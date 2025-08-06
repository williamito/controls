import copy
import numpy as np
from enum import Enum
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from scripts.utils import *



class ReferenceFrame(Enum):
    BASE = 0
    LOCAL = 1


class RobotKinematics:
    
    """Parent class to define methods common to both DH and URDF representations"""

    def forward_kinematics(self, robot, q, target_link_name=None):
        
        """compute forward kinematics (worldTtool)"""

        baseTn = self._forward_kinematics_baseTn(robot, q, target_link_name)

        return robot.worldTbase @ baseTn @ robot.nTtool

    def _forward_kinematics_baseTn(self, robot, q, target_link_name):
        
        """compute forward kinematics (baseTn)"""
        
        raise NotImplementedError
    
    def calc_geom_jacobian(self, robot, q, target=None, reference_frame=ReferenceFrame.BASE):
        
        """Compute geometric Jacobian for target (link or frame) wrt BASE or LOCAL frame"""

        raise NotImplementedError

    def _inverse_kinematics_step_baseTn(self, robot, q_start, T_desired, target_link_name=None, use_orientation=True, k=0.8, n_iter=50):
        
        """compute inverse kinematics (T_desired must be expressed in baseTn)"""

        # don't override current joint positions
        q = copy.deepcopy(q_start)

        for _ in range(n_iter):
            # compute current pose baseTn
            T_current = self._forward_kinematics_baseTn(robot, q, target_link_name)

            # compute linear error
            err_lin = RobotUtils.calc_lin_err(T_current, T_desired)

            # decide whether to use full jacobian or not
            if use_orientation:
                err_ang = RobotUtils.calc_ang_err(T_current, T_desired)  # compute angular error
                error = np.concatenate((err_lin, err_ang))  # total error
                J_geom = self.calc_geom_jacobian(robot, q)  # full jacobian
            else:
                error = err_lin  # total error
                J_geom = self.calc_geom_jacobian(robot, q)[:3, :]  # take only the position part

            # stop if error is minimum
            if np.linalg.norm(error) < 1e-5:
                break

            # Damped Least Squares Right-Pseudo-Inverse
            J_pinv = RobotUtils.dls_right_pseudoinv(J_geom)

            # keep integrating resulting joint positions
            q += k * (J_pinv @ error)

        return q

    def inverse_kinematics(self, robot, q_start, desired_worldTtool, target_link_name=None, use_orientation=True, k=0.8, n_iter=50):
        
        """compute inverse kinematics (T_desired must be expressed in worldTtool)
        It is performed an interpolation both for linear and angular components"""

        # I compute ikine with baseTn
        desired_baseTn = (RobotUtils.inv_homog_mat(robot.worldTbase)
                          @ desired_worldTtool
                          @ RobotUtils.inv_homog_mat(robot.nTtool))

        # don't override current joint positions
        q = copy.deepcopy(q_start)

        # init interpolator
        n_steps = self._interp_init(self._forward_kinematics_baseTn(robot, q, target_link_name), desired_baseTn)

        for i in range(0, n_steps + 1):
            # current setpoint as baseTn
            T_desired_interp = self._interp_execute(i)

            # get updated joint positions
            q = self._inverse_kinematics_step_baseTn(robot, q, T_desired_interp, target_link_name, use_orientation, k, n_iter)

        # check final error
        current_worldTtool = self.forward_kinematics(robot, q, target_link_name)
        err_lin = RobotUtils.calc_lin_err(current_worldTtool, desired_worldTtool)
        lin_error_norm = np.linalg.norm(err_lin)
        assert lin_error_norm < 1e-2, (f"[ERROR] Large position error ({lin_error_norm:.4f}). Check target reachability (position/orientation)")

        return q

    def check_joint_limits(self, robot, q_vec): 
    
        """raise an error in case mechanical joint limits are exceeded"""

        for i, q in enumerate(q_vec):
            assert robot.mech_joint_limits_low[i] <= q <= robot.mech_joint_limits_up[i], (f"[ERROR] Joint limits out of bound. J{i + 1} = {q}, but limits are ({robot.mech_joint_limits_low[i]}, {robot.mech_joint_limits_up[i]})")

    def _interp_init(self, T_start, T_final):
        
        """Initialize interpolator parameters"""

        # init
        self.t_start = T_start[:3, 3]
        self.t_final = T_final[:3, 3]
        R_start = T_start[:3, :3]
        R_final = T_final[:3, :3]
        
        # step size for trajectory
        delta_trans = 0.01  # meters
        delta_rot = 0.05    # radians
        
        # linear distance
        trans_dist = RobotUtils.calc_distance(self.t_final, self.t_start)
        n_steps_trans = trans_dist / delta_trans
        
        # angular distance
        rotvec = R.from_matrix(R_final @ R_start.T).as_rotvec() # axis*angle
        ang_dist = np.linalg.norm(rotvec) # angle modulus
        n_steps_rot = ang_dist / delta_rot
        
        # total steps
        self.n_steps = int(np.ceil(max(n_steps_trans, n_steps_rot)))

        # Create SLERP object
        times = [0, 1]
        rotations = R.from_matrix([R_start, R_final])
        self.slerp = Slerp(times, rotations)

        return self.n_steps

    def _interp_execute(self, i):
        
        """Compute Cartesian pose setpoint for the current step"""

        # n_steps == 0 means Tgoal == Tinit
        # In this way I also avoid division by zero
        if self.n_steps == 0:
            s = 1.0
        else:
            s = i / self.n_steps  # compute current step

        t_interp = (1 - s) * self.t_start + s * self.t_final
        R_interp = self.slerp(s).as_matrix()

        # compute current setpoint
        T_interp = np.eye(4)
        T_interp[:3, :3] = R_interp
        T_interp[:3, 3] = t_interp

        return T_interp
    


    
class DH_Kinematics(RobotKinematics):

    """Child class in charge to override methods specific for DH representation"""

    def __init__(self):

        super().__init__()  # Initialize RobotKinematics

    def _forward_kinematics_baseTn(self, robot, q, target_link_name=None):
        
        """compute forward kinematics (baseTn)"""

        T = np.eye(4)
        DOF = len(q)

        # loop over DH frames
        for i in range(DOF):
            dh = robot.dh_table[i]
            T_link = RobotUtils.calc_dh_matrix(dh, q[i])
            T = T @ T_link

        return T
    
    def calc_geom_jac_0(self, robot, q, target_link_name=None):
        
        """compute geometrical jacobian for base-frame wrt base-frame"""

        DOF = len(q)
        J = np.zeros((6, DOF))
        P = np.zeros((3, DOF+1))
        z = np.zeros((3, DOF+1))
        base_P_i = np.eye(4)
        
        # define z0
        z[:,0] = np.array([0,0,1])
        
        for i in range(DOF):
            
            # get Pi and zi
            i_T_ip1 = RobotUtils.calc_dh_matrix(robot.dh_table[i], q[i])
            base_P_i = base_P_i @ i_T_ip1
            P[:,i+1] = base_P_i[:3,3]
            z[:,i+1] = base_P_i[:3,2]
            
        for i in range(DOF):
            
            # compose jacobian matrix
            J[:3, i] = np.cross(z[:,i] , P[:,DOF] - P[:,i])
            J[3:, i] = z[:,i] 
            
        return J
    
    def calc_geom_jac_n(self, robot, q, base_T_n):
        
        """compute geometrical jacobian for n-frame wrt n-frame"""
        
        DOF = len(q)
        Jn = np.zeros((6, DOF))
        T = np.zeros((6,6))
        R = base_T_n[:3,:3]
        
        # get jacobian in base frame
        J0 = self.calc_geom_jac_0(robot, q)
        
        # compose mapping matrix
        T[:3,:3] = R.T
        T[3:,3:] = R.T
        
        # compute jacobian in n-frame
        Jn = T @ J0
        
        return Jn

    def calc_geom_jacobian(self, robot, q, target=None, reference_frame=ReferenceFrame.BASE):
        
        """Compute geometric Jacobian for n-frame (target) wrt BASE or LOCAL frame"""
        
        if (reference_frame == ReferenceFrame.LOCAL):
            J = self.calc_geom_jac_n(robot, q, target)
        elif (reference_frame == ReferenceFrame.BASE):
            J = self.calc_geom_jac_0(robot, q)

        return J
    



class URDF_Kinematics(RobotKinematics):

    """Child class in charge to override methods specific for URDF representation"""

    def __init__(self):

        super().__init__()  # Initialize RobotKinematics

    def _forward_kinematics_baseTn(self, robot, q, target_link_name="base_link"):

        """compute forward kinematics (baseTn)"""

        # get full joint chain from base to target
        chain = self.get_joint_chain(robot, "base_link", target_link_name)

        T = np.eye(4)
        q_index = 0  # index into q (which only contains movable joints)

        for joint in chain:
            if robot.joint_type(joint) == "fixed":
                # just apply fixed joint transform (no q)
                T = T @ RobotUtils.calc_urdf_joint_transform(joint, 0.0)
            else:
                # apply transform with actual joint variable
                T = T @ RobotUtils.calc_urdf_joint_transform(joint, q[q_index])
                q_index += 1

        return T

    def calc_geom_jacobian(self, robot, q, target=None, reference_frame=ReferenceFrame.BASE):
        
        """
        Compute geometric Jacobian for target link wrt BASE or LOCAL frame.
        q must correspond to robot's movable joints in the chain.
        """

        # get chain joints (only movable ones)
        chain = self.get_joint_chain(robot, "base_link", target)
        # chain = [j for j in robot.get_chain("base_link", target, links=False) if robot.is_movable(j)]
        DOF = len(chain)

        J = np.zeros((6, DOF))
        P = np.zeros((3, DOF + 1))
        z = np.zeros((3, DOF + 1))
        T = np.eye(4)

        # z0 (z-axis of base)
        z[:, 0] = np.array([0, 0, 1])

        for i, joint in enumerate(chain):
            T = T @ RobotUtils.calc_urdf_joint_transform(joint, q[i])
            P[:, i + 1] = T[:3, 3]
            z[:, i + 1] = T[:3, 2]

        # build Jacobian
        for i, joint in enumerate(chain):
            if robot.joint_type(joint) == "revolute":
                J[:3, i] = np.cross(z[:, i], P[:, DOF] - P[:, i])
                J[3:, i] = z[:, i]
            elif robot.joint_type(joint) == "prismatic":
                J[:3, i] = z[:, i]
                J[3:, i] = np.zeros(3)

        # if LOCAL frame requested
        if reference_frame == ReferenceFrame.LOCAL:
            R = T[:3, :3]  # from base to target
            T_map = np.zeros((6, 6))
            T_map[:3, :3] = R.T
            T_map[3:, 3:] = R.T
            J = T_map @ J

        return J
    
    def get_joint_chain(robot, base_link_name, target_link_name):

        """Returns the list of joints connecting base_link_name to target_link_name"""




        pass
