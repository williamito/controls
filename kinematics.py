import copy

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


class Robot:
    
    # Follow this convention: theta , d, a, alpha
    ROBOT_DH_TABLES = {
        "so100": [
            [0, 0.0542, 0.0304, np.pi / 2],
            [0, 0.0, 0.116, 0.0],
            [0, 0.0, 0.1347, 0.0],
            [0, 0.0, 0.0, -np.pi / 2],
            [0, 0.0609, 0.0, 0.0],  # to increase length and include also gripper: [0, 0.155, 0.0, 0.0],
        ]
    }
    
    # define mass for link i (n° links = n° joints +1)
    LINK_MASS = {
        "so100": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    }
    
    # define COM for link i wrt origin frame i (n° links = n° joints +1)
    LINK_COM = {
        "so100": [
            np.array([0.0, 0.05, 0.0]),
            np.array([0.0, 0.05, 0.0]),
            np.array([0.0, 0.05, 0.0]),
            np.array([0.0, 0.05, 0.0]),
            np.array([0.0, 0.05, 0.0]),
            np.array([0.0, 0.05, 0.0]),
        ]
    } 
    
    # define Inertia for link i wrt origin frame i (n° links = n° joints +1)
    LINK_INERTIA = {
        "so100": [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0542, 0.0, 0.0, 0.0, 0.0304]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0542, 0.0, 0.0, 0.0, 0.0304]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1347]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1347]),
        ]
    }
    
    # mechanical joint limitis
    MECH_JOINT_LIMITS_LOW = {"so100": np.array([-2.2000, -3.1416, 0.0000, -2.0000, -3.1416, -0.2000])}
    MECH_JOINT_LIMITS_UP = {"so100": np.array([2.2000, 0.2000, 3.1416, 1.8000, 3.1416, 2.0000])}

    # set worldTbase frame (base-frame DH aligned wrt SO100 simulator)
    WORLD_T_TOOL = {
        "so100": np.array([[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, -0.0453], [0.0, 0.0, 1.0, 0.0647], [0.0, 0.0, 0.0, 1.0]])
    }

    # set nTtool frame (n-frame DH aligned wrt SO100 simulator)
    N_T_TOOL = {
        "so100": np.array([[0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    }

    def __init__(self, robot_type="so100"):
        
        if robot_type not in Robot.ROBOT_DH_TABLES:
            raise ValueError(f"Unknown robot type: {robot_type}. Available: {list(Robot.ROBOT_DH_TABLES.keys())}")

        # set robot model
        self.robot_type = robot_type
        self.dh_table = Robot.ROBOT_DH_TABLES[robot_type]
        self.link_mass = Robot.LINK_MASS[robot_type]
        self.link_com = Robot.LINK_COM[robot_type]
        self.link_inertia = Robot.LINK_INERTIA[robot_type]
        self.mech_joint_limits_low = Robot.MECH_JOINT_LIMITS_LOW[robot_type]
        self.mech_joint_limits_up = Robot.MECH_JOINT_LIMITS_UP[robot_type]
        self.worldTbase = Robot.WORLD_T_TOOL[robot_type]
        self.nTtool = Robot.N_T_TOOL[robot_type]

    def from_dh_to_mech(self, q_dh):
        
        """convert joint positions from DH to mechanical coordinates"""

        beta = np.deg2rad(14.45)  # make reference to dh2.png in README.md

        q_mech = np.zeros_like(q_dh)
        q_mech[0] = q_dh[0]
        q_mech[1] = -q_dh[1] - beta
        q_mech[2] = -q_dh[2] + beta
        q_mech[3] = -q_dh[3] - np.pi / 2
        q_mech[4] = -q_dh[4] - np.pi / 2

        return q_mech

    def from_mech_to_dh(self, q_mech):
        
        """convert joint positions from mechanical to DH coordinates"""

        beta = np.deg2rad(14.45)  # make reference to dh2.png in README.md

        q_dh = np.zeros_like(q_mech)
        q_dh[0] = q_mech[0]
        q_dh[1] = -q_mech[1] - beta
        q_dh[2] = -q_mech[2] + beta
        q_dh[3] = -q_mech[3] - np.pi / 2
        q_dh[4] = -q_mech[4] - np.pi / 2

        return q_dh[:-1]  # skip last DOF because it is the gripper

    def check_joint_limits(self, q_vec):
        
        """raise an error in case mechanical joint limits are exceeded"""

        for i, q in enumerate(q_vec):
            assert self.mech_joint_limits_low[i] <= q <= self.mech_joint_limits_up[i], (f"[ERROR] Joint limits out of bound. J{i + 1} = {q}, but limits are ({self.mech_joint_limits_low[i]}, {self.mech_joint_limits_up[i]})")



class RobotUtils:
    
    @staticmethod
    def calc_distance(p1, p2):
        
        """compute distance between two 3D vectors"""

        return np.linalg.norm(p2 - p1)

    @staticmethod
    def inv_homog_mat(T):
        
        """invert homogenous transformation matrix"""

        R = T[:3, :3]
        t = T[:3, 3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        return T_inv

    @staticmethod
    def calc_lin_err(T_current, T_desired):
        
        """compute linear error between 2 homogenous transformations"""

        return T_desired[:3, 3] - T_current[:3, 3]

    @staticmethod
    def calc_ang_err(T_current, T_desired):
        
        """compute angular error between two homogenous transformations (axis-angle notation)"""

        R_current = T_current[:3, :3]
        R_desired = T_desired[:3, :3]
        return 0.5 * (np.cross(R_current[:, 0], R_desired[:, 0])
                + np.cross(R_current[:, 1], R_desired[:, 1])
                + np.cross(R_current[:, 2], R_desired[:, 2]))

    @staticmethod
    def calc_dh_matrix(dh, theta):
        
        """compute dh matrix"""

        _, d, a, alpha = dh
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)

        return np.array([[ct, -st * ca, st * sa, a * ct], 
                         [st, ct * ca, -ct * sa, a * st], 
                         [0, sa, ca, d], 
                         [0, 0, 0, 1]])   
    
    @staticmethod
    def calc_geom_jac(robot, q):
        
        """compute geometrical jacobian wrt base-frame"""

        DOF = len(q)
        J = np.zeros((6, DOF))
        P = np.zeros((3, DOF+1))
        z = np.zeros((3, DOF+1))
        base_P_i = np.eye(4)
        
        # define z0
        z[:,0] = np.array([0,0,1])
        
        for i in range(DOF):
            
            i_T_ip1 = RobotUtils.calc_dh_matrix(robot.dh_table[i], q[i])
            base_P_i = base_P_i @ i_T_ip1
            P[:,i+1] = base_P_i[:3,3]
            z[:,i+1] = base_P_i[:3,2]
            
        for i in range(DOF):
            
            # compose jacobian matrix
            J[:3, i] = np.cross(z[:,i] , P[:,DOF] - P[:,i])
            J[3:, i] = z[:,i] 
            
        return J

    @staticmethod
    def dls_right_pseudoinv(J, lambda_val=0.001):
        
        """compute Damped Least Squares Right-Pseudo-Inverse"""
               
        JT = J.T
        JTJ = JT @ J
        J_pinv = np.linalg.inv(JTJ + lambda_val * np.eye(JTJ.shape[0])) @ JT  
        
        return J_pinv

    

class RobotKinematics:
    
    def __init__(self):
        pass

    def forward_kinematics(self, robot, q):
        
        """compute forward kinematics (worldTtool)"""

        baseTn = self._forward_kinematics_baseTn(robot, q)

        return robot.worldTbase @ baseTn @ robot.nTtool

    def _forward_kinematics_baseTn(self, robot, q):
        
        """compute forward kinematics (baseTn)"""

        T = np.eye(4)
        DOF = len(q)

        for i in range(DOF):
            dh = robot.dh_table[i]
            T_link = RobotUtils.calc_dh_matrix(dh, q[i])
            T = T @ T_link

        return T

    def _inverse_kinematics_step_baseTn(self, robot, q_start, T_desired, use_orientation=True, k=0.8, n_iter=50):
        
        """compute inverse kinematics (T_desired must be expressed in baseTn)"""

        # don't override current joint positions
        q = copy.deepcopy(q_start)

        for _ in range(n_iter):
            # compute current pose baseTn
            T_current = self._forward_kinematics_baseTn(robot, q)

            # compute linear error
            err_lin = RobotUtils.calc_lin_err(T_current, T_desired)

            # decide whether to use full jacobian or not
            if use_orientation:
                err_ang = RobotUtils.calc_ang_err(T_current, T_desired)  # compute angular error
                error = np.concatenate((err_lin, err_ang))  # total error
                J_geom = RobotUtils.calc_geom_jac(robot, q)  # full jacobian
            else:
                error = err_lin  # total error
                J_geom = RobotUtils.calc_geom_jac(robot, q)[:3, :]  # take only the position part

            # stop if error is minimum
            if np.linalg.norm(error) < 1e-5:
                break

            # Damped Least Squares Right-Pseudo-Inverse
            J_pinv = RobotUtils.dls_right_pseudoinv(J_geom)

            # keep integrating resulting joint positions
            q += k * (J_pinv @ error)

        return q

    def inverse_kinematics(self, robot, q_start, desired_worldTtool, use_orientation=True, k=0.8, n_iter=50):
        
        """compute inverse kinematics (T_desired must be expressed in worldTtool)
        It is performed an interpolation both for linear and angular components"""

        # I compute ikine with baseTn
        desired_baseTn = (RobotUtils.inv_homog_mat(robot.worldTbase)
                          @ desired_worldTtool
                          @ RobotUtils.inv_homog_mat(robot.nTtool))

        # don't override current joint positions
        q = copy.deepcopy(q_start)

        # init interpolator
        n_steps = self._interp_init(self._forward_kinematics_baseTn(robot, q), desired_baseTn)

        for i in range(0, n_steps + 1):
            # current setpoint as baseTn
            T_desired_interp = self._interp_execute(i)

            # get updated joint positions
            q = self._inverse_kinematics_step_baseTn(robot, q, T_desired_interp, use_orientation, k, n_iter)

        # check final error
        current_worldTtool = self.forward_kinematics(robot, q)
        err_lin = RobotUtils.calc_lin_err(current_worldTtool, desired_worldTtool)
        lin_error_norm = np.linalg.norm(err_lin)
        assert lin_error_norm < 1e-2, (f"[ERROR] Large position error ({lin_error_norm:.4f}). Check target reachability (position/orientation)")

        return q

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
    
    
