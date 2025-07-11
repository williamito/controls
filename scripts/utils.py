import copy
import numpy as np



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
        "so100": np.array([[0.0, 1.0, 0.0, 0.0], 
                           [-1.0, 0.0, 0.0, -0.0453], 
                           [0.0, 0.0, 1.0, 0.0647], 
                           [0.0, 0.0, 0.0, 1.0]])
    }

    # set nTtool frame (n-frame DH aligned wrt SO100 simulator)
    N_T_TOOL = {
        "so100": np.array([[0.0, 0.0, -1.0, 0.0], 
                           [1.0, 0.0, 0.0, 0.0], 
                           [0.0, -1.0, 0.0, 0.0], 
                           [0.0, 0.0, 0.0, 1.0]])
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
        
        """convert joint positions from DH to mechanical coordinates (for SO100)"""

        beta = np.deg2rad(14.45)  # make reference to dh2.png in README.md

        q_mech = np.zeros_like(q_dh)
        q_mech[0] = q_dh[0]
        q_mech[1] = -q_dh[1] - beta
        q_mech[2] = -q_dh[2] + beta
        q_mech[3] = -q_dh[3] - np.pi / 2
        q_mech[4] = -q_dh[4] - np.pi / 2

        return q_mech

    def from_mech_to_dh(self, q_mech):
        
        """convert joint positions from mechanical to DH coordinates (for SO100)"""

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
    def calc_geom_jac_0(robot, q):
        
        """compute geometrical jacobian wrt base-frame"""

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

    @staticmethod
    def calc_geom_jac_n(robot, q, base_T_n):
        
        """compute geometrical jacobian wrt n-frame"""
        
        DOF = len(q)
        Jn = np.zeros((6, DOF))
        T = np.zeros((6,6))
        R = base_T_n[:3,:3]
        
        # get jacobian in base frame
        J0 = RobotUtils.calc_geom_jac_0(robot, q)
        
        # compose mapping matrix
        T[:3,:3] = R.T
        T[3:,3:] = R.T
        
        # compute jacobian in n-frame
        Jn = T @ J0
        
        return Jn
        
    @staticmethod
    def dls_right_pseudoinv(J, lambda_val=0.001):
        
        """compute Damped Least Squares Right-Pseudo-Inverse"""
               
        JT = J.T
        JTJ = JT @ J
        J_pinv = np.linalg.inv(JTJ + lambda_val * np.eye(JTJ.shape[0])) @ JT  
        
        return J_pinv

    @staticmethod
    def dls_right_pseudoinv_weighted(J, W, lambda_val=0.001):
        
        """compute Damped Least Squares Right-Pseudo-Inverse weighted"""
               
        JT = J.T
        JTJ = JT @ J
        J_pinv = np.linalg.inv(JTJ + lambda_val * W) @ JT 
        
        return J_pinv