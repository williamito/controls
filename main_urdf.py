import os

from scripts.model import *
from scripts.kinematics import *
from scripts.dynamics import *


    # --> sarebbe bello usarla pulita e nel main in cui uso pybullet per mostrare robot e muoverlo con mia libreria!

    # def inverse_kinematics(self, robot, q_start, desired_worldTtool, target_link_name=None, use_orientation=True, k=0.8, n_iter=50):
        
    #     """compute inverse kinematics (T_desired must be expressed in worldTtool)
    #     It is performed an interpolation both for linear and angular components"""

    #     # I compute ikine with baseTn
    #     desired_baseTn = (RobotUtils.inv_homog_mat(robot.worldTbase)
    #                       @ desired_worldTtool
    #                       @ RobotUtils.inv_homog_mat(robot.nTtool))

    #     # don't override current joint positions
    #     q = copy.deepcopy(q_start)

    #     # init interpolator
    #     n_steps = self._interp_init(self._forward_kinematics_baseTn(robot, q, target_link_name), desired_baseTn)

    #     for i in range(0, n_steps + 1):

    #         # current setpoint as baseTn
    #         T_desired_interp = self._interp_execute(i)

    #         # get updated joint positions
    #         q = self._inverse_kinematics_step_baseTn(robot, q, T_desired_interp, target_link_name, use_orientation, k, n_iter)
        
    #     # check final error
    #     current_worldTtool = self.forward_kinematics(robot, q, target_link_name)
    #     err_lin = RobotUtils.calc_lin_err(current_worldTtool, desired_worldTtool)
    #     lin_error_norm = np.linalg.norm(err_lin)
    #     assert lin_error_norm < 1e-2, (f"[ERROR] Large position error ({lin_error_norm:.4f}). Check target reachability (position/orientation)")

    #     return q



# 0) investigare singolarità o moto strano in pybullet con mia libreria
# 1) fixare fatto che posso dare in input q a dim=23, non a mano che sto a tagliare a dim=6 --> poi togliere rifirimenti a G1 nel codice
# 2) aggiungere main_pybullet.py dove uso mia lib + visualizzo in pybullet
# 3) riscrivi readme includendo urdf explanation + requirements.txt env + fai merge

# 4) testare DH dynamics su pybullet
# 5) fare andare URDF dynamics su SO100


## URDF Loader ##

print("\n\nLOADING ROBOT MODEL:\n\n")

# Get directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct relative path to URDF
urdf_path = os.path.join(script_dir, "urdfs", "so100", "so100.urdf")
urdf_path = os.path.join(script_dir, "urdfs", "g1", "g1_23dof.urdf")

# Instantiate URDF loader
urdf_loader = URDF_handler()
urdf_loader.load(urdf_path)

# Wrap in RobotModel
robot_model = RobotModel(urdf_loader)

# Info
robot_model.print_model_properties()
print("Number of joints: ", robot_model.get_n_joints())
print("Number of links: ", robot_model.get_n_links())

## kinematics ##

print("\n\nKINEMATICS EXAMPLE:\n\n")

# init
kin = URDF_Kinematics()

# get current joint positions   
q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # SO100
q_init = np.zeros(6)  # GO1 --> n_dof = 6 for target "right_ankle_roll_link"
print("q_init: ", np.rad2deg(q_init))

# compute start pose
# T_start = kin.forward_kinematics(robot_model, q_init, target_link_name="gripper") # SO100
T_start = kin.forward_kinematics(robot_model, q_init, target_link_name="right_ankle_roll_link") # G1
print("\nT_start = \n", T_start)

# Define relative goal pose G1
T_goal = T_start.copy()
T_goal[:3, 3] += np.array([0.2, 0.0, 0.3])
print("\nT_goal = \n", T_goal)

# # Define relative goal pose SO100
# T_goal = T_start.copy()
# T_goal[:3, 3] += np.array([0.0, 0.0, 0.2])
# print("\nT_goal = \n", T_goal)

# # Define absolute goal pose SO100
# T_goal = np.array([
#     [1.0, 0.0, 0.0,  0.0],
#     [0.0, 1.0, 0.0, -0.17],
#     [0.0, 0.0, 1.0,  0.37],
#     [0.0, 0.0, 0.0,  1.0]
# ])

# IK with internal interpolation
# q_final = kin.inverse_kinematics(robot_model, q_init, T_goal, target_link_name="gripper", use_orientation=True, k=0.8, n_iter=1) # SO100
q_final = kin.inverse_kinematics(robot_model, q_init, T_goal, target_link_name="right_ankle_roll_link", use_orientation=True, k=0.8, n_iter=1) # G1
print("\nFinal joint angles = ", q_final)

# T_final = kin.forward_kinematics(robot_model, q_final, target_link_name="gripper")  # SO100
T_final = kin.forward_kinematics(robot_model, q_final, target_link_name="right_ankle_roll_link")  # G1
print("\nFinal pose direct kinematics = \n", T_final)

print("\nerr_lin = ", RobotUtils.calc_lin_err(T_goal, T_final))
print("err_ang = ", RobotUtils.calc_ang_err(T_goal, T_final))

# raise an error in case joint limits are exceeded
kin.check_joint_limits(robot_model, q_final)

# ## dynamics ## 

# print("\n\nDYNAMICS EXAMPLE:\n\n")

# # init
# dyn = RobotDynamics()

# # init parameters
# q = np.array([0.1, 0.2, 0.1, 0.2, 0.1]) 
# qdot = np.array([0.7, 0.7, 0.7, 0.7, 0.7]) # --> -0.6120  -7.6320  -3.2212   0.4416   0.1600 e qddot = 0
# qddot = np.array([0.7, 0.7, 0.7, 0.7, 0.7]) # --> -0.1444  -7.3537  -3.0630   0.4483   0.3428 e qdot != 0
# Fext = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]) # expressed wrt n-frame --> 3.4263  -8.6711  -5.0112  -2.1744   1.3428 e qdot, qddot != 0

# # inverse dynamics
# torques = dyn.inverse_dynamics(robot_model, q, qdot, qddot, Fext = Fext) 
# print("torques with RNE: \n", torques)

# # compute B, Cqdot, G for the whole robotic model
# B, Cqdot, G = dyn.get_robot_model(robot_model, q, qdot)
# print("\nB: \n", B)
# print("\nCqdot: \n", Cqdot)
# print("\nG: \n", G)

# # transform force
# f_ext_tool = np.array([0.0, 0.0, 2.0, 0.0, 1.0, 0.0])
# tool_T_n = RobotUtils.inv_homog_mat(robot_model.nTtool)
# f_ext_n = dyn.transform_force(f_ext_tool, tool_T_n)
# print("\nforce expressed in n-frame: \n", f_ext_n)


# print("\n\nJACOBIANS EXAMPLE:\n\n")

# # Print chain list from base_link to flange
# chain = kin.get_joint_chain(robot_model, "base_link", "flange")
# for idx, joint in enumerate(chain):
#     print(idx, joint.name, joint.parent, joint.child)

# # jacobians
# q = np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.2]) 
# base_T_n = kin._forward_kinematics_baseTn(robot_model, q)
# J0 = kin.calc_geom_jacobian(robot_model, q, target="flange")
# Jn = kin.calc_geom_jacobian(robot_model, q, target="flange", reference_frame=ReferenceFrame.LOCAL)
# print("\ngeometric jacobian in base-frame: \n", J0)
# print("\ngeometric jacobian in n-frame: \n", Jn)

