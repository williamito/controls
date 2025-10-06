import os

from scripts.model import *
from scripts.kinematics import *
from scripts.dynamics import *


# pip install "networkx>=2.6,<3"
# pip install "numpy<1.24"

# - networkx==2.2 --> unica diversa lei, vedere quale giusta

# - mujoco==3.3.6
# - numpy==1.23.5
# - urdfpy==0.0.22


# 1) creare 2 metodi per IKINE_FULL (risultato finale diretto) e IKINE_STEP (risultati intermedi e devo fare loop nel main) + visualizza in pybullet
# 2) riscrivi readme includendo urdf explanation 
# 3) testa create env.yaml from scratch + fai merge



# 4) investigare singolarità o moto strano o slerp orientation in pybullet con mia libreria
# 5) testare DH dynamics su mujoco
# 6) fare andare URDF dynamics su SO100



## URDF Loader ##

print("\n\nLOADING ROBOT MODEL:\n\n")

# Get directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct relative path to URDF
urdf_path = os.path.join(script_dir, "models", "so101", "so101.urdf") 

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

# define ee target link name
ee_name = "gripper_link"

# get current joint positions (only movable joints, no fixed joints and no floating base pose) 
q_init = np.zeros(6) 
print("q_init: ", np.rad2deg(q_init))

# compute start pose
T_start = kin.forward_kinematics(robot_model, q_init, target_link_name=ee_name) 
print("\nT_start = \n", T_start)

# Define relative goal pose
T_goal = T_start.copy()
T_goal[:3, 3] += np.array([-0.1, 0.0, 0.1])
print("\nT_goal = \n", T_goal)

# # Define absolute goal pose
# T_goal = np.array([
#     [1.0, 0.0, 0.0,  0.0],
#     [0.0, 1.0, 0.0, -0.17],
#     [0.0, 0.0, 1.0,  0.37],
#     [0.0, 0.0, 0.0,  1.0]
# ])
# print("\nT_goal = \n", T_goal)

# IK with internal interpolation
q_final = kin.inverse_kinematics(robot_model, q_init, T_goal, target_link_name=ee_name, use_orientation=True, k=0.8, n_iter=1) 
print("\nFinal joint angles = ", q_final)

T_final = kin.forward_kinematics(robot_model, q_final, target_link_name=ee_name)  
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


print("\n\nJACOBIANS EXAMPLE:\n\n")

# Print chain list from base_link to flange
chain = kin.get_joint_chain(robot_model, robot_model.root_link, ee_name)
for idx, joint in enumerate(chain):
    print(idx, joint.name, joint.parent, joint.child)

# jacobians
q = np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.2]) 
base_T_n = kin._forward_kinematics_baseTn(robot_model, q, target_link_name=ee_name)
J0 = kin.calc_geom_jacobian(robot_model, q, target=ee_name)
Jn = kin.calc_geom_jacobian(robot_model, q, target=ee_name, reference_frame=ReferenceFrame.LOCAL)
print("\ngeometric jacobian in base-frame: \n", J0)
print("\ngeometric jacobian in n-frame: \n", Jn)

