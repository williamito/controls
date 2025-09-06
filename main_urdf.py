import os

from scripts.model import *
from scripts.kinematics import *
from scripts.dynamics import *


# 0) Include source_link oltre al target_link)
# 1) fare andare IK di ABB Robot --> confronta jacobians con pybullet (magari sbaglio a calcolarli)
# 2) fare andare DK + IK SO100
# 3) fare andare URDF dynamics su SO100 (come DH)
# 4) testare dynamics su pybullet???


# lerobot 
# qpos_start = [-np.pi/2, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2]
# ee_start = [-0.1936, -0.0452,  0.1743]

# The expected final position should be

# qpos_goal = [-1.5708, -1.7102,  1.1232,  1.3598, -1.5709,  1.5706]
# ee_goal = [-0.1952, -0.0452,  0.2714]


# 6 axis
# Start q:  [0. 0. 0. 0. 0. 0.]
# T_start: 
#  [[ 0.          0.          1.          2.11249995]
#  [ 0.          1.          0.          0.        ]
#  [-1.          0.          0.          2.25999999]
#  [ 0.          0.          0.          1.        ]]
# T_final: 
#  [[1.  0.  0.  2.3]
#  [0.  1.  0.  0.3]
#  [0.  0.  1.  2.8]
#  [0.  0.  0.  1. ]]
# Final q:  [ 1.29899961e-01  4.32965089e-01 -7.31214305e-01 -4.08254762e-04
#  -1.27558318e+00 -1.29778771e-01]
# Final pose: 
#  [[ 9.99967623e-01 -6.57766597e-07  8.04692895e-03  2.29901719e+00]
#  [ 6.63310164e-07  1.00000000e+00 -6.86235942e-07  2.98462093e-01]
#  [-8.04692895e-03  6.91551334e-07  9.99967623e-01  2.79722095e+00]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]





## URDF Loader ##

print("\n\nLOADING ROBOT MODEL:\n\n")

# Get directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct relative path to URDF
urdf_path = os.path.join(script_dir, "urdfs", "abb_irb6700_150_320", "abb_irb6700_150_320.urdf")
# urdf_path = os.path.join(script_dir, "urdfs", "abb_irb6700_150_320", "so100.urdf")

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
# q_init = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2])
q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
print("q_init: ", np.rad2deg(q_init))

# compute start pose
T_start = kin.forward_kinematics(robot_model, q_init, target_link_name="flange")
# T_start = kin.forward_kinematics(robot_model, q_init, target_link_name="gripper")
print("\nT_start = \n", T_start)

# Define goal pose
# T_goal = T_start.copy()
# T_goal[:3, 3] += np.array([0.0, 0.0, 0.0])
# print("\nT_goal = \n", T_goal)

T_goal = np.array([
    [1.0, 0.0, 0.0, 2.3],
    [0.0, 1.0, 0.0, 0.3],
    [0.0, 0.0, 1.0, 2.8],
    [0.0, 0.0, 0.0, 1.0]
])
print("\nT_goal = \n", T_goal)



# IK with internal interpolation
q_final = kin.inverse_kinematics(robot_model, q_init, T_goal, target_link_name="flange", use_orientation=True, k=0.8, n_iter=1) 
print("\nFinal joint angles = ", q_final)

T_final = kin.forward_kinematics(robot_model, q_final, target_link_name="flange")
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
chain = kin.get_joint_chain(robot_model, "base_link", "flange")
for idx, joint in enumerate(chain):
    print(idx, joint.name, joint.parent, joint.child)

# jacobians
q = np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.2]) 
base_T_n = kin._forward_kinematics_baseTn(robot_model, q)
J0 = kin.calc_geom_jacobian(robot_model, q, target="flange")
Jn = kin.calc_geom_jacobian(robot_model, q, target="flange", reference_frame=ReferenceFrame.LOCAL)
print("\ngeometric jacobian in base-frame: \n", J0)
print("\ngeometric jacobian in n-frame: \n", Jn)

