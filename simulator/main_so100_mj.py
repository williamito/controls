# import pybullet as p
# import time
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# from scripts.model import *
# from scripts.kinematics import *
# from scripts.dynamics import *


# # run from /controls folder directly: python -m simulator.main_so100


# ### INIT PYBULLET ###

# # pybullet params
# physicsClient = p.connect(p.GUI)
# p.setGravity(0, 0, -9.81)

# # Get directory where this script is located
# script_dir = os.path.dirname(os.path.abspath(__file__))

# # Construct relative path to URDF
# urdf_path = os.path.join(script_dir, "..", "urdfs", "so100", "so100.urdf") 

# # Load URDF
# robot_id = p.loadURDF(urdf_path, useFixedBase=True)

# # Find index corresponding to the flange
# effector_name = "gripper"
# end_effector_index = -1

# for i in range(p.getNumJoints(robot_id)):
#     joint_info = p.getJointInfo(robot_id, i)
#     link_name = joint_info[12].decode("utf-8")
#     if link_name == effector_name:
#         end_effector_index = i
#         break

# print("End effector index:", end_effector_index)

# # Find indeces corresponding to movable joints
# movable_joints = []
# for i in range(p.getNumJoints(robot_id)):
#     joint_info = p.getJointInfo(robot_id, i)
#     joint_type = joint_info[2]
#     if joint_type in [p.JOINT_PRISMATIC, p.JOINT_REVOLUTE]:
#         movable_joints.append(i)

# N_DOF = len(movable_joints)
# print("Movable joints: ", N_DOF)


# ### INIT CONTROLS LIBRARY ###

# # Instantiate URDF loader
# urdf_loader = URDF_handler()
# urdf_loader.load(urdf_path)

# # Wrap in RobotModel
# robot_model = RobotModel(urdf_loader)

# # Info
# robot_model.print_model_properties()
# print("Number of joints: ", robot_model.get_n_joints())
# print("Number of links: ", robot_model.get_n_links())


# ### SIMULATION ###

# time.sleep(5)

# # Read initial joint state
# q_init = [p.getJointState(robot_id, i)[0] for i in movable_joints]
# q_init = np.array(q_init) 
# print("Start q: ", q_init)

# # Get initial pose
# link_state = p.getLinkState(robot_id, end_effector_index, computeForwardKinematics=True)
# current_pos = np.array(link_state[4]) 
# current_ori = np.array(link_state[5])
# current_mat = np.array(p.getMatrixFromQuaternion(current_ori)).reshape(3, 3)
# T_start = np.eye(4)
# T_start[:3, :3] = current_mat
# T_start[:3, 3] = current_pos
# print("T_start: \n", T_start)

# ### KINEMATICS ###

# # init
# kin = URDF_Kinematics()

# # get current joint positions (only movable joints, no fixed joints and no floating base pose) 
# # q_init = np.zeros(6)  # SO100
# print("q_init: ", np.rad2deg(q_init))

# # compute start pose
# T_start = kin.forward_kinematics(robot_model, q_init, target_link_name="gripper") # SO100
# print("\nT_start = \n", T_start)

# # # Define relative goal pose SO100
# # T_goal = T_start.copy()
# # T_goal[:3, 3] += np.array([0.0, 0.0, 0.2])
# # print("\nT_goal = \n", T_goal)

# # Define absolute goal pose SO100
# T_goal = np.array([
#     [1.0, 0.0, 0.0,  0.0],
#     [0.0, 1.0, 0.0, -0.17],
#     [0.0, 0.0, 1.0,  0.37],
#     [0.0, 0.0, 0.0,  1.0]
# ])
# print("\nT_goal = \n", T_goal)

# # IK with internal interpolation
# q_final = kin.inverse_kinematics(robot_model, q_init, T_goal, target_link_name="gripper", use_orientation=True, k=0.8, n_iter=1) # SO100
# print("\nFinal joint angles = ", q_final)

# T_final = kin.forward_kinematics(robot_model, q_final, target_link_name="gripper")  # SO100
# print("\nFinal pose direct kinematics = \n", T_final)

# print("\nerr_lin = ", RobotUtils.calc_lin_err(T_goal, T_final))
# print("err_ang = ", RobotUtils.calc_ang_err(T_goal, T_final))

# # raise an error in case joint limits are exceeded
# kin.check_joint_limits(robot_model, q_final)


# sim_time = 2.0  # seconds
# dt = 1/240.0
# steps = int(sim_time / dt)

# for t in range(steps):
#     s = t / steps
#     q_interp = (1-s)*q_init + s*q_final  # simple linear interpolation
#     for i, pos in zip(movable_joints, q_interp):
#         p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=pos, force=500, maxVelocity=2.0)
#     p.stepSimulation()
#     time.sleep(dt)






import mujoco
import mujoco.viewer
import os

# Get directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct relative path to URDF
xml_path = os.path.join(script_dir, "..", "models", "so100", "xml", "so100.xml")
print(xml_path)

# Load the model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("Number of joints:", model.njnt)
print("Number of actuators:", model.nu)
print("Number of nq:", model.nq) # dimension of generalized coordinates (qpos) --> total DOF in the system
print("Simulator sample time [s]: ", model.opt.timestep) # update frequency in the simulator equations

# list joint names
joint_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for i in range(model.njnt)
]
print("Joint names:", joint_names)

# list actuator names
actuator_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    for i in range(model.nu)
]
print("Actuator names:", actuator_names)

# current state
qpos = data.qpos.copy()   # joint positions
qvel = data.qvel.copy()   # joint velocities
print("qpos: ", qpos)
print("qvel: ", qvel)

# Launch the viewer
# viewer = mujoco.viewer.launch(model, data)

import numpy as np

with mujoco.viewer.launch_passive(model, data) as viewer:

    t0 = data.time

    while viewer.is_running():

        # Apply a simple sinusoidal control to joint 0
        data.ctrl[0] = 1 * np.sin(0.25*np.pi * (data.time - t0))
        
        mujoco.mj_step(model, data)
        viewer.sync()