import os
import time
import numpy as np
import mujoco
import mujoco.viewer

from scripts.model import *
from scripts.kinematics import *
from scripts.dynamics import *


# run from /controls folder directly: python -m simulator.main_so100_mj.py


### INIT MUJOCO ###

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


### INIT CONTROLS LIBRARY ###

# Get directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct relative path to URDF
urdf_path = os.path.join(script_dir, "..", "models", "so100", "urdf", "so100.urdf") 

# Instantiate URDF loader
urdf_loader = URDF_handler()
urdf_loader.load(urdf_path)

# Wrap in RobotModel
robot_model = RobotModel(urdf_loader)

# Info
robot_model.print_model_properties()
print("Number of joints: ", robot_model.get_n_joints())
print("Number of links: ", robot_model.get_n_links())


### SIMULATION INIT ###

# Read initial joint state
q_init = q_init = data.qpos.copy()
q_init = np.array(q_init) 
print("Start q: ", q_init)

# Get initial pose from Mujoco

# Make sure the model state is up to date
mujoco.mj_forward(model, data)

# Get Mujoco body id for the end effector (replace "gripper" with your EE name)
ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")

# Extract position and orientation
current_pos = data.xpos[ee_body_id].copy()     # shape (3,)
current_quat = data.xquat[ee_body_id].copy()   # shape (4,)

# Convert quaternion to rotation matrix
current_mat = np.zeros((3, 3))
mujoco.mju_quat2Mat(current_mat.ravel(), current_quat)

# Build homogeneous transform
T_start = np.eye(4)
T_start[:3, :3] = current_mat
T_start[:3, 3]  = current_pos

print("T_start:\n", T_start)

### KINEMATICS ###

# init
kin = URDF_Kinematics()

# get current joint positions (only movable joints, no fixed joints and no floating base pose) 
# q_init = np.zeros(6)  # SO100
print("q_init: ", np.rad2deg(q_init))

# compute start pose
T_start = kin.forward_kinematics(robot_model, q_init, target_link_name="gripper") # SO100
print("\nT_start = \n", T_start)

# # Define relative goal pose SO100
# T_goal = T_start.copy()
# T_goal[:3, 3] += np.array([0.0, 0.0, 0.2])
# print("\nT_goal = \n", T_goal)

# Define absolute goal pose SO100
T_goal = np.array([
    [1.0, 0.0, 0.0,  0.0],
    [0.0, 1.0, 0.0, -0.17],
    [0.0, 0.0, 1.0,  0.37],
    [0.0, 0.0, 0.0,  1.0]
])
print("\nT_goal = \n", T_goal)

# IK with internal interpolation
q_final = kin.inverse_kinematics(robot_model, q_init, T_goal, target_link_name="gripper", use_orientation=True, k=0.8, n_iter=1) # SO100
print("\nFinal joint angles = ", q_final)

T_final = kin.forward_kinematics(robot_model, q_final, target_link_name="gripper")  # SO100
print("\nFinal pose direct kinematics = \n", T_final)

print("\nerr_lin = ", RobotUtils.calc_lin_err(T_goal, T_final))
print("err_ang = ", RobotUtils.calc_ang_err(T_goal, T_final))

# raise an error in case joint limits are exceeded
kin.check_joint_limits(robot_model, q_final)


### SIMULATION INIT ###

sim_time = 2.0  # seconds
dt = model.opt.timestep
steps = int(sim_time / dt)

with mujoco.viewer.launch_passive(model, data) as viewer:

    # Run trajectory once
    for t in range(steps):

        if not viewer.is_running():
            break

        s = t / steps
        q_interp = (1 - s) * q_init + s * q_final
        data.ctrl[:] = q_interp
        
        mujoco.mj_step(model, data)
        viewer.sync()

    # Hold final position indefinitely
    while viewer.is_running():
        data.ctrl[:] = 0.0  # or q_final if using position control
        mujoco.mj_step(model, data)
        viewer.sync()