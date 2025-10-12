import os
import time
import numpy as np
import mujoco
import mujoco.viewer

from scripts.model import *
from scripts.kinematics import *
from scripts.dynamics import *


# 1) try others urdf/xml
# 2) print dkine mujoco/lib con q_pos != 0


### INIT MUJOCO ###

# Get directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct relative path to URDF
xml_path = os.path.join(script_dir, "..", "models", "g1", "g1_29dof_rev_1_0.xml") 
print(xml_path)

# Load the model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("\nMUJOCO DATA:\n")
print("Number of joints:", model.njnt)
print("Number of actuators:", model.nu)
print("Number of nq:", model.nq) # dimension of generalized coordinates (qpos) --> total DOF in the system
print("Number of links:", model.nbody)
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

# list link names
link_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    for i in range(model.nbody)
]
print("Link names:", link_names)

# current state
qpos = data.qpos.copy()   # joint positions
qvel = data.qvel.copy()   # joint velocities
print("qpos: ", qpos)
print("qvel: ", qvel)

### INIT CONTROLS LIBRARY ###

# Get directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct relative path to URDF
urdf_path = os.path.join(script_dir, "..", "models", "g1", "g1_29dof_rev_1_0.urdf") 

# Instantiate URDF loader
urdf_loader = URDF_loader()
urdf_loader.load(urdf_path)

# Wrap in RobotModel
robot_model = RobotModel(urdf_loader)

# Info
print("\nCONTROLS LIB DATA:\n")
robot_model.print_model_properties()
print("Number of joints: ", robot_model.get_n_joints())
print("Number of links: ", robot_model.get_n_links())

### SIMULATION INIT ###

print("\nSIMULATION\n")

# turn off gravity
model.opt.gravity[:] = 0
data.qpos[:7] = np.array([0. ,   0. ,   0.793 ,  1. ,   0.   , 0. ,   0.])
data.qvel[:6] = 0 # zero linear/angular base velocity

# Read initial joint state
q_init = data.qpos.copy()
q_init = np.array(q_init) 
print("q_init [RAD]: ", q_init)
print("q_init [DEG]: ", np.rad2deg(q_init))

# urdf2mujoco conversion due to joint names different order
q = q_init[7:].copy()

# Make sure the model state is up to date
mujoco.mj_forward(model, data)

# Get Mujoco body id for the end effector 
ee_name_mj = "right_wrist_pitch_link" 
ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_name_mj)
assert ee_body_id !=-1, (f"[ERROR] {ee_name_mj} is not a valid EE link name available in .xml file.")

# Extract position and orientation
current_pos = data.xpos[ee_body_id].copy()     
current_quat = data.xquat[ee_body_id].copy()  
current_mat = np.zeros((3, 3))
mujoco.mju_quat2Mat(current_mat.ravel(), current_quat)

# Build homogeneous transform
T_start = np.eye(4)
T_start[:3, :3] = current_mat
T_start[:3, 3]  = current_pos
print("T_start Mujoco:\n", T_start)

### KINEMATICS ###

# init
kin = URDF_Kinematics()

# define ee target link name
ee_name_cl = "right_wrist_pitch_link"

# compute start pose
T_start = kin.forward_kinematics(robot_model, q, target_link_name=ee_name_cl) 
print("\nT_start Lib:\n", T_start)

# Define relative goal pose (baseTn)
T_goal = T_start.copy()
T_goal[:3, 3] += np.array([0.1, 0.0, 0.0])
# rot_rel = R.from_euler('zy', [90, 0], degrees=True).as_matrix()
# T_goal[:3, :3] = T_start[:3, :3] @ rot_rel # rotation wrt EE frame
print("\nT_goal = \n", T_goal)

# # Define absolute goal pose (baseTn)
# T_goal = np.array([
#     [0.0721, -0.8863, -0.4575,  0.15],
#     [0.0336,  0.4606, -0.887, 0.22],
#     [0.9968,  0.0486,  0.063,  0.22],
#     [0.0, 0.0, 0.0,  1.0]
# ])
# print("\nT_goal = \n", T_goal)

### SIMULATION START ###

# ikine params
use_orientation=True
k=0.8
n_iter=50

# init interpolator
n_steps = kin._interp_init(kin._forward_kinematics_baseTn(robot_model, q, ee_name_cl), T_goal, freq = 1.0/model.opt.timestep, trans_speed = 0.3, rot_speed = 0.3)

with mujoco.viewer.launch_passive(model, data) as viewer:

    # import time
    # time.sleep(4)

    # Run trajectory once
    for i in range(0, n_steps + 1):

        if not viewer.is_running():
            break

        # current setpoint as baseTn
        T_desired_interp = kin._interp_execute(i)

        # get updated joint positions
        q = kin._inverse_kinematics_step_baseTn(robot_model, q, T_desired_interp, ee_name_cl, use_orientation, k, n_iter)

        # go back in Mujoco domain
        data.ctrl[:] = q
        data.qpos[:7] = np.array([0. ,   0. ,   0.793 ,  1. ,   0.   , 0. ,   0.])
        data.qvel[:6] = 0 # zero linear/angular base velocity
        
        mujoco.mj_step(model, data)
        viewer.sync()

    # Hold final position indefinitely
    while viewer.is_running():

        # check final error
        T_current = kin._forward_kinematics_baseTn(robot_model, q, ee_name_cl)
        err_lin = RobotUtils.calc_lin_err(T_current, T_goal)
        lin_error_norm = np.linalg.norm(err_lin)
        assert lin_error_norm < 1e-2, (f"[ERROR] Large position error ({lin_error_norm:.4f}). Check target reachability (position/orientation)")

        # apply last action
        data.ctrl[:] = q 
        data.qpos[:7] = np.array([0. ,   0. ,   0.793 ,1. ,   0.   , 0. ,   0.])
        data.qvel[:6] = 0 # zero linear/angular base velocity
        mujoco.mj_step(model, data)
        viewer.sync()




