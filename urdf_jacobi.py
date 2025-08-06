# -*- coding: utf-8 -*-

import pybullet as p
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import os

from kinematics_lib import *



### INIT PYBULLET ###

# pybullet params
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)

# Get directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct relative path to URDF
urdf_path = os.path.join(script_dir, "..", "robot-urdfs", "merged_robots", "merged_robots.urdf")

# Load URDF
robot_id = p.loadURDF(urdf_path, useFixedBase=True)

# Find index corresponding to the flange
effector_name = "flange"
end_effector_index = -1

for i in range(p.getNumJoints(robot_id)):
    joint_info = p.getJointInfo(robot_id, i)
    link_name = joint_info[12].decode("utf-8")
    if link_name == effector_name:
        end_effector_index = i
        break

print("End effector index:", end_effector_index)

# Find indeces corresponding to movable joints
movable_joints = []
for i in range(p.getNumJoints(robot_id)):
    joint_info = p.getJointInfo(robot_id, i)
    joint_type = joint_info[2]
    if joint_type in [p.JOINT_PRISMATIC, p.JOINT_REVOLUTE]:
        movable_joints.append(i)

N_DOF = len(movable_joints)


time.sleep(10)

### SIMULATION ###

# Init
dt = 0.01
t = 0
n_iter = 10


# utils Class
utils = RobotUtils()

# Target pose as homogeneous matrix
target_pos = np.array([4.7, 0.5, 2.5]) # -> it starts in: [2.713 0.43  2.81 ]
target_ori = p.getQuaternionFromEuler([0, 0, 0])
target_mat = np.array(p.getMatrixFromQuaternion(target_ori)).reshape(3, 3)
T_final = np.eye(4)
T_final[:3, :3] = target_mat
T_final[:3, 3] = target_pos

# Read initial joint state
current_q = [p.getJointState(robot_id, i)[0] for i in movable_joints]
q = np.array(current_q)

# Get initial pose
link_state = p.getLinkState(robot_id, end_effector_index, computeForwardKinematics=True)
current_pos = np.array(link_state[4])
current_ori = np.array(link_state[5])
current_mat = np.array(p.getMatrixFromQuaternion(current_ori)).reshape(3, 3)
T_start = np.eye(4)
T_start[:3, :3] = current_mat
T_start[:3, 3] = current_pos

# Plotting errors
e_pos = []
e_ang = []
e_pos_step = []
e_ang_step = []

# Init interpolator
sim_time = utils.interp_init(T_start, T_final)

while t <= sim_time:
    
    # Interpolation
    T_des = utils.interp_execute(t)

    for _ in range(n_iter):
        
        # Get current FK (for Jacobian and orientation)
        link_state = p.getLinkState(robot_id, end_effector_index, computeForwardKinematics=True)
        current_pos = np.array(link_state[4])
        current_ori = np.array(link_state[5])
        current_mat = np.array(p.getMatrixFromQuaternion(current_ori)).reshape(3, 3)
        T_current = np.eye(4)
        T_current[:3, :3] = current_mat
        T_current[:3, 3] = current_pos
    
        # Compute linear and angular errors
        err_lin = utils.calc_lin_err(T_current, T_des)
        err_ang = utils.calc_ang_err(T_current, T_des)
        error = np.concatenate((err_lin, err_ang))
        e_pos_step.append(np.linalg.norm(err_lin))
        e_ang_step.append(np.linalg.norm(err_ang))
        
        # Stop if converged
        if np.linalg.norm(error) < 1e-5:
            break
            
        # Jacobian (from PyBullet)
        zero_vec = [0.0] * N_DOF
        J_lin, J_ang = p.calculateJacobian(robot_id, end_effector_index, [0, 0, 0], list(q), zero_vec, zero_vec)
        J = np.concatenate((J_lin, J_ang), axis=0)[:, :N_DOF]
    
        # DLS pseudoinverse
        J_pinv = utils.dls_right_pseudoinv(J)
    
        # IK update
        q += 0.1 * (J_pinv @ error)
    
        # Apply to simulation
        for i, pos in zip(movable_joints, q):
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=pos)

        # Step simulation
        p.stepSimulation()
        time.sleep(1 / 240.0)
    
    t += dt
    
    # Average errors over iteration cycle
    e_pos.append(sum(e_pos_step)/len(e_pos_step))
    e_ang.append(sum(e_ang_step)/len(e_ang_step))

    # Debug
    state = p.getLinkState(robot_id, end_effector_index, computeForwardKinematics=True)
    actual_pos = state[4]
    print(f"Target: {np.round(T_des[:3, 3], 3)} | Actual: {np.round(actual_pos, 3)}")


### PLOT ERRORS ###

# Timeline in seconds
time = np.arange(len(e_pos)) * dt

# Plot linear error
plt.figure(figsize=(8, 4))
plt.plot(time, e_pos, label='Linear Error [m]', color='blue')
plt.axhline(y=0.01, color='red', linestyle='--', label='Threshold 0.01 m') 
plt.xlabel('Time [s]')
plt.ylabel('Linear Error [m]')
plt.title('Average Linear Position Error Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot angular error
plt.figure(figsize=(8, 4))
plt.plot(time, e_ang, label='Angular Error [rad]', color='green')
plt.axhline(y=0.01, color='red', linestyle='--', label='Threshold 0.01 rad') 
plt.xlabel('Time [s]')
plt.ylabel('Angular Error [rad]')
plt.title('Average Angular Orientation Error Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

