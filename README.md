# Robot Kinematics Module

This module provides an implementation of forward and inverse kinematics for robotic manipulators using Denavit–Hartenberg (DH) parameters. It supports:

- **Forward kinematics**
- **Inverse kinematics via Damped Least Squares Inverse Jacobian**
- **Pose interpolation (position + orientation) using SLERP**

An example configuration is included for the SO100 robotic arm from [LeRobot](https://github.com/huggingface/lerobot), but the module is adaptable to any N-dof robot defined by standard DH parameters.

---

## Classes

### `Robot`
Defines a robot model from a predefined set (e.g. `"so100"`), with attributes:

- `dh_table`: DH table as a list of $[ \theta, d, a, \alpha ]$ entries.
- `mech_joint_limits_low`: mechanical joint position limits lower bound
- `mech_joint_limits_up`: mechanical joint position limits upper bound
- `worldTbase`: 4x4 homogeneous transform.
- `nTtool`: 4x4 homogeneous transform.
- `from_dh_to_mech()`: DH angles to mechanical angles conversion.
- `from_dh_to_mech()`: mechanical angles to DH angles conversion.
- `check_joint_limits()`: check joint limits.

---

### `RobotUtils`
Collection of static methods:
- `calc_dh_matrix(dh, θ)`: returns the homogeneous transform using standard DH convention.
- `calc_lin_err(T1, T2)`: linear position error.
- `calc_ang_err(T1, T2)`: angular error.
- `inv_homog_mat(T)`: efficiently inverts a 4x4 transformation.
- `calc_geom_jac(...)`: compute geometrical Jacobian wrt base-frame.
- `dls_right_pseudoinv(...)`: Damped Least Squares pseudoinverse.

---

### `RobotKinematics`
Main class for computing kinematics:

#### `forward_kinematics(robot, q)`
Returns the tool pose in the world frame:

$$
^{world}T_{tool} = ^{world}T_{base} \cdot ^{base}T_n(q) \cdot ^nT_{tool}
$$

#### `inverse_kinematics(...)`
Computes inverse kinematics using iterative pose interpolation and inverse Jacobian method. Optional orientation tracking.

#### Internal helpers:
- `_forward_kinematics_baseTn`: computes fkine from base-frame to n-frame.
- `_inverse_kinematics_step_baseTn`: performs one step of iterative IK.
- `_interp_init`, `_interp_execute`: Pose interpolation (position + orientation).

---

## DH Frames

DH (Denavit–Hartenberg) frames provide a systematic and compact way to model the kinematics of robotic arms. Each joint and link transformation is represented by a standard set of parameters, allowing for consistent and scalable computation of **forward kinematics, inverse kinematics and Jacobians**.

Each robot uses the **standard DH convention**:

- $\theta$: variable joint angle (actuated)
- $d, a, \alpha$: constant link parameters defined by the mechanical structure

Once the DH table is ready, the homogeneous transformation from frame( i-1 ) to frame( i ) is:

$$
^{i-1}A_{i} =
\begin{bmatrix}
\cos\theta & -\sin\theta\cos\alpha & \sin\theta\sin\alpha & a\cos\theta \\
\sin\theta & \cos\theta\cos\alpha & -\cos\theta\sin\alpha & a\sin\theta \\
0 & \sin\alpha & \cos\alpha & d \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

Finally, the forward kinematics is computed as:

$$
^{base}T_{n}(q) = A_1(\theta_1) \cdot A_2(\theta_2) \cdot \dots \cdot A_n(\theta_n)
$$

### SO100 DH Frames

<p align="center">
  <img src="./images/dh1.png" alt="DH"/><br>
  <em>Figure: DH frames and DH table computed for the SO100 robotic arm</em>
</p>

<p align="center">
  <img src="./images/dh2.png" alt="DH"/><br>
  <em>Figure: DH angles Vs mechanical angles</em>
</p>

---

## Example (in `main.py`)

- Initializes the `"so100"` robot model.
- Transform mechanical angles in DH angles
- Define a goal pose worldTtool.
- Solves IK with only position tracking.
- Prints joint angles and final pose with direct kinematics.
- Transform DH angles in mechanical angles
- Check mechanical angles are within their physical limits

---

## Full contributions

- Forward kinematics using DH tables
- Jacobian computation using DH tables.
- Inverse kinematics using Jacobian and dump-least square method to avoid singularities.
- Pose interpolation: linear (position) + SLERP (orientation)
- DH angles to mechanical angles conversion (and viceversa).
- Out of Bound joint position limits checker
