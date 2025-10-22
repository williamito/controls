import unittest
import numpy as np
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.model import *
from scripts.kinematics import *
from scripts.dynamics import *

class TestKinematics(unittest.TestCase):
    def setUp(self):
        dh_loader = DH_loader() # instantiate DH loader
        self.robot = RobotModel(dh_loader) # wrap in RobotModel
        self.kin = DH_Kinematics()
        self.q_init = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2])
        self.q_init_dh = self.robot.convert_mech_to_dh(self.q_init)
        self.T_start = self.kin.forward_kinematics(self.robot, self.q_init_dh)

    def test_basic_usage(self):
        "test basic usage"
        T_goal = self.T_start.copy()
        T_goal[:3, 3] += np.array([0.0, 0.0, -0.1])
        q_final_dh = self.kin.inverse_kinematics(self.robot, self.q_init_dh, T_goal, use_orientation=True)
        q_final_mech = self.robot.convert_dh_to_mech(q_final_dh)
        self.kin.check_joint_limits(self.robot, q_final_mech)

    def test_unreachable_pose(self):
        "test assert error when final pose is far from T_goal"
        T_goal = self.T_start.copy()
        T_goal[:3, 3] += np.array([1.0, 0.1, 0.1])  # set an UNREACHABLE goal pose to trigger error
        # check method
        with self.assertRaises(AssertionError) as context:
            self.kin.inverse_kinematics(self.robot, self.q_init_dh, T_goal, use_orientation=False)
        # Check the correct ERROR is triggered
        self.assertIn("Large position error", str(context.exception))

    def test_joint_limits(self):
        "test assert error when joint limits are out of bound"
        T_goal = self.T_start.copy()
        T_goal[:3, 3] += np.array([0.3, -0.2, -0.3])
        q_final_dh = self.kin.inverse_kinematics(self.robot, self.q_init_dh, T_goal, use_orientation=False)
        q_final_mech = self.robot.convert_dh_to_mech(q_final_dh)
        # check method
        with self.assertRaises(AssertionError) as context:
            self.kin.check_joint_limits(self.robot, q_final_mech)
        # Check the correct ERROR is triggered
        self.assertIn("Joint limits out of bound", str(context.exception))

    def test_dh2mechanical(self):
        "validate dh2mechanical conversion against known values"
        q_dh = np.array([-1.57079633, 1.31859625, -1.31859625, -3.14159265, 0.0])
        expected_mech = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2])
        q_mech = self.robot.convert_dh_to_mech(q_dh)
        assert np.allclose(q_mech, expected_mech, atol=1e-5), f"Expected {expected_mech}, got {q_mech}"

    def test_mechanical2dh(self):
        "validate mechanical2dh conversion against known values"
        q_mech = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2])
        expected_dh = np.array([-1.57079633, 1.31859625, -1.31859625, -3.14159265, 0.0])
        q_dh = self.robot.convert_mech_to_dh(q_mech)
        assert np.allclose(q_dh, expected_dh, atol=1e-5), f"Expected {expected_dh}, got {q_dh}"

    def test_interpolator_slerp(self):
        "test SLERP interpolation with goal orientation triggering dot(q0,q1) < 0"
        
        # helper to create a homogeneous transform from translation + rotation
        def make_transform(t, rot_matrix):
            T = np.eye(4)
            T[:3, :3] = rot_matrix
            T[:3, 3] = t
            return T

        # define start and goal poses
        T_start = make_transform(
            t=np.array([0.0, 0.0, 0.0]),
            rot_matrix=np.eye(3)
        )
        T_goal = make_transform(
            t=np.array([0.5, 0.2, 0.3]),   
            rot_matrix = R.from_rotvec(np.pi * np.array([-1, 0, 0])).as_matrix()
        )

        # initialize interpolator
        n_steps = self.kin._interp_init(T_start, T_goal, freq=100, trans_speed=0.1, rot_speed=0.5)

        # collect interpolated poses
        interpolated_poses = []
        for i in range(n_steps + 1):
            T_interp = self.kin._interp_execute(i)
            interpolated_poses.append(T_interp)

        # extract rotation vectors
        rvecs = [R.from_matrix(T[:3, :3]).as_rotvec() for T in interpolated_poses]
        rvecs = np.array(rvecs) # N,3

        # print max angular jump between steps
        angular_jumps = np.linalg.norm(np.diff(rvecs, axis=0), axis=1)
        assert np.max(angular_jumps) < 1e-2, f"Got great jump during orientation interpolation {np.max(angular_jumps)}" 


if __name__ == "__main__":
    unittest.main()
