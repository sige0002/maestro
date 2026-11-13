"""SO101 contract tests; use SO101_URDF to select the canonical model."""
import os
from pathlib import Path
import unittest
from types import SimpleNamespace
import yaml

import numpy as np

from maestro.so101.kinematics import Kinematics
from maestro.so101.api import SO101RobotApi
from maestro.so101.perception import point_on_plane


URDF = Path(os.environ.get('SO101_URDF', '/so101/models/so101/so101_new_calib_camera.urdf'))


class PlaneTests(unittest.TestCase):
    def test_downward_camera_projection(self):
        pose = np.diag([1., -1., -1., 1.])
        pose[:3, 3] = [0, 0, 1]
        k = [[100, 0, 50], [0, 100, 50], [0, 0, 1]]
        np.testing.assert_allclose(point_on_plane([60, 70], k, [], pose, 0), [.1, -.2, 0])
        with self.assertRaisesRegex(ValueError, 'behind'):
            point_on_plane([50, 50], k, [], pose, 2)


@unittest.skipUnless(URDF.exists(), 'Set SO101_URDF to the canonical SO101 camera URDF')
class SO101ContractTests(unittest.TestCase):
    def setUp(self):
        self.kin = Kinematics(URDF)
        self.config = yaml.safe_load((Path(__file__).resolve().parents[1] /
                                      'config/so101.example.yaml').read_text())
        self.config['urdf'] = str(URDF)

    def test_invalid_targets_are_rejected_without_publishing(self):
        class NoPublish:
            def publish(self, positions):
                raise AssertionError('Invalid target reached ROS')
        api = SO101RobotApi(NoPublish(), self.config)
        for target in ([0]*5, [float('nan')]*6, [float('inf')]*6, [100]*6):
            with self.subTest(target=target), self.assertRaises(ValueError):
                api.move_joints(target)

    def test_bad_freshness_and_motion_settings_fail_before_connecting(self):
        for key in ('max_age_s', 'command_hz', 'max_velocity_rad_s',
                    'joint_tolerance_rad', 'settle_timeout_s'):
            for value in (float('nan'), 0, -1, True, 'fast'):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    SO101RobotApi(None, {**self.config, key: value})

    def test_unreachable_cartesian_target_fails(self):
        with self.assertRaisesRegex(ValueError, 'Unreachable'):
            self.kin.inverse([10, 0, 0], np.zeros(6))

    def test_gripper_does_not_change_fixed_finger_frame(self):
        closed = np.array([.1, .2, -.3, .4, -.2, 0])
        opened = closed.copy()
        opened[5] = .65
        np.testing.assert_allclose(self.kin.forward(closed), self.kin.forward(opened))

    def test_home_verification_uses_measured_all_six_joints(self):
        measured = np.asarray(self.config['home_rad'], dtype=float)
        transport = SimpleNamespace(snapshot=lambda **kwargs: {'joints': measured.copy()})
        api = SO101RobotApi(transport, self.config)
        self.assertTrue(api.verify_home()['returned_home'])
        for index in range(6):
            measured[index] += .1
            with self.assertRaisesRegex(RuntimeError, 'Initial pose not reached'):
                api.verify_home()
            measured[index] -= .1


if __name__ == '__main__':
    unittest.main()
