"""The benchmark score must require both settled placement and measured return."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from maestro.so101.evaluate import evaluate, inside_tray


class EvaluationTests(unittest.TestCase):
    def test_whole_cube_must_fit_inside_tray(self):
        self.assertTrue(inside_tray([.14, -.24]))
        self.assertTrue(inside_tray([.164, -.216]))
        self.assertFalse(inside_tray([.166, -.24]))
        self.assertFalse(inside_tray([.14, -.266]))
        self.assertFalse(inside_tray([float('nan'), -.24]))

    def score(self, points, home):
        transport = SimpleNamespace(
            snapshot=lambda: {'overview': np.zeros((4, 4, 3), dtype=np.uint8),
                              'wrist': np.zeros((4, 4, 3), dtype=np.uint8), 'joints': np.zeros(6)},
            camera_geometry=lambda name: (SimpleNamespace(k=None, d=None), None))
        robot = SimpleNamespace(transport=transport, verify_home=Mock(side_effect=home))
        with tempfile.TemporaryDirectory() as tmp, \
                patch('maestro.so101.planning.GraspPlanner', return_value=SimpleNamespace(floor_z=0)), \
                patch('maestro.so101.evaluate.color_top', return_value=([0, 0], None)), \
                patch('maestro.so101.evaluate.point_on_plane', side_effect=points), \
                patch('maestro.so101.evaluate.time.sleep'):
            return evaluate(robot, 'green', Path(tmp))

    def test_placement_without_return_fails(self):
        p = np.array([.14, -.24, .034])
        result = self.score([p, p], [RuntimeError('not home'), RuntimeError('not home')])
        self.assertFalse(result['success'])

    def test_placement_that_does_not_remain_in_tray_fails(self):
        points = [np.array([.14, -.24, .034]), np.array([.20, -.1, .034])]
        self.assertFalse(self.score(points, [{'returned_home': True}]*2)['success'])

    def test_stable_placement_and_return_pass(self):
        p = np.array([.14, -.24, .034])
        self.assertTrue(self.score([p, p], [{'returned_home': True}]*2)['success'])
