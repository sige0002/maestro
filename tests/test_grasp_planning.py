"""Geometry checks for the shared SO101 model, without ROS or model calls."""
from pathlib import Path
import unittest
import numpy as np


@unittest.skipUnless(Path('/so101/scripts/models/model.py').exists(), 'Shared SO101 assets required')
class PlanningTests(unittest.TestCase):
    def setUp(self):
        from maestro.so101.planning import GraspPlanner
        self.planner = GraspPlanner()

    def test_observed_obstacle_blocks_arm(self):
        p = self.planner
        q = np.array([0, 0, 0, 0, 0, .65])
        self.assertTrue(p.valid(q))
        p.pose(q)
        wrist = p.data.body('wrist_link').xpos.copy()
        p.set_objects({'blue': wrist, 'green': [.2, -.1, .013]}, 'green')
        self.assertFalse(p.valid(q))
        with self.assertRaises(ValueError):
            p.path(q, q)

    def test_robot_model_has_no_simulator_free_object_state(self):
        # Object ground truth cannot enter perception via this planning model.
        self.assertEqual(self.planner.model.nq, 6)

    def test_folded_home_and_return_with_released_cube(self):
        import yaml
        from model import HOME_RAD
        config = yaml.safe_load(Path('config/so101.example.yaml').read_text())
        home = np.asarray(config['home_rad'])
        np.testing.assert_allclose(home, HOME_RAD)
        p = self.planner
        self.assertTrue(p.valid(home))
        p.gravity_command(home)
        dest = [.13, -.24, p.floor_z+.019]
        p.set_objects({'green': dest, 'blue': [.24, -.06, p.floor_z+.015]}, 'green')
        p.target_geom = None
        retreat = p.ik([.13, -.24, p.floor_z+.08], home, .65)
        path = p.path(retreat, home)
        self.assertTrue(all(p.edge(a, b) for a, b in zip(path, path[1:])))

    def test_blocked_direct_path_gets_a_checked_detour(self):
        p = self.planner
        a = np.array([0,0,0,0,0,.65])
        b = p.ik([.2,-.1,p.floor_z+.06],a,.65)
        p.pose(a+.3*(b-a))
        obstacle = p.data.body('gripper_link').xpos.copy()
        p.set_objects({'blue': obstacle},'green')
        self.assertTrue(p.valid(a) and p.valid(b))
        self.assertFalse(p.edge(a,b))
        path = p.path(a,b)
        self.assertGreater(len(path),2)
        self.assertTrue(all(p.edge(x,y) for x,y in zip(path,path[1:])))

    def test_grasp_ik_and_edge_check(self):
        p = self.planner
        start = np.array([0, 0, 0, 0, 0, .65])
        target = np.array([.2, -.1, p.floor_z+.06])
        goal = p.ik(target, start, .65)
        self.assertLess(np.linalg.norm(p.pose(goal)-target), .003)
        path = p.path(start, goal)
        self.assertTrue(all(p.edge(a,b) for a,b in zip(path,path[1:])))
        with self.assertRaises(ValueError):
            p.ik([5, 0, 0], start, .65)
