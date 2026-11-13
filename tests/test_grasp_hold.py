"""Prevent unloading the gripper when starting a new carried-object segment."""
import unittest
import importlib.util
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'MuJoCo planning dependency required')
class GraspHoldTests(unittest.TestCase):
    def test_contact_blocked_measured_jaw_is_not_reissued_as_open_command(self):
        from maestro.so101.manipulation import TabletopManipulation
        commands = []
        # The cube holds the measured jaw at .12 while the commanded close is .05.
        measured = np.array([0, 0, 0, 0, 0, .12])
        transport = SimpleNamespace(snapshot=lambda: {'joints': measured.copy()},
                                    publish=lambda q: commands.append(q.copy()))
        planner = SimpleNamespace(model=SimpleNamespace(geom_contype=np.array([1])),
                                  carried_geom=0, check_count=0,
                                  path=lambda a,b: [a.copy(),b.copy()],
                                  gravity_command=lambda q: q.copy())
        robot = SimpleNamespace(transport=transport, kinematics=SimpleNamespace(validate=lambda q:q))
        tool = TabletopManipulation.__new__(TabletopManipulation)
        tool.robot, tool.transport, tool.planner, tool.events = robot, transport, planner, []
        with patch('maestro.so101.manipulation.time.sleep'):
            tool.execute(np.array([.02,0,0,0,0,.05]), 'Carry', contact=True,
                         minimum_duration=.1, settle=0)
        self.assertGreater(len(commands), 1)
        np.testing.assert_allclose([q[5] for q in commands], .05, atol=1e-12)
