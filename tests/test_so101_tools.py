"""Tool contracts that protect physical grasp commands across object widths."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from maestro.so101.tools import RobotTools


class GripperToolTests(unittest.TestCase):
    def test_failed_open_does_not_claim_release_or_forget_carried_object(self):
        robot=RobotTools.__new__(RobotTools)
        robot.transport=SimpleNamespace(snapshot=lambda:{'joints':np.array([0,0,0,0,0,.2])})
        robot.holding=True;robot.selected=0;robot.grip_command=.17
        robot.objects=[{'center':np.array([.2,0,.02])}]
        robot.config={'gripper_open_rad':.65}
        robot.planner=SimpleNamespace(pose=lambda q:np.array([.1,-.2,.02]),set_carrying=Mock())
        robot._sync_obstacles=Mock()
        robot.motion=SimpleNamespace(execute=Mock(side_effect=ValueError('tray collision')))
        with self.assertRaisesRegex(ValueError,'tray collision'):
            RobotTools.open_gripper.__wrapped__(robot)
        self.assertTrue(robot.holding)
        self.assertEqual(robot.grip_command,.17)
        np.testing.assert_array_equal(robot.objects[0]['center'],[.2,0,.02])
        robot.planner.set_carrying.assert_called_with(True)

    def test_closing_stops_at_contact_and_keeps_pressure_command(self):
        for contact_angle in (.22,.08):
            with self.subTest(contact_angle=contact_angle):
                robot=RobotTools.__new__(RobotTools)
                measured=np.array([0,0,0,0,0,.65],dtype=float)
                robot.transport=SimpleNamespace(snapshot=lambda: {'joints':measured.copy()})
                robot.config={'gripper_closed_rad':0}
                commands=[]
                def execute(goal,*args,**kwargs):
                    commands.append(float(goal[5]))
                    measured[:]=goal
                    measured[5]=max(goal[5],contact_angle)
                robot.motion=SimpleNamespace(execute=execute)
                robot.selected=0
                robot.objects=[{'half_size':np.array([.02,.01,.015])}]
                model=SimpleNamespace(geom_size=np.zeros((1,3)),geom_rbound=np.zeros(1),geom_aabb=np.zeros((1,6)))
                planner=SimpleNamespace(model=model,carried_geom=0,hand=0,pose=lambda q:None,
                    data=SimpleNamespace(body=lambda i:SimpleNamespace(xmat=np.eye(3).ravel())),set_carrying=Mock())
                robot.planner=planner
                robot._sync_obstacles=lambda:None
                result=RobotTools.close_gripper.__wrapped__(robot)
                self.assertTrue(result['grasp_detected'])
                self.assertGreater(contact_angle-robot.grip_command,.015)
                self.assertLess(contact_angle-robot.grip_command,.046)
                self.assertEqual(commands[-1],robot.grip_command)
                self.assertNotEqual(robot.grip_command,measured[5])
                planner.set_carrying.assert_called_once_with(True)
