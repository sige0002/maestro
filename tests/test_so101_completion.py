"""Completion must wait for measured home and reobservation after return."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from maestro.so101.runner import Plan, run


class CompletionTests(unittest.TestCase):
    def test_model_completion_requires_return_and_new_observation(self):
        plan = Plan(reason_and_plan='placed', task_complete=True, actor_code='')
        chat = SimpleNamespace(send=Mock(return_value=(plan, plan.model_dump_json())))
        home = {'returned_home': True, 'max_error_rad': .002}
        robot = SimpleNamespace(verify_home=Mock(side_effect=[RuntimeError('not home'), home]),
                                reset=Mock(), command_log=[])
        with tempfile.TemporaryDirectory() as tmp, \
                patch('maestro.vlm.planner_chat', return_value=chat), \
                patch('maestro.so101.runner.observe', return_value=[]) as observe:
            result = run(robot, 'place cube', Path(tmp), 2)
        self.assertEqual(result['home'], home)
        robot.reset.assert_called_once()
        self.assertEqual(observe.call_count, 2)
        self.assertEqual(chat.send.call_count, 2)

    def test_failed_return_cannot_report_completion(self):
        plan = Plan(reason_and_plan='placed', task_complete=True, actor_code='')
        chat = SimpleNamespace(send=lambda *args: (plan, plan.model_dump_json()))
        robot = SimpleNamespace(verify_home=Mock(side_effect=RuntimeError('not home')),
                                reset=Mock(side_effect=RuntimeError('blocked')), command_log=[])
        with tempfile.TemporaryDirectory() as tmp, \
                patch('maestro.vlm.planner_chat', return_value=chat), \
                patch('maestro.so101.runner.observe', return_value=[]), \
                self.assertRaisesRegex(RuntimeError, 'blocked'):
            run(robot, 'place cube', Path(tmp), 2)
