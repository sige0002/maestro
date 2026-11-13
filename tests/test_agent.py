"""Shared Maestro execution: feedback-driven correction and persistent namespace."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from maestro.agent import VLMCodeExtractor, Replan, execute_plan_loop
from maestro.so101.agent_runner import TrialLogger


class AgentTests(unittest.TestCase):
    def test_error_reaches_model_and_corrected_code_runs_with_persistent_state(self):
        messages=[]
        responses=[
            Replan(reason_and_plan='Correct failed key',success=False,task_complete=False,
                   actor_code='answer = saved["value"] + 1\nprint(answer)'),
            Replan(reason_and_plan='Image and state verify outcome',success=True,task_complete=True,actor_code='')]
        def send(message,schema):
            messages.append(message)
            r=responses.pop(0)
            return r,r.model_dump_json()
        robot=SimpleNamespace(instruction='test',get_robot_state=lambda: {},replan_hints=[])
        editor=SimpleNamespace(feedback=lambda: np.zeros((10,10,3),dtype=np.uint8))
        with tempfile.TemporaryDirectory() as tmp:
            agent=VLMCodeExtractor(SimpleNamespace(send=send),robot,editor,TrialLogger(tmp))
            agent.actor_code='saved = {"value": 4}\nprint(saved["wrong"])'
            self.assertTrue(execute_plan_loop(agent,3))
            self.assertEqual(agent.local_vars['answer'],5)
            self.assertIn('KeyError',str(messages[0]))
            self.assertEqual(len(list(Path(tmp).glob('*-actor.py'))),2)

    def test_model_claim_does_not_bypass_home_guard(self):
        result=Replan(reason_and_plan='done',success=True,task_complete=True,actor_code='')
        chat=SimpleNamespace(send=lambda *args:(result,result.model_dump_json()))
        robot=SimpleNamespace(instruction='test',get_robot_state=lambda:{},replan_hints=[])
        editor=SimpleNamespace(feedback=lambda:np.zeros((10,10,3),dtype=np.uint8))
        with tempfile.TemporaryDirectory() as tmp:
            agent=VLMCodeExtractor(chat,robot,editor,TrialLogger(tmp));agent.actor_code='print("done")'
            self.assertFalse(execute_plan_loop(agent,2,completion_guard=lambda:'not home'))
