"""Run Maestro's VLM plan / execute / observe loop on ROS 2 SO101."""
import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import traceback

import numpy as np
from PIL import Image
from pydantic import BaseModel
import yaml

from .api import SO101RobotApi
from .transport import RosTransport


class Plan(BaseModel):
    reason_and_plan: str
    task_complete: bool
    actor_code: str


def observe(robot, directory, step):
    snapshot = robot.transport.snapshot()
    state = robot.get_robot_state()
    message = [f'Measured robot state (absolute rad, base_link metres): {json.dumps(state)}']
    for name in robot.config['cameras']:
        image = Image.fromarray(snapshot[name])
        image.save(directory / f'{step:02d}-{name}.png')
        message.extend([f'Camera: {name}', image])
    (directory / f'{step:02d}-state.json').write_text(json.dumps(state, indent=2))
    return message


def run(robot, instruction, directory, max_steps):
    from maestro import vlm
    chat = vlm.planner_chat()
    prompt = Path(__file__).with_name('prompt.txt').read_text()
    namespace = {'robot': robot, 'np': np}
    feedback = 'No actions have been executed yet.'
    for step in range(max_steps + 1):
        message = [prompt, f'User instruction: {instruction}', f'Last execution: {feedback}',
                   *observe(robot, directory, step)]
        plan, raw = chat.send(message, Plan)
        (directory / f'{step:02d}-plan.json').write_text(raw)
        print(plan.reason_and_plan, flush=True)
        if plan.task_complete:
            try:
                home = robot.verify_home()
            except RuntimeError:
                # Completion includes return, even if generated actor code omitted it.
                # Reobserve afterwards so the model checks the outcome after motion.
                robot.reset()
                feedback = 'Returned to initial pose. Verify the task outcome in the new images.'
                continue
            return {'status': 'model_reported_complete', 'steps': step,
                    'reason': plan.reason_and_plan, 'home': home, 'commands': robot.command_log}
        if step == max_steps:
            break
        (directory / f'{step:02d}-actor.py').write_text(plan.actor_code)
        output = io.StringIO()
        try:
            # Same coding-agent execution model as the original Maestro runner.
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(compile(plan.actor_code, '<maestro-actor>', 'exec'), namespace, namespace)
            feedback = output.getvalue() or 'Code finished without output.'
        except Exception:
            feedback = output.getvalue() + traceback.format_exc()
        print(feedback, flush=True)
        (directory / f'{step:02d}-execution.txt').write_text(feedback)
    return {'status': 'step_limit', 'commands': robot.command_log}


def check_vlm_credentials():
    """Fail before opening ROS when a requested model has no credentials."""
    import os
    from maestro.config import cfg
    provider = cfg.vlm.provider
    if provider == 'local':
        import requests
        settings = cfg.vlm.local
        response = requests.get(settings['base_url'].rstrip('/') + '/models', timeout=10)
        response.raise_for_status()
        if settings['model'] not in [m['id'] for m in response.json()['data']]:
            raise RuntimeError('Configured local VLM is not loaded')
        return
    keys = {'gemini': 'GEMINI_KEY', 'astra': 'OPENAI_API_KEY'}
    if provider not in keys:
        raise ValueError(f'Unsupported Maestro VLM provider: {provider}')
    key = keys[provider]
    if not os.environ.get(key, '').strip():
        raise RuntimeError(f'Set {key} in Maestro .env before running natural-language instructions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/so101.example.yaml')
    parser.add_argument('-i', '--instruction')
    parser.add_argument('--check', action='store_true', help='Read live images/state without commanding motion')
    parser.add_argument('--smoke', action='store_true', help='Small joint move and return; no VLM')
    parser.add_argument('--max-steps', type=int, default=20)
    parser.add_argument('--output', default='runs/so101')
    parser.add_argument('--record', action='store_true', help='Record live overview/wrist video')
    args = parser.parse_args()
    if sum((bool(args.instruction), args.check, args.smoke)) != 1:
        parser.error('Choose one of --instruction, --check, --smoke')
    if not 1 <= args.max_steps <= 100:
        parser.error('--max-steps must be between 1 and 100')
    config = yaml.safe_load(Path(args.config).read_text())
    directory = Path(args.output) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory.mkdir(parents=True)
    transport = None
    video = None
    try:
        if args.instruction:
            check_vlm_credentials()
        # Validate motion configuration and URDF before creating ROS resources.
        robot = SO101RobotApi(None, config)
        transport = RosTransport(config)
        robot.transport = transport
        transport.wait_ready()
        robot.directory = directory
        if args.record:
            from .recording import TrialVideo
            video = TrialVideo(transport, directory / 'live.mp4')
            robot.video = video
            video.start()
        observe(robot, directory, 0)
        if args.check:
            result = {'status': 'observations_ready', 'state': robot.get_robot_state()}
        elif args.smoke:
            start = transport.snapshot()['joints']
            goal = start.copy()
            goal[0] += .12
            robot.move_joints(goal)
            observe(robot, directory, 1)
            robot.move_joints(start)
            observe(robot, directory, 2)
            result = {'status': 'motion_smoke_passed', 'commands': robot.command_log,
                      'vlm_used': False}
        else:
            robot.reset()
            result = run(robot, args.instruction, directory, args.max_steps)
        (directory / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({'run': str(directory), **result}, indent=2), flush=True)
        if result['status'] == 'step_limit':
            raise SystemExit(2)
    except Exception as error:
        (directory / 'result.json').write_text(json.dumps({
            'status': 'failed', 'error': str(error), 'instruction': args.instruction}, indent=2))
        raise
    finally:
        if video is not None:
            video.close()
        if transport is not None:
            transport.close()


if __name__ == '__main__':
    main()
