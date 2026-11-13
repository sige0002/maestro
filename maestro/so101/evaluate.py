"""Read-only, camera-based evaluator for the fixed SO101 Cell cube benchmark.

No model completion claims or simulator dynamic object state enter this score.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import yaml

from .api import SO101RobotApi
from .manipulation import color_top
from .perception import point_on_plane
from .transport import RosTransport


def inside_tray(top_xy):
    # Cell tray centre (.14, -.24), inner half-width .04, cube half-width .015.
    # The whole 30 mm cube must fit, assuming its faces remain axis-aligned.
    return bool(np.all(np.abs(np.asarray(top_xy)-[.14, -.24]) <= .025))


def evaluate(robot, color, directory):
    directory.mkdir(parents=True, exist_ok=True)
    from .planning import GraspPlanner
    floor_z = GraspPlanner().floor_z
    samples = []
    for index in range(2):
        if index:
            time.sleep(2)
        obs = robot.transport.snapshot()
        info, camera = robot.transport.camera_geometry('overview')
        Image.fromarray(obs['overview']).save(directory / f'eval-{index}-overview.png')
        Image.fromarray(obs['wrist']).save(directory / f'eval-{index}-wrist.png')
        try:
            uv, _ = color_top(obs['overview'], color)
            # Measured fixture floor height; cube top on the tray's 4 mm floor.
            top = point_on_plane(uv, info.k, info.d, camera, floor_z+.034)
            placed = inside_tray(top[:2])
            placement = {'top_base_link_m': top.tolist(), 'inside_tray': placed}
        except ValueError as error:
            placement = {'inside_tray': False, 'error': str(error)}
        try:
            home = robot.verify_home()
        except RuntimeError as error:
            home = {'returned_home': False, 'error': str(error)}
        samples.append({'placement': placement, 'home': home,
                        'joints_rad': obs['joints'].tolist()})
    result = {'color': color, 'source': 'live_camera_and_joint_feedback', 'samples': samples,
              'success': all(s['placement']['inside_tray'] and s['home']['returned_home'] for s in samples)}
    (directory / 'evaluation.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/so101.yaml')
    parser.add_argument('--color', choices=('red', 'green', 'blue'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if not config.get('use_sim_time') or not config.get('sim_manipulation'):
        raise ValueError('This evaluator is for the calibrated MuJoCo Cell scene only')
    transport = RosTransport(config)
    try:
        transport.wait_ready()
        robot = SO101RobotApi(transport, config)
        print(json.dumps(evaluate(robot, args.color, args.output), indent=2))
    finally:
        transport.close()


if __name__ == '__main__':
    main()
