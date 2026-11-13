"""Post-trial camera/geometry placement and measured-home check; no motion."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from maestro import vlm
from maestro.perception_utils import draw_segmentation, segment

from .api import SO101RobotApi
from .planning import GraspPlanner
from .rgbd import point_cloud, select_pointed_mask
from .transport import RosTransport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/so101.yaml')
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    sources=[Path(__file__),Path(__file__).with_name('rgbd.py')]
    (args.output/'manifest.json').write_text(json.dumps({'label':args.label,
        'robot_config':config,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2))
    transport = RosTransport(config)
    try:
        transport.wait_ready()
        api = SO101RobotApi(transport, config)
        planner = GraspPlanner()
        tray = planner.model.body('tray').pos.copy()
        frames = []
        for i in range(2):
            rgb, depth, stamp = transport.rgbd()
            info, camera = transport.camera_geometry('overview')
            detections = segment(rgb, args.label, service=config['segmentation_backend'])
            if not len(detections['scores']):
                raise RuntimeError('Evaluator could not segment the requested object')
            pointing = vlm.point([rgb],[args.label])
            if len(pointing) != 1:
                raise RuntimeError('Evaluator could not unambiguously point to the requested object')
            y,x = pointing[0]['point']
            best = select_pointed_mask(detections,[x*(rgb.shape[1]-1)/1000,y*(rgb.shape[0]-1)/1000])
            cloud = point_cloud(depth, info.k, camera)
            points = cloud[detections['masks'][best].astype(bool) & np.isfinite(cloud).all(axis=-1)]
            if len(points) < 25:
                raise RuntimeError('Insufficient depth for placement verification')
            lo, hi = np.percentile(points, [1, 99], axis=0)
            # Static tray interior is 80 x 80 mm; dynamic objects are camera-derived.
            inside = bool(np.all(lo[:2] >= tray[:2]-.04) and np.all(hi[:2] <= tray[:2]+.04)
                          and hi[2] > tray[2]+.006 and hi[2] < tray[2]+.10)
            home = api.verify_home()
            frames.append({'visible_extent_min_m':lo.tolist(), 'visible_extent_max_m':hi.tolist(),
                           'inside_tray':inside, 'home':home, 'score':float(detections['scores'][best]),
                           'selected_index':best,'pointing':pointing,
                           'sensor_stamp_s':stamp.sec+stamp.nanosec*1e-9})
            Image.fromarray(rgb).save(args.output/f'{i}-overview.png')
            Image.fromarray(draw_segmentation(rgb,detections)).save(args.output/f'{i}-segmentation.png')
            selected={k:detections[k][best:best+1] for k in ('masks','boxes','scores','labels')}
            Image.fromarray(draw_segmentation(rgb,selected)).save(args.output/f'{i}-selected.png')
            np.savez_compressed(args.output/f'{i}-detections.npz',**detections)
            np.savez_compressed(args.output/f'{i}-depth.npz', depth=depth, camera=camera, intrinsics=info.k)
            if i == 0:
                time.sleep(2)
        centres = [(np.array(f['visible_extent_min_m'])+f['visible_extent_max_m'])/2 for f in frames]
        drift = float(np.linalg.norm(centres[1]-centres[0]))
        result = {'success':all(f['inside_tray'] for f in frames) and drift < .005,
                  'method':'post-trial depth+segmentation extent against static tray interior; measured six-joint home',
                  'limitation':'same segmentation backend as agent; visible extent only, not dynamic object poses; depth source explicitly recorded',
                  'depth':transport.depth_metadata,
                  'segmentation_backend':config['segmentation_backend'],
                  'label':args.label,'drift_m':drift,'frames':frames}
        (args.output/'evaluation.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
        if not result['success']:
            raise SystemExit(2)
    except Exception as error:
        (args.output/'evaluation.json').write_text(json.dumps({'success':False,'error':str(error)},indent=2))
        raise
    finally:
        transport.close()


if __name__ == '__main__':
    main()
