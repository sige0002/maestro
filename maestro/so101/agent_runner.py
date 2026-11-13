"""SO101 lifecycle around Maestro's shared coding-agent harness."""
import argparse
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path

from PIL import Image
import yaml

from maestro import vlm
from maestro.agent import VLMCodeExtractor, plan_first_step, execute_plan_loop
from .runner import check_vlm_credentials
from .tools import RobotTools, serial
from .transport import RosTransport
from .recording import TrialVideo


class TrialLogger:
    def __init__(self, directory):
        self.log_dir=Path(directory)
        self.timestamp_log_file=self.log_dir/'timestamps.log'
        self.step=0

    def log_text(self,text):
        with (self.log_dir/'log.txt').open('a') as f:
            f.write(text+'\n')

    def save_image(self,image,step,j):
        image.save(self.log_dir/f'{self.step:02d}-feedback-{j}.png')


def observe(robot,directory,step):
    obs=robot.transport.snapshot()
    for name in robot.config['cameras']:
        Image.fromarray(obs[name]).save(directory/f'{step:02d}-{name}.png')
    state=robot.get_robot_state()
    (directory/f'{step:02d}-state.json').write_text(json.dumps(state,indent=2,default=serial))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/so101.yaml')
    parser.add_argument('-i','--instruction')
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--record',action='store_true')
    parser.add_argument('--max-steps',type=int,default=20)
    parser.add_argument('--output',default='runs/so101-agent')
    parser.add_argument('--history',type=Path,help='Prior trial log for Maestro in-context trial improvement')
    args=parser.parse_args()
    if bool(args.instruction)==args.check:
        parser.error('Choose --instruction or --check')
    if not 1<=args.max_steps<=40:
        parser.error('max-steps must be in 1..40')
    config=yaml.safe_load(Path(args.config).read_text())
    if not config.get('use_sim_time'):
        raise ValueError('This RGB-D tool backend is currently validated only for MuJoCo')
    directory=Path(args.output)/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory.mkdir(parents=True)
    sources=[Path(__file__),Path(__file__).with_name('tools.py'),Path(__file__).with_name('agent_prompt.txt'),
             Path(__file__).parents[1]/'agent.py',Path(__file__).parents[1]/'geometry.py']
    sources += [Path(__file__).with_name(n) for n in ('planning.py','transport.py','rgbd.py','grasps.py','manipulation.py','monocular_depth.py')]
    model_source=Path('/so101/scripts/models/model.py')
    if model_source.exists():
        sources.append(model_source)
    (directory/'source').mkdir()
    for source in sources:
        (directory/'source'/source.name).write_bytes(source.read_bytes())
    (directory/'manifest.json').write_text(json.dumps({'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                                                     'robot_config':config,'instruction':args.instruction,
                                                     'vlm':{'provider':vlm.PROVIDER,'model':vlm.SETTINGS.get('model'),
                                                            'max_tokens':vlm.SETTINGS.get('max_tokens')}},indent=2))
    transport=video=None
    try:
        if args.instruction:
            check_vlm_credentials()
        transport=RosTransport(config);transport.wait_ready()
        robot=RobotTools(transport,config,directory)
        if args.record:
            video=TrialVideo(transport,directory/'live.mp4');robot.video=video;video.start()
        logger=TrialLogger(directory)
        observe(robot,directory,0)
        scan_result=robot.scan_workspace()
        if args.check:
            result={'status':'observations_ready','state':robot.get_robot_state(), 'depth':transport.depth_metadata, 'scan':scan_result}
        else:
            robot.instruction=args.instruction
            chat=vlm.planner_chat()
            prompt=Path(__file__).with_name('agent_prompt.txt').read_text()
            history=args.history.read_text() if args.history else ''
            code=plan_first_step(chat,robot,robot.image_editor,logger,prompt,history)
            extractor=VLMCodeExtractor(chat,robot,robot.image_editor,logger)
            extractor.actor_code=code
            success=execute_plan_loop(extractor,args.max_steps,
                on_step=lambda step:observe(robot,directory,step),completion_guard=robot.completion_guard)
            result={'status':'model_reported_complete' if success else 'step_limit',
                    'instruction':args.instruction,'steps':logger.step,
                    'state':robot.get_robot_state(),'home_verified':robot.completion_guard() is None,
                    'harness':'maestro.agent.VLMCodeExtractor + execute_plan_loop',
                    'backends':{'depth':transport.depth_metadata,'perception':config.get('segmentation_backend','rgbd_components')+' + Maestro_VLM_pointing',
                                'grasp':'SO101_RGBD_geometry','motion':'SO101_IK_sampled_collision_RRT'}}
        (directory/'result.json').write_text(json.dumps(result,indent=2,default=serial))
        print(json.dumps({'run':str(directory),**result},indent=2,default=serial),flush=True)
        if result['status']=='step_limit':
            raise SystemExit(2)
    except KeyboardInterrupt:
        (directory/'result.json').write_text(json.dumps({'status':'interrupted','success':False},indent=2))
        raise
    except Exception as error:
        (directory/'result.json').write_text(json.dumps({'status':'failed','error':str(error)},indent=2))
        raise
    finally:
        if video:video.close()
        if transport:transport.close()


if __name__=='__main__':
    main()
