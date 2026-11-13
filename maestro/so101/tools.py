"""SO101 embodiment tools for the shared Maestro agent.

There is deliberately no task-completing macro. The agent composes perception,
gripper, grasp, Cartesian motion, geometry, visualization and return itself.
"""
import functools
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from maestro import vlm
from maestro.geometry import GeometryTools, world_to_image
from maestro.perception_utils import draw_segmentation, segment
from maestro.scan_cache import SceneScanCache

from .api import SO101RobotApi
from .grasps import geometry_candidates
from .kinematics import JOINTS, vector
from .manipulation import TabletopManipulation
from .planning import GraspPlanner
from .rgbd import (
    lift_pixel,
    object_from_mask,
    objects_above_table,
    objects_from_masks,
    point_cloud,
    select_pointed_mask,
)


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def tool(method):
    @functools.wraps(method)
    def invoke(self, *args, **kwargs):
        start = time.time()
        self._event({'event': 'tool_start', 'tool': method.__name__, 'args': args, 'kwargs': kwargs})
        if self.video:
            self.video.phase = method.__name__
        try:
            result = method(self, *args, **kwargs)
        except Exception as error:
            self._event({'event': 'tool_error', 'tool': method.__name__, 'error': str(error)})
            raise
        self._event({'event': 'tool_end', 'tool': method.__name__, 'result': result,
                     'elapsed_s': time.time()-start})
        print(method.__name__ + ': ' + json.dumps(result, default=serial), flush=True)
        return result
    return invoke


class ImageEditor:
    """Buffered overlays projected onto fresh RGB images, as in ImageEditApi."""
    def __init__(self, robot):
        self.robot = robot
        self.actions = []

    def draw_point(self, position):
        self.draw_point_with_label(position, 'point')

    def draw_point_with_label(self, position, label):
        self.actions.append(('point', vector(position, 3, 'point'), str(label)))

    def draw_pose(self, position, orientation):
        self.actions.append(('pose', vector(position, 3, 'position'), vector(orientation, 3, 'orientation')))

    def feedback(self):
        images = self.robot.transport.snapshot()
        panels = []
        for name in ('overview', 'wrist'):
            canvas = images[name].copy()
            info, camera = self.robot.transport.camera_geometry(name)
            k = np.asarray(info.k).reshape(3,3)
            def project(p, k=k, camera=camera):
                pixel, _ = world_to_image(p, k, camera)
                return None if pixel is None else tuple(np.rint(pixel).astype(int))
            for kind, position, value in self.actions:
                xy = project(position)
                if xy is None:
                    continue
                if kind == 'point':
                    cv2.circle(canvas, xy, 5, (255,255,0), 2)
                    cv2.putText(canvas, value[:35], (xy[0]+6,xy[1]-6), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,0), 1)
                else:
                    rot = Rotation.from_euler('xyz',value).as_matrix()
                    for axis, color in enumerate(((255,0,0),(0,255,0),(0,100,255))):
                        end = project(position+.03*rot[:,axis])
                        if end is not None:
                            cv2.line(canvas,xy,end,color,2)
            cv2.putText(canvas,name,(10,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            panels.append(canvas)
        self.actions.clear()
        return np.concatenate(panels, axis=1)


class RobotTools(GeometryTools):
    def __init__(self, transport, config, directory, video=None):
        self.transport, self.config = transport, config
        self.directory, self.video = Path(directory), video
        self.low_level = SO101RobotApi(transport, config)
        self.kinematics = self.low_level.kinematics
        self.planner = GraspPlanner()
        # Reuse the checked trajectory executor, never its pick_place macro.
        self.motion = TabletopManipulation.__new__(TabletopManipulation)
        self.motion.robot, self.motion.transport = self, transport
        self.motion.planner, self.motion.events = self.planner, []
        self.image_editor = ImageEditor(self)
        self.objects, self.scan = [], None
        self.point_cache = {}
        self.scan_cache = SceneScanCache(str(self.directory/'scan-cache'))
        self.selected = None
        self.holding = False
        self.grip_command = config['gripper_closed_rad']
        self.instruction = ''
        self.replan_hints = [
            'All joint positions, including gripper, are absolute radians. Inspect grasp_detected and the images.',
            'Use locate_objects for relevant objects AND destinations before grasping. Cache measured positions.',
            'After a tool error, inspect the current images and state, then change your code. Do not repeat failed coordinates blindly.',
            'A successful grasp remains between the fingers during lift. A gripper at its closing target is empty.',
            'SO101 has five arm axes: some orientations/positions are unreachable. Use small vertical approaches and lifts.',
            'Use image_editor.draw_point/draw_pose to check geometry. Perception returns visible surface points, not object centres.',
            'After releasing, retreat vertically before reset. Complete only when the instruction AND return to home are verified.',
            'Do not call unavailable task macros. Assemble the steps from the documented tools.'
        ]

    def _event(self, event):
        with (self.directory/'tools.jsonl').open('a') as f:
            f.write(json.dumps({'time':time.time(), **event},default=serial)+'\n')

    def _log_timestamp(self, kind, detail):
        self._event({'event':kind,'detail':detail})

    def _sync_obstacles(self):
        p = self.planner
        for name in p.object_names:
            geom = p.model.geom('observed_'+name).id
            p.model.geom_contype[geom] = p.model.geom_conaffinity[geom] = 0
        for i, obj in enumerate(self.objects):
            name = f'slot{i}'
            geom = p.model.geom('observed_'+name).id
            body = p.model.body('observed_body_'+name).id
            p.data.mocap_pos[p.model.body_mocapid[body]] = obj['center']
            p.model.geom_size[geom] = obj['half_size']
            p.model.geom_rbound[geom] = np.linalg.norm(obj['half_size'])
            p.model.geom_aabb[geom,3:] = obj['half_size']
            p.model.geom_contype[geom] = p.model.geom_conaffinity[geom] = int(not(self.holding and self.selected == i))
        p.target_geom = None if self.selected is None else p.model.geom('observed_slot'+str(self.selected)).id

    @tool
    def scan_workspace(self):
        if self.holding:
            raise RuntimeError('Release held object before refreshing the scene map')
        rgb, depth, _stamp = self.transport.rgbd('overview')
        info, camera = self.transport.camera_geometry('overview')
        self.scan = {'rgb':rgb,'depth':depth,'info':info,'camera':camera,'time':time.time()}
        self.point_cache.clear()
        self.scan_cache.set([{'left_image':rgb,'depth':depth,
                              'intrinsics':np.asarray(info.k).reshape(3,3),'cam_pose':camera,
                              'target_joints':self.transport.snapshot()['joints'],
                              'timestamp':time.time(),'source':self.transport.depth_metadata['source']}])
        self.scan_cache.save_to_disk()
        if self.config.get('segmentation_backend') in ('groundedsam', 'sam3'):
            detections = segment(rgb, self.config.get('workspace_segmentation_prompt',
                                                      'colored object. block. cylinder.'),
                                 service=self.config['segmentation_backend'])
            scene_name = f'workspace-segmentation-{time.time_ns()}'
            np.savez_compressed(self.directory/(scene_name+'.npz'), **detections)
            Image.fromarray(draw_segmentation(rgb,detections)).save(self.directory/(scene_name+'.png'))
            self.objects = objects_from_masks(detections,point_cloud(depth,info.k,camera),self.planner.floor_z)
            self._event({'event':'workspace_segmentation','archive':scene_name+'.npz',
                         'backend':str(detections.get('backend', 'GroundingDINO+SAM')),
                         'accepted_indices':[o['detection_index'] for o in self.objects]})
        else:
            self.objects = objects_above_table(depth,info.k,camera,self.planner.floor_z)
        if len(self.objects) > 16:
            raise RuntimeError('More objects than collision-map capacity')
        self.selected = None
        self._sync_obstacles()
        name = f'scan-{time.time_ns()}'
        Image.fromarray(rgb).save(self.directory/(name+'.png'))
        np.savez_compressed(self.directory/(name+'.npz'),depth=depth,intrinsics=info.k,camera_pose=camera)
        (self.directory/(name+'-depth.json')).write_text(json.dumps(self.transport.depth_metadata,indent=2))
        return {'views':1,**self.transport.depth_metadata,'objects':len(self.objects),
                'supports':'separated compact tabletop objects; not touching assemblies'}

    @tool
    def refresh_scan(self):
        self.scan_cache.invalidate()
        return self.scan_workspace()

    @tool
    def locate_objects(self, labels):
        if self.scan is None:
            self.scan_workspace()
        s=self.scan
        detections=vlm.point([s['rgb']],labels)
        self._event({'event':'pointing_response','requested':labels,'detections':detections})
        h,w=s['depth'].shape
        result=[]
        for d in detections:
            if d['label'] not in labels or d['pose_index'] != 1:
                continue
            y,x=d['point']; pixel=[x*(w-1)/1000,y*(h-1)/1000]
            xyz=lift_pixel(pixel,s['depth'],s['info'].k,s['camera'])
            self.point_cache[d['label']] = pixel
            self.image_editor.draw_point_with_label(xyz,d['label'])
            result.append({'label':d['label'],'2d_position':pixel,'3d_position':xyz.tolist()})
        return result

    def _object_for(self, label):
        if self.config.get('segmentation_backend') in ('groundedsam', 'sam3'):
            obj, _detection = self._segmented_object(label)
            if self.objects:
                index = min(range(len(self.objects)), key=lambda i: np.linalg.norm(self.objects[i]['center']-obj['center']))
            else:
                index = 0
            if not self.objects or np.linalg.norm(self.objects[index]['center']-obj['center']) > .035:
                if len(self.objects) >= 16:
                    raise RuntimeError('Collision-map capacity exceeded')
                index = len(self.objects)
                self.objects.append(obj)
            else:
                self.objects[index] = obj
            self._sync_obstacles()
            return index, obj
        detections=self.locate_objects([label])
        if not detections or not self.objects:
            raise ValueError(f'No observed object for {label}')
        pixel=np.asarray(detections[0]['2d_position'])
        index=min(range(len(self.objects)),key=lambda i:np.linalg.norm(pixel-self.objects[i]['pixel']))
        if np.linalg.norm(pixel-self.objects[index]['pixel']) > 35:
            raise ValueError('VLM point does not match a separated RGB-D object')
        return index,self.objects[index]

    def _segmented_object(self, label):
        if self.scan is None:
            self.scan_workspace()
        s = self.scan
        detections = segment(s['rgb'], label, service=self.config['segmentation_backend'])
        name = f'segmentation-{time.time_ns()}'
        np.savez_compressed(self.directory/(name+'.npz'), **detections)
        Image.fromarray(draw_segmentation(s['rgb'], detections)).save(self.directory/(name+'.png'))
        if len(detections['scores']) == 0:
            raise ValueError(f'Segmentation did not detect {label}')
        if label not in self.point_cache:
            self.locate_objects([label])
        if label not in self.point_cache:
            raise ValueError('VLM did not point to the requested object; cannot disambiguate segmentation')
        index = select_pointed_mask(detections,self.point_cache[label])
        mask = detections['masks'][index].astype(bool)
        cloud = point_cloud(s['depth'], s['info'].k, s['camera'])
        obj = object_from_mask(mask, cloud, self.planner.floor_z)
        self._event({'event':'segmentation','backend':str(detections.get('backend', 'GroundingDINO+SAM')),
                     'requested':label,'score':float(detections['scores'][index]),
                     'box':detections['boxes'][index], 'archive':name+'.npz',
                     'selected_index':index,'point_pixel':self.point_cache[label]})
        return obj, detections

    @tool
    def locate_object_by_segmentation(self, object_name, z_offset=0):
        _index,obj=self._object_for(object_name)
        yy,xx=np.where(obj['mask'])
        position=obj['top'].copy();position[2]+=float(z_offset)
        self.image_editor.draw_point_with_label(position,object_name)
        return {'3d_position':position.tolist(),'box':[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())],
                'backend':self.config.get('segmentation_backend','rgbd_components')}

    @tool
    def locate_object_part(self, object_name, part_description):
        if self.config.get('segmentation_backend') in ('groundedsam', 'sam3'):
            obj, _ = self._segmented_object(object_name)
            yy,xx = np.where(obj['mask'])
            chosen = np.linspace(0,len(xx)-1,min(24,len(xx)),dtype=int)
            pixels = np.stack([xx[chosen],yy[chosen]],axis=1)
            image = self.scan['rgb'].copy()
            for i,xy in enumerate(pixels):
                cv2.circle(image,tuple(xy),3,(255,255,0),-1)
                cv2.putText(image,str(i),(int(xy[0])+3,int(xy[1])-3),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,0),1)
            selections = vlm.select_points(Image.fromarray(image),pixels.tolist(),part_description)
            result=[]
            for selected in selections:
                i=selected['index']
                if not 0 <= i < len(pixels):
                    raise ValueError('Part selector returned an invalid point index')
                xyz=lift_pixel(pixels[i],self.scan['depth'],self.scan['info'].k,self.scan['camera'])
                result.append({'label':selected['label'],'index':i,'3d_position':xyz.tolist()})
                self.image_editor.draw_point_with_label(xyz,selected['label'])
            return result
        return self.locate_objects([part_description+' of '+object_name])

    @tool
    def propose_grasps(self, object_name):
        index,obj=self._object_for(object_name)
        self.selected=index;self._sync_obstacles()
        seed=self.transport.snapshot()['joints']
        # Geometry candidates for this five-axis gripper, not Franka GraspGen weights.
        candidates=[]
        for position,orientation in geometry_candidates(obj,self.planner.floor_z):
            try:
                joints=self.planner.ik(position,seed,self.config['gripper_open_rad'],orientation)
                self.planner.ik(np.array(position)+[0,0,.05],joints,self.config['gripper_open_rad'],orientation)
                candidates.append((position,orientation))
                if len(candidates) == 6:
                    break
            except ValueError:
                continue
        if not candidates:
            raise ValueError('No reachable grasp of the observed object; change approach or target')
        self._event({'event':'grasp_backend','backend':'RGBD_geometry_SO101','object':object_name,
                     'half_size':obj['half_size'],'center':obj['center']})
        self.image_editor.draw_pose(*candidates[0])
        return candidates

    @tool
    def get_robot_state(self):
        q=self.transport.snapshot(images=False)['joints']; p=self.planner
        position=p.pose(q).copy()
        ori=Rotation.from_matrix(p.data.body(p.hand).xmat.reshape(3,3) @ np.diag([1,-1,-1])).as_euler('xyz')
        detected=bool(self.holding and q[5] > self.grip_command+.005)
        return {'robot_state':{'position':position.tolist(),'orientation':ori.tolist(),
                              'gripper_position_rad':float(q[5]),'grasp_detected':detected},
                'joint_positions_rad':dict(zip(JOINTS,q.tolist())), 'home_rad':self.config['home_rad']}

    @tool
    def move_to_pose(self, position, orientation=None):
        goal=vector(position,3,'position')
        if orientation is not None:
            orientation=vector(orientation,3,'orientation')
        p=self.planner; q=self.transport.snapshot()['joints']; start=p.pose(q).copy()
        if not self.holding and self.objects:
            nearest=min(range(len(self.objects)),key=lambda i:np.linalg.norm(self.objects[i]['center']-goal))
            if np.linalg.norm(self.objects[nearest]['center']-goal) < .03:
                self.selected=nearest;self._sync_obstacles()
        opening=self.grip_command if self.holding else q[5]
        # Near-vertical motion is sampled in Cartesian space to avoid a bowed descent.
        count=max(1,int(np.ceil(abs(goal[2]-start[2])/.008))) if np.linalg.norm(goal[:2]-start[:2]) < .008 else 1
        for xyz in np.linspace(start,goal,count+1)[1:]:
            q=self.transport.snapshot()['joints']
            joints=p.ik(xyz,q,opening,orientation)
            support=bool(self.holding and self.selected is not None and xyz[2] < p.floor_z+self.objects[self.selected]['half_size'][2]+.012)
            self.motion.execute(joints,'move_to_pose',contact=self.holding,require_hold=not support,
                                minimum_duration=.3 if count>1 else 1.5,settle=.15 if count>1 else .7)
        actual=p.pose(self.transport.snapshot()['joints']).copy()
        if np.linalg.norm(actual-goal) > .008:
            raise RuntimeError(f'Cartesian tracking error: requested {goal.tolist()}, measured {actual.tolist()}')
        if not self.holding and self.selected is not None and np.linalg.norm(actual-self.objects[self.selected]['center']) > .065:
            self.selected=None;self._sync_obstacles()
        return {'success':True,'measured_position':actual.tolist(),'position_error_m':float(np.linalg.norm(actual-goal))}

    @tool
    def open_gripper(self):
        q=self.transport.snapshot()['joints']
        was_holding=self.holding
        old_center=None if self.selected is None else self.objects[self.selected]['center'].copy()
        if self.holding and self.selected is not None:
            self.objects[self.selected]['center']=self.planner.pose(q).copy()
        self.holding=False
        self.planner.set_carrying(False)
        self._sync_obstacles()
        q[5]=self.config['gripper_open_rad']
        try:
            self.motion.execute(q,'open_gripper',settle=.5)
        except Exception:
            # Planning can reject opening near a tray wall before any release.
            # Preserve the carried state and pressure target for recovery.
            self.holding=was_holding
            if old_center is not None:
                self.objects[self.selected]['center']=old_center
            self.planner.set_carrying(was_holding)
            self._sync_obstacles()
            raise
        return {'gripper_position_rad':float(self.transport.snapshot()['joints'][5])}

    @tool
    def close_gripper(self):
        q=self.transport.snapshot()['joints']
        lower=self.config['gripper_closed_rad']
        actual=q.copy()
        # Stop at observed resistance instead of over-closing every object to
        # a cube-specific angle. The maintained target remains absolute radians.
        openings=np.linspace(q[5],lower,max(2,int(np.ceil(abs(q[5]-lower)/.03))+1))[1:]
        for opening in openings:
            q=self.transport.snapshot()['joints'];q[5]=opening
            self.grip_command=float(opening)
            self.motion.execute(q,'close_gripper',contact=True,require_hold=False,
                                minimum_duration=.2,settle=.4)
            actual=self.transport.snapshot()['joints']
            if actual[5] > self.grip_command+.015:
                break
        self.holding=bool(actual[5] > self.grip_command+.005)
        if not self.holding:
            raise RuntimeError('Empty grasp: measured jaws reached closing command; inspect image and replan')
        if self.selected is None:
            raise RuntimeError('Contact detected without a perceived object; cannot plan carried volume')
        p=self.planner
        p.pose(actual)
        rotation=p.data.body(p.hand).xmat.reshape(3,3)
        p.model.geom_size[p.carried_geom]=np.abs(rotation.T) @ self.objects[self.selected]['half_size']
        p.model.geom_rbound[p.carried_geom]=np.linalg.norm(p.model.geom_size[p.carried_geom])
        p.model.geom_aabb[p.carried_geom,3:]=p.model.geom_size[p.carried_geom]
        p.set_carrying(True)
        self._sync_obstacles()
        return {'grasp_detected':True,'gripper_position_rad':float(actual[5]),
                'maintained_command_rad':self.grip_command}

    @tool
    def execute_grasp(self, grasps):
        if not grasps:
            raise ValueError('No grasp candidates')
        position,orientation=grasps[0]
        position=vector(position,3,'grasp position')
        self.open_gripper()
        self.move_to_pose(position+[0,0,.05],orientation)
        self.move_to_pose(position,orientation)
        self.close_gripper()
        # Verify physical support before handing control back to a slow VLM.
        # This is a grasp tool, never transport to a destination or task completion.
        self.move_to_pose(position+[0,0,.025],orientation)
        time.sleep(.5)
        if not self.get_robot_state()['robot_state']['grasp_detected']:
            raise RuntimeError('Object was not retained during the grasp verification lift')
        return {'grasp_detected':True,'test_lift_m':.025,
                'measured_position':self.planner.pose(self.transport.snapshot()['joints']).tolist()}

    @tool
    def reset(self):
        if self.holding:
            raise RuntimeError('Release the held object and retreat before returning home')
        self.motion.execute(np.asarray(self.config['home_rad'],dtype=float),'Return to initial pose')
        return self.low_level.verify_home()

    def completion_guard(self):
        if self.holding:
            return 'An object is still held. Finish placement, release, retreat and reset.'
        try:
            self.low_level.verify_home()
        except RuntimeError as error:
            return str(error)+'. Call reset after retreat, then inspect the final images.'
        return None
