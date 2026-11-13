"""Perception-driven tabletop cube manipulation through the ROS adapter."""
import time
import cv2
import numpy as np

from .perception import point_on_plane
from .planning import GraspPlanner


def color_top(image, color):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ranges = {'green': [(35, 90)], 'blue': [(95, 135)], 'red': [(0, 12), (170, 179)]}
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for lo, hi in ranges[color]:
        mask |= cv2.inRange(hsv, (lo, 80, 35), (hi, 255, 255))
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        raise ValueError(f'No visible {color} cube')
    candidates = [i for i in range(1, count) if 30 <= stats[i, cv2.CC_STAT_AREA] <= 5000]
    if not candidates:
        raise ValueError(f'No compact {color} cube region')
    index = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    region = components == index
    if region.sum() < 30:
        raise ValueError(f'{color} cube too small or occluded')
    # The top face of these matte calibration cubes is the brightly lit face.
    # Keep its interior; side-face pixels are at a different metric height.
    value = hsv[:, :, 2]
    top = region & (value >= np.percentile(value[region], 75))
    yy, xx = np.where(top)
    return np.array([xx.mean(), yy.mean()]), region


def tray_floor_pixel(image, seed):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 0, 220), (179, 35, 255))
    yy, xx = np.indices(mask.shape)
    mask[(xx-seed[0])**2+(yy-seed[1])**2 > 70**2] = 0
    count, components, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates = [i for i in range(1, count) if 150 < stats[i, cv2.CC_STAT_AREA] < 15000]
    if not candidates:
        raise ValueError('Tray interior is not visible enough for placement')
    index = min(candidates, key=lambda i: np.linalg.norm(centers[i]-seed))
    return centers[index]


class TabletopManipulation:
    def __init__(self, robot):
        self.robot = robot
        self.transport = robot.transport
        self.planner = GraspPlanner()
        self.events = []

    def phase(self, name):
        print('PHASE:', name, flush=True)
        if getattr(self.robot, 'video', None):
            self.robot.video.phase = name
        self.events.append({'phase': name, 'time': time.time()})

    def observe_return_obstacles(self):
        """Include visible calibration cubes in standalone home paths."""
        if not self.robot.config.get('calibrated_overview'):
            raise RuntimeError('Calibrated overview is required for planned return')
        image = self.transport.snapshot()['overview']
        info, camera = self.transport.camera_geometry('overview')
        objects = {}
        for color in ('red', 'green', 'blue'):
            try:
                uv, _ = color_top(image, color)
            except ValueError:
                continue
            objects[color] = point_on_plane(uv, info.k, info.d, camera,
                                            self.planner.floor_z+.03)-[0, 0, .015]
        self.planner.set_objects(objects, 'green')
        self.planner.target_geom = None

    def perceive(self, color):
        from maestro import vlm
        self.phase('VLM object localization')
        image = self.transport.snapshot()['overview']
        info, camera = self.transport.camera_geometry('overview')
        # Ask the actual VLM to identify the requested cube and destination.
        detections = vlm.point([image], [color+' cube', 'empty tray interior'])
        self.events.append({'vlm_detections': detections})
        print('VLM DETECTIONS:', detections, flush=True)
        by_label = {d['label']: d for d in detections}
        if color+' cube' not in by_label or 'empty tray interior' not in by_label:
            raise ValueError(f'VLM could not identify target and tray: {detections}')
        h, w = image.shape[:2]
        pixel, mask = color_top(image, color)
        y, x = by_label[color+' cube']['point']
        pointed = np.array([x*(w-1)/1000, y*(h-1)/1000])
        if np.linalg.norm(pixel-pointed) > 50:
            raise ValueError('VLM target disagrees with color segmentation')
        objects = {}
        for name in ('red', 'green', 'blue'):
            try:
                uv, _ = color_top(image, name)
                top = point_on_plane(uv, info.k, info.d, camera, self.planner.floor_z+.03)
                objects[name] = top-[0, 0, .015]
            except ValueError:
                if name == color:
                    raise
        y, x = by_label['empty tray interior']['point']
        tray_pixel = tray_floor_pixel(image, np.array([x*(w-1)/1000, y*(h-1)/1000]))
        destination = point_on_plane(tray_pixel, info.k, info.d,
                                     camera, self.planner.floor_z+.004)
        self.planner.set_objects(objects, color)
        result = {'objects': {k:v.tolist() for k,v in objects.items()},
                  'destination': destination.tolist(), 'vlm_detections': detections,
                  'target_top_pixel': pixel.tolist()}
        print('PERCEPTION:', result, flush=True)
        self.events.append({'perception': result})
        return objects[color], destination

    def execute(self, goal, phase, contact=False, minimum_duration=1.5, settle=1.5, require_hold=True):
        self.phase(phase)
        start = self.transport.snapshot()['joints']
        carrying = bool(self.planner.model.geom_contype[self.planner.carried_geom])
        if carrying:
            # Measured jaw position includes the object's thickness. Starting a
            # new interpolation from that value would momentarily release grip
            # pressure. Preserve the closing command throughout every transit.
            start[5] = goal[5]
        path = self.planner.path(start, goal)
        for target in path[1:]:
            start = self.transport.snapshot()['joints']
            if carrying:
                start[5] = target[5] = goal[5]
            duration = max(minimum_duration, 1.5*float(np.max(np.abs(target-start)))/.35)
            count = int(np.ceil(duration*30))
            for step in range(1, count+1):
                t = step/count
                desired = start+(target-start)*(3*t*t-2*t*t*t)
                command = self.planner.gravity_command(desired)
                self.robot.kinematics.validate(command)
                self.transport.publish(command)
                time.sleep(duration/count)
            # Hold the absolute position command; real physics determines contact.
            deadline = time.monotonic()+settle
            while time.monotonic() < deadline:
                self.transport.publish(self.planner.gravity_command(target))
                time.sleep(1/30)
            actual = self.transport.snapshot()['joints']
            error = np.abs(actual-target)
            if np.max(error[:5]) > .06 or (not contact and error[5] > .08):
                raise RuntimeError(f'Tracking failure during {phase}: {error.tolist()}')
            if require_hold and contact and self.planner.model.geom_contype[self.planner.carried_geom] and actual[5] < goal[5]+.005:
                raise RuntimeError('Gripper closed without expected object resistance; possible dropped grasp')
        self.events.append({'motion': phase, 'target': goal.tolist(),
                            'measured': self.transport.snapshot()['joints'].tolist(),
                            'collision_checks': self.planner.check_count})
        if getattr(self.robot, 'directory', None):
            from PIL import Image
            Image.fromarray(self.transport.snapshot()['overview']).save(
                self.robot.directory / f'phase-{len(self.events):03d}.png')

    def pick_place(self, color='green'):
        if color not in ('red', 'green', 'blue'):
            raise ValueError('This calibrated tabletop tool supports red, green and blue 30 mm cubes')
        source, dest = self.perceive(color)
        p = self.planner
        q = self.transport.snapshot()['joints']
        q[5] = .65
        self.execute(q, 'Open gripper')
        def move(xyz, opening, name):
            seed = self.transport.snapshot()['joints']
            self.execute(p.ik(xyz, seed, opening), name, contact=opening < .3)
        move([*source[:2], p.floor_z+.06], .65, 'Approach cube vertically')
        # Cartesian descent prevents the bowed hand path of a single joint-space
        # interpolation from pushing the cube before the fingers surround it.
        for height in np.linspace(.06, .014, 10)[1:]:
            seed = self.transport.snapshot()['joints']
            goal = p.ik([*source[:2], p.floor_z+height], seed, .65)
            self.execute(goal, 'Vertical descent', minimum_duration=.25, settle=.1)
        q = self.transport.snapshot()['joints']; q[5] = .10
        self.execute(q, 'Close gripper on cube', contact=True)
        p.set_carrying(True)
        goal = p.ik([*source[:2],p.floor_z+.08],self.transport.snapshot()['joints'],.10)
        self.execute(goal,'Lift cube',contact=True,settle=.2)
        # Use a continuous checked transit, high enough for the cube's lower
        # edge to clear the tray rim. Avoid repeated stop/start accelerations.
        goal = p.ik([*dest[:2],p.floor_z+.08],self.transport.snapshot()['joints'],.10)
        self.execute(goal,'Carry cube above tray',contact=True,settle=.1)
        goal = p.ik([*dest[:2],p.floor_z+.042],self.transport.snapshot()['joints'],.10)
        # Support at the destination can unload the jaws. Placement is checked
        # from the camera after release; transit still requires grasp resistance.
        self.execute(goal,'Lower cube into tray',contact=True,settle=.1,require_hold=False)
        p.set_carrying(False)
        p.model.geom_contype[p.target_geom] = p.model.geom_conaffinity[p.target_geom] = 0
        q = self.transport.snapshot()['joints']; q[5] = .65
        self.execute(q, 'Release cube')
        move([*dest[:2], p.floor_z+.08], .65, 'Retreat from tray')
        self.phase('Verify placement from camera')
        image = self.transport.snapshot()['overview']
        info, camera = self.transport.camera_geometry('overview')
        uv, _ = color_top(image, color)
        final = point_on_plane(uv, info.k, info.d, camera, p.floor_z+.034)
        error = float(np.linalg.norm(final[:2]-dest[:2]))
        if error >= .025:
            raise RuntimeError(f'Cube not observed in tray: offset {error:.3f}m')
        # The released cube is now an obstacle at its camera-observed location.
        # Do not retain its old source position or allow finger contact on return.
        body = p.model.body('observed_body_'+color).id
        p.data.mocap_pos[p.model.body_mocapid[body]] = final-[0, 0, .015]
        p.model.geom_contype[p.target_geom] = p.model.geom_conaffinity[p.target_geom] = 1
        p.target_geom = None
        self.execute(np.asarray(self.robot.config['home_rad'], dtype=float), 'Return to initial pose')
        home = self.robot.verify_home()
        self.phase('Verify placement after return to initial pose')
        uv, _ = color_top(self.transport.snapshot()['overview'], color)
        final = point_on_plane(uv, info.k, info.d, camera, p.floor_z+.034)
        error = float(np.linalg.norm(final[:2]-dest[:2]))
        result = {'source': source.tolist(), 'destination': dest.tolist(),
                  'observed_final_top': final.tolist(), 'placement_xy_error_m': error,
                  'home': home, 'success': error < .025, 'events': self.events}
        print('PLACEMENT:', result, flush=True)
        if not result['success']:
            raise RuntimeError(f'Cube not observed in tray: offset {error:.3f}m')
        return result
