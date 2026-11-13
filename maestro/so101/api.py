"""Maestro tools implemented with SO101's canonical joint and image topics."""
import time

import numpy as np

from .kinematics import JOINTS, Kinematics, vector


class SO101RobotApi:
    def __init__(self, transport, config):
        self.transport = transport
        self.config = config
        self.kinematics = Kinematics(config['urdf'])
        self.command_log = []
        for key in ('max_age_s', 'command_hz', 'max_velocity_rad_s',
                    'joint_tolerance_rad', 'settle_timeout_s'):
            if key not in config or isinstance(config[key], bool) or not isinstance(config[key], (float, int)) \
                    or not np.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f'{key} must be positive and finite')
        self.kinematics.validate(config['home_rad'])
        for key in ('gripper_open_rad', 'gripper_closed_rad'):
            target = np.array(config['home_rad'], dtype=float)
            target[5] = config[key]
            self.kinematics.validate(target)

    def locate_objects(self, labels, plane_z_m):
        """Locate visible points on a known horizontal surface in base_link.

        Use the fixed overview camera. plane_z_m is the height of the pointed
        surface, including object height, not necessarily the tabletop height.
        """
        from maestro import vlm
        from .perception import point_on_plane
        if not self.config.get('calibrated_overview', False):
            raise RuntimeError('Overview camera calibration has not been declared valid')
        image = self.transport.snapshot()['overview']
        info, pose = self.transport.camera_geometry('overview')
        if info.distortion_model not in ('', 'plumb_bob', 'rational_polynomial'):
            raise ValueError(f'Unsupported distortion model: {info.distortion_model}')
        height, width = image.shape[:2]
        if (width, height) != (info.width, info.height):
            raise ValueError('CameraInfo size differs from RGB image')
        detections = vlm.point([image], labels)
        result = []
        for item in detections:
            y, x = item['point']
            if item['pose_index'] != 1 or not (0 <= x <= 1000 and 0 <= y <= 1000):
                raise ValueError('Invalid VLM image coordinates')
            position = point_on_plane([x*(width-1)/1000, y*(height-1)/1000],
                                      info.k, info.d, pose, plane_z_m)
            result.append({'label': item['label'], '3d_position': position.tolist(),
                           'plane_z_m': plane_z_m})
        print(result, flush=True)
        return result

    def get_robot_state(self):
        q = self.transport.snapshot(images=False)['joints']
        return {'joint_positions_rad': dict(zip(JOINTS, q.tolist())),
                'eef_pose_base_link': self.kinematics.forward(q).tolist()}

    def move_joints(self, positions, duration=2.0):
        target = self.kinematics.validate(positions)
        start = self.transport.snapshot()['joints']
        self.kinematics.validate(start)
        if not np.isfinite(duration) or not 0 < duration <= 30:
            raise ValueError('duration must be in (0, 30] seconds')
        duration = max(float(duration), 1.5 * float(np.max(np.abs(target-start))) /
                       self.config['max_velocity_rad_s'])
        if duration > 30:
            raise ValueError('Motion duration exceeds 30 seconds')
        ticks = max(1, int(np.ceil(duration*self.config['command_hz'])))
        began = time.monotonic()
        for tick in range(1, ticks+1):
            t = tick/ticks
            q = start + (target-start)*(3*t*t-2*t*t*t)
            self.transport.publish(q)
            # Discovery or scheduling delays must not cause a burst of queued
            # setpoints that exceeds the intended command rate.
            time.sleep(max(duration/ticks, began + duration*t-time.monotonic()))
        deadline = time.monotonic() + self.config['settle_timeout_s']
        while time.monotonic() < deadline:
            measured = self.transport.snapshot(images=False)['joints']
            error = float(np.max(np.abs(measured-target)))
            if error <= self.config['joint_tolerance_rad']:
                result = {'target_rad': target.tolist(), 'measured_rad': measured.tolist(),
                          'max_error_rad': error, 'duration_s': duration}
                self.command_log.append(result)
                print(result, flush=True)
                return result
            time.sleep(.05)
        raise RuntimeError(f'Joint target not reached: max error {error:.4f}rad')

    def move_to_pose(self, position, orientation=None, duration=2.0):
        from scipy.spatial.transform import Rotation
        position = vector(position, 3, 'position')
        commanded_position = position.copy()
        for attempt in range(4):
            seed = self.transport.snapshot(images=False)['joints']
            target = self.kinematics.inverse(commanded_position, seed, orientation)
            result = self.move_joints(target, duration)
            # Let the servo settle before evaluating residual gravity/tracking error.
            time.sleep(.5)
            measured = self.transport.snapshot(images=False)['joints']
            measured_pose = self.kinematics.forward(measured)
            residual = position - measured_pose[:3, 3]
            error = float(np.linalg.norm(residual))
            angle = 0.0
            if orientation is not None:
                desired = Rotation.from_euler('xyz', orientation).as_matrix()
                angle = float(np.linalg.norm(Rotation.from_matrix(
                    desired.T @ measured_pose[:3, :3]).as_rotvec()))
            if error <= .005 and angle <= .2:
                return {**result, 'measured_rad': measured.tolist(),
                        'position_error_m': error, 'corrections': attempt}
            # Bounded outer Cartesian feedback loop. Coordinates remain canonical;
            # every new absolute joint command still passes limits/speed checks.
            if error > .03 or angle > .2:
                break
            commanded_position += residual
        raise RuntimeError(f'Cartesian target not reached: position error {error:.4f}m, angle {angle:.3f}rad')

    def open_gripper(self):
        q = self.transport.snapshot(images=False)['joints']
        q[5] = self.config['gripper_open_rad']
        return self.move_joints(q)

    def close_gripper(self):
        q = self.transport.snapshot(images=False)['joints']
        q[5] = self.config['gripper_closed_rad']
        return self.move_joints(q)

    def reset(self):
        if self.config.get('sim_manipulation') and self.config.get('use_sim_time'):
            from .manipulation import TabletopManipulation
            tool = TabletopManipulation(self)
            tool.observe_return_obstacles()
            tool.execute(np.asarray(self.config['home_rad'], dtype=float), 'Return to initial pose')
            return self.verify_home()
        return self.move_joints(self.config['home_rad'], duration=4)

    def verify_home(self):
        target = np.asarray(self.config['home_rad'], dtype=float)
        actual = self.transport.snapshot(images=False)['joints']
        error = float(np.max(np.abs(actual-target)))
        if error > min(.06, self.config['joint_tolerance_rad']):
            raise RuntimeError(f'Initial pose not reached: max error {error:.4f}rad')
        return {'target_rad': target.tolist(), 'measured_rad': actual.tolist(),
                'max_error_rad': error, 'returned_home': True}

    def pick_place_cube(self, color='green'):
        """Perceive a 30 mm cube and tray, then execute collision-checked grasping."""
        if not self.config.get('sim_manipulation') or not self.config.get('use_sim_time'):
            raise RuntimeError('Calibrated MuJoCo tabletop manipulation must be explicitly enabled')
        from .manipulation import TabletopManipulation
        manipulation = TabletopManipulation(self)
        try:
            return manipulation.pick_place(color)
        finally:
            if getattr(self, 'directory', None):
                import json
                import time
                (self.directory / f'manipulation-events-{time.time_ns()}.json').write_text(
                    json.dumps(manipulation.events, indent=2))
