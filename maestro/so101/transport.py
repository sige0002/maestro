"""Direct ROS transport; no HTTP bridge or hardware driver in Maestro."""
import io
import threading
import time

import numpy as np
from PIL import Image
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, CompressedImage, JointState, Image as DepthImage
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, TransformListener

from .kinematics import JOINTS


class RosTransport:
    def __init__(self, config):
        self.config = config
        self.depth_backend=config.get('depth_backend','depth_anything')
        if self.depth_backend not in ('depth_anything','mujoco_sensor'):
            raise ValueError('Unknown depth_backend')
        self.depth_metadata={}
        self.latest = {}
        self.lock = threading.Lock()
        rclpy.init()
        self.node = rclpy.create_node('maestro_so101', cli_args=[
            '--ros-args', '-r', '/tf:=/so101/tf', '-r', '/tf_static:=/so101/tf_static',
            '-r', '/clock:=/so101/clock'], parameter_overrides=[
                Parameter('use_sim_time', value=config.get('use_sim_time', False))])
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        # Observation-only sessions must not become a second command source:
        # the SO101 controller rejects commands while multiple publishers exist.
        self.publisher = None
        self.subscriptions = [self.node.create_subscription(
            JointState, config['state_topic'], self._state, qos_profile_sensor_data)]
        for name, topic in config['cameras'].items():
            self.subscriptions.append(self.node.create_subscription(
                CompressedImage, topic, lambda msg, key=name: self._image(key, msg), qos_profile_sensor_data))
        for name, topic in config.get('camera_info_topics', {}).items():
            self.subscriptions.append(self.node.create_subscription(
                CameraInfo, topic, lambda msg, key=name: self._store(key + '_info', msg),
                qos_profile_sensor_data))
        depth_topics=config.get('depth_topics', {}) if self.depth_backend=='mujoco_sensor' else {}
        for name, topic in depth_topics.items():
            self.subscriptions.append(self.node.create_subscription(
                DepthImage, topic, lambda msg, key=name: self._depth(key, msg), qos_profile_sensor_data))
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def _store(self, key, value):
        with self.lock:
            self.latest[key] = (time.monotonic(), value)

    def _state(self, msg):
        if len(msg.name) != len(msg.position) or len(set(msg.name)) != len(msg.name):
            return
        values = dict(zip(msg.name, msg.position))
        if all(n in values and np.isfinite(values[n]) for n in JOINTS):
            self._store('joints', np.array([values[n] for n in JOINTS]))

    def _image(self, key, msg):
        try:
            image = Image.open(io.BytesIO(bytes(msg.data))).convert('RGB')
            pixels = np.array(image)
            with self.lock:
                now = time.monotonic()
                self.latest[key] = (now, pixels)
                self.latest[key + '_stamp'] = (now, msg.header.stamp)
        except (ValueError, OSError) as error:
            self.node.get_logger().warning(f'Bad image on {key}: {error}')

    def _depth(self, key, msg):
        if msg.encoding != '32FC1' or msg.step < msg.width * 4 or len(msg.data) != msg.step * msg.height:
            self.node.get_logger().warning('Rejecting malformed metric depth image')
            return
        dtype = '>f4' if msg.is_bigendian else '<f4'
        depth = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=bytes(msg.data),
                           strides=(msg.step, 4)).astype(np.float32)
        with self.lock:
            now = time.monotonic()
            self.latest[key + '_depth'] = (now, depth)
            self.latest[key + '_depth_stamp'] = (now, msg.header.stamp)

    def rgbd(self, name='overview', timeout=3):
        if self.depth_backend=='depth_anything':
            from .monocular_depth import infer_depth
            with self.lock:
                if name not in self.latest or time.monotonic()-self.latest[name][0] >= self.config['max_age_s']:
                    raise RuntimeError('Missing/stale RGB input for depth estimation')
                rgb=self.latest[name][1].copy()
                stamp=self.latest[name+'_stamp'][1]
            depth,self.depth_metadata=infer_depth(rgb,self.config.get('depth_anything',{}))
            return rgb,depth,stamp
        self.depth_metadata={'source':'MuJoCo_RGBD_sensor'}
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                keys = [name, name+'_depth', name+'_stamp', name+'_depth_stamp']
                if all(k in self.latest and time.monotonic()-self.latest[k][0] < self.config['max_age_s'] for k in keys):
                    rgb, depth, rgb_stamp, depth_stamp = [self.latest[k][1] for k in keys]
                    delta = abs(rgb_stamp.sec-depth_stamp.sec + (rgb_stamp.nanosec-depth_stamp.nanosec)*1e-9)
                    if delta < .001 and rgb.shape[:2] == depth.shape:
                        return rgb.copy(), depth.copy(), depth_stamp
            time.sleep(.01)
        raise RuntimeError(f'No fresh synchronized RGB-D frame for {name}')

    def snapshot(self, images=True):
        keys = ['joints', *self.config['cameras']] if images else ['joints']
        with self.lock:
            stale = [k for k in keys if k not in self.latest or
                     time.monotonic() - self.latest[k][0] > self.config['max_age_s']]
            if stale:
                raise RuntimeError(f'Missing/stale ROS observations: {stale}')
            return {k: self.latest[k][1].copy() for k in keys}

    def wait_ready(self, timeout=30):
        end = time.monotonic() + timeout
        while True:
            try:
                return self.snapshot()
            except RuntimeError:
                if time.monotonic() > end:
                    raise
                time.sleep(.1)

    def camera_geometry(self, name):
        with self.lock:
            if name + '_info' not in self.latest:
                raise RuntimeError(f'No CameraInfo for {name}; calibrated geometry is required')
            received, info = self.latest[name + '_info']
            if time.monotonic() - received > self.config['max_age_s']:
                raise RuntimeError(f'Stale CameraInfo for {name}')
        transform = self.tf_buffer.lookup_transform('base_link', info.header.frame_id, rclpy.time.Time())
        t, q = transform.transform.translation, transform.transform.rotation
        pose = np.eye(4)
        pose[:3, 3] = [t.x, t.y, t.z]
        pose[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        return info, pose

    def publish(self, positions):
        if self.publisher is None:
            if self.node.count_publishers(self.config['command_topic']) != 0:
                raise RuntimeError('Another command publisher is active; use SO101 launcher to stop it')
            self.publisher = self.node.create_publisher(JointState, self.config['command_topic'], 1)
            deadline = time.monotonic() + 3
            while self.publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
                time.sleep(.05)
        self.snapshot(images=False)
        if self.publisher.get_subscription_count() == 0:
            raise RuntimeError('No subscriber to joint_commands')
        if self.node.count_publishers(self.config['command_topic']) != 1:
            raise RuntimeError('Another command publisher is active; use SO101 launcher to stop it')
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(JOINTS)
        msg.position = [float(x) for x in positions]
        self.publisher.publish(msg)

    def close(self):
        self.executor.shutdown(timeout_sec=3)
        self.thread.join(timeout=3)
        self.node.destroy_node()
        rclpy.try_shutdown()
