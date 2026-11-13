"""Canonical URDF coordinates, in metres and absolute radians."""
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')


def vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must contain {size} finite numbers')
    return result


class Kinematics:
    def __init__(self, urdf, tip='gripper_frame_link'):
        root = ET.parse(urdf).getroot()
        joints = {j.attrib['name']: j for j in root.findall('joint')}
        self.limits = np.array([[float(joints[n].find('limit').attrib[k])
                                for k in ('lower', 'upper')] for n in JOINTS])
        children = {j.find('child').attrib['link']: j for j in joints.values()}
        chain = []
        while tip != 'base_link':
            joint = children[tip]
            origin = joint.find('origin')
            pose = np.eye(4)
            if origin is not None:
                pose[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                pose[:3, :3] = Rotation.from_euler('xyz', np.fromstring(
                    origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            axis = joint.find('axis')
            axis = np.fromstring(axis.get('xyz', '1 0 0'), sep=' ') if axis is not None else np.zeros(3)
            name = joint.attrib['name']
            chain.append((JOINTS.index(name) if name in JOINTS else None, pose, axis))
            tip = joint.find('parent').attrib['link']
        self.chain = list(reversed(chain))

    def validate(self, q):
        q = vector(q, 6, 'joint positions')
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            raise ValueError('Joint target is outside URDF limits; target rejected without clipping')
        return q

    def forward(self, q):
        q = vector(q, 6, 'joint positions')
        pose = np.eye(4)
        for index, origin, axis in self.chain:
            pose = pose @ origin
            if index is not None:
                turn = np.eye(4)
                turn[:3, :3] = Rotation.from_rotvec(axis * q[index]).as_matrix()
                pose = pose @ turn
        return pose

    def inverse(self, position, seed, orientation=None):
        target = vector(position, 3, 'position')
        seed = self.validate(seed)
        desired = None if orientation is None else Rotation.from_euler(
            'xyz', vector(orientation, 3, 'orientation')).as_matrix()

        def residual(arm):
            pose = self.forward(np.r_[arm, seed[5]])
            error = (pose[:3, 3] - target) * 10
            if desired is not None:
                error = np.r_[error, Rotation.from_matrix(desired.T @ pose[:3, :3]).as_rotvec() * .3]
            return error

        result = least_squares(residual, seed[:5], bounds=(self.limits[:5, 0], self.limits[:5, 1]),
                               max_nfev=250, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        q = self.validate(np.r_[result.x, seed[5]])
        pose = self.forward(q)
        error = np.linalg.norm(pose[:3, 3] - target)
        angle = 0 if desired is None else np.linalg.norm(
            Rotation.from_matrix(desired.T @ pose[:3, :3]).as_rotvec())
        if not result.success or error > .005 or angle > .15:
            raise ValueError(f'Unreachable SO101 pose: position error={error:.4f}m, angle error={angle:.3f}rad')
        return q
