"""Embodiment-independent geometry helpers from Maestro's robot API."""
import numpy as np
from scipy.spatial.transform import Rotation


def euler_to_rmat(euler):
    return Rotation.from_euler('xyz', euler).as_matrix()


def rmat_to_euler(matrix):
    return Rotation.from_matrix(matrix).as_euler('xyz')


class GeometryTools:
    def measure_distance(self, position1, position2):
        self._log_timestamp("FUNCTION_CALL", f"measure_distance(position1={position1}, position2={position2})")
        position1 = np.array(position1)
        position2 = np.array(position2)
        return np.linalg.norm(position1 - position2)

    def measure_vector(self, position1, position2):
        self._log_timestamp("FUNCTION_CALL", f"measure_vector(position1={position1}, position2={position2})")
        position1 = np.array(position1)
        position2 = np.array(position2)
        return position2 - position1

    def euler_to_rotation_matrix(self, euler):
        self._log_timestamp("FUNCTION_CALL", f"euler_to_rotation_matrix(euler={euler})")
        euler = np.array(euler)
        return euler_to_rmat(euler)

    def rotation_matrix_to_euler(self, rotation_matrix):
        self._log_timestamp("FUNCTION_CALL", f"rotation_matrix_to_euler(rotation_matrix={rotation_matrix.tolist() if isinstance(rotation_matrix, np.ndarray) else rotation_matrix})")
        return rmat_to_euler(np.array(rotation_matrix))

    def measure_relative_rotation(self, vector1, vector2):
        self._log_timestamp("FUNCTION_CALL", f"measure_relative_rotation(vector1={vector1}, vector2={vector2})")
        vector1 = np.array(vector1) / np.linalg.norm(vector1)
        vector2 = np.array(vector2) / np.linalg.norm(vector2)
        v = np.cross(vector1, vector2)
        c = np.dot(vector1, vector2)
        if c > 0.9999:
            return np.array([0., 0., 0.])
        if c < -0.9999:
            axis = np.array([1., 0., 0.])
            if abs(vector1[0]) > 0.9:
                axis = np.array([0., 1., 0.])
            axis = axis - vector1 * np.dot(axis, vector1)
            axis = axis / np.linalg.norm(axis)
            return Rotation.from_rotvec(axis * np.pi).as_euler('xyz')
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s * s))
        euler = rmat_to_euler(R)
        return euler

    def rotate_by(self, current_orientation, rotation):
        self._log_timestamp("FUNCTION_CALL", f"rotate_by(current_orientation={current_orientation}, rotation={rotation})")
        current_orientation = np.array(current_orientation)
        rotation = np.array(rotation)
        rmat1 = euler_to_rmat(current_orientation)
        rmat2 = euler_to_rmat(rotation)
        rmat = rmat2 @ rmat1
        return rmat_to_euler(rmat)


def world_to_image( position, intrinsic, extrinsic):
    position = np.array(position, dtype=np.float32)
    position_homogeneous = np.append(position, 1.0)
    camera_position = np.linalg.inv(extrinsic) @ position_homogeneous
    if camera_position[2] <= 0:
        return None, None
    image_position = intrinsic @ camera_position[:3]
    image_position /= image_position[2]
    z_cam = camera_position[2]
    fx = intrinsic[0, 0]
    scale = fx / z_cam
    return image_position[:2], scale
