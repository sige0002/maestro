"""RGB ray/plane projection with explicit metric height, not guessed depth."""
import cv2
import numpy as np


def point_on_plane(pixel_xy, intrinsic, distortion, camera_to_base, z):
    if not np.isfinite(z):
        raise ValueError('Plane height must be finite base_link metres')
    k = np.asarray(intrinsic, dtype=float).reshape(3, 3)
    pose = np.asarray(camera_to_base, dtype=float).reshape(4, 4)
    if not np.all(np.isfinite(k)) or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Invalid camera intrinsics')
    pixel = np.asarray(pixel_xy, dtype=float).reshape(1, 1, 2)
    if not np.all(np.isfinite(pixel)) or not np.all(np.isfinite(pose)):
        raise ValueError('Invalid pixel or camera transform')
    normalized = cv2.undistortPoints(pixel, k, np.asarray(distortion, dtype=float)
                                     if len(distortion) else None)[0, 0]
    ray = pose[:3, :3] @ np.r_[normalized, 1]
    if abs(ray[2]) < 1e-8:
        raise ValueError('Camera ray is parallel to plane')
    distance = (z-pose[2, 3])/ray[2]
    if distance <= 0:
        raise ValueError('Plane is behind the camera')
    return pose[:3, 3] + distance * ray
