"""Shared Maestro scan cache, independent of DROID hardware imports."""
import os
import json
import time
from datetime import datetime
import numpy as np
import PIL.Image as Image
from maestro.camera_geometry import camera_matrix

class SceneScanCache:
    """Cache wrist-camera scene scans to avoid repeated rescans.

    Stores for each candidate wrist pose:
    - target_joints: absolute 7-DoF joint configuration used to capture images
    - left_image, right_image: RGB numpy arrays captured at that pose
    - timestamp: capture time
    The cache is invalidated by refresh_scan, or when a target cannot be found or verified in the cached views.
    """
    def __init__(self, base_dir: str):
        self.entries = []
        self.valid = False
        self.base_dir = base_dir

    def set(self, entries):
        self.entries = entries
        self.valid = True

    def get(self):
        return self.entries if self.valid else []

    def invalidate(self):
        self.entries = []
        self.valid = False

    def _ensure_dir(self):
        try:
            os.makedirs(self.base_dir, exist_ok=True)
        except Exception:
            pass

    def save_to_disk(self, robot_api=None):
        """Persist current cache entries to disk under base_dir and write meta.json.

        Args:
            robot_api: Optional RealFrankaRobotApi instance for dual-save and logging
        """
        if not self.valid or len(self.entries) == 0:
            return
        self._ensure_dir()

        # Generate timestamp for this save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # Clear existing files in base_dir (latest_data)
        try:
            for name in os.listdir(self.base_dir):
                try:
                    os.remove(os.path.join(self.base_dir, name))
                except Exception:
                    pass
        except Exception:
            pass

        meta = []
        for i, entry in enumerate(self.entries, start=1):
            # Save to latest_data (no timestamp)
            left_path = os.path.join(self.base_dir, f"pose_{i}_left.png")
            right_path = os.path.join(self.base_dir, f"pose_{i}_right.png")
            depth_path_npy = os.path.join(self.base_dir, f"pose_{i}_depth.npy")

            try:
                Image.fromarray(entry['left_image']).save(left_path)
                if entry.get('right_image') is not None:
                    Image.fromarray(entry['right_image']).save(right_path)
                # If depth is available, save it too
                if 'depth' in entry and isinstance(entry['depth'], np.ndarray):
                    try:
                        np.save(depth_path_npy, entry['depth'])
                    except Exception:
                        pass
                # If robot_api is available, also save timestamped versions via dual-save
                if robot_api:
                    robot_api._save_dual(entry['left_image'], f"scan_cache/pose_{i}_left", "png",
                                       timestamp, context="scan_workspace")
                    if entry.get('right_image') is not None:
                        robot_api._save_dual(entry['right_image'], f"scan_cache/pose_{i}_right", "png",
                                           timestamp, context="scan_workspace")
                    if 'depth' in entry and isinstance(entry['depth'], np.ndarray):
                        robot_api._save_dual(entry['depth'], f"scan_cache/pose_{i}_depth", "npy",
                                           timestamp, is_image=False, context="scan_workspace")
                    robot_api._log_timestamp("FILE_SAVE", f"scan_cache pose {i} images saved")
                    robot_api._log_timestamp("FILE_SAVE", f"scan_cache pose {i} depth saved")
            except Exception:
                # Skip if saving fails for this entry
                continue

            meta_entry = {
                'pose_index': i,
                'target_joints': np.array(entry['target_joints']).tolist(),
                'timestamp': float(entry.get('timestamp', time.time())),
                'left_image': os.path.basename(left_path),
                'right_image': os.path.basename(right_path) if entry.get('right_image') is not None else None,
                'source': entry.get('source', 'stereo'),
            }
            if os.path.exists(depth_path_npy):
                meta_entry['depth_file'] = os.path.basename(depth_path_npy)
            # Persist intrinsics and camera pose if present
            if 'intrinsics' in entry:
                try:
                    meta_entry['intrinsics'] = camera_matrix(entry['intrinsics']).tolist()
                except Exception:
                    pass
            if 'cam_pose' in entry:
                try:
                    meta_entry['cam_pose'] = np.array(entry['cam_pose']).tolist()
                except Exception:
                    pass
            meta.append(meta_entry)

        # Save metadata
        try:
            meta_path = os.path.join(self.base_dir, 'meta.json')
            with open(meta_path, 'w') as f:
                json.dump(meta, f)

            # Save timestamped metadata if robot_api available
            if robot_api:
                robot_api._save_dual(meta, "scan_cache/meta", "json",
                                   timestamp, is_image=False, context="scan_workspace")
                robot_api._log_timestamp("FILE_SAVE", f"scan_cache metadata saved with {len(meta)} poses")
        except Exception:
            pass

    def load_from_disk(self):
        """Load cached entries from disk if available."""
        self._ensure_dir()
        meta_path = os.path.join(self.base_dir, 'meta.json')
        if not os.path.exists(meta_path):
            return []
        try:
            with open(meta_path, 'r') as stream:
                meta = json.load(stream)
        except Exception:
            return []
        loaded = []
        for m in meta:
            left_path = os.path.join(self.base_dir, m.get('left_image', ''))
            right_path = os.path.join(self.base_dir, m['right_image']) if m.get('right_image') else None
            if not os.path.exists(left_path) or (right_path and not os.path.exists(right_path)):
                continue
            try:
                left_img = np.array(Image.open(left_path))
                right_img = np.array(Image.open(right_path)) if right_path else None
                depth_file = m.get('depth_file')
                loaded.append({
                    'target_joints': np.array(m.get('target_joints', []), dtype=float),
                    'left_image': left_img,
                    'right_image': right_img,
                    'timestamp': float(m.get('timestamp', time.time())),
                    'depth': np.load(os.path.join(self.base_dir, depth_file)) if depth_file else None,
                    'intrinsics': m.get('intrinsics'),
                    'cam_pose': m.get('cam_pose'),
                    'source': m.get('source', 'stereo'),
                })
            except Exception:
                continue
        return loaded

    def ensure_loaded_from_disk(self):
        if not self.valid:
            entries = self.load_from_disk()
            if len(entries) > 0:
                self.set(entries)
