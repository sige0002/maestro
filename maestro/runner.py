"""Main VLM pipeline for robot control.

This module provides the main execution loop for running robot trials with:
- Natural language instruction processing by the planner VLM (see maestro.vlm)
- Video recording with automatic saving on Ctrl+C
- Continuous trial execution with user prompts
"""

# Quiet third-party import noise for a clean console (our own output is unaffected).
import warnings as _warnings, logging as _logging
_warnings.filterwarnings("ignore", category=FutureWarning)
_warnings.filterwarnings("ignore", category=DeprecationWarning)
_warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
for _n in ("httpx", "openai", "google_genai", "OpenGL", "OpenGL.acceleratesupport", "OpenGL.arrays.arraydatatype"):
    _logging.getLogger(_n).setLevel(_logging.WARNING)
# gym prints an "unmaintained" banner straight to stderr on import (not a Python warning);
# import it once with stderr muted so the banner never reaches the user's console.
import os as _os, contextlib as _ctx
with open(_os.devnull, "w") as _dn, _ctx.redirect_stderr(_dn):
    try:
        import gym as _gym
        _gym.logger.set_level(40)  # ERROR
    except Exception:
        pass

import numpy as np
import argparse
from maestro.api import RealFrankaRobotApi, ImageEditApi
import PIL.Image as Image
import re

import os
from maestro.config import cfg, service_url, REPO_ROOT
import atexit
import signal
import json
import shutil
from datetime import datetime
from pathlib import Path
import time
import sys
import io
from copy import deepcopy

from droid.robot_env_threadsafe import RobotEnvThreadSafe as RobotEnv
from pydantic import BaseModel
from maestro import vlm
from maestro.agent import VLMCodeExtractor, plan_first_step, execute_plan_loop
from maestro.agent import FirstStepPlan as FirstStepPlan, Replan as Replan, format_floats as format_floats
import requests
from gevent import spawn as gevent_spawn, sleep as gevent_sleep
from gevent.event import Event as GeventEvent
from droid.misc.parameters import droid_version, hand_camera_id, robot_serial_number, robot_type

# DROID-format trajectory recording for data collection (DROID's own trajectory utilities).
from maestro.trajectory_recorder import TrajectoryWriter, save_calibration_info
from droid.camera_utils.info import camera_type_dict

    


def past_trial_logs(folder_path):
    trial_count = 0
    result_str = ""
    for root, dirs, files in sorted(os.walk(folder_path), key=lambda x: x[0]):
        if "log.txt" in files:
            trial_count += 1
            log_path = os.path.join(root, "log.txt")
            with open(log_path, "r", encoding="utf-8") as f:
                data = f.read().strip()
            result_str += f" Trial {trial_count}:\n{data}\n"

    return result_str
    
class Logger:
    def __init__(self, instruction, base_log_dir="logs_all_tasks"):
        postfix = re.sub(r'[^\w\s]', '', instruction).replace(' ', '_')[:50]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = Path(base_log_dir) / f'{timestamp}_{postfix}'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "log.txt"
        # Dedicated timestamp log for visualization alignment
        self.timestamp_log_file = self.log_dir / "timestamps.log"
        self.step = 0
        self.instruction = instruction
        # Initialize timestamps.log with trial metadata
        self._init_timestamp_log()

    def log_text(self, text):
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{text}\n")
        except Exception as e:
            with open(self.log_file, "a", encoding="utf-8", errors="ignore") as f:
                f.write("Error writing to log file: " + str(e) + "\n")

    def save_image(self, image, step, j):
        img_path = self.log_dir / f"step_{self.step}_{j}.png"
        image.save(img_path)
        self.log_text(f"Saved image for step {self.step}: {img_path}")
        
    def save_crop_image(self, image, object_name):
        img_path = self.log_dir / f"{object_name}_seg.png"
        image.save(img_path)
    
    def _init_timestamp_log(self):
        """Initialize the timestamp log with trial metadata."""
        try:
            with open(self.timestamp_log_file, "w", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                f.write(f"# Trial Visualization Log\n")
                f.write(f"# Started: {timestamp}\n")
                f.write(f"# Task: {self.instruction}\n")
                f.write(f"# Format: timestamp | event_type | details\n")
                f.write(f"# Event types: TRIAL_START, FUNCTION_CALL, FILE_SAVE, FILE_LOAD, PLANNER_RESPONSE, SUBTASK_START, SUBTASK_END, CODE_EXEC, SCAN_POSE, SCAN_CAPTURE, SEGMENTATION_START, SEGMENTATION_COMPLETE, SEGMENTATION_SUCCESS, SEGMENTATION_FAILED, SEGMENTATION_ERROR\n")
                f.write(f"{timestamp} | TRIAL_START | Task: {self.instruction}\n")
        except Exception as e:
            print(f"Failed to initialize timestamp log: {e}")


class DataRecorder:
    """Background recorder that logs a DROID-format trajectory during trials (data collection)."""

    def __init__(self, env, robot_api, logger, controller_name="VLM_Maestro", post_process=True):
        self.env = env
        self.robot_api = robot_api
        self.logger = logger
        self.controller_name = controller_name
        self.post_process = post_process
        self.control_hz = getattr(env, "control_hz", 15)
        self._stop_event = GeventEvent()
        self._greenlet = None
        self._writer = None
        self._save_dir = None
        self._recording_dir = None
        self._metadata = None
        self._prev_cartesian = None
        self._prev_gripper = None
        self._prev_time = None
        self._controller_template = {
            "movement_enabled": True,
            "success": False,
            "failure": False,
        }
        self._scan_workspace_active = False

    def start(self):
        if self._writer is not None:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        data_root = self.logger.log_dir / "recordings"
        data_root.mkdir(parents=True, exist_ok=True)
        self._save_dir = data_root / timestamp
        self._recording_dir = self._save_dir / "recordings"
        self._recording_dir.mkdir(parents=True, exist_ok=True)
        try:
            save_calibration_info(str(self._save_dir / "calibration.json"))
        except Exception as exc:
            print(f"[DataRecorder] Failed to save calibration snapshot: {exc}")
        self._metadata = {
            "time": timestamp,
            "robot_serial_number": f"{robot_type}-{robot_serial_number}",
            "version_number": droid_version,
            "controller": self.controller_name,
        }
        try:
            with open(self._save_dir / "info.json", "w", encoding="utf-8") as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as exc:
            print(f"[DataRecorder] Failed to save info.json: {exc}")
        trajectory_path = self._save_dir / "trajectory.h5"
        self._writer = TrajectoryWriter(
            str(trajectory_path),
            metadata=self._metadata,
            post_process=self.post_process,
        )
        self._stop_event.clear()
        self._greenlet = gevent_spawn(self._log_loop)

    def stop(self):
        if self._writer is None:
            return
        self._stop_event.set()
        if self._greenlet is not None:
            self._greenlet.join(timeout=5)
            self._greenlet = None
        final_status = deepcopy(self._controller_template)
        try:
            self._writer.close(metadata=final_status)
        except Exception as exc:
            print(f"[DataRecorder] Failed to close writer: {exc}")
        self._writer = None
        self._copy_recordings()

    def _log_loop(self):
        last_error = None
        while not self._stop_event.is_set():
            loop_start = time.time()
            try:
                obs = self.env.get_observation()
            except Exception as exc:
                # Print each distinct error once; a failed camera would repeat it every poll.
                if str(exc) != last_error:
                    last_error = str(exc)
                    print(f"[DataRecorder] Observation error: {exc}")
                gevent_sleep(0.05)
                continue
            last_error = None
            obs = deepcopy(obs)
            self._overwrite_camera_types(obs)
            obs["controller_info"] = deepcopy(self._controller_template)
            obs.setdefault("timestamp", {})
            self._scan_workspace_active = getattr(self.robot_api, "is_scanning_wrist", False)
            record_step = not self._scan_workspace_active
            obs["timestamp"]["skip_action"] = not record_step
            now_ms = int(time.time_ns() // 1_000_000)
            obs["timestamp"]["control"] = {
                "step_start": now_ms,
                "policy_start": now_ms,
                "sleep_start": now_ms,
                "control_start": now_ms,
            }
            if record_step:
                action = self._build_action(obs.get("robot_state", {}))
                timestep = {"observation": obs, "action": action}
                obs["timestamp"]["control"]["step_end"] = int(time.time_ns() // 1_000_000)
                timestep = self._sanitize_for_hdf5(timestep)
                try:
                    self._writer.write_timestep(timestep)
                except Exception as exc:
                    print(f"[DataRecorder] Failed to write timestep: {exc}")
            elapsed = time.time() - loop_start
            wait_time = max(0.0, (1 / self.control_hz) - elapsed)
            if wait_time > 0 and not self._stop_event.is_set():
                gevent_sleep(wait_time)

    def _build_action(self, robot_state):
        cartesian_pos = np.array(robot_state.get("cartesian_position", np.zeros(6)), dtype=np.float32)
        joint_pos = np.array(robot_state.get("joint_positions", []), dtype=np.float32)
        joint_vel = np.array(robot_state.get("joint_velocities", np.zeros_like(joint_pos)), dtype=np.float32)
        gripper_pos = float(robot_state.get("gripper_position", 0.0))
        now = time.time()
        if self._prev_time is None:
            cartesian_vel = np.zeros_like(cartesian_pos)
            gripper_vel = 0.0
        else:
            dt = max(now - self._prev_time, 1e-3)
            cartesian_vel = (cartesian_pos - self._prev_cartesian) / dt
            gripper_vel = (gripper_pos - self._prev_gripper) / dt
        self._prev_cartesian = cartesian_pos
        self._prev_gripper = gripper_pos
        self._prev_time = now
        action_dict = {
            "cartesian_position": cartesian_pos,
            "cartesian_velocity": cartesian_vel,
            "joint_position": joint_pos,
            "joint_velocity": joint_vel,
            "gripper_position": np.array(gripper_pos, dtype=np.float32),
            "gripper_velocity": np.array(gripper_vel, dtype=np.float32),
            "target_cartesian_position": cartesian_pos,
            "target_gripper_position": np.array(gripper_pos, dtype=np.float32),
            "robot_state": deepcopy(robot_state),
        }
        return action_dict

    def _copy_recordings(self):
        if self._recording_dir is None:
            return
        src_dir = self.logger.log_dir / "SVO"
        if not src_dir.exists():
            return
        for file in src_dir.glob("*"):
            if not file.is_file():
                continue
            if file.suffix == ".svo":
                dest_name = f"{file.stem}.svo2"
            else:
                dest_name = file.name
            dest = self._recording_dir / dest_name
            try:
                shutil.copy2(file, dest)
            except Exception as exc:
                print(f"[DataRecorder] Failed to copy {file} -> {dest}: {exc}")

    def _overwrite_camera_types(self, obs):
        available_serials = {cam_key.split("_")[0] for cam_key in obs.get("image", {}).keys()}
        obs["camera_type"] = {
            serial: camera_type_dict[serial]
            for serial in camera_type_dict
            if serial in available_serials
        }

    def _sanitize_for_hdf5(self, data):
        if isinstance(data, dict):
            return {k: self._sanitize_for_hdf5(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._sanitize_for_hdf5(v) for v in data]
        if isinstance(data, tuple):
            return tuple(self._sanitize_for_hdf5(v) for v in data)
        if isinstance(data, np.ndarray) and data.dtype.kind == "U":
            return data.astype("S")
        return data






def run_trial(robot: RealFrankaRobotApi, image_editor: ImageEditApi, env: RobotEnv, instruction, in_context_example: str, code_file=None):
    """Run a single trial with the given instruction.
    
    Args:
        robot: Robot API instance for controlling the robot
        image_editor: API for image visualization and editing
        env: Robot environment
        instruction: Natural language instruction for the task
        code_file: Optional path to code file to execute directly

    Returns:
        True if the planner reported the task complete.
    """
    robot.instruction = instruction  # replans judge success against this trial's instruction
    logger = Logger(instruction, base_log_dir=os.path.join(cfg.paths['data_dir'], "logs_all_tasks"))
    robot.logger = logger
    # Create the per-trial grasp_data subfolder
    try:
        trial_name = logger.log_dir.name
        trial_gd = Path(os.path.join(cfg.paths['data_dir'], "grasp_data")) / trial_name
        trial_gd.mkdir(parents=True, exist_ok=True)
        # Update scan cache to use trial-specific directory
        robot.scan_cache.base_dir = str(trial_gd / "latest_data" / "scan_cache")
        robot.scan_cache.invalidate()  # this trial scans its own scene; never plan around the last one
    except Exception:
        pass
    
    # Start video recording with direct-to-disk writing for memory efficiency
    video_path = str(logger.log_dir / "video.mp4")
    robot.start_video_recorder(fps=20, output_path=video_path)
    
    # Don't catch KeyboardInterrupt here - let it propagate to main loop
    if code_file is not None:
        code = open(code_file, 'r', encoding='utf-8').read()
    requests.post(f"{service_url('curobo')}/clean_map")
    robot.open_gripper()
    robot.move_to_pose([0.35, 0.0, 0.38], [3.14, 0., 0.])
    # Prime wrist cache once for faster, robust perception
    try:
        robot.scan_workspace()
    except Exception as e:
        print(f"Initial wrist scan failed (will lazily rescan on demand): {e}")

    prompt_file = str(REPO_ROOT / "maestro" / "prompts" / "maestro_prompt.txt")
    prompt = open(prompt_file, 'r').read()
    chat = vlm.planner_chat()
    if code_file is None:
        actor_code = plan_first_step(chat, robot, image_editor, logger, prompt, in_context_example)
    else:
        actor_code = code

    code_extractor = VLMCodeExtractor(chat, robot, image_editor, logger)
    code_extractor.actor_code = actor_code
    max_attempts = 20
    env.camera_reader.start_recording(logger.log_dir)
    # Records a DROID-format trajectory (states/actions + per-camera videos) for data collection.
    data_recorder = DataRecorder(env, robot, logger)
    data_recorder.start()
    task_complete = False
    try:
        if code_file is not None:
            code_extractor.execute_code()
        else:
            task_complete = execute_plan_loop(code_extractor, max_attempts)
    finally:
        # Close the SVO files before the DataRecorder copies them.
        try:
            env.camera_reader.stop_recording()
        except Exception:
            pass
        data_recorder.stop()
    return task_complete


# Global robot instance for signal handler
_global_robot = None

def main(args):
    global _global_robot
    
    if not hand_camera_id:
        raise SystemExit("Set cameras.hand_camera_id in config/maestro.yaml to your wrist ZED's serial number.")
    print("Connecting to the robot and cameras...")
    env = RobotEnv()
    if hand_camera_id not in env.camera_reader.camera_dict:
        raise SystemExit(f"Wrist camera {hand_camera_id} not found. Check its USB 3 cable and "
                         "cameras.hand_camera_id in config/maestro.yaml.")
    env.reset()
    use_motionplanner = bool(cfg.robot.use_motionplanner) if (cfg.robot and cfg.robot.use_motionplanner is not None) else True  # config: robot.use_motionplanner
    robot = RealFrankaRobotApi(env, args.instruction, use_motionplanner)
    _global_robot = robot  # Store globally for signal handler to access
    
    image_editor = robot.image_editor
    instruction = args.instruction
    
    # Set up cleanup handlers for graceful shutdown
    def cleanup_on_exit():
        """Clean up resources on program exit.
        
        This runs when the program exits normally (not via Ctrl+C).
        Ensures video is saved.
        """
        if _global_robot and getattr(_global_robot, '_video_active', False):
            print("\n[EXIT] Saving video...")
            _global_robot.stop_video_recorder()
    
    atexit.register(cleanup_on_exit)
    
    # Handle Ctrl+C to save video before exiting 
    def signal_handler(signum, frame):
        """Handle SIGINT (Ctrl+C) to ensure video is saved.
        
        This runs immediately when Ctrl+C is pressed, saves the video,
        then re-raises KeyboardInterrupt to be caught by the main loop.
        """
        if _global_robot:
            print("\n[Ctrl+C] Saving video...")
            _global_robot.stop_video_recorder()
        raise KeyboardInterrupt  # Re-raise to be caught by main loop
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Main trial loop - keeps running until user chooses to exit
    while True:
        try:
            in_context_example = past_trial_logs(args.log) if args.log else ""
            task_complete = run_trial(robot, image_editor, env, instruction, in_context_example, args.code)
            
            # Trial ended normally (not interrupted)
            if task_complete:
                print("\nPlanner reported the task complete.")
            else:
                print("\nTrial ended without the planner reporting the task complete.")
            robot.stop_video_recorder()
            
            # Ask if user wants to run another trial
            print("\nDo you want to run another trial?")
            try:
                instruction = input("Enter instruction (or press Enter to exit): ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            
            if not instruction:
                break  # User pressed Enter without instruction - exit
                
            print("Starting new trial...")
                
        except KeyboardInterrupt:
            # Ctrl+C pressed - trial was interrupted
            # Note: Video is already saved by the signal handler
            print("\nTrial interrupted.")
            
            # Ask if user wants to run another trial after interruption
            print("\nDo you want one more trial?")
            try:
                instruction = input("Enter instruction (or press Enter to exit): ").strip()
            except (KeyboardInterrupt, EOFError):
                break  # Another Ctrl+C or EOF - exit immediately
                
            if not instruction:
                break  # User pressed Enter without instruction - exit
                
            print("Starting new trial...")

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--instruction", type=str, default="", help="Instruction for the VLM to follow. This will be appended to the maestro prompt.")
    parser.add_argument("-c", "--code", type=str, default=None, help="Manually set the path of code file to be executed")
    parser.add_argument("-l", "--log", type=str, default=None, help="Path to the log file")
    args = parser.parse_args(args)
    if not args.instruction and not args.code:
        parser.error("give a task with -i, or a code file to run with -c")
    return args

if __name__ == "__main__":  
    main(parse_args())

