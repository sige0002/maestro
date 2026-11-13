"""Shared Maestro coding-agent execution and visual replanning.

Extracted from the upstream runner so every embodiment executes the same agent,
not a separate imitation of its loop. Hardware lifecycle stays in each runner.
"""
from __future__ import annotations
from datetime import datetime
import io
import sys
from pathlib import Path
import PIL.Image as Image
from pydantic import BaseModel

class Replan(BaseModel):
    """Response model for replanning after each step execution."""
    reason_and_plan: str
    success: bool
    task_complete: bool
    actor_code: str


class FirstStepPlan(BaseModel):
    """Response model for the initial planning step."""
    describe_scene: str
    reason_and_plan: str
    actor_code: str


def format_floats(obj):
    if isinstance(obj, float):
        return round(obj, 2)
    elif isinstance(obj, list):
        return [format_floats(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: format_floats(v) for k, v in obj.items()}
    return obj


class _Tee(io.TextIOBase):
    """Write to several text streams at once."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class VLMCodeExtractor:
    def __init__(self, chat, robot, image_editor, logger):
        self.chat = chat
        self.robot = robot
        self.image_editor = image_editor
        # Names visible to actor code
        self.local_vars = {
            "robot": self.robot,
            "image_editor": self.image_editor,
        }
        self.logger = logger
        self.output = ""
        self.actor_code = ""

    def execute_code(self):
        code = self.actor_code
        Path(self.logger.log_dir, f"{self.logger.step:02d}-actor.py").write_text(code)
        print("Executing code...", code)

        # Log code execution start
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        try:
            with open(self.logger.timestamp_log_file, "a", encoding="utf-8") as f:
                # Extract first line of code as subtask indicator
                first_line = code.strip().split('\n')[0] if code.strip() else "No code"
                f.write(f"{timestamp} | CODE_EXEC | Starting: {first_line[:100]}...\n")
        except Exception:
            pass

        # Show tool output live and capture it for the planner's replan.
        stdout_capture = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(old_stdout, stdout_capture)
        sys.stderr = _Tee(old_stderr, stdout_capture)
        try:
            # Use self.local_vars for both globals and locals so imports and variables
            # are stored in the same namespace and accessible in comprehensions
            if '__builtins__' not in self.local_vars:
                self.local_vars['__builtins__'] = __builtins__
            exec(code, self.local_vars, self.local_vars)
        except Exception as e:
            print(f"Error executing code: {e}")
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
        print("Execution finished.")
        output = stdout_capture.getvalue()
        self.logger.log_text(f"Code output: {output}")
        self.output = output
        return output

    def replan(self, i):
        state = self.robot.get_robot_state()
        image = self.image_editor.feedback()
        image = Image.fromarray(image)
        self.logger.save_image(image, i, 98)
        replan_result = self.send_message([
            f"Goal Instruction: {self.robot.instruction}",
            f"Step {i} execution results: Code Output (the code output is partial and not always reliable. Judge success from the sub step goal or the final goal instruction together with the current image; the image is the best way to judge whether the step succeeded): {self.output}",
            f"Final State (robot only; do not over-weight this, but do check the hints below): {format_floats(state)}",
            *getattr(self.robot, "replan_hints", DEFAULT_REPLAN_HINTS),
            "Final Image:", image,
            (
                "Describe the execution result and compare it with the goal or your sub step goal, is this step successful? If the image shows the whole goal instruction achieved, set task_complete to true; the trial then ends and actor_code is not run. Otherwise, if the step succeeded, output the code for the next step. If it failed, summarize what caused the failure, find the problem of your code, write code to return to some free state, and then rewrite a better code for the task."
            )
        ])
        self.output = ""
        Path(self.logger.log_dir, f"{self.logger.step:02d}-replan.json").write_text(
            replan_result.model_dump_json(indent=2))
        print(replan_result)

        # Log the planner's reason_and_plan with timestamp for visualization sync
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        try:
            with open(self.logger.timestamp_log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} | PLANNER_RESPONSE | Step {i} reason_and_plan: {replan_result.reason_and_plan}\n")
        except Exception:
            pass

        self.logger.log_text(f"Step {i} replanning response:\n")
        self.logger.log_text("Success: "+ str(replan_result.success)+'\n')
        self.logger.log_text("Task complete: "+ str(replan_result.task_complete)+'\n')
        self.logger.log_text("Reason and plan"+replan_result.reason_and_plan+'\n')
        self.logger.log_text(f"Actor code:\n{replan_result.actor_code}\n")
        actor_code = replan_result.actor_code
        self.actor_code = actor_code
        self.logger.step += 1
        return replan_result

    def send_message(self, content):
        replan_result, _ = self.chat.send(content, Replan)
        return replan_result


DEFAULT_REPLAN_HINTS = [
            "Hint 1: Remember to check the gripper status indicated by the gripper_fraction_closed key, where 0 means open and 1 means closed. When you want to check whether a grasp is successful, make sure gripper_fraction_closed is somewhere between 0 and 1. Usually it cannot be very close to 1.0, which means the gripper is fully closed with nothing in between: you probably missed the object or it slipped, so replan and try again. But if you are grasping something very thin, like the edge of a cup, gripper_fraction_closed can be very close to 1.0; check visually whether the grasp succeeded in this case.",
            "Hint 2: If you see errors like 'Error executing code: Expecting value: line 1 column 1 (char 0)' and you cannot move the arm to the desired pose, the combination of the target pose and the current gripper orientation is most likely causing a kinematic issue, such as a joint limit or singularity. To resolve this (usually after grasping), instead of keeping the grasp orientation, first rotate the gripper to a neutral, downward-facing orientation while keeping the arm as a whole at a central, elevated position. This change in configuration will likely let the motion planner find a valid path for the next move.",
            "Hint 3: There are three perception modules, locate_objects, locate_object_by_segmentation and locate_object_part, used to locate objects or a certain part of an object to interact with. locate_objects suits most cases; it uses the VLM's built-in pointing and is relatively fast and accurate. locate_object_by_segmentation and locate_object_part use GroundedSAM bounding boxes. locate_object_by_segmentation picks the center of the bounding box, which is also quick; locate_object_part selects keypoints from the bounding box and can be slow. The rule of thumb: first try locate_objects (it works for an object or a specific part of an object); if it does not work, consider rescanning the scene or switching to locate_object_by_segmentation (fast) or locate_object_part (slow but accurate).",
            "Hint 4: Use the perception modules before generating and executing a grasp. At the very beginning of your first step, detect every relevant object in the subtask without skipping any (for a multi-stage task, the objects of each stage: for wiping the whiteboard, detect the whiteboard and the eraser; for putting A into B, detect both A and B). For multi-step pick-and-place tasks, detect all objects once at the beginning using locate_objects, store the results in a dictionary like `object_positions = {item['label']: np.array(item['3d_position']) for item in results}`, then reuse this dictionary in later steps instead of calling locate_objects again. This saves about 25-35 seconds per redundant call. Objects typically don't move between your operations, so their positions remain valid. The perception modules may not work correctly while the gripper holds something (they use the wrist camera), so avoid calling them when the robot is already holding something (in the middle of a subtask). In later steps, if you need to interact with an object precisely at a specific part or location, you can use the locate_object_part function. You cannot detect or estimate keypoints of two objects in a single locate_object_by_segmentation or locate_object_part call; call it for each object separately.",
            "Hint 5: If the Goal Instruction has multiple stages, do not skip any of them.",
            "Hint 6: If you want a functional grasp (grasping a certain part of an object for tool use, like the handle of a knife to cut), think twice before using propose_grasps/execute_grasp (their grasp poses are usually too general for tool use). Try estimating the grasp pose yourself from the object's feature points returned by the locate_object_part function.",
            "Hint 7: If locate_objects doesn't work, locate_object_by_segmentation is cheap, fast and usually reliable; try it first unless you want to interact with the object precisely at a specific part or location.",
            "Hint 8: It is always safe to move to a safe height before moving horizontally, to avoid hitting other objects on the table.",
            "Hint 9: Before you consider any grasp, answer this question: what do you want to do after the grasp? Do you need perception right after grasping to detect other objects to interact with? If yes, always do the perception first, since your camera will be blocked after grasping.",
            "Hint 10: For precise orientation and spatial reasoning, always use the vector functions such as measure_relative_rotation, rotate_by, measure_vector and measure_distance, rather than estimating a target pose directly.",
            "Hint 11: Error executing code: 'dt' usually means the motion planner failed (either because of a potential collision along the planned path or a potential joint limit violation). Replan and try again, or change your method. If you see this several times in a row, first move back to the neutral pose, then try to move to the desired pose again. Almost always this is because of an expected collision, not a joint singularity, so think about how a collision could happen and how to avoid it; this becomes a path planning problem. Consider whether you want to keep a top-down orientation (this pose sometimes collides when placing an object in a cluttered space), and note that raising the object higher can sometimes solve the problem.",
            "Hint 12: For perception, if you cannot find an object, first consider rescanning the scene with the refresh_scan function, then try again. If you still cannot find it, switch to another perception module like locate_object_by_segmentation or locate_object_part.",
]


def plan_first_step(chat, robot, image_editor, logger, prompt, history=""):
    """The upstream first-step schema and perception-before-grasp instruction."""
    result, raw = chat.send([
        prompt, "Goal instruction: " + robot.instruction,
        "History Code and failure analysis:\n" + history,
        f"State: {format_floats(robot.get_robot_state())}; Image:",
        Image.fromarray(image_editor.feedback()),
        "Describe the scene and plan the steps. Write code for the first step only. "
        "Before grasping, determine what to do after the grasp and perceive relevant "
        "destinations now, since the wrist camera can be blocked after grasping."
    ], FirstStepPlan)
    logger.log_text(raw)
    Path(logger.log_dir, "first-plan.json").write_text(raw)
    return result.actor_code


def execute_plan_loop(extractor, max_steps=20, on_step=None, completion_guard=None):
    """Execute, observe, replan; failed code and images inform the next attempt."""
    for step in range(1, max_steps + 1):
        extractor.execute_code()
        if on_step is not None:
            on_step(step)
        result = extractor.replan(step)
        if result.task_complete:
            if completion_guard is None:
                return True
            error = completion_guard()
            if error is None:
                return True
            extractor.output += "\nCompletion rejected by measured-state check: " + error
            # The agent must generate the corrective action itself.
            result = extractor.replan(step)
            if result.task_complete:
                return False
    return False
