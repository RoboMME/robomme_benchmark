"""ASPIRE's LIBERO typed API backed by RoboMME-Plus public one-step control.

Reuses upstream trajectory execution and gripper feedback loops. Perception,
IK, planning, program execution and the learning workflow remain ASPIRE's.
"""
from __future__ import annotations

from pathlib import Path
import uuid

import numpy as np

from benchmark_io.episode import Episode
from benchmark_io.protocol import Study
from cap_harness.contracts import CameraObservation, Observation, Pose, RobotState, StepResult, TaskContext
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.geometry import matrix_to_quaternion_wxyz
from cap_harness.libero.adapter import LiberoAdapter, PANDA_JOINT_NAMES


def pose(matrix):
    return Pose(position=matrix[:3, 3], quaternion_wxyz=matrix_to_quaternion_wxyz(matrix[:3, :3]), frame="robot_base")


class RoboMMEPlusAdapter(LiberoAdapter):
    # Native Panda joint-position control can hold intermediate finger widths.
    gripper_holds_intermediate = True

    def __init__(self, *, study_root, output_root, purpose="development", run_observer=None, episode_factory=Episode):
        super().__init__(run_observer=run_observer, post_settle_steps=0, camera_width=256, camera_height=256)
        self.study = Study.open(study_root)
        if self.study.config["method"] != "aspire":
            raise ValueError("Not an ASPIRE study")
        self.output_root = Path(output_root)
        self.purpose, self.episode_factory = purpose, episode_factory
        self.episode = None
        self._demonstration = ()

    @property
    def embodiment(self):
        return "robomme_plus"

    def _require_env(self):
        if self.episode is None or self._closed:
            raise RuntimeError("No active episode")
        # Only trusted upstream planning-context code uses this guard.
        return self.episode

    def reset(self, task_ref, seed):
        if self.episode is not None or self._closed:
            raise RuntimeError("Create a fresh adapter for each budgeted attempt")
        task = task_ref if isinstance(task_ref, str) else task_ref.task_name
        if task not in self.study.config["records"]:
            raise ValueError("Only released training tasks may be reset in development")
        # 'seed' here is the released episode index. No procedural seed API.
        self.episode = self.episode_factory(task, seed, study=self.study,
            output=self.output_root / f"{task}_{uuid.uuid4().hex[:12]}", purpose=self.purpose)
        self.control_frequency = self.episode.frequency
        self.control_period_s = 1.0 / self.control_frequency
        self.horizon = self.episode.limit
        self._seed = seed
        task_order = self.study.config.get("task_order", sorted(self.study.config["records"]))
        self._task_context = TaskContext(suite="robomme_plus", task_id=task_order.index(task),
            task_name=task, language=self.episode.instruction, family="robomme_plus",
            metadata=dict(control_frequency_hz=self.control_frequency, max_steps=self.horizon,
                          conditioning_video_frames=len(self.episode.frames)-1,
                          conditioning_video_sampling=getattr(self.episode,'video_sampling',None),
                          conditioning_video_tool="robomme_plus.get_conditioning_video"))
        self._demonstration = tuple(self._observation(frame) for frame in self.episode.frames[:-1])
        return self.get_observation()

    def _state(self, frame):
        # Use LIBERO's Panda tool reference: panda_hand translated +0.1 m in local Z.
        # This lets ASPIRE's existing PyRoki/cuRobo clients keep their hand conversion.
        # The native TCP can differ (e.g. stick); joint actions need no tool-tip assumption.
        base_from_world = np.linalg.inv(self.episode.robot_base_world)
        from scipy.spatial.transform import Rotation
        tcp_world = np.eye(4)
        tcp_world[:3, :3] = Rotation.from_euler("xyz", frame["eef_state"][3:6]).as_matrix()
        tcp_world[:3, 3] = frame["eef_state"][:3]
        hand = base_from_world @ tcp_world @ self.episode.native_tcp_to_hand
        hand[:3, 3] += hand[:3, :3] @ np.array([0., 0., .1])
        return RobotState(joint_positions={"primary": frame["joint_state"]},
            joint_velocities={"primary": np.zeros(7)},
            end_effector_poses={"primary": pose(hand)},
            gripper_positions={"primary": float(np.clip(np.mean(frame["gripper_state"]) / .04, 0, 1))},
            base_frame="robot_base", joint_names={"primary": PANDA_JOINT_NAMES},
            timestamp_s=self._timestamp(frame), embodiment=self.embodiment)

    def _timestamp(self, frame):
        return (frame["frame_index"] if frame["phase"] == "video_demo" else frame["control_step"]) / self.control_frequency

    def _observation(self, frame):
        base_from_world = np.linalg.inv(self.episode.robot_base_world)
        cameras = {}
        for name, data in frame["cameras"].items():
            cameras[name] = CameraObservation(rgb=data["rgb"], depth_m=data["depth_m"],
                intrinsics=data["intrinsics"], frame=name,
                camera_pose=pose(base_from_world @ data["camera_to_world"]),
                timestamp_s=self._timestamp(frame))
        return Observation(cameras=cameras, robot_state=self._state(frame), task_context=self._task_context,
                           timestamp_s=self._timestamp(frame))

    def get_observation(self):
        self._require_env()
        return self._observation(self.episode.current)

    def get_robot_state(self):
        self._require_env()
        return self._state(self.episode.current)

    def get_conditioning_video(self):
        """Ordered reset-video observations; excludes the first execution frame."""
        self._require_env()
        return self._demonstration

    def native_step(self, action):
        self._require_env()
        if self.episode.terminated or self.episode.truncated:
            return StepResult(ok=False, observation=self.get_observation(),
                terminated=self.episode.terminated, truncated=self.episode.truncated,
                error=ApiError(ErrorCode.ADAPTER_FAILED, 'Episode has ended; no further control steps'))
        if action.embodiment != self.embodiment or set(action.arms) != {"primary"}:
            raise ValueError("Use a single RoboMME-Plus primary-arm joint command")
        command = action.arms["primary"]
        gripper = command.gripper_position
        if gripper is None:
            gripper = self.get_robot_state().gripper_positions["primary"]
        native = np.r_[command.target, 2 * gripper - 1]
        if self._run_observer is not None:
            self._run_observer.before_step(action)
        try:
            self.episode.step(native)
            self._elapsed_steps = self.episode.steps
            result = StepResult(ok=True, observation=self.get_observation(),
                               terminated=self.episode.terminated, truncated=self.episode.truncated)
        except Exception as exc:
            result = StepResult(ok=False, error=ApiError(ErrorCode.ADAPTER_FAILED,
                               f"RoboMME-Plus step failed: {type(exc).__name__}: {exc}"))
        if self._run_observer is not None:
            self._run_observer.after_step(action, result)
        return result

    def check_success(self):
        self._require_env()
        return self.episode.success

    def compute_reward(self):
        return None

    def get_task_metadata(self):
        return dict(task=self._task_context.task_name, language=self._task_context.language,
                    control_frequency_hz=self.control_frequency, max_steps=self.horizon)

    def get_controller_metadata(self):
        return dict(arm_names=["primary"], camera_names=["front", "wrist"],
                    control_frequency_hz=self.control_frequency, controller="pd_joint_pos",
                    joint_command_semantics="Absolute seven-joint target; one native control tick",
                    gripper_semantics="0 closed, 1 open; native -1 closed, +1 open; ignored on stick tasks",
                    tool_reference="panda_hand + local [0,0,0.1] m, in robot_base",
                    native_tcp_to_panda_hand=self.episode.native_tcp_to_hand.tolist(),
                    robot_base_to_world=self.episode.robot_base_world.tolist(),
                    joint_velocity_semantics="Unavailable from public observations; zero placeholders")

    def close(self):
        if not self._closed:
            if self.episode is not None:
                self.episode.close()
            self._closed = True
