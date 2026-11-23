"""Evaluation observations, demonstration playback and execution budgeting."""

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
    PandaStickMotionPlanningSolver,
)

import robomme_plus.evaluation.batches as planner_denseStep
import robomme_plus.control.reset as reset_panda
import robomme_plus.evaluation.goals as task_goal
from robomme_plus.scoring.failures import clear_failure
from robomme_plus.common.logging_utils import logger

# Pose continuousness and RPY statistics logic unified in shared util to avoid divergent implementations.
from robomme_plus.control.rotations import build_endeffector_pose_dict


def _tensor_to_numpy(value: Any, dtype: np.dtype) -> np.ndarray:
    """Convert a single Tensor to an ndarray of specified dtype; if already ndarray, only convert dtype."""
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return arr


from robomme_plus.evaluation.clock import ExecutionClock
from robomme_plus.evaluation.grounding import fill_grounding, front_segmentation, subgoal_objects


class DemonstrationWrapper(gym.Wrapper):
    """
    Demonstration wrapper (does not include video saving function).

    Main functions:
    1. Automatically generate demonstration Trajectory after environment reset, using Motion Planner.
    2. Record data such as frames, actions, states, subgoals during demonstration for downstream tasks.
    """


    def __init__(
        self,
        env,
        gui_render=False,
        episode_metadata=None,
        include_front_depth=False,
        include_wrist_depth=False,
        include_front_camera_extrinsic=False,
        include_wrist_camera_extrinsic=False,
        include_front_camera_intrinsic=False,
        include_wrist_camera_intrinsic=False,
    ):
        self.execution_clock = ExecutionClock(episode_metadata)
        self._terminal_batch = None
        self._last_step_info = {}
        self.gui_render = gui_render
        self.include_front_depth = include_front_depth
        self.include_wrist_depth = include_wrist_depth
        self.include_front_camera_extrinsic = include_front_camera_extrinsic
        self.include_wrist_camera_extrinsic = include_wrist_camera_extrinsic
        self.include_front_camera_intrinsic = include_front_camera_intrinsic
        self.include_wrist_camera_intrinsic = include_wrist_camera_intrinsic

        super().__init__(env)
        # Tasks may declare a default perturbation; pass False to disable it.
        self.unwrapped.use_demonstrationwrapper = True

        self.demonstration_record_traj = (
            False  # Whether currently in "demonstration recording" phase
        )

        # Consecutive steps without executing "demonstration task", used for truncate judgment
        self.execution_clock.reset()
        self._terminal_batch = None
        # Demonstration trajectory data for this episode (filled by get_demonstration_trajectory)
        self.demonstration_data = None
        self.current_subgoal_segment_filled = None
        # Result of replacing placeholders in current subgoal text with coordinates
        # Whether this episode is judged as successful (for downstream data saving etc)
        self.episode_success = False

        # Total attempts (including first) for screw planning retry during demonstration phase
        self._demo_screw_max_attempts = 1
        # Total attempts (including first) for RRT* planning retry after screw failure
        self._demo_rrt_max_attempts = 3
        # Whether current demonstration task experienced planning failure (for task-level continuation)
        self._current_demo_task_screw_failed = False
        # End-effector pose continuousness cache (wxyz / XYZ-RPY):
        # - _prev_ee_quat_wxyz: Save "sign-aligned" quaternion of previous frame
        # - _prev_ee_rpy_xyz: Save "unwrapped" continuous RPY of previous frame
        # These two caches jointly determine cross-frame continuousness behavior, lifecycle limited to single episode.
        self._prev_ee_quat_wxyz = None
        self._prev_ee_rpy_xyz = None

    def reset(self, **kwargs):
        """Reset environment and generate demonstration trajectory, then execute one initial action step and return unified batch."""
        metadata = self.execution_clock.metadata
        if metadata is not None and kwargs.get("seed") not in (None, metadata["seed"]):
            raise ValueError(
                "Reset seed differs from the supplied episode metadata"
            )
        # Reset latch state
        # Reset non-demonstration step counter to avoid cross-episode accumulation
        self.execution_clock.reset()
        self._terminal_batch = None
        self._last_step_info = {}
        clear_failure(self.unwrapped)
        # Start each episode with clean cache to avoid cross-episode pollution:
        # Do not allow "previous frame pose" from last game to affect current game's first frame unwrapping result.
        self._prev_ee_quat_wxyz = None
        self._prev_ee_rpy_xyz = None

        super().reset(**kwargs)
        for task in self.unwrapped.task_list:
            if not task.get("demonstration", False):
                task.pop("solve", None)
        self.episode_success = False
        # Generate demonstration trajectory batch
        demo_batch = self.get_demonstration_trajectory()

        # Select gripper and initial action based on environment: PatternLock/RouteStick use stick and require online generated action
        if (
            getattr(self.unwrapped, "task_name", self.unwrapped.spec.id)
            == "PatternLock"
            or getattr(self.unwrapped, "task_name", self.unwrapped.spec.id)
            == "RouteStick"
        ):
            gripper = "stick"
        else:
            gripper = None
        if (
            getattr(self.unwrapped, "task_name", self.unwrapped.spec.id)
            == "PatternLock"
            or getattr(self.unwrapped, "task_name", self.unwrapped.spec.id)
            == "RouteStick"
        ):
            action = (
                self.unwrapped.swing_qpos
            )  # These two types of environments require online generated initial action
        else:
            action = reset_panda.get_reset_panda_param("action", gripper=gripper)

        reset_panda.reset_execution_pose(self.unwrapped, gripper=gripper)
        # Hold the initialized pose for one observation; the execution clock
        # starts below, after the task's execution-start hook.
        init_batch = self._step_batch(action)
        merged_batch = planner_denseStep.concat_step_batches([demo_batch, init_batch])
        merged_batch = self._filter_no_record_from_step_batch(merged_batch)
        self.demonstration_data = merged_batch
        execution_start = getattr(self.unwrapped, "on_execution_start", None)
        if callable(execution_start):
            execution_start()

        # Unpack the batch to return only obs and info, but keep the full batch in self.demonstration_data
        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = (
            merged_batch
        )
        info_flat = planner_denseStep.flatten_info(info_batch)
        info_flat.pop("_record_frame", None)
        if info_flat.get("status") != "fail":
            clear_failure(self.unwrapped)
        self.execution_clock.start()
        info_flat.update(self.execution_clock.information())
        return obs_batch, info_flat

    @property
    def steps_without_demonstration(self):
        return self.execution_clock.steps

    def _filter_no_record_from_step_batch(self, batch):
        """
        Only used before reset return: Filter out frames where info_batch['subgoal'] is "NO RECORD".

        Return contract consistent with input batch:
        (obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch)
        """
        if not (isinstance(batch, tuple) and len(batch) == 5):
            return batch
        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = batch

        if (
            not isinstance(reward_batch, torch.Tensor)
            or not isinstance(terminated_batch, torch.Tensor)
            or not isinstance(truncated_batch, torch.Tensor)
        ):
            return batch
        if not isinstance(info_batch, dict):
            return batch

        n = int(reward_batch.numel())
        if n == 0:
            return batch
        if int(terminated_batch.numel()) != n or int(truncated_batch.numel()) != n:
            return batch

        record_frames = info_batch.get("_record_frame")
        if not isinstance(record_frames, list) or len(record_frames) != n:
            return batch

        keep_indices = [
            idx
            for idx, record in enumerate(record_frames)
            if record
        ]
        if len(keep_indices) == n:
            return batch
        # Defensive fallback: Avoid accessing [-1] on empty batch after filtering.
        if len(keep_indices) == 0:
            return batch

        index_reward = torch.as_tensor(
            keep_indices, dtype=torch.long, device=reward_batch.device
        )
        index_terminated = torch.as_tensor(
            keep_indices, dtype=torch.long, device=terminated_batch.device
        )
        index_truncated = torch.as_tensor(
            keep_indices, dtype=torch.long, device=truncated_batch.device
        )

        def _filter_columnar_dict(batch_dict):
            if not isinstance(batch_dict, dict):
                return batch_dict
            filtered = {}
            for key, value in batch_dict.items():
                if isinstance(value, list) and len(value) == n:
                    filtered[key] = [value[i] for i in keep_indices]
                else:
                    filtered[key] = value
            return filtered

        filtered_obs_batch = _filter_columnar_dict(obs_batch)
        filtered_info_batch = _filter_columnar_dict(info_batch)
        filtered_reward_batch = reward_batch.index_select(0, index_reward)
        filtered_terminated_batch = terminated_batch.index_select(0, index_terminated)
        filtered_truncated_batch = truncated_batch.index_select(0, index_truncated)
        return (
            filtered_obs_batch,
            filtered_reward_batch,
            filtered_terminated_batch,
            filtered_truncated_batch,
            filtered_info_batch,
        )

    def _augment_obs_and_info(self, obs, info, action):
        """Extract current step data directly from obs and merge into obs and info to return, bypassing list buffer intermediate."""
        language_goal = task_goal.get_language_goal(
            self.env, getattr(self.unwrapped, "task_name", self.unwrapped.spec.id)
        )

        env_id = getattr(self.unwrapped, "task_name", self.unwrapped.spec.id)
        subgoal_text = getattr(self, "current_task_name", "Unknown")
        grounded_subgoal = self.current_subgoal_segment_filled

        # Extract frames, state, velocity etc directly from obs (no longer read from self.frames etc list)
        image = obs["sensor_data"]["base_camera"]["rgb"][0]
        wrist_image = obs["sensor_data"]["hand_camera"]["rgb"][0]
        state = self.agent.robot.qpos
        # end_effector_velocity = self.agent.robot.links[9].get_linear_velocity()[0], self.agent.robot.links[9].get_angular_velocity()[0]

        # Output end-effector pose as dict, containing pose/quat/rpy representations; also update continuousness cache.
        # squeeze out batch dim: (1, 3) -> (3,), (1, 4) -> (4,)
        _tcp_p = self.agent.tcp.pose.p
        _tcp_q = self.agent.tcp.pose.q
        if _tcp_p.ndim > 1:
            _tcp_p = _tcp_p.squeeze(0)
        if _tcp_q.ndim > 1:
            _tcp_q = _tcp_q.squeeze(0)
        robot_endeffector_pose, self._prev_ee_quat_wxyz, self._prev_ee_rpy_xyz = (
            build_endeffector_pose_dict(
                _tcp_p,
                _tcp_q,
                self._prev_ee_quat_wxyz,
                self._prev_ee_rpy_xyz,
            )
        )

        # ───────── Apply internal inline numpy conversion ─────────
        image_np = _tensor_to_numpy(image, np.uint8)
        wrist_image_np = _tensor_to_numpy(wrist_image, np.uint8)

        robot_endeffector_pose_np = {
            "pose": _tensor_to_numpy(robot_endeffector_pose["pose"], np.float32),
            "quat": _tensor_to_numpy(robot_endeffector_pose["quat"], np.float32),
            "rpy": _tensor_to_numpy(robot_endeffector_pose["rpy"], np.float32),
        }

        eef_state_list_f64 = np.concatenate(
            [
                robot_endeffector_pose_np["pose"].flatten()[:3],
                robot_endeffector_pose_np["rpy"].flatten()[:3],
            ]
        ).astype(np.float64, copy=False)

        # Extract gripper state from the last 2 dims of joint positions
        state_flat = (
            state.detach().cpu().numpy().flatten()
            if hasattr(state, "cpu")
            else np.asarray(state).flatten()
        )

        is_stick_env = getattr(self.unwrapped, "task_name", self.unwrapped.spec.id) in (
            "PatternLock",
            "RouteStick",
        )
        if is_stick_env:
            gripper_state = np.zeros(2, dtype=np.float64)
        else:
            gripper_state = (
                state_flat[7:9]
                if len(state_flat) >= 9
                else np.zeros(2, dtype=np.float64)
            )

        # Only keep first 7 joint dims for joint_state_list
        joint_state = state_flat[:7]

        # ───────── Build new_obs (always-present fields first) ─────────
        new_obs = {
            "front_rgb_list": image_np,
            "wrist_rgb_list": wrist_image_np,
            "joint_state_list": joint_state,
            # 'end_effector_pose_raw': robot_endeffector_pose_np,  # Kept for quick restore if needed.
            "eef_state_list": eef_state_list_f64,
            "gripper_state_list": gripper_state,
        }
        if self.include_front_depth:
            new_obs["front_depth_list"] = _tensor_to_numpy(
                obs["sensor_data"]["base_camera"]["depth"][0], np.int16
            )
        if self.include_wrist_depth:
            new_obs["wrist_depth_list"] = _tensor_to_numpy(
                obs["sensor_data"]["hand_camera"]["depth"][0], np.int16
            )
        if self.include_front_camera_extrinsic:
            _ext = _tensor_to_numpy(
                obs["sensor_param"]["base_camera"]["extrinsic_cv"], np.float32
            )
            if _ext.ndim == 3:
                _ext = _ext.squeeze(0)
            new_obs["front_camera_extrinsic_list"] = _ext
        if self.include_wrist_camera_extrinsic:
            _ext = _tensor_to_numpy(
                obs["sensor_param"]["hand_camera"]["extrinsic_cv"], np.float32
            )
            if _ext.ndim == 3:
                _ext = _ext.squeeze(0)
            new_obs["wrist_camera_extrinsic_list"] = _ext

        # ───────── Build new_info (always-present fields first) ─────────
        new_info = {
            **{key: info[key] for key in ("success", "fail", "trial_reset", "trial_index",
                                        "trials_remaining", "message", "failure_reason") if key in info},
            "_record_frame": subgoal_text != "NO RECORD",
            "task_goal": language_goal,
            "simple_subgoal_online": subgoal_text,
            "grounded_subgoal_online": grounded_subgoal,
        }
        if self.include_front_camera_intrinsic:
            _intr = _tensor_to_numpy(
                obs["sensor_param"]["base_camera"]["intrinsic_cv"], np.float32
            )
            if _intr.ndim == 3:
                _intr = _intr.squeeze(0)
            new_info["front_camera_intrinsic"] = _intr
        if self.include_wrist_camera_intrinsic:
            _intr = _tensor_to_numpy(
                obs["sensor_param"]["hand_camera"]["intrinsic_cv"], np.float32
            )
            if _intr.ndim == 3:
                _intr = _intr.squeeze(0)
            new_info["wrist_camera_intrinsic"] = _intr

        variation = getattr(self.unwrapped, "_perceptual_variation", None)
        if variation is not None and variation.noise is not None:
            new_obs, new_info = variation.noise.transform(new_obs, new_info)
        return new_obs, new_info


    def _compute_segmentation_and_fill_subgoal(self, obs):
        native = self.unwrapped
        template = getattr(native, "current_subgoal_segment", None)
        simple = getattr(native, "current_task_name", "Unknown")
        if simple == "All tasks completed":
            return simple, False
        id_map = getattr(native, "segmentation_id_map", None) or {}
        objects = subgoal_objects(getattr(native, "current_segment", None))
        return fill_grounding(
            template, simple, objects, front_segmentation(obs), id_map
        ), False

    def _normalize_action_for_env_step(self, action) -> np.ndarray:
        from robomme_plus.control.actions import normalize_joint_action

        return normalize_joint_action(self.unwrapped, action)

    def _step_batch(self, action):
        """Internal step returning full batch format (dict-of-lists for both obs and info).

        Used by reset() and other internal callers that need batch-compatible output
        for concat_step_batches.
        """
        if self._terminal_batch is not None:
            return self._terminal_batch
        normalized_action = self._normalize_action_for_env_step(action)
        obs, reward, terminated, truncated, info = super().step(normalized_action)
        if self.execution_clock.advance():
            truncated = torch.tensor([True])
        if self.execution_clock.active and not bool(info.get("success", False)):
            status = "fail" if bool(terminated.any()) else "timeout" if bool(truncated.any()) else "ongoing"
            retry = self._retry_execution_raw(status)
            if retry is not None:
                obs, info = retry
                reward = torch.zeros_like(reward)
                terminated = torch.zeros_like(terminated)
                truncated = torch.zeros_like(truncated)

        # ---------- Subgoal segmentation and placeholder filling: Internally parse segmentation from obs, calculate center, fill placeholders ----------
        filled_text, failed_match = self._compute_segmentation_and_fill_subgoal(obs)
        current_subgoal_segment = getattr(
            self.unwrapped, "current_subgoal_segment", None
        )
        self.current_subgoal_segment_filled = (
            filled_text if filled_text is not None else current_subgoal_segment
        )

        # ---------- Update episode_success based on terminated and info["success"] ----------
        if terminated.any():
            if info.get("success") == torch.tensor([True]) or (
                isinstance(info.get("success"), torch.Tensor)
                and info.get("success").item()
            ):
                self.episode_success = True
                # print("Episode success detected, data will be saved")
            else:
                self.episode_success = False
                # print("Episode failed, data will be discarded")

        obs, info = self._augment_obs_and_info(obs, info, normalized_action)

        # Compute status field from terminated/truncated/success
        raw_success = info.get("success")
        is_success = (
            isinstance(raw_success, torch.Tensor) and raw_success.item()
        ) or raw_success is True
        if is_success:
            info["status"] = "success"
        elif terminated.any():
            info["status"] = "fail"
        elif truncated.any():
            info["status"] = "timeout"
        else:
            info["status"] = "ongoing"

        info.update(self.execution_clock.information())
        self._last_step_info = info.copy()
        batch = planner_denseStep.to_step_batch(
            [(obs, reward, terminated, truncated, info)]
        )
        if self.execution_clock.active and (
            bool(terminated.any()) or bool(truncated.any())
        ):
            self._terminal_batch = batch
        return batch

    def _retry_execution_raw(self, status):
        if not self.execution_clock.active or status not in ("fail", "timeout"):
            return None
        retry = getattr(self.unwrapped, "retry_execution_trial", None)
        if not callable(retry):
            return None
        notice = retry(status, self.execution_clock.steps)
        if notice is None:
            return None
        clear_failure(self.unwrapped)
        self.execution_clock.restart_trial()
        self._terminal_batch = None
        self.episode_success = False
        self._prev_ee_quat_wxyz = self._prev_ee_rpy_xyz = None
        info = self.unwrapped.get_info()
        info.update(notice)
        obs = self.unwrapped.get_obs(info=info)
        return obs, info

    def try_execution_retry(self, status):
        """Also handle budget expiry caused by an action-error penalty."""
        retry = self._retry_execution_raw(status)
        if retry is None:
            return None
        raw_obs, info = retry
        self.current_subgoal_segment_filled, _ = self._compute_segmentation_and_fill_subgoal(raw_obs)
        obs, info = self._augment_obs_and_info(raw_obs, info, None)
        info.update(status="ongoing", **self.execution_clock.information())
        self._last_step_info = info.copy()
        obs_batch, _, _, _, info_batch = planner_denseStep.to_step_batch([(obs, 0., False, False, info)])
        flat = planner_denseStep.flatten_info(info_batch)
        flat.pop("_record_frame", None)
        return obs_batch, 0., False, False, flat

    def observe_current(self):
        """Return one current frame without stepping physics or scoring again."""
        raw_obs = self.unwrapped.get_obs(info={})
        filled, _ = self._compute_segmentation_and_fill_subgoal(raw_obs)
        self.current_subgoal_segment_filled = filled
        current_info = {k: v for k, v in self._last_step_info.items()
                        if k not in ("trial_reset", "message", "trials_remaining")}
        obs, info = self._augment_obs_and_info(raw_obs, current_info, None)
        info["status"] = self._last_step_info.get("status", "ongoing")
        info.update(self.execution_clock.information())
        obs_batch, _, _, _, info_batch = planner_denseStep.to_step_batch(
            [(obs, 0.0, False, False, info)]
        )
        info_flat = planner_denseStep.flatten_info(info_batch)
        info_flat.pop("_record_frame", None)
        return obs_batch, info_flat

    def step(self, action):
        """Execute one step and return (obs_batch, reward, terminated, truncated, info).

        obs_batch is dict[str, list]; info is a flat dict (last values only).

        The outer FailAwareWrapper reports action errors and charges their budget penalty.
        """
        batch = self._step_batch(action)
        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = batch
        info_flat = planner_denseStep.flatten_info(info_batch)
        # Demonstration collection calls step() internally. Keep its recording
        # marker until reset() filters out the hidden reset motions; public
        # execution observations do not need this internal field.
        if self.execution_clock.active:
            info_flat.pop("_record_frame", None)
        return (
            obs_batch,
            reward_batch[-1],
            terminated_batch[-1],
            truncated_batch[-1],
            info_flat,
        )

    def close(self):
        """Close environment, release resources (this wrapper no longer saves video)."""
        super().close()
        return None

    def get_demonstration_trajectory(self):
        """
        Generate Demonstration Trajectory.

        Flow:
        1. Select appropriate Motion Planner (PandaArm or PandaStick) based on environment ID.
        2. Iterate task list (task_list), find tasks marked as demonstration.
        3. For each demonstration task, wrap entire solve call with _collect_dense_steps,
           monkey-patch planner.env.step to collect all env.step calls
           (including move_to_pose_with_screw, follow_path, direct env.step and all other paths).
        4. Return unified batch (obs/info dict values as list, reward/terminated/truncated as 1D tensor).
        """
        # Lazy load FailAware planner; fallback to original planner implementation if import fails
        try:
            from robomme_plus.control.planner import (
                FailAwarePandaArmMotionPlanningSolver,
                FailAwarePandaStickMotionPlanningSolver,
                ScrewPlanFailure,
            )
        except Exception as exc:
            logger.debug(
                f"[DemonstrationWrapper] Warning: failed to import planner_fail_safe, fallback to base planners: {exc}"
            )
            FailAwarePandaArmMotionPlanningSolver = PandaArmMotionPlanningSolver
            FailAwarePandaStickMotionPlanningSolver = PandaStickMotionPlanningSolver
            ScrewPlanFailure = RuntimeError

        # Select motion planner by environment: PatternLock/RouteStick use stick planner, others use arm planner
        if (
            getattr(self.unwrapped, "task_name", self.unwrapped.spec.id)
            == "PatternLock"
            or getattr(self.unwrapped, "task_name", self.unwrapped.spec.id)
            == "RouteStick"
        ):
            planner = FailAwarePandaStickMotionPlanningSolver(
                self,
                debug=False,
                vis=False,
                base_pose=self.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                joint_vel_limits=0.3,
            )
        else:
            planner = FailAwarePandaArmMotionPlanningSolver(
                self,
                debug=False,
                vis=False,
                base_pose=self.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                **getattr(self.unwrapped, "reference_planner_options", lambda: {})(),
            )

        # Wrap screw call at planner instance level: automatic switch to RRT* retry after screw failure
        original_move_to_pose_with_screw = planner.move_to_pose_with_screw
        original_move_to_pose_with_rrt = planner.move_to_pose_with_RRTStar

        def _move_to_pose_with_screw_then_rrt_retry(*args, **kwargs):
            for attempt in range(1, self._demo_screw_max_attempts + 1):
                try:
                    result = original_move_to_pose_with_screw(*args, **kwargs)
                except ScrewPlanFailure as exc:
                    logger.debug(
                        f"[DemonstrationWrapper] screw planning failed "
                        f"(attempt {attempt}/{self._demo_screw_max_attempts}): {exc}"
                    )
                    continue

                # Compatible with non-FailAware fallback scenario: Original planner may return -1 directly
                if isinstance(result, int) and result == -1:
                    logger.debug(
                        f"[DemonstrationWrapper] screw planning returned -1 "
                        f"(attempt {attempt}/{self._demo_screw_max_attempts})"
                    )
                    continue

                return result

            logger.debug(
                "[DemonstrationWrapper] screw planning exhausted; "
                f"fallback to RRT* (max {self._demo_rrt_max_attempts} attempts)"
            )

            # A curved RRT fallback is not a valid PatternLock line. Let its
            # task controller try the continuous Cartesian fallback instead.
            if self.unwrapped.task_name == "PatternLock":
                return -1

            for attempt in range(1, self._demo_rrt_max_attempts + 1):
                try:
                    result = original_move_to_pose_with_rrt(*args, **kwargs)
                except Exception as exc:
                    logger.debug(
                        f"[DemonstrationWrapper] RRT* planning failed "
                        f"(attempt {attempt}/{self._demo_rrt_max_attempts}): {exc}"
                    )
                    continue

                if isinstance(result, int) and result == -1:
                    logger.debug(
                        f"[DemonstrationWrapper] RRT* planning returned -1 "
                        f"(attempt {attempt}/{self._demo_rrt_max_attempts})"
                    )
                    continue

                return result

            self._current_demo_task_screw_failed = True
            logger.debug(
                "[DemonstrationWrapper] screw->RRT* planning exhausted; return -1"
            )
            return -1

        planner.move_to_pose_with_screw = _move_to_pose_with_screw_then_rrt_retry
        tasks = getattr(self, "task_list", [])
        self.task_list_length = len(tasks)
        logger.debug(f"Task list length: {self.task_list_length}")

        demonstration_tasks = [
            task for task in tasks if task.get("demonstration", False)
        ]
        self.non_demonstration_task_length = len(tasks) - len(demonstration_tasks)
        logger.debug(
            f"Non-demonstration task length: {self.non_demonstration_task_length}"
        )

        all_collected_steps = []

        # Iterate and execute each demonstration task: Set demonstration_record_traj=True, call task's solve(planner)
        # Wrap entire solve with _collect_dense_steps, monkey-patch planner.env.step,
        # to collect all env.step calls (including follow_path, direct env.step etc underlying paths)
        for idx, task_entry in enumerate(demonstration_tasks):
            self.unwrapped.demonstration_record_traj = True
            self._current_demo_task_screw_failed = False
            task_name = task_entry.get("name", f"Task {idx}")
            logger.debug(
                f"Executing task {idx + 1}/{len(demonstration_tasks)}: {task_name}"
            )

            solve_callable = task_entry.get("solve")
            if not callable(solve_callable):
                raise ValueError(f"Task '{task_name}' must supply a callable 'solve'.")

            self.evaluate(solve_complete_eval=True)

            def _solve_task():
                try:
                    solve_result = solve_callable(self, planner)
                except ScrewPlanFailure as exc:
                    raise RuntimeError(f"Demonstration task '{task_name}' failed") from exc
                if ((isinstance(solve_result, int) and solve_result == -1)
                        or self._current_demo_task_screw_failed):
                    raise RuntimeError(f"Demonstration task '{task_name}' failed after planning retries")
                return solve_result

            try:
                task_steps = planner_denseStep._collect_dense_steps(
                    planner,
                    _solve_task,
                )
            except Exception:
                # Never provide a partial demonstration to an evaluated policy.
                self.unwrapped.demonstration_record_traj = False
                raise
            all_collected_steps.extend(task_steps)
            self.evaluate(solve_complete_eval=True)

        self.unwrapped.demonstration_record_traj = False  # Demonstration ends, subsequent steps perform subgoal judgment normally
        return planner_denseStep.to_step_batch(all_collected_steps)
