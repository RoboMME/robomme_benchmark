"""
DemonstrationWrapper: Wrap another layer outside Robomme environment to automatically generate demonstration trajectories and record frames/actions/states/subgoals, etc.

- Call get_demonstration_trajectory() after reset, use Motion Planner to execute tasks marked with demonstration and record trajectory.
- step receives joint space action, performs segmentation and subgoal placeholder filling, trajectory recording, truncate and success judgment. ee_pose->joint is handled by outer EndeffectorDemonstrationWrapper.
- Does not include video saving function; reset/step returns unified dense batch; step injects current step frames/subgoal etc via _augment_obs_and_info.
"""
import copy
import re
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import h5py
import numpy as np
import sapien.physx as physx
import torch
import cv2
import colorsys
import imageio

from mani_skill import get_commit_info
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils, sapien_utils
from mani_skill.utils.io_utils import dump_json
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.structs.types import Array
from mani_skill.utils.wrappers import CPUGymWrapper

from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.motionplanner_stick import PandaStickMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from ..robomme_env.utils import task_goal
from ..robomme_env.utils.vqa_options import get_vqa_options

from ..robomme_env.utils import reset_panda

from ..robomme_env.utils import planner_denseStep
# Pose continuousness and RPY statistics logic unified in shared util to avoid divergent implementations.
from ..robomme_env.utils.rpy_util import build_endeffector_pose_dict

from ..logging_utils import logger

class DemonstrationWrapper(gym.Wrapper):
    """
    Demonstration wrapper (does not include video saving function).

    Main functions:
    1. Automatically generate demonstration Trajectory after environment reset, using Motion Planner.
    2. Record data such as frames, actions, states, subgoals during demonstration for downstream tasks.
    """
    def __init__(self, env, max_steps_without_demonstration, gui_render, **kwargs):
        # **kwargs for compatibility with old calls (e.g. save_video=..., action_space=...), no longer used
        # Max steps without demonstration: truncate episode if demonstration task not executed exceeding this
        self.max_steps_without_demonstration = max_steps_without_demonstration
        self.gui_render = gui_render

        super().__init__(env)
        self.unwrapped.use_demonstrationwrapper = True

        self.demonstration_record_traj = False  # Whether currently in "demonstration recording" phase

        # Consecutive steps without executing "demonstration task", used for truncate judgment
        self.steps_without_demonstration = 0
        # Prevent re-entering step in "append extra step at termination" logic
        self._doing_extra_step = False
        # Demonstration trajectory data for this episode (filled by get_demonstration_trajectory)
        self.demonstration_data = None
        # Result of replacing placeholders in current subgoal text with coordinates
        self.current_subgoal_segment_filled = None
        # Whether this episode is judged as successful (for downstream data saving etc)
        self.episode_success = False

        self._failed_match_save_count = 0
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

        # Consistent with RecordWrapper: Generate high-distinctiveness color map by object ID for segmentation visualization
        def generate_color_map(n=100, s_min=0.70, s_max=0.95, v_min=0.78, v_max=0.95):
            phi = 0.6180339887498948
            color_map = {}
            for i in range(1, n + 1):
                h = (i * phi) % 1.0
                s = s_min + (s_max - s_min) * ((i % 7) / 6)
                v = v_min + (v_max - v_min) * (((i * 3) % 5) / 4)
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                color_map[i] = [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))]
            return color_map

        self.color_map = generate_color_map(10000)




    def reset(self, **kwargs):
        """Reset environment and generate demonstration trajectory, then execute one initial action step and return unified batch."""
        # Reset latch state
        self.last_subgoal_segment = None
        self.latched_replacements = None
        self._failed_match_save_count = 0
        # Reset non-demonstration step counter to avoid cross-episode accumulation
        self.steps_without_demonstration = 0
        # Start each episode with clean cache to avoid cross-episode pollution:
        # Do not allow "previous frame pose" from last game to affect current game's first frame unwrapping result.
        self._prev_ee_quat_wxyz = None
        self._prev_ee_rpy_xyz = None

        super().reset(**kwargs)
        self.episode_success = False
        # Generate demonstration trajectory batch
        demo_batch = self.get_demonstration_trajectory()

        # Select gripper and initial action based on environment: PatternLock/RouteStick use stick and require online generated action
        if self.unwrapped.spec.id == "PatternLock" or self.unwrapped.spec.id == "RouteStick":
            gripper = "stick"
        else:
            gripper = None
        if self.unwrapped.spec.id == "PatternLock" or self.unwrapped.spec.id == "RouteStick":
            action = self.unwrapped.swing_qpos  # These two types of environments require online generated initial action
        else:
            action = reset_panda.get_reset_panda_param("action", gripper=gripper)

        # Execute one initial step, append to demonstration trajectory batch
        init_batch = self._step_batch(action)
        merged_batch = planner_denseStep.concat_step_batches([demo_batch, init_batch])
        merged_batch = self._filter_no_record_from_step_batch(merged_batch)
        self.demonstration_data = merged_batch
        
        # Unpack the batch to return only obs and info, but keep the full batch in self.demonstration_data
        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = merged_batch
        info_flat = self._flatten_info_batch(info_batch)
        return obs_batch, info_flat

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

        subgoal_list = info_batch.get("simple_subgoal_online")
        if not isinstance(subgoal_list, list) or len(subgoal_list) != n:
            return batch

        keep_indices = [
            idx for idx, subgoal in enumerate(subgoal_list)
            if str(subgoal).strip() != "NO RECORD"
        ]
        if len(keep_indices) == n:
            return batch
        # Defensive fallback: Avoid accessing [-1] on empty batch after filtering.
        if len(keep_indices) == 0:
            return batch

        index_reward = torch.as_tensor(keep_indices, dtype=torch.long, device=reward_batch.device)
        index_terminated = torch.as_tensor(keep_indices, dtype=torch.long, device=terminated_batch.device)
        index_truncated = torch.as_tensor(keep_indices, dtype=torch.long, device=truncated_batch.device)

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
        language_goal = task_goal.get_language_goal(self.env, self.unwrapped.spec.id)

        base_obs = obs if isinstance(obs, dict) else {}
        env_id = self.unwrapped.spec.id
        dummy_target = {"obj": None, "name": None, "seg_id": None, "click_point": None, "centroid_point": None}
        raw_options = get_vqa_options(self, None, dummy_target, env_id)
        available_options = [
            {"label": opt.get("label"), "action": opt.get("action", "Unknown"), "need_parameter": bool(opt.get("available"))}
            for opt in raw_options
        ]

        # Extract frames, state, velocity etc directly from obs (no longer read from self.frames etc list)
        image = obs['sensor_data']['base_camera']['rgb'][0]
        wrist_image = obs['sensor_data']['hand_camera']['rgb'][0]
        state = self.agent.robot.qpos
        # end_effector_velocity = self.agent.robot.links[9].get_linear_velocity()[0], self.agent.robot.links[9].get_angular_velocity()[0]
        subgoal_text = getattr(self, 'current_task_name', 'Unknown')
        grounded_subgoal = self.current_subgoal_segment_filled

        base_camera_depth = obs["sensor_data"]["base_camera"]["depth"][0]
        base_camera_segmentation = obs["sensor_data"]["base_camera"]["segmentation"][0]
        wrist_camera_depth = obs["sensor_data"]["hand_camera"]["depth"][0]
        base_camera_extrinsic_opencv = obs["sensor_param"]["base_camera"]["extrinsic_cv"]
        base_camera_intrinsic_opencv = obs["sensor_param"]["base_camera"]["intrinsic_cv"]
        base_camera_cam2world_opengl = obs["sensor_param"]["base_camera"]["cam2world_gl"]
        wrist_camera_extrinsic_opencv = obs["sensor_param"]["hand_camera"]["extrinsic_cv"]
        wrist_camera_intrinsic_opencv = obs["sensor_param"]["hand_camera"]["intrinsic_cv"]
        wrist_camera_cam2world_opengl = obs["sensor_param"]["hand_camera"]["cam2world_gl"]
        # Output end-effector pose as dict, containing pose/quat/rpy representations; also update continuousness cache.
        robot_endeffector_pose, self._prev_ee_quat_wxyz, self._prev_ee_rpy_xyz = \
            build_endeffector_pose_dict(
                self.agent.tcp.pose.p,
                self.agent.tcp.pose.q,
                self._prev_ee_quat_wxyz,
                self._prev_ee_rpy_xyz,
            )

        # Extract gripper state from the last 2 dims of joint positions
        state_flat = state.cpu().numpy().flatten() if hasattr(state, 'cpu') else np.asarray(state).flatten()
        gripper_state = state_flat[-2:]
        # Only keep first 7 joint dims for joint_state_list
        joint_state = state_flat[:7]

        new_obs = {
            'maniskill_obs': base_obs,
            'front_rgb_list': image,
            'wrist_rgb_list': wrist_image,
            'joint_state_list': joint_state,
            # 'velocity': end_effector_velocity,
            'front_depth_list': base_camera_depth,
            # 'front_camera_segmentation': base_camera_segmentation,
            'wrist_depth_list': wrist_camera_depth,
            # 'front_camera_cam2world_opengl': base_camera_cam2world_opengl,
            # 'wrist_camera_cam2world_opengl': wrist_camera_cam2world_opengl,
            'end_effector_pose_raw': robot_endeffector_pose,
            'eef_state_list': (
                robot_endeffector_pose['pose'].detach().cpu().flatten().tolist()[:3]
                + robot_endeffector_pose['rpy'].detach().cpu().flatten().tolist()[:3]
            ),
            'gripper_state_list': gripper_state,
            'front_camera_extrinsic_list': base_camera_extrinsic_opencv,
            'wrist_camera_extrinsic_list': wrist_camera_extrinsic_opencv,
        }
        new_info = {
            **info,
            'simple_subgoal_online': subgoal_text,
            'grounded_subgoal_online': grounded_subgoal,
            'available_options': available_options,
            'task_goal': language_goal,
            'front_camera_intrinsic': base_camera_intrinsic_opencv,
            'wrist_camera_intrinsic': wrist_camera_intrinsic_opencv,
        }
        return new_obs, new_info

    def _add_red_border(self, frame, border_width=5):
        """Draw red border on four sides of image, used to highlight demonstration frames (currently not used for video saving)."""
        frame_with_border = frame.copy()
        frame_with_border[:border_width, :] = [255, 0, 0]
        frame_with_border[-border_width:, :] = [255, 0, 0]
        frame_with_border[:, :border_width] = [255, 0, 0]
        frame_with_border[:, -border_width:] = [255, 0, 0]
        return frame_with_border

    TEXT_AREA_HEIGHT = 80  # Fixed font black border height

    def _add_text_to_frame(self, frame, text, position='top_right'):
        """Append black text area above frame and stitch, supporting multi-line and auto-wrap. Black border height fixed to TEXT_AREA_HEIGHT."""
        if text is None:
            text = ""
        text_area_height = self.TEXT_AREA_HEIGHT
        if not text and not (isinstance(text, (list, tuple)) and any(text)):
            text_area = np.zeros((text_area_height, frame.shape[1], 3), dtype=np.uint8)
            return np.vstack((text_area, frame))

        if isinstance(text, str):
            text_list = [text]
        else:
            text_list = list(text) if text else []

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        thickness = 1
        max_width = max(1, frame.shape[1] - 20)

        lines = []
        for text_item in text_list:
            if text_item is None:
                continue
            text_item = str(text_item).strip()
            if not text_item:
                continue
            words = text_item.replace(',', ' ').split()
            if not words:
                continue
            current_line = words[0]
            for word in words[1:]:
                test_line = f"{current_line} {word}"
                (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
                if text_width <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            lines.append(current_line)

        if not lines:
            text_area = np.zeros((text_area_height, frame.shape[1], 3), dtype=np.uint8)
            return np.vstack((text_area, frame))

        line_height = 20
        text_area = np.zeros((text_area_height, frame.shape[1], 3), dtype=np.uint8)
        text_area[:] = (0, 0, 0)
        max_visible_lines = (text_area_height - 15) // line_height
        for i, line in enumerate(lines[:max_visible_lines]):
            y_position = 15 + i * line_height
            cv2.putText(text_area, line, (10, y_position), font, font_scale, (255, 255, 255), thickness)

        return np.vstack((text_area, frame))

    def save_frame_as_image(self, output_path: Union[str, Path], frame: np.ndarray, text=None):
        """
        Overlay single frame with text and save as image.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined = self._add_text_to_frame(np.asarray(frame).copy(), text)
        if combined.ndim == 2:
            combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)
        scale = 2
        out_h, out_w = combined.shape[0] * scale, combined.shape[1] * scale
        combined = cv2.resize(combined, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        imageio.imwrite(str(output_path), combined)

    def _compute_segmentation_and_fill_subgoal(
        self,
        obs: Dict,
    ) -> Tuple[Optional[str], bool]:
        """
        Parse base camera segmentation from observation, build object ID mapping cared by current task, calculate target object pixel center on image, and replace placeholders (like <target>) in current subgoal text with specific coordinates <y, x>.
        Support latching: Result is reused after successful fill for same subgoal; latch cleared when subgoal changes.

        Args:
            obs: Current step observation, must contain sensor_data.base_camera.segmentation (and optional rgb etc).

        Returns:
            filled_text: Subgoal text after placeholder replacement; consistent with current_subgoal_segment if no subgoal or no replacement.
            failed_match: True if text has placeholder but no valid fill in this frame and no latch (used for saving failed frames etc).
        """
        current_subgoal_segment = getattr(self.unwrapped, 'current_subgoal_segment', None)
        current_task_name = getattr(self, 'current_task_name', 'Unknown')

        # ---------- Parse base camera segmentation from obs, and build active_segments / segment_ids_by_index / vis_obj_id_list ----------
        segmentation = None
        try:
            segmentation = obs['sensor_data']['base_camera']['segmentation']
        except Exception:
            segmentation = None

        segmentation_2d = None
        active_segments = []
        segment_ids_by_index = {}
        vis_obj_id_list = []

        if segmentation is not None:
            if hasattr(segmentation, "cpu"):
                segmentation = segmentation.cpu().numpy()
            segmentation = np.asarray(segmentation)
            if segmentation.ndim > 2:
                segmentation = segmentation[0]
            segmentation_2d = segmentation.squeeze()

            # Segmentation object (current_segment) and ID mapping cared by current task, used for subsequent center calculation and placeholder filling
            current_segment = getattr(self, "current_segment", None)
            if isinstance(current_segment, (list, tuple)):
                active_segments = list(current_segment)
            elif current_segment is None:
                active_segments = []
            else:
                active_segments = [current_segment]

            # Establish "Object -> Segmentation ID" mapping by active_segments index, for calculating center segment by segment
            segment_ids_by_index = {idx: [] for idx in range(len(active_segments))}
            segmentation_id_map = getattr(self, "segmentation_id_map", None)
            if isinstance(segmentation_id_map, dict):
                for obj_id, obj in sorted(segmentation_id_map.items()):
                    if active_segments:
                        for idx, target in enumerate(active_segments):
                            if obj is target:
                                vis_obj_id_list.append(obj_id)
                                segment_ids_by_index[idx].append(obj_id)
                                break
                    # Set workspace table to black in color map, for distinction in segmentation visualization
                    if getattr(obj, "name", None) == 'table-workspace':
                        self.color_map[obj_id] = [0, 0, 0]

        # No fill when no segmentation data, directly return original text and mismatch
        if segmentation_2d is None:
            return (current_subgoal_segment, False)

        def center_from_ids(segmentation_mask: np.ndarray, ids: List):
            """
            Calculate pixel center (centroid) of the object on image based on segmentation mask and object ID list.
            Return (center [y, x] or None, no_object_flag_this).
            no_object_flag_this is True when ids is not empty but no corresponding pixels in mask.
            """
            if not ids:
                return None, False
            mask = np.isin(segmentation_mask, ids)
            if not np.any(mask):
                return None, True
            coords = np.argwhere(mask)
            if coords.size == 0:
                return None, True
            center_y = int(coords[:, 0].mean())
            center_x = int(coords[:, 1].mean())
            return [center_y, center_x], False

        # Clear latch when subgoal changes, subsequent calculation will use current frame and may re-latch
        if current_subgoal_segment != self.last_subgoal_segment:
            self.last_subgoal_segment = current_subgoal_segment
            self.latched_replacements = None

        # Calculate pixel center segment by segment (or single center for whole image) according to objects cared by current task
        segment_centers = []
        no_object_flag = False
        if active_segments:
            for idx in range(len(active_segments)):
                center, no_obj = center_from_ids(segmentation_2d, segment_ids_by_index.get(idx, []))
                segment_centers.append(center)
                no_object_flag = no_object_flag or no_obj
        else:
            center, no_obj = center_from_ids(segmentation_2d, vis_obj_id_list)
            segment_centers.append(center)
            no_object_flag = no_obj

        # No placeholder replacement needed when no subgoal text, return directly
        if not current_subgoal_segment:
            return (current_subgoal_segment, False)

        # Match all placeholders (format <...>) using regex
        placeholder_pattern = re.compile(r'<[^>]*>')
        placeholders = list(placeholder_pattern.finditer(current_subgoal_segment))
        placeholder_count = len(placeholders)

        final_replacements = None
        missing_placeholder = False

        # Prioritize latched replacement results; generate replacement string using current frame center when no latch
        if self.latched_replacements is not None:
            final_replacements = self.latched_replacements
        else:
            # Format each center as "<y, x>" string, None for undetected center
            normalized_centers = []
            for center in segment_centers:
                if center is None:
                    normalized_centers.append(None)
                    continue
                center_y, center_x = center
                normalized_centers.append(f'<{center_y}, {center_x}>')

            if placeholder_count > 0 and normalized_centers:
                replacements = normalized_centers.copy()
                # If only one center but multiple placeholders, reuse that center; if insufficient centers, pad with None
                if len(replacements) == 1 and placeholder_count > 1:
                    replacements = replacements * placeholder_count
                elif len(replacements) < placeholder_count:
                    replacements.extend([None] * (placeholder_count - len(replacements)))
                # Latch only when all placeholders can be replaced by non-None, to avoid latching incomplete results
                temp_missing_placeholder = any(r is None for r in replacements)
                if not temp_missing_placeholder:
                    self.latched_replacements = replacements
                final_replacements = replacements

        # Apply replacement: Assemble final text by placeholder order, degrade to current_task_name as whole sentence if any placeholder misses replacement
        if final_replacements and placeholder_count > 0:
            new_text_parts = []
            last_idx = 0
            for idx, match in enumerate(placeholders):
                new_text_parts.append(current_subgoal_segment[last_idx:match.start()])
                replacement_text = final_replacements[idx] if idx < len(final_replacements) else None
                if replacement_text is None:
                    missing_placeholder = True
                else:
                    new_text_parts.append(replacement_text)
                last_idx = match.end()
            new_text_parts.append(current_subgoal_segment[last_idx:])
            filled_text = current_task_name if missing_placeholder else ''.join(new_text_parts)
            # Regard as match failure when no latch and (valid replacement not given in this frame or still missing items)
            failed_match = self.latched_replacements is None and (final_replacements is None or missing_placeholder)
            return (filled_text, failed_match)
        else:
            # Also record as match failure when there are placeholders but no replacement result and no latch
            failed_match = placeholder_count > 0 and self.latched_replacements is None
            return (current_subgoal_segment, failed_match)

    _STICK_ENV_IDS = ("PatternLock", "RouteStick")

    def _normalize_action_for_env_step(self, action) -> np.ndarray:
        """
        Normalize external action to the dimensionality required by the wrapped env.step.
        - PatternLock/RouteStick: accept len>=7 and pass first 7 dims.
        - Other envs: accept len>=8 and pass first 8 dims.
        """
        env_spec = getattr(self.unwrapped, "spec", None)
        env_id = getattr(env_spec, "id", "<unknown_env>")
        action_arr = np.asarray(action, dtype=np.float64).flatten()
        if env_id in self._STICK_ENV_IDS:
            if action_arr.size < 7:
                raise ValueError(f"[{env_id}] action must have at least 7 elements, got {action_arr.size}")
            return action_arr[:7]
        if action_arr.size < 8:
            raise ValueError(f"[{env_id}] action must have at least 8 elements, got {action_arr.size}")
        return action_arr[:8]

    @staticmethod
    def _flatten_info_batch(info_batch: dict) -> dict:
        """Convert columnar info dict-of-lists to flat dict by taking the last value of each key."""
        return {k: v[-1] if isinstance(v, list) and v else v for k, v in info_batch.items()}

    def _step_batch(self, action):
        """Internal step returning full batch format (dict-of-lists for both obs and info).

        Used by reset() and other internal callers that need batch-compatible output
        for concat_step_batches.
        """
        normalized_action = self._normalize_action_for_env_step(action)
        obs, reward, terminated, truncated, info = super().step(normalized_action)

        # ---------- Subgoal segmentation and placeholder filling: Internally parse segmentation from obs, calculate center, fill placeholders ----------
        filled_text, failed_match = self._compute_segmentation_and_fill_subgoal(obs)
        current_subgoal_segment = getattr(self.unwrapped, 'current_subgoal_segment', None)
        self.current_subgoal_segment_filled = filled_text if filled_text is not None else current_subgoal_segment

        # ---------- Non-demonstration step count: Truncate if exceeding limit ----------
        if self.current_task_demonstration == False:
            self.steps_without_demonstration += 1
            if self.steps_without_demonstration >= self.max_steps_without_demonstration:
                truncated = torch.tensor([True])

        # ---------- Update episode_success based on terminated and info["success"] ----------
        if terminated.any():
            if info.get("success") == torch.tensor([True]) or (isinstance(info.get("success"), torch.Tensor) and info.get("success").item()):
                self.episode_success = True
                # print("Episode success detected, data will be saved")
            else:
                self.episode_success = False
                # print("Episode failed, data will be discarded")

        # ---------- Execute extra step at termination, so last frame is also recorded (action same as previous step) ----------
        if terminated.any() and not self._doing_extra_step:
            # Save RPY continuousness cache before recursive extra step.
            # Reason: Inner extra step should not change previous frame baseline of "current outer return step",
            # otherwise it will pollute continuousness results on outer timeline.
            cached_prev_quat = None if self._prev_ee_quat_wxyz is None else self._prev_ee_quat_wxyz.detach().clone()
            cached_prev_rpy = None if self._prev_ee_rpy_xyz is None else self._prev_ee_rpy_xyz.detach().clone()
            self._doing_extra_step = True
            try:
                self._step_batch(normalized_action)
            finally:
                self._doing_extra_step = False
                # Restore outer cache, ensuring "extra step only used for recording frames", not interfering with outer continuousness state.
                self._prev_ee_quat_wxyz = cached_prev_quat
                self._prev_ee_rpy_xyz = cached_prev_rpy

        obs, info = self._augment_obs_and_info(obs, info, normalized_action)

        # Compute status field from terminated/truncated/success
        raw_success = info.get("success")
        is_success = (isinstance(raw_success, torch.Tensor) and raw_success.item()) or raw_success is True
        if is_success:
            info["status"] = "success"
        elif terminated.any():
            info["status"] = "fail"
        elif truncated.any():
            info["status"] = "timeout"
        else:
            info["status"] = "ongoing"

        return planner_denseStep.to_step_batch([(obs, reward, terminated, truncated, info)])

    def step(self, action):
        """Execute one step and return (obs_batch, reward, terminated, truncated, info).

        obs_batch is dict[str, list]; info is a flat dict (last values only).
        """
        batch = self._step_batch(action)
        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = batch
        info_flat = self._flatten_info_batch(info_batch)
        return (obs_batch, reward_batch[-1], terminated_batch[-1], truncated_batch[-1], info_flat)

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
            from ..robomme_env.utils.planner_fail_safe import (
                FailAwarePandaArmMotionPlanningSolver,
                FailAwarePandaStickMotionPlanningSolver,
                ScrewPlanFailure,
            )
        except Exception as exc:
            logger.debug(f"[DemonstrationWrapper] Warning: failed to import planner_fail_safe, fallback to base planners: {exc}")
            FailAwarePandaArmMotionPlanningSolver = PandaArmMotionPlanningSolver
            FailAwarePandaStickMotionPlanningSolver = PandaStickMotionPlanningSolver
            ScrewPlanFailure = RuntimeError

        # Select motion planner by environment: PatternLock/RouteStick use stick planner, others use arm planner
        if self.unwrapped.spec.id == "PatternLock" or self.unwrapped.spec.id == "RouteStick":
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
            logger.debug("[DemonstrationWrapper] screw->RRT* planning exhausted; return -1")
            return -1

        planner.move_to_pose_with_screw = _move_to_pose_with_screw_then_rrt_retry
        tasks = getattr(self, 'task_list', [])
        self.task_list_length = len(tasks)
        logger.debug(f"Task list length: {self.task_list_length}")

        demonstration_tasks = [task for task in tasks if task.get("demonstration", False)]
        self.non_demonstration_task_length = len(tasks) - len(demonstration_tasks)
        logger.debug(f"Non-demonstration task length: {self.non_demonstration_task_length}")

        all_collected_steps = []

        # Iterate and execute each demonstration task: Set demonstration_record_traj=True, call task's solve(planner)
        # Wrap entire solve with _collect_dense_steps, monkey-patch planner.env.step,
        # to collect all env.step calls (including follow_path, direct env.step etc underlying paths)
        for idx, task_entry in enumerate(demonstration_tasks):
            self.unwrapped.demonstration_record_traj = True
            self._current_demo_task_screw_failed = False
            task_name = task_entry.get("name", f"Task {idx}")
            logger.debug(f"Executing task {idx+1}/{len(demonstration_tasks)}: {task_name}")

            solve_callable = task_entry.get("solve")
            if not callable(solve_callable):
                raise ValueError(f"Task '{task_name}' must supply a callable 'solve'.")

            self.evaluate(solve_complete_eval=True)

            def _solve_task_without_hard_fail():
                # Avoid solve returning -1 causing _collect_dense_steps to discard collected steps of this task
                try:
                    solve_result = solve_callable(self, planner)
                except ScrewPlanFailure as exc:
                    self._current_demo_task_screw_failed = True
                    logger.debug(f"[DemonstrationWrapper] task '{task_name}' screw failure: {exc}")
                    return None
                if isinstance(solve_result, int) and solve_result == -1:
                    self._current_demo_task_screw_failed = True
                    logger.debug(f"[DemonstrationWrapper] task '{task_name}' returned -1 after screw->RRT* retries")
                    return None
                return solve_result

            task_steps = planner_denseStep._collect_dense_steps(
                planner,
                _solve_task_without_hard_fail,
            )
            if task_steps == -1:
                # Theoretically should not hit (_solve_task_without_hard_fail has swallowed -1)
                logger.debug(f"[DemonstrationWrapper] task '{task_name}' returned -1 from collector; continuing")
            else:
                all_collected_steps.extend(task_steps)

            if self._current_demo_task_screw_failed:
                logger.debug(f"[DemonstrationWrapper] task '{task_name}' marked failed after screw->RRT* retries; continuing")
            self.evaluate(solve_complete_eval=True)

        self.unwrapped.demonstration_record_traj = False  # Demonstration ends, subsequent steps perform subgoal judgment normally
        return planner_denseStep.to_step_batch(all_collected_steps)
