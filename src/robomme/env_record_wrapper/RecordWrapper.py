import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

import gymnasium as gym
import h5py
import numpy as np
import sapien
import sapien.physx as physx
import torch
import cv2
import colorsys

from mani_skill import get_commit_info
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
    PandaStickMotionPlanningSolver,
)
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils, sapien_utils
from mani_skill.utils.io_utils import dump_json
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.structs.types import Array
from mani_skill.utils.visualization.misc import (
    images_to_video,
    put_info_on_image,
    tile_images,
)
from mani_skill.utils.wrappers import CPUGymWrapper
import imageio
from ..robomme_env.utils import task_goal
from ..robomme_env.utils.segmentation_utils import (
    process_segmentation,
    create_segmentation_visuals,
)
from ..robomme_env.utils.rpy_util import build_endeffector_pose_dict

class FailsafeTimeout(RuntimeError):
    """Exception raised when Robomme failsafe terminates episode early."""
    pass


class RobommeRecordWrapper(gym.Wrapper):
    """
    Robomme record wrapper.
    
    Main functions:
    1. Record Robomme rollout data (obs, action, state, etc) to HDF5 file.
    2. Generate composite video including base/wrist camera views, segmentation masks, and visualization results.
    3. Handle segmentation logic, including object recognition and center calculation.
    """
    def __init__(self, env,
     Robomme_dataset=None,Robomme_env=None,Robomme_episode=None,Robomme_seed=None,save_video=False):
        # Initialize parent first to ensure self.env exists
        super().__init__(env)
        self.unwrapped.use_demonstrationwrapper=False


        # Save config as attribute to avoid triggering __getattr__

        self.Robomme_dataset = Robomme_dataset
        self.Robomme_episode = Robomme_episode
        self.Robomme_env = Robomme_env
        self.Robomme_seed = Robomme_seed
        self.save_video = save_video



        # Track if failsafe triggered to avoid repeated exceptions
        self._failsafe_triggered = False

        # New: Buffer for temporary data storage, write in batch before write()
        # Avoid IO operation every step to improve efficiency
        self.buffer = []
        self.episode_success = False

        # Cache for subgoal segmentation tracking
        self.previous_subgoal_segment = None
        self.current_subgoal_segment_filled = None
        self.segmentation_points = []  # Cache segmentation center points
        self.previous_subgoal_segment_online = None
        self.current_subgoal_segment_online_filled = None
        self.segmentation_points_online = []  # Cache online segmentation target points

        # Video buffer
        self.video_frames = []  # Store combined video frames
        self.no_object_video_frames = []  # Save separately when target missing in video frame, for debugging

        # End-effector pose continuousness cache (wxyz / XYZ-RPY), lifecycle limited to single episode
        self._prev_ee_quat_wxyz = None
        self._prev_ee_rpy_xyz = None

        self.h5_file = None

        if not self.Robomme_dataset:
            raise ValueError("RobommeRecord=True requires Robomme_dataset path")

        # Create HDF5 folder; allow user to pass single h5 file or parent directory, automatically deduce output path
        base_path = Path(self.Robomme_dataset).resolve()
        if base_path.suffix == '.h5' or base_path.suffix == '.hdf5':
            # If file path provided, use its parent directory
            self.output_root = base_path.parent
            hdf5_folder_name = base_path.stem + "_hdf5_files"
        else:
            # If directory path provided, use directly
            self.output_root = base_path
            hdf5_folder_name = "hdf5_files"

        # Create folder to save HDF5 file
        self.hdf5_dir = self.output_root / hdf5_folder_name
        self.hdf5_dir.mkdir(parents=True, exist_ok=True)

        # HDF5 file saved in new created folder
        h5_filename = f"{self.Robomme_env}_ep{self.Robomme_episode}_seed{self.Robomme_seed}.h5"
        self.dataset_path = self.hdf5_dir / h5_filename

        # Generate unique filename by env/episode/seed convention for batch analysis
        # Open in 'a' mode, delete and recreate if file corrupted
        try:
            self.h5_file = h5py.File(self.dataset_path, "a")
        except OSError as exc:
            if self.dataset_path.exists():
                # Delete truncated/corrupted file and recreate a clean one
                print(f"Failed to open existing dataset ({exc}); recreating file.")
                self.dataset_path.unlink()
                self.h5_file = h5py.File(self.dataset_path, "w")
            else:
                raise
        print(f"Recording data to {self.dataset_path}")
        print(f"HDF5 files will be saved in folder: {self.hdf5_dir}")

        # Color lookup table generated once at initialization, avoid repeated construction in step
        # Used to assign fixed color to different segmentation IDs
        def generate_color_map(n=100, s_min=0.70, s_max=0.95, v_min=0.78, v_max=0.95):
            """
            Generate 1..n color dictionary, value [R,G,B] (0-255).
            - Hue uses golden ratio step to avoid clustering
            - Saturation/Value fluctuates in small cycles to enhance separability
            """
            phi = 0.6180339887498948  # Golden ratio step
            color_map = {}
            for i in range(1, n + 1):
                h = (i * phi) % 1.0
                s = s_min + (s_max - s_min) * ((i % 7) / 6)        # 7-step cycle saturation
                v = v_min + (v_max - v_min) * (((i * 3) % 5) / 4)  # 5-step cycle value
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                color_map[i] = [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))]
            return color_map

        # Usage
        color_map = generate_color_map(10000)
        #color_map[16] = [0, 0, 0]  # Fix 16 as black (table)
        self.color_map=color_map

    def _add_red_border(self, frame, border_width=10):
        """Add red border to image, usually used to mark Demonstration phase."""
        frame_with_border = frame.copy()
        # Add red border (RGB: 255, 0, 0)
        frame_with_border[:border_width, :] = [255, 0, 0]  # Top
        frame_with_border[-border_width:, :] = [255, 0, 0]  # Bottom
        frame_with_border[:, :border_width] = [255, 0, 0]  # Left
        frame_with_border[:, -border_width:] = [255, 0, 0]  # Right
        return frame_with_border

    def _add_text_to_frame(self, frame, text, position='top_right'):
        """
        Create filled text area above frame, auto-wrapping if needed.
        
        Args:
            frame: Image frame to add text to
            text: Single string or list of strings. Each item in list will display on separate line.
            position: Position argument (retained for compatibility, actually always stacked on top)
        """
        if not text:
            return frame

        # Convert single string to list for uniform processing
        if isinstance(text, str):
            text_list = [text]
        else:
            text_list = text

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        max_width = max(1, frame.shape[1] - 20)

        lines = []

        # Process each text item separately
        for text_item in text_list:
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
            return frame

        line_height = 20
        text_area_height = max(50, len(lines) * line_height + 10)
        text_area = np.zeros((text_area_height, frame.shape[1], 3), dtype=np.uint8)

        for i, line in enumerate(lines):
            y_position = 15 + i * line_height
            cv2.putText(text_area, line, (10, y_position), font, font_scale, (255, 255, 255), thickness)

        # Stack text area above frame; position argument retained for compatibility.
        return np.vstack((text_area, frame))

    def _video_should_record(self, current_task_name):
        """Determine if current step needs video pipeline execution."""
        return self.save_video and current_task_name != "NO RECORD"

    def _video_prepare_step_frames(
        self,
        base_frame,
        wrist_frame,
        segmentation,
        segmentation_result,
        segmentation_result_online,
    ):
        """
        Prepare material for single step video construction, returning base images for planner/online rows.
        """
        # Use deepcopy to avoid modifying original frames that will be saved to HDF5
        base_camera_frame_for_video = copy.deepcopy(base_frame)
        wrist_camera_frame_for_video = copy.deepcopy(wrist_frame)
        segmentation_for_video = copy.deepcopy(segmentation)
        segmentation_result_for_video = copy.deepcopy(segmentation_result)
        segmentation_result_online_for_video = copy.deepcopy(segmentation_result_online)

        # Resize wrist camera image to match base camera
        if base_camera_frame_for_video.shape[:2] != wrist_camera_frame_for_video.shape[:2]:
            wrist_camera_frame_for_video = cv2.resize(
                wrist_camera_frame_for_video,
                (base_camera_frame_for_video.shape[1], base_camera_frame_for_video.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # Generate segmentation visualization image (change color) and target image with red dot
        (
            segmentation_vis,
            segmentation_result_vis,
            target_for_video,
        ) = create_segmentation_visuals(
            segmentation_for_video,
            segmentation_result_for_video,
            base_camera_frame_for_video,
            self.color_map,
            self.segmentation_points,
        )

        (
            segmentation_vis_online,
            segmentation_result_vis_online,
            target_for_video_online,
        ) = create_segmentation_visuals(
            segmentation_for_video,
            segmentation_result_online_for_video,
            base_camera_frame_for_video,
            self.color_map,
            self.segmentation_points_online,
        )

        # Final video frame structure: base | wrist | original segmentation | filtered segmentation | base+red dot
        combined = np.hstack(
            [
                base_camera_frame_for_video,
                wrist_camera_frame_for_video,
                segmentation_vis,
                segmentation_result_vis,
                target_for_video,
            ]
        )
        combined_online = np.hstack(
            [
                base_camera_frame_for_video,
                wrist_camera_frame_for_video,
                segmentation_vis_online,
                segmentation_result_vis_online,
                target_for_video_online,
            ]
        )

        return {
            "combined": combined,
            "combined_online": combined_online,
        }

    def _video_compose_planner_online_rows(
        self,
        prepared,
        subgoal_text,
        grounded_text,
        subgoal_online_text,
        grounded_online_text,
    ):
        """Stitch and overlay planner / online text rows."""
        combined = prepared["combined"]
        combined_online = prepared["combined_online"]

        # Add subgoal_text and grounded_subgoal text for first row (PLANNER view)
        combined = self._add_text_to_frame(
            combined,
            ["PLANNER:", subgoal_text, grounded_text],
            position="top_right",
        )

        # Add ONLINE: marker and online text for second row (ONLINE view)
        combined_online = self._add_text_to_frame(
            combined_online,
            ["ONLINE:", subgoal_online_text, grounded_online_text],
            position="top_right",
        )

        # Stack two video streams vertically
        return np.vstack([combined, combined_online])

    def _video_apply_overlays(self, frame, is_demonstration, language_goal):
        """Apply demonstration red border and language goal text."""
        # If demonstration phase, add red border to entire frame
        if is_demonstration:
            frame = self._add_red_border(frame)

        return self._add_text_to_frame(frame, [language_goal], position="top_right")

    def _video_append_step_frame(self, frame, no_object_flag):
        """Append single step video frame to corresponding buffer."""
        self.video_frames.append(frame)
        if no_object_flag == True:
            self.no_object_video_frames.append(frame)

    def _video_build_filename_parts(self, language_goal, difficulty):
        """Build suffix in video filename."""
        sanitized_goal = (
            language_goal.replace(" ", "_").replace("/", "_")
            if language_goal
            else "no_goal"
        )
        difficulty_tag = (
            str(difficulty).replace(" ", "_").replace("/", "_")
            if difficulty
            else None
        )
        filename_suffix = sanitized_goal
        if difficulty_tag:
            filename_suffix = (
                f"{difficulty_tag}_{filename_suffix}"
                if filename_suffix
                else difficulty_tag
            )
        return {"filename_suffix": filename_suffix}

    def _video_write_mp4(self, frames, output_path):
        """Write mp4 with unified parameters."""
        with imageio.get_writer(
            output_path.as_posix(), fps=30, codec="libx264", quality=8
        ) as writer:
            for frame in frames:
                writer.append_data(frame)

    def _video_flush_episode_files(self, success, video_prefix, filename_suffix):
        """Write current episode video (main video and no-object video)."""
        if not self.save_video:
            return

        if len(self.video_frames) == 0 and len(self.no_object_video_frames) == 0:
            return

        videos_dir = self.output_root / "videos"

        if len(self.video_frames) > 0:
            try:
                videos_dir.mkdir(parents=True, exist_ok=True)
                if success:
                    combined_video_path = videos_dir / f"{video_prefix}_{filename_suffix}.mp4"
                    self._video_write_mp4(self.video_frames, combined_video_path)
                    print(f"Saved combined video to {combined_video_path}")
                else:
                    combined_video_path = (
                        videos_dir / f"FAILED_{video_prefix}_{filename_suffix}.mp4"
                    )
                    self._video_write_mp4(self.video_frames, combined_video_path)
                    print(f"Saved failed episode video to {combined_video_path}")
            except Exception as e:
                if success:
                    print(
                        f"Warning: Failed to save combined video for episode {self.Robomme_episode}: {e}"
                    )
                else:
                    print(
                        f"Warning: Failed to save failed episode video for episode {self.Robomme_episode}: {e}"
                    )

        if len(self.no_object_video_frames) > 0:
            try:
                videos_dir.mkdir(parents=True, exist_ok=True)
                if success:
                    no_object_video_path = (
                        videos_dir
                        / f"success_NO_OBJECT_{video_prefix}_{filename_suffix}.mp4"
                    )
                    self._video_write_mp4(
                        self.no_object_video_frames, no_object_video_path
                    )
                    print(f"Saved no-object video to {no_object_video_path}")
                else:
                    no_object_video_path = (
                        videos_dir
                        / f"FAILED_NO_OBJECT_{video_prefix}_{filename_suffix}.mp4"
                    )
                    self._video_write_mp4(
                        self.no_object_video_frames, no_object_video_path
                    )
                    print(f"Saved failed no-object video to {no_object_video_path}")
            except Exception as e:
                if success:
                    print(
                        f"Warning: Failed to save no-object video for episode {self.Robomme_episode}: {e}"
                    )
                else:
                    print(
                        f"Warning: Failed to save failed no-object video for episode {self.Robomme_episode}: {e}"
                    )

    def _init_fk_planner(self):
        """Initialize mplib FK planner after env.reset().

        Stores pinocchio_model, ee_link_idx and robot_base_pose for
        forward-kinematics computation in _joint_action_to_ee_pose_dict().
        Sets self._fk_available = False on failure so callers can fall back.
        """
        try:
            _STICK_IDS = ("PatternLock", "RouteStick")
            env_id = getattr(getattr(self.unwrapped, "spec", None), "id", None) or self.Robomme_env
            use_stick = env_id in _STICK_IDS

            solver_cls = PandaStickMotionPlanningSolver if use_stick else PandaArmMotionPlanningSolver
            solver_kwargs = dict(
                debug=False,
                vis=False,
                base_pose=self.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
            )
            if use_stick:
                solver_kwargs["joint_vel_limits"] = 0.3
            solver = solver_cls(self, **solver_kwargs)
            self._mplib_planner = solver.planner
            self._ee_link_idx = self._mplib_planner.link_name_2_idx[
                self._mplib_planner.move_group
            ]
            self._robot_base_pose = self.unwrapped.agent.robot.pose
            self._fk_qpos_size = len(self._mplib_planner.user_joint_names)
            self._fk_available = True
        except Exception as exc:
            print(f"[RecordWrapper] FK planner init failed, eef_action_raw/eef_action "
                  f"will be zeros: {exc}")
            self._mplib_planner = None
            self._ee_link_idx = None
            self._robot_base_pose = None
            self._fk_available = False

    def _joint_action_to_ee_pose_dict(self, action):
        """Compute end-effector pose dict from joint_action via forward kinematics.

        Uses the same build_endeffector_pose_dict pipeline as eef_state_raw
        (normalization, sign alignment, RPY unwrapping) but with independent
        prev-frame caches so that state and action continuity do not interfere.

        Returns None if FK is unavailable or the action is invalid.
        """
        if not self._fk_available or action is None:
            return None

        try:
            if isinstance(action, torch.Tensor):
                action_np = action.detach().cpu().numpy()
            else:
                action_np = np.asarray(action)
            action_np = action_np.astype(np.float64).flatten()

            arm_qpos = action_np[:7]
            if self._fk_qpos_size == 7:
                # stick 机器人：pinocchio 只有 7 维，不拼 finger
                full_qpos = arm_qpos
            else:
                # 标准 panda：pinocchio 有 9 维，拼两个 finger
                gripper = float(action_np[7]) if action_np.size > 7 else -1.0
                finger_pos = max(gripper, 0.0) if gripper >= 0 else 0.04
                full_qpos = np.concatenate([arm_qpos, [finger_pos, finger_pos]])

            pmodel = self._mplib_planner.pinocchio_model
            pmodel.compute_forward_kinematics(full_qpos)
            fk_result = pmodel.get_link_pose(self._ee_link_idx)

            p_base = fk_result[:3]
            q_base_wxyz = fk_result[3:]

            pose_in_base = sapien.Pose(p_base, q_base_wxyz)
            world_pose = self._robot_base_pose * pose_in_base

            position_t = torch.as_tensor(
                np.asarray(world_pose.p, dtype=np.float64), dtype=torch.float64
            )
            quat_wxyz_t = torch.as_tensor(
                np.asarray(world_pose.q, dtype=np.float64), dtype=torch.float64
            )

            pose_dict, self._prev_action_ee_quat_wxyz, self._prev_action_ee_rpy_xyz = (
                build_endeffector_pose_dict(
                    position_t,
                    quat_wxyz_t,
                    self._prev_action_ee_quat_wxyz,
                    self._prev_action_ee_rpy_xyz,
                )
            )
            return pose_dict
        except Exception as exc:
            print(f"[RecordWrapper] FK computation failed: {exc}")
            return None

    def reset(self, **kwargs):
        # Reset continuousness cache per episode to avoid cross-episode pollution
        self._prev_ee_quat_wxyz = None
        self._prev_ee_rpy_xyz = None
        self._prev_action_ee_quat_wxyz = None
        self._prev_action_ee_rpy_xyz = None
        self._current_keypoint_action = None  # Persist keypoint_action (7D ndarray)
        self._failsafe_triggered = False
        result = super().reset(**kwargs)
        self._init_fk_planner()
        return result

    def _consume_pending_keypoint(self) -> bool:
        """
        Check and consume env._pending_keypoint if it exists.
        Converts keypoint_p/keypoint_q into a 7D keypoint_action
        [position(3), rpy(3), gripper(1)] and stores it in
        self._current_keypoint_action.

        Returns True if a keypoint was consumed (i.e. this step is a keyframe).
        """
        env_unwrapped = getattr(self.env, 'unwrapped', self.env)
        if not (hasattr(env_unwrapped, '_pending_keypoint') and env_unwrapped._pending_keypoint is not None):
            return False

        current_keypoint = env_unwrapped._pending_keypoint

        if 'keypoint_p' not in current_keypoint or 'keypoint_q' not in current_keypoint:
            raise ValueError(
                f"_pending_keypoint missing keypoint_p/keypoint_q: {current_keypoint}"
            )

        keypoint_p_np = np.asarray(current_keypoint['keypoint_p']).reshape(-1)
        keypoint_q_np = np.asarray(current_keypoint['keypoint_q']).reshape(-1)
        if keypoint_p_np.size != 3 or keypoint_q_np.size != 4:
            raise ValueError(
                f"_pending_keypoint keypoint shape invalid: p={keypoint_p_np.shape}, q={keypoint_q_np.shape}"
            )

        # Reuse prev_quat/prev_rpy of current frame to convert keypoint quat to continuous RPY
        kp_pose_dict, _, _ = build_endeffector_pose_dict(
            torch.as_tensor(keypoint_p_np),
            torch.as_tensor(keypoint_q_np),
            self._prev_ee_quat_wxyz,
            self._prev_ee_rpy_xyz,
        )
        kp_type = current_keypoint.get('keypoint_type', 'unknown')
        gripper_val = 1.0 if kp_type == 'open' else -1.0
        self._current_keypoint_action = np.concatenate([
            kp_pose_dict['pose'].detach().cpu().numpy().flatten()[:3],
            kp_pose_dict['rpy'].detach().cpu().numpy().flatten()[:3],
            [gripper_val],
        ])

        env_unwrapped._pending_keypoint = None
        return True

    def _backfill_keypoint_actions_in_buffer(self) -> None:
        # Backfill keypoint_action in buffer before close():
        # 1. Steps [0, k0]: filled with k0's keypoint_action (first keyframe)
        # 2. Steps (k_prev, k_curr]: filled with k_curr's keypoint_action (between adjacent keyframes)
        # 3. Steps (k_last, end]: filled with k_last's keypoint_action (after last keyframe)
        # Also handles the single-keyframe case (len == 1) by filling the entire buffer.
        if not self.buffer:
            return

        def _as_bool_flag(value) -> bool:
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    return False
                return bool(value.detach().cpu().bool().any().item())
            if isinstance(value, np.ndarray):
                if value.size == 0:
                    return False
                return bool(np.asarray(value).astype(bool).any())
            return bool(value)

        def _validate_keypoint_action(kf_idx):
            """Validate and return the 7D keypoint_action for a keyframe index, or None."""
            kf_action = self.buffer[kf_idx].get("action", {})
            kp = kf_action.get("keypoint_action", None)
            if kp is None:
                print(
                    f"Warning: keyframe {kf_idx} has None keypoint_action, skip backfill"
                )
                return None
            target_kp = np.asarray(kp).flatten()
            if target_kp.size != 7:
                print(
                    f"Warning: keyframe {kf_idx} keypoint_action shape invalid {target_kp.shape}, "
                    f"skip backfill"
                )
                return None
            if not np.isfinite(target_kp).all():
                print(
                    f"Warning: keyframe {kf_idx} keypoint_action has non-finite values, "
                    f"skip backfill"
                )
                return None
            return target_kp

        def _fill_range(start, end_exclusive, target_kp):
            """Fill buffer[start:end_exclusive] with target_kp."""
            for fill_idx in range(start, end_exclusive):
                fill_action = self.buffer[fill_idx].setdefault("action", {})
                fill_action["keypoint_action"] = target_kp.copy()

        keyframe_indices = []
        for idx, record_data in enumerate(self.buffer):
            info_data = record_data.get("info", {})
            if _as_bool_flag(info_data.get("is_keyframe", False)):
                keyframe_indices.append(idx)

        if len(keyframe_indices) == 0:
            return

        # --- Handle single keyframe: fill entire buffer with its action ---
        if len(keyframe_indices) == 1:
            kf_idx = keyframe_indices[0]
            target_kp = _validate_keypoint_action(kf_idx)
            if target_kp is not None:
                _fill_range(0, len(self.buffer), target_kp)
            return

        # --- Multiple keyframes ---

        # 1. Fill [0, k0] with k0's keypoint_action (leading steps before/including first keyframe)
        first_kf = keyframe_indices[0]
        first_kp = _validate_keypoint_action(first_kf)
        if first_kp is not None:
            _fill_range(0, first_kf + 1, first_kp)

        # 2. Fill (k_prev, k_curr] intervals between adjacent keyframes
        for prev_idx, curr_idx in zip(keyframe_indices, keyframe_indices[1:]):
            target_kp = _validate_keypoint_action(curr_idx)
            if target_kp is None:
                continue
            _fill_range(prev_idx + 1, curr_idx + 1, target_kp)

        # 3. Fill (k_last, end] with last keyframe's keypoint_action (trailing steps)
        last_kf = keyframe_indices[-1]
        last_kp = _validate_keypoint_action(last_kf)
        if last_kp is not None and last_kf + 1 < len(self.buffer):
            _fill_range(last_kf + 1, len(self.buffer), last_kp)

    def step(self, action):
        self.no_object_flag=False
        obs, reward, terminated, truncated, info = super().step(action)


        # Parse raw observation: RGB, Segmentation Mask all keep data after torch->numpy, ensure direct write to HDF5
        base_camera_frame = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
        base_camera_depth = obs['sensor_data']['base_camera']['depth'][0].cpu().numpy()
        wrist_camera_frame = obs['sensor_data']['hand_camera']['rgb'][0].cpu().numpy()
        wrist_camera_depth = obs['sensor_data']['hand_camera']['depth'][0].cpu().numpy()

        base_camera_extrinsic=obs['sensor_param']['base_camera']['extrinsic_cv'].reshape(3, 4)
        base_camera_intrinsic=obs['sensor_param']['base_camera']['intrinsic_cv'].reshape(3, 3)
        wrist_camera_extrinsic=obs['sensor_param']['hand_camera']['extrinsic_cv'].reshape(3, 4)
        wrist_camera_intrinsic=obs['sensor_param']['hand_camera']['intrinsic_cv'].reshape(3, 3)
        
        
        segmentation=obs['sensor_data']['base_camera']['segmentation'].cpu().numpy()[0]

        # Get current subgoal name and online planning subgoal name
        current_subgoal_segment = getattr(self.unwrapped, 'current_subgoal_segment', None)
        current_subgoal_segment_online = getattr(self.unwrapped, 'current_subgoal_segment_online', None)
        current_task_name_online = getattr(self.unwrapped, 'current_task_name_online', getattr(self, 'current_task_name_online', 'Unknown'))
        
        # Process offline planning segmentation info: Generate visualization, calculate target center, fill placeholders in subgoal text
        segmentation_output = process_segmentation(
            segmentation=segmentation,
            segmentation_id_map=getattr(self, "segmentation_id_map", None),
            color_map=self.color_map,
            current_segment=getattr(self, "current_segment", None),
            current_subgoal_segment=current_subgoal_segment,
            previous_subgoal_segment=self.previous_subgoal_segment,
            current_task_name=getattr(self, 'current_task_name', 'Unknown'),
            existing_points=self.segmentation_points,
            existing_subgoal_filled=self.current_subgoal_segment_filled,
        )
        segmentation_result = segmentation_output["segmentation_result"]
        self.segmentation_points = segmentation_output["segmentation_points"]
        self.current_subgoal_segment_filled = segmentation_output[
            "current_subgoal_segment_filled"
        ]
        self.no_object_flag = segmentation_output["no_object_flag"]
        self.previous_subgoal_segment = segmentation_output[
            "updated_previous_subgoal_segment"
        ]
        self.vis_obj_id_list = segmentation_output["vis_obj_id_list"]

        # Process online planning segmentation info (logic same as above, but for online target)
        segmentation_output_online = process_segmentation(
            segmentation=segmentation,
            segmentation_id_map=getattr(self, "segmentation_id_map", None),
            color_map=self.color_map,
            current_segment=getattr(self, "current_segment_online", None),
            current_subgoal_segment=current_subgoal_segment_online,
            previous_subgoal_segment=self.previous_subgoal_segment_online,
            current_task_name=current_task_name_online,
            existing_points=self.segmentation_points_online,
            existing_subgoal_filled=self.current_subgoal_segment_online_filled,
        )
        segmentation_result_online = segmentation_output_online["segmentation_result"]
        self.segmentation_points_online = segmentation_output_online["segmentation_points"]
        self.current_subgoal_segment_filled = segmentation_output[
            "current_subgoal_segment_filled"
        ]
        # Note: This might should be online filling result? But original code overwrote self.current_subgoal_segment_online_filled
        self.current_subgoal_segment_online_filled = segmentation_output_online[
            "current_subgoal_segment_filled"
        ]
        self.no_object_flag_online = segmentation_output_online["no_object_flag"]
        self.previous_subgoal_segment_online = segmentation_output_online[
            "updated_previous_subgoal_segment"
        ]
        self.vis_obj_id_list_online = segmentation_output_online["vis_obj_id_list"]

        current_task=self.current_task_name if hasattr(self, 'current_task_name') else "Unknown"
        
        # Video recording logic: Execute only when task name is not NO RECORD and video saving enabled
        if self._video_should_record(current_task):
                
            # If demonstration task, add red border (video only, does not affect HDF5)
            is_demonstration = getattr(self, 'current_task_demonstration', False)
            subgoal_text = getattr(self, 'current_task_name', 'Unknown')
            subgoal_online_text = getattr(self, 'current_task_name_online', 'Unknown')

            language_goal = task_goal.get_language_goal(self.env, self.Robomme_env)
            prepared = self._video_prepare_step_frames(
                base_camera_frame,
                wrist_camera_frame,
                segmentation,
                segmentation_result,
                segmentation_result_online,
            )
            combined = self._video_compose_planner_online_rows(
                prepared,
                subgoal_text,
                self.current_subgoal_segment_filled,
                subgoal_online_text,
                self.current_subgoal_segment_online_filled,
            )
            combined = self._video_apply_overlays(
                combined,
                is_demonstration,
                language_goal,
            )
            self._video_append_step_frame(combined, self.no_object_flag)

            #print(self.current_task_name)

            # Buffer data instead of writing directly to HDF5 (using raw frames, no border)
            #print(f"End-effector linear velocity: {self.agent.robot.links[9].get_linear_velocity().tolist()[0]}, angular velocity: {self.agent.robot.links[9].get_angular_velocity().tolist()[0]}")
            # end_effector_velocity = self.agent.robot.links[9].get_linear_velocity().tolist()[0] + self.agent.robot.links[9].get_angular_velocity().tolist()[0]

            # Process keypoint info: Read pending keypoint from env (refresh cache if exists)
            # Convert to 7D keypoint_action: [position(3), rpy(3), gripper(1)]
            # Cache by "current latest keypoint" here; finally do backward fill post-processing before close().
            is_keyframe = self._consume_pending_keypoint()

            eef_pose_dict, self._prev_ee_quat_wxyz, self._prev_ee_rpy_xyz = build_endeffector_pose_dict(
                self.agent.tcp.pose.p,
                self.agent.tcp.pose.q,
                self._prev_ee_quat_wxyz,
                self._prev_ee_rpy_xyz,
            )

            def _to_numpy(value):
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu().numpy()
                return np.asarray(value)

            joint_state = self.agent.robot.qpos.cpu().numpy() if hasattr(self.agent.robot.qpos, 'cpu') else self.agent.robot.qpos
            joint_state = np.asarray(joint_state).flatten()
            # gripper is at indices 7-8; joint_state stored as first 7 dims
            gripper_state = joint_state[7:9] if joint_state.size >= 9 else np.zeros(2)
            gripper_close = bool(np.any(gripper_state < 0.03))
            joint_state = joint_state[:7]

            eef_action = np.concatenate([
                _to_numpy(eef_pose_dict['pose']).flatten()[:3],
                _to_numpy(eef_pose_dict['rpy']).flatten()[:3],
                _to_numpy(action).flatten()[-1:] if action is not None else np.array([-1.0]),
            ])


            # FK from joint_action -> eef_action_raw (pose/quat/rpy) and eef_action (7D)
            action_pose_dict = self._joint_action_to_ee_pose_dict(action)
            if action_pose_dict is not None:
                fk_pose = _to_numpy(action_pose_dict['pose']).flatten()[:3]
                fk_quat = _to_numpy(action_pose_dict['quat']).flatten()[:4]
                fk_rpy = _to_numpy(action_pose_dict['rpy']).flatten()[:3]
                gripper_val = (
                    _to_numpy(action).flatten()[-1:]
                    if action is not None
                    else np.array([-1.0])
                )
                fk_eef_action = np.concatenate([fk_pose, fk_rpy, gripper_val])
            else:
                fk_pose = np.zeros(3)
                fk_quat = np.zeros(4)
                fk_rpy = np.zeros(3)
                fk_eef_action = np.zeros(7)

            record_data = {
                'obs': {
                    'front_rgb': base_camera_frame,
                    'wrist_rgb': wrist_camera_frame,
                    'front_depth': base_camera_depth,
                    'wrist_depth': wrist_camera_depth,
                    'joint_state': joint_state,

                    'gripper_state': gripper_state,
                    'is_gripper_close': gripper_close,
                    # 'eef_velocity': end_effector_velocity,
                    'front_camera_segmentation': segmentation,
                    'front_camera_segmentation_result': segmentation_result,
                    'front_camera_extrinsic': base_camera_extrinsic,
                    'wrist_camera_extrinsic': wrist_camera_extrinsic,
                    'eef_state_raw': {
                        'pose': _to_numpy(eef_pose_dict['pose']).flatten(),
                        'quat': _to_numpy(eef_pose_dict['quat']).flatten(),
                        'rpy': _to_numpy(eef_pose_dict['rpy']).flatten(),
                    },
                },
                'action': {
                    'joint_action': action,
                    'keypoint_action': self._current_keypoint_action,  # 7D ndarray or None (backward fill done before close())
                    'eef_action_raw': {
                        'pose': fk_pose,
                        'quat': fk_quat,
                        'rpy': fk_rpy,
                    },
                    'eef_action': fk_eef_action,
                    'choice_action': "{}",
                },
                'info': {
                    'simple_subgoal': subgoal_text,
                    'simple_subgoal_online': subgoal_online_text,
                    'grounded_subgoal': self.current_subgoal_segment_filled,
                    'grounded_subgoal_online': self.current_subgoal_segment_online_filled,
                    'is_video_demo': self.current_task_demonstration if hasattr(self, 'current_task_demonstration') else False,
                    'is_keyframe': is_keyframe,
                },
                '_setup_camera_intrinsics': {
                    'front_camera_intrinsic': base_camera_intrinsic,
                    'wrist_camera_intrinsic': wrist_camera_intrinsic,
                },
            }

            self.buffer.append(record_data)


        # Check if episode successful
        if terminated.any():
            if info.get("success") == torch.tensor([True]) or (isinstance(info.get("success"), torch.Tensor) and info.get("success").item()):
                self.episode_success = True
                print("Episode success detected, data will be saved")
            else:
                self.episode_success = False
                print("Episode failed, data will be discarded")

        # Failsafe: enforce a hard cap on episode length so planners can't run forever
        # Keep English comment to retain original meaning: Force truncate when planner stuck, protect recording process
        # Force terminate episode if environment steps exceed preset safety limit (2000 steps)
        fail_safe_limit = 2000
        env_steps = int(getattr(self.env.unwrapped, "elapsed_steps", getattr(self.env, "elapsed_steps", 0)))
        #print(env_steps)
        if env_steps >= fail_safe_limit:

            # Mark episode as truncated due to failsafe
            if isinstance(truncated, torch.Tensor):
                truncated = torch.ones_like(truncated, dtype=torch.bool)
            elif isinstance(truncated, np.ndarray):
                truncated = np.ones_like(truncated, dtype=bool)
            else:
                truncated = True

            if isinstance(terminated, torch.Tensor):
                terminated = torch.zeros_like(terminated, dtype=torch.bool)
            elif isinstance(terminated, np.ndarray):
                terminated = np.zeros_like(terminated, dtype=bool)
            else:
                terminated = False

            info = dict(info)
            info["TimeLimit.truncated"] = True
            info["failsafe_elapsed_steps"] = env_steps
            self.episode_success = False
            print(f"Failsafe triggered at {env_steps} steps; terminating episode early.")
            if not self._failsafe_triggered:
                self._failsafe_triggered = True
                raise FailsafeTimeout(f"Episode exceeded failsafe limit ({env_steps} >= {fail_safe_limit})")

        return obs, reward, terminated, truncated, info

    def close(self):
        # Generate language goal (needed for both success and failure video filenames)
        language_goal = ""
        difficulty = getattr(self.env.unwrapped, 'difficulty', None)
       
        # language_goal mainly used for video naming and HDF5 metadata, needed for both failure/success
        language_goal=task_goal.get_language_goal(self.env,self.Robomme_env)
        filename_parts = self._video_build_filename_parts(language_goal, difficulty)
        filename_suffix = filename_parts["filename_suffix"]
        fail_recover_suffix = ""
        if getattr(self.env, "use_fail_planner", False):
            fail_mode = getattr(self.env, "fail", None)
            if fail_mode == "xy":
                fail_recover_suffix = "_FailRecoverXY"
            elif fail_mode == "z":
                fail_recover_suffix = "_FailRecoverZ"
            else:
                fail_recover_suffix = "_FailRecover"
        video_prefix = f"{self.Robomme_env}_ep{self.Robomme_episode}_seed{self.Robomme_seed}{fail_recover_suffix}"

        # Write data to HDF5 only when episode successful
        if self.episode_success:
            print(f"Writing {len(self.buffer)} records to HDF5...")

            # Consume any unconsumed _pending_keypoint left by the last planner action.
            # This fixes the case where _record_keypoint() is the final action in a planner
            # function with no subsequent step() call to consume it.
            if self.buffer and self._consume_pending_keypoint():
                last_record = self.buffer[-1]
                last_record.setdefault("action", {})["keypoint_action"] = (
                    self._current_keypoint_action.copy()
                )
                last_record.setdefault("info", {})["is_keyframe"] = True
                print("Consumed trailing _pending_keypoint in close(), "
                      f"marked buffer[-1] (timestep {len(self.buffer) - 1}) as keyframe.")

            # keypoint_action backfill handled uniformly before writing to disk, avoiding changing real-time logic during step().
            self._backfill_keypoint_actions_in_buffer()

            # HDF5 hierarchy: episode_xxx / timestep_xxx, convenient for retrieval by environment and round
            # env_group_name = f"env_{self.Robomme_env}"
            # env_group = self.h5_file.require_group(env_group_name)
            episode_group_name = f"episode_{self.Robomme_episode}"
            # if episode_group_name in env_group:
            #     del env_group[episode_group_name]
            if episode_group_name in self.h5_file:
                del self.h5_file[episode_group_name]
            episode_group = self.h5_file.create_group(episode_group_name)

            # Write all buffered data
            for record_timestep, record_data in enumerate(self.buffer):
                base_group_name = f"timestep_{record_timestep}"
                group_name = base_group_name
                duplicate_index = 1
                # Avoid collisions when multiple records share the same timestep
                while group_name in episode_group:
                    group_name = f"{base_group_name}_dup{duplicate_index}"
                    duplicate_index += 1

                ts_group = episode_group.create_group(group_name)

                # ── obs sub group ──
                obs_group = ts_group.create_group("obs")
                obs_data = record_data['obs']
                obs_group.create_dataset("front_rgb", data=obs_data['front_rgb'])
                obs_group.create_dataset("wrist_rgb", data=obs_data['wrist_rgb'])
                obs_group.create_dataset("front_depth", data=obs_data['front_depth'])
                obs_group.create_dataset("wrist_depth", data=obs_data['wrist_depth'])

                obs_group.create_dataset("joint_state", data=obs_data['joint_state'])

                obs_group.create_dataset("gripper_state", data=obs_data['gripper_state'])
                obs_group.create_dataset("is_gripper_close", data=obs_data['is_gripper_close'])

                # obs_group.create_dataset("eef_velocity", data=obs_data['eef_velocity'])
                # obs_group.create_dataset("front_camera_segmentation", data=obs_data['front_camera_segmentation'])
                # obs_group.create_dataset("front_camera_segmentation_result", data=obs_data['front_camera_segmentation_result'])
                obs_group.create_dataset("front_camera_extrinsic", data=obs_data['front_camera_extrinsic'])
                obs_group.create_dataset("wrist_camera_extrinsic", data=obs_data['wrist_camera_extrinsic'])

                eef_state_raw_group = obs_group.create_group("eef_state_raw")
                eef_state_raw_group.create_dataset("pose", data=obs_data['eef_state_raw']['pose'])
                eef_state_raw_group.create_dataset("quat", data=obs_data['eef_state_raw']['quat'])
                eef_state_raw_group.create_dataset("rpy", data=obs_data['eef_state_raw']['rpy'])

                # ── action sub group ──
                action_group = ts_group.create_group("action")
                action_data_dict = record_data['action']

                # Action may be None (e.g. planner not yet output), write string to avoid h5py dtype error
                if action_data_dict['joint_action'] is None:
                    action_group.create_dataset("joint_action", data="None", dtype=h5py.special_dtype(vlen=str))
                else:
                    action_data = action_data_dict['joint_action']
                    if isinstance(action_data, torch.Tensor):
                        action_data = action_data.cpu().numpy()
                    if isinstance(action_data, list):
                        action_data = np.array(action_data)
                    
                    # joint_action ensure 8 dims, fill one -1 if 7 dims
                    if isinstance(action_data, np.ndarray):
                        if action_data.shape == (7,):
                            action_data = np.concatenate([action_data, [-1]])
                        elif action_data.shape == (1, 7):
                            action_data = action_data.flatten()
                            action_data = np.concatenate([action_data, [-1]])
                            action_data = action_data.reshape(1, 8)
                    action_group.create_dataset("joint_action", data=action_data)

                # eef_action_raw information (pose/quat/rpy sub-datasets)
                eef_action_raw_group = action_group.create_group("eef_action_raw")
                eef_action_raw_group.create_dataset("pose", data=action_data_dict['eef_action_raw']['pose'])
                eef_action_raw_group.create_dataset("quat", data=action_data_dict['eef_action_raw']['quat'])
                eef_action_raw_group.create_dataset("rpy", data=action_data_dict['eef_action_raw']['rpy'])

                # eef_action: 7-dim [pose(3), rpy(3), gripper(1)]
                action_group.create_dataset("eef_action", data=action_data_dict['eef_action'])

                # Write keypoint_action (7D: pos(3)+rpy(3)+gripper(1), value backfilled before close())
                kp_action = action_data_dict.get('keypoint_action', None)
                if kp_action is None:
                    kp_action = np.zeros(7)
                    
                action_group.create_dataset("keypoint_action", data=kp_action)

                # choice_action: empty dict string placeholder
                action_group.create_dataset("choice_action", data=action_data_dict.get('choice_action', '{}'), dtype=h5py.special_dtype(vlen=str))

                # ── info sub group ──
                info_group = ts_group.create_group("info")
                info_data = record_data['info']

                # Process string task name, ensure correct encoding
                task_name = info_data['simple_subgoal']
                if isinstance(task_name, str):
                    task_name_encoded = task_name.encode('utf-8')
                else:
                    task_name_encoded = task_name
                info_group.create_dataset("simple_subgoal", data=task_name_encoded)

                online_task_name = info_data.get('simple_subgoal_online', 'Unknown')
                if isinstance(online_task_name, str):
                    task_name_encoded = online_task_name.encode('utf-8')
                else:
                    task_name_encoded = online_task_name
                info_group.create_dataset("simple_subgoal_online", data=task_name_encoded)

                task_name = info_data['grounded_subgoal']
                if isinstance(task_name, str):
                    task_name_encoded = task_name.encode('utf-8')
                else:
                    task_name_encoded = task_name
                info_group.create_dataset("grounded_subgoal", data=task_name_encoded)

                task_name_online = info_data.get('grounded_subgoal_online', 'Unknown')
                if isinstance(task_name_online, str):
                    task_name_encoded = task_name_online.encode('utf-8')
                else:
                    task_name_encoded = task_name_online
                info_group.create_dataset("grounded_subgoal_online", data=task_name_encoded)

                info_group.create_dataset("is_video_demo", data=info_data['is_video_demo'])
                info_group.create_dataset("is_keyframe", data=info_data['is_keyframe'])

            # Write setup info (seed, difficulty, task list, camera intrinsics)
            setup_group = episode_group.create_group(f"setup")
            setup_group.create_dataset("seed", data=self.Robomme_seed)
            setup_group.create_dataset(
                "available_multi_choices",
                data="",
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            setup_group.create_dataset(
                    "difficulty",
                    data=difficulty,
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
            env_unwrapped = getattr(self.env, "unwrapped", self.env)
            fail_recover_mode = getattr(env_unwrapped, "fail_recover_mode", None)
            if fail_recover_mode is not None:
                setup_group.create_dataset(
                    "fail_recover_mode",
                    data=str(fail_recover_mode),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
            fail_recover_seed_anchor = getattr(env_unwrapped, "fail_recover_seed_anchor", None)
            if fail_recover_seed_anchor is not None:
                setup_group.create_dataset(
                    "fail_recover_seed_anchor",
                    data=int(fail_recover_seed_anchor),
                )
            fail_recover_xy_signs = getattr(env_unwrapped, "fail_recover_xy_signs", None)
            if fail_recover_xy_signs is not None:
                xy_signs_np = np.asarray(fail_recover_xy_signs).reshape(-1)
                if xy_signs_np.size == 2:
                    setup_group.create_dataset("fail_recover_xy_signs", data=xy_signs_np)
                else:
                    print(
                        "Warning: skip writing fail_recover_xy_signs due to invalid size "
                        f"{xy_signs_np.size}"
                    )
            fail_recover_xy_signed_offset = getattr(env_unwrapped, "fail_recover_xy_signed_offset", None)
            if fail_recover_xy_signed_offset is not None:
                xy_signed_offset_np = np.asarray(fail_recover_xy_signed_offset).reshape(-1)
                if xy_signed_offset_np.size == 2:
                    setup_group.create_dataset(
                        "fail_recover_xy_signed_offset", data=xy_signed_offset_np
                    )
                else:
                    print(
                        "Warning: skip writing fail_recover_xy_signed_offset due to invalid size "
                        f"{xy_signed_offset_np.size}"
                    )

            # Camera intrinsics: Save only once per episode (take value from first buffer)
            if self.buffer:
                intrinsics = self.buffer[0].get('_setup_camera_intrinsics', {})
                if 'front_camera_intrinsic' in intrinsics:
                    setup_group.create_dataset("front_camera_intrinsic", data=intrinsics['front_camera_intrinsic'].reshape(3, 3))
                if 'wrist_camera_intrinsic' in intrinsics:
                    setup_group.create_dataset("wrist_camera_intrinsic", data=intrinsics['wrist_camera_intrinsic'].reshape(3, 3))

            if language_goal:
                setup_group.create_dataset("task_goal", data=language_goal)

            # Save success video (if enabled). Filename contains language goal/difficulty for easy lookup
            # Note: Video save failure should not affect HDF5 data saving
            self._video_flush_episode_files(
                success=True,
                video_prefix=video_prefix,
                filename_suffix=filename_suffix,
            )

            print(f"Successfully saved episode {self.Robomme_episode}")
        else:
            print(f"Episode {self.Robomme_episode} failed, discarding {len(self.buffer)} records")

            # Save failure video (if enabled), but do not write HDF5
            # Note: Video save failure should not throw exception
            self._video_flush_episode_files(
                success=False,
                video_prefix=video_prefix,
                filename_suffix=filename_suffix,
            )

            # If episode failed, delete created group (if any)
            episode_group_name = f"episode_{self.Robomme_episode}"
            if episode_group_name in self.h5_file:
                del self.h5_file[episode_group_name]
                print(f"Deleted episode group: {episode_group_name}")

        # Clear buffer to prevent repeated writing if close called multiple times
        self.buffer.clear()
        self.video_frames.clear()
        self.no_object_video_frames.clear()

        # Close HDF5 file
        if self.h5_file:
            self.h5_file.close()

        return super().close()
