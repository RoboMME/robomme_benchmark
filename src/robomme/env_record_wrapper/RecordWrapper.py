import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

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
from ..robomme_env.utils.vqa_options import get_vqa_options
from ..robomme_env.utils.segmentation_utils import (
    process_segmentation,
    create_segmentation_visuals,
)
from ..robomme_env.utils.rpy_util import build_endeffector_pose_dict
from ..robomme_env.utils.oracle_action_matcher import map_action_text_to_option_label
from ..robomme_env.utils.choice_action_mapping import (
    extract_actor_position_xyz,
    project_world_to_pixel,
)
from ..robomme_env.utils import swap_contact_monitoring as swapContact
from ..env_record_wrapper import object_log as objectlog

from ..logging_utils import logger

class FailsafeTimeout(RuntimeError):
    """Robomme 的 failsafe 提前截断 episode 时抛出的异常。"""
    pass


def _is_online_subgoal_completed(current_task_index, task_list) -> bool:
    """判断 online 子目标推进是否已经越过全部任务。"""
    if task_list is None:
        return False
    try:
        num_tasks = len(task_list)
    except Exception:
        return False
    if num_tasks <= 0:
        return False
    try:
        current_task_index = int(current_task_index)
    except (TypeError, ValueError):
        return False
    return current_task_index >= num_tasks


def _make_color_map(n=10000, s_min=0.70, s_max=0.95, v_min=0.78, v_max=0.95):
    """
    生成 1..n 的颜色字典，值为 [R, G, B]（0-255）。

    这里的颜色表只在 wrapper 初始化时构造一次，后续所有 segmentation
    可视化都复用它，避免在 step 中重复创建。
    - Hue 使用黄金分割步长，减少相邻 id 颜色扎堆
    - Saturation / Value 做小周期波动，提高视觉区分度
    """
    phi = 0.6180339887498948  # Golden ratio step
    color_map = {}
    for i in range(1, n + 1):
        h = (i * phi) % 1.0
        s = s_min + (s_max - s_min) * ((i % 7) / 6)  # 7-step cycle saturation
        v = v_min + (v_max - v_min) * (((i * 3) % 5) / 4)  # 5-step cycle value
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        color_map[i] = [
            int(round(r * 255)),
            int(round(g * 255)),
            int(round(b * 255)),
        ]
    return color_map


def _to_numpy(value):
    """把 tensor / list / ndarray 统一转成 numpy，供录制和 HDF5 写入使用。"""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class RobommeRecordWrapper(gym.Wrapper):
    FAILSAFE_STEP_LIMIT = 2000  # hard cap on steps per episode

    """
    Robomme 录制包装器。

    这个类同时承担三类职责：
    1. 把 rollout 中每一步的观测、动作和辅助信息缓存在内存里，并在 close 时落盘到 HDF5。
    2. 生成回放视频，把 base / wrist 画面、分割结果、目标点和文本 overlay 拼到一起。
    3. 维护若干跨 step 的缓存，例如 segmentation 目标点、choice action、waypoint、
       末端执行器姿态连续性等。

    设计上要特别注意：这里既有“只影响视频展示”的数据，也有“进入 HDF5 schema”
    的数据，二者不能混淆，尤其不能让视频上的边框或 overlay 反向污染原始记录。
    """
    def __init__(
        self,
        env,
        dataset=None,
        env_id=None,
        episode=None,
        seed=None,
        save_video=False,
        record_hdf5=True,
    ):
        # 先初始化父类，确保 self.env / self.unwrapped 等 gym.Wrapper 基础设施可用。
        super().__init__(env)
        self.unwrapped.use_demonstrationwrapper = False

        # 基础配置：这些字段主要用于输出路径命名和 close 时写元数据。
        self.dataset = dataset
        self.episode = episode
        self.env_id = env_id
        self.seed = seed
        self.save_video = save_video
        self.record_hdf5 = record_hdf5

        # failsafe 只应触发一次异常。后续如果 close / 上层清理逻辑再次触发检查，
        # 需要避免重复 raise 干扰调用方。
        self._failsafe_triggered = False

        # episode 级缓存：
        # - buffer: step 期间先写入内存，close 时再统一刷到 HDF5
        # - episode_success: 只在 episode 成功时才保留 HDF5；失败时视频可保留，HDF5 丢弃
        self.buffer = []
        self.episode_success = False

        # segmentation 相关缓存：
        # process_segmentation 会根据上一步的状态决定是否重用中心点、是否判定目标缺失，
        # 因此这些字段必须跨 step 保持。
        self.previous_subgoal_segment = None
        self.current_subgoal_segment_filled = None
        self.segmentation_points = []  # Cache segmentation center points
        self.previous_subgoal_segment_online = None
        self.current_subgoal_segment_online_filled = None
        self.segmentation_points_online = []  # Cache online segmentation target points

        # choice_action 相关缓存：
        # 当前实现通过 task_index 边界来刷新 choice label，避免相邻任务文字相同
        # 时仅靠 subgoal 文本造成误判。
        self._current_choice_action_text = ""  # Source choice action text from env task entry
        self._current_choice_label = ""  # Resolved option label (a/b/c/d/...)
        self._prev_task_index = -1           # Task index from previous step, used to detect subgoal switch
        self._prev_is_video_demo = False     # Track demonstration->non-demonstration boundary

        # 视频缓存：
        # 主视频保存所有帧；no_object_video_frames 只保存”当前目标缺失”的帧，便于调试分割失败。
        self.video_frames = []  # Store combined video frames
        self.no_object_video_frames = []  # Save separately when target missing in video frame, for debugging
        self._video_target_frame_size = None
        self._episode_object_log_flushed = False

        # 末端执行器姿态连续性缓存：
        # build_endeffector_pose_dict 会做四元数符号对齐和 RPY 展开，这里保留上一步结果，
        # 避免姿态表示在相邻帧之间跳变。
        self._prev_ee_quat_wxyz = None
        self._prev_ee_rpy_xyz = None

        # HDF5 句柄只在 record_hdf5=True 时真正打开。
        self.h5_file = None

        if not self.dataset:
            raise ValueError("RobommeRecordWrapper requires dataset/output path")

        # dataset 既支持传入目录，也支持直接传入一个 .h5/.hdf5 文件路径。
        # 如果传的是文件路径，这里会自动把目录和 hdf5 子目录名拆出来。
        base_path = Path(self.dataset).resolve()
        if base_path.suffix == '.h5' or base_path.suffix == '.hdf5':
            self.output_root = base_path.parent
            hdf5_folder_name = base_path.stem + "_hdf5_files"
        else:
            self.output_root = base_path
            hdf5_folder_name = "hdf5_files"

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.hdf5_dir = None
        self.dataset_path = None

        if self.record_hdf5:
            # Create folder to save HDF5 file
            self.hdf5_dir = self.output_root / hdf5_folder_name
            self.hdf5_dir.mkdir(parents=True, exist_ok=True)

            # HDF5 file saved in new created folder
            h5_filename = f"{self.env_id}_ep{self.episode}_seed{self.seed}.h5"
            self.dataset_path = self.hdf5_dir / h5_filename

            # Generate unique filename by env/episode/seed convention for batch analysis
            # Open in 'a' mode, delete and recreate if file corrupted
            try:
                self.h5_file = h5py.File(self.dataset_path, "a")
            except OSError as exc:
                if self.dataset_path.exists():
                    # Delete truncated/corrupted file and recreate a clean one
                    logger.debug(f"Failed to open existing dataset ({exc}); recreating file.")
                    self.dataset_path.unlink()
                    self.h5_file = h5py.File(self.dataset_path, "w")
                else:
                    raise
            logger.debug(f"Recording data to {self.dataset_path}")
            logger.debug(f"HDF5 files will be saved in folder: {self.hdf5_dir}")
        else:
            logger.debug(f"Recording videos only under {self.output_root}")

        # segmentation 可视化固定配色表。这里只初始化一次，后续所有 step 复用。
        color_map = _make_color_map()
        # color_map[16] = [0, 0, 0]  # Fix 16 as black (table)
        self.color_map = color_map

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
        """判断当前 step 是否需要走视频流水线。"""
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
        为单步视频拼接准备素材。

        这里会深拷贝原始图像和 segmentation，后续所有可视化修改都基于副本完成，
        以确保写入 HDF5 的仍然是原始观测，不会被红框、文字或红点污染。
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
        choice_action_payload,
        task_index,
        is_completed,
    ):
        """
        把 planner 行和 online 行拼成最终视频帧，并叠加关键 schema 字段。

        这样在看 mp4 时就能直接核对：
        - 当前子目标文本
        - grounded subgoal
        - choice_action
        - online 侧子目标
        """
        combined = prepared["combined"]
        combined_online = prepared["combined_online"]
        choice_action_json = json.dumps(choice_action_payload, ensure_ascii=False)

        # Add planner-side schema fields to first row for direct video inspection.
        combined = self._add_text_to_frame(
            combined,
            [
                "PLANNER:",
                f"info.simple_subgoal: {subgoal_text}",
                f"info.grounded_subgoal: {grounded_text}",
                f"action.choice_action: {choice_action_json}",
                f"info.is_completed: {bool(is_completed)}",
                f"task_index: {task_index}",
            ],
            position="top_right",
        )

        # Add online-side schema fields to second row for direct video inspection.
        combined_online = self._add_text_to_frame(
            combined_online,
            [
                "ONLINE:",
                f"info.simple_subgoal_online: {subgoal_online_text}",
                f"info.grounded_subgoal_online: {grounded_online_text}",
            ],
            position="top_right",
        )

        # Stack two video streams vertically
        return np.vstack([combined, combined_online])

    def _normalize_language_goal_list(self, language_goal):
        """把 language_goal 标准化为 list[str]，并仅过滤 None。"""
        if language_goal is None:
            return []
        if isinstance(language_goal, str):
            raw_items = [language_goal]
        elif isinstance(language_goal, (list, tuple)):
            raw_items = list(language_goal)
        else:
            raw_items = [language_goal]

        normalized = []
        for item in raw_items:
            if item is None:
                continue
            normalized.append(str(item))
        return normalized

    def _sanitize_filename_component(self, text):
        """把任意文本清洗成安全的文件名片段。"""
        text = str(text).strip()
        if not text:
            return ""
        text = text.replace("/", "_").replace("\\", "_")
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r"[^A-Za-z0-9._-]", "_", text)
        text = re.sub(r"_+", "_", text)
        return text.strip("._")

    def _truncate_filename_with_hash(self, text, max_len=220):
        """对过长文件名做截断，并追加稳定 hash。"""
        if len(text) <= max_len:
            return text
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        marker = "__HASH__"
        keep_len = max(1, max_len - len(marker) - len(digest))
        return f"{text[:keep_len]}{marker}{digest}"

    def _video_apply_overlays(self, frame, is_demonstration, language_goal):
        """给视频帧叠加 demonstration 红框和 language goal 文本。"""
        # 红框只用于回放展示，不应该进入 HDF5 原始图像。
        if is_demonstration:
            frame = self._add_red_border(frame)

        normalized_goals = self._normalize_language_goal_list(language_goal)
        return self._add_text_to_frame(frame, normalized_goals, position="top_right")

    def _video_append_step_frame(self, frame, no_object_flag):
        """
        追加单帧到视频缓存，并统一尺寸。

        文本 overlay 的行数可变，会让帧高发生变化；如果不在这里统一尺寸，
        imageio 写 mp4 时会因为输入帧大小不一致而报错。
        """
        h, w = frame.shape[:2]
        if self._video_target_frame_size is None:
            self._video_target_frame_size = (h, w)
        else:
            target_h, target_w = self._video_target_frame_size
            if (h, w) != (target_h, target_w):
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        self.video_frames.append(frame)
        if no_object_flag == True:
            self.no_object_video_frames.append(frame)

    def _video_build_filename_parts(self, language_goal, difficulty):
        """根据 language goal 和 difficulty 生成视频文件名后缀。"""
        normalized_goals = self._normalize_language_goal_list(language_goal)
        sanitized_goals = []
        for goal_text in normalized_goals:
            sanitized_goal = self._sanitize_filename_component(goal_text)
            if sanitized_goal:
                sanitized_goals.append(sanitized_goal)
        goal_tag = "__ALT__".join(sanitized_goals) if sanitized_goals else "no_goal"
        goal_tag = self._truncate_filename_with_hash(goal_tag, max_len=180)

        difficulty_tag = None
        if difficulty is not None:
            difficulty_tag = self._sanitize_filename_component(difficulty)
            if not difficulty_tag:
                difficulty_tag = None

        filename_suffix = goal_tag
        if difficulty_tag:
            filename_suffix = (
                f"{difficulty_tag}_{filename_suffix}"
                if filename_suffix
                else difficulty_tag
            )
        filename_suffix = self._truncate_filename_with_hash(filename_suffix, max_len=220)
        return {"filename_suffix": filename_suffix}

    def _video_write_mp4(self, frames, output_path):
        """用统一编码参数写出 mp4。"""
        with imageio.get_writer(
            output_path.as_posix(), fps=30, codec="libx264", quality=8
        ) as writer:
            for frame in frames:
                writer.append_data(frame)

    def _video_flush_episode_files(self, success, video_prefix, filename_suffix):
        """
        把当前 episode 的视频缓存写盘。

        主视频和 no-object 调试视频分开写；任何视频写失败都只记录日志，
        不应反向影响主流程。
        """
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
                    logger.debug(f"Saved combined video to {combined_video_path}")
                else:
                    combined_video_path = (
                        videos_dir / f"FAILED_{video_prefix}_{filename_suffix}.mp4"
                    )
                    self._video_write_mp4(self.video_frames, combined_video_path)
                    logger.debug(f"Saved failed episode video to {combined_video_path}")
            except Exception as e:
                if success:
                    logger.debug(
                        f"Warning: Failed to save combined video for episode {self.episode}: {e}"
                    )
                else:
                    logger.debug(
                        f"Warning: Failed to save failed episode video for episode {self.episode}: {e}"
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
                    logger.debug(f"Saved no-object video to {no_object_video_path}")
                else:
                    no_object_video_path = (
                        videos_dir
                        / f"FAILED_NO_OBJECT_{video_prefix}_{filename_suffix}.mp4"
                    )
                    self._video_write_mp4(
                        self.no_object_video_frames, no_object_video_path
                    )
                    logger.debug(f"Saved failed no-object video to {no_object_video_path}")
            except Exception as e:
                if success:
                    logger.debug(
                        f"Warning: Failed to save no-object video for episode {self.episode}: {e}"
                    )
                else:
                    logger.debug(
                        f"Warning: Failed to save failed no-object video for episode {self.episode}: {e}"
                    )

    def _init_fk_planner(self):
        """Initialize mplib FK planner after env.reset().

        Stores pinocchio_model, ee_link_idx and robot_base_pose for
        forward-kinematics computation in _joint_action_to_ee_pose_dict().
        Sets self._fk_available = False on failure so callers can fall back.
        """
        try:
            _STICK_IDS = ("PatternLock", "RouteStick")
            env_id = getattr(getattr(self.unwrapped, "spec", None), "id", None) or self.env_id
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
            self.planner = solver
            self._mplib_planner = solver.planner
            self._ee_link_idx = self._mplib_planner.link_name_2_idx[
                self._mplib_planner.move_group
            ]
            self._robot_base_pose = self.unwrapped.agent.robot.pose
            self._fk_qpos_size = len(self._mplib_planner.user_joint_names)
            self._fk_available = True
        except Exception as exc:
            logger.debug(f"[RecordWrapper] FK planner init failed, eef_action_raw/eef_action "
                  f"will be zeros: {exc}")
            self.planner = None
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

        这个 helper 的作用是把 joint_action 映射到与 eef_state_raw 同口径的
        pose / quat / rpy，方便后续写出 eef_action_raw / eef_action。
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
                # stick robot: pinocchio has only 7 dims; do not append finger joints
                full_qpos = arm_qpos
            else:
                # standard panda: pinocchio has 9 dims; append two finger joints
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
            logger.debug(f"[RecordWrapper] FK computation failed: {exc}")
            return None

    def reset(self, **kwargs):
        # 每个 episode 重新开始时，必须把所有跨帧连续性缓存清零，
        # 否则上一个 episode 的姿态展开结果会污染当前 episode。
        self._prev_ee_quat_wxyz = None
        self._prev_ee_rpy_xyz = None
        self._prev_action_ee_quat_wxyz = None
        self._prev_action_ee_rpy_xyz = None
        self._current_waypoint_action = None  # Persist waypoint_action (7D ndarray)
        self._failsafe_triggered = False
        self._video_target_frame_size = None
        self._episode_object_log_flushed = False
        # choice_action 相关状态也要按 episode 重置。
        self._current_choice_action_text = ""
        self._current_choice_label = ""
        self._prev_task_index = -1
        result = super().reset(**kwargs)
        self._prev_is_video_demo = bool(
            getattr(self.unwrapped, "current_task_demonstration", False)
        )
        self._init_fk_planner()
        # Stick environment (stick end-effector, no gripper) identifier: pinocchio model has only 7 user joints
        # When _fk_available=False, _fk_qpos_size is undefined, default treated as non-stick
        self.is_stick_env = getattr(self, '_fk_qpos_size', 9) == 7
        return result

    def _resolve_choice_label(self, choice_action_text, task_index) -> str:
        """把环境里的 choice action 文本解析成 schema 需要的选项标签。"""
        if not isinstance(choice_action_text, str) or not choice_action_text:
            return ""

        selected_target = {
            "obj": None,
            "name": None,
            "seg_id": None,
        }
        env_id = getattr(getattr(self.unwrapped, "spec", None), "id", None) or self.env_id

        try:
            solve_options = get_vqa_options(
                self.env,
                getattr(self, "planner", None),
                selected_target,
                env_id,
            )
        except Exception as exc:
            logger.debug(
                "[RecordWrapper] Failed to build VQA options for label mapping: "
                f"env={env_id}, task_index={task_index}, source_action='{choice_action_text}', error={exc}"
            )
            return ""

        matched_label = map_action_text_to_option_label(choice_action_text, solve_options)
        if matched_label is None:
            logger.debug(
                "[RecordWrapper] Choice label mapping missing, writing empty label: "
                f"env={env_id}, task_index={task_index}, source_action='{choice_action_text}'"
            )
            return ""
        return matched_label

    @staticmethod
    def _collect_choice_segment_candidates(item: Any, out: List[Any]) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                RobommeRecordWrapper._collect_choice_segment_candidates(child, out)
            return
        if isinstance(item, dict):
            for child in item.values():
                RobommeRecordWrapper._collect_choice_segment_candidates(child, out)
            return
        if item is not None:
            out.append(item)

    def _get_choice_action_position_3d(self) -> List[float]:
        """
        从 current_segment 中尽量提取 choice action 对应物体的三维坐标。

        current_segment 的结构在不同任务里可能是单对象、列表、嵌套字典等，
        所以这里先做统一展开，再尝试抽取位置。
        """
        current_segment = getattr(self, "current_segment", None)
        candidates: List[Any] = []
        self._collect_choice_segment_candidates(current_segment, candidates)
        for candidate in candidates:
            pos = extract_actor_position_xyz(candidate)
            if pos is not None:
                return pos.astype(np.float64).tolist()
        return []

    def _build_choice_action_payload(
        self,
        *,
        label: str,
        position_3d: List[float],
        front_camera_intrinsic: Any,
        front_camera_extrinsic: Any,
        front_rgb_frame: Any,
    ) -> dict:
        """
        生成 action.choice_action 对应的 payload。

        其中：
        - `choice` 是标准化后的选项字母
        - `point` 是投影到前视相机上的 [y, x] 像素坐标
        """
        image_shape = np.asarray(front_rgb_frame).shape
        position_2d = project_world_to_pixel(
            world_xyz=position_3d,
            intrinsic_cv=front_camera_intrinsic,
            extrinsic_cv=front_camera_extrinsic,
            image_shape=image_shape,
        )
        point_yx: List[int] = []
        if position_2d is not None and len(position_2d) >= 2:
            # project_world_to_pixel returns [x, y]; schema stores [y, x].
            point_yx = [int(position_2d[1]), int(position_2d[0])]
        choice = label.strip().upper() if isinstance(label, str) else ""
        return {
            "choice": choice,
            "point": point_yx,
        }

    def _refresh_pending_waypoint(self, expected_is_demo: bool) -> bool:
        """
        从 env._pending_waypoint 刷新 self._current_waypoint_action。

        会把 waypoint_p / waypoint_q 转成 7D waypoint_action：
        [position(3), rpy(3), gripper(1)]。

        只有与当前 demo phase 一致的 waypoint 才会被消费；
        跨 phase 的旧 waypoint 会被直接丢弃，避免演示阶段遗留关键点污染正常阶段。

        Returns True if a pending waypoint exists and current cache was refreshed.
        """
        env_unwrapped = getattr(self.env, 'unwrapped', self.env)
        if not (hasattr(env_unwrapped, '_pending_waypoint') and env_unwrapped._pending_waypoint is not None):
            return False

        current_waypoint = env_unwrapped._pending_waypoint
        waypoint_phase_is_demo = current_waypoint.get("waypoint_phase_is_demo")
        if waypoint_phase_is_demo is not None:
            if bool(waypoint_phase_is_demo) != bool(expected_is_demo):
                env_unwrapped._pending_waypoint = None
                logger.debug(
                    "Dropped cross-phase pending waypoint: "
                    f"tagged_is_demo={bool(waypoint_phase_is_demo)}, "
                    f"expected_is_demo={bool(expected_is_demo)}"
                )
                return False

        if 'waypoint_p' not in current_waypoint or 'waypoint_q' not in current_waypoint:
            raise ValueError(
                f"_pending_waypoint missing waypoint_p/waypoint_q: {current_waypoint}"
            )

        waypoint_p_np = np.asarray(current_waypoint['waypoint_p']).reshape(-1)
        waypoint_q_np = np.asarray(current_waypoint['waypoint_q']).reshape(-1)
        if waypoint_p_np.size != 3 or waypoint_q_np.size != 4:
            raise ValueError(
                f"_pending_waypoint waypoint shape invalid: p={waypoint_p_np.shape}, q={waypoint_q_np.shape}"
            )

        # Reuse prev_quat/prev_rpy of current frame to convert waypoint quat to continuous RPY
        kp_pose_dict, _, _ = build_endeffector_pose_dict(
            torch.as_tensor(waypoint_p_np),
            torch.as_tensor(waypoint_q_np),
            self._prev_ee_quat_wxyz,
            self._prev_ee_rpy_xyz,
        )
        if getattr(self, 'is_stick_env', False):
            # Stick environment has no gripper, waypoint intent is fixed to -1.0
            gripper_val = -1.0
        else:
            kp_type = current_waypoint.get('waypoint_type', 'unknown')
            gripper_val = 1.0 if kp_type == 'open' else -1.0
        self._current_waypoint_action = np.concatenate([
            kp_pose_dict['pose'].detach().cpu().numpy().flatten()[:3],
            kp_pose_dict['rpy'].detach().cpu().numpy().flatten()[:3],
            [gripper_val],
        ])

        return True

    def _clear_waypoint_caches_on_demo_end(self, *, clear_pending_waypoint: bool = True) -> None:
        """在 demonstration -> 非 demonstration 切换时清空 waypoint 缓存。"""
        self._current_waypoint_action = None
        if clear_pending_waypoint:
            env_unwrapped = getattr(self.env, "unwrapped", self.env)
            if hasattr(env_unwrapped, "_pending_waypoint"):
                env_unwrapped._pending_waypoint = None
        logger.debug("Cleared waypoint caches at demo->non-demo transition.")

    def _step_parse_raw_obs(self, obs):
        """
        从 obs 中抽出录制真正需要的原始字段。

        这里统一做 torch -> numpy 和矩阵 reshape，避免后续 helper 到处写重复样板代码。
        """
        return {
            "base_camera_frame": obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy(),
            "base_camera_depth": obs["sensor_data"]["base_camera"]["depth"][0].cpu().numpy(),
            "wrist_camera_frame": obs["sensor_data"]["hand_camera"]["rgb"][0].cpu().numpy(),
            "wrist_camera_depth": obs["sensor_data"]["hand_camera"]["depth"][0].cpu().numpy(),
            "base_camera_extrinsic": obs["sensor_param"]["base_camera"]["extrinsic_cv"].reshape(3, 4),
            "base_camera_intrinsic": obs["sensor_param"]["base_camera"]["intrinsic_cv"].reshape(3, 3),
            "wrist_camera_extrinsic": obs["sensor_param"]["hand_camera"]["extrinsic_cv"].reshape(3, 4),
            "wrist_camera_intrinsic": obs["sensor_param"]["hand_camera"]["intrinsic_cv"].reshape(3, 3),
            "segmentation": obs["sensor_data"]["base_camera"]["segmentation"].cpu().numpy()[0],
        }

    def _step_process_segmentation(self, segmentation, cur_task_index):
        """
        处理 planner / online 两条 segmentation 分支，并更新相关缓存。

        它既负责生成本帧需要的 segmentation 结果，也负责维护：
        - segmentation_points / segmentation_points_online
        - previous_subgoal_segment / previous_subgoal_segment_online
        - choice_action 的 task 边界刷新
        """
        current_subgoal_segment = getattr(
            self.unwrapped, "current_subgoal_segment", None
        )
        current_subgoal_segment_online = getattr(
            self.unwrapped, "current_subgoal_segment_online", None
        )
        current_task_name_online = getattr(
            self.unwrapped,
            "current_task_name_online",
            getattr(self, "current_task_name_online", "Unknown"),
        )

        segmentation_output = process_segmentation(
            segmentation=segmentation,
            segmentation_id_map=getattr(self, "segmentation_id_map", None),
            color_map=self.color_map,
            current_segment=getattr(self, "current_segment", None),
            current_subgoal_segment=current_subgoal_segment,
            previous_subgoal_segment=self.previous_subgoal_segment,
            current_task_name=getattr(self, "current_task_name", "Unknown"),
            existing_points=self.segmentation_points,
            existing_subgoal_filled=self.current_subgoal_segment_filled,
        )

        choice_action_subgoal_boundary = False
        if cur_task_index != self._prev_task_index:
            self._current_choice_action_text = getattr(
                self.unwrapped, "current_choice_label", ""
            )
            self._current_choice_label = self._resolve_choice_label(
                self._current_choice_action_text,
                cur_task_index,
            )
            self._prev_task_index = cur_task_index
            choice_action_subgoal_boundary = True
            self.previous_subgoal_segment = segmentation_output[
                "updated_previous_subgoal_segment"
            ]
        self.segmentation_points = segmentation_output["segmentation_points"]
        self.current_subgoal_segment_filled = segmentation_output[
            "current_subgoal_segment_filled"
        ]
        self.no_object_flag = segmentation_output["no_object_flag"]
        self.vis_obj_id_list = segmentation_output["vis_obj_id_list"]

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
        self.segmentation_points_online = segmentation_output_online[
            "segmentation_points"
        ]
        self.current_subgoal_segment_filled = segmentation_output[
            "current_subgoal_segment_filled"
        ]
        # 注意：这里刻意保留原有行为。
        # 虽然看起来还可以继续抽象或精简，但任何赋值顺序变化都可能影响后续字段内容。
        self.current_subgoal_segment_online_filled = segmentation_output_online[
            "current_subgoal_segment_filled"
        ]
        self.no_object_flag_online = segmentation_output_online["no_object_flag"]
        self.previous_subgoal_segment_online = segmentation_output_online[
            "updated_previous_subgoal_segment"
        ]
        self.vis_obj_id_list_online = segmentation_output_online["vis_obj_id_list"]

        current_task = (
            self.current_task_name if hasattr(self, "current_task_name") else "Unknown"
        )
        subgoal_text = getattr(self, "current_task_name", "Unknown")
        subgoal_online_text = getattr(self, "current_task_name_online", "Unknown")
        is_completed = _is_online_subgoal_completed(
            cur_task_index,
            getattr(self.unwrapped, "task_list", None),
        )

        return {
            "segmentation_result": segmentation_output["segmentation_result"],
            "segmentation_result_online": segmentation_output_online["segmentation_result"],
            "choice_action_subgoal_boundary": choice_action_subgoal_boundary,
            "current_task": current_task,
            "subgoal_text": subgoal_text,
            "subgoal_online_text": subgoal_online_text,
            "grounded_subgoal": self.current_subgoal_segment_filled,
            "grounded_subgoal_online": self.current_subgoal_segment_online_filled,
            "task_index": cur_task_index,
            "is_completed": is_completed,
        }

    def _step_compute_eef_and_action(self, action, eef_pose_dict):
        """
        计算写入 buffer 的机器人状态和动作派生量。

        这里会同时准备：
        - 当前真实末端状态 eef_state_raw
        - 基于 joint_action 反解出来的 eef_action_raw / eef_action
        """
        joint_state = (
            self.agent.robot.qpos.cpu().numpy()
            if hasattr(self.agent.robot.qpos, "cpu")
            else self.agent.robot.qpos
        )
        joint_state = np.asarray(joint_state).flatten()

        # Stick 环境没有 gripper 关节；标准 Panda 的夹爪关节在 qpos[7:9]。
        if getattr(self, "is_stick_env", False):
            gripper_state = np.zeros(2)
        else:
            gripper_state = joint_state[7:9] if joint_state.size >= 9 else np.zeros(2)
        gripper_close = bool(np.any(gripper_state < 0.03))
        joint_state = joint_state[:7]

        eef_action = np.concatenate(
            [
                _to_numpy(eef_pose_dict["pose"]).flatten()[:3],
                _to_numpy(eef_pose_dict["rpy"]).flatten()[:3],
                # Stick environment: action end is the 7th joint angle, not gripper command.
                np.array([-1.0])
                if getattr(self, "is_stick_env", False)
                else (
                    _to_numpy(action).flatten()[-1:]
                    if action is not None
                    else np.array([-1.0])
                ),
            ]
        )

        action_pose_dict = self._joint_action_to_ee_pose_dict(action)
        if action_pose_dict is not None:
            fk_pose = _to_numpy(action_pose_dict["pose"]).flatten()[:3]
            fk_quat = _to_numpy(action_pose_dict["quat"]).flatten()[:4]
            fk_rpy = _to_numpy(action_pose_dict["rpy"]).flatten()[:3]
            # stick robot has no gripper, use -1; standard panda takes gripper from action[-1]
            if self._fk_qpos_size == 7:
                gripper_val = np.array([-1.0])
            else:
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

        return {
            "joint_state": joint_state,
            "gripper_state": gripper_state,
            "gripper_close": gripper_close,
            "eef_state_raw": {
                "pose": _to_numpy(eef_pose_dict["pose"]).flatten(),
                "quat": _to_numpy(eef_pose_dict["quat"]).flatten(),
                "rpy": _to_numpy(eef_pose_dict["rpy"]).flatten(),
            },
            "eef_action_raw": {
                "pose": fk_pose,
                "quat": fk_quat,
                "rpy": fk_rpy,
            },
            "eef_action": eef_action,
            "fk_eef_action": fk_eef_action,
        }

    def _step_build_record(self, raw_obs, seg_results, eef_results, action, choice_action_payload):
        """
        组装单步 record_data。

        这是 step 与 close 之间的中间契约：
        step 只负责把字段准备完整并塞进 buffer；
        close 再统一把 buffer 写成最终 HDF5 结构。
        """
        return {
            "obs": {
                "front_rgb": raw_obs["base_camera_frame"],
                "wrist_rgb": raw_obs["wrist_camera_frame"],
                "front_depth": raw_obs["base_camera_depth"],
                "wrist_depth": raw_obs["wrist_camera_depth"],
                "joint_state": eef_results["joint_state"],
                "gripper_state": eef_results["gripper_state"],
                "is_gripper_close": eef_results["gripper_close"],
                # "eef_velocity": end_effector_velocity,
                "front_camera_segmentation": raw_obs["segmentation"],
                "front_camera_segmentation_result": seg_results["segmentation_result"],
                "front_camera_extrinsic": raw_obs["base_camera_extrinsic"],
                "wrist_camera_extrinsic": raw_obs["wrist_camera_extrinsic"],
                "eef_state_raw": eef_results["eef_state_raw"],
            },
            "action": {
                "joint_action": action,
                "waypoint_action": (
                    self._current_waypoint_action.copy()
                    if self._current_waypoint_action is not None
                    else None
                ),
                "eef_action_raw": eef_results["eef_action_raw"],
                "eef_action": eef_results["fk_eef_action"],
                "choice_action": json.dumps(choice_action_payload),
            },
            "info": {
                "simple_subgoal": seg_results["subgoal_text"],
                "simple_subgoal_online": seg_results["subgoal_online_text"],
                "grounded_subgoal": seg_results["grounded_subgoal"],
                "grounded_subgoal_online": seg_results["grounded_subgoal_online"],
                "is_completed": seg_results["is_completed"],
                "is_video_demo": (
                    self.current_task_demonstration
                    if hasattr(self, "current_task_demonstration")
                    else False
                ),
                "is_subgoal_boundary": seg_results["choice_action_subgoal_boundary"],
            },
            "_setup_camera_intrinsics": {
                "front_camera_intrinsic": raw_obs["base_camera_intrinsic"],
                "wrist_camera_intrinsic": raw_obs["wrist_camera_intrinsic"],
            },
        }

    def _step_record_video(self, raw_obs, seg_results, _eef_results, language_goal, choice_action_payload):
        """执行单步视频合成流水线。"""
        is_demonstration = getattr(self, "current_task_demonstration", False)
        prepared = self._video_prepare_step_frames(
            raw_obs["base_camera_frame"],
            raw_obs["wrist_camera_frame"],
            raw_obs["segmentation"],
            seg_results["segmentation_result"],
            seg_results["segmentation_result_online"],
        )
        combined = self._video_compose_planner_online_rows(
            prepared,
            seg_results["subgoal_text"],
            seg_results["grounded_subgoal"],
            seg_results["subgoal_online_text"],
            seg_results["grounded_subgoal_online"],
            choice_action_payload=choice_action_payload,
            task_index=seg_results["task_index"],
            is_completed=seg_results["is_completed"],
        )
        combined = self._video_apply_overlays(
            combined,
            is_demonstration,
            language_goal,
        )
        self._video_append_step_frame(combined, self.no_object_flag)

    def _step_check_success(self, terminated, info):
        """在 episode 终止时，根据 info.success 更新 episode_success。"""
        if terminated.any():
            if info.get("success") == torch.tensor([True]) or (
                isinstance(info.get("success"), torch.Tensor)
                and info.get("success").item()
            ):
                self.episode_success = True
            else:
                self.episode_success = False

    def _step_check_failsafe(self, terminated, truncated, info, env_steps):
        """
        执行超时保护。

        当环境步数超过 FAILSAFE_STEP_LIMIT 时，强制把 episode 标记为 truncated，
        并抛出 FailsafeTimeout，避免规划器无限卡住。
        """
        if env_steps < self.FAILSAFE_STEP_LIMIT:
            return terminated, truncated, info

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
        logger.debug(
            f"Failsafe triggered at {env_steps} steps; terminating episode early."
        )
        if not self._failsafe_triggered:
            self._failsafe_triggered = True
            raise FailsafeTimeout(
                f"Episode exceeded failsafe limit ({env_steps} >= {self.FAILSAFE_STEP_LIMIT})"
            )
        return terminated, truncated, info

    def step(self, action):
        """
        单步录制主流程。

        控制流分成五段：
        1. 处理 demonstration phase 变化和 waypoint 刷新
        2. 执行 env.step
        3. 解析 obs 并更新 segmentation / choice_action 状态
        4. 如需录制，则生成视频帧并把 record_data 追加到 buffer
        5. 更新成功状态并做 failsafe 检查
        """
        self.no_object_flag = False
        # 先看 demonstration phase 是否切换；如果切了，需要清掉旧缓存，
        # 再尝试刷新当前 step 对应的 pending waypoint。
        pre_step_is_demo = bool(
            getattr(self.unwrapped, "current_task_demonstration", False)
        )
        if self._prev_is_video_demo != pre_step_is_demo:
            self._clear_waypoint_caches_on_demo_end(clear_pending_waypoint=False)
        # waypoint 在 planner 执行前产生，所以要在 env.step() 前刷新。
        self._refresh_pending_waypoint(expected_is_demo=pre_step_is_demo)
        obs, reward, terminated, truncated, info = super().step(action)

        post_step_is_demo = bool(
            getattr(self.unwrapped, "current_task_demonstration", False)
        )
        self._prev_is_video_demo = post_step_is_demo

        raw_obs = self._step_parse_raw_obs(obs)
        cur_task_index = getattr(self.unwrapped, "current_task_index", -1)
        seg_results = self._step_process_segmentation(
            raw_obs["segmentation"],
            cur_task_index,
        )

        # 保持原始行为：只有进入视频分支时，当前 step 的 record_data 才会写入 buffer。
        if self._video_should_record(seg_results["current_task"]):
            language_goal = task_goal.get_language_goal(self.env, self.env_id)
            choice_action_payload = self._build_choice_action_payload(
                label=self._current_choice_label,
                position_3d=self._get_choice_action_position_3d(),
                front_camera_intrinsic=raw_obs["base_camera_intrinsic"],
                front_camera_extrinsic=raw_obs["base_camera_extrinsic"],
                front_rgb_frame=raw_obs["base_camera_frame"],
            )
            eef_pose_dict, self._prev_ee_quat_wxyz, self._prev_ee_rpy_xyz = (
                build_endeffector_pose_dict(
                    self.agent.tcp.pose.p,
                    self.agent.tcp.pose.q,
                    self._prev_ee_quat_wxyz,
                    self._prev_ee_rpy_xyz,
                )
            )
            eef_results = self._step_compute_eef_and_action(action, eef_pose_dict)
            self._step_record_video(
                raw_obs,
                seg_results,
                eef_results,
                language_goal,
                choice_action_payload,
            )
            self.buffer.append(
                self._step_build_record(
                    raw_obs,
                    seg_results,
                    eef_results,
                    action,
                    choice_action_payload,
                )
            )
        # 成功判定和 failsafe 与是否录视频无关，因此始终执行。
        self._step_check_success(terminated, info)
        env_steps = int(
            getattr(
                self.env.unwrapped,
                "elapsed_steps",
                getattr(self.env, "elapsed_steps", 0),
            )
        )
        terminated, truncated, info = self._step_check_failsafe(
            terminated,
            truncated,
            info,
            env_steps,
        )

        return obs, reward, terminated, truncated, info

    def _h5_encode_str(self, text):
        """统一处理 HDF5 字符串字段编码，减少重复 if/encode 逻辑。"""
        if isinstance(text, str):
            return text.encode("utf-8")
        return text

    def _h5_write_timestep(self, ts_group, record_data):
        """
        把一个 timestep 的 record_data 写入 HDF5。

        写入结构严格对应 buffer 中的 schema：
        obs / action / info 三个子 group。
        """
        obs_group = ts_group.create_group("obs")
        obs_data = record_data["obs"]
        obs_group.create_dataset("front_rgb", data=obs_data["front_rgb"])
        obs_group.create_dataset("wrist_rgb", data=obs_data["wrist_rgb"])
        obs_group.create_dataset("front_depth", data=obs_data["front_depth"])
        obs_group.create_dataset("wrist_depth", data=obs_data["wrist_depth"])
        obs_group.create_dataset("joint_state", data=obs_data["joint_state"])
        obs_group.create_dataset("gripper_state", data=obs_data["gripper_state"])
        obs_group.create_dataset("is_gripper_close", data=obs_data["is_gripper_close"])

        # obs_group.create_dataset("eef_velocity", data=obs_data['eef_velocity'])
        # obs_group.create_dataset("front_camera_segmentation", data=obs_data['front_camera_segmentation'])
        # obs_group.create_dataset("front_camera_segmentation_result", data=obs_data['front_camera_segmentation_result'])
        obs_group.create_dataset(
            "front_camera_extrinsic",
            data=obs_data["front_camera_extrinsic"],
        )
        obs_group.create_dataset(
            "wrist_camera_extrinsic",
            data=obs_data["wrist_camera_extrinsic"],
        )

        # Temporarily disabled by request: do not generate obs/eef_state_raw in HDF5.
        # Keep code commented for future restoration.
        # eef_state_raw_group = obs_group.create_group("eef_state_raw")
        # eef_state_raw_group.create_dataset("pose", data=obs_data['eef_state_raw']['pose'])
        # eef_state_raw_group.create_dataset("quat", data=obs_data['eef_state_raw']['quat'])
        # eef_state_raw_group.create_dataset("rpy", data=obs_data['eef_state_raw']['rpy'])

        # 对外暴露的 eef_state 是 6D：[位置 3 维 + RPY 3 维]。
        eef_state = np.concatenate(
            [
                np.asarray(obs_data["eef_state_raw"]["pose"]).flatten()[:3],
                np.asarray(obs_data["eef_state_raw"]["rpy"]).flatten()[:3],
            ]
        ).astype(np.float32)
        obs_group.create_dataset("eef_state", data=eef_state)

        action_group = ts_group.create_group("action")
        action_data_dict = record_data["action"]

        # planner 尚未输出动作时，显式写字符串 "None"，避免 h5py dtype 推断失败。
        if action_data_dict["joint_action"] is None:
            action_group.create_dataset(
                "joint_action",
                data="None",
                dtype=h5py.special_dtype(vlen=str),
            )
        else:
            action_data = action_data_dict["joint_action"]
            if isinstance(action_data, torch.Tensor):
                action_data = action_data.cpu().numpy()
            if isinstance(action_data, list):
                action_data = np.array(action_data)

            # joint_action ensure 8 dims, fill -1.0 (float) if 7 dims
            # Stick environment action is 7D, pad with gripper placeholder -1.0 to align to standard 8D
            if isinstance(action_data, np.ndarray):
                if action_data.shape == (7,):
                    action_data = np.concatenate([action_data, [-1.0]])
                elif action_data.shape == (1, 7):
                    action_data = action_data.flatten()
                    action_data = np.concatenate([action_data, [-1.0]])
                    action_data = action_data.reshape(1, 8)
            action_group.create_dataset("joint_action", data=action_data)

        # Temporarily disabled by request: do not generate action/eef_action_raw in HDF5.
        # Keep code commented for future restoration.
        # eef_action_raw_group = action_group.create_group("eef_action_raw")
        # eef_action_raw_group.create_dataset("pose", data=action_data_dict['eef_action_raw']['pose'])
        # eef_action_raw_group.create_dataset("quat", data=action_data_dict['eef_action_raw']['quat'])
        # eef_action_raw_group.create_dataset("rpy", data=action_data_dict['eef_action_raw']['rpy'])

        action_group.create_dataset("eef_action", data=action_data_dict["eef_action"])

        # waypoint_action 为空时，使用 NaN sentinel，而不是省略字段。
        kp_action = action_data_dict.get("waypoint_action", None)
        if kp_action is None:
            kp_action = np.full(7, np.nan, dtype=np.float32)
        action_group.create_dataset("waypoint_action", data=kp_action)

        action_group.create_dataset(
            "choice_action",
            data=action_data_dict.get("choice_action", "{}"),
            dtype=h5py.special_dtype(vlen=str),
        )

        info_group = ts_group.create_group("info")
        info_data = record_data["info"]
        info_group.create_dataset(
            "simple_subgoal",
            data=self._h5_encode_str(info_data["simple_subgoal"]),
        )
        info_group.create_dataset(
            "simple_subgoal_online",
            data=self._h5_encode_str(info_data.get("simple_subgoal_online", "Unknown")),
        )
        info_group.create_dataset(
            "grounded_subgoal",
            data=self._h5_encode_str(info_data["grounded_subgoal"]),
        )
        info_group.create_dataset(
            "grounded_subgoal_online",
            data=self._h5_encode_str(info_data.get("grounded_subgoal_online", "Unknown")),
        )
        info_group.create_dataset(
            "is_completed",
            data=bool(info_data.get("is_completed", False)),
        )
        info_group.create_dataset("is_video_demo", data=info_data["is_video_demo"])
        info_group.create_dataset(
            "is_subgoal_boundary",
            data=info_data["is_subgoal_boundary"],
        )

    def _h5_write_setup_group(self, episode_group, language_goal_list, difficulty):
        """
        写 episode 级的一次性元数据。

        这些内容不属于某个 step，而属于整个 rollout 上下文，例如：
        seed、difficulty、available choices、相机内参、task_goal。
        """
        setup_group = episode_group.create_group("setup")
        setup_group.create_dataset("seed", data=self.seed)
        try:
            selected_target = {
                "obj": None,
                "name": None,
                "seg_id": None,
            }

            env_id = getattr(getattr(self.unwrapped, "spec", None), "id", None) or self.env_id
            solve_options = get_vqa_options(
                self.env,
                getattr(self, "planner", None),
                selected_target,
                env_id,
            )
            available_options = [
                {
                    "label": opt.get("label"),
                    "action": opt.get("action", "Unknown"),
                    "need_parameter": bool(opt.get("available")),
                }
                for opt in solve_options
            ]
            available_multi_choices_str = json.dumps(available_options)
        except Exception as e:
            logger.debug(f"[RecordWrapper] Failed to compute available_multi_choices: {e}")
            available_multi_choices_str = ""

        setup_group.create_dataset(
            "available_multi_choices",
            data=available_multi_choices_str,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        setup_group.create_dataset(
            "difficulty",
            data=difficulty,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        # Temporarily disabled by request: do not generate fail_recover_* setup fields in HDF5.
        # Keep code commented for future restoration.
        # env_unwrapped = getattr(self.env, "unwrapped", self.env)
        # fail_recover_mode = getattr(env_unwrapped, "fail_recover_mode", None)
        # if fail_recover_mode is not None:
        #     setup_group.create_dataset(
        #         "fail_recover_mode",
        #         data=str(fail_recover_mode),
        #         dtype=h5py.string_dtype(encoding="utf-8"),
        #     )
        # fail_recover_seed_anchor = getattr(env_unwrapped, "fail_recover_seed_anchor", None)
        # if fail_recover_seed_anchor is not None:
        #     setup_group.create_dataset(
        #         "fail_recover_seed_anchor",
        #         data=int(fail_recover_seed_anchor),
        #     )
        # fail_recover_xy_signs = getattr(env_unwrapped, "fail_recover_xy_signs", None)
        # if fail_recover_xy_signs is not None:
        #     xy_signs_np = np.asarray(fail_recover_xy_signs).reshape(-1)
        #     if xy_signs_np.size == 2:
        #         setup_group.create_dataset("fail_recover_xy_signs", data=xy_signs_np)
        #     else:
        #         logger.debug(
        #             "Warning: skip writing fail_recover_xy_signs due to invalid size "
        #             f"{xy_signs_np.size}"
        #         )
        # fail_recover_xy_signed_offset = getattr(env_unwrapped, "fail_recover_xy_signed_offset", None)
        # if fail_recover_xy_signed_offset is not None:
        #     xy_signed_offset_np = np.asarray(fail_recover_xy_signed_offset).reshape(-1)
        #     if xy_signed_offset_np.size == 2:
        #         setup_group.create_dataset(
        #             "fail_recover_xy_signed_offset", data=xy_signed_offset_np
        #         )
        #     else:
        #         logger.debug(
        #             "Warning: skip writing fail_recover_xy_signed_offset due to invalid size "
        #             f"{xy_signed_offset_np.size}"
        #         )

        # 相机内参按设计在整个 episode 内不变，因此只从第一帧里取一次。
        if self.buffer:
            intrinsics = self.buffer[0].get("_setup_camera_intrinsics", {})
            if "front_camera_intrinsic" in intrinsics:
                setup_group.create_dataset(
                    "front_camera_intrinsic",
                    data=intrinsics["front_camera_intrinsic"].reshape(3, 3),
                )
            if "wrist_camera_intrinsic" in intrinsics:
                setup_group.create_dataset(
                    "wrist_camera_intrinsic",
                    data=intrinsics["wrist_camera_intrinsic"].reshape(3, 3),
                )

        if language_goal_list:
            setup_group.create_dataset(
                "task_goal",
                data=np.asarray(language_goal_list, dtype=object),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

    def _h5_flush_episode(self, episode_group_name, language_goal_list, difficulty):
        """
        把当前 episode 的整段 buffer 刷到 HDF5。

        这里主要负责控制流，不直接处理字段细节；
        字段细节由 _h5_write_timestep / _h5_write_setup_group 负责。
        """
        if episode_group_name in self.h5_file:
            del self.h5_file[episode_group_name]
        episode_group = self.h5_file.create_group(episode_group_name)

        for record_timestep, record_data in enumerate(self.buffer):
            base_group_name = f"timestep_{record_timestep}"
            group_name = base_group_name
            duplicate_index = 1
            while group_name in episode_group:
                group_name = f"{base_group_name}_dup{duplicate_index}"
                duplicate_index += 1

            ts_group = episode_group.create_group(group_name)
            self._h5_write_timestep(ts_group, record_data)

        self._h5_write_setup_group(episode_group, language_goal_list, difficulty)

    def _episode_object_log_flush(self) -> None:
        """
        Episode object logging 的收口并落盘：把 env 内累积的 object-log 状态写成
        episode_object_logs.jsonl 中的一行（定义见 doc/episode_object_logging_mechanism.md）。

        管线位置（最后一跳）：初始化空 _episode_object_log_state → reset 写入
        bin_list / cube_list / target_cube_list → step 中追加 swap_events 并在
        swap_contact_state 累计碰撞 → close 时调用本方法收拢为 JSONL。

        步骤简述：
        1. 若 _episode_object_log_flushed 已置位则返回，避免重复写入。
        2. 从 swap_contact_state 取摘要，并把 summary 作为 close 阶段的补充信息。
        3. objectlog.flush_episode_object_log 组装 record 并追加落盘。
        """
        if self._episode_object_log_flushed:
            return

        unwrapped = self.env.unwrapped
        contact_summary = None
        swap_contact_state = getattr(unwrapped, "swap_contact_state", None)
        if swap_contact_state is not None:
            contact_summary = swapContact.get_swap_contact_summary(swap_contact_state)

        objectlog.flush_episode_object_log(
            unwrapped,
            output_root=self.output_root,
            env_id=self.env_id,
            episode=self.episode,
            seed=self.seed,
            contact_summary=contact_summary,
        )
        self._episode_object_log_flushed = True

    def close(self):
        """
        episode 结束时的统一收尾入口。

        根据 episode_success 和 record_hdf5 的组合走三条路径：
        - 成功且启用 HDF5：写 HDF5 + 写成功视频
        - 失败且启用 HDF5：丢弃 HDF5，只保留失败视频
        - 纯视频模式：只写视频
        """
        # Generate language goal (needed for both success and failure video filenames)
        language_goal_list = []
        difficulty = getattr(self.env.unwrapped, 'difficulty', None)

        # language_goal mainly used for video naming and HDF5 metadata, needed for both failure/success
        language_goal_list = task_goal.get_language_goal(self.env, self.env_id)
        language_goal_list = self._normalize_language_goal_list(language_goal_list)
        filename_parts = self._video_build_filename_parts(language_goal_list, difficulty)
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
        video_prefix = f"{self.env_id}_ep{self.episode}_seed{self.seed}{fail_recover_suffix}"

        self._episode_object_log_flush()

        # 只有成功 episode 才保留 HDF5；失败时仍可能需要视频做排查。
        if self.record_hdf5 and self.episode_success:
            logger.debug(f"Writing {len(self.buffer)} records to HDF5...")
            episode_group_name = f"episode_{self.episode}"
            self._h5_flush_episode(
                episode_group_name,
                language_goal_list,
                difficulty,
            )

            # Save success video (if enabled). Filename contains language goal/difficulty for easy lookup
            # Note: Video save failure should not affect HDF5 data saving
            self._video_flush_episode_files(
                success=True,
                video_prefix=video_prefix,
                filename_suffix=filename_suffix,
            )

            logger.debug(f"Successfully saved episode {self.episode}")
        elif self.record_hdf5:
            logger.debug(f"Episode {self.episode} failed, discarding {len(self.buffer)} records")

            # Save failure video (if enabled), but do not write HDF5
            # Note: Video save failure should not throw exception
            self._video_flush_episode_files(
                success=False,
                video_prefix=video_prefix,
                filename_suffix=filename_suffix,
            )

            # If episode failed, delete created group (if any)
            episode_group_name = f"episode_{self.episode}"
            if episode_group_name in self.h5_file:
                del self.h5_file[episode_group_name]
                logger.debug(f"Deleted episode group: {episode_group_name}")

        else:
            logger.debug(
                f"Episode {self.episode} closed in video-only mode "
                f"({'success' if self.episode_success else 'failed'})."
            )
            self._video_flush_episode_files(
                success=self.episode_success,
                video_prefix=video_prefix,
                filename_suffix=filename_suffix,
            )

        # Clear buffer to prevent repeated writing if close called multiple times
        self.buffer.clear()
        self.video_frames.clear()
        self.no_object_video_frames.clear()

        # Close HDF5 file
        if self.h5_file:
            self.h5_file.close()

        return super().close()
