import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

import gymnasium as gym
import h5py
import numpy as np
import sapien.physx as physx
import torch
import cv2
import colorsys

from mani_skill import get_commit_info
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
from ..HistoryBench_env.util import task_goal
class FailsafeTimeout(RuntimeError):
    """Raised when the HistoryBench failsafe stops an episode early."""
    pass


class HistoryBenchRecordWrapper(gym.Wrapper):
    """Wrapper that logs HistoryBench rollouts into HDF5 and optional videos."""
    def __init__(self, env,
     HistoryBench_dataset=None,HistoryBench_env=None,HistoryBench_episode=None,HistoryBench_seed=None,save_video=False):
        # 先初始化父类，确保 self.env 存在
        super().__init__(env)
        self.unwrapped.use_demonstrationwrapper=False


        # 保存配置为自身属性，避免触发 __getattr__

        self.HistoryBench_dataset = HistoryBench_dataset
        self.HistoryBench_episode = HistoryBench_episode
        self.HistoryBench_env = HistoryBench_env
        self.HistoryBench_seed = HistoryBench_seed
        self.save_video = save_video



        # Track whether the failsafe has been triggered to avoid repeated exceptions
        self._failsafe_triggered = False

        # 新增：用于暂存数据的缓冲区，在 write() 前批量写入
        self.buffer = []
        self.episode_success = False

        # Cache for subgoal segment tracking
        self.previous_subgoal_segment = None
        self.current_subgoal_segment_filled = None
        self.segmentation_points = []  # Cache for segmentation center point(s)

        # 视频缓冲区
        self.video_frames = []  # 存储组合后的视频帧
        self.no_object_video_frames = []  # 视频帧中缺失目标时单独保存

        self.h5_file = None

        if not self.HistoryBench_dataset:
            raise ValueError("HistoryBenchRecord=True 需要提供 HistoryBench_dataset 路径")

        # 创建 HDF5 文件夹；允许用户直接传单个 h5 文件或上层目录，自动推导输出路径
        base_path = Path(self.HistoryBench_dataset).resolve()
        if base_path.suffix == '.h5' or base_path.suffix == '.hdf5':
            # 如果提供的是文件路径，使用其父目录
            self.output_root = base_path.parent
            hdf5_folder_name = base_path.stem + "_hdf5_files"
        else:
            # 如果提供的是目录路径，直接使用
            self.output_root = base_path
            hdf5_folder_name = "hdf5_files"

        # 创建保存 HDF5 文件的文件夹
        self.hdf5_dir = self.output_root / hdf5_folder_name
        self.hdf5_dir.mkdir(parents=True, exist_ok=True)

        # HDF5 文件保存在新创建的文件夹中
        h5_filename = f"{self.HistoryBench_env}_ep{self.HistoryBench_episode}_seed{self.HistoryBench_seed}.h5"
        self.dataset_path = self.hdf5_dir / h5_filename

        # 按照 env/episode/seed 约定生成唯一文件名，便于后续批量分析
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

                # Color map: assign different colors to different segmentation IDs
        # 颜色查找表在初始化时生成一次，避免 step 中不断构造
        def generate_color_map(n=100, s_min=0.70, s_max=0.95, v_min=0.78, v_max=0.95):
            """
            生成 1..n 的颜色字典，值为 [R,G,B]（0-255）。
            - 色调使用黄金比例步进，避免聚堆
            - 饱和度/亮度做小周期波动，增强可分性
            """
            phi = 0.6180339887498948  # 黄金比例步长
            color_map = {}
            for i in range(1, n + 1):
                h = (i * phi) % 1.0
                s = s_min + (s_max - s_min) * ((i % 7) / 6)        # 7 步循环饱和度
                v = v_min + (v_max - v_min) * (((i * 3) % 5) / 4)  # 5 步循环亮度
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                color_map[i] = [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))]
            return color_map

        # 用法
        color_map = generate_color_map(10000)
        #color_map[16] = [0, 0, 0]  # 固定16为黑色 桌面
        self.color_map=color_map

    def _add_red_border(self, frame, border_width=10):
        """Add a red border around the frame."""
        frame_with_border = frame.copy()
        # Add red border (RGB: 255, 0, 0)
        frame_with_border[:border_width, :] = [255, 0, 0]  # Top
        frame_with_border[-border_width:, :] = [255, 0, 0]  # Bottom
        frame_with_border[:, :border_width] = [255, 0, 0]  # Left
        frame_with_border[:, -border_width:] = [255, 0, 0]  # Right
        return frame_with_border

    def _add_text_to_frame(self, frame, text, position='top_right'):
        """Create a padded text area following the requested wrapping logic and stack it above the frame.

        Args:
            frame: The image frame to add text to
            text: Either a single string or a list of strings. Each list item will be displayed on separate lines.
            position: Position parameter (retained for compatibility)
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

    def step(self, action):
        self.no_object_flag=False
        obs, reward, terminated, truncated, info = super().step(action)


        # 解析原始观测：RGB、分割 Mask 全部保留 torch->numpy 之后的数据，确保可直接写入 HDF5
        base_camera_frame = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
        wrist_camera_frame = obs['sensor_data']['hand_camera']['rgb'][0].cpu().numpy()
        segmentation=obs['sensor_data']['base_camera']['segmentation'].cpu().numpy()[0]

        #generate segmentation
        # current_segment 可能是单个 Actor 或 Actor 列表，这里统一展开，方便对齐后续 placeholder
        current_segment = getattr(self, "current_segment", None)
        if isinstance(current_segment, (list, tuple)):
            active_segments = list(current_segment)
        elif current_segment is None:
            active_segments = []
        else:
            active_segments = [current_segment]

        # segment_ids_by_index 会记录 “第几个 segment -> 对应的 seg id 列表”，用于计算独立中心点
        segment_ids_by_index = {idx: [] for idx in range(len(active_segments))}
        self.vis_obj_id_list=[]
        table_id = None
        for obj_id, obj in sorted(self.segmentation_id_map.items()):
            if active_segments:
                for idx, target in enumerate(active_segments):
                    if obj is target:
                        self.vis_obj_id_list.append(obj_id)
                        segment_ids_by_index[idx].append(obj_id)
                        break
                            
            if obj.name=='table-workspace':
                table_id = obj_id
                self.color_map[table_id] = [0, 0, 0]  # Set table color to black
        # segmentation_result 只保留当前任务关心的对象，其他像素置零，方便后续定位中心点
        segmentation_result =np.where(np.isin(segmentation,self.vis_obj_id_list), segmentation, 0)
        segmentation_2d = segmentation.squeeze() if segmentation.ndim > 2 else segmentation
        segmentation_result_2d = segmentation_result.squeeze() if segmentation_result.ndim > 2 else segmentation_result


        # Add center coordinates to current_subgoal_segment within <>
        # Only update if current_subgoal_segment has changed from previous
        current_subgoal_segment = getattr(self.unwrapped, 'current_subgoal_segment', None)

        if current_subgoal_segment != self.previous_subgoal_segment:
            # 仅在切换子目标时重新计算中心点，避免每帧都完整扫描
            # Subgoal has changed, recalculate segmentation points and update filled version
            def compute_center_from_ids(self,segmentation_mask, ids):
                """在二维 segmentation mask 中求指定 ID 集合的像素中心，返回[y,x]"""
                if not ids:
                    return None
                mask = np.isin(segmentation_mask, ids)
                if not np.any(mask):
                    self.no_object_flag=True
                    return None
                coords = np.argwhere(mask)
                if coords.size == 0:
                    return None
                center_y = int(coords[:, 0].mean())
                center_x = int(coords[:, 1].mean())
                return [center_y, center_x]

            segment_centers = []
            if active_segments:
                for idx in range(len(active_segments)):
                    segment_centers.append(
                        compute_center_from_ids(
                            self,segmentation_2d, segment_ids_by_index.get(idx, [])
                        )
                    )
            else:
                segment_centers.append(
                    compute_center_from_ids(self,segmentation_2d, self.vis_obj_id_list)
                )

            # 记录所有 segment 各自的中心坐标，便于给多目标任务标记不同点
            self.segmentation_points = [
                center for center in segment_centers if center is not None
            ]

            if current_subgoal_segment:
                import re

                subgoal_text = getattr(self, 'current_task_name', 'Unknown')
                seg_shape = (
                    segmentation_result_2d.shape
                    if segmentation_result_2d.ndim >= 2
                    else (256, 256)
                )
                normalized_centers = []
                for center in segment_centers:
                    if center is None:
                        normalized_centers.append(None)
                        continue
                    center_y, center_x = center
                    denom_y = max(seg_shape[0] - 1, 1)
                    denom_x = max(seg_shape[1] - 1, 1)
                    norm_y = min(max(center_y / denom_y, 0.0), 1.0)
                    norm_x = min(max(center_x / denom_x, 0.0), 1.0)
                    normalized_centers.append(f'<{norm_y:.4f}, {norm_x:.4f}>')

                # 子目标字符串里可能出现多个 <>，依次替换为对应 segment 的中心坐标（不足则保留原文本）
                placeholder_pattern = re.compile(r'<[^>]*>')
                placeholders = list(placeholder_pattern.finditer(current_subgoal_segment))
                placeholder_count = len(placeholders)
                if placeholder_count > 0 and normalized_centers:
                    replacements = normalized_centers.copy()
                    if len(replacements) == 1 and placeholder_count > 1:
                        replacements = replacements * placeholder_count
                    elif len(replacements) < placeholder_count:
                        replacements.extend([None] * (placeholder_count - len(replacements)))

                    missing_placeholder = False
                    new_text_parts = []
                    last_idx = 0
                    for idx, match in enumerate(placeholders):
                        new_text_parts.append(current_subgoal_segment[last_idx:match.start()])
                        replacement_text = replacements[idx]
                        if replacement_text is None:
                            # new_text_parts.append(match.group(0))  # 原逻辑：保留占位符本身
                            missing_placeholder = True
                        else:
                            new_text_parts.append(replacement_text)
                        last_idx = match.end()
                    new_text_parts.append(current_subgoal_segment[last_idx:])
                    # 缺失时用子目标文本替换整个文本
                    self.current_subgoal_segment_filled = subgoal_text if missing_placeholder else ''.join(new_text_parts)
                else:
                    self.current_subgoal_segment_filled = current_subgoal_segment
            else:
                self.current_subgoal_segment_filled = current_subgoal_segment

            # Update the previous subgoal
            self.previous_subgoal_segment = current_subgoal_segment
        # else: keep both segmentation_points and current_subgoal_segment_filled unchanged
           

        current_task=self.current_task_name if hasattr(self, 'current_task_name') else "Unknown"
        if self.save_video and current_task!='NO RECORD':
                
            # 如果是 demonstration 任务，添加红色边框（仅用于视频，不影响HDF5）
            is_demonstration = getattr(self, 'current_task_demonstration', False)
            subgoal_text = getattr(self, 'current_task_name', 'Unknown')



            # Use deepcopy to avoid modifying original frames that will be saved to HDF5
            base_camera_frame_for_video = copy.deepcopy(base_camera_frame)
            wrist_camera_frame_for_video = copy.deepcopy(wrist_camera_frame)
            segmentation_for_video=copy.deepcopy(segmentation)
            segmentation_result_for_video=copy.deepcopy(segmentation_result)


            if base_camera_frame_for_video.shape[:2] != wrist_camera_frame_for_video.shape[:2]:
                wrist_camera_frame_for_video = cv2.resize(
                    wrist_camera_frame_for_video,
                    (base_camera_frame_for_video.shape[1], base_camera_frame_for_video.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            # Convert segmentation masks to RGB for visualization
            # Create colormap for segmentation visualization
            segmentation_vis = np.zeros((*segmentation_for_video.shape[:2], 3), dtype=np.uint8)
            segmentation_result_vis = np.zeros((*segmentation_result_for_video.shape[:2], 3), dtype=np.uint8)



            # Ensure segmentation masks are 2D
            seg_2d = segmentation_for_video.squeeze() if segmentation_for_video.ndim > 2 else segmentation_for_video
            seg_result_2d = segmentation_result_for_video.squeeze() if segmentation_result_for_video.ndim > 2 else segmentation_result_for_video

            # Apply colors to segmentation_vis
            for seg_id in np.unique(seg_2d):
                if seg_id > 0:
                    color = self.color_map.get(seg_id, [255, 255, 255])  # Default white for unmapped IDs
                    mask = seg_2d == seg_id
                    segmentation_vis[mask] = color

            # Apply colors to segmentation_result_vis (same color map)
            # segmentation_result_vis 仅渲染与当前任务相关的对象，便于展示 grounding 结果
            for seg_id in np.unique(seg_result_2d):
                if seg_id > 0:
                    color = self.color_map.get(seg_id, [255, 255, 255])  # Default white for unmapped IDs
                    mask = seg_result_2d == seg_id
                    segmentation_result_vis[mask] = color

            # Resize segmentation visualizations to match camera frame size
            if segmentation_vis.shape[:2] != base_camera_frame_for_video.shape[:2]:
                segmentation_vis = cv2.resize(
                    segmentation_vis,
                    (base_camera_frame_for_video.shape[1], base_camera_frame_for_video.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            if segmentation_result_vis.shape[:2] != base_camera_frame_for_video.shape[:2]:
                segmentation_result_vis = cv2.resize(
                    segmentation_result_vis,
                    (base_camera_frame_for_video.shape[1], base_camera_frame_for_video.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            # Plot segmentation points onto target frame with a red dot (diameter 5px)
            diameter = 5
            target_for_video = copy.deepcopy(base_camera_frame_for_video)
            if self.segmentation_points:
                for center_y, center_x in self.segmentation_points:
                    cv2.circle(target_for_video, (center_x, center_y), diameter, (255, 0, 0), -1)

            # 最终视频帧拼接结构：base | wrist | 原 segmentation | filtered segmentation | base+红点
            combined=np.hstack([base_camera_frame_for_video, wrist_camera_frame_for_video,
                              segmentation_vis, segmentation_result_vis,target_for_video])
            
            # combined=np.hstack([base_camera_frame_for_video, wrist_camera_frame_for_video,
            #                   ])



            if is_demonstration:
                combined = self._add_red_border(combined)


            language_goal = task_goal.get_language_goal(self.env, self.HistoryBench_env)
            combined = self._add_text_to_frame(combined, [language_goal,subgoal_text,self.current_subgoal_segment_filled], position='top_right')
            #wrist_camera_frame_for_video = self._add_text_to_frame(wrist_camera_frame_for_video, language_goal, position='top_right')

            self.video_frames.append(combined)

            if self.no_object_flag==True:
                self.no_object_video_frames.append(combined)

            #print(self.current_task_name)

            # 暂存数据到缓冲区而不是直接写入HDF5（使用原始帧，无边框）
            record_timestep = len(self.buffer)
            # record_data 中同时保存观测、行为、语言信息，write() 时用 record_timestep 还原顺序
            #todo:在这里获取robot的末端执行器速度 并打印
            print()
            record_data = {
                'record_timestep': record_timestep,
                'robot_endeffector_p': self.agent.robot.links[9].pose.p.cpu().numpy(),
                'robot_endeffector_q': self.agent.robot.links[9].pose.q.cpu().numpy(),
                'image': base_camera_frame,
                'wrist_image': wrist_camera_frame,
                'action': action,
                'state': self.agent.robot.qpos.cpu().numpy() if hasattr(self.agent.robot.qpos, 'cpu') else self.agent.robot.qpos,
                'subgoal': self.current_task_name if hasattr(self, 'current_task_name') else "Unknown",
                'demonstration': self.current_task_demonstration if hasattr(self, 'current_task_demonstration') else False,
                'segmentation':segmentation,
                'segmentation_result':segmentation_result,
                'subgoal_grounded': self.current_subgoal_segment_filled
            }

            self.buffer.append(record_data)


        # 检查 episode 是否成功
        if terminated.any():
            if info.get("success") == torch.tensor([True]) or (isinstance(info.get("success"), torch.Tensor) and info.get("success").item()):
                self.episode_success = True
                print("Episode success detected, data will be saved")
            else:
                self.episode_success = False
                print("Episode failed, data will be discarded")

        # Failsafe: enforce a hard cap on episode length so planners can't run forever
        # 仍使用英文注释保留原含义：当 planner 卡住不结束时强制截断，保护录制流程
        fail_safe_limit = 3000
        env_steps = int(getattr(self.env.unwrapped, "elapsed_steps", getattr(self.env, "elapsed_steps", 0)))
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
       
        # language_goal 主要用于视频命名和 HDF5 中的 metadata，失败/成功都需要
        language_goal=task_goal.get_language_goal(self.env,self.HistoryBench_env)
        sanitized_goal = language_goal.replace(" ", "_").replace("/", "_") if language_goal else "no_goal"
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

        # 只有在 episode 成功时才写入数据到 HDF5
        if self.episode_success:
            print(f"Writing {len(self.buffer)} records to HDF5...")

            # HDF5 层级: env_xxx / episode_xxx / record_timestep_xxx，便于按环境和轮次检索
            env_group_name = f"env_{self.HistoryBench_env}"
            env_group = self.h5_file.require_group(env_group_name)
            episode_group_name = f"episode_{self.HistoryBench_episode}"
            if episode_group_name in env_group:
                del env_group[episode_group_name]
            episode_group = env_group.create_group(episode_group_name)

            # 写入所有缓冲的数据
            for record_data in self.buffer:
                record_timestep = record_data['record_timestep']
                if isinstance(record_timestep, (torch.Tensor, np.ndarray)):
                    record_timestep = int(record_timestep.item())
                else:
                    record_timestep = int(record_timestep)

                base_group_name = f"record_timestep_{record_timestep}"
                group_name = base_group_name
                duplicate_index = 1
                # Avoid collisions when multiple records share the same timestep
                while group_name in episode_group:
                    group_name = f"{base_group_name}_dup{duplicate_index}"
                    duplicate_index += 1

                ts_group = episode_group.create_group(group_name)

                ts_group.create_dataset("record_timestep", data=record_timestep)

                #ts_group.create_dataset("real_timestep", data=record_data['real_timestep'])
                ts_group.create_dataset("robot_endeffector_p", data=record_data['robot_endeffector_p'])
                ts_group.create_dataset("robot_endeffector_q", data=record_data['robot_endeffector_q'])
                ts_group.create_dataset("image", data=record_data['image'])
                ts_group.create_dataset("wrist_image", data=record_data['wrist_image'])
                ts_group.create_dataset("segmentation", data=record_data['segmentation'])
                ts_group.create_dataset("segmentation_result", data=record_data['segmentation_result'])

                # 动作有可能是 None（例如 planner 尚未输出），写入字符串避免 h5py 报 dtype 错误
                if record_data['action'] is None:
                    ts_group.create_dataset("action", data="None", dtype=h5py.special_dtype(vlen=str))
                else:
                    ts_group.create_dataset("action", data=record_data['action'])

                ts_group.create_dataset("state", data=record_data['state'])

                # 处理字符串任务名
                task_name = record_data['subgoal']
                if isinstance(task_name, str):
                    task_name_encoded = task_name.encode('utf-8')
                else:
                    task_name_encoded = task_name
                ts_group.create_dataset("subgoal", data=task_name_encoded)
                

                task_name = record_data['subgoal_grounded']
                if isinstance(task_name, str):
                    task_name_encoded = task_name.encode('utf-8')
                else:
                    task_name_encoded = task_name
                ts_group.create_dataset("subgoal_grounded", data=task_name_encoded)
                
                ts_group.create_dataset("demonstration", data=record_data['demonstration'])

            # 写入 setup 信息
            setup_group = episode_group.create_group(f"setup")
            setup_group.create_dataset("seed", data=self.HistoryBench_seed)
            setup_group.create_dataset(
                    "difficulty",
                    data=difficulty,
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )

            # 记录任务列表（如果存在），便于离线复现每个 subgoal 的语义
            if hasattr(self, 'task_list'):
                tasks_group = setup_group.create_group("task_list")
                for i, task_entry in enumerate(self.task_list):
                    if isinstance(task_entry, dict):
                        task_name = task_entry.get("name", "Unknown")
                        demonstration = bool(task_entry.get("demonstration", False))
                    else:
                        if len(task_entry) == 2:
                            _, task_name = task_entry
                            demonstration = False
                        else:
                            _, task_name, demonstration = task_entry

                    task_name_encoded = task_name.encode('utf-8')
                    tasks_group.create_dataset(f"task_{i}_name", data=task_name_encoded)
                    tasks_group.create_dataset(f"task_{i}_demonstration", data=demonstration)

            if language_goal:
                setup_group.create_dataset("language goal", data=language_goal)

            # 保存成功视频（如果启用）。文件名包含语言目标/难度，方便查找
            if self.save_video and len(self.video_frames) > 0:
                videos_dir = Path("/data/hongzefu/videos")
                videos_dir.mkdir(parents=True, exist_ok=True)
                combined_video_path = videos_dir / f"{self.HistoryBench_env}_ep{self.HistoryBench_episode}_{filename_suffix}.mp4"

                with imageio.get_writer(combined_video_path.as_posix(), fps=60, codec="libx264", quality=8) as writer:
                    for combined_frame in self.video_frames:
                        writer.append_data(combined_frame)
                print(f"Saved combined video to {combined_video_path}")
            if self.save_video and len(self.no_object_video_frames) > 0:
                videos_dir = Path("/data/hongzefu/videos")
                videos_dir.mkdir(parents=True, exist_ok=True)
                no_object_video_path = videos_dir / f"NO_OBJECT_{self.HistoryBench_env}_ep{self.HistoryBench_episode}_{filename_suffix}.mp4"

                with imageio.get_writer(no_object_video_path.as_posix(), fps=60, codec="libx264", quality=8) as writer:
                    for combined_frame in self.no_object_video_frames:
                        writer.append_data(combined_frame)
                print(f"Saved no-object video to {no_object_video_path}")

            print(f"Successfully saved episode {self.HistoryBench_episode}")
        else:
            print(f"Episode {self.HistoryBench_episode} failed, discarding {len(self.buffer)} records")

            # 保存失败视频（如果启用），但不写入 HDF5
            if self.save_video and len(self.video_frames) > 0:
                videos_dir = Path("videos")
                videos_dir.mkdir(parents=True, exist_ok=True)
                # Add FAILED_ prefix to the filename
                combined_video_path = videos_dir / f"FAILED_{self.HistoryBench_env}_ep{self.HistoryBench_episode}_{filename_suffix}.mp4"

                with imageio.get_writer(combined_video_path.as_posix(), fps=60, codec="libx264", quality=8) as writer:
                    for combined_frame in self.video_frames:
                        writer.append_data(combined_frame)
                print(f"Saved failed episode video to {combined_video_path}")
            if self.save_video and len(self.no_object_video_frames) > 0:
                videos_dir = Path("videos")
                videos_dir.mkdir(parents=True, exist_ok=True)
                no_object_video_path = videos_dir / f"FAILED_NO_OBJECT_{self.HistoryBench_env}_ep{self.HistoryBench_episode}_{filename_suffix}.mp4"

                with imageio.get_writer(no_object_video_path.as_posix(), fps=60, codec="libx264", quality=8) as writer:
                    for combined_frame in self.no_object_video_frames:
                        writer.append_data(combined_frame)
                print(f"Saved failed no-object video to {no_object_video_path}")

            # 如果 episode 失败，删除已经创建的 group（如果有的话）
            env_group_name = f"env_{self.HistoryBench_env}"
            episode_group_name = f"episode_{self.HistoryBench_episode}"
            if env_group_name in self.h5_file:
                env_group = self.h5_file[env_group_name]
                if episode_group_name in env_group:
                    del env_group[episode_group_name]
                    print(f"Deleted episode group: {episode_group_name}")

        # 清空缓冲区，防止 close 被多次调用时重复写入
        self.buffer.clear()
        self.video_frames.clear()
        self.no_object_video_frames.clear()

        # 关闭 HDF5 文件
        if self.h5_file:
            self.h5_file.close()

        return super().close()
