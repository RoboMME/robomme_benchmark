import copy
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

from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.motionplanner_stick import PandaStickMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from ..HistoryBench_env.util import task_goal

from ..HistoryBench_env.util import reset_panda


def load_episode_metadata(metadata_path: Union[str, Path, None]) -> Dict[Tuple[str, int], Dict[str, object]]:
    """
    从 JSON 文件读取每集的元数据（metadata）；如果缺失或无效则返回空字典。
    用于恢复特定 episode 的配置（如 seed、难度等）。
    """

    metadata_index: Dict[Tuple[str, int], Dict[str, object]] = {}
    if not metadata_path:
        return metadata_index

    path = Path(metadata_path)
    if not path.exists():
        print(f"Metadata file not found, skipping: {path}")
        return metadata_index

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # pragma: no cover - informational logging only
        print(f"Failed to read metadata {path}: {exc}")
        return metadata_index

    default_task = str(payload.get("env_id") or "").strip()
    for record in payload.get("records", []):
        task_name = str(record.get("task") or default_task or "").strip()
        episode = record.get("episode")
        if not task_name or episode is None:
            continue
        try:
            episode_idx = int(episode)
        except (TypeError, ValueError):
            continue
        metadata_index[(task_name, episode_idx)] = record

    if metadata_index:
        print(f"Loaded {len(metadata_index)} metadata records from {path}")
    else:
        print(f"No valid metadata entries found in {path}")
    return metadata_index


def get_episode_metadata(
    metadata_index: Dict[Tuple[str, int], Dict[str, object]],
    task: str,
    episode: int,
) -> Optional[Dict[str, object]]:
    """查找特定 (task, episode) 配对的元数据条目。"""

    if not metadata_index:
        return None
    return metadata_index.get((task, episode))


class EpisodeConfigResolver:
    """
    Episode 配置解析器。
    
    辅助类，用于解析每个 episode 的种子（seed）和难度（difficulty），并构建包装好的环境。
    数据来源可以是已有的 HDF5 数据集，也可以是元数据文件。
    """

    def __init__(
        self,
        env_id: str,
        dataset: Optional[h5py.File],
        metadata_path: Union[str, Path, None],
        render_mode: str,
        gui_render: bool,
        max_steps_without_demonstration: int,
        save_video: bool = False,
    ):
        self.env_id = env_id
        self.render_mode = render_mode
        self.gui_render = gui_render
        self.max_steps_without_demonstration = max_steps_without_demonstration
        self.save_video = save_video
        self.metadata_index = load_episode_metadata(metadata_path)

        self.env_dataset = None
        if dataset is not None:
            env_group = f"env_{env_id}"
            if env_group not in dataset:
                raise KeyError(f"Dataset missing group '{env_group}'")
            self.env_dataset = dataset[env_group]

    def resolve_episode(self, episode: int):
        """根据 dataset 或 metadata 解析 episode 的配置。"""
        episode_dataset = None
        seed = None
        difficulty_hint = None

        if self.env_dataset is not None:
            episode_key = f"episode_{episode}"
            if episode_key not in self.env_dataset:
                raise KeyError(f"No data for episode {episode} in env_{self.env_id}")

            episode_dataset = self.env_dataset[episode_key]
            seed = int(episode_dataset["setup"]["seed"][()])

        metadata = get_episode_metadata(self.metadata_index, self.env_id, episode)
        if metadata:
            metadata_seed = metadata.get("seed")
            if metadata_seed is not None:
                try:
                    seed = int(metadata_seed)
                except (TypeError, ValueError):
                    print(f"[{self.env_id}] Invalid metadata seed for episode {episode}: {metadata_seed}")
            difficulty_hint = metadata.get("difficulty")

        return seed, difficulty_hint, episode_dataset

    def make_env_for_episode(self, episode: int):
        """为特定 episode 创建并配置环境。"""
        seed, difficulty_hint, episode_dataset = self.resolve_episode(episode)
        env_kwargs = dict(
            obs_mode="rgb+depth+segmentation",
            control_mode="pd_joint_pos",
            render_mode=self.render_mode,
            reward_mode="dense",
            max_episode_steps=99999,
        )
        if seed is not None:
            env_kwargs["HistoryBench_seed"] = seed
        if difficulty_hint:
            env_kwargs["HistoryBench_difficulty"] = difficulty_hint
        seed_desc = seed if seed is not None else "default"
        difficulty_str = f", difficulty={difficulty_hint}" if difficulty_hint else ""
        print(f"[{self.env_id}] Episode {episode}: seed={seed_desc}{difficulty_str}")

        env = gym.make(self.env_id, **env_kwargs)
        env = DemonstrationWrapper(
            env,
            max_steps_without_demonstration=self.max_steps_without_demonstration,
            gui_render=self.gui_render,
            save_video=self.save_video,
        )
        return env, episode_dataset, seed, difficulty_hint


class DemonstrationWrapper(gym.Wrapper):
    """
    Demonstration 包装器。
    
    主要功能：
    1. 在环境 reset 后自动生成演示轨迹（Trajectory），使用 Motion Planner。
    2. 记录演示过程中的帧、动作、状态等数据。
    3. 支持视频录制，可视化演示过程。
    """
    def __init__(self, env,max_steps_without_demonstration,gui_render,save_video=False):
        self.max_steps_without_demonstration=max_steps_without_demonstration
        self.gui_render=gui_render
        self.save_video=save_video

        # 先初始化父类，确保 self.env 存在
        super().__init__(env)
        self.unwrapped.use_demonstrationwrapper=True


        # self.video_path = video_path
        self.frames = []
        self.wrist_frames = []
        self.actions = []
        self.states = []
        self.subgoal=[]
        self.subgoal_grounded=[]
        self.video_frames=[]
        self.no_object_video_frames=[]
        self.demonstration_record_traj = False
        self.velocity=[]


        self.steps_without_demonstration=0
        self._doing_extra_step = False  # avoid re-entrance when adding highlight step
        self.demonstration_data = None
        self.previous_subgoal_segment = None
        self.current_subgoal_segment_filled = None
        self.segmentation_points = []
        self.no_object_flag = False
        self.episode_success = False  # 初始化 episode_success 属性

        # 与 RecordWrapper 保持一致的分割上色表
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
        obs = super().reset(**kwargs)
        # 重置 episode_success 状态
        self.episode_success = False
        # Reset 后自动调用 get_demonstration_trajectory 生成演示轨迹
        # 这确保了每次 episode 开始时都有对应的演示数据
        self.demonstration_data = self.get_demonstration_trajectory()

        if self.unwrapped.spec.id=="PatternLock" or self.unwrapped.spec.id == "RouteStick":
             gripper="stick"
        else:
             gripper=None
        if self.unwrapped.spec.id=="PatternLock" or self.unwrapped.spec.id == "RouteStick": 
            action=self.unwrapped.swing_qpos#对于这两个环境 需要得到在线生成的action！
        else:
            action=reset_panda.get_reset_panda_param("action",gripper=gripper)
        

        # Only add the extra step when a video demonstration task exists (no length check)
        has_video_demo = any(
            task.get("demonstration", False)
            for task in getattr(self, "task_list", [])
        )

        obs, _, _, _, info = self.step(action)

        return obs, info


    def _add_red_border(self, frame, border_width=10):
        frame_with_border = frame.copy()
        frame_with_border[:border_width, :] = [255, 0, 0]
        frame_with_border[-border_width:, :] = [255, 0, 0]
        frame_with_border[:, :border_width] = [255, 0, 0]
        frame_with_border[:, -border_width:] = [255, 0, 0]
        return frame_with_border

    def _add_text_to_frame(self, frame, text, position='top_right'):
        if not text:
            return frame

        if isinstance(text, str):
            text_list = [text]
        else:
            text_list = text

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        max_width = max(1, frame.shape[1] - 20)

        lines = []
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

        return np.vstack((text_area, frame))


    def step(self, action):

        self.no_object_flag = False
        obs, reward, terminated, truncated, info = super().step(action)

        base_camera_frame = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
        wrist_camera_frame = obs['sensor_data']['hand_camera']['rgb'][0].cpu().numpy()

        segmentation = None
        segmentation_result = None
        segmentation_2d = None
        segmentation_result_2d = None

        try:
            segmentation = obs['sensor_data']['base_camera']['segmentation']
        except Exception:
            segmentation = None

        if segmentation is not None:
            if hasattr(segmentation, "cpu"):
                segmentation = segmentation.cpu().numpy()
            segmentation = np.asarray(segmentation)
            if segmentation.ndim > 2:
                segmentation = segmentation[0]
            segmentation_2d = segmentation.squeeze()

            current_segment = getattr(self, "current_segment", None)
            if isinstance(current_segment, (list, tuple)):
                active_segments = list(current_segment)
            elif current_segment is None:
                active_segments = []
            else:
                active_segments = [current_segment]

            segment_ids_by_index = {idx: [] for idx in range(len(active_segments))}
            vis_obj_id_list = []
            segmentation_id_map = getattr(self, "segmentation_id_map", None)
            if isinstance(segmentation_id_map, dict):
                for obj_id, obj in sorted(segmentation_id_map.items()):
                    if active_segments:
                        for idx, target in enumerate(active_segments):
                            if obj is target:
                                vis_obj_id_list.append(obj_id)
                                segment_ids_by_index[idx].append(obj_id)
                                break
                    if getattr(obj, "name", None) == 'table-workspace':
                        self.color_map[obj_id] = [0, 0, 0]

            if vis_obj_id_list:
                segmentation_result = np.where(np.isin(segmentation_2d, vis_obj_id_list), segmentation_2d, 0)
            else:
                segmentation_result = segmentation_2d
            segmentation_result_2d = segmentation_result.squeeze()

        # 处理子目标分割和占位符填充逻辑
        current_subgoal_segment = getattr(self.unwrapped, 'current_subgoal_segment', None)
        if current_subgoal_segment != self.previous_subgoal_segment:
            def compute_center_from_ids(wrapper_self, segmentation_mask, ids):
                """计算指定 ID 集合的分割掩码中心点。"""
                if not ids:
                    return None
                mask = np.isin(segmentation_mask, ids)
                if not np.any(mask):
                    wrapper_self.no_object_flag = True
                    return None
                coords = np.argwhere(mask)
                if coords.size == 0:
                    return None
                center_y = int(coords[:, 0].mean())
                center_x = int(coords[:, 1].mean())
                return [center_y, center_x]

            segment_centers = []
            if segmentation_result_2d is not None and segmentation_2d is not None:
                if segmentation_result_2d is not None and segmentation_result_2d.size > 0:
                    if 'active_segments' in locals() and active_segments:
                        for idx in range(len(active_segments)):
                            segment_centers.append(
                                compute_center_from_ids(self, segmentation_2d, segment_ids_by_index.get(idx, []))
                            )
                    else:
                        target_ids = vis_obj_id_list if 'vis_obj_id_list' in locals() else []
                        segment_centers.append(
                            compute_center_from_ids(self, segmentation_2d, target_ids)
                        )
                self.segmentation_points = [center for center in segment_centers if center is not None]

                # 如果存在子目标文本，尝试替换其中的占位符（如 <target>）为具体坐标
                if current_subgoal_segment:
                    import re
                    subgoal_text = getattr(self, 'current_task_name', 'Unknown')
                    seg_shape = segmentation_result_2d.shape if segmentation_result_2d.ndim >= 2 else (256, 256)
                    normalized_centers = []
                    for center in segment_centers:
                        if center is None:
                            normalized_centers.append(None)
                            continue
                        center_y, center_x = center
                        # 直接写入像素坐标，不再归一化到 [0, 1]
                        normalized_centers.append(f'<{center_y}, {center_x}>')

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
                                # new_text_parts.append(match.group(0))  # 原逻辑：保留占位符
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
            else:
                self.segmentation_points = []
                self.current_subgoal_segment_filled = current_subgoal_segment

            self.previous_subgoal_segment = current_subgoal_segment

        # # Collect frames for video
        current_task=self.current_task_name if hasattr(self, 'current_task_name') else "Unknown"
        if current_task!='NO RECORD':
            image = base_camera_frame
            wrist_image = wrist_camera_frame
            state=self.agent.robot.qpos.cpu().numpy() if hasattr(self.agent.robot.qpos, 'cpu') else self.agent.robot.qpos
            end_effector_velocity = self.agent.robot.links[9].get_linear_velocity().tolist()[0] + self.agent.robot.links[9].get_angular_velocity().tolist()[0]

            subgoal_text = getattr(self, 'current_task_name', 'Unknown')
            grounded_subgoal = self.current_subgoal_segment_filled
            language_goal = task_goal.get_language_goal(self.env, self.unwrapped.spec.id)
            self.frames.append(image)
            self.wrist_frames.append(wrist_image)
            self.actions.append(action)
            self.states.append(state)
            self.velocity.append(end_effector_velocity)
            self.subgoal.append(subgoal_text)
            self.subgoal_grounded.append(grounded_subgoal)

            # 与 RecordWrapper 保持一致的视频帧组合（base | wrist | segmentation | filtered | base+dot）
            if self.save_video:
                is_demonstration = getattr(self, 'current_task_demonstration', False)
                base_frame_video = copy.deepcopy(image)
                wrist_frame_video = copy.deepcopy(wrist_image)
                segmentation_for_video = copy.deepcopy(segmentation) if segmentation is not None else np.zeros(base_frame_video.shape[:2], dtype=np.int32)
                segmentation_result_for_video = copy.deepcopy(segmentation_result) if segmentation_result is not None else np.zeros(base_frame_video.shape[:2], dtype=np.int32)

                if base_frame_video.shape[:2] != wrist_frame_video.shape[:2]:
                    wrist_frame_video = cv2.resize(
                        wrist_frame_video,
                        (base_frame_video.shape[1], base_frame_video.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )

                seg_2d = segmentation_for_video.squeeze() if segmentation_for_video.ndim > 2 else segmentation_for_video
                seg_result_2d = segmentation_result_for_video.squeeze() if segmentation_result_for_video.ndim > 2 else segmentation_result_for_video

                segmentation_vis = np.zeros((*seg_2d.shape, 3), dtype=np.uint8)
                segmentation_result_vis = np.zeros((*seg_result_2d.shape, 3), dtype=np.uint8)

                for seg_id in np.unique(seg_2d):
                    if seg_id > 0:
                        mask = seg_2d == seg_id
                        segmentation_vis[mask] = self.color_map.get(seg_id, [255, 255, 255])

                for seg_id in np.unique(seg_result_2d):
                    if seg_id > 0:
                        mask = seg_result_2d == seg_id
                        segmentation_result_vis[mask] = self.color_map.get(seg_id, [255, 255, 255])

                if segmentation_vis.shape[:2] != base_frame_video.shape[:2]:
                    segmentation_vis = cv2.resize(
                        segmentation_vis,
                        (base_frame_video.shape[1], base_frame_video.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

                if segmentation_result_vis.shape[:2] != base_frame_video.shape[:2]:
                    segmentation_result_vis = cv2.resize(
                        segmentation_result_vis,
                        (base_frame_video.shape[1], base_frame_video.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

                target_frame = copy.deepcopy(base_frame_video)
                if self.segmentation_points:
                    for center_y, center_x in self.segmentation_points:
                        cv2.circle(target_frame, (center_x, center_y), 5, (255, 0, 0), -1)

                combined = np.hstack([
                    base_frame_video,
                    wrist_frame_video,
                    segmentation_vis,
                    segmentation_result_vis,
                    target_frame,
                ])

                if is_demonstration:
                    combined = self._add_red_border(combined)
                combined = self._add_text_to_frame(
                    combined,
                    [language_goal, subgoal_text, grounded_subgoal],
                    position='top_right',
                )
                self.video_frames.append(combined)
                # if self.no_object_flag:
                #     self.no_object_video_frames.append(combined)




                #start counting from 
        if self.current_task_demonstration==False:
            self.steps_without_demonstration+=1
            if self.steps_without_demonstration>=self.max_steps_without_demonstration:
                truncated=torch.tensor([True])
            #print(self.steps_without_demonstration)


        # 检查 episode 是否成功
        if terminated.any():
            if info.get("success") == torch.tensor([True]) or (isinstance(info.get("success"), torch.Tensor) and info.get("success").item()):
                self.episode_success = True
                print("Episode success detected, data will be saved")
            else:
                self.episode_success = False
                print("Episode failed, data will be discarded")

        # 在终止时追加一次“多余动作”，动作直接复用上一次 action
        if terminated.any() and not self._doing_extra_step:
            self._doing_extra_step = True
            try:
                # 复用完整 DemonstrationWrapper.step 流程（包含记录/可视化等）

                self.step(action)
            finally:
                self._doing_extra_step = False

        return obs, reward, terminated, truncated, info

    def close(self):
        # 保存演示视频到 replay_videos 目录
        if self.save_video and len(self.video_frames)>0:
            videos_dir = Path("/data/hongzefu/dataset_generate/replay_videos")
            videos_dir.mkdir(parents=True, exist_ok=True)

            language_goal = task_goal.get_language_goal(self.env,self.unwrapped.spec.id)
            sanitized_goal = language_goal.replace(" ", "_").replace("/", "_") if language_goal else "no_goal"
            seed = getattr(self.env.unwrapped, "HistoryBench_seed", None)
            seed_tag = f"_seed{seed}" if seed is not None else ""

            if self.episode_success == True:
                video_path = videos_dir / f"DEMO_{self.unwrapped.spec.id}{seed_tag}_{sanitized_goal}.mp4"
            else:
                 video_path = videos_dir / f"DEMO_FAILED_{self.unwrapped.spec.id}{seed_tag}_{sanitized_goal}.mp4"
            try:
                with imageio.get_writer(video_path.as_posix(), fps=30, codec="libx264", quality=8) as writer:
                    for frame in self.video_frames:
                        writer.append_data(frame)
                print(f"Saved demonstration video to {video_path}")
            except (ValueError, Exception) as e:
                # 如果保存视频时出错（如帧大小不一致），保存一个空白视频
                print(f"Error saving video: {e}. Saving blank video instead.")
                # 获取第一个帧的大小，如果不存在则使用默认大小
                if len(self.video_frames) > 0:
                    first_frame = self.video_frames[0]
                    if isinstance(first_frame, np.ndarray):
                        frame_shape = first_frame.shape
                        blank_frame = np.zeros(frame_shape, dtype=first_frame.dtype)
                    else:
                        # 如果第一个帧不是numpy数组，使用默认大小
                        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # 创建一个只包含空白帧的短视频（1秒，30帧）
                with imageio.get_writer(video_path.as_posix(), fps=30, codec="libx264", quality=8) as writer:
                    for _ in range(30):
                        writer.append_data(blank_frame)
                print(f"Saved blank video to {video_path}")
        # if self.save_video and len(self.no_object_video_frames)>0:
        #     videos_dir = Path("/data/hongzefu/dataset_generate/replay_videos")
        #     videos_dir.mkdir(parents=True, exist_ok=True)
        #
        #     language_goal = task_goal.get_language_goal(self.env,self.unwrapped.spec.id)
        #     sanitized_goal = language_goal.replace(" ", "_").replace("/", "_") if language_goal else "no_goal"
        #     seed = getattr(self.env.unwrapped, "HistoryBench_seed", None)
        #     seed_tag = f"_seed{seed}" if seed is not None else ""
        #
        #     video_path = videos_dir / f"DEMO_NO_OBJECT_{self.unwrapped.spec.id}{seed_tag}_{sanitized_goal}.mp4"
        #     with imageio.get_writer(video_path.as_posix(), fps=30, codec="libx264", quality=8) as writer:
        #         for frame in self.no_object_video_frames:
        #             writer.append_data(frame)
        #     print(f"Saved demonstration no-object video to {video_path}")

        self.video_frames.clear()
        self.no_object_video_frames.clear()
        super().close()
        return None

    def get_demonstration_trajectory(self):
        """
        生成演示轨迹（Demonstration Trajectory）。
        
        流程：
        1. 根据环境 ID 选择合适的 Motion Planner（PandaArm 或 PandaStick）。
        2. 遍历任务列表（task_list），找到标记为 demonstration 的任务。
        3. 对每个演示任务，调用其 solve 回调函数，让 Motion Planner 执行规划。
        4. 记录执行过程中的帧、动作、状态等。
        5. 返回收集到的轨迹数据。
        """
        #######for video demonstration
        if self.unwrapped.spec.id=="PatternLock" or self.unwrapped.spec.id == "RouteStick":
                    planner = PandaStickMotionPlanningSolver(
                            self,
                            debug=False,
                            vis=False,
                            base_pose= self.unwrapped.agent.robot.pose,
                            visualize_target_grasp_pose=False,
                            print_env_info=False,
                            joint_vel_limits=0.3,
                        )
        else:
                    planner = PandaArmMotionPlanningSolver(
                            self,
                            debug=False,
                            vis=False,
                            base_pose=self.unwrapped.agent.robot.pose,
                            visualize_target_grasp_pose=False,
                            print_env_info=False,
                        )
        tasks = getattr(self, 'task_list', [])

        self.task_list_length = len(tasks)#记录任务列表长度
        print(f"Task list length: {self.task_list_length}")

        demonstration_tasks = [task for task in tasks if task.get("demonstration", False)]
        self.non_demonstration_task_length = len(tasks) - len(demonstration_tasks)  # 记录非demonstration任务长度
        print(f"Non-demonstration task length: {self.non_demonstration_task_length}")

        for idx, task_entry in enumerate(demonstration_tasks):
            self.unwrapped.demonstration_record_traj=True
            task_name = task_entry.get("name", f"Task {idx}")
            print(f"Executing task {idx+1}/{len(demonstration_tasks)}: {task_name}")

            solve_callable = task_entry.get("solve")
            if not callable(solve_callable):
                raise ValueError(f"Task '{task_name}' must supply a callable 'solve'.")
            
            #solve_callable(self, planner)

            evaluation = self.evaluate(solve_complete_eval=True)
            solve_callable(self, planner)
            evaluation = self.evaluate(solve_complete_eval=True)
        #######for video demonstration

        

        language_goal=task_goal.get_language_goal(self.env,self.env.unwrapped.spec.id)
        self.unwrapped.demonstration_record_traj=False #标记video结束 正常开始判断subgoal
        return {
            'frames': self.frames,
            'wrist_frames': self.wrist_frames,
            'actions': self.actions,
            'states': self.states,
            'velocity':self.velocity,
            'subgoal':self.subgoal,
            'subgoal_grounded': self.subgoal_grounded,  # 所有步骤《》占位符填充为坐标后的文本序列
            'language goal':language_goal,
        }
