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
from ..HistoryBench_env.util.segmentation_utils import (
    process_segmentation,
    create_segmentation_visuals,
)
class FailsafeTimeout(RuntimeError):
    """当 HistoryBench failsafe 提前终止 episode 时抛出的异常。"""
    pass


class HistoryBenchRecordWrapper(gym.Wrapper):
    """
    HistoryBench 记录包装器。
    
    主要功能：
    1. 将 HistoryBench 的 rollout 数据（观测、动作、状态等）记录到 HDF5 文件中。
    2. 生成包含 base/wrist 相机视角、分割掩码、可视化结果的合成视频。
    3. 处理分割（segmentation）逻辑，包括目标识别和中心点计算。
    """
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



        # 追踪 failsafe 是否已被触发，避免重复抛出异常
        self._failsafe_triggered = False

        # 新增：用于暂存数据的缓冲区，在 write() 前批量写入
        # 避免每一步都进行 IO 操作，提高效率
        self.buffer = []
        self.episode_success = False

        # 用于子目标分割跟踪的缓存
        self.previous_subgoal_segment = None
        self.current_subgoal_segment_filled = None
        self.segmentation_points = []  # 缓存分割中心点
        self.previous_subgoal_segment_online = None
        self.current_subgoal_segment_online_filled = None
        self.segmentation_points_online = []  # 缓存即使在线分割目标点

        # 视频缓冲区
        self.video_frames = []  # 存储组合后的视频帧
        self.no_object_video_frames = []  # 视频帧中缺失目标时单独保存，用于调试

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
        # 使用 'a' 模式打开，如果文件损坏则删除重建
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

        # 颜色查找表在初始化时生成一次，避免 step 中不断构造
        # 用于给不同的分割 ID 分配固定颜色
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
        """给图像添加红色边框，通常用于标记 Demonstration 阶段。"""
        frame_with_border = frame.copy()
        # Add red border (RGB: 255, 0, 0)
        frame_with_border[:border_width, :] = [255, 0, 0]  # Top
        frame_with_border[-border_width:, :] = [255, 0, 0]  # Bottom
        frame_with_border[:, :border_width] = [255, 0, 0]  # Left
        frame_with_border[:, -border_width:] = [255, 0, 0]  # Right
        return frame_with_border

    def _add_text_to_frame(self, frame, text, position='top_right'):
        """
        在帧上方创建一个填充的文本区域，并根据需要自动换行。
        
        Args:
            frame: 要添加文本的图像帧
            text: 单个字符串或字符串列表。列表中的每一项将显示在不同的行上。
            position: 位置参数（保留用于兼容性，实际总是堆叠在上方）
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
        base_camera_depth = obs['sensor_data']['base_camera']['depth'][0].cpu().numpy()
        wrist_camera_frame = obs['sensor_data']['hand_camera']['rgb'][0].cpu().numpy()
        wrist_camera_depth = obs['sensor_data']['hand_camera']['depth'][0].cpu().numpy()

        base_camera_extrinsic_opencv=obs['sensor_param']['base_camera']['extrinsic_cv']
        base_camera_intrinsic_opencv=obs['sensor_param']['base_camera']['intrinsic_cv']
        base_camera_cam2world_opengl=obs['sensor_param']['base_camera']['cam2world_gl']
        wrist_camera_extrinsic_opencv=obs['sensor_param']['hand_camera']['extrinsic_cv']
        wrist_camera_intrinsic_opencv=obs['sensor_param']['hand_camera']['intrinsic_cv']
        wrist_camera_cam2world_opengl=obs['sensor_param']['hand_camera']['cam2world_gl']
        
        
        segmentation=obs['sensor_data']['base_camera']['segmentation'].cpu().numpy()[0]

        # 获取当前子目标名称和在线规划子目标名称
        current_subgoal_segment = getattr(self.unwrapped, 'current_subgoal_segment', None)
        current_subgoal_segment_online = getattr(self.unwrapped, 'current_subgoal_segment_online', None)
        current_task_name_online = getattr(self.unwrapped, 'current_task_name_online', getattr(self, 'current_task_name_online', 'Unknown'))
        
        # 处理离线规划的分割信息：生成可视化结果、计算目标中心点、填充子目标文本中的占位符
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

        # 处理在线规划的分割信息（逻辑同上，但针对 online 目标）
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
        # 注意：这里可能应该是 online 的填充结果？但原代码是复写了 self.current_subgoal_segment_online_filled
        self.current_subgoal_segment_online_filled = segmentation_output_online[
            "current_subgoal_segment_filled"
        ]
        self.no_object_flag_online = segmentation_output_online["no_object_flag"]
        self.previous_subgoal_segment_online = segmentation_output_online[
            "updated_previous_subgoal_segment"
        ]
        self.vis_obj_id_list_online = segmentation_output_online["vis_obj_id_list"]

        current_task=self.current_task_name if hasattr(self, 'current_task_name') else "Unknown"
        
        # 视频录制逻辑：仅在任务名称不为 NO RECORD 且启用视频保存时执行
        if self.save_video and current_task!='NO RECORD':
                
            # 如果是 demonstration 任务，添加红色边框（仅用于视频，不影响HDF5）
            is_demonstration = getattr(self, 'current_task_demonstration', False)
            subgoal_text = getattr(self, 'current_task_name', 'Unknown')
            subgoal_online_text = getattr(self, 'current_task_name_online', 'Unknown')



            # Use deepcopy to avoid modifying original frames that will be saved to HDF5
            base_camera_frame_for_video = copy.deepcopy(base_camera_frame)
            wrist_camera_frame_for_video = copy.deepcopy(wrist_camera_frame)
            segmentation_for_video=copy.deepcopy(segmentation)
            segmentation_result_for_video=copy.deepcopy(segmentation_result)
            segmentation_result_online_for_video=copy.deepcopy(segmentation_result_online)

            # 调整 wrist 相机图像大小以匹配 base 相机
            if base_camera_frame_for_video.shape[:2] != wrist_camera_frame_for_video.shape[:2]:
                wrist_camera_frame_for_video = cv2.resize(
                    wrist_camera_frame_for_video,
                    (base_camera_frame_for_video.shape[1], base_camera_frame_for_video.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            # 生成分割可视化图像（上色）和带红点的目标图像
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

            # 最终视频帧拼接结构：base | wrist | 原 segmentation | filtered segmentation | base+红点
            combined=np.hstack([base_camera_frame_for_video, wrist_camera_frame_for_video,
                              segmentation_vis, segmentation_result_vis,target_for_video])
            combined_online=np.hstack([base_camera_frame_for_video, wrist_camera_frame_for_video,
                              segmentation_vis_online, segmentation_result_vis_online,target_for_video_online])

            # 为第一行添加 subgoal_text 和 grounded_subgoal 文本（PLANNER 视角）
            combined = self._add_text_to_frame(combined, ["PLANNER:",subgoal_text, self.current_subgoal_segment_filled], position='top_right')

            # 为第二行添加 ONLINE: 标记和 subgoal_online_text 和 grounded_subgoal_online 文本（ONLINE 视角）
            combined_online = self._add_text_to_frame(combined_online, ["ONLINE:", subgoal_online_text, self.current_subgoal_segment_online_filled], position='top_right')

            # 将两行视频流垂直堆叠
            combined=np.vstack([combined,combined_online])

            # combined=np.hstack([base_camera_frame_for_video, wrist_camera_frame_for_video,
            #                   ])


            # 如果是演示阶段，给整个帧加红框
            if is_demonstration:
                combined = self._add_red_border(combined)


            language_goal = task_goal.get_language_goal(self.env, self.HistoryBench_env)
            combined = self._add_text_to_frame(combined, [language_goal], position='top_right')
            #wrist_camera_frame_for_video = self._add_text_to_frame(wrist_camera_frame_for_video, language_goal, position='top_right')

            self.video_frames.append(combined)

            if self.no_object_flag==True:
                self.no_object_video_frames.append(combined)

            #print(self.current_task_name)

            # 暂存数据到缓冲区而不是直接写入HDF5（使用原始帧，无边框）
            record_timestep = len(self.buffer)
            # record_data 中同时保存观测、行为、语言信息，write() 时用 record_timestep 还原顺序
            #print(f"End-effector linear velocity: {self.agent.robot.links[9].get_linear_velocity().tolist()[0]}, angular velocity: {self.agent.robot.links[9].get_angular_velocity().tolist()[0]}")
            end_effector_velocity = self.agent.robot.links[9].get_linear_velocity().tolist()[0] + self.agent.robot.links[9].get_angular_velocity().tolist()[0]

            # 处理keypoint信息：读取env中的待记录keypoint（如果存在则记录一次后清除）
            current_keypoint = None
            env_unwrapped = getattr(self.env, 'unwrapped', self.env)
            if hasattr(env_unwrapped, '_pending_keypoint') and env_unwrapped._pending_keypoint is not None:
                # 获取待记录的keypoint并立即清除，确保每个keypoint只记录一次
                current_keypoint = env_unwrapped._pending_keypoint

                if 'keypoint_p' not in current_keypoint or 'keypoint_q' not in current_keypoint:
                    raise ValueError(
                        f"_pending_keypoint missing keypoint_p/keypoint_q: {current_keypoint}"
                    )

                keypoint_p_np = np.asarray(current_keypoint['keypoint_p'], dtype=np.float32).reshape(-1)
                keypoint_q_np = np.asarray(current_keypoint['keypoint_q'], dtype=np.float32).reshape(-1)
                if keypoint_p_np.size != 3 or keypoint_q_np.size != 4:
                    raise ValueError(
                        f"_pending_keypoint keypoint shape invalid: p={keypoint_p_np.shape}, q={keypoint_q_np.shape}"
                    )

                current_keypoint['keypoint_p'] = keypoint_p_np
                current_keypoint['keypoint_q'] = keypoint_q_np
                
                env_unwrapped._pending_keypoint = None

            record_data = {
                'record_timestep': record_timestep,
                'robot_endeffector_p': self.agent.tcp.pose.p.cpu().numpy(),
                'robot_endeffector_q': self.agent.tcp.pose.q.cpu().numpy(),
                'image': base_camera_frame,
                'wrist_image': wrist_camera_frame,
                'base_camera_depth': base_camera_depth,
                'wrist_camera_depth': wrist_camera_depth,
                'base_camera_extrinsic_opencv': base_camera_extrinsic_opencv,
                'base_camera_intrinsic_opencv': base_camera_intrinsic_opencv,
                'base_camera_cam2world_opengl': base_camera_cam2world_opengl,
                'wrist_camera_extrinsic_opencv': wrist_camera_extrinsic_opencv,
                'wrist_camera_intrinsic_opencv': wrist_camera_intrinsic_opencv,
                'wrist_camera_cam2world_opengl': wrist_camera_cam2world_opengl,
                'action': action,
                'state': self.agent.robot.qpos.cpu().numpy() if hasattr(self.agent.robot.qpos, 'cpu') else self.agent.robot.qpos,
                'velocity': end_effector_velocity,
                'simple_subgoal': subgoal_text,
                'simple_subgoal_online': subgoal_online_text,
                'demonstration': self.current_task_demonstration if hasattr(self, 'current_task_demonstration') else False,
                'segmentation':segmentation,
                'segmentation_result':segmentation_result,
                'grounded_subgoal': self.current_subgoal_segment_filled,
                'grounded_subgoal_online': self.current_subgoal_segment_online_filled,
                'keypoint': current_keypoint if current_keypoint else None
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
        # 如果环境步数超过预设的安全限制（2000步），则强制终止 episode
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
        fail_recover_suffix = ""
        if getattr(self.env, "use_fail_planner", False):
            fail_mode = getattr(self.env, "fail", None)
            if fail_mode == "xy":
                fail_recover_suffix = "_FailRecoverXY"
            elif fail_mode == "z":
                fail_recover_suffix = "_FailRecoverZ"
            else:
                fail_recover_suffix = "_FailRecover"
        video_prefix = f"{self.HistoryBench_env}_ep{self.HistoryBench_episode}_seed{self.HistoryBench_seed}{fail_recover_suffix}"

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
                ts_group.create_dataset("base_camera_depth", data=record_data['base_camera_depth'])
                ts_group.create_dataset("wrist_camera_depth", data=record_data['wrist_camera_depth'])
                ts_group.create_dataset("base_camera_extrinsic_opencv", data=record_data['base_camera_extrinsic_opencv'])
                ts_group.create_dataset("base_camera_intrinsic_opencv", data=record_data['base_camera_intrinsic_opencv'])
                ts_group.create_dataset("base_camera_cam2world_opengl", data=record_data['base_camera_cam2world_opengl'])
                ts_group.create_dataset("wrist_camera_extrinsic_opencv", data=record_data['wrist_camera_extrinsic_opencv'])
                ts_group.create_dataset("wrist_camera_intrinsic_opencv", data=record_data['wrist_camera_intrinsic_opencv'])
                ts_group.create_dataset("wrist_camera_cam2world_opengl", data=record_data['wrist_camera_cam2world_opengl'])
                ts_group.create_dataset("segmentation", data=record_data['segmentation'])
                ts_group.create_dataset("segmentation_result", data=record_data['segmentation_result'])

                # 动作有可能是 None（例如 planner 尚未输出），写入字符串避免 h5py 报 dtype 错误
                if record_data['action'] is None:
                    ts_group.create_dataset("action", data="None", dtype=h5py.special_dtype(vlen=str))
                else:
                    action_data = record_data['action']
                    if isinstance(action_data, torch.Tensor):
                        action_data = action_data.cpu().numpy()
                    if isinstance(action_data, list):
                        action_data = np.array(action_data)

                    # action保证8维度 如果是7维度则填充一个-1
                    # action保证8维度 如果是7维度则填充一个-1
                    if isinstance(action_data, np.ndarray):
                        if action_data.shape == (7,):
                            action_data = np.concatenate([action_data, [-1]])
                        elif action_data.shape == (1, 7):
                            action_data = action_data.flatten()
                            action_data = np.concatenate([action_data, [-1]])
                            action_data = action_data.reshape(1, 8)
                    
                    ts_group.create_dataset("action", data=action_data)

                state_data = record_data['state']
                # state保证9维度 如果是7维度则填充两个0
                if isinstance(state_data, np.ndarray):
                    if state_data.shape == (7,):
                        state_data = np.concatenate([state_data, [0, 0]])
                    elif state_data.shape == (1, 7):
                        state_data = state_data.flatten()
                        state_data = np.concatenate([state_data, [0, 0]])
                        state_data = state_data.reshape(1, 9)

                ts_group.create_dataset("state", data=state_data)
                ts_group.create_dataset("velocity", data=record_data['velocity'])

                # 处理字符串任务名，确保编码正确
                task_name = record_data['simple_subgoal']
                if isinstance(task_name, str):
                    task_name_encoded = task_name.encode('utf-8')
                else:
                    task_name_encoded = task_name
                ts_group.create_dataset("simple_subgoal", data=task_name_encoded)

                online_task_name = record_data.get('simple_subgoal_online', 'Unknown')
                if isinstance(online_task_name, str):
                    task_name_encoded = online_task_name.encode('utf-8')
                else:
                    task_name_encoded = online_task_name
                ts_group.create_dataset("simple_subgoal_online", data=task_name_encoded)

                task_name = record_data['grounded_subgoal']
                if isinstance(task_name, str):
                    task_name_encoded = task_name.encode('utf-8')
                else:
                    task_name_encoded = task_name
                ts_group.create_dataset("grounded_subgoal", data=task_name_encoded)

                task_name_online = record_data.get('grounded_subgoal_online', 'Unknown')
                if isinstance(task_name_online, str):
                    task_name_encoded = task_name_online.encode('utf-8')
                else:
                    task_name_encoded = task_name_online
                ts_group.create_dataset("grounded_subgoal_online", data=task_name_encoded)

                ts_group.create_dataset("demonstration", data=record_data['demonstration'])

                # 写入keypoint信息（如果存在）
                keypoint = record_data.get('keypoint', None)
                # 如果是demonstration就不记录这个keypoint
                # if keypoint and not record_data['demonstration']:
                if keypoint:  # 注释掉 demonstration 限制，现在 demonstration 阶段也会记录 keypoint
                    ts_group.create_dataset("keypoint_p", data=keypoint['keypoint_p'])
                    ts_group.create_dataset("keypoint_q", data=keypoint['keypoint_q'])
                    
                    solve_function_name = keypoint.get('solve_function', 'unknown')
                    if isinstance(solve_function_name, str):
                        solve_function_name_encoded = solve_function_name.encode('utf-8')
                    else:
                        solve_function_name_encoded = solve_function_name
                    ts_group.create_dataset("keypoint_solve_function", data=solve_function_name_encoded)
                    
                    keypoint_type = keypoint.get('keypoint_type', 'unknown')
                    if isinstance(keypoint_type, str):
                        keypoint_type_encoded = keypoint_type.encode('utf-8')
                    else:
                        keypoint_type_encoded = keypoint_type
                    ts_group.create_dataset("keypoint_type", data=keypoint_type_encoded)

            # 写入 setup 信息（种子、难度、任务列表）
            setup_group = episode_group.create_group(f"setup")
            setup_group.create_dataset("seed", data=self.HistoryBench_seed)
            setup_group.create_dataset(
                    "difficulty",
                    data=difficulty,
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )

            # 记录任务列表（如果存在），便于离线复现每个 simple_subgoal 的语义
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
            # 注意：视频保存失败不应该影响HDF5数据的保存
            if self.save_video and len(self.video_frames) > 0:
                try:
                    videos_dir = self.output_root / "videos"
                    videos_dir.mkdir(parents=True, exist_ok=True)
                    combined_video_path = videos_dir / f"{video_prefix}_{filename_suffix}.mp4"

                    with imageio.get_writer(combined_video_path.as_posix(), fps=30, codec="libx264", quality=8) as writer:
                        for combined_frame in self.video_frames:
                            writer.append_data(combined_frame)
                    print(f"Saved combined video to {combined_video_path}")
                except Exception as e:
                    print(f"Warning: Failed to save combined video for episode {self.HistoryBench_episode}: {e}")
                    # 视频保存失败不影响HDF5数据保存
            if self.save_video and len(self.no_object_video_frames) > 0:
                try:
                    videos_dir = self.output_root / "videos"
                    videos_dir.mkdir(parents=True, exist_ok=True)
                    no_object_video_path = videos_dir / f"success_NO_OBJECT_{video_prefix}_{filename_suffix}.mp4"

                    with imageio.get_writer(no_object_video_path.as_posix(), fps=30, codec="libx264", quality=8) as writer:
                        for combined_frame in self.no_object_video_frames:
                            writer.append_data(combined_frame)
                    print(f"Saved no-object video to {no_object_video_path}")
                except Exception as e:
                    print(f"Warning: Failed to save no-object video for episode {self.HistoryBench_episode}: {e}")
                    # 视频保存失败不影响HDF5数据保存

            print(f"Successfully saved episode {self.HistoryBench_episode}")
        else:
            print(f"Episode {self.HistoryBench_episode} failed, discarding {len(self.buffer)} records")

            # 保存失败视频（如果启用），但不写入 HDF5
            # 注意：视频保存失败不应该抛出异常
            if self.save_video and len(self.video_frames) > 0:
                try:
                    videos_dir = self.output_root / "videos"
                    videos_dir.mkdir(parents=True, exist_ok=True)
                    # Add FAILED_ prefix to the filename
                    combined_video_path = videos_dir / f"FAILED_{video_prefix}_{filename_suffix}.mp4"

                    with imageio.get_writer(combined_video_path.as_posix(), fps=30, codec="libx264", quality=8) as writer:
                        for combined_frame in self.video_frames:
                            writer.append_data(combined_frame)
                    print(f"Saved failed episode video to {combined_video_path}")
                except Exception as e:
                    print(f"Warning: Failed to save failed episode video for episode {self.HistoryBench_episode}: {e}")
            if self.save_video and len(self.no_object_video_frames) > 0:
                try:
                    videos_dir = self.output_root / "videos"
                    videos_dir.mkdir(parents=True, exist_ok=True)
                    no_object_video_path = videos_dir / f"FAILED_NO_OBJECT_{video_prefix}_{filename_suffix}.mp4"

                    with imageio.get_writer(no_object_video_path.as_posix(), fps=30, codec="libx264", quality=8) as writer:
                        for combined_frame in self.no_object_video_frames:
                            writer.append_data(combined_frame)
                    print(f"Saved failed no-object video to {no_object_video_path}")
                except Exception as e:
                    print(f"Warning: Failed to save failed no-object video for episode {self.HistoryBench_episode}: {e}")

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
