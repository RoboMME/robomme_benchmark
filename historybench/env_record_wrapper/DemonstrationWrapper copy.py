"""
DemonstrationWrapper：在 HistoryBench 环境外再包一层，用于自动生成演示轨迹并记录帧/动作/状态/子目标等。

- reset 后调用 get_demonstration_trajectory()，用 Motion Planner 执行带 demonstration 标记的任务并记录轨迹。
- step 接收关节空间动作，做分割与子目标占位符填充、轨迹记录、truncate 与成功判断。ee_pose→关节 由外层 EndeffectorDemonstrationWrapper 负责。
- 不包含视频保存功能；obs/info 通过 _augment_obs_and_info 注入演示数据。
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
from ..HistoryBench_env.util import task_goal

from ..HistoryBench_env.util import reset_panda


class DemonstrationWrapper(gym.Wrapper):
    """
    Demonstration 包装器（不包含视频保存功能）。

    主要功能：
    1. 在环境 reset 后自动生成演示轨迹（Trajectory），使用 Motion Planner。
    2. 记录演示过程中的帧、动作、状态、子目标等数据，供下游任务使用。
    """
    def __init__(self, env, max_steps_without_demonstration, gui_render, **kwargs):
        # **kwargs 兼容旧调用（如 save_video=..., action_space=...），已不再使用
        # 无演示步数上限：超过此步数未执行演示任务则 truncate episode
        self.max_steps_without_demonstration = max_steps_without_demonstration
        self.gui_render = gui_render

        super().__init__(env)
        self.unwrapped.use_demonstrationwrapper = True

        # 演示轨迹缓冲区：每步记录的观测与动作
        self.frames = []              # 基座相机 RGB 帧列表
        self.wrist_frames = []        # 腕部相机 RGB 帧列表
        self.actions = []             # 动作序列
        self.states = []              # 机器人状态（如 qpos）序列
        self.subgoal = []             # 子目标文本序列（原始，含占位符）
        self.subgoal_grounded = []    # 子目标文本序列（占位符已替换为坐标）
        self.demonstration_record_traj = False  # 当前是否处于“演示记录”阶段
        self.velocity = []            # 末端执行器速度序列


        # 连续未执行“演示任务”的步数，用于 truncate 判断
        self.steps_without_demonstration = 0
        # 防止在“终止时追加一步”逻辑中重复进入 step
        self._doing_extra_step = False
        # 本次 episode 的演示轨迹数据（由 get_demonstration_trajectory 填充）
        self.demonstration_data = None
        # 当前子目标文本中占位符替换为坐标后的结果
        self.current_subgoal_segment_filled = None
        # 本 episode 是否被判定为成功（用于下游是否保存数据等）
        self.episode_success = False

        self._failed_match_save_count = 0

        # 与 RecordWrapper 一致：按物体 ID 生成区分度高的颜色表，用于分割可视化
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
        """重置环境并生成演示轨迹，再执行一步初始动作后返回增强后的 obs 与 info。"""
        # Reset latching state
        self.last_subgoal_segment = None
        self.latched_replacements = None
        self._failed_match_save_count = 0

        obs = super().reset(**kwargs)
        self.episode_success = False
        # 生成演示轨迹：内部会执行所有带 demonstration 标记的任务并记录帧/动作/状态
        self.demonstration_data = self.get_demonstration_trajectory()

        # 根据环境选择夹爪与初始动作：PatternLock/RouteStick 使用 stick 且需在线生成的 action
        if self.unwrapped.spec.id == "PatternLock" or self.unwrapped.spec.id == "RouteStick":
            gripper = "stick"
        else:
            gripper = None
        if self.unwrapped.spec.id == "PatternLock" or self.unwrapped.spec.id == "RouteStick":
            action = self.unwrapped.swing_qpos  # 这两类环境需使用在线生成的初始 action
        else:
            action = reset_panda.get_reset_panda_param("action", gripper=gripper)

        # 执行一步初始动作，使观测与记录与演示阶段对齐
        obs, _, _, _, info = self.step(action)
        obs, info = self._augment_obs_and_info(obs, info)
        return obs, info

    def _augment_obs_and_info(self, obs, info):
        """将演示轨迹数据（帧、动作、状态、速度、子目标历史等）合并进 obs 和 info 后返回。只取各 list 的最后一个，仍以 list 形式给出。"""
        language_goal = task_goal.get_language_goal(self.env, self.unwrapped.spec.id)
        new_obs = {
            **obs,
            'frames': [self.frames[-1]],
            'wrist_frames': [self.wrist_frames[-1]],
            'actions': [self.actions[-1]],
            'states': [self.states[-1]],
            'velocity': [self.velocity[-1]],
            'language_goal': language_goal,
        }
        new_info = {
            **info,
            'subgoal': [self.subgoal[-1]],
            'subgoal_grounded': [self.subgoal_grounded[-1]],
        }
        return new_obs, new_info

    def _add_red_border(self, frame, border_width=5):
        """在图像四边绘制红色边框，用于高亮演示帧（当前未用于保存视频）。"""
        frame_with_border = frame.copy()
        frame_with_border[:border_width, :] = [255, 0, 0]
        frame_with_border[-border_width:, :] = [255, 0, 0]
        frame_with_border[:, :border_width] = [255, 0, 0]
        frame_with_border[:, -border_width:] = [255, 0, 0]
        return frame_with_border

    TEXT_AREA_HEIGHT = 80  # 固定字体黑边高度

    def _add_text_to_frame(self, frame, text, position='top_right'):
        """在帧上方追加黑色文本区域并拼接，支持多行与自动换行。黑边高度固定为 TEXT_AREA_HEIGHT。"""
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

    def save_video(self, output_path: Union[str, Path], fps: int = 20):
        """
        将 self.frames 与 self.subgoal_grounded 一一对应保存为视频；
        每帧上方为黑色区域，写入当前 subgoal_grounded 文本（自动换行）。
        """
        n = min(len(self.frames), len(self.subgoal_grounded))
        if n == 0:
            return
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        target_h, target_w = None, None
        scale = 2  # 分辨率放大两倍，长宽比固定
        with imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=8) as writer:
            for i in range(n):
                frame = np.asarray(self.frames[i]).copy()
                text = self.subgoal_grounded[i] if i < len(self.subgoal_grounded) else ""
                combined = self._add_text_to_frame(frame, text)
                if combined.ndim == 2:
                    combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)
                if target_h is None:
                    target_h, target_w = combined.shape[:2]
                if combined.shape[0] != target_h or combined.shape[1] != target_w:
                    combined = cv2.resize(combined, (int(target_w), int(target_h)), interpolation=cv2.INTER_LINEAR)
                out_w, out_h = int(target_w) * scale, int(target_h) * scale
                combined = cv2.resize(combined, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                writer.append_data(combined)

    def save_frame_as_image(self, output_path: Union[str, Path], frame: np.ndarray, text=None):
        """
        将单帧与文本叠加（与 save_video 中单帧逻辑一致）并保存为图片。
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
        从观测中解析基座相机分割，构建当前任务关心的物体 ID 映射，计算目标物体在图像上的
        像素中心，并将当前子目标文本中的占位符（如 <target>）替换为具体坐标 <y, x>。
        支持锁存：同一子目标在首次成功填充后会沿用该结果；子目标变化时清空锁存并重新计算。

        参数:
            obs: 环境当前步观测，需包含 sensor_data.base_camera.segmentation（及可选 rgb 等）。

        返回:
            filled_text: 占位符替换后的子目标文本；无子目标或未替换时与 current_subgoal_segment 一致。
            failed_match: 文本中存在占位符但本帧未能得到有效填充且无锁存时为 True（用于保存失败帧等）。
        """
        current_subgoal_segment = getattr(self.unwrapped, 'current_subgoal_segment', None)
        current_task_name = getattr(self, 'current_task_name', 'Unknown')

        # ---------- 从 obs 解析基座相机分割，并构建 active_segments / segment_ids_by_index / vis_obj_id_list ----------
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

            # 当前任务关心的分割物体（current_segment）与 ID 映射，用于后续中心点计算与占位符填充
            current_segment = getattr(self, "current_segment", None)
            if isinstance(current_segment, (list, tuple)):
                active_segments = list(current_segment)
            elif current_segment is None:
                active_segments = []
            else:
                active_segments = [current_segment]

            # 按 active_segments 索引建立「物体 -> 分割 ID」的映射，供逐段算中心
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
                    # 桌面在颜色表中置黑，便于分割可视化时区分
                    if getattr(obj, "name", None) == 'table-workspace':
                        self.color_map[obj_id] = [0, 0, 0]

        # 无分割数据时不做填充，直接返回原文本与未匹配
        if segmentation_2d is None:
            return (current_subgoal_segment, False)

        def center_from_ids(segmentation_mask: np.ndarray, ids: List):
            """
            根据分割掩码与物体 ID 列表计算该物体在图像上的像素中心（质心）。
            返回 (center [y, x] 或 None, no_object_flag_this)。
            当 ids 非空但掩码中无对应像素时，no_object_flag_this 为 True。
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

        # 子目标变化时清空锁存，后续将用当前帧重新计算并可能重新锁存
        if current_subgoal_segment != self.last_subgoal_segment:
            self.last_subgoal_segment = current_subgoal_segment
            self.latched_replacements = None

        # 按当前任务关心的物体逐段计算像素中心（或整张图单一中心）
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

        # 无子目标文本时无需占位符替换，直接返回
        if not current_subgoal_segment:
            return (current_subgoal_segment, False)

        # 用正则匹配所有占位符（形如 <...>）
        placeholder_pattern = re.compile(r'<[^>]*>')
        placeholders = list(placeholder_pattern.finditer(current_subgoal_segment))
        placeholder_count = len(placeholders)

        final_replacements = None
        missing_placeholder = False

        # 优先使用已锁存的替换结果；无锁存时用当前帧中心生成替换串
        if self.latched_replacements is not None:
            final_replacements = self.latched_replacements
        else:
            # 将每个中心格式化为 "<y, x>" 字符串，未检测到的中心为 None
            normalized_centers = []
            for center in segment_centers:
                if center is None:
                    normalized_centers.append(None)
                    continue
                center_y, center_x = center
                normalized_centers.append(f'<{center_y}, {center_x}>')

            if placeholder_count > 0 and normalized_centers:
                replacements = normalized_centers.copy()
                # 若仅有一个中心但多个占位符，则重复使用该中心；若中心不足则用 None 补齐
                if len(replacements) == 1 and placeholder_count > 1:
                    replacements = replacements * placeholder_count
                elif len(replacements) < placeholder_count:
                    replacements.extend([None] * (placeholder_count - len(replacements)))
                # 仅当所有占位符都能被非 None 替换时才锁存，避免锁存不完整结果
                temp_missing_placeholder = any(r is None for r in replacements)
                if not temp_missing_placeholder:
                    self.latched_replacements = replacements
                final_replacements = replacements

        # 应用替换：按占位符顺序拼出最终文本，若有任一占位符缺替换则用 current_task_name 作为整句回退
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
            # 无锁存且（本帧未给出有效替换或仍有缺项）时视为匹配失败
            failed_match = self.latched_replacements is None and (final_replacements is None or missing_placeholder)
            return (filled_text, failed_match)
        else:
            # 有占位符但无替换结果且无锁存，也记为匹配失败
            failed_match = placeholder_count > 0 and self.latched_replacements is None
            return (current_subgoal_segment, failed_match)

    def step(self, action):
        """执行一步：分割与子目标占位符填充、轨迹记录、truncate 与成功判断。接收关节空间动作。"""
        obs, reward, terminated, truncated, info = super().step(action)

        # ---------- 子目标分割与占位符填充：内部从 obs 解析分割并计算中心、填充占位符 ----------
        filled_text, failed_match = self._compute_segmentation_and_fill_subgoal(obs)
        current_subgoal_segment = getattr(self.unwrapped, 'current_subgoal_segment', None)
        self.current_subgoal_segment_filled = filled_text if filled_text is not None else current_subgoal_segment

        # ---------- 轨迹记录：非 “NO RECORD” 任务时追加当前帧、动作、状态、子目标等 ----------
        current_task = self.current_task_name if hasattr(self, 'current_task_name') else "Unknown"
        if current_task != 'NO RECORD':
            base_camera_frame = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
            wrist_camera_frame = obs['sensor_data']['hand_camera']['rgb'][0].cpu().numpy()
            image = base_camera_frame
            wrist_image = wrist_camera_frame
            # 当前步的机器人关节位置与末端线速度+角速度（用于轨迹数据）
            state = self.agent.robot.qpos.cpu().numpy() if hasattr(self.agent.robot.qpos, 'cpu') else self.agent.robot.qpos
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

            # # 子目标占位符未匹配上时，将当前帧与填充后子目标文本保存为图片，便于调试（目录写死为 failed_match）
            # if failed_match:
            #     save_dir = Path("/data/hongzefu/dataset_generate/failed_match")
            #     self._failed_match_save_count += 1  # 本 episode 内失败保存计数，用于文件名去重
            #     env_id = getattr(self, "save_failed_match_env_id", None)
            #     episode = getattr(self, "save_failed_match_episode", None)
            #     if env_id is not None and episode is not None:
            #         basename = f"failed_match_{env_id}_ep{episode}_{self._failed_match_save_count:04d}.png"
            #     else:
            #         basename = f"failed_match_{self._failed_match_save_count:04d}.png"
            #     out_path = save_dir / basename
            #     self.save_frame_as_image(out_path, image, grounded_subgoal)  # 图上叠加当前 grounded 子目标文本

        # ---------- 非演示步计数：超过上限则 truncate ----------
        if self.current_task_demonstration == False:
            self.steps_without_demonstration += 1
            if self.steps_without_demonstration >= self.max_steps_without_demonstration:
                truncated = torch.tensor([True])

        # ---------- 根据 terminated 与 info["success"] 更新 episode_success ----------
        if terminated.any():
            if info.get("success") == torch.tensor([True]) or (isinstance(info.get("success"), torch.Tensor) and info.get("success").item()):
                self.episode_success = True
                print("Episode success detected, data will be saved")
            else:
                self.episode_success = False
                print("Episode failed, data will be discarded")

        # ---------- 终止时多执行一步，使最后一帧也被记录（动作与上一步相同） ----------
        if terminated.any() and not self._doing_extra_step:
            self._doing_extra_step = True
            try:
                self.step(action)
            finally:
                self._doing_extra_step = False

        obs, info = self._augment_obs_and_info(obs, info)
        return obs, reward, terminated, truncated, info

    def close(self):
        """关闭环境，释放资源（本包装器不再保存视频）。"""
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
        # 按环境选择运动规划器：PatternLock/RouteStick 用 stick 规划器，其余用机械臂规划器
        if self.unwrapped.spec.id == "PatternLock" or self.unwrapped.spec.id == "RouteStick":
            planner = PandaStickMotionPlanningSolver(
                self,
                debug=False,
                vis=False,
                base_pose=self.unwrapped.agent.robot.pose,
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
        self.task_list_length = len(tasks)
        print(f"Task list length: {self.task_list_length}")

        demonstration_tasks = [task for task in tasks if task.get("demonstration", False)]
        self.non_demonstration_task_length = len(tasks) - len(demonstration_tasks)
        print(f"Non-demonstration task length: {self.non_demonstration_task_length}")

        # 依次执行每个演示任务：设 demonstration_record_traj=True，调用任务的 solve(planner)
        for idx, task_entry in enumerate(demonstration_tasks):
            self.unwrapped.demonstration_record_traj = True
            task_name = task_entry.get("name", f"Task {idx}")
            print(f"Executing task {idx+1}/{len(demonstration_tasks)}: {task_name}")

            solve_callable = task_entry.get("solve")
            if not callable(solve_callable):
                raise ValueError(f"Task '{task_name}' must supply a callable 'solve'.")

            self.evaluate(solve_complete_eval=True)
            solve_callable(self, planner)
            self.evaluate(solve_complete_eval=True)

        language_goal = task_goal.get_language_goal(self.env, self.env.unwrapped.spec.id)
        self.unwrapped.demonstration_record_traj = False  # 演示结束，后续 step 正常做 subgoal 判断
        return {
            'frames': self.frames,
            'wrist_frames': self.wrist_frames,
            'actions': self.actions,
            'states': self.states,
            'velocity': self.velocity,
            'subgoal': self.subgoal,
            'subgoal_grounded': self.subgoal_grounded,
            'language goal': language_goal,
        }
