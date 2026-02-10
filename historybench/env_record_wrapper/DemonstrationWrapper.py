"""
DemonstrationWrapper：在 HistoryBench 环境外再包一层，用于自动生成演示轨迹并记录帧/动作/状态/子目标等。

- reset 后调用 get_demonstration_trajectory()，用 Motion Planner 执行带 demonstration 标记的任务并记录轨迹。
- step 接收关节空间动作，做分割与子目标占位符填充、轨迹记录、truncate 与成功判断。ee_pose→关节 由外层 EndeffectorDemonstrationWrapper 负责。
- 不包含视频保存功能；reset/step 返回统一 dense batch；step 通过 _augment_obs_and_info 注入当前步的 frames、subgoal 等。
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
from ..HistoryBench_env.util.vqa_options import get_vqa_options

from ..HistoryBench_env.util import reset_panda

from . import planner_denseStep


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

        self.demonstration_record_traj = False  # 当前是否处于“演示记录”阶段

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
        # demonstration 阶段 screw 规划失败重试：总尝试次数（含首次）
        self._demo_screw_max_attempts = 1
        # screw 失败后的 RRT* 规划重试：总尝试次数（含首次）
        self._demo_rrt_max_attempts = 3
        # 当前 demonstration task 是否出现规划失败（用于 task 级继续执行）
        self._current_demo_task_screw_failed = False
        # 末端姿态连续化缓存（wxyz / XYZ-RPY）：
        # - _prev_ee_quat_wxyz：保存“符号对齐后”的上一帧四元数表示
        # - _prev_ee_rpy_xyz：保存“unwrap 后”的上一帧连续 RPY
        # 这两个缓存共同决定跨帧连续化行为，生命周期限定在单个 episode 内。
        self._prev_ee_quat_wxyz = None
        self._prev_ee_rpy_xyz = None

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
        """重置环境并生成演示轨迹，再执行一步初始动作后返回统一批次。"""
        # 重置锁存状态
        self.last_subgoal_segment = None
        self.latched_replacements = None
        self._failed_match_save_count = 0
        # 每个 episode 从干净缓存开始，避免跨 episode 污染：
        # 不允许上一局的“上一帧姿态”影响当前局第一帧的展开结果。
        self._prev_ee_quat_wxyz = None
        self._prev_ee_rpy_xyz = None

        super().reset(**kwargs)
        self.episode_success = False
        # 生成演示轨迹批次
        demo_batch = self.get_demonstration_trajectory()

        # 根据环境选择夹爪与初始动作：PatternLock/RouteStick 使用 stick 且需在线生成的 action
        if self.unwrapped.spec.id == "PatternLock" or self.unwrapped.spec.id == "RouteStick":
            gripper = "stick"
        else:
            gripper = None
        if self.unwrapped.spec.id == "PatternLock" or self.unwrapped.spec.id == "RouteStick":
            action = self.unwrapped.swing_qpos  # 这两类环境需使用在线生成的初始 action
        else:
            action = reset_panda.get_reset_panda_param("action", gripper=gripper)

        # 执行一步初始动作，拼接到演示轨迹批次
        init_batch = self.step(action)
        merged_batch = planner_denseStep.concat_step_batches([demo_batch, init_batch])
        self.demonstration_data = merged_batch
        return merged_batch

    def _normalize_quat_wxyz(self, quat: torch.Tensor) -> torch.Tensor:
        """
        归一化四元数（wxyz）。

        规则：
        1. 仅当“元素有限 + 范数有限 + 范数>阈值”时按常规归一化；
        2. 对零范数或 NaN/Inf，回退为单位四元数 [1, 0, 0, 0]；
        3. 输出与输入形状保持一致，便于批处理。
        """
        quat = torch.as_tensor(quat)
        # 有效性判定：输入数值可用且可安全归一化。
        quat_norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
        finite_quat = torch.all(torch.isfinite(quat), dim=-1, keepdim=True)
        finite_norm = torch.isfinite(quat_norm)
        valid = finite_quat & finite_norm & (quat_norm > 1e-12)

        # 无效项先用 1.0 作为安全除数，随后整体由 where 覆盖 fallback。
        safe_norm = torch.where(valid, quat_norm, torch.ones_like(quat_norm))
        normalized = quat / safe_norm
        # fallback 统一为单位四元数，确保后续转换稳定。
        fallback = torch.zeros_like(normalized)
        fallback[..., 0] = 1.0
        return torch.where(valid.expand_as(normalized), normalized, fallback)

    def _align_quat_sign_with_prev(self, quat: torch.Tensor) -> torch.Tensor:
        """
        四元数符号对齐（q 与 -q 等价）：
        通过与上一帧点积判断是否翻转，减少表示层面的突变。

        注意：
        - 若无上一帧缓存，或形状不一致，直接返回当前 quat；
        - shape 不一致通常意味着 batch 结构变化，此时不强行对齐。
        """
        prev = self._prev_ee_quat_wxyz
        if prev is None:
            return quat
        if prev.shape != quat.shape:
            return quat

        prev = prev.to(device=quat.device, dtype=quat.dtype)
        # dot<0 时代表位于单位球对径点，翻转可获得更连续的表示。
        dot = torch.sum(quat * prev, dim=-1, keepdim=True)
        sign = torch.where(dot < 0, -torch.ones_like(dot), torch.ones_like(dot))
        return quat * sign

    def _quat_wxyz_to_rpy_xyz(self, quat: torch.Tensor) -> torch.Tensor:
        """
        将 wxyz 四元数转换为 XYZ 顺序的 RPY（弧度）。

        说明：
        - 输出是欧拉主值（未 unwrap）；
        - pitch 在 asin 前做 clamp，防止浮点误差越界。
        """
        w, x, y, z = quat.unbind(dim=-1)

        # roll (X)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # pitch (Y)：asin 入参裁剪到 [-1, 1]。
        sinp = 2.0 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

        # yaw (Z)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return torch.stack((roll, pitch, yaw), dim=-1)

    def _unwrap_rpy_with_prev(self, rpy: torch.Tensor) -> torch.Tensor:
        """
        RPY 展开（unwrap）：基于上一帧，把相邻差分折叠到 (-pi, pi] 后再累加。

        结果特性：
        - 优先保证跨帧无跳变；
        - 输出是“连续角”，可超出 [-pi, pi]（不是主值范围）。
        """
        prev = self._prev_ee_rpy_xyz
        if prev is None:
            return rpy
        if prev.shape != rpy.shape:
            return rpy

        prev = prev.to(device=rpy.device, dtype=rpy.dtype)
        pi = torch.as_tensor(np.pi, dtype=rpy.dtype, device=rpy.device)
        two_pi = torch.as_tensor(2.0 * np.pi, dtype=rpy.dtype, device=rpy.device)
        # 相邻帧差分 -> 映射到 (-pi, pi] -> 加回上一帧，得到连续表示。
        delta = rpy - prev
        delta = torch.remainder(delta + pi, two_pi) - pi
        return prev + delta

    def _build_robot_endeffector_pose_xyzrpy(self) -> torch.Tensor:
        """
        构建 xyz+rpy（连续）形式的末端位姿。

        流水线：
        1) 读取当前 tcp pose 的 p/q；
        2) quat 归一化；
        3) 与上一帧做四元数符号对齐；
        4) quat -> rpy 主值；
        5) 基于上一帧做 unwrap，得到连续 RPY；
        6) 更新缓存（对齐后 quat + unwrap 后 rpy）；
        7) 输出 [x, y, z, roll, pitch, yaw]。

        备注：该流程只做连续化稳定处理，不尝试全局最优欧拉参数化。
        """
        robot_endeffector_p = self.agent.tcp.pose.p
        robot_endeffector_q = self.agent.tcp.pose.q

        quat_normalized = self._normalize_quat_wxyz(robot_endeffector_q)
        quat_aligned = self._align_quat_sign_with_prev(quat_normalized)
        rpy_xyz = self._quat_wxyz_to_rpy_xyz(quat_aligned)
        rpy_xyz_unwrapped = self._unwrap_rpy_with_prev(rpy_xyz)

        # 缓存“本帧最终参与连续化的表示”，供下一帧继续使用。
        self._prev_ee_quat_wxyz = quat_aligned.detach().clone()
        self._prev_ee_rpy_xyz = rpy_xyz_unwrapped.detach().clone()
        return torch.cat((robot_endeffector_p, rpy_xyz_unwrapped), dim=-1)

    def _augment_obs_and_info(self, obs, info, action):
        """直接从 obs 提取当前步数据并合并进 obs 和 info 后返回，不经过 list 缓冲区中转。"""
        language_goal = task_goal.get_language_goal(self.env, self.unwrapped.spec.id)
        base_obs = obs if isinstance(obs, dict) else {}
        env_id = self.unwrapped.spec.id
        dummy_target = {"obj": None, "name": None, "seg_id": None, "click_point": None, "centroid_point": None}
        raw_options = get_vqa_options(self, None, dummy_target, env_id)
        available_options = [
            {"action": opt.get("label", "未知"), "need_parameter": bool(opt.get("available"))}
            for opt in raw_options
        ]

        # 直接从 obs 提取帧、状态、速度等（不再从 self.frames 等 list 读取）
        image = obs['sensor_data']['base_camera']['rgb'][0]
        wrist_image = obs['sensor_data']['hand_camera']['rgb'][0]
        state = self.agent.robot.qpos
        end_effector_velocity = self.agent.robot.links[9].get_linear_velocity()[0], self.agent.robot.links[9].get_angular_velocity()[0]
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
        # 这里统一产出 xyz+rpy（连续）而非历史的 xyz+quat，
        # 便于下游直接做角度连续性统计与控制。
        robot_endeffector_pose = self._build_robot_endeffector_pose_xyzrpy()

        new_obs = {
            'maniskill_obs': base_obs,
            'front_camera': image,
            'wrist_camera': wrist_image,
            'joint_states': state,
            'velocity': end_effector_velocity,
            'front_camera_depth': base_camera_depth,
            'front_camera_segmentation': base_camera_segmentation,
            'wrist_camera_depth': wrist_camera_depth,
            'front_camera_extrinsic_opencv': base_camera_extrinsic_opencv,
            'front_camera_intrinsic_opencv': base_camera_intrinsic_opencv,
            'front_camera_cam2world_opengl': base_camera_cam2world_opengl,
            'wrist_camera_extrinsic_opencv': wrist_camera_extrinsic_opencv,
            'wrist_camera_intrinsic_opencv': wrist_camera_intrinsic_opencv,
            'wrist_camera_cam2world_opengl': wrist_camera_cam2world_opengl,
            'robot_endeffector_pose': robot_endeffector_pose,
        }
        new_info = {
            **info,
            'subgoal': subgoal_text,
            'subgoal_grounded': grounded_subgoal,
            'available_options': available_options,
            'language_goal': language_goal,
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

    def save_frame_as_image(self, output_path: Union[str, Path], frame: np.ndarray, text=None):
        """
        将单帧与文本叠加并保存为图片。
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

    def step(self, action):
        """执行一步并返回统一批次（N=1）。"""
        normalized_action = self._normalize_action_for_env_step(action)
        obs, reward, terminated, truncated, info = super().step(normalized_action)

        # ---------- 子目标分割与占位符填充：内部从 obs 解析分割并计算中心、填充占位符 ----------
        filled_text, failed_match = self._compute_segmentation_and_fill_subgoal(obs)
        current_subgoal_segment = getattr(self.unwrapped, 'current_subgoal_segment', None)
        self.current_subgoal_segment_filled = filled_text if filled_text is not None else current_subgoal_segment

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
            # 在递归额外 step 前保存 RPY 连续化缓存。
            # 原因：内层额外 step 不应改变“当前外层返回步”的上一帧基准，
            # 否则会污染外层时序上的连续化结果。
            cached_prev_quat = None if self._prev_ee_quat_wxyz is None else self._prev_ee_quat_wxyz.detach().clone()
            cached_prev_rpy = None if self._prev_ee_rpy_xyz is None else self._prev_ee_rpy_xyz.detach().clone()
            self._doing_extra_step = True
            try:
                self.step(normalized_action)
            finally:
                self._doing_extra_step = False
                # 恢复外层缓存，保证“额外 step 仅用于记录帧”，不干扰外层连续化状态。
                self._prev_ee_quat_wxyz = cached_prev_quat
                self._prev_ee_rpy_xyz = cached_prev_rpy

        obs, info = self._augment_obs_and_info(obs, info, normalized_action)
        return planner_denseStep.to_step_batch([(obs, reward, terminated, truncated, info)])

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
        3. 对每个演示任务，用 _collect_dense_steps 包裹整个 solve 调用，
           monkey-patch planner.env.step 以收集所有 env.step 调用
           （包括 move_to_pose_with_screw、follow_path、直接 env.step 等所有路径）。
        4. 返回统一批次（obs/info 字典值为 list，reward/terminated/truncated 为一维张量）。
        """
        # 懒加载 FailAware 规划器；若导入失败，回退到原 planner 实现
        try:
            from planner_fail_safe import (
                FailAwarePandaArmMotionPlanningSolver,
                FailAwarePandaStickMotionPlanningSolver,
                ScrewPlanFailure,
            )
        except Exception as exc:
            print(f"[DemonstrationWrapper] Warning: failed to import planner_fail_safe, fallback to base planners: {exc}")
            FailAwarePandaArmMotionPlanningSolver = PandaArmMotionPlanningSolver
            FailAwarePandaStickMotionPlanningSolver = PandaStickMotionPlanningSolver
            ScrewPlanFailure = RuntimeError

        # 按环境选择运动规划器：PatternLock/RouteStick 用 stick 规划器，其余用机械臂规划器
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

        # 在 planner 实例层包装 screw 调用：screw 失败后自动切到 RRT* 重试
        original_move_to_pose_with_screw = planner.move_to_pose_with_screw
        original_move_to_pose_with_rrt = planner.move_to_pose_with_RRTStar

        def _move_to_pose_with_screw_then_rrt_retry(*args, **kwargs):
            for attempt in range(1, self._demo_screw_max_attempts + 1):
                try:
                    result = original_move_to_pose_with_screw(*args, **kwargs)
                except ScrewPlanFailure as exc:
                    print(
                        f"[DemonstrationWrapper] screw planning failed "
                        f"(attempt {attempt}/{self._demo_screw_max_attempts}): {exc}"
                    )
                    continue

                # 兼容非 FailAware 回退场景：原 planner 可能直接返回 -1
                if isinstance(result, int) and result == -1:
                    print(
                        f"[DemonstrationWrapper] screw planning returned -1 "
                        f"(attempt {attempt}/{self._demo_screw_max_attempts})"
                    )
                    continue

                return result

            print(
                "[DemonstrationWrapper] screw planning exhausted; "
                f"fallback to RRT* (max {self._demo_rrt_max_attempts} attempts)"
            )

            for attempt in range(1, self._demo_rrt_max_attempts + 1):
                try:
                    result = original_move_to_pose_with_rrt(*args, **kwargs)
                except Exception as exc:
                    print(
                        f"[DemonstrationWrapper] RRT* planning failed "
                        f"(attempt {attempt}/{self._demo_rrt_max_attempts}): {exc}"
                    )
                    continue

                if isinstance(result, int) and result == -1:
                    print(
                        f"[DemonstrationWrapper] RRT* planning returned -1 "
                        f"(attempt {attempt}/{self._demo_rrt_max_attempts})"
                    )
                    continue

                return result

            self._current_demo_task_screw_failed = True
            print("[DemonstrationWrapper] screw->RRT* planning exhausted; return -1")
            return -1

        planner.move_to_pose_with_screw = _move_to_pose_with_screw_then_rrt_retry
        tasks = getattr(self, 'task_list', [])
        self.task_list_length = len(tasks)
        print(f"Task list length: {self.task_list_length}")

        demonstration_tasks = [task for task in tasks if task.get("demonstration", False)]
        self.non_demonstration_task_length = len(tasks) - len(demonstration_tasks)
        print(f"Non-demonstration task length: {self.non_demonstration_task_length}")

        all_collected_steps = []

        # 依次执行每个演示任务：设 demonstration_record_traj=True，调用任务的 solve(planner)
        # 用 _collect_dense_steps 包裹整个 solve，monkey-patch planner.env.step，
        # 以收集所有 env.step 调用（包括 follow_path、直接 env.step 等底层路径）
        for idx, task_entry in enumerate(demonstration_tasks):
            self.unwrapped.demonstration_record_traj = True
            self._current_demo_task_screw_failed = False
            task_name = task_entry.get("name", f"Task {idx}")
            print(f"Executing task {idx+1}/{len(demonstration_tasks)}: {task_name}")

            solve_callable = task_entry.get("solve")
            if not callable(solve_callable):
                raise ValueError(f"Task '{task_name}' must supply a callable 'solve'.")

            self.evaluate(solve_complete_eval=True)

            def _solve_task_without_hard_fail():
                # 避免 solve 返回 -1 导致 _collect_dense_steps 丢弃本 task 已收集的 step
                try:
                    solve_result = solve_callable(self, planner)
                except ScrewPlanFailure as exc:
                    self._current_demo_task_screw_failed = True
                    print(f"[DemonstrationWrapper] task '{task_name}' screw failure: {exc}")
                    return None
                if isinstance(solve_result, int) and solve_result == -1:
                    self._current_demo_task_screw_failed = True
                    print(f"[DemonstrationWrapper] task '{task_name}' returned -1 after screw->RRT* retries")
                    return None
                return solve_result

            task_steps = planner_denseStep._collect_dense_steps(
                planner,
                _solve_task_without_hard_fail,
            )
            if task_steps == -1:
                # 理论上不应命中（_solve_task_without_hard_fail 已吞掉 -1）
                print(f"[DemonstrationWrapper] task '{task_name}' returned -1 from collector; continuing")
            else:
                all_collected_steps.extend(task_steps)

            if self._current_demo_task_screw_failed:
                print(f"[DemonstrationWrapper] task '{task_name}' marked failed after screw->RRT* retries; continuing")
            self.evaluate(solve_complete_eval=True)

        self.unwrapped.demonstration_record_traj = False  # 演示结束，后续 step 正常做 subgoal 判断
        return planner_denseStep.to_step_batch(all_collected_steps)
