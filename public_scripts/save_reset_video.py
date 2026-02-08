# -*- coding: utf-8 -*-
# 用于保存 reset 阶段（演示阶段）带字幕视频的公共工具。

import os
from typing import Dict, List, Any

import numpy as np
import cv2
import imageio
import torch

TEXT_AREA_HEIGHT = 60


def _frame_to_numpy(frame: Any) -> np.ndarray:
    """Convert frame-like input to CPU numpy array for OpenCV/imageio writing."""
    if isinstance(frame, torch.Tensor):
        frame = frame.detach()
        if frame.is_cuda:
            frame = frame.cpu()
        frame = frame.numpy()
    else:
        frame = np.asarray(frame)
    return frame


def _flatten_column(batch_dict: Dict[str, List[Any]], key: str) -> List[Any]:
    out = []
    for item in (batch_dict or {}).get(key, []) or []:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            out.extend([x for x in item if x is not None])
        else:
            out.append(item)
    return out


def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    return frame


def _concat_left_right(left_frame: Any, right_frame: Any) -> np.ndarray:
    left = _ensure_rgb(_frame_to_numpy(left_frame))
    right = _ensure_rgb(_frame_to_numpy(right_frame))
    if left.shape[0] != right.shape[0]:
        target_h = left.shape[0]
        scale = target_h / max(1, right.shape[0])
        target_w = max(1, int(round(right.shape[1] * scale)))
        right = cv2.resize(right, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return np.hstack((left, right))


def add_text_to_frame(
    frame: np.ndarray,
    text: Any,
    text_area_height: int = TEXT_AREA_HEIGHT,
) -> np.ndarray:
    """在帧顶部叠加字幕（黑底+白字），样式与 DemonstrationWrapper.save_video 一致。"""
    frame = _frame_to_numpy(frame).copy()
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    if text is None:
        text = ""
    if isinstance(text, (list, tuple)):
        text = " | ".join(str(t).strip() for t in text if t)
    text = str(text).strip()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1
    max_width = max(1, frame.shape[1] - 20)
    lines = []
    if text:
        words = text.replace(",", " ").split()
        if words:
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


def save_listStep_video(
    obs_batch: Dict[str, List[Any]],
    reward_batch: Any,
    terminated_batch: Any,
    truncated_batch: Any,
    info_batch: Dict[str, List[Any]],
    save_path: str,
    fps: int = 20,
) -> bool:
    """
    保存 reset 阶段（演示阶段）视频，并使用 subgoal_grounded 作为字幕。

    从 obs_batch["image"] 提取图像帧，
    从 info_batch["subgoal_grounded"] 提取字幕，
    并写入 save_path 指定的视频文件。

    Args:
        obs_batch: 列式观测字典（dict-of-list）。
        reward_batch: 一维 reward 张量（未使用，仅保持函数签名一致）。
        terminated_batch: 一维 terminated 张量（未使用）。
        truncated_batch: 一维 truncated 张量（未使用）。
        info_batch: 列式 info 字典（dict-of-list）。
        save_path: 输出视频路径（如 .mp4）。
        fps: 输出视频帧率。

    Returns:
        至少写入一帧时返回 True，否则返回 False。
    """
    image = _flatten_column(obs_batch, "image")
    base_camera = _flatten_column(obs_batch, "base_camera")
    wrist_camera = _flatten_column(obs_batch, "wrist_camera")

    if base_camera and wrist_camera:
        n_pair = min(len(base_camera), len(wrist_camera))
        image = [_concat_left_right(base_camera[i], wrist_camera[i]) for i in range(n_pair)]
    elif base_camera:
        image = [_frame_to_numpy(f) for f in base_camera]
    elif wrist_camera:
        image = [_frame_to_numpy(f) for f in wrist_camera]

    subgoal_grounded = _flatten_column(info_batch, "subgoal_grounded")

    n_reset = min(len(image), len(subgoal_grounded))
    if n_reset == 0:
        return False

    out_dir = os.path.dirname(os.path.abspath(save_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with imageio.get_writer(save_path, fps=fps, codec="libx264", quality=8) as writer:
        for i in range(n_reset):
            frame = _frame_to_numpy(image[i])
            caption = subgoal_grounded[i] if i < len(subgoal_grounded) else ""
            combined = add_text_to_frame(frame, caption)
            writer.append_data(combined)
    print(f"Saved: {save_path}")
    return True
