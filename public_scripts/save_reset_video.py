# -*- coding: utf-8 -*-
# Shared utilities for saving reset-phase (demonstration) videos with captions.

import os
from typing import List, Any, Optional

import numpy as np
import cv2
import imageio

TEXT_AREA_HEIGHT = 60


def add_text_to_frame(
    frame: np.ndarray,
    text: Any,
    text_area_height: int = TEXT_AREA_HEIGHT,
) -> np.ndarray:
    """Overlay caption on top of frame (black bar + text), same style as DemonstrationWrapper.save_video."""
    frame = np.asarray(frame).copy()
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
    obs_list: List[Optional[dict]],
    reward_list: List[Any],
    terminated_list: List[Any],
    truncated_list: List[Any],
    info_list: List[Optional[dict]],
    save_path: str,
    fps: int = 20,
) -> bool:
    """
    Save the reset-phase (demonstration) video with captions from subgoal_grounded.

    Extracts frames from each obs in obs_list (key 'frames') and captions from
    each info in info_list (key 'subgoal_grounded'), then writes a captioned
    video to save_path.

    Args:
        obs_list: List of observation dicts from env.reset().
        reward_list: List of rewards (unused, for consistent signature).
        terminated_list: List of terminated flags (unused).
        truncated_list: List of truncated flags (unused).
        info_list: List of info dicts from env.reset().
        save_path: Output video file path (e.g. .mp4).
        fps: Frames per second for the output video.

    Returns:
        True if a video was written (at least one frame), False otherwise.
    """
    def _extract_frames(obs_root: dict) -> list:
        frames_local = obs_root.get("frames", [])
        if frames_local:
            return list(frames_local)
        ms_obs = obs_root.get("maniskill_obs", {})
        sensor_data = ms_obs.get("sensor_data", {}) if isinstance(ms_obs, dict) else {}
        base_cam = sensor_data.get("base_camera", {}) if isinstance(sensor_data, dict) else {}
        rgb = base_cam.get("rgb") if isinstance(base_cam, dict) else None
        if rgb is None:
            return []
        try:
            if hasattr(rgb, "cpu"):
                rgb = rgb.cpu().numpy()
            rgb = np.asarray(rgb)
            if rgb.ndim >= 4:
                return [rgb[0]]
        except Exception:
            return []
        return []

    frames = []
    for o in obs_list:
        obs_root = o or {}
        frames.extend(_extract_frames(obs_root))
    subgoal_grounded = []
    for i in info_list:
        if i:
            subgoal_grounded.extend(i.get("subgoal_grounded", []))

    n_reset = min(len(frames), len(subgoal_grounded))
    if n_reset == 0:
        return False

    out_dir = os.path.dirname(os.path.abspath(save_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with imageio.get_writer(save_path, fps=fps, codec="libx264", quality=8) as writer:
        for i in range(n_reset):
            frame = np.asarray(frames[i])
            caption = subgoal_grounded[i] if i < len(subgoal_grounded) else ""
            combined = add_text_to_frame(frame, caption)
            writer.append_data(combined)
    print(f"Saved: {save_path}")
    return True
