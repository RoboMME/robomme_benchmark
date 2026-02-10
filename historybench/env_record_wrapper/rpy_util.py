"""
RPY 连续化工具：供 wrapper 与公共脚本共享。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def normalize_quat_wxyz_torch(quat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    归一化 wxyz 四元数。

    对非法输入（零范数/NaN/Inf）回退到单位四元数 [1, 0, 0, 0]。
    """
    quat = torch.as_tensor(quat)
    quat_norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
    finite_quat = torch.all(torch.isfinite(quat), dim=-1, keepdim=True)
    finite_norm = torch.isfinite(quat_norm)
    valid = finite_quat & finite_norm & (quat_norm > eps)

    safe_norm = torch.where(valid, quat_norm, torch.ones_like(quat_norm))
    normalized = quat / safe_norm
    fallback = torch.zeros_like(normalized)
    fallback[..., 0] = 1.0
    return torch.where(valid.expand_as(normalized), normalized, fallback)


def align_quat_sign_with_prev_torch(quat: torch.Tensor, prev_quat: torch.Tensor | None) -> torch.Tensor:
    """
    与上一帧四元数表示做符号对齐。

    若 dot(quat, prev_quat) < 0，则翻转当前四元数符号。
    """
    if prev_quat is None:
        return quat
    if prev_quat.shape != quat.shape:
        return quat

    prev = prev_quat.to(device=quat.device, dtype=quat.dtype)
    dot = torch.sum(quat * prev, dim=-1, keepdim=True)
    sign = torch.where(dot < 0, -torch.ones_like(dot), torch.ones_like(dot))
    return quat * sign


def quat_wxyz_to_rpy_xyz_torch(quat: torch.Tensor) -> torch.Tensor:
    """
    将 wxyz 四元数转换为 XYZ 顺序 RPY（弧度）。
    """
    w, x, y, z = quat.unbind(dim=-1)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)


def unwrap_rpy_with_prev_torch(rpy: torch.Tensor, prev_rpy: torch.Tensor | None) -> torch.Tensor:
    """
    相对上一帧做 RPY 展开：将差分折叠到 (-pi, pi] 后再累加。
    """
    if prev_rpy is None:
        return rpy
    if prev_rpy.shape != rpy.shape:
        return rpy

    prev = prev_rpy.to(device=rpy.device, dtype=rpy.dtype)
    pi = torch.as_tensor(np.pi, dtype=rpy.dtype, device=rpy.device)
    two_pi = torch.as_tensor(2.0 * np.pi, dtype=rpy.dtype, device=rpy.device)
    delta = rpy - prev
    delta = torch.remainder(delta + pi, two_pi) - pi
    return prev + delta


def summarize_and_print_rpy_sequence(rpy_sequence: Any, label: str = "") -> dict[str, Any]:
    """
    汇总一段 RPY 序列，并打印弧度/角度两份报告。
    """
    rpy = np.asarray(rpy_sequence, dtype=np.float64)
    if rpy.size == 0:
        summary = {
            "count": 0,
            "axis_min_rad": None,
            "axis_max_rad": None,
            "axis_first_rad": None,
            "axis_last_rad": None,
            "axis_min_deg": None,
            "axis_max_deg": None,
            "axis_first_deg": None,
            "axis_last_deg": None,
            "axis_max_abs_delta_rad": [0.0, 0.0, 0.0],
            "axis_max_abs_delta_deg": [0.0, 0.0, 0.0],
            "prev_step_max_abs_delta_peak_rad": 0.0,
            "prev_step_max_abs_delta_peak_deg": 0.0,
            "prev_step_peak_transition": None,
        }
        prefix = f"{label} " if label else ""
        print(f"{prefix}RPY summary: no RPY samples.")
        return summary

    if rpy.ndim == 1:
        if rpy.shape[0] == 3:
            rpy = rpy.reshape(1, 3)
        elif rpy.shape[0] % 3 == 0:
            rpy = rpy.reshape(-1, 3)
        else:
            raise ValueError(f"Cannot reshape 1D rpy_sequence of shape {rpy.shape} to (*, 3)")
    elif rpy.shape[-1] == 3:
        rpy = rpy.reshape(-1, 3)
    else:
        raise ValueError(f"rpy_sequence last dimension must be 3, got shape {rpy.shape}")

    count = int(rpy.shape[0])
    axis_min_rad = np.min(rpy, axis=0)
    axis_max_rad = np.max(rpy, axis=0)
    axis_first_rad = rpy[0]
    axis_last_rad = rpy[-1]
    rpy_deg = np.rad2deg(rpy)
    axis_min_deg = np.min(rpy_deg, axis=0)
    axis_max_deg = np.max(rpy_deg, axis=0)
    axis_first_deg = rpy_deg[0]
    axis_last_deg = rpy_deg[-1]

    if count < 2:
        axis_max_abs_delta_rad = np.zeros(3, dtype=np.float64)
        axis_max_abs_delta_deg = np.zeros(3, dtype=np.float64)
        prev_step_max_abs_delta_peak_rad = 0.0
        prev_step_max_abs_delta_peak_deg = 0.0
        prev_step_peak_transition = None
    else:
        diff = np.diff(rpy, axis=0)
        abs_diff = np.abs(diff)
        axis_max_abs_delta_rad = np.max(abs_diff, axis=0)
        axis_max_abs_delta_deg = np.rad2deg(axis_max_abs_delta_rad)
        prev_step_max_abs_delta = np.max(abs_diff, axis=1)
        peak_idx = int(np.argmax(prev_step_max_abs_delta))
        prev_step_max_abs_delta_peak_rad = float(prev_step_max_abs_delta[peak_idx])
        prev_step_max_abs_delta_peak_deg = float(np.rad2deg(prev_step_max_abs_delta_peak_rad))
        prev_step_peak_transition = [peak_idx, peak_idx + 1]

    summary = {
        "count": count,
        "axis_min_rad": axis_min_rad.tolist(),
        "axis_max_rad": axis_max_rad.tolist(),
        "axis_first_rad": axis_first_rad.tolist(),
        "axis_last_rad": axis_last_rad.tolist(),
        "axis_min_deg": axis_min_deg.tolist(),
        "axis_max_deg": axis_max_deg.tolist(),
        "axis_first_deg": axis_first_deg.tolist(),
        "axis_last_deg": axis_last_deg.tolist(),
        "axis_max_abs_delta_rad": axis_max_abs_delta_rad.tolist(),
        "axis_max_abs_delta_deg": axis_max_abs_delta_deg.tolist(),
        "prev_step_max_abs_delta_peak_rad": float(prev_step_max_abs_delta_peak_rad),
        "prev_step_max_abs_delta_peak_deg": float(prev_step_max_abs_delta_peak_deg),
        "prev_step_peak_transition": prev_step_peak_transition,
    }

    prefix = f"{label} " if label else ""
    print(f"{prefix}RPY summary (rad):")
    print(f"  count={count}")
    print(
        "  roll: "
        f"min={axis_min_rad[0]:.6f}, max={axis_max_rad[0]:.6f}, "
        f"first={axis_first_rad[0]:.6f}, last={axis_last_rad[0]:.6f}"
    )
    print(
        "  pitch: "
        f"min={axis_min_rad[1]:.6f}, max={axis_max_rad[1]:.6f}, "
        f"first={axis_first_rad[1]:.6f}, last={axis_last_rad[1]:.6f}"
    )
    print(
        "  yaw: "
        f"min={axis_min_rad[2]:.6f}, max={axis_max_rad[2]:.6f}, "
        f"first={axis_first_rad[2]:.6f}, last={axis_last_rad[2]:.6f}"
    )
    print(
        "  axis_max_abs_delta_rad (roll,pitch,yaw)="
        f"[{axis_max_abs_delta_rad[0]:.6f}, {axis_max_abs_delta_rad[1]:.6f}, {axis_max_abs_delta_rad[2]:.6f}]"
    )
    print(
        f"  prev_step_max_abs_delta_peak_rad={prev_step_max_abs_delta_peak_rad:.6f}, "
        f"transition={prev_step_peak_transition}"
    )
    print(f"{prefix}RPY summary (deg):")
    print(f"  count={count}")
    print(
        "  roll: "
        f"min={axis_min_deg[0]:.6f}, max={axis_max_deg[0]:.6f}, "
        f"first={axis_first_deg[0]:.6f}, last={axis_last_deg[0]:.6f}"
    )
    print(
        "  pitch: "
        f"min={axis_min_deg[1]:.6f}, max={axis_max_deg[1]:.6f}, "
        f"first={axis_first_deg[1]:.6f}, last={axis_last_deg[1]:.6f}"
    )
    print(
        "  yaw: "
        f"min={axis_min_deg[2]:.6f}, max={axis_max_deg[2]:.6f}, "
        f"first={axis_first_deg[2]:.6f}, last={axis_last_deg[2]:.6f}"
    )
    print(
        "  axis_max_abs_delta_deg (roll,pitch,yaw)="
        f"[{axis_max_abs_delta_deg[0]:.6f}, {axis_max_abs_delta_deg[1]:.6f}, {axis_max_abs_delta_deg[2]:.6f}]"
    )
    print(
        f"  prev_step_max_abs_delta_peak_deg={prev_step_max_abs_delta_peak_deg:.6f}, "
        f"transition={prev_step_peak_transition}"
    )

    return summary
