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


def rpy_xyz_to_quat_wxyz_torch(rpy: torch.Tensor) -> torch.Tensor:
    """
    将 XYZ 顺序 RPY（弧度）转换为 wxyz 四元数。

    与 quat_wxyz_to_rpy_xyz_torch 互为逆操作。
    使用 intrinsic XYZ（= extrinsic ZYX）Tait-Bryan 组合：
        q = q_x(roll) ⊗ q_y(pitch) ⊗ q_z(yaw)
    输出经过归一化，防止浮点积累误差。
    """
    roll, pitch, yaw = rpy.unbind(dim=-1)

    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    w = cr * cp * cy - sr * sp * sy
    x = sr * cp * cy + cr * sp * sy
    y = cr * sp * cy - sr * cp * sy
    z = cr * cp * sy + sr * sp * cy

    quat = torch.stack((w, x, y, z), dim=-1)
    return normalize_quat_wxyz_torch(quat)


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


def build_endeffector_pose_dict(
    position: torch.Tensor,
    quat_wxyz: torch.Tensor,
    prev_ee_quat_wxyz: torch.Tensor | None,
    prev_ee_rpy_xyz: torch.Tensor | None,
    eps: float = 1e-12,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """
    末端位姿连续化流水线。

    流水线：
    1) quat 归一化；
    2) 与上一帧做四元数符号对齐；
    3) quat -> rpy 主值；
    4) 基于上一帧做 unwrap，得到连续 RPY；
    5) 更新缓存（对齐后 quat + unwrap 后 rpy）；
    6) 输出 {"pose": xyz, "quat": wxyz, "rpy": [roll, pitch, yaw]}。

    输入：
      - position: xyz 位置
      - quat_wxyz: 当前帧 wxyz 四元数
      - prev_ee_quat_wxyz / prev_ee_rpy_xyz: 上一帧缓存（None = 无缓存）

    返回：
      - pose_dict: {"pose": position, "quat": 对齐后 quat, "rpy": 连续化 RPY}
      - new_prev_quat: 更新后的缓存 quat（detach+clone）
      - new_prev_rpy: 更新后的缓存 rpy（detach+clone）
    """
    quat_normalized = normalize_quat_wxyz_torch(quat_wxyz, eps=eps)
    quat_aligned = align_quat_sign_with_prev_torch(quat_normalized, prev_ee_quat_wxyz)
    rpy_xyz = quat_wxyz_to_rpy_xyz_torch(quat_aligned)
    rpy_xyz_unwrapped = unwrap_rpy_with_prev_torch(rpy_xyz, prev_ee_rpy_xyz)

    new_prev_quat = quat_aligned.detach().clone()
    new_prev_rpy = rpy_xyz_unwrapped.detach().clone()

    pose_dict = {
        "pose": position,          # xyz 位置
        "quat": quat_aligned,      # wxyz 四元数（归一化 + 符号对齐后）
        "rpy": rpy_xyz_unwrapped,  # 连续化 RPY (roll, pitch, yaw)
    }
    return pose_dict, new_prev_quat, new_prev_rpy


def summarize_and_print_rpy_sequence(rpy_sequence: Any, label: str = "") -> dict[str, Any]:
    """
    汇总一段 RPY 序列，并打印仅含 count 和 delta 的报告。
    """
    rpy = np.asarray(rpy_sequence, dtype=np.float64)
    if rpy.size == 0:
        summary = {
            "count": 0,
            "axis_max_abs_delta_rad": [0.0, 0.0, 0.0],
            "axis_max_abs_delta_deg": [0.0, 0.0, 0.0],
            "axis_max_abs_delta_transition": [None, None, None],
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

    if count < 2:
        axis_max_abs_delta_rad = np.zeros(3, dtype=np.float64)
        axis_max_abs_delta_deg = np.zeros(3, dtype=np.float64)
        axis_max_abs_delta_transition = [None, None, None]
    else:
        diff = np.diff(rpy, axis=0)
        abs_diff = np.abs(diff)
        axis_max_abs_delta_rad = np.max(abs_diff, axis=0)
        axis_max_abs_delta_deg = np.rad2deg(axis_max_abs_delta_rad)

        peak_indices = np.argmax(abs_diff, axis=0)
        axis_max_abs_delta_transition = [[int(i), int(i) + 1] for i in peak_indices]

    summary = {
        "count": count,
        "axis_max_abs_delta_rad": axis_max_abs_delta_rad.tolist(),
        "axis_max_abs_delta_deg": axis_max_abs_delta_deg.tolist(),
        "axis_max_abs_delta_transition": axis_max_abs_delta_transition,
    }

    prefix = f"{label} " if label else ""
    print(f"{prefix}RPY summary (rad):")
    print(f"  count={count}")
    print(
        "  axis_max_abs_delta_rad (roll,pitch,yaw)="
        f"[{axis_max_abs_delta_rad[0]:.6f}, {axis_max_abs_delta_rad[1]:.6f}, {axis_max_abs_delta_rad[2]:.6f}]"
    )
    print(f"  transitions={axis_max_abs_delta_transition}")
    print(f"{prefix}RPY summary (deg):")
    print(
        "  axis_max_abs_delta_deg (roll,pitch,yaw)="
        f"[{axis_max_abs_delta_deg[0]:.6f}, {axis_max_abs_delta_deg[1]:.6f}, {axis_max_abs_delta_deg[2]:.6f}]"
    )

    return summary
