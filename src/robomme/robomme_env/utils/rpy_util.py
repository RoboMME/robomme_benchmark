"""
RPY continuous tool: shared by wrapper and public scripts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def normalize_quat_wxyz_torch(quat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize wxyz quaternion.

    Fallback to unit quaternion [1, 0, 0, 0] for invalid input (zero norm/NaN/Inf).
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
    Align sign with previous frame's quaternion representation.

    If dot(quat, prev_quat) < 0, flip current quaternion sign.
    """
    if prev_quat is None:
        return quat
    if prev_quat.shape != quat.shape:
        return quat

    prev = prev_quat.to(device=quat.device, dtype=quat.dtype)
    dot = torch.sum(quat * prev, dim=-1, keepdim=True)
    sign = torch.where(dot < 0, -torch.ones_like(dot), torch.ones_like(dot))
    return quat * sign


from scipy.spatial.transform import Rotation


def quat_wxyz_to_rpy_xyz_torch(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert wxyz quaternion to XYZ order RPY (radians).
    Use scipy.spatial.transform.Rotation implementation.
    Note: This process blocks gradient propagation and involves CPU/GPU data transfer.
    """
    # Keep input tensor device and dtype
    device = quat.device
    dtype = quat.dtype
    
    # Convert to numpy (CPU)
    quat_np = quat.detach().cpu().numpy()
    
    # scipy needs xyzw format, input is wxyz
    # If single vector (4,) -> (1, 4) processing, squeeze at the end
    is_single = quat_np.ndim == 1
    if is_single:
        quat_np = quat_np[None, :]
        
    # wxyz -> xyzw
    # quat_np: [..., 4] -> w, x, y, z
    w = quat_np[..., 0]
    x = quat_np[..., 1]
    y = quat_np[..., 2]
    z = quat_np[..., 3]
    
    # Re-stack as xyzw
    quat_xyzw = np.stack([x, y, z, w], axis=-1)
    
    # Create Rotation object
    try:
        rot = Rotation.from_quat(quat_xyzw)
        # Convert to euler 'xyz'
        rpy_np = rot.as_euler('xyz', degrees=False)
    except ValueError as e:
        # Handle all-zero or invalid quaternion, fallback to 0
        # scipy is strict, errors on zero norm
        # Simple handling: catch exception and return all 0s, or ensure normalization during preprocessing
        # Here normalize_quat_wxyz_torch is already called externally,
        # but for robustness, return 0 if error occurs
        rpy_np = np.zeros((quat_np.shape[0], 3))

    if is_single:
        rpy_np = rpy_np[0]
        
    return torch.from_numpy(rpy_np).to(device=device, dtype=dtype)


def rpy_xyz_to_quat_wxyz_torch(rpy: torch.Tensor) -> torch.Tensor:
    """
    Convert XYZ order RPY (radians) to wxyz quaternion.
    Use scipy.spatial.transform.Rotation implementation.
    Inverse operation of quat_wxyz_to_rpy_xyz_torch.
    Note: This process blocks gradient propagation and involves CPU/GPU data transfer.
    """
    device = rpy.device
    dtype = rpy.dtype
    
    rpy_np = rpy.detach().cpu().numpy()
    
    is_single = rpy_np.ndim == 1
    if is_single:
        rpy_np = rpy_np[None, :]
        
    # scipy euler 'xyz' -> quat (xyzw)
    rot = Rotation.from_euler('xyz', rpy_np, degrees=False)
    quat_xyzw = rot.as_quat()
    
    # xyzw -> wxyz
    x = quat_xyzw[..., 0]
    y = quat_xyzw[..., 1]
    z = quat_xyzw[..., 2]
    w = quat_xyzw[..., 3]
    
    quat_wxyz = np.stack([w, x, y, z], axis=-1)
    
    if is_single:
        quat_wxyz = quat_wxyz[0]

    # Output normalized (scipy default normalized), convert back to tensor directly
    return torch.from_numpy(quat_wxyz).to(device=device, dtype=dtype)


def unwrap_rpy_with_prev_torch(rpy: torch.Tensor, prev_rpy: torch.Tensor | None) -> torch.Tensor:
    """
    Unwrap RPY relative to previous frame: fold difference into (-pi, pi] then accumulate.
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
    End-effector pose continuous pipeline.

    Pipeline:
    1) quat normalization;
    2) Align quaternion sign with previous frame;
    3) quat -> rpy principal value;
    4) Unwrap based on previous frame to get continuous RPY;
    5) Update cache (aligned quat + unwrapped rpy);
    6) Output {"pose": xyz, "quat": wxyz, "rpy": [roll, pitch, yaw]}.

    Input:
      - position: xyz position
      - quat_wxyz: current frame wxyz quaternion
      - prev_ee_quat_wxyz / prev_ee_rpy_xyz: previous frame cache (None = no cache)

    Return:
      - pose_dict: {"pose": position, "quat": aligned quat, "rpy": continuous RPY}
      - new_prev_quat: updated cache quat (detach+clone)
      - new_prev_rpy: updated cache rpy (detach+clone)
    """
    quat_normalized = normalize_quat_wxyz_torch(quat_wxyz, eps=eps)
    quat_aligned = align_quat_sign_with_prev_torch(quat_normalized, prev_ee_quat_wxyz)
    rpy_xyz = quat_wxyz_to_rpy_xyz_torch(quat_aligned)
    rpy_xyz_unwrapped = unwrap_rpy_with_prev_torch(rpy_xyz, prev_ee_rpy_xyz)

    new_prev_quat = quat_aligned.detach().clone()
    new_prev_rpy = rpy_xyz_unwrapped.detach().clone()

    pose_dict = {
        "pose": position,          # xyz position
        "quat": quat_aligned,      # wxyz quaternion (normalized + sign aligned)
        "rpy": rpy_xyz_unwrapped,  # continuous RPY (roll, pitch, yaw)
    }
    return pose_dict, new_prev_quat, new_prev_rpy


def summarize_and_print_rpy_sequence(rpy_sequence: Any, label: str = "") -> dict[str, Any]:
    """
    Summarize an RPY sequence and print report containing only count and delta.
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
        logger.debug(f"{prefix}RPY summary: no RPY samples.")
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
    logger.debug(f"{prefix}RPY summary (rad):")
    logger.debug(f"  count={count}")
    logger.debug(
        "  axis_max_abs_delta_rad (roll,pitch,yaw)="
        f"[{axis_max_abs_delta_rad[0]:.6f}, {axis_max_abs_delta_rad[1]:.6f}, {axis_max_abs_delta_rad[2]:.6f}]"
    )
    logger.debug(f"  transitions={axis_max_abs_delta_transition}")
    logger.debug(f"{prefix}RPY summary (deg):")
    logger.debug(
        "  axis_max_abs_delta_deg (roll,pitch,yaw)="
        f"[{axis_max_abs_delta_deg[0]:.6f}, {axis_max_abs_delta_deg[1]:.6f}, {axis_max_abs_delta_deg[2]:.6f}]"
    )

    return summary
