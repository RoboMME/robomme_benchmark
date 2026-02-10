#!/usr/bin/env python3
"""
轻量验证脚本：四元数(wxyz) -> RPY(XYZ) 连续化逻辑。

设计目标：
1. 用最小依赖（仅 numpy）复现 wrapper 中的关键连续化流程；
2. 对比 naive 与 stable 两条管线的角度跳变表现；
3. 提供可复用的 RPY 汇总函数，给 replay 脚本直接调用并输出 JSON 友好结果。
"""

import sys

import numpy as np


EPS = 1e-12
SIGN_FLIP_TOL = 1e-6


def normalize_quat_wxyz(quat):
    """
    归一化四元数（wxyz）。

    规则：
    - 输入最后一维必须是 4；
    - 范数过小、非有限值（NaN/Inf）回退为单位四元数 [1, 0, 0, 0]；
    - 输出与输入形状一致（除 dtype 统一为 float64）。
    """
    quat = np.asarray(quat, dtype=np.float64)
    if quat.shape[-1] != 4:
        raise ValueError(f"Quaternion last dimension must be 4, got shape {quat.shape}")

    # 有效性判定：元素有限 + 范数有限 + 范数大于阈值。
    quat_norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    finite_quat = np.isfinite(quat).all(axis=-1, keepdims=True)
    finite_norm = np.isfinite(quat_norm)
    valid = finite_quat & finite_norm & (quat_norm > EPS)

    # 无效项先用 1 做安全除数，再由 where 覆盖为 fallback，避免除零告警污染。
    safe_norm = np.where(valid, quat_norm, 1.0)
    normalized = quat / safe_norm

    # fallback 始终采用单位四元数（w=1, xyz=0）。
    fallback = np.zeros_like(normalized)
    fallback[..., 0] = 1.0
    return np.where(valid, normalized, fallback)


def align_quat_sign_with_prev(quat, prev_quat):
    """
    四元数符号对齐：当 dot(quat, prev_quat) < 0 时翻转当前符号。

    说明：
    - q 与 -q 表示同一旋转；该步骤用于减小跨帧表示不连续导致的数值抖动；
    - 若无上一帧或形状不一致，直接返回当前 quat。
    """
    if prev_quat is None:
        return quat
    if prev_quat.shape != quat.shape:
        return quat

    # dot<0 代表位于球面对径点，翻转后可选取与上一帧更“接近”的表示。
    dot = np.sum(quat * prev_quat, axis=-1, keepdims=True)
    sign = np.where(dot < 0.0, -1.0, 1.0)
    return quat * sign


def quat_wxyz_to_rpy_xyz(quat):
    """
    四元数(wxyz)转欧拉角(roll, pitch, yaw)，约定为 XYZ，单位弧度。

    返回形状：(..., 3)
    """
    quat = np.asarray(quat, dtype=np.float64)
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]

    # roll (X)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (Y)：asin 前做 clip，避免浮点误差导致越界。
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    # yaw (Z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack((roll, pitch, yaw), axis=-1)


def unwrap_rpy_with_prev(rpy, prev_rpy):
    """
    RPY 展开（unwrap）：把相邻帧差分折叠到 (-pi, pi]，再加回上一帧。

    结果特性：
    - 输出跨帧连续；
    - 数值可超出 [-pi, pi]（连续表示，不是主值表示）。
    """
    if prev_rpy is None:
        return rpy
    if prev_rpy.shape != rpy.shape:
        return rpy

    delta = rpy - prev_rpy
    # 把差分映射到 (-pi, pi]，对应“最短角度增量”。
    delta = np.remainder(delta + np.pi, 2.0 * np.pi) - np.pi
    return prev_rpy + delta


class StableRPYTracker:
    """
    按帧维护连续化状态的跟踪器。

    内部状态：
    - prev_quat：上一帧对齐后的四元数表示；
    - prev_rpy：上一帧 unwrap 后的连续 RPY。
    """

    def __init__(self):
        self.prev_quat = None
        self.prev_rpy = None

    def reset(self):
        """清空跨帧状态。用于模拟 episode 边界。"""
        self.prev_quat = None
        self.prev_rpy = None

    def step(self, quat_wxyz):
        """
        处理单帧输入，返回连续 RPY。

        流程：normalize -> 符号对齐 -> quat->rpy -> unwrap -> 更新缓存。
        """
        quat = normalize_quat_wxyz(quat_wxyz)
        quat = align_quat_sign_with_prev(quat, self.prev_quat)
        rpy = quat_wxyz_to_rpy_xyz(quat)
        rpy = unwrap_rpy_with_prev(rpy, self.prev_rpy)
        self.prev_quat = np.array(quat, copy=True)
        self.prev_rpy = np.array(rpy, copy=True)
        return rpy


def euler_xyz_to_quat_wxyz(roll, pitch, yaw):
    """
    欧拉角(XYZ)转四元数(wxyz)，输入/输出均支持批量广播。
    """
    roll = np.asarray(roll, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)

    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr * cp * cy - sr * sp * sy
    x = sr * cp * cy + cr * sp * sy
    y = cr * sp * cy - sr * cp * sy
    z = cr * cp * sy + sr * sp * cy
    return np.stack((w, x, y, z), axis=-1)


def run_naive(quats):
    """naive 管线：仅 normalize + quat->rpy，不做符号对齐与 unwrap。"""
    quats = normalize_quat_wxyz(quats)
    return quat_wxyz_to_rpy_xyz(quats)


def run_stable(quats):
    """stable 管线：逐帧执行符号对齐与 unwrap，输出连续 RPY。"""
    tracker = StableRPYTracker()
    out = [tracker.step(q) for q in quats]
    return np.asarray(out, dtype=np.float64)


def max_abs_delta(rpy_seq):
    """
    计算每轴相邻帧最大绝对差分。

    定义：max_t |rpy[t] - rpy[t-1]|，返回 [d_roll, d_pitch, d_yaw]。
    """
    rpy_seq = np.asarray(rpy_seq, dtype=np.float64)
    if rpy_seq.shape[0] < 2:
        return np.zeros(3, dtype=np.float64)
    return np.max(np.abs(np.diff(rpy_seq, axis=0)), axis=0)


def summarize_and_print_rpy_sequence(rpy_sequence, label=""):
    """
    对一段 RPY 序列做“episode 级”汇总，并同时打印弧度与角度两份结果。

    统计口径说明（与 evaluate_dataset_replay-verifyRPY.py 保持一致）：
    1. 输入按时间顺序排列，视作 rpy[t]。
    2. 相邻差分定义为 diff[t] = rpy[t] - rpy[t-1]（t>=1）。
    3. 每步的“最大角差”定义为 max(|d_roll|, |d_pitch|, |d_yaw|)。
    4. prev_step_max_abs_delta_peak 是上述“每步最大角差”的全序列峰值。
    5. prev_step_peak_transition 记录峰值对应的步迁移 [t-1, t]（JSON 友好的 list）。
    6. 除打印外，返回 dict 结果用于后续写入 JSON，字段均为基础类型/list/None，
       可直接 json.dump，不依赖自定义编码器。

    Parameters
    ----------
    rpy_sequence:
        Sequence of RPY rows. Accepts shape (N, 3) or values convertible to it.
    label:
        Optional prefix label for printing.

    Returns
    -------
    dict with fixed keys:
        count,
        axis_min_rad/axis_max_rad/axis_first_rad/axis_last_rad,
        axis_min_deg/axis_max_deg/axis_first_deg/axis_last_deg,
        axis_max_abs_delta_rad/axis_max_abs_delta_deg,
        prev_step_max_abs_delta_peak_rad/prev_step_max_abs_delta_peak_deg,
        prev_step_peak_transition
    """

    # 统一转 float64，保证统计与打印精度稳定（跨 torch/numpy 输入保持一致行为）。
    rpy = np.asarray(rpy_sequence, dtype=np.float64)
    if rpy.size == 0:
        # 空序列：打印提示并返回“零差值 + None 统计项”，避免调用方做额外判空分支。
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

    # 兼容多种输入形状，最终规范化为 (N, 3)：
    # - (3,) 视为 1 条样本
    # - (3k,) 视为 k 条样本
    # - (..., 3) flatten 为 (-1, 3)
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
        # 单样本无“前后步差分”，将差值类指标置 0，transition 置 None。
        axis_max_abs_delta_rad = np.zeros(3, dtype=np.float64)
        axis_max_abs_delta_deg = np.zeros(3, dtype=np.float64)
        prev_step_max_abs_delta_peak_rad = 0.0
        prev_step_max_abs_delta_peak_deg = 0.0
        prev_step_peak_transition = None
    else:
        # diff: 逐轴相邻差分；abs_diff: 逐轴相邻绝对差。
        diff = np.diff(rpy, axis=0)
        abs_diff = np.abs(diff)
        # 每轴的最大绝对相邻差分（roll/pitch/yaw 各自一个值）。
        axis_max_abs_delta_rad = np.max(abs_diff, axis=0)
        axis_max_abs_delta_deg = np.rad2deg(axis_max_abs_delta_rad)
        # 每一步在三个轴中的最大绝对相邻差分，用于定位“最突变”的那一步。
        prev_step_max_abs_delta = np.max(abs_diff, axis=1)
        peak_idx = int(np.argmax(prev_step_max_abs_delta))
        prev_step_max_abs_delta_peak_rad = float(prev_step_max_abs_delta[peak_idx])
        prev_step_max_abs_delta_peak_deg = float(np.rad2deg(prev_step_max_abs_delta_peak_rad))
        prev_step_peak_transition = [peak_idx, peak_idx + 1]

    # 返回值使用 list/float/None，保证直接写入 JSON 不会遇到 numpy 序列化错误。
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

    # 终端输出分两段：先弧度，再角度，方便和上游/人类阅读两侧需求同时对齐。
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


def scenario_sign_flip_invariance():
    """
    场景1：仅做四元数符号翻转 q / -q 交替，物理姿态保持不变。

    预期：naive 与 stable 都应近似无跳变（用于验证“不变性”）。
    """
    base_q = euler_xyz_to_quat_wxyz(0.31, -0.27, 1.03)
    quats = np.stack([base_q if i % 2 == 0 else -base_q for i in range(40)], axis=0)
    naive = run_naive(quats)
    stable = run_stable(quats)
    return {
        "name": "sign_flip_invariance",
        "naive_max_delta": max_abs_delta(naive),
        "stable_max_delta": max_abs_delta(stable),
        "naive_rpy": naive,
        "stable_rpy": stable,
    }


def scenario_cross_pi_yaw():
    """
    场景2：yaw 跨越 ±pi 边界（170° -> 190°）。

    预期：naive 在边界附近产生大跳变；stable 经过 unwrap 后连续。
    """
    yaw = np.deg2rad(np.linspace(170.0, 190.0, 81))
    zeros = np.zeros_like(yaw)
    quats = euler_xyz_to_quat_wxyz(zeros, zeros, yaw)
    naive = run_naive(quats)
    stable = run_stable(quats)
    return {
        "name": "cross_pi_yaw",
        "naive_max_delta": max_abs_delta(naive),
        "stable_max_delta": max_abs_delta(stable),
        "naive_rpy": naive,
        "stable_rpy": stable,
    }


def scenario_smooth_multi_turn_with_flips():
    """
    场景3：平滑多圈旋转 + 人工符号翻转。

    构造：
    - roll/pitch 为小幅平滑扰动；
    - yaw 跨多圈（显著超过 2*pi）；
    - 随机挑选索引进行 q->-q。
    """
    rng = np.random.default_rng(0)
    n = 400
    t = np.linspace(0.0, 1.0, n)

    roll = 0.05 * np.sin(2.0 * np.pi * 3.0 * t)
    pitch = 0.03 * np.sin(2.0 * np.pi * 2.0 * t + 0.3)
    yaw = np.linspace(-0.5, 8.5 * np.pi, n) + 0.1 * np.sin(2.0 * np.pi * 5.0 * t)

    quats = euler_xyz_to_quat_wxyz(roll, pitch, yaw)
    flip_indices = rng.choice(np.arange(1, n - 1), size=80, replace=False)
    quats[flip_indices] *= -1.0

    naive = run_naive(quats)
    stable = run_stable(quats)
    return {
        "name": "smooth_multi_turn_with_flips",
        "naive_max_delta": max_abs_delta(naive),
        "stable_max_delta": max_abs_delta(stable),
        "naive_rpy": naive,
        "stable_rpy": stable,
    }


def scenario_episode_reset_isolation():
    """
    场景4：验证 episode reset 隔离。

    方法：
    - 先喂一段序列 A 更新 tracker；
    - reset 清空状态；
    - 再喂序列 B，并与“新 tracker 直接跑 B”结果比较。
    """
    tracker = StableRPYTracker()

    yaw_a = np.deg2rad(np.linspace(-160.0, 160.0, 60))
    quats_a = euler_xyz_to_quat_wxyz(np.zeros_like(yaw_a), np.zeros_like(yaw_a), yaw_a)
    for q in quats_a:
        tracker.step(q)

    tracker.reset()

    yaw_b = np.deg2rad(np.linspace(-30.0, 35.0, 50))
    roll_b = 0.02 * np.sin(np.linspace(0.0, 2.0 * np.pi, 50))
    pitch_b = 0.01 * np.cos(np.linspace(0.0, 2.0 * np.pi, 50))
    quats_b = euler_xyz_to_quat_wxyz(roll_b, pitch_b, yaw_b)
    quats_b[::7] *= -1.0

    reset_run = np.asarray([tracker.step(q) for q in quats_b], dtype=np.float64)
    fresh_run = run_stable(quats_b)
    max_abs_diff = float(np.max(np.abs(reset_run - fresh_run)))
    return {
        "name": "episode_reset_isolation",
        "max_abs_diff": max_abs_diff,
    }


def scenario_pose_dimension():
    """维度场景：检查 xyz+rpy 拼接后的末维是否为 6。"""
    tracker = StableRPYTracker()
    pos = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    quat = euler_xyz_to_quat_wxyz(0.2, -0.1, 0.4)
    rpy = tracker.step(quat)
    pose_xyzrpy = np.concatenate((pos, rpy), axis=-1)
    return {
        "name": "pose_dimension",
        "shape": pose_xyzrpy.shape,
    }


def main():
    """
    运行全部场景并按阈值做验收。

    失败返回 1，成功返回 0。
    """
    failures = []

    # 依次执行场景，收集对比指标。
    sign_flip = scenario_sign_flip_invariance()
    cross_pi = scenario_cross_pi_yaw()
    smooth = scenario_smooth_multi_turn_with_flips()
    reset_iso = scenario_episode_reset_isolation()
    pose_dim = scenario_pose_dimension()

    print(
        f"[{sign_flip['name']}] naive max|delta| (r,p,y): {sign_flip['naive_max_delta']} | "
        f"stable max|delta| (r,p,y): {sign_flip['stable_max_delta']}"
    )
    print(
        f"[{cross_pi['name']}] naive max|delta| (r,p,y): {cross_pi['naive_max_delta']} | "
        f"stable max|delta| (r,p,y): {cross_pi['stable_max_delta']}"
    )
    print(
        f"[{smooth['name']}] naive max|delta| (r,p,y): {smooth['naive_max_delta']} | "
        f"stable max|delta| (r,p,y): {smooth['stable_max_delta']}"
    )
    print(f"[{reset_iso['name']}] max abs diff after reset vs fresh: {reset_iso['max_abs_diff']:.12f}")
    print(f"[{pose_dim['name']}] xyz+rpy shape: {pose_dim['shape']}")

    # 验收1：q/-q 不变性（两条管线都应稳定）。
    if not (
        np.all(sign_flip["naive_max_delta"] < SIGN_FLIP_TOL)
        and np.all(sign_flip["stable_max_delta"] < SIGN_FLIP_TOL)
    ):
        failures.append("sign_flip_invariance should be stable for both naive and stable pipelines")

    # 验收2：±pi 穿越场景中 stable 连续且 yaw 单调。
    cross_pi_stable_yaw = cross_pi["stable_rpy"][:, 2]
    cross_pi_stable_monotonic = bool(np.all(np.diff(cross_pi_stable_yaw) >= -1e-9))
    if not (
        cross_pi["naive_max_delta"][2] > 3.0
        and cross_pi["stable_max_delta"][2] < 0.2
        and cross_pi_stable_monotonic
    ):
        failures.append("cross_pi_yaw thresholds or monotonic continuity check failed")

    # 验收3：多圈旋转 + 翻转下 stable 显著抑制 yaw 跳变。
    if not (smooth["naive_max_delta"][2] > 3.0 and smooth["stable_max_delta"][2] < 0.2):
        failures.append("smooth_multi_turn_with_flips thresholds check failed")

    # 验收4：reset 后结果应与“新 tracker”一致。
    if not (reset_iso["max_abs_diff"] < 1e-9):
        failures.append("episode_reset_isolation failed (reset run differs from fresh run)")

    # 验收5：pose 维度为 6（xyz+rpy）。
    if pose_dim["shape"] != (6,):
        failures.append("pose_dimension check failed: expected shape (6,)")

    if failures:
        print("\nFAIL")
        for idx, failure in enumerate(failures, start=1):
            print(f"{idx}. {failure}")
        return 1

    print("\nPASS: all continuity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
