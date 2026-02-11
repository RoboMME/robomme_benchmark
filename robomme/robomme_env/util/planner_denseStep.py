"""
逐步收集规划工具。

所有逐步收集接口统一返回 5 元组：
    (obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch)

- obs_batch/info_batch: dict[str, list]，键来自单步字典。
- reward_batch: torch.float32，形状为 [N]。
- terminated_batch/truncated_batch: torch.bool，形状为 [N]。
"""

import numpy as np
import torch


def _to_scalar(value):
    """将标量或张量值转换为 Python 标量。"""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0
        return value.reshape(-1)[0].item()
    return value


def _snapshot_value(value):
    """对可能复用底层内存的值做快照，避免后续步骤覆盖前序帧。"""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: _snapshot_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_snapshot_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_value(v) for v in value)
    return value


def _snapshot_step(out):
    """对单步输出做深拷贝快照。"""
    if not (isinstance(out, tuple) and len(out) == 5):
        return out
    obs, reward, terminated, truncated, info = out
    return (
        _snapshot_value(obs),
        _snapshot_value(reward),
        _snapshot_value(terminated),
        _snapshot_value(truncated),
        _snapshot_value(info),
    )


def _is_columnar_dict(batch_dict, n):
    if not isinstance(batch_dict, dict):
        return False
    for value in batch_dict.values():
        if not isinstance(value, list):
            return False
        if len(value) != n:
            return False
    return True


def _output_to_steps(out):
    """
    将 step 输出规范化为“原始逐步 tuple 列表”。
    同时支持单步 tuple 和统一批次 tuple 两种输入格式。
    """
    if isinstance(out, tuple) and len(out) == 5:
        obs_part, reward_part, terminated_part, truncated_part, info_part = out
        if (
            isinstance(reward_part, torch.Tensor)
            and isinstance(terminated_part, torch.Tensor)
            and isinstance(truncated_part, torch.Tensor)
            and reward_part.ndim == 1
            and terminated_part.ndim == 1
            and truncated_part.ndim == 1
        ):
            n = int(reward_part.numel())
            if (
                terminated_part.numel() == n
                and truncated_part.numel() == n
                and _is_columnar_dict(obs_part, n)
                and _is_columnar_dict(info_part, n)
            ):
                steps = []
                obs_keys = list(obs_part.keys())
                info_keys = list(info_part.keys())
                for idx in range(n):
                    obs = {k: _snapshot_value(obs_part[k][idx]) for k in obs_keys}
                    info = {k: _snapshot_value(info_part[k][idx]) for k in info_keys}
                    steps.append(
                        (
                            obs,
                            _snapshot_value(reward_part[idx]),
                            _snapshot_value(terminated_part[idx]),
                            _snapshot_value(truncated_part[idx]),
                            info,
                        )
                    )
                return steps
    return [_snapshot_step(out)]


def _dicts_to_columnar_dict(dict_steps):
    """
    将逐步字典转换为 dict[str, list]，缺失键使用 None 补齐。
    """
    n = len(dict_steps)
    out = {}
    for idx, item in enumerate(dict_steps):
        current = item if isinstance(item, dict) else {}
        for key in current:
            if key not in out:
                out[key] = [None] * idx
        for key in out:
            out[key].append(current.get(key, None))
    for key in out:
        if len(out[key]) < n:
            out[key].extend([None] * (n - len(out[key])))
    return out


def empty_step_batch():
    """按统一契约返回一个空 batch。"""
    return (
        {},
        torch.empty(0, dtype=torch.float32),
        torch.empty(0, dtype=torch.bool),
        torch.empty(0, dtype=torch.bool),
        {},
    )


def to_step_batch(collected_steps):
    """
    将收集到的逐步 tuple 转换为统一 batch 输出。
    collected_steps: [(obs, reward, terminated, truncated, info), ...]
    """
    if not collected_steps:
        return empty_step_batch()

    obs_steps = [x[0] for x in collected_steps]
    reward_steps = [_to_scalar(x[1]) for x in collected_steps]
    terminated_steps = [bool(_to_scalar(x[2])) for x in collected_steps]
    truncated_steps = [bool(_to_scalar(x[3])) for x in collected_steps]
    info_steps = [x[4] for x in collected_steps]

    obs_batch = _dicts_to_columnar_dict(obs_steps)
    info_batch = _dicts_to_columnar_dict(info_steps)
    reward_batch = torch.tensor(reward_steps, dtype=torch.float32)
    terminated_batch = torch.tensor(terminated_steps, dtype=torch.bool)
    truncated_batch = torch.tensor(truncated_steps, dtype=torch.bool)
    return (obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch)


def concat_step_batches(batches):
    """
    将多个统一 batch 拼接为一个统一 batch。
    """
    valid = []
    for batch in batches:
        if batch is None:
            continue
        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = batch
        if reward_batch.numel() == 0:
            continue
        valid.append((obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch))
    if not valid:
        return empty_step_batch()

    obs_out = {}
    info_out = {}
    reward_out = []
    terminated_out = []
    truncated_out = []
    n_total = 0

    for obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch in valid:
        n = int(reward_batch.numel())
        for key in obs_batch:
            if key not in obs_out:
                obs_out[key] = [None] * n_total
        for key in obs_out:
            values = obs_batch.get(key, None)
            if values is None:
                obs_out[key].extend([None] * n)
            else:
                obs_out[key].extend(list(values))

        for key in info_batch:
            if key not in info_out:
                info_out[key] = [None] * n_total
        for key in info_out:
            values = info_batch.get(key, None)
            if values is None:
                info_out[key].extend([None] * n)
            else:
                info_out[key].extend(list(values))

        reward_out.append(reward_batch.reshape(-1).to(torch.float32))
        terminated_out.append(terminated_batch.reshape(-1).to(torch.bool))
        truncated_out.append(truncated_batch.reshape(-1).to(torch.bool))
        n_total += n

    return (
        obs_out,
        torch.cat(reward_out, dim=0) if reward_out else torch.empty(0, dtype=torch.float32),
        torch.cat(terminated_out, dim=0) if terminated_out else torch.empty(0, dtype=torch.bool),
        torch.cat(truncated_out, dim=0) if truncated_out else torch.empty(0, dtype=torch.bool),
        info_out,
    )


def _collect_dense_steps(planner, fn):
    """
    运行 fn() 时拦截 planner.env.step，收集原始逐步 tuple。
    若 fn 返回 -1，则返回 -1；否则返回收集结果列表。
    """
    collected = []
    original_step = planner.env.step

    def _step(action):
        out = original_step(action)
        collected.extend(_output_to_steps(out))
        return out

    planner.env.step = _step
    try:
        result = fn()
        if result == -1:
            return -1
        return collected
    finally:
        planner.env.step = original_step


def _run_with_dense_collection(planner, fn):
    """
    运行 fn() 并返回统一批次；若 fn 返回 -1 则返回 -1。
    """
    collected = _collect_dense_steps(planner, fn)
    if collected == -1:
        return -1
    return to_step_batch(collected)


def move_to_pose_with_RRTStar(planner, pose):
    """
    调用 planner.move_to_pose_with_RRTStar(pose) 并返回统一批次。
    规划失败时返回 -1。
    """
    return _run_with_dense_collection(
        planner, lambda: planner.move_to_pose_with_RRTStar(pose)
    )


def move_to_pose_with_screw(planner, pose):
    """
    调用 planner.move_to_pose_with_screw(pose) 并返回统一批次。
    规划失败时返回 -1。
    """
    return _run_with_dense_collection(
        planner, lambda: planner.move_to_pose_with_screw(pose)
    )


def close_gripper(planner):
    """
    调用 planner.close_gripper() 并返回统一批次。
    失败时返回 -1。
    """
    return _run_with_dense_collection(planner, lambda: planner.close_gripper())


def open_gripper(planner):
    """
    调用 planner.open_gripper() 并返回统一批次。
    失败时返回 -1。
    """
    return _run_with_dense_collection(planner, lambda: planner.open_gripper())


# ---- 调用关系 ----
#
# _collect_dense_steps:
#   - DemonstrationWrapper.get_demonstration_trajectory()
#     包裹整个 solve_callable，monkey-patch planner.env.step 收集所有底层 step
#
# _run_with_dense_collection:
#   - OraclePlannerDemonstrationWrapper
#     包裹 solve_options 的 solve()，收集所有底层 step 并直接返回统一批次
#
# move_to_pose_with_RRTStar:
#   - MultiStepDemonstrationWrapper 中单步执行移动
#     （MultiStepDemonstrationWrapper.py 第 106 行）
#
# move_to_pose_with_screw:
#   - 目前无外部调用，作为与 move_to_pose_with_RRTStar 对称的 API 保留
#
# close_gripper:
#   - MultiStepDemonstrationWrapper 中执行夹爪关闭
#     （MultiStepDemonstrationWrapper.py 第 112 行）
#
# open_gripper:
#   - MultiStepDemonstrationWrapper 中执行夹爪打开
#     （MultiStepDemonstrationWrapper.py 第 121 行）
