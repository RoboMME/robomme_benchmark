"""
Step-by-step planning collection tool.

All step-by-step collection interfaces uniformly return a 5-tuple:
    (obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch)

- obs_batch/info_batch: dict[str, list], keys from single step dictionary.
- reward_batch: torch.float32, shape [N].
- terminated_batch/truncated_batch: torch.bool, shape [N].
"""

import numpy as np
import torch


def _collapse_singleton_lists(value, key=None):
    """Recursively unwrap singleton lists while preserving non-singleton lists."""
    if key == "task_goal":
        return value
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _to_scalar(value):
    """Convert scalar or tensor value to Python scalar."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0
        return value.reshape(-1)[0].item()
    return value


def _snapshot_value(value):
    """Snapshot values that might reuse underlying memory to avoid overwriting previous frames in subsequent steps."""
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
    """Deep copy snapshot of single step output."""
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
    Normalize step output to "list of raw step tuples".
    Supports both single step tuple and unified batch tuple input formats.
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
    Convert step dictionaries to dict[str, list], filling missing keys with None.
    """
    n = len(dict_steps)
    out = {}
    for idx, item in enumerate(dict_steps):
        current = item if isinstance(item, dict) else {}
        for key in current:
            if key not in out:
                out[key] = [None] * idx
        for key in out:
            out[key].append(_collapse_singleton_lists(current.get(key, None), key=key))
    for key in out:
        if len(out[key]) < n:
            out[key].extend([None] * (n - len(out[key])))
    return out


def empty_step_batch():
    """Return an empty batch following the unified contract."""
    return (
        {},
        torch.empty(0, dtype=torch.float32),
        torch.empty(0, dtype=torch.bool),
        torch.empty(0, dtype=torch.bool),
        {},
    )


def to_step_batch(collected_steps):
    """
    Convert collected step tuples to unified batch output.
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
    Concatenate multiple unified batches into one unified batch.
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
                obs_out[key].extend(_collapse_singleton_lists(v, key=key) for v in values)

        for key in info_batch:
            if key not in info_out:
                info_out[key] = [None] * n_total
        for key in info_out:
            values = info_batch.get(key, None)
            if values is None:
                info_out[key].extend([None] * n)
            else:
                info_out[key].extend(_collapse_singleton_lists(v, key=key) for v in values)

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
    Intercept planner.env.step when running fn(), collecting raw step tuples.
    If fn returns -1, return -1; otherwise return collected result list.
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
    Run fn() and return unified batch; return -1 if fn returns -1.
    """
    collected = _collect_dense_steps(planner, fn)
    if collected == -1:
        return -1
    return to_step_batch(collected)


def move_to_pose_with_RRTStar(planner, pose):
    """
    Call planner.move_to_pose_with_RRTStar(pose) and return unified batch.
    Return -1 on planning failure.
    """
    return _run_with_dense_collection(
        planner, lambda: planner.move_to_pose_with_RRTStar(pose)
    )


def move_to_pose_with_screw(planner, pose):
    """
    Call planner.move_to_pose_with_screw(pose) and return unified batch.
    Return -1 on planning failure.
    """
    return _run_with_dense_collection(
        planner, lambda: planner.move_to_pose_with_screw(pose)
    )


def close_gripper(planner):
    """
    Call planner.close_gripper() and return unified batch.
    Return -1 on failure.
    """
    return _run_with_dense_collection(planner, lambda: planner.close_gripper())


def open_gripper(planner):
    """
    Call planner.open_gripper() and return unified batch.
    Return -1 on failure.
    """
    return _run_with_dense_collection(planner, lambda: planner.open_gripper())


# ---- Call Relationships ----
#
# _collect_dense_steps:
#   - DemonstrationWrapper.get_demonstration_trajectory()
#     Wrap entire solve_callable, monkey-patch planner.env.step to collect all underlying steps
#
# _run_with_dense_collection:
#   - OraclePlannerDemonstrationWrapper
#     Wrap solve() in solve_options, collect all underlying steps and directly return unified batch
#
# move_to_pose_with_RRTStar:
#   - Execute single step move in MultiStepDemonstrationWrapper
#     (MultiStepDemonstrationWrapper.py line 106)
#
# move_to_pose_with_screw:
#   - Currently no external call, reserved as symmetric API to move_to_pose_with_RRTStar
#
# close_gripper:
#   - Execute gripper close in MultiStepDemonstrationWrapper
#     (MultiStepDemonstrationWrapper.py line 112)
#
# open_gripper:
#   - Execute gripper open in MultiStepDemonstrationWrapper
#     (MultiStepDemonstrationWrapper.py line 121)
