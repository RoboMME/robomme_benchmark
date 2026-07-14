#!/usr/bin/env python3
"""离线比较官方与重新生成 RoboMME 数据中的 joint action。

该工具只把文件/episode、seed/difficulty、timestep、joint 数据契约以及最终
``info/is_completed`` 作为验证条件。joint 数值差异会完整统计，但不设置容差
阈值，也不参与 ``validation_passed`` 或退出码 0/1 的判定。

默认范围是正式恢复产物的 16 个任务、episode 0--8。所有 HDF5 都以只读模式
打开；输入、JSON 和 Markdown 路径都必须位于 workspace 内且不得经过符号链接。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

import h5py
import numpy as np


CANDIDATE_COMMIT = "a3842d1b77bc79e2f70cefcbab136207e7067065"
REFERENCE_REVISION = "a5e4e25ffe8af34f64944f9533d06455ce5f8337"
ROOT_UV_LOCK_SHA256 = (
    "983de83f7b22c98b96c3c25a39958b4f5920e3232cfaa209c89542ef5639ac03"
)
CANDIDATE_UV_LOCK_SHA256 = (
    "af4a645421c486ca1b1f27f5e54e8043497434b4efc49d2cbbf5eaa1b79d532e"
)
EXPECTED_TASK_IDS = (
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
)
DEFAULT_EPISODES = tuple(range(9))
DEFAULT_REFERENCE_DIR = "data/robomme_data_h5"
DEFAULT_GENERATED_DIR = (
    f"artifacts/generated/{CANDIDATE_COMMIT}/official-train-episodes-0-8"
)
DEFAULT_JSON_REPORT = (
    "artifacts/reports/generated/"
    f"{CANDIDATE_COMMIT}/official-train-episodes-0-8/joint_16x9/comparison.json"
)
DEFAULT_MARKDOWN_REPORT = "scripts/reports/DATASET_COMPARISON_16x9.md"

EPISODE_PATTERN = re.compile(r"^episode_(0|[1-9][0-9]*)$")
TIMESTEP_PATTERN = re.compile(r"^timestep_(0|[1-9][0-9]*)$")
EPISODE_TOKEN_PATTERN = re.compile(r"^(0|[1-9][0-9]*)(?:[-:](0|[1-9][0-9]*))?$")
_MISSING = object()


class UnsafePathError(ValueError):
    """输入或输出路径不满足仓库边界和无符号链接约束。"""


class InputReadError(RuntimeError):
    """输入 HDF5 已定位，但无法以只读方式打开或读取。"""


def _absolute_path(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_workspace(path: str | Path) -> Path:
    workspace = _absolute_path(path)
    if not workspace.is_dir() or workspace.is_symlink():
        raise UnsafePathError(f"workspace-root 不是无符号链接的已有目录：{workspace}")
    resolved = workspace.resolve(strict=True)
    if resolved != workspace:
        raise UnsafePathError(
            f"workspace-root 路径中包含符号链接：{workspace} -> {resolved}"
        )
    return workspace


def _safe_input_directory(
    path: str | Path,
    *,
    workspace: Path,
    label: str,
) -> Path:
    directory = _absolute_path(path)
    if not _is_relative_to(directory, workspace):
        raise UnsafePathError(f"{label} 位于仓库外：{directory}")
    if not directory.is_dir() or directory.is_symlink():
        raise UnsafePathError(f"{label} 不是无符号链接的已有目录：{directory}")
    resolved = directory.resolve(strict=True)
    if resolved != directory or not _is_relative_to(resolved, workspace):
        raise UnsafePathError(f"{label} 路径中包含符号链接：{directory} -> {resolved}")

    try:
        entries = list(os.scandir(directory))
    except OSError as exc:
        raise UnsafePathError(f"无法扫描 {label}：{exc}") from exc
    symlinks = sorted(entry.name for entry in entries if entry.is_symlink())
    if symlinks:
        raise UnsafePathError(
            f"{label} 顶层禁止符号链接：{', '.join(symlinks)}"
        )
    return directory


def _safe_input_file(path: Path, *, root: Path, label: str) -> Path:
    if not _is_relative_to(path, root):
        raise UnsafePathError(f"{label} 位于输入目录外：{path}")
    if not path.is_file() or path.is_symlink():
        raise UnsafePathError(f"{label} 不是无符号链接的普通文件：{path}")
    resolved = path.resolve(strict=True)
    if resolved != path or not _is_relative_to(resolved, root):
        raise UnsafePathError(f"{label} 路径中包含符号链接：{path} -> {resolved}")
    return path


def _safe_output_path(
    path: str | Path,
    *,
    workspace: Path,
    input_roots: Sequence[Path],
    label: str,
) -> Path:
    output = _absolute_path(path)
    if output == workspace or not _is_relative_to(output, workspace):
        raise UnsafePathError(f"{label} 必须位于仓库内：{output}")
    for input_root in input_roots:
        if output == input_root or _is_relative_to(output, input_root):
            raise UnsafePathError(f"{label} 不能写入只读输入目录：{output}")

    current = workspace
    for part in output.parent.relative_to(workspace).parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise UnsafePathError(f"{label} 路径中包含符号链接：{current}")
        if current.exists() and not current.is_dir():
            raise UnsafePathError(f"{label} 的父路径不是目录：{current}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.resolve(strict=True) != output.parent:
        raise UnsafePathError(f"{label} 父路径中包含符号链接：{output.parent}")
    if output.is_symlink() or (output.exists() and not output.is_file()):
        raise UnsafePathError(f"{label} 必须是普通文件或尚不存在：{output}")
    return output


def _parse_episode_tokens(tokens: Sequence[str]) -> tuple[int, ...]:
    episodes: list[int] = []
    for token in tokens:
        for part in token.split(","):
            item = part.strip()
            match = EPISODE_TOKEN_PATTERN.fullmatch(item)
            if match is None:
                raise ValueError(
                    "episode 必须是非负整数或闭区间 START-END/START:END："
                    f"{item!r}"
                )
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) is not None else start
            if end < start:
                raise ValueError(f"episode 区间终点小于起点：{item!r}")
            episodes.extend(range(start, end + 1))
    if not episodes:
        raise ValueError("至少需要一个 episode")
    if len(set(episodes)) != len(episodes):
        raise ValueError("episode 范围不能重叠或重复")
    return tuple(sorted(episodes))


def _validate_scope(
    task_ids: Sequence[str], episodes: Sequence[int]
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    normalized_tasks = tuple(task_ids)
    normalized_episodes = tuple(episodes)
    if not normalized_tasks or len(set(normalized_tasks)) != len(normalized_tasks):
        raise ValueError("tasks 必须非空且不能重复")
    unknown = sorted(set(normalized_tasks) - set(EXPECTED_TASK_IDS))
    if unknown:
        raise ValueError(f"未知任务：{', '.join(unknown)}")
    if not normalized_episodes or len(set(normalized_episodes)) != len(
        normalized_episodes
    ):
        raise ValueError("episodes 必须非空且不能重复")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in normalized_episodes
    ):
        raise ValueError("episodes 必须全部为非负整数")
    return normalized_tasks, tuple(sorted(normalized_episodes))


def _error(
    errors: list[dict[str, Any]],
    code: str,
    message: str,
    **details: Any,
) -> None:
    item: dict[str, Any] = {"code": code, "message": message}
    item.update(details)
    errors.append(item)


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _strict_scalar_equal(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


@dataclass
class CompletionCounter:
    expected: int = 0
    valid: int = 0
    successful: int = 0

    def observe(self, value: bool | None) -> None:
        if value is None:
            return
        self.valid += 1
        self.successful += int(value)

    def summary(self) -> dict[str, Any]:
        return {
            "expected_trajectories": self.expected,
            "valid_boolean_statuses": self.valid,
            "successful_trajectories": self.successful,
            "success_rate": self.successful / self.expected if self.expected else None,
        }


@dataclass
class JointAccumulator:
    """流式累计 joint 指标；只为精确分位数保留扁平 absolute diff。"""

    path_count: int = 0
    element_count: int = 0
    strict_equal_path_count: int = 0
    strict_equal_element_count: int = 0
    difference_sum: float = 0.0
    squared_difference_sum: float = 0.0
    max_abs_difference: float = -1.0
    worst_difference: dict[str, Any] | None = None
    _absolute_chunks: list[np.ndarray[Any, np.dtype[np.float64]]] = field(
        default_factory=list,
        repr=False,
    )

    def update(
        self,
        reference: np.ndarray[Any, Any],
        generated: np.ndarray[Any, Any],
        *,
        path: str,
    ) -> None:
        reference_float = np.asarray(reference, dtype=np.float64)
        generated_float = np.asarray(generated, dtype=np.float64)
        absolute = np.abs(reference_float - generated_float).reshape(-1)
        equal = np.equal(reference, generated).reshape(-1)

        self.path_count += 1
        self.element_count += int(equal.size)
        equal_count = int(np.count_nonzero(equal))
        self.strict_equal_element_count += equal_count
        self.strict_equal_path_count += int(equal_count == equal.size)
        self.difference_sum += float(np.sum(absolute, dtype=np.float64))
        self.squared_difference_sum += float(
            np.sum(np.square(absolute), dtype=np.float64)
        )
        self._absolute_chunks.append(absolute.copy())

        if absolute.size:
            flat_index = int(np.argmax(absolute))
            candidate = float(absolute[flat_index])
            if candidate > self.max_abs_difference:
                source_flat = np.asarray(reference).reshape(-1)
                generated_flat = np.asarray(generated).reshape(-1)
                element_index = [
                    int(item)
                    for item in np.unravel_index(flat_index, reference.shape)
                ]
                self.max_abs_difference = candidate
                self.worst_difference = {
                    "path": path,
                    "flat_index": flat_index,
                    "element_index": element_index,
                    "reference_value": _python_scalar(source_flat[flat_index]),
                    "generated_value": _python_scalar(generated_flat[flat_index]),
                    "absolute_difference": candidate,
                }

    def summary(self) -> dict[str, Any]:
        differing_elements = self.element_count - self.strict_equal_element_count
        differing_paths = self.path_count - self.strict_equal_path_count
        if self.element_count:
            mean = self.difference_sum / self.element_count
            rmse = math.sqrt(self.squared_difference_sum / self.element_count)
            values = np.concatenate(self._absolute_chunks)
            p50, p95, p99 = (
                float(item) for item in np.quantile(values, (0.50, 0.95, 0.99))
            )
            maximum: float | None = max(self.max_abs_difference, 0.0)
        else:
            mean = None
            rmse = None
            p50 = p95 = p99 = None
            maximum = None
        return {
            "joint_path_count": self.path_count,
            "joint_element_count": self.element_count,
            "joint_strict_equal_path_count": self.strict_equal_path_count,
            "joint_differing_path_count": differing_paths,
            "joint_strict_equal_element_count": self.strict_equal_element_count,
            "joint_differing_element_count": differing_elements,
            "joint_difference_element_ratio": (
                differing_elements / self.element_count if self.element_count else None
            ),
            "max_abs_difference": maximum,
            "mean_abs_difference": mean,
            "rmse": rmse,
            "p50_abs_difference": p50,
            "p95_abs_difference": p95,
            "p99_abs_difference": p99,
            "worst_difference": self.worst_difference,
            "joint_strict_equal": bool(
                self.path_count > 0 and differing_elements == 0
            ),
        }


def _hard_group(
    parent: h5py.Group | h5py.File,
    name: str,
    *,
    errors: list[dict[str, Any]],
    code: str,
    message: str,
    details: dict[str, Any],
) -> h5py.Group | None:
    link = parent.get(name, getlink=True)
    if not isinstance(link, h5py.HardLink):
        _error(errors, code, message, path=f"{parent.name}/{name}", **details)
        return None
    obj = parent.get(name)
    if not isinstance(obj, h5py.Group):
        _error(errors, code, message, path=f"{parent.name}/{name}", **details)
        return None
    return obj


def _hard_dataset(
    parent: h5py.Group,
    name: str,
    *,
    errors: list[dict[str, Any]],
    code: str,
    message: str,
    details: dict[str, Any],
) -> h5py.Dataset | None:
    link = parent.get(name, getlink=True)
    if not isinstance(link, h5py.HardLink):
        _error(errors, code, message, path=f"{parent.name}/{name}", **details)
        return None
    obj = parent.get(name)
    if not isinstance(obj, h5py.Dataset):
        _error(errors, code, message, path=f"{parent.name}/{name}", **details)
        return None
    return obj


def _read_setup_value(
    episode_group: h5py.Group,
    field_name: str,
    *,
    role: str,
    task_id: str,
    episode: int,
    errors: list[dict[str, Any]],
) -> Any:
    details = {"role": role, "task_id": task_id, "episode": episode}
    setup = _hard_group(
        episode_group,
        "setup",
        errors=errors,
        code="setup_missing_or_invalid",
        message="episode/setup 必须是普通 hard-link group",
        details=details,
    )
    if setup is None:
        return _MISSING
    dataset = _hard_dataset(
        setup,
        field_name,
        errors=errors,
        code="setup_field_missing_or_invalid",
        message=f"episode/setup/{field_name} 必须是普通 hard-link dataset",
        details=details,
    )
    if dataset is None:
        return _MISSING
    if dataset.shape != ():
        _error(
            errors,
            "setup_field_not_scalar",
            f"episode/setup/{field_name} 必须是标量",
            path=dataset.name,
            actual_shape=list(dataset.shape),
            **details,
        )
        return _MISSING
    try:
        return _python_scalar(dataset[()])
    except (OSError, RuntimeError, TypeError, UnicodeError, ValueError) as exc:
        raise InputReadError(f"无法读取 {role} {task_id} {dataset.name}：{exc}") from exc


def _episode_indices(
    data: h5py.File,
    *,
    role: str,
    task_id: str,
    selected: tuple[int, ...],
    errors: list[dict[str, Any]],
) -> tuple[int, ...]:
    malformed = sorted(name for name in data.keys() if EPISODE_PATTERN.fullmatch(name) is None)
    for name in malformed:
        _error(
            errors,
            "unexpected_top_level_entry",
            "HDF5 顶层只允许规范的 episode_N group",
            role=role,
            task_id=task_id,
            path=f"/{name}",
        )
    indices = tuple(
        sorted(
            int(match.group(1))
            for name in data.keys()
            if (match := EPISODE_PATTERN.fullmatch(name)) is not None
        )
    )
    missing = sorted(set(selected) - set(indices))
    if missing:
        _error(
            errors,
            "selected_episode_missing",
            "HDF5 缺少所选 episode",
            role=role,
            task_id=task_id,
            missing=missing,
        )
    return indices


def _episode_group(
    data: h5py.File,
    *,
    role: str,
    task_id: str,
    episode: int,
    errors: list[dict[str, Any]],
) -> h5py.Group | None:
    return _hard_group(
        data,
        f"episode_{episode}",
        errors=errors,
        code="episode_missing_or_invalid",
        message="所选 episode 必须是普通 hard-link group",
        details={"role": role, "task_id": task_id, "episode": episode},
    )


def _timestep_indices(
    episode_group: h5py.Group,
    *,
    role: str,
    task_id: str,
    episode: int,
    errors: list[dict[str, Any]],
) -> tuple[int, ...]:
    malformed = sorted(
        name
        for name in episode_group.keys()
        if name.startswith("timestep_") and TIMESTEP_PATTERN.fullmatch(name) is None
    )
    for name in malformed:
        _error(
            errors,
            "timestep_name_invalid",
            "timestep 名称必须严格为 timestep_N，且 N 不含前导零",
            role=role,
            task_id=task_id,
            episode=episode,
            path=f"{episode_group.name}/{name}",
        )
    indices = tuple(
        sorted(
            int(match.group(1))
            for name in episode_group.keys()
            if (match := TIMESTEP_PATTERN.fullmatch(name)) is not None
        )
    )
    details = {"role": role, "task_id": task_id, "episode": episode}
    if not indices:
        _error(
            errors,
            "no_timesteps",
            "每个所选 episode 至少必须包含 timestep_0",
            path=episode_group.name,
            **details,
        )
    elif indices != tuple(range(indices[-1] + 1)):
        _error(
            errors,
            "timestep_sequence_not_contiguous",
            "timestep 必须严格连续为 0..N",
            path=episode_group.name,
            expected=list(range(indices[-1] + 1)),
            actual=list(indices),
            **details,
        )
    return indices


def _read_final_completion(
    episode_group: h5py.Group,
    timesteps: tuple[int, ...],
    *,
    role: str,
    task_id: str,
    episode: int,
    errors: list[dict[str, Any]],
) -> bool | None:
    if not timesteps:
        return None
    final_timestep = timesteps[-1]
    details = {
        "role": role,
        "task_id": task_id,
        "episode": episode,
        "timestep": final_timestep,
    }
    timestep_group = _hard_group(
        episode_group,
        f"timestep_{final_timestep}",
        errors=errors,
        code="final_timestep_invalid",
        message="最终 timestep 必须是普通 hard-link group",
        details=details,
    )
    if timestep_group is None:
        return None
    info = _hard_group(
        timestep_group,
        "info",
        errors=errors,
        code="final_info_missing_or_invalid",
        message="最终 timestep/info 必须是普通 hard-link group",
        details=details,
    )
    if info is None:
        return None
    completed = _hard_dataset(
        info,
        "is_completed",
        errors=errors,
        code="final_is_completed_missing_or_invalid",
        message="最终 info/is_completed 必须是普通 hard-link dataset",
        details=details,
    )
    if completed is None:
        return None
    if completed.shape != ():
        _error(
            errors,
            "final_is_completed_not_scalar",
            "最终 info/is_completed 必须是标量",
            path=completed.name,
            actual_shape=list(completed.shape),
            **details,
        )
        return None
    try:
        raw = completed[()]
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise InputReadError(f"无法读取 {role} {task_id} {completed.name}：{exc}") from exc
    if completed.dtype.kind != "b" or not isinstance(raw, (bool, np.bool_)):
        _error(
            errors,
            "final_is_completed_not_strict_bool",
            "最终 info/is_completed 必须是严格 bool 标量",
            path=completed.name,
            actual_dtype=str(completed.dtype),
            **details,
        )
        return None
    value = bool(raw)
    if not value:
        _error(
            errors,
            "trajectory_not_completed",
            "最终 info/is_completed 为 False，轨迹未成功完成",
            path=completed.name,
            **details,
        )
    return value


def _read_joint_action(
    timestep_group: h5py.Group,
    *,
    role: str,
    task_id: str,
    episode: int,
    timestep: int,
    errors: list[dict[str, Any]],
) -> tuple[np.ndarray[Any, Any], np.dtype[Any], tuple[int, ...]] | None:
    details = {
        "role": role,
        "task_id": task_id,
        "episode": episode,
        "timestep": timestep,
    }
    action = _hard_group(
        timestep_group,
        "action",
        errors=errors,
        code="action_group_missing_or_invalid",
        message="timestep/action 必须是普通 hard-link group",
        details=details,
    )
    if action is None:
        return None
    dataset = _hard_dataset(
        action,
        "joint_action",
        errors=errors,
        code="joint_action_missing_or_invalid",
        message="timestep/action/joint_action 必须是普通 hard-link dataset",
        details=details,
    )
    if dataset is None:
        return None
    if dataset.dtype.kind not in "iuf":
        _error(
            errors,
            "joint_action_not_real_numeric",
            "joint_action dtype 必须是实数 numeric dtype",
            path=dataset.name,
            actual_dtype=str(dataset.dtype),
            **details,
        )
        return None
    if dataset.size == 0:
        _error(
            errors,
            "joint_action_empty",
            "joint_action 不能为空",
            path=dataset.name,
            **details,
        )
        return None
    try:
        value = np.asarray(dataset[()])
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise InputReadError(f"无法读取 {role} {task_id} {dataset.name}：{exc}") from exc
    if not bool(np.all(np.isfinite(value))):
        _error(
            errors,
            "joint_action_non_finite",
            "joint_action 包含 NaN 或 infinity",
            path=dataset.name,
            **details,
        )
        return None
    return value, np.dtype(dataset.dtype), tuple(dataset.shape)


def _compare_joint_path(
    reference_timestep: h5py.Group,
    generated_timestep: h5py.Group,
    *,
    task_id: str,
    episode: int,
    timestep: int,
    accumulators: Sequence[JointAccumulator],
    errors: list[dict[str, Any]],
) -> None:
    reference = _read_joint_action(
        reference_timestep,
        role="reference",
        task_id=task_id,
        episode=episode,
        timestep=timestep,
        errors=errors,
    )
    generated = _read_joint_action(
        generated_timestep,
        role="generated",
        task_id=task_id,
        episode=episode,
        timestep=timestep,
        errors=errors,
    )
    if reference is None or generated is None:
        return
    reference_value, reference_dtype, reference_shape = reference
    generated_value, generated_dtype, generated_shape = generated
    path = f"record_dataset_{task_id}.h5:/episode_{episode}/timestep_{timestep}/action/joint_action"
    if reference_dtype != generated_dtype:
        _error(
            errors,
            "joint_action_dtype_mismatch",
            "官方与生成 joint_action dtype 不同",
            task_id=task_id,
            episode=episode,
            timestep=timestep,
            path=path,
            reference_dtype=str(reference_dtype),
            generated_dtype=str(generated_dtype),
        )
        return
    if reference_shape != generated_shape:
        _error(
            errors,
            "joint_action_shape_mismatch",
            "官方与生成 joint_action shape 不同",
            task_id=task_id,
            episode=episode,
            timestep=timestep,
            path=path,
            reference_shape=list(reference_shape),
            generated_shape=list(generated_shape),
        )
        return
    for accumulator in accumulators:
        accumulator.update(reference_value, generated_value, path=path)


def _completion_pair(
    expected: int,
) -> tuple[CompletionCounter, CompletionCounter]:
    return CompletionCounter(expected=expected), CompletionCounter(expected=expected)


def _completion_summary(
    reference: CompletionCounter,
    generated: CompletionCounter,
) -> dict[str, Any]:
    return {
        "definition": "final_timestep/info/is_completed strict boolean",
        "reference": reference.summary(),
        "generated": generated.summary(),
    }


def _inventory(directory: Path) -> list[str]:
    return sorted(
        entry.name
        for entry in os.scandir(directory)
        if entry.is_file(follow_symlinks=False)
        and entry.name.endswith(".h5")
    )


def _add_inventory_errors(
    *,
    role: str,
    actual: Sequence[str],
    selected_tasks: Sequence[str],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = {f"record_dataset_{task}.h5" for task in selected_tasks}
    all_known = {f"record_dataset_{task}.h5" for task in EXPECTED_TASK_IDS}
    actual_set = set(actual)
    missing = sorted(selected - actual_set)
    unknown = sorted(actual_set - all_known)
    unselected = sorted((actual_set & all_known) - selected)
    if missing:
        _error(
            errors,
            "task_file_missing",
            "缺少所选任务 HDF5 文件",
            role=role,
            files=missing,
        )
    if unknown:
        _error(
            errors,
            "unknown_task_file",
            "存在不属于固定 16 任务的 HDF5 文件",
            role=role,
            files=unknown,
        )
    return {
        "files": list(actual),
        "selected_missing": missing,
        "known_unselected": unselected,
        "unknown": unknown,
    }


def compare_joint_actions(
    *,
    reference_dir: str | Path = DEFAULT_REFERENCE_DIR,
    generated_dir: str | Path = DEFAULT_GENERATED_DIR,
    json_report: str | Path = DEFAULT_JSON_REPORT,
    markdown_report: str | Path = DEFAULT_MARKDOWN_REPORT,
    workspace_root: str | Path = ".",
    task_ids: Sequence[str] = EXPECTED_TASK_IDS,
    episodes: Sequence[int] = DEFAULT_EPISODES,
) -> dict[str, Any]:
    """执行只读 joint 审查，写出 JSON/Markdown，并返回同一份摘要。"""

    selected_tasks, selected_episodes = _validate_scope(task_ids, episodes)
    workspace = _safe_workspace(workspace_root)
    reference_root = _safe_input_directory(
        reference_dir,
        workspace=workspace,
        label="reference-dir",
    )
    generated_root = _safe_input_directory(
        generated_dir,
        workspace=workspace,
        label="generated-dir",
    )
    if os.path.samefile(reference_root, generated_root):
        raise UnsafePathError(
            "reference-dir 与 generated-dir 不能指向同一个目录"
        )
    json_path = _safe_output_path(
        json_report,
        workspace=workspace,
        input_roots=(reference_root, generated_root),
        label="json-report",
    )
    markdown_path = _safe_output_path(
        markdown_report,
        workspace=workspace,
        input_roots=(reference_root, generated_root),
        label="markdown-report",
    )
    if json_path == markdown_path:
        raise UnsafePathError("json-report 与 markdown-report 不能是同一个文件")
    is_default_scope = (
        selected_tasks == EXPECTED_TASK_IDS
        and selected_episodes == DEFAULT_EPISODES
    )
    if not is_default_scope:
        default_json_path = _absolute_path(DEFAULT_JSON_REPORT)
        default_markdown_path = _absolute_path(DEFAULT_MARKDOWN_REPORT)
        if json_path == default_json_path or markdown_path == default_markdown_path:
            raise UnsafePathError(
                "非默认 tasks/episodes 必须显式指定不同的 --json-report 和 "
                "--markdown-report，禁止覆盖固定 16×9 报告"
            )

    errors: list[dict[str, Any]] = []
    reference_inventory = _inventory(reference_root)
    generated_inventory = _inventory(generated_root)
    inventory = {
        "reference": _add_inventory_errors(
            role="reference",
            actual=reference_inventory,
            selected_tasks=selected_tasks,
            errors=errors,
        ),
        "generated": _add_inventory_errors(
            role="generated",
            actual=generated_inventory,
            selected_tasks=selected_tasks,
            errors=errors,
        ),
    }

    expected_trajectories = len(selected_tasks) * len(selected_episodes)
    reference_completion, generated_completion = _completion_pair(
        expected_trajectories
    )
    global_joint = JointAccumulator()
    task_reports: list[dict[str, Any]] = []

    for task_id in selected_tasks:
        file_name = f"record_dataset_{task_id}.h5"
        reference_file = reference_root / file_name
        generated_file = generated_root / file_name
        if not reference_file.is_file() or not generated_file.is_file():
            continue
        reference_file = _safe_input_file(
            reference_file,
            root=reference_root,
            label=f"reference {task_id}",
        )
        generated_file = _safe_input_file(
            generated_file,
            root=generated_root,
            label=f"generated {task_id}",
        )
        if os.path.samefile(reference_file, generated_file):
            raise UnsafePathError(
                f"官方与生成 {task_id} HDF5 不能是同一个文件或硬链接"
            )

        task_joint = JointAccumulator()
        task_reference_completion, task_generated_completion = _completion_pair(
            len(selected_episodes)
        )
        episode_reports: list[dict[str, Any]] = []
        try:
            reference_h5 = h5py.File(reference_file, "r")
            try:
                generated_h5 = h5py.File(generated_file, "r")
            except (OSError, RuntimeError, TypeError, ValueError):
                reference_h5.close()
                raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise InputReadError(
                f"无法只读打开 {task_id} HDF5：{type(exc).__name__}: {exc}"
            ) from exc

        try:
            with reference_h5, generated_h5:
                reference_indices = _episode_indices(
                    reference_h5,
                    role="reference",
                    task_id=task_id,
                    selected=selected_episodes,
                    errors=errors,
                )
                generated_indices = _episode_indices(
                    generated_h5,
                    role="generated",
                    task_id=task_id,
                    selected=selected_episodes,
                    errors=errors,
                )
                for episode in selected_episodes:
                    episode_joint = JointAccumulator()
                    episode_report: dict[str, Any] = {
                        "episode": episode,
                        "reference_present": episode in reference_indices,
                        "generated_present": episode in generated_indices,
                        "reference_final_is_completed": None,
                        "generated_final_is_completed": None,
                    }
                    reference_episode = _episode_group(
                        reference_h5,
                        role="reference",
                        task_id=task_id,
                        episode=episode,
                        errors=errors,
                    )
                    generated_episode = _episode_group(
                        generated_h5,
                        role="generated",
                        task_id=task_id,
                        episode=episode,
                        errors=errors,
                    )
                    if reference_episode is None or generated_episode is None:
                        episode_report["joint"] = episode_joint.summary()
                        episode_reports.append(episode_report)
                        continue

                    for field_name in ("seed", "difficulty"):
                        reference_value = _read_setup_value(
                            reference_episode,
                            field_name,
                            role="reference",
                            task_id=task_id,
                            episode=episode,
                            errors=errors,
                        )
                        generated_value = _read_setup_value(
                            generated_episode,
                            field_name,
                            role="generated",
                            task_id=task_id,
                            episode=episode,
                            errors=errors,
                        )
                        episode_report[f"reference_{field_name}"] = (
                            None if reference_value is _MISSING else reference_value
                        )
                        episode_report[f"generated_{field_name}"] = (
                            None if generated_value is _MISSING else generated_value
                        )
                        if (
                            reference_value is not _MISSING
                            and generated_value is not _MISSING
                            and not _strict_scalar_equal(
                                reference_value, generated_value
                            )
                        ):
                            _error(
                                errors,
                                "setup_value_mismatch",
                                f"官方与生成 setup/{field_name} 不一致",
                                task_id=task_id,
                                episode=episode,
                                field=field_name,
                                reference=reference_value,
                                generated=generated_value,
                            )

                    reference_timesteps = _timestep_indices(
                        reference_episode,
                        role="reference",
                        task_id=task_id,
                        episode=episode,
                        errors=errors,
                    )
                    generated_timesteps = _timestep_indices(
                        generated_episode,
                        role="generated",
                        task_id=task_id,
                        episode=episode,
                        errors=errors,
                    )
                    episode_report["reference_timestep_count"] = len(
                        reference_timesteps
                    )
                    episode_report["generated_timestep_count"] = len(
                        generated_timesteps
                    )
                    if reference_timesteps != generated_timesteps:
                        _error(
                            errors,
                            "timestep_inventory_mismatch",
                            "官方与生成 timestep 集合不一致",
                            task_id=task_id,
                            episode=episode,
                            reference=list(reference_timesteps),
                            generated=list(generated_timesteps),
                        )

                    reference_completed = _read_final_completion(
                        reference_episode,
                        reference_timesteps,
                        role="reference",
                        task_id=task_id,
                        episode=episode,
                        errors=errors,
                    )
                    generated_completed = _read_final_completion(
                        generated_episode,
                        generated_timesteps,
                        role="generated",
                        task_id=task_id,
                        episode=episode,
                        errors=errors,
                    )
                    episode_report["reference_final_is_completed"] = (
                        reference_completed
                    )
                    episode_report["generated_final_is_completed"] = (
                        generated_completed
                    )
                    reference_completion.observe(reference_completed)
                    generated_completion.observe(generated_completed)
                    task_reference_completion.observe(reference_completed)
                    task_generated_completion.observe(generated_completed)

                    for timestep in sorted(
                        set(reference_timesteps) & set(generated_timesteps)
                    ):
                        reference_timestep = _hard_group(
                            reference_episode,
                            f"timestep_{timestep}",
                            errors=errors,
                            code="timestep_missing_or_invalid",
                            message="timestep 必须是普通 hard-link group",
                            details={
                                "role": "reference",
                                "task_id": task_id,
                                "episode": episode,
                                "timestep": timestep,
                            },
                        )
                        generated_timestep = _hard_group(
                            generated_episode,
                            f"timestep_{timestep}",
                            errors=errors,
                            code="timestep_missing_or_invalid",
                            message="timestep 必须是普通 hard-link group",
                            details={
                                "role": "generated",
                                "task_id": task_id,
                                "episode": episode,
                                "timestep": timestep,
                            },
                        )
                        if reference_timestep is None or generated_timestep is None:
                            continue
                        _compare_joint_path(
                            reference_timestep,
                            generated_timestep,
                            task_id=task_id,
                            episode=episode,
                            timestep=timestep,
                            accumulators=(episode_joint, task_joint, global_joint),
                            errors=errors,
                        )
                    episode_report["joint"] = episode_joint.summary()
                    episode_reports.append(episode_report)
        except InputReadError:
            raise
        except (OSError, RuntimeError, TypeError, UnicodeError, ValueError) as exc:
            raise InputReadError(
                f"读取 {task_id} HDF5 时失败：{type(exc).__name__}: {exc}"
            ) from exc

        task_reports.append(
            {
                "task_id": task_id,
                "completion": _completion_summary(
                    task_reference_completion,
                    task_generated_completion,
                ),
                "joint": task_joint.summary(),
                "episodes": episode_reports,
            }
        )

    validation_passed = not errors
    joint_summary = global_joint.summary()
    joint_scope_error_codes = {
        "task_file_missing",
        "unknown_task_file",
        "unexpected_top_level_entry",
        "selected_episode_missing",
        "episode_missing_or_invalid",
        "timestep_name_invalid",
        "no_timesteps",
        "timestep_sequence_not_contiguous",
        "timestep_inventory_mismatch",
        "timestep_missing_or_invalid",
        "action_group_missing_or_invalid",
        "joint_action_missing_or_invalid",
        "joint_action_not_real_numeric",
        "joint_action_empty",
        "joint_action_non_finite",
        "joint_action_dtype_mismatch",
        "joint_action_shape_mismatch",
    }
    joint_scope_complete = not any(
        item["code"] in joint_scope_error_codes for item in errors
    )
    joint_strict_equal = bool(
        joint_scope_complete and joint_summary["joint_strict_equal"]
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "comparison_kind": "offline_joint_action_report",
        "candidate_commit": CANDIDATE_COMMIT,
        "reference_revision": REFERENCE_REVISION,
        "dependency_locks": {
            "root_uv_lock_sha256": ROOT_UV_LOCK_SHA256,
            "candidate_uv_lock_sha256": CANDIDATE_UV_LOCK_SHA256,
        },
        "workspace_root": str(workspace),
        "reference_dir": str(reference_root),
        "generated_dir": str(generated_root),
        "reference_open_mode": "read_only",
        "generated_open_mode": "read_only",
        "json_report": str(json_path),
        "markdown_report": str(markdown_path),
        "task_ids": list(selected_tasks),
        "episodes": list(selected_episodes),
        "file_inventory": inventory,
        "counts": {
            "selected_tasks": len(selected_tasks),
            "selected_episodes_per_task": len(selected_episodes),
            "selected_trajectories": expected_trajectories,
            "reference_h5_files": len(reference_inventory),
            "generated_h5_files": len(generated_inventory),
            "compared_joint_paths": joint_summary["joint_path_count"],
            "compared_joint_elements": joint_summary["joint_element_count"],
        },
        "completion": _completion_summary(
            reference_completion,
            generated_completion,
        ),
        "joint": joint_summary,
        "tasks": task_reports,
        "validation_passed": validation_passed,
        "joint_scope_complete": joint_scope_complete,
        "joint_strict_equal": joint_strict_equal,
        "joint_difference_report_only": True,
        "joint_tolerance_policy": None,
        "errors": errors,
        "error_count": len(errors),
        "exit_code": 0 if validation_passed else 1,
    }
    try:
        json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(_render_markdown(summary), encoding="utf-8")
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise UnsafePathError(f"无法写入审查报告：{type(exc).__name__}: {exc}") from exc
    return summary


def _format_number(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.12e}"
    return str(value)


def _completion_cell(summary: dict[str, Any]) -> str:
    return (
        f"{summary['successful_trajectories']}/"
        f"{summary['expected_trajectories']}"
    )


def _render_markdown(summary: dict[str, Any]) -> str:
    completion = summary["completion"]
    joint = summary["joint"]
    counts = summary["counts"]
    lines = [
        (
            "# RoboMME dataset "
            f"{counts['selected_tasks']}×{counts['selected_episodes_per_task']} "
            "离线 joint 对比"
        ),
        "",
        "## 结论",
        "",
        f"- 契约验证：{'通过' if summary['validation_passed'] else '失败'}。",
        (
            "- 官方离线完成率："
            f"{_completion_cell(completion['reference'])}；生成离线完成率："
            f"{_completion_cell(completion['generated'])}。成功仅按每条轨迹最终 "
            "`info/is_completed` 的严格布尔值统计。"
        ),
        (
            "- joint 严格相等："
            f"{'是' if summary['joint_strict_equal'] else '否'}；差异只报告，"
            "未设置或应用任何容差阈值，也不参与契约通过判定。"
        ),
        "- 本报告不声明字节级一致、非 joint 全内容一致、数值容差通过或行为回放一致。",
        "",
        "## 固定输入与范围",
        "",
        f"- 候选 commit：`{summary['candidate_commit']}`",
        f"- 官方 revision：`{summary['reference_revision']}`",
        (
            "- root `uv.lock` SHA-256："
            f"`{summary['dependency_locks']['root_uv_lock_sha256']}`"
        ),
        (
            "- candidate `uv.lock` SHA-256："
            f"`{summary['dependency_locks']['candidate_uv_lock_sha256']}`"
        ),
        f"- 官方目录（只读）：`{summary['reference_dir']}`",
        f"- 生成目录（只读）：`{summary['generated_dir']}`",
        f"- 任务数：{len(summary['task_ids'])}",
        f"- episode：`{', '.join(str(item) for item in summary['episodes'])}`",
        (
            f"- 官方 HDF5 文件数：{counts['reference_h5_files']}；"
            f"生成 HDF5 文件数：{counts['generated_h5_files']}；"
            f"所选轨迹数：{counts['selected_trajectories']}。"
        ),
        "",
        "## 可复现命令",
        "",
        "```bash",
        "uv run --locked scripts/compare_joint_actions.py",
        "```",
        "",
        f"机器 JSON：`{summary['json_report']}`",
        "",
        "## 全局 joint 统计",
        "",
        "所有误差统计均以全部已验证 joint 元素为总体，包含严格相等元素。",
        "",
        "| 指标 | 值 |",
        "| --- | ---: |",
    ]
    metric_rows = (
        ("joint 路径数", "joint_path_count"),
        ("joint 元素数", "joint_element_count"),
        ("严格相等路径数", "joint_strict_equal_path_count"),
        ("差异路径数", "joint_differing_path_count"),
        ("严格相等元素数", "joint_strict_equal_element_count"),
        ("差异元素数", "joint_differing_element_count"),
        ("差异元素比例", "joint_difference_element_ratio"),
        ("最大绝对差", "max_abs_difference"),
        ("平均绝对差", "mean_abs_difference"),
        ("RMSE", "rmse"),
        ("p50 绝对差", "p50_abs_difference"),
        ("p95 绝对差", "p95_abs_difference"),
        ("p99 绝对差", "p99_abs_difference"),
    )
    lines.extend(f"| {label} | {_format_number(joint[key])} |" for label, key in metric_rows)
    worst = joint.get("worst_difference")
    if worst is not None:
        lines.extend(
            [
                "",
                "最差位置："
                f"`{worst['path']}`，元素索引 `{worst['element_index']}`；"
                f"官方值 `{worst['reference_value']}`，生成值 "
                f"`{worst['generated_value']}`，绝对差 "
                f"`{_format_number(worst['absolute_difference'])}`。",
            ]
        )

    lines.extend(
        [
            "",
            "## 逐任务摘要",
            "",
            "| 任务 | 官方完成 | 生成完成 | 路径 | 元素 | 差异元素 | 差异比例 | 最大绝对差 | RMSE | 严格相等 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for task in summary["tasks"]:
        task_completion = task["completion"]
        task_joint = task["joint"]
        lines.append(
            "| {task} | {reference} | {generated} | {paths} | {elements} | "
            "{different} | {ratio} | {maximum} | {rmse} | {strict} |".format(
                task=task["task_id"],
                reference=_completion_cell(task_completion["reference"]),
                generated=_completion_cell(task_completion["generated"]),
                paths=_format_number(task_joint["joint_path_count"]),
                elements=_format_number(task_joint["joint_element_count"]),
                different=_format_number(task_joint["joint_differing_element_count"]),
                ratio=_format_number(task_joint["joint_difference_element_ratio"]),
                maximum=_format_number(task_joint["max_abs_difference"]),
                rmse=_format_number(task_joint["rmse"]),
                strict="是" if task_joint["joint_strict_equal"] else "否",
            )
        )

    lines.extend(
        [
            "",
            "## 逐 episode 摘要",
            "",
            "| 任务 | Episode | 官方完成 | 生成完成 | 路径 | 元素 | 差异元素 | 差异比例 | 最大绝对差 | p95 | 严格相等 |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for task in summary["tasks"]:
        for episode in task["episodes"]:
            episode_joint = episode["joint"]
            lines.append(
                "| {task} | {episode} | {reference} | {generated} | {paths} | "
                "{elements} | {different} | {ratio} | {maximum} | {p95} | "
                "{strict} |".format(
                    task=task["task_id"],
                    episode=episode["episode"],
                    reference=_format_number(
                        episode["reference_final_is_completed"]
                    ),
                    generated=_format_number(
                        episode["generated_final_is_completed"]
                    ),
                    paths=_format_number(episode_joint["joint_path_count"]),
                    elements=_format_number(episode_joint["joint_element_count"]),
                    different=_format_number(
                        episode_joint["joint_differing_element_count"]
                    ),
                    ratio=_format_number(
                        episode_joint["joint_difference_element_ratio"]
                    ),
                    maximum=_format_number(episode_joint["max_abs_difference"]),
                    p95=_format_number(episode_joint["p95_abs_difference"]),
                    strict="是" if episode_joint["joint_strict_equal"] else "否",
                )
            )

    if summary["errors"]:
        lines.extend(
            [
                "",
                "## 验证错误",
                "",
                f"共 {summary['error_count']} 项；完整结构化内容见 JSON。",
                "",
                "| code | task | episode | message |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for error in summary["errors"][:100]:
            lines.append(
                "| {code} | {task} | {episode} | {message} |".format(
                    code=error["code"],
                    task=error.get("task_id", ""),
                    episode=error.get("episode", ""),
                    message=str(error["message"]).replace("|", "\\|"),
                )
            )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "只读验证 RoboMME 16×9 文件/seed/完成状态并报告 joint 差异；"
            "joint 差异不设阈值且不影响退出码 0/1。"
        )
    )
    parser.add_argument(
        "--reference-dir",
        default=DEFAULT_REFERENCE_DIR,
        help=f"官方参考 HDF5 目录，默认 {DEFAULT_REFERENCE_DIR}",
    )
    parser.add_argument(
        "--generated-dir",
        default=DEFAULT_GENERATED_DIR,
        help=f"重新生成 HDF5 目录，默认 {DEFAULT_GENERATED_DIR}",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=EXPECTED_TASK_IDS,
        default=list(EXPECTED_TASK_IDS),
        help="任务列表，默认全部 16 个固定任务",
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        default=["0-8"],
        metavar="N|START-END",
        help="episode 编号或闭区间，可用逗号组合；默认 0-8",
    )
    parser.add_argument(
        "--json-report",
        default=DEFAULT_JSON_REPORT,
        help=f"仓库内机器报告路径，默认 {DEFAULT_JSON_REPORT}",
    )
    parser.add_argument(
        "--markdown-report",
        default=DEFAULT_MARKDOWN_REPORT,
        help=f"仓库内 Markdown 报告路径，默认 {DEFAULT_MARKDOWN_REPORT}",
    )
    parser.add_argument(
        "--workspace-root",
        default=".",
        help="路径边界使用的仓库根目录，默认当前目录",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        episodes = _parse_episode_tokens(args.episodes)
        summary = compare_joint_actions(
            reference_dir=args.reference_dir,
            generated_dir=args.generated_dir,
            json_report=args.json_report,
            markdown_report=args.markdown_report,
            workspace_root=args.workspace_root,
            task_ids=args.tasks,
            episodes=episodes,
        )
    except (UnsafePathError, InputReadError, OSError, ValueError) as exc:
        print(f"输入、路径或 HDF5 不可读取：{exc}", file=sys.stderr)
        return 2

    if not summary["validation_passed"]:
        print(
            "结构、metadata 或完成状态验证失败；"
            f"JSON：{summary['json_report']}；Markdown：{summary['markdown_report']}",
            file=sys.stderr,
        )
        return 1
    print(
        "离线契约验证通过；joint 严格相等="
        f"{summary['joint_strict_equal']}（差异仅报告）；"
        f"JSON：{summary['json_report']}；Markdown：{summary['markdown_report']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
