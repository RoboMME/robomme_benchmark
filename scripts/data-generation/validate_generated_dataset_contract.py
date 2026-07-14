#!/usr/bin/env python3
"""验证生成数据的 episode、元数据和 HDF5 setup 契约。"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


H5_NAME_PATTERN = re.compile(r"^record_dataset_(?P<env>.+)\.h5$")
EPISODE_NAME_PATTERN = re.compile(r"^episode_(?P<episode>0|[1-9][0-9]*)$")
TIMESTEP_NAME_PATTERN = re.compile(r"^timestep_(?P<timestep>0|[1-9][0-9]*)$")
COMPLETED_SIMPLE_SUBGOAL = "All tasks completed"


class UnsafePathError(ValueError):
    """表示无法安全地读取输入或写入报告。"""


def _absolute_path(path: str | Path) -> Path:
    """返回不解析符号链接的绝对词法路径。"""
    return Path(os.path.abspath(os.fspath(path)))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_workspace_root(path: str | Path) -> Path:
    workspace = _absolute_path(path)
    if not workspace.is_dir():
        raise UnsafePathError(f"workspace-root 不是已有目录：{workspace}")
    if workspace.is_symlink():
        raise UnsafePathError(f"workspace-root 不能是符号链接：{workspace}")
    resolved = workspace.resolve(strict=True)
    if resolved != workspace:
        raise UnsafePathError(
            f"workspace-root 路径中包含符号链接：{workspace} -> {resolved}"
        )
    return workspace


def _safe_existing_directory(
    path: str | Path,
    *,
    workspace: Path,
    label: str,
) -> Path:
    directory = _absolute_path(path)
    if not _is_relative_to(directory, workspace):
        raise UnsafePathError(f"{label} 位于仓库外：{directory}")
    if not directory.is_dir():
        raise UnsafePathError(f"{label} 不是已有目录：{directory}")
    if directory.is_symlink():
        raise UnsafePathError(f"{label} 不能是符号链接：{directory}")
    resolved = directory.resolve(strict=True)
    if resolved != directory:
        raise UnsafePathError(
            f"{label} 路径中包含符号链接：{directory} -> {resolved}"
        )
    if not _is_relative_to(resolved, workspace):
        raise UnsafePathError(f"{label} 解析后位于仓库外：{resolved}")
    return directory


def _safe_report_path(path: str | Path, *, workspace: Path) -> Path:
    report = _absolute_path(path)
    if report == workspace or not _is_relative_to(report, workspace):
        raise UnsafePathError(f"report 必须位于仓库内：{report}")

    relative_parent = report.parent.relative_to(workspace)
    current = workspace
    for part in relative_parent.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise UnsafePathError(f"report 路径中包含符号链接：{current}")
        if current.exists() and not current.is_dir():
            raise UnsafePathError(f"report 的父路径不是目录：{current}")

    report.parent.mkdir(parents=True, exist_ok=True)
    if report.parent.resolve(strict=True) != report.parent:
        raise UnsafePathError(f"report 父路径中包含符号链接：{report.parent}")
    if report.is_symlink() or (report.exists() and not report.is_file()):
        raise UnsafePathError(f"report 必须是普通文件或尚不存在：{report}")
    return report


def _reject_report_inside_inputs(
    report: Path,
    *,
    generated_dir: str | Path,
    metadata_root: str | Path,
) -> None:
    """在创建报告父目录前，排除输入树内的任何写入位置。"""
    for label, raw_input in (
        ("generated-dir", generated_dir),
        ("metadata-root", metadata_root),
    ):
        lexical_root = _absolute_path(raw_input)
        candidate_roots = {lexical_root}
        try:
            candidate_roots.add(lexical_root.resolve(strict=True))
        except OSError:
            pass
        for input_root in candidate_roots:
            if report == input_root or _is_relative_to(report, input_root):
                raise UnsafePathError(
                    f"report 不能写入只读输入目录 {label}：{report}"
                )


def _append_error(
    errors: list[dict[str, Any]],
    code: str,
    message: str,
    **details: Any,
) -> None:
    item: dict[str, Any] = {"code": code, "message": message}
    item.update(details)
    errors.append(item)


def _scan_tree(
    root: Path,
    *,
    errors: list[dict[str, Any]],
    reject_temp: bool,
) -> tuple[list[str], list[str]]:
    """不跟随链接地扫描目录，并返回链接与临时残留的相对路径。"""
    symlinks: list[str] = []
    temp_entries: list[str] = []
    try:
        for directory, directory_names, file_names in os.walk(
            root, topdown=True, followlinks=False
        ):
            base = Path(directory)
            kept_directories: list[str] = []
            for name in sorted(directory_names):
                entry = base / name
                relative = entry.relative_to(root).as_posix()
                if entry.is_symlink():
                    symlinks.append(relative)
                else:
                    kept_directories.append(name)
                if reject_temp and name.startswith("temp_"):
                    temp_entries.append(relative)
            directory_names[:] = kept_directories

            for name in sorted(file_names):
                entry = base / name
                relative = entry.relative_to(root).as_posix()
                if entry.is_symlink():
                    symlinks.append(relative)
                if reject_temp and name.startswith("temp_"):
                    temp_entries.append(relative)
    except OSError as exc:
        _append_error(
            errors,
            "tree_scan_failed",
            f"无法扫描目录：{type(exc).__name__}: {exc}",
            path=str(root),
        )

    for relative in symlinks:
        _append_error(
            errors,
            "symbolic_link",
            "输入树中禁止符号链接",
            path=str(root / relative),
        )
    for relative in temp_entries:
        _append_error(
            errors,
            "temporary_entry",
            "生成目录中存在 temp_* 临时残留",
            path=str(root / relative),
        )
    return symlinks, temp_entries


def _load_metadata(
    path: Path,
    *,
    env_id: str,
    role: str,
    errors: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any] | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _append_error(
            errors,
            "metadata_unreadable",
            f"无法读取 {role} metadata：{type(exc).__name__}: {exc}",
            env_id=env_id,
            path=str(path),
        )
        return {}, None

    if not isinstance(payload, dict):
        _append_error(
            errors,
            "metadata_not_object",
            f"{role} metadata 顶层必须是 JSON object",
            env_id=env_id,
            path=str(path),
        )
        return {}, None

    if payload.get("env_id") != env_id:
        _append_error(
            errors,
            "metadata_env_mismatch",
            f"{role} metadata 的 env_id 与文件名不一致",
            env_id=env_id,
            path=str(path),
            expected=env_id,
            actual=payload.get("env_id"),
        )

    records = payload.get("records")
    if not isinstance(records, list):
        _append_error(
            errors,
            "metadata_records_not_list",
            f"{role} metadata 的 records 必须是 list",
            env_id=env_id,
            path=str(path),
        )
        return {}, payload

    record_count = payload.get("record_count")
    if (
        not isinstance(record_count, int)
        or isinstance(record_count, bool)
        or record_count != len(records)
    ):
        _append_error(
            errors,
            "metadata_record_count_mismatch",
            f"{role} metadata 的 record_count 必须等于 records 长度",
            env_id=env_id,
            path=str(path),
            expected=len(records),
            actual=record_count,
        )

    index: dict[int, dict[str, Any]] = {}
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            _append_error(
                errors,
                "metadata_record_not_object",
                f"{role} metadata 的 record 必须是 object",
                env_id=env_id,
                path=str(path),
                position=position,
            )
            continue
        episode = record.get("episode")
        if (
            not isinstance(episode, int)
            or isinstance(episode, bool)
            or episode < 0
        ):
            _append_error(
                errors,
                "metadata_episode_invalid",
                f"{role} metadata 的 episode 必须是非负整数",
                env_id=env_id,
                path=str(path),
                position=position,
                actual=episode,
            )
            continue
        if episode in index:
            _append_error(
                errors,
                "metadata_episode_duplicate",
                f"{role} metadata 中 episode 重复",
                env_id=env_id,
                path=str(path),
                episode=episode,
            )
            continue
        for field in ("seed", "difficulty"):
            if field not in record:
                _append_error(
                    errors,
                    "metadata_field_missing",
                    f"{role} metadata record 缺少 {field}",
                    env_id=env_id,
                    path=str(path),
                    episode=episode,
                    field=field,
                )
        index[episode] = record
    return index, payload


def _exact_equal(left: Any, right: Any) -> bool:
    """JSON 标量必须类型和值都相同，避免 True 与 1 被误判相等。"""
    return type(left) is type(right) and left == right


def _read_h5_scalar(
    setup: h5py.Group,
    field: str,
    *,
    env_id: str,
    episode: int,
    errors: list[dict[str, Any]],
) -> Any | None:
    link = setup.get(field, getlink=True)
    if link is None:
        _append_error(
            errors,
            "h5_setup_field_missing",
            f"HDF5 setup 缺少 {field}",
            env_id=env_id,
            episode=episode,
            path=f"/episode_{episode}/setup/{field}",
        )
        return None
    if not isinstance(link, h5py.HardLink):
        _append_error(
            errors,
            "h5_non_hard_link",
            "HDF5 契约字段不能使用 soft/external link",
            env_id=env_id,
            episode=episode,
            path=f"/episode_{episode}/setup/{field}",
        )
        return None
    dataset = setup.get(field)
    if not isinstance(dataset, h5py.Dataset):
        _append_error(
            errors,
            "h5_setup_field_not_dataset",
            f"HDF5 setup/{field} 必须是 dataset",
            env_id=env_id,
            episode=episode,
            path=f"/episode_{episode}/setup/{field}",
        )
        return None
    if dataset.shape != ():
        _append_error(
            errors,
            "h5_setup_field_not_scalar",
            f"HDF5 setup/{field} 必须是标量",
            env_id=env_id,
            episode=episode,
            path=f"/episode_{episode}/setup/{field}",
            actual_shape=list(dataset.shape),
        )
        return None

    try:
        value = dataset[()]
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
    except (OSError, RuntimeError, TypeError, UnicodeError, ValueError) as exc:
        _append_error(
            errors,
            "h5_setup_field_unreadable",
            f"无法读取 HDF5 setup/{field}：{type(exc).__name__}: {exc}",
            env_id=env_id,
            episode=episode,
            path=f"/episode_{episode}/setup/{field}",
        )
        return None
    return value


def _inspect_episode_trajectory(
    episode_group: h5py.Group,
    *,
    env_id: str,
    episode: int,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    """验证单个 episode 的连续 timestep 与最终成功标志。"""
    episode_path = f"/episode_{episode}"
    malformed_names = sorted(
        name
        for name in episode_group.keys()
        if name.startswith("timestep_")
        and TIMESTEP_NAME_PATTERN.fullmatch(name) is None
    )
    for name in malformed_names:
        _append_error(
            errors,
            "h5_timestep_name_invalid",
            "timestep 名称必须严格为 timestep_N，且 N 不含前导零",
            env_id=env_id,
            episode=episode,
            path=f"{episode_path}/{name}",
        )

    timestep_indices = sorted(
        int(match.group("timestep"))
        for name in episode_group.keys()
        if (match := TIMESTEP_NAME_PATTERN.fullmatch(name)) is not None
    )
    trajectory: dict[str, Any] = {
        "timestep_count": len(timestep_indices),
        "final_timestep": timestep_indices[-1] if timestep_indices else None,
        "final_is_completed": None,
        "final_simple_subgoal": None,
    }
    if not timestep_indices:
        _append_error(
            errors,
            "h5_no_timesteps",
            "每个 HDF5 episode 至少必须包含 timestep_0",
            env_id=env_id,
            episode=episode,
            path=episode_path,
        )
        return trajectory

    expected_indices = list(range(timestep_indices[-1] + 1))
    if timestep_indices != expected_indices:
        _append_error(
            errors,
            "h5_timestep_sequence_not_contiguous",
            "episode 顶层 timestep 必须严格连续为 0..N",
            env_id=env_id,
            episode=episode,
            path=episode_path,
            expected=expected_indices,
            actual=timestep_indices,
        )

    timestep_groups: dict[int, h5py.Group] = {}
    for timestep in timestep_indices:
        name = f"timestep_{timestep}"
        link = episode_group.get(name, getlink=True)
        group = episode_group.get(name)
        if not isinstance(link, h5py.HardLink) or not isinstance(group, h5py.Group):
            _append_error(
                errors,
                "h5_timestep_not_group",
                "每个 timestep 必须是普通 hard-link group",
                env_id=env_id,
                episode=episode,
                timestep=timestep,
                path=f"{episode_path}/{name}",
            )
            continue
        timestep_groups[timestep] = group

    final_timestep = timestep_indices[-1]
    final_group = timestep_groups.get(final_timestep)
    if final_group is None:
        return trajectory
    final_path = f"{episode_path}/timestep_{final_timestep}"

    info_link = final_group.get("info", getlink=True)
    info_group = final_group.get("info")
    if not isinstance(info_link, h5py.HardLink) or not isinstance(
        info_group, h5py.Group
    ):
        _append_error(
            errors,
            "h5_final_info_missing_or_invalid",
            "最终 timestep/info 必须是普通 hard-link group",
            env_id=env_id,
            episode=episode,
            timestep=final_timestep,
            path=f"{final_path}/info",
        )
        return trajectory

    completed_link = info_group.get("is_completed", getlink=True)
    completed_dataset = info_group.get("is_completed")
    completed_path = f"{final_path}/info/is_completed"
    if completed_link is None:
        _append_error(
            errors,
            "h5_final_is_completed_missing",
            "最终 timestep/info 缺少 is_completed",
            env_id=env_id,
            episode=episode,
            timestep=final_timestep,
            path=completed_path,
        )
    elif not isinstance(completed_link, h5py.HardLink) or not isinstance(
        completed_dataset, h5py.Dataset
    ):
        _append_error(
            errors,
            "h5_final_is_completed_not_dataset",
            "最终 is_completed 必须是普通 hard-link dataset",
            env_id=env_id,
            episode=episode,
            timestep=final_timestep,
            path=completed_path,
        )
    elif completed_dataset.shape != ():
        _append_error(
            errors,
            "h5_final_is_completed_not_scalar",
            "最终 is_completed 必须是标量 dataset",
            env_id=env_id,
            episode=episode,
            timestep=final_timestep,
            path=completed_path,
            actual_shape=list(completed_dataset.shape),
        )
    else:
        try:
            raw_completed = completed_dataset[()]
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            _append_error(
                errors,
                "h5_final_is_completed_unreadable",
                f"无法读取最终 is_completed：{type(exc).__name__}: {exc}",
                env_id=env_id,
                episode=episode,
                timestep=final_timestep,
                path=completed_path,
            )
        else:
            if completed_dataset.dtype.kind != "b" or not isinstance(
                raw_completed, (bool, np.bool_)
            ):
                _append_error(
                    errors,
                    "h5_final_is_completed_not_bool",
                    "最终 is_completed 必须是严格 bool 标量",
                    env_id=env_id,
                    episode=episode,
                    timestep=final_timestep,
                    path=completed_path,
                    actual_dtype=str(completed_dataset.dtype),
                )
            else:
                completed = bool(raw_completed)
                trajectory["final_is_completed"] = completed
                if not completed:
                    _append_error(
                        errors,
                        "h5_final_not_completed",
                        "最终 timestep 的 is_completed 必须为 True",
                        env_id=env_id,
                        episode=episode,
                        timestep=final_timestep,
                        path=completed_path,
                        expected=True,
                        actual=False,
                    )

    subgoal_link = info_group.get("simple_subgoal", getlink=True)
    subgoal_dataset = info_group.get("simple_subgoal")
    subgoal_path = f"{final_path}/info/simple_subgoal"
    if subgoal_link is None:
        _append_error(
            errors,
            "h5_final_simple_subgoal_missing",
            "最终 timestep/info 缺少 simple_subgoal",
            env_id=env_id,
            episode=episode,
            timestep=final_timestep,
            path=subgoal_path,
        )
    elif not isinstance(subgoal_link, h5py.HardLink) or not isinstance(
        subgoal_dataset, h5py.Dataset
    ):
        _append_error(
            errors,
            "h5_final_simple_subgoal_not_dataset",
            "最终 simple_subgoal 必须是普通 hard-link dataset",
            env_id=env_id,
            episode=episode,
            timestep=final_timestep,
            path=subgoal_path,
        )
    elif subgoal_dataset.shape != ():
        _append_error(
            errors,
            "h5_final_simple_subgoal_not_scalar",
            "最终 simple_subgoal 必须是标量 dataset",
            env_id=env_id,
            episode=episode,
            timestep=final_timestep,
            path=subgoal_path,
            actual_shape=list(subgoal_dataset.shape),
        )
    else:
        try:
            simple_subgoal = subgoal_dataset[()]
            if hasattr(simple_subgoal, "item"):
                simple_subgoal = simple_subgoal.item()
            if isinstance(simple_subgoal, bytes):
                simple_subgoal = simple_subgoal.decode("utf-8")
        except (OSError, RuntimeError, TypeError, UnicodeError, ValueError) as exc:
            _append_error(
                errors,
                "h5_final_simple_subgoal_unreadable",
                f"无法读取最终 simple_subgoal：{type(exc).__name__}: {exc}",
                env_id=env_id,
                episode=episode,
                timestep=final_timestep,
                path=subgoal_path,
            )
        else:
            trajectory["final_simple_subgoal"] = simple_subgoal
            if simple_subgoal != COMPLETED_SIMPLE_SUBGOAL:
                _append_error(
                    errors,
                    "h5_final_simple_subgoal_mismatch",
                    "最终 simple_subgoal 必须表明全部任务完成",
                    env_id=env_id,
                    episode=episode,
                    timestep=final_timestep,
                    path=subgoal_path,
                    expected=COMPLETED_SIMPLE_SUBGOAL,
                    actual=simple_subgoal,
                )
    return trajectory


def _inspect_h5(
    path: Path,
    *,
    env_id: str,
    errors: list[dict[str, Any]],
) -> tuple[
    list[int],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    episodes: list[int] = []
    setup_values: dict[int, dict[str, Any]] = {}
    trajectory_summaries: dict[int, dict[str, Any]] = {}
    try:
        with h5py.File(path, "r") as data:
            unexpected = [
                name for name in data.keys() if EPISODE_NAME_PATTERN.fullmatch(name) is None
            ]
            for name in sorted(unexpected):
                _append_error(
                    errors,
                    "h5_unexpected_top_level_entry",
                    "HDF5 顶层只允许规范的 episode_N group",
                    env_id=env_id,
                    path=f"/{name}",
                )

            episodes = sorted(
                int(match.group("episode"))
                for name in data.keys()
                if (match := EPISODE_NAME_PATTERN.fullmatch(name)) is not None
            )
            expected_episodes = list(range(len(episodes)))
            if not episodes:
                _append_error(
                    errors,
                    "h5_no_episodes",
                    "HDF5 至少必须包含 episode_0",
                    env_id=env_id,
                    path=str(path),
                )
            elif episodes != expected_episodes:
                _append_error(
                    errors,
                    "h5_episode_sequence_not_contiguous",
                    "HDF5 顶层 episode 必须严格连续为 0..N-1",
                    env_id=env_id,
                    path=str(path),
                    expected=expected_episodes,
                    actual=episodes,
                )

            for episode in episodes:
                episode_name = f"episode_{episode}"
                episode_link = data.get(episode_name, getlink=True)
                episode_group = data.get(episode_name)
                if not isinstance(episode_link, h5py.HardLink) or not isinstance(
                    episode_group, h5py.Group
                ):
                    _append_error(
                        errors,
                        "h5_episode_not_group",
                        "HDF5 episode 必须是普通 group",
                        env_id=env_id,
                        episode=episode,
                        path=f"/{episode_name}",
                    )
                    continue
                setup_link = episode_group.get("setup", getlink=True)
                setup_group = episode_group.get("setup")
                if not isinstance(setup_link, h5py.HardLink) or not isinstance(
                    setup_group, h5py.Group
                ):
                    _append_error(
                        errors,
                        "h5_setup_missing_or_invalid",
                        "HDF5 episode/setup 必须是普通 group",
                        env_id=env_id,
                        episode=episode,
                        path=f"/{episode_name}/setup",
                    )
                else:
                    setup_values[episode] = {
                        field: _read_h5_scalar(
                            setup_group,
                            field,
                            env_id=env_id,
                            episode=episode,
                            errors=errors,
                        )
                        for field in ("seed", "difficulty")
                    }
                trajectory_summaries[episode] = _inspect_episode_trajectory(
                    episode_group,
                    env_id=env_id,
                    episode=episode,
                    errors=errors,
                )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _append_error(
            errors,
            "h5_unreadable",
            f"无法读取 HDF5：{type(exc).__name__}: {exc}",
            env_id=env_id,
            path=str(path),
        )
    return episodes, setup_values, trajectory_summaries


def _compare_records(
    *,
    env_id: str,
    episodes: list[int],
    generated_records: dict[int, dict[str, Any]],
    source_records: dict[int, dict[str, Any]],
    setup_values: dict[int, dict[str, Any]],
    trajectory_summaries: dict[int, dict[str, Any]],
    errors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_set = set(episodes)
    generated_set = set(generated_records)
    if generated_set != expected_set:
        _append_error(
            errors,
            "generated_metadata_episode_set_mismatch",
            "生成 metadata 的 episode 集合必须与 HDF5 完全一致",
            env_id=env_id,
            expected=sorted(expected_set),
            actual=sorted(generated_set),
            missing=sorted(expected_set - generated_set),
            extra=sorted(generated_set - expected_set),
        )

    episode_summaries: list[dict[str, Any]] = []
    for episode in episodes:
        generated = generated_records.get(episode)
        source = source_records.get(episode)
        setup = setup_values.get(episode)
        trajectory = trajectory_summaries.get(
            episode,
            {
                "timestep_count": 0,
                "final_timestep": None,
                "final_is_completed": None,
                "final_simple_subgoal": None,
            },
        )
        summary: dict[str, Any] = {
            "episode": episode,
            "timestep_count": trajectory["timestep_count"],
            "trajectory": trajectory,
        }

        if source is None:
            _append_error(
                errors,
                "source_metadata_episode_missing",
                "源 metadata 缺少生成 episode",
                env_id=env_id,
                episode=episode,
            )
        if generated is not None:
            summary["generated_metadata"] = {
                field: generated.get(field) for field in ("seed", "difficulty")
            }
        if source is not None:
            summary["source_metadata"] = {
                field: source.get(field) for field in ("seed", "difficulty")
            }
        if setup is not None:
            summary["h5_setup"] = setup

        if generated is not None and source is not None:
            for field in ("seed", "difficulty"):
                actual = generated.get(field)
                expected = source.get(field)
                if not _exact_equal(actual, expected):
                    _append_error(
                        errors,
                        "generated_metadata_value_mismatch",
                        f"生成 metadata 的 {field} 与源 metadata 不一致",
                        env_id=env_id,
                        episode=episode,
                        field=field,
                        expected=expected,
                        actual=actual,
                    )

        if setup is not None:
            expected_record = generated if generated is not None else source
            if expected_record is not None:
                for field in ("seed", "difficulty"):
                    actual = setup.get(field)
                    expected = expected_record.get(field)
                    if not _exact_equal(actual, expected):
                        _append_error(
                            errors,
                            "h5_setup_value_mismatch",
                            f"HDF5 setup/{field} 与 metadata 不一致",
                            env_id=env_id,
                            episode=episode,
                            field=field,
                            expected=expected,
                            actual=actual,
                        )
        episode_summaries.append(summary)
    return episode_summaries


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_generated_dataset_contract(
    generated_dir: str | Path,
    metadata_root: str | Path,
    workspace_root: str | Path,
    report: str | Path,
    expected_envs: list[str] | None = None,
    expected_episodes: int | None = None,
) -> dict[str, Any]:
    """验证生成目录且仅向显式报告路径写入 JSON。"""
    workspace = _safe_workspace_root(workspace_root)
    report_candidate = _absolute_path(report)
    _reject_report_inside_inputs(
        report_candidate,
        generated_dir=generated_dir,
        metadata_root=metadata_root,
    )
    report_path = _safe_report_path(report_candidate, workspace=workspace)
    errors: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "schema_version": 1,
        "workspace_root": str(workspace),
        "generated_directory": str(_absolute_path(generated_dir)),
        "metadata_root": str(_absolute_path(metadata_root)),
        "report": str(report_path),
        "expected": {
            "environments": expected_envs,
            "episode_indices": (
                list(range(expected_episodes))
                if expected_episodes is not None
                else None
            ),
        },
        "counts": {
            "h5_files": 0,
            "environments": 0,
            "episodes": 0,
            "timesteps": 0,
            "errors": 0,
        },
        "environments": [],
        "filesystem": {
            "generated_symlinks": [],
            "metadata_symlinks": [],
            "temporary_entries": [],
        },
        "errors": errors,
        "passed": False,
    }

    try:
        generated = _safe_existing_directory(
            generated_dir, workspace=workspace, label="generated-dir"
        )
        metadata = _safe_existing_directory(
            metadata_root, workspace=workspace, label="metadata-root"
        )
        if _is_relative_to(report_path, generated) or _is_relative_to(
            report_path, metadata
        ):
            raise UnsafePathError(
                "report 不能写入只读输入目录 generated-dir 或 metadata-root"
            )
    except UnsafePathError as exc:
        _append_error(errors, "unsafe_input_path", str(exc))
        payload["counts"]["errors"] = len(errors)
        _write_report(report_path, payload)
        return payload

    generated_symlinks, temporary_entries = _scan_tree(
        generated, errors=errors, reject_temp=True
    )
    metadata_symlinks, _ = _scan_tree(
        metadata, errors=errors, reject_temp=False
    )
    payload["filesystem"] = {
        "generated_symlinks": generated_symlinks,
        "metadata_symlinks": metadata_symlinks,
        "temporary_entries": temporary_entries,
    }

    h5_files: list[tuple[str, Path]] = []
    try:
        for entry in sorted(generated.iterdir(), key=lambda item: item.name):
            match = H5_NAME_PATTERN.fullmatch(entry.name)
            if match is not None and entry.is_file() and not entry.is_symlink():
                h5_files.append((match.group("env"), entry))
    except OSError as exc:
        _append_error(
            errors,
            "generated_directory_unreadable",
            f"无法列出 generated-dir：{type(exc).__name__}: {exc}",
            path=str(generated),
        )

    if not h5_files:
        _append_error(
            errors,
            "no_generated_h5",
            "generated-dir 至少必须包含一个 record_dataset_<env>.h5",
            path=str(generated),
        )

    if expected_envs is not None:
        duplicate_expected_envs = sorted(
            {
                env_id
                for env_id in expected_envs
                if expected_envs.count(env_id) > 1
            }
        )
        if duplicate_expected_envs:
            _append_error(
                errors,
                "expected_environment_duplicate",
                "调用方给出的预期环境存在重复值",
                actual=duplicate_expected_envs,
            )
        actual_envs = {env_id for env_id, _ in h5_files}
        expected_env_set = set(expected_envs)
        missing_envs = sorted(expected_env_set - actual_envs)
        extra_envs = sorted(actual_envs - expected_env_set)
        if missing_envs or extra_envs:
            _append_error(
                errors,
                "generated_environment_set_mismatch",
                "生成 HDF5 文件集合与调用方请求不一致",
                expected=sorted(expected_env_set),
                actual=sorted(actual_envs),
                missing=missing_envs,
                extra=extra_envs,
            )

    total_episodes = 0
    total_timesteps = 0
    environment_summaries: list[dict[str, Any]] = []
    for env_id, h5_path in h5_files:
        generated_metadata_path = generated / f"record_dataset_{env_id}_metadata.json"
        source_metadata_path = metadata / f"record_dataset_{env_id}_metadata.json"
        environment_summary: dict[str, Any] = {
            "env_id": env_id,
            "h5_path": str(h5_path),
            "h5_bytes": h5_path.stat().st_size,
            "generated_metadata_path": str(generated_metadata_path),
            "source_metadata_path": str(source_metadata_path),
            "episodes": [],
        }

        episodes, setup_values, trajectory_summaries = _inspect_h5(
            h5_path, env_id=env_id, errors=errors
        )
        if expected_episodes is not None:
            expected_episode_indices = list(range(expected_episodes))
            if episodes != expected_episode_indices:
                _append_error(
                    errors,
                    "generated_episode_set_mismatch",
                    "HDF5 episode 集合与调用方请求不一致",
                    env_id=env_id,
                    expected=expected_episode_indices,
                    actual=episodes,
                )
        total_episodes += len(episodes)
        environment_timesteps = sum(
            int(summary["timestep_count"])
            for summary in trajectory_summaries.values()
        )
        total_timesteps += environment_timesteps
        environment_summary["episode_indices"] = episodes
        environment_summary["episode_count"] = len(episodes)
        environment_summary["timestep_count"] = environment_timesteps

        if not generated_metadata_path.is_file() or generated_metadata_path.is_symlink():
            _append_error(
                errors,
                "generated_metadata_missing",
                "HDF5 缺少同名生成 metadata JSON",
                env_id=env_id,
                path=str(generated_metadata_path),
            )
            generated_records: dict[int, dict[str, Any]] = {}
        else:
            generated_records, _ = _load_metadata(
                generated_metadata_path,
                env_id=env_id,
                role="生成",
                errors=errors,
            )

        if not source_metadata_path.is_file() or source_metadata_path.is_symlink():
            _append_error(
                errors,
                "source_metadata_missing",
                "metadata-root 缺少同名源 metadata JSON",
                env_id=env_id,
                path=str(source_metadata_path),
            )
            source_records: dict[int, dict[str, Any]] = {}
        else:
            source_records, _ = _load_metadata(
                source_metadata_path,
                env_id=env_id,
                role="源",
                errors=errors,
            )

        environment_summary["episodes"] = _compare_records(
            env_id=env_id,
            episodes=episodes,
            generated_records=generated_records,
            source_records=source_records,
            setup_values=setup_values,
            trajectory_summaries=trajectory_summaries,
            errors=errors,
        )
        environment_summaries.append(environment_summary)

    payload["environments"] = environment_summaries
    payload["counts"] = {
        "h5_files": len(h5_files),
        "environments": len(environment_summaries),
        "episodes": total_episodes,
        "timesteps": total_timesteps,
        "errors": len(errors),
    }
    payload["passed"] = not errors
    _write_report(report_path, payload)
    return payload


def _positive_episode_count(raw_value: str) -> int:
    """将 CLI episode 数解析为正整数。"""
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是正整数") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("必须大于等于 1")
    return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只读验证生成 HDF5、生成 metadata 与源 metadata 的契约一致性。"
    )
    parser.add_argument("--generated-dir", required=True, type=Path)
    parser.add_argument("--metadata-root", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--expected-env",
        action="append",
        dest="expected_envs",
        help="调用方预期生成的环境；可重复传入。省略时保持旧的自动发现行为。",
    )
    parser.add_argument(
        "--expected-episodes",
        type=_positive_episode_count,
        help="每个预期环境必须包含 episode_0 到 episode_(N-1)。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = validate_generated_dataset_contract(
            generated_dir=args.generated_dir,
            metadata_root=args.metadata_root,
            workspace_root=args.workspace_root,
            report=args.report,
            expected_envs=args.expected_envs,
            expected_episodes=args.expected_episodes,
        )
    except UnsafePathError as exc:
        print(f"契约验证无法安全启动：{exc}", file=sys.stderr)
        return 2

    if result["passed"]:
        print(
            "生成数据契约验证通过："
            f"{result['counts']['environments']} env，"
            f"{result['counts']['episodes']} episodes；报告：{result['report']}"
        )
        return 0
    print(
        "生成数据契约验证失败："
        f"{result['counts']['errors']} 项错误；报告：{result['report']}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
