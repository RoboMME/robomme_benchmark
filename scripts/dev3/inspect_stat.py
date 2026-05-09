"""Combined dataset inspection: task-goal distribution + xy 渲染编排。

inspect_stat.py 现在只负责：
1. **distribution pipeline**：HDF5 → episode_task_metadata.csv + 每个 env 的
   <env>_distribution.png（与拆分前完全一致，env-specific parsing 留在这里）。
2. **xy pipeline 编排**：4 次薄调用，分别由 4 个 suite 模块负责渲染：
   - counting_inspect (BinFill / PickXtimes / SwingXtimes / StopCube)
   - permanance_inspect (VideoUnmask / VideoUnmaskSwap / ButtonUnmask / ButtonUnmaskSwap)
   - reference_inspect  (PickHighlight / VideoRepick / VideoPlaceButton / VideoPlaceOrder)
   - imitation_inspect  (MoveCube / InsertPeg / PatternLock / RouteStick)
3. **seed-dedup 报告**：HDF5 + 4 个 suite 各自返回的 visible_objects skipped
   汇总打印 ``[Skip-older-seed]`` 块。

xy 相关的所有数据结构、发现/去重、panel 绘制原语统一搬到
``env_specific_extraction/xy_common.py``。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("MPLBACKEND", "Agg")

import h5py
import numpy as np

# env_specific_extraction/ 下的模块通过 sys.path 注入导入（目录名含下划线可以作为
# package 导入，但为保持与 permanence.py 一致的导入方式，继续走 sys.path 注入）。
_SCRIPT_DIR = Path(__file__).resolve().parent
_ENV_SPECIFIC_DIR = _SCRIPT_DIR / "env_specific_extraction"
if str(_ENV_SPECIFIC_DIR) not in sys.path:
    sys.path.insert(0, str(_ENV_SPECIFIC_DIR))

import xy_common  # scripts/dev3/env_specific_extraction/xy_common.py
import counting_inspect as counting_inspect_module
import permanance_inspect as permanance_inspect_module
import reference_inspect as reference_inspect_module
import imitation_inspect as imitation_inspect_module

DEFAULT_BASE_DIR = Path("/data/hongzefu/robomme_benchmark_cvpr2026-heldoutSeed/runs/replay_videos")
DEFAULT_MAX_PER_DIFFICULTY = 1000

# ---------------------------------------------------------------------------
# Distribution-pipeline constants (from dataset-distribution.py)
# ---------------------------------------------------------------------------
MISSING_TASK_GOAL = "<missing task_goal>"
FIXED_TASK_GOAL_LABEL = "fixed_task_goal"

COLOR_ORDER = ["red", "blue", "green"]
COLOR_SIGNATURE_ORDER = [
    "red",
    "blue",
    "green",
    "red+blue",
    "red+green",
    "blue+green",
    "red+blue+green",
    "none",
    "unknown",
]
COLOR_SEQUENCE_ORDER = (
    ["red", "blue", "green"]
    + [f"{first}->{second}" for first, second in permutations(COLOR_ORDER, 2)]
    + ["unknown"]
)
BEFORE_AFTER_ORDER = ["before", "after", "unknown"]
DIFFICULTY_ORDER = ["easy", "medium", "hard"]
DIFFICULTY_SPLIT_ENVS = {
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
}

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}
ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
}

CSV_FIELDS = [
    "env_id",
    "episode",
    "difficulty",
    "canonical_goal",
    "color_signature",
    "put_in_color",
    "put_in_numbers",
    "target_color",
    "first_target_color",
    "second_target_color",
    "third_target_color",
    "target_color_sequence",
    "pickup_count",
    "repeat_count",
    "before_after",
    "target_order",
    "stop_visit",
    "red_count",
    "blue_count",
    "green_count",
    "parse_error",
]

UNMASK_ENVS = {
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    "VideoUnmask",
    "VideoUnmaskSwap",
}
CONSTANT_GOAL_ENVS = {
    "InsertPeg",
    "MoveCube",
    "PatternLock",
    "RouteStick",
}
AGGREGATED_H5_PREFIX = "record_dataset_"
# Reset 阶段 env-specific 元数据已迁移到 reset_segmentation_pngs/<env>_ep*_seed*/
# visible_objects.json 顶层（取代旧的 HDF5 setup/pickhighlight_metadata 与
# setup/videorepick_metadata 数据集）。
VISIBLE_OBJECTS_JSON_FILENAME = "visible_objects.json"
SELECTED_TARGET_JSON_KEY = "selected_target"
VIDEOREPICK_METADATA_KEY = "videorepick_metadata"
SPLIT_H5_PATTERN = re.compile(
    r"^(?P<env_id>.+?)_ep(?P<episode>\d+)(?:_seed(?P<seed>\d+))?$"
)

# XY-pipeline 相关常量（VISIBLE_OBJECT_JSON_FILENAME / 颜色 / 样式表 / env-id 等）
# 现统一定义在 env_specific_extraction/xy_common.py，由 4 个 suite-specific inspect
# 模块共享，inspect_stat.py 不再直接引用它们。


# ---------------------------------------------------------------------------
# Shared HDF5 helpers
# ---------------------------------------------------------------------------


def _episode_sort_key(name: str) -> tuple[int, str]:
    prefix = "episode_"
    if name.startswith(prefix):
        suffix = name[len(prefix):]
        if suffix.isdigit():
            return (0, f"{int(suffix):09d}")
    return (1, name)


def _decode_scalar(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _normalize_task_goal(raw_value: object) -> list[str]:
    if isinstance(raw_value, np.ndarray):
        flattened: Iterable[object] = raw_value.reshape(-1).tolist()
        return [_decode_scalar(item) for item in flattened]
    if isinstance(raw_value, (list, tuple)):
        return [_decode_scalar(item) for item in raw_value]
    return [_decode_scalar(raw_value)]


def _decode_dataset_string(value: object, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        flattened = value.reshape(-1).tolist()
        if not flattened:
            return default
        return _decode_scalar(flattened[0])
    return _decode_scalar(value)


# ---------------------------------------------------------------------------
# Distribution-pipeline parsing
# ---------------------------------------------------------------------------


def _parse_int_word(word: str) -> int | None:
    return NUMBER_WORDS.get(word.lower())


def _parse_ordinal_word(word: str) -> int | None:
    return ORDINAL_WORDS.get(word.lower())


def _ordered_unique_colors(goal: str) -> list[str]:
    found: list[str] = []
    for match in re.finditer(r"\b(red|blue|green)\b", goal):
        color = match.group(1)
        if color not in found:
            found.append(color)
    return found


def _parse_target_color(goal: str) -> str | None:
    match = re.search(r"\b(red|blue|green)\s+cube\b", goal)
    if match:
        return match.group(1)
    colors = _ordered_unique_colors(goal)
    return colors[0] if colors else None


def _parse_repeat_count(goal: str) -> int | None:
    if "again" in goal or "once again" in goal:
        return 1
    if "twice" in goal:
        return 2
    match = re.search(r"\b(one|two|three|four|five|six)\s+times\b", goal)
    if match:
        return _parse_int_word(match.group(1))
    return None


def _parse_target_order(goal: str) -> int | None:
    match = re.search(r"\b(first|second|third|fourth|fifth|sixth)\s+target\b", goal)
    if match:
        return _parse_ordinal_word(match.group(1))
    return None


def _parse_stop_visit(goal: str) -> int | None:
    match = re.search(r"\b(first|second|third|fourth|fifth|sixth)\s+time\b", goal)
    if match:
        return _parse_ordinal_word(match.group(1))
    match = re.search(r"\b(first|second|third|fourth|fifth|sixth)\s+visit\b", goal)
    if match:
        return _parse_ordinal_word(match.group(1))
    return None


def _parse_before_after(goal: str) -> str | None:
    match = re.search(r"\b(before|after)\b(?: the button was pressed| the button)", goal)
    if match:
        return match.group(1)
    return None


def _parse_unmask_fields(goal: str) -> tuple[str | None, int | None, str | None]:
    colors = re.findall(r"\b(red|blue|green)\s+cube\b", goal)
    if not colors:
        colors = _ordered_unique_colors(goal)
    if not colors:
        return (None, None, "failed to parse color sequence")
    unique_colors: list[str] = []
    for color in colors:
        if not unique_colors or color != unique_colors[-1]:
            unique_colors.append(color)
    sequence = "->".join(unique_colors)
    return (sequence, len(unique_colors), None)


def _parse_unmask_target_colors(goal: str) -> tuple[str | None, str | None]:
    colors = re.findall(r"\b(red|blue|green)\s+cube\b", goal)
    if not colors:
        colors = _ordered_unique_colors(goal)

    unique_colors: list[str] = []
    for color in colors:
        if not unique_colors or color != unique_colors[-1]:
            unique_colors.append(color)

    first_color = unique_colors[0] if unique_colors else None
    second_color = unique_colors[1] if len(unique_colors) > 1 else "none"
    return (first_color, second_color)


def _parse_binfill_fields(goal: str) -> tuple[dict[str, int | str], str | None]:
    counts = {f"{color}_count": 0 for color in COLOR_ORDER}
    for count_word, color in re.findall(
        r"\b(one|two|three|four|five|six)\s+(red|blue|green)\s+cube(?:s)?\b",
        goal,
    ):
        count_value = _parse_int_word(count_word)
        if count_value is None:
            return ({}, f"unknown count word '{count_word}'")
        counts[f"{color}_count"] = count_value

    present_colors = [color for color in COLOR_ORDER if counts[f"{color}_count"] > 0]
    put_in_color = len(present_colors)
    put_in_numbers = sum(counts.values())
    if "put the cubes into the bin" in goal and not present_colors:
        color_signature = "none"
    elif present_colors:
        color_signature = "+".join(present_colors)
    else:
        color_signature = "unknown"

    parsed_fields: dict[str, int | str] = {
        "color_signature": color_signature,
        "put_in_color": put_in_color,
        "put_in_numbers": put_in_numbers,
    }
    parsed_fields.update(counts)
    if color_signature == "unknown":
        return (parsed_fields, "failed to parse BinFill color counts")
    return (parsed_fields, None)


def _merge_error_messages(existing_error: object, errors: list[str]) -> str:
    merged: list[str] = []

    existing_text = str(existing_error).strip()
    if existing_text:
        merged.extend(
            part.strip() for part in existing_text.split(";") if part.strip()
        )

    merged.extend(error.strip() for error in errors if error and error.strip())

    unique_errors: list[str] = []
    for error in merged:
        if error not in unique_errors:
            unique_errors.append(error)
    return "; ".join(unique_errors)


def _load_visible_objects_for_episode(
    segmentation_dir: Path | None,
    env_id: str,
    episode_idx: int,
) -> tuple[dict | None, str | None]:
    """按 (env_id, episode_idx) 在 segmentation_dir 下找 visible_objects.json
    并加载。同一 episode 多 seed 时取 max seed（与 inspect_stat 别处去重一致）。
    返回 (payload, error_message)；找不到/损坏返回 (None, error_message)。"""
    if segmentation_dir is None:
        return None, "segmentation_dir not provided"
    if not segmentation_dir.is_dir():
        return None, f"segmentation_dir not found: {segmentation_dir}"

    candidates: list[tuple[int, Path]] = []
    for episode_dir in segmentation_dir.iterdir():
        if not episode_dir.is_dir():
            continue
        match = SPLIT_H5_PATTERN.match(episode_dir.name)
        if not match:
            continue
        if match.group("env_id") != env_id:
            continue
        if int(match.group("episode")) != episode_idx:
            continue
        seed_str = match.group("seed")
        seed_val = int(seed_str) if seed_str is not None else -1
        json_path = episode_dir / VISIBLE_OBJECTS_JSON_FILENAME
        if json_path.is_file():
            candidates.append((seed_val, json_path))

    if not candidates:
        return None, (
            f"no {VISIBLE_OBJECTS_JSON_FILENAME} for {env_id} episode {episode_idx} "
            f"under {segmentation_dir}"
        )

    candidates.sort(key=lambda pair: pair[0])
    _, json_path = candidates[-1]
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"failed to load {json_path}: {type(exc).__name__}: {exc}"

    if not isinstance(payload, dict):
        return None, f"{json_path}: visible_objects payload is not a dict"
    return payload, None


def _parse_videorepick_setup_fields(
    vobjects_payload: dict | None,
) -> tuple[dict[str, int | str], str | None]:
    """从 visible_objects.json 顶层 'videorepick_metadata' 字段读 VideoRepick
    元数据（target_cube_1_color + num_repeats）。"""
    if vobjects_payload is None:
        return ({}, f"missing {VIDEOREPICK_METADATA_KEY}")

    payload = vobjects_payload.get(VIDEOREPICK_METADATA_KEY)
    if not isinstance(payload, dict):
        return ({}, f"missing {VIDEOREPICK_METADATA_KEY}")

    target_color = payload.get("target_cube_1_color")
    if target_color not in COLOR_ORDER:
        return ({}, "invalid videorepick target_cube_1_color")

    repeat_count = payload.get("num_repeats")
    if isinstance(repeat_count, bool) or not isinstance(repeat_count, int):
        return ({}, "invalid videorepick num_repeats type")
    if repeat_count < 1:
        return ({}, "invalid videorepick num_repeats value")

    return (
        {
            "target_color": target_color,
            "repeat_count": repeat_count,
        },
        None,
    )


def _parse_pickhighlight_setup_fields(
    vobjects_payload: dict | None,
) -> tuple[dict[str, int | str], str | None]:
    """从 visible_objects.json 顶层 'selected_target' 字段读 PickHighlight 的
    target_cube_colors（即 reference.enrich_visible_payload 写入的
    selected_target.task_target_colors）。"""
    if vobjects_payload is None:
        return ({}, f"missing {SELECTED_TARGET_JSON_KEY}")

    selected_target = vobjects_payload.get(SELECTED_TARGET_JSON_KEY)
    if not isinstance(selected_target, dict):
        return ({}, f"missing {SELECTED_TARGET_JSON_KEY}")

    target_cube_colors = selected_target.get("task_target_colors")
    if not isinstance(target_cube_colors, list) or not target_cube_colors:
        return ({}, "invalid pickhighlight task_target_colors")

    normalized_colors: list[str] = []
    for color_name in target_cube_colors:
        normalized_color = str(color_name).strip().lower()
        if normalized_color not in COLOR_ORDER:
            return ({}, f"invalid pickhighlight target color {color_name!r}")
        normalized_colors.append(normalized_color)

    max_target_slots = 3
    if len(normalized_colors) > max_target_slots:
        return ({}, f"unexpected pickhighlight target count {len(normalized_colors)}")

    padded_colors = normalized_colors + ["none"] * (max_target_slots - len(normalized_colors))
    return (
        {
            "first_target_color": padded_colors[0],
            "second_target_color": padded_colors[1],
            "third_target_color": padded_colors[2],
        },
        None,
    )


def _default_row(env_id: str, episode_name: str) -> dict[str, object]:
    episode_number = episode_name.removeprefix("episode_")
    return {
        "env_id": env_id,
        "episode": int(episode_number) if episode_number.isdigit() else episode_name,
        "difficulty": "",
        "canonical_goal": "",
        "color_signature": "",
        "put_in_color": "",
        "put_in_numbers": "",
        "target_color": "",
        "first_target_color": "",
        "second_target_color": "",
        "third_target_color": "",
        "target_color_sequence": "",
        "pickup_count": "",
        "repeat_count": "",
        "before_after": "",
        "target_order": "",
        "stop_visit": "",
        "red_count": "",
        "blue_count": "",
        "green_count": "",
        "parse_error": "",
    }


def _parse_semantic_fields(
    row: dict[str, object],
    setup_group: h5py.Group | None = None,
    vobjects_payload: dict | None = None,
) -> None:
    """解析 env-specific 语义字段。

    PickHighlight / VideoRepick 的 reset 元数据来自 visible_objects.json 顶层
    (vobjects_payload)，由调用方按 (env_id, episode_idx) 加载好传入。
    其他 env 的字段仍从 task_goal 字符串里解析，与 setup_group 无关。
    """
    env_id = str(row["env_id"])
    goal = str(row["canonical_goal"]).lower()
    errors: list[str] = []
    existing_error = row.get("parse_error", "")

    if env_id == "VideoRepick":
        if goal == MISSING_TASK_GOAL.lower():
            errors.append("missing task_goal")

        parsed_fields, error = _parse_videorepick_setup_fields(vobjects_payload)
        if parsed_fields:
            row.update(parsed_fields)
        if error:
            errors.append(error)

        row["parse_error"] = _merge_error_messages(existing_error, errors)
        return

    if env_id == "PickHighlight":
        if goal == MISSING_TASK_GOAL.lower():
            errors.append("missing task_goal")

        parsed_fields, error = _parse_pickhighlight_setup_fields(vobjects_payload)
        if parsed_fields:
            row.update(parsed_fields)
        if error:
            errors.append(error)

        row["parse_error"] = _merge_error_messages(existing_error, errors)
        return

    if goal == MISSING_TASK_GOAL.lower():
        row["parse_error"] = _merge_error_messages(existing_error, ["missing task_goal"])
        return

    if env_id == "BinFill":
        parsed_fields, error = _parse_binfill_fields(goal)
        if parsed_fields:
            row.update(parsed_fields)
        if error:
            errors.append(error)
    elif env_id in {"PickXtimes", "SwingXtimes"}:
        row["target_color"] = _parse_target_color(goal) or "unknown"
        repeat_count = _parse_repeat_count(goal)
        if repeat_count is None:
            repeat_count = 1
        row["repeat_count"] = repeat_count
        if row["target_color"] == "unknown":
            errors.append("failed to parse target_color")
    elif env_id == "VideoPlaceButton":
        row["target_color"] = _parse_target_color(goal) or "unknown"
        row["before_after"] = _parse_before_after(goal) or "unknown"
        if row["target_color"] == "unknown":
            errors.append("failed to parse target_color")
        if row["before_after"] == "unknown":
            errors.append("failed to parse before_after")
    elif env_id == "VideoPlaceOrder":
        row["target_color"] = _parse_target_color(goal) or "unknown"
        target_order = _parse_target_order(goal)
        row["target_order"] = target_order if target_order is not None else "unknown"
        if row["target_color"] == "unknown":
            errors.append("failed to parse target_color")
        if row["target_order"] == "unknown":
            errors.append("failed to parse target_order")
    elif env_id == "StopCube":
        stop_visit = _parse_stop_visit(goal)
        row["stop_visit"] = stop_visit if stop_visit is not None else "unknown"
        if row["stop_visit"] == "unknown":
            errors.append("failed to parse stop_visit")
    elif env_id in UNMASK_ENVS:
        target_color = _parse_target_color(goal)
        first_target_color, second_target_color = _parse_unmask_target_colors(goal)
        sequence, pickup_count, error = _parse_unmask_fields(goal)
        row["target_color"] = target_color or "unknown"
        row["first_target_color"] = first_target_color or "unknown"
        row["second_target_color"] = second_target_color or "unknown"
        row["target_color_sequence"] = sequence or "unknown"
        row["pickup_count"] = pickup_count if pickup_count is not None else "unknown"
        if row["target_color"] == "unknown":
            errors.append("failed to parse target_color")
        if row["first_target_color"] == "unknown":
            errors.append("failed to parse first_target_color")
        if row["second_target_color"] == "unknown":
            errors.append("failed to parse second_target_color")
        if error:
            errors.append(error)
    elif env_id in CONSTANT_GOAL_ENVS:
        return

    row["parse_error"] = _merge_error_messages(existing_error, errors)


# ---------------------------------------------------------------------------
# HDF5 discovery + dedup
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SplitH5Entry:
    path: Path
    env_id: str
    episode: int
    seed: int


def _iter_h5_paths(dataset_root: Path) -> list[Path]:
    if dataset_root.is_file():
        return [dataset_root] if dataset_root.suffix in {".h5", ".hdf5"} else []
    if not dataset_root.exists():
        return []

    h5_paths = [
        path
        for pattern in ("*.h5", "*.hdf5")
        for path in dataset_root.glob(pattern)
        if path.is_file()
    ]
    return sorted(h5_paths)


def _classify_h5_paths(
    paths: list[Path],
) -> tuple[list[Path], list[_SplitH5Entry], list[str]]:
    """Split paths into aggregated files, parsed per-episode entries, and warnings."""
    aggregated: list[Path] = []
    split_entries: list[_SplitH5Entry] = []
    warnings: list[str] = []

    for path in paths:
        stem = path.stem
        if stem.startswith(AGGREGATED_H5_PREFIX):
            aggregated.append(path)
            continue
        match = SPLIT_H5_PATTERN.match(stem)
        if match is None:
            warnings.append(f"{path.name}: skipped unsupported HDF5 naming pattern")
            continue
        seed_text = match.group("seed")
        if seed_text is None:
            warnings.append(f"{path.name}: missing seed in filename")
            continue
        split_entries.append(
            _SplitH5Entry(
                path=path,
                env_id=match.group("env_id"),
                episode=int(match.group("episode")),
                seed=int(seed_text),
            )
        )

    return aggregated, split_entries, warnings


def _dedup_split_h5_entries(
    entries: list[_SplitH5Entry],
) -> tuple[list[_SplitH5Entry], list[_SplitH5Entry]]:
    """Keep max-seed entry per (env_id, episode); return (kept, skipped)."""
    grouped: dict[tuple[str, int], list[_SplitH5Entry]] = defaultdict(list)
    for entry in entries:
        grouped[(entry.env_id, entry.episode)].append(entry)

    kept: list[_SplitH5Entry] = []
    skipped: list[_SplitH5Entry] = []
    for (env_id, episode), bucket in grouped.items():
        bucket_sorted = sorted(bucket, key=lambda e: e.seed)
        kept.append(bucket_sorted[-1])
        skipped.extend(bucket_sorted[:-1])
    kept.sort(key=lambda e: (e.env_id, e.episode))
    skipped.sort(key=lambda e: (e.env_id, e.episode, e.seed))
    return kept, skipped


def _peek_episode_difficulty(episode_group: h5py.Group) -> str:
    setup_group = episode_group.get("setup")
    if setup_group is None or "difficulty" not in setup_group:
        return ""
    return _decode_dataset_string(setup_group["difficulty"][()]).strip().lower()


def _resolve_split_episode_group(
    handle: h5py.File, expected_episode_name: str
) -> tuple[str, h5py.Group] | None:
    if expected_episode_name in handle:
        episode_group = handle[expected_episode_name]
        if isinstance(episode_group, h5py.Group):
            return expected_episode_name, episode_group

    episode_group_names = sorted(
        (
            key
            for key, value in handle.items()
            if isinstance(value, h5py.Group) and key.startswith("episode_")
        ),
        key=_episode_sort_key,
    )
    if len(episode_group_names) == 1:
        episode_name = episode_group_names[0]
        return episode_name, handle[episode_name]
    return None


def _append_episode_row(
    rows: list[dict[str, object]],
    warnings: list[str],
    *,
    env_id: str,
    episode_name: str,
    episode_group: h5py.Group,
    segmentation_dir: Path | None = None,
) -> None:
    row = _default_row(env_id, episode_name)
    setup_group = episode_group.get("setup")

    if setup_group is None:
        row["canonical_goal"] = MISSING_TASK_GOAL
        row["parse_error"] = "missing setup group"
        rows.append(row)
        warnings.append(f"{env_id}/{episode_name}: missing setup group")
        return

    if "difficulty" in setup_group:
        row["difficulty"] = _decode_dataset_string(setup_group["difficulty"][()])

    if "task_goal" in setup_group:
        task_goal_values = _normalize_task_goal(setup_group["task_goal"][()])
        row["canonical_goal"] = (
            task_goal_values[0] if task_goal_values else MISSING_TASK_GOAL
        )
    else:
        row["canonical_goal"] = MISSING_TASK_GOAL
        row["parse_error"] = "missing task_goal"
        warnings.append(f"{env_id}/{episode_name}: missing task_goal")

    # PickHighlight / VideoRepick 的 reset 元数据已迁移到 visible_objects.json
    # 顶层（不再写 HDF5 setup group）；按 (env_id, episode_idx) 加载对应 payload。
    vobjects_payload: dict | None = None
    if env_id in {"PickHighlight", "VideoRepick"} and segmentation_dir is not None:
        episode_suffix = episode_name.removeprefix("episode_")
        if episode_suffix.isdigit():
            payload, vload_error = _load_visible_objects_for_episode(
                segmentation_dir, env_id, int(episode_suffix)
            )
            vobjects_payload = payload
            if vload_error and payload is None:
                # 找不到 visible_objects.json：降级让 _parse_*_setup_fields 报
                # missing 错误，走与原行为一致的 warning 路径
                pass

    _parse_semantic_fields(row, setup_group, vobjects_payload)
    if row["parse_error"]:
        warnings.append(f"{env_id}/{episode_name}: {row['parse_error']}")
    rows.append(row)


def _read_episode_rows(
    aggregated_paths: list[Path],
    kept_split_entries: list[_SplitH5Entry],
    *,
    env_filter: Optional[str],
    max_per_difficulty: int | None,
    segmentation_dir: Path | None = None,
) -> tuple[list[dict[str, object]], list[str], dict[tuple[str, int], str]]:
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    difficulty_counts_by_env: dict[str, Counter[str]] = {}
    difficulty_map: dict[tuple[str, int], str] = {}

    def _difficulty_cap_reached(env_id: str, difficulty: str) -> bool:
        if max_per_difficulty is None or max_per_difficulty <= 0:
            return False
        if difficulty not in DIFFICULTY_ORDER:
            return False
        counts = difficulty_counts_by_env.setdefault(env_id, Counter())
        return counts[difficulty] >= max_per_difficulty

    def _bump_difficulty(env_id: str, difficulty: str) -> None:
        if difficulty in DIFFICULTY_ORDER:
            counts = difficulty_counts_by_env.setdefault(env_id, Counter())
            counts[difficulty] += 1

    def _record_difficulty(env_id: str, episode_name: str, difficulty: str) -> None:
        if not difficulty:
            return
        episode_suffix = episode_name.removeprefix("episode_")
        if not episode_suffix.isdigit():
            return
        difficulty_map[(env_id, int(episode_suffix))] = difficulty

    for h5_path in aggregated_paths:
        env_id = h5_path.stem.removeprefix(AGGREGATED_H5_PREFIX)
        if env_filter is not None and env_id != env_filter:
            continue
        try:
            with h5py.File(h5_path, "r") as handle:
                for episode_name in sorted(handle.keys(), key=_episode_sort_key):
                    episode_group = handle[episode_name]
                    if not isinstance(episode_group, h5py.Group):
                        warnings.append(
                            f"{env_id}/{episode_name}: expected HDF5 group"
                        )
                        continue
                    difficulty = _peek_episode_difficulty(episode_group)
                    _record_difficulty(env_id, episode_name, difficulty)
                    if _difficulty_cap_reached(env_id, difficulty):
                        continue
                    _append_episode_row(
                        rows,
                        warnings,
                        env_id=env_id,
                        episode_name=episode_name,
                        episode_group=episode_group,
                        segmentation_dir=segmentation_dir,
                    )
                    _bump_difficulty(env_id, difficulty)
        except Exception as exc:
            warnings.append(f"{h5_path.name}: failed to open ({exc})")

    for entry in kept_split_entries:
        if env_filter is not None and entry.env_id != env_filter:
            continue
        try:
            with h5py.File(entry.path, "r") as handle:
                expected_name = f"episode_{entry.episode}"
                resolved = _resolve_split_episode_group(handle, expected_name)
                if resolved is None:
                    warnings.append(
                        f"{entry.path.name}: could not resolve episode group"
                    )
                    continue
                episode_name, episode_group = resolved
                difficulty = _peek_episode_difficulty(episode_group)
                _record_difficulty(entry.env_id, episode_name, difficulty)
                if _difficulty_cap_reached(entry.env_id, difficulty):
                    continue
                _append_episode_row(
                    rows,
                    warnings,
                    env_id=entry.env_id,
                    episode_name=episode_name,
                    episode_group=episode_group,
                    segmentation_dir=segmentation_dir,
                )
                _bump_difficulty(entry.env_id, difficulty)
        except Exception as exc:
            warnings.append(f"{entry.path.name}: failed to open ({exc})")

    return rows, warnings, difficulty_map


def _write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


# ---------------------------------------------------------------------------
# Distribution-pipeline plotting
# ---------------------------------------------------------------------------


def _plot_order(counter: Counter[object], preferred: list[str] | None = None) -> list[str]:
    labels = [str(label) for label in counter.keys()]
    if preferred is None:
        numeric_labels = [label for label in labels if label.isdigit()]
        if len(numeric_labels) == len(labels):
            return [str(value) for value in sorted(int(label) for label in labels)]
        return sorted(labels)

    ordered = [label for label in preferred if label in labels]
    remaining = sorted(label for label in labels if label not in preferred)
    return ordered + remaining


def _subplot_shape(num_plots: int) -> tuple[int, int]:
    if num_plots <= 1:
        return (1, 1)
    if num_plots == 2:
        return (1, 2)
    if num_plots <= 4:
        return (2, 2)
    return (2, 3)


def _ordinal_label(value: str) -> str:
    suffix = "th"
    if value.endswith("1") and value != "11":
        suffix = "st"
    elif value.endswith("2") and value != "12":
        suffix = "nd"
    elif value.endswith("3") and value != "13":
        suffix = "rd"
    return f"{value}{suffix}"


def _display_label(field: str, raw_label: str) -> str:
    if field == "target_order":
        return _ordinal_label(raw_label)
    return raw_label


def _series_counter(rows: list[dict[str, object]], field: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = row.get(field, "")
        if value == "":
            continue
        counter[str(value)] += 1
    return counter


def _annotate_bars(ax, bars, counts: list[int], total: int) -> None:
    ymax = max(counts, default=0)
    offset = max(ymax * 0.02, 0.5)
    for bar, count in zip(bars, counts):
        percentage = (count / total * 100.0) if total else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{count}\n{percentage:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _axis_upper(max_count: int) -> float:
    return max(max_count * 1.2, max_count + 2, 1)


def _plot_counter(
    ax,
    *,
    field: str,
    title: str,
    counter: Counter[str],
    total: int,
    color: str,
    preferred_order: list[str] | None = None,
    include_all_preferred: bool = False,
    upper_bound: float | None = None,
) -> None:
    if include_all_preferred and preferred_order:
        ordered_labels = list(preferred_order)
    else:
        ordered_labels = _plot_order(counter, preferred_order)
    counts = [counter.get(label, 0) for label in ordered_labels]
    display_labels = [_display_label(field, label) for label in ordered_labels]
    bars = ax.bar(range(len(ordered_labels)), counts, color=color, edgecolor="black")
    ax.set_title(title)
    ax.set_ylabel("Episodes")
    ax.set_xticks(range(len(ordered_labels)))
    ax.set_xticklabels(display_labels, rotation=25, ha="right")
    upper = max(counts, default=0)
    ax.set_ylim(0, upper_bound if upper_bound is not None else _axis_upper(upper))
    _annotate_bars(ax, bars, counts, total)


def _constant_goal_counter(rows: list[dict[str, object]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    if rows:
        counter[FIXED_TASK_GOAL_LABEL] = len(rows)
    return counter


def _figure_specs_for_env(env_id: str) -> list[dict[str, object]]:
    if env_id == "BinFill":
        return [
            {
                "field": "put_in_color",
                "title": "Put-In Color Count",
                "color": "#4C78A8",
                "preferred_order": [str(value) for value in range(1, 4)] + ["unknown"],
            },
            {
                "field": "put_in_numbers",
                "title": "Put-In Cube Count",
                "color": "#54A24B",
                "preferred_order": [str(value) for value in range(1, 7)] + ["unknown"],
            },
        ]
    if env_id in {"PickXtimes", "SwingXtimes"}:
        return [
            {
                "field": "target_color",
                "title": "Target Color",
                "color": "#72B7B2",
                "preferred_order": COLOR_ORDER + ["unknown"],
            },
            {
                "field": "repeat_count",
                "title": "Repeat Count",
                "color": "#F58518",
                "preferred_order": [str(value) for value in range(1, 7)] + ["unknown"],
            },
        ]
    if env_id == "VideoPlaceButton":
        return [
            {
                "field": "target_color",
                "title": "Target Color",
                "color": "#72B7B2",
                "preferred_order": COLOR_ORDER + ["unknown"],
            },
            {
                "field": "before_after",
                "title": "Before / After",
                "color": "#F2CF5B",
                "preferred_order": BEFORE_AFTER_ORDER,
            },
        ]
    if env_id == "VideoPlaceOrder":
        return [
            {
                "field": "target_color",
                "title": "Target Color",
                "color": "#72B7B2",
                "preferred_order": COLOR_ORDER + ["unknown"],
            },
            {
                "field": "target_order",
                "title": "Target Order",
                "color": "#B279A2",
                "preferred_order": [str(value) for value in range(1, 7)] + ["unknown"],
            },
        ]
    if env_id == "VideoRepick":
        return [
            {
                "field": "target_color",
                "title": "Target Color",
                "color": "#72B7B2",
                "preferred_order": COLOR_ORDER + ["unknown"],
            },
            {
                "field": "repeat_count",
                "title": "Repeat Count",
                "color": "#F58518",
                "preferred_order": [str(value) for value in range(1, 7)] + ["unknown"],
            }
        ]
    if env_id == "PickHighlight":
        return [
            {
                "field": "first_target_color",
                "title": "1st Target Color",
                "color": "#72B7B2",
                "preferred_order": COLOR_ORDER + ["none", "unknown"],
            },
            {
                "field": "second_target_color",
                "title": "2nd Target Color",
                "color": "#F58518",
                "preferred_order": COLOR_ORDER + ["none", "unknown"],
            },
            {
                "field": "third_target_color",
                "title": "3rd Target Color",
                "color": "#B279A2",
                "preferred_order": COLOR_ORDER + ["none", "unknown"],
            },
        ]
    if env_id == "StopCube":
        return [
            {
                "field": "stop_visit",
                "title": "Stop Visit",
                "color": "#B279A2",
                "preferred_order": [str(value) for value in range(1, 7)] + ["unknown"],
            }
        ]
    if env_id in UNMASK_ENVS:
        specs: list[dict[str, object]] = []
        # ButtonUnmask / VideoUnmask（非 swap）的 pickup_count panel 信息量低，
        # 用户要求隐藏；ButtonUnmaskSwap / VideoUnmaskSwap 仍保留。
        if env_id not in {"ButtonUnmask", "VideoUnmask"}:
            specs.append({
                "field": "pickup_count",
                "title": "Pickup Count",
                "color": "#F58518",
                "preferred_order": [str(value) for value in range(1, 4)] + ["unknown"],
            })
        specs.append({
            "field": "first_target_color",
            "title": "1st Target Color",
            "color": "#72B7B2",
            "preferred_order": COLOR_ORDER + ["unknown"],
        })
        specs.append({
            "field": "second_target_color",
            "title": "2nd Target Color",
            "color": "#B279A2",
            "preferred_order": COLOR_ORDER + ["none", "unknown"],
        })
        return specs
    return [
        {
            "field": FIXED_TASK_GOAL_LABEL,
            "title": "Fixed Task Goal",
            "color": "#9C755F",
            "preferred_order": [FIXED_TASK_GOAL_LABEL],
        },
        {
            "field": "difficulty",
            "title": "Difficulty",
            "color": "#4C78A8",
            "preferred_order": ["easy", "medium", "hard", "unknown"],
        },
    ]


def _render_env_figure(env_id: str, rows: list[dict[str, object]], output_path: Path, plt) -> None:
    plot_specs = _figure_specs_for_env(env_id)
    if env_id in DIFFICULTY_SPLIT_ENVS:
        rows_by_difficulty = {
            difficulty: [
                row
                for row in rows
                if str(row.get("difficulty", "")).strip().lower() == difficulty
            ]
            for difficulty in DIFFICULTY_ORDER
        }
        upper_bounds_by_field: dict[str, float] = {}
        for spec in plot_specs:
            field = str(spec["field"])
            max_count = 0
            preferred_order = spec.get("preferred_order")
            for difficulty_rows in rows_by_difficulty.values():
                counter = _series_counter(difficulty_rows, field)
                if preferred_order:
                    counts = [counter.get(str(label), 0) for label in preferred_order]
                else:
                    counts = list(counter.values())
                max_count = max(max_count, max(counts, default=0))
            upper_bounds_by_field[field] = _axis_upper(max_count)

        fig, axes = plt.subplots(
            len(DIFFICULTY_ORDER),
            len(plot_specs),
            figsize=(6 * len(plot_specs), 4.2 * len(DIFFICULTY_ORDER)),
            squeeze=False,
            sharey="col",
        )
        total = len(rows)

        for row_index, difficulty in enumerate(DIFFICULTY_ORDER):
            difficulty_rows = rows_by_difficulty[difficulty]
            difficulty_total = len(difficulty_rows)

            for col_index, spec in enumerate(plot_specs):
                ax = axes[row_index, col_index]
                field = str(spec["field"])
                counter = _series_counter(difficulty_rows, field)
                title = str(spec["title"]) if row_index == 0 else ""
                _plot_counter(
                    ax,
                    field=field,
                    title=title,
                    counter=counter,
                    total=difficulty_total,
                    color=str(spec["color"]),
                    preferred_order=spec.get("preferred_order"),
                    include_all_preferred=True,
                    upper_bound=upper_bounds_by_field[field],
                )

                if col_index == 0:
                    ax.set_ylabel(f"{difficulty.title()} (n={difficulty_total})\nEpisodes")
                else:
                    ax.set_ylabel("")

        fig.suptitle(f"{env_id} task distribution by difficulty (n={total})", fontsize=16, y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        return

    num_plots = len(plot_specs)
    nrows, ncols = _subplot_shape(num_plots)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes_array = np.atleast_1d(axes).reshape(-1)
    total = len(rows)

    for ax, spec in zip(axes_array, plot_specs):
        field = str(spec["field"])
        if field == FIXED_TASK_GOAL_LABEL:
            counter = _constant_goal_counter(rows)
        else:
            counter = _series_counter(rows, field)
        _plot_counter(
            ax,
            field=field,
            title=str(spec["title"]),
            counter=counter,
            total=total,
            color=str(spec["color"]),
            preferred_order=spec.get("preferred_order"),
        )

    for ax in axes_array[num_plots:]:
        ax.axis("off")

    fig.suptitle(f"{env_id} task distribution (n={total})", fontsize=16, y=0.98)
    if env_id in CONSTANT_GOAL_ENVS:
        canonical_goal = str(rows[0]["canonical_goal"]) if rows else ""
        subtitle = (
            "No task-goal semantic split in setup metadata. "
            f"Canonical goal: {canonical_goal}"
        )
        fig.text(
            0.5,
            0.93,
            textwrap.fill(subtitle, width=110),
            ha="center",
            va="top",
            fontsize=10,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")



# ---------------------------------------------------------------------------
# CLI + orchestration
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a generated dataset: parse HDF5 task goals into per-env "
            "distribution plots and aggregate visible_objects.json into per-env "
            "XY scatter collages, all written to a single output folder. "
            "For each (env_id, episode), only the artifact with the largest seed "
            "is consumed; older seeds are explicitly listed and skipped."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base directory under runs/replay_videos; hdf5-dir, segmentation-dir, and output-dir are derived from it.",
    )
    parser.add_argument(
        "--hdf5-dir",
        type=Path,
        default=None,
        help="Directory containing per-episode <env>_ep<n>_seed<s>.h5 (or record_dataset_*.h5) files. Defaults to <base-dir>/hdf5_files.",
    )
    parser.add_argument(
        "--segmentation-dir",
        type=Path,
        default=None,
        help="Directory containing reset segmentation episode folders with visible_objects.json. Defaults to <base-dir>/reset_segmentation_pngs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Single shared output directory for CSV + all PNGs. Defaults to <base-dir>/inspect-stat.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Optional env_id filter applied to both pipelines.",
    )
    return parser


def _print_skipped_seed_block(
    skipped_h5: list[_SplitH5Entry],
    skipped_json: list,
) -> None:
    print("=" * 72)
    print("[Skip-older-seed] Episodes with multiple seeds — only max-seed kept")
    print("=" * 72)
    if not skipped_h5 and not skipped_json:
        print("No older-seed duplicates found.")
        print()
        return

    if skipped_h5:
        print(f"  HDF5 ({len(skipped_h5)} dropped):")
        for entry in skipped_h5:
            print(
                f"    - env={entry.env_id} ep={entry.episode} "
                f"seed={entry.seed} path={entry.path}"
            )
    else:
        print("  HDF5: none")

    if skipped_json:
        print(f"  visible_objects.json ({len(skipped_json)} dropped):")
        for entry in skipped_json:
            print(
                f"    - env={entry.env_id} ep={entry.episode} "
                f"seed={entry.seed} path={entry.path}"
            )
    else:
        print("  visible_objects.json: none")
    print()


def _run_distribution_pipeline(
    hdf5_dir: Path,
    output_dir: Path,
    env_filter: Optional[str],
    segmentation_dir: Path | None = None,
) -> tuple[
    list[_SplitH5Entry],
    list[_SplitH5Entry],
    list[Path],
    dict[tuple[str, int], str],
]:
    h5_paths = _iter_h5_paths(hdf5_dir)
    aggregated, split_entries, classify_warnings = _classify_h5_paths(h5_paths)
    kept_split, skipped_split = _dedup_split_h5_entries(split_entries)

    if env_filter is not None:
        aggregated = [
            path
            for path in aggregated
            if path.stem.removeprefix(AGGREGATED_H5_PREFIX) == env_filter
        ]
        kept_for_processing = [entry for entry in kept_split if entry.env_id == env_filter]
    else:
        kept_for_processing = kept_split

    rows, read_warnings, difficulty_map = _read_episode_rows(
        aggregated,
        kept_for_processing,
        env_filter=env_filter,
        max_per_difficulty=DEFAULT_MAX_PER_DIFFICULTY,
        segmentation_dir=segmentation_dir,
    )

    csv_path = output_dir / "episode_task_metadata.csv"
    _write_csv(rows, csv_path)

    plt = xy_common._get_pyplot(show=False)
    rows_by_env: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        rows_by_env.setdefault(str(row["env_id"]), []).append(row)

    figure_paths: list[Path] = []
    for env_id in sorted(rows_by_env):
        figure_path = output_dir / f"{env_id}_distribution.png"
        _render_env_figure(env_id, rows_by_env[env_id], figure_path, plt)
        figure_paths.append(figure_path)

    plt.close("all")

    print("=" * 72)
    print("[Distribution] Task-goal distribution summary")
    print("=" * 72)
    print(f"  HDF5 input:   {hdf5_dir}")
    print(f"  CSV:          {csv_path}")
    for env_id in sorted(rows_by_env):
        env_rows = rows_by_env[env_id]
        parse_issues = sum(1 for row in env_rows if row["parse_error"])
        figure_path = output_dir / f"{env_id}_distribution.png"
        print(
            f"  {env_id}: episodes={len(env_rows)} "
            f"parse_issues={parse_issues} figure={figure_path}"
        )

    all_warnings = classify_warnings + read_warnings
    if all_warnings:
        print("  Warnings:")
        for warning in all_warnings:
            print(f"    - {warning}")
    print()

    return kept_split, skipped_split, figure_paths, difficulty_map


def main() -> None:
    args = _build_arg_parser().parse_args()
    base_dir = args.base_dir.resolve()
    hdf5_dir = (args.hdf5_dir or base_dir / "hdf5_files").resolve()
    segmentation_dir = (args.segmentation_dir or base_dir / "reset_segmentation_pngs").resolve()
    inspect_dir = (args.output_dir or base_dir / "inspect-stat").resolve()
    task_goal_dir = inspect_dir / "task-goal"
    xy_dir = inspect_dir / "xy"
    task_goal_dir.mkdir(parents=True, exist_ok=True)
    xy_dir.mkdir(parents=True, exist_ok=True)

    print(f"Task-goal dir:  {task_goal_dir}")
    print(f"XY dir:         {xy_dir}")
    print(f"HDF5 dir:       {hdf5_dir}")
    print(f"Segmentation:   {segmentation_dir}")
    if args.env:
        print(f"Env filter:     {args.env}")
    print()

    # Pipeline 1: pre-discover + dedup HDF5, then run distribution
    # PickHighlight / VideoRepick 的 reset 元数据已迁移到 visible_objects.json，
    # 把 segmentation_dir 传下去让 distribution pipeline 也能从中读取。
    kept_h5, skipped_h5, _, difficulty_map = _run_distribution_pipeline(
        hdf5_dir,
        task_goal_dir,
        args.env,
        segmentation_dir=segmentation_dir,
    )

    # Pipeline 2: xy 编排 — 4 个 suite 各自一次薄调用，渲染 visible_objects 散点图
    kept_count, skipped_count = counting_inspect_module.visualize(
        segmentation_dir=segmentation_dir,
        output_dir=xy_dir,
        env_id=args.env,
        difficulty_by_env_episode=difficulty_map,
    )

    kept_perm, skipped_perm = permanance_inspect_module.visualize(
        segmentation_dir=segmentation_dir,
        output_dir=inspect_dir / "permanance_inspect",
        env_id=args.env,
    )

    kept_ref, skipped_ref = reference_inspect_module.visualize(
        segmentation_dir=segmentation_dir,
        output_dir=xy_dir,
        env_id=args.env,
        difficulty_by_env_episode=difficulty_map,
    )

    kept_imit, skipped_imit = imitation_inspect_module.visualize(
        segmentation_dir=segmentation_dir,
        output_dir=xy_dir,
        env_id=args.env,
        difficulty_by_env_episode=difficulty_map,
    )

    # 4 个 suite 的 kept / skipped 是按 (env, episode) 分区的视图；env 跨 suite
    # 不重叠，因此简单拼接就等价于一次全局 dedup 的结果。
    kept_json = kept_count + kept_perm + kept_ref + kept_imit
    skipped_json = skipped_count + skipped_perm + skipped_ref + skipped_imit

    # Report seed-dedup results (consolidated for both pipelines)
    _print_skipped_seed_block(skipped_h5, skipped_json)

    print(
        f"[Done] kept_hdf5={len(kept_h5)} skipped_hdf5={len(skipped_h5)} "
        f"kept_json={len(kept_json)} skipped_json={len(skipped_json)} "
        f"kept_counting={len(kept_count)} skipped_counting={len(skipped_count)} "
        f"kept_permanence={len(kept_perm)} skipped_permanence={len(skipped_perm)} "
        f"kept_reference={len(kept_ref)} skipped_reference={len(skipped_ref)} "
        f"kept_imitation={len(kept_imit)} skipped_imitation={len(skipped_imit)}"
    )


if __name__ == "__main__":
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    try:
        main()
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(0)
