"""Combined dataset inspection: task-goal distribution + visible-object XY scatter.

Replaces ``dataset-distribution.py`` + ``aggregate-visible-object-xy.py`` with a
single entry point that:

1. Discovers per-episode artifacts in two input dirs (HDF5 + visible_objects JSON).
2. For each ``(env_id, episode)`` keeps only the entry with the **largest seed**;
   every dropped artifact is printed in a ``[Skip-older-seed]`` block.
3. Runs both visualizations and writes all output (CSV + per-env PNGs) flat into
   one shared output directory.
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

DEFAULT_HDF5_DIR = Path("runs/replay_videos/hdf5_files")
DEFAULT_SEGMENTATION_DIR = Path("runs/replay_videos/reset_segmentation_pngs")
DEFAULT_OUTPUT_DIR = Path("./tmp/inspect-stat")
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
PICKHIGHLIGHT_METADATA_FIELD = "pickhighlight_metadata"
VIDEOREPICK_METADATA_FIELD = "videorepick_metadata"
SPLIT_H5_PATTERN = re.compile(
    r"^(?P<env_id>.+?)_ep(?P<episode>\d+)(?:_seed(?P<seed>\d+))?$"
)

# ---------------------------------------------------------------------------
# XY-pipeline constants (from aggregate-visible-object-xy.py)
# ---------------------------------------------------------------------------
VISIBLE_OBJECT_JSON_FILENAME = "visible_objects.json"
XY_LIMIT = 0.3
AGGREGATE_DPI = 300

XY_DEFAULT_PNG_SUFFIX = "_xy.png"
VIDEOREPICK_EASY_MEDIUM_SUFFIX = "_xy_easy_medium.png"
VIDEOREPICK_HARD_SUFFIX = "_xy_hard.png"
BINFILL_ENV_ID = "BinFill"
BINFILL_BOARD_WITH_HOLE_NAME = "board_with_hole"
VIDEOREPICK_ENV_ID = "VideoRepick"
INSERTPEG_ENV_ID = "InsertPeg"
MOVECUBE_ENV_ID = "MoveCube"
ROUTESTICK_ENV_ID = "RouteStick"
INSERTPEG_BOX_WITH_HOLE_NAME = "box_with_hole"
MOVECUBE_GOAL_SITE_NAME = "goal_site"
VIDEOREPICK_DIFFICULTY_CYCLE = ("easy", "easy", "medium", "hard")
ENV_METADATA_ROOT = Path("src/robomme/env_metadata")
BIN_PANEL_ENVS = {
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmask",
    "ButtonUnmaskSwap",
}
BIN_NAME_PATTERN = re.compile(r"^bin_(\d+)$")

ALL_CATEGORY_COLORS = {
    "cube": "#2ca02c",
    "button": "#9467bd",
    "peg": "#4e79a7",
    "bin": "#f28e2b",
    "goal_site": "#59a14f",
    "box_with_hole": "#e15759",
    "target": "#e15759",
    "other": "#7f7f7f",
}
ALL_OBJECT_PANEL_ORDER = (
    "cube",
    "button",
    "peg",
    "bin",
    "goal_site",
    "box_with_hole",
    "target",
    "other",
)
CUBE_COLOR_MAP = {
    "red": "#d62728",
    "green": "#2ca02c",
    "blue": "#1f77b4",
    "unknown": "#7f7f7f",
}
BUTTON_STYLE_MAP = {
    "button_base": {"color": "#9467bd", "marker": "s", "label": "button_base"},
    "button_cap": {"color": "#5b2a86", "marker": "X", "label": "button_cap"},
    "button_other": {"color": "#c084fc", "marker": "o", "label": "button_other"},
}
PEG_PART_STYLE_MAP = {
    "peg_head": {"color": "#4e79a7", "marker": "o", "label": "peg_head"},
    "peg_tail": {"color": "#f28e2b", "marker": "X", "label": "peg_tail"},
}
TARGET_STYLE = {"color": "#e15759", "marker": "^", "label": "target"}
GOAL_SITE_STYLE = {"color": "#59a14f", "marker": "^", "label": MOVECUBE_GOAL_SITE_NAME}
BOX_WITH_HOLE_STYLE = {
    "color": "#e15759",
    "marker": "D",
    "label": INSERTPEG_BOX_WITH_HOLE_NAME,
}
BINFILL_BOARD_STYLE = {
    "color": TARGET_STYLE["color"],
    "marker": TARGET_STYLE["marker"],
    "label": BINFILL_BOARD_WITH_HOLE_NAME,
}
BIN_COLOR_CYCLE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


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


def _parse_videorepick_setup_fields(
    setup_group: h5py.Group | None,
) -> tuple[dict[str, int | str], str | None]:
    if setup_group is None or VIDEOREPICK_METADATA_FIELD not in setup_group:
        return ({}, f"missing {VIDEOREPICK_METADATA_FIELD}")

    payload_raw = _decode_dataset_string(
        setup_group[VIDEOREPICK_METADATA_FIELD][()],
        default="",
    ).strip()
    if not payload_raw:
        return ({}, f"missing {VIDEOREPICK_METADATA_FIELD}")

    try:
        payload = json.loads(payload_raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return ({}, f"invalid {VIDEOREPICK_METADATA_FIELD} json ({exc})")

    if not isinstance(payload, dict):
        return ({}, f"invalid {VIDEOREPICK_METADATA_FIELD} payload type")

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
    setup_group: h5py.Group | None,
) -> tuple[dict[str, int | str], str | None]:
    if setup_group is None or PICKHIGHLIGHT_METADATA_FIELD not in setup_group:
        return ({}, f"missing {PICKHIGHLIGHT_METADATA_FIELD}")

    payload_raw = _decode_dataset_string(
        setup_group[PICKHIGHLIGHT_METADATA_FIELD][()],
        default="",
    ).strip()
    if not payload_raw:
        return ({}, f"missing {PICKHIGHLIGHT_METADATA_FIELD}")

    try:
        payload = json.loads(payload_raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return ({}, f"invalid {PICKHIGHLIGHT_METADATA_FIELD} json ({exc})")

    if not isinstance(payload, dict):
        return ({}, f"invalid {PICKHIGHLIGHT_METADATA_FIELD} payload type")

    target_cube_colors = payload.get("target_cube_colors")
    if not isinstance(target_cube_colors, list) or not target_cube_colors:
        return ({}, "invalid pickhighlight target_cube_colors")

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
) -> None:
    env_id = str(row["env_id"])
    goal = str(row["canonical_goal"]).lower()
    errors: list[str] = []
    existing_error = row.get("parse_error", "")

    if env_id == "VideoRepick":
        if goal == MISSING_TASK_GOAL.lower():
            errors.append("missing task_goal")

        parsed_fields, error = _parse_videorepick_setup_fields(setup_group)
        if parsed_fields:
            row.update(parsed_fields)
        if error:
            errors.append(error)

        row["parse_error"] = _merge_error_messages(existing_error, errors)
        return

    if env_id == "PickHighlight":
        if goal == MISSING_TASK_GOAL.lower():
            errors.append("missing task_goal")

        parsed_fields, error = _parse_pickhighlight_setup_fields(setup_group)
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

    _parse_semantic_fields(row, setup_group)
    if row["parse_error"]:
        warnings.append(f"{env_id}/{episode_name}: {row['parse_error']}")
    rows.append(row)


def _read_episode_rows(
    aggregated_paths: list[Path],
    kept_split_entries: list[_SplitH5Entry],
    *,
    env_filter: Optional[str],
    max_per_difficulty: int | None,
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    difficulty_counts_by_env: dict[str, Counter[str]] = {}

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
                    if _difficulty_cap_reached(env_id, difficulty):
                        continue
                    _append_episode_row(
                        rows,
                        warnings,
                        env_id=env_id,
                        episode_name=episode_name,
                        episode_group=episode_group,
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
                if _difficulty_cap_reached(entry.env_id, difficulty):
                    continue
                _append_episode_row(
                    rows,
                    warnings,
                    env_id=entry.env_id,
                    episode_name=episode_name,
                    episode_group=episode_group,
                )
                _bump_difficulty(entry.env_id, difficulty)
        except Exception as exc:
            warnings.append(f"{entry.path.name}: failed to open ({exc})")

    return rows, warnings


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
        return [
            {
                "field": "pickup_count",
                "title": "Pickup Count",
                "color": "#F58518",
                "preferred_order": [str(value) for value in range(1, 4)] + ["unknown"],
            },
            {
                "field": "first_target_color",
                "title": "1st Target Color",
                "color": "#72B7B2",
                "preferred_order": COLOR_ORDER + ["unknown"],
            },
            {
                "field": "second_target_color",
                "title": "2nd Target Color",
                "color": "#B279A2",
                "preferred_order": COLOR_ORDER + ["none", "unknown"],
            },
        ]
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
# XY-pipeline data structures + parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisibleObjectPoint:
    env_id: str
    episode: int
    seed: int
    name: str
    world_x: float
    world_y: float
    world_z: float
    semantic: str
    cube_color: str
    button_kind: str
    difficulty: Optional[str] = None
    bin_index: Optional[int] = None
    peg_part: Optional[str] = None


@dataclass(frozen=True)
class _VisibleObjectsFile:
    path: Path
    env_id: str
    episode: int
    seed: int
    objects: list[dict]


def _safe_float_triplet(value: object) -> Optional[tuple[float, float, float]]:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return float(arr[0]), float(arr[1]), float(arr[2])


def _is_binfill_board_with_hole(env_id: str, name: str) -> bool:
    return env_id == BINFILL_ENV_ID and name == BINFILL_BOARD_WITH_HOLE_NAME


def _bin_index(name: str) -> Optional[int]:
    match = BIN_NAME_PATTERN.fullmatch(name)
    if match is None:
        return None
    return int(match.group(1))


def _peg_part(name: str) -> Optional[str]:
    if name in PEG_PART_STYLE_MAP:
        return name
    return None


def _is_insertpeg_box_with_hole(env_id: str, name: str) -> bool:
    return env_id == INSERTPEG_ENV_ID and name == INSERTPEG_BOX_WITH_HOLE_NAME


def _is_movecube_goal_site(env_id: str, name: str) -> bool:
    return env_id == MOVECUBE_ENV_ID and name == MOVECUBE_GOAL_SITE_NAME


def _semantic_category(env_id: str, name: str) -> str:
    if _is_binfill_board_with_hole(env_id, name):
        return "target"
    if env_id in {INSERTPEG_ENV_ID, MOVECUBE_ENV_ID} and _peg_part(name) is not None:
        return "peg"
    if _is_insertpeg_box_with_hole(env_id, name):
        return "box_with_hole"
    if _is_movecube_goal_site(env_id, name):
        return "goal_site"
    if env_id in BIN_PANEL_ENVS and _bin_index(name) is not None:
        return "bin"
    lower = name.lower()
    if "cube" in lower:
        return "cube"
    if "button" in lower:
        return "button"
    if "target" in lower:
        return "target"
    return "other"


def _cube_color(name: str) -> str:
    lower = name.lower()
    for color in ("red", "green", "blue"):
        if color in lower:
            return color
    return "unknown"


def _button_kind(name: str) -> str:
    lower = name.lower()
    if "button_base" in lower:
        return "button_base"
    if "button_cap" in lower:
        return "button_cap"
    return "button_other"


def _discover_visible_object_files(input_dir: Path) -> list[_VisibleObjectsFile]:
    files: list[_VisibleObjectsFile] = []
    for json_path in sorted(input_dir.rglob(VISIBLE_OBJECT_JSON_FILENAME)):
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            print(f"[Warn] Skip invalid JSON file {json_path}: {type(exc).__name__}: {exc}")
            continue

        env_id = payload.get("env_id")
        episode = payload.get("episode")
        seed = payload.get("seed")
        objects = payload.get("objects")

        if not isinstance(env_id, str) or not isinstance(episode, int) or not isinstance(seed, int):
            print(f"[Warn] Skip malformed metadata file {json_path}: missing env_id/episode/seed")
            continue
        if not isinstance(objects, list):
            print(f"[Warn] Skip malformed objects list in {json_path}")
            continue

        files.append(
            _VisibleObjectsFile(
                path=json_path,
                env_id=env_id,
                episode=episode,
                seed=seed,
                objects=objects,
            )
        )
    return files


def _dedup_visible_object_files(
    entries: list[_VisibleObjectsFile],
) -> tuple[list[_VisibleObjectsFile], list[_VisibleObjectsFile]]:
    grouped: dict[tuple[str, int], list[_VisibleObjectsFile]] = defaultdict(list)
    for entry in entries:
        grouped[(entry.env_id, entry.episode)].append(entry)

    kept: list[_VisibleObjectsFile] = []
    skipped: list[_VisibleObjectsFile] = []
    for bucket in grouped.values():
        bucket_sorted = sorted(bucket, key=lambda e: e.seed)
        kept.append(bucket_sorted[-1])
        skipped.extend(bucket_sorted[:-1])
    kept.sort(key=lambda e: (e.env_id, e.episode))
    skipped.sort(key=lambda e: (e.env_id, e.episode, e.seed))
    return kept, skipped


def _load_episode_difficulty_map() -> dict[str, dict[int, str]]:
    difficulty_by_env_episode: dict[str, dict[int, str]] = defaultdict(dict)
    if not ENV_METADATA_ROOT.is_dir():
        return {}

    for metadata_path in sorted(ENV_METADATA_ROOT.glob("*/record_dataset_*_metadata.json")):
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            print(
                f"[Warn] Skip invalid env metadata {metadata_path}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        env_id = payload.get("env_id")
        records = payload.get("records")
        if not isinstance(env_id, str) or not isinstance(records, list):
            print(f"[Warn] Skip malformed env metadata {metadata_path}")
            continue

        env_map = difficulty_by_env_episode[env_id]
        for record in records:
            if not isinstance(record, dict):
                continue
            episode = record.get("episode")
            difficulty = record.get("difficulty")
            if not isinstance(episode, int) or not isinstance(difficulty, str):
                continue
            existing = env_map.get(episode)
            if existing is not None and existing != difficulty:
                print(
                    f"[Warn] Conflicting difficulty metadata for env={env_id} "
                    f"episode={episode}: keep {existing}, ignore {difficulty}"
                )
                continue
            env_map[episode] = difficulty

    return {env_id: dict(env_map) for env_id, env_map in difficulty_by_env_episode.items()}


def _fallback_difficulty(env_id: str, episode: int) -> Optional[str]:
    if env_id == VIDEOREPICK_ENV_ID:
        return VIDEOREPICK_DIFFICULTY_CYCLE[episode % len(VIDEOREPICK_DIFFICULTY_CYCLE)]
    return None


def _resolve_difficulty(
    difficulty_by_env_episode: dict[str, dict[int, str]],
    env_id: str,
    episode: int,
) -> Optional[str]:
    env_map = difficulty_by_env_episode.get(env_id, {})
    if episode in env_map:
        return env_map[episode]
    return _fallback_difficulty(env_id, episode)


def _build_points_from_files(
    files: list[_VisibleObjectsFile],
    difficulty_by_env_episode: dict[str, dict[int, str]],
) -> tuple[dict[str, list[VisibleObjectPoint]], Counter, Counter]:
    points_by_env: dict[str, list[VisibleObjectPoint]] = defaultdict(list)
    skipped = Counter(objects=0)
    episodes_by_env: dict[str, set[tuple[int, int]]] = defaultdict(set)

    for file_entry in files:
        episodes_by_env[file_entry.env_id].add((file_entry.episode, file_entry.seed))
        difficulty = _resolve_difficulty(
            difficulty_by_env_episode, file_entry.env_id, file_entry.episode
        )

        for obj in file_entry.objects:
            if not isinstance(obj, dict):
                skipped["objects"] += 1
                print(f"[Warn] Skip non-dict object in {file_entry.path}")
                continue

            name = obj.get("name")
            world_xyz = _safe_float_triplet(obj.get("world_xyz"))
            if not isinstance(name, str) or not name.strip() or world_xyz is None:
                skipped["objects"] += 1
                print(f"[Warn] Skip malformed object in {file_entry.path}: {obj!r}")
                continue

            semantic = _semantic_category(file_entry.env_id, name)
            points_by_env[file_entry.env_id].append(
                VisibleObjectPoint(
                    env_id=file_entry.env_id,
                    episode=file_entry.episode,
                    seed=file_entry.seed,
                    name=name,
                    world_x=world_xyz[0],
                    world_y=world_xyz[1],
                    world_z=world_xyz[2],
                    semantic=semantic,
                    cube_color=_cube_color(name),
                    button_kind=_button_kind(name),
                    difficulty=difficulty,
                    bin_index=_bin_index(name),
                    peg_part=_peg_part(name),
                )
            )

    episode_counts = Counter(
        {env_id: len(episode_keys) for env_id, episode_keys in episodes_by_env.items()}
    )
    return points_by_env, skipped, episode_counts


# ---------------------------------------------------------------------------
# XY-pipeline plotting
# ---------------------------------------------------------------------------


def _xy_rot_cw_90(x_pos: float, y_pos: float) -> tuple[float, float]:
    """俯视图顺时针旋转 90°：显示 (y, -x)。"""
    return y_pos, -x_pos


def _prepare_axis(ax, title: str, point_count: int) -> None:
    ax.set_xlim(-XY_LIMIT, XY_LIMIT)
    ax.set_ylim(-XY_LIMIT, XY_LIMIT)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World Y")
    ax.set_ylabel("-World X")
    ax.grid(True, alpha=0.45)
    ax.set_title(f"{title}\npoints={point_count}")


def _save_combined_figure(fig, output_path: Path, plt) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=AGGREGATE_DPI)
    plt.close(fig)
    return output_path


def _plot_all_objects(ax, points: list[VisibleObjectPoint]) -> None:
    from matplotlib.lines import Line2D

    grouped: dict[str, list[VisibleObjectPoint]] = defaultdict(list)
    for point in points:
        grouped[point.semantic].append(point)

    legend_handles: list[Line2D] = []
    for semantic in ALL_OBJECT_PANEL_ORDER:
        semantic_points = grouped.get(semantic, [])
        if not semantic_points:
            continue
        rotated_points = [
            _xy_rot_cw_90(point.world_x, point.world_y)
            for point in semantic_points
        ]
        xs = [point[0] for point in rotated_points]
        ys = [point[1] for point in rotated_points]
        ax.scatter(
            xs,
            ys,
            s=44,
            alpha=0.72,
            c=ALL_CATEGORY_COLORS[semantic],
            edgecolors="black",
            linewidths=0.35,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=ALL_CATEGORY_COLORS[semantic],
                markeredgecolor="black",
                markersize=8,
                label=semantic,
            )
        )

    _prepare_axis(ax, "All Visible Objects (Rotated XY)", len(points))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")
    else:
        ax.text(0.0, 0.0, "No data", ha="center", va="center")


def _cube_label(env_id: str, color_name: str) -> str:
    if env_id == ROUTESTICK_ENV_ID and color_name == "unknown":
        return "obstacle"
    return f"cube_{color_name}"


def _plot_cube_objects(ax, env_id: str, points: list[VisibleObjectPoint]) -> None:
    from matplotlib.lines import Line2D

    cube_points = [point for point in points if point.semantic == "cube"]

    legend_handles: list[Line2D] = []
    for color_name in ("red", "green", "blue", "unknown"):
        bucket = [point for point in cube_points if point.cube_color == color_name]
        if not bucket:
            continue
        rotated_points = [_xy_rot_cw_90(point.world_x, point.world_y) for point in bucket]
        ax.scatter(
            [point[0] for point in rotated_points],
            [point[1] for point in rotated_points],
            s=48,
            alpha=0.78,
            c=CUBE_COLOR_MAP[color_name],
            edgecolors="black",
            linewidths=0.35,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=CUBE_COLOR_MAP[color_name],
                markeredgecolor="black",
                markersize=8,
                label=_cube_label(env_id, color_name),
            )
        )

    _prepare_axis(ax, "Cube (Rotated XY)", len(cube_points))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")
    else:
        ax.text(0.0, 0.0, "No cube data", ha="center", va="center")


def _plot_button_objects(ax, points: list[VisibleObjectPoint]) -> None:
    from matplotlib.lines import Line2D

    button_points = [point for point in points if point.semantic == "button"]

    legend_handles: list[Line2D] = []
    for kind in ("button_base", "button_cap", "button_other"):
        bucket = [point for point in button_points if point.button_kind == kind]
        if not bucket:
            continue
        style = BUTTON_STYLE_MAP[kind]
        rotated_points = [_xy_rot_cw_90(point.world_x, point.world_y) for point in bucket]
        ax.scatter(
            [point[0] for point in rotated_points],
            [point[1] for point in rotated_points],
            s=62,
            alpha=0.8,
            c=style["color"],
            marker=style["marker"],
            edgecolors="black",
            linewidths=0.45,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="none",
                markerfacecolor=style["color"],
                markeredgecolor="black",
                markersize=8,
                label=style["label"],
            )
        )

    _prepare_axis(ax, "Button (Rotated XY)", len(button_points))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")
    else:
        ax.text(0.0, 0.0, "No button data", ha="center", va="center")


def _peg_pair_distance_sq(head_point: VisibleObjectPoint, tail_point: VisibleObjectPoint) -> float:
    dx = head_point.world_x - tail_point.world_x
    dy = head_point.world_y - tail_point.world_y
    return (dx * dx) + (dy * dy)


def _nearest_peg_pairs_for_episode(
    head_points: list[VisibleObjectPoint],
    tail_points: list[VisibleObjectPoint],
) -> list[tuple[VisibleObjectPoint, VisibleObjectPoint]]:
    pair_count = min(len(head_points), len(tail_points))
    if pair_count == 0:
        return []

    if pair_count <= 7:
        best_pairs: list[tuple[VisibleObjectPoint, VisibleObjectPoint]] = []
        best_distance_sq = float("inf")
        if len(head_points) <= len(tail_points):
            for tail_perm in permutations(tail_points, pair_count):
                candidate_pairs = list(zip(head_points, tail_perm))
                candidate_distance_sq = sum(
                    _peg_pair_distance_sq(head_point, tail_point)
                    for head_point, tail_point in candidate_pairs
                )
                if candidate_distance_sq < best_distance_sq:
                    best_distance_sq = candidate_distance_sq
                    best_pairs = candidate_pairs
        else:
            for head_perm in permutations(head_points, pair_count):
                candidate_pairs = list(zip(head_perm, tail_points))
                candidate_distance_sq = sum(
                    _peg_pair_distance_sq(head_point, tail_point)
                    for head_point, tail_point in candidate_pairs
                )
                if candidate_distance_sq < best_distance_sq:
                    best_distance_sq = candidate_distance_sq
                    best_pairs = candidate_pairs
        return best_pairs

    remaining_heads = list(head_points)
    remaining_tails = list(tail_points)
    greedy_pairs: list[tuple[VisibleObjectPoint, VisibleObjectPoint]] = []
    while remaining_heads and remaining_tails:
        best_pair: tuple[int, int] | None = None
        best_distance_sq = float("inf")
        for head_index, head_point in enumerate(remaining_heads):
            for tail_index, tail_point in enumerate(remaining_tails):
                candidate_distance_sq = _peg_pair_distance_sq(head_point, tail_point)
                if candidate_distance_sq < best_distance_sq:
                    best_distance_sq = candidate_distance_sq
                    best_pair = (head_index, tail_index)
        if best_pair is None:
            break
        head_index, tail_index = best_pair
        greedy_pairs.append((remaining_heads.pop(head_index), remaining_tails.pop(tail_index)))
    return greedy_pairs


def _nearest_episode_peg_pairs(
    peg_points: Iterable[VisibleObjectPoint],
) -> list[tuple[VisibleObjectPoint, VisibleObjectPoint]]:
    peg_points_by_episode: dict[tuple[int, int], dict[str, list[VisibleObjectPoint]]] = defaultdict(
        lambda: {"peg_head": [], "peg_tail": []}
    )
    for point in peg_points:
        if point.peg_part not in {"peg_head", "peg_tail"}:
            continue
        peg_points_by_episode[(point.episode, point.seed)][point.peg_part].append(point)

    pairs: list[tuple[VisibleObjectPoint, VisibleObjectPoint]] = []
    for episode_key in sorted(peg_points_by_episode):
        grouped_points = peg_points_by_episode[episode_key]
        pairs.extend(
            _nearest_peg_pairs_for_episode(
                grouped_points["peg_head"],
                grouped_points["peg_tail"],
            )
        )
    return pairs


def _plot_peg_objects(ax, points: list[VisibleObjectPoint]) -> None:
    from matplotlib.lines import Line2D

    peg_points = [point for point in points if point.semantic == "peg"]
    for head_point, tail_point in _nearest_episode_peg_pairs(peg_points):
        head_xy = _xy_rot_cw_90(head_point.world_x, head_point.world_y)
        tail_xy = _xy_rot_cw_90(tail_point.world_x, tail_point.world_y)
        ax.plot(
            [head_xy[0], tail_xy[0]],
            [head_xy[1], tail_xy[1]],
            color="#6b7280",
            alpha=0.35,
            linewidth=1.2,
            zorder=1,
        )

    legend_handles: list[Line2D] = []
    for peg_name in ("peg_head", "peg_tail"):
        bucket = [point for point in peg_points if point.peg_part == peg_name]
        if not bucket:
            continue
        style = PEG_PART_STYLE_MAP[peg_name]
        rotated_points = [_xy_rot_cw_90(point.world_x, point.world_y) for point in bucket]
        ax.scatter(
            [point[0] for point in rotated_points],
            [point[1] for point in rotated_points],
            s=62,
            alpha=0.8,
            c=style["color"],
            marker=style["marker"],
            edgecolors="black",
            linewidths=0.45,
            zorder=2,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="none",
                markerfacecolor=style["color"],
                markeredgecolor="black",
                markersize=8,
                label=style["label"],
            )
        )

    _prepare_axis(ax, "Peg Head/Tail (Rotated XY)", len(peg_points))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")
    else:
        ax.text(0.0, 0.0, "No peg data", ha="center", va="center")


def _plot_bin_objects(ax, points: list[VisibleObjectPoint]) -> None:
    from matplotlib.lines import Line2D

    bin_points = [point for point in points if point.semantic == "bin"]

    grouped: dict[Optional[int], list[VisibleObjectPoint]] = defaultdict(list)
    for point in bin_points:
        grouped[point.bin_index].append(point)

    legend_handles: list[Line2D] = []
    ordered_keys = sorted(
        grouped,
        key=lambda value: (value is None, -1 if value is None else value),
    )
    for key in ordered_keys:
        bucket = grouped[key]
        if not bucket:
            continue
        color = BIN_COLOR_CYCLE[(key or 0) % len(BIN_COLOR_CYCLE)]
        label = f"bin_{key}" if key is not None else "bin_unknown"
        rotated_points = [_xy_rot_cw_90(point.world_x, point.world_y) for point in bucket]
        ax.scatter(
            [point[0] for point in rotated_points],
            [point[1] for point in rotated_points],
            s=62,
            alpha=0.8,
            c=color,
            marker="s",
            edgecolors="black",
            linewidths=0.45,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=8,
                label=label,
            )
        )

    _prepare_axis(ax, "Bin (Rotated XY)", len(bin_points))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")
    else:
        ax.text(0.0, 0.0, "No bin data", ha="center", va="center")


def _plot_goal_site_objects(ax, points: list[VisibleObjectPoint]) -> None:
    goal_points = [point for point in points if point.semantic == "goal_site"]

    if goal_points:
        rotated_points = [_xy_rot_cw_90(point.world_x, point.world_y) for point in goal_points]
        ax.scatter(
            [point[0] for point in rotated_points],
            [point[1] for point in rotated_points],
            s=70,
            alpha=0.8,
            c=GOAL_SITE_STYLE["color"],
            marker=GOAL_SITE_STYLE["marker"],
            edgecolors="black",
            linewidths=0.45,
            label=GOAL_SITE_STYLE["label"],
        )
        ax.legend(loc="upper right")
    else:
        ax.text(0.0, 0.0, "No goal_site data", ha="center", va="center")

    _prepare_axis(ax, "goal_site (Rotated XY)", len(goal_points))


def _plot_box_with_hole_objects(ax, points: list[VisibleObjectPoint]) -> None:
    box_points = [point for point in points if point.semantic == "box_with_hole"]

    if box_points:
        rotated_points = [_xy_rot_cw_90(point.world_x, point.world_y) for point in box_points]
        ax.scatter(
            [point[0] for point in rotated_points],
            [point[1] for point in rotated_points],
            s=70,
            alpha=0.8,
            c=BOX_WITH_HOLE_STYLE["color"],
            marker=BOX_WITH_HOLE_STYLE["marker"],
            edgecolors="black",
            linewidths=0.45,
            label=BOX_WITH_HOLE_STYLE["label"],
        )
        ax.legend(loc="upper right")
    else:
        ax.text(0.0, 0.0, "No box_with_hole data", ha="center", va="center")

    _prepare_axis(ax, "box_with_hole (Rotated XY)", len(box_points))


def _target_panel_points(
    env_id: str,
    points: Iterable[VisibleObjectPoint],
) -> list[VisibleObjectPoint]:
    if env_id == BINFILL_ENV_ID:
        return [
            point
            for point in points
            if point.name == BINFILL_BOARD_WITH_HOLE_NAME
        ]
    return [point for point in points if point.semantic == "target"]


def _target_panel_title(env_id: str) -> str:
    if env_id == BINFILL_ENV_ID:
        return f"{BINFILL_BOARD_WITH_HOLE_NAME} (Rotated XY)"
    return "Target (Rotated XY)"


def _target_panel_empty_text(env_id: str) -> str:
    if env_id == BINFILL_ENV_ID:
        return f"No {BINFILL_BOARD_WITH_HOLE_NAME} data"
    return "No target data"


def _target_panel_style(env_id: str) -> dict[str, str]:
    if env_id == BINFILL_ENV_ID:
        return BINFILL_BOARD_STYLE
    return TARGET_STYLE


def _plot_target_objects(
    ax,
    env_id: str,
    points: list[VisibleObjectPoint],
) -> None:
    target_points = _target_panel_points(env_id, points)
    style = _target_panel_style(env_id)

    if target_points:
        rotated_points = [_xy_rot_cw_90(point.world_x, point.world_y) for point in target_points]
        ax.scatter(
            [point[0] for point in rotated_points],
            [point[1] for point in rotated_points],
            s=70,
            alpha=0.8,
            c=style["color"],
            marker=style["marker"],
            edgecolors="black",
            linewidths=0.45,
            label=style["label"],
        )
        ax.legend(loc="upper right")
    else:
        ax.text(0.0, 0.0, _target_panel_empty_text(env_id), ha="center", va="center")

    _prepare_axis(ax, _target_panel_title(env_id), len(target_points))


def _category_counts(points: Iterable[VisibleObjectPoint]) -> Counter:
    counts: Counter[str] = Counter()
    for point in points:
        counts[point.semantic] += 1
    return counts


def _count_unique_episodes(points: Iterable[VisibleObjectPoint]) -> int:
    return len({(point.episode, point.seed) for point in points})


def _panel_specs_for_env(env_id: str) -> tuple[str, ...]:
    if env_id == INSERTPEG_ENV_ID:
        return ("all", "peg", "box_with_hole")
    if env_id == MOVECUBE_ENV_ID:
        return ("all", "peg", "goal_site", "cube")
    if env_id in BIN_PANEL_ENVS:
        return ("all", "cube", "button", "bin")
    return ("all", "cube", "button", "target")


def _plot_panel(ax, panel_key: str, env_id: str, points: list[VisibleObjectPoint]) -> None:
    if panel_key == "all":
        _plot_all_objects(ax, points)
        return
    if panel_key == "cube":
        _plot_cube_objects(ax, env_id, points)
        return
    if panel_key == "button":
        _plot_button_objects(ax, points)
        return
    if panel_key == "peg":
        _plot_peg_objects(ax, points)
        return
    if panel_key == "bin":
        _plot_bin_objects(ax, points)
        return
    if panel_key == "goal_site":
        _plot_goal_site_objects(ax, points)
        return
    if panel_key == "box_with_hole":
        _plot_box_with_hole_objects(ax, points)
        return
    if panel_key == "target":
        _plot_target_objects(ax, env_id, points)
        return
    raise ValueError(f"Unsupported panel key: {panel_key}")


def _render_collage(
    output_dir: Path,
    env_id: str,
    points: list[VisibleObjectPoint],
    episode_count: int,
    title_suffix: Optional[str],
    output_filename: str,
    plt,
) -> Path:
    panel_specs = _panel_specs_for_env(env_id)
    fig, axes = plt.subplots(
        1,
        len(panel_specs),
        figsize=(7 * len(panel_specs), 7),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    title = f"{env_id} | episodes={episode_count} | points={len(points)}"
    if title_suffix:
        title = f"{title} | {title_suffix}"
    fig.suptitle(title, fontsize=18)

    for ax, panel_key in zip(axes, panel_specs):
        _plot_panel(ax, panel_key, env_id, points)

    return _save_combined_figure(fig, output_dir / output_filename, plt)


def _render_xy_env(
    output_dir: Path,
    env_id: str,
    points: list[VisibleObjectPoint],
    episode_count: int,
    plt,
) -> Counter:
    output_dir.mkdir(parents=True, exist_ok=True)

    if env_id == VIDEOREPICK_ENV_ID:
        split_specs = (
            ("easy+medium", {"easy", "medium"}, f"{env_id}{VIDEOREPICK_EASY_MEDIUM_SUFFIX}"),
            ("hard", {"hard"}, f"{env_id}{VIDEOREPICK_HARD_SUFFIX}"),
        )
        for split_name, difficulties, output_filename in split_specs:
            split_points = [point for point in points if point.difficulty in difficulties]
            if not split_points:
                print(f"[XY] {env_id}: skip empty split={split_name}")
                continue
            output_path = _render_collage(
                output_dir,
                env_id,
                split_points,
                _count_unique_episodes(split_points),
                f"difficulty={split_name}",
                output_filename,
                plt,
            )
            print(f"[XY] {env_id}: split={split_name} -> {output_path}")
    else:
        output_path = _render_collage(
            output_dir,
            env_id,
            points,
            episode_count,
            None,
            f"{env_id}{XY_DEFAULT_PNG_SUFFIX}",
            plt,
        )
        print(f"[XY] {env_id}: -> {output_path}")

    return _category_counts(points)


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
        "--hdf5-dir",
        type=Path,
        default=DEFAULT_HDF5_DIR,
        help="Directory containing per-episode <env>_ep<n>_seed<s>.h5 (or record_dataset_*.h5) files.",
    )
    parser.add_argument(
        "--segmentation-dir",
        type=Path,
        default=DEFAULT_SEGMENTATION_DIR,
        help="Directory containing reset segmentation episode folders with visible_objects.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Single shared output directory for CSV + all PNGs.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Optional env_id filter applied to both pipelines.",
    )
    parser.add_argument(
        "--max-per-difficulty",
        type=int,
        default=DEFAULT_MAX_PER_DIFFICULTY,
        help=(
            "Distribution pipeline only: max episodes per difficulty per env. "
            "Set to 0 or negative to disable."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Distribution pipeline only: display figures interactively after saving.",
    )
    return parser


def _get_pyplot(show: bool):
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def _print_skipped_seed_block(
    skipped_h5: list[_SplitH5Entry],
    skipped_json: list[_VisibleObjectsFile],
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
    max_per_difficulty: int,
    show: bool,
) -> tuple[list[_SplitH5Entry], list[_SplitH5Entry], list[Path]]:
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

    rows, read_warnings = _read_episode_rows(
        aggregated,
        kept_for_processing,
        env_filter=env_filter,
        max_per_difficulty=max_per_difficulty,
    )

    csv_path = output_dir / "episode_task_metadata.csv"
    _write_csv(rows, csv_path)

    plt = _get_pyplot(show)
    rows_by_env: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        rows_by_env.setdefault(str(row["env_id"]), []).append(row)

    figure_paths: list[Path] = []
    for env_id in sorted(rows_by_env):
        figure_path = output_dir / f"{env_id}_distribution.png"
        _render_env_figure(env_id, rows_by_env[env_id], figure_path, plt)
        figure_paths.append(figure_path)

    if show:
        plt.show()
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

    return kept_split, skipped_split, figure_paths


def _run_xy_pipeline(
    segmentation_dir: Path,
    output_dir: Path,
    env_filter: Optional[str],
) -> tuple[list[_VisibleObjectsFile], list[_VisibleObjectsFile]]:
    if not segmentation_dir.is_dir():
        print("=" * 72)
        print("[XY] visible-object aggregation summary")
        print("=" * 72)
        print(
            f"  Segmentation dir does not exist or is not a directory: "
            f"{segmentation_dir} — skipping XY pipeline."
        )
        print()
        return [], []

    files = _discover_visible_object_files(segmentation_dir)
    if env_filter is not None:
        files = [entry for entry in files if entry.env_id == env_filter]

    kept_files, skipped_files = _dedup_visible_object_files(files)

    difficulty_by_env_episode = _load_episode_difficulty_map()
    points_by_env, skipped_objects, episode_counts = _build_points_from_files(
        kept_files, difficulty_by_env_episode
    )

    print("=" * 72)
    print("[XY] visible-object aggregation summary")
    print("=" * 72)
    print(f"  Segmentation dir: {segmentation_dir}")
    if env_filter:
        print(f"  Env filter:       {env_filter}")

    if not points_by_env:
        env_part = f" for env={env_filter}" if env_filter else ""
        print(
            f"  No visible_objects.json data found under {segmentation_dir}{env_part}."
        )
        print()
        return kept_files, skipped_files

    plt = _get_pyplot(show=False)

    total_points = 0
    overall_counts: Counter[str] = Counter()
    for env_id in sorted(points_by_env):
        points = points_by_env[env_id]
        counts = _render_xy_env(
            output_dir,
            env_id,
            points,
            episode_counts.get(env_id, 0),
            plt,
        )
        total_points += len(points)
        overall_counts.update(counts)
        print(
            f"  {env_id}: episodes={episode_counts.get(env_id, 0)} "
            f"points={len(points)} cube={counts.get('cube', 0)} "
            f"button={counts.get('button', 0)} peg={counts.get('peg', 0)} "
            f"bin={counts.get('bin', 0)} goal_site={counts.get('goal_site', 0)} "
            f"box_with_hole={counts.get('box_with_hole', 0)} "
            f"target={counts.get('target', 0)} other={counts.get('other', 0)}"
        )

    plt.close("all")

    print(
        f"  [Total] envs={len(points_by_env)} total_points={total_points} "
        f"cube={overall_counts.get('cube', 0)} button={overall_counts.get('button', 0)} "
        f"peg={overall_counts.get('peg', 0)} bin={overall_counts.get('bin', 0)} "
        f"goal_site={overall_counts.get('goal_site', 0)} "
        f"box_with_hole={overall_counts.get('box_with_hole', 0)} "
        f"target={overall_counts.get('target', 0)} other={overall_counts.get('other', 0)} "
        f"skipped_objects={skipped_objects.get('objects', 0)}"
    )
    print()

    return kept_files, skipped_files


def main() -> None:
    args = _build_arg_parser().parse_args()
    hdf5_dir = args.hdf5_dir.resolve()
    segmentation_dir = args.segmentation_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir:     {output_dir}")
    print(f"HDF5 dir:       {hdf5_dir}")
    print(f"Segmentation:   {segmentation_dir}")
    if args.env:
        print(f"Env filter:     {args.env}")
    print()

    # Pipeline 1: pre-discover + dedup HDF5, then run distribution
    kept_h5, skipped_h5, _ = _run_distribution_pipeline(
        hdf5_dir,
        output_dir,
        args.env,
        args.max_per_difficulty,
        args.show,
    )

    # Pipeline 2: pre-discover + dedup JSON, then run XY aggregation
    kept_json, skipped_json = _run_xy_pipeline(
        segmentation_dir,
        output_dir,
        args.env,
    )

    # Report seed-dedup results (consolidated for both pipelines)
    _print_skipped_seed_block(skipped_h5, skipped_json)

    print(
        f"[Done] kept_hdf5={len(kept_h5)} skipped_hdf5={len(skipped_h5)} "
        f"kept_json={len(kept_json)} skipped_json={len(skipped_json)}"
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
