from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import sys
import textwrap
from collections import Counter
from itertools import permutations
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

DEFAULT_DATASET_ROOT = Path("/data/hongzefu/data-0306")
DEFAULT_OUTPUT_DIR = Path("./tmp/dataset-distribution")
DEFAULT_MAX_PER_DIFFICULTY = 1000
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
    "PickHighlight",
}
AGGREGATED_H5_PREFIX = "record_dataset_"
VIDEOREPICK_METADATA_FIELD = "videorepick_metadata"
SPLIT_H5_PATTERN = re.compile(
    r"^(?P<env_id>.+?)_ep(?P<episode>\d+)(?:_seed\d+)?$"
)


def _episode_sort_key(name: str) -> tuple[int, str]:
    prefix = "episode_"
    if name.startswith(prefix):
        suffix = name[len(prefix) :]
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


def _normalize_difficulty(value: object) -> str:
    return str(value).strip().lower()


def _peek_episode_difficulty(episode_group: h5py.Group) -> str:
    setup_group = episode_group.get("setup")
    if setup_group is None or "difficulty" not in setup_group:
        return ""
    return _normalize_difficulty(_decode_dataset_string(setup_group["difficulty"][()]))


def _parse_split_h5_identity(h5_path: Path) -> tuple[str, str] | None:
    match = SPLIT_H5_PATTERN.match(h5_path.stem)
    if match is None:
        return None
    env_id = match.group("env_id")
    episode_name = f"episode_{int(match.group('episode'))}"
    return env_id, episode_name


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
    dataset_root: Path, max_per_difficulty: int | None = None
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    difficulty_counts_by_env: dict[str, Counter[str]] = {}

    for h5_path in _iter_h5_paths(dataset_root):
        try:
            with h5py.File(h5_path, "r") as handle:
                if h5_path.stem.startswith(AGGREGATED_H5_PREFIX):
                    env_id = h5_path.stem.removeprefix(AGGREGATED_H5_PREFIX)
                    difficulty_counts = difficulty_counts_by_env.setdefault(
                        env_id, Counter()
                    )
                    for episode_name in sorted(handle.keys(), key=_episode_sort_key):
                        episode_group = handle[episode_name]
                        if not isinstance(episode_group, h5py.Group):
                            warnings.append(
                                f"{env_id}/{episode_name}: expected HDF5 group"
                            )
                            continue
                        difficulty = _peek_episode_difficulty(episode_group)
                        if (
                            max_per_difficulty is not None
                            and max_per_difficulty > 0
                            and difficulty in DIFFICULTY_ORDER
                            and difficulty_counts[difficulty] >= max_per_difficulty
                        ):
                            continue
                        _append_episode_row(
                            rows,
                            warnings,
                            env_id=env_id,
                            episode_name=episode_name,
                            episode_group=episode_group,
                        )
                        if difficulty in DIFFICULTY_ORDER:
                            difficulty_counts[difficulty] += 1
                    continue

                split_identity = _parse_split_h5_identity(h5_path)
                if split_identity is None:
                    warnings.append(
                        f"{h5_path.name}: skipped unsupported HDF5 naming pattern"
                    )
                    continue

                env_id, expected_episode_name = split_identity
                difficulty_counts = difficulty_counts_by_env.setdefault(
                    env_id, Counter()
                )
                resolved_episode = _resolve_split_episode_group(
                    handle, expected_episode_name
                )
                if resolved_episode is None:
                    warnings.append(
                        f"{h5_path.name}: could not resolve episode group"
                    )
                    continue

                episode_name, episode_group = resolved_episode
                difficulty = _peek_episode_difficulty(episode_group)
                if (
                    max_per_difficulty is not None
                    and max_per_difficulty > 0
                    and difficulty in DIFFICULTY_ORDER
                    and difficulty_counts[difficulty] >= max_per_difficulty
                ):
                    continue
                _append_episode_row(
                    rows,
                    warnings,
                    env_id=env_id,
                    episode_name=episode_name,
                    episode_group=episode_group,
                )
                if difficulty in DIFFICULTY_ORDER:
                    difficulty_counts[difficulty] += 1
        except Exception as exc:
            warnings.append(f"{h5_path.name}: failed to open ({exc})")

    return rows, warnings


def _write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse episode task goals and save per-env distribution plots."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        #default=DEFAULT_DATASET_ROOT,
        default=Path("/data/hongzefu/robomme_benchmark-heldOutSeed/runs/replay_videos/hdf5_files"),
        help=(
            "Directory or HDF5 file containing either record_dataset_*.h5 "
            "files or per-episode *_ep*_seed*.h5 files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNGs and CSV summary will be written.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving them.",
    )
    parser.add_argument(
        "--max-per-difficulty",
        type=int,
        default=DEFAULT_MAX_PER_DIFFICULTY,
        help=(
            "Maximum number of episodes to read for each difficulty. "
            "Set to 0 or a negative value to disable the limit."
        ),
    )
    return parser


def _get_pyplot(show: bool):
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def _print_summary(
    rows: list[dict[str, object]],
    warnings: list[str],
    output_dir: Path,
    csv_path: Path,
) -> None:
    print(f"Output directory: {output_dir}")
    print(f"Episode metadata CSV: {csv_path}")

    rows_by_env: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        rows_by_env.setdefault(str(row["env_id"]), []).append(row)

    for env_id in sorted(rows_by_env):
        env_rows = rows_by_env[env_id]
        parse_issues = sum(1 for row in env_rows if row["parse_error"])
        figure_path = output_dir / f"{env_id}.png"
        print(
            f"{env_id}: episodes={len(env_rows)} "
            f"parse_issues={parse_issues} figure={figure_path}"
        )

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, warnings = _read_episode_rows(
        dataset_root, max_per_difficulty=args.max_per_difficulty
    )
    csv_path = output_dir / "episode_task_metadata.csv"
    _write_csv(rows, csv_path)

    plt = _get_pyplot(args.show)

    rows_by_env: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        rows_by_env.setdefault(str(row["env_id"]), []).append(row)

    for env_id in sorted(rows_by_env):
        figure_path = output_dir / f"{env_id}.png"
        _render_env_figure(env_id, rows_by_env[env_id], figure_path, plt)

    if args.show:
        plt.show()
    plt.close("all")

    _print_summary(rows, warnings, output_dir, csv_path)


if __name__ == "__main__":
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    try:
        main()
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(0)
