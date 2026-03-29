"""Aggregate reset visible object XY positions across episodes and plot by semantic group."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

VISIBLE_OBJECT_JSON_FILENAME = "visible_objects.json"
DEFAULT_INPUT_DIR = Path("runs/replay_videos/reset_segmentation_pngs")
DEFAULT_OUTPUT_DIR = Path("runs/replay_videos/aggregate_visible_object_xy")
XY_LIMIT = 0.3
AGGREGATE_DPI = 300

ALL_OBJECTS_PNG = "all_objects_xy.png"
CUBE_PNG = "cube_xy.png"
BUTTON_PNG = "button_xy.png"
TARGET_PNG = "target_xy.png"
HORIZONTAL_COLLAGE_PNG = "all_panels_horizontal.png"
VIDEOREPICK_EASY_MEDIUM_PNG = "all_panels_horizontal_easy_medium.png"
VIDEOREPICK_HARD_PNG = "all_panels_horizontal_hard.png"
BINFILL_ENV_ID = "BinFill"
BINFILL_BOARD_WITH_HOLE_NAME = "board_with_hole"
VIDEOREPICK_ENV_ID = "VideoRepick"
INSERTPEG_ENV_ID = "InsertPeg"
MOVECUBE_ENV_ID = "MoveCube"
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate visible_objects.json files and export per-env XY scatter plots "
            "for all objects, cubes, buttons, and environment-specific targets/bins/groups."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing episode reset folders with visible_objects.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where aggregated XY plots will be written.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Optional env_id filter. When omitted, process all envs found in input-dir.",
    )
    return parser


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


def _discover_json_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob(VISIBLE_OBJECT_JSON_FILENAME))


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


def _load_points(
    input_dir: Path,
    env_filter: Optional[str],
    difficulty_by_env_episode: dict[str, dict[int, str]],
) -> tuple[dict[str, list[VisibleObjectPoint]], Counter, Counter]:
    points_by_env: dict[str, list[VisibleObjectPoint]] = defaultdict(list)
    skipped = Counter(files=0, objects=0)
    episodes_by_env: dict[str, set[tuple[int, int]]] = defaultdict(set)

    for json_path in _discover_json_files(input_dir):
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            print(f"[Warn] Skip invalid JSON file {json_path}: {type(exc).__name__}: {exc}")
            skipped["files"] += 1
            continue

        env_id = payload.get("env_id")
        episode = payload.get("episode")
        seed = payload.get("seed")
        objects = payload.get("objects")

        if not isinstance(env_id, str) or not isinstance(episode, int) or not isinstance(seed, int):
            print(f"[Warn] Skip malformed metadata file {json_path}: missing env_id/episode/seed")
            skipped["files"] += 1
            continue
        if env_filter is not None and env_id != env_filter:
            continue
        if not isinstance(objects, list):
            print(f"[Warn] Skip malformed objects list in {json_path}")
            skipped["files"] += 1
            continue

        episodes_by_env[env_id].add((episode, seed))
        difficulty = _resolve_difficulty(difficulty_by_env_episode, env_id, episode)

        for obj in objects:
            if not isinstance(obj, dict):
                skipped["objects"] += 1
                print(f"[Warn] Skip non-dict object in {json_path}")
                continue

            name = obj.get("name")
            world_xyz = _safe_float_triplet(obj.get("world_xyz"))
            if not isinstance(name, str) or not name.strip() or world_xyz is None:
                skipped["objects"] += 1
                print(f"[Warn] Skip malformed object in {json_path}: {obj!r}")
                continue

            semantic = _semantic_category(env_id, name)
            points_by_env[env_id].append(
                VisibleObjectPoint(
                    env_id=env_id,
                    episode=episode,
                    seed=seed,
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


def _save_combined_figure(fig, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=AGGREGATE_DPI)
    plt.close(fig)
    return output_path


def _cleanup_legacy_panel_files(env_output_dir: Path) -> None:
    for filename in (ALL_OBJECTS_PNG, CUBE_PNG, BUTTON_PNG, TARGET_PNG):
        stale_path = env_output_dir / filename
        if stale_path.exists():
            stale_path.unlink()


def _cleanup_collage_files(env_id: str, env_output_dir: Path) -> None:
    stale_filenames = [HORIZONTAL_COLLAGE_PNG]
    if env_id == VIDEOREPICK_ENV_ID:
        stale_filenames.extend(
            [VIDEOREPICK_EASY_MEDIUM_PNG, VIDEOREPICK_HARD_PNG]
        )

    for filename in stale_filenames:
        stale_path = env_output_dir / filename
        if stale_path.exists():
            stale_path.unlink()


def _plot_all_objects(ax, points: list[VisibleObjectPoint]) -> None:
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


def _plot_cube_objects(ax, points: list[VisibleObjectPoint]) -> None:
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
                label=f"cube_{color_name}",
            )
        )

    _prepare_axis(ax, "Cube (Rotated XY)", len(cube_points))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")
    else:
        ax.text(0.0, 0.0, "No cube data", ha="center", va="center")


def _plot_button_objects(ax, points: list[VisibleObjectPoint]) -> None:
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


def _plot_peg_objects(ax, points: list[VisibleObjectPoint]) -> None:
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


def _plot_bin_objects(ax, points: list[VisibleObjectPoint]) -> None:
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
    counts = Counter()
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
        _plot_cube_objects(ax, points)
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
    env_output_dir: Path,
    env_id: str,
    points: list[VisibleObjectPoint],
    episode_count: int,
    title_suffix: Optional[str],
    output_filename: str,
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
    fig.suptitle(
        title,
        fontsize=18,
    )

    for ax, panel_key in zip(axes, panel_specs):
        _plot_panel(ax, panel_key, env_id, points)

    return _save_combined_figure(fig, env_output_dir / output_filename)


def _render_env(output_root: Path, env_id: str, points: list[VisibleObjectPoint], episode_count: int) -> Counter:
    env_output_dir = output_root / env_id
    env_output_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_legacy_panel_files(env_output_dir)
    _cleanup_collage_files(env_id, env_output_dir)

    if env_id == VIDEOREPICK_ENV_ID:
        split_specs = (
            ("easy+medium", {"easy", "medium"}, VIDEOREPICK_EASY_MEDIUM_PNG),
            ("hard", {"hard"}, VIDEOREPICK_HARD_PNG),
        )
        for split_name, difficulties, output_filename in split_specs:
            split_points = [point for point in points if point.difficulty in difficulties]
            if not split_points:
                print(f"[Env] {env_id}: skip empty split={split_name}")
                continue
            output_path = _render_collage(
                env_output_dir,
                env_id,
                split_points,
                _count_unique_episodes(split_points),
                f"difficulty={split_name}",
                output_filename,
            )
            print(f"[Env] {env_id}: split={split_name} output -> {output_path}")
    else:
        output_path = _render_collage(
            env_output_dir,
            env_id,
            points,
            episode_count,
            None,
            HORIZONTAL_COLLAGE_PNG,
        )
        print(f"[Env] {env_id}: output -> {output_path}")

    return _category_counts(points)


def main() -> None:
    args = _build_parser().parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"--input-dir does not exist or is not a directory: {input_dir}")

    difficulty_by_env_episode = _load_episode_difficulty_map()
    points_by_env, skipped, episode_counts = _load_points(
        input_dir,
        args.env,
        difficulty_by_env_episode,
    )
    if not points_by_env:
        env_part = f" for env={args.env}" if args.env else ""
        raise SystemExit(f"No visible_objects.json data found under {input_dir}{env_part}.")

    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    if args.env:
        print(f"Env filter: {args.env}")

    total_points = 0
    overall_counts = Counter()
    for env_id in sorted(points_by_env):
        points = points_by_env[env_id]
        counts = _render_env(output_dir, env_id, points, episode_counts.get(env_id, 0))
        total_points += len(points)
        overall_counts.update(counts)
        print(
            f"[Env] {env_id}: episodes={episode_counts.get(env_id, 0)} "
            f"points={len(points)} cube={counts.get('cube', 0)} "
            f"button={counts.get('button', 0)} peg={counts.get('peg', 0)} "
            f"bin={counts.get('bin', 0)} goal_site={counts.get('goal_site', 0)} "
            f"box_with_hole={counts.get('box_with_hole', 0)} "
            f"target={counts.get('target', 0)} "
            f"other={counts.get('other', 0)}"
        )

    print(
        f"[Summary] envs={len(points_by_env)} total_points={total_points} "
        f"cube={overall_counts.get('cube', 0)} button={overall_counts.get('button', 0)} "
        f"peg={overall_counts.get('peg', 0)} bin={overall_counts.get('bin', 0)} "
        f"goal_site={overall_counts.get('goal_site', 0)} "
        f"box_with_hole={overall_counts.get('box_with_hole', 0)} "
        f"target={overall_counts.get('target', 0)} other={overall_counts.get('other', 0)} "
        f"skipped_files={skipped.get('files', 0)} skipped_objects={skipped.get('objects', 0)}"
    )


if __name__ == "__main__":
    main()
