"""Aggregate reset visible object XY positions across episodes and plot by semantic group."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
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

ALL_CATEGORY_COLORS = {
    "cube": "#2ca02c",
    "button": "#9467bd",
    "target": "#e15759",
    "other": "#7f7f7f",
}
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
TARGET_STYLE = {"color": "#e15759", "marker": "^", "label": "target"}


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate visible_objects.json files and export per-env XY scatter plots "
            "for all objects, cubes, buttons, and targets."
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


def _semantic_category(name: str) -> str:
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


def _load_points(
    input_dir: Path,
    env_filter: Optional[str],
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

            semantic = _semantic_category(name)
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


def _plot_all_objects(ax, points: list[VisibleObjectPoint]) -> None:
    grouped: dict[str, list[VisibleObjectPoint]] = defaultdict(list)
    for point in points:
        grouped[point.semantic].append(point)

    legend_handles: list[Line2D] = []
    for semantic in ("cube", "button", "target", "other"):
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


def _plot_target_objects(ax, points: list[VisibleObjectPoint]) -> None:
    target_points = [point for point in points if point.semantic == "target"]

    if target_points:
        rotated_points = [_xy_rot_cw_90(point.world_x, point.world_y) for point in target_points]
        ax.scatter(
            [point[0] for point in rotated_points],
            [point[1] for point in rotated_points],
            s=70,
            alpha=0.8,
            c=TARGET_STYLE["color"],
            marker=TARGET_STYLE["marker"],
            edgecolors="black",
            linewidths=0.45,
            label=TARGET_STYLE["label"],
        )
        ax.legend(loc="upper right")
    else:
        ax.text(0.0, 0.0, "No target data", ha="center", va="center")

    _prepare_axis(ax, "Target (Rotated XY)", len(target_points))


def _category_counts(points: Iterable[VisibleObjectPoint]) -> Counter:
    counts = Counter()
    for point in points:
        counts[point.semantic] += 1
    return counts


def _render_env(output_root: Path, env_id: str, points: list[VisibleObjectPoint], episode_count: int) -> Counter:
    env_output_dir = output_root / env_id
    env_output_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_legacy_panel_files(env_output_dir)

    fig, axes = plt.subplots(1, 4, figsize=(28, 7), sharex=True, sharey=True)
    fig.suptitle(
        f"{env_id} | episodes={episode_count} | points={len(points)}",
        fontsize=18,
    )

    _plot_all_objects(axes[0], points)
    _plot_cube_objects(axes[1], points)
    _plot_button_objects(axes[2], points)
    _plot_target_objects(axes[3], points)

    output_path = _save_combined_figure(
        fig,
        env_output_dir / HORIZONTAL_COLLAGE_PNG,
    )
    print(f"[Env] {env_id}: output -> {output_path}")

    return _category_counts(points)


def main() -> None:
    args = _build_parser().parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"--input-dir does not exist or is not a directory: {input_dir}")

    points_by_env, skipped, episode_counts = _load_points(input_dir, args.env)
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
            f"button={counts.get('button', 0)} target={counts.get('target', 0)} "
            f"other={counts.get('other', 0)}"
        )

    print(
        f"[Summary] envs={len(points_by_env)} total_points={total_points} "
        f"cube={overall_counts.get('cube', 0)} button={overall_counts.get('button', 0)} "
        f"target={overall_counts.get('target', 0)} other={overall_counts.get('other', 0)} "
        f"skipped_files={skipped.get('files', 0)} skipped_objects={skipped.get('objects', 0)}"
    )


if __name__ == "__main__":
    main()
