"""inspect-stat xy pipeline 的通用构件。

把原本散在 inspect_stat.py 里的：
- XY 渲染常量（XY_LIMIT / AGGREGATE_DPI / 颜色 / 样式表 / env-id）
- VisibleObjectPoint / _VisibleObjectsFile / _PickupCubeRecord 三个 dataclass
- visible_objects.json 与 pickup snapshot 的发现 + 去重 + 解析
- 全部 panel 绘制 + per-env collage 渲染 + VideoRepick 的 split 调度

整体迁移到本模块，让 4 个 suite-specific 的 inspect 模块（counting / permanance /
reference / imitation）和 inspect_stat.py 自身都从这里取共享逻辑，避免循环 import。

函数 / 常量名一律保留下划线前缀和命名，inspect_stat.py 里残留的 distribution
pipeline 不读这些，只是 xy pipeline 的 import 入口换到这里。
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")


# ---------------------------------------------------------------------------
# XY-pipeline constants
# ---------------------------------------------------------------------------

VISIBLE_OBJECT_JSON_FILENAME = "visible_objects.json"
XY_LIMIT = 0.3
AGGREGATE_DPI = 300

SNAPSHOT_DIRNAME = "snapshots"
PICKUP_SNAPSHOT_FILENAME_PATTERN = re.compile(
    r"^(?P<env_id>.+?)_ep(?P<episode>\d+)_seed(?P<seed>\d+)_after_no_record_reset\.json$"
)

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


@dataclass(frozen=True)
class _PickupCubeRecord:
    env_id: str
    episode: int
    seed: int
    difficulty: Optional[str]
    name: str
    color: str
    world_x: float
    world_y: float
    world_z: float


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


def _discover_pickup_snapshot_records(
    snapshot_dir: Path,
    env_filter: Optional[str],
    difficulty_by_env_episode: dict[tuple[str, int], str],
) -> list[_PickupCubeRecord]:
    """扫描 snapshot 目录，按 episode/seed 加载 solve_pickup_cubes 的第一条记录。

    遵循 No silent fallbacks：异常文件用 [Warn] 打印并跳过，不静默吞掉。
    后续由 _dedup_pickup_records 做 max-seed-per-episode 去重，与 visible_objects 一致。
    """
    if not snapshot_dir.is_dir():
        return []

    records: list[_PickupCubeRecord] = []
    for json_path in sorted(snapshot_dir.glob("*.json")):
        match = PICKUP_SNAPSHOT_FILENAME_PATTERN.match(json_path.name)
        if match is None:
            continue
        env_id = match.group("env_id")
        if env_filter is not None and env_id != env_filter:
            continue
        try:
            episode = int(match.group("episode"))
            seed = int(match.group("seed"))
        except ValueError:
            print(f"[Warn] Skip snapshot with non-integer episode/seed: {json_path}")
            continue

        try:
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[Warn] Skip invalid snapshot JSON {json_path}: {type(exc).__name__}: {exc}")
            continue

        solve_entries = payload.get("solve_pickup_cubes")
        if not isinstance(solve_entries, list) or not solve_entries:
            print(f"[Warn] Skip snapshot without solve_pickup_cubes entries: {json_path}")
            continue

        first = solve_entries[0]
        if not isinstance(first, dict):
            print(f"[Warn] Skip snapshot with malformed solve_pickup_cubes[0]: {json_path}")
            continue

        world_xyz = _safe_float_triplet(first.get("position_xyz"))
        if world_xyz is None:
            print(f"[Warn] Skip snapshot with malformed position_xyz: {json_path}")
            continue

        name = first.get("name") or ""
        if not isinstance(name, str):
            name = ""
        color_field = first.get("color")
        color = (
            color_field
            if isinstance(color_field, str) and color_field
            else _cube_color(name)
        )

        difficulty = difficulty_by_env_episode.get((env_id, episode))

        records.append(
            _PickupCubeRecord(
                env_id=env_id,
                episode=episode,
                seed=seed,
                difficulty=difficulty,
                name=name,
                color=color,
                world_x=world_xyz[0],
                world_y=world_xyz[1],
                world_z=world_xyz[2],
            )
        )
    return records


def _dedup_pickup_records(
    records: list[_PickupCubeRecord],
) -> list[_PickupCubeRecord]:
    """同 (env_id, episode) 多 seed 时只保留 max seed，与 visible_objects 一致。"""
    grouped: dict[tuple[str, int], list[_PickupCubeRecord]] = defaultdict(list)
    for rec in records:
        grouped[(rec.env_id, rec.episode)].append(rec)
    kept: list[_PickupCubeRecord] = []
    for bucket in grouped.values():
        bucket_sorted = sorted(bucket, key=lambda e: e.seed)
        kept.append(bucket_sorted[-1])
    kept.sort(key=lambda e: (e.env_id, e.episode))
    return kept


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


def _build_points_from_files(
    files: list[_VisibleObjectsFile],
    difficulty_by_env_episode: dict[tuple[str, int], str],
) -> tuple[dict[str, list[VisibleObjectPoint]], Counter, Counter, list[tuple[str, int]]]:
    points_by_env: dict[str, list[VisibleObjectPoint]] = defaultdict(list)
    skipped = Counter(objects=0)
    episodes_by_env: dict[str, set[tuple[int, int]]] = defaultdict(set)
    missing_difficulty: list[tuple[str, int]] = []

    for file_entry in files:
        episodes_by_env[file_entry.env_id].add((file_entry.episode, file_entry.seed))
        difficulty = difficulty_by_env_episode.get(
            (file_entry.env_id, file_entry.episode)
        )
        if difficulty is None:
            missing_difficulty.append((file_entry.env_id, file_entry.episode))

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
    return points_by_env, skipped, episode_counts, missing_difficulty


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


def _plot_videorepick_pickup_panel(
    ax,
    pickup_records: list[_PickupCubeRecord],
) -> None:
    """VideoRepick xy 子图最右侧绘制 pickup cube（按 cube 颜色着色）。"""
    if not pickup_records:
        ax.text(0.0, 0.0, "no pickup cube data", ha="center", va="center")
        _prepare_axis(ax, "pickup", 0)
        return

    by_color: dict[str, list[_PickupCubeRecord]] = defaultdict(list)
    for rec in pickup_records:
        by_color[rec.color or "unknown"].append(rec)

    plotted = 0
    for color in ("red", "green", "blue", "unknown"):
        bucket = by_color.get(color)
        if not bucket:
            continue
        rotated = [_xy_rot_cw_90(rec.world_x, rec.world_y) for rec in bucket]
        ax.scatter(
            [pt[0] for pt in rotated],
            [pt[1] for pt in rotated],
            s=110,
            alpha=0.85,
            c=CUBE_COLOR_MAP.get(color, CUBE_COLOR_MAP["unknown"]),
            marker="*",
            edgecolors="black",
            linewidths=0.5,
            label=f"pickup_{color}",
        )
        plotted += len(bucket)

    if plotted:
        ax.legend(loc="upper right")
    _prepare_axis(ax, "pickup", plotted)


def _plot_panel(
    ax,
    panel_key: str,
    env_id: str,
    points: list[VisibleObjectPoint],
    pickup_records: Optional[list[_PickupCubeRecord]] = None,
) -> None:
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
        # VideoRepick：右侧 target panel 复用为 pickup cube 视图（其余 env 行为不变）
        if env_id == VIDEOREPICK_ENV_ID:
            _plot_videorepick_pickup_panel(ax, pickup_records or [])
            return
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
    pickup_records: Optional[list[_PickupCubeRecord]] = None,
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
        _plot_panel(
            ax,
            panel_key,
            env_id,
            points,
            pickup_records=pickup_records,
        )

    return _save_combined_figure(fig, output_dir / output_filename, plt)


def _render_xy_env(
    output_dir: Path,
    env_id: str,
    points: list[VisibleObjectPoint],
    episode_count: int,
    plt,
    pickup_records: Optional[list[_PickupCubeRecord]] = None,
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
            split_pickup = (
                [rec for rec in (pickup_records or []) if rec.difficulty in difficulties]
                if pickup_records is not None
                else None
            )
            output_path = _render_collage(
                output_dir,
                env_id,
                split_points,
                _count_unique_episodes(split_points),
                f"difficulty={split_name}",
                output_filename,
                plt,
                pickup_records=split_pickup,
            )
            print(
                f"[XY] {env_id}: split={split_name} -> {output_path}"
                f" pickup_points={len(split_pickup) if split_pickup is not None else 0}"
            )
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


def _get_pyplot(show: bool):
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt
