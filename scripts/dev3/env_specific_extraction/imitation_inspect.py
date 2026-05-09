"""Imitation 套件（MoveCube / InsertPeg / PatternLock / RouteStick）专用
xy 渲染入口。

模板对齐 reference_inspect.py：可被 inspect_stat.py 调用，也可独立运行。
单一公开接口：

    visualize(segmentation_dir, output_dir, env_id=None,
              difficulty_by_env_episode=None) -> (kept, skipped)

四个 imitation env 的 figure 布局：

- **InsertPeg** 2 行 × 3 列：第 1 行 = ("all", "peg", "box_with_hole") 复用
  xy_common._plot_panel；第 2 行 = (left/right 占比柱状图 / insert-end head/tail
  占比柱状图 / 选中端 xy 散点 + box 中心)。数据来源 = visible_objects.json
  顶层 ``insertpeg_choice`` 字段。
- **MoveCube** 2 行 × 4 列：第 1 行 = ("all", "peg", "goal_site", "cube")
  复用；第 2 行 = 跨 4 列的 way 占比柱状图。数据来源 = visible_objects.json
  顶层 ``movecube_choice`` 字段。
- **PatternLock / RouteStick** 单行：保持原 xy_common._render_xy_env 单行多列
  collage（imitation 套件内这 2 个 env 不写 reset 阶段 choice 字段，所以无第
  2 行）。

Discover/dedup 模式仿 reference_inspect：每个 env 各自独立扫描 visible_objects.json
顶层字段，跳过缺字段的旧 episode（打 [Warn]）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("MPLBACKEND", "Agg")

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import imitation as imitation_module  # noqa: E402
import xy_common  # noqa: E402


IMITATION_ENV_IDS: frozenset[str] = frozenset(
    {"MoveCube", "InsertPeg", "PatternLock", "RouteStick"}
)

# 双行布局 env：写了 reset 阶段 choice 字段
INSERTPEG_ENV_ID = "InsertPeg"
MOVECUBE_ENV_ID = "MoveCube"
TWO_ROW_ENV_IDS: frozenset[str] = frozenset({INSERTPEG_ENV_ID, MOVECUBE_ENV_ID})

VISIBLE_OBJECTS_JSON_FILENAME = "visible_objects.json"

_DEFAULT_BASE = Path("/data/hongzefu/robomme_benchmark_cvpr2026-heldoutSeed/runs/replay_videos")
DEFAULT_SEGMENTATION_DIR = _DEFAULT_BASE / "reset_segmentation_pngs"
DEFAULT_OUTPUT_DIR = _DEFAULT_BASE / "inspect-stat" / "xy"

_BAR_DEFAULT_COLOR = "#1f77b4"  # tab:blue

# InsertPeg 第 2 行第 2 列：选中端散点的颜色（按 insert_end label 区分）和 marker
# （按 direction label 区分）
_INSERT_END_COLOR_MAP = {"head": "#1f77b4", "tail": "#ff7f0e"}  # tab:blue / tab:orange
_DIRECTION_MARKER_MAP = {"left": "o", "right": "s"}            # 圆=left / 方=right
_BOX_CENTER_COLOR = "#bb0000"
_BOX_CENTER_MARKER = "X"

# MoveCube way 柱状图：3 个 way 的颜色，与已有 panel 风格保持一致
_MOVECUBE_WAY_BAR_COLORS = {
    "peg_push": "#1f77b4",       # tab:blue
    "gripper_push": "#2ca02c",   # tab:green
    "grasp_putdown": "#d62728",  # tab:red
}


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _load_visible_objects(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _annotate_bars(ax: Any, counts: list[int], total: int) -> None:
    """每个 bar 顶端写 count (pct%)。"""
    if total <= 0:
        return
    for rect, count in zip(ax.patches, counts):
        if count <= 0:
            continue
        pct = 100.0 * count / total
        ax.annotate(
            f"{count} ({pct:.0f}%)",
            xy=(rect.get_x() + rect.get_width() / 2.0, rect.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _draw_categorical_bar(
    ax: Any,
    title: str,
    labels: list[str],
    counts: list[int],
    *,
    bar_colors: Optional[list[str]] = None,
) -> None:
    """画分类柱状图（labels 与 counts 等长）。empty 数据写 'no data' 占位。"""
    if not labels:
        ax.text(
            0.5,
            0.5,
            "no data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(f"{title}\nepisodes=0")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    x_positions = list(range(len(labels)))
    colors = bar_colors if bar_colors is not None else [_BAR_DEFAULT_COLOR] * len(labels)
    ax.bar(x_positions, counts, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("count")
    total = sum(counts)
    ax.set_title(f"{title}\nepisodes={total}")
    ax.grid(axis="y", alpha=0.3)
    if counts:
        max_count = max(counts)
        ax.set_ylim(0, max(1, int(max_count * 1.15) + 1))
    _annotate_bars(ax, counts, total)


# ---------------------------------------------------------------------------
# Discover/dedup：InsertPeg
# ---------------------------------------------------------------------------


@dataclass
class InsertPegChoiceRecord:
    path: Path
    env_id: str
    episode: int
    seed: int
    choice: dict   # visible_objects.json 顶层 'insertpeg_choice' dict


def _validate_insertpeg_choice(choice: dict) -> bool:
    """关键字段齐备且类型/值合法时返回 True。"""
    if not isinstance(choice, dict):
        return False
    if int(choice.get("direction", 0)) not in (-1, 1):
        return False
    if int(choice.get("obj_flag", 0)) not in (-1, 1):
        return False
    if str(choice.get("direction_label", "")) not in ("left", "right"):
        return False
    if str(choice.get("insert_end_label", "")) not in ("head", "tail"):
        return False
    for k in ("peg_head_xy", "peg_tail_xy", "insert_end_xy", "box_xy"):
        v = choice.get(k)
        if not (isinstance(v, (list, tuple)) and len(v) >= 2):
            return False
    return True


def _discover_insertpeg_choice_records(
    segmentation_dir: Path,
    env_filter: Optional[str] = None,
) -> list[InsertPegChoiceRecord]:
    """递归扫描 segmentation_dir 下所有 visible_objects.json，提取
    payload['insertpeg_choice']。

    - 只处理 env_id == 'InsertPeg'，其它 env 跳过
    - env_filter 非空且 != 'InsertPeg' 时直接返回空列表
    - 字段缺失 / 类型不合法时打 [Warn] 并跳过（说明该条 episode 数据来自旧
      rollout，需要重跑）
    """
    seg_dir = Path(segmentation_dir)
    if not seg_dir.is_dir():
        return []
    if env_filter is not None and env_filter != INSERTPEG_ENV_ID:
        return []

    results: list[InsertPegChoiceRecord] = []
    for json_path in sorted(seg_dir.rglob(VISIBLE_OBJECTS_JSON_FILENAME)):
        try:
            payload = _load_visible_objects(json_path)
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"[Warn] Skip invalid visible_objects JSON {json_path}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        env_id = payload.get("env_id")
        episode = payload.get("episode")
        seed = payload.get("seed")
        if (
            not isinstance(env_id, str)
            or not isinstance(episode, int)
            or not isinstance(seed, int)
        ):
            print(
                f"[Warn] Skip visible_objects JSON {json_path} missing "
                f"env_id/episode/seed"
            )
            continue
        if env_id != INSERTPEG_ENV_ID:
            continue

        choice = payload.get(imitation_module.INSERTPEG_CHOICE_KEY)
        if not _validate_insertpeg_choice(choice):
            print(
                f"[Warn] {env_id} ep{episode} seed{seed}: "
                f"visible_objects.json missing/invalid 'insertpeg_choice' "
                f"(re-run rollout to populate). Skipping {json_path}."
            )
            continue
        results.append(
            InsertPegChoiceRecord(
                path=json_path,
                env_id=env_id,
                episode=episode,
                seed=seed,
                choice=choice,
            )
        )
    return results


def _dedup_insertpeg_choice_records(
    entries: list[InsertPegChoiceRecord],
) -> tuple[list[InsertPegChoiceRecord], list[InsertPegChoiceRecord]]:
    """同 (env_id, episode) 多 seed 时只保留 max seed —— 与 reference 的 dedup
    行为完全一致。"""
    grouped: dict[tuple[str, int], list[InsertPegChoiceRecord]] = {}
    for entry in entries:
        grouped.setdefault((entry.env_id, entry.episode), []).append(entry)
    kept: list[InsertPegChoiceRecord] = []
    skipped: list[InsertPegChoiceRecord] = []
    for bucket in grouped.values():
        bucket_sorted = sorted(bucket, key=lambda e: e.seed)
        kept.append(bucket_sorted[-1])
        skipped.extend(bucket_sorted[:-1])
    kept.sort(key=lambda e: (e.env_id, e.episode))
    skipped.sort(key=lambda e: (e.env_id, e.episode, e.seed))
    return kept, skipped


# ---------------------------------------------------------------------------
# Discover/dedup：MoveCube
# ---------------------------------------------------------------------------


@dataclass
class MoveCubeChoiceRecord:
    path: Path
    env_id: str
    episode: int
    seed: int
    way: str   # one of MOVECUBE_WAYS


def _discover_movecube_choice_records(
    segmentation_dir: Path,
    env_filter: Optional[str] = None,
) -> list[MoveCubeChoiceRecord]:
    seg_dir = Path(segmentation_dir)
    if not seg_dir.is_dir():
        return []
    if env_filter is not None and env_filter != MOVECUBE_ENV_ID:
        return []

    results: list[MoveCubeChoiceRecord] = []
    for json_path in sorted(seg_dir.rglob(VISIBLE_OBJECTS_JSON_FILENAME)):
        try:
            payload = _load_visible_objects(json_path)
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"[Warn] Skip invalid visible_objects JSON {json_path}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        env_id = payload.get("env_id")
        episode = payload.get("episode")
        seed = payload.get("seed")
        if (
            not isinstance(env_id, str)
            or not isinstance(episode, int)
            or not isinstance(seed, int)
        ):
            print(
                f"[Warn] Skip visible_objects JSON {json_path} missing "
                f"env_id/episode/seed"
            )
            continue
        if env_id != MOVECUBE_ENV_ID:
            continue

        choice = payload.get(imitation_module.MOVECUBE_CHOICE_KEY)
        if not isinstance(choice, dict):
            print(
                f"[Warn] {env_id} ep{episode} seed{seed}: "
                f"visible_objects.json missing 'movecube_choice' field "
                f"(re-run rollout to populate). Skipping {json_path}."
            )
            continue
        way = str(choice.get("way", ""))
        if way not in imitation_module.MOVECUBE_WAYS:
            print(
                f"[Warn] {env_id} ep{episode} seed{seed}: "
                f"movecube_choice.way={way!r} not in MOVECUBE_WAYS. "
                f"Skipping {json_path}."
            )
            continue
        results.append(
            MoveCubeChoiceRecord(
                path=json_path,
                env_id=env_id,
                episode=episode,
                seed=seed,
                way=way,
            )
        )
    return results


def _dedup_movecube_choice_records(
    entries: list[MoveCubeChoiceRecord],
) -> tuple[list[MoveCubeChoiceRecord], list[MoveCubeChoiceRecord]]:
    grouped: dict[tuple[str, int], list[MoveCubeChoiceRecord]] = {}
    for entry in entries:
        grouped.setdefault((entry.env_id, entry.episode), []).append(entry)
    kept: list[MoveCubeChoiceRecord] = []
    skipped: list[MoveCubeChoiceRecord] = []
    for bucket in grouped.values():
        bucket_sorted = sorted(bucket, key=lambda e: e.seed)
        kept.append(bucket_sorted[-1])
        skipped.extend(bucket_sorted[:-1])
    kept.sort(key=lambda e: (e.env_id, e.episode))
    skipped.sort(key=lambda e: (e.env_id, e.episode, e.seed))
    return kept, skipped


# ---------------------------------------------------------------------------
# InsertPeg：第 2 行第 3 列「选中端 xy 散点」面板
# ---------------------------------------------------------------------------


def _draw_insertpeg_insert_end_xy_panel(
    ax: Any,
    records: list[InsertPegChoiceRecord],
) -> None:
    """背景层 = 所有 record 的 peg_head + peg_tail 灰点；前景层 = 每个 record
    的 insert_end 一点（颜色 = insert_end head/tail，marker = direction left/right）；
    box 中心用 'X' 标出。"""
    from matplotlib.lines import Line2D

    # 背景：所有 head/tail 灰点（reference 统计可对照）
    bg_xy: list[tuple[float, float]] = []
    box_xy: list[tuple[float, float]] = []
    for rec in records:
        ch = rec.choice
        head = ch.get("peg_head_xy")
        tail = ch.get("peg_tail_xy")
        box = ch.get("box_xy")
        if isinstance(head, (list, tuple)) and len(head) >= 2:
            bg_xy.append(xy_common._xy_rot_cw_90(float(head[0]), float(head[1])))
        if isinstance(tail, (list, tuple)) and len(tail) >= 2:
            bg_xy.append(xy_common._xy_rot_cw_90(float(tail[0]), float(tail[1])))
        if isinstance(box, (list, tuple)) and len(box) >= 2:
            box_xy.append(xy_common._xy_rot_cw_90(float(box[0]), float(box[1])))

    if bg_xy:
        ax.scatter(
            [p[0] for p in bg_xy],
            [p[1] for p in bg_xy],
            s=20,
            color="lightgray",
            alpha=0.5,
            edgecolors="none",
            label="all peg_head + peg_tail",
            zorder=1,
        )

    # 前景：按 (insert_end_label, direction_label) 分组绘制，4 种组合
    fg_groups: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for rec in records:
        ch = rec.choice
        end_label = str(ch.get("insert_end_label", "head"))
        dir_label = str(ch.get("direction_label", "left"))
        end_xy = ch.get("insert_end_xy")
        if not (isinstance(end_xy, (list, tuple)) and len(end_xy) >= 2):
            continue
        key = (end_label, dir_label)
        fg_groups.setdefault(key, []).append(
            xy_common._xy_rot_cw_90(float(end_xy[0]), float(end_xy[1]))
        )

    legend_handles: list[Line2D] = []
    # 顺序：(head, left) (head, right) (tail, left) (tail, right) — 视觉一致
    for end_label in ("head", "tail"):
        for dir_label in ("left", "right"):
            pts = fg_groups.get((end_label, dir_label))
            if not pts:
                continue
            color = _INSERT_END_COLOR_MAP[end_label]
            marker = _DIRECTION_MARKER_MAP[dir_label]
            ax.scatter(
                [p[0] for p in pts],
                [p[1] for p in pts],
                s=80,
                c=color,
                marker=marker,
                edgecolors="black",
                linewidths=0.6,
                alpha=0.9,
                zorder=3,
                label=f"insert={end_label}, dir={dir_label} (n={len(pts)})",
            )
            legend_handles.append(
                Line2D(
                    [0], [0],
                    marker=marker, linestyle="",
                    markersize=8,
                    markerfacecolor=color,
                    markeredgecolor="black",
                    label=f"insert={end_label}, dir={dir_label} (n={len(pts)})",
                )
            )

    if box_xy:
        ax.scatter(
            [p[0] for p in box_xy],
            [p[1] for p in box_xy],
            s=120,
            c=_BOX_CENTER_COLOR,
            marker=_BOX_CENTER_MARKER,
            linewidths=1.0,
            zorder=4,
            label=f"box center (n={len(box_xy)})",
        )
        legend_handles.append(
            Line2D(
                [0], [0],
                marker=_BOX_CENTER_MARKER, linestyle="",
                markersize=10,
                markerfacecolor=_BOX_CENTER_COLOR,
                markeredgecolor=_BOX_CENTER_COLOR,
                label=f"box center (n={len(box_xy)})",
            )
        )

    ax.set_xlim(-xy_common.XY_LIMIT, xy_common.XY_LIMIT)
    ax.set_ylim(-xy_common.XY_LIMIT, xy_common.XY_LIMIT)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World Y")
    ax.set_ylabel("-World X")
    ax.grid(True, alpha=0.45)
    total = sum(len(v) for v in fg_groups.values())
    ax.set_title(f"Insert-end XY (color=end, marker=dir)\nepisodes={total}")
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)


# ---------------------------------------------------------------------------
# InsertPeg / MoveCube 的多行 figure 渲染入口
# ---------------------------------------------------------------------------


def _render_insertpeg_two_row_figure(
    output_dir: Path,
    points: list[xy_common.VisibleObjectPoint],
    episode_count: int,
    records: list[InsertPegChoiceRecord],
    plt: Any,
) -> Path:
    """InsertPeg 2 行 × 3 列 figure：第 1 行 = ("all", "peg", "box_with_hole")
    复用 xy_common._plot_panel；第 2 行 = (left/right bar / head/tail bar /
    insert_end xy)。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{INSERTPEG_ENV_ID}{xy_common.XY_DEFAULT_PNG_SUFFIX}"

    panel_specs = xy_common._panel_specs_for_env(INSERTPEG_ENV_ID)
    n_cols = len(panel_specs)

    fig = plt.figure(figsize=(7 * n_cols, 7 * 2))
    gs = fig.add_gridspec(2, n_cols)

    # Row 0: 现有 visible-objects panel collage
    for col_idx, panel_key in enumerate(panel_specs):
        ax = fig.add_subplot(gs[0, col_idx])
        xy_common._plot_panel(ax, panel_key, INSERTPEG_ENV_ID, points)

    # Row 1 col 0: left/right 占比柱状图
    direction_counter: Counter[str] = Counter()
    for rec in records:
        direction_counter[str(rec.choice.get("direction_label", "unknown"))] += 1
    direction_labels = ["left", "right"]
    direction_counts = [direction_counter.get(lbl, 0) for lbl in direction_labels]
    ax_dir = fig.add_subplot(gs[1, 0])
    _draw_categorical_bar(
        ax_dir,
        title="Insertion direction (left/right)",
        labels=direction_labels,
        counts=direction_counts,
    )

    # Row 1 col 1: insert_end head/tail 占比柱状图
    end_counter: Counter[str] = Counter()
    for rec in records:
        end_counter[str(rec.choice.get("insert_end_label", "unknown"))] += 1
    end_labels = ["head", "tail"]
    end_counts = [end_counter.get(lbl, 0) for lbl in end_labels]
    end_colors = [_INSERT_END_COLOR_MAP[lbl] for lbl in end_labels]
    ax_end = fig.add_subplot(gs[1, 1])
    _draw_categorical_bar(
        ax_end,
        title="Insert-end (peg head / tail)",
        labels=end_labels,
        counts=end_counts,
        bar_colors=end_colors,
    )

    # Row 1 col 2: insert_end xy 散点 + box 中心
    ax_xy = fig.add_subplot(gs[1, 2])
    _draw_insertpeg_insert_end_xy_panel(ax_xy, records)

    title = (
        f"{INSERTPEG_ENV_ID} | episodes={episode_count} | points={len(points)} "
        f"| insertpeg_choice={len(records)}"
    )
    fig.suptitle(title, fontsize=18)
    return xy_common._save_combined_figure(fig, output_path, plt)


def _render_movecube_two_row_figure(
    output_dir: Path,
    points: list[xy_common.VisibleObjectPoint],
    episode_count: int,
    records: list[MoveCubeChoiceRecord],
    plt: Any,
) -> Path:
    """MoveCube 2 行 × 4 列 figure：第 1 行 = ("all", "peg", "goal_site",
    "cube") 复用 xy_common._plot_panel；第 2 行 = 跨 4 列的 way 占比柱状图。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{MOVECUBE_ENV_ID}{xy_common.XY_DEFAULT_PNG_SUFFIX}"

    panel_specs = xy_common._panel_specs_for_env(MOVECUBE_ENV_ID)
    n_cols = len(panel_specs)

    fig = plt.figure(figsize=(7 * n_cols, 7 * 2))
    gs = fig.add_gridspec(2, n_cols)

    # Row 0: 现有 visible-objects panel collage
    for col_idx, panel_key in enumerate(panel_specs):
        ax = fig.add_subplot(gs[0, col_idx])
        xy_common._plot_panel(ax, panel_key, MOVECUBE_ENV_ID, points)

    # Row 1: 跨 4 列的 way 占比柱状图
    way_counter: Counter[str] = Counter()
    for rec in records:
        way_counter[rec.way] += 1
    way_labels = list(imitation_module.MOVECUBE_WAYS)
    way_counts = [way_counter.get(w, 0) for w in way_labels]
    way_colors = [_MOVECUBE_WAY_BAR_COLORS[w] for w in way_labels]
    ax_way = fig.add_subplot(gs[1, 0:n_cols])
    _draw_categorical_bar(
        ax_way,
        title="Move-cube way (peg_push / gripper_push / grasp_putdown)",
        labels=way_labels,
        counts=way_counts,
        bar_colors=way_colors,
    )

    title = (
        f"{MOVECUBE_ENV_ID} | episodes={episode_count} | points={len(points)} "
        f"| movecube_choice={len(records)}"
    )
    fig.suptitle(title, fontsize=18)
    return xy_common._save_combined_figure(fig, output_path, plt)


# ---------------------------------------------------------------------------
# 公开接口：visualize
# ---------------------------------------------------------------------------


def visualize(
    segmentation_dir: Path,
    output_dir: Path,
    *,
    env_id: Optional[str] = None,
    difficulty_by_env_episode: Optional[dict[tuple[str, int], str]] = None,
) -> tuple[list, list]:
    """渲染 Imitation 套件每个 env 的 xy collage。

    Parameters
    ----------
    segmentation_dir:
        包含各 episode 子目录（内有 visible_objects.json）的根目录。
    output_dir:
        最终 PNG 写入位置（通常是 ``inspect-stat/xy``）。
    env_id:
        指定只处理哪个 env；``None`` = 处理 4 个全部。传入非 imitation env
        时静默返回空列表。
    difficulty_by_env_episode:
        来自 distribution pipeline 的 (env, episode) -> difficulty 映射。
        Imitation 套件本身没有 difficulty 拆分，但保留以与其余 suite 接口
        一致（也是 _build_points_from_files 的必要输入）。

    Returns
    -------
    (kept, skipped) : tuple[list[xy_common._VisibleObjectsFile], ...]
    """
    segmentation_dir = Path(segmentation_dir)
    output_dir = Path(output_dir)

    if env_id is not None and env_id not in IMITATION_ENV_IDS:
        return [], []

    files = xy_common._discover_visible_object_files(segmentation_dir)
    files = [
        entry
        for entry in files
        if entry.env_id in IMITATION_ENV_IDS
        and (env_id is None or entry.env_id == env_id)
    ]

    kept, skipped = xy_common._dedup_visible_object_files(files)
    if not kept:
        env_part = f" for env_id={env_id!r}" if env_id else ""
        print(
            f"[Imitation-inspect] No visible_objects.json found "
            f"under {segmentation_dir}{env_part}."
        )
        return kept, skipped

    points_by_env, skipped_objects, episode_counts, _ = (
        xy_common._build_points_from_files(kept, difficulty_by_env_episode or {})
    )

    # 收集 InsertPeg choice records
    insertpeg_records_by_env: dict[str, list[InsertPegChoiceRecord]] = {}
    if env_id is None or env_id == INSERTPEG_ENV_ID:
        raw_ip = _discover_insertpeg_choice_records(segmentation_dir, env_filter=env_id)
        kept_ip, skipped_ip = _dedup_insertpeg_choice_records(raw_ip)
        for rec in kept_ip:
            insertpeg_records_by_env.setdefault(rec.env_id, []).append(rec)
        print(
            f"  InsertPeg-choice records: kept={len(kept_ip)} "
            f"skipped(dup)={len(skipped_ip)}"
        )

    # 收集 MoveCube choice records
    movecube_records_by_env: dict[str, list[MoveCubeChoiceRecord]] = {}
    if env_id is None or env_id == MOVECUBE_ENV_ID:
        raw_mc = _discover_movecube_choice_records(segmentation_dir, env_filter=env_id)
        kept_mc, skipped_mc = _dedup_movecube_choice_records(raw_mc)
        for rec in kept_mc:
            movecube_records_by_env.setdefault(rec.env_id, []).append(rec)
        print(
            f"  MoveCube-choice records: kept={len(kept_mc)} "
            f"skipped(dup)={len(skipped_mc)}"
        )

    plt = xy_common._get_pyplot(show=False)
    for eid in sorted(points_by_env):
        points = points_by_env[eid]
        if eid == INSERTPEG_ENV_ID:
            ip_records = insertpeg_records_by_env.get(eid, [])
            _render_insertpeg_two_row_figure(
                output_dir,
                points,
                episode_counts.get(eid, 0),
                ip_records,
                plt,
            )
            counts = xy_common._category_counts(points)
            print(
                f"  {eid}: episodes={episode_counts.get(eid, 0)} "
                f"points={len(points)} "
                f"insertpeg_records={len(ip_records)} "
                f"peg={counts.get('peg', 0)} "
                f"box_with_hole={counts.get('box_with_hole', 0)} "
                f"other={counts.get('other', 0)}"
            )
        elif eid == MOVECUBE_ENV_ID:
            mc_records = movecube_records_by_env.get(eid, [])
            _render_movecube_two_row_figure(
                output_dir,
                points,
                episode_counts.get(eid, 0),
                mc_records,
                plt,
            )
            counts = xy_common._category_counts(points)
            print(
                f"  {eid}: episodes={episode_counts.get(eid, 0)} "
                f"points={len(points)} "
                f"movecube_records={len(mc_records)} "
                f"cube={counts.get('cube', 0)} "
                f"peg={counts.get('peg', 0)} "
                f"goal_site={counts.get('goal_site', 0)} "
                f"other={counts.get('other', 0)}"
            )
        else:
            # PatternLock / RouteStick：保持原单行 fallback
            counts = xy_common._render_xy_env(
                output_dir,
                eid,
                points,
                episode_counts.get(eid, 0),
                plt,
            )
            print(
                f"  {eid}: episodes={episode_counts.get(eid, 0)} "
                f"points={len(points)} cube={counts.get('cube', 0)} "
                f"button={counts.get('button', 0)} peg={counts.get('peg', 0)} "
                f"bin={counts.get('bin', 0)} goal_site={counts.get('goal_site', 0)} "
                f"box_with_hole={counts.get('box_with_hole', 0)} "
                f"target={counts.get('target', 0)} other={counts.get('other', 0)}"
            )
    plt.close("all")

    print(
        f"[Imitation-inspect] envs={len(points_by_env)} "
        f"kept={len(kept)} skipped={len(skipped)} "
        f"skipped_objects={skipped_objects.get('objects', 0)}"
    )
    return kept, skipped


_VALID_ENV_IDS = sorted(IMITATION_ENV_IDS)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Imitation 套件专用 xy 可视化：扫描 visible_objects.json，为 "
            "MoveCube / InsertPeg / PatternLock / RouteStick 各生成对应的 "
            "xy collage。InsertPeg 与 MoveCube 走 2 行布局（第 2 行展示 "
            "reset 阶段写入的 choice 字段对应的 distribution）；"
            "PatternLock / RouteStick 保持单行 collage。"
        )
    )
    parser.add_argument(
        "--segmentation-dir",
        type=Path,
        default=DEFAULT_SEGMENTATION_DIR,
        help="包含各 episode 子目录（内有 visible_objects.json）的根目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="PNG 输出目录。",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=None,
        choices=_VALID_ENV_IDS,
        help=(
            f"只处理指定的 imitation env，不传则处理全部四个。"
            f"可选值：{_VALID_ENV_IDS}"
        ),
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    segmentation_dir = args.segmentation_dir.resolve()
    output_dir = args.output_dir.resolve()

    print(f"Segmentation dir: {segmentation_dir}")
    print(f"Output dir:       {output_dir}")
    if args.env_id:
        print(f"Env filter:       {args.env_id}")
    print()

    kept, skipped = visualize(
        segmentation_dir=segmentation_dir,
        output_dir=output_dir,
        env_id=args.env_id,
    )

    print()
    print(f"[Done] kept={len(kept)} skipped={len(skipped)}")


if __name__ == "__main__":
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    try:
        main()
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(0)
