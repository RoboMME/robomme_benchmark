"""Reference 套件（PickHighlight / VideoRepick / VideoPlaceButton / VideoPlaceOrder）
专用 xy 渲染入口。

模板对齐 permanance_inspect.py：可被 inspect_stat.py 调用，也可独立运行。
单一公开接口：

    visualize(segmentation_dir, output_dir, env_id=None,
              difficulty_by_env_episode=None) -> (kept, skipped)

四个 reference env 的 figure 布局：

- **PickHighlight** 2 行：第 1 行 = ("all","cube","button","target") 1×4
  collage；第 2 行 = (Panel A 主语义占比 / Panel B 次语义占比 / Panel C
  task_target xy 分布)。数据来源 = visible_objects.json 顶层
  ``selected_target`` 字段。
- **VideoPlaceButton / VideoPlaceOrder** 3 行：第 1 行 / 第 2 行与 PickHighlight
  一致；第 3 行 = 单 panel 满列 swap_pair xy 散点 + 双向箭头：背景 all_candidates
  灰点、前景 swap_target_a/b 端点 + 箭头连接，按 episode 着色（仿
  permanance_inspect 的 _plot_permanence_swaps_panel）。第 3 行数据来源 =
  visible_objects.json 顶层 ``videoplace_swap_pair`` 字段（与 selected_target
  平级独立顶层字段）。
- **VideoRepick** 2 行：第 1 行 = 同款 1×4 collage（最后一列 target 为空）；
  第 2 行 = 单 panel 满列 pickup target xy overlay：背景 spawned_cubes 灰点、
  前景 target_cube_1 按 RGB 着色。数据来源 = visible_objects.json 顶层
  ``videorepick_metadata`` 字段（与 selected_target 解耦）。

数据层（discover / dedup / sidecar 解析）走 reference.py；可视化层（panel
绘制 / figure 拼装）一律落在本文件 —— reference.py 只负责数据产生，
不参与渲染。
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("MPLBACKEND", "Agg")

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import reference as reference_module  # noqa: E402
import xy_common  # noqa: E402


REFERENCE_ENV_IDS: frozenset[str] = frozenset(
    {"PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"}
)

# 3 行布局：第 3 行画 swap_pair 散点 + 双向箭头（仿 permanence）
VIDEOPLACE_ENV_IDS: frozenset[str] = frozenset(
    {"VideoPlaceButton", "VideoPlaceOrder"}
)

# 2 行布局：第 2 行 = selected_target overlay（PickHighlight 无 swap，保持 2 行）
PICKHIGHLIGHT_ROW_ENV_IDS: frozenset[str] = frozenset({"PickHighlight"})

# selected_target 适用的 env 全集（PickHighlight + 两个 VideoPlace），用于
# discover/dedup 时的 env 过滤
TWO_ROW_ENV_IDS: frozenset[str] = VIDEOPLACE_ENV_IDS | PICKHIGHLIGHT_ROW_ENV_IDS

# swap_pair 端点的着色 palette（按 episode 取色）
VIDEOPLACE_SWAP_EPISODE_COLORS: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#bcbd22",
]

# VideoRepick swap_pair 着色 palette（按 swap_index 取色，与
# permanance_inspect.PERMANENCE_SWAP_INDEX_COLORS 视觉风格对齐）
VIDEOREPICK_SWAP_INDEX_COLORS: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
]


_DEFAULT_BASE = Path("/data/hongzefu/robomme_benchmark_cvpr2026-heldoutSeed/runs/replay_videos")
DEFAULT_SEGMENTATION_DIR = _DEFAULT_BASE / "reset_segmentation_pngs"
DEFAULT_OUTPUT_DIR = _DEFAULT_BASE / "inspect-stat" / "xy"


# ---------------------------------------------------------------------------
# 第 2 行 panel：bar / xy 通用工具
# ---------------------------------------------------------------------------


_BAR_DEFAULT_COLOR = "#1f77b4"  # tab:blue


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
    """画分类柱状图（labels 与 counts 等长）。"""
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
        # 顶部留出 ~15% 空间给 annotation
        ax.set_ylim(0, max(1, int(max_count * 1.15) + 1))
    _annotate_bars(ax, counts, total)


# ---------------------------------------------------------------------------
# 第 2 行 panel A/B：每个 env 的占比聚合
# ---------------------------------------------------------------------------


def _bar_specs_for_env(
    env_id: str,
    records: list[reference_module.SelectedTargetRecord],
) -> tuple[tuple[str, list[str], list[int], Optional[list[str]]],
           tuple[str, list[str], list[int], Optional[list[str]]]]:
    """返回 (Panel A spec, Panel B spec)。每个 spec = (title, labels, counts, bar_colors_or_None)。"""

    if env_id == "VideoPlaceButton":
        timing_counter: Counter[str] = Counter()
        index_counter: Counter[int] = Counter()
        for rec in records:
            st = rec.selected_target
            timing_counter[str(st.get("timing", "unknown"))] += 1
            indices = st.get("task_target_indices") or []
            if indices:
                index_counter[int(indices[0])] += 1

        ordered_timing_labels = ["before", "after"]
        timing_labels = ordered_timing_labels + sorted(
            k for k in timing_counter if k not in ordered_timing_labels
        )
        timing_counts = [timing_counter.get(lbl, 0) for lbl in timing_labels]

        max_idx = max(list(index_counter.keys()) + [1])
        index_labels = [f"target_{i}" for i in range(max_idx + 1)]
        index_counts = [index_counter.get(i, 0) for i in range(max_idx + 1)]

        return (
            ("Panel A: timing (target_target_language)", timing_labels, timing_counts, None),
            ("Panel B: target_index", index_labels, index_counts, None),
        )

    if env_id == "VideoPlaceOrder":
        order_counter: Counter[int] = Counter()
        index_counter = Counter()
        for rec in records:
            st = rec.selected_target
            order_position = st.get("order_position")
            if isinstance(order_position, int):
                order_counter[order_position] += 1
            indices = st.get("task_target_indices") or []
            if indices:
                index_counter[int(indices[0])] += 1

        if order_counter:
            order_keys = sorted(order_counter.keys())
            order_labels = [_ordinal_label(k) for k in order_keys]
            order_counts = [order_counter[k] for k in order_keys]
        else:
            order_labels, order_counts = [], []

        if index_counter:
            max_idx = max(index_counter.keys())
            index_labels = [f"target_{i}" for i in range(max_idx + 1)]
            index_counts = [index_counter.get(i, 0) for i in range(max_idx + 1)]
        else:
            index_labels, index_counts = [], []

        return (
            ("Panel A: order_position (which_in_subset)", order_labels, order_counts, None),
            ("Panel B: target_index", index_labels, index_counts, None),
        )

    if env_id == "PickHighlight":
        color_counter: Counter[str] = Counter()
        size_counter: Counter[int] = Counter()
        for rec in records:
            st = rec.selected_target
            colors = st.get("task_target_colors") or []
            for c in colors:
                color_counter[str(c) or "unknown"] += 1
            highlight_count = st.get("highlight_count")
            if isinstance(highlight_count, int):
                size_counter[highlight_count] += 1

        ordered_color_labels = ["red", "green", "blue"]
        color_labels = ordered_color_labels + sorted(
            k for k in color_counter if k not in ordered_color_labels
        )
        color_counts = [color_counter.get(lbl, 0) for lbl in color_labels]
        color_bar_colors = [
            xy_common.CUBE_COLOR_MAP.get(lbl, xy_common.CUBE_COLOR_MAP["unknown"])
            for lbl in color_labels
        ]

        if size_counter:
            size_keys = sorted(size_counter.keys())
            size_labels = [str(k) for k in size_keys]
            size_counts = [size_counter[k] for k in size_keys]
        else:
            size_labels, size_counts = [], []

        return (
            (
                "Panel A: highlighted cube color",
                color_labels,
                color_counts,
                color_bar_colors,
            ),
            ("Panel B: highlight_count (K)", size_labels, size_counts, None),
        )

    raise ValueError(f"Unsupported TWO_ROW env_id: {env_id}")


def _ordinal_label(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 11 -> '11th' ..."""
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ---------------------------------------------------------------------------
# 第 2 行 panel C：task_target xy 分布
# ---------------------------------------------------------------------------


_TASK_TARGET_FG_COLOR = "#ff7f0e"  # tab:orange — 单一前景色，与 cube R/G/B 区分


def _draw_task_target_xy_panel(
    ax: Any,
    records: list[reference_module.SelectedTargetRecord],
) -> None:
    """背景层 = all_candidates 灰点；前景层 = task_target 单一橙色点（不区分 cube 颜色）。"""

    bg_xy: list[tuple[float, float]] = []
    fg_xy: list[tuple[float, float]] = []

    for rec in records:
        st = rec.selected_target
        for cand in st.get("all_candidates", []) or []:
            pos = cand.get("position_xy")
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                bg_xy.append(xy_common._xy_rot_cw_90(float(pos[0]), float(pos[1])))

        for pos in st.get("task_target_positions_xy") or []:
            if not (isinstance(pos, (list, tuple)) and len(pos) >= 2):
                continue
            fg_xy.append(xy_common._xy_rot_cw_90(float(pos[0]), float(pos[1])))

    legend_drawn = False
    if bg_xy:
        ax.scatter(
            [p[0] for p in bg_xy],
            [p[1] for p in bg_xy],
            s=20,
            color="lightgray",
            alpha=0.4,
            edgecolors="none",
            label="all_candidates",
        )
        legend_drawn = True

    if fg_xy:
        ax.scatter(
            [p[0] for p in fg_xy],
            [p[1] for p in fg_xy],
            s=80,
            c=_TASK_TARGET_FG_COLOR,
            edgecolors="black",
            linewidths=0.6,
            label="task_target",
        )
        legend_drawn = True

    ax.set_xlim(-xy_common.XY_LIMIT, xy_common.XY_LIMIT)
    ax.set_ylim(-xy_common.XY_LIMIT, xy_common.XY_LIMIT)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World Y")
    ax.set_ylabel("-World X")
    ax.grid(True, alpha=0.45)
    ax.set_title(
        f"Panel C: task_target XY\nselected_points={len(fg_xy)}"
    )
    if legend_drawn:
        ax.legend(loc="upper right", fontsize=8)


# ---------------------------------------------------------------------------
# 第 3 行 panel（VideoPlaceButton / VideoPlaceOrder）：swap_pair 散点 + 双向箭头
# ---------------------------------------------------------------------------


def _draw_videoplace_swap_pair_panel(
    ax: Any,
    swap_records: list[reference_module.VideoPlaceSwapPairRecord],
    selected_records: list[reference_module.SelectedTargetRecord],
) -> int:
    """背景层 = selected_records 的 all_candidates 灰色点；前景层 = 每个
    has_swap=True 的 swap_record 画 a/b 端点 + 双向箭头（按 episode 着色）。
    返回参与绘制的 swap pair 总数（has_swap=True 的 episode 计数）。"""
    from matplotlib.lines import Line2D

    bg_xy: list[tuple[float, float]] = []
    for rec in selected_records:
        st = rec.selected_target
        for cand in st.get("all_candidates", []) or []:
            pos = cand.get("position_xy")
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                bg_xy.append(xy_common._xy_rot_cw_90(float(pos[0]), float(pos[1])))

    if bg_xy:
        ax.scatter(
            [p[0] for p in bg_xy],
            [p[1] for p in bg_xy],
            s=20,
            color="lightgray",
            alpha=0.4,
            edgecolors="none",
            label="all_candidates",
            zorder=1,
        )

    pair_count = 0
    for rec in swap_records:
        sp = rec.swap_pair
        if not sp.get("has_swap"):
            continue
        a_xy = sp.get("swap_target_a_position_xy")
        b_xy = sp.get("swap_target_b_position_xy")
        if not (
            isinstance(a_xy, (list, tuple)) and len(a_xy) >= 2
            and isinstance(b_xy, (list, tuple)) and len(b_xy) >= 2
        ):
            continue
        color = VIDEOPLACE_SWAP_EPISODE_COLORS[
            rec.episode % len(VIDEOPLACE_SWAP_EPISODE_COLORS)
        ]
        ax_x, ax_y = xy_common._xy_rot_cw_90(float(a_xy[0]), float(a_xy[1]))
        bx_x, bx_y = xy_common._xy_rot_cw_90(float(b_xy[0]), float(b_xy[1]))

        ax.annotate(
            "",
            xy=(bx_x, bx_y), xytext=(ax_x, ax_y),
            arrowprops=dict(
                arrowstyle="<->", color=color, lw=1.4, alpha=0.7,
                shrinkA=4, shrinkB=4,
            ),
            zorder=3,
        )
        ax.scatter(
            [ax_x, bx_x], [ax_y, bx_y],
            s=60, c=color, edgecolors="black", linewidths=0.5,
            alpha=0.9, zorder=4,
        )
        mid_x = (ax_x + bx_x) / 2
        mid_y = (ax_y + bx_y) / 2
        ax.text(
            mid_x, mid_y,
            f"ep{rec.episode}",
            fontsize=6, alpha=0.7,
        )
        pair_count += 1

    ax.set_xlim(-xy_common.XY_LIMIT, xy_common.XY_LIMIT)
    ax.set_ylim(-xy_common.XY_LIMIT, xy_common.XY_LIMIT)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World Y")
    ax.set_ylabel("-World X")
    ax.grid(True, alpha=0.45)

    title = (
        f"Panel D: swap_pair XY (a ↔ b)\n"
        f"episodes_with_swap={pair_count} / total_records={len(swap_records)}"
    )
    ax.set_title(title)

    if pair_count == 0 and not bg_xy:
        ax.text(
            0.0, 0.0,
            "No swap data (only hard difficulty has swap)",
            ha="center", va="center",
        )
    elif pair_count == 0:
        ax.text(
            0.0, 0.0,
            "No swap pairs (all records are easy/medium)",
            ha="center", va="center",
            zorder=5,
        )
    else:
        legend_handles = [
            Line2D(
                [0], [0],
                marker="o", linestyle="", markersize=6,
                markerfacecolor="lightgray", markeredgecolor="none",
                label="all_candidates",
            ),
            Line2D(
                [0], [0],
                marker="o", linestyle="-", markersize=8,
                color=VIDEOPLACE_SWAP_EPISODE_COLORS[0],
                markeredgecolor="black",
                label="swap pair (a ↔ b, color=episode)",
            ),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    return pair_count


# ---------------------------------------------------------------------------
# 2 行 figure 渲染入口
# ---------------------------------------------------------------------------


def _render_two_row_figure(
    output_dir: Path,
    env_id: str,
    points: list[xy_common.VisibleObjectPoint],
    episode_count: int,
    selected_records: list[reference_module.SelectedTargetRecord],
    plt: Any,
) -> Path:
    """对 PickHighlight 渲染 2 行 figure（VideoPlaceButton / VideoPlaceOrder
    走 _render_three_row_figure_videoplace 升到 3 行）。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{env_id}{xy_common.XY_DEFAULT_PNG_SUFFIX}"

    panel_specs = ("all", "cube", "button", "target")
    n_cols = len(panel_specs)

    fig = plt.figure(figsize=(7 * n_cols, 7 * 2))
    gs = fig.add_gridspec(2, n_cols)

    # Row 0: 4 个 visible-objects panel（与现有 collage 完全一致的样式）
    row0_axes = []
    for col_idx, panel_key in enumerate(panel_specs):
        ax = fig.add_subplot(gs[0, col_idx])
        xy_common._plot_panel(ax, panel_key, env_id, points)
        row0_axes.append(ax)

    # Row 1: bar1 / bar2 / xy(span 2)
    bar_a_spec, bar_b_spec = _bar_specs_for_env(env_id, selected_records)
    ax_bar_a = fig.add_subplot(gs[1, 0])
    _draw_categorical_bar(
        ax_bar_a,
        title=bar_a_spec[0],
        labels=bar_a_spec[1],
        counts=bar_a_spec[2],
        bar_colors=bar_a_spec[3],
    )
    ax_bar_b = fig.add_subplot(gs[1, 1])
    _draw_categorical_bar(
        ax_bar_b,
        title=bar_b_spec[0],
        labels=bar_b_spec[1],
        counts=bar_b_spec[2],
        bar_colors=bar_b_spec[3],
    )
    ax_xy = fig.add_subplot(gs[1, 2:n_cols])
    _draw_task_target_xy_panel(ax_xy, selected_records)

    title = (
        f"{env_id} | episodes={episode_count} | points={len(points)} "
        f"| selected_target={len(selected_records)}"
    )
    fig.suptitle(title, fontsize=18)
    return xy_common._save_combined_figure(fig, output_path, plt)


# ---------------------------------------------------------------------------
# 3 行 figure 渲染入口（VideoPlaceButton / VideoPlaceOrder）
# ---------------------------------------------------------------------------


def _render_three_row_figure_videoplace(
    output_dir: Path,
    env_id: str,
    points: list[xy_common.VisibleObjectPoint],
    episode_count: int,
    selected_records: list[reference_module.SelectedTargetRecord],
    swap_records: list[reference_module.VideoPlaceSwapPairRecord],
    plt: Any,
) -> tuple[Path, int]:
    """对 VideoPlaceButton / VideoPlaceOrder 渲染 3 行 figure：

    第 1 行 / 第 2 行 = 与 _render_two_row_figure 完全对齐
    第 3 行 = 单 panel 满列 swap_pair 散点 + 双向箭头

    返回 (output_path, swap_pair_count)。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{env_id}{xy_common.XY_DEFAULT_PNG_SUFFIX}"

    panel_specs = ("all", "cube", "button", "target")
    n_cols = len(panel_specs)

    fig = plt.figure(figsize=(7 * n_cols, 7 * 3))
    gs = fig.add_gridspec(3, n_cols)

    # Row 0: 4 个 visible-objects panel
    for col_idx, panel_key in enumerate(panel_specs):
        ax = fig.add_subplot(gs[0, col_idx])
        xy_common._plot_panel(ax, panel_key, env_id, points)

    # Row 1: bar1 / bar2 / xy(span 2)
    bar_a_spec, bar_b_spec = _bar_specs_for_env(env_id, selected_records)
    ax_bar_a = fig.add_subplot(gs[1, 0])
    _draw_categorical_bar(
        ax_bar_a,
        title=bar_a_spec[0],
        labels=bar_a_spec[1],
        counts=bar_a_spec[2],
        bar_colors=bar_a_spec[3],
    )
    ax_bar_b = fig.add_subplot(gs[1, 1])
    _draw_categorical_bar(
        ax_bar_b,
        title=bar_b_spec[0],
        labels=bar_b_spec[1],
        counts=bar_b_spec[2],
        bar_colors=bar_b_spec[3],
    )
    ax_xy = fig.add_subplot(gs[1, 2:n_cols])
    _draw_task_target_xy_panel(ax_xy, selected_records)

    # Row 2: swap_pair 单 panel 满列
    ax_swap = fig.add_subplot(gs[2, 0:n_cols])
    swap_pair_count = _draw_videoplace_swap_pair_panel(
        ax_swap, swap_records, selected_records
    )

    title = (
        f"{env_id} | episodes={episode_count} | points={len(points)} "
        f"| selected_target={len(selected_records)} "
        f"| swap_records={len(swap_records)} swap_episodes={swap_pair_count}"
    )
    fig.suptitle(title, fontsize=18)
    return xy_common._save_combined_figure(fig, output_path, plt), swap_pair_count


# ---------------------------------------------------------------------------
# VideoRepick 专属：第 2 行 pickup target xy overlay panel
# ---------------------------------------------------------------------------


def _draw_videorepick_pickup_target_xy_panel(
    ax: Any,
    records: list[reference_module.VideoRepickRecord],
) -> None:
    """背景层 = all_candidates 灰点；前景层 = target_cube_1 按 cube 颜色着色。"""

    bg_xy: list[tuple[float, float]] = []
    fg_by_color: dict[str, list[tuple[float, float]]] = {}
    color_counts: Counter[str] = Counter()

    for rec in records:
        md = rec.metadata
        for cand in md.get("all_candidates", []) or []:
            pos = cand.get("position_xy")
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                bg_xy.append(xy_common._xy_rot_cw_90(float(pos[0]), float(pos[1])))

        target_xy = md.get("target_cube_1_position_xy")
        color_name = str(md.get("target_cube_1_color", "unknown")) or "unknown"
        if isinstance(target_xy, (list, tuple)) and len(target_xy) >= 2:
            fg_by_color.setdefault(color_name, []).append(
                xy_common._xy_rot_cw_90(float(target_xy[0]), float(target_xy[1]))
            )
            color_counts[color_name] += 1

    if bg_xy:
        ax.scatter(
            [p[0] for p in bg_xy],
            [p[1] for p in bg_xy],
            s=20,
            color="lightgray",
            alpha=0.4,
            edgecolors="none",
            label="spawned_cubes",
        )

    ordered_colors = ["red", "green", "blue"] + sorted(
        c for c in fg_by_color if c not in ("red", "green", "blue")
    )
    for color_name in ordered_colors:
        pts = fg_by_color.get(color_name)
        if not pts:
            continue
        bar_color = xy_common.CUBE_COLOR_MAP.get(
            color_name, xy_common.CUBE_COLOR_MAP["unknown"]
        )
        ax.scatter(
            [p[0] for p in pts],
            [p[1] for p in pts],
            s=80,
            c=bar_color,
            edgecolors="black",
            linewidths=0.6,
            label=f"target_cube_1={color_name} (n={len(pts)})",
        )

    ax.set_xlim(-xy_common.XY_LIMIT, xy_common.XY_LIMIT)
    ax.set_ylim(-xy_common.XY_LIMIT, xy_common.XY_LIMIT)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World Y")
    ax.set_ylabel("-World X")
    ax.grid(True, alpha=0.45)

    total_target = sum(color_counts.values())
    by_color_text = " ".join(
        f"{c}={color_counts.get(c, 0)}" for c in ordered_colors if color_counts.get(c)
    )
    title = f"Pickup target XY\nepisodes={total_target}"
    if by_color_text:
        title = f"{title} | {by_color_text}"
    ax.set_title(title)
    if bg_xy or any(fg_by_color.values()):
        ax.legend(loc="upper right", fontsize=8)


def _render_two_row_figure_videorepick(
    output_dir: Path,
    points: list[xy_common.VisibleObjectPoint],
    episode_count: int,
    videorepick_records: list[reference_module.VideoRepickRecord],
    plt: Any,
) -> Path:
    """VideoRepick 2 行 figure：第 1 行复用 1×3 collage，第 2 行单 panel 满列。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"VideoRepick{xy_common.XY_DEFAULT_PNG_SUFFIX}"

    panel_specs = ("all", "cube", "button")
    n_cols = len(panel_specs)

    fig = plt.figure(figsize=(7 * n_cols, 7 * 2))
    gs = fig.add_gridspec(2, n_cols)

    for col_idx, panel_key in enumerate(panel_specs):
        ax = fig.add_subplot(gs[0, col_idx])
        xy_common._plot_panel(ax, panel_key, "VideoRepick", points)

    ax_xy = fig.add_subplot(gs[1, 0:n_cols])
    _draw_videorepick_pickup_target_xy_panel(ax_xy, videorepick_records)

    title = (
        f"VideoRepick | episodes={episode_count} | points={len(points)} "
        f"| videorepick_records={len(videorepick_records)}"
    )
    fig.suptitle(title, fontsize=18)
    return xy_common._save_combined_figure(fig, output_path, plt)


# ---------------------------------------------------------------------------
# VideoRepick 专属：第 3 行 swap_pair 散点 + 双向箭头 panel
# ---------------------------------------------------------------------------


def _draw_videorepick_swap_pair_panel(
    ax: Any,
    records: list[reference_module.VideoRepickRecord],
) -> int:
    """背景层 = 所有 record 的 all_candidates 灰点；前景层 = 每个 record 的
    swap_pairs：按 swap_index % 4 着色 → 双向 ``<->`` 箭头 + 端点 scatter +
    中点 ``ep{ep}#s{idx}`` 文字。空数据显示 'No swap data'。

    视觉风格 mirror permanance_inspect._plot_permanence_swaps_panel（按
    swap_index 着色），而非 VideoPlace 的 episode 着色——与 VideoUnmaskSwap /
    ButtonUnmaskSwap 的 swap panel 一致。
    """
    from matplotlib.lines import Line2D

    bg_xy: list[tuple[float, float]] = []
    for rec in records:
        for cand in rec.metadata.get("all_candidates", []) or []:
            pos = cand.get("position_xy")
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                bg_xy.append(xy_common._xy_rot_cw_90(float(pos[0]), float(pos[1])))

    if bg_xy:
        ax.scatter(
            [p[0] for p in bg_xy],
            [p[1] for p in bg_xy],
            s=20,
            color="lightgray",
            alpha=0.4,
            edgecolors="none",
            label="all_candidates",
            zorder=1,
        )

    pair_count = 0
    seen_swap_indices: set[int] = set()
    for rec in records:
        swap_pairs = rec.metadata.get("swap_pairs") or []
        for pair in swap_pairs:
            if not isinstance(pair, dict):
                continue
            swap_idx = int(pair.get("swap_index", 0))
            seen_swap_indices.add(swap_idx)
            color = VIDEOREPICK_SWAP_INDEX_COLORS[
                swap_idx % len(VIDEOREPICK_SWAP_INDEX_COLORS)
            ]
            a_xy = pair.get("cube_a_position_xy")
            b_xy = pair.get("cube_b_position_xy")
            if not (
                isinstance(a_xy, (list, tuple)) and len(a_xy) >= 2
                and isinstance(b_xy, (list, tuple)) and len(b_xy) >= 2
            ):
                continue
            ax_x, ax_y = xy_common._xy_rot_cw_90(float(a_xy[0]), float(a_xy[1]))
            bx_x, bx_y = xy_common._xy_rot_cw_90(float(b_xy[0]), float(b_xy[1]))

            ax.annotate(
                "",
                xy=(bx_x, bx_y), xytext=(ax_x, ax_y),
                arrowprops=dict(
                    arrowstyle="<->", color=color, lw=1.4, alpha=0.7,
                    shrinkA=4, shrinkB=4,
                ),
                zorder=3,
            )
            ax.scatter(
                [ax_x, bx_x], [ax_y, bx_y],
                s=60, c=color, edgecolors="black", linewidths=0.5,
                alpha=0.9, zorder=4,
            )
            mid_x = (ax_x + bx_x) / 2
            mid_y = (ax_y + bx_y) / 2
            ax.text(
                mid_x, mid_y,
                f"ep{rec.episode}#s{swap_idx}",
                fontsize=6, alpha=0.7,
            )
            pair_count += 1

    ax.set_xlim(-xy_common.XY_LIMIT, xy_common.XY_LIMIT)
    ax.set_ylim(-xy_common.XY_LIMIT, xy_common.XY_LIMIT)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World Y")
    ax.set_ylabel("-World X")
    ax.grid(True, alpha=0.45)
    ax.set_title(
        f"VideoRepick swaps (Rotated XY)\n"
        f"pairs={pair_count} / records={len(records)}"
    )

    if pair_count == 0 and not bg_xy:
        ax.text(0.0, 0.0, "No swap data", ha="center", va="center")
    elif pair_count == 0:
        ax.text(
            0.0, 0.0,
            "No swap pairs (all records are hard difficulty)",
            ha="center", va="center", zorder=5,
        )
    else:
        legend_handles: list[Line2D] = []
        for swap_idx in sorted(seen_swap_indices):
            color = VIDEOREPICK_SWAP_INDEX_COLORS[
                swap_idx % len(VIDEOREPICK_SWAP_INDEX_COLORS)
            ]
            legend_handles.append(
                Line2D(
                    [0], [0],
                    marker="o", linestyle="-", color=color, markersize=8,
                    label=f"swap #{swap_idx}",
                )
            )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7)

    return pair_count


def _render_three_row_figure_videorepick(
    output_dir: Path,
    points: list[xy_common.VisibleObjectPoint],
    episode_count: int,
    videorepick_records: list[reference_module.VideoRepickRecord],
    plt: Any,
) -> tuple[Path, int]:
    """VideoRepick 3 行 figure：第 1 行 1×3 collage（all/cube/button），第 2 行
    pickup target xy overlay 单 panel，第 3 行 swap_pair 散点 + 双向箭头单
    panel。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"VideoRepick{xy_common.XY_DEFAULT_PNG_SUFFIX}"

    panel_specs = ("all", "cube", "button")
    n_cols = len(panel_specs)

    fig = plt.figure(figsize=(7 * n_cols, 7 * 3))
    gs = fig.add_gridspec(3, n_cols)

    for col_idx, panel_key in enumerate(panel_specs):
        ax = fig.add_subplot(gs[0, col_idx])
        xy_common._plot_panel(ax, panel_key, "VideoRepick", points)

    ax_xy = fig.add_subplot(gs[1, 0:n_cols])
    _draw_videorepick_pickup_target_xy_panel(ax_xy, videorepick_records)

    ax_swap = fig.add_subplot(gs[2, 0:n_cols])
    swap_pair_count = _draw_videorepick_swap_pair_panel(ax_swap, videorepick_records)

    title = (
        f"VideoRepick | episodes={episode_count} | points={len(points)} "
        f"| videorepick_records={len(videorepick_records)} "
        f"| swap_pairs={swap_pair_count}"
    )
    fig.suptitle(title, fontsize=18)
    return xy_common._save_combined_figure(fig, output_path, plt), swap_pair_count


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
    """渲染 Reference 套件每个 env 的 xy collage。

    Parameters
    ----------
    segmentation_dir:
        包含各 episode 子目录（内有 visible_objects.json）的根目录。
    output_dir:
        最终 PNG 写入位置（通常是 ``inspect-stat/xy``）。
    env_id:
        指定只处理哪个 env；``None`` = 处理 4 个全部。传入非 reference env
        时静默返回空列表。
    difficulty_by_env_episode:
        来自 distribution pipeline 的 (env, episode) -> difficulty 映射。
        Reference 套件不做 difficulty 拆分，但保留以与其它 suite 接口一致
        （也是 _build_points_from_files 的必要输入）。

    Returns
    -------
    (kept, skipped) : tuple[list[xy_common._VisibleObjectsFile], ...]
    """
    segmentation_dir = Path(segmentation_dir)
    output_dir = Path(output_dir)

    if env_id is not None and env_id not in REFERENCE_ENV_IDS:
        return [], []

    files = xy_common._discover_visible_object_files(segmentation_dir)
    files = [
        entry
        for entry in files
        if entry.env_id in REFERENCE_ENV_IDS
        and (env_id is None or entry.env_id == env_id)
    ]

    kept, skipped = xy_common._dedup_visible_object_files(files)
    if not kept:
        env_part = f" for env_id={env_id!r}" if env_id else ""
        print(
            f"[Reference-inspect] No visible_objects.json found "
            f"under {segmentation_dir}{env_part}."
        )
        return kept, skipped

    points_by_env, skipped_objects, episode_counts, _ = (
        xy_common._build_points_from_files(kept, difficulty_by_env_episode or {})
    )

    # --- 收集 selected_target 记录（仅 TWO_ROW envs 用得到）---
    selected_records_by_env: dict[str, list[reference_module.SelectedTargetRecord]] = {}
    raw_records = reference_module.discover_selected_target_records(
        segmentation_dir, env_filter=env_id
    )
    raw_records = [
        rec for rec in raw_records if rec.env_id in TWO_ROW_ENV_IDS
    ]
    kept_records, skipped_records = reference_module.dedup_selected_target_records(raw_records)
    for rec in kept_records:
        selected_records_by_env.setdefault(rec.env_id, []).append(rec)
    print(
        f"  Selected-target records: kept={len(kept_records)} "
        f"skipped(dup)={len(skipped_records)}"
    )

    # --- 收集 VideoRepick records（与 selected_target 解耦的独立路径）---
    videorepick_records_by_env: dict[
        str, list[reference_module.VideoRepickRecord]
    ] = {}
    raw_vr = reference_module.discover_videorepick_records(
        segmentation_dir, env_filter=env_id
    )
    kept_vr, skipped_vr = reference_module.dedup_videorepick_records(raw_vr)
    for rec in kept_vr:
        videorepick_records_by_env.setdefault(rec.env_id, []).append(rec)
    print(
        f"  VideoRepick records: kept={len(kept_vr)} "
        f"skipped(dup)={len(skipped_vr)}"
    )

    # --- 收集 VideoPlace swap_pair records（与 selected_target 解耦的独立路径）---
    swap_records_by_env: dict[
        str, list[reference_module.VideoPlaceSwapPairRecord]
    ] = {}
    raw_swap = reference_module.discover_videoplace_swap_pair_records(
        segmentation_dir, env_filter=env_id
    )
    kept_swap, skipped_swap = reference_module.dedup_videoplace_swap_pair_records(
        raw_swap
    )
    for rec in kept_swap:
        swap_records_by_env.setdefault(rec.env_id, []).append(rec)
    print(
        f"  VideoPlace swap-pair records: kept={len(kept_swap)} "
        f"skipped(dup)={len(skipped_swap)}"
    )

    plt = xy_common._get_pyplot(show=False)
    for eid in sorted(points_by_env):
        points = points_by_env[eid]
        if eid in VIDEOPLACE_ENV_IDS:
            selected_records = selected_records_by_env.get(eid, [])
            swap_records = swap_records_by_env.get(eid, [])
            _, swap_pair_count = _render_three_row_figure_videoplace(
                output_dir,
                eid,
                points,
                episode_counts.get(eid, 0),
                selected_records,
                swap_records,
                plt,
            )
            counts = xy_common._category_counts(points)
            print(
                f"  {eid}: episodes={episode_counts.get(eid, 0)} "
                f"points={len(points)} "
                f"selected_records={len(selected_records)} "
                f"swap_records={len(swap_records)} "
                f"swap_pairs={swap_pair_count} "
                f"cube={counts.get('cube', 0)} "
                f"button={counts.get('button', 0)} "
                f"target={counts.get('target', 0)} "
                f"other={counts.get('other', 0)}"
            )
        elif eid in PICKHIGHLIGHT_ROW_ENV_IDS:
            selected_records = selected_records_by_env.get(eid, [])
            _render_two_row_figure(
                output_dir,
                eid,
                points,
                episode_counts.get(eid, 0),
                selected_records,
                plt,
            )
            counts = xy_common._category_counts(points)
            print(
                f"  {eid}: episodes={episode_counts.get(eid, 0)} "
                f"points={len(points)} "
                f"selected_records={len(selected_records)} "
                f"cube={counts.get('cube', 0)} "
                f"button={counts.get('button', 0)} "
                f"target={counts.get('target', 0)} "
                f"other={counts.get('other', 0)}"
            )
        elif eid == reference_module.VIDEOREPICK_ENV_ID:
            vr_records = videorepick_records_by_env.get(eid, [])
            has_swap_record = any(
                int(rec.metadata.get("swap_times", 0) or 0) >= 1
                for rec in vr_records
            )
            if has_swap_record:
                _, swap_pair_count = _render_three_row_figure_videorepick(
                    output_dir,
                    points,
                    episode_counts.get(eid, 0),
                    vr_records,
                    plt,
                )
            else:
                _render_two_row_figure_videorepick(
                    output_dir,
                    points,
                    episode_counts.get(eid, 0),
                    vr_records,
                    plt,
                )
                swap_pair_count = 0
            counts = xy_common._category_counts(points)
            print(
                f"  {eid}: episodes={episode_counts.get(eid, 0)} "
                f"points={len(points)} "
                f"videorepick_records={len(vr_records)} "
                f"swap_pairs={swap_pair_count} "
                f"cube={counts.get('cube', 0)} "
                f"button={counts.get('button', 0)} "
                f"target={counts.get('target', 0)} "
                f"other={counts.get('other', 0)}"
            )
        else:
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
        f"[Reference-inspect] envs={len(points_by_env)} "
        f"kept={len(kept)} skipped={len(skipped)} "
        f"skipped_objects={skipped_objects.get('objects', 0)}"
    )
    return kept, skipped


_VALID_ENV_IDS = sorted(REFERENCE_ENV_IDS)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reference 套件专用 xy 可视化：扫描 visible_objects.json，为四个 "
            "reference env 各渲染一张 PNG。第 1 行恒为 visible-objects 1×4 "
            "collage；后续行因 env 而异：PickHighlight 走 2 行（第 2 行 "
            "selected_target 三 panel）；VideoPlaceButton / VideoPlaceOrder "
            "走 3 行（第 3 行新增 swap_pair 散点 + 双向箭头，数据来自顶层 "
            "videoplace_swap_pair 字段）；VideoRepick 走 2 行（第 2 行 "
            "pickup target xy overlay 单 panel 满列）。"
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
            f"只处理指定的 reference env，不传则处理全部四个。"
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
