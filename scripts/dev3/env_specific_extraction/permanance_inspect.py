"""Permanence 套件（VideoUnmask / VideoUnmaskSwap / ButtonUnmask / ButtonUnmaskSwap）
专用可视化入口。

可被 inspect_stat.py 调用，也可独立运行。对外只暴露一个公开接口：
    visualize(segmentation_dir, output_dir, env_id=None)

调用一次 visualize() 直接产出完整的 3 行 PNG 到 inspect-stat/xy/{env_id}_xy.png：
- 第 1 行：visible-objects 4 个面板（与 inspect_stat 普通 xy 图保持一致）
- 第 2 行：permanence cubes + permanence swaps 双面板
- 第 3 行：first / second pickup bin 双面板（颜色=被遮挡的 cube 颜色）

permanance_inspect/ 目录不再被创建——`output_dir` 仅作 anchor，用来定位
inspect-stat 根目录下的 xy/ 子目录。

数据层（discover / dedup / sidecar 解析）走 permanence.py；
可视化层（panel 绘制 / 3 行 figure 拼装）一律落在本文件——
permanence.py 只负责数据产生，不参与渲染。
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

os.environ.setdefault("MPLBACKEND", "Agg")

# permanence.py / xy_common.py 都在同一目录下，直接 sys.path 注入
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import permanence as permanence_module  # noqa: E402
import xy_common  # noqa: E402

# ---------------------------------------------------------------------------
# 默认路径
# ---------------------------------------------------------------------------

_DEFAULT_BASE = Path("/data/hongzefu/robomme_benchmark_cvpr2026-heldoutSeed/runs/replay_videos")
DEFAULT_SEGMENTATION_DIR = _DEFAULT_BASE / "reset_segmentation_pngs"
DEFAULT_OUTPUT_DIR = _DEFAULT_BASE / "inspect-stat" / "permanance_inspect"


# ---------------------------------------------------------------------------
# 可视化常量（permanence 套件 visualization 的单一事实来源）
# ---------------------------------------------------------------------------

_PERMANENCE_PANEL_LIMIT = 0.3

CUBE_COLOR_MAP: dict[str, str] = {
    "red": "#d62728",
    "green": "#2ca02c",
    "blue": "#1f77b4",
    "unknown": "#7f7f7f",
}

PERMANENCE_SWAP_INDEX_COLORS: list[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
]


# ---------------------------------------------------------------------------
# 内部：axes-level 几何与轴样式
# ---------------------------------------------------------------------------


def _xy_rot_cw_90(x_pos: float, y_pos: float) -> tuple[float, float]:
    """俯视图顺时针旋转 90°：显示 (y, -x)，与 inspect_stat 一致。"""
    return y_pos, -x_pos


def _prepare_panel_axis(ax: Any, title: str, point_count: int) -> None:
    ax.set_xlim(-_PERMANENCE_PANEL_LIMIT, _PERMANENCE_PANEL_LIMIT)
    ax.set_ylim(-_PERMANENCE_PANEL_LIMIT, _PERMANENCE_PANEL_LIMIT)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World Y")
    ax.set_ylabel("-World X")
    ax.grid(True, alpha=0.45)
    ax.set_title(f"{title}\npoints={point_count}")
    # 3 行 figure 用 sharex/sharey=True，matplotlib 默认会隐藏非底行/
    # 非首列子图的 tick labels。permanence 视图要求每个 2D 散点 panel
    # 都显示自己的世界坐标，这里显式打开。
    ax.tick_params(axis="x", labelbottom=True)
    ax.tick_params(axis="y", labelleft=True)


def _unshare_axes(ax: Any) -> None:
    """脱离 row-shared ±0.3 浮点 xlim/ylim 共享组，并重开 autoscale。

    pair-freq heatmap 与 1D 分布柱状图都需要整数刻度 / 自定义 limit，
    与 row 0/1/2 左侧 panel 共享的世界坐标轴不兼容。matplotlib 3.6+ 的
    get_shared_*_axes() 返回只读 GrouperView，必须走底层 _shared_axes
    Grouper 才能 remove。共享组里其他 panel 已显式 set_xlim/ylim 关闭了
    autoscale，移除后必须重开，否则 imshow / bar 不会撑开 limits。"""
    for axis_name in ("x", "y"):
        try:
            ax._shared_axes[axis_name].remove(ax)
        except (KeyError, ValueError):
            pass
    ax.set_autoscalex_on(True)
    ax.set_autoscaley_on(True)


# ---------------------------------------------------------------------------
# 内部：permanence cubes / swaps 面板渲染
# ---------------------------------------------------------------------------


def _plot_permanence_cubes_panel(
    ax: Any,
    env_id: str,
    files: Iterable[Any],
) -> int:
    """渲染 cube reset 位置散点：颜色按 cube 颜色（不区分 difficulty），
    并用细线把 cube 连到对应的 bin。无数据时显示 'No cube data'。"""
    from matplotlib.lines import Line2D

    files_list = list(files or [])
    plotted = 0
    seen_colors: set[str] = set()

    for entry in files_list:
        for cube in entry.payload.get("cubes", []):
            color_name = cube.get("color_name", "unknown")
            seen_colors.add(color_name)
            color = CUBE_COLOR_MAP.get(color_name, CUBE_COLOR_MAP["unknown"])
            pos = cube.get("position_xy") or [0.0, 0.0]
            bin_pos = cube.get("bin_position_xy") or pos
            x, y = _xy_rot_cw_90(float(pos[0]), float(pos[1]))
            bx, by = _xy_rot_cw_90(float(bin_pos[0]), float(bin_pos[1]))
            ax.plot([x, bx], [y, by], color=color, alpha=0.25, linewidth=0.8)
            ax.scatter(
                x, y,
                s=70, alpha=0.85, c=color, marker="o",
                edgecolors="black", linewidths=0.5,
            )
            bin_idx = cube.get("bin_index")
            if bin_idx is not None:
                ax.text(
                    x + 0.005, y + 0.005,
                    f"ep{entry.episode}/b{bin_idx}",
                    fontsize=6, alpha=0.6,
                )
            plotted += 1

    if plotted:
        legend_handles: list[Line2D] = []
        for color_name in sorted(seen_colors):
            color = CUBE_COLOR_MAP.get(color_name, CUBE_COLOR_MAP["unknown"])
            legend_handles.append(
                Line2D(
                    [0], [0],
                    marker="o", linestyle="", markersize=8,
                    markerfacecolor=color, markeredgecolor="black",
                    label=f"cube_{color_name}",
                )
            )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7)
    else:
        ax.text(0.0, 0.0, "No cube data", ha="center", va="center")

    _prepare_panel_axis(ax, "Permanence cubes (Rotated XY)", plotted)
    return plotted


def _plot_permanence_swaps_panel(
    ax: Any,
    env_id: str,
    files: Iterable[Any],
) -> int:
    """渲染 swap pair 双向箭头：每对 (bin_a, bin_b) 按 swap_index 着色，
    bin 全集画成淡灰色背景点。非 Swap env 或无 swap_pairs 时显示
    'No swap data'。"""
    from matplotlib.lines import Line2D

    files_list = list(files or [])
    is_swap_env = env_id in permanence_module.SWAP_ENV_IDS

    pair_count = 0
    seen_swap_indices: set[int] = set()

    if is_swap_env:
        for entry in files_list:
            for bin_info in entry.payload.get("bins", []):
                pos = bin_info.get("position_xy") or [0.0, 0.0]
                bx, by = _xy_rot_cw_90(float(pos[0]), float(pos[1]))
                ax.scatter(bx, by, s=20, color="lightgray", alpha=0.4, zorder=1)

            for pair in entry.payload.get("swap_pairs", []):
                swap_idx = int(pair.get("swap_index", 0))
                seen_swap_indices.add(swap_idx)
                color = PERMANENCE_SWAP_INDEX_COLORS[
                    swap_idx % len(PERMANENCE_SWAP_INDEX_COLORS)
                ]
                a_xy = pair.get("bin_a_position_xy") or [0.0, 0.0]
                b_xy = pair.get("bin_b_position_xy") or [0.0, 0.0]
                ax_x, ax_y = _xy_rot_cw_90(float(a_xy[0]), float(a_xy[1]))
                bx_x, bx_y = _xy_rot_cw_90(float(b_xy[0]), float(b_xy[1]))

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
                    f"ep{entry.episode}#s{swap_idx}",
                    fontsize=6, alpha=0.7,
                )
                pair_count += 1

    if pair_count:
        legend_handles: list[Line2D] = []
        for swap_idx in sorted(seen_swap_indices):
            color = PERMANENCE_SWAP_INDEX_COLORS[
                swap_idx % len(PERMANENCE_SWAP_INDEX_COLORS)
            ]
            legend_handles.append(
                Line2D(
                    [0], [0],
                    marker="o", linestyle="-", color=color, markersize=8,
                    label=f"swap #{swap_idx}",
                )
            )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7)
    else:
        ax.text(0.0, 0.0, "No swap data", ha="center", va="center")

    _prepare_panel_axis(ax, "Permanence swaps (Rotated XY)", pair_count)
    return pair_count


# ---------------------------------------------------------------------------
# 内部：first / second pickup bin 散点（按 cube 颜色着色）
# ---------------------------------------------------------------------------


_PICKUP_PANEL_TITLES = {
    0: "First pickup bin (Rotated XY)",
    1: "Second pickup bin (Rotated XY)",
}


def _plot_pickup_bin_panel(
    ax: Any,
    env_id: str,
    files: Iterable[Any],
    pickup_index: int,
) -> int:
    """渲染第 (pickup_index+1) 次 pickup 的 bin 位置：颜色=被遮挡 cube 的颜色
    （不区分 difficulty）。pickup 顺序与 cubes_payload 顺序对齐——非 swap env 是
    spawned_bins[i]，swap env 是 selected_bins[i]，两者在 permanence.py 写入
    sidecar 时都按 cube/bin pair 顺序展开。无数据时显示 'No pickup data'。"""
    from matplotlib.lines import Line2D

    files_list = list(files or [])
    plotted = 0
    seen_colors: set[str] = set()

    for entry in files_list:
        cubes = entry.payload.get("cubes") or []
        if pickup_index >= len(cubes):
            continue
        cube = cubes[pickup_index]
        if not isinstance(cube, dict):
            continue
        bin_pos = cube.get("bin_position_xy")
        if bin_pos is None or len(bin_pos) < 2:
            continue

        color_name = cube.get("color_name", "unknown")
        seen_colors.add(color_name)
        color = CUBE_COLOR_MAP.get(color_name, CUBE_COLOR_MAP["unknown"])

        bx, by = _xy_rot_cw_90(float(bin_pos[0]), float(bin_pos[1]))
        ax.scatter(
            bx, by,
            s=70, alpha=0.85, c=color, marker="o",
            edgecolors="black", linewidths=0.5,
        )
        ax.text(
            bx + 0.005, by + 0.005,
            f"ep{entry.episode}",
            fontsize=6, alpha=0.6,
        )
        plotted += 1

    if plotted:
        legend_handles: list[Line2D] = []
        for color_name in sorted(seen_colors):
            color = CUBE_COLOR_MAP.get(color_name, CUBE_COLOR_MAP["unknown"])
            legend_handles.append(
                Line2D(
                    [0], [0],
                    marker="o", linestyle="", markersize=8,
                    markerfacecolor=color, markeredgecolor="black",
                    label=f"cube_{color_name}",
                )
            )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7)
    else:
        ax.text(0.0, 0.0, "No pickup data", ha="center", va="center")

    title = _PICKUP_PANEL_TITLES.get(
        pickup_index, f"Pickup #{pickup_index + 1} bin (Rotated XY)"
    )
    _prepare_panel_axis(ax, title, plotted)
    return plotted


# ---------------------------------------------------------------------------
# 内部：first / second pickup bin_index 1D 分布（不分 cube 颜色）
# ---------------------------------------------------------------------------


_DISTRIBUTION_PANEL_TITLES = {
    0: "First pickup bin selection (counts)",
    1: "Second pickup bin selection (counts)",
}


def _plot_pickup_bin_selection_distribution_panel(
    ax: Any,
    env_id: str,
    files: Iterable[Any],
    pickup_index: int,
) -> int:
    """渲染第 (pickup_index+1) 次 pickup 的 bin_index 1D 分布柱状图：
    X = bin_index（bin_0 .. bin_(N-1)），Y = episode 计数，纯色柱不堆叠。
    无数据时显示 'No pickup data'。"""
    from collections import Counter
    import matplotlib.ticker as mticker

    _unshare_axes(ax)
    # share group 里 row 0/1 的 "bin" panel 把 yticks 设成了字符串
    # ("bin_0".."bin_n")，share=True 时会同步过来。_unshare_axes 只移除
    # share，不清 ticks 本身。这里显式把 Y 轴 locator/formatter 重置回
    # 数值整数 ticks，否则计数轴会显示成 bin_* 字符串挤在 0 附近。
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

    files_list = list(files or [])
    counts: Counter = Counter()
    max_bin_idx = -1

    for entry in files_list:
        bins_len = len(entry.payload.get("bins", []) or [])
        if bins_len > 0:
            max_bin_idx = max(max_bin_idx, bins_len - 1)
        cubes = entry.payload.get("cubes") or []
        if pickup_index >= len(cubes):
            continue
        cube = cubes[pickup_index]
        if not isinstance(cube, dict):
            continue
        bin_idx = cube.get("bin_index")
        if bin_idx is None:
            continue
        bin_idx_int = int(bin_idx)
        counts[bin_idx_int] += 1
        max_bin_idx = max(max_bin_idx, bin_idx_int)

    total = sum(counts.values())
    title = _DISTRIBUTION_PANEL_TITLES.get(
        pickup_index, f"Pickup #{pickup_index + 1} bin selection (counts)"
    )

    if total == 0 or max_bin_idx < 0:
        ax.set_title(f"{title}\nepisodes=0")
        ax.text(0.5, 0.5, "No pickup data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return 0

    n_bins = max_bin_idx + 1
    xs = list(range(n_bins))
    ys = [counts.get(i, 0) for i in xs]
    y_max = max(ys)

    ax.bar(xs, ys, color="#1f77b4", edgecolor="black", linewidth=0.5)
    for x, y in zip(xs, ys):
        if y > 0:
            ax.text(x, y, str(y), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"bin_{i}" for i in xs])
    # row-0/1/2 share group 里其他 panel 已让非首列子图隐藏 tick label，
    # 移除 share 后还要显式打开。
    ax.tick_params(axis="x", labelbottom=True)
    ax.tick_params(axis="y", labelleft=True)
    ax.set_xlim(-0.5, n_bins - 0.5)
    ax.set_ylim(0, y_max * 1.15 if y_max > 0 else 1)
    ax.set_xlabel("bin index")
    ax.set_ylabel("episode count")
    ax.grid(True, axis="y", alpha=0.45)
    ax.set_title(f"{title}\nepisodes={total}")
    return total


# ---------------------------------------------------------------------------
# 内部：pair frequency heatmap（仅 swap env）
# ---------------------------------------------------------------------------


def _plot_pair_frequency_panel(
    ax: Any,
    env_id: str,
    files: Iterable[Any],
    bin_count: int,
    plt_module: Any,
) -> int:
    """渲染 bin 对频次 heatmap (bin_count × bin_count)，仅统计 episode bin
    数恰好等于 ``bin_count`` 的 swap_pairs。非 swap env 或无匹配 episode
    时降级到提示文字。返回参与统计的 swap pair 总数。"""
    from collections import Counter

    _unshare_axes(ax)

    title = f"Pair freq ({bin_count}-bin)"

    if env_id not in permanence_module.SWAP_ENV_IDS:
        ax.set_title(title)
        ax.text(0.5, 0.5, "Not a swap env", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return 0

    matched = [
        e for e in (files or [])
        if len(e.payload.get("bins", [])) == bin_count
    ]
    n_eps = len(matched)

    counter: Counter = Counter()
    for entry in matched:
        for pair in entry.payload.get("swap_pairs", []):
            a = pair.get("bin_a_index")
            b = pair.get("bin_b_index")
            if a is None or b is None:
                continue
            if not (0 <= int(a) < bin_count and 0 <= int(b) < bin_count):
                continue
            i, j = sorted([int(a), int(b)])
            counter[(i, j)] += 1

    total = sum(counter.values())

    if n_eps == 0:
        ax.set_title(f"{title}\nepisodes=0 | swaps=0")
        ax.text(0.5, 0.5, f"No {bin_count}-bin episodes", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return 0

    mat = [[0] * bin_count for _ in range(bin_count)]
    for (i, j), c in counter.items():
        mat[i][j] = c
        mat[j][i] = c

    im = ax.imshow(mat, cmap="YlOrRd", origin="lower")
    max_val = max((max(row) for row in mat), default=0)
    threshold = max_val * 0.6 if max_val > 0 else 1
    for i in range(bin_count):
        for j in range(bin_count):
            v = mat[i][j]
            if i == j:
                ax.text(j, i, "—", ha="center", va="center",
                        color="gray", fontsize=10)
            else:
                color = "white" if v > threshold else "black"
                ax.text(j, i, str(v), ha="center", va="center",
                        color=color, fontsize=10)

    ax.set_xticks(range(bin_count))
    ax.set_yticks(range(bin_count))
    ax.set_xticklabels([f"bin_{i}" for i in range(bin_count)])
    ax.set_yticklabels([f"bin_{i}" for i in range(bin_count)])
    # share group 里其他 panel（左侧 cubes/swaps）会让非首列子图的 y tick label
    # 被自动隐藏；移除 share 之后还要显式打开，否则 heatmap 只剩 x 轴。
    ax.tick_params(axis="x", labelbottom=True)
    ax.tick_params(axis="y", labelleft=True)
    ax.set_xlim(-0.5, bin_count - 0.5)
    ax.set_ylim(-0.5, bin_count - 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title}\nepisodes={n_eps} | swaps={total}")
    plt_module.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return total


# ---------------------------------------------------------------------------
# 内部：渲染 3 行 figure
# ---------------------------------------------------------------------------


def _render_three_row_figure(
    xy_dir: Path,
    env_id: str,
    points: list,
    perm_files: list,
    episode_count: int,
    plt,
) -> Path:
    """生成 3 行布局：第 1 行 visible-objects；第 2 行 permanence cubes + swaps
    （swap env 还附加 pair-freq heatmap）；第 3 行 first / second pickup bin。"""
    import numpy as np

    panel_specs = xy_common._panel_specs_for_env(env_id)
    n_top = len(panel_specs)
    # 不要用 sharex/sharey=True：row-0 col-3 的 "bin" panel 用字符串
    # ticks ("bin_0".."bin_n") + ax._sharex 引用会把 ticks sync 到整个
    # share group，导致其它 2D 散点 panel 的数值刻度被覆盖成 "bin_0"
    # 单 tick。本视图每个 panel 都显式 set_xlim/set_ylim，share 本来
    # 也没有实际作用，直接 sharex=False/sharey=False 让 axes 从一开始
    # 就独立。
    fig, axes = plt.subplots(
        3,
        n_top,
        figsize=(7 * n_top, 21),
        sharex=False,
        sharey=False,
    )
    axes = np.atleast_2d(axes)

    fig.suptitle(
        f"{env_id} | episodes={episode_count} | points={len(points)} | "
        f"permanence_episodes={len(perm_files)}",
        fontsize=18,
    )

    is_swap_env = env_id in permanence_module.SWAP_ENV_IDS

    # 第 1 行：visible-objects 4 个面板（按 _panel_specs_for_env 给出的顺序）；
    # swap env 隐藏中间两个 panel（cube / button）。
    hidden_top_keys = {"cube", "button"} if is_swap_env else set()
    for ax, key in zip(axes[0], panel_specs):
        if key in hidden_top_keys:
            ax.axis("off")
        else:
            xy_common._plot_panel(ax, key, env_id, points)
            # row-0 走 xy_common._plot_panel，share=True 会让非首列 panel
            # 隐藏 y tick labels、非底行 panel 隐藏 x tick labels。permanence
            # 视图要求每个 2D 散点 panel 都显式显示自己的坐标。
            ax.tick_params(axis="x", labelbottom=True)
            ax.tick_params(axis="y", labelleft=True)

    # 第 2 行：左 2 格放 cubes / swaps；swap env 在右 2 格放 pair-freq heatmap
    _plot_permanence_cubes_panel(axes[1, 0], env_id, perm_files)
    _plot_permanence_swaps_panel(axes[1, 1], env_id, perm_files)
    if is_swap_env and n_top >= 4:
        _plot_pair_frequency_panel(axes[1, 2], env_id, perm_files, 3, plt)
        _plot_pair_frequency_panel(axes[1, 3], env_id, perm_files, 4, plt)
        for j in range(4, n_top):
            axes[1, j].axis("off")
    else:
        for j in range(2, n_top):
            axes[1, j].axis("off")

    # 第 3 行：左 2 格放 first / second pickup bin 2D 散点；右 2 格放
    # first / second pickup bin_index 1D 选择分布柱状图。
    _plot_pickup_bin_panel(axes[2, 0], env_id, perm_files, pickup_index=0)
    _plot_pickup_bin_panel(axes[2, 1], env_id, perm_files, pickup_index=1)
    if n_top >= 4:
        _plot_pickup_bin_selection_distribution_panel(
            axes[2, 2], env_id, perm_files, pickup_index=0
        )
        _plot_pickup_bin_selection_distribution_panel(
            axes[2, 3], env_id, perm_files, pickup_index=1
        )
        for j in range(4, n_top):
            axes[2, j].axis("off")
    else:
        for j in range(2, n_top):
            axes[2, j].axis("off")

    output_path = xy_dir / f"{env_id}_xy.png"
    return xy_common._save_combined_figure(fig, output_path, plt)


# ---------------------------------------------------------------------------
# 单一公开接口
# ---------------------------------------------------------------------------


def visualize(
    segmentation_dir: Path,
    output_dir: Path,
    env_id: Optional[str] = None,
) -> tuple[list, list]:
    """发现 permanence sidecar + visible_objects → 渲染 3 行 PNG 到 inspect-stat/xy/。

    `output_dir` 仅做 anchor：实际写入路径为 ``output_dir.parent / "xy" / {env}_xy.png``。
    旧的 ``permanance_inspect/`` 子目录不再被创建。

    Parameters
    ----------
    segmentation_dir:
        包含 ``reset_segmentation_pngs`` 各 episode 子目录的根目录。
    output_dir:
        历史接口参数，仅用于推导 inspect-stat 根目录与 xy/ 子目录。
    env_id:
        指定只处理哪个 permanence env（必须是 PERMANENCE_ENV_IDS 之一）；
        ``None`` 表示处理全部四个。传入非 permanence env 时静默返回空列表。

    Returns
    -------
    (kept, skipped) : tuple[list[PermanenceFile], list[PermanenceFile]]
    """
    segmentation_dir = Path(segmentation_dir)
    output_dir = Path(output_dir)
    inspect_dir = output_dir.parent
    xy_dir = inspect_dir / "xy"

    if env_id is not None and env_id not in permanence_module.PERMANENCE_ENV_IDS:
        return [], []

    # 1) 发现并去重 permanence sidecar
    perm_files = permanence_module.discover_permanence_files(
        segmentation_dir, env_filter=env_id
    )
    if not perm_files:
        env_part = f" for env_id={env_id!r}" if env_id else ""
        print(
            f"[Permanance-inspect] No permanence_init_state.json found "
            f"under {segmentation_dir}{env_part}."
        )
        return [], []

    kept, skipped = permanence_module.dedup_permanence_files(perm_files)

    perm_by_env: dict[str, list] = {}
    for entry in kept:
        perm_by_env.setdefault(entry.env_id, []).append(entry)
    for bucket in perm_by_env.values():
        bucket.sort(key=lambda e: e.episode)

    # 2) 发现并去重 visible_objects（仅 permanence env 子集）
    visible_files = xy_common._discover_visible_object_files(segmentation_dir)
    visible_files = [
        f
        for f in visible_files
        if f.env_id in permanence_module.PERMANENCE_ENV_IDS
        and (env_id is None or f.env_id == env_id)
    ]
    visible_kept, _ = xy_common._dedup_visible_object_files(visible_files)
    points_by_env, _, episode_counts, _ = xy_common._build_points_from_files(
        visible_kept, difficulty_by_env_episode={}
    )

    # 3) 渲染：每个 permanence env 一张 3×N 的 xy PNG
    xy_dir.mkdir(parents=True, exist_ok=True)
    plt = xy_common._get_pyplot(show=False)

    for eid in sorted(perm_by_env):
        out_path = _render_three_row_figure(
            xy_dir,
            eid,
            points_by_env.get(eid, []),
            perm_by_env[eid],
            episode_counts.get(eid, 0),
            plt,
        )
        print(
            f"[Permanance-inspect] {eid}: episodes={len(perm_by_env[eid])} "
            f"points={len(points_by_env.get(eid, []))} -> {out_path}"
        )

    plt.close("all")

    if skipped:
        print(
            f"[Permanance-inspect] Skipped {len(skipped)} older-seed duplicate(s):"
        )
        for entry in skipped:
            print(
                f"  - env={entry.env_id} ep={entry.episode} seed={entry.seed}"
            )

    return kept, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_VALID_ENV_IDS = sorted(permanence_module.PERMANENCE_ENV_IDS)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Permanence 套件专用可视化：扫描 permanence_init_state.json + "
            "visible_objects.json，为 VideoUnmask / VideoUnmaskSwap / "
            "ButtonUnmask / ButtonUnmaskSwap 各生成一张 3 行 PNG（行 1 visible-"
            "objects、行 2 cubes + swaps、行 3 first/second pickup bin），"
            "写入 inspect-stat/xy/。"
        )
    )
    parser.add_argument(
        "--segmentation-dir",
        type=Path,
        default=DEFAULT_SEGMENTATION_DIR,
        help="包含各 episode 子目录（内有 permanence_init_state.json / "
        "visible_objects.json）的根目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="历史接口参数：实际输出位置为 <output-dir>/../xy/{env_id}_xy.png；"
        "本目录本身不会被创建。",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=None,
        choices=_VALID_ENV_IDS,
        help=(
            f"只处理指定的 permanence env，不传则处理全部四个。"
            f"可选值：{_VALID_ENV_IDS}"
        ),
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    segmentation_dir = args.segmentation_dir.resolve()
    output_dir = args.output_dir.resolve()

    print(f"Segmentation dir: {segmentation_dir}")
    print(f"Output anchor:    {output_dir}")
    print(f"Effective xy dir: {output_dir.parent / 'xy'}")
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
