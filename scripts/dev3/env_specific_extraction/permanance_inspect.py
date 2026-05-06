"""Permanence 套件（VideoUnmask / VideoUnmaskSwap / ButtonUnmask / ButtonUnmaskSwap）
专用可视化入口。

可被 inspect_stat.py 调用，也可独立运行。对外只暴露一个公开接口：
    visualize(segmentation_dir, output_dir, env_id=None)

调用一次 visualize() 直接产出完整的 2 行 PNG 到 inspect-stat/xy/{env_id}_xy.png：
- 第 1 行：visible-objects 4 个面板（与 inspect_stat 普通 xy 图保持一致）
- 第 2 行：permanence cubes + permanence swaps 双面板

permanance_inspect/ 目录不再被创建——`output_dir` 仅作 anchor，用来定位
inspect-stat 根目录下的 xy/ 子目录。

数据层（discover / dedup / sidecar 解析）走 permanence.py；
可视化层（panel 绘制 / 2 行 figure 拼装）一律落在本文件——
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

# permanence.py 在同一目录下，直接 sys.path 注入
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# inspect_stat.py 位于上一级 scripts/dev3/，同样 sys.path 注入；真正的
# import 推迟到 visualize() 内部，避免和 inspect_stat 的双向 import 冲突
# （inspect_stat.py 在 module-level 导入本模块）。
_INSPECT_STAT_DIR = _SCRIPT_DIR.parent
if str(_INSPECT_STAT_DIR) not in sys.path:
    sys.path.insert(0, str(_INSPECT_STAT_DIR))

import permanence as permanence_module  # noqa: E402

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

PERMANENCE_DIFFICULTY_MARKERS: dict[str, str] = {
    "easy": "o",
    "medium": "s",
    "hard": "^",
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


# ---------------------------------------------------------------------------
# 内部：permanence cubes / swaps 面板渲染
# ---------------------------------------------------------------------------


def _plot_permanence_cubes_panel(
    ax: Any,
    env_id: str,
    files: Iterable[Any],
) -> int:
    """渲染 cube reset 位置散点：颜色按 cube 颜色、marker 按 difficulty，
    并用细线把 cube 连到对应的 bin。无数据时显示 'No cube data'。"""
    from matplotlib.lines import Line2D

    files_list = list(files or [])
    plotted = 0
    seen_difficulties: set[str] = set()
    seen_colors: set[str] = set()

    for entry in files_list:
        difficulty = str(entry.payload.get("difficulty", ""))
        marker = PERMANENCE_DIFFICULTY_MARKERS.get(difficulty, "x")
        seen_difficulties.add(difficulty)

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
                s=70, alpha=0.85, c=color, marker=marker,
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
        for diff in sorted(seen_difficulties):
            marker = PERMANENCE_DIFFICULTY_MARKERS.get(diff, "x")
            legend_handles.append(
                Line2D(
                    [0], [0],
                    marker=marker, linestyle="", markersize=8,
                    markerfacecolor="white", markeredgecolor="black",
                    label=f"difficulty: {diff or 'unknown'}",
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
# 内部：渲染 2 行 figure
# ---------------------------------------------------------------------------


def _render_two_row_figure(
    xy_dir: Path,
    env_id: str,
    points: list,
    perm_files: list,
    episode_count: int,
    inspect_stat_module,
    plt,
) -> Path:
    """生成 2 行布局：上行 visible-objects 面板，下行 permanence cubes + swaps。"""
    import numpy as np

    panel_specs = inspect_stat_module._panel_specs_for_env(env_id)
    n_top = len(panel_specs)
    fig, axes = plt.subplots(
        2,
        n_top,
        figsize=(7 * n_top, 14),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    fig.suptitle(
        f"{env_id} | episodes={episode_count} | points={len(points)} | "
        f"permanence_episodes={len(perm_files)}",
        fontsize=18,
    )

    # 第 1 行：visible-objects 4 个面板（按 _panel_specs_for_env 给出的顺序）
    for ax, key in zip(axes[0], panel_specs):
        inspect_stat_module._plot_panel(ax, key, env_id, points)

    # 第 2 行：左 2 格放 cubes / swaps，其余隐藏
    _plot_permanence_cubes_panel(axes[1, 0], env_id, perm_files)
    _plot_permanence_swaps_panel(axes[1, 1], env_id, perm_files)
    for j in range(2, n_top):
        axes[1, j].axis("off")

    output_path = xy_dir / f"{env_id}_xy.png"
    return inspect_stat_module._save_combined_figure(fig, output_path, plt)


# ---------------------------------------------------------------------------
# 单一公开接口
# ---------------------------------------------------------------------------


def visualize(
    segmentation_dir: Path,
    output_dir: Path,
    env_id: Optional[str] = None,
) -> tuple[list, list]:
    """发现 permanence sidecar + visible_objects → 渲染 2 行 PNG 到 inspect-stat/xy/。

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
    # 延迟 import 避免与 inspect_stat 的循环 import
    import inspect_stat as inspect_stat_module  # noqa: WPS433

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
    visible_files = inspect_stat_module._discover_visible_object_files(segmentation_dir)
    visible_files = [
        f
        for f in visible_files
        if f.env_id in permanence_module.PERMANENCE_ENV_IDS
        and (env_id is None or f.env_id == env_id)
    ]
    visible_kept, _ = inspect_stat_module._dedup_visible_object_files(visible_files)
    points_by_env, _, episode_counts, _ = inspect_stat_module._build_points_from_files(
        visible_kept, difficulty_by_env_episode={}
    )

    # 3) 渲染：每个 permanence env 一张 2×N 的 xy PNG
    xy_dir.mkdir(parents=True, exist_ok=True)
    plt = inspect_stat_module._get_pyplot(show=False)

    for eid in sorted(perm_by_env):
        out_path = _render_two_row_figure(
            xy_dir,
            eid,
            points_by_env.get(eid, []),
            perm_by_env[eid],
            episode_counts.get(eid, 0),
            inspect_stat_module,
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
            "ButtonUnmask / ButtonUnmaskSwap 各生成一张 2 行 PNG（上行 visible-"
            "objects、下行 cubes + swaps），写入 inspect-stat/xy/。"
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
