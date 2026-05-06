"""Permanence 套件（VideoUnmask / VideoUnmaskSwap / ButtonUnmask / ButtonUnmaskSwap）
专用可视化脚本。

可独立运行，也可被 inspect_stat.py 导入。对外只暴露一个公开接口：
    visualize(segmentation_dir, output_dir, env_id=None)

所有绘图逻辑与 inspect_stat.py 的 permanence panel 保持完全一致（使用相同常量
与相同 _plot_* 函数），以确保两处产出的图像视觉相同。
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLBACKEND", "Agg")

# permanence.py 在同一目录下，直接 sys.path 注入
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import permanence as permanence_module  # noqa: E402

# ---------------------------------------------------------------------------
# 默认路径
# ---------------------------------------------------------------------------

_DEFAULT_BASE = Path("/data/hongzefu/robomme_benchmark_cvpr2026-heldoutSeed/runs/replay_videos")
DEFAULT_SEGMENTATION_DIR = _DEFAULT_BASE / "reset_segmentation_pngs"
DEFAULT_OUTPUT_DIR = _DEFAULT_BASE / "inspect-stat" / "permanance_inspect"

# ---------------------------------------------------------------------------
# 常量（与 inspect_stat.py 完全一致，保证视觉相同）
# ---------------------------------------------------------------------------

XY_LIMIT = 0.3
AGGREGATE_DPI = 300

CUBE_COLOR_MAP = {
    "red": "#d62728",
    "green": "#2ca02c",
    "blue": "#1f77b4",
    "unknown": "#7f7f7f",
}

_PERMANENCE_DIFFICULTY_MARKERS: dict[str, str] = {"easy": "o", "medium": "s", "hard": "^"}
_PERMANENCE_SWAP_INDEX_COLORS: list[str] = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# ---------------------------------------------------------------------------
# 私有工具（从 inspect_stat.py 原样复制）
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


# ---------------------------------------------------------------------------
# 私有绘图（从 inspect_stat.py 原样移入，代码一字不差）
# ---------------------------------------------------------------------------


def _plot_permanence_cubes_objects(
    ax,
    env_id: str,
    permanence_files: Optional[list],
) -> None:
    """渲染 cube reset 位置散点：颜色按 cube 颜色、marker 按 difficulty，
    并用细线把 cube 连到对应的 bin。无数据时显示 'No cube data'。"""
    from matplotlib.lines import Line2D

    files = list(permanence_files or [])
    plotted = 0
    seen_difficulties: set[str] = set()
    seen_colors: set[str] = set()

    for entry in files:
        difficulty = str(entry.payload.get("difficulty", ""))
        marker = _PERMANENCE_DIFFICULTY_MARKERS.get(difficulty, "x")
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
                x,
                y,
                s=70,
                alpha=0.85,
                c=color,
                marker=marker,
                edgecolors="black",
                linewidths=0.5,
            )
            bin_idx = cube.get("bin_index")
            if bin_idx is not None:
                ax.text(
                    x + 0.005,
                    y + 0.005,
                    f"ep{entry.episode}/b{bin_idx}",
                    fontsize=6,
                    alpha=0.6,
                )
            plotted += 1

    if plotted:
        legend_handles: list[Line2D] = []
        for color_name in sorted(seen_colors):
            color = CUBE_COLOR_MAP.get(color_name, CUBE_COLOR_MAP["unknown"])
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=8,
                    markerfacecolor=color,
                    markeredgecolor="black",
                    label=f"cube_{color_name}",
                )
            )
        for diff in sorted(seen_difficulties):
            marker = _PERMANENCE_DIFFICULTY_MARKERS.get(diff, "x")
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    linestyle="",
                    markersize=8,
                    markerfacecolor="white",
                    markeredgecolor="black",
                    label=f"difficulty: {diff or 'unknown'}",
                )
            )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7)
    else:
        ax.text(0.0, 0.0, "No cube data", ha="center", va="center")

    _prepare_axis(ax, "Permanence cubes (Rotated XY)", plotted)


def _plot_permanence_swaps_objects(
    ax,
    env_id: str,
    permanence_files: Optional[list],
) -> None:
    """渲染 swap pair 双向箭头：每对 (bin_a, bin_b) 按 swap_index 着色，
    bin 全集画成淡灰色背景点。非 Swap env 或无 swap_pairs 时显示
    'No swap data'。"""
    from matplotlib.lines import Line2D

    files = list(permanence_files or [])
    is_swap_env = env_id in permanence_module.SWAP_ENV_IDS

    pair_count = 0
    seen_swap_indices: set[int] = set()

    if is_swap_env:
        for entry in files:
            for bin_info in entry.payload.get("bins", []):
                pos = bin_info.get("position_xy") or [0.0, 0.0]
                bx, by = _xy_rot_cw_90(float(pos[0]), float(pos[1]))
                ax.scatter(bx, by, s=20, color="lightgray", alpha=0.4, zorder=1)

            for pair in entry.payload.get("swap_pairs", []):
                swap_idx = int(pair.get("swap_index", 0))
                seen_swap_indices.add(swap_idx)
                color = _PERMANENCE_SWAP_INDEX_COLORS[
                    swap_idx % len(_PERMANENCE_SWAP_INDEX_COLORS)
                ]
                a_xy = pair.get("bin_a_position_xy") or [0.0, 0.0]
                b_xy = pair.get("bin_b_position_xy") or [0.0, 0.0]
                ax_x, ax_y = _xy_rot_cw_90(float(a_xy[0]), float(a_xy[1]))
                bx_x, bx_y = _xy_rot_cw_90(float(b_xy[0]), float(b_xy[1]))

                ax.annotate(
                    "",
                    xy=(bx_x, bx_y),
                    xytext=(ax_x, ax_y),
                    arrowprops=dict(
                        arrowstyle="<->",
                        color=color,
                        lw=1.4,
                        alpha=0.7,
                        shrinkA=4,
                        shrinkB=4,
                    ),
                    zorder=3,
                )
                ax.scatter(
                    [ax_x, bx_x],
                    [ax_y, bx_y],
                    s=60,
                    c=color,
                    edgecolors="black",
                    linewidths=0.5,
                    alpha=0.9,
                    zorder=4,
                )
                mid_x = (ax_x + bx_x) / 2
                mid_y = (ax_y + bx_y) / 2
                ax.text(
                    mid_x,
                    mid_y,
                    f"ep{entry.episode}#s{swap_idx}",
                    fontsize=6,
                    alpha=0.7,
                )
                pair_count += 1

    if pair_count:
        legend_handles: list[Line2D] = []
        for swap_idx in sorted(seen_swap_indices):
            color = _PERMANENCE_SWAP_INDEX_COLORS[
                swap_idx % len(_PERMANENCE_SWAP_INDEX_COLORS)
            ]
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="-",
                    color=color,
                    markersize=8,
                    label=f"swap #{swap_idx}",
                )
            )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7)
    else:
        ax.text(0.0, 0.0, "No swap data", ha="center", va="center")

    _prepare_axis(ax, "Permanence swaps (Rotated XY)", pair_count)


# ---------------------------------------------------------------------------
# 私有渲染（新增：把两个 panel 拼成独立 1×2 figure）
# ---------------------------------------------------------------------------


def _render_permanence_collage(
    output_dir: Path,
    env_id: str,
    permanence_files: list,
    plt,
) -> Path:
    """为单个 env 渲染 permanence_cubes | permanence_swaps 双面板图。"""
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    n_episodes = len(permanence_files)
    fig.suptitle(
        f"{env_id} | permanence | episodes={n_episodes}",
        fontsize=18,
    )

    _plot_permanence_cubes_objects(axes[0], env_id, permanence_files)
    _plot_permanence_swaps_objects(axes[1], env_id, permanence_files)

    output_path = output_dir / f"{env_id}_permanence.png"
    return _save_combined_figure(fig, output_path, plt)


def _get_pyplot(show: bool = False):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# ---------------------------------------------------------------------------
# 单一公开接口
# ---------------------------------------------------------------------------


def visualize(
    segmentation_dir: Path,
    output_dir: Path,
    env_id: Optional[str] = None,
) -> tuple[list, list]:
    """发现 permanence sidecar → 去重 → 渲染 cubes + swaps 双面板图。

    Parameters
    ----------
    segmentation_dir:
        包含 ``reset_segmentation_pngs`` 各 episode 子目录的根目录。
    output_dir:
        输出目录，每个 env 生成 ``{env_id}_permanence.png``。
    env_id:
        指定只画哪个 permanence env（必须是 PERMANENCE_ENV_IDS 之一）；
        ``None`` 表示处理全部四个。传入非 permanence env 时静默返回空列表。

    Returns
    -------
    (kept, skipped) : tuple[list[PermanenceFile], list[PermanenceFile]]
    """
    segmentation_dir = Path(segmentation_dir)
    output_dir = Path(output_dir)

    if env_id is not None and env_id not in permanence_module.PERMANENCE_ENV_IDS:
        return [], []

    files = permanence_module.discover_permanence_files(
        segmentation_dir, env_filter=env_id
    )
    if not files:
        env_part = f" for env_id={env_id!r}" if env_id else ""
        print(
            f"[Permanance-inspect] No permanence_init_state.json found "
            f"under {segmentation_dir}{env_part}."
        )
        return [], []

    kept, skipped = permanence_module.dedup_permanence_files(files)

    by_env: dict[str, list] = {}
    for entry in kept:
        by_env.setdefault(entry.env_id, []).append(entry)
    for bucket in by_env.values():
        bucket.sort(key=lambda e: e.episode)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _get_pyplot(show=False)

    for eid, env_files in sorted(by_env.items()):
        out_path = _render_permanence_collage(output_dir, eid, env_files, plt)
        print(f"[Permanance-inspect] {eid}: episodes={len(env_files)} -> {out_path}")

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
            "Permanence 套件专用可视化：扫描 permanence_init_state.json sidecar，"
            "为 VideoUnmask / VideoUnmaskSwap / ButtonUnmask / ButtonUnmaskSwap "
            "各生成一张 cube reset 位置 + swap pair 双面板图。"
        )
    )
    parser.add_argument(
        "--segmentation-dir",
        type=Path,
        default=DEFAULT_SEGMENTATION_DIR,
        help="包含各 episode 子目录（内有 permanence_init_state.json）的根目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录，每个 env 生成 {env_id}_permanence.png。",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=None,
        choices=_VALID_ENV_IDS,
        help=(
            f"只画指定的 permanence env，不传则处理全部四个。"
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
