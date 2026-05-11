"""permanence 任务位置生成的 lightweight 演示脚本（多桶均匀采样可视化）.

生成机制使用 torch.Generator + manual_seed，与 permanence 任务
(src/robomme/robomme_env/VideoUnmaskSwap.py:177-184) 保持同一 RNG stream 风格，
便于直接对照 demo 与 permanence 任务的种子流。

两个距离类超参数严格对齐 permanence 任务：
- MIN_CENTER_DIST = 0.135：对应 VideoUnmask/ButtonUnmask 的 Poisson disk 间隔
  r_sep = (bin_half_size + min_gap) * 2 = (0.0275 + 0.04) * 2，配合 panda
  cube_half_size=0.02（object_generation.py:1080）。
- BUTTON_MIN_CENTER_DIST = 0.15：对应 ButtonUnmaskSwap 中两按钮最坏间隔——
  名义 y-spacing 0.2 减去左右各自的 randomize_range 0.025 = 0.15。

每个 episode 流程：
1. 在 [-XY_HALF, XY_HALF]^2 上均匀采样 n_bins 个 (x, y)，最小中心距
   MIN_CENTER_DIST（rejection sampling）；落入 button strip
   [BUTTON_X_MIN, BUTTON_X_MAX] x [BUTTON_Y_MIN, BUTTON_Y_MAX] 的样本被拒绝，
   保证 cube/bin 不与 button 位置重叠（对应 permanence 中 avoid=[button_obb]）。
2. 随机洗牌 (red, green, blue, "no_cube" * (n_bins - 3)) 多重集，给每个 bin
   分配颜色（torch.randperm，与 VideoUnmask.py:191 一致）。
3. 用独立 pickup_generator（seed * KNUTH_HASH + 1）从 3 个有色 bin 中无放回挑出
   n_pickups 个作为目标（与 VideoUnmaskSwap.py:181 一致）。
4. 用独立 swap_generator（seed * KNUTH_HASH + 2）循环 n_swaps 次抽取 (i, j) 对
   作为 swap 对象。
5. 仅当 n_buttons > 0：用独立 button_generator（seed * KNUTH_HASH + 3；offset 3
   是 demo 私有的，permanence ButtonUnmask 实际从主 stream 抽，但 demo 用独立
   stream 以便在不影响 bin/pickup/swap 的前提下叠加 button 采样）在 vertical
   strip [BUTTON_X_MIN, BUTTON_X_MAX] x [BUTTON_Y_MIN, BUTTON_Y_MAX] =
   [-0.25, -0.15] x [-0.2, 0.2]（x 对应 permanence x ~ -0.2 button strip；y
   范围扩到完整 demo 工作区，避免 rejection sampling 在 0.15 最小中心距下出现
   几何死区）上均匀采样 n_buttons 个位置。

输出：默认 1000 episodes 的 2 行多列 PNG 一张。
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 距离/范围/调色板常量与 generator 函数都从 src 下的正式 utility re-import，
# 保持单一真相（避免 demo 与 env 各持一份副本而漂移）。
from robomme.robomme_env.utils.permanance_task_pos_generator import (
    BUTTON_MIN_CENTER_DIST,
    BUTTON_X_MAX,
    BUTTON_X_MIN,
    BUTTON_Y_MAX,
    BUTTON_Y_MIN,
    COLORS,
    MIN_CENTER_DIST,
    XY_HALF,
    permanance_task_pos_generator,
)

# === 调色板与 CLI 默认输出（绘图/CLI 资产，与超参数概念正交）===
COLOR_TO_RGB: dict[str, str] = {
    "red": "#d62728",
    "green": "#2ca02c",
    "blue": "#1f77b4",
    "no_cube": "#7f7f7f",
}
BUTTON_COLOR = "#7f7f7f"
DEFAULT_OUT = Path("runs/lightweight_bin_demo/bin_distribution.png")


@dataclass(frozen=True)
class BinSample:
    episode: int
    bin_index: int
    x: float
    y: float
    color: str
    pickup_order: int | None


@dataclass(frozen=True)
class EpisodeSwaps:
    episode: int
    pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ButtonSample:
    episode: int
    button_index: int
    x: float
    y: float


def _prepare_axis(ax: plt.Axes, title: str, point_count: int) -> None:
    ax.set_xlim(-XY_HALF * 1.15, XY_HALF * 1.15)
    ax.set_ylim(-XY_HALF * 1.15, XY_HALF * 1.15)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{title}  (n={point_count})", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.plot(
        [-XY_HALF, XY_HALF, XY_HALF, -XY_HALF, -XY_HALF],
        [-XY_HALF, -XY_HALF, XY_HALF, XY_HALF, -XY_HALF],
        linestyle="--",
        color="black",
        linewidth=1.0,
        alpha=0.7,
    )


def _draw_color_panel(ax: plt.Axes, samples: list[BinSample], color: str) -> None:
    pts = [s for s in samples if s.color == color]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    ax.scatter(xs, ys, s=18, c=COLOR_TO_RGB[color], alpha=0.7, edgecolors="none")
    _prepare_axis(ax, f"{color} bin", len(pts))


def _draw_no_cube_panel(ax: plt.Axes, samples: list[BinSample], slot: int) -> None:
    # 每个 episode 的 no_cube bin 按 bin_index 升序展示。
    # samples 按 (episode, bin_index) 排好序，因此每个 episode 第 k 个遇到的
    # no_cube 就放在 slot=k 这个面板上。
    counts: dict[int, int] = {}
    pts: list[BinSample] = []
    for s in samples:
        if s.color != "no_cube":
            continue
        idx = counts.get(s.episode, 0)
        if idx == slot:
            pts.append(s)
        counts[s.episode] = idx + 1
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    ax.scatter(xs, ys, s=18, c=COLOR_TO_RGB["no_cube"], alpha=0.7, edgecolors="none")
    _prepare_axis(ax, f"no_cube #{slot + 1} bin", len(pts))


def _draw_all_colors_panel(ax: plt.Axes, samples: list[BinSample]) -> None:
    cs = [COLOR_TO_RGB[s.color] for s in samples]
    xs = [s.x for s in samples]
    ys = [s.y for s in samples]
    ax.scatter(xs, ys, s=10, c=cs, alpha=0.55, edgecolors="none")
    _prepare_axis(ax, "all 4 colors combined", len(samples))


def _draw_pickup_panel(ax: plt.Axes, samples: list[BinSample], order: int) -> None:
    pts = [s for s in samples if s.pickup_order == order]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    cs = [COLOR_TO_RGB[p.color] for p in pts]
    ax.scatter(xs, ys, s=22, c=cs, alpha=0.75, edgecolors="white", linewidths=0.4)
    label = f"pickup #{order + 1}"
    _prepare_axis(ax, label, len(pts))


def _draw_buttons_panel(ax: plt.Axes, button_samples: list[ButtonSample]) -> None:
    xs = [b.x for b in button_samples]
    ys = [b.y for b in button_samples]
    ax.scatter(xs, ys, s=18, c=BUTTON_COLOR, alpha=0.7, edgecolors="none")
    # button strip 坐标轴和其他 panel 的 workspace 不一样：strip 在
    # x ∈ [-0.25, -0.15]，超出 [-XY_HALF, XY_HALF]，所以单独设范围。
    pad_x = (BUTTON_X_MAX - BUTTON_X_MIN) * 0.15
    pad_y = (BUTTON_Y_MAX - BUTTON_Y_MIN) * 0.15
    ax.set_xlim(BUTTON_X_MIN - pad_x, BUTTON_X_MAX + pad_x)
    ax.set_ylim(BUTTON_Y_MIN - pad_y, BUTTON_Y_MAX + pad_y)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"buttons  (n={len(button_samples)})", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.plot(
        [BUTTON_X_MIN, BUTTON_X_MAX, BUTTON_X_MAX, BUTTON_X_MIN, BUTTON_X_MIN],
        [BUTTON_Y_MIN, BUTTON_Y_MIN, BUTTON_Y_MAX, BUTTON_Y_MAX, BUTTON_Y_MIN],
        linestyle="--",
        color="black",
        linewidth=1.0,
        alpha=0.7,
    )


def _draw_swap_pair_panel(
    ax: plt.Axes, episode_swaps_list: list[EpisodeSwaps], n_bins: int
) -> None:
    freq = np.zeros((n_bins, n_bins), dtype=int)
    total_swaps = 0
    for ep_swaps in episode_swaps_list:
        for a, b in ep_swaps.pairs:
            freq[a, b] += 1
            total_swaps += 1

    display = freq.astype(float).copy()
    mask = np.tri(n_bins, n_bins, k=0, dtype=bool)
    display[mask] = np.nan

    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad(color="#fff7e6")
    ax.imshow(display, cmap=cmap, aspect="equal")

    for i in range(n_bins):
        for j in range(n_bins):
            if i < j:
                ax.text(
                    j, i, str(freq[i, j]),
                    ha="center", va="center",
                    color="black", fontsize=11,
                )
            else:
                ax.text(
                    j, i, "—",
                    ha="center", va="center",
                    color="#888888", fontsize=11,
                )

    ax.set_xticks(range(n_bins))
    ax.set_yticks(range(n_bins))
    ax.set_xticklabels([f"bin_{i}" for i in range(n_bins)])
    ax.set_yticklabels([f"bin_{i}" for i in range(n_bins)])
    ax.set_title(
        f"Pair freq ({n_bins} bin)\n"
        f"episodes={len(episode_swaps_list)} | swaps={total_swaps}",
        fontsize=11,
    )


def render_figure(
    samples: list[BinSample],
    episode_swaps_list: list[EpisodeSwaps],
    button_samples: list[ButtonSample],
    num_buttons: int,
    out_path: Path,
) -> None:
    """把聚合后的 samples / swaps / buttons 渲染成 2 行 N 列的 PNG.

    layout 数字均从数据 / 调色板派生（不再依赖模块全局常量）：
    - n_bins:           samples 中最大 bin_index + 1
    - n_colored_cubes:  COLORS 调色板长度 - 1（末项是 "no_cube"）
    - n_no_cube_bins:   n_bins - n_colored_cubes
    - n_pickups:        samples 中最大 pickup_order + 1
    """
    n_bins = max(s.bin_index for s in samples) + 1
    n_colored_cubes = len(COLORS) - 1
    n_no_cube_bins = n_bins - n_colored_cubes
    n_pickups = max(
        (s.pickup_order for s in samples if s.pickup_order is not None),
        default=-1,
    ) + 1

    row0_panels = n_colored_cubes + n_no_cube_bins  # = n_bins
    # combined + N pickups + swap_freq，再加一个可选的 buttons-only panel
    row1_panels = 2 + n_pickups + (1 if num_buttons > 0 else 0)
    n_cols = max(row0_panels, row1_panels)
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 5, 10))

    # 第 0 行：3 个有色 cube panel + 每个 slot 一个 no_cube panel
    _draw_color_panel(axes[0, 0], samples, "red")
    _draw_color_panel(axes[0, 1], samples, "green")
    _draw_color_panel(axes[0, 2], samples, "blue")
    for k in range(n_no_cube_bins):
        _draw_no_cube_panel(axes[0, n_colored_cubes + k], samples, k)
    for c in range(row0_panels, n_cols):
        axes[0, c].set_visible(False)

    # 第 1 行：combined / pickup #1..N / swap pair freq / [buttons]
    _draw_all_colors_panel(axes[1, 0], samples)
    for p in range(n_pickups):
        _draw_pickup_panel(axes[1, 1 + p], samples, p)
    _draw_swap_pair_panel(axes[1, 1 + n_pickups], episode_swaps_list, n_bins)
    if num_buttons > 0:
        _draw_buttons_panel(axes[1, 2 + n_pickups], button_samples)
    for c in range(row1_panels, n_cols):
        axes[1, c].set_visible(False)

    fig.suptitle(
        f"Lightweight {n_bins}-bin demo: xy in [-{XY_HALF}, {XY_HALF}], "
        f"min_center_dist={MIN_CENTER_DIST}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_summary(
    args: argparse.Namespace,
    samples: list[BinSample],
    episode_swaps_list: list[EpisodeSwaps],
    button_samples: list[ButtonSample],
    failed_seeds: list[tuple[int, str]],
) -> None:
    color_counts = Counter(s.color for s in samples)
    pick_counts = Counter(
        (s.color, s.pickup_order) for s in samples if s.pickup_order is not None
    )
    swap_pair_counts = Counter(
        pair for ep in episode_swaps_list for pair in ep.pairs
    )
    n_total = args.n_episodes
    n_failed = len(failed_seeds)
    n_success = n_total - n_failed
    success_rate = n_success / n_total if n_total > 0 else 0.0
    print(
        f"wrote {args.out}  "
        f"({n_success}/{n_total} episodes succeeded = {success_rate:.1%}, "
        f"{len(samples)} bin samples)"
    )
    if n_failed > 0:
        skipped = ", ".join(str(s) for s, _ in failed_seeds[:10])
        more = f" (and {n_failed - 10} more)" if n_failed > 10 else ""
        print(f"skipped seeds (first 10): {skipped}{more}")
    print(f"color counts: {dict(color_counts)}")
    print(f"pickup counts (color, order): {dict(pick_counts)}")
    print(f"swap pair counts (i<j)=count: {dict(sorted(swap_pair_counts.items()))}")
    if args.num_buttons > 0:
        print(
            f"button samples: {len(button_samples)} "
            f"({n_success} succeeded episodes x {args.num_buttons} buttons)"
        )


def _print_debug_episode(
    seed: int, n_bins: int, n_swaps: int, n_buttons: int, n_pickups: int
) -> None:
    result = permanance_task_pos_generator(n_bins, n_swaps, n_buttons, n_pickups, seed)
    if "fail" in result:
        print(f"[debug seed={seed}] FAILED: {result['fail']}")
        return
    bin_colors = result["bin_colors"]
    pickup_map = result["pickup_map"]
    chosen = sorted(pickup_map.keys(), key=lambda i: pickup_map[i])
    print(
        f"[debug seed={seed}] bin_colors={bin_colors}  "
        f"chosen_bin_indices={chosen}  swap_pairs={list(result['swap_pairs'])}"
    )
    if n_buttons > 0:
        print(f"[debug seed={seed}] button_positions={result['button_positions']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=1000)
    parser.add_argument("--n-bins", type=int, default=6)
    parser.add_argument("--n-swaps", type=int, default=3)
    parser.add_argument("--n-pickups", type=int, default=2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--num-buttons",
        type=int,
        default=2,
        help=(
            "如果 > 0，每个 episode 额外在 vertical strip "
            f"[{BUTTON_X_MIN}, {BUTTON_X_MAX}] x [{BUTTON_Y_MIN}, "
            f"{BUTTON_Y_MAX}] 中采样 N 个 button（对应 permanence 的 x ~ -0.2 "
            "button strip；y 范围扩到完整 demo workspace 以避免 0.15 最小中心距"
            "下出现几何死区），最小中心距 "
            f"{BUTTON_MIN_CENTER_DIST}（值与 ButtonUnmaskSwap 最坏情况 "
            "0.2 - 2*0.025 = 0.15 一致）。bin 采样会拒绝落入此 strip 的位置，"
            "保证 cube 与 button 不重叠。"
        ),
    )
    parser.add_argument(
        "--debug-single-episode",
        type=int,
        default=None,
        help="若设置，则额外打印这个 seed 下的 bin_colors 和 chosen_bin_indices "
        "（便于和 permanence 任务内部对照）。",
    )
    args = parser.parse_args()

    samples: list[BinSample] = []
    episode_swaps_list: list[EpisodeSwaps] = []
    button_samples_all: list[ButtonSample] = []
    failed_seeds: list[tuple[int, str]] = []
    for ep in range(args.n_episodes):
        seed = args.base_seed + ep
        result = permanance_task_pos_generator(
            args.n_bins, args.n_swaps, args.num_buttons, args.n_pickups, seed,
        )
        if "fail" in result:
            # CLAUDE.md 要求显式暴露错误：打印一行并跳过这个 seed，其余 seed 继续
            print(f"[skip seed={seed}] RuntimeError: {result['fail']}")
            failed_seeds.append((seed, result["fail"]))
            continue
        samples.extend(
            BinSample(ep, i, x, y, c, result["pickup_map"].get(i))
            for i, ((x, y), c) in enumerate(
                zip(result["bin_positions"], result["bin_colors"])
            )
        )
        episode_swaps_list.append(EpisodeSwaps(ep, tuple(result["swap_pairs"])))
        button_samples_all.extend(
            ButtonSample(ep, i, x, y)
            for i, (x, y) in enumerate(result["button_positions"])
        )

    render_figure(
        samples, episode_swaps_list, button_samples_all, args.num_buttons, args.out
    )
    _print_summary(args, samples, episode_swaps_list, button_samples_all, failed_seeds)
    if args.debug_single_episode is not None:
        _print_debug_episode(
            args.debug_single_episode,
            args.n_bins,
            args.n_swaps,
            args.num_buttons,
            args.n_pickups,
        )


if __name__ == "__main__":
    main()
