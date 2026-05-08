"""Lightweight 4-bin uniform sampling demo for permanence randomization reference.

Uses torch.Generator + manual_seed in the same pattern as
src/robomme/robomme_env/VideoUnmaskSwap.py:177-184 so the demo's RNG stream is
directly comparable to permanence task seeds.

The two distance hyper-parameters are calibrated to match permanence:
- MIN_CENTER_DIST = 0.135 mirrors VideoUnmask/ButtonUnmask's Poisson disk
  separation r_sep = (bin_half_size + min_gap) * 2 = (0.0275 + 0.04) * 2 with
  panda's cube_half_size=0.02 (object_generation.py:1080).
- BUTTON_MIN_CENTER_DIST = 0.15 mirrors ButtonUnmaskSwap's worst-case button
  separation: nominal y-spacing 0.2 between button_left/button_right minus
  randomize_range 0.025 each side -> 0.15 in the closest configuration.

Each episode:
1. Samples N_BINS bin (x, y) positions uniformly in [-XY_HALF, XY_HALF]^2 with a
   minimum centre-to-centre separation MIN_CENTER_DIST (rejection sampling on
   torch.rand). Samples falling inside the button strip
   [BUTTON_X_MIN, BUTTON_X_MAX] x [BUTTON_Y_MIN, BUTTON_Y_MAX] are rejected so
   cubes/bins never spawn where buttons live (mirrors permanence's
   avoid=[button_obb] for bins).
2. Shuffles the 4-tuple ("red", "green", "blue", "no_cube") and assigns one
   colour to each bin (torch.randperm, same as VideoUnmask.py:191).
3. Picks 2 bins out of the 3 coloured-cube bins without replacement, deriving
   pickup_generator via seed * 2654435761 + 1 (same as VideoUnmaskSwap.py:181).
4. If --num-buttons N > 0, derives button_generator via seed * 2654435761 + 3
   (offset 3 is demo-local; permanence ButtonUnmask draws buttons from the main
   stream, but the demo offsets to a separate stream so it can be added without
   shifting the bin/pickup/swap RNG streams) and uniformly samples N button
   (x, y) in the vertical strip [BUTTON_X_MIN, BUTTON_X_MAX] x [BUTTON_Y_MIN,
   BUTTON_Y_MAX] = [-0.25, -0.15] x [-0.2, 0.2] (matching permanence's x ~ -0.2
   button strip; y range widened to demo workspace full width so rejection
   sampling at min_dist 0.15 has no geometric dead zone) with rejection
   sampling, min center-dist = BUTTON_MIN_CENTER_DIST.

Output: a single 2x3 PNG covering 200 episodes (800 bin samples) by default.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

XY_HALF = 0.2
MIN_CENTER_DIST = 0.135
N_BINS = 6
N_COLORED_CUBES = 3
N_NO_CUBE_BINS = N_BINS - N_COLORED_CUBES
N_EPISODES = 1000
N_PICKUPS = 2
SWAP_TIMES = 3
KNUTH_HASH = 2654435761

# Button strip mirrors permanence ButtonUnmask/ButtonUnmaskSwap x position
# (vertical strip at x ~ -0.2 with y varying). The y range is widened to the
# full demo workspace [-0.2, 0.2] so that rejection sampling at min center-dist
# 0.15 has no geometric dead zone (max corner distance from strip mid-point
# (-0.2, 0) is sqrt(0.05^2 + 0.2^2) ~= 0.206 > 0.15).
# Bin sampling rejects positions that fall inside this strip so cubes never
# land where buttons would (mirrors permanence's avoid=[button_obb] for bins).
BUTTON_X_MIN = -0.25
BUTTON_X_MAX = -0.15
BUTTON_Y_MIN = -0.2
BUTTON_Y_MAX = 0.2
BUTTON_MIN_CENTER_DIST = 0.15
BUTTON_COLOR = "#7f7f7f"

COLORS: tuple[str, ...] = ("red", "green", "blue", "no_cube")
COLOR_TO_RGB: dict[str, str] = {
    "red": "#d62728",
    "green": "#2ca02c",
    "blue": "#1f77b4",
    "no_cube": "#7f7f7f",
}
# Multiset assigned to bins: 3 unique colored cubes + (N_BINS - 3) no_cube placeholders
COLOR_POOL: tuple[str, ...] = (
    COLORS[:N_COLORED_CUBES] + ("no_cube",) * (N_BINS - N_COLORED_CUBES)
)
assert len(COLOR_POOL) == N_BINS, (
    f"COLOR_POOL size {len(COLOR_POOL)} must equal N_BINS={N_BINS}"
)

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


def sample_bin_positions(
    generator: torch.Generator,
    xy_half: float,
    n_bins: int,
    min_dist: float,
    max_attempts: int = 2000,
) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    min_dist_sq = min_dist * min_dist
    for _ in range(max_attempts):
        if len(pts) == n_bins:
            return pts
        u = torch.rand(2, generator=generator)
        cx = float((u[0].item() - 0.5) * 2.0 * xy_half)
        cy = float((u[1].item() - 0.5) * 2.0 * xy_half)
        # Reject samples landing inside the button strip so cubes never
        # spawn where buttons live.
        if (BUTTON_X_MIN <= cx <= BUTTON_X_MAX
                and BUTTON_Y_MIN <= cy <= BUTTON_Y_MAX):
            continue
        if all((cx - x) ** 2 + (cy - y) ** 2 >= min_dist_sq for x, y in pts):
            pts.append((cx, cy))
    raise RuntimeError(
        f"failed to place {n_bins} bins after {max_attempts} attempts "
        f"(xy_half={xy_half}, min_dist={min_dist})"
    )


def sample_button_positions(
    generator: torch.Generator,
    n_buttons: int,
    max_attempts: int = 2000,
) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    min_dist_sq = BUTTON_MIN_CENTER_DIST * BUTTON_MIN_CENTER_DIST
    x_span = BUTTON_X_MAX - BUTTON_X_MIN
    y_span = BUTTON_Y_MAX - BUTTON_Y_MIN
    for _ in range(max_attempts):
        if len(pts) == n_buttons:
            return pts
        u = torch.rand(2, generator=generator)
        cx = float(BUTTON_X_MIN + u[0].item() * x_span)
        cy = float(BUTTON_Y_MIN + u[1].item() * y_span)
        if all((cx - x) ** 2 + (cy - y) ** 2 >= min_dist_sq for x, y in pts):
            pts.append((cx, cy))
    raise RuntimeError(
        f"failed to place {n_buttons} buttons after {max_attempts} attempts "
        f"(x_range=[{BUTTON_X_MIN}, {BUTTON_X_MAX}], "
        f"y_range=[{BUTTON_Y_MIN}, {BUTTON_Y_MAX}], "
        f"min_dist={BUTTON_MIN_CENTER_DIST})"
    )


def generate_episode(
    episode_seed: int, episode_idx: int, num_buttons: int = 0
) -> tuple[list[BinSample], EpisodeSwaps, list[ButtonSample]]:
    generator = torch.Generator()
    generator.manual_seed(episode_seed)
    pickup_generator = torch.Generator()
    pickup_generator.manual_seed(episode_seed * KNUTH_HASH + 1)
    swap_generator = torch.Generator()
    swap_generator.manual_seed(episode_seed * KNUTH_HASH + 2)

    positions = sample_bin_positions(generator, XY_HALF, N_BINS, MIN_CENTER_DIST)

    shuffle_indices = torch.randperm(N_BINS, generator=generator).tolist()
    bin_colors = [COLOR_POOL[i] for i in shuffle_indices]

    target_indices = [i for i, c in enumerate(bin_colors) if c != "no_cube"]
    sel = torch.randperm(len(target_indices), generator=pickup_generator)[:N_PICKUPS].tolist()
    chosen_bin_indices = [target_indices[k] for k in sel]
    pickup_map = {chosen_bin_indices[0]: 0, chosen_bin_indices[1]: 1}

    swap_pairs: list[tuple[int, int]] = []
    for _ in range(SWAP_TIMES):
        idx = torch.randperm(N_BINS, generator=swap_generator)[:2].tolist()
        a, b = sorted((idx[0], idx[1]))
        swap_pairs.append((a, b))

    samples = [
        BinSample(
            episode=episode_idx,
            bin_index=i,
            x=positions[i][0],
            y=positions[i][1],
            color=bin_colors[i],
            pickup_order=pickup_map.get(i),
        )
        for i in range(N_BINS)
    ]
    swaps = EpisodeSwaps(episode=episode_idx, pairs=tuple(swap_pairs))

    button_samples: list[ButtonSample] = []
    if num_buttons > 0:
        button_generator = torch.Generator()
        button_generator.manual_seed(episode_seed * KNUTH_HASH + 3)
        button_positions = sample_button_positions(button_generator, num_buttons)
        button_samples = [
            ButtonSample(
                episode=episode_idx,
                button_index=i,
                x=button_positions[i][0],
                y=button_positions[i][1],
            )
            for i in range(num_buttons)
        ]

    return samples, swaps, button_samples


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
    # Per-episode no_cube bins are shown in bin_index ascending order.
    # samples[] is generated in (episode, bin_index) order, so the k-th no_cube
    # encountered for an episode is the slot=k panel.
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
    # Button strip axis differs from the workspace box used by other panels
    # because the strip lives at x in [-0.25, -0.15], outside [-XY_HALF, XY_HALF].
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
    row0_panels = N_COLORED_CUBES + N_NO_CUBE_BINS  # = N_BINS
    # combined + N pickups + swap_freq, plus optional buttons-only panel.
    row1_panels = 2 + N_PICKUPS + (1 if num_buttons > 0 else 0)
    n_cols = max(row0_panels, row1_panels)
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 5, 10))

    # Row 0: colored cubes (red/green/blue) + per-slot no_cube panels
    _draw_color_panel(axes[0, 0], samples, "red")
    _draw_color_panel(axes[0, 1], samples, "green")
    _draw_color_panel(axes[0, 2], samples, "blue")
    for k in range(N_NO_CUBE_BINS):
        _draw_no_cube_panel(axes[0, N_COLORED_CUBES + k], samples, k)
    for c in range(row0_panels, n_cols):
        axes[0, c].set_visible(False)

    # Row 1: combined, pickup #1..N, swap pair freq, [buttons]
    _draw_all_colors_panel(axes[1, 0], samples)
    for p in range(N_PICKUPS):
        _draw_pickup_panel(axes[1, 1 + p], samples, p)
    _draw_swap_pair_panel(axes[1, 1 + N_PICKUPS], episode_swaps_list, N_BINS)
    if num_buttons > 0:
        _draw_buttons_panel(axes[1, 2 + N_PICKUPS], button_samples)
    for c in range(row1_panels, n_cols):
        axes[1, c].set_visible(False)

    fig.suptitle(
        f"Lightweight {N_BINS}-bin demo: xy in [-{XY_HALF}, {XY_HALF}], "
        f"min_center_dist={MIN_CENTER_DIST}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=N_EPISODES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--num-buttons",
        type=int,
        default=2,
        help="If > 0, additionally sample N buttons per episode in the vertical "
        f"strip [{BUTTON_X_MIN}, {BUTTON_X_MAX}] x [{BUTTON_Y_MIN}, "
        f"{BUTTON_Y_MAX}] (matching permanence's x~-0.2 button strip; y range "
        "is the full demo workspace width so rejection sampling has no "
        f"geometric dead zone) with min center-dist {BUTTON_MIN_CENTER_DIST} "
        "(value mirrors ButtonUnmaskSwap's worst-case 0.2 - 2*0.025 = 0.15 "
        "two-button separation). Bin sampling rejects positions inside this "
        "strip so cubes never overlap buttons.",
    )
    parser.add_argument(
        "--debug-single-episode",
        type=int,
        default=None,
        help="If set, also print bin_colors and chosen_bin_indices for this seed value "
        "(useful to cross-check against permanence task internals).",
    )
    args = parser.parse_args()

    samples: list[BinSample] = []
    episode_swaps_list: list[EpisodeSwaps] = []
    button_samples_all: list[ButtonSample] = []
    failed_seeds: list[tuple[int, str]] = []
    for ep in range(args.n_episodes):
        episode_seed = args.base_seed + ep
        try:
            ep_samples, ep_swaps, ep_buttons = generate_episode(
                episode_seed, ep, args.num_buttons
            )
        except RuntimeError as exc:
            # Surface the failure loudly per CLAUDE.md, but skip this seed and
            # continue so the remaining seeds still produce a figure.
            print(f"[skip seed={episode_seed}] RuntimeError: {exc}")
            failed_seeds.append((episode_seed, str(exc)))
            continue
        samples.extend(ep_samples)
        episode_swaps_list.append(ep_swaps)
        button_samples_all.extend(ep_buttons)

    render_figure(
        samples, episode_swaps_list, button_samples_all, args.num_buttons, args.out
    )

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
            f"button samples: {len(button_samples_all)} "
            f"({n_success} succeeded episodes x {args.num_buttons} buttons)"
        )

    if args.debug_single_episode is not None:
        debug_seed = args.debug_single_episode
        debug_samples, debug_swaps, debug_buttons = generate_episode(
            debug_seed, debug_seed, args.num_buttons
        )
        debug_colors = [s.color for s in debug_samples]
        debug_chosen = [
            s.bin_index for s in sorted(debug_samples, key=lambda s: (s.pickup_order is None, s.pickup_order))
            if s.pickup_order is not None
        ]
        print(
            f"[debug seed={debug_seed}] bin_colors={debug_colors}  "
            f"chosen_bin_indices={debug_chosen}  swap_pairs={list(debug_swaps.pairs)}"
        )
        if args.num_buttons > 0:
            debug_button_xy = [(b.x, b.y) for b in debug_buttons]
            print(
                f"[debug seed={debug_seed}] button_positions={debug_button_xy}"
            )


if __name__ == "__main__":
    main()
