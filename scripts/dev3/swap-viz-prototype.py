"""Prototype: explore better visualizations for ButtonUnmaskSwap swap pairs.

This is a stand-alone exploration script. It synthesizes dummy swap data that
mirrors the schema of `permanence_init_state.json` produced by
`scripts/dev3/env-specific-extraction/permanence.py`, then renders 5 candidate
visualizations side-by-side so we can decide which form to integrate into
`inspect-stat.py` later.

Run:
    uv run python scripts/dev3/swap-viz-prototype.py --n-episodes 90 --seed 0
"""

from __future__ import annotations

import argparse
import datetime as _dt
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle


# 与 inspect-stat.py 中 _PERMANENCE_SWAP_INDEX_COLORS 保持一致
SWAP_INDEX_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# 4 个 bin 中心（与 ButtonUnmaskSwap 截图布局一致：2x2 grid）
BIN_CENTERS = {
    0: (-0.10, -0.20),
    1: (0.10, -0.20),
    2: (0.10, 0.20),
    3: (-0.10, 0.20),
}

DIFFICULTIES = ("easy", "medium", "hard")
DIFF_COLORS = {"easy": "#bde0fe", "medium": "#ffd6a5", "hard": "#ffadad"}

# 与 inspect-stat.py:1568-1570 同一约定：俯视图顺时针旋转 90°
def rot_cw_90(x: float, y: float) -> tuple[float, float]:
    return y, -x


# ---------------------------------------------------------------------------
# Dummy data
# ---------------------------------------------------------------------------


@dataclass
class DummyEpisode:
    episode: int
    difficulty: str
    bins: list[dict]
    swap_pairs: list[dict]


def _jitter(rng: random.Random, scale: float = 0.02) -> float:
    return rng.uniform(-scale, scale)


def _bins_for_episode(rng: random.Random) -> list[dict]:
    bins = []
    for idx, (cx, cy) in BIN_CENTERS.items():
        bins.append(
            {
                "index": idx,
                "name": f"bin_{idx}",
                "position_xy": [cx + _jitter(rng), cy + _jitter(rng)],
            }
        )
    return bins


def _pick_pair(rng: random.Random, difficulty: str) -> tuple[int, int]:
    """Slight difficulty-based bias to make patterns visible.

    - easy:   uniform random among adjacent slots (sides of the square)
    - medium: uniform random among any pair
    - hard:   biased towards diagonals (across the square)
    """
    sides = [(0, 1), (1, 2), (2, 3), (3, 0)]
    diagonals = [(0, 2), (1, 3)]
    all_pairs = sides + diagonals
    if difficulty == "easy":
        a, b = rng.choice(sides)
    elif difficulty == "hard":
        a, b = rng.choice(diagonals if rng.random() < 0.7 else all_pairs)
    else:
        a, b = rng.choice(all_pairs)
    if rng.random() < 0.5:
        a, b = b, a
    return a, b


def make_dummy_episodes(n_episodes: int, seed: int) -> list[DummyEpisode]:
    rng = random.Random(seed)
    out: list[DummyEpisode] = []
    for ep in range(n_episodes):
        difficulty = DIFFICULTIES[ep % len(DIFFICULTIES)]
        bins = _bins_for_episode(rng)
        swap_times = {"easy": 1, "medium": 2, "hard": 3}[difficulty]
        # step windows tile [50, 250]
        window_edges = [50 + i * (200 // swap_times) for i in range(swap_times + 1)]
        pairs: list[dict] = []
        for k in range(swap_times):
            a_idx, b_idx = _pick_pair(rng, difficulty)
            a_xy = bins[a_idx]["position_xy"]
            b_xy = bins[b_idx]["position_xy"]
            pairs.append(
                {
                    "swap_index": k,
                    "step_window": [window_edges[k], window_edges[k + 1]],
                    "bin_a_index": a_idx,
                    "bin_a_name": f"bin_{a_idx}",
                    "bin_a_position_xy": list(a_xy),
                    "bin_b_index": b_idx,
                    "bin_b_name": f"bin_{b_idx}",
                    "bin_b_position_xy": list(b_xy),
                }
            )
        out.append(
            DummyEpisode(
                episode=ep,
                difficulty=difficulty,
                bins=bins,
                swap_pairs=pairs,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Shared axis helpers
# ---------------------------------------------------------------------------


def _setup_xy_axis(ax, title: str, points: int) -> None:
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("World Y")
    ax.set_ylabel("-World X")
    ax.set_title(f"{title}\npoints={points}", fontsize=10)


def _all_pair_count(eps: list[DummyEpisode]) -> Counter:
    """Undirected (i,j) pair counts across all episodes."""
    c: Counter = Counter()
    for e in eps:
        for p in e.swap_pairs:
            i, j = sorted([p["bin_a_index"], p["bin_b_index"]])
            c[(i, j)] += 1
    return c


# ---------------------------------------------------------------------------
# Candidate A — baseline (replicates current inspect-stat.py)
# ---------------------------------------------------------------------------


def candidate_A_baseline(ax, eps: list[DummyEpisode]) -> None:
    pair_count = 0
    seen_idx: set[int] = set()
    for e in eps:
        for b in e.bins:
            bx, by = rot_cw_90(*b["position_xy"])
            ax.scatter(bx, by, s=20, color="lightgray", alpha=0.4, zorder=1)
        for p in e.swap_pairs:
            si = p["swap_index"]
            seen_idx.add(si)
            color = SWAP_INDEX_COLORS[si % len(SWAP_INDEX_COLORS)]
            ax_x, ax_y = rot_cw_90(*p["bin_a_position_xy"])
            bx_x, bx_y = rot_cw_90(*p["bin_b_position_xy"])
            ax.annotate(
                "",
                xy=(bx_x, bx_y),
                xytext=(ax_x, ax_y),
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
            ax.text(
                (ax_x + bx_x) / 2,
                (ax_y + bx_y) / 2,
                f"ep{e.episode}#s{si}",
                fontsize=5, alpha=0.6,
            )
            pair_count += 1

    handles = [
        Line2D([0], [0], marker="o", linestyle="-",
               color=SWAP_INDEX_COLORS[i % len(SWAP_INDEX_COLORS)],
               markersize=8, label=f"swap #{i}")
        for i in sorted(seen_idx)
    ]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=7)
    _setup_xy_axis(ax, "A) Baseline (overlay arrows)", pair_count)


# ---------------------------------------------------------------------------
# Candidate B — bin-pair frequency heat matrix
# ---------------------------------------------------------------------------


def candidate_B_heatmap(ax, eps: list[DummyEpisode]) -> None:
    n = len(BIN_CENTERS)
    counts = _all_pair_count(eps)
    mat = [[0] * n for _ in range(n)]
    for (i, j), c in counts.items():
        mat[i][j] = c
        mat[j][i] = c

    arr = [[mat[i][j] for j in range(n)] for i in range(n)]
    im = ax.imshow(arr, cmap="YlOrRd", origin="lower")
    for i in range(n):
        for j in range(n):
            v = mat[i][j]
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color="gray", fontsize=10)
            else:
                color = "white" if v > max(max(r) for r in arr) * 0.6 else "black"
                ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=10)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"bin_{i}" for i in range(n)])
    ax.set_yticklabels([f"bin_{i}" for i in range(n)])
    ax.set_title(f"B) Pair frequency matrix\ntotal swaps={sum(counts.values())}",
                 fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ---------------------------------------------------------------------------
# Candidate C — aggregated flow on bin layout
# ---------------------------------------------------------------------------


def candidate_C_aggregated(ax, eps: list[DummyEpisode]) -> None:
    counts = _all_pair_count(eps)
    total = sum(counts.values())
    # bin participation degree (sum over pairs touching this bin)
    deg: Counter = Counter()
    for (i, j), c in counts.items():
        deg[i] += c
        deg[j] += c

    # node positions (rotated frame)
    node_xy = {i: rot_cw_90(*c) for i, c in BIN_CENTERS.items()}

    max_count = max(counts.values()) if counts else 1
    for (i, j), c in counts.items():
        x0, y0 = node_xy[i]
        x1, y1 = node_xy[j]
        lw = 0.8 + 5.0 * c / max_count
        # Curve to disambiguate between pairs sharing endpoints
        rad = 0.18 if (i + j) % 2 == 0 else -0.18
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-",
            connectionstyle=f"arc3,rad={rad}",
            color="#444",
            lw=lw,
            alpha=0.55,
            zorder=2,
        )
        ax.add_patch(arrow)
        # label count near midpoint
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my, str(c), fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
                zorder=3)

    max_deg = max(deg.values()) if deg else 1
    for i, (x, y) in node_xy.items():
        size = 200 + 600 * deg[i] / max_deg
        ax.scatter(x, y, s=size, color="#1f77b4", edgecolors="black",
                   linewidths=1.2, alpha=0.9, zorder=4)
        ax.text(x, y, f"bin_{i}\n({deg[i]})", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold", zorder=5)

    _setup_xy_axis(ax, "C) Aggregated flow\n(line width = freq)", total)


# ---------------------------------------------------------------------------
# Candidate D — facet by swap_index
# ---------------------------------------------------------------------------


def candidate_D_by_swap_index(fig, gridspec, eps: list[DummyEpisode]) -> None:
    """Returns a list of axes; each axis = one swap_index slot."""
    max_k = max((len(e.swap_pairs) for e in eps), default=1)
    axes = []
    for k in range(max_k):
        ax = fig.add_subplot(gridspec[k])
        axes.append(ax)
        # background bin slots
        for i, c in BIN_CENTERS.items():
            bx, by = rot_cw_90(*c)
            ax.scatter(bx, by, s=120, color="lightgray", edgecolors="gray",
                       linewidths=0.8, zorder=1)
            ax.text(bx, by, f"bin_{i}", fontsize=7, ha="center", va="center", zorder=2)

        cnt = 0
        for e in eps:
            for p in e.swap_pairs:
                if p["swap_index"] != k:
                    continue
                ax_x, ax_y = rot_cw_90(*p["bin_a_position_xy"])
                bx_x, bx_y = rot_cw_90(*p["bin_b_position_xy"])
                color = SWAP_INDEX_COLORS[k % len(SWAP_INDEX_COLORS)]
                ax.annotate(
                    "",
                    xy=(bx_x, bx_y),
                    xytext=(ax_x, ax_y),
                    arrowprops=dict(
                        arrowstyle="<->", color=color, lw=1.0, alpha=0.35,
                        shrinkA=3, shrinkB=3,
                    ),
                    zorder=3,
                )
                cnt += 1
        _setup_xy_axis(ax, f"D{k}) swap_index = {k}", cnt)
    return axes


# ---------------------------------------------------------------------------
# Candidate E — temporal swim lane (episode × step)
# ---------------------------------------------------------------------------


def candidate_E_swimlane(ax, eps: list[DummyEpisode]) -> None:
    # sort episodes by difficulty
    ordered = sorted(eps, key=lambda e: (DIFFICULTIES.index(e.difficulty), e.episode))
    # difficulty bands
    diff_runs: list[tuple[str, int, int]] = []
    if ordered:
        cur = ordered[0].difficulty
        start = 0
        for i, e in enumerate(ordered):
            if e.difficulty != cur:
                diff_runs.append((cur, start, i - 1))
                cur = e.difficulty
                start = i
        diff_runs.append((cur, start, len(ordered) - 1))

    y_max = 270
    for d, lo, hi in diff_runs:
        ax.add_patch(Rectangle((lo - 0.4, 0), (hi - lo + 0.8), y_max,
                               facecolor=DIFF_COLORS[d], alpha=0.35, zorder=0))
        ax.text((lo + hi) / 2, y_max - 10, d,
                ha="center", va="top", fontsize=9, color="#333", zorder=1)

    # per-pair color
    pair_color: dict[tuple[int, int], str] = {}
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for idx, pair in enumerate(sorted({tuple(sorted([p["bin_a_index"], p["bin_b_index"]]))
                                       for e in eps for p in e.swap_pairs})):
        pair_color[pair] = palette[idx % len(palette)]

    cnt = 0
    for col, e in enumerate(ordered):
        for p in e.swap_pairs:
            pair = tuple(sorted([p["bin_a_index"], p["bin_b_index"]]))
            color = pair_color[pair]
            s0, s1 = p["step_window"]
            ax.plot([col, col], [s0, s1], color=color, lw=2.2, alpha=0.9, zorder=2)
            cnt += 1

    handles = [
        Line2D([0], [0], color=c, lw=3, label=f"bin_{i}↔bin_{j}")
        for (i, j), c in pair_color.items()
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6, ncol=2)
    ax.set_xlim(-0.5, len(ordered) - 0.5)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("episode (sorted by difficulty)")
    ax.set_ylabel("step")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(f"E) Temporal swim lane\nsegments={cnt}", fontsize=10)


# ---------------------------------------------------------------------------
# Compose figure
# ---------------------------------------------------------------------------


def render_all(eps: list[DummyEpisode], out_path: Path, env_id: str) -> None:
    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 2, wspace=0.32)

    ax_a = fig.add_subplot(gs[0])
    candidate_A_baseline(ax_a, eps)

    ax_b = fig.add_subplot(gs[1])
    candidate_B_heatmap(ax_b, eps)

    fig.suptitle(
        f"{env_id} dummy | episodes={len(eps)} | "
        f"total swaps={sum(len(e.swap_pairs) for e in eps)}",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env-id", default="ButtonUnmaskSwap")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "runs"
        / "swap-viz-prototype",
    )
    args = ap.parse_args()

    eps = make_dummy_episodes(args.n_episodes, args.seed)
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "candidates.png"
    render_all(eps, out_path, args.env_id)
    print(f"[swap-viz-prototype] wrote {out_path}")
    print(f"  episodes={len(eps)}  swaps={sum(len(e.swap_pairs) for e in eps)}")
    print(
        "  difficulties=",
        Counter(e.difficulty for e in eps),
    )


if __name__ == "__main__":
    main()
