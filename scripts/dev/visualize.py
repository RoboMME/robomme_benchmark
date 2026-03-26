"""把 snapshot JSON 聚合成一张总览图。

默认读取 `runs/replay_videos/snapshots/*.json`，每个 JSON 在同一张总图里占一个子图。
每个子图使用俯视角 (x, y) 展示：
- `bin` 位置：方框。
- `cube` 位置：带颜色的圆点。

运行示例：
    uv run python scripts/dev/visualize.py
    uv run python scripts/dev/visualize.py --output runs/replay_videos/snapshots/overview.png
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

DEFAULT_INPUT_DIR = Path("runs/replay_videos/snapshots")
DEFAULT_OUTPUT_PATH = DEFAULT_INPUT_DIR / "overview.png"

BIN_EDGE_COLOR = "#111827"
BIN_FILLED_FACE_COLOR = "#d1d5db"
BIN_EMPTY_FACE_COLOR = "#ffffff"
FALLBACK_CUBE_COLORS = [
    "#ef4444",
    "#22c55e",
    "#3b82f6",
    "#f59e0b",
    "#a855f7",
    "#ec4899",
    "#14b8a6",
    "#f97316",
]
NAMED_CUBE_COLORS = {
    "red": "#ef4444",
    "green": "#22c55e",
    "blue": "#3b82f6",
    "yellow": "#f59e0b",
    "orange": "#f97316",
    "purple": "#a855f7",
    "pink": "#ec4899",
    "cyan": "#06b6d4",
    "brown": "#92400e",
    "gray": "#6b7280",
    "grey": "#6b7280",
    "white": "#f9fafb",
    "black": "#111827",
}


@dataclass(frozen=True)
class BinSnapshot:
    index: int
    name: str | None
    position_xyz: tuple[float, float, float]
    has_cube_under_bin: bool


@dataclass(frozen=True)
class CubeSnapshot:
    name: str | None
    color: str | None
    position_xyz: tuple[float, float, float]
    paired_bin_index: int | None
    paired_bin_name: str | None


@dataclass(frozen=True)
class SceneSnapshot:
    path: Path
    env_id: str
    episode: int
    seed: int
    difficulty: str | None
    bins: tuple[BinSnapshot, ...]
    cubes: tuple[CubeSnapshot, ...]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read snapshot JSON files and plot bin/cube positions in a single figure."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing snapshot JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path of the generated overview image.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI.",
    )
    return parser


def _position_xyz(raw_position: object, *, path: Path, field_name: str) -> tuple[float, float, float]:
    if not isinstance(raw_position, list) or len(raw_position) < 3:
        raise ValueError(f"{path}: `{field_name}` must be a list with at least 3 numbers.")
    return (float(raw_position[0]), float(raw_position[1]), float(raw_position[2]))


def _load_snapshot(path: Path) -> SceneSnapshot:
    payload = json.loads(path.read_text(encoding="utf-8"))
    env_id = str(payload["env_id"])
    bins_payload = payload.get("bins", [])
    cubes_payload = payload.get("cubes", [])

    bins = tuple(
        BinSnapshot(
            index=int(bin_item["index"]),
            name=bin_item.get("name"),
            position_xyz=_position_xyz(
                bin_item["position_xyz"],
                path=path,
                field_name="bins[].position_xyz",
            ),
            has_cube_under_bin=bool(bin_item.get("has_cube_under_bin", False)),
        )
        for bin_item in bins_payload
    )
    cubes = tuple(
        CubeSnapshot(
            name=cube_item.get("name"),
            color=cube_item.get("color"),
            position_xyz=_position_xyz(
                cube_item["position_xyz"],
                path=path,
                field_name="cubes[].position_xyz",
            ),
            paired_bin_index=(
                None
                if cube_item.get("paired_bin_index") is None
                else int(cube_item["paired_bin_index"])
            ),
            paired_bin_name=cube_item.get("paired_bin_name"),
        )
        for cube_item in cubes_payload
    )

    return SceneSnapshot(
        path=path,
        env_id=env_id,
        episode=int(payload.get("episode", 0)),
        seed=int(payload.get("seed", 0)),
        difficulty=payload.get("difficulty"),
        bins=bins,
        cubes=cubes,
    )


def _load_snapshots(snapshot_paths: list[Path]) -> list[SceneSnapshot]:
    scenes = [_load_snapshot(path) for path in sorted(snapshot_paths)]
    scenes.sort(key=lambda scene: (scene.env_id, scene.episode, scene.seed, scene.path.name))
    return scenes


def _resolve_cube_color(color_name: str | None, fallback_index: int) -> str:
    if color_name:
        lowered = color_name.strip().lower()
        if lowered in NAMED_CUBE_COLORS:
            return NAMED_CUBE_COLORS[lowered]
        if mcolors.is_color_like(color_name):
            return color_name
    return FALLBACK_CUBE_COLORS[fallback_index % len(FALLBACK_CUBE_COLORS)]


def _subplot_shape(num_plots: int) -> tuple[int, int]:
    if num_plots <= 1:
        return 1, 1
    ncols = min(3, math.ceil(math.sqrt(num_plots)))
    nrows = math.ceil(num_plots / ncols)
    return nrows, ncols


def _axis_limits(scenes: list[SceneSnapshot]) -> tuple[tuple[float, float], tuple[float, float], float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for scene in scenes:
        xs.extend(bin_item.position_xyz[0] for bin_item in scene.bins)
        ys.extend(bin_item.position_xyz[1] for bin_item in scene.bins)
        xs.extend(cube_item.position_xyz[0] for cube_item in scene.cubes)
        ys.extend(cube_item.position_xyz[1] for cube_item in scene.cubes)

    if not xs or not ys:
        return (-1.0, 1.0), (-1.0, 1.0), 0.05, 0.05

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = max(0.02, (x_max - x_min) * 0.2)
    y_pad = max(0.02, (y_max - y_min) * 0.2)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad), x_pad, y_pad


def _snapshot_text(scene: SceneSnapshot) -> str:
    lines = [
        f"env: {scene.env_id}",
        f"episode: {scene.episode}",
        f"seed: {scene.seed}",
    ]
    if scene.difficulty:
        lines.append(f"difficulty: {scene.difficulty}")
    return "\n".join(lines)


def _plot_snapshot(
    ax: Axes,
    scene: SceneSnapshot,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    x_pad: float,
    y_pad: float,
) -> None:
    label_dx = x_pad * 0.12
    label_dy = y_pad * 0.12

    encountered_colors: dict[str, str] = {}

    for bin_item in scene.bins:
        x_pos, y_pos, _ = bin_item.position_xyz
        face_color = (
            BIN_FILLED_FACE_COLOR if bin_item.has_cube_under_bin else BIN_EMPTY_FACE_COLOR
        )
        ax.scatter(
            x_pos,
            y_pos,
            s=360,
            marker="s",
            facecolors=face_color,
            edgecolors=BIN_EDGE_COLOR,
            linewidths=1.2,
            alpha=0.75,
            zorder=1,
        )
        ax.text(
            x_pos + label_dx,
            y_pos + label_dy,
            f"B{bin_item.index}",
            fontsize=7,
            color=BIN_EDGE_COLOR,
            zorder=4,
        )

    for cube_index, cube_item in enumerate(scene.cubes):
        x_pos, y_pos, _ = cube_item.position_xyz
        cube_color = _resolve_cube_color(cube_item.color, cube_index)
        label = cube_item.color or cube_item.name or f"cube_{cube_index}"
        encountered_colors[label] = cube_color
        ax.scatter(
            x_pos,
            y_pos,
            s=120,
            marker="o",
            c=[cube_color],
            edgecolors=BIN_EDGE_COLOR,
            linewidths=0.9,
            alpha=0.98,
            zorder=3,
        )

    marker_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markeredgecolor=BIN_EDGE_COLOR,
            markerfacecolor=BIN_FILLED_FACE_COLOR,
            markersize=10,
            label="bin (has cube)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markeredgecolor=BIN_EDGE_COLOR,
            markerfacecolor=BIN_EMPTY_FACE_COLOR,
            markersize=10,
            label="bin (empty)",
        ),
    ]
    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markeredgecolor=BIN_EDGE_COLOR,
            markerfacecolor=color_value,
            markersize=8,
            label=color_name,
        )
        for color_name, color_value in sorted(encountered_colors.items())
    ]
    ax.legend(
        handles=marker_handles + color_handles,
        fontsize=8,
        loc="upper left",
        frameon=True,
    )

    ax.text(
        0.98,
        0.02,
        _snapshot_text(scene),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    ax.set_title(f"{scene.env_id} | ep{scene.episode} | seed{scene.seed}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)


def _save_figure(scenes: list[SceneSnapshot], output_path: Path, dpi: int) -> None:
    nrows, ncols = _subplot_shape(len(scenes))
    x_limits, y_limits, x_pad, y_pad = _axis_limits(scenes)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6.4 * ncols, 5.8 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()

    for axis, scene in zip(axes_flat, scenes, strict=False):
        _plot_snapshot(
            axis,
            scene,
            x_limits=x_limits,
            y_limits=y_limits,
            x_pad=x_pad,
            y_pad=y_pad,
        )

    for axis in axes_flat[len(scenes) :]:
        axis.axis("off")

    fig.suptitle("Snapshot Overview: one subplot per JSON", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = _build_parser().parse_args()
    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    snapshot_paths = sorted(input_dir.glob("*.json"))
    if not snapshot_paths:
        raise SystemExit(f"No snapshot JSON files found in: {input_dir}")

    scenes = _load_snapshots(snapshot_paths)
    _save_figure(scenes, output_path=output_path, dpi=args.dpi)

    print(f"Loaded {len(snapshot_paths)} snapshot JSON files from {input_dir}")
    print(f"Saved overview figure to {output_path}")


if __name__ == "__main__":
    main()
