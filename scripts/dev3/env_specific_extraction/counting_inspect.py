"""Counting 套件（BinFill / PickXtimes / SwingXtimes / StopCube）专用 xy 渲染入口。

模板对齐 permanance_inspect.py：可被 inspect_stat.py 调用，也可独立运行。
单一公开接口：

    visualize(segmentation_dir, output_dir, env_id=None,
              difficulty_by_env_episode=None) -> (kept, skipped)

`output_dir` 直接写入 ``inspect-stat/xy/{env_id}_xy.png`` 的目标路径
（counting 没有 pickup overlay 之类的特殊处理，所以 difficulty 参数仅占位以
保持 4 个 suite 接口一致；BinFill 的 board_with_hole 当作 target 的特例
由 xy_common._panel_specs_for_env / _plot_target_objects 内部处理）。
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLBACKEND", "Agg")

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import xy_common  # noqa: E402


COUNTING_ENV_IDS: frozenset[str] = frozenset(
    {"BinFill", "PickXtimes", "SwingXtimes", "StopCube"}
)


_DEFAULT_BASE = Path("/data/hongzefu/robomme_benchmark_cvpr2026-heldoutSeed/runs/replay_videos")
DEFAULT_SEGMENTATION_DIR = _DEFAULT_BASE / "reset_segmentation_pngs"
DEFAULT_OUTPUT_DIR = _DEFAULT_BASE / "inspect-stat" / "xy"


def visualize(
    segmentation_dir: Path,
    output_dir: Path,
    *,
    env_id: Optional[str] = None,
    difficulty_by_env_episode: Optional[dict[tuple[str, int], str]] = None,
) -> tuple[list, list]:
    """渲染 Counting 套件每个 env 的 xy collage。

    Parameters
    ----------
    segmentation_dir:
        包含各 episode 子目录（内有 visible_objects.json）的根目录。
    output_dir:
        最终 PNG 写入位置（通常是 ``inspect-stat/xy``）。
    env_id:
        指定只处理哪个 env；``None`` = 处理 4 个全部。传入非 counting env
        时静默返回空列表。
    difficulty_by_env_episode:
        来自 distribution pipeline 的 (env, episode) -> difficulty 映射。
        Counting 套件的 panel 渲染本身不使用，但保留以与其余 suite 接口一致
        （也用于将来可能的扩展）。

    Returns
    -------
    (kept, skipped) : tuple[list[xy_common._VisibleObjectsFile], ...]
    """
    segmentation_dir = Path(segmentation_dir)
    output_dir = Path(output_dir)

    if env_id is not None and env_id not in COUNTING_ENV_IDS:
        return [], []

    files = xy_common._discover_visible_object_files(segmentation_dir)
    files = [
        entry
        for entry in files
        if entry.env_id in COUNTING_ENV_IDS
        and (env_id is None or entry.env_id == env_id)
    ]

    kept, skipped = xy_common._dedup_visible_object_files(files)
    if not kept:
        env_part = f" for env_id={env_id!r}" if env_id else ""
        print(
            f"[Counting-inspect] No visible_objects.json found "
            f"under {segmentation_dir}{env_part}."
        )
        return kept, skipped

    points_by_env, skipped_objects, episode_counts, _ = (
        xy_common._build_points_from_files(kept, difficulty_by_env_episode or {})
    )

    plt = xy_common._get_pyplot(show=False)
    for eid in sorted(points_by_env):
        points = points_by_env[eid]
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
        f"[Counting-inspect] envs={len(points_by_env)} "
        f"kept={len(kept)} skipped={len(skipped)} "
        f"skipped_objects={skipped_objects.get('objects', 0)}"
    )
    return kept, skipped


_VALID_ENV_IDS = sorted(COUNTING_ENV_IDS)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Counting 套件专用 xy 可视化：扫描 visible_objects.json，为 "
            "BinFill / PickXtimes / SwingXtimes / StopCube 各生成一张 1×4 "
            "PNG（all / cube / button / target），写入 inspect-stat/xy/。"
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
            f"只处理指定的 counting env，不传则处理全部四个。"
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
