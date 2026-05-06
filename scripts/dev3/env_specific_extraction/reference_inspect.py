"""Reference 套件（PickHighlight / VideoRepick / VideoPlaceButton / VideoPlaceOrder）
专用 xy 渲染入口。

模板对齐 permanance_inspect.py：可被 inspect_stat.py 调用，也可独立运行。
单一公开接口：

    visualize(segmentation_dir, output_dir, env_id=None,
              difficulty_by_env_episode=None, snapshot_dir=None) -> (kept, skipped)

VideoRepick 比较特殊，需要：
- 按 difficulty 拆成 easy_medium 和 hard 两张 PNG（由 xy_common._render_xy_env 内部
  根据 env_id == VIDEOREPICK_ENV_ID 自动走 split 逻辑）；
- 在拆分后右侧 panel 叠加 pickup cube 散点（来自 snapshot_dir 下的
  ``<env_id>_ep<n>_seed<s>_after_no_record_reset.json``）。
所以 visualize() 接受 ``snapshot_dir`` + ``difficulty_by_env_episode`` 两个 kwarg。
其余 3 个 reference env（PickHighlight / VideoPlaceButton / VideoPlaceOrder）走标准
1×4 collage（all / cube / button / target），与 counting 完全一致。
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


REFERENCE_ENV_IDS: frozenset[str] = frozenset(
    {"PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"}
)


_DEFAULT_BASE = Path("/data/hongzefu/robomme_benchmark_cvpr2026-heldoutSeed/runs/replay_videos")
DEFAULT_SEGMENTATION_DIR = _DEFAULT_BASE / "reset_segmentation_pngs"
DEFAULT_OUTPUT_DIR = _DEFAULT_BASE / "inspect-stat" / "xy"
DEFAULT_SNAPSHOT_DIR = _DEFAULT_BASE / xy_common.SNAPSHOT_DIRNAME


def visualize(
    segmentation_dir: Path,
    output_dir: Path,
    *,
    env_id: Optional[str] = None,
    difficulty_by_env_episode: Optional[dict[tuple[str, int], str]] = None,
    snapshot_dir: Optional[Path] = None,
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
        用于把 VideoRepick 拆成 easy_medium / hard 两张 PNG，以及决定每条
        pickup snapshot 走哪一张 PNG。
    snapshot_dir:
        包含 ``<env_id>_ep<n>_seed<s>_after_no_record_reset.json`` 的目录。
        仅 VideoRepick 使用；其他 env 即使路径有效也不会读取。

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

    difficulty_map = difficulty_by_env_episode or {}
    points_by_env, skipped_objects, episode_counts, missing_difficulty = (
        xy_common._build_points_from_files(kept, difficulty_map)
    )

    if missing_difficulty:
        unique_missing = sorted(set(missing_difficulty))
        print(
            f"[Reference-inspect] [Warn] {len(unique_missing)} episodes have "
            f"visible_objects.json but no matching HDF5 difficulty — "
            f"VideoRepick split routing will skip them:"
        )
        for eid, episode in unique_missing:
            print(f"    - env={eid} ep={episode}")

    # VideoRepick pickup overlay：snapshot_dir 仅在 VideoRepick 在范围内时才有意义
    pickup_by_env: dict[str, list] = {}
    if snapshot_dir is not None:
        snapshot_dir_path = Path(snapshot_dir)
        if snapshot_dir_path.is_dir():
            # 只为 VideoRepick 加载 pickup（filter 用 env_id 时直接限定，
            # 否则限定到 VideoRepick env_id 上以避免读其他 env 的 snapshot）
            effective_pickup_filter = (
                env_id
                if env_id is not None
                else xy_common.VIDEOREPICK_ENV_ID
            )
            all_pickup = xy_common._discover_pickup_snapshot_records(
                snapshot_dir_path, effective_pickup_filter, difficulty_map
            )
            for rec in xy_common._dedup_pickup_records(all_pickup):
                pickup_by_env.setdefault(rec.env_id, []).append(rec)
            print(
                f"  Snapshot dir:     {snapshot_dir_path} "
                f"(pickup records loaded: "
                f"{sum(len(v) for v in pickup_by_env.values())})"
            )
        else:
            print(
                f"[Reference-inspect] [Warn] Snapshot dir not found: "
                f"{snapshot_dir_path} — VideoRepick pickup overlay will be skipped."
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
            pickup_records=pickup_by_env.get(eid),
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
            "Reference 套件专用 xy 可视化：扫描 visible_objects.json，为 "
            "PickHighlight / VideoRepick / VideoPlaceButton / VideoPlaceOrder "
            "渲染 xy collage。VideoRepick 会按 difficulty 拆成 easy_medium 与 "
            "hard 两张 PNG，并叠加 pickup cube 散点。"
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
        "--snapshot-dir",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR,
        help="VideoRepick pickup snapshot 所在目录。",
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
    snapshot_dir = args.snapshot_dir.resolve()

    print(f"Segmentation dir: {segmentation_dir}")
    print(f"Output dir:       {output_dir}")
    print(f"Snapshot dir:     {snapshot_dir}")
    if args.env_id:
        print(f"Env filter:       {args.env_id}")
    print()

    kept, skipped = visualize(
        segmentation_dir=segmentation_dir,
        output_dir=output_dir,
        env_id=args.env_id,
        snapshot_dir=snapshot_dir,
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
