"""Permanence 套件（VideoUnmask / VideoUnmaskSwap / ButtonUnmask / ButtonUnmaskSwap）
专用可视化入口。

可被 inspect_stat.py 调用，也可独立运行。对外只暴露一个公开接口：
    visualize(segmentation_dir, output_dir, env_id=None)

调用一次 visualize() 直接产出完整的 2 行 PNG 到 inspect-stat/xy/{env_id}_xy.png：
- 第 1 行：visible-objects 4 个面板（与 inspect_stat 普通 xy 图保持一致）
- 第 2 行：permanence cubes + permanence swaps 双面板

permanance_inspect/ 目录不再被创建——`output_dir` 仅作 anchor，用来定位
inspect-stat 根目录下的 xy/ 子目录。

permanence-specific 的两个面板绘制逻辑统一收敛在
`scripts/dev3/env_specific_extraction/permanence.py` 中
（plot_permanence_cubes_panel / plot_permanence_swaps_panel）。
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
    permanence_module.plot_permanence_cubes_panel(axes[1, 0], env_id, perm_files)
    permanence_module.plot_permanence_swaps_panel(axes[1, 1], env_id, perm_files)
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
