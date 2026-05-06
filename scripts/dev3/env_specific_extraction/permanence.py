"""Permanence 套件 reset 时刻 cube/swap 状态的提取、序列化与渲染。

适用 env：ButtonUnmask / ButtonUnmaskSwap / VideoUnmask / VideoUnmaskSwap。

为什么独立成模块：
- Env-rollout-parallel-segmentation.py 已经超过 2000 行，permanence 相关逻辑
  (从 env 读属性 / 写 JSON / 做可视化) 不应继续往主脚本里堆。
- 同时 inspect-stat.py 也需要消费同一份 sidecar 数据；把读写两端的契约
  集中在一个文件里能让 schema 始终一致。

env 文件本身不做任何修改：本模块直接从 env.unwrapped 读 _load_scene 已经
设置好的属性 (color_names / spawned_bins / cube_bin_pairs / swap_schedule
/ target_cube_<color>)。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

PERMANENCE_ENV_IDS: frozenset[str] = frozenset(
    {"ButtonUnmask", "ButtonUnmaskSwap", "VideoUnmask", "VideoUnmaskSwap"}
)
SWAP_ENV_IDS: frozenset[str] = frozenset({"ButtonUnmaskSwap", "VideoUnmaskSwap"})

PERMANENCE_JSON_FILENAME = "permanence_init_state.json"

# 与 env 内部 cube_colors 常量保持一致（red/green/blue 对应 RGBA）
COLOR_NAME_TO_RGBA: dict[str, tuple[float, float, float, float]] = {
    "red": (1.0, 0.0, 0.0, 1.0),
    "green": (0.0, 1.0, 0.0, 1.0),
    "blue": (0.0, 0.0, 1.0, 1.0),
}

# 与 inspect-stat.py / Env-rollout 保持一致的目录命名 regex
_DIR_NAME_PATTERN = re.compile(
    r"^(?P<env_id>.+?)_ep(?P<episode>\d+)_seed(?P<seed>\d+)$"
)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _to_jsonable(value: Any) -> Any:
    """递归把 numpy / torch 张量转成 JSON 可序列化的 Python 原生类型。"""
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    # torch.Tensor 不强行 import，靠 duck-typing
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy().tolist()
    return value


def _actor_xy(actor: Any) -> list[float]:
    """从 actor.pose.p 提取 (x, y)。pose.p 形状通常为 [B, 3] 或 [3]。"""
    if actor is None:
        raise ValueError("cannot read pose from None actor")
    pose = getattr(actor, "pose", None)
    if pose is None:
        raise ValueError(f"actor {actor!r} missing .pose attribute")
    p = pose.p
    if hasattr(p, "detach"):
        p = p.detach()
    if hasattr(p, "cpu"):
        p = p.cpu()
    arr = np.asarray(p)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.ndim != 1 or arr.shape[0] < 2:
        raise ValueError(f"unexpected pose.p shape {arr.shape} for actor {actor!r}")
    return [float(arr[0]), float(arr[1])]


def _actor_name(actor: Any) -> str:
    name = getattr(actor, "name", None)
    return str(name) if name else f"<unnamed:{id(actor)}>"


def _index_in_bins(bin_actor: Any, spawned_bins: list[Any]) -> Optional[int]:
    """反查 bin_actor 在 spawned_bins 里的下标。优先 is，再按 name 比对。"""
    for i, candidate in enumerate(spawned_bins):
        if candidate is bin_actor:
            return i
    target_name = getattr(bin_actor, "name", None)
    if target_name is not None:
        for i, candidate in enumerate(spawned_bins):
            if getattr(candidate, "name", None) == target_name:
                return i
    return None


# ---------------------------------------------------------------------------
# 提取与写入
# ---------------------------------------------------------------------------


def extract_permanence_init_state(env: Any) -> Optional[dict]:
    """从 env 实例提取 permanence 套件的 init 状态 dict。

    非 Permanence env 直接返回 None；Permanence env 缺关键属性按
    no-silent-fallbacks 原则直接 raise。
    """
    base = getattr(env, "unwrapped", env)
    env_class_name = type(base).__name__
    if env_class_name not in PERMANENCE_ENV_IDS:
        return None

    color_names = list(getattr(base, "color_names", []))
    spawned_bins = list(getattr(base, "spawned_bins", []))
    if not color_names:
        raise ValueError(f"{env_class_name}.color_names is empty after reset")
    if not spawned_bins:
        raise ValueError(f"{env_class_name}.spawned_bins is empty after reset")

    cubes_payload: list[dict] = []
    cube_bin_pairs = getattr(base, "cube_bin_pairs", None)
    if cube_bin_pairs:
        # *Swap：env 已经把 (cube, bin) 配好对，颜色按 color_names 顺序对齐
        for i, (cube_actor, bin_actor) in enumerate(cube_bin_pairs):
            color_name = color_names[i] if i < len(color_names) else "unknown"
            bin_idx = _index_in_bins(bin_actor, spawned_bins)
            cubes_payload.append(
                {
                    "name": _actor_name(cube_actor),
                    "color_name": color_name,
                    "color_rgba": list(COLOR_NAME_TO_RGBA.get(color_name, (1.0, 1.0, 1.0, 1.0))),
                    "position_xy": _actor_xy(cube_actor),
                    "bin_index": bin_idx,
                    "bin_name": _actor_name(bin_actor),
                    "bin_position_xy": _actor_xy(bin_actor),
                }
            )
    else:
        # 非 Swap：cube i 与 spawned_bins[i] 一一对应（i in 0..min(3, len(bins))-1）
        n = min(3, len(spawned_bins), len(color_names))
        for i in range(n):
            color_name = color_names[i]
            cube_actor = getattr(base, f"target_cube_{color_name}", None)
            if cube_actor is None:
                cube_actor = getattr(base, f"target_cube_{i}", None)
            if cube_actor is None:
                raise ValueError(
                    f"{env_class_name}: missing target_cube_{color_name} / target_cube_{i}"
                )
            bin_actor = spawned_bins[i]
            cubes_payload.append(
                {
                    "name": _actor_name(cube_actor),
                    "color_name": color_name,
                    "color_rgba": list(COLOR_NAME_TO_RGBA.get(color_name, (1.0, 1.0, 1.0, 1.0))),
                    "position_xy": _actor_xy(cube_actor),
                    "bin_index": i,
                    "bin_name": _actor_name(bin_actor),
                    "bin_position_xy": _actor_xy(bin_actor),
                }
            )

    swap_pairs_payload: list[dict] = []
    if env_class_name in SWAP_ENV_IDS:
        swap_schedule = getattr(base, "swap_schedule", None) or []
        for swap_idx, entry in enumerate(swap_schedule):
            if not isinstance(entry, (tuple, list)) or len(entry) < 4:
                continue
            bin_a, bin_b, start_step, end_step = entry[0], entry[1], entry[2], entry[3]
            if bin_a is None or bin_b is None:
                continue
            swap_pairs_payload.append(
                {
                    "swap_index": swap_idx,
                    "step_window": [int(start_step), int(end_step)],
                    "bin_a_index": _index_in_bins(bin_a, spawned_bins),
                    "bin_a_name": _actor_name(bin_a),
                    "bin_a_position_xy": _actor_xy(bin_a),
                    "bin_b_index": _index_in_bins(bin_b, spawned_bins),
                    "bin_b_name": _actor_name(bin_b),
                    "bin_b_position_xy": _actor_xy(bin_b),
                }
            )

    bins_payload: list[dict] = [
        {
            "index": i,
            "name": _actor_name(bin_actor),
            "position_xy": _actor_xy(bin_actor),
        }
        for i, bin_actor in enumerate(spawned_bins)
    ]

    state = {
        "env_id": env_class_name,
        "seed": int(getattr(base, "seed", -1)),
        "difficulty": str(getattr(base, "difficulty", "")),
        "color_names": list(color_names),
        "bins": bins_payload,
        "cubes": cubes_payload,
        "swap_pairs": swap_pairs_payload,
    }
    swap_times = getattr(base, "swap_times", None)
    if swap_times is not None:
        state["swap_times"] = int(swap_times)
    return _to_jsonable(state)


def write_permanence_init_state(
    env: Any,
    env_id: str,
    episode_idx: int,
    seed: int,
    reset_output_dir: Path,
) -> Optional[Path]:
    """提取 permanence init state 并写入 sidecar JSON。非 Permanence env 返回 None。"""
    state = extract_permanence_init_state(env)
    if state is None:
        return None

    payload = {
        "env_id": env_id,
        "episode": int(episode_idx),
        "seed": int(seed),
        **state,
    }
    # extract 已经填了 env_id（来自 class name），这里以传入的 env_id 为准
    payload["env_id"] = env_id

    reset_output_dir = Path(reset_output_dir)
    reset_output_dir.mkdir(parents=True, exist_ok=True)
    out_path = reset_output_dir / PERMANENCE_JSON_FILENAME
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return out_path


# ---------------------------------------------------------------------------
# 加载与发现（inspect-stat 端）
# ---------------------------------------------------------------------------


@dataclass
class PermanenceFile:
    path: Path
    env_id: str
    episode: int
    seed: int
    payload: dict


def load_permanence_init_state(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_permanence_files(
    segmentation_dir: Path,
    env_filter: Optional[str] = None,
) -> list[PermanenceFile]:
    """递归扫描 segmentation_dir 下所有 permanence_init_state.json。

    - 用 payload 的 env_id/episode/seed 作为权威来源；目录名解析仅作 sanity check。
    - env_filter 非空时过滤 env_id（精确匹配）。
    - 文件结构异常用 [Warn] 打印并跳过，与 inspect-stat 现有风格一致。
    """
    seg_dir = Path(segmentation_dir)
    if not seg_dir.is_dir():
        return []

    results: list[PermanenceFile] = []
    for json_path in sorted(seg_dir.rglob(PERMANENCE_JSON_FILENAME)):
        try:
            payload = load_permanence_init_state(json_path)
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"[Warn] Skip invalid permanence JSON {json_path}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        env_id = payload.get("env_id")
        episode = payload.get("episode")
        seed = payload.get("seed")
        if not isinstance(env_id, str) or not isinstance(episode, int) or not isinstance(seed, int):
            print(
                f"[Warn] Skip permanence JSON {json_path} missing env_id/episode/seed"
            )
            continue

        if env_filter is not None and env_id != env_filter:
            continue

        results.append(
            PermanenceFile(
                path=json_path,
                env_id=env_id,
                episode=episode,
                seed=seed,
                payload=payload,
            )
        )
    return results


def dedup_permanence_files(
    entries: list[PermanenceFile],
) -> tuple[list[PermanenceFile], list[PermanenceFile]]:
    """同 (env_id, episode) 多 seed 时只保留 max seed，与 visible_objects 一致。"""
    grouped: dict[tuple[str, int], list[PermanenceFile]] = {}
    for entry in entries:
        grouped.setdefault((entry.env_id, entry.episode), []).append(entry)
    kept: list[PermanenceFile] = []
    skipped: list[PermanenceFile] = []
    for bucket in grouped.values():
        bucket_sorted = sorted(bucket, key=lambda e: e.seed)
        kept.append(bucket_sorted[-1])
        skipped.extend(bucket_sorted[:-1])
    kept.sort(key=lambda e: (e.env_id, e.episode))
    skipped.sort(key=lambda e: (e.env_id, e.episode, e.seed))
    return kept, skipped


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

# 桌面可视范围与 inspect-stat / Env-rollout 的俯视图保持一致
_PLOT_LIMIT = 0.3
_DIFFICULTY_MARKERS: dict[str, str] = {
    "easy": "o",
    "medium": "s",
    "hard": "^",
}


def _xy_rot_cw_90(x: float, y: float) -> tuple[float, float]:
    """与 Env-rollout._xy_rot_cw_90 一致：俯视图顺时针旋转 90° -> (y, -x)。"""
    return y, -x


def _setup_xy_axis(ax: Any, title: str) -> None:
    ax.set_xlim(-_PLOT_LIMIT, _PLOT_LIMIT)
    ax.set_ylim(-_PLOT_LIMIT, _PLOT_LIMIT)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World Y")
    ax.set_ylabel("-World X")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)


def render_cubes_figure(
    env_id: str,
    files: Iterable[PermanenceFile],
    output_path: Path,
) -> None:
    """渲染该 env 所有 episode 的 cube reset 位置散点图。"""
    import matplotlib.pyplot as plt

    files = list(files)

    fig, ax = plt.subplots(figsize=(8, 8))
    _setup_xy_axis(
        ax,
        f"Permanence: cube positions @ reset\n{env_id} (n_episodes={len(files)})",
    )

    seen_difficulties: set[str] = set()
    seen_colors: set[str] = set()

    for entry in files:
        difficulty = str(entry.payload.get("difficulty", ""))
        marker = _DIFFICULTY_MARKERS.get(difficulty, "x")
        seen_difficulties.add(difficulty)

        for cube in entry.payload.get("cubes", []):
            color_name = cube.get("color_name", "unknown")
            seen_colors.add(color_name)
            rgba = COLOR_NAME_TO_RGBA.get(color_name, (0.5, 0.5, 0.5, 1.0))
            pos = cube.get("position_xy") or [0.0, 0.0]
            bin_pos = cube.get("bin_position_xy") or pos
            x, y = _xy_rot_cw_90(float(pos[0]), float(pos[1]))
            bx, by = _xy_rot_cw_90(float(bin_pos[0]), float(bin_pos[1]))

            # 用细线把 cube 连到它对应的 bin，便于观察 cube↔bin 关联
            ax.plot([x, bx], [y, by], color=rgba, alpha=0.25, linewidth=0.8)
            ax.scatter(
                x, y,
                s=70, color=rgba, marker=marker,
                edgecolors="black", linewidths=0.6, alpha=0.9,
            )
            bin_idx = cube.get("bin_index")
            if bin_idx is not None:
                ax.text(
                    x + 0.005, y + 0.005, f"ep{entry.episode}/b{bin_idx}",
                    fontsize=6, alpha=0.6,
                )

    # 自定义 legend：颜色对应 cube color，marker 对应 difficulty
    legend_handles = []
    for color_name in sorted(seen_colors):
        rgba = COLOR_NAME_TO_RGBA.get(color_name, (0.5, 0.5, 0.5, 1.0))
        legend_handles.append(
            plt.Line2D(
                [0], [0], marker="o", linestyle="", markersize=8,
                markerfacecolor=rgba, markeredgecolor="black",
                label=f"cube: {color_name}",
            )
        )
    for diff in sorted(seen_difficulties):
        marker = _DIFFICULTY_MARKERS.get(diff, "x")
        legend_handles.append(
            plt.Line2D(
                [0], [0], marker=marker, linestyle="", markersize=8,
                markerfacecolor="white", markeredgecolor="black",
                label=f"difficulty: {diff or 'unknown'}",
            )
        )
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def render_swaps_figure(
    env_id: str,
    files: Iterable[PermanenceFile],
    output_path: Path,
) -> None:
    """渲染该 env 所有 episode 的 swap pair 连线图。

    每个 swap pair 用一条带箭头的线把两个 bin 中心连起来；不同 swap_index
    用不同颜色（按 step_window 时间先后排序），episode 之间叠加在同一张图上。
    """
    import matplotlib.pyplot as plt

    files = [f for f in files if f.payload.get("swap_pairs")]
    if not files:
        # 没有 swap pair（非 Swap 环境或所有 episode 的 swap 都为空）则不出图
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    _setup_xy_axis(
        ax,
        f"Permanence: swap pairs @ reset\n{env_id} (n_episodes={len(files)})",
    )

    swap_index_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    seen_swap_indices: set[int] = set()

    for entry in files:
        # 同时把所有 bin 画成淡灰色背景点，方便看 swap 端点位置
        for bin_info in entry.payload.get("bins", []):
            bx, by = _xy_rot_cw_90(*bin_info["position_xy"])
            ax.scatter(bx, by, s=20, color="lightgray", alpha=0.4, zorder=1)

        for pair in entry.payload.get("swap_pairs", []):
            swap_idx = int(pair.get("swap_index", 0))
            seen_swap_indices.add(swap_idx)
            color = swap_index_colors[swap_idx % len(swap_index_colors)]
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
                s=60, color=color, edgecolors="black", linewidths=0.5,
                alpha=0.9, zorder=4,
            )
            mid_x = (ax_x + bx_x) / 2
            mid_y = (ax_y + bx_y) / 2
            ax.text(
                mid_x, mid_y, f"ep{entry.episode}#s{swap_idx}",
                fontsize=6, alpha=0.7,
            )

    legend_handles = []
    for swap_idx in sorted(seen_swap_indices):
        color = swap_index_colors[swap_idx % len(swap_index_colors)]
        legend_handles.append(
            plt.Line2D(
                [0], [0], marker="o", linestyle="-", color=color, markersize=8,
                label=f"swap #{swap_idx}",
            )
        )
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_permanence_pipeline(
    segmentation_dir: Path,
    output_dir: Path,
    env_filter: Optional[str] = None,
) -> tuple[list[PermanenceFile], list[PermanenceFile]]:
    """inspect-stat 端入口：扫描 sidecar，按 env 渲染 cubes / swaps 两张图。

    返回 (kept, skipped) 用于和 visible_objects / hdf5 的去重报告共显示。
    """
    out_dir = Path(output_dir) / "permanence"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = discover_permanence_files(segmentation_dir, env_filter=env_filter)
    if not files:
        env_part = f" for env={env_filter!r}" if env_filter else ""
        print(f"[Permanence] No permanence_init_state.json found under {segmentation_dir}{env_part}.")
        return [], []

    kept, skipped = dedup_permanence_files(files)

    by_env: dict[str, list[PermanenceFile]] = {}
    for entry in kept:
        by_env.setdefault(entry.env_id, []).append(entry)

    for env_id, env_files in sorted(by_env.items()):
        env_files.sort(key=lambda e: e.episode)
        cubes_path = out_dir / f"{env_id}_cubes.png"
        render_cubes_figure(env_id, env_files, cubes_path)
        print(f"[Permanence] Cubes plot written: {cubes_path}")

        if env_id in SWAP_ENV_IDS:
            swaps_path = out_dir / f"{env_id}_swaps.png"
            # 注意：render_swaps_figure 内部判定无 swap pair 时不出图
            render_swaps_figure(env_id, env_files, swaps_path)
            if swaps_path.is_file():
                print(f"[Permanence] Swaps plot written: {swaps_path}")
            else:
                print(f"[Permanence] No swap pairs found for {env_id}; skip swaps plot.")

    return kept, skipped


# ---------------------------------------------------------------------------
# Axes-level 面板渲染（供 inspect-stat 集成模式复用，逻辑与 render_cubes_figure /
# render_swaps_figure 一致，但不创建 figure / 不保存文件）
# ---------------------------------------------------------------------------

# 与 inspect-stat 视觉常量保持一致（CUBE_COLOR_MAP 之前散落在 permanance_inspect.py，
# 现统一收敛到 permanence.py，作为本套件 visualization 的单一事实来源）。
CUBE_COLOR_MAP: dict[str, str] = {
    "red": "#d62728",
    "green": "#2ca02c",
    "blue": "#1f77b4",
    "unknown": "#7f7f7f",
}

PERMANENCE_DIFFICULTY_MARKERS: dict[str, str] = dict(_DIFFICULTY_MARKERS)
PERMANENCE_SWAP_INDEX_COLORS: list[str] = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# inspect-stat collage 的视图轴与 _PLOT_LIMIT 等价
_PERMANENCE_PANEL_LIMIT = _PLOT_LIMIT


def _prepare_permanence_panel_axis(ax: Any, title: str, point_count: int) -> None:
    ax.set_xlim(-_PERMANENCE_PANEL_LIMIT, _PERMANENCE_PANEL_LIMIT)
    ax.set_ylim(-_PERMANENCE_PANEL_LIMIT, _PERMANENCE_PANEL_LIMIT)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World Y")
    ax.set_ylabel("-World X")
    ax.grid(True, alpha=0.45)
    ax.set_title(f"{title}\npoints={point_count}")


def plot_permanence_cubes_panel(
    ax: Any,
    env_id: str,
    files: Iterable[PermanenceFile],
) -> int:
    """渲染 cube reset 位置散点：颜色按 cube 颜色、marker 按 difficulty，
    并用细线把 cube 连到对应的 bin。无数据时显示 'No cube data'。

    返回成功绘制的 cube 数量（用于上层统计）。
    """
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

    _prepare_permanence_panel_axis(ax, "Permanence cubes (Rotated XY)", plotted)
    return plotted


def plot_permanence_swaps_panel(
    ax: Any,
    env_id: str,
    files: Iterable[PermanenceFile],
) -> int:
    """渲染 swap pair 双向箭头：每对 (bin_a, bin_b) 按 swap_index 着色，
    bin 全集画成淡灰色背景点。非 Swap env 或无 swap_pairs 时显示
    'No swap data'。

    返回成功绘制的 swap pair 数量。
    """
    from matplotlib.lines import Line2D

    files_list = list(files or [])
    is_swap_env = env_id in SWAP_ENV_IDS

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

    _prepare_permanence_panel_axis(ax, "Permanence swaps (Rotated XY)", pair_count)
    return pair_count


__all__ = [
    "PERMANENCE_ENV_IDS",
    "SWAP_ENV_IDS",
    "PERMANENCE_JSON_FILENAME",
    "PermanenceFile",
    "CUBE_COLOR_MAP",
    "PERMANENCE_DIFFICULTY_MARKERS",
    "PERMANENCE_SWAP_INDEX_COLORS",
    "extract_permanence_init_state",
    "write_permanence_init_state",
    "load_permanence_init_state",
    "discover_permanence_files",
    "dedup_permanence_files",
    "render_cubes_figure",
    "render_swaps_figure",
    "run_permanence_pipeline",
    "plot_permanence_cubes_panel",
    "plot_permanence_swaps_panel",
]
