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


__all__ = [
    "PERMANENCE_ENV_IDS",
    "SWAP_ENV_IDS",
    "PERMANENCE_JSON_FILENAME",
    "COLOR_NAME_TO_RGBA",
    "PermanenceFile",
    "extract_permanence_init_state",
    "write_permanence_init_state",
    "load_permanence_init_state",
    "discover_permanence_files",
    "dedup_permanence_files",
]
