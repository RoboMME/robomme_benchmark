"""Reference 套件 reset 时刻 task_target 选择状态的提取、序列化与发现。

适用 env：VideoPlaceButton / VideoPlaceOrder / PickHighlight。

为什么独立成模块（与 permanence.py 同款理由）：
- Env-rollout-parallel-segmentation.py 已超过 2000 行，reference 相关
  逻辑（从 env 读属性 / enrich payload / inspect 端 discover）不应继续往
  主脚本里堆。
- inspect-stat 端也要消费同一份 schema；把读写两端的契约集中在一个文件里
  能让格式始终一致。

env 文件本身不做任何修改：本模块直接从 env.unwrapped 读 _load_scene 已经
设置好的属性 (target_target / targets / target_color_name /
target_target_language / which_in_subset / which_targets_to_pick /
target_cubes / target_cube_colors / all_cubes / all_cube_names)。

与 permanence 的关键差异：reference 不写独立 sidecar 文件，而是把
selected_target dict 原地塞进 visible_objects.json 的顶层。这样 inspect
端无需额外路径，复用 visible_objects.json 的发现逻辑即可读到。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

REFERENCE_TARGET_ENV_IDS: frozenset[str] = frozenset(
    {"VideoPlaceButton", "VideoPlaceOrder", "PickHighlight"}
)

VIDEO_PLACE_BUTTON_ENV_ID = "VideoPlaceButton"
VIDEO_PLACE_ORDER_ENV_ID = "VideoPlaceOrder"
PICK_HIGHLIGHT_ENV_ID = "PickHighlight"

KIND_VIDEO_PLACE_BUTTON = "video_place_button"
KIND_VIDEO_PLACE_ORDER = "video_place_order"
KIND_PICK_HIGHLIGHT = "pick_highlight"

VISIBLE_OBJECTS_JSON_FILENAME = "visible_objects.json"
SELECTED_TARGET_JSON_KEY = "selected_target"

# 与 inspect-stat / permanence 保持一致的目录命名 regex
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


def _index_in_list(target: Any, candidates: list[Any]) -> Optional[int]:
    """反查 target 在 candidates 里的下标。优先 is，再按 name 比对。"""
    for i, candidate in enumerate(candidates):
        if candidate is target:
            return i
    target_name = getattr(target, "name", None)
    if target_name is not None:
        for i, candidate in enumerate(candidates):
            if getattr(candidate, "name", None) == target_name:
                return i
    return None


def _candidate_dict(idx: int, actor: Any) -> dict:
    return {
        "index": int(idx),
        "name": _actor_name(actor),
        "position_xy": _actor_xy(actor),
    }


# ---------------------------------------------------------------------------
# 三个 env 各自的提取
# ---------------------------------------------------------------------------


def _extract_video_place_button(base: Any) -> dict:
    targets = list(getattr(base, "targets", []) or [])
    target_target = getattr(base, "target_target", None)
    timing = getattr(base, "target_target_language", None)
    target_color_name = getattr(base, "target_color_name", None)

    if not targets:
        raise ValueError("VideoPlaceButton.targets is empty after reset")
    if target_target is None:
        raise ValueError("VideoPlaceButton.target_target is None after reset")
    if timing not in ("before", "after"):
        raise ValueError(
            f"VideoPlaceButton.target_target_language must be 'before'|'after', "
            f"got {timing!r}"
        )

    selected_idx = _index_in_list(target_target, targets)
    if selected_idx is None:
        raise ValueError(
            "VideoPlaceButton.target_target not found in targets list"
        )

    color_name = str(target_color_name) if target_color_name else "unknown"

    return {
        "kind": KIND_VIDEO_PLACE_BUTTON,
        "timing": str(timing),
        "task_target_indices": [selected_idx],
        "task_target_names": [_actor_name(target_target)],
        "task_target_positions_xy": [_actor_xy(target_target)],
        "task_target_colors": [color_name],
        "all_candidates": [_candidate_dict(i, t) for i, t in enumerate(targets)],
    }


def _extract_video_place_order(base: Any) -> dict:
    targets = list(getattr(base, "targets", []) or [])
    target_target = getattr(base, "target_target", None)
    which_in_subset = getattr(base, "which_in_subset", None)
    which_targets_to_pick = list(getattr(base, "which_targets_to_pick", []) or [])
    target_color_name = getattr(base, "target_color_name", None)

    if not targets:
        raise ValueError("VideoPlaceOrder.targets is empty after reset")
    if target_target is None:
        raise ValueError("VideoPlaceOrder.target_target is None after reset")
    if which_in_subset is None:
        raise ValueError("VideoPlaceOrder.which_in_subset is None after reset")
    if not which_targets_to_pick:
        raise ValueError(
            "VideoPlaceOrder.which_targets_to_pick is empty after reset"
        )

    selected_idx = _index_in_list(target_target, targets)
    if selected_idx is None:
        raise ValueError(
            "VideoPlaceOrder.target_target not found in targets list"
        )

    color_name = str(target_color_name) if target_color_name else "unknown"
    order_position_int = int(which_in_subset)

    return {
        "kind": KIND_VIDEO_PLACE_ORDER,
        "order_position": order_position_int,
        "subset_size": len(which_targets_to_pick),
        "task_target_indices": [selected_idx],
        "task_target_names": [_actor_name(target_target)],
        "task_target_positions_xy": [_actor_xy(target_target)],
        "task_target_colors": [color_name],
        "all_candidates": [_candidate_dict(i, t) for i, t in enumerate(targets)],
    }


def _extract_pick_highlight(base: Any) -> dict:
    all_cubes = list(getattr(base, "all_cubes", []) or [])
    target_cubes = list(getattr(base, "target_cubes", []) or [])
    target_cube_colors = list(getattr(base, "target_cube_colors", []) or [])

    if not all_cubes:
        raise ValueError("PickHighlight.all_cubes is empty after reset")
    if not target_cubes:
        raise ValueError("PickHighlight.target_cubes is empty after reset")
    if len(target_cube_colors) != len(target_cubes):
        raise ValueError(
            f"PickHighlight: target_cubes ({len(target_cubes)}) and "
            f"target_cube_colors ({len(target_cube_colors)}) length mismatch"
        )

    indices: list[int] = []
    names: list[str] = []
    positions: list[list[float]] = []
    colors: list[str] = []
    for cube, color_name in zip(target_cubes, target_cube_colors):
        cube_idx = _index_in_list(cube, all_cubes)
        if cube_idx is None:
            raise ValueError(
                f"PickHighlight.target_cubes contains cube {cube!r} not in all_cubes"
            )
        indices.append(cube_idx)
        names.append(_actor_name(cube))
        positions.append(_actor_xy(cube))
        colors.append(str(color_name) if color_name else "unknown")

    return {
        "kind": KIND_PICK_HIGHLIGHT,
        "highlight_count": len(target_cubes),
        "task_target_indices": indices,
        "task_target_names": names,
        "task_target_positions_xy": positions,
        "task_target_colors": colors,
        "all_candidates": [_candidate_dict(i, c) for i, c in enumerate(all_cubes)],
    }


# ---------------------------------------------------------------------------
# 公开接口：提取 + payload enrich
# ---------------------------------------------------------------------------


def extract_selected_target(env: Any, env_id: Optional[str] = None) -> Optional[dict]:
    """从 env 实例提取 reference 套件的 selected_target dict。

    非 reference target env 直接返回 None；目标 env 缺关键属性时按
    no-silent-fallbacks 原则直接 raise。

    env_id 可选：传入时按 env_id 走 dispatch；未传时回退到
    type(env.unwrapped).__name__（仅当类名恰好是支持的 env_id 时生效）。
    """
    base = getattr(env, "unwrapped", env)
    resolved_env_id = env_id if env_id is not None else type(base).__name__
    if resolved_env_id not in REFERENCE_TARGET_ENV_IDS:
        return None

    if resolved_env_id == VIDEO_PLACE_BUTTON_ENV_ID:
        state = _extract_video_place_button(base)
    elif resolved_env_id == VIDEO_PLACE_ORDER_ENV_ID:
        state = _extract_video_place_order(base)
    elif resolved_env_id == PICK_HIGHLIGHT_ENV_ID:
        state = _extract_pick_highlight(base)
    else:
        raise AssertionError(
            f"unreachable: resolved_env_id={resolved_env_id!r}"
        )
    return _to_jsonable(state)


def enrich_visible_payload(payload: dict, env: Any, env_id: str) -> None:
    """原地修改 visible_objects.json 的 payload —— 给 reference target env
    添加顶层 'selected_target' key。其他 env_id 是 no-op。

    Env-rollout-parallel-segmentation.py 在写 visible_objects.json 之前
    调用一次即可。
    """
    if env_id not in REFERENCE_TARGET_ENV_IDS:
        return
    state = extract_selected_target(env, env_id=env_id)
    if state is None:
        # extract 同样判断了 env_id；理论上走到这里只能是 mismatch，但保留
        # 保护以避免静默失败
        raise ValueError(
            f"enrich_visible_payload: extract_selected_target returned None "
            f"for env_id={env_id!r}"
        )
    payload[SELECTED_TARGET_JSON_KEY] = state


# ---------------------------------------------------------------------------
# Inspect 端：从 visible_objects.json 发现并去重
# ---------------------------------------------------------------------------


@dataclass
class SelectedTargetRecord:
    path: Path
    env_id: str
    episode: int
    seed: int
    selected_target: dict


def _load_visible_objects(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_selected_target_records(
    segmentation_dir: Path,
    env_filter: Optional[str] = None,
) -> list[SelectedTargetRecord]:
    """递归扫描 segmentation_dir 下所有 visible_objects.json，提取
    payload['selected_target']。

    - 用 payload 的 env_id/episode/seed 作为权威来源；目录名仅作 fallback。
    - env_filter 非空时按 env_id 过滤（精确匹配）。
    - env 不在 REFERENCE_TARGET_ENV_IDS 时跳过（PickHighlight/VideoPlace*
      之外的 env 本来就没 selected_target 字段，无需警告）。
    - env 在范围内但缺 selected_target 字段时打 [Warn]（说明该条 episode
      数据来自旧 rollout，需要重跑）。
    """
    seg_dir = Path(segmentation_dir)
    if not seg_dir.is_dir():
        return []

    results: list[SelectedTargetRecord] = []
    for json_path in sorted(seg_dir.rglob(VISIBLE_OBJECTS_JSON_FILENAME)):
        try:
            payload = _load_visible_objects(json_path)
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"[Warn] Skip invalid visible_objects JSON {json_path}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        env_id = payload.get("env_id")
        episode = payload.get("episode")
        seed = payload.get("seed")
        if (
            not isinstance(env_id, str)
            or not isinstance(episode, int)
            or not isinstance(seed, int)
        ):
            print(
                f"[Warn] Skip visible_objects JSON {json_path} missing "
                f"env_id/episode/seed"
            )
            continue

        if env_id not in REFERENCE_TARGET_ENV_IDS:
            continue
        if env_filter is not None and env_id != env_filter:
            continue

        selected_target = payload.get(SELECTED_TARGET_JSON_KEY)
        if not isinstance(selected_target, dict):
            print(
                f"[Warn] {env_id} ep{episode} seed{seed}: "
                f"visible_objects.json missing 'selected_target' field "
                f"(re-run rollout to populate). Skipping {json_path}."
            )
            continue

        results.append(
            SelectedTargetRecord(
                path=json_path,
                env_id=env_id,
                episode=episode,
                seed=seed,
                selected_target=selected_target,
            )
        )
    return results


def dedup_selected_target_records(
    entries: list[SelectedTargetRecord],
) -> tuple[list[SelectedTargetRecord], list[SelectedTargetRecord]]:
    """同 (env_id, episode) 多 seed 时只保留 max seed —— 与 visible_objects /
    permanence 一致。"""
    grouped: dict[tuple[str, int], list[SelectedTargetRecord]] = {}
    for entry in entries:
        grouped.setdefault((entry.env_id, entry.episode), []).append(entry)
    kept: list[SelectedTargetRecord] = []
    skipped: list[SelectedTargetRecord] = []
    for bucket in grouped.values():
        bucket_sorted = sorted(bucket, key=lambda e: e.seed)
        kept.append(bucket_sorted[-1])
        skipped.extend(bucket_sorted[:-1])
    kept.sort(key=lambda e: (e.env_id, e.episode))
    skipped.sort(key=lambda e: (e.env_id, e.episode, e.seed))
    return kept, skipped


__all__ = [
    "REFERENCE_TARGET_ENV_IDS",
    "VIDEO_PLACE_BUTTON_ENV_ID",
    "VIDEO_PLACE_ORDER_ENV_ID",
    "PICK_HIGHLIGHT_ENV_ID",
    "KIND_VIDEO_PLACE_BUTTON",
    "KIND_VIDEO_PLACE_ORDER",
    "KIND_PICK_HIGHLIGHT",
    "SELECTED_TARGET_JSON_KEY",
    "VISIBLE_OBJECTS_JSON_FILENAME",
    "SelectedTargetRecord",
    "extract_selected_target",
    "enrich_visible_payload",
    "discover_selected_target_records",
    "dedup_selected_target_records",
]
