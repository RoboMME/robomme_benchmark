"""Reference 套件 reset 时刻 env-specific 数据的提取、序列化与发现。

适用 env：VideoPlaceButton / VideoPlaceOrder / PickHighlight / VideoRepick。

为什么独立成模块（与 permanence.py 同款理由）：
- Env-rollout-parallel-segmentation.py 已超过 2000 行，reference 相关
  逻辑（从 env 读属性 / enrich payload / inspect 端 discover）不应继续往
  主脚本里堆。
- inspect-stat 端也要消费同一份 schema；把读写两端的契约集中在一个文件里
  能让格式始终一致。

env 文件本身不做任何修改：本模块直接从 env.unwrapped 读 _load_scene 已经
设置好的属性 (target_target / targets / target_color_name /
target_target_language / which_in_subset / which_targets_to_pick /
target_cubes / target_cube_colors / all_cubes / all_cube_names /
target_cube_1 / num_repeats)。

数据写入策略：reference 不写独立 sidecar 文件，所有字段都原地塞进
visible_objects.json 的顶层 ——
- VideoPlaceButton / VideoPlaceOrder / PickHighlight → 'selected_target'
- VideoRepick                                       → 'videorepick_metadata'
- VideoPlaceButton / VideoPlaceOrder                → 'videoplace_swap_pair'
                                                       （与 'selected_target' 平级独立顶层字段）
这样 inspect 端无需额外路径，复用 visible_objects.json 的发现逻辑即可读到。
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

# VideoPlaceButton / VideoPlaceOrder 共享 hard 难度的 swap 机制
VIDEOPLACE_ENV_IDS: frozenset[str] = frozenset(
    {"VideoPlaceButton", "VideoPlaceOrder"}
)

VIDEO_PLACE_BUTTON_ENV_ID = "VideoPlaceButton"
VIDEO_PLACE_ORDER_ENV_ID = "VideoPlaceOrder"
PICK_HIGHLIGHT_ENV_ID = "PickHighlight"
VIDEOREPICK_ENV_ID = "VideoRepick"

KIND_VIDEO_PLACE_BUTTON = "video_place_button"
KIND_VIDEO_PLACE_ORDER = "video_place_order"
KIND_PICK_HIGHLIGHT = "pick_highlight"

VISIBLE_OBJECTS_JSON_FILENAME = "visible_objects.json"
SELECTED_TARGET_JSON_KEY = "selected_target"
VIDEOREPICK_METADATA_KEY = "videorepick_metadata"
VIDEOPLACE_SWAP_PAIR_KEY = "videoplace_swap_pair"

_VIDEOREPICK_COLOR_TOLERANCE = 0.05
_VIDEOREPICK_COLOR_RGB_MAP: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "green": (0.0, 1.0, 0.0),
}
_VIDEOREPICK_COLOR_NAME_PATTERN = re.compile(r"\b(red|blue|green)\b")

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
# VideoRepick：从 actor 渲染材质 / actor.name 推断 cube 颜色（容差 0.05）
# ---------------------------------------------------------------------------


def _normalize_rgba(base_color: Any) -> Optional[tuple[float, float, float, float]]:
    try:
        color_values = [float(value) for value in list(base_color)]
    except Exception:
        return None
    if len(color_values) == 3:
        color_values.append(1.0)
    if len(color_values) < 4:
        return None
    return (
        color_values[0],
        color_values[1],
        color_values[2],
        color_values[3],
    )


def _match_rgb_to_name(rgba: tuple[float, float, float, float]) -> Optional[str]:
    for color_name, expected_rgb in _VIDEOREPICK_COLOR_RGB_MAP.items():
        if all(
            abs(rgba[channel_idx] - expected_rgb[channel_idx])
            <= _VIDEOREPICK_COLOR_TOLERANCE
            for channel_idx in range(3)
        ):
            return color_name
    return None


def _iter_actor_rgba_colors(actor: Any):
    for entity in list(getattr(actor, "_objs", []) or []):
        for component in list(getattr(entity, "components", []) or []):
            render_shapes = getattr(component, "render_shapes", None)
            if render_shapes is None:
                continue
            for render_shape in list(render_shapes):
                material = getattr(render_shape, "material", None)
                if material is None:
                    get_material = getattr(render_shape, "get_material", None)
                    if callable(get_material):
                        try:
                            material = get_material()
                        except Exception:
                            material = None
                if material is None:
                    continue
                base_color = getattr(material, "base_color", None)
                if base_color is None:
                    get_base_color = getattr(material, "get_base_color", None)
                    if callable(get_base_color):
                        try:
                            base_color = get_base_color()
                        except Exception:
                            base_color = None
                if base_color is None:
                    continue
                rgba = _normalize_rgba(base_color)
                if rgba is not None:
                    yield rgba


def _color_from_actor_name(actor_name: object) -> Optional[str]:
    if not actor_name:
        return None
    match = _VIDEOREPICK_COLOR_NAME_PATTERN.search(str(actor_name).lower())
    if match is None:
        return None
    return match.group(1)


def _color_from_actor(actor: Any) -> str:
    """从 actor 渲染材质 base_color 或 actor.name 推断颜色名（容差 0.05）。
    无法判定时 raise。"""
    for rgba in _iter_actor_rgba_colors(actor):
        color_name = _match_rgb_to_name(rgba)
        if color_name is not None:
            return color_name
    actor_name = getattr(actor, "name", None)
    color_name = _color_from_actor_name(actor_name)
    if color_name is not None:
        return color_name
    raise ValueError(
        "failed to resolve cube color from render material or actor name"
    )


def _extract_videorepick_metadata(env: Any) -> dict:
    """从 env.unwrapped 提取 VideoRepick 的 target_cube_1 颜色/位置 + num_repeats +
    spawned_cubes 候选列表。缺关键属性 / 类型不合法时 raise。"""
    base = getattr(env, "unwrapped", env)
    target_cube = getattr(base, "target_cube_1", None)
    if target_cube is None:
        raise ValueError("VideoRepick: target_cube_1 not set after reset")
    color_name = _color_from_actor(target_cube)
    if color_name not in {"red", "blue", "green"}:
        raise ValueError(
            f"VideoRepick: invalid target_cube_1 color {color_name!r}"
        )
    num_repeats = getattr(base, "num_repeats", None)
    if isinstance(num_repeats, bool) or not isinstance(num_repeats, int):
        raise ValueError(
            f"VideoRepick: num_repeats has invalid type "
            f"{type(num_repeats).__name__}"
        )
    if num_repeats < 1:
        raise ValueError(
            f"VideoRepick: num_repeats must be >= 1, got {num_repeats}"
        )

    spawned_cubes = list(getattr(base, "spawned_cubes", []) or [])
    if not spawned_cubes:
        raise ValueError(
            "VideoRepick: spawned_cubes is empty after reset"
        )
    target_idx = _index_in_list(target_cube, spawned_cubes)
    if target_idx is None:
        raise ValueError(
            "VideoRepick: target_cube_1 not found in spawned_cubes"
        )

    return {
        "target_cube_1_color": color_name,
        "num_repeats": int(num_repeats),
        "target_cube_1_index": int(target_idx),
        "target_cube_1_name": _actor_name(target_cube),
        "target_cube_1_position_xy": _actor_xy(target_cube),
        "all_candidates": [
            _candidate_dict(i, cube) for i, cube in enumerate(spawned_cubes)
        ],
    }


# ---------------------------------------------------------------------------
# VideoPlaceButton / VideoPlaceOrder：swap 配对信息提取
# ---------------------------------------------------------------------------


def _extract_videoplace_swap_pair(env: Any, env_id: str) -> dict:
    """从 env 提取 VideoPlaceButton / VideoPlaceOrder 的 swap 配对信息。

    永远返回 dict（与 VideoRepick metadata 模式对齐）：
    - hard 难度且 swap_target_a/b 非 None → has_swap=True，填字段
    - easy/medium 难度（swap_target_a/b == None）→ has_swap=False，
      所有 a/b 字段为 None
    """
    if env_id not in VIDEOPLACE_ENV_IDS:
        raise ValueError(
            f"_extract_videoplace_swap_pair called with unsupported env_id={env_id!r}"
        )
    base = getattr(env, "unwrapped", env)
    targets = list(getattr(base, "targets", []) or [])
    if not targets:
        raise ValueError(f"{env_id}.targets is empty after reset")

    swap_a = getattr(base, "swap_target_a", None)
    swap_b = getattr(base, "swap_target_b", None)
    has_swap = swap_a is not None and swap_b is not None
    if not has_swap:
        return {
            "env_id": env_id,
            "has_swap": False,
            "swap_target_a_index": None,
            "swap_target_a_name": None,
            "swap_target_a_position_xy": None,
            "swap_target_b_index": None,
            "swap_target_b_name": None,
            "swap_target_b_position_xy": None,
        }

    a_idx = _index_in_list(swap_a, targets)
    b_idx = _index_in_list(swap_b, targets)
    if a_idx is None or b_idx is None:
        raise ValueError(
            f"{env_id}: swap_target_a/b not found in targets list (corrupted state)"
        )
    return {
        "env_id": env_id,
        "has_swap": True,
        "swap_target_a_index": int(a_idx),
        "swap_target_a_name": _actor_name(swap_a),
        "swap_target_a_position_xy": _actor_xy(swap_a),
        "swap_target_b_index": int(b_idx),
        "swap_target_b_name": _actor_name(swap_b),
        "swap_target_b_position_xy": _actor_xy(swap_b),
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
    """原地修改 visible_objects.json 的 payload。

    PickHighlight / VideoPlaceButton / VideoPlaceOrder
        → payload['selected_target'] = extract_selected_target(...)
    VideoPlaceButton / VideoPlaceOrder（额外，与 selected_target 平级）
        → payload['videoplace_swap_pair'] = {has_swap, swap_target_{a,b}_*}
    VideoRepick
        → payload['videorepick_metadata'] = {target_cube_1_color, num_repeats}
    其他 env_id
        → no-op
    """
    if env_id in REFERENCE_TARGET_ENV_IDS:
        state = extract_selected_target(env, env_id=env_id)
        if state is None:
            raise ValueError(
                f"enrich_visible_payload: extract_selected_target returned None "
                f"for env_id={env_id!r}"
            )
        payload[SELECTED_TARGET_JSON_KEY] = state
        if env_id in VIDEOPLACE_ENV_IDS:
            payload[VIDEOPLACE_SWAP_PAIR_KEY] = _to_jsonable(
                _extract_videoplace_swap_pair(env, env_id)
            )
        return
    if env_id == VIDEOREPICK_ENV_ID:
        payload[VIDEOREPICK_METADATA_KEY] = _to_jsonable(
            _extract_videorepick_metadata(env)
        )
        return
    # 其他 env：no-op


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


# ---------------------------------------------------------------------------
# Inspect 端：VideoRepick 专用发现 / 去重（与 selected_target 解耦）
# ---------------------------------------------------------------------------


@dataclass
class VideoRepickRecord:
    path: Path
    env_id: str
    episode: int
    seed: int
    metadata: dict


def discover_videorepick_records(
    segmentation_dir: Path,
    env_filter: Optional[str] = None,
) -> list[VideoRepickRecord]:
    """递归扫描 segmentation_dir 下所有 visible_objects.json，提取
    payload['videorepick_metadata']。

    - 只处理 env_id == 'VideoRepick'，其它 env 跳过。
    - env_filter 非空且 != 'VideoRepick' 时直接返回空列表。
    - metadata 缺 'target_cube_1_position_xy' / 'all_candidates' 时打 [Warn]
      并跳过（说明该条 episode 数据来自旧 rollout，需要重跑）。
    """
    seg_dir = Path(segmentation_dir)
    if not seg_dir.is_dir():
        return []
    if env_filter is not None and env_filter != VIDEOREPICK_ENV_ID:
        return []

    results: list[VideoRepickRecord] = []
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

        if env_id != VIDEOREPICK_ENV_ID:
            continue

        metadata = payload.get(VIDEOREPICK_METADATA_KEY)
        if not isinstance(metadata, dict):
            print(
                f"[Warn] {env_id} ep{episode} seed{seed}: "
                f"visible_objects.json missing 'videorepick_metadata' field "
                f"(re-run rollout to populate). Skipping {json_path}."
            )
            continue
        target_xy = metadata.get("target_cube_1_position_xy")
        if not (isinstance(target_xy, (list, tuple)) and len(target_xy) >= 2):
            print(
                f"[Warn] {env_id} ep{episode} seed{seed}: "
                f"videorepick_metadata missing 'target_cube_1_position_xy' "
                f"(re-run rollout to populate). Skipping {json_path}."
            )
            continue
        if not isinstance(metadata.get("all_candidates"), list):
            print(
                f"[Warn] {env_id} ep{episode} seed{seed}: "
                f"videorepick_metadata missing 'all_candidates' "
                f"(re-run rollout to populate). Skipping {json_path}."
            )
            continue

        results.append(
            VideoRepickRecord(
                path=json_path,
                env_id=env_id,
                episode=episode,
                seed=seed,
                metadata=metadata,
            )
        )
    return results


def dedup_videorepick_records(
    entries: list[VideoRepickRecord],
) -> tuple[list[VideoRepickRecord], list[VideoRepickRecord]]:
    """同 (env_id, episode) 多 seed 时只保留 max seed —— 与
    dedup_selected_target_records 一致。"""
    grouped: dict[tuple[str, int], list[VideoRepickRecord]] = {}
    for entry in entries:
        grouped.setdefault((entry.env_id, entry.episode), []).append(entry)
    kept: list[VideoRepickRecord] = []
    skipped: list[VideoRepickRecord] = []
    for bucket in grouped.values():
        bucket_sorted = sorted(bucket, key=lambda e: e.seed)
        kept.append(bucket_sorted[-1])
        skipped.extend(bucket_sorted[:-1])
    kept.sort(key=lambda e: (e.env_id, e.episode))
    skipped.sort(key=lambda e: (e.env_id, e.episode, e.seed))
    return kept, skipped


# ---------------------------------------------------------------------------
# Inspect 端：VideoPlaceButton / VideoPlaceOrder swap_pair 专用发现 / 去重
# （与 selected_target 解耦的独立路径，仿 VideoRepick）
# ---------------------------------------------------------------------------


@dataclass
class VideoPlaceSwapPairRecord:
    path: Path
    env_id: str
    episode: int
    seed: int
    swap_pair: dict   # 永远是 dict（has_swap True/False）


def discover_videoplace_swap_pair_records(
    segmentation_dir: Path,
    env_filter: Optional[str] = None,
) -> list[VideoPlaceSwapPairRecord]:
    """递归扫描 segmentation_dir 下所有 visible_objects.json，提取
    payload['videoplace_swap_pair']。

    - 只处理 env_id ∈ VIDEOPLACE_ENV_IDS（VideoPlaceButton / VideoPlaceOrder）
    - env_filter 非空且不在 VIDEOPLACE_ENV_IDS 时直接返回空列表
    - swap_pair 字段缺失 / 类型不合法时打 [Warn] 并跳过（说明该条 episode
      数据来自旧 rollout，需要重跑）
    """
    seg_dir = Path(segmentation_dir)
    if not seg_dir.is_dir():
        return []
    if env_filter is not None and env_filter not in VIDEOPLACE_ENV_IDS:
        return []

    results: list[VideoPlaceSwapPairRecord] = []
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

        if env_id not in VIDEOPLACE_ENV_IDS:
            continue
        if env_filter is not None and env_id != env_filter:
            continue

        swap_pair = payload.get(VIDEOPLACE_SWAP_PAIR_KEY)
        if not isinstance(swap_pair, dict):
            print(
                f"[Warn] {env_id} ep{episode} seed{seed}: "
                f"visible_objects.json missing 'videoplace_swap_pair' field "
                f"(re-run rollout to populate). Skipping {json_path}."
            )
            continue
        if not isinstance(swap_pair.get("has_swap"), bool):
            print(
                f"[Warn] {env_id} ep{episode} seed{seed}: "
                f"videoplace_swap_pair missing/invalid 'has_swap' "
                f"(re-run rollout to populate). Skipping {json_path}."
            )
            continue

        results.append(
            VideoPlaceSwapPairRecord(
                path=json_path,
                env_id=env_id,
                episode=episode,
                seed=seed,
                swap_pair=swap_pair,
            )
        )
    return results


def dedup_videoplace_swap_pair_records(
    entries: list[VideoPlaceSwapPairRecord],
) -> tuple[list[VideoPlaceSwapPairRecord], list[VideoPlaceSwapPairRecord]]:
    """同 (env_id, episode) 多 seed 时只保留 max seed —— 与
    dedup_videorepick_records 行为完全一致。"""
    grouped: dict[tuple[str, int], list[VideoPlaceSwapPairRecord]] = {}
    for entry in entries:
        grouped.setdefault((entry.env_id, entry.episode), []).append(entry)
    kept: list[VideoPlaceSwapPairRecord] = []
    skipped: list[VideoPlaceSwapPairRecord] = []
    for bucket in grouped.values():
        bucket_sorted = sorted(bucket, key=lambda e: e.seed)
        kept.append(bucket_sorted[-1])
        skipped.extend(bucket_sorted[:-1])
    kept.sort(key=lambda e: (e.env_id, e.episode))
    skipped.sort(key=lambda e: (e.env_id, e.episode, e.seed))
    return kept, skipped


__all__ = [
    "REFERENCE_TARGET_ENV_IDS",
    "VIDEOPLACE_ENV_IDS",
    "VIDEO_PLACE_BUTTON_ENV_ID",
    "VIDEO_PLACE_ORDER_ENV_ID",
    "PICK_HIGHLIGHT_ENV_ID",
    "VIDEOREPICK_ENV_ID",
    "KIND_VIDEO_PLACE_BUTTON",
    "KIND_VIDEO_PLACE_ORDER",
    "KIND_PICK_HIGHLIGHT",
    "SELECTED_TARGET_JSON_KEY",
    "VIDEOREPICK_METADATA_KEY",
    "VIDEOPLACE_SWAP_PAIR_KEY",
    "VISIBLE_OBJECTS_JSON_FILENAME",
    "SelectedTargetRecord",
    "VideoRepickRecord",
    "VideoPlaceSwapPairRecord",
    "extract_selected_target",
    "enrich_visible_payload",
    "discover_selected_target_records",
    "dedup_selected_target_records",
    "discover_videorepick_records",
    "dedup_videorepick_records",
    "discover_videoplace_swap_pair_records",
    "dedup_videoplace_swap_pair_records",
]
