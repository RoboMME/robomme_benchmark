import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from ..logging_utils import logger


EPISODE_OBJECT_LOG_FILENAME = "episode_object_logs.jsonl"
_EPISODE_OBJECT_LOG_STATE_ATTR = "_episode_object_log_state"


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _to_jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {
            str(key): _to_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return _to_python_scalar(value)


def _empty_episode_object_log_state() -> dict[str, list[Any]]:
    return {
        "bin_list": [],
        "cube_list": [],
        "target_cube_list": [],
        "swap_events": [],
        "collision_events": [],
    }


def _get_episode_object_log_state(env: Any) -> dict[str, list[Any]]:
    state = getattr(env, _EPISODE_OBJECT_LOG_STATE_ATTR, None)
    if not isinstance(state, dict):
        state = _empty_episode_object_log_state()
        setattr(env, _EPISODE_OBJECT_LOG_STATE_ATTR, state)
    return state


def init_episode_object_log_state(env: Any) -> None:
    setattr(env, _EPISODE_OBJECT_LOG_STATE_ATTR, _empty_episode_object_log_state())


def extract_actor_world_position(actor: Any) -> Optional[list[float]]:
    if actor is None:
        return None

    pose = getattr(actor, "pose", None)
    if pose is None:
        return None

    position = getattr(pose, "p", None)
    if position is None:
        return None

    if isinstance(position, torch.Tensor):
        position = position.detach().cpu().numpy()
    else:
        position = np.asarray(position)

    position = position.reshape(-1)
    if position.size < 3:
        return None
    return [float(position[0]), float(position[1]), float(position[2])]


def _serialize_actor_item(item: Any) -> Optional[dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    actor = item.get("actor")
    return {
        "name": getattr(actor, "name", None),
        "position": extract_actor_world_position(actor),
        "color": item.get("color"),
    }


def _serialize_actor_list(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    serialized = []
    for item in items:
        payload = _serialize_actor_item(item)
        if payload is not None:
            serialized.append(payload)
    return serialized


def record_reset_objects(
    env: Any,
    *,
    bin_list: list[dict[str, Any]],
    cube_list: list[dict[str, Any]],
    target_cube_list: list[dict[str, Any]],
) -> None:
    state = _get_episode_object_log_state(env)
    state["bin_list"] = _serialize_actor_list(bin_list)
    state["cube_list"] = _serialize_actor_list(cube_list)
    state["target_cube_list"] = _serialize_actor_list(target_cube_list)
    state["swap_events"] = []


def append_episode_object_swap_event(
    env: Any,
    *,
    swap_index: int,
    object_a: Any,
    object_b: Any,
) -> None:
    state = _get_episode_object_log_state(env)
    state["swap_events"].append(
        {
            "swap_index": int(swap_index),
            "object_a": getattr(object_a, "name", None),
            "object_b": getattr(object_b, "name", None),
        }
    )


def build_episode_object_collision_event(
    contact_summary: Optional[dict[str, Any]],
) -> dict[str, Any]:
    summary = contact_summary or {}
    return {
        "swap_contact_detected": bool(summary.get("swap_contact_detected", False)),
        "first_contact_step": summary.get("first_contact_step"),
        "contact_pairs": _to_jsonable(summary.get("contact_pairs", [])),
        "max_force_norm": float(summary.get("max_force_norm", 0.0)),
        "max_force_pair": summary.get("max_force_pair"),
        "max_force_step": summary.get("max_force_step"),
        "pair_max_force": {
            str(key): float(value)
            for key, value in (summary.get("pair_max_force", {}) or {}).items()
        },
    }


def append_episode_object_collision_event(
    env: Any,
    *,
    contact_summary: Optional[dict[str, Any]],
) -> None:
    state = _get_episode_object_log_state(env)
    state["collision_events"].append(
        build_episode_object_collision_event(contact_summary)
    )


def build_episode_object_log_record(
    env: Any,
    *,
    env_id: Optional[str],
    episode: Optional[int],
    seed: Optional[int],
) -> dict[str, Any]:
    state = _get_episode_object_log_state(env)
    return {
        "env": env_id,
        "episode": None if episode is None else int(episode),
        "seed": None if seed is None else int(seed),
        "bin_list": _to_jsonable(state.get("bin_list", [])),
        "cube_list": _to_jsonable(state.get("cube_list", [])),
        "target_cube_list": _to_jsonable(state.get("target_cube_list", [])),
        "swap_events": _to_jsonable(state.get("swap_events", [])),
        "collision_events": _to_jsonable(state.get("collision_events", [])),
    }


def append_episode_object_log_record(
    output_root: Any,
    record: dict[str, Any],
) -> Optional[Path]:
    jsonl_path = Path(output_root) / EPISODE_OBJECT_LOG_FILENAME
    try:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(_to_jsonable(record), ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.debug(f"Warning: failed to append episode object log to {jsonl_path}: {exc}")
        return None
    return jsonl_path
