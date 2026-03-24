import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..logging_utils import logger


EPISODE_OBJECT_LOG_SCHEMA_VERSION = 1
EPISODE_OBJECT_LOG_FILENAME = "episode_object_logs.jsonl"


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


def empty_episode_object_log() -> Dict[str, Any]:
    return {
        "cube_bins": [],
        "target_cube": None,
        "swap_events": [],
    }


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


def build_object_descriptor(
    *,
    actor: Any = None,
    object_type: Optional[str],
    name: Optional[str] = None,
    actor_name: Optional[str] = None,
    color: Optional[str] = None,
    bin_index: Optional[int] = None,
    position_key: str = "world_position",
) -> Dict[str, Any]:
    resolved_actor_name = actor_name or getattr(actor, "name", None)
    resolved_name = name or resolved_actor_name
    resolved_bin_index = None if bin_index is None else int(bin_index)
    descriptor = {
        "type": object_type,
        "name": resolved_name,
        "actor_name": resolved_actor_name,
        "color": color,
        "bin_index": resolved_bin_index,
        position_key: extract_actor_world_position(actor),
    }
    return descriptor


def build_cube_bin_entry(
    *,
    bin_actor: Any,
    cube_actor: Any,
    color: Optional[str],
    bin_index: Optional[int],
) -> Dict[str, Any]:
    return {
        "bin_index": None if bin_index is None else int(bin_index),
        "color": color,
        "bin": build_object_descriptor(
            actor=bin_actor,
            object_type="bin",
            bin_index=bin_index,
        ),
        "cube": build_object_descriptor(
            actor=cube_actor,
            object_type="cube",
            color=color,
            bin_index=bin_index,
        ),
        "bin_world_position": extract_actor_world_position(bin_actor),
        "cube_world_position": extract_actor_world_position(cube_actor),
    }


def normalize_episode_object_log(payload: Any) -> Dict[str, Any]:
    normalized = empty_episode_object_log()
    if not isinstance(payload, dict):
        return normalized

    cube_bins = payload.get("cube_bins", [])
    if isinstance(cube_bins, list):
        normalized["cube_bins"] = _to_jsonable(cube_bins)

    target_cube = payload.get("target_cube", None)
    if target_cube is None or isinstance(target_cube, dict):
        normalized["target_cube"] = _to_jsonable(target_cube)

    swap_events = payload.get("swap_events", [])
    if isinstance(swap_events, list):
        normalized["swap_events"] = _to_jsonable(swap_events)

    return normalized


def build_episode_object_log_record(
    *,
    env: Optional[str],
    episode: Optional[int],
    seed: Optional[int],
    difficulty: Optional[str],
    episode_success: bool,
    object_log: Any,
) -> Dict[str, Any]:
    record = {
        "schema_version": EPISODE_OBJECT_LOG_SCHEMA_VERSION,
        "env": env,
        "episode": None if episode is None else int(episode),
        "seed": None if seed is None else int(seed),
        "difficulty": difficulty,
        "episode_success": bool(episode_success),
        "object_log": normalize_episode_object_log(object_log),
    }
    return record


def append_episode_object_log_record(
    output_root: Any,
    record: Dict[str, Any],
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
