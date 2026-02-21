import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def find_exact_label_option_index(target_label: Any, options: List[dict]) -> int:
    """Return option index only when target_label exactly equals option label."""
    if not isinstance(target_label, str):
        return -1
    for idx, opt in enumerate(options):
        if opt.get("label") == target_label:
            return idx
    return -1


def map_action_text_to_option_label(action_text: Any, options: List[dict]) -> Optional[str]:
    """Map exact option action text to its option label for recording-time conversion."""
    if not isinstance(action_text, str):
        return None
    for opt in options:
        if opt.get("action") == action_text:
            label = opt.get("label")
            if isinstance(label, str) and label:
                return label
            return None
    return None


def normalize_and_clip_point_xy(
    point_like: Any,
    width: int,
    height: int,
) -> Optional[Tuple[int, int]]:
    """Normalize arbitrary point-like input into clipped (x, y)."""
    if point_like is None:
        return None
    if not isinstance(point_like, (list, tuple, np.ndarray)) or len(point_like) < 2:
        return None
    try:
        x = int(float(point_like[0]))
        y = int(float(point_like[1]))
    except (TypeError, ValueError):
        return None
    x = max(0, min(x, int(width) - 1))
    y = max(0, min(y, int(height) - 1))
    return x, y


def _collect_candidates(item: Any, out: List[Any]) -> None:
    if isinstance(item, (list, tuple)):
        for child in item:
            _collect_candidates(child, out)
        return
    if isinstance(item, dict):
        for child in item.values():
            _collect_candidates(child, out)
        return
    if item is not None:
        out.append(item)


def _unique_candidates(available: Any) -> List[Any]:
    candidates: List[Any] = []
    _collect_candidates(available, candidates)
    # Keep object identity uniqueness to avoid redundant scans.
    return list(dict.fromkeys(candidates))


def _target_ids_for_actor(seg_id_map: Dict[int, Any], actor: Any) -> List[int]:
    return [int(seg_id) for seg_id, obj in seg_id_map.items() if obj is actor]


def _scan_actor_masks(
    seg_raw: np.ndarray,
    seg_id_map: Dict[int, Any],
    actor: Any,
    click_xy: Tuple[int, int],
) -> Tuple[Optional[Tuple[int, Tuple[int, int]]], Optional[Tuple[int, Tuple[int, int]]]]:
    """
    Return (hit, observed):
    - hit: clicked (seg_id, centroid) for this actor if click is inside mask.
    - observed: first visible (seg_id, centroid) for this actor.
    """
    cx, cy = click_xy
    observed: Optional[Tuple[int, Tuple[int, int]]] = None

    for target_id in _target_ids_for_actor(seg_id_map, actor):
        mask = seg_raw == target_id
        if not np.any(mask):
            continue

        ys, xs = np.nonzero(mask)
        centroid_point = (int(xs.mean()), int(ys.mean()))
        if observed is None:
            observed = (target_id, centroid_point)

        if bool(mask[cy, cx]):
            return (target_id, centroid_point), observed

    return None, observed


def select_target_with_point(
    seg_raw: np.ndarray,
    seg_id_map: Dict[int, Any],
    available: Any,
    point_like: Any,
) -> Optional[Dict[str, Any]]:
    """
    Two-stage matching:
    1) If click point hits a visible candidate mask, return that actor immediately.
    2) Otherwise randomly sample one actor from candidate list as fallback.
    """
    if seg_raw is None:
        return None

    h, w = seg_raw.shape[:2]
    point_xy = normalize_and_clip_point_xy(point_like, width=w, height=h)
    if point_xy is None:
        return None

    unique_candidates = _unique_candidates(available)
    if not unique_candidates:
        return None

    cx, cy = point_xy
    observed_info: Dict[Any, Tuple[int, Tuple[int, int]]] = {}

    for actor in unique_candidates:
        hit, observed = _scan_actor_masks(
            seg_raw=seg_raw,
            seg_id_map=seg_id_map,
            actor=actor,
            click_xy=point_xy,
        )
        if observed is not None and actor not in observed_info:
            observed_info[actor] = observed

        if hit is not None:
            target_id, centroid_point = hit
            return {
                "obj": actor,
                "name": getattr(actor, "name", f"id_{target_id}"),
                "seg_id": target_id,
                "click_point": (int(cx), int(cy)),
                "centroid_point": centroid_point,
                "selection_mode": "hit",
                "used_random_fallback": False,
            }

    fallback_actor = random.choice(unique_candidates)
    fallback_seg_id: Optional[int] = None
    fallback_centroid: Optional[Tuple[int, int]] = None
    if fallback_actor in observed_info:
        fallback_seg_id, fallback_centroid = observed_info[fallback_actor]

    return {
        "obj": fallback_actor,
        "name": getattr(fallback_actor, "name", "unknown"),
        "seg_id": fallback_seg_id,
        "click_point": (int(cx), int(cy)),
        "centroid_point": fallback_centroid,
        "selection_mode": "fallback_random",
        "used_random_fallback": True,
    }
