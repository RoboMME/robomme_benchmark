"""Utility helpers for oracle planner action matching and target selection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _flatten_available(available: Any) -> List[Any]:
    """Flatten nested available objects into a linear candidate list."""
    out: List[Any] = []

    def _collect(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for x in item:
                _collect(x)
            return
        if isinstance(item, dict):
            for x in item.values():
                _collect(x)
            return
        if item is not None:
            out.append(item)

    _collect(available)
    return out


def _parse_point_yx(point_like: Any) -> Optional[Tuple[int, int]]:
    """Parse [y, x]-style point into integer tuple."""
    if not isinstance(point_like, (list, tuple)) or len(point_like) < 2:
        return None
    try:
        y = int(point_like[0])
        x = int(point_like[1])
    except (TypeError, ValueError):
        return None
    return y, x


def find_exact_option_index(action_label: Any, option_metadata: List[Dict[str, Any]]) -> int:
    """Find exact option index by label, return -1 if not found."""
    if not isinstance(action_label, str):
        return -1
    for i, opt in enumerate(option_metadata):
        if opt.get("label") == action_label:
            return i
    return -1


def select_target_with_point(
    seg_raw: Optional[np.ndarray],
    seg_id_map: Dict[Any, Any],
    available: Any,
    point_like: Any,
) -> Optional[Dict[str, Any]]:
    """
    Select target actor using strict+nearest policy.

    Policy:
    1. Parse/clip point [y, x] in image range.
    2. Try direct hit on seg_raw[y, x] and require hit object in available set.
    3. If direct hit fails, use nearest centroid among available objects with valid masks.
    4. Return None when target cannot be determined.
    """
    if seg_raw is None:
        return None

    point_yx = _parse_point_yx(point_like)
    if point_yx is None:
        return None

    if not isinstance(seg_raw, np.ndarray) or seg_raw.size == 0:
        return None

    h, w = seg_raw.shape[:2]
    y, x = point_yx
    y = max(0, min(y, h - 1))
    x = max(0, min(x, w - 1))

    candidates = _flatten_available(available)
    if not candidates:
        return None
    candidate_ids = {id(obj) for obj in candidates}

    # Build object -> seg_ids map once from seg_id_map.
    obj_to_seg_ids: Dict[int, List[int]] = {}
    for raw_sid, obj in (seg_id_map or {}).items():
        try:
            sid = int(raw_sid)
        except (TypeError, ValueError):
            continue
        if obj is None:
            continue
        oid = id(obj)
        if oid in candidate_ids:
            obj_to_seg_ids.setdefault(oid, []).append(sid)

    # Direct hit first.
    try:
        direct_sid = int(seg_raw[y, x])
    except Exception:
        direct_sid = None

    if direct_sid is not None:
        direct_obj = (seg_id_map or {}).get(direct_sid)
        if direct_obj is not None and id(direct_obj) in candidate_ids:
            return {
                "obj": direct_obj,
                "name": getattr(direct_obj, "name", f"id_{direct_sid}"),
                "seg_id": direct_sid,
                "click_point": (int(x), int(y)),
                "centroid_point": (int(x), int(y)),
            }

    # Nearest centroid among available objects with valid masks.
    best: Optional[Dict[str, Any]] = None
    min_dist = float("inf")

    for obj in candidates:
        oid = id(obj)
        for sid in obj_to_seg_ids.get(oid, []):
            mask = seg_raw == sid
            if not np.any(mask):
                continue
            ys, xs = np.nonzero(mask)
            cx = float(xs.mean())
            cy = float(ys.mean())
            dist = (cx - x) ** 2 + (cy - y) ** 2
            if dist < min_dist:
                min_dist = dist
                best = {
                    "obj": obj,
                    "name": getattr(obj, "name", f"id_{sid}"),
                    "seg_id": sid,
                    "click_point": (int(x), int(y)),
                    "centroid_point": (int(cx), int(cy)),
                }

    return best
