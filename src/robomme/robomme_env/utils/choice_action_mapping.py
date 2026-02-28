from typing import Any, Dict, List, Optional

import numpy as np


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


def _to_numpy_array(value: Any, dtype: np.dtype = np.float64) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    try:
        arr = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError):
        return None
    return arr


def normalize_pixel_xy(pixel_like: Any) -> Optional[np.ndarray]:
    arr = _to_numpy_array(pixel_like, dtype=np.float64)
    if arr is None:
        return None
    arr = arr.reshape(-1)
    if arr.size < 2:
        return None
    pixel = arr[:2]
    if not np.all(np.isfinite(pixel)):
        return None
    return pixel


def normalize_position_xyz(position_like: Any) -> Optional[np.ndarray]:
    arr = _to_numpy_array(position_like, dtype=np.float64)
    if arr is None:
        return None
    arr = arr.reshape(-1)
    if arr.size < 3:
        return None
    pos = arr[:3]
    if not np.all(np.isfinite(pos)):
        return None
    return pos


def extract_actor_position_xyz(actor: Any) -> Optional[np.ndarray]:
    pose = getattr(actor, "pose", None)
    if pose is None and hasattr(actor, "get_pose"):
        try:
            pose = actor.get_pose()
        except Exception:
            return None
    if pose is None:
        return None
    pos = getattr(pose, "p", None)
    if pos is None:
        return None
    return normalize_position_xyz(pos)


def _normalize_intrinsic_cv(intrinsic_cv: Any) -> Optional[np.ndarray]:
    intrinsic = _to_numpy_array(intrinsic_cv, dtype=np.float64)
    if intrinsic is None:
        return None
    intrinsic = intrinsic.reshape(-1)
    if intrinsic.size < 9:
        return None
    intrinsic = intrinsic[:9].reshape(3, 3)
    if not np.all(np.isfinite(intrinsic)):
        return None
    return intrinsic


def _normalize_extrinsic_cv(extrinsic_cv: Any) -> Optional[np.ndarray]:
    extrinsic = _to_numpy_array(extrinsic_cv, dtype=np.float64)
    if extrinsic is None:
        return None
    extrinsic = extrinsic.reshape(-1)
    if extrinsic.size < 12:
        return None
    extrinsic = extrinsic[:12].reshape(3, 4)
    if not np.all(np.isfinite(extrinsic)):
        return None
    return extrinsic


def _normalize_image_shape(image_shape: Any) -> Optional[tuple[int, int]]:
    if image_shape is None:
        return None
    try:
        shape_arr = np.asarray(image_shape, dtype=np.int64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if shape_arr.size < 2:
        return None
    h = int(shape_arr[0])
    w = int(shape_arr[1])
    if h <= 0 or w <= 0:
        return None
    return h, w


def project_world_to_pixel(
    world_xyz: Any,
    intrinsic_cv: Any,
    extrinsic_cv: Any,
    image_shape: Any,
) -> Optional[List[int]]:
    world = normalize_position_xyz(world_xyz)
    intrinsic = _normalize_intrinsic_cv(intrinsic_cv)
    extrinsic = _normalize_extrinsic_cv(extrinsic_cv)
    hw = _normalize_image_shape(image_shape)
    if world is None or intrinsic is None or extrinsic is None or hw is None:
        return None

    def _project(extrinsic_mat: np.ndarray) -> Optional[List[int]]:
        world_h = np.concatenate([world, [1.0]], axis=0)
        camera_xyz = extrinsic_mat @ world_h
        z = float(camera_xyz[2])
        if not np.isfinite(z) or z <= 1e-8:
            return None

        pixel_h = intrinsic @ camera_xyz
        x = float(pixel_h[0] / z)
        y = float(pixel_h[1] / z)
        if not np.isfinite(x) or not np.isfinite(y):
            return None

        px = int(np.rint(x))
        py = int(np.rint(y))
        h, w = hw
        if px < 0 or px >= w or py < 0 or py >= h:
            return None
        return [px, py]

    # Most pipelines use extrinsic_cv as world->camera.
    projected = _project(extrinsic)
    if projected is not None:
        return projected

    # Fallback: treat extrinsic_cv as camera->world and invert to world->camera.
    extrinsic_4x4 = np.eye(4, dtype=np.float64)
    extrinsic_4x4[:3, :4] = extrinsic
    try:
        world_to_camera = np.linalg.inv(extrinsic_4x4)[:3, :4]
    except np.linalg.LinAlgError:
        return None
    return _project(world_to_camera)


def select_target_with_position(
    available: Any,
    position_like: Any,
) -> Optional[Dict[str, Any]]:
    target_pos = normalize_position_xyz(position_like)
    if target_pos is None:
        return None

    unique_candidates = _unique_candidates(available)
    if not unique_candidates:
        return None

    best_actor: Optional[Any] = None
    best_pos: Optional[np.ndarray] = None
    best_dist: Optional[float] = None

    for actor in unique_candidates:
        actor_pos = extract_actor_position_xyz(actor)
        if actor_pos is None:
            continue
        dist = float(np.linalg.norm(actor_pos - target_pos))
        if best_dist is None or dist < best_dist:
            best_actor = actor
            best_pos = actor_pos
            best_dist = dist

    if best_actor is None or best_pos is None or best_dist is None:
        return None

    return {
        "obj": best_actor,
        "name": getattr(best_actor, "name", "unknown"),
        "position": best_pos.astype(np.float64).tolist(),
        "match_distance": best_dist,
        "selection_mode": "nearest_position",
    }


def select_target_with_pixel(
    available: Any,
    pixel_like: Any,
    intrinsic_cv: Any,
    extrinsic_cv: Any,
    image_shape: Any,
) -> Optional[Dict[str, Any]]:
    target_pixel = normalize_pixel_xy(pixel_like)
    if target_pixel is None:
        return None

    unique_candidates = _unique_candidates(available)
    if not unique_candidates:
        return None

    best_actor: Optional[Any] = None
    best_pos: Optional[np.ndarray] = None
    best_pixel: Optional[List[int]] = None
    best_dist: Optional[float] = None

    for actor in unique_candidates:
        actor_pos = extract_actor_position_xyz(actor)
        if actor_pos is None:
            continue
        projected = project_world_to_pixel(
            actor_pos,
            intrinsic_cv=intrinsic_cv,
            extrinsic_cv=extrinsic_cv,
            image_shape=image_shape,
        )
        if projected is None:
            continue
        projected_np = np.asarray(projected, dtype=np.float64)
        dist = float(np.linalg.norm(projected_np - target_pixel))
        if best_dist is None or dist < best_dist:
            best_actor = actor
            best_pos = actor_pos
            best_pixel = projected
            best_dist = dist

    if best_actor is None or best_pos is None or best_dist is None or best_pixel is None:
        return None

    return {
        "obj": best_actor,
        "name": getattr(best_actor, "name", "unknown"),
        "position": best_pos.astype(np.float64).tolist(),
        "projected_pixel": [int(best_pixel[0]), int(best_pixel[1])],
        "match_distance": best_dist,
        "selection_mode": "nearest_pixel_projection",
    }
