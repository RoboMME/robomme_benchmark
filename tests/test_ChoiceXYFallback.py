"""
Visualization test: mimic the full process of select_target_with_point.

Flow:
1. Build synthetic seg_raw、seg_id_map、available、point_like
2. Call select_target_with_point to get selected target
3. Draw segmentation + click point + centroid, and save to tests/ directory.
"""
from pathlib import Path

import numpy as np
import random

# Optional: if cv2 is available, use it to save images
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Load only oracle_action_matcher to avoid pulling dependencies like RecordWrapper/gymnasium
import importlib.util

_src_root = Path(__file__).resolve().parents[1]
_matcher_path = _src_root / "src" / "robomme" / "robomme_env" / "utils" / "oracle_action_matcher.py"
_spec = importlib.util.spec_from_file_location("oracle_action_matcher", _matcher_path)
_matcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_matcher)
normalize_and_clip_point_xy = _matcher.normalize_and_clip_point_xy
select_target_with_point = _matcher.select_target_with_point


def _make_mock_actor(name: str):
    """Build a mock actor with a name, used for seg_id_map and available."""
    return type("MockActor", (), {"name": name})()


def _build_synthetic_scene(height=240, width=320):
    """
    Build a synthetic scene (both 2 objects are visible in the segment):
    - seg_raw: two object regions with seg_id 1 and 2
    - seg_id_map: 1 -> actor_a, 2 -> actor_b
    - available: [actor_a, actor_b]
    """
    seg_raw = np.zeros((height, width), dtype=np.int64)
    seg_raw[40:120, 30:130] = 1
    seg_raw[80:200, 180:280] = 2

    actor_a = _make_mock_actor("object_A")
    actor_b = _make_mock_actor("object_B")
    seg_id_map = {1: actor_a, 2: actor_b}
    available = [actor_a, actor_b]

    return seg_raw, seg_id_map, available


def _build_synthetic_scene_three_objects_one_missing(height=240, width=320):
    """
    Build a scene with 3 objects where only 2 are visible in the segment:
    - seg_raw: only seg_id 1 and 2 have pixels; seg_id 3 is absent (object C is invisible)
    - seg_id_map: 1 -> A, 2 -> B, 3 -> C
    - available: [A, B, C]
    When click misses all masks, fallback may select C; then seg_id/centroid is None.
    """
    seg_raw = np.zeros((height, width), dtype=np.int64)
    seg_raw[40:120, 30:130] = 1   # object A visible
    seg_raw[80:200, 180:280] = 2  # object B visible
    # Object C maps to seg_id=3, but seg_raw contains no 3 -> C is not in the segment

    actor_a = _make_mock_actor("object_A")
    actor_b = _make_mock_actor("object_B")
    actor_c = _make_mock_actor("object_C")
    seg_id_map = {1: actor_a, 2: actor_b, 3: actor_c}
    available = [actor_a, actor_b, actor_c]

    return seg_raw, seg_id_map, available


def _colorize_seg(seg_raw, color_map=None):
    """Convert seg_raw to a BGR image for visualization."""
    if color_map is None:
        color_map = {
            1: [0, 180, 255],   # BGR orange
            2: [255, 180, 0],   # BGR blue
        }
    h, w = seg_raw.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[:] = [40, 40, 40]  # dark gray background
    for seg_id in np.unique(seg_raw):
        if seg_id <= 0:
            continue
        color = color_map.get(seg_id, [200, 200, 200])
        mask = seg_raw == seg_id
        vis[mask] = color
    return vis


def _draw_result(vis_bgr, result, label_prefix=""):
    """Draw click_point (red) and centroid_point (green) on vis_bgr, and annotate text."""
    if result is None:
        return vis_bgr
    out = vis_bgr.copy()
    # Click point: red circle
    click = result.get("click_point")
    if click is not None:
        cx, cy = click
        cv2.circle(out, (int(cx), int(cy)), 8, (0, 0, 255), 2)
        cv2.putText(
            out, f"{label_prefix}click", (cx + 12, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
        )
    # Centroid: green circle
    centroid = result.get("centroid_point")
    if centroid is not None:
        cx, cy = centroid
        cv2.circle(out, (int(cx), int(cy)), 6, (0, 255, 0), -1)
        cv2.putText(
            out, f"{label_prefix}centroid", (cx + 12, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )
    name = result.get("name", "?")
    seg_id = result.get("seg_id", "?")
    cv2.putText(
        out, f"obj={name} seg_id={seg_id}", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
    )
    return out


def test_select_target_with_point_process_visualization():
    """Mimic the full select_target_with_point process and generate visualization images."""
    if not HAS_CV2:
        raise RuntimeError("This visualization test requires opencv-python, install with: uv add opencv-python")

    out_dir = Path(__file__).resolve().parent  # tests/
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_raw, seg_id_map, available = _build_synthetic_scene()
    h, w = seg_raw.shape[:2]

    # case 1：click lands on object 1 -> should hit object_A
    point_on_obj1 = (80, 70)  # (x,y) inside seg_id=1 region
    clipped1 = normalize_and_clip_point_xy(point_on_obj1, width=w, height=h)
    assert clipped1 is not None

    result_hit = select_target_with_point(
        seg_raw=seg_raw,
        seg_id_map=seg_id_map,
        available=available,
        point_like=point_on_obj1,
    )
    assert result_hit is not None
    assert result_hit.get("name") == "object_A"
    assert result_hit.get("seg_id") == 1
    assert result_hit.get("centroid_point") is not None

    # Visualization: hit case
    vis_base = _colorize_seg(seg_raw)
    vis_hit = _draw_result(vis_base.copy(), result_hit, "hit ")
    path_hit = out_dir / "select_target_hit.png"
    cv2.imwrite(str(path_hit), vis_hit)
    print(f"Hit case:      obj={result_hit['name']} seg_id={result_hit['seg_id']} -> {path_hit}")

    # case 2（fallback）：3 objects, only 2 in segment; when click misses, fallback may pick the one not in segment
    seg_raw_3, seg_id_map_3, available_3 = _build_synthetic_scene_three_objects_one_missing()
    h3, w3 = seg_raw_3.shape[:2]
    point_on_bg = (160, 120)

    # Try multiple seeds to get both outcomes: visible object selected / invisible object selected (object_C)
    result_fallback_visible = None
    result_fallback_invisible = None
    for seed in range(50):
        random.seed(seed)
        res = select_target_with_point(
            seg_raw=seg_raw_3,
            seg_id_map=seg_id_map_3,
            available=available_3,
            point_like=point_on_bg,
        )
        assert res is not None and res.get("click_point") is not None
        if res.get("name") == "object_C" and res.get("seg_id") is None:
            result_fallback_invisible = res
            if result_fallback_visible is not None:
                break
        elif res.get("seg_id") is not None:
            result_fallback_visible = res
            if result_fallback_invisible is not None:
                break

    # Should at least randomly pick a visible object; then try more seeds to get an invisible object
    assert result_fallback_visible is not None, "fallback should be able to select an object that is in the segment"
    for seed in range(50, 200):
        random.seed(seed)
        res = select_target_with_point(
            seg_raw=seg_raw_3,
            seg_id_map=seg_id_map_3,
            available=available_3,
            point_like=point_on_bg,
        )
        if res.get("name") == "object_C" and res.get("seg_id") is None:
            result_fallback_invisible = res
            break
    assert result_fallback_invisible is not None, "fallback should be able to select an object not in the segment (object_C)"

    color_map_3 = {1: [0, 180, 255], 2: [255, 180, 0], 3: [200, 200, 200]}
    vis_base_3 = _colorize_seg(seg_raw_3, color_map=color_map_3)
    vis_fallback_visible = _draw_result(vis_base_3.copy(), result_fallback_visible, "fallback_visible ")
    vis_fallback_invisible = _draw_result(vis_base_3.copy(), result_fallback_invisible, "fallback_invisible ")

    path_fallback_visible = out_dir / "select_target_fallback_visible.png"
    path_fallback_invisible = out_dir / "select_target_fallback_invisible.png"
    cv2.imwrite(str(path_fallback_visible), vis_fallback_visible)
    cv2.imwrite(str(path_fallback_invisible), vis_fallback_invisible)

    print(f"Fallback (visible):   obj={result_fallback_visible['name']} seg_id={result_fallback_visible['seg_id']} -> {path_fallback_visible}")
    print(f"Fallback (invisible): obj={result_fallback_invisible['name']} seg_id={result_fallback_invisible['seg_id']} -> {path_fallback_invisible}")
    assert path_hit.exists() and path_fallback_visible.exists() and path_fallback_invisible.exists()


def test_normalize_and_clip_point_xy_visualization():
    """Visualize normalize_and_clip_point_xy: valid ranges of different inputs on the image."""
    if not HAS_CV2:
        raise RuntimeError("This visualization test requires opencv-python, install with: uv add opencv-python")

    out_dir = Path(__file__).resolve().parent  # tests/
    out_dir.mkdir(parents=True, exist_ok=True)

    w, h = 320, 240
    vis = np.ones((h, w, 3), dtype=np.uint8) * 240
    cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 128, 0), 1)

    test_inputs = [
        ((50, 50), "normal"),
        ((100, 150), "normal"),
        ((-10, 100), "clip_x"),
        ((400, 100), "clip_x"),
        ([80, 80], "list"),
    ]
    y_text = 20
    for pt, desc in test_inputs:
        clipped = normalize_and_clip_point_xy(pt, width=w, height=h)
        if clipped is None:
            cv2.putText(vis, f"None: {desc}", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            y_text += 18
            continue
        cx, cy = clipped
        was_clipped = (isinstance(pt, (list, tuple)) and len(pt) >= 2 and (pt[0] != cx or pt[1] != cy))
        color = (0, 0, 255) if was_clipped else (0, 128, 0)
        cv2.circle(vis, (int(cx), int(cy)), 4, color, -1)
        cv2.putText(vis, f"{desc}->({cx},{cy})", (cx + 6, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        y_text += 18

    path = out_dir / "normalize_clip_points.png"
    cv2.imwrite(str(path), vis)
    print(f"Normalize/clip points -> {path}")
    assert path.exists()
