"""
可视化 test：模仿 select_target_with_point 的完整过程。

流程：
1. 构造合成 seg_raw、seg_id_map、available、point_like
2. 调用 select_target_with_point 得到选中目标
3. 绘制分割图 + 点击点 + 质心，保存到 tests/ 目录。
"""
from pathlib import Path

import numpy as np
import random

# 可选：若环境有 cv2 则用于保存图像
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# 仅加载 oracle_action_matcher，避免拉取 RecordWrapper/gymnasium 等依赖
import importlib.util

_src_root = Path(__file__).resolve().parents[1]
_matcher_path = _src_root / "src" / "robomme" / "env_record_wrapper" / "oracle_action_matcher.py"
_spec = importlib.util.spec_from_file_location("oracle_action_matcher", _matcher_path)
_matcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_matcher)
normalize_and_clip_point_xy = _matcher.normalize_and_clip_point_xy
select_target_with_point = _matcher.select_target_with_point


def _make_mock_actor(name: str):
    """构造带 name 的 mock actor，用于 seg_id_map 和 available。"""
    return type("MockActor", (), {"name": name})()


def _build_synthetic_scene(height=240, width=320):
    """
    构造合成场景（2 个对象均在 segment 中）：
    - seg_raw: 两个物体区域，seg_id 分别为 1、2
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
    构造 3 个对象、仅 2 个在 segment 中可见的场景：
    - seg_raw: 只有 seg_id 1、2 有像素；seg_id 3 不在图中（物体 C 不可见）
    - seg_id_map: 1 -> A, 2 -> B, 3 -> C
    - available: [A, B, C]
    点击不命中任何 mask 时，fallback 可能选中 C，此时 seg_id/centroid 为 None。
    """
    seg_raw = np.zeros((height, width), dtype=np.int64)
    seg_raw[40:120, 30:130] = 1   # 物体 A 可见
    seg_raw[80:200, 180:280] = 2  # 物体 B 可见
    # 物体 C 对应 seg_id=3，但 seg_raw 中没有任何 3 -> C 不在 segment 中

    actor_a = _make_mock_actor("object_A")
    actor_b = _make_mock_actor("object_B")
    actor_c = _make_mock_actor("object_C")
    seg_id_map = {1: actor_a, 2: actor_b, 3: actor_c}
    available = [actor_a, actor_b, actor_c]

    return seg_raw, seg_id_map, available


def _colorize_seg(seg_raw, color_map=None):
    """把 seg_raw 转成 BGR 图，用于可视化。"""
    if color_map is None:
        color_map = {
            1: [0, 180, 255],   # BGR 橙
            2: [255, 180, 0],   # BGR 蓝
        }
    h, w = seg_raw.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[:] = [40, 40, 40]  # 背景深灰
    for seg_id in np.unique(seg_raw):
        if seg_id <= 0:
            continue
        color = color_map.get(seg_id, [200, 200, 200])
        mask = seg_raw == seg_id
        vis[mask] = color
    return vis


def _draw_result(vis_bgr, result, label_prefix=""):
    """在 vis_bgr 上画 click_point（红）和 centroid_point（绿），并加文字。"""
    if result is None:
        return vis_bgr
    out = vis_bgr.copy()
    # 点击点：红色圆
    click = result.get("click_point")
    if click is not None:
        cx, cy = click
        cv2.circle(out, (int(cx), int(cy)), 8, (0, 0, 255), 2)
        cv2.putText(
            out, f"{label_prefix}click", (cx + 12, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
        )
    # 质心：绿色圆
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
    """模仿 select_target_with_point 的完整过程并生成可视化图。"""
    if not HAS_CV2:
        raise RuntimeError("此可视化测试需要 opencv-python，请安装: uv add opencv-python")

    out_dir = Path(__file__).resolve().parent  # tests/
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_raw, seg_id_map, available = _build_synthetic_scene()
    h, w = seg_raw.shape[:2]

    # 用例 1：点击落在物体 1 上 -> 应命中 object_A
    point_on_obj1 = (80, 70)  # (x,y) 在 seg_id=1 区域内
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

    # 可视化：hit 用例
    vis_base = _colorize_seg(seg_raw)
    vis_hit = _draw_result(vis_base.copy(), result_hit, "hit ")
    path_hit = out_dir / "select_target_hit.png"
    cv2.imwrite(str(path_hit), vis_hit)
    print(f"Hit case:      obj={result_hit['name']} seg_id={result_hit['seg_id']} -> {path_hit}")

    # 用例 2（fallback）：3 个对象，仅 2 个在 segment 中；点击不命中时可能选中「不在 segment 中」的那个
    seg_raw_3, seg_id_map_3, available_3 = _build_synthetic_scene_three_objects_one_missing()
    h3, w3 = seg_raw_3.shape[:2]
    point_on_bg = (160, 120)

    # 多试几次种子，得到两种结果：选中可见对象 / 选中不可见对象（object_C）
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

    # 至少应能随机到可见对象；再试更多种子以得到不可见对象
    assert result_fallback_visible is not None, "fallback 应能选中在 segment 中的对象"
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
    assert result_fallback_invisible is not None, "fallback 应能选中不在 segment 中的对象（object_C）"

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
    """可视化 normalize_and_clip_point_xy：不同输入在图像上的合法范围。"""
    if not HAS_CV2:
        raise RuntimeError("此可视化测试需要 opencv-python，请安装: uv add opencv-python")

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
