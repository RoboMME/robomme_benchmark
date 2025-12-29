import os
import sys

# 添加项目根目录到 Python 路径，以便正确导入模块
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import cv2
import torch

from historybench.HistoryBench_env.util.vqa_options import get_vqa_options


# NLP 语义匹配（可选）
_NLP_MODEL = None
_ST_UTIL = None
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    print("Loading NLP Model (all-MiniLM-L6-v2)...")
    _NLP_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    _ST_UTIL = st_util
    print("NLP Model loaded.")
except ImportError:
    print("Warning: sentence-transformers not found. NLP matching will fail.")
except Exception as e:
    print(f"Error loading NLP model: {e}")


# =============================================================================
# 辅助函数
# =============================================================================

def _prepare_frame(frame):
    """预处理帧数据为 uint8 格式"""
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        max_val = float(np.max(frame)) if frame.size else 0.0
        if max_val <= 1.0:
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    return frame


def _prepare_segmentation_visual(segmentation, color_map, target_hw):
    """将分割数据转换为可视化图像"""
    if segmentation is None:
        return None, None

    seg = segmentation
    if hasattr(seg, "cpu"):
        seg = seg.cpu().numpy()
    seg = np.asarray(seg)
    if seg.ndim > 2:
        seg = seg[0]
    seg_2d = seg.squeeze().astype(np.int64)

    h, w = seg_2d.shape[:2]
    seg_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    unique_ids = np.unique(seg_2d)
    for seg_id in unique_ids:
        if seg_id <= 0:
            continue
        color = color_map.get(int(seg_id))
        if color is None:
            continue
        seg_rgb[seg_2d == seg_id] = color
    seg_bgr = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)

    target_h, target_w = target_hw
    if seg_bgr.shape[:2] != (target_h, target_w):
        seg_bgr = cv2.resize(seg_bgr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    return seg_bgr, seg_2d


def _fetch_segmentation(env):
    """从环境获取分割数据"""
    obs = env.unwrapped.get_obs(unflattened=True)
    return obs["sensor_data"]["base_camera"]["segmentation"]


def _build_solve_options(env, planner, selected_target, env_id):
    """构建可用的动作选项"""
    return get_vqa_options(env, planner, selected_target, env_id)


def _find_best_semantic_match(user_query, options):
    """使用 NLP 语义匹配找到最佳选项"""
    if _NLP_MODEL is None or _ST_UTIL is None:
        return -1, 0.0
    
    if not options:
        return -1, 0.0

    labels = [opt.get("label", "") for opt in options]
    query_text = str(user_query or "").strip()

    try:
        query_embedding = _NLP_MODEL.encode(query_text, convert_to_tensor=True)
        corpus_embeddings = _NLP_MODEL.encode(labels, convert_to_tensor=True)
        cos_scores = _ST_UTIL.cos_sim(query_embedding, corpus_embeddings)[0]
        best_idx = int(torch.argmax(cos_scores).item())
        best_score = float(cos_scores[best_idx].item())
    except Exception as exc:
        print(f"  [NLP] Semantic match failed ({exc}); defaulting to option 1.")
        return 0, 0.0

    print(f"  [NLP] Closest Match: '{query_text}' -> '{labels[best_idx]}' (Score: {best_score:.4f})")
    
    return best_idx, best_score


# =============================================================================
# 核心函数: step_before 和 step_after
# =============================================================================

def step_before(env, planner, env_id, color_map, use_segmentation=False):
    """
    在执行动作之前调用，获取当前环境状态和可用选项。
    
    Args:
        env: 环境对象
        planner: 规划器对象
        env_id: 环境 ID
        color_map: 颜色映射表
        use_segmentation: 是否使用分割可视化
    
    Returns:
        seg_vis: 分割可视化图像 (BGR)
        seg_raw: 原始分割数据
        base_frames: 基础相机帧列表
        wrist_frames: 腕部相机帧列表
        available_options: 可用动作选项列表
    """
    # 1. 获取帧数据
    base_frames = getattr(env, "frames", [])
    if not base_frames:
        base_frames = getattr(env.unwrapped, "frames", []) or []
        
    wrist_frames = getattr(env, "wrist_frames", [])
    if not wrist_frames:
        wrist_frames = getattr(env.unwrapped, "wrist_frames", []) or []

    # 2. 获取分割数据
    seg_data = _fetch_segmentation(env)
    
    # 3. 确定分辨率
    seg_hw = (255, 255)  # 默认
    if base_frames and len(base_frames) > 0:
        seg_hw = base_frames[-1].shape[:2]
    elif seg_data is not None:
        try:
            temp = seg_data
            if hasattr(temp, "cpu"):
                temp = temp.cpu().numpy()
            temp = np.asarray(temp)
            if temp.ndim > 2:
                temp = temp[0]
            seg_hw = temp.shape[:2]
        except Exception:
            pass

    # 4. 处理分割可视化
    seg_vis = None
    seg_raw = None

    if use_segmentation:
        seg_vis, seg_raw = _prepare_segmentation_visual(seg_data, color_map, seg_hw)
    else:
        _, seg_raw = (_prepare_segmentation_visual(seg_data, color_map, seg_hw) 
                      if seg_data is not None else (None, None))
        if base_frames:
            vis_frame = _prepare_frame(base_frames[-1])
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            if vis_frame.shape[:2] != seg_hw:
                vis_frame = cv2.resize(vis_frame, (seg_hw[1], seg_hw[0]), interpolation=cv2.INTER_LINEAR)
            seg_vis = vis_frame
    
    if seg_vis is None:
        seg_vis = np.zeros((seg_hw[0], seg_hw[1], 3), dtype=np.uint8)

    # 5. 构建可用选项
    dummy_target = {"obj": None, "name": None, "seg_id": None, "click_point": None, "centroid_point": None}
    raw_options = _build_solve_options(env, planner, dummy_target, env_id)
    available_options = [{"action": opt.get("label", "Unknown"), "need_parameter": bool(opt.get("available"))} 
                         for opt in raw_options]

    return seg_vis, seg_raw, base_frames, wrist_frames, available_options


def step_after(env, planner, env_id, seg_vis, seg_raw, base_frames, wrist_frames, command_dict):
    """
    在收到命令后执行动作并返回评估结果。
    
    Args:
        env: 环境对象
        planner: 规划器对象
        env_id: 环境 ID
        seg_vis: 分割可视化图像
        seg_raw: 原始分割数据
        base_frames: 基础相机帧列表
        wrist_frames: 腕部相机帧列表
        command_dict: 命令字典，包含 'action' 和 'point'
    
    Returns:
        evaluation: 评估结果字典
    """
    # 1. 构建选项
    selected_target = {"obj": None, "name": None, "seg_id": None, "click_point": None, "centroid_point": None}
    solve_options = _build_solve_options(env, planner, selected_target, env_id)
    
    target_action = command_dict.get("action")
    target_param = command_dict.get("point")

    if "action" not in command_dict:
        return None
    if target_action is None:
        return None

    # 2. 查找匹配的动作选项
    found_idx = -1
    for i, opt in enumerate(solve_options):
        if opt.get("label") == target_action or str(i + 1) == str(target_action):
            found_idx = i
            break
    
    # 3. 如果精确匹配失败，尝试语义匹配
    if found_idx == -1 and isinstance(target_action, str) and not target_action.isdigit():
        print(f"Attempting semantic match for: '{target_action}'")
        found_idx, score = _find_best_semantic_match(target_action, solve_options)
    
    if found_idx == -1:
        print(f"Error: Action '{target_action}' not found in current options.")
        return None

    # 4. 处理点击坐标，解析目标对象
    if target_param is not None and seg_raw is not None:
        cx, cy = target_param
        h, w = seg_raw.shape[:2]
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))
        
        seg_id_map = getattr(env.unwrapped, "segmentation_id_map", {}) or {}
        
        # 收集可用候选对象
        candidates = []
        def _collect(item):
            if isinstance(item, (list, tuple)):
                for x in item:
                    _collect(x)
            elif isinstance(item, dict):
                for x in item.values():
                    _collect(x)
            else:
                if item:
                    candidates.append(item)
        
        avail = solve_options[found_idx].get("available")
        if avail:
            _collect(avail)
            best_cand = None
            min_dist = float('inf')
            for actor in candidates:
                target_ids = [sid for sid, obj in seg_id_map.items() if obj is actor]
                for tid in target_ids:
                    tid = int(tid)
                    mask = (seg_raw == tid)
                    if np.any(mask):
                        ys, xs = np.nonzero(mask)
                        center_x, center_y = xs.mean(), ys.mean()
                        dist = (center_x - cx) ** 2 + (center_y - cy) ** 2
                        if dist < min_dist:
                            min_dist = dist
                            best_cand = {
                                "obj": actor,
                                "name": getattr(actor, "name", f"id_{tid}"),
                                "seg_id": tid,
                                "click_point": (int(cx), int(cy)),
                                "centroid_point": (int(center_x), int(center_y))
                            }
            if best_cand:
                selected_target.update(best_cand)
            else:
                selected_target["click_point"] = (int(cx), int(cy))
        else:
            selected_target["click_point"] = (int(cx), int(cy))

    # 5. 执行动作
    print(f"Executing Option: {found_idx + 1} - {solve_options[found_idx].get('label')}")
    solve_options[found_idx].get("solve")()

    # 6. 评估结果
    env.unwrapped.evaluate()
    evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
    print(f"Evaluation: {evaluation}")
    return evaluation

