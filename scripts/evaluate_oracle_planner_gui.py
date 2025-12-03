import os
import sys
import select
import numpy as np
import sapien
import gymnasium as gym
import cv2
import colorsys
import h5py
from pathlib import Path
import torch

# --- NLP Imports ---
try:
    # [FIX] Rename 'util' to 'st_util' to avoid conflict with historybench.util
    from sentence_transformers import SentenceTransformer, util as st_util
except ImportError as exc:
    raise ImportError(
        "Missing dependency 'sentence-transformers'."
        " Semantic matching relies on it; please install via `pip install sentence-transformers`."
    ) from exc

print("Loading NLP Model (all-MiniLM-L6-v2)...")
_NLP_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
print("NLP Model loaded.")

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None

# Ensure script can find root modules
_ROOT = os.path.abspath(os.path.dirname(__file__))
_PARENT = os.path.dirname(_ROOT)
for _path in (_PARENT, _ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# --- Project Imports ---
from historybench.env_record_wrapper import EpisodeConfigResolver
from historybench.HistoryBench_env import *
from historybench.env_record_wrapper import *
from historybench.HistoryBench_env.util import *
from historybench.HistoryBench_env.util import task_goal
from historybench.HistoryBench_env.util.vqa_options import get_vqa_options

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
    PandaStickMotionPlanningSolver,
)

def _on_mouse_click(event, x, y, flags, param):
    """Callback for capturing mouse clicks."""
    click_state = param
    if event == cv2.EVENT_LBUTTONDOWN and isinstance(click_state, dict):
        click_state["coords"] = (x, y)
        print(f"  -> Mouse Clicked at: {x}, {y}")

def _generate_color_map(n=10000, s_min=0.70, s_max=0.95, v_min=0.78, v_max=0.95):
    phi = 0.6180339887498948
    color_map = {}
    for i in range(1, n + 1):
        h = (i * phi) % 1.0
        s = s_min + (s_max - s_min) * ((i % 7) / 6)
        v = v_min + (v_max - v_min) * (((i * 3) % 5) / 4)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        color_map[i] = [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))]
    return color_map

def _print_terminal_options(options):
    """Mirror GUI option list in terminal for quick reference."""
    if not options:
        return
    print("\n[Options]")
    for idx, opt in enumerate(options):
        label = opt.get("label", f"Option {idx + 1}")
        print(f"  {idx + 1}. {label}")

def _sync_table_color(env, color_map):
    seg_id_map = getattr(env.unwrapped, "segmentation_id_map", None)
    if not isinstance(seg_id_map, dict):
        return
    for obj_id, obj in seg_id_map.items():
        if getattr(obj, "name", None) == "table-workspace":
            color_map[obj_id] = [0, 0, 0]

def _tensor_to_bool(value):
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)

def _prepare_frame(frame):
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
    try:
        obs = env.unwrapped.get_obs(unflattened=True)
        return obs["sensor_data"]["base_camera"]["segmentation"]
    except Exception:
        return None

def _build_solve_options(env, planner, selected_target, env_id):
    return get_vqa_options(env, planner, selected_target, env_id)

def _display_step_image(title, img_bgr, window_size=(6, 4)):
    """Show an image in a non-blocking OpenCV window."""
    win_name = title
    try:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        if window_size:
            target_w = max(100, int(window_size[0] * 100))
            target_h = max(100, int(window_size[1] * 100))
            cv2.resizeWindow(win_name, target_w, target_h)
        cv2.imshow(win_name, img_bgr)
        cv2.waitKey(1)
    except Exception:
        win_name = None
    return win_name

def _cleanup_figures(figures, close_all=False):
    """Close OpenCV preview windows."""
    if figures:
        for win_name in figures:
            if not win_name:
                continue
            try:
                cv2.destroyWindow(win_name)
            except Exception:
                pass
        figures.clear()

    if close_all:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

def _find_best_semantic_match(user_query, options):
    """
    Finds the index of the option label that best matches the user_query semantically.
    ALWAYS returns the best match, regardless of score.
    """
    if _NLP_MODEL is None:
        return -1, 0.0
    
    if not options or not user_query:
        return -1, 0.0

    # Extract labels from options
    labels = [opt.get("label", "") for opt in options]
    
    # Encode and compute similarity
    query_embedding = _NLP_MODEL.encode(user_query, convert_to_tensor=True)
    corpus_embeddings = _NLP_MODEL.encode(labels, convert_to_tensor=True)
    
    # [FIX] Use st_util.cos_sim instead of util.cos_sim
    cos_scores = st_util.cos_sim(query_embedding, corpus_embeddings)[0]

    # Find the best score (Argmax)
    best_score = torch.max(cos_scores)
    best_idx = torch.argmax(cos_scores).item()

    # LOGIC CHANGE: No threshold check. We always accept the result.
    print(f"  [NLP] Closest Match: '{user_query}' -> '{labels[best_idx]}' (Score: {best_score:.4f})")
    
    return best_idx, best_score.item()

# --- Core Logic Function ---
def _prompt_next_task_gui(
    options,
    env,
    selected_target,
    env_id,
    seg_vis,
    seg_raw,
    base_frames,
    wrist_frames,
    fps=12,
    seg_hw=(360, 480),
    video_hw=(180, 240),
    simulated_inputs=None,
    external_state=None,
):
    """
    Updates state based on simulated_inputs and returns the annotated image.
    If external_state is None, starts with a fresh state.
    """
    if simulated_inputs is None:
        simulated_inputs = []

    if external_state is not None:
        state = external_state
        state["seg_raw"] = seg_raw
    else:
        state = {
            "buttons": [],
            "selection": None,
            "seg_raw": seg_raw,
            "seg_rect": None,
            "pending_option": None,
        }
    
    if seg_vis is not None:
        h, w = seg_vis.shape[:2]
        state["seg_rect"] = (0, 0, w - 1, h - 1)
    else:
        state["seg_rect"] = (0, 0, seg_hw[1] - 1, seg_hw[0] - 1)

    # --- Helper Inner Functions ---
    def _reset_selected_target():
        selected_target["obj"] = None
        selected_target["name"] = None
        selected_target["seg_id"] = None
        selected_target["click_point"] = None
        selected_target["centroid_point"] = None

    def _collect_available_targets(selected_idx=None):
        candidates = []
        seen = set()

        def _flatten_items(item):
            if item is None:
                return []
            if isinstance(item, dict):
                flattened = []
                for value in item.values():
                    flattened.extend(_flatten_items(value))
                return flattened
            if isinstance(item, (list, tuple, set)):
                flattened = []
                for value in item:
                    flattened.extend(_flatten_items(value))
                return flattened
            return [item]

        def _extend_from_iterable(iterable):
            for actor in iterable:
                if actor is None:
                    continue
                identifier = id(actor)
                if identifier not in seen:
                    seen.add(identifier)
                    candidates.append(actor)

        def _extend_from_option(idx):
            if idx is None or not (0 <= idx < len(options)):
                return
            avail = options[idx].get("available")
            if avail is not None:
                _extend_from_iterable(_flatten_items(avail))

        if selected_idx is not None:
            _extend_from_option(selected_idx)
            return candidates

        for idx in range(len(options)):
            _extend_from_option(idx)

        if not candidates:
            fallback = getattr(env, "spawned_cubes", None)
            if fallback:
                try:
                    _extend_from_iterable(list(fallback))
                except TypeError:
                    pass

        return candidates

    def _handle_logic_click(global_x, global_y):
        seg_rect = state.get("seg_rect")
        seg_raw_local = state.get("seg_raw")

        if seg_rect and seg_raw_local is not None:
            x1, y1, x2, y2 = seg_rect
            if x1 <= global_x <= x2 and y1 <= global_y <= y2:
                disp_w = max(1, x2 - x1 + 1)
                disp_h = max(1, y2 - y1 + 1)
                seg_h, seg_w = seg_raw_local.shape[:2]
                sx = int((global_x - x1) * seg_w / disp_w)
                sy = int((global_y - y1) * seg_h / disp_h)

                if 0 <= sx < seg_w and 0 <= sy < seg_h:
                    seg_id_map = getattr(env.unwrapped, "segmentation_id_map", {}) or {}
                    pending_option = state.get("pending_option")

                    if pending_option is None:
                        return

                    available_targets = _collect_available_targets(pending_option)
                    click_disp = (global_x - x1, global_y - y1)

                    candidates = []
                    if available_targets:
                        for actor in available_targets:
                            target_seg_id = None
                            for sid, obj in seg_id_map.items():
                                if obj is actor:
                                    target_seg_id = int(sid)
                                    break
                            if target_seg_id is None:
                                continue

                            mask = seg_raw_local == target_seg_id
                            if not np.any(mask):
                                continue

                            ys, xs = np.nonzero(mask)
                            cx = float(xs.mean())
                            cy = float(ys.mean())
                            disp_cx = int(round(cx * disp_w / seg_w))
                            disp_cy = int(round(cy * disp_h / seg_h))
                            dist2 = (disp_cx - click_disp[0]) ** 2 + (disp_cy - click_disp[1]) ** 2
                            candidates.append(
                                {
                                    "actor": actor,
                                    "seg_id": target_seg_id,
                                    "disp_point": (disp_cx, disp_cy),
                                    "dist2": dist2,
                                }
                            )

                    chosen = None
                    if candidates:
                        candidates.sort(key=lambda item: item["dist2"])
                        chosen = candidates[0]

                    if chosen is not None:
                        obj = chosen["actor"]
                        selected_target["obj"] = obj
                        selected_target["name"] = getattr(obj, "name", f"id_{chosen['seg_id']}")
                        selected_target["seg_id"] = chosen["seg_id"]
                        selected_target["click_point"] = (int(click_disp[0]), int(click_disp[1]))
                        selected_target["centroid_point"] = chosen["disp_point"]
                        print(f"AUTO: Target Selected -> {selected_target['name']}")
                    else:
                        seg_id = int(seg_raw_local[sy, sx])
                        obj = seg_id_map.get(seg_id)
                        if obj:
                            selected_target["obj"] = obj
                            selected_target["name"] = getattr(obj, "name", f"id_{seg_id}")
                            selected_target["seg_id"] = seg_id
                            selected_target["click_point"] = (int(click_disp[0]), int(click_disp[1]))
                            mask = seg_raw_local == seg_id
                            if np.any(mask):
                                ys, xs = np.nonzero(mask)
                                cx = int(round(xs.mean() * disp_w / seg_w))
                                cy = int(round(ys.mean() * disp_h / seg_h))
                                selected_target["centroid_point"] = (cx, cy)
                            print(f"AUTO: Target Selected via ID -> {selected_target['name']}")

    # --- Logic Loop ---
    for i, cmd in enumerate(simulated_inputs):
        if not cmd:
            continue
        
        cmd = str(cmd).strip()
        print(f"AUTO INPUT [{i}]: {cmd}")
        tokens = cmd.replace(",", " ").split()

        # CASE A: Option Selection (e.g. "1")
        if len(tokens) == 1 and tokens[0].isdigit():
            idx = int(tokens[0]) - 1
            if 0 <= idx < len(options):
                option_avail = _collect_available_targets(idx)
                if option_avail:
                    if state.get("pending_option") == idx:
                        # Trying to confirm
                        obj = selected_target.get("obj")
                        if obj is not None:
                            state["selection"] = idx
                        else:
                            print("AUTO: Cannot confirm, no target selected.")
                    else:
                        state["pending_option"] = idx
                        state["selection"] = None
                        _reset_selected_target()
                        print(f"AUTO: Option {idx + 1} armed.")
                else:
                    state["selection"] = idx
            else:
                print(f"AUTO: Invalid index {idx}")

        # CASE B: Coordinates (e.g. "150 150")
        elif len(tokens) >= 2:
            try:
                cx = int(float(tokens[0]))
                cy = int(float(tokens[1]))
                seg_rect = state.get("seg_rect")
                if seg_rect:
                    global_x = seg_rect[0] + cx
                    global_y = seg_rect[1] + cy
                    _handle_logic_click(global_x, global_y)
            except ValueError:
                print("AUTO: Coord parse error")

    # --- Render Annotated Image ---
    if seg_vis is not None:
        display_img = seg_vis.copy()
    else:
        h, w = seg_hw
        display_img = np.zeros((h, w, 3), dtype=np.uint8)

    if selected_target.get("click_point"):
        cv2.circle(display_img, selected_target["click_point"], 6, (0, 0, 255), 2)
    if selected_target.get("centroid_point"):
        cv2.circle(display_img, selected_target["centroid_point"], 6, (0, 255, 255), 2)

    return state, display_img


# =============================================================================
# Refactored Step Functions
# =============================================================================

def get_obs_and_segmentation(env, planner, env_id, color_map, use_segmentation=False, use_visualize=True):
    """
    Step A: Visualization Preparation & Initial Render.
    Returns visual data AND a list of available options with parameter requirements.

    Parameters
    ----------
    env : gym.Env
        已加载的 HistoryBench 环境实例。
    planner : PandaArmMotionPlanningSolver | PandaStickMotionPlanningSolver
        对应环境的规划器，实现特定动作求解。
    env_id : str
        当前评估的环境标识，用于构建选项。
    color_map : dict
        语义分割每个目标对应的 BGR 颜色映射。
    use_segmentation : bool
        是否展示语义分割图。
    use_visualize : bool
        是否打开 OpenCV 可视化窗口。

    Returns
    -------
    fig_a : str | None
        OpenCV 窗口名称（若已创建）。
    seg_vis : np.ndarray
        用于展示的 RGB 图像（可能是 segmentation 或相机画面）。
    seg_raw : np.ndarray | None
        原始分割 ID 图（后续点击逻辑使用）。
    base_frames : list
        来自 env.frames 的 Base 相机帧序列。
    wrist_frames : list
        来自 env.wrist_frames 的手腕帧序列。
    available_options : list of dict
        当前可解动作信息列表（{"action": label, "need_parameter": bool}）。
    """
    # 1. Visual Preparation
    base_frames = getattr(env, "frames", []) or []
    wrist_frames = getattr(env, "wrist_frames", []) or []
    seg_data = _fetch_segmentation(env)
    
    seg_hw = (360, 480)
    seg_vis = None
    seg_raw = None

    if use_segmentation:
        seg_vis, seg_raw = _prepare_segmentation_visual(seg_data, color_map, seg_hw)
    else:
        _, seg_raw = (
            _prepare_segmentation_visual(seg_data, color_map, seg_hw)
            if seg_data is not None
            else (None, None)
        )
        if base_frames:
            vis_frame = _prepare_frame(base_frames[-1])
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            if vis_frame.shape[:2] != seg_hw:
                vis_frame = cv2.resize(
                    vis_frame, (seg_hw[1], seg_hw[0]), interpolation=cv2.INTER_LINEAR
                )
            seg_vis = vis_frame
    
    # Safety fallback
    if seg_vis is None:
        seg_vis = np.zeros((seg_hw[0], seg_hw[1], 3), dtype=np.uint8)

    # 2. Build Options (To return structure)
    dummy_target = {"obj": None, "name": None, "seg_id": None, "click_point": None, "centroid_point": None}
    raw_options = _build_solve_options(env, planner, dummy_target, env_id)
    
    available_options = []
    for opt in raw_options:
        label = opt.get("label", "Unknown")
        # Logic: If 'available' list exists and is not empty, we need a parameter (coords) to select one.
        # If 'available' is None or empty, it's usually a global action or one that auto-selects.
        has_targets = opt.get("available")
        need_param = True if has_targets else False
        available_options.append({"action": label, "need_parameter": need_param})

    # 3. Render Step A (Static Plot)
    print("\n[Step A] Initial Render & Option Generation")
    
    _, img_a = _prompt_next_task_gui(
        [], # No options needed for initial render logic, just drawing frame
        env,
        dummy_target,
        env_id,
        seg_vis=seg_vis,
        seg_raw=seg_raw,
        base_frames=base_frames,
        wrist_frames=wrist_frames,
        simulated_inputs=[],
        external_state=None
    )

    fig_a = None
    if use_visualize:
        fig_a = _display_step_image("Step A: Initial State", img_a)
    
    return  fig_a, seg_vis, seg_raw, base_frames, wrist_frames, available_options


def get_input_from_gui(env, planner, env_id, seg_vis, mouse_select=True, use_visualize=True):
    """
    Step B: Input Collection.

    Parameters
    ----------
    env : gym.Env
        当前交互的环境实例。
    planner : PandaArmMotionPlanningSolver | PandaStickMotionPlanningSolver
        与环境绑定的运动规划器，供选项构造使用。
    env_id : str
        当前环境的标识，用于选项逻辑。
    seg_vis : np.ndarray
        Step A 输出的可视化帧，用于鼠标点击 feedback。
    mouse_select : bool
        是否启用鼠标点击交互。
    use_visualize : bool
        是否弹出 OpenCV 窗口，影响鼠标交互可用性。

    Returns
    -------
    command_dict : dict
        包含用户意图的命令，格式为 {"action": label, "point": (x, y) | None}。
    mouse_win_name : str | None
        如果创建了鼠标选择窗口，返回其名称以便 later cleanup。
    """
    # Create temporary options for display purposes
    temp_target = {}
    solve_options = _build_solve_options(env, planner, temp_target, env_id)

    _print_terminal_options(solve_options)
    
    # If visualization is disabled, we cannot use mouse selection.
    if not use_visualize:
        mouse_select = False
        
    mouse_win_name = "Select Target (Mouse)" if mouse_select else None
    
    # Use the passed seg_vis image directly
    reference_img = seg_vis

    print("\n[Input] Collecting Commands...")
    
    # 1. Option Selection
    cmd1 = input("  Command 1 (Option selection, e.g. '1' or 'Pick red cube'): ")
    
    selected_action_name = "Unknown"
    selected_idx = -1
    skip_coords = False
    
    # NOTE: Logic to identify index purely for GUI feedback is tricky with NLP here, 
    # so we pass the raw string and let Step C handle the heavy lifting (NLP matching).
    # However, if it IS an integer, we can check parameter needs now.
    
    if cmd1.strip().isdigit():
        try:
            # Parse input to get index
            idx = int(cmd1.strip()) - 1
            if 0 <= idx < len(solve_options):
                selected_idx = idx
                opt = solve_options[idx]
                selected_action_name = opt.get("label", "Unknown")
                
                # Check if the selected option has 'available' targets
                available_targets = opt.get("available")
                if not available_targets:
                    print(f"  -> Option {idx + 1} ({selected_action_name}) has no specific targets available.")
                    print("  -> Skipping coordinate selection.")
                    skip_coords = True
        except ValueError:
            pass
    else:
        # If text input, we simply assume it's an action name and pass it through.
        selected_action_name = cmd1.strip()

    # 2. Coordinates Selection
    coords = None

    if not skip_coords:
        if mouse_select and use_visualize:
            print("  Command 2: Please select a point on the popup OpenCV window...")
            click_state = {"coords": None}
            
            cv2.namedWindow(mouse_win_name)
            cv2.setMouseCallback(mouse_win_name, _on_mouse_click, click_state)

            while click_state["coords"] is None:
                cv2.imshow(mouse_win_name, reference_img) 
                key = cv2.waitKey(20)
                if key == 27:  # ESC
                    break

            if click_state["coords"]:
                coords = (click_state['coords'][0], click_state['coords'][1])
                print(f"  -> Captured Coordinates: {coords}")
            else:
                print("  -> No click detected. Defaulting to (0, 0)")
                coords = (0, 0)
        else:
            if not use_visualize:
                print("  (Visualization disabled: Mouse selection unavailable)")
            
            user_coords = input("  Command 2 (Coords, e.g. '150 150'): ")
            try:
                parts = user_coords.strip().split()
                if len(parts) >= 2:
                    coords = (int(parts[0]), int(parts[1]))
                else:
                    coords = (0, 0)
            except ValueError:
                coords = (0, 0)
            
            mouse_win_name = None 

    # Construct the final command dictionary
    command_dict = {
        "action": selected_action_name,
        "point": coords
    }

    return command_dict, mouse_win_name


def solve_and_evaluate_from_command(env, planner, env_id, seg_vis, seg_raw, base_frames, wrist_frames, command_dict, use_visualize=True):
    """
    Step C: Logic Processing & Execution.
    Takes dictionary command, translates to simulated inputs for internal logic, and executes.

    Parameters
    ----------
    env : gym.Env
        当前交互的环境实例。
    planner : PandaArmMotionPlanningSolver | PandaStickMotionPlanningSolver
        对应的运动规划器。
    env_id : str
        当前评估任务的标识符。
    seg_vis : np.ndarray
        分割可视化帧。
    seg_raw : np.ndarray | None
        原始语义分割 ID 图。
    base_frames : list
        Base 相机帧序列。
    wrist_frames : list
        手腕视角帧序列。
    command_dict : dict
        Step B 中构建的命令，包含 "action" 与可选 "point"。
    use_visualize : bool
        是否展示 Step C 的最终结果窗口。

    Returns
    -------
    evaluation : dict | None
        env.evaluate 的返回（若命令失效则为 None）。
    fig_c : str | None
        OpenCV 窗口名称（若可视化成功展示）。
    """
    
    # --- INTERNAL STATE INITIALIZATION ---
    selected_target = {
        "obj": None,
        "name": None,
        "seg_id": None,
        "click_point": None,
        "centroid_point": None,
    }
    
    # Build options bound to this specific selected_target instance
    solve_options = _build_solve_options(env, planner, selected_target, env_id)

    # --- DICT TO SIMULATED INPUT TRANSLATION ---
    target_action = command_dict.get("action")
    target_param = command_dict.get("point")

    # Validate command format
    if "action" not in command_dict:
        print("Error: Invalid command format. Expected format: {'action': 'ActionName', ...}")
        print(f"Received: {command_dict}")
        return None, None

    if target_action is None: # Allow integer or string
        print("Error: 'action' cannot be None.")
        return None, None

    # --- FIND OPTION INDEX (Strict OR NLP) ---
    found_idx = -1
    
    # 1. Try Strict Matching / Integer Parsing
    for i, opt in enumerate(solve_options):
        # Match label string OR match index (e.g. user typed "1")
        if opt.get("label") == target_action or str(i + 1) == str(target_action):
            found_idx = i
            break
            
    # 2. Try Semantic Matching (if strict failed and input is a string)
    if found_idx == -1 and isinstance(target_action, str) and not target_action.isdigit():
        print(f"Attempting semantic match for: '{target_action}'")
        found_idx, score = _find_best_semantic_match(target_action, solve_options)
            
    # --- CONSTRUCT INPUTS ---
    simulated_inputs = []
    
    if found_idx != -1:
        # 1. Select the option
        simulated_inputs.append(str(found_idx + 1))
        
        # 2. Provide coordinates if they exist
        if target_param is not None:
             simulated_inputs.append(f"{target_param[0]} {target_param[1]}")
        else:
             if solve_options[found_idx].get("available"):
                 simulated_inputs.append("0 0") # Fallback dummy if logic demands coord

        # 3. Confirm selection (repeat option index)
        simulated_inputs.append(str(found_idx + 1))
    else:
        print(f"Error: Action '{target_action}' not found in current options (Semantic match failed).")
        return None, None

    print(f"\n[Step C] Processing Dictionary: {command_dict}")
    print(f"       -> Simulated Sequence: {simulated_inputs}")

    # --- 1. Process Logic & Update State (Internally) ---
    final_internal_state, img_c = _prompt_next_task_gui(
        solve_options,
        env,
        selected_target,
        env_id,
        seg_vis=seg_vis,
        seg_raw=seg_raw,
        base_frames=base_frames,
        wrist_frames=wrist_frames,
        simulated_inputs=simulated_inputs,
        external_state=None
    )

    # --- 2. Visual Confirmation ---
    fig_c = None
    if use_visualize:
        fig_c = _display_step_image("Step C: Target Confirmation", img_c)

    # --- 3. Execution ---
    selection_idx = final_internal_state.get("selection")
    
    if selection_idx is not None:
        print(f"Execution. Selected Option: {selection_idx + 1}")
        solve_entry = solve_options[selection_idx]
        print(f"Solving: {solve_entry.get('label')}")
        solve_entry.get("solve")()
    else:
        print("Error: No selection was confirmed.")
        return None, fig_c # Return None evaluation to indicate execution failure

    # --- 4. Evaluation ---
    env.unwrapped.evaluate()
    evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
    
    print(evaluation)
    
    return evaluation, fig_c


# =============================================================================
# Main Loop
# =============================================================================

def main():
    # 主入口函数，负责按照固定配置加载环境、循环评估指定任务并展示 GUI 交互。
    num_episodes = 100
    gui_render = True
    max_steps_without_demonstration = 3000
    dataset_root = Path("/data/hongzefu/dataset_demonstration")
    use_segmentation = False
    mouse_select = True
    use_visualization = True 

    
    env_id_list =[
        # "PickXtimes",
        #  "StopCube",
        #"SwingXtimes",
        # "BinFill",

        #"VideoUnmaskSwap",
        #"VideoUnmask",
        #"ButtonUnmaskSwap",
        #"ButtonUnmask",

        #"VideoRepick",
        #"VideoPlaceButton",
        #"VideoPlaceOrder",
        #"PickHighlight",

        #"InsertPeg",
        'MoveCube',
        #"PatternLock",
        #"RouteStick"

        ]

    render_mode = "human" if gui_render else "rgb_array"

    for env_id in env_id_list:
        # 对每个环境 ID 在数据集中找到对应的演示数据文件
        dataset_path = dataset_root / f"record_dataset_{env_id}.h5"
        metadata_path = dataset_root / f"record_dataset_{env_id}_metadata.json"
        
        if not dataset_path.exists():
            print(f"[{env_id}] Dataset file {dataset_path} not found; skipping env.")
            continue

        with h5py.File(dataset_path, "r") as dataset:
            resolver = EpisodeConfigResolver(
                env_id=env_id,
                dataset=dataset,
                metadata_path=metadata_path if metadata_path.exists() else None,
                render_mode=render_mode,
                gui_render=gui_render,
                max_steps_without_demonstration=max_steps_without_demonstration,
            )

            for episode in range(num_episodes):
                # 当前只运行指定的 episode 号（此处硬编码为 15）用于调试
                if episode != 15:
                    continue

                env, episode_dataset, seed, difficulty = resolver.make_env_for_episode(episode)

                print(f"--- Running online evaluation for episode:{episode} ---")
                env.reset()

                language_goal = env.demonstration_data.get('language goal')

                # 生成语义分割颜色映射，并确保桌面区域使用黑色避免干扰
                color_map = _generate_color_map()
                _sync_table_color(env, color_map)

                if env_id in ("PatternLock", "RouteStick"):
                    planner = PandaStickMotionPlanningSolver(
                        env,
                        debug=False,
                        vis=gui_render,
                        base_pose=env.unwrapped.agent.robot.pose,
                        visualize_target_grasp_pose=False,
                        print_env_info=False,
                        joint_vel_limits=0.3,
                    )
                else:
                    planner = PandaArmMotionPlanningSolver(
                        env,
                        debug=False,
                        vis=gui_render,
                        base_pose=env.unwrapped.agent.robot.pose,
                        visualize_target_grasp_pose=False,
                        print_env_info=False,
                    )

                env.unwrapped.evaluate()
                tasks = list(getattr(env.unwrapped, "task_list", []) or [])
                
                # 先读取任务列表，若为空则直接关闭环境。实际选项仍在后续 Step A 中动态生成。
                if not tasks:
                    env.close()
                    continue

                figures_to_close = []
                try:
                    while True:
                        # [Step A] 获取当前帧视觉数据并构建可行动作列表
                        # 输入: env/planner/env_id 以及颜色映射color_map和可视化开关。
                        # 输出: 
                        #   - fig_a: 若启动可视化则返回 OpenCV 窗口名，用于后续清理；
                        #   - seg_vis: RGB 预览（分割或相机视角），供鼠标点击反馈；
                        #   - seg_raw: 索引式语义分割（用于点击后解析目标 ID）；
                        #   - base_frames/wrist_frames: 记录每步堆栈的相机和手腕帧；
                        #   - available_options: 在当前状态可选动作及其是否需要参数；
                        fig_a, seg_vis, seg_raw, base_frames, wrist_frames, available_options = get_obs_and_segmentation(
                            env,
                            planner,
                            env_id,
                            color_map,
                            use_segmentation=use_segmentation,
                            use_visualize=use_visualization
                        )
                        if fig_a:
                            figures_to_close.append(fig_a)

                        # Print the options list for verification
                        print(f"Available Actions: {available_options}")

                        #[Step B] Input Collection
                        # 输入:
                        #   - seg_vis: Step A 返回的可视化帧（用于鼠标点击定位目标）；
                        #   - 鼠标交互开关 + 是否启用窗口；
                        # 输出:
                        #   - command_dict: {"action": label, "point": (x, y) | None}，后续字典驱动动作执行；
                        # command_dict 包含 {"action": label, "point": (x, y) | None}，point 使用 OpenCV 坐标系：origin 在图像左上角，x 向右递增，y 向下递增。
                        #   - fig_b: 若启用点击交互则为 OpenCV 窗口名（便于清理）。
                        print(f"Language Goal: {language_goal}")
                        command_dict, fig_b = get_input_from_gui(
                             env,
                             planner,
                             env_id,
                             seg_vis,
                             mouse_select=mouse_select,
                             use_visualize=use_visualization
                        )
                        if fig_b:
                             figures_to_close.append(fig_b)

                        # [Step C] 执行逻辑处理并调用求解器
                        # 输入:
                        #   - env/planner/env_id: 主动参与求解、执行动作的核心对象；
                        #   - seg_vis/seg_raw: 分割与视觉帧，用于点击逻辑匹配；
                        #   - base_frames/wrist_frames: 可供内部逻辑回溯的帧缓存（当前实现未直接使用但保留接口）；
                        #   - command_dict: Step B 生成的意图指令；
                        # 输出:
                        #   - evaluation: env.evaluate() 的结果（包含 success/fail 等状态）；
                        #   - fig_c: 若启用展示则为 Step C 结果图窗口；
                        # command_dict 包含 {"action": label, "point": (x, y) | None}，point 使用 OpenCV 坐标系：origin 在图像左上角，x 向右递增，y 向下递增。
                        
                        evaluation, fig_c = solve_and_evaluate_from_command(
                            env,
                            planner,
                            env_id,
                            seg_vis,
                            seg_raw,
                            base_frames,
                            wrist_frames,
                            command_dict,
                            use_visualize=use_visualization
                        )
                        if fig_c:
                            figures_to_close.append(fig_c)

                        _cleanup_figures(figures_to_close, close_all=True)
                        figures_to_close = []

                        # Check if evaluation failed due to invalid command format
                        if evaluation is None:
                            print("Skipping evaluation due to invalid command format or failed NLP match.")
                            continue

                        fail_flag = evaluation.get("fail", False)
                        success_flag = evaluation.get("success", False)
                        if _tensor_to_bool(fail_flag):
                            print("Encountered failure condition; stopping task sequence.")
                            break

                        if _tensor_to_bool(success_flag):
                            print("Task completed successfully.")
                            break
                                
                finally:
                    _cleanup_figures(figures_to_close, close_all=True)

                env.close()

if __name__ == "__main__":
    main()
