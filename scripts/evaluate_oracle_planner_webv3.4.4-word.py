import os
import sys
import select
import json
import numpy as np
import sapien
import gymnasium as gym
import cv2
import colorsys
import h5py
from pathlib import Path
import torch
import gradio as gr
import tempfile
import shutil
import imageio


# --- NLP Imports ---
try:
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
_HARDCODED_DATASET_ROOT = Path("/data/hongzefu/historybench-v4.8.3/dataset_json")
DEFAULT_DATASET_ROOT = (
    _HARDCODED_DATASET_ROOT
    if _HARDCODED_DATASET_ROOT.exists()
    else Path(__file__).resolve().parents[1] / "dataset_json"
)
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

from history_bench_sim.chat_api.api import GeminiModel, QwenModel, OpenAIModel
from history_bench_sim.chat_api.prompts import *

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
    PandaStickMotionPlanningSolver,
)

TASK_WITH_DEMO = [
    "VideoUnmask", "VideoUnmaskSwap", "VideoPlaceButton", "VideoPlaceOrder",
    "VideoRepick", "MoveCube", "InsertPeg", "PatternLock", "RouteStick"
]

# =============================================================================
# Helper Functions (Retained)
# =============================================================================

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

        obs = env.unwrapped.get_obs(unflattened=True)
        return obs["sensor_data"]["base_camera"]["segmentation"]


def _build_solve_options(env, planner, selected_target, env_id):
    return get_vqa_options(env, planner, selected_target, env_id)

def _display_step_image(title, img_bgr, window_size=(6, 4)):
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

def _register_figure_for_cleanup(figures, window_name):
    """Track OpenCV windows so they can be closed later."""
    if figures is None:
        return
    if window_name:
        figures.append(window_name)

def _find_best_semantic_match(user_query, options):
    if _NLP_MODEL is None:
        return -1, 0.0
    
    if not options:
        return -1, 0.0

    labels = [opt.get("label", "") for opt in options]
    query_text = str(user_query or "").strip()

    try:
        query_embedding = _NLP_MODEL.encode(query_text, convert_to_tensor=True)
        corpus_embeddings = _NLP_MODEL.encode(labels, convert_to_tensor=True)
        cos_scores = st_util.cos_sim(query_embedding, corpus_embeddings)[0]
        best_idx = torch.argmax(cos_scores).item()
        best_score = cos_scores[best_idx].item()
    except Exception as exc:
        print(f"  [NLP] Semantic match failed ({exc}); defaulting to option 1.")
        return 0, 0.0

    print(f"  [NLP] Closest Match: '{query_text}' -> '{labels[best_idx]}' (Score: {best_score:.4f})")
    
    return best_idx, best_score

# --- Core GUI Logic Functions (Retained) ---
def _prompt_next_task_gui(options, env, selected_target, env_id, seg_vis, seg_raw, base_frames, wrist_frames, fps=12, seg_hw=(255, 255), video_hw=(255, 255), simulated_inputs=None, external_state=None):
    # ... [Same as provided code] ...
    # (Abbreviated to avoid duplication, assuming logic remains identical to original)
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
            if item is None: return []
            if isinstance(item, dict):
                flattened = []
                for value in item.values(): flattened.extend(_flatten_items(value))
                return flattened
            if isinstance(item, (list, tuple, set)):
                flattened = []
                for value in item: flattened.extend(_flatten_items(value))
                return flattened
            return [item]
        def _extend_from_iterable(iterable):
            for actor in iterable:
                if actor is None: continue
                identifier = id(actor)
                if identifier not in seen:
                    seen.add(identifier)
                    candidates.append(actor)
        def _extend_from_option(idx):
            if idx is None or not (0 <= idx < len(options)): return
            avail = options[idx].get("available")
            if avail is not None: _extend_from_iterable(_flatten_items(avail))
        if selected_idx is not None:
            _extend_from_option(selected_idx)
            return candidates
        for idx in range(len(options)): _extend_from_option(idx)
        if not candidates:
            fallback = getattr(env, "spawned_cubes", None)
            if fallback:
                try: _extend_from_iterable(list(fallback))
                except TypeError: pass
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
                    if pending_option is None: return
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
                            if target_seg_id is None: continue
                            mask = seg_raw_local == target_seg_id
                            if not np.any(mask): continue
                            ys, xs = np.nonzero(mask)
                            cx, cy = float(xs.mean()), float(ys.mean())
                            disp_cx, disp_cy = int(round(cx * disp_w / seg_w)), int(round(cy * disp_h / seg_h))
                            dist2 = (disp_cx - click_disp[0]) ** 2 + (disp_cy - click_disp[1]) ** 2
                            candidates.append({"actor": actor, "seg_id": target_seg_id, "disp_point": (disp_cx, disp_cy), "dist2": dist2})
                    chosen = None
                    if candidates:
                        candidates.sort(key=lambda item: item["dist2"])
                        chosen = candidates[0]
                    # 首选：依据点击点在分割图中的距离，锁定最近的候选 actor
                    if chosen is not None:
                        obj = chosen["actor"]
                        selected_target["obj"] = obj
                        selected_target["name"] = getattr(obj, "name", f"id_{chosen['seg_id']}")
                        selected_target["seg_id"] = chosen["seg_id"]
                        selected_target["click_point"] = (int(click_disp[0]), int(click_disp[1]))
                        selected_target["centroid_point"] = chosen["disp_point"]
                        print(f"AUTO: Target Selected -> {selected_target['name']}")
                    else:
                        # 兜底一：若距离比较失败，则直接读取点击像素的 seg_id 来确定对象
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
                                cx, cy = int(round(xs.mean() * disp_w / seg_w)), int(round(ys.mean() * disp_h / seg_h))
                                selected_target["centroid_point"] = (cx, cy)
                            print(f"AUTO: Target Selected via ID -> {selected_target['name']}")
                        elif available_targets:
                            # 兜底二：当前帧完全无法定位像素时，至少返回一个可操作 actor，避免 selection 为空
                            fallback_actor = next((actor for actor in available_targets if actor is not None), None)
                            if fallback_actor is not None:
                                selected_target["obj"] = fallback_actor
                                selected_target["name"] = getattr(fallback_actor, "name", "fallback")
                                selected_target["seg_id"] = None
                                click_pt = (int(click_disp[0]), int(click_disp[1]))
                                selected_target["click_point"] = click_pt
                                selected_target["centroid_point"] = click_pt
                                print(f"AUTO: Fallback target -> {selected_target['name']}")

    # --- Logic Loop ---
    for i, cmd in enumerate(simulated_inputs):
        if not cmd: continue
        cmd = str(cmd).strip()
        print(f"AUTO INPUT [{i}]: {cmd}")
        tokens = cmd.replace(",", " ").split()
        if len(tokens) == 1 and tokens[0].isdigit():
            idx = int(tokens[0]) - 1
            if 0 <= idx < len(options):
                option_avail = _collect_available_targets(idx)
                if option_avail:
                    if state.get("pending_option") == idx:
                        obj = selected_target.get("obj")
                        if obj is not None: state["selection"] = idx
                        else: print("AUTO: Cannot confirm, no target selected.")
                    else:
                        state["pending_option"] = idx
                        state["selection"] = None
                        _reset_selected_target()
                        print(f"AUTO: Option {idx + 1} armed.")
                else: state["selection"] = idx
            else: print(f"AUTO: Invalid index {idx}")
        elif len(tokens) >= 2:
            try:
                cx, cy = int(float(tokens[0])), int(float(tokens[1]))
                seg_rect = state.get("seg_rect")
                if seg_rect:
                    global_x, global_y = seg_rect[0] + cx, seg_rect[1] + cy
                    _handle_logic_click(global_x, global_y)
            except ValueError: print("AUTO: Coord parse error")

    # --- Render Annotated Image ---
    if seg_vis is not None: display_img = seg_vis.copy()
    else:
        h, w = seg_hw
        display_img = np.zeros((h, w, 3), dtype=np.uint8)
    if selected_target.get("click_point"): cv2.circle(display_img, selected_target["click_point"], 6, (0, 0, 255), 2)
    if selected_target.get("centroid_point"): cv2.circle(display_img, selected_target["centroid_point"], 6, (0, 255, 255), 2)
    return state, display_img

def step_before(env, planner, env_id, color_map, use_segmentation=False, use_visualize=False, figures_to_close=None):
    base_frames = getattr(env, "frames", [])
    if not base_frames:
        base_frames = getattr(env.unwrapped, "frames", []) or []
        
    wrist_frames = getattr(env, "wrist_frames", [])
    if not wrist_frames:
        wrist_frames = getattr(env.unwrapped, "wrist_frames", []) or []

    seg_data = _fetch_segmentation(env)
    
    # Detect resolution from env data instead of hardcoding
    #seg_hw = (360, 480)
    if base_frames and len(base_frames) > 0:
        seg_hw = base_frames[-1].shape[:2]
    elif seg_data is not None:
        try:
            temp = seg_data
            if hasattr(temp, "cpu"): temp = temp.cpu().numpy()
            temp = np.asarray(temp)
            if temp.ndim > 2: temp = temp[0]
            seg_hw = temp.shape[:2]
        except Exception:
            pass

    seg_vis = None
    seg_raw = None

    if use_segmentation:
        seg_vis, seg_raw = _prepare_segmentation_visual(seg_data, color_map, seg_hw)
    else:
        _, seg_raw = (_prepare_segmentation_visual(seg_data, color_map, seg_hw) if seg_data is not None else (None, None))
        if base_frames:
            vis_frame = _prepare_frame(base_frames[-1])
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            if vis_frame.shape[:2] != seg_hw:
                vis_frame = cv2.resize(vis_frame, (seg_hw[1], seg_hw[0]), interpolation=cv2.INTER_LINEAR)
            seg_vis = vis_frame
    
    if seg_vis is None: seg_vis = np.zeros((seg_hw[0], seg_hw[1], 3), dtype=np.uint8)

    dummy_target = {"obj": None, "name": None, "seg_id": None, "click_point": None, "centroid_point": None}
    raw_options = _build_solve_options(env, planner, dummy_target, env_id)
    available_options = [{"action": opt.get("label", "Unknown"), "need_parameter": bool(opt.get("available"))} for opt in raw_options]

    print("\n[Step A] Initial Render & Option Generation")
    _, img_a = _prompt_next_task_gui([], env, dummy_target, env_id, seg_vis=seg_vis, seg_raw=seg_raw, base_frames=base_frames, wrist_frames=wrist_frames, simulated_inputs=[], external_state=None)

    fig_a = _display_step_image("Step A: Initial State", img_a) if use_visualize else None
    _register_figure_for_cleanup(figures_to_close, fig_a)
    return seg_vis, seg_raw, base_frames, wrist_frames, available_options

def get_input_from_gui(env, planner, env_id, seg_vis, mouse_select=False, use_visualize=False, figures_to_close=None):
    temp_target = {}
    solve_options = _build_solve_options(env, planner, temp_target, env_id)
    _print_terminal_options(solve_options)
    
    if not use_visualize: mouse_select = False
    mouse_win_name = "Select Target (Mouse)" if mouse_select else None
    reference_img = seg_vis
    print("\n[Input] Collecting Commands...")
    
    cmd1 = input("  Command 1 (Option selection, e.g. '1' or 'Pick red cube'): ")
    selected_action_name, skip_coords = cmd1.strip(), False
    
    if cmd1.strip().isdigit():
        try:
            idx = int(cmd1.strip()) - 1
            if 0 <= idx < len(solve_options):
                selected_action_name = solve_options[idx].get("label", "Unknown")
                if not solve_options[idx].get("available"):
                    print(f"  -> Option {idx + 1} ({selected_action_name}) has no specific targets available.")
                    print("  -> Skipping coordinate selection.")
                    skip_coords = True
        except ValueError: pass

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
                if key == 27: break
            if click_state["coords"]:
                coords = (click_state['coords'][0], click_state['coords'][1])
                print(f"  -> Captured Coordinates: {coords}")
            else:
                print("  -> No click detected. Defaulting to (0, 0)")
                coords = (0, 0)
        else:
            if not use_visualize: print("  (Visualization disabled: Mouse selection unavailable)")
            user_coords = input("  Command 2 (Coords, e.g. '150 150'): ")
            try:
                parts = user_coords.strip().split()
                coords = (int(parts[0]), int(parts[1])) if len(parts) >= 2 else (0, 0)
            except ValueError: coords = (0, 0)
            mouse_win_name = None 

    command_dict = {"action": selected_action_name, "point": coords}
    _register_figure_for_cleanup(figures_to_close, mouse_win_name)
    return command_dict

def step_after(env, planner, env_id, seg_vis, seg_raw, base_frames, wrist_frames, command_dict, use_visualize=False, figures_to_close=None):
    selected_target = {"obj": None, "name": None, "seg_id": None, "click_point": None, "centroid_point": None}
    solve_options = _build_solve_options(env, planner, selected_target, env_id)
    target_action = command_dict.get("action")
    target_param = command_dict.get("point")

    if "action" not in command_dict: return None
    if target_action is None: return None

    found_idx = -1
    for i, opt in enumerate(solve_options):
        if opt.get("label") == target_action or str(i + 1) == str(target_action):
            found_idx = i
            break
            
    if found_idx == -1 and isinstance(target_action, str) and not target_action.isdigit():
        print(f"Attempting semantic match for: '{target_action}'")
        found_idx, score = _find_best_semantic_match(target_action, solve_options)
            
    simulated_inputs = []
    if found_idx != -1:
        simulated_inputs.append(str(found_idx + 1))
        if target_param is not None: simulated_inputs.append(f"{target_param[0]} {target_param[1]}")
        else:
             if solve_options[found_idx].get("available"): simulated_inputs.append("0 0")
        simulated_inputs.append(str(found_idx + 1))
    else:
        print(f"Error: Action '{target_action}' not found in current options.")
        return None

    print(f"\n[Step C] Processing Dictionary: {command_dict}")
    print(f"       -> Simulated Sequence: {simulated_inputs}")

    final_internal_state, img_c = _prompt_next_task_gui(solve_options, env, selected_target, env_id, seg_vis=seg_vis, seg_raw=seg_raw, base_frames=base_frames, wrist_frames=wrist_frames, simulated_inputs=simulated_inputs, external_state=None)
    fig_c = _display_step_image("Step C: Target Confirmation", img_c) if use_visualize else None
    _register_figure_for_cleanup(figures_to_close, fig_c)

    selection_idx = final_internal_state.get("selection")
    if selection_idx is not None:
        print(f"Execution. Selected Option: {selection_idx + 1}")
        solve_options[selection_idx].get("solve")()
    else:
        return None

    env.unwrapped.evaluate()
    evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
    print(evaluation)
    return evaluation

# =============================================================================
# NEW CLASS: EpisodeConfigResolverForOraclePlanner
# =============================================================================

class EpisodeConfigResolverForOraclePlanner:
    """
    Wraps the logic for initializing environments, datasets, planners, and
    visualization tools for the Oracle Planner workflow.
    """
    def __init__(self, dataset_root=DEFAULT_DATASET_ROOT, gui_render=True, max_steps_without_demonstration=3000):
        self.dataset_root = Path(dataset_root)
        self.gui_render = gui_render
        self.max_steps = max_steps_without_demonstration
        self.render_mode = "human" if gui_render else "rgb_array"
        
        # Internal state tracking
        self.current_dataset_file = None
        self.current_env_id = None

    def initialize_episode(self, env_id, episode_idx):
        """
        Loads the dataset, creates the environment, generates color maps,
        and initializes the appropriate planner for a specific episode.
        
        Returns:
            env, planner, color_map, language_goal, tasks
        """
        metadata_path = self.dataset_root / f"record_dataset_{env_id}_metadata.json"
        
        if not metadata_path.exists():
            print(f"[{env_id}] Dataset file {metadata_path} not found; skipping env.")
            return None, None, None, None

        # Manage H5 file lifecycle: Close previous if env_id changed or if it exists
        # (Simple approach: we re-open per call or check if same. 
        # Here we re-open to match original logic scope, but store handle to close later)
        self._cleanup_dataset() 
        self.current_env_id = env_id

        resolver = EpisodeConfigResolver(
            env_id=env_id,
            dataset=None,
            metadata_path=metadata_path if metadata_path.exists() else None,
            render_mode=self.render_mode,
            gui_render=self.gui_render,
            max_steps_without_demonstration=self.max_steps,
        )

        env, episode_dataset, seed, difficulty = resolver.make_env_for_episode(episode_idx)
        print(f"--- Running online evaluation for episode:{episode_idx} ---")
        env.reset()

        language_goal = env.demonstration_data.get('language goal')

        # Generate semantic segmentation color map
        color_map = _generate_color_map()
        _sync_table_color(env, color_map)

        # Initialize Planner
        if env_id in ("PatternLock", "RouteStick"):
            planner = PandaStickMotionPlanningSolver(
                env,
                debug=False,
                vis=self.gui_render,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                joint_vel_limits=0.3,
            )
        else:
            planner = PandaArmMotionPlanningSolver(
                env,
                debug=False,
                vis=self.gui_render,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
            )

        env.unwrapped.evaluate()
        tasks = list(getattr(env.unwrapped, "task_list", []) or [])

        return env, planner, color_map, language_goal

    def _cleanup_dataset(self):
        if self.current_dataset_file:
            try:
                self.current_dataset_file.close()
            except Exception:
                pass
            self.current_dataset_file = None

    def close(self):
        self._cleanup_dataset()

    def get_num_episodes(self, env_id):
        """
        Public helper to fetch how many episodes are recorded for an env.
        """
        return self._get_num_episodes_for_env(env_id)

    def _get_num_episodes_for_env(self, env_id):
        metadata_path = self.dataset_root / f"record_dataset_{env_id}_metadata.json"
        if not metadata_path.exists():
            print(f"[{env_id}] Metadata {metadata_path} not found; cannot determine episode count.")
            return 0

        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            print(f"[{env_id}] Failed to read metadata {metadata_path}: {exc}")
            return 0

        record_count = payload.get("record_count")
        if isinstance(record_count, int) and record_count >= 0:
            return record_count

        records = payload.get("records")
        if isinstance(records, list):
            return len(records)

        print(f"[{env_id}] Metadata does not declare episodes; defaulting to 0.")
        return 0


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

class OracleGradioService:
    def __init__(self):
        self.oracle_resolver = EpisodeConfigResolverForOraclePlanner(
            gui_render=False,  # Disable window popups, use rgb_array for streaming
            max_steps_without_demonstration=3000
        )
        self.env_id_list = [
            "PickXtimes",
            "StopCube",
            "SwingXtimes",
            "BinFill",
            "VideoUnmaskSwap",
            "VideoUnmask",
            "ButtonUnmaskSwap",
            "ButtonUnmask",
            "VideoRepick",
            "VideoPlaceButton",
            "VideoPlaceOrder",
            "PickHighlight",
            "InsertPeg",
            "MoveCube",
            "PatternLock",
            "RouteStick",
        ]
        self.current_env_id = self.env_id_list[0]
        self.current_episode = 42
        
        self.env = None
        self.planner = None
        self.color_map = None
        self.language_goal = ""
        
        self.seg_vis = None
        self.seg_raw = None
        self.base_frames = []
        self.wrist_frames = []
        self.available_options = []
        
        self.last_click = (0, 0)
        
        # API related
        self.api = None
        self.step_idx = 0
        self.frame_idx = 0
        self.save_dir = None
        
        # Track frame indices for video slicing between actions
        self.last_action_base_frame_idx = 0
        self.last_action_wrist_frame_idx = 0
        
    def _get_video_path(self, start_idx=None, end_idx=None):
        if not self.base_frames:
            return None
        
        frames = self.base_frames
        if len(frames) == 0:
            return None
        
        # Slice frames if indices are provided
        if start_idx is not None or end_idx is not None:
            start = start_idx if start_idx is not None else 0
            end = end_idx if end_idx is not None else len(frames)
            frames = frames[start:end]
            if len(frames) == 0:
                return None
            
        print(f"Starting video generation for {len(frames)} frames...")

        # Strategy 1: Try imageio (usually more robust)
        try:
            import imageio
            # Create a temp file
            tfile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            tfile.close()
            video_path = tfile.name
            
            print("Attempting to use imageio for video generation...")
            # imageio expects RGB frames, which matches base_frames format
            imageio.mimwrite(video_path, frames, fps=12, quality=8, macro_block_size=None)
            print(f"Video generated successfully using imageio: {video_path}")
            return video_path
        except ImportError:
            print("imageio module not found.")
        except Exception as e:
            print(f"imageio generation failed: {e}")

        # Strategy 2: OpenCV VideoWriter
        try:
            tfile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            tfile.close()
            video_path = tfile.name
            
            h, w = frames[0].shape[:2]
            print(f"Attempting to use OpenCV VideoWriter ({w}x{h})...")
            
            # Try mp4v first as it is more commonly available without extra h264 libs
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 12.0, (w, h))
            
            if not out.isOpened():
                print("Failed with mp4v, trying avc1...")
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(video_path, fourcc, 12.0, (w, h))
            
            if out.isOpened():
                print("VideoWriter opened. Writing frames...")
                for i, frame in enumerate(frames):
                    if i % 100 == 0:
                        print(f"  Writing frame {i}/{len(frames)}")
                    frame_u8 = np.asarray(frame).astype(np.uint8)
                    frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                out.release()
                print(f"Video generated successfully using OpenCV: {video_path}")
                return video_path
            else:
                print("Failed to open OpenCV VideoWriter with both mp4v and avc1.")
        except Exception as e:
            print(f"OpenCV generation failed: {e}")

        # Strategy 3: PIL GIF (Last resort)
        try:
            from PIL import Image as PILImage
            tfile = tempfile.NamedTemporaryFile(suffix='.gif', delete=False)
            tfile.close()
            video_path = tfile.name
            
            print("Attempting to generate GIF using PIL (fallback)...")
            pil_frames = [PILImage.fromarray(frame.astype(np.uint8)) for frame in frames]
            # Save as GIF
            pil_frames[0].save(
                video_path,
                save_all=True,
                append_images=pil_frames[1:],
                optimize=False,
                duration=int(1000/12), # ms per frame
                loop=0
            )
            print(f"GIF generated successfully: {video_path}")
            return video_path
        except Exception as e:
            print(f"PIL GIF generation failed: {e}")
            
        return None

    def _get_wrist_video_path(self, start_idx=None, end_idx=None):
        """生成 wrist camera 的视频路径"""
        if not self.wrist_frames:
            return None
        
        frames = self.wrist_frames
        if len(frames) == 0:
            return None
        
        # Slice frames if indices are provided
        if start_idx is not None or end_idx is not None:
            start = start_idx if start_idx is not None else 0
            end = end_idx if end_idx is not None else len(frames)
            frames = frames[start:end]
            if len(frames) == 0:
                return None
            
        print(f"Starting wrist video generation for {len(frames)} frames...")

        # Strategy 1: Try imageio (usually more robust)
        try:
            import imageio
            # Create a temp file
            tfile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            tfile.close()
            video_path = tfile.name
            
            print("Attempting to use imageio for wrist video generation...")
            # imageio expects RGB frames, which matches wrist_frames format
            imageio.mimwrite(video_path, frames, fps=12, quality=8, macro_block_size=None)
            print(f"Wrist video generated successfully using imageio: {video_path}")
            return video_path
        except ImportError:
            print("imageio module not found.")
        except Exception as e:
            print(f"imageio wrist video generation failed: {e}")

        # Strategy 2: OpenCV VideoWriter
        try:
            tfile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            tfile.close()
            video_path = tfile.name
            
            h, w = frames[0].shape[:2]
            print(f"Attempting to use OpenCV VideoWriter for wrist video ({w}x{h})...")
            
            # Try mp4v first as it is more commonly available without extra h264 libs
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 12.0, (w, h))
            
            if not out.isOpened():
                print("Failed with mp4v, trying avc1...")
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(video_path, fourcc, 12.0, (w, h))
            
            if out.isOpened():
                print("VideoWriter opened. Writing wrist frames...")
                for i, frame in enumerate(frames):
                    if i % 100 == 0:
                        print(f"  Writing wrist frame {i}/{len(frames)}")
                    frame_u8 = np.asarray(frame).astype(np.uint8)
                    frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                out.release()
                print(f"Wrist video generated successfully using OpenCV: {video_path}")
                return video_path
            else:
                print("Failed to open OpenCV VideoWriter with both mp4v and avc1.")
        except Exception as e:
            print(f"OpenCV wrist video generation failed: {e}")

        # Strategy 3: PIL GIF (Last resort)
        try:
            from PIL import Image as PILImage
            tfile = tempfile.NamedTemporaryFile(suffix='.gif', delete=False)
            tfile.close()
            video_path = tfile.name
            
            print("Attempting to generate wrist GIF using PIL (fallback)...")
            pil_frames = [PILImage.fromarray(frame.astype(np.uint8)) for frame in frames]
            # Save as GIF
            pil_frames[0].save(
                video_path,
                save_all=True,
                append_images=pil_frames[1:],
                optimize=False,
                duration=int(1000/12), # ms per frame
                loop=0
            )
            print(f"Wrist GIF generated successfully: {video_path}")
            return video_path
        except Exception as e:
            print(f"PIL wrist GIF generation failed: {e}")
            
        return None

    def _get_base_video_update(self, start_idx=None, end_idx=None):
        """获取 base camera 视频的更新对象（带自动播放）"""
        video_path = self._get_video_path(start_idx, end_idx)
        if video_path:
            return gr.update(value=video_path, autoplay=True)
        return gr.update(value=None)

    def _get_wrist_video_update(self, start_idx=None, end_idx=None):
        """获取 wrist camera 视频的更新对象（带自动播放）"""
        video_path = self._get_wrist_video_path(start_idx, end_idx)
        if video_path:
            return gr.update(value=video_path, autoplay=True)
        return gr.update(value=None)
    
    def _get_synced_video_updates(self, base_start_idx=None, base_end_idx=None, wrist_start_idx=None, wrist_end_idx=None):
        """同时获取两个视频的更新对象，确保同步播放"""
        base_path = self._get_video_path(base_start_idx, base_end_idx)
        wrist_path = self._get_wrist_video_path(wrist_start_idx, wrist_end_idx)
        
        base_update = gr.update(value=base_path, autoplay=True) if base_path else gr.update(value=None)
        wrist_update = gr.update(value=wrist_path, autoplay=True) if wrist_path else gr.update(value=None)
        
        return base_update, wrist_update

    def _get_viewer_snapshot(self):
        """
        Rewritten env.render() wrapper to support dynamic streaming.
        Returns the current environment frame (rgb_array).
        """
        if not self.env:
            return None
            
        try:
            # STRICTLY use env.render() only, as requested.
            img = self.env.render()
            
            if img is None:
                return None

            if hasattr(img, "cpu"):
                img = img.cpu().numpy()
            img = np.asarray(img)
            
            # Debug: Print shape if it's unexpected (only once to avoid spam)
            # print(f"[Viewer] Raw shape: {img.shape}")
            
            # Squeeze to remove batch dimensions, e.g. (1, 1, 512, 512, 3) -> (512, 512, 3)
            img = img.squeeze()
            
            # Handle PyTorch convention (C, H, W) -> (H, W, C)
            # If shape is (3, H, W), transpose.
            if img.ndim == 3 and img.shape[0] == 3 and img.shape[1] > 3 and img.shape[2] > 3:
                img = np.transpose(img, (1, 2, 0))
            
            # Normalize if float [0, 1]
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            # Handle RGBA -> RGB
            if img.ndim == 3 and img.shape[-1] == 4:
                img = img[..., :3]
            
            # Ensure HxWxC
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
                
            return img
        except Exception as e:
            print(f"[Viewer] Render failed: {e}")
            return None

    def _get_black_image(self):
        """
        返回纯黑色图片（用于替换原有的snapshot机制）
        """
        # 返回一个256x256的纯黑色RGB图片
        return np.zeros((256, 256, 3), dtype=np.uint8)

    def _get_streaming_views(self):
        # 流式更新只刷新 viewer snapshot，不刷新视频
        # 视频只在执行 action 后刷新（在 run_step 中返回）
        return gr.skip(), gr.skip()

    def load_env(self, env_id, episode_idx):
        try:
            episode_idx = int(episode_idx)
            self.current_env_id = env_id
            self.current_episode = episode_idx
            
            if self.env:
                try:
                    self.env.close()
                except:
                    pass
            
            self.env, self.planner, self.color_map, self.language_goal = self.oracle_resolver.initialize_episode(env_id, episode_idx)
            
            if self.env is None:
                return (
                    f"Failed to load Env: {env_id}, Episode: {episode_idx}",
                    None,
                    None,
                    gr.update(value=None),
                    gr.update(value=None),
                    gr.Radio(choices=[], value=None),
                    "",
                )
            
            # Initialize API
            model_name = "gemini-2.5-flash"
            self.save_dir = f"oracle_planning_gui/{model_name}/{env_id}/ep{episode_idx}"
            if os.path.exists(self.save_dir):
                shutil.rmtree(self.save_dir)
            os.makedirs(self.save_dir, exist_ok=True)

            with open(os.path.join(self.save_dir, "language_goal.txt"), "w") as f:
                f.write(self.language_goal)
            
            if "gemini" in model_name:
                self.api = GeminiModel(save_dir=self.save_dir, task_id=env_id, model_name=model_name, task_goal=self.language_goal, subgoal_type="oracle_planner")
            elif "qwen" in model_name:
                self.api = QwenModel(save_dir=self.save_dir, task_id=env_id, model_name=model_name, task_goal=self.language_goal, subgoal_type="oracle_planner")
            else:
                self.api = OpenAIModel(save_dir=self.save_dir, task_id=env_id, model_name=model_name, task_goal=self.language_goal, subgoal_type="oracle_planner")
            
            self.step_idx = 0
            self.frame_idx = 0
            self.last_action_base_frame_idx = 0
            self.last_action_wrist_frame_idx = 0
                
            # Perform initial step
            self._update_state()
            
            # Set the starting indices to the current frame length (episode history length)
            # This ensures that the first action's video starts after the episode history
            self.last_action_base_frame_idx = len(self.base_frames)
            self.last_action_wrist_frame_idx = len(self.wrist_frames)
            
            # base_update, wrist_update = self._get_synced_video_updates()
            return (
                f"Loaded {env_id} Episode {episode_idx}",
                self._get_display_image(),
                self._get_video_path(),
                gr.update(value=None),  # Base camera video hidden
                gr.update(value=None),  # Wrist camera video hidden
                gr.Radio(choices=self._get_options_labels(), value=None),
                self.language_goal,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return (
                f"Error loading: {str(e)}",
                None,
                None,
                gr.update(value=None),
                gr.update(value=None),
                gr.Radio(choices=[], value=None),
                "",
            )

    def _update_state(self):
        self.seg_vis, self.seg_raw, self.base_frames, self.wrist_frames, self.available_options = step_before(
            self.env,
            self.planner,
            self.current_env_id,
            self.color_map,
            use_segmentation=False,
            use_visualize=False,
            figures_to_close=[]
        )

    def _get_display_image(self):
        if self.seg_vis is not None:
            # CV2 is BGR, Gradio expects RGB
            display_img = cv2.cvtColor(self.seg_vis.copy(), cv2.COLOR_BGR2RGB)
            # 在图像上绘制点击位置
            if self.last_click != (0, 0):
                x, y = self.last_click
                # 绘制一个红色圆圈标记点击位置
                cv2.circle(display_img, (x, y), 8, (255, 0, 0), 2)
                # 绘制一个十字标记
                cv2.line(display_img, (x - 10, y), (x + 10, y), (255, 0, 0), 2)
                cv2.line(display_img, (x, y - 10), (x, y + 10), (255, 0, 0), 2)
            return display_img
        return np.zeros((256, 256, 3), dtype=np.uint8)

    def _get_options_labels(self):
        # Returns list of strings like "1. Pick red cube"
        return [f"{i+1}. {opt['action']}" for i, opt in enumerate(self.available_options)]

    def set_click(self, evt: gr.SelectData):
        self.last_click = (evt.index[0], evt.index[1])
        return (
            self._get_display_image(),
            f"Selected Coords: {self.last_click}"
        )

    def call_gemini_prediction(self):
        if not self.env or not self.api:
            return (
                "Environment or API not loaded.",
                self._get_display_image(),
                gr.Radio(choices=[], value=None),
                "(0, 0)"
            )
        
        # Construct Query
        if self.step_idx == 0:
            if self.current_env_id in TASK_WITH_DEMO:
                if self.api.use_multi_images_as_video:
                    text_query = DEMO_TEXT_QUERY_multi_image.format(task_goal=self.language_goal)
                else:
                    text_query = DEMO_TEXT_QUERY.format(task_goal=self.language_goal)
            else:
                text_query = IMAGE_TEXT_QUERY.format(task_goal=self.language_goal)
        else:
            if self.api.use_multi_images_as_video:
                text_query = VIDEO_TEXT_QUERY_multi_image.format(task_goal=self.language_goal)
            else:
                text_query = VIDEO_TEXT_QUERY.format(task_goal=self.language_goal)
        
        input_data = self.api.prepare_input_data(self.base_frames[self.frame_idx:], text_query, self.step_idx)
        
        # Check if we have frames
        frames_slice = self.base_frames[self.frame_idx:]
        if len(frames_slice) == 0:
             print(f"Warning: No new frames found since frame_idx {self.frame_idx}. Total frames: {len(self.base_frames)}")
             # Fallback: use last frame or fail gracefully
             if len(self.base_frames) > 0:
                 # Just use the last frame to avoid crash, but this implies logic error elsewhere
                 frames_slice = [self.base_frames[-1]]
             else:
                 return (
                    "Error: No frames available.",
                    self._get_display_image(),
                    gr.Radio(choices=self._get_options_labels(), value=None),
                    "(0, 0)"
                )
        
        input_data = self.api.prepare_input_data(frames_slice, text_query, self.step_idx)
        
        print(f"Calling Gemini... Step: {self.step_idx}, Frame Start: {self.frame_idx}")
        try:
            response, points = self.api.call(input_data)
        except Exception as e:
             import traceback
             traceback.print_exc()
             return (
                f"API Call Failed: {e}",
                self._get_display_image(),
                gr.Radio(choices=self._get_options_labels(), value=None),
                "(0, 0)"
            )

        if response is None:
             return (
                "API returned None.",
                self._get_display_image(),
                gr.Radio(choices=self._get_options_labels(), value=None),
                "(0, 0)"
            )
            
        # Draw points for debug
        if points and len(points) > 0:
            # Update last click to the first point so it shows up in red
            # points are (row, col) -> (y, x).
            pt = points[0]
            # self.last_click = (int(pt[1]), int(pt[0])) # Optional: update click to first point from 'points' list?
            # actually response['subgoal']['point'] is the main one.
            
            # Save debug image
            anno_image = self.base_frames[-1].copy()
            for point in points:
                cv2.circle(anno_image, (point[1], point[0]), 5, (255, 255, 0), -1)
            imageio.imwrite(os.path.join(self.save_dir, f"anno_step_{self.step_idx}_image.png"), anno_image)
            self.api.add_frame_hold(anno_image)

        command_dict = response.get('subgoal', {})
        
        predicted_action = command_dict.get('action')
        predicted_point = command_dict.get('point')
        
        if predicted_point:
             # Apply the same flip as in oraclev3.py (y, x) -> (x, y)
             predicted_point = predicted_point[::-1]
             self.last_click = (int(predicted_point[0]), int(predicted_point[1]))

        # Find matching option index
        options = self._get_options_labels()
        selected_option = None
        
        # Helper to strip numbering "1. "
        def clean_opt(o):
            return o.split('. ', 1)[1] if '. ' in o else o

        if predicted_action:
            for opt in options:
                if clean_opt(opt) == predicted_action:
                    selected_option = opt
                    break
            
            if not selected_option:
                # Try loose match
                for opt in options:
                    if predicted_action.lower() in opt.lower():
                        selected_option = opt
                        break
        
        log_msg = f"Gemini Prediction:\nAction: {predicted_action}\nPoint: {predicted_point}\nResponse: {response}"
        
        return (
            log_msg,
            self._get_display_image(),
            gr.Radio(choices=options, value=selected_option),
            f"{self.last_click}"
        )

    def run_step(self, option_str):
        if not self.env:
            return (
                "Environment not loaded.",
                self._get_display_image(),
                None,
                gr.update(value=None),
                gr.update(value=None),
                gr.Radio(choices=[], value=None),
                "",
            )
        
        if not option_str:
            # base_update, wrist_update = self._get_synced_video_updates()
            return (
                "Please select an action.",
                self._get_display_image(),
                gr.skip(),
                gr.skip(),  # Base camera video
                gr.skip(),  # Wrist camera video
                gr.Radio(choices=self._get_options_labels(), value=None),
                self.language_goal,
            )

        # Parse option index from string "1. Action Name"
        try:
            idx = int(option_str.split('.')[0]) - 1
        except:
            # base_update, wrist_update = self._get_synced_video_updates()
            return (
                "Invalid option format.",
                self._get_display_image(),
                gr.skip(),
                gr.skip(),  # Base camera video
                gr.skip(),  # Wrist camera video
                gr.Radio(choices=self._get_options_labels(), value=None),
                self.language_goal,
            )

        if not (0 <= idx < len(self.available_options)):
             # base_update, wrist_update = self._get_synced_video_updates()
             return (
                 "Option index out of range.",
                 self._get_display_image(),
                 gr.skip(),
                 gr.skip(),  # Base camera video
                 gr.skip(),  # Wrist camera video
                 gr.Radio(choices=self._get_options_labels(), value=None),
                 self.language_goal,
             )

        selected_opt = self.available_options[idx]
        action_name = selected_opt['action']
        
        # Prepare command
        command_dict = {
            "action": action_name,
            "point": self.last_click
        }
        
        # Save current frame count before action execution to correctly set frame_idx later
        # Note: If self.base_frames is a reference to env.frames, it might grow during step_after.
        # But at this exact line, step_after hasn't run yet.
        pre_action_len = len(self.base_frames)
        pre_action_wrist_len = len(self.wrist_frames)
        
        # Save the starting frame indices for this action (where the video should start)
        action_start_base_idx = self.last_action_base_frame_idx
        action_start_wrist_idx = self.last_action_wrist_frame_idx
        
        log_msg = f"Executing: {command_dict}\n"
        
        # Run Step After
        try:
            evaluation = step_after(
                self.env,
                self.planner,
                self.current_env_id,
                self.seg_vis,
                self.seg_raw,
                self.base_frames,
                self.wrist_frames,
                command_dict,
                use_visualize=False,
                figures_to_close=[]
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            # After action execution, update the last action frame indices
            self.last_action_base_frame_idx = len(self.base_frames)
            self.last_action_wrist_frame_idx = len(self.wrist_frames)
            base_update, wrist_update = self._get_synced_video_updates(
                base_start_idx=action_start_base_idx,
                base_end_idx=self.last_action_base_frame_idx,
                wrist_start_idx=action_start_wrist_idx,
                wrist_end_idx=self.last_action_wrist_frame_idx
            )
            return (
                f"Error during execution: {e}",
                self._get_display_image(),
                gr.skip(),
                base_update,  # Base camera video with autoplay
                wrist_update,  # Wrist camera video with autoplay
                gr.Radio(choices=self._get_options_labels(), value=None),
                self.language_goal,
            )

        status = "Continue"
        if evaluation:
            if _tensor_to_bool(evaluation.get("success", False)):
                status = "SUCCESS"
            elif _tensor_to_bool(evaluation.get("fail", False)):
                status = "FAILURE"
        
        log_msg += f"Result: {status}\n"
        
        # Update the last action frame indices after action execution
        # These will be the starting point for the next action's video
        self.last_action_base_frame_idx = len(self.base_frames)
        self.last_action_wrist_frame_idx = len(self.wrist_frames)

        if status in ["SUCCESS", "FAILURE"]:
            base_update, wrist_update = self._get_synced_video_updates(
                base_start_idx=action_start_base_idx,
                base_end_idx=self.last_action_base_frame_idx,
                wrist_start_idx=action_start_wrist_idx,
                wrist_end_idx=self.last_action_wrist_frame_idx
            )
            return (
                log_msg + " (End of Episode)",
                self._get_display_image(),
                gr.skip(),
                base_update,  # Base camera video with autoplay
                wrist_update,  # Wrist camera video with autoplay
                gr.Radio(choices=[], value=None),
                self.language_goal,
            )
        
        # Next Step
        self._update_state()
        
        # CRITICAL FIX: frame_idx should be the index where the NEW frames start.
        # The new frames were generated during step_after.
        # So they start at the OLD length.
        # We saved it in `pre_action_len`.
        self.frame_idx = pre_action_len
        self.step_idx += 1
        base_update, wrist_update = self._get_synced_video_updates(
            base_start_idx=action_start_base_idx,
            base_end_idx=self.last_action_base_frame_idx,
            wrist_start_idx=action_start_wrist_idx,
            wrist_end_idx=self.last_action_wrist_frame_idx
        )
        return (
            log_msg,
            self._get_display_image(),
            gr.skip(),
            base_update,  # Base camera video with autoplay
            wrist_update,  # Wrist camera video with autoplay
            gr.Radio(choices=self._get_options_labels(), value=None),
            self.language_goal,
        )


def main():
    service = OracleGradioService()
    
    with gr.Blocks(title="Oracle Planner Web Interface") as demo:
        gr.Markdown("# 🚀 Human Planning Demo! 🚀")
        
        with gr.Row():
            with gr.Column(scale=1):
                env_dd = gr.Dropdown(choices=service.env_id_list, value=service.current_env_id, label="Environment ID")
                ep_num = gr.Number(value=42, label="Episode Index", precision=0)
                load_btn = gr.Button("Load/Reset Episode 💾")
                
                goal_box = gr.Textbox(label="Instruction/Task Goal", interactive=False, lines=3)
                
                options_radio = gr.Radio(label="Action Selection", choices=[])
                coords_box = gr.Textbox(label="Target Coordinates (Click to Select)", value="(0, 0)", interactive=False)
            
                with gr.Row():
                    gemini_btn = gr.Button("Ask Gemini 🤔🙌", variant="secondary")
                    exec_btn = gr.Button("Execute Action! 🦾🔥", variant="primary")
                
                log_output = gr.Textbox(label="Status/Logs", lines=5)
                
            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Keypoint Selector (Click to Select) 🎯")
                        img_display = gr.Image(label="", interactive=True)
                    with gr.Column():
                        gr.Markdown("## Video Demonstration (Single Playback Only) 🎥")
                        # 将竖向高度压缩到原先的一半左右，避免占用过高空间
                        video_display = gr.Video(label="", height=256)
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Execution Feed: Desk View 📷")
                        base_display = gr.Video(label="", height=256)
                    with gr.Column():
                        gr.Markdown("## Execution Feed: Robot View 🤖")
                        wrist_display = gr.Video(label="", height=256)
        
        # Event Wiring
        load_btn.click(
            fn=service.load_env,
            inputs=[env_dd, ep_num],
           outputs=[log_output, img_display, video_display, base_display, wrist_display, options_radio, goal_box]
            # outputs=[log_output, img_display, video_display, base_display, wrist_display, options_radio, goal_box]
        )
        
        img_display.select(
            fn=service.set_click,
            inputs=None,
            outputs=[img_display, coords_box]
        )
        
        gemini_btn.click(
            fn=service.call_gemini_prediction,
            inputs=None,
            outputs=[log_output, img_display, options_radio, coords_box]
        )
        
        exec_btn.click(
            fn=service.run_step,
            inputs=[options_radio],
            outputs=[log_output, img_display, video_display, base_display, wrist_display, options_radio, goal_box]
        )
        
        # 动态streaming: 使用 gr.Timer 替代 deprecated 的 every 参数
        timer = gr.Timer(value=0.2)
        timer.tick(
            fn=service._get_streaming_views, 
            inputs=None, 
            outputs=[base_display, wrist_display]
        )
        
        # 添加 JavaScript 来同步播放两个视频，确保同时开始播放
        sync_script_html = gr.HTML(
            value="""<script>
        (function() {
            let baseVideoEl = null;
            let wristVideoEl = null;
            let isSyncing = false;
            let baseReady = false;
            let wristReady = false;
            let bothReadyCallback = null;
            
            function findVideos() {
                const containers = document.querySelectorAll('.gradio-video, [class*="video"]');
                baseVideoEl = null;
                wristVideoEl = null;
                
                containers.forEach(container => {
                    const label = container.querySelector('label');
                    const video = container.querySelector('video');
                    if (label && video) {
                        const labelText = (label.textContent || label.innerText || '').toLowerCase();
                        if (labelText.includes('base camera') || labelText.includes('base')) {
                            baseVideoEl = video;
                        } else if (labelText.includes('wrist camera') || labelText.includes('wrist')) {
                            wristVideoEl = video;
                        }
                    }
                });
                
                return baseVideoEl && wristVideoEl;
            }
            
            function checkBothReady() {
                if (baseVideoEl && wristVideoEl && baseReady && wristReady) {
                    // 两个视频都准备好了，同时开始播放
                    if (bothReadyCallback) {
                        clearTimeout(bothReadyCallback);
                        bothReadyCallback = null;
                    }
                    
                    // 重置状态
                    baseReady = false;
                    wristReady = false;
                    
                    // 确保两个视频都从0开始
                    baseVideoEl.currentTime = 0;
                    wristVideoEl.currentTime = 0;
                    
                    // 同时播放
                    Promise.all([
                        baseVideoEl.play().catch(() => {}),
                        wristVideoEl.play().catch(() => {})
                    ]).then(() => {
                        // 播放后同步时间
                        if (Math.abs(baseVideoEl.currentTime - wristVideoEl.currentTime) > 0.1) {
                            wristVideoEl.currentTime = baseVideoEl.currentTime;
                        }
                    });
                }
            }
            
            function setupSync() {
                if (!baseVideoEl || !wristVideoEl) return;
                
                // 重置状态
                baseReady = false;
                wristReady = false;
                
                // 移除旧的事件监听器（通过克隆元素）
                const baseClone = baseVideoEl.cloneNode(true);
                const wristClone = wristVideoEl.cloneNode(true);
                baseVideoEl.parentNode.replaceChild(baseClone, baseVideoEl);
                wristVideoEl.parentNode.replaceChild(wristClone, wristVideoEl);
                baseVideoEl = baseClone;
                wristVideoEl = wristClone;
                
                // 监听视频加载完成事件
                function onBaseCanPlay() {
                    baseReady = true;
                    checkBothReady();
                }
                
                function onWristCanPlay() {
                    wristReady = true;
                    checkBothReady();
                }
                
                baseVideoEl.addEventListener('canplay', onBaseCanPlay);
                wristVideoEl.addEventListener('canplay', onWristCanPlay);
                
                // 如果视频已经加载完成，立即标记为ready
                if (baseVideoEl.readyState >= 2) {
                    baseReady = true;
                    checkBothReady();
                }
                if (wristVideoEl.readyState >= 2) {
                    wristReady = true;
                    checkBothReady();
                }
                
                // 同步播放/暂停
                function syncPlay(sourceVideo, targetVideo) {
                    if (isSyncing) return;
                    isSyncing = true;
                    
                    if (targetVideo.readyState >= 2) {
                        targetVideo.currentTime = sourceVideo.currentTime;
                        if (!sourceVideo.paused && targetVideo.paused) {
                            targetVideo.play().catch(() => {});
                        } else if (sourceVideo.paused && !targetVideo.paused) {
                            targetVideo.pause();
                        }
                    }
                    
                    setTimeout(() => { isSyncing = false; }, 50);
                }
                
                // 同步时间
                function syncTime(sourceVideo, targetVideo) {
                    if (isSyncing || targetVideo.paused) return;
                    if (Math.abs(sourceVideo.currentTime - targetVideo.currentTime) > 0.15) {
                        isSyncing = true;
                        targetVideo.currentTime = sourceVideo.currentTime;
                        setTimeout(() => { isSyncing = false; }, 50);
                    }
                }
                
                // 监听播放/暂停事件
                baseVideoEl.addEventListener('play', () => syncPlay(baseVideoEl, wristVideoEl));
                baseVideoEl.addEventListener('pause', () => {
                    if (!wristVideoEl.paused) wristVideoEl.pause();
                });
                baseVideoEl.addEventListener('timeupdate', () => syncTime(baseVideoEl, wristVideoEl));
                
                wristVideoEl.addEventListener('play', () => syncPlay(wristVideoEl, baseVideoEl));
                wristVideoEl.addEventListener('pause', () => {
                    if (!baseVideoEl.paused) baseVideoEl.pause();
                });
                wristVideoEl.addEventListener('timeupdate', () => syncTime(wristVideoEl, baseVideoEl));
            }
            
            function initSync() {
                if (findVideos()) {
                    setupSync();
                }
            }
            
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', initSync);
            } else {
                initSync();
            }
            
            const observer = new MutationObserver(() => {
                setTimeout(initSync, 100);
            });
            observer.observe(document.body, { childList: true, subtree: true });
        })();
        </script>""",
            visible=False
        )

    print("Starting Gradio Server...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

if __name__ == "__main__":
    main()
