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

# --- NLP Imports ---
# NLP 语义匹配（基于字符的 Edit Distance）
from rapidfuzz import process, fuzz
print("Loading NLP Module (rapidfuzz Edit Distance)...")

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

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
    PandaStickMotionPlanningSolver,
)

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
    """使用基于字符的编辑距离（rapidfuzz）找到最佳选项"""
    if not options:
        return -1, 0.0

    labels = [opt.get("label", "") for opt in options]
    query_text = str(user_query or "").strip()

    try:
        # 使用 rapidfuzz 提取最佳匹配
        # process.extractOne 返回 (match, score, index)
        # score 范围 0-100
        result = process.extractOne(query_text, labels, scorer=fuzz.ratio)
        
        if result:
            match_text, score, best_idx = result
            best_score = score / 100.0
        else:
            return -1, 0.0

    except Exception as exc:
        print(f"  [NLP] Edit Distance match failed ({exc}); defaulting to option 1.")
        return 0, 0.0

    print(f"  [NLP] Closest Match (Edit Distance): '{query_text}' -> '{labels[best_idx]}' (Score: {best_score:.4f})")
    
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
    base_frames = getattr(env, "frames", []) or []
    wrist_frames = getattr(env, "wrist_frames", []) or []
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
            metadata_path=metadata_path if metadata_path.exists() else None,
            render_mode=self.render_mode,
            gui_render=self.gui_render,
            max_steps_without_demonstration=self.max_steps,
        )

        env, seed, difficulty = resolver.make_env_for_episode(episode_idx)
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
# Main Loop (Refactored)
# =============================================================================

def main():
    # Initialization Wrapper
    oracle_resolver = EpisodeConfigResolverForOraclePlanner(
        gui_render=True,#环境本身是否开启gui渲染
        max_steps_without_demonstration=3000
    )

    use_segmentation = False
    mouse_select = True
    use_visualization = True 
    
    env_id_list = [
        #"PickXtimes",
        #"StopCube",
        #"SwingXtimes",
        #"BinFill",
       # "VideoUnmaskSwap",
        #"VideoUnmask",
        #"ButtonUnmaskSwap",
        # "ButtonUnmask",
        # "VideoRepick",
         #"VideoPlaceButton",
        # "VideoPlaceOrder",
        "VideoPlaceOrder",
         #"InsertPeg",
        #'MoveCube',
        #"PatternLock",
       #"RouteStick"
    ]

    try:
        for env_id in env_id_list:
            num_episodes = oracle_resolver.get_num_episodes(env_id)
            if num_episodes <= 0:
                print(f"[{env_id}] No episodes detected; skipping.")
                continue
            for episode in range(num_episodes):
                if episode != 42:
                    continue

                # --- WRAPPED INITIALIZATION  ---
                env, planner, color_map, language_goal = oracle_resolver.initialize_episode(env_id, episode)

                figures_to_close = []
              
                while True:
                    # [Step A] 获取当前帧视觉数据并构建可行动作列表
                    seg_vis, seg_raw, base_frames, wrist_frames, available_options = step_before(
                        env,
                        planner,
                        env_id,
                        color_map,
                        use_segmentation=use_segmentation,#需要可视化这三个设置为true
                        use_visualize=use_visualization,
                        figures_to_close=figures_to_close
                    )

                    print(f"Available Actions: {available_options}")

                    # # [Step B] Input Collection
                    print(f"Language Goal: {language_goal}")
                    command_dict = get_input_from_gui(#从gui里获取用户输入的命令
                            env,
                            planner,
                            env_id,
                            seg_vis,
                            mouse_select=mouse_select,
                            use_visualize=use_visualization,
                            figures_to_close=figures_to_close
                    )
                    #command_dict={'action': 'Pick red cube', 'point': (255, 255)}
                    #command_dict={'action': 'Pick red cube', 'point': (255, 255)}
                 

                    # [Step C] 执行逻辑处理并调用求解器
                    evaluation = step_after(
                        env,
                        planner,
                        env_id,
                        seg_vis,
                        seg_raw,
                        base_frames,
                        wrist_frames,
                        command_dict,
                        use_visualize=use_visualization,#需要可视化这两个设置为true
                        figures_to_close=figures_to_close
                    )

                    _cleanup_figures(figures_to_close, close_all=True)#每个step需要清理可视化的图像窗口，以防混淆
                    figures_to_close = []

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
                            
               
                    _cleanup_figures(figures_to_close, close_all=True)
 
    finally:
        oracle_resolver.close()

if __name__ == "__main__":
    main()
