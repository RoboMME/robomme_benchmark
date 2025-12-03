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

def get_obs_and_segmentation(env, env_id, color_map, use_segmentation=False, use_visualize=True):
    """
    Step A: Visualization Preparation & Initial Render.
    No longer requires real options or selected_target. Uses dummies for simple render.
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
    
    # Safety fallback: Ensure seg_vis is valid
    if seg_vis is None:
        seg_vis = np.zeros((seg_hw[0], seg_hw[1], 3), dtype=np.uint8)

    # 2. Render Step A (Static Plot)
    print("\n[Step A] Initial Render")
    
    # Use dummy data just for image generation
    dummy_target = {"obj": None, "name": None, "seg_id": None, "click_point": None, "centroid_point": None}
    
    _, img_a = _prompt_next_task_gui(
        [], # No options needed for initial render
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
    
    return  fig_a, seg_vis, seg_raw, base_frames, wrist_frames


def get_input_from_gui(env, planner, env_id, seg_vis, mouse_select=True, use_visualize=True):
    """
    Step B: Input Collection.
    Returns dictionary: {"action": "Name", "parameter": (x, y)} or None parameter.
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
    cmd1 = input("  Command 1 (Option selection, e.g. '1'): ")
    
    selected_action_name = "Unknown"
    selected_idx = -1
    skip_coords = False

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

    # 2. Coordinates Selection
    coords = None

    if not skip_coords and selected_idx != -1:
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
        "parameter": coords
    }

    return command_dict, mouse_win_name


def solve_and_evaluate_from_command(env, planner, env_id, seg_vis, seg_raw, base_frames, wrist_frames, command_dict, use_visualize=True):
    """
    Step C: Logic Processing & Execution.
    Takes dictionary command, translates to simulated inputs for internal logic, and executes.
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
    target_param = command_dict.get("parameter")
    
    # Find the index corresponding to the action name
    found_idx = -1
    for i, opt in enumerate(solve_options):
        if opt.get("label") == target_action:
            found_idx = i
            break
            
    simulated_inputs = []
    
    if found_idx != -1:
        # 1. Select the option
        simulated_inputs.append(str(found_idx + 1))
        
        # 2. Provide coordinates if they exist
        if target_param is not None:
             simulated_inputs.append(f"{target_param[0]} {target_param[1]}")
        else:
             # If param is None, we skip coords input, or pass dummy if strictly required by old logic.
             # Based on previous logic, if targets are available, coords are needed.
             # If not available (global action), just selecting the option again confirms it.
             # Safest bet is to pass "0 0" only if we think logic needs it, but let's rely on
             # the logic that 'available' check determines flow.
             # However, to simulate "skipping" input, we just don't add the coord string.
             # BUT, if the internal logic expects a click to confirm target selection:
             if solve_options[found_idx].get("available"):
                 simulated_inputs.append("0 0") # Fallback dummy if logic demands coord

        # 3. Confirm selection (repeat option index)
        simulated_inputs.append(str(found_idx + 1))
    else:
        print(f"Error: Action '{target_action}' not found in current options.")

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

    # --- 4. Evaluation ---
    env.unwrapped.evaluate()
    evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
    
    fail_flag = evaluation.get("fail", False)
    success_flag = evaluation.get("success", False)
    
    print(evaluation)
    
    return evaluation, fig_c


# =============================================================================
# Main Loop
# =============================================================================

def main():
    num_episodes = 100
    gui_render = True
    max_steps_without_demonstration = 3000
    dataset_root = Path("/data/hongzefu/dataset_demonstration")
    use_segmentation = False
    mouse_select = True
    use_visualization = True 

    
    env_id_list =[
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
        'MoveCube',
        "PatternLock",
        "RouteStick"

        ]

    render_mode = "human" if gui_render else "rgb_array"

    for env_id in env_id_list:
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
                if episode != 0:
                    continue

                env, episode_dataset, seed, difficulty = resolver.make_env_for_episode(episode)

                print(f"--- Running online evaluation for episode:{episode} ---")
                env.reset()

                language_goal = env.demonstration_data.get('language goal')

               

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
                
                # We check tasks, but option building happens inside steps now.
                if not tasks:
                    env.close()
                    continue

                figures_to_close = []
                try:
                    while True:
                        # [Step A] Visual Prep
                        fig_a, seg_vis, seg_raw, base_frames, wrist_frames = get_obs_and_segmentation(
                            env, 
                            env_id,
                            color_map,
                            use_segmentation=use_segmentation,
                            use_visualize=use_visualization
                        )
                        if fig_a:
                            figures_to_close.append(fig_a)

                        #[Step B] Input Collection - Returns Dict now
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

                        # [Step C] Process Logic & Solve - Accepts Dict now
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