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

# Ensure script can find root modules
_ROOT = os.path.abspath(os.path.dirname(__file__))
_PARENT = os.path.dirname(_ROOT)
for _path in (_PARENT, _ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from historybench.env_record_wrapper import EpisodeConfigResolver
from historybench.HistoryBench_env import *
from historybench.env_record_wrapper import *
from historybench.HistoryBench_env.util import *
from historybench.HistoryBench_env.util import task_goal
from historybench.HistoryBench_env.util.vqa_options import get_vqa_options

import torch
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None
else:
    # Adjust font path if necessary for your system
    _BANNER_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        _BANNER_FONT = ImageFont.truetype(_BANNER_FONT_PATH, 18)
    except OSError:
        _BANNER_FONT = None

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
    PandaStickMotionPlanningSolver,
)


def _render_text_with_pillow(image, lines, font, line_h, color=(0, 255, 0), offset=(10, 18)):
    if font is None or Image is None or ImageDraw is None:
        for idx, line in enumerate(lines):
            y_pos = offset[1] + idx * line_h
            cv2.putText(
                image,
                line,
                (offset[0], y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return image

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    for idx, line in enumerate(lines):
        y_pos = offset[1] + idx * line_h
        draw.text((offset[0], y_pos), line, font=font, fill=tuple(color))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _get_font_height(font, sample_text="Example"):
    if font is None:
        return 0
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(sample_text)
        return bbox[3] - bbox[1]
    if hasattr(font, "getsize"):
        return font.getsize(sample_text)[1]
    return 0


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


def _print_terminal_options(options, gui_state):
    """Mirror GUI option list in terminal for quick reference."""
    if not options:
        return
    selection = (gui_state or {}).get("selection")
    pending = (gui_state or {}).get("pending_option")
    print("\n[Options]")
    for idx, opt in enumerate(options):
        prefix = "> " if idx in (selection, pending) else ""
        label = opt.get("label", f"Option {idx + 1}")
        print(f"  {idx + 1}. {prefix}{label}")


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


def _compose_vertical_layout(
    frame_idx,
    seg_vis,
    base_frames,
    wrist_frames,
    seg_hw=(360, 480),
    video_hw=(180, 240),
):
    seg_h, seg_w = seg_hw
    vid_h, vid_w = video_hw
    width = max(seg_w, vid_w * 2)

    blank_seg = np.zeros((seg_h, seg_w, 3), dtype=np.uint8)
    seg_vis = seg_vis if seg_vis is not None else blank_seg
    if seg_vis.shape[:2] != (seg_h, seg_w):
        seg_vis = cv2.resize(seg_vis, (seg_w, seg_h), interpolation=cv2.INTER_NEAREST)

    seg_canvas = np.zeros((seg_h, width, 3), dtype=np.uint8)
    seg_x = (width - seg_w) // 2
    seg_canvas[:, seg_x : seg_x + seg_w] = seg_vis
    seg_rect = (seg_x, 0, seg_x + seg_w - 1, seg_h - 1)

    base_frames = base_frames or []
    wrist_frames = wrist_frames or []

    def _get_frame(arr, idx):
        if not arr:
            return None
        if idx < 0:
            idx = len(arr) - 1
        idx = idx % len(arr)
        return arr[idx]

    def _prep(arr):
        blank = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)
        if arr is None:
            return blank
        frame = _prepare_frame(arr)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if frame.shape[:2] != (vid_h, vid_w):
            frame = cv2.resize(frame, (vid_w, vid_h), interpolation=cv2.INTER_LINEAR)
        return frame

    video_idx = min(frame_idx, len(base_frames) - 1) if base_frames else 0
    base_video = _prep(_get_frame(base_frames, video_idx))
    wrist_video = _prep(_get_frame(wrist_frames, video_idx)) if wrist_frames else _prep(None)

    video_row = np.zeros((vid_h, width, 3), dtype=np.uint8)
    vid_x = (width - vid_w * 2) // 2
    video_row[:, vid_x : vid_x + vid_w] = base_video
    video_row[:, vid_x + vid_w : vid_x + 2 * vid_w] = wrist_video

    grid = np.vstack([seg_canvas, video_row])
    return grid, seg_rect


def _wrap_text_for_button(text, max_width, font, font_scale, thickness):
    if not text:
        return [""]
    words = text.split()
    if not words:
        return [text]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        width = cv2.getTextSize(candidate, font, font_scale, thickness)[0][0]
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_buttons_bottom(grid, options, button_h=42, margin=8):
    width = grid.shape[1]
    total = len(options)
    panel_h = total * (button_h + margin) + margin
    panel = np.zeros((panel_h, width, 3), dtype=np.uint8)
    buttons = []

    for idx_in_list, opt in enumerate(options):
        y1 = margin + idx_in_list * (button_h + margin)
        y2 = y1 + button_h
        x1 = margin
        x2 = width - margin

        rect_color = (90, 140, 255)
        text_color = (255, 255, 255)
        cv2.rectangle(panel, (x1, y1), (x2, y2), rect_color, thickness=-1)
        label = opt.get("label", f"Option {idx_in_list + 1}")
        lines = _wrap_text_for_button(
            label, x2 - x1 - 12, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        line_spacing = 4
        line_sizes = [
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0] for line in lines
        ]
        text_height = sum(size[1] for size in line_sizes) + line_spacing * (len(lines) - 1)
        cursor_y = y1 + (button_h - text_height) // 2
        for line, size in zip(lines, line_sizes):
            cursor_y += size[1]
            cv2.putText(
                panel,
                line,
                (x1 + 6, cursor_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1,
                cv2.LINE_AA,
            )
            cursor_y += line_spacing
        buttons.append(
            {
                "idx": idx_in_list,
                "rect": (x1, grid.shape[0] + y1, x2, grid.shape[0] + y2),
            }
        )

    combined = np.vstack([grid, panel])
    return combined, buttons


def _build_solve_options(env, planner, selected_target, env_id):
    return get_vqa_options(env, planner, selected_target, env_id)

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
    Renders ONLY the seg_vis image with point annotations.
    Processes inputs based on persistent state.
    """
    if simulated_inputs is None:
        simulated_inputs = []

    window_name = "Demonstration (Automated)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Use external state if provided, otherwise create new
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

    # --- 2. Helper Functions (Logic only) ---

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
        # Handles logic of a click without requiring actual mouse event
        seg_rect = state.get("seg_rect")
        seg_raw_local = state.get("seg_raw")

        if seg_rect and seg_raw_local is not None:
            x1, y1, x2, y2 = seg_rect
            # Check if inside segmentation rect
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
                        print("AUTO: No option armed before coordinates.")
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
                        # Fallback to direct segmentation ID under pixel
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

    # --- 3. Rendering Logic (SIMPLIFIED) ---

    def render_current_state(simulated_step_idx, last_cmd):
        """Renders ONLY the annotated segmentation image."""
        
        # 1. Prepare Display Image
        if seg_vis is not None:
            # Copy strictly needed to avoid drawing on the original buffer if reused
            display_img = seg_vis.copy()
        else:
            # Fallback blank image
            h, w = seg_hw
            display_img = np.zeros((h, w, 3), dtype=np.uint8)

        # 2. Draw Target Indicators (Red for click, Yellow for centroid)
        if selected_target.get("click_point"):
            cv2.circle(display_img, selected_target["click_point"], 6, (0, 0, 255), 2)
        if selected_target.get("centroid_point"):
            cv2.circle(display_img, selected_target["centroid_point"], 6, (0, 255, 255), 2)

        # 3. Update State for Logic
        # Since the image takes up the entire window, the rect starts at 0,0
        h, w = display_img.shape[:2]
        state["seg_rect"] = (0, 0, w - 1, h - 1)
        state["buttons"] = [] # No buttons used

        # 4. Display
        cv2.imshow(window_name, display_img)
        cv2.waitKey(20)

    # --- 4. Main Processing Loop ---

    # Render once initially to update visuals and calc rects
    render_current_state(-1, "WAITING")

    # If inputs provided, process them
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

                # Logic: If option needs target, arm it. If confirmed, select it.
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
                    # No target needed
                    state["selection"] = idx
            else:
                print(f"AUTO: Invalid index {idx}")

        # CASE B: Coordinates (e.g. "150 150")
        elif len(tokens) >= 2:
            try:
                cx = int(float(tokens[0]))
                cy = int(float(tokens[1]))

                # Try to use existing rects to resolve click
                # If coords are small (relative), assume they are inside seg rect
                seg_rect = state.get("seg_rect")
                if seg_rect:
                    # Assuming local coords relative to image start (which is now 0,0)
                    global_x = seg_rect[0] + cx
                    global_y = seg_rect[1] + cy
                    _handle_logic_click(global_x, global_y)
            except ValueError:
                print("AUTO: Coord parse error")

        # Render result of this step
        render_current_state(i, cmd)

    return state

def main():
    num_episodes = 100
    gui_render = True
    max_steps_without_demonstration = 3000
    dataset_root = Path("/data/hongzefu/dataset_demonstration")
    use_segmentation = False

    env_id_list = [
        "VideoRepick",
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
                if episode != 79:
                    continue

                env, episode_dataset, seed, difficulty = resolver.make_env_for_episode(episode)

                print(f"--- Running online evaluation for episode:{episode} ---")
                env.reset()

                color_map = _generate_color_map()
                _sync_table_color(env, color_map)
                selected_target = {
                    "obj": None,
                    "name": None,
                    "seg_id": None,
                    "click_point": None,
                    "centroid_point": None,
                }

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
                solve_options = _build_solve_options(env, planner, selected_target, env_id)

                if not tasks or not solve_options:
                    env.close()
                    continue

                ordinal = 0
                
                # --- STATEFUL GUI LOOP ---
                # Initialize persistent GUI state for this task sequence
                gui_state = None 

                while True:
                    base_frames = getattr(env, "frames", []) or []
                    wrist_frames = getattr(env, "wrist_frames", []) or []
                    seg_data = _fetch_segmentation(env)

                    seg_vis = None
                    seg_raw = None
                    seg_hw = (360, 480)

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
                        else:
                            seg_vis = np.zeros((seg_hw[0], seg_hw[1], 3), dtype=np.uint8)

                    # 1. RENDER (No input) - Just to show state or initial screen
                    gui_state = _prompt_next_task_gui(
                        solve_options,
                        env,
                        selected_target,
                        env_id,
                        seg_vis=seg_vis,
                        seg_raw=seg_raw,
                        base_frames=base_frames,
                        wrist_frames=wrist_frames,
                        simulated_inputs=[], 
                        external_state=gui_state 
                    )

                    # 2. CAPTURE INPUT (Terminal)
                    _print_terminal_options(solve_options, gui_state)
                    user_cmd = input(f"\n[Step {ordinal}] Enter command (e.g. '1', '150 150') or '1234' to confirm: ")
                    
                    if user_cmd.lower() == 'exit':
                        print("Exiting manually.")
                        break

                    # 3. CHECK CONFIRMATION
                    if user_cmd == "1234":
                        selection_idx = gui_state.get("selection") if gui_state else None
                        if selection_idx is not None:
                            print(f"Confirmed. Executing Option {selection_idx + 1}")
                            ordinal += 1
                            solve_entry = solve_options[selection_idx]
                            print(f"Executing: {solve_entry.get('label')}")
                            solve_entry.get("solve")()

                            # Evaluation
                            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
                            
                            # Reset GUI state for next sub-task
                            gui_state = None 
                            selected_target["obj"] = None # Clear target selection

                            if _tensor_to_bool(evaluation.get("fail", False)):
                                print("Failure condition.")
                                break
                            if _tensor_to_bool(evaluation.get("success", False)):
                                print("Success.")
                                break
                            
                            # Continue to next loop iteration (Start next sub-task)
                            continue 
                        else:
                            print("Cannot execute: No valid selection made yet.")
                            # Continue loop to ask for input again
                            continue

                    # 4. UPDATE GUI (With input)
                    gui_state = _prompt_next_task_gui(
                        solve_options,
                        env,
                        selected_target,
                        env_id,
                        seg_vis=seg_vis,
                        seg_raw=seg_raw,
                        base_frames=base_frames,
                        wrist_frames=wrist_frames,
                        simulated_inputs=[user_cmd], # Apply user input
                        external_state=gui_state
                    )

                cv2.destroyAllWindows()
                env.unwrapped.evaluate()
                env.close()


if __name__ == "__main__":
    main()
