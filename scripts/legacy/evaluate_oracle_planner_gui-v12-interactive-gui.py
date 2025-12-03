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

# 保证脚本直接运行时能找到工程根目录下的模块
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
    """用 Pillow 在给定图像绘制换行文本，返回依然为 BGR 的 numpy 图像。"""
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


def _get_font_height(font, sample_text="例"):
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
):
    """
    GUI函数：只负责渲染传入的静态图像，不进行环境数据获取。
    """
    window_name = "Demonstration (press number to choose, q/Esc to stop)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        language_goal = task_goal.get_language_goal(env, env_id)
    except Exception as exc:
        print(f"[warn] failed to get language goal: {exc}")
        language_goal = ""

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1

    def _wrap_text_line(text, max_width):
        words = (text or "").split()
        if not words:
            return []
        lines = []
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            (w, _), _ = cv2.getTextSize(trial, font, font_scale, thickness)
            if w <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    delay = max(1, int(1000 / max(fps, 1)))
    frame_idx = 0
    paused = False
    selection = None
    state = {
        "buttons": [],
        "selection": None,
        "seg_raw": seg_raw, # 直接使用传入的静态 seg_raw
        "seg_rect": None,
        "pending_option": None,
    }

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
            added = False
            for actor in iterable:
                if actor is None:
                    continue
                identifier = id(actor)
                if identifier in seen:
                    continue
                seen.add(identifier)
                candidates.append(actor)
                added = True
            return added

        def _extend_from_option(idx):
            if idx is None or not (0 <= idx < len(options)):
                return False
            avail = options[idx].get("available")
            if avail is None:
                return False
            flattened = _flatten_items(avail)
            return _extend_from_iterable(flattened)

        if selected_idx is not None:
            _extend_from_option(selected_idx)
            return candidates

        for idx in range(len(options)):
            _extend_from_option(idx)

        if candidates:
            return candidates

        fallback = getattr(env, "spawned_cubes", None)
        if fallback:
            try:
                iterable = list(fallback)
            except TypeError:
                iterable = []
            _extend_from_iterable(iterable)

        return candidates

    def _handle_click(global_x, global_y):
        target_chosen = False
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
                        print("Select an option button before choosing a target point.")
                        return True

                    available_targets = _collect_available_targets(pending_option)
                    if not available_targets:
                        print("The selected option has no available targets to click.")
                        return True
                    click_disp = (global_x - x1, global_y - y1)

                    candidates = []
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
                        seg_id = chosen["seg_id"]
                        selected_target["obj"] = obj
                        selected_target["name"] = getattr(obj, "name", f"id_{seg_id}")
                        selected_target["seg_id"] = seg_id
                        selected_target["click_point"] = (int(click_disp[0]), int(click_disp[1]))
                        selected_target["centroid_point"] = chosen["disp_point"]
                        target_chosen = True
                        print(
                            f"Selected nearest available target via segmentation id={seg_id}, "
                            f"obj name={selected_target['name']}"
                        )
                    else:
                        seg_id = int(seg_raw_local[sy, sx])
                        obj = seg_id_map.get(seg_id)
                        if obj is not None:
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
                            else:
                                selected_target["centroid_point"] = None
                            target_chosen = True
                            print(
                                f"Selected target via segmentation id={seg_id}, "
                                f"obj name={selected_target['name']}"
                            )
                        else:
                            print(f"No object found for segmentation id={seg_id}")
                return True

        for btn in state["buttons"]:
            x1, y1, x2, y2 = btn["rect"]
            if x1 <= global_x <= x2 and y1 <= global_y <= y2:
                idx = btn["idx"]
                option_avail = _collect_available_targets(idx)
                if option_avail:
                    pending = state.get("pending_option")
                    if pending == idx:
                        obj = selected_target.get("obj")
                        if obj is None:
                            print("Target not selected yet; click on the segmentation view first.")
                            return True
                        ids = {id(actor) for actor in option_avail}
                        if id(obj) not in ids:
                            print("Selected target does not belong to this option; please reselect.")
                            _reset_selected_target()
                            return True
                        state["selection"] = idx
                        state["pending_option"] = None
                    else:
                        state["pending_option"] = idx
                        state["selection"] = None
                        _reset_selected_target()
                        print(
                            "Option armed. Click on the segmentation map to pick one of its targets, then click the same button again to confirm."
                        )
                else:
                    state["selection"] = idx
                    state["pending_option"] = None
                return True
        return False

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        _handle_click(x, y)

    cv2.setMouseCallback(window_name, on_mouse)

    print("\n可用选项（输入数字直接选择，或在终端输入 x y 坐标模拟点击，q 退出）:")
    for idx, opt in enumerate(options):
        label = opt.get("label", f"Option {idx + 1}")
        print(f"  {idx + 1}: {label}")

    while True:
        # 复制一份显示用的图像，以免 marker 污染原图
        seg_vis_display = seg_vis.copy() if seg_vis is not None else None

        if seg_vis_display is not None and selected_target.get("click_point"):
            cv2.circle(seg_vis_display, selected_target["click_point"], 6, (0, 0, 255), 2)
        if seg_vis_display is not None and selected_target.get("centroid_point"):
            cv2.circle(seg_vis_display, selected_target["centroid_point"], 6, (0, 255, 255), 2)

        # 这里直接使用传入的 base_frames 和 wrist_frames
        grid, seg_rect = _compose_vertical_layout(
            frame_idx,
            seg_vis_display,
            base_frames,
            wrist_frames,
            seg_hw=seg_hw,
            video_hw=video_hw,
        )
        numbered_options = []
        for idx, opt in enumerate(options):
            labeled = dict(opt)
            labeled["label"] = f"{idx + 1}. {opt.get('label', f'Option {idx + 1}')}"
            numbered_options.append(labeled)

        combined, buttons = _draw_buttons_bottom(grid, numbered_options)
        selected_name = selected_target.get("name") or "None"
        banner_lines = []
        if language_goal:
            banner_lines.extend(_wrap_text_line(f"Goal: {language_goal}", combined.shape[1] - 20))
        banner_lines.append(f"Selected target: {selected_name}")
        banner_lines.extend(
            _wrap_text_line(
                "Press/enter a number to arm, pick a target on the segmentation view (mouse or terminal 'x y'), then press the same number again to confirm. Options without targets run immediately. Space pause, q/Esc exit.",
                combined.shape[1] - 20,
            )
        )
        line_h = 22
        if _BANNER_FONT is not None:
            font_height = _get_font_height(_BANNER_FONT, "例") + 4
            line_h = max(line_h, font_height)
        banner_h = max(30, 12 + line_h * len(banner_lines))
        banner = np.zeros((banner_h, combined.shape[1], 3), dtype=np.uint8)
        banner = _render_text_with_pillow(
            banner,
            banner_lines,
            _BANNER_FONT,
            line_h,
            color=(0, 255, 0),
            offset=(10, 18),
        )
        combined = np.vstack([banner, combined])
        banner_h = banner.shape[0]

        adjusted_buttons = []
        for btn in buttons:
            x1, y1, x2, y2 = btn["rect"]
            adjusted_buttons.append({"idx": btn["idx"], "rect": (x1, y1 + banner_h, x2, y2 + banner_h)})

        adjusted_seg_rect = (
            None
            if seg_rect is None
            else (seg_rect[0], seg_rect[1] + banner_h, seg_rect[2], seg_rect[3] + banner_h)
        )

        state["buttons"] = adjusted_buttons
        state["seg_rect"] = adjusted_seg_rect

        cv2.imshow(window_name, combined)
        key = cv2.waitKey(100 if paused else delay) & 0xFF

        if key in (ord("q"), 27):
            selection = None
            break
        elif key == ord(" "):
            paused = not paused
        elif ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            if idx < len(options):
                option_avail = _collect_available_targets(idx)
                if option_avail:
                    if state.get("pending_option") == idx:
                        obj = selected_target.get("obj")
                        if obj is not None and any(id(obj) == id(actor) for actor in option_avail):
                            selection = idx
                            break
                        print("Target not selected or mismatched; click segmentation then press the number again.")
                    else:
                        state["pending_option"] = idx
                        state["selection"] = None
                        _reset_selected_target()
                        print("Option armed. Click on segmentation to pick target, then press the same number to confirm.")
                else:
                    selection = idx
                    break
        elif state["selection"] is not None:
            selection = state["selection"]
            break

        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
        except Exception:
            ready = []
        if ready:
            line = sys.stdin.readline().strip()
            if line:
                lowered = line.lower()
                if lowered in ("q", "quit", "exit"):
                    selection = None
                    break
                tokens = line.replace(",", " ").split()
                if len(tokens) == 1 and tokens[0].isdigit():
                    idx = int(tokens[0]) - 1
                    if 0 <= idx < len(options):
                        option_avail = _collect_available_targets(idx)
                        if option_avail:
                            if state.get("pending_option") == idx:
                                obj = selected_target.get("obj")
                                if obj is not None and any(id(obj) == id(actor) for actor in option_avail):
                                    selection = idx
                                    break
                                print("Target not selected or mismatched; enter coordinates then press the same number again.")
                            else:
                                state["pending_option"] = idx
                                state["selection"] = None
                                _reset_selected_target()
                                print(
                                    f"Option {idx + 1} armed via terminal. Enter 'x y' to pick a target, then press {idx + 1} again to confirm."
                                )
                        else:
                            selection = idx
                            break
                    else:
                        print(f"Invalid option index {tokens[0]}; valid range: 1-{len(options)}")
                elif len(tokens) >= 2:
                    try:
                        cx = int(float(tokens[0]))
                        cy = int(float(tokens[1]))
                        handled = _handle_click(cx, cy)
                        if not handled:
                            seg_rect = state.get("seg_rect")
                            if seg_rect is not None:
                                x1, y1, x2, y2 = seg_rect
                                disp_w = x2 - x1 + 1
                                disp_h = y2 - y1 + 1
                                if 0 <= cx < disp_w and 0 <= cy < disp_h:
                                    handled = _handle_click(x1 + cx, y1 + cy)
                        if handled:
                            continue
                    except ValueError:
                        print(f"Unrecognized input '{line}'.")
                else:
                    print(f"Unrecognized input '{line}'.")

        if not paused:
            # 这里的 base_frames 是外部传入的固定列表
            if base_frames and frame_idx < len(base_frames) - 1:
                frame_idx += 1
            else:
                paused = True

    cv2.destroyWindow(window_name)
    return selection


def main():
    num_episodes = 100
    gui_render = True
    max_steps_without_demonstration = 3000
    dataset_root = Path("/data/hongzefu/dataset_demonstration")
    use_segmentation = False  # 不使用分割显示模式

    env_id_list = [
        "VideoRepick",
        #"BinFill",
        #"ButtonUnmask",
        #"ButtonUnmaskSwap",
        # "InsertPeg",
        # "MoveCube",
        # "PatternLock",
        # "PickHighlight",
        # "PickXtimes",
        # "RouteStick",
        # "StopCube",
        # "SwingXtimes",
        # "VideoPlaceButton",
        # "VideoPlaceOrder",
        # "VideoUnmask",
        #"VideoUnmaskSwap",
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
                
                seed_desc = seed if seed is not None else "default"
                difficulty_desc = difficulty if difficulty else "default"
                print(
                    f"--- Running online evaluation for episode:{episode}, env:{env_id}, "
                    f"seed:{seed_desc}, difficulty:{difficulty_desc} ---"
                )

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

                if not tasks:
                    print("No tasks defined for this environment; skipping execution")
                    env.close()
                    continue
                if not solve_options:
                    print("No solve options available; skipping execution")
                    env.close()
                    continue

                print(f"{env_id}: Task list has {len(tasks)} tasks; {len(solve_options)} solve options available")

                ordinal = 0

                while True:
                    # 在调用 GUI 之前获取并处理好所有图像数据
                    base_frames = getattr(env, "frames", []) or []
                    wrist_frames = getattr(env, "wrist_frames", []) or []
                    
                    # 尝试获取 segmentation 数据用于后续点击计算
                    seg_data = _fetch_segmentation(env)
                    
                    seg_vis = None
                    seg_raw = None
                    seg_hw = (360, 480)

                    if use_segmentation:
                        # 模式1: 使用分割图作为主视图
                        seg_vis, seg_raw = _prepare_segmentation_visual(seg_data, color_map, seg_hw)
                    else:
                        # 模式2: 使用 RGB 图作为主视图
                        # 依然需要准备 seg_raw 用于点击逻辑(如果 seg_data 存在)
                        _, seg_raw = _prepare_segmentation_visual(seg_data, color_map, seg_hw) if seg_data is not None else (None, None)
                        
                        # 准备 用于选择的 RGB/分割 视图
                        if base_frames:
                            vis_frame = _prepare_frame(base_frames[-1])
                            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                            if vis_frame.shape[:2] != seg_hw:
                                vis_frame = cv2.resize(vis_frame, (seg_hw[1], seg_hw[0]), interpolation=cv2.INTER_LINEAR)
                            seg_vis = vis_frame
                        else:
                            seg_vis = np.zeros((seg_hw[0], seg_hw[1], 3), dtype=np.uint8)

                    # 调用纯渲染的 GUI 函数
                    solve_idx = _prompt_next_task_gui(
                        solve_options,
                        env,
                        selected_target,
                        env_id,
                        seg_vis=seg_vis,          # 传入处理好的主显示图
                        seg_raw=seg_raw,          # 传入处理好的原始分割ID图
                        base_frames=base_frames,  # 传入当前积累的视频列表
                        wrist_frames=wrist_frames,# 传入当前积累的视频列表
                        fps=120,
                        seg_hw=seg_hw
                    )
                    
                    if solve_idx is None:
                        print("Stopping by user choice.")
                        break

                    ordinal += 1
                    solve_entry = solve_options[solve_idx]
                    solve_name = solve_entry.get("label", f"Solve {solve_idx}")
                    print(f"Executing selected solve {ordinal}: {solve_name}")

                    solve_callable = solve_entry.get("solve")
                    if not callable(solve_callable):
                        raise ValueError(f"Solve '{solve_name}' must be callable.")

                    env.unwrapped.evaluate(solve_complete_eval=True)
                    solve_callable()
                    evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
                    fail_flag = evaluation.get("fail", False)
                    success_flag = evaluation.get("success", False)

                    print(evaluation)

                    if _tensor_to_bool(fail_flag):
                        print("Encountered failure condition; stopping task sequence.")
                        break

                    if _tensor_to_bool(success_flag):
                        print("All tasks completed successfully.")
                        break

                env.unwrapped.evaluate()
                env.close()
                print(f"--- Finished online evaluation for episode:{episode}, env:{env_id} ---")


if __name__ == "__main__":
    main()