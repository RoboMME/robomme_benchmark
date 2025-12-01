import os
import sys

# 保证脚本直接运行时能找到工程根目录下的模块
_ROOT = os.path.abspath(os.path.dirname(__file__))
_PARENT = os.path.dirname(_ROOT)
for _path in (_PARENT, _ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import sapien
import gymnasium as gym
import cv2
import colorsys
import h5py
from pathlib import Path
from historybench.env_record_wrapper import EpisodeConfigResolver


from historybench.HistoryBench_env import *  # 引入自定义环境注册与工具
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


def _render_text_with_pillow(image, lines, font, line_h, color=(0, 255, 0), offset=(10, 18)):
    """
    用 Pillow 在给定图像绘制换行文本，返回依然为 BGR 的 numpy 图像。
    如果字体或 Pillow 不可用，就回退到 OpenCV 。
    """
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

if Image is None:
    _BANNER_FONT = None


def _get_font_height(font, sample_text="例"):
    """返回字体的像素高度."""
    if font is None:
        return 0
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(sample_text)
        return bbox[3] - bbox[1]
    if hasattr(font, "getsize"):
        return font.getsize(sample_text)[1]
    return 0

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
    PandaStickMotionPlanningSolver,
)


def _generate_color_map(n=10000, s_min=0.70, s_max=0.95, v_min=0.78, v_max=0.95):
    """与演示包装器一致的颜色表，保证分割可视化稳定。"""
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
    """
    若 env.segmentation_id_map 里有名为 table-workspace 的对象，则强制将其颜色设为黑色。
    便于分割可视化时弱化背景桌面。
    """
    seg_id_map = getattr(env.unwrapped, "segmentation_id_map", None)
    if not isinstance(seg_id_map, dict):
        return
    for obj_id, obj in seg_id_map.items():
        if getattr(obj, "name", None) == "table-workspace":
            color_map[obj_id] = [0, 0, 0]


def _tensor_to_bool(value):
    """
    将多种标志值安全地转换为 Python bool。
    - 支持 torch.Tensor：先转到 CPU，再转 bool
    - 支持 numpy 数组：只要有任意非零元素就视为 True
    - None 视为 False，其他类型走内置 bool 逻辑
    """
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)


def _prepare_frame(frame):
    """
    将任意帧数据规范化为 uint8 RGB。
    - 若为 float 且最大值<=1，按 0~255 缩放
    - 若为灰度图，扩展为 3 通道
    - 返回值不改动输入对象本身
    """
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
    """
    将分割 mask 转成可点击的伪彩色图，同时返回原始二维 mask。
    - segmentation: torch/np，形状(1,H,W)或(H,W)
    - color_map: dict[id]->RGB
    - target_hw: 最终显示大小 (h, w)
    """
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
    """尝试从环境获取 base_camera segmentation。"""
    try:
        obs = env.unwrapped.get_obs(unflattened=True)
        return obs["sensor_data"]["base_camera"]["segmentation"]
    except Exception:
        return None


def _compose_vertical_layout(
    env,
    frame_idx,
    seg_vis,
    seg_hw=(360, 480),
    video_hw=(180, 240),
):
    """
    竖向布局：
    - 顶部：分割大图（可点击）
    - 中部：两个小的历史播放（base / wrist）
    - 按钮区域由后续函数叠加
    返回：组合图、分割区域矩形（用于点击映射）
    """
    seg_h, seg_w = seg_hw
    vid_h, vid_w = video_hw
    width = max(seg_w, vid_w * 2)

    blank_seg = np.zeros((seg_h, seg_w, 3), dtype=np.uint8)
    seg_vis = seg_vis if seg_vis is not None else blank_seg
    if seg_vis.shape[:2] != (seg_h, seg_w):
        seg_vis = cv2.resize(seg_vis, (seg_w, seg_h), interpolation=cv2.INTER_NEAREST)

    # 分割图居中放置
    seg_canvas = np.zeros((seg_h, width, 3), dtype=np.uint8)
    seg_x = (width - seg_w) // 2
    seg_canvas[:, seg_x : seg_x + seg_w] = seg_vis
    seg_rect = (seg_x, 0, seg_x + seg_w - 1, seg_h - 1)

    # 取历史帧（base/wrist）作为中部小图
    base_frames = getattr(env, "frames", []) or []
    wrist_frames = getattr(env, "wrist_frames", []) or []

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
    """按宽度包装按钮文字，返回不超过 max_width 的多行列表。"""
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
    """
    每行只放一个按钮，避免并列排布。
    - grid: 已拼好的 2x2 图
    - options: [{'label': str, 'solve': callable}, ...]
    - 返回拼接后的图像以及每个按钮的矩形坐标，用于点击检测
    """
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
    """
    按环境对象属性动态生成可用的 solve 列表。
    - 支持通过分割点击选择 target cube
    - 根据 env_id 拉取 VQA 选项模板
    - 返回列表元素包含 label 与 solve 可调用对象
    """
    return get_vqa_options(env, planner, selected_target, env_id)


def _prompt_next_task_gui(
    options,
    env,
    color_map,
    selected_target,
    env_id,
    fps=12,
    seg_hw=(360, 480),
    video_hw=(180, 240),
):
    """
    播放演示并等待用户选择下一个 solve。
    - 顶部大分割图可点击选目标；中部两个小窗播放历史视频；底部按钮
    - 支持数字键 1-9 选择；空格暂停/恢复；q/Esc 退出
    - 鼠标可直接点击分割图选中 target cube（来自 obs['sensor_data']['base_camera']['segmentation']）
    - 每轮循环都会读取最新的 env.frames，确保展示与任务同步
    """
    window_name = "Demonstration (press number to choose, q/Esc to stop)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        language_goal = task_goal.get_language_goal(env, env_id)
    except Exception as exc:  # 防御性兜底，不影响主流程
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
    state = {"buttons": [], "selection": None, "seg_raw": None, "seg_rect": None}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # 优先处理分割点击：右上角区域
        seg_rect = state.get("seg_rect")
        seg_raw = state.get("seg_raw")
        if seg_rect and seg_raw is not None:
            x1, y1, x2, y2 = seg_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                disp_w = max(1, x2 - x1 + 1)
                disp_h = max(1, y2 - y1 + 1)
                seg_h, seg_w = seg_raw.shape[:2]
                sx = int((x - x1) * seg_w / disp_w)
                sy = int((y - y1) * seg_h / disp_h)
                if 0 <= sx < seg_w and 0 <= sy < seg_h:
                    seg_id = int(seg_raw[sy, sx])
                    obj = getattr(env.unwrapped, "segmentation_id_map", {}).get(seg_id)
                    if obj is not None:
                        selected_target["obj"] = obj
                        selected_target["name"] = getattr(obj, "name", f"id_{seg_id}")
                        selected_target["seg_id"] = seg_id
                        # 记录点击点用于可视化
                        selected_target["click_point"] = (int((x - x1)), int((y - y1)))
                        print(f"Selected target via segmentation id={seg_id}, obj name={selected_target['name']}")
                    else:
                        print(f"No object found for segmentation id={seg_id}")
                return

        # 支持鼠标点击底部按钮直接选择 solve
        for btn in state["buttons"]:
            x1, y1, x2, y2 = btn["rect"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                state["selection"] = btn["idx"]
                break

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        seg_raw = _fetch_segmentation(env)
        seg_vis, seg_raw = _prepare_segmentation_visual(seg_raw, color_map, seg_hw)

        # 如果之前点击过，标记在分割可视化上
        if seg_vis is not None and selected_target.get("click_point"):
            cv2.circle(seg_vis, selected_target["click_point"], 6, (0, 0, 255), 2)

        grid, seg_rect = _compose_vertical_layout(env, frame_idx, seg_vis, seg_hw=seg_hw, video_hw=video_hw)
        combined, buttons = _draw_buttons_bottom(grid, options)
        selected_name = selected_target.get("name") or "None"
        banner_lines = []
        if language_goal:
            banner_lines.extend(_wrap_text_line(f"Goal: {language_goal}", combined.shape[1] - 20))
        banner_lines.append(f"Selected target: {selected_name}")
        banner_lines.extend(
            _wrap_text_line(
                "Click segmentation to pick target cube; use buttons/number keys to run solve. Space pause, q/Esc exit.",
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
        state["seg_raw"] = seg_raw
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
                selection = idx
                break
        elif state["selection"] is not None:
            selection = state["selection"]
            break
        # 其他按键忽略

        if not paused:
            # 播放索引自增；到达最后一帧后自动暂停等待用户操作
            base_len = len(getattr(env, "frames", []) or [])
            if base_len and frame_idx < base_len - 1:
                frame_idx += 1
            else:
                paused = True  # 播放到最后一帧后停止

    cv2.destroyWindow(window_name)
    return selection


def main():
    """ 
    在线评估入口。
    - 按照演示数据逐集驱动 env
    - 每次 solve 前后调用 env.unwrapped.evaluate() 观察成功/失败
    - 通过 GUI/数字键选择要执行的策略
    """

    # 配置区：按需修改
    num_episodes = 1  # 每个环境重复次数
    gui_render = True  # 是否开启 GUI 渲染
    max_steps_without_demonstration = 3000  # 演示间隔上限
    dataset_root = Path("/data/hongzefu/dataset_demonstration")

    env_id_list = [
        # "VideoRepick",
        # "BinFill",
        # "ButtonUnmask",
        # "ButtonUnmaskSwap",
        # "InsertPeg",
        # "MoveCube",
        # "PatternLock",
        # "PickHighlight",
        # "PickXtimes",
        "RouteStick",
        # "StopCube",
        # "SwingXtimes",
        # "VideoPlaceButton",
        # "VideoPlaceOrder",
        # "VideoUnmask",
        # "VideoUnmaskSwap",
    ]

    # GUI/无 GUI 渲染模式
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
                env, episode_dataset, seed, difficulty = resolver.make_env_for_episode(episode)
                seed_desc = seed if seed is not None else "default"
                difficulty_desc = difficulty if difficulty else "default"
                print(
                    f"--- Running online evaluation for episode:{episode}, env:{env_id}, "
                    f"seed:{seed_desc}, difficulty:{difficulty_desc} ---"
                )

                env.reset()

                # 分割颜色表与用户点击的目标对象占位
                color_map = _generate_color_map()
                _sync_table_color(env, color_map)
                selected_target = {"obj": None, "name": None, "seg_id": None, "click_point": None}

                # 根据环境类型选择不同的规划器
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

                # 先 evaluate 一次以初始化任务列表
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
                    # 实时显示视频并等待用户选择 solve；返回 None 代表退出
                    solve_idx = _prompt_next_task_gui(
                        solve_options, env, color_map, selected_target, env_id, fps=120
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

                    # 在执行前后各评估一次，便于及时发现失败或成功
                    evaluation = env.unwrapped.evaluate(solve_complete_eval=True)  # 运行前评估
                    solve_callable()
                    evaluation = env.unwrapped.evaluate(solve_complete_eval=True)  # 运行后评估
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

                print(
                    f"--- Finished online evaluation for episode:{episode}, env:{env_id} ---"
                )


if __name__ == "__main__":
    main()
