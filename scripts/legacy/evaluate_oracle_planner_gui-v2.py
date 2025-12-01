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


from historybench.HistoryBench_env import *  # 引入自定义环境注册与工具
from historybench.env_record_wrapper import *
from historybench.HistoryBench_env.util import *

import torch

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
    PandaStickMotionPlanningSolver,
)


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


def _compose_quadrant_grid(env, frame_idx, target_hw=(240, 320)):
    """
    将当前环境帧拼成 2x2 方格用于预览。
    - 左列：按 frame_idx 播放的视频帧（base / wrist）
    - 右列：最新帧（base / wrist）
    - 若不存在手腕视角则补黑图；内部转换为 BGR 返回
    """
    h_target, w_target = target_hw
    blank = np.zeros((h_target, w_target, 3), dtype=np.uint8)

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
        if arr is None:
            return blank.copy()
        frame = _prepare_frame(arr)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if frame.shape[:2] != (h_target, w_target):
            frame = cv2.resize(frame, (w_target, h_target), interpolation=cv2.INTER_LINEAR)
        return frame

    video_idx = min(frame_idx, len(base_frames) - 1) if base_frames else 0
    base_video = _prep(_get_frame(base_frames, video_idx))
    wrist_video = _prep(_get_frame(wrist_frames, video_idx)) if wrist_frames else blank.copy()
    base_latest = _prep(_get_frame(base_frames, -1))
    wrist_latest = _prep(_get_frame(wrist_frames, -1)) if wrist_frames else blank.copy()

    top_row = np.hstack([base_video, base_latest])
    bottom_row = np.hstack([wrist_video, wrist_latest])
    return np.vstack([top_row, bottom_row])


def _draw_buttons_bottom(grid, options, button_w=180, button_h=42, margin=8):
    """
    在底部绘制交互按钮。
    - grid: 已拼好的 2x2 图
    - options: [{'label': str, 'solve': callable}, ...]
    - 返回拼接后的图像以及每个按钮的矩形坐标，用于点击检测
    """
    width = grid.shape[1]
    total = len(options)
    cols = max(1, (width - margin) // (button_w + margin))
    cols = max(1, cols)
    rows = max(1, int(np.ceil(total / cols))) if total > 0 else 1

    panel_h = rows * (button_h + margin) + margin
    panel = np.zeros((panel_h, width, 3), dtype=np.uint8)
    buttons = []

    for idx_in_list, opt in enumerate(options):
        row = idx_in_list // cols
        col = idx_in_list % cols

        x1 = margin + col * (button_w + margin)
        y1 = margin + row * (button_h + margin)
        x2 = min(x1 + button_w, width - margin)
        y2 = min(y1 + button_h, panel_h - margin)
        if y2 - y1 < 20:
            continue

        rect_color = (90, 140, 255)
        text_color = (255, 255, 255)
        cv2.rectangle(panel, (x1, y1), (x2, y2), rect_color, thickness=-1)
        label = opt.get("label", f"Option {idx_in_list + 1}")
        if len(label) > 28:
            label = label[:25] + "..."
        cv2.putText(
            panel,
            label,
            (x1 + 6, y2 - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
            cv2.LINE_AA,
        )
        buttons.append(
            {
                "idx": idx_in_list,
                "rect": (x1, grid.shape[0] + y1, x2, grid.shape[0] + y2),
            }
        )

    combined = np.vstack([grid, panel])
    return combined, buttons


def _build_solve_options(env, planner):
    """
    按环境对象属性动态生成可用的 solve 列表。
    - 目前只处理 cube 与按钮类任务，后续可继续扩展
    - 返回列表元素包含 label 与 solve 可调用对象
    """
    base = env.unwrapped
    options = []

    if hasattr(base, "target_cube_1"):
        options.append(
            {
                "label": "pickup cube1",
                "solve": lambda: solve_pickup(env, planner, obj=base.target_cube_1),
            }
        )
        options.append(
            {
                "label": "putdown",
                "solve": lambda: solve_putdown_whenhold(env, planner, obj=base.target_cube_1, release_z=0.01),
            }
        )
    if hasattr(base, "button_left"):
        options.append(
            {
                "label": "press button",
                "solve": lambda: solve_button(env, planner, obj=base.button_left),
            }
        )
    return options


def _prompt_next_task_gui(options, env, fps=12):
    """
    播放演示并等待用户选择下一个 solve。
    - 左列播放录制视频，右列实时最新帧，底部按钮可点击
    - 支持数字键 1-9 选择；空格暂停/恢复；q/Esc 退出
    - 每轮循环都会读取最新的 env.frames，确保展示与任务同步
    """
    window_name = "Demonstration (press number to choose, q/Esc to stop)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    delay = max(1, int(1000 / max(fps, 1)))
    frame_idx = 0
    paused = False
    selection = None
    state = {"buttons": [], "selection": None}

    def on_mouse(event, x, y, flags, param):
        # 支持鼠标点击底部按钮直接选择 solve
        if event == cv2.EVENT_LBUTTONDOWN:
            for btn in state["buttons"]:
                x1, y1, x2, y2 = btn["rect"]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    state["selection"] = btn["idx"]
                    break

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        grid = _compose_quadrant_grid(env, frame_idx)
        # 叠加提示
        y = 26
        cv2.putText(
            grid,
            "Left: video playback (base/wrist). Right: latest frame. Space pause, 1-9 select, q/Esc exit.",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        combined, buttons = _draw_buttons_bottom(grid, options)
        cv2.putText(
            combined,
            "Click buttons below or press number keys to pick a solve.",
            (10, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        state["buttons"] = buttons

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
    - 循环遍历 env_id_list，逐个 episode 运行
    - 每次 solve 前后调用 env.unwrapped.evaluate() 观察成功/失败
    - 通过 GUI/数字键选择要执行的策略
    """

    # 配置区：按需修改
    num_episodes = 1  # 每个环境重复次数
    gui_render = True  # 是否开启 GUI 渲染
    test_seed_offset = 0  # 随机种子偏移

    env_id_list = ["VideoRepick"]  # 需要评测的环境列表

    # GUI/无 GUI 渲染模式
    render_mode = "human" if gui_render else "rgb_array"

    for env_id in env_id_list:
        for episode in range(num_episodes):
            seed = test_seed_offset + episode
            print(
                f"--- Running online evaluation for episode:{episode}, env:{env_id}, seed:{seed} ---"
            )

            env = gym.make(
                env_id,
                obs_mode="rgb+depth+segmentation",
                control_mode="pd_joint_pos",
                render_mode=render_mode,  # human 时窗口渲染；rgb_array 时后台渲染
                reward_mode="dense",
                HistoryBench_seed=seed,
                max_episode_steps=99999,
            )
            # 包装演示记录器，保证长时间无演示时自动触发；gui_render 控制是否即时渲染
            env = DemonstrationWrapper(env, max_steps_without_demonstration=3000, gui_render=gui_render)
            env.reset()



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

            solve_options = _build_solve_options(env, planner)

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
                solve_idx = _prompt_next_task_gui(solve_options, env, fps=120)
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
