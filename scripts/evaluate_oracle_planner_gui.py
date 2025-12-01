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
    """将多种张量/数组/None 格式的标志位统一转成 Python bool。"""
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)


def _prepare_frame(frame):
    """确保帧数据为 uint8 RGB。"""
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


def _compose_display_frame(env, frame_idx):
    """
    直接读取 env.frames/env.wrist_frames，拼接为可展示的 BGR 帧。
    若无帧则返回纯黑背景。
    """
    base_frames = getattr(env, "frames", []) or []
    wrist_frames = getattr(env, "wrist_frames", []) or []

    if not base_frames:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    idx = frame_idx % len(base_frames)
    base = base_frames[idx]
    wrist = wrist_frames[idx] if idx < len(wrist_frames) else None

    prepared_base = _prepare_frame(base)
    if wrist is not None:
        prepared_wrist = _prepare_frame(wrist)
        if prepared_wrist.shape[:2] != prepared_base.shape[:2]:
            prepared_wrist = cv2.resize(
                prepared_wrist,
                (prepared_base.shape[1], prepared_base.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        prepared_base = np.concatenate((prepared_base, prepared_wrist), axis=1)

    return cv2.cvtColor(prepared_base, cv2.COLOR_RGB2BGR)


def _draw_buttons(display, tasks, remaining_indices, button_w=220, button_h=44, margin=8):
    """
    在右侧绘制多列按钮面板，返回拼接后的图像和按钮位置信息。
    自动分列以适配高度，尽量容纳更多按钮。
    """
    h = display.shape[0]
    max_rows = max(1, (h - 2 * margin) // (button_h + margin))
    total = len(remaining_indices)
    num_cols = max(1, int(np.ceil(total / max_rows)))

    panel_width = num_cols * (button_w + margin) + margin
    panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
    buttons = []

    for col in range(num_cols):
        for row in range(max_rows):
            idx_in_list = col * max_rows + row
            if idx_in_list >= total:
                break
            task_idx = remaining_indices[idx_in_list]

            x1 = margin + col * (button_w + margin)
            y1 = margin + row * (button_h + margin)
            x2 = x1 + button_w
            y2 = min(y1 + button_h, h - margin)
            if y2 - y1 < 20:
                continue

            rect_color = (90, 140, 255)
            text_color = (255, 255, 255)
            cv2.rectangle(panel, (x1, y1), (x2, y2), rect_color, thickness=-1)
            name = tasks[task_idx].get("name", f"Task {task_idx}")
            label = f"[{task_idx + 1}] {name}"
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
                    "idx": task_idx,
                    "rect": (display.shape[1] + x1, y1, display.shape[1] + x2, y2),
                }
            )

    combined = np.hstack([display, panel])
    return combined, buttons


def _prompt_next_task_gui(tasks, remaining_indices, env, fps=12):
    """
    GUI 播放 env.frames，点击右侧按钮或按数字键直接选择 solve；Esc/q 退出，空格暂停。
    每次循环都会使用最新的 env.frames，完成一个 solve 后可实时看到更新。
    """
    window_name = "Demonstration (press number to choose, q/Esc to stop)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    delay = max(1, int(1000 / max(fps, 1)))
    latest_count = len(getattr(env, "frames", []) or [])
    frame_idx = max(latest_count - 1, 0)
    paused = False
    selection = None
    state = {"buttons": [], "selection": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for btn in state["buttons"]:
                x1, y1, x2, y2 = btn["rect"]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    state["selection"] = btn["idx"]
                    break

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        display = _compose_display_frame(env, frame_idx)

        y = 30
        cv2.putText(
            display,
            "Click buttons or press 1-9 to select; Space pause; q/Esc stop",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        combined, buttons = _draw_buttons(display, tasks, remaining_indices)
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
            if idx in remaining_indices:
                selection = idx
                break
        elif state["selection"] is not None:
            selection = state["selection"]
            break
        # 其他按键忽略

        if not paused:
            frame_idx += 1

    cv2.destroyWindow(window_name)
    return selection


def main():
    """
    在线评估入口：
    - 按用户逐个选择的 solve 运行
    - 每次 solve 结束立即 evaluate，查看成功/失败
    - 可通过回车停止后续 solve
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
            env = DemonstrationWrapper(env, max_steps_without_demonstration=3000,gui_render=gui_render)
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

            if not tasks:
                print("No tasks defined for this environment; skipping execution")
            else:
                print(f"{env_id}: Task list has {len(tasks)} tasks")

                remaining_indices = list(range(len(tasks)))
                ordinal = 0

                while remaining_indices:
                    # 交互式选择下一个 solve，直接按键选择
                    task_idx = _prompt_next_task_gui(tasks, remaining_indices, env, fps=120)
                    
                    if task_idx is None:
                        print("Stopping by user choice.")
                        break

                    ordinal += 1
                    task_entry = tasks[task_idx]
                    task_name = task_entry.get("name", f"Task {task_idx}")
                    print(f"Executing selected task {ordinal}: {task_name}")

                    solve_callable = task_entry.get("solve")
                    if not callable(solve_callable):
                        raise ValueError(
                            f"Task '{task_name}' must supply a callable 'solve'."
                        )

                    evaluation = env.unwrapped.evaluate(solve_complete_eval=True)  # 运行前评估
                    solve_callable(env, planner)
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

                    # 已执行任务从候选列表移除，避免重复
                    remaining_indices = [i for i in remaining_indices if i != task_idx]

                env.unwrapped.evaluate()

            env.close()

            print(
                f"--- Finished online evaluation for episode:{episode}, env:{env_id} ---"
            )


if __name__ == "__main__":
    main()
