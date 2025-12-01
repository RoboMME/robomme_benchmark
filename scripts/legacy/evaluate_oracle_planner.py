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


def _prompt_next_task(tasks, remaining_indices):
    """提示用户选择下一个要执行的 solve（1 基索引，回车停止）。"""
    print("Available solves:")
    for idx in remaining_indices:
        name = tasks[idx].get("name", f"Task {idx}")
        print(f"  [{idx + 1}] {name}")

    selection = input(
        "Select the next solve index to run (1-based, empty to stop): "
    ).strip()

    if not selection:
        return None

    try:
        idx = int(selection) - 1
    except ValueError:
        print(f"Ignoring invalid selection: {selection}")
        return None

    if idx not in remaining_indices:
        print(f"Selection {selection} not in remaining solves; please pick again.")
        return None

    return idx


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
                    # 交互式选择下一个 solve，回车停止
                    task_idx = _prompt_next_task(tasks, remaining_indices)
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
