# -*- coding: utf-8 -*-
# 脚本功能：按 episode 运行 end-effector 评测（不读取 h5 数据集）
# 特点：跳过 demonstration 阶段后执行 dummy action，并保留 obs/info 展开逻辑

import os
import sys

# 将包根目录及上级目录加入 sys.path，便于作为脚本直接运行（不依赖 PYTHONPATH）
_ROOT = os.path.abspath(os.path.dirname(__file__))
_PARENT = os.path.dirname(_ROOT)
for _path in (_PARENT, _ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import torch

# HistoryBench 环境及工具
from historybench.HistoryBench_env import *
from historybench.HistoryBench_env.util import *
# 用于根据元数据创建环境
from historybench.env_record_wrapper import (
    BenchmarkEnvBuilder,
)

DEFAULT_ENV_IDS = [
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

def _flatten_column(batch_dict, key):
    out = []
    for item in (batch_dict or {}).get(key, []) or []:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            out.extend([x for x in item if x is not None])
        else:
            out.append(item)
    return out


def _last_info(info_batch, n):
    if n <= 0:
        return {}
    idx = n - 1
    return {k: v[idx] for k, v in (info_batch or {}).items() if len(v) > idx and v[idx] is not None}


def _get_dummy_action(robot_endeffector_p, robot_endeffector_q, actions):
    # """用当前 robot_endeffector_p、robot_endeffector_q、actions 的最后一个元素拼成 8 维 dummy action (3 pose + 4 quat + 1 gripper)。"""
    # p = np.ravel(np.array(robot_endeffector_p[-1], dtype=np.float32)) if robot_endeffector_p else np.zeros(3, dtype=np.float32)
    # q = np.ravel(np.array(robot_endeffector_q[-1], dtype=np.float32)) if robot_endeffector_q else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # last_action = np.ravel(np.array(actions[-1], dtype=np.float32)) if actions else np.array([-1.0], dtype=np.float32)
    # gripper = float(last_action[-1]) if last_action.size > 0 else -1.0
    # return np.concatenate([p, q, np.array([gripper], dtype=np.float32)])
    return np.array([-6.0499899e-02, -2.8136521e-08,  5.2110010e-01, -7.3355800e-08,
  1.0000000e+00, -2.0861623e-07, -1.8728323e-09,  1.0000000e+00], dtype=np.float32)


def main():
    """
    主流程：按 episode 运行评测，不读取 h5 数据集。
    - 每个 episode 在 demonstration 结束后执行 dummy action。
    - 保留 obs/info 展开逻辑，用于调试和后续处理。
    """
    # ---------- 全局配置 ----------
    gui_render = False
    max_steps = 3000
    env_id_list = list(DEFAULT_ENV_IDS)
    print(f"Running envs: {env_id_list}")

    for env_id in env_id_list:
        # ---------- 按 env_id 创建配置解析器 ----------
        config_resolver = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="train",
            action_space="ee_pose",
            gui_render=gui_render,
        )

        for episode in range(50):
            # ---------- 为当前 episode 创建环境 ----------
            env, seed, difficulty = config_resolver.make_env_for_episode(episode)

            

            # ---------- 重置环境（含 demonstration 阶段） ----------
            obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.reset()

            # 从 obs_batch / info_batch 抽取并合并成列表
            maniskill_obs = (obs_batch or {}).get("maniskill_obs", [])
            base_camera = _flatten_column(obs_batch, "base_camera")
            wrist_camera = _flatten_column(obs_batch, "wrist_camera")
            base_camera_depth = _flatten_column(obs_batch, "base_camera_depth")
            base_camera_segmentation = _flatten_column(obs_batch, "base_camera_segmentation")
            wrist_camera_depth = _flatten_column(obs_batch, "wrist_camera_depth")
            base_camera_extrinsic_opencv = _flatten_column(obs_batch, "base_camera_extrinsic_opencv")
            base_camera_intrinsic_opencv = _flatten_column(obs_batch, "base_camera_intrinsic_opencv")
            base_camera_cam2world_opengl = _flatten_column(obs_batch, "base_camera_cam2world_opengl")
            wrist_camera_extrinsic_opencv = _flatten_column(obs_batch, "wrist_camera_extrinsic_opencv")
            wrist_camera_intrinsic_opencv = _flatten_column(obs_batch, "wrist_camera_intrinsic_opencv")
            wrist_camera_cam2world_opengl = _flatten_column(obs_batch, "wrist_camera_cam2world_opengl")
            robot_endeffector_p = _flatten_column(obs_batch, "robot_endeffector_p")
            robot_endeffector_q = _flatten_column(obs_batch, "robot_endeffector_q")
            actions = _flatten_column(obs_batch, "actions")
            states = _flatten_column(obs_batch, "states")
            velocity = _flatten_column(obs_batch, "velocity")
            language_goal_list = (obs_batch or {}).get("language_goal", [])
            language_goal = language_goal_list[0] if language_goal_list else None
            subgoal = _flatten_column(info_batch, "subgoal")
            subgoal_grounded = _flatten_column(info_batch, "subgoal_grounded")
            n = int(reward_batch.numel()) if hasattr(reward_batch, "numel") else 0
            info = _last_info(info_batch, n)
            terminated = bool(terminated_batch[-1].item()) if n > 0 else False
            truncated = bool(truncated_batch[-1].item()) if n > 0 else False

            # ---------- 按 step 运行：action 固定为 dummy ----------
            episode_success = False
            step = 0
            while step < max_steps:
                dummy_action = _get_dummy_action(robot_endeffector_p, robot_endeffector_q, actions)

                obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.step(dummy_action)
                print("dummy_action: ", dummy_action)
                
                # 从 batch 读取（供调试或后续逻辑）
                maniskill_obs = (obs_batch or {}).get("maniskill_obs", [])
                # 与 jointangle 回放一致：batch 中图像键为 "base_camera" / "wrist_camera"
                base_camera = _flatten_column(obs_batch, "base_camera")
                wrist_camera = _flatten_column(obs_batch, "wrist_camera")
                base_camera_depth = _flatten_column(obs_batch, "base_camera_depth")
                base_camera_segmentation = _flatten_column(obs_batch, "base_camera_segmentation")
                wrist_camera_depth = _flatten_column(obs_batch, "wrist_camera_depth")
                base_camera_extrinsic_opencv = _flatten_column(obs_batch, "base_camera_extrinsic_opencv")
                base_camera_intrinsic_opencv = _flatten_column(obs_batch, "base_camera_intrinsic_opencv")
                base_camera_cam2world_opengl = _flatten_column(obs_batch, "base_camera_cam2world_opengl")
                wrist_camera_extrinsic_opencv = _flatten_column(obs_batch, "wrist_camera_extrinsic_opencv")
                wrist_camera_intrinsic_opencv = _flatten_column(obs_batch, "wrist_camera_intrinsic_opencv")
                wrist_camera_cam2world_opengl = _flatten_column(obs_batch, "wrist_camera_cam2world_opengl")
                robot_endeffector_p = _flatten_column(obs_batch, "robot_endeffector_p")
                robot_endeffector_q = _flatten_column(obs_batch, "robot_endeffector_q")
                actions = _flatten_column(obs_batch, "actions")
                state = _flatten_column(obs_batch, "states")
                velocity = _flatten_column(obs_batch, "velocity")
                language_goal_list = (obs_batch or {}).get("language_goal", [])
            
                subgoal = _flatten_column(info_batch, "subgoal")
                subgoal_grounded = _flatten_column(info_batch, "subgoal_grounded")
                available_options = _flatten_column(info_batch, "available_options")
                n = int(reward_batch.numel()) if hasattr(reward_batch, "numel") else 0
                info = _last_info(info_batch, n)
                terminated = bool(terminated_batch[-1].item()) if n > 0 else False
                truncated = bool(truncated_batch[-1].item()) if n > 0 else False

                step += 1
                if gui_render:
                    env.render()
                if truncated:
                    print(f"[{env_id}] episode {episode} 步数超限，步 {step}。")
                    break
                if terminated:
                    if info.get("success") == torch.tensor([True]) or (
                        isinstance(info.get("success"), torch.Tensor) and info.get("success").item()
                    ):
                        print(f"[{env_id}] episode {episode} 成功。")
                        episode_success = True
                    elif info.get("fail", False):
                        print(f"[{env_id}] episode {episode} 失败。")
                    break

            env.close()


if __name__ == "__main__":
    main()
