# -*- coding: utf-8 -*-
# 脚本功能：统一评测入口，支持 joint_angle / ee_pose / keypoint / oracle_planner 四种 action_space。

import os
import sys

# 将包根目录、上级目录及 scripts 加入 sys.path，便于作为脚本直接运行（不依赖 PYTHONPATH）
_ROOT = os.path.abspath(os.path.dirname(__file__))
_PARENT = os.path.dirname(_ROOT)
_SCRIPTS = os.path.join(_PARENT, "scripts")
for _path in (_PARENT, _ROOT, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import torch

# HistoryBench 环境及工具（保持与现有评测脚本一致的导入方式）
from historybench.HistoryBench_env import *
from historybench.HistoryBench_env.util import *
from historybench.env_record_wrapper import (
    BenchmarkEnvBuilder,
)

# 只启用一个 ACTION_SPACE；其他选项保留在注释中供手动切换
ACTION_SPACE = "joint_angle"
#ACTION_SPACE = "ee_pose"
#ACTION_SPACE = "keypoint"
#ACTION_SPACE = "oracle_planner"

GUI_RENDER = True
MAX_STEPS = 3000

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

OBS_SUPER_KEYS = [
    "maniskill_obs",
    "base_camera",
    "wrist_camera",
    "base_camera_depth",
    "base_camera_segmentation",
    "wrist_camera_depth",
    "base_camera_extrinsic_opencv",
    "base_camera_intrinsic_opencv",
    "base_camera_cam2world_opengl",
    "wrist_camera_extrinsic_opencv",
    "wrist_camera_intrinsic_opencv",
    "wrist_camera_cam2world_opengl",
    "robot_endeffector_p",
    "robot_endeffector_q",
    "actions",
    "states",
    "velocity",
    "language_goal",
]

INFO_SUPER_KEYS = [
    "subgoal",
    "subgoal_grounded",
    "available_options",
]






def _get_dummy_action(action_space):
    
    if action_space == "joint_angle":
        return np.array(
            [0.0, 0.0, 0.0, -1.5707964, 0.0, 1.5707964, 0.7853982, 1.0],
            dtype=np.float32,
        )
    if action_space == "ee_pose":
        return np.array(
            [
                -6.0499899e-02,
                -2.8136521e-08,
                5.2110010e-01,
                -7.3355800e-08,
                1.0000000e00,
                -2.0861623e-07,
                -1.8728323e-09,
                1.0000000e00,
            ],
            dtype=np.float32,
        )
    if action_space == "keypoint":
        return np.array(
            [-0.120827354, 0.17769682, 0.15, 0.0, 0.972572, 0.23260213, 0.0, 1.0],
            dtype=np.float32,
        )
    if action_space == "oracle_planner":
        return {
            "action": "pick up the cube",
            "point": [0, 0],
        }
    raise ValueError(f"Unsupported ACTION_SPACE: {action_space}")


def main():
    env_id_list = BenchmarkEnvBuilder.get_task_list()
    print(f"Running envs: {env_id_list}")
    print(f"Using action_space: {ACTION_SPACE}")

    for env_id in env_id_list:
        env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="train",
            action_space=ACTION_SPACE,
            gui_render=GUI_RENDER,
        )
        episode_count = env_builder.get_episode_num()
        print(f"[{env_id}] episode_count from metadata: {episode_count}")

        for episode in range(episode_count):
            env, seed, difficulty = env_builder.make_env_for_episode(episode)
            obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.reset()

            # 保持四个原评测脚本中的调试变量语义
            maniskill_obs = obs_batch["maniskill_obs"]
            base_camera = obs_batch["base_camera"]
            wrist_camera = obs_batch["wrist_camera"]
            base_camera_depth = obs_batch["base_camera_depth"]
            base_camera_segmentation = obs_batch["base_camera_segmentation"]
            wrist_camera_depth = obs_batch["wrist_camera_depth"]
            base_camera_extrinsic_opencv = obs_batch["base_camera_extrinsic_opencv"]
            base_camera_intrinsic_opencv = obs_batch["base_camera_intrinsic_opencv"]
            base_camera_cam2world_opengl = obs_batch["base_camera_cam2world_opengl"]
            wrist_camera_extrinsic_opencv = obs_batch["wrist_camera_extrinsic_opencv"]
            wrist_camera_intrinsic_opencv = obs_batch["wrist_camera_intrinsic_opencv"]
            wrist_camera_cam2world_opengl = obs_batch["wrist_camera_cam2world_opengl"]
            robot_endeffector_p = obs_batch["robot_endeffector_p"]
            robot_endeffector_q = obs_batch["robot_endeffector_q"]
            actions = obs_batch["actions"]
            states = obs_batch["states"]
            velocity = obs_batch["velocity"]
            language_goal_list = obs_batch["language_goal"]
            language_goal = language_goal_list[0] if language_goal_list else None

            subgoal = info_batch["subgoal"]
            subgoal_grounded = info_batch["subgoal_grounded"]
            available_options = info_batch["available_options"]

         
            info ={k: v[-1] for k, v in info_batch.items()}
            terminated = bool(terminated_batch[-1].item())
            truncated = bool(truncated_batch[-1].item())

            episode_success = False
            step = 0
            while step < MAX_STEPS:
                dummy_action = _get_dummy_action(ACTION_SPACE)
                obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.step(dummy_action)
                print("dummy_action: ", dummy_action)

                # 保持四个原评测脚本中的调试变量语义
                maniskill_obs = obs_batch["maniskill_obs"]
                base_camera = obs_batch["base_camera"]
                wrist_camera = obs_batch["wrist_camera"]
                base_camera_depth = obs_batch["base_camera_depth"]
                base_camera_segmentation = obs_batch["base_camera_segmentation"]
                wrist_camera_depth = obs_batch["wrist_camera_depth"]
                base_camera_extrinsic_opencv = obs_batch["base_camera_extrinsic_opencv"]
                base_camera_intrinsic_opencv = obs_batch["base_camera_intrinsic_opencv"]
                base_camera_cam2world_opengl = obs_batch["base_camera_cam2world_opengl"]
                wrist_camera_extrinsic_opencv = obs_batch["wrist_camera_extrinsic_opencv"]
                wrist_camera_intrinsic_opencv = obs_batch["wrist_camera_intrinsic_opencv"]
                wrist_camera_cam2world_opengl = obs_batch["wrist_camera_cam2world_opengl"]
                robot_endeffector_p = obs_batch["robot_endeffector_p"]
                robot_endeffector_q = obs_batch["robot_endeffector_q"]
                actions = obs_batch["actions"]
                states = obs_batch["states"]
                velocity = obs_batch["velocity"]
                language_goal_list = obs_batch["language_goal"]

                subgoal = info_batch["subgoal"]
                subgoal_grounded = info_batch["subgoal_grounded"]
                available_options = info_batch["available_options"]

         
                info ={k: v[-1] for k, v in info_batch.items()}
                terminated = bool(terminated_batch[-1].item()) 
                truncated = bool(truncated_batch[-1].item()) 

                step += 1
                if GUI_RENDER:
                    env.render()
                if truncated:
                    print(f"[{env_id}] episode {episode} 步数超限，步 {step}。")
                    break
                if terminated:
                    succ = info.get("success")
                    if succ == torch.tensor([True]) or (
                        isinstance(succ, torch.Tensor) and succ.item()
                    ):
                        print(f"[{env_id}] episode {episode} 成功。")
                        episode_success = True
                    elif info.get("fail", False):
                        print(f"[{env_id}] episode {episode} 失败。")
                    break


            env.close()


if __name__ == "__main__":
    main()
