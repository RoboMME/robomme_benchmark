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
# ACTION_SPACE = "ee_pose"
# ACTION_SPACE = "keypoint"
# ACTION_SPACE = "oracle_planner"

GUI_RENDER = True
MAX_STEPS = 3000
EPISODE_COUNT = 50

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


def _extract_obs_union(obs_batch):
    return {key: _flatten_column(obs_batch, key) for key in OBS_SUPER_KEYS}


def _extract_info_union(info_batch):
    return {key: _flatten_column(info_batch, key) for key in INFO_SUPER_KEYS}


def _last_info(info_batch, n):
    if n <= 0:
        return {}
    idx = n - 1
    return {
        k: v[idx]
        for k, v in (info_batch or {}).items()
        if len(v) > idx and v[idx] is not None
    }


def _get_dummy_action(action_space, obs_union, info_union):
    # 当前需求为固定写死 dummy action；保留 obs/info 参数用于统一接口。
    _ = obs_union
    _ = info_union

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
    env_id_list = list(DEFAULT_ENV_IDS)
    print(f"Running envs: {env_id_list}")
    print(f"Using action_space: {ACTION_SPACE}")

    for env_id in env_id_list:
        config_resolver = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="train",
            action_space=ACTION_SPACE,
            gui_render=GUI_RENDER,
        )

        for episode in range(EPISODE_COUNT):
            env, seed, difficulty = config_resolver.make_env_for_episode(episode)
            obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.reset()

            # reset 后统一做 obs/info 固定超集抽取
            obs_union = _extract_obs_union(obs_batch)
            info_union = _extract_info_union(info_batch)

            # 保持四个原评测脚本中的调试变量语义
            maniskill_obs = obs_union["maniskill_obs"]
            base_camera = obs_union["base_camera"]
            wrist_camera = obs_union["wrist_camera"]
            base_camera_depth = obs_union["base_camera_depth"]
            base_camera_segmentation = obs_union["base_camera_segmentation"]
            wrist_camera_depth = obs_union["wrist_camera_depth"]
            base_camera_extrinsic_opencv = obs_union["base_camera_extrinsic_opencv"]
            base_camera_intrinsic_opencv = obs_union["base_camera_intrinsic_opencv"]
            base_camera_cam2world_opengl = obs_union["base_camera_cam2world_opengl"]
            wrist_camera_extrinsic_opencv = obs_union["wrist_camera_extrinsic_opencv"]
            wrist_camera_intrinsic_opencv = obs_union["wrist_camera_intrinsic_opencv"]
            wrist_camera_cam2world_opengl = obs_union["wrist_camera_cam2world_opengl"]
            robot_endeffector_p = obs_union["robot_endeffector_p"]
            robot_endeffector_q = obs_union["robot_endeffector_q"]
            actions = obs_union["actions"]
            states = obs_union["states"]
            velocity = obs_union["velocity"]
            language_goal_list = obs_union["language_goal"]
            language_goal = language_goal_list[0] if language_goal_list else None

            subgoal = info_union["subgoal"]
            subgoal_grounded = info_union["subgoal_grounded"]
            available_options = info_union["available_options"]

            n = int(reward_batch.numel()) if hasattr(reward_batch, "numel") else 0
            info = _last_info(info_batch, n)
            terminated = bool(terminated_batch[-1].item()) if n > 0 else False
            truncated = bool(truncated_batch[-1].item()) if n > 0 else False

            episode_success = False
            step = 0
            while step < MAX_STEPS:
                dummy_action = _get_dummy_action(ACTION_SPACE, obs_union, info_union)
                obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.step(dummy_action)
                print("dummy_action: ", dummy_action)

                # step 后统一做 obs/info 固定超集抽取
                obs_union = _extract_obs_union(obs_batch)
                info_union = _extract_info_union(info_batch)

                # 保持四个原评测脚本中的调试变量语义
                maniskill_obs = obs_union["maniskill_obs"]
                base_camera = obs_union["base_camera"]
                wrist_camera = obs_union["wrist_camera"]
                base_camera_depth = obs_union["base_camera_depth"]
                base_camera_segmentation = obs_union["base_camera_segmentation"]
                wrist_camera_depth = obs_union["wrist_camera_depth"]
                base_camera_extrinsic_opencv = obs_union["base_camera_extrinsic_opencv"]
                base_camera_intrinsic_opencv = obs_union["base_camera_intrinsic_opencv"]
                base_camera_cam2world_opengl = obs_union["base_camera_cam2world_opengl"]
                wrist_camera_extrinsic_opencv = obs_union["wrist_camera_extrinsic_opencv"]
                wrist_camera_intrinsic_opencv = obs_union["wrist_camera_intrinsic_opencv"]
                wrist_camera_cam2world_opengl = obs_union["wrist_camera_cam2world_opengl"]
                robot_endeffector_p = obs_union["robot_endeffector_p"]
                robot_endeffector_q = obs_union["robot_endeffector_q"]
                actions = obs_union["actions"]
                states = obs_union["states"]
                velocity = obs_union["velocity"]
                language_goal_list = obs_union["language_goal"]

                subgoal = info_union["subgoal"]
                subgoal_grounded = info_union["subgoal_grounded"]
                available_options = info_union["available_options"]

                n = int(reward_batch.numel()) if hasattr(reward_batch, "numel") else 0
                info = _last_info(info_batch, n)
                terminated = bool(terminated_batch[-1].item()) if n > 0 else False
                truncated = bool(truncated_batch[-1].item()) if n > 0 else False

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

            _ = (
                maniskill_obs,
                base_camera,
                wrist_camera,
                base_camera_depth,
                base_camera_segmentation,
                wrist_camera_depth,
                base_camera_extrinsic_opencv,
                base_camera_intrinsic_opencv,
                base_camera_cam2world_opengl,
                wrist_camera_extrinsic_opencv,
                wrist_camera_intrinsic_opencv,
                wrist_camera_cam2world_opengl,
                robot_endeffector_p,
                robot_endeffector_q,
                actions,
                states,
                velocity,
                language_goal,
                subgoal,
                subgoal_grounded,
                available_options,
                episode_success,
                seed,
                difficulty,
                terminated,
                truncated,
            )
            env.close()


if __name__ == "__main__":
    main()
