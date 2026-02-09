# -*- coding: utf-8 -*-
# 脚本功能：按 episode 运行 keypoint 评测（不读取 h5 数据集）
# 特点：跳过 demonstration 阶段后执行 dummy keypoint（robot_endeffector_p/q + actions 最后一维），与 evaluate_endeffector 一致

import os
import sys

# 将包根目录、上级目录及 scripts 加入 sys.path（MultiStepDemonstrationWrapper 需从 scripts 导入 planner_fail_safe）
_ROOT = os.path.abspath(os.path.dirname(__file__))
_PARENT = os.path.dirname(_ROOT)
_SCRIPTS = os.path.join(_PARENT, "scripts")
for _path in (_PARENT, _ROOT, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import torch

# HistoryBench 环境及工具
from historybench.HistoryBench_env import *
# 用于根据元数据创建环境
from historybench.env_record_wrapper import (
    EpisodeConfigResolver,
)

# 数据集根目录：需包含 record_dataset_{env_id}_metadata.json
DATASET_ROOT = os.path.join(_PARENT, "dataset_json")
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

# 定死每步 step 使用的 8 维 action（keypoint_p 3 + keypoint_q 4 + gripper 1）
DUMMY_ACTION = np.array(
  [-0.120827354, 0.17769682, 0.15, 0.0, 0.972572, 0.23260213, 0.0, 1.0],
    dtype=np.float32,
)


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





def main():
    """
    主流程：按 episode 运行 keypoint 评测，不读取 h5 数据集。
    - 每个 episode 在 demonstration 结束后执行 dummy keypoint（与 evaluate_endeffector 同构）。
    - 保留 obs/info 展开逻辑，用于调试和后续处理。
    """
    # ---------- 全局配置 ----------
    gui_render = True
    max_steps = 3000
    render_mode = "human" if gui_render else "rgb_array"
    env_id_list = list(DEFAULT_ENV_IDS)
    print(f"Running envs: {env_id_list}")

    for env_id in env_id_list:
        # ---------- 按 env_id 创建配置解析器 ----------
        metadata_path = os.path.join(DATASET_ROOT, f"record_dataset_{env_id}_metadata.json")
        config_resolver = EpisodeConfigResolver(
            env_id=env_id,
            metadata_path=metadata_path,
            render_mode=render_mode,
            gui_render=gui_render,
            max_steps_without_demonstration=max_steps,
            action_space="keypoint",
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
            available_options = _flatten_column(info_batch, "available_options")

            n = int(reward_batch.numel()) if hasattr(reward_batch, "numel") else 0  # type: ignore[union-attr]
            info = _last_info(info_batch, n)
            terminated = bool(terminated_batch[-1].item()) if n > 0 else False  # type: ignore[union-attr]
            truncated = bool(truncated_batch[-1].item()) if n > 0 else False  # type: ignore[union-attr]

            # ---------- 按 step 运行：action 定死为 DUMMY_ACTION ----------
            episode_success = False
            step = 0
            while step < max_steps:
                obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.step(DUMMY_ACTION)

                # 从 batch 读取（供调试或后续逻辑）
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

                subgoal = _flatten_column(info_batch, "subgoal")
                subgoal_grounded = _flatten_column(info_batch, "subgoal_grounded")
                available_options = _flatten_column(info_batch, "available_options")

                n = int(reward_batch.numel()) if hasattr(reward_batch, "numel") else 0  # type: ignore[union-attr]
                info = _last_info(info_batch, n)
                terminated = bool(terminated_batch[-1].item()) if n > 0 else False  # type: ignore[union-attr]
                truncated = bool(truncated_batch[-1].item()) if n > 0 else False  # type: ignore[union-attr]

                step += 1
                if gui_render:
                    env.render()
                if truncated:
                    print(f"[{env_id}] episode {episode} 步数超限，步 {step}。")
                    break
                if terminated:
                    succ = info.get("success")
                    if succ == torch.tensor([True]) or (
                        isinstance(succ, torch.Tensor) and succ is not None and succ.item()
                    ):
                        print(f"[{env_id}] episode {episode} 成功。")
                        episode_success = True
                    elif info.get("fail", False):
                        print(f"[{env_id}] episode {episode} 失败。")
                    break

            env.close()


if __name__ == "__main__":
    main()
