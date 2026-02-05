# -*- coding: utf-8 -*-
# 脚本功能：从 dataset_generate 读取数据集并回放轨迹
# 特点：跳过 demonstration 阶段，仅执行非演示动作；使用末端执行器位姿 + IK 生成关节动作进行回放

import os
import sys
from pathlib import Path
from typing import List, Optional

# 将包根目录及上级目录加入 sys.path，便于作为脚本直接运行（不依赖 PYTHONPATH）
_ROOT = os.path.abspath(os.path.dirname(__file__))
_PARENT = os.path.dirname(_ROOT)
for _path in (_PARENT, _ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import gymnasium as gym
import torch

from save_reset_video import save_listStep_video

# HistoryBench 环境及工具
from historybench.HistoryBench_env import *
from historybench.HistoryBench_env.util import *
# 用于根据元数据创建环境、从 h5 读取 episode 动作/位姿
from historybench.env_record_wrapper import (
    EpisodeConfigResolver,
    EpisodeDatasetResolver,
)

# 数据集根目录：需包含 record_dataset_{env_id}_metadata.json 与 record_dataset_{env_id}.h5
DATASET_ROOT = "/data/hongzefu/dataset_generate"


def main():
    """
    主流程：从 DATASET_ROOT 读取数据集，按 episode 回放。
    - 每个 episode 仅回放非 demonstration 步（跳过演示阶段）。
    - 使用数据集中记录的末端执行器位姿 (ee pose)，经 IK 得到关节角后再执行，以验证末端轨迹回放效果。
    """
    # ---------- 全局配置 ----------
    gui_render = False
    max_steps = 3000
    render_mode = "human" if gui_render else "rgb_array"
    env_id_list = ["VideoRepick"]

    for env_id in env_id_list:
        # ---------- 按 env_id 创建配置解析器与数据集路径 ----------
        metadata_path = f"{DATASET_ROOT}/record_dataset_{env_id}_metadata.json"
        config_resolver = EpisodeConfigResolver(
            env_id=env_id,
            metadata_path=metadata_path,
            render_mode=render_mode,
            gui_render=gui_render,
            max_steps_without_demonstration=max_steps,
            action_space="ee_pose",
        )
        h5_path = f"{DATASET_ROOT}/record_dataset_{env_id}.h5"
        out_video_dir = os.path.join(DATASET_ROOT, "videos")

        for episode in range(10):
            # ---------- 为当前 episode 创建环境与数据集解析器 ----------
            env, seed, difficulty = config_resolver.make_env_for_episode(episode)
            env.save_failed_match_env_id = env_id
            env.save_failed_match_episode = episode
            dataset_resolver = EpisodeDatasetResolver(
                env_id=env_id,
                episode=episode,
                dataset_path=h5_path,
            )

            # ---------- 重置环境（含 demonstration 阶段），并保存 reset 带字幕视频 ----------
            obs_list, reward_list, terminated_list, truncated_list, info_list = env.reset()

            # 从 obs_list / info_list 抽取并合并成新列表
            frames = []
            wrist_frames = []
            actions = []
            states = []
            velocity = []
            for o in obs_list:
                obs_root = o or {}
                _ms_obs = obs_root.get("maniskill_obs", {})
                frames.extend(obs_root.get("frames", []))
                wrist_frames.extend(obs_root.get("wrist_frames", []))
                actions.extend(obs_root.get("actions", []))
                states.extend(obs_root.get("states", []))
                velocity.extend(obs_root.get("velocity", []))
            first_obs = obs_list[0] if obs_list else {}
            language_goal = (first_obs or {}).get("language_goal")
            subgoal = []
            subgoal_grounded = []
            for i in info_list:
                if i:
                    subgoal.extend(i.get("subgoal", []))
                    subgoal_grounded.extend(i.get("subgoal_grounded", []))

            reset_captioned_path = os.path.join(out_video_dir, f"replay_ee_{env_id}_ep{episode}_reset_captioned.mp4")
            save_listStep_video(obs_list, reward_list, terminated_list, truncated_list, info_list, reset_captioned_path)

            # ---------- 按 step 回放：action = [eep, eeq, gripper]，wrapper 内做 IK ----------
            step = 0
            while True:
                action = dataset_resolver.get_ee_pose_gripper(step)
                if action is None:
                    break
                obs, reward, terminated, truncated, info = env.step(action)

                # 从 obs / info 读取（供调试或后续逻辑）
                obs_root = obs or {}
                _ms_obs = obs_root.get("maniskill_obs", {})
                image = obs_root.get("frames", []) if obs_root.get("frames") else None
                wrist_image = obs_root.get("wrist_frames", []) if obs_root.get("wrist_frames") else None
                last_action = obs_root.get("actions", []) if obs_root.get("actions") else None
                state = obs_root.get("states", []) if obs_root.get("states") else None
                velocity = obs_root.get("velocity", []) if obs_root.get("velocity") else None
                language_goal = obs_root.get("language_goal")
                subgoal = info.get("subgoal", []) if info else []
                print(subgoal)
                subgoal_grounded = info.get("subgoal_grounded", []) if info else []

                step += 1
                if gui_render:
                    env.render()
                if truncated:
                    print(f"[{env_id}] episode {episode} 步数超限，步 {step}。")
                    break
                if terminated.any():
                    if info.get("success") == torch.tensor([True]) or (
                        isinstance(info.get("success"), torch.Tensor) and info.get("success").item()
                    ):
                        print(f"[{env_id}] episode {episode} 成功。")
                    elif info.get("fail", False):
                        print(f"[{env_id}] episode {episode} 失败。")
                    break

            # ---------- 保存本 episode 回放视频并关闭资源 ----------
            os.makedirs(out_video_dir, exist_ok=True)
            out_video_path = os.path.join(out_video_dir, f"replay_ee_{env_id}_ep{episode}.mp4")
            env.save_video(out_video_path)
            print(f"Saved video: {out_video_path}")
            dataset_resolver.close()
            env.close()


if __name__ == "__main__":
    main()
