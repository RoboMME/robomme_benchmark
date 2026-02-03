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
from PIL import Image

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
    # 是否开启 GUI 渲染（弹窗显示）；False 则仅 rgb_array 不弹窗
    gui_render = True
    # 单 episode 最大步数（不含 demonstration）
    max_steps = 3000

    # gymnasium 的 render_mode：human 弹窗，rgb_array 仅返回数组
    render_mode = "human" if gui_render else "rgb_array"

    # 要评估的环境 ID 列表，可扩展多个
    env_id_list = ["VideoRepick"]

    for env_id in env_id_list:
        # 该环境对应的元数据路径（episode 配置、种子、难度等）
        metadata_path = f"{DATASET_ROOT}/record_dataset_{env_id}_metadata.json"

        # 根据元数据创建环境的解析器，用于按 episode 生成 env（action_space=ee_pose 时 wrapper 内做 IK）
        config_resolver = EpisodeConfigResolver(
            env_id=env_id,
            metadata_path=metadata_path,
            render_mode=render_mode,
            gui_render=gui_render,
            max_steps_without_demonstration=max_steps,
            action_space="ee_pose",
        )

        # 该环境对应的 h5 数据集路径（动作、末端位姿等按 step 存储）
        h5_path = f"{DATASET_ROOT}/record_dataset_{env_id}.h5"
        # 对前 10 个 episode 分别回放
        for episode in range(10):
            # 为当前 episode 创建环境（含正确 seed、难度等）
            env, seed, difficulty = config_resolver.make_env_for_episode(episode)

            # 从 h5 中按 step 读取当前 episode 的动作与末端位姿
            dataset_resolver = EpisodeDatasetResolver(
                env_id=env_id,
                episode=episode,
                dataset_path=h5_path,
            )
            # 重置环境，得到 demonstration 结束后的初始 obs / info
            obs, info = env.reset()

            # ---------- 从 obs 读取（reset 后为 demonstration 阶段最后一帧的数据）----------
            frames = obs.get("frames", []) if obs else []
            wrist_frames = obs.get("wrist_frames", []) if obs else []
            actions = obs.get("actions", []) if obs else []
            states = obs.get("states", []) if obs else []
            velocity = obs.get("velocity", []) if obs else []
            language_goal = obs.get("language_goal") if obs else None

            # ---------- 从 info 读取子目标等 ----------
            subgoal = info.get("subgoal_history", []) if info else []
            subgoal_grounded = info.get("subgoal_grounded_history", []) if info else []


            # ---------- 按 step 回放：action = [eep, eeq, gripper]，wrapper 内做 IK ----------
            step = 0
            while True:
                # 从数据集中取当前步的原始动作（取 gripper 最后一维）
                action_original = dataset_resolver.get_action(step)
                # 从数据集中取当前步记录的末端执行器位姿：位置 p、四元数 q（世界系）
                ee_p, ee_q = dataset_resolver.get_ee_pose(step)

                # 若该步没有末端位姿（例如已到 episode 末尾），结束回放
                if ee_p is None or ee_q is None:
                    break

                gripper = float(action_original[-1]) if action_original is not None else -1.0
                action = np.concatenate([
                    np.asarray(ee_p).flatten(),
                    np.asarray(ee_q).flatten(),
                    [gripper],
                ])
                obs, reward, terminated, truncated, info = env.step(action)

                # ---------- step 之后从 obs/info 更新（供后续逻辑或调试使用）----------
                image = obs.get("frames", [])[-1] if obs.get("frames") else None
                wrist_image = obs.get("wrist_frames", [])[-1] if obs.get("wrist_frames") else None
                last_action = obs.get("actions", [])[-1] if obs.get("actions") else None
                state = obs.get("states", [])[-1] if obs.get("states") else None
                velocity = obs.get("velocity", [])[-1] if obs.get("velocity") else None
                language_goal = obs.get("language_goal") if obs else None
                subgoal = info.get("subgoal_history", []) if info else []
                subgoal_grounded = info.get("subgoal_grounded_history", []) if info else []

                step += 1
                if gui_render:
                    env.render()
                # 步数达到上限
                if truncated:
                    print(f"[{env_id}] episode {episode} 步数超限，步 {step}。")
                    break
                # 任务结束（成功或失败）
                if terminated.any():
                    if info.get("success") == torch.tensor([True]) or (
                        isinstance(info.get("success"), torch.Tensor) and info.get("success").item()
                    ):
                        print(f"[{env_id}] episode {episode} 成功。")
                    elif info.get("fail", False):
                        print(f"[{env_id}] episode {episode} 失败。")
                    break

            # 保存回放视频（frames + subgoal_grounded）
            out_video_path = os.path.join(DATASET_ROOT, "videos", f"replay_ee_{env_id}_ep{episode}.mp4")
            env.save_video(out_video_path)
            print(f"Saved video: {out_video_path}")

            # 当前 episode 结束，关闭数据集句柄和环境
            dataset_resolver.close()
            env.close()


if __name__ == "__main__":
    main()
