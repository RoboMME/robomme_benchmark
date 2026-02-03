# 从 dataset_generate 读取数据集并回放轨迹（跳过 demonstration 阶段，只执行非演示动作）

import os
import sys
from pathlib import Path
from typing import List, Optional

# 将包根目录加入 path，便于作为脚本直接运行
_ROOT = os.path.abspath(os.path.dirname(__file__))
_PARENT = os.path.dirname(_ROOT)
for _path in (_PARENT, _ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import gymnasium as gym
import torch
from PIL import Image

from historybench.HistoryBench_env import *
from historybench.HistoryBench_env.util import *
from historybench.env_record_wrapper import (
    EpisodeConfigResolver,
    EpisodeDatasetResolver,
    DemonstrationWrapper,
)

# 数据集根目录（包含 record_dataset_*_metadata.json 与 record_dataset_*.h5）
DATASET_ROOT = "/data/hongzefu/dataset_generate"





def main():
    """
    从 DATASET_ROOT 读取数据集，回放所有 episode；
    每个 episode 仅回放非 demonstration 的动作（跳过 demonstration 步）。
    """
    gui_render = False
    max_steps = 3000

    render_mode = "human" if gui_render else "rgb_array"

    env_id_list =["VideoPlaceOrder"]

    for env_id in env_id_list:
        metadata_path = f"{DATASET_ROOT}/record_dataset_{env_id}_metadata.json"

        config_resolver = EpisodeConfigResolver(
            env_id=env_id,
            metadata_path=metadata_path,
            render_mode=render_mode,
            gui_render=gui_render,
            max_steps_without_demonstration=max_steps,
        )

      

        h5_path = f"{DATASET_ROOT}/record_dataset_{env_id}.h5"
        out_video_dir = os.path.join(DATASET_ROOT, "videos")
        for episode in range(10):
            env, seed, difficulty = config_resolver.make_env_for_episode(episode)
            env.save_failed_match_env_id = env_id
            env.save_failed_match_episode = episode
            dataset_resolver = EpisodeDatasetResolver(
                env_id=env_id,
                episode=episode,
                dataset_path=h5_path,
            )
            obs, info = env.reset()

            # 从 obs 读取
            frames = obs.get('frames', []) if obs else []
            wrist_frames = obs.get('wrist_frames', []) if obs else []
            actions = obs.get('actions', []) if obs else []
            states = obs.get('states', []) if obs else []
            velocity = obs.get('velocity', []) if obs else []
            language_goal = obs.get('language_goal') if obs else None

            # 从 info 读取
            subgoal = info.get('subgoal_history', []) if info else []
            subgoal_grounded = info.get('subgoal_grounded_history', []) if info else []


            step = 0
            while True:
                action = dataset_resolver.get_action(step)
                obs, reward, terminated, truncated, info = env.step(action)

                # 从 obs 读取
                image = obs.get('frames', [])[-1] if obs.get('frames') else None
                wrist_image = obs.get('wrist_frames', [])[-1] if obs.get('wrist_frames') else None
                last_action = obs.get('actions', [])[-1] if obs.get('actions') else None
                state = obs.get('states', [])[-1] if obs.get('states') else None
                velocity = obs.get('velocity', [])[-1] if obs.get('velocity') else None
                language_goal = obs.get('language_goal') if obs else None
                # 从 info 读取
                subgoal = info.get('subgoal_history', []) if info else []
                subgoal_grounded = info.get('subgoal_grounded_history', []) if info else []

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


            
            out_video_path = os.path.join(DATASET_ROOT, "videos", f"replay_{env_id}_ep{episode}.mp4")
            env.save_video(out_video_path)
            print(f"Saved video: {out_video_path}")
            dataset_resolver.close()
            env.close()

if __name__ == "__main__":
    main()
