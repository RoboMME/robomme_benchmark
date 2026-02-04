import os
import sys
import json
import numpy as np
from pathlib import Path

# Add parent directory and scripts to Python path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "scripts"))
import gymnasium as gym
from historybench.env_record_wrapper import (
    EpisodeConfigResolver,
    EpisodeDatasetResolver,
)
from historybench.HistoryBench_env import *
from save_reset_video import save_listStep_video

import torch

OUTPUT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/data/hongzefu/dataset_generate")


def read_metadata(metadata_path):
    """
    从 metadata JSON 文件读取所有 episode 配置

    Args:
        metadata_path: metadata JSON 文件路径

    Returns:
        list: 包含所有 episode 记录的列表，每个记录包含 task、episode、seed、difficulty
    """
    if not Path(metadata_path).exists():
        print(f"Metadata file not found: {metadata_path}")
        return []

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        episode_records = metadata.get('records', [])
        return episode_records


def main():
    """
    Main function to run the simulation using keypoints from dataset.
    Uses EpisodeDatasetResolver.get_keypoint(step) per episode; mirrors evaluate_endeffector-replayv2 structure.
    """

    env_id_list = [
        "SwingXtimes",
    ]

    gui_render = True
    render_mode = "human" if gui_render else "rgb_array"
    max_steps_without_demonstration = 200

    for env_id in env_id_list:
        metadata_path = DATASET_ROOT / f"record_dataset_{env_id}_metadata.json"
        h5_path = DATASET_ROOT / f"record_dataset_{env_id}.h5"

        episode_records = read_metadata(metadata_path)
        if not episode_records:
            print(f"No episode records found for {env_id}; skipping")
            continue

        config_resolver = EpisodeConfigResolver(
            env_id=env_id,
            metadata_path=str(metadata_path),
            render_mode=render_mode,
            gui_render=gui_render,
            max_steps_without_demonstration=max_steps_without_demonstration,
            action_space="keypoint",
        )

        for episode_record in episode_records:
            episode = episode_record["episode"]
            if episode != 0:
                continue
            seed = episode_record.get("seed")
            difficulty = episode_record.get("difficulty")

            print(f"--- Running simulation for episode:{episode}, env: {env_id}, seed: {seed}, difficulty: {difficulty} ---")

            env, resolved_seed, resolved_difficulty = config_resolver.make_env_for_episode(episode)
            dataset_resolver = EpisodeDatasetResolver(
                env_id=env_id,
                episode=episode,
                dataset_path=h5_path,
            )
            obs_list, reward_list, terminated_list, truncated_list, info_list = env.reset()

              # ---------- 从每个 obs 读取 frame 等，构建列表 ----------
            frames = []
            wrist_frames = []
            actions = []
            states = []
            velocity = []
            for o in obs_list:
                if o:
                    frames.extend(o.get("frames", []))
                    wrist_frames.extend(o.get("wrist_frames", []))
                    actions.extend(o.get("actions", []))
                    states.extend(o.get("states", []))
                    velocity.extend(o.get("velocity", []))
            language_goal = obs_list[0].get("language_goal") if obs_list and obs_list[0] else None

            # ---------- 从每个 info 读取子目标等 ----------
            subgoal = []
            subgoal_grounded = []
            for i in info_list:
                if i:
                    subgoal.extend(i.get("subgoal", []))
                    subgoal_grounded.extend(i.get("subgoal_grounded", []))

            # 用 reset 后的 frames 和 subgoal_grounded 直接保存为带字幕视频
            out_video_dir = DATASET_ROOT / "videos"
            os.makedirs(out_video_dir, exist_ok=True)
            reset_captioned_path = os.path.join(out_video_dir, f"replay_ee_{env_id}_ep{episode}_reset_captioned.mp4")
            if save_listStep_video(
                obs_list, reward_list, terminated_list, truncated_list, info_list, reset_captioned_path
            ):
                print(f"Saved reset captioned video: {reset_captioned_path}")





            video_dir = DATASET_ROOT / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            out_video_path = video_dir / f"replay_kp_{env_id}_ep{episode}.mp4"
            fps = 20

            step = 0
            while True:
                action = dataset_resolver.get_keypoint(step)
                if action is None:
                    break

                print(f"  Executing keypoint {step+1}: keypoint_p: {action[:3]}")
                action = action.astype(np.float32)

                obs_list, reward_list, terminated_list, truncated_list, info_list = env.step(action)

                # 从每个 obs 读取 frame 等，构建列表
                frames = []
                wrist_frames = []
                actions = []
                states = []
                velocity = []
                for o in obs_list:
                    if o:
                        frames.extend(o.get('frames', []))
                        wrist_frames.extend(o.get('wrist_frames', []))
                        actions.extend(o.get('actions', []))
                        states.extend(o.get('states', []))
                        velocity.extend(o.get('velocity', []))
                language_goal = obs_list[0].get('language_goal') if obs_list and obs_list[0] else None

                # 从每个 info 读取
                subgoal = []
                subgoal_grounded = []
                for i in info_list:
                    if i:
                        subgoal.extend(i.get('subgoal', []))
                        subgoal_grounded.extend(i.get('subgoal_grounded', []))

                # 用最后一步的 terminated/truncated/info 做循环判断
                terminated = terminated_list[-1] if terminated_list else False
                truncated = truncated_list[-1] if truncated_list else False
                info = info_list[-1] if info_list else {}

                # Save captioned video for this step (每个 step 保存一个带字幕的视频)
                kp_captioned_path = video_dir / f"replay_kp_{env_id}_ep{episode}_kp{step}_captioned.mp4"
                if save_listStep_video(
                    obs_list, reward_list, terminated_list, truncated_list, info_list, str(kp_captioned_path), fps=fps
                ):
                    print(f"Saved keypoint step video: {kp_captioned_path}")



                if gui_render:
                    env.render()
                step += 1

                if truncated:
                    print(f"[{env_id}] episode {episode} 步数超限。")
                    break
                if terminated.any():
                    if info.get("success") == torch.tensor([True]) or (
                        isinstance(info.get("success"), torch.Tensor) and info.get("success").item()
                    ):
                        print(f"[{env_id}] episode {episode} 成功。")
                    elif info.get("fail", False):
                        print(f"[{env_id}] episode {episode} 失败。")
                    break

            env.save_video(str(out_video_path))
            print(f"Saved video: {out_video_path}")

            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
            print(f"Final evaluation for episode {episode}: {evaluation}")

            dataset_resolver.close()
            env.close()
            print(f"--- Finished Running simulation for episode:{episode}, env: {env_id} ---")


if __name__ == "__main__":
    main()
