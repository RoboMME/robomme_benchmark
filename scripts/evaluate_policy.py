import os
import sys

# Ensure the package root is importable when running as a script
_ROOT = os.path.abspath(os.path.dirname(__file__))
_PARENT = os.path.dirname(_ROOT)
for _path in (_PARENT, _ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import sapien
import gymnasium as gym
from gymnasium.utils.save_video import save_video

from historybench.env_record_wrapper import DemonstrationWrapper
from historybench.HistoryBench_env import *
# from util import *
import torch
import imageio

from historybench.HistoryBench_env.util import reset_panda

def main():
    """
    Main function to run the simulation and record data for multiple seeds.
    """

    num_episodes = 1
    gui_render=True
    max_steps=500
    test_seed_offset=500
    

    env_id_list=[
    'BinFill'
    ]

    if gui_render==True:
        render_mode="human"
    else:
        render_mode="rgb_array"

    for env_id in env_id_list:
        for episode in range(num_episodes):
            seed=episode+test_seed_offset

            print(f"--- Running simulation for episode:{episode},env: {env_id}, seed:{seed} ---")


            env= gym.make(
                env_id,
                obs_mode="rgb",
                control_mode="pd_joint_pos",
                render_mode=render_mode,
                reward_mode="dense",
                HistoryBench_seed = seed,
                max_episode_steps=99999
            )
            env = DemonstrationWrapper(env, max_steps_without_demonstration=max_steps,gui_render=gui_render)
            env.reset()

            # Demonstration data is now automatically generated in reset()
            demonstration_data = env.demonstration_data

            #Save demonstration videos
            if len(demonstration_data['frames']) > 0:
                video_dir = os.path.join(_ROOT, "demonstration_videos")
                os.makedirs(video_dir, exist_ok=True)

                base_video_path = os.path.join(video_dir, f"evaluate:{env_id}_seed{seed}_base_camera.mp4")
                wrist_video_path = os.path.join(video_dir, f"evaluate:{env_id}_seed{seed}_wrist_camera.mp4")

                imageio.mimsave(base_video_path, demonstration_data['frames'], fps=30)
                imageio.mimsave(wrist_video_path, demonstration_data['wrist_frames'], fps=30)

                print(f"Saved demonstration videos to {base_video_path} and {wrist_video_path}")
                print(f"Recorded {len(demonstration_data['actions'])} actions and {len(demonstration_data['states'])} states")

            # Initialize trajectory data dictionary
            trajectory_data = {
                'frames': [],
                'wrist_frames': [],
                'actions': [],
                'states': []
            }
            while True:
                action=None
                obs, reward, terminated, truncated, info = env.step(action)

                image = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
                wrist_image = obs['sensor_data']['hand_camera']['rgb'][0].cpu().numpy()
                state = env.agent.robot.qpos.cpu().numpy() if hasattr(env.agent.robot.qpos, 'cpu') else env.agent.robot.qpos
                trajectory_data['frames'].append(image)
                trajectory_data['wrist_frames'].append(wrist_image)
                trajectory_data['actions'].append(action)
                trajectory_data['states'].append(state)

                if gui_render:
                    env.render()

                if truncated:
                    print("time limit!")
                    break
                elif terminated:
                    if info.get("success", False):
                        print("success")
                    if info.get("fail", False):
                        print("fail")
                    break

           

            env.close()
            print(f"--- Finished Running simulation for episode:{episode},env: {env_id} ---")


if __name__ == "__main__":
    main()
