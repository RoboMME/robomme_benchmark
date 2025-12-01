import os
import sys
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

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


from historybench.HistoryBench_env import *
from historybench.HistoryBench_env.util import *
from historybench.env_record_wrapper import *
import torch
import imageio
import h5py

import re


def main():
    """
    Main function to run the simulation and record data for multiple seeds.
    """

    num_episodes = 1
    gui_render=False
    max_steps=3000
    


    env_id_list = [
        # "VideoRepick",
        # "BinFill",
        # "ButtonUnmask",
        # "ButtonUnmaskSwap",
        # "InsertPeg",
        # "MoveCube",
        "PatternLock",
        # "PickHighlight",
        # "PickXtimes",
        # "RouteStick",
        # "StopCube",
        # "SwingXtimes",
        # "VideoPlaceButton",
        # "VideoPlaceOrder",
        # "VideoUnmask",
        # "VideoUnmaskSwap",
    ]



    if gui_render==True:
        render_mode="human"
    else:
        render_mode="rgb_array"

    for env_id in env_id_list:
        dataset_path= f"/data/hongzefu/dataset_demonstration/record_dataset_{env_id}.h5"
        dataset = h5py.File(dataset_path, "r")
        metadata_path =  f"/data/hongzefu/dataset_demonstration/record_dataset_{env_id}_metadata.json"

        resolver = EpisodeConfigResolver(
            env_id=env_id,
            dataset=None,
            metadata_path=metadata_path,
            render_mode=render_mode,
            gui_render=gui_render,
            max_steps_without_demonstration=max_steps,
        )

        for episode in range(num_episodes):
            env, episode_dataset, seed, difficulty = resolver.make_env_for_episode(episode)
            import pdb ;pdb.set_trace()
           
            env.reset()

            # Demonstration data is now automatically generated in reset()
            demonstration_data = env.demonstration_data
            frames = demonstration_data.get('frames', []) if demonstration_data else []
            wrist_frames = demonstration_data.get('wrist_frames', []) if demonstration_data else []
            actions= demonstration_data.get('actions', []) if demonstration_data else []
            states = demonstration_data.get('states', []) if demonstration_data else []
            velocity = demonstration_data.get('velocity', []) if demonstration_data else []
            subgoal = demonstration_data.get('subgoal', []) if demonstration_data else []
            subgoal_grounded = demonstration_data.get('subgoal_grounded', []) if demonstration_data else []
            language_goal = demonstration_data.get('language goal') if demonstration_data else None


            while True:

            
                action = None

                obs, reward, terminated, truncated, info = env.step(action)
                

                image=env.frames[-1]
                wrist_image=env.wrist_frames[-1]
                last_action=env.actions[-1]
                state=env.states[-1]
                velocity=env.velocity[-1]
                subgoal_grounded=env.subgoal_grounded[-1]


                if info["success"] == torch.tensor([True]):
                    print(info["success"])

                if gui_render:
                    env.render()

                if truncated:
                    print("time limit!")
                    break
                if terminated:
                    obs, reward, terminated, truncated, info = env.step(action)
                    if info.get("success", False):
                        print("success")
                    if info.get("fail", False):
                        print("fail")
                    break

                #print(f"Step: {env.elapsed_steps}")
            env.close()

if __name__ == "__main__":
    main()
