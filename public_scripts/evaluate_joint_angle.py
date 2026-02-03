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
from PIL import Image


def main():
    """
    Main function to run the simulation and record data for multiple seeds.
    """

    num_episodes = 1
    gui_render=True
    max_steps=3000
    


    env_id_list = [
        "VideoRepick",
        #"BinFill",
        # "ButtonUnmask",
        # "ButtonUnmaskSwap",


        # "InsertPeg",
        # "MoveCube",
        # "PatternLock",
        # "PickHighlight",
        # "PickXtimes",
        # "RouteStick",
        # "StopCube",

        # "SwingXtimes",
       # "VideoPlaceButton",
        #"VideoPlaceOrder",
        # "VideoUnmask",
        # "VideoUnmaskSwap",
    ]



    if gui_render==True:
        render_mode="human"
    else:
        render_mode="rgb_array"

    for env_id in env_id_list:

        metadata_path =  f"/data/hongzefu/historybench-v5.6.19.1-gradio-changeStopcube3/dataset_json/record_dataset_{env_id}_metadata.json"

        resolver = EpisodeConfigResolver(
            env_id=env_id,
            metadata_path=metadata_path,
            render_mode=render_mode,
            gui_render=gui_render,
            max_steps_without_demonstration=max_steps,
        )

        for episode in range(num_episodes):
            episode=98
            env, seed, difficulty = resolver.make_env_for_episode(episode)
           
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


            #保存最后一张frame和wrist_frame 左右拼接成一张图片
            image = np.concatenate([frames[-1], wrist_frames[-1]], axis=1)
            image = Image.fromarray(image)
            image.save(f"last_frame_{env_id}_{episode}.png")




            while True:

            
                action = None

                action=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

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


                if info["success"] == torch.tensor([True]):
                    print(info["success"])

                if gui_render:
                    test=env.render()
                    print(test)
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
