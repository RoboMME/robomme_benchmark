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
    gui_render=True
    max_steps=3000
    

    # env_id_list=['VideoUnmaskSwap']
    # dataset_path= f"/data/hongzefu/dataset_generate/record_dataset_VideoUnmaskSwap.h5"
    env_id_list=['BinFill']
    dataset_path= f"/data/hongzefu/data_1129_3/record_dataset_BinFill.h5"
    h5_path = dataset_path
    dataset = h5py.File(h5_path, "r")
    metadata_path = Path(h5_path).with_name(f"{Path(h5_path).stem}_metadata.json")

    if gui_render==True:
        render_mode="human"
    else:
        render_mode="rgb_array"

    for env_id in env_id_list:
        env_dataset = dataset[f"env_{env_id}"]
        resolver = EpisodeConfigResolver(
            env_id=env_id,
            dataset=dataset,
            metadata_path=metadata_path,
            render_mode=render_mode,
            gui_render=gui_render,
            max_steps_without_demonstration=max_steps,
        )
        episode_indexs=sorted(
                int(k.split("_")[1]) for k in env_dataset.keys() if k.startswith("episode_")
            )

        for episode in range(num_episodes):
            env, episode_dataset, seed, difficulty_hint = resolver.make_env_for_episode(6)

            language_goal=episode_dataset["setup"]["language goal"][()]
            print(f"setting task for {language_goal}")

            env.reset()

            # Demonstration data is now automatically generated in reset()
            demonstration_data = env.demonstration_data
            
            timestep_indexes = sorted(
                int(m.group(1))
                for k in episode_dataset.keys()
                if (m := re.search(r'^record_timestep_(\d+)$', k))
            )
            # add timestep_indexs variable to align timestep and to stop automatically

            for step in timestep_indexes:

                print(step)
                timestep_group = episode_dataset[f"record_timestep_{step}"]


                if timestep_group['demonstration']==True:
                    continue

                action = np.asarray(timestep_group["action"][()])

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
                    obs, reward, terminated, truncated, info = env.step(action)#highlight显示
                    if info.get("success", False):
                        print("success")
                    if info.get("fail", False):
                        print("fail")
                    break

            env.close()

if __name__ == "__main__":
    main()
