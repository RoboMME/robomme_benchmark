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
    test_seed_offset=0
    

    env_id_list=['VideoRepick']
    dataset_path= f"/data/hongzefu/dataset_generate/record_dataset_VideoRepick.h5"
    h5_path = dataset_path
    dataset = h5py.File(h5_path, "r")


    if gui_render==True:
        render_mode="human"
    else:
        render_mode="rgb_array"

    for env_id in env_id_list:
        env_dataset = dataset[f"env_{env_id}"]
        episode_indexs=sorted(
                int(k.split("_")[1]) for k in env_dataset.keys() if k.startswith("episode_")
            )

        for episode in range(num_episodes):
            episode_dataset=dataset[f"env_{env_id}"][f"episode_{episode}"]
            seed=episode_dataset["setup"]["seed"][()]
            seed=int(seed)

            language_goal=episode_dataset["setup"]["language goal"][()]
            print(f"setting task for {language_goal}")
        



            env= gym.make(
                env_id,
                obs_mode="rgb+depth+segmentation",
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

                image = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
                wrist_image = obs['sensor_data']['hand_camera']['rgb'][0].cpu().numpy()

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
