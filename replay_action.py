import gymnasium as gym
import numpy as np
import sapien
import h5py

#from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode

import re
from env_record_wrapper import *
from HistoryBench_env import *

import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

dataset_path= "/data/hongzefu/record_dataset_ButtonUnmaskSwap.h5"
episode=10

def main():
    h5_path = dataset_path
    dataset = h5py.File(h5_path, "r")

    env_id_list=['ButtonUnmaskSwap']
    for env_id in env_id_list:

        env_dataset = dataset[f"env_{env_id}"]
        episode_indexs=sorted(
                int(k.split("_")[1]) for k in env_dataset.keys() if k.startswith("episode_")
            )

        for episode in episode_indexs:

            episode_dataset=dataset[f"env_{env_id}"][f"episode_{episode}"]
            seed=episode_dataset["setup"]["seed"][()]
            seed=int(seed)

            language_goal=episode_dataset["setup"]["language goal"][()]
            print(f"setting task for {language_goal}")

            env: PickCubeXtimesEnv = gym.make(
                id=env_id,
                obs_mode="rgb",
                control_mode="pd_joint_pos",
                render_mode="human",
                reward_mode="dense",
                HistoryBench_seed=seed,
            )
            env.reset()
            timestep_indexes = sorted(
                int(m.group(1))
                for k in episode_dataset.keys()
                if (m := re.search(r'^record_timestep_(\d+)$', k))
            )
           #add timestep_indexs variable to align timestep and to stop automatically

            for step in timestep_indexes:
                    print(step)
                    timestep_group = episode_dataset[f"record_timestep_{step}"]
                    if "action" not in timestep_group:
                        print(f"[replay] timestep_{step} 缺少 'action'，结束回放。")
                        break

                    action = np.asarray(timestep_group["action"][()])

                    obs, reward, terminated, truncated, info = env.step(action)

                    image = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
                    wrist_image = obs['sensor_data']['hand_camera']['rgb'][0].cpu().numpy()

                    if info["success"]==torch.tensor([True]):
                        print(info["success"])
                    env.render()

            env.close()


if __name__ == "__main__":
    main()
