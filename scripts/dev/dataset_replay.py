# -*- coding: utf-8 -*-
# Script function: Unified dataset replay entry point, supporting 5 action spaces: joint_angle / ee_pose / ee_quat / keypoint / oracle_planner.
# Consistent with evaluate.py main loop; difference is actions come from EpisodeDatasetResolver.

import os
import re
from typing import Any, Optional

import numpy as np
import torch

from robomme.robomme_env import *
from robomme.robomme_env.utils import *
from robomme.env_record_wrapper import (
    BenchmarkEnvBuilder,
    EpisodeDatasetResolver,
)
from robomme.robomme_env.utils.save_reset_video import save_robomme_video

# Only enable one ACTION_SPACE; others are commented out for manual switching
#ACTION_SPACE = "joint_angle"
ACTION_SPACE = "ee_pose"

#ACTION_SPACE = "keypoint"
#ACTION_SPACE = "oracle_planner"

GUI_RENDER = False

DATASET_ROOT = "/data/hongzefu/data_0217"

DEFAULT_ENV_IDS = [
    # "PickXtimes",
    # "StopCube",
    # "SwingXtimes",
    #"BinFill",
     #"VideoUnmaskSwap",
     "VideoUnmask",
    # "ButtonUnmaskSwap",
    # "ButtonUnmask",
     #"VideoRepick",
    # "VideoPlaceButton",
    # "VideoPlaceOrder",
   # "PickHighlight",
    # "InsertPeg",
    # "MoveCube",
    #"PatternLock",
   #  "RouteStick",
]

OUT_VIDEO_DIR = "/data/hongzefu/dataset_replay"
MAX_STEPS = 1000




def _parse_oracle_command(subgoal_text: Optional[str]) -> Optional[dict[str, Any]]:
    if not subgoal_text:
        return None
    point = None
    match = re.search(r"<\s*(-?\d+)\s*,\s*(-?\d+)\s*>", subgoal_text)
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        # Dataset text is usually <x, y>, Oracle wrapper expects [row, col], i.e., [y, x]
        point = [y, x]
    return {"action": subgoal_text, "point": point}


def main():
    env_id_list = BenchmarkEnvBuilder.get_task_list()
    print(f"Running envs: {env_id_list}")
    print(f"Using action_space: {ACTION_SPACE}")

    #for env_id in env_id_list:
    for env_id in DEFAULT_ENV_IDS:
        env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="train",
            action_space=ACTION_SPACE,
            gui_render=GUI_RENDER,
        )
        episode_count = env_builder.get_episode_num()
        print(f"[{env_id}] episode_count from metadata: {episode_count}")

        env = None
        for episode in range(episode_count):

            env = env_builder.make_env_for_episode(episode, max_steps=MAX_STEPS)
            dataset_resolver = EpisodeDatasetResolver(
                env_id=env_id,
                episode=episode,
                dataset_directory=DATASET_ROOT,
            )

            # ======== Reset ========
            # obs: dict-of-lists (columnar batch, list length = number of demo frames)
            # info: flat dict (last frame values only)
            obs, info = env.reset()

            # --- Explicitly read all obs fields (each is a list) ---
            maniskill_obs = obs["maniskill_obs"]
            front_rgb_list = obs["front_rgb_list"]
            wrist_rgb_list = obs["wrist_rgb_list"]
            front_depth_list = obs["front_depth_list"]
            wrist_depth_list = obs["wrist_depth_list"]
            end_effector_pose_raw = obs["end_effector_pose_raw"]
            eef_state_list = obs["eef_state_list"]
            joint_state_list = obs["joint_state_list"]
            # velocity = obs["velocity"]
            gripper_state_list = obs["gripper_state_list"]
            front_camera_extrinsic_list = obs["front_camera_extrinsic_list"]
            wrist_camera_extrinsic_list = obs["wrist_camera_extrinsic_list"]

            # --- Explicitly read all info fields (flat dict, last frame values) ---
            task_goal = info["task_goal"]
            simple_subgoal_online = info["simple_subgoal_online"]
            grounded_subgoal_online = info["grounded_subgoal_online"]
            available_multi_choices = info.get("available_multi_choices")
            front_camera_intrinsic = info["front_camera_intrinsic"]
            wrist_camera_intrinsic = info["wrist_camera_intrinsic"]
            status = info.get("status")


            # --- Video saving variable preparation (reset phase) ---
            reset_base_frames = [torch.as_tensor(f).detach().cpu().numpy().copy() for f in front_rgb_list]
            reset_wrist_frames = [torch.as_tensor(f).detach().cpu().numpy().copy() for f in wrist_rgb_list]
            reset_subgoal_grounded = [grounded_subgoal_online] * len(front_rgb_list)

            step = 0
            episode_success = False
            rollout_base_frames: list[np.ndarray] = []
            rollout_wrist_frames: list[np.ndarray] = []
            rollout_subgoal_grounded: list[Any] = []

            # ======== Step loop ========
            while True:
                replay_key = ACTION_SPACE
                action = dataset_resolver.get_step(replay_key, step)
                if ACTION_SPACE == "oracle_planner":
                    action = _parse_oracle_command(action)
                if action is None:
                    break

                # step returns: obs (dict-of-lists), reward (scalar tensor),
                #               terminated (scalar tensor), truncated (scalar tensor), info (flat dict)
                obs, reward, terminated, truncated, info = env.step(action)

                # --- Explicitly read all obs fields (dict-of-lists, typically 1 element per list) ---
                maniskill_obs = obs["maniskill_obs"]
                front_rgb_list = obs["front_rgb_list"]
                wrist_rgb_list = obs["wrist_rgb_list"]
                front_depth_list = obs["front_depth_list"]
                wrist_depth_list = obs["wrist_depth_list"]
                end_effector_pose_raw = obs["end_effector_pose_raw"]
                eef_state_list = obs["eef_state_list"]
                joint_state_list = obs["joint_state_list"]
                gripper_state_list = obs["gripper_state_list"]
                front_camera_extrinsic_list = obs["front_camera_extrinsic_list"]
                wrist_camera_extrinsic_list = obs["wrist_camera_extrinsic_list"]

                # --- Explicitly read all info fields (flat dict) ---
                task_goal = info["task_goal"]
                simple_subgoal_online = info["simple_subgoal_online"]
                grounded_subgoal_online = info["grounded_subgoal_online"]
                available_multi_choices = info.get("available_multi_choices")
                front_camera_intrinsic = info["front_camera_intrinsic"]
                wrist_camera_intrinsic = info["wrist_camera_intrinsic"]
                status = info.get("status")

                # --- Video saving variable preparation (replay phase) ---
                rollout_base_frames.extend(
                    torch.as_tensor(f).detach().cpu().numpy().copy() for f in front_rgb_list
                )
                rollout_wrist_frames.extend(
                    torch.as_tensor(f).detach().cpu().numpy().copy() for f in wrist_rgb_list
                )
                rollout_subgoal_grounded.extend([grounded_subgoal_online] * len(front_rgb_list))

                terminated_flag = bool(terminated.item())
                truncated_flag = bool(truncated.item())

                step += 1
                if GUI_RENDER:
                    env.render()
                if truncated_flag:
                    print(f"[{env_id}] episode {episode} steps exceeded, step {step}.")
                    break
                if terminated_flag:
                    if status == "success":
                        print(f"[{env_id}] episode {episode} success.")
                        episode_success = True
                    elif status == "fail":
                        print(f"[{env_id}] episode {episode} failed.")
                    break

            # ======== Video saving ========
            save_robomme_video(
                reset_base_frames=reset_base_frames,
                reset_wrist_frames=reset_wrist_frames,
                rollout_base_frames=rollout_base_frames,
                rollout_wrist_frames=rollout_wrist_frames,
                reset_subgoal_grounded=reset_subgoal_grounded,
                rollout_subgoal_grounded=rollout_subgoal_grounded,
                out_video_dir=OUT_VIDEO_DIR,
                action_space=ACTION_SPACE,
                env_id=env_id,
                episode=episode,
                episode_success=episode_success,
            )

        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
