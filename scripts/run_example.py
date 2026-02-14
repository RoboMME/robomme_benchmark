"""
Run a single benchmark episode and save the rollout as a video.

Use this script to sanity-check the environment and action space
without running full evaluation.
"""

import os
from typing import Literal

import cv2
import imageio
import numpy as np
import torch
import tyro

# Register envs and expose action space constants (consistent with evaluation scripts)
from robomme.robomme_env import *
from robomme.robomme_env.utils import *
from robomme.env_record_wrapper import BenchmarkEnvBuilder
from robomme.robomme_env.utils import (
    EE_POSE_ACTION_SPACE,
    JOINT_ACTION_SPACE,
    KEYPOINT_ACTION_SPACE,
    MULTI_CHOICE_ACTION_SPACE,
)


GUI_RENDER = False
MAX_STEPS = 200
VIDEO_FPS = 30
VIDEO_OUTPUT_DIR = "sample_run_videos"


def _add_small_noise(
    action: np.ndarray, noise_level: float = 0.0
) -> np.ndarray:
    """Add Gaussian noise to the first `dim` dimensions of the action."""
    noise = np.random.normal(0, noise_level, action.shape)
    noise[..., -1:] = 0.0
    return action + noise


def generate_sample_actions(action_space: str):
    if action_space == JOINT_ACTION_SPACE:
        base = np.array(
            [0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, np.pi / 4, 1.0],
            dtype=np.float32,
        )
        while True:
            yield _add_small_noise(base.copy(), noise_level=0.01)
    elif action_space == EE_POSE_ACTION_SPACE:
        base = np.array(
            [0.0, 0.0, 0.52, -np.pi, 0.0, 0.0, 1.0],  # x, y, z, roll, pitch, yaw, gripper
            dtype=np.float32,
        )
        while True:
            yield _add_small_noise(base.copy(), noise_level=0.01)
    elif action_space == KEYPOINT_ACTION_SPACE:
        # TODO: hongze makes this correct — predefined keypoints for PickXtimes episode 0
        waypoints = [
            [-0.01452322, 0.00461263, 0.06710568],
            [0.00062228, 0.00461263, 0.06710568],
            [0.01576777, 0.00461263, 0.06710568],
        ]
        for wp in waypoints:
            yield np.array(wp, dtype=np.float32)
    elif action_space == MULTI_CHOICE_ACTION_SPACE:
        # TODO: hongze makes this correct, give the correct choices for PickXtimes episode 0
        choices = [{"choice": "A", "point": [2, 1]}, {"choice": "B", "point": [2, 2]}, {"choice": "C", "point": None}]
        for choice in choices:
            yield choice
    else:
        raise ValueError(f"Unsupported action space: {action_space}")


def _frame_from_obs(obs, is_video_demo: bool = False) -> np.ndarray:
    """Build a single side-by-side frame from front and wrist camera obs."""
    front = obs["front_camera"][0].cpu().numpy()
    wrist = obs["wrist_camera"][0].cpu().numpy()
    frame = np.hstack([front, wrist]).astype(np.uint8)
    if is_video_demo:
        frame = cv2.rectangle(
            frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 10
        )
    return frame


def main(
    action_space_type: Literal["joint_angle", "ee_pose", "keypoint", "multi_choice"] = "joint_angle",
    dataset: Literal["train", "test", "val"] = "test",
) -> None:
    env_id_list = BenchmarkEnvBuilder.get_task_list()
    print(f"All RoboMME tasks: {env_id_list}")
    print(f"Using action_space: {action_space_type}")

    env_id = "PickXtimes"
    episode_idx = 0

    print(f"Running task: {env_id}")
    env_builder = BenchmarkEnvBuilder(
        env_id=env_id,
        dataset=dataset,
        action_space=action_space_type,
        gui_render=GUI_RENDER,
    )
    episode_count = env_builder.get_episode_num()
    print(f"[{env_id}] contains {episode_count} episodes in {dataset} dataset")

    if episode_count == 0:
        print(f"No episodes in {dataset} for {env_id}. Exiting.")
        return

    env, _, _ = env_builder.make_env_for_episode(episode_idx) # TODO: hongze put the maxsteps as input here 
    obs, info = env.reset()

    # Obs values are lists: length 1 for no video, >1 for video; last element is current.
    frames = []
    n_obs = len(obs["front_camera"])
    for i in range(n_obs):
        frame = _frame_from_obs(
            {k: [v[i]] for k, v in obs.items()},
            is_video_demo=(i < n_obs - 1),
        )
        frames.append(frame)

    task_goal = info["language_goal"][0]
    print(f"Task goal: {task_goal}")
    print(f"Oracle simple subgoal: {info['subgoal'][-1]}")
    print(f"Oracle grounded subgoal: {info['subgoal_grounded'][-1]}")

    step = 0
    action_gen = generate_sample_actions(action_space_type)
    while step < MAX_STEPS:
        action = next(action_gen)
        obs, _, terminated, _, info = env.step(action)

        frames.append(_frame_from_obs(obs))
        step += 1

        if GUI_RENDER:
            env.render()
        
        if step == MAX_STEPS:
            print(f"Step {step} exceeded, terminating episode {episode_idx}.")
            break
        if terminated[-1]: #TODO: hongze remove redundant nested lists 
            if info.get("success", False)[-1][-1]:
                print(f"[{env_id}] episode {episode_idx} success.")
            elif info.get("fail", False)[-1][-1]:
                print(f"[{env_id}] episode {episode_idx} failed.")
            break

    env.close()

    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
    video_path = os.path.join(
        VIDEO_OUTPUT_DIR, f"{env_id}_ep{episode_idx}_{action_space_type}.mp4"
    )
    imageio.mimsave(video_path, frames, fps=VIDEO_FPS)
    print(f"Saved video to {video_path}")


if __name__ == "__main__":
    tyro.cli(main)
