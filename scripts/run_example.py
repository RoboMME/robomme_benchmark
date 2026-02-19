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

VIDEO_FPS = 30
VIDEO_OUTPUT_DIR = "sample_run_videos"
MAX_STEPS = 100


def _add_small_noise(
    action: np.ndarray, noise_level: float = 0.0
) -> np.ndarray:
    """Add Gaussian noise to the first `dim` dimensions of the action."""
    noise = np.random.normal(0, noise_level, action.shape)
    noise[..., -1:] = 0.0
    return action + noise


def generate_sample_actions(action_space: str, task_id: str=None):
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
    front = obs["front_rgb_list"][0].cpu().numpy()
    wrist = obs["wrist_rgb_list"][0].cpu().numpy()
    frame = np.hstack([front, wrist]).astype(np.uint8)
    if is_video_demo:
        frame = cv2.rectangle(
            frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 10
        )
    return frame


def main(
    action_space_type: Literal["joint_angle", "ee_pose", "keypoint", "multi_choice"] = "joint_angle",
    dataset: Literal["train", "test", "val"] = "test",
    task_id: Literal["BinFill", "PickXtimes", "SwingXtimes", "StopCube", "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap", "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder", "MoveCube", "InsertPeg", "PatternLock", "RouteStick"] = "RouteStick",
    episode_idx: int = 0, # [0, 100) for train, [0, 50) for test and val
) -> None:
    task_id_list = BenchmarkEnvBuilder.get_task_list()
    print(f"All RoboMME tasks: {task_id_list}")
    print(f"Using action_space: {action_space_type}")
    print(f"Running task: {task_id}, episode: {episode_idx}, dataset: {dataset}")
    
    assert task_id in task_id_list, f"Invalid env_id: {task_id}. Allowed env_ids: {task_id_list}"
    if dataset == "train":
        assert 0 <= episode_idx < 100, f"Invalid episode_idx: {episode_idx}. Allowed episode_idx: [0, 100)"
    else:   # test or val
        assert 0 <= episode_idx < 50, f"Invalid episode_idx: {episode_idx}. Allowed episode_idx: [0, 50)"
    
    env_builder = BenchmarkEnvBuilder(
        env_id=task_id,
        dataset=dataset,
        action_space=action_space_type,
        gui_render=GUI_RENDER,
        max_steps=MAX_STEPS,
    )
    episode_count = env_builder.get_episode_num()
    print(f"[{task_id}] contains {episode_count} episodes in {dataset} dataset")


    env = env_builder.make_env_for_episode(episode_idx)
    print(f"seed={env.unwrapped.Robomme_seed}, difficulty={env.unwrapped.Robomme_difficulty}")
    obs, info = env.reset()

    # Obs values are lists: length 1 for no video, >1 for video; last element is current.
    frames = []
    n_obs = len(obs["front_rgb_list"])
    for i in range(n_obs):
        frame = _frame_from_obs(
            {k: [v[i]] for k, v in obs.items()},
            is_video_demo=(i < n_obs - 1),
        )
        frames.append(frame)

    task_goal = info["task_goal"]
    print(f"Task goal: {task_goal}")
    print(f"Oracle simple subgoal: {info['simple_subgoal_online']}")
    print(f"Oracle grounded subgoal: {info['grounded_subgoal_online']}")

    step = 0
    action_gen = generate_sample_actions(action_space_type)
    
    while True:
        action = next(action_gen)
        obs, _, terminated, truncated, info = env.step(action)

        frames.append(_frame_from_obs(obs))
        step += 1

        if GUI_RENDER:
            env.render()

        if terminated or truncated:
            if info.get("status") == "success":
                print(f"[{task_id}] episode {episode_idx} is successful.")
            elif info.get("status") == "fail":
                print(f"[{task_id}] episode {episode_idx} is failed.")
            elif info.get("status") == "timeout":
                print(f"[{task_id}] episode {episode_idx} is timeout.")
            break

    env.close()

    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
    video_path = os.path.join(
        VIDEO_OUTPUT_DIR, f"{task_id}_ep{episode_idx}_{action_space_type}.mp4"
    )
    imageio.mimsave(video_path, frames, fps=VIDEO_FPS)
    print(f"Saved video to {video_path}")


if __name__ == "__main__":
    tyro.cli(main)
