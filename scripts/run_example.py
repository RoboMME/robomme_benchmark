"""
Run a single benchmark episode and save the rollout as a video.

Use this script to sanity-check the environment and action space
without running full evaluation.
"""

from pathlib import Path
from typing import Generator, Literal, Optional, Union

import cv2
import imageio
import numpy as np
import tyro

from robomme.env_record_wrapper import BenchmarkEnvBuilder
from robomme.robomme_env.utils import (
    EE_POSE_ACTION_SPACE,
    JOINT_ACTION_SPACE,
    KEYPOINT_ACTION_SPACE,
    MULTI_CHOICE_ACTION_SPACE,
)

# --- Configuration ---
GUI_RENDER = False
VIDEO_FPS = 30
VIDEO_OUTPUT_DIR = "sample_run_videos"
MAX_STEPS = 100

# Episode index limits
TRAIN_EPISODE_LIMIT = 100
TEST_VAL_EPISODE_LIMIT = 50

# Action generation constants
NOISE_LEVEL = 0.01
VIDEO_FRAME_BORDER_COLOR = (255, 0, 0)  # Blue border for video frames
VIDEO_FRAME_BORDER_THICKNESS = 10

# Type aliases
TaskID = Literal[
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
]

ActionSpaceType = Literal["joint_angle", "ee_pose", "keypoint", "multi_choice"]
DatasetType = Literal["train", "test", "val"]


def _add_small_noise(
    action: np.ndarray, noise_level: float = 0.0
) -> np.ndarray:
    noise = np.random.normal(0, noise_level, action.shape)
    noise[..., -1:] = 0.0  # Preserve gripper action
    return action + noise


def generate_sample_actions(
    action_space: str, task_id: Optional[str] = None
) -> Generator[Union[np.ndarray, dict], None, None]:
    if action_space == JOINT_ACTION_SPACE:
        # Joint angles: [base_x, base_y, base_theta, shoulder, elbow, wrist, gripper_angle, gripper_state]
        base = np.array(
            [0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, np.pi / 4, 1.0],
            dtype=np.float32,
        )
        while True:
            yield _add_small_noise(base.copy(), noise_level=NOISE_LEVEL)

    elif action_space == EE_POSE_ACTION_SPACE:
        # End-effector pose: [x, y, z, roll, pitch, yaw, gripper]
        base = np.array(
            [0.0, 0.0, 0.52, -np.pi, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        while True:
            yield _add_small_noise(base.copy(), noise_level=NOISE_LEVEL)

    elif action_space == KEYPOINT_ACTION_SPACE:
        # TODO: hongze makes this correct — predefined keypoints for PickXtimes episode 0
        waypoints = [
            [-0.01452322, 0.00461263, 0.06710568],
            [0.00062228, 0.00461263, 0.06710568],
            [0.01576777, 0.00461263, 0.06710568],
        ]
        for waypoint in waypoints:
            yield np.array(waypoint, dtype=np.float32)

    elif action_space == MULTI_CHOICE_ACTION_SPACE:
        # TODO: hongze makes this correct, give the correct choices for PickXtimes episode 0

        choices = [
            {"choice": "A", "point": [2, 1]},
            {"choice": "B", "point": [2, 2]},
            {"choice": "C", "point": None},
        ]
        for choice in choices:
            yield choice

    else:
        raise ValueError(f"Unsupported action space: {action_space}")


def _frame_from_obs(obs: dict, is_video_demo: bool = False) -> np.ndarray:
    front = obs["front_rgb_list"][0].cpu().numpy()
    wrist = obs["wrist_rgb_list"][0].cpu().numpy()
    frame = np.hstack([front, wrist]).astype(np.uint8)

    if is_video_demo:
        height, width = frame.shape[:2]
        frame = cv2.rectangle(
            frame,
            (0, 0),
            (width, height),
            VIDEO_FRAME_BORDER_COLOR,
            VIDEO_FRAME_BORDER_THICKNESS,
        )

    return frame


def _validate_episode_index(
    episode_idx: int, dataset: DatasetType
) -> None:
    if dataset == "train":
        if not (0 <= episode_idx < TRAIN_EPISODE_LIMIT):
            raise ValueError(
                f"Invalid episode_idx: {episode_idx}. "
                f"Allowed episode_idx: [0, {TRAIN_EPISODE_LIMIT})"
            )
    else:  # test or val
        if not (0 <= episode_idx < TEST_VAL_EPISODE_LIMIT):
            raise ValueError(
                f"Invalid episode_idx: {episode_idx}. "
                f"Allowed episode_idx: [0, {TEST_VAL_EPISODE_LIMIT})"
            )


def _determine_outcome_message(task_id: str, episode_idx: int, status: str) -> str:
    status_messages = {
        "success": f"[{task_id}] episode {episode_idx} is successful.",
        "fail": f"[{task_id}] episode {episode_idx} is failed.",
        "timeout": f"[{task_id}] episode {episode_idx} is timeout.",
    }
    return status_messages.get(status, f"[{task_id}] episode {episode_idx} completed.")


def _save_video(
    frames: list[np.ndarray],
    task_id: str,
    episode_idx: int,
    action_space_type: str,
) -> Path:
    video_dir = Path(VIDEO_OUTPUT_DIR)
    video_dir.mkdir(parents=True, exist_ok=True)

    video_name = f"{task_id}_ep{episode_idx}_{action_space_type}.mp4"
    video_path = video_dir / video_name

    imageio.mimsave(str(video_path), frames, fps=VIDEO_FPS)
    return video_path


def main(
    action_space_type: ActionSpaceType = "joint_angle",
    dataset: DatasetType = "test",
    task_id: TaskID = "RouteStick",
    episode_idx: int = 0,
) -> None:
    """
    Run a single benchmark episode and save the rollout as a video.

    Args:
        action_space_type: Type of action space to use.
        dataset: Dataset split to use (train, test, or val).
        task_id: Task identifier.
        episode_idx: Episode index to run.
            - For train: [0, 100)
            - For test/val: [0, 50)

    Raises:
        ValueError: If task_id or episode_idx is invalid.
    """
    task_id_list = BenchmarkEnvBuilder.get_task_list()
    print(f"All RoboMME tasks: {task_id_list}")
    print(f"Using action_space: {action_space_type}")
    print(f"Running task: {task_id}, episode: {episode_idx}, dataset: {dataset}")

    # Validate inputs
    if task_id not in task_id_list:
        raise ValueError(
            f"Invalid task_id: {task_id}. Allowed task_ids: {task_id_list}"
        )
    _validate_episode_index(episode_idx, dataset)

    # Create environment builder
    env_builder = BenchmarkEnvBuilder(
        env_id=task_id,
        dataset=dataset,
        action_space=action_space_type,
        gui_render=GUI_RENDER,
        max_steps=MAX_STEPS,
    )
    episode_count = env_builder.get_episode_num()
    print(
        f"[{task_id}] contains {episode_count} episodes in {dataset} dataset"
    )

    # Create and reset environment
    env = env_builder.make_env_for_episode(episode_idx)
    print(
        f"seed={env.unwrapped.Robomme_seed}, "
        f"difficulty={env.unwrapped.Robomme_difficulty}"
    )
    obs, info = env.reset()

    # Capture initial frames
    # Obs values are lists: length 1 for no video, >1 for video; last element is current
    frames = []
    n_obs = len(obs["front_rgb_list"])

    for i in range(n_obs):
        single_obs = {k: [v[i]] for k, v in obs.items()}
        is_video_demo = i < n_obs - 1
        frames.append(_frame_from_obs(single_obs, is_video_demo=is_video_demo))

    # Print task information
    task_goal = info["task_goal"]
    print(f"Task goal: {task_goal}")
    print(f"Oracle simple subgoal: {info['simple_subgoal_online']}")
    print(f"Oracle grounded subgoal: {info['grounded_subgoal_online']}")

    # Run episode
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
            status = info.get("status", "unknown")
            print(_determine_outcome_message(task_id, episode_idx, status))
            break

    env.close()

    # Save video
    video_path = _save_video(frames, task_id, episode_idx, action_space_type)
    print(f"Saved video to {video_path}")


if __name__ == "__main__":
    tyro.cli(main)
