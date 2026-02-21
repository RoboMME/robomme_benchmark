"""
Replay episodes from HDF5 datasets and save rollout videos.

Loads recorded joint actions from record_dataset_<Task>.h5, steps the environment,
and writes side-by-side front/wrist camera videos to disk.
"""

from pathlib import Path
from typing import Literal, Optional

import cv2
import h5py
import imageio
import numpy as np

from robomme.env_record_wrapper import BenchmarkEnvBuilder
from robomme.robomme_env.utils import (
    EE_POSE_ACTION_SPACE,
    JOINT_ACTION_SPACE,
    KEYPOINT_ACTION_SPACE,
    MULTI_CHOICE_ACTION_SPACE,
)

# --- Configuration ---
GUI_RENDER = False
REPLAY_VIDEO_DIR = "replay_videos"
VIDEO_FPS = 30

# Video frame annotation constants
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


def _frame_from_obs(obs: dict, is_video_frame: bool = False) -> np.ndarray:
    front = obs["front_rgb_list"][0].cpu().numpy()
    wrist = obs["wrist_rgb_list"][0].cpu().numpy()
    frame = np.concatenate([front, wrist], axis=1).astype(np.uint8)

    if is_video_frame:
        height, width = frame.shape[:2]
        frame = cv2.rectangle(
            frame,
            (0, 0),
            (width, height),
            VIDEO_FRAME_BORDER_COLOR,
            VIDEO_FRAME_BORDER_THICKNESS,
        )

    return frame


def _first_execution_step(episode_data: h5py.Group) -> int:
    step_idx = 0
    timestep_key = f"timestep_{step_idx}"

    while timestep_key in episode_data:
        timestep_group = episode_data[timestep_key]
        if "info" in timestep_group and "is_video_demo" in timestep_group["info"]:
            if not timestep_group["info"]["is_video_demo"][()]:
                break
        step_idx += 1
        timestep_key = f"timestep_{step_idx}"

    return step_idx


def _load_action_from_timestep(
    episode_data: h5py.Group, step_idx: int, action_space_type: str
) -> np.ndarray:
    timestep_key = f"timestep_{step_idx}"
    action_group = episode_data[timestep_key]["action"]

    if action_space_type == JOINT_ACTION_SPACE:
        action_data = action_group["joint_action"][()]
    elif action_space_type == EE_POSE_ACTION_SPACE:
        action_data = action_group["eef_action"][()]
    elif action_space_type == KEYPOINT_ACTION_SPACE: # TODO: Hongze make this correct
        raise NotImplementedError(
            f"Keypoint action space type is not supported for dataset replay."
        )
    elif action_space_type == MULTI_CHOICE_ACTION_SPACE: 
        # TODO: Hongze make this correct
        raise NotImplementedError(
            f"Multi-choice action space type is not supported for dataset replay."
        )
    else:
        raise ValueError(f"Unknown action space type: {action_space_type}")

    return np.asarray(action_data, dtype=np.float32)



def _determine_outcome(info: dict) -> str:
    status = info.get("status", "")
    if status == "success":
        return "success"
    elif status == "fail":
        return "fail"
    return "unknown"


def _save_video(
    frames: list[np.ndarray],
    task_id: str,
    episode_idx: int,
    task_goal: str,
    outcome: str,
    action_space_type: str,
) -> Path:
    safe_goal = task_goal.replace(" ", "_").replace("/", "_")
    video_dir = Path(REPLAY_VIDEO_DIR) / action_space_type
    video_dir.mkdir(parents=True, exist_ok=True)

    video_name = (
        f"{outcome}_{task_id}_ep{episode_idx}_{safe_goal}_step-{len(frames)}.mp4"
    )
    video_path = video_dir / video_name

    imageio.mimsave(str(video_path), frames, fps=VIDEO_FPS)
    return video_path


def _get_episode_indices(data: h5py.File) -> list[int]:
    return sorted(
        int(key.split("_")[1])
        for key in data.keys()
        if key.startswith("episode_")
    )


def process_episode(
    env_data: h5py.File,
    episode_idx: int,
    task_id: str,
    action_space_type: str,
) -> None:
    """
    Replay one episode from HDF5 data, record frames, and save a video.

    Args:
        env_data: Open HDF5 file containing episode data.
        episode_idx: Index of the episode to replay.
        task_id: Task identifier.
        action_space_type: Type of action space (joint_angle, ee_pose, etc.).
    """
    episode_key = f"episode_{episode_idx}"
    episode_data = env_data[episode_key]
    task_goal = episode_data["setup"]["task_goal"][()].decode()

    # Count total timesteps
    total_steps = sum(
        1 for k in episode_data.keys() if k.startswith("timestep_")
    )

    # Find first execution step (skip video demo steps)
    step_idx = _first_execution_step(episode_data)
    print(f"Execution start step index: {step_idx}")

    # Create and configure environment
    env_builder = BenchmarkEnvBuilder(
        env_id=task_id,
        dataset="train",
        action_space=action_space_type,
        gui_render=GUI_RENDER,
    )
    env = env_builder.make_env_for_episode(episode_idx)
    print(
        f"seed={env.unwrapped.Robomme_seed}, "
        f"difficulty={env.unwrapped.Robomme_difficulty}"
    )
    print(
        f"task_name: {task_id}, episode_idx: {episode_idx}, "
        f"task_goal: {task_goal}"
    )

    # Reset environment and capture initial frames
    obs, _ = env.reset()
    frames = []
    n_obs = len(obs["front_rgb_list"])

    # Process initial observations (video frames + current frame)
    # Last element is current frame, others are video demo frames
    for i in range(n_obs):
        single_obs = {k: [v[i]] for k, v in obs.items()}
        is_video_frame = i < n_obs - 1
        frames.append(_frame_from_obs(single_obs, is_video_frame=is_video_frame))

    print(f"Initial frames (video + current): {len(frames)}")

    # Replay episode steps
    outcome = "unknown"
    while step_idx < total_steps:
        action = _load_action_from_timestep(episode_data, step_idx, action_space_type)
        obs, _, terminated, truncated, info = env.step(action)
        frames.append(_frame_from_obs(obs))

        if GUI_RENDER:
            env.render()

        if terminated or truncated:
            outcome = _determine_outcome(info)
            break

        step_idx += 1

    env.close()

    # Save video
    video_path = _save_video(
        frames, task_id, episode_idx, task_goal, outcome, action_space_type
    )
    print(f"Saved video to {video_path}")


def replay(
    h5_data_dir: str = "data/robomme_h5_data",
    task_id: Optional[TaskID] = None,
    action_space_type: ActionSpaceType = "joint_angle",
    replay_number: int = 100,
) -> None:
    """
    Replay episodes from HDF5 dataset files and save rollout videos.

    Args:
        h5_data_dir: Directory containing HDF5 dataset files.
        task_id: Specific task ID to replay. If None, replays all tasks.
        action_space_type: Type of action space to use for replay.
        replay_number: Maximum number of episodes to replay per task.

    Raises:
        ValueError: If task_id is invalid.
    """
    task_id_list = BenchmarkEnvBuilder.get_task_list()

    if task_id is None:
        replay_list = task_id_list
    else:
        if task_id not in task_id_list:
            raise ValueError(
                f"Invalid task_id: {task_id}. "
                f"Allowed task_ids: {task_id_list}"
            )
        replay_list = [task_id]

    h5_data_path = Path(h5_data_dir)

    for current_task_id in replay_list:
        file_name = f"record_dataset_{current_task_id}.h5"
        file_path = h5_data_path / file_name

        if not file_path.exists():
            print(f"Skipping {current_task_id}: file not found: {file_path}")
            continue

        with h5py.File(file_path, "r") as data:
            episode_indices = _get_episode_indices(data)
            num_episodes = len(episode_indices)
            num_to_replay = min(replay_number, num_episodes)

            print(
                f"Task: {current_task_id}, "
                f"has {num_episodes} episodes, "
                f"replaying {num_to_replay}"
            )

            for episode_idx in episode_indices[:num_to_replay]:
                process_episode(data, episode_idx, current_task_id, action_space_type)
                import pdb; pdb.set_trace()


if __name__ == "__main__":
    import tyro
    tyro.cli(replay)
