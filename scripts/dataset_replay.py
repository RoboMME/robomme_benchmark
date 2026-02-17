"""
Replay episodes from HDF5 datasets and save rollout videos.

Loads recorded joint actions from record_dataset_<Task>.h5, steps the environment,
and writes side-by-side front/wrist camera videos to disk.
"""

import os

import cv2
import h5py
import imageio
import numpy as np

from robomme.robomme_env import *
from robomme.robomme_env.utils import *
from robomme.env_record_wrapper import BenchmarkEnvBuilder
from robomme.robomme_env.utils import JOINT_ACTION_SPACE

# --- Config ---
GUI_RENDER = False
REPLAY_VIDEO_DIR = "replay_videos"
VIDEO_FPS = 30


def _frame_from_obs(obs: dict, is_video_frame: bool = False) -> np.ndarray:
    """Build a single side-by-side frame from front and wrist camera obs."""
    front = obs["front_camera"][0].cpu().numpy()
    wrist = obs["wrist_camera"][0].cpu().numpy()
    frame = np.concatenate([front, wrist], axis=1).astype(np.uint8)
    if is_video_frame:
        frame = cv2.rectangle(
            frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 10
        )
    return frame


def _first_execution_step(episode_data) -> int:
    """Return the first step index that is not a video-demo step."""
    step_idx = 0
    while episode_data[f"timestep_{step_idx}"]["info"]["is_video_demo"][()]:
        step_idx += 1
    return step_idx


def process_episode(env_data: h5py.File, episode_idx: int, env_id: str) -> None:
    """Replay one episode from HDF5 data, record frames, and save a video."""
    episode_data = env_data[f"episode_{episode_idx}"]
    task_goal = episode_data["setup"]["task_goal"][()].decode()
    total_steps = sum(1 for k in episode_data.keys() if k.startswith("timestep_"))

    step_idx = _first_execution_step(episode_data)
    print(f"Execution start step index: {step_idx}")

    env_builder = BenchmarkEnvBuilder(
        env_id=env_id,
        dataset="train",
        action_space=JOINT_ACTION_SPACE,
        gui_render=GUI_RENDER,
    )
    env = env_builder.make_env_for_episode(episode_idx)
    print(f"seed={env.unwrapped.Robomme_seed}, difficulty={env.unwrapped.Robomme_difficulty}")
    setup_group = episode_data.get("setup")
    if setup_group is not None:
        env_unwrapped = getattr(env, "unwrapped", env)
        reserved_setup_keys = {
            "seed",
            "difficulty",
            "task_goal",
            "subgoal_list",
            "front_camera_intrinsic",
            "wrist_camera_intrinsic",
        }
        for key, value in setup_group.items():
            if key in reserved_setup_keys or isinstance(value, h5py.Group):
                continue
            merged_value = value[()]
            if isinstance(merged_value, (bytes, np.bytes_)):
                merged_value = merged_value.decode("utf-8")
            elif isinstance(merged_value, np.ndarray) and merged_value.dtype.kind == "S":
                merged_value = np.char.decode(merged_value, "utf-8")
            setattr(env_unwrapped, key, merged_value)
    print(f"task_name: {env_id}, episode_idx: {episode_idx}, task_goal: {task_goal}")

    obs, info = env.reset()
    # Obs lists: length 1 = no video, length > 1 = video; last element is current.
    frames = []
    n_obs = len(obs["front_camera"])
    for i in range(n_obs):
        single_obs = {k: [v[i]] for k, v in obs.items()}
        frames.append(_frame_from_obs(single_obs, is_video_frame=(i < n_obs - 1)))
    print(f"Initial frames (video + current): {len(frames)}")

    outcome = "unknown"
    try:
        while step_idx < total_steps:
            action = np.asarray(
                episode_data[f"timestep_{step_idx}"]["action"]["joint_action"][()],
                dtype=np.float32,
            )
            obs, _, terminated, _, info = env.step(action)
            frames.append(_frame_from_obs(obs))

            if GUI_RENDER:
                env.render()

            # TODO: hongze makes this correct
            # there are two many nested lists here, need to flatten them
            if terminated[-1]:
                if info.get("success", False)[-1][-1]:
                    outcome = "success"
                if info.get("fail", False)[-1][-1]:
                    outcome = "fail"
                break
            step_idx += 1
    finally:
        env.close()

    safe_goal = task_goal.replace(" ", "_").replace("/", "_")
    os.makedirs(REPLAY_VIDEO_DIR, exist_ok=True)
    video_name = f"{outcome}_{env_id}_ep{episode_idx}_{safe_goal}_step-{len(frames)}.mp4"
    video_path = os.path.join(REPLAY_VIDEO_DIR, video_name)
    imageio.mimsave(video_path, frames, fps=VIDEO_FPS)
    print(f"Saved video to {video_path}")


def replay(h5_data_dir: str = "data/robomme_h5_data") -> None:
    """Replay all episodes from all task HDF5 files in the given directory."""
    env_id_list = BenchmarkEnvBuilder.get_task_list()
    for env_id in env_id_list:
        file_name = f"record_dataset_{env_id}.h5"
        file_path = os.path.join(h5_data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Skipping {env_id}: file not found: {file_path}")
            continue

        with h5py.File(file_path, "r") as data:
            episode_indices = sorted(
                int(k.split("_")[1])
                for k in data.keys()
                if k.startswith("episode_")
            )
            print(f"Task: {env_id}, has {len(episode_indices)} episodes")
            for episode_idx in episode_indices[:1]:
                process_episode(data, episode_idx, env_id)


if __name__ == "__main__":
    import tyro
    tyro.cli(replay)
