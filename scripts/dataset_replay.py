"""
Replay episodes from HDF5 datasets and save rollout videos.
Loads recorded actions from record_dataset_<Task>.h5, steps the environment
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
from pathlib import Path
from typing import Any, Dict, Literal, Union

import cv2
import h5py
import imageio
import numpy as np
import torch

from robomme.env_record_wrapper import BenchmarkEnvBuilder

GUI_RENDER = False
REPLAY_VIDEO_DIR = "replay_videos"
VIDEO_FPS = 30
VIDEO_BORDER_COLOR = (255, 0, 0)
VIDEO_BORDER_THICKNESS = 10

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


ActionSpaceType = Literal["joint_angle", "ee_pose", "waypoint", "multi_choice"]

def _to_numpy(t) -> np.ndarray:
    return t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def _frame_from_obs(
    front: np.ndarray | torch.Tensor,
    wrist: np.ndarray | torch.Tensor,
    is_video_demo: bool = False,
) -> np.ndarray:
    frame = np.hstack([_to_numpy(front), _to_numpy(wrist)]).astype(np.uint8)
    if is_video_demo:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, h),
                      VIDEO_BORDER_COLOR, VIDEO_BORDER_THICKNESS)
    return frame


def _extract_frames(obs: dict, is_video_demo_fn=None) -> list[np.ndarray]:
    n = len(obs["front_rgb_list"])
    return [
        _frame_from_obs(
            obs["front_rgb_list"][i],
            obs["wrist_rgb_list"][i],
            is_video_demo=(is_video_demo_fn(i) if is_video_demo_fn else False),
        )
        for i in range(n)
    ]


def _is_video_demo(ts: h5py.Group) -> bool:
    info = ts.get("info")
    if info is None or "is_video_demo" not in info:
        return False
    return bool(np.reshape(np.asarray(info["is_video_demo"][()]), -1)[0])


def _is_subgoal_boundary(ts: h5py.Group) -> bool:
    info = ts.get("info")
    if info is None or "is_subgoal_boundary" not in info:
        return False
    return bool(np.reshape(np.asarray(info["is_subgoal_boundary"][()]), -1)[0])


def _decode_h5_str(raw) -> str:
    """Uniformly decode bytes / numpy bytes / str from HDF5 to str."""
    if isinstance(raw, np.ndarray):
        raw = raw.flatten()[0]
    if isinstance(raw, (bytes, np.bytes_)):
        raw = raw.decode("utf-8")
    return raw


def _build_action_sequence(
    episode_data: h5py.Group, action_space_type: str
) -> list[Union[np.ndarray, Dict[str, Any]]]:
    """
    Scan the entire episode and return the deduplicated action sequence:
    - joint_angle / ee_pose: actions of all non-video-demo steps (sequential, not deduplicated)
    - waypoint: remove adjacent duplicate waypoint_action (like EpisodeDatasetResolver)
    - multi_choice: choice_action (JSON dict) only for steps where is_subgoal_boundary=True
    """
    timestep_keys = sorted(
        (k for k in episode_data.keys() if k.startswith("timestep_")),
        key=lambda k: int(k.split("_")[1]),
    )

    actions: list[Union[np.ndarray, Dict[str, Any]]] = []
    prev_waypoint: np.ndarray | None = None

    for key in timestep_keys:
        ts = episode_data[key]
        if _is_video_demo(ts):
            continue

        action_grp = ts.get("action")
        if action_grp is None:
            continue

        if action_space_type == "joint_angle":
            if "joint_action" not in action_grp:
                continue
            actions.append(np.asarray(action_grp["joint_action"][()], dtype=np.float32))

        elif action_space_type == "ee_pose":
            if "eef_action" not in action_grp:
                continue
            actions.append(np.asarray(action_grp["eef_action"][()], dtype=np.float32))

        elif action_space_type == "waypoint":
            if "waypoint_action" not in action_grp:
                continue
            wa = np.asarray(action_grp["waypoint_action"][()], dtype=np.float32).flatten()
            if wa.shape != (7,) or not np.all(np.isfinite(wa)):
                continue
            # Remove adjacent duplicates
            if prev_waypoint is None or not np.array_equal(wa, prev_waypoint):
                actions.append(wa)
                prev_waypoint = wa.copy()

        elif action_space_type == "multi_choice":
            if not _is_subgoal_boundary(ts):
                continue
            if "choice_action" not in action_grp:
                continue
            raw = _decode_h5_str(action_grp["choice_action"][()])
            try:
                payload = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            choice = payload.get("choice")
            if not isinstance(choice, str) or not choice.strip():
                continue
            if "point" not in payload:
                continue
            actions.append({"choice": choice, "point": payload.get("point")})

        else:
            raise ValueError(f"Unknown action space type: {action_space_type}")

    return actions


def _save_video(
    frames: list[np.ndarray],
    task_id: str,
    episode_idx: int,
    task_goal: str,
    outcome: str,
    action_space_type: str,
) -> Path:
    video_dir = Path(REPLAY_VIDEO_DIR) / action_space_type
    video_dir.mkdir(parents=True, exist_ok=True)
    name = f"{outcome}_{task_id}_ep{episode_idx}_{task_goal}.mp4"
    path = video_dir / name
    imageio.mimsave(str(path), frames, fps=VIDEO_FPS)
    return path


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
    action_space_type: ActionSpaceType,
) -> None:
    """Replay one episode from HDF5 data, record frames, and save a video."""
    episode_data = env_data[f"episode_{episode_idx}"]
    task_goal = episode_data["setup"]["task_goal"][()][0].decode()
    action_sequence = _build_action_sequence(episode_data, action_space_type)

    env = BenchmarkEnvBuilder(
        env_id=task_id,
        dataset="train",
        action_space=action_space_type,
        gui_render=GUI_RENDER,
    ).make_env_for_episode(episode_idx)

    print(f"\nTask: {task_id}, Episode: {episode_idx}")
    print(f"Task goal: {task_goal}")
    print(f"Total actions after dedup: {len(action_sequence)}")

    obs, _ = env.reset()
    frames = _extract_frames(
        obs, is_video_demo_fn=lambda i, n=len(obs["front_rgb_list"]): i < n - 1
    )

    outcome = "unknown"
    for seq_idx, action in enumerate(action_sequence):
        try:
            obs, _, terminated, truncated, info = env.step(action)
            frames.extend(_extract_frames(obs))
        except Exception as e:
            print(f"Error at seq_idx {seq_idx}: {e}")
            break

        if GUI_RENDER:
            env.render()
        if terminated or truncated:
            outcome = info.get("status", "unknown")
            print(f"Outcome: {outcome}")
            break

    env.close()
    path = _save_video(frames, task_id, episode_idx, task_goal, outcome, action_space_type)
    print(f"Saved video to {path}\n")


def replay(
    h5_data_dir: str = "/data/hongzefu/data_0226",
    action_space_type: ActionSpaceType = "ee_pose",
    replay_number: int = 10,
) -> None:
    """Replay episodes from HDF5 dataset files and save rollout videos."""
    for task_id in BenchmarkEnvBuilder.get_task_list():
        file_path = Path(h5_data_dir) / f"record_dataset_{task_id}.h5"

        if not file_path.exists():
            print(f"Skipping {task_id}: file not found: {file_path}")
            continue

        with h5py.File(file_path, "r") as data:
            episode_indices = _get_episode_indices(data)
            for episode_idx in episode_indices[:min(replay_number, len(episode_indices))]:
                process_episode(data, episode_idx, task_id, action_space_type)


if __name__ == "__main__":
    import tyro
    tyro.cli(replay)
