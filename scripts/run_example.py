"""
Run a single benchmark episode and save the rollout as a video.

Use this script to sanity-check the environment and action space
"""

from pathlib import Path
from typing import Literal

import cv2
import imageio
import numpy as np
import torch
import tyro

from robomme.env_record_wrapper import BenchmarkEnvBuilder
from robomme.robomme_env.utils import generate_sample_actions

GUI_RENDER = False
VIDEO_FPS = 30
VIDEO_OUTPUT_DIR = "runs/sample_run_videos"
MAX_STEPS = 300
EPISODE_LIMITS = {"train": 100, "test": 50, "val": 50}
VIDEO_BORDER_COLOR = (255, 0, 0)
VIDEO_BORDER_THICKNESS = 10
TaskID = Literal[
    "BinFill", "PickXtimes", "SwingXtimes", "StopCube",
    "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap",
    "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
    "All",
]
ActionSpaceType = Literal["joint_angle", "ee_pose", "waypoint", "multi_choice"]
DatasetType = Literal["train", "test", "val"]


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


def _validate_episode_index(episode_idx: int, dataset: DatasetType) -> None:
    if episode_idx == -1:
        return
    limit = EPISODE_LIMITS[dataset]
    if not 0 <= episode_idx < limit:
        raise ValueError(
            f"Invalid episode_idx {episode_idx} for '{dataset}'; allowed: [0, {limit})"
        )


def _save_video(
    frames: list[np.ndarray],
    task_id: str,
    episode_idx: int,
    action_space_type: str,
    task_goal: str,
) -> Path:
    video_dir = Path(VIDEO_OUTPUT_DIR) / action_space_type
    video_dir.mkdir(parents=True, exist_ok=True)
    path = video_dir / f"{task_id}_ep{episode_idx}_{task_goal}.mp4"
    imageio.mimsave(str(path), frames, fps=VIDEO_FPS)
    return path


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



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    dataset: DatasetType = "test",
    task_id: TaskID = "PickXtimes",
    action_space_type: ActionSpaceType = "joint_angle",
    episode_idx: int = 0,
) -> None:
    """
    Run a single benchmark episode and save the rollout as a video.

    Args:
        action_space_type: Type of action space to use.
        dataset: Dataset split (train / test / val).
        task_id: Task identifier, or "All" to run every task.
        episode_idx: Episode index (-1 = All episodes).
    """
    task_ids = (
        BenchmarkEnvBuilder.get_task_list() if task_id == "All" else [task_id]
    )
    _validate_episode_index(episode_idx, dataset)

    for tid in task_ids:
        env_builder = BenchmarkEnvBuilder(
            env_id=tid,
            dataset=dataset,
            action_space=action_space_type,
            gui_render=GUI_RENDER,
            max_steps=MAX_STEPS,
        )
        episodes = (
            list(range(env_builder.get_episode_num()))
            if episode_idx == -1
            else [episode_idx]
        )

        for ep in episodes:
            print(f"\nRunning task: {tid}, episode: {ep}, action_space: {action_space_type}, dataset: {dataset}")
            env = env_builder.make_env_for_episode(ep)
            obs, info = env.reset()
            
            if action_space_type == "multi_choice":
                print(f"Available multi choices: {info['available_multi_choices']}")
                
            task_goal = info["task_goal"][0]
            print(f"Task goal: {task_goal}")

            frames = _extract_frames(
                obs, is_video_demo_fn=lambda i, n=len(obs["front_rgb_list"]): i < n - 1
            )

            action_gen = generate_sample_actions(action_space_type, env=env)
            for action in action_gen:
                print(f"Action: {action}")
                obs, _, terminated, truncated, info = env.step(action)
                status = info.get("status", "unknown")
                if status == "error":
                    print(f"Step error: {info.get('error_message', 'unknown error')}")
                    break
                frames.extend(_extract_frames(obs))

                if GUI_RENDER:
                    env.render()
                if terminated or truncated:
                    print(f"Outcome: {status} | env_id: {tid} | episode: {ep}")
                    break

            env.close()
            path = _save_video(frames, tid, ep, action_space_type, task_goal)
            print(f"Saved video: {path}\n")


if __name__ == "__main__":
    tyro.cli(main)
