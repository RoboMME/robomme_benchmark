"""
Replay episodes from an HDF5 dataset and save videos.

Read recorded end-effector pose actions (eef_action) from record_dataset_<Task>.h5,
replay them in an environment wrapped by EE_POSE_ACTION_SPACE,
and finally save side-by-side front/wrist camera videos to disk.
"""

import os
from typing import Tuple

import cv2
import h5py
import imageio
import numpy as np

from robomme.robomme_env import *
from robomme.robomme_env.utils import *
from robomme.env_record_wrapper import BenchmarkEnvBuilder
from robomme.robomme_env.utils import EE_POSE_ACTION_SPACE

# --- Config ---
GUI_RENDER = False
REPLAY_VIDEO_DIR = "replay_videos"
VIDEO_FPS = 30
MAX_STEPS = 1000


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


def process_episode(
    h5_file_path: str, episode_idx: int, env_id: str, gui_render: bool = False,
) -> None:
    """Replay one episode in HDF5: read EE pose actions, run the environment, and save video.

    Each worker process opens the HDF5 file independently to avoid cross-process shared file handles.
    """
    with h5py.File(h5_file_path, "r") as env_data:
        episode_data = env_data[f"episode_{episode_idx}"]
        task_goal = episode_data["setup"]["task_goal"][()].decode()
        total_steps = sum(1 for k in episode_data.keys() if k.startswith("timestep_"))

        step_idx = _first_execution_step(episode_data)
        print(f"[ep{episode_idx}] execution start step index: {step_idx}")

        env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="train",
            action_space=EE_POSE_ACTION_SPACE,
            gui_render=gui_render,
        )
        env = env_builder.make_env_for_episode(episode_idx, max_steps=MAX_STEPS)
        print(f"[ep{episode_idx}] task: {env_id}, goal: {task_goal}")

        obs, info = env.reset()
        frames = []
        n_obs = len(obs["front_camera"])
        for i in range(n_obs):
            single_obs = {k: [v[i]] for k, v in obs.items()}
            frames.append(_frame_from_obs(single_obs, is_video_frame=(i < n_obs - 1)))
        print(f"[ep{episode_idx}] initial frame count (demo video + current frame): {len(frames)}")

        outcome = "unknown"
        try:
            while step_idx < total_steps:
                action = np.asarray(
                    episode_data[f"timestep_{step_idx}"]["action"]["eef_action"][()],
                    dtype=np.float32,
                )
                obs, _, terminated, _, info = env.step(action)
                frames.append(_frame_from_obs(obs))

                if gui_render:
                    env.render()

                # TODO: hongze makes this correct
                # there are two many nested lists here, need to flatten them
                if terminated:
                    if info.get("success", False)[-1][-1]:
                        outcome = "success"
                    if info.get("fail", False)[-1][-1]:
                        outcome = "fail"
                    break
                step_idx += 1
        finally:
            env.close()

    # Save replay video
    safe_goal = task_goal.replace(" ", "_").replace("/", "_")
    os.makedirs(REPLAY_VIDEO_DIR, exist_ok=True)
    video_name = f"{outcome}_{env_id}_ep{episode_idx}_{safe_goal}_step-{len(frames)}.mp4"
    video_path = os.path.join(REPLAY_VIDEO_DIR, video_name)
    imageio.mimsave(video_path, frames, fps=VIDEO_FPS)
    print(f"[ep{episode_idx}] Video saved to {video_path}")


def _worker_init(gpu_id_queue) -> None:
    """Pool worker initializer that binds a GPU before CUDA initialization.

    When each worker starts, it takes one GPU ID from the queue and sets env vars,
    ensuring all later CUDA ops in that process run on the assigned GPU.
    """
    gpu_id = gpu_id_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[Worker PID {os.getpid()}] bind GPU {gpu_id}")


def _process_episode_worker(args: Tuple[str, int, str, bool]) -> str:
    """multiprocessing worker entrypoint: unpack args and call process_episode."""
    h5_file_path, episode_idx, env_id, gui_render = args
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    try:
        process_episode(h5_file_path, episode_idx, env_id, gui_render=gui_render)
        return f"OK: {env_id} ep{episode_idx} (GPU {gpu_id})"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"FAIL: {env_id} ep{episode_idx} (GPU {gpu_id}): {e}"


def replay(
    h5_data_dir: str = "/data/hongzefu/dataset_generate",
    num_workers: int = 20,
    gui_render: bool = False,
    gpu_ids: str = "0,1",
) -> None:
    """Iterate through all task HDF5 files in the given directory and replay multiple episodes per env in parallel.

    Args:
        h5_data_dir: Directory containing HDF5 datasets.
        num_workers: Number of parallel workers per env.
        gui_render: Whether to enable GUI rendering (recommended off in multiprocessing).
        gpu_ids: Comma-separated GPU ID list; workers use them in round-robin order.
                 For example, "0,1" alternates assignment between GPU 0 and GPU 1.
    """
    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    gpu_id_list = [int(g.strip()) for g in gpu_ids.split(",")]
    print(f"Using GPUs: {gpu_id_list}, workers: {num_workers}")

    env_id_list = BenchmarkEnvBuilder.get_task_list()
    for env_id in env_id_list:
        file_name = f"record_dataset_{env_id}.h5"
        file_path = os.path.join(h5_data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Skip {env_id}: file does not exist: {file_path}")
            continue

        # Quickly read episode list and close file
        with h5py.File(file_path, "r") as data:
            episode_indices = sorted(
                int(k.split("_")[1])
                for k in data.keys()
                if k.startswith("episode_")
            )
        print(f"task: {env_id}, total {len(episode_indices)} episodes, "
              f"workers: {num_workers}, GPUs: {gpu_id_list}")

        # Build worker argument list
        worker_args = [
            (file_path, ep_idx, env_id, gui_render)
            for ep_idx in episode_indices
        ]

        # Create a new GPU assignment queue for each round; each worker grabs one GPU ID at startup
        gpu_id_queue = ctx.Queue()
        for i in range(num_workers):
            gpu_id_queue.put(gpu_id_list[i % len(gpu_id_list)])

        # Parallel replay (initializer binds GPU when each worker starts)
        with ctx.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(gpu_id_queue,),
        ) as pool:
            results = pool.map(_process_episode_worker, worker_args)

        for r in results:
            print(r)


if __name__ == "__main__":
    import tyro
    tyro.cli(replay)
