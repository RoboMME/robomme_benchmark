"""
从 HDF5 数据集回放 episode 并保存视频。

从 record_dataset_<Task>.h5 中读取已录制的末端执行器位姿动作 (eef_action)，
输入到以 EE_POSE_ACTION_SPACE 包裹的环境中执行回放，
最后将前置/腕部相机的左右拼接视频写入磁盘。
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
    """回放 HDF5 中的一个 episode：读取 EE 位姿动作、环境执行、保存视频。

    每个 worker 进程独立打开 HDF5 文件，避免跨进程共享文件句柄。
    """
    with h5py.File(h5_file_path, "r") as env_data:
        episode_data = env_data[f"episode_{episode_idx}"]
        task_goal = episode_data["setup"]["task_goal"][()].decode()
        total_steps = sum(1 for k in episode_data.keys() if k.startswith("timestep_"))

        step_idx = _first_execution_step(episode_data)
        print(f"[ep{episode_idx}] 执行起始步索引: {step_idx}")

        env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="train",
            action_space=EE_POSE_ACTION_SPACE,
            gui_render=gui_render,
        )
        env, _, _ = env_builder.make_env_for_episode(episode_idx)
        print(f"[ep{episode_idx}] 任务: {env_id}, 目标: {task_goal}")

        obs, info = env.reset()
        frames = []
        n_obs = len(obs["front_camera"])
        for i in range(n_obs):
            single_obs = {k: [v[i]] for k, v in obs.items()}
            frames.append(_frame_from_obs(single_obs, is_video_frame=(i < n_obs - 1)))
        print(f"[ep{episode_idx}] 初始帧数（视频 + 当前帧）: {len(frames)}")

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
                if terminated[-1]:
                    if info.get("success", False)[-1][-1]:
                        outcome = "success"
                    if info.get("fail", False)[-1][-1]:
                        outcome = "fail"
                    break
                step_idx += 1
        finally:
            env.close()

    # 保存回放视频
    safe_goal = task_goal.replace(" ", "_").replace("/", "_")
    os.makedirs(REPLAY_VIDEO_DIR, exist_ok=True)
    video_name = f"{outcome}_{env_id}_ep{episode_idx}_{safe_goal}_step-{len(frames)}.mp4"
    video_path = os.path.join(REPLAY_VIDEO_DIR, video_name)
    imageio.mimsave(video_path, frames, fps=VIDEO_FPS)
    print(f"[ep{episode_idx}] 视频已保存到 {video_path}")


def _worker_init(gpu_id_queue) -> None:
    """Pool worker 初始化函数，在 CUDA 初始化前绑定 GPU。

    每个 worker 进程启动时从队列中取出一个 GPU ID 并设置环境变量，
    确保该进程的所有后续 CUDA 操作都在指定 GPU 上执行。
    """
    gpu_id = gpu_id_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[Worker PID {os.getpid()}] 绑定 GPU {gpu_id}")


def _process_episode_worker(args: Tuple[str, int, str, bool]) -> str:
    """multiprocessing worker 入口，解包参数并调用 process_episode。"""
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
    """遍历指定目录下所有任务的 HDF5 文件，对每个 env 的多 episode 并行回放。

    参数:
        h5_data_dir: HDF5 数据集所在目录。
        num_workers: 每个 env 的并行 worker 数量。
        gui_render: 是否开启 GUI 渲染（多进程下建议关闭）。
        gpu_ids: 逗号分隔的 GPU ID 列表，worker 按轮询方式交替使用。
                 例如 "0,1" 表示在 GPU 0 和 GPU 1 之间交替分配。
    """
    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    gpu_id_list = [int(g.strip()) for g in gpu_ids.split(",")]
    print(f"使用 GPU: {gpu_id_list}, workers: {num_workers}")

    env_id_list = BenchmarkEnvBuilder.get_task_list()
    for env_id in env_id_list:
        file_name = f"record_dataset_{env_id}.h5"
        file_path = os.path.join(h5_data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"跳过 {env_id}: 文件不存在: {file_path}")
            continue

        # 快速读取 episode 列表后关闭文件
        with h5py.File(file_path, "r") as data:
            episode_indices = sorted(
                int(k.split("_")[1])
                for k in data.keys()
                if k.startswith("episode_")
            )
        print(f"任务: {env_id}, 共 {len(episode_indices)} 个 episode, "
              f"workers: {num_workers}, GPUs: {gpu_id_list}")

        # 构造 worker 参数列表
        worker_args = [
            (file_path, ep_idx, env_id, gui_render)
            for ep_idx in episode_indices
        ]

        # 每轮创建新的 GPU 分配队列，worker 启动时各取一个 GPU ID
        gpu_id_queue = ctx.Queue()
        for i in range(num_workers):
            gpu_id_queue.put(gpu_id_list[i % len(gpu_id_list)])

        # 并行回放（initializer 在每个 worker 进程启动时绑定 GPU）
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
