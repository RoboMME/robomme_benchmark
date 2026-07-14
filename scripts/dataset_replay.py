"""
Replay episodes from HDF5 datasets and save rollout videos.
Loads recorded actions from record_dataset_<Task>.h5, steps the environment
"""

import contextlib
import os
import multiprocessing as mp
import queue
import time
import traceback
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
REPLAY_VIDEO_DIR = "runs/replay_videos"
VIDEO_FPS = 30
VIDEO_BORDER_COLOR = (255, 0, 0)
VIDEO_BORDER_THICKNESS = 10
GPU_DEVICE_IDS = (0, 1)
SPAWN_START_TIMEOUT_SECONDS = 300
TASK_TIMEOUT_SECONDS = 14_400

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
) -> dict[str, Any]:
    """Replay one episode and return an auditable result for its task worker."""
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
    step_error: str | None = None
    for seq_idx, action in enumerate(action_sequence):
        obs, _, terminated, truncated, info = env.step(action)
        frames.extend(_extract_frames(obs))

        if GUI_RENDER:
            env.render()
        if info.get("status") == "error":
            step_error = str(info.get("error_message", "environment returned status=error"))
            print(f"Error at seq_idx {seq_idx}: {step_error}")
            break
        if terminated or truncated:
            outcome = info.get("status", "unknown")
            print(
                f"Outcome: {outcome} | task_id: {task_id} | episode: {episode_idx}"
            )
            break

    env.close()
    video_path = _save_video(
        frames, task_id, episode_idx, task_goal, outcome, action_space_type
    )
    print(f"Saved video to {video_path}\n")
    return {
        "episode_idx": episode_idx,
        "outcome": outcome,
        "step_error": step_error,
        "video_path": str(video_path),
    }


def _replay_task(
    task_id: str,
    h5_data_dir: str,
    action_space_type: ActionSpaceType,
    replay_number: int,
) -> dict[str, Any]:
    """Open one task HDF5 file inside its owning child process and replay it."""
    file_path = Path(h5_data_dir) / f"record_dataset_{task_id}.h5"
    if not file_path.is_file():
        raise FileNotFoundError(f"Missing HDF5 file for {task_id}: {file_path}")

    with h5py.File(file_path, "r") as data:
        episode_indices = _get_episode_indices(data)[:replay_number]
        episodes = [
            process_episode(data, episode_idx, task_id, action_space_type)
            for episode_idx in episode_indices
        ]

    return {
        "task_id": task_id,
        "h5_path": str(file_path),
        "episodes_requested": replay_number,
        "episodes_replayed": len(episodes),
        "episodes": episodes,
    }


def _task_worker(
    task_id: str,
    gpu_device: int,
    h5_data_dir: str,
    action_space_type: ActionSpaceType,
    replay_number: int,
    replay_log_dir: str,
    ready_queue,
    release_event,
    result_queue,
) -> None:
    """Wait at the parent barrier, then run exactly one task and report its result."""
    log_path = Path(replay_log_dir) / f"{task_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any]

    with log_path.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            try:
                print(f"task_id={task_id} assigned_gpu={gpu_device}")
                ready_queue.put({"task_id": task_id, "gpu_device": gpu_device})
                if not release_event.wait(SPAWN_START_TIMEOUT_SECONDS):
                    raise TimeoutError("parent did not release the spawn barrier")
                task_result = _replay_task(
                    task_id, h5_data_dir, action_space_type, replay_number
                )
                result = {
                    "task_id": task_id,
                    "gpu_device": gpu_device,
                    "status": "ok",
                    "log_path": str(log_path),
                    **task_result,
                }
            except BaseException as exc:
                traceback.print_exc()
                result = {
                    "task_id": task_id,
                    "gpu_device": gpu_device,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "log_path": str(log_path),
                }

    result_queue.put(result)


def _stop_processes(processes: list[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join()


def _write_replay_summary(replay_log_dir: Path, summary: dict[str, Any]) -> Path:
    summary_path = replay_log_dir / "replay_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary_path


def _run_parallel_replay(
    task_ids: list[str],
    h5_data_dir: str,
    action_space_type: ActionSpaceType,
    replay_number: int,
    replay_log_dir: str,
) -> None:
    """Spawn one synchronized process per task and fail if any task is incomplete."""
    if len(task_ids) != 16:
        raise RuntimeError(f"Expected exactly 16 RoboMME tasks, found {len(task_ids)}")

    log_dir = Path(replay_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    ready_queue = ctx.Queue()
    result_queue = ctx.Queue()
    release_event = ctx.Event()
    processes: list[mp.Process] = []
    assignments = {
        task_id: GPU_DEVICE_IDS[index % len(GPU_DEVICE_IDS)]
        for index, task_id in enumerate(task_ids)
    }

    had_visible_devices = "CUDA_VISIBLE_DEVICES" in os.environ
    previous_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        for task_id in task_ids:
            gpu_device = assignments[task_id]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
            process = ctx.Process(
                target=_task_worker,
                name=f"replay-{task_id}",
                args=(
                    task_id,
                    gpu_device,
                    h5_data_dir,
                    action_space_type,
                    replay_number,
                    str(log_dir),
                    ready_queue,
                    release_event,
                    result_queue,
                ),
            )
            process.start()
            processes.append(process)
    finally:
        if had_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible_devices or ""
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    ready_tasks: set[str] = set()
    startup_deadline = time.monotonic() + SPAWN_START_TIMEOUT_SECONDS
    while len(ready_tasks) < len(task_ids) and time.monotonic() < startup_deadline:
        try:
            ready_message = ready_queue.get(
                timeout=max(0.1, startup_deadline - time.monotonic())
            )
        except queue.Empty:
            break
        ready_tasks.add(ready_message["task_id"])

    startup_failures = [
        f"{process.name} exited before the barrier with code {process.exitcode}"
        for process in processes
        if process.exitcode is not None and process.exitcode != 0
    ]
    missing_ready = sorted(set(task_ids) - ready_tasks)
    if missing_ready:
        startup_failures.append(f"workers not ready before timeout: {missing_ready}")
    if startup_failures:
        _stop_processes(processes)
        summary_path = _write_replay_summary(
            log_dir,
            {
                "task_ids": task_ids,
                "gpu_assignments": assignments,
                "ready_tasks": sorted(ready_tasks),
                "results": [],
                "failures": startup_failures,
            },
        )
        raise RuntimeError(f"Parallel replay startup failed; summary: {summary_path}")

    release_event.set()
    completion_deadline = time.monotonic() + TASK_TIMEOUT_SECONDS
    for process in processes:
        process.join(timeout=max(0.0, completion_deadline - time.monotonic()))

    timed_out = [process.name for process in processes if process.is_alive()]
    if timed_out:
        _stop_processes(processes)

    results: dict[str, dict[str, Any]] = {}
    result_deadline = time.monotonic() + 5
    while len(results) < len(task_ids) and time.monotonic() < result_deadline:
        try:
            result = result_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        results[result["task_id"]] = result

    failures = [
        f"timed out: {process_name}" for process_name in timed_out
    ] + [
        f"{process.name} exited with code {process.exitcode}"
        for process in processes
        if process.exitcode not in (0, None)
    ] + [
        f"missing worker result: {task_id}"
        for task_id in task_ids
        if task_id not in results
    ] + [
        f"{task_id}: {result.get('error', 'worker returned error')}"
        for task_id, result in results.items()
        if result["status"] != "ok"
    ]

    summary_path = _write_replay_summary(
        log_dir,
        {
            "task_ids": task_ids,
            "gpu_assignments": assignments,
            "ready_tasks": sorted(ready_tasks),
            "process_exit_codes": {process.name: process.exitcode for process in processes},
            "results": [results[task_id] for task_id in task_ids if task_id in results],
            "failures": failures,
        },
    )
    if failures:
        raise RuntimeError(f"Parallel replay failed; summary: {summary_path}")

    print(f"Parallel replay completed for 16 tasks. Summary: {summary_path}")


def replay(
    h5_data_dir: str = "data/robomme_data_h5",
    action_space_type: ActionSpaceType = "joint_angle",
    replay_number: int = 10,
    replay_log_dir: str = "runs/replay_logs",
) -> None:
    """Synchronously start 16 task workers, balanced across GPU 0 and GPU 1."""
    _run_parallel_replay(
        list(BenchmarkEnvBuilder.get_task_list()),
        h5_data_dir,
        action_space_type,
        replay_number,
        replay_log_dir,
    )


if __name__ == "__main__":
    import tyro
    tyro.cli(replay)
