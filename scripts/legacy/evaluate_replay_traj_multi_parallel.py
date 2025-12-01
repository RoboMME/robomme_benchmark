import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import sys
import re
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure the package root is importable when running as a script
_ROOT = os.path.abspath(os.path.dirname(__file__))
_PARENT = os.path.dirname(_ROOT)
for _path in (_PARENT, _ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import sapien
import gymnasium as gym

from historybench.env_record_wrapper import DemonstrationWrapper


from historybench.HistoryBench_env import *
from historybench.HistoryBench_env.util import *
import torch
import h5py

DEFAULT_ENVS = [
    # "PickXtimes",
    # "StopCube",
    # "SwingXtimes",
    # "BinFill",

    # "VideoUnmaskSwap",
    # "VideoUnmask",
    # "ButtonUnmaskSwap",
    # "ButtonUnmask",

    # "VideoRepick",
    # "VideoPlaceButton",
    # "VideoPlaceOrder",
    # "PickHighlight",

    "InsertPeg",
    # "MoveCube",
    "PatternLock",
    "RouteStick",

]


def _parse_args():
    """Parse command line options for replay execution."""

    parser = argparse.ArgumentParser(description="Replay stored datasets inside HistoryBench environments.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        #default=_PARENT,
        default="/data/hongzefu/dataset_generate",
        help="Directory that contains record_dataset_*.h5 files. Defaults to repository parent.",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=DEFAULT_ENVS,
        help="Specific environment IDs to replay. Defaults to a predefined list of HistoryBench tasks.",
    )
    parser.add_argument(  
        "--num-episodes",
        type=int,
        default=-1,
        help="Maximum number of episodes to replay for each environment. Use a negative value to replay all.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3000,
        help="Maximum number of steps to run without receiving demonstration data.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Enable GUI rendering instead of headless RGB rendering.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Optional offset added to the stored seed before replaying.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Maximum number of episodes to evaluate in parallel for each environment.",
    )
    return parser.parse_args()


def _discover_env_ids(dataset_dir):
    """Return a sorted list of env IDs inferred from dataset filenames."""

    env_ids = []
    prefix = "record_dataset_"
    suffix = ".h5"
    for entry in sorted(os.listdir(dataset_dir)):
        if entry.startswith(prefix) and entry.endswith(suffix):
            env_ids.append(entry[len(prefix) : -len(suffix)])
    return env_ids


def _as_bool(value):
    """Safely convert tensors/arrays/None to bool without raising."""

    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return False
        flat = value.detach().cpu().reshape(-1)
        return bool(flat[0].item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
        return bool(np.reshape(value, -1)[0].item())
    return bool(value) if value is not None else False


def _prepare_frame(frame):
    """Convert observation arrays to uint8 RGB frames."""

    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        max_val = float(np.max(frame)) if frame.size else 0.0
        if max_val <= 1.0:
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    return frame


def _add_border(frame, color=(255, 0, 0), thickness=6):
    """Draw a solid colored border around the frame."""

    if frame.ndim != 3 or frame.shape[2] < 3:
        return frame

    annotated = frame.copy()
    h, w = annotated.shape[:2]
    annotated[:thickness, :] = color
    annotated[-thickness:, :] = color
    annotated[:, :thickness] = color
    annotated[:, -thickness:] = color
    return annotated


def _prepare_demonstration_frames(demo_data, border_color=(255, 0, 0), border_thickness=6):
    """Return demonstration frames with wrist views and a border annotation."""

    if not demo_data:
        return []

    base_frames = demo_data.get("frames", []) or []
    wrist_frames = demo_data.get("wrist_frames", []) or []

    annotated_frames = []
    for idx, base_frame in enumerate(base_frames):
        prepared_base = _prepare_frame(base_frame)
        prepared_wrist = None
        if idx < len(wrist_frames):
            prepared_wrist = _prepare_frame(wrist_frames[idx])

        if prepared_wrist is not None:
            try:
                combined = np.concatenate((prepared_base, prepared_wrist), axis=1)
            except ValueError:
                combined = prepared_base
        else:
            combined = prepared_base

        annotated_frames.append(_add_border(combined, color=border_color, thickness=border_thickness))

    return annotated_frames


def _save_episode_video(frames, env_id, episode_idx, step_idx, prefix="success", fps=20):
    """Video writing handled by the DemonstrationWrapper; disabled here."""
    return


def _replay_episode(
    env_id,
    episode,
    dataset_path,
    render_mode,
    gui_render,
    max_steps,
    seed_offset,
):
    """Worker process to replay a single episode."""

    with h5py.File(dataset_path, "r") as dataset:
        env_dataset = dataset[f"env_{env_id}"]
        episode_dataset = env_dataset[f"episode_{episode}"]
        seed = int(episode_dataset["setup"]["seed"][()]) + seed_offset
        language_goal = episode_dataset["setup"]["language goal"][()]

        print(f"[{env_id}] Episode {episode} | goal: {language_goal!r} | seed: {seed}")

        env = gym.make(
            env_id,
            obs_mode="rgb+depth+segmentation",
            control_mode="pd_joint_pos",
            render_mode=render_mode,
            reward_mode="dense",
            HistoryBench_seed=seed,
            max_episode_steps=99999
        )
        env = DemonstrationWrapper(
            env,
            max_steps_without_demonstration=max_steps,
            gui_render=gui_render,
            save_video=True
        )

        try:
            env.reset()
            demonstration_data = env.demonstration_data
            demonstration_frames = _prepare_demonstration_frames(demonstration_data)
            if demonstration_frames:
                print(
                    f"[{env_id}] Episode {episode} prepended {len(demonstration_frames)} demonstration frames with border."
                )

            timestep_indexes = sorted(
                int(m.group(1))
                for k in episode_dataset.keys()
                if (m := re.search(r'^record_timestep_(\d+)$', k))
            )

            episode_video_frames = list(demonstration_frames)

            for step in timestep_indexes:
                #print(f"[{env_id}] Episode {episode} -> step {step}")
                timestep_group = episode_dataset[f"record_timestep_{step}"]
                if _as_bool(timestep_group['demonstration'][()]):
                    continue

                action = np.asarray(timestep_group["action"][()])

                obs, reward, terminated, truncated, info = env.step(action)

                image = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
                wrist_image = obs['sensor_data']['hand_camera']['rgb'][0].cpu().numpy()

                base_frame = _prepare_frame(image)
                wrist_frame = _prepare_frame(wrist_image)

                try:
                    combined_frame = np.concatenate((base_frame, wrist_frame), axis=1)
                except ValueError:
                    combined_frame = base_frame

                episode_video_frames.append(combined_frame)

                success_flag = _as_bool(info.get("success"))
                fail_flag = _as_bool(info.get("fail"))

                if success_flag:
                    print(f"[{env_id}] Episode {episode} success detail: {info.get('success')}")

                if gui_render:
                    env.render()

                if truncated:
                    print(f"[{env_id}] Episode {episode} truncated at step {step}")
                    _save_episode_video(episode_video_frames, env_id, episode, step, prefix="fail")
                    break
                if terminated:
                    obs, reward, terminated, truncated, info = env.step(action)#highlight显示
                    if success_flag:
                        print(f"[{env_id}] Episode {episode} succeeded at step {step}")
                        _save_episode_video(episode_video_frames, env_id, episode, step, prefix="success")
                    else:
                        if fail_flag:
                            print(f"[{env_id}] Episode {episode} failed at step {step}")
                        else:
                            print(f"[{env_id}] Episode {episode} terminated without explicit success flag")
                        _save_episode_video(episode_video_frames, env_id, episode, step, prefix="fail")
                    break
        finally:
            env.close()

    return episode


def main():
    """
    Main function to run the simulation and record data for multiple seeds.
    """

    args = _parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    env_id_list = args.envs if args.envs else _discover_env_ids(dataset_dir)
    if not env_id_list:
        print(f"No dataset files discovered in {dataset_dir}; nothing to replay.")
        return

    num_episodes = None if args.num_episodes is not None and args.num_episodes < 0 else args.num_episodes
    gui_render = args.gui
    max_steps = args.max_steps
    seed_offset = args.seed_offset

    render_mode = "human" if gui_render else "rgb_array"

    for env_id in env_id_list:
        dataset_path = os.path.join(dataset_dir, f"record_dataset_{env_id}.h5")
        if not os.path.exists(dataset_path):
            print(f"Dataset file missing for {env_id}: {dataset_path}, skipping.")
            continue

        print(f"=== Replaying dataset for {env_id} ({dataset_path}) ===")

        with h5py.File(dataset_path, "r") as dataset:
            env_dataset_key = f"env_{env_id}"
            if env_dataset_key not in dataset:
                print(f"No group '{env_dataset_key}' in {dataset_path}; skipping.")
                continue

            env_dataset = dataset[env_dataset_key]
            episode_indices = sorted(
                int(k.split("_")[1])
                for k in env_dataset.keys()
                if k.startswith("episode_")
            )

        if not episode_indices:
            print(f"No episodes found in dataset for {env_id}, skipping.")
            continue

        if num_episodes is not None:
            selected_episode_indices = episode_indices[:num_episodes]
        else:
            selected_episode_indices = episode_indices

        worker_count = max(1, min(args.max_workers, len(selected_episode_indices)))
        if gui_render and worker_count > 1:
            print("GUI render requested; forcing single worker to avoid multiple windows.")
            worker_count = 1

        if worker_count == 1:
            for episode in selected_episode_indices:
                _replay_episode(
                    env_id,
                    episode,
                    dataset_path,
                    render_mode,
                    gui_render,
                    max_steps,
                    seed_offset,
                )
        else:
            print(f"Running {env_id} with up to {worker_count} parallel episodes...")
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_to_episode = {
                    executor.submit(
                        _replay_episode,
                        env_id,
                        episode,
                        dataset_path,
                        render_mode,
                        gui_render,
                        max_steps,
                        seed_offset,
                    ): episode
                    for episode in selected_episode_indices
                }

                for future in as_completed(future_to_episode):
                    episode = future_to_episode[future]
                    try:
                        finished_episode = future.result()
                        print(f"[{env_id}] Episode {finished_episode} finished.")
                    except Exception as exc:
                        print(f"[{env_id}] Episode {episode} failed with error: {exc}")

if __name__ == "__main__":
    main()
