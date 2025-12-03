import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import sapien
from typing import Any, Dict, Iterable, List, Optional
import h5py

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym
from gymnasium.utils.save_video import save_video

from historybench.env_record_wrapper import HistoryBenchRecordWrapper, FailsafeTimeout
from historybench.HistoryBench_env import *


from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver

from mani_skill.examples.motionplanning.panda.motionplanner_stick import PandaStickMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)

# from util import *
import torch



# 所有模块名称的列表
# DEFAULT_ENVS = [
# "BinFill",
#    "PickXtimes",
#     "SwingXtimes",
#    "ButtonUnmask",
#  "VideoUnmask",
#    "PickHighlight",
#     "VideoUnmaskSwap",
# "VideoRepick",
#  "VideoPlaceButton",
# "VideoPlaceOrder",
#  "ButtonUnmaskSwap",
# 'MoveCube',
# "InsertPeg",
# "StopCube",
# 'PatternLock',
# 'RouteStick'

# ]


DEFAULT_ENVS =[
# "PickXtimes",
# "StopCube",
# "SwingXtimes",
# "BinFill",

# "VideoUnmaskSwap",
# "VideoUnmask",
# "ButtonUnmaskSwap",
# "ButtonUnmask",

#  "VideoRepick",
# "VideoPlaceButton",
# "VideoPlaceOrder",
# "PickHighlight",

"InsertPeg",
'MoveCube',
"PatternLock",
"RouteStick"

]
ENV_ID_TO_CODE = {name: idx + 1 for idx, name in enumerate(DEFAULT_ENVS)}
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1]

def _tensor_to_bool(value) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)


def _split_episode_indices(num_episodes: int, max_chunks: int) -> List[List[int]]:
    if num_episodes <= 0:
        return []

    chunk_count = min(max_chunks, num_episodes)
    base_size, remainder = divmod(num_episodes, chunk_count)

    chunks: List[List[int]] = []
    start = 0
    for chunk_idx in range(chunk_count):
        stop = start + base_size + (1 if chunk_idx < remainder else 0)
        chunks.append(list(range(start, stop)))
        start = stop

    return chunks


def _get_difficulty_from_ratio(episode: int, difficulty_ratio: Optional[str]) -> Optional[str]:
    """Calculate difficulty based on episode index and ratio.

    Args:
        episode: Episode index
        difficulty_ratio: Ratio string like "211" meaning easy:medium:hard = 2:1:1

    Returns:
        One of "easy", "medium", "hard", or None if no ratio specified
    """
    if not difficulty_ratio:
        return None

    # Parse ratio string (e.g., "211" -> [2, 1, 1])
    try:
        ratios = [int(c) for c in difficulty_ratio]
        if len(ratios) != 3:
            raise ValueError("Difficulty ratio must have 3 digits")
    except (ValueError, TypeError):
        raise ValueError(f"Invalid difficulty ratio: {difficulty_ratio}. Expected 3 digits like '211'")

    # Create difficulty sequence
    difficulties = ["easy"] * ratios[0] + ["medium"] * ratios[1] + ["hard"] * ratios[2]
    total = sum(ratios)

    if total == 0:
        return None

    # Map episode to difficulty using modulo
    return difficulties[episode % total]


def _run_episode_attempt(
    env_id: str,
    episode: int,
    seed: int,
    temp_dataset_path: Path,
    save_video: bool,
    difficulty: Optional[str],
) -> bool:
    """Run a single episode attempt and report success/failure."""
    print(f"--- Running simulation for episode:{episode}, seed:{seed}, env: {env_id} ---")

    env = gym.make(
        env_id,
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        HistoryBench_seed=seed,
        max_episode_steps=200,
        HistoryBench_difficulty=difficulty,
    )
    env = HistoryBenchRecordWrapper(
        env,
        HistoryBench_dataset=str(temp_dataset_path),
        HistoryBench_env=env_id,
        HistoryBench_episode=episode,
        HistoryBench_seed=seed,
        save_video=save_video,

    )

    episode_successful = False

    try:
        env.reset()

        if env_id == "PatternLock" or env_id == "RouteStick":
            planner = PandaStickMotionPlanningSolver(
                env,
                debug=False,
                vis=False,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                joint_vel_limits=0.3,
            )
        else:
            planner = PandaArmMotionPlanningSolver(
                env,
                debug=False,
                vis=False,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
            )

        env.unwrapped.evaluate()
        tasks = list(getattr(env.unwrapped, "task_list", []) or [])

        if not tasks:
            print("No tasks defined for this environment; skipping execution")
            episode_successful = True
        else:
            print(f"{env_id}: Task list has {len(tasks)} tasks")

            for idx, task_entry in enumerate(tasks):
                task_name = task_entry.get("name", f"Task {idx}")
                print(f"Executing task {idx + 1}/{len(tasks)}: {task_name}")

                solve_callable = task_entry.get("solve")
                if not callable(solve_callable):
                    raise ValueError(
                        f"Task '{task_name}' must supply a callable 'solve'."
                    )

                try:
                    env.unwrapped.evaluate(solve_complete_eval=True)
                    solve_callable(env, planner)
                    evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
                except FailsafeTimeout as exc:
                    print(f"Failsafe: {exc}")
                    break

                fail_flag = evaluation.get("fail", False)
                success_flag = evaluation.get("success", False)

                if _tensor_to_bool(success_flag):
                    print("All tasks completed successfully.")
                    episode_successful = True
                    break

                if _tensor_to_bool(fail_flag):
                    print("Encountered failure condition; stopping task sequence.")
                    break

            else:
                evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
                episode_successful = _tensor_to_bool(evaluation.get("success", False))

        # Prefer the wrapper's own success signal in case evaluation misses it
        episode_successful = episode_successful or _tensor_to_bool(
            getattr(env, "episode_success", False)
        )

    finally:
        env.close()

    status_text = "SUCCESS" if episode_successful else "FAILED"
    print(
        f"--- Finished Running simulation for episode:{episode}, seed:{seed}, env: {env_id} [{status_text}] ---"
    )

    return episode_successful


def run_env_dataset(
    env_id: str,
    episode_indices: Iterable[int],
    temp_folder: Path,
    save_video: bool,
    difficulty_ratio: Optional[str],
) -> List[Dict[str, Any]]:
    """Run dataset generation and save to a temporary folder.

    Args:
        env_id: Environment ID
        episode_indices: List of episode indices to run
        temp_folder: Temporary folder to save episode files
        save_video: Whether to save videos
        difficulty_ratio: Optional difficulty ratio string like "211" for easy:medium:hard = 2:1:1

    Returns:
        List of episode metadata records produced by this batch
    """
    temp_folder.mkdir(parents=True, exist_ok=True)
    episode_indices = list(episode_indices)
    if not episode_indices:
        return []

    if env_id not in ENV_ID_TO_CODE:
        raise ValueError(f"Environment {env_id} missing from ENV_ID_TO_CODE mapping")
    env_code = ENV_ID_TO_CODE[env_id]

    # Use a temporary h5 file path for the wrapper
    # The wrapper will create individual episode files in a subfolder
    temp_dataset_path = temp_folder / f"temp_chunk.h5"
    episode_records: List[Dict[str, Any]] = []

    for episode in episode_indices:
        # Calculate difficulty for this episode based on ratio
        difficulty = _get_difficulty_from_ratio(episode, difficulty_ratio)
        if difficulty:
            print(f"Episode {episode} assigned difficulty: {difficulty}")

        base_seed = env_code * 1000 + (episode % 100) * 10 -1 + 20000  # env (2 digits) | episode (2 digits) | attempt (1 digit) #add 20000 for test
        attempt = 1
        while True:
            seed = base_seed + (attempt % 10)  # rightmost digit cycles with attempts
            success = _run_episode_attempt(
                env_id=env_id,
                episode=episode,
                seed=seed,
                temp_dataset_path=temp_dataset_path,
                save_video=save_video,
                difficulty=difficulty,
            )
            if success:
                episode_records.append(
                    {
                        "task": env_id,
                        "episode": episode,
                        "seed": seed,
                        "difficulty": difficulty,
                    }
                )
                break

            attempt += 1
            print(
                f"Episode {episode} failed; retrying with new seed {seed} (attempt {attempt})"
            )

    return episode_records


def _merge_dataset_from_folder(
    env_id: str,
    temp_folder: Path,
    final_dataset_path: Path,
) -> None:
    """Merge all episode files from temporary folder into final dataset.

    Args:
        env_id: Environment ID
        temp_folder: Temporary folder containing episode files
        final_dataset_path: Final output HDF5 file path
    """
    if not temp_folder.exists() or not temp_folder.is_dir():
        print(f"Warning: Temporary folder {temp_folder} does not exist")
        return

    final_dataset_path.parent.mkdir(parents=True, exist_ok=True)

    # Find the subfolder created by HistoryBenchRecordWrapper
    # It creates directories with suffix "_hdf5_files"
    hdf5_folders = list(temp_folder.glob("*_hdf5_files"))

    if not hdf5_folders:
        print(f"Warning: No HDF5 folders found in {temp_folder}")
        return

    print(f"Merging episodes from {temp_folder} into {final_dataset_path}")

    with h5py.File(final_dataset_path, "a") as final_file:
        for hdf5_folder in sorted(hdf5_folders):
            # Get all h5 files in the folder
            h5_files = sorted(hdf5_folder.glob("*.h5"))

            if not h5_files:
                print(f"Warning: No h5 files found in {hdf5_folder}")
                continue

            print(f"Found {len(h5_files)} episode files in {hdf5_folder.name}")

            # Merge each episode file
            for h5_file in h5_files:
                print(f"  - Merging {h5_file.name}")
                with h5py.File(h5_file, "r") as episode_file:
                    for env_group_name, src_env_group in episode_file.items():
                        if env_group_name not in final_file:
                            final_file.copy(src_env_group, env_group_name)
                            continue

                        dest_env_group = final_file[env_group_name]
                        if not isinstance(dest_env_group, h5py.Group):
                            print(f"    Warning: {env_group_name} is not a group, skipping...")
                            continue

                        for episode_name in src_env_group.keys():
                            if episode_name in dest_env_group:
                                print(f"    Warning: Episode {episode_name} already exists, overwriting...")
                                del dest_env_group[episode_name]
                            src_env_group.copy(episode_name, dest_env_group, name=episode_name)

    # Clean up the temporary folder after successful merge
    try:
        shutil.rmtree(temp_folder)
        print(f"Cleaned up temporary folder: {temp_folder}")
    except Exception as e:
        print(f"Warning: Failed to remove temporary folder {temp_folder}: {e}")


def _save_episode_metadata(
    records: List[Dict[str, Any]],
    metadata_path: Path,
    env_id: str,
) -> None:
    """Save per-episode seed/difficulty metadata to JSON."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_records = sorted(records, key=lambda rec: rec.get("episode", -1))
    metadata = {
        "env_id": env_id,
        "record_count": len(sorted_records),
        "records": sorted_records,
    }
    try:
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)
        print(f"Saved episode metadata to {metadata_path}")
    except Exception as exc:
        print(f"Warning: Failed to save episode metadata to {metadata_path}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HistoryBench dataset generator")
    parser.add_argument(
        "--env",
        "-e",
        type=str,
        nargs="+",
        default=None,
        help="Environment IDs to run. Provide one or multiple values; defaults to built-in HistoryBench environments.",
    )
    parser.add_argument(
        "--episodes",
        "-n",
        type=int,
        default=3,
        help="Number of episodes to generate per environment (default: 50)",
    )

    parser.add_argument(
        "--difficulty",
        "-d",
        type=str,
        default='211',
        help="Difficulty ratio as a 3-digit string (easy:medium:hard). Example: '211' means 2 easy, 1 medium, 1 hard per cycle.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where dataset files will be written.",
    )
    parser.add_argument(
        "--save-video",
        dest="save_video",
        action="store_true",
        default=True,
        help="Enable video recording via HistoryBenchRecordWrapper (default: enabled).",
    )
    parser.add_argument(
        "--no-save-video",
        dest="save_video",
        action="store_false",
        help="Disable video recording.",
    )
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=32,
        help="Number of parallel workers when running multiple environments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_inputs = args.env or DEFAULT_ENVS
    env_ids: List[str] = []
    for raw_env in env_inputs:
        env_ids.extend(env.strip() for env in raw_env.split(",") if env.strip())

    if not env_ids:
        env_ids = DEFAULT_ENVS.copy()

    dataset_dir: Path = args.dataset_dir
    num_workers = max(1, args.max_workers)
    episode_indices = list(range(args.episodes))

    for env_id in env_ids:
        # Create a shared temporary folder for all episodes
        temp_folder =  Path(f"/data/hongzefu/dataset_generate/temp_{env_id}_episodes")
        final_dataset_path =  Path(f"/data/hongzefu/dataset_generate/record_dataset_{env_id}.h5")

        print(f"\n{'='*80}")
        print(f"Environment: {env_id}")
        print(f"Episodes: {args.episodes}")
        print(f"Workers: {num_workers}")
        print(f"Temporary folder: {temp_folder}")
        print(f"Final dataset: {final_dataset_path}")
        print(f"{'='*80}\n")

        episode_records: List[Dict[str, Any]] = []

        if num_workers > 1:
            episode_chunks = _split_episode_indices(args.episodes, num_workers)

            if len(episode_chunks) <= 1:
                # Single chunk, run directly
                chunk = episode_chunks[0] if episode_chunks else []
                episode_records = run_env_dataset(
                    env_id,
                    chunk,
                    temp_folder,
                    args.save_video,
                    args.difficulty,
                )
            else:
                worker_count = len(episode_chunks)
                print(
                    f"Running {env_id} with {worker_count} workers across {args.episodes} episodes..."
                )

                # Each worker writes to the same temporary folder
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    future_to_chunk = {
                        executor.submit(
                            run_env_dataset,
                            env_id,
                            chunk,
                            temp_folder,  # All workers use the same temp folder
                            args.save_video,
                            args.difficulty,
                        ): chunk
                        for chunk in episode_chunks
                    }

                    for future in as_completed(future_to_chunk):
                        chunk = future_to_chunk[future]
                        chunk_label = (chunk[0], chunk[-1]) if chunk else ("?", "?")
                        try:
                            records = future.result()
                            episode_records.extend(records)
                            print(f"✓ Completed episodes {chunk_label[0]}-{chunk_label[1]} for {env_id}")
                        except Exception as exc:
                            print(
                                f"✗ Environment {env_id} failed on episodes "
                                f"{chunk_label[0]}-{chunk_label[1]} with error: {exc}"
                            )

            # Merge all episodes from the temporary folder into final dataset
            print(f"\nMerging all episodes into final dataset...")
            _merge_dataset_from_folder(
                env_id,
                temp_folder,
                final_dataset_path,
            )
        else:
            # Single worker mode
            episode_records = run_env_dataset(
                env_id,
                episode_indices,
                temp_folder,
                args.save_video,
                args.difficulty,
            )

            # Merge episodes into final dataset
            print(f"\nMerging all episodes into final dataset...")
            _merge_dataset_from_folder(
                env_id,
                temp_folder,
                final_dataset_path,
            )

        metadata_path = final_dataset_path.with_name(
            f"{final_dataset_path.stem}_metadata.json"
        )
        _save_episode_metadata(episode_records, metadata_path, env_id)

        print(f"\n✓ Finished! Final dataset saved to: {final_dataset_path}\n")

    print("✓ All requested environments processed.")


if __name__ == "__main__":
    main()
