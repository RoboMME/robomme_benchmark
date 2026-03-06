import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import sapien
from typing import Any, Dict, Iterable, List, Optional
import h5py

# Add parent directory to Python path to import robomme module
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "scripts"))
import gymnasium as gym
from gymnasium.utils.save_video import save_video

# Import Robomme related environment wrappers and exceptions
from robomme.env_record_wrapper import RobommeRecordWrapper, FailsafeTimeout
from robomme.robomme_env import *
from robomme.robomme_env.utils.SceneGenerationError import SceneGenerationError


from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)

# from util import *
import torch

# Import planners and related exceptions
from robomme.robomme_env.utils.planner_fail_safe import (
    FailAwarePandaArmMotionPlanningSolver,
    FailAwarePandaStickMotionPlanningSolver,
    ScrewPlanFailure,
)

"""

Script function: Parallel generation of Robomme environment datasets.
This script supports multi-process parallel environment simulation, generating HDF5 datasets containing RGB, depth, segmentation, etc.
Key features include:
1. Configure environment list and parameters.
2. Parallel execution of multiple episode simulations.
3. Use FailAware planner to attempt to solve tasks.
4. Record data and save as HDF5 file.
5. Merge multiple temporarily generated HDF5 files into a final dataset.
"""

# List of all supported environment module names
DEFAULT_ENVS =[
"PickXtimes",
"StopCube",
"SwingXtimes",
"BinFill",

# "VideoUnmaskSwap",
# "VideoUnmask",
# "ButtonUnmaskSwap",
# "ButtonUnmask",

# "VideoRepick",
# "VideoPlaceButton",
# "VideoPlaceOrder",
# "PickHighlight",

# "InsertPeg",
# 'MoveCube',
# "PatternLock",
# "RouteStick"
    ]

# Map environment names to unique integer codes for random seed generation
ENV_ID_TO_CODE = {name: idx + 1 for idx, name in enumerate(DEFAULT_ENVS)}
# Seed offset: Add 500,000 on top of original 10k-170k base
SEED_OFFSET = 500_000*2
# Hard-coded output root directory
DATASET_OUTPUT_ROOT = Path("/data/hongzefu/data-0306")

def _tensor_to_bool(value) -> bool:
    """
    Helper function: Convert Tensor or numpy array to Python bool.
    Used to handle success/failure flags from different sources.
    """
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)


def _split_episode_indices(num_episodes: int, max_chunks: int) -> List[List[int]]:
    """
    Helper function: Split total episodes into multiple chunks for parallel processing.
    
    Args:
        num_episodes: Total number of episodes
        max_chunks: Maximum number of chunks (usually equals number of workers)
        
    Returns:
        List of episode index lists
    """
    if num_episodes <= 0:
        return []

    chunk_count = min(max_chunks, num_episodes)
    base_size, remainder = divmod(num_episodes, chunk_count)

    chunks: List[List[int]] = []
    start = 0
    for chunk_idx in range(chunk_count):
        # If there is a remainder, the first `remainder` chunks get one extra episode
        stop = start + base_size + (1 if chunk_idx < remainder else 0)
        chunks.append(list(range(start, stop)))
        start = stop

    return chunks


def _get_difficulty_from_ratio(episode: int, difficulty_ratio: Optional[str]) -> Optional[str]:
    """
    Calculate difficulty for current episode based on episode index and ratio.
    
    Args:
        episode: Episode index
        difficulty_ratio: Difficulty ratio string, e.g., "211" means easy:medium:hard = 2:1:1
    
    Returns:
        One of "easy", "medium", "hard", or None (if no ratio specified)
    """
    if not difficulty_ratio:
        return None

    # Parse ratio string (e.g. "211" -> [2, 1, 1])
    try:
        ratios = [int(c) for c in difficulty_ratio]
        if len(ratios) != 3:
            raise ValueError("Difficulty ratio must have 3 digits")
    except (ValueError, TypeError):
        raise ValueError(f"Invalid difficulty ratio: {difficulty_ratio}. Expected 3 digits like '211'")

    # Create difficulty sequence, e.g. [easy, easy, medium, hard]
    difficulties = ["easy"] * ratios[0] + ["medium"] * ratios[1] + ["hard"] * ratios[2]
    total = sum(ratios)

    if total == 0:
        return None

    # Use modulo operation to map episode to difficulty
    return difficulties[episode % total]


def _run_episode_attempt(
    env_id: str,
    episode: int,
    seed: int,
    temp_dataset_path: Path,
    save_video: bool,
    difficulty: Optional[str],
    profile_tasks: bool = False,
) -> bool:
    """
    Run a single episode attempt and report success or failure.
    
    Main steps:
    1. Initialize environment parameters and Gym environment.
    2. Apply RobommeRecordWrapper for data recording.
    3. Select appropriate planner (PandaStick or PandaArm) based on environment type.
    4. Get task list and execute tasks one by one.
    5. Use planner to solve tasks and handle potential planning failures.
    6. Check task execution result (fail/success).
    7. Return whether episode was ultimately successful.
    """
    episode_start = time.perf_counter()
    print(f"--- Running simulation for episode:{episode}, seed:{seed}, env: {env_id} ---")
    if profile_tasks:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        print(
            f"[PROFILE] episode={episode} seed={seed} CUDA_VISIBLE_DEVICES={cuda_visible} "
            f"torch.cuda.is_available={torch.cuda.is_available()} "
            f"torch.cuda.device_count={torch.cuda.device_count()}"
        )

    env: Optional[gym.Env] = None
    try:
        setup_start = time.perf_counter()
        # 1. Environment parameter configuration
        env_kwargs = dict(
            obs_mode="rgb+depth+segmentation",  # Observation mode: RGB + depth + segmentation
            control_mode="pd_joint_pos",        # Control mode: Position control
            render_mode="rgb_array",            # Render mode
            reward_mode="dense",                # Reward mode
            seed=seed,             # Random seed
            difficulty=difficulty, # Difficulty setting
        )
        
        # Special failure recovery settings for first few episodes (for testing/demo only)
        if episode <= 5:
            env_kwargs["robomme_failure_recovery"] = True
            if episode <=2:
                env_kwargs["robomme_failure_recovery_mode"] = "z"  # z-axis recovery
            else:
                env_kwargs["robomme_failure_recovery_mode"] = "xy" # xy-axis recovery


        env = gym.make(env_id, **env_kwargs)
        
        # 2. Wrap environment to record data
        env = RobommeRecordWrapper(
            env,
            dataset=str(temp_dataset_path), # Data save path
            env_id=env_id,
            episode=episode,
            seed=seed,
            save_video=save_video,

        )

        episode_successful = False


        env.reset()
        if profile_tasks:
            print(
                f"[PROFILE] episode={episode} env_setup_and_reset_s="
                f"{time.perf_counter() - setup_start:.3f}"
            )

        # 3. Select planner
        # PatternLock and RouteStick require Stick planner, others use Arm planner
        if env_id == "PatternLock" or env_id == "RouteStick":
            planner = FailAwarePandaStickMotionPlanningSolver(
                env,
                debug=False,
                vis=False,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                joint_vel_limits=0.3,
            )
        else:
            planner = FailAwarePandaArmMotionPlanningSolver(
                env,
                debug=False,
                vis=False,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
            )
        if profile_tasks:
            print(
                f"[PROFILE] episode={episode} planner_init_s="
                f"{time.perf_counter() - setup_start:.3f}"
            )

        env.unwrapped.evaluate()
        # Get environment task list
        tasks = list(getattr(env.unwrapped, "task_list", []) or [])

        print(f"{env_id}: Task list has {len(tasks)} tasks")

        # 4. Iterate and execute all subtasks
        for idx, task_entry in enumerate(tasks):
            task_name = task_entry.get("name", f"Task {idx}")
            print(f"Executing task {idx + 1}/{len(tasks)}: {task_name}")

            solve_callable = task_entry.get("solve")
            if not callable(solve_callable):
                raise ValueError(
                    f"Task '{task_name}' must supply a callable 'solve'."
                )

            # Evaluate before executing solve
            env.unwrapped.evaluate(solve_complete_eval=True)
            screw_failed = False
            task_start = time.perf_counter()
            try:
                # 5. Call planner to solve current task
                solve_callable(env, planner)
            except ScrewPlanFailure as exc:
                # Plan failure handling
                screw_failed = True
                print(f"Screw plan failure during '{task_name}': {exc}")
                env.unwrapped.failureflag = torch.tensor([True])
                env.unwrapped.successflag = torch.tensor([False])
                env.unwrapped.current_task_failure = True
            except FailsafeTimeout as exc:
                # Timeout handling
                print(f"Failsafe: {exc}")
                break

            # Evaluation after task execution
            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
            if profile_tasks:
                print(
                    f"[PROFILE] episode={episode} task_index={idx} "
                    f"task_name={task_name!r} elapsed_s={time.perf_counter() - task_start:.3f}"
                )

            fail_flag = evaluation.get("fail", False)
            success_flag = evaluation.get("success", False)

            # 6. Check success/failure conditions
            if _tensor_to_bool(success_flag):
                print("All tasks completed successfully.")
                episode_successful = True
                break

            if screw_failed or _tensor_to_bool(fail_flag):
                print("Encountered failure condition; stopping task sequence.")
                break

        else:
            # If loop ends normally (no break), check success again
            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
            episode_successful = _tensor_to_bool(evaluation.get("success", False))

        # 7. Prioritize wrapper success signal (double check)
        episode_successful = episode_successful or _tensor_to_bool(
            getattr(env, "episode_success", False)
        )

    except SceneGenerationError as exc:# Scene generation failure may occur in environments like swingxtimes
        print(
            f"Scene generation failed for env {env_id}, episode {episode}, seed {seed}: {exc}"
        )
        episode_successful = False
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as close_exc:
                # Even if close() fails, if episode was successful, return success
                # because HDF5 data was written before close() (in write() method)
                print(f"Warning: Exception during env.close() for episode {episode}, seed {seed}: {close_exc}")
                # If episode was successful, exception in close() should not affect return value
                # episode_successful was determined before close()

    status_text = "SUCCESS" if episode_successful else "FAILED"
    if profile_tasks:
        print(
            f"[PROFILE] episode={episode} total_elapsed_s="
            f"{time.perf_counter() - episode_start:.3f}"
        )
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
    profile_tasks: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run dataset generation for a batch of episodes and save data to temporary folder.
    
    Args:
        env_id: Environment ID
        episode_indices: List of episode indices to run
        temp_folder: Temporary folder to save data
        save_video: Whether to save video
        difficulty_ratio: Difficulty ratio string
    
    Returns:
        List of generated episode metadata records
    """
    temp_folder.mkdir(parents=True, exist_ok=True)
    episode_indices = list(episode_indices)
    if not episode_indices:
        return []

    if env_id not in ENV_ID_TO_CODE:
        raise ValueError(f"Environment {env_id} missing from ENV_ID_TO_CODE mapping")
    env_code = ENV_ID_TO_CODE[env_id]

    # Use a temporary h5 file path to pass to wrapper
    # Note: wrapper will actually create separate episode files in subfolders under this path's directory
    temp_dataset_path = temp_folder / f"temp_chunk.h5"
    episode_records: List[Dict[str, Any]] = []

    for episode in episode_indices:
        # Calculate difficulty for current episode
        difficulty = _get_difficulty_from_ratio(episode, difficulty_ratio)
        if difficulty:
            print(f"Episode {episode} assigned difficulty: {difficulty}")

        # Generate base seed: Add offset to original formula, ensure unique within batch for env/episode
        base_seed = SEED_OFFSET + env_code * 10000 + episode * 100
        attempt = 0

        # Retry loop: If current seed fails, try next seed until success
        max_attempts = 100  # Add max retry limit to avoid infinite loop
        episode_success = False
        while attempt < max_attempts:
            seed = base_seed + attempt  # Use unique seed for each attempt

            try:
                success = _run_episode_attempt(
                    env_id=env_id,
                    episode=episode,
                    seed=seed,
                    temp_dataset_path=temp_dataset_path,
                    save_video=save_video,
                    difficulty=difficulty,
                    profile_tasks=profile_tasks,
                )

                if success:
                    # Record successful episode information
                    episode_records.append(
                        {
                            "task": env_id,
                            "episode": episode,
                            "seed": seed,
                            "difficulty": difficulty,
                        }
                    )
                    episode_success = True
                    break  # Break retry loop on success, proceed to next episode

                attempt += 1
                next_seed = base_seed + attempt
                print(
                    f"Episode {episode} failed; retrying with new seed {next_seed} (attempt {attempt + 1})"
                )

            except Exception as e:
                attempt += 1
                if attempt >= max_attempts:
                    break

    return episode_records


def _merge_dataset_from_folder(
    env_id: str,
    temp_folder: Path,
    final_dataset_path: Path,
) -> None:
    """
    Merge all episode files from temporary folder into final dataset.
    
    Args:
        env_id: Environment ID
        temp_folder: Temporary folder containing episode files
        final_dataset_path: Final output HDF5 file path
    """
    if not temp_folder.exists() or not temp_folder.is_dir():
        print(f"Warning: Temporary folder {temp_folder} does not exist")
        return

    final_dataset_path.parent.mkdir(parents=True, exist_ok=True)

    # Find subfolders created by RobommeRecordWrapper
    # It usually creates directories ending with "_hdf5_files"
    hdf5_folders = list(temp_folder.glob("*_hdf5_files"))

    if not hdf5_folders:
        print(f"Warning: No HDF5 folders found in {temp_folder}")
        return

    print(f"Merging episodes from {temp_folder} into {final_dataset_path}")

    # Open final HDF5 file in append mode
    with h5py.File(final_dataset_path, "a") as final_file:
        for hdf5_folder in sorted(hdf5_folders):
            # Get all h5 files in folder
            h5_files = sorted(hdf5_folder.glob("*.h5"))

            if not h5_files:
                print(f"Warning: No h5 files found in {hdf5_folder}")
                continue

            print(f"Found {len(h5_files)} episode files in {hdf5_folder.name}")

            # Merge each episode file
            for h5_file in h5_files:
                print(f"  - Merging {h5_file.name}")

                try:
                    with h5py.File(h5_file, "r") as episode_file:
                        file_keys = list(episode_file.keys())
                        if len(file_keys) == 0:
                            print(f"    Warning: {h5_file.name} is empty, skipping...")
                            continue
                        
                        for env_group_name, src_env_group in episode_file.items():
                            episode_keys = list(src_env_group.keys()) if isinstance(src_env_group, h5py.Group) else []
                            if len(episode_keys) == 0:
                                print(f"    Warning: {env_group_name} in {h5_file.name} has no episodes, skipping...")
                                continue
                            
                            # If environment group (e.g. 'PickXtimes') does not exist, copy directly
                            if env_group_name not in final_file:
                                final_file.copy(src_env_group, env_group_name)
                                continue

                            dest_env_group = final_file[env_group_name]
                            if not isinstance(dest_env_group, h5py.Group):
                                print(f"    Warning: {env_group_name} is not a group, skipping...")
                                continue

                            # If environment group exists, copy episodes one by one
                            for episode_name in src_env_group.keys():
                                if episode_name in dest_env_group:
                                    print(f"    Warning: Episode {episode_name} already exists, overwriting...")
                                    del dest_env_group[episode_name]
                                src_env_group.copy(episode_name, dest_env_group, name=episode_name)
                except Exception as e:
                    print(f"    Error merging {h5_file.name}: {e}")
                    continue

    # Clean up temporary folder after successful merge
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
    """Save seed/difficulty metadata for each episode to JSON file."""
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
    parser = argparse.ArgumentParser(description="Robomme dataset generator")
    parser.add_argument(
        "--env",
        "-e",
        type=str,
        nargs="+",
        default=None,
        help="Environment IDs to run. Provide one or more values; defaults to all built-in Robomme environments.",
    )
    parser.add_argument(
        "--episodes",
        "-n",
        type=int,
        default=5,
        help="Number of episodes to generate per environment (default: 100)",
    )

    parser.add_argument(
        "--difficulty",
        "-d",
        type=str,
        default='211',
        help="Difficulty ratio, 3-digit string (easy:medium:hard). E.g.: '211' means 2 easy, 1 medium, 1 hard per cycle.",
    )
    parser.add_argument(
        "--save-video",
        dest="save_video",
        action="store_true",
        default=True,
        help="Enable video recording via RobommeRecordWrapper (default: enabled).",
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
        default=20,
        help="Number of parallel workers when running multiple environments.",
    )
    parser.add_argument(
        "--profile-tasks",
        action="store_true",
        help="Print per-episode and per-task timing plus CUDA visibility diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_inputs = args.env or DEFAULT_ENVS
    env_ids: List[str] = []
    # Parse environment list argument, support comma separation
    for raw_env in env_inputs:
        env_ids.extend(env.strip() for env in raw_env.split(",") if env.strip())

    if not env_ids:
        env_ids = DEFAULT_ENVS.copy()

    num_workers = max(1, args.max_workers)
    episode_indices = list(range(args.episodes))

    for env_id in env_ids:
        # Create shared temporary folder for all episodes
        temp_folder = DATASET_OUTPUT_ROOT / f"temp_{env_id}_episodes"
        final_dataset_path = DATASET_OUTPUT_ROOT / f"record_dataset_{env_id}.h5"

        print(f"\n{'='*80}")
        print(f"Environment: {env_id}")
        print(f"Episodes: {args.episodes}")
        print(f"Workers: {num_workers}")
        print(f"Temporary folder: {temp_folder}")
        print(f"Final dataset: {final_dataset_path}")
        print(f"{'='*80}\n")

        episode_records: List[Dict[str, Any]] = []

        if num_workers > 1:
            # 1. Split task chunks
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
                    args.profile_tasks,
                )
            else:
                worker_count = len(episode_chunks)
                print(
                    f"Running {env_id} with {worker_count} workers across {args.episodes} episodes..."
                )

                # 2. Parallel execution
                # Each worker writes to same temp folder (but uses different file/dir names)
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    future_to_chunk = {
                        executor.submit(
                            run_env_dataset,
                            env_id,
                            chunk,
                            temp_folder,  # All workers use the same temporary folder path
                            args.save_video,
                            args.difficulty,
                            args.profile_tasks,
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

            # 3. Merge all episode files into final dataset
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
                args.profile_tasks,
            )

            # Merge episodes into final dataset
            print(f"\nMerging all episodes into final dataset...")
            _merge_dataset_from_folder(
                env_id,
                temp_folder,
                final_dataset_path,
            )

        # 4. Save metadata
        metadata_path = final_dataset_path.with_name(
            f"{final_dataset_path.stem}_metadata.json"
        )
        _save_episode_metadata(episode_records, metadata_path, env_id)

        print(f"\n✓ Finished! Final dataset saved to: {final_dataset_path}\n")

    print("✓ All requested environments processed.")


if __name__ == "__main__":
    main()
