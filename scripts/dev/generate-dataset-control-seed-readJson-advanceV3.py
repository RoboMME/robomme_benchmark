import os

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Set
import h5py

import gymnasium as gym

# Import Robomme related environment wrappers and exception classes
from robomme.env_record_wrapper import RobommeRecordWrapper, FailsafeTimeout
from robomme.robomme_env import *
from robomme.robomme_env.utils.SceneGenerationError import SceneGenerationError

# from util import *
import torch

# Import planner and related exception classes
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

"VideoUnmaskSwap",
"VideoUnmask",
"ButtonUnmaskSwap",
"ButtonUnmask",

"VideoRepick",
"VideoPlaceButton",
"VideoPlaceOrder",
"PickHighlight",

"InsertPeg",
'MoveCube',
"PatternLock",
"RouteStick"
    ]

# Reference dataset metadata root directory: used to read difficulty and seed
SOURCE_METADATA_ROOT = Path("/data/hongzefu/robomme_benchmark/src/robomme/env_metadata/1206")
VALID_DIFFICULTIES: Set[str] = {"easy", "medium", "hard"}
DATASET_SCREW_MAX_ATTEMPTS = 3
DATASET_RRT_MAX_ATTEMPTS = 3


def _load_env_metadata_records(
    env_id: str,
    metadata_root: Path,
) -> List[Dict[str, Any]]:
    """
    Read metadata records for an environment from the reference directory to control difficulty and seed.
    """
    metadata_path = metadata_root / f"record_dataset_{env_id}_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found for env '{env_id}': {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        payload = json.load(metadata_file)

    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(
            f"Metadata file has no valid 'records' list: {metadata_path}"
        )

    normalized_records: List[Dict[str, Any]] = []
    for idx, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise ValueError(
                f"Invalid metadata record at index {idx} in {metadata_path}"
            )
        if "episode" not in raw_record or "seed" not in raw_record or "difficulty" not in raw_record:
            raise ValueError(
                f"Metadata record missing episode/seed/difficulty at index {idx} in {metadata_path}"
            )

        try:
            episode = int(raw_record["episode"])
            seed = int(raw_record["seed"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Metadata record has non-integer episode/seed at index {idx} in {metadata_path}"
            ) from exc

        difficulty_raw = str(raw_record["difficulty"]).strip().lower()
        if difficulty_raw not in VALID_DIFFICULTIES:
            raise ValueError(
                f"Metadata record has invalid difficulty '{raw_record['difficulty']}' "
                f"at index {idx} in {metadata_path}. Expected one of {sorted(VALID_DIFFICULTIES)}."
            )

        normalized_records.append(
            {
                "episode": episode,
                "seed": seed,
                "difficulty": difficulty_raw,
            }
        )

    normalized_records.sort(key=lambda rec: rec["episode"])
    print(
        f"Loaded {len(normalized_records)} metadata records for {env_id} from {metadata_path}"
    )
    return normalized_records


def _build_seed_candidates_from_metadata(
    episode: int,
    metadata_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Construct candidate (seed, difficulty) list for current episode.
    Strictly use only the seed from metadata for the same episode, no cross-episode fallback.
    """
    if not metadata_records:
        return []

    same_episode_records = [rec for rec in metadata_records if rec["episode"] == episode]
    if not same_episode_records:
        return []
    if len(same_episode_records) > 1:
        raise ValueError(
            f"Found duplicated metadata records for episode {episode}. "
            "Strict mode requires exactly one source record per episode."
        )

    rec = same_episode_records[0]
    return [{"seed": int(rec["seed"]), "difficulty": rec["difficulty"]}]

def _tensor_to_bool(value) -> bool:
    """
    Helper function: Convert Tensor or numpy array to Python bool type.
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
    Helper function: Split total episodes into multiple chunks for parallel processing by different processes.
    
    Args:
        num_episodes: Total number of episodes
        max_chunks: Max number of chunks (usually equals number of workers)
        
    Returns:
        List containing lists of episode indices
    """
    if num_episodes <= 0:
        return []

    chunk_count = min(max_chunks, num_episodes)
    base_size, remainder = divmod(num_episodes, chunk_count)

    chunks: List[List[int]] = []
    start = 0
    for chunk_idx in range(chunk_count):
        # If there is a remainder, allocate one extra episode to the first 'remainder' chunks
        stop = start + base_size + (1 if chunk_idx < remainder else 0)
        chunks.append(list(range(start, stop)))
        start = stop

    return chunks


def _run_episode_attempt(
    env_id: str,
    episode: int,
    seed: int,
    temp_dataset_path: Path,
    save_video: bool,
    difficulty: Optional[str],
) -> bool:
    """
    Run a single episode attempt and report success or failure.
    
    Main steps:
    1. Initialize environment parameters and Gym environment.
    2. Apply RobommeRecordWrapper for data recording.
    3. Select appropriate planner based on environment type (PandaStick or PandaArm).
    4. Get task list and execute tasks one by one.
    5. Use planner to solve task and handle possible planning failures.
    6. Check task execution result (fail/success).
    7. Return whether episode is finally successful.
    """
    print(f"--- Running simulation for episode:{episode}, seed:{seed}, env: {env_id} ---")

    env: Optional[gym.Env] = None
    try:
        # 1. Environment parameter configuration
        env_kwargs = dict(
            obs_mode="rgb+depth+segmentation",  # Observation mode: RGB + Depth + Segmentation
            control_mode="pd_joint_pos",        # Control mode: Position control
            render_mode="rgb_array",            # Render mode
            reward_mode="dense",                # Reward mode
            seed=seed,             # Random seed
            difficulty=difficulty, # Difficulty setting
        )
        
        # Special failure recovery settings for the first few episodes (for testing or demonstration purposes only)
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

        original_move_to_pose_with_screw = planner.move_to_pose_with_screw
        original_move_to_pose_with_rrt = planner.move_to_pose_with_RRTStar

        def _move_to_pose_with_screw_then_rrt_retry(*args, **kwargs):
            for attempt in range(1, DATASET_SCREW_MAX_ATTEMPTS + 1):
                try:
                    result = original_move_to_pose_with_screw(*args, **kwargs)
                except ScrewPlanFailure as exc:
                    print(
                        f"[DatasetGen] screw planning failed "
                        f"(attempt {attempt}/{DATASET_SCREW_MAX_ATTEMPTS}): {exc}"
                    )
                    continue

                if isinstance(result, int) and result == -1:
                    print(
                        f"[DatasetGen] screw planning returned -1 "
                        f"(attempt {attempt}/{DATASET_SCREW_MAX_ATTEMPTS})"
                    )
                    continue

                return result

            print(
                "[DatasetGen] screw planning exhausted; "
                f"fallback to RRT* (max {DATASET_RRT_MAX_ATTEMPTS} attempts)"
            )

            for attempt in range(1, DATASET_RRT_MAX_ATTEMPTS + 1):
                try:
                    result = original_move_to_pose_with_rrt(*args, **kwargs)
                except Exception as exc:
                    print(
                        f"[DatasetGen] RRT* planning failed "
                        f"(attempt {attempt}/{DATASET_RRT_MAX_ATTEMPTS}): {exc}"
                    )
                    continue

                if isinstance(result, int) and result == -1:
                    print(
                        f"[DatasetGen] RRT* planning returned -1 "
                        f"(attempt {attempt}/{DATASET_RRT_MAX_ATTEMPTS})"
                    )
                    continue

                return result

            print("[DatasetGen] screw->RRT* planning exhausted; return -1")
            return -1

        planner.move_to_pose_with_screw = _move_to_pose_with_screw_then_rrt_retry

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

            # Evaluate once before executing solve
            env.unwrapped.evaluate(solve_complete_eval=True)
            screw_failed = False
            try:
                # 5. Call planner to solve current task
                solve_result = solve_callable(env, planner)
                if isinstance(solve_result, int) and solve_result == -1:
                    screw_failed = True
                    print(f"Screw->RRT* planning exhausted during '{task_name}'")
                    env.unwrapped.failureflag = torch.tensor([True])
                    env.unwrapped.successflag = torch.tensor([False])
                    env.unwrapped.current_task_failure = True
            except ScrewPlanFailure as exc:
                # Planning failure handling
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

        # 7. Prioritize wrapper's success signal (double check)
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
                # Even if close() fails, return success if episode was successful
                # Because HDF5 data was written before close() (in write() method)
                print(f"Warning: Exception during env.close() for episode {episode}, seed {seed}: {close_exc}")
                # If episode was successful, close() exception should not affect return value
                # episode_successful was determined before close()

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
    metadata_records: List[Dict[str, Any]],
    gpu_id: int,
) -> List[Dict[str, Any]]:
    """
    Run dataset generation for a batch of episodes and save data to temporary folder.
    
    Args:
        env_id: Environment ID
        episode_indices: List of episode indices to run
        temp_folder: Temporary folder to save data
        save_video: Whether to save video
        metadata_records: Records from reference dataset metadata
        gpu_id: GPU ID to use
    
    Returns:
        Generated episode metadata record list
    """
    # Set GPU used by current process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    temp_folder.mkdir(parents=True, exist_ok=True)
    episode_indices = list(episode_indices)
    if not episode_indices:
        return []

    if env_id not in DEFAULT_ENVS:
        raise ValueError(f"Unsupported environment: {env_id}")

    # Pass a temporary h5 file path to wrapper
    # Note: wrapper will actually create separate episode files in a subfolder of that path's directory
    temp_dataset_path = temp_folder / f"temp_chunk.h5"
    episode_records: List[Dict[str, Any]] = []

    for episode in episode_indices:
        candidate_pairs = _build_seed_candidates_from_metadata(episode, metadata_records)
        if not candidate_pairs:
            print(f"Episode {episode}: no metadata candidate seeds found, skipping.")
            continue

        episode_success = False
        MAX_RETRY_ATTEMPTS = 20

        for attempt_idx, candidate in enumerate(candidate_pairs, start=1):
            base_seed = int(candidate["seed"])
            difficulty = str(candidate["difficulty"])
            
            current_seed = base_seed
            for retry_count in range(MAX_RETRY_ATTEMPTS):
                if retry_count > 0:
                    current_seed += 1

                print(
                    f"Episode {episode} attempt {retry_count + 1}/{MAX_RETRY_ATTEMPTS} "
                    f"with seed={current_seed} (base={base_seed}, diff={difficulty})"
                )

                try:
                    success = _run_episode_attempt(
                        env_id=env_id,
                        episode=episode,
                        seed=current_seed,
                        temp_dataset_path=temp_dataset_path,
                        save_video=save_video,
                        difficulty=difficulty,
                    )

                    if success:
                        # Record successful episode information
                        episode_records.append(
                            {
                                "task": env_id,
                                "episode": episode,
                                "seed": current_seed,
                                "difficulty": difficulty,
                            }
                        )
                        episode_success = True
                        break  # Break retry loop (seed increment loop)
                    
                    print(
                        f"Episode {episode} failed with seed {current_seed}; retrying with seed+1..."
                    )
                except Exception as exc:
                    print(
                        f"Episode {episode} exception with seed {current_seed}: {exc}; retrying with seed+1..."
                    )
            
            if episode_success:
                break # Break candidate loop

        if not episode_success:
            print(
                f"Episode {episode} failed with strict source metadata seed; "
                "metadata will not be recorded for this episode."
            )

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

    # Open final HDF5 file for append mode writing
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

    # Keep videos: wrapper writes videos to 'videos' under temp dir, move to final dir before cleanup
    temp_videos_dir = temp_folder / "videos"
    final_videos_dir = final_dataset_path.parent / "videos"
    if temp_videos_dir.exists() and temp_videos_dir.is_dir():
        final_videos_dir.mkdir(parents=True, exist_ok=True)
        moved_count = 0
        for video_path in sorted(temp_videos_dir.glob("*.mp4")):
            target_path = final_videos_dir / video_path.name
            if target_path.exists():
                stem = target_path.stem
                suffix = target_path.suffix
                index = 1
                while True:
                    candidate = final_videos_dir / f"{stem}_dup{index}{suffix}"
                    if not candidate.exists():
                        target_path = candidate
                        break
                    index += 1
            try:
                shutil.move(str(video_path), str(target_path))
                moved_count += 1
            except Exception as exc:
                print(f"Warning: Failed to move video {video_path.name}: {exc}")
        if moved_count > 0:
            print(f"Moved {moved_count} videos to {final_videos_dir}")

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
    parser = argparse.ArgumentParser(description="Robomme Dataset Generator")
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
        default=100,
        help="Number of episodes generated per environment (Default: 100)",
    )
    parser.add_argument(
        "--save-video",
        dest="save_video",
        action="store_true",
        default=True,
        help="Enable video recording via RobommeRecordWrapper (Default: Enabled).",
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
        "--gpus",
        type=str,
        default="1",
        help="GPU selection. Supported values: '0', '1', '0,1' (or '1,0'). Default: '0'.",
    )
    return parser.parse_args()


def _parse_gpu_ids(gpu_spec: str) -> List[int]:
    """Parse user GPU spec string to a deduplicated GPU id list."""
    valid_gpu_ids = {0, 1}
    raw_tokens = [token.strip() for token in gpu_spec.split(",") if token.strip()]
    if not raw_tokens:
        raise ValueError("GPU spec is empty. Use one of: 0, 1, 0,1")

    gpu_ids: List[int] = []
    for token in raw_tokens:
        try:
            gpu_id = int(token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid GPU id '{token}'. Supported values are 0 and 1."
            ) from exc

        if gpu_id not in valid_gpu_ids:
            raise ValueError(
                f"Unsupported GPU id '{gpu_id}'. Supported values are 0 and 1."
            )
        if gpu_id not in gpu_ids:
            gpu_ids.append(gpu_id)

    if not gpu_ids:
        raise ValueError("No valid GPU id provided. Use one of: 0, 1, 0,1")
    return gpu_ids


def main() -> None:
    args = parse_args()
    env_inputs = args.env or DEFAULT_ENVS
    env_ids: List[str] = []
    # Parse environment list arguments, support comma separation
    for raw_env in env_inputs:
        env_ids.extend(env.strip() for env in raw_env.split(",") if env.strip())

    if not env_ids:
        env_ids = DEFAULT_ENVS.copy()

    num_workers = max(1, args.max_workers)
    gpu_spec = args.gpus
    gpu_ids = _parse_gpu_ids(gpu_spec)
    episode_indices = list(range(args.episodes))

    for env_id in env_ids:
        source_metadata_records = _load_env_metadata_records(
            env_id=env_id,
            metadata_root=SOURCE_METADATA_ROOT,
        )

        # Create shared temporary folder for all episodes
        temp_folder =  Path(f"/data/hongzefu/data_0226/temp_{env_id}_episodes")
        final_dataset_path =  Path(f"/data/hongzefu/data_0226/record_dataset_{env_id}.h5")
        #final_dataset_path =  Path(f"/data/hongzefu/dataset_generate/record_dataset_{env_id}.h5")

        print(f"\n{'='*80}")
        print(f"Environment: {env_id}")
        print(f"Episodes: {args.episodes}")
        print(f"Workers: {num_workers}")
        if len(gpu_ids) == 1:
            print(f"GPU mode: Single GPU ({gpu_ids[0]})")
        else:
            print(f"GPU mode: Multi GPU ({','.join(str(gpu) for gpu in gpu_ids)})")
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
                    source_metadata_records,
                    gpu_ids[0],
                )
            else:
                worker_count = len(episode_chunks)
                print(
                    f"Running {env_id} with {worker_count} workers across {args.episodes} episodes..."
                )
                
                future_to_chunk = {}
                futures = []
                if len(gpu_ids) == 1:
                    print(
                        f"Assigning all {len(episode_chunks)} chunks to GPU {gpu_ids[0]} ({num_workers} workers)"
                    )
                else:
                    print(
                        f"Assigning {len(episode_chunks)} chunks across GPUs {','.join(str(gpu) for gpu in gpu_ids)}"
                    )

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    for chunk_idx, chunk in enumerate(episode_chunks):
                        assigned_gpu = gpu_ids[chunk_idx % len(gpu_ids)]
                        f = executor.submit(
                            run_env_dataset,
                            env_id,
                            chunk,
                            temp_folder,
                            args.save_video,
                            source_metadata_records,
                            assigned_gpu,
                        )
                        future_to_chunk[f] = (chunk, assigned_gpu)
                        futures.append(f)

                    for future in as_completed(futures):
                        chunk, assigned_gpu = future_to_chunk[future]
                        chunk_label = (chunk[0], chunk[-1]) if chunk else ("?", "?")
                        try:
                            records = future.result()
                            episode_records.extend(records)
                            print(
                                f"✓ Completed episodes {chunk_label[0]}-{chunk_label[1]} for {env_id} on GPU {assigned_gpu}"
                            )
                        except Exception as exc:
                            print(
                                f"✗ Environment {env_id} failed on episodes "
                                f"{chunk_label[0]}-{chunk_label[1]} (GPU {assigned_gpu}) with error: {exc}"
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
                source_metadata_records,
                gpu_ids[0], # gpu_id
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
