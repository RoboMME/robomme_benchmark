# -*- coding: utf-8 -*-
# Script function: Unified dataset replay entry point, supports four action_spaces: joint_angle / ee_pose / keypoint / oracle_planner.
# Consistent with subgoal_evaluate_func.py's main loop and debug fields; the difference is that actions come from EpisodeDatasetResolver.
# [New] Support parallel multi-process replay and alternate task assignment between two GPUs.

import os
import sys
import argparse
import concurrent.futures
import multiprocessing as mp
from typing import Any, Optional

import numpy as np
import torch

from robomme.robomme_env import *
from robomme.robomme_env.utils import *
from robomme.env_record_wrapper import (
    BenchmarkEnvBuilder,
    EpisodeDatasetResolver,
)
from robomme.robomme_env.utils.save_reset_video import save_robomme_video

# Enable only one ACTION_SPACE; other options are kept in comments for manual switching
#ACTION_SPACE = "joint_angle"
#ACTION_SPACE = "ee_pose"

ACTION_SPACE = "keypoint"
#ACTION_SPACE = "oracle_planner"

GUI_RENDER = False

DATASET_ROOT = "/data/hongzefu/data_0220"
OVERRIDE_METADATA_PATH = "/data/hongzefu/data_0220"

# ######## Video saving variables (output directory) start ########
# Video output directory: Independently hardcoded, not aligned with h5 path or env_id
OUT_VIDEO_DIR = "/data/hongzefu/dataset_replay-0220"
# ######## Video saving variables (output directory) end ########
MAX_STEPS = 1000

DEFAULT_ENV_IDS = [
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
    "MoveCube",
     "PatternLock",
     "RouteStick",
 ]

def _parse_oracle_command(choice_action: Optional[Any]) -> Optional[dict[str, Any]]:
    if not isinstance(choice_action, dict):
        return None
    label = choice_action.get("label")
    if not isinstance(label, str) or not label:
        return None
    return choice_action


def _align_blue_box_mask(mask_like: Any, target_len: int) -> list[bool]:
    n = max(0, int(target_len))
    if n == 0:
        return []
    if not isinstance(mask_like, (list, tuple)):
        return [False] * n
    mask = [bool(x) for x in mask_like[:n]]
    if len(mask) < n:
        mask.extend([False] * (n - len(mask)))
    return mask


def init_worker(gpu_id: int):
    """
    Worker process initialization function, sets CUDA_VISIBLE_DEVICES.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # print(f"[Worker] Initialized on GPU {gpu_id} (PID: {os.getpid()})")

def evaluate_episode(
    env_id: str,
    episode: int,
    dataset_root: str,
    override_metadata_path: str,
    action_space: str,
    out_video_dir: str,
    gui_render: bool
) -> str:
    """
    Evaluation logic for a single Episode.
    """
    # Reconstruct Envs and Resolver (avoid passing complex objects across processes)
    env_builder = BenchmarkEnvBuilder(
        env_id=env_id,
        dataset="train",
        action_space=action_space,
        gui_render=gui_render,
        override_metadata_path=override_metadata_path,
    )

    env = None
    dataset_resolver = None
    
    try:
        env = env_builder.make_env_for_episode(episode, max_steps=MAX_STEPS)
        dataset_resolver = EpisodeDatasetResolver(
            env_id=env_id,
            episode=episode,
            dataset_directory=dataset_root,
        )

        # obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.reset()
        obs_batch, info_batch = env.reset()

        # Maintain debug variable semantics from subgoal_evaluate_func.py
        # Note: These local variables in multi-processing can be simplified if printing is not needed, but unpacking logic is retained for consistency.
        maniskill_obs = obs_batch["maniskill_obs"]
        front_camera = obs_batch["front_rgb_list"]
        wrist_camera = obs_batch["wrist_rgb_list"]
        # Other variables unpacking skipped unless used downstream

        task_goal_list = info_batch["task_goal"]
        # task_goal = task_goal_list[0] if task_goal_list else None
        
        info = {k: v[-1] if isinstance(v, list) and v else v for k, v in info_batch.items()}
        # terminated = bool(terminated_batch[-1].item())
        # truncated = bool(truncated_batch[-1].item())

        # ######## Video saving variable preparation (reset phase) start ########
        reset_base_frames = [torch.as_tensor(f).detach().cpu().numpy().copy() for f in front_camera]
        reset_wrist_frames = [torch.as_tensor(f).detach().cpu().numpy().copy() for f in wrist_camera]
        _subgoal = info_batch.get("grounded_subgoal_online", "")
        reset_subgoal_grounded = _subgoal if isinstance(_subgoal, list) else [_subgoal] * len(reset_base_frames)
        # ######## Video saving variable preparation (reset phase) end ########

        # ######## Video saving variable initialization start ########
        step = 0
        episode_success = False
        rollout_base_frames: list[np.ndarray] = []
        rollout_wrist_frames: list[np.ndarray] = []
        rollout_subgoal_grounded: list[Any] = []
        rollout_oracle_fallback_blue_box_mask: list[bool] = []
        # ######## Video saving variable initialization end ########

        while True:
            replay_key = action_space
            action = dataset_resolver.get_step(replay_key, step)
            if action_space == "oracle_planner":
                action = _parse_oracle_command(action)
            if action is None:
                break

            obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.step(action)

            # Maintain debug variable semantics from subgoal_evaluate_func.py
            front_camera = obs_batch["front_rgb_list"]
            wrist_camera = obs_batch["wrist_rgb_list"]

            subgoal_grounded = info_batch["grounded_subgoal_online"]

            # ######## Video saving variable preparation (replay phase) start ########
            rollout_base_frames.extend(torch.as_tensor(f).detach().cpu().numpy().copy() for f in front_camera)
            rollout_wrist_frames.extend(torch.as_tensor(f).detach().cpu().numpy().copy() for f in wrist_camera)
            if isinstance(subgoal_grounded, list):
                rollout_subgoal_grounded.extend(subgoal_grounded)
            else:
                rollout_subgoal_grounded.extend([subgoal_grounded] * len(front_camera))
            if action_space == "oracle_planner":
                raw_mask = info_batch.get("oracle_random_fallback_blue_box_mask", [])
            else:
                raw_mask = []
            rollout_oracle_fallback_blue_box_mask.extend(
                _align_blue_box_mask(raw_mask, target_len=len(front_camera))
            )
            # ######## Video saving variable preparation (replay phase) end ########

            info = {k: v[-1] if isinstance(v, list) and v else v for k, v in info_batch.items()}
            terminated = bool(terminated_batch.item())
            truncated = bool(truncated_batch.item())

            step += 1
            if gui_render:
                env.render()
            
            if truncated:
                # print(f"[{env_id}] episode {episode} step limit exceeded, step {step}.")
                break
            if terminated:
                succ = info.get("success")
                if succ == torch.tensor([True]) or (
                    isinstance(succ, torch.Tensor) and succ.item()
                ):
                    # print(f"[{env_id}] episode {episode} success.")
                    episode_success = True
                elif info.get("fail", False):
                    # print(f"[{env_id}] episode {episode} failed.")
                    pass
                break

        # ######## Video saving section start ########
        save_robomme_video(
            reset_base_frames=reset_base_frames,
            reset_wrist_frames=reset_wrist_frames,
            rollout_base_frames=rollout_base_frames,
            rollout_wrist_frames=rollout_wrist_frames,
            reset_subgoal_grounded=reset_subgoal_grounded,
            rollout_subgoal_grounded=rollout_subgoal_grounded,
            out_video_dir=out_video_dir,
            action_space=action_space,
            env_id=env_id,
            episode=episode,
            episode_success=episode_success,
            rollout_blue_box_mask=(
                rollout_oracle_fallback_blue_box_mask
                if action_space == "oracle_planner"
                else None
            ),
        )
        # ######## Video saving section end ########

        status = "Success" if episode_success else "Ended"
        if not episode_success and info.get("fail", False):
            status = "Failed"
        return f"[{env_id}] episode {episode} {status} (step {step})"

    except (FileNotFoundError, KeyError) as exc:
        return f"[{env_id}] episode {episode} data missing, skip. {exc}"
    except Exception as exc:
        # import traceback
        # traceback.print_exc()
        return f"[{env_id}] episode {episode} replay exception, skip. {exc}"
    finally:
        if dataset_resolver is not None:
            dataset_resolver.close()
        if env is not None:
            env.close()

def _parse_gpus(s: str) -> list[int]:
    """Parse --gpus: '0' -> [0], '1' -> [1], '0,1' -> [0, 1]."""
    allowed = {"0", "1", "0,1", "1,0"}
    v = s.strip()
    if v not in allowed:
        raise argparse.ArgumentTypeError(
            f"--gpus must be one of: 0, 1, 0,1 (got {s!r})"
        )
    if "," in v:
        return [int(x) for x in v.split(",")]
    return [int(v)]

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay dataset for one env_id in parallel.")
    parser.add_argument(
        "--envid",
        required=False,
        type=str,
        default=None,
        help="Single environment id to replay.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Total max workers (split across GPUs when using 2 GPUs).",
    )
    parser.add_argument(
        "--gpus",
        type=_parse_gpus,
        default=[1],
        help="GPUs to use: '0' (GPU 0 only), '1' (GPU 1 only), '0,1' (both). Default: 0.",
    )
    return parser.parse_args()

def process_env_id(env_id: str, max_workers_total: int, gpu_ids: list[int]):
    # Simple calculation of episode count (do not instantiate env_builder to avoid overhead, or lightweight instantiation)
    # To get episode_count, we need to instantiate env_builder once
    # But we only need the metadata parsing part
    temp_builder = BenchmarkEnvBuilder(
        env_id=env_id,
        dataset="train",
        action_space=ACTION_SPACE,
        gui_render=False, # Just to read metadata
        override_metadata_path=OVERRIDE_METADATA_PATH,
    )
    episode_count = temp_builder.get_episode_num()
    print(f"[{env_id}] Total episodes found in metadata: {episode_count}")
    print(f"Parallel execution with max_workers={max_workers_total} on GPU(s) {gpu_ids}")
    
    if episode_count == 0:
        print("No episodes to replay.")
        return

    n_gpus = len(gpu_ids)
    if n_gpus == 1:
        mw0 = max(max_workers_total, 1)
        mw1 = 0
        print(f"Pool (GPU {gpu_ids[0]}): {mw0} workers")
    else:
        mw0 = (max_workers_total + 1) // 2
        mw1 = max_workers_total // 2
        if mw0 == 0:
            mw0 = 1
        if mw1 == 0 and max_workers_total > 1:
            mw1 = 1
        print(f"Pool 0 (GPU {gpu_ids[0]}): {mw0} workers")
        print(f"Pool 1 (GPU {gpu_ids[1]}): {mw1} workers")

    futures = []

    if n_gpus == 1:
        g0 = gpu_ids[0]
        with concurrent.futures.ProcessPoolExecutor(max_workers=mw0, initializer=init_worker, initargs=(g0,)) as executor0:
            for episode in range(episode_count):
                future = executor0.submit(
                    evaluate_episode,
                    env_id=env_id,
                    episode=episode,
                    dataset_root=DATASET_ROOT,
                    override_metadata_path=OVERRIDE_METADATA_PATH,
                    action_space=ACTION_SPACE,
                    out_video_dir=OUT_VIDEO_DIR,
                    gui_render=GUI_RENDER
                )
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                print(res)
    else:
        g0, g1 = gpu_ids[0], gpu_ids[1]
        with concurrent.futures.ProcessPoolExecutor(max_workers=mw0, initializer=init_worker, initargs=(g0,)) as executor0, \
             concurrent.futures.ProcessPoolExecutor(max_workers=mw1, initializer=init_worker, initargs=(g1,)) as executor1:
            for episode in range(episode_count):
                if episode % 2 == 0:
                    executor = executor0
                else:
                    executor = executor1
                    if mw1 == 0:
                        executor = executor0
                future = executor.submit(
                    evaluate_episode,
                    env_id=env_id,
                    episode=episode,
                    dataset_root=DATASET_ROOT,
                    override_metadata_path=OVERRIDE_METADATA_PATH,
                    action_space=ACTION_SPACE,
                    out_video_dir=OUT_VIDEO_DIR,
                    gui_render=GUI_RENDER
                )
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                print(res)

def main():
    # Force use of spawn to avoid PyTorch/CUDA fork issues
    mp.set_start_method("spawn", force=True)
    
    args = _parse_args()
    env_ids = [args.envid] if args.envid else DEFAULT_ENV_IDS
    max_workers_total = args.max_workers
    gpu_ids = args.gpus

    print(f"Plan to replay envs: {env_ids} (gpus={gpu_ids})")
    for env_id in env_ids:
        print(f"=== Processing {env_id} ===")
        process_env_id(env_id, max_workers_total, gpu_ids)

if __name__ == "__main__":
    main()
