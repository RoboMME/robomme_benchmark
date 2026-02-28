# -*- coding: utf-8 -*-
# Script function: Unified dataset replay entry point, supports four action_spaces: joint_angle / ee_pose / waypoint / multi_choice.
# Consistent with subgoal_evaluate_func.py's main loop and debug fields; the difference is that actions come from EpisodeDatasetResolver.
# [New] Support parallel multi-process replay and alternate task assignment between two GPUs.

import os
import sys
import argparse
import concurrent.futures
import multiprocessing as mp
from typing import Any, Optional

import cv2
import numpy as np
import torch

from robomme.robomme_env import *
from robomme.robomme_env.utils import *
from robomme.env_record_wrapper import (
    BenchmarkEnvBuilder,
    EpisodeDatasetResolver,
)
from robomme.env_record_wrapper.OraclePlannerDemonstrationWrapper import (
    OraclePlannerDemonstrationWrapper,
)
from robomme.robomme_env.utils.choice_action_mapping import (
    _unique_candidates,
    extract_actor_position_xyz,
    project_world_to_pixel,
    select_target_with_pixel,
)
from robomme.robomme_env.utils.save_reset_video import save_robomme_video

AVAILABLE_ACTION_SPACES = [
    "joint_angle",
    "ee_pose",
    "waypoint",
    "multi_choice",
]

GUI_RENDER = False

DATASET_ROOT = "/data/hongzefu/data_0226-test"
OVERRIDE_METADATA_PATH = "/data/hongzefu/data_0226-test"

# ######## Video saving variables (output directory) start ########
# Video output directory: Independently hardcoded, not aligned with h5 path or env_id
OUT_VIDEO_DIR = "/data/hongzefu/dataset_replay-0226-test"
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


def _to_numpy_copy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    return np.array(value, copy=True)


def _to_frame_list(frames_like: Any) -> list[np.ndarray]:
    if frames_like is None:
        return []
    if isinstance(frames_like, torch.Tensor):
        arr = frames_like.detach().cpu().numpy()
        if arr.ndim == 3:
            return [np.array(arr, copy=True)]
        if arr.ndim == 4:
            return [np.array(x, copy=True) for x in arr]
        return []
    if isinstance(frames_like, np.ndarray):
        if frames_like.ndim == 3:
            return [np.array(frames_like, copy=True)]
        if frames_like.ndim == 4:
            return [np.array(x, copy=True) for x in frames_like]
        return []
    if isinstance(frames_like, (list, tuple)):
        out = []
        for frame in frames_like:
            if frame is None:
                continue
            out.append(_to_numpy_copy(frame))
        return out
    try:
        arr = np.asarray(frames_like)
    except Exception:
        return []
    if arr.ndim == 3:
        return [np.array(arr, copy=True)]
    if arr.ndim == 4:
        return [np.array(x, copy=True) for x in arr]
    return []


def _normalize_pixel_xy(pixel_like: Any) -> Optional[list[int]]:
    if not isinstance(pixel_like, (list, tuple, np.ndarray)):
        return None
    if len(pixel_like) < 2:
        return None
    try:
        x = float(pixel_like[0])
        y = float(pixel_like[1])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return [int(np.rint(x)), int(np.rint(y))]


def _find_oracle_wrapper(env_like: Any) -> Optional[OraclePlannerDemonstrationWrapper]:
    current = env_like
    visited: set[int] = set()
    for _ in range(16):
        if current is None:
            return None
        if isinstance(current, OraclePlannerDemonstrationWrapper):
            return current
        obj_id = id(current)
        if obj_id in visited:
            return None
        visited.add(obj_id)
        current = getattr(current, "env", None)
    return None


def _collect_multi_choice_visualization(
    env_like: Any,
    command: dict[str, Any],
) -> tuple[list[list[int]], Optional[list[int]], Optional[list[int]]]:
    clicked_pixel = _normalize_pixel_xy(command.get("position"))
    oracle_wrapper = _find_oracle_wrapper(env_like)
    if oracle_wrapper is None:
        return [], clicked_pixel, None

    try:
        _selected_target, solve_options = oracle_wrapper._build_step_options()
        found_idx, _ = oracle_wrapper._resolve_command(command, solve_options)
    except Exception:
        return [], clicked_pixel, None

    if found_idx is None or found_idx < 0 or found_idx >= len(solve_options):
        return [], clicked_pixel, None

    option = solve_options[found_idx]
    available = option.get("available")
    intrinsic_cv = getattr(oracle_wrapper, "_front_camera_intrinsic_cv", None)
    extrinsic_cv = getattr(oracle_wrapper, "_front_camera_extrinsic_cv", None)
    image_shape = getattr(oracle_wrapper, "_front_rgb_shape", None)

    candidate_pixels: list[list[int]] = []
    if available is not None:
        for actor in _unique_candidates(available):
            actor_pos = extract_actor_position_xyz(actor)
            if actor_pos is None:
                continue
            projected = project_world_to_pixel(
                actor_pos,
                intrinsic_cv=intrinsic_cv,
                extrinsic_cv=extrinsic_cv,
                image_shape=image_shape,
            )
            if projected is None:
                continue
            candidate_pixels.append([int(projected[0]), int(projected[1])])

    matched_pixel: Optional[list[int]] = None
    if available is not None and clicked_pixel is not None:
        matched = select_target_with_pixel(
            available=available,
            pixel_like=clicked_pixel,
            intrinsic_cv=intrinsic_cv,
            extrinsic_cv=extrinsic_cv,
            image_shape=image_shape,
        )
        if isinstance(matched, dict):
            matched_pixel = _normalize_pixel_xy(matched.get("projected_pixel"))

    return candidate_pixels, clicked_pixel, matched_pixel


def _make_blackboard(frame_like: Any) -> np.ndarray:
    frame = _to_numpy_copy(frame_like)
    if frame.ndim < 2:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    h, w = int(frame.shape[0]), int(frame.shape[1])
    if h <= 0 or w <= 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return np.zeros((h, w, 3), dtype=np.uint8)


def _draw_candidate_blackboard(
    frame_like: Any,
    candidate_pixels: list[list[int]],
) -> np.ndarray:
    board = _make_blackboard(frame_like)
    for pixel in candidate_pixels:
        if len(pixel) < 2:
            continue
        cv2.circle(board, (int(pixel[0]), int(pixel[1])), 4, (0, 255, 255), 1)
    return board


def _draw_selection_blackboard(
    frame_like: Any,
    clicked_pixel: Optional[list[int]],
    matched_pixel: Optional[list[int]],
) -> np.ndarray:
    board = _make_blackboard(frame_like)
    if clicked_pixel is not None:
        cv2.drawMarker(
            board,
            (int(clicked_pixel[0]), int(clicked_pixel[1])),
            (255, 255, 0),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=10,
            thickness=1,
        )
    if matched_pixel is not None:
        cv2.circle(board, (int(matched_pixel[0]), int(matched_pixel[1])), 5, (255, 0, 0), 2)
    return board


def init_worker(gpu_id: int):
    """
    Worker process initialization function, sets CUDA_VISIBLE_DEVICES.
    """
    from robomme.logging_utils import setup_logging
    setup_logging(level="DEBUG")
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
        env = env_builder.make_env_for_episode(
            episode,
            max_steps=MAX_STEPS,
            include_maniskill_obs=True,
            include_front_depth=True,
            include_wrist_depth=True,
            include_front_camera_extrinsic=True,
            include_wrist_camera_extrinsic=True,
            include_available_multi_choices=True,
            include_front_camera_intrinsic=True,
            include_wrist_camera_intrinsic=True,
        )
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
        front_camera = _to_frame_list(obs_batch["front_rgb_list"])
        wrist_camera = _to_frame_list(obs_batch["wrist_rgb_list"])
        # Other variables unpacking skipped unless used downstream

        task_goal_list = info_batch["task_goal"]
        # task_goal = task_goal_list[0] if task_goal_list else None
        
        info = {k: v[-1] if isinstance(v, list) and v else v for k, v in info_batch.items()}
        # terminated = bool(terminated_batch[-1].item())
        # truncated = bool(truncated_batch[-1].item())

        # ######## Video saving variable preparation (reset phase) start ########
        reset_base_frames = [_to_numpy_copy(f) for f in front_camera]
        reset_wrist_frames = [_to_numpy_copy(f) for f in wrist_camera]
        reset_right_frames = (
            [_make_blackboard(f) for f in reset_base_frames]
            if action_space == "multi_choice"
            else None
        )
        reset_far_right_frames = (
            [_make_blackboard(f) for f in reset_base_frames]
            if action_space == "multi_choice"
            else None
        )
        _subgoal = info_batch.get("grounded_subgoal_online", "")
        reset_subgoal_grounded = _subgoal if isinstance(_subgoal, list) else [_subgoal] * len(reset_base_frames)
        # ######## Video saving variable preparation (reset phase) end ########

        # ######## Video saving variable initialization start ########
        step = 0
        episode_success = False
        rollout_base_frames: list[np.ndarray] = []
        rollout_wrist_frames: list[np.ndarray] = []
        rollout_right_frames: list[np.ndarray] = []
        rollout_far_right_frames: list[np.ndarray] = []
        rollout_subgoal_grounded: list[Any] = []
        # ######## Video saving variable initialization end ########

        while True:
            replay_key = action_space
            action = dataset_resolver.get_step(replay_key, step)
            if action_space == "multi_choice":
                action = _parse_oracle_command(action)
            if action is None:
                break

            candidate_pixels: list[list[int]] = []
            clicked_pixel: Optional[list[int]] = None
            matched_pixel: Optional[list[int]] = None
            if action_space == "multi_choice":
                candidate_pixels, clicked_pixel, matched_pixel = _collect_multi_choice_visualization(
                    env, action
                )

            obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.step(action)

            # Maintain debug variable semantics from subgoal_evaluate_func.py
            front_camera = _to_frame_list(obs_batch["front_rgb_list"])
            wrist_camera = _to_frame_list(obs_batch["wrist_rgb_list"])

            subgoal_grounded = info_batch["grounded_subgoal_online"]

            # ######## Video saving variable preparation (replay phase) start ########
            rollout_base_frames.extend(_to_numpy_copy(f) for f in front_camera)
            rollout_wrist_frames.extend(_to_numpy_copy(f) for f in wrist_camera)
            if action_space == "multi_choice":
                for base_frame in front_camera:
                    rollout_right_frames.append(
                        _draw_candidate_blackboard(
                            base_frame,
                            candidate_pixels=candidate_pixels,
                        )
                    )
                    rollout_far_right_frames.append(
                        _draw_selection_blackboard(
                            base_frame,
                            clicked_pixel=clicked_pixel,
                            matched_pixel=matched_pixel,
                        )
                    )
            if isinstance(subgoal_grounded, list):
                rollout_subgoal_grounded.extend(subgoal_grounded)
            else:
                rollout_subgoal_grounded.extend([subgoal_grounded] * len(front_camera))
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
            reset_right_frames=reset_right_frames if action_space == "multi_choice" else None,
            rollout_right_frames=rollout_right_frames if action_space == "multi_choice" else None,
            reset_far_right_frames=(
                reset_far_right_frames if action_space == "multi_choice" else None
            ),
            rollout_far_right_frames=(
                rollout_far_right_frames if action_space == "multi_choice" else None
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

def _parse_action_spaces(s: str) -> list[str]:
    tokens = [x.strip() for x in s.split(",") if x.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError(
            "--action_spaces cannot be empty. "
            f"Allowed action spaces: {AVAILABLE_ACTION_SPACES}"
        )

    selected: list[str] = []
    seen: set[str] = set()
    invalid: list[str] = []

    for token in tokens:
        if token not in AVAILABLE_ACTION_SPACES:
            invalid.append(token)
            continue
        if token in seen:
            continue
        seen.add(token)
        selected.append(token)

    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid action space(s): {invalid}. "
            f"Allowed action spaces: {AVAILABLE_ACTION_SPACES}"
        )
    if not selected:
        raise argparse.ArgumentTypeError(
            "--action_spaces has no valid value after parsing. "
            f"Allowed action spaces: {AVAILABLE_ACTION_SPACES}"
        )
    return selected

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
        default=20,
        help="Total max workers (split across GPUs when using 2 GPUs).",
    )
    parser.add_argument(
        "--gpus",
        type=_parse_gpus,
        default=[1],
        help="GPUs to use: '0' (GPU 0 only), '1' (GPU 1 only), '0,1' (both). Default: 0.",
    )
    parser.add_argument(
        "--action_spaces",
        type=_parse_action_spaces,
        #default=AVAILABLE_ACTION_SPACES.copy(),
        default=["multi_choice",],
        help=(
            "Comma-separated action spaces to replay in order. "
            "Available: joint_angle,ee_pose,waypoint,multi_choice. "
            "Default: joint_angle,ee_pose,waypoint,multi_choice."
        ),
    )
    return parser.parse_args()

def process_env_id(
    env_id: str,
    max_workers_total: int,
    gpu_ids: list[int],
    action_spaces: list[str],
):
    # Simple calculation of episode count (do not instantiate env_builder to avoid overhead, or lightweight instantiation)
    # To get episode_count, we need to instantiate env_builder once
    # But we only need the metadata parsing part
    temp_builder = BenchmarkEnvBuilder(
        env_id=env_id,
        dataset="train",
        action_space=action_spaces[0],
        gui_render=False, # Just to read metadata
        override_metadata_path=OVERRIDE_METADATA_PATH,
    )
    episode_count = temp_builder.get_episode_num()
    print(f"[{env_id}] episodes={episode_count}")
    print(f"Parallel execution with max_workers={max_workers_total} on GPU(s) {gpu_ids}")
    
    if episode_count == 0:
        print(f"[{env_id}] No episodes to replay, skip.")
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

    for action_space in action_spaces:
        print(f"[{env_id}] >>> action_space={action_space}")
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
                        action_space=action_space,
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
                        action_space=action_space,
                        out_video_dir=OUT_VIDEO_DIR,
                        gui_render=GUI_RENDER
                    )
                    futures.append(future)
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    print(res)
        print(f"[{env_id}] <<< action_space={action_space} done")

def main():
    from robomme.logging_utils import setup_logging
    setup_logging(level="DEBUG")
    # Force use of spawn to avoid PyTorch/CUDA fork issues
    mp.set_start_method("spawn", force=True)
    
    args = _parse_args()
    env_ids = [args.envid] if args.envid else DEFAULT_ENV_IDS
    max_workers_total = args.max_workers
    gpu_ids = args.gpus
    action_spaces = args.action_spaces

    print(f"Plan to replay envs: {env_ids} (gpus={gpu_ids})")
    print(f"Available action spaces: {AVAILABLE_ACTION_SPACES}")
    print(f"Selected action spaces: {action_spaces}")
    for env_id in env_ids:
        print(f"=== Processing {env_id} ===")
        process_env_id(env_id, max_workers_total, gpu_ids, action_spaces)

if __name__ == "__main__":
    main()
