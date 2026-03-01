# -*- coding: utf-8 -*-
# Script function: Unified dataset replay entry point, supporting 4 action spaces: joint_angle / ee_pose / waypoint / multi_choice.
# Consistent with subgoal_evaluate_func.py main loop; difference is actions come from EpisodeDatasetResolver.

import os
from typing import Any, Optional



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



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

# Only enable one ACTION_SPACE; others are commented out for manual switching
ACTION_SPACE = "joint_angle"


GUI_RENDER = False

DATASET_ROOT = "/data/hongzefu/data_0226"

DEFAULT_ENV_IDS = [
#"PickXtimes",
 #"StopCube",
#"SwingXtimes",
 "BinFill",
# "VideoUnmaskSwap",
# "VideoUnmask",
# "ButtonUnmaskSwap",
# "ButtonUnmask",
# "VideoRepick",
# "VideoPlaceButton",
# "VideoPlaceOrder",
#"PickHighlight",
#"InsertPeg",
#"MoveCube",
 #"PatternLock",
 #"RouteStick",
]

OUT_VIDEO_DIR = "/data/hongzefu/dataset_replay"
MAX_STEPS = 1000


def _parse_oracle_command(choice_action: Optional[Any]) -> Optional[dict[str, Any]]:
    if not isinstance(choice_action, dict):
        return None
    choice = choice_action.get("choice")
    if not isinstance(choice, str) or not choice.strip():
        return None
    point = choice_action.get("point")
    if not isinstance(point, (list, tuple, np.ndarray)) or len(point) != 2:
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


def _normalize_point_yx_to_pixel_xy(point_like: Any) -> Optional[list[int]]:
    if not isinstance(point_like, (list, tuple, np.ndarray)):
        return None
    if len(point_like) < 2:
        return None
    try:
        y = float(point_like[0])
        x = float(point_like[1])
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
    clicked_pixel = _normalize_point_yx_to_pixel_xy(command.get("point"))
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




def main():
    from robomme.logging_utils import setup_logging
    setup_logging(level="DEBUG")
    env_id_list = BenchmarkEnvBuilder.get_task_list()
    print(f"Running envs: {env_id_list}")
    print(f"Using action_space: {ACTION_SPACE}")

    #for env_id in env_id_list:
    for env_id in DEFAULT_ENV_IDS:
        env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="train",
            action_space=ACTION_SPACE,
            gui_render=GUI_RENDER,
        )
        episode_count = env_builder.get_episode_num()
        print(f"[{env_id}] episode_count from metadata: {episode_count}")

        env = None
        for episode in range(episode_count):
            if episode !=15:
                continue

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
            try:
                dataset_resolver = EpisodeDatasetResolver(
                    env_id=env_id,
                    episode=episode,
                    dataset_directory=DATASET_ROOT,
                )
            except KeyError as e:
                print(f"[{env_id}] Episode {episode} missing in H5, skipping. ({e})")
                if env is not None:
                    env.close()
                continue

            # ======== Reset ========
            # obs: dict-of-lists (columnar batch, list length = number of demo frames)
            # info: flat dict (last frame values only)
            obs, info = env.reset()

            # --- Explicitly read all obs fields (each is a list) ---
            maniskill_obs = obs["maniskill_obs"]
            front_rgb_list = _to_frame_list(obs["front_rgb_list"])
            wrist_rgb_list = _to_frame_list(obs["wrist_rgb_list"])
            front_depth_list = obs["front_depth_list"]
            wrist_depth_list = obs["wrist_depth_list"]
            end_effector_pose_raw = obs["end_effector_pose_raw"]
            eef_state_list = obs["eef_state_list"]
            joint_state_list = obs["joint_state_list"]
            # velocity = obs["velocity"]
            gripper_state_list = obs["gripper_state_list"]
            front_camera_extrinsic_list = obs["front_camera_extrinsic_list"]
            wrist_camera_extrinsic_list = obs["wrist_camera_extrinsic_list"]

            # --- Explicitly read all info fields (flat dict, last frame values) ---
            task_goal = info["task_goal"]
            simple_subgoal_online = info["simple_subgoal_online"]
            grounded_subgoal_online = info["grounded_subgoal_online"]
            available_multi_choices = info.get("available_multi_choices")
            front_camera_intrinsic = info["front_camera_intrinsic"]
            wrist_camera_intrinsic = info["wrist_camera_intrinsic"]
            status = info.get("status")


            # --- Video saving variable preparation (reset phase) ---
            reset_base_frames = [_to_numpy_copy(f) for f in front_rgb_list]
            reset_wrist_frames = [_to_numpy_copy(f) for f in wrist_rgb_list]
            reset_right_frames = (
                [_make_blackboard(f) for f in reset_base_frames]
                if ACTION_SPACE == "multi_choice"
                else None
            )
            reset_far_right_frames = (
                [_make_blackboard(f) for f in reset_base_frames]
                if ACTION_SPACE == "multi_choice"
                else None
            )
            reset_subgoal_grounded = [grounded_subgoal_online] * len(front_rgb_list)

            step = 0
            episode_success = False
            rollout_base_frames: list[np.ndarray] = []
            rollout_wrist_frames: list[np.ndarray] = []
            rollout_right_frames: list[np.ndarray] = []
            rollout_far_right_frames: list[np.ndarray] = []
            rollout_subgoal_grounded: list[Any] = []

            # ======== Step loop ========
            while True:
                replay_key = ACTION_SPACE
                action = dataset_resolver.get_step(replay_key, step)
                if ACTION_SPACE == "multi_choice":
                    action = _parse_oracle_command(action)
                if action is None:
                    break

                candidate_pixels: list[list[int]] = []
                clicked_pixel: Optional[list[int]] = None
                matched_pixel: Optional[list[int]] = None
                if ACTION_SPACE == "multi_choice":
                    candidate_pixels, clicked_pixel, matched_pixel = _collect_multi_choice_visualization(
                        env, action
                    )

                # step returns: obs (dict-of-lists), reward (scalar tensor),
                #               terminated (scalar tensor), truncated (scalar tensor), info (flat dict)
                obs, reward, terminated, truncated, info = env.step(action)

                # --- Explicitly read all obs fields (dict-of-lists, typically 1 element per list) --                maniskill_obs = obs["maniskill_obs"]
                front_rgb_list = _to_frame_list(obs["front_rgb_list"])
                wrist_rgb_list = _to_frame_list(obs["wrist_rgb_list"])
                front_depth_list = obs["front_depth_list"]
                wrist_depth_list = obs["wrist_depth_list"]
                end_effector_pose_raw = obs["end_effector_pose_raw"]
                eef_state_list = obs["eef_state_list"]
                joint_state_list = obs["joint_state_list"]
                gripper_state_list = obs["gripper_state_list"]
                front_camera_extrinsic_list = obs["front_camera_extrinsic_list"]
                wrist_camera_extrinsic_list = obs["wrist_camera_extrinsic_list"]

                # --- Explicitly read all info fields (flat dict) ---
                task_goal = info["task_goal"]
                simple_subgoal_online = info["simple_subgoal_online"]
                grounded_subgoal_online = info["grounded_subgoal_online"]
                available_multi_choices = info.get("available_multi_choices")
                front_camera_intrinsic = info["front_camera_intrinsic"]
                wrist_camera_intrinsic = info["wrist_camera_intrinsic"]
                status = info.get("status")

                # --- Video saving variable preparation (replay phase) ---
                rollout_base_frames.extend(
                    _to_numpy_copy(f) for f in front_rgb_list
                )
                rollout_wrist_frames.extend(
                    _to_numpy_copy(f) for f in wrist_rgb_list
                )
                if ACTION_SPACE == "multi_choice":
                    for base_frame in front_rgb_list:
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
                rollout_subgoal_grounded.extend([grounded_subgoal_online] * len(front_rgb_list))

                terminated_flag = bool(terminated.item())
                truncated_flag = bool(truncated.item())

                step += 1
                if GUI_RENDER:
                    env.render()
                if truncated_flag:
                    print(f"[{env_id}] episode {episode} steps exceeded, step {step}.")
                    break
                if terminated_flag:
                    if status == "success":
                        print(f"[{env_id}] episode {episode} success.")
                        episode_success = True
                    elif status == "fail":
                        print(f"[{env_id}] episode {episode} failed.")
                    break

            # ======== Video saving ========
            save_robomme_video(
                reset_base_frames=reset_base_frames,
                reset_wrist_frames=reset_wrist_frames,
                rollout_base_frames=rollout_base_frames,
                rollout_wrist_frames=rollout_wrist_frames,
                reset_subgoal_grounded=reset_subgoal_grounded,
                rollout_subgoal_grounded=rollout_subgoal_grounded,
                out_video_dir=OUT_VIDEO_DIR,
                action_space=ACTION_SPACE,
                env_id=env_id,
                episode=episode,
                episode_success=episode_success,
                reset_right_frames=reset_right_frames if ACTION_SPACE == "multi_choice" else None,
                rollout_right_frames=rollout_right_frames if ACTION_SPACE == "multi_choice" else None,
                reset_far_right_frames=(
                    reset_far_right_frames if ACTION_SPACE == "multi_choice" else None
                ),
                rollout_far_right_frames=(
                    rollout_far_right_frames if ACTION_SPACE == "multi_choice" else None
                ),
            )

        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
