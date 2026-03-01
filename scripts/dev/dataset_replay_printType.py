# -*- coding: utf-8 -*-
# Script function: Unified dataset replay entry point, supporting 4 action spaces: joint_angle / ee_pose / waypoint / multi_choice.
# Consistent with subgoal_evaluate_func.py main loop; difference is actions come from EpisodeDatasetResolver.

import os
from typing import Any, Optional

import numpy as np
import torch

from robomme.robomme_env import *
from robomme.robomme_env.utils import *
from robomme.env_record_wrapper import (
    BenchmarkEnvBuilder,
    EpisodeDatasetResolver,
)

# Only enable one ACTION_SPACE; others are commented out for manual switching
#ACTION_SPACE = "joint_angle"
ACTION_SPACE = "waypoint"


GUI_RENDER = False

DATASET_ROOT = "/data/hongzefu/data_0225"

DEFAULT_ENV_IDS = [
    # "PickXtimes",
    # "StopCube",
    # "SwingXtimes",
    #"BinFill",
    "VideoUnmaskSwap",
    # "VideoUnmask",
    # "ButtonUnmaskSwap",
    # "ButtonUnmask",
     #"VideoRepick",
    # "VideoPlaceButton",
    # "VideoPlaceOrder",
   # "PickHighlight",
    # "InsertPeg",
    # "MoveCube",
    # "PatternLock",
    # "RouteStick",
]

MAX_STEPS = 1000


def _describe(value: Any, indent: int = 0) -> str:
    """Recursively describe a value's type, shape, and content summary."""
    prefix = "  " * indent
    if isinstance(value, torch.Tensor):
        return f"{prefix}Tensor  dtype={value.dtype}  shape={tuple(value.shape)}  device={value.device}"
    elif isinstance(value, np.ndarray):
        return f"{prefix}ndarray  dtype={value.dtype}  shape={value.shape}"
    elif isinstance(value, list):
        if len(value) == 0:
            return f"{prefix}list[]  (empty)"
        lines = [f"{prefix}list[{len(value)}]"]
        for i, item in enumerate(value):
            lines.append(f"{prefix}  [{i}]: {_describe(item, 0)}")
            if i >= 2:
                lines.append(f"{prefix}  ... (only first 3 shown)")
                break
        return "\n".join(lines)
    elif isinstance(value, dict):
        lines = [f"{prefix}dict  keys={list(value.keys())}"]
        for k, v in value.items():
            lines.append(f"{prefix}  '{k}': {_describe(v, 0)}")
        return "\n".join(lines)
    elif isinstance(value, (int, float, bool, str)):
        return f"{prefix}{type(value).__name__}  value={repr(value)}"
    elif value is None:
        return f"{prefix}None"
    else:
        return f"{prefix}{type(value).__name__}  repr={repr(value)[:80]}"


def _print_obs(obs: dict, tag: str):
    """Print data formats of all fields in the obs dict."""
    print(f"\n{'='*60}")
    print(f"[{tag}] obs fields:")
    print(f"{'='*60}")
    # maniskill_obs not printed (data volume is large)
    _ = obs["maniskill_obs"]
    front_rgb_list = obs["front_rgb_list"]
    wrist_rgb_list = obs["wrist_rgb_list"]
    front_depth_list = obs["front_depth_list"]
    wrist_depth_list = obs["wrist_depth_list"]
    end_effector_pose_raw = obs["end_effector_pose_raw"]
    eef_state_list = obs["eef_state_list"]
    joint_state_list = obs["joint_state_list"]

    gripper_state_list = obs["gripper_state_list"]
    front_camera_extrinsic_list = obs["front_camera_extrinsic_list"]
    wrist_camera_extrinsic_list = obs["wrist_camera_extrinsic_list"]

    fields = {
        "front_rgb_list": front_rgb_list,
        "wrist_rgb_list": wrist_rgb_list,
        "front_depth_list": front_depth_list,
        "wrist_depth_list": wrist_depth_list,
        "end_effector_pose_raw": end_effector_pose_raw,
        "eef_state_list": eef_state_list,
        "joint_state_list": joint_state_list,

        "gripper_state_list": gripper_state_list,
        "front_camera_extrinsic_list": front_camera_extrinsic_list,
        "wrist_camera_extrinsic_list": wrist_camera_extrinsic_list,
    }
    for name, val in fields.items():
        print(f"  obs['{name}']:")
        print(_describe(val, indent=2))
    return fields


def _print_info(info: dict, tag: str):
    """Print data formats of all fields in the info dict."""
    print(f"\n[{tag}] info fields:")
    print(f"{'-'*60}")
    task_goal = info["task_goal"]
    simple_subgoal_online = info["simple_subgoal_online"]
    grounded_subgoal_online = info["grounded_subgoal_online"]
    available_multi_choices = info.get("available_multi_choices")
    front_camera_intrinsic = info["front_camera_intrinsic"]
    wrist_camera_intrinsic = info["wrist_camera_intrinsic"]
    status = info.get("status")

    fields = {
        "task_goal": task_goal,
        "simple_subgoal_online": simple_subgoal_online,
        "grounded_subgoal_online": grounded_subgoal_online,
        "available_multi_choices": available_multi_choices,
        "front_camera_intrinsic": front_camera_intrinsic,
        "wrist_camera_intrinsic": wrist_camera_intrinsic,
        "status": status,
    }
    for name, val in fields.items():
        print(f"  info['{name}']:")
        print(_describe(val, indent=2))
    return fields


def _print_step_extras(reward, terminated, truncated, tag: str):
    """Print data formats of reward / terminated / truncated."""
    print(f"\n[{tag}] reward / terminated / truncated:")
    print(f"{'-'*60}")
    print(f"  reward:     {_describe(reward, 0)}")
    print(f"  terminated: {_describe(terminated, 0)}")
    print(f"  truncated:  {_describe(truncated, 0)}")


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


def main():
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
                dataset_directory=DATASET_ROOT,
            )

            # obs: dict-of-lists (columnar batch, list length = number of demo frames)
            # info: flat dict (last frame values only)
            obs, info = env.reset()

            # --- Print all obs / info field types (reset) ---
            _print_obs(obs, tag=f"{env_id} ep{episode} RESET")
            _print_info(info, tag=f"{env_id} ep{episode} RESET")

            step = 0
            episode_success = False

            # ======== Step loop ========
            while True:
                replay_key = ACTION_SPACE
                action = dataset_resolver.get_step(replay_key, step)
                if ACTION_SPACE == "multi_choice":
                    action = _parse_oracle_command(action)
                if action is None:
                    break

                # step returns: obs (dict-of-lists), reward (scalar tensor),
                #               terminated (scalar tensor), truncated (scalar tensor), info (flat dict)
                obs, reward, terminated, truncated, info = env.step(action)

                # --- Print all obs / info / reward / terminated / truncated field types (step) ---
                _print_obs(obs, tag=f"{env_id} ep{episode} STEP{step}")
                _print_info(info, tag=f"{env_id} ep{episode} STEP{step}")
                _print_step_extras(reward, terminated, truncated, tag=f"{env_id} ep{episode} STEP{step}")

                terminated_flag = bool(terminated.item())
                truncated_flag = bool(truncated.item())

                step += 1
                if GUI_RENDER:
                    env.render()
                if truncated_flag:
                    print(f"[{env_id}] episode {episode} steps exceeded, step {step}.")
                    break
                if terminated_flag:
                    status = info.get("status")
                    if status == "success":
                        print(f"[{env_id}] episode {episode} success.")
                        episode_success = True
                    elif status == "fail":
                        print(f"[{env_id}] episode {episode} failed.")
                    break

        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
