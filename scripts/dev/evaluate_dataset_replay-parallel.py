# -*- coding: utf-8 -*-
# Script function: Unified dataset replay entry point, supports four action_spaces: joint_angle / ee_pose / keypoint / oracle_planner.
# Consistent with subgoal_evaluate_func.py's main loop and debug fields; the difference is that actions come from EpisodeDatasetResolver.

import os
import sys
import argparse
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
ACTION_SPACE = "ee_pose"
#ACTION_SPACE = "keypoint"
#ACTION_SPACE = "oracle_planner"

GUI_RENDER = False

DATASET_ROOT = "/data/hongzefu/dataset_generate-rpy4-v2"
OVERRIDE_METADATA_PATH = "/data/hongzefu/dataset_generate-rpy4-v2"

# ######## Video saving variables (output directory) start ########
# Video output directory: Independently hardcoded, not aligned with h5 path or env_id
OUT_VIDEO_DIR = "/data/hongzefu/dataset_replay-v2"
# ######## Video saving variables (output directory) end ########
MAX_STEPS = 1000

def _parse_oracle_command(choice_action: Optional[Any]) -> Optional[dict[str, Any]]:
    if not isinstance(choice_action, dict):
        return None
    label = choice_action.get("label")
    if not isinstance(label, str) or not label:
        return None
    return choice_action


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay dataset for one env_id.")
    parser.add_argument(
        "--envid",
        required=True,
        type=str,
        help="Single environment id to replay.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    env_id = args.envid

    env_id_list = BenchmarkEnvBuilder.get_task_list()
    print(f"Available envs: {env_id_list}")
    print(f"Using action_space: {ACTION_SPACE}")
    print(f"Target env: {env_id}")

    if env_id not in env_id_list:
        print(f"Invalid --envid: {env_id}")
        print(f"Valid env ids: {env_id_list}")
        sys.exit(2)

    env_builder = BenchmarkEnvBuilder(
        env_id=env_id,
        dataset="train",
        action_space=ACTION_SPACE,
        gui_render=GUI_RENDER,
        override_metadata_path=OVERRIDE_METADATA_PATH,
    )
    episode_count = env_builder.get_episode_num()
    print(f"[{env_id}] episode_count from metadata: {episode_count}")

    for episode in range(episode_count):
        env = None
        dataset_resolver = None
        try:
            env = env_builder.make_env_for_episode(episode, max_steps=MAX_STEPS)
            dataset_resolver = EpisodeDatasetResolver(
                env_id=env_id,
                episode=episode,
                dataset_directory=DATASET_ROOT,
            )

            # obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.reset()
            obs_batch, info_batch = env.reset()

            # Maintain debug variable semantics from subgoal_evaluate_func.py
            maniskill_obs = obs_batch["maniskill_obs"]
            front_camera = obs_batch["front_camera"]
            wrist_camera = obs_batch["wrist_camera"]
            front_camera_depth = obs_batch["front_camera_depth"]
            wrist_camera_depth = obs_batch["wrist_camera_depth"]
            end_effector_pose_raw = obs_batch["end_effector_pose_raw"]
            end_effector_pose = obs_batch["end_effector_pose"]
            joint_states = obs_batch["joint_states"]
            velocity = obs_batch["velocity"]
            language_goal_list = info_batch["language_goal"]
            language_goal = language_goal_list[0] if language_goal_list else None

            subgoal = info_batch["subgoal"]
            subgoal_grounded = info_batch["subgoal_grounded"]
            available_options = info_batch["available_options"]
            front_camera_extrinsic_opencv = info_batch["front_camera_extrinsic_opencv"]
            front_camera_intrinsic_opencv = info_batch["front_camera_intrinsic_opencv"]
            wrist_camera_extrinsic_opencv = info_batch["wrist_camera_extrinsic_opencv"]
            wrist_camera_intrinsic_opencv = info_batch["wrist_camera_intrinsic_opencv"]

            info = {k: v[-1] for k, v in info_batch.items()}
            # terminated = bool(terminated_batch[-1].item())
            # truncated = bool(truncated_batch[-1].item())

            # ######## Video saving variable preparation (reset phase) start ########
            reset_base_frames = [torch.as_tensor(f).detach().cpu().numpy().copy() for f in front_camera]
            reset_wrist_frames = [torch.as_tensor(f).detach().cpu().numpy().copy() for f in wrist_camera]
            reset_subgoal_grounded = subgoal_grounded
            # ######## Video saving variable preparation (reset phase) end ########

            # ######## Video saving variable initialization start ########
            step = 0
            episode_success = False
            rollout_base_frames: list[np.ndarray] = []
            rollout_wrist_frames: list[np.ndarray] = []
            rollout_subgoal_grounded: list[Any] = []
            # ######## Video saving variable initialization end ########

            while True:
                replay_key = ACTION_SPACE
                action = dataset_resolver.get_step(replay_key, step)
                if ACTION_SPACE == "oracle_planner":
                    action = _parse_oracle_command(action)
                if action is None:
                    break

                obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.step(action)

                # Maintain debug variable semantics from subgoal_evaluate_func.py
                maniskill_obs = obs_batch["maniskill_obs"]
                front_camera = obs_batch["front_camera"]
                wrist_camera = obs_batch["wrist_camera"]
                front_camera_depth = obs_batch["front_camera_depth"]
                wrist_camera_depth = obs_batch["wrist_camera_depth"]
                end_effector_pose_raw = obs_batch["end_effector_pose_raw"]
                end_effector_pose = obs_batch["end_effector_pose"]
                joint_states = obs_batch["joint_states"]
                velocity = obs_batch["velocity"]

                language_goal_list = info_batch["language_goal"]
                subgoal = info_batch["subgoal"]
                subgoal_grounded = info_batch["subgoal_grounded"]
                available_options = info_batch["available_options"]
                front_camera_extrinsic_opencv = info_batch["front_camera_extrinsic_opencv"]
                front_camera_intrinsic_opencv = info_batch["front_camera_intrinsic_opencv"]
                wrist_camera_extrinsic_opencv = info_batch["wrist_camera_extrinsic_opencv"]
                wrist_camera_intrinsic_opencv = info_batch["wrist_camera_intrinsic_opencv"]

                # ######## Video saving variable preparation (replay phase) start ########
                rollout_base_frames.extend(torch.as_tensor(f).detach().cpu().numpy().copy() for f in front_camera)
                rollout_wrist_frames.extend(torch.as_tensor(f).detach().cpu().numpy().copy() for f in wrist_camera)
                rollout_subgoal_grounded.extend(subgoal_grounded)
                # ######## Video saving variable preparation (replay phase) end ########

                info = {k: v[-1] for k, v in info_batch.items()}
                terminated = bool(terminated_batch.item())
                truncated = bool(truncated_batch.item())

                step += 1
                if GUI_RENDER:
                    env.render()
                if truncated:
                    print(f"[{env_id}] episode {episode} step limit exceeded, step {step}.")
                    break
                if terminated:
                    succ = info.get("success")
                    if succ == torch.tensor([True]) or (
                        isinstance(succ, torch.Tensor) and succ.item()
                    ):
                        print(f"[{env_id}] episode {episode} success.")
                        episode_success = True
                    elif info.get("fail", False):
                        print(f"[{env_id}] episode {episode} failed.")
                    break

            # ######## Video saving section start ########
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
            )
            # ######## Video saving section end ########

        except (FileNotFoundError, KeyError) as exc:
            print(f"[{env_id}] episode {episode} data missing, skipping. {exc}")
            continue
        except Exception as exc:
            print(f"[{env_id}] episode {episode} replay exception, skipping. {exc}")
            continue
        finally:
            if dataset_resolver is not None:
                dataset_resolver.close()
            if env is not None:
                env.close()


if __name__ == "__main__":
    main()
