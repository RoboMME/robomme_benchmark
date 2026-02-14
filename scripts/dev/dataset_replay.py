# -*- coding: utf-8 -*-
# Script function: Unified dataset replay entry point, supporting 5 action spaces: joint_angle / ee_pose / ee_quat / keypoint / oracle_planner.
# Consistent with evaluate.py main loop; difference is actions come from EpisodeDatasetResolver.

import os
import re
import sys
from typing import Any, Optional

# # Add package root, parent dir, and scripts to sys.path for direct execution (no PYTHONPATH needed)
# _ROOT = os.path.abspath(os.path.dirname(__file__))
# _PARENT = os.path.dirname(_ROOT)
# _SCRIPTS = os.path.join(_PARENT, "scripts")
# for _path in (_PARENT, _ROOT, _SCRIPTS):
#     if _path not in sys.path:
#         sys.path.insert(0, _path)

import numpy as np
import torch
import cv2
import imageio

from robomme.robomme_env import *
from robomme.robomme_env.utils import *
from robomme.env_record_wrapper import (
    BenchmarkEnvBuilder,
    EpisodeDatasetResolver,
)
from robomme.robomme_env.utils.save_reset_video import save_robomme_video

# Only enable one ACTION_SPACE; others are commented out for manual switching
ACTION_SPACE = "joint_angle"
#ACTION_SPACE = "ee_pose"
#ACTION_SPACE = "ee_quat"
#ACTION_SPACE = "keypoint"
#ACTION_SPACE = "oracle_planner"

GUI_RENDER = False
MAX_STEPS = 3000
DATASET_ROOT = "data_0213"
#OVERRIDE_METADATA_PATH = "/data/hongzefu/dataset_generate-b4"   

DEFAULT_ENV_IDS = [
    # "PickXtimes",
    # "StopCube",
    # "SwingXtimes",
    #"BinFill",
    # "VideoUnmaskSwap",
    # "VideoUnmask",
    # "ButtonUnmaskSwap",
    # "ButtonUnmask",
     "VideoRepick",
    # "VideoPlaceButton",
    # "VideoPlaceOrder",
   # "PickHighlight",
    # "InsertPeg",
    # "MoveCube",
    # "PatternLock",
    # "RouteStick",
]

# ######## Video saving variables (output dir) start ########
# Video output directory: Independent fixed path, not aligned with h5 path or env_id
OUT_VIDEO_DIR = "/data/hongzefu/dataset_replay"
# ######## Video saving variables (output dir) end ########

def _parse_oracle_command(subgoal_text: Optional[str]) -> Optional[dict[str, Any]]:
    if not subgoal_text:
        return None
    point = None
    match = re.search(r"<\s*(-?\d+)\s*,\s*(-?\d+)\s*>", subgoal_text)
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        # Dataset text is usually <x, y>, Oracle wrapper expects [row, col], i.e., [y, x]
        point = [y, x]
    return {"action": subgoal_text, "point": point}


def main():
    env_id_list = BenchmarkEnvBuilder.get_task_list()
    print(f"Running envs: {env_id_list}")
    print(f"Using action_space: {ACTION_SPACE}")

    

    #for env_id in env_id_list:
    for env_id in ['RouteStick']:
        env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="train",
            action_space=ACTION_SPACE,
            gui_render=GUI_RENDER,
            #override_metadata_path=OVERRIDE_METADATA_PATH,
        )
        episode_count = env_builder.get_episode_num()
        print(f"[{env_id}] episode_count from metadata: {episode_count}")

        for episode in range(episode_count):

            env = None
            dataset_resolver = None
            env, seed, difficulty = env_builder.make_env_for_episode(episode)
            import pdb; pdb.set_trace()
            dataset_resolver = EpisodeDatasetResolver(
                env_id=env_id,
                episode=episode,
                dataset_directory=DATASET_ROOT,
                
            )

            # obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.reset()
            obs, info = env.reset()
            
            total_images = []
            front_rgb_list = obs["front_camera"]
            wrist_rgb_list = obs["wrist_camera"]
            
            for i in range(len(front_rgb_list)):
                front_rgb = front_rgb_list[i].cpu().numpy()
                wrist_rgb = wrist_rgb_list[i].cpu().numpy()
                concat_image = np.concatenate([front_rgb, wrist_rgb], axis=1)
                if i < len(front_rgb_list) - 1:
                    # add a red border to the image to indicate the video input
                    concat_image = cv2.rectangle(concat_image, (0, 0), 
                            (concat_image.shape[1], concat_image.shape[0]), (255, 0, 0), 10)
                total_images.append(concat_image)
            
            imageio.mimsave(f"test_env_eval.mp4", total_images, fps=30)
            import pdb; pdb.set_trace()
            
            

            # Keep debug variable semantics from evaluate.py
            maniskill_obs = obs_batch["maniskill_obs"]
            front_camera = obs_batch["front_camera"]
            wrist_camera = obs_batch["wrist_camera"]
            front_camera_depth = obs_batch["front_camera_depth"]
            wrist_camera_depth = obs_batch["wrist_camera_depth"]
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

            # #todo: Save the last two front camera images, stitch them side-by-side and add annotations
            # if len(front_camera) >= 2:
            #     def _tensor_to_numpy_img(f):
            #         img = torch.as_tensor(f).detach().cpu().numpy()
            #         if img.dtype != np.uint8:
            #             # Assume [0,1] float, convert to [0,255] uint8
            #             if img.max() <= 1.0:
            #                 img = (img * 255).astype(np.uint8)
            #             else:
            #                 img = img.astype(np.uint8)
            #         return img.copy()  # Ensure writable copy

            #     def _draw_text_with_wrap(img, text, position=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, color=(0, 255, 0), thickness=1):
            #         """Draw text with automatic wrapping"""
            #         if not text:
            #             return img
                    
            #         img_h, img_w = img.shape[:2]
            #         x, y = position
            #         line_height = int(30 * font_scale) + 5
                    
            #         words = text.split(' ')
            #         current_line = ""
                    
            #         # Simple word-by-word wrapping logic
            #         for word in words:
            #             test_line = current_line + word + " "
            #             (w, h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            #             if x + w > img_w - 10:  # Leave right margin
            #                 # Draw current line
            #                 cv2.putText(img, current_line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            #                 # Reset new line
            #                 current_line = word + " "
            #                 y += line_height
            #             else:
            #                 current_line = test_line
                    
            #         # Draw last line
            #         if current_line:
            #             cv2.putText(img, current_line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
                    
            #         return img

            #     img_prev = _tensor_to_numpy_img(front_camera[-2])
            #     img_curr = _tensor_to_numpy_img(front_camera[-1])

            #     # Add corresponding subgoal to each image
            #     # Note: Length of subgoal_grounded may match front_camera, take second to last and last
            #     subgoal_text_prev = str(subgoal_grounded[-2]) if len(subgoal_grounded) >= 2 else "No Subgoal"
            #     subgoal_text_curr = str(subgoal_grounded[-1]) if subgoal_grounded else "No Subgoal"
                
            #     # Draw text
            #     _draw_text_with_wrap(img_prev, f"Prev: {subgoal_text_prev}")
            #     _draw_text_with_wrap(img_curr, f"Curr: {subgoal_text_curr}")

            #     # Horizontal stitching
            #     concat_img = np.hstack((img_prev, img_curr))
                
            #     # Convert to BGR for saving
            #     concat_img_bgr = cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR)
                
            #     save_path = os.path.join(OUT_VIDEO_DIR, f"{env_id}-{episode}-reset-comparison.png")
            #     os.makedirs(OUT_VIDEO_DIR, exist_ok=True)
            #     cv2.imwrite(save_path, concat_img_bgr)
            #     print(f"[{env_id}] episode {episode} reset comparison image saved to {save_path}")
            

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

            while step < MAX_STEPS:
                replay_key = ACTION_SPACE
                action = dataset_resolver.get_step(replay_key, step)
                if ACTION_SPACE == "oracle_planner":
                    action = _parse_oracle_command(action)
                if action is None:
                    break

                obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.step(action)

                # Keep debug variable semantics from evaluate.py
                maniskill_obs = obs_batch["maniskill_obs"]
                front_camera = obs_batch["front_camera"]
                wrist_camera = obs_batch["wrist_camera"]
                front_camera_depth = obs_batch["front_camera_depth"]
                wrist_camera_depth = obs_batch["wrist_camera_depth"]
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
                terminated = bool(terminated_batch[-1].item())
                truncated = bool(truncated_batch[-1].item())

                step += 1
                if GUI_RENDER:
                    env.render()
                if truncated:
                    print(f"[{env_id}] episode {episode} steps exceeded, step {step}.")
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

            # ######## Video saving part start ########
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
            # ######## Video saving part end ########

        env.close()


if __name__ == "__main__":
    main()
