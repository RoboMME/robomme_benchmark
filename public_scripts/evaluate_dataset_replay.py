# -*- coding: utf-8 -*-
# 脚本功能：统一 dataset replay 入口，支持 joint_angle / ee_pose / keypoint / oracle_planner 四种 action_space。
# 与 evaluate.py 的主循环与调试字段保持一致；差异在于动作来自 EpisodeDatasetResolver。

import os
import re
import sys
from typing import Any, Optional

# 将包根目录、上级目录及 scripts 加入 sys.path，便于作为脚本直接运行（不依赖 PYTHONPATH）
_ROOT = os.path.abspath(os.path.dirname(__file__))
_PARENT = os.path.dirname(_ROOT)
_SCRIPTS = os.path.join(_PARENT, "scripts")
for _path in (_PARENT, _ROOT, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import torch

from historybench.HistoryBench_env import *
from historybench.HistoryBench_env.util import *
from historybench.env_record_wrapper import (
    BenchmarkEnvBuilder,
    EpisodeDatasetResolver,
)
from save_reset_video import save_listStep_video

# 只启用一个 ACTION_SPACE；其他选项保留在注释中供手动切换
#ACTION_SPACE = "joint_angle"
ACTION_SPACE = "ee_pose"
#ACTION_SPACE = "keypoint"
# ACTION_SPACE = "oracle_planner"

GUI_RENDER = True
MAX_STEPS = 3000
DATASET_ROOT = "/data/hongzefu/dataset_generate"

DEFAULT_ENV_IDS = [
    "PickXtimes",
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
    # "InsertPeg",
    # "MoveCube",
    # "PatternLock",
    # "RouteStick",
]

ACTION_SPACE_TO_VIDEO_DIR = {
    "joint_angle": "jointangle",
    "ee_pose": "endeffector",
    "keypoint": "keypoint",
    "oracle_planner": "oracleplanner",
}

ACTION_SPACE_TO_VIDEO_PREFIX = {
    "joint_angle": "jointangle",
    "ee_pose": "ee",
    "keypoint": "keypoint",
    "oracle_planner": "oracle",
}

# 视频输出目录：独立固定写死，不与 h5 路径或 env_id 对齐
OUT_VIDEO_DIR = "/data/hongzefu/dataset_generate/videos/replay"

def _parse_oracle_command(subgoal_text: Optional[str]) -> Optional[dict[str, Any]]:
    if not subgoal_text:
        return None
    point = None
    match = re.search(r"<\s*(-?\d+)\s*,\s*(-?\d+)\s*>", subgoal_text)
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        # 数据集文本通常是 <x, y>，Oracle wrapper 期望 [row, col]，即 [y, x]
        point = [y, x]
    return {"action": subgoal_text, "point": point}


def _get_replay_action(
    action_space: str,
    dataset_resolver: EpisodeDatasetResolver,
    step: int,
) -> Optional[Any]:
    if action_space == "oracle_planner":
        subgoal_text = dataset_resolver.get_grounded_subgoal(step)
        return _parse_oracle_command(subgoal_text)
    elif action_space == "ee_pose":
        return dataset_resolver.get_ee_pose_gripper(step)
    elif action_space == "keypoint":
        return dataset_resolver.get_keypoint(step)
    else:  # joint_angle (default)
        return dataset_resolver.get_action(step)


def main():
 
    env_id_list = list(DEFAULT_ENV_IDS)
    print(f"Running envs: {env_id_list}")
    print(f"Using action_space: {ACTION_SPACE}")

    os.makedirs(OUT_VIDEO_DIR, exist_ok=True)

    for env_id in env_id_list:
        env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="train",
            action_space=ACTION_SPACE,
            gui_render=GUI_RENDER,
        )
        episode_count = env_builder.get_episode_num()
        print(f"[{env_id}] episode_count from metadata: {episode_count}")

        h5_path = f"{DATASET_ROOT}/record_dataset_{env_id}.h5"

        for episode in range(episode_count):
            env = None
            dataset_resolver = None
            try:
                env, seed, difficulty = env_builder.make_env_for_episode(episode)
                dataset_resolver = EpisodeDatasetResolver(
                    env_id=env_id,
                    episode=episode,
                    dataset_path=h5_path,
                )

                obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.reset()

                # 保持 evaluate.py 中的调试变量语义
                maniskill_obs = obs_batch["maniskill_obs"]
                base_camera = obs_batch["base_camera"]
                wrist_camera = obs_batch["wrist_camera"]
                base_camera_depth = obs_batch["base_camera_depth"]
                base_camera_segmentation = obs_batch["base_camera_segmentation"]
                wrist_camera_depth = obs_batch["wrist_camera_depth"]
                base_camera_extrinsic_opencv = obs_batch["base_camera_extrinsic_opencv"]
                base_camera_intrinsic_opencv = obs_batch["base_camera_intrinsic_opencv"]
                base_camera_cam2world_opengl = obs_batch["base_camera_cam2world_opengl"]
                wrist_camera_extrinsic_opencv = obs_batch["wrist_camera_extrinsic_opencv"]
                wrist_camera_intrinsic_opencv = obs_batch["wrist_camera_intrinsic_opencv"]
                wrist_camera_cam2world_opengl = obs_batch["wrist_camera_cam2world_opengl"]
                robot_endeffector_p = obs_batch["robot_endeffector_p"]
                robot_endeffector_q = obs_batch["robot_endeffector_q"]
                actions = obs_batch["actions"]
                states = obs_batch["states"]
                velocity = obs_batch["velocity"]
                language_goal_list = obs_batch["language_goal"]
                language_goal = language_goal_list[0] if language_goal_list else None

                subgoal = info_batch["subgoal"]
                subgoal_grounded = info_batch["subgoal_grounded"]
                available_options = info_batch["available_options"]

                info = {k: v[-1] for k, v in info_batch.items()}
                terminated = bool(terminated_batch[-1].item())
                truncated = bool(truncated_batch[-1].item())

                reset_base_frames = []
                for frame in base_camera:
                    f = frame
                    if hasattr(f, "detach"):
                        f = f.detach()
                    if hasattr(f, "cpu"):
                        f = f.cpu()
                    reset_base_frames.append(np.asarray(f).copy())

                reset_wrist_frames = []
                for frame in wrist_camera:
                    f = frame
                    if hasattr(f, "detach"):
                        f = f.detach()
                    if hasattr(f, "cpu"):
                        f = f.cpu()
                    reset_wrist_frames.append(np.asarray(f).copy())

                reset_subgoal_grounded = list(subgoal_grounded) if subgoal_grounded else []

                step = 0
                episode_success = False
                replay_base_frames: list[np.ndarray] = []
                replay_wrist_frames: list[np.ndarray] = []
                replay_subgoal_grounded: list[Any] = []

                while step < MAX_STEPS:
                    action = _get_replay_action(ACTION_SPACE, dataset_resolver, step)
                    if action is None:
                        break

                    obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.step(action)

                    # 保持 evaluate.py 中的调试变量语义
                    maniskill_obs = obs_batch["maniskill_obs"]
                    base_camera = obs_batch["base_camera"]
                    wrist_camera = obs_batch["wrist_camera"]
                    base_camera_depth = obs_batch["base_camera_depth"]
                    base_camera_segmentation = obs_batch["base_camera_segmentation"]
                    wrist_camera_depth = obs_batch["wrist_camera_depth"]
                    base_camera_extrinsic_opencv = obs_batch["base_camera_extrinsic_opencv"]
                    base_camera_intrinsic_opencv = obs_batch["base_camera_intrinsic_opencv"]
                    base_camera_cam2world_opengl = obs_batch["base_camera_cam2world_opengl"]
                    wrist_camera_extrinsic_opencv = obs_batch["wrist_camera_extrinsic_opencv"]
                    wrist_camera_intrinsic_opencv = obs_batch["wrist_camera_intrinsic_opencv"]
                    wrist_camera_cam2world_opengl = obs_batch["wrist_camera_cam2world_opengl"]
                    robot_endeffector_p = obs_batch["robot_endeffector_p"]
                    robot_endeffector_q = obs_batch["robot_endeffector_q"]
                    actions = obs_batch["actions"]
                    states = obs_batch["states"]
                    velocity = obs_batch["velocity"]
                    language_goal_list = obs_batch["language_goal"]

                    subgoal = info_batch["subgoal"]
                    subgoal_grounded = info_batch["subgoal_grounded"]
                    available_options = info_batch["available_options"]

                    for frame in base_camera:
                        f = frame
                        if hasattr(f, "detach"):
                            f = f.detach()
                        if hasattr(f, "cpu"):
                            f = f.cpu()
                        replay_base_frames.append(np.asarray(f).copy())

                    for frame in wrist_camera:
                        f = frame
                        if hasattr(f, "detach"):
                            f = f.detach()
                        if hasattr(f, "cpu"):
                            f = f.cpu()
                        replay_wrist_frames.append(np.asarray(f).copy())

                    for text in subgoal_grounded:
                        if text is not None:
                            replay_subgoal_grounded.append(text)

                    info = {k: v[-1] for k, v in info_batch.items()}
                    terminated = bool(terminated_batch[-1].item())
                    truncated = bool(truncated_batch[-1].item())

                    step += 1
                    if GUI_RENDER:
                        env.render()
                    if truncated:
                        print(f"[{env_id}] episode {episode} 步数超限，步 {step}。")
                        break
                    if terminated:
                        succ = info.get("success")
                        if succ == torch.tensor([True]) or (
                            isinstance(succ, torch.Tensor) and succ.item()
                        ):
                            print(f"[{env_id}] episode {episode} 成功。")
                            episode_success = True
                        elif info.get("fail", False):
                            print(f"[{env_id}] episode {episode} 失败。")
                        break

                success_prefix = "success" if episode_success else "fail"
                mode_prefix = ACTION_SPACE_TO_VIDEO_PREFIX[ACTION_SPACE]
                out_video_path = os.path.join(
                    OUT_VIDEO_DIR,
                    f"{success_prefix}_replay_{mode_prefix}_{env_id}_ep{episode}.mp4",
                )

                merged_base_frames = reset_base_frames + replay_base_frames
                merged_wrist_frames = reset_wrist_frames + replay_wrist_frames
                merged_subgoal_grounded = reset_subgoal_grounded + replay_subgoal_grounded

                if merged_base_frames or merged_wrist_frames:
                    obs_video = {
                        "base_camera": merged_base_frames,
                        "wrist_camera": merged_wrist_frames,
                    }
                    info_video = {"subgoal_grounded": merged_subgoal_grounded}

                    if reset_base_frames and reset_wrist_frames:
                        reset_highlight_count = min(len(reset_base_frames), len(reset_wrist_frames))
                    elif reset_base_frames:
                        reset_highlight_count = len(reset_base_frames)
                    else:
                        reset_highlight_count = len(reset_wrist_frames)

                    save_listStep_video(
                        obs_video,
                        reward_batch,
                        terminated_batch,
                        truncated_batch,
                        info_video,
                        out_video_path,
                        highlight_prefix_count=reset_highlight_count,
                    )
                    print(f"Saved video: {out_video_path}")
                else:
                    print(f"Skipped video (no frames): {out_video_path}")

            except (FileNotFoundError, KeyError) as exc:
                print(f"[{env_id}] episode {episode} 数据缺失，跳过。{exc}")
                continue
            except Exception as exc:
                print(f"[{env_id}] episode {episode} 回放异常，跳过。{exc}")
                continue
            finally:
                if dataset_resolver is not None:
                    dataset_resolver.close()
                if env is not None:
                    env.close()


if __name__ == "__main__":
    main()
