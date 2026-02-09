# -*- coding: utf-8 -*-
# 脚本功能：统一 dataset replay 入口，支持 joint_angle / ee_pose / keypoint / oracle_planner 四种 action_space。
# 与 evaluate.py 的主循环与调试字段保持一致；差异在于动作来自 EpisodeDatasetResolver。

import os
import re
import sys
from typing import Any, Dict, List, Optional

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
ACTION_SPACE = "joint_angle"
# ACTION_SPACE = "ee_pose"
# ACTION_SPACE = "keypoint"
# ACTION_SPACE = "oracle_planner"

GUI_RENDER = False
MAX_STEPS = 3000
DATASET_ROOT = "/data/hongzefu/dataset_generate"

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


def _flatten_column(batch_dict: Optional[Dict[str, List[Any]]], key: str) -> List[Any]:
    out = []
    for item in (batch_dict or {}).get(key, []) or []:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            out.extend([x for x in item if x is not None])
        else:
            out.append(item)
    return out


def _last_info(info_batch: Optional[Dict[str, List[Any]]], n: int) -> Dict[str, Any]:
    if n <= 0:
        return {}
    idx = n - 1
    return {
        k: v[idx]
        for k, v in (info_batch or {}).items()
        if len(v) > idx and v[idx] is not None
    }


def _to_numpy_frame(frame: Any) -> np.ndarray:
    if hasattr(frame, "detach"):
        frame = frame.detach()
    if hasattr(frame, "cpu"):
        frame = frame.cpu()
    return np.asarray(frame).copy()


def _copy_frame_list(frames: List[Any]) -> List[np.ndarray]:
    return [_to_numpy_frame(frame) for frame in frames]


def _success_from_info(info: Dict[str, Any]) -> bool:
    succ = info.get("success")
    if succ == torch.tensor([True]):
        return True
    if isinstance(succ, torch.Tensor):
        try:
            return bool(succ.item())
        except Exception:
            return False
    return bool(succ)


def _parse_oracle_command(subgoal_text: Optional[str]) -> Optional[Dict[str, Any]]:
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
    return dataset_resolver.get_action(step)


def _get_out_video_dir(action_space: str) -> str:
    suffix = ACTION_SPACE_TO_VIDEO_DIR[action_space]
    out_video_dir = os.path.join(DATASET_ROOT, "videos", suffix)
    os.makedirs(out_video_dir, exist_ok=True)
    return out_video_dir


def main():
    if ACTION_SPACE not in ACTION_SPACE_TO_VIDEO_DIR:
        raise ValueError(
            f"Unsupported ACTION_SPACE: {ACTION_SPACE}. "
            f"Allowed: {sorted(ACTION_SPACE_TO_VIDEO_DIR)}"
        )

    env_id_list = list(DEFAULT_ENV_IDS)
    print(f"Running envs: {env_id_list}")
    print(f"Using action_space: {ACTION_SPACE}")

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
        out_video_dir = _get_out_video_dir(ACTION_SPACE)

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

                n = int(reward_batch.numel()) if hasattr(reward_batch, "numel") else 0
                info = _last_info(info_batch, n)
                terminated = bool(terminated_batch[-1].item()) if n > 0 else False
                truncated = bool(truncated_batch[-1].item()) if n > 0 else False

                reset_base_frames = _copy_frame_list(_flatten_column(obs_batch, "base_camera"))
                reset_wrist_frames = _copy_frame_list(_flatten_column(obs_batch, "wrist_camera"))
                reset_subgoal_grounded = list(_flatten_column(info_batch, "subgoal_grounded"))

                step = 0
                episode_success = False
                replay_base_frames: List[np.ndarray] = []
                replay_wrist_frames: List[np.ndarray] = []
                replay_subgoal_grounded: List[Any] = []

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

                    replay_base_frames.extend(_copy_frame_list(_flatten_column(obs_batch, "base_camera")))
                    replay_wrist_frames.extend(_copy_frame_list(_flatten_column(obs_batch, "wrist_camera")))
                    replay_subgoal_grounded.extend(_flatten_column(info_batch, "subgoal_grounded"))

                    n = int(reward_batch.numel()) if hasattr(reward_batch, "numel") else 0
                    info = _last_info(info_batch, n)
                    terminated = bool(terminated_batch[-1].item()) if n > 0 else False
                    truncated = bool(truncated_batch[-1].item()) if n > 0 else False

                    step += 1
                    if GUI_RENDER:
                        env.render()
                    if truncated:
                        print(f"[{env_id}] episode {episode} 步数超限，步 {step}。")
                        break
                    if terminated:
                        if _success_from_info(info):
                            print(f"[{env_id}] episode {episode} 成功。")
                            episode_success = True
                        elif info.get("fail", False):
                            print(f"[{env_id}] episode {episode} 失败。")
                        break

                success_prefix = "success" if episode_success else "fail"
                mode_prefix = ACTION_SPACE_TO_VIDEO_PREFIX[ACTION_SPACE]
                out_video_path = os.path.join(
                    out_video_dir,
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
