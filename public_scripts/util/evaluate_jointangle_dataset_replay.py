# 从 dataset_generate 读取数据集并回放轨迹（跳过 demonstration 阶段，只执行非演示动作）

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

# 将包根目录加入 path，便于作为脚本直接运行
_ROOT = os.path.abspath(os.path.dirname(__file__))
_PARENT = os.path.dirname(_ROOT)
for _path in (_PARENT, _ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import gymnasium as gym
import torch

from historybench.HistoryBench_env import *
from historybench.HistoryBench_env.util import *
from historybench.env_record_wrapper import (
    EpisodeConfigResolver,
    EpisodeDatasetResolver,
    DemonstrationWrapper,
)

from save_reset_video import save_listStep_video

# 数据集根目录（包含 record_dataset_*_metadata.json 与 record_dataset_*.h5）
DATASET_ROOT = "/data/hongzefu/dataset_generate"
DEFAULT_ENV_IDS = [
#     "PickXtimes",
#     "StopCube",
#     "SwingXtimes",
#     "BinFill",
#     "VideoUnmaskSwap",
#     "VideoUnmask",
#     "ButtonUnmaskSwap",
#     "ButtonUnmask",
#     "VideoRepick",
#     "VideoPlaceButton",
#     "VideoPlaceOrder",
#     "PickHighlight",
  "InsertPeg",
#     "MoveCube",
#     "PatternLock",
#     "RouteStick",
]


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Replay joint-angle dataset trajectories."
    )
    parser.add_argument(
        "--env-id",
        dest="env_ids",
        action="append",
        default=None,
        help=(
            "Specify env id(s) to run. Can be passed multiple times or as comma-separated values. "
            "Default: run all 16 envs."
        ),
    )
    return parser.parse_args()


def _resolve_env_ids(env_args):
    if not env_args:
        return list(DEFAULT_ENV_IDS)

    selected = []
    for raw in env_args:
        for env_id in raw.split(","):
            env_id = env_id.strip()
            if not env_id:
                continue
            if env_id not in DEFAULT_ENV_IDS:
                raise ValueError(
                    f"Unknown env_id '{env_id}'. Available: {', '.join(DEFAULT_ENV_IDS)}"
                )
            if env_id not in selected:
                selected.append(env_id)

    return selected if selected else list(DEFAULT_ENV_IDS)


def _flatten_column(batch_dict, key):
    out = []
    for item in (batch_dict or {}).get(key, []) or []:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            out.extend([x for x in item if x is not None])
        else:
            out.append(item)
    return out


def _last_info(info_batch, n):
    if n <= 0:
        return {}
    idx = n - 1
    return {k: v[idx] for k, v in (info_batch or {}).items() if len(v) > idx and v[idx] is not None}


def main():
    """
    从 DATASET_ROOT 读取数据集，回放所有 episode；
    每个 episode 仅回放非 demonstration 的动作（跳过 demonstration 步）。
    """
    # ---------- 全局配置 ----------
    gui_render = False
    max_steps = 3000
    render_mode = "human" if gui_render else "rgb_array"
    args = _parse_args()
    env_id_list = _resolve_env_ids(args.env_ids)
    print(f"Running envs: {env_id_list}")

    for env_id in env_id_list:
        # ---------- 按 env_id 创建配置解析器与数据集路径 ----------
        metadata_path = f"{DATASET_ROOT}/record_dataset_{env_id}_metadata.json"
        config_resolver = EpisodeConfigResolver(
            env_id=env_id,
            metadata_path=metadata_path,
            render_mode=render_mode,
            gui_render=gui_render,
            max_steps_without_demonstration=max_steps,
        )
        h5_path = f"{DATASET_ROOT}/record_dataset_{env_id}.h5"
        out_video_dir = os.path.join("/data/hongzefu/dataset_generate", "videos", "jointangle")
        os.makedirs(out_video_dir, exist_ok=True)

        for episode in range(50):
            env = None
            dataset_resolver = None
            # 只跑以下 (env_id, episode)，其余跳过
            # RUN_EPISODES = {

            #     ("VideoPlaceButton", 9),
            #     ("VideoPlaceButton", 3),

            # }
            # if (env_id, episode) not in RUN_EPISODES:
            #     continue

            try:
                # ---------- 为当前 episode 创建环境与数据集解析器 ----------
                env, seed, difficulty = config_resolver.make_env_for_episode(episode)
                env.save_failed_match_env_id = env_id
                env.save_failed_match_episode = episode
                dataset_resolver = EpisodeDatasetResolver(
                    env_id=env_id,
                    episode=episode,
                    dataset_path=h5_path,
                )

                # ---------- 重置环境（含 demonstration 阶段），并保存 reset 带字幕视频 ----------
                obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.reset()

                # 从 obs_batch / info_batch 抽取并合并成列表
                maniskill_obs = (obs_batch or {}).get("maniskill_obs", [])
                base_camera = _flatten_column(obs_batch, "base_camera")
                wrist_camera = _flatten_column(obs_batch, "wrist_camera")
                base_camera_depth = _flatten_column(obs_batch, "base_camera_depth")
                base_camera_segmentation = _flatten_column(obs_batch, "base_camera_segmentation")
                wrist_camera_depth = _flatten_column(obs_batch, "wrist_camera_depth")
                base_camera_extrinsic_opencv = _flatten_column(obs_batch, "base_camera_extrinsic_opencv")
                base_camera_intrinsic_opencv = _flatten_column(obs_batch, "base_camera_intrinsic_opencv")
                base_camera_cam2world_opengl = _flatten_column(obs_batch, "base_camera_cam2world_opengl")
                wrist_camera_extrinsic_opencv = _flatten_column(obs_batch, "wrist_camera_extrinsic_opencv")
                wrist_camera_intrinsic_opencv = _flatten_column(obs_batch, "wrist_camera_intrinsic_opencv")
                wrist_camera_cam2world_opengl = _flatten_column(obs_batch, "wrist_camera_cam2world_opengl")
                robot_endeffector_p = _flatten_column(obs_batch, "robot_endeffector_p")
                robot_endeffector_q = _flatten_column(obs_batch, "robot_endeffector_q")
                actions = _flatten_column(obs_batch, "actions")
                states = _flatten_column(obs_batch, "states")
                velocity = _flatten_column(obs_batch, "velocity")
                language_goal_list = (obs_batch or {}).get("language_goal", [])
                language_goal = language_goal_list[0] if language_goal_list else None
                subgoal = _flatten_column(info_batch, "subgoal")
                subgoal_grounded = _flatten_column(info_batch, "subgoal_grounded")

                # 保留 reset 阶段帧与字幕，后续会拼接到 replay 视频前缀。
                reset_base_frames = []
                for frame in base_camera:
                    if hasattr(frame, "cpu"):
                        frame = frame.cpu()
                    reset_base_frames.append(np.asarray(frame).copy())
                reset_wrist_frames = []
                for frame in wrist_camera:
                    if hasattr(frame, "cpu"):
                        frame = frame.cpu()
                    reset_wrist_frames.append(np.asarray(frame).copy())
                reset_subgoal_grounded = list(subgoal_grounded) if subgoal_grounded else []

                # ---------- 按 step 回放：从数据集取关节角动作执行；每步收集帧与字幕（拷贝）用于保存视频 ----------
                step = 0
                episode_success = False
                replay_base_frames = []
                replay_wrist_frames = []
                replay_subgoal_grounded = []
                while True:
                    action = dataset_resolver.get_action(step)
                    if action is None:
                        break
                    obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.step(action)

                    # 从 batch 读取（供调试或后续逻辑）
                    maniskill_obs = (obs_batch or {}).get("maniskill_obs", [])
                    base_camera = _flatten_column(obs_batch, "base_camera")
                    wrist_camera = _flatten_column(obs_batch, "wrist_camera")
                    base_camera_depth = _flatten_column(obs_batch, "base_camera_depth")
                    base_camera_segmentation = _flatten_column(obs_batch, "base_camera_segmentation")
                    wrist_camera_depth = _flatten_column(obs_batch, "wrist_camera_depth")
                    base_camera_extrinsic_opencv = _flatten_column(obs_batch, "base_camera_extrinsic_opencv")
                    base_camera_intrinsic_opencv = _flatten_column(obs_batch, "base_camera_intrinsic_opencv")
                    base_camera_cam2world_opengl = _flatten_column(obs_batch, "base_camera_cam2world_opengl")
                    wrist_camera_extrinsic_opencv = _flatten_column(obs_batch, "wrist_camera_extrinsic_opencv")
                    wrist_camera_intrinsic_opencv = _flatten_column(obs_batch, "wrist_camera_intrinsic_opencv")
                    wrist_camera_cam2world_opengl = _flatten_column(obs_batch, "wrist_camera_cam2world_opengl")
                    robot_endeffector_p = _flatten_column(obs_batch, "robot_endeffector_p")
                    robot_endeffector_q = _flatten_column(obs_batch, "robot_endeffector_q")
                    last_action = _flatten_column(obs_batch, "actions")
                    state = _flatten_column(obs_batch, "states")
                    velocity = _flatten_column(obs_batch, "velocity")
                    language_goal_list = (obs_batch or {}).get("language_goal", [])
                    language_goal = language_goal_list[-1] if language_goal_list else None
                    subgoal = _flatten_column(info_batch, "subgoal")
                    subgoal_grounded = _flatten_column(info_batch, "subgoal_grounded")
                    # 每步拷贝当前帧与字幕，避免与 env 内部 buffer 共享导致视频每帧相同
                    if base_camera:
                        frame = base_camera[-1]
                        if hasattr(frame, "cpu"):
                            frame = frame.cpu()
                        replay_base_frames.append(np.asarray(frame).copy())
                    if wrist_camera:
                        frame = wrist_camera[-1]
                        if hasattr(frame, "cpu"):
                            frame = frame.cpu()
                        replay_wrist_frames.append(np.asarray(frame).copy())
                    if subgoal_grounded:
                        replay_subgoal_grounded.append(subgoal_grounded[-1])

                    n = int(reward_batch.numel()) if hasattr(reward_batch, "numel") else 0
                    info = _last_info(info_batch, n)
                    terminated = bool(terminated_batch[-1].item()) if n > 0 else False
                    truncated = bool(truncated_batch[-1].item()) if n > 0 else False

                    step += 1
                    if gui_render:
                        env.render()
                    if truncated:
                        print(f"[{env_id}] episode {episode} 步数超限，步 {step}。")
                        break
                    if terminated:
                        if info.get("success") == torch.tensor([True]) or (
                            isinstance(info.get("success"), torch.Tensor) and info.get("success").item()
                        ):
                            print(f"[{env_id}] episode {episode} 成功。")
                            episode_success = True
                        elif info.get("fail", False):
                            print(f"[{env_id}] episode {episode} 失败。")
                        break

                # ---------- 保存本 episode 回放视频（用本循环内收集的帧与字幕，不调用 env.save_video） ----------
                success_prefix = "success" if episode_success else "fail"
                out_video_path = os.path.join(out_video_dir, f"{success_prefix}_replay_{env_id}_ep{episode}.mp4")
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
