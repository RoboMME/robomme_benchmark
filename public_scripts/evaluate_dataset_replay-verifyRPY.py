# -*- coding: utf-8 -*-
# 脚本功能：统一 dataset replay 入口，支持 joint_angle / ee_pose / keypoint / oracle_planner 四种 action_space。
# 与 evaluate.py 的主循环与调试字段保持一致；差异在于动作来自 EpisodeDatasetResolver。

import os
import re
import sys
import json
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
# RPY 汇总函数改为从共享 util 导入，避免依赖自验证脚本文件。
from historybench.env_record_wrapper.rpy_util import summarize_and_print_rpy_sequence
from save_reset_video import save_robomme_video

# 只启用一个 ACTION_SPACE；其他选项保留在注释中供手动切换
ACTION_SPACE = "joint_angle"
#ACTION_SPACE = "ee_pose"
#ACTION_SPACE = "keypoint"
#ACTION_SPACE = "oracle_planner"
#ACTION_SPACE = "oracle_planner"

GUI_RENDER = False
MAX_STEPS = 3000
DATASET_ROOT = "/data/hongzefu/dataset_generate"

DEFAULT_ENV_IDS = [
    #"PickXtimes",
    # "StopCube",
    # "SwingXtimes",
    # "BinFill",
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

# ######## 视频保存变量（输出目录）开始 ########
# 视频输出目录：独立固定写死，不与 h5 路径或 env_id 对齐
OUT_VIDEO_DIR = "/data/hongzefu/dataset_replay"
# RPY 汇总 JSON 输出路径：按 action_space 区分文件，避免多模式互相覆盖。
OUT_RPY_SUMMARY_JSON = os.path.join(OUT_VIDEO_DIR, f"rpy_summary_{ACTION_SPACE}.json")
# ######## 视频保存变量（输出目录）结束 ########

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


def _extract_rpy_from_obs_batch(obs_batch: dict[str, Any]) -> list[np.ndarray]:
    """
    从 columnar 结构的 obs_batch 中提取 RPY 序列（最后三维）。

    输入兼容说明：
    - obs_batch["robot_endeffector_pose"] 的元素可能是 torch.Tensor / np.ndarray / list
    - 形状可能是 (6,), (1, 6), (N, 6) 等，统一 reshape 后按行提取

    返回值：
    - list[np.ndarray]，每个元素是 shape=(3,) 的 [roll, pitch, yaw]（float64）
    - 保持输入原有顺序，供后续按时间顺序汇总与写 JSON。
    """
    rpy_rows: list[np.ndarray] = []
    pose_column = (obs_batch or {}).get("robot_endeffector_pose", None)
    if pose_column is None:
        return rpy_rows

    for item in pose_column:
        if item is None:
            continue
        if isinstance(item, torch.Tensor):
            pose_arr = item.detach().cpu().numpy()
        else:
            pose_arr = np.asarray(item)
        if pose_arr.size == 0:
            continue

        pose_arr = np.asarray(pose_arr, dtype=np.float64)
        if pose_arr.ndim == 1:
            pose_arr = pose_arr.reshape(1, -1)
        else:
            pose_arr = pose_arr.reshape(-1, pose_arr.shape[-1])
        if pose_arr.shape[-1] < 3:
            continue

        # 统一取最后 3 维作为 RPY。前面 3 维是 xyz，不参与此处统计。
        for row in pose_arr[:, -3:]:
            rpy_rows.append(np.asarray(row, dtype=np.float64).copy())
    return rpy_rows


def _write_ordered_rpy_summaries_json(path: str, summaries: list[dict[str, Any]]) -> None:
    """
    将当前已完成 episode 的汇总按“运行顺序”写入 JSON。

    设计点：
    - 每次写全量列表（覆盖写），中途异常时仍能保留已完成部分；
    - summaries 内每条记录含 order_index，顺序与回放执行顺序一致；
    - 使用 ensure_ascii=False + indent=2，便于人工查看与后处理脚本读取。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"summaries": summaries}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    env_id_list = BenchmarkEnvBuilder.get_task_list()
    print(f"Running envs: {env_id_list}")
    print(f"Using action_space: {ACTION_SPACE}")
    # 进程内累计：每完成一个 episode 追加一条，保证 order_index 单调递增。
    ordered_rpy_summaries: list[dict[str, Any]] = []

    

    for env_id in env_id_list:
        env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="train",
            action_space=ACTION_SPACE,
            gui_render=GUI_RENDER,
        )
        episode_count = env_builder.get_episode_num()
        print(f"[{env_id}] episode_count from metadata: {episode_count}")

        for episode in range(episode_count):
            env = None
            dataset_resolver = None
            try:
                env, seed, difficulty = env_builder.make_env_for_episode(episode)
                dataset_resolver = EpisodeDatasetResolver(
                    env_id=env_id,
                    episode=episode,
                    dataset_directory=DATASET_ROOT,
                )

                obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.reset()

                # 保持 evaluate.py 中的调试变量语义
                maniskill_obs = obs_batch["maniskill_obs"]
                front_camera = obs_batch["front_camera"]
                wrist_camera = obs_batch["wrist_camera"]
                front_camera_depth = obs_batch["front_camera_depth"]
                front_camera_segmentation = obs_batch["front_camera_segmentation"]
                wrist_camera_depth = obs_batch["wrist_camera_depth"]
                front_camera_extrinsic_opencv = obs_batch["front_camera_extrinsic_opencv"]
                front_camera_intrinsic_opencv = obs_batch["front_camera_intrinsic_opencv"]
                front_camera_cam2world_opengl = obs_batch["front_camera_cam2world_opengl"]
                wrist_camera_extrinsic_opencv = obs_batch["wrist_camera_extrinsic_opencv"]
                wrist_camera_intrinsic_opencv = obs_batch["wrist_camera_intrinsic_opencv"]
                wrist_camera_cam2world_opengl = obs_batch["wrist_camera_cam2world_opengl"]
                robot_endeffector_pose = obs_batch["robot_endeffector_pose"]
                joint_states = obs_batch["joint_states"]
                velocity = obs_batch["velocity"]
                language_goal_list = info_batch["language_goal"]
                language_goal = language_goal_list[0] if language_goal_list else None

                subgoal = info_batch["subgoal"]
                subgoal_grounded = info_batch["subgoal_grounded"]
                available_options = info_batch["available_options"]

                info = {k: v[-1] for k, v in info_batch.items()}
                terminated = bool(terminated_batch[-1].item())
                truncated = bool(truncated_batch[-1].item())

                # ######## 视频保存变量准备（reset 阶段）开始 ########
                reset_base_frames = [torch.as_tensor(f).detach().cpu().numpy().copy() for f in front_camera]
                reset_wrist_frames = [torch.as_tensor(f).detach().cpu().numpy().copy() for f in wrist_camera]
                reset_subgoal_grounded = subgoal_grounded
                # ######## 视频保存变量准备（reset 阶段）结束 ########

                # ######## 视频保存变量初始化开始 ########
                step = 0
                episode_success = False
                rollout_base_frames: list[np.ndarray] = []
                rollout_wrist_frames: list[np.ndarray] = []
                rollout_subgoal_grounded: list[Any] = []
                # 单个 episode 的 RPY 序列，统计范围包含 reset + replay。
                episode_rpy_seq: list[np.ndarray] = []
                # 先计入 reset 返回批次中的所有 RPY（可能是多步 demonstration + init）。
                episode_rpy_seq.extend(_extract_rpy_from_obs_batch(obs_batch))
                # ######## 视频保存变量初始化结束 ########

                while step < MAX_STEPS:
                    replay_key = ACTION_SPACE
                    action = dataset_resolver.get_step(replay_key, step)
                    if ACTION_SPACE == "oracle_planner":
                        action = _parse_oracle_command(action)
                    if action is None:
                        break

                    obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.step(action)

                    # 保持 evaluate.py 中的调试变量语义
                    maniskill_obs = obs_batch["maniskill_obs"]
                    front_camera = obs_batch["front_camera"]
                    wrist_camera = obs_batch["wrist_camera"]
                    front_camera_depth = obs_batch["front_camera_depth"]
                    front_camera_segmentation = obs_batch["front_camera_segmentation"]
                    wrist_camera_depth = obs_batch["wrist_camera_depth"]
                    front_camera_extrinsic_opencv = obs_batch["front_camera_extrinsic_opencv"]
                    front_camera_intrinsic_opencv = obs_batch["front_camera_intrinsic_opencv"]
                    front_camera_cam2world_opengl = obs_batch["front_camera_cam2world_opengl"]
                    wrist_camera_extrinsic_opencv = obs_batch["wrist_camera_extrinsic_opencv"]
                    wrist_camera_intrinsic_opencv = obs_batch["wrist_camera_intrinsic_opencv"]
                    wrist_camera_cam2world_opengl = obs_batch["wrist_camera_cam2world_opengl"]
                    robot_endeffector_pose = obs_batch["robot_endeffector_pose"]
                    # 每次 env.step 后把该批次所有 RPY 追加到当前 episode 序列。
                    episode_rpy_seq.extend(_extract_rpy_from_obs_batch(obs_batch))

                    joint_states = obs_batch["joint_states"]
                    velocity = obs_batch["velocity"]



                    language_goal_list = info_batch["language_goal"]
                    subgoal = info_batch["subgoal"]
                    subgoal_grounded = info_batch["subgoal_grounded"]
                    available_options = info_batch["available_options"]

                    # ######## 视频保存变量准备（replay 阶段）开始 ########
                    rollout_base_frames.extend(torch.as_tensor(f).detach().cpu().numpy().copy() for f in front_camera)
                    rollout_wrist_frames.extend(torch.as_tensor(f).detach().cpu().numpy().copy() for f in wrist_camera)
                    rollout_subgoal_grounded.extend(subgoal_grounded)
                    # ######## 视频保存变量准备（replay 阶段）结束 ########

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

                # ######## 视频保存部分开始 ########
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
                episode_summary = summarize_and_print_rpy_sequence(
                    episode_rpy_seq,
                    label=f"[{env_id}] episode {episode}",
                )
                # 记录到有序列表，后续统一写 JSON。
                episode_record = {
                    "order_index": len(ordered_rpy_summaries),
                    "env_id": env_id,
                    "episode": int(episode),
                    "action_space": ACTION_SPACE,
                    "summary": episode_summary,
                }
                ordered_rpy_summaries.append(episode_record)
                # 每个 episode 完成后立即落盘，便于中途中断时保留已完成结果。
                _write_ordered_rpy_summaries_json(OUT_RPY_SUMMARY_JSON, ordered_rpy_summaries)
                # ######## 视频保存部分结束 ########

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
    # 全部任务结束后再写一次，确保文件是最新完整状态。
    _write_ordered_rpy_summaries_json(OUT_RPY_SUMMARY_JSON, ordered_rpy_summaries)
    print(f"Saved ordered RPY summaries to: {OUT_RPY_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
