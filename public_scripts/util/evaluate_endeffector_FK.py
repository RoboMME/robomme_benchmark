# -*- coding: utf-8 -*-
# 脚本功能：从 dataset_generate 读取数据集并回放轨迹
# 特点：跳过 demonstration 阶段，仅执行非演示动作；使用末端执行器位姿 + IK 生成关节动作进行回放

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

# 将包根目录及上级目录加入 sys.path，便于作为脚本直接运行（不依赖 PYTHONPATH）
_ROOT = os.path.abspath(os.path.dirname(__file__))
_PARENT = os.path.dirname(_ROOT)
for _path in (_PARENT, _ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import gymnasium as gym
import torch

from save_reset_video import save_listStep_video

# HistoryBench 环境及工具
from historybench.HistoryBench_env import *
from historybench.HistoryBench_env.util import *
# 用于根据元数据创建环境、从 h5 读取 episode 动作/位姿
from historybench.env_record_wrapper import (
    EpisodeConfigResolver,
    EpisodeDatasetResolver,
)

# 数据集根目录：需包含 record_dataset_{env_id}_metadata.json 与 record_dataset_{env_id}.h5
DATASET_ROOT = "/data/hongzefu/data_1206"
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


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Replay end-effector dataset trajectories."
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


def _init_fk_context(env):
    """
    Initialize FK context from current environment:
    - robot articulation
    - pinocchio model
    - ee link index for panda_hand_tcp
    """
    robot = env.unwrapped.agent.robot
    pin_model = robot.create_pinocchio_model()
    links = robot.get_links()
    ee_link_idx = None
    link_names = []
    for link in links:
        if hasattr(link, "get_name"):
            name = link.get_name()
        else:
            name = getattr(link, "name", "")
        link_names.append(name)
        if name == "panda_hand_tcp":
            ee_link_idx = link.index
    if ee_link_idx is None:
        raise RuntimeError(
            "FK init failed: link 'panda_hand_tcp' not found. "
            f"Available links: {link_names}"
        )
    return robot, pin_model, ee_link_idx


def _joint_to_ee_action_with_fk(joint_action, env, pin_model, ee_link_idx, step):
    """
    Convert joint action to ee_pose action using FK.
    Input joint_action: [arm(>=7), gripper(optional)]
    Output ee action: [ee_p(3), ee_q(4), gripper(1)]
    """
    env_id = getattr(getattr(env.unwrapped, "spec", None), "id", "<unknown_env>")
    action = np.asarray(joint_action, dtype=np.float64).flatten()
    if action.size < 7:
        raise ValueError(
            f"[{env_id}] FK replay expects joint action length >= 7 at step {step}, got {action.size}"
        )
    arm_q = action[:7]
    gripper = float(action[-1]) if action.size >= 8 else -1.0

    robot = env.unwrapped.agent.robot
    current_qpos = robot.get_qpos()
    current_qpos_np = (
        current_qpos.detach().cpu().numpy()
        if hasattr(current_qpos, "detach")
        else np.asarray(current_qpos)
    )
    full_qpos = np.asarray(current_qpos_np[0], dtype=np.float64).copy()
    full_qpos[:7] = arm_q
    try:
        pin_model.compute_forward_kinematics(full_qpos)
        ee_pose_local = pin_model.get_link_pose(ee_link_idx)
        ee_pose_world = robot.pose.sp * ee_pose_local
    except Exception as exc:
        raise RuntimeError(
            f"[{env_id}] FK failed at step {step}: {exc}"
        ) from exc

    ee_p = np.asarray(ee_pose_world.p, dtype=np.float64).reshape(-1)[:3]
    ee_q = np.asarray(ee_pose_world.q, dtype=np.float64).reshape(-1)[:4]
    return np.concatenate([ee_p, ee_q, np.array([gripper], dtype=np.float64)])


def main():
    """
    主流程：从 DATASET_ROOT 读取数据集，按 episode 回放。
    - 每个 episode 仅回放非 demonstration 步（跳过演示阶段）。
    - 使用数据集中记录的末端执行器位姿 (ee pose)，经 IK 得到关节角后再执行，以验证末端轨迹回放效果。
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
            action_space="ee_pose",
        )
        h5_path = f"{DATASET_ROOT}/record_dataset_{env_id}.h5"
        out_video_dir = os.path.join("/data/hongzefu/dataset_generate", "videos", "ee-fk")
        os.makedirs(out_video_dir, exist_ok=True)

        for episode in range(50):

            
            # ---------- 为当前 episode 创建环境与数据集解析器 ----------
            env, seed, difficulty = config_resolver.make_env_for_episode(episode)
            env.save_failed_match_env_id = env_id
            env.save_failed_match_episode = episode
            robot, pin_model, ee_link_idx = _init_fk_context(env)
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
            n = int(reward_batch.numel()) if hasattr(reward_batch, "numel") else 0
            info = _last_info(info_batch, n)
            terminated = bool(terminated_batch[-1].item()) if n > 0 else False
            truncated = bool(truncated_batch[-1].item()) if n > 0 else False

            reset_captioned_path = os.path.join(out_video_dir, f"replay_ee_fk_{env_id}_ep{episode}_reset_captioned.mp4")
            # save_listStep_video 支持 base/wrist 左右拼接
            reset_obs_for_video = {
                "base_camera": base_camera,
                "wrist_camera": wrist_camera,
            } if (base_camera or wrist_camera) else {}
            # if save_listStep_video(reset_obs_for_video, reward_batch, terminated_batch, truncated_batch, info_batch, reset_captioned_path):
            #     print(f"Saved reset captioned video: {reset_captioned_path}")
            # else:
            #     print(f"WARNING: Reset video not saved (no frames or no subgoal_grounded): {reset_captioned_path}")


            # ---------- 按 step 回放：读取 joint action，经 FK 转 ee_pose(8维) 后执行 ----------
            episode_success = False
            step = 0
            replay_base_frames = []
            replay_wrist_frames = []
            replay_subgoal_grounded = []
            while True:
                joint_action = dataset_resolver.get_action(step)
                if joint_action is None:
                    break
                ee_action = _joint_to_ee_action_with_fk(
                    joint_action=joint_action,
                    env=env,
                    pin_model=pin_model,
                    ee_link_idx=ee_link_idx,
                    step=step,
                )
                obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.step(ee_action)

                # 从 batch 读取（供调试或后续逻辑）
                maniskill_obs = (obs_batch or {}).get("maniskill_obs", [])
                # 与 jointangle 回放一致：batch 中图像键为 "base_camera" / "wrist_camera"
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

            # ---------- 保存本 episode 回放视频（用本循环内收集的帧与字幕）并关闭资源 ----------
            success_prefix = "success" if episode_success else "fail"
            out_video_path = os.path.join(out_video_dir, f"{success_prefix}_replay_ee_fk_{env_id}_ep{episode}.mp4")
            if (replay_base_frames or replay_wrist_frames) and replay_subgoal_grounded:
                obs_video = {
                    "base_camera": replay_base_frames,
                    "wrist_camera": replay_wrist_frames,
                }
                info_video = {"subgoal_grounded": replay_subgoal_grounded}
                save_listStep_video(obs_video, reward_batch, terminated_batch, truncated_batch, info_video, out_video_path)
                print(f"Saved video: {out_video_path}")
            else:
                print(f"Skipped video (no frames or no subtitles): {out_video_path}")
            dataset_resolver.close()
            env.close()


if __name__ == "__main__":
    main()
