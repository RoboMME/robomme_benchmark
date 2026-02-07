import os


import sys
import h5py
import numpy as np
import sapien
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym
from gymnasium.utils.save_video import save_video

from historybench.env_record_wrapper import DemonstrationWrapper
from historybench.HistoryBench_env import *


from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)

# from util import *
import torch

from planner_fail_safe import (
    FailAwarePandaArmMotionPlanningSolver,
    FailAwarePandaStickMotionPlanningSolver,
    ScrewPlanFailure,
)

OUTPUT_ROOT = Path(__file__).resolve().parents[1]
TARGET_SEED = 10010300
TARGET_DIFFICULTY = "hard"


def resolve_dataset_h5_path(env_id, seed, difficulty, episode):
    """
    兼容两种数据落盘格式：
    1) 单文件: record_dataset_{env}_seed{seed}_{difficulty}.h5
    2) 分片文件: record_dataset_{env}_seed{seed}_{difficulty}_hdf5_files/{env}_ep{episode}_seed{seed}.h5
    """
    base_file = Path(
        f"/data/hongzefu/dataset_generate/record_dataset_{env_id}_seed{seed}_{difficulty}.h5"
    )
    if base_file.exists():
        return base_file

    split_file = (
        base_file.parent
        / f"{base_file.stem}_hdf5_files"
        / f"{env_id}_ep{episode}_seed{seed}.h5"
    )
    if split_file.exists():
        return split_file

    return base_file


def read_keypoints_from_h5(h5_file_path, env_id=None):
    """
    从 HDF5 文件读取指定 env_id 的所有 keypoint
    
    Args:
        h5_file_path: HDF5 文件路径
        env_id: 环境 ID
        
    Returns:
        list: 包含所有 keypoint 的列表，每个 keypoint 包含 episode、timestep、position_p、quaternion_q、solve_function、keypoint_type
    """
    keypoints_data = []
    
    if not Path(h5_file_path).exists():
        print(f"HDF5 file not found: {h5_file_path}")
        return keypoints_data
    
    with h5py.File(h5_file_path, 'r') as f:
        env_group = None
        if env_id is not None:
            env_group_name = f"env_{env_id}"
            if env_group_name in f:
                env_group = f[env_group_name]
            else:
                print(f"Environment group '{env_group_name}' not found; fallback to first group in HDF5")
        if env_group is None:
            root_keys = list(f.keys())
            if not root_keys:
                print(f"No groups found in HDF5 file: {h5_file_path}")
                return keypoints_data
            env_group = f[root_keys[0]]
            print(f"Using env group from file: {root_keys[0]}")
        
        # 遍历所有episode
        for episode_name in env_group.keys():
            if episode_name == 'setup':
                continue  # 跳过setup group
                
            episode_group = env_group[episode_name]
            episode_num = int(episode_name.replace('episode_', ''))
            
            # 遍历所有timestep
            for timestep_name in episode_group.keys():
                if timestep_name == 'setup':
                    continue  # 跳过setup group
                
                timestep_group = episode_group[timestep_name]
                timestep_num = int(timestep_name.replace('record_timestep_', ''))
                
                # 检查是否有keypoint
                if 'keypoint_p' in timestep_group:
                    position_p = timestep_group['keypoint_p'][()].tolist()
                    quaternion_q = timestep_group['keypoint_q'][()].tolist()
                    
                    solve_function = timestep_group['keypoint_solve_function'][()]
                    if isinstance(solve_function, bytes):
                        solve_function = solve_function.decode('utf-8')
                    else:
                        solve_function = str(solve_function)
                    
                    keypoint_type = timestep_group['keypoint_type'][()]
                    if isinstance(keypoint_type, bytes):
                        keypoint_type = keypoint_type.decode('utf-8')
                    else:
                        keypoint_type = str(keypoint_type)
                    
                    action = timestep_group['action'][()].tolist()
                    demonstration = timestep_group['demonstration'][()]
                    
                    keypoint_info = {
                        'episode': episode_num,
                        'timestep': timestep_num,
                        'position_p': position_p,
                        'quaternion_q': quaternion_q,
                        'solve_function': solve_function,
                        'keypoint_type': keypoint_type,
                        'action': action,
                        'demonstration': demonstration,
                    }
                    keypoints_data.append(keypoint_info)
    
    # 按 episode 和 timestep 排序
    keypoints_data.sort(key=lambda x: (x['episode'], x['timestep']))
    return keypoints_data


def main():
    """
    Main function to run the simulation using keypoints from dataset.
    """

    env_id = "ButtonUnmask"
    episode = 0
    seed = TARGET_SEED
    difficulty = TARGET_DIFFICULTY
    dataset_path = resolve_dataset_h5_path(env_id, seed, difficulty, episode)
    print(f"Using dataset path: {dataset_path}")
    # 读取 HDF5 文件中的所有 keypoint
    all_keypoints = read_keypoints_from_h5(dataset_path, env_id)
    if not all_keypoints:
        print(f"No keypoints found for {env_id}; exiting")
        return

    print(f"Found {len(all_keypoints)} keypoints for {env_id}")
    print(f"--- Running simulation for episode:{episode}, env: {env_id}, seed: {seed}, difficulty: {difficulty} ---")
    
    env_kwargs = dict(
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_pos",
        render_mode="human",
        reward_mode="dense",
        max_episode_steps=99999,
        HistoryBench_seed=seed,
        HistoryBench_difficulty=difficulty,
    )
    env = gym.make(env_id, **env_kwargs)
    env = DemonstrationWrapper(
        env,
        max_steps_without_demonstration=200,
        gui_render=True,
    )
    env.reset(seed=seed)
        
        # 初始化 planner
    if env_id in ("PatternLock", "RouteStick"):
        planner = FailAwarePandaStickMotionPlanningSolver(
            env,
            debug=False,
            vis=True,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            
        )
    else:
        planner = FailAwarePandaArmMotionPlanningSolver(
            env,
            debug=False,
            vis=True,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=True,
            print_env_info=False,
        )
        
        # 获取当前 episode 的所有 keypoint（按 timestep 排序）
    episode_keypoints = [kp for kp in all_keypoints if kp['episode'] == episode]
    episode_keypoints.sort(key=lambda x: x['timestep'])
    
    if not episode_keypoints:
        print(f"No keypoints found for episode {episode}; exiting")
        env.close()
        return
    
    print(f"Executing {len(episode_keypoints)} keypoints for episode {episode}")
        
        
    print("Task list:")
    for i, task in enumerate(env.unwrapped.task_list):
        task_name = task.get("name", "Unknown")
        print(f"  Task {i+1}: {task_name}")
        
        # 执行所有 keypoint
    for idx, kp in enumerate(episode_keypoints):
        if kp['demonstration']:
            continue

        keypoint_p = np.array(kp['position_p'], dtype=np.float32)
        keypoint_q = np.array(kp['quaternion_q'], dtype=np.float32)
        keypoint_type = kp['keypoint_type']
        timestep = kp['timestep']
            
        print(f"  Executing keypoint {idx+1}/{len(episode_keypoints)}: timestep={timestep}, type={keypoint_type}")
            
            # 根据 keypoint_type 决定夹爪操作
        gripper_action = kp['action'][-1]
            
            # 移动到 keypoint pose
        try:
                pose = sapien.Pose(p=keypoint_p, q=keypoint_q)
                print (f"keypoint_p: {keypoint_p}")

                # 获取当前末端位置
                current_pose = env.unwrapped.agent.tcp.pose
                current_p = current_pose.p
                if hasattr(current_p, 'cpu'):
                    current_p = current_p.cpu().numpy()
                if current_p.ndim > 1:
                    current_p = current_p.flatten()
                
                # 计算距离偏差
                dist = np.linalg.norm(current_p - keypoint_p)
                
                # 如果距离极小，直接获取当前观测并跳过移动
                if dist < 0.001:  # 阈值可调整，例如 5mm
                    print(f"Target too close (dist={dist:.6f}), skipping planner motion.")
                else:
                    #planner.move_to_pose_with_RRTStar(pose)
                    planner.move_to_pose_with_screw(pose)

                # 对比 keypoint 目标位姿与 gripper 实际到达位姿（TCP）
                reached_pose = env.unwrapped.agent.tcp.pose
                reached_p = reached_pose.p
                if hasattr(reached_p, 'cpu'):
                    reached_p = reached_p.cpu().numpy()
                reached_p = np.asarray(reached_p, dtype=np.float32).reshape(-1)[:3]

                reached_q = reached_pose.q
                if hasattr(reached_q, 'cpu'):
                    reached_q = reached_q.cpu().numpy()
                reached_q = np.asarray(reached_q, dtype=np.float32).reshape(-1)[:4]

                pos_delta = reached_p - keypoint_p
                pos_err = np.linalg.norm(pos_delta)

                kp_q_n = keypoint_q / (np.linalg.norm(keypoint_q) + 1e-12)
                reached_q_n = reached_q / (np.linalg.norm(reached_q) + 1e-12)
                quat_dot = np.clip(np.abs(np.dot(kp_q_n, reached_q_n)), -1.0, 1.0)
                ori_err_deg = np.degrees(2.0 * np.arccos(quat_dot))

                print(f"    Target keypoint p: {keypoint_p}")
                print(f"    Reached  tcp    p: {reached_p}")
                print(f"    Position delta  : {pos_delta} (L2={pos_err:.6f} m, {pos_err * 1000.0:.2f} mm)")
                print(f"    Orientation err : {ori_err_deg:.3f} deg")
                  
                #import pdb; pdb.set_trace()
                # try:
                #     planner.move_to_pose_with_screw(pose)
                # except Exception as exc:
                    
                #     break


                # 如果是 grasp_pose，移动到位置后关闭夹爪
                if gripper_action == -1:
                    try:
                        planner.close_gripper()
                    except Exception as exc:
                        print("no gripper")
                    
                elif gripper_action == 1:
                    try:
                        planner.open_gripper()
                    except Exception as exc:
                        print("no gripper")
                
                # 更新观测数据
                env.render()
                    
        except ScrewPlanFailure as exc:
            print(f"    Screw plan failure for keypoint timestep {timestep}: {exc}")
            break
        except Exception as exc:
            print(f"    Error executing keypoint timestep {timestep}: {exc}")
            break
    
    # 执行完成后进行评估
    evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
    print(f"Final evaluation for episode {episode}: {evaluation}")
    
    env.close()
    print(f"--- Finished Running simulation for episode:{episode}, env: {env_id} ---")


if __name__ == "__main__":
    main()
