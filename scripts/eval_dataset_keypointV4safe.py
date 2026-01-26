import os


import sys
import json
import h5py
import numpy as np
import sapien
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym
from gymnasium.utils.save_video import save_video

from historybench.env_record_wrapper import HistoryBenchRecordWrapper, EpisodeConfigResolver
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


def read_metadata(metadata_path):
    """
    从 metadata JSON 文件读取所有 episode 配置
    
    Args:
        metadata_path: metadata JSON 文件路径
        
    Returns:
        list: 包含所有 episode 记录的列表，每个记录包含 task、episode、seed、difficulty
    """
    if not Path(metadata_path).exists():
        print(f"Metadata file not found: {metadata_path}")
        return []
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        episode_records = metadata.get('records', [])
        return episode_records


def read_keypoints_from_h5(h5_file_path, env_id):
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
        env_group_name = f"env_{env_id}"
        if env_group_name not in f:
            print(f"Environment group '{env_group_name}' not found in HDF5 file")
            return keypoints_data
        
        env_group = f[env_group_name]
        
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

    env_id_list = [
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
'MoveCube',
"PatternLock",
"RouteStick"
        ]

    dataset_root = Path("/data/hongzefu/dataset_generate")

    dataset_root = Path("/nfs/turbo/coe-chaijy-unreplicated/hongzefu/dataset_generate")

    for env_id in env_id_list:
        dataset_path = dataset_root / f"record_dataset_{env_id}.h5"
        metadata_path = dataset_root / f"record_dataset_{env_id}_metadata.json"
        
        # 读取 metadata 中的所有 episode 配置
        episode_records = read_metadata(metadata_path)
        if not episode_records:
            print(f"No episode records found for {env_id}; skipping")
            continue
        
        # 读取 HDF5 文件中的所有 keypoint
        all_keypoints = read_keypoints_from_h5(dataset_path, env_id)
        if not all_keypoints:
            print(f"No keypoints found for {env_id}; skipping")
            continue
        
        print(f"Found {len(episode_records)} episodes and {len(all_keypoints)} keypoints for {env_id}")
        
        # 初始化 EpisodeConfigResolver
        resolver = EpisodeConfigResolver(
            env_id=env_id,
            dataset=None,
            metadata_path=metadata_path,
            render_mode="human",
            gui_render=True,
            max_steps_without_demonstration=200,
        )
        
        # 遍历所有 episode
        for episode_record in episode_records:

            # if episode_record['episode'] != 0:
            #     continue

            episode = episode_record['episode']
            seed = episode_record.get('seed')
            difficulty = episode_record.get('difficulty')
            
            print(f"--- Running simulation for episode:{episode}, env: {env_id}, seed: {seed}, difficulty: {difficulty} ---")
            
            # 使用 EpisodeConfigResolver 创建环境
            env, episode_dataset, resolved_seed, resolved_difficulty = resolver.make_env_for_episode(episode)
            env.reset()
        
            
            # 初始化 planner
            if env_id in ("PatternLock", "RouteStick"):
                planner = FailAwarePandaStickMotionPlanningSolver(
                    env,
                    debug=False,
                    vis=True,
                    base_pose=env.unwrapped.agent.robot.pose,
                    visualize_target_grasp_pose=False,
                    print_env_info=False,
                    joint_vel_limits=0.3,
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
                print(f"No keypoints found for episode {episode}; skipping")
                env.close()
                continue
            
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
                        planner.move_to_pose_with_RRTStar(pose)
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
