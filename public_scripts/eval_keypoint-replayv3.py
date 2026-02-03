import os


import sys
import json
import h5py
import numpy as np
import sapien
from pathlib import Path

# Add parent directory and scripts to Python path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "scripts"))
import gymnasium as gym
from gymnasium.utils.save_video import save_video

from historybench.env_record_wrapper import (
    HistoryBenchRecordWrapper,
    EpisodeConfigResolver,
    RRTPlanFailure,
)
from historybench.HistoryBench_env import *


from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)

# from util import *
import torch

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
# "PickXtimes",
# "StopCube",
# "SwingXtimes",
# "BinFill",

# "VideoUnmaskSwap",
# "VideoUnmask",
# "ButtonUnmaskSwap",
# "ButtonUnmask",

#"VideoRepick",
#  "VideoPlaceButton",
 "VideoPlaceOrder",
# "PickHighlight",

# "InsertPeg",
# 'MoveCube',
# "PatternLock",
# "RouteStick"
        ]

    dataset_root = Path("/data/hongzefu/dataset_generate")

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
        
        # 初始化 EpisodeConfigResolver（action_space=keypoint 时内部会包好 MultiStepDemonstrationWrapper）
        resolver = EpisodeConfigResolver(
            env_id=env_id,
            metadata_path=metadata_path,
            render_mode="human",
            gui_render=True,
            max_steps_without_demonstration=200,
            action_space="keypoint",
        )
        
        # 遍历所有 episode
        for episode_record in episode_records:


            episode = episode_record['episode']
            seed = episode_record.get('seed')
            difficulty = episode_record.get('difficulty')
            
            print(f"--- Running simulation for episode:{episode}, env: {env_id}, seed: {seed}, difficulty: {difficulty} ---")
            
            # 使用 EpisodeConfigResolver 创建环境（已含 MultiStepDemonstrationWrapper）
            env, resolved_seed, resolved_difficulty = resolver.make_env_for_episode(episode)
            env.reset()

            # 获取当前 episode 的所有 keypoint（按 timestep 排序）
            episode_keypoints = [kp for kp in all_keypoints if kp['episode'] == episode]
            episode_keypoints.sort(key=lambda x: x['timestep'])
            
            if not episode_keypoints:
                print(f"No keypoints found for episode {episode}; skipping")
                env.close()
                continue
            
            gui_render = True
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
                print(f"keypoint_p: {keypoint_p}")

                gripper_action = kp['action'][-1]
                action = np.concatenate([
                    keypoint_p,
                    keypoint_q,
                    [float(gripper_action)],
                ]).astype(np.float32)

                try:
                    obs, reward, terminated, truncated, info = env.step(action)
                    # 从 obs 读取
                    image = obs.get('frames', [])[-1] if obs.get('frames') else None
                    wrist_image = obs.get('wrist_frames', [])[-1] if obs.get('wrist_frames') else None
                    last_action = obs.get('actions', [])[-1] if obs.get('actions') else None
                    state = obs.get('states', [])[-1] if obs.get('states') else None
                    velocity = obs.get('velocity', [])[-1] if obs.get('velocity') else None
                    language_goal = obs.get('language_goal') if obs else None
                    # 从 info 读取
                    subgoal = info.get('subgoal_history', []) if info else []
                    subgoal_grounded = info.get('subgoal_grounded_history', []) if info else []

                    # 从 obs 读取
                    frames = obs.get('frames', []) if obs else []
                    wrist_frames = obs.get('wrist_frames', []) if obs else []
                    actions = obs.get('actions', []) if obs else []
                    states = obs.get('states', []) if obs else []
                    velocity = obs.get('velocity', []) if obs else []
                    language_goal = obs.get('language_goal') if obs else None
                    # 从 info 读取
                    subgoal = info.get('subgoal_history', []) if info else []
                    subgoal_grounded = info.get('subgoal_grounded_history', []) if info else []

                    if gui_render:
                        env.render()
                    if truncated:
                        print(f"[{env_id}] episode {episode} 步数超限。")
                        break
                    if terminated.any():
                        if info.get("success") == torch.tensor([True]) or (
                            isinstance(info.get("success"), torch.Tensor) and info.get("success").item()
                        ):
                            print(f"[{env_id}] episode {episode} 成功。")
                        elif info.get("fail", False):
                            print(f"[{env_id}] episode {episode} 失败。")
                        break
          
                except RRTPlanFailure as exc:
                    print(f"    RRT plan failure for keypoint timestep {timestep}: {exc}")
                    break

            


            # 保存回放视频（frames + subgoal_grounded）
            video_dir = dataset_root / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            out_video_path = video_dir / f"replay_kp_{env_id}_ep{episode}.mp4"
            env.save_video(str(out_video_path))
            print(f"Saved video: {out_video_path}")

            # 执行完成后进行评估
            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
            print(f"Final evaluation for episode {episode}: {evaluation}")
            

            env.close()
            print(f"--- Finished Running simulation for episode:{episode}, env: {env_id} ---")


if __name__ == "__main__":
    main()
