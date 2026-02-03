"""
核心理念：将“整个 Episode”视为一个原子操作。仅保留与环境交互部分，无 query model。

阶段 A 初始化 -> 阶段 B 步骤循环 (step_before -> command_dict -> step_after) -> 阶段 C 保存状态。
若发生异常：env.close(), del env, gc.collect()，记录 sim_error 后继续下一 episode；无 episode 重试。
"""
import os
import sys

# 添加项目根目录到 Python 路径，以便正确导入模块
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import json
import shutil
import torch
import gc

from historybench.env_record_wrapper import EpisodeConfigResolver
from pathlib import Path


# =============================================================================
# 辅助函数
# =============================================================================

def _tensor_to_bool(value):
    """将 tensor 或其他类型转换为布尔值"""
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)


def load_episode_status(json_path):
    """
    读取JSON状态文件
    返回: 状态字典，如果文件不存在则返回空字典
    """
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load episode status from {json_path}: {e}")
        return {}

def save_episode_status(json_path, model_name, env_id, episode, status):
    """
    保存episode状态到JSON文件
    使用原子写入（先写临时文件再重命名）确保数据一致性
    
    Args:
        json_path: JSON文件路径
        model_name: 模型名称
        env_id: 环境ID
        episode: episode编号
        status: 状态 ("success", "fail", "sim_error")
    """
    # 生成episode key
    episode_key = f"{model_name}/{env_id}/ep{episode}"
    
    # 加载现有状态
    status_dict = load_episode_status(json_path)
    
    # 更新状态
    status_dict[episode_key] = status
    
    # 确保目录存在
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # 原子写入：先写临时文件，再重命名
    temp_path = json_path + ".tmp"
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(status_dict, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, json_path)
    except Exception as e:
        print(f"Warning: Failed to save episode status to {json_path}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def check_episode_status(json_path, model_name, env_id, episode):
    """
    检查episode是否已有记录状态
    
    Args:
        json_path: JSON文件路径
        model_name: 模型名称
        env_id: 环境ID
        episode: episode编号
    
    Returns:
        (has_status, status): (是否有记录, 状态值) 如果没有记录则返回 (False, None)
    """
    episode_key = f"{model_name}/{env_id}/ep{episode}"
    status_dict = load_episode_status(json_path)
    
    if episode_key in status_dict:
        status = status_dict[episode_key]
        # 只返回success、fail、sim_error，忽略api_error
        if status in ["success", "fail", "sim_error"]:
            return True, status
    
    return False, None


def mock_model(base_frames, text_query, step_idx, language_goal):
    """
    Mock model: returns a fixed command_dict per step for env-only testing.
    No API or prompts involved.

    Args:
        base_frames: list of base frames (unused)
        text_query: unused
        step_idx: step index (determines which action to return)
        language_goal: unused

    Returns:
        command_dict: dict with 'action' and 'point' for step_after
    """
    actions = [
        {"action": "pick up the cube", "point": [256, 256]},
        {"action": "put it into the bin", "point": [256, 256]},
        {"action": "press the button", "point": None},
    ]
    if step_idx < len(actions):
        return actions[step_idx]
    return actions[-1]


def get_num_episodes(dataset_root, env_id):
    metadata_path = dataset_root / f"record_dataset_{env_id}_metadata.json"
    if not metadata_path.exists():
        print(f"[{env_id}] Metadata {metadata_path} not found.")
        return 0
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
            return payload.get("record_count", 0)
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return 0

def main():    
    # Dataset Root
    dataset_root = Path("/data/hongzefu/robomme-v0.1b7/dataset_json")
    
    env_id_list = [
        # "PickXtimes",
        # "StopCube",
        # "SwingXtimes",
        "BinFill",

    #     "VideoUnmaskSwap",
    #     "VideoUnmask",
    #     "ButtonUnmaskSwap",
    #     "ButtonUnmask",

    #     "VideoRepick",
    #     "VideoPlaceButton",
    #      "VideoPlaceOrder",
    #     "PickHighlight",

    #     "InsertPeg",
    #     'MoveCube',
    #     "PatternLock",
    #     "RouteStick"
    ]

    # 定义JSON状态文件路径
    status_json_path = os.path.join("/home/hongzefu", "oracle_planning_results", "episode_status.json")
    
    for env_id in env_id_list:
        num_episodes = get_num_episodes(dataset_root, env_id)

        # Initialize Resolver per Env
        metadata_path = dataset_root / f"record_dataset_{env_id}_metadata.json"
        resolver = EpisodeConfigResolver(
            env_id=env_id,
            metadata_path=metadata_path,
            render_mode="human",
            gui_render=True,
            max_steps_without_demonstration=1000,
            action_space="oracle_planner"
        )

        #for episode in range(num_episodes):
        for episode in range(2):
            # if episode !=1:
            #     continue
            
            model_name = "env_only"
            save_dir = os.path.join("/data/hongzefu", "oracle_planning_results", model_name, env_id, f"ep{episode}")
            
            # 使用 EpisodeConfigResolver
            env, _, _ = resolver.make_env_for_episode(episode)

            try:
                # 阶段 A：初始化 (reset)
                obs, info = env.reset()
                language_goal = obs["language_goal"]
                
                success = "fail"

                os.makedirs(save_dir, exist_ok=True)

                with open(os.path.join(save_dir, "language_goal.txt"), "w") as f:
                    f.write(str(language_goal) if language_goal is not None else "")

                step_idx = 0
                frame_idx = 0
                max_query_times = 10

                # 阶段 B：执行步骤循环
                while True:
                    if step_idx >= max_query_times:
                        print(f"Max query times ({max_query_times}) reached, stopping.")
                        break

                    # 从 obs 获取数据
                    base_frames = obs["base_frames"]
                    available_options = obs["available_options"]
                    
                    print("num of base_frames", len(base_frames) - frame_idx)
                    print("num of wrist_frames", len(obs["wrist_frames"]) - frame_idx)
                    print(available_options)

                    if len(base_frames) <= frame_idx:
                        print(f"Warning: No new frames available at step {step_idx}. Exiting loop.")
                        break

                    command_dict = mock_model(base_frames[frame_idx:], "", step_idx, language_goal)
                    if command_dict["point"] is not None:
                        command_dict["point"] = command_dict["point"][::-1]

                    print(f"\nCommand: {command_dict}")

                    frame_idx = len(base_frames)
                    step_idx += 1

                    # 步骤 2：执行 step (相当于 step_after + step_before)
                    obs, reward, terminated, truncated, info = env.step(command_dict)

                    if terminated:
                        fail_flag = info.get("fail", False)
                        success_flag = info.get("success", False)

                        if _tensor_to_bool(fail_flag):
                            success = "fail"
                            print("Encountered failure condition; stopping task sequence.")
                        
                        if _tensor_to_bool(success_flag):
                            success = "success"
                            print("Task completed successfully.")
                        break

                # 阶段 C：成功标记与保存
                env.close()
                if success in ["success", "fail"]:
                    save_episode_status(status_json_path, model_name, env_id, episode, success)

            except Exception as e:
                print(f"Episode {episode} crashed. Error: {e}")
                import traceback
                traceback.print_exc()

                try:
                    env.close()
                except Exception:
                    pass
                
                gc.collect()
                save_episode_status(status_json_path, model_name, env_id, episode, "sim_error")
                      
    # oracle_resolver.close() # No longer needed as we create per loop
    
if __name__ == "__main__":
    main()
