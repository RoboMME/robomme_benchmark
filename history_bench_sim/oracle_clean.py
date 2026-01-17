"""
核心理念：将“整个 Episode”视为一个原子操作
我们将逻辑分为两类错误处理：

1. 软错误（Soft Failures）：如 API 超时、网络波动。
   策略：原地等待并重试（由 API 类内部处理）。

2. 硬错误（Hard Failures）：如 step_before/after 报错、环境初始化失败、规划器错误、关键数据准备失败。
   策略：销毁环境，重新开始当前 Episode。

详细实施机制（Safe Eval）：
第一层：Episode 级“大循环”保护
在遍历 episode 的循环内部，建立一个“尝试-销毁-重建”的机制。
- 设置重试计数器（如 max_episode_retries = 3）。
- 开启 while 循环进行尝试。
- 进入 try 块（最大的保护伞）：
    - 阶段 A：初始化 (initialize_episode, API init)
    - 阶段 B：执行步骤循环 (step_before -> prepare_input -> call_api -> step_after)
    - 阶段 C：成功标记与保存
- 进入 except 块（捕获环境/仿真崩溃）：
    - 捕获所有 Exception。
    - 执行“清理战场”：env.close(), del env, del api, gc.collect()。
    - 计数器增加，准备下一次重建环境重试。
    - 若超过最大次数，跳过当前 Episode。
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


from history_bench_sim.oracle_logic_clean import step_before, step_after
from scripts.evaluate_oracle_planner_gui import EpisodeConfigResolverForOraclePlanner


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
    Mock model 返回固定的5个序列动作: pick, place, pick, place, press button
    
    Args:
        base_frames: 所有 base frames 列表（不处理）
        text_query: 文本查询（不使用）
        step_idx: 步骤索引（用于确定返回哪个动作）
        language_goal: 语言目标（不使用）
    
    Returns:
        command_dict: 根据 step_idx 返回对应的命令字典
    """
    # 定义固定的5个动作序列
    actions = [
        {"action": "pick up the cube", "point": [256, 256]},
        {"action": "put it into the bin", "point": [256, 256]},
        {"action": "press the button", "point": None}  # press button 动作，point 为 None
    ]
    
    # 根据 step_idx 返回对应的动作，如果超出范围则循环使用最后一个动作
    if step_idx < len(actions):
        return actions[step_idx]
    else:
        return actions[-1]  # 超出范围时返回最后一个动作

def main():    
    # Initialization Wrapper
    oracle_resolver = EpisodeConfigResolverForOraclePlanner(
        gui_render=False,
        max_steps_without_demonstration=1000
    )
    
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
        num_episodes = oracle_resolver.get_num_episodes(env_id)

        #for episode in range(num_episodes):
        for episode in range(2):
            # if episode !=1:
            #     continue
            
            model_name = "test"  # "gemini-2.5-pro" # "gpt-4o-mini", "gemini-er", "qwen-vl"， "local" 
            save_dir = os.path.join("/home/hongzefu", "oracle_planning_results", model_name, env_id, f"ep{episode}")
            
            # 检查episode状态（断点继续）- 只使用JSON状态判断
            has_status, recorded_status = check_episode_status(status_json_path, model_name, env_id, episode)
            if has_status:
                print(f"[断点继续] Episode {episode} 已有状态记录: {recorded_status}，跳过执行。")
                continue
            
            current_episode_try = 0
            max_episode_retries = 3  #一个episode最大重试次数
            
            # --- Episode 级重试循环 (处理仿真器崩溃/规划器错误) ---
            # 硬错误（Hard Failures）：如 step_before/after 报错、环境初始化失败。策略是 销毁环境，重新开始当前 Episode。
            while current_episode_try < max_episode_retries:
                try:
                    # 阶段 A：初始化
                    env, planner, color_map, language_goal = oracle_resolver.initialize_episode(env_id, episode)
                    success = "fail"
                                
                    # 如果目录存在但未完成，删除并重新创建
                    if os.path.exists(save_dir):
                        print(f"[断点继续] 检测到未完成的episode {episode}，删除旧文件重新开始")
                        shutil.rmtree(save_dir)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    
                    with open(os.path.join(save_dir, "language_goal.txt"), "w") as f:
                        f.write(str(language_goal) if language_goal is not None else "")

                    step_idx = 0
                    frame_idx = 0
                    max_query_times = 10  #对于binfill 10 不够
                    
                    # 阶段 B：执行步骤循环
                    while True:
                        if step_idx >= max_query_times:
                            print(f"Max query times ({max_query_times}) reached, stopping.")
                            break

                        # 步骤 1：执行 step_before
                        # 直接运行。如果这里报错，会直接跳到外层的 except，触发重启。
                        seg_vis, seg_raw, base_frames, wrist_frames, available_options = step_before(
                            env,
                            planner,
                            env_id,
                            color_map
                        )
                        print("num of base_frames", len(base_frames)-frame_idx)
                        print("num of wrist_frames", len(wrist_frames)-frame_idx)
                        print(available_options)
                        
                        # 检查是否有新的帧可用
                        if len(base_frames) <= frame_idx:
                            print(f"Warning: No new frames available at step {step_idx}. Exiting loop.")
                            raise Exception("No new frames available, triggering episode retry")
                        
                        # ------------------------ Call Mock Model ------------------------------------
                        # 使用 mock model 生成 command_dict，输入所有 base_frames 不进行处理
                        command_dict = mock_model(base_frames[frame_idx:], "", step_idx, language_goal)
                        
                        # TODO: will be fixed in the future
                        if command_dict['point'] is not None:
                            command_dict['point'] = command_dict['point'][::-1]  
                        
                        print(f"\nCommand: {command_dict}")
                        
                        frame_idx = len(base_frames)
                        step_idx += 1
                        
                        # 步骤 3：执行 step_after
                        # 直接运行。如果物理引擎在这一步崩溃，跳到外层 except，触发重启。               
                        evaluation = step_after(
                            env,
                            planner,
                            env_id,
                            seg_vis,
                            seg_raw,
                            base_frames,
                            wrist_frames,
                            command_dict
                        )
                        
                        if evaluation is None:
                            print("Evaluation is None, skipping this step")
                            break
                        
                        fail_flag = evaluation.get("fail", False)
                        success_flag = evaluation.get("success", False)
                        
                        # 步骤 4：判断结束
                        if _tensor_to_bool(fail_flag):
                            success = "fail"
                            print("Encountered failure condition; stopping task sequence.")
                            break

                        if _tensor_to_bool(success_flag):
                            success = "success"
                            print("Task completed successfully.")
                            break
                    
                    # 阶段 C：成功标记
                    # 如果代码能走到这里（没有报错），说明本 Episode 跑通了。
                    del env

                    # 记录状态（success或fail，api_error不记录）
                    if success in ["success", "fail"]:
                        save_episode_status(status_json_path, model_name, env_id, episode, success)
                    break # Success break out of retry loop

                except Exception as e:
                    # 进入 except 块（捕获环境/仿真崩溃）：
                    print(f"Episode {episode} crashed on try {current_episode_try + 1}. Error: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # 关键动作：清理战场。
                    env_local = locals().get('env')
                    if env_local is not None:
                        try:
                            env_local.close()
                        except:
                            pass
                        del env_local
                    
                    # 强制进行 Python 垃圾回收 gc.collect()（对仿真器很重要）。
                    gc.collect()
                    current_episode_try += 1
                    
                    # 判断是否放弃
                    if current_episode_try >= max_episode_retries:
                        print(f"Skipping episode {episode} due to persistent errors.")
                        # 记录sim_error状态
                        save_episode_status(status_json_path, model_name, env_id, episode, "sim_error")
                        break
                      
    oracle_resolver.close()
    
if __name__ == "__main__":
    main()
