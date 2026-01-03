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
import cv2
import imageio
import json
import shutil
import torch
import gc
import glob


from history_bench_sim.chat_api.api import *
from history_bench_sim.chat_api.prompts import *
from history_bench_sim.oracle_logic import step_before, step_after
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


def process_patternlock_images(frames, env_id):
    """
    如果env_id是PatternLock，将图片旋转180度并在右上角画上坐标轴
    
    Args:
        frames: 图片列表（numpy数组列表）
        env_id: 环境ID
    
    Returns:
        处理后的图片列表
    """
    if env_id != "PatternLock":
        return frames
    
    processed_frames = []
    for frame in frames:
        # 复制图片以避免修改原始数据
        img = frame.copy()
        
        # 旋转180度
        img = cv2.rotate(img, cv2.ROTATE_180)
        
        # 获取图片尺寸
        h, w = img.shape[:2]
        
        # 在右上角画坐标轴
        # 设置坐标轴区域（右上角，留出一些边距）
        margin = 20
        axis_size = 80
        start_x = w - axis_size - margin
        start_y = margin
        
        # 绘制中心点
        center_x = start_x + axis_size // 2
        center_y = start_y + axis_size // 2
        cv2.circle(img, (center_x, center_y), 3, (0, 0, 0), -1)
        
        # 绘制箭头和标签
        arrow_length = 15  # 缩小箭头长度
        label_offset = 12  # 标签与箭头端点之间的距离
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4  # 缩小字体
        thickness = 1  # 加粗字体
        arrow_thickness = 3  # 加粗箭头
        
        # Forward (向上，在旋转后的坐标系中)
        cv2.arrowedLine(img, (center_x, center_y), (center_x, center_y - arrow_length), (0, 255, 0), arrow_thickness, tipLength=0.3)
        cv2.putText(img, "forward", (center_x - 25, center_y - arrow_length - label_offset), font, font_scale, (0, 255, 0), thickness)
        
        # Backward (向下)
        cv2.arrowedLine(img, (center_x, center_y), (center_x, center_y + arrow_length), (255, 0, 0), arrow_thickness, tipLength=0.3)
        cv2.putText(img, "backward", (center_x - 30, center_y + arrow_length + label_offset + 8), font, font_scale, (255, 0, 0), thickness)
        
        # Left (向左)
        cv2.arrowedLine(img, (center_x, center_y), (center_x - arrow_length, center_y), (0, 0, 255), arrow_thickness, tipLength=0.3)
        cv2.putText(img, "left", (center_x - arrow_length - label_offset - 20, center_y + 5), font, font_scale, (0, 0, 255), thickness)
        
        # Right (向右)
        cv2.arrowedLine(img, (center_x, center_y), (center_x + arrow_length, center_y), (255, 255, 0), arrow_thickness, tipLength=0.3)
        cv2.putText(img, "right", (center_x + arrow_length + label_offset, center_y + 5), font, font_scale, (255, 255, 0), thickness)
        
        processed_frames.append(img)
    
    return processed_frames


TASK_WITH_DEMO = [
    "VideoUnmask", "VideoUnmaskSwap", "VideoPlaceButton", "VideoPlaceOrder",
    "VideoRepick", "MoveCube", "InsertPeg", "PatternLock", "RouteStick"
]

def check_episode_completed(save_dir, episode, language_goal):
    """
    检查episode是否已完成
    返回: (is_completed, reason)
    """
    # 检查保存目录是否存在
    if not os.path.exists(save_dir):
        return False, "save_dir不存在"
    
    # 检查conversation.json是否存在
    conversation_file = os.path.join(save_dir, "conversation.json")
    if not os.path.exists(conversation_file):
        return False, "conversation.json不存在"
    
    # 检查父目录中是否有对应的视频文件
    parent_dir = os.path.dirname(save_dir)
    video_pattern = f"*_ep{episode}_*.mp4"
    matching_videos = glob.glob(os.path.join(parent_dir, video_pattern))
    
    if not matching_videos:
        return False, "视频文件不存在"
    
    # 检查视频文件是否完整（文件大小大于0）
    for video_path in matching_videos:
        if os.path.getsize(video_path) == 0:
            return False, "视频文件为空"
    
    return True, "所有文件都存在且完整"

def main():    
    # Initialization Wrapper
    oracle_resolver = EpisodeConfigResolverForOraclePlanner(
        gui_render=False,
        max_steps_without_demonstration=1000
    )
    
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
        #"PatternLock",
        "RouteStick"
    ]

    for env_id in env_id_list:
        num_episodes = oracle_resolver.get_num_episodes(env_id)

        #for episode in range(num_episodes):
        for episode in range(10):
            # if episode !=2:
            #     continue
            
            model_name = "gemini-2.5-pro"  # "gemini-2.5-pro" # "gpt-4o-mini", "gemini-er", "qwen-vl"， "local" 
            save_dir = os.path.join("/home/hongzefu", "oracle_planning_results", model_name, env_id, f"ep{episode}")
            
            # 获取language_goal用于检查（需要先初始化才能获取，但我们可以先检查保存的文件）
            # 先尝试读取已保存的language_goal.txt
            language_goal_for_check = None
            language_goal_file = os.path.join(save_dir, "language_goal.txt")
            if os.path.exists(language_goal_file):
                with open(language_goal_file, "r") as f:
                    language_goal_for_check = f.read().strip()
            
            # 检查episode是否已完成
            is_completed, reason = check_episode_completed(save_dir, episode, language_goal_for_check)
            if is_completed:
                print(f"[断点继续] Episode {episode} 已完成，跳过执行。原因: {reason}")
                continue
            
            current_episode_try = 0
            max_episode_retries = 3
            
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
                        f.write(language_goal)
                    
                    if "gemini" in model_name:
                        api = GeminiModel(save_dir=save_dir, task_id=env_id, model_name=model_name, task_goal=language_goal, subgoal_type="oracle_planner")
                    elif "qwen" in model_name:
                        api = QwenModel(save_dir=save_dir, task_id=env_id, model_name=model_name, task_goal=language_goal, subgoal_type="oracle_planner")
                    elif "local" in model_name:
                        api = LocalModel(save_dir=save_dir, task_id=env_id, model_name=model_name, task_goal=language_goal, subgoal_type="oracle_planner")
                    else:
                        api = OpenAIModel(save_dir=save_dir, task_id=env_id, model_name=model_name, task_goal=language_goal, subgoal_type="oracle_planner")


                    step_idx = 0
                    frame_idx = 0
                    max_query_times = 10
                    
                    response = None
                    text_query = ""
                    
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
                        
                        # ------------------------ Call Gemini API ------------------------------------
                    
                        if step_idx == 0:
                            if env_id in TASK_WITH_DEMO:
                                if api.use_multi_images_as_video:
                                    text_query = DEMO_TEXT_QUERY_multi_image.format(task_goal=language_goal)
                                else:
                                    text_query = DEMO_TEXT_QUERY.format(task_goal=language_goal)
                            else:
                                text_query = IMAGE_TEXT_QUERY.format(task_goal=language_goal)
                        else:
                            if api.use_multi_images_as_video:
                                text_query = VIDEO_TEXT_QUERY_multi_image.format(task_goal=language_goal)
                            else:
                                text_query = VIDEO_TEXT_QUERY.format(task_goal=language_goal)
                        
                        # 数据准备
                        # 如果env_id是PatternLock，先处理图片（旋转180度并添加坐标轴）
                        # processed_frames = process_patternlock_images(base_frames[frame_idx:], env_id)
                        # 如果它因为任何原因（如帧数据为空、IO错误）失败抛出异常，也会触发外层的 except 块，导致环境销毁和重建。
                        input_data = api.prepare_input_data(base_frames[frame_idx:], text_query, step_idx)

                    #使用gui画出图
                        # cv2.imshow("base_frames[-1]", base_frames[-1])
                        # cv2.waitKey(0)
                        # cv2.destroyWindow("base_frames[-1]")


                        # 步骤 2：API 调用（保持原有的原地重试逻辑）
                        # 这里保留之前的“API 内部重试循环”，因为网络错误不需要重启环境。
                        response, points = api.call(input_data)
                        
                        #points=[(255, 255)]#test

                        if response is None:
                            print("Response is None, skipping this step")
                            break
                        
                        # Draw the points for debugging              
                        if points and len(points) > 0:
                            anno_image = base_frames[-1].copy()
                            for point in points:
                                cv2.circle(anno_image, (point[1], point[0]), 5, (255, 255, 0), -1)
                            imageio.imwrite(os.path.join(save_dir, f"anno_step_{step_idx}_image.png"), anno_image)
                            api.add_frame_hold(anno_image)
                        
                        command_dict = response['subgoal']
                        # TODO: will be fixed in the future
                        if command_dict['point'] is not None:
                            command_dict['point'] = command_dict['point'][::-1]  
                        
                        print(f"\nResponse: {response}")              
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
                    if response is not None:
                        # 如果env_id是PatternLock，先处理图片（旋转180度并添加坐标轴）
                        # processed_frames = process_patternlock_images(base_frames[frame_idx:], env_id)
                        api.prepare_input_data(base_frames[frame_idx:], text_query, step_idx)
                    else:
                        success = "api_error"
                    
                    api.save_conversation()
                    api.save_final_video(os.path.join(os.path.dirname(save_dir), f"{success}_ep{episode}_{language_goal}.mp4"))
                    api.clear_uploaded_files() #only for gemini
                    del api
                    del env
                    #import pdb; pdb.set_trace()
                    break # Success break out of retry loop

                except Exception as e:
                    # 进入 except 块（捕获环境/仿真崩溃）：
                    print(f"Episode {episode} crashed on try {current_episode_try + 1}. Error: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # 关键动作：清理战场。
                    if 'env' in locals() and env is not None:
                        try:
                            env.close()
                        except:
                            pass
                    
                    if 'api' in locals() and api is not None:
                         # api cleanup if needed
                         pass
                    
                    if 'env' in locals(): del env
                    if 'api' in locals(): del api
                    
                    # 强制进行 Python 垃圾回收 gc.collect()（对仿真器很重要）。
                    gc.collect()
                    current_episode_try += 1
                    
                    # 判断是否放弃
                    if current_episode_try >= max_episode_retries:
                        print(f"Skipping episode {episode} due to persistent errors.")
                        break
                      
    oracle_resolver.close()
    
if __name__ == "__main__":
    main()
