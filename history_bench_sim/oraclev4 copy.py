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


TASK_WITH_DEMO = [
    "VideoUnmask", "VideoUnmaskSwap", "VideoPlaceButton", "VideoPlaceOrder",
    "VideoRepick", "MoveCube", "InsertPeg", "PatternLock", "RouteStick"
]

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
        "PatternLock",
        "RouteStick"
    ]

    for env_id in env_id_list:
        num_episodes = oracle_resolver.get_num_episodes(env_id)

        for episode in range(num_episodes):
        #for episode in range(10):
            # if episode !=2:
            #     continue

            env, planner, color_map, language_goal = oracle_resolver.initialize_episode(env_id, episode)
            model_name = "local"  # "gemini-2.5-pro" # "gpt-4o-mini", "gemini-er", "qwen-vl"
            success = "fail"
            save_dir = os.path.join("history_bench_sim", "oracle_planning", model_name, env_id, f"ep{episode}")
                        
            if os.path.exists(save_dir):
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
            max_query_times = 20
            
            while True:
                if step_idx >= max_query_times:
                    print(f"Max query times ({max_query_times}) reached, stopping.")
                    break

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
                    print(f"Warning: No new frames available at step {step_idx}. Skipping API call.")
                    continue
                
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
                
                input_data = api.prepare_input_data(base_frames[frame_idx:], text_query, step_idx)

               #使用gui画出图
                # cv2.imshow("base_frames[-1]", base_frames[-1])
                # cv2.waitKey(0)
                # cv2.destroyWindow("base_frames[-1]")


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
                
                # ------------------------------------------------------------                
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
                if _tensor_to_bool(fail_flag):
                    success = "fail"
                    print("Encountered failure condition; stopping task sequence.")
                    break

                if _tensor_to_bool(success_flag):
                    success = "success"
                    print("Task completed successfully.")
                    break
            
            
            if response is not None:
                api.prepare_input_data(base_frames[frame_idx:], text_query, step_idx)
            else:
                success = "api_error"
            
            api.save_conversation()
            api.save_final_video(os.path.join(os.path.dirname(save_dir), f"{success}_ep{episode}_{language_goal}.mp4"))
            api.clear_uploaded_files() #only for gemini
            del api
            #import pdb; pdb.set_trace()
                      
    oracle_resolver.close()
    
if __name__ == "__main__":
    main()
