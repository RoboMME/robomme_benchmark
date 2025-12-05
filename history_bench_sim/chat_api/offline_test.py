import datetime
import time
import os
import json
import shutil
import numpy as np
from moviepy import VideoFileClip
from copy import deepcopy
import re
from history_bench_sim.chat_api.prompts import *
import cv2

from history_bench_sim.chat_api.api import GeminiModel, OpenAIModel
from history_bench_sim.chat_api.utils import *



if __name__ == "__main__":
    import imageio
    video_path = "chat_api/videos"
    
    USE_MULTI_IMAGES_AS_VIDEO = False
    SUBGOAL_TYPE = "oracle_planner" # simple_subgoal, grounded_subgoal, oracle_planner
    if USE_MULTI_IMAGES_AS_VIDEO:
        VIDEO_TEXT_QUERY = VIDEO_TEXT_QUERY_multi_image

    model_name = "gpt-4o"
    task_id = "VideoUnmaskSwap"
    
    video = read_video_moviepy(f"{video_path}/{task_id}.mp4", use_concatenated_image=False)
        
    print(f"video length: {len(video)}", f"half length: {len(video)/2}")
    
    save_dir = f"examples/history_bench_sim/chat_api/runs/{model_name}/{task_id}/{SUBGOAL_TYPE}"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    total_len = len(video)
    step_idx = 0
    
    task_goal_dict = {
        'RouteStick': "watch the video carefully, then move the closed gripper to navigate around the sticks, following the same path as demonstrated in the past",
        "PatternLock": "watch the video carefully, then move the closed gripper to trace the same path as before",
        "VideoPlaceButton": "watch the video carefully, then place the blue cube on the target right before the button was pressed",
        "VideoPlaceOrder": "watch the video carefully, then place the blue cube on the third target it was previously placed on",
        "PickXtimes": "pick up the green cube and place it on the target, repeating this action three times, then press the button to stop",
        "SwingXtimes": "pick up the green cube and move it to the top of the right-side target, then move it to the top of the left-side target, repeating this back and forth motion three times, then press the button to stop",
        "StopCube": "press the button to stop the cube just as it reaches the target for the third time.",
        "MoveCube": "watch the video carefully, then move the cube to the target in the same manner as before",
        "ButtonUnmask": "first press the button, then pick up the container hiding the blue cube",
        "VideoUnmask": "watch the video carefully, then pick up the container hiding the red cube",
        "VideoRepick": "watch the video carefully, then repeatedly pick up and put down the same cube that has been picked up before for three times, finally press the button to finish",
        "InsertPeg": "watch the video carefully, then insert the same peg into the hole in the same way as you did before",
        "PickHighlight": "first press the button, then pick up all highlighted cubes, finally press the button again to stop",
        "VideoUnmaskSwap": "watch the video carefully, then pick up the container hiding the blue cube, finally pick up another container hiding the green cube",
        "ButtonUnmaskSwap": "first press both buttons on the table, then pick up the container hiding the blue cube, finally pick up another container hiding the green cube",
        "BinFill": "put one blue cube into the bin and press the button to stop",   
    }
    exec_idx_dict = {
        "BinFill": 0,
        "PickXtimes": 0,
        "StopCube": 0,
        "SwingXtimes": 0,
        
        "PickHighlight": 0,
        "VideoRepick": 280,
        "VideoPlaceButton": 800,
        "VideoPlaceOrder": 850,
        
        "VideoUnmask": 81,
        "ButtonUnmask": 0,
        "ButtonUnmaskSwap": 0,
        "VideoUnmaskSwap": 219,
        
        "MoveCube": 160,
        "InsertPeg": 325,
        "PatternLock": 197,
        "RouteStick": 162,
    }
    step_size_dict = {
        "BinFill": 48,
        "SwingXtimes": 48,
        "RouteStick": 32,
        "PickHighlight": 48,
        "PickXtimes": 48,
        "InsertPeg": 48,
        "StopCube": 16,
        "PatternLock": 32,
        "ButtonUnmask": 64,
        "ButtonUnmaskSwap": 64,
        "VideoUnmask": 64,
        "VideoUnmaskSwap": 64,
        "VideoPlaceButton": 48,
        "VideoPlaceOrder": 48,
        "VideoRepick": 48,
        "MoveCube": 64,
    }
    
    exec_idx = exec_idx_dict[task_id]
    task_goal = task_goal_dict[task_id]
    step_size = step_size_dict[task_id]
    
    
    if "gemini" in model_name:
        api = GeminiModel(save_dir=save_dir, task_id=task_id, model_name=model_name, task_goal=task_goal, subgoal_type=SUBGOAL_TYPE,
                        use_multi_images_as_video=USE_MULTI_IMAGES_AS_VIDEO)
    else:
        api = OpenAIModel(save_dir=save_dir, task_id=task_id, model_name=model_name, task_goal=task_goal, subgoal_type=SUBGOAL_TYPE,
                        use_multi_images_as_video=USE_MULTI_IMAGES_AS_VIDEO)
        
    
    if exec_idx > 0: # has demo
        video_clip = video[:exec_idx+1] # remember to include the first execution frame
        input_data = api.prepare_input_data(video_clip, DEMO_TEXT_QUERY.format(task_goal=task_goal), step_idx)
                    
        response, points = api.call(input_data)
        step_idx += 1
    
    else: # no demo
        image = video[0]  
        input_data = api.prepare_input_data(image, IMAGE_TEXT_QUERY.format(task_goal=task_goal), step_idx)
        response, points = api.call(input_data)
        step_idx += 1
        
    import pdb; pdb.set_trace()
        
    if SUBGOAL_TYPE == "grounded_subgoal":
        # get the previous image 
        # extract the <y, x> from the response
        # draw a circle on the image
        # save the image
        
        anno_image = deepcopy(video[exec_idx])
        if points:
            for point in points:
                y, x = point
                print(f"x: {x}, y: {y}")
                cv2.circle(anno_image, (x, y), 5, (255, 0, 0), -1)
            imageio.imwrite(os.path.join(save_dir, f"annotated_step_{step_idx}_image.png"), anno_image)
            import pdb; pdb.set_trace()

    
    if SUBGOAL_TYPE == "oracle_planner":
        anno_image = deepcopy(video[exec_idx])        
        if points:
            for point in points:
                y, x = point
                print(f"x: {x}, y: {y}")
                cv2.circle(anno_image, (x, y), 5, (255, 0, 0), -1)
            imageio.imwrite(os.path.join(save_dir, f"annotated_step_{step_idx}_image.png"), anno_image)
            import pdb; pdb.set_trace()
        
        
        
    for i in range(exec_idx+1, total_len, step_size): # use step_size to mock action chunk execution
        video_clip = video[i:i+step_size]
        
        if task_id == "PickHighlight":
            text_query = PICK_HIGHLIGHT_VIDEO_TEXT_QUERY
        else:
            text_query = VIDEO_TEXT_QUERY
        input_data = api.prepare_input_data(video_clip, VIDEO_TEXT_QUERY.format(task_goal=task_goal), step_idx)
    
        response, points = api.call(input_data)
        step_idx += 1
        
        if SUBGOAL_TYPE == "grounded_subgoal":
            # get the previous image 
            # extract the <y, x> from the response
            # draw a circle on the image
            # save the image
            
            anno_image = deepcopy(video[exec_idx])
            point = api.extract_point(response, anno_image.shape)
            if point is not None:
                y, x = point
                print(f"x: {x}, y: {y}")
                cv2.circle(anno_image, (x, y), 5, (255, 0, 0), -1)
                imageio.imwrite(os.path.join(save_dir, f"annotated_step_{step_idx}_image.png"), anno_image)
                import pdb; pdb.set_trace()
        
            
        
        
        if SUBGOAL_TYPE == "oracle_planner":
            anno_image = deepcopy(video[exec_idx])
            
            point = api.extract_point(response, anno_image.shape)
            if point is not None:
                y, x = point
                print(f"x: {x}, y: {y}")
                cv2.circle(anno_image, (x, y), 5, (255, 0, 0), -1)
                imageio.imwrite(os.path.join(save_dir, f"annotated_step_{step_idx}_image.png"), anno_image)
                import pdb; pdb.set_trace()

    api.save_conversation()
    
    