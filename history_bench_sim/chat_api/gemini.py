import datetime
import google.generativeai as genai
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



class Gemini:
    
    def __init__(self, save_dir: str = None, task_id: str = None, model_name: str = 'gemini-2.5-flash-lite', task_goal: str = None, subgoal_type: str = "simple_subgoal"):
        self.save_dir = save_dir
        self.task_id = task_id
        self.conversation_history = []
        self.model_name = model_name
        self.subgoal_type = subgoal_type
        assert subgoal_type in ["simple_subgoal", "grounded_subgoal", "oracle_planner"]
        
        if subgoal_type == "simple_subgoal":
            _prompt_dict = prompt_dict_simple
        elif subgoal_type == "grounded_subgoal":
            _prompt_dict = prompt_dict_grounded
        else:
            _prompt_dict = prompt_dict_oracle_planner
        
        system_prompt = _prompt_dict[task_id].replace("<task_goal>", task_goal)
        print(system_prompt)
        self.init_model(system_prompt, model_name)
        
        
    def init_model(self, system_prompt: str, model_name: str):
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt
        )        
        self.chat = self.model.start_chat(history=[])
    
    
    def call(self, input_data: dict):
        # input image or video
        # output subgoal
        image_path = input_data['image_path']
        video_path = input_data['video_path']
        text_query = input_data['text_query']
        
        try:
            print('--------------------------------')
            if image_path is not None:
                print(f"Processing image: {image_path}")
                print(f"Text query: {text_query}")
                response = self._process_image(image_path, text_query)
            elif video_path is not None:
                print(f"Processing video: {video_path}")
                print(f"Text query: {text_query}")
                response = self._process_video(video_path, text_query)
            else:
                print(f"Text query: {text_query}")
                response = self._process_text(text_query)
                
            print(f"\nResponse:\n{response}")
            print('--------------------------------')
            return response
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return None
    
    
    def _process_image(self, image_path: str | list[str], text_query: str = "What should the robot do in this situation?"):
        if isinstance(image_path, list):
            image_files = [genai.upload_file(path=path) for path in image_path]
            while all(file.state.name == "PROCESSING" for file in image_files):
                time.sleep(0.1)
                image_files = [genai.get_file(file.name) for file in image_files]
            if any(file.state.name == "FAILED" for file in image_files):
                raise ValueError(f"Image processing failed: {image_files[0].state.name}")
            
            response = self.chat.send_message([text_query, *image_files])
            
            self.conversation_history.append({
                "turn": len(self.conversation_history) + 1,
                "type": "image",
                "path": [os.path.basename(path) for path in image_path],
                "query": text_query,
                "response": response.text
            })
            
        else:
            image_file = genai.upload_file(path=image_path)
            
            while image_file.state.name == "PROCESSING":
                time.sleep(0.1)
                image_file = genai.get_file(image_file.name)
            
            if image_file.state.name == "FAILED":
                raise ValueError(f"Image processing failed: {image_file.state.name}")
            
            response = self.chat.send_message([text_query, image_file])
        
            self.conversation_history.append({
                "turn": len(self.conversation_history) + 1,
                "type": "image",
                "path": os.path.basename(image_path),
                "query": text_query,
                "response": response.text
            })
        
        return response.text

    
    def _process_video(self, video_path: str, text_query: str = "What should the robot do based on this video?"):
        video_file = genai.upload_file(path=video_path)
        
        while video_file.state.name == "PROCESSING":
            time.sleep(0.1)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.state.name}")
        
        response = self.chat.send_message([text_query, video_file])
        
        self.conversation_history.append({
            "turn": len(self.conversation_history) + 1,
            "type": "video",
            "path": os.path.basename(video_path),
            "query": text_query,
            "response": response.text
        })
        
        return response.text
    
    def _process_text(self, user_query: str = "What should the robot do based on this text?"):
        print(f"\nProcessing text: {user_query}")
        response = self.chat.send_message(user_query)
        
        self.conversation_history.append({
            "turn": len(self.conversation_history) + 1,
            "type": "text",
            "query": user_query,
            "response": response.text
        })
        
        print(f"Response: {response.text}")
        return response.text
    
    
    def save_conversation(self):
        with open(os.path.join(self.save_dir, "conversation.json"), "w") as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)



def parse_markdown_json(text):
    """Parse JSON that may be wrapped in markdown code fences."""
    text = text.strip()
    
    # Try to extract from markdown code fence
    match = re.search(r'```(?:json|JSON)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = text
    
    # Parse the JSON
    return json.loads(json_str.strip())


def parse_point_yx_from_response(string):
    # extract "press the button at <316, 149>" the <num, num> pattern and get the int numbers
    match = re.search(r'at <(\d+), (\d+)>', string)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return None


def downsample_video_to_images(video_clip, max_num_images=20, min_interval=8):
    indices = np.arange(0, len(video_clip), min_interval)
    if len(indices) > max_num_images:
        indices = indices[::len(indices)//max_num_images]
    return [video_clip[i] for i in indices]




def read_video_moviepy(video_path):
    clip = VideoFileClip(video_path)
    frames = [frame for frame in clip.iter_frames()]
    clip.close()
    return frames

if __name__ == "__main__":
    import imageio
    video_path = "examples/history_bench_sim/chat_api/videos"
    
    USE_MULTI_IMAGES_AS_VIDEO = False
    SUBGOAL_TYPE = "oracle_planner"
    if USE_MULTI_IMAGES_AS_VIDEO:
        VIDEO_TEXT_QUERY = VIDEO_TEXT_QUERY_multi_image

    model_name = "gemini-2.5-pro"
    task_id = "BinFill"
    
    video = read_video_moviepy(f"{video_path}/{task_id}.mp4")
        
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
        "VideoPlaceButton": 850,
        "VideoPlaceOrder": 850,
        
        "VideoUnmask": 81,
        "ButtonUnmask": 0,
        "ButtonUnmaskSwap": 0,
        "VideoUnmaskSwap": 219,
        
        "MoveCube": 160,
        "InsertPeg": 287,
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
    
    api = Gemini(save_dir=save_dir, task_id=task_id, model_name=model_name, task_goal=task_goal, subgoal_type=SUBGOAL_TYPE)
      
    if exec_idx > 0: # has demo
        video_clip = video[:exec_idx]
        if not USE_MULTI_IMAGES_AS_VIDEO:
            imageio.mimsave(os.path.join(save_dir, f"step_{step_idx}_video.mp4"), video_clip, fps=30)
            input_data = {  
                "image_path": None,
                "video_path": os.path.join(save_dir, f"step_{step_idx}_video.mp4"),
                "text_query": DEMO_TEXT_QUERY.format(task_goal=task_goal),
            }
        else:
            images = downsample_video_to_images(video_clip)
            os.makedirs(os.path.join(save_dir, f"step_{step_idx}_images"), exist_ok=True)
            for i, image in enumerate(images):
                imageio.imwrite(os.path.join(save_dir, f"step_{step_idx}_images", f"{i}.png"), image)
            input_data = {
                "image_path": [os.path.join(save_dir, f"step_{step_idx}_images", f"{i}.png") for i in range(len(images))],
                "video_path": None,
                "text_query": DEMO_TEXT_QUERY.format(task_goal=task_goal),
            }
            
        response = api.call(input_data)
        step_idx += 1
    
    else: # no demo
        image = video[0]  
        imageio.imwrite(os.path.join(save_dir, f"step_{step_idx}_image.png"), image)
        text_query = IMAGE_TEXT_QUERY.format(task_goal=task_goal)
        if task_id == "PickHighlight":
            text_query = PICK_HIGHLIGHT_IMAGE_TEXT_QUERY.format(task_goal=task_goal)
        input_data = {  
            "image_path": os.path.join(save_dir, f"step_{step_idx}_image.png"),
            "video_path": None,
            "text_query": text_query,
        }
        response = api.call(input_data)
        step_idx += 1
        
    if SUBGOAL_TYPE == "grounded_subgoal":
        # get the previous image 
        # extract the <y, x> from the response
        # draw a circle on the image
        # save the image
        
        anno_image = deepcopy(video[exec_idx])
        point = parse_point_yx_from_response(response)
        if point is not None:
            x = int(point[1] * anno_image.shape[1] / 1000)
            y = int(point[0] * anno_image.shape[0] / 1000)
            if  0<=x<=255 and 0<=y<=255:    
                cv2.circle(anno_image, (x, y), 5, (255, 0, 0), -1)
                imageio.imwrite(os.path.join(save_dir, f"annotated_step_{step_idx}_image.png"), anno_image)
                import pdb; pdb.set_trace()
    
    if SUBGOAL_TYPE == "oracle_planner":
        anno_image = deepcopy(video[exec_idx])
        resp_dict = parse_markdown_json(response)
        point = resp_dict['subgoal']['point']
        if point is not None:
            x = int(point[1] * anno_image.shape[1] / 1000)
            y = int(point[0] * anno_image.shape[0] / 1000)
            if  0<=x<=255 and 0<=y<=255:    
                cv2.circle(anno_image, (x, y), 5, (255, 0, 0), -1)
                imageio.imwrite(os.path.join(save_dir, f"annotated_step_{step_idx}_image.png"), anno_image)
                import pdb; pdb.set_trace()
        
        
    for i in range(exec_idx+1, total_len, step_size): # use step_size to mock action chunk execution
        video_clip = video[i:i+step_size]
        if not USE_MULTI_IMAGES_AS_VIDEO:
            imageio.mimsave(os.path.join(save_dir, f"step_{step_idx}_video.mp4"), video_clip, fps=30)
            if task_id == "PickHighlight":
                text_query = PICK_HIGHLIGHT_VIDEO_TEXT_QUERY
            else:
                text_query = VIDEO_TEXT_QUERY
            input_data = {
                "image_path": None,
                "video_path": os.path.join(save_dir, f"step_{step_idx}_video.mp4"),
                "text_query": text_query,
            }
        else:
            images = downsample_video_to_images(video_clip)
            os.makedirs(os.path.join(save_dir, f"step_{step_idx}_images"), exist_ok=True)
            for i, image in enumerate(images):
                imageio.imwrite(os.path.join(save_dir, f"step_{step_idx}_images", f"{i}.png"), image)
            input_data = {
                "image_path": [os.path.join(save_dir, f"step_{step_idx}_images", f"{i}.png") for i in range(len(images))],
                "video_path": None,
                "text_query": text_query,
            }
        response = api.call(input_data)
        step_idx += 1
        
                
        if SUBGOAL_TYPE == "grounded_subgoal":
            # get the previous image 
            # extract the <y, x> from the response
            # draw a circle on the image
            # save the image
            
            anno_image = deepcopy(video[i+step_size-1])
            point = parse_point_yx_from_response(response)
            if point is not None:
                x = int(point[1] * anno_image.shape[1] / 1000)
                y = int(point[0] * anno_image.shape[0] / 1000)
                if  0<=x<=255 and 0<=y<=255:    
                    cv2.circle(anno_image, (x, y), 5, (255, 0, 0), -1)
                    imageio.imwrite(os.path.join(save_dir, f"annotated_step_{step_idx}_image.png"), anno_image)
                    import pdb; pdb.set_trace()
        
        if SUBGOAL_TYPE == "oracle_planner":
            anno_image = deepcopy(video[i+step_size-1])
            resp_dict = parse_markdown_json(response)
            point = resp_dict['subgoal']['point']
            if point is not None:
                x = int(point[1] * anno_image.shape[1] / 1000)
                y = int(point[0] * anno_image.shape[0] / 1000)
                if  0<=x<=255 and 0<=y<=255:    
                    cv2.circle(anno_image, (x, y), 5, (255, 0, 0), -1)
                    imageio.imwrite(os.path.join(save_dir, f"annotated_step_{step_idx}_image.png"), anno_image)
                    import pdb; pdb.set_trace()
            
    api.save_conversation()
    
    