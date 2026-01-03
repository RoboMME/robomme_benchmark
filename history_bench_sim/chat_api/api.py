import os
import json
import time
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Dict
import google.generativeai as genai
from openai import OpenAI, AzureOpenAI

from .prompts import *

import numpy as np
import imageio
import re
import base64
import shutil
import cv2




def parse_markdown_json(text):
    """Parse JSON that may be wrapped in markdown code fences."""
    text = text.strip()
    
    match = re.search(r'"subgoal"\s*:\s*(\{.*?\})', text, re.DOTALL)

    if match:
        subgoal_str = match.group(1)
        subgoal = json.loads(subgoal_str)
        out = {"subgoal": subgoal}
    else:
        return None
    return out


def parse_point_from_str(string):
    # extract "press the button at <316, 149>" the <num, num> pattern and get the int numbers
    match = re.search(r'at <(\d+), (\d+)>', string)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return None



def downsample_video_to_images(video_clip, max_num_images=20, min_interval=8):
    indices = np.arange(0, len(video_clip), min_interval)
    if len(indices) > max_num_images:
        indices = np.linspace(0, len(video_clip)-1, max_num_images).astype(int)
    return [video_clip[i] for i in indices]

class BaseModel(ABC):
    """Base class for vision-language models"""
    
    def __init__(self, save_dir: str = None, task_id: str = None, 
                 model_name: str = None, task_goal: str = None, 
                 subgoal_type: str = "simple_subgoal", 
                 use_multi_images_as_video: bool = False,
                 image_size: tuple = (256, 256)):
        """
        初始化基础模型
        
        Args:
            save_dir: 保存对话历史和图像的目录路径
            task_id: 任务标识符（如 "PatternLock", "BinFill", "PickHighlight" 等）
            model_name: 模型名称或路径
            task_goal: 任务目标描述（如 "解锁图案锁"）
            subgoal_type: 子目标类型，决定使用哪种提示模板
                - "simple_subgoal": 简单子目标，输出纯文本子目标
                - "grounded_subgoal": 带定位信息的子目标，输出包含坐标点
                - "oracle_planner": Oracle规划器，输出结构化的action和point
            use_multi_images_as_video: 是否将多张图片作为视频处理
            image_size: 图像尺寸，用于坐标归一化
        """
        # 步骤 1：初始化保存目录
        # 如果目录已存在，先删除再创建，确保每次运行都是干净的环境
        self.save_dir = save_dir
        if save_dir:
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir, exist_ok=True)
        
        # 步骤 2：保存基本参数
        self.task_id = task_id
        self.conversation_history = []  # 用于记录所有对话历史，便于后续保存和调试
        self.model_name = model_name
        self.subgoal_type = subgoal_type
        self.use_multi_images_as_video = use_multi_images_as_video
        self.image_size = image_size
        
        # 步骤 3：验证 subgoal_type 的有效性
        assert subgoal_type in ["simple_subgoal", "grounded_subgoal", "oracle_planner"]
        
        # 步骤 4：根据 subgoal_type 选择对应的提示字典
        # 每个字典包含不同任务（task_id）的提示模板
        # Load appropriate prompt dictionary
        if subgoal_type == "simple_subgoal":
            _prompt_dict = prompt_dict_simple
        elif subgoal_type == "grounded_subgoal":
            _prompt_dict = prompt_dict_grounded
        else:
            _prompt_dict = prompt_dict_oracle_planner
        
        # 步骤 5：构建 system_prompt（系统提示词）
        # 
        # system_prompt 的组成结构：
        # 
        # 1. 基础模板部分（来自 prompts/base.py）：
        #    - SYSTEM_PROMPT_with_DEMO: 用于 simple_subgoal 和 grounded_subgoal
        #      包含：角色定义、输出格式说明（JSON格式要求）
        #    - SYSTEM_PROMPT_ORACLE_PLANNER: 用于 oracle_planner
        #      包含：角色定义、输出格式说明（包含action和point的结构化格式）
        # 
        # 2. 任务特定内容（来自 prompts/{TaskName}.py，如 PatternLock.py）：
        #    - subgoals: 该任务所有可能的子目标动作列表
        #      例如 PatternLock: "move forward", "move left", "move right" 等
        #    - example: 任务分解示例，展示如何将任务目标分解为子目标序列
        #      例如："1. move forward, 2. move left, 3. move forward-left, ..."
        #    - notes: 任务特定的重要说明
        #      例如：坐标变换规则、目标识别规则、完成条件等
        # 
        # 3. 模板组装过程（在 prompts/{TaskName}.py 中完成）：
        #    TaskName_SYSTEM_PROMPT = SYSTEM_PROMPT_with_DEMO.format(
        #        subgoals=subgoals,           # 填充 {subgoals} 占位符
        #        example=example + notes       # 填充 {example} 占位符
        #    )
        # 
        # 4. 最终 system_prompt 的完整结构：
        #    - 角色定义："You are assisting with predicting the next language subgoal for a robot."
        #    - 可能的子目标列表：根据任务类型列出所有可用的子目标动作
        #    - 示例说明：展示如何分解任务和预测子目标
        #    - 任务特定说明：坐标变换、识别规则等
        #    - 输出格式要求：
        #      * 第一帧输入：输出 subgoal_sequence 和第一个 subgoal
        #      * 视频片段输入：输出 description 和下一个 subgoal
        #      * JSON 格式规范
        # 
        # 5. task_goal 替换：
        #    模板中可能包含 <task_goal> 占位符，会被实际的 task_goal 参数替换
        #    例如："The task goal is '<task_goal>'" -> "The task goal is '解锁图案锁'"
        # 
        # 6. 不同 subgoal_type 的区别：
        #    - simple_subgoal: 输出纯文本子目标，如 "move forward"
        #    - grounded_subgoal: 输出带坐标的子目标，如 "pick up the object at <316, 149>"
        #    - oracle_planner: 输出结构化格式，如 {"action": "move forward", "point": [y, x]}
        # 
        # 从字典中获取任务特定的提示模板，并替换 <task_goal> 占位符
        system_prompt = _prompt_dict[task_id].replace("<task_goal>", task_goal)
        
        # 保存 system_prompt 为实例变量，以便后续保存到 conversation.json
        self.system_prompt = system_prompt
        
        # 打印最终的 system_prompt，便于调试和查看
        print(f"\"\"\"\nSystem prompt:\n{system_prompt}\n\"\"\"")
        
        # 步骤 6：初始化模型，传入 system_prompt 和 model_name
        # system_prompt 将作为系统消息添加到对话历史中，指导模型的行为
        self.init_model(system_prompt, model_name)
        
        # 步骤 7：初始化状态变量
        self.total_images = []      # 记录所有处理过的图像
        self.subgoals = []          # 记录所有预测的子目标
        self.last_subgoal = None    # 记录最后一个子目标，用于上下文传递
    
    
    def _normalize_point(self, point: tuple, image_size: tuple | None) -> tuple:
        if image_size is None:
            return point
        x = int(point[1] * image_size[1] / 1000)
        x = np.clip(x, 0, image_size[1])
        y = int(point[0] * image_size[0] / 1000)
        y = np.clip(y, 0, image_size[0])
        return y, x


    def normalize_point_in_response(self, response: dict) -> dict:
        points = []
        if self.subgoal_type == "grounded_subgoal":
            subgoal = response['subgoal']
            matches = re.findall(r'<(\d+), (\d+)>', subgoal)
            for match in matches:
                y = int(match[0])
                x = int(match[1])
                point = self._normalize_point((y, x), self.image_size)
                points.append(point)
                original = f'<{match[0]}, {match[1]}>'
                subgoal = subgoal.replace(original, f'<{point[0]}, {point[1]}>')
            response['subgoal'] = subgoal
        elif self.subgoal_type == "oracle_planner":
            subgoal = response.get("subgoal")
            # 兼容模型可能返回字符串或缺少 point 的情况
            if not isinstance(subgoal, dict):
                subgoal = {"action": subgoal, "point": None}
            if "point" not in subgoal:
                subgoal["point"] = None
            response["subgoal"] = subgoal

            point = subgoal.get("point")
            if point is not None:
                point = self._normalize_point(point, self.image_size)
                response['subgoal']['point'] = point
                points.append(point)
    
        return response, points
    
    @abstractmethod
    def init_model(self, system_prompt: str, model_name: str):
        """Initialize the model with system prompt"""
        pass
    
    def call(self, input_data: dict) -> Optional[str]:
        """Main entry point for processing inputs"""
        image_path = input_data.get('image_path')
        video_path = input_data.get('video_path')
        text_query = input_data.get('text_query', '')
        
        try:
            print('--------------------------------')
            if image_path is not None:
                print(f"Processing image: {image_path}")
                print(f"Text query: {text_query}")
                print("calling api...")
                response = self._process_image(image_path, text_query)
            elif video_path is not None:
                print(f"Processing video: {video_path}")
                print(f"Text query: {text_query}")
                print("calling api...")
                response = self._process_video(video_path, text_query)
            else:
                print(f"Text query: {text_query}")
                print("calling api...")
                response = self._process_text(text_query)
                
            print(f"\nResponse:\n{response}")
            print('--------------------------------')
            response = parse_markdown_json(response)
            response, points = self.normalize_point_in_response(response)
            self.last_subgoal = response['subgoal']
            return response, points
        
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            self.last_subgoal = None
            return None, None
    
    def prepare_input_data(self, image_query: Union[np.ndarray, List[np.ndarray]], text_query: str, step_idx: int) -> dict:
        """Prepare input data for the model"""
        # 检查空列表情况
        if isinstance(image_query, list) and len(image_query) == 0:
            raise ValueError(f"image_query is empty at step {step_idx}. Cannot prepare input data without images.")
        
        if not isinstance(image_query, list) or (isinstance(image_query, list) and len(image_query) <= 3):
            # will consider it as a single image, take the last one
            imageio.imwrite(os.path.join(self.save_dir, f"step_{step_idx}_image.png"), image_query[-1])
            input_data = {
                "image_path": os.path.join(self.save_dir, f"step_{step_idx}_image.png"),
                "video_path": None,
                "text_query": text_query,
            }
            self.total_images.append(image_query[-1])
            self.subgoals.append(self.last_subgoal)
            return input_data
        else:
            self.total_images.extend(image_query)
            self.subgoals.extend([self.last_subgoal] * len(image_query))
            if self.use_multi_images_as_video:
                images = downsample_video_to_images(image_query)
                os.makedirs(os.path.join(self.save_dir, f"step_{step_idx}_images"), exist_ok=True)
                for i, image in enumerate(images):
                    imageio.imwrite(os.path.join(self.save_dir, f"step_{step_idx}_images", f"{i}.png"), image)
                input_data = {
                    "image_path": [os.path.join(self.save_dir, f"step_{step_idx}_images", f"{i}.png") for i in range(len(images))],
                    "video_path": None,
                    "text_query": text_query,
                }
                return input_data
            else:
                imageio.mimsave(os.path.join(self.save_dir, f"step_{step_idx}_video.mp4"), image_query, fps=30)
                input_data = {
                    "image_path": None,
                    "video_path": os.path.join(self.save_dir, f"step_{step_idx}_video.mp4"),
                    "text_query": text_query,
                }
                return input_data
    
    def save_conversation(self):
        """Save conversation history to JSON"""
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            conversation_data = {
                "system_prompt": self.system_prompt,
                "conversation_history": self.conversation_history
            }
            with open(os.path.join(self.save_dir, "conversation.json"), "w") as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)

    def _wrap_text(self, text: str, font, font_scale, thickness, max_width):
        """Wrap text to fit within max_width pixels"""
        words = text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            
            if text_width <= max_width - 20:  # 20 pixels margin
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Single word is too long, add it anyway
                    lines.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines

    def save_final_video(self, video_path: str):
        if len(self.total_images) != len(self.subgoals):
            raise ValueError(f"Total images and subgoals must have the same length")
        final_images = []
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        line_height = 25
        
        max_text_height = 80
        last_subgoal = None
        for i, (image, subgoal) in enumerate(zip(self.total_images, self.subgoals)):
            if subgoal is None:
                subgoal = "[initializing...]"
            
            if isinstance(subgoal, dict):
                subgoal = subgoal['action'] if subgoal['point'] is None else f"{subgoal['action']} <{subgoal['point'][0]}, {subgoal['point'][1]}>"
            
            # Wrap text to fit image width
            lines = self._wrap_text(subgoal, font, font_scale, thickness, image.shape[1])
            
            # Create black image with the maximum text height for all frames
            black_image = np.zeros((image.shape[0] + max_text_height, image.shape[1], 3), dtype=np.uint8)
                        
            # Add each line of text
            for i, line in enumerate(lines):
                y_position = (i + 1) * line_height
                black_image = cv2.putText(black_image, line, (10, y_position), font, font_scale, (255, 255, 255), thickness)
            
            black_image[max_text_height:, :] = image
            final_images.append(black_image)
            if subgoal != last_subgoal:
                final_images.extend([black_image] * 30)
            last_subgoal = subgoal
        imageio.mimsave(video_path, final_images, fps=45)
        
        
    def add_frame_hold(self, image: np.ndarray, hold_len: int = 10):
        for i in range(hold_len):
            self.total_images.append(image.copy())
            self.subgoals.append(self.last_subgoal)
    
class GeminiModel(BaseModel):
    """Gemini model implementation"""
    
    def init_model(self, system_prompt: str, model_name: str):
        self.model = genai.GenerativeModel(
            model_name=model_name or 'gemini-2.5-flash-lite',
            system_instruction=system_prompt
        )
        self.chat = self.model.start_chat(history=[])
        self.all_uploaded_file_names = []
    
    def _process_image(self, image_path: Union[str, List[str]], 
                      text_query: str = "What should the robot do in this situation?") -> str:
        if isinstance(image_path, list):
            image_files = [genai.upload_file(path=path) for path in image_path]
            self.all_uploaded_file_names.extend([image_file.name for image_file in image_files])
            while all(file.state.name == "PROCESSING" for file in image_files):
                time.sleep(0.1)
                image_files = [genai.get_file(file.name) for file in image_files]
            if any(file.state.name == "FAILED" for file in image_files):
                raise ValueError(f"Image processing failed")
            
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
            self.all_uploaded_file_names.append(image_file.name)
            
            while image_file.state.name == "PROCESSING":
                time.sleep(0.1)
                image_file = genai.get_file(image_file.name)
            
            if image_file.state.name == "FAILED":
                raise ValueError(f"Image processing failed")
            
            response = self.chat.send_message([text_query, image_file])
            
            self.conversation_history.append({
                "turn": len(self.conversation_history) + 1,
                "type": "image",
                "path": os.path.basename(image_path),
                "query": text_query,
                "response": response.text
            })
        
        return response.text
    
    def _process_video(self, video_path: str, 
                      text_query: str = "What should the robot do based on this video?") -> str:
        video_file = genai.upload_file(path=video_path)
        self.all_uploaded_file_names.append(video_file.name)
        
        while video_file.state.name == "PROCESSING":
            time.sleep(0.1)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed")
        
        response = self.chat.send_message([text_query, video_file])
        
        self.conversation_history.append({
            "turn": len(self.conversation_history) + 1,
            "type": "video",
            "path": os.path.basename(video_path),
            "query": text_query,
            "response": response.text
        })
        
        return response.text
    
    def _process_text(self, user_query: str) -> str:
        response = self.chat.send_message(user_query)
        
        self.conversation_history.append({
            "turn": len(self.conversation_history) + 1,
            "type": "text",
            "query": user_query,
            "response": response.text
        })
        
        return response.text

    def clear_uploaded_files(self):
        try:
            for file_name in self.all_uploaded_file_names:
                genai.delete_file(file_name)
            print(f"Cleared {len(self.all_uploaded_file_names)} uploaded files")
            self.all_uploaded_file_names = []
        except Exception as e:
            print(f"Error clearing uploaded files: {e}")


class OpenAIModel(BaseModel):
    
    def init_model(self, system_prompt: str, model_name: str):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.conv = self.client.conversations.create()
        self.message = [{
            "role": "system",
            "content": [
                {"type": "input_text", "text": system_prompt},
            ],
        }]
        self.step = 0
        self.use_multi_images_as_video = True
    
    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        
        return f"data:image/jpeg;base64,{b64}"
    
    def print_message(self):
        print("--------Message History Start--------")
        for message in self.message:
            # do not print base64_image, use [IMAGE] instead
            print(f"Role: {message['role']}")
            for content in message['content']:
                if content['type'] == 'input_image':
                    print(f"  [IMAGE]")
                else:
                    print(f"  {content['text']}")
        print("--------Message History End--------")
    
    def _process_image(self, image_path: Union[str, List[str]], 
                      text_query: str = "What should the robot do in this situation?") -> str:
        content = []        
        image_paths = image_path if isinstance(image_path, list) else [image_path]
        
        for path in image_paths:
            base64_image = self._encode_image(path)
            content.append({
                "type": "input_image",
                "image_url": base64_image
            })
        content.append({"type": "input_text", "text": text_query})        
        self.message.append({
            "role": "user",
            "content": content
        })
                
        response = self.client.responses.create(
            model=self.model_name,
            conversation=self.conv.id,
            input=self.message[-2:]
        )
        
        assistant_message = response.output[0].content[0].text
        self.message.append({
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": assistant_message},
            ],
        })
                
        self.conversation_history.append({
            "turn": len(self.conversation_history) + 1,
            "type": "image",
            "path": [os.path.basename(p) for p in image_paths] if isinstance(image_path, list) else os.path.basename(image_path),
            "query": text_query,
            "response": assistant_message
        })
    
        return assistant_message
    
    
    def _process_text(self, user_query: str) -> str:
        self.messages.append({"role": "user", "content": user_query})
        
        response = self.client.responses.create(
            model=self.model_name,
            conversation=self.conv.id,
            input=self.message[-2:]
        )
        
        assistant_message = response.output[0].content[0].text
        self.message.append({
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": assistant_message},
            ],
        })
        
        self.conversation_history.append({
            "turn": len(self.conversation_history) + 1,
            "type": "text",
            "query": user_query,
            "response": assistant_message
        })
        
        return assistant_message

    def clear_uploaded_files(self):
        """Placeholder method for compatibility with other model classes.
        OpenAIModel doesn't upload files to external services, so this is a no-op."""
        pass

class QwenModel(BaseModel):
    
    def init_model(self, system_prompt: str, model_name: str):
        # Use DASHSCOPE_API_KEY if available, else OPENAI_API_KEY
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model_name = model_name or "qwen-vl-max"
        self.messages = [{
            "role": "system",
            "content": system_prompt
        }]
        self.use_multi_images_as_video = True

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _process_image(self, image_path: Union[str, List[str]], 
                      text_query: str = "What should the robot do in this situation?") -> str:
        content = []        
        image_paths = image_path if isinstance(image_path, list) else [image_path]
        
        for path in image_paths:
            base64_image = self._encode_image(path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": base64_image
                }
            })
        content.append({"type": "text", "text": text_query})        
        
        self.messages.append({
            "role": "user",
            "content": content
        })
                
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages
        )
        
        assistant_message = response.choices[0].message.content
        self.messages.append({
            "role": "assistant",
            "content": assistant_message
        })
                
        self.conversation_history.append({
            "turn": len(self.conversation_history) + 1,
            "type": "image",
            "path": [os.path.basename(p) for p in image_paths] if isinstance(image_path, list) else os.path.basename(image_path),
            "query": text_query,
            "response": assistant_message
        })
    
        return assistant_message
    
    def _process_text(self, user_query: str) -> str:
        self.messages.append({"role": "user", "content": user_query})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages
        )
        
        assistant_message = response.choices[0].message.content
        self.messages.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        self.conversation_history.append({
            "turn": len(self.conversation_history) + 1,
            "type": "text",
            "query": user_query,
            "response": assistant_message
        })
        
        return assistant_message

    def _process_video(self, video_path: str, text_query: str) -> str:
        # If this is called, it means use_multi_images_as_video was False or prepare_input_data logic bypassed
        # For Qwen via OpenAI compatible API, handling video directly is complex (usually frames).
        # We'll treat it as not implemented or try to use frames if we had them.
        # But prepare_input_data for BaseModel will handle video->images if configured.
        raise NotImplementedError("Video processing not directly supported for Qwen via OpenAI API in this implementation. Use image sequence.")

    def clear_uploaded_files(self):
        """Placeholder method for compatibility with other model classes.
        QwenModel doesn't upload files to external services, so this is a no-op."""
        pass

class LocalModel(BaseModel):
    
    def init_model(self, system_prompt: str, model_name: str):
        """
        初始化本地模型客户端和对话系统
        
        Args:
            system_prompt: 系统提示词，用于指导模型的行为和输出格式
                system_prompt 的组成（在 BaseModel.__init__ 中构建）：
                1. 根据 subgoal_type 选择提示字典：
                   - "simple_subgoal" -> prompt_dict_simple
                   - "grounded_subgoal" -> prompt_dict_grounded
                   - "oracle_planner" -> prompt_dict_oracle_planner
                2. 从字典中根据 task_id 获取任务特定的提示模板
                   （如 "PatternLock", "BinFill", "PickHighlight" 等）
                3. 提示模板包含以下部分：
                   - 基础模板（如 SYSTEM_PROMPT_with_DEMO, SYSTEM_PROMPT_ORACLE_PLANNER）
                   - 任务特定的 subgoals 列表（可能的子目标动作）
                   - 任务特定的 example 示例（展示如何分解任务）
                   - 任务特定的 notes 说明（如坐标变换、目标识别规则等）
                   - 输出格式说明（JSON 格式要求）
                4. 模板中的 <task_goal> 占位符会被实际的 task_goal 替换
                5. 最终生成的 system_prompt 用于指导模型：
                   - 理解任务目标和可能的子目标
                   - 根据输入（图片/视频）预测下一个子目标
                   - 按照指定格式输出 JSON 响应
            model_name: 模型名称或路径
        """
        # 步骤 1：初始化 OpenAI 客户端，连接到本地模型服务
        # base_url 指向本地部署的模型服务（通常是 vLLM 或其他兼容 OpenAI API 的服务）
        self.client = OpenAI(
            api_key="EMPTY",  # 本地服务不需要真实的 API key
            base_url="http://localhost:22003/v1",  # 本地模型服务的地址
            timeout=3600  # 超时时间设置为 3600 秒（1小时），因为视频处理可能耗时较长
        )
        
        # 步骤 2：设置模型名称/路径
        # 如果 model_name 是 "local" 或为空，使用默认的 Qwen3-VL-32B-Instruct 模型路径
        # 否则使用传入的 model_name（可以是其他模型路径）
        if model_name == "local" or not model_name:
            self.model_name = "/nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/Qwen3-VL-32B-Instruct"
        else:
            self.model_name = model_name
        
        # 步骤 3：初始化对话消息列表，添加系统提示词
        # system_prompt 包含了任务说明、子目标列表、示例和输出格式要求
        # 这个系统提示词会在整个对话过程中指导模型的行为
        self.messages = [{
            "role": "system",
            "content": system_prompt
        }]
        
        # 步骤 4：设置视频处理模式
        # 直接走视频推理，不再拆多图
        # 当有多帧图像时，会将其打包成视频文件而不是多张图片
        self.use_multi_images_as_video = False

    def prepare_input_data(self, image_query: Union[np.ndarray, List[np.ndarray]], text_query: str, step_idx: int) -> dict:
        """Override: 始终将当前帧序列打包成mp4供视频模型使用"""
        if isinstance(image_query, list):
            if len(image_query) == 0:
                raise ValueError(f"image_query is empty at step {step_idx}. Cannot prepare input data without images.")
            frames = image_query
        else:
            frames = [image_query]
        video_path = os.path.join(self.save_dir, f"step_{step_idx}_video.mp4")
        # 将当前帧序列编码为 mp4，供后续视频模型直接消费
        imageio.mimsave(video_path, frames, fps=30)
        self.total_images.extend(frames)
        self.subgoals.extend([self.last_subgoal] * len(frames))
        return {
            "image_path": None,
            "video_path": video_path,
            "text_query": text_query,
        }

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _process_image(self, image_path: Union[str, List[str]], 
                      text_query: str = "What should the robot do in this situation?") -> str:
        content = []        
        image_paths = image_path if isinstance(image_path, list) else [image_path]
        
        for path in image_paths:
            base64_image = self._encode_image(path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": base64_image
                }
            })
        content.append({"type": "text", "text": text_query})        
        
        self.messages.append({
            "role": "user",
            "content": content
        })
                
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            max_tokens=2048
        )
        
        assistant_message = response.choices[0].message.content
        self.messages.append({
            "role": "assistant",
            "content": assistant_message
        })
                
        self.conversation_history.append({
            "turn": len(self.conversation_history) + 1,
            "type": "image",
            "path": [os.path.basename(p) for p in image_paths] if isinstance(image_path, list) else os.path.basename(image_path),
            "query": text_query,
            "response": assistant_message
        })
    
        return assistant_message
    
    def _process_text(self, user_query: str) -> str:
        self.messages.append({"role": "user", "content": user_query})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            max_tokens=2048
        )
        
        assistant_message = response.choices[0].message.content
        self.messages.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        self.conversation_history.append({
            "turn": len(self.conversation_history) + 1,
            "type": "text",
            "query": user_query,
            "response": assistant_message
        })
        
        return assistant_message

    def _process_video(self, video_path: str, text_query: str) -> str:
        """
        处理视频输入并调用API获取响应
        
        步骤：
        1. 读取本地视频文件并转换为base64编码
        2. 构建data URL格式的视频数据
        3. 组装包含视频和文本查询的消息内容
        4. 将用户消息添加到对话历史
        5. 调用API进行视频理解
        6. 提取并保存助手回复
        7. 记录到会话历史
        8. 返回助手消息
        """
        # 步骤 1：读取本地视频文件并转换为base64编码
        # 将本地 mp4 转为 data:video/mp4;base64 直接随消息发送，无需额外上传
        with open(video_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        
        # 步骤 2：构建data URL格式，用于在消息中嵌入视频数据
        data_url = f"data:video/mp4;base64,{b64}"
        
        # 步骤 3：组装消息内容，包含视频URL和文本查询
        content = [
            {
                "type": "video_url",
                "video_url": {
                    "url": data_url
                }
            },
            {"type": "text", "text": text_query}
        ]
        
        # 步骤 4：将用户消息（包含视频和文本查询）添加到对话历史
        self.messages.append({
            "role": "user",
            "content": content
        })
        
        # 步骤 5：调用API进行视频理解，传入完整的对话历史
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            max_tokens=2048
        )
        
        # 步骤 6：从API响应中提取助手回复内容
        assistant_message = response.choices[0].message.content
        
        # 步骤 7：将助手回复添加到对话历史，保持对话上下文
        self.messages.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        # 步骤 8：将本次交互记录到会话历史，用于后续保存和调试
        self.conversation_history.append({
            "turn": len(self.conversation_history) + 1,
            "type": "video",
            "path": os.path.basename(video_path),
            "query": text_query,
            "response": assistant_message
        })
    
        # 步骤 9：返回助手消息内容
        return assistant_message

    def clear_uploaded_files(self):
        """Placeholder method for compatibility with other model classes.
        LocalModel doesn't upload files to external services, so this is a no-op."""
        pass
