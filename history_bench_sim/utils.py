import warnings
import os
import logging
import numpy as np
import cv2
import imageio
import re

def suppress_warnings():
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
    warnings.filterwarnings("ignore", message=".*env.task_list.*")
    warnings.filterwarnings("ignore", message=".*env.elapsed_steps.*")
    warnings.filterwarnings("ignore", message=".*panda_wristcam is not in the task's list of supported robots.*")
    warnings.filterwarnings("ignore", message=".*No initial pose set for actor builder.*")

    warnings.filterwarnings("ignore", category=UserWarning, module="mani_skill")

    # Suppress ManiSkill warnings - comprehensive approach
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

    # Set up logging to suppress all warnings
    logging.basicConfig(level=logging.CRITICAL)
    logging.getLogger("mani_skill").setLevel(logging.CRITICAL)
    logging.getLogger("mani_skill").propagate = False
    

def pack_buffer(image_buffer, state_buffer, exec_start_idx=0):
    image_output = np.stack(image_buffer, axis=0).astype(np.uint8)[:, None]
    state_output = np.stack(state_buffer, axis=0).astype(np.float32)
    return {
        "images": image_output,
        "state": state_output,
        "add_buffer": True,
        "exec_start_idx": exec_start_idx,
    }
    
def check_args(args):
    assert args.symbolic_memory in ["simple_subgoal", "grounded_subgoal", "action_history", None]



class RolloutRecorder:
    def __init__(self, save_dir: str, task_goal: str, fps: int = 30):
        self.save_dir = save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        self.total_images = []
        self.fps = fps
        self.task_goal = task_goal
        
    def record(self, image: np.ndarray, wrist_image: np.ndarray, state: np.ndarray, action: np.ndarray=None, is_video: bool=False, subgoal: str = None):
        if not is_video:
            concat_image = np.concatenate([image, wrist_image], axis=1)
        else:
            concat_image = np.concatenate(
            [cv2.rectangle(image, (0, 0), (256, 256), (255, 0, 0), 2), 
                cv2.rectangle(wrist_image, (0, 0), (256, 256), (255, 0, 0), 2)],
            axis=1
        )
        frame_text = "Frame: " + str(len(self.total_images))
        frame_text_area = self.add_text_area(frame_text, concat_image.shape)
        
        goal_text = "Task Goal: " + self.task_goal
        goal_text_area = self.add_text_area(goal_text, concat_image.shape)
        
        if subgoal is not None:
            subgoal_text = "Subgoal: " + subgoal
            subgoal_text_area = self.add_text_area(subgoal_text, concat_image.shape)
            concat_image = np.concatenate([subgoal_text_area, concat_image], axis=0)
        
        state_text = "State: " + ','.join([f"{i:.4f}" for i in state])
        state_text_area = self.add_text_area(state_text, concat_image.shape)
        
        action_text = 'Action: ' + ','.join([f"{i:.4f}" for i in action]) if action is not None else "None"
        action_text_area = self.add_text_area(action_text, concat_image.shape)
        # Concatenate text area on top of image
        concat_image = np.concatenate([frame_text_area, goal_text_area, action_text_area, state_text_area, concat_image], axis=0)
        self.total_images.append(concat_image)
    
    def add_text_area(self, text: str, concat_image_shape: tuple):        
        # Calculate text wrapping
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        max_width = concat_image_shape[1] - 20  # Leave 10px margin on each side
        lines = []
        words = text.replace(',', ' ').split()
        current_line = words[0]
        for word in words[1:]:
            test_line = current_line + ' ' + word
            (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            
            if text_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        
        lines.append(current_line)  # Add the last line
        
        # Create text area with dynamic height
        line_height = 20
        text_area_height = max(50, len(lines) * line_height + 10)
        text_area = np.zeros((text_area_height, concat_image_shape[1], 3), dtype=np.uint8)
        
        # Draw each line
        for i, line in enumerate(lines):
            y_position = 15 + i * line_height
            text_area = cv2.putText(text_area, line, (10, y_position), font, font_scale, (255, 255, 255), thickness)
        
        return text_area
             
    def save_video(self, filename: str):
        imageio.mimsave(os.path.join(self.save_dir, filename), self.total_images, fps=self.fps)
    