
import datetime
import google.generativeai as genai
import time
import os
import json
import shutil
import numpy as np
from moviepy import VideoFileClip
import re
from history_bench_sim.chat_api.prompts import *



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




def read_video_moviepy(video_path, use_concatenated_image=True):
    clip = VideoFileClip(video_path)
    frames = [frame for frame in clip.iter_frames()]
    clip.close()
    
    if not use_concatenated_image:
        new_frames = []
        for frame in frames:
            width = frame.shape[1] // 2
            new_fr = frame[:,:width,:]
            new_frames.append(new_fr)
        return new_frames
        
    return frames
