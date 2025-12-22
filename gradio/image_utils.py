"""
图像处理工具模块
无状态的图像处理函数
"""
import numpy as np
import tempfile
import os
import traceback
from PIL import Image, ImageDraw
import cv2


def save_video(frames, suffix=""):
    """
    视频保存函数 - 使用imageio生成视频
    
    优化点：
    1. 使用imageio.mimwrite，不依赖FFmpeg编码器
    2. 直接处理RGB帧，无需颜色空间转换
    3. 自动处理编码，简单可靠
    """
    if not frames or len(frames) == 0:
        return None
    
    try:
        import imageio
        
        # 准备帧：确保是uint8格式的numpy数组
        processed_frames = []
        for f in frames:
            if not isinstance(f, np.ndarray):
                f = np.array(f)
            # 确保是uint8格式
            if f.dtype != np.uint8:
                if np.max(f) <= 1.0:
                    f = (f * 255).astype(np.uint8)
                else:
                    f = f.clip(0, 255).astype(np.uint8)
            # 处理灰度图
            if len(f.shape) == 2:
                f = np.stack([f] * 3, axis=-1)
            # imageio期望RGB格式，frames已经是RGB
            processed_frames.append(f)
        
        fd, path = tempfile.mkstemp(suffix=f"_{suffix}.mp4")
        os.close(fd)
        
        # imageio.mimwrite会自动处理编码
        imageio.mimwrite(path, processed_frames, fps=10.0, quality=8, macro_block_size=None)
        
        return path
    except ImportError:
        print("Error: imageio module not found. Please install it: pip install imageio imageio-ffmpeg")
        return None
    except Exception as e:
        print(f"Error in save_video: {e}")
        traceback.print_exc()
        return None


def concatenate_frames_horizontally(frames1, frames2):
    """
    将两个帧序列左右拼接成一个帧序列
    
    Args:
        frames1: 左侧视频帧列表（base frames）
        frames2: 右侧视频帧列表（wrist frames）
    
    Returns:
        拼接后的帧列表
    """
    if not frames1 and not frames2:
        return []
    
    # 确定最大帧数
    max_frames = max(len(frames1), len(frames2))
    concatenated_frames = []
    
    for i in range(max_frames):
        # 获取当前帧，如果某个序列较短，重复最后一帧
        frame1 = frames1[min(i, len(frames1) - 1)] if frames1 else None
        frame2 = frames2[min(i, len(frames2) - 1)] if frames2 else None
        
        # 转换为numpy数组并确保格式正确
        if frame1 is not None:
            if not isinstance(frame1, np.ndarray):
                frame1 = np.array(frame1)
            if frame1.dtype != np.uint8:
                if np.max(frame1) <= 1.0:
                    frame1 = (frame1 * 255).astype(np.uint8)
                else:
                    frame1 = frame1.clip(0, 255).astype(np.uint8)
            if len(frame1.shape) == 2:
                frame1 = np.stack([frame1] * 3, axis=-1)
        else:
            # 如果frame1为空，创建一个黑色帧
            if frame2 is not None:
                h, w = frame2.shape[:2]
                frame1 = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                continue
        
        if frame2 is not None:
            if not isinstance(frame2, np.ndarray):
                frame2 = np.array(frame2)
            if frame2.dtype != np.uint8:
                if np.max(frame2) <= 1.0:
                    frame2 = (frame2 * 255).astype(np.uint8)
                else:
                    frame2 = frame2.clip(0, 255).astype(np.uint8)
            if len(frame2.shape) == 2:
                frame2 = np.stack([frame2] * 3, axis=-1)
        else:
            # 如果frame2为空，创建一个黑色帧
            h, w = frame1.shape[:2]
            frame2 = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 确保两个帧的高度相同，如果不同则调整
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        if h1 != h2:
            # 调整到相同高度（使用较小的那个）
            target_h = min(h1, h2)
            if h1 != target_h:
                frame1 = cv2.resize(frame1, (w1, target_h), interpolation=cv2.INTER_LINEAR)
            if h2 != target_h:
                frame2 = cv2.resize(frame2, (w2, target_h), interpolation=cv2.INTER_LINEAR)
        
        # 左右拼接
        concatenated_frame = np.concatenate([frame1, frame2], axis=1)
        concatenated_frames.append(concatenated_frame)
    
    return concatenated_frames


def draw_marker(img, x, y):
    """Draws a red circle and cross at (x, y)."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    img = img.copy()
    draw = ImageDraw.Draw(img)
    r = 5
    # Circle
    draw.ellipse((x-r, y-r, x+r, y+r), outline="red", width=2)
    # Cross
    draw.line((x-r, y, x+r, y), fill="red", width=2)
    draw.line((x, y-r, x, y+r), fill="red", width=2)
    return img
