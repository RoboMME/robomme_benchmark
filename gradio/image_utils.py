"""
图像处理工具模块
无状态的图像处理函数
"""
import numpy as np
import tempfile
import os
import traceback
import math
from PIL import Image, ImageDraw, ImageFont
import cv2
from config import VIDEO_PLAYBACK_FPS


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
        imageio.mimwrite(path, processed_frames, fps=VIDEO_PLAYBACK_FPS, quality=8, macro_block_size=None)

        return path
    except ImportError:
        print("Error: imageio module not found. Please install it: pip install imageio imageio-ffmpeg")
        return None
    except Exception as e:
        print(f"Error in save_video: {e}")
        traceback.print_exc()
        return None


def concatenate_frames_horizontally(frames1, frames2=None, env_id=None):
    """
    处理 base frames 序列，添加标注和坐标系（已移除 wrist camera）
    
    Args:
        frames1: base frames 视频帧列表
        frames2: 已弃用，保留以保持向后兼容，但不会被使用
        env_id: 环境ID，用于决定是否显示坐标系（可选）
    
    Returns:
        处理后的帧列表
    """
    # 需要显示坐标系的任务列表
    COORDINATE_AXES_ENVS = ["PatternLock", "RouteStick", "InsertPeg", "SwingXtimes"]
    show_coordinate_axes = env_id in COORDINATE_AXES_ENVS if env_id else False
    if not frames1:
        return []
    
    concatenated_frames = []
    
    for i in range(len(frames1)):
        # 获取当前帧
        frame1 = frames1[i] if i < len(frames1) else frames1[-1]
        
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
            continue
        
        # 获取帧的宽度和高度
        actual_h, actual_w1 = frame1.shape[:2]
        
        # 确定左侧和右侧边框宽度
        left_border_width = 0
        right_border_width = 0
        if show_coordinate_axes:
            if env_id == "RouteStick":
                left_border_width = 150  # RouteStick 任务的左侧边框宽度（用于坐标系）
                right_border_width = 240  # RouteStick 任务的右侧边框宽度（用于旋转方向示意图）
            else:
                left_border_width = 150  # 其他任务的左侧边框宽度
        
        if show_coordinate_axes:
            # 添加左侧黑色边框用于绘制坐标系
            left_border = np.zeros((actual_h, left_border_width, 3), dtype=np.uint8)
            
            # 拼接（包含左侧边框）
            concatenated_frame = np.concatenate([left_border, frame1], axis=1)
            
            # 如果是 RouteStick 任务，添加右侧黑色边框
            if env_id == "RouteStick" and right_border_width > 0:
                right_border = np.zeros((actual_h, right_border_width, 3), dtype=np.uint8)
                concatenated_frame = np.concatenate([concatenated_frame, right_border], axis=1)
            
            # 转换为PIL图像以便在黑色边框区域绘制坐标系
            concatenated_pil = Image.fromarray(concatenated_frame)
            
            # 在左侧黑色边框绘制 base camera 坐标系（旋转180度）
            left_border_pil = Image.new('RGB', (left_border_width, actual_h), (0, 0, 0))
            left_border_pil = draw_coordinate_axes(left_border_pil, position="left", rotate_180=True, env_id=env_id)
            
            # 将坐标系绘制到拼接后的图像上
            concatenated_pil.paste(left_border_pil, (0, 0))
            
            # 如果是 RouteStick 任务，在右侧黑色边框绘制旋转方向示意图
            if env_id == "RouteStick" and right_border_width > 0:
                right_border_pil = Image.new('RGB', (right_border_width, actual_h), (0, 0, 0))
                right_border_pil = draw_coordinate_axes(right_border_pil, position="right", rotate_180=False, env_id=env_id)
                
                # 计算右侧边框在拼接图像中的位置
                right_border_x = left_border_width + actual_w1
                concatenated_pil.paste(right_border_pil, (right_border_x, 0))
            
            # 转换回numpy数组
            concatenated_frame = np.array(concatenated_pil)
        else:
            # 不显示坐标系，直接使用原帧
            concatenated_frame = frame1
        
        concatenated_frames.append(concatenated_frame)
    
    return concatenated_frames


def draw_coordinate_axes(img, position="right", rotate_180=False, env_id=None):
    """
    在图片外的黑色区域绘制坐标系，标注 forward/backward/left/right
    
    Args:
        img: PIL Image 或 numpy array
        position: "left" 或 "right"，指定在左侧还是右侧绘制
        rotate_180: 如果为 True，将坐标系顺时针旋转180度（用于 base camera）
        env_id: 环境ID，用于决定是否绘制特殊示意图（如 RouteStick 的旋转方向）
    
    Returns:
        PIL Image with coordinate axes drawn
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    img = img.copy()
    draw = ImageDraw.Draw(img)
    
    # 获取图片尺寸
    width, height = img.size
    
    # 如果是 RouteStick 任务且位置在右侧，绘制旋转方向示意图
    if env_id == "RouteStick" and position == "right":
        # 绘制四个半圆箭头示意图（2×2 网格）
        # 示意图位置：在图像右侧，2×2 网格布局
        illustration_width = 220  # 示意图区域宽度（已弃用，保留以保持兼容性）
        
        # 尝试加载字体
        try:
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            try:
                small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
            except:
                small_font = ImageFont.load_default()
        
        line_color = (255, 255, 255)  # 白色
        semicircle_radius = 18  # 半圆半径
        arrow_size = 6  # 箭头大小
        grid_spacing_x = 35  # 水平格子间距（增加以分开左右两列）
        grid_spacing_y = 20  # 垂直格子间距（保持原值）
        line_width = 2  # 线宽
        
        # 计算网格布局
        # 网格总宽度：2个半圆 + 1个间距 = 2*18*2 + 35 = 107px
        grid_width = semicircle_radius * 2 + grid_spacing_x
        grid_height = semicircle_radius * 2 + grid_spacing_y
        
        # 网格中心位置（位于右侧黑色区域的中心）
        # width 为右侧黑色区域的宽度（240），将网格中心放在区域中心
        grid_center_x = width // 2
        grid_center_y = height // 2
        
        # 计算四个格子的中心位置
        # 左列中心 x
        left_col_x = grid_center_x - grid_spacing_x // 2 - semicircle_radius
        # 右列中心 x
        right_col_x = grid_center_x + grid_spacing_x // 2 + semicircle_radius
        # 上行（clockwise）中心 y
        cw_row_y = grid_center_y - grid_spacing_y // 2 - semicircle_radius
        # 下行（counterclockwise）中心 y
        ccw_row_y = grid_center_y + grid_spacing_y // 2 + semicircle_radius
        
        # 1. 绘制 left clockwise（左上格）：左半圆，右→左（顺时针），箭头在左端朝上
        lcw_center_x = left_col_x
        lcw_center_y = cw_row_y
        # 左半圆：从 0°（右）到 180°（左），向左凸出
        # 圆心在格子中心右侧，使半圆向左凸出
        lcw_circle_center_x = lcw_center_x + semicircle_radius
        lcw_circle_center_y = lcw_center_y
        arc_points_lcw = []
        for angle_deg in range(0, 181, 5):  # 从 0° 到 180°（顺时针）
            angle_rad = math.radians(angle_deg)
            x = lcw_circle_center_x + semicircle_radius * math.cos(angle_rad)
            y = lcw_circle_center_y + semicircle_radius * math.sin(angle_rad)
            arc_points_lcw.append((x, y))
        # 绘制圆弧线段
        for i in range(len(arc_points_lcw) - 1):
            draw.line([arc_points_lcw[i], arc_points_lcw[i+1]], fill=line_color, width=line_width)
        # 箭头在左端（180°位置），朝上
        arrow_x_lcw = lcw_circle_center_x + semicircle_radius * math.cos(math.radians(180))
        arrow_y_lcw = lcw_circle_center_y + semicircle_radius * math.sin(math.radians(180))
        # 箭头朝上（在180°位置，切线方向向上）
        draw.polygon(
            [(arrow_x_lcw, arrow_y_lcw - arrow_size),  # 尖端向上
             (arrow_x_lcw - arrow_size, arrow_y_lcw + arrow_size // 2),
             (arrow_x_lcw + arrow_size, arrow_y_lcw + arrow_size // 2)],
            fill=line_color
        )
        # 添加标签 "L CW"
        lcw_text = "L CW"
        lcw_bbox = draw.textbbox((0, 0), lcw_text, font=small_font)
        lcw_text_width = lcw_bbox[2] - lcw_bbox[0]
        lcw_text_height = lcw_bbox[3] - lcw_bbox[1]
        lcw_text_x = lcw_center_x - lcw_text_width // 2
        lcw_text_y = lcw_center_y + semicircle_radius + 5
        draw.rectangle(
            [(lcw_text_x - 2, lcw_text_y - 2),
             (lcw_text_x + lcw_text_width + 2, lcw_text_y + lcw_text_height + 2)],
            fill=(0, 0, 0)
        )
        draw.text((lcw_text_x, lcw_text_y), lcw_text, fill=line_color, font=small_font)
        
        # 2. 绘制 right clockwise（右上格）：右半圆，左→右（顺时针），箭头在右端朝上
        rcw_center_x = right_col_x
        rcw_center_y = cw_row_y
        # 右半圆：从 180°（左）到 0°（右），向右凸出
        # 圆心在格子中心左侧，使半圆向右凸出
        rcw_circle_center_x = rcw_center_x - semicircle_radius
        rcw_circle_center_y = rcw_center_y
        arc_points_rcw = []
        for angle_deg in range(180, -1, -5):  # 从 180° 到 0°（顺时针）
            angle_rad = math.radians(angle_deg)
            x = rcw_circle_center_x + semicircle_radius * math.cos(angle_rad)
            y = rcw_circle_center_y + semicircle_radius * math.sin(angle_rad)
            arc_points_rcw.append((x, y))
        # 绘制圆弧线段
        for i in range(len(arc_points_rcw) - 1):
            draw.line([arc_points_rcw[i], arc_points_rcw[i+1]], fill=line_color, width=line_width)
        # 箭头在右端（0°位置），朝上
        arrow_x_rcw = rcw_circle_center_x + semicircle_radius * math.cos(math.radians(0))
        arrow_y_rcw = rcw_circle_center_y + semicircle_radius * math.sin(math.radians(0))
        # 箭头朝上（在0°位置，切线方向向上）
        draw.polygon(
            [(arrow_x_rcw, arrow_y_rcw - arrow_size),  # 尖端向上
             (arrow_x_rcw - arrow_size, arrow_y_rcw + arrow_size // 2),
             (arrow_x_rcw + arrow_size, arrow_y_rcw + arrow_size // 2)],
            fill=line_color
        )
        # 添加标签 "R CW"
        rcw_text = "R CW"
        rcw_bbox = draw.textbbox((0, 0), rcw_text, font=small_font)
        rcw_text_width = rcw_bbox[2] - rcw_bbox[0]
        rcw_text_height = rcw_bbox[3] - rcw_bbox[1]
        rcw_text_x = rcw_center_x - rcw_text_width // 2
        rcw_text_y = rcw_center_y + semicircle_radius + 5
        draw.rectangle(
            [(rcw_text_x - 2, rcw_text_y - 2),
             (rcw_text_x + rcw_text_width + 2, rcw_text_y + rcw_text_height + 2)],
            fill=(0, 0, 0)
        )
        draw.text((rcw_text_x, rcw_text_y), rcw_text, fill=line_color, font=small_font)
        
        # 3. 绘制 left counterclockwise（左下格）：左半圆，左→右（逆时针），箭头在右端朝下
        lccw_center_x = left_col_x
        lccw_center_y = ccw_row_y
        # 左半圆：从 180°（左）到 0°（右），向左凸出
        # 圆心在格子中心右侧，使半圆向左凸出
        lccw_circle_center_x = lccw_center_x + semicircle_radius
        lccw_circle_center_y = lccw_center_y
        arc_points_lccw = []
        for angle_deg in range(180, -1, -5):  # 从 180° 到 0°（逆时针）
            angle_rad = math.radians(angle_deg)
            x = lccw_circle_center_x + semicircle_radius * math.cos(angle_rad)
            y = lccw_circle_center_y + semicircle_radius * math.sin(angle_rad)
            arc_points_lccw.append((x, y))
        # 绘制圆弧线段
        for i in range(len(arc_points_lccw) - 1):
            draw.line([arc_points_lccw[i], arc_points_lccw[i+1]], fill=line_color, width=line_width)
        # 箭头在右端（0°位置），朝下
        arrow_x_lccw = lccw_circle_center_x + semicircle_radius * math.cos(math.radians(0))
        arrow_y_lccw = lccw_circle_center_y + semicircle_radius * math.sin(math.radians(0))
        # 箭头朝下（在0°位置，切线方向向下）
        draw.polygon(
            [(arrow_x_lccw, arrow_y_lccw + arrow_size),  # 尖端向下
             (arrow_x_lccw - arrow_size, arrow_y_lccw - arrow_size // 2),
             (arrow_x_lccw + arrow_size, arrow_y_lccw - arrow_size // 2)],
            fill=line_color
        )
        # 添加标签 "L CCW"
        lccw_text = "L CCW"
        lccw_bbox = draw.textbbox((0, 0), lccw_text, font=small_font)
        lccw_text_width = lccw_bbox[2] - lccw_bbox[0]
        lccw_text_height = lccw_bbox[3] - lccw_bbox[1]
        lccw_text_x = lccw_center_x - lccw_text_width // 2
        lccw_text_y = lccw_center_y + semicircle_radius + 5
        draw.rectangle(
            [(lccw_text_x - 2, lccw_text_y - 2),
             (lccw_text_x + lccw_text_width + 2, lccw_text_y + lccw_text_height + 2)],
            fill=(0, 0, 0)
        )
        draw.text((lccw_text_x, lccw_text_y), lccw_text, fill=line_color, font=small_font)
        
        # 4. 绘制 right counterclockwise（右下格）：右半圆，右→左（逆时针），箭头在左端朝下
        rccw_center_x = right_col_x
        rccw_center_y = ccw_row_y
        # 右半圆：从 0°（右）到 180°（左），向右凸出
        # 圆心在格子中心左侧，使半圆向右凸出
        rccw_circle_center_x = rccw_center_x - semicircle_radius
        rccw_circle_center_y = rccw_center_y
        arc_points_rccw = []
        for angle_deg in range(0, 181, 5):  # 从 0° 到 180°（逆时针）
            angle_rad = math.radians(angle_deg)
            x = rccw_circle_center_x + semicircle_radius * math.cos(angle_rad)
            y = rccw_circle_center_y + semicircle_radius * math.sin(angle_rad)
            arc_points_rccw.append((x, y))
        # 绘制圆弧线段
        for i in range(len(arc_points_rccw) - 1):
            draw.line([arc_points_rccw[i], arc_points_rccw[i+1]], fill=line_color, width=line_width)
        # 箭头在左端（180°位置），朝下
        arrow_x_rccw = rccw_circle_center_x + semicircle_radius * math.cos(math.radians(180))
        arrow_y_rccw = rccw_circle_center_y + semicircle_radius * math.sin(math.radians(180))
        # 箭头朝下（在180°位置，切线方向向下）
        draw.polygon(
            [(arrow_x_rccw, arrow_y_rccw + arrow_size),  # 尖端向下
             (arrow_x_rccw - arrow_size, arrow_y_rccw - arrow_size // 2),
             (arrow_x_rccw + arrow_size, arrow_y_rccw - arrow_size // 2)],
            fill=line_color
        )
        # 添加标签 "R CCW"
        rccw_text = "R CCW"
        rccw_bbox = draw.textbbox((0, 0), rccw_text, font=small_font)
        rccw_text_width = rccw_bbox[2] - rccw_bbox[0]
        rccw_text_height = rccw_bbox[3] - rccw_bbox[1]
        rccw_text_x = rccw_center_x - rccw_text_width // 2
        rccw_text_y = rccw_center_y + semicircle_radius + 5
        draw.rectangle(
            [(rccw_text_x - 2, rccw_text_y - 2),
             (rccw_text_x + rccw_text_width + 2, rccw_text_y + rccw_text_height + 2)],
            fill=(0, 0, 0)
        )
        draw.text((rccw_text_x, rccw_text_y), rccw_text, fill=line_color, font=small_font)
        
        # 右侧区域只绘制旋转示意图，不绘制坐标系，直接返回
        return img
    
    # 坐标系位置（在黑色边框内）
    axis_size = 60  # 坐标系大小
    
    # 如果是 RouteStick 任务且位置在左侧，坐标系位于左侧区域
    # 与右侧旋转示意图对称，距离左边缘75像素
    if env_id == "RouteStick" and position == "left":
        # 坐标系中心位于左侧区域，距离左边缘75像素
        center_x_pos = 75
        origin_x = center_x_pos - axis_size // 2
    else:
        # 坐标轴中心位于边框宽度的中心
        origin_x = width // 2 - axis_size // 2
    origin_y = height // 2 - axis_size // 2
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            font = ImageFont.load_default()
    
    # 绘制坐标轴（十字形）
    axis_length = axis_size - 20
    center_x = origin_x + axis_size // 2
    center_y = origin_y + axis_size // 2
    
    # 绘制坐标轴线条（白色，带半透明效果）
    line_color = (255, 255, 255)  # 白色
    line_width = 2
    
    # 根据是否旋转180度，调整方向
    if rotate_180:
        # 旋转180度：forward变成backward，left变成right
        # 水平轴（left-right，但方向相反）
        draw.line(
            [(center_x - axis_length // 2, center_y), 
             (center_x + axis_length // 2, center_y)],
            fill=line_color, width=line_width
        )
        
        # 垂直轴（forward-backward，但方向相反）
        draw.line(
            [(center_x, center_y - axis_length // 2), 
             (center_x, center_y + axis_length // 2)],
            fill=line_color, width=line_width
        )
        
        # 绘制箭头（旋转180度后的方向）
        arrow_size = 5
        # Forward 箭头（现在在下方，原来是上方）
        draw.polygon(
            [(center_x, center_y + axis_length // 2),
             (center_x - arrow_size, center_y + axis_length // 2 - arrow_size),
             (center_x + arrow_size, center_y + axis_length // 2 - arrow_size)],
            fill=line_color
        )
        # Backward 箭头（现在在上方，原来是下方）
        draw.polygon(
            [(center_x, center_y - axis_length // 2),
             (center_x - arrow_size, center_y - axis_length // 2 + arrow_size),
             (center_x + arrow_size, center_y - axis_length // 2 + arrow_size)],
            fill=line_color
        )
        # Right 箭头（现在在左侧，原来是右侧）
        draw.polygon(
            [(center_x - axis_length // 2, center_y),
             (center_x - axis_length // 2 + arrow_size, center_y - arrow_size),
             (center_x - axis_length // 2 + arrow_size, center_y + arrow_size)],
            fill=line_color
        )
        # Left 箭头（现在在右侧，原来是左侧）
        draw.polygon(
            [(center_x + axis_length // 2, center_y),
             (center_x + axis_length // 2 - arrow_size, center_y - arrow_size),
             (center_x + axis_length // 2 - arrow_size, center_y + arrow_size)],
            fill=line_color
        )
        
        # 添加文字标签（旋转180度后的位置）
        text_color = (255, 255, 255)  # 白色文字
        
        # Forward (现在在下方)
        forward_text = "forward"
        forward_bbox = draw.textbbox((0, 0), forward_text, font=font)
        forward_width = forward_bbox[2] - forward_bbox[0]
        forward_x = center_x - forward_width // 2
        forward_y = center_y + axis_length // 2 + 5
        draw.rectangle(
            [(forward_x - 2, forward_y - 2), 
             (forward_x + forward_width + 2, forward_y + (forward_bbox[3] - forward_bbox[1]) + 2)],
            fill=(0, 0, 0)
        )
        draw.text((forward_x, forward_y), forward_text, fill=text_color, font=font)
        
        # Backward (现在在上方)
        backward_text = "backward"
        backward_bbox = draw.textbbox((0, 0), backward_text, font=font)
        backward_width = backward_bbox[2] - backward_bbox[0]
        backward_x = center_x - backward_width // 2
        backward_y = center_y - axis_length // 2 - 20
        draw.rectangle(
            [(backward_x - 2, backward_y - 2), 
             (backward_x + backward_width + 2, backward_y + (backward_bbox[3] - backward_bbox[1]) + 2)],
            fill=(0, 0, 0)
        )
        draw.text((backward_x, backward_y), backward_text, fill=text_color, font=font)
        
        # Right (现在在左侧)
        right_text = "right"
        right_bbox = draw.textbbox((0, 0), right_text, font=font)
        right_width = right_bbox[2] - right_bbox[0]
        right_x = center_x - axis_length // 2 - right_width - 5
        right_y = center_y - (right_bbox[3] - right_bbox[1]) // 2
        draw.rectangle(
            [(right_x - 2, right_y - 2), 
             (right_x + right_width + 2, right_y + (right_bbox[3] - right_bbox[1]) + 2)],
            fill=(0, 0, 0)
        )
        draw.text((right_x, right_y), right_text, fill=text_color, font=font)
        
        # Left (现在在右侧)
        left_text = "left"
        left_bbox = draw.textbbox((0, 0), left_text, font=font)
        left_width = left_bbox[2] - left_bbox[0]
        left_x = center_x + axis_length // 2 + 5
        left_y = center_y - (left_bbox[3] - left_bbox[1]) // 2
        draw.rectangle(
            [(left_x - 2, left_y - 2), 
             (left_x + left_width + 2, left_y + (left_bbox[3] - left_bbox[1]) + 2)],
            fill=(0, 0, 0)
        )
        draw.text((left_x, left_y), left_text, fill=text_color, font=font)
    else:
        # 正常方向（不旋转）
        # 水平轴（left-right）
        draw.line(
            [(center_x - axis_length // 2, center_y), 
             (center_x + axis_length // 2, center_y)],
            fill=line_color, width=line_width
        )
        
        # 垂直轴（forward-backward）
        draw.line(
            [(center_x, center_y - axis_length // 2), 
             (center_x, center_y + axis_length // 2)],
            fill=line_color, width=line_width
        )
        
        # 绘制箭头（在轴的两端）
        arrow_size = 5
        # Forward (上) 箭头
        draw.polygon(
            [(center_x, center_y - axis_length // 2),
             (center_x - arrow_size, center_y - axis_length // 2 + arrow_size),
             (center_x + arrow_size, center_y - axis_length // 2 + arrow_size)],
            fill=line_color
        )
        # Backward (下) 箭头
        draw.polygon(
            [(center_x, center_y + axis_length // 2),
             (center_x - arrow_size, center_y + axis_length // 2 - arrow_size),
             (center_x + arrow_size, center_y + axis_length // 2 - arrow_size)],
            fill=line_color
        )
        # Right (右) 箭头
        draw.polygon(
            [(center_x + axis_length // 2, center_y),
             (center_x + axis_length // 2 - arrow_size, center_y - arrow_size),
             (center_x + axis_length // 2 - arrow_size, center_y + arrow_size)],
            fill=line_color
        )
        # Left (左) 箭头
        draw.polygon(
            [(center_x - axis_length // 2, center_y),
             (center_x - axis_length // 2 + arrow_size, center_y - arrow_size),
             (center_x - axis_length // 2 + arrow_size, center_y + arrow_size)],
            fill=line_color
        )
        
        # 添加文字标签
        text_color = (255, 255, 255)  # 白色文字
        
        # Forward (上)
        forward_text = "forward"
        forward_bbox = draw.textbbox((0, 0), forward_text, font=font)
        forward_width = forward_bbox[2] - forward_bbox[0]
        forward_x = center_x - forward_width // 2
        forward_y = center_y - axis_length // 2 - 20
        draw.rectangle(
            [(forward_x - 2, forward_y - 2), 
             (forward_x + forward_width + 2, forward_y + (forward_bbox[3] - forward_bbox[1]) + 2)],
            fill=(0, 0, 0)
        )
        draw.text((forward_x, forward_y), forward_text, fill=text_color, font=font)
        
        # Backward (下)
        backward_text = "backward"
        backward_bbox = draw.textbbox((0, 0), backward_text, font=font)
        backward_width = backward_bbox[2] - backward_bbox[0]
        backward_x = center_x - backward_width // 2
        backward_y = center_y + axis_length // 2 + 5
        draw.rectangle(
            [(backward_x - 2, backward_y - 2), 
             (backward_x + backward_width + 2, backward_y + (backward_bbox[3] - backward_bbox[1]) + 2)],
            fill=(0, 0, 0)
        )
        draw.text((backward_x, backward_y), backward_text, fill=text_color, font=font)
        
        # Right (右)
        right_text = "right"
        right_bbox = draw.textbbox((0, 0), right_text, font=font)
        right_width = right_bbox[2] - right_bbox[0]
        right_x = center_x + axis_length // 2 + 5
        right_y = center_y - (right_bbox[3] - right_bbox[1]) // 2
        draw.rectangle(
            [(right_x - 2, right_y - 2), 
             (right_x + right_width + 2, right_y + (right_bbox[3] - right_bbox[1]) + 2)],
            fill=(0, 0, 0)
        )
        draw.text((right_x, right_y), right_text, fill=text_color, font=font)
        
        # Left (左)
        left_text = "left"
        left_bbox = draw.textbbox((0, 0), left_text, font=font)
        left_width = left_bbox[2] - left_bbox[0]
        left_x = center_x - axis_length // 2 - left_width - 5
        left_y = center_y - (left_bbox[3] - left_bbox[1]) // 2
        draw.rectangle(
            [(left_x - 2, left_y - 2), 
             (left_x + left_width + 2, left_y + (left_bbox[3] - left_bbox[1]) + 2)],
            fill=(0, 0, 0)
        )
        draw.text((left_x, left_y), left_text, fill=text_color, font=font)
    
    return img


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
