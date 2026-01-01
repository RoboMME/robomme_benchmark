"""
自动同步视频打标工具
支持从HDF5文件中读取数据，进行关键帧标记和坐标标注
"""

import gradio as gr
import cv2
import json
import numpy as np
import os
import sys
import subprocess
import tempfile
import shutil
import h5py

# 全局变量：存储每个数据集实际使用的端口
dataset_actual_ports = {}


def convert_video_with_imageio(frames, fps, output_path):
    """
    使用imageio将帧数组转换为浏览器兼容的MP4视频
    
    参数:
        frames: 帧数组列表，每个元素是一个numpy数组（BGR或RGB格式）
        fps: 视频帧率
        output_path: 输出视频文件路径
        
    返回:
        bool: 转换成功返回True，失败返回False
    """
    try:
        import imageio
    except ImportError:
        # imageio库未安装，无法进行转换
        return False
    
    try:
        # 准备帧数据：确保是uint8格式的RGB numpy数组
        processed_frames = []
        for f in frames:
            # 如果不是numpy数组，转换为numpy数组
            if not isinstance(f, np.ndarray):
                f = np.array(f)
            
            # 确保数据类型为uint8（0-255范围）
            if f.dtype != np.uint8:
                # 如果数据在0-1范围内，需要乘以255
                if np.max(f) <= 1.0:
                    f = (f * 255).astype(np.uint8)
                else:
                    # 否则直接裁剪到0-255范围并转换类型
                    f = f.clip(0, 255).astype(np.uint8)
            
            # 如果是3通道BGR图像，转换为RGB（浏览器需要RGB格式）
            if len(f.shape) == 3 and f.shape[2] == 3:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            # 如果是灰度图，转换为3通道RGB
            elif len(f.shape) == 2:
                f = np.stack([f] * 3, axis=-1)
            
            processed_frames.append(f)
        
        # 使用imageio.mimwrite创建视频，使用H.264编码以确保浏览器兼容性
        imageio.mimwrite(output_path, processed_frames, fps=fps, quality=8, macro_block_size=None, codec='libx264')
        
        return True
    except Exception as e:
        # 转换过程中发生异常，返回False
        print(f"视频转换失败: {e}")
        return False

class AutoSyncTagger:
    """
    自动同步视频打标器类
    支持从HDF5文件中读取数据，进行关键帧标记和坐标标注
    """
    
    def __init__(self, video_path, output_json_path, output_video_path, target_width=960):
        """
        初始化打标器
        
        参数:
            video_path: HDF5文件路径（.hdf5 或 .h5）
            output_json_path: 输出JSON标注文件路径
            output_video_path: 输出带标注的视频文件路径
            target_width: 目标显示宽度，超过此宽度会自动缩放（默认960像素）
        """
        # 保存路径参数
        self.video_path = video_path
        self.output_json_path = output_json_path
        self.output_video_path = output_video_path
        self.target_width = target_width
        
        # 关键帧数据字典：{帧索引: {"option": 选项值, "point": [x, y]}}
        self.keyframes = {}
        # 帧缓存列表，存储所有预加载的帧数据
        self.frames_cache = []
        
        # 验证文件类型
        if not (video_path.endswith('.hdf5') or video_path.endswith('.h5')):
            raise ValueError(f"只支持HDF5文件格式，当前文件: {video_path}")
        
        # ========== HDF5文件处理模式 ==========
        # 从 HDF5 文件加载帧数据
        frames, fps = self._load_from_hdf5(video_path, fps=30)
        self.frames_cache = frames
        self.fps = fps
        self.total_frames = len(frames)
        
        # 获取第一帧的尺寸用于缩放计算
        if len(frames) > 0:
            self.orig_h, self.orig_w = frames[0].shape[:2]
        else:
            raise ValueError("HDF5 文件中没有帧数据")
        
        # 计算缩放比例：如果原始宽度超过目标宽度，则进行缩放
        self.scale_ratio = 1.0
        if self.orig_w > self.target_width:
            self.scale_ratio = self.target_width / self.orig_w
        self.resize_dims = (int(self.orig_w * self.scale_ratio), int(self.orig_h * self.scale_ratio))
        
        # 应用缩放（如果需要）
        if self.scale_ratio != 1.0:
            print(f"应用缩放: {self.orig_w}x{self.orig_h} -> {self.resize_dims[0]}x{self.resize_dims[1]}")
            scaled_frames = []
            for frame in self.frames_cache:
                # 使用INTER_AREA插值方法进行高质量缩放
                scaled_frame = cv2.resize(frame, self.resize_dims, interpolation=cv2.INTER_AREA)
                scaled_frames.append(scaled_frame)
            self.frames_cache = scaled_frames
        
        # 为 UI 播放器创建临时视频文件（使用 imageio 创建浏览器兼容的视频）
        # 临时视频保存在系统临时目录，以符合 Gradio 的安全要求
        temp_dir = tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True)
        fd, temp_video_path = tempfile.mkstemp(suffix='.mp4', dir=temp_dir)
        os.close(fd)
        
        if convert_video_with_imageio(self.frames_cache, self.fps, temp_video_path):
            # 更新 video_path 为临时视频文件路径，用于 UI 播放器
            self.video_path = temp_video_path
        else:
            # 如果创建失败，删除临时文件
            os.remove(temp_video_path)

    def _load_from_hdf5(self, hdf5_path, fps=30):
        """
        从 HDF5 文件读取 right_shoulder 相机数据并转换为 BGR 格式的帧数组
        
        参数:
            hdf5_path: HDF5 文件路径
            fps: 帧率，默认 30
            
        返回:
            frames: BGR 格式的帧数组列表
            fps: 帧率
            
        异常:
            ValueError: 当HDF5文件结构不符合预期时抛出
        """
        try:
            with h5py.File(hdf5_path, 'r') as f:
                # 检查并读取 observation 组
                if 'observation' not in f:
                    raise ValueError(f"HDF5 文件中未找到 'observation' 组")
                
                obs = f['observation']
                # 检查并读取 camera_rgb 组
                if 'camera_rgb' not in obs:
                    raise ValueError(f"HDF5 文件中未找到 'observation/camera_rgb' 组")
                
                camera_rgb = obs['camera_rgb']
                # 检查并读取 right_shoulder 相机数据
                if 'right_shoulder' not in camera_rgb:
                    raise ValueError(f"HDF5 文件中未找到 'observation/camera_rgb/right_shoulder' 数据")
                
                right_shoulder_data = camera_rgb['right_shoulder']
                # 数据形状应该是 (N, 256, 256, 3)，RGB 格式，uint8
                num_frames, height, width, channels = right_shoulder_data.shape
                
                print(f"正在从 HDF5 加载 {num_frames} 帧 (分辨率: {width}x{height})...")
                
                frames = []
                for i in range(num_frames):
                    # 读取一帧 (RGB 格式)，确保是 numpy 数组
                    frame_rgb = np.array(right_shoulder_data[i])
                    
                    # 转换为 BGR 格式（OpenCV 使用 BGR）
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    
                    frames.append(frame_bgr)
                    
                    # 每50帧显示一次进度
                    if (i + 1) % 50 == 0:
                        sys.stdout.write(f"\r加载: {i + 1}/{num_frames}")
                        sys.stdout.flush()
                
                print("\nHDF5 加载完成！")
                
                return frames, fps
                
        except Exception as e:
            raise ValueError(f"从 HDF5 文件读取数据失败: {e}")

    def _draw_overlay(self, frame_bgr, frame_idx):
        """
        在帧上绘制标记（用于UI显示）
        
        参数:
            frame_bgr: BGR格式的帧图像（会被原地修改）
            frame_idx: 帧索引
            
        返回:
            frame_bgr: 绘制标记后的帧图像
        """
        if frame_idx in self.keyframes:
            # 获取该关键帧的数据
            data = self.keyframes[frame_idx]
            h, w, _ = frame_bgr.shape
            
            # 根据缩放比例计算线条粗细
            thick = max(2, int(5 * self.scale_ratio))
            
            # 绘制红色边框（关键帧标记）
            cv2.rectangle(frame_bgr, (0, 0), (w, h), (0, 0, 255), thick * 2)
            
            # 显示关键帧信息文本（帧索引和选项）
            info_text = f"KEY: {frame_idx} | {data.get('option', '')}"
            cv2.putText(frame_bgr, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0 * self.scale_ratio, (0, 0, 255), thick)
            
            # 如果有关键点坐标，绘制红色圆点标记
            if "point" in data:
                # 将原始坐标转换为显示坐标（考虑缩放）
                orig_pt = data["point"]
                disp_x = int(orig_pt[0] * self.scale_ratio)
                disp_y = int(orig_pt[1] * self.scale_ratio)
                # 绘制实心红色圆点
                cv2.circle(frame_bgr, (disp_x, disp_y), 10, (0, 0, 255), -1)
                
        return frame_bgr

    def get_frame_image(self, frame_idx):
        """
        获取指定帧的图像（带标记），用于UI显示
        
        参数:
            frame_idx: 帧索引
            
        返回:
            RGB格式的numpy数组图像
        """
        idx = int(frame_idx)
        # 确保索引在有效范围内
        if idx < 0:
            idx = 0
        if idx >= len(self.frames_cache):
            idx = len(self.frames_cache) - 1
        
        # 复制帧数据（避免修改原始缓存）
        frame_bgr = self.frames_cache[idx].copy()
        # 绘制标记
        frame_bgr = self._draw_overlay(frame_bgr, idx)
        # 转换为RGB格式（Gradio需要RGB格式）
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _format_time(self, seconds):
        """
        时间格式化辅助函数：将秒数转换为 "MM:SS.ss" 格式
        
        参数:
            seconds: 秒数（浮点数）
            
        返回:
            格式化后的时间字符串，格式为 "MM:SS.ss"
        """
        m, s = divmod(seconds, 60)
        return f"{int(m):02d}:{s:05.2f}"

    def get_timeline_text(self):
        """
        生成时间线文本，显示所有已标记的关键帧信息
        
        返回:
            格式化的时间线文本字符串，包含帧索引、时间、选项和坐标信息
        """
        sorted_keys = sorted(list(self.keyframes.keys()))
        if not sorted_keys:
            return "暂无标记"
        
        lines = []
        # 添加标题行
        lines.append(f"总计标记: {len(sorted_keys)} 帧")
        lines.append("-" * 30)
        
        # 遍历所有关键帧，生成详细信息
        for k in sorted_keys:
            # 计算时间戳
            t = k / self.fps
            time_str = self._format_time(t)
            
            # 获取关键帧数据
            data = self.keyframes[k]
            opt = data.get("option", "N/A")
            pt = data.get("point", [])
            
            # 格式化坐标字符串
            pt_str = f"({pt[0]}, {pt[1]})" if pt else "No Point"
            
            # 添加一行信息：帧索引 | 时间 | 选项 | 坐标
            lines.append(f"Frame {k:<5} | Time {time_str} | {opt} | {pt_str}")
        
        return "\n".join(lines)

    def toggle_keyframe(self, frame_idx, option_val, point_str):
        """
        标记或更新关键帧
        
        如果该帧已存在标记，则更新；如果不存在，则添加新标记。
        只有 "pickup block" 和 "pickup cup" 会保存坐标点，其他选项不保存坐标。
        
        参数:
            frame_idx: 帧索引
            option_val: 选项值（动作类型）
            point_str: 坐标字符串，格式为 "X: 123, Y: 456"（显示分辨率下的坐标）
            
        返回:
            tuple: (当前帧图像, 状态消息, 时间线文本)
        """
        idx = int(frame_idx)
        msg = ""
        
        # 检查选项：只有 "pickup block" 和 "pickup cup" 允许保存坐标
        allowed_options = ["pickup block", "pickup cup"]
        
        # 解析坐标字符串（仅当选项允许时）
        point = []
        if option_val in allowed_options:
            try:
                # 格式预期 "X: 123, Y: 456"
                parts = point_str.replace("X:", "").replace("Y:", "").split(",")
                if len(parts) == 2:
                    # 获取显示分辨率下的坐标
                    disp_x = int(parts[0].strip())
                    disp_y = int(parts[1].strip())
                    # 转换回原始分辨率下的坐标（考虑缩放）
                    orig_x = int(disp_x / self.scale_ratio)
                    orig_y = int(disp_y / self.scale_ratio)
                    point = [orig_x, orig_y]
            except:
                # 解析失败时，point保持为空列表
                pass
        # 如果选项不在允许列表中，point保持为空列表（不保存坐标）
            
        if idx in self.keyframes:
            # 如果该帧已存在标记，则更新（覆盖旧数据）
            if option_val in allowed_options:
                self.keyframes[idx] = {"option": option_val, "point": point}
            else:
                # 不允许保存坐标的选项，只保存option，不保存point
                self.keyframes[idx] = {"option": option_val}
            msg = f"帧 {idx} 已更新: {option_val}"
        else:
            # 如果该帧不存在标记，则添加新标记
            if option_val in allowed_options:
                self.keyframes[idx] = {"option": option_val, "point": point}
            else:
                # 不允许保存坐标的选项，只保存option，不保存point
                self.keyframes[idx] = {"option": option_val}
            msg = f"帧 {idx} 已标记: {option_val}"
        
        # 返回：当前图像，状态消息，更新后的时间线文本
        return self.get_frame_image(idx), msg, self.get_timeline_text()

    def remove_keyframe(self, frame_idx):
        """
        删除指定帧的关键帧标记
        
        参数:
            frame_idx: 帧索引
            
        返回:
            tuple: (当前帧图像, 状态消息, 时间线文本)
        """
        idx = int(frame_idx)
        if idx in self.keyframes:
            # 删除该帧的标记
            del self.keyframes[idx]
            msg = f"帧 {idx} 已移除"
        else:
            msg = f"帧 {idx} 未标记"
        
        return self.get_frame_image(idx), msg, self.get_timeline_text()

    def _get_keyframe_data_for_frame(self, frame_idx):
        """
        查找当前帧应该显示的关键帧数据（区间逻辑）
        
        使用区间逻辑：找到小于等于当前帧的最大关键帧（最近的前一个关键帧）。
        从该关键帧开始，到下一个关键帧之前，都显示这个关键帧的数据。
        如果当前帧之前没有关键帧，则不显示任何数据。
        
        参数:
            frame_idx: 当前帧索引
            
        返回:
            tuple: (关键帧数据dict, 关键帧索引) 或 (None, None)
                   关键帧数据格式: {"option": 选项值, "point": [x, y]}
        """
        sorted_keys = sorted(list(self.keyframes.keys()))
        if not sorted_keys:
            return None, None
        
        # 找到小于等于 frame_idx 的最大关键帧（最近的前一个关键帧）
        target_kf = None
        for k in sorted_keys:
            if k <= frame_idx:
                target_kf = k
            else:
                # 已经超过当前帧，停止查找
                break
        
        # 如果找到了关键帧，返回其数据和索引；否则返回None
        if target_kf is not None:
            return self.keyframes[target_kf], target_kf
        else:
            return None, None

    def _wrap_text(self, text, font_scale, font_thickness, max_width):
        """
        将文本自动换行，返回多行文本列表
        
        参数:
            text: 要换行的文本
            font_scale: 字体大小
            font_thickness: 字体粗细
            max_width: 最大宽度（像素）
            
        返回:
            list: 换行后的文本列表
        """
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            # 测试添加这个词后的宽度
            test_line = current_line + (" " if current_line else "") + word
            (test_width, _), _ = cv2.getTextSize(
                test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            if test_width <= max_width:
                # 可以添加这个词
                current_line = test_line
            else:
                # 当前行已满，开始新行
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        # 添加最后一行
        if current_line:
            lines.append(current_line)
        
        return lines if lines else [text]
    
    def _draw_overlay_for_export(self, frame_bgr, frame_idx):
        """
        在帧上绘制标记（用于导出视频）
        
        与_draw_overlay不同，此函数使用区间逻辑，会在视频外的上下黑框内显示字幕，
        并绘制关键点和边框。字幕支持自动换行，字体缩小一倍。
        
        参数:
            frame_bgr: BGR格式的帧图像（会被原地修改）
            frame_idx: 帧索引
            
        返回:
            frame_bgr: 绘制标记后的帧图像（包含上下黑框）
        """
        h, w, _ = frame_bgr.shape
        thick = max(2, int(5 * self.scale_ratio))
        
        # ========== 1. 获取关键帧数据（使用区间逻辑）==========
        # 获取该帧应该显示的关键帧数据（使用区间逻辑：找到小于等于当前帧的最大关键帧）
        data, target_keyframe_idx = self._get_keyframe_data_for_frame(frame_idx)
        
        # 字幕字体大小（缩小一倍：从0.75改为0.375）
        font_scale = 0.375 * self.scale_ratio
        font_thickness = max(1, int(0.75 * self.scale_ratio))
        
        # 计算黑框高度（根据字体大小动态计算）
        (_, text_height), baseline = cv2.getTextSize(
            "Test", cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        line_height = text_height + baseline + 5  # 行高（包含间距）
        black_bar_height = int(line_height * 3)  # 上下黑框各3行高度
        
        # 创建新的画布（包含上下黑框）
        new_h = h + black_bar_height * 2
        new_frame = np.zeros((new_h, w, 3), dtype=np.uint8)
        
        # 将原视频放在中间（不包含左上角信息）
        new_frame[black_bar_height:black_bar_height + h, :] = frame_bgr
        
        # ========== 2. 在上黑框显示帧索引和时间戳 ==========
        # 计算当前帧的时间戳
        time_seconds = frame_idx / self.fps
        time_str = self._format_time(time_seconds)
        frame_info_text = f"Frame: {frame_idx} | Time: {time_str}"
        
        # 计算文本位置（上黑框居中）
        (info_text_width, info_text_height), info_baseline = cv2.getTextSize(
            frame_info_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        info_x = (w - info_text_width) // 2
        info_y = black_bar_height // 2 + info_text_height // 2
        
        # 在上黑框绘制帧索引和时间戳（白色）
        cv2.putText(
            new_frame,
            frame_info_text,
            (info_x, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
        
        # ========== 3. 在下黑框显示关键帧信息（三行）==========
        if data and target_keyframe_idx is not None:
            # 获取选项和坐标信息
            opt_text = data.get("option", "")
            point = data.get("point", [])
            
            # 查找用于显示的关键帧编号
            sorted_keys = sorted(list(self.keyframes.keys()))
            # 获取找到的关键帧编号
            keyframe_num = sorted_keys.index(target_keyframe_idx) + 1
            total_keyframes = len(sorted_keys)
            
            # 构建三行文本
            line1_text = f"Keyframe {keyframe_num}/{total_keyframes}"
            line2_text = opt_text if opt_text else "N/A"
            line3_text = f"({point[0]}, {point[1]})" if point else "No Point"
            
            # 计算下黑框的起始Y坐标
            bottom_bar_start_y = black_bar_height + h
            # 计算三行文本的总高度
            total_bottom_height = line_height * 3
            # 计算起始Y位置（下黑框居中）
            start_y = bottom_bar_start_y + (black_bar_height - total_bottom_height) // 2 + text_height
            
            # 绘制三行文本（居中）
            lines = [line1_text, line2_text, line3_text]
            for i, line in enumerate(lines):
                (line_width, _), _ = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                line_x = (w - line_width) // 2
                line_y = start_y + i * line_height
                
                # 绘制文本（白色）
                cv2.putText(
                    new_frame,
                    line,
                    (line_x, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    font_thickness
                )
        
        # ========== 4. 如果当前帧是关键帧，绘制红色边框（在视频区域）==========
        if frame_idx in self.keyframes:
            cv2.rectangle(
                new_frame,
                (0, black_bar_height),
                (w - 1, black_bar_height + h - 1),
                (0, 0, 255),
                thick * 2
            )
        
        # ========== 5. 如果有关键点坐标，绘制红色圆点标记（在视频区域）==========
        if data and target_keyframe_idx is not None:
            point = data.get("point", [])
            if point:
                # 将原始坐标转换为显示坐标（考虑缩放）
                disp_x = int(point[0] * self.scale_ratio)
                disp_y = int(point[1] * self.scale_ratio) + black_bar_height  # 加上上黑框高度
                # 绘制实心红色圆点（半径7像素）
                cv2.circle(new_frame, (disp_x, disp_y), 7, (0, 0, 255), -1)
                # 绘制圆点外圈（更明显的标记）
                cv2.circle(new_frame, (disp_x, disp_y), 10, (0, 0, 255), 2)
        
        return new_frame

    def save_all_data(self, progress=gr.Progress()):
        """
        保存所有数据：JSON标注文件和带标记的视频文件
        
        只有 "pickup block" 和 "pickup cup" 会保存坐标，其他选项不保存坐标。
        
        参数:
            progress: Gradio进度条对象，用于显示导出进度
            
        返回:
            str: 保存结果消息
        """
        # ========== 1. 保存 JSON 标注文件 ==========
        sorted_keys = sorted(list(self.keyframes.keys()))
        # 构建导出格式的数据（将帧索引转换为字符串作为键）
        export_keyframes = {}
        allowed_options = ["pickup block", "pickup cup"]
        for k in sorted_keys:
            frame_data = self.keyframes[k]
            option_val = frame_data.get("option", "")
            # 只有允许的选项才保存坐标
            if option_val in allowed_options:
                # 保存完整数据（包括坐标）
                export_keyframes[str(k)] = frame_data
            else:
                # 不保存坐标，只保存option
                export_keyframes[str(k)] = {"option": option_val}
            
        json_data = {"keyframes": export_keyframes}
        try:
            with open(self.output_json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
            json_msg = "✅ JSON 已保存"
        except Exception as e:
            return f"❌ JSON 保存失败: {e}"

        # ========== 2. 导出带标记的 MP4 视频 ==========
        if not self.frames_cache:
            return "❌ 视频缓存为空，无法导出"

        try:
            total = len(self.frames_cache)
            # 处理所有帧，添加标记
            processed_frames_bgr = []
            for i, frame in enumerate(self.frames_cache):
                # 复制帧（避免修改原始缓存）
                frame_copy = frame.copy()
                # 使用导出绘制逻辑添加标记（frame_copy是BGR格式）
                frame_copy = self._draw_overlay_for_export(frame_copy, i)
                processed_frames_bgr.append(frame_copy)
                
                # 每10帧更新一次进度
                if i % 10 == 0:
                    progress((i + 1) / total, desc="正在渲染导出视频...")
            
            # 使用imageio保存视频，使用H.264编码以确保兼容性
            # convert_video_with_imageio会自动将BGR转换为RGB
            if convert_video_with_imageio(processed_frames_bgr, self.fps, self.output_video_path):
                video_msg = "✅ MP4 已导出"
            else:
                # 如果imageio不可用，回退到OpenCV的H.264编码
                try:
                    h, w, _ = self.frames_cache[0].shape
                    # 尝试使用H.264编码器（更兼容的格式）
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264编码
                    out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (w, h))
                    
                    if not out.isOpened():
                        # 如果avc1不可用，尝试XVID
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (w, h))
                    
                    if out.isOpened():
                        # processed_frames_bgr已经是BGR格式，直接写入
                        for frame_bgr in processed_frames_bgr:
                            out.write(frame_bgr)
                        out.release()
                        video_msg = "✅ MP4 已导出（使用OpenCV）"
                    else:
                        return f"{json_msg}\n❌ 视频导出失败: 无法创建视频写入器"
                except Exception as e2:
                    return f"{json_msg}\n❌ 视频导出失败: {e2}"
        except Exception as e:
            return f"{json_msg}\n❌ 视频导出失败: {e}"

        # 返回保存结果摘要
        return f"{json_msg}\n{video_msg}\n关键帧数: {len(sorted_keys)}\n视频路径: {self.output_video_path}"

    def sync_from_player(self, time_float):
        """
        从视频播放器同步帧索引
        
        当用户在视频播放器中暂停或视频结束时，根据当前播放时间同步到对应的帧。
        
        参数:
            time_float: 视频播放时间（秒），浮点数
            
        返回:
            tuple: (目标帧索引, 目标帧图像)
        """
        try:
            if time_float is None:
                # 如果时间为None，返回第一帧
                return 0, self.get_frame_image(0)
            
            # 将时间转换为帧索引
            seconds = float(time_float)
            target_frame = int(seconds * self.fps)
            # 确保帧索引在有效范围内
            target_frame = min(max(target_frame, 0), self.total_frames - 1)
            
            return target_frame, self.get_frame_image(target_frame)
        except Exception as e:
            print(f"同步错误: {e}")
            # 发生错误时返回第一帧
            return 0, self.get_frame_image(0)

    def handle_image_click(self, evt: gr.SelectData, frame_idx, option_val):
        """
        处理图片点击事件：更新坐标显示，并在图上绘制点击点
        
        当用户在预览图像上点击时，获取点击坐标并显示在图像上。
        只有 "pickup block" 和 "pickup cup" 允许点击。
        
        参数:
            evt: Gradio的SelectData事件对象，包含点击位置信息
            frame_idx: 当前显示的帧索引
            option_val: 当前选中的选项值
            
        返回:
            tuple: (坐标字符串, 带点击标记的图像)
        """
        # 检查选项：只有 "pickup block" 和 "pickup cup" 允许点击
        allowed_options = ["pickup block", "pickup cup"]
        if option_val not in allowed_options:
            # 不允许点击，返回当前坐标和图像（不更新）
            base_img_rgb = self.get_frame_image(frame_idx)
            # 获取当前坐标显示（如果有的话）
            if frame_idx in self.keyframes:
                point = self.keyframes[frame_idx].get("point", [])
                if point:
                    coord_str = f"X: {int(point[0] * self.scale_ratio)}, Y: {int(point[1] * self.scale_ratio)}"
                else:
                    coord_str = "X: 0, Y: 0"
            else:
                coord_str = "X: 0, Y: 0"
            return coord_str, base_img_rgb
        
        # Gradio的evt.index是 [x, y] 格式（列，行）
        x, y = evt.index[0], evt.index[1]
        coord_str = f"X: {x}, Y: {y}"
        
        # 获取基础图像（已包含之前的标记）
        base_img_rgb = self.get_frame_image(frame_idx)
        # 转换为BGR格式以便OpenCV处理
        img_bgr = cv2.cvtColor(base_img_rgb, cv2.COLOR_RGB2BGR)
        
        # 绘制黄色圆点表示当前点击位置（带黑色边框）
        cv2.circle(img_bgr, (x, y), 8, (0, 255, 255), -1)  # 黄色实心圆
        cv2.circle(img_bgr, (x, y), 8, (0, 0, 0), 1)  # 黑色边框
        
        # 转换回RGB格式返回
        out_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return coord_str, out_img

    def create_ui(self, options=None):
        """
        创建Gradio用户界面
        
        参数:
            options: 选项列表，如果为None则使用默认选项
        
        返回:
            gr.Blocks: Gradio界面对象
        """
        # 如果没有提供选项，使用默认选项
        if options is None:
            options = ["Option 1", "Option 2", "Option 3"]
        
        # JavaScript代码：从视频元素获取当前播放时间
        js_get_time = "(x) => document.querySelector('video').currentTime"

        with gr.Blocks(title="Auto-Sync Tagger", theme=gr.themes.Soft()) as demo:
            gr.Markdown(f"### ⚡ 自动同步打标工具 (带时间线)")
            
            # 隐藏的时间值，用于JavaScript传递视频播放时间
            hidden_time = gr.Number(value=0.0, visible=False)
            
            with gr.Row():
                # ========== 左列：视频播放器和时间线 ==========
                with gr.Column(scale=5):
                    gr.Markdown("**1. 宏观浏览** (点击暂停 -> 右侧同步)")
                    
                    # 视频播放器组件（使用format参数确保浏览器兼容性）
                    native_video = gr.Video(
                        value=self.video_path, 
                        label="原生播放器", 
                        interactive=False,
                        format="mp4"  # 明确指定MP4格式
                    )
                    
                    # 分隔线
                    gr.Markdown("---")
                    
                    # 时间线显示区域（显示所有已标记的关键帧信息）
                    timeline_box = gr.TextArea(
                        label="📋 已标记关键帧时间线", 
                        value="暂无标记", 
                        lines=15, 
                        interactive=False,
                        text_align="left"
                    )

                # ========== 右列：帧编辑器和控制面板 ==========
                with gr.Column(scale=4):
                    gr.Markdown("**2. 微观编辑器**")
                    
                    # 精确帧预览图像（interactive=True允许点击获取坐标）
                    editor_img = gr.Image(label="精确帧预览 (点击选择坐标)", interactive=True)
                    
                    # 选项和坐标显示行
                    with gr.Row():
                        # 动作类型选择器
                        option_selector = gr.Radio(
                            choices=options, 
                            value=options[0] if options else "Option 1", 
                            label="选择动作类型"
                        )
                        # 当前选中坐标显示（显示分辨率下的坐标）
                        coord_display = gr.Textbox(
                            label="当前选中坐标 (显示分辨率)", 
                            value="X: 0, Y: 0", 
                            interactive=False
                        )

                    # 帧索引滑块
                    slider = gr.Slider(minimum=0, maximum=self.total_frames-1, step=1, label="帧索引", value=0)
                    
                    # 上一帧/下一帧按钮
                    with gr.Row():
                        btn_prev = gr.Button("⬅️ 上一帧")
                        btn_next = gr.Button("➡️ 下一帧")
                    
                    # 标记和删除按钮
                    with gr.Row():
                        btn_mark = gr.Button("🔴 标记 / 更新", variant="primary")
                        btn_remove = gr.Button("🗑️ 删除当前帧", variant="secondary")
                    
                    # 状态显示和保存按钮
                    status = gr.Textbox(label="最新操作状态", lines=1)
                    btn_save_all = gr.Button("💾 保存 JSON + MP4", variant="stop")

            # ========== 事件绑定 ==========
            
            # 视频播放器同步：当视频暂停或结束时，同步到对应帧
            native_video.pause(fn=self.sync_from_player, inputs=[hidden_time], outputs=[slider, editor_img], js=js_get_time)
            native_video.end(fn=self.sync_from_player, inputs=[hidden_time], outputs=[slider, editor_img], js=js_get_time)

            # 滑块改变：当用户拖动滑块时，更新显示的帧
            slider.change(fn=self.get_frame_image, inputs=[slider], outputs=[editor_img])

            # 上一帧/下一帧按钮：步进式导航
            def step(curr, delta):
                """计算新的帧索引（带边界检查）"""
                val = min(max(curr + delta, 0), self.total_frames - 1)
                return val
            
            btn_prev.click(fn=lambda x: step(x, -1), inputs=[slider], outputs=[slider])
            btn_next.click(fn=lambda x: step(x, 1), inputs=[slider], outputs=[slider])

            # 图片点击：当用户在预览图像上点击时，获取坐标并显示
            # 只有 "pickup block" 和 "pickup cup" 允许点击
            editor_img.select(
                fn=self.handle_image_click,
                inputs=[slider, option_selector],
                outputs=[coord_display, editor_img]
            )

            # 标记按钮：标记或更新当前帧的关键帧信息
            btn_mark.click(
                fn=self.toggle_keyframe, 
                inputs=[slider, option_selector, coord_display], 
                outputs=[editor_img, status, timeline_box] 
            )

            # 删除按钮：删除当前帧的关键帧标记
            btn_remove.click(
                fn=self.remove_keyframe,
                inputs=[slider],
                outputs=[editor_img, status, timeline_box]
            )

            # 保存按钮：保存JSON标注文件和带标记的视频
            btn_save_all.click(fn=self.save_all_data, inputs=[], outputs=[status])
            
            # 初始化：页面加载时显示第一帧
            demo.load(fn=lambda: self.get_frame_image(0), outputs=[editor_img])

        return demo

def extract_ep_number(filename):
    """
    从文件名中提取ep后的数字，用于排序
    
    参数:
        filename: 文件名
        
    返回:
        int: ep后的数字，如果找不到则返回0
    """
    import re
    # 查找 -ep 或 ep 后的数字
    match = re.search(r'[_-]ep(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # 如果找不到，返回0（会排在前面）
    return 0

def get_hdf5_files(directory):
    """
    获取目录中所有的 hdf5 和 h5 文件
    
    参数:
        directory: 目录路径
        
    返回:
        list: 按ep后数字排序后的文件路径列表
    """
    hdf5_files = []
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith('.hdf5') or filename.endswith('.h5'):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath):
                    hdf5_files.append(filepath)
    # 按ep后的数字排序
    hdf5_files.sort(key=lambda x: extract_ep_number(os.path.basename(x)))
    return hdf5_files

def is_file_annotated(filename, output_dir):
    """
    检查文件是否已标注（输出目录中是否存在对应的JSON文件）
    
    参数:
        filename: HDF5文件名
        output_dir: 输出目录
        
    返回:
        bool: 已标注返回True，否则返回False
    """
    if not filename:
        return False
    base_name = os.path.splitext(filename)[0]
    json_path = os.path.join(output_dir, f"{base_name}_keyframes.json")
    return os.path.exists(json_path)

def get_file_list_data(file_choices, output_dir):
    """
    生成文件列表数据，用于 Dataframe 显示
    
    参数:
        file_choices: 文件名列表
        output_dir: 输出目录
        
    返回:
        list: [[状态图标, 文件名], ...]
    """
    data = []
    for f in file_choices:
        annotated = is_file_annotated(f, output_dir)
        status_icon = "✅" if annotated else "⬜"
        data.append([status_icon, f])
    return data

def get_progress_info(file_choices, output_dir):
    """
    生成进度信息文本
    
    参数:
        file_choices: 文件名列表
        output_dir: 输出目录
        
    返回:
        str: 进度信息 Markdown 文本
    """
    total = len(file_choices)
    if total == 0:
        return "### 进度: 0/0 (0%)"
    
    annotated_count = sum(1 for f in file_choices if is_file_annotated(f, output_dir))
    percent = (annotated_count / total) * 100
    return f"### 进度: {annotated_count}/{total} ({percent:.1f}%)"

def create_batch_annotation_ui(input_dir, output_dir, options=None):
    """
    创建批量标注界面，支持逐个标注多个 HDF5 文件
    
    参数:
        input_dir: 输入目录路径（包含 HDF5 文件）
        output_dir: 输出目录路径（保存标注结果）
        options: 选项列表，如果为None则使用默认选项
        
    返回:
        gr.Blocks: Gradio界面对象
    """
    # 如果没有提供选项，使用默认选项
    if options is None:
        options = ["Option 1", "Option 2", "Option 3"]
    # 获取所有 HDF5 文件
    hdf5_files = get_hdf5_files(input_dir)
    
    if not hdf5_files:
        print(f"错误: 在 {input_dir} 中未找到任何 HDF5 文件")
        return None
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建文件选择列表（显示文件名）
    file_choices = [os.path.basename(f) for f in hdf5_files]
    
    # 全局变量存储当前打标器
    current_tagger = [None]
    current_file_idx = [0]
    
    def load_file_by_name(filename):
        """
        根据文件名加载文件并创建打标器
        
        参数:
            filename: 文件名
            
        返回:
            tuple: (文件信息, 视频路径, 第一帧图像, 滑块值, 滑块最大值, 时间线文本)
        """
        if not filename or filename not in file_choices:
            return "请选择文件", None, None, 0, "暂无标记"
        
        # 找到文件路径
        file_idx = file_choices.index(filename)
        hdf5_path = hdf5_files[file_idx]
        
        # 生成输出文件路径
        base_name = os.path.splitext(filename)[0]
        output_json = os.path.join(output_dir, f"{base_name}_keyframes.json")
        output_video = os.path.join(output_dir, f"{base_name}_labeled.mp4")
        
        try:
            # 创建打标器实例
            tagger = AutoSyncTagger(hdf5_path, output_json, output_video)
            current_tagger[0] = tagger
            current_file_idx[0] = file_idx
            
            # 获取第一帧图像
            first_frame = tagger.get_frame_image(0)
            
            info_text = f"文件: {filename}\n进度: {file_idx + 1}/{len(hdf5_files)}\n总帧数: {tagger.total_frames}\nFPS: {tagger.fps:.2f}"
            
            # 返回滑块的值（设置为 0，因为旧版 Gradio 不支持动态更新 maximum）
            slider_value = 0
            
            return info_text, tagger.video_path, first_frame, slider_value, tagger.get_timeline_text()
        except Exception as e:
            error_msg = f"加载文件失败: {filename}\n错误: {str(e)}"
            print(error_msg)
            current_tagger[0] = None
            return error_msg, None, None, 0, "加载失败"
    
    # JavaScript代码：从视频元素获取当前播放时间
    js_get_time = "(x) => document.querySelector('video').currentTime"
    
    # 从输入目录提取数据集名称
    dataset_name = os.path.basename(input_dir.rstrip('/'))
    
    with gr.Blocks(title=f"批量 HDF5 标注工具 - {dataset_name}") as demo:
        gr.Markdown(f"### 📁 批量 HDF5 文件标注工具 - **{dataset_name}**")
        gr.Markdown(f"**输入目录**: `{input_dir}`  |  **输出目录**: `{output_dir}`")
        
        # 隐藏组件
        hidden_time = gr.Number(value=0.0, visible=False)
        # 保持 file_dropdown 存在但隐藏，用于作为中间状态同步
        file_dropdown = gr.Dropdown(choices=file_choices, value=file_choices[0] if file_choices else None, visible=False)

        with gr.Row():
            # ========== 左侧边栏：文件列表 ==========
            with gr.Column(scale=1, min_width=300):
                progress_bar = gr.Markdown(value=get_progress_info(file_choices, output_dir))
                
                # 文件列表 Dataframe
                file_list = gr.Dataframe(
                    headers=["状态", "文件名"],
                    value=get_file_list_data(file_choices, output_dir),
                    interactive=False, # 防止用户编辑表格
                    wrap=True
                )
                
                # 刷新列表按钮
                btn_refresh_list = gr.Button("🔄 刷新列表")
                
            # ========== 右侧主内容区域 ==========
            with gr.Column(scale=5):
                # 文件导航和信息
                with gr.Row():
                    file_info_display = gr.Textbox(label="当前文件信息", lines=3, interactive=False)
                
                with gr.Row():
                    # ========== 左列：视频播放器和时间线 ==========
                    with gr.Column(scale=5):
                        gr.Markdown("**1. 宏观浏览** (点击暂停 -> 右侧同步)")
                        
                        # 视频播放器组件
                        native_video = gr.Video(
                            value=None, 
                            label="原生播放器", 
                            interactive=False,
                            format="mp4"
                        )
                        
                        gr.Markdown("---")
                        
                        # 时间线显示区域
                        timeline_box = gr.TextArea(
                            label="📋 已标记关键帧时间线", 
                            value="暂无标记", 
                            lines=15, 
                            interactive=False,
                            text_align="left"
                        )

                    # ========== 右列：帧编辑器和控制面板 ==========
                    with gr.Column(scale=4):
                        gr.Markdown("**2. 微观编辑器**")
                        
                        # 精确帧预览图像
                        editor_img = gr.Image(label="精确帧预览 (点击选择坐标)", interactive=True)
                        
                        # 选项和坐标显示行
                        with gr.Row():
                            option_selector = gr.Radio(
                                choices=options, 
                                value=options[0] if options else "Option 1", 
                                label="选择动作类型"
                            )
                            coord_display = gr.Textbox(
                                label="当前选中坐标 (显示分辨率)", 
                                value="X: 0, Y: 0", 
                                interactive=False
                            )

                        # 帧索引滑块
                        slider = gr.Slider(minimum=0, maximum=10000, step=1, label="帧索引", value=0)
                        
                        # 上一帧/下一帧按钮
                        with gr.Row():
                            btn_prev = gr.Button("⬅️ 上一帧")
                            btn_next = gr.Button("➡️ 下一帧")
                        
                        # 标记和删除按钮
                        with gr.Row():
                            btn_mark = gr.Button("🔴 标记 / 更新", variant="primary")
                            btn_remove = gr.Button("🗑️ 删除当前帧", variant="secondary")
                        
                        # 状态显示和保存按钮
                        status = gr.Textbox(label="最新操作状态", lines=1)
                        btn_save_current = gr.Button("💾 保存当前文件", variant="stop")

        # ========== 事件绑定 ==========
        
        def update_frame_image(frame_idx):
            """更新帧图像"""
            if current_tagger[0] is None:
                return None
            # 确保帧索引在有效范围内
            frame_idx = int(frame_idx)
            max_frame = current_tagger[0].total_frames - 1
            frame_idx = min(max(frame_idx, 0), max_frame)
            return current_tagger[0].get_frame_image(frame_idx)
        
        def toggle_keyframe_wrapper(frame_idx, option_val, point_str):
            """标记关键帧的包装函数"""
            if current_tagger[0] is None:
                return None, "请先选择文件", "无文件"
            return current_tagger[0].toggle_keyframe(frame_idx, option_val, point_str)
        
        def remove_keyframe_wrapper(frame_idx):
            """删除关键帧的包装函数"""
            if current_tagger[0] is None:
                return None, "请先选择文件", "无文件"
            return current_tagger[0].remove_keyframe(frame_idx)
        
        def sync_from_player_wrapper(time_float):
            """从播放器同步的包装函数"""
            if current_tagger[0] is None:
                return 0, None
            return current_tagger[0].sync_from_player(time_float)
        
        def save_current_file_wrapper():
            """保存当前文件的包装函数，并更新列表状态"""
            if current_tagger[0] is None:
                return "请先选择文件", get_file_list_data(file_choices, output_dir), get_progress_info(file_choices, output_dir)
            
            result_msg = current_tagger[0].save_all_data()
            
            # 更新列表数据和进度
            new_list_data = get_file_list_data(file_choices, output_dir)
            new_progress = get_progress_info(file_choices, output_dir)
            
            return result_msg, new_list_data, new_progress
        
        def step_frame(curr, delta):
            """步进帧"""
            if current_tagger[0] is None:
                return 0
            # 确保不超过总帧数
            max_frame = current_tagger[0].total_frames - 1
            val = min(max(curr + delta, 0), max_frame)
            return val
            
        def on_select_file(evt: gr.SelectData):
            """处理文件列表点击"""
            # evt.index 是 [row, col]
            row_idx = evt.index[0]
            if 0 <= row_idx < len(file_choices):
                return file_choices[row_idx]
            return None

        # 刷新列表
        def refresh_file_list():
            return get_file_list_data(file_choices, output_dir), get_progress_info(file_choices, output_dir)

        btn_refresh_list.click(
            fn=refresh_file_list,
            inputs=[],
            outputs=[file_list, progress_bar]
        )

        # 文件列表点击 -> 更新下拉框
        file_list.select(
            fn=on_select_file,
            inputs=[],
            outputs=[file_dropdown]
        )

        # 下拉框改变 -> 加载文件
        file_dropdown.change(
            fn=load_file_by_name,
            inputs=[file_dropdown],
            outputs=[file_info_display, native_video, editor_img, slider, timeline_box]
        )
        
        # 视频播放器同步
        native_video.pause(fn=sync_from_player_wrapper, inputs=[hidden_time], outputs=[slider, editor_img], js=js_get_time)
        native_video.end(fn=sync_from_player_wrapper, inputs=[hidden_time], outputs=[slider, editor_img], js=js_get_time)

        # 滑块改变（添加验证，确保不超过总帧数）
        def slider_change_wrapper(frame_idx):
            """滑块改变包装函数，添加边界检查"""
            if current_tagger[0] is None:
                return None
            frame_idx = int(frame_idx)
            max_frame = current_tagger[0].total_frames - 1
            frame_idx = min(max(frame_idx, 0), max_frame)
            return current_tagger[0].get_frame_image(frame_idx)
        
        slider.change(fn=slider_change_wrapper, inputs=[slider], outputs=[editor_img])

        # 上一帧/下一帧按钮
        btn_prev.click(fn=lambda x: step_frame(x, -1), inputs=[slider], outputs=[slider])
        btn_next.click(fn=lambda x: step_frame(x, 1), inputs=[slider], outputs=[slider])

        # 图片点击
        def handle_click_with_frame(evt: gr.SelectData, frame_idx, option_val):
            """处理图片点击，包含帧索引和选项值"""
            if current_tagger[0] is None:
                return "X: 0, Y: 0", None
            try:
                return current_tagger[0].handle_image_click(evt, int(frame_idx), option_val)
            except Exception as e:
                print(f"处理图片点击错误: {e}")
                return "X: 0, Y: 0", None
        
        editor_img.select(
            fn=handle_click_with_frame,
            inputs=[slider, option_selector],
            outputs=[coord_display, editor_img]
        )

        # 标记按钮
        btn_mark.click(
            fn=toggle_keyframe_wrapper, 
            inputs=[slider, option_selector, coord_display], 
            outputs=[editor_img, status, timeline_box] 
        )

        # 删除按钮
        btn_remove.click(
            fn=remove_keyframe_wrapper,
            inputs=[slider],
            outputs=[editor_img, status, timeline_box]
        )

        # 保存当前文件按钮
        btn_save_current.click(
            fn=save_current_file_wrapper, 
            inputs=[], 
            outputs=[status, file_list, progress_bar]
        )
        
        # 初始化：加载第一个文件
        if file_choices:
            demo.load(
                fn=lambda: load_file_by_name(file_choices[0]),
                inputs=[],
                outputs=[file_info_display, native_video, editor_img, slider, timeline_box]
            )

    return demo

def is_port_available(port, host='0.0.0.0'):
    """
    检查端口是否可用
    
    参数:
        port: 端口号
        host: 主机地址，默认为 '0.0.0.0'
        
    返回:
        bool: 端口可用返回True，否则返回False
    """
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False

def find_free_port(start_port, max_attempts=100):
    """
    从指定端口开始查找可用端口
    
    参数:
        start_port: 起始端口号
        max_attempts: 最大尝试次数
        
    返回:
        int: 可用的端口号，如果找不到则返回None
    """
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    return None

def launch_dataset_ui(dataset_name, input_dir, output_dir, preferred_port, options=None, title_suffix=""):
    """
    为指定数据集启动UI界面
    
    参数:
        dataset_name: 数据集名称
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        preferred_port: 首选端口号（如果被占用会自动查找可用端口）
        options: 选项列表，如果为None则使用默认选项
        title_suffix: 标题后缀
    """
    # 检查首选端口是否可用，如果不可用则查找可用端口
    if is_port_available(preferred_port):
        actual_port = preferred_port
        port_status = "使用首选端口"
    else:
        print(f"⚠️  端口 {preferred_port} 被占用，正在查找可用端口...")
        actual_port = find_free_port(preferred_port + 1)
        if actual_port is None:
            print(f"❌ 无法为 {dataset_name} 找到可用端口（从 {preferred_port + 1} 开始查找）")
            return
        port_status = f"端口 {preferred_port} 被占用，已切换到 {actual_port}"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建批量标注界面
    demo = create_batch_annotation_ui(input_dir, output_dir, options)
    
    if demo is not None:
        # 启动 Web 界面
        print(f"\n{'='*60}")
        print(f"启动 {dataset_name} 数据集标注界面")
        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        print(f"端口: {actual_port} ({port_status})")
        print(f"访问地址: http://0.0.0.0:{actual_port}")
        print(f"{'='*60}\n")
        
        # 将实际使用的端口保存到全局变量，以便主程序可以获取
        global dataset_actual_ports
        dataset_actual_ports[dataset_name] = actual_port
        
        demo.queue().launch(
            server_name="0.0.0.0", 
            server_port=actual_port,
            share=False, 
            theme=gr.themes.Soft()
        )
    else:
        print(f"无法创建 {dataset_name} 界面，请检查输入目录中是否有 HDF5 文件")

if __name__ == "__main__":
    """
    主程序入口
    为四个数据集创建独立的UI界面，每个使用不同的端口
    """
    import threading
    import time
    
    # 数据集配置：名称、输入目录、输出目录、端口号、选项列表
    datasets_config = [
        {
            "name": "DrawPattern",
            "input_dir": "/data/hongzefu/historybench_real_dataset/DrawPattern",
            "output_dir": "/data/hongzefu/historybench_real_dataset/annotation_results/DrawPattern",
            "port": 7860,
            "options": ["move left", "move right", "move forward", "move backward"]
        },
        {
            "name": "PutFruits",
            "input_dir": "/data/hongzefu/historybench_real_dataset/PutFruits",
            "output_dir": "/data/hongzefu/historybench_real_dataset/annotation_results/PutFruits",
            "port": 7861,
            "options": ["pickup fruit", "putdown fruit", "press button"]
        },
        {
            "name": "RepickBlock",
            "input_dir": "/data/hongzefu/historybench_real_dataset/RepickBlock",
            "output_dir": "/data/hongzefu/historybench_real_dataset/annotation_results/RepickBlock",
            "port": 7862,
            "options": ["pickup block", "putdown block"]
        },
        {
            "name": "TrackBlock",
            "input_dir": "/data/hongzefu/historybench_real_dataset/TrackBlock",
            "output_dir": "/data/hongzefu/historybench_real_dataset/annotation_results/TrackBlock",
            "port": 7863,
            "options": ["pickup cup", "putdown cup"]
        }
    ]
    
    # 为每个数据集创建独立的线程启动UI
    threads = []
    for config in datasets_config:
        thread = threading.Thread(
            target=launch_dataset_ui,
            args=(config["name"], config["input_dir"], config["output_dir"], config["port"], config.get("options")),
            daemon=True
        )
        thread.start()
        threads.append(thread)
        # 延迟启动，避免端口冲突
        time.sleep(2)
    
    # 等待所有服务启动（给一些时间让端口分配完成）
    time.sleep(3)
    
    print("\n" + "="*60)
    print("所有数据集界面已启动！")
    print("="*60)
    print("\n访问地址：")
    for config in datasets_config:
        # 获取实际使用的端口
        actual_port = dataset_actual_ports.get(config["name"], config["port"])
        preferred = "" if actual_port == config["port"] else f" (首选 {config['port']} 被占用)"
        print(f"  {config['name']:15s} -> http://0.0.0.0:{actual_port}{preferred}")
    print("\n按 Ctrl+C 停止所有服务\n")
    print("="*60 + "\n")
    
    # 保持主线程运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在关闭所有服务...")