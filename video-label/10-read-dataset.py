"""
自动同步视频打标工具
支持从HDF5文件或视频文件中读取数据，进行关键帧标记和坐标标注
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
    支持从HDF5文件或视频文件中读取数据，进行关键帧标记和坐标标注
    """
    
    def __init__(self, video_path, output_json_path, output_video_path, target_width=960):
        """
        初始化打标器
        
        参数:
            video_path: 输入视频文件路径或HDF5文件路径
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
        
        # 检测输入类型：HDF5 文件还是视频文件
        is_hdf5 = video_path.endswith('.hdf5') or video_path.endswith('.h5')
        
        if is_hdf5:
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
            
            # HDF5 模式不需要视频文件，设置 cap 为 None
            self.cap = None
            
            # 为 UI 播放器创建临时视频文件（使用 imageio 创建浏览器兼容的视频）
            fd, temp_video_path = tempfile.mkstemp(suffix='.mp4', dir=os.path.dirname(video_path) if os.path.dirname(video_path) else None)
            os.close(fd)
            
            if convert_video_with_imageio(self.frames_cache, self.fps, temp_video_path):
                # 更新 video_path 为临时视频文件路径，用于 UI 播放器
                self.video_path = temp_video_path
            else:
                # 如果创建失败，删除临时文件
                os.remove(temp_video_path)
        else:
            # ========== 视频文件处理模式 ==========
            # 使用OpenCV打开视频文件
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise ValueError(f"无法打开视频: {video_path}")
            
            # 获取视频属性
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # 获取视频编码格式（FOURCC）
            fourcc_code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc_code >> 8 * i) & 0xFF) for i in range(4)])
            
            # 获取原始视频尺寸
            self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 计算缩放比例
            self.scale_ratio = 1.0
            if self.orig_w > self.target_width:
                self.scale_ratio = self.target_width / self.orig_w
            self.resize_dims = (int(self.orig_w * self.scale_ratio), int(self.orig_h * self.scale_ratio))
            
            # 检查编码格式兼容性：浏览器通常只支持H.264编码
            browser_compatible_codecs = ['H264', 'avc1', 'X264', 'x264']
            needs_conversion = fourcc_str not in browser_compatible_codecs and fourcc_str != 'FMP4'
            
            # 预加载所有视频帧到内存
            self.preload_video()
            
            # 如果编码不兼容（包括FMP4），使用imageio重新编码为H.264格式
            if fourcc_str == 'FMP4' or needs_conversion:
                # 创建临时转换后的视频文件
                fd, converted_path = tempfile.mkstemp(suffix='.mp4', dir=os.path.dirname(video_path))
                os.close(fd)
                
                # 使用imageio从frames_cache重新编码为浏览器兼容格式
                if convert_video_with_imageio(self.frames_cache, self.fps, converted_path):
                    self.cap.release()
                    self.video_path = converted_path
                    self.cap = cv2.VideoCapture(converted_path)
                else:
                    # 转换失败，删除临时文件
                    os.remove(converted_path)

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

    def preload_video(self):
        """
        从视频文件预加载所有帧到内存（仅用于视频文件模式）
        
        注意:
            HDF5 模式下不需要此操作，因为帧数据已经在初始化时加载
        """
        if self.cap is None:
            # HDF5 模式下不需要此操作
            return
        
        print(f"正在预加载 {self.total_frames} 帧 (FPS: {self.fps:.2f})...")
        cnt = 0
        
        # 逐帧读取视频
        while True:
            ret, frame = self.cap.read()
            if not ret:
                # 读取失败或到达视频末尾，退出循环
                break
            
            # 如果需要缩放，对帧进行缩放处理
            if self.scale_ratio != 1.0:
                frame = cv2.resize(frame, self.resize_dims, interpolation=cv2.INTER_AREA)
            
            # 将处理后的帧添加到缓存
            self.frames_cache.append(frame) 
            cnt += 1
            
            # 每100帧显示一次进度
            if cnt % 100 == 0:
                sys.stdout.write(f"\r加载: {cnt}/{self.total_frames}")
                sys.stdout.flush()
        
        print("\n预加载完成！")
        # 释放视频捕获对象
        self.cap.release()

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
        支持同时保存选项值和坐标点。
        
        参数:
            frame_idx: 帧索引
            option_val: 选项值（动作类型）
            point_str: 坐标字符串，格式为 "X: 123, Y: 456"（显示分辨率下的坐标）
            
        返回:
            tuple: (当前帧图像, 状态消息, 时间线文本)
        """
        idx = int(frame_idx)
        msg = ""
        
        # 解析坐标字符串
        point = []
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
            
        if idx in self.keyframes:
            # 如果该帧已存在标记，则更新（覆盖旧数据）
            self.keyframes[idx] = {"option": option_val, "point": point}
            msg = f"帧 {idx} 已更新: {option_val}"
        else:
            # 如果该帧不存在标记，则添加新标记
            self.keyframes[idx] = {"option": option_val, "point": point}
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
        
        使用区间逻辑：如果当前帧不是关键帧，则使用最近的前一个关键帧的数据。
        如果当前帧之后没有关键帧，则使用最后一个关键帧的数据。
        
        参数:
            frame_idx: 当前帧索引
            
        返回:
            dict: 关键帧数据 {"option": 选项值, "point": [x, y]}，如果没有关键帧则返回None
        """
        sorted_keys = sorted(list(self.keyframes.keys()))
        if not sorted_keys:
            return None
        
        # 找到第一个大于等于 frame_idx 的关键帧
        target_kf = None
        for k in sorted_keys:
            if k >= frame_idx:
                target_kf = k
                break
        
        # 如果没找到（frame_idx > 最后一个关键帧），使用最后一个关键帧
        if target_kf is None:
            target_kf = sorted_keys[-1]
            
        return self.keyframes[target_kf]

    def _draw_overlay_for_export(self, frame_bgr, frame_idx):
        """
        在帧上绘制标记（用于导出视频）
        
        与_draw_overlay不同，此函数使用区间逻辑，会在视频底部显示字幕，
        并绘制关键点和边框。
        
        参数:
            frame_bgr: BGR格式的帧图像（会被原地修改）
            frame_idx: 帧索引
            
        返回:
            frame_bgr: 绘制标记后的帧图像
        """
        # 获取该帧应该显示的关键帧数据（使用区间逻辑）
        data = self._get_keyframe_data_for_frame(frame_idx)
        if data:
            h, w, _ = frame_bgr.shape
            thick = max(2, int(5 * self.scale_ratio))
            
            # 获取选项和坐标信息
            opt_text = data.get("option", "")
            point = data.get("point", [])
            
            # 构建字幕文本：显示选项和坐标
            if point:
                subtitle_text = f"{opt_text} | ({point[0]}, {point[1]})"
            else:
                subtitle_text = opt_text
            
            # 计算字幕字体大小和粗细（根据缩放比例调整）
            font_scale = 0.75 * self.scale_ratio
            font_thickness = max(1, int(1.5 * self.scale_ratio))
            (text_width, text_height), baseline = cv2.getTextSize(
                subtitle_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # 计算字幕位置（底部居中）
            padding = 10
            subtitle_y = h - 30
            subtitle_x = (w - text_width) // 2
            
            # 绘制字幕背景框（黑色背景，用于提高文字可读性）
            cv2.rectangle(
                frame_bgr,
                (subtitle_x - padding, subtitle_y - text_height - padding),
                (subtitle_x + text_width + padding, subtitle_y + baseline + padding),
                (0, 0, 0),
                -1
            )
            
            # 绘制字幕文本（白色）
            cv2.putText(
                frame_bgr, 
                subtitle_text, 
                (subtitle_x, subtitle_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (255, 255, 255), 
                font_thickness
            )
            
            # 如果当前帧是关键帧，绘制红色边框
            if frame_idx in self.keyframes:
                cv2.rectangle(frame_bgr, (0, 0), (w, h), (0, 0, 255), thick * 2)
            
            # 如果有关键点坐标，绘制红色圆点标记
            if point:
                # 将原始坐标转换为显示坐标（考虑缩放）
                disp_x = int(point[0] * self.scale_ratio)
                disp_y = int(point[1] * self.scale_ratio)
                # 绘制实心红色圆点（半径7像素）
                cv2.circle(frame_bgr, (disp_x, disp_y), 7, (0, 0, 255), -1)
                
        return frame_bgr

    def save_all_data(self, progress=gr.Progress()):
        """
        保存所有数据：JSON标注文件和带标记的视频文件
        
        参数:
            progress: Gradio进度条对象，用于显示导出进度
            
        返回:
            str: 保存结果消息
        """
        # ========== 1. 保存 JSON 标注文件 ==========
        sorted_keys = sorted(list(self.keyframes.keys()))
        # 构建导出格式的数据（将帧索引转换为字符串作为键）
        export_keyframes = {}
        for k in sorted_keys:
            export_keyframes[str(k)] = self.keyframes[k]
            
        json_data = {"path": self.video_path, "keyframes": export_keyframes}
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
            # 获取视频尺寸
            h, w, _ = self.frames_cache[0].shape
            # 使用mp4v编码器创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (w, h))
            
            total = len(self.frames_cache)
            # 逐帧处理并写入视频
            for i, frame in enumerate(self.frames_cache):
                # 复制帧（避免修改原始缓存）
                frame_copy = frame.copy()
                # 使用导出绘制逻辑添加标记
                frame_copy = self._draw_overlay_for_export(frame_copy, i)
                # 写入视频
                out.write(frame_copy)
                
                # 每10帧更新一次进度
                if i % 10 == 0:
                    progress((i + 1) / total, desc="正在渲染导出视频...")
            
            # 释放视频写入器
            out.release()
            video_msg = "✅ MP4 已导出"
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

    def handle_image_click(self, evt: gr.SelectData, frame_idx):
        """
        处理图片点击事件：更新坐标显示，并在图上绘制点击点
        
        当用户在预览图像上点击时，获取点击坐标并显示在图像上。
        
        参数:
            evt: Gradio的SelectData事件对象，包含点击位置信息
            frame_idx: 当前显示的帧索引
            
        返回:
            tuple: (坐标字符串, 带点击标记的图像)
        """
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

    def create_ui(self):
        """
        创建Gradio用户界面
        
        返回:
            gr.Blocks: Gradio界面对象
        """
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
                            choices=["Option 1", "Option 2", "Option 3"], 
                            value="Option 1", 
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
            editor_img.select(
                fn=self.handle_image_click,
                inputs=[slider],
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

def get_hdf5_files(directory):
    """
    获取目录中所有的 hdf5 和 h5 文件
    
    参数:
        directory: 目录路径
        
    返回:
        list: 排序后的文件路径列表
    """
    hdf5_files = []
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith('.hdf5') or filename.endswith('.h5'):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath):
                    hdf5_files.append(filepath)
    # 按文件名排序
    hdf5_files.sort()
    return hdf5_files

def create_batch_annotation_ui(input_dir, output_dir):
    """
    创建批量标注界面，支持逐个标注多个 HDF5 文件
    
    参数:
        input_dir: 输入目录路径（包含 HDF5 文件）
        output_dir: 输出目录路径（保存标注结果）
        
    返回:
        gr.Blocks: Gradio界面对象
    """
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
    
    def get_next_file_name(current_filename):
        """获取下一个文件名"""
        if not current_filename or current_filename not in file_choices:
            return file_choices[0] if file_choices else None
        current_idx = file_choices.index(current_filename)
        next_idx = min(current_idx + 1, len(file_choices) - 1)
        return file_choices[next_idx]
    
    def get_prev_file_name(current_filename):
        """获取上一个文件名"""
        if not current_filename or current_filename not in file_choices:
            return file_choices[0] if file_choices else None
        current_idx = file_choices.index(current_filename)
        prev_idx = max(current_idx - 1, 0)
        return file_choices[prev_idx]
    
    def save_and_next_file(current_filename):
        """保存当前文件并切换到下一个文件"""
        if current_tagger[0] is None:
            return "没有可保存的文件", current_filename, "无文件", None, None, 0, "无文件"
        
        # 保存当前文件的标注
        try:
            save_msg = current_tagger[0].save_all_data()
        except Exception as e:
            save_msg = f"保存失败: {str(e)}"
        
        # 切换到下一个文件
        next_filename = get_next_file_name(current_filename)
        if next_filename == current_filename:
            # 保持当前文件不变，只更新保存状态
            info, video_path, frame, slider_val, timeline = load_file_by_name(current_filename)
            return f"{save_msg}\n(已是最后一个文件)", current_filename, info, video_path, frame, slider_val, timeline
        
        # 加载下一个文件
        info, video_path, frame, slider_val, timeline = load_file_by_name(next_filename)
        return f"{save_msg}\n已切换到下一个文件", next_filename, info, video_path, frame, slider_val, timeline
    
    # JavaScript代码：从视频元素获取当前播放时间
    js_get_time = "(x) => document.querySelector('video').currentTime"
    
    with gr.Blocks(title="批量 HDF5 标注工具") as demo:
        gr.Markdown(f"### 📁 批量 HDF5 文件标注工具")
        gr.Markdown(f"**输入目录**: `{input_dir}`  |  **输出目录**: `{output_dir}`  |  **文件总数**: {len(hdf5_files)}")
        
        # 文件选择区域
        with gr.Row():
            file_dropdown = gr.Dropdown(
                choices=file_choices,
                value=file_choices[0] if file_choices else None,
                label="选择要标注的文件",
                interactive=True
            )
            file_info_display = gr.Textbox(label="文件信息", lines=3, interactive=False)
            with gr.Column():
                btn_prev_file = gr.Button("⬅️ 上一个文件", variant="secondary")
                btn_next_file = gr.Button("➡️ 下一个文件", variant="secondary")
                btn_save_next = gr.Button("💾 保存并下一个", variant="stop")
        
        save_status = gr.Textbox(label="保存状态", lines=2, interactive=False, visible=False)
        
        # 隐藏的时间值，用于JavaScript传递视频播放时间
        hidden_time = gr.Number(value=0.0, visible=False)
        
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
                        choices=["Option 1", "Option 2", "Option 3"], 
                        value="Option 1", 
                        label="选择动作类型"
                    )
                    coord_display = gr.Textbox(
                        label="当前选中坐标 (显示分辨率)", 
                        value="X: 0, Y: 0", 
                        interactive=False
                    )

                # 帧索引滑块（设置一个足够大的最大值，因为旧版 Gradio 不支持动态更新 maximum）
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
            """保存当前文件的包装函数"""
            if current_tagger[0] is None:
                return "请先选择文件"
            return current_tagger[0].save_all_data()
        
        def step_frame(curr, delta):
            """步进帧"""
            if current_tagger[0] is None:
                return 0
            # 确保不超过总帧数
            max_frame = current_tagger[0].total_frames - 1
            val = min(max(curr + delta, 0), max_frame)
            return val
        
        # 文件选择事件
        file_dropdown.change(
            fn=load_file_by_name,
            inputs=[file_dropdown],
            outputs=[file_info_display, native_video, editor_img, slider, timeline_box]
        )
        
        # 上一个/下一个文件按钮
        btn_prev_file.click(
            fn=lambda f: (get_prev_file_name(f),) + load_file_by_name(get_prev_file_name(f)),
            inputs=[file_dropdown],
            outputs=[file_dropdown, file_info_display, native_video, editor_img, slider, timeline_box]
        )
        
        btn_next_file.click(
            fn=lambda f: (get_next_file_name(f),) + load_file_by_name(get_next_file_name(f)),
            inputs=[file_dropdown],
            outputs=[file_dropdown, file_info_display, native_video, editor_img, slider, timeline_box]
        )
        
        # 保存并下一个文件
        btn_save_next.click(
            fn=save_and_next_file,
            inputs=[file_dropdown],
            outputs=[save_status, file_dropdown, file_info_display, native_video, editor_img, slider, timeline_box]
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
        def handle_click_with_frame(evt: gr.SelectData, frame_idx):
            """处理图片点击，包含帧索引"""
            if current_tagger[0] is None:
                return "X: 0, Y: 0", None
            try:
                return current_tagger[0].handle_image_click(evt, int(frame_idx))
            except Exception as e:
                print(f"处理图片点击错误: {e}")
                return "X: 0, Y: 0", None
        
        editor_img.select(
            fn=handle_click_with_frame,
            inputs=[slider],
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
        btn_save_current.click(fn=save_current_file_wrapper, inputs=[], outputs=[status])
        
        # 初始化：加载第一个文件
        if file_choices:
            demo.load(
                fn=lambda: load_file_by_name(file_choices[0]),
                inputs=[],
                outputs=[file_info_display, native_video, editor_img, slider, timeline_box]
            )

    return demo

if __name__ == "__main__":
    """
    主程序入口
    从 DrawPattern 目录读取所有 HDF5 文件，创建批量标注界面
    """
    # 输入目录：包含 HDF5 文件的目录
    input_dir = "/data/hongzefu/historybench_real_dataset/DrawPattern"
    
    # 输出目录：保存标注结果的目录
    output_dir = "/data/hongzefu/historybench_real_dataset/annotation_results/DrawPattern"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建批量标注界面
    demo = create_batch_annotation_ui(input_dir, output_dir)
    
    if demo is not None:
        # 启动 Web 界面
        # server_name="0.0.0.0" 允许从任何网络接口访问
        # share=False 不创建公共链接
        demo.queue().launch(server_name="0.0.0.0", share=False, theme=gr.themes.Soft())
    else:
        print("无法创建界面，请检查输入目录中是否有 HDF5 文件")