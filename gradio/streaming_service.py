"""
流媒体服务模块
处理MJPEG流、帧队列和后台监控线程
"""
import queue
import threading
import time
import numpy as np
import cv2
from fastapi.responses import StreamingResponse
from state_manager import (
    get_session, 
    get_frame_queue_info, 
    FRAME_QUEUES
)
from image_utils import concatenate_frames_horizontally

# --- Streaming Configuration ---
STREAMING_MONITOR_INTERVAL = 0.1  # 监控线程检查间隔（秒）


class FrameQueueManager:
    """帧队列管理器"""
    
    @staticmethod
    def init_queue(uid, pre_base_count, pre_wrist_count):
        """初始化队列并启动监控"""
        if uid not in FRAME_QUEUES:
            FRAME_QUEUES[uid] = {
                "frame_queue": queue.Queue(),
                "last_base_count": pre_base_count,
                "last_wrist_count": pre_wrist_count,
                "streaming_active": True
            }
        else:
            old_queue_size = FRAME_QUEUES[uid]["frame_queue"].qsize()
            FRAME_QUEUES[uid]["streaming_active"] = True
            FRAME_QUEUES[uid]["last_base_count"] = pre_base_count
            FRAME_QUEUES[uid]["last_wrist_count"] = pre_wrist_count
            
            # 只有在队列中有frames且pre_base_count/pre_wrist_count不为0时才清空队列
            # 如果pre_base_count/pre_wrist_count为0，说明这是第一次初始化，应该保留队列中的初始frames
            should_clear = old_queue_size > 0 and (pre_base_count > 0 or pre_wrist_count > 0)
            
            if should_clear:
                # 清空之前的队列（新action开始）
                while not FRAME_QUEUES[uid]["frame_queue"].empty():
                    try:
                        FRAME_QUEUES[uid]["frame_queue"].get_nowait()
                    except queue.Empty:
                        break
        
        # 启动监控线程
        monitor_thread = threading.Thread(
            target=continuous_frame_monitor,
            args=(uid, pre_base_count, pre_wrist_count),
            daemon=True
        )
        monitor_thread.start()
    
    @staticmethod
    def cleanup_queue(uid):
        """清理指定session的队列"""
        if uid in FRAME_QUEUES:
            queue_info = FRAME_QUEUES[uid]
            # 清空队列
            while not queue_info["frame_queue"].empty():
                try:
                    queue_info["frame_queue"].get_nowait()
                except queue.Empty:
                    break
            queue_info["streaming_active"] = False
    
    @staticmethod
    def get_queue_info(uid):
        """获取队列信息"""
        return get_frame_queue_info(uid)


def monitor_frames_and_enqueue(uid, pre_base_count, pre_wrist_count):
    """
    监控session的帧变化，将新帧加入队列
    
    Args:
        uid: session ID
        pre_base_count: 上次检查时的base_frames数量
        pre_wrist_count: 上次检查时的wrist_frames数量
    
    Returns:
        (current_base_count, current_wrist_count): 当前帧数
    """
    session = get_session(uid)
    if not session or uid not in FRAME_QUEUES:
        return pre_base_count, pre_wrist_count
    
    queue_info = FRAME_QUEUES[uid]
    if not queue_info["streaming_active"]:
        return pre_base_count, pre_wrist_count
    
    # 检查新帧
    current_base_count = len(session.base_frames)
    current_wrist_count = len(session.wrist_frames)
    
    # 获取新增的帧
    if current_base_count > pre_base_count or current_wrist_count > pre_wrist_count:
        new_base = session.base_frames[pre_base_count:current_base_count] if current_base_count > pre_base_count else []
        new_wrist = session.wrist_frames[pre_wrist_count:current_wrist_count] if current_wrist_count > pre_wrist_count else []
        
        # 拼接并加入队列
        if new_base or new_wrist:
            concatenated = concatenate_frames_horizontally(new_base, new_wrist)
            for frame in concatenated:
                try:
                    queue_info["frame_queue"].put(frame, block=False)
                except queue.Full:
                    # 队列已满，跳过此帧（可选：可以限制队列大小）
                    print(f"Warning: Frame queue full for {uid}, dropping frame")
                    break
    
    return current_base_count, current_wrist_count


def continuous_frame_monitor(uid, pre_base_count, pre_wrist_count):
    """
    持续监控帧变化并加入队列的线程函数
    
    Args:
        uid: session ID
        pre_base_count: 执行前的base_frames数量
        pre_wrist_count: 执行前的wrist_frames数量
    """
    last_base_count = pre_base_count
    last_wrist_count = pre_wrist_count
    
    while True:
        if uid not in FRAME_QUEUES:
            break
        
        queue_info = FRAME_QUEUES[uid]
        if not queue_info["streaming_active"]:
            break
        
        # 监控帧变化
        last_base_count, last_wrist_count = monitor_frames_and_enqueue(
            uid, last_base_count, last_wrist_count
        )
        
        # 等待一段时间后再次检查
        time.sleep(STREAMING_MONITOR_INTERVAL)


def cleanup_frame_queue(uid):
    """
    清理指定session的队列
    
    Args:
        uid: session ID
    """
    FrameQueueManager.cleanup_queue(uid)


def generate_mjpeg_stream(uid: str):
    """
    MJPEG 流式生成器：从队列中读取帧并生成 MJPEG 流
    
    Args:
        uid: session ID
    
    Yields:
        JPEG 编码的图片字节流（按照 MJPEG 格式）
    """
    # 发送 MJPEG 头部
    boundary = b"frame"
    
    while True:
        if uid not in FRAME_QUEUES:
            # Session 不存在，等待一段时间后重试
            time.sleep(0.1)
            continue
        
        queue_info = FRAME_QUEUES[uid]
        frame_queue = queue_info["frame_queue"]
        
        try:
            # 从队列中获取一帧（阻塞等待，最多等待 0.1 秒）
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                # 队列为空，发送一个空帧或等待
                time.sleep(0.01)
                continue
            
            # 确保帧是 numpy array 且格式正确
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
            
            # 确保是 uint8 格式
            if frame.dtype != np.uint8:
                if np.max(frame) <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.clip(0, 255).astype(np.uint8)
            
            # 确保是 RGB 格式（3通道）
            if len(frame.shape) == 2:
                frame = np.stack([frame] * 3, axis=-1)
            elif len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            
            # OpenCV 使用 BGR，如果帧是 RGB 格式，需要转换为 BGR
            # 假设输入的 frame 是 RGB 格式（从 concatenate_frames_horizontally 来的）
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # 转换为 BGR 供 OpenCV 使用
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            # 使用 OpenCV 将帧编码为 JPEG
            success, jpeg_bytes = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            if success:
                # 按照 MJPEG 格式发送帧
                yield (b'--' + boundary + b'\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(jpeg_bytes)).encode() + b'\r\n\r\n' +
                       jpeg_bytes.tobytes() + b'\r\n')
            else:
                # 编码失败，跳过此帧
                continue
                
        except Exception as e:
            # 发生错误，记录并继续
            print(f"Error in MJPEG stream generator for {uid}: {e}")
            time.sleep(0.1)
            continue


def create_video_feed_route(fastapi_app):
    """
    创建FastAPI视频流路由
    
    Args:
        fastapi_app: FastAPI应用实例
    """
    @fastapi_app.get("/video_feed/{uid}")
    async def video_feed(uid: str):
        """
        MJPEG 流式视频端点
        
        Args:
            uid: session ID
        
        Returns:
            StreamingResponse: MJPEG 格式的视频流
        """
        return StreamingResponse(
            generate_mjpeg_stream(uid),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
