"""
流媒体服务模块
处理MJPEG流、帧队列和后台监控线程

本模块负责：
1. 从 ProcessSessionProxy 的本地缓存读取视频帧
2. 监控帧变化并将新帧加入队列
3. 生成MJPEG流式视频供浏览器播放

注意：session.base_frames 和 session.wrist_frames 来自 ProcessSessionProxy 的本地缓存，
这些数据由后台同步线程从工作进程实时更新。
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

# Global map for stream generations to prevent race conditions
STREAM_GENERATIONS = {}  # {uid: int_generation_id}


class FrameQueueManager:
    """帧队列管理器"""
    
    @staticmethod
    def init_queue(uid, pre_base_count, pre_wrist_count):
        """
        初始化队列并启动监控线程
        
        Args:
            uid: session ID
            pre_base_count: 当前已有的 base_frames 数量（用于标记监控起始点）
            pre_wrist_count: 当前已有的 wrist_frames 数量（用于标记监控起始点）
        
        逻辑说明：
        - 如果队列不存在，创建新队列
        - 如果队列已存在，更新监控起始点；如果队列中有旧帧且 pre_count > 0，清空队列（新任务开始）
        - 启动后台监控线程，持续监控帧变化并加入队列
        """
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
            
            # 如果队列中有旧帧且 pre_count > 0，说明是新任务开始，需要清空队列
            # 如果 pre_count = 0，说明是第一次初始化，保留队列中的初始帧
            should_clear = old_queue_size > 0 and (pre_base_count > 0 or pre_wrist_count > 0)
            
            if should_clear:
                # 清空之前的队列（新任务开始）
                while not FRAME_QUEUES[uid]["frame_queue"].empty():
                    try:
                        FRAME_QUEUES[uid]["frame_queue"].get_nowait()
                    except queue.Empty:
                        break
        
        # 启动后台监控线程，持续监控帧变化
        monitor_thread = threading.Thread(
            target=continuous_frame_monitor,
            args=(uid, pre_base_count, pre_wrist_count),
            daemon=True
        )
        monitor_thread.start()
    
    @staticmethod
    def cleanup_queue(uid):
        """
        清理指定session的队列
        
        清理操作包括：
        1. 清空队列中的所有帧
        2. 标记流为非活跃状态
        3. 删除队列条目，强制旧的 MJPEG 生成器退出（检测到 uid not in FRAME_QUEUES）
        4. 递增 generation ID，使旧的流生成器检测到不匹配并自动退出
        """
        if uid in FRAME_QUEUES:
            queue_info = FRAME_QUEUES[uid]
            # 清空队列
            while not queue_info["frame_queue"].empty():
                try:
                    queue_info["frame_queue"].get_nowait()
                except queue.Empty:
                    break
            queue_info["streaming_active"] = False
            # 彻底删除队列条目，强制终止旧的 MJPEG 生成器
            del FRAME_QUEUES[uid]
            
        # 递增 generation ID，使旧的流生成器检测到不匹配并自动退出
        STREAM_GENERATIONS[uid] = STREAM_GENERATIONS.get(uid, 0) + 1
    
    @staticmethod
    def get_queue_info(uid):
        """获取队列信息"""
        return get_frame_queue_info(uid)


def monitor_frames_and_enqueue(uid, pre_base_count, pre_wrist_count):
    """
    监控session的帧变化，将新帧加入队列
    
    此函数从 ProcessSessionProxy 的本地缓存读取帧数据。
    这些帧由代理的后台同步线程从工作进程实时更新。
    
    工作原理：
    - 比较当前帧数量与上次检查时的数量
    - 如果发现新帧（数量增加），提取新增的帧并拼接
    - 将拼接后的帧加入队列供 MJPEG 流使用
    
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
    
    # 检查新帧（从 ProcessSessionProxy 的本地缓存读取）
    current_base_count = len(session.base_frames) if session.base_frames else 0
    current_wrist_count = len(session.wrist_frames) if session.wrist_frames else 0
    
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
    持续监控帧变化并加入队列的后台线程函数
    
    此函数在独立线程中运行，定期检查帧变化并将新帧加入队列。
    当队列被清理或流被停止时，线程自动退出。
    
    Args:
        uid: session ID
        pre_base_count: 监控起始点的base_frames数量
        pre_wrist_count: 监控起始点的wrist_frames数量
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
    
    功能特性：
    1. 从队列中读取新帧并发送
    2. 如果队列为空，使用 keep-alive 机制定期重发最后一帧以保持连接
    3. 如果从未发送过帧（队列初始化时为空），直接从 session 获取当前帧作为初始帧
    4. 使用 generation ID 机制防止旧流干扰新流
    
    Args:
        uid: session ID
    
    Yields:
        JPEG 编码的图片字节流（按照 MJPEG 格式）
    """
    # 捕获当前 generation ID，用于检测流是否已被新任务替换
    my_generation = STREAM_GENERATIONS.get(uid, 0)
    
    # MJPEG 格式边界标识
    boundary = b"frame"
    
    # Keep-alive 机制：缓存最后一帧，定期重发以保持连接活跃
    last_yielded_frame_bytes = None
    last_yield_time = time.time()
    KEEP_ALIVE_INTERVAL = 0.5  # 至少每 0.5 秒发送一帧
    
    while True:
        # 检查流是否已被新任务替换（cleanup_queue 会递增 generation ID）
        current_generation = STREAM_GENERATIONS.get(uid, 0)
        if current_generation != my_generation:
            print(f"Stream generation mismatch for {uid}: {my_generation} vs {current_generation}. Terminating old stream.")
            break
            
        if uid not in FRAME_QUEUES:
            # Session 不存在，等待一段时间后重试
            time.sleep(0.1)
            continue
        
        queue_info = FRAME_QUEUES[uid]
        frame_queue = queue_info["frame_queue"]
        
        try:
            frame_to_send = None
            
            # 从队列中获取一帧（阻塞等待，最多等待 0.1 秒）
            try:
                frame = frame_queue.get(timeout=0.1)
                frame_to_send = frame
            except queue.Empty:
                # 队列为空时的处理策略
                if last_yielded_frame_bytes and (time.time() - last_yield_time > KEEP_ALIVE_INTERVAL):
                    # 策略1: 如果已有缓存帧，定期重发以保持连接活跃
                    try:
                        yield (b'--' + boundary + b'\r\n'
                               b'Content-Type: image/jpeg\r\n'
                               b'Content-Length: ' + str(len(last_yielded_frame_bytes)).encode() + b'\r\n\r\n' +
                               last_yielded_frame_bytes + b'\r\n')
                        last_yield_time = time.time()
                    except Exception as e:
                        print(f"Error sending keep-alive frame for {uid}: {e}")
                        break
                elif last_yielded_frame_bytes is None:
                    # 策略2: 如果从未发送过帧（队列初始化时为空），直接从 session 获取当前帧
                    # 这解决了新任务加载时队列尚未填充导致的空白屏幕问题
                    try:
                        session = get_session(uid)
                        if session:
                            last_base_frame = session.base_frames[-1] if session.base_frames else None
                            last_wrist_frame = session.wrist_frames[-1] if session.wrist_frames else None
                            
                            if last_base_frame is not None or last_wrist_frame is not None:
                                current_frames = concatenate_frames_horizontally(
                                    [last_base_frame] if last_base_frame is not None else [],
                                    [last_wrist_frame] if last_wrist_frame is not None else []
                                )
                                if current_frames:
                                    frame_to_send = current_frames[0]
                    except Exception as e:
                        print(f"Error fetching fallback frame for {uid}: {e}")

                time.sleep(0.01)
                
                # 如果获取到了 fallback 帧，继续处理；否则继续循环等待
                if frame_to_send is None:
                    continue
            
            # Process the new frame
            if frame_to_send is not None:
                frame = frame_to_send
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
                    current_bytes = jpeg_bytes.tobytes()
                    last_yielded_frame_bytes = current_bytes  # Update cache
                    try:
                        yield (b'--' + boundary + b'\r\n'
                               b'Content-Type: image/jpeg\r\n'
                               b'Content-Length: ' + str(len(current_bytes)).encode() + b'\r\n\r\n' +
                               current_bytes + b'\r\n')
                        last_yield_time = time.time()
                    except GeneratorExit:
                        # 客户端断开连接
                        print(f"Client disconnected for {uid}")
                        break
                    except Exception as e:
                        print(f"Error sending frame for {uid}: {e}")
                        break
                else:
                    # 编码失败，跳过此帧
                    continue
                
        except Exception as e:
            # 发生错误，记录并退出（防止僵尸线程窃取队列数据）
            print(f"Error in MJPEG stream generator for {uid}: {e}")
            break


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
