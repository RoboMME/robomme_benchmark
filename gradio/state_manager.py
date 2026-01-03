"""
状态管理模块
管理所有全局状态和Session生命周期

本模块负责：
1. 创建和管理 ProcessSessionProxy 实例（每个用户一个）
2. 存储任务索引、坐标点击、选项选择等UI状态
3. 管理视频帧队列（用于MJPEG流式传输）
4. 提供线程安全的访问接口
5. 清理会话资源（当用户重复登录时，自动清理旧会话的进程和状态）

注意：GLOBAL_SESSIONS 中存储的是 ProcessSessionProxy 对象，而不是 OracleSession。
实际的 OracleSession 运行在独立的工作进程中，通过代理对象进行通信。
当同一用户第二次登录时，系统会自动清理旧会话的所有资源（进程、RAM、VRAM、状态数据等）。
"""
import uuid
import threading
import traceback
import queue
from process_session import ProcessSessionProxy

# --- 全局会话存储 ---
# 存储所有用户的 ProcessSessionProxy 实例
# 每个用户登录时会创建一个代理，代理会启动一个独立的工作进程运行 OracleSession
GLOBAL_SESSIONS = {}

# --- 任务索引存储（用于进度显示） ---
# 存储每个session的任务索引和总任务数，用于直接读取Progress
TASK_INDEX_MAP = {}  # {uid: {"task_index": int, "total_tasks": int}}

# --- 坐标点击跟踪（两次动作之间） ---
# 跟踪每个session从上次action_execute到现在的所有coordinate_click
COORDINATE_CLICKS = {}  # {uid: [{"coordinates": {"x": x, "y": y}, "coords_str": "...", "image_array": ..., "timestamp": "..."}, ...]}

# --- 选项选择跟踪（两次动作之间） ---
# 跟踪每个session从上次action_execute到现在的所有option_select
OPTION_SELECTS = {}  # {uid: [{"option_idx": idx, "option_label": label, "timestamp": "..."}, ...]}

# --- 视频帧队列存储（用于流式传输） ---
# 全局队列存储：{uid: {"frame_queue": queue.Queue, "last_base_count": int, "last_wrist_count": int, "streaming_active": bool}}
# 注意：帧数据来自 ProcessSessionProxy 的本地缓存，由后台同步线程从工作进程实时更新
FRAME_QUEUES = {}

# --- UI阶段存储 ---
# 存储每个session的UI阶段："watching_demo" 或 "executing_task"
UI_PHASE_MAP = {}  # {uid: "watching_demo" | "executing_task"}

# --- Execute 次数跟踪 ---
# 跟踪每个用户每个任务的 execute 次数
# 键格式: "{username}:{env_id}:{episode_idx}"
EXECUTE_COUNTS = {}  # {task_key: count}

# --- 任务开始时间跟踪 ---
# 跟踪每个任务的开始时间
# 键格式: "{username}:{env_id}:{episode_idx}"
# 值: ISO 格式的时间字符串
TASK_START_TIMES = {}  # {task_key: "2025-12-28T14:01:25.372278"}

# 线程锁，用于保护全局状态的访问
_state_lock = threading.Lock()


def get_session(uid):
    """
    获取指定uid的session（ProcessSessionProxy实例）
    
    Args:
        uid: 会话ID
        
    Returns:
        ProcessSessionProxy: 代理对象，提供与 OracleSession 相同的接口
    """
    with _state_lock:
        return GLOBAL_SESSIONS.get(uid)


def create_session():
    """
    创建新的session并返回uid
    
    此函数会：
    1. 生成一个唯一的会话ID（UUID）
    2. 创建一个 ProcessSessionProxy 实例
    3. ProcessSessionProxy 会自动启动一个独立的工作进程运行 OracleSession
    4. 将代理对象存储到 GLOBAL_SESSIONS 中
    
    Returns:
        str: 新创建的会话ID
    """
    uid = str(uuid.uuid4())
    session = ProcessSessionProxy()
    with _state_lock:
        GLOBAL_SESSIONS[uid] = session
    return uid


def get_task_index(uid):
    """获取任务索引信息"""
    with _state_lock:
        return TASK_INDEX_MAP.get(uid)


def set_task_index(uid, task_index, total_tasks):
    """设置任务索引信息"""
    with _state_lock:
        TASK_INDEX_MAP[uid] = {
            "task_index": task_index,
            "total_tasks": total_tasks
        }


def get_coordinate_clicks(uid):
    """获取坐标点击列表"""
    with _state_lock:
        return COORDINATE_CLICKS.get(uid, [])


def clear_coordinate_clicks(uid):
    """清空坐标点击列表"""
    with _state_lock:
        if uid in COORDINATE_CLICKS:
            COORDINATE_CLICKS[uid] = []


def add_coordinate_click(uid, click_data):
    """添加坐标点击"""
    with _state_lock:
        if uid not in COORDINATE_CLICKS:
            COORDINATE_CLICKS[uid] = []
        COORDINATE_CLICKS[uid].append(click_data)


def get_option_selects(uid):
    """获取选项选择列表"""
    with _state_lock:
        return OPTION_SELECTS.get(uid, [])


def clear_option_selects(uid):
    """清空选项选择列表"""
    with _state_lock:
        if uid in OPTION_SELECTS:
            OPTION_SELECTS[uid] = []


def add_option_select(uid, select_data):
    """添加选项选择"""
    with _state_lock:
        if uid not in OPTION_SELECTS:
            OPTION_SELECTS[uid] = []
        OPTION_SELECTS[uid].append(select_data)


def get_frame_queue_info(uid):
    """获取帧队列信息"""
    with _state_lock:
        return FRAME_QUEUES.get(uid)


def get_ui_phase(uid):
    """获取UI阶段"""
    with _state_lock:
        return UI_PHASE_MAP.get(uid, "watching_demo")  # 默认为观看示范阶段


def set_ui_phase(uid, phase):
    """设置UI阶段
    
    Args:
        uid: session ID
        phase: "watching_demo" 或 "executing_task"
    """
    with _state_lock:
        UI_PHASE_MAP[uid] = phase


def reset_ui_phase(uid):
    """重置UI阶段为初始阶段（watching_demo）"""
    with _state_lock:
        UI_PHASE_MAP[uid] = "watching_demo"


def _get_task_key(username, env_id, episode_idx):
    """生成任务键（用于跟踪 execute 次数）"""
    return f"{username}:{env_id}:{episode_idx}"


def get_execute_count(username, env_id, episode_idx):
    """
    获取指定任务的 execute 次数
    
    Args:
        username: 用户名
        env_id: 环境ID
        episode_idx: Episode索引
        
    Returns:
        int: execute 次数，如果任务不存在则返回 0
    """
    with _state_lock:
        task_key = _get_task_key(username, env_id, episode_idx)
        return EXECUTE_COUNTS.get(task_key, 0)


def increment_execute_count(username, env_id, episode_idx):
    """
    增加指定任务的 execute 次数
    
    Args:
        username: 用户名
        env_id: 环境ID
        episode_idx: Episode索引
        
    Returns:
        int: 增加后的 execute 次数
    """
    with _state_lock:
        task_key = _get_task_key(username, env_id, episode_idx)
        current_count = EXECUTE_COUNTS.get(task_key, 0)
        EXECUTE_COUNTS[task_key] = current_count + 1
        return EXECUTE_COUNTS[task_key]


def reset_execute_count(username, env_id, episode_idx):
    """
    重置指定任务的 execute 次数为 0
    
    Args:
        username: 用户名
        env_id: 环境ID
        episode_idx: Episode索引
    """
    with _state_lock:
        task_key = _get_task_key(username, env_id, episode_idx)
        EXECUTE_COUNTS[task_key] = 0


def get_task_start_time(username, env_id, episode_idx):
    """
    获取指定任务的开始时间
    
    Args:
        username: 用户名
        env_id: 环境ID
        episode_idx: Episode索引
        
    Returns:
        str: ISO 格式的时间字符串，如果任务不存在则返回 None
    """
    with _state_lock:
        task_key = _get_task_key(username, env_id, episode_idx)
        return TASK_START_TIMES.get(task_key)


def set_task_start_time(username, env_id, episode_idx, start_time):
    """
    设置指定任务的开始时间
    
    Args:
        username: 用户名
        env_id: 环境ID
        episode_idx: Episode索引
        start_time: ISO 格式的时间字符串
    """
    with _state_lock:
        task_key = _get_task_key(username, env_id, episode_idx)
        TASK_START_TIMES[task_key] = start_time


def clear_task_start_time(username, env_id, episode_idx):
    """
    清除指定任务的开始时间记录
    
    Args:
        username: 用户名
        env_id: 环境ID
        episode_idx: Episode索引
    """
    with _state_lock:
        task_key = _get_task_key(username, env_id, episode_idx)
        if task_key in TASK_START_TIMES:
            del TASK_START_TIMES[task_key]


def cleanup_session(uid):
    """
    清理指定会话的所有资源
    
    此函数会清理与指定 uid 相关的所有资源：
    1. 关闭 ProcessSessionProxy（会终止工作进程，释放 RAM/VRAM）
    2. 从 GLOBAL_SESSIONS 中移除
    3. 清理所有相关的状态数据（任务索引、坐标点击、选项选择、帧队列、UI阶段）
    4. 清理流生成ID（用于终止旧的MJPEG流）
    
    Args:
        uid: 要清理的会话ID
    """
    if not uid:
        return
    
    # 先获取需要清理的帧队列信息（在锁外，避免死锁）
    frame_queue_info = None
    with _state_lock:
        if uid in FRAME_QUEUES:
            frame_queue_info = FRAME_QUEUES[uid]
    
    # 在锁外停止帧队列的流式传输（避免死锁）
    if frame_queue_info:
        try:
            frame_queue_info["streaming_active"] = False
            # 清空队列中的所有帧
            while not frame_queue_info["frame_queue"].empty():
                try:
                    frame_queue_info["frame_queue"].get_nowait()
                except queue.Empty:
                    break
        except Exception as e:
            print(f"Error stopping frame queue for {uid}: {e}")
    
    # 清理流生成ID（在锁外，因为它在 streaming_service 模块中）
    try:
        from streaming_service import STREAM_GENERATIONS
        if uid in STREAM_GENERATIONS:
            STREAM_GENERATIONS[uid] = STREAM_GENERATIONS.get(uid, 0) + 1
    except Exception as e:
        print(f"Error updating stream generation for {uid}: {e}")
    
    with _state_lock:
        # 1. 关闭 ProcessSessionProxy（终止工作进程）
        session = GLOBAL_SESSIONS.get(uid)
        if session:
            try:
                print(f"Cleaning up session {uid}: closing ProcessSessionProxy...")
                session.close()
                print(f"Session {uid}: ProcessSessionProxy closed successfully")
            except Exception as e:
                print(f"Error closing ProcessSessionProxy for {uid}: {e}")
                traceback.print_exc()
        
        # 2. 从 GLOBAL_SESSIONS 中移除
        if uid in GLOBAL_SESSIONS:
            del GLOBAL_SESSIONS[uid]
            print(f"Session {uid}: removed from GLOBAL_SESSIONS")
        
        # 3. 清理帧队列
        if uid in FRAME_QUEUES:
            del FRAME_QUEUES[uid]
            print(f"Session {uid}: frame queue cleaned up")
        
        # 4. 清理任务索引
        if uid in TASK_INDEX_MAP:
            del TASK_INDEX_MAP[uid]
            print(f"Session {uid}: task index cleaned up")
        
        # 5. 清理坐标点击
        if uid in COORDINATE_CLICKS:
            del COORDINATE_CLICKS[uid]
            print(f"Session {uid}: coordinate clicks cleaned up")
        
        # 6. 清理选项选择
        if uid in OPTION_SELECTS:
            del OPTION_SELECTS[uid]
            print(f"Session {uid}: option selects cleaned up")
        
        # 7. 清理UI阶段
        if uid in UI_PHASE_MAP:
            del UI_PHASE_MAP[uid]
            print(f"Session {uid}: UI phase cleaned up")
        
        # 注意：不清理 EXECUTE_COUNTS，因为它是按任务跟踪的，不是按 session 跟踪的
        # 如果需要清理，应该在任务切换时调用 reset_execute_count
    
    print(f"Session {uid}: all resources cleaned up successfully")
