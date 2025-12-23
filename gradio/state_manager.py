"""
状态管理模块
管理所有全局状态和Session生命周期

本模块负责：
1. 创建和管理 ProcessSessionProxy 实例（每个用户一个）
2. 存储任务索引、坐标点击、选项选择等UI状态
3. 管理视频帧队列（用于MJPEG流式传输）
4. 提供线程安全的访问接口

注意：GLOBAL_SESSIONS 中存储的是 ProcessSessionProxy 对象，而不是 OracleSession。
实际的 OracleSession 运行在独立的工作进程中，通过代理对象进行通信。
"""
import uuid
import threading
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
