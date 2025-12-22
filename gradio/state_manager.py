"""
状态管理模块
管理所有全局状态和Session生命周期
"""
import uuid
import threading
from oracle_logic import OracleSession

# --- Global Session Storage ---
GLOBAL_SESSIONS = {}

# --- Task Index Storage (for Progress display) ---
# 存储每个session的任务索引和总任务数，用于直接读取Progress
TASK_INDEX_MAP = {}  # {uid: {"task_index": int, "total_tasks": int}}

# --- Coordinate Click Tracking (between actions) ---
# 跟踪每个session从上次action_execute到现在的所有coordinate_click
COORDINATE_CLICKS = {}  # {uid: [{"coordinates": {"x": x, "y": y}, "coords_str": "...", "image_array": ..., "timestamp": "..."}, ...]}

# --- Option Select Tracking (between actions) ---
# 跟踪每个session从上次action_execute到现在的所有option_select
OPTION_SELECTS = {}  # {uid: [{"option_idx": idx, "option_label": label, "timestamp": "..."}, ...]}

# --- Frame Queue Storage for Streaming ---
# 全局队列存储：{uid: {"frame_queue": queue.Queue, "last_base_count": int, "last_wrist_count": int, "streaming_active": bool}}
FRAME_QUEUES = {}

# --- UI Phase Storage ---
# 存储每个session的UI阶段："watching_demo" 或 "executing_task"
UI_PHASE_MAP = {}  # {uid: "watching_demo" | "executing_task"}

# 线程锁，用于保护全局状态的访问
_state_lock = threading.Lock()


def get_session(uid):
    """获取指定uid的session"""
    with _state_lock:
        return GLOBAL_SESSIONS.get(uid)


def create_session():
    """创建新的session并返回uid"""
    uid = str(uuid.uuid4())
    session = OracleSession()
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


def set_frame_queue_info(uid, queue_info):
    """设置帧队列信息"""
    with _state_lock:
        FRAME_QUEUES[uid] = queue_info


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


def clear_session_data(uid):
    """清理指定session的所有相关数据"""
    with _state_lock:
        # 清理坐标点击
        if uid in COORDINATE_CLICKS:
            COORDINATE_CLICKS[uid] = []
        # 清理选项选择
        if uid in OPTION_SELECTS:
            OPTION_SELECTS[uid] = []
        # 注意：不清理 GLOBAL_SESSIONS 和 TASK_INDEX_MAP，因为它们可能仍在使用
        # FRAME_QUEUES 由 streaming_service 管理清理
        # UI_PHASE_MAP 在加载新任务时会重置
