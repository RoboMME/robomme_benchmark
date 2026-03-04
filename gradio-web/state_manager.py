"""
状态管理模块
管理所有全局状态和Session生命周期

本模块负责：
1. 创建和管理 ProcessSessionProxy 实例（每个用户一个）
2. 存储任务索引等UI状态
3. 提供线程安全的访问接口
4. 清理会话资源（当用户重复登录时，自动清理旧会话的进程和状态）

注意：GLOBAL_SESSIONS 中存储的是 ProcessSessionProxy 对象，而不是 OracleSession。
实际的 OracleSession 运行在独立的工作进程中，通过代理对象进行通信。
当同一用户第二次登录时，系统会自动清理旧会话的所有资源（进程、RAM、VRAM、状态数据等）。
"""
import logging
import uuid
import threading
import time
from process_session import ProcessSessionProxy
LOGGER = logging.getLogger("robomme.state_manager")

# --- 全局会话存储 ---
# 存储所有用户的 ProcessSessionProxy 实例
# 每个用户登录时会创建一个代理，代理会启动一个独立的工作进程运行 OracleSession
GLOBAL_SESSIONS = {}

# --- 任务索引存储（用于进度显示） ---
# 存储每个session的任务索引和总任务数，用于直接读取Progress
TASK_INDEX_MAP = {}  # {uid: {"task_index": int, "total_tasks": int}}

# --- UI阶段存储 ---
# 存储每个session的UI阶段："watching_demo" 或 "executing_task"
UI_PHASE_MAP = {}  # {uid: "watching_demo" | "executing_task"}

# --- Execute 次数跟踪 ---
# 跟踪每个会话每个任务的 execute 次数
# 键格式: "{uid}:{env_id}:{episode_idx}"
EXECUTE_COUNTS = {}  # {task_key: count}

# --- 任务开始时间跟踪 ---
# 跟踪每个任务的开始时间
# 键格式: "{uid}:{env_id}:{episode_idx}"
# 值: ISO 格式的时间字符串
TASK_START_TIMES = {}  # {task_key: "2025-12-28T14:01:25.372278"}

# --- Session活动时间跟踪 ---
# 跟踪每个session的最后活动时间（用于超时检测）
SESSION_LAST_ACTIVITY = {}  # {uid: timestamp} - timestamp是time.time()返回的浮点数
SESSION_TIMEOUT_WARNED = {}  # {uid: bool} - 跟踪已警告的session，避免重复警告

# --- 播放按钮状态跟踪 ---
# 跟踪每个session的播放按钮是否已被点击（用于execute按钮条件控制）
PLAY_BUTTON_CLICKED = {}  # {uid: bool} - 跟踪播放按钮是否已被点击

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
    5. 初始化最后活动时间为当前时间
    
    Returns:
        str: 新创建的会话ID
    """
    uid = str(uuid.uuid4())
    session = ProcessSessionProxy()
    with _state_lock:
        GLOBAL_SESSIONS[uid] = session
        SESSION_LAST_ACTIVITY[uid] = time.time()
    LOGGER.info("create_session uid=%s total_sessions=%s", uid, len(GLOBAL_SESSIONS))
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


def set_play_button_clicked(uid, clicked=True):
    """
    设置播放按钮是否已被点击
    
    Args:
        uid: 会话ID
        clicked: 是否已被点击（默认 True）
    """
    with _state_lock:
        PLAY_BUTTON_CLICKED[uid] = clicked


def get_play_button_clicked(uid):
    """
    获取播放按钮是否已被点击
    
    Args:
        uid: 会话ID
        
    Returns:
        bool: 如果已被点击返回 True，否则返回 False
    """
    with _state_lock:
        return PLAY_BUTTON_CLICKED.get(uid, False)


def reset_play_button_clicked(uid):
    """
    重置播放按钮点击状态
    
    Args:
        uid: 会话ID
    """
    with _state_lock:
        if uid in PLAY_BUTTON_CLICKED:
            del PLAY_BUTTON_CLICKED[uid]


def _get_task_key(uid, env_id, episode_idx):
    """生成任务键（用于跟踪 execute 次数）"""
    return f"{uid}:{env_id}:{episode_idx}"


def get_execute_count(uid, env_id, episode_idx):
    """
    获取指定任务的 execute 次数
    
    Args:
        uid: 会话ID
        env_id: 环境ID
        episode_idx: Episode索引
        
    Returns:
        int: execute 次数，如果任务不存在则返回 0
    """
    with _state_lock:
        task_key = _get_task_key(uid, env_id, episode_idx)
        return EXECUTE_COUNTS.get(task_key, 0)


def increment_execute_count(uid, env_id, episode_idx):
    """
    增加指定任务的 execute 次数
    
    Args:
        uid: 会话ID
        env_id: 环境ID
        episode_idx: Episode索引
        
    Returns:
        int: 增加后的 execute 次数
    """
    with _state_lock:
        task_key = _get_task_key(uid, env_id, episode_idx)
        current_count = EXECUTE_COUNTS.get(task_key, 0)
        EXECUTE_COUNTS[task_key] = current_count + 1
        return EXECUTE_COUNTS[task_key]


def reset_execute_count(uid, env_id, episode_idx):
    """
    重置指定任务的 execute 次数为 0
    
    Args:
        uid: 会话ID
        env_id: 环境ID
        episode_idx: Episode索引
    """
    with _state_lock:
        task_key = _get_task_key(uid, env_id, episode_idx)
        EXECUTE_COUNTS[task_key] = 0


def get_task_start_time(uid, env_id, episode_idx):
    """
    获取指定任务的开始时间
    
    Args:
        uid: 会话ID
        env_id: 环境ID
        episode_idx: Episode索引
        
    Returns:
        str: ISO 格式的时间字符串，如果任务不存在则返回 None
    """
    with _state_lock:
        task_key = _get_task_key(uid, env_id, episode_idx)
        return TASK_START_TIMES.get(task_key)


def set_task_start_time(uid, env_id, episode_idx, start_time):
    """
    设置指定任务的开始时间
    
    Args:
        uid: 会话ID
        env_id: 环境ID
        episode_idx: Episode索引
        start_time: ISO 格式的时间字符串
    """
    with _state_lock:
        task_key = _get_task_key(uid, env_id, episode_idx)
        TASK_START_TIMES[task_key] = start_time


def clear_task_start_time(uid, env_id, episode_idx):
    """
    清除指定任务的开始时间记录
    
    Args:
        uid: 会话ID
        env_id: 环境ID
        episode_idx: Episode索引
    """
    with _state_lock:
        task_key = _get_task_key(uid, env_id, episode_idx)
        if task_key in TASK_START_TIMES:
            del TASK_START_TIMES[task_key]


def cleanup_session(uid):
    """
    清理指定会话的所有资源
    
    此函数会清理与指定 uid 相关的所有资源：
    1. 关闭 ProcessSessionProxy（会终止工作进程，释放 RAM/VRAM）
    2. 从 GLOBAL_SESSIONS 中移除
    3. 清理所有相关的状态数据（任务索引、UI阶段）
    
    Args:
        uid: 要清理的会话ID
    """
    if not uid:
        return
    
    with _state_lock:
        # 1. 关闭 ProcessSessionProxy（终止工作进程）
        session = GLOBAL_SESSIONS.get(uid)
        if session:
            try:
                LOGGER.info("cleanup_session uid=%s closing ProcessSessionProxy", uid)
                session.close()
                LOGGER.info("cleanup_session uid=%s proxy closed", uid)
            except Exception as e:
                LOGGER.exception("cleanup_session uid=%s proxy close failed: %s", uid, e)
        
        # 2. 从 GLOBAL_SESSIONS 中移除
        if uid in GLOBAL_SESSIONS:
            del GLOBAL_SESSIONS[uid]
            LOGGER.debug("cleanup_session uid=%s removed from GLOBAL_SESSIONS", uid)
        
        # 3. 清理任务索引
        if uid in TASK_INDEX_MAP:
            del TASK_INDEX_MAP[uid]
            LOGGER.debug("cleanup_session uid=%s task index cleaned", uid)

        # 4. 清理UI阶段
        if uid in UI_PHASE_MAP:
            del UI_PHASE_MAP[uid]
        
        # 清理播放按钮状态
        if uid in PLAY_BUTTON_CLICKED:
            del PLAY_BUTTON_CLICKED[uid]
            LOGGER.debug("cleanup_session uid=%s play button state cleaned", uid)
        
        # 5. 清理活动时间跟踪
        if uid in SESSION_LAST_ACTIVITY:
            del SESSION_LAST_ACTIVITY[uid]
            LOGGER.debug("cleanup_session uid=%s last activity cleaned", uid)
        
        # 6. 清理超时警告标志
        if uid in SESSION_TIMEOUT_WARNED:
            del SESSION_TIMEOUT_WARNED[uid]
            LOGGER.debug("cleanup_session uid=%s timeout warning flag cleaned", uid)
        
        # 注意：不清理 EXECUTE_COUNTS，因为它是按任务跟踪的，不是按 session 跟踪的
        # 如果需要清理，应该在任务切换时调用 reset_execute_count
    
    LOGGER.info("cleanup_session uid=%s done", uid)


def update_session_activity(uid):
    """
    更新指定session的最后活动时间为当前时间
    
    Args:
        uid: 会话ID
    """
    with _state_lock:
        SESSION_LAST_ACTIVITY[uid] = time.time()
        # 如果之前被警告过，清除警告标志
        if uid in SESSION_TIMEOUT_WARNED:
            del SESSION_TIMEOUT_WARNED[uid]
            LOGGER.debug("update_session_activity uid=%s cleared timeout warning", uid)


def get_session_activity(uid):
    """
    获取指定session的最后活动时间
    
    Args:
        uid: 会话ID
        
    Returns:
        float: 最后活动时间戳（time.time()），如果session不存在则返回None
    """
    with _state_lock:
        return SESSION_LAST_ACTIVITY.get(uid)


def check_and_cleanup_timeout_sessions():
    """
    检查所有session，清理超时的session
    
    此函数会：
    1. 检查所有活跃session的最后活动时间
    2. 如果超过SESSION_TIMEOUT秒且未警告，设置警告标志并记录日志
    3. 如果已警告且超过警告时间（再等5秒），调用cleanup_session清理资源
    """
    from config import SESSION_TIMEOUT
    
    current_time = time.time()
    timeout_sessions = []
    warned_sessions_to_cleanup = []
    
    with _state_lock:
        # 获取所有活跃的session uid
        active_uids = list(GLOBAL_SESSIONS.keys())
    
    # 在锁外检查，避免长时间持有锁
    for uid in active_uids:
        with _state_lock:
            last_activity = SESSION_LAST_ACTIVITY.get(uid)
            is_warned = SESSION_TIMEOUT_WARNED.get(uid, False)
        
        if last_activity is None:
            # 如果session没有活动记录，跳过（可能是刚创建的）
            continue
        
        elapsed = current_time - last_activity
        
        if elapsed > SESSION_TIMEOUT:
            if not is_warned:
                # 首次超时，设置警告标志
                with _state_lock:
                    SESSION_TIMEOUT_WARNED[uid] = True
                timeout_sessions.append(uid)
                LOGGER.warning("session timeout warning uid=%s elapsed=%.2fs limit=%ss", uid, elapsed, SESSION_TIMEOUT)
            elif elapsed > SESSION_TIMEOUT + 5:
                # 已警告且再等5秒仍未活动，标记为需要清理
                warned_sessions_to_cleanup.append(uid)
    
    # 清理超时的session
    for uid in warned_sessions_to_cleanup:
        LOGGER.warning(
            "session timeout cleanup uid=%s elapsed_limit=%ss",
            uid,
            SESSION_TIMEOUT + 5,
        )
        cleanup_session(uid)
        # cleanup_session内部会清理SESSION_LAST_ACTIVITY和SESSION_TIMEOUT_WARNED


# 后台监控线程相关变量
_timeout_monitor_thread = None
_timeout_monitor_running = False
_timeout_monitor_lock = threading.Lock()


def _timeout_monitor_loop():
    """
    后台监控线程的主循环
    每5秒检查一次所有session的超时状态
    """
    global _timeout_monitor_running
    while _timeout_monitor_running:
        try:
            check_and_cleanup_timeout_sessions()
        except Exception as e:
            LOGGER.exception("timeout monitor loop error: %s", e)
        
        # 每5秒检查一次
        for _ in range(50):  # 5秒 = 50 * 0.1秒
            if not _timeout_monitor_running:
                break
            time.sleep(0.1)


def start_timeout_monitor():
    """
    启动后台监控线程
    在应用启动时调用此函数
    """
    global _timeout_monitor_thread, _timeout_monitor_running
    
    with _timeout_monitor_lock:
        if _timeout_monitor_running:
            LOGGER.info("timeout monitor already running")
            return
        
        _timeout_monitor_running = True
        _timeout_monitor_thread = threading.Thread(
            target=_timeout_monitor_loop,
            daemon=True,
            name="SessionTimeoutMonitor"
        )
        _timeout_monitor_thread.start()
        LOGGER.info("session timeout monitor started")


def stop_timeout_monitor():
    """
    停止后台监控线程
    在应用关闭时调用此函数
    """
    global _timeout_monitor_thread, _timeout_monitor_running
    
    with _timeout_monitor_lock:
        if not _timeout_monitor_running:
            return
        
        _timeout_monitor_running = False
        if _timeout_monitor_thread:
            _timeout_monitor_thread.join(timeout=2.0)
            LOGGER.info("session timeout monitor stopped")
