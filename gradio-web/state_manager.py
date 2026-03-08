"""
状态管理模块
管理所有全局状态和 Session 生命周期。

GLOBAL_SESSIONS 中存储的是 ProcessSessionProxy，而不是 OracleSession。
实际的 OracleSession 运行在独立工作进程中，通过代理对象进行通信。
"""

import logging
import threading

from process_session import ProcessSessionProxy

LOGGER = logging.getLogger("robomme.state_manager")

# --- 全局会话存储 ---
GLOBAL_SESSIONS = {}
ACTIVE_SESSION_SLOTS = set()

# --- 任务索引存储（用于进度显示） ---
TASK_INDEX_MAP = {}  # {uid: {"task_index": int, "total_tasks": int}}

# --- UI阶段存储 ---
UI_PHASE_MAP = {}  # {uid: "watching_demo" | "executing_task"}

# --- Execute 次数跟踪 ---
EXECUTE_COUNTS = {}  # {"{uid}:{env_id}:{episode_idx}": count}

# --- 任务开始时间跟踪 ---
TASK_START_TIMES = {}  # {"{uid}:{env_id}:{episode_idx}": iso_timestamp}

# --- 播放按钮状态跟踪 ---
PLAY_BUTTON_CLICKED = {}  # {uid: bool}

_state_lock = threading.Lock()


def get_session(uid):
    """获取指定 uid 的 ProcessSessionProxy。"""
    with _state_lock:
        return GLOBAL_SESSIONS.get(uid)


def _try_reserve_session_slot_locked(uid, session_concurrency_limit):
    if uid in ACTIVE_SESSION_SLOTS:
        return True

    if len(ACTIVE_SESSION_SLOTS) >= int(session_concurrency_limit):
        LOGGER.info(
            "try_reserve_session_slot rejected uid=%s active_slots=%s limit=%s",
            uid,
            len(ACTIVE_SESSION_SLOTS),
            session_concurrency_limit,
        )
        return False

    ACTIVE_SESSION_SLOTS.add(uid)
    LOGGER.info(
        "try_reserve_session_slot acquired uid=%s active_slots=%s",
        uid,
        len(ACTIVE_SESSION_SLOTS),
    )
    return True


def try_reserve_session_slot(uid):
    """
    Try to reserve a session slot without blocking.

    Returns:
        bool: whether the slot was acquired
    """
    if not uid:
        raise ValueError("Session uid cannot be empty")

    from config import SESSION_CONCURRENCY_LIMIT

    with _state_lock:
        return _try_reserve_session_slot_locked(uid, SESSION_CONCURRENCY_LIMIT)


def release_session_slot(uid):
    if not uid:
        return

    with _state_lock:
        if uid in ACTIVE_SESSION_SLOTS:
            ACTIVE_SESSION_SLOTS.remove(uid)
            LOGGER.info(
                "release_session_slot uid=%s active_slots=%s",
                uid,
                len(ACTIVE_SESSION_SLOTS),
            )


def try_create_session(uid):
    """
    Try to create a ProcessSessionProxy without blocking on session slot wait.

    Returns:
        bool: whether the session is ready
    """
    if not uid:
        raise ValueError("Session uid cannot be empty")

    with _state_lock:
        if GLOBAL_SESSIONS.get(uid) is not None:
            return True
        if not _try_reserve_session_slot_locked(uid, _get_session_concurrency_limit()):
            return False
        try:
            GLOBAL_SESSIONS[uid] = ProcessSessionProxy()
        except Exception:
            ACTIVE_SESSION_SLOTS.discard(uid)
            raise

    LOGGER.info("try_create_session uid=%s total_sessions=%s", uid, len(GLOBAL_SESSIONS))
    return True


def create_session(uid):
    """
    为指定 session key 创建 ProcessSessionProxy。

    超出并发上限时立即失败，不执行排队等待。
    """
    if not uid:
        raise ValueError("Session uid cannot be empty")

    ready = try_create_session(uid)
    if not ready:
        raise RuntimeError("No session slots available")
    LOGGER.info("create_session uid=%s total_sessions=%s", uid, len(GLOBAL_SESSIONS))
    return uid


def _get_session_concurrency_limit():
    from config import SESSION_CONCURRENCY_LIMIT

    return SESSION_CONCURRENCY_LIMIT


def get_task_index(uid):
    """获取任务索引信息。"""
    with _state_lock:
        return TASK_INDEX_MAP.get(uid)


def set_task_index(uid, task_index, total_tasks):
    """设置任务索引信息。"""
    with _state_lock:
        TASK_INDEX_MAP[uid] = {
            "task_index": task_index,
            "total_tasks": total_tasks,
        }


def get_ui_phase(uid):
    """获取 UI 阶段。"""
    with _state_lock:
        return UI_PHASE_MAP.get(uid, "watching_demo")


def set_ui_phase(uid, phase):
    """设置 UI 阶段。"""
    with _state_lock:
        UI_PHASE_MAP[uid] = phase


def reset_ui_phase(uid):
    """重置 UI 阶段为初始阶段。"""
    with _state_lock:
        UI_PHASE_MAP[uid] = "watching_demo"


def set_play_button_clicked(uid, clicked=True):
    """设置播放按钮是否已被点击。"""
    with _state_lock:
        PLAY_BUTTON_CLICKED[uid] = clicked


def get_play_button_clicked(uid):
    """获取播放按钮是否已被点击。"""
    with _state_lock:
        return PLAY_BUTTON_CLICKED.get(uid, False)


def reset_play_button_clicked(uid):
    """重置播放按钮点击状态。"""
    with _state_lock:
        PLAY_BUTTON_CLICKED.pop(uid, None)


def _get_task_key(uid, env_id, episode_idx):
    return f"{uid}:{env_id}:{episode_idx}"


def get_execute_count(uid, env_id, episode_idx):
    """获取指定任务的 execute 次数。"""
    with _state_lock:
        task_key = _get_task_key(uid, env_id, episode_idx)
        return EXECUTE_COUNTS.get(task_key, 0)


def increment_execute_count(uid, env_id, episode_idx):
    """增加指定任务的 execute 次数。"""
    with _state_lock:
        task_key = _get_task_key(uid, env_id, episode_idx)
        current_count = EXECUTE_COUNTS.get(task_key, 0)
        EXECUTE_COUNTS[task_key] = current_count + 1
        return EXECUTE_COUNTS[task_key]


def reset_execute_count(uid, env_id, episode_idx):
    """重置指定任务的 execute 次数为 0。"""
    with _state_lock:
        task_key = _get_task_key(uid, env_id, episode_idx)
        EXECUTE_COUNTS[task_key] = 0


def get_task_start_time(uid, env_id, episode_idx):
    """获取指定任务的开始时间。"""
    with _state_lock:
        task_key = _get_task_key(uid, env_id, episode_idx)
        return TASK_START_TIMES.get(task_key)


def set_task_start_time(uid, env_id, episode_idx, start_time):
    """设置指定任务的开始时间。"""
    with _state_lock:
        task_key = _get_task_key(uid, env_id, episode_idx)
        TASK_START_TIMES[task_key] = start_time


def clear_task_start_time(uid, env_id, episode_idx):
    """清除指定任务的开始时间记录。"""
    with _state_lock:
        task_key = _get_task_key(uid, env_id, episode_idx)
        TASK_START_TIMES.pop(task_key, None)


def cleanup_session(uid):
    """清理指定会话的所有资源。"""
    if not uid:
        return

    session = None
    task_prefix = f"{uid}:"

    with _state_lock:
        session = GLOBAL_SESSIONS.pop(uid, None)
        TASK_INDEX_MAP.pop(uid, None)
        UI_PHASE_MAP.pop(uid, None)
        PLAY_BUTTON_CLICKED.pop(uid, None)

        execute_keys = [key for key in EXECUTE_COUNTS if key.startswith(task_prefix)]
        task_start_keys = [key for key in TASK_START_TIMES if key.startswith(task_prefix)]

        for key in execute_keys:
            del EXECUTE_COUNTS[key]
        for key in task_start_keys:
            del TASK_START_TIMES[key]

    if session is not None:
        try:
            LOGGER.info("cleanup_session uid=%s closing ProcessSessionProxy", uid)
            session.close()
            LOGGER.info("cleanup_session uid=%s proxy closed", uid)
        except Exception as exc:
            LOGGER.exception("cleanup_session uid=%s proxy close failed: %s", uid, exc)
    release_session_slot(uid)

    from user_manager import user_manager

    user_manager.cleanup_session(uid)
    LOGGER.info("cleanup_session uid=%s done", uid)
