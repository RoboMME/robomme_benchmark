"""
Gradio回调函数模块
响应UI事件，调用业务逻辑，返回UI更新
"""
import logging
import os
import queue
import re
import threading
import time
from datetime import datetime

import gradio as gr
import numpy as np
from PIL import Image

from state_manager import (
    cleanup_session,
    get_execute_count,
    get_play_button_clicked,
    get_session,
    increment_execute_count,
    reset_play_button_clicked,
    reset_execute_count,
    set_play_button_clicked,
    set_task_start_time,
    set_ui_phase,
    try_create_session,
)
from image_utils import draw_marker, save_video, concatenate_frames_horizontally
from user_manager import user_manager
from config import (
    EXECUTE_LIMIT_OFFSET,
    KEYFRAME_DOWNSAMPLE_FACTOR,
    LIVE_OBS_REFRESH_HZ,
    UI_TEXT,
    USE_SEGMENTED_VIEW,
    get_live_obs_elem_classes,
    get_ui_action_text,
    should_show_demo_video,
)
from process_session import ScrewPlanFailureError
from note_content import get_task_hint


# --- live_obs refresh queue state ---
# Each uid keeps its own FIFO queue and sampling cursor.
_LIVE_OBS_REFRESH = {}
_LIVE_OBS_REFRESH_LOCK = threading.Lock()
LOGGER = logging.getLogger("robomme.callbacks")


def _ui_text(section, key, **kwargs):
    template = UI_TEXT[section][key]
    return template.format(**kwargs) if kwargs else template


_LIVE_OBS_UPDATE_SKIP = object()


def _action_selection_log():
    return format_log_markdown(_ui_text("log", "action_selection_prompt"))


def _point_selection_log():
    return format_log_markdown(_ui_text("log", "point_selection_prompt"))


def _session_error_text():
    return _ui_text("log", "session_error")


def _entry_rejected_text():
    return _ui_text("progress", "entry_rejected")


def touch_session(uid):
    """Re-emit the current session key to refresh gr.State TTL."""
    if not uid:
        return None
    # Keep the browser-side uid even when the backend session is stubbed or not yet materialized.
    return uid


def cleanup_user_session(uid):
    """Unified cleanup entry for gr.State TTL deletion and unload hooks."""
    if not uid:
        return
    with _LIVE_OBS_REFRESH_LOCK:
        _LIVE_OBS_REFRESH.pop(uid, None)
    cleanup_session(uid)


def cleanup_current_request_session(request: gr.Request):
    """Clean up the current browser tab's session on unload."""
    cleanup_user_session(getattr(request, "session_hash", None))


def _live_obs_update(
    *,
    value=_LIVE_OBS_UPDATE_SKIP,
    interactive=None,
    visible=None,
    waiting_for_point=False,
):
    kwargs = {
        "elem_classes": get_live_obs_elem_classes(waiting_for_point=waiting_for_point),
    }
    if value is not _LIVE_OBS_UPDATE_SKIP:
        kwargs["value"] = value
    if interactive is not None:
        kwargs["interactive"] = interactive
    if visible is not None:
        kwargs["visible"] = visible
    return gr.update(**kwargs)


def _parse_option_idx(option_value):
    if isinstance(option_value, tuple):
        _, option_idx = option_value
        return option_idx
    return option_value


def _option_requires_coords(session, option_value) -> bool:
    option_idx = _parse_option_idx(option_value)
    if not isinstance(option_idx, int):
        return False
    raw_solve_options = getattr(session, "raw_solve_options", None)
    if not isinstance(raw_solve_options, list):
        return False
    if not (0 <= option_idx < len(raw_solve_options)):
        return False
    return bool(raw_solve_options[option_idx].get("available"))


def _should_enqueue_sample(sample_index: int) -> bool:
    factor = max(1, int(KEYFRAME_DOWNSAMPLE_FACTOR))
    return sample_index % factor == 0


def _live_obs_refresh_interval_sec() -> float:
    return 1.0 / max(float(LIVE_OBS_REFRESH_HZ), 1.0)


def _uid_for_log(uid):
    if not uid:
        return "<none>"
    text = str(uid)
    return text if len(text) <= 12 else f"{text[:8]}..."


def capitalize_first_letter(text: str) -> str:
    """确保字符串的第一个字母大写，其余字符保持不变"""
    if not text:
        return text
    if len(text) == 1:
        return text.upper()
    return text[0].upper() + text[1:]




def _ui_option_label(session, opt_label: str, opt_idx: int) -> str:
    """
    仅在 Gradio UI 层对选项显示文案做覆盖（不改底层 env/options 生成逻辑）。
    优先使用 raw_solve_options 中的原始 label/action 重新组装显示文本，
    并按 env_id 做 display-only action 文案映射。
    """
    try:
        option_index = int(opt_idx)
    except (TypeError, ValueError):
        return opt_label

    raw_solve_options = getattr(session, "raw_solve_options", None)
    if not isinstance(raw_solve_options, list):
        return opt_label
    if not (0 <= option_index < len(raw_solve_options)):
        return opt_label

    raw_option = raw_solve_options[option_index]
    if not isinstance(raw_option, dict):
        return opt_label

    raw_label = str(raw_option.get("label", "")).strip()
    raw_action = str(raw_option.get("action", "")).strip()
    mapped_action = get_ui_action_text(getattr(session, "env_id", None), raw_action)

    if raw_label and mapped_action:
        return f"{raw_label}. {mapped_action}"
    if mapped_action:
        return mapped_action
    if raw_label:
        return raw_label
    return opt_label


def format_log_markdown(log_message):
    """
    将日志消息标准化为纯文本，供 Textbox 展示。

    Args:
        log_message: 纯文本日志消息（可以是多行）

    Returns:
        str: 清洗后的纯文本日志字符串
    """
    if log_message is None:
        return ""
    return str(log_message).replace("\r\n", "\n").replace("\r", "\n")


def show_task_hint(uid, current_hint=""):
    """
    按需加载任务提示内容（仅在用户点击"Task Hint"按钮时调用）
    On-demand loading of task hint based on current session's env_id.
    支持切换显示/隐藏：如果当前提示为空则显示，如果不为空则隐藏。
    
    【修改说明】
    此函数用于实现任务提示的延迟加载和切换显示功能。用户点击"Task Hint"按钮时：
    - 如果当前提示内容为空，则从当前session中读取env_id并加载对应的提示内容
    - 如果当前提示内容不为空，则清空提示内容（隐藏）
    
    Args:
        uid: 用户会话的唯一标识符，用于获取当前session对象
        current_hint: 当前提示内容的文本，用于判断是否显示/隐藏
        
    Returns:
        str: 根据当前环境ID返回的任务提示内容（Markdown格式），
             如果当前提示不为空则返回空字符串（隐藏），
             如果session不存在或env_id未加载则返回空字符串或错误提示
    """
    # 如果当前提示内容不为空，则切换为隐藏（返回空字符串）
    if current_hint and current_hint.strip():
        return ""
    
    # 从全局状态管理器中获取当前用户的session对象
    session = get_session(uid)
    if not session:
        # 如果session不存在，返回空字符串（前端不会显示任何内容）
        return ""
    
    # 从session对象中获取当前加载的环境ID（env_id）
    # 使用getattr安全获取属性，如果不存在则返回None
    env_id = getattr(session, 'env_id', None)
    if not env_id:
        # 如果环境ID未加载，返回提示信息
        return "No environment loaded."
    
    # 根据环境ID调用get_task_hint函数获取对应的任务提示内容
    # 该函数会根据不同的env_id返回不同的提示文本（如PickXtimes、VideoPlaceOrder等）
    return get_task_hint(env_id)


def on_video_end(uid):
    """
    Called when the demonstration video finishes playing.
    Updates the system log to prompt for action selection.
    """
    return _action_selection_log()


def on_demo_video_play(uid):
    """Mark the demo video as consumed and disable the play button."""
    if not uid:
        LOGGER.warning("on_demo_video_play: missing uid")
        raise gr.Error(_session_error_text())

    if not get_session(uid):
        LOGGER.warning(
            "on_demo_video_play: missing session uid=%s; disabling button anyway",
            _uid_for_log(uid),
        )

    already_clicked = get_play_button_clicked(uid)
    if not already_clicked:
        set_play_button_clicked(uid, True)
    LOGGER.debug(
        "demo video play clicked uid=%s already_clicked=%s",
        _uid_for_log(uid),
        already_clicked,
    )
    return gr.update(visible=True, interactive=False)


def switch_to_execute_phase(uid):
    """Disable controls and point clicking during execute playback."""
    if uid:
        session = get_session(uid)
        base_count = len(getattr(session, "base_frames", []) or []) if session else 0
        LOGGER.debug(
            "switch_to_execute_phase uid=%s base_frames=%s",
            _uid_for_log(uid),
            base_count,
        )
        with _LIVE_OBS_REFRESH_LOCK:
            _LIVE_OBS_REFRESH[uid] = {
                "frame_queue": queue.Queue(),
                "last_base_count": base_count,
                "sample_index": 0,
            }
    return (
        gr.update(interactive=False),  # options_radio
        gr.update(interactive=False),  # exec_btn
        gr.update(interactive=False),  # restart_episode_btn
        gr.update(interactive=False),  # next_task_btn
        _live_obs_update(interactive=False),  # img_display
        gr.update(interactive=False),  # reference_action_btn
    )


def switch_to_action_phase(uid=None):
    """Switch display to action phase and restore control panel interactions."""
    if uid:
        LOGGER.debug("switch_to_action_phase uid=%s", _uid_for_log(uid))
        with _LIVE_OBS_REFRESH_LOCK:
            _LIVE_OBS_REFRESH.pop(uid, None)
    return (
        gr.update(interactive=True),  # options_radio
        gr.update(),  # exec_btn (keep execute_step result)
        gr.update(),  # restart_episode_btn (keep execute_step result)
        gr.update(),  # next_task_btn (keep execute_step result)
        _live_obs_update(interactive=True),  # img_display
        gr.update(interactive=True),  # reference_action_btn
    )


def _get_live_obs_refresh_state(uid, base_count=0):
    with _LIVE_OBS_REFRESH_LOCK:
        if uid not in _LIVE_OBS_REFRESH:
            _LIVE_OBS_REFRESH[uid] = {
                "frame_queue": queue.Queue(),
                "last_base_count": int(base_count),
                "sample_index": 0,
            }
        return _LIVE_OBS_REFRESH[uid]


def _enqueue_live_obs_frames(uid, base_frames):
    """
    Push newly appended base_frames into per-uid FIFO queue with configurable downsampling.
    """
    if not uid:
        return 0
    frames = base_frames or []
    state = _get_live_obs_refresh_state(uid, base_count=len(frames))
    frame_queue = state["frame_queue"]
    current_count = len(frames)
    last_count = int(state.get("last_base_count", 0))

    # Session/task reset: history shrank.
    if current_count < last_count:
        with _LIVE_OBS_REFRESH_LOCK:
            state["frame_queue"] = queue.Queue()
            state["last_base_count"] = current_count
            state["sample_index"] = 0
        return 0

    if current_count <= last_count:
        return frame_queue.qsize()

    new_frames = frames[last_count:current_count]
    sample_index = int(state.get("sample_index", 0))
    for frame in new_frames:
        if _should_enqueue_sample(sample_index) and frame is not None:
            frame_queue.put(frame)
        sample_index += 1

    with _LIVE_OBS_REFRESH_LOCK:
        state["last_base_count"] = current_count
        state["sample_index"] = sample_index
    return frame_queue.qsize()


def _wait_for_live_obs_queue_drain(uid, max_wait_sec=None, empty_grace_sec=0.2, poll_sec=0.05):
    """
    Wait for timer-driven live_obs refresh to consume queued frames before phase switch.
    """
    if not uid:
        return
    with _LIVE_OBS_REFRESH_LOCK:
        state0 = _LIVE_OBS_REFRESH.get(uid)
        queue0 = state0.get("frame_queue") if state0 else None
        initial_qsize = int(queue0.qsize()) if queue0 is not None else 0
    if max_wait_sec is None:
        # Timer-driven playback + small buffer, capped to keep UI responsive.
        max_wait_sec = min(30.0, max(2.0, initial_qsize * (_live_obs_refresh_interval_sec() + 0.02) + 1.0))

    start = time.time()
    empty_since = None
    while True:
        if (time.time() - start) >= max_wait_sec:
            break
        with _LIVE_OBS_REFRESH_LOCK:
            state = _LIVE_OBS_REFRESH.get(uid)
            frame_queue = state.get("frame_queue") if state else None
        if frame_queue is None:
            break
        if frame_queue.qsize() > 0:
            empty_since = None
        else:
            if empty_since is None:
                empty_since = time.time()
            elif (time.time() - empty_since) >= empty_grace_sec:
                break
        time.sleep(poll_sec)


def _prepare_refresh_frame(frame):
    """Normalize cached frame to an RGB uint8 PIL image for gr.Image."""
    if frame is None:
        return None
    frame_arr = np.asarray(frame)
    if frame_arr.dtype != np.uint8:
        max_val = float(np.max(frame_arr)) if frame_arr.size else 0.0
        if max_val <= 1.0:
            frame_arr = (frame_arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame_arr = frame_arr.clip(0, 255).astype(np.uint8)
    if frame_arr.ndim == 2:
        frame_arr = np.stack([frame_arr] * 3, axis=-1)
    elif frame_arr.ndim == 3 and frame_arr.shape[2] == 4:
        frame_arr = frame_arr[:, :, :3]
    return Image.fromarray(frame_arr)


def refresh_live_obs(uid, ui_phase):
    """
    Poll latest cached frame during execute phase.
    Updates live_obs using the configured gr.Timer interval.
    """
    if ui_phase != "execution_playback":
        return gr.update()
    session = get_session(uid)
    if not session:
        return gr.update()

    base_frames = getattr(session, "base_frames", None) or []
    if not base_frames:
        return gr.update()

    _enqueue_live_obs_frames(uid, base_frames)
    state = _get_live_obs_refresh_state(uid, base_count=len(base_frames))
    frame_queue = state["frame_queue"]

    if frame_queue.empty():
        return gr.update()

    latest = frame_queue.get()
    env_id = getattr(session, "env_id", None)
    stitched = concatenate_frames_horizontally([latest], env_id=env_id)
    if stitched:
        latest = stitched[-1]

    img = _prepare_refresh_frame(latest)
    if img is None:
        return gr.update()
    return _live_obs_update(value=img, interactive=False)


def on_video_end_transition(uid):
    """Called when demo video finishes. Transition from video to action phase."""
    return (
        gr.update(visible=False),  # video_phase_group
        gr.update(visible=True),   # action_phase_group
        gr.update(visible=True),   # control_panel_group
        _action_selection_log(),
        gr.update(visible=False, interactive=False),  # watch_demo_video_btn
    )


def _task_load_failed_response(uid, message):
    LOGGER.warning("task_load_failed uid=%s message=%s", _uid_for_log(uid), message)
    return (
        uid,
        gr.update(visible=True),  # main_interface
        _live_obs_update(value=None, interactive=False),  # img_display
        format_log_markdown(message),  # log_output
        gr.update(choices=[], value=None),  # options_radio
        "",  # goal_box
        _ui_text("coords", "not_needed"),  # coords_box
        gr.update(value=None, visible=False),  # video_display
        gr.update(visible=False, interactive=False),  # watch_demo_video_btn
        "",  # task_info_box
        "",  # progress_info_box
        gr.update(interactive=False),  # restart_episode_btn
        gr.update(interactive=False),  # next_task_btn
        gr.update(interactive=False),  # exec_btn
        gr.update(visible=False),  # video_phase_group
        gr.update(visible=False),  # action_phase_group
        gr.update(visible=False),  # control_panel_group
        gr.update(value=""),  # task_hint_display
        gr.update(interactive=False),  # reference_action_btn
    )


def _load_status_task(uid, status):
    """Load status.current_task to session and build the standard UI update tuple."""
    current_task = status.get("current_task") if isinstance(status, dict) else None
    if not current_task:
        return _task_load_failed_response(uid, _ui_text("errors", "load_missing_task"))

    env_id = current_task.get("env_id")
    ep_num = current_task.get("episode_idx")
    if env_id is None or ep_num is None:
        return _task_load_failed_response(uid, _ui_text("errors", "load_invalid_task"))

    try:
        completed_count = int(status.get("completed_count", 0))
    except (TypeError, ValueError):
        completed_count = 0
    progress_text = f"Completed: {completed_count}"
    LOGGER.info(
        "load_status_task uid=%s env=%s episode=%s completed=%s",
        _uid_for_log(uid),
        env_id,
        ep_num,
        completed_count,
    )

    session = get_session(uid)
    if session is None:
        LOGGER.warning("load_status_task missing session uid=%s", _uid_for_log(uid))
        return _task_load_failed_response(uid, _session_error_text())

    LOGGER.debug("loading episode env=%s episode=%s uid=%s", env_id, ep_num, _uid_for_log(uid))

    with _LIVE_OBS_REFRESH_LOCK:
        _LIVE_OBS_REFRESH.pop(uid, None)
    reset_play_button_clicked(uid)
    reset_execute_count(uid, env_id, int(ep_num))

    img, load_msg = session.load_episode(env_id, int(ep_num))
    actual_env_id = getattr(session, "env_id", None) or env_id
    LOGGER.debug(
        "load_episode result uid=%s env=%s episode=%s img_none=%s message=%s",
        _uid_for_log(uid),
        actual_env_id,
        ep_num,
        img is None,
        load_msg,
    )

    if img is not None:
        start_time = datetime.now().isoformat()
        set_task_start_time(uid, env_id, int(ep_num), start_time)

    if img is None:
        set_ui_phase(uid, "executing_task")
        return (
            uid,
            gr.update(visible=True),  # main_interface
            _live_obs_update(value=None, interactive=False),  # img_display
            format_log_markdown(_ui_text("errors", "load_episode_error", load_msg=load_msg)),  # log_output
            gr.update(choices=[], value=None),  # options_radio
            "",  # goal_box
            _ui_text("coords", "not_needed"),  # coords_box
            gr.update(value=None, visible=False),  # video_display
            gr.update(visible=False, interactive=False),  # watch_demo_video_btn
            f"{actual_env_id} (Episode {ep_num})",  # task_info_box
            progress_text,  # progress_info_box
            gr.update(interactive=True),  # restart_episode_btn
            gr.update(interactive=True),  # next_task_btn
            gr.update(interactive=False),  # exec_btn
            gr.update(visible=False),  # video_phase_group
            gr.update(visible=True),  # action_phase_group
            gr.update(visible=True),  # control_panel_group
            gr.update(value=get_task_hint(env_id) if env_id else ""),  # task_hint_display
            gr.update(interactive=False),  # reference_action_btn
        )

    if session.env_id == "VideoPlaceButton" and session.language_goal:
        goal_text = get_videoplacebutton_goal(session.language_goal)
    else:
        goal_text = capitalize_first_letter(session.language_goal) if session.language_goal else ""

    options = session.available_options
    radio_choices = []
    for opt_label, opt_idx in options:
        opt_label = _ui_option_label(session, opt_label, opt_idx)
        if 0 <= opt_idx < len(session.raw_solve_options):
            opt = session.raw_solve_options[opt_idx]
            if opt.get("available"):
                opt_label_with_hint = f"{opt_label}{_ui_text('actions', 'point_required_suffix')}"
            else:
                opt_label_with_hint = opt_label
        else:
            opt_label_with_hint = opt_label
        radio_choices.append((opt_label_with_hint, opt_idx))
    LOGGER.debug(
        "options prepared uid=%s env=%s count=%s",
        _uid_for_log(uid),
        actual_env_id,
        len(radio_choices),
    )

    demo_video_path = None
    should_show = should_show_demo_video(actual_env_id) if actual_env_id else False
    initial_log_msg = _action_selection_log()

    if should_show:
        if session.demonstration_frames:
            try:
                demo_video_path = save_video(session.demonstration_frames, "demo")
                if demo_video_path:
                    file_exists = os.path.exists(demo_video_path)
                    file_size = os.path.getsize(demo_video_path) if file_exists else 0
                    if not (file_exists and file_size > 0):
                        demo_video_path = None
            except Exception:
                demo_video_path = None
        LOGGER.debug(
            "demo video decision uid=%s env=%s should_show=%s has_path=%s",
            _uid_for_log(uid),
            actual_env_id,
            should_show,
            bool(demo_video_path),
        )

    has_demo_video = bool(demo_video_path)
    if has_demo_video:
        initial_log_msg = format_log_markdown(_ui_text("log", "demo_video_prompt"))

    img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)

    if has_demo_video:
        set_ui_phase(uid, "executing_task")

        return (
            uid,
            gr.update(visible=True),  # main_interface
            _live_obs_update(value=img, interactive=False),  # img_display
            initial_log_msg,  # log_output
            gr.update(choices=radio_choices, value=None),  # options_radio
            goal_text,  # goal_box
            _ui_text("coords", "not_needed"),  # coords_box
            gr.update(value=demo_video_path, visible=True),  # video_display
            gr.update(visible=True, interactive=True),  # watch_demo_video_btn
            f"{actual_env_id} (Episode {ep_num})",  # task_info_box
            progress_text,  # progress_info_box
            gr.update(interactive=True),  # restart_episode_btn
            gr.update(interactive=True),  # next_task_btn
            gr.update(interactive=True),  # exec_btn
            gr.update(visible=True),  # video_phase_group
            gr.update(visible=False),  # action_phase_group
            gr.update(visible=False),  # control_panel_group
            gr.update(value=get_task_hint(actual_env_id)),  # task_hint_display
            gr.update(interactive=True),  # reference_action_btn
        )

    set_ui_phase(uid, "executing_task")

    return (
        uid,
        gr.update(visible=True),  # main_interface
        _live_obs_update(value=img, interactive=False),  # img_display
        initial_log_msg,  # log_output
        gr.update(choices=radio_choices, value=None),  # options_radio
        goal_text,  # goal_box
        _ui_text("coords", "not_needed"),  # coords_box
        gr.update(value=None, visible=False),  # video_display (no video)
        gr.update(visible=False, interactive=False),  # watch_demo_video_btn
        f"{actual_env_id} (Episode {ep_num})",  # task_info_box
        progress_text,  # progress_info_box
        gr.update(interactive=True),  # restart_episode_btn
        gr.update(interactive=True),  # next_task_btn
        gr.update(interactive=True),  # exec_btn
        gr.update(visible=False),  # video_phase_group
        gr.update(visible=True),  # action_phase_group
        gr.update(visible=True),  # control_panel_group
        gr.update(value=get_task_hint(actual_env_id)),  # task_hint_display
        gr.update(interactive=True),  # reference_action_btn
    )


def init_session_and_load_task(uid):
    """Initialize the Gradio session and load the current task."""
    if not uid:
        return _task_load_failed_response(uid, _session_error_text())

    if get_session(uid) is None and not try_create_session(uid):
        return _task_load_failed_response(uid, _entry_rejected_text())

    return _load_initialized_session_task(uid)


def _load_initialized_session_task(uid):
    LOGGER.debug("init_session_and_load_task: init_session uid=%s", _uid_for_log(uid))
    success, msg, status = user_manager.init_session(uid)
    LOGGER.debug(
        "init_session_and_load_task result uid=%s success=%s msg=%s",
        _uid_for_log(uid),
        success,
        msg,
    )

    if not success:
        LOGGER.warning("init_session_and_load_task failed uid=%s msg=%s", _uid_for_log(uid), msg)
        return _task_load_failed_response(uid, msg)
    LOGGER.debug("init_session_and_load_task success uid=%s -> load_status_task", _uid_for_log(uid))
    return _load_status_task(uid, status)


def try_init_session_and_load_task(uid):
    """Try to initialize the session and reject immediately when full."""
    if not uid:
        return {
            "status": "ready",
            "load_result": _task_load_failed_response(uid, _session_error_text()),
        }

    if get_session(uid) is None:
        ready = try_create_session(uid)
        if not ready:
            message = _entry_rejected_text()
            LOGGER.info(
                "try_init_session_and_load_task rejected uid=%s",
                _uid_for_log(uid),
            )
            return {
                "status": "rejected",
                "uid": uid,
                "message": message,
                "load_result": _task_load_failed_response(uid, message),
            }

    return {
        "status": "ready",
        "load_result": _load_initialized_session_task(uid),
    }


def load_next_task_wrapper(uid):
    """Move to a random episode within the same env and reload task."""
    if not uid or get_session(uid) is None:
        return _task_load_failed_response(uid, _session_error_text())

    LOGGER.info("load_next_task_wrapper uid=%s", _uid_for_log(uid))
    status = user_manager.next_episode_same_env(uid)
    if not status:
        return _task_load_failed_response(uid, _ui_text("errors", "next_task_failed"))
    return _load_status_task(uid, status)


def restart_episode_wrapper(uid):
    """Reload the current env + episode."""
    if not uid or get_session(uid) is None:
        return _task_load_failed_response(uid, _session_error_text())

    LOGGER.info("restart_episode_wrapper uid=%s", _uid_for_log(uid))
    status = user_manager.get_session_status(uid)
    current_task = status.get("current_task") if isinstance(status, dict) else None
    if not current_task:
        return _task_load_failed_response(uid, _ui_text("errors", "restart_missing_task"))

    env_id = current_task.get("env_id")
    ep_num = current_task.get("episode_idx")
    if env_id is None or ep_num is None:
        return _task_load_failed_response(uid, _ui_text("errors", "restart_invalid_task"))

    return _load_status_task(uid, status)


def switch_env_wrapper(uid, selected_env):
    """Switch env from Current Task dropdown and randomly assign an episode."""
    if not uid or get_session(uid) is None:
        return _task_load_failed_response(uid, _session_error_text())

    LOGGER.info(
        "switch_env_wrapper uid=%s selected_env=%s",
        _uid_for_log(uid),
        selected_env,
    )
    if selected_env:
        status = user_manager.switch_env_and_random_episode(uid, selected_env)
    else:
        status = user_manager.get_session_status(uid)

    if not status:
        return _task_load_failed_response(
            uid,
            _ui_text("errors", "switch_env_failed", selected_env=selected_env),
        )

    return _load_status_task(uid, status)


def on_map_click(uid, option_value, evt: gr.SelectData):
    """
    处理图片点击事件
    """
    session = get_session(uid)
    if not session:
        LOGGER.warning("on_map_click: missing session uid=%s", _uid_for_log(uid))
        return (
            _live_obs_update(value=None, interactive=False),
            _ui_text("coords", "not_needed"),
            format_log_markdown(_session_error_text()),
        )
        
    # Check if current option actually needs coordinates
    needs_coords = _option_requires_coords(session, option_value)
    
    if not needs_coords:
        LOGGER.debug(
            "on_map_click ignored uid=%s option=%s needs_coords=%s",
            _uid_for_log(uid),
            option_value,
            needs_coords,
        )
        # Return current state without changes (or reset to default message if needed, but it should already be there)
        # We return the clean image and the "No need" message to enforce state
        base_img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)
        return _live_obs_update(value=base_img, interactive=False), _ui_text("coords", "not_needed"), _action_selection_log()

    x, y = evt.index[0], evt.index[1]
    LOGGER.debug(
        "on_map_click uid=%s option=%s coords=(%s,%s)",
        _uid_for_log(uid),
        option_value,
        x,
        y,
    )
    
    # Get clean image from session
    base_img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)
    
    # Draw marker
    marked_img = draw_marker(base_img, x, y)
    
    coords_str = f"{x}, {y}"
    
    return _live_obs_update(value=marked_img, interactive=True), coords_str, _action_selection_log()


def _is_valid_coords_text(coords_text: str) -> bool:
    if not isinstance(coords_text, str):
        return False
    text = coords_text.strip()
    if text in {
        "",
        _ui_text("coords", "select_point"),
        _ui_text("coords", "not_needed"),
    }:
        return False
    if "," not in text:
        return False
    try:
        x_raw, y_raw = text.split(",", 1)
        int(x_raw.strip())
        int(y_raw.strip())
    except Exception:
        return False
    return True


def on_option_select(uid, option_value, coords_str=None, suppress_next_option_change=False):
    """
    处理选项选择事件
    """
    default_msg = _ui_text("coords", "not_needed")

    if suppress_next_option_change:
        LOGGER.debug(
            "on_option_select suppressed uid=%s option=%s",
            _uid_for_log(uid),
            option_value,
        )
        return gr.update(), gr.update(), gr.update(), False
    
    if option_value is None:
        LOGGER.debug("on_option_select uid=%s option=None", _uid_for_log(uid))
        session = get_session(uid) if uid else None
        base_img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW) if session else _LIVE_OBS_UPDATE_SKIP
        return default_msg, _live_obs_update(value=base_img, interactive=False), _action_selection_log(), False
    
    session = get_session(uid)
    if not session:
        LOGGER.warning("on_option_select: missing session uid=%s", _uid_for_log(uid))
        return default_msg, _live_obs_update(interactive=False), format_log_markdown(_session_error_text()), False
    
    option_idx = _parse_option_idx(option_value)
    base_img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)

    # Determine coords message
    if _option_requires_coords(session, option_idx):
        LOGGER.debug(
            "on_option_select uid=%s option=%s requires_coords=True valid_coords=%s",
            _uid_for_log(uid),
            option_idx,
            _is_valid_coords_text(coords_str),
        )
        return (
            _ui_text("coords", "select_point"),
            _live_obs_update(value=base_img, interactive=True, waiting_for_point=True),
            _point_selection_log(),
            False,
        )
    
    LOGGER.debug("on_option_select uid=%s option=%s requires_coords=False", _uid_for_log(uid), option_idx)
    return default_msg, _live_obs_update(value=base_img, interactive=False), _action_selection_log(), False


def on_reference_action(uid, current_option_value=None):
    """
    自动获取并回填当前步参考 action + 像素坐标（不执行）。
    """
    session = get_session(uid)
    if not session:
        LOGGER.warning("on_reference_action: missing session uid=%s", _uid_for_log(uid))
        return (
            _live_obs_update(value=None, interactive=False),
            gr.update(),
            _ui_text("coords", "not_needed"),
            format_log_markdown(_session_error_text()),
            False,
        )

    LOGGER.info("on_reference_action uid=%s env=%s", _uid_for_log(uid), getattr(session, "env_id", None))
    current_img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)

    try:
        reference = session.get_reference_action()
    except Exception as exc:
        LOGGER.exception("on_reference_action failed uid=%s", _uid_for_log(uid))
        return (
            _live_obs_update(value=current_img, interactive=False),
            gr.update(),
            gr.update(),
            format_log_markdown(_ui_text("log", "reference_action_error", error=exc)),
            False,
        )

    if not isinstance(reference, dict) or not reference.get("ok", False):
        message = _ui_text("errors", "reference_action_resolve_failed")
        if isinstance(reference, dict) and reference.get("message"):
            message = str(reference.get("message"))
        return (
            _live_obs_update(value=current_img, interactive=False),
            gr.update(),
            gr.update(),
            format_log_markdown(_ui_text("log", "reference_action_status", message=message)),
            False,
        )

    option_idx = reference.get("option_idx")
    current_option_idx = _parse_option_idx(current_option_value)
    option_label = str(reference.get("option_label", "")).strip()
    option_action = str(reference.get("option_action", "")).strip()
    option_action = get_ui_action_text(getattr(session, "env_id", None), option_action)
    need_coords = bool(reference.get("need_coords", False))
    coords_xy = reference.get("coords_xy")
    suppress_next_option_change = option_idx != current_option_idx

    updated_img = current_img
    coords_text = _ui_text("coords", "not_needed")
    log_text = _ui_text(
        "log",
        "reference_action_message",
        option_label=option_label,
        option_action=option_action,
    ).strip()

    if need_coords and isinstance(coords_xy, (list, tuple)) and len(coords_xy) >= 2:
        x = int(coords_xy[0])
        y = int(coords_xy[1])
        updated_img = draw_marker(current_img, x, y)
        coords_text = f"{x}, {y}"
        log_text = _ui_text(
            "log",
            "reference_action_message_with_coords",
            option_label=option_label,
            option_action=option_action,
            coords_text=coords_text,
        )
    LOGGER.debug(
        "on_reference_action resolved uid=%s option_idx=%s need_coords=%s coords=%s",
        _uid_for_log(uid),
        option_idx,
        need_coords,
        coords_xy,
    )

    return (
        _live_obs_update(value=updated_img, interactive=False),
        gr.update(value=option_idx),
        coords_text,
        format_log_markdown(log_text),
        suppress_next_option_change,
    )


def init_app(request: gr.Request):
    """
    处理初始页面加载，直接初始化会话并加载首个任务。

    Args:
        request: Gradio Request 对象，包含查询参数 / Gradio Request object containing query parameters

    Returns:
        初始化后的UI状态
    """
    try:
        uid = getattr(request, "session_hash", None)
        LOGGER.info("init_app: session_hash=%s", _uid_for_log(uid))
        LOGGER.info("init_app: created uid=%s", _uid_for_log(uid))
        result = try_init_session_and_load_task(uid)
        if isinstance(result, dict) and result.get("status") == "ready":
            LOGGER.debug(
                "init_app: init_session_and_load_task returned %s outputs",
                len(result.get("load_result", ()) or ()),
            )
        return result
    except Exception as e:
        LOGGER.exception("init_app exception")
        # Return a safe fallback that hides the loading overlay and shows error
        return {
            "status": "ready",
            "load_result": _task_load_failed_response("", _ui_text("errors", "init_failed", error=e)),
        }


def precheck_execute_inputs(uid, option_idx, coords_str):
    """
    Native precheck for execute action.
    Replaces frontend JS interception by validating inputs server-side before phase switch.
    """
    session = get_session(uid)
    if not session:
        LOGGER.error("precheck_execute_inputs: missing session uid=%s", _uid_for_log(uid))
        raise gr.Error(_session_error_text())

    parsed_option_idx = _parse_option_idx(option_idx)

    if parsed_option_idx is None:
        LOGGER.debug("precheck_execute_inputs uid=%s missing option", _uid_for_log(uid))
        raise gr.Error(_ui_text("log", "execute_missing_action"))

    needs_coords = _option_requires_coords(session, parsed_option_idx)

    if needs_coords and not _is_valid_coords_text(coords_str):
        LOGGER.debug(
            "precheck_execute_inputs uid=%s option=%s requires_coords but coords invalid: %s",
            _uid_for_log(uid),
            parsed_option_idx,
            coords_str,
        )
        raise gr.Error(_ui_text("coords", "select_point_before_execute"))
    LOGGER.debug(
        "precheck_execute_inputs passed uid=%s option=%s needs_coords=%s",
        _uid_for_log(uid),
        parsed_option_idx,
        needs_coords,
    )


def execute_step(uid, option_idx, coords_str):
    LOGGER.info(
        "execute_step start uid=%s option=%s coords=%s",
        _uid_for_log(uid),
        option_idx,
        coords_str,
    )
    session = get_session(uid)
    if not session:
        LOGGER.error("execute_step missing session uid=%s", _uid_for_log(uid))
        return (
            _live_obs_update(value=None, interactive=False),
            format_log_markdown(_session_error_text()),
            gr.update(),
            gr.update(),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
    
    # 检查 execute 次数限制（在执行前检查，如果达到限制则模拟失败状态）
    execute_limit_reached = False
    if uid and session.env_id is not None and session.episode_idx is not None:
        # 从 session 读取 non_demonstration_task_length，如果存在则加上配置的偏移量作为限制，否则不设置限制
        max_execute = None
        if hasattr(session, 'non_demonstration_task_length') and session.non_demonstration_task_length is not None:
            max_execute = session.non_demonstration_task_length + EXECUTE_LIMIT_OFFSET
        
        if max_execute is not None:
            current_count = get_execute_count(uid, session.env_id, session.episode_idx)
            if current_count >= max_execute:
                execute_limit_reached = True
            LOGGER.debug(
                "execute limit check uid=%s env=%s ep=%s current=%s max=%s reached=%s",
                _uid_for_log(uid),
                session.env_id,
                session.episode_idx,
                current_count,
                max_execute,
                execute_limit_reached,
            )
    
    # Ensure at least one cached frame exists for timer-based refresh.
    if not session.base_frames:
        LOGGER.debug("execute_step uid=%s base_frames empty; triggering update_observation", _uid_for_log(uid))
        session.update_observation(use_segmentation=USE_SEGMENTED_VIEW)
    
    option_idx = _parse_option_idx(option_idx)
    if option_idx is None:
        LOGGER.debug("execute_step uid=%s aborted: option_idx is None", _uid_for_log(uid))
        return (
            _live_obs_update(value=session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW), interactive=False),
            format_log_markdown(_ui_text("log", "execute_missing_action")),
            gr.update(),
            gr.update(),
            gr.update(interactive=False),
            gr.update(interactive=True),
        )

    needs_coords = _option_requires_coords(session, option_idx)
    
    # 如果选项需要坐标，检查是否已经点击了图片
    if needs_coords:
        if not _is_valid_coords_text(coords_str):
            LOGGER.debug(
                "execute_step uid=%s option=%s missing valid coords, coords_str=%s",
                _uid_for_log(uid),
                option_idx,
                coords_str,
            )
            current_img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)
            error_msg = _ui_text("coords", "select_point_before_execute")
            return _live_obs_update(value=current_img, interactive=False), format_log_markdown(error_msg), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

    # Parse coords
    click_coords = None
    if coords_str and "," in coords_str:
        try:
            parts = coords_str.split(",")
            click_coords = (int(parts[0].strip()), int(parts[1].strip()))
        except:
            pass
    
    # Execute
    # 如果达到 execute 次数限制，模拟失败状态（使用和任务失败一样的机制）
    if execute_limit_reached:
        # 获取选项标签用于状态消息
        option_label = None
        if session.available_options:
            for label, idx in session.available_options:
                if idx == option_idx:
                    option_label = _ui_option_label(session, label, idx)
                    break
        
        # 模拟失败状态，使用和 oracle_logic.py 中任务失败一样的格式
        status = f"Executing: {option_label or 'Action'}"
        status += " | FAILED"  # 和任务失败一样的格式
        done = True  # 设置为完成，触发任务完成流程
        
        # 获取当前图片
        img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)
        
        # 增加 execute 计数（因为这也算一次尝试）
        if uid and session.env_id is not None and session.episode_idx is not None:
            new_count = increment_execute_count(uid, session.env_id, session.episode_idx)
            LOGGER.warning(
                "execute limit reached uid=%s env=%s ep=%s count=%s",
                _uid_for_log(uid),
                session.env_id,
                session.episode_idx,
                new_count,
            )
    else:
        # 正常执行
        # 异常处理：所有异常（ScrewPlanFailure 和其他执行错误）都会显示弹窗通知
        LOGGER.info(
            "executing action uid=%s env=%s ep=%s option=%s coords=%s",
            _uid_for_log(uid),
            getattr(session, "env_id", None),
            getattr(session, "episode_idx", None),
            option_idx,
            click_coords,
        )
        try:
            img, status, done = session.execute_action(option_idx, click_coords)
        except ScrewPlanFailureError as e:
            # 捕获 screw plan 失败异常，显示弹窗通知
            error_message = str(e)
            gr.Info(f"Robot cannot reach position, Refresh the page and try again.")
            # 返回当前状态，在状态消息中显示错误信息
            current_img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)
            status = f"Screw plan failed: {error_message}"
            done = False
            # 继续正常返回流程
            img = current_img
            LOGGER.warning("execute_step screw_plan_failure uid=%s error=%s", _uid_for_log(uid), error_message)
        except RuntimeError as e:
            # 捕获所有其他执行错误，显示弹窗通知
            error_message = str(e)
            gr.Info(f"Cannot find suitable target, Refresh the page and try again.")
            # 返回当前状态，在状态消息中显示错误信息
            current_img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)
            status = f"Error: {error_message}"
            done = False
            # 继续正常返回流程
            img = current_img
            LOGGER.warning("execute_step runtime_error uid=%s error=%s", _uid_for_log(uid), error_message)
        
        # 增加 execute 计数（无论成功或失败都计数，因为用户已经执行了一次操作）
        if uid and session.env_id is not None and session.episode_idx is not None:
            new_count = increment_execute_count(uid, session.env_id, session.episode_idx)
            LOGGER.debug(
                "execute count incremented uid=%s env=%s ep=%s count=%s",
                _uid_for_log(uid),
                session.env_id,
                session.episode_idx,
                new_count,
            )

    # Execute frames are produced in batch when execute_action returns from worker process.
    # Enqueue them now, then wait briefly for the configured timer to drain FIFO playback.
    _enqueue_live_obs_frames(uid, getattr(session, "base_frames", None))
    _wait_for_live_obs_queue_drain(uid)
    LOGGER.debug("execute_step playback drain complete uid=%s", _uid_for_log(uid))
    
    # 注意：执行阶段画面由 live_obs 的配置化轮询间隔刷新。
    
    progress_update = gr.update()  # 默认不更新 progress
    task_update = gr.update()
    
    if done:
        # 确定最终状态用于日志记录
        final_log_status = "failed"
        if "SUCCESS" in status:
            final_log_status = "success"
        
        # Episode完成时，格式化System Log的状态消息
        # 使用固定模板，所有行长度一致（32个字符），无空行
        if final_log_status == "success":
            status = _ui_text("log", "episode_success_banner")
        else:
            status = _ui_text("log", "episode_failed_banner")

        # 更新累计完成计数，不再推进固定任务索引
        if uid:
            seed = getattr(session, 'seed', None)
            user_status = user_manager.complete_current_task(
                uid,
                env_id=session.env_id,
                episode_idx=session.episode_idx,
                status=final_log_status,
                difficulty=session.difficulty if hasattr(session, 'difficulty') and session.difficulty is not None else None,
                language_goal=session.language_goal,
                seed=seed
            )
            if user_status:
                completed_count = user_status.get("completed_count", 0)
                task_update = f"{session.env_id} (Episode {session.episode_idx})"
                progress_update = f"Completed: {completed_count}"
                LOGGER.info(
                    "task complete uid=%s env=%s ep=%s final=%s completed_count=%s",
                    _uid_for_log(uid),
                    session.env_id,
                    session.episode_idx,
                    final_log_status,
                    completed_count,
                )
    
    # 根据视图模式重新获取图片
    img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)
        
    restart_episode_update = gr.update(interactive=True)
    next_task_update = gr.update(interactive=True)
    exec_btn_update = gr.update(interactive=False) if done else gr.update(interactive=True)
    
    # 格式化日志消息为 HTML 格式（支持颜色显示）
    formatted_status = format_log_markdown(status)
    LOGGER.debug(
        "execute_step done uid=%s env=%s ep=%s done=%s exec_btn_interactive=%s",
        _uid_for_log(uid),
        getattr(session, "env_id", None),
        getattr(session, "episode_idx", None),
        done,
        not done,
    )
    
    return (
        _live_obs_update(value=img, interactive=False),
        formatted_status,
        task_update,
        progress_update,
        restart_episode_update,
        next_task_update,
        exec_btn_update,
    )
