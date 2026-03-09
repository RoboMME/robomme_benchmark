"""
Gradio回调函数模块
响应UI事件，调用业务逻辑，返回UI更新
"""
import logging
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np

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
    UI_TEXT,
    USE_SEGMENTED_VIEW,
    get_live_obs_elem_classes,
    get_ui_action_text,
    should_show_demo_video,
)
from process_session import ScrewPlanFailureError
from note_content import get_task_hint


# --- execute video temp files ---
_EXECUTION_VIDEO_PATHS = {}
_EXECUTION_VIDEO_LOCK = threading.Lock()
LOGGER = logging.getLogger("robomme.callbacks")


def _ui_text(section, key, **kwargs):
    template = UI_TEXT[section][key]
    return template.format(**kwargs) if kwargs else template


_LIVE_OBS_UPDATE_SKIP = object()


def _action_selection_log():
    return format_log_markdown(_ui_text("log", "action_selection_prompt"))


def _point_selection_log():
    return format_log_markdown(_ui_text("log", "point_selection_prompt"))


def _get_raw_option_label(session, option_idx):
    try:
        option_index = int(option_idx)
    except (TypeError, ValueError):
        return None

    raw_solve_options = getattr(session, "raw_solve_options", None)
    if not isinstance(raw_solve_options, list):
        return None
    if not (0 <= option_index < len(raw_solve_options)):
        return None

    raw_option = raw_solve_options[option_index]
    if not isinstance(raw_option, dict):
        return None

    label = str(raw_option.get("label", "")).strip()
    return label or None


def _execution_video_log(session, option_idx, fallback_status=None):
    label = _get_raw_option_label(session, option_idx)
    if label:
        return format_log_markdown(_ui_text("log", "execute_action_prompt", label=label))
    if fallback_status is None:
        return None
    return format_log_markdown(fallback_status)


def _default_post_execute_log_state():
    return {
        "preserve_terminal_log": False,
        "terminal_log_value": None,
        "preserve_execute_video_log": False,
        "execute_video_log_value": None,
    }


def _normalize_post_execute_log_state(state):
    """Normalize terminal-log preservation payloads across callback boundaries."""
    if isinstance(state, dict):
        preserve_terminal_log = bool(state.get("preserve_terminal_log", False))
        terminal_log_value = state.get("terminal_log_value")
        if terminal_log_value is None:
            preserve_terminal_log = False
        else:
            terminal_log_value = str(terminal_log_value)
        preserve_execute_video_log = bool(state.get("preserve_execute_video_log", False))
        execute_video_log_value = state.get("execute_video_log_value")
        if execute_video_log_value is None:
            preserve_execute_video_log = False
        else:
            execute_video_log_value = str(execute_video_log_value)
        return {
            "preserve_terminal_log": preserve_terminal_log,
            "terminal_log_value": terminal_log_value,
            "preserve_execute_video_log": preserve_execute_video_log,
            "execute_video_log_value": execute_video_log_value,
        }
    return _default_post_execute_log_state()


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
    _clear_execution_video_path(uid)
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


def _uid_for_log(uid):
    if not uid:
        return "<none>"
    text = str(uid)
    return text if len(text) <= 12 else f"{text[:8]}..."


def _delete_temp_video(path):
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        LOGGER.warning("failed to delete temp video: %s", path, exc_info=True)


def _clear_execution_video_path(uid):
    if not uid:
        return
    with _EXECUTION_VIDEO_LOCK:
        old_path = _EXECUTION_VIDEO_PATHS.pop(uid, None)
    _delete_temp_video(old_path)


def _set_execution_video_path(uid, path):
    if not uid:
        return
    with _EXECUTION_VIDEO_LOCK:
        old_path = _EXECUTION_VIDEO_PATHS.get(uid)
        _EXECUTION_VIDEO_PATHS[uid] = path
    if old_path and old_path != path:
        _delete_temp_video(old_path)


def _build_radio_choices(session):
    radio_choices = []
    options = getattr(session, "available_options", None) or []
    raw_solve_options = getattr(session, "raw_solve_options", None) or []
    for opt_label, opt_idx in options:
        ui_label = _ui_option_label(session, opt_label, opt_idx)
        if 0 <= opt_idx < len(raw_solve_options) and raw_solve_options[opt_idx].get("available"):
            ui_label = f"{ui_label}{_ui_text('actions', 'point_required_suffix')}"
        radio_choices.append((ui_label, opt_idx))
    return radio_choices


def _coerce_video_source_frames(frames):
    if not isinstance(frames, list):
        return []
    valid = []
    for frame in frames:
        if frame is None:
            continue
        frame_arr = np.asarray(frame)
        if frame_arr.ndim not in {2, 3}:
            continue
        if frame_arr.dtype.kind in {"U", "S", "O"}:
            continue
        valid.append(frame_arr)
    return valid


def _fallback_execution_frames(session):
    base_frames = getattr(session, "base_frames", None) or []
    if base_frames:
        return [np.asarray(base_frames[-1])]
    try:
        pil_image = session.get_pil_image(use_segmented=False)
    except Exception:
        return []
    if pil_image is None:
        return []
    frame_arr = np.asarray(pil_image)
    if frame_arr.ndim not in {2, 3}:
        return []
    if frame_arr.dtype.kind in {"U", "S", "O"}:
        return []
    return [frame_arr]


def _build_execution_video_update(uid, session):
    raw_frames = _coerce_video_source_frames(getattr(session, "last_execution_frames", None))
    if not raw_frames:
        raw_frames = _fallback_execution_frames(session)
    stitched_frames = concatenate_frames_horizontally(
        raw_frames,
        env_id=getattr(session, "env_id", None),
    )
    if not stitched_frames:
        _clear_execution_video_path(uid)
        return gr.update(value=None, visible=False)

    suffix = f"execute_{int(time.time() * 1000)}"
    video_path = save_video(stitched_frames, suffix=suffix)
    if not video_path:
        _clear_execution_video_path(uid)
        return gr.update(value=None, visible=False)
    if not (os.path.exists(video_path) and os.path.getsize(video_path) > 0):
        _clear_execution_video_path(uid)
        return gr.update(value=None, visible=False)

    _set_execution_video_path(uid, video_path)
    return gr.update(
        value=video_path,
        visible=True,
        autoplay=True,
        playback_position=0,
    )


def capitalize_first_letter(text: str) -> str:
    """确保字符串的第一个字母大写，其余字符保持不变"""
    if not text:
        return text
    if len(text) == 1:
        return text.upper()
    return text[0].upper() + text[1:]


def _format_choice_prefix(text: str) -> str:
    """Display multi-choice labels like a/b/c/d as uppercase without changing action text."""
    if not isinstance(text, str):
        return text

    stripped = text.strip()
    if not stripped:
        return stripped

    prefix, dot, rest = stripped.partition(".")
    if dot and prefix.isalpha() and len(prefix) <= 4:
        return f"{prefix.upper()}.{rest}"
    if stripped.isalpha() and len(stripped) <= 4:
        return stripped.upper()
    return stripped




def _ui_option_label(session, opt_label: str, opt_idx: int) -> str:
    """
    仅在 Gradio UI 层对选项显示文案做覆盖（不改底层 env/options 生成逻辑）。
    优先使用 raw_solve_options 中的原始 label/action 重新组装显示文本，
    并按 env_id 做 display-only action 文案映射。
    """
    try:
        option_index = int(opt_idx)
    except (TypeError, ValueError):
        return _format_choice_prefix(opt_label)

    raw_solve_options = getattr(session, "raw_solve_options", None)
    if not isinstance(raw_solve_options, list):
        return _format_choice_prefix(opt_label)
    if not (0 <= option_index < len(raw_solve_options)):
        return _format_choice_prefix(opt_label)

    raw_option = raw_solve_options[option_index]
    if not isinstance(raw_option, dict):
        return _format_choice_prefix(opt_label)

    raw_label = _format_choice_prefix(str(raw_option.get("label", "")).strip())
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
    """Disable controls and point clicking while execute work is running."""
    if uid:
        LOGGER.debug("switch_to_execute_phase uid=%s", _uid_for_log(uid))
    return (
        gr.update(interactive=False),  # options_radio
        gr.update(interactive=False),  # exec_btn
        gr.update(interactive=False),  # restart_episode_btn
        gr.update(interactive=False),  # next_task_btn
        _live_obs_update(interactive=False),  # img_display
        gr.update(interactive=False),  # reference_action_btn
        gr.update(interactive=False),  # task_hint_display
    )


def switch_to_action_phase(uid=None):
    """Switch display to action phase and restore control panel interactions."""
    if uid:
        LOGGER.debug("switch_to_action_phase uid=%s", _uid_for_log(uid))
    return (
        gr.update(interactive=True),  # options_radio
        gr.update(),  # exec_btn (keep execute_step result)
        gr.update(),  # restart_episode_btn (keep execute_step result)
        gr.update(),  # next_task_btn (keep execute_step result)
        _live_obs_update(interactive=True),  # img_display
        gr.update(interactive=True),  # reference_action_btn
    )


def on_demo_video_end_transition(uid, ui_phase=None):
    """Transition from demo video phase back to the action phase."""
    LOGGER.debug(
        "on_demo_video_end_transition uid=%s ui_phase=%s",
        _uid_for_log(uid),
        ui_phase,
    )
    return (
        gr.update(visible=False),  # video_phase_group
        gr.update(visible=True),   # action_phase_group
        gr.update(visible=True),   # control_panel_group
        _action_selection_log(),  # log_output
        gr.update(visible=False, interactive=False),  # watch_demo_video_btn
        "action_point",  # ui_phase_state
    )


def on_video_end_transition(uid, ui_phase=None):
    """Backward-compatible alias for legacy tests and demo end handling."""
    return on_demo_video_end_transition(uid, ui_phase)


def _normalize_post_execute_controls_state(state):
    """Normalize legacy bool and dict payloads for execute-video exit transitions."""
    if isinstance(state, dict):
        exec_interactive = bool(state.get("exec_btn_interactive", True))
        reference_interactive = bool(state.get("reference_action_interactive", True))
        return {
            "exec_btn_interactive": exec_interactive,
            "reference_action_interactive": reference_interactive,
        }
    legacy_exec_interactive = bool(state)
    return {
        "exec_btn_interactive": legacy_exec_interactive,
        "reference_action_interactive": legacy_exec_interactive,
    }


def on_execute_video_end_transition(
    uid,
    post_execute_controls_state=True,
    post_execute_log_state=None,
):
    """Transition from execute video phase back to the action phase."""
    controls_state = _normalize_post_execute_controls_state(post_execute_controls_state)
    log_state = _normalize_post_execute_log_state(post_execute_log_state)
    next_log_state = _default_post_execute_log_state()
    log_update = gr.update(value=_action_selection_log())
    if log_state["preserve_terminal_log"]:
        log_update = gr.update(value=log_state["terminal_log_value"])
        next_log_state = log_state
    LOGGER.debug(
        "on_execute_video_end_transition uid=%s controls_state=%s log_state=%s",
        _uid_for_log(uid),
        controls_state,
        log_state,
    )
    return (
        gr.update(visible=False),  # execution_video_group
        gr.update(visible=True),   # action_phase_group
        gr.update(visible=True),   # control_panel_group
        gr.update(interactive=True),  # options_radio
        gr.update(interactive=controls_state["exec_btn_interactive"]),  # exec_btn
        gr.update(interactive=True),  # restart_episode_btn
        gr.update(interactive=True),  # next_task_btn
        _live_obs_update(interactive=False),  # img_display
        log_update,  # log_output
        gr.update(interactive=controls_state["reference_action_interactive"]),  # reference_action_btn
        gr.update(interactive=True),  # task_hint_display
        next_log_state,  # post_execute_log_state
        "action_point",  # ui_phase_state
    )


def _task_load_failed_response(uid, message):
    LOGGER.warning("task_load_failed uid=%s message=%s", _uid_for_log(uid), message)
    _clear_execution_video_path(uid)
    return (
        uid,
        gr.update(visible=True),  # main_interface
        _live_obs_update(value=None, interactive=False),  # img_display
        format_log_markdown(message),  # log_output
        gr.update(choices=[], value=None),  # options_radio
        "",  # goal_box
        _ui_text("coords", "not_needed"),  # coords_box
        gr.update(value=None, visible=False, autoplay=False, playback_position=0),  # video_display
        gr.update(value=None, visible=False, playback_position=0),  # execute_video_display
        gr.update(visible=False, interactive=False),  # watch_demo_video_btn
        "",  # task_info_box
        "",  # progress_info_box
        gr.update(interactive=False),  # restart_episode_btn
        gr.update(interactive=False),  # next_task_btn
        gr.update(interactive=False),  # exec_btn
        gr.update(visible=False),  # video_phase_group
        gr.update(visible=False),  # execution_video_group
        gr.update(visible=False),  # action_phase_group
        gr.update(visible=False),  # control_panel_group
        gr.update(value=""),  # task_hint_display
        gr.update(interactive=False),  # reference_action_btn
        _default_post_execute_log_state(),  # post_execute_log_state
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

    _clear_execution_video_path(uid)
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
            gr.update(value=None, visible=False, autoplay=False, playback_position=0),  # video_display
            gr.update(value=None, visible=False, playback_position=0),  # execute_video_display
            gr.update(visible=False, interactive=False),  # watch_demo_video_btn
            f"{actual_env_id} (Episode {ep_num})",  # task_info_box
            progress_text,  # progress_info_box
            gr.update(interactive=True),  # restart_episode_btn
            gr.update(interactive=True),  # next_task_btn
            gr.update(interactive=False),  # exec_btn
            gr.update(visible=False),  # video_phase_group
            gr.update(visible=False),  # execution_video_group
            gr.update(visible=True),  # action_phase_group
            gr.update(visible=True),  # control_panel_group
            gr.update(value=get_task_hint(env_id) if env_id else ""),  # task_hint_display
            gr.update(interactive=False),  # reference_action_btn
            _default_post_execute_log_state(),  # post_execute_log_state
        )

    goal_text = capitalize_first_letter(session.language_goal) if session.language_goal else ""

    radio_choices = _build_radio_choices(session)
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
            gr.update(value=demo_video_path, visible=True, autoplay=False, playback_position=0),  # video_display
            gr.update(value=None, visible=False, playback_position=0),  # execute_video_display
            gr.update(visible=True, interactive=True),  # watch_demo_video_btn
            f"{actual_env_id} (Episode {ep_num})",  # task_info_box
            progress_text,  # progress_info_box
            gr.update(interactive=True),  # restart_episode_btn
            gr.update(interactive=True),  # next_task_btn
            gr.update(interactive=True),  # exec_btn
            gr.update(visible=True),  # video_phase_group
            gr.update(visible=False),  # execution_video_group
            gr.update(visible=False),  # action_phase_group
            gr.update(visible=False),  # control_panel_group
            gr.update(value=get_task_hint(actual_env_id)),  # task_hint_display
            gr.update(interactive=True),  # reference_action_btn
            _default_post_execute_log_state(),  # post_execute_log_state
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
        gr.update(value=None, visible=False, autoplay=False, playback_position=0),  # video_display (no video)
        gr.update(value=None, visible=False, playback_position=0),  # execute_video_display
        gr.update(visible=False, interactive=False),  # watch_demo_video_btn
        f"{actual_env_id} (Episode {ep_num})",  # task_info_box
        progress_text,  # progress_info_box
        gr.update(interactive=True),  # restart_episode_btn
        gr.update(interactive=True),  # next_task_btn
        gr.update(interactive=True),  # exec_btn
        gr.update(visible=False),  # video_phase_group
        gr.update(visible=False),  # execution_video_group
        gr.update(visible=True),  # action_phase_group
        gr.update(visible=True),  # control_panel_group
        gr.update(value=get_task_hint(actual_env_id)),  # task_hint_display
        gr.update(interactive=True),  # reference_action_btn
        _default_post_execute_log_state(),  # post_execute_log_state
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


def on_option_select(
    uid,
    option_value,
    coords_str=None,
    suppress_next_option_change=False,
    post_execute_log_state=None,
):
    """
    处理选项选择事件
    """
    default_msg = _ui_text("coords", "not_needed")
    normalized_post_execute_log_state = _normalize_post_execute_log_state(post_execute_log_state)

    if suppress_next_option_change:
        LOGGER.debug(
            "on_option_select suppressed uid=%s option=%s",
            _uid_for_log(uid),
            option_value,
        )
        return gr.update(), gr.update(), gr.update(), False, normalized_post_execute_log_state

    if normalized_post_execute_log_state["preserve_terminal_log"]:
        LOGGER.debug(
            "on_option_select preserving terminal log uid=%s option=%s",
            _uid_for_log(uid),
            option_value,
        )
        return (
            gr.update(),
            gr.update(),
            gr.update(value=normalized_post_execute_log_state["terminal_log_value"]),
            False,
            normalized_post_execute_log_state,
        )

    if normalized_post_execute_log_state["preserve_execute_video_log"]:
        LOGGER.debug(
            "on_option_select preserving execution-video log uid=%s option=%s",
            _uid_for_log(uid),
            option_value,
        )
        return (
            gr.update(),
            gr.update(),
            gr.update(value=normalized_post_execute_log_state["execute_video_log_value"]),
            False,
            normalized_post_execute_log_state,
        )
    
    if option_value is None:
        LOGGER.debug("on_option_select uid=%s option=None", _uid_for_log(uid))
        session = get_session(uid) if uid else None
        base_img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW) if session else _LIVE_OBS_UPDATE_SKIP
        return (
            default_msg,
            _live_obs_update(value=base_img, interactive=False),
            _action_selection_log(),
            False,
            normalized_post_execute_log_state,
        )
    
    session = get_session(uid)
    if not session:
        LOGGER.warning("on_option_select: missing session uid=%s", _uid_for_log(uid))
        return (
            default_msg,
            _live_obs_update(interactive=False),
            format_log_markdown(_session_error_text()),
            False,
            normalized_post_execute_log_state,
        )
    
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
            normalized_post_execute_log_state,
        )
    
    LOGGER.debug("on_option_select uid=%s option=%s requires_coords=False", _uid_for_log(uid), option_idx)
    return (
        default_msg,
        _live_obs_update(value=base_img, interactive=False),
        _action_selection_log(),
        False,
        normalized_post_execute_log_state,
    )


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

    def _response(
        *,
        img_update,
        log_update,
        task_update=gr.update(),
        progress_update=gr.update(),
        restart_update=gr.update(interactive=True),
        next_update=gr.update(interactive=True),
        exec_update=gr.update(interactive=True),
        execute_video_update=None,
        options_update=gr.update(interactive=True),
        coords_update=None,
        reference_update=gr.update(interactive=True),
        task_hint_update=gr.update(interactive=True),
        post_execute_controls_state=None,
        post_execute_log_state=None,
        show_execution_video=False,
        ui_phase="action_point",
    ):
        if execute_video_update is None:
            execute_video_update = gr.update(value=None, visible=False, playback_position=0)
        if coords_update is None:
            coords_update = _ui_text("coords", "not_needed")
        if post_execute_controls_state is None:
            post_execute_controls_state = {
                "exec_btn_interactive": True,
                "reference_action_interactive": True,
            }
        normalized_post_execute_controls_state = _normalize_post_execute_controls_state(
            post_execute_controls_state
        )
        normalized_post_execute_log_state = _normalize_post_execute_log_state(post_execute_log_state)
        return (
            img_update,
            log_update,
            task_update,
            progress_update,
            restart_update,
            next_update,
            exec_update,
            execute_video_update,
            gr.update(visible=not show_execution_video),  # action_phase_group
            gr.update(visible=True),  # control_panel_group
            gr.update(visible=show_execution_video),  # execution_video_group
            options_update,
            coords_update,
            reference_update,
            task_hint_update,
            normalized_post_execute_controls_state,
            normalized_post_execute_log_state,
            ui_phase,
        )

    session = get_session(uid)
    if not session:
        LOGGER.error("execute_step missing session uid=%s", _uid_for_log(uid))
        return _response(
            img_update=_live_obs_update(value=None, interactive=False),
            log_update=format_log_markdown(_session_error_text()),
            restart_update=gr.update(interactive=False),
            next_update=gr.update(interactive=False),
            exec_update=gr.update(interactive=False),
            options_update=gr.update(interactive=False),
            reference_update=gr.update(interactive=False),
            task_hint_update=gr.update(interactive=False),
            post_execute_controls_state={
                "exec_btn_interactive": False,
                "reference_action_interactive": False,
            },
            post_execute_log_state=_default_post_execute_log_state(),
            show_execution_video=False,
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

    # Ensure at least one cached frame exists for fallback clip generation.
    if not session.base_frames:
        LOGGER.debug("execute_step uid=%s base_frames empty; triggering update_observation", _uid_for_log(uid))
        session.update_observation(use_segmentation=USE_SEGMENTED_VIEW)
    if hasattr(session, "last_execution_frames"):
        session.last_execution_frames = []

    option_idx = _parse_option_idx(option_idx)
    if option_idx is None:
        LOGGER.debug("execute_step uid=%s aborted: option_idx is None", _uid_for_log(uid))
        return _response(
            img_update=_live_obs_update(value=session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW), interactive=False),
            log_update=format_log_markdown(_ui_text("log", "execute_missing_action")),
            exec_update=gr.update(interactive=True),
            options_update=gr.update(choices=_build_radio_choices(session), value=None, interactive=True),
            reference_update=gr.update(interactive=True),
            task_hint_update=gr.update(interactive=True),
            post_execute_controls_state={
                "exec_btn_interactive": True,
                "reference_action_interactive": True,
            },
            post_execute_log_state=_default_post_execute_log_state(),
            show_execution_video=False,
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
            return _response(
                img_update=_live_obs_update(value=current_img, interactive=False),
                log_update=format_log_markdown(error_msg),
                exec_update=gr.update(interactive=True),
                options_update=gr.update(choices=_build_radio_choices(session), value=None, interactive=True),
                coords_update=coords_str,
                reference_update=gr.update(interactive=True),
                task_hint_update=gr.update(interactive=True),
                post_execute_controls_state={
                    "exec_btn_interactive": True,
                    "reference_action_interactive": True,
                },
                post_execute_log_state=_default_post_execute_log_state(),
                show_execution_video=False,
            )

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
    execute_video_update = _build_execution_video_update(uid, session)
    show_execution_video = execute_video_update.get("visible") is True
    radio_choices = _build_radio_choices(session)
    post_execute_controls_state = {
        "exec_btn_interactive": not done,
        "reference_action_interactive": not done,
    }
    post_execute_log_state = _default_post_execute_log_state()
    restart_episode_update = gr.update(interactive=False) if show_execution_video else gr.update(interactive=True)
    next_task_update = gr.update(interactive=False) if show_execution_video else gr.update(interactive=True)
    exec_btn_update = (
        gr.update(interactive=False)
        if show_execution_video
        else gr.update(interactive=post_execute_controls_state["exec_btn_interactive"])
    )
    options_update = gr.update(
        choices=radio_choices,
        value=None,
        interactive=not show_execution_video,
    )
    coords_update = _ui_text("coords", "not_needed")
    reference_update = gr.update(interactive=not show_execution_video)
    task_hint_update = gr.update(interactive=not show_execution_video)
    
    # 格式化日志消息为 HTML 格式（支持颜色显示）
    formatted_status = format_log_markdown(status)
    if show_execution_video and not done:
        formatted_status = _execution_video_log(session, option_idx, fallback_status=status) or formatted_status
        post_execute_log_state = {
            "preserve_terminal_log": False,
            "terminal_log_value": None,
            "preserve_execute_video_log": True,
            "execute_video_log_value": formatted_status,
        }
    if done:
        post_execute_log_state = {
            "preserve_terminal_log": True,
            "terminal_log_value": formatted_status,
            "preserve_execute_video_log": False,
            "execute_video_log_value": None,
        }
    LOGGER.debug(
        "execute_step done uid=%s env=%s ep=%s done=%s post_execute_controls=%s post_execute_log=%s show_execution_video=%s",
        _uid_for_log(uid),
        getattr(session, "env_id", None),
        getattr(session, "episode_idx", None),
        done,
        post_execute_controls_state,
        post_execute_log_state,
        show_execution_video,
    )

    return _response(
        img_update=_live_obs_update(value=img, interactive=False),
        log_update=formatted_status,
        task_update=task_update,
        progress_update=progress_update,
        restart_update=restart_episode_update,
        next_update=next_task_update,
        exec_update=exec_btn_update,
        execute_video_update=execute_video_update,
        options_update=options_update,
        coords_update=coords_update,
        reference_update=reference_update,
        task_hint_update=task_hint_update,
        post_execute_controls_state=post_execute_controls_state,
        post_execute_log_state=post_execute_log_state,
        show_execution_video=show_execution_video,
        ui_phase="execution_video" if show_execution_video else "action_point",
    )
