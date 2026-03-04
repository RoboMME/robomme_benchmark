"""
Gradio回调函数模块
响应UI事件，调用业务逻辑，返回UI更新
"""
import gradio as gr
import numpy as np
import time
import threading
import queue
import os
import re
from datetime import datetime
from PIL import Image
from state_manager import (
    get_session,
    create_session,
    set_ui_phase,
    reset_ui_phase,
    get_execute_count,
    increment_execute_count,
    reset_execute_count,
    set_task_start_time,
    update_session_activity,
    get_session_activity,
    cleanup_session,
    reset_play_button_clicked,
    GLOBAL_SESSIONS,
    SESSION_LAST_ACTIVITY,
    _state_lock,
)
from image_utils import draw_marker, save_video, concatenate_frames_horizontally
from user_manager import user_manager
from config import USE_SEGMENTED_VIEW, should_show_demo_video, SESSION_TIMEOUT, EXECUTE_LIMIT_OFFSET
from process_session import ScrewPlanFailureError, ProcessSessionProxy
from note_content import get_task_hint


# --- live_obs refresh queue state ---
# Each uid keeps its own FIFO queue and sampling cursor.
_LIVE_OBS_REFRESH = {}
_LIVE_OBS_REFRESH_LOCK = threading.Lock()


def capitalize_first_letter(text: str) -> str:
    """确保字符串的第一个字母大写，其余字符保持不变"""
    if not text:
        return text
    if len(text) == 1:
        return text.upper()
    return text[0].upper() + text[1:]


def get_videoplacebutton_goal(original_goal: str) -> str:
    """
    为 VideoPlaceButton 任务构造新的任务目标
    匹配 "cube on the target" 并替换为新的目标格式
    """
    if not original_goal:
        return ""
    
    original_lower = original_goal.lower()
    
    # 匹配 "cube on the target" 并替换
    if "cube on the target" in original_lower:
        # 使用正则表达式进行不区分大小写的替换
        pattern = re.compile(re.escape("cube on the target"), re.IGNORECASE)
        new_goal = pattern.sub("cube on the target that it was previously placed on", original_goal)
        return capitalize_first_letter(new_goal)
    else:
        # 如果无法匹配，保持原始任务目标不变
        return capitalize_first_letter(original_goal)


def _ui_option_label(session, opt_label: str, opt_idx: int) -> str:
    """
    仅在 Gradio UI 层对选项显示文案做覆盖（不改底层 env/options 生成逻辑）。
    目前只对 RouteStick 任务把 4 个长句 label 显示为短 label。
    """
    env_id = getattr(session, "env_id", None)
    if env_id == "RouteStick":
        routestick_map = {
            0: "move left clockwise",
            1: "move right clockwise",
            2: "move left counterclockwise",
            3: "move right counterclockwise",
        }
        return routestick_map.get(int(opt_idx), opt_label)
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


def show_loading_info():
    """
    显示加载环境的全屏遮罩层提示信息
    
    功能说明：
    - 此函数在用户点击登录/加载任务等按钮时被调用
    - 返回包含全屏遮罩层的 HTML 字符串，用于显示加载提示
    - 遮罩层会覆盖整个页面，防止用户在加载过程中进行其他操作
    - 加载完成后，回调函数会返回空字符串 "" 来清空 loading_overlay 组件，从而隐藏遮罩层
    
    工作流程：
    1. 用户点击按钮（如 Login、Next Task 等）
    2. 按钮的 click 事件首先调用此函数，显示遮罩层
    3. 然后通过 .then() 链式调用实际的加载函数（如 login_and_load_task）
    4. 加载函数执行完成后，返回 gr.update(visible=False) 隐藏遮罩层

    Returns:
        gr.update: 显示 loading overlay group
    """
    return gr.update(visible=True)


def on_video_end(uid):
    """
    Called when the demonstration video finishes playing.
    Updates the system log to prompt for action selection.
    """
    return format_log_markdown("please select the action below 👇🏻,\nsome actions also need to select keypoint")


def switch_to_execute_phase(uid):
    """Disable controls and keypoint clicking during execute playback."""
    if uid:
        session = get_session(uid)
        base_count = len(getattr(session, "base_frames", []) or []) if session else 0
        with _LIVE_OBS_REFRESH_LOCK:
            _LIVE_OBS_REFRESH[uid] = {
                "frame_queue": queue.Queue(),
                "last_base_count": base_count,
                "take_next": True,  # downsample x2 by enqueueing every other frame
            }
    return (
        gr.update(interactive=False),  # options_radio
        gr.update(interactive=False),  # exec_btn
        gr.update(interactive=False),  # restart_episode_btn
        gr.update(interactive=False),  # next_task_btn
        gr.update(interactive=False),  # img_display
        gr.update(interactive=False),  # reference_action_btn
    )


def switch_to_action_phase(uid=None):
    """Switch display to action phase and restore control panel interactions."""
    if uid:
        with _LIVE_OBS_REFRESH_LOCK:
            _LIVE_OBS_REFRESH.pop(uid, None)
    return (
        gr.update(interactive=True),  # options_radio
        gr.update(),  # exec_btn (keep execute_step result)
        gr.update(),  # restart_episode_btn (keep execute_step result)
        gr.update(),  # next_task_btn (keep execute_step result)
        gr.update(interactive=True),  # img_display
        gr.update(interactive=True),  # reference_action_btn
    )


def _get_live_obs_refresh_state(uid, base_count=0):
    with _LIVE_OBS_REFRESH_LOCK:
        if uid not in _LIVE_OBS_REFRESH:
            _LIVE_OBS_REFRESH[uid] = {
                "frame_queue": queue.Queue(),
                "last_base_count": int(base_count),
                "take_next": True,  # downsample x2 by enqueueing every other frame
            }
        return _LIVE_OBS_REFRESH[uid]


def _enqueue_live_obs_frames(uid, base_frames):
    """
    Push newly appended base_frames into per-uid FIFO queue with x2 downsampling.
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
            state["take_next"] = True
        return 0

    if current_count <= last_count:
        return frame_queue.qsize()

    new_frames = frames[last_count:current_count]
    take_next = bool(state.get("take_next", True))
    for frame in new_frames:
        if take_next and frame is not None:
            frame_queue.put(frame)
        take_next = not take_next

    with _LIVE_OBS_REFRESH_LOCK:
        state["last_base_count"] = current_count
        state["take_next"] = take_next
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
        # 0.1s tick playback + small buffer, capped to keep UI responsive.
        max_wait_sec = min(30.0, max(2.0, initial_qsize * 0.12 + 1.0))

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
    Updates live_obs every 0.1s via gr.Timer.
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
    return gr.update(value=img, interactive=False)


def on_video_end_transition(uid):
    """Called when demo video finishes. Transition from video to action phase."""
    return (
        gr.update(visible=False),  # video_phase_group
        gr.update(visible=True),   # action_phase_group
        gr.update(visible=True),   # control_panel_group
        format_log_markdown("please select the action below 👇🏻,\nsome actions also need to select keypoint")
    )


def _task_load_failed_response(uid, message):
    return (
        uid,
        gr.update(visible=True),  # main_interface
        gr.update(value=None, interactive=False),  # img_display
        format_log_markdown(message),  # log_output
        gr.update(choices=[], value=None),  # options_radio
        "",  # goal_box
        "No need for coordinates",  # coords_box
        gr.update(value=None, visible=False),  # video_display
        "",  # task_info_box
        "",  # progress_info_box
        gr.update(interactive=False),  # restart_episode_btn
        gr.update(interactive=False),  # next_task_btn
        gr.update(interactive=False),  # exec_btn
        gr.update(visible=False),  # video_phase_group
        gr.update(visible=False),  # action_phase_group
        gr.update(visible=False),  # control_panel_group
        gr.update(value=""),  # task_hint_display
        gr.update(visible=False),  # loading_overlay
        gr.update(interactive=False),  # reference_action_btn
    )


def _load_status_task(uid, status):
    """Load status.current_task to session and build the standard UI update tuple."""
    current_task = status.get("current_task") if isinstance(status, dict) else None
    if not current_task:
        return _task_load_failed_response(uid, "Error loading task: missing current_task")

    env_id = current_task.get("env_id")
    ep_num = current_task.get("episode_idx")
    if env_id is None or ep_num is None:
        return _task_load_failed_response(uid, "Error loading task: invalid task payload")

    try:
        completed_count = int(status.get("completed_count", 0))
    except (TypeError, ValueError):
        completed_count = 0
    progress_text = f"Completed: {completed_count}"

    session = get_session(uid)
    if session is None:
        print(f"Session {uid} not found, creating new session")
        session = ProcessSessionProxy()
        with _state_lock:
            GLOBAL_SESSIONS[uid] = session
            SESSION_LAST_ACTIVITY[uid] = time.time()
        print(f"New session created for {uid}")

    print(f"Loading {env_id} Ep {ep_num} for {uid}")

    with _LIVE_OBS_REFRESH_LOCK:
        _LIVE_OBS_REFRESH.pop(uid, None)
    reset_play_button_clicked(uid)
    reset_execute_count(uid, env_id, int(ep_num))

    img, load_msg = session.load_episode(env_id, int(ep_num))
    actual_env_id = getattr(session, "env_id", None) or env_id

    if img is not None:
        start_time = datetime.now().isoformat()
        set_task_start_time(uid, env_id, int(ep_num), start_time)

    if img is None:
        set_ui_phase(uid, "executing_task")
        return (
            uid,
            gr.update(visible=True),  # main_interface
            gr.update(value=None, interactive=False),  # img_display
            format_log_markdown(f"Error: {load_msg}"),  # log_output
            gr.update(choices=[], value=None),  # options_radio
            "",  # goal_box
            "No need for coordinates",  # coords_box
            gr.update(value=None, visible=False),  # video_display
            f"{actual_env_id} (Episode {ep_num})",  # task_info_box
            progress_text,  # progress_info_box
            gr.update(interactive=True),  # restart_episode_btn
            gr.update(interactive=True),  # next_task_btn
            gr.update(interactive=False),  # exec_btn
            gr.update(visible=False),  # video_phase_group
            gr.update(visible=True),  # action_phase_group
            gr.update(visible=True),  # control_panel_group
            gr.update(value=get_task_hint(env_id) if env_id else ""),  # task_hint_display
            gr.update(visible=False),  # loading_overlay
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
                opt_label_with_hint = f"{opt_label} (click mouse 🖱️ to select 🎯)"
            else:
                opt_label_with_hint = opt_label
        else:
            opt_label_with_hint = opt_label
        radio_choices.append((opt_label_with_hint, opt_idx))

    demo_video_path = None
    has_demo_video = False
    should_show = should_show_demo_video(actual_env_id) if actual_env_id else False
    initial_log_msg = format_log_markdown("please select the action below 👇🏻,\nsome actions also need to select keypoint")

    if should_show:
        has_demo_video = True
        initial_log_msg = format_log_markdown('press "Watch Video Input🎬" to watch a video\nNote: you can only watch the video once')
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

    img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)

    if has_demo_video:
        set_ui_phase(uid, "executing_task")

        return (
            uid,
            gr.update(visible=True),  # main_interface
            gr.update(value=img, interactive=False),  # img_display
            initial_log_msg,  # log_output
            gr.update(choices=radio_choices, value=None),  # options_radio
            goal_text,  # goal_box
            "No need for coordinates",  # coords_box
            gr.update(value=demo_video_path, visible=True),  # video_display
            f"{actual_env_id} (Episode {ep_num})",  # task_info_box
            progress_text,  # progress_info_box
            gr.update(interactive=True),  # restart_episode_btn
            gr.update(interactive=True),  # next_task_btn
            gr.update(interactive=True),  # exec_btn
            gr.update(visible=True),  # video_phase_group
            gr.update(visible=False),  # action_phase_group
            gr.update(visible=False),  # control_panel_group
            gr.update(value=get_task_hint(actual_env_id)),  # task_hint_display
            gr.update(visible=False),  # loading_overlay
            gr.update(interactive=True),  # reference_action_btn
        )

    set_ui_phase(uid, "executing_task")

    return (
        uid,
        gr.update(visible=True),  # main_interface
        gr.update(value=img, interactive=False),  # img_display
        initial_log_msg,  # log_output
        gr.update(choices=radio_choices, value=None),  # options_radio
        goal_text,  # goal_box
        "No need for coordinates",  # coords_box
        gr.update(value=None, visible=False),  # video_display (no video)
        f"{actual_env_id} (Episode {ep_num})",  # task_info_box
        progress_text,  # progress_info_box
        gr.update(interactive=True),  # restart_episode_btn
        gr.update(interactive=True),  # next_task_btn
        gr.update(interactive=True),  # exec_btn
        gr.update(visible=False),  # video_phase_group
        gr.update(visible=True),  # action_phase_group
        gr.update(visible=True),  # control_panel_group
        gr.update(value=get_task_hint(actual_env_id)),  # task_hint_display
        gr.update(visible=False),  # loading_overlay
        gr.update(interactive=True),  # reference_action_btn
    )


def init_session_and_load_task(uid):
    """Initialize the Gradio session and load the current task."""
    if not uid:
        uid = create_session()

    print(f"[DEBUG init_session_and_load_task] Calling user_manager.init_session({uid})")
    success, msg, status = user_manager.init_session(uid)
    print(f"[DEBUG init_session_and_load_task] init_session result: success={success}, msg={msg}")

    if uid:
        update_session_activity(uid)

    if not success:
        print(f"[DEBUG init_session_and_load_task] Failed, returning error response")
        return _task_load_failed_response(uid, msg)
    print(f"[DEBUG init_session_and_load_task] Success, loading task...")
    return _load_status_task(uid, status)


def load_next_task_wrapper(uid):
    """Move to a random episode within the same env and reload task."""

    if not uid:
        uid = create_session()

    if uid:
        update_session_activity(uid)

    status = user_manager.next_episode_same_env(uid)
    if not status:
        return _task_load_failed_response(uid, "Failed to load next task")
    return _load_status_task(uid, status)


def restart_episode_wrapper(uid):
    """Reload the current env + episode."""
    if not uid:
        uid = create_session()

    if uid:
        update_session_activity(uid)

    status = user_manager.get_session_status(uid)
    current_task = status.get("current_task") if isinstance(status, dict) else None
    if not current_task:
        return _task_load_failed_response(uid, "Failed to restart episode: missing current task")

    env_id = current_task.get("env_id")
    ep_num = current_task.get("episode_idx")
    if env_id is None or ep_num is None:
        return _task_load_failed_response(uid, "Failed to restart episode: invalid task payload")

    return _load_status_task(uid, status)


def switch_env_wrapper(uid, selected_env):
    """Switch env from Current Task dropdown and randomly assign an episode."""
    if not uid:
        uid = create_session()

    if uid:
        update_session_activity(uid)

    if selected_env:
        status = user_manager.switch_env_and_random_episode(uid, selected_env)
    else:
        status = user_manager.get_session_status(uid)

    if not status:
        return _task_load_failed_response(uid, f"Failed to switch environment to '{selected_env}'")

    return _load_status_task(uid, status)


def on_map_click(uid, option_value, evt: gr.SelectData):
    """
    处理图片点击事件
    """
    # 更新session活动时间（点击图片操作）
    if uid:
        update_session_activity(uid)
    
    session = get_session(uid)
    if not session:
        return None, "Session Error"
        
    # Check if current option actually needs coordinates
    needs_coords = False
    if option_value is not None:
        # Parse option index similar to on_option_select
        option_idx = None
        if isinstance(option_value, tuple):
             _, option_idx = option_value
        else:
             option_idx = option_value
             
        if option_idx is not None and 0 <= option_idx < len(session.raw_solve_options):
             opt = session.raw_solve_options[option_idx]
             if opt.get("available"):
                 needs_coords = True
    
    if not needs_coords:
        # Return current state without changes (or reset to default message if needed, but it should already be there)
        # We return the clean image and the "No need" message to enforce state
        base_img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)
        return base_img, "No need for coordinates"

    x, y = evt.index[0], evt.index[1]
    
    # Get clean image from session
    base_img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)
    
    # Draw marker
    marked_img = draw_marker(base_img, x, y)
    
    coords_str = f"{x}, {y}"
    
    return marked_img, coords_str


def _is_valid_coords_text(coords_text: str) -> bool:
    if not isinstance(coords_text, str):
        return False
    text = coords_text.strip()
    if text in {"", "please click the keypoint selection image", "No need for coordinates"}:
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


def on_option_select(uid, option_value, coords_str=None):
    """
    处理选项选择事件
    """
    default_msg = "No need for coordinates"
    
    if option_value is None:
        return default_msg, gr.update(interactive=False)
    
    # 更新session活动时间（选择选项操作）
    if uid:
        update_session_activity(uid)
    
    session = get_session(uid)
    if not session:
        return default_msg, gr.update(interactive=False)
    
    # option_value 是 (label, idx) 元组或直接是 idx
    if isinstance(option_value, tuple):
        _, option_idx = option_value
    else:
        option_idx = option_value

    # Determine coords message
    if 0 <= option_idx < len(session.raw_solve_options):
        opt = session.raw_solve_options[option_idx]
        if opt.get("available"):
             if _is_valid_coords_text(coords_str):
                 return coords_str, gr.update(interactive=True)
             return "please click the keypoint selection image", gr.update(interactive=True)
    
    return default_msg, gr.update(interactive=False)


def on_reference_action(uid):
    """
    自动获取并回填当前步参考 action + 像素坐标（不执行）。
    """
    if uid:
        update_session_activity(uid)

    session = get_session(uid)
    if not session:
        return (
            None,
            gr.update(),
            "No need for coordinates",
            format_log_markdown("Session Error"),
        )

    current_img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)

    try:
        reference = session.get_reference_action()
    except Exception as exc:
        return (
            current_img,
            gr.update(),
            gr.update(),
            format_log_markdown(f"Ground Truth Action Error: {exc}"),
        )

    if not isinstance(reference, dict) or not reference.get("ok", False):
        message = "Failed to resolve ground truth action."
        if isinstance(reference, dict) and reference.get("message"):
            message = str(reference.get("message"))
        return (
            current_img,
            gr.update(),
            gr.update(),
            format_log_markdown(f"Ground Truth Action: {message}"),
        )

    option_idx = reference.get("option_idx")
    option_label = str(reference.get("option_label", "")).strip()
    option_action = str(reference.get("option_action", "")).strip()
    need_coords = bool(reference.get("need_coords", False))
    coords_xy = reference.get("coords_xy")

    updated_img = current_img
    coords_text = "No need for coordinates"
    log_text = f"Ground Truth Action: {option_label}. {option_action}".strip()

    if need_coords and isinstance(coords_xy, (list, tuple)) and len(coords_xy) >= 2:
        x = int(coords_xy[0])
        y = int(coords_xy[1])
        updated_img = draw_marker(current_img, x, y)
        coords_text = f"{x}, {y}"
        log_text = f"Ground Truth Action: {option_label}. {option_action} | coords: {coords_text}"

    return (
        updated_img,
        gr.update(value=option_idx),
        coords_text,
        format_log_markdown(log_text),
    )


def init_app(request: gr.Request):
    """
    处理初始页面加载，直接初始化会话并加载首个任务。

    Args:
        request: Gradio Request 对象，包含查询参数 / Gradio Request object containing query parameters

    Returns:
        初始化后的UI状态
    """
    import traceback
    _ = request  # Query params are intentionally ignored in session-based mode.
    try:
        print(f"[DEBUG init_app] Creating session...")
        uid = create_session()
        print(f"[DEBUG init_app] Session created: {uid}")
        result = init_session_and_load_task(uid)
        print(f"[DEBUG init_app] init_session_and_load_task returned {len(result)} elements")
        return result
    except Exception as e:
        print(f"[ERROR init_app] Exception: {e}")
        traceback.print_exc()
        # Return a safe fallback that hides the loading overlay and shows error
        return _task_load_failed_response("", f"Initialization error: {e}")


def precheck_execute_inputs(uid, option_idx, coords_str):
    """
    Native precheck for execute action.
    Replaces frontend JS interception by validating inputs server-side before phase switch.
    """
    if uid:
        update_session_activity(uid)

    session = get_session(uid)
    if not session:
        raise gr.Error("Session Error")

    parsed_option_idx = option_idx
    if isinstance(option_idx, tuple):
        _, parsed_option_idx = option_idx

    if parsed_option_idx is None:
        raise gr.Error("Error: No action selected")

    needs_coords = False
    if (
        isinstance(parsed_option_idx, int)
        and 0 <= parsed_option_idx < len(session.raw_solve_options)
    ):
        opt = session.raw_solve_options[parsed_option_idx]
        needs_coords = bool(opt.get("available"))

    if needs_coords and not _is_valid_coords_text(coords_str):
        raise gr.Error("please click the keypoint selection image before execute!")


def execute_step(uid, option_idx, coords_str):
    # 检查session是否超时（在更新活动时间之前检查）
    last_activity = get_session_activity(uid)
    if last_activity is not None:
        elapsed = time.time() - last_activity
        if elapsed > SESSION_TIMEOUT:
            raise gr.Error(f"Session已超时：超过 {SESSION_TIMEOUT} 秒未活动。请刷新页面重新登录。")
    
    # 更新session的最后活动时间
    update_session_activity(uid)
    
    session = get_session(uid)
    if not session:
        return None, format_log_markdown("Session Error"), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=False)
    
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
    
    # Ensure at least one cached frame exists for timer-based refresh.
    if not session.base_frames:
        session.update_observation(use_segmentation=USE_SEGMENTED_VIEW)
    
    if option_idx is None:
        return session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW), format_log_markdown("Error: No action selected"), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

    # 检查当前选项是否需要坐标
    needs_coords = False
    if option_idx is not None and 0 <= option_idx < len(session.raw_solve_options):
        opt = session.raw_solve_options[option_idx]
        if opt.get("available"):
            needs_coords = True
    
    # 如果选项需要坐标，检查是否已经点击了图片
    if needs_coords:
        # 检查 coords_str 是否是有效的坐标（不是提示信息）
        is_valid_coords = False
        if coords_str and "," in coords_str:
            try:
                parts = coords_str.split(",")
                x = int(parts[0].strip())
                y = int(parts[1].strip())
                # 如果成功解析为数字，且不是提示信息，则认为是有效坐标
                if coords_str.strip() not in ["please click the keypoint selection image", "No need for coordinates"]:
                    is_valid_coords = True
            except:
                pass
        
        # 如果需要坐标但没有有效坐标，返回错误提示
        if not is_valid_coords:
            current_img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)
            error_msg = "please click the keypoint selection image before execute!"
            return current_img, format_log_markdown(error_msg), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

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
            print(f"Execute limit reached for {uid}:{session.env_id}:{session.episode_idx} (count: {new_count})")
    else:
        # 正常执行
        # 异常处理：所有异常（ScrewPlanFailure 和其他执行错误）都会显示弹窗通知
        print(f"Executing step: Opt {option_idx}, Coords {click_coords}")
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
        
        # 增加 execute 计数（无论成功或失败都计数，因为用户已经执行了一次操作）
        if uid and session.env_id is not None and session.episode_idx is not None:
            new_count = increment_execute_count(uid, session.env_id, session.episode_idx)
            print(f"Execute count for {uid}:{session.env_id}:{session.episode_idx} = {new_count}")

    # Execute frames are produced in batch when execute_action returns from worker process.
    # Enqueue them now, then wait briefly for the 0.1s timer to drain FIFO playback.
    _enqueue_live_obs_frames(uid, getattr(session, "base_frames", None))
    _wait_for_live_obs_queue_drain(uid)
    
    # 注意：执行阶段画面由 live_obs 的 0.1s 轮询刷新。
    
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
            status = "********************************\n****   episode success      ****\n********************************\n  ---please press change episode----   "
        else:
            status = "********************************\n****   episode failed       ****\n********************************\n  ---please press change episode----   "

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
    
    # 根据视图模式重新获取图片
    img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)
        
    restart_episode_update = gr.update(interactive=True)
    next_task_update = gr.update(interactive=True)
    exec_btn_update = gr.update(interactive=False) if done else gr.update(interactive=True)
    
    # 格式化日志消息为 HTML 格式（支持颜色显示）
    formatted_status = format_log_markdown(status)
    
    return (
        img,
        formatted_status,
        task_update,
        progress_update,
        restart_episode_update,
        next_task_update,
        exec_btn_update,
    )
