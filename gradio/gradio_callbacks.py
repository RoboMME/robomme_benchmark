"""
Gradio回调函数模块
响应UI事件，调用业务逻辑，返回UI更新
"""
import gradio as gr
import numpy as np
import time
import traceback
import queue
import os
from datetime import datetime
from state_manager import (
    get_session,
    create_session,
    get_task_index,
    set_task_index,
    get_coordinate_clicks,
    clear_coordinate_clicks,
    add_coordinate_click,
    get_option_selects,
    clear_option_selects,
    add_option_select,
    set_ui_phase,
    reset_ui_phase,
    get_execute_count,
    increment_execute_count,
    reset_execute_count,
    set_task_start_time,
    update_session_activity,
    get_session_activity,
    cleanup_session,
    set_play_button_clicked,
    get_play_button_clicked,
    reset_play_button_clicked,
    GLOBAL_SESSIONS,
    SESSION_LAST_ACTIVITY,
    _state_lock,
)
from streaming_service import FrameQueueManager, cleanup_frame_queue
from image_utils import draw_marker, save_video, concatenate_frames_horizontally
from user_manager import user_manager, LeaseLost
from logger import log_user_action, create_new_attempt, has_existing_actions
from config import USE_SEGMENTED_VIEW, REFERENCE_VIEW_HEIGHT, DEMO_VIDEO_HEIGHT, should_show_demo_video, SESSION_TIMEOUT, EXECUTE_LIMIT_OFFSET
from process_session import ScrewPlanFailureError, ProcessSessionProxy
from note_content import get_task_hint


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


def format_log_html(log_message):
    """
    将纯文本日志消息格式化为带颜色的 HTML 格式
    
    Args:
        log_message: 纯文本日志消息（可以是多行）
    
    Returns:
        str: 格式化的 HTML 字符串，成功消息显示绿色，错误消息显示红色
    """
    if not log_message or not log_message.strip():
        return ""
    
    # 先检查整个消息是否包含成功或失败的关键词，如果包含，整个消息都使用相同颜色
    message_upper = log_message.upper()
    global_color = None
    if any(keyword in message_upper for keyword in ["SUCCESS", "成功", "EPISODE SUCCESS"]):
        global_color = "#28a745"  # 绿色
    elif any(keyword in message_upper for keyword in ["FAILED", "失败", "ERROR", "EPISODE FAILED"]):
        global_color = "#dc3545"  # 红色
    
    # 将消息按行分割
    lines = log_message.split('\n')
    formatted_lines = []
    
    for line in lines:
        # 检查是否是空行（只包含空白字符）
        if not line.strip():
            # 保留空行，但使用最小高度的div
            if global_color:
                formatted_lines.append(f'<div style="color: {global_color}; margin: 0; padding: 0; line-height: 1.4; height: 0.7em;"></div>')
            else:
                formatted_lines.append('<div style="margin: 0; padding: 0; line-height: 1.4; height: 0.7em;"></div>')
            continue
        
        # 保留原始行的空格（不strip），用于正确显示格式化边框
        original_line = line
        
        # 如果整个消息有全局颜色，使用全局颜色；否则按行判断
        if global_color:
            color = global_color
        else:
            # 判断消息类型并设置颜色（使用strip后的行进行判断）
            line_upper = original_line.strip().upper()
            if any(keyword in line_upper for keyword in ["SUCCESS", "成功", "EPISODE SUCCESS"]):
                color = "#28a745"  # 绿色
            elif any(keyword in line_upper for keyword in ["FAILED", "失败", "ERROR", "EPISODE FAILED"]):
                color = "#dc3545"  # 红色
            else:
                color = "inherit"  # 默认颜色
        
        # 转义 HTML 特殊字符（保留空格）
        escaped_line = original_line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        # 创建带颜色的 div，设置 white-space: pre 保留空格，margin 和 padding 为 0 确保无间距
        formatted_lines.append(f'<div style="color: {color}; margin: 0; padding: 0; line-height: 1.4; white-space: pre;">{escaped_line}</div>')
    
    # 包装在容器中，保持紧凑布局
    # 使用 white-space: normal 让div正常换行，但每个div内部使用 white-space: pre 保留空格
    # 注意：不设置 font-size，使用 CSS 中定义的字体大小（.compact-log 和 #log_output）
    html_content = ''.join(formatted_lines)
    result = f'<div style="font-family: monospace; line-height: 1.4;">{html_content}</div>'
    
    return result


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


def get_tutorial_video_path(env_id):
    """
    根据环境ID获取对应的教程视频文件路径（仅在episode 98时使用）
    
    Args:
        env_id: 环境ID（如 "VideoPlaceOrder", "InsertPeg" 等）
    
    Returns:
        str: 视频文件的完整路径，如果文件不存在则返回 None
    """
    if not env_id:
        return None
    
    # 直接使用 env_id 添加 .mp4 后缀（视频文件名是大写的，如 BinFill.mp4）
    video_filename = env_id + ".mp4"
    video_path = os.path.join("/home/hongzefu/historybench-v5.6.16-gradio-final-video/gradio/videos", video_filename)
    
    # 检查文件是否存在
    if os.path.exists(video_path):
        return video_path
    else:
        return None


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
    4. 加载函数执行完成后，返回空字符串给 loading_overlay，遮罩层消失
    
    Returns:
        str: 包含全屏遮罩层 HTML 的字符串
            - 返回 HTML 字符串时：显示遮罩层
            - 返回空字符串 "" 时：隐藏遮罩层（由回调函数在加载完成后返回）
    
    样式说明：
    - 使用 .loading-overlay 类作为全屏遮罩层容器
    - 使用 .loading-content 类作为中央的白色提示卡片
    - 显示英文提示信息："Loading environment, please wait..."
    """
    # 构建全屏遮罩层的 HTML 结构
    # 外层 div 使用 .loading-overlay 类，实现全屏半透明遮罩效果
    # 内层 div 使用 .loading-content 类，显示中央的白色提示卡片
    overlay_html = '''
    <div class="loading-overlay">
        <div class="loading-content">
            <!-- 加载图标：使用时钟表情符号 ⏳ 作为视觉提示，完全不透明 -->
            <div style="font-size: 40px; margin-bottom: 15px; opacity: 1;">⏳</div>
            <!-- 加载提示文本：英文提示，完全不透明，使用深色确保清晰可见 -->
            <div style="font-size: 18px; color: #000000; opacity: 1;">
                Loading environment, please wait...
            </div>
        </div>
    </div>
    '''
    return overlay_html


def on_video_end(uid):
    """
    Called when the demonstration video finishes playing.
    Updates the system log to prompt for action selection.
    """
    return format_log_html("please select the action below 👇🏻,\nsome actions also need to select keypoint")


def login_and_load_task(username, uid):
    """
    Handle user login and load their current task.
    处理用户登录并加载当前任务。
    """
    if not uid:
        uid = create_session()
    
    # Pass uid to login for force takeover mechanism
    success, msg, status = user_manager.login(username, uid)
    
    # 更新session活动时间（登录操作）
    if uid:
        update_session_activity(uid)
    
    if not success:
        # Login failed
        # 登录失败
        return (
            uid,
            gr.update(visible=True), # login_group
            gr.update(visible=False), # main_interface
            msg, # login_message
            gr.update(value=None, interactive=False), format_log_html(""), # img, log_output (空字符串，但也要格式化)
            gr.update(choices=[], value=None), # options
            "", "No need for coordinates", # goal, coords
            gr.update(value="<div id='combined_view_html'><p>等待登录...</p></div>"), 
            gr.update(value=None, visible=False),  # video_display - 登录失败时隐藏
            gr.update(value="", visible=False),  # no_video_display - 登录失败时隐藏
            "", "", # task_info, progress_info
            gr.update(interactive=True), # login_btn
            gr.update(interactive=False), # next_task_btn
            gr.update(interactive=False), # exec_btn
            gr.update(visible=False), # demo_video_group
            gr.update(visible=False), # combined_view_group
            gr.update(visible=False), # operation_zone_group
            gr.update(visible=False, interactive=True),  # play_video_btn
            gr.update(visible=False),  # coords_group
            gr.update(value=""),  # task_hint_display - 任务提示显示（清空）
            gr.update(value=None, visible=False),  # tutorial_video_display - 登录失败时隐藏
            ""  # loading_overlay - 【关键】清空加载遮罩层：返回空字符串，清空 loading_overlay 组件内容，使全屏遮罩层自动隐藏
        )
    
    # #region agent log
    import json as json_module
    try:
        with open('/home/hongzefu/historybench-v5.6.11b5-debug/.cursor/debug.log', 'a', encoding='utf-8') as f:
            f.write(json_module.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"gradio_callbacks.py:259","message":"login_and_load_task after login","data":{"username":username,"status":status},"timestamp":int(__import__('time').time()*1000)})+"\n")
    except: pass
    # #endregion
    
    # Login success - Load current task
    if status["is_done_all"]:
        # 保存任务索引（已完成所有任务）
        set_task_index(uid, status['total_tasks'] - 1, status['total_tasks'])
        task_info = get_task_index(uid)
        task_idx = task_info["task_index"]
        total = task_info["total_tasks"]
        # 生成 HTML 内容，包含 MJPEG 流（添加时间戳防止缓存）
        import random
        random_id = random.randint(0, 1000000)
        combined_html = f'<div id="combined_view_html"><img src="/video_feed/{uid}?r={random_id}" style="max-width: 100%; height: {REFERENCE_VIEW_HEIGHT}; width: auto; margin: 0 auto; display: block; border-radius: 8px; object-fit: contain;" alt="Desk View | Robot View" /></div>'
        # 已完成所有任务，直接显示操作区域
        set_ui_phase(uid, "executing_task")
        return (
            uid,
            gr.update(visible=False), # login_group
            gr.update(visible=True), # main_interface
            f"Welcome {username}. You have completed all tasks!", # login_message (hidden)
            gr.update(value=None, interactive=False), format_log_html("All tasks completed! Thank you."), 
            gr.update(choices=[], value=None),
            "All tasks completed.", "No need for coordinates", 
            gr.update(value=combined_html), 
            gr.update(value=None, visible=False),  # video_display - 所有任务完成时隐藏
            gr.update(value="", visible=False),  # no_video_display - 所有任务完成时隐藏
            "No active task", f"Progress: {total}/{total}",
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False), # exec_btn
            gr.update(visible=True),  # demo_video_group (始终显示)
            gr.update(visible=True),  # combined_view_group (始终显示)
            gr.update(visible=True),  # operation_zone_group (始终显示)
            gr.update(visible=True, interactive=False),  # play_video_btn (显示但禁用)
            gr.update(visible=False),  # coords_group
            gr.update(value=""),  # task_hint_display - 所有任务完成，无提示
            gr.update(value=None, visible=False),  # tutorial_video_display - 所有任务完成时隐藏
            ""  # loading_overlay - 【关键】清空加载遮罩层：返回空字符串，清空 loading_overlay 组件内容，使全屏遮罩层自动隐藏
        )
        
    current_task = status["current_task"]
    env_id = current_task["env_id"]
    ep_num = current_task["episode_idx"]
    
    
    # 特殊处理episode98：如果当前任务是episode98且已有成功记录，自动跳过并推进到下一个任务
    # 这确保用户在episode98成功后关闭重进时，不会再次加载episode98
    if ep_num == 98:
        has_success = user_manager.has_episode98_success(username, env_id)
        if has_success:
            # 已有成功记录，但索引可能没有推进（用户关闭时没有点击Next Task）
            # 使用complete_current_task推进索引（会创建新的进度记录，但这是合理的，因为标记了任务完成）
            # 注意：这里会调用complete_current_task，但传入的env_id和episode_idx是当前任务的
            # 需要从任务列表中获取正确的信息
            print(f"Episode98 for {username}/{env_id} already succeeded, skipping to next task")
            
            # 从任务列表中获取当前任务的完整信息（包括difficulty等）
            tasks = user_manager.user_tasks.get(username, [])
            if tasks and status.get("current_index") is not None:
                current_idx = status["current_index"]
                if current_idx < len(tasks):
                    current_task_info = tasks[current_idx]
                    
                    # #region agent log
                    import json as json_module
                    try:
                        with open('/home/hongzefu/historybench-v5.6.11b5-debug/.cursor/debug.log', 'a', encoding='utf-8') as f:
                            f.write(json_module.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"gradio_callbacks.py:317","message":"current_task_info before complete_current_task","data":{"username":username,"current_idx":current_idx,"current_task_info":current_task_info},"timestamp":int(__import__('time').time()*1000)})+"\n")
                    except: pass
                    # #endregion
                    
                    # 推进任务索引（这会创建新的进度记录，标记任务已完成）
                    # 使用"success"状态，因为episode98已经成功
                    next_status = user_manager.complete_current_task(
                        username,
                        env_id=env_id,
                        episode_idx=ep_num,
                        status="success",  # 使用success状态
                        difficulty=current_task_info.get("difficulty"),
                        language_goal=None,  # 不需要language_goal，因为只是推进索引
                        seed=None  # 不需要seed
                    )
                    
                    # 重新获取任务状态
                    if next_status:
                        status = next_status
                        if status.get("is_done_all"):
                            # 如果所有任务完成，返回完成状态
                            set_task_index(uid, status['total_tasks'] - 1, status['total_tasks'])
                            task_info = get_task_index(uid)
                            task_idx = task_info["task_index"]
                            total = task_info["total_tasks"]
                            import random
                            random_id = random.randint(0, 1000000)
                            combined_html = f'<div id="combined_view_html"><img src="/video_feed/{uid}?r={random_id}" style="max-width: 100%; height: {REFERENCE_VIEW_HEIGHT}; width: auto; margin: 0 auto; display: block; border-radius: 8px; object-fit: contain;" alt="Desk View | Robot View" /></div>'
                            set_ui_phase(uid, "executing_task")
                            return (
                                uid,
                                gr.update(visible=False),
                                gr.update(visible=True),
                                f"Welcome {username}. You have completed all tasks!",
                                gr.update(value=None, interactive=False), format_log_html("All tasks completed! Thank you."), 
                                gr.update(choices=[], value=None),
                                "All tasks completed.", "No need for coordinates", 
                                gr.update(value=combined_html), 
                                gr.update(value=None, visible=False),
                                gr.update(value="", visible=False),
                                "No active task", f"Progress: {total}/{total}",
                                gr.update(interactive=True),
                                gr.update(interactive=False),
                                gr.update(interactive=False),
                                gr.update(visible=True),
                                gr.update(visible=True),
                                gr.update(visible=True),
                                gr.update(visible=True, interactive=False),
                                gr.update(visible=False),
                                gr.update(value=""),
                                gr.update(value=None, visible=False),  # tutorial_video_display - episode98已跳过，隐藏
                                ""
                            )
                        else:
                            # 更新当前任务信息
                            if status.get("current_task"):
                                current_task = status["current_task"]
                                env_id = current_task["env_id"]
                                ep_num = current_task["episode_idx"]
                                print(f"Skipped to next task: {env_id} Ep {ep_num}")
                                
    
    # Load the environment
    session = get_session(uid)


    # 【修复】如果session不存在（可能被free try mode销毁了），创建一个新的session
    # 场景：用户在free try mode下点击"Back to Mode Selection"时，session会被销毁
    # 当用户切换到Record Mode时，需要重新创建session才能正常加载环境
    if session is None:
        print(f"Session {uid} not found, creating new session for {username}")
        # 创建新的ProcessSessionProxy实例（会启动独立的工作进程）
        session = ProcessSessionProxy()
        # 使用线程锁保护全局状态，将新session注册到全局会话存储中
        with _state_lock:
            GLOBAL_SESSIONS[uid] = session
            SESSION_LAST_ACTIVITY[uid] = time.time()  # 更新最后活动时间
        print(f"New session created for {uid} (User: {username})")
    
    print(f"Loading {env_id} Ep {ep_num} for {uid} (User: {username})")
    
    # 清理帧队列（新episode开始）
    cleanup_frame_queue(uid)
    
    # 清空该session的coordinate_clicks和option_selects（新episode开始）
    clear_coordinate_clicks(uid)
    clear_option_selects(uid)
    
    # 重置播放按钮点击状态（新任务开始）
    reset_play_button_clicked(uid)
    
    # 重置该任务的 execute 计数（新任务开始）
    reset_execute_count(username, env_id, int(ep_num))
    
    
    img, load_msg = session.load_episode(env_id, int(ep_num))
    
    
    # 【修复】在load_episode之后，使用session.env_id来判断是否显示视频
    # 这样可以确保使用的是实际加载的环境ID，而不是可能过时的局部变量
    actual_env_id = getattr(session, 'env_id', None) or env_id  # 如果session.env_id不存在，回退到局部变量
    
    # 成功加载 episode 后，记录任务开始时间
    if img is not None:
        start_time = datetime.now().isoformat()
        set_task_start_time(username, env_id, int(ep_num), start_time)
    
    if img is None:
        # 即使加载失败，也保存任务索引
        set_task_index(uid, status['current_index'], status['total_tasks'])
        task_info = get_task_index(uid)
        task_idx = task_info["task_index"]
        total = task_info["total_tasks"]
        # 生成 HTML 内容，包含 MJPEG 流
        import random
        random_id = random.randint(0, 1000000)
        combined_html = f'<div id="combined_view_html"><img src="/video_feed/{uid}?r={random_id}" style="max-width: 100%; height: {REFERENCE_VIEW_HEIGHT}; width: auto; margin: 0 auto; display: block; border-radius: 8px; object-fit: contain;" alt="Desk View | Robot View" /></div>'
        # 加载失败，直接进入执行阶段
        set_ui_phase(uid, "executing_task")
        
        # 如果环境在 DEMO_VIDEO_ENV_IDS 中，无论如何都显示视频播放器（即使为空）
        # 【修复】使用actual_env_id（从session.env_id获取）而不是局部变量env_id
        should_show = should_show_demo_video(actual_env_id) if actual_env_id else False
        
        if should_show:
            video_display_update = gr.update(value=None, visible=True)  # 显示视频播放器（即使为空）
            no_video_display_update = gr.update(value="", visible=False)
        else:
            video_display_update = gr.update(value=None, visible=False)  # 环境不在列表中，隐藏视频
            no_video_display_update = gr.update(visible=True, value=f"<div style='color: black; font-size: 20px; text-align: center; height: {DEMO_VIDEO_HEIGHT}; display: flex; align-items: center; justify-content: center;'>No video</div>")
        
        # 仅在episode 98时显示教程视频
        if int(ep_num) == 98:
            tutorial_video_path = get_tutorial_video_path(actual_env_id)
            if tutorial_video_path:
                tutorial_video_update = gr.update(value=tutorial_video_path, visible=True)
            else:
                tutorial_video_update = gr.update(value=None, visible=False)
        else:
            tutorial_video_update = gr.update(value=None, visible=False)
        
        return (
            uid,
            gr.update(visible=False), # login_group
            gr.update(visible=True), # main_interface
            f"Error loading task for {username}",
            gr.update(value=None, interactive=False), format_log_html(f"Error: {load_msg}"),
            gr.update(choices=[], value=None),
            "", "No need for coordinates", 
            gr.update(value=combined_html), 
            video_display_update,  # video_display - 根据环境是否在列表中决定显示/隐藏
            no_video_display_update,  # no_video_display
            f"Task: {actual_env_id} (Ep {ep_num})", f"Progress: {task_idx + 1}/{total}",
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False), # exec_btn
            gr.update(visible=True),  # demo_video_group (始终显示)
            gr.update(visible=True),  # combined_view_group (始终显示)
            gr.update(visible=True),  # operation_zone_group (始终显示)
            gr.update(visible=True, interactive=False),  # play_video_btn (显示但禁用)
            gr.update(visible=False),  # coords_group
            gr.update(value=get_task_hint(env_id) if env_id else ""),  # task_hint_display - 任务提示（自动加载）
            tutorial_video_update,  # tutorial_video_display - 仅在episode 98时显示
            ""  # loading_overlay - 【关键】清空加载遮罩层：返回空字符串，清空 loading_overlay 组件内容，使全屏遮罩层自动隐藏
        )
        
    # Success loading
    goal_text = capitalize_first_letter(session.language_goal) if session.language_goal else ""
    
    # 检查是否为 episode 98 (trial mode)
    if int(ep_num) == 98:
        gr.Info("This is tutorial mode.")
        capitalized_goal = capitalize_first_letter(session.language_goal) if session.language_goal else ""
        goal_text = f"[[tutorial mode]]\n{capitalized_goal}"
    
    options = session.available_options
    # 生成选项列表，如果选项需要坐标选择，在标签后添加提示  
    radio_choices = []
    for opt_label, opt_idx in options:
        opt_label = _ui_option_label(session, opt_label, opt_idx)
        # 检查该选项是否需要坐标
        if 0 <= opt_idx < len(session.raw_solve_options):
            opt = session.raw_solve_options[opt_idx]
            if opt.get("available"):
                # 需要坐标，在英文标签后添加提示
                opt_label_with_hint = f"{opt_label} (click mouse 🖱️ to select 🎯)"
            else:
                opt_label_with_hint = opt_label
        else:
            opt_label_with_hint = opt_label
        radio_choices.append((opt_label_with_hint, opt_idx))
    
    # 保存任务索引到全局映射，供Progress直接读取
    set_task_index(uid, status['current_index'], status['total_tasks'])
    
    demo_video_path = None
    has_demo_video = False
    # 只用环境是否在 DEMO_VIDEO_ENV_IDS 中判断是否显示演示视频
    # 【修复】使用actual_env_id（从session.env_id获取）而不是局部变量env_id
    # 这样可以确保使用的是实际加载的环境ID，而不是可能过时的局部变量
    should_show = should_show_demo_video(actual_env_id) if actual_env_id else False
    
    # Set initial log message based on whether video is shown
    initial_log_msg = format_log_html("please select the action below 👇🏻,\nsome actions also need to select keypoint")
    
    if should_show:
        has_demo_video = True  # 环境在列表中，标记为需要显示视频
        initial_log_msg = format_log_html('press "Watch Video Input🎬" to watch a video\nNote: you can only watch the video once') # Show Watch Video prompt
        # 尝试生成视频（即使没有 demonstration_frames 也尝试）
        if session.demonstration_frames:
            try:
                demo_video_path = save_video(session.demonstration_frames, "demo")
                
                
                # 验证视频文件是否真实存在且有效
                if demo_video_path:
                    file_exists = os.path.exists(demo_video_path)
                    file_size = os.path.getsize(demo_video_path) if file_exists else 0
                    
                    
                    if not (file_exists and file_size > 0):
                        # 视频文件不存在或无效，但保持 has_demo_video = True
                        demo_video_path = None
            except Exception as e:
                # 保存失败，但保持 has_demo_video = True（视频播放器将为空）
                demo_video_path = None
                pass
        else:
            # 如果没有 demonstration_frames，demo_video_path 保持为 None，但 has_demo_video = True
            pass


    # 从TASK_INDEX_MAP直接读取Progress
    task_info = get_task_index(uid)
    task_idx = task_info["task_index"]
    total = task_info["total_tasks"]
    
    # 根据视图模式重新获取图片
    img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)
    
    # 生成 HTML 内容，包含 MJPEG 流
    # 使用随机参数强制浏览器重新建立连接，避免缓存旧流导致显示问题
    import random
    random_id = random.randint(0, 1000000)
    combined_html = f'<div id="combined_view_html"><img src="/video_feed/{uid}?r={random_id}" style="max-width: 100%; height: {REFERENCE_VIEW_HEIGHT}; width: auto; margin: 0 auto; display: block; border-radius: 8px; object-fit: contain;" alt="Desk View | Robot View" /></div>'
    
    # 根据是否有示范视频决定UI阶段
    if has_demo_video:
        # 有示范视频：同时显示演示视频和执行界面
        set_ui_phase(uid, "executing_task")  # 设置为执行阶段
        
        # 初始化Reference Views队列（有demo video时，也需要立即显示Reference Views）
        if session.base_frames:
            from state_manager import FRAME_QUEUES
            
            # 初始化队列：传入当前frames数量作为监控起始点
            # 监控线程会从这些帧之后开始检测新帧，不会重复添加已存在的帧
            current_base_count = len(session.base_frames) if session.base_frames else 0
            
            if uid not in FRAME_QUEUES:
                FrameQueueManager.init_queue(uid, current_base_count)
            
            # 手动添加最后一帧到队列（重复多次），确保初始显示
            # 这是可选的优化，因为 generate_mjpeg_stream 有 fallback 机制
            last_base_frame = session.base_frames[-1] if session.base_frames else None
            
            if last_base_frame is not None:
                env_id_for_concat = getattr(session, 'env_id', None)
                last_frames = concatenate_frames_horizontally(
                    [last_base_frame],
                    env_id=env_id_for_concat
                )
                
                queue_info = FRAME_QUEUES.get(uid)
                if queue_info and last_frames:
                    last_frame = last_frames[0]
                    # 重复加入最后一帧10次，确保初始显示
                    for _ in range(10):
                        try:
                            frame_copy = np.copy(last_frame) if isinstance(last_frame, np.ndarray) else last_frame
                            queue_info["frame_queue"].put(frame_copy, block=False)
                        except queue.Full:
                            break
        
        
        video_display_update = gr.update(value=demo_video_path, visible=True)
        no_video_display_update = gr.update(value="", visible=False)
        
        # 仅在episode 98时显示教程视频
        if int(ep_num) == 98:
            tutorial_video_path = get_tutorial_video_path(actual_env_id)
            if tutorial_video_path:
                tutorial_video_update = gr.update(value=tutorial_video_path, visible=True)
            else:
                tutorial_video_update = gr.update(value=None, visible=False)
        else:
            tutorial_video_update = gr.update(value=None, visible=False)
        
        return (
            uid,
            gr.update(visible=False), # login_group
            gr.update(visible=True),  # main_interface
            f"Logged in as {username}", 
            gr.update(value=img, interactive=False), 
            initial_log_msg,
            gr.update(choices=radio_choices, value=None),
            goal_text, 
            "No need for coordinates", 
            gr.update(value=combined_html), 
            video_display_update,  # video_display - 有视频时显示
            no_video_display_update,  # no_video_display - 有视频时隐藏并清空内容
            f"Current Task: {actual_env_id}\n(Episode {ep_num})",
            f"Progress: {task_idx + 1}/{total}",
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=True), # exec_btn (直接启用，不需要等待确认)
            gr.update(visible=True),  # demo_video_group (始终显示)
            gr.update(visible=True),  # combined_view_group (始终显示)
            gr.update(visible=True),  # operation_zone_group (始终显示)
            gr.update(visible=True, interactive=True),  # play_video_btn (始终显示)
            gr.update(visible=False),  # coords_group (初始化时隐藏)
            gr.update(value=get_task_hint(actual_env_id)),  # task_hint_display - 任务提示（自动加载，页面加载完就显示）
            tutorial_video_update,  # tutorial_video_display - 仅在episode 98时显示
            ""  # loading_overlay - 【关键】清空加载遮罩层：返回空字符串，清空 loading_overlay 组件内容，使全屏遮罩层自动隐藏
        )
    else:
        # 环境不在 DEMO_VIDEO_ENV_IDS 中：直接进入执行阶段
        set_ui_phase(uid, "executing_task")

        
        # 初始化Reference Views队列（如果环境不在列表中，需要立即显示Reference Views）
        # 注意：即使手动添加了初始帧，generate_mjpeg_stream 也有 fallback 机制直接从 session 获取帧
        # 这确保了即使队列初始化时序有问题，用户也能看到当前状态
        if session.base_frames:
            from state_manager import FRAME_QUEUES
            
            # 初始化队列：传入当前frames数量作为监控起始点
            # 监控线程会从这些帧之后开始检测新帧，不会重复添加已存在的帧
            current_base_count = len(session.base_frames) if session.base_frames else 0
            
            if uid not in FRAME_QUEUES:
                FrameQueueManager.init_queue(uid, current_base_count)
            
            # 手动添加最后一帧到队列（重复多次），确保初始显示
            # 这是可选的优化，因为 generate_mjpeg_stream 有 fallback 机制
            last_base_frame = session.base_frames[-1] if session.base_frames else None
            
            if last_base_frame is not None:
                env_id = getattr(session, 'env_id', None)
                last_frames = concatenate_frames_horizontally(
                    [last_base_frame],
                    env_id=env_id
                )
                
                queue_info = FRAME_QUEUES.get(uid)
                if queue_info and last_frames:
                    last_frame = last_frames[0]
                    # 重复加入最后一帧10次，确保初始显示
                    for _ in range(10):
                        try:
                            frame_copy = np.copy(last_frame) if isinstance(last_frame, np.ndarray) else last_frame
                            queue_info["frame_queue"].put(frame_copy, block=False)
                        except queue.Full:
                            break
        
        # 环境不在 DEMO_VIDEO_ENV_IDS 中，隐藏视频并显示 "No video" 提示
        
        no_video_html = f"<div style='color: black; font-size: 20px; text-align: center; height: {DEMO_VIDEO_HEIGHT}; display: flex; align-items: center; justify-content: center;'>No video</div>"
        video_display_update = gr.update(value=None, visible=False)
        no_video_display_update = gr.update(visible=True, value=no_video_html)
        
        # 仅在episode 98时显示教程视频
        if int(ep_num) == 98:
            tutorial_video_path = get_tutorial_video_path(actual_env_id)
            if tutorial_video_path:
                tutorial_video_update = gr.update(value=tutorial_video_path, visible=True)
            else:
                tutorial_video_update = gr.update(value=None, visible=False)
        else:
            tutorial_video_update = gr.update(value=None, visible=False)
        
        return (
            uid,
            gr.update(visible=False), # login_group
            gr.update(visible=True),  # main_interface
            f"Logged in as {username}", 
            gr.update(value=img, interactive=False), 
            initial_log_msg,
            gr.update(choices=radio_choices, value=None),
            goal_text, 
            "No need for coordinates", 
            gr.update(value=combined_html), 
            video_display_update,  # video_display - 无视频时隐藏，value=None确保hasDemoVideo()返回false
            no_video_display_update,  # no_video_display - 无视频时显示黑色文字（占用和演示视频相同的高度）
            f"Current Task: {actual_env_id}\n(Episode {ep_num})",
            f"Progress: {task_idx + 1}/{total}",
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=True), # exec_btn
            gr.update(visible=True), # demo_video_group (无视频也显示)
            gr.update(visible=True),  # combined_view_group (始终显示)
            gr.update(visible=True),  # operation_zone_group (始终显示)
            gr.update(visible=True, interactive=False),  # play_video_btn (无视频时禁用)
            gr.update(visible=False),  # coords_group (初始化时隐藏)
            gr.update(value=get_task_hint(actual_env_id)),  # task_hint_display - 任务提示（自动加载，页面加载完就显示）
            tutorial_video_update,  # tutorial_video_display - 仅在episode 98时显示
            ""  # loading_overlay - 【关键】清空加载遮罩层：返回空字符串，清空 loading_overlay 组件内容，使全屏遮罩层自动隐藏
        )


def play_demo_video(play_btn, uid_state=None):
    """
    播放示范视频并禁用按钮
    
    Args:
        play_btn: 播放按钮组件
        uid_state: 会话ID（可选，用于状态跟踪）
    """
    # 如果提供了 uid，设置播放按钮已被点击的状态
    if uid_state:
        set_play_button_clicked(uid_state, clicked=True)
    
    # 返回禁用状态的播放按钮
    return gr.update(interactive=False)


def confirm_demo_watched(uid, username):
    """
    用户确认已观看示范视频，切换到执行任务阶段
    """
    # Check lease
    if username:
        try:
            user_manager.assert_lease(username, uid)
        except LeaseLost as e:
            raise gr.Error(f"You have been logged in elsewhere. This page is no longer valid. Please refresh the page to log in again.\\n{str(e)}")
    
    # 更新session活动时间（确认观看演示操作）
    if uid:
        update_session_activity(uid)
    
    # 设置阶段为执行任务
    set_ui_phase(uid, "executing_task")
    
    # 初始化Reference Views队列（确认demo后，需要显示Reference Views）
    session = get_session(uid)
    if session and session.base_frames:
        from state_manager import FRAME_QUEUES
        
        # 初始化队列（如果还没有，或者队列存在但不活跃）
        # 传入当前frames数量，这样监控线程就知道这些frames已存在，不会将它们作为"新"frames加入队列
        current_base_count = len(session.base_frames) if session.base_frames else 0
        
        # 检查队列是否存在且活跃，如果不存在或不活跃，则初始化
        queue_info = FRAME_QUEUES.get(uid)
        if not queue_info or not queue_info.get("streaming_active", False):
            FrameQueueManager.init_queue(uid, current_base_count)
        
        # 只获取最后一帧并处理
        last_base_frame = session.base_frames[-1] if session.base_frames else None
        # 传入 env_id 确保首帧也能绘制坐标系
        env_id_for_concat = getattr(session, 'env_id', None)
        
        if last_base_frame is not None:
            # 使用concatenate_frames_horizontally处理单帧（传入只包含最后一帧的列表）
            last_frames = concatenate_frames_horizontally(
                [last_base_frame],
                env_id=env_id_for_concat
            )
            
            # 只加入最后一帧（重复多次以确保持续显示）
            queue_info = FRAME_QUEUES.get(uid)
            if queue_info and last_frames:
                last_frame = last_frames[0]
                # 重复加入最后一帧10次，确保即使被快速消费也能持续显示
                frames_added = 0
                for _ in range(10):
                    try:
                        # 复制帧以避免引用问题
                        frame_copy = np.copy(last_frame) if isinstance(last_frame, np.ndarray) else last_frame
                        queue_info["frame_queue"].put(frame_copy, block=False)
                        frames_added += 1
                    except queue.Full:
                        break
    
    # 返回UI更新：两个组都显示（这个函数不再被使用，因为不再有第一阶段切换）
    # 保留此函数以防将来需要，但返回值与新的布局一致
    env_id = session.env_id if session and hasattr(session, 'env_id') and session.env_id else ""
    return (
        gr.update(visible=True),   # demo_video_group (始终显示)
        gr.update(visible=True),   # combined_view_group (始终显示)
        gr.update(visible=True),   # operation_zone_group (始终显示)
        gr.update(visible=True, interactive=False),  # play_video_btn (始终显示)
        gr.update(interactive=True),  # exec_btn - 启用执行按钮
        gr.update(visible=False),   # coords_group (还未选择选项，隐藏)
        gr.update(value="")  # task_hint_display - 任务提示（此函数不再使用，但保留兼容性）
    )


def load_next_task_wrapper(username, uid):
    """
    Wrapper to just reload the user's current status (which should be next task if updated).
    如果当前任务已有 actions，则创建新的 attempt。
    对于 user_test，next task 时跳转回 env_id 选择界面。
    For user_test, jump back to env_id selection interface when next task.
    
    特殊处理episode98：如果episode98没有成功记录，保持在当前episode98，不推进索引。
    """
    
    if username:
        # Check lease before proceeding
        try:
            user_manager.assert_lease(username, uid)
        except LeaseLost as e:
            # Raise error to be caught by Gradio and displayed
            raise gr.Error(f"You have been logged in elsewhere. This page is no longer valid. Please refresh the page to log in again.\\n{str(e)}")
        
        # 更新session活动时间（加载下一个任务操作）
        if uid:
            update_session_activity(uid)
        
        # 检查所有ep98是否都完成且没有其他任务完成
        if user_manager.check_all_ep98_completed_and_no_other_tasks(username):
            gr.Info("Tutorial Finished, start testing!")
        
        success, msg, status = user_manager.login(username, uid)
        if success and not status["is_done_all"]:
            current_task = status["current_task"]
            env_id = current_task["env_id"]
            ep_num = current_task["episode_idx"]
            
            # 特殊处理episode98：检查是否有成功记录
            if ep_num == 98:
                has_success = user_manager.has_episode98_success(username, env_id)
                if not has_success:
                    # 没有成功记录，保持在当前episode98（不推进索引）
                    # 检查是否已有actions，如果有则创建新的attempt
                    if has_existing_actions(username, env_id, ep_num):
                        create_new_attempt(username, env_id, ep_num)
                    # 继续加载当前episode98（login_and_load_task会使用status中的当前任务）
                else:
                    # 有成功记录，正常推进到下一个任务
                    # 检查当前任务是否已有 actions，如果有则创建新的 attempt
                    if has_existing_actions(username, env_id, ep_num):
                        create_new_attempt(username, env_id, ep_num)
            else:
                # 非episode98，正常处理
                # 检查当前任务是否已有 actions，如果有则创建新的 attempt
                if has_existing_actions(username, env_id, ep_num):
                    create_new_attempt(username, env_id, ep_num)
    
    return login_and_load_task(username, uid)


def on_map_click(uid, username, option_value, evt: gr.SelectData):
    """
    处理图片点击事件
    """
    # Check lease
    if username:
        try:
            user_manager.assert_lease(username, uid)
        except LeaseLost as e:
            raise gr.Error(f"You have been logged in elsewhere. This page is no longer valid. Please refresh the page to log in again.\\n{str(e)}")
    
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
    
    # 将 PIL Image 转换为 numpy array (RGB 格式)
    image_array = None
    if base_img is not None:
        try:
            # 确保是 RGB 格式
            if base_img.mode != "RGB":
                base_img = base_img.convert("RGB")
            # 转换为 numpy array
            image_array = np.array(base_img, dtype=np.uint8)
        except Exception as e:
            print(f"Error converting image to array in on_map_click: {e}")
            traceback.print_exc()
    
    # 将坐标点击存储到临时列表，等待在action_execute时一起记录
    add_coordinate_click(uid, {
        "coordinates": {"x": x, "y": y},
        "coords_str": coords_str,
        "image_array": image_array,  # 新增：图片数组
        "timestamp": datetime.now().isoformat()
    })
    
    return marked_img, coords_str


def on_option_select(uid, username, option_value):
    """
    处理选项选择事件，记录用户选择了哪个选项
    """
    default_msg = "No need for coordinates"
    
    if option_value is None:
        return default_msg, gr.update(interactive=False), gr.update(visible=False)
    
    # Check lease
    if username:
        try:
            user_manager.assert_lease(username, uid)
        except LeaseLost as e:
            raise gr.Error(f"You have been logged in elsewhere. This page is no longer valid. Please refresh the page to log in again.\\n{str(e)}")
    
    # 更新session活动时间（选择选项操作）
    if uid:
        update_session_activity(uid)
    
    session = get_session(uid)
    if not session:
        return default_msg, gr.update(interactive=False), gr.update(visible=False)
    
    # option_value 是 (label, idx) 元组或直接是 idx
    if isinstance(option_value, tuple):
        option_label, option_idx = option_value
    else:
        option_idx = option_value
        # 从 available_options 中查找标签
        option_label = None
        if session.available_options:
            for label, idx in session.available_options:
                if idx == option_idx:
                    option_label = _ui_option_label(session, label, idx)
                    break
    
    # 将选项选择存储到临时列表，等待在action_execute时一起记录
    add_option_select(uid, {
        "option_idx": option_idx,
        "option_label": option_label,
        "timestamp": datetime.now().isoformat()
    })

    # Determine coords message
    if 0 <= option_idx < len(session.raw_solve_options):
        opt = session.raw_solve_options[option_idx]
        if opt.get("available"):
             return "please click the keypoint selection image", gr.update(interactive=True), gr.update(visible=True)
    
    return default_msg, gr.update(interactive=False), gr.update(visible=False)


def init_app(request: gr.Request):
    """
    处理初始页面加载。
    如果URL中包含 'user' 或 'username' 查询参数，直接登录并进入主界面。
    Handle initial page load.
    If URL contains 'user' or 'username' query parameters, automatically login and show main interface.
    
    支持的URL格式 / Supported URL formats：
    - http://host:port/?user=username
    - http://host:port/?username=username
    
    Args:
        request: Gradio Request 对象，包含查询参数 / Gradio Request object containing query parameters
    
    Returns:
        根据是否自动登录返回不同的UI状态 / Different UI states based on auto-login status
    """
    params = request.query_params if request else {}
    # 支持 'user' 和 'username' 两种参数名称
    username = params.get('user') or params.get('username')
    
    # Default outputs if no user param
    # outputs: 
    # 0. uid
    # 1. loading_group
    # 2. login_group
    # ...
    
    default_outputs = (
        None, 
        gr.update(visible=False), # loading_group (hide it)
        gr.update(visible=True), # login_group (show it)
        gr.update(visible=False), # main_interface
        "", 
        gr.update(value=None, interactive=False), format_log_html(""),  # img, log_output (空字符串，但也要格式化)
        gr.update(choices=[], value=None), 
        "", "No need for coordinates", 
        gr.update(value="<div id='combined_view_html'><p>等待登录...</p></div>"), 
        gr.update(value=None, visible=False),  # video_display - 初始状态隐藏
        gr.update(value="", visible=False),  # no_video_display - 初始状态隐藏
        "", "", 
        gr.update(interactive=True), 
        gr.update(interactive=False), 
        gr.update(interactive=False),
        "",  # username_state
        gr.update(visible=False), # demo_video_group
        gr.update(visible=False), # combined_view_group
        gr.update(visible=False), # operation_zone_group
        gr.update(visible=False, interactive=True),  # play_video_btn
        gr.update(visible=False),  # coords_group (初始化时隐藏)
        gr.update(value=""),  # task_hint_display (初始化时清空)
        gr.update(value=None, visible=False)  # tutorial_video_display (初始化时隐藏)
    )
    
    if username:
        # 检查用户是否存在
        # Check if user exists
        if username in user_manager.user_tasks:
            # 直接登录并加载任务，进入主界面
            # Directly login and load task, show main interface
            print(f"URL Auto-Login: Detected '{username}', automatically logging in and loading task.")
            
            # 创建新的会话ID
            uid = create_session()
            
            # 调用 login_and_load_task 进行登录和任务加载
            login_results = login_and_load_task(username, uid)
            
            # login_and_load_task 返回的格式：
            # (uid, login_group, main_interface, login_msg, img_display, log_output, 
            #  options_radio, goal_box, coords_box, combined_display, video_display, no_video_display,
            #  task_info_box, progress_info_box, login_btn, next_task_btn, exec_btn,
            #  demo_video_group, combined_view_group, operation_zone_group, 
            #  play_video_btn, coords_group, task_hint_display, tutorial_video_display, loading_overlay)
            
            # init_app 需要的返回格式：
            # (uid, loading_group, login_group, main_interface, login_msg, img_display, 
            #  log_output, options_radio, goal_box, coords_box, combined_display, 
            #  video_display, no_video_display, task_info_box, progress_info_box, login_btn, next_task_btn, 
            #  exec_btn, username_state, demo_video_group, combined_view_group, 
            #  operation_zone_group, play_video_btn, coords_group, task_hint_display, tutorial_video_display)
            
            # 转换返回值格式：添加 loading_group 和 username_state，移除 loading_overlay
            return (
                login_results[0],                    # uid
                gr.update(visible=False),            # loading_group (隐藏加载界面)
                login_results[1],                    # login_group
                login_results[2],                   # main_interface
                login_results[3],                   # login_msg
                login_results[4],                  # img_display
                login_results[5],                  # log_output
                login_results[6],                  # options_radio
                login_results[7],                  # goal_box
                login_results[8],                  # coords_box
                login_results[9],                  # combined_display
                login_results[10],                 # video_display
                login_results[11],                 # no_video_display
                login_results[12],                # task_info_box
                login_results[13],                 # progress_info_box
                login_results[14],                 # login_btn
                login_results[15],                 # next_task_btn
                login_results[16],                 # exec_btn
                username,                           # username_state
                login_results[17],                 # demo_video_group
                login_results[18],                 # combined_view_group
                login_results[19],                 # operation_zone_group
                login_results[20],                 # play_video_btn
                login_results[21],                 # coords_group
                login_results[22],                 # task_hint_display
                login_results[23]                  # tutorial_video_display
            )
        else:
            # 用户名不存在，显示错误消息但仍显示登录界面
            # Username does not exist, show error message but still show login interface
            print(f"自动登录失败: 用户名 '{username}' 不存在于用户列表中")
            error_msg = f"⚠️ 用户名 '{username}' 不存在。请从下拉列表中选择有效的用户名。"
            return (
                None,
                gr.update(visible=False),  # loading_group
                gr.update(visible=True),   # login_group (显示登录界面)
                gr.update(visible=False),  # main_interface
                error_msg,                 # login_msg (显示错误消息)
                gr.update(value=None, interactive=False), format_log_html(""),  # img, log_output (空字符串，但也要格式化)
                gr.update(choices=[], value=None),
                "", "No need for coordinates",
                gr.update(value="<div id='combined_view_html'><p>等待登录...</p></div>"), 
                gr.update(value=None, visible=False),  # video_display - 用户名不存在时隐藏
                gr.update(value="", visible=False),  # no_video_display - 用户名不存在时隐藏
                "", "",
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=False),
                "",  # username_state
                gr.update(visible=False), # demo_video_group
                gr.update(visible=False), # combined_view_group
                gr.update(visible=False), # operation_zone_group
                gr.update(visible=False, interactive=True),  # play_video_btn
                gr.update(visible=False),  # coords_group
                gr.update(value=""),  # task_hint_display (清空)
                gr.update(value=None, visible=False)  # tutorial_video_display (初始状态隐藏)
            )
    
    return default_outputs


def execute_step(uid, username, option_idx, coords_str):
    # 记录用户按下 execute 按钮的瞬间时间戳
    execute_timestamp = datetime.now().isoformat()
    
    # 检查session是否超时（在更新活动时间之前检查）
    last_activity = get_session_activity(uid)
    if last_activity is not None:
        elapsed = time.time() - last_activity
        if elapsed > SESSION_TIMEOUT:
            raise gr.Error(f"Session已超时：超过 {SESSION_TIMEOUT} 秒未活动。请刷新页面重新登录。")
    
    # 更新session的最后活动时间
    update_session_activity(uid)
    
    # Check lease first
    if username:
        try:
            user_manager.assert_lease(username, uid)
        except LeaseLost as e:
            raise gr.Error(f"LeaseLost: {str(e)}")
    
    session = get_session(uid)
    if not session:
        return None, format_log_html("Session Error"), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=False)
    
    # 检查播放按钮是否已被点击（如果有演示视频）
    # 注意：前端已经做了完整的验证，这里作为额外的安全检查
    if session.env_id and should_show_demo_video(session.env_id):
        # 检查是否有演示视频
        if session.demonstration_frames:
            play_button_clicked = get_play_button_clicked(uid)
            if not play_button_clicked:
                current_img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)
                error_msg = "Please click 'Start Demonstration Video' button before executing."
                # 记录日志
                print(f"[{execute_timestamp}] User {username} (uid: {uid}) attempted to execute without clicking 'Start Demonstration Video' button. env_id: {session.env_id}, episode_idx: {session.episode_idx}")
                # 使用 gr.Info 显示提示信息
                gr.Info(error_msg)
                return current_img, format_log_html(error_msg), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), gr.update(visible=False)
    
    # 检查 execute 次数限制（在执行前检查，如果达到限制则模拟失败状态）
    execute_limit_reached = False
    if username and session.env_id is not None and session.episode_idx is not None:
        # 从 session 读取 non_demonstration_task_length，如果存在则加上配置的偏移量作为限制，否则不设置限制
        max_execute = None
        if hasattr(session, 'non_demonstration_task_length') and session.non_demonstration_task_length is not None:
            max_execute = session.non_demonstration_task_length + EXECUTE_LIMIT_OFFSET
        
        if max_execute is not None:
            current_count = get_execute_count(username, session.env_id, session.episode_idx)
            if current_count >= max_execute:
                execute_limit_reached = True
    
    # 检查并初始化Reference Views（如果frames为空或队列不存在）
    from state_manager import FRAME_QUEUES
    frames_exist = session.base_frames
    queue_exists = uid in FRAME_QUEUES
    
    if not frames_exist:
        # 从环境中读取初始frames
        session.update_observation(use_segmentation=USE_SEGMENTED_VIEW)
        
        # 如果有frames了，将最后一帧加入队列
        if session.base_frames:
            
            # 初始化队列（如果还没有）
            # 传入当前frames数量，这样监控线程就知道这些frames已存在，不会将它们作为"新"frames加入队列
            current_base_count = len(session.base_frames) if session.base_frames else 0
            env_id_for_concat = getattr(session, 'env_id', None)
            
            if uid not in FRAME_QUEUES:
                FrameQueueManager.init_queue(uid, current_base_count)
            
            # 只获取最后一帧并处理
            last_base_frame = session.base_frames[-1] if session.base_frames else None
            
            if last_base_frame is not None:
                # 使用concatenate_frames_horizontally处理单帧（传入只包含最后一帧的列表）
                last_frames = concatenate_frames_horizontally(
                    [last_base_frame],
                    env_id=env_id_for_concat
                )
                
                # 只加入最后一帧（重复多次以确保持续显示）
                queue_info = FRAME_QUEUES.get(uid)
                if queue_info and last_frames:
                    last_frame = last_frames[0]
                    # 重复加入最后一帧10次，确保即使被快速消费也能持续显示
                    frames_added = 0
                    for _ in range(10):
                        try:
                            # 复制帧以避免引用问题
                            frame_copy = np.copy(last_frame) if isinstance(last_frame, np.ndarray) else last_frame
                            queue_info["frame_queue"].put(frame_copy, block=False)
                            frames_added += 1
                        except queue.Full:
                            break
    elif frames_exist and not queue_exists:
        # frames存在但队列不存在，初始化队列并加入最后一帧
        # 初始化队列（传入当前frames数量，这样监控线程就知道这些frames已存在）
        current_base_count = len(session.base_frames) if session.base_frames else 0
        
        FrameQueueManager.init_queue(uid, current_base_count)
        
        # 只获取最后一帧并处理
        last_base_frame = session.base_frames[-1] if session.base_frames else None
        env_id_for_concat = getattr(session, 'env_id', None)
        
        if last_base_frame is not None:
            # 使用concatenate_frames_horizontally处理单帧（传入只包含最后一帧的列表）
            last_frames = concatenate_frames_horizontally(
                [last_base_frame],
                env_id=env_id_for_concat
            )
            
            # 只加入最后一帧（重复多次以确保持续显示）
            queue_info = FRAME_QUEUES.get(uid)
            if queue_info and last_frames:
                last_frame = last_frames[0]
                # 重复加入最后一帧10次，确保即使被快速消费也能持续显示
                frames_added = 0
                for _ in range(10):
                    try:
                        # 复制帧以避免引用问题
                        frame_copy = np.copy(last_frame) if isinstance(last_frame, np.ndarray) else last_frame
                        queue_info["frame_queue"].put(frame_copy, block=False)
                        frames_added += 1
                    except queue.Full:
                        break
    
    if option_idx is None:
        return session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW), format_log_html("Error: No action selected"), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), gr.update(visible=False)

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
            return current_img, format_log_html(error_msg), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), gr.update(visible=True)

    # Parse coords
    click_coords = None
    if coords_str and "," in coords_str:
        try:
            parts = coords_str.split(",")
            click_coords = (int(parts[0].strip()), int(parts[1].strip()))
        except:
            pass
    
    # 在执行 action 之前记录当前帧数
    # 这些帧数将作为监控线程的起始点，确保只监控新产生的帧
    pre_base_frame_count = len(session.base_frames)
    
    # 【重要修复】清空队列中的旧帧，确保从当前execute的第一个frame开始播放
    # 修复问题：当第一个任务执行完毕但livestream还没播放完时，直接执行下一个任务
    # 会导致livestream跳回开头重新播放。通过清空队列，确保每次execute都从新帧开始
    if uid in FRAME_QUEUES:
        queue_info = FRAME_QUEUES.get(uid)
        if queue_info:
            while not queue_info["frame_queue"].empty():
                try:
                    queue_info["frame_queue"].get_nowait()
                except queue.Empty:
                    break
    
    # 初始化队列和启动监控线程（用于流式输出）
    # 使用当前帧数作为起始点，这样监控线程只会添加execute后新产生的帧
    FrameQueueManager.init_queue(uid, pre_base_frame_count)
    
    # 在执行前获取当前图片（用于记录最后执行的坐标对应的图片）
    pre_execute_image = None
    if click_coords:
        try:
            pre_execute_pil = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)
            # 转换为 numpy array (RGB格式)
            pre_execute_image = np.array(pre_execute_pil)
            if len(pre_execute_image.shape) == 2:
                pre_execute_image = np.stack([pre_execute_image] * 3, axis=-1)
            elif len(pre_execute_image.shape) == 3 and pre_execute_image.shape[2] == 4:
                pre_execute_image = pre_execute_image[:, :, :3]
        except Exception as e:
            print(f"Error getting pre-execute image: {e}")
            
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
        if username and session.env_id is not None and session.episode_idx is not None:
            new_count = increment_execute_count(username, session.env_id, session.episode_idx)
            print(f"Execute limit reached for {username}:{session.env_id}:{session.episode_idx} (count: {new_count})")
    else:
        # 正常执行
        # 异常处理：所有异常（ScrewPlanFailure 和其他执行错误）都会显示弹窗通知
        print(f"Executing step: Opt {option_idx}, Coords {click_coords}")
        try:
            img, status, done = session.execute_action(option_idx, click_coords)
        except ScrewPlanFailureError as e:
            # 捕获 screw plan 失败异常，显示弹窗通知
            error_message = str(e)
            gr.Info(f"Robot cannot reach position")
            # 返回当前状态，在状态消息中显示错误信息
            current_img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)
            status = f"Screw plan failed: {error_message}"
            done = False
            # 继续正常返回流程
            img = current_img
        except RuntimeError as e:
            # 捕获所有其他执行错误，显示弹窗通知
            error_message = str(e)
            gr.Info(f"Cannot find suitable target")
            # 返回当前状态，在状态消息中显示错误信息
            current_img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)
            status = f"Error: {error_message}"
            done = False
            # 继续正常返回流程
            img = current_img
        
        # 增加 execute 计数（无论成功或失败都计数，因为用户已经执行了一次操作）
        if username and session.env_id is not None and session.episode_idx is not None:
            new_count = increment_execute_count(username, session.env_id, session.episode_idx)
            print(f"Execute count for {username}:{session.env_id}:{session.episode_idx} = {new_count}")
    
    # 等待一小段时间，确保continuous_frame_monitor线程处理完所有帧
    # 不再手动调用monitor_frames_and_enqueue，因为continuous_frame_monitor线程已经在处理
    time.sleep(0.3)
    
    # 标记流式输出完成（让Timer有机会读取最后的帧）
    from state_manager import FRAME_QUEUES
    if uid in FRAME_QUEUES:
        # 停止流式输出，continuous_frame_monitor线程会检测到并退出
        FRAME_QUEUES[uid]["streaming_active"] = False
    
    # 记录执行操作（包含从上次action到这次action之间的所有coordinate_click）
    if username and session.env_id is not None and session.episode_idx is not None:
        try:
            # 获取选项标签
            option_label = None
            if session.available_options:
                for label, idx in session.available_options:
                    if idx == option_idx:
                        option_label = _ui_option_label(session, label, idx)
                        break
            
            # 获取从上次action_execute到现在的所有option_select
            option_selects_before_execute = get_option_selects(uid).copy()
            clear_option_selects(uid)
            
            # 获取从上次action_execute到现在的所有coordinate_click
            # 这些点击已经包含了 image_array（在 on_map_click 中添加）
            coordinate_clicks_before_execute = get_coordinate_clicks(uid).copy()
            clear_coordinate_clicks(uid)
            
            # 获取最后执行的坐标和图片
            final_coordinates = None
            final_coords_str = None
            final_image_array = None
            if click_coords:
                final_coordinates = {"x": click_coords[0], "y": click_coords[1]}
                final_coords_str = f"{click_coords[0]},{click_coords[1]}"
                final_image_array = pre_execute_image  # 使用执行前的图片
            
            # 获取 option_list（从 session.raw_solve_options 获取）
            option_list = None
            if hasattr(session, 'raw_solve_options') and session.raw_solve_options:
                option_list = session.raw_solve_options
            
            # 获取任务元数据（从 session 对象获取）
            task_status = status  # 使用当前执行状态
            task_difficulty = None
            if hasattr(session, 'difficulty') and session.difficulty is not None:
                task_difficulty = session.difficulty
            task_language_goal = None
            if hasattr(session, 'language_goal') and session.language_goal is not None:
                task_language_goal = session.language_goal
            task_seed = None
            if hasattr(session, 'seed') and session.seed is not None:
                task_seed = session.seed
            
            log_user_action(
                username=username,
                env_id=session.env_id,
                episode_idx=session.episode_idx,
                action_data={
                    "execute_timestamp": execute_timestamp,  # 用户按下 execute 按钮的瞬间时间戳
                    "option_idx": option_idx,  # execute时使用的option（最后一次选择的）
                    "option_label": option_label,
                    "final_coordinates": final_coordinates,  # 最后执行的坐标
                    "final_coords_str": final_coords_str,  # 最后执行的坐标字符串
                    "final_image_array": final_image_array,  # 最后执行时的图片
                    "option_selects_before_execute": option_selects_before_execute,  # execute之前所有的option选择
                    "coordinate_clicks_before_execute": coordinate_clicks_before_execute,  # execute之前所有的坐标点击（已包含 image_array）
                    "status": status,
                    "done": done
                },
                option_list=option_list,
                status=task_status,
                difficulty=task_difficulty,
                language_goal=task_language_goal,
                seed=task_seed
            )
        except Exception as e:
            print(f"Error logging action execute: {e}")
            traceback.print_exc()
    
    # 注意：不再在这里生成完整视频，而是通过 MJPEG 流式输出
    # combined_display 现在使用 HTML + MJPEG 流，不需要手动更新
    # 不再返回 combined_display 的更新，避免显示加载动画
    
    progress_update = gr.update()  # 不更新 progress，保持原值
    task_update = gr.update()
    
    if done:
        # 确定最终状态用于日志记录
        final_log_status = "failed"
        if "SUCCESS" in status:
            final_log_status = "success"
        
        # Episode完成时，格式化System Log的状态消息
        # 使用固定模板，所有行长度一致（32个字符），无空行
        if final_log_status == "success":
            status = "********************************\n****   episode success      ****\n********************************\n  ---please press next task----   "
        else:
            status = "********************************\n****   episode failed       ****\n********************************\n  ---please press next task----   "

        # Update user progress (但不更新 progress_info_box，等用户按 next task/refresh 时再更新)
        if username:
            # 判断是否为 episode_idx == 98
            ep_val = getattr(session, 'episode_idx', None)
            ep_is_98 = False
            if ep_val is not None:
                if isinstance(ep_val, int):
                    ep_is_98 = (ep_val == 98)
                else:
                    ep_is_98 = (str(ep_val) == "98")
            
            # episode98失败时不推进索引，成功时推进索引
            if ep_is_98 and (final_log_status == "failed"):
                # 跳过 complete_current_task，不推进任务索引
                gr.Info("---please press Next Task to redo it again---")
                task_update = "Task Failed. Press Next Task to retry same episode."
            else:
                # 正常推进任务索引并生成下一任务提示（包括episode98成功的情况）
                seed = getattr(session, 'seed', None)
                user_status = user_manager.complete_current_task(
                    username,
                    env_id=session.env_id,
                    episode_idx=session.episode_idx,
                    status=final_log_status,
                    difficulty=session.difficulty if hasattr(session, 'difficulty') and session.difficulty is not None else None,
                    language_goal=session.language_goal,
                    seed=seed
                )
                if user_status:
                    if user_status["is_done_all"]:
                        task_update = "All tasks completed!"
                        # progress_update 保持为 gr.update()，不改变
                    else:
                        next_env = user_status["current_task"]["env_id"]
                        next_ep = user_status["current_task"]["episode_idx"]
                        task_update = f"Task Completed! Next: {next_env} (Ep {next_ep})"
                        # progress_update 保持为 gr.update()，不改变
    
    # 根据视图模式重新获取图片
    img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)
        
    next_task_update = gr.update(interactive=True) if done else gr.update(interactive=False)
    exec_btn_update = gr.update(interactive=False) if done else gr.update(interactive=True)
    
    # 执行后隐藏Coords区域
    coords_group_update = gr.update(visible=False)
    
    # 格式化日志消息为 HTML 格式（支持颜色显示）
    formatted_status = format_log_html(status)
    
    return img, formatted_status, task_update, progress_update, next_task_update, exec_btn_update, coords_group_update
