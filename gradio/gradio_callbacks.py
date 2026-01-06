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
    GLOBAL_SESSIONS,
    SESSION_LAST_ACTIVITY,
    _state_lock,
)
from streaming_service import FrameQueueManager, cleanup_frame_queue
from image_utils import draw_marker, save_video, concatenate_frames_horizontally
from user_manager import user_manager, LeaseLost
from logger import log_user_action, create_new_attempt, has_existing_actions
from config import USE_SEGMENTED_VIEW, REFERENCE_VIEW_HEIGHT, should_show_demo_video, SESSION_TIMEOUT, EXECUTE_LIMIT_OFFSET
from process_session import ScrewPlanFailureError, ProcessSessionProxy
from note_content import get_task_hint


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
            gr.update(visible=False), # env_selection_group
            gr.update(visible=False), # main_interface
            msg, # login_message
            gr.update(value=None, interactive=False), None, # img, status
            gr.update(choices=[], value=None), # options
            "", "No need for coordinates", # goal, coords
            gr.update(value="<div id='combined_view_html'><p>等待登录...</p></div>"), None, # combined_html, demo_video
            "", "", # task_info, progress_info
            gr.update(interactive=True), # login_btn
            gr.update(interactive=False), # next_task_btn
            gr.update(interactive=False), # exec_btn
            gr.update(visible=False), # demo_video_group
            gr.update(visible=False), # combined_view_group
            gr.update(visible=False), # operation_zone_group
            gr.update(visible=False),  # confirm_demo_btn
            gr.update(visible=False, interactive=True),  # play_video_btn
            gr.update(visible=False),  # coords_group
            # 【修改】任务提示改为延迟加载：不再在登录失败时自动加载提示内容
            # 初始值设为空字符串，用户需要点击"Show Hint"按钮才会显示提示
            gr.update(value=""),  # note2 - 任务提示（延迟加载，初始为空）
            gr.update(value=""),  # note2_demo - 演示任务提示（延迟加载，初始为空）
            ""  # loading_overlay - 【关键】清空加载遮罩层：返回空字符串，清空 loading_overlay 组件内容，使全屏遮罩层自动隐藏
        )
    
    # 特殊处理：如果是 user_test，显示 env_id 选择界面
    # Special handling: if user_test, show env_id selection interface
    if username.endswith("_test"):
        return (
            uid,
            gr.update(visible=False), # login_group
            gr.update(visible=True),  # env_selection_group
            gr.update(visible=False), # main_interface
            f"Logged in as {username}. Please select an environment ID.", # login_message
            gr.update(value=None, interactive=False), "", # img, status
            gr.update(choices=[], value=None), # options
            "", "No need for coordinates", # goal, coords
            gr.update(value="<div id='combined_view_html'><p>等待选择环境...</p></div>"), None, # combined_html, demo_video
            "", "", # task_info, progress_info
            gr.update(interactive=True), # login_btn
            gr.update(interactive=False), # next_task_btn
            gr.update(interactive=False), # exec_btn
            gr.update(visible=False), # demo_video_group
            gr.update(visible=False), # combined_view_group
            gr.update(visible=False), # operation_zone_group
            gr.update(visible=False),  # confirm_demo_btn
            gr.update(visible=False, interactive=True),  # play_video_btn
            gr.update(visible=False),  # coords_group
            gr.update(value=""),  # note2
            gr.update(value=""),  # note2_demo
            ""  # loading_overlay - 【关键】清空加载遮罩层：返回空字符串，清空 loading_overlay 组件内容，使全屏遮罩层自动隐藏
        )
    
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
            gr.update(visible=False), # env_selection_group
            gr.update(visible=True), # main_interface
            f"Welcome {username}. You have completed all tasks!", # login_message (hidden)
            gr.update(value=None, interactive=False), "All tasks completed! Thank you.", 
            gr.update(choices=[], value=None),
            "All tasks completed.", "No need for coordinates", 
            gr.update(value=combined_html), None,
            "No active task", f"Progress: {total}/{total}",
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False), # exec_btn
            gr.update(visible=False), # demo_video_group
            gr.update(visible=False), # combined_view_group
            gr.update(visible=True),  # operation_zone_group
            gr.update(visible=False),  # confirm_demo_btn
            gr.update(visible=False, interactive=True),  # play_video_btn
            gr.update(visible=False),  # coords_group
            gr.update(value=""),  # note2
            gr.update(value=""),  # note2_demo
            ""  # loading_overlay - 【关键】清空加载遮罩层：返回空字符串，清空 loading_overlay 组件内容，使全屏遮罩层自动隐藏
        )

    current_task = status["current_task"]
    env_id = current_task["env_id"]
    ep_num = current_task["episode_idx"]
    
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
    
    # 重置该任务的 execute 计数（新任务开始）
    reset_execute_count(username, env_id, int(ep_num))
    
    img, load_msg = session.load_episode(env_id, int(ep_num))
    
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
         return (
            uid,
            gr.update(visible=False), # login_group
            gr.update(visible=False), # env_selection_group
            gr.update(visible=True), # main_interface
            f"Error loading task for {username}",
            gr.update(value=None, interactive=False), f"Error: {load_msg}",
            gr.update(choices=[], value=None),
            "", "No need for coordinates", 
            gr.update(value=combined_html), None,
            f"Task: {env_id} (Ep {ep_num})", f"Progress: {task_idx + 1}/{total}",
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False), # exec_btn
            gr.update(visible=False), # demo_video_group
            gr.update(visible=False), # combined_view_group
            gr.update(visible=True),  # operation_zone_group
            gr.update(visible=False),  # confirm_demo_btn
            gr.update(visible=False, interactive=True),  # play_video_btn
            gr.update(visible=False),  # coords_group
            gr.update(value=""),  # note2
            gr.update(value=""),  # note2_demo
            ""  # loading_overlay - 【关键】清空加载遮罩层：返回空字符串，清空 loading_overlay 组件内容，使全屏遮罩层自动隐藏
        )
        
    # Success loading
    goal_text = f"{session.language_goal}"
    options = session.available_options
    # 生成选项列表，如果选项需要坐标选择，在标签后添加提示
    radio_choices = []
    for opt_label, opt_idx in options:
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
    # 只有ENV_IDS为video的才显示demonstration videos
    if session.demonstration_frames and should_show_demo_video(env_id):
        try:
            demo_video_path = save_video(session.demonstration_frames, "demo")
            has_demo_video = True
        except: pass


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
        # 有示范视频：第一阶段 - 观看示范视频
        reset_ui_phase(uid)  # 设置为 "watching_demo"
        
        return (
            uid,
            gr.update(visible=False), # login_group
            gr.update(visible=False), # env_selection_group
            gr.update(visible=True),  # main_interface
            f"Logged in as {username}", 
            gr.update(value=img, interactive=False), 
            f"Ready. Task {task_idx + 1}/{total}: {env_id}",
            gr.update(choices=radio_choices, value=None),
            goal_text, 
            "No need for coordinates", 
            gr.update(value=combined_html), 
            demo_video_path,
            f"Current Task: {env_id} (Episode {ep_num})",
            f"Progress: {task_idx + 1}/{total}",
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False), # exec_btn (第一阶段禁用)
            gr.update(visible=True),  # demo_video_group (第一阶段显示)
            gr.update(visible=False), # combined_view_group (第一阶段隐藏，正确)
            gr.update(visible=False), # operation_zone_group (第一阶段隐藏)
            gr.update(visible=True, interactive=False),  # confirm_demo_btn (第一阶段显示，初始禁用)
            gr.update(visible=True, interactive=True),  # play_video_btn (第一阶段显示)
            gr.update(visible=False),  # coords_group (初始化时隐藏)
            # 【修改】任务提示延迟加载功能（有示范视频的情况）：
            # 之前：任务加载时自动调用 get_task_hint(env_id) 显示提示内容
            # 现在：初始值设为空字符串，用户需要点击"Show Hint"按钮才会通过 show_task_hint() 函数加载并显示提示
            # 这样可以减少不必要的计算，提升页面加载速度，同时让用户按需查看提示
            gr.update(value=""),  # note2 - 任务提示（延迟加载，初始为空，点击按钮后显示）
            gr.update(value=""),  # note2_demo - 演示任务提示（延迟加载，初始为空，点击按钮后显示）
            ""  # loading_overlay - 【关键】清空加载遮罩层：返回空字符串，清空 loading_overlay 组件内容，使全屏遮罩层自动隐藏
        )
    else:
        # 没有示范视频：直接进入执行阶段
        set_ui_phase(uid, "executing_task")

        
        # 初始化Reference Views队列（如果没有demo video，需要立即显示Reference Views）
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
        
        return (
            uid,
            gr.update(visible=False), # login_group
            gr.update(visible=False), # env_selection_group
            gr.update(visible=True),  # main_interface
            f"Logged in as {username}", 
            gr.update(value=img, interactive=False), 
            f"Ready. Task {task_idx + 1}/{total}: {env_id}",
            gr.update(choices=radio_choices, value=None),
            goal_text, 
            "No need for coordinates", 
            gr.update(value=combined_html), 
            None,  # demo_video_path
            f"Current Task: {env_id} (Episode {ep_num})",
            f"Progress: {task_idx + 1}/{total}",
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=True), # exec_btn
            gr.update(visible=False), # demo_video_group (无视频，隐藏)
            gr.update(visible=True),  # combined_view_group (修复：应该显示)
            gr.update(visible=True),  # operation_zone_group (直接显示)
            gr.update(visible=False), # confirm_demo_btn (无视频，隐藏)
            gr.update(visible=False, interactive=True),  # play_video_btn
            gr.update(visible=False),  # coords_group (初始化时隐藏)
            # 【修改】任务提示延迟加载功能（无示范视频的情况）：
            # 之前：任务加载时自动调用 get_task_hint(env_id) 显示提示内容
            # 现在：初始值设为空字符串，用户需要点击"Show Hint"按钮才会通过 show_task_hint() 函数加载并显示提示
            # 这样可以减少不必要的计算，提升页面加载速度，同时让用户按需查看提示
            gr.update(value=""),  # note2 - 任务提示（延迟加载，初始为空，点击按钮后显示）
            gr.update(value=""),  # note2_demo - 演示任务提示（延迟加载，初始为空，点击按钮后显示）
            ""  # loading_overlay - 【关键】清空加载遮罩层：返回空字符串，清空 loading_overlay 组件内容，使全屏遮罩层自动隐藏
        )


def play_demo_video(play_btn):
    """
    播放示范视频并禁用按钮，同时启用Start Task按钮
    """
    # 返回禁用状态的播放按钮，和启用状态的确认按钮
    return gr.update(interactive=False), gr.update(interactive=True)


def confirm_demo_watched(uid, username):
    """
    用户确认已观看示范视频，切换到执行任务阶段
    """
    # Check lease
    if username:
        try:
            user_manager.assert_lease(username, uid)
        except LeaseLost as e:
            raise gr.Error(f"You have been logged in elsewhere. This page is no longer valid. Please refresh the page to log in again.\n{str(e)}")
    
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
    
    # 返回UI更新：隐藏示范视频，显示Combined View和操作区域
    # 同时更新 exec_btn 为可交互状态
    env_id = session.env_id if session and hasattr(session, 'env_id') and session.env_id else ""
    return (
        gr.update(visible=False),  # demo_video_group
        gr.update(visible=True),   # combined_view_group (修复：应该显示)
        gr.update(visible=True),   # operation_zone_group
        gr.update(visible=False),  # confirm_demo_btn
        gr.update(visible=False, interactive=False),  # play_video_btn (确认后隐藏)
        gr.update(interactive=True),  # exec_btn - 启用执行按钮
        gr.update(visible=False),   # coords_group (确认demo后，还未选择选项，隐藏)
        # 【修改】任务提示延迟加载功能（确认观看演示视频后）：
        # 之前：确认观看演示视频后自动调用 get_task_hint(env_id) 显示提示内容
        # 现在：初始值设为空字符串，用户需要点击"Show Hint"按钮才会通过 show_task_hint() 函数加载并显示提示
        # 这样可以减少不必要的计算，提升页面加载速度，同时让用户按需查看提示
        gr.update(value=""),  # note2 - 任务提示（延迟加载，初始为空，用户必须点击按钮才显示）
        gr.update(value="")  # note2_demo - 演示任务提示（延迟加载，初始为空，用户必须点击按钮才显示）
    )


def select_env_id(username, uid, env_id):
    """
    为 user_test 用户选择 env_id 并加载对应的任务。
    Select env_id for user_test and load the corresponding task.
    """
    if not username.endswith("_test"):
        # 如果不是 user_test，返回错误
        # If not user_test, return error
        return (
            uid,
            gr.update(visible=True), # login_group
            gr.update(visible=False), # env_selection_group
            gr.update(visible=False), # main_interface
            "Error: This function is only for user_test", # login_msg
            gr.update(value=None, interactive=False), "", # img, status
            gr.update(choices=[], value=None), # options
            "", "No need for coordinates", # goal, coords
            gr.update(value="<div id='combined_view_html'><p>错误...</p></div>"), None, # combined_html, demo_video
            "", "", # task_info, progress_info
            gr.update(interactive=True), # login_btn
            gr.update(interactive=False), # next_task_btn
            gr.update(interactive=False), # exec_btn
            gr.update(visible=False), # demo_video_group
            gr.update(visible=False), # combined_view_group
            gr.update(visible=False), # operation_zone_group
            gr.update(visible=False),  # confirm_demo_btn
            gr.update(visible=False, interactive=True),  # play_video_btn
            gr.update(visible=False),  # coords_group
            gr.update(value=""),  # note2
            gr.update(value=""),  # note2_demo
            ""  # loading_overlay - 【关键】清空加载遮罩层：返回空字符串，清空 loading_overlay 组件内容，使全屏遮罩层自动隐藏
        )
    
    if not uid:
        uid = create_session()
    
    # Check lease
    try:
        user_manager.assert_lease(username, uid)
    except LeaseLost as e:
        raise gr.Error(f"You have been logged in elsewhere. This page is no longer valid. Please refresh the page to log in again.\n{str(e)}")
    
    # 更新session活动时间（选择环境ID操作）
    if uid:
        update_session_activity(uid)
    
    # 为 user_test 加载指定的 env_id，episode_idx 固定为 99
    # Load specific env_id for user_test, episode_idx fixed to 99
    episode_idx = 99
    
    # Load the environment
    session = get_session(uid)
    print(f"Loading {env_id} Ep {episode_idx} for {uid} (User: {username})")
    
    # 清理帧队列（新episode开始）
    # Clear frame queue (start of new episode)
    cleanup_frame_queue(uid)
    
    # 清空该session的coordinate_clicks和option_selects（新episode开始）
    # Clear coordinate_clicks and option_selects for this session (start of new episode)
    clear_coordinate_clicks(uid)
    clear_option_selects(uid)
    
    # 重置该任务的 execute 计数（新任务开始）
    # Reset execute count for this task (start of new task)
    reset_execute_count(username, env_id, episode_idx)
    
    img, load_msg = session.load_episode(env_id, episode_idx)
    
    # 成功加载 episode 后，记录任务开始时间
    # After successfully loading episode, record task start time
    if img is not None:
        start_time = datetime.now().isoformat()
        set_task_start_time(username, env_id, episode_idx, start_time)
    
    if img is None:
        # 加载失败
        # Load failed
        import random
        random_id = random.randint(0, 1000000)
        combined_html = f'<div id="combined_view_html"><img src="/video_feed/{uid}?r={random_id}" style="max-width: 100%; height: {REFERENCE_VIEW_HEIGHT}; width: auto; margin: 0 auto; display: block; border-radius: 8px; object-fit: contain;" alt="Desk View | Robot View" /></div>'
        set_ui_phase(uid, "executing_task")
        return (
            uid,
            gr.update(visible=False), # login_group
            gr.update(visible=False), # env_selection_group
            gr.update(visible=True), # main_interface
            f"Error loading task: {load_msg}",
            gr.update(value=None, interactive=False), f"Error: {load_msg}",
            gr.update(choices=[], value=None),
            "", "No need for coordinates", 
            gr.update(value=combined_html), None,
            f"Task: {env_id} (Ep {episode_idx})", "Progress: N/A",
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False), # exec_btn
            gr.update(visible=False), # demo_video_group
            gr.update(visible=False), # combined_view_group
            gr.update(visible=True),  # operation_zone_group
            gr.update(visible=False),  # confirm_demo_btn
            gr.update(visible=False, interactive=True),  # play_video_btn
            gr.update(visible=False),  # coords_group
            # 【修改】任务提示延迟加载功能（select_env_id函数 - 加载失败情况）：
            # 之前：任务加载失败时也会调用 get_task_hint(env_id) 显示提示内容
            # 现在：初始值设为空字符串，用户需要点击"Show Hint"按钮才会通过 show_task_hint() 函数加载并显示提示
            gr.update(value=""),  # note2 - 任务提示（延迟加载，初始为空）
            gr.update(value=""),  # note2_demo - 演示任务提示（延迟加载，初始为空）
            ""  # loading_overlay - 【关键】清空加载遮罩层：返回空字符串，清空 loading_overlay 组件内容，使全屏遮罩层自动隐藏
        )
    
    # Success loading
    goal_text = f"{session.language_goal}"
    options = session.available_options
    # 生成选项列表
    # Generate option list
    radio_choices = []
    for opt_label, opt_idx in options:
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
    if session.demonstration_frames and should_show_demo_video(env_id):
        try:
            demo_video_path = save_video(session.demonstration_frames, "demo")
            has_demo_video = True
        except: pass
    
    # 根据视图模式重新获取图片
    # Re-acquire image based on view mode
    img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)
    
    # 生成 HTML 内容，包含 MJPEG 流
    # Generate HTML content containing MJPEG stream
    import random
    random_id = random.randint(0, 1000000)
    combined_html = f'<div id="combined_view_html"><img src="/video_feed/{uid}?r={random_id}" style="max-width: 100%; height: {REFERENCE_VIEW_HEIGHT}; width: auto; margin: 0 auto; display: block; border-radius: 8px; object-fit: contain;" alt="Desk View | Robot View" /></div>'
    
    # 根据是否有示范视频决定UI阶段
    # Determine UI phase based on whether there is a demonstration video
    if has_demo_video:
        reset_ui_phase(uid)  # 设置为 "watching_demo" / Set to "watching_demo"
        return (
            uid,
            gr.update(visible=False), # login_group
            gr.update(visible=False), # env_selection_group
            gr.update(visible=True),  # main_interface
            f"Logged in as {username}", 
            gr.update(value=img, interactive=False), 
            f"Ready. Task: {env_id}",
            gr.update(choices=radio_choices, value=None),
            goal_text, 
            "No need for coordinates", 
            gr.update(value=combined_html), 
            demo_video_path,
            f"Current Task: {env_id} (Episode {episode_idx})",
            "Progress: N/A",
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False), # exec_btn (第一阶段禁用 / disabled in first phase)
            gr.update(visible=True),  # demo_video_group (第一阶段显示 / shown in first phase)
            gr.update(visible=False), # combined_view_group (第一阶段隐藏 / hidden in first phase)
            gr.update(visible=False), # operation_zone_group (第一阶段隐藏 / hidden in first phase)
            gr.update(visible=True, interactive=False),  # confirm_demo_btn (第一阶段显示，初始禁用 / shown, initially disabled)
            gr.update(visible=True, interactive=True),  # play_video_btn (第一阶段显示 / shown)
            gr.update(visible=False),  # coords_group (初始化时隐藏 / hidden initially)
            # 【修改】任务提示延迟加载功能（select_env_id函数 - 有示范视频的情况）：
            # 之前：任务加载时自动调用 get_task_hint(env_id) 显示提示内容
            # 现在：初始值设为空字符串，用户需要点击"Show Hint"按钮才会通过 show_task_hint() 函数加载并显示提示
            gr.update(value=""),  # note2 - 任务提示（延迟加载，初始为空）
            gr.update(value=""),  # note2_demo - 演示任务提示（延迟加载，初始为空）
            ""  # loading_overlay - 【关键】清空加载遮罩层：返回空字符串，清空 loading_overlay 组件内容，使全屏遮罩层自动隐藏
        )
    else:
        # 没有示范视频：直接进入执行阶段
        # No demonstration video: proceed directly to execution phase
        set_ui_phase(uid, "executing_task")
        
        # 初始化Reference Views队列
        # Initialize Reference Views queue
        if session.base_frames:
            from state_manager import FRAME_QUEUES
            
            current_base_count = len(session.base_frames) if session.base_frames else 0
            
            if uid not in FRAME_QUEUES:
                FrameQueueManager.init_queue(uid, current_base_count)
            
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
                    for _ in range(10):
                        try:
                            frame_copy = np.copy(last_frame) if isinstance(last_frame, np.ndarray) else last_frame
                            queue_info["frame_queue"].put(frame_copy, block=False)
                        except queue.Full:
                            break
        
        return (
            uid,
            gr.update(visible=False), # login_group
            gr.update(visible=False), # env_selection_group
            gr.update(visible=True),  # main_interface
            f"Logged in as {username}", 
            gr.update(value=img, interactive=False), 
            f"Ready. Task: {env_id}",
            gr.update(choices=radio_choices, value=None),
            goal_text, 
            "No need for coordinates", 
            gr.update(value=combined_html), 
            None,  # demo_video_path
            f"Current Task: {env_id} (Episode {episode_idx})",
            "Progress: N/A",
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=True), # exec_btn
            gr.update(visible=False), # demo_video_group (无视频，隐藏 / no video, hidden)
            gr.update(visible=True),  # combined_view_group
            gr.update(visible=True),  # operation_zone_group (直接显示 / shown directly)
            gr.update(visible=False), # confirm_demo_btn (无视频，隐藏 / no video, hidden)
            gr.update(visible=False, interactive=True),  # play_video_btn
            gr.update(visible=False),  # coords_group (初始化时隐藏 / hidden initially)
            # 【修改】任务提示延迟加载功能（select_env_id函数 - 无示范视频的情况）：
            # 之前：任务加载时自动调用 get_task_hint(env_id) 显示提示内容
            # 现在：初始值设为空字符串，用户需要点击"Show Hint"按钮才会通过 show_task_hint() 函数加载并显示提示
            gr.update(value=""),  # note2 - 任务提示（延迟加载，初始为空）
            gr.update(value=""),  # note2_demo - 演示任务提示（延迟加载，初始为空）
            ""  # loading_overlay - 【关键】清空加载遮罩层：返回空字符串，清空 loading_overlay 组件内容，使全屏遮罩层自动隐藏
        )


def load_next_task_wrapper(username, uid):
    """
    Wrapper to just reload the user's current status (which should be next task if updated).
    如果当前任务已有 actions，则创建新的 attempt。
    对于 user_test，next task 时跳转回 env_id 选择界面。
    For user_test, jump back to env_id selection interface when next task.
    """
    # 特殊处理：如果是 user_test，next task 时跳转回选择界面
    # Special handling: if user_test, jump back to selection interface on next task
    if username.endswith("_test"):
        return (
            uid,
            gr.update(visible=False), # login_group
            gr.update(visible=True),  # env_selection_group
            gr.update(visible=False), # main_interface
            f"Please select an environment ID to continue.", # login_msg
            gr.update(value=None, interactive=False), "", # img, status
            gr.update(choices=[], value=None), # options
            "", "No need for coordinates", # goal, coords
            gr.update(value="<div id='combined_view_html'><p>等待选择环境...</p></div>"), None, # combined_html, demo_video
            "", "", # task_info, progress_info
            gr.update(interactive=True), # login_btn
            gr.update(interactive=False), # next_task_btn
            gr.update(interactive=False), # exec_btn
            gr.update(visible=False), # demo_video_group
            gr.update(visible=False), # combined_view_group
            gr.update(visible=False), # operation_zone_group
            gr.update(visible=False),  # confirm_demo_btn
            gr.update(visible=False, interactive=True),  # play_video_btn
            gr.update(visible=False),  # coords_group
            gr.update(value=""),  # note2
            gr.update(value=""),  # note2_demo
            ""  # loading_overlay - 【关键】清空加载遮罩层：返回空字符串，清空 loading_overlay 组件内容，使全屏遮罩层自动隐藏
        )
    
    if username:
        # Check lease before proceeding
        try:
            user_manager.assert_lease(username, uid)
        except LeaseLost as e:
            # Raise error to be caught by Gradio and displayed
            raise gr.Error(f"You have been logged in elsewhere. This page is no longer valid. Please refresh the page to log in again.\n{str(e)}")
        
        # 更新session活动时间（加载下一个任务操作）
        if uid:
            update_session_activity(uid)
        
        success, msg, status = user_manager.login(username, uid)
        if success and not status["is_done_all"]:
            current_task = status["current_task"]
            env_id = current_task["env_id"]
            ep_num = current_task["episode_idx"]
            
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
            raise gr.Error(f"You have been logged in elsewhere. This page is no longer valid. Please refresh the page to log in again.\n{str(e)}")
    
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
            raise gr.Error(f"You have been logged in elsewhere. This page is no longer valid. Please refresh the page to log in again.\n{str(e)}")
    
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
                    option_label = label
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


def switch_to_record_mode(username, uid):
    """
    Switch to record mode: proceed with login using original username.
    切换到记录模式：使用原始用户名继续登录。
    
    功能说明：
    - 用户在模式选择页面点击 "Record Mode" 按钮时调用此函数
    - 使用原始用户名（不含 _test 后缀）调用 login_and_load_task() 进行登录和任务加载
    - 返回值中包含 loading_overlay 的输出（空字符串），用于清空加载遮罩层
    
    返回值说明：
    - 返回 login_and_load_task() 的结果，但将 landing_group 设置为隐藏
    - 最后一个返回值是 loading_overlay（空字符串），用于清空遮罩层
    """
    if not username:
        # 如果没有用户名，返回默认状态，包括 loading_overlay 的空字符串
        return (None, gr.update(visible=True)) + (gr.update(visible=True),) + tuple([gr.update()] * 23) + ("",)  # 最后一个是 loading_overlay - 清空遮罩层

    # Call login_and_load_task with original username
    # 使用原始用户名调用 login_and_load_task
    results = login_and_load_task(username, uid)
    
    # We need to return:
    # uid, landing_group(hidden), + rest of results (starting from login_group)
    # results[0] is uid
    # results[1] is login_group
    
    return (results[0], gr.update(visible=False)) + results[1:]


def switch_to_test_mode(username, uid):
    """
    Switch to test mode: append _test to username and proceed with login.
    切换到测试模式：在用户名后附加 _test 并继续登录。
    
    功能说明：
    - 用户在模式选择页面点击 "Free Try Mode" 按钮时调用此函数
    - 在用户名后添加 _test 后缀，构造测试用户名（如 user1 -> user1_test）
    - 使用测试用户名调用 login_and_load_task() 进行登录和任务加载
    - 返回值中包含 loading_overlay 的输出（空字符串），用于清空加载遮罩层
    
    返回值说明：
    - 返回 login_and_load_task() 的结果，但将 landing_group 设置为隐藏
    - 在 exec_btn 和 demo_video_group 之间插入 test_username 到 username_state
    - 最后一个返回值是 loading_overlay（空字符串），用于清空遮罩层
    """
    if not username:
        # 如果没有用户名，返回默认状态，包括 loading_overlay 的空字符串
        return (None, gr.update(visible=True)) + (gr.update(visible=True),) + tuple([gr.update()] * 23) + ("",)  # 最后一个是 loading_overlay - 清空遮罩层

    # Construct test username
    # 构造测试用户名
    if not username.endswith("_test"):
        parts = username.split('_')
        if len(parts) >= 2 and parts[0].startswith("user"):
             test_username = f"{parts[0]}_test"
        else:
             test_username = f"{username}_test"
    else:
        test_username = username
        
    print(f"Switching to Test Mode: {username} -> {test_username}")
    
    # Call login_and_load_task with test username
    # 使用测试用户名调用 login_and_load_task
    results = login_and_load_task(test_username, uid)
    
    # Updated return to include username_state
    # results[0] is uid
    # results[1] ... results[16] are login_group ... exec_btn
    # results[17] ... are demo_video_group ...
    
    # We need to insert test_username between exec_btn and demo_video_group
    # 我们需要在 exec_btn 和 demo_video_group 之间插入 test_username
    
    return (
        results[0],                 # uid
        gr.update(visible=False),   # landing_group
    ) + results[1:17] + (            # login_group ... exec_btn
        test_username,              # username_state (NEW)
    ) + results[17:]                 # demo_video_group ...


def back_to_landing_page(username, uid):
    """
    从执行界面返回到模式选择页面（Landing Page）。
    允许用户在成功进入环境执行界面后，通过点击按钮回退到模式选择页面，重新选择测试模式或录制模式。
    
    Args:
        username: 当前用户名（可能以 _test 结尾）
        uid: 当前会话的唯一标识符
    
    Returns:
        返回UI状态更新，用于显示模式选择页面（landing page）
    """
    if not username:
        # 如果没有用户名，显示登录页面
        return (
            uid,                                    # uid_state: 会话唯一标识符
            gr.update(visible=False),               # loading_group: 隐藏加载组
            gr.update(visible=False),               # landing_group: 隐藏模式选择组
            gr.update(visible=True),                # login_group: 显示登录组
            gr.update(visible=False),               # env_selection_group: 隐藏环境选择组
            gr.update(visible=False),               # main_interface: 隐藏主界面
            "",                                     # login_msg: 登录消息（空）
            gr.update(value=None, interactive=False), None,  # img_display: 图片显示（清空），log_output: 日志输出（无）
            gr.update(choices=[], value=None),      # options_radio: 选项单选（清空）
            "", "No need for coordinates",         # goal_box: 目标框（清空），coords_box: 坐标框（无需坐标）
            gr.update(value="<div id='combined_view_html'><p>等待登录...</p></div>"), None,  # combined_display: 组合视图（等待登录），video_display: 视频显示（无）
            "", "",                                 # task_info_box: 任务信息（清空），progress_info_box: 进度信息（清空）
            gr.update(interactive=True),            # login_btn: 登录按钮（可交互）
            gr.update(interactive=False),           # next_task_btn: 下一个任务按钮（不可交互）
            gr.update(interactive=False),           # exec_btn: 执行按钮（不可交互）
            "",                                     # username_state: 用户名状态（清空）
            gr.update(visible=False),               # demo_video_group: 演示视频组（隐藏）
            gr.update(visible=False),               # combined_view_group: 组合视图组（隐藏）
            gr.update(visible=False),               # operation_zone_group: 操作区域组（隐藏）
            gr.update(visible=False),               # confirm_demo_btn: 确认演示按钮（隐藏）
            gr.update(visible=False, interactive=True),  # play_video_btn: 播放视频按钮（隐藏但可交互）
            gr.update(visible=False),               # coords_group: 坐标组（隐藏）
            gr.update(value=""),     # note2: 提示信息（清空）
            gr.update(value="")      # note2_demo: 演示提示信息（清空）
        )
    
    # 【Free Try Mode 立即销毁】如果用户名以 _test 结尾，立即销毁环境释放资源
    # 目的：在free try mode下，用户返回模式选择页面时立即释放环境资源（RAM/VRAM），而不是等待超时
    # 这样可以避免资源浪费，让其他用户能够更快地使用系统资源
    if username.endswith("_test") and uid:
        print(f"Free Try Mode detected: {username}, destroying session {uid} immediately")
        try:
            # 调用cleanup_session销毁环境，包括：
            # 1. 关闭ProcessSessionProxy（终止工作进程，释放RAM/VRAM）
            # 2. 清理所有相关的状态数据（任务索引、坐标点击、选项选择、帧队列等）
            # 3. 清理流生成ID（终止旧的MJPEG流）
            cleanup_session(uid)
            print(f"Session {uid} destroyed successfully for free try mode user {username}")
        except Exception as e:
            # 如果销毁过程中出现异常，记录错误但不影响UI返回
            print(f"Error destroying session {uid} for free try mode: {e}")
            traceback.print_exc()
    
    # 提取原始用户名（如果存在 _test 后缀则去掉）
    # 例如：user1_test -> user1, user1_VideoPlaceOrder -> user1
    if username.endswith("_test"):
        # 如果用户名以 _test 结尾，提取基础用户名
        parts = username.split('_')
        if len(parts) >= 2 and parts[0].startswith("user"):
            # 如果格式是 user1_test，提取 user1
            original_username = parts[0]
        else:
            # 否则直接去掉 _test 后缀
            original_username = username.replace("_test", "")
    else:
        # 如果用户名不以 _test 结尾，尝试提取基础用户名
        # 例如：user1_VideoPlaceOrder -> user1
        parts = username.split('_')
        if len(parts) >= 2 and parts[0].startswith("user"):
            # 如果格式是 user1_XXX，提取 user1
            original_username = parts[0]
        else:
            # 否则使用原用户名
            original_username = username
    
    # 检查原始用户名是否存在于 user_tasks 中
    # 如果不存在，尝试查找匹配的用户名
    if original_username not in user_manager.user_tasks:
        # 获取所有可用的用户名列表
        available_users = list(user_manager.user_tasks.keys())
        # 查找匹配的用户名（以原始用户名开头或完全匹配）
        matching_users = [u for u in available_users if u.startswith(original_username + "_") or u == original_username]
        if matching_users:
            # 使用第一个匹配的用户（优先选择非测试版本）
            # 例如：如果有 user1_VideoPlaceOrder 和 user1_test，优先选择 user1_VideoPlaceOrder
            non_test_users = [u for u in matching_users if not u.endswith("_test")]
            original_username = non_test_users[0] if non_test_users else matching_users[0]
        else:
            # 如果找不到匹配，使用原用户名
            original_username = username
    
    print(f"返回模式选择页面: {username} -> {original_username}")
    
    # 返回显示模式选择页面的状态（与 init_app 显示 landing_group 时相同）
    return (
        uid,                                    # uid_state: 会话唯一标识符（保持不变）
        gr.update(visible=False),               # loading_group: 隐藏加载组
        gr.update(visible=True),                # landing_group: 显示模式选择组（关键：显示模式选择页面）
        gr.update(visible=False),               # login_group: 隐藏登录组
        gr.update(visible=False),               # env_selection_group: 隐藏环境选择组
        gr.update(visible=False),               # main_interface: 隐藏主界面
        f"Welcome {original_username}",        # login_msg: 欢迎消息
        gr.update(value=None, interactive=False), None,  # img_display: 图片显示（清空），log_output: 日志输出（无）
        gr.update(choices=[], value=None),      # options_radio: 选项单选（清空）
        "", "No need for coordinates",         # goal_box: 目标框（清空），coords_box: 坐标框（无需坐标）
        gr.update(value="<div id='combined_view_html'><p>Select Mode...</p></div>"), None,  # combined_display: 组合视图（选择模式），video_display: 视频显示（无）
        "", "",                                 # task_info_box: 任务信息（清空），progress_info_box: 进度信息（清空）
        gr.update(interactive=True),             # login_btn: 登录按钮（可交互）
        gr.update(interactive=False),           # next_task_btn: 下一个任务按钮（不可交互）
        gr.update(interactive=False),           # exec_btn: 执行按钮（不可交互）
        original_username,                      # username_state: 用户名状态（设置为原始用户名，不含 _test）
        gr.update(visible=False),               # demo_video_group: 演示视频组（隐藏）
        gr.update(visible=False),               # combined_view_group: 组合视图组（隐藏）
        gr.update(visible=False),               # operation_zone_group: 操作区域组（隐藏）
        gr.update(visible=False),               # confirm_demo_btn: 确认演示按钮（隐藏）
        gr.update(visible=False, interactive=True),  # play_video_btn: 播放视频按钮（隐藏但可交互）
        gr.update(visible=False),               # coords_group: 坐标组（隐藏）
        gr.update(value=""),      # note2: 提示信息（清空）
        gr.update(value="")       # note2_demo: 演示提示信息（清空）
    )


def init_app(request: gr.Request):
    """
    处理初始页面加载。
    如果URL中包含 'user' 或 'username' 查询参数，显示选择模式页面 (Landing Page)。
    Handle initial page load.
    If URL contains 'user' or 'username' query parameters, show Select Mode page (Landing Page).
    
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
    # 2. landing_group (NEW)
    # 3. login_group
    # ...
    
    default_outputs = (
        None, 
        gr.update(visible=False), # loading_group (hide it)
        gr.update(visible=False), # landing_group (hide it)
        gr.update(visible=True), # login_group (show it)
        gr.update(visible=False), # env_selection_group
        gr.update(visible=False), # main_interface
        "", 
        gr.update(value=None, interactive=False), None, 
        gr.update(choices=[], value=None), 
        "", "No need for coordinates", 
        gr.update(value="<div id='combined_view_html'><p>等待登录...</p></div>"), None, 
        "", "", 
        gr.update(interactive=True), 
        gr.update(interactive=False), 
        gr.update(interactive=False),
        "",  # username_state
        gr.update(visible=False), # demo_video_group
        gr.update(visible=False), # combined_view_group
        gr.update(visible=False), # operation_zone_group
        gr.update(visible=False), # confirm_demo_btn
        gr.update(visible=False, interactive=True),  # play_video_btn
        gr.update(visible=False),  # coords_group (初始化时隐藏)
        gr.update(value=""),  # note2
        gr.update(value="")  # note2_demo
    )
    
    if username:
        # 检查用户是否存在
        # Check if user exists
        if username in user_manager.user_tasks:
            # 显示 Landing Page
            # Show Landing Page
            print(f"URL Login: Detected '{username}', showing landing page.")
            
            return (
                None,                       # uid
                gr.update(visible=False),   # loading_group
                gr.update(visible=True),    # landing_group (SHOW)
                gr.update(visible=False),   # login_group
                gr.update(visible=False),   # env_selection_group
                gr.update(visible=False),   # main_interface
                f"Welcome {username}",      # login_msg
                gr.update(value=None, interactive=False), None, # img, status
                gr.update(choices=[], value=None), # options
                "", "No need for coordinates", # goal, coords
                gr.update(value="<div id='combined_view_html'><p>Select Mode...</p></div>"), None, # combined, video
                "", "", # task, progress
                gr.update(interactive=True), # login_btn
                gr.update(interactive=False), # next_task_btn
                gr.update(interactive=False), # exec_btn
                username,                   # username_state (Set this!)
                gr.update(visible=False),   # demo_video_group
                gr.update(visible=False),   # combined_view_group
                gr.update(visible=False),   # operation_zone_group
                gr.update(visible=False),   # confirm_demo_btn
                gr.update(visible=False, interactive=True), # play_video_btn
                gr.update(visible=False),   # coords_group
                gr.update(value=""), # note2
                gr.update(value="")  # note2_demo
            )
        else:
            # 用户名不存在，显示错误消息但仍显示登录界面
            # Username does not exist, show error message but still show login interface
            print(f"自动登录失败: 用户名 '{username}' 不存在于用户列表中")
            error_msg = f"⚠️ 用户名 '{username}' 不存在。请从下拉列表中选择有效的用户名。"
            return (
                None,
                gr.update(visible=False),  # loading_group
                gr.update(visible=False),  # landing_group
                gr.update(visible=True),   # login_group (显示登录界面)
                gr.update(visible=False),  # env_selection_group
                gr.update(visible=False),  # main_interface
                error_msg,                 # login_msg (显示错误消息)
                gr.update(value=None, interactive=False), None,
                gr.update(choices=[], value=None),
                "", "No need for coordinates",
                gr.update(value="<div id='combined_view_html'><p>等待登录...</p></div>"), None,
                "", "",
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=False),
                "",  # username_state
                gr.update(visible=False), # demo_video_group
                gr.update(visible=False), # combined_view_group
                gr.update(visible=False), # operation_zone_group
                gr.update(visible=False), # confirm_demo_btn
                gr.update(visible=False, interactive=True),  # play_video_btn
                gr.update(visible=False),  # coords_group
                gr.update(value=""),  # note2
                gr.update(value="")  # note2_demo
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
        return None, "Session Error", gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=False)
    
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
        return session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW), "Error: No action selected", gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), gr.update(visible=False)

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
            return current_img, error_msg, gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), gr.update(visible=True)

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
                    option_label = label
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
                        option_label = label
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
        # 使用固定模板，所有行长度一致（32个字符）
        if final_log_status == "success":
            status = """********************************
****   episode success      ****
********************************
  ---please press next task----   """
        else:
            status = """********************************
****   episode failed       ****
********************************
  ---please press next task----   """

        # Update user progress (但不更新 progress_info_box，等用户按 next task/refresh 时再更新)
        if username:
            # 获取 seed（直接使用 session.seed，如果不存在则为 None）
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
    
    return img, status, task_update, progress_update, next_task_update, exec_btn_update, coords_group_update
