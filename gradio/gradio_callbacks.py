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
)
from streaming_service import FrameQueueManager, cleanup_frame_queue
from image_utils import draw_marker, save_video, concatenate_frames_horizontally
from user_manager import user_manager, LeaseLost
from logger import log_user_action, create_new_attempt, has_existing_actions
from config import USE_SEGMENTED_VIEW, REFERENCE_VIEW_HEIGHT, should_show_demo_video
from process_session import ScrewPlanFailureError
from note_content import get_task_hint


def login_and_load_task(username, uid):
    """
    Handle user login and load their current task.
    """
    if not uid:
        uid = create_session()
    
    # Pass uid to login for force takeover mechanism
    success, msg, status = user_manager.login(username, uid)
    
    if not success:
        # Login failed
        return (
            uid,
            gr.update(visible=True), # login_group
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
            gr.update(value=get_task_hint("")),  # note2
            gr.update(value=get_task_hint(""))  # note2_demo
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
            gr.update(value=get_task_hint("")),  # note2
            gr.update(value=get_task_hint(""))  # note2_demo
        )

    current_task = status["current_task"]
    env_id = current_task["env_id"]
    ep_num = current_task["episode_idx"]
    
    # Load the environment
    session = get_session(uid)
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
            gr.update(visible=False),
            gr.update(visible=True),
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
            gr.update(value=get_task_hint(env_id)),  # note2
            gr.update(value=get_task_hint(env_id))  # note2_demo
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
            gr.update(visible=False), # Login hidden
            gr.update(visible=True),  # Main visible
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
            gr.update(value=get_task_hint(env_id)),  # note2
            gr.update(value=get_task_hint(env_id))  # note2_demo
        )
    else:
        # 没有示范视频：直接进入执行阶段
        set_ui_phase(uid, "executing_task")

        
        # 初始化Reference Views队列（如果没有demo video，需要立即显示Reference Views）
        # 注意：即使手动添加了初始帧，generate_mjpeg_stream 也有 fallback 机制直接从 session 获取帧
        # 这确保了即使队列初始化时序有问题，用户也能看到当前状态
        if session.base_frames or session.wrist_frames:
            from state_manager import FRAME_QUEUES
            
            # 初始化队列：传入当前frames数量作为监控起始点
            # 监控线程会从这些帧之后开始检测新帧，不会重复添加已存在的帧
            current_base_count = len(session.base_frames) if session.base_frames else 0
            current_wrist_count = len(session.wrist_frames) if session.wrist_frames else 0
            
            if uid not in FRAME_QUEUES:
                FrameQueueManager.init_queue(uid, current_base_count, current_wrist_count)
            
            # 手动添加最后一帧到队列（重复多次），确保初始显示
            # 这是可选的优化，因为 generate_mjpeg_stream 有 fallback 机制
            last_base_frame = session.base_frames[-1] if session.base_frames else None
            last_wrist_frame = session.wrist_frames[-1] if session.wrist_frames else None
            
            if last_base_frame is not None or last_wrist_frame is not None:
                env_id = getattr(session, 'env_id', None)
                last_frames = concatenate_frames_horizontally(
                    [last_base_frame] if last_base_frame is not None else [],
                    [last_wrist_frame] if last_wrist_frame is not None else [],
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
            gr.update(visible=False), # Login hidden
            gr.update(visible=True),  # Main visible
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
            gr.update(value=get_task_hint(env_id)),  # note2
            gr.update(value=get_task_hint(env_id))  # note2_demo
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
    
    # 设置阶段为执行任务
    set_ui_phase(uid, "executing_task")
    
    # 初始化Reference Views队列（确认demo后，需要显示Reference Views）
    session = get_session(uid)
    if session and (session.base_frames or session.wrist_frames):
        from state_manager import FRAME_QUEUES
        
        # 初始化队列（如果还没有，或者队列存在但不活跃）
        # 传入当前frames数量，这样监控线程就知道这些frames已存在，不会将它们作为"新"frames加入队列
        current_base_count = len(session.base_frames) if session.base_frames else 0
        current_wrist_count = len(session.wrist_frames) if session.wrist_frames else 0
        
        # 检查队列是否存在且活跃，如果不存在或不活跃，则初始化
        queue_info = FRAME_QUEUES.get(uid)
        if not queue_info or not queue_info.get("streaming_active", False):
            FrameQueueManager.init_queue(uid, current_base_count, current_wrist_count)
        
        # 只获取最后一帧并拼接
        last_base_frame = session.base_frames[-1] if session.base_frames else None
        last_wrist_frame = session.wrist_frames[-1] if session.wrist_frames else None
        
        if last_base_frame is not None or last_wrist_frame is not None:
            # 使用concatenate_frames_horizontally处理单帧（传入只包含最后一帧的列表）
            last_frames = concatenate_frames_horizontally(
                [last_base_frame] if last_base_frame is not None else [],
                [last_wrist_frame] if last_wrist_frame is not None else []
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
        gr.update(value=get_task_hint(env_id)),  # note2
        gr.update(value=get_task_hint(env_id))  # note2_demo
    )


def load_next_task_wrapper(username, uid):
    """
    Wrapper to just reload the user's current status (which should be next task if updated).
    如果当前任务已有 actions，则创建新的 attempt。
    """
    if username:
        # Check lease before proceeding
        try:
            user_manager.assert_lease(username, uid)
        except LeaseLost as e:
            # Raise error to be caught by Gradio and displayed
            raise gr.Error(f"You have been logged in elsewhere. This page is no longer valid. Please refresh the page to log in again.\n{str(e)}")
        
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


def init_app(request: gr.Request):
    """
    处理初始页面加载。
    如果URL中包含 'user' 或 'username' 查询参数，自动使用该用户名登录。
    
    支持的URL格式：
    - http://host:port/?user=username
    - http://host:port/?username=username
    
    Args:
        request: Gradio Request 对象，包含查询参数
    
    Returns:
        根据是否自动登录返回不同的UI状态
    """
    params = request.query_params if request else {}
    # 支持 'user' 和 'username' 两种参数名称
    username = params.get('user') or params.get('username')
    
    # Default outputs if no auto-login
    # uid, loading_group, login_group, main_interface, login_msg, img, log, options, goal, coords, combined, video, task, progress, login_btn, next_btn, exec_btn, username_state, demo_video_group, combined_view_group, operation_zone_group, confirm_demo_btn, coords_group, note2, note2_demo
    default_outputs = (
        None, 
        gr.update(visible=False), # loading_group (hide it)
        gr.update(visible=True), # login_group (show it)
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
        gr.update(value=get_task_hint("")),  # note2
        gr.update(value=get_task_hint(""))  # note2_demo
    )
    
    if username:
        # 检查用户是否存在
        if username in user_manager.user_tasks:
            # 自动登录
            print(f"自动登录: 从URL参数检测到用户名 '{username}'，正在自动登录...")
            # login_and_load_task 会在 uid=None 时自动创建新的 session
            results = login_and_load_task(username, None)
            
            # results[0] is uid
            # results[1] is login_group update
            # results[2] is main_interface update
            # ...
            # results[15] is exec_btn update
            # results[16] is demo_video_group update
            # ...
            # results[20] is coords_group update
            
            # 构建返回结果，确保 loading_group 隐藏，并在正确位置插入 username_state
            # outputs 顺序: uid, loading_group, login_group, ..., exec_btn, username_state, demo_video_group, ..., coords_group
            new_results = (
                results[0],                 # uid (outputs[0])
                gr.update(visible=False),   # loading_group (outputs[1])
            ) + results[1:16] + (           # login_group 到 exec_btn (outputs[2:17])
                username,                   # username_state (outputs[17])
            ) + results[16:]                # demo_video_group 到 coords_group (outputs[18:23])
            
            print(f"自动登录成功: 用户 '{username}' (uid: {results[0]})")
            return new_results
        else:
            # 用户名不存在，显示错误消息但仍显示登录界面
            print(f"自动登录失败: 用户名 '{username}' 不存在于用户列表中")
            error_msg = f"⚠️ 用户名 '{username}' 不存在。请从下拉列表中选择有效的用户名。"
            return (
                None,
                gr.update(visible=False),  # loading_group
                gr.update(visible=True),   # login_group (显示登录界面)
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
                gr.update(value=get_task_hint("")),  # note2
                gr.update(value=get_task_hint(""))  # note2_demo
            )
    
    return default_outputs


def execute_step(uid, username, option_idx, coords_str):
    # 记录用户按下 execute 按钮的瞬间时间戳
    execute_timestamp = datetime.now().isoformat()
    
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
        # 从 session 读取 non_demonstration_task_length，如果存在则加1作为限制，否则不设置限制
        max_execute = None
        if hasattr(session, 'non_demonstration_task_length') and session.non_demonstration_task_length is not None:
            max_execute = session.non_demonstration_task_length + 1
        
        if max_execute is not None:
            current_count = get_execute_count(username, session.env_id, session.episode_idx)
            if current_count >= max_execute:
                execute_limit_reached = True
    
    # 检查并初始化Reference Views（如果frames为空或队列不存在）
    from state_manager import FRAME_QUEUES
    frames_exist = session.base_frames or session.wrist_frames
    queue_exists = uid in FRAME_QUEUES
    
    if not frames_exist:
        # 从环境中读取初始frames
        session.update_observation(use_segmentation=USE_SEGMENTED_VIEW)
        
        # 如果有frames了，将最后一帧加入队列
        if session.base_frames or session.wrist_frames:
            
            # 初始化队列（如果还没有）
            # 传入当前frames数量，这样监控线程就知道这些frames已存在，不会将它们作为"新"frames加入队列
            current_base_count = len(session.base_frames) if session.base_frames else 0
            current_wrist_count = len(session.wrist_frames) if session.wrist_frames else 0
            
            if uid not in FRAME_QUEUES:
                FrameQueueManager.init_queue(uid, current_base_count, current_wrist_count)
            
            # 只获取最后一帧并拼接
            last_base_frame = session.base_frames[-1] if session.base_frames else None
            last_wrist_frame = session.wrist_frames[-1] if session.wrist_frames else None
            
            if last_base_frame is not None or last_wrist_frame is not None:
                # 使用concatenate_frames_horizontally处理单帧（传入只包含最后一帧的列表）
                last_frames = concatenate_frames_horizontally(
                    [last_base_frame] if last_base_frame is not None else [],
                    [last_wrist_frame] if last_wrist_frame is not None else []
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
        current_wrist_count = len(session.wrist_frames) if session.wrist_frames else 0
        
        FrameQueueManager.init_queue(uid, current_base_count, current_wrist_count)
        
        # 只获取最后一帧并拼接
        last_base_frame = session.base_frames[-1] if session.base_frames else None
        last_wrist_frame = session.wrist_frames[-1] if session.wrist_frames else None
        
        if last_base_frame is not None or last_wrist_frame is not None:
            # 使用concatenate_frames_horizontally处理单帧（传入只包含最后一帧的列表）
            last_frames = concatenate_frames_horizontally(
                [last_base_frame] if last_base_frame is not None else [],
                [last_wrist_frame] if last_wrist_frame is not None else []
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
    pre_wrist_frame_count = len(session.wrist_frames)
    
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
    FrameQueueManager.init_queue(uid, pre_base_frame_count, pre_wrist_frame_count)
    
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
