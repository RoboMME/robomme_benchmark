"""
Gradio回调函数模块
响应UI事件，调用业务逻辑，返回UI更新
"""
import gradio as gr
import numpy as np
import time
import traceback
import queue
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
    get_ui_phase,
    set_ui_phase,
    reset_ui_phase,
)
from streaming_service import FrameQueueManager, cleanup_frame_queue
from image_utils import draw_marker, save_video, concatenate_frames_horizontally
from user_manager import user_manager, LeaseLost
from logger import log_session, log_user_action, create_new_attempt, has_existing_actions
from config import USE_SEGMENTED_VIEW, REFERENCE_VIEW_HEIGHT, should_show_demo_video


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
            gr.update(visible=False)  # confirm_demo_btn
        )
    
    # Login success - Load current task
    if status["is_done_all"]:
        # 保存任务索引（已完成所有任务）
        set_task_index(uid, status['total_tasks'] - 1, status['total_tasks'])
        task_info = get_task_index(uid)
        task_idx = task_info["task_index"]
        total = task_info["total_tasks"]
        # 生成 HTML 内容，包含 MJPEG 流
        combined_html = f'<div id="combined_view_html"><img src="/video_feed/{uid}" style="max-width: 100%; height: {REFERENCE_VIEW_HEIGHT}; width: auto; margin: 0 auto; display: block; border: 2px solid #3b82f6; border-radius: 8px; object-fit: contain;" alt="Desk View | Robot View" /></div>'
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
            gr.update(visible=False)  # confirm_demo_btn
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
    
    img, load_msg = session.load_episode(env_id, int(ep_num))
    
    if img is None:
         # 即使加载失败，也保存任务索引
         set_task_index(uid, status['current_index'], status['total_tasks'])
         task_info = get_task_index(uid)
         task_idx = task_info["task_index"]
         total = task_info["total_tasks"]
         # 生成 HTML 内容，包含 MJPEG 流
         combined_html = f'<div id="combined_view_html"><img src="/video_feed/{uid}" style="max-width: 100%; height: {REFERENCE_VIEW_HEIGHT}; width: auto; margin: 0 auto; display: block; border: 2px solid #3b82f6; border-radius: 8px; object-fit: contain;" alt="Desk View | Robot View" /></div>'
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
            gr.update(visible=False)  # confirm_demo_btn
        )
        
    # Success loading
    goal_text = f"{session.language_goal}"
    options = session.available_options
    radio_choices = [(opt_label, opt_idx) for opt_label, opt_idx in options]
    
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
    combined_html = f'<div id="combined_view_html"><img src="/video_feed/{uid}" style="max-width: 100%; height: {REFERENCE_VIEW_HEIGHT}; width: auto; margin: 0 auto; display: block; border: 2px solid #3b82f6; border-radius: 8px; object-fit: contain;" alt="Desk View | Robot View" /></div>'
    
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
            gr.update(visible=True),  # confirm_demo_btn (第一阶段显示)
            gr.update(visible=False)  # coords_group (初始化时隐藏)
        )
    else:
        # 没有示范视频：直接进入执行阶段
        set_ui_phase(uid, "executing_task")
        
        # 初始化Reference Views队列（如果没有demo video，需要立即显示Reference Views）
        if session.base_frames or session.wrist_frames:
            from state_manager import FRAME_QUEUES
            
            # 初始化队列（如果还没有）
            # 注意：使用0作为pre_base_count/pre_wrist_count，表示这是初始frames，不应该被清空
            if uid not in FRAME_QUEUES:
                FrameQueueManager.init_queue(uid, 0, 0)
            
            # 拼接初始frames
            initial_frames = concatenate_frames_horizontally(
                session.base_frames, 
                session.wrist_frames
            )
            
            # 将初始frames加入队列
            queue_info = FRAME_QUEUES.get(uid)
            if queue_info:
                for frame in initial_frames:
                    try:
                        queue_info["frame_queue"].put(frame, block=False)
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
            gr.update(visible=False)  # coords_group (初始化时隐藏)
        )


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
        
        # 初始化队列（如果还没有）
        # 使用0作为pre_base_count/pre_wrist_count，表示这是初始frames
        if uid not in FRAME_QUEUES:
            FrameQueueManager.init_queue(uid, 0, 0)
        
        # 拼接初始frames
        initial_frames = concatenate_frames_horizontally(
            session.base_frames, 
            session.wrist_frames
        )
        
        # 将初始frames加入队列
        queue_info = FRAME_QUEUES.get(uid)
        if queue_info:
            for frame in initial_frames:
                try:
                    queue_info["frame_queue"].put(frame, block=False)
                except queue.Full:
                    break
    
    # 返回UI更新：隐藏示范视频，显示Combined View和操作区域
    # 同时更新 exec_btn 为可交互状态
    return (
        gr.update(visible=False),  # demo_video_group
        gr.update(visible=True),   # combined_view_group (修复：应该显示)
        gr.update(visible=True),   # operation_zone_group
        gr.update(visible=False),  # confirm_demo_btn
        gr.update(interactive=True),  # exec_btn - 启用执行按钮
        gr.update(visible=False)   # coords_group (确认demo后，还未选择选项，隐藏)
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
             return "please click the image", gr.update(interactive=True), gr.update(visible=True)
    
    return default_msg, gr.update(interactive=False), gr.update(visible=False)


def init_app(request: gr.Request):
    """
    Handle initial page load. 
    If 'user' query parameter is present, automatically login as that user.
    """
    params = request.query_params
    username = params.get('user')
    
    # Default outputs if no auto-login
    # uid, loading_group, login_group, main_interface, login_msg, img, log, options, goal, coords, combined, video, task, progress, login_btn, next_btn, exec_btn, username_state, demo_video_group, combined_view_group, operation_zone_group, confirm_demo_btn, coords_group
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
        gr.update(visible=False)  # coords_group (初始化时隐藏)
    )
    
    if username:
        # Check if user exists
        if username in user_manager.user_tasks:
            # Auto login
            # We need to pass a uid. Let's create one or pass None and let logic handle it.
            # login_and_load_task handles uid=None by creating a new one.
            results = login_and_load_task(username, None)
            
            # results[0] is uid
            # results[1] is login_group update
            # results[2] is main_interface update
            
            # New structure:
            # (uid, loading_group=False, login_group=False, main_interface=True, ...rest...)
            
            # Since login_and_load_task returns login_group update as results[1], we can use it but maybe force it to False just in case
            # Actually results[1] should be visible=False from login_and_load_task on success
            
            new_results = (
                results[0],                 # uid
                gr.update(visible=False),   # loading_group
            ) + results[1:] + (username,)
            
            return new_results
    
    return default_outputs


def execute_step(uid, username, option_idx, coords_str):
    # Check lease first
    if username:
        try:
            user_manager.assert_lease(username, uid)
        except LeaseLost as e:
            raise gr.Error(f"LeaseLost: {str(e)}")
    
    session = get_session(uid)
    if not session:
        return None, "Session Error", gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=False)
    
    # 检查并初始化Reference Views（如果frames为空或队列不存在）
    from state_manager import FRAME_QUEUES
    frames_exist = session.base_frames or session.wrist_frames
    queue_exists = uid in FRAME_QUEUES
    
    if not frames_exist:
        # 从环境中读取初始frames
        session.update_observation(use_segmentation=USE_SEGMENTED_VIEW)
        
        # 如果有frames了，将初始frames加入队列
        if session.base_frames or session.wrist_frames:
            
            # 初始化队列（如果还没有）
            if uid not in FRAME_QUEUES:
                FrameQueueManager.init_queue(uid, 0, 0)
            
            # 拼接初始frames
            initial_frames = concatenate_frames_horizontally(
                session.base_frames, 
                session.wrist_frames
            )
            
            # 将初始frames加入队列
            queue_info = FRAME_QUEUES.get(uid)
            if queue_info:
                for frame in initial_frames:
                    try:
                        queue_info["frame_queue"].put(frame, block=False)
                    except queue.Full:
                        break
    elif frames_exist and not queue_exists:
        # frames存在但队列不存在，初始化队列并加入frames
        # 初始化队列
        FrameQueueManager.init_queue(uid, len(session.base_frames), len(session.wrist_frames))
        
        # 拼接初始frames
        initial_frames = concatenate_frames_horizontally(
            session.base_frames, 
            session.wrist_frames
        )
        
        # 将初始frames加入队列
        queue_info = FRAME_QUEUES.get(uid)
        if queue_info:
            for frame in initial_frames:
                try:
                    queue_info["frame_queue"].put(frame, block=False)
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
                if coords_str.strip() not in ["please click the image", "No need for coordinates"]:
                    is_valid_coords = True
            except:
                pass
        
        # 如果需要坐标但没有有效坐标，返回错误提示
        if not is_valid_coords:
            current_img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)
            error_msg = "please click the image before execute!"
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
    pre_base_frame_count = len(session.base_frames)
    pre_wrist_frame_count = len(session.wrist_frames)
    
    # 检查是否是第一次execute（队列已存在且有frames，且pre_base_count等于当前frames数量）
    # 如果是第一次execute，不应该清空队列中的初始frames
    is_first_execute = False
    if uid in FRAME_QUEUES:
        queue_info = FRAME_QUEUES.get(uid)
        if queue_info and queue_info["frame_queue"].qsize() > 0:
            # 如果pre_base_count等于当前frames数量，说明这是第一次execute，不应该清空队列
            if pre_base_frame_count == len(session.base_frames) and pre_wrist_frame_count == len(session.wrist_frames):
                is_first_execute = True
    
    # 初始化队列和启动监控线程（用于流式输出）
    # 如果是第一次execute，使用0作为pre_base_count/pre_wrist_count，避免清空队列
    if is_first_execute:
        FrameQueueManager.init_queue(uid, 0, 0)
    else:
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
    print(f"Executing step: Opt {option_idx}, Coords {click_coords}")
    img, status, done = session.execute_action(option_idx, click_coords)
    
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
            
            log_user_action(
                username=username,
                env_id=session.env_id,
                episode_idx=session.episode_idx,
                action_data={
                    "option_idx": option_idx,  # execute时使用的option（最后一次选择的）
                    "option_label": option_label,
                    "final_coordinates": final_coordinates,  # 最后执行的坐标
                    "final_coords_str": final_coords_str,  # 最后执行的坐标字符串
                    "final_image_array": final_image_array,  # 最后执行时的图片
                    "option_selects_before_execute": option_selects_before_execute,  # execute之前所有的option选择
                    "coordinate_clicks_before_execute": coordinate_clicks_before_execute,  # execute之前所有的坐标点击（已包含 image_array）
                    "status": status,
                    "done": done
                }
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
        status += " [EPISODE COMPLETE]"
        
        # Determine final status for logging
        final_log_status = "failed"
        if "SUCCESS" in status:
            final_log_status = "success"

        # Log session data to experiment_logs.jsonl
        try:
            log_session({
                "uid": uid,
                "username": username if username else "unknown",
                "env_id": session.env_id,
                "episode_idx": session.episode_idx,
                "language_goal": session.language_goal,
                "difficulty": session.difficulty if hasattr(session, 'difficulty') and session.difficulty is not None else None,
                "finished": True,
                "status": final_log_status
            })
        except Exception as e:
            print(f"Error logging session: {e}")
            traceback.print_exc()
        
        # Update user progress (但不更新 progress_info_box，等用户按 next task/refresh 时再更新)
        if username:
            user_status = user_manager.complete_current_task(username)
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
