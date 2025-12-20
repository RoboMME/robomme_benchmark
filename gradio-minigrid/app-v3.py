import gradio as gr
import uuid
import numpy as np
import tempfile
import os
import traceback
from datetime import datetime
from PIL import Image, ImageDraw
from oracle_logic import OracleSession
from concurrent.futures import ThreadPoolExecutor
from user_manager import user_manager
from logger import log_session, log_user_action, create_new_attempt, has_existing_actions

# --- Global Session Storage ---
GLOBAL_SESSIONS = {}

# --- Task Index Storage (for Progress display) ---
# 存储每个session的任务索引和总任务数，用于直接读取Progress
TASK_INDEX_MAP = {}  # {uid: {"task_index": int, "total_tasks": int}}

# --- Coordinate Click Tracking (between actions) ---
# 跟踪每个session从上次action_execute到现在的所有coordinate_click
COORDINATE_CLICKS = {}  # {uid: [{"x": x, "y": y, "coords_str": "...", "image_array": ..., "timestamp": "..."}, ...]}

# --- Option Select Tracking (between actions) ---
# 跟踪每个session从上次action_execute到现在的所有option_select
OPTION_SELECTS = {}  # {uid: [{"option_idx": idx, "option_label": label, "timestamp": "..."}, ...]}

ENV_IDS = [
    "VideoPlaceOrder", "PickXtimes", "StopCube", "SwingXtimes", 
    "BinFill", "VideoUnmaskSwap", "VideoUnmask", "ButtonUnmaskSwap", 
    "ButtonUnmask", "VideoRepick", "VideoPlaceButton", "InsertPeg", 
    "MoveCube", "PatternLock", "RouteStick"
]


# --- Configuration ---
RESTRICT_VIDEO_PLAYBACK = True  # Set to False to enable controls
USE_SEGMENTED_VIEW = False  # Set to True to use segmented view, False to use original image

# --- Helper Functions ---

def get_session(uid):
    return GLOBAL_SESSIONS.get(uid)

def create_session():
    uid = str(uuid.uuid4())
    session = OracleSession()
    GLOBAL_SESSIONS[uid] = session
    return uid

def get_image_with_view_mode(uid, use_segmented=True):
    """
    根据视图模式获取图片
    
    Args:
        uid: session ID
        use_segmented: 是否使用分割视图
    
    Returns:
        PIL Image
    """
    session = get_session(uid)
    if not session:
        return None
    return session.get_pil_image(use_segmented=use_segmented)

# --- User Management Helpers ---

def login_and_load_task(username, uid):
    """
    Handle user login and load their current task.
    """
    if not uid:
        uid = create_session()
    
    success, msg, status = user_manager.login(username)
    
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
            None, None, # combined_video, demo_video
            "", "", # task_info, progress_info
            gr.update(interactive=True), # login_btn
            gr.update(interactive=False), # next_task_btn
            gr.update(interactive=False) # exec_btn
        )
    
    # Login success - Load current task
    if status["is_done_all"]:
        # 保存任务索引（已完成所有任务）
        TASK_INDEX_MAP[uid] = {
            "task_index": status['total_tasks'] - 1,  # 最后一个任务的索引
            "total_tasks": status['total_tasks']
        }
        task_idx = TASK_INDEX_MAP[uid]["task_index"]
        total = TASK_INDEX_MAP[uid]["total_tasks"]
        return (
            uid,
            gr.update(visible=False), # login_group
            gr.update(visible=True), # main_interface
            f"Welcome {username}. You have completed all tasks!", # login_message (hidden)
            gr.update(value=None, interactive=False), "All tasks completed! Thank you.", 
            gr.update(choices=[], value=None),
            "All tasks completed.", "No need for coordinates", 
            None, None,
            "No active task", f"Progress: {total}/{total}",
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False) # exec_btn
        )

    current_task = status["current_task"]
    env_id = current_task["env_id"]
    ep_num = current_task["episode_idx"]
    
    # Load the environment
    session = get_session(uid)
    print(f"Loading {env_id} Ep {ep_num} for {uid} (User: {username})")
    
    # 清空该session的coordinate_clicks和option_selects（新episode开始）
    if uid in COORDINATE_CLICKS:
        COORDINATE_CLICKS[uid] = []
    if uid in OPTION_SELECTS:
        OPTION_SELECTS[uid] = []
    
    img, load_msg = session.load_episode(env_id, int(ep_num))
    
    if img is None:
         # 即使加载失败，也保存任务索引
         TASK_INDEX_MAP[uid] = {
             "task_index": status['current_index'],
             "total_tasks": status['total_tasks']
         }
         task_idx = TASK_INDEX_MAP[uid]["task_index"]
         total = TASK_INDEX_MAP[uid]["total_tasks"]
         return (
            uid,
            gr.update(visible=False),
            gr.update(visible=True),
            f"Error loading task for {username}",
            gr.update(value=None, interactive=False), f"Error: {load_msg}",
            gr.update(choices=[], value=None),
            "", "No need for coordinates", 
            None, None,
            f"Task: {env_id} (Ep {ep_num})", f"Progress: {task_idx + 1}/{total}",
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False) # exec_btn
        )
        
    # Success loading
    goal_text = f"{session.language_goal}"
    options = session.available_options
    radio_choices = [(opt_label, opt_idx) for opt_label, opt_idx in options]
    
    # 保存任务索引到全局映射，供Progress直接读取
    TASK_INDEX_MAP[uid] = {
        "task_index": status['current_index'],
        "total_tasks": status['total_tasks']
    }
    
    demo_video_path = None
    if session.demonstration_frames:
        try:
            demo_video_path = save_video(session.demonstration_frames, "demo")
        except: pass

    # 从TASK_INDEX_MAP直接读取Progress
    task_idx = TASK_INDEX_MAP[uid]["task_index"]
    total = TASK_INDEX_MAP[uid]["total_tasks"]
    
    # 根据视图模式重新获取图片
    img = session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW)
    
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
        None, 
        demo_video_path,
        f"Current Task: {env_id} (Episode {ep_num})",
        f"Progress: {task_idx + 1}/{total}",
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=True) # exec_btn
    )

def load_next_task_wrapper(username, uid):
    """
    Wrapper to just reload the user's current status (which should be next task if updated).
    如果当前任务已有 actions，则创建新的 attempt。
    """
    if username:
        success, msg, status = user_manager.login(username)
        if success and not status["is_done_all"]:
            current_task = status["current_task"]
            env_id = current_task["env_id"]
            ep_num = current_task["episode_idx"]
            
            # 检查当前任务是否已有 actions，如果有则创建新的 attempt
            if has_existing_actions(username, env_id, ep_num):
                create_new_attempt(username, env_id, ep_num)
    
    return login_and_load_task(username, uid)

def save_video(frames, suffix=""):
    """
    视频保存函数 - 使用imageio生成视频
    
    优化点：
    1. 使用imageio.mimwrite，不依赖FFmpeg编码器
    2. 直接处理RGB帧，无需颜色空间转换
    3. 自动处理编码，简单可靠
    """
    if not frames or len(frames) == 0:
        return None
    
    try:
        import imageio
        
        # 准备帧：确保是uint8格式的numpy数组
        processed_frames = []
        for f in frames:
            if not isinstance(f, np.ndarray):
                f = np.array(f)
            # 确保是uint8格式
            if f.dtype != np.uint8:
                if np.max(f) <= 1.0:
                    f = (f * 255).astype(np.uint8)
                else:
                    f = f.clip(0, 255).astype(np.uint8)
            # 处理灰度图
            if len(f.shape) == 2:
                f = np.stack([f] * 3, axis=-1)
            # imageio期望RGB格式，frames已经是RGB
            processed_frames.append(f)
        
        fd, path = tempfile.mkstemp(suffix=f"_{suffix}.mp4")
        os.close(fd)
        
        # imageio.mimwrite会自动处理编码
        imageio.mimwrite(path, processed_frames, fps=10.0, quality=8, macro_block_size=None)
        
        return path
    except ImportError:
        print("Error: imageio module not found. Please install it: pip install imageio imageio-ffmpeg")
        return None
    except Exception as e:
        print(f"Error in save_video: {e}")
        traceback.print_exc()
        return None

def concatenate_frames_horizontally(frames1, frames2):
    """
    将两个帧序列左右拼接成一个帧序列
    
    Args:
        frames1: 左侧视频帧列表（base frames）
        frames2: 右侧视频帧列表（wrist frames）
    
    Returns:
        拼接后的帧列表
    """
    if not frames1 and not frames2:
        return []
    
    # 确定最大帧数
    max_frames = max(len(frames1), len(frames2))
    concatenated_frames = []
    
    for i in range(max_frames):
        # 获取当前帧，如果某个序列较短，重复最后一帧
        frame1 = frames1[min(i, len(frames1) - 1)] if frames1 else None
        frame2 = frames2[min(i, len(frames2) - 1)] if frames2 else None
        
        # 转换为numpy数组并确保格式正确
        if frame1 is not None:
            if not isinstance(frame1, np.ndarray):
                frame1 = np.array(frame1)
            if frame1.dtype != np.uint8:
                if np.max(frame1) <= 1.0:
                    frame1 = (frame1 * 255).astype(np.uint8)
                else:
                    frame1 = frame1.clip(0, 255).astype(np.uint8)
            if len(frame1.shape) == 2:
                frame1 = np.stack([frame1] * 3, axis=-1)
        else:
            # 如果frame1为空，创建一个黑色帧
            if frame2 is not None:
                h, w = frame2.shape[:2]
                frame1 = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                continue
        
        if frame2 is not None:
            if not isinstance(frame2, np.ndarray):
                frame2 = np.array(frame2)
            if frame2.dtype != np.uint8:
                if np.max(frame2) <= 1.0:
                    frame2 = (frame2 * 255).astype(np.uint8)
                else:
                    frame2 = frame2.clip(0, 255).astype(np.uint8)
            if len(frame2.shape) == 2:
                frame2 = np.stack([frame2] * 3, axis=-1)
        else:
            # 如果frame2为空，创建一个黑色帧
            h, w = frame1.shape[:2]
            frame2 = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 确保两个帧的高度相同，如果不同则调整
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        if h1 != h2:
            # 调整到相同高度（使用较小的那个）
            import cv2
            target_h = min(h1, h2)
            if h1 != target_h:
                frame1 = cv2.resize(frame1, (w1, target_h), interpolation=cv2.INTER_LINEAR)
            if h2 != target_h:
                frame2 = cv2.resize(frame2, (w2, target_h), interpolation=cv2.INTER_LINEAR)
        
        # 左右拼接
        concatenated_frame = np.concatenate([frame1, frame2], axis=1)
        concatenated_frames.append(concatenated_frame)
    
    return concatenated_frames

def draw_marker(img, x, y):
    """Draws a red circle and cross at (x, y)."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    img = img.copy()
    draw = ImageDraw.Draw(img)
    r = 5
    # Circle
    draw.ellipse((x-r, y-r, x+r, y+r), outline="red", width=2)
    # Cross
    draw.line((x-r, y, x+r, y), fill="red", width=2)
    draw.line((x, y-r, x, y+r), fill="red", width=2)
    return img

# --- Callback Functions ---

def load_env(uid, env_id, ep_num):
    if not uid:
        uid = create_session()
    
    session = get_session(uid)
    print(f"Loading {env_id} Ep {ep_num} for {uid}")
    
    # 清空该session的coordinate_clicks和option_selects（新episode开始）
    if uid in COORDINATE_CLICKS:
        COORDINATE_CLICKS[uid] = []
    if uid in OPTION_SELECTS:
        OPTION_SELECTS[uid] = []
    
    img, msg = session.load_episode(env_id, int(ep_num))
    
    if img is None:
        return (
            uid, 
            gr.update(value=None, interactive=False), 
            "Error loading episode", 
            gr.update(choices=[], value=None), 
            "", 
            "No need for coordinates", 
            None, None
        )

    # Goal
    goal_text = f"{session.language_goal}"
    
    # Options
    options = session.available_options
    radio_choices = [(opt_label, opt_idx) for opt_label, opt_idx in options]
    
    # Reset video placeholders
    
    # Save demonstration video if available
    demo_video_path = None
    if session.demonstration_frames:
        try:
            demo_video_path = save_video(session.demonstration_frames, "demo")
        except Exception as e:
            print(f"Error saving demo video: {e}")
            
    return (
        uid,
        gr.update(value=img, interactive=False), 
        f"Loaded {env_id} Ep {ep_num}. Status: Ready", 
        gr.update(choices=radio_choices, value=None), 
        goal_text, 
        "No need for coordinates", # Clear coords
        None, # Clear combined video
        demo_video_path # Demonstration video
    )

def on_map_click(uid, username, option_value, evt: gr.SelectData):
#     如果 用户选择选项 (on_option_select):
#     如果 选项需要目标 (available=True):
#         UI变更为: 图片可点击 (interactive=True), 提示 "Please click"
#     否则:
#         UI变更为: 图片不可点击 (interactive=False), 提示 "No need"

# 如果 用户点击图片 (on_map_click):
#     后端检查: 当前选项是否真的需要目标?
#     如果 不需要:
#         拦截点击，不记录坐标，不画点，返回 "No need"
#     如果 需要:
#         记录坐标，画红点，返回坐标值
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
    
    # 将坐标点击存储到临时列表，等待在action_execute时一起记录
    if uid not in COORDINATE_CLICKS:
        COORDINATE_CLICKS[uid] = []
    
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
            import traceback
            traceback.print_exc()
    
    COORDINATE_CLICKS[uid].append({
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
        return default_msg, gr.update(interactive=False)
    
    session = get_session(uid)
    if not session:
        return default_msg, gr.update(interactive=False)
    
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
    if uid not in OPTION_SELECTS:
        OPTION_SELECTS[uid] = []
    
    OPTION_SELECTS[uid].append({
        "option_idx": option_idx,
        "option_label": option_label,
        "timestamp": datetime.now().isoformat()
    })

    # Determine coords message
    if 0 <= option_idx < len(session.raw_solve_options):
        opt = session.raw_solve_options[option_idx]
        if opt.get("available"):
             return "please click the image", gr.update(interactive=True)
    
    return default_msg, gr.update(interactive=False)

def init_app(request: gr.Request):
    """
    Handle initial page load. 
    If 'user' query parameter is present, automatically login as that user.
    """
    params = request.query_params
    username = params.get('user')
    
    # Default outputs if no auto-login
    # uid, loading_group, login_group, main_interface, login_msg, img, log, options, goal, coords, combined, video, task, progress, login_btn, next_btn, exec_btn, username_state
    default_outputs = (
        None, 
        gr.update(visible=False), # loading_group (hide it)
        gr.update(visible=True), # login_group (show it)
        gr.update(visible=False), # main_interface
        "", 
        gr.update(value=None, interactive=False), None, 
        gr.update(choices=[], value=None), 
        "", "No need for coordinates", 
        None, None, 
        "", "", 
        gr.update(interactive=True), 
        gr.update(interactive=False), 
        gr.update(interactive=False),
        "" 
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
    session = get_session(uid)
    if not session:
        return None, "Session Error", None, gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=False)
    
    if option_idx is None:
        return session.get_pil_image(use_segmented=USE_SEGMENTED_VIEW), "Error: No action selected", None, gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

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
            option_selects_before_execute = []
            if uid in OPTION_SELECTS:
                option_selects_before_execute = OPTION_SELECTS[uid].copy()
                # 清空列表，为下次action做准备
                OPTION_SELECTS[uid] = []
            
            # 获取从上次action_execute到现在的所有coordinate_click
            # 这些点击已经包含了 image_array（在 on_map_click 中添加）
            coordinate_clicks_before_execute = []
            if uid in COORDINATE_CLICKS:
                coordinate_clicks_before_execute = COORDINATE_CLICKS[uid].copy()
                # 清空列表，为下次action做准备
                COORDINATE_CLICKS[uid] = []
            
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
    
    # 只取新增的帧来生成视频（从 execute action 之后新增的帧开始）
    new_base_frames = session.base_frames[pre_base_frame_count:]
    new_wrist_frames = session.wrist_frames[pre_wrist_frame_count:]
    
    # 将两个视频左右拼接成一个视频
    concatenated_frames = concatenate_frames_horizontally(new_base_frames, new_wrist_frames)
    
    # 生成拼接后的视频
    combined_video_path = None
    try:
        combined_video_path = save_video(concatenated_frames, "combined")
    except Exception as e:
        print(f"Error generating combined video: {e}")
        traceback.print_exc()
    
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
                "total_steps": len(session.history) if hasattr(session, 'history') and session.history else 0,
                "total_frames": len(session.base_frames) if hasattr(session, 'base_frames') else 0,
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
    
    return img, status, combined_video_path, task_update, progress_update, next_task_update, exec_btn_update

# --- JS for Video (no sync needed for single video) ---
SYNC_JS = ""  # No longer needed since we have a single combined video

CSS = """
#live_obs { border: 4px solid #3b82f6; border-radius: 8px; }
#control_panel { border: 1px solid #e5e7eb; padding: 15px; border-radius: 8px; background-color: #f9fafb; }
.compact-log textarea { max-height: 120px !important; font-family: monospace; font-size: 0.85em; }
.ref-zone { border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; margin-bottom: 10px; }
"""
if RESTRICT_VIDEO_PLAYBACK:
    CSS += """
    #demo_video {
        pointer-events: none;
    }
    """

# --- UI Construction ---

with gr.Blocks(title="Oracle Planner Interface", js=SYNC_JS, css=CSS) as demo:
    gr.Markdown("## HistoryBench Oracle Planner Interface (v2)")
    
    # State
    uid_state = gr.State(value=None)
    username_state = gr.State(value="")
    
    # --- Loading Section (Visible initially) ---
    with gr.Group(visible=True) as loading_group:
        gr.Markdown("### Logging in and setting up environment... Please wait.")

    # --- Login Section ---
    with gr.Group(visible=False) as login_group:
        gr.Markdown("### User Login")
        with gr.Row():
            # Get available usernames from user_manager
            available_users = list(user_manager.user_tasks.keys())
            username_input = gr.Dropdown(
                choices=available_users,
                label="Username",
                value=None
            )
            login_btn = gr.Button("Login", variant="primary")
        login_msg = gr.Markdown("")

    # --- Main Interface (Hidden initially) ---
    with gr.Group(visible=False) as main_interface:
        
        # --- Top Container: Reference Zone (35-40% Height) ---
        with gr.Row(elem_classes="ref-zone"):
            # Left: Text Info (30%)
            with gr.Column(scale=3):
                with gr.Group():
                     gr.Markdown("### 1. Task Info")
                     with gr.Row():
                         task_info_box = gr.Textbox(label="Current Task", interactive=False, show_label=False, scale=2)
                         progress_info_box = gr.Textbox(label="Progress", interactive=False, show_label=False, scale=1)
                
                with gr.Group():
                     gr.Markdown("### 2. Task Goal")
                     goal_box = gr.Textbox(label="Instruction", lines=3, interactive=False, show_label=False)
                
                with gr.Group():
                     gr.Markdown("### 3. System Log")
                     log_output = gr.Textbox(label="Log", lines=6, interactive=False, elem_classes="compact-log", show_label=False)

            # Right: Reference Views (70%)
            with gr.Column(scale=7):
                 gr.Markdown("### Reference Views")
                 with gr.Row():
                     # Demo Video
                     video_elem_id = "demo_video" if RESTRICT_VIDEO_PLAYBACK else None
                     video_autoplay = True if RESTRICT_VIDEO_PLAYBACK else False
                     
                     video_display = gr.Video(
                        label="Demonstration (示范)", 
                        interactive=False, 
                        height=300, 
                        elem_id=video_elem_id, 
                        autoplay=video_autoplay
                     )
                     
                     # Desk + Robot View (Combined)
                     combined_display = gr.Video(
                        label="Desk View (侧视) | Robot View (第一人称)", 
                        interactive=False, 
                        autoplay=True, 
                        height=300,
                        elem_id="combined_view"
                     )

        # --- Bottom Container: Operation Zone (60-65% Height) ---
        with gr.Row():
            # Left: Live Observation (Main)
            with gr.Column(scale=1):
                 gr.Markdown("### Live Observation (交互主视图)")
                 img_display = gr.Image(
                    label="Live Observation", 
                    interactive=False, 
                    type="pil", 
                    elem_id="live_obs",
                    show_label=False
                 )

            # Right: Control Panel
            with gr.Column(scale=2, elem_id="control_panel"):
                 gr.Markdown("### Control Panel")
                 
                 with gr.Row():
                     # Left sub-column: Actions
                     with gr.Column(scale=1):
                         gr.Markdown("**1. Action**")
                         options_radio = gr.Radio(choices=[], label="Action", type="value", show_label=False)
                     
                     # Right sub-column: Coords & Execute
                     with gr.Column(scale=1):
                         gr.Markdown("**2. Coords**")
                         coords_box = gr.Textbox(label="Coords", value="", interactive=False, show_label=False)
                         
                         gr.Markdown("**3. Execute**")
                         exec_btn = gr.Button("EXECUTE", variant="stop", size="lg")
                         
                         gr.Markdown("---")
                         next_task_btn = gr.Button("Next Task", variant="secondary", interactive=False)

    # --- Event Wiring ---

    # 1. Login
    login_btn.click(
        fn=login_and_load_task,
        inputs=[username_input, uid_state],
        outputs=[
            uid_state, 
            login_group, 
            main_interface, 
            login_msg, 
            img_display, 
            log_output, 
            options_radio, 
            goal_box, 
            coords_box, 
            combined_display, 
            video_display,
            task_info_box,
            progress_info_box,
            login_btn,
            next_task_btn,
            exec_btn
        ]
    ).then(
        fn=lambda u: u,
        inputs=[username_input],
        outputs=[username_state]
    )
    
    # 1.5 Next Task
    next_task_btn.click(
        fn=load_next_task_wrapper,
        inputs=[username_state, uid_state],
        outputs=[
            uid_state, 
            login_group, 
            main_interface, 
            login_msg, 
            img_display, 
            log_output, 
            options_radio, 
            goal_box, 
            coords_box, 
            combined_display, 
            video_display,
            task_info_box,
            progress_info_box,
            login_btn,
            next_task_btn,
            exec_btn
        ]
    )

    # 2. Image Click
    img_display.select(
        fn=on_map_click,
        inputs=[uid_state, username_state, options_radio],
        outputs=[img_display, coords_box]
    )
    
    # 2.5. Option Select
    options_radio.change(
        fn=on_option_select,
        inputs=[uid_state, username_state, options_radio],
        outputs=[coords_box, img_display]
    )

    # 3. Execute
    exec_btn.click(
        fn=execute_step,
        inputs=[uid_state, username_state, options_radio, coords_box],
        outputs=[img_display, log_output, combined_display, task_info_box, progress_info_box, next_task_btn, exec_btn]
    )
    
    # 5. Timer for Streaming (Keep-Alive / Real-time view)
    timer = gr.Timer(value=2.0)
    
    def _get_streaming_views(uid):
        # This function can be used to pull latest frames if the backend is running asynchronously
        # For this synchronous Oracle setup, we just return nothing or keep alive.
        # If we return values, we might overwrite the user's annotation on the image.
        pass 
        
    timer.tick(_get_streaming_views, inputs=[uid_state], outputs=[])

    # 6. Auto Login on Load
    demo.load(
        fn=init_app,
        inputs=[],
        outputs=[
            uid_state,
            loading_group,
            login_group, 
            main_interface, 
            login_msg, 
            img_display, 
            log_output, 
            options_radio, 
            goal_box, 
            coords_box, 
            combined_display, 
            video_display,
            task_info_box,
            progress_info_box,
            login_btn,
            next_task_btn,
            exec_btn,
            username_state
        ]
    )

if __name__ == "__main__":
    # Ensure session created for imports
    create_session()
    
    import socket
    def find_free_port(start_port=7860):
        for port in range(start_port, start_port + 20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port
        return 7860
        
    port = find_free_port()
    print(f"Starting server on port {port}")
    
    # Launch with prevent_thread_lock=True so we can print links
    _, local_url, share_url = demo.launch(server_name="0.0.0.0", server_port=port, share=True, prevent_thread_lock=True)
    
    print("\n" + "="*60)
    print("USER SPECIFIC LINKS:")
    print("="*60)
    
    # Use share_url if available, otherwise local_url
    target_url = share_url if share_url else local_url
    
    if target_url:
        # Sort users for better readability
        sorted_users = sorted(list(user_manager.user_tasks.keys()))
        for username in sorted_users:
            # quote username just in case, though usually simple strings
            import urllib.parse
            safe_username = urllib.parse.quote(username)
            print(f"User: {username:<15} -> {target_url}?user={safe_username}")
    else:
        print("Could not determine base URL.")
        
    print("="*60 + "\n")
    
    # Keep the server running
    demo.block_thread()
