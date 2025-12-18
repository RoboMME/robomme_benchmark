import gradio as gr
import uuid
import numpy as np
import tempfile
import os
import traceback
from PIL import Image, ImageDraw
from oracle_logic import OracleSession
from concurrent.futures import ThreadPoolExecutor
from user_manager import user_manager
from logger import log_session

# --- Global Session Storage ---
GLOBAL_SESSIONS = {}

ENV_IDS = [
    "VideoPlaceOrder", "PickXtimes", "StopCube", "SwingXtimes", 
    "BinFill", "VideoUnmaskSwap", "VideoUnmask", "ButtonUnmaskSwap", 
    "ButtonUnmask", "VideoRepick", "VideoPlaceButton", "InsertPeg", 
    "MoveCube", "PatternLock", "RouteStick"
]

# --- Configuration ---
RESTRICT_VIDEO_PLAYBACK = False  # Set to False to enable controls

# --- Helper Functions ---

def get_session(uid):
    return GLOBAL_SESSIONS.get(uid)

def create_session():
    uid = str(uuid.uuid4())
    session = OracleSession()
    GLOBAL_SESSIONS[uid] = session
    return uid

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
            None, None, # img, status
            gr.update(choices=[], value=None), # options
            "", "", # goal, coords
            None, None, # combined_video, demo_video
            "", "", # task_info, progress_info
            gr.update(interactive=True) # login_btn
        )
    
    # Login success - Load current task
    if status["is_done_all"]:
        return (
            uid,
            gr.update(visible=False), # login_group
            gr.update(visible=True), # main_interface
            f"Welcome {username}. You have completed all tasks!", # login_message (hidden)
            None, "All tasks completed! Thank you.", 
            gr.update(choices=[], value=None),
            "All tasks completed.", "", 
            None, None,
            "No active task", f"Progress: {status['completed_count']}/{status['total_tasks']} (100%)",
            gr.update(interactive=True)
        )

    current_task = status["current_task"]
    env_id = current_task["env_id"]
    ep_num = current_task["episode_idx"]
    
    # Load the environment
    session = get_session(uid)
    print(f"Loading {env_id} Ep {ep_num} for {uid} (User: {username})")
    
    img, load_msg = session.load_episode(env_id, int(ep_num))
    
    if img is None:
         return (
            uid,
            gr.update(visible=False),
            gr.update(visible=True),
            f"Error loading task for {username}",
            None, f"Error: {load_msg}",
            gr.update(choices=[], value=None),
            "", "", 
            None, None,
            f"Task: {env_id} (Ep {ep_num})", f"Progress: {status['current_index'] + 1}/{status['total_tasks']}",
            gr.update(interactive=True)
        )
        
    # Success loading
    goal_text = f"{session.language_goal}"
    options = session.available_options
    radio_choices = [(opt_label, opt_idx) for opt_label, opt_idx in options]
    
    demo_video_path = None
    if session.demonstration_frames:
        try:
            demo_video_path = save_video(session.demonstration_frames, "demo")
        except: pass

    return (
        uid,
        gr.update(visible=False), # Login hidden
        gr.update(visible=True),  # Main visible
        f"Logged in as {username}", 
        img, 
        f"Ready. Task {status['current_index'] + 1}/{status['total_tasks']}: {env_id}",
        gr.update(choices=radio_choices, value=None),
        goal_text, 
        "", 
        None, 
        demo_video_path,
        f"Current Task: {env_id} (Episode {ep_num})",
        f"Progress: {status['current_index'] + 1}/{status['total_tasks']} Completed: {status['completed_count']}",
        gr.update(interactive=True)
    )

def load_next_task_wrapper(username, uid):
    """Wrapper to just reload the user's current status (which should be next task if updated)."""
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
    
    img, msg = session.load_episode(env_id, int(ep_num))
    
    if img is None:
        return (
            uid, 
            None, 
            "Error loading episode", 
            gr.update(choices=[], value=None), 
            "", 
            "", 
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
        img, 
        f"Loaded {env_id} Ep {ep_num}. Status: Ready", 
        gr.update(choices=radio_choices, value=None), 
        goal_text, 
        "", # Clear coords
        None, # Clear combined video
        demo_video_path # Demonstration video
    )

def on_map_click(uid, evt: gr.SelectData):
    session = get_session(uid)
    if not session:
        return None, "Session Error"
    
    x, y = evt.index[0], evt.index[1]
    
    # Get clean image from session
    base_img = session.get_pil_image()
    
    # Draw marker
    marked_img = draw_marker(base_img, x, y)
    
    coords_str = f"{x}, {y}"
    return marked_img, coords_str

def execute_step(uid, username, option_idx, coords_str):
    session = get_session(uid)
    if not session:
        return None, "Session Error", None, gr.update(), gr.update()
    
    if option_idx is None:
        return session.get_pil_image(), "Error: No action selected", None, gr.update(), gr.update()

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
            
    # Execute
    print(f"Executing step: Opt {option_idx}, Coords {click_coords}")
    img, status, done = session.execute_action(option_idx, click_coords)
    
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
    
    progress_update = gr.update()
    task_update = gr.update()
    
    if done:
        status += " [EPISODE COMPLETE]"
        
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
                "status": "completed"
            })
        except Exception as e:
            print(f"Error logging session: {e}")
            traceback.print_exc()
        
        # Update user progress
        if username:
            user_status = user_manager.complete_current_task(username)
            if user_status:
                if user_status["is_done_all"]:
                    task_update = "All tasks completed!"
                    progress_update = f"Progress: {user_status['completed_count']}/{user_status['total_tasks']} (100%)"
                else:
                    next_env = user_status["current_task"]["env_id"]
                    next_ep = user_status["current_task"]["episode_idx"]
                    task_update = f"Task Completed! Next: {next_env} (Ep {next_ep})"
                    progress_update = f"Progress: {user_status['current_index'] + 1}/{user_status['total_tasks']} Completed: {user_status['completed_count']}"
        
    return img, status, combined_video_path, task_update, progress_update

# --- JS for Video (no sync needed for single video) ---
SYNC_JS = ""  # No longer needed since we have a single combined video

CSS = ""
if RESTRICT_VIDEO_PLAYBACK:
    CSS = """
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
    
    # --- Login Section ---
    with gr.Group(visible=True) as login_group:
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
        with gr.Row():
            # --- Left Column: Controls (Scale 1) ---
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 1. Task Info")
                    task_info_box = gr.Textbox(label="Current Task", interactive=False)
                    progress_info_box = gr.Textbox(label="Progress", interactive=False)
                    
                    with gr.Row():
                        next_task_btn = gr.Button("Load Next / Current Task", variant="secondary")
                    
                    # Hidden fields for compatibility if needed, or just remove usage
                    # keeping them defined but hidden/unused for now to minimize structure change impact if referenced elsewhere
                    # env_dd = gr.Dropdown(choices=ENV_IDS, value="PickXtimes", label="Environment ID", visible=False)
                    # ep_num = gr.Number(value=0, label="Episode Index", precision=0, visible=False)
                    # load_btn = gr.Button("Load Environment", variant="primary", visible=False)
                
                with gr.Group():
                    gr.Markdown("### 2. Task Goal")
                    goal_box = gr.Textbox(label="Instruction", lines=2, interactive=False)
                
                with gr.Group():
                    gr.Markdown("### 3. Action & Interaction")
                    options_radio = gr.Radio(choices=[], label="Available Actions", type="value")
                    coords_box = gr.Textbox(label="Selected Coordinates (x, y)", value="")
                    
                    exec_btn = gr.Button("Execute Action", variant="stop")
                
                gr.Markdown("### 4. Logs")
                log_output = gr.Textbox(label="System Log", lines=5, interactive=False)

            # --- Right Column: Visuals (Scale 2) ---
            with gr.Column(scale=2):
                # Top: Image & Demo Video
                with gr.Row():
                    img_display = gr.Image(label="Live Observation (Click to Select)", interactive=True, type="pil")
                    # Placeholder for demo video if needed, or maybe just hidden if no data
                    video_elem_id = "demo_video" if RESTRICT_VIDEO_PLAYBACK else None
                    video_autoplay = True if RESTRICT_VIDEO_PLAYBACK else False
                    
                    video_display = gr.Video(
                        label="Demonstration", 
                        interactive=False, 
                        height=300, 
                        elem_id=video_elem_id, 
                        autoplay=video_autoplay
                    )
                
                # Bottom: Execution Feedback (Combined View)
                combined_display = gr.Video(label="Desk View (Left) | Robot View (Right)", elem_id="combined_view", interactive=False, autoplay=True)

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
            login_btn
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
            login_btn
        ]
    )

    # 2. Image Click
    img_display.select(
        fn=on_map_click,
        inputs=[uid_state],
        outputs=[img_display, coords_box]
    )

    # 3. Execute
    exec_btn.click(
        fn=execute_step,
        inputs=[uid_state, username_state, options_radio, coords_box],
        outputs=[img_display, log_output, combined_display, task_info_box, progress_info_box]
    )
    
    # 5. Timer for Streaming (Keep-Alive / Real-time view)
    timer = gr.Timer(value=2.0)
    
    def _get_streaming_views(uid):
        # This function can be used to pull latest frames if the backend is running asynchronously
        # For this synchronous Oracle setup, we just return nothing or keep alive.
        # If we return values, we might overwrite the user's annotation on the image.
        pass 
        
    timer.tick(_get_streaming_views, inputs=[uid_state], outputs=[])

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
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
