"""
配置常量模块
"""
# --- Configuration ---
VIDEO_PLAYBACK_FPS = 35.0  # Shared output frame rate for demonstration and execute video playback
USE_SEGMENTED_VIEW = False  # Set to True to use segmented view, False to use original image

# 主界面两列宽度比例 (Point Selection : Right Panel)
POINT_SELECTION_SCALE = 1
CONTROL_PANEL_SCALE = 2

# 右侧顶部并排比例 (Action Selection : System Log)
RIGHT_TOP_ACTION_SCALE = 2
RIGHT_TOP_LOG_SCALE = 1

# 全局界面字号（不作用于页面主标题）
UI_GLOBAL_FONT_SIZE = "24px"

# Session / queue 配置
SESSION_TIMEOUT = 90  # 30秒无用户主动操作后，交由 gr.State TTL 自动回收 session
SESSION_CONCURRENCY_ID = "session_actions"
SESSION_CONCURRENCY_LIMIT = 10


# 兜底执行次数配置
EXECUTE_LIMIT_OFFSET = 4  # 兜底执行次数 = non_demonstration_task_length + EXECUTE_LIMIT_OFFSET

TASK_NAME_LIST = [
    "BinFill",
    "StopCube",
    "PickXtimes",
    "SwingXtimes",
    
    "ButtonUnmask",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmaskSwap",
    
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
]


# 应该显示demonstration videos的环境ID列表
DEMO_VIDEO_ENV_IDS = [
    "VideoPlaceOrder",
    "VideoUnmaskSwap",
    "VideoUnmask",
    "VideoRepick",
    "VideoPlaceButton",
    "InsertPeg",
    "MoveCube",
    "PatternLock",
    "RouteStick"
]

UI_TEXT = {
    "log": {
        "action_selection_prompt": "Please select the action.\nActions with 🎯 need to select a point on the image as input",
        "point_selection_prompt": "Current action needs location input, please click on the image to select key pixel",
        "point_selected_message": "Select: {label} | point <{x}, {y}>",
        "execute_action_prompt": "Executing: {label}",
        "execute_action_prompt_with_coords": "Executing: {label} | point <{coords_text}>",
        "demo_video_prompt": 'Press "Watch Video Input 🎬" to watch a video\nNote: you can only watch the video once',
        "session_error": "Session expired. Please refresh the page and try again.",
        "reference_action_error": "Ground Truth Action Error: {error}",
        "reference_action_message": "Ground Truth Action: {option_label}",
        "reference_action_message_with_coords": "Ground Truth Action: {option_label} | point <{coords_text}>",
        "reference_action_status": "Ground Truth Action: {message}",
        "execute_missing_action": "Error: No action selected",
        "episode_success_banner": "********************************\n****   episode success      ****\n********************************\n  ---please press change episode----   ",
        "episode_failed_banner": "********************************\n****   episode failed       ****\n********************************\n  ---please press change episode----   ",
    },
    "coords": {
        "not_needed": "No need for coordinates",
        "select_point": "Please click the point selection image",
        "select_point_before_execute": "Please click the point selection image before execute!",
    },
    "actions": {
        "point_required_suffix": " 🎯",
    },
    "progress": {
        "episode_loading": "The episode is loading...",
        "queue_wait": "Lots of people are playing! Please wait...",
        "entry_rejected": "Too many users are trying the demo right now. Please try again later.",
    },
    "errors": {
        "load_missing_task": "Error loading task: missing current_task",
        "load_invalid_task": "Error loading task: invalid task payload",
        "load_episode_error": "Error: {load_msg}",
        "next_task_failed": "Failed to load next task",
        "restart_missing_task": "Failed to restart episode: missing current task",
        "restart_invalid_task": "Failed to restart episode: invalid task payload",
        "switch_env_failed": "Failed to switch environment to '{selected_env}'",
        "init_failed": "Initialization error: {error}",
        "reference_action_resolve_failed": "Failed to resolve ground truth action.",
    },
}

UI_ACTION_TEXT_OVERRIDES = {
    "PatternLock": {
        "move forward": "move forward↓",
        "move backward": "move backward↑",
        "move left": "move left→",
        "move right": "move right←",
        "move forward-left": "move forward-left↘︎",
        "move forward-right": "move forward-right↙︎",
        "move backward-left": "move backward-left↗︎",
        "move backward-right": "move backward-right↖︎",
    },
}

ROUTESTICK_OVERLAY_ACTION_TEXTS = [
    "move to the nearest left target by circling around the stick clockwise",
    "move to the nearest left target by circling around the stick counterclockwise",
    "move to the nearest right target by circling around the stick clockwise",
    "move to the nearest right target by circling around the stick counterclockwise",
]

LIVE_OBS_BASE_CLASS = "live-obs-resizable"
LIVE_OBS_POINT_WAIT_CLASS = "live-obs-point-waiting"


def get_live_obs_elem_classes(waiting_for_point=False):
    classes = [LIVE_OBS_BASE_CLASS]
    if waiting_for_point:
        classes.append(LIVE_OBS_POINT_WAIT_CLASS)
    return classes


def get_ui_action_text(env_id, action_text):
    """
    Return display-only action text overrides for a specific env/action pair.
    Falls back to the original action text when no override is configured.
    """
    if not isinstance(action_text, str):
        return action_text
    if not isinstance(env_id, str) or not env_id:
        return action_text
    env_overrides = UI_ACTION_TEXT_OVERRIDES.get(env_id, {})
    return env_overrides.get(action_text, action_text)

def should_show_demo_video(env_id):
    """
    判断指定的环境ID是否应该显示demonstration video
    只有DEMO_VIDEO_ENV_IDS列表中的环境才显示demonstration videos
    """
    return env_id in DEMO_VIDEO_ENV_IDS
