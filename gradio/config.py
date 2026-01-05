"""
配置常量模块
"""
# --- Configuration ---
RESTRICT_VIDEO_PLAYBACK = True  # Restrict controls; we will force autoplay via JS
VIDEO_PLAYBACK_FPS = 30.0  # Frame rate for demonstration video playback
USE_SEGMENTED_VIEW = False  # Set to True to use segmented view, False to use original image
REFERENCE_VIEW_HEIGHT = "40vh"  # Height of the reference view image

# Operation Zone 三列宽度比例 (Live Observation : Action : Control)
LIVE_OBSERVATION_SCALE = 3  # Live Observation 列的宽度比例
ACTION_SCALE = 3  # Action 列的宽度比例
CONTROL_SCALE = 3  # Control 列的宽度比例

# Session超时配置
SESSION_TIMEOUT = 300  # Session超时时间（秒），如果30秒内没有execute_step操作，将自动回收session


# 【环境ID列表】
# 所有可用的测试环境ID，共16个环境
# 注意：PickHighlight 已添加到列表中，确保所有环境ID都可用
ENV_IDS = [
    "VideoPlaceOrder", "PickXtimes", "StopCube", "SwingXtimes", 
    "BinFill", "VideoUnmaskSwap", "VideoUnmask", "ButtonUnmaskSwap", 
    "ButtonUnmask", "VideoRepick", "VideoPlaceButton", "InsertPeg", 
    "MoveCube", "PatternLock", "RouteStick", "PickHighlight"  # 【新增】PickHighlight 环境ID
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

def should_show_demo_video(env_id):
    """
    判断指定的环境ID是否应该显示demonstration video
    只有DEMO_VIDEO_ENV_IDS列表中的环境才显示demonstration videos
    """
    return env_id in DEMO_VIDEO_ENV_IDS
