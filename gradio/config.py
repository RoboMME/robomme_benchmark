"""
配置常量模块
"""
# --- Configuration ---
RESTRICT_VIDEO_PLAYBACK = True  # Set to False to enable controls
USE_SEGMENTED_VIEW = False  # Set to True to use segmented view, False to use original image
REFERENCE_VIEW_HEIGHT = "40vh"  # Height of the reference view image

ENV_IDS = [
    "VideoPlaceOrder", "PickXtimes", "StopCube", "SwingXtimes", 
    "BinFill", "VideoUnmaskSwap", "VideoUnmask", "ButtonUnmaskSwap", 
    "ButtonUnmask", "VideoRepick", "VideoPlaceButton", "InsertPeg", 
    "MoveCube", "PatternLock", "RouteStick"
]
