from .RouteStick import RouteStick_SYSTEM_PROMPT, RouteStick_SYSTEM_PROMPT_GROUNDED, RouteStick_SYSTEM_PROMPT_ORACLE_PLANNER
from .BinFill import BinFill_SYSTEM_PROMPT, BinFill_SYSTEM_PROMPT_GROUNDED, BinFill_SYSTEM_PROMPT_ORACLE_PLANNER
from .SwingXtimes import SwingXtimes_SYSTEM_PROMPT, SwingXtimes_SYSTEM_PROMPT_GROUNDED, SwingXtimes_SYSTEM_PROMPT_ORACLE_PLANNER
from .PickXtimes import PickXtimes_SYSTEM_PROMPT, PickXtimes_SYSTEM_PROMPT_GROUNDED, PickXtimes_SYSTEM_PROMPT_ORACLE_PLANNER
from .StopCube import StopCube_SYSTEM_PROMPT, StopCube_SYSTEM_PROMPT_GROUNDED, StopCube_SYSTEM_PROMPT_ORACLE_PLANNER
from .InsertPeg import InsertPeg_SYSTEM_PROMPT, InsertPeg_SYSTEM_PROMPT_GROUNDED, InsertPeg_SYSTEM_PROMPT_ORACLE_PLANNER
from .PatternLock import PatternLock_SYSTEM_PROMPT_ORACLE_PLANNER
from .ButtonUnmask import ButtonUnmask_SYSTEM_PROMPT, ButtonUnmask_SYSTEM_PROMPT_GROUNDED, ButtonUnmask_SYSTEM_PROMPT_ORACLE_PLANNER
from .VideoUnmask import VideoUnmask_SYSTEM_PROMPT, VideoUnmask_SYSTEM_PROMPT_GROUNDED, VideoUnmask_SYSTEM_PROMPT_ORACLE_PLANNER
from .VideoUnmaskSwap import VideoUnmaskSwap_SYSTEM_PROMPT, VideoUnmaskSwap_SYSTEM_PROMPT_GROUNDED, VideoUnmaskSwap_SYSTEM_PROMPT_ORACLE_PLANNER
from .ButtonUnmaskSwap import ButtonUnmaskSwap_SYSTEM_PROMPT, ButtonUnmaskSwap_SYSTEM_PROMPT_GROUNDED, ButtonUnmaskSwap_SYSTEM_PROMPT_ORACLE_PLANNER
from .VideoPlaceButton import VideoPlaceButton_SYSTEM_PROMPT, VideoPlaceButton_SYSTEM_PROMPT_GROUNDED, VideoPlaceButton_SYSTEM_PROMPT_ORACLE_PLANNER
from .VideoPlaceOrder import VideoPlaceOrder_SYSTEM_PROMPT, VideoPlaceOrder_SYSTEM_PROMPT_GROUNDED, VideoPlaceOrder_SYSTEM_PROMPT_ORACLE_PLANNER
from .VideoRepick import VideoRepick_SYSTEM_PROMPT, VideoRepick_SYSTEM_PROMPT_GROUNDED, VideoRepick_SYSTEM_PROMPT_ORACLE_PLANNER
from .MoveCube import MoveCube_SYSTEM_PROMPT, MoveCube_SYSTEM_PROMPT_GROUNDED, MoveCube_SYSTEM_PROMPT_ORACLE_PLANNER

from .PickHighlight import PickHighlight_SYSTEM_PROMPT, PICK_HIGHLIGHT_IMAGE_TEXT_QUERY, PICK_HIGHLIGHT_VIDEO_TEXT_QUERY, PickHighlight_SYSTEM_PROMPT_GROUNDED, PickHighlight_SYSTEM_PROMPT_ORACLE_PLANNER
from .base import IMAGE_TEXT_QUERY, VIDEO_TEXT_QUERY, DEMO_TEXT_QUERY, VIDEO_TEXT_QUERY_multi_image, DEMO_TEXT_QUERY_multi_image

prompt_dict_simple = {
    "RouteStick": RouteStick_SYSTEM_PROMPT,
    "BinFill": BinFill_SYSTEM_PROMPT,
    "PickHighlight": PickHighlight_SYSTEM_PROMPT,
    "SwingXtimes": SwingXtimes_SYSTEM_PROMPT,
    "PickXtimes": PickXtimes_SYSTEM_PROMPT,
    "StopCube": StopCube_SYSTEM_PROMPT,
    "InsertPeg": InsertPeg_SYSTEM_PROMPT,
    "ButtonUnmask": ButtonUnmask_SYSTEM_PROMPT,
    "VideoUnmask": VideoUnmask_SYSTEM_PROMPT,
    "VideoUnmaskSwap": VideoUnmaskSwap_SYSTEM_PROMPT,
    "ButtonUnmaskSwap": ButtonUnmaskSwap_SYSTEM_PROMPT,
    "VideoPlaceButton": VideoPlaceButton_SYSTEM_PROMPT,
    "VideoPlaceOrder": VideoPlaceOrder_SYSTEM_PROMPT,
    "VideoRepick": VideoRepick_SYSTEM_PROMPT,
    "MoveCube": MoveCube_SYSTEM_PROMPT
}

prompt_dict_grounded = {
    "BinFill": BinFill_SYSTEM_PROMPT_GROUNDED,
    "ButtonUnmask": ButtonUnmask_SYSTEM_PROMPT_GROUNDED,
    "ButtonUnmaskSwap": ButtonUnmaskSwap_SYSTEM_PROMPT_GROUNDED,
    "InsertPeg": InsertPeg_SYSTEM_PROMPT_GROUNDED,
    "MoveCube": MoveCube_SYSTEM_PROMPT_GROUNDED,
    "PickHighlight": PickHighlight_SYSTEM_PROMPT_GROUNDED,
    "PickXtimes": PickXtimes_SYSTEM_PROMPT_GROUNDED,
    "RouteStick": RouteStick_SYSTEM_PROMPT_GROUNDED,
    "StopCube": StopCube_SYSTEM_PROMPT_GROUNDED,
    "SwingXtimes": SwingXtimes_SYSTEM_PROMPT_GROUNDED,
    "VideoPlaceButton": VideoPlaceButton_SYSTEM_PROMPT_GROUNDED,
    "VideoPlaceOrder": VideoPlaceOrder_SYSTEM_PROMPT_GROUNDED,
    "VideoRepick": VideoRepick_SYSTEM_PROMPT_GROUNDED,
    "VideoUnmask": VideoUnmask_SYSTEM_PROMPT_GROUNDED,
    "VideoUnmaskSwap": VideoUnmaskSwap_SYSTEM_PROMPT_GROUNDED,
}

prompt_dict_oracle_planner = {
    "BinFill": BinFill_SYSTEM_PROMPT_ORACLE_PLANNER,
    "ButtonUnmask": ButtonUnmask_SYSTEM_PROMPT_ORACLE_PLANNER,
    "ButtonUnmaskSwap": ButtonUnmaskSwap_SYSTEM_PROMPT_ORACLE_PLANNER,
    "InsertPeg": InsertPeg_SYSTEM_PROMPT_ORACLE_PLANNER,
    "PatternLock": PatternLock_SYSTEM_PROMPT_ORACLE_PLANNER,
    "MoveCube": MoveCube_SYSTEM_PROMPT_ORACLE_PLANNER,
    "PickHighlight": PickHighlight_SYSTEM_PROMPT_ORACLE_PLANNER,
    "PickXtimes": PickXtimes_SYSTEM_PROMPT_ORACLE_PLANNER,
    "RouteStick": RouteStick_SYSTEM_PROMPT_ORACLE_PLANNER,
    "StopCube": StopCube_SYSTEM_PROMPT_ORACLE_PLANNER,
    "SwingXtimes": SwingXtimes_SYSTEM_PROMPT_ORACLE_PLANNER,
    "VideoPlaceButton": VideoPlaceButton_SYSTEM_PROMPT_ORACLE_PLANNER,
    "VideoPlaceOrder": VideoPlaceOrder_SYSTEM_PROMPT_ORACLE_PLANNER,
    "VideoRepick": VideoRepick_SYSTEM_PROMPT_ORACLE_PLANNER,
    "VideoUnmask": VideoUnmask_SYSTEM_PROMPT_ORACLE_PLANNER,
    "VideoUnmaskSwap": VideoUnmaskSwap_SYSTEM_PROMPT_ORACLE_PLANNER,
}