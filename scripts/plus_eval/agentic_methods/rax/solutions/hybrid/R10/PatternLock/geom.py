import json
import numpy as np
from pathlib import Path
from PIL import Image

WS = Path("/workspace")


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def world_points(depth_m, K, pose):
    """depth HxW metres -> HxWx3 world points (OpenGL camera convention)."""
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    d = depth_m
    cam = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], -1)
    R = quat_to_R(pose["quaternion_wxyz"])
    return cam @ R.T + np.array(pose["position_m"])


def load_frame(frame, camera="front"):
    files = {(f["camera"], f["kind"]): f["path"] for f in frame["files"]}
    rgb = np.asarray(Image.open(WS / files[(camera, "rgb")]).convert("RGB")).astype(np.int32)
    dep = None
    if (camera, "depth") in files:
        dep = np.asarray(Image.open(WS / files[(camera, "depth")])).astype(np.float64) * 0.001
    return rgb, dep
