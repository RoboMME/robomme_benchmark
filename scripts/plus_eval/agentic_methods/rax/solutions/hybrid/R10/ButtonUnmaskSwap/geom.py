import json
import numpy as np
from PIL import Image

ROOT = "/workspace/"


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_frame(frame_meta):
    """Return dict cam -> (rgb uint8 HxWx3, depth_m HxW, K, pose) from a manifest frame entry."""
    files = {(f["camera"], f["kind"]): f["path"] for f in frame_meta["files"]}
    out = {}
    for cam in ("front", "wrist"):
        if (cam, "rgb") not in files:
            continue
        rgb = np.asarray(Image.open(ROOT + files[(cam, "rgb")]).convert("RGB"))
        dep = None
        if (cam, "depth") in files:
            dep = np.asarray(Image.open(ROOT + files[(cam, "depth")])).astype(np.float32) * 0.001
        out[cam] = dict(rgb=rgb, depth=dep,
                        K=np.array(frame_meta["camera_intrinsics"][cam]["matrix"]),
                        pose=frame_meta["camera_poses"][cam])
    return out


def backproject(depth, K, pose):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    d = depth
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    cam = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], -1)
    R = quat_to_R(pose["quaternion_wxyz"])
    pts = cam.reshape(-1, 3) @ R.T + np.array(pose["position_m"])
    return pts.reshape(H, W, 3)
