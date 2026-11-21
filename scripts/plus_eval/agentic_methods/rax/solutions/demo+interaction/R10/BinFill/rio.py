"""Robot I/O helpers: decode MCP replies, observe, step, back-project."""
import json, io, base64, math
import numpy as np
from PIL import Image

def _blocks(reply):
    if isinstance(reply, dict) and "content" in reply:
        return reply["content"]
    return reply

def decode(reply):
    blocks = _blocks(reply)
    meta = None
    for b in blocks:
        t = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
        if t == "text":
            txt = b["text"] if isinstance(b, dict) else b.text
            try:
                meta = json.loads(txt); break
            except Exception:
                continue
    imgs = {}
    if meta is not None:
        for a in meta.get("attachments", []) or []:
            idx = a.get("content_index")
            if idx is None or idx >= len(blocks):
                continue
            b = blocks[idx]
            data = b["data"] if isinstance(b, dict) else b.data
            im = Image.open(io.BytesIO(base64.b64decode(data)))
            if a["kind"] == "rgb":
                imgs[(a["camera"], "rgb")] = np.array(im.convert("RGB"))
            else:
                imgs[(a["camera"], "depth")] = np.array(im).astype(np.float32) * 0.001
    return meta, imgs

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def backproject(meta, cam, u, v, d):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    p = np.array([(u-cx)*d/fx, -(v-cy)*d/fy, -d])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"])
    return R @ p + np.array(pose["position_m"])

def pointcloud(meta, imgs, cam):
    """World xyz per pixel (H,W,3); nan where depth invalid."""
    d = imgs[(cam, "depth")]
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = d.shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    P = np.stack([(uu-cx)*d/fx, -(vv-cy)*d/fy, -d], -1)
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"])
    Wp = P @ R.T + np.array(pose["position_m"])
    Wp[d <= 0] = np.nan
    return Wp
