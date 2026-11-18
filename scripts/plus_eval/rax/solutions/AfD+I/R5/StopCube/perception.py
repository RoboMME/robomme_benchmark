import base64, io, json
import numpy as np
from PIL import Image

def decode(reply):
    content = reply["content"]
    meta = json.loads(content[0]["text"])
    imgs = {}
    for att in meta.get("attachments", []):
        blk = content[att["content_index"]]
        data = base64.b64decode(blk["data"])
        im = Image.open(io.BytesIO(data))
        arr = np.array(im)
        if att["kind"] == "depth":
            arr = arr.astype(np.float32) * 0.001
        imgs[(att["camera"], att["kind"])] = arr
    return meta, imgs

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def backproject(meta, depth, cam="front"):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d = depth
    pc = np.stack([(u - K[0, 2]) * d / K[0, 0], -(v - K[1, 2]) * d / K[1, 1], -d], -1)
    pw = pc @ R.T + t
    pw[d <= 0] = np.nan
    return pw

def hsv(a):
    a = a.astype(np.float32) / 255; mx = a.max(-1); mn = a.min(-1); d = mx - mn + 1e-6
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    s = (mx - mn) / (mx + 1e-6)
    return h, s, mx
