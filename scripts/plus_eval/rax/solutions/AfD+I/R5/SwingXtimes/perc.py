import base64, io, json
import numpy as np
from PIL import Image

def parse(reply):
    blocks = reply["content"]
    meta = json.loads(blocks[0]["text"])
    imgs = {}
    for a in meta.get("attachments", []):
        b = blocks[a["content_index"]]
        im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
        arr = np.asarray(im)
        imgs[(a["camera"], a["kind"])] = arr
    return meta, imgs

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def backproject(meta, imgs, cam="front"):
    """Return HxWx3 world points (nan where invalid)."""
    d = imgs[(cam, "depth")].astype(np.float64) * meta["depth"][cam].get("meters_per_unit", 0.001)
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    pc = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], -1)
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    pw = pc @ R.T + t
    pw[d <= 0] = np.nan
    return pw
