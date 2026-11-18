import json, base64, io, time
import numpy as np
from PIL import Image

def parse(reply):
    blocks = reply.get("content", [])
    meta = json.loads(next(b["text"] for b in blocks if b.get("type") == "text"))
    imgs = {}
    for a in meta.get("attachments", []):
        b = blocks[a["content_index"]]
        if b.get("type") != "image":
            continue
        im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
        arr = np.array(im)
        if a["kind"] == "depth":
            arr = arr.astype(np.float32) * 0.001
        else:
            arr = np.array(im.convert("RGB"))
        imgs[(a["camera"], a["kind"])] = arr
    return meta, imgs

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def deproject(meta, cam, u, v, d):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    p = np.array([(u-cx)*d/fx, -(v-cy)*d/fy, -d])
    pose = meta["camera_poses"][cam]
    return quat_to_R(pose["quaternion_wxyz"]) @ p + np.array(pose["position_m"])

def cloud(meta, depth, cam):
    """world xyz for every pixel (H,W,3)"""
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W]
    d = depth
    P = np.stack([(uu-cx)*d/fx, -(vv-cy)*d/fy, -d], -1)
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"])
    return P @ R.T + np.array(pose["position_m"])
