import json, base64, io
import numpy as np
from PIL import Image

def parse(reply):
    blocks = reply["content"]
    meta = json.loads(blocks[0]["text"])
    imgs = {}
    for a in meta.get("attachments", []):
        b = blocks[a["content_index"]]
        data = base64.b64decode(b["data"])
        im = Image.open(io.BytesIO(data))
        if a["kind"] == "rgb":
            imgs[(a["camera"], "rgb")] = np.array(im.convert("RGB"))
        else:
            imgs[(a["camera"], "depth")] = np.array(im).astype(np.float32) * 0.001
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
    v, u = np.mgrid[0:H, 0:W]
    d = depth
    xc = (u - K[0, 2]) * d / K[0, 0]  # pixel center convention tested below
    yc = -(v - K[1, 2]) * d / K[1, 1]
    zc = -d
    P = np.stack([xc, yc, zc], -1) @ R.T + t
    return P

def project(meta, pts, cam="front"):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    pc = (np.atleast_2d(pts) - t) @ R
    d = -pc[:, 2]
    u = pc[:, 0] / d * K[0, 0] + K[0, 2]
    v = -pc[:, 1] / d * K[1, 1] + K[1, 2]
    return np.stack([u, v], -1)
