import base64, io, json, math, time
import numpy as np
from PIL import Image

def parse(reply):
    blocks = reply.get("content", [])
    meta = json.loads(next(b["text"] for b in blocks if b.get("type") == "text"))
    imgs = {}
    for a in meta.get("attachments", []) or []:
        b = blocks[a["content_index"]]
        if b.get("type") != "image":
            continue
        raw = base64.b64decode(b["data"])
        im = Image.open(io.BytesIO(raw))
        arr = np.array(im)
        imgs[(a["camera"], a["kind"])] = arr
    return meta, imgs

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def backproject(meta, depth, cam, u, v):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    d = float(depth[int(round(v)), int(round(u))]) * 0.001
    if d <= 0:
        return None
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    pc = np.array([(u-cx)*d/fx, -(v-cy)*d/fy, -d])
    return quat_to_R(pose["quaternion_wxyz"]) @ pc + np.array(pose["position_m"])

def project(meta, cam, p):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"])
    pc = R.T @ (np.asarray(p) - np.array(pose["position_m"]))
    d = -pc[2]
    u = K[0,0]*pc[0]/d + K[0,2]
    v = -K[1,1]*pc[1]/d + K[1,2]
    return u, v, d
