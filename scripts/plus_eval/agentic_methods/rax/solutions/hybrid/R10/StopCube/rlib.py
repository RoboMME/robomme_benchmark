import base64, io, json, time
import numpy as np
from PIL import Image

def decode(result):
    blocks = result.get("content", [])
    meta = json.loads(blocks[0]["text"])
    imgs = {}
    for a in meta.get("attachments", []):
        b = blocks[a["content_index"]]
        if b.get("type") != "image":
            continue
        im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
        if a["kind"] == "rgb":
            imgs[(a["camera"], "rgb")] = np.asarray(im.convert("RGB"))
        else:
            imgs[(a["camera"], "depth")] = np.asarray(im).astype(np.float32) * 0.001
    return meta, imgs

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def deproject(meta, cam, u, v, d):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    u = np.asarray(u, float); v = np.asarray(v, float); d = np.asarray(d, float)
    pc = np.stack([(u-cx)*d/fx, -(v-cy)*d/fy, -d], -1)
    return pc @ R.T + t

def project(meta, cam, P):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    pc = (np.asarray(P) - t) @ R
    d = -pc[..., 2]
    return np.stack([pc[..., 0]*fx/d + cx, -pc[..., 1]*fy/d + cy], -1), d

def world_map(meta, imgs, cam="front"):
    D = imgs[(cam, "depth")]
    vv, uu = np.mgrid[0:D.shape[0], 0:D.shape[1]]
    P = deproject(meta, cam, uu, vv, D)
    P[D <= 0] = np.nan
    return P
