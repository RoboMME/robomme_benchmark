import base64, io, json
import numpy as np
from PIL import Image

def parse(reply):
    blocks = reply.get("content", [])
    meta = None
    for b in blocks:
        if b.get("type") == "text":
            try:
                meta = json.loads(b["text"]); break
            except Exception:
                continue
    if meta is None:
        raise RuntimeError("no json block")
    imgs = {}
    for a in meta.get("attachments", []) or []:
        idx = a.get("content_index")
        if idx is None or idx >= len(blocks):
            continue
        b = blocks[idx]
        if b.get("type") != "image":
            continue
        im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
        if a.get("kind") == "depth":
            arr = np.array(im).astype(np.float32) * 0.001
        else:
            arr = np.array(im.convert("RGB"))
        imgs[(a.get("camera"), a.get("kind"))] = arr
    meta["_imgs"] = imgs
    return meta

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def backproject(meta, cam, us, vs, ds):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    us = np.asarray(us, float); vs = np.asarray(vs, float); ds = np.asarray(ds, float)
    pc = np.stack([(us-K[0, 2])*ds/K[0, 0], -(vs-K[1, 2])*ds/K[1, 1], -ds], -1)
    return pc @ R.T + t

def project(meta, cam, pts):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    pc = (np.asarray(pts, float) - t) @ R
    d = -pc[..., 2]
    u = pc[..., 0]*K[0, 0]/d + K[0, 2]
    v = -pc[..., 1]*K[1, 1]/d + K[1, 2]
    return u, v, d
