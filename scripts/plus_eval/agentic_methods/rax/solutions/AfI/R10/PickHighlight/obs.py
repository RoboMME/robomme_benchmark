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

def backproject(meta, cam, depth, us=None, vs=None):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    H, W = depth.shape
    if us is None:
        vs, us = np.mgrid[0:H, 0:W]
    d = depth[vs, us] if np.ndim(us) else depth[int(vs), int(us)]
    x = (us - K[0, 2]) * d / K[0, 0]
    y = -(vs - K[1, 2]) * d / K[1, 1]
    z = -d
    P = np.stack([x, y, z], -1)
    return P @ R.T + t
