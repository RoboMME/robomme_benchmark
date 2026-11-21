import json, base64, io
import numpy as np
from PIL import Image

def parse(reply):
    blocks = reply.get('content', [])
    meta = json.loads(blocks[0]['text'])
    imgs = {}
    for att in meta.get('attachments', []) or []:
        b = blocks[att['content_index']]
        data = base64.b64decode(b['data'])
        im = Image.open(io.BytesIO(data))
        arr = np.array(im)
        imgs[(att['camera'], att['kind'])] = arr
    return meta, imgs

def depth_m(arr, mpu=0.001):
    d = arr.astype(np.float64) * mpu
    d[arr == 0] = np.nan
    return d

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def backproject(depth, K, pose, us=None, vs=None):
    """depth HxW meters -> world points HxWx3 (OpenGL convention)."""
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    H, W = depth.shape
    if us is None:
        vs, us = np.mgrid[0:H, 0:W]
    d = depth[vs, us] if depth.ndim == 2 and us.shape != depth.shape else depth
    cam = np.stack([(us - cx) * d / fx, -(vs - cy) * d / fy, -d], -1)
    R = quat_to_R(pose['quaternion_wxyz'])
    return cam @ R.T + np.array(pose['position_m'])
