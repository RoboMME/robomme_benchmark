import json, base64, io, os
import numpy as np
from PIL import Image

WS = '/workspace'

def packet(reply):
    for b in reply.get('content', []):
        if b.get('type') == 'text':
            return json.loads(b['text'])
    raise RuntimeError('no text block')

def images(reply, pk=None):
    pk = pk or packet(reply)
    out = {}
    blocks = reply['content']
    for a in pk.get('attachments', []):
        b = blocks[a['content_index']]
        im = Image.open(io.BytesIO(base64.b64decode(b['data'])))
        arr = np.asarray(im)
        if a['kind'] == 'depth':
            arr = arr.astype(np.float64) * 0.001
        else:
            arr = np.asarray(im.convert('RGB'))
        out[(a['camera'], a['kind'])] = arr
    return out

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

class Cam:
    def __init__(self, intr, pose):
        K = np.array(intr['matrix'], float)
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        self.R = quat_to_R(pose['quaternion_wxyz'])
        self.t = np.array(pose['position_m'], float)
    def backproject(self, u, v, d):
        u = np.asarray(u, float); v = np.asarray(v, float); d = np.asarray(d, float)
        pc = np.stack([(u - self.cx) * d / self.fx, -(v - self.cy) * d / self.fy, -d], -1)
        return pc @ self.R.T + self.t
    def project(self, P):
        pc = (np.asarray(P, float) - self.t) @ self.R
        d = -pc[..., 2]
        u = pc[..., 0] * self.fx / d + self.cx
        v = -pc[..., 1] * self.fy / d + self.cy
        return u, v, d
    def world_map(self, depth):
        H, W = depth.shape
        vv, uu = np.mgrid[0:H, 0:W]
        P = self.backproject(uu, vv, depth)
        P[depth <= 0] = np.nan
        return P

def cams(pk):
    return {c: Cam(pk['camera_intrinsics'][c], pk['camera_poses'][c]) for c in pk['camera_intrinsics']}
