"""Observation decoding, geometry and color segmentation for MoveCube."""
import base64, io, json, math
from pathlib import Path
import numpy as np
from PIL import Image

WS = Path('/workspace')

def decode(result):
    """MCP result -> (meta dict, {(camera,kind): np.array})"""
    content = result['content']
    meta = json.loads(content[0]['text'])
    imgs = {}
    for att in meta.get('attachments', []):
        blk = content[att['content_index']]
        data = base64.b64decode(blk['data'])
        im = Image.open(io.BytesIO(data))
        if att['kind'] == 'depth':
            arr = np.array(im).astype(np.float32) * meta['depth'][att['camera']].get('meters_per_unit', 0.001)
        else:
            arr = np.array(im.convert('RGB'))
        imgs[(att['camera'], att['kind'])] = arr
    return meta, imgs

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def axangle_to_R(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-9: return np.eye(3)
    k = v / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K

def R_to_axangle(R):
    c = (np.trace(R) - 1) / 2; c = max(-1.0, min(1.0, c)); th = math.acos(c)
    if th < 1e-6: return np.zeros(3)
    if abs(th - math.pi) < 1e-3:
        # near pi
        M = (R + np.eye(3)) / 2; k = np.sqrt(np.maximum(np.diag(M), 0))
        i = int(np.argmax(k)); k = M[:, i] / math.sqrt(max(M[i, i], 1e-9))
        return k * th
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return k * th

class Cam:
    def __init__(self, meta, cam):
        K = np.array(meta['camera_intrinsics'][cam]['matrix'], float)
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        p = meta['camera_poses'][cam]
        self.R = quat_to_R(p['quaternion_wxyz']); self.t = np.array(p['position_m'], float)

    def backproject(self, u, v, d):
        u = np.asarray(u, float); v = np.asarray(v, float); d = np.asarray(d, float)
        pc = np.stack([(u - self.cx) * d / self.fx, -(v - self.cy) * d / self.fy, -d], -1)
        return pc @ self.R.T + self.t

    def project(self, P):
        P = np.atleast_2d(P)
        pc = (P - self.t) @ self.R  # world->cam
        d = -pc[:, 2]
        u = pc[:, 0] / d * self.fx + self.cx
        v = -pc[:, 1] / d * self.fy + self.cy
        return np.stack([u, v, d], -1)

def world_points(meta, imgs, cam, mask):
    D = imgs[(cam, 'depth')]
    vs, us = np.nonzero(mask & (D > 0))
    if len(us) == 0: return np.zeros((0, 3))
    return Cam(meta, cam).backproject(us, vs, D[vs, us])

# ---------- colour masks ----------
def _ch(a):
    a = a.astype(int); return a[..., 0], a[..., 1], a[..., 2]

def m_red(a):
    R, G, B = _ch(a); return (R > 120) & (G < 60) & (B < 60) & (R - G > 80)
def m_purple(a):
    R, G, B = _ch(a); return (B > 150) & (R > 120) & (G < 150) & (B - G > 50) & (R - G > 30)
def m_yellow(a):
    R, G, B = _ch(a); return (R > 140) & (G > 120) & (B < 100) & (abs(R - G) < 50) & (G - B > 70)
def m_teal(a):
    R, G, B = _ch(a); return (G - R > 30) & (B - R > 25) & (G > 70) & (abs(G - B) < 35) & (B < 170)
def m_blue(a):  # ramp
    R, G, B = _ch(a); return (B > 140) & (B - R > 50) & (G > 120) & (B - G > 10)
def m_salmon(a):  # pink tool parts (plate, hook tip, ball tip)
    R, G, B = _ch(a); return (R > 200) & (G > 130) & (G < 200) & (B > 110) & (B < 180) & (R - G > 35)
def m_gray(a):
    a = a.astype(int); mx = a.max(-1); mn = a.min(-1); return (mx - mn < 30) & (mx > 140)
def m_slate(a):  # hammer head (dark blue-gray)
    R, G, B = _ch(a); mx = np.maximum(np.maximum(R, G), B); mn = np.minimum(np.minimum(R, G), B)
    return (mx - mn < 45) & (mx < 140) & (mx > 50) & (B >= R + 5)
def m_tan(a):  # wooden hammer handle
    R, G, B = _ch(a); return (R > 170) & (G > 140) & (B > 90) & (B < 150) & (R - B > 50) & (R - G < 50) & (R - G > 15)

def components(mask, min_area=3):
    """Simple 4-connected components; returns list of (area, ys, xs)."""
    from collections import deque
    H, W = mask.shape; seen = np.zeros_like(mask, bool); out = []
    ys0, xs0 = np.nonzero(mask)
    for y0, x0 in zip(ys0, xs0):
        if seen[y0, x0]: continue
        q = deque([(y0, x0)]); seen[y0, x0] = True; ys = []; xs = []
        while q:
            y, x = q.popleft(); ys.append(y); xs.append(x)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True; q.append((yy, xx))
        if len(ys) >= min_area: out.append((len(ys), np.array(ys), np.array(xs)))
    out.sort(key=lambda c: -c[0])
    return out

def blob_world(meta, imgs, maskfn, cam='front', min_area=3, which=0):
    a = imgs[(cam, 'rgb')]; comps = components(maskfn(a), min_area)
    if len(comps) <= which: return None
    area, ys, xs = comps[which]
    m = np.zeros(a.shape[:2], bool); m[ys, xs] = True
    P = world_points(meta, imgs, cam, m)
    if len(P) == 0: return None
    return dict(area=area, uv=(xs.mean(), ys.mean()), P=np.median(P, 0), pts=P, top=P[:, 2].max())
