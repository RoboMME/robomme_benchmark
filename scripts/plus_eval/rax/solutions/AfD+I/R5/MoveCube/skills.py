import math, json, time
import numpy as np
from perception import *

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

class Scene:
    """front-camera perception of the current observation"""
    def __init__(self, rb):
        self.rb = rb

    def grab(self):
        m = self.rb.observe()
        ims = self.rb.images()
        fr = {'front_rgb': ims[('front', 'rgb')], 'front_depth': ims[('front', 'depth')],
              'K': m['camera_intrinsics'], 'poses': m['camera_poses']}
        self.F = fr
        self.img = fr['front_rgb']
        self.P = world_map(fr['front_depth'], fr['K']['front'], fr['poses']['front'])
        self.m = color_masks(self.img)
        return self

    def _frame_from_meta(self, m):
        # current observation files: use the latest saved frame of the frame sequence manifest
        seq = m.get('frame_sequence') or {}
        man = json.loads((WS / seq['manifest_path']).read_text())
        fr = man['frames'][-1]
        return load_frame(fr)

    def pts(self, cat, min_px=6, k=0):
        bl = blobs(self.m[cat], min_px)
        if len(bl) <= k: return None, None
        return blob_world(bl[k], self.P), bl[k]

    def cube(self):
        bl = blobs(self.m['red'], 15)
        if not bl: return None
        pts = blob_world(bl[0], self.P)
        if len(pts) < 5: return None
        top = pts[pts[:, 2] > pts[:, 2].max() - 0.008]
        return np.array([top[:, 0].mean(), top[:, 1].mean(), 0.02])

    def cube_top(self):
        bl = blobs(self.m['red'], 15)
        if not bl: return 0.0
        pts = blob_world(bl[0], self.P)
        if len(pts) < 5: return 0.0
        return float(np.percentile(pts[:, 2], 95))

    def target(self):
        t = self._target()
        if t is not None and getattr(self, 'tgt_cache', None) is None:
            self.tgt_cache = t
        return getattr(self, 'tgt_cache', None) if getattr(self, 'tgt_cache', None) is not None else t

    def _target(self):
        mk = self.m['target']
        # target rings: merge all target-colored pixels near the largest cluster
        bl = blobs(mk, 2)
        if not bl: return None
        pts = np.concatenate([blob_world(b, self.P) for b in bl])
        pts = pts[pts[:, 2] < 0.03]
        if len(pts) == 0: return None
        c = np.median(pts[:, :2], 0)
        near = pts[np.linalg.norm(pts[:, :2] - c, axis=1) < 0.05]
        return np.array([near[:, 0].mean(), near[:, 1].mean(), 0.01])

TIP_R = {'ball': 0.016, 'scoop': 0.03, 'hook': 0.02, 'hammer': 0.015, 'yellow': 0.02}

def find_teal_tools(sc):
    teal = [b for b in blobs(sc.m['teal'], 15)]
    handles = []
    for b in teal:
        pts = blob_world(b, sc.P)
        if len(pts) < 10: continue
        pts = pts[pts[:, 2] < 0.08]
        if len(pts) < 10: continue
        xy = pts[:, :2]
        ext = np.ptp(xy, 0).max()
        if ext > 0.15: continue
        handles.append({'c': pts.mean(0), 'zmax': float(np.percentile(pts[:, 2], 95)), 'n': len(pts), 'pts': pts})
    tips = []
    for b in blobs(sc.m['pink'], 5):
        pts = blob_world(b, sc.P)
        pts = pts[(pts[:, 2] < 0.08) & (pts[:, 2] > -0.01)] if len(pts) else pts
        if len(pts) < 4: continue
        cls, ext = pink_class(pts)
        tips.append({'c': pts.mean(0), 'cls': cls, 'ext': ext, 'n': len(pts), 'pts': pts})
    for h in handles:
        xy = h['pts'][:, :2]; mu = xy.mean(0)
        w, V = np.linalg.eigh(np.cov((xy - mu).T)); h['ax'] = V[:, 1]
    def cost(h, t):
        d = t['c'][:2] - h['c'][:2]; L = np.linalg.norm(d)
        if L < 0.08 or L > 0.45: return 1e9
        perp = abs(d[0] * h['ax'][1] - d[1] * h['ax'][0])
        r = perp / L
        if r > 0.33 and perp > 0.05: return 1e9
        return r + 0.1 * L
    owner = {}
    for ti, t in enumerate(tips):
        cs = [cost(h, t) for h in handles]
        if cs and min(cs) < 1e8: owner[ti] = int(np.argmin(cs))
    tools = []
    for hi, h in enumerate(handles):
        cand = [tips[ti] for ti, o in owner.items() if o == hi]
        if not cand: continue
        big = max(cand, key=lambda t: t['n'])
        pts = np.concatenate([t['pts'] for t in cand if np.linalg.norm(t['c'][:2] - big['c'][:2]) < 0.07])
        cls, ext = pink_class(pts)
        tip = {'c': pts.mean(0), 'cls': cls, 'ext': ext, 'n': len(pts), 'pts': pts}
        tools.append({'handle': h, 'tip': tip, 'cls': cls, 'dist': float(np.linalg.norm(h['c'][:2] - tip['c'][:2]))})
    return tools, handles, tips

def find_hammer(sc):
    bl = blobs(sc.m['beige'], 10)
    dk = blobs(sc.m['dark'], 15)
    best = None
    for b in bl[:4]:
        pts = blob_world(b, sc.P); pts = pts[pts[:, 2] < 0.10] if len(pts) else pts
        if len(pts) < 10: continue
        if best is None or len(pts) > len(best): best = pts
    if best is None: return None
    pts = best; xy = pts[:, :2]; mu = xy.mean(0)
    w, V = np.linalg.eigh(np.cov((xy - mu).T)); ax = V[:, 1]
    s = (xy - mu) @ ax
    e1, e2 = mu + ax * s.min(), mu + ax * s.max()
    head = None
    for b in dk[:4]:
        hp = blob_world(b, sc.P)
        if len(hp) < 10: continue
        hc = hp.mean(0)
        if min(np.linalg.norm(hc[:2] - e1), np.linalg.norm(hc[:2] - e2)) < 0.12:
            head = hc; break
    if head is not None and np.linalg.norm(head[:2] - e1) < np.linalg.norm(head[:2] - e2):
        near, far = e1, e2
    else:
        near, far = e2, e1
    L = np.linalg.norm(far - near)
    g = near + (far - near) * 0.35   # grasp 35% from the head end
    loc = pts[np.linalg.norm(pts[:, :2] - g, axis=1) < 0.025]
    z = float(np.percentile(loc[:, 2], 90)) if len(loc) >= 5 else float(np.percentile(pts[:, 2], 90))
    return {'grasp': np.array([g[0], g[1], z]), 'tip': np.array([far[0], far[1], z]), 'len': L, 'head': head}

def ang(v):
    return math.atan2(v[1], v[0])

def cur_yaw(rb):
    from scipy.spatial.transform import Rotation as Rr
    Rm = Rr.from_rotvec(rb.state[3:6]) * Rr.from_rotvec([math.pi, 0, 0]).inv()
    return float(Rm.as_euler('zyx')[0])

def grasp_at(rb, p, yaw, z_grasp, log=print):
    """open gripper, go above p with yaw, descend, close. returns True if something is held"""
    rb.move_to([p[0], p[1], 0.13], -1, max_steps=80, speed=0.8, yaw=yaw)
    rb.move_to([p[0], p[1], 0.13], -1, max_steps=30, yaw=yaw, tol=0.005)
    rb.move_to([p[0], p[1], z_grasp + 0.04], -1, max_steps=40, speed=0.5, yaw=yaw)
    rb.move_to([p[0], p[1], z_grasp], -1, max_steps=40, speed=0.25, tol=0.003, yaw=yaw)
    rb.grip(1, 14)
    w = float(rb.state[6] + rb.state[7])
    log('grasp width', round(w, 4), 'tcp', rb.state[:3].round(3))
    return w > 0.004

def measure_pink_tip(sc, expect, radius=0.10):
    tips = []
    for b in blobs(sc.m['pink'], 3):
        pts = blob_world(b, sc.P)
        pts = pts[~np.isnan(pts).any(-1)] if len(pts) else pts
        if len(pts) < 3: continue
        if np.linalg.norm(pts.mean(0)[:2] - expect[:2]) < radius: tips.append(pts)
    if not tips: return None
    pts = np.concatenate(tips)
    return pts.mean(0)

def measure_beige_tip(sc, tcp, expect, radius=0.10):
    best = []
    for b in blobs(sc.m['beige'], 3):
        pts = blob_world(b, sc.P)
        if len(pts) < 3: continue
        best.append(pts)
    if not best: return None
    pts = np.concatenate(best)
    pts = pts[np.linalg.norm(pts[:, :2] - tcp[:2], axis=1) < 0.30]
    if len(pts) < 5: return None
    d = np.linalg.norm(pts[:, :2] - tcp[:2], axis=1)
    far = pts[d > np.percentile(d, 90)]
    c = far.mean(0)
    if np.linalg.norm(c[:2] - expect[:2]) > radius: return None
    return c
