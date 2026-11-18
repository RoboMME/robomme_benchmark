import json, base64, io, time
import numpy as np
from PIL import Image

def parse(reply):
    """reply: list of MCP content blocks -> (meta dict, {(camera,kind): ndarray})"""
    blocks = reply['content'] if isinstance(reply, dict) and 'content' in reply else reply
    meta = None
    for b in blocks:
        if b.get('type') == 'text':
            try:
                meta = json.loads(b['text']); break
            except Exception:
                pass
    imgs = {}
    if meta:
        for att in meta.get('attachments', []):
            b = blocks[att['content_index']]
            if b.get('type') == 'image':
                im = Image.open(io.BytesIO(base64.b64decode(b['data'])))
                imgs[(att['camera'], att['kind'])] = np.array(im)
    return meta, imgs

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log
        self.meta = None; self.imgs = None; self.ended = False; self.success = False

    def observe(self):
        self.meta, self.imgs = parse(self.ctx.call('robomme_observe'))
        self._check(self.meta)
        return self.meta

    def _check(self, meta):
        if not meta: return
        if meta.get('episode_ended'):
            self.ended = True
        for s in meta.get('steps', []) or []:
            if isinstance(s, dict):
                if s.get('task_complete'): self.success = True
                if s.get('episode_ended'): self.ended = True
        if meta.get('task_complete'): self.success = True
        tr = meta.get('transition') or {}
        if tr.get('task_complete'): self.success = True
        if tr.get('episode_ended'): self.ended = True
        for s in tr.get('steps', []) or []:
            if isinstance(s, dict):
                if s.get('task_complete'): self.success = True
                if s.get('episode_ended'): self.ended = True

    @property
    def state(self):
        return np.array(self.meta['state'], float)

    def remaining(self):
        return self.meta.get('episode_limits', {}).get('remaining_episode_steps', 999)

    def step(self, rows):
        if self.ended: return self.meta
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        rem = self.remaining()
        if rem is not None and rem <= 0:
            self.ended = True; return self.meta
        rows = rows[:max(1, rem)]
        self.meta, self.imgs = parse(self.ctx.call('robomme_step', actions=rows))
        self._check(self.meta)
        return self.meta

    def pix2world(self, pts, camera='front', r=1):
        sel = [{'camera': camera, 'u': float(u), 'v': float(v)} for u, v in pts]
        rep = self.ctx.call('robomme_pixel_to_position', observation_step_id=int(self.meta['step_id']),
                            selections=sel, depth_window_radius=r)
        blocks = rep['content'] if isinstance(rep, dict) else rep
        for b in blocks:
            if b.get('type') == 'text':
                return json.loads(b['text'])
        return None

    def move_to(self, target, grip, tol=0.004, max_steps=60, rot_hold=True):
        """Closed-loop move of EE to world target (x,y,z) with gripper command grip."""
        target = np.asarray(target, float)
        n = 0
        while n < max_steps and not self.ended:
            p = self.state[:3]
            d = target - p
            if np.linalg.norm(d) < tol:
                break
            k = min(8, max_steps - n)
            # plan k rows toward target, limited 0.05 m/step*0.8
            step = d / 0.05
            mag = np.abs(step).max()
            nrows = int(np.clip(np.ceil(mag / 0.7), 1, k))
            row = list(np.clip(step / nrows, -0.7, 0.7)) + [0, 0, 0, grip]
            self.step([row] * nrows)
            n += nrows
        return self.state[:3]

    def grip(self, g, n=10):
        self.step([[0, 0, 0, 0, 0, 0, g]] * n)

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def backproject(meta, depth_png, camera, us, vs):
    K = np.array(meta['camera_intrinsics'][camera]['matrix'], float)
    pose = meta['camera_poses'][camera]
    R = quat_to_R(pose['quaternion_wxyz']); t = np.array(pose['position_m'], float)
    us = np.asarray(us); vs = np.asarray(vs)
    d = depth_png[vs.astype(int), us.astype(int)].astype(float) * 0.001
    ok = d > 0
    cam = np.stack([(us - K[0, 2]) * d / K[0, 0], -(vs - K[1, 2]) * d / K[1, 1], -d], axis=1)
    w = cam @ R.T + t
    return w[ok]
