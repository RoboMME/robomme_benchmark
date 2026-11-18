import json, math, time
from pathlib import Path
import numpy as np
from PIL import Image

WS = Path('/workspace')

def jcall(ctx, tool, **kw):
    r = ctx.call(tool, **kw)
    for b in r.get('content', []):
        if b.get('type') == 'text':
            try:
                return json.loads(b['text'])
            except Exception:
                pass
    return {}

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def load_frame(fr):
    files = {(f['camera'], f['kind']): f['path'] for f in fr['files']}
    out = {}
    for (cam, kind), p in files.items():
        a = np.array(Image.open(WS / p))
        if kind == 'depth':
            a = a.astype(np.float32) * 0.001
        else:
            a = a[..., :3]
        out[(cam, kind)] = a
    return out

def backproject(depth, K, pose, us, vs):
    """us, vs arrays of pixel coords -> world points Nx3"""
    fx, cx, fy, cy = K[0][0], K[0][2], K[1][1], K[1][2]
    us = np.asarray(us, float); vs = np.asarray(vs, float)
    d = depth[vs.astype(int), us.astype(int)]
    cam = np.stack([(us-cx)*d/fx, -(vs-cy)*d/fy, -d], 1)
    R = quat_to_R(pose['quaternion_wxyz'])
    return cam @ R.T + np.array(pose['position_m']), d

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx
        self.log = log
        self.frames = []   # list of (step_id, frame dict)
        self.last = None
        self.ended = False
        self.complete = False
        self.step_id = 0
        self.limit_remaining = None

    def observe(self):
        j = jcall(self.ctx, 'robomme_observe')
        self.last = j
        self.step_id = j.get('step_id', self.step_id)
        self.state = j['state']
        seq = j.get('frame_sequence')
        if seq:
            self._read_seq(seq)
        return j

    def _read_seq(self, seq):
        try:
            m = json.loads((WS / seq['manifest_path']).read_text())
        except Exception as e:
            self.log('seq read fail', e); return
        for fr in m.get('frames', []):
            self.frames.append(fr)
        tr = m.get('transition') or {}
        if tr.get('episode_ended'):
            self.ended = True
        if tr.get('task_complete'):
            self.complete = True
        for s in tr.get('steps', []):
            if s.get('episode_ended'):
                self.ended = True
            if s.get('task_complete'):
                self.complete = True

    def step(self, rows):
        if self.ended:
            return None
        rows = [[float(max(-1, min(1, v))) for v in r] for r in rows]
        j = jcall(self.ctx, 'robomme_step', actions=rows)
        self.last = j
        self.step_id = j.get('step_id', self.step_id)
        if 'state' in j:
            self.state = j['state']
        if j.get('episode_ended'):
            self.ended = True
        if j.get('task_complete'):
            self.complete = True
        lim = j.get('episode_limits') or {}
        self.limit_remaining = lim.get('remaining_episode_steps')
        seq = j.get('frame_sequence')
        if seq:
            self._read_seq(seq)
        return j

    def pos(self):
        return np.array(self.state[:3])

    def grip_open(self):
        return self.state[6] + self.state[7]

    def move_to(self, target, grip, max_steps=60, tol=0.004, chunk=5, gain=1.0, yaw_target=None, maxv=1.0):
        """closed-loop move in chunks. grip: -1 open, +1 close"""
        target = np.array(target, float)
        n = 0
        while n < max_steps and not self.ended:
            err = target - self.pos()
            yerr = 0.0
            if yaw_target is not None:
                yerr = wrap(yaw_target - self.yaw())
            if np.linalg.norm(err) < tol and abs(yerr) < 0.03:
                break
            k = min(chunk, max_steps - n)
            # per-step command: fraction of error, measured lag means ~proportional
            a = err / 0.05 * gain
            s = np.max(np.abs(a))
            if s > maxv:
                a = a / s * maxv
            r = 0.0
            if yaw_target is not None:
                r = max(-1, min(1, yerr / 0.5 * gain))
            rows = [[a[0], a[1], a[2], 0, 0, r, grip]] * k
            self.step(rows)
            n += k
        return n

    def hold(self, k, grip):
        return self.step([[0, 0, 0, 0, 0, 0, grip]] * k)

    def yaw(self):
        R = rotvec_to_R(self.state[3:6])
        # gripper x-axis direction in world xy
        return math.atan2(R[1, 0], R[0, 0])

def rotvec_to_R(rv):
    rv = np.array(rv, float)
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
