"""Robot helpers: observation decoding, depth back-projection, closed-loop motion."""
import base64, io, json, time
import numpy as np
from PIL import Image

LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def decode(blocks):
    """MCP result (dict with content, or block list) -> (meta dict, {(camera,kind): ndarray})."""
    if isinstance(blocks, dict):
        blocks = blocks.get('content', [])
    meta = None
    for b in blocks:
        if b.get('type') == 'text':
            try:
                meta = json.loads(b['text'])
                break
            except Exception:
                continue
    imgs = {}
    if meta is None:
        return {}, imgs
    for att in meta.get('attachments', []) or []:
        idx = att.get('content_index')
        if idx is None or idx >= len(blocks):
            continue
        b = blocks[idx]
        data = b.get('data')
        if not data:
            continue
        im = Image.open(io.BytesIO(base64.b64decode(data)))
        arr = np.asarray(im)
        if att.get('kind') == 'rgb':
            arr = np.asarray(im.convert('RGB'))
        imgs[(att.get('camera'), att.get('kind'))] = arr
    return meta, imgs


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(meta, depth_mm, cam, us, vs):
    """pixels -> world points (N,3); invalid depth -> nan."""
    K = np.array(meta['camera_intrinsics'][cam]['matrix'], float)
    pose = meta['camera_poses'][cam]
    R = quat_to_R(pose['quaternion_wxyz'])
    t = np.array(pose['position_m'], float)
    us = np.asarray(us, float); vs = np.asarray(vs, float)
    ui = np.clip(np.round(us).astype(int), 0, depth_mm.shape[1] - 1)
    vi = np.clip(np.round(vs).astype(int), 0, depth_mm.shape[0] - 1)
    d = depth_mm[vi, ui].astype(float) * 0.001
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.stack([(us - cx) * d / fx, -(vs - cy) * d / fy, -d], -1)
    pw = pc @ R.T + t
    pw[d <= 0] = np.nan
    return pw


def project(meta, cam, P):
    """world points (N,3) -> pixels (N,2)."""
    K = np.array(meta['camera_intrinsics'][cam]['matrix'], float)
    pose = meta['camera_poses'][cam]
    R = quat_to_R(pose['quaternion_wxyz'])
    t = np.array(pose['position_m'], float)
    pc = (np.asarray(P, float) - t) @ R  # world->cam
    d = -pc[:, 2]
    u = K[0, 2] + K[0, 0] * pc[:, 0] / d
    v = K[1, 2] - K[1, 1] * pc[:, 1] / d
    return np.stack([u, v], -1)


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = None
        self.imgs = {}
        self.ended = False
        self.complete = False
        self.steps_used = 0

    def observe(self):
        blocks = self.ctx.call('robomme_observe')
        self.meta, self.imgs = decode(blocks)
        return self.meta

    @property
    def state(self):
        return np.array(self.meta['state'], float)

    def remaining(self):
        lim = self.meta.get('episode_limits') or {}
        return lim.get('remaining_episode_steps', 10 ** 6)

    def step(self, rows):
        if self.ended:
            return None
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        blocks = self.ctx.call('robomme_step', actions=rows)
        meta, imgs = decode(blocks)
        if meta:
            if 'state' in meta:
                self.meta, self.imgs = meta, imgs
            else:
                self.meta.update({k: v for k, v in meta.items() if k in ('episode_limits', 'episode_ended')})
            if meta.get('episode_ended'):
                self.ended = True
            for s in meta.get('steps', []) or []:
                if s.get('task_complete'):
                    self.complete = True
                if s.get('episode_ended'):
                    self.ended = True
            if meta.get('task_complete'):
                self.complete = True
            tr = meta.get('transition') or {}
            if tr.get('task_complete'):
                self.complete = True
            if tr.get('episode_ended'):
                self.ended = True
            for s in tr.get('steps', []) or []:
                if s.get('task_complete'):
                    self.complete = True
                if s.get('episode_ended'):
                    self.ended = True
        return meta

    def move_to(self, target, grip, tol=0.004, max_iter=12, max_rows=40, gain=1.0):
        """Closed-loop translation to a world xyz, keeping orientation, gripper fixed."""
        target = np.asarray(target, float)
        for it in range(max_iter):
            if self.ended:
                return False
            p = self.state[:3]
            err = target - p
            dist = np.linalg.norm(err)
            if dist < tol:
                return True
            # per-step cap 0.03 m (0.6 action units), fewer steps when close
            n = int(np.clip(np.ceil(dist / 0.025), 1, max_rows))
            delta = err / n * gain
            row = list(np.clip(delta / 0.05, -1, 1)) + [0.0, 0.0, 0.0, grip]
            self.step([row] * n)
        return np.linalg.norm(target - self.state[:3]) < 3 * tol

    def gripper(self, grip, n=12):
        self.step([[0, 0, 0, 0, 0, 0, grip]] * n)
