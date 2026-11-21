import base64, io, json, math, time
import numpy as np
from PIL import Image

LOG = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)

def parse(reply):
    """MCP result -> (meta dict, {(camera, kind): np.array})"""
    content = reply['content'] if isinstance(reply, dict) and 'content' in reply else reply
    meta = None
    for b in content:
        if b.get('type') == 'text':
            try:
                meta = json.loads(b['text']); break
            except Exception:
                pass
    imgs = {}
    if meta:
        for att in meta.get('attachments', []):
            ci = att.get('content_index')
            if ci is None or ci >= len(content):
                continue
            b = content[ci]
            data = b.get('data')
            if not data:
                continue
            im = Image.open(io.BytesIO(base64.b64decode(data)))
            arr = np.array(im)
            if att.get('kind') == 'rgb':
                arr = np.array(im.convert('RGB'))
            imgs[(att.get('camera'), att.get('kind'))] = arr
    return meta or {}, imgs

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def backproject(meta, depth, cam, us, vs):
    K = np.array(meta['camera_intrinsics'][cam]['matrix'])
    pose = meta['camera_poses'][cam]
    R = quat_to_R(pose['quaternion_wxyz']); t = np.array(pose['position_m'])
    mpu = meta.get('depth', {}).get(cam, {}).get('meters_per_unit', 0.001)
    us = np.asarray(us); vs = np.asarray(vs)
    d = depth[vs, us].astype(float) * mpu
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.stack([(us - cx) * d / fx, -(vs - cy) * d / fy, -d], -1)
    pw = pc @ R.T + t
    return pw, d > 0

class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = None; self.imgs = None
        self.ended = False; self.success = False
        self.steps_used = 0

    def observe(self):
        r = self.ctx.call('robomme_observe')
        self.meta, self.imgs = parse(r)
        return self.meta

    @property
    def state(self):
        return np.array(self.meta['state'], dtype=float)

    def remaining(self):
        lim = self.meta.get('episode_limits', {})
        return lim.get('remaining_episode_steps', 999)

    def step(self, rows):
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        r = self.ctx.call('robomme_step', actions=rows)
        meta, imgs = parse(r)
        if meta.get('state') is None or not imgs:
            self.observe()
        else:
            self.meta, self.imgs = meta, imgs
        m = meta
        if m.get('episode_ended'):
            self.ended = True
        tr = m.get('transition') or {}
        if tr.get('episode_ended'):
            self.ended = True
        if tr.get('task_complete'):
            self.success = True
        for s in (m.get('steps', []) or []) + (tr.get('steps', []) or []):
            if isinstance(s, dict):
                if s.get('task_complete'):
                    self.success = True
                if s.get('episode_ended'):
                    self.ended = True
        if m.get('task_complete'):
            self.success = True
        return m

    def move_to(self, target, grip, tol=0.004, max_iter=8, max_step=0.8, settle=False):
        """Move EE position to target (world m) keeping orientation pointing down."""
        target = np.asarray(target, float)
        for it in range(max_iter):
            if self.ended:
                return False
            p = self.state[:3]
            d = target - p
            dist = np.linalg.norm(d)
            if dist < tol:
                return True
            n = int(math.ceil(dist / (0.05 * max_step)))
            n = max(1, min(n, 40))
            per = d / n / 0.05
            rot = self.rot_correction()
            rows = []
            for k in range(n):
                r = list(per) + (list(rot) if k == 0 else [0, 0, 0]) + [grip]
                rows.append(r)
            if self.remaining() < len(rows) + 2:
                rows = rows[:max(1, self.remaining() - 1)]
            self.step(rows)
        return np.linalg.norm(target - self.state[:3]) < tol * 3

    def rot_correction(self, target_yaw=None):
        """Small correction toward gripper-down orientation (axis-angle (pi,0,0) family)."""
        rv = self.state[3:6]
        ang = np.linalg.norm(rv)
        if ang < 1e-6:
            return [0, 0, 0]
        Rc = axang_to_R(rv)
        Rt = self.target_R if getattr(self, 'target_R', None) is not None else axang_to_R(np.array([math.pi, 0, 0]))
        Re = Rt @ Rc.T
        e = R_to_axang(Re)
        e = np.clip(e / 0.5, -0.6, 0.6)
        if np.linalg.norm(e) < 0.01:
            return [0, 0, 0]
        return list(e)

    def gripper(self, g, n=8):
        rows = [[0, 0, 0, 0, 0, 0, g] for _ in range(n)]
        self.step(rows)

def axang_to_R(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx

def R_to_axang(R):
    c = (np.trace(R) - 1) / 2
    c = min(1, max(-1, c)); th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if abs(th - math.pi) < 1e-3:
        w, V = np.linalg.eigh((R + np.eye(3)) / 2)
        k = V[:, np.argmax(w)]
        return k * th
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return k * th

def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

R_DOWN = axang_to_R(np.array([math.pi, 0, 0]))

def rotate_to(rb, Rt, tol=0.03, max_calls=10, grip=-1):
    rb.target_R = Rt
    for _ in range(max_calls):
        if rb.ended:
            return False
        Rc = axang_to_R(rb.state[3:6])
        e = R_to_axang(Rt @ Rc.T)
        if np.linalg.norm(e) < tol:
            return True
        r = np.clip(e / 0.5, -0.8, 0.8)
        rb.step([[0, 0, 0] + list(r) + [grip]])
    return False

def yaw_delta(cube_yaw, cur_close_angle):
    """rotation (rad) about world z to align closing direction to a cube face normal, in [-pi/4, pi/4]."""
    d = (cube_yaw - cur_close_angle) % (math.pi / 2)
    if d > math.pi / 4:
        d -= math.pi / 2
    return d

def closing_angle(rb):
    """World xy angle of gripper closing direction (wrist cam x axis)."""
    q = rb.meta['camera_poses']['wrist']['quaternion_wxyz']
    R = quat_to_R(q)
    x = R[:, 0]
    return math.atan2(x[1], x[0])
