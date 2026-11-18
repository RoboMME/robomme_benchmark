"""Robot I/O and simple closed-loop motion for the Panda (official 7-D delta rows)."""
import base64, io, json, time
import numpy as np
from PIL import Image
from geom import backproject

POS_SCALE = 0.05   # 1.0 -> 0.05 m
ROT_SCALE = 0.5    # 1.0 -> 0.5 rad

def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)

# ---------- rotations ----------
def rotvec_to_R(r):
    r = np.asarray(r, float); th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx

def R_to_rotvec(R):
    c = (np.trace(R) - 1) / 2
    c = min(1.0, max(-1.0, c)); th = np.arccos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > np.pi - 1e-4:
        # near pi: use symmetric part
        M = (R + np.eye(3)) / 2
        k = np.sqrt(np.maximum(np.diag(M), 0))
        i = np.argmax(k)
        k[(i + 1) % 3] = np.copysign(k[(i + 1) % 3], M[i, (i + 1) % 3])
        k[(i + 2) % 3] = np.copysign(k[(i + 2) % 3], M[i, (i + 2) % 3])
        return k / np.linalg.norm(k) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

R_DOWN = rotvec_to_R([np.pi, 0, 0])   # gripper pointing down, yaw 0

def down_R(yaw):
    return Rz(yaw) @ R_DOWN

def current_yaw(state):
    R = rotvec_to_R(state[3:6]) @ R_DOWN.T
    return np.arctan2(R[1, 0], R[0, 0])

def wrap_half_pi(a):
    """Gripper is symmetric under 180 deg: wrap to (-pi/2, pi/2]."""
    return (a + np.pi / 2) % np.pi - np.pi / 2

# ---------- observation ----------
class Obs:
    def __init__(self, result):
        content = result['content']
        self.meta = json.loads(next(b['text'] for b in content if b['type'] == 'text'))
        m = self.meta
        self.step_id = m.get('step_id')
        self.state = np.array(m['state'], float) if m.get('state') is not None else None
        self.ended = bool(m.get('episode_ended'))
        self.task_complete = m.get('task_complete')
        self.limits = m.get('episode_limits') or {}
        self.images = {}
        self.K = m.get('camera_intrinsics') or {}
        self.poses = m.get('camera_poses') or {}
        for att in m.get('attachments', []) or []:
            b = content[att['content_index']]
            if b.get('type') != 'image':
                continue
            im = Image.open(io.BytesIO(base64.b64decode(b['data'])))
            if att['kind'] == 'rgb':
                self.images[(att['camera'], 'rgb')] = np.array(im.convert('RGB')).astype(np.float32)
            else:
                mpu = ((m.get('depth') or {}).get(att['camera']) or {}).get('meters_per_unit', 0.001) or 0.001
                self.images[(att['camera'], 'depth')] = np.array(im).astype(np.float32) * mpu
        self._P = {}

    def rgb(self, cam='front'):
        return self.images.get((cam, 'rgb'))

    def points(self, cam='front'):
        if cam not in self._P:
            d = self.images.get((cam, 'depth'))
            if d is None or cam not in self.K or cam not in self.poses:
                return None
            self._P[cam] = backproject(d, self.K[cam]['matrix'], self.poses[cam])
        return self._P[cam]

    def remaining(self):
        return self.limits.get('remaining_episode_steps')

def observe(ctx):
    return Obs(ctx.call('robomme_observe'))

class EpisodeEnded(Exception):
    pass

class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.obs = observe(ctx)
        self.steps = 0
        self.gripper = -1.0

    @property
    def state(self):
        return self.obs.state

    def step(self, rows):
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        res = self.ctx.call('robomme_step', actions=rows)
        o = Obs(res)
        self.steps += len(rows)
        if o.state is None or not o.images:
            o2 = observe(self.ctx)
            o2.ended = o2.ended or o.ended
            o2.task_complete = o.task_complete if o.task_complete is not None else o2.task_complete
            o = o2
        self.obs = o
        if o.ended:
            raise EpisodeEnded(f'episode ended task_complete={o.task_complete}')
        return o

    def refresh(self):
        self.obs = observe(self.ctx)
        return self.obs

    def hold(self, n, gripper=None):
        g = self.gripper if gripper is None else gripper
        self.gripper = g
        return self.step([[0, 0, 0, 0, 0, 0, g]] * n)

    def move_to(self, target, yaw=None, gripper=None, tol=0.004, rot_tol=0.03, max_steps=80,
                chunk=4, vmax=1.0, settle=0):
        """Closed-loop move of the TCP to world `target` with gripper down at `yaw`."""
        g = self.gripper if gripper is None else gripper
        self.gripper = g
        target = np.asarray(target, float)
        used = 0
        while used < max_steps:
            s = self.state
            err = target - s[:3]
            Rc = rotvec_to_R(s[3:6])
            if yaw is None:
                Rt = down_R(current_yaw(s))
            else:
                Rt = down_R(yaw)
            rerr = R_to_rotvec(Rt @ Rc.T)
            if np.linalg.norm(err) < tol and np.linalg.norm(rerr) < rot_tol:
                break
            rows = []
            n = min(chunk, max_steps - used)
            # plan n rows from current error (open-loop within the chunk)
            for i in range(n):
                frac = 1.0 / max(1, (n - i))
                dp = np.clip(err / POS_SCALE, -vmax, vmax)
                dr = np.clip(rerr / ROT_SCALE, -1, 1)
                rows.append(list(dp) + list(dr) + [g])
                err = err - dp * POS_SCALE
                rerr = rerr - dr * ROT_SCALE
                if np.linalg.norm(err) < 1e-4 and np.linalg.norm(rerr) < 1e-3:
                    break
            self.step(rows)
            used += len(rows)
        if settle:
            self.hold(settle, g)
        s = self.state
        return float(np.linalg.norm(target - s[:3])), used
