"""Motion helpers for VideoPlaceOrder (Panda, ee-delta actions)."""
import json, math
import numpy as np

def meta(reply):
    if isinstance(reply, dict):
        reply = reply.get('content', [])
    for b in reply:
        if isinstance(b, dict) and b.get('type') == 'text':
            try:
                return json.loads(b['text'])
            except Exception:
                pass
    raise RuntimeError('no json in reply')

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx
        self.log = log
        self.ended = False
        self.complete = False
        self.state = None
        self.remaining = None
        self.last = None

    def observe(self):
        m = meta(self.ctx.call('robomme_observe'))
        self.last = m
        self.state = np.array(m['state'], float)
        lim = m.get('episode_limits') or {}
        self.remaining = lim.get('remaining_episode_steps', self.remaining)
        if m.get('episode_ended'):
            self.ended = True
        return m

    def step(self, rows):
        if self.ended:
            return None
        if self.remaining is not None:
            rows = rows[:max(0, self.remaining)]
            if not rows:
                return None
        m = meta(self.ctx.call('robomme_step', actions=[[float(v) for v in r] for r in rows]))
        self.last = m
        if 'state' in m:
            self.state = np.array(m['state'], float)
        lim = m.get('episode_limits') or {}
        self.remaining = lim.get('remaining_episode_steps', self.remaining)
        for s in (m.get('steps') or []) + ((m.get('transition') or {}).get('steps') or []):
            if s.get('task_complete'):
                self.complete = True
            if s.get('episode_ended'):
                self.ended = True
        if m.get('task_complete'):
            self.complete = True
        if m.get('episode_ended'):
            self.ended = True
        return m

    def move_to(self, target, grip, tol=0.004, max_steps=80, chunk=4, gain=1.0, vmax=1.0):
        """Closed-loop translation to target (world m), holding orientation pointing down."""
        target = np.asarray(target, float)
        used = 0
        while used < max_steps and not self.ended:
            if self.state is None:
                self.observe()
            pos = self.state[:3]
            err = target - pos
            if np.linalg.norm(err) < tol:
                return True
            rows = []
            # plan a few steps open-loop with predicted position
            p = pos.copy()
            for _ in range(chunk):
                e = target - p
                a = np.clip(gain * e / 0.05, -vmax, vmax)
                rot = self.rot_correction()
                rows.append(list(a) + list(rot) + [grip])
                p = p + a * 0.05
                if np.linalg.norm(target - p) < tol * 0.5:
                    break
            self.step(rows)
            used += len(rows)
        return np.linalg.norm(target - self.state[:3]) < tol * 2

    def rot_correction(self):
        """Keep gripper pointing straight down (axis-angle ~ (pi,0,0)) - small proportional fix."""
        rv = self.state[3:6]
        R = rotvec_to_mat(rv)
        Rt = rotvec_to_mat(np.array([math.pi, 0, 0]))
        Rt = Rt @ rotz(self.yaw_target if hasattr(self, 'yaw_target') else 0.0)
        Rerr = Rt @ R.T
        w = mat_to_rotvec(Rerr)
        return list(np.clip(w / 0.5 * 0.8, -0.5, 0.5))

    def grip(self, val, n=10):
        pos = self.state[:3]
        rows = [[0, 0, 0] + self.rot_correction() + [val] for _ in range(n)]
        return self.step(rows)

def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rotvec_to_mat(rv):
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K

def mat_to_rotvec(R):
    c = (np.trace(R) - 1) / 2
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi
        M = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(M), 0, None))
        i = int(np.argmax(k))
        k = M[:, i] / math.sqrt(M[i, i])
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th
