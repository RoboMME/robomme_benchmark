import json, math, time
import numpy as np

def parse(reply):
    """reply: MCP result (dict with content) or list of blocks -> json dict of first text block."""
    content = reply.get('content', reply) if isinstance(reply, dict) else reply
    for b in content:
        if b.get('type') == 'text':
            return json.loads(b['text'])
    raise RuntimeError('no text block')

def axis_angle_to_R(r):
    r = np.asarray(r, float); th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K

def R_to_axis_angle(R):
    c = (np.trace(R) - 1) / 2; c = max(-1.0, min(1.0, c)); th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if abs(th - math.pi) < 1e-3:
        M = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(M))); v = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return v * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return v * th

def wrap(a, period):
    return (a + period / 2) % period - period / 2

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log; self.last = None; self.ended = False
        self.task_complete = None; self.grip = -1.0

    def observe(self):
        r = self.ctx.call('robomme_observe'); d = parse(r); self.last = d
        return d

    def state(self):
        return np.array(self.last['state'])

    def remaining(self):
        lim = self.last.get('episode_limits') or {}
        return lim.get('remaining_episode_steps', 999)

    def step(self, rows):
        if self.ended:
            return self.last
        rows = [[float(max(-1, min(1, x))) for x in r] for r in rows]
        r = self.ctx.call('robomme_step', actions=rows)
        d = parse(r); self.last = d
        if d.get('episode_ended'):
            self.ended = True
            tr = d.get('transition') or {}
            if tr.get('task_complete') is not None:
                self.task_complete = tr.get('task_complete')
            for s in (tr.get('steps') or []) + (d.get('steps') or []):
                if s.get('task_complete') is not None:
                    self.task_complete = s.get('task_complete')
            self.log(f"EPISODE ENDED task_complete={self.task_complete} keys={list(d.keys())}")
        return d

    def closing_yaw(self):
        R = axis_angle_to_R(self.state()[3:6])
        ax = R[:, 1]
        return math.atan2(ax[1], ax[0])

    def ee_R(self):
        return axis_angle_to_R(self.state()[3:6])

    def move(self, target, yaw=None, vmax=0.4, tol=0.004, ytol=0.03, max_iter=8, grip=None, chunk=12):
        """Closed-loop move of TCP to target xyz; yaw = desired closing axis angle (mod pi)."""
        if grip is not None:
            self.grip = grip
        target = np.asarray(target, float)
        for it in range(max_iter):
            if self.ended:
                return False
            st = self.state(); err = target - st[:3]
            dyaw = 0.0
            if yaw is not None:
                dyaw = wrap(yaw - self.closing_yaw(), math.pi)
            dist = np.linalg.norm(err)
            if dist < tol and abs(dyaw) < ytol:
                return True
            n = int(math.ceil(max(dist / (0.05 * vmax), abs(dyaw) / (0.5 * 0.5), 1)))
            n = min(n, chunk)
            rows = []
            for k in range(n):
                frac = 1.0 / (n - k)
                # remaining error assumed to shrink linearly
                d = err * (1.0 / n) / 0.05
                rz = (dyaw / n) / 0.5
                rows.append([d[0], d[1], d[2], 0.0, 0.0, rz, self.grip])
            self.step(rows)
        st = self.state()
        return np.linalg.norm(target - st[:3]) < 3 * tol

    def gripper(self, g, n=12):
        self.grip = g
        st = self.state()
        self.step([[0, 0, 0, 0, 0, 0, g]] * n)
        return self.state()[6:8].sum()
