import json, math
import numpy as np

def parse(reply):
    """MCP reply (list of content blocks or dict) -> JSON metadata dict."""
    if isinstance(reply, dict) and 'content' in reply:
        reply = reply['content']
    if isinstance(reply, list):
        for b in reply:
            if isinstance(b, dict) and b.get('type') == 'text':
                try: return json.loads(b['text'])
                except Exception: continue
    if isinstance(reply, dict): return reply
    raise RuntimeError('cannot parse reply')

def aa_to_R(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-9: return np.eye(3)
    k = v / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K

def R_to_aa(R):
    th = math.acos(max(-1, min(1, (np.trace(R) - 1) / 2)))
    if th < 1e-9: return np.zeros(3)
    if abs(th - math.pi) < 1e-4:
        # handle pi
        A = (R + np.eye(3)) / 2; k = np.sqrt(np.maximum(np.diag(A), 0))
        i = int(np.argmax(k)); k = A[:, i] / math.sqrt(max(A[i, i], 1e-12))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th

def yaw_of(aa):
    R = aa_to_R(aa)
    # gripper x-axis projected to world XY
    return math.atan2(R[1, 0], R[0, 0])

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log; self.last = None; self.ended = False; self.complete = False
        self.grip = -1.0

    def observe(self):
        self.last = parse(self.ctx.call('robomme_observe'))
        return self.last

    def state(self):
        return np.array(self.last['state'], float)

    def step(self, rows):
        rows = [[float(max(-1, min(1, v))) for v in r] for r in rows]
        meta = parse(self.ctx.call('robomme_step', actions=rows))
        self.last = meta
        self._check_end(meta)
        return meta

    def _check_end(self, meta):
        if meta.get('episode_ended'): self.ended = True
        if meta.get('task_complete'): self.complete = True
        steps = list(meta.get('steps', []) or []) + list((meta.get('transition') or {}).get('steps', []) or [])
        for s in steps:
            if isinstance(s, dict):
                if s.get('task_complete'): self.complete = True
                if s.get('episode_ended'): self.ended = True

    def remaining(self):
        lim = self.last.get('episode_limits', {}) if self.last else {}
        return lim.get('remaining_episode_steps', 9999)

    def move_to(self, target, yaw=None, speed=0.6, tol=0.004, yaw_tol=0.03, max_iter=12, grip=None):
        """Closed-loop Cartesian move (world xyz), optional absolute yaw (rad)."""
        if grip is not None: self.grip = grip
        target = np.asarray(target, float)
        for it in range(max_iter):
            if self.ended: return False
            s = self.state(); pos = s[:3]
            err = target - pos
            dyaw = 0.0
            if yaw is not None:
                dyaw = (yaw - yaw_of(s[3:6]) + math.pi) % (2 * math.pi) - math.pi
            dist = np.linalg.norm(err)
            if dist < tol and abs(dyaw) < yaw_tol:
                return True
            # number of rows to cover the distance at `speed`
            n = int(max(1, min(25, math.ceil(max(dist / (0.05 * speed), abs(dyaw) / (0.5 * 0.5))))))
            if dist < 0.02: n = max(1, min(n, 4))
            rows = []
            for i in range(n):
                d = err / n / 0.05
                rz = dyaw / n / 0.5
                rows.append([d[0], d[1], d[2], 0.0, 0.0, rz, self.grip])
            self.step(rows)
            if self.remaining() <= 0: return False
        return False

    def gripper(self, value, n=8):
        self.grip = value
        s = self.state()
        return self.step([[0, 0, 0, 0, 0, 0, value]] * n)
