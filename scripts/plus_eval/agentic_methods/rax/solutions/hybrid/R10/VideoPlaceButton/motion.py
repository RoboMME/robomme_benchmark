"""Simple closed-loop Cartesian motion on top of robomme_step."""
import json
import numpy as np

def parse(reply):
    for b in reply.get('content', []):
        if b.get('type') == 'text':
            try:
                return json.loads(b['text'])
            except Exception:
                pass
    return {}

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx
        self.log = log
        self.last = None
        self.ended = False
        self.complete = False
        self.grip = -1.0

    def observe(self):
        r = parse(self.ctx.call('robomme_observe'))
        self.last = r
        return r

    def state(self):
        if self.last is None:
            self.observe()
        return np.array(self.last['state'], float)

    def remaining(self):
        lim = (self.last or {}).get('episode_limits') or {}
        return lim.get('remaining_episode_steps', 999)

    def step(self, rows):
        if self.ended:
            return self.last
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        rem = self.remaining()
        if rem is not None and rem <= 0:
            self.ended = True
            return self.last
        rows = rows[:max(1, min(100, rem))]
        r = parse(self.ctx.call('robomme_step', actions=rows))
        self.last = r
        if r.get('episode_ended'):
            self.ended = True
        for s in r.get('steps', []) or []:
            if isinstance(s, dict) and s.get('task_complete'):
                self.complete = True
        if r.get('task_complete'):
            self.complete = True
        tr = r.get('transition') or {}
        if tr.get('task_complete'):
            self.complete = True
        if tr.get('episode_ended'):
            self.ended = True
        for s in tr.get('steps', []) or []:
            if isinstance(s, dict):
                if s.get('task_complete'):
                    self.complete = True
                if s.get('episode_ended'):
                    self.ended = True
        return r

    def move_to(self, target, grip=None, tol=0.006, max_iter=12, vmax=0.03, chunk=None):
        """Move TCP to world target xyz keeping orientation pointing down."""
        if grip is not None:
            self.grip = grip
        target = np.asarray(target, float)
        for it in range(max_iter):
            st = self.state()
            d = target - st[:3]
            dist = float(np.linalg.norm(d))
            if dist < tol or self.ended:
                break
            k = int(np.ceil(dist / vmax))
            k = max(1, min(k, 20))
            per = d / k
            rot = self.rot_correction(st)
            rows = []
            for j in range(k):
                r = list(per / 0.05) + (list(rot) if j == 0 else [0, 0, 0]) + [self.grip]
                rows.append(r)
            # settle rows
            rows.append([0, 0, 0, 0, 0, 0, self.grip])
            self.step(rows)
        st = self.state()
        return float(np.linalg.norm(target - st[:3]))

    def rot_correction(self, st, gain=0.8):
        """Rotation delta (in action units) to bring the gripper back to pointing straight down,
        with yaw kept. Small-angle: compare current rotation with the down-pointing one."""
        from scipy.spatial.transform import Rotation as Rr
        cur = Rr.from_rotvec(st[3:6])
        # desired: z axis of tool pointing world -z, keep current yaw
        z = cur.apply([0, 0, 1.0])
        err = np.cross(z, [0, 0, -1.0])  # axis * sin(angle)
        n = np.linalg.norm(err)
        if n < 1e-4:
            return np.zeros(3)
        ang = np.arcsin(min(1.0, n))
        if np.dot(z, [0, 0, -1.0]) < 0:
            ang = np.pi - ang
        vec = err / n * ang * gain
        return np.clip(vec / 0.5, -1, 1)

    def gripper(self, g, n=8):
        self.grip = g
        return self.step([[0, 0, 0, 0, 0, 0, g]] * n)

    def closing_angle(self, st=None):
        from scipy.spatial.transform import Rotation as Rr
        st = self.state() if st is None else st
        y = Rr.from_rotvec(st[3:6]).apply([0, 1.0, 0])
        return float(np.arctan2(y[1], y[0]))

    def align_yaw(self, face_yaw, tol=0.03, max_iter=8):
        """Rotate about world z so the closing axis is parallel to a cube face normal (mod 90deg)."""
        err = 0.0
        for it in range(max_iter):
            if self.ended:
                break
            cur = self.closing_angle()
            err = (face_yaw - cur + np.pi / 4) % (np.pi / 2) - np.pi / 4
            if abs(err) < tol:
                break
            k = max(1, int(np.ceil(abs(err) / 0.2)))
            per = err / k / 0.5
            st = self.state()
            rows = [[0, 0, 0, 0, 0, per, self.grip] for _ in range(k)] + [[0, 0, 0, 0, 0, 0, self.grip]] * 2
            self.step(rows)
        return err

    def yaw_by(self, dyaw, n=None):
        n = n or max(1, int(np.ceil(abs(dyaw) / 0.25)))
        per = dyaw / n / 0.5
        self.step([[0, 0, 0, 0, 0, per, self.grip]] * n + [[0, 0, 0, 0, 0, 0, self.grip]])
