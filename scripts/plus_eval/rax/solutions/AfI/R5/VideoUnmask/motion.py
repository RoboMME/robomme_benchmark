import json, math
import numpy as np


def packet(reply):
    for b in reply.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                pass
    return {}


class Robot:
    def __init__(self, ctx, log=print):
        self.ctx, self.log = ctx, log
        self.last = None
        self.ended = False
        self.success = False

    def observe(self):
        if self.ended and self.last is not None:
            return self.last
        try:
            rep = self.ctx.call("robomme_observe")
        except RuntimeError as e:
            if "nded" in str(e):
                self.ended = True
                return self.last or {}
            raise
        self.last = packet(rep)
        self.ended = bool(self.last.get("episode_ended", False)) or self.ended
        return self.last

    def state(self):
        if self.last is None:
            self.observe()
        return np.array(self.last["state"], float)

    def remaining(self):
        lim = (self.last or {}).get("episode_limits", {})
        return lim.get("remaining_episode_steps", 10**6)

    def step(self, rows):
        if self.ended:
            return self.last
        rows = [[float(max(-1.0, min(1.0, x))) for x in r] for r in rows]
        try:
            p = packet(self.ctx.call("robomme_step", actions=rows))
        except RuntimeError as e:
            if "nded" in str(e):
                self.ended = True
                return self.last
            raise
        self.last = p if "state" in p else self.last
        tr = p.get("transition") or {}
        if tr.get("task_complete"):
            self.success = True
        if tr.get("episode_ended"):
            self.ended = True
        for s in (p.get("steps") or []) + (tr.get("steps") or []):
            if s.get("task_complete"):
                self.success = True
            if s.get("episode_ended"):
                self.ended = True
        if p.get("task_complete"):
            self.success = True
        if p.get("episode_ended"):
            self.ended = True
        return p

    def move_to(self, target, grip, max_steps=40, tol=0.004, vmax=1.0, chunk=4):
        """Closed-loop translation toward target (world m), orientation held."""
        target = np.array(target, float)
        n = 0
        while n < max_steps and not self.ended:
            pos = self.state()[:3]
            err = target - pos
            if np.linalg.norm(err) < tol:
                break
            rows = []
            # plan a chunk assuming each row achieves its delta
            p = pos.copy()
            for _ in range(min(chunk, max_steps - n)):
                e = target - p
                d = np.clip(e / 0.05, -vmax, vmax)
                if np.linalg.norm(e) < tol:
                    break
                rows.append(list(d) + [0, 0, 0, grip])
                p = p + d * 0.05
            if not rows:
                break
            self.step(rows)
            n += len(rows)
        return self.state()[:3]

    def grip(self, g, n=8):
        pos = self.state()[:3]
        return self.step([[0, 0, 0, 0, 0, 0, g]] * n)


def aa_to_mat(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def yaw_of(state):
    R = aa_to_mat(state[3:6])
    return math.atan2(R[1, 0], R[0, 0])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rotate_yaw_to(robot, target_yaw, grip, max_iter=16, tol=math.radians(2.0)):
    for _ in range(max_iter):
        if robot.ended:
            break
        y = yaw_of(robot.state())
        e = wrap(target_yaw - y)
        if abs(e) < tol:
            break
        d = max(-1.0, min(1.0, e / 0.5))
        robot.step([[0, 0, 0, 0, 0, d, grip], [0, 0, 0, 0, 0, 0, grip]])
    return yaw_of(robot.state())
