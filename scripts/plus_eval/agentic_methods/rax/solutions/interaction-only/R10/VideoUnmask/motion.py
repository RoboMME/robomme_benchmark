import json
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
        self.complete = False
        self.grip = -1.0

    def observe(self):
        self.last = packet(self.ctx.call("robomme_observe"))
        return self.last

    def state(self):
        return np.array(self.last["state"], float)

    def remaining(self):
        lim = (self.last or {}).get("episode_limits") or {}
        return lim.get("remaining_episode_steps", 999)

    def step(self, rows):
        if self.ended:
            return self.last
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        p = packet(self.ctx.call("robomme_step", actions=rows))
        tr = p.get("transition") or {}
        for src in [p, tr] + list(tr.get("steps") or []) + list(p.get("steps") or []):
            if isinstance(src, dict):
                if src.get("task_complete"):
                    self.complete = True
                if src.get("episode_ended"):
                    self.ended = True
        if "state" in p:
            self.last = p
        else:
            self.observe()
        return p

    def move_to(self, target, grip=None, tol=0.004, max_iter=12, speed=0.5, chunk=4):
        """Closed-loop translation to world target (orientation held)."""
        if grip is not None:
            self.grip = grip
        target = np.asarray(target, float)
        for _ in range(max_iter):
            if self.ended:
                return False
            pos = self.state()[:3]
            err = target - pos
            if np.linalg.norm(err) < tol:
                return True
            rows = []
            # plan a chunk of steps assuming each step realises its command
            p = pos.copy()
            for _k in range(chunk):
                e = target - p
                d = np.clip(e / 0.05, -speed, speed)
                if np.linalg.norm(e) < tol * 0.5:
                    break
                rows.append([d[0], d[1], d[2], 0, 0, 0, self.grip])
                p = p + d * 0.05
            if not rows:
                rows = [[0, 0, 0, 0, 0, 0, self.grip]]
            self.step(rows)
        return np.linalg.norm(target - self.state()[:3]) < tol * 3

    def set_gripper(self, g, n=8):
        self.grip = g
        self.step([[0, 0, 0, 0, 0, 0, g]] * n)

    def rotate_z(self, dyaw, n=None):
        # world-frame rotation-vector delta around z
        n = n or max(1, int(np.ceil(abs(dyaw) / 0.25)))
        per = dyaw / n / 0.5
        self.step([[0, 0, 0, 0, 0, per, self.grip]] * n)


def rotvec_to_R(rv):
    rv = np.asarray(rv, float)
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def gripper_yaw(state):
    R = rotvec_to_R(state[3:6])
    x = R[:, 0]
    return float(np.arctan2(x[1], x[0]))


def wrap90(a):
    """wrap angle (rad) into [-pi/4, pi/4) modulo pi/2"""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def rotate_to_yaw(robot, target_yaw, tol_deg=2.0, max_iter=10):
    for _ in range(max_iter):
        if robot.ended:
            return False
        cur = gripper_yaw(robot.state())
        err = (target_yaw - cur + np.pi) % (2 * np.pi) - np.pi
        if abs(np.degrees(err)) < tol_deg:
            return True
        n = 4
        per = float(np.clip(err / 0.5 / 2.0, -0.6, 0.6))  # response ~40-50%, spread over steps
        robot.step([[0, 0, 0, 0, 0, per, robot.grip]] * n)
    return False
