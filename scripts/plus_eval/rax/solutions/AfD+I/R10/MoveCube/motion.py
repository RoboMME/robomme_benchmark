import json, math
import numpy as np

def meta(reply):
    for b in reply:
        if isinstance(b, dict) and b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                pass
    if isinstance(reply, dict):
        if "content" in reply:
            return meta(reply["content"])
        return reply
    return {}

def rv_to_R(rv):
    rv = np.asarray(rv, float); th = np.linalg.norm(rv)
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K

def R_to_rv(Rm):
    c = max(-1.0, min(1.0, (np.trace(Rm) - 1) / 2)); th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: use symmetric part
        M = (Rm + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(M), 0, None))
        i = int(np.argmax(k)); k = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return k * th
    w = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]]) / (2 * math.sin(th))
    return w * th

R0 = np.diag([1.0, -1.0, -1.0])

def R_down(yaw):
    """gripper pointing down with finger (y) axis at world angle yaw."""
    a = yaw + math.pi / 2
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R0

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx
        self.log = log
        self.last = None
        self.ended = False
        self.complete = False
        self.grip = -1.0
        self.hold_yaw = None   # if set, keep gripper pointing down at this finger yaw

    def rot_err(self):
        if self.hold_yaw is None:
            return np.zeros(3)
        Rc = rv_to_R(self.state()[3:6])
        return R_to_rv(R_down(self.hold_yaw) @ Rc.T)

    def observe(self):
        r = self.ctx.call("robomme_observe")
        first = self.last is None
        self.last = meta(r)
        self.last_raw = r.get("content", r) if isinstance(r, dict) else r
        if first and "state" in self.last:
            st = self.last["state"]
            self.grip = 1.0 if (st[6] + st[7]) < 0.07 else -1.0
        return self.last

    def state(self):
        if self.last is None:
            self.observe()
        return np.array(self.last["state"], dtype=float)

    def remaining(self):
        lim = (self.last or {}).get("episode_limits", {})
        return lim.get("remaining_episode_steps", 10**6)

    def step(self, rows):
        if self.ended:
            return self.last
        rows = [[float(max(-1, min(1, v))) for v in r] for r in rows]
        r = self.ctx.call("robomme_step", actions=rows)
        m = meta(r)
        self.last_raw = r.get("content", r) if isinstance(r, dict) else r
        if "state" in m:
            self.last = m
        else:
            self.last = {**(self.last or {}), **m}
        if m.get("episode_ended"):
            self.ended = True
        for s in m.get("steps", []) or []:
            if isinstance(s, dict) and s.get("task_complete"):
                self.complete = True
            if isinstance(s, dict) and s.get("episode_ended"):
                self.ended = True
        if m.get("task_complete"):
            self.complete = True
        return m

    def move_to(self, target, grip=None, tol=0.004, max_steps=80, speed=1.0, rot_target=None):
        """closed loop move of TCP position to target (world m)."""
        if grip is not None:
            self.grip = grip
        target = np.array(target, dtype=float)
        used = 0
        while used < max_steps and not self.ended:
            p = self.state()[:3]
            d = target - p
            dist = np.linalg.norm(d)
            if dist < tol:
                break
            n = int(min(10, max(1, dist / 0.04)))
            rows = []
            # plan n rows along remaining displacement, each at most speed*0.05 m
            per = d / max(1, math.ceil(dist / (0.05 * speed)))
            k = max(1, min(n, math.ceil(dist / (0.05 * speed))))
            e = self.rot_err()
            for j in range(k):
                v = per / 0.05
                w = e / max(k, 1) / 0.5 if j < k else 0 * e
                w = np.clip(w, -0.3, 0.3)
                rows.append([v[0], v[1], v[2], w[0], w[1], w[2], self.grip])
            self.step(rows)
            used += len(rows)
        return self.state()[:3]

    def gripper(self, g, n=12):
        self.grip = g
        return self.step([[0, 0, 0, 0, 0, 0, g]] * n)

    def orient(self, yaw, iters=6, rate=0.15):
        """closed-loop orientation to pointing-down with given finger yaw."""
        self.hold_yaw = yaw
        for _ in range(iters):
            if self.ended:
                break
            e = self.rot_err(); a = np.linalg.norm(e)
            if a < 0.02:
                break
            n = max(1, int(math.ceil(a / rate)))
            w = e / n / 0.5
            self.step([[0, 0, 0, w[0], w[1], w[2], self.grip]] * n)
        return np.linalg.norm(self.rot_err())

    def rotate_z(self, angle, steps=None):
        # world-frame rotation about z by angle (rad)
        n = steps or max(1, math.ceil(abs(angle) / 0.25))
        a = angle / n / 0.5
        return self.step([[0, 0, 0, 0, 0, a, self.grip]] * n)
