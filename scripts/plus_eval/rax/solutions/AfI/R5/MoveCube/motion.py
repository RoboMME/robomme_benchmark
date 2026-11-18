"""Motion helpers for Panda via robomme_step (world-frame deltas)."""
import math, time
import numpy as np
from perc import parse_reply

POS_SCALE = 0.05
ROT_SCALE = 0.5
LOG = []

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)

def rotvec_to_R(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-9: return np.eye(3)
    k = v / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K

def R_to_rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2)); th = math.acos(c)
    if th < 1e-7: return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi
        M = (R + np.eye(3)) / 2; i = int(np.argmax(np.diag(M)))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-12)); return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th

def down_R(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Rz @ np.diag([1.0, -1.0, -1.0])

def yaw_of(state):
    R = rotvec_to_R(state[3:6]); return math.atan2(R[1, 0], R[0, 0])

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

class Robot:
    def __init__(self, ctx):
        self.ctx = ctx; self.meta = None; self.imgs = None; self.ended = False
        self.complete = False; self.grip = -1.0; self.steps_used = 0; self.remaining = None

    def observe(self):
        self.meta, self.imgs = parse_reply(self.ctx.call("robomme_observe"))
        self._lim(self.meta); return self.meta

    def _lim(self, meta):
        lim = meta.get("episode_limits") or {}
        if "remaining_episode_steps" in lim: self.remaining = lim["remaining_episode_steps"]
        if meta.get("episode_ended"): self.ended = True
        tr = meta.get("transition") or {}
        if tr.get("episode_ended"): self.ended = True
        for st in (tr.get("steps") or []) + (meta.get("steps") or []):
            if st.get("task_complete"): self.complete = True
            if st.get("episode_ended"): self.ended = True
        if meta.get("task_complete"): self.complete = True

    @property
    def state(self):
        return np.array(self.meta["state"], float)

    def step(self, rows):
        if self.ended: return False
        rows = [[float(max(-1, min(1, x))) for x in r] for r in rows]
        if self.remaining is not None:
            rows = rows[:max(0, self.remaining)]
        if not rows: self.ended = True; return False
        self.meta, self.imgs = parse_reply(self.ctx.call("robomme_step", actions=rows))
        self.steps_used += len(rows); self._lim(self.meta)
        return not self.ended

    def hold(self, n, grip=None):
        if grip is not None: self.grip = grip
        return self.step([[0, 0, 0, 0, 0, 0, self.grip]] * n)

    def gripper(self, g, n=12):
        self.grip = g; return self.hold(n)

    def move_to(self, xyz, yaw=None, tol=0.004, yaw_tol=0.03, max_iter=8, vmax=0.8, chunk=25, R_des=None):
        """Closed-loop move: open-loop chunks of constant delta rows, then re-measure."""
        xyz = np.asarray(xyz, float)
        stalls = 0; prev = None
        for it in range(max_iter):
            st = self.state; p = st[:3]
            if prev is not None:
                if np.linalg.norm(xyz - p) > 0.012 and np.linalg.norm(p - prev) < 0.002:
                    stalls += 1
                    if stalls >= 2:
                        log("move_to stalled", np.round(xyz, 3), np.round(p, 3)); return False
                else:
                    stalls = 0
            prev = p.copy()
            if R_des is None:
                Rd = down_R(yaw if yaw is not None else yaw_of(st))
            else:
                Rd = R_des
            Rc = rotvec_to_R(st[3:6])
            rerr = R_to_rotvec(Rd @ Rc.T)
            dp = xyz - p
            if np.linalg.norm(dp) < tol and np.linalg.norm(rerr) < yaw_tol:
                return True
            n_p = int(math.ceil(np.max(np.abs(dp)) / (POS_SCALE * vmax)))
            n_r = int(math.ceil(np.max(np.abs(rerr)) / (ROT_SCALE * 0.6)))
            n = max(1, n_p, n_r); n = min(n, chunk)
            frac = min(1.0, n / max(1, max(n_p, n_r)))
            dpos = dp * frac / n / POS_SCALE; drot = rerr * frac / n / ROT_SCALE
            row = list(dpos) + list(drot) + [self.grip]
            if not self.step([row] * n): return False
            # settle one step if we were close
        st = self.state
        log("move_to not converged", xyz, st[:3])
        return np.linalg.norm(xyz - st[:3]) < 3 * tol
