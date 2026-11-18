"""Robot wrapper: MCP calls, state tracking, closed-loop Cartesian motion primitives."""
import json
import math
import time
import traceback

import numpy as np

import rax_io

POS_SCALE = 0.05     # m per unit action
ROT_SCALE = 0.5      # rad per unit action
OPEN, CLOSE = -1.0, 1.0


# --------------------------------------------------------------------------- rotations
def rotvec_to_R(r):
    r = np.asarray(r, dtype=float)
    th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def R_to_rotvec(R):
    c = (np.trace(R) - 1) / 2
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: axis from the diagonal
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.maximum(np.diag(A), 0))
        i = int(np.argmax(k))
        if k[i] < 1e-9:
            return np.zeros(3)
        k = A[:, i] / k[i]
        k = k / np.linalg.norm(k)
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def yaw_of(R):
    """Yaw of the first TCP axis that is not the (near-vertical) approach axis."""
    vert = int(np.argmax(np.abs(R[2, :])))
    ref = 0 if vert != 0 else 1
    return math.atan2(R[1, ref], R[0, ref])


def wrap90(a):
    """wrap angle to [-pi/4, pi/4)."""
    q = math.pi / 2
    return (a + q / 2) % q - q / 2


class EpisodeOver(Exception):
    pass


class Robot:
    def __init__(self, ctx, log):
        self.ctx = ctx
        self.log = log
        self.state = None
        self.ended = False
        self.complete = False
        self.steps = 0
        self.step_limit = None
        self.remaining = None
        self.last_json = {}
        self.R_ref = None
        self.R_des = None
        self.rot_mode = "world"   # or "body" or "off"
        self.rot_flips = 0
        self.t0 = time.time()
        self.last_obs = None
        self.calls = 0
        self.resp = 1.0          # EMA of achieved / commanded translation per row

    # ------------------------------------------------------------------ io
    def _absorb(self, j):
        self.last_json = j
        s = rax_io.get_state(j)
        if s is not None:
            self.state = s
        if rax_io.episode_ended(j):
            self.ended = True
        if rax_io.task_complete(j):
            self.complete = True
        lim = rax_io.find_key(j, ["episode_limits"])
        if isinstance(lim, dict):
            for k, v in lim.items():
                kl = k.lower()
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if "remaining" in kl or "left" in kl:
                        self.remaining = int(v)
                    elif ("max" in kl or "limit" in kl) and "step" in kl:
                        self.step_limit = int(v)
                    elif kl in ("steps", "step_limit", "max_steps", "horizon"):
                        self.step_limit = int(v)
        sid = rax_io.get_step_id(j)
        if isinstance(sid, int):
            self.steps = max(self.steps, sid)

    def status(self):
        try:
            r = self.ctx.call("robomme_status")
            j = rax_io.reply_json(r)
            self._absorb(j)
            return j
        except Exception as e:
            self.log("robomme_status failed: %r" % e)
            return {}

    def observe(self):
        r = self.ctx.call("robomme_observe")
        self.calls += 1
        o = rax_io.build_obs(r)
        self._absorb(o.json)
        self.last_obs = o
        self.last_reply = r
        return o

    def pixel_to_position(self, step_id, sels, radius=1):
        out = []
        for k in range(0, len(sels), 32):
            chunk = sels[k:k + 32]
            args = {"observation_step_id": int(step_id) if step_id is not None else 0,
                    "selections": [{"camera": c, "u": float(u), "v": float(v)} for c, u, v in chunk],
                    "depth_window_radius": int(radius)}
            r = self.ctx.call("robomme_pixel_to_position", **args)
            self.calls += 1
            out.extend(rax_io.parse_points_reply(r, len(chunk)))
        return out

    def remaining_steps(self):
        if self.remaining is not None:
            return self.remaining
        if self.step_limit is not None:
            return self.step_limit - self.steps
        return 10 ** 6

    def send(self, rows):
        if self.ended:
            raise EpisodeOver()
        rows = [[float(max(-1.0, min(1.0, x))) for x in r] for r in rows]
        rem = self.remaining_steps()
        if rem <= 0:
            self.ended = True
            raise EpisodeOver()
        rows = rows[:max(1, min(100, rem))]
        try:
            r = self.ctx.call("robomme_step", actions=rows)
        except RuntimeError as e:
            self.log("robomme_step error: %s" % str(e)[:300])
            # do not retry blindly: refresh state
            self.status()
            try:
                self.observe()
            except Exception as e2:
                self.log("observe after error failed: %r" % e2)
            if self.ended:
                raise EpisodeOver()
            return None
        self.calls += 1
        j = rax_io.reply_json(r)
        before = self.steps
        self._absorb(j)
        if self.steps == before:
            self.steps += len(rows)
            if self.remaining is not None and rax_io.find_key(j, ["episode_limits"]) is None:
                self.remaining -= len(rows)
        if rax_io.get_state(j) is None:
            try:
                self.observe()
            except Exception as e:
                self.log("observe failed: %r" % e)
        if self.ended:
            self.log("episode ended (complete=%s) at step %s" % (self.complete, self.steps))
            raise EpisodeOver()
        return j

    # ------------------------------------------------------------------ state helpers
    @property
    def pos(self):
        return np.array(self.state[:3])

    @property
    def R(self):
        return rotvec_to_R(self.state[3:6])

    @property
    def width(self):
        return float(self.state[6] + self.state[7])

    def rot_cmd(self, n_rows):
        """Per-row rotation command (action units) to drive R toward R_des."""
        if self.R_des is None or self.rot_mode == "off":
            return np.zeros(3), 0.0
        Rc = self.R
        e = R_to_rotvec(self.R_des @ Rc.T)
        err = float(np.linalg.norm(e))
        if self.rot_mode == "body":
            e = Rc.T @ e
        per = e / max(1, n_rows)
        m = np.linalg.norm(per)
        if m > 0.12:
            per = per * 0.12 / m
        return per / ROT_SCALE, err

    def rot_error(self):
        if self.R_des is None:
            return 0.0
        return float(np.linalg.norm(R_to_rotvec(self.R_des @ self.R.T)))

    # ------------------------------------------------------------------ motions
    def move_to(self, target, grip, speed=0.025, tol=0.005, max_calls=40, rows_per_call=6,
                rot_tol=0.03, label=""):
        target = np.asarray(target, dtype=float)
        stall = 0
        last_d = None
        for it in range(max_calls):
            p = self.pos
            e = target - p
            d = float(np.linalg.norm(e))
            rerr = self.rot_error() if self.rot_mode != "off" else 0.0
            if d < tol and rerr < rot_tol:
                return True, d
            n_need = max(1, int(math.ceil(d / speed)))
            n = min(rows_per_call, n_need)
            per = e / n_need
            # lag compensation: only far from the target, bounded, so it cannot overshoot much
            gain = 1.0
            if d > 0.03 and self.resp < 0.8:
                gain = min(2.0, 0.8 / max(self.resp, 0.3))
                if np.linalg.norm(per * gain) * n > d - 0.01:
                    gain = max(1.0, (d - 0.01) / max(1e-6, np.linalg.norm(per) * n))
            per = per * gain
            rot, _ = self.rot_cmd(max(1, min(n_need, 4)))
            row = list(per / POS_SCALE) + list(rot) + [grip]
            rerr0 = rerr
            self.send([row] * n)
            # rotation convention safety check
            if self.rot_mode != "off" and rerr0 > 0.08:
                rerr1 = self.rot_error()
                if rerr1 > rerr0 + 0.03:
                    self.rot_flips += 1
                    old = self.rot_mode
                    self.rot_mode = "body" if old == "world" else "off"
                    if self.rot_flips >= 2:
                        self.rot_mode = "off"
                    self.log("rotation error grew %.3f->%.3f; mode %s->%s" % (rerr0, rerr1, old, self.rot_mode))
            d1 = float(np.linalg.norm(target - self.pos))
            cmd = float(np.linalg.norm(per)) * n
            if cmd > 0.01:
                ach = float(np.dot(e / max(d, 1e-9), p + 0 - p)) if False else (d - d1)
                self.resp = 0.7 * self.resp + 0.3 * float(np.clip(ach / cmd, 0.05, 1.5))
            expected = min(d, n * speed)
            if d - d1 < 0.12 * expected and d > tol:
                stall += 1
            else:
                stall = 0
            if stall >= 3:
                self.log("move_to[%s] stalled at d=%.3f" % (label, d1))
                return False, d1
            last_d = d1
        return False, float(np.linalg.norm(target - self.pos))

    def hold(self, grip, n=8):
        rot, _ = self.rot_cmd(4)
        self.send([[0, 0, 0] + list(rot) + [grip]] * n)

    def gripper(self, grip, n=12):
        """Actuate gripper while holding the pose; return final width."""
        tgt = self.pos.copy()
        rows = []
        for _ in range(n):
            rows.append([0, 0, 0, 0, 0, 0, grip])
        self.send(rows[: n // 2])
        e = (tgt - self.pos) / POS_SCALE
        rot, _ = self.rot_cmd(4)
        self.send([list(np.clip(e / max(1, n - n // 2), -0.2, 0.2)) + list(rot) + [grip]] * (n - n // 2))
        # wait until the fingers settle (at most 3 more short holds)
        for _ in range(3):
            w0 = self.width
            self.send([[0, 0, 0] + list(rot * 0) + [grip]] * 4)
            if abs(self.width - w0) < 0.001:
                break
        return self.width
