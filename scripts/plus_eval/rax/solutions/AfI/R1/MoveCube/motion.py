"""Robot interface: stepping, probes, closed-loop moves on measured state."""
import math
import time

import numpy as np

import geom
import mcio
from perception import Obs

POS_SCALE = 0.05
ROT_SCALE = 0.5


class EpisodeEnded(Exception):
    pass


class OutOfBudget(Exception):
    pass


class Robot:
    def __init__(self, ctx, log=print, t0=None, wall_limit=1100.0):
        self.ctx = ctx
        self.log = log
        self.t0 = t0 or time.time()
        self.wall_limit = wall_limit
        self.reply = None
        self.state = None
        self.step_id = 0
        self.ended = False
        self.complete = False
        self.max_steps = None
        self.remaining = None
        self.M = np.eye(3) * POS_SCALE      # measured displacement per unit command (columns)
        self.Minv = np.linalg.inv(self.M)
        self.rot_mode = "world"
        self.approach_axis = 2              # local axis pointing down
        self.closing_axis = 1               # local axis along which fingers close
        self.R_down0 = None
        self.calls = 0
        self.steps_sent = 0

    # ---------------- basic I/O ----------------
    def wall(self):
        return time.time() - self.t0

    def _absorb(self, reply):
        self.reply = reply
        m = mcio.meta(reply)
        s = mcio.get_state(m)
        if s is not None:
            self.state = s
        sid = mcio.get_step_id(m)
        if sid is not None:
            self.step_id = sid
        if mcio.get_bool(m, "episode_ended"):
            self.ended = True
        if mcio.get_bool(m, "task_complete") or mcio.get_bool(m, "success"):
            self.complete = True
        lim = mcio.find_shallow(m, "episode_limits", lambda x: isinstance(x, dict))
        if isinstance(lim, dict):
            for k, v in lim.items():
                kl = k.lower()
                if isinstance(v, int) and not isinstance(v, bool):
                    if "remain" in kl:
                        self.remaining = v
                    elif "max" in kl or "limit" in kl or kl in ("steps", "step_limit", "horizon"):
                        self.max_steps = v
        return m

    def observe(self):
        r = self.ctx.call("robomme_observe")
        self.calls += 1
        self._absorb(r)
        return r

    def status(self):
        try:
            r = self.ctx.call("robomme_status")
            m = self._absorb(r)
            return m
        except Exception as e:
            self.log("robomme_status failed", repr(e)[:300])
            return {}

    def steps_left(self):
        if self.remaining is not None:
            return self.remaining
        if self.max_steps is not None:
            return self.max_steps - (self.step_id or 0)
        return 10 ** 6

    def send(self, rows):
        """Send rows (list of 7-lists). Returns last reply. Raises EpisodeEnded."""
        if self.ended:
            raise EpisodeEnded()
        if self.wall() > self.wall_limit:
            raise OutOfBudget("wall")
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        left = self.steps_left()
        if left <= 0:
            raise OutOfBudget("steps")
        rows = rows[:max(1, left)]
        r = None
        for i in range(0, len(rows), 100):
            r = self.ctx.call("robomme_step", actions=rows[i:i + 100])
            self.calls += 1
            self.steps_sent += len(rows[i:i + 100])
            self._absorb(r)
            if self.remaining is not None:
                self.remaining = self.remaining  # authoritative from reply when present
            if self.ended:
                raise EpisodeEnded()
        return r

    def obs(self):
        """Obs built from the latest reply (images of the final frame)."""
        return Obs(self.reply)

    # ---------------- state helpers ----------------
    @property
    def pos(self):
        return np.array(self.state[:3])

    @property
    def R(self):
        return geom.rotvec_to_R(self.state[3:6])

    @property
    def grip_width(self):
        return float(self.state[6] + self.state[7])

    def yaw(self, R=None):
        R = self.R if R is None else R
        c = R[:, self.closing_axis]
        return math.atan2(c[1], c[0])

    def R_des(self, yaw):
        """Downward-pointing orientation whose closing axis has world yaw `yaw`."""
        base = self.R_down0
        cur = math.atan2(base[1, self.closing_axis], base[0, self.closing_axis])
        return geom.Rz(yaw - cur) @ base

    def rot_err(self, Rd):
        e = geom.R_to_rotvec(Rd @ self.R.T)  # world-frame error
        return e

    def rot_cmd(self, e_world):
        if self.rot_mode == "world":
            return e_world
        return self.R.T @ e_world

    # ---------------- probes ----------------
    def setup_orientation(self):
        R = self.R
        down = np.array([0, 0, -1.0])
        dots = [R[:, i] @ down for i in range(3)]
        ai = int(np.argmax(np.abs(dots)))
        self.approach_axis = ai
        sign = 1 if dots[ai] > 0 else -1
        a = R[:, ai] * sign
        # minimal rotation making approach axis exactly down
        v = np.cross(a, down)
        s = np.linalg.norm(v)
        if s > 1e-9:
            ang = math.atan2(s, a @ down)
            Rc = geom.rotvec_to_R(v / s * ang)
        else:
            Rc = np.eye(3)
        self.R_down0 = Rc @ R
        if self.approach_axis != 2:
            self.closing_axis = [i for i in range(3) if i != ai][1]
        self.log(f"orient: approach axis {ai} (sign {sign}, tilt {math.degrees(math.acos(min(1, abs(dots[ai])))):.1f} deg),"
                 f" closing axis {self.closing_axis}, yaw0 {math.degrees(self.yaw()):.1f}")

    def probe(self, grip=-1.0):
        """Measure translation gain matrix and rotation composition (about 12 steps)."""
        g = grip
        cols = [None, None, None]
        ok = True
        for ax, amt in ((2, 0.3), (0, 0.25), (1, 0.25)):
            p0 = self.pos.copy()
            row = [0.0] * 7
            row[ax] = amt
            row[6] = g
            self.send([row, row, [0, 0, 0, 0, 0, 0, g]])
            d = (self.pos - p0) / (2 * amt)
            cols[ax] = d
        M = np.array(cols).T
        self.log("probe M (per unit cmd):", np.round(M, 4).tolist())
        diag = np.diag(M)
        if np.all(diag > 0.02) and np.all(np.abs(M - np.diag(diag)) < 0.4 * diag.min()):
            gains = np.clip(diag, 0.02, 0.08)
            self.M = np.diag(gains)
        elif abs(np.linalg.det(M)) > 1e-6 and np.linalg.cond(M) < 10:
            self.M = M
            self.log("WARNING: non-diagonal translation map, using measured M")
        else:
            self.log("WARNING: probe unreliable; keeping nominal map")
            ok = False
        self.Minv = np.linalg.inv(self.M)
        # rotation probe: +0.1 rad about world z (2 rows of 0.1)
        R0 = self.R
        self.send([[0, 0, 0, 0, 0, 0.2, g], [0, 0, 0, 0, 0, 0.2, g], [0, 0, 0, 0, 0, 0, g]])
        R1 = self.R
        cmd = np.array([0, 0, 0.2])
        dw = geom.R_to_rotvec(R1 @ R0.T)
        db = geom.R_to_rotvec(R0.T @ R1)
        ew, eb = np.linalg.norm(dw - cmd), np.linalg.norm(db - cmd)
        self.rot_mode = "world" if ew <= eb else "body"
        self.log(f"rot probe: world-delta {np.round(dw, 3).tolist()} body-delta {np.round(db, 3).tolist()} -> {self.rot_mode}")
        return ok

    # ---------------- moves ----------------
    def move_to(self, target, yaw=None, grip=-1.0, tol=0.004, rtol=0.03, vmax=0.025, wmax=0.12,
                max_iter=12, max_rows=40, settle=0, label=""):
        """Closed-loop move to an absolute world position (and downward orientation with yaw)."""
        target = np.asarray(target, float)
        Rd = self.R_des(yaw) if yaw is not None else self.R_des(self.yaw())
        last_err = None
        stall = 0
        for it in range(max_iter):
            err = target - self.pos
            e = self.rot_err(Rd)
            en, rn = float(np.linalg.norm(err)), float(np.linalg.norm(e))
            if en < tol and rn < rtol:
                if settle:
                    self.send([[0, 0, 0, 0, 0, 0, grip]] * settle)
                return True
            n = int(np.clip(math.ceil(max(en / vmax, rn / wmax)), 1, max_rows))
            dp = self.Minv @ (err / n)
            dr = self.rot_cmd(e) / ROT_SCALE / n
            sc = max(1.0, np.max(np.abs(dp)), np.max(np.abs(dr)))
            row = list(dp / sc) + list(dr / sc) + [grip]
            self.send([row] * n)
            if last_err is not None and en > 0.6 * last_err and en > 0.01:
                stall += 1
                if stall >= 3:
                    self.log(f"move_to {label}: stalled at err {en:.3f}")
                    return False
            else:
                stall = 0
            last_err = en
        err = float(np.linalg.norm(target - self.pos))
        self.log(f"move_to {label}: max_iter, err {err:.4f} rot {float(np.linalg.norm(self.rot_err(Rd))):.3f}")
        return err < 3 * tol

    def linear(self, delta, grip, speed=0.01, yaw=None):
        """Open-loop straight line by `delta` (world m) at `speed` m/step, holding orientation."""
        delta = np.asarray(delta, float)
        n = max(1, int(math.ceil(np.linalg.norm(delta) / speed)))
        Rd = self.R_des(yaw if yaw is not None else self.yaw())
        start = self.pos.copy()
        rows = []
        for i in range(n):
            pass
        # send in chunks re-aiming at the straight line
        done = 0
        while done < n:
            k = min(5, n - done)
            goal = start + delta * (done + k) / n
            err = goal - self.pos
            dp = self.Minv @ (err / k)
            dr = self.rot_cmd(self.rot_err(Rd)) / ROT_SCALE / max(k, 3)
            sc = max(1.0, np.max(np.abs(dp)), np.max(np.abs(dr)))
            row = list(dp / sc) + list(dr / sc) + [grip]
            self.send([row] * k)
            done += k
        return self.pos - start

    def gripper(self, grip, n=10):
        self.send([[0, 0, 0, 0, 0, 0, grip]] * n)
        return self.grip_width
