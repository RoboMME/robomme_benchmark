"""Robot I/O wrapper and motion primitives over the MCP context."""
import math
import time
import traceback

import numpy as np

import io_util as U
from geom import rotvec_to_R, rot_err_world, R_to_rotvec

POS_SCALE = 0.05   # m per unit action
ROT_SCALE = 0.5    # rad per unit action


class EpisodeOver(Exception):
    pass


class Robot:
    def __init__(self, ctx, deadline_s=1100.0):
        self.ctx = ctx
        self.t_start = time.time()
        self.deadline = deadline_s
        self.meta = None
        self.reply = None
        self.state = None
        self.step_id = None
        self.steps_sent = 0
        self.ended = False
        self.complete = False
        self.n_obs = 0
        self.n_step_calls = 0
        self.saved = set()
        self.step_limit = None
        self.errors = 0
        self.Tmap = None        # 3x3: commanded (normalised) -> world displacement correction (inverse)
        self.rot_local = False  # rotation deltas interpreted in the EE frame?
        self.gain = None
        self.grip_sign = 1.0     # multiply gripper commands (docs: -1 open, +1 close)

    # ---------- raw calls ----------
    def _save_first(self, name, reply):
        if name in self.saved:
            return
        self.saved.add(name)
        U.save_json("raw_%s.json" % name, U.strip_blobs(reply))

    def elapsed(self):
        return time.time() - self.t_start

    def _update_flags(self, meta):
        if U.task_complete(meta):
            self.complete = True
        if U.episode_ended(meta):
            self.ended = True

    def observe(self):
        rep = self.ctx.call("robomme_observe")
        self.n_obs += 1
        self._save_first("observe", rep)
        meta = U.reply_meta(rep)
        self.reply, self.meta = rep, meta
        st = U.get_state(meta)
        if st is not None:
            self.state = np.array(st, dtype=float)
        sid = U.get_step_id(meta)
        if sid is not None:
            self.step_id = sid
        self._update_flags(meta)
        return rep, meta

    def status(self):
        try:
            rep = self.ctx.call("robomme_status")
            self._save_first("status", rep)
            meta = U.reply_meta(rep)
            self._update_flags(meta)
            lim = self.parse_limit(meta)
            if lim:
                self.step_limit = lim
            return meta
        except Exception as e:
            U.log("status failed", e)
            return {}

    @staticmethod
    def parse_limit(meta):
        cands = []
        for p, k, v in U.walk(meta):
            if isinstance(v, int) and not isinstance(v, bool):
                kl = str(k).lower()
                pl = " ".join(str(x) for x in p).lower()
                if ("limit" in kl or "max" in kl or "horizon" in kl) and "step" in (kl + pl) and 10 <= v <= 100000:
                    cands.append(v)
        return max(cands) if cands else None

    def remaining_steps(self):
        if self.step_limit and self.step_id is not None:
            return self.step_limit - self.step_id
        return None

    def step(self, rows):
        if self.ended:
            raise EpisodeOver()
        rows = [[float(min(1.0, max(-1.0, x))) for x in r[:6]] + [float(self.grip_sign * (1.0 if r[6] > 0 else -1.0))]
                for r in rows]
        if not rows:
            return
        try:
            rep = self.ctx.call("robomme_step", actions=rows)
        except Exception as e:
            self.errors += 1
            U.log("step error:", str(e)[:400])
            # do not retry blindly: check the episode
            try:
                self.observe()
            except Exception as e2:
                U.log("observe after step error failed", str(e2)[:200])
            st = self.status()
            if self.ended or self.errors > 6:
                raise EpisodeOver()
            time.sleep(0.5)
            return
        self.errors = 0  # consecutive-error counter
        self.n_step_calls += 1
        self.steps_sent += len(rows)
        self._save_first("step", rep)
        meta = U.reply_meta(rep)
        self._update_flags(meta)
        st = U.get_state(meta)
        sid = U.get_step_id(meta)
        if sid is not None:
            self.step_id = sid
        lim = self.parse_limit(meta)
        if lim:
            self.step_limit = lim
        # authoritative current state from a fresh observe (cheap, no physics)
        if self.ended:
            raise EpisodeOver()
        try:
            self.observe()
        except Exception as e:
            U.log("observe failed", str(e)[:200])
            if st is not None:
                self.state = np.array(st, dtype=float)
        if self.ended:
            raise EpisodeOver()

    # ---------- state helpers ----------
    def pos(self):
        return self.state[:3].copy()

    def R(self):
        return rotvec_to_R(self.state[3:6])

    def width(self):
        return float(self.state[6] + self.state[7])

    # ---------- primitives ----------
    def move_to(self, target, R_goal, grip, tol=0.004, rot_tol=0.03, vmax=0.012, wmax=0.12,
                chunk=8, max_rows=200, label=""):
        """Closed-loop Cartesian move. Returns (ok, final_dist)."""
        target = np.asarray(target, dtype=float)
        rows_used = 0
        stall = 0
        last_d = None
        while True:
            p = self.pos()
            e = target - p
            d = float(np.linalg.norm(e))
            er = rot_err_world(R_goal, self.R()) if R_goal is not None else np.zeros(3)
            dr = float(np.linalg.norm(er))
            if d < tol and dr < rot_tol:
                return True, d
            if rows_used >= max_rows:
                U.log("move_to %s: row budget exhausted d=%.4f dr=%.3f" % (label, d, dr))
                return False, d
            if self.elapsed() > self.deadline:
                U.log("move_to %s: wall deadline" % label)
                return False, d
            v = vmax if d > 0.03 else max(0.004, min(vmax, 0.006))
            n = max(1, int(math.ceil(d / v)), int(math.ceil(dr / wmax)))
            k = min(n, chunk if d > 0.03 else 3)
            dp = e / n
            drot = er / n if dr >= rot_tol * 0.5 else np.zeros(3)
            cmd_p = dp / POS_SCALE
            if self.Tmap is not None:
                cmd_p = self.Tmap @ cmd_p
            cmd_r = drot / ROT_SCALE
            if self.rot_local:
                cmd_r = self.R().T @ cmd_r
            row = list(cmd_p) + list(cmd_r) + [grip]
            self.step([row] * k)
            rows_used += k
            nd = float(np.linalg.norm(target - self.pos()))
            if last_d is not None or True:
                expected = min(d, k * float(np.linalg.norm(dp)))
                if d - nd < 0.15 * expected and dr < rot_tol:
                    stall += 1
                else:
                    stall = 0
            last_d = nd
            if stall >= 4:
                U.log("move_to %s: stalled at d=%.4f" % (label, nd))
                return False, nd

    def probe(self, grip=-1):
        """Measure how commanded deltas move the measured TCP (frame/sign/gain check)."""
        p0, R0 = self.pos(), self.R()
        M = np.zeros((3, 3))
        a = 0.3
        nrow = 2
        for ax in (2, 0, 1):
            cmd = np.zeros(3)
            cmd[ax] = a
            b = self.pos()
            self.step([list(cmd) + [0, 0, 0, grip]] * nrow + [[0, 0, 0, 0, 0, 0, grip]])
            M[:, ax] = (self.pos() - b) / (a * POS_SCALE * nrow)
            self.step([list(-cmd) + [0, 0, 0, grip]] * nrow + [[0, 0, 0, 0, 0, 0, grip]])
        g = float(np.mean(np.linalg.norm(M, axis=0)))
        U.log("probe translation map M=\n%s gain %.2f" % (np.round(M, 3), g))
        self.gain = g
        if g > 0.05:
            Mn = M / g
            off = np.abs(Mn - np.eye(3)).max()
            if off > 0.5 and abs(np.linalg.det(Mn)) > 0.3:
                self.Tmap = np.linalg.inv(Mn)
                U.log("probe: action frame differs from world -> using inverse map", np.round(self.Tmap, 3))
        # rotation: +0.1 rad about world z
        Rb = self.R()
        wcmd = np.array([0, 0, 0.2])
        self.step([[0, 0, 0] + list(wcmd) + [grip]] * nrow + [[0, 0, 0, 0, 0, 0, grip]])
        w = R_to_rotvec(self.R() @ Rb.T)
        cw = wcmd * ROT_SCALE * nrow
        cl = Rb @ cw
        def cos(x, y):
            nx, ny = np.linalg.norm(x), np.linalg.norm(y)
            return float(x @ y / (nx * ny)) if nx > 1e-6 and ny > 1e-6 else 0.0
        s_w, s_l = cos(w, cw), cos(w, cl)
        U.log("probe rotation: measured %s world-cos %.2f local-cos %.2f" % (np.round(w, 3), s_w, s_l))
        if s_l > s_w + 0.3 and s_l > 0.5:
            self.rot_local = True
            U.log("probe: rotation deltas behave as EE-frame -> rot_local")
        # return to start pose
        self.move_to(p0, R0, grip, tol=0.004, label="probe-return", max_rows=40)
        # gripper sign check
        w0 = self.width()
        self.step([[0, 0, 0, 0, 0, 0, 1]] * 5)
        w1 = self.width()
        self.step([[0, 0, 0, 0, 0, 0, -1]] * 6)
        w2 = self.width()
        U.log("probe gripper: widths %.4f -> close %.4f -> open %.4f" % (w0, w1, w2))
        if w1 > w0 + 0.005 and w2 < w1 - 0.005:
            self.grip_sign = -1.0
            U.log("probe: gripper sign flipped")
            self.step([[0, 0, 0, 0, 0, 0, -1]] * 6)

    def hold(self, grip, n, R_goal=None):
        rows = []
        er = rot_err_world(R_goal, self.R()) if R_goal is not None else np.zeros(3)
        if self.rot_local:
            er = self.R().T @ er
        for i in range(n):
            rows.append([0, 0, 0] + (list(er / ROT_SCALE / max(n, 1)) if i == 0 else [0, 0, 0]) + [grip])
        self.step(rows)

    def gripper(self, grip, n=10, settle_tol=0.0008, R_goal=None):
        """Open (-1) or close (+1); returns final width."""
        self.hold(grip, max(4, n // 2), R_goal)
        w_prev = self.width()
        for _ in range(4):
            self.hold(grip, 3, None)
            w = self.width()
            if abs(w - w_prev) < settle_tol:
                break
            w_prev = w
        return self.width()
