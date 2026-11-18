"""Robot I/O and motion primitives over the MCP tools (robomme_*)."""
import json
import math
import time
import traceback

import numpy as np

from vpo_util import (bfs_find, is_num_list, reply_meta, rotvec_to_mat, mat_to_rotvec, rotz)

POS_SCALE = 0.05   # m per unit action (documented)
ROT_SCALE = 0.5    # rad per unit action (documented)


class EpisodeOver(Exception):
    pass


class Robot:
    def __init__(self, ctx, log, wall_deadline):
        self.ctx = ctx
        self.log = log
        self.wall_deadline = wall_deadline
        self.state = None
        self.step_id = 0
        self.ended = False
        self.complete = False
        self.remaining = None
        self.max_steps = None
        self.steps_sent = 0
        self.errors = 0
        self.last_result = None
        self.last_meta = None
        self.grip = -1.0
        self.rot_sign = 1.0
        self.rot_ok = None

    # ------------------------------------------------------------------ I/O
    def _absorb(self, meta):
        if not isinstance(meta, dict):
            return
        st = None
        for cand in (meta.get("state"), (meta.get("observation") or {}).get("state") if isinstance(meta.get("observation"), dict) else None):
            if is_num_list(cand, 8):
                st = cand
                break
        if st is None:
            st = bfs_find(meta, "state", lambda v: is_num_list(v, 8))
        if st is not None:
            self.state = np.asarray(st, dtype=float)
        sid = meta.get("step_id")
        if not isinstance(sid, int):
            sid = bfs_find(meta, "step_id", lambda v: isinstance(v, int) and not isinstance(v, bool))
        if isinstance(sid, int):
            self.step_id = sid
        for k in ("episode_ended", "terminated", "done"):
            v = bfs_find(meta, k, lambda v: isinstance(v, bool))
            if v is True:
                self.ended = True
        v = bfs_find(meta, "task_complete", lambda v: isinstance(v, bool))
        if v is True:
            self.complete = True
        lim = bfs_find(meta, "episode_limits")
        if isinstance(lim, dict):
            for k, val in lim.items():
                kl = str(k).lower()
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    if "remain" in kl:
                        self.remaining = int(val)
                    elif ("max" in kl or "limit" in kl or "horizon" in kl) and "time" not in kl and "second" not in kl:
                        self.max_steps = int(val)
            if self.remaining is None and self.max_steps is not None:
                self.remaining = self.max_steps - self.step_id

    def call(self, tool, **kw):
        return self.ctx.call(tool, **kw)

    def observe(self):
        res = self.call("robomme_observe")
        meta = reply_meta(res)
        self._absorb(meta)
        self.last_result, self.last_meta = res, meta
        return res, meta

    def status(self):
        try:
            res = self.call("robomme_status")
            meta = reply_meta(res)
            self._absorb(meta)
            return meta
        except Exception as e:
            self.log.ev("status_error", err=str(e)[:300])
            return {}

    def check_time(self):
        if time.time() > self.wall_deadline:
            raise EpisodeOver("wall deadline")

    def step(self, rows):
        """Send rows (list of 7-lists). Raises EpisodeOver when the episode ends."""
        if self.ended:
            raise EpisodeOver("ended")
        self.check_time()
        clean = []
        for r in rows:
            rr = [float(max(-1.0, min(1.0, x))) if math.isfinite(float(x)) else 0.0 for x in r]
            clean.append(rr)
        if self.remaining is not None:
            if self.remaining <= 0:
                self.ended = True
                raise EpisodeOver("no steps left")
            clean = clean[:max(1, self.remaining)]
        try:
            res = self.call("robomme_step", actions=clean)
        except Exception as e:
            self.errors += 1
            self.log.ev("step_error", err=str(e)[:500], n=len(clean))
            # Physics may have advanced: re-read the true state, never blindly repeat.
            try:
                self.observe()
            except Exception as e2:
                self.log.ev("observe_after_error_failed", err=str(e2)[:300])
            self.status()
            if self.errors > 8:
                raise EpisodeOver("too many tool errors")
            if self.ended:
                raise EpisodeOver("ended")
            return None
        self.steps_sent += len(clean)
        meta = reply_meta(res)
        prev = self.state
        self.state = None
        self._absorb(meta)
        self.last_result, self.last_meta = res, meta
        if self.state is None:
            try:
                self.observe()
            except Exception:
                self.state = prev
        if self.remaining is not None and not isinstance(bfs_find(meta, "episode_limits"), dict):
            self.remaining -= len(clean)
        if self.ended:
            raise EpisodeOver("ended (task_complete=%s)" % self.complete)
        return meta

    # ------------------------------------------------------------ geometry
    @property
    def pos(self):
        return self.state[:3].copy()

    @property
    def R(self):
        return rotvec_to_mat(self.state[3:6])

    @property
    def width(self):
        return float(self.state[6] + self.state[7])

    def yaw(self):
        R = self.R
        return math.atan2(R[1, 0], R[0, 0])

    # --------------------------------------------------------- primitives
    def hold(self, n, grip=None):
        g = self.grip if grip is None else grip
        return self.step([[0, 0, 0, 0, 0, 0, g]] * n)

    def set_gripper(self, close, n=12, min_n=6):
        """Close (+1) or open (-1); wait until the finger sum settles."""
        self.grip = 1.0 if close else -1.0
        sent = 0
        last = self.width
        while sent < n + 12:
            k = min_n if sent == 0 else 3
            self.hold(k)
            sent += k
            w = self.width
            if sent >= n and abs(w - last) < 0.0008:
                break
            last = w
        return self.width

    def move_to(self, target, speed=0.02, tol=0.004, max_steps=160, chunk=5, name="move", floor=None):
        """Closed-loop straight-line move of the TCP to `target` (world m)."""
        target = np.asarray(target, dtype=float).copy()
        if floor is not None and target[2] < floor:
            target[2] = floor
        used = 0
        best = 1e9
        stall = 0
        t0 = self.step_id
        start = self.pos
        while used < max_steps:
            err = target - self.pos
            d = float(np.linalg.norm(err))
            if d < tol:
                self.log.ev("move_done", name=name, d=round(d, 4), steps=used, pos=_r(self.pos))
                return True
            if d < best - 0.0015:
                best = d
                stall = 0
            else:
                stall += 1
            if stall >= 6:
                self.log.ev("move_stall", name=name, d=round(d, 4), steps=used, pos=_r(self.pos), target=_r(target))
                return d < 3 * tol
            if d > 2.5 * speed:
                n = max(1, min(chunk, int(d / speed) - 1))
                per = speed
            else:
                n = 1
                per = min(speed, 0.8 * d) if d > 0.006 else d
            dvec = err / d * per / POS_SCALE
            row = [dvec[0], dvec[1], dvec[2], 0, 0, 0, self.grip]
            self.step([row] * n)
            used += n
        d = float(np.linalg.norm(target - self.pos))
        self.log.ev("move_timeout", name=name, d=round(d, 4), steps=used, pos=_r(self.pos), target=_r(target))
        return d < 3 * tol

    def rotate_yaw(self, dyaw, hold_pos=None, max_steps=40):
        """Rotate the gripper about world z by dyaw (rad) with sign self-check."""
        if abs(dyaw) < math.radians(3):
            return True
        R_t = rotz(dyaw) @ self.R
        hold_pos = self.pos if hold_pos is None else np.asarray(hold_pos, dtype=float)
        e0 = float(np.linalg.norm(mat_to_rotvec(R_t @ self.R.T)))
        used, flips = 0, 0
        best = e0
        since_best = 0
        while used < max_steps:
            e = mat_to_rotvec(R_t @ self.R.T)
            en = float(np.linalg.norm(e))
            if en < math.radians(2.5):
                self.log.ev("yaw_done", err_deg=round(math.degrees(en), 2), steps=used, sign=self.rot_sign)
                self.rot_ok = True
                return True
            if en < best - math.radians(0.5):
                best, since_best = en, 0
            else:
                since_best += 1
            if since_best >= 3:
                if en > e0 * 0.9 and flips == 0:
                    self.rot_sign *= -1.0
                    flips += 1
                    since_best = 0
                    best = en
                    e0 = en
                    self.log.ev("yaw_flip_sign", err_deg=round(math.degrees(en), 2))
                else:
                    self.log.ev("yaw_stall", err_deg=round(math.degrees(en), 2), steps=used)
                    self.rot_ok = en < math.radians(10)
                    return False
            rv = self.rot_sign * np.clip(0.6 * e / ROT_SCALE, -0.3, 0.3)
            perr = (hold_pos - self.pos) / POS_SCALE
            perr = np.clip(0.8 * perr, -0.3, 0.3)
            n = 2 if en > math.radians(12) else 1
            self.step([[perr[0], perr[1], perr[2], rv[0], rv[1], rv[2], self.grip]] * n)
            used += n
        self.log.ev("yaw_timeout", steps=used)
        return False


def _r(v, k=4):
    return [round(float(x), k) for x in v]
