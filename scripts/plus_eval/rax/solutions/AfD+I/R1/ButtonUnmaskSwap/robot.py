"""Robot I/O wrapper and motion primitives."""
import math
import time
import traceback
import numpy as np

from geom import rotvec_to_R, R_to_rotvec, Rz
from obsio import parse, log, remaining_steps, max_steps

POS_SCALE = 0.05
ROT_SCALE = 0.5


class EpisodeOver(Exception):
    pass


class Robot:
    def __init__(self, ctx, world, t_start):
        self.ctx = ctx
        self.world = world
        self.r = None
        self.t_start = t_start
        self.ncalls = 0
        self.lat = []
        self.step_id = 0
        self.ended = False
        self.complete = None
        self.grip = -1.0
        self.track_enabled = False
        self.remaining = None
        self.max_steps = None
        self.on_reply = None
        self.errors = 0
        self.wall_limit = None  # seconds since t_start; send() stops the episode cleanly past it
        # action-convention self check (documented: world-frame deltas)
        self.pos_sign = 1.0
        self.rot_body = False
        self._pos_ev = []
        self._rot_ev = []

    # ------------------------------------------------------------ io
    def observe(self):
        res = self.ctx.call('robomme_observe')
        r = parse(res)
        self._ingest(r, res)
        return r

    def _ingest(self, r, res=None):
        self.r = r
        if r.step is not None:
            self.step_id = r.step
        if r.ended:
            self.ended = True
        if r.complete is not None:
            self.complete = r.complete
        rem = remaining_steps(r.meta)
        if rem is not None:
            self.remaining = rem
        ms = max_steps(r.meta)
        if ms is not None:
            self.max_steps = ms
        if self.on_reply:
            try:
                self.on_reply(r)
            except Exception as e:
                log('on_reply error', e, traceback.format_exc()[-800:])

    def steps_left(self):
        if self.remaining is not None:
            return self.remaining
        if self.max_steps is not None:
            return self.max_steps - self.step_id
        return 480 - self.step_id

    def send(self, rows):
        if self.ended:
            raise EpisodeOver()
        if self.wall_limit is not None and self.elapsed() > self.wall_limit:
            log('WALL LIMIT reached (%.0fs); stopping cleanly' % self.elapsed())
            self.ended = True
            raise EpisodeOver()
        rows = [[float(np.clip(x, -1, 1)) for x in row] for row in rows]
        prev_state = None if self.r is None or self.r.state is None else np.array(self.r.state, float)
        # apply convention corrections (identity unless the self-check found otherwise)
        if self.pos_sign < 0 or self.rot_body:
            Rb = self.R() if self.r is not None and self.r.state is not None else np.eye(3)
            out = []
            for row in rows:
                row = list(row)
                row[0:3] = [self.pos_sign * x for x in row[0:3]]
                if self.rot_body:
                    row[3:6] = list(Rb.T @ np.array(row[3:6]))
                out.append(row)
            rows = out
        t = time.time()
        try:
            res = self.ctx.call('robomme_step', actions=rows)
        except RuntimeError as e:
            self.errors += 1
            log('step error', str(e)[:500])
            # physics may have advanced; re-observe
            try:
                self.observe()
            except Exception as e2:
                log('observe after error failed', str(e2)[:300])
            if self.errors > 20:
                raise EpisodeOver()
            if self.ended:
                raise EpisodeOver()
            return self.r
        self.lat.append(time.time() - t)
        self.ncalls += 1
        r = parse(res)
        self._ingest(r, res)
        try:
            self._check_conventions(prev_state, rows)
        except Exception as e:
            log('convention check error', e)
        if self.ended:
            log('episode ended at step', self.step_id, 'complete=', self.complete)
            raise EpisodeOver()
        return r

    def _check_conventions(self, prev, rows):
        if prev is None or self.r is None or self.r.state is None or len(rows) != 1:
            return
        cur = np.array(self.r.state, float)
        row = np.array(rows[0], float)
        # translation: commanded (as sent, before our sign correction) vs measured
        cmd = row[0:3] * POS_SCALE * (self.pos_sign)  # intended world displacement
        dp = cur[:3] - prev[:3]
        if np.linalg.norm(cmd) > 0.006 and np.linalg.norm(dp) > 0.003:
            self._pos_ev.append(float(cmd @ dp / (np.linalg.norm(cmd) * np.linalg.norm(dp))))
            if len(self._pos_ev) in (4, 8) :
                m = float(np.median(self._pos_ev[-4:]))
                log('convention check: translation cos median %.2f (n=%d)' % (m, len(self._pos_ev)))
                if m < -0.5:
                    self.pos_sign *= -1
                    log('WARNING: translation sign flipped')
                    self._pos_ev = []
        # rotation: world-frame hypothesis vs body-frame hypothesis
        e_sent = row[3:6]
        if np.linalg.norm(e_sent) * ROT_SCALE > 0.03:
            R0 = rotvec_to_R(prev[3:6])
            R1 = rotvec_to_R(cur[3:6])
            dw = R_to_rotvec(R1 @ R0.T)
            if np.linalg.norm(dw) > 0.015:
                hw = e_sent  # world hypothesis (what the sent row means if world-frame)
                hb = R0 @ e_sent  # body hypothesis
                cw = float(hw @ dw / (np.linalg.norm(hw) * np.linalg.norm(dw)))
                cb = float(hb @ dw / (np.linalg.norm(hb) * np.linalg.norm(dw)))
                self._rot_ev.append((cw, cb))
                if len(self._rot_ev) in (4, 8):
                    a = np.median(np.array(self._rot_ev[-4:]), 0)
                    log('convention check: rotation cos world %.2f body %.2f (rot_body=%s)' % (a[0], a[1], self.rot_body))
                    # rows were already transformed if rot_body: in that case 'world' hypothesis of the
                    # sent row is the body-mode command; keep simple: switch only from default mode
                    if not self.rot_body and a[1] > 0.8 and a[0] < 0.5:
                        self.rot_body = True
                        log('WARNING: rotation deltas look body-frame; switching')
                        self._rot_ev = []

    def elapsed(self):
        return time.time() - self.t_start

    # ------------------------------------------------------------ state
    def pos(self):
        return np.array(self.r.state[:3], float)

    def R(self):
        return rotvec_to_R(self.r.state[3:6])

    def opening(self):
        return float(self.r.state[6] + self.r.state[7])

    # ------------------------------------------------------------ primitives
    def row_towards(self, p_t, R_t, grip, vmax=0.4, gain=0.8, rmax=0.3, rgain=0.7):
        p = self.pos()
        dp = (np.asarray(p_t) - p) * gain / POS_SCALE
        n = np.linalg.norm(dp)
        if n > vmax:
            dp = dp / n * vmax
        row = [dp[0], dp[1], dp[2], 0.0, 0.0, 0.0, grip]
        if R_t is not None:
            e = R_to_rotvec(R_t @ self.R().T) * rgain / ROT_SCALE
            ne = np.linalg.norm(e)
            if ne > rmax:
                e = e / ne * rmax
            row[3:6] = list(e)
        return row

    def hold(self, n=1, grip=None):
        g = self.grip if grip is None else grip
        self.grip = g
        return self.send([[0, 0, 0, 0, 0, 0, g]] * n)

    def move(self, p_t, R_t=None, grip=None, tol=0.006, rtol=0.06, max_steps=60, vmax=0.4,
             target_fn=None, stop_fn=None, label=''):
        """closed-loop move; target_fn() may update (p_t, R_t) each step. returns ok"""
        g = self.grip if grip is None else grip
        self.grip = g
        best = 1e9
        stall = 0
        for k in range(max_steps):
            if target_fn is not None:
                try:
                    upd = target_fn()
                    if upd is not None:
                        p_t, R_t2 = upd
                        if R_t2 is not None:
                            R_t = R_t2
                except Exception as e:
                    log('target_fn error', e)
            err = float(np.linalg.norm(np.asarray(p_t) - self.pos()))
            rerr = float(np.linalg.norm(R_to_rotvec(R_t @ self.R().T))) if R_t is not None else 0.0
            if stop_fn is not None and stop_fn():
                return True
            if err < tol and rerr < rtol:
                return True
            if err < best - 0.001:
                best = err
                stall = 0
            else:
                stall += 1
            if stall > 12 and rerr < rtol:
                log('move %s stalled err=%.4f' % (label, err))
                return False
            row = self.row_towards(p_t, R_t, g, vmax=vmax)
            self.send([row])
        log('move %s timeout err=%.4f rerr=%.3f' % (label, err, rerr))
        return False

    def gripper(self, g, n=6, R_t=None, p_t=None):
        self.grip = g
        p_t = self.pos() if p_t is None else p_t
        prev = None
        for k in range(n):
            row = self.row_towards(p_t, R_t, g, vmax=0.1)
            self.send([row])
            o = self.opening()
            if prev is not None and abs(o - prev) < 0.0004 and k >= 2:
                break
            prev = o
        return self.opening()
