"""Closed-loop Cartesian motion with relative 7-D rows."""
import math
import numpy as np

from geom import mat_to_rotvec, rotz
from mcpio import log

POS_SCALE = 0.05
ROT_SCALE = 0.5


class Motion:
    def __init__(self, robot):
        self.rb = robot
        self.R_ref = None       # orientation to hold (gripper down, initial yaw)
        self.yaw_off = 0.0      # current commanded yaw offset (rad) about world z relative to R_ref
        self.grip = -1.0
        self.eff_hist = []
        self.eff = 0.8          # running estimate of achieved / commanded translation per step
        self.rot_body = False   # flip to body-frame rotation commands if world-frame diverges
        self.rot_strikes = 0
        self.rot_good = 0

    def init_ref(self):
        self.R_ref = self.rb.R().copy()
        self.yaw_off = 0.0

    def R_goal(self, yaw=None):
        y = self.yaw_off if yaw is None else yaw
        return rotz(y) @ self.R_ref

    def rows_toward(self, goal, Rg, k, vmax, rmax=0.12):
        p = self.rb.tcp()
        R = self.rb.R()
        e = np.asarray(goal, float) - p
        dist = float(np.linalg.norm(e))
        eff = min(1.0, max(0.4, self.eff))
        # commanded per-step offset; compensate for partial tracking (eff) without overshooting
        step = min(vmax / eff, 0.048, 0.9 * dist / (k * eff)) if dist > 1e-9 else 0.0
        d = e / dist * step if dist > 1e-9 else np.zeros(3)
        rv = mat_to_rotvec(Rg @ R.T)
        ang = float(np.linalg.norm(rv))
        rstep = min(rmax, ang / k)
        r = rv / ang * rstep if ang > 1e-9 else np.zeros(3)
        if self.rot_body:
            r = R.T @ r
        row = list(d / POS_SCALE) + list(r / ROT_SCALE) + [self.grip]
        return [row] * k, dist, ang

    def move(self, goal, yaw=None, tol=0.004, rot_tol=0.035, vmax=0.02, max_steps=120, chunk=6, label='', learn=True):
        """Servo the TCP to goal (world m) with the held orientation. Returns final distance."""
        Rg = self.R_goal(yaw)
        used = 0
        stall = 0
        prev = None
        dist = ang = None
        prev_ang = None
        last_rot_cmd = 0.0
        while used < max_steps:
            self._refresh()
            p = self.rb.tcp()
            dist = float(np.linalg.norm(np.asarray(goal) - p))
            ang = float(np.linalg.norm(mat_to_rotvec(Rg @ self.rb.R().T)))
            # rotation-frame sanity check
            if prev_ang is not None and last_rot_cmd > 0.03:
                if ang > prev_ang + 0.3 * last_rot_cmd:
                    self.rot_strikes += 1
                    log('rotation diverging (%s frame): %.3f -> %.3f strikes %d' % ('body' if self.rot_body else 'world', prev_ang, ang, self.rot_strikes))
                    if self.rot_strikes >= 2 and self.rot_good < 3:
                        self.rot_body = not self.rot_body
                        self.rot_strikes = 0
                        log('SWITCH rotation command frame ->', 'body' if self.rot_body else 'world')
                elif ang < prev_ang - 0.3 * last_rot_cmd:
                    self.rot_good += 1
                    self.rot_strikes = 0
            if dist < tol and ang < rot_tol:
                break
            if prev is not None:
                prog_d = prev - dist
                prog_a = (prev_ang - ang) if prev_ang is not None else 0.0
                if prog_d < 0.0015 and prog_a < 0.01 and (dist >= tol or ang >= rot_tol):
                    stall += 1
                else:
                    stall = 0
                if stall >= 4:
                    log('move stall', label, 'dist %.4f ang %.3f' % (dist, ang))
                    break
            prev = dist
            prev_ang = ang
            if dist > 0.12 and chunk >= 6:
                k = 10          # long free-space leg: fewer round trips
            elif dist > 0.04:
                k = chunk
            elif dist > 0.012:
                k = 3
            else:
                k = 2
            k = max(1, min(k, max_steps - used))
            rows, _, _ = self.rows_toward(goal, Rg, k, vmax)
            last_rot_cmd = float(np.linalg.norm(np.array(rows[0][3:6]) * ROT_SCALE)) * k
            cmd = np.array(rows[0][:3]) * POS_SCALE * k
            self.rb.step(rows)
            used += k
            # learn tracking efficiency from free-space moves only (large commands, not contact moves)
            if learn and np.linalg.norm(cmd) > 0.012:
                try:
                    self._refresh()
                    got = float((self.rb.tcp() - p) @ cmd) / float(cmd @ cmd)
                    if 0.05 < got < 1.6:
                        self.eff_hist.append(got)
                        self.eff_hist = self.eff_hist[-12:]
                        self.eff = float(np.median(self.eff_hist))
                except Exception:
                    pass
        self._refresh()
        p = self.rb.tcp()
        dist = float(np.linalg.norm(np.asarray(goal) - p))
        log('move %s done steps=%d dist=%.4f ang=%.3f eff=%.2f tcp=%s' % (label, used, dist, ang if ang is not None else -1, self.eff, np.round(p, 4).tolist()))
        return dist

    def hold(self, n, grip=None):
        if grip is not None:
            self.grip = grip
        Rg = self.R_goal()
        rows = []
        for _ in range(n):
            rows.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.grip])
        if rows:
            self.rb.step(rows)
        self._refresh()

    def set_gripper(self, g, n=8):
        self.grip = g
        self.hold(n)

    def _refresh(self):
        pass  # Robot.state() observes automatically when the last reply carried no state

    def descend_until(self, z_goal, xy, yaw=None, vmax=0.01, max_steps=60, label=''):
        """Move down to z_goal at xy; stop on stall (contact). Returns final z."""
        goal = np.array([xy[0], xy[1], z_goal])
        return self.move(goal, yaw=yaw, tol=0.003, vmax=vmax, max_steps=max_steps, chunk=3, label=label)
