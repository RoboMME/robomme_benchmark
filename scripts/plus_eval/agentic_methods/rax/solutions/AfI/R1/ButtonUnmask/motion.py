"""Robot wrapper: bounded relative-delta motion with step accounting and terminal detection."""
import math
import time

import numpy as np

from geom import R_to_axang, axang_to_R
from obs import find_key, parse_obs, reply_meta, walk

POS_SCALE = 0.05
ROT_SCALE = 0.5


def parse_limit(meta):
    best = None
    for p, v in walk(meta):
        if not p or not isinstance(p[-1], str) or not isinstance(v, int) or isinstance(v, bool):
            continue
        k = p[-1].lower()
        if k in ("max_steps", "step_limit", "max_episode_steps", "horizon", "episode_step_limit",
                 "max_control_steps", "limit", "steps_limit", "total_steps"):
            if 20 <= v <= 5000:
                best = v if best is None else min(best, v)
    return best


class EpisodeOver(Exception):
    pass


class Robot:
    def __init__(self, ctx, log, limit=None):
        self.ctx, self.log = ctx, log
        self.limit = limit
        self.steps = 0
        self.ended = False
        self.complete = False
        self.obs = None
        self.state = None
        self.grip = -1.0
        self.calls = 0
        self.last_step_wall = time.time()
        self.errors = 0
        self.rot_frame = "world"
        self.xy_rot = 0.0
        self.z_sign = 1.0
        self._mv_samples = []
        self._rot_hist = []

    # ------------------------------------------------------------ io
    def observe(self, images=True):
        r = self.ctx.call("robomme_observe")
        o = parse_obs(r, want_images=images)
        if o.state is not None:
            self.state = o.state
        if isinstance(o.step_id, int):
            self.steps = max(self.steps, o.step_id)
        if o.ended:
            self.ended = True
        if o.complete:
            self.complete = True
        if images:
            self.obs = o
        return o

    def remaining(self):
        if self.limit is None:
            return 10 ** 6
        return self.limit - self.steps

    def send(self, rows):
        if self.ended:
            raise EpisodeOver()
        clean = []
        for r in rows:
            r = [float(x) if np.isfinite(x) else 0.0 for x in r]
            r = [max(-1.0, min(1.0, x)) for x in r]
            clean.append(r)
        if not clean:
            return
        rem = self.remaining()
        if rem <= 0:
            self.ended = True
            raise EpisodeOver()
        clean = clean[:max(1, min(100, rem))]
        try:
            r = self.ctx.call("robomme_step", actions=clean)
            self.calls += 1
        except Exception as e:
            self.errors += 1
            self.log(f"robomme_step error: {str(e)[:300]}")
            self._recheck()
            if self.errors > 8:
                raise
            return
        self.errors = 0
        self.last_step_wall = time.time()
        meta = reply_meta(r)
        o = parse_obs(r, want_images=False)
        if isinstance(o.step_id, int) and o.step_id >= self.steps:
            self.steps = o.step_id
        else:
            self.steps += len(clean)
        if o.ended:
            self.ended = True
        if o.complete:
            self.complete = True
        # always refresh from a fresh observation: the step reply may hold per-step records
        try:
            self.observe(images=False)
        except Exception as e:
            self.log(f"observe after step failed: {str(e)[:200]}")
            if o.state is not None:
                self.state = o.state
        if self.ended:
            self.log(f"episode ended at step {self.steps} complete={self.complete}")
            raise EpisodeOver()

    def _recheck(self):
        try:
            r = self.ctx.call("robomme_status")
            meta = reply_meta(r)
            for p, v in walk(meta):
                if p and isinstance(p[-1], str) and p[-1].lower() in ("episode_ended", "ended", "done") and v is True:
                    self.ended = True
            self.observe(images=False)
        except Exception as e:
            self.log(f"recheck failed: {str(e)[:200]}")

    # ------------------------------------------------------------ state
    def pos(self):
        return np.array(self.state[:3], float)

    def R(self):
        return axang_to_R(self.state[3:6])

    def opening(self):
        return float(self.state[6] + self.state[7])

    def finger_axis(self):
        return self.R()[:, 1]

    # ------------------------------------------------------------ motion
    def rot_err(self, Rd):
        if Rd is None:
            return np.zeros(3)
        return R_to_axang(Rd @ self.R().T)

    def hold(self, n, grip=None, dz=0.0):
        if grip is not None:
            self.grip = grip
        rows = [[0, 0, dz / POS_SCALE, 0, 0, 0, self.grip]] * n
        self.send(rows)

    def move_to(self, target, Rd=None, grip=None, vmax=0.025, tol=0.004, rtol=0.03, max_steps=60, chunk=3,
                hook=None, rmax=0.12, label=""):
        if grip is not None:
            self.grip = grip
        target = np.asarray(target, float)
        used = 0
        hist = []
        while used < max_steps:
            err = target - self.pos()
            dist = float(np.linalg.norm(err))
            rerr = self.rot_err(Rd)
            rang = float(np.linalg.norm(rerr))
            if dist < tol and rang < rtol:
                break
            hist.append(dist + 0.05 * rang)
            if len(hist) > 6 and hist[-6] - hist[-1] < 0.002:
                self.log(f"move_to[{label}] stalled dist={dist:.4f} rot={rang:.3f}")
                break
            if dist > 2.5 * vmax:
                k = max(1, min(chunk, int(dist / vmax) - 1, max_steps - used))
                d = err / dist * vmax
            else:
                k = 1
                d = err if dist <= vmax else err / dist * vmax
            nrot = max(k, int(math.ceil(rang / rmax)))
            rd = rerr / nrot if rang > 1e-6 else np.zeros(3)
            if self.rot_frame == "body":
                rd = self.R().T @ rd
            c, s_ = math.cos(self.xy_rot), math.sin(self.xy_rot)
            dc = np.array([c * d[0] - s_ * d[1], s_ * d[0] + c * d[1], self.z_sign * d[2]])
            row = [dc[0] / POS_SCALE, dc[1] / POS_SCALE, dc[2] / POS_SCALE,
                   rd[0] / ROT_SCALE, rd[1] / ROT_SCALE, rd[2] / ROT_SCALE, self.grip]
            p_before = self.pos()
            self.send([row] * k)
            used += k
            self._check_frames(d * k, self.pos() - p_before, rang, float(np.linalg.norm(self.rot_err(Rd))))
            if hook is not None:
                hook()
        return float(np.linalg.norm(target - self.pos()))

    def _check_frames(self, cmd, ach, r_before, r_after):
        """Guard against a mis-assumed action frame (documented: world frame)."""
        try:
            cxy, axy = cmd[:2], ach[:2]
            nc, na = np.linalg.norm(cxy), np.linalg.norm(axy)
            if nc > 0.015 and na > 0.4 * nc:
                ang = math.atan2(cxy[0] * axy[1] - cxy[1] * axy[0], float(cxy @ axy))
                self._mv_samples.append(ang)
                last = self._mv_samples[-3:]
                if len(last) == 3 and all(abs(a) > math.radians(30) for a in last) and np.ptp(last) < math.radians(25):
                    self.xy_rot -= float(np.mean(last))
                    self.log(f"FRAME GUARD: xy action frame rotated by {math.degrees(np.mean(last)):.1f} deg; compensating")
                    self._mv_samples = []
            if abs(cmd[2]) > 0.015 and abs(ach[2]) > 0.4 * abs(cmd[2]) and np.sign(cmd[2]) != np.sign(ach[2]):
                self._zflip = getattr(self, "_zflip", 0) + 1
                if self._zflip >= 2:
                    self.z_sign *= -1
                    self._zflip = 0
                    self.log("FRAME GUARD: z action sign flipped")
            if r_before > 0.08:
                self._rot_hist.append(r_after - r_before)
                h = self._rot_hist[-3:]
                if len(h) == 3 and all(x > 0.005 for x in h):
                    self.rot_frame = "body" if self.rot_frame == "world" else "world"
                    self._rot_hist = []
                    self.log(f"FRAME GUARD: rotation error growing; switching rotation frame to {self.rot_frame}")
            elif r_before <= 0.08:
                self._rot_hist = []
        except Exception:
            pass
