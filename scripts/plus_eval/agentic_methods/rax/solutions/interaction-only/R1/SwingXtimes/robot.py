"""Robot I/O: observe, step with sanitised rows, closed-loop moves, pixel queries."""
import math
import time

import numpy as np

import geom
from mcpio import call, parse_obs, find_key, log, describe

POS_SCALE = 0.05
ROT_SCALE = 0.5
MAX_ROW_TRANS = 0.6   # of the 0.05 m scale -> at most 3 cm commanded per tick


class EpisodeOver(Exception):
    pass


class Robot:
    def __init__(self, ctx, wall_limit_s=1080.0):
        self.ctx = ctx
        self.t0 = time.time()
        self.wall_limit = wall_limit_s
        self.obs = None
        self.ended = False
        self.complete = False
        self.grip_cmd = -1.0
        self.R_des = None
        self.rot_sign = 1.0       # world-frame rotation delta sign (verified on first use)
        self.rot_verified = False
        self.rot_broken = False
        self.steps_sent = 0
        self.calls = 0
        self.first_reply_logged = False
        self.open_sum = 0.08
        self.step_errors = 0
        self.err_total = 0
        self.gain = 1.0

    # ------------------------------------------------------------------ basic I/O
    def time_left(self):
        return self.wall_limit - (time.time() - self.t0)

    def _check(self, o):
        if o.ended:
            self.ended = True
        if o.complete:
            self.complete = True

    def observe(self, images=True):
        r = call(self.ctx, "robomme_observe")
        self.calls += 1
        o = parse_obs(r, want_images=images)
        if not self.first_reply_logged:
            log("observe reply structure:", describe(r))
            self.first_reply_logged = True
        if o.state is None and self.obs is not None:
            o.state = self.obs.state
        self.obs = o
        self._check(o)
        return o

    def step(self, rows):
        if self.ended or self.complete:
            raise EpisodeOver()
        if self.time_left() < 0:
            raise EpisodeOver("wall")
        clean = []
        for r in rows:
            a = np.nan_to_num(np.asarray(r, dtype=np.float64).ravel()[:7], nan=0.0)
            if a.size < 7:
                a = np.concatenate([a, np.zeros(7 - a.size)])
            a = np.clip(a, -1.0, 1.0)
            clean.append([float(round(x, 5)) for x in a])
        if not clean:
            return self.obs
        rem = self.obs.remaining if self.obs is not None else None
        if rem is not None:
            if rem <= 0:
                self.ended = True
                raise EpisodeOver("no steps left")
            clean = clean[:max(1, rem)]
        out = None
        for i in range(0, len(clean), 100):
            chunk = clean[i:i + 100]
            try:
                r = call(self.ctx, "robomme_step", actions=chunk)
            except Exception as e:
                # Do not retry: physics may have advanced. Re-read the state and let the
                # caller's closed loop decide the next action.
                self.calls += 1
                self.step_errors += 1
                log("robomme_step error (%d): %s" % (self.step_errors, str(e)[:300]))
                self.err_total += 1
                if self.step_errors >= 5 or self.err_total > 40:
                    raise EpisodeOver("too many step errors")
                o = self.observe(images=False)
                if o.ended or o.complete or o.state is None:
                    raise EpisodeOver("ended after step error")
                return o
            self.calls += 1
            self.step_errors = 0
            self.steps_sent += len(chunk)
            o = parse_obs(r, want_images=False)
            if o.state is None:
                o2 = self.observe(images=False)
                o.state = o2.state
            if o.remaining is None and self.obs is not None and self.obs.remaining is not None:
                o.remaining = self.obs.remaining - len(chunk)
            self.obs = o
            self._check(o)
            out = o
            if self.ended or self.complete:
                break
        return out

    # ------------------------------------------------------------------ orientation
    def rot_correction(self, n):
        """Per-row rotation delta that removes the orientation error over n rows."""
        if self.R_des is None or self.obs is None or not self.rot_verified:
            return np.zeros(3)
        R = geom.rotvec_to_R(self.obs.rotvec)
        err = geom.R_to_rotvec(self.R_des @ R.T)
        if np.linalg.norm(err) < math.radians(1.5):
            return np.zeros(3)
        return self.rot_sign * err / max(1, n) / ROT_SCALE

    def tilt_deg(self):
        R = geom.rotvec_to_R(self.obs.rotvec)
        return math.degrees(math.acos(max(-1, min(1, -R[2, 2]))))

    def rotate_yaw(self, dyaw, tol_deg=3.0):
        """Rotate the tool about world z by dyaw (rad), verified from the measured state."""
        R0 = geom.rotvec_to_R(self.obs.rotvec)
        return self.rotate_to(geom.Rz(dyaw) @ R0, tol_deg)

    def orient_err_deg(self, target):
        R = geom.rotvec_to_R(self.obs.rotvec)
        return math.degrees(np.linalg.norm(geom.R_to_rotvec(target @ R.T)))

    def rotate_to(self, target, tol_deg=3.0):
        """Closed-loop rotation to an absolute orientation; flips the delta sign once if the
        first attempt makes the error worse (the frame convention was never observed live)."""
        if self.rot_broken:
            log("rotation skipped: deltas did not behave as world-frame rotations earlier")
            return False
        bad = 0
        for attempt in range(6):
            R = geom.rotvec_to_R(self.obs.rotvec)
            err = geom.R_to_rotvec(target @ R.T)
            e = np.linalg.norm(err)
            if e < math.radians(tol_deg):
                break
            if bad >= 2:
                self.rot_broken = not self.rot_verified
                log("rotation not converging; giving up (rot_broken=%s)" % self.rot_broken)
                break
            n = max(2, int(math.ceil(e / 0.15)))
            row = list(np.zeros(3)) + list(self.rot_sign * err / n / ROT_SCALE) + [self.grip_cmd]
            self.step([row] * n + [[0, 0, 0, 0, 0, 0, self.grip_cmd]] * 2)
            R2 = geom.rotvec_to_R(self.obs.rotvec)
            e2 = np.linalg.norm(geom.R_to_rotvec(target @ R2.T))
            log("rotate attempt %d err %.1f -> %.1f deg (sign %+d)" % (attempt, math.degrees(e), math.degrees(e2), self.rot_sign))
            if e2 > e * 0.9:
                bad += 1
                if not self.rot_verified:
                    self.rot_sign = -self.rot_sign
                    log("rotation did not converge; flipping rotation sign")
            elif e2 < e * 0.7:
                self.rot_verified = True
                bad = 0
        R = geom.rotvec_to_R(self.obs.rotvec)
        final = math.degrees(np.linalg.norm(geom.R_to_rotvec(target @ R.T)))
        return final < 3 * tol_deg

    # ------------------------------------------------------------------ translation
    def move_to(self, target, tol=0.006, vmax=0.02, max_iter=7, settle=1, label=""):
        target = np.asarray(target, dtype=np.float64)
        e0 = None
        for it in range(max_iter):
            p = self.obs.pos
            err = target - p
            d = float(np.linalg.norm(err))
            if e0 is None:
                e0 = d
            if d < tol:
                return True, d
            n = int(min(100, max(1, math.ceil(d / vmax))))
            rot = self.rot_correction(n)
            g = self.gain
            tr = err / n / POS_SCALE / g
            nt = float(np.linalg.norm(tr))
            if nt > MAX_ROW_TRANS:            # keep per-tick targets moderate (held cube)
                tr = tr * (MAX_ROW_TRANS / nt)
            row = list(tr) + list(rot) + [self.grip_cmd]
            rows = [row] * n + [[0, 0, 0, 0, 0, 0, self.grip_cmd]] * settle
            prev = p.copy()
            self.step(rows)
            moved = float(np.linalg.norm(self.obs.pos - prev))
            if d > 0.02 and n >= 2 and nt <= MAX_ROW_TRANS and \
                    label not in ("grasp-z", "lower", "press", "touch-button"):
                along = float(np.dot(self.obs.pos - prev, err / d))
                g_true = max(0.05 * d, along) * g / d
                self.gain = float(np.clip(0.5 * g + 0.5 * g_true, 0.3, 1.25))
                if abs(self.gain - g) > 0.05:
                    log("motion gain estimate %.2f -> %.2f (%s)" % (g, self.gain, label))
            if it >= 2 and moved < 0.002:
                log("move %s stalled at %.3f m from target" % (label, float(np.linalg.norm(target - self.obs.pos))))
                break
        d = float(np.linalg.norm(target - self.obs.pos))
        return d < tol * 2.5, d

    def gripper(self, g, n=12):
        self.grip_cmd = float(g)
        rows = [[0, 0, 0, 0, 0, 0, self.grip_cmd]] * n
        self.step(rows)
        return self.obs.grip

    def hold(self, n=3):
        self.step([[0, 0, 0, 0, 0, 0, self.grip_cmd]] * n)

    # ------------------------------------------------------------------ pixel queries
    def query(self, sels, radius=0):
        """sels: list of (camera, u, v). Returns list of xyz or None."""
        out = []
        sid = self.obs.step_id if self.obs.step_id is not None else 0
        for i in range(0, len(sels), 32):
            chunk = sels[i:i + 32]
            r = call(self.ctx, "robomme_pixel_to_position", observation_step_id=int(sid),
                     selections=[{"camera": c, "u": float(u), "v": float(v)} for c, u, v in chunk],
                     depth_window_radius=int(radius))
            self.calls += 1
            out.extend(parse_points(r, len(chunk)))
        return out


def _xyz(v):
    try:
        a = np.array(v, dtype=np.float64).ravel()
        if a.size == 3 and np.all(np.isfinite(a)):
            return a
    except Exception:
        pass
    return None


def parse_points(reply, n):
    lst = None
    for key in ("points", "results", "selections", "positions", "world_points", "items", "_list"):
        v = find_key(reply.meta, key)
        if isinstance(v, list) and len(v) == n:
            lst = v
            break
    if lst is None:
        # any list of length n in the metadata
        stack = [reply.meta]
        while stack and lst is None:
            cur = stack.pop()
            if isinstance(cur, dict):
                for x in cur.values():
                    if isinstance(x, list) and len(x) == n and all(isinstance(y, (dict, list)) for y in x):
                        lst = x
                        break
                    stack.append(x)
    res = [None] * n
    if lst is None:
        return res
    for i, it in enumerate(lst):
        p = None
        if isinstance(it, dict):
            if "valid" in it and not it.get("valid"):
                res[i] = None
                continue
            for k in ("position", "world", "world_position", "point", "xyz", "position_m",
                      "world_xyz", "p", "world_point", "position_world"):
                if k in it:
                    p = _xyz(it[k])
                    if p is not None:
                        break
            if p is None:
                for k, x in it.items():
                    if k in ("u", "v"):
                        continue
                    p = _xyz(x)
                    if p is not None:
                        break
        else:
            p = _xyz(it)
        res[i] = p
    return res
