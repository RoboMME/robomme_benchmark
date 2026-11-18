"""Robot wrapper: stepping, closed-loop Cartesian moves, gripper and orientation hold."""
import math
import time
import numpy as np

from io_utils import log, reply_json, find_key, find_state, truthy
from geom import rotvec_to_mat, mat_to_rotvec, rotz

POS_SCALE = 0.05
ROT_SCALE = 0.5
WALL_BUDGET_S = 900.0   # controller limit is 1200 s; leave margin for slow formal workers


def _safe(x):
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return float(max(-1.0, min(1.0, round(v, 5))))


class EpisodeOver(Exception):
    pass


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.state = None
        self.step_id = None
        self.ended = False
        self.complete = False
        self.remaining = None
        self.limit = None
        self.grip = -1.0
        self.R_ref = None
        self.rot_mode = "world"   # world | body | off
        self.rot_bad = 0
        self.last_reply = None
        self.last_step_result = None
        self.n_steps_sent = 0
        self.errors = 0
        self.t_last_step = time.time()
        self.deadline = time.time() + WALL_BUDGET_S

    # ------------------------------------------------------------ replies
    def absorb(self, result):
        j = reply_json(result)
        self.last_reply = result
        s = find_state(j)
        if s is not None:
            self.state = np.asarray(s, dtype=float)
        sid = find_key(j, "step_id")
        if isinstance(sid, int):
            self.step_id = sid
        ee = find_key(j, "episode_ended")
        if ee is not None and truthy(ee):
            self.ended = True
        tc = find_key(j, "task_complete")
        if tc is not None and truthy(tc):
            self.complete = True
        if find_key(j, "terminated") is True or find_key(j, "truncated") is True:
            self.ended = True
        lim = find_key(j, "episode_limits")
        if isinstance(lim, dict):
            for k in ("remaining_steps", "steps_remaining", "remaining"):
                if isinstance(lim.get(k), (int, float)):
                    self.remaining = int(lim[k])
            for k in ("max_steps", "step_limit", "max_episode_steps", "limit"):
                if isinstance(lim.get(k), (int, float)):
                    self.limit = int(lim[k])
            if self.remaining is None and self.limit is not None and self.step_id is not None:
                self.remaining = self.limit - self.step_id
        return j

    def observe(self):
        r = self.ctx.call("robomme_observe")
        self.absorb(r)
        return r

    def status(self):
        try:
            r = self.ctx.call("robomme_status")
            self.absorb(r)
            return reply_json(r)
        except Exception as e:
            log("robomme_status failed", err=str(e)[:300])
            return {}

    @property
    def pos(self):
        return self.state[:3].copy()

    @property
    def R(self):
        return rotvec_to_mat(self.state[3:6])

    @property
    def fingers(self):
        return float(self.state[6] + self.state[7])

    def yaw(self):
        R = self.R
        return math.atan2(R[1, 0], R[0, 0])

    # ------------------------------------------------------------ stepping
    def step(self, rows):
        if self.ended:
            raise EpisodeOver()
        if time.time() > self.deadline:
            log("wall-clock deadline reached; stopping control")
            self.ended = True
            raise EpisodeOver()
        rows = [[_safe(x) for x in r] for r in rows]
        if self.remaining is not None:
            if self.remaining <= 0:
                self.ended = True
                raise EpisodeOver()
            rows = rows[:max(1, self.remaining)]
        try:
            r = self.ctx.call("robomme_step", actions=rows)
        except Exception as e:
            self.errors += 1
            log("robomme_step error", err=str(e)[:400], n=len(rows))
            self.status()
            try:
                self.observe()
            except Exception as e2:
                log("observe after step error failed", err=str(e2)[:300])
            if self.errors > 25:
                raise EpisodeOver()
            return None
        self.n_steps_sent += len(rows)
        self.t_last_step = time.time()
        prev_sid = self.step_id
        self.last_step_result = r
        j = self.absorb(r)
        if find_state(j) is None and not self.ended:
            # no state in the motion reply: refresh it, never act on a stale pose
            try:
                self.observe()
            except Exception as e:
                log("observe after step failed", err=str(e)[:200])
        if not isinstance(find_key(j, "step_id"), int):
            self.step_id = (prev_sid or 0) + len(rows) if prev_sid is not None else self.n_steps_sent
        if self.ended:
            log("episode ended", step=self.step_id, complete=self.complete)
            raise EpisodeOver()
        return r

    def rot_cmd(self):
        """Rotation command (3,) holding R_ref."""
        if self.R_ref is None or self.rot_mode == "off":
            return np.zeros(3)
        Rc = self.R
        if self.rot_mode == "world":
            e = mat_to_rotvec(self.R_ref @ Rc.T)
        else:
            e = mat_to_rotvec(Rc.T @ self.R_ref)
        ang = float(np.linalg.norm(e))
        if ang < 0.01:
            return np.zeros(3)
        c = e / ROT_SCALE * 0.6
        n = float(np.linalg.norm(c))
        if n > 0.35:
            c = c / n * 0.35
        return c

    def rot_err(self):
        if self.R_ref is None:
            return 0.0
        return float(np.linalg.norm(mat_to_rotvec(self.R_ref @ self.R.T)))

    def _check_rot_progress(self, before, after, commanded):
        if not commanded or self.rot_mode == "off":
            return
        if after > before + 0.02:
            self.rot_bad += 1
        elif after < before - 0.005:
            self.rot_bad = max(0, self.rot_bad - 1)
        if self.rot_bad >= 3:
            old = self.rot_mode
            self.rot_mode = {"world": "body", "body": "off"}.get(self.rot_mode, "off")
            self.rot_bad = 0
            log("rotation control mode switch", old=old, new=self.rot_mode, err=after)

    def hold(self, n=1, grip=None):
        g = self.grip if grip is None else grip
        rows = [[0, 0, 0, 0, 0, 0, g]] * n
        return self.step(rows)

    def gripper(self, g, n=8, settle=True, min_steps=6):
        """Command gripper (-1 open, +1 close) for up to n steps, stopping when fingers settle."""
        self.grip = float(g)
        prev = self.fingers
        done = 0
        stable = 0
        while done < n:
            k = 2
            rc = self.rot_cmd()
            self.step([[0, 0, 0, rc[0], rc[1], rc[2], self.grip]] * k)
            done += k
            f = self.fingers
            if abs(f - prev) < 0.001:
                stable += 1
                if settle and stable >= 2 and done >= min_steps:
                    break
            else:
                stable = 0
            prev = f
        return self.fingers

    def move_to(self, target, vmax=0.4, tol=0.005, max_steps=120, stall_calls=6, rows_far=4, label=""):
        """Closed-loop move of the TCP to target (world xyz). Returns (ok, dist, steps)."""
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)) or self.state is None:
            log("move_to refused: invalid target/state", label=label, target=str(target))
            return False, float("inf"), 0
        used = 0
        hist = []
        vstep = vmax * POS_SCALE
        while used < max_steps:
            err = target - self.pos
            dist = float(np.linalg.norm(err))
            if dist < tol and (self.rot_mode == "off" or self.rot_err() < 0.05):
                return True, dist, used
            hist.append(dist)
            if len(hist) > stall_calls and hist[-stall_calls - 1] - dist < 0.002 and dist < 0.05:
                log("move stall", label=label, dist=dist, steps=used)
                return False, dist, used
            if len(hist) > 3 * stall_calls and hist[-3 * stall_calls - 1] - dist < 0.003:
                log("move stall (far)", label=label, dist=dist, steps=used)
                return False, dist, used
            rbefore = self.rot_err()
            rc = self.rot_cmd()
            commanded_rot = bool(np.any(rc != 0))
            if dist > 0.05:
                n = int(min(rows_far, max(1, (dist - 0.03) // vstep)))
                d = err / dist * vstep
                rows = [[d[0] / POS_SCALE, d[1] / POS_SCALE, d[2] / POS_SCALE, rc[0], rc[1], rc[2], self.grip]]
                rows += [[d[0] / POS_SCALE, d[1] / POS_SCALE, d[2] / POS_SCALE, 0, 0, 0, self.grip]] * (n - 1)
            else:
                d = err.copy()
                nd = float(np.linalg.norm(d))
                if nd > vstep:
                    d = d / nd * vstep
                rows = [[d[0] / POS_SCALE, d[1] / POS_SCALE, d[2] / POS_SCALE, rc[0], rc[1], rc[2], self.grip]]
            self.step(rows)
            used += len(rows)
            self._check_rot_progress(rbefore, self.rot_err(), commanded_rot)
        err = target - self.pos
        return False, float(np.linalg.norm(err)), used

    def set_yaw_ref(self, yaw_delta):
        """Rotate the reference orientation about world z by yaw_delta."""
        if self.R_ref is None:
            self.R_ref = self.R
        self.R_ref = rotz(yaw_delta) @ self.R_ref

    def init_ref(self):
        R = self.R
        down = np.array([0, 0, -1.0])
        dz = float(R[:, 2] @ down)
        if dz > 0.9 or dz <= 0.5:
            # dz <= 0.5: orientation convention unclear -> never reorient blindly
            self.R_ref = R
            if dz <= 0.5:
                log("tool z axis not downward; keeping current orientation", dz=dz)
        else:
            x = R[:, 0].copy()
            x[2] = 0
            if np.linalg.norm(x) < 1e-3:
                x = np.array([1.0, 0, 0])
            x /= np.linalg.norm(x)
            zc = down
            y = np.cross(zc, x)
            self.R_ref = np.stack([x, y, zc], axis=1)
            log("initial gripper not pointing down; reference set to down", dz=dz)
        return dz
