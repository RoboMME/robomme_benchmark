"""Closed-loop end-effector motion over the RoboMME MCP step tool."""
import json, math, time
import numpy as np
from perception import decode, axangle_to_R, R_to_axangle
from geom import wrap, yaw_of, R_down, R_goal

POS_SCALE = 0.05
ROT_SCALE = 0.5

class EpisodeEnded(Exception):
    pass

class Robot:
    def __init__(self, ctx, log=print, step_budget_reserve=5):
        self.ctx = ctx; self.log = log
        self.meta = None; self.imgs = None; self.grip = -1.0
        self.ended = False; self.task_complete = None
        self.reserve = step_budget_reserve
        self.n_calls = 0
        self.yaw_u = None
        self.pitch = 0.0
        self.t0 = time.time()

    # ---------- observation ----------
    def observe(self):
        res = self.ctx.call('robomme_observe')
        self.meta, self.imgs = decode(res)
        self._check_end(self.meta)
        self._upd_yaw()
        return self.meta, self.imgs

    def _upd_yaw(self):
        y = yaw_of(self.R)
        if self.yaw_u is None: self.yaw_u = y
        else: self.yaw_u = self.yaw_u + wrap(y - wrap(self.yaw_u))

    def _check_end(self, meta):
        tr = meta.get('transition') or {}
        for s in tr.get('steps', []) or []:
            if s.get('task_complete'):
                self.task_complete = True
            if s.get('episode_ended'):
                self.ended = True
        if tr.get('task_complete') or meta.get('task_complete'):
            self.task_complete = True
        if tr.get('episode_ended') or meta.get('episode_ended'):
            self.ended = True

    @property
    def state(self):
        return np.array(self.meta['state'], float)

    @property
    def pos(self):
        return self.state[:3]

    @property
    def R(self):
        return axangle_to_R(self.state[3:6])

    @property
    def remaining(self):
        lim = self.meta.get('episode_limits') or {}
        return lim.get('remaining_episode_steps', 10**6)

    # ---------- stepping ----------
    def step(self, rows):
        if self.ended:
            raise EpisodeEnded()
        rows = [[float(max(-1.0, min(1.0, x))) for x in r] for r in rows]
        if self.meta is not None:
            rem = self.remaining
            if rem - len(rows) < self.reserve:
                rows = rows[:max(0, rem - self.reserve)]
                if not rows:
                    raise EpisodeEnded('step budget reserve reached')
        out = None
        for i in range(0, len(rows), 100):
            res = self.ctx.call('robomme_step', actions=rows[i:i + 100])
            self.n_calls += 1
            self.meta, self.imgs = decode(res)
            self._check_end(self.meta)
            self._upd_yaw()
            out = self.meta
            if self.ended:
                self.log(f'[robot] episode ended task_complete={self.task_complete} step={self.meta.get("step_id")}')
                raise EpisodeEnded()
        return out

    def hold(self, n, grip=None):
        g = self.grip if grip is None else grip
        self.grip = g
        return self.step([[0, 0, 0, 0, 0, 0, g]] * n)

    def set_gripper(self, g, n=12):
        return self.hold(n, grip=g)

    def gripper_opening(self):
        s = self.state; return s[6] + s[7]

    # ---------- motion ----------
    def rot_err(self, R_target):
        return R_to_axangle(R_target @ self.R.T)

    def move(self, target, R_target=None, speed=0.6, rspeed=0.5, tol=0.004, rtol=0.03,
             max_iter=12, chunk_max=40, label=''):
        """Move TCP to world target (and orientation). speed: fraction of max per-step delta."""
        target = np.asarray(target, float)
        if self.meta is None: self.observe()
        last = None
        for it in range(max_iter):
            err = target - self.pos
            if last is not None and np.linalg.norm(err) > 0.8 * last and np.linalg.norm(err) > 0.01 and it >= 2:
                self.log(f'[move {label}] stalled err={np.linalg.norm(err):.4f} pos={np.round(self.pos,3)}')
                return False
            last = np.linalg.norm(err)
            rerr = self.rot_err(R_target) if R_target is not None else np.zeros(3)
            d, rd = np.linalg.norm(err), np.linalg.norm(rerr)
            if d < tol and rd < rtol:
                return True
            n_pos = d / (POS_SCALE * speed)
            n_rot = rd / (ROT_SCALE * rspeed)
            n = int(min(chunk_max, max(1, math.ceil(max(n_pos, n_rot)))))
            dp = err / (POS_SCALE * n)
            dr = rerr / (ROT_SCALE * n) if R_target is not None else np.zeros(3)
            if d < 0.02 and n <= 3:
                # final approach: a few gentle steps
                n = max(n, 2); dp = err / (POS_SCALE * n)
            rows = [list(dp) + list(dr) + [self.grip]] * n
            self.step(rows)
        err = np.linalg.norm(target - self.pos)
        self.log(f'[move {label}] not converged err={err:.4f}')
        return err < 3 * tol

    def goto(self, target, speed=0.6, tol=0.004, **kw):
        return self.move(target, R_goal(self.yaw_u, self.pitch), speed=speed, rspeed=kw.pop('rspeed', 0.2), tol=tol, **kw)

    def move_line(self, target, speed=0.3, R_target=None, label='', tol=0.004):
        """Straight-line motion with bounded per-step delta, correcting cross-track error."""
        target = np.asarray(target, float)
        for it in range(30):
            err = target - self.pos; d = np.linalg.norm(err)
            if d < tol: return True
            n = int(min(10, max(1, math.ceil(d / (POS_SCALE * speed)))))
            dp = err / (POS_SCALE * n)
            dr = (self.rot_err(R_target) / (ROT_SCALE * max(n, 4))) if R_target is not None else np.zeros(3)
            self.step([list(dp) + list(dr) + [self.grip]] * n)
        return False

def yaw_R(R, dyaw):
    c, s = math.cos(dyaw), math.sin(dyaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Rz @ R


def rotate_yaw(robot, target_u, rate=0.03, hold_pos=None, max_steps=400, label=''):
    """Rotate the down-pointing gripper about world z to the unwrapped yaw target_u at
    `rate` rad/step, holding position and keeping the tool pointing down."""
    R = robot
    p0 = np.array(R.pos if hold_pos is None else hold_pos, float)
    used = 0; stall = 0
    while used < max_steps:
        rem = target_u - R.yaw_u
        if abs(rem) < 0.015: return True
        n = int(min(10, max(1, math.ceil(abs(rem) / rate))))
        dyaw = math.copysign(min(rate, abs(rem) / n), rem)
        # tilt correction toward pointing down at the current yaw
        rerr = R_to_axangle(R_goal(R.yaw_u, R.pitch) @ R.R.T)
        dp = (p0 - R.pos) / (POS_SCALE * max(n, 3))
        row = list(np.clip(dp, -0.3, 0.3)) + [rerr[0] / (ROT_SCALE * max(n, 3)), rerr[1] / (ROT_SCALE * max(n, 3)),
                                               dyaw / ROT_SCALE, R.grip]
        before = R.yaw_u
        R.step([row] * n); used += n
        prog = (R.yaw_u - before) / (dyaw * n) if abs(dyaw * n) > 1e-6 else 1
        if prog < 0.3:
            stall += 1
            if stall >= 3:
                R.log(f'[rotate {label}] stalled at yaw {math.degrees(R.yaw_u):.1f} target {math.degrees(target_u):.1f}')
                return False
        else:
            stall = 0
    return abs(target_u - R.yaw_u) < 0.05
