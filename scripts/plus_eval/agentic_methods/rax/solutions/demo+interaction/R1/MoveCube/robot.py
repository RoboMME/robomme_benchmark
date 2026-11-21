"""Motion primitives over robomme_step (relative world-frame delta rows)."""
import time
import numpy as np
from mcpio import log, split_result, find_key, as_state, terminal_flags, parse_obs, strip_big, dump_json
from geom import rotvec_to_R, R_to_rotvec, Rz, wrap

POS_SCALE = 0.05
ROT_SCALE = 0.5


class EpisodeOver(Exception):
    pass


class Robot:
    def __init__(self, ctx, max_steps=None, deadline=None):
        self.ctx = ctx
        self.state = None
        self.step_id = 0
        self.max_steps = max_steps
        self.ended = False
        self.success = None
        self.grip = -1.0
        self.deadline = deadline
        self.n_calls = 0
        self.last_meta = None
        self.yaw_total = 0.0

    # ---------- low level ----------
    def steps_left(self):
        if self.max_steps is None:
            return 10 ** 6
        return self.max_steps - (self.step_id or 0)

    def send(self, rows):
        if self.ended:
            raise EpisodeOver()
        if self.deadline is not None and time.time() > self.deadline:
            log('wall deadline reached'); self.ended = True; raise EpisodeOver()
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        if self.max_steps is not None:
            left = self.steps_left()
            if left <= 0:
                self.ended = True; raise EpisodeOver()
            rows = rows[:max(1, left)]
        for i in range(0, len(rows), 100):
            chunk = rows[i:i + 100]
            r = self.ctx.call('robomme_step', actions=chunk)
            self.n_calls += 1
            meta, _ = split_result(r)
            self.last_meta = meta
            if self.n_calls <= 2:
                dump_json('step_reply_%d.json' % self.n_calls, strip_big(meta))
            st = find_key(meta, 'state')
            s = as_state(st) if st is not None else None
            if s is not None:
                self.state = s
            else:
                self.no_state_replies = getattr(self, 'no_state_replies', 0) + 1
                if self.no_state_replies <= 3:
                    log('step reply without state; refreshing via observe')
            sid = find_key(meta, 'step_id')
            if isinstance(sid, (int, float)):
                self.step_id = int(sid)
            else:
                self.step_id = (self.step_id or 0) + len(chunk)
            ended, succ = terminal_flags(meta)
            if succ:
                self.success = True
            if ended or find_key(meta, 'episode_ended') is True:
                self.ended = True
                log('EPISODE ENDED at step', self.step_id, 'success', self.success,
                    'meta', str(strip_big({k: meta.get(k) for k in list(meta)[:30] if k not in ('state',)}))[:1500])
                raise EpisodeOver()
            if s is None:
                try:
                    o = parse_obs(self.ctx.call('robomme_observe'))
                    if o.state is not None:
                        self.state = o.state
                    if o.ended:
                        self.ended = True; raise EpisodeOver()
                except EpisodeOver:
                    raise
                except Exception as e:
                    log('observe refresh failed', e)
        return self.state

    def observe(self):
        r = self.ctx.call('robomme_observe')
        o = parse_obs(r)
        if o.state is not None:
            self.state = o.state
        if isinstance(o.step_id, (int, float)):
            self.step_id = int(o.step_id)
        if o.ended:
            self.ended = True
        return o

    # ---------- state helpers ----------
    @property
    def pos(self):
        return np.array(self.state[:3])

    @property
    def R(self):
        return rotvec_to_R(self.state[3:6])

    @property
    def opening(self):
        return float(self.state[6] + self.state[7])

    def finger_yaw(self):
        """World yaw of the finger-closing axis (gripper y axis projected on xy)."""
        y = self.R[:, 1]
        return float(np.arctan2(y[1], y[0]))

    # ---------- motion ----------
    def hold(self, n, grip=None):
        g = self.grip if grip is None else grip
        self.grip = g
        return self.send([[0, 0, 0, 0, 0, 0, g]] * int(n))

    def gripper(self, close, n=12):
        g = 1.0 if close else -1.0
        self.grip = g
        return self.send([[0, 0, 0, 0, 0, 0, g]] * int(n))

    def move(self, p_target, R_target=None, max_lin=0.02, max_rot=0.12, tol=0.004, rot_tol=0.03,
             max_steps=150, chunk=6, grip=None):
        """Closed-loop chunked motion to an absolute position (and orientation)."""
        g = self.grip if grip is None else grip
        self.grip = g
        p_target = np.asarray(p_target, float)
        used = 0
        while used < max_steps:
            e = p_target - self.pos
            w = R_to_rotvec(R_target @ self.R.T) if R_target is not None else np.zeros(3)
            en, wn = np.linalg.norm(e), np.linalg.norm(w)
            if en < tol and wn < rot_tol:
                return True
            n = int(max(1, np.ceil(en / max_lin), np.ceil(wn / max_rot)))
            k = int(min(n, chunk, max_steps - used))
            dp = e / n; dw = w / n
            row = list(dp / POS_SCALE) + list(dw / ROT_SCALE) + [g]
            self.send([row] * k)
            used += k
        e = np.linalg.norm(p_target - self.pos)
        log('move: not converged, err %.4f' % e)
        return False

    def move_linear(self, p_target, R_target=None, speed=0.008, max_steps=200, tol=0.003, grip=None):
        """Slow straight-line motion (pushing), one step per row, re-planned each chunk."""
        return self.move(p_target, R_target, max_lin=speed, max_rot=0.05, tol=tol, rot_tol=0.05,
                         max_steps=max_steps, chunk=4, grip=grip)
