"""Robot wrapper: state tracking, closed-loop Cartesian moves, back-projection batching."""
import math
import time

import numpy as np

import mcp_io as io_
from mcp_io import log

POS_SCALE = 0.05     # m per unit action
ROT_SCALE = 0.5      # rad per unit action
KEEPALIVE_S = 60.0   # s without a control step before perception inserts a zero step


def rotvec_to_mat(r):
    r = np.asarray(r, dtype=float)
    th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def mat_to_rotvec(R):
    R = np.asarray(R, dtype=float)
    c = (np.trace(R) - 1) / 2
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: axis from the symmetric part
        A = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(A)))
        k = A[:, i] / math.sqrt(max(A[i, i], 1e-12))
        return k / np.linalg.norm(k) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


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
        self.max_steps = None
        self.last_result = None
        self.last_metas = []
        self.rot_world = True          # rotation rows interpreted in world frame (documented)
        self.steps_sent = 0
        self.ref_front = None
        self.cams = {'front': None, 'wrist': None}
        self.grip_cmd = -1.0
        self.order_hint = None       # for unlabelled image pairs: 'front_first' / 'wrist_first'
        self.cam_how = None
        self.last_unlabelled = []
        self.p2p_calls = 0
        self.p2p_time = 0.0
        self.last_control_t = time.time()

    # ------------------------------------------------------------------ raw calls
    def _absorb(self, result):
        metas = io_.meta_of(result)
        self.last_result, self.last_metas = result, metas
        st = io_.find_state(metas)
        if st is not None:
            self.state = st
        sid = io_.find_step_id(metas)
        if sid is not None:
            self.step_id = sid
        if io_.episode_ended(metas):
            self.ended = True
        if io_.task_complete(metas):
            self.complete = True
            self.ended = True
        rem = io_.remaining_steps(metas)
        if rem is not None:
            self.remaining = rem
        mx = io_.max_steps(metas)
        if mx is not None:
            self.max_steps = mx
        return metas

    def _take_images(self, res, metas):
        cams, how, unl = io_.assign_rgb(res, metas, self.ref_front, self.step_id, self.order_hint)
        self.cam_how = how
        self.last_unlabelled = unl
        if how in ('manifest', 'heuristic') and len(unl) >= 2 and cams['front'] is not None and self.order_hint is None:
            first_is_front = cams['front'] is unl[0]
            if how == 'manifest':
                self.order_hint = 'front_first' if first_is_front else 'wrist_first'
                log('camera order from manifest:', self.order_hint)
        for k in cams:
            if cams[k] is not None:
                self.cams[k] = cams[k]

    def observe(self):
        res = self.ctx.call('robomme_observe')
        metas = self._absorb(res)
        self._take_images(res, metas)
        if self.ref_front is None and self.cams['front'] is not None and self.cam_how in ('labels', 'manifest', 'hint'):
            self.ref_front = self.cams['front'].astype(np.int16)
        return res, metas

    def identify_cameras(self):
        """Unlabelled attachments: move 1 cm up and back; the static front view changes least."""
        if self.cam_how not in ('heuristic',) or len(self.last_unlabelled) < 2 or self.order_hint is not None:
            return
        a0, b0 = [x.astype(np.int16) for x in self.last_unlabelled[:2]]
        self.send([[0, 0, 0.2, 0, 0, 0, self.grip_cmd]] * 2)
        unl = self.last_unlabelled
        if len(unl) >= 2:
            da = float(np.abs(unl[0].astype(np.int16) - a0).mean())
            db = float(np.abs(unl[1].astype(np.int16) - b0).mean())
            self.order_hint = 'front_first' if da <= db else 'wrist_first'
            log('camera order by motion test: change first=%.2f second=%.2f -> %s' % (da, db, self.order_hint))
        self.send([[0, 0, -0.2, 0, 0, 0, self.grip_cmd]] * 2)
        self.observe()
        if self.cams['front'] is not None:
            self.ref_front = self.cams['front'].astype(np.int16)

    def status(self):
        try:
            res = self.ctx.call('robomme_status')
            self._absorb(res)
            return io_.meta_of(res)
        except Exception as e:
            log('status failed', repr(e)[:200])
            return []

    def send(self, rows, want_images=True):
        if self.ended:
            raise EpisodeOver()
        rows = [[float(max(-1.0, min(1.0, x))) for x in r] for r in rows]
        if self.remaining is not None:
            if self.remaining <= 0:
                raise EpisodeOver()
            rows = rows[:max(1, self.remaining)]
        prev_sid = self.step_id
        prev_state = self.state
        res = self.ctx.call('robomme_step', actions=rows)
        self.last_control_t = time.time()
        metas = self._absorb(res)
        self.steps_sent += len(rows)
        if self.remaining is not None and io_.remaining_steps(metas) is None:
            self.remaining -= len(rows)
        if self.step_id is None or (prev_sid is not None and self.step_id <= prev_sid):
            self.step_id = (prev_sid or 0) + len(rows)   # guess; observe() refreshes it
        if want_images:
            self._take_images(res, metas)
        if self.state is prev_state or self.state is None:
            # state not in the step reply: refresh explicitly
            try:
                self.observe()
            except Exception as e:
                log('observe after step failed', repr(e)[:200])
        if self.ended:
            log('episode ended; complete =', self.complete)
        return res

    # ------------------------------------------------------------------ state helpers
    @property
    def pos(self):
        return np.array(self.state[0:3], dtype=float)

    @property
    def R(self):
        return rotvec_to_mat(self.state[3:6])

    @property
    def finger_sum(self):
        try:
            v = float(self.state[6]) + float(self.state[7])
            return v if math.isfinite(v) else None
        except Exception:
            return None

    # ------------------------------------------------------------------ motion
    def rot_delta(self, R_target):
        """Rotation vector to command (world or local convention)."""
        Rc = self.R
        if self.rot_world:
            return mat_to_rotvec(R_target @ Rc.T)
        return mat_to_rotvec(Rc.T @ R_target)

    def hold(self, n, grip=None):
        g = self.grip_cmd if grip is None else grip
        self.grip_cmd = g
        done = 0
        while done < n and not self.ended:
            k = min(10, n - done)
            self.send([[0, 0, 0, 0, 0, 0, g]] * k)
            done += k

    def goto(self, target, R_target=None, grip=None, tol=0.004, rot_tol=0.03, vmax=0.022,
             chunk=5, max_calls=40, min_z=None, label=''):
        """Closed-loop move of the TCP to `target` (world m). Returns (ok, why)."""
        g = self.grip_cmd if grip is None else grip
        self.grip_cmd = g
        target = np.array(target, dtype=float)
        if min_z is not None:
            target[2] = max(target[2], min_z)
        hist = []
        rot_hist = []
        for call in range(max_calls):
            if self.ended:
                return False, 'ended'
            if self.state is None:
                return False, 'no_state'
            err = target - self.pos
            dist = float(np.linalg.norm(err))
            rv = self.rot_delta(R_target) if R_target is not None else np.zeros(3)
            rerr = float(np.linalg.norm(rv))
            hist.append(dist)
            rot_hist.append(rerr)
            if dist < tol and rerr < rot_tol:
                return True, 'reached'
            # rotation-convention self check: error grew after rotating twice
            if R_target is not None and len(rot_hist) >= 3 and rot_hist[-1] > rot_hist[-2] + 0.01 \
                    and rot_hist[-2] > rot_hist[-3] + 0.01 and rot_hist[-3] > 0.05:
                self.rot_world = not self.rot_world
                log('rotation error grows; switching rotation convention, world =', self.rot_world)
                rot_hist.clear()
                rv = self.rot_delta(R_target)
                rerr = float(np.linalg.norm(rv))
            # stall: no progress over 4 calls
            if len(hist) >= 5 and hist[-5] - hist[-1] < 0.002 and rerr < rot_tol * 3:
                return False, 'stall dist=%.4f' % dist
            need = max(1, int(math.ceil(dist / vmax)), int(math.ceil(rerr / 0.12)))
            n = min(need, chunk)
            dp = err / need
            dr = rv / need
            row = list(dp / POS_SCALE) + list(dr / ROT_SCALE) + [g]
            self.send([row] * n)
        return False, 'max_calls dist=%.4f' % float(np.linalg.norm(target - self.pos))

    # ------------------------------------------------------------------ perception
    def p2p_latency(self):
        return self.p2p_time / self.p2p_calls if self.p2p_calls else 0.0

    def backproject(self, cam, pixels, budget_s=None):
        """pixels: list of (u, v). Returns list of np.array(3) or None, same order."""
        out = []
        t0 = time.time()
        for i in range(0, len(pixels), 32):
            if budget_s is not None and time.time() - t0 > budget_s:
                out.extend([None] * (len(pixels) - len(out)))
                log('backproject budget exceeded at', i, 'of', len(pixels))
                break
            if time.time() - self.last_control_t > KEEPALIVE_S and not self.ended:
                # runtime kills a controller that does not advance control for 180 s
                log('keepalive zero step during perception')
                try:
                    self.send([[0, 0, 0, 0, 0, 0, self.grip_cmd]])
                    self.observe()
                except EpisodeOver:
                    raise
                except Exception as e:
                    log('keepalive failed', repr(e)[:200])
            chunk = pixels[i:i + 32]
            sels = [{'camera': cam, 'u': float(u), 'v': float(v)} for (u, v) in chunk]
            tc = time.time()
            try:
                args = dict(observation_step_id=int(self.step_id if self.step_id is not None else 0),
                            selections=sels, depth_window_radius=0)
                res = self.ctx.call('robomme_pixel_to_position', **args)
                pts = io_.parse_points(io_.meta_of(res), len(chunk))
            except Exception as e:
                log('pixel_to_position failed', repr(e)[:300])
                pts = [None] * len(chunk)
            self.p2p_calls += 1
            self.p2p_time += time.time() - tc
            out.extend([np.array(p) if p is not None else None for p in pts])
        return out
