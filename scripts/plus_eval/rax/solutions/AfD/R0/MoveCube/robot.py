"""Thin robot wrapper: observation caching, closed-loop Cartesian moves, gripper, pixel->3D."""
import math
import time

import numpy as np

import io_utils as iu

POS_SCALE = 0.05   # metres per unit action
ROT_SCALE = 0.5    # radians per unit action


def rotvec_to_mat(r):
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def mat_to_rotvec(R):
    c = (np.trace(R) - 1) / 2
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: axis from the largest diagonal entry
        M = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(M)))
        ax = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return ax / np.linalg.norm(ax) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class EpisodeOver(Exception):
    pass


class Robot:
    def __init__(self, ctx, log, t0=None, wall_limit=1100.0):
        self.ctx = ctx
        self.log = log
        self.t0 = t0 or time.time()
        self.wall_limit = wall_limit
        self.meta = {}
        self.images = {}
        self.state = None
        self.step_id = None
        self.ended = False
        self.success = False
        self.grip = -1.0
        self.steps_used = 0
        self.max_steps = None
        self.n_errors = 0
        self.R0 = None
        self.home = None
        self.close_axis = 1  # EE-frame axis along which the fingers close
        self.rot_local = False  # True if rotation deltas turn out to act in the tool frame
        self.rot_bad = 0
        self.approach_axis = 2

    # ------------------------------------------------------------------ io
    def time_left(self):
        return self.wall_limit - (time.time() - self.t0)

    def observe(self):
        res = self.ctx.call('robomme_observe')
        self._ingest(res)
        return self.meta

    def _ingest(self, res, from_step=False):
        meta = iu.first_json(res)
        self.meta = meta
        ims = iu.get_images(res, meta)
        if ims:
            self.images = {k: _to256(v) for k, v in ims.items()}
        st = iu.get_state(meta)
        if st is not None:
            self.state = st
        sid = iu.get_step_id(meta)
        if sid is not None:
            self.step_id = sid
        if iu.task_complete(meta):
            self.success = True
        if iu.episode_ended(meta):
            self.ended = True
        lim = iu.find_key(meta, ('max_steps', 'step_limit', 'max_episode_steps'), lambda x: isinstance(x, int))
        if lim:
            self.max_steps = lim
        used = iu.find_key(meta, ('steps_used', 'elapsed_steps', 'step_count'), lambda x: isinstance(x, int))
        if used is not None:
            self.steps_used = used
        elif self.step_id is not None:
            self.steps_used = max(self.steps_used, self.step_id)

    def read_limits(self):
        try:
            res = self.ctx.call('robomme_status')
            meta = iu.first_json(res)
            self.log('robomme_status: ' + _short(meta))
            lim = iu.find_key(meta, ('max_steps', 'step_limit', 'max_episode_steps', 'horizon'), lambda x: isinstance(x, int))
            if lim:
                self.max_steps = lim
            if iu.episode_ended(meta):
                self.ended = True
        except Exception as e:
            self.log(f'robomme_status failed: {e}')

    def steps_left(self):
        if self.max_steps is None:
            return 10 ** 6
        return self.max_steps - self.steps_used

    def send(self, rows):
        if self.ended:
            raise EpisodeOver()
        if self.time_left() < 25:
            raise EpisodeOver('wall time')
        rows = [[float(max(-1.0, min(1.0, x))) for x in r] for r in rows]
        if self.max_steps is not None:
            left = self.steps_left()
            if left <= 1:
                raise EpisodeOver('steps')
            rows = rows[:max(1, left - 1)]
        try:
            res = self.ctx.call('robomme_step', actions=rows)
        except RuntimeError as e:
            self.n_errors += 1
            self.log(f'step error #{self.n_errors}: {str(e)[:300]}')
            try:
                self.observe()
            except Exception as e2:
                self.log(f'observe after error failed: {e2}')
            if self.n_errors > 8 or 'ended' in str(e).lower():
                self.ended = self.ended or ('ended' in str(e).lower())
                raise EpisodeOver(str(e)[:200])
            return False
        self.steps_used += len(rows)
        self._ingest(res, from_step=True)
        if iu.get_state(self.meta) is None:
            self.observe()
        if self.ended:
            raise EpisodeOver('ended')
        return True

    # ------------------------------------------------------------------ pose
    @property
    def pos(self):
        return self.state[:3].copy()

    @property
    def R(self):
        return rotvec_to_mat(self.state[3:6])

    @property
    def fingers(self):
        return float(self.state[6] + self.state[7])

    def init_frames(self):
        self.R0 = self.R
        self.home = self.pos
        # approach axis: EE axis closest to world -z
        dots = [self.R0[:, i] @ np.array([0, 0, -1.0]) for i in range(3)]
        self.approach_axis = int(np.argmax(dots))
        self.log(f'home pos {np.round(self.home, 3)} approach axis {self.approach_axis} dots {np.round(dots, 2)}')

    def close_dir(self, R=None):
        R = self.R if R is None else R
        d = R[:, self.close_axis].copy()
        d[2] = 0
        n = np.linalg.norm(d)
        return d / n if n > 1e-6 else np.array([0, 1.0, 0])

    def yaw_for_close_dir(self, want_dir_xy):
        """Yaw offset psi (about world z, relative to R0) so the fingers close along want_dir (mod pi);
        choose the solution with the smallest |psi|."""
        c0 = self.close_dir(self.R0)
        a0 = math.atan2(c0[1], c0[0])
        aw = math.atan2(want_dir_xy[1], want_dir_xy[0])
        psi = wrap(aw - a0)
        if psi > math.pi / 2:
            psi -= math.pi
        elif psi < -math.pi / 2:
            psi += math.pi
        return psi

    def R_des(self, psi):
        return rotz(psi) @ self.R0

    def current_psi(self):
        Rr = self.R @ self.R0.T
        return math.atan2(Rr[1, 0], Rr[0, 0])

    # ------------------------------------------------------------------ motion
    def move_to(self, target, psi=None, vmax=0.02, wmax=0.12, tol=0.004, atol=0.03, max_iter=60, chunk=8, label=''):
        """Closed-loop Cartesian move. target: xyz world; psi: yaw offset relative to R0 (None: keep)."""
        target = np.asarray(target, float)
        Rd = self.R_des(self.current_psi() if psi is None else psi)
        last_d = None; stall = 0
        for it in range(max_iter):
            p = self.pos
            dp = target - p
            d = float(np.linalg.norm(dp))
            e = mat_to_rotvec(Rd @ self.R.T)
            ang = float(np.linalg.norm(e))
            if d < tol and ang < atol:
                return True
            m = max(1, int(math.ceil(d / vmax)), int(math.ceil(ang / wmax)))
            k = min(m, chunk)
            if m == 1:
                row_p = dp * 0.9
                row_r = e * 0.9
            else:
                row_p = dp / m
                row_r = e / m
            if self.rot_local:
                row_r = self.R.T @ row_r
            row = list(row_p / POS_SCALE) + list(row_r / ROT_SCALE) + [self.grip]
            self.send([row] * k)
            # rotation-convention check: a commanded rotation of >0.1 rad should reduce the error
            if k * float(np.linalg.norm(row_r)) > 0.1:
                e2 = float(np.linalg.norm(mat_to_rotvec(Rd @ self.R.T)))
                if e2 > ang + 0.02:
                    self.rot_bad += 1
                    self.log(f'rotation error grew {ang:.3f}->{e2:.3f} (bad={self.rot_bad})')
                    if self.rot_bad >= 2:
                        self.rot_local = not self.rot_local
                        self.rot_bad = 0
                        self.log(f'switch rotation convention: local={self.rot_local}')
                elif e2 < ang - 0.02:
                    self.rot_bad = 0
            if last_d is not None and d > last_d - 0.0005 and ang < atol * 3:
                stall += 1
                if stall >= 6:
                    self.log(f'move_to {label}: stalled at d={d:.4f} ang={ang:.3f}')
                    return False
            else:
                stall = 0
            last_d = d
        self.log(f'move_to {label}: max_iter, d={d:.4f}')
        return False

    def hold(self, n, grip=None):
        if grip is not None:
            self.grip = grip
        rows = [[0, 0, 0, 0, 0, 0, self.grip]] * n
        while rows:
            self.send(rows[:50])
            rows = rows[50:]

    def set_gripper(self, g, n=12):
        self.grip = g
        self.hold(n)

    # ------------------------------------------------------------------ perception
    def pixels_to_points(self, sels, camera='front', radius=1):
        """sels: list of (u,v). Uses the current observation's step id."""
        out = []
        sels = list(sels)
        for i in range(0, len(sels), 32):
            chunk = sels[i:i + 32]
            args = {'observation_step_id': int(self.step_id if self.step_id is not None else 0),
                    'selections': [{'camera': camera, 'u': float(u), 'v': float(v)} for u, v in chunk]}
            if radius is not None:
                args['depth_window_radius'] = int(radius)
            pts = None
            for attempt in range(3):
                try:
                    res = self.ctx.call('robomme_pixel_to_position', **args)
                    meta = iu.first_json(res)
                    pts = iu.parse_points(meta, len(chunk))
                    break
                except EpisodeOver:
                    raise
                except Exception as e:
                    self.log(f'pixel_to_position failed (try {attempt}): {str(e)[:200]}')
                    if attempt == 2:
                        break
                    try:
                        if attempt == 1 and not self.ended:
                            self.hold(1)  # fresh observation/step id
                        self.observe()
                    except EpisodeOver:
                        raise
                    except Exception as e2:
                        self.log(f'observe during pixel retry failed: {e2}')
                    args['observation_step_id'] = int(self.step_id if self.step_id is not None else 0)
            if pts is None:
                pts = [(None, None)] * len(chunk)
            out.extend(pts)
        return out


def _to256(a):
    if a is None:
        return None
    if a.shape[0] == 256 and a.shape[1] == 256:
        return a
    from PIL import Image
    return np.array(Image.fromarray(a).resize((256, 256)))


def _short(meta):
    import json
    try:
        s = json.dumps(meta)
    except Exception:
        s = str(meta)
    return s[:1500]
