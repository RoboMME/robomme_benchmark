"""Thin defensive wrapper around the RoboMME MCP robot tools + motion primitives."""
import json
import math
import time
import numpy as np

import io_utils as iu
from geometry import rotvec_to_mat, mat_to_rotvec

POS_SCALE = 0.05
ROT_SCALE = 0.5


class EpisodeEnded(Exception):
    pass


class Robot:
    def __init__(self, ctx, log):
        self.ctx = ctx
        self.log = log
        self.state = None
        self.step_id = None
        self.steps_used = 0
        self.max_steps = None
        self.ended = False
        self.complete = False
        self.grip_cmd = -1.0
        self.last_meta = {}
        self.n_calls = 0
        self.rot_mode = 0          # 0 world-frame rotvec (documented), 1 EE-frame, 2 negated
        self.rot_bad = 0
        self.p2p_warned = False

    # ------------------------------------------------------------ tools
    def call(self, tool, **kw):
        self.n_calls += 1
        return self.ctx.call(tool, **kw)

    def observe(self):
        r = self.call('robomme_observe')
        meta = iu.reply_json(r)
        self._absorb(meta)
        return r, meta

    def status(self):
        try:
            r = self.call('robomme_status')
            meta = iu.reply_json(r)
        except Exception as e:
            self.log('robomme_status failed: %r' % (e,))
            return {}
        ms = iu.max_steps_from(meta)
        if ms:
            self.max_steps = ms
        if iu.episode_ended(meta):
            self.ended = True
        return meta

    def _absorb(self, meta):
        if not isinstance(meta, dict):
            return
        self.last_meta = meta
        s = iu.parse_state(meta)
        if s is not None:
            self.state = s
        sid = iu.parse_step_id(meta)
        if sid is not None:
            self.step_id = sid
        if iu.episode_ended(meta):
            self.ended = True
        if iu.task_complete(meta):
            self.complete = True
        ms = iu.max_steps_from(meta)
        if ms and not self.max_steps:
            self.max_steps = ms

    def remaining(self):
        if self.max_steps is None:
            return 10 ** 6
        used = self.step_id if self.step_id is not None else self.steps_used
        return self.max_steps - used

    def p2p(self, sels, radius=0):
        """sels: list of (camera, u, v). Returns list of (xyz or None, spread)."""
        out = []
        for i in range(0, len(sels), 32):
            chunk = sels[i:i + 32]
            args = dict(observation_step_id=int(self.step_id or 0), depth_window_radius=int(radius),
                        selections=[{'camera': c, 'u': float(u), 'v': float(v)} for c, u, v in chunk])
            got = None
            for attempt in range(2):
                try:
                    r = self.call('robomme_pixel_to_position', **args)
                    meta = iu.reply_json(r)
                    got = parse_p2p(meta, len(chunk))
                    if attempt == 0 and all(g[0] is None for g in got) and not self.p2p_warned:
                        self.p2p_warned = True
                        self.log('p2p: no points parsed; reply sample: %s' % (json.dumps(meta)[:600],))
                    break
                except Exception as e:
                    self.log('p2p failed (attempt %d): %r' % (attempt, str(e)[:300],))
                    if attempt == 0:
                        try:
                            self.observe()
                            args['observation_step_id'] = int(self.step_id or 0)
                        except Exception:
                            pass
            out.extend(got if got is not None else [(None, None)] * len(chunk))
        return out

    def step(self, rows):
        if self.ended:
            raise EpisodeEnded()
        rows = [[float(max(-1.0, min(1.0, x))) for x in r] for r in rows]
        if self.max_steps is not None and self.remaining() <= 0:
            self.ended = True
            raise EpisodeEnded()
        if self.max_steps is not None:
            rows = rows[:max(1, self.remaining())]
        r = self.call('robomme_step', actions=rows)
        self.steps_used += len(rows)
        meta = iu.reply_json(r)
        prev_sid = self.step_id
        self._absorb(meta)
        if self.step_id == prev_sid or iu.parse_state(meta) is None:
            # make sure state/step_id are fresh
            try:
                _, m2 = self.observe()
            except Exception as e:
                self.log('observe after step failed %r' % (e,))
        if self.ended:
            raise EpisodeEnded()
        return meta

    # ------------------------------------------------------------ pose
    @property
    def pos(self):
        return np.array(self.state[:3])

    @property
    def R(self):
        return rotvec_to_mat(self.state[3:6])

    @property
    def opening(self):
        return float(self.state[6] + self.state[7])

    def move_to(self, target, R_des=None, vmax=0.02, tol=0.004, rot_tol=0.03, max_rows=60,
                rows_per_call=6, settle=True):
        """Closed-loop Cartesian move.  Returns final position error (m)."""
        target = np.asarray(target, float)
        if R_des is None:
            R_des = self.R
        rmax = 0.12  # rad per step
        used = 0
        err = float(np.linalg.norm(target - self.pos))
        stall = 0
        last_err = err
        last_rerr = None
        while used < max_rows:
            dp = target - self.pos
            err = float(np.linalg.norm(dp))
            er = mat_to_rotvec(R_des @ self.R.T)
            rerr = float(np.linalg.norm(er))
            if last_rerr is not None and last_rerr > 0.06:
                if rerr > last_rerr + 0.004:
                    self.rot_bad += 1
                    if self.rot_bad >= 2:
                        self.rot_mode = (self.rot_mode + 1) % 3
                        self.rot_bad = 0
                        self.log('rotation error grew (%.3f -> %.3f): switching rotation mode to %d' % (
                            last_rerr, rerr, self.rot_mode))
                elif rerr < last_rerr - 0.004:
                    self.rot_bad = 0
            if err < tol and rerr < rot_tol:
                break
            n_lin = int(math.ceil(err / vmax)) if err > tol else 1
            n_rot = int(math.ceil(rerr / rmax)) if rerr > rot_tol else 1
            n = max(1, min(rows_per_call, max(n_lin, n_rot)))
            k = max(n_lin, n_rot, 1)
            step_p = dp / k
            step_r = er / k
            if self.rot_mode == 1:
                step_r = self.R.T @ step_r
            elif self.rot_mode == 2:
                step_r = -step_r
            if rerr <= rot_tol:
                step_r = np.zeros(3)
            row = list(np.clip(step_p / POS_SCALE, -1, 1)) + list(np.clip(step_r / ROT_SCALE, -1, 1)) + [self.grip_cmd]
            self.step([row] * n)
            used += n
            last_rerr = rerr if rerr > rot_tol else None
            if err > last_err - 0.0005 and rerr <= rot_tol:
                stall += 1
            else:
                stall = 0
            last_err = err
            if stall >= 6:
                self.log('move_to stalled err=%.4f' % err)
                break
        return float(np.linalg.norm(target - self.pos))

    def gripper(self, val, n=8):
        self.grip_cmd = float(val)
        rows = [[0, 0, 0, 0, 0, 0, self.grip_cmd]] * n
        self.step(rows)

    def hold(self, n=3):
        self.step([[0, 0, 0, 0, 0, 0, self.grip_cmd]] * n)


def parse_p2p(meta, n):
    """Extract per-selection world points from a pixel_to_position reply."""
    lst = None
    if isinstance(meta, dict):
        for k in ('results', 'points', 'positions', 'selections', 'world_points', '_list', 'items'):
            v = meta.get(k)
            if isinstance(v, list) and len(v) == n:
                lst = v
                break
        if lst is None:
            # any list of length n
            def search(o, depth=0):
                if depth > 4:
                    return None
                if isinstance(o, dict):
                    for v in o.values():
                        if isinstance(v, list) and len(v) == n and all(isinstance(x, (dict, list)) for x in v):
                            return v
                    for v in o.values():
                        r = search(v, depth + 1)
                        if r is not None:
                            return r
                return None
            lst = search(meta)
    out = []
    if lst is None:
        return [(None, None)] * n
    for it in lst:
        xyz, spread = None, None
        if isinstance(it, list) and iu.is_num_list(it, 3):
            xyz = np.array(it, float)
        elif isinstance(it, dict):
            valid = it.get('valid', True)
            for k in ('position', 'world_position', 'world', 'point', 'xyz', 'position_m', 'world_point',
                      'world_xyz', 'world_position_m', 'point_world'):
                v = it.get(k)
                if iu.is_num_list(v, 3):
                    xyz = np.array(v, float)
                    break
                if isinstance(v, dict):
                    if all(c in v for c in 'xyz'):
                        xyz = np.array([v['x'], v['y'], v['z']], float)
                        break
            if xyz is None:
                for k, v in it.items():
                    kl = k.lower()
                    if iu.is_num_list(v, 3) and not any(s in kl for s in ('pixel', 'uv', 'camera', 'cam_', 'normal', 'ray', 'origin')):
                        xyz = np.array(v, float)
                        break
            for k in ('depth_spread_m', 'depth_spread', 'spread'):
                if isinstance(it.get(k), (int, float)):
                    spread = float(it[k])
            if valid is False:
                xyz = None
            if xyz is not None and not np.all(np.isfinite(xyz)):
                xyz = None
        out.append((xyz, spread))
    return out
