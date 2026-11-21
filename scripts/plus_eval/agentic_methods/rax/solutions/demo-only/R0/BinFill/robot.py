"""Robot access: observe, step, pixel queries, dense geometry with self-validation."""
import math
import time
import numpy as np

from mcpio import log, parse_observation, parse_meta, parse_points, walk, is_num_list
from geom import Camera, rotvec_to_mat


class EpisodeOver(Exception):
    pass


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.obs = None
        self.steps = 0          # executed control steps (our count / reported)
        self.limit = None
        self.ended = False
        self.success = None
        self.dense_ok = {}      # cam -> (conv, depth_kind, offset) or False
        self.n_calls = 0
        self.last_grip = -1.0
        self._fresh = False

    # ------------------------------------------------------------ basic calls
    def status(self):
        try:
            r = self.ctx.call('robomme_status')
            m = parse_meta(r)
            log('status keys', list(m.keys())[:30])
            log('status', str(m)[:1500])
            for p, k, v in walk(m):
                kl = str(k).lower()
                if self.limit is None and isinstance(v, (int, float)) and not isinstance(v, bool) and \
                        kl in ('max_steps', 'step_limit', 'max_episode_steps', 'episode_step_limit', 'horizon', 'limit', 'steps_limit'):
                    self.limit = int(v)
            for p, k, v in walk(m):
                if str(k).lower() in ('episode_limits', 'limits') and isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float)) and not isinstance(vv, bool) and 'step' in kk.lower() and 'remain' not in kk.lower() and 'used' not in kk.lower() and 'elapsed' not in kk.lower():
                            if self.limit is None:
                                self.limit = int(vv)
            return m
        except Exception as e:
            log('status failed', repr(e)[:300])
            return {}

    def observe(self):
        # observation has no physics: safe to retry on transient tool errors
        last = None
        for k in range(3):
            try:
                r = self.ctx.call('robomme_observe')
                break
            except Exception as e:
                last = e
                log('observe error (try %d)' % k, repr(e)[:300])
                time.sleep(0.3 * (k + 1))
        else:
            raise last
        self.n_calls += 1
        o = parse_observation(r)
        if o.state is None and self.obs is not None:
            o.state = self.obs.state
        self.obs = o
        self._fresh = o.state is not None
        self._update_term(o.meta)
        if o.step_id is not None:
            self.steps = max(self.steps, o.step_id)
        return o

    def _update_term(self, meta):
        for p, k, v in walk(meta):
            kl = str(k).lower()
            if kl in ('episode_ended', 'terminated', 'done', 'ended') and v is True:
                self.ended = True
            if kl in ('task_complete', 'success') and v is True:
                self.success = True
            if kl in ('episode_limits', 'limits') and isinstance(v, dict) and self.limit is None:
                for kk, vv in v.items():
                    if isinstance(vv, (int, float)) and not isinstance(vv, bool) and 'step' in kk.lower() and \
                            not any(w in kk.lower() for w in ('remain', 'used', 'elapsed', 'current')):
                        self.limit = int(vv)

    def remaining(self):
        if self.limit is None:
            return 10 ** 6
        return self.limit - self.steps

    def step(self, rows):
        """Send rows; returns parsed reply meta. Raises EpisodeOver when ended."""
        if self.ended:
            raise EpisodeOver()
        rows = [[float(max(-1.0, min(1.0, x))) for x in r] for r in rows]
        if not rows:
            return {}
        for i in range(0, len(rows), 100):
            chunk = rows[i:i + 100]
            r = None
            for attempt in range(2):
                before = self.steps
                try:
                    r = self.ctx.call('robomme_step', actions=chunk)
                    break
                except Exception as e:
                    log('STEP ERROR', repr(e)[:500])
                    # re-observe to learn whether the chunk executed; retry only if it clearly did not
                    try:
                        o = self.observe()
                    except Exception as e2:
                        log('observe after step error failed', repr(e2)[:300])
                        raise e
                    if self.ended:
                        raise EpisodeOver()
                    sid = o.step_id
                    if attempt == 0 and sid is not None and sid == before and self.steps == before:
                        log('step chunk did not execute (step_id %s); retrying once' % sid)
                        continue
                    if sid is not None and sid > before:
                        log('step chunk executed despite error (step_id %s -> %s)' % (before, sid))
                        r = None
                        break
                    raise
            if r is None:
                self.last_grip = chunk[-1][6]
                continue
            self.n_calls += 1
            m = parse_meta(r)
            self._update_term(m)
            sid = None
            for p, k, v in walk(m):
                if k == 'step_id' and isinstance(v, int):
                    sid = v if sid is None else max(sid, v)
            if sid is not None:
                self.steps = max(self.steps, sid)
            else:
                self.steps += len(chunk)
            # update state from reply when present
            st = m.get('state')
            if not is_num_list(st, 8):
                c = [v for p, k, v in walk(m) if k == 'state' and is_num_list(v, 8)]
                st = c[-1] if c else None
            if st is not None and self.obs is not None:
                self.obs.state = [float(x) for x in st]
                self.obs.stale = True
                self._fresh = True
            else:
                self._fresh = False
            self.last_grip = rows[-1][6]
            if self.ended:
                raise EpisodeOver()
        return m

    # ------------------------------------------------------------ state helpers
    def state(self):
        if self.obs is None or self.obs.state is None or not self._fresh:
            self.observe()
        if self.obs.state is None:
            raise RuntimeError('no robot state available')
        return self.obs.state

    def tcp(self):
        return np.array(self.state()[:3])

    def R(self):
        return rotvec_to_mat(self.state()[3:6])

    def fingers(self):
        s = self.state()
        return float(s[6] + s[7])

    # ------------------------------------------------------------ geometry
    def pixels_tool(self, cam, uvs, obs=None):
        obs = obs or self.obs
        out = []
        for i in range(0, len(uvs), 32):
            sel = [{'camera': cam, 'u': float(u), 'v': float(v)} for (u, v) in uvs[i:i + 32]]
            kw = dict(selections=sel)
            if obs is not None and obs.step_id is not None:
                kw['observation_step_id'] = int(obs.step_id)
            res = None
            for k in range(2):
                try:
                    r = self.ctx.call('robomme_pixel_to_position', depth_window_radius=1, **kw)
                    res = parse_points(r, len(sel))
                    break
                except Exception as e:
                    log('pixel_to_position failed (try %d)' % k, repr(e)[:300])
            out += res if res is not None else [{'pos': None, 'spread': None} for _ in sel]
        return out

    def validate_dense(self, obs=None):
        """Compare own back-projection with the tool for each camera; pick convention."""
        obs = obs or self.obs
        for cam in ('front', 'wrist'):
            if self.dense_ok.get(cam):
                continue
            if cam not in obs.cams or cam not in obs.depth:
                log('dense: no calib/depth for', cam, 'cams', list(obs.cams), 'depth', list(obs.depth))
                self.dense_ok[cam] = False
                continue
            d = obs.depth[cam]
            H, W = d.shape
            uvs = []
            for (fu, fv) in [(0.3, 0.6), (0.7, 0.6), (0.5, 0.8), (0.2, 0.9), (0.8, 0.9), (0.5, 0.45), (0.35, 0.75), (0.65, 0.75)]:
                u, v = int(fu * W), int(fv * H)
                if d[v, u] > 0:
                    uvs.append((u, v))
            pts = self.pixels_tool(cam, uvs, obs)
            pairs = [(uv, p['pos']) for uv, p in zip(uvs, pts) if p['pos'] is not None]
            if len(pairs) < 3:
                log('dense: tool gave too few points for', cam)
                self.dense_ok[cam] = False
                continue
            best = None
            base = obs.cams[cam]
            for conv in ('opengl', 'opencv'):
                for dk in ('z', 'ray'):
                    for off in (0.0, 0.5):
                        c = Camera(base.K, base.R, base.t, conv=conv, depth_kind=dk, pix_offset=off)
                        errs = []
                        for (u, v), P in pairs:
                            Q = c.backproject(u, v, d[v, u])
                            errs.append(float(np.linalg.norm(Q - P)))
                        e = float(np.median(errs))
                        if best is None or e < best[0]:
                            best = (e, conv, dk, off)
            log('dense validation', cam, 'best', best)
            if best[0] < 0.012:
                self.dense_ok[cam] = best[1:]
            else:
                self.dense_ok[cam] = False

    def camera(self, cam, obs=None):
        obs = obs or self.obs
        cfg = self.dense_ok.get(cam)
        if not cfg or cam not in obs.cams:
            return None
        b = obs.cams[cam]
        return Camera(b.K, b.R, b.t, conv=cfg[0], depth_kind=cfg[1], pix_offset=cfg[2])

    def dense(self, cam, obs=None):
        obs = obs or self.obs
        c = self.camera(cam, obs)
        if c is None or cam not in obs.depth:
            return None
        d = obs.depth[cam]
        P = c.dense(d)
        P[d <= 0] = np.nan
        return P

    def points(self, cam, uvs, obs=None):
        """3-D points for pixels: dense model if validated else the tool."""
        obs = obs or self.obs
        P = self.dense(cam, obs)
        if P is not None:
            out = []
            H, W = P.shape[:2]
            for (u, v) in uvs:
                ui, vi = int(round(u)), int(round(v))
                ui, vi = min(max(ui, 0), W - 1), min(max(vi, 0), H - 1)
                # median of a 3x3 window of valid points
                w = P[max(vi - 1, 0):vi + 2, max(ui - 1, 0):ui + 2].reshape(-1, 3)
                w = w[~np.isnan(w).any(1)]
                out.append(np.median(w, 0) if len(w) else None)
            return out
        return [p['pos'] for p in self.pixels_tool(cam, uvs, obs)]
