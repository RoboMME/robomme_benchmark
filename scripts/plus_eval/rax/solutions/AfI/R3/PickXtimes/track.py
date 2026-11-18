"""Target (purple bullseye) detection and motion model for PickXtimes.

Evidence (rehearsal 1, 2026-11-16): the target can move continuously along a closed loop
(~0.004 m/step, period ~160 steps) and stops while the cube rests on it; it resumes when the cube is
lifted. In ep1 it was static.  The model: position = L(m) where m is 'moving time' (steps while moving).
"""
import math
import numpy as np
import rlib
from rlib import log

MIN_FRAC = 0.70          # accept a detection if pixel count >= MIN_FRAC * expected (unoccluded)
STATIC_EPS = 0.006       # m: total motion below this over the static window => static
STATIC_WINDOW = 30       # steps
CAL_MAX = 330            # steps of calibration at most
LOOP_TOL = 0.015         # m: detection must lie this close to the loop to be used for phase sync


def detect_target(obs, cam):
    """Return (xy(2), count, mean_depth) of the purple target, world frame; None if not seen."""
    rgb = obs.rgb.get(cam)
    if rgb is None:
        return None
    mask = rlib.color_mask(rgb, 'purple')
    n = int(mask.sum())
    if n < 8:
        return None
    ys, xs = np.nonzero(mask)
    d = obs.depth[cam][ys, xs]
    ok = d > 0
    ys, xs, d = ys[ok], xs[ok], d[ok]
    if len(d) < 8:
        return None
    K = obs.K[cam]
    pc = np.stack([(xs - K[0, 2]) / K[0, 0] * d, -(ys - K[1, 2]) / K[1, 1] * d, -d], 1)
    Rm = rlib.quat_to_R(obs.pose[cam]['quaternion_wxyz'])
    t = np.array(obs.pose[cam]['position_m'])
    pw = pc @ Rm.T + t
    # keep the main cluster (within 7 cm of the median)
    med = np.median(pw, 0)
    keep = np.linalg.norm(pw[:, :2] - med[:2], axis=1) < 0.07
    if keep.sum() < 8:
        return None
    pw = pw[keep]
    c = pw.mean(0)
    return c[:2], int(keep.sum()), float(np.mean(d[keep])), float(c[2])


class Tracker:
    def __init__(self):
        self.k = {}            # cam -> count*depth^2 for an unoccluded view (calibrated at step 0)
        self.hist = []         # (step, x, y, cam, frac)
        self.mode = None       # 'static' | 'loop' | 'linear'
        self.static_xy = None
        self.loop = None       # array (T, 2) sampled per moving-step
        self.T = None
        self.m_ref = None      # moving time at step s_ref
        self.s_ref = None
        self.moving = True
        self.stop_xy = None
        self.v_loop = None

    # ---------------- detections ----------------
    def calib_scale(self, obs):
        for cam in ('wrist', 'front'):
            r = detect_target(obs, cam)
            if r is not None:
                self.k[cam] = r[1] * r[2] ** 2

    def good_detections(self, obs):
        out = []
        for cam in ('wrist', 'front'):
            r = detect_target(obs, cam)
            if r is None:
                continue
            xy, n, d, z = r
            if cam in self.k:
                frac = n * d * d / self.k[cam]
            else:
                frac = 1.0
            if frac >= MIN_FRAC and z < 0.03:
                out.append((cam, np.asarray(xy), frac))
        return out

    def add(self, obs):
        """Record the best detection of this observation; returns (xy, cam) or None."""
        dets = self.good_detections(obs)
        if not dets:
            return None
        dets.sort(key=lambda t: (t[0] != 'wrist', -t[2]))
        cam, xy, frac = dets[0]
        self.hist.append((int(obs.step_id), float(xy[0]), float(xy[1]), cam, float(frac)))
        return xy, cam

    # ---------------- calibration ----------------
    def calibrate(self, R, max_steps=CAL_MAX):
        """Hold the arm still (it is at the reset pose) and watch the target."""
        o = R.last
        self.calib_scale(o)
        self.add(o)
        s0 = o.step_id
        while not R.ended:
            o = R.hold(2)          # R.on_obs records the detection
            el = o.step_id - s0
            H = [h for h in self.hist if h[0] >= s0]
            if len(H) < 3:
                if el > 40:
                    break
                continue
            P = np.array([[h[1], h[2]] for h in H])
            if el >= STATIC_WINDOW and self.mode is None and len(P) >= 6:
                k = max(2, len(P) // 4)
                span = float(np.linalg.norm(np.median(P[-k:], 0) - np.median(P[:k], 0)))
                if span < STATIC_EPS:
                    self.mode = 'static'
                    self.static_xy = np.median(P, 0)
                    log(f'[track] static target at {self.static_xy.round(4)} span={span:.4f}')
                    return True
            if el >= 60:
                T = self._find_period(H)
                if T is not None:
                    self._build_loop(H, T)
                    log(f'[track] loop model T={self.T} n={len(H)} at step {o.step_id}')
                    return True
            if el >= max_steps:
                break
        # fall back: linear model from the last samples
        H = [h for h in self.hist if h[0] >= s0]
        if len(H) >= 3:
            self.mode = 'linear'
            log('[track] no period found; linear fallback')
            return True
        return False

    def _interp(self, H, steps):
        s = np.array([h[0] for h in H], float)
        x = np.array([h[1] for h in H]); y = np.array([h[2] for h in H])
        return np.stack([np.interp(steps, s, x), np.interp(steps, s, y)], 1)

    def _find_period(self, H):
        s = np.array([h[0] for h in H], float)
        s0, s1 = s[0], s[-1]
        if s1 - s0 < 60:
            return None
        P0 = self._interp(H, [s0])[0]
        steps = np.arange(s0, s1 + 1)
        P = self._interp(H, steps)
        d = np.linalg.norm(P - P0, axis=1)
        if d.max() < 0.03:
            return None
        far = np.nonzero(d > 0.03)[0]
        first_far = far[0]
        cand = None
        for i in range(first_far, len(d)):
            if d[i] < 0.012 and (i + 1 >= len(d) or d[i + 1] >= d[i]) and d[i] <= d[i - 1]:
                cand = i
                break
        if cand is None or cand < 40:
            return None
        # need a few samples beyond the candidate period to refine
        if s1 - s0 < cand + 8:
            return None
        best = None
        for T in range(max(30, cand - 8), cand + 9):
            n = int(s1 - s0 - T)
            if n < 5:
                continue
            a = self._interp(H, s0 + np.arange(n))
            b = self._interp(H, s0 + T + np.arange(n))
            e = float(np.mean(np.linalg.norm(a - b, axis=1)))
            if best is None or e < best[0]:
                best = (e, T)
        if best is None or best[0] > 0.01:
            return None
        return best[1]

    def _build_loop(self, H, T):
        s = np.array([h[0] for h in H], float)
        s0 = s[0]
        # average over all cycles: position at phase p = mean of samples at s0+p+kT
        loop = np.zeros((T, 2)); cnt = np.zeros(T)
        steps = np.arange(s0, s[-1] + 1)
        P = self._interp(H, steps)
        for i, st in enumerate(steps):
            p = int(st - s0) % T
            loop[p] += P[i]; cnt[p] += 1
        loop /= np.maximum(cnt, 1)[:, None]
        self.loop = loop
        self.T = T
        self.mode = 'loop'
        self.v_loop = float(np.mean(np.linalg.norm(np.roll(loop, -1, 0) - loop, axis=1)))
        self.s_ref = int(s0)
        self.m_ref = 0.0
        self.moving = True

    # ---------------- prediction ----------------
    def L(self, m):
        T = self.T
        m = m % T
        i0 = int(math.floor(m)); f = m - i0
        a = self.loop[i0 % T]; b = self.loop[(i0 + 1) % T]
        return a * (1 - f) + b * f

    def m_at(self, step):
        if self.mode != 'loop':
            return float(step)
        if not self.moving:
            return self.m_ref
        return self.m_ref + (step - self.s_ref)

    def predict(self, step):
        if self.mode == 'static':
            return self.static_xy.copy()
        if self.mode == 'loop':
            if not self.moving and self.stop_xy is not None:
                return self.stop_xy.copy()
            return self.L(self.m_at(step))
        if self.mode == 'linear':
            H = self.hist[-12:]
            s = np.array([h[0] for h in H], float)
            P = np.array([[h[1], h[2]] for h in H])
            if len(H) < 2 or s[-1] - s[0] < 1:
                return P[-1]
            A = np.stack([np.ones_like(s), s - s[-1]], 1)
            cx = np.linalg.lstsq(A, P[:, 0], rcond=None)[0]
            cy = np.linalg.lstsq(A, P[:, 1], rcond=None)[0]
            dt = step - s[-1]
            return np.array([cx[0] + cx[1] * dt, cy[0] + cy[1] * dt])
        return None

    def phase_of(self, xy, m_guess, window=25):
        """Moving time near m_guess whose loop point is closest to xy; (m, dist)."""
        best = None
        for dm in np.arange(-window, window + 0.25, 0.25):
            m = m_guess + dm
            d = float(np.linalg.norm(self.L(m) - xy))
            if best is None or d < best[1]:
                best = (m, d)
        return best

    # ---------------- events ----------------
    def mark_stopped(self, xy, step):
        """Target observed stationary (cube on it) at xy."""
        if self.mode != 'loop':
            return
        m, d = self.phase_of(np.asarray(xy), self.m_at(step), window=40)
        self.m_ref = m
        self.s_ref = step
        self.moving = False
        self.stop_xy = np.asarray(xy, float)
        log(f'[track] stopped at m={m:.1f} (loop dist {d:.4f})')

    def mark_resumed(self, step):
        if self.mode != 'loop' or self.moving:
            return
        self.s_ref = step
        self.moving = True
        log(f'[track] resumed at step {step} m={self.m_ref:.1f}')

    def resync(self, since_step, max_corr=20):
        """Correct the phase from recent detections (after since_step) while moving."""
        if self.mode != 'loop' or not self.moving:
            return None
        res = []
        for h in self.hist:
            if h[0] < since_step:
                continue
            xy = np.array([h[1], h[2]])
            mp = self.m_at(h[0])
            m, d = self.phase_of(xy, mp, window=max_corr)
            if d < LOOP_TOL and abs(m - mp) < max_corr - 0.5:
                res.append(m - mp)
        if len(res) < 2:
            return None
        corr = float(np.median(res))
        self.m_ref += corr
        log(f'[track] resync n={len(res)} corr={corr:+.2f} steps')
        return corr

    def n_good_since(self, step):
        return sum(1 for h in self.hist if h[0] >= step)

    def relock(self, since_step, prior_window=30):
        """Global phase search over one period using detections since since_step (target moving).
        Prefers offsets within prior_window of the current model unless a global one is clearly better."""
        if self.mode != 'loop' or not self.moving:
            return None
        H = [h for h in self.hist if h[0] >= since_step]
        if len(H) < 3:
            return None
        steps = np.array([h[0] for h in H], float)
        obs = np.array([[h[1], h[2]] for h in H])
        T = self.T
        base = self.m_at(int(steps[0]))
        best_g, best_p = None, None
        for off in np.arange(-T / 2, T / 2, 0.5):
            m = base + off + (steps - steps[0])
            pts = np.array([self.L(mm) for mm in m])
            cost = float(np.mean(np.minimum(np.linalg.norm(pts - obs, axis=1), 0.03)))
            if best_g is None or cost < best_g[0]:
                best_g = (cost, off)
            if abs(off) <= prior_window and (best_p is None or cost < best_p[0]):
                best_p = (cost, off)
        pick = best_p if best_p[0] < best_g[0] + 0.004 else best_g
        if pick[0] > 0.012:
            log(f'[track] relock rejected cost={pick[0]:.4f} n={len(H)}')
            return None
        self.m_ref += pick[1]
        log(f'[track] relock n={len(H)} off={pick[1]:+.1f} cost={pick[0]:.4f} (global {best_g[1]:+.1f}/{best_g[0]:.4f})')
        return pick[1]
