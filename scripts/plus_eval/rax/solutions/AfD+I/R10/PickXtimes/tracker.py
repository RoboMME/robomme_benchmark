"""Moving-target tracker: full-ring measurements, period model, prediction."""
import math
import numpy as np
import robo, percep
from robo import log

class TargetTracker:
    def __init__(self):
        self.hist = []          # (step, x, y)
        self.D0 = None          # nominal ring diameter (m)
        self.period = None
        self.period_err = None
        self._last_fit_n = 0

    def measure(self, meta, accept_partial=False):
        step = meta.get("step_id", 0)
        best = None
        for cam in ("front", "wrist"):
            rgb = meta.get("rgb", {}).get(cam)
            if rgb is None or cam not in meta.get("depth_m", {}):
                continue
            mk = robo.mask_target(rgb)
            if mk.sum() < 12:
                continue
            P = percep.world_cloud(meta, cam)
            ok = mk & percep._inws(P) & (P[..., 2] < 0.03) & (P[..., 2] > -0.02)
            comps = robo.components(ok, 4)
            if not comps:
                continue
            n0, (vs0, us0) = comps[0]
            c0 = np.nanmedian(P[vs0, us0][:, :2], 0)
            pts = [P[vs, us] for n, (vs, us) in comps
                   if np.linalg.norm(np.nanmedian(P[vs, us][:, :2], 0) - c0) < 0.06]
            pts = np.concatenate(pts)
            lo, hi = pts[:, :2].min(0), pts[:, :2].max(0)
            ext = hi - lo
            if self.D0 is None:
                if len(pts) > 60 and abs(ext[0] - ext[1]) < 0.006:
                    self.D0 = float(ext.mean())
                    log(f"tracker: ring diameter {self.D0:.3f} from {cam}")
                else:
                    continue
            full = ext.min() > 0.85 * self.D0 and ext.max() < 1.25 * self.D0
            if not full and not accept_partial:
                continue
            xy = (lo + hi) / 2
            score = len(pts) * (2 if cam == "wrist" else 1)
            if best is None or score > best[1]:
                best = (xy, score, cam)
        if best is not None:
            if not self.hist or self.hist[-1][0] != step:
                self.hist.append((step, float(best[0][0]), float(best[0][1])))
            return best[0]
        return None

    # ----- model -----
    def _arr(self):
        return np.array(self.hist, float) if self.hist else np.zeros((0, 3))

    def moving(self):
        H = self._arr()
        if len(H) < 3:
            return None
        return float(np.ptp(H[:, 1]) + np.ptp(H[:, 2])) > 0.01

    def fit_period(self):
        H = self._arr()
        if len(H) < 20 or len(H) == self._last_fit_n:
            return self.period
        self._last_fit_n = len(H)
        t, x, y = H[:, 0], H[:, 1], H[:, 2]
        span = t[-1] - t[0]
        best = None
        for P in np.arange(30, min(600, span - 10) + 0.5, 0.5):
            m = t - P >= t[0]
            if m.sum() < 10:
                continue
            xi = np.interp(t[m] - P, t, x); yi = np.interp(t[m] - P, t, y)
            # only use pairs where interpolation source is dense (gap <= 6 steps)
            idx = np.searchsorted(t, t[m] - P)
            idx = np.clip(idx, 1, len(t) - 1)
            gap = t[idx] - t[idx - 1]
            g = gap <= 8
            if g.sum() < 8:
                continue
            e = np.hypot(xi - x[m], yi - y[m])[g]
            err = float(np.median(e))
            if best is None or err < best[1]:
                best = (P, err, int(g.sum()))
        if best is not None and best[1] < 0.004:
            # refine to the smallest fitting multiple? keep best
            self.period, self.period_err = best[0], best[1]
        return self.period

    def predict(self, step):
        H = self._arr()
        if len(H) == 0:
            return None
        t, x, y = H[:, 0], H[:, 1], H[:, 2]
        if not self.moving():
            return np.array([np.median(x[-10:]), np.median(y[-10:])])
        P = self.fit_period()
        if P is not None:
            vals = []
            ts = step
            while ts > t[-1]:
                ts -= P
            while ts >= t[0]:
                idx = int(np.clip(np.searchsorted(t, ts), 1, len(t) - 1))
                if t[idx] - t[idx - 1] <= 10 and t[idx - 1] <= ts <= t[idx]:
                    vals.append([np.interp(ts, t, x), np.interp(ts, t, y)])
                ts -= P
            if vals:
                return np.median(np.array(vals), 0)
        # constant velocity from recent measurements
        rec = H[H[:, 0] >= t[-1] - 12]
        if len(rec) >= 2 and rec[-1, 0] > rec[0, 0]:
            v = (rec[-1, 1:] - rec[0, 1:]) / (rec[-1, 0] - rec[0, 0])
            dt = min(step - t[-1], 8)
            return rec[-1, 1:] + v * dt
        return np.array([x[-1], y[-1]])


class PathModel:
    """Closed path of the moving target learned from one clean period of history."""
    def __init__(self, tracker):
        self.tr = tracker
        self.P = None
        self.ref = None   # (tau array 0..P, x, y)

    def build(self):
        T = self.tr
        P = T.fit_period()
        if P is None:
            return False
        H = T._arr()
        t = H[:, 0]
        # use the latest window of length P with dense coverage
        t_end = t[-1]
        sel = H[t >= t_end - P - 1e-6]
        if len(sel) < 10:
            return False
        tau = np.arange(0, P, 1.0)
        ts = t_end - P + tau
        x = np.interp(ts, t, H[:, 1]); y = np.interp(ts, t, H[:, 2])
        self.P, self.ref = P, (tau, x, y)
        self.t0 = t_end - P  # path(tau) at absolute time t0 + tau (before any pause)
        log(f"path model: period {P}, err {T.period_err:.4f}, x[{x.min():.3f},{x.max():.3f}] y[{y.min():.3f},{y.max():.3f}]")
        return True

    def at(self, tau):
        taus, x, y = self.ref
        tt = np.mod(tau, self.P)
        xs = np.append(x, x[0]); ys = np.append(y, y[0]); tg = np.append(taus, self.P)
        return np.array([np.interp(tt, tg, xs), np.interp(tt, tg, ys)])

    def phase_of(self, meas):
        """meas: list of (t, x, y) recent fresh measurements (>=1). Returns tau at last meas time."""
        taus = np.arange(0, self.P, 0.5)
        tl = meas[-1][0]
        cost = np.zeros(len(taus))
        for (tm, xm, ym) in meas:
            q = self.at(taus - (tl - tm))
            cost += np.hypot(q[0] - xm, q[1] - ym)
        i = int(np.argmin(cost))
        return taus[i], cost[i] / len(meas)

    def predict(self, step, meas):
        tau, c = self.phase_of(meas)
        return self.at(tau + (step - meas[-1][0])), c
