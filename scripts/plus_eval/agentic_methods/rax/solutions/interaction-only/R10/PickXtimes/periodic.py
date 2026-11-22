"""Target history from every observation, period estimation and periodic prediction."""
import numpy as np
from robo import log
from tracking import detect_target_cam


class PeriodicTracker:
    def __init__(self):
        self.hist = {}      # step -> (x, y, clean)
        self.period = None
        self.static = False
        self._last_est = -999

    def update(self, R):
        try:
            step = int(R.meta.get("step_id", 0))
            d = detect_target_cam(R, "front")
            if d is None or d[1] < 20:
                return
            p = d[0]
            tcp = np.array(R.state[:3])
            clean = bool(np.linalg.norm(tcp[:2] - p[:2]) > 0.12 or tcp[2] > 0.32)
            old = self.hist.get(step)
            if old is None or (clean and not old[2]):
                self.hist[step] = (float(p[0]), float(p[1]), clean)
        except Exception as e:
            log("tracker update error", e)

    def clean_array(self):
        ks = sorted(k for k, v in self.hist.items() if v[2])
        if not ks:
            return np.zeros((0, 3))
        return np.array([[k, self.hist[k][0], self.hist[k][1]] for k in ks], float)

    def last(self):
        A = self.clean_array()
        if len(A) == 0:
            ks = sorted(self.hist)
            if not ks:
                return None
            return np.array([ks[-1], *self.hist[ks[-1]][:2]])
        return A[-1]

    def estimate(self):
        A = self.clean_array()
        if len(A) < 10:
            return
        span = A[-1, 0] - A[0, 0]
        mv = np.linalg.norm(A[:, 1:3] - A[:, 1:3].mean(0), axis=1)
        if span >= 15 and mv.max() < 0.004:
            self.static = True
            return
        self.static = False
        if A[-1, 0] - self._last_est < 10 and self.period is not None:
            return
        self._last_est = A[-1, 0]
        t = A[:, 0]; X = A[:, 1:3]
        best = None
        for P in range(20, int(span) - 5):
            # compare each sample with interpolated position P steps earlier
            m = (t - P) >= t[0]
            if m.sum() < 8:
                continue
            tq = t[m] - P
            xi = np.interp(tq, t, X[:, 0]); yi = np.interp(tq, t, X[:, 1])
            # only trust interpolation where samples near tq exist
            idx = np.searchsorted(t, tq)
            idx = np.clip(idx, 1, len(t) - 1)
            gap = t[idx] - t[idx - 1]
            ok = gap <= 6
            if ok.sum() < 8:
                continue
            err = np.hypot(X[m][ok, 0] - xi[ok], X[m][ok, 1] - yi[ok])
            e = float(np.median(err))
            if best is None or e < best[1] - 1e-4:
                best = (P, e, int(ok.sum()))
            if e < 0.0025 and ok.sum() >= 12:
                best = (P, e, int(ok.sum()))
                break
        if best is not None and best[1] < 0.004 and best[2] >= 12:
            if self.period != best[0]:
                log("tracker: period", best[0], "err", round(best[1], 4), "pairs", best[2])
            self.period = best[0]

    def predict(self, step):
        """Predicted target xy at `step` (may be future). Returns (xy, confidence str)."""
        A = self.clean_array()
        if len(A) == 0:
            l = self.last()
            return (None if l is None else l[1:3]), "none"
        if self.static:
            return A[-8:, 1:3].mean(0), "static"
        if self.period:
            P = self.period
            t = A[:, 0]
            cands = []
            k = int(np.ceil((step - t[-1]) / P))
            for j in range(max(k, 0), max(k, 0) + 40):
                tq = step - j * P
                if tq < t[0]:
                    break
                if tq > t[-1]:
                    continue
                i = np.searchsorted(t, tq)
                i = min(max(i, 1), len(t) - 1)
                if t[i] - t[i - 1] <= 6:
                    w = (tq - t[i - 1]) / max(t[i] - t[i - 1], 1e-9)
                    cands.append(A[i - 1, 1:3] * (1 - w) + A[i, 1:3] * w)
                if len(cands) >= 3:
                    break
            if cands:
                return np.mean(cands, 0), "periodic"
        # local polynomial extrapolation from recent clean samples
        H = A[A[:, 0] >= A[-1, 0] - 12]
        if len(H) >= 4 and H[-1, 0] - H[0, 0] >= 3:
            tt = H[:, 0] - H[-1, 0]
            deg = 2 if len(H) >= 7 else 1
            out = [np.polyval(np.polyfit(tt, H[:, k], deg), step - H[-1, 0]) for k in (1, 2)]
            return np.array(out), "poly"
        return A[-1, 1:3], "last"

    def speed(self):
        A = self.clean_array()
        if len(A) < 3:
            return None
        H = A[-10:]
        dt = H[-1, 0] - H[0, 0]
        return float(np.linalg.norm(H[-1, 1:3] - H[0, 1:3]) / dt) if dt > 0 else None
