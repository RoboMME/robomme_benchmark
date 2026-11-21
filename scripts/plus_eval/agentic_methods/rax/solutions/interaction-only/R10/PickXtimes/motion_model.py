import math
import numpy as np


class TargetModel:
    """Fit p(t) = c + a cos(w t) + b sin(w t) (static if w=0) to (step, x, y) observations."""

    def __init__(self):
        self.obs = []  # (t, x, y, weight)
        self.params = None
        self.rms = None

    def add(self, t, xy, w=1.0):
        self.obs.append((float(t), float(xy[0]), float(xy[1]), float(w)))

    def _fit_w(self, T, X, W, w):
        if w == 0:
            A = np.ones((len(T), 1))
        else:
            A = np.stack([np.ones_like(T), np.cos(w * T), np.sin(w * T)], 1)
        Aw = A * W[:, None]
        coef, *_ = np.linalg.lstsq(Aw, X * W[:, None], rcond=None)
        res = X - A @ coef
        return coef, res

    def fit(self, min_period=30, max_period=600):
        if len(self.obs) < 3:
            return None
        O = np.array(self.obs)
        T, X, W = O[:, 0], O[:, 1:3], O[:, 3]
        keep = np.ones(len(T), bool)
        best = None
        for it in range(3):
            Tk, Xk, Wk = T[keep], X[keep], W[keep]
            span = Tk.max() - Tk.min()
            cands = [0.0]
            if span > 10:
                for P in np.geomspace(min_period, max_period, 400):
                    cands.append(2 * math.pi / P)
            best = None
            for w in cands:
                coef, res = self._fit_w(Tk, Xk, Wk, w)
                rms = float(np.sqrt(np.mean(np.sum(res ** 2, 1))))
                # penalise moving models slightly so static wins when equal
                score = rms + (0.0 if w == 0 else 0.0005)
                if best is None or score < best[0]:
                    best = (score, w, coef, rms)
            # refine w locally
            if best[1] > 0:
                w0 = best[1]
                for w in np.linspace(w0 * 0.97, w0 * 1.03, 61):
                    coef, res = self._fit_w(Tk, Xk, Wk, w)
                    rms = float(np.sqrt(np.mean(np.sum(res ** 2, 1))))
                    if rms + 0.0005 < best[0]:
                        best = (rms + 0.0005, w, coef, rms)
            _, w, coef, rms = best
            # outlier rejection over all points
            pred = self.predict_with(w, coef, T)
            err = np.linalg.norm(pred - X, axis=1)
            new_keep = err < max(0.012, 3 * rms)
            if new_keep.sum() < 3 or (new_keep == keep).all():
                keep = new_keep if new_keep.sum() >= 3 else keep
                break
            keep = new_keep
        self.params = (best[1], best[2])
        self.rms = best[3]
        self.n_used = int(keep.sum())
        return self.params

    @staticmethod
    def predict_with(w, coef, t):
        t = np.atleast_1d(np.asarray(t, float))
        if w == 0:
            return np.repeat(coef[0:1], len(t), 0)
        A = np.stack([np.ones_like(t), np.cos(w * t), np.sin(w * t)], 1)
        return A @ coef

    def predict(self, t):
        w, coef = self.params
        return self.predict_with(w, coef, t)[0]

    @property
    def moving(self):
        return self.params is not None and self.params[0] > 0

    def period(self):
        if not self.moving:
            return None
        return 2 * math.pi / self.params[0]
