"""Target trajectory model: per-step detections, analog (previous-lap) forecasting, local fallback."""
import numpy as np
from robo import log
from tracking import detect_target_cam


class TargetModel:
    def __init__(self):
        self.hist = {}  # step -> (x, y)
        self.last_any = None
        self.max_gap = 4
        self.cube_xy = None  # resting cube position (occludes the target when on it)
        self.ever_moved = False

    def update(self, R):
        try:
            step = int(R.meta.get("step_id", 0))
            d = detect_target_cam(R, "front")
            if d is None or d[1] < 20:
                return
            p = d[0]
            self.last_any = (step, float(p[0]), float(p[1]))
            tcp = np.array(R.state[:3])
            arm_clear = not self.occluded(R, p, tcp)
            cube_clear = self.cube_xy is None or np.linalg.norm(np.asarray(self.cube_xy[:2]) - p[:2]) > 0.07
            if arm_clear and cube_clear:
                self.hist[step] = (float(p[0]), float(p[1]))
                sp = self.recent_speed()
                if sp is not None and sp > 0.0015 and len(self.hist) > 6:
                    self.ever_moved = True
        except Exception as e:
            log("target model update error", e)

    def occluded(self, R, p, tcp):
        """True if the gripper column (TCP-0.03 .. TCP+0.30, radius 0.07) may cut the front
        camera ray to target point p."""
        cam = np.array(R.meta["camera_poses"]["front"]["position_m"], float)
        p = np.asarray(p, float)
        for h in np.linspace(max(tcp[2] - 0.03, 0.0), tcp[2] + 0.30, 14):
            if h >= cam[2]:
                break
            a = h / cam[2] if cam[2] > 0 else 0
            xy = p[:2] + (cam[:2] - p[:2]) * a
            if np.linalg.norm(xy - tcp[:2]) < 0.07:
                return True
        return False

    def fresh(self, step, n=4, within=6):
        ks = [k for k in self.hist if k >= step - within]
        return len(ks) >= n

    def arr(self):
        ks = sorted(self.hist)
        return np.array([[k, *self.hist[k]] for k in ks], float) if ks else np.zeros((0, 3))

    def recent_speed(self, n=6):
        A = self.arr()
        if len(A) < 2:
            return None
        H = A[-n:]
        dt = H[-1, 0] - H[0, 0]
        return float(np.linalg.norm(H[-1, 1:] - H[0, 1:]) / dt) if dt > 0 else None

    def _interp(self, A, tq):
        t = A[:, 0]
        if tq < t[0] or tq > t[-1]:
            return None
        i = int(np.searchsorted(t, tq))
        if i < len(t) and t[i] == tq:
            return A[i, 1:]
        i = min(max(i, 1), len(t) - 1)
        if t[i] - t[i - 1] > 4:
            return None
        w = (tq - t[i - 1]) / (t[i] - t[i - 1])
        return A[i - 1, 1:] * (1 - w) + A[i, 1:] * w

    def grid(self):
        A = self.arr()
        if len(A) < 2:
            return None
        t = A[:, 0]
        g = np.arange(int(t[0]), int(t[-1]) + 1)
        X = np.stack([np.interp(g, t, A[:, 1]), np.interp(g, t, A[:, 2])], 1)
        i = np.clip(np.searchsorted(t, g), 1, len(t) - 1)
        gap = t[i] - t[i - 1]
        exact = np.isin(g, t)
        valid = exact | (gap <= self.max_gap)
        return g, X, valid

    def analog(self, horizon, win=(0, 2, 4, 6), min_lag=25):
        """Forecast the position `horizon` steps after the latest sample from the best-matching
        earlier stretch of the trajectory. Returns (xy, err, s) or None."""
        G = self.grid()
        if G is None:
            return None
        g, X, valid = G
        n = len(g)
        if n < 30:
            return None
        win = np.array(win)
        now = n - 1
        if not valid[now - win].all():
            return None
        cur = X[now - win]
        if np.linalg.norm(cur[0] - cur[-1]) < 0.004:
            return None
        hz = int(round(horizon))
        cand = np.arange(win[-1], now - min_lag + 1)
        if hz > 0:
            cand = cand[cand + hz <= now]
        if len(cand) == 0:
            return None
        idx = cand[:, None] - win[None, :]
        ok = valid[idx].all(1) & valid[np.clip(cand + hz, 0, now)]
        if not ok.any():
            return None
        cand = cand[ok]; idx = idx[ok]
        e = np.linalg.norm(X[idx] - cur[None], axis=2).mean(1)
        j = int(np.argmin(e))
        if e[j] > 0.004:
            return None
        s = cand[j]
        fut = X[min(s + hz, now)] + (cur[0] - X[s])
        return fut, float(e[j]), int(g[s])

    def predict(self, step):
        """Target xy at absolute step `step` and a label."""
        A = self.arr()
        if len(A) == 0:
            if self.last_any is not None:
                return np.array(self.last_any[1:]), "last_any"
            return None, "none"
        t_now = A[-1, 0]
        h = step - t_now
        sp = self.recent_speed()
        if sp is not None and sp < 0.0008 and len(A) >= 4 and t_now - A[-4, 0] < 15:
            return A[-4:, 1:].mean(0), "static"
        if h <= 0:
            p = self._interp(A, step)
            if p is not None:
                return p, "hist"
        a = self.analog(h)
        if a is not None:
            return a[0], "analog"
        H = A[A[:, 0] >= t_now - 8]
        if len(H) >= 3 and H[-1, 0] - H[0, 0] >= 2:
            tt = H[:, 0] - t_now
            v = np.array([np.polyfit(tt, H[:, k], 1)[0] for k in (1, 2)])
            sp = np.linalg.norm(v)
            if sp > 0.012:
                v = v / sp * 0.012
            hh = min(max(h, 0), 12)
            return A[-1, 1:] + v * hh, "poly"
        return A[-1, 1:], "last"

    def has_lap(self):
        A = self.arr()
        if len(A) < 20:
            return False
        sp = self.recent_speed()
        if sp is not None and sp < 0.0008:
            return True
        return self.analog(10) is not None
