"""Visit counting and arrival prediction for the StopCube cube.

Works in any planar metric frame (world metres or pixels): positions are (x, y) per
integer control step; the target is a point; `thr` is the visit distance threshold.
"""
import numpy as np


def _interp_track(track, steps):
    pts = [track[s] for s in steps]
    return np.array(pts, float)


class VisitModel:
    def __init__(self, target_xy, thr, log=print):
        self.target = np.array(target_xy, float)
        self.thr = float(thr)
        self.track = {}           # step -> np.array([x, y])
        self.visits = []          # dicts: t (refined), step (int min), dmin
        self.log = log
        self._last_visit_step = -999
        self.near_passes = []     # local minima above thr (not counted)

    def set_target(self, target_xy, thr=None):
        self.target = np.array(target_xy, float)
        if thr is not None:
            self.thr = float(thr)

    def dist(self, s):
        p = self.track.get(s)
        return None if p is None else float(np.hypot(*(p - self.target)))

    def add(self, s, xy):
        if xy is None:
            self.track.pop(s, None)
        else:
            self.track[s] = np.array(xy, float)

    # ---------------------------------------------------------- visit detection
    def recount(self, now):
        """Recompute visits from the whole track (robust to revised detections)."""
        steps = sorted(k for k in self.track if k <= now)
        visits, near = [], []
        if len(steps) < 3:
            self.visits = visits
            return visits
        d = {s: self.dist(s) for s in steps}
        i = 0
        n = len(steps)
        while i < n:
            s = steps[i]
            if d[s] < self.thr:
                # a contiguous in-zone run (allow 2-step gaps of missing detections)
                j = i
                while j + 1 < n and steps[j + 1] - steps[j] <= 3 and d[steps[j + 1]] < self.thr * 1.15:
                    j += 1
                run = steps[i:j + 1]
                smin = min(run, key=lambda k: d[k])
                # the run must have ended (cube left) or the min must be followed by increase
                ended = j + 1 < n or now - run[-1] >= 3
                after = [k for k in steps if k > smin and k <= now]
                rising = len(after) >= 2 and d[after[-1]] > d[smin] + 0.25 * self.thr
                # entering from step 0 already in zone and leaving: does not count as reaching
                started_inside = (run[0] == steps[0]) and run[0] <= 1 and smin == run[0]
                if (ended or rising) and not started_inside:
                    visits.append({"step": smin, "t": self.refine_tmin(smin), "dmin": d[smin],
                                   "run": (run[0], run[-1])})
                i = j + 1
            else:
                i += 1
        self.visits = visits
        return visits

    def refine_tmin(self, smin, half=4):
        """Sub-step time of closest approach from a local quadratic fit of the trajectory."""
        ks = [k for k in range(smin - half, smin + half + 1) if k in self.track]
        if len(ks) < 4:
            return float(smin)
        t = np.array(ks, float) - smin
        P = np.array([self.track[k] for k in ks])
        A = np.stack([np.ones_like(t), t, t * t], 1)
        try:
            cx = np.linalg.lstsq(A, P[:, 0], rcond=None)[0]
            cy = np.linalg.lstsq(A, P[:, 1], rcond=None)[0]
        except Exception:
            return float(smin)
        tt = np.linspace(-1.5, 1.5, 301)
        x = cx[0] + cx[1] * tt + cx[2] * tt * tt
        y = cy[0] + cy[1] * tt + cy[2] * tt * tt
        dd = np.hypot(x - self.target[0], y - self.target[1])
        return float(smin + tt[int(np.argmin(dd))])

    # ---------------------------------------------------------- prediction
    def eta_kinematic(self, s, n_fit=5, horizon=12.0):
        ks = [k for k in range(s - 8, s + 1) if k in self.track][-n_fit:]
        if len(ks) < 3 or ks[-1] != s:
            return None
        t = np.array(ks, float) - s
        P = np.array([self.track[k] for k in ks])
        deg = 2 if len(ks) >= 4 else 1
        A = np.stack([t ** i for i in range(deg + 1)], 1)
        cx = np.linalg.lstsq(A, P[:, 0], rcond=None)[0]
        cy = np.linalg.lstsq(A, P[:, 1], rcond=None)[0]
        # limit curvature extrapolation: blend quadratic and linear beyond 4 steps
        tt = np.linspace(0, horizon, int(horizon * 20) + 1)
        def ev(c, tq):
            return sum(c[i] * tq ** i for i in range(len(c)))
        x, y = ev(cx, tt), ev(cy, tt)
        dd = np.hypot(x - self.target[0], y - self.target[1])
        # first local minimum
        for i in range(1, len(dd) - 1):
            if dd[i] <= dd[i - 1] and dd[i] <= dd[i + 1]:
                return {"eta": float(tt[i]), "dmin": float(dd[i]), "method": "kin"}
        if dd[0] <= dd[-1]:
            return {"eta": 0.0, "dmin": float(dd[0]), "method": "kin"}
        return {"eta": float(tt[-1]), "dmin": float(dd[-1]), "method": "kin", "beyond": True}

    def eta_profile(self, s, lookback=16):
        """Match the current position to approach segments of previous visits."""
        p = self.track.get(s)
        pp = self.track.get(s - 1)
        if p is None or not self.visits:
            return None
        vel = None if pp is None else p - pp
        speed = None if vel is None else float(np.hypot(*vel))
        ests = []
        for v in self.visits:
            tk = v["t"]
            ks = [k for k in range(int(np.floor(tk)) - lookback, int(np.ceil(tk)) + 1) if k in self.track]
            if len(ks) < 4 or ks[0] >= s:
                continue
            best = None
            for a, b in zip(ks[:-1], ks[1:]):
                if b - a > 2:
                    continue
                pa, pb = self.track[a], self.track[b]
                seg = pb - pa
                L2 = float(seg @ seg)
                if L2 < 1e-12:
                    continue
                lam = float(np.clip((p - pa) @ seg / L2, 0, 1))
                q = pa + lam * seg
                dist = float(np.hypot(*(p - q)))
                tau = a + lam * (b - a)
                if vel is not None and speed > 1e-9:
                    cosang = float(vel @ seg) / (speed * np.sqrt(L2))
                    if cosang < 0.5:
                        continue
                if best is None or dist < best[0]:
                    best = (dist, tau, np.sqrt(L2) / (b - a))
            if best is None:
                continue
            dist, tau, seg_speed = best
            eta = tk - tau
            if eta < -0.5:
                continue
            # speed ratio correction (cube may speed up / slow down between cycles)
            ratio = 1.0
            if speed is not None and seg_speed > 1e-9:
                ratio = float(np.clip(seg_speed / max(speed, 1e-9), 0.8, 1.25))
            ests.append({"eta": eta * ratio, "match": dist, "visit_t": tk, "ratio": ratio})
        if not ests:
            return None
        ests.sort(key=lambda e: e["match"])
        good = [e for e in ests if e["match"] < max(1.5 * ests[0]["match"], 0.15 * self.thr)][:3]
        eta = float(np.mean([e["eta"] for e in good]))
        return {"eta": eta, "match": good[0]["match"], "n": len(good), "method": "prof",
                "spread": float(np.ptp([e["eta"] for e in good])) if len(good) > 1 else 0.0}
