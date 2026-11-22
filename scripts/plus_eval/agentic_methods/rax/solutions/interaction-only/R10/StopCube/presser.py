"""Timing logic: count cube passes over target and decide when to fire the press."""
import numpy as np

def k_circle(ts, Ps, T, t_now):
    """Fit a circle to recent positions; return (steps until closest approach to T, miss distance, radius)
    or None when the path is nearly straight (radius > 0.4 m) or the fit is degenerate."""
    ts = np.asarray(ts, float); P = np.asarray(Ps, float); T = np.asarray(T, float)
    A = np.c_[2 * P, np.ones(len(P))]; b = (P ** 2).sum(1)
    sol, res, rk, _ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]; r2 = sol[2] + c @ c
    if rk < 3 or r2 <= 0:
        return None
    r = float(np.sqrt(r2))
    if r > 0.4 or r < 0.01:
        return None
    ang = np.unwrap(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))
    fit = np.polyfit(ts[-4:], ang[-4:], 1); w = fit[0]
    if abs(w) < 1e-3:
        return None
    th_now = np.polyval(fit, t_now)
    thT = np.arctan2(T[1] - c[1], T[0] - c[0])
    dth = (thT - th_now + np.pi) % (2 * np.pi) - np.pi
    k = dth / w
    if k < -1:
        k += 2 * np.pi / abs(w)
    return float(k), float(abs(np.linalg.norm(T - c) - r)), r

class PassCounter:
    def __init__(self, r_in=0.035, r_out=0.045, latency=2.0, press_on=5, done=0, margin=0.3):
        self.margin = margin
        self.r_in, self.r_out, self.latency, self.press_on = r_in, r_out, latency, press_on
        self.done = done          # completed passes
        self.inzone = False
        self.hist = []            # (step, xy)
        self.entries = []
        self.min_d = None

    def update(self, step, cube_xy, target_xy):
        """Return (fire, info)."""
        T = np.asarray(target_xy, float)
        if cube_xy is None:
            return False, {"why": "no cube"}
        p = np.asarray(cube_xy, float)
        self.hist.append((step, p))
        d = float(np.linalg.norm(p - T))
        if not self.inzone and d < self.r_in:
            self.inzone = True; self.entries.append(step); self.min_d = d
            # missed-pass recovery: a gap of ~2+ lap periods means a pass went unseen (e.g. occlusion)
            if len(self.entries) >= 3:
                gaps = np.diff(self.entries[:-1])
                med = float(np.median(gaps)) if len(gaps) else 0.0
                gap = self.entries[-1] - self.entries[-2]
                if med >= 8 and gap > 1.6 * med:
                    extra = int(round(gap / med)) - 1
                    self.done += max(0, extra); self.recovered = getattr(self, "recovered", 0) + extra
        elif self.inzone:
            self.min_d = min(self.min_d, d)
            if d > self.r_out:
                self.inzone = False; self.done += 1
        info = {"d": round(d, 4), "done": self.done, "inzone": self.inzone}
        if len(self.hist) < 3:
            return False, info
        (s0, p0), (s2, p2) = self.hist[-3], self.hist[-1]
        if s2 - s0 <= 0:
            return False, info
        v = (p2 - p0) / (s2 - s0)
        sp = float(np.linalg.norm(v)); info["speed"] = round(sp, 4)
        if sp < 1e-3:
            return False, info
        k = float(-(p - T) @ v / (sp * sp))       # linear: steps until closest approach
        dmin = float(np.linalg.norm(p + k * v - T))
        info.update(klin=round(k, 2))
        # circle fit over the last 6 observations (paths are closed loops; small loops curve fast)
        circ = None
        if len(self.hist) >= 6 and self.hist[-1][0] - self.hist[-6][0] <= 9:
            try:
                circ = k_circle([h[0] for h in self.hist[-6:]], [h[1] for h in self.hist[-6:]], T, step)
            except Exception:
                circ = None
        if circ is not None:
            k, dmin, rad = circ
            info.update(k=round(k, 2), dmin=round(dmin, 4), circ_r=round(rad, 3))
        # quadratic fit over the last 5 observations (actual step times) if contiguous enough
        if len(self.hist) >= 5 and self.hist[-1][0] - self.hist[-5][0] <= 7:
            try:
                ts = np.array([h[0] for h in self.hist[-5:]], float) - step
                Ps = np.array([h[1] for h in self.hist[-5:]])
                A = np.stack([ts ** 2, ts, np.ones_like(ts)], 1)
                c = np.linalg.lstsq(A, Ps, rcond=None)[0]
                ks = np.linspace(-1.0, 8.0, 181)
                pos = (ks[:, None] ** 2) * c[0] + ks[:, None] * c[1] + c[2]
                dd = np.linalg.norm(pos - T, axis=1)
                j = int(np.argmin(dd))
                if circ is None and 0 < j < len(ks) - 1:       # interior minimum only
                    k, dmin = float(ks[j]), float(dd[j])
                    info["quad"] = True
            except Exception:
                pass
        if "k" not in info:
            info.update(k=round(k, 2), dmin=round(dmin, 4))
        # which pass is upcoming / ongoing: if in zone, current pass = done+1
        current = self.done + 1
        fire = (current == self.press_on and dmin < 0.03 and -0.5 <= k <= self.latency + self.margin)
        info["pass"] = current
        return fire, info
