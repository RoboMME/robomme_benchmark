"""Live target tracker: detects the (possibly moving) target disc on every observation."""
import math
import numpy as np
from perception import world_points, color_mask, components

DIAM_MIN = 0.062   # bbox extent (m) above which a detection is treated as complete (disc ~0.078)


def detect_target(meta, imgs, cam):
    if (cam, "depth") not in imgs:
        return None
    rgb = imgs[(cam, "rgb")]
    P = world_points(meta, imgs, cam)
    z = P[..., 2]
    fin = np.isfinite(z)
    zt = z[fin & (z > -0.05) & (z < 0.05)]
    tz = float(np.median(zt)) if zt.size else 0.0
    m = color_mask(rgb, "purple", smin=0.25) & fin & (z < tz + 0.03) & (z > tz - 0.02)
    if m.sum() < 5:
        return None
    cs = components(m, min_px=3)
    n, ys, xs = cs[0]
    c0 = np.nanmedian(P[ys, xs], 0)
    ally, allx = np.nonzero(m)
    pts = P[ally, allx]
    sel = np.linalg.norm(pts[:, :2] - c0[:2], axis=1) < 0.075
    pts = pts[sel]
    if len(pts) < 5:
        return None
    ex, ey = float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))
    c = np.array([(pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 1].min() + pts[:, 1].max()) / 2])
    complete = ex >= DIAM_MIN and ey >= DIAM_MIN and ex < 0.095 and ey < 0.095
    return {"xy": c, "z": float(np.nanmax(pts[:, 2])), "npx": int(len(pts)), "complete": bool(complete),
            "ext": (ex, ey), "cam": cam, "table_z": tz}


class Tracker:
    def __init__(self, log=print):
        self.hist = []      # (step, xy, cam) of complete detections
        self.last_any = None
        self.disc_z = None
        self.ever_moving = False
        self.log = log

    def update(self, meta, imgs):
        step = int(meta.get("step_id", 0))
        best = None
        for cam in ("wrist", "front"):
            try:
                d = detect_target(meta, imgs, cam)
            except Exception:
                d = None
            if d is None:
                continue
            self.last_any = (step, d)
            if d["complete"] and best is None:
                best = d
        if best is not None:
            if self.hist and self.hist[-1][0] == step:
                return best
            self.hist.append((step, best["xy"].copy(), best["cam"]))
            if self.disc_z is None or best["cam"] == "wrist":
                self.disc_z = best["z"]
            if not self.ever_moving:
                for s0, p0, _ in reversed(self.hist):
                    if step - s0 > 60:
                        break
                    if np.linalg.norm(p0 - best["xy"]) > 0.02:
                        self.ever_moving = True
                        self.log(f"tracker: target is moving (step {step})")
                        break
            if len(self.hist) > 3000:
                self.hist = self.hist[-2000:]
        return best

    def recent(self, now, window):
        return [(s, p) for s, p, _ in self.hist if now - s <= window]

    def speed(self, now, window=20):
        r = self.recent(now, window)
        if len(r) < 2 or r[-1][0] - r[0][0] < 3:
            return None
        (s0, p0), (s1, p1) = r[0], r[-1]
        return float(np.linalg.norm(p1 - p0) / (s1 - s0))

    def last_age(self, now):
        return (now - self.hist[-1][0]) if self.hist else 10 ** 6

    def predict(self, now, ahead):
        """Predicted target xy at step now+ahead from recent complete detections."""
        r = self.recent(now, 24)
        if not r:
            if self.hist:
                return self.hist[-1][1].copy(), False
            return None, False
        if len(r) == 1:
            return r[0][1].copy(), True
        t = np.array([s for s, _ in r], float)
        P = np.array([p for _, p in r])
        tq = now + ahead
        span = t[-1] - t[0]
        if span < 2:
            return P[-1].copy(), True
        deg = 2 if (len(r) >= 6 and span >= 10) else 1
        out = np.zeros(2)
        tc = t - t[-1]
        for k in range(2):
            co = np.polyfit(tc, P[:, k], deg)
            out[k] = np.polyval(co, tq - t[-1])
        # sanity: never extrapolate more than 6 cm from the last seen point
        dv = out - P[-1]
        nd = np.linalg.norm(dv)
        if nd > 0.06:
            out = P[-1] + dv / nd * 0.06
        return out, True

    def nearest_path_point(self, xy, now, window=500):
        r = self.recent(now, window)
        if not r:
            return None
        P = np.array([p for _, p in r])
        i = int(np.argmin(np.linalg.norm(P - np.asarray(xy)[:2], axis=1)))
        return P[i].copy()
