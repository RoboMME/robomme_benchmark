"""Multi-target tracker for identical containers on the table (world XY)."""
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


def _assign(C):
    """Min-cost assignment; falls back to greedy."""
    if C.size == 0:
        return [], []
    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(C)
        return list(r), list(c)
    rr, cc = [], []
    used_r, used_c = set(), set()
    for k in np.argsort(C, axis=None):
        i, j = np.unravel_index(k, C.shape)
        if i in used_r or j in used_c:
            continue
        rr.append(i)
        cc.append(j)
        used_r.add(i)
        used_c.add(j)
    return rr, cc


class Track:
    def __init__(self, tid, pos, step):
        self.id = tid
        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.zeros(2)
        self.last_seen = step
        self.last_step = step
        self.held = False
        self.yaw = None
        self.yaw_step = -1
        self.ztop = None
        self.hist = [(step, float(pos[0]), float(pos[1]))]
        self.n_seen = 1


class Tracker:
    def __init__(self, scale=1.0, gate0=0.03, gate_growth=0.003, gate_max=0.07, min_sep=0.03,
                 vmax=0.004, coast_max=0.03):
        # all distances in metres * scale (scale lets the offline pixel test reuse it)
        self.s = scale
        self.gate0 = gate0 * scale
        self.gate_growth = gate_growth * scale
        self.gate_max = gate_max * scale
        self.min_sep = min_sep * scale
        self.vmax = vmax * scale
        self.coast_max = coast_max * scale
        self.static = False  # shuffle over: no prediction, smoothed updates
        self.tracks = []
        self.step = 0
        self.max_speed_hist = []

    def init(self, positions, step):
        self.tracks = [Track(i, p, step) for i, p in enumerate(positions)]
        self.step = step

    def predict(self, step):
        for t in self.tracks:
            dt = step - t.last_step
            if dt <= 0 or t.held:
                continue
            if self.static:
                t.vel[:] = 0
                t.last_step = step
                continue
            missed = step - t.last_seen
            # shuffle still running: keep constant velocity for short occlusions
            damp = 0.97 ** max(0, missed - 1)
            newp = t.pos + t.vel * dt * damp
            # never coast further than coast_max from the last confirmed position
            if t.hist:
                last = np.array(t.hist[-1][1:3])
                dd = newp - last
                nd = np.linalg.norm(dd)
                if nd > self.coast_max:
                    newp = last + dd * (self.coast_max / nd)
            t.pos = newp
            if missed > 8:
                t.vel *= 0.9 ** dt
            t.last_step = step

    def update(self, meas, step, meas_info=None, visible_fn=None):
        """meas: (M,2) measured centres. Returns dict of assignment diagnostics."""
        self.predict(step)
        active = [t for t in self.tracks if not t.held]
        M = 0 if meas is None else len(meas)
        diag = {"n_meas": M, "assigned": 0, "unassigned_meas": 0}
        if not active or M == 0:
            self.step = step
            return diag
        P = np.array([t.pos for t in active])
        meas = np.asarray(meas, dtype=np.float64)
        D = np.linalg.norm(P[:, None, :] - meas[None, :, :], axis=2)
        gates = np.array([min(self.gate_max, self.gate0 + self.gate_growth * (step - t.last_seen)) for t in active])
        BIG = 1e6
        C = np.where(D <= gates[:, None], D, BIG)
        r, c = _assign(C)
        used = set()
        for i, j in zip(r, c):
            if C[i, j] >= BIG:
                continue
            t = active[i]
            dt = max(1, step - t.last_seen)
            newp = meas[j]
            # velocity from last confirmed position
            if t.hist:
                s0, x0, y0 = t.hist[-1]
                vel_obs = (newp - np.array([x0, y0])) / max(1, step - s0)
                t.vel = 0.6 * t.vel + 0.4 * vel_obs if dt <= 3 else vel_obs * 0.5
                sp = np.linalg.norm(t.vel)
                if sp > self.vmax:
                    t.vel *= self.vmax / sp
            if self.static:
                t.vel[:] = 0
                a = 0.25 if np.linalg.norm(newp - t.pos) < 0.012 * self.s else 0.6
                t.pos = (1 - a) * t.pos + a * newp
            else:
                t.pos = newp.copy()
            t.last_seen = step
            t.last_step = step
            t.n_seen += 1
            t.hist.append((step, float(newp[0]), float(newp[1])))
            if len(t.hist) > 400:
                t.hist = t.hist[-400:]
            if meas_info is not None and j < len(meas_info):
                info = meas_info[j]
                if info.get("yaw") is not None and info.get("yaw_ok", True):
                    t.yaw = info["yaw"]
                    t.yaw_step = step
                if info.get("ztop") is not None:
                    t.ztop = info["ztop"] if t.ztop is None else 0.8 * t.ztop + 0.2 * info["ztop"]
            used.add(j)
        diag["assigned"] = len(used)
        diag["unassigned_meas"] = M - len(used)
        # keep tracks physically separated (coasting tracks must not collapse onto others)
        self._separate(active)
        self.step = step
        return diag

    def _separate(self, active):
        for a in range(len(active)):
            for b in range(a + 1, len(active)):
                ta, tb = active[a], active[b]
                d = tb.pos - ta.pos
                n = np.linalg.norm(d)
                if n < self.min_sep:
                    # move the less recently seen one away
                    if n < 1e-9:
                        d = np.array([1.0, 0.0])
                        n = 1.0
                    push = (self.min_sep - n) * d / n
                    if ta.last_seen < tb.last_seen:
                        ta.pos -= push
                    elif tb.last_seen < ta.last_seen:
                        tb.pos += push

    def speeds(self, window=6):
        """Recent max speed (units/step) over tracks seen in the window."""
        out = []
        for t in self.tracks:
            h = [x for x in t.hist if x[0] >= self.step - window]
            if len(h) >= 2:
                (s0, x0, y0), (s1, x1, y1) = h[0], h[-1]
                if s1 > s0:
                    out.append(np.hypot(x1 - x0, y1 - y0) / (s1 - s0))
        return out

    def summary(self):
        return [{"id": t.id, "pos": [round(float(t.pos[0]) / self.s, 4), round(float(t.pos[1]) / self.s, 4)],
                 "seen": int(t.last_seen), "held": t.held,
                 "yaw": None if t.yaw is None else round(float(t.yaw), 3)} for t in self.tracks]
