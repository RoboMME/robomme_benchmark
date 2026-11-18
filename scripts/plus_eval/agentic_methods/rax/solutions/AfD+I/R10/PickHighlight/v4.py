"""v4 logic: continuous front-camera tracking with shuffle-aware picking."""
import numpy as np
from motion import log
from tracker2 import Tracker2
from percep import components

FRONT_XBIAS = -0.009   # front centroid sits ~9 mm +x of the cube centre


class Scene:
    def __init__(self, R, T):
        self.R, self.T = R, T
        self.motion = []      # (step, maxd among tracks far from TCP)
        self.starts = []      # shuffle start steps
        self.active = False
        self.quiet = 0

    def hook(self, R):
        if ("front", "depth") not in R.imgs:
            return
        P = R.world("front")
        st = R.state()
        prev = self.T.pos.copy()
        self.T.update_P(P, st[:3], hold=(not self.active and self.quiet >= 6))
        d = np.linalg.norm(self.T.pos - prev, axis=1)
        far = np.linalg.norm(self.T.pos - st[None, :2], axis=1) > 0.08
        df = np.sort(d[far])[::-1] if far.any() else np.zeros(1)
        md = float(df[1]) if len(df) >= 2 else 0.0  # shuffles move >=2 cubes at once
        step = int(R.last.get("step_id", 0))
        self.motion.append((step, md))
        if md > 0.002:
            if not self.active and self.quiet >= 10:
                self.starts.append(step)
                log("shuffle start", step, "starts", self.starts)
            self.active = True
            self.quiet = 0
        else:
            self.quiet += 1
            if self.quiet >= 6:
                self.active = False

    def period(self):
        if len(self.starts) >= 2:
            ds = np.diff(self.starts)
            p = float(np.median(ds))
            if 40 <= p <= 400:
                return p
        return None

    def steps_to_next(self):
        p = self.period()
        if p is None or not self.starts:
            return None
        now = int(self.R.last.get("step_id", 0))
        nxt = self.starts[-1] + p
        while nxt < now - 5:
            nxt += p
        return nxt - now

    def safe_window(self, need):
        if self.active or self.quiet < 4:
            return False
        s = self.steps_to_next()
        if s is None:
            return True
        return s >= need or s < -40  # s<-40: overdue, assume no more


def servo_xy(R, target_fn, z, n=1, vmax=0.025):
    """One step toward target_fn() at height z."""
    st = R.state()
    t = target_fn()
    err = np.array([t[0] - st[0], t[1] - st[1], z - st[2]])
    d = np.linalg.norm(err)
    if d > vmax:
        err = err / d * vmax
    a = err / 0.05 * 1.3
    R.step([[a[0], a[1], a[2], 0, 0, 0, R.grip]])
    return d


def hover_until(R, S, target_fn, z, need, cap=300, tol=0.006):
    """Hover above target (following it) until close and a safe window opens."""
    for i in range(cap):
        if R.ended:
            return False
        d = servo_xy(R, target_fn, z)
        if d < tol and S.safe_window(need):
            return True
    return False


def wrist_cube_near(R, xy, radius):
    R.observe()
    P = R.world("wrist")
    m = (P[..., 2] > 0.034) & (P[..., 2] < 0.07)
    best = None
    for c in components(m, 30):
        pts = P[c[:, 0], c[:, 1]]
        top = pts[pts[:, 2] > np.percentile(pts[:, 2], 50)][:, :2]
        cxy = top.mean(0)
        ext = (top.max(0) - top.min(0)).max()
        if ext > 0.07:
            near = top[np.linalg.norm(top - xy[None], axis=1) < 0.03]
            if len(near) < 20:
                continue
            cxy = near.mean(0)
        dd = float(np.linalg.norm(cxy - xy))
        if best is None or dd < best[0]:
            best = (dd, cxy)
    if best is None or best[0] > radius:
        return None
    return best[1]
