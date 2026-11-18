"""Voronoi/centroid tracker for N identical containers from fused RGB-D top-face points."""
import numpy as np

R_ASSOC = 0.045     # m: points farther than this from every prediction are ignored
MIN_PTS = 12        # grid cells needed to update a track
GRID = 0.004
MIN_BRIGHT = 140   # container tops >=147; gray occluder pillar 110-136 (ep2)
TOP_ZLO, TOP_ZHI = 0.058, 0.09   # container top face band (top z ~0.073)
HAND_HALF = (0.05, 0.125)        # hand mask half-sizes (across, along finger-opening axis)
HAND_MASK_Z = 0.13   # v2: fingertips (TCP) above the top band -> no hand cells in band; ep2 showed z<0.25 masking biased the target track


def ee_yaw(state):
    """Yaw of the EE x-axis in world (gripper pointing down)."""
    from scipy.spatial.transform import Rotation as Rot
    R = Rot.from_rotvec(state[3:6]).as_matrix()
    return float(np.arctan2(R[1, 0], R[0, 0])), R


def top_points(views, state, xmin=-0.47, mask_hand=True):
    """World XY of container-top cells (deduplicated on a grid), with the robot hand masked."""
    pts = []
    for v in views.values():
        wp, ok = v.world_points()
        z = wp[..., 2]
        rgb = v.rgb.astype(np.int16)
        bright = (rgb.min(2) >= MIN_BRIGHT) & ((rgb.max(2) - rgb.min(2)) < 40)
        m = ok & bright & (z > TOP_ZLO) & (z < TOP_ZHI) & (wp[..., 0] > xmin) & (np.abs(wp[..., 1]) < 0.45) & (wp[..., 0] < 0.4)
        pts.append(wp[m][:, :2])
    P = np.concatenate(pts) if pts else np.zeros((0, 2))
    if mask_hand and state is not None and state[2] < HAND_MASK_Z and len(P):
        yaw, R = ee_yaw(state)
        ex, ey = R[:2, 0], R[:2, 1]
        d = P - state[:2]
        a, b = np.abs(d @ ex), np.abs(d @ ey)
        P = P[~((a < HAND_HALF[0]) & (b < HAND_HALF[1]))]
    if len(P) == 0:
        return P
    cells = np.unique(np.floor(P / GRID).astype(int), axis=0)
    return (cells + 0.5) * GRID


class T:
    def __init__(self, i, p, label, step):
        self.id, self.label, self.p, self.v = i, label, np.asarray(p, float), np.zeros(2)
        self.last, self.miss, self.n, self.hist = step, 0, 0, [(step, p[0], p[1], 0)]


class Tracker:
    def __init__(self, xys, labels, step):
        self.tracks = [T(i, p, l, step) for i, (p, l) in enumerate(zip(xys, labels))]
        self.step = step

    def update(self, P, step, frozen=(), iters=3):
        dt = max(1, step - self.step)
        pred = np.array([t.p + t.v * min(dt, 10) for t in self.tracks])
        est = pred.copy()
        cnt = np.zeros(len(self.tracks), int)
        if len(P):
            for _ in range(iters):
                D = np.linalg.norm(P[:, None, :] - est[None], axis=2)
                j = D.argmin(1)
                dm = D[np.arange(len(P)), j]
                for i in range(len(self.tracks)):
                    sel = (j == i) & (dm < R_ASSOC)
                    cnt[i] = sel.sum()
                    if cnt[i] >= MIN_PTS:
                        est[i] = P[sel].mean(0)
                    else:
                        est[i] = pred[i]
        for i, t in enumerate(self.tracks):
            if t.id in frozen:
                t.hist.append((step, t.p[0], t.p[1], -1)); t.v[:] = 0; t.last = step; continue
            if cnt[i] >= MIN_PTS:
                vnew = (est[i] - t.p) / (step - t.last if step > t.last else 1)
                t.v = 0.6 * t.v + 0.4 * vnew
                t.p = est[i]; t.last = step; t.miss = 0; t.n = int(cnt[i])
            else:
                t.p = pred[i]; t.last = step; t.v *= 0.8; t.miss += 1; t.n = int(cnt[i])
            t.hist.append((step, t.p[0], t.p[1], t.n))
        self.step = step
        return cnt

    def by_label(self, label):
        return [t for t in self.tracks if t.label == label]
