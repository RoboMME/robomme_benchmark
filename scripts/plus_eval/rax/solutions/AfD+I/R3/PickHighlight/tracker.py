"""Multi-cube tracker on the static front camera (world xy), carrying highlight flags."""
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


def _assign(cost):
    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    # greedy fallback
    pairs = []
    used_r, used_c = set(), set()
    for idx in np.argsort(cost, axis=None):
        i, j = np.unravel_index(idx, cost.shape)
        if i in used_r or j in used_c:
            continue
        pairs.append((int(i), int(j)))
        used_r.add(i)
        used_c.add(j)
    return pairs


class Tracker:
    def __init__(self, xys, gate=0.04, regate=0.15):
        self.pos = np.array(xys, dtype=float).reshape(-1, 2)
        self.vel = np.zeros_like(self.pos)
        self.hl = np.zeros(len(self.pos), bool)
        self.picked = np.zeros(len(self.pos), bool)
        self.last_seen = np.zeros(len(self.pos), int)
        self.gate = gate
        self.regate = regate
        self.last_step = 0
        self.last_motion_step = -1000
        self.motion_hist = []

    def update(self, dets, step, frozen_xy=None, frozen_r=0.0):
        """dets: (M,2) detections. Tracks within frozen_r of frozen_xy are not updated
        (arm occlusion / held cube) and detections there are ignored."""
        dets = np.asarray(dets, float).reshape(-1, 2)
        active = np.ones(len(self.pos), bool)
        if frozen_xy is not None and frozen_r > 0:
            fz = np.asarray(frozen_xy, float)
            active = np.linalg.norm(self.pos - fz, axis=1) > frozen_r
            if len(dets):
                dets = dets[np.linalg.norm(dets - fz, axis=1) > frozen_r]
        dt = max(1, step - self.last_step)
        self.last_step = step
        moved = 0
        if len(dets) and active.any():
            idx = np.nonzero(active)[0]
            pred = self.pos[idx] + self.vel[idx] * min(dt, 5)
            cost = np.linalg.norm(pred[:, None, :] - dets[None, :, :], axis=2)
            # also allow matching to the un-predicted position
            cost = np.minimum(cost, np.linalg.norm(self.pos[idx][:, None, :] - dets[None, :, :], axis=2))
            big = 10.0
            c2 = np.where(cost < self.gate, cost, big)
            for i, j in _assign(c2):
                if c2[i, j] >= big:
                    continue
                k = idx[i]
                d = dets[j] - self.pos[k]
                if np.linalg.norm(d) > 0.004:
                    moved += 1
                self.vel[k] = 0.5 * self.vel[k] + 0.5 * d / dt
                self.pos[k] = dets[j]
                self.last_seen[k] = step
        # Re-acquire lost tracks with unclaimed detections (wider gate).
        if len(dets) and active.any():
            claimed = set()
            for k in np.nonzero(self.last_seen == step)[0]:
                dd = np.linalg.norm(dets - self.pos[k], axis=1)
                j = int(np.argmin(dd))
                if dd[j] < 1e-9:
                    claimed.add(j)
            lost = [k for k in np.nonzero(active)[0] if self.last_seen[k] != step]
            free = [j for j in range(len(dets)) if j not in claimed]
            if lost and free:
                cost = np.linalg.norm(self.pos[lost][:, None, :] - dets[free][None, :, :], axis=2)
                big = 10.0
                c2 = np.where(cost < self.regate, cost, big)
                for i, j in _assign(c2):
                    if c2[i, j] >= big:
                        continue
                    k = lost[i]
                    self.pos[k] = dets[free[j]]
                    self.vel[k] = 0
                    self.last_seen[k] = step
                    moved += 1
        # decay velocity of unmatched tracks
        unseen = self.last_seen != step
        self.vel[unseen] *= 0.5
        self.motion_hist.append((step, moved))
        if moved >= 2:
            self.last_motion_step = step
        return moved

    def quiet_for(self, step):
        return step - self.last_motion_step
