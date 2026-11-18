"""Identity tracker for the cubes in world xy (Hungarian assignment with constant-velocity prediction)."""
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

GATE_M = 0.06          # max association distance per update (cubes move ~9 mm/step during shuffles)
MOVE_EPS_M = 0.004     # displacement that counts as motion between two updates


def assign(cost):
    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(cost)
        return list(zip(r, c))
    pairs, used_r, used_c = [], set(), set()
    for idx in np.argsort(cost, axis=None):
        i, j = np.unravel_index(idx, cost.shape)
        if i in used_r or j in used_c:
            continue
        pairs.append((i, j)); used_r.add(i); used_c.add(j)
    return pairs


class Tracker:
    def __init__(self, xys):
        self.pos = np.array(xys, float).reshape(-1, 2)
        self.vel = np.zeros_like(self.pos)
        self.last_step = None
        self.coast = np.zeros(len(self.pos), int)
        self.hl_score = np.zeros(len(self.pos))      # max disc score while discs visible
        self.hl_frames = np.zeros(len(self.pos), int)
        self.picked = np.zeros(len(self.pos), bool)
        self.n = len(self.pos)

    def predict(self, step):
        if self.last_step is None:
            return self.pos.copy()
        dt = max(0, step - self.last_step)
        # v2: only extrapolate tracks that were matched recently (coasting tracks used to drift off-table)
        ok = (self.coast <= 2)[:, None]
        return self.pos + np.where(ok, self.vel * min(dt, 4), 0.0)

    def update(self, dets, step, gate=GATE_M, frozen=None):
        """dets: Nx2 detections. frozen: bool mask of tracks to keep fixed (e.g. under the gripper);
        detections within 5 cm of a frozen track are discarded. Returns (max disp, n_matched)."""
        dets = np.array(dets, float).reshape(-1, 2)
        pred = self.predict(step)
        if frozen is not None and np.any(frozen) and len(dets):
            fz = self.pos[frozen]
            keep = np.sqrt(((dets[:, None] - fz[None]) ** 2).sum(2)).min(1) > 0.05
            dets = dets[keep]
            pred[frozen] = self.pos[frozen]
        dt = 1 if self.last_step is None else max(1, step - self.last_step)
        newpos = pred.copy()
        matched = np.zeros(self.n, bool)
        used = np.zeros(len(dets), bool)
        maxdisp = 0.0
        if len(dets):
            cost = np.sqrt(((pred[:, None] - dets[None]) ** 2).sum(2))
            big = cost.copy(); big[big > gate] = 1e3
            if frozen is not None:
                big[np.asarray(frozen, bool)] = 1e3
            for i, j in assign(big):
                if cost[i, j] <= gate and not (frozen is not None and frozen[i]):
                    newpos[i] = dets[j]; matched[i] = True; used[j] = True
        disp = np.linalg.norm(newpos - self.pos, axis=1)
        for i in range(self.n):
            if matched[i]:
                v = (newpos[i] - self.pos[i]) / dt
                self.vel[i] = 0.5 * self.vel[i] + 0.5 * v if disp[i] > MOVE_EPS_M else 0.3 * self.vel[i]
                self.coast[i] = 0
                maxdisp = max(maxdisp, disp[i])
            elif frozen is not None and frozen[i]:
                self.vel[i] = 0
            else:
                self.coast[i] += 1
                if self.coast[i] > 2:
                    self.vel[i] = 0
        self.last_disp = np.where(matched, disp, 0.0)
        self.last_matched = matched
        self.last_unused = dets[~used] if len(dets) else dets
        self.pos = newpos
        self.last_step = step
        return maxdisp, int(matched.sum())

    def copy(self):
        import copy as _c
        return _c.deepcopy(self)

    def summary(self):
        return " ".join("%d:(%.3f,%.3f)%s%s" % (i, p[0], p[1], "H" if self.hl_score[i] > 0.4 else "",
                                              "P" if self.picked[i] else "") for i, p in enumerate(self.pos))
