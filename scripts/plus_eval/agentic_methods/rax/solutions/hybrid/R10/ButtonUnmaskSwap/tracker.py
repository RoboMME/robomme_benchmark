import numpy as np
from percep import gather, clusters, kmeans_track, classify_color
from scipy.optimize import linear_sum_assignment

CZ_LO, CZ_HI = 0.045, 0.13


class Tracker:
    """Track N identical containers: one-to-one cluster matching when all are separated,
    otherwise Lloyd (k-means) iterations seeded at the prediction."""

    def __init__(self, cubes):
        self.colors = [c for c, _ in cubes]
        self.pos = np.array([xy for _, xy in cubes], float)
        self.vel = np.zeros_like(self.pos)
        self.last_step = None
        self.history = []
        self.still = 0

    def ee_mask(self, obs):
        ee = obs.state[:3]
        if ee[2] < 0.14:
            return ee
        return None

    def update(self, obs):
        stp = obs.step if obs.step is not None else (self.last_step or 0) + 1
        dt = 1 if self.last_step is None else max(1, stp - self.last_step)
        self.last_step = stp
        P, C = gather(obs, CZ_LO, CZ_HI, ee_mask=self.ee_mask(obs))
        cl = [c for c in clusters(P, C) if 60 < c["n"] < 260]
        pred = self.pos + self.vel
        newpos = None
        if len(cl) == len(self.pos):
            X = np.array([c["xy"] for c in cl])
            D = np.linalg.norm(pred[:, None] - X[None], axis=2)
            r, c = linear_sum_assignment(D)
            if D[r, c].max() < 0.05:
                newpos = X[c[np.argsort(r)]]
        mode = "clusters"
        if newpos is None:
            newpos, counts = kmeans_track(P, pred.copy())
            mode = "kmeans"
            # duplicate guard: two centres collapsing onto one container -> weaker one keeps its prediction
            for _ in range(3):
                Dn = np.linalg.norm(newpos[:, None] - newpos[None], axis=2) + np.eye(len(newpos))
                i, j = np.unravel_index(Dn.argmin(), Dn.shape)
                if Dn[i, j] >= 0.035:
                    break
                k = i if counts[i] < counts[j] else j
                newpos[k] = self.pos[k]
                mode = "kmeans_dup"
        move = float(np.linalg.norm(newpos - self.pos, axis=1).max())
        self.vel = 0.5 * (newpos - self.pos)
        self.pos = newpos
        if mode == "clusters" and move / dt < 0.0006:
            self.still += dt
        else:
            self.still = 0
        self.history.append((stp, mode, len(cl), round(move, 4)))
        return mode, len(cl), move

    def where(self, color):
        return self.pos[self.colors.index(color)]


def init_cubes(obs0):
    import percep
    Pt, _ = gather(obs0, 0.045, 0.6)
    percep.set_static(Pt)
    P, C = gather(obs0, 0.012, 0.06)
    cl = clusters(P, C)
    objs = [(classify_color(c["color"]), c["xy"], c["n"]) for c in cl if c["ztop"] < 0.05]
    return objs


def split_buttons(objs, obs_cov):
    P, C = gather(obs_cov, CZ_LO, CZ_HI)
    cl = [c for c in clusters(P, C) if c["n"] > 60]
    X = np.array([c["xy"] for c in cl]) if cl else np.zeros((0, 2))
    cubes, buttons = [], []
    for col, xy, n in objs:
        d = np.linalg.norm(X - xy, axis=1).min() if len(X) else 1.0
        if d < 0.035:
            cubes.append((col, xy))
        else:
            buttons.append(xy)
    return cubes, buttons
