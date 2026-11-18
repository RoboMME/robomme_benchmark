"""Seeded physical motion primitives for the hard-seed experiment."""
import hashlib
from itertools import combinations
import numpy as np


def rng_for(seed, domain):
    digest = hashlib.sha256(f'hard-motion-v1/{seed}/{domain}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little'))


class RateSchedule:
    """Signed speed multiplier with smooth transitions, pauses and reversals."""
    def __init__(self, seed, domain, horizon=30000):
        rng = rng_for(seed, domain)
        self.rates, self.events = [], []
        previous = 1.
        while len(self.rates) < horizon:
            segments = [('travel', float(rng.uniform(.65, 1.25)), int(rng.integers(55, 96))),
                        ('pause', 0., int(rng.integers(12, 23))),
                        ('reverse', -float(rng.uniform(.3, .55)), int(rng.integers(12, 23))),
                        ('travel', float(rng.uniform(.8, 1.2)), int(rng.integers(45, 86)))]
            for kind, rate, duration in segments:
                start = len(self.rates)
                blend = min(8, duration // 2)
                ease = .5 - .5 * np.cos(np.linspace(0, np.pi, blend))
                self.rates.extend((previous + (rate - previous) * ease).tolist())
                self.rates.extend([rate] * (duration - blend))
                self.events.append(dict(kind=kind, start=start, end=len(self.rates), multiplier=rate))
                previous = rate
        self.rates = np.asarray(self.rates)
        self.phase = np.r_[0., np.cumsum(self.rates)]

    def rate(self, step):
        return float(self.rates[min(max(int(step), 0), len(self.rates)-1)])

    def value(self, step):
        return float(np.interp(max(0., step), np.arange(len(self.phase)), self.phase))


def minimum_gap(paths, radii, *, moving_only=False):
    """Minimum surface separation over every continuous linear frame interval."""
    paths = np.asarray(paths)
    i, j = np.triu_indices(paths.shape[1], 1)
    if moving_only:
        active=(np.linalg.norm(np.diff(paths,axis=0),axis=-1)>1e-10).any(axis=0)
        mask=active[i]|active[j]
        i,j=i[mask],j[mask]
    if not len(i):
        return float('inf')
    rel = paths[:, i] - paths[:, j]
    a, delta = rel[:-1], np.diff(rel, axis=0)
    t = np.clip(-np.sum(a * delta, axis=-1) / np.maximum(np.sum(delta * delta, axis=-1), 1e-15), 0, 1)
    gap = np.linalg.norm(a + t[..., None] * delta, axis=-1) - np.asarray(radii)[i] - np.asarray(radii)[j]
    return float(gap.min())


def path_clear(paths, radii, *, bounds, circles=(), valid=None, max_diameter=None, margin=.006, moving_only=False):
    radii = np.asarray(radii)
    if minimum_gap(paths, radii,moving_only=moving_only) < margin:
        return False
    bounds = np.asarray(bounds)
    if max_diameter is not None:
        if np.linalg.norm(paths[:, :, None] - paths[:, None, :], axis=-1).max() > max_diameter:
            return False
    if moving_only:
        active=(np.linalg.norm(np.diff(paths,axis=0),axis=-1)>1e-10).any(axis=0)
        paths,radii=paths[:,active],radii[active]
    if np.any(paths < bounds[:, 0] + radii[None, :, None]) or np.any(paths > bounds[:, 1] - radii[None, :, None]):
        return False
    a, delta = paths[:-1], np.diff(paths, axis=0)
    for center, radius in circles:
        rel = a - np.asarray(center)
        t = np.clip(-np.sum(rel * delta, axis=-1) / np.maximum(np.sum(delta * delta, axis=-1), 1e-15), 0, 1)
        if np.any(np.linalg.norm(rel + t[..., None] * delta, axis=-1) < radius + radii + margin):
            return False
    return valid is None or bool(valid(paths))


def swap_plan(points, radii, rng, *, steps=88, speed=.0045, bounds, circles=(),
              valid=None, allowed=None, max_diameter=None, margin=None):
    """Physically exchange two actor positions; never exchange labels in place."""
    points, radii = np.asarray(points), np.asarray(radii)
    eligible = list(range(len(points))) if allowed is None else list(allowed)
    pairs = list(combinations(eligible, 2))
    rng.shuffle(pairs)
    u = np.linspace(0., 1., steps + 1)
    ease = u*u*(3-2*u)
    for i, j in pairs:
        delta = points[j] - points[i]
        distance = np.linalg.norm(delta)
        if not .08 <= distance <= .23:
            continue
        axis = delta / distance
        normal = np.array([-axis[1], axis[0]])
        center = (points[i]+points[j])/2
        for style in rng.permutation(['ellipse', 'bow', 'asymmetric_bow']):
            for sign in rng.permutation([-1, 1]):
                width = float(max(radii[i], radii[j]) + rng.uniform(.015, .035))
                if style == 'ellipse':
                    along = -.5 * distance * np.cos(np.pi * ease)
                else:
                    along = (ease-.5) * distance
                lateral = sign * width * np.sin(np.pi * ease)
                if style == 'asymmetric_bow':
                    lateral *= 1 + .22*np.sin(2*np.pi*ease)
                offset = along[:, None]*axis + lateral[:, None]*normal
                paths = np.repeat(points[None], steps+1, axis=0)
                paths[:, i], paths[:, j] = center+offset, center-offset
                paths[0] = points
                paths[-1, i], paths[-1, j] = points[j], points[i]
                max_speed = float(np.linalg.norm(np.diff(paths, axis=0), axis=-1).max())
                if max_speed > speed:
                    continue
                if path_clear(paths, radii, bounds=bounds, circles=circles, valid=valid,
                              max_diameter=max_diameter, margin=.006+speed if margin is None else margin,
                              moving_only=True):
                    return paths, dict(pair=[int(i), int(j)], style=str(style), duration_steps=steps,
                                       max_speed_m_per_step=max_speed, minimum_pair_gap_m=minimum_gap(paths, radii),
                                       start_xy=points[[i,j]].tolist(), end_xy=paths[-1,[i,j]].tolist())
    return None
