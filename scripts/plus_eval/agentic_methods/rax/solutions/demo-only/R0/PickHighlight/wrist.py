"""Wrist-camera helpers: closing-axis calibration and cube refinement at hover."""
import math

import numpy as np

from mcp_io import log
import scene


def closing_axis(robot, table_z):
    """World xy direction of the wrist image's horizontal axis (= finger closing axis)."""
    robot.observe()
    pairs = []
    for v in (90, 130, 170, 210):
        pairs.append((40, v)); pairs.append((216, v))
    pts = robot.backproject('wrist', pairs)
    dirs = []
    for i in range(0, len(pairs), 2):
        a, b = pts[i], pts[i + 1]
        if a is None or b is None:
            continue
        if abs(a[2] - b[2]) > 0.02:
            continue
        d = b[:2] - a[:2]
        n = np.linalg.norm(d)
        if n > 0.01:
            dirs.append(d / n)
    if not dirs:
        log('closing axis: no valid pairs')
        return None
    d = np.mean(dirs, axis=0)
    d /= np.linalg.norm(d)
    log('closing axis world xy', np.round(d, 3).tolist(), 'from', len(dirs), 'pairs')
    return d


def _blob_points(robot, comp, k):
    ys, xs = comp
    sel = np.linspace(0, len(ys) - 1, min(k, len(ys))).astype(int)
    pix = [(int(xs[i]), int(ys[i])) for i in sel]
    U, V, P = scene.sample_points(robot, 'wrist', pix, budget_s=15)
    return U, V, P


def _fit(P, top_z, s, exp_xy):
    if len(P) < 6:
        return None
    m = np.abs(P[:, 2] - top_z) < 0.008
    if m.sum() < 8:
        return None
    xy = P[m, :2]
    c, yaw, a, b = scene.min_area_rect(xy)
    if np.linalg.norm(c - exp_xy) > 0.04:
        return None
    if not (0.5 * s < max(a, b) < 1.6 * s):
        return None
    return dict(xy=np.array(c), yaw=yaw if min(a, b) > 0.5 * s else None, top_z=float(np.median(P[m, 2])),
                n=int(m.sum()), rect=(float(a), float(b)))


def refine(robot, exp_xy, top_z, s, color, table_z):
    """Locate the target cube's top face from the wrist camera. Returns dict or None."""
    robot.observe()
    img = robot.cams.get('wrist')
    if img is None:
        log('refine: no wrist image')
        return None
    best = None
    if color is not None:
        d = np.sqrt(((img.astype(float) - np.asarray(color)[None, None, :]) ** 2).sum(-1))
        mask = d < 42
        comps = sorted([c for c in scene.label(mask) if len(c[0]) >= 30], key=lambda c: -len(c[0]))[:5]
        for comp in comps:
            U, V, P = _blob_points(robot, comp, 12)
            if len(P) < 4:
                continue
            med = np.median(P, axis=0)
            if abs(med[2] - top_z) > 0.015 or np.linalg.norm(med[:2] - exp_xy) > 0.045:
                continue
            dist = np.linalg.norm(med[:2] - exp_xy)
            if best is None or dist < best[0]:
                best = (dist, comp)
        if best is not None:
            U, V, P = _blob_points(robot, best[1], 96)
            f = _fit(P, top_z, s, exp_xy)
            if f is not None:
                log('refine(colour): xy=(%.3f,%.3f) yaw=%s n=%d rect=%s shift=%.3f' % (
                    f['xy'][0], f['xy'][1], None if f['yaw'] is None else round(math.degrees(f['yaw']), 1),
                    f['n'], np.round(f['rect'], 3).tolist(), np.linalg.norm(f['xy'] - exp_xy)))
                return f
    # colour-free fallback: coarse grid, then densify around matching points
    H, W = img.shape[:2]
    grid = [(u, v) for v in range(8, H, 16) for u in range(8, W, 16)]
    U, V, P = scene.sample_points(robot, 'wrist', grid, budget_s=25)
    if len(P) == 0:
        return None
    m = (np.abs(P[:, 2] - top_z) < 0.012) & (np.sqrt(((P[:, :2] - exp_xy) ** 2).sum(1)) < 0.04)
    if m.sum() == 0:
        log('refine: grid found no top-face points near target')
        return None
    u0, u1 = int(U[m].min()) - 24, int(U[m].max()) + 24
    v0, v1 = int(V[m].min()) - 24, int(V[m].max()) + 24
    dense = [(u, v) for v in range(max(0, v0), min(H, v1), 4) for u in range(max(0, u0), min(W, u1), 4)]
    U2, V2, P2 = scene.sample_points(robot, 'wrist', dense[:640], budget_s=25)
    if len(P2) == 0:
        return None
    near = np.sqrt(((P2[:, :2] - exp_xy) ** 2).sum(1)) < 0.045
    f = _fit(P2[near], top_z, s, exp_xy)
    if f is not None:
        log('refine(grid): xy=(%.3f,%.3f) yaw=%s n=%d shift=%.3f' % (
            f['xy'][0], f['xy'][1], None if f['yaw'] is None else round(math.degrees(f['yaw']), 1), f['n'],
            np.linalg.norm(f['xy'] - exp_xy)))
    return f
