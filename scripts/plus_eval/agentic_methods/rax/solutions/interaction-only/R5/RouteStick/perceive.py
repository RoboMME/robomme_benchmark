"""Landmark detection (white discs, standing blocks) from front RGB-D."""
import numpy as np
from scipy import ndimage
import rslib

def _clusters(pts, radius=0.03, min_n=3):
    """Greedy xy clustering; returns list of (centroid xy, count)."""
    pts = pts[np.isfinite(pts).all(1)]
    out = []
    used = np.zeros(len(pts), bool)
    order = np.arange(len(pts))
    while True:
        idx = np.nonzero(~used)[0]
        if len(idx) == 0: break
        seed = pts[idx[0], :2]
        for _ in range(5):
            d = np.linalg.norm(pts[:, :2] - seed, axis=1)
            mem = (d < radius) & ~used
            if mem.sum() == 0: break
            seed = pts[mem, :2].mean(0)
        d = np.linalg.norm(pts[:, :2] - seed, axis=1)
        mem = (d < radius) & ~used
        if mem.sum() == 0: mem[idx[0]] = True
        used |= mem
        if mem.sum() >= min_n: out.append((seed, int(mem.sum())))
    return out

def detect(meta, rgb, depth, robot_xy_exclude=(-0.42,), cam="front"):
    P = rslib.deproject(meta, cam, depth)
    rgb = rgb.astype(int)
    valid = np.isfinite(P).all(-1)
    reg = valid & (P[..., 0] > robot_xy_exclude[0]) & (np.abs(P[..., 1]) < 0.6) & (P[..., 0] < 0.45)
    # discs: bright, low saturation, near table height
    white = (rgb.min(-1) > 150) & ((rgb.max(-1) - rgb.min(-1)) < 45) & reg & (P[..., 2] < 0.03) & (P[..., 2] > 0.005)
    discs = [c for c, n in _clusters(P[white], 0.025, 3)]
    # blocks: points well above table, cluster tops
    high = reg & (P[..., 2] > 0.04) & (P[..., 2] < 0.25)
    blocks = []
    hp = P[high]
    for c, n in _clusters(hp, 0.04, 15):
        blocks.append(c)
    return np.array(discs).reshape(-1, 2), np.array(blocks).reshape(-1, 2), P

def order_along_line(pts):
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c)
    d = vt[0]
    if d[1] < 0: d = -d  # orient toward +y
    t = (pts - c) @ d
    return pts[np.argsort(t)], c, d

def fit_rigid(A, B):
    """Return R(2x2), t with B ~ A R^T + t (least squares)."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1] *= -1; R = Vt.T @ U.T
    t = cb - ca @ R.T
    return R, t


def detect_discs(meta, imgs_by_cam):
    """Front-camera discs, plus wrist-camera discs not already found (fills occlusions)."""
    D = np.zeros((0, 2))
    for cam in ("front", "wrist"):
        if (cam, "rgb") not in imgs_by_cam: continue
        try:
            Dc = detect(meta, imgs_by_cam[(cam, "rgb")], imgs_by_cam[(cam, "depth")], cam=cam)[0]
        except Exception:
            continue
        for p in Dc:
            if len(D) == 0 or np.min(np.linalg.norm(D - p, axis=1)) > 0.03:
                D = np.vstack([D, p])
    return D

def consistent_order(Dv, De):
    """Order video discs along their line; choose live ordering (fwd/rev) with least rotation."""
    Dv = order_along_line(Dv)[0]; De = order_along_line(De)[0]
    if len(Dv) != len(De): return Dv, De
    best = None
    for cand in (De, De[::-1]):
        R, t = fit_rigid(Dv, cand)
        ang = abs(np.arctan2(R[1, 0], R[0, 0]))
        if best is None or ang < best[0]: best = (ang, cand)
    return Dv, best[1]

def match_mismatch(Dv, De):
    """Counts differ: align by centroid+principal axis (both senses), assign nearest live disc
    to each video disc (predicted position if none within 4 cm). Returns De aligned to Dv."""
    Dv = order_along_line(Dv)[0]
    _, cv, dv = order_along_line(Dv); _, ce, de = order_along_line(De)
    best = None
    for sgn in (1, -1):
        ang = np.arctan2(sgn * de[1], sgn * de[0]) - np.arctan2(dv[1], dv[0])
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        pred = (Dv - cv) @ R.T + ce
        for _ in range(3):  # refine with matched pairs
            d = np.linalg.norm(pred[:, None] - De[None], axis=2)
            j = d.argmin(1); ok = d.min(1) < 0.04
            if ok.sum() >= 2:
                R2, t2 = fit_rigid(Dv[ok], De[j[ok]]); pred = Dv @ R2.T + t2
        d = np.linalg.norm(pred[:, None] - De[None], axis=2)
        cost = np.minimum(d.min(1), 0.04).sum() + abs(np.arctan2(np.sin(ang), np.cos(ang))) * 0.01
        if best is None or cost < best[0]: best = (cost, pred, d)
    _, pred, d = best
    out = pred.copy()
    for i in range(len(Dv)):
        if d[i].min() < 0.04: out[i] = De[d[i].argmin()]
    return Dv, out

def drop_outliers(De, n):
    """Remove detections farthest from the principal line until n remain."""
    De = np.array(De)
    while len(De) > n:
        _, c, d = order_along_line(De)
        nrm = np.array([-d[1], d[0]])
        r = np.abs((De - c) @ nrm)
        De = np.delete(De, int(np.argmax(r)), axis=0)
    return De

def stick_tip(meta, imgs_by_cam, max_depth=0.3):
    """Stick tip (world) from wrist depth: mean of the lowest stick points. Returns (tip, n) or (None, 0)."""
    if ("wrist", "depth") not in imgs_by_cam: return None, 0
    dep = imgs_by_cam[("wrist", "depth")]
    m = (dep > 0) & (dep < max_depth)
    if m.sum() < 30: return None, int(m.sum())
    P = rslib.deproject(meta, "wrist", dep)[m]
    P = P[np.isfinite(P).all(1)]
    lo = P[np.argsort(P[:, 2])[:15]]
    return np.median(lo, 0), int(m.sum())

def block_points(meta, imgs_by_cam, xmin=-0.42):
    rgb = imgs_by_cam[("front", "rgb")]; dep = imgs_by_cam[("front", "depth")]
    P = rslib.deproject(meta, "front", dep)
    m = np.isfinite(P).all(-1) & (P[..., 2] > 0.035) & (P[..., 2] < 0.3) & (P[..., 0] > xmin) & (P[..., 0] < 0.45)
    return P[m][:, :2]

def red_near(meta, imgs_by_cam, xy, r=0.03):
    """True if red (registered) disc pixels are near xy in the front view, or the wrist view is mostly red-disc."""
    hits = 0
    for cam in ("front", "wrist"):
        if (cam, "rgb") not in imgs_by_cam or (cam, "depth") not in imgs_by_cam: continue
        rgb = imgs_by_cam[(cam, "rgb")].astype(int)
        red = (rgb[..., 0] > 150) & (rgb[..., 1] < 70) & (rgb[..., 2] < 70)
        if red.sum() == 0: continue
        P = rslib.deproject(meta, cam, imgs_by_cam[(cam, "depth")])[red]
        P = P[np.isfinite(P).all(1)]
        if len(P) and (np.linalg.norm(P[:, :2] - xy, axis=1) < r).sum() >= 5: hits += 1
    return hits > 0
