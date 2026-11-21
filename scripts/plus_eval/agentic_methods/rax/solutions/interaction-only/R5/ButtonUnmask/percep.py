"""Perception for ButtonUnmask: height-map object detection from front RGB-D.

detect(obs) -> list of dicts {kind: 'cube'|'container'|'button', xy, ztop, color, n}
Pure functions on an Obs-like object (rgb, depth, cloud()).
"""
import re
import numpy as np
import cv2

TABLE_X = (-0.32, 0.26)
TABLE_Y = (-0.32, 0.32)
CELL = 0.004
ZMIN = 0.010          # object points above table
CONTAINER_Z = 0.055   # top above this -> container (container top ~0.073, cube ~0.033)
CUBE_TOP_MIN = 0.020

# reference colours (front RGB, measured ep1)
PROTOS = {
    'red': (231, 8, 8), 'green': (6, 231, 6), 'blue': (7, 7, 231), 'yellow': (212, 212, 7),
    'orange': (231, 160, 5), 'purple': (176, 81, 214), 'white': (222, 222, 222),
}
COLOR_WORDS = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'white', 'black', 'pink',
               'cyan', 'brown', 'gray', 'grey', 'magenta', 'violet']


def parse_targets(instruction):
    """Ordered list of colour words for 'container hiding the X cube' phrases."""
    s = instruction.lower()
    found = re.findall(r"hiding (the |all )?(\w+) cube(s?)", s)
    if found:
        return [w for _, w, _ in found]
    # fallback: colour words in order of appearance
    toks = re.findall(r"[a-z]+", s)
    return [t for t in toks if t in COLOR_WORDS]


def parse_targets_full(instruction):
    """[(colour, want_all)] in order."""
    s = instruction.lower()
    out = []
    for m in re.finditer(r"(all containers|the one|the container|container|containers)?[^,.]*?hiding (the |all )?(\w+) cube(s?)", s):
        word = m.group(3)
        plural = bool(m.group(4)) or (m.group(1) or '').startswith('all') or (m.group(2) or '').startswith('all')
        out.append((word, plural))
    return out


def classify_color(rgb):
    """Hue-based colour name (robust to front/wrist brightness). Returns (name, saturation*100)."""
    r, g, b = [float(x) / 255 for x in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    sat = 0 if mx <= 0 else (mx - mn) / mx
    if mx < 0.18:
        return 'black', sat * 100
    if sat < 0.25:
        return ('white' if mx > 0.55 else 'gray'), sat * 100
    if mx == r:
        h = (60 * ((g - b) / (mx - mn))) % 360
    elif mx == g:
        h = 60 * ((b - r) / (mx - mn)) + 120
    else:
        h = 60 * ((r - g) / (mx - mn)) + 240
    for lo, hi, name in [(0, 12, 'red'), (12, 50, 'orange'), (50, 75, 'yellow'), (75, 165, 'green'),
                         (165, 200, 'cyan'), (200, 262, 'blue'), (262, 300, 'purple'), (300, 340, 'pink'),
                         (340, 361, 'red')]:
        if lo <= h < hi:
            return name, sat * 100
    return 'red', sat * 100


def heightmap(obs, cam='front', exclude=()):
    """Returns (zmax grid, rgb grid). cam: name or tuple of names. exclude: list of (xy, radius)."""
    cams = (cam,) if isinstance(cam, str) else tuple(cam)
    Ps, Cs = [], []
    for c in cams:
        Pc = obs.cloud(c).reshape(-1, 3)
        Cc = obs.rgb[c].reshape(-1, 3).astype(float)
        if c == 'wrist':  # drop gripper fingers / very near points
            near = obs.depth[c].reshape(-1) < 0.10
            Pc = Pc[~near]; Cc = Cc[~near]
        Ps.append(Pc); Cs.append(Cc)
    P = np.concatenate(Ps); rgb = np.concatenate(Cs)
    ok = np.isfinite(P[:, 0]) & (P[:, 2] > ZMIN) & (P[:, 2] < 0.15)
    ok &= (P[:, 0] > TABLE_X[0]) & (P[:, 0] < TABLE_X[1]) & (P[:, 1] > TABLE_Y[0]) & (P[:, 1] < TABLE_Y[1])
    for xy, r in exclude:
        ok &= np.hypot(P[:, 0] - xy[0], P[:, 1] - xy[1]) > r
    P, rgb = P[ok], rgb[ok]
    nx = int((TABLE_X[1] - TABLE_X[0]) / CELL) + 1
    ny = int((TABLE_Y[1] - TABLE_Y[0]) / CELL) + 1
    ix = ((P[:, 0] - TABLE_X[0]) / CELL).astype(int)
    iy = ((P[:, 1] - TABLE_Y[0]) / CELL).astype(int)
    Z = np.zeros((nx, ny))
    C = np.zeros((nx, ny, 3))
    order = np.argsort(P[:, 2])  # later (higher) overwrite
    Z[ix[order], iy[order]] = P[order, 2]
    C[ix[order], iy[order]] = rgb[order]
    return Z, C


def cell_xy(i, j):
    return np.array([TABLE_X[0] + (i + 0.5) * CELL, TABLE_Y[0] + (j + 0.5) * CELL])


def detect(obs, cam='front', exclude=()):
    Z, C = heightmap(obs, cam, exclude)
    occ = (Z > ZMIN).astype(np.uint8)
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(occ)
    out = []
    for k in range(1, n):
        ii, jj = np.nonzero(lab == k)
        z = Z[ii, jj]
        if len(ii) < 4:
            continue
        zt = float(z.max())
        if zt > CONTAINER_Z:
            sel = z > zt - 0.012
            kind = 'container'
        elif zt > CUBE_TOP_MIN:
            sel = z > zt - 0.010
            kind = 'cube'
        else:
            continue
        xy = np.array([cell_xy(i, j) for i, j in zip(ii[sel], jj[sel])]).mean(0)
        col = C[ii[sel], jj[sel]].mean(0)
        name, dist = classify_color(col)
        # containers may contain several merged objects: split by top cells later if needed
        out.append(dict(kind=kind, xy=xy, ztop=zt, color=name, cdist=float(dist), rgb=col.round().tolist(),
                        n=int(sel.sum()), area=int(len(ii))))
    return out


def detect_color_blobs(obs, cam='front', exclude=()):
    """Saturated colour blobs (cubes) directly in the image; robust to merging with white containers."""
    img = obs.rgb[cam].astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    P = obs.cloud(cam)
    s, v = hsv[..., 1].astype(float) / 255, hsv[..., 2].astype(float) / 255
    out = []
    for name, proto in PROTOS.items():
        if name == 'white':
            continue
        d = np.linalg.norm(img.astype(float) - np.array(proto, float), axis=-1)
        m = (d < 45) & (s > 0.5) & (v > 0.35)
        m &= np.isfinite(P[..., 2]) & (P[..., 2] > 0.004) & (P[..., 2] < 0.06)
        m &= (P[..., 0] > TABLE_X[0]) & (P[..., 0] < TABLE_X[1])
        for xy, r in exclude:
            m &= np.hypot(P[..., 0] - xy[0], P[..., 1] - xy[1]) > r
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < 4:
                continue
            pts = P[lab == k]
            out.append(dict(kind='cube', color=name, xy=np.median(pts[:, :2], 0), ztop=float(np.max(pts[:, 2])),
                            n=int(stats[k, cv2.CC_STAT_AREA])))
    return out
