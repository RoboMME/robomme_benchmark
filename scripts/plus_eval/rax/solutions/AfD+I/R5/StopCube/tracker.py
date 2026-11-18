import numpy as np
from scipy import ndimage
from perception import decode, backproject, hsv

CUBE_ZMIN, CUBE_ZMAX = 0.022, 0.075

def high_blobs(pw, zmin=CUBE_ZMIN, zmax=CUBE_ZMAX, minpix=6):
    z = pw[..., 2]
    mk = (z > zmin) & (z < zmax) & np.isfinite(z)
    lab, n = ndimage.label(mk)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < minpix:
            continue
        P = pw[ys, xs]
        out.append(dict(n=len(xs), uv=(xs.mean(), ys.mean()), xyz=np.nanmedian(P, 0), zmax=float(np.nanmax(P[:, 2]))))
    return out

def target_points(pw, rgb):
    z = pw[..., 2]
    h, s, v = hsv(rgb)
    flat = (z > 0.009) & (z < 0.021)
    tablelike = (h < 45) & (s > 0.3)
    mk = flat & ~tablelike
    return mk

def cube_top(pw, button_xy, exclude_r=0.06, target_xy=None, search_r=0.25):
    """Return xy of cube top (points with z near the top) not near the button."""
    z = pw[..., 2]
    mk = (z > CUBE_ZMIN) & (z < CUBE_ZMAX) & np.isfinite(z)
    P = pw[mk]
    if target_xy is not None and len(P):
        d = np.hypot(P[:, 0] - target_xy[0], P[:, 1] - target_xy[1])
        P = P[d < search_r]
    if button_xy is not None and len(P):
        d = np.hypot(P[:, 0] - button_xy[0], P[:, 1] - button_xy[1])
        P = P[d > exclude_r]
    if len(P) < 4:
        return None
    # largest cluster: take points within 4cm of median
    med = np.median(P[:, :2], 0)
    d = np.hypot(P[:, 0] - med[0], P[:, 1] - med[1])
    P = P[d < 0.04]
    if len(P) < 4:
        return None
    return np.median(P[:, :2], 0), len(P)
