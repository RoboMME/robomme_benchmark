import numpy as np
from vision import deproject, components

def above_table(rgb, dep, intr, pose, zmin=0.006, zmax=0.08):
    P = deproject(dep, intr, pose)
    m = (dep > 0) & (P[..., 2] > zmin) & (P[..., 2] < zmax) & (P[..., 0] > -0.45) & (P[..., 0] < 0.35) & (np.abs(P[..., 1]) < 0.45)
    return P, m

def find_button(rgb, dep, intr, pose):
    P, m = above_table(rgb, dep, intr, pose, zmin=0.004)
    mx = rgb.max(-1); mn = rgb.min(-1)
    low = (mx - mn < 40) & (mx > 80)
    best = None
    for xs, ys in components(m & low, 8):
        Q = P[ys, xs]
        lo, hi = Q[:, :2].min(0), Q[:, :2].max(0)
        ext = hi - lo
        score = len(xs) * (1.0 if np.all(ext < 0.11) else 0.2)
        if best is None or score > best[0]:
            best = (score, dict(xy=Q[:, :2].mean(0), lo=lo, hi=hi, ext=ext, n=len(xs), ztop=float(np.percentile(Q[:, 2], 95))))
    return None if best is None else best[1]

def drop_tall_attached(P, dep, m, ztall=0.075, ddepth=0.02):
    """remove mask components that continue (depth-continuously) into tall structures."""
    from scipy import ndimage
    tall = (dep > 0) & (P[..., 2] > ztall)
    if not tall.any():
        return m
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    out = m.copy()
    dil = ndimage.grey_dilation(lab, size=(5, 5))
    ring = (dil > 0) & ~m & tall
    ys, xs = np.nonzero(ring)
    if len(ys) == 0:
        return m
    # for each ring pixel compare its depth with the max depth window of the component
    labs = dil[ys, xs]
    bad = set()
    dmax = ndimage.maximum(dep, lab, index=np.arange(1, n + 1)); dmin = ndimage.minimum(dep, lab, index=np.arange(1, n + 1))
    for y, x, l in zip(ys, xs, labs):
        if l in bad:
            continue
        d = dep[y, x]
        if dmin[l - 1] - ddepth < d < dmax[l - 1] + ddepth:
            bad.add(l)
    for l in bad:
        out[lab == l] = False
    return out

def object_points(rgb, dep, intr, pose, button=None, margin=0.012):
    P, m = above_table(rgb, dep, intr, pose)
    if button is not None:
        lo = np.asarray(button["lo"]) - margin; hi = np.asarray(button["hi"]) + margin
        inb = (P[..., 0] > lo[0]) & (P[..., 0] < hi[0]) & (P[..., 1] > lo[1]) & (P[..., 1] < hi[1])
        m = m & ~inb
    m = drop_tall_attached(P, dep, m)
    ys, xs = np.nonzero(m)
    return P[ys, xs], np.stack([xs, ys], 1)

def object_blobs(rgb, dep, intr, pose, button=None, min_px=15):
    pts, uv = object_points(rgb, dep, intr, pose, button)
    mask = np.zeros(dep.shape, bool); mask[uv[:, 1], uv[:, 0]] = True
    idx = -np.ones(dep.shape, int); idx[uv[:, 1], uv[:, 0]] = np.arange(len(uv))
    out = []
    for xs, ys in components(mask, min_px):
        Q = pts[idx[ys, xs]]
        out.append(dict(xy=Q[:, :2].mean(0), n=len(xs), pts=Q, zmax=float(Q[:, 2].max()), uv=(xs.mean(), ys.mean())))
    return out


def merge_buttons(a, b):
    if a is None:
        return b
    if b is None or np.linalg.norm(np.asarray(a["xy"]) - np.asarray(b["xy"])) > 0.06:
        return a
    lo = np.minimum(a["lo"], b["lo"]); hi = np.maximum(a["hi"], b["hi"])
    out = dict(a); out.update(lo=lo, hi=hi, ext=hi - lo, xy=(lo + hi) / 2)
    return out
