import numpy as np
from geom import backproject

def label(mask):
    """4-connected component labelling (numpy + python BFS); returns list of index arrays."""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]; seen[y0, x0] = True; pts = []
        while stack:
            y, x = stack.pop(); pts.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True; stack.append((yy, xx))
        comps.append(np.array(pts))
    return comps

def world_points(depth_mm, K, pose):
    return backproject(depth_mm.astype(float) * 0.001, K, pose)

def detect_targets(P, zlo=0.008, zhi=0.03, min_area=6):
    """Raised target discs: points with height zlo..zhi above table. Returns Nx3 centroids."""
    z = P[..., 2]
    mask = np.isfinite(z) & (z > zlo) & (z < zhi)
    out = []
    for c in label(mask):
        if len(c) < min_area:
            continue
        pts = P[c[:, 0], c[:, 1]]
        out.append(np.nanmedian(pts, 0))
    return np.array(out)

def fit_grid(pts, n=5):
    """Fit an n x n square grid. Returns dict(center, theta, spacing, idx(Nx2), resid)."""
    xy = pts[:, :2]
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min(1)
    s = float(np.median(nn))
    # orientation from neighbour vectors of length ~s
    angs = []
    for i in range(len(xy)):
        for j in range(len(xy)):
            if i != j and abs(d[i, j] - s) < 0.25 * s:
                v = xy[j] - xy[i]
                angs.append(np.arctan2(v[1], v[0]))
    angs = np.array(angs)
    # mean of 4*angle on circle
    th = np.arctan2(np.mean(np.sin(4 * angs)), np.mean(np.cos(4 * angs))) / 4.0  # in (-45,45]
    c = xy.mean(0)
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])  # rotate by -th
    loc = (xy - c) @ R.T / s
    # refine center so that indices are integer-centred
    off = (n - 1) / 2.0
    idx = np.round(loc + off).astype(int)
    resid = np.linalg.norm(loc + off - idx, axis=1)
    return dict(center=c, theta=float(th), spacing=s, idx=idx, resid=resid, local=loc)

def grid_to_world(g, ij, n=5):
    th, s, c = g["theta"], g["spacing"], g["center"]
    off = (n - 1) / 2.0
    loc = (np.asarray(ij, float) - off) * s
    Rinv = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return loc @ Rinv.T + c

def red_mask(rgb):
    a = rgb.astype(int)
    return (a[..., 0] > 140) & (a[..., 1] < 80) & (a[..., 2] < 80) & (a[..., 0] - a[..., 1] > 90)
