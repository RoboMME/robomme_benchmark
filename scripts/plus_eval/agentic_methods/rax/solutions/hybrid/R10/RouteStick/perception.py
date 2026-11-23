import numpy as np
from PIL import Image
from geom import backproject

def load_rgbd(rgb_path, depth_path, mpu=0.001):
    rgb = np.asarray(Image.open(rgb_path).convert("RGB")).astype(float)
    dep = np.asarray(Image.open(depth_path)).astype(float) * mpu
    return rgb, dep

def world_map(dep, K, pose):
    H, W = dep.shape
    v, u = np.mgrid[0:H, 0:W].astype(float)
    P = backproject(u, v, dep, K, pose)
    P[dep <= 0] = np.nan
    return P

def _components(mask):
    """4-connected components; returns list of (N,2) pixel arrays (v,u)."""
    H, W = mask.shape
    lab = -np.ones((H, W), int); comps = []
    for v0, u0 in zip(*np.nonzero(mask)):
        if lab[v0, u0] >= 0: continue
        stack = [(v0, u0)]; lab[v0, u0] = len(comps); pix = []
        while stack:
            v, u = stack.pop(); pix.append((v, u))
            for dv, du in ((1,0),(-1,0),(0,1),(0,-1)):
                a, b = v+dv, u+du
                if 0 <= a < H and 0 <= b < W and mask[a, b] and lab[a, b] < 0:
                    lab[a, b] = len(comps); stack.append((a, b))
        comps.append(np.array(pix))
    return comps

def detect_targets(rgb, dep, K, pose, n_expected=5):
    """Bright low-saturation disc pixels slightly above the table."""
    P = world_map(dep, K, pose)
    mx = rgb.max(2); mn = rgb.min(2)
    z = P[..., 2]
    mask = (mn > 150) & ((mx - mn) < 45) & (z > 0.004) & (z < 0.035)
    mask &= np.isfinite(z)
    comps = [c for c in _components(mask) if len(c) >= 6]
    cents = []
    for c in comps:
        pts = P[c[:, 0], c[:, 1]]
        cents.append((len(c), np.nanmedian(pts, 0), c.mean(0)))
    cents.sort(key=lambda t: -t[0])
    return cents

def detect_bars(rgb, dep, K, pose):
    """Saturated or dark tall objects (bars); returns centroid of top-ish points per component."""
    P = world_map(dep, K, pose)
    z = P[..., 2]
    mask = np.isfinite(z) & (z > 0.03) & (z < 0.25) & (np.abs(P[..., 0]) < 0.6)
    comps = [c for c in _components(mask) if len(c) >= 15]
    out = []
    for c in comps:
        pts = P[c[:, 0], c[:, 1]]
        out.append((len(c), np.nanmedian(pts, 0), c.mean(0), np.nanmax(pts[:, 2])))
    out.sort(key=lambda t: -t[0])
    return out

ROW_X_MIN = -0.45   # exclude robot base region (base sits near x=-0.56)

def find_row_targets(rgb, dep, K, pose, n=5):
    """Return (n,3) world positions of the row targets ordered by world y, or None."""
    cands = [c for c in detect_targets(rgb, dep, K, pose) if c[0] >= 15 and c[1][0] > ROW_X_MIN]
    if len(cands) < n:
        return None, cands
    pts = np.array([c[1] for c in cands[:n]])
    # order along principal axis of the row, oriented to increasing y
    xy = pts[:, :2] - pts[:, :2].mean(0)
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    ax = vt[0] if vt[0][1] >= 0 else -vt[0]
    order = np.argsort(xy @ ax)
    return pts[order], cands

def stick_tip(f_files, K, pose, tcp, max_depth=0.2):
    """Estimate the stick tip (world) from a wrist RGB-D frame: gray pixels near the
    camera connected to the bottom border; tip = stick pixel farthest from the camera centre
    along the image (topmost-in-component), refined with depth."""
    rgb, dep = load_rgbd(f_files[0], f_files[1])
    P = world_map(dep, K, pose)
    mx = rgb.max(2); mn = rgb.min(2)
    gray = ((mx - mn) < 14) & (mn > 90) & (dep > 0) & (dep < max_depth)
    comps = _components(gray)
    best = None
    for c in comps:
        if len(c) < 50: continue
        if c[:, 0].max() < 250: continue   # must touch the bottom border
        if best is None or len(c) > len(best): best = c
    if best is None:
        return None, None
    pts = P[best[:, 0], best[:, 1]]
    dd = dep[best[:, 0], best[:, 1]]
    # tip: the deepest part of the stick (farthest from camera)
    k = max(5, len(best) // 50)
    idx = np.argsort(-dd)[:k]
    tip = np.nanmedian(pts[idx], 0)
    return tip, dict(n=len(best), tip_px=best[idx].mean(0).tolist(), depth=float(np.median(dd[idx])))

def find_bars(rgb, dep, K, pose, n=4, zlo=0.06, zhi=0.125):
    """Bar positions (world xy of the visible part between zlo and zhi), ordered along y."""
    P = world_map(dep, K, pose)
    z = P[..., 2]
    mask = np.isfinite(z) & (z > zlo) & (z < zhi) & (P[..., 0] > ROW_X_MIN)
    cs = [c for c in _components(mask) if len(c) >= 8]
    cs.sort(key=lambda c: -len(c))
    pts = np.array([np.nanmedian(P[c[:, 0], c[:, 1]], 0) for c in cs[:n]])
    if len(pts) == 0:
        return None
    return pts[np.argsort(pts[:, 1])]
