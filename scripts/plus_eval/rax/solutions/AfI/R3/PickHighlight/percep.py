"""Perception: segment cubes (saturated colour blobs above table) and the button from one camera."""
import numpy as np
from geom import world_map


def label(mask):
    """4-connected component labelling. Returns list of (k,2) arrays of (y, x) pixel indices."""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        if n == 0:
            return []
        ys, xs = np.nonzero(lab)
        ls = lab[ys, xs]
        order = np.argsort(ls, kind="stable")
        ys, xs, ls = ys[order], xs[order], ls[order]
        cuts = np.flatnonzero(np.diff(ls)) + 1
        return [np.stack([a, b], 1) for a, b in zip(np.split(ys, cuts), np.split(xs, cuts))]
    except ImportError:
        pass
    H, W = mask.shape
    seen = np.zeros_like(mask, bool)
    comps = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]; seen[y0, x0] = True; pix = []
        while stack:
            y, x = stack.pop(); pix.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True; stack.append((yy, xx))
        comps.append(np.array(pix))
    return comps


def hsv(rgb):
    r, g, b = [rgb[..., i].astype(float) / 255 for i in range(3)]
    mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn + 1e-9
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    s = np.where(mx > 0, d / (mx + 1e-9), 0)
    return h, s, mx


def blobs(rgb, depth_m, K, pose, min_z=0.01, max_z=0.15, min_px=6):
    """Objects above the table: pixels whose world z > min_z, excluding robot (z>max_z)."""
    Wm = world_map(depth_m, K, pose)
    z = Wm[..., 2]
    mask = (z > min_z) & (z < max_z) & (depth_m > 0)
    out = []
    for pix in label(mask):
        if len(pix) < min_px:
            continue
        ys, xs = pix[:, 0], pix[:, 1]
        pts = Wm[ys, xs]
        col = rgb[ys, xs].astype(float)
        h, s, v = hsv(col[None])
        out.append(dict(n=len(pix), uv=(float(xs.mean()), float(ys.mean())),
                        xyz=np.median(pts, 0), ztop=float(np.percentile(pts[:, 2], 90)),
                        ext=(pts[:, :2].max(0) - pts[:, :2].min(0)),
                        rgb=col.mean(0), hue=float(np.median(h)), sat=float(np.median(s)),
                        val=float(np.median(v))))
    return out


def cube_mask(rgb):
    """Coloured (saturated) pixels; cubes vary in colour per episode (green, purple, ...).
    Used together with a height window, so the (coloured) table is excluded by depth."""
    h, s, v = hsv(rgb)
    return (s > 0.15) & (v > 0.2)


def disc_mask(rgb):
    a = rgb.astype(int)
    return (a.min(2) > 200) & ((a.max(2) - a.min(2)) < 25)


SINGLE_AREA_M2 = 0.0019   # approx visible area of one cube (top + sides) above z=0.022, wrist top-down view


def _kmeans(P, k, iters=15):
    # deterministic init: farthest-point
    C = [P[np.argmax(((P - P.mean(0)) ** 2).sum(1))]]
    while len(C) < k:
        d = np.min([((P - c) ** 2).sum(1) for c in C], 0)
        C.append(P[np.argmax(d)])
    C = np.array(C)
    for _ in range(iters):
        lab = np.argmin(((P[:, None] - C[None]) ** 2).sum(2), 1)
        C = np.array([P[lab == q].mean(0) if np.any(lab == q) else C[q] for q in range(k)])
    return C, lab


def detect_cubes(rgb, depth_m, K, pose, zlo=0.022, zhi=0.075, min_px=5, split=True):
    """Return (list of dict(xy, z, n, area, uv, hue), world map) for cubes in one camera image."""
    Wm = world_map(depth_m, K, pose)
    z = Wm[..., 2]
    m = cube_mask(rgb) & (z > zlo) & (z < zhi) & (depth_m > 0)
    h, s, v = hsv(rgb)
    fx = K[0][0]
    out = []
    for pix in label(m):
        if len(pix) < min_px:
            continue
        ys, xs = pix[:, 0], pix[:, 1]
        pts = Wm[ys, xs]
        area = float(((depth_m[ys, xs] / fx) ** 2).sum())
        k = 1
        if split and area > 1.6 * SINGLE_AREA_M2:
            k = int(min(4, max(2, round(area / SINGLE_AREA_M2))))
        if k == 1:
            groups = [np.arange(len(pts))]
        else:
            C, lab = _kmeans(pts[:, :2], k)
            groups = [np.flatnonzero(lab == q) for q in range(k)]
        for g in groups:
            if len(g) < min_px:
                continue
            out.append(dict(xy=np.median(pts[g, :2], 0), z=float(np.max(pts[g, 2])), n=len(g),
                            area=area / k, split=k, uv=(float(xs[g].mean()), float(ys[g].mean())),
                            hue=float(np.median(h[ys[g], xs[g]]))))
    return out, Wm


def detect_button(rgb, depth_m, K, pose):
    """White/grey low-saturation object on the table, ~7 cm plate. Returns dict(xy, ztop, n) or None."""
    Wm = world_map(depth_m, K, pose)
    z = Wm[..., 2]
    h, s, v = hsv(rgb)
    m = (s < 0.12) & (v > 0.35) & (z > 0.008) & (z < 0.06) & (depth_m > 0)
    best = None
    for pix in label(m):
        if len(pix) < 15:
            continue
        ys, xs = pix[:, 0], pix[:, 1]
        pts = Wm[ys, xs]
        ext = np.ptp(pts[:, :2], 0)
        if ext.max() > 0.12 or ext.min() < 0.03:
            continue
        zt = float(np.percentile(pts[:, 2], 95))
        if zt < 0.02:
            continue  # flat disc, not the button
        cap = pts[pts[:, 2] > zt - 0.006]
        cxy = np.median(cap[:, :2], 0) if len(cap) >= 5 else (pts[:, :2].max(0) + pts[:, :2].min(0)) / 2
        cand = dict(xy=cxy, ztop=zt, n=len(pix), ext=ext)
        if best is None or cand["n"] > best["n"]:
            best = cand
    return best


def disc_score(rgb, Wm, xy, r_in=0.028, r_out=0.045, exclude_xy=None, exclude_half=0.05):
    """Fraction of pixels in an annulus around xy (near table height) that are white disc pixels.
    Pixels within the button plate square (exclude_xy +- exclude_half) are ignored."""
    d = np.sqrt(((Wm[..., :2] - np.asarray(xy)) ** 2).sum(2))
    ann = (d > r_in) & (d < r_out) & (Wm[..., 2] < 0.02)
    if exclude_xy is not None:
        e = np.abs(Wm[..., :2] - np.asarray(exclude_xy)).max(2) < exclude_half
        ann &= ~e
    n = int(ann.sum())
    if n < 5:
        return 0.0, n
    return float(disc_mask(rgb)[ann].mean()), n


def cube_yaw(Wm, rgb, xy, r=0.035, ztop=0.033):
    """Estimate cube yaw (rad, mod pi/2, in [-pi/4, pi/4)) from top-face points around xy."""
    d = np.sqrt(((Wm[..., :2] - np.asarray(xy)) ** 2).sum(2))
    m = (d < r) & (Wm[..., 2] > ztop) & (Wm[..., 2] < 0.07) & cube_mask(rgb)
    P = Wm[m][:, :2]
    if len(P) < 8:
        return None, len(P)
    P = P - P.mean(0)
    best = None
    for a in np.deg2rad(np.arange(-45, 45, 1.0)):
        c, s = np.cos(a), np.sin(a)
        u = P @ np.array([c, s]); v = P @ np.array([-s, c])
        area = np.ptp(u) * np.ptp(v)
        if best is None or area < best[0]:
            best = (area, a)
    return float(best[1]), len(P)
