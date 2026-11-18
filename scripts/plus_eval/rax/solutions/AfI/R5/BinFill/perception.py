"""Color/depth perception for BinFill: cubes, bin, button, conveyor mat. Public RGB-D only.

Tested layouts: ep1 static wooden table (dark lighting, cube top z~0.041),
ep2 conveyor mat (bright lighting, cube top z~0.056, distractor colours yellow/orange/cyan/magenta).
"""
import math
import numpy as np
from obs import world_map

try:
    from scipy import ndimage as _nd
except Exception:  # pragma: no cover
    _nd = None

CUBE_TOP_Z = (0.025, 0.085)   # top face of a cube resting on table (0.041) or mat (0.056)


def label(mask):
    if _nd is not None:
        return _nd.label(mask)
    H, W = mask.shape; lab = np.zeros((H, W), int); n = 0
    for y in range(H):
        for x in range(W):
            if mask[y, x] and not lab[y, x]:
                n += 1; st = [(y, x)]; lab[y, x] = n
                while st:
                    a, b = st.pop()
                    for c, d in ((a+1, b), (a-1, b), (a, b+1), (a, b-1)):
                        if 0 <= c < H and 0 <= d < W and mask[c, d] and not lab[c, d]:
                            lab[c, d] = n; st.append((c, d))
    return lab, n


def blobs(mask, min_pix):
    lab, n = label(mask)
    if n == 0:
        return []
    if _nd is not None:
        idx = _nd.find_objects(lab)
        out = []
        for i, sl in enumerate(idx, 1):
            if sl is None:
                continue
            sub = lab[sl] == i
            if sub.sum() < min_pix:
                continue
            ys, xs = np.nonzero(sub)
            out.append((ys + sl[0].start, xs + sl[1].start))
        return out
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) >= min_pix:
            out.append((ys, xs))
    return out


def color_masks(rgb):
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    return {
        "red": (r > 35) & (r > 3 * g + 10) & (r > 3 * b + 10),
        "blue": (b > 28) & (b > 3 * r + 10) & (b > 3 * g + 10),
        "green": (g > 28) & (g > 3 * r + 10) & (g > 3 * b + 10),
    }


def square_yaw(xy):
    """Yaw in [-pi/4, pi/4) of the min-area bounding square of 2-D points."""
    best = (1e9, 0.0)
    c = xy - xy.mean(0)
    for deg in np.arange(-45, 45, 3.0):
        a = math.radians(deg)
        R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        q = c @ R.T
        area = np.ptp(q[:, 0]) * np.ptp(q[:, 1])
        if area < best[0]:
            best = (area, a)
    return best[1]


def detect_cubes(meta, imgs, cam, min_pix=6, W=None, colors=None):
    """List of dicts: color, top (xy of top-face median, z of top), yaw, npix, ntop, uv, cam."""
    rgb = imgs[(cam, "rgb")]
    if W is None:
        W = world_map(meta, imgs, cam)
    out = []
    for color, m in color_masks(rgb).items():
        if colors is not None and color not in colors:
            continue
        for ys, xs in blobs(m, min_pix):
            P = W[ys, xs]; ok = np.isfinite(P).all(1)
            if ok.sum() < min_pix:
                continue
            P = P[ok]
            ztop = float(np.percentile(P[:, 2], 90))
            top = P[P[:, 2] > ztop - 0.006]
            if len(top) < 3:
                top = P
            cxy = np.median(top[:, :2], 0)
            yaw = square_yaw(top[:, :2]) if len(top) >= 8 else 0.0
            out.append(dict(color=color, top=np.array([cxy[0], cxy[1], ztop]),
                            yaw=yaw, npix=int(len(xs)), ntop=int(len(top)),
                            uv=(float(xs.mean()), float(ys.mean())), cam=cam))
    return out


def resting_cubes(dets, bin_xy=None, min_top=5):
    out = []
    for d in dets:
        t = d["top"]
        if not (CUBE_TOP_Z[0] < t[2] < CUBE_TOP_Z[1]) or d["ntop"] < min_top:
            continue
        if bin_xy is not None and np.linalg.norm(t[:2] - bin_xy[:2]) < 0.07:
            continue
        out.append(d)
    return out


def _lowsat(rgb, tol=14):
    rgb = rgb.astype(int)
    return (rgb.max(-1) - rgb.min(-1)) < tol


def find_bin(meta, imgs, cam, W=None, near=None):
    """Bin: dark interior blob (z<0.045) + its low-saturation rim.  Returns dict(center, rim_z, n, cam)."""
    rgb = imgs[(cam, "rgb")].astype(int)
    if W is None:
        W = world_map(meta, imgs, cam)
    Z = W[..., 2]
    fin = np.isfinite(Z)
    dark = (rgb.max(-1) < 22) & fin & (Z > -0.02) & (Z < 0.045)
    best = None
    for ys, xs in blobs(dark, 12):
        c = np.median(W[ys, xs], 0)
        if abs(c[1]) > 0.6 or c[0] < -0.6 or c[0] > 0.4:
            continue
        if near is not None and np.linalg.norm(c[:2] - near[:2]) > 0.12:
            continue
        if best is None or len(xs) > best[1]:
            best = (c, len(xs), ys, xs)
    if best is None:
        return None
    c, n, ys, xs = best
    sat = rgb.max(-1) - rgb.min(-1)
    rim = fin & (Z > 0.022) & (Z < 0.075) & (sat < 70) & ~((rgb.max(-1) < 22))
    D = np.linalg.norm(W[..., :2] - c[:2], axis=-1)
    rim &= D < 0.085
    P = W[rim]
    Pd = W[ys, xs]
    allp = np.concatenate([P[:, :2], Pd[:, :2]], 0) if len(P) else Pd[:, :2]
    if len(P) >= 25:
        lo = np.percentile(allp, 3, axis=0); hi = np.percentile(allp, 97, axis=0)
        mid = (lo + hi) / 2
        rim_z = float(np.percentile(P[:, 2], 90))
    else:
        mid = c[:2]; rim_z = 0.05
    return dict(center=np.array([mid[0], mid[1]]), dark_center=c[:2], rim_z=rim_z,
                n=int(n), nrim=int(len(P)), cam=cam)


def find_button(meta, imgs, cam, bin_xy=None, W=None, near=None):
    """Button cap: low-saturation blob with top z 0.026-0.045 (cap ~0.033), base at ~0.016."""
    rgb = imgs[(cam, "rgb")].astype(int)
    if W is None:
        W = world_map(meta, imgs, cam)
    Z = W[..., 2]
    m = _lowsat(rgb, 14) & np.isfinite(Z) & (Z > 0.025) & (Z < 0.045) & (rgb.max(-1) > 30)
    best = None
    for ys, xs in blobs(m, 6):
        P = W[ys, xs]
        c = np.median(P, 0)
        if bin_xy is not None and np.linalg.norm(c[:2] - bin_xy[:2]) < 0.10:
            continue
        if abs(c[1]) > 0.55 or c[0] < -0.48 or c[0] > 0.35:
            continue
        ext = np.ptp(P[:, :2], 0)
        if ext.max() > 0.07:      # bigger than a button cap
            continue
        if near is not None and np.linalg.norm(c[:2] - near[:2]) > 0.05:
            continue
        score = len(xs)
        if best is None or score > best[1]:
            best = (np.array([c[0], c[1], float(np.percentile(P[:, 2], 90))]), score)
    return None if best is None else best[0]


def find_mat(meta, imgs, cam="front", W=None):
    """Conveyor mat: large low-saturation region with z 0.006-0.026 (not table, not button)."""
    rgb = imgs[(cam, "rgb")].astype(int)
    if W is None:
        W = world_map(meta, imgs, cam)
    Z = W[..., 2]
    m = _lowsat(rgb, 16) & np.isfinite(Z) & (Z > 0.006) & (Z < 0.026)
    best = None
    for ys, xs in blobs(m, 400):
        if best is None or len(xs) > len(best[0]):
            best = (xs, ys)
    if best is None:
        return None
    P = W[best[1], best[0]]
    return dict(lo=np.percentile(P[:, :2], 1, axis=0), hi=np.percentile(P[:, :2], 99, axis=0),
                z=float(np.median(P[:, 2])), n=len(P))


def match_velocity(dets0, dets1, dt, max_disp=0.045):
    """Median displacement per step of same-colour nearest-neighbour matches (static: ~0)."""
    if dt <= 0:
        return None, 0
    disp = []
    for d1 in dets1:
        best = None
        for d0 in dets0:
            if d0["color"] != d1["color"]:
                continue
            dd = d1["top"][:2] - d0["top"][:2]
            n = np.linalg.norm(dd)
            if n < max_disp and (best is None or n < best[0]):
                best = (n, dd)
        if best is not None:
            disp.append(best[1])
    if len(disp) < 2:
        return None, len(disp)
    return np.median(np.array(disp), 0) / dt, len(disp)
