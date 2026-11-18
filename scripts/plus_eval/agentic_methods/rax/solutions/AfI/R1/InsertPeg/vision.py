"""Depth / RGB perception helpers: calibration, back-projection, segmentation."""
import numpy as np
from collections import deque

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

from mcpio import log, find_key, find_all_keys, _is_num_list


# ---------------------------------------------------------------- image IO
def load_depth(path, mpu=0.001):
    im = Image.open(path)
    a = np.array(im)
    if a.ndim == 3:
        a = a[..., 0]
    d = a.astype(np.float64) * float(mpu or 0.001)
    d[a == 0] = np.nan
    return d


def load_rgb(path):
    im = Image.open(path).convert("RGB")
    return np.array(im).astype(np.float64)


# ---------------------------------------------------------------- calibration
class Calib:
    """World point = M @ [u*d, v*d, d, 1] with u=column, v=row (pixel index)."""

    def __init__(self, M, source="?", resid=None):
        self.M = np.asarray(M, dtype=float)
        self.source = source
        self.resid = resid

    def backproject(self, depth):
        H, W = depth.shape
        v, u = np.mgrid[0:H, 0:W].astype(np.float64)
        d = depth
        X = np.stack([u * d, v * d, d, np.ones_like(d)], axis=-1)
        P = X @ self.M.T
        P[~np.isfinite(d)] = np.nan
        return P

    def project(self, P):
        """World point -> (u, v, d) by solving the linear model (3x3 part)."""
        A = self.M[:, :3]
        b = np.asarray(P, float) - self.M[:, 3]
        x = np.linalg.solve(A, b)
        d = x[2]
        return x[0] / d, x[1] / d, d


def fit_calib(samples, depth):
    """samples: list of (u, v, P) from pixel_to_position. Tries both readings of the
    tool's (u, v): (column, row) and (row, column). Returns (Calib, rms) or None."""
    best = None
    H, W = depth.shape
    for swap in (False, True):
        rows, ys = [], []
        for (u, v, P) in samples:
            col, row = (v, u) if swap else (u, v)
            ci, ri = int(round(col)), int(round(row))
            if not (0 <= ci < W and 0 <= ri < H):
                continue
            d = depth[ri, ci]
            if not np.isfinite(d):
                continue
            rows.append([col * d, row * d, d, 1.0])
            ys.append(P)
        if len(rows) < 8:
            continue
        A = np.array(rows)
        Y = np.array(ys)
        Mt, *_ = np.linalg.lstsq(A, Y, rcond=None)
        res = A @ Mt - Y
        rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
        if best is None or rms < best[1]:
            best = (Calib(Mt.T, "fit_swap" if swap else "fit", rms), rms, swap)
    return best


def _quat_wxyz_to_mat(q):
    w, x, y, z = [float(a) for a in q]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def calib_from_json(j, cam="front"):
    """Build calibration from camera_intrinsics / camera_poses (OpenGL convention)."""
    try:
        intr = find_key(j, ["camera_intrinsics"])
        poses = find_key(j, ["camera_poses"])
        if intr is None or poses is None:
            return None
        ci = intr.get(cam) if isinstance(intr, dict) else None
        if ci is None and isinstance(intr, list):
            ci = next((c for c in intr if isinstance(c, dict) and c.get("camera") == cam), None)
        cp = poses.get(cam) if isinstance(poses, dict) else None
        if cp is None and isinstance(poses, list):
            cp = next((c for c in poses if isinstance(c, dict) and c.get("camera") == cam), None)
        if ci is None or cp is None:
            return None
        K = None
        for k in ("K", "matrix", "intrinsic_matrix", "camera_matrix", "intrinsics"):
            v = ci.get(k) if isinstance(ci, dict) else None
            if v is not None:
                K = np.array(v, dtype=float).reshape(3, 3)
                break
        if K is None and isinstance(ci, dict) and "fx" in ci:
            K = np.array([[ci["fx"], 0, ci.get("cx", 127.5)], [0, ci["fy"], ci.get("cy", 127.5)], [0, 0, 1]], float)
        if K is None and isinstance(ci, list):
            K = np.array(ci, dtype=float).reshape(3, 3)
        if K is None:
            return None
        pos = None
        for k in ('position_m', "position", "p", "translation", "xyz"):
            if isinstance(cp, dict) and _is_num_list(cp.get(k), 3):
                pos = np.array(cp[k], float)
                break
        q = None
        for k in ("quaternion_wxyz", "q", "quaternion", "wxyz"):
            if isinstance(cp, dict) and _is_num_list(cp.get(k), 4):
                q = cp[k]
                break
        R = None
        if q is not None:
            R = _quat_wxyz_to_mat(q)
        elif isinstance(cp, dict) and cp.get("matrix") is not None:
            T = np.array(cp["matrix"], float).reshape(4, 4)
            R, pos = T[:3, :3], T[:3, 3]
        if R is None or pos is None:
            return None
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        # camera (OpenGL): x right, y up, looks along -z. cam = [ (u-cx)/fx d, -(v-cy)/fy d, -d ]
        C = np.array([[1 / fx, 0, -cx / fx], [0, -1 / fy, cy / fy], [0, 0, -1.0]])
        M = np.zeros((3, 4))
        M[:, :3] = R @ C
        M[:, 3] = pos
        return Calib(M, "json")
    except Exception:
        return None


def _pt_from(v):
    if _is_num_list(v, 3):
        return [float(x) for x in v]
    if isinstance(v, dict) and all(isinstance(v.get(k), (int, float)) for k in ("x", "y", "z")):
        return [float(v["x"]), float(v["y"]), float(v["z"])]
    return None


_PT_KEYS = ("position", "world_position", "point", "world", "xyz", "position_m", "world_point",
            "point_world", "world_xyz", "position_world", "world_position_m", "point_m")


def parse_p2p(j, n):
    """Extract up to n (point or None, raw) from a pixel_to_position reply, in order."""
    cands = []

    def point_of(o):
        for k in _PT_KEYS:
            if k in o:
                p = _pt_from(o[k])
                if p is not None:
                    return p
        for k, v in o.items():
            kl = str(k).lower()
            if ("pos" in kl or "world" in kl or "point" in kl or "xyz" in kl) and "pixel" not in kl:
                p = _pt_from(v)
                if p is not None:
                    return p
        if all(isinstance(o.get(k), (int, float)) for k in ("x", "y", "z")):
            return [float(o["x"]), float(o["y"]), float(o["z"])]
        if depth_ok[0]:
            for v in o.values():
                if isinstance(v, dict):
                    depth_ok[0] = False
                    p = point_of(v)
                    depth_ok[0] = True
                    if p is not None:
                        return p
        return None

    depth_ok = [True]

    def walk(o, depth=0):
        if depth > 7:
            return
        if isinstance(o, dict):
            pt = point_of(o)
            is_entry = pt is not None or ("valid" in o and ("u" in o or "v" in o or "camera" in o))
            if is_entry:
                valid = o.get("valid", True)
                cands.append((pt if valid is not False else None, o))
                return
            for v in o.values():
                walk(v, depth + 1)
        elif isinstance(o, list):
            if len(o) >= 1 and all(_is_num_list(x, 3) for x in o) and len(o) != 3:
                for x in o:
                    cands.append(([float(a) for a in x], None))
                return
            for v in o:
                walk(v, depth + 1)

    walk(j)
    return cands[:n]


# ---------------------------------------------------------------- segmentation
def table_height(P):
    z = P[..., 2]
    z = z[np.isfinite(z)]
    if z.size < 100:
        return None
    lo, hi = np.percentile(z, 1), np.percentile(z, 99)
    bins = np.arange(lo - 0.002, hi + 0.004, 0.002)
    if len(bins) < 3:
        return float(np.median(z))
    h, e = np.histogram(z, bins=bins)
    i = int(np.argmax(h))
    sel = z[(z >= e[i] - 0.002) & (z <= e[i + 1] + 0.002)]
    return float(np.median(sel))


def components(mask, P, max_jump=0.015, min_size=15):
    """Connected components of mask with 3-D continuity. Returns label image, list of sizes."""
    H, W = mask.shape
    lab = -np.ones((H, W), dtype=np.int32)
    sizes = []
    cur = 0
    for r0 in range(H):
        row = mask[r0]
        for c0 in np.nonzero(row)[0]:
            if lab[r0, c0] >= 0:
                continue
            q = deque([(r0, c0)])
            lab[r0, c0] = cur
            n = 0
            while q:
                r, c = q.popleft()
                n += 1
                p = P[r, c]
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and lab[rr, cc] < 0:
                        if np.sum((P[rr, cc] - p) ** 2) < max_jump * max_jump:
                            lab[rr, cc] = cur
                            q.append((rr, cc))
            sizes.append(n)
            cur += 1
    # drop tiny
    out = -np.ones_like(lab)
    keep = []
    for i, s in enumerate(sizes):
        if s >= min_size:
            out[lab == i] = len(keep)
            keep.append(s)
    return out, keep


def comp_stats(P, rgb, sel, tz):
    pts = P[sel]
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < 5:
        return None
    zmax = float(np.percentile(pts[:, 2], 98))
    top = pts[pts[:, 2] > zmax - 0.005]
    if len(top) < 5:
        top = pts
    xy = top[:, :2]
    c0 = xy.mean(axis=0)
    cov = np.cov((xy - c0).T) if len(xy) > 2 else np.eye(2) * 1e-6
    w, V = np.linalg.eigh(cov)
    ax = V[:, 1]
    yaw = float(np.arctan2(ax[1], ax[0]))
    proj = (xy - c0) @ ax
    perp = (xy - c0) @ V[:, 0]
    a_lo, a_hi = np.percentile(proj, 1), np.percentile(proj, 99)
    p_lo, p_hi = np.percentile(perp, 1), np.percentile(perp, 99)
    length = float(a_hi - a_lo)
    width = float(p_hi - p_lo)
    # mid-range centre: side-face points lie on the footprint edge, so this is unbiased
    c = c0 + ax * 0.5 * (a_lo + a_hi) + V[:, 0] * 0.5 * (p_lo + p_hi)
    col = rgb[sel].mean(axis=0) if rgb is not None else np.zeros(3)
    return {
        "n": int(sel.sum()), "c_top": c, "c3": pts.mean(axis=0), "height": zmax - tz,
        "yaw": yaw, "axis": ax, "length": length, "width": width, "color": col,
        "zmin": float(np.percentile(pts[:, 2], 2)),
    }


def half_colors(P, rgb, sel, st):
    """Mean colour of the two halves along the principal axis (+axis half, -axis half)."""
    pts = P[sel]
    cols = rgb[sel]
    ok = np.all(np.isfinite(pts), axis=1)
    pts, cols = pts[ok], cols[ok]
    s = (pts[:, :2] - st["c_top"]) @ st["axis"]
    a = cols[s > 0.01]
    b = cols[s < -0.01]
    if len(a) < 5 or len(b) < 5:
        return None
    return a.mean(axis=0), b.mean(axis=0)
