"""Green-object segmentation and back-projection helpers (pure numpy/PIL)."""
import numpy as np
from PIL import Image


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(u, v, d, K, pose):
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    pc = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], axis=-1)
    R = quat_to_rot(pose["quaternion_wxyz"])
    return pc @ R.T + np.asarray(pose["position_m"])


def project(pw, K, pose):
    R = quat_to_rot(pose["quaternion_wxyz"])
    pc = (np.asarray(pw) - np.asarray(pose["position_m"])) @ R
    d = -pc[..., 2]
    u = K[0][0] * pc[..., 0] / d + K[0][2]
    v = -K[1][1] * pc[..., 1] / d + K[1][2]
    return u, v, d


OBJ_HUE = None  # learned per episode (set_object_hue); None -> legacy green rule
HUE_TOL = 0.06
SAT_MIN = 0.3


def set_object_hue(h):
    global OBJ_HUE
    OBJ_HUE = None if h is None else float(h)


def green_mask(rgb):
    """Object-colour mask (name kept for compatibility)."""
    if OBJ_HUE is None:
        a = rgb.astype(np.int32)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        return (g > r + 25) & (g > b + 25) & (g > 60)
    hsv = rgb_to_hsv(rgb.astype(np.float64) / 255.0)
    dh = np.abs(hsv[..., 0] - OBJ_HUE); dh = np.minimum(dh, 1 - dh)
    return (dh < HUE_TOL) & (hsv[..., 1] > SAT_MIN) & (hsv[..., 2] > 0.12)


def label(mask):
    """4-connected components; returns label image and count."""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    n = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]:
            continue
        n += 1
        stack = [(y0, x0)]
        lab[y0, x0] = n
        while stack:
            y, x = stack.pop()
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n
                    stack.append((yy, xx))
    return lab, n


def green_points_world(rgb, depth_m, K, pose):
    """World xyz of every valid green pixel (N,3) plus pixel coords."""
    m = green_mask(rgb) & (depth_m > 0)
    vs, us = np.nonzero(m)
    d = depth_m[vs, us]
    pw = backproject(us.astype(float), vs.astype(float), d, K, pose)
    return pw, us, vs


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def load_depth(path, scale=0.001):
    return np.asarray(Image.open(path)).astype(np.float64) * scale


def min_enclosing_circle(xy, iters=400):
    c = xy.mean(0).copy()
    for i in range(1, iters + 1):
        d = np.linalg.norm(xy - c, axis=1)
        far = xy[d.argmax()]
        c = c + (far - c) / (i + 1)
    return c, np.linalg.norm(xy - c, axis=1).max()


EXCLUDE_XY = []  # (xy, radius) world regions to ignore (button)


def object_pose_from_view(rgb, depth_m, K, pose, near_xy, radius=0.04, ztop=None):
    """Locate the object nearest near_xy. Returns dict(center, R, centroid, bite_vec, n_top, n_all, top_xy...)."""
    pw, us, vs = green_points_world(rgb, depth_m, K, pose)
    for c, r in EXCLUDE_XY:
        if len(pw):
            pw = pw[np.linalg.norm(pw[:, :2] - np.asarray(c), axis=1) > r]
    if len(pw) == 0:
        return None
    sel = (np.linalg.norm(pw[:, :2] - near_xy, axis=1) < radius) & (pw[:, 2] < 0.08)
    allp = pw[sel]
    if len(allp) < 10:
        return None
    # refine: recentre on these points and reselect
    c0 = allp[:, :2].mean(0)
    sel = (np.linalg.norm(pw[:, :2] - c0, axis=1) < 0.03) & (pw[:, 2] < 0.08)
    allp = pw[sel]
    zmax = float(np.percentile(allp[:, 2], 98))
    if ztop is None:
        ztop = zmax - 0.009
    top = allp[allp[:, 2] > ztop]
    out = dict(n_all=int(len(allp)), n_top=int(len(top)), zmax=zmax, top_xy=top[:, :2], all_xy=allp[:, :2])
    if len(top) >= 15:
        c, R = min_enclosing_circle(top[:, :2])
        g = top[:, :2].mean(0)
        b = c - g
        out.update(center=c, R=float(R), centroid=g, bite_vec=b, bite_norm=float(np.linalg.norm(b)))
    else:
        out.update(center=allp[:, :2].mean(0), R=None, centroid=allp[:, :2].mean(0), bite_vec=np.zeros(2), bite_norm=0.0)
    return out


def rgb_to_hsv(a):
    """a: (...,3) float in [0,1] -> hsv (hue in [0,1))."""
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1); mn = a.min(-1); d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-6
    rm = m & (mx == r); gm = m & (mx == g) & ~rm; bm = m & ~rm & ~gm
    h[rm] = ((g - b)[rm] / d[rm]) % 6
    h[gm] = (b - r)[gm] / d[gm] + 2
    h[bm] = (r - g)[bm] / d[bm] + 4
    h = h / 6.0
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0)
    return np.stack([h, s, mx], -1)


def learn_object_hue(rgb, depth_m, K, pose, exclude_xy=None):
    """Median hue of saturated pixels raised 8-70 mm above the table (objects)."""
    vs, us = np.nonzero(depth_m > 0)
    pw = backproject(us.astype(float), vs.astype(float), depth_m[vs, us], K, pose)
    hsv = rgb_to_hsv(rgb[vs, us].astype(np.float64) / 255.0)
    sel = (pw[:, 2] > 0.008) & (pw[:, 2] < 0.07) & (hsv[:, 1] > SAT_MIN) & (np.abs(pw[:, 1]) < 0.5) & (pw[:, 0] > -0.5)
    if sel.sum() < 30:
        return None, int(sel.sum())
    h = hsv[sel, 0]
    # circular median via histogram mode
    hist, edges = np.histogram(h, bins=60, range=(0, 1))
    k = int(np.argmax(hist)); c = (edges[k] + edges[k + 1]) / 2
    dh = np.abs(h - c); dh = np.minimum(dh, 1 - dh)
    hh = h[dh < 0.08]
    # unwrap around c
    hh = np.where(hh - c > 0.5, hh - 1, np.where(c - hh > 0.5, hh + 1, hh))
    return float(np.median(hh) % 1.0), int(sel.sum())


def closing_width(xy, center, yaw, hw=0.009):
    """Simulated parallel-jaw closing: finger axis at angle yaw+90deg. Returns (width, mid_offset)."""
    ax = np.array([-np.sin(yaw), np.cos(yaw)]); pp = np.array([np.cos(yaw), np.sin(yaw)])
    d = xy - center
    a = d @ ax; p = d @ pp
    band = np.abs(p) < hw
    if band.sum() < 3:
        return None, None
    lo, hi = a[band].min(), a[band].max()
    return float(hi - lo), float((hi + lo) / 2)
