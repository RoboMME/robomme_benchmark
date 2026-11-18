"""Perception: find colored cubes, the bullseye target and the button from RGB-D."""
import numpy as np

try:
    from scipy import ndimage
except Exception:  # pragma: no cover
    ndimage = None


def hsv(img):
    x = img.astype(float) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = x.max(-1)
    mn = x.min(-1)
    df = mx - mn + 1e-9
    h = np.zeros_like(mx)
    m = mx == r
    h[m] = ((g - b)[m] / df[m]) % 6
    m2 = (mx == g) & ~m
    h[m2] = ((b - r)[m2] / df[m2]) + 2
    m3 = (mx == b) & ~m & ~m2
    h[m3] = ((r - g)[m3] / df[m3]) + 4
    h = h * 60.0
    s = np.where(mx > 0, (mx - mn) / (mx + 1e-9), 0)
    return h, s, mx


COLOR_HUES = {"red": (345, 12), "yellow": (45, 70), "green": (90, 160), "blue": (205, 260),
              "purple": (265, 320)}


def color_mask(img, name, smin=0.55, vmin=0.25):
    h, s, v = hsv(img)
    lo, hi = COLOR_HUES[name]
    if lo > hi:
        hm = (h >= lo) | (h <= hi)
    else:
        hm = (h >= lo) & (h <= hi)
    return hm & (s >= smin) & (v >= vmin)


def label(mask):
    if ndimage is not None:
        lab, n = ndimage.label(mask)
        return lab, n
    # simple BFS fallback
    H, W = mask.shape
    lab = np.zeros((H, W), int)
    n = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and lab[i, j] == 0:
                n += 1
                st = [(i, j)]
                lab[i, j] = n
                while st:
                    a, b = st.pop()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        c, d = a + da, b + db
                        if 0 <= c < H and 0 <= d < W and mask[c, d] and lab[c, d] == 0:
                            lab[c, d] = n
                            st.append((c, d))
    return lab, n


def blobs(mask, pw, min_px=4, zmin=None, zmax=None):
    """Connected components of mask -> list of dicts (pixel centroid, world centroid, top z, count)."""
    m = mask & np.isfinite(pw[..., 0])
    if zmin is not None:
        m &= pw[..., 2] > zmin
    if zmax is not None:
        m &= pw[..., 2] < zmax
    lab, n = label(m)
    out = []
    for k in range(1, n + 1):
        sel = lab == k
        cnt = int(sel.sum())
        if cnt < min_px:
            continue
        vs, us = np.nonzero(sel)
        P = pw[sel]
        out.append({"u": float(us.mean()), "v": float(vs.mean()), "n": cnt,
                    "xyz": np.nanmedian(P, axis=0), "ztop": float(np.nanpercentile(P[:, 2], 90)),
                    "zmax": float(np.nanmax(P[:, 2]))})
    out.sort(key=lambda b: -b["n"])
    return out


def find_cubes(obs, cam="front", color="green", exclude_xy=None):
    pw = obs.world_points(cam)
    img = obs.rgb[cam]
    m = color_mask(img, color)
    bl = blobs(m, pw, min_px=3, zmin=0.004, zmax=0.12)
    return bl


def find_target(obs, cam="front"):
    """Bullseye: purple/magenta rings plus white, flat on the table."""
    pw = obs.world_points(cam)
    img = obs.rgb[cam]
    m = color_mask(img, "purple", smin=0.3, vmin=0.3)
    bl = blobs(m, pw, min_px=3, zmax=0.03)
    return bl


def find_button(obs, cam="front"):
    """Button: a light grey low base (low saturation, high value) above the table."""
    pw = obs.world_points(cam)
    img = obs.rgb[cam]
    h, s, v = hsv(img)
    m = (s < 0.15) & (v > 0.55)
    bl = blobs(m, pw, min_px=4, zmin=0.004, zmax=0.08)
    # drop anything near the robot base (x < -0.4) / out of table
    bl = [b for b in bl if -0.45 < b["xyz"][0] < 0.4 and abs(b["xyz"][1]) < 0.45]
    return bl
