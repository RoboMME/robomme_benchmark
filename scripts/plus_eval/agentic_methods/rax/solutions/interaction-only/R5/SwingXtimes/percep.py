"""Color-based perception for SwingXtimes."""
import numpy as np
import cv2


HUES = {"green": (45, 85), "blue": (100, 130), "yellow": (22, 38), "purple": (130, 160),
        "orange": (8, 20), "cyan": (85, 100), "pink": (160, 175)}


def color_mask(rgb, name):
    r = rgb[..., 0].astype(int); g = rgb[..., 1].astype(int); b = rgb[..., 2].astype(int)
    if name == "red":
        return (r > 80) & (g < 50) & (b < 50) & (r - g > 60)
    if name in HUES:
        hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(int)
        h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        lo, hi = HUES[name]
        hm = ((h >= lo) & (h <= hi)) if lo <= hi else ((h >= lo) | (h <= hi))
        return hm & (sat > 150) & (val > 50)
    if name == "white":
        mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b)
        return (mx - mn <= 14) & (mx >= 70)
    raise ValueError(name)


def largest_component(mask, min_px=4):
    m = mask.astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < min_px:
        return None
    return lab == i


def components(mask, min_px=4):
    m = mask.astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            out.append((stats[i, cv2.CC_STAT_AREA], lab == i, cent[i]))
    out.sort(key=lambda t: -t[0])
    return out


def cube_estimate(obs, cam, color="red"):
    """Return (center_xy, top_z, yaw_mod90, npts) from top-face points, or None."""
    rgb = obs.rgb(cam)
    best = None
    for a, mc, cc in components(color_mask(rgb, color), min_px=3)[:6]:
        me = cv2.erode(mc.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        if me.sum() >= 6:
            mc = me
        Pc = obs.deproject_mask(cam, mc)
        Pc = Pc[(Pc[:, 2] > 0.012) & (Pc[:, 2] < 0.4)] if len(Pc) else Pc
        if len(Pc) >= 3 and (best is None or len(Pc) > len(best)):
            best = Pc
    if best is None:
        return None
    P = best
    ztop = np.percentile(P[:, 2], 90)
    top = P[P[:, 2] > ztop - 0.006]
    if len(top) < 3:
        top = P
    c = np.median(top[:, :2], axis=0)
    yaw = None
    if len(top) >= 8:
        pts = top[:, :2].astype(np.float32)
        rect = cv2.minAreaRect(pts)
        yaw = np.deg2rad(rect[2]) % (np.pi / 2)
        c = np.array(rect[0], dtype=float)
    return c, float(ztop), yaw, len(top)
