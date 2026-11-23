import numpy as np
from scipy import ndimage

def green_mask(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (g > r + 25) & (g > b + 5)

def button_mask(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b)
    return (mx - mn < 18) & (mx > 60)

def blobs(mask, P, min_px=6, zmin=0.005):
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < min_px:
            continue
        pts = P[ys, xs]
        ok = (pts[:, 2] > zmin) & (pts[:, 2] < 0.3)
        if ok.sum() < 3:
            continue
        pts = pts[ok]
        top = pts[:, 2].max()
        sel = pts[pts[:, 2] > top - 0.006]
        out.append(dict(uv=(float(xs.mean()), float(ys.mean())), n=int(len(ys)),
                        xyz=pts.mean(0), top=float(top), topxy=sel[:, :2].mean(0)))
    return out


def square_yaw(xy):
    """Yaw (rad, in [-pi/4,pi/4)) of a square footprint from its top-face xy points."""
    xy = xy - xy.mean(0)
    best, ba = None, 0.0
    for a in np.deg2rad(np.arange(-45, 45, 1.5)):
        c, s = np.cos(a), np.sin(a)
        q = xy @ np.array([[c, -s], [s, c]])  # rotate by -a
        area = np.ptp(q[:, 0]) * np.ptp(q[:, 1])
        if best is None or area < best:
            best, ba = area, a
    return float(ba)


def cube_top_points(obs, cam, near_xy, radius=0.035):
    P = obs.cloud(cam)
    m = green_mask(obs.rgb[cam]) & (P[..., 2] > 0.005) & (P[..., 2] < 0.3)
    pts = P[m]
    if len(pts) == 0:
        return pts
    d = np.linalg.norm(pts[:, :2] - np.asarray(near_xy)[None], axis=1)
    pts = pts[d < radius]
    if len(pts) == 0:
        return pts
    top = np.percentile(pts[:, 2], 95)
    return pts[pts[:, 2] > top - 0.006]
