import numpy as np
from obs import backproject

def _label(mask):
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
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n
                    stack.append((yy, xx))
    return lab, n

def world_cloud(meta, imgs, cam="front"):
    d = imgs[(cam, "depth")]
    P = backproject(meta, cam, d)
    P[d <= 0] = np.nan
    return P

def saturation(rgb):
    rgb = rgb.astype(np.float32)
    mx = rgb.max(-1); mn = rgb.min(-1)
    return (mx - mn) / np.maximum(mx, 1)

def table_color_mask(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    # wood: r>g>b, warm
    return (r > b + 25) & (r >= g - 5) & (g > b + 10)

def find_cubes(meta, imgs, cam="front", zmin=0.033, zmax=0.08, min_px=6, xy_box=(-0.45, 0.35, -0.4, 0.4)):
    """Return list of dicts: center xyz (top-face centroid), top z, pixel count, mean rgb."""
    P = world_cloud(meta, imgs, cam)
    rgb = imgs[(cam, "rgb")]
    sat = saturation(rgb)
    z = P[..., 2]
    x0, x1, y0, y1 = xy_box
    with np.errstate(invalid="ignore"):
        m = (z > zmin) & (z < zmax) & (sat > 0.15)
        m &= (P[..., 0] > x0) & (P[..., 0] < x1) & (P[..., 1] > y0) & (P[..., 1] < y1)
    lab, n = _label(m)
    out = []
    for k in range(1, n + 1):
        sel = lab == k
        if sel.sum() < min_px:
            continue
        pts = P[sel]
        ztop = np.nanpercentile(pts[:, 2], 90)
        top = pts[pts[:, 2] > ztop - 0.008]
        c = np.nanmean(top, 0)
        out.append(dict(xyz=c, ztop=float(ztop), npx=int(sel.sum()), rgb=rgb[sel].mean(0), pts=pts, top=top))
    return out

def white_mask_world(meta, imgs, cam="front"):
    P = world_cloud(meta, imgs, cam)
    rgb = imgs[(cam, "rgb")].astype(int)
    with np.errstate(invalid="ignore"):
        m = (rgb.min(-1) > 215) & (saturation(rgb) < 0.12) & (P[..., 2] < 0.022) & (P[..., 2] > 0.004)
    return P, m

def highlight_scores(meta, imgs, cubes, cam="front", r_in=0.02, r_out=0.05):
    """Fraction of table pixels in annulus around each cube xy that are white."""
    P, wm = white_mask_world(meta, imgs, cam)
    with np.errstate(invalid="ignore"):
        tab = (P[..., 2] < 0.022) & (P[..., 2] > -0.01)
    xy = P[..., :2]
    scores = []
    for c in cubes:
        dd = np.linalg.norm(xy - c["xyz"][:2], axis=-1)
        with np.errstate(invalid="ignore"):
            ann = tab & (dd > r_in) & (dd < r_out)
        n = ann.sum()
        scores.append(float((wm & ann).sum() / n) if n > 0 else 0.0)
    return scores

def rect_yaw(pts_xy):
    """Yaw (rad, mod pi/2, in [-pi/4, pi/4)) of min-area rectangle of points, plus side length estimate."""
    pts = pts_xy - pts_xy.mean(0)
    best = None
    for a in np.deg2rad(np.arange(0, 90, 1.5)):
        c, s = np.cos(a), np.sin(a)
        q = pts @ np.array([[c, -s], [s, c]])
        ext = np.percentile(q, 98, 0) - np.percentile(q, 2, 0)
        area = ext[0] * ext[1]
        if best is None or area < best[0]:
            best = (area, a, ext)
    a = best[1]
    a = (a + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return float(a), best[2]

def find_button(meta, imgs, cam="front", xy_box=(-0.45, 0.35, -0.4, 0.4)):
    P = world_cloud(meta, imgs, cam)
    rgb = imgs[(cam, "rgb")]
    sat = saturation(rgb)
    z = P[..., 2]
    x0, x1, y0, y1 = xy_box
    with np.errstate(invalid="ignore"):
        m = (z > 0.018) & (z < 0.06) & (sat < 0.12) & (rgb.max(-1) > 60)
        m &= (P[..., 0] > x0) & (P[..., 0] < x1) & (P[..., 1] > y0) & (P[..., 1] < y1)
    lab, n = _label(m)
    best = None
    for k in range(1, n + 1):
        sel = lab == k
        if sel.sum() < 8:
            continue
        if best is None or sel.sum() > best.sum():
            best = sel
    if best is None:
        return None
    pts = P[best]
    ztop = float(np.nanpercentile(pts[:, 2], 95))
    cap = pts[pts[:, 2] > ztop - 0.006]
    lo = np.nanpercentile(cap[:, :2], 3, 0); hi = np.nanpercentile(cap[:, :2], 97, 0)
    return dict(xy=(lo + hi) / 2, ztop=ztop, npx=int(best.sum()), pts=pts)
