import numpy as np
from rlib import cloud

def hsv(rgb):
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = x.max(-1); mn = x.min(-1); d = mx - mn + 1e-6
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    s = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0)
    return h, s, mx

def label(mask):
    """simple connected components (4-neigh), returns list of index arrays"""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32); n = 0; comps = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]:
            continue
        n += 1; stack = [(y0, x0)]; lab[y0, x0] = n; pts = []
        while stack:
            y, x = stack.pop(); pts.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n; stack.append((yy, xx))
        comps.append(np.array(pts))
    return comps

def masks(rgb):
    h, s, v = hsv(rgb)
    cube = (s > 0.2) & (v > 0.2)
    white = (s < 0.12) & (v > 0.85)
    return cube, white

def geo_cube_mask(P, dep, rgb, table_z=0.0):
    z = P[..., 2] - table_z
    m = (dep > 0) & (z > 0.022) & (z < 0.065) & (P[..., 0] > -0.55) & (P[..., 0] < 0.35) & (np.abs(P[..., 1]) < 0.45)
    h, s, v = hsv(rgb)
    m &= (s > 0.12)   # exclude white/grey button and grey robot
    return m

def table_z(P, rgb):
    h, s, v = hsv(rgb)
    tab = (h > 20) & (h < 50) & (s > 0.2) & (s < 0.6)
    z = P[..., 2][tab]
    return float(np.median(z)) if z.size else 0.0

def detect(meta, imgs, cam):
    rgb = imgs[(cam, "rgb")]; dep = imgs[(cam, "depth")]
    P = cloud(meta, dep, cam)
    valid = dep > 0
    _, white = masks(rgb)
    tz = 0.0
    cube = geo_cube_mask(P, dep, rgb, tz)
    out = {"table_z": tz, "cubes": [], "white": []}
    for name, m in (("cubes", cube), ("white", white)):
        for c in label(m & valid):
            if len(c) < 6:
                continue
            pts = P[c[:, 0], c[:, 1]]
            out[name].append({"n": len(c), "uv": c.mean(0)[::-1].tolist(),
                              "xyz": np.median(pts, 0).tolist(),
                              "zmax": float(np.percentile(pts[:, 2], 90)),
                              "pts": pts})
    return out

def find_button(meta, imgs, cam="wrist"):
    """button = grey cap (low sat, mid value) raised above table, adjacent to white base"""
    rgb = imgs[(cam, "rgb")]; dep = imgs[(cam, "depth")]
    P = cloud(meta, dep, cam)
    h, s, v = hsv(rgb)
    grey = (s < 0.08) & (v > 0.35) & (v < 0.75) & (dep > 0) & (P[..., 2] > 0.008) & (P[..., 2] < 0.08)
    cube, white = masks(rgb)
    whiteb = white & (dep > 0) & (P[..., 2] > 0.008) & (P[..., 2] < 0.04)
    best = None
    for c in label(grey):
        if len(c) < 4:
            continue
        pts = P[c[:, 0], c[:, 1]]
        ctr = np.median(pts, 0)
        # white base nearby
        W = P[whiteb]
        nw = int((np.linalg.norm(W[:, :2] - ctr[:2], axis=1) < 0.05).sum()) if len(W) else 0
        sc = nw + len(c)
        if nw > 5 and (best is None or sc > best[0]):
            best = (sc, ctr, float(np.percentile(pts[:, 2], 90)), len(c), nw)
    return best

def highlight_votes(meta, imgs, cam, cubes, button_xy, r=0.045):
    """for each cube xy, count table-level white pixels within r (excluding button area)"""
    rgb = imgs[(cam, "rgb")]; dep = imgs[(cam, "depth")]
    P = cloud(meta, dep, cam)
    cube, white = masks(rgb)
    m = white & (dep > 0) & (P[..., 2] < 0.025) & (P[..., 2] > -0.01)
    W = P[m][:, :2]
    if button_xy is not None and len(W):
        W = W[np.linalg.norm(W - np.asarray(button_xy)[:2], axis=1) > 0.05]
    votes = []
    for c in cubes:
        xy = np.asarray(c["xyz"][:2])
        votes.append(int((np.linalg.norm(W - xy, axis=1) < r).sum()) if len(W) else 0)
    return votes

def cube_pose(pts):
    """pts: Nx3 world points of a cube blob. returns center xy, top z, yaw (rad, mod pi/2)"""
    zt = np.percentile(pts[:, 2], 90)
    top = pts[pts[:, 2] > zt - 0.006]
    if len(top) < 5:
        top = pts
    xy = top[:, :2]
    best = None
    for a in np.deg2rad(np.arange(0, 90, 2.0)):
        c, s = np.cos(a), np.sin(a)
        q = xy @ np.array([[c, -s], [s, c]])
        lo, hi = np.percentile(q, 3, 0), np.percentile(q, 97, 0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            ctr = ((lo + hi) / 2) @ np.array([[c, s], [-s, c]])
            best = (area, a, ctr, hi - lo)
    return best[2], float(zt), float(best[1]), best[3]

def find_cube_near(meta, imgs, cam, xy, rmax=0.05):
    d = detect(meta, imgs, cam)
    best = None
    for c in d["cubes"]:
        if c["n"] < 15 or c["zmax"] < 0.02 or c["zmax"] > 0.07:
            continue
        ctr, zt, yaw, ext = cube_pose(c["pts"])
        dd = np.linalg.norm(np.asarray(ctr) - np.asarray(xy[:2]))
        if dd < rmax and (best is None or dd < best[0]):
            best = (dd, {"xy": ctr, "ztop": zt, "yaw": yaw, "ext": ext, "n": c["n"], "dist": dd})
    return None if best is None else best[1]
