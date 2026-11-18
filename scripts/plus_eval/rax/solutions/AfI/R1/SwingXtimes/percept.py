"""Classical RGB-D scene perception for the front camera: table, cubes, flat targets, button."""
import math

import numpy as np

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None

COLOR_HUES = {
    "red": 0, "orange": 30, "yellow": 55, "lime": 90, "green": 120, "cyan": 180, "teal": 175,
    "blue": 225, "navy": 230, "purple": 280, "violet": 275, "magenta": 305, "pink": 330,
}
ACHROMATIC = {"white", "black", "gray", "grey", "silver"}


def rgb_to_hsv(rgb):
    a = rgb.astype(np.float64) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1)
    mn = a.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-9
    rm = m & (mx == r)
    gm = m & (mx == g) & ~rm
    bm = m & ~rm & ~gm
    h[rm] = ((g - b)[rm] / d[rm]) % 6
    h[gm] = ((b - r)[gm] / d[gm]) + 2
    h[bm] = ((r - g)[bm] / d[bm]) + 4
    h = h * 60.0
    s = np.where(mx > 1e-9, d / np.maximum(mx, 1e-9), 0)
    return h, s, mx


def label(mask):
    if ndi is not None:
        lab, n = ndi.label(mask, structure=np.ones((3, 3)))
        return lab, n
    # fallback: iterative flood fill
    H, W = mask.shape
    lab = np.zeros((H, W), dtype=np.int32)
    n = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and lab[i, j] == 0:
                n += 1
                stack = [(i, j)]
                lab[i, j] = n
                while stack:
                    y, x = stack.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            yy, xx = y + dy, x + dx
                            if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and lab[yy, xx] == 0:
                                lab[yy, xx] = n
                                stack.append((yy, xx))
    return lab, n


def dilate(mask, r):
    if r <= 0:
        return mask
    if ndi is not None:
        return ndi.binary_dilation(mask, structure=np.ones((2 * r + 1, 2 * r + 1)))
    out = mask.copy()
    H, W = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            sh = np.zeros_like(mask)
            ys = slice(max(0, dy), H + min(0, dy))
            yd = slice(max(0, -dy), H + min(0, -dy))
            xs = slice(max(0, dx), W + min(0, dx))
            xd = slice(max(0, -dx), W + min(0, -dx))
            sh[yd, xd] = mask[ys, xs]
            out |= sh
    return out


def hue_dist(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def circ_mean_hue(h, w=None):
    rad = np.deg2rad(h)
    if w is None:
        w = np.ones_like(rad)
    return float(np.rad2deg(math.atan2(np.sum(w * np.sin(rad)), np.sum(w * np.cos(rad)))) % 360)


def min_area_rect_yaw(xy):
    """Yaw (mod 90 deg) and side lengths of the min-area rectangle around xy points."""
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 4:
        return 0.0, (0.0, 0.0)
    c = xy - xy.mean(0)
    best = (float("inf"), 0.0, (0, 0))
    for deg in np.arange(0, 90, 1.0):
        a = math.radians(deg)
        R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        p = c @ R.T
        lo = np.percentile(p, 3, axis=0)
        hi = np.percentile(p, 97, axis=0)
        ext = hi - lo
        area = ext[0] * ext[1]
        if area < best[0]:
            best = (area, a, (float(ext[0]), float(ext[1])))
    return best[1], best[2]


class Blob(dict):
    __getattr__ = dict.get


def table_height(P, valid, tcp):
    z = P[..., 2][valid]
    xy = P[..., :2][valid]
    near = np.linalg.norm(xy - tcp[:2], axis=1) < 1.0
    zz = z[near & (z < tcp[2] + 0.05) & (z > tcp[2] - 0.8)]
    if zz.size < 100:
        zz = z
    lo, hi = np.percentile(zz, 1), np.percentile(zz, 99)
    bins = np.arange(lo, hi + 0.004, 0.004)
    if len(bins) < 3:
        return float(np.median(zz))
    hist, edges = np.histogram(zz, bins=bins)
    i = int(np.argmax(hist))
    mode = 0.5 * (edges[i] + edges[i + 1])
    sel = zz[np.abs(zz - mode) < 0.008]
    return float(np.median(sel)) if sel.size else float(mode)


def analyse(rgb, P, tcp, log=print):
    """Return a dict with table_z, cubes (saturated elevated blobs), elevated blobs, flat targets."""
    H, W, _ = rgb.shape
    valid = np.all(np.isfinite(P), axis=-1)
    tz = table_height(P, valid, tcp)
    h = np.where(valid, P[..., 2] - tz, np.nan)
    hue, sat, val = rgb_to_hsv(rgb)
    xy = P[..., :2]
    dxy = np.where(valid, np.linalg.norm(xy - tcp[:2], axis=-1), np.inf)
    work = valid & (dxy < 0.85)
    near_tcp = valid & (np.linalg.norm(np.nan_to_num(P) - tcp, axis=-1) < 0.07)
    hh = np.nan_to_num(h, nan=-1)
    elevated = work & (hh > 0.010) & (hh < 0.16) & ~near_tcp
    flat = work & (hh > -0.012) & (hh < 0.009)
    out = {"table_z": tz, "cubes": [], "elevated": [], "targets": []}

    def features(m, kind):
        cnt = int(m.sum())
        pts = P[m]
        top = float(np.percentile(pts[:, 2], 92))
        topm = m & (P[..., 2] > top - 0.010)
        tp = P[topm] if topm.sum() >= 4 else pts
        yaw, sides = min_area_rect_yaw(tp[:, :2])
        ext = np.percentile(pts[:, :2], 97, axis=0) - np.percentile(pts[:, :2], 3, axis=0)
        ring = dilate(m, 3) & ~dilate(m, 1) & valid
        ring_table = float(np.mean(np.abs(hh[ring]) < 0.008)) if ring.sum() else 0.0
        ring_elev = float(np.mean(hh[ring] > 0.008)) if ring.sum() else 0.0
        vv, uu = np.nonzero(m)
        return Blob(kind=kind, n=cnt, center=np.array([float(np.median(tp[:, 0])), float(np.median(tp[:, 1])), top]),
                    top_h=top - tz, hue=circ_mean_hue(hue[m], sat[m]), sat=float(np.mean(sat[m])),
                    val=float(np.mean(val[m])), rgb=rgb[m].mean(0), yaw=yaw, sides=sides,
                    ext=float(max(ext)), ring_table=ring_table, ring_elev=ring_elev,
                    uv=(float(uu.mean()), float(vv.mean())), mask=m, max_h=float(np.nanmax(hh[m])))

    # ---------------- elevated, saturated blobs (cube candidates)
    satmask = elevated & (sat > 0.30) & (val > 0.12)
    lab, n = label(satmask)
    for i in range(1, n + 1):
        m = lab == i
        if int(m.sum()) >= 6:
            out["cubes"].append(features(m, "sat"))

    # ---------------- all elevated blobs (button, obstacles, achromatic cubes)
    lab2, n2 = label(elevated)
    for i in range(1, n2 + 1):
        m = lab2 == i
        if int(m.sum()) >= 8:
            out["elevated"].append(features(m, "elev"))

    # ---------------- flat markings on the table (targets)
    tab = flat & ~dilate(elevated, 4)
    if tab.sum() > 200:
        trgb = rgb[tab].astype(np.float64)
        ref = np.median(trgb, axis=0)
        s = trgb.sum(1, keepdims=True) + 1e-6
        chroma = trgb / s
        refc = ref / (ref.sum() + 1e-6)
        cd = np.linalg.norm(chroma - refc, axis=1)
        bright = trgb.mean(1) / (ref.mean() + 1e-6)
        cd_thr = max(0.05, float(np.percentile(cd, 60)) * 3.0)
        mark_vals = (cd > cd_thr) | (bright > 1.30) | (bright < 0.40)
        mark = np.zeros((H, W), dtype=bool)
        mark[tab] = mark_vals
        log("table ref rgb", np.round(ref, 1).tolist(), "chroma thr %.3f" % cd_thr,
            "marked %d / %d" % (int(mark.sum()), int(tab.sum())))
        lab3, n3 = label(mark)
        comps = []
        for i in range(1, n3 + 1):
            m = lab3 == i
            cnt = int(m.sum())
            if cnt < 6:
                continue
            pts = P[m]
            c = np.median(pts[:, :2], axis=0)
            comps.append({"n": cnt, "c": c, "pts": pts[:, :2], "mask": m})
        # merge components whose centroids lie close (rings of one bullseye)
        clusters = []
        for cp in sorted(comps, key=lambda x: -x["n"]):
            placed = False
            for cl in clusters:
                if np.linalg.norm(cl["c"] - cp["c"]) < 0.035:
                    cl["parts"].append(cp)
                    allp = np.concatenate([p["pts"] for p in cl["parts"]])
                    cl["n"] += cp["n"]
                    cl["mask"] = cl["mask"] | cp["mask"]
                    placed = True
                    break
            if not placed:
                clusters.append({"c": cp["c"].copy(), "n": cp["n"], "parts": [cp], "mask": cp["mask"].copy()})
        for cl in clusters:
            allp = np.concatenate([p["pts"] for p in cl["parts"]])
            lo, hi = np.percentile(allp, 2, axis=0), np.percentile(allp, 98, axis=0)
            ext = hi - lo
            centre = (lo + hi) / 2 if len(cl["parts"]) > 1 else np.median(allp, axis=0)
            cov = np.cov((allp - allp.mean(0)).T) if len(allp) > 3 else np.eye(2)
            ev = np.sort(np.linalg.eigvalsh(cov))
            elong = float(math.sqrt(ev[1] / max(ev[0], 1e-12)))
            m = cl["mask"]
            vv, uu = np.nonzero(m)
            out["targets"].append(Blob(kind="flat", n=cl["n"], center=np.array([float(centre[0]), float(centre[1]), tz]),
                                       ext=float(max(ext)), ext_min=float(min(ext)), elong=elong,
                                       hue=circ_mean_hue(hue[m], sat[m]), sat=float(np.mean(sat[m])),
                                       val=float(np.mean(val[m])), uv=(float(uu.mean()), float(vv.mean())),
                                       parts=len(cl["parts"]), mask=m))
    return out


def cube_score(b, color):
    """Lower is better. color is a name from the instruction or None."""
    if not (0.012 <= b.top_h <= 0.10) or b.ext > 0.12:
        return None
    s = 0.0
    if color in COLOR_HUES:
        hd = hue_dist(b.hue, COLOR_HUES[color])
        s += hd / 30.0
    # bigger blobs more reliable; cubes are surrounded by table, not by a base
    s += 0.6 * (1.0 - b.ring_table)
    s += 0.3 * abs(b.top_h - 0.04) / 0.04
    s -= 0.1 * min(b.n, 200) / 200.0
    return s


def achromatic_score(b, color):
    if not (0.012 <= b.top_h <= 0.10) or b.ext > 0.09 or b.ring_table < 0.6:
        return None
    s = 0.6 * (1.0 - b.ring_table) + 0.3 * abs(b.top_h - 0.04) / 0.04 + 0.5
    if color in ("white", "silver"):
        s += (1.0 - b.val)
    elif color == "black":
        s += b.val
    return s


def pick_cube(scene, color, exclude=None):
    cands = []
    pool = scene["cubes"]
    if color in ACHROMATIC:
        pool = []
    for b in pool:
        if exclude is not None and np.linalg.norm(b.center[:2] - exclude[:2]) < 0.03:
            continue
        s = cube_score(b, color)
        if s is not None:
            cands.append((s, b))
    if not cands:
        for b in scene["elevated"]:
            if exclude is not None and np.linalg.norm(b.center[:2] - exclude[:2]) < 0.03:
                continue
            s = achromatic_score(b, color)
            if s is not None:
                cands.append((s, b))
    cands.sort(key=lambda x: x[0])
    return cands


def pick_targets(scene, avoid_xy=()):
    """Choose the two flat markings most likely to be the twin targets."""
    c = []
    for t in scene["targets"]:
        if t.ext < 0.025 or t.ext > 0.35 or t.elong > 3.0 or t.n < 12:
            continue
        if any(np.linalg.norm(t.center[:2] - a[:2]) < 0.03 for a in avoid_xy):
            continue
        c.append(t)
    best, bs = None, float("inf")
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            a, b = c[i], c[j]
            d = np.linalg.norm(a.center[:2] - b.center[:2])
            if d < 0.07:
                continue
            size = abs(math.log((a.n + 1) / (b.n + 1)))
            ext = abs(math.log((a.ext + 1e-3) / (b.ext + 1e-3)))
            col = hue_dist(a.hue, b.hue) / 60.0 * min(a.sat, b.sat) + abs(a.val - b.val)
            s = size + ext + col - 0.5 * math.log(a.n * b.n + 1) / 5.0
            s -= 0.5 * min(a.sat, b.sat)                      # targets are usually coloured
            for t in (a, b):                                  # plausible target diameter
                if t.ext < 0.04 or t.ext > 0.25:
                    s += 0.5
            if s < bs:
                bs, best = s, (a, b)
    return best, c


def cube_like(scene, e):
    for b in scene["cubes"]:
        if np.linalg.norm(b.center[:2] - e.center[:2]) < 0.03 and b.ring_table > 0.7 and b.ext < 0.07 \
                and abs(b.n - e.n) < 0.3 * e.n:
            return True
    return False


def pick_button(scene, cube_xy, target_xys):
    cands = []
    for e in scene["elevated"]:
        if np.linalg.norm(e.center[:2] - cube_xy[:2]) < 0.04:
            continue
        if e.ext > 0.2 or e.top_h > 0.15 or e.top_h < 0.008:
            continue
        s = abs(e.ext - 0.06) / 0.06 - 0.3 * e.sat
        if cube_like(scene, e):
            s += 2.0
        cands.append((s, e))
    cands.sort(key=lambda x: x[0])
    return cands


def pick_targets_elevated(scene, avoid_xy=()):
    """Fallback when no flat twin markings exist: two similar low raised objects."""
    c = []
    for e in scene["elevated"]:
        if e.ext < 0.03 or e.ext > 0.2 or e.top_h > 0.08:
            continue
        if cube_like(scene, e):
            continue
        if any(np.linalg.norm(e.center[:2] - a[:2]) < 0.04 for a in avoid_xy):
            continue
        c.append(e)
    best, bs = None, float("inf")
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            a, b = c[i], c[j]
            if np.linalg.norm(a.center[:2] - b.center[:2]) < 0.08:
                continue
            s = abs(math.log(a.ext / b.ext)) + abs(a.top_h - b.top_h) / 0.01 + abs(math.log((a.n + 1) / (b.n + 1)))
            if s < bs:
                bs, best = s, (a, b)
    return best
