"""Cube detection in RGB-D (world points), table estimation, video tracking."""
import math

import numpy as np

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None
try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

# Tunables (never tested live; derived from typical ManiSkill tabletop scale)
MIN_H = 0.008          # m above table counted as object
MAX_H = 0.30
SAT_MIN = 0.28         # HSV saturation for coloured cubes
VAL_MIN = 0.10
MIN_AREA = 10          # pixels
MAX_EXTENT = 0.14      # m, xy extent of one detection
WORK_RADIUS = 0.85     # m around the reference xy


def hsv(rgb):
    a = rgb.astype(np.float64) / 255.0
    mx = a.max(-1)
    mn = a.min(-1)
    s = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return s, mx


def table_height(P, valid, ref_xy=None, radius=1.0):
    z = P[..., 2][valid]
    if ref_xy is not None:
        dxy = np.linalg.norm(P[..., :2][valid] - np.asarray(ref_xy)[None, :2], axis=1)
        z = z[dxy < radius]
    z = z[np.isfinite(z)]
    z = z[(z > -2) & (z < 3)]
    if z.size < 50:
        return None
    bins = np.arange(z.min(), z.max() + 0.004, 0.003)
    if bins.size < 2:
        return float(np.median(z))
    hist, edges = np.histogram(z, bins=bins)
    # prefer the highest plane holding at least 40% of the largest peak's count
    peak = hist.max()
    cand = np.where(hist >= 0.4 * peak)[0]
    # a plane: local maximum
    best = int(np.argmax(hist))
    if cand.size:
        # choose highest candidate that is a broad plane (count >= 4% of points)
        good = [i for i in cand if hist[i] >= 0.04 * z.size]
        if good:
            best = max(good)
    zc = 0.5 * (edges[best] + edges[best + 1])
    sel = z[np.abs(z - zc) < 0.006]
    return float(np.median(sel)) if sel.size else float(zc)


def plane_candidates(P, valid, ref_xy=None, radius=1.0, max_n=4):
    """Heights of broad horizontal planes (histogram peaks), most populated first."""
    z = P[..., 2][valid]
    if ref_xy is not None:
        dxy = np.linalg.norm(P[..., :2][valid] - np.asarray(ref_xy)[None, :2], axis=1)
        z = z[dxy < radius]
    z = z[np.isfinite(z) & (z > -2) & (z < 3)]
    if z.size < 50:
        return []
    bins = np.arange(z.min(), z.max() + 0.004, 0.003)
    if bins.size < 3:
        return [float(np.median(z))]
    hist, edges = np.histogram(z, bins=bins)
    order = np.argsort(-hist)
    out = []
    for i in order:
        if hist[i] < 0.02 * z.size or len(out) >= max_n:
            break
        zc = 0.5 * (edges[i] + edges[i + 1])
        if all(abs(zc - o) > 0.03 for o in out):
            sel = z[np.abs(z - zc) < 0.006]
            out.append(float(np.median(sel)) if sel.size else float(zc))
    return out


def detect(P, valid, rgb, table_z, ref_xy=None, use_sat=True, exclude_xyz=None, exclude_r=0.0):
    """Detect coloured blobs standing above the table. Returns list of dicts."""
    if ndi is None or table_z is None:
        return []
    h = P[..., 2] - table_z
    mask = valid & (h > MIN_H) & (h < MAX_H)
    if ref_xy is not None:
        d = np.linalg.norm(P[..., :2] - np.asarray(ref_xy)[None, None, :2], axis=-1)
        mask &= d < WORK_RADIUS
    s, v = hsv(rgb)
    if use_sat:
        mask &= (s > SAT_MIN) & (v > VAL_MIN)
    if exclude_xyz is not None and exclude_r > 0:
        d = np.linalg.norm(P - np.asarray(exclude_xyz)[None, None, :], axis=-1)
        mask &= d > exclude_r
    try:
        mask = ndi.binary_opening(mask, structure=np.ones((2, 2)))
    except Exception:
        pass
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    dets = []
    if n == 0:
        return dets
    idx = np.arange(1, n + 1)
    areas = ndi.sum(np.ones_like(lab), lab, idx)
    for i, area in zip(idx, areas):
        if area < MIN_AREA:
            continue
        m = lab == i
        pts = P[m]
        z = pts[:, 2]
        ztop = float(np.percentile(z, 95))
        top = pts[z > ztop - 0.007]
        if top.shape[0] < 3:
            top = pts
        cxy = np.median(top[:, :2], axis=0)
        ext = np.percentile(pts[:, :2], 95, axis=0) - np.percentile(pts[:, :2], 5, axis=0)
        if max(ext) > MAX_EXTENT:
            # try splitting big blobs into cube-sized parts
            for sub in _split_blob(m, P, rgb):
                dets.append(_make_det(sub, P, rgb, table_z))
            continue
        col = rgb[m].reshape(-1, 3).mean(0)
        vs, us = np.nonzero(m)
        dets.append({
            "xy": cxy, "ztop": ztop, "h": ztop - table_z, "area": int(area),
            "color": col, "chroma": chroma(col), "ext": ext,
            "uv": (float(us.mean()), float(vs.mean())), "mask": m,
            "top_pts": top[:, :2],
            "pos": np.array([cxy[0], cxy[1], ztop]),
        })
    dets = [d for d in dets if d is not None]
    return merge_fragments(dets, P, rgb, table_z)


MERGE_EXTENT = 0.056   # union of fragments must fit inside one cube footprint (~4-5 cm)


def merge_fragments(dets, P, rgb, table_z):
    """Merge pieces of one object split by an occluder (e.g. gripper fingers)."""
    changed = True
    while changed and len(dets) > 1:
        changed = False
        best = None
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                a, b = dets[i], dets[j]
                if np.linalg.norm(a["xy"] - b["xy"]) > MERGE_EXTENT:
                    continue
                if abs(a["ztop"] - b["ztop"]) > 0.03:
                    continue
                if color_dist(a["chroma"], b["chroma"]) > 0.15:
                    continue
                m = a["mask"] | b["mask"]
                pts = P[m][:, :2]
                ext = np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0)
                if max(ext) > MERGE_EXTENT:
                    continue
                d = np.linalg.norm(a["xy"] - b["xy"])
                if best is None or d < best[0]:
                    best = (d, i, j, m)
        if best is not None:
            _, i, j, m = best
            nd = _make_det(m, P, rgb, table_z)
            if nd is not None:
                dets = [d for k, d in enumerate(dets) if k not in (i, j)] + [nd]
                changed = True
    return dets


def _make_det(m, P, rgb, table_z):
    pts = P[m]
    if pts.shape[0] < MIN_AREA:
        return None
    z = pts[:, 2]
    ztop = float(np.percentile(z, 95))
    top = pts[z > ztop - 0.007]
    if top.shape[0] < 3:
        top = pts
    cxy = np.median(top[:, :2], axis=0)
    ext = np.percentile(pts[:, :2], 95, axis=0) - np.percentile(pts[:, :2], 5, axis=0)
    if max(ext) > MAX_EXTENT * 1.3:
        return None
    col = rgb[m].reshape(-1, 3).mean(0)
    vs, us = np.nonzero(m)
    return {"xy": cxy, "ztop": ztop, "h": ztop - table_z, "area": int(m.sum()), "color": col,
            "chroma": chroma(col), "ext": ext, "uv": (float(us.mean()), float(vs.mean())), "mask": m,
            "top_pts": top[:, :2], "pos": np.array([cxy[0], cxy[1], ztop])}


def _split_blob(m, P, rgb, max_k=6):
    """Split a large blob by colour/position k-means in world xy (simple)."""
    vs, us = np.nonzero(m)
    pts = P[m][:, :2]
    if pts.shape[0] < 2 * MIN_AREA:
        return []
    ext = np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0)
    k = int(min(max_k, max(2, round(float(ext[0] * ext[1]) / (0.045 * 0.045)))))
    rng = np.random.default_rng(0)
    cent = pts[rng.choice(pts.shape[0], k, replace=False)]
    for _ in range(15):
        d = np.linalg.norm(pts[:, None, :] - cent[None], axis=-1)
        a = d.argmin(1)
        for j in range(k):
            if (a == j).any():
                cent[j] = pts[a == j].mean(0)
    subs = []
    for j in range(k):
        mm = np.zeros_like(m)
        sel = a == j
        mm[vs[sel], us[sel]] = True
        subs.append(mm)
    return subs


def chroma(col):
    c = np.asarray(col, float)
    s = c.sum()
    return c / s if s > 1 else np.array([1 / 3] * 3)


def color_dist(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).sum())


def local_table_z(P, valid, xy, all_masks=None, r_in=0.04, r_out=0.10, global_z=None):
    d = np.linalg.norm(P[..., :2] - np.asarray(xy)[None, None, :2], axis=-1)
    m = valid & (d > r_in) & (d < r_out)
    if all_masks is not None:
        m &= ~all_masks
    z = P[..., 2][m]
    if global_z is not None:
        z = z[np.abs(z - global_z) < 0.05]
    if z.size < 20:
        return global_z
    # lower mode: the support plane
    zs = np.sort(z)
    lo = zs[: max(10, int(0.6 * zs.size))]
    return float(np.median(lo))


def cube_yaw(top_xy):
    """Yaw (rad, modulo 90 deg) minimising the bounding-box area of top-face points."""
    p = np.asarray(top_xy, float)
    if p.shape[0] < 8:
        return None, None
    p = p - p.mean(0)
    best = (1e9, 0.0)
    for deg in np.arange(0, 90, 1.5):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        q = p @ np.array([[c, -s], [s, c]])
        e = np.percentile(q, 97, axis=0) - np.percentile(q, 3, axis=0)
        area = e[0] * e[1]
        if area < best[0]:
            best = (area, a)
    # quality: ratio of best to worst
    return best[1], math.sqrt(best[0])


# ----------------------------------------------------------------------------- tracking
class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 0

    def update(self, t, dets, ee=None, ee_prev=None, grip_closed=None):
        preds = []
        for tr in self.tracks:
            p = tr["pos"].copy()
            if ee is not None and ee_prev is not None and tr.get("held"):
                p = p + (np.asarray(ee) - np.asarray(ee_prev))
            preds.append(p)
        nT, nD = len(self.tracks), len(dets)
        assign = {}
        if nT and nD:
            C = np.zeros((nT, nD))
            for i, tr in enumerate(self.tracks):
                for j, d in enumerate(dets):
                    C[i, j] = np.linalg.norm(preds[i] - d["pos"]) + 0.6 * color_dist(tr["chroma"], d["chroma"]) \
                        + (0.02 if tr["n"] < 5 else 0.0)
            gate = np.array([0.20 if tr.get("held") else 0.09 for tr in self.tracks])
            if linear_sum_assignment is not None:
                ri, ci = linear_sum_assignment(C)
            else:
                ri, ci = [], []
                used = set()
                for i in np.argsort(C.min(1)):
                    j = int(np.argmin(C[i]))
                    if j not in used:
                        ri.append(i)
                        ci.append(j)
                        used.add(j)
            for i, j in zip(ri, ci):
                if C[i, j] < gate[i]:
                    assign[int(i)] = int(j)
        used = set(assign.values())
        for i, tr in enumerate(self.tracks):
            if i in assign:
                d = dets[assign[i]]
                tr["pos"] = d["pos"].copy()
                tr["chroma"] = 0.8 * tr["chroma"] + 0.2 * d["chroma"]
                tr["last"] = t
                tr["n"] += 1
                tr["hist"].append((t, d["pos"].copy(), d["h"]))
                tr["det"] = d
            else:
                tr["pos"] = preds[i]
                tr["det"] = None
        for j, d in enumerate(dets):
            if j not in used:
                self.tracks.append({"id": self.next_id, "pos": d["pos"].copy(), "chroma": d["chroma"].copy(),
                                    "color": d["color"], "first": t, "last": t, "n": 1,
                                    "hist": [(t, d["pos"].copy(), d["h"])], "held": False, "det": d})
                self.next_id += 1
        # held flags for next prediction
        for tr in self.tracks:
            held = False
            if ee is not None and grip_closed:
                dxy = np.linalg.norm(tr["pos"][:2] - np.asarray(ee)[:2])
                dz = np.asarray(ee)[2] - tr["pos"][2]
                held = dxy < 0.045 and -0.05 < dz < 0.08
            tr["held"] = held

    def lifted_tracks(self, rise=0.03):
        out = []
        for tr in self.tracks:
            if tr["n"] < 3:
                continue
            hs = [h for _, _, h in tr["hist"]]
            base = float(np.median(hs[: max(1, min(5, len(hs)))]))
            for (t, p, h) in tr["hist"]:
                if h > base + rise:
                    out.append((t, tr, base))
                    break
        out.sort(key=lambda x: x[0])
        return out

    def displacement(self, tr):
        ps = [p for _, p, _ in tr["hist"]]
        if len(ps) < 2:
            return 0.0
        return float(np.linalg.norm(ps[-1][:2] - ps[0][:2]))
