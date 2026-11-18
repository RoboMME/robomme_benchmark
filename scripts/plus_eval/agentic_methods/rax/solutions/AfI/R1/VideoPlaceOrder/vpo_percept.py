"""Perception and video reasoning for VideoPlaceOrder (pure functions, offline-testable)."""
import math
import re

import numpy as np

from vpo_util import wrap_quarter

# ---------------------------------------------------------------------------
# Tunables (untested live: AfI with a single simulator start; ranges are design guesses)
# ---------------------------------------------------------------------------
BAND_LO = 0.012     # m above table: lowest point counted as an object (targets assumed thinner)
BAND_HI = 0.11      # m above table: highest point counted as a cube-like object
CELL = 0.01         # m: xy occupancy grid for clustering
MAX_EXTENT = 0.11   # m: largest xy extent of a cube-like cluster
MIN_POINTS = 8      # depth points in a cluster
OPEN_MARGIN = 0.010 # m: finger-sum drop below "open" that counts as closed
EMPTY_CLOSE = 0.004 # m: finger sum below this while closed = closed on nothing
HOLD_TOL = 0.006    # m: finger-sum tolerance around the holding plateau


# ---------------------------------------------------------------------------
# Table plane
# ---------------------------------------------------------------------------
def estimate_table_z(P, valid, center_xy=None, radius=0.45, z_max=None):
    """Mode of the height histogram of depth points near center_xy."""
    pts = P[valid]
    if pts.shape[0] == 0:
        return None
    if center_xy is not None:
        dxy = np.linalg.norm(pts[:, :2] - np.asarray(center_xy)[None, :2], axis=1)
        sel = pts[dxy < radius]
        if sel.shape[0] > 200:
            pts = sel
    z = pts[:, 2]
    if z_max is not None:
        zz = z[z < z_max]
        if zz.size > 200:
            z = zz
    lo, hi = np.percentile(z, 1), np.percentile(z, 99)
    if hi - lo < 1e-3:
        return float(np.median(z))
    bins = np.arange(lo, hi + 0.004, 0.004)
    if bins.size < 3:
        return float(np.median(z))
    h, e = np.histogram(z, bins=bins)
    i = int(np.argmax(h))
    zm = 0.5 * (e[i] + e[i + 1])
    near = z[np.abs(z - zm) < 0.006]
    return float(np.median(near)) if near.size else float(zm)


# ---------------------------------------------------------------------------
# Clustering of above-table points
# ---------------------------------------------------------------------------
def _label(grid):
    try:
        from scipy import ndimage
        lab, n = ndimage.label(grid, structure=np.ones((3, 3)))
        return lab, n
    except Exception:
        lab = np.zeros(grid.shape, dtype=int)
        n = 0
        H, W = grid.shape
        for i in range(H):
            for j in range(W):
                if grid[i, j] and lab[i, j] == 0:
                    n += 1
                    stack = [(i, j)]
                    lab[i, j] = n
                    while stack:
                        a, b = stack.pop()
                        for da in (-1, 0, 1):
                            for db in (-1, 0, 1):
                                x, y = a + da, b + db
                                if 0 <= x < H and 0 <= y < W and grid[x, y] and lab[x, y] == 0:
                                    lab[x, y] = n
                                    stack.append((x, y))
        return lab, n


def find_clusters(P, valid, table_z, roi_center=None, roi_radius=0.6, exclude=None, rgb=None,
                  band_lo=BAND_LO, band_hi=BAND_HI):
    """Cube-like clusters among above-table points.

    exclude: list of (xyz, radius) spheres (e.g. around the gripper) to drop.
    Returns list of dicts sorted by point count (desc).
    """
    pts = P[valid]
    cols = rgb[valid] if rgb is not None else None
    if pts.shape[0] == 0 or table_z is None:
        return []
    h = pts[:, 2] - table_z
    m = (h > band_lo) & (h < band_hi)
    if roi_center is not None:
        m &= np.linalg.norm(pts[:, :2] - np.asarray(roi_center)[None, :2], axis=1) < roi_radius
    if exclude:
        # (xyz, radius, z_below): drop points within `radius` (xy) of xyz and higher than xyz.z - z_below
        for ex in exclude:
            c, r = np.asarray(ex[0], dtype=float), ex[1]
            zb = ex[2] if len(ex) > 2 else r
            near = np.linalg.norm(pts[:, :2] - c[None, :2], axis=1) < r
            m &= ~(near & (pts[:, 2] > c[2] - zb))
    pts = pts[m]
    if cols is not None:
        cols = cols[m]
    if pts.shape[0] < MIN_POINTS:
        return []
    mn = pts[:, :2].min(axis=0) - CELL
    idx = np.floor((pts[:, :2] - mn) / CELL).astype(int)
    shape = idx.max(axis=0) + 2
    if shape[0] * shape[1] > 4_000_000:
        return []
    grid = np.zeros(shape, dtype=bool)
    grid[idx[:, 0], idx[:, 1]] = True
    lab, n = _label(grid)
    pl = lab[idx[:, 0], idx[:, 1]]
    out = []
    for k in range(1, n + 1):
        sel = pl == k
        cp = pts[sel]
        if cp.shape[0] < MIN_POINTS:
            continue
        ext = cp[:, :2].max(axis=0) - cp[:, :2].min(axis=0)
        zmax = float(np.percentile(cp[:, 2], 97))
        top = cp[cp[:, 2] > zmax - 0.008]
        if top.shape[0] < 3:
            top = cp
        c = {
            "n": int(cp.shape[0]),
            "center": [float(top[:, 0].mean()), float(top[:, 1].mean()), zmax],
            "centroid": [float(v) for v in cp.mean(axis=0)],
            "top_z": zmax,
            "height": zmax - table_z,
            "extent": [float(ext[0]), float(ext[1])],
            "yaw": estimate_yaw(top[:, :2]) if top.shape[0] >= 6 else None,
            "top_n": int(top.shape[0]),
        }
        if cols is not None:
            c["rgb"] = [float(v) for v in np.median(cols[sel], axis=0)]
        out.append(c)
    out.sort(key=lambda c: -c["n"])
    return out


def estimate_yaw(xy):
    """Yaw (mod 90 deg, in [-45, 45)) of a square footprint by min-area box search."""
    xy = np.asarray(xy, dtype=float)
    xy = xy - xy.mean(axis=0)
    best, best_a = None, 0.0
    for deg in range(0, 90, 2):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, s], [-s, c]])
        q = xy @ R.T
        ext = q.max(axis=0) - q.min(axis=0)
        area = ext[0] * ext[1]
        if best is None or area < best:
            best, best_a = area, a
    return wrap_quarter(best_a)


def is_cube_like(c, size_hint=None):
    if max(c["extent"]) > MAX_EXTENT:
        return False
    if c["height"] < 0.012 or c["height"] > 0.10:
        return False
    return True


def choose_cube(clusters, anchors, size_hint=None, log=None):
    """Pick the task cube.

    anchors: ordered list of (name, xy, weight) expected cube locations.
    Returns (cluster or None, reason).
    """
    cands = [c for c in clusters if is_cube_like(c, size_hint)]
    if not cands:
        return None, "no cube-like cluster"
    for name, xy, rad in anchors:
        if xy is None:
            continue
        best = None
        for c in cands:
            d = math.hypot(c["center"][0] - xy[0], c["center"][1] - xy[1])
            if d < rad and (best is None or d < best[0]):
                best = (d, c)
        if best is not None:
            return best[1], "near %s (%.3f m)" % (name, best[0])

    def score(c):
        s = 0.0
        if size_hint:
            s += 3.0 * abs(c["height"] - size_hint)
            s += 1.0 * abs(max(c["extent"]) - size_hint * 1.2)
        for name, xy, rad in anchors:
            if xy is not None:
                s += 0.3 * math.hypot(c["center"][0] - xy[0], c["center"][1] - xy[1])
                break
        return s
    cands.sort(key=score)
    return cands[0], "best score"


# ---------------------------------------------------------------------------
# Video: grasp/release events from the per-frame robot state
# ---------------------------------------------------------------------------
def extract_events(times, states):
    """times: list of float; states: list of 8-vectors (or None).

    Returns dict with placements (ordered), grasps, segments, diagnostics.
    """
    rows = [(t, s) for t, s in zip(times, states) if s is not None and len(s) >= 8]
    info = {"n_frames": len(rows), "placements": [], "grasps": [], "segments": [], "usable": False}
    if len(rows) < 5:
        info["why"] = "too few states"
        return info
    t = np.array([r[0] for r in rows], dtype=float)
    S = np.array([r[1][:8] for r in rows], dtype=float)
    w = S[:, 6] + S[:, 7]
    pos = S[:, :3]
    info["w_range"] = [float(w.min()), float(w.max())]
    info["pos_range"] = float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0)))
    if w.max() - w.min() < 0.006 or info["pos_range"] < 0.02:
        info["why"] = "state does not vary (w %.4f, pos %.3f)" % (w.max() - w.min(), info["pos_range"])
        return info
    w_open = float(np.percentile(w, 95))
    # Holding = finger sum on the cube-width plateau (mode of partially-closed widths).
    part = w[(w > EMPTY_CLOSE) & (w < w_open - OPEN_MARGIN)]
    if part.size >= 3:
        hist, edges = np.histogram(part, bins=np.arange(EMPTY_CLOSE, w_open, 0.002))
        if hist.size:
            i0 = int(np.argmax(hist))
            w_mode = float(np.median(part[(part >= edges[i0] - 0.002) & (part <= edges[i0 + 1] + 0.002)]))
        else:
            w_mode = float(np.median(part))
    else:
        w_mode = None
    info["w_mode"] = w_mode
    if w_mode is None:
        info["why"] = "gripper never closed on an object"
        return info
    closed = (np.abs(w - w_mode) < HOLD_TOL) & (w > EMPTY_CLOSE)
    c = closed.copy()
    for i in range(1, len(c) - 1):
        if closed[i - 1] and closed[i + 1] and not closed[i]:
            c[i] = True
    segs = []
    i = 0
    n = len(c)
    while i < n:
        if c[i]:
            j = i
            while j + 1 < n and c[j + 1]:
                j += 1
            segs.append((i, j))
            i = j + 1
        else:
            i += 1
    for (a, b) in segs:
        wh = float(np.median(w[a:b + 1]))
        dur = float(t[b] - t[a])
        seg = {
            "start": a, "end": b, "t0": float(t[a]), "t1": float(t[b]),
            "w_hold": wh,
            "grasp_pos": [float(v) for v in pos[a]],
            "release_pos": [float(v) for v in pos[b]],
            "min_z": float(pos[a:b + 1, 2].min()),
            "ends_closed": bool(b == n - 1),
            "starts_closed": bool(a == 0),
            "moved": float(np.linalg.norm(pos[b, :2] - pos[a, :2])),
            "dur": dur,
        }
        seg["holding"] = wh > EMPTY_CLOSE
        seg["short"] = (b - a + 1) < 3 or dur < 0.3
        seg["counted"] = bool(seg["holding"] and not (seg["short"] and seg["moved"] < 0.03))
        info["segments"].append(seg)
    held = [s for s in info["segments"] if s["counted"]]
    for s in held:
        info["grasps"].append({"pos": s["grasp_pos"], "t": s["t0"], "w_hold": s["w_hold"],
                               "from_start": s["starts_closed"]})
        if not s["ends_closed"]:
            info["placements"].append({"pos": s["release_pos"], "t": s["t1"], "w_hold": s["w_hold"],
                                       "moved": s["moved"]})
    info["holding_at_end"] = bool(held and held[-1]["ends_closed"])
    info["w_open"] = w_open
    info["usable"] = len(info["placements"]) > 0
    if held:
        info["w_hold_median"] = float(np.median([s["w_hold"] for s in held]))
    return info


# ---------------------------------------------------------------------------
# Instruction parsing
# ---------------------------------------------------------------------------
ORD_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
}
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
             "nine": 9, "ten": 10}
CONTEXT_WORDS = ("time", "target", "place", "placement", "visit", "position", "spot", "location", "stop", "order")


def parse_instruction(text):
    """Return dict(mode='index'|'sequence'|'unknown', index=int (0-based or negative), why=str)."""
    if not text:
        return {"mode": "unknown", "index": None, "why": "empty"}
    s = text.lower().replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    # Relative-to-end phrases first
    m = re.search(r"\b(second|third|fourth|2nd|3rd|4th)\s+(?:to|from)\s+(?:the\s+)?(?:last|final|end)\b", s)
    if m:
        k = {"second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}[m.group(1)]
        return {"mode": "index", "index": -k, "why": m.group(0)}
    if re.search(r"\b(penultimate|next to last|one before (?:the )?last)\b", s):
        return {"mode": "index", "index": -2, "why": "penultimate"}
    hits = []
    for mm in re.finditer(r"\b(" + "|".join(ORD_WORDS) + r")\b", s):
        hits.append((mm.start(), ORD_WORDS[mm.group(1)], mm.group(0)))
    for mm in re.finditer(r"\b(\d+)\s*(st|nd|rd|th)\b", s):
        hits.append((mm.start(), int(mm.group(1)), mm.group(0)))
    for mm in re.finditer(r"(?:number|no\.|#|position|step|order)\s*(\d+)\b", s):
        hits.append((mm.start(), int(mm.group(1)), mm.group(0)))
    for mm in re.finditer(r"\b(last|final|latest|most recent|most recently)\b", s):
        hits.append((mm.start(), -1, mm.group(0)))
    # Drop "first" used as an adverb of sequence ("first pick up the cube, then ...")
    filt = []
    for pos, k, word in hits:
        tail = s[pos:pos + 40]
        if word == "first" and re.match(r"first,?\s+(pick|grab|grasp|take|lift|move)\b", tail):
            continue
        filt.append((pos, k, word))
    hits = filt
    if len(hits) == 1:
        return {"mode": "index", "index": hits[0][1] - 1 if hits[0][1] > 0 else hits[0][1], "why": hits[0][2]}
    if len(hits) > 1:
        def ctx_score(h):
            pos, k, word = h
            win = s[max(0, pos - 25): pos + len(word) + 25]
            return sum(1 for w in CONTEXT_WORDS if w in win)
        best = max(hits, key=lambda h: (ctx_score(h), h[0]))
        return {"mode": "index", "index": best[1] - 1 if best[1] > 0 else best[1],
                "why": "multi:%s" % [h[2] for h in hits]}
    m = re.search(r"\b(?:the\s+)?(" + "|".join(NUM_WORDS) + r")\s+(?:time|placement)", s)
    if m:
        return {"mode": "index", "index": NUM_WORDS[m.group(1)] - 1, "why": m.group(0)}
    if re.search(r"\b(reverse|reversed|backwards|opposite order)\b", s):
        return {"mode": "reverse", "index": None, "why": "reverse words"}
    if re.search(r"\b(same order|in order|sequence|order|sequentially|one by one|each)\b", s):
        return {"mode": "sequence", "index": None, "why": "sequence words"}
    return {"mode": "unknown", "index": None, "why": "no ordinal"}


def resolve_index(idx, n):
    """Map a parsed (0-based or negative) index onto n placements (clamped)."""
    if n <= 0 or idx is None:
        return None
    if idx < 0:
        return max(0, n + idx)
    return min(idx, n - 1)


# ---------------------------------------------------------------------------
# Vision fallback for the video (no usable states): colour tracking of the cube
# ---------------------------------------------------------------------------
def color_rest_positions(frames_rgb, times, cube_rgb, P_exec, valid_exec, table_z,
                         col_tol=45.0, min_rest_s=0.8, move_tol=0.02):
    """Track pixels changed from frame 0 and close to the cube colour.

    Returns ordered list of rest positions [{pos, t0, t1}] (world xy via the execution depth).
    """
    if not frames_rgb or cube_rgb is None:
        return []
    ref = frames_rgb[0].astype(float)
    cube = np.asarray(cube_rgb, dtype=float)
    track = []  # (t, xy or None)
    for im, t in zip(frames_rgb, times):
        a = im.astype(float)
        changed = np.abs(a - ref).sum(axis=2) > 40
        close = np.linalg.norm(a - cube[None, None, :], axis=2) < col_tol
        m = changed & close
        if m.sum() < 6:
            track.append((t, None))
            continue
        vv, uu = np.nonzero(m)
        # largest blob: use median as a robust centre
        u0, v0 = int(np.median(uu)), int(np.median(vv))
        if not valid_exec[v0, u0]:
            track.append((t, None))
            continue
        track.append((t, P_exec[v0, u0, :2].copy()))
    rests = []
    cur = None
    for t, xy in track:
        if xy is None:
            if cur is not None:
                if cur["t1"] - cur["t0"] >= min_rest_s:
                    rests.append(cur)
                cur = None
            continue
        if cur is not None and np.linalg.norm(xy - cur["xy"]) < move_tol:
            cur["t1"] = t
            cur["n"] += 1
            cur["xy"] = (cur["xy"] * (cur["n"] - 1) + xy) / cur["n"]
        else:
            if cur is not None and cur["t1"] - cur["t0"] >= min_rest_s:
                rests.append(cur)
            cur = {"xy": xy.copy(), "t0": t, "t1": t, "n": 1}
    if cur is not None and cur["t1"] - cur["t0"] >= min_rest_s:
        rests.append(cur)
    merged = []
    for r in rests:
        if merged and np.linalg.norm(merged[-1]["xy"] - r["xy"]) < move_tol:
            merged[-1]["t1"] = r["t1"]
        else:
            merged.append(r)
    return [{"pos": [float(r["xy"][0]), float(r["xy"][1]), None], "t0": r["t0"], "t1": r["t1"]} for r in merged]
