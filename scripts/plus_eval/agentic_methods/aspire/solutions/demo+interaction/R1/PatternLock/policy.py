# PatternLock v00 -- initial program from offline scene analysis (demo+interaction, R=1).
#
# Strategy (all task-dependent quantities are perceived at runtime):
#   1. Detect the 5x5 lattice of white target discs in the conditioning video's first front frame
#      and in the live front frame (pure-Python blob detector + convex-hull-corner homography).
#   2. Read the pattern from the conditioning video: the ordered cells whose disc turns red
#      (touch feedback) in the front camera.  The video robot_state gives the end-effector pose at
#      every touch and the height profile between touches.
#   3. Map video cells to live cells by the cyclic corner assignment with the smallest corner
#      displacement (the live board may be shifted / rotated).
#   4. Localize each live target disc in 3-D (public depth + mask_to_point_cloud), add the
#      stick-tip offset measured in the video, and replay touches with the video's transit profile.
# Every phase is guarded; `result` is populated even when a phase fails.
#
# v01 (live_tip_contact) -- repair of the v00 episode-1 failure (zero touches registered):
#   The live drawing stick is NOT the stick seen in the conditioning video.  At identical joints
#   and wrist-camera pose (video frame 0 vs live frame 0) the stick ends at a different wrist
#   pixel, and on the expert execution the touched disc sits next to that pixel.  The video's
#   EE-minus-disc offset therefore left the live tip ~4 cm above and ~2-3 cm beside each disc.
#   v01 keeps v00's perception / pattern / mapping / disc localization unchanged and
#     a. measures the live stick tip at start from the public wrist depth (mask_to_point_cloud
#        on near-range wrist pixels; the lowest points are the tip) and expresses it in the EE frame;
#     b. places the TIP on each disc top, and on the first touch searches the contact height
#        downward in bounded steps, holding still and polling the red touch feedback each tick
#        (the video shows a ~5-tick lag between arrival and the red change);
#     c. reuses the registered height for later touches (bounded retries), and lifts during
#        transit so the tip never sweeps over intermediate discs at contact height.
#   If the tip calibration is implausible, it falls back to the video offset and a deeper,
#   coarser height search.
import math

report = {"version": "v01_live_tip_contact", "phases": {}, "errors": [], "warnings": []}
result = report

CFG = {
    "n_side": None,   # lattice size is derived at runtime (select_lattice tries 3..8 per side)
    "bg_radius": 7, "max_sat": 60, "min_lum": 300, "min_contrast": 60,
    "min_area": 3, "max_area": 200, "max_dim": 22, "merge_px": 7.0, "lattice_min_area": 12,
    "red_ratio": 0.4, "red_min_r": 80, "red_lift_frac": 0.3, "red_min_px": 15, "red_gap": 3,
    "hover_m": 0.06, "slide_chunk_m": 0.03, "transit_chunk_m": 0.06, "lift_min_m": 0.008,
    "default_lift_m": 0.03, "touch_tol": 0.005, "move_tol": 0.01, "dwell_ticks": 2,
    "correction_m": 0.005, "max_offset_xy_m": 0.04, "max_api_err_m": 0.015,
    # v01: live stick-tip calibration (wrist depth) and contact search with red feedback
    "stick_near_m": 0.35, "stick_near_frac": 0.6, "stick_min_px": 30, "tip_band_m": 0.004,
    "tip_max_norm_m": 0.25, "tip_max_up_m": 0.005,
    "cal_h0_m": 0.012, "cal_step_m": 0.004, "cal_n": 6, "cal_ext_step_m": 0.008, "cal_ext_n": 3,
    "fb_step_m": 0.008, "fb_n": 9, "fb_floor_above_disc_m": 0.02,
    "reg_dwell_max": 10, "reg_margin_m": 0.002, "later_retry": 2, "wrist_red_inc": 25,
    "transit_lift_m": 0.03, "approach_h_m": 0.05, "tick_reserve": 8, "lift_chunk_m": 0.15,
    "later_dwell_extra": 4, "press_max_steps": 15, "descend_max_steps": 25,
}
STATE = {"terminal": False, "ticks": 0, "max_steps": None, "n_moves": 0}


def note_error(where, err):
    msg = str(err)
    report["errors"].append({"where": where, "error": msg[:400]})
    if "ended" in msg or "closing" in msg:
        STATE["terminal"] = True


def r4(x):
    try:
        return round(float(x), 4)
    except Exception:
        return None


def vec(p):
    return [float(p[0]), float(p[1]), float(p[2])]


# ---------------------------------------------------------------- image helpers (pure python)

def roi_rows(arr, v0, v1, u0, u1):
    try:
        return arr[v0:v1, u0:u1].tolist()
    except Exception:
        pass
    out = []
    for v in range(v0, v1):
        row = arr[v]
        line = []
        for u in range(u0, u1):
            px = row[u]
            line.append([int(px[0]), int(px[1]), int(px[2])])
        out.append(line)
    return out


def img_shape(arr):
    try:
        return int(arr.shape[0]), int(arr.shape[1])
    except Exception:
        return len(arr), len(arr[0])


def find_disc_blobs(rows):
    h = len(rows)
    w = len(rows[0])
    lum = []
    sat = []
    for v in range(h):
        rr = rows[v]
        lrow = []
        srow = []
        for u in range(w):
            px = rr[u]
            a = int(px[0])
            b = int(px[1])
            c = int(px[2])
            lrow.append(a + b + c)
            srow.append(max(a, b, c) - min(a, b, c))
        lum.append(lrow)
        sat.append(srow)
    integ = [[0] * (w + 1) for k in range(h + 1)]
    for v in range(h):
        acc = 0
        prev = integ[v]
        cur = integ[v + 1]
        lrow = lum[v]
        for u in range(w):
            acc += lrow[u]
            cur[u + 1] = prev[u + 1] + acc
    rad = CFG["bg_radius"]
    cand = [[False] * w for k in range(h)]
    for v in range(h):
        v0 = max(0, v - rad)
        v1 = min(h, v + rad + 1)
        for u in range(w):
            if sat[v][u] > CFG["max_sat"] or lum[v][u] < CFG["min_lum"]:
                continue
            u0 = max(0, u - rad)
            u1 = min(w, u + rad + 1)
            area = (v1 - v0) * (u1 - u0)
            tot = integ[v1][u1] - integ[v0][u1] - integ[v1][u0] + integ[v0][u0]
            if lum[v][u] - tot / area > CFG["min_contrast"]:
                cand[v][u] = True
    seen = [[False] * w for k in range(h)]
    blobs = []
    for v in range(h):
        for u in range(w):
            if not cand[v][u] or seen[v][u]:
                continue
            stack = [(v, u)]
            seen[v][u] = True
            pts = []
            while len(stack) > 0:
                cell = stack.pop()
                pv = cell[0]
                pu = cell[1]
                pts.append(cell)
                for dv, du in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nv = pv + dv
                    nu = pu + du
                    if 0 <= nv < h and 0 <= nu < w and cand[nv][nu] and not seen[nv][nu]:
                        seen[nv][nu] = True
                        stack.append((nv, nu))
                if len(pts) > CFG["max_area"] * 4:
                    break
            n = len(pts)
            vs = [p[0] for p in pts]
            us = [p[1] for p in pts]
            bw = max(us) - min(us) + 1
            bh = max(vs) - min(vs) + 1
            if CFG["min_area"] <= n <= CFG["max_area"] and bw <= CFG["max_dim"] and bh <= CFG["max_dim"]:
                blobs.append({"u": sum(us) / float(n), "v": sum(vs) / float(n), "area": n, "bw": bw, "bh": bh})
    return blobs


def merge_blobs(blobs, merge_px):
    items = [dict(b) for b in blobs]
    changed = True
    guard = 0
    while changed and guard < 2000:
        guard += 1
        changed = False
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                a = items[i]
                b = items[j]
                if (a["u"] - b["u"]) ** 2 + (a["v"] - b["v"]) ** 2 <= merge_px * merge_px:
                    tot = a["area"] + b["area"]
                    items[i] = {"u": (a["u"] * a["area"] + b["u"] * b["area"]) / tot,
                                "v": (a["v"] * a["area"] + b["v"] * b["area"]) / tot,
                                "area": tot, "bw": max(a["bw"], b["bw"]), "bh": max(a["bh"], b["bh"])}
                    items.pop(j)
                    changed = True
                    break
            if changed:
                break
    return items


def solve_linear(a, b):
    n = len(b)
    m = [list(a[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = col
        for r in range(col + 1, n):
            if abs(m[r][col]) > abs(m[piv][col]):
                piv = r
        if abs(m[piv][col]) < 1e-12:
            return None
        tmp = m[col]
        m[col] = m[piv]
        m[piv] = tmp
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                if f != 0.0:
                    for c in range(col, n + 1):
                        m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def homography(src, dst):
    a = []
    b = []
    for k in range(4):
        x = src[k][0]
        y = src[k][1]
        tx = dst[k][0]
        ty = dst[k][1]
        a.append([x, y, 1.0, 0.0, 0.0, 0.0, -tx * x, -tx * y])
        b.append(tx)
        a.append([0.0, 0.0, 0.0, x, y, 1.0, -ty * x, -ty * y])
        b.append(ty)
    h = solve_linear(a, b)
    if h is None:
        return None
    return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]]


def apply_h(hm, x, y):
    d = hm[2][0] * x + hm[2][1] * y + hm[2][2]
    if abs(d) < 1e-12:
        return (1e9, 1e9)
    return ((hm[0][0] * x + hm[0][1] * y + hm[0][2]) / d, (hm[1][0] * x + hm[1][1] * y + hm[1][2]) / d)


def cross3(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points):
    pts = sorted(set((p[0], p[1]) for p in points))
    if len(pts) < 3:
        return pts
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross3(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross3(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def hull_corners(points):
    hull = convex_hull(points)
    n = len(hull)
    if n < 4:
        return None
    scored = []
    for i in range(n):
        p = hull[i - 1]
        q = hull[i]
        r = hull[(i + 1) % n]
        ax = p[0] - q[0]
        ay = p[1] - q[1]
        bx = r[0] - q[0]
        by = r[1] - q[1]
        na = math.sqrt(ax * ax + ay * ay)
        nb = math.sqrt(bx * bx + by * by)
        if na < 1e-9 or nb < 1e-9:
            continue
        scored.append(((ax * bx + ay * by) / (na * nb), i))
    scored.sort(reverse=True)
    if len(scored) < 4:
        return None
    idx = sorted(s[1] for s in scored[:4])
    return [hull[i] for i in idx]


def fit_lattice(blobs, ns):
    pts = [(b["u"], b["v"]) for b in blobs]
    corners = hull_corners(pts)
    if corners is None:
        return None
    g = float(ns - 1)
    hm = homography(corners, [(0.0, 0.0), (g, 0.0), (g, g), (0.0, g)])
    if hm is None:
        return None
    cells = {}
    resid = 0.0
    worst = 0.0
    for k in range(len(pts)):
        ab = apply_h(hm, pts[k][0], pts[k][1])
        ia = int(round(ab[0]))
        ib = int(round(ab[1]))
        e = math.sqrt((ab[0] - ia) ** 2 + (ab[1] - ib) ** 2)
        resid += e
        if e > worst:
            worst = e
        key = (ia, ib)
        if ia < 0 or ib < 0 or ia > ns - 1 or ib > ns - 1 or key in cells:
            return {"ok": False, "corners": corners, "worst": 9.0}
        cells[key] = k
    return {"ok": len(cells) == ns * ns and worst < 0.3, "corners": corners, "h": hm,
            "cells": cells, "mean_resid": resid / max(1, len(pts)), "worst": worst}


def blob_dist(a, b):
    return math.sqrt((a["u"] - b["u"]) ** 2 + (a["v"] - b["v"]) ** 2)


def lambda_free_key(group_size):
    # sort key: prefer the lattice size whose cell count is closest to the group size
    def key(ns):
        return abs(group_size - ns * ns)
    return key


def select_lattice(blobs):
    cands = [b for b in blobs if b["area"] >= CFG["lattice_min_area"]]
    log = {"n_cands": len(cands)}
    if len(cands) < 9:
        log["why"] = "too few candidates"
        return None, log
    n = len(cands)
    nn = []
    for i in range(n):
        best = 1e9
        for j in range(n):
            if i != j:
                d = blob_dist(cands[i], cands[j])
                if d < best:
                    best = d
        nn.append(best)
    adj = [[] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if blob_dist(cands[i], cands[j]) <= 1.6 * min(nn[i], nn[j]) + 1.0:
                adj[i].append(j)
                adj[j].append(i)
    comp = [-1] * n
    groups = []
    for i in range(n):
        if comp[i] >= 0:
            continue
        stack = [i]
        comp[i] = len(groups)
        members = []
        while len(stack) > 0:
            k = stack.pop()
            members.append(k)
            for j in adj[k]:
                if comp[j] < 0:
                    comp[j] = comp[i]
                    stack.append(j)
        groups.append(members)
    groups.sort(key=len, reverse=True)
    log["group_sizes"] = [len(g) for g in groups[:5]]
    tries = []
    for members in groups[:3]:
        sizes = [ns for ns in (3, 4, 5, 6, 7, 8) if ns * ns <= len(members)]
        sizes.sort(key=lambda_free_key(len(members)))
        for ns in sizes:
            tries.append((members, ns))
    for item in tries:
        members = item[0]
        ns = item[1]
        need = ns * ns
        sel = [cands[k] for k in members]
        guard = 0
        while len(sel) > need and guard < 60:
            guard += 1
            if len(sel) - 1 > need:
                us = sorted(b["u"] for b in sel)
                vs = sorted(b["v"] for b in sel)
                mu = us[len(us) // 2]
                mv = vs[len(vs) // 2]
                far = 0
                fd = -1.0
                for q in range(len(sel)):
                    d = (sel[q]["u"] - mu) ** 2 + (sel[q]["v"] - mv) ** 2
                    if d > fd:
                        fd = d
                        far = q
                sel = sel[:far] + sel[far + 1:]
            else:
                best_sel = None
                best_w = 1e9
                for drop in range(len(sel)):
                    trial = sel[:drop] + sel[drop + 1:]
                    fit = fit_lattice(trial, ns)
                    w = 1e9
                    if fit is not None and fit["ok"]:
                        w = fit["worst"]
                    if w < best_w:
                        best_w = w
                        best_sel = trial
                if best_sel is None:
                    break
                sel = best_sel
        fit = fit_lattice(sel, ns)
        if fit is not None and fit["ok"]:
            log["fit_mean_resid"] = r4(fit["mean_resid"])
            log["fit_worst"] = r4(fit["worst"])
            log["n_side"] = ns
            return {"blobs": sel, "fit": fit, "n_side": ns}, log
    log["why"] = "no group fits a lattice"
    return None, log


def lattice_from_rgb(rgb):
    hw = img_shape(rgb)
    rows = roi_rows(rgb, 0, hw[0], 0, hw[1])
    blobs = merge_blobs(find_disc_blobs(rows), CFG["merge_px"])
    lat, log = select_lattice(blobs)
    log["n_blobs"] = len(blobs)
    if lat is None:
        return None, log
    cells = {}
    for key in lat["fit"]["cells"]:
        b = lat["blobs"][lat["fit"]["cells"][key]]
        cells[key] = (b["u"], b["v"])
    radius = {}
    for key in cells:
        best = 1e9
        for other in cells:
            if other != key:
                d = math.sqrt((cells[key][0] - cells[other][0]) ** 2 + (cells[key][1] - cells[other][1]) ** 2)
                if d < best:
                    best = d
        radius[key] = 0.5 * best
    us = [cells[k][0] for k in cells]
    vs = [cells[k][1] for k in cells]
    rmax = max(radius[k] for k in radius)
    roi = [max(0, int(min(vs) - 2 * rmax - 2)), min(hw[0], int(max(vs) + rmax + 3)),
           max(0, int(min(us) - rmax - 2)), min(hw[1], int(max(us) + rmax + 3))]
    log["corners_px"] = [[r4(c[0]), r4(c[1])] for c in lat["fit"]["corners"]]
    return {"cells": cells, "radius": radius, "corners": lat["fit"]["corners"], "roi": roi, "shape": hw,
            "n_side": lat["n_side"]}, log


def red_counts(rgb, lat):
    roi = lat["roi"]
    rows = roi_rows(rgb, roi[0], roi[1], roi[2], roi[3])
    ratio = CFG["red_ratio"]
    rmin = CFG["red_min_r"]
    reds = []
    for dv in range(len(rows)):
        rr = rows[dv]
        for du in range(len(rr)):
            px = rr[du]
            r = px[0]
            if r < rmin:
                continue
            if px[1] < ratio * r and px[2] < ratio * r:
                reds.append((du + roi[2], dv + roi[0]))
    counts = {}
    cells = lat["cells"]
    radius = lat["radius"]
    keys = list(cells.keys())
    for pt in reds:
        best = None
        bd = 1e9
        for k in keys:
            du = pt[0] - cells[k][0]
            dv = pt[1] - (cells[k][1] - CFG["red_lift_frac"] * radius[k])
            d = math.sqrt(du * du + dv * dv)
            if d < bd:
                bd = d
                best = k
        if best is not None and bd <= 1.3 * radius[best]:
            counts[best] = counts.get(best, 0) + 1
    return counts, len(reds)


def red_events(series):
    min_px = CFG["red_min_px"]
    gap = CFG["red_gap"]
    events = []
    last_red = {}
    n = len(series)
    for i in range(n):
        counts = series[i]
        top = 0
        for key in counts:
            if counts[key] > top:
                top = counts[key]
        fresh = []
        for key in counts:
            c = counts[key]
            if c < min_px or c < 0.3 * top:
                continue
            nxt_ok = (i + 1 >= n) or (series[i + 1].get(key, 0) >= min_px)
            if not nxt_ok:
                continue
            prev = last_red.get(key, None)
            if prev is None or i - prev > gap:
                fresh.append((c, key))
            last_red[key] = i
        fresh.sort(reverse=True)
        for item in fresh:
            events.append({"cell": item[1], "sample": i, "px": item[0]})
    return events


# ---------------------------------------------------------------- geometry helpers

def quat_to_rot(q):
    w = float(q[0])
    x = float(q[1])
    y = float(q[2])
    z = float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    w = w / n
    x = x / n
    y = y / n
    z = z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def camera_model(cam):
    k = cam.intrinsics
    pose = cam.camera_pose
    return {"fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]), "cy": float(k[1][2]),
            "R": quat_to_rot(pose.quaternion_wxyz), "t": vec(pose.position)}


def depth_at(depth, u, v, hw):
    vals = []
    iu = int(round(u))
    iv = int(round(v))
    for dv in (-1, 0, 1):
        for du in (-1, 0, 1):
            pv = iv + dv
            pu = iu + du
            if 0 <= pv < hw[0] and 0 <= pu < hw[1]:
                try:
                    d = float(depth[pv][pu])
                except Exception:
                    continue
                if d > 0.0 and d < 10.0:
                    vals.append(d)
    if len(vals) == 0:
        return None
    vals.sort()
    return vals[len(vals) // 2]


CONVENTIONS = ["opencv", "opengl", "xfwd_yleft_zup"]


def project(model, u, v, d, conv):
    xn = (u - model["cx"]) / model["fx"]
    yn = (v - model["cy"]) / model["fy"]
    if conv == "opencv":
        p = (xn * d, yn * d, d)
    elif conv == "opengl":
        p = (xn * d, -yn * d, -d)
    else:
        p = (d, -xn * d, -yn * d)
    r = model["R"]
    t = model["t"]
    return [r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0],
            r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1],
            r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2]]


def lattice_3d(lat, cam, conv):
    model = camera_model(cam)
    depth = cam.depth_m
    out = {}
    for key in lat["cells"]:
        uv = lat["cells"][key]
        d = depth_at(depth, uv[0], uv[1], lat["shape"])
        if d is None:
            continue
        out[key] = project(model, uv[0], uv[1], d, conv)
    return out


def planarity(points3d):
    zs = [points3d[k][2] for k in points3d]
    if len(zs) < 4:
        return 9.0
    mz = sum(zs) / len(zs)
    return math.sqrt(sum((z - mz) ** 2 for z in zs) / len(zs))


def median(values):
    s = sorted(values)
    if len(s) == 0:
        return None
    return s[len(s) // 2]


def quantile(values, frac):
    s = sorted(values)
    if len(s) == 0:
        return None
    return s[int((len(s) - 1) * frac)]


def cloud_summary(pc):
    pts = pc.points
    n = len(pts)
    if n < 3:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    zs = [float(p[2]) for p in pts]
    return {"n": n, "center": [median(xs), median(ys), quantile(zs, 0.9)],
            "span": [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)]}


def api_disc_point(lat, key, cam_name, base_frame, hw, mode):
    uv = lat["cells"][key]
    info = {"cell": [key[0], key[1]], "mode": mode}
    pc = None
    if mode == "own_mask":
        rad = 1.6
        mask = [[False] * hw[1] for k in range(hw[0])]
        for v in range(int(uv[1] - 2), int(uv[1] + 3)):
            for u in range(int(uv[0] - 2), int(uv[0] + 3)):
                if 0 <= v < hw[0] and 0 <= u < hw[1] and (u - uv[0]) ** 2 + (v - uv[1]) ** 2 <= rad * rad:
                    mask[v][u] = True
        pc = mask_to_point_cloud(mask, cam_name, target_frame=base_frame)
    else:
        seg = segment_points(cam_name, [[int(round(uv[0])), int(round(uv[1]))]])
        info["seg_ok"] = bool(seg.ok)
        if not seg.ok or len(seg.segmentations) == 0:
            info["why"] = "segment_points empty"
            return None, info
        cand = seg.segmentations[0]
        try:
            info["score"] = r4(cand.score)
        except Exception:
            pass
        pc = mask_to_point_cloud(cand, cam_name, target_frame=base_frame)
    summ = cloud_summary(pc)
    if summ is None:
        info["why"] = "too few points"
        return None, info
    info["n"] = summ["n"]
    info["span"] = [r4(s) for s in summ["span"]]
    if mode == "segment_points" and (summ["span"][0] > 0.08 or summ["span"][1] > 0.08):
        info["why"] = "mask too large for one disc"
        return None, info
    info["center"] = [r4(c) for c in summ["center"]]
    return summ["center"], info


def dist3(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def dist2(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ---------------------------------------------------------------- motion helpers

def read_terminal(res):
    term = False
    try:
        if res.terminated:
            term = True
    except Exception:
        pass
    try:
        if res.truncated:
            term = True
    except Exception:
        pass
    return term


def ee_now():
    st = get_robot_state()
    pose = st.end_effector_poses["primary"]
    return vec(pose.position), [float(q) for q in pose.quaternion_wxyz], st


def move_ee(pos, quat, tol, max_steps, tag, log):
    rec = {"tag": tag, "target": [r4(p) for p in pos]}
    log.append(rec)
    if STATE["terminal"]:
        rec["skipped"] = "terminal"
        return False
    if STATE["max_steps"] is not None and STATE["ticks"] >= STATE["max_steps"] - 5:
        rec["skipped"] = "tick budget"
        return False
    STATE["n_moves"] += 1
    try:
        res = move_to_pose(Pose([pos[0], pos[1], pos[2]], quat, BASE["frame"]), tolerance=tol, max_steps=max_steps)
    except Exception as err:
        rec["error"] = str(err)[:300]
        note_error("move_ee:" + tag, err)
        return False
    ok = False
    try:
        ok = bool(res.ok)
    except Exception:
        pass
    rec["ok"] = ok
    try:
        rec["status"] = str(res.status)
    except Exception:
        pass
    try:
        n = int(res.steps_executed)
        rec["steps"] = n
        STATE["ticks"] += n
    except Exception:
        pass
    if read_terminal(res):
        rec["terminal"] = True
        STATE["terminal"] = True
        return ok
    if not ok:
        try:
            rec["err"] = str(res.error)[:300]
        except Exception:
            pass
    try:
        cur = ee_now()
        rec["reached"] = [r4(p) for p in cur[0]]
        rec["err_m"] = r4(dist3(cur[0], pos))
        rec["t"] = r4(cur[2].timestamp_s)
    except Exception as err:
        note_error("move_ee.state:" + tag, err)
    return ok


def wrist_red_total():
    o = get_observation()
    w = o.cameras["wrist"].rgb
    hw = img_shape(w)
    rows = roi_rows(w, 0, hw[0], 0, hw[1])
    n = 0
    for v in range(0, hw[0], 2):
        rr = rows[v]
        for u in range(0, hw[1], 2):
            px = rr[u]
            if px[0] >= CFG["red_min_r"] and px[1] < CFG["red_ratio"] * px[0] and px[2] < CFG["red_ratio"] * px[0]:
                n += 1
    return n, o


def dwell(n, log):
    for tick in range(n):
        if STATE["terminal"]:
            return
        try:
            st = get_robot_state()
            reply = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=st.joint_positions["primary"],
                gripper_position=st.gripper_positions["primary"],
                embodiment=st.embodiment)}))
            STATE["ticks"] += 1
            if read_terminal(reply):
                STATE["terminal"] = True
                log.append({"tag": "dwell", "terminal": True})
                return
        except Exception as err:
            note_error("dwell", err)
            return


def lerp(a, b, f):
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]


def chunked_move(start, goal, quat, chunk, tol, tag, log):
    d = dist3(start, goal)
    n = max(1, int(math.ceil(d / chunk)))
    ok = True
    for k in range(1, n + 1):
        if STATE["terminal"]:
            return False
        ok = move_ee(lerp(start, goal, k / float(n)), quat, tol, 60, tag + "_" + str(k) + "of" + str(n), log)
    return ok


def mat_vec(m, v):
    return [m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]]


def mat_t_vec(m, v):
    return [m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
            m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
            m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2]]


def calibrate_tip(obs, ee_pos, ee_quat):
    # Live stick tip from the public wrist depth: near-range wrist pixels are the stick / hand;
    # the lowest of their robot_base points is the tip.  Returned in the EE frame.
    info = {}
    cam = obs.cameras["wrist"]
    depth = cam.depth_m
    hw = img_shape(depth)
    samp = []
    for v in range(0, hw[0], 8):
        for u in range(0, hw[1], 8):
            d = float(depth[v][u])
            if d > 0.0 and d < 100.0:
                samp.append(d)
    if len(samp) < 10:
        info["why"] = "no wrist depth"
        return None, info
    far = median(samp)
    near = min(CFG["stick_near_m"], CFG["stick_near_frac"] * far)
    info["far_m"] = r4(far)
    info["near_m"] = r4(near)
    mask = []
    n = 0
    umin = hw[1]
    umax = -1
    vmin = hw[0]
    vmax = -1
    for v in range(hw[0]):
        row = depth[v]
        line = []
        for u in range(hw[1]):
            d = float(row[u])
            ok = d > 0.005 and d < near
            line.append(ok)
            if ok:
                n += 1
                umin = min(umin, u)
                umax = max(umax, u)
                vmin = min(vmin, v)
                vmax = max(vmax, v)
        mask.append(line)
    info["n_px"] = n
    info["bbox_uv"] = [umin, vmin, umax, vmax]
    if n < CFG["stick_min_px"]:
        info["why"] = "too few near pixels"
        return None, info
    pts = []
    try:
        pc = mask_to_point_cloud(mask, "wrist", target_frame=BASE["frame"])
        pts = pc.points
        info["source"] = "mask_to_point_cloud"
    except Exception as err:
        info["api_error"] = str(err)[:200]
    if len(pts) < CFG["stick_min_px"] and CONV["name"] is not None:
        # fallback: back-project with the wrist intrinsics/pose and the convention verified in Phase 2
        model = camera_model(cam)
        pts = []
        for v in range(hw[0]):
            for u in range(hw[1]):
                if mask[v][u]:
                    pts.append(project(model, u, v, float(depth[v][u]), CONV["name"]))
        info["source"] = "manual_" + str(CONV["name"])
    info["n_pts"] = len(pts)
    if len(pts) < CFG["stick_min_px"]:
        info["why"] = "too few points"
        return None, info
    zs = sorted([float(p[2]) for p in pts])
    zt = zs[min(2, len(zs) - 1)]
    sel = [p for p in pts if float(p[2]) <= zt + CFG["tip_band_m"]]
    if len(sel) == 0:
        info["why"] = "no valid tip points"
        return None, info
    tip = [median([float(p[0]) for p in sel]), median([float(p[1]) for p in sel]), zt]
    w = [tip[0] - ee_pos[0], tip[1] - ee_pos[1], tip[2] - ee_pos[2]]
    info["tip_world"] = [r4(x) for x in tip]
    info["tip_minus_ee_world"] = [r4(x) for x in w]
    info["n_tip_pts"] = len(sel)
    norm = math.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)
    if norm > CFG["tip_max_norm_m"] or w[2] > CFG["tip_max_up_m"]:
        info["why"] = "implausible tip offset"
        return None, info
    local = mat_t_vec(quat_to_rot(ee_quat), w)
    info["tip_local"] = [r4(x) for x in local]
    return local, info


def wrist_red_obs(o):
    w = o.cameras["wrist"].rgb
    hw = img_shape(w)
    rows = roi_rows(w, 0, hw[0], 0, hw[1])
    n = 0
    for v in range(0, hw[0], 2):
        rr = rows[v]
        for u in range(0, hw[1], 2):
            px = rr[u]
            if px[0] >= CFG["red_min_r"] and px[1] < CFG["red_ratio"] * px[0] and px[2] < CFG["red_ratio"] * px[0]:
                n += 1
    return n


def budget_low():
    return STATE["max_steps"] is not None and STATE["ticks"] >= STATE["max_steps"] - CFG["tick_reserve"]


def red_state(cell):
    o = get_observation()
    cnt, ntot = red_counts(o.cameras["front"].rgb, LIVE["lat"])
    return cnt.get(cell, 0), wrist_red_obs(o), cnt


def wait_register(cell, n_max, log):
    # Hold still and poll the red touch feedback each tick (front target cell, wrist increase).
    rec = {"polls": 0}
    if STATE["terminal"]:
        # the episode ended on arrival (normally: the last pattern disc registered)
        rec["terminal"] = True
        return True, rec
    f, w, cnt = red_state(cell)
    base_w = w
    rec["front0"] = f
    rec["wrist0"] = w
    if f >= CFG["red_min_px"]:
        rec["front"] = f
        rec["wrist"] = w
        rec["cnt"] = cnt
        return True, rec
    for i in range(n_max):
        if STATE["terminal"] or budget_low():
            rec["stopped"] = "terminal" if STATE["terminal"] else "tick budget"
            break
        dwell(1, log)
        if STATE["terminal"]:
            rec["terminal"] = True
            rec["polls"] = i + 1
            return True, rec
        f, w, cnt = red_state(cell)
        rec["polls"] = i + 1
        if f >= CFG["red_min_px"] or w - base_w >= CFG["wrist_red_inc"]:
            rec["front"] = f
            rec["wrist"] = w
            rec["cnt"] = cnt
            return True, rec
    rec["front"] = f
    rec["wrist"] = w
    rec["cnt"] = cnt
    return False, rec


BASE = {"frame": "robot_base"}

# ================================================================ Phase 0: context and metadata
try:
    ctx = get_task_context()
    report["goal"] = str(ctx.language)
    report["task_name"] = str(ctx.task_name)
    st0 = get_robot_state()
    BASE["frame"] = st0.base_frame
    report["base_frame"] = str(st0.base_frame)
    report["embodiment"] = str(st0.embodiment)
    report["start_ee"] = [r4(p) for p in vec(st0.end_effector_poses["primary"].position)]
    report["start_quat"] = [r4(q) for q in st0.end_effector_poses["primary"].quaternion_wxyz]
    report["start_joints"] = [r4(q) for q in st0.joint_positions["primary"]]
    report["start_gripper"] = r4(st0.gripper_positions["primary"])
    report["start_t"] = r4(st0.timestamp_s)
except Exception as err:
    note_error("phase0.context", err)
try:
    meta = robomme_plus.get_task_metadata()
    report["task_metadata"] = str(meta)[:3000]
    try:
        STATE["max_steps"] = int(meta["max_steps"])
    except Exception as err:
        note_error("phase0.max_steps", err)
    report["max_steps"] = STATE["max_steps"]
except Exception as err:
    note_error("phase0.task_metadata", err)
try:
    report["controller_metadata"] = str(robomme_plus.get_controller_metadata())[:3000]
except Exception as err:
    note_error("phase0.controller_metadata", err)

# ================================================================ Phase 1: live lattice
LIVE = {"lat": None, "obs": None, "cam": None}
try:
    obs = get_observation()
    LIVE["obs"] = obs
    report["cameras"] = [str(c) for c in obs.cameras]
    cam = obs.cameras["front"]
    LIVE["cam"] = cam
    report["live_front_pose"] = {"p": [r4(x) for x in vec(cam.camera_pose.position)],
                                 "q": [r4(x) for x in cam.camera_pose.quaternion_wxyz],
                                 "frame": str(cam.camera_pose.frame)}
    report["live_front_K"] = [[r4(cam.intrinsics[i][j]) for j in range(3)] for i in range(3)]
    lat, log = lattice_from_rgb(cam.rgb)
    report["phases"]["live_lattice"] = log
    LIVE["lat"] = lat
    if lat is None:
        report["warnings"].append("live lattice not found")
    else:
        c0, n0 = red_counts(cam.rgb, lat)
        report["live_initial_red_px"] = n0
except Exception as err:
    note_error("phase1.live_lattice", err)

# ================================================================ Phase 2: projection convention
CONV = {"name": None, "api_mode": None}
try:
    if LIVE["lat"] is not None:
        lat = LIVE["lat"]
        scores = {}
        live3 = {}
        for conv in CONVENTIONS:
            pts = lattice_3d(lat, LIVE["cam"], conv)
            live3[conv] = pts
            sc = planarity(pts)
            camz = float(LIVE["cam"].camera_pose.position[2])
            zs = [pts[kk][2] for kk in pts]
            if len(zs) == 0 or sum(zs) / len(zs) >= camz:
                sc += 1.0
            scores[conv] = r4(sc)
        report["phases"]["conv_planarity_std_m"] = scores
        probes = []
        api_pts = {}
        for key in ((0, 0), (4, 4), (2, 2), (0, 4)):
            got = None
            for mode in ("own_mask", "segment_points"):
                if CONV["api_mode"] is not None and mode != CONV["api_mode"]:
                    continue
                try:
                    got, info = api_disc_point(lat, key, "front", BASE["frame"], lat["shape"], mode)
                    probes.append(info)
                    if got is not None:
                        CONV["api_mode"] = mode
                        break
                except Exception as err:
                    probes.append({"cell": [key[0], key[1]], "mode": mode, "error": str(err)[:200]})
            if got is not None:
                api_pts[key] = got
        report["phases"]["api_probes"] = probes
        errs = {}
        for conv in CONVENTIONS:
            if len(api_pts) > 0:
                e = [dist3(live3[conv][k], api_pts[k]) for k in api_pts if k in live3[conv]]
                errs[conv] = r4(max(e)) if len(e) > 0 else None
        report["phases"]["conv_api_err_m"] = errs
        best = None
        if len(errs) > 0:
            for conv in CONVENTIONS:
                if errs.get(conv) is not None and (best is None or errs[conv] < errs[best]):
                    best = conv
            if best is not None and errs[best] > CFG["max_api_err_m"]:
                report["warnings"].append("best projection convention disagrees with API by " + str(errs[best]))
        if best is None:
            for conv in CONVENTIONS:
                if best is None or scores[conv] < scores[best]:
                    best = conv
        CONV["name"] = best
        report["projection_convention"] = best
        LIVE["pts3"] = live3[best]
        LIVE["api_pts"] = api_pts
except Exception as err:
    note_error("phase2.convention", err)

# ================================================================ Phase 2b (v01): live stick-tip calibration (start pose, before any motion)
TIP = {"local": None, "world_touch": None}
try:
    if LIVE["obs"] is not None:
        cur0 = ee_now()
        local, cinfo = calibrate_tip(LIVE["obs"], cur0[0], cur0[1])
        report["phases"]["tip_calibration"] = cinfo
        TIP["local"] = local
        if local is None:
            report["warnings"].append("live stick-tip calibration failed: " + str(cinfo.get("why")))
except Exception as err:
    note_error("phase2b.tip_calibration", err)

# ================================================================ Phase 3: conditioning video
VID = {"lat": None, "events": [], "ee": [], "ts": []}
try:
    video = robomme_plus.get_conditioning_video()
    nvid = len(video)
    report["video_frames"] = nvid
    if nvid > 0:
        report["video_t0"] = r4(video[0].timestamp_s)
        report["video_t1"] = r4(video[nvid - 1].timestamp_s)
        vlat = None
        for fi in (0, nvid - 1, nvid // 2):
            vlat, vlog = lattice_from_rgb(video[fi].cameras["front"].rgb)
            vlog["frame"] = fi
            report["phases"]["video_lattice"] = vlog
            if vlat is not None:
                VID["lat_frame"] = fi
                break
        VID["lat"] = vlat
        series = []
        red_tot = []
        for fi in range(nvid):
            fr = video[fi]
            try:
                pose = fr.robot_state.end_effector_poses["primary"]
                VID["ee"].append(vec(pose.position))
                VID.setdefault("quats", []).append([float(q) for q in pose.quaternion_wxyz])
                VID["ts"].append(float(fr.timestamp_s))
            except Exception as err:
                VID["ee"].append(None)
                VID.setdefault("quats", []).append(None)
                VID["ts"].append(None)
                if fi == 0:
                    note_error("phase3.video_state", err)
            if vlat is not None:
                cnt, ntot = red_counts(fr.cameras["front"].rgb, vlat)
                series.append(cnt)
                red_tot.append(ntot)
        report["video_red_px_per_frame"] = red_tot
        report["video_ee_mm"] = [[int(round(1000 * p[0])), int(round(1000 * p[1])), int(round(1000 * p[2]))]
                                 if p is not None else None for p in VID["ee"]]
        if vlat is not None:
            VID["events"] = red_events(series)
            report["video_events_raw"] = [{"cell": [e["cell"][0], e["cell"][1]], "sample": e["sample"],
                                           "px": e["px"]} for e in VID["events"]]
        try:
            vcam = video[VID.get("lat_frame", 0)].cameras["front"]
            report["video_front_pose"] = {"p": [r4(x) for x in vec(vcam.camera_pose.position)],
                                          "q": [r4(x) for x in vcam.camera_pose.quaternion_wxyz]}
            if vlat is not None and CONV["name"] is not None:
                VID["pts3"] = lattice_3d(vlat, vcam, CONV["name"])
                report["video_lattice_planarity_m"] = r4(planarity(VID["pts3"]))
        except Exception as err:
            note_error("phase3.video_3d", err)
    else:
        report["warnings"].append("conditioning video empty")
except Exception as err:
    note_error("phase3.video", err)

# ================================================================ Phase 4: pattern + touch geometry
PAT = {"cells": [], "touch_ee": [], "touch_z": [], "lifts": [], "samples": []}
try:
    seq = []
    samples = []
    for e in VID["events"]:
        if e["cell"] in seq:
            report["warnings"].append("repeat cell dropped " + str([e["cell"][0], e["cell"][1]]))
            continue
        seq.append(e["cell"])
        samples.append(e["sample"])
    if len(seq) == 0 and len(VID["ee"]) > 0 and VID.get("pts3") is not None:
        # fallback: EE height dips mapped to the nearest video disc
        report["warnings"].append("no red events; using EE dips")
        zs = [p[2] for p in VID["ee"] if p is not None]
        zmin = min(zs)
        inside = False
        for i in range(len(VID["ee"])):
            p = VID["ee"][i]
            if p is None:
                continue
            low = p[2] < zmin + 0.01
            if low and not inside:
                best = None
                bd = 1e9
                for k in VID["pts3"]:
                    d = dist2(p, VID["pts3"][k])
                    if d < bd:
                        bd = d
                        best = k
                if best is not None and best not in seq:
                    seq.append(best)
                    samples.append(i)
            inside = low
    PAT["cells"] = seq
    PAT["samples"] = samples
    report["pattern_video_cells"] = [[c[0], c[1]] for c in seq]
    # touch EE: lowest EE within +-5 samples of the event
    PAT["touch_quat"] = []
    for s in samples:
        best = None
        bq = None
        for i in range(max(0, s - 5), min(len(VID["ee"]), s + 6)):
            p = VID["ee"][i]
            if p is not None and (best is None or p[2] < best[2]):
                best = p
                bq = VID["quats"][i]
        PAT["touch_ee"].append(best)
        PAT["touch_quat"].append(bq)
    report["touch_quat_video"] = [[r4(x) for x in q] if q is not None else None for q in PAT["touch_quat"]]
    report["touch_ee_video"] = [[r4(x) for x in p] if p is not None else None for p in PAT["touch_ee"]]
    # transit profile between touches (relative lift over the touch height)
    for k in range(len(samples) - 1):
        a = samples[k]
        b = samples[k + 1]
        lo = a + (b - a) // 4
        hi = b - (b - a) // 4
        zs = [VID["ee"][i][2] for i in range(lo, hi + 1) if 0 <= i < len(VID["ee"]) and VID["ee"][i] is not None]
        za = PAT["touch_ee"][k]
        zb = PAT["touch_ee"][k + 1]
        if len(zs) > 0 and za is not None and zb is not None:
            PAT["lifts"].append(max(zs) - 0.5 * (za[2] + zb[2]))
        else:
            PAT["lifts"].append(None)
    report["video_lifts_m"] = [r4(x) for x in PAT["lifts"]]
    # tip offset: video touch EE minus video disc (3-D)
    offs = []
    if VID.get("pts3") is not None:
        for k in range(len(seq)):
            p = PAT["touch_ee"][k]
            if p is not None and seq[k] in VID["pts3"]:
                q = VID["pts3"][seq[k]]
                offs.append([p[0] - q[0], p[1] - q[1], p[2] - q[2]])
    report["tip_offsets_video"] = [[r4(x) for x in o] for o in offs]
    if len(offs) > 0:
        off = [median([o[0] for o in offs]), median([o[1] for o in offs]), median([o[2] for o in offs])]
        spread = max(dist2(o, off) for o in offs)
        report["tip_offset_xy_spread_m"] = r4(spread)
        if math.sqrt(off[0] ** 2 + off[1] ** 2) > CFG["max_offset_xy_m"]:
            report["warnings"].append("tip xy offset implausible; zeroed")
            off = [0.0, 0.0, off[2]]
        if abs(off[2]) > 0.3:
            report["warnings"].append("tip z offset implausible; offset not used")
        else:
            PAT["offset"] = off
            report["tip_offset_used"] = [r4(x) for x in off]
except Exception as err:
    note_error("phase4.pattern", err)

# ================================================================ Phase 5: video->live mapping and targets
TGT = {"points": [], "cells": [], "discs": []}
try:
    vlat = VID["lat"]
    llat = LIVE["lat"]
    shift = 0
    if vlat is not None and llat is not None:
        vc = vlat["corners"]
        lc = llat["corners"]
        best = None
        costs = []
        for s in range(4):
            c = sum(math.sqrt((vc[i][0] - lc[(i + s) % 4][0]) ** 2 + (vc[i][1] - lc[(i + s) % 4][1]) ** 2) for i in range(4))
            costs.append(r4(c))
            if best is None or c < best:
                best = c
                shift = s
        report["corner_shift_costs_px"] = costs
    report["corner_shift"] = shift
    nside = None
    if vlat is not None:
        nside = vlat["n_side"]
    elif llat is not None:
        nside = llat["n_side"]
    if vlat is not None and llat is not None and vlat["n_side"] != llat["n_side"]:
        report["warnings"].append("video and live lattice sizes differ; live lattice not used")
        llat = None
    report["n_side"] = nside
    ns1 = (nside if nside is not None else 1) - 1
    grid_corner = [(0, 0), (ns1, 0), (ns1, ns1), (0, ns1)]

    def map_cell(cell):
        # video grid coords -> live grid coords: the live lattice's corner list is rotated by `shift`,
        # i.e. live corner (i + shift) % 4 plays video corner i.  Rotate the grid accordingly.
        a = cell[0]
        b = cell[1]
        for k in range(shift):
            # one cyclic step of the corner order (0,0)->(g,0)->(g,g)->(0,g) is a 90-degree grid turn
            na = ns1 - b
            nb = a
            a = na
            b = nb
        return (a, b)

    # sanity: mapping of the corners must be consistent with the shift definition
    check_ok = True
    for i in range(4):
        if map_cell(grid_corner[i]) != grid_corner[(i + shift) % 4]:
            check_ok = False
    report["map_check_ok"] = check_ok
    if not check_ok:
        report["warnings"].append("corner mapping self-check failed")
    quat = None
    for q in PAT.get("touch_quat", []):
        if q is not None:
            quat = q
            break
    try:
        cur = ee_now()
        report["live_quat"] = [r4(q) for q in cur[1]]
        if quat is None:
            quat = cur[1]
    except Exception as err:
        note_error("phase5.ee", err)
    TGT["quat"] = quat
    off = PAT.get("offset", None)
    for k in range(len(PAT["cells"])):
        vcell = PAT["cells"][k]
        lcell = map_cell(vcell) if llat is not None else vcell
        disc = None
        src = None
        if llat is not None and lcell in llat["cells"]:
            if LIVE.get("api_pts") is not None and lcell in LIVE["api_pts"]:
                disc = LIVE["api_pts"][lcell]
                src = "api"
            elif CONV["api_mode"] is not None:
                try:
                    got, info = api_disc_point(llat, lcell, "front", BASE["frame"], llat["shape"], CONV["api_mode"])
                    if got is not None:
                        disc = got
                        src = "api"
                except Exception as err:
                    note_error("phase5.api_disc", err)
            if disc is None and LIVE.get("pts3") is not None and lcell in LIVE["pts3"]:
                disc = LIVE["pts3"][lcell]
                src = "projection"
        tp = None
        if disc is not None and off is not None:
            tp = [disc[0] + off[0], disc[1] + off[1], disc[2] + off[2]]
            src = src + "+offset"
        elif disc is not None and PAT["touch_ee"][k] is not None:
            tp = [disc[0], disc[1], PAT["touch_ee"][k][2]]
            src = src + "+video_z"
        elif PAT["touch_ee"][k] is not None:
            tp = list(PAT["touch_ee"][k])
            src = "video_ee_only"
            report["warnings"].append("target from raw video EE (assumes unchanged board)")
        if tp is not None:
            TGT["points"].append(tp)
            TGT["cells"].append(lcell)
            TGT["discs"].append(disc)
        report.setdefault("targets", []).append({"video_cell": [vcell[0], vcell[1]], "live_cell": [lcell[0], lcell[1]],
                                                 "disc": [r4(x) for x in disc] if disc is not None else None,
                                                 "target": [r4(x) for x in tp] if tp is not None else None,
                                                 "source": src})
    if len(TGT["points"]) == 0:
        report["warnings"].append("no targets derived")
except Exception as err:
    note_error("phase5.targets", err)

# ================================================================ Phase 5b (v01): live tip offset at the touch orientation
try:
    if TIP["local"] is not None and TGT.get("quat") is not None:
        TIP["world_touch"] = mat_vec(quat_to_rot(TGT["quat"]), TIP["local"])
        report["tip_minus_ee_touch_world"] = [r4(x) for x in TIP["world_touch"]]
        if PAT.get("offset") is not None:
            vo = PAT["offset"]
            report["live_vs_video_tip_delta_m"] = [r4(-TIP["world_touch"][i] - vo[i]) for i in range(3)]
except Exception as err:
    note_error("phase5b.tip_world", err)
    TIP["world_touch"] = None

# ================================================================ Phase 6 (v01): tip-on-disc touches with red feedback
MOVES = []
report["moves"] = MOVES
TOUCH = []
report["touches"] = TOUCH
CONTACT = {"h": None, "found": False, "mode": None, "tried": []}
report["contact"] = CONTACT
DONE = []


def touch_point(k, h):
    # EE position that puts the live stick tip h metres above the top of target disc k
    d = TGT["discs"][k]
    if d is not None and TIP["world_touch"] is not None:
        w = TIP["world_touch"]
        return [d[0] - w[0], d[1] - w[1], d[2] - w[2] + h]
    p = TGT["points"][k]
    return [p[0], p[1], p[2] + h]


def h_floor():
    if TIP["world_touch"] is not None:
        return -(CFG["cal_step_m"] * (CFG["cal_n"] - 1) - CFG["cal_h0_m"]) - CFG["cal_ext_step_m"] * CFG["cal_ext_n"]
    return -CFG["fb_step_m"] * (CFG["fb_n"] - 1)


def search_heights():
    hs = []
    if TIP["world_touch"] is not None:
        CONTACT["mode"] = "live_tip"
        for i in range(CFG["cal_n"]):
            hs.append(CFG["cal_h0_m"] - CFG["cal_step_m"] * i)
        last = hs[len(hs) - 1]
        for i in range(CFG["cal_ext_n"]):
            hs.append(last - CFG["cal_ext_step_m"] * (i + 1))
    else:
        CONTACT["mode"] = "video_offset_fallback"
        for i in range(CFG["fb_n"]):
            hs.append(-CFG["fb_step_m"] * i)
    return hs


def fb_ok(k, h):
    # fallback mode: never command the EE closer than fb_floor_above_disc_m to the disc top
    if TIP["world_touch"] is not None or TGT["discs"][k] is None:
        return True
    return touch_point(k, h)[2] >= TGT["discs"][k][2] + CFG["fb_floor_above_disc_m"]


def note_touch(k, ok, rec, h):
    rec["k"] = k
    rec["cell"] = [TGT["cells"][k][0], TGT["cells"][k][1]]
    rec["h"] = r4(h)
    rec["registered"] = bool(ok)
    if not STATE["terminal"]:
        try:
            cur = ee_now()
            rec["ee"] = [r4(x) for x in cur[0]]
            rec["t"] = r4(cur[2].timestamp_s)
        except Exception as err:
            note_error("note_touch", err)
    cnt = rec.pop("cnt", None)
    if cnt is not None:
        rec["red_cells"] = [[c[0], c[1], cnt[c]] for c in cnt if cnt[c] >= CFG["red_min_px"]]
        extra = [[c[0], c[1]] for c in cnt if cnt[c] >= CFG["red_min_px"] and c != TGT["cells"][k] and c not in DONE]
        if len(extra) > 0:
            rec["unexpected_red"] = extra
            report["warnings"].append("unexpected red cells at touch " + str(k) + ": " + str(extra))
    if ok:
        DONE.append(TGT["cells"][k])
    TOUCH.append(rec)


try:
    pts = TGT["points"]
    quat = TGT.get("quat", None)
    if quat is None:
        quat = ee_now()[1]
    if len(pts) == 0 or LIVE["lat"] is None:
        report["warnings"].append("no targets or no live lattice; holding one tick only")
        dwell(1, MOVES)
    else:
        hs = search_heights()
        report["search_heights_m"] = [r4(h) for h in hs]
        hover = touch_point(0, CFG["approach_h_m"])
        ok = move_ee(hover, quat, CFG["move_tol"], 200, "hover_first", MOVES)
        if not ok and not STATE["terminal"]:
            try:
                res = move_to_pose(Pose(hover, quat, BASE["frame"]), tolerance=CFG["move_tol"], max_steps=200,
                                   strategy=MotionStrategy(trajectory_planner="curobo"))
                MOVES.append({"tag": "hover_first_curobo", "ok": bool(res.ok), "steps": int(res.steps_executed)})
                STATE["ticks"] += int(res.steps_executed)
                if read_terminal(res):
                    STATE["terminal"] = True
            except Exception as err:
                note_error("hover_first_curobo", err)
        # ---- first touch: descend in bounded steps until the disc registers
        found = False
        rec = {}
        h = hs[0]
        for i in range(len(hs)):
            if STATE["terminal"] or budget_low():
                break
            h = hs[i]
            if not fb_ok(0, h):
                CONTACT["tried"].append({"h": r4(h), "skipped": "fallback floor"})
                break
            move_ee(touch_point(0, h), quat, CFG["touch_tol"], CFG["press_max_steps"], "search_0_" + str(i), MOVES)
            ok, rec = wait_register(TGT["cells"][0], CFG["reg_dwell_max"], MOVES)
            CONTACT["tried"].append({"h": r4(h), "ok": bool(ok), "polls": rec.get("polls"),
                                     "front": rec.get("front"), "wrist": rec.get("wrist")})
            if ok:
                found = True
                break
        CONTACT["found"] = found
        CONTACT["h"] = h if found else 0.0
        # later touches wait for the observed registration lag plus a margin (bounded)
        later_dwell = CFG["reg_dwell_max"]
        if found and rec.get("polls") is not None:
            later_dwell = min(CFG["reg_dwell_max"], int(rec["polls"]) + CFG["later_dwell_extra"])
        CONTACT["later_dwell"] = later_dwell
        if not found:
            report["warnings"].append("first touch never registered during the height search")
        note_touch(0, found, rec, h)
        h_use = CONTACT["h"] - CFG["reg_margin_m"] if found else CONTACT["h"]
        h_use = max(h_use, h_floor())
        # ---- later touches: lift, transit, descend to the registered height, poll, bounded retries
        for k in range(1, len(pts)):
            if STATE["terminal"] or budget_low():
                break
            cur = ee_now()[0]
            up_z = touch_point(k - 1, h_use)[2] + CFG["transit_lift_m"]
            up0 = [cur[0], cur[1], max(cur[2], up_z)]
            move_ee(up0, quat, CFG["move_tol"], 40, "lift_" + str(k), MOVES)
            up1 = touch_point(k, h_use + CFG["transit_lift_m"])
            chunked_move(up0, up1, quat, CFG["lift_chunk_m"], CFG["move_tol"], "transit_" + str(k), MOVES)
            hk = h_use
            move_ee(touch_point(k, hk), quat, CFG["touch_tol"], CFG["descend_max_steps"], "touch_" + str(k), MOVES)
            ok, rec = wait_register(TGT["cells"][k], later_dwell, MOVES)
            tries = 0
            while not ok and tries < CFG["later_retry"] and not STATE["terminal"] and not budget_low():
                nh = hk - CFG["cal_step_m"]
                if nh < h_floor() or not fb_ok(k, nh):
                    break
                hk = nh
                tries += 1
                move_ee(touch_point(k, hk), quat, CFG["touch_tol"], CFG["press_max_steps"], "retry_" + str(k) + "_" + str(tries), MOVES)
                ok, rec = wait_register(TGT["cells"][k], CFG["reg_dwell_max"], MOVES)
            rec["retries"] = tries
            note_touch(k, ok, rec, hk)
            if ok and hk < h_use:
                h_use = hk
        if not STATE["terminal"]:
            try:
                cur = ee_now()[0]
                move_ee([cur[0], cur[1], cur[2] + CFG["hover_m"]], quat, CFG["move_tol"], 60, "retreat", MOVES)
                dwell(3, MOVES)
            except Exception as err:
                note_error("retreat", err)
except Exception as err:
    note_error("phase6.execute", err)

report["ticks_counted"] = STATE["ticks"]
report["terminal_seen"] = STATE["terminal"]
report["n_moves"] = STATE["n_moves"]
try:
    if not STATE["terminal"]:
        report["end_t"] = r4(get_robot_state().timestamp_s)
except Exception as err:
    note_error("final_state", err)
result = report
