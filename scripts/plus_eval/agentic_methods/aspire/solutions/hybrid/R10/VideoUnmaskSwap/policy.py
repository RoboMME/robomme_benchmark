import math

# VideoUnmaskSwap v00 -- track every container through the conditioning video (RGB-D, both
# cameras fused top-down in the robot base frame), keep tracking live until the shuffle stops,
# then lift the containers hiding the goal's colours in goal order.
# Every geometric quantity (table height, container height/size, camera convention, cube
# colours/positions, container positions) is derived at runtime from public observations.

report = {"version": "v02"}
STATE = {"ticks": 0, "ended": False}

COLOR_WORDS = ["red", "green", "blue", "yellow", "orange", "purple", "black", "white",
               "pink", "brown", "gray", "grey", "cyan"]


def rows_of(arr):
    try:
        return arr.tolist()
    except Exception:
        return [list(row) for row in arr]


def parse_goal(text):
    t = text.lower()
    for ch in ",.;:!":
        t = t.replace(ch, " ")
    words = t.split()
    start = 0
    if "then" in words:
        start = words.index("then") + 1
    seq = []
    for i in range(start, len(words)):
        w = words[i]
        if w in COLOR_WORDS:
            nxt = words[i + 1] if i + 1 < len(words) else ""
            if w == "grey":
                w = "gray"
            seq.append({"color": w, "all": nxt == "cubes"})
    return seq


def rgb_hsv(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    v = mx / 255.0
    s = 0.0 if mx <= 0 else (mx - mn) / mx
    if mx == mn:
        h = 0.0
    elif mx == r:
        h = (60.0 * (g - b) / (mx - mn)) % 360.0
    elif mx == g:
        h = 60.0 * (b - r) / (mx - mn) + 120.0
    else:
        h = 60.0 * (r - g) / (mx - mn) + 240.0
    return h, s, v


def classify(r, g, b):
    h, s, v = rgb_hsv(r, g, b)
    if v < 0.22:
        return "black"
    if s < 0.25:
        return "white" if v > 0.65 else "gray"
    if h < 15.0 or h >= 330.0:
        return "red"
    if h < 42.0:
        return "orange"
    if h < 75.0:
        return "yellow"
    if h < 165.0:
        return "green"
    if h < 195.0:
        return "cyan"
    if h < 255.0:
        return "blue"
    return "purple"


PROTO_HUE = {"red": 0.0, "orange": 30.0, "yellow": 58.0, "green": 120.0, "cyan": 180.0,
             "blue": 230.0, "purple": 285.0, "pink": 320.0}


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def qmat(q):
    w, x, y, z = [float(c) for c in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


CONVS = {
    "cv": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "gl": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
    "sapien": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
}

SETUP_CACHE = {}


def cam_setup(cam, conv, ranged, stride, roi):
    pose = cam.camera_pose
    t = [float(c) for c in pose.position]
    q = [float(c) for c in pose.quaternion_wxyz]
    K = rows_of(cam.intrinsics)
    fx = float(K[0][0])
    fy = float(K[1][1])
    cx = float(K[0][2])
    cy = float(K[1][2])
    key = str((conv, ranged, stride, roi, [round(c, 5) for c in t], [round(c, 5) for c in q],
               round(fx, 3), round(cx, 3)))
    hit = SETUP_CACHE.get(key)
    if hit is not None:
        return hit
    A = matmul(qmat(q), CONVS[conv])
    u0, v0, u1, v1 = roi
    rows = []
    for v in range(v0, v1, stride):
        yn = (v - cy) / fy
        ulist = []
        for u in range(u0, u1, stride):
            xn = (u - cx) / fx
            sc = 1.0
            if ranged:
                sc = 1.0 / math.sqrt(xn * xn + yn * yn + 1.0)
            ax = (A[0][0] * xn + A[0][1] * yn + A[0][2]) * sc
            ay = (A[1][0] * xn + A[1][1] * yn + A[1][2]) * sc
            az = (A[2][0] * xn + A[2][1] * yn + A[2][2]) * sc
            ulist.append((u, ax, ay, az))
        rows.append((v, ulist))
    out = {"rows": rows, "t": t}
    if len(SETUP_CACHE) > 40:
        SETUP_CACHE.clear()
    SETUP_CACHE[key] = out
    return out


def project(depth_rows, setup, zlo, zhi, rgb_rows=None):
    tx, ty, tz = setup["t"]
    pts = []
    for v, ulist in setup["rows"]:
        drow = depth_rows[v]
        crow = rgb_rows[v] if rgb_rows is not None else None
        for u, ax, ay, az in ulist:
            d = drow[u]
            if not (d > 0.0) or d > 8.0:
                continue
            z = d * az + tz
            if z < zlo or z > zhi:
                continue
            if crow is not None:
                c = crow[u]
                pts.append((d * ax + tx, d * ay + ty, z, u, v, c[0], c[1], c[2]))
            else:
                pts.append((d * ax + tx, d * ay + ty, z, u, v))
    return pts


def densest(vals, width):
    vals = sorted(vals)
    best = 0
    bi = 0
    j = 0
    for i in range(len(vals)):
        while vals[i] - vals[j] > width:
            j += 1
        if i - j + 1 > best:
            best = i - j + 1
            bi = j
    seg = vals[bi:bi + best]
    return sum(seg) / max(1, len(seg)), best


def pct(vals, f):
    s = sorted(vals)
    if not s:
        return 0.0
    return s[min(len(s) - 1, max(0, int(round((len(s) - 1) * f))))]


def main():
    # ---------------------------------------------------------------- task context / goal
    goal = get_task_context().language
    report["goal"] = goal
    targets_spec = parse_goal(goal)
    report["parsed"] = targets_spec
    try:
        meta = robomme_plus.get_task_metadata()
        try:
            report["max_steps"] = int(meta["max_steps"])
        except Exception:
            report["max_steps"] = None
        try:
            report["sampling"] = str(meta["conditioning_video_sampling"])[:200]
        except Exception:
            pass
    except Exception as exc:
        report["meta_error"] = str(exc)[:200]

    state0 = get_robot_state()
    base = state0.base_frame
    emb = state0.embodiment
    home_q = [float(c) for c in state0.end_effector_poses["primary"].quaternion_wxyz]
    home_p = [float(c) for c in state0.end_effector_poses["primary"].position]
    report["home_p"] = [round(c, 3) for c in home_p]

    live0 = get_observation()
    cams = [n for n in ("front", "wrist") if n in live0.cameras]
    report["cams"] = cams

    # ---------------------------------------------------------------- camera convention + table
    conv_scores = []
    for conv in ("cv", "gl", "sapien"):
        for ranged in (False, True):
            total = 0.0
            zs_front = None
            okc = True
            for cname in cams:
                cam = live0.cameras[cname]
                dr = rows_of(cam.depth_m)
                H = len(dr)
                W = len(dr[0])
                st = cam_setup(cam, conv, ranged, 8, (0, 0, W, H))
                pts = project(dr, st, -100.0, 100.0)
                zs = [p[2] for p in pts]
                if len(zs) < 20:
                    okc = False
                    break
                zc, cnt = densest(zs, 0.006)
                frac = cnt / float(len(zs))
                if st["t"][2] < zc:
                    okc = False
                total += 1.0 - frac
                if cname == "front":
                    zs_front = zc
            if okc:
                conv_scores.append((total, conv, ranged, zs_front))
    conv_scores.sort()
    report["conv_scores"] = [(round(s[0], 3), s[1], s[2]) for s in conv_scores[:4]]
    if not conv_scores:
        raise ValueError("no camera convention gives a table plane below the camera")
    CONV = conv_scores[0][1]
    RANGED = conv_scores[0][2]

    # table height from the live front camera at a finer stride
    cam = live0.cameras[cams[0]]
    dr = rows_of(cam.depth_m)
    st = cam_setup(cam, CONV, RANGED, 4, (0, 0, len(dr[0]), len(dr)))
    TABLE_Z, tcount = densest([p[2] for p in project(dr, st, -100.0, 100.0)], 0.004)
    report["table_z"] = round(TABLE_Z, 4)

    BAND_LO = 0.006
    BAND_HI = 0.20


    def ee_z_of(obs):
        try:
            return float(obs.robot_state.end_effector_poses["primary"].position[2])
        except Exception:
            return 99.0


    def frame_points(obs, strides, rois, want_rgb=False, hz_lo=BAND_LO):
        """Band points (above-table, below gripper) fused from both cameras."""
        zhi = min(TABLE_Z + BAND_HI, ee_z_of(obs) - 0.02)
        out = []
        for cname in cams:
            cam = obs.cameras[cname]
            dr = rows_of(cam.depth_m)
            roi = rois.get(cname)
            if roi is None:
                roi = (0, 0, len(dr[0]), len(dr))
            st = cam_setup(cam, CONV, RANGED, strides[cname], roi)
            rr = rows_of(cam.rgb) if want_rgb else None
            pts = project(dr, st, TABLE_Z + hz_lo, zhi, rr)
            for p in pts:
                out.append((cname,) + p)
        return out


    def ckey(x, y, cs):
        return (int(math.floor(x / cs)), int(math.floor(y / cs)))


    # ---------------------------------------------------------------- conditioning video
    video = robomme_plus.get_conditioning_video()
    NV = len(video)
    report["video_frames"] = NV
    if NV < 10:
        raise ValueError("conditioning video too short for tracking")

    coarse_step = max(1, NV // 48)
    coarse_idx = list(range(0, NV, coarse_step))
    if coarse_idx[-1] != NV - 1:
        coarse_idx.append(NV - 1)
    full_rois = {}
    coarse = []
    cell_count = {}
    for fi in coarse_idx:
        pts = frame_points(video[fi], {"front": 4, "wrist": 4}, full_rois)
        cells = set()
        for p in pts:
            cells.add(ckey(p[1], p[2], 0.01))
        for k in cells:
            cell_count[k] = cell_count.get(k, 0) + 1
        coarse.append((fi, pts))
    bg = set()
    for k, c in cell_count.items():
        if c >= max(0.9 * len(coarse_idx), len(coarse_idx) - 1):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    bg.add((k[0] + dx, k[1] + dy))
    report["bg_cells"] = len(bg)


    def nonbg(pts):
        return [p for p in pts if ckey(p[1], p[2], 0.01) not in bg]


    h95s = []
    roi_box = {}
    for fi, pts in coarse:
        nb = nonbg(pts)
        hs = [p[3] - TABLE_Z for p in nb]
        h95s.append((fi, pct(hs, 0.95) if len(hs) >= 10 else 0.0, len(hs)))
        for p in nb:
            b = roi_box.get(p[0])
            u = p[4]
            v = p[5]
            if b is None:
                roi_box[p[0]] = [u, v, u, v]
            else:
                if u < b[0]:
                    b[0] = u
                if v < b[1]:
                    b[1] = v
                if u > b[2]:
                    b[2] = u
                if v > b[3]:
                    b[3] = v
    late = [h for fi, h, n in h95s if fi >= NV // 2 and n >= 10]
    HC = pct(late, 0.5) if late else 0.05
    report["container_h"] = round(HC, 4)
    report["h95_coarse"] = [(fi, round(h, 3), n) for fi, h, n in h95s[:14]]

    vid_rois = {}
    for cname in cams:
        b = roi_box.get(cname)
        cam = video[0].cameras[cname]
        H = len(rows_of(cam.depth_m))
        W = H
        try:
            W = int(cam.rgb.shape[1])
        except Exception:
            pass
        if b is None:
            vid_rois[cname] = (0, 0, W, H)
        else:
            vid_rois[cname] = (max(0, b[0] - 16), max(0, b[1] - 16), min(W, b[2] + 17), min(H, b[3] + 17))
    report["vid_rois"] = vid_rois

    # first covered frame
    FC = None
    prev_c = 0
    for fi, h, n in h95s:
        if h > 0.75 * HC and n >= 20:
            FC = fi
            break
        prev_c = fi
    if FC is None:
        raise ValueError("cover time not found")
    for fi in range(prev_c, FC + 1):
        nb = nonbg(frame_points(video[fi], {"front": 2, "wrist": 3}, vid_rois))
        hs = [p[3] - TABLE_Z for p in nb]
        if len(hs) >= 20 and pct(hs, 0.95) > 0.75 * HC:
            FC = fi
            break
    report["cover_frame"] = FC


    def grid_clusters(pts, cs, min_pts):
        cellmap = {}
        for i in range(len(pts)):
            k = ckey(pts[i][1], pts[i][2], cs)
            cellmap.setdefault(k, []).append(i)
        seen = set()
        clusters = []
        for k in cellmap:
            if k in seen:
                continue
            stack = [k]
            seen.add(k)
            members = []
            while stack:
                c = stack.pop()
                members.extend(cellmap[c])
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nk = (c[0] + dx, c[1] + dy)
                        if nk in cellmap and nk not in seen:
                            seen.add(nk)
                            stack.append(nk)
            if len(members) >= min_pts:
                clusters.append([pts[i] for i in members])
        return clusters


    # cubes: the last uncovered frame with low objects only
    cubes = []
    cube_frame = None
    for fi in range(FC - 1, max(-1, FC - 12), -1):
        nb = nonbg(frame_points(video[fi], {"front": 1, "wrist": 2}, vid_rois, want_rgb=True,
                                hz_lo=0.004))
        if not nb:
            continue
        hs = [p[3] - TABLE_Z for p in nb]
        if pct(hs, 0.95) > 0.75 * HC:
            continue
        cl = grid_clusters(nb, 0.01, 6)
        if cl:
            cube_frame = fi
            for c in cl:
                hs2 = sorted(p[3] for p in c)
                ztop = hs2[len(hs2) // 2]
                up = [p for p in c if p[3] >= ztop]
                r = sum(p[6] for p in up) / len(up)
                g = sum(p[7] for p in up) / len(up)
                b = sum(p[8] for p in up) / len(up)
                cubes.append({"x": sum(p[1] for p in c) / len(c), "y": sum(p[2] for p in c) / len(c),
                              "n": len(c), "rgb": [int(r), int(g), int(b)],
                              "color": classify(r, g, b), "h": round(hs2[-1] - TABLE_Z, 3)})
            break
    report["cube_frame"] = cube_frame
    report["cubes"] = [(c["color"], round(c["x"], 3), round(c["y"], 3), c["n"], c["rgb"], c["h"])
                       for c in cubes]
    if not cubes:
        raise ValueError("no cubes found before cover")

    # ---------------------------------------------------------------- tracking
    UP_FRAC = 0.8


    def upper_cells(pts, frac=UP_FRAC):
        thr = TABLE_Z + frac * HC
        cells = {}
        for p in pts:
            if p[3] < thr:
                continue
            k = ckey(p[1], p[2], 0.005)
            if k in cells:
                continue
            if ckey(p[1], p[2], 0.01) in bg:
                continue
            cells[k] = ((k[0] + 0.5) * 0.005, (k[1] + 0.5) * 0.005)
        return list(cells.values())


    def kmeans(centers, cells, rg, iters):
        n = len(centers)
        cur = [list(c) for c in centers]
        counts = [0] * n
        rg2 = rg * rg
        for it in range(iters):
            sx = [0.0] * n
            sy = [0.0] * n
            sw = [0] * n
            for x, y in cells:
                best = -1
                bd = rg2
                for j in range(n):
                    dx = x - cur[j][0]
                    dy = y - cur[j][1]
                    dd = dx * dx + dy * dy
                    if dd < bd:
                        bd = dd
                        best = j
                if best >= 0:
                    sx[best] += x
                    sy[best] += y
                    sw[best] += 1
            for j in range(n):
                if sw[j] >= 3:
                    cur[j] = [sx[j] / sw[j], sy[j] / sw[j]]
            counts = sw
        return cur, counts


    init_pts = nonbg(frame_points(video[FC], {"front": 2, "wrist": 3}, vid_rois))
    cells0 = upper_cells(init_pts)
    seeds = [[c["x"], c["y"]] for c in cubes]
    cent, cnts = kmeans(seeds, cells0, 0.05, 3)
    # extra (empty) containers: unclaimed upper cells far from every seed
    unclaimed = []
    for x, y in cells0:
        dmin = min(math.hypot(x - c[0], y - c[1]) for c in cent)
        if dmin > 0.04:
            unclaimed.append(("u", x, y, 0.0))
    extra = grid_clusters(unclaimed, 0.01, 8)
    for c in extra:
        cent.append([sum(p[1] for p in c) / len(c), sum(p[2] for p in c) / len(c)])
    cent, cnts = kmeans(cent, cells0, 0.05, 3)
    # container radius from assignment spread
    spreads = []
    for j in range(len(cent)):
        ds = []
        for x, y in cells0:
            dmin = None
            jb = -1
            for k in range(len(cent)):
                d = math.hypot(x - cent[k][0], y - cent[k][1])
                if dmin is None or d < dmin:
                    dmin = d
                    jb = k
            if jb == j and dmin < 0.05:
                ds.append(dmin)
        if len(ds) >= 5:
            spreads.append(pct(ds, 0.9))
    R90 = pct(spreads, 0.5) if spreads else 0.025
    RG = min(0.06, max(0.02, 1.4 * R90))
    DUP = R90
    WS_PAD = 0.10
    REACQ_MAX = 0.25
    report["r90"] = round(R90, 4)
    report["rg"] = round(RG, 4)

    tracks = []
    for j in range(len(cent)):
        cube = cubes[j] if j < len(cubes) else None
        tracks.append({"x": cent[j][0], "y": cent[j][1], "vx": 0.0, "vy": 0.0,
                       "color": cube["color"] if cube else "none", "miss": 0, "maxmiss": 0,
                       "cube_rgb": cube["rgb"] if cube else None})
    report["init_tracks"] = [(t["color"], round(t["x"], 3), round(t["y"], 3), cnts[j])
                             for j, t in enumerate(tracks)]
    stats = {"min_sep": 9.0, "min_sep_frame": -1, "max_unclaimed": 0, "unclaimed_frame": -1}


    TYP = {"n": float(pct(cnts, 0.5)) if cnts else 20.0}
    EVENTS = []
    report["reacq"] = EVENTS
    for t in tracks:
        t["weak"] = 0

    def track_step(cells, dt, tag):
        preds = [[t["x"] + t["vx"] * dt, t["y"] + t["vy"] * dt] for t in tracks]
        cur, cnt = kmeans(preds, cells, RG, 3)
        healthy = [c for c in cnt if c >= 3]
        if len(healthy) >= max(1, len(tracks) // 2):
            TYP["n"] = 0.9 * TYP["n"] + 0.1 * pct(healthy, 0.75)
        for j, t in enumerate(tracks):
            if cnt[j] >= 3:
                nx, ny = cur[j]
                if dt > 1e-6:
                    t["vx"] = 0.6 * t["vx"] + 0.4 * (nx - t["x"]) / dt
                    t["vy"] = 0.6 * t["vy"] + 0.4 * (ny - t["y"]) / dt
                t["x"] = nx
                t["y"] = ny
                t["miss"] = 0
            else:
                t["x"] = preds[j][0]
                t["y"] = preds[j][1]
                t["vx"] *= 0.7
                t["vy"] *= 0.7
                t["miss"] += 1
                if t["miss"] > t["maxmiss"]:
                    t["maxmiss"] = t["miss"]
            if cnt[j] < 0.35 * TYP["n"]:
                t["weak"] += 1
            else:
                t["weak"] = 0
        n = len(tracks)
        for a in range(n):
            for b in range(a + 1, n):
                d = math.hypot(tracks[a]["x"] - tracks[b]["x"], tracks[a]["y"] - tracks[b]["y"])
                if d < stats["min_sep"]:
                    stats["min_sep"] = d
                    stats["min_sep_frame"] = tag
                if d < DUP:
                    loser = a if cnt[a] < cnt[b] else b
                    tracks[loser]["weak"] = max(tracks[loser]["weak"], 2)
        orphans = []
        for x, y in cells:
            ok = False
            for t in tracks:
                if (x - t["x"]) ** 2 + (y - t["y"]) ** 2 < RG * RG:
                    ok = True
                    break
            if not ok:
                orphans.append(("o", x, y, 0.0))
        un = len(orphans)
        if un > stats["max_unclaimed"]:
            stats["max_unclaimed"] = un
            stats["unclaimed_frame"] = tag
        if un >= max(6, 0.35 * TYP["n"]):
            clusters = grid_clusters(orphans, 0.01, max(6, int(0.35 * TYP["n"])))
            clusters.sort(key=len, reverse=True)
            wx0 = min(t["x"] for t in tracks) - WS_PAD
            wx1 = max(t["x"] for t in tracks) + WS_PAD
            wy0 = min(t["y"] for t in tracks) - WS_PAD
            wy1 = max(t["y"] for t in tracks) + WS_PAD
            for c in clusters:
                ox = sum(p[1] for p in c) / len(c)
                oy = sum(p[2] for p in c) / len(c)
                if ox < wx0 or ox > wx1 or oy < wy0 or oy > wy1:
                    continue
                best = None
                for j, t in enumerate(tracks):
                    if t["weak"] < 2:
                        continue
                    d = math.hypot(t["x"] - ox, t["y"] - oy)
                    if d > REACQ_MAX:
                        continue
                    if best is None or d < best[0]:
                        best = (d, j)
                if best is None:
                    if len(EVENTS) < 60:
                        EVENTS.append((str(tag), "orphan", round(ox, 3), round(oy, 3), len(c)))
                    continue
                t = tracks[best[1]]
                if len(EVENTS) < 60:
                    EVENTS.append((str(tag), best[1], round(best[0], 3), round(ox, 3), round(oy, 3),
                                   len(c)))
                t["x"] = ox
                t["y"] = oy
                t["vx"] = 0.0
                t["vy"] = 0.0
                t["weak"] = 0
                t["miss"] = 0


    TRACE = []
    report["trace"] = TRACE
    prev_t = float(video[FC].timestamp_s)
    for fi in range(FC + 1, NV):
        fr = video[fi]
        ts = float(fr.timestamp_s)
        cells = upper_cells(frame_points(fr, {"front": 1, "wrist": 2}, vid_rois))
        track_step(cells, ts - prev_t, fi)
        prev_t = ts
        if fi % 10 == 0:
            TRACE.append((fi, round(ts, 2), [(round(t["x"], 3), round(t["y"], 3)) for t in tracks]))
    report["video_track_end"] = [(t["color"], round(t["x"], 3), round(t["y"], 3), t["maxmiss"])
                                 for t in tracks]

    # ---------------------------------------------------------------- live tracking until still


    def hold_tick(grip):
        s = get_robot_state()
        sr = step(RobotAction(arms={"primary": ArmCommand(
            "joint_position", s.joint_positions["primary"], gripper_position=grip,
            embodiment=s.embodiment)}))
        STATE["ticks"] += 1
        okv = True
        try:
            okv = bool(sr.ok)
        except Exception:
            pass
        try:
            if sr.terminated or sr.truncated:
                STATE["ended"] = True
        except Exception:
            pass
        return okv


    live_rois = {"wrist": None}
    fr0 = nonbg(frame_points(live0, {"front": 2, "wrist": 3}, {}))
    b = None
    for p in fr0:
        if p[0] != "front":
            continue
        if b is None:
            b = [p[4], p[5], p[4], p[5]]
        else:
            b = [min(b[0], p[4]), min(b[1], p[5]), max(b[2], p[4]), max(b[3], p[5])]
    if b is not None:
        fw = len(rows_of(live0.cameras["front"].depth_m)[0])
        fh = len(rows_of(live0.cameras["front"].depth_m))
        live_rois["front"] = (max(0, b[0] - 20), max(0, b[1] - 20), min(fw, b[2] + 21),
                              min(fh, b[3] + 21))
    LIVE_ROIS = {}
    for k, v in live_rois.items():
        if v is not None:
            LIVE_ROIS[k] = v
    track_step(upper_cells(fr0), 0.05, "live0")
    hist = [[(t["x"], t["y"]) for t in tracks]]
    wait_ticks = 0
    step_fail = 0
    for k in range(300):
        if not hold_tick(1.0):
            step_fail += 1
            if step_fail > 3:
                break
        if STATE["ended"]:
            break
        wait_ticks += 1
        obs = get_observation()
        track_step(upper_cells(frame_points(obs, {"front": 2, "wrist": 3}, LIVE_ROIS)), 0.05,
                   "L" + str(k))
        hist.append([(t["x"], t["y"]) for t in tracks])
        if len(hist) > 12:
            old = hist[-11]
            mv = max(math.hypot(hist[-1][j][0] - old[j][0], hist[-1][j][1] - old[j][1])
                     for j in range(len(tracks)))
            if mv < 0.004:
                break
    report["wait_ticks"] = wait_ticks
    report["step_fail"] = step_fail
    report["live_tracks"] = [(t["color"], round(t["x"], 3), round(t["y"], 3), t["maxmiss"])
                             for t in tracks]
    report["track_stats"] = {"min_sep": round(stats["min_sep"], 4),
                             "min_sep_frame": stats["min_sep_frame"],
                             "max_unclaimed": stats["max_unclaimed"],
                             "unclaimed_frame": stats["unclaimed_frame"]}

    # ---------------------------------------------------------------- target order
    order = []
    used = set()
    for spec in targets_spec:
        idx = [j for j, t in enumerate(tracks) if t["color"] == spec["color"] and j not in used]
        if not idx:
            # nearest-hue fallback among saturated, unused cube tracks
            ph = PROTO_HUE.get(spec["color"])
            best = None
            for j, t in enumerate(tracks):
                if j in used or t["cube_rgb"] is None or ph is None:
                    continue
                h, s, v = rgb_hsv(t["cube_rgb"][0], t["cube_rgb"][1], t["cube_rgb"][2])
                if s < 0.3:
                    continue
                dh = abs(((h - ph + 180.0) % 360.0) - 180.0)
                if best is None or dh < best[0]:
                    best = (dh, j)
            if best is not None:
                idx = [best[1]]
                report.setdefault("fallback", []).append((spec["color"], best[1], round(best[0], 1)))
        if not spec["all"]:
            idx = idx[:1]
        for j in idx:
            used.add(j)
            order.append(j)
    report["order"] = [(j, tracks[j]["color"]) for j in order]

    # ---------------------------------------------------------------- picking


    def rect_fit(xy):
        best = None
        for k in range(30):
            a = math.radians(3.0 * k)
            c = math.cos(a)
            s = math.sin(a)
            us = [p[0] * c + p[1] * s for p in xy]
            vs = [-p[0] * s + p[1] * c for p in xy]
            ulo = pct(us, 0.03)
            uhi = pct(us, 0.97)
            vlo = pct(vs, 0.03)
            vhi = pct(vs, 0.97)
            area = (uhi - ulo) * (vhi - vlo)
            if best is None or area < best[0]:
                uc = 0.5 * (ulo + uhi)
                vc = 0.5 * (vlo + vhi)
                best = (area, a, uc * c - vc * s, uc * s + vc * c, uhi - ulo, vhi - vlo)
        return best


    def local_points(j):
        obs = get_observation()
        pts = nonbg(frame_points(obs, {"front": 2, "wrist": 2}, {}))
        cells = upper_cells(pts)
        centers = [[t["x"], t["y"]] for t in tracks]
        cur, cnt = kmeans(centers, cells, RG, 2)
        mine = []
        for x, y in cells:
            best = -1
            bd = RG * RG
            for k in range(len(cur)):
                dd = (x - cur[k][0]) ** 2 + (y - cur[k][1]) ** 2
                if dd < bd:
                    bd = dd
                    best = k
            if best == j:
                mine.append((x, y))
        tops = []
        for x, y in upper_cells(pts, 0.8):
            if (x - cur[j][0]) ** 2 + (y - cur[j][1]) ** 2 < RG * RG:
                dself = (x - cur[j][0]) ** 2 + (y - cur[j][1]) ** 2
                if all(dself <= (x - cur[k][0]) ** 2 + (y - cur[k][1]) ** 2
                       for k in range(len(cur))):
                    tops.append((x, y))
        zs = [p[3] for p in pts if (p[1] - cur[j][0]) ** 2 + (p[2] - cur[j][1]) ** 2 < R90 * R90]
        return cur, mine, zs, tops


    def wrap90(d):
        while d > math.pi / 2:
            d -= math.pi
        while d <= -math.pi / 2:
            d += math.pi
        return d


    def goto(p, q, tol, steps):
        r = move_to_pose(Pose([float(c) for c in p], q, base), tolerance=tol, max_steps=steps)
        n = 0
        try:
            n = int(r.steps_executed)
        except Exception:
            pass
        STATE["ticks"] += n
        try:
            if r.terminated or r.truncated:
                STATE["ended"] = True
        except Exception:
            pass
        okv = False
        try:
            okv = bool(r.ok)
        except Exception:
            pass
        return okv, n


    R0 = qmat(home_q)
    yaw0 = math.atan2(R0[1][1], R0[0][1])
    picks = []
    for j in order:
        if STATE["ended"]:
            break
        t = tracks[j]
        info = {"track": j, "color": t["color"]}
        cur, mine, zs, tops = local_points(j)
        cx = cur[j][0]
        cy = cur[j][1]
        top = pct(zs, 0.95) if len(zs) >= 5 else TABLE_Z + HC
        pre_z = top + 0.10
        ok1, n1 = goto((cx, cy, pre_z), home_q, 0.015, 200)
        # refine from close range
        cur, mine, zs, tops = local_points(j)
        info["n_mine"] = len(mine)
        info["n_tops"] = len(tops)
        fit = rect_fit(tops) if len(tops) >= 8 else (rect_fit(mine) if len(mine) >= 8 else None)
        if fit is not None:
            cx, cy = fit[2], fit[3]
            ang = fit[1]
            info["fit"] = [round(fit[4], 3), round(fit[5], 3), round(math.degrees(ang), 1)]
        else:
            cx = cur[j][0]
            cy = cur[j][1]
            ang = yaw0
        if len(zs) >= 5:
            top = pct(zs, 0.95)
        options = []
        for b2 in (ang, ang + math.pi / 2):
            d = wrap90(b2 - yaw0)
            block = 0
            for k, o in enumerate(cur):
                if k == j:
                    continue
                dx = o[0] - cx
                dy = o[1] - cy
                along = abs(dx * math.cos(b2) + dy * math.sin(b2))
                across = abs(-dx * math.sin(b2) + dy * math.cos(b2))
                if across < 1.6 * R90 and along < 4.0 * R90:
                    block += 1
            options.append((block, abs(d), d))
        options.sort()
        delta = options[0][2]
        q = qmul([math.cos(delta / 2), 0.0, 0.0, math.sin(delta / 2)], home_q)
        grasp_z = max(TABLE_Z + 0.012, top - 0.45 * HC)
        info["xy"] = [round(cx, 3), round(cy, 3)]
        info["top_h"] = round(top - TABLE_Z, 3)
        info["delta_deg"] = round(math.degrees(delta), 1)
        info["block"] = options[0][0]
        ok2, n2 = goto((cx, cy, pre_z), q, 0.01, 150)
        ok3, n3 = goto((cx, cy, grasp_z), q, 0.008, 150)
        rc = close_gripper()
        try:
            STATE["ticks"] += int(rc.steps_executed)
        except Exception:
            pass
        grip = float(get_robot_state().gripper_positions["primary"])
        info["grip"] = round(grip, 3)
        if grip < 0.03 and not STATE["ended"]:
            open_gripper()
            goto((cx, cy, pre_z), q, 0.015, 150)
            cur, mine, zs, tops = local_points(j)
            fit = rect_fit(tops) if len(tops) >= 8 else None
            if fit is not None:
                cx, cy = fit[2], fit[3]
            info["retry_xy"] = [round(cx, 3), round(cy, 3)]
            goto((cx, cy, grasp_z), q, 0.008, 150)
            close_gripper()
            grip = float(get_robot_state().gripper_positions["primary"])
            info["grip_retry"] = round(grip, 3)
        ok4, n4 = goto((cx, cy, grasp_z + 0.12), q, 0.015, 150)
        info["moves"] = [ok1, ok2, ok3, ok4]
        info["ticks"] = [n1, n2, n3, n4]
        if STATE["ended"]:
            picks.append(info)
            break
        ok5, n5 = goto((cx, cy, grasp_z + 0.012), q, 0.015, 150)
        open_gripper()
        ok6, n6 = goto((cx, cy, pre_z), q, 0.02, 150)
        info["ticks"] = info["ticks"] + [n5, n6]
        picks.append(info)
    report["picks"] = picks
    report["ticks_counted"] = STATE["ticks"]
    report["ended"] = STATE["ended"]


try:
    main()
except Exception as exc:
    report["error"] = str(exc)[:300]
result = report
