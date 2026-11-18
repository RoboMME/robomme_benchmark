# VideoUnmaskSwap v00 -- initial program from offline scene analysis (released recordings).
# Strategy (all per-episode quantities perceived at runtime):
#   1. Conditioning video, front camera: find the empty-table frame (video[0]), the last frame
#      before the white containers drop, and the cubes visible in it (difference vs video[0]).
#      Cube colours come from that frame; wrist-camera cubes hidden from the front view
#      (behind an occluder) are added by RGB-D projection when calibration allows.
#   2. Track every container through all retained video frames in front-image space:
#      warm-started nearest-centre assignment of whitish non-static pixels, constant-velocity
#      prediction scaled by frame timestamps, coasting when occluded.
#   3. Live: hold the arm joints tick by tick and keep tracking until the containers settle.
#   4. Parse the goal into an ordered colour list; for each wanted container: project its
#      pixels with mask_to_point_cloud, estimate centre/top/yaw, grasp top-down, lift, put
#      back, release, rise.  Every phase is guarded and logged into `result`.
import math

result = {"version": "v01_cube_cover", "phase": "start", "errors": [], "log": []}

WHITE_MIN = 120          # whitish container pixel: every channel >= this
WHITE_SPREAD = 35        # and max-min channel spread below this
TRACK_RADIUS = 14.0      # px: a pixel joins its nearest predicted track centre within this
MIN_TRACK_PIX = 3        # stride-2 samples needed to update a track (else it coasts)
VIDEO_STRIDE = 2
ROI_MARGIN = 40
SETTLE_MIN_TICKS = 20
SETTLE_MAX_TICKS = 160
SETTLE_WINDOW = 10
SETTLE_PIX = 1.5
ORPHAN_CELL = 6          # px grid for clustering unclaimed container pixels
ORPHAN_MIN = 12          # stride-2 samples for an unclaimed blob to re-acquire a lost track
DOUBLE_PIX = 9.0         # two centres closer than this sit on one container
REACQ_PIX = 70.0         # a lost track may jump at most this far to an unclaimed blob
VEL_KEEP = 0.8           # velocity smoothing (fraction of old velocity kept)
COAST_DECAY = 0.6        # velocity decay per frame while coasting in the open
OCCLUDED_DECAY = 0.97    # velocity decay per frame while coasting behind static structure
LIFT_ABOVE_TOP = 0.10
GRASP_BELOW_TOP = 0.02
HELD_MIN = 0.03          # gripper reading above this after close = something between the fingers
REVEAL_MIN_PIX = 8       # saturated pixels needed to name a revealed cube
MAX_RETRIES = 2
TABLE_HUE_TOL = 6.0      # v01: table-like = within this hue (deg) of the frame's median table hue
TABLE_SAT_TOL = 0.15     # v01: ... and within this saturation of the median table saturation
CUBE_MIN_PIX = 6
CUBE_SPLIT_PIX = 30      # v01: a second colour name needs this many pixels to count as its own cube
CUBE_SAME_PIX = 7.0      # v01: colour-name centroids closer than this belong to one cube
ORANGE_PURE_FRAC = 0.7   # v01: a blob is an orange cube only if mostly orange (table fringe reads orange)
CUBE_FRINGE_PIX = 20     # v01: blobs smaller than this next to a bigger cube are edge fringe

COLOR_WORDS = ["red", "green", "blue", "yellow", "orange", "purple", "black", "white",
               "pink", "brown", "gray", "grey", "cyan"]


def log(msg):
    try:
        if len(result["log"]) < 400:
            result["log"].append(str(msg)[:300])
    except Exception:
        pass


def err(where, exc):
    try:
        result["errors"].append(where + ": " + str(exc)[:400])
    except Exception:
        pass


def rows_of(arr):
    try:
        return arr.tolist()
    except Exception:
        out = []
        for y in range(len(arr)):
            out.append([arr[y][x] for x in range(len(arr[y]))])
        return out


def rgb_rows(obs, cam):
    return rows_of(obs.cameras[cam].rgb)


def is_whitish(p):
    r = p[0]
    g = p[1]
    b = p[2]
    if r < WHITE_MIN or g < WHITE_MIN or b < WHITE_MIN:
        return False
    return max(r, g, b) - min(r, g, b) < WHITE_SPREAD


def hue_of(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    c = mx - mn
    if c <= 0:
        return 0.0
    if mx == r:
        h = (60.0 * (g - b) / c) % 360.0
    elif mx == g:
        h = 60.0 * (b - r) / c + 120.0
    else:
        h = 60.0 * (r - g) / c + 240.0
    return h


def colour_name(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 70:
        return "black"
    if mx - mn < 40:
        if mn > 130:
            return "white"
        return "gray"
    h = hue_of(r, g, b)
    if h < 15 or h >= 330:
        return "red"
    if h < 48:
        return "orange"
    if h < 72:
        return "yellow"
    if h < 170:
        return "green"
    if h < 265:
        return "blue"
    return "purple"


def is_table(p):
    r = p[0]
    g = p[1]
    b = p[2]
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 40 or mx - mn < 35 or mx != r:
        return False
    h = hue_of(r, g, b)
    return 3.0 <= h <= 42.0


def dilate(flags, w, h, rad):
    out = [False] * (w * h)
    for idx in range(w * h):
        if flags[idx]:
            y0 = idx // w
            x0 = idx - y0 * w
            for yy in range(max(0, y0 - rad), min(h, y0 + rad + 1)):
                base = yy * w
                for xx in range(max(0, x0 - rad), min(w, x0 + rad + 1)):
                    out[base + xx] = True
    return out


def components(flags, w, h, minpix):
    seen = [False] * (w * h)
    comps = []
    for idx in range(w * h):
        if flags[idx] and not seen[idx]:
            seen[idx] = True
            stack = [idx]
            pts = []
            while stack:
                cur = stack.pop()
                pts.append(cur)
                cy = cur // w
                cx = cur - cy * w
                for nb in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    ny = nb[0]
                    nx = nb[1]
                    if 0 <= ny < h and 0 <= nx < w:
                        ni = ny * w + nx
                        if flags[ni] and not seen[ni]:
                            seen[ni] = True
                            stack.append(ni)
            if len(pts) >= minpix:
                comps.append(pts)
    return comps


def median(vals):
    s = sorted(vals)
    if not s:
        return 0.0
    return s[len(s) // 2]


def quantile(vals, frac):
    s = sorted(vals)
    if not s:
        return 0.0
    return s[int((len(s) - 1) * frac)]


def detect_cubes(rows, bg, static, w, h):
    """Cubes = pixels differing from the empty-table frame, outside static structure."""
    flags = [False] * (w * h)
    for y in range(h):
        row = rows[y]
        brow = bg[y]
        base = y * w
        for x in range(w):
            p = row[x]
            q = brow[x]
            if abs(p[0] - q[0]) + abs(p[1] - q[1]) + abs(p[2] - q[2]) > 60 and not static[base + x]:
                flags[base + x] = True
    cubes = []
    for pts in components(flags, w, h, 6):
        xs = []
        ys = []
        rs = []
        gs = []
        bs = []
        for idx in pts:
            y = idx // w
            x = idx - y * w
            xs.append(x)
            ys.append(y)
            p = rows[y][x]
            rs.append(p[0])
            gs.append(p[1])
            bs.append(p[2])
        mr = median(rs)
        mg = median(gs)
        mb = median(bs)
        cubes.append({"x": sum(xs) / float(len(xs)), "y": sum(ys) / float(len(ys)), "n": len(pts),
                      "rgb": [int(mr), int(mg), int(mb)], "color": colour_name(mr, mg, mb),
                      "x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys)})
    return cubes


def sat_of(p):
    mx = max(p[0], p[1], p[2])
    if mx <= 0:
        return 0.0
    return (mx - min(p[0], p[1], p[2])) / float(mx)


def table_model(rows, w, h):
    """Median hue/saturation of loosely table-coloured pixels in this frame (perceived, not fixed)."""
    hs = []
    ss = []
    for y in range(0, h, 4):
        row = rows[y]
        for x in range(0, w, 4):
            p = row[x]
            if is_table(p):
                hs.append(hue_of(p[0], p[1], p[2]))
                ss.append(sat_of(p))
    if len(hs) < 20:
        return None
    return (median(hs), median(ss))


def table_like(p, model):
    if model is None:
        return is_table(p)
    if max(p[0], p[1], p[2]) < 60:
        return False
    return abs(hue_of(p[0], p[1], p[2]) - model[0]) < TABLE_HUE_TOL and abs(sat_of(p) - model[1]) < TABLE_SAT_TOL


def detect_cubes_cover(prev, cover, static, w, h):
    """v01 cube detector that needs no empty-table frame.

    The live conditioning video already shows the cubes in frame 0, so v00's difference to frame 0
    found nothing. Instead: a cube pixel is a pixel of the last pre-drop frame that a container
    (non-static white) covers in the first container frame and that is not table-coloured.
    Pixels are grouped by colour name so touching cubes of different colours stay separate.
    """
    model = table_model(prev, w, h)
    names = [None] * (w * h)
    for y in range(h):
        prow = prev[y]
        crow = cover[y]
        base = y * w
        for x in range(w):
            if static[base + x]:
                continue
            c = crow[x]
            if not is_whitish(c):
                continue
            p = prow[x]
            if is_whitish(p):
                if abs(p[0] - c[0]) + abs(p[1] - c[1]) + abs(p[2] - c[2]) > 60:
                    names[base + x] = "white"
                continue
            if table_like(p, model):
                continue
            names[base + x] = colour_name(p[0], p[1], p[2])
    seen = [False] * (w * h)
    cubes = []
    for idx in range(w * h):
        if names[idx] is None or seen[idx]:
            continue
        seen[idx] = True
        stack = [idx]
        pts = []
        while stack:
            cur = stack.pop()
            pts.append(cur)
            cy = cur // w
            cx = cur - cy * w
            for nb in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if 0 <= nb[0] < h and 0 <= nb[1] < w:
                    ni = nb[0] * w + nb[1]
                    if not seen[ni] and names[ni] is not None:
                        seen[ni] = True
                        stack.append(ni)
        if len(pts) < CUBE_MIN_PIX:
            continue
        # one connected blob may hold two touching cubes. Per-pixel names are noisy on shaded
        # faces (yellow->orange, purple/blue->red) and the table-blended fringe reads 'orange', so
        # split only by where the names are: names whose centroids lie apart are separate cubes.
        hist = {}
        for q in pts:
            nm = names[q]
            hist[nm] = hist.get(nm, 0) + 1
        if hist.get("orange", 0) >= ORANGE_PURE_FRAC * len(pts):
            cand = ["orange"]
        else:
            cand = [nm for nm in hist if nm != "orange" and hist[nm] >= max(8, 0.1 * len(pts))]
            if not cand:
                cand = [nm for nm in hist]
        cand = sorted(cand, key=hist.get, reverse=True)
        cent = {}
        for nm in cand:
            sel = [q for q in pts if names[q] == nm]
            cent[nm] = (sum(q - (q // w) * w for q in sel) / float(len(sel)), sum(q // w for q in sel) / float(len(sel)))
        seeds = []
        for nm in cand:
            home = None
            for sd in seeds:
                if math.sqrt((cent[sd[0]][0] - cent[nm][0]) ** 2 + (cent[sd[0]][1] - cent[nm][1]) ** 2) < CUBE_SAME_PIX:
                    home = sd
                    break
            if home is None:
                seeds.append([nm, [nm]])
            else:
                home[1].append(nm)
        groups = []
        if len(seeds) == 1:
            groups.append((seeds[0][0], pts))
        else:
            for sd in seeds:
                gp = [q for q in pts if names[q] in sd[1]]
                if len(gp) >= CUBE_SPLIT_PIX:
                    groups.append((sd[0], gp))
            if not groups:
                groups.append((seeds[0][0], pts))
        for grp in groups:
            gp = grp[1]
            xs = [q - (q // w) * w for q in gp]
            ys = [q // w for q in gp]
            rs = []
            gs = []
            bs = []
            for q in gp:
                p = prev[q // w][q - (q // w) * w]
                rs.append(p[0])
                gs.append(p[1])
                bs.append(p[2])
            cubes.append({"x": sum(xs) / float(len(gp)), "y": sum(ys) / float(len(gp)), "n": len(gp),
                          "blob_n": len(pts), "color": grp[0], "names": hist,
                          "rgb": [int(median(rs)), int(median(gs)), int(median(bs))]})
    keep = []
    for c in cubes:
        fringe = False
        if c["n"] < CUBE_FRINGE_PIX:
            for o in cubes:
                if o["n"] > c["n"] and math.sqrt((o["x"] - c["x"]) ** 2 + (o["y"] - c["y"]) ** 2) < 12.0:
                    fringe = True
        if not fringe:
            keep.append(c)
    return keep, model


def fg_points(rows, static, w, h, roi, stride):
    x0 = max(0, int(roi[0]))
    x1 = min(w - 1, int(roi[1]))
    y0 = max(0, int(roi[2]))
    y1 = min(h - 1, int(roi[3]))
    pts = []
    for y in range(y0, y1 + 1, stride):
        row = rows[y]
        base = y * w
        for x in range(x0, x1 + 1, stride):
            p = row[x]
            if p[2] >= WHITE_MIN and p[1] >= WHITE_MIN and p[0] >= WHITE_MIN:
                if max(p[0], p[1], p[2]) - min(p[0], p[1], p[2]) < WHITE_SPREAD and not static[base + x]:
                    pts.append((x, y))
    return pts


def tracks_roi(tracks, w, h):
    if not tracks:
        return (0, w - 1, 0, h - 1)
    xs = [t["x"] for t in tracks]
    ys = [t["y"] for t in tracks]
    return (max(0, min(xs) - ROI_MARGIN), min(w - 1, max(xs) + ROI_MARGIN),
            max(0, min(ys) - ROI_MARGIN), min(h - 1, max(ys) + ROI_MARGIN))


def assign(pts, cx, cy):
    n = len(cx)
    r2 = TRACK_RADIUS * TRACK_RADIUS
    sx = [0.0] * n
    sy = [0.0] * n
    cnt = [0] * n
    for p in pts:
        best = -1
        bd = r2
        for k in range(n):
            dx = p[0] - cx[k]
            dy = p[1] - cy[k]
            d2 = dx * dx + dy * dy
            if d2 < bd:
                bd = d2
                best = k
        if best >= 0:
            sx[best] += p[0]
            sy[best] += p[1]
            cnt[best] += 1
    return sx, sy, cnt


OCCL = {"mask": None, "reacq": 0, "split": 0}


def in_occluder(x, y, w, h):
    m = OCCL["mask"]
    if m is None:
        return False
    xi = int(x)
    yi = int(y)
    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return False
    return m[yi * w + xi]


def orphan_blobs(pts, cx, cy, w, h):
    """Clusters of foreground samples farther than TRACK_RADIUS from every track centre."""
    r2 = TRACK_RADIUS * TRACK_RADIUS
    cells = {}
    for p in pts:
        far = True
        for k in range(len(cx)):
            dx = p[0] - cx[k]
            dy = p[1] - cy[k]
            if dx * dx + dy * dy < r2:
                far = False
                break
        if far:
            key = (p[0] // ORPHAN_CELL, p[1] // ORPHAN_CELL)
            if key in cells:
                cells[key].append(p)
            else:
                cells[key] = [p]
    blobs = []
    used = set()
    for key in cells:
        if key in used:
            continue
        used.add(key)
        stack = [key]
        members = []
        while stack:
            cur = stack.pop()
            members.extend(cells[cur])
            for dk in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nk = (cur[0] + dk[0], cur[1] + dk[1])
                if nk in cells and nk not in used:
                    used.add(nk)
                    stack.append(nk)
        if len(members) >= ORPHAN_MIN:
            bx = sum(q[0] for q in members) / float(len(members))
            by = sum(q[1] for q in members) / float(len(members))
            blobs.append((bx, by, len(members)))
    return blobs


def track_update(tracks, pts, dt, predict, iters, w, h):
    n = len(tracks)
    px = []
    py = []
    for t in tracks:
        if predict:
            px.append(t["x"] + t["vx"] * dt)
            py.append(t["y"] + t["vy"] * dt)
        else:
            px.append(t["x"])
            py.append(t["y"])
    cx = list(px)
    cy = list(py)
    cnt = [0] * n
    for it in range(iters):
        sx, sy, cnt = assign(pts, cx, cy)
        nx = list(cx)
        ny = list(cy)
        for k in range(n):
            if cnt[k] >= MIN_TRACK_PIX:
                nx[k] = sx[k] / cnt[k]
                ny[k] = sy[k] / cnt[k]
            else:
                nx[k] = px[k]
                ny[k] = py[k]
        cx = nx
        cy = ny
    lost = [cnt[k] < MIN_TRACK_PIX for k in range(n)]
    jumped = [False] * n
    if predict:
        # two centres on one container: the weaker one is treated as lost
        for a in range(n):
            for b in range(a + 1, n):
                if not lost[a] and not lost[b]:
                    if (cx[a] - cx[b]) ** 2 + (cy[a] - cy[b]) ** 2 < DOUBLE_PIX * DOUBLE_PIX:
                        weak = a if cnt[a] < cnt[b] else b
                        lost[weak] = True
                        OCCL["split"] += 1
        if any(lost):
            keep_x = [cx[k] for k in range(n) if not lost[k]]
            keep_y = [cy[k] for k in range(n) if not lost[k]]
            blobs = orphan_blobs(pts, keep_x, keep_y, w, h)
            for bl in blobs:
                best = -1
                bd = REACQ_PIX * REACQ_PIX
                for k in range(n):
                    if lost[k]:
                        d2 = (px[k] - bl[0]) ** 2 + (py[k] - bl[1]) ** 2
                        if d2 < bd:
                            bd = d2
                            best = k
                if best >= 0:
                    cx[best] = bl[0]
                    cy[best] = bl[1]
                    cnt[best] = bl[2]
                    lost[best] = False
                    jumped[best] = True
                    tracks[best]["vx"] = 0.0
                    tracks[best]["vy"] = 0.0
                    tracks[best]["reacq"] = tracks[best].get("reacq", 0) + 1
                    OCCL["reacq"] += 1
            for k in range(n):
                if lost[k] and cnt[k] >= MIN_TRACK_PIX:
                    # doubled centre with no orphan to take: fall back to its own prediction
                    cx[k] = px[k]
                    cy[k] = py[k]
    for k in range(n):
        t = tracks[k]
        nxk = min(w - 1.0, max(0.0, cx[k]))
        nyk = min(h - 1.0, max(0.0, cy[k]))
        if predict and dt > 0:
            if not lost[k]:
                if not jumped[k]:
                    vx = (nxk - t["x"]) / dt
                    vy = (nyk - t["y"]) / dt
                    t["vx"] = VEL_KEEP * t["vx"] + (1.0 - VEL_KEEP) * vx
                    t["vy"] = VEL_KEEP * t["vy"] + (1.0 - VEL_KEEP) * vy
                t["coast"] = 0
            else:
                decay = OCCLUDED_DECAY if in_occluder(nxk, nyk, w, h) else COAST_DECAY
                t["vx"] = decay * t["vx"]
                t["vy"] = decay * t["vy"]
                t["coast"] = t.get("coast", 0) + 1
                t["coast_total"] = t.get("coast_total", 0) + 1
        t["x"] = nxk
        t["y"] = nyk
        t["n"] = cnt[k]


APPROACH = {}


def update_approach(tracks):
    n = len(tracks)
    for a in range(n):
        for b in range(a + 1, n):
            d = math.sqrt((tracks[a]["x"] - tracks[b]["x"]) ** 2 + (tracks[a]["y"] - tracks[b]["y"]) ** 2)
            key = (a, b)
            if key not in APPROACH or d < APPROACH[key]:
                APPROACH[key] = d


def approach(a, b):
    if a > b:
        a, b = b, a
    return APPROACH.get((a, b), 999.0)


def parse_goal(goal):
    """Ordered (colour, all_flag) list from the instruction text."""
    text = goal.lower().replace(",", " , ").replace(".", " . ")
    words = text.split()
    order = []
    clause_all = False
    for i in range(len(words)):
        wd = words[i]
        if wd in (",", ".", "then", "followed", "finally"):
            clause_all = False
        if wd == "all":
            clause_all = True
        if wd in COLOR_WORDS:
            nxt = words[i + 1] if i + 1 < len(words) else ""
            plural = clause_all or nxt == "cubes"
            name = "gray" if wd == "grey" else wd
            order.append({"color": name, "all": plural})
    return order


# ---------------------------------------------------------------- geometry helpers
def quat_to_mat(q):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def cam_point(u, v, d, K, conv):
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    a = (u - cx) / fx * d
    b = (v - cy) / fy * d
    if conv == "opencv":
        return (a, b, d)
    if conv == "opengl":
        return (a, -b, -d)
    return (d, -a, -b)  # ros: x forward, y left, z up


def deproject(u, v, d, cam, conv):
    pc = cam_point(u, v, d, cam["K"], conv)
    R = cam["R"]
    t = cam["t"]
    return (R[0][0] * pc[0] + R[0][1] * pc[1] + R[0][2] * pc[2] + t[0],
            R[1][0] * pc[0] + R[1][1] * pc[1] + R[1][2] * pc[2] + t[1],
            R[2][0] * pc[0] + R[2][1] * pc[1] + R[2][2] * pc[2] + t[2])


def project(pw, cam, conv):
    R = cam["R"]
    t = cam["t"]
    d0 = pw[0] - t[0]
    d1 = pw[1] - t[1]
    d2 = pw[2] - t[2]
    pc = (R[0][0] * d0 + R[1][0] * d1 + R[2][0] * d2,
          R[0][1] * d0 + R[1][1] * d1 + R[2][1] * d2,
          R[0][2] * d0 + R[1][2] * d1 + R[2][2] * d2)
    if conv == "opencv":
        a = pc[0]
        b = pc[1]
        d = pc[2]
    elif conv == "opengl":
        a = pc[0]
        b = -pc[1]
        d = -pc[2]
    else:
        a = -pc[1]
        b = -pc[2]
        d = pc[0]
    if d <= 1e-6:
        return None
    K = cam["K"]
    return (K[0][0] * a / d + K[0][2], K[1][1] * b / d + K[1][2])


def camera_info(obs, name):
    c = obs.cameras[name]
    K = rows_of(c.intrinsics)
    pose = c.camera_pose
    q = [float(v) for v in pose.quaternion_wxyz]
    return {"K": [[float(v) for v in r] for r in K], "R": quat_to_mat(q),
            "t": [float(v) for v in pose.position], "q": q, "frame": str(pose.frame)}


def calibrate_convention(rows, depth, cam, w, h):
    """Choose the camera-axis convention under which table pixels form the flattest plane."""
    best = None
    report = {}
    for conv in ("opencv", "opengl", "ros"):
        zs = []
        for y in range(h // 3, h, 8):
            for x in range(4, w - 4, 8):
                d = depth[y][x]
                if d > 0.05 and d < 5.0 and is_table(rows[y][x]):
                    zs.append(deproject(x, y, d, cam, conv)[2])
        if len(zs) < 20:
            report[conv] = {"n": len(zs)}
            continue
        mu = sum(zs) / len(zs)
        sd = math.sqrt(sum((z - mu) * (z - mu) for z in zs) / len(zs))
        report[conv] = {"n": len(zs), "z_mean": round(mu, 4), "z_sd": round(sd, 4)}
        if best is None or sd < best[1]:
            best = (conv, sd, mu)
    return best, report


# ---------------------------------------------------------------- robot helpers
STATE = {"ended": False, "ticks": 0}
result["motions"] = []


def record_motion(name, res):
    entry = {"call": name}
    try:
        entry["ok"] = bool(res.ok)
    except Exception:
        entry["ok"] = None
    try:
        entry["status"] = str(res.status)
    except Exception:
        pass
    try:
        entry["steps"] = int(res.steps_executed)
        STATE["ticks"] += int(res.steps_executed)
    except Exception:
        pass
    try:
        entry["error"] = str(res.error)[:200]
    except Exception:
        pass
    try:
        entry["terminated"] = bool(res.terminated)
        if entry["terminated"]:
            STATE["ended"] = True
    except Exception:
        pass
    try:
        entry["truncated"] = bool(res.truncated)
        if entry["truncated"]:
            STATE["ended"] = True
    except Exception:
        pass
    result["motions"].append(entry)
    return entry


def ended_error(exc):
    s = str(exc)
    return "Episode has ended" in s or "execution is closing" in s


def hold_step(state, grip):
    reply = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=list(state.joint_positions["primary"]),
        gripper_position=grip, embodiment=state.embodiment)}))
    STATE["ticks"] += 1
    try:
        if bool(reply.terminated) or bool(reply.truncated):
            STATE["ended"] = True
    except Exception:
        pass
    return reply


def topdown_quat(psi_deg):
    a = math.radians(psi_deg) / 2.0
    return (0.0, math.cos(a), math.sin(a), 0.0)


def move_pose(name, xyz, psi, tol, max_steps):
    if STATE["ended"]:
        return None
    pose = Pose((float(xyz[0]), float(xyz[1]), float(xyz[2])), topdown_quat(psi), BASE["frame"])
    res = move_to_pose(pose, tolerance=tol, max_steps=max_steps)
    e = record_motion(name, res)
    e["target"] = [round(float(v), 4) for v in xyz] + [round(psi, 1)]
    try:
        p = get_robot_state().end_effector_poses["primary"].position
        e["reached"] = [round(float(v), 4) for v in p]
    except Exception:
        pass
    return res


BASE = {"frame": "robot_base"}

# ================================================================ main program
try:
    ctx = get_task_context()
    goal = str(ctx.language)
    result["goal"] = goal
    result["task_name"] = str(ctx.task_name)
    state0 = get_robot_state()
    BASE["frame"] = state0.base_frame
    result["base_frame"] = str(state0.base_frame)
    result["embodiment"] = str(state0.embodiment)
    try:
        md = robomme_plus.get_task_metadata()
        result["task_metadata"] = dict(md)
    except Exception as exc:
        err("task_metadata", exc)
    try:
        cm = robomme_plus.get_controller_metadata()
        result["controller_metadata"] = dict(cm)
    except Exception as exc:
        err("controller_metadata", exc)
    try:
        ee0 = state0.end_effector_poses["primary"]
        result["ee0"] = {"pos": [round(float(v), 4) for v in ee0.position],
                         "quat": [round(float(v), 4) for v in ee0.quaternion_wxyz]}
        result["joints0"] = [round(float(v), 4) for v in state0.joint_positions["primary"]]
        result["gripper0"] = float(state0.gripper_positions["primary"])
    except Exception as exc:
        err("state0", exc)
    result["order"] = parse_goal(goal)
except Exception as exc:
    err("setup", exc)

MAX_STEPS = 1300
try:
    MAX_STEPS = int(result["task_metadata"]["max_steps"])
except Exception:
    pass
result["max_steps_used"] = MAX_STEPS

TRACKS = []
CAL = {"conv": None, "front": None}
LIVE_STATIC_FALLBACK = [None]
W = 256
H = 256
result["phase"] = "video"
try:
    video = robomme_plus.get_conditioning_video()
    frames = [f for f in video]
    nf = len(frames)
    result["video_frames"] = nf
    if nf > 0:
        result["video_t"] = [round(float(frames[0].timestamp_s), 3), round(float(frames[-1].timestamp_s), 3)]
        result["video_cameras"] = [str(c) for c in frames[0].cameras]
        H = len(frames[0].cameras["front"].rgb)
        W = len(frames[0].cameras["front"].rgb[0])
        result["image_hw"] = [H, W]
        bg = rgb_rows(frames[0], "front")
        white0 = [False] * (W * H)
        occl0 = [False] * (W * H)
        for y in range(H):
            row = bg[y]
            for x in range(W):
                p = row[x]
                if is_whitish(p):
                    white0[y * W + x] = True
                if not is_table(p):
                    occl0[y * W + x] = True
        # v01: frame 0 already contains the cubes, so 'static' white must persist through the video
        # (frame 0, middle and last frame), and the occluder excludes cube-coloured pixels below.
        white_first0 = list(white0)
        for j in (nf // 2, nf - 1):
            if j > 0:
                rj = rgb_rows(frames[j], "front")
                for y in range(H):
                    row = rj[y]
                    for x in range(W):
                        if white0[y * W + x] and not is_whitish(row[x]):
                            white0[y * W + x] = False
        static_v = dilate(white0, W, H, 2)
        LIVE_STATIC_FALLBACK[0] = static_v
        static_first0 = dilate(white_first0, W, H, 2)
        OCCL["mask"] = dilate(occl0, W, H, 2)
        result["static_video_pix"] = sum(1 for f in static_v if f)
        result["occluder_pix"] = sum(1 for f in occl0 if f)
        # first frame containing containers
        first = -1
        counts = []
        for i in range(nf):
            rows = rgb_rows(frames[i], "front")
            pts = fg_points(rows, static_first0, W, H, (0, W - 1, 0, H - 1), 4)
            counts.append(len(pts))
            if len(pts) >= 40:
                first = i
                break
        result["first_container_frame"] = first
        result["first_counts_tail"] = counts[-6:]
        if first >= 1:
            cube_rows = rgb_rows(frames[first - 1], "front")
            cover_rows = rgb_rows(frames[first], "front")
            cubes, tmodel = detect_cubes_cover(cube_rows, cover_rows, static_v, W, H)
            result["table_model"] = None if tmodel is None else [round(tmodel[0], 1), round(tmodel[1], 3)]
            result["front_cubes"] = [{"x": round(c["x"], 1), "y": round(c["y"], 1), "n": c["n"],
                                      "rgb": c["rgb"], "color": c["color"], "blob_n": c["blob_n"],
                                      "names": c["names"]} for c in cubes]
            # cube pixels are not occluders (frame 0 shows the cubes in the live video)
            for c in cubes:
                for yy in range(max(0, int(c["y"]) - 8), min(H, int(c["y"]) + 9)):
                    for xx in range(max(0, int(c["x"]) - 8), min(W, int(c["x"]) + 9)):
                        OCCL["mask"][yy * W + xx] = False
            result["occluder_pix_after_cubes"] = sum(1 for f in OCCL["mask"] if f)
            for c in cubes:
                if c["color"] in ("gray",) and c["n"] < 8:
                    continue
                TRACKS.append({"x": c["x"], "y": c["y"], "vx": 0.0, "vy": 0.0, "color": c["color"],
                               "src": "front", "coast": 0, "n": 0})
            # wrist-assisted hidden cubes (front occluded) via RGB-D projection
            try:
                hidden = []
                fcam = camera_info(frames[0], "front")
                wcam = camera_info(frames[0], "wrist")
                result["front_cam"] = {"t": [round(v, 4) for v in fcam["t"]], "q": [round(v, 4) for v in fcam["q"]],
                                       "K": [[round(v, 2) for v in r] for r in fcam["K"]], "frame": fcam["frame"]}
                result["wrist_cam"] = {"t": [round(v, 4) for v in wcam["t"]], "q": [round(v, 4) for v in wcam["q"]],
                                       "frame": wcam["frame"]}
                fdepth = rows_of(frames[0].cameras["front"].depth_m)
                fcal, frep = calibrate_convention(bg, fdepth, fcam, W, H)
                wbg = rgb_rows(frames[0], "wrist")
                wdepth0 = rows_of(frames[0].cameras["wrist"].depth_m)
                wcal, wrep = calibrate_convention(wbg, wdepth0, wcam, W, H)
                result["calib_front"] = frep
                result["calib_wrist"] = wrep
                if fcal is not None and wcal is not None and fcal[0] == wcal[0] and fcal[1] < 0.01 and wcal[1] < 0.01 \
                        and abs(fcal[2] - wcal[2]) < 0.03:
                    conv = fcal[0]
                    result["convention"] = conv
                    CAL["conv"] = conv
                    CAL["front"] = fcam
                    wrows = rgb_rows(frames[first - 1], "wrist")
                    wdepth = rows_of(frames[first - 1].cameras["wrist"].depth_m)
                    wstatic = [is_whitish(p) for row in wbg for p in row]
                    for j in (nf // 2, nf - 1):
                        rj = rgb_rows(frames[j], "wrist")
                        for y in range(H):
                            for x in range(W):
                                if wstatic[y * W + x] and not is_whitish(rj[y][x]):
                                    wstatic[y * W + x] = False
                    wstatic = dilate(wstatic, W, H, 2)
                    wcover = rgb_rows(frames[first], "wrist")
                    wcubes, wmodel = detect_cubes_cover(wrows, wcover, wstatic, W, H)
                    wlist = []
                    for c in wcubes:
                        u = int(c["x"])
                        v = int(c["y"])
                        d = wdepth[v][u]
                        if not (d > 0.02 and d < 5.0):
                            continue
                        pw = deproject(u, v, d, wcam, conv)
                        uv = project(pw, fcam, conv)
                        item = {"color": c["color"], "n": c["n"], "wx": round(c["x"], 1), "wy": round(c["y"], 1),
                                "base": [round(pw[0], 4), round(pw[1], 4), round(pw[2], 4)]}
                        if uv is not None:
                            item["front_uv"] = [round(uv[0], 1), round(uv[1], 1)]
                        wlist.append(item)
                        if uv is None or c["n"] < 10 or c["color"] in ("gray", "white"):
                            continue
                        fu = int(uv[0])
                        fv = int(uv[1])
                        if not (0 <= fu < W and 0 <= fv < H):
                            continue
                        near_same = False
                        for t in TRACKS:
                            if t["color"] == c["color"] and abs(t["x"] - uv[0]) + abs(t["y"] - uv[1]) < 30:
                                near_same = True
                        occluded = False
                        for yy in range(max(0, fv - 3), min(H, fv + 4)):
                            for xx in range(max(0, fu - 3), min(W, fu + 4)):
                                if OCCL["mask"][yy * W + xx]:
                                    occluded = True
                        item["near_same_front"] = near_same
                        item["front_occluded"] = occluded
                        if occluded and not near_same:
                            hidden.append({"x": uv[0], "y": uv[1], "vx": 0.0, "vy": 0.0, "color": c["color"],
                                           "src": "wrist", "coast": 0, "n": 0})
                    result["wrist_cubes"] = wlist
                for t in hidden:
                    TRACKS.append(t)
                result["hidden_added"] = [{"color": t["color"], "x": round(t["x"], 1), "y": round(t["y"], 1)} for t in hidden]
            except Exception as exc:
                err("wrist_hidden", exc)
            if not TRACKS:
                log("no cube tracks from the video; live phase will seed unknown container tracks")
                first = nf - 1
            # refine track centres on the first container frame
            rows = rgb_rows(frames[first], "front")
            pts = fg_points(rows, static_v, W, H, (0, W - 1, 0, H - 1), VIDEO_STRIDE)
            for rep in range(3 if TRACKS else 0):
                track_update(TRACKS, pts, 0.0, False, 2, W, H)
            result["tracks_init"] = [{"color": t["color"], "x": round(t["x"], 1), "y": round(t["y"], 1), "n": t["n"],
                                      "src": t["src"]} for t in TRACKS]
            # video tracking
            snapshots = []
            last_t = float(frames[first].timestamp_s)
            for i in range(first + 1, nf):
                fr = frames[i]
                ts = float(fr.timestamp_s)
                dt = ts - last_t
                last_t = ts
                rows = rgb_rows(fr, "front")
                pts = fg_points(rows, static_v, W, H, tracks_roi(TRACKS, W, H), VIDEO_STRIDE)
                if TRACKS:
                    track_update(TRACKS, pts, dt, True, 2, W, H)
                    update_approach(TRACKS)
                if i % 64 == 0 or i == nf - 1:
                    snapshots.append({"i": i, "t": round(ts, 2),
                                      "pos": [[round(t["x"], 1), round(t["y"], 1), t["n"]] for t in TRACKS]})
            result["video_snapshots"] = snapshots
            result["video_reacquisitions"] = OCCL["reacq"]
            result["video_splits"] = OCCL["split"]
            result["video_end_t"] = last_t
            try:
                fcam_end = camera_info(frames[-1], "front")
                result["front_cam_end_t"] = [round(v, 4) for v in fcam_end["t"]]
            except Exception as exc:
                err("front_cam_end", exc)
except Exception as exc:
    err("video", exc)

# ---------------------------------------------------------------- live settle tracking
result["phase"] = "live_track"
LIVE = {}
try:
    obs = get_observation()
    result["live_cameras"] = [str(c) for c in obs.cameras]
    try:
        lcam = camera_info(obs, "front")
        result["live_front_cam_t"] = [round(v, 4) for v in lcam["t"]]
        result["live_front_cam_q"] = [round(v, 4) for v in lcam["q"]]
    except Exception as exc:
        err("live_cam", exc)
    if len(TRACKS) == 0:
        # v01 fallback: no video identities -> track the live containers as 'unknown' so that the
        # settle/localize/pick phases still run (identity then comes only from the reveal check)
        try:
            rows0 = rgb_rows(obs, "front")
            static0 = LIVE_STATIC_FALLBACK[0]
            if static0 is None:
                static0 = [False] * (W * H)
            pts0 = fg_points(rows0, static0, W, H, (0, W - 1, 0, H - 1), VIDEO_STRIDE)
            for bl in orphan_blobs(pts0, [], [], W, H):
                if bl[2] >= ORPHAN_MIN:
                    TRACKS.append({"x": bl[0], "y": bl[1], "vx": 0.0, "vy": 0.0, "color": "unknown",
                                   "src": "live_fallback", "coast": 0, "n": bl[2]})
            result["live_fallback_tracks"] = [[round(t["x"], 1), round(t["y"], 1), t["n"]] for t in TRACKS]
        except Exception as exc:
            err("live_fallback", exc)
    if len(TRACKS) > 0:
        rows = rgb_rows(obs, "front")
        static_l = [False] * (W * H)
        lim2 = (TRACK_RADIUS + 8.0) * (TRACK_RADIUS + 8.0)
        for y in range(H):
            row = rows[y]
            for x in range(W):
                if is_whitish(row[x]):
                    far = True
                    for t in TRACKS:
                        dx = x - t["x"]
                        dy = y - t["y"]
                        if dx * dx + dy * dy < lim2:
                            far = False
                            break
                    if far:
                        static_l[y * W + x] = True
        static_l = dilate(static_l, W, H, 2)
        occl_l = [False] * (W * H)
        for y in range(H):
            row = rows[y]
            for x in range(W):
                if not is_table(row[x]) and not is_whitish(row[x]):
                    occl_l[y * W + x] = True
        OCCL["mask"] = dilate(occl_l, W, H, 2)
        result["occluder_live_pix"] = sum(1 for f in OCCL["mask"] if f)
        result["static_live_pix"] = sum(1 for f in static_l if f)
        LIVE["static"] = static_l
        pts = fg_points(rows, static_l, W, H, tracks_roi(TRACKS, W, H), VIDEO_STRIDE)
        track_update(TRACKS, pts, 0.05, True, 2, W, H)
        hist = [[(t["x"], t["y"]) for t in TRACKS]]
        state = get_robot_state()
        grip0 = float(state.gripper_positions["primary"])
        ticks = 0
        settled = False
        motion_log = []
        while ticks < SETTLE_MAX_TICKS and not STATE["ended"]:
            reply = hold_step(state, grip0)
            ticks += 1
            if STATE["ended"]:
                break
            try:
                o2 = reply.observation
                rows = rgb_rows(o2, "front")
            except Exception:
                o2 = get_observation()
                rows = rgb_rows(o2, "front")
            pts = fg_points(rows, static_l, W, H, tracks_roi(TRACKS, W, H), VIDEO_STRIDE)
            track_update(TRACKS, pts, 0.05, True, 2, W, H)
            update_approach(TRACKS)
            hist.append([(t["x"], t["y"]) for t in TRACKS])
            if len(hist) > SETTLE_WINDOW:
                old = hist[-1 - SETTLE_WINDOW]
                mv = 0.0
                for k in range(len(TRACKS)):
                    d = math.sqrt((TRACKS[k]["x"] - old[k][0]) ** 2 + (TRACKS[k]["y"] - old[k][1]) ** 2)
                    mv = max(mv, d)
                if ticks % 5 == 0:
                    motion_log.append([ticks, round(mv, 2)])
                if ticks >= SETTLE_MIN_TICKS and mv < SETTLE_PIX:
                    settled = True
                    break
        result["settle_ticks"] = ticks
        result["settled"] = settled
        result["settle_motion"] = motion_log
        LIVE["rows"] = rows
    result["tracks_final"] = [{"color": t["color"], "x": round(t["x"], 1), "y": round(t["y"], 1), "n": t["n"],
                               "coast_total": t.get("coast_total", 0), "src": t["src"]} for t in TRACKS]
except Exception as exc:
    err("live_track", exc)
    if ended_error(exc):
        STATE["ended"] = True


# ---------------------------------------------------------------- 3-D container estimates
def own_pixels(pts, tracks):
    r2 = TRACK_RADIUS * TRACK_RADIUS
    own = [[] for t in tracks]
    for p in pts:
        best = -1
        bd = r2
        for k in range(len(tracks)):
            d2 = (p[0] - tracks[k]["x"]) ** 2 + (p[1] - tracks[k]["y"]) ** 2
            if d2 < bd:
                bd = d2
                best = k
        if best >= 0:
            own[best].append(p)
    return own


def cloud_from_pixels(pix, depth_rows):
    """Base-frame points for own pixels: provider projection first, then own calibrated projection."""
    info = {}
    if len(pix) >= 15:
        try:
            mask = [[False] * W for yy in range(H)]
            for p in pix:
                mask[p[1]][p[0]] = True
            pc = mask_to_point_cloud(mask, "front", target_frame=BASE["frame"])
            pts = [(float(q[0]), float(q[1]), float(q[2])) for q in pc.points]
            if len(pts) >= 10:
                info["src"] = "own_mask"
                return pts, info
            info["own_mask_n"] = len(pts)
        except Exception as exc:
            info["own_mask_error"] = str(exc)[:160]
        if CAL["conv"] is not None and depth_rows is not None:
            pts = []
            for p in pix:
                d = depth_rows[p[1]][p[0]]
                if d > 0.02 and d < 5.0:
                    pts.append(deproject(p[0], p[1], d, CAL["front"], CAL["conv"]))
            if len(pts) >= 10:
                info["src"] = "own_projection"
                return pts, info
    return None, info


def describe_cloud(cloud_pts, info):
    zs = [q[2] for q in cloud_pts]
    ztop = quantile(zs, 0.9)
    top = [q for q in cloud_pts if q[2] >= ztop - 0.012]
    if len(top) < 5:
        top = cloud_pts
    mx = median([q[0] for q in top])
    my = median([q[1] for q in top])
    near = [q for q in top if (q[0] - mx) ** 2 + (q[1] - my) ** 2 < 0.05 * 0.05]
    if len(near) >= 5:
        top = near
        mx = median([q[0] for q in top])
        my = median([q[1] for q in top])
    best_ang = 0.0
    best_area = None
    best_ext = (0.0, 0.0)
    for step_i in range(30):
        ang = step_i * 3.0
        ca = math.cos(math.radians(ang))
        sa = math.sin(math.radians(ang))
        us = [(q[0] - mx) * ca + (q[1] - my) * sa for q in top]
        vs = [-(q[0] - mx) * sa + (q[1] - my) * ca for q in top]
        eu = quantile(us, 0.95) - quantile(us, 0.05)
        ev = quantile(vs, 0.95) - quantile(vs, 0.05)
        area = eu * ev
        if best_area is None or area < best_area:
            best_area = area
            best_ang = ang
            best_ext = (eu, ev)
    info["xy"] = [round(mx, 4), round(my, 4)]
    info["z_top"] = round(ztop, 4)
    info["z_low"] = round(quantile(zs, 0.05), 4)
    info["n_pts"] = len(cloud_pts)
    info["n_top"] = len(top)
    info["face_angle_deg"] = best_ang
    info["top_extent"] = [round(best_ext[0], 4), round(best_ext[1], 4)]
    return info


result["phase"] = "localize"
result["containers"] = []
try:
    if len(TRACKS) > 0 and not STATE["ended"]:
        obs = get_observation()
        rows = rgb_rows(obs, "front")
        LIVE["before_rows"] = rows
        try:
            depth_rows = rows_of(obs.cameras["front"].depth_m)
        except Exception:
            depth_rows = None
        static_l = LIVE.get("static", [False] * (W * H))
        pts = fg_points(rows, static_l, W, H, tracks_roi(TRACKS, W, H), 1)
        # containers no track claims (hidden at the start or lost) become 'unknown' tracks
        try:
            blobs = orphan_blobs([p for p in pts if (p[0] % 2 == 0 and p[1] % 2 == 0)],
                                 [t["x"] for t in TRACKS], [t["y"] for t in TRACKS], W, H)
            added = []
            for bl in blobs:
                if bl[2] >= 2 * ORPHAN_MIN:
                    TRACKS.append({"x": bl[0], "y": bl[1], "vx": 0.0, "vy": 0.0, "color": "unknown",
                                   "src": "orphan", "coast": 0, "n": bl[2]})
                    added.append([round(bl[0], 1), round(bl[1], 1), bl[2]])
            result["orphan_tracks_added"] = added
        except Exception as exc:
            err("orphans", exc)
        own = own_pixels(pts, TRACKS)
        for k in range(len(TRACKS)):
            t = TRACKS[k]
            info = {"k": k, "color": t["color"], "px": [round(t["x"], 1), round(t["y"], 1)], "npix": len(own[k])}
            cloud_pts, cinfo = cloud_from_pixels(own[k], depth_rows)
            for key in cinfo:
                info[key] = cinfo[key]
            if cloud_pts is None:
                try:
                    seg = segment_points("front", [[int(t["x"]), int(t["y"])]])
                    info["seg_ok"] = bool(seg.ok)
                    if seg.ok and len(seg.segmentations) > 0:
                        cand = seg.segmentations[0]
                        try:
                            info["seg_box"] = [round(float(v), 1) for v in cand.box_xyxy]
                        except Exception:
                            pass
                        pc = mask_to_point_cloud(cand, "front", target_frame=BASE["frame"])
                        cloud_pts = [(float(q[0]), float(q[1]), float(q[2])) for q in pc.points]
                        info["src"] = "segment_points"
                except Exception as exc:
                    info["seg_error"] = str(exc)[:160]
            if cloud_pts is not None and len(cloud_pts) >= 10:
                describe_cloud(cloud_pts, info)
            result["containers"].append(info)
except Exception as exc:
    err("localize", exc)
    if ended_error(exc):
        STATE["ended"] = True


# ---------------------------------------------------------------- reveal check
def reveal_colour(before, after, x, y):
    """Colour of a cube newly visible where the lifted container stood (front camera)."""
    counts = {}
    for yy in range(max(0, int(y) - 8), min(H, int(y) + 18)):
        for xx in range(max(0, int(x) - 14), min(W, int(x) + 15)):
            b = before[yy][xx]
            a = after[yy][xx]
            if not is_whitish(b):
                continue
            mx = max(a[0], a[1], a[2])
            mn = min(a[0], a[1], a[2])
            if mx - mn < 60 or mx < 60:
                continue
            name = colour_name(a[0], a[1], a[2])
            counts[name] = counts.get(name, 0) + 1
    best = None
    bn = 0
    for name in counts:
        if counts[name] > bn:
            bn = counts[name]
            best = name
    if bn < REVEAL_MIN_PIX:
        return None, counts
    return best, counts


# ---------------------------------------------------------------- pick sequence
result["phase"] = "pick"
result["picks"] = []


def grasp_yaws(c, conts, psi0):
    x = c["xy"][0]
    y = c["xy"][1]
    options = []
    for phi in (c["face_angle_deg"], c["face_angle_deg"] + 90.0):
        ux = math.cos(math.radians(phi))
        uy = math.sin(math.radians(phi))
        clear = 1.0
        for o in conts:
            if o["k"] == c["k"]:
                continue
            for sgn in (1.0, -1.0):
                fx = x + sgn * 0.055 * ux
                fy = y + sgn * 0.055 * uy
                clear = min(clear, math.sqrt((o["xy"][0] - fx) ** 2 + (o["xy"][1] - fy) ** 2))
        psi = phi + 90.0
        while psi - psi0 > 90.0:
            psi -= 180.0
        while psi - psi0 <= -90.0:
            psi += 180.0
        options.append([clear, psi])
    if options[1][0] > options[0][0]:
        options = [options[1], options[0]]
    return options


def pick_one(c, conts, psi0, expected):
    """Hover, descend, close, lift, check the revealed cube, lower, open, rise."""
    x = c["xy"][0]
    y = c["xy"][1]
    top = c["z_top"]
    opts = grasp_yaws(c, conts, psi0)
    pick = {"expected": expected, "k": c["k"], "label": c["color"], "xy": [x, y], "top": top,
            "yaw_options": [[round(o[0], 3), round(o[1], 1)] for o in opts], "tick_start": STATE["ticks"]}
    result["picks"].append(pick)
    hover = top + LIFT_ABOVE_TOP
    grasp_z = top - GRASP_BELOW_TOP
    psi = None
    for o in opts:
        for cand in (o[1], o[1] + 180.0 if o[1] < psi0 else o[1] - 180.0):
            if STATE["ended"]:
                return pick
            r = move_pose("hover", (x, y, hover), cand, 0.02, 250)
            ok = False
            try:
                ok = bool(r.ok)
            except Exception:
                ok = False
            if ok:
                psi = cand
                break
        if psi is not None:
            break
    if psi is None:
        pick["failed"] = "hover unreachable for every yaw"
        return pick
    pick["psi"] = round(psi, 1)
    held = False
    for attempt in range(2):
        gz = grasp_z - 0.012 * attempt
        move_pose("descend", (x, y, gz), psi, 0.01, 150)
        if STATE["ended"]:
            return pick
        record_motion("close_gripper", close_gripper())
        if STATE["ended"]:
            return pick
        try:
            gval = float(get_robot_state().gripper_positions["primary"])
        except Exception:
            gval = -1.0
        pick["grip_after_close_" + str(attempt)] = round(gval, 3)
        if gval > HELD_MIN:
            held = True
            grasp_z = gz
            break
        record_motion("open_gripper", open_gripper())
        if STATE["ended"]:
            return pick
    pick["held"] = held
    move_pose("lift", (x, y, hover), psi, 0.02, 150)
    if STATE["ended"]:
        return pick
    try:
        pick["grip_after_lift"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
        after = rgb_rows(get_observation(), "front")
        seen, counts = reveal_colour(LIVE["before_rows"], after, c["px"][0], c["px"][1])
        pick["revealed"] = seen
        pick["reveal_counts"] = counts
    except Exception as exc:
        pick["reveal_error"] = str(exc)[:160]
        if ended_error(exc):
            STATE["ended"] = True
            return pick
    move_pose("lower", (x, y, grasp_z + 0.005), psi, 0.015, 150)
    if STATE["ended"]:
        return pick
    record_motion("open_gripper", open_gripper())
    if STATE["ended"]:
        return pick
    move_pose("rise", (x, y, hover), psi, 0.02, 150)
    pick["tick_end"] = STATE["ticks"]
    return pick


try:
    conts = [c for c in result["containers"] if "xy" in c]
    psi0 = 0.0
    try:
        q0 = [float(v) for v in get_robot_state().end_effector_poses["primary"].quaternion_wxyz]
        psi0 = math.degrees(2.0 * math.atan2(q0[2], q0[1]))
        result["psi0_deg"] = round(psi0, 1)
    except Exception as exc:
        err("psi0", exc)
    used = set()
    queue = []
    for item in result.get("order", []):
        queue.append(item)
    result["queue"] = [dict(q) for q in queue]
    if conts and queue and not STATE["ended"]:
        record_motion("open_gripper", open_gripper())
    for item in queue:
        if STATE["ended"]:
            break
        colour = item["color"]
        labelled = [c for c in conts if c["color"] == colour and c["k"] not in used]
        if not labelled:
            labelled = [c for c in conts if c["color"] == "unknown" and c["k"] not in used][:1]
            if labelled:
                log("no container labelled " + colour + "; trying an unclaimed container")
        if not labelled:
            result["picks"].append({"expected": colour, "skipped": "no container labelled with this colour"})
            continue
        if not item["all"]:
            labelled = labelled[:1]
        todo = list(labelled)
        retries = 0
        while todo and not STATE["ended"]:
            c = todo.pop(0)
            used.add(c["k"])
            pk = pick_one(c, conts, psi0, colour)
            seen = pk.get("revealed")
            if seen is None or seen == colour or colour in ("white", "black", "gray") or retries >= MAX_RETRIES:
                continue
            # wrong container: relabel it, and swap labels with its nearest-approach partner of that colour
            retries += 1
            c["color"] = seen
            partners = [o for o in conts if o["k"] not in used and o["color"] in (seen, "unknown")]
            best = None
            for o in partners:
                if best is None or approach(c["k"], o["k"]) < approach(c["k"], best["k"]):
                    best = o
            if best is not None:
                log("relabel k%d as %s; k%d (was %s) now %s" % (c["k"], seen, best["k"], best["color"], colour))
                best["color"] = colour
                todo.insert(0, best)
except Exception as exc:
    err("pick", exc)
    if ended_error(exc):
        STATE["ended"] = True

result["phase"] = "done"
result["ended"] = STATE["ended"]
result["ticks_counted"] = STATE["ticks"]
