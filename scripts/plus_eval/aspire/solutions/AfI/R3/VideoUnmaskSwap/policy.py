# VideoUnmaskSwap v02 -- UNTESTED minimal repair of v01 after start 3 (0 simulator runs).
# Changes vs v01: (1) coloured objects still visible at the end of pass 1 are excluded from the
# hidden cubes and from the tracking start frame; extra tracks near them are dropped;
# (2) a target colour with no visible cube uses one unexplained container (no cube) instead of
# the nearest-hue container; tracks are never lifted twice.
# Start 1 (v00, ep 0) lifted the wrong container: its component tracker merged touching
# containers (7 containers moving simultaneously for 35 s) and the tracks drifted off the table.
# v01 tracks only the container TOP faces (a height band derived from the last video frame),
# whose XY footprints never touch, at 2-px resolution inside a workspace ROI derived from the
# observed cubes, with a per-frame k-means assignment (nearest constant-velocity prediction).
# Tracks are seeded at every coloured cube seen before the containers covered them, so each
# track carries its cube's colour.  The lift plan follows the goal's colour order and lifts every
# container whose cube is labelled with that colour, putting each one back after the lift.
import math

report = {"version": "v02_static_distractor", "log": [], "errors": [], "calls": []}
result = report

VID_STEP = 4          # pixel subsampling for video frames
LIVE_STEP = 2         # pixel subsampling for the live frame
HMIN = 0.008          # min height above table for a foreground cell
HMAX = 0.30           # max height above table (excludes most of the arm at rest)
BASE_R2 = 0.20 * 0.20 # exclude cells within 0.20 m (XY) of the robot base origin
LINK_DZ = 0.045
LINK_D2 = 0.045 * 0.045
SAT_COLOR = 0.40
VAL_COLOR = 0.20
CONT_MIN_N = 4
GATE = 0.12
CLAIM_R = 0.03
GRASP_BELOW_TOP = 0.018
PRE_ABOVE_TOP = 0.10
LIFT_ABOVE_TOP = 0.15

COLOR_REF = {"red": 0.0, "orange": 30.0, "yellow": 58.0, "green": 120.0, "cyan": 185.0,
             "blue": 225.0, "purple": 275.0, "violet": 275.0, "magenta": 310.0, "pink": 330.0}
NB = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def note(msg):
    report["log"].append(str(msg))


def to_list(arr):
    try:
        return arr.tolist()
    except Exception:
        return [list(row) for row in arr]


def sub_rows(arr, s):
    try:
        return arr[::s, ::s].tolist()
    except Exception:
        pass
    full = to_list(arr)
    out = []
    for v in range(0, len(full), s):
        row = full[v]
        out.append([row[u] for u in range(0, len(row), s)])
    return out


def is_terminal(res):
    flag = False
    for name in ("terminated", "truncated"):
        try:
            if name == "terminated":
                flag = flag or bool(res.terminated)
            else:
                flag = flag or bool(res.truncated)
        except Exception:
            pass
    return flag


def log_call(name, res):
    entry = {"call": name}
    try:
        entry["ok"] = bool(res.ok)
    except Exception:
        pass
    try:
        entry["status"] = str(res.status)
    except Exception:
        pass
    try:
        entry["steps"] = int(res.steps_executed)
    except Exception:
        pass
    try:
        entry["terminated"] = bool(res.terminated)
        entry["truncated"] = bool(res.truncated)
    except Exception:
        pass
    try:
        entry["error"] = str(res.error)
    except Exception:
        pass
    try:
        entry["t"] = round(float(get_robot_state().timestamp_s), 3) if not is_terminal(res) else None
    except Exception:
        pass
    report["calls"].append(entry)
    return entry


def quat_to_R(q):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def hsv(r, g, b):
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = mx - mn
    if mx <= 0:
        return 0.0, 0.0, 0.0
    s = d / mx
    if d <= 1e-6:
        h = 0.0
    elif mx == r:
        h = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        h = 60.0 * ((b - r) / d + 2.0)
    else:
        h = 60.0 * ((r - g) / d + 4.0)
    return h, s, mx


def hue_dist(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def quantile(vals, frac):
    o = sorted(vals)
    if not o:
        return 0.0
    return o[int((len(o) - 1) * frac)]


def cam_info(cam):
    K = to_list(cam.intrinsics)
    pose = cam.camera_pose
    t = [float(v) for v in pose.position]
    q = [float(v) for v in pose.quaternion_wxyz]
    return K, t, q


def make_grid(K, R, H, W, s, conv):
    fx = float(K[0][0])
    fy = float(K[1][1])
    cx = float(K[0][2])
    cy = float(K[1][2])
    grid = []
    for gj in range((H + s - 1) // s):
        v = gj * s
        for gi in range((W + s - 1) // s):
            u = gi * s
            a = (u - cx) / fx
            b = (v - cy) / fy
            if conv == "opencv":
                l0, l1, l2 = a, b, 1.0
            elif conv == "opengl":
                l0, l1, l2 = a, -b, -1.0
            else:
                l0, l1, l2 = 1.0, -a, -b
            dx = R[0][0] * l0 + R[0][1] * l1 + R[0][2] * l2
            dy = R[1][0] * l0 + R[1][1] * l1 + R[1][2] * l2
            dz = R[2][0] * l0 + R[2][1] * l1 + R[2][2] * l2
            grid.append((gi, gj, dx, dy, dz))
    return grid


def table_peak(D, grid, t):
    zs = []
    for cell in grid:
        d = D[cell[1]][cell[0]]
        if not (d > 0.05 and d < 4.0):
            continue
        zs.append(t[2] + d * cell[4])
    if len(zs) < 50:
        return 0.0, 0.0, len(zs)
    hist = {}
    for z in zs:
        k = int(math.floor(z / 0.005))
        hist[k] = hist.get(k, 0) + 1
    best_k = None
    best_c = -1
    for k in hist:
        c = hist[k] + hist.get(k - 1, 0) + hist.get(k + 1, 0)
        if c > best_c:
            best_c = c
            best_k = k
    zc = (best_k + 0.5) * 0.005
    near = [z for z in zs if abs(z - zc) < 0.01]
    return best_c / float(len(zs)), quantile(near, 0.5), len(zs)


def extract(D, C, grid, t, table_z):
    """Foreground cells: key (gi,gj) -> (x, y, z, colored, hue, sat, val)."""
    cells = {}
    tx = t[0]
    ty = t[1]
    tz = t[2]
    for cell in grid:
        gi = cell[0]
        gj = cell[1]
        d = D[gj][gi]
        if not (d > 0.05 and d < 4.0):
            continue
        z = tz + d * cell[4]
        h = z - table_z
        if h < HMIN or h > HMAX:
            continue
        x = tx + d * cell[2]
        y = ty + d * cell[3]
        if x * x + y * y < BASE_R2:
            continue
        colored = False
        hu = 0.0
        sa = 0.0
        va = 0.0
        if C is not None:
            px = C[gj][gi]
            hu, sa, va = hsv(float(px[0]), float(px[1]), float(px[2]))
            colored = sa >= SAT_COLOR and va >= VAL_COLOR and h < 0.12
        cells[(gi, gj)] = (x, y, z, colored, hu, sa, va)
    return cells


def components(cells, want_colored, table_z, s):
    seen = set()
    comps = []
    for key in cells:
        if key in seen:
            continue
        if cells[key][3] != want_colored:
            continue
        stack = [key]
        seen.add(key)
        members = []
        while stack:
            k = stack.pop()
            members.append(k)
            p = cells[k]
            for off in NB:
                nk = (k[0] + off[0], k[1] + off[1])
                if nk in seen or nk not in cells:
                    continue
                q = cells[nk]
                if q[3] != want_colored:
                    continue
                if abs(q[2] - p[2]) < LINK_DZ and (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 < LINK_D2:
                    seen.add(nk)
                    stack.append(nk)
        n = len(members)
        sx = 0.0
        sy = 0.0
        su = 0.0
        sv = 0.0
        hx = 0.0
        hy = 0.0
        ssat = 0.0
        sval = 0.0
        zs = []
        pts = []
        for k in members:
            p = cells[k]
            sx += p[0]
            sy += p[1]
            su += k[0] * s
            sv += k[1] * s
            zs.append(p[2])
            pts.append((p[0], p[1], p[2]))
            hx += math.cos(math.radians(p[4])) * p[5]
            hy += math.sin(math.radians(p[4])) * p[5]
            ssat += p[5]
            sval += p[6]
        hue = math.degrees(math.atan2(hy, hx)) % 360.0
        comps.append({"x": sx / n, "y": sy / n, "top": quantile(zs, 0.9) - table_z,
                      "n": n, "u": su / n, "v": sv / n, "hue": hue, "sat": ssat / n,
                      "val": sval / n, "pts": pts})
    return comps


def compact(c):
    return [round(c["x"], 3), round(c["y"], 3), round(c["top"], 3), c["n"],
            int(c["hue"]), round(c["sat"], 2), round(c["val"], 2)]


def render(D, C, grid, t, table_z):
    rows = {}
    for cell in grid:
        gi = cell[0]
        gj = cell[1]
        d = D[gj][gi]
        ch = "#"
        if d > 0.05 and d < 4.0:
            z = t[2] + d * cell[4]
            h = z - table_z
            ch = "."
            if h < -0.03:
                ch = "_"
            elif h >= HMIN:
                ch = str(min(9, int(h / 0.02)))
                if C is not None:
                    px = C[gj][gi]
                    hu, sa, va = hsv(float(px[0]), float(px[1]), float(px[2]))
                    if sa >= SAT_COLOR and va >= VAL_COLOR:
                        best = "?"
                        bd = 999.0
                        for name in COLOR_REF:
                            dd = hue_dist(hu, COLOR_REF[name])
                            if dd < bd:
                                bd = dd
                                best = name
                        ch = best[0].upper()
        rows.setdefault(gj, []).append(ch)
    return ["".join(rows[k]) for k in sorted(rows)]


def min_rect(xy):
    best = None
    for k in range(45):
        a = math.radians(k * 2.0)
        ca = math.cos(a)
        sa = math.sin(a)
        us = [p[0] * ca + p[1] * sa for p in xy]
        vs = [-p[0] * sa + p[1] * ca for p in xy]
        u0 = quantile(us, 0.03)
        u1 = quantile(us, 0.97)
        v0 = quantile(vs, 0.03)
        v1 = quantile(vs, 0.97)
        area = (u1 - u0) * (v1 - v0)
        if best is None or area < best[0]:
            best = (area, a, (u0 + u1) / 2.0, (v0 + v1) / 2.0, u1 - u0, v1 - v0)
    a = best[1]
    ca = math.cos(a)
    sa = math.sin(a)
    cx = best[2] * ca - best[3] * sa
    cy = best[2] * sa + best[3] * ca
    return cx, cy, a, best[4], best[5]




TOP_STEP = 2          # pixel subsampling for container-top tracking in the video
ROI_R = 0.35          # workspace radius around the observed cubes (m)
ASSIGN_R = 0.045      # a top cell belongs to a track only within this XY radius of its prediction
MIN_TOP = 3           # fewer assigned top cells than this (per TOP_STEP^2 pixels) -> coast
PUT_BACK_ABOVE = 0.006


def label_color(hue):
    best = None
    for name in COLOR_REF:
        if name == "violet":
            continue
        d = hue_dist(hue, COLOR_REF[name])
        if best is None or d < best[0]:
            best = (d, name)
    return best[1]


def roi_grid(K, R, H, W, s, t, plane_z, center, radius):
    out = []
    for cell in make_grid(K, R, H, W, s, best_conv):
        if cell[4] >= -1e-6:
            continue
        d = (plane_z - t[2]) / cell[4]
        if d <= 0:
            continue
        x = t[0] + d * cell[2]
        y = t[1] + d * cell[3]
        if (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius * radius:
            out.append(cell)
    return out


def top_cells(D, grid, t, lo, hi):
    pts = []
    tx = t[0]
    ty = t[1]
    tz = t[2]
    for cell in grid:
        d = D[cell[1]][cell[0]]
        if not (d > 0.05 and d < 4.0):
            continue
        h = tz + d * cell[4] - table_z
        if h < lo or h > hi:
            continue
        pts.append((tx + d * cell[2], ty + d * cell[3], h + table_z, cell[0], cell[1]))
    return pts


def kmeans_step(pts, centers, radius, iters, min_pts):
    r2 = radius * radius
    groups = []
    cur = [c for c in centers]
    for it in range(iters):
        groups = [[] for c in cur]
        for p in pts:
            bi = -1
            bd = r2
            for i in range(len(cur)):
                d = (p[0] - cur[i][0]) ** 2 + (p[1] - cur[i][1]) ** 2
                if d < bd:
                    bd = d
                    bi = i
            if bi >= 0:
                groups[bi].append(p)
        nxt = []
        for i in range(len(cur)):
            g = groups[i]
            if len(g) >= min_pts:
                nxt.append((sum([p[0] for p in g]) / len(g), sum([p[1] for p in g]) / len(g)))
            else:
                nxt.append(centers[i])
        cur = nxt
    return cur, groups


# ---------------------------------------------------------------- task context
ctx = get_task_context()
goal = str(ctx.language)
report["goal"] = goal
try:
    meta = robomme_plus.get_task_metadata()
    report["task_metadata"] = {str(k): str(meta[k]) for k in meta}
except Exception as exc:
    report["errors"].append("task_metadata: " + str(exc))

state0 = get_robot_state()
base_frame = state0.base_frame
ee0 = state0.end_effector_poses["primary"]
q_home = [float(v) for v in ee0.quaternion_wxyz]
report["ee0"] = {"pos": [round(float(v), 4) for v in ee0.position], "quat": q_home}

words = []
cur = ""
for ch in goal.lower():
    if ch.isalpha():
        cur += ch
    else:
        if cur:
            words.append(cur)
        cur = ""
if cur:
    words.append(cur)
targets = []
for w in words:
    if w in COLOR_REF and w not in targets:
        targets.append(w)
report["target_colors"] = targets

# ---------------------------------------------------------------- conditioning video: calibration
video = robomme_plus.get_conditioning_video()
NV = len(video)
report["video_frames"] = NV
if NV == 0:
    raise ValueError("no conditioning video")
cam0 = video[0].cameras["front"]
K, tcam, qcam = cam_info(cam0)
Rcam = quat_to_R(qcam)
D0 = to_list(cam0.depth_m)
H = len(D0)
W = len(D0[0])
conv_scores = {}
best_conv = None
best_frac = -1.0
table_z = 0.0
D0s = sub_rows(cam0.depth_m, VID_STEP)
for conv in ("opencv", "opengl", "ros"):
    g = make_grid(K, Rcam, H, W, VID_STEP, conv)
    frac, zpk, nvalid = table_peak(D0s, g, tcam)
    conv_scores[conv] = [round(frac, 3), round(zpk, 4), nvalid]
    if zpk > tcam[2] - 0.1:
        frac = frac * 0.1  # a table above the camera is the mirrored (wrong) axis convention
    if frac > best_frac:
        best_frac = frac
        best_conv = conv
        table_z = zpk
report["conv_scores"] = conv_scores
report["conv"] = best_conv
report["table_z"] = round(table_z, 4)
grid_v = make_grid(K, Rcam, H, W, VID_STEP, best_conv)

# ---------------------------------------------------------------- pass 1: coloured cubes while visible
cube_tl = []
maps = {}
last_seen = -1
pass1_end = NV - 1
for fi in range(NV):
    cam = video[fi].cameras["front"]
    D = sub_rows(cam.depth_m, VID_STEP)
    C = sub_rows(cam.rgb, VID_STEP)
    cells = extract(D, C, grid_v, tcam, table_z)
    cubes = [c for c in components(cells, True, table_z, VID_STEP) if c["n"] >= 2]
    cube_tl.append([[c["x"], c["y"], c["hue"], c["n"]] for c in cubes])
    if fi == 0:
        maps["0"] = render(D, C, grid_v, tcam, table_z)
    if cubes:
        last_seen = fi
    elif last_seen >= 0 and fi - last_seen >= 12:
        pass1_end = fi
        break
report["cube_last_seen"] = last_seen
report["pass1_end"] = pass1_end

clusters = []
for fi in range(len(cube_tl)):
    for cb in cube_tl[fi]:
        hit = None
        for cl in clusters:
            if (cl["x"] - cb[0]) ** 2 + (cl["y"] - cb[1]) ** 2 < 0.04 ** 2 and hue_dist(cl["hue"], cb[2]) < 35.0:
                hit = cl
                break
        if hit is None:
            clusters.append({"x": cb[0], "y": cb[1], "hue": cb[2], "count": 1, "first": fi, "last": fi})
        else:
            hit["x"] = 0.7 * hit["x"] + 0.3 * cb[0]
            hit["y"] = 0.7 * hit["y"] + 0.3 * cb[1]
            hit["count"] += 1
            hit["last"] = fi
clusters = [cl for cl in clusters if cl["count"] >= 3]
for cl in clusters:
    cl["label"] = label_color(cl["hue"])
# v02: a coloured object still visible at the end of pass 1 is not a cube hidden by a container
# (start 3, ep 1: a static magenta object visible in all 512 frames kept last_seen at 511, so
# the container tracker started at the last frame and never tracked the swaps).
persistent = [cl for cl in clusters if cl["last"] > pass1_end - 12]
hidden = [cl for cl in clusters if cl["last"] <= pass1_end - 12]
report["persistent_colored"] = [[round(c["x"], 3), round(c["y"], 3), int(c["hue"]), c["label"], c["count"],
                                 c["first"], c["last"]] for c in persistent]
if hidden:
    clusters = hidden
    last_seen = max([cl["last"] for cl in hidden])
    report["cube_last_seen_hidden"] = last_seen
report["cube_clusters"] = [[round(c["x"], 3), round(c["y"], 3), int(c["hue"]), c["label"], c["count"],
                            c["first"], c["last"]] for c in clusters]
if not clusters:
    raise ValueError("no cubes seen in the conditioning video")

# lift plan: language order; "all containers hiding green cubes" -> every cube labelled green
plan = []
for color in targets:
    group = [i for i in range(len(clusters)) if clusters[i]["label"] == color]
    if not group:
        best = None
        for i in range(len(clusters)):
            d = hue_dist(clusters[i]["hue"], COLOR_REF[color])
            if best is None or d < best[0]:
                best = (d, i)
        group = [best[1]]
        report["errors"].append("no cube labelled %s; an unexplained container is used if any, else nearest hue" % color)
        plan.append({"color": color, "clusters": group, "fallback": True})
        continue
    plan.append({"color": color, "clusters": group, "fallback": False})
report["plan"] = plan

# ---------------------------------------------------------------- container top height (last frame)
camL = video[NV - 1].cameras["front"]
DL = sub_rows(camL.depth_m, VID_STEP)
CL = sub_rows(camL.rgb, VID_STEP)
cellsL = extract(DL, CL, grid_v, tcam, table_z)
maps[str(NV - 1)] = render(DL, CL, grid_v, tcam, table_z)
tops = sorted([c["top"] for c in components(cellsL, False, table_z, VID_STEP)
               if c["n"] >= CONT_MIN_N and c["top"] >= 0.025 and c["top"] < 0.15])
if not tops:
    raise ValueError("no containers in the last video frame")
ctop = tops[len(tops) // 2]
band_lo = ctop - 0.010   # start 1 live depth: ledge at +0.050, top at +0.072 -> keep the ledge out
band_hi = ctop + 0.012
report["container_top_h"] = round(ctop, 4)
report["maps"] = maps

ccx = sum([c["x"] for c in clusters]) / len(clusters)
ccy = sum([c["y"] for c in clusters]) / len(clusters)
roi = roi_grid(K, Rcam, H, W, TOP_STEP, tcam, table_z + ctop, (ccx, ccy), ROI_R)
report["roi_cells"] = len(roi)

# ---------------------------------------------------------------- pass 2: track container tops
init_f = min(NV - 1, last_seen + 3)
report["init_frame"] = init_f
D = sub_rows(video[init_f].cameras["front"].depth_m, TOP_STEP)
pts = top_cells(D, roi, tcam, band_lo, band_hi)
seeds = [(c["x"], c["y"]) for c in clusters]
centers, groups = kmeans_step(pts, seeds, 0.05, 3, MIN_TOP)
# containers without a cube: leftover top cells forming a blob
left = []
for p in pts:
    near = False
    for c in centers:
        if (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 < 0.05 ** 2:
            near = True
            break
    if not near:
        left.append(p)
extra = []
for p in left:
    hit = None
    for e in extra:
        if (p[0] - e[0] / e[2]) ** 2 + (p[1] - e[1] / e[2]) ** 2 < 0.035 ** 2:
            hit = e
            break
    if hit is None:
        extra.append([p[0], p[1], 1])
    else:
        hit[0] += p[0]
        hit[1] += p[1]
        hit[2] += 1
tracks = []
for i in range(len(centers)):
    tracks.append({"x": centers[i][0], "y": centers[i][1], "vx": 0.0, "vy": 0.0, "coast": 0,
                   "cube": i, "path": [], "seed_shift": round(((centers[i][0] - seeds[i][0]) ** 2 + (centers[i][1] - seeds[i][1]) ** 2) ** 0.5, 3),
                   "n0": len(groups[i]) if groups else 0})
for e in extra:
    near_pers = False
    for pc in persistent:
        if (e[0] / e[2] - pc["x"]) ** 2 + (e[1] / e[2] - pc["y"]) ** 2 < 0.08 ** 2:
            near_pers = True
    if e[2] >= 2 * MIN_TOP and not near_pers:
        tracks.append({"x": e[0] / e[2], "y": e[1] / e[2], "vx": 0.0, "vy": 0.0, "coast": 0,
                       "cube": -1, "path": [], "seed_shift": 0.0, "n0": e[2]})
report["init_top_cells"] = len(pts)

min_sep = 9.0
min_sep_frame = -1
for fi in range(init_f + 1, NV):
    D = sub_rows(video[fi].cameras["front"].depth_m, TOP_STEP)
    pts = top_cells(D, roi, tcam, band_lo, band_hi)
    preds = [(t["x"] + t["vx"], t["y"] + t["vy"]) for t in tracks]
    centers, groups = kmeans_step(pts, preds, ASSIGN_R, 2, MIN_TOP)
    for i in range(len(tracks)):
        tr = tracks[i]
        nx, ny = centers[i]
        if len(groups[i]) >= MIN_TOP:
            tr["vx"] = 0.5 * tr["vx"] + 0.5 * (nx - tr["x"])
            tr["vy"] = 0.5 * tr["vy"] + 0.5 * (ny - tr["y"])
        else:
            tr["coast"] += 1
            tr["vx"] *= 0.8
            tr["vy"] *= 0.8
        tr["x"] = nx
        tr["y"] = ny
        if fi % 4 == 0 or fi == NV - 1:
            tr["path"].append([fi, round(nx, 3), round(ny, 3), len(groups[i])])
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            d = ((tracks[i]["x"] - tracks[j]["x"]) ** 2 + (tracks[i]["y"] - tracks[j]["y"]) ** 2) ** 0.5
            if d < min_sep:
                min_sep = d
                min_sep_frame = fi
report["min_track_sep"] = [round(min_sep, 3), min_sep_frame]
report["tracks"] = [{"cube": t["cube"], "final": [round(t["x"], 3), round(t["y"], 3)], "coast": t["coast"],
                     "seed_shift": t["seed_shift"], "n0": t["n0"], "path": t["path"]} for t in tracks]
result = report

# ---------------------------------------------------------------- live frame: one more tracking step
obs = get_observation()
lcam = obs.cameras["front"]
Kl, tl, ql = cam_info(lcam)
report["live_cam_shift"] = round(abs(tl[0] - tcam[0]) + abs(tl[1] - tcam[1]) + abs(tl[2] - tcam[2]), 5)
roi1 = roi_grid(Kl, quat_to_R(ql), H, W, 1, tl, table_z + ctop, (ccx, ccy), ROI_R)
Dl = to_list(lcam.depth_m)
lpts = top_cells(Dl, roi1, tl, band_lo, band_hi)
preds = [(t["x"] + t["vx"], t["y"] + t["vy"]) for t in tracks]
lcenters, lgroups = kmeans_step(lpts, preds, ASSIGN_R, 3, MIN_TOP * 4)
report["live_tracks"] = [[round(lcenters[i][0], 3), round(lcenters[i][1], 3), len(lgroups[i])]
                         for i in range(len(tracks))]
result = report

R_home = quat_to_R(q_home)
if R_home[2][2] > -0.9:
    note("home EE z-axis not downward (%.3f); using [0,1,0,0]" % R_home[2][2])
    q_home = [0.0, 1.0, 0.0, 0.0]
    R_home = quat_to_R(q_home)
close_yaw = math.atan2(R_home[1][1], R_home[0][1])


def grasp_quat(yaw_side):
    delta = yaw_side - close_yaw
    delta = ((delta + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0
    qz = [math.cos(delta / 2.0), 0.0, 0.0, math.sin(delta / 2.0)]
    return qmul(qz, q_home), delta


def goto(xyz, quat, tol, steps, name):
    res = move_to_pose(Pose((float(xyz[0]), float(xyz[1]), float(xyz[2])), quat, base_frame),
                       tolerance=tol, max_steps=steps)
    log_call(name, res)
    return res


def hold(ticks, grip):
    st = get_robot_state()
    for tick in range(ticks):
        last = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=st.joint_positions["primary"],
            gripper_position=grip, embodiment=st.embodiment)}))
        if is_terminal(last):
            log_call("hold", last)
            return True
    return False


def geometry_of(track_index):
    g = lgroups[track_index]
    if len(g) < MIN_TOP * 4:
        return None
    zs = [p[2] for p in g]
    cx, cy, ang, wu, wv = min_rect([(p[0], p[1]) for p in g])
    best = None
    for p in g:
        d = (p[0] - cx) ** 2 + (p[1] - cy) ** 2
        if best is None or d < best[0]:
            best = (d, p)
    return {"cx": cx, "cy": cy, "top": quantile(zs, 0.5), "yaw": ang, "wu": wu, "wv": wv,
            "n": len(g), "u": best[1][3], "v": best[1][4]}


def sam_check(geom):
    out = {}
    try:
        seg = segment_points("front", [[int(geom["u"]), int(geom["v"])]])
        out["ok"] = bool(seg.ok)
        if seg.ok and len(seg.segmentations) > 0:
            cand = seg.segmentations[0]
            out["score"] = round(float(cand.score), 3)
            pc = mask_to_point_cloud(cand, "front", target_frame=base_frame)
            sp = to_list(pc.points)
            out["n"] = len(sp)
            if len(sp) >= 30:
                zs = [float(p[2]) for p in sp]
                top = quantile(zs, 0.9)
                tp = [(float(p[0]), float(p[1])) for p in sp if float(p[2]) >= top - 0.012]
                cx, cy, ang, wu, wv = min_rect(tp)
                out["geom"] = {"cx": cx, "cy": cy, "top": top, "yaw": ang, "wu": wu, "wv": wv}
    except Exception as exc:
        out["error"] = str(exc)
    return out


# ---------------------------------------------------------------- lift sequence
done = False
used_tracks = []
report["picks"] = []
pos_now = (float(ee0.position[0]), float(ee0.position[1]))
for stage in plan:
    if done:
        break
    remaining = [ti for ti in range(len(tracks)) if tracks[ti]["cube"] in stage["clusters"]]
    if stage["fallback"]:
        # v02: a colour seen in no visible cube can only hide under a container that covered no
        # visible cube (one guess); a container known to hide another colour is never right.
        unexplained = [ti for ti in range(len(tracks)) if tracks[ti]["cube"] == -1 and ti not in used_tracks
                       and len(lgroups[ti]) >= MIN_TOP * 4]
        if unexplained:
            bestu = None
            for ti in unexplained:
                d = (lcenters[ti][0] - pos_now[0]) ** 2 + (lcenters[ti][1] - pos_now[1]) ** 2
                if bestu is None or d < bestu[0]:
                    bestu = (d, ti)
            remaining = [bestu[1]]
    remaining = [ti for ti in remaining if ti not in used_tracks]
    while remaining and not done:
        best = None
        for ti in remaining:
            d = (lcenters[ti][0] - pos_now[0]) ** 2 + (lcenters[ti][1] - pos_now[1]) ** 2
            if best is None or d < best[0]:
                best = (d, ti)
        ti = best[1]
        used_tracks.append(ti)
        remaining = [r for r in remaining if r != ti]
        pick = {"color": stage["color"], "track": ti}
        report["picks"].append(pick)
        geom = geometry_of(ti)
        if geom is None:
            report["errors"].append("track %d has no live top cells" % ti)
            continue
        pick["geom_own"] = {k: round(geom[k], 4) for k in geom}
        sam = sam_check(geom)
        pick["sam"] = {k: (sam[k] if k != "geom" else {kk: round(sam[k][kk], 4) for kk in sam[k]}) for k in sam}
        gx = geom["cx"]
        gy = geom["cy"]
        top = geom["top"]
        yaw = geom["yaw"]
        pick["geom_used"] = "own"
        if "geom" in sam:
            sg = sam["geom"]
            if ((sg["cx"] - gx) ** 2 + (sg["cy"] - gy) ** 2) ** 0.5 < 0.02 and abs(sg["top"] - top) < 0.02:
                pick["geom_used"] = "own+sam_agree"
        quat, delta = grasp_quat(yaw)
        pick["yaw_delta_deg"] = round(math.degrees(delta), 1)
        result = report
        r = open_gripper()
        if is_terminal(r):
            done = True
            break
        r = goto((gx, gy, top + PRE_ABOVE_TOP), quat, 0.015, 200, "pre_grasp")
        if is_terminal(r):
            done = True
            break
        r = goto((gx, gy, top - GRASP_BELOW_TOP), quat, 0.01, 120, "descend")
        if is_terminal(r):
            done = True
            break
        r = close_gripper()
        log_call("close", r)
        if is_terminal(r):
            done = True
            break
        pick["grip_width"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
        r = goto((gx, gy, top + LIFT_ABOVE_TOP), quat, 0.015, 150, "lift")
        if is_terminal(r):
            done = True
            break
        if hold(5, 0.0):
            done = True
            break
        pick["grip_after_lift"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
        r = goto((gx, gy, top - GRASP_BELOW_TOP + PUT_BACK_ABOVE), quat, 0.015, 150, "put_back")
        if is_terminal(r):
            done = True
            break
        r = open_gripper()
        log_call("release", r)
        if is_terminal(r) or hold(3, 1.0):
            done = True
            break
        r = goto((gx, gy, top + LIFT_ABOVE_TOP), quat, 0.02, 150, "retreat")
        if is_terminal(r):
            done = True
            break
        pos_now = (gx, gy)
if not done:
    done = hold(20, 1.0)
report["terminal_seen"] = done
result = report
