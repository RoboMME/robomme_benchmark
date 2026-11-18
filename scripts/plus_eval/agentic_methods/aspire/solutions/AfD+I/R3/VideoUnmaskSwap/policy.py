# VideoUnmaskSwap v00 -- track containers through the conditioning video (front + wrist RGB-D,
# fused in robot_base XY), continue tracking live until they settle, then lift the containers
# hiding the goal's colours in the goal's order (lift, put back, release).
# Uses only public APIs; every task-dependent choice is derived from this episode's goal,
# conditioning video and live observations.
import math

report = {"version": "v01", "events": []}


def note(key, value):
    report[key] = value


def ev(msg):
    report["events"].append(msg)


# ---------------------------------------------------------------- basic helpers
def to_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        out = []
        for row in arr:
            out.append([[int(p[0]), int(p[1]), int(p[2])] for p in row])
        return out


def to_depth_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        return [[float(v) for v in row] for row in arr]


def classify(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mn >= 120 and mx - mn <= 30:
        return 1
    if mx < 45:
        return 2
    if mx - mn < 80:
        return 0
    if r > 120 and g < 60 and b < 60:
        return 10
    if g > 120 and g > r + 50 and g > b + 60:
        return 11
    if b > 120 and b > r + 50 and b > g + 50 and g < 110:
        return 12
    if r > 120 and g > 120 and b < min(r, g) - 90 and abs(r - g) < 45:
        return 13
    if r > 120 and b < 22 and g > 0.6 * r and g < 0.8 * r:
        return 14
    if r > 100 and b > 120 and g < r - 40 and g < b - 50:
        return 15
    return 0


COLOR_NAMES = {1: "white", 2: "black", 10: "red", 11: "green", 12: "blue", 13: "yellow", 14: "orange", 15: "purple"}


def quantile(values, frac):
    vals = sorted(values)
    if len(vals) == 0:
        return None
    return vals[int((len(vals) - 1) * frac)]


def median(values):
    return quantile(values, 0.5)


def quat_to_rot(q):
    w = float(q[0])
    x = float(q[1])
    y = float(q[2])
    z = float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w = w / n
    x = x / n
    y = y / n
    z = z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def calib_of(cam):
    k = cam.intrinsics
    pose = cam.camera_pose
    return {
        "fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]), "cy": float(k[1][2]),
        "R": quat_to_rot(pose.quaternion_wxyz),
        "t": [float(pose.position[0]), float(pose.position[1]), float(pose.position[2])],
    }


def project(u, v, d, cal, conv):
    xc = (u - cal["cx"]) * d / cal["fx"]
    yc = (v - cal["cy"]) * d / cal["fy"]
    zc = d
    if conv == 1:
        yc = -yc
        zc = -zc
    R = cal["R"]
    t = cal["t"]
    return (
        R[0][0] * xc + R[0][1] * yc + R[0][2] * zc + t[0],
        R[1][0] * xc + R[1][1] * yc + R[1][2] * zc + t[1],
        R[2][0] * xc + R[2][1] * yc + R[2][2] * zc + t[2],
    )


def components(cells):
    left = set(cells)
    comps = []
    while left:
        seed = left.pop()
        stack = [seed]
        comp = [seed]
        while stack:
            cur = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cur[0] + dx, cur[1] + dy)
                    if nb in left:
                        left.discard(nb)
                        stack.append(nb)
                        comp.append(nb)
        comps.append(comp)
    return comps


ORPHANS = []


def track_step(points, centers, gate, min_pts, iters):
    k = len(centers)
    cur = list(centers)
    counts = [0] * k
    gate2 = gate * gate
    orph = []
    for it in range(iters):
        sx = [0.0] * k
        sy = [0.0] * k
        counts = [0] * k
        orph = []
        for p in points:
            best = -1
            bd = gate2
            for i in range(k):
                dx = p[0] - cur[i][0]
                dy = p[1] - cur[i][1]
                dd = dx * dx + dy * dy
                if dd < bd:
                    bd = dd
                    best = i
            if best >= 0:
                sx[best] += p[0]
                sy[best] += p[1]
                counts[best] += 1
            else:
                orph.append(p)
        nxt = []
        for i in range(k):
            if counts[i] >= min_pts:
                nxt.append((sx[i] / counts[i], sy[i] / counts[i]))
            else:
                nxt.append(centers[i])
        cur = nxt
    ORPHANS[:] = orph
    return cur, counts


RECOVER = []


def recover_lost(centers, counts, gate, min_pts, rad, tag):
    # Re-attach a track that lost all support to an orphan blob of container size (only when the
    # pairing is unambiguous: exactly one lost track and one orphan blob).
    lost = [i for i in range(len(centers)) if counts[i] < min_pts]
    if len(lost) != 1 or len(ORPHANS) < 2 * min_pts:
        return centers
    cellmap = {}
    for p in ORPHANS:
        key = (int(math.floor(p[0] / rad)), int(math.floor(p[1] / rad)))
        cellmap.setdefault(key, []).append(p)
    blobs = []
    for comp in components(list(cellmap.keys())):
        members = []
        for key in comp:
            members.extend(cellmap[key])
        if len(members) >= 2 * min_pts:
            blobs.append(members)
    if len(blobs) != 1:
        return centers
    b = blobs[0]
    cx = sum(p[0] for p in b) / len(b)
    cy = sum(p[1] for p in b) / len(b)
    out = list(centers)
    i = lost[0]
    if math.hypot(out[i][0] - cx, out[i][1] - cy) > 3.0 * gate:
        return centers
    RECOVER.append([tag, i, [round(out[i][0], 3), round(out[i][1], 3)], [round(cx, 3), round(cy, 3)], len(b)])
    out[i] = (cx, cy)
    return out


def assign_points(points, centers, gate):
    groups = [[] for c in centers]
    gate2 = gate * gate
    for p in points:
        best = -1
        bd = gate2
        for i in range(len(centers)):
            dx = p[0] - centers[i][0]
            dy = p[1] - centers[i][1]
            dd = dx * dx + dy * dy
            if dd < bd:
                bd = dd
                best = i
        if best >= 0:
            groups[best].append(p)
    return groups


# ---------------------------------------------------------------- task inputs
ctx = get_task_context()
goal = str(ctx.language)
note("goal", goal)
max_steps = None
try:
    meta = robomme_plus.get_task_metadata()
    try:
        max_steps = int(meta["max_steps"])
    except Exception:
        max_steps = None
    try:
        note("metadata", {"max_steps": max_steps, "task": str(meta["task_name"]) if "task_name" in meta else None,
                          "sampling": str(meta["conditioning_video_sampling"]) if "conditioning_video_sampling" in meta else None})
    except Exception as exc:
        note("metadata_err", str(exc))
except Exception as exc:
    note("metadata_err", str(exc))
try:
    cmeta = robomme_plus.get_controller_metadata()
    note("controller_keys", sorted([str(k) for k in cmeta]))
except Exception as exc:
    note("controller_err", str(exc))

video = robomme_plus.get_conditioning_video()
N = len(video)
note("video_frames", N)
if N > 0:
    note("video_t", [float(video[0].timestamp_s), float(video[N - 1].timestamp_s)])

# goal parsing: ordered list of (colour, all_flag)
COLOR_WORDS = ["red", "green", "blue", "yellow", "orange", "purple", "black", "white"]
low = goal.lower()
mentions = []
for cw in COLOR_WORDS:
    pos = low.find(cw)
    while pos >= 0:
        mentions.append((pos, cw))
        pos = low.find(cw, pos + 1)
mentions = sorted(mentions)
plan_spec = []
for idx, m in enumerate(mentions):
    pos = m[0]
    cw = m[1]
    start = mentions[idx - 1][0] if idx > 0 else 0
    seg = low[start:pos + len(cw) + 7]
    after = low[pos + len(cw):pos + len(cw) + 7]
    is_all = ("all " in low[max(start, pos - 30):pos]) or after.startswith(" cubes")
    plan_spec.append((cw, is_all))
note("plan_spec", plan_spec)

CAMS = ["front", "wrist"]
STRIDE = {"front": 2, "wrist": 4}

# ---------------------------------------------------------------- calibration convention
conv = 0
table_z = None
try:
    f0 = video[0] if N > 0 else get_observation()
    cam = f0.cameras["front"]
    cal = calib_of(cam)
    rgb = to_rows(cam.rgb)
    dep = to_depth_rows(cam.depth_m)
    H = len(rgb)
    W = len(rgb[0])
    zs = [[], []]
    for y in range(H // 3, H, 6):
        for x in range(0, W, 6):
            p = rgb[y][x]
            d = float(dep[y][x])
            if d > 0.05 and d < 10.0 and classify(int(p[0]), int(p[1]), int(p[2])) == 0:
                for cv in (0, 1):
                    zs[cv].append(project(x + 0.5, y + 0.5, d, cal, cv)[2])
    spread = []
    for cv in (0, 1):
        spread.append(quantile(zs[cv], 0.9) - quantile(zs[cv], 0.1))
    meds = [median(zs[0]), median(zs[1])]
    conv = 0 if spread[0] <= spread[1] else 1
    if meds[conv] > cal["t"][2] and meds[1 - conv] < cal["t"][2]:
        conv = 1 - conv
    table_z = median(zs[conv])
    note("calib", {"spread": spread, "meds": meds, "conv": conv, "table_z": table_z, "cam_z": cal["t"][2], "n": len(zs[0]), "H": H, "W": W})
except Exception as exc:
    note("calib_err", str(exc))

ZLO = 0.004
ZHI = 0.15


def frame_cells(obs, cams, static_masks):
    """Per camera: list of (gx, gy, cls) for non-zero, non-static cells; plus calib and depth rows."""
    out = {}
    for cname in cams:
        try:
            cam = obs.cameras[cname]
        except Exception:
            continue
        s = STRIDE[cname]
        rgb = to_rows(cam.rgb)
        H = len(rgb)
        W = len(rgb[0])
        smask = static_masks.get(cname) if static_masks is not None else None
        cells = []
        for gy in range(H // s):
            row = rgb[gy * s]
            srow = smask[gy] if smask is not None else None
            for gx in range(W // s):
                if srow is not None and srow[gx]:
                    continue
                p = row[gx * s]
                c = classify(p[0], p[1], p[2])
                if c != 0:
                    cells.append((gx, gy, c))
        out[cname] = {"cells": cells, "cam": cam}
    return out


def cells_to_xyz(fc, cname, wanted):
    cam = fc[cname]["cam"]
    cal = calib_of(cam)
    dep = to_depth_rows(cam.depth_m)
    s = STRIDE[cname]
    pts = []
    for cell in fc[cname]["cells"]:
        if cell[2] != wanted:
            continue
        d = float(dep[cell[1] * s][cell[0] * s])
        if not (d > 0.02 and d < 10.0):
            continue
        p = project(cell[0] * s + 0.5, cell[1] * s + 0.5, d, cal, conv)
        if table_z is not None and (p[2] < table_z + ZLO or p[2] > table_z + ZHI):
            continue
        pts.append((p[0], p[1], p[2], cname))
    return pts


# ---------------------------------------------------------------- static masks (video)
static_masks = {}
if N > 0:
    samp = list(range(0, N, 4))
    for cname in CAMS:
        try:
            cam = video[0].cameras[cname]
        except Exception:
            continue
        s = STRIDE[cname]
        H = len(cam.rgb)
        W = len(cam.rgb[0])
        cw = [[0] * (W // s) for y in range(H // s)]
        cc = [[0] * (W // s) for y in range(H // s)]
        for fi in samp:
            rgb = to_rows(video[fi].cameras[cname].rgb)
            for gy in range(H // s):
                row = rgb[gy * s]
                for gx in range(W // s):
                    p = row[gx * s]
                    c = classify(p[0], p[1], p[2])
                    if c == 1:
                        cw[gy][gx] += 1
                    elif c >= 2:
                        cc[gy][gx] += 1
        lim = 0.8 * len(samp)
        mask = [[(cw[gy][gx] >= lim or cc[gy][gx] >= lim) for gx in range(W // s)] for gy in range(H // s)]
        static_masks[cname] = mask
        note("static_" + cname, sum(sum(1 for v in row if v) for row in mask))

# ---------------------------------------------------------------- main video pass
white_xy = []      # per frame: list of (x, y, z, cam)
white_cnt = []     # per frame: count of non-static whitish cells (all cams)
frame_store = []   # per frame: cells per cam (kept small: cells only)
for fi in range(N):
    fcells = frame_cells(video[fi], CAMS, static_masks)
    cnt = 0
    for cname in fcells:
        cnt += sum(1 for c in fcells[cname]["cells"] if c[2] == 1)
    white_cnt.append(cnt)
    pts = []
    for cname in fcells:
        pts.extend(cells_to_xyz(fcells, cname, 1))
    white_xy.append(pts)
    frame_store.append(dict((cname, fcells[cname]["cells"]) for cname in fcells))
note("white_cnt_sample", white_cnt[::32])

# cover frame
fc_idx = None
fb_idx = None
if N > 0:
    late = white_cnt[N // 2:]
    ref = median(late) if len(late) > 0 else 0
    for i in range(N):
        if ref > 0 and white_cnt[i] >= 0.5 * ref:
            fc_idx = i
            break
    if fc_idx is not None:
        for i in range(fc_idx - 1, -1, -1):
            if any(any(c[2] >= 2 for c in frame_store[i][cn]) for cn in frame_store[i]):
                fb_idx = i
                break
note("cover_frame", fc_idx)
note("cube_frame", fb_idx)

# ---------------------------------------------------------------- cubes at the pre-cover frame
cubes = []
if fb_idx is not None:
    vb = frame_cells(video[fb_idx], CAMS, static_masks)
    cov_i = min(N - 1, fc_idx + 3)
    vcov = frame_store[cov_i]
    for cname in vb:
        s = STRIDE[cname]
        covset = set((c[0], c[1]) for c in vcov.get(cname, []) if c[2] == 1)
        colored = {}
        for c in vb[cname]["cells"]:
            if c[2] >= 2:
                colored[(c[0], c[1])] = c[2]
        nwhite_cells = 0
        cam = vb[cname]["cam"]
        cal = calib_of(cam)
        dep = to_depth_rows(cam.depth_m)
        if white_cnt[fb_idx] < 0.2 * ref:
            for c in vb[cname]["cells"]:
                if c[2] == 1:
                    colored[(c[0], c[1])] = 1
        for comp in components(list(colored.keys())):
            if len(comp) < 2:
                continue
            votes = {}
            for p in comp:
                votes[colored[p]] = votes.get(colored[p], 0) + 1
            best = None
            for kk in votes:
                if best is None or votes[kk] > votes[best]:
                    best = kk
            cov = sum(1 for p in comp if p in covset) / float(len(comp))
            xyz = []
            for p in comp:
                d = float(dep[p[1] * s][p[0] * s])
                if d > 0.02 and d < 10.0:
                    xyz.append(project(p[0] * s + 0.5, p[1] * s + 0.5, d, cal, conv))
            if len(xyz) == 0:
                continue
            cx = median([q[0] for q in xyz])
            cy = median([q[1] for q in xyz])
            cz = median([q[2] for q in xyz])
            rec = {"color": COLOR_NAMES[best], "xy": (cx, cy), "z": cz, "cam": cname, "n": len(comp),
                   "cov": round(cov, 2), "px": (round(sum(p[0] for p in comp) * s / len(comp)), round(sum(p[1] for p in comp) * s / len(comp)))}
            if best == 1:
                cov = 1.0 if len(comp) <= 60 // s else 0.0
            ok = cov >= 0.5 and (table_z is None or cz < table_z + 0.08)
            rec["ok"] = ok
            cubes.append(rec)
note("cube_candidates", [{"color": c["color"], "xy": [round(c["xy"][0], 3), round(c["xy"][1], 3)], "z": round(c["z"], 3),
                          "cam": c["cam"], "n": c["n"], "cov": c["cov"], "px": c["px"], "ok": c["ok"]} for c in cubes])

# merge cubes across cameras (same colour, close in XY)
MERGE_D = 0.03
tracks = []
for c in cubes:
    if not c["ok"]:
        continue
    merged = False
    for t in tracks:
        if t["color"] == c["color"] and math.hypot(t["xy"][0] - c["xy"][0], t["xy"][1] - c["xy"][1]) < MERGE_D:
            t["cams"].append(c["cam"])
            if c["cam"] == "front":
                t["xy"] = c["xy"]
            merged = True
            break
    if not merged:
        tracks.append({"color": c["color"], "xy": c["xy"], "cams": [c["cam"]]})
note("tracks_init", [{"color": t["color"], "xy": [round(t["xy"][0], 3), round(t["xy"][1], 3)], "cams": t["cams"]} for t in tracks])

# ---------------------------------------------------------------- container scale + extra tracks
centers = [t["xy"] for t in tracks]
rad = None
gate = None
min_pts = 2
if fc_idx is not None and len(centers) > 0:
    pts0 = white_xy[fc_idx]
    grp = assign_points(pts0, centers, 10.0)
    rms = []
    for gi in range(len(grp)):
        g = grp[gi]
        if len(g) >= 3:
            mx = sum(p[0] for p in g) / len(g)
            my = sum(p[1] for p in g) / len(g)
            rms.append(math.sqrt(sum((p[0] - mx) ** 2 + (p[1] - my) ** 2 for p in g) / len(g)))
    rad = median(rms) if len(rms) > 0 else None
    if rad is not None:
        gate = 2.5 * rad
        # leftover points far from every cube track -> untracked containers
        left = []
        for p in pts0:
            dmin = min(math.hypot(p[0] - c[0], p[1] - c[1]) for c in centers)
            if dmin > 2.5 * rad:
                left.append(p)
        # cluster leftover by grid linking at rad
        cellmap = {}
        for p in left:
            key = (int(math.floor(p[0] / rad)), int(math.floor(p[1] / rad)))
            cellmap.setdefault(key, []).append(p)
        cnts = [len(g) for g in grp if len(g) > 0]
        med_cnt = median(cnts) if len(cnts) > 0 else 0
        for comp in components(list(cellmap.keys())):
            members = []
            for key in comp:
                members.extend(cellmap[key])
            if med_cnt > 0 and len(members) >= 0.4 * med_cnt:
                cx = sum(p[0] for p in members) / len(members)
                cy = sum(p[1] for p in members) / len(members)
                tracks.append({"color": "unknown", "xy": (cx, cy), "cams": ["leftover"]})
                centers.append((cx, cy))
        min_pts = max(2, int(0.15 * med_cnt))
note("scale", {"rad": rad, "gate": gate, "min_pts": min_pts, "K": len(centers)})

# ---------------------------------------------------------------- track through the video
hist = []
if gate is not None:
    for fi in range(fc_idx, N):
        centers, counts = track_step(white_xy[fi], centers, gate, min_pts, 4)
        centers = recover_lost(centers, counts, gate, min_pts, rad, fi)
        if (fi - fc_idx) % 24 == 0 or fi == N - 1:
            hist.append([fi, [[round(c[0], 3), round(c[1], 3)] for c in centers], counts])
note("video_track", hist)

# compact dump of projected container points (mm), every 2nd frame, for offline replay
dump = []
try:
    for fi in range(0, N, 2):
        pts = white_xy[fi]
        stepk = max(1, len(pts) // 120)
        dump.append([fi, [[int(round(p[0] * 1000)), int(round(p[1] * 1000)), int(round(p[2] * 1000)), 0 if p[3] == "front" else 1] for p in pts[::stepk]]])
except Exception as exc:
    note("dump_err", str(exc))
note("dump_video", dump)

# static XY cells (pillars / static objects) and tall obstacles, from the pre-cover frame's depth
static_xy = set()
XYCELL = 0.01
obstacles = []
obstacle_top = None
if fb_idx is not None and table_z is not None:
    try:
        vfr = video[fb_idx]
        rpos = vfr.robot_state.end_effector_poses["primary"].position
        ree = (float(rpos[0]), float(rpos[1]))
        note("video_ee", [round(float(v), 3) for v in rpos])
        for cname in CAMS:
            cam = vfr.cameras[cname]
            cal = calib_of(cam)
            dep = to_depth_rows(cam.depth_m)
            s = STRIDE[cname]
            H = len(dep)
            W = len(dep[0])
            for gy in range(H // s):
                for gx in range(W // s):
                    d = float(dep[gy * s][gx * s])
                    if not (d > 0.02 and d < 10.0):
                        continue
                    q = project(gx * s + 0.5, gy * s + 0.5, d, cal, conv)
                    if q[2] < table_z + 0.045:
                        continue
                    # skip the robot: near the base->EE segment in XY
                    ax = ree[0]
                    ay = ree[1]
                    L2 = ax * ax + ay * ay
                    tt = 0.0
                    if L2 > 0:
                        tt = max(0.0, min(1.0, (q[0] * ax + q[1] * ay) / L2))
                    dseg = math.hypot(q[0] - tt * ax, q[1] - tt * ay)
                    if dseg < 0.12:
                        continue
                    if q[2] < table_z + ZHI:
                        kx = int(math.floor(q[0] / XYCELL))
                        ky = int(math.floor(q[1] / XYCELL))
                        for ddx in (-1, 0, 1):
                            for ddy in (-1, 0, 1):
                                static_xy.add((kx + ddx, ky + ddy))
                    if q[2] > table_z + 0.10:
                        obstacles.append((q[0], q[1], q[2]))
        if len(obstacles) >= 10:
            obstacle_top = quantile([o[2] for o in obstacles], 0.98)
    except Exception as exc:
        note("static_xy_err", str(exc))
note("static_xy_cells", len(static_xy))
note("obstacles", {"n": len(obstacles), "top": obstacle_top,
                   "xy_min": [round(min(o[0] for o in obstacles), 3), round(min(o[1] for o in obstacles), 3)] if len(obstacles) > 0 else None,
                   "xy_max": [round(max(o[0] for o in obstacles), 3), round(max(o[1] for o in obstacles), 3)] if len(obstacles) > 0 else None})
obst_xy = obstacles[::max(1, len(obstacles) // 200)] if obstacle_top is not None else []


def live_points(obs):
    fcells = frame_cells(obs, CAMS, {"front": static_masks.get("front")})
    pts = []
    for cname in fcells:
        for p in cells_to_xyz(fcells, cname, 1):
            key = (int(math.floor(p[0] / XYCELL)), int(math.floor(p[1] / XYCELL)))
            if cname != "front" and key in static_xy:
                continue
            pts.append(p)
    return pts


# ---------------------------------------------------------------- live continuation until settled
terminated = False
state = get_robot_state()
hold = state.joint_positions["primary"]
hold_list = [float(v) for v in hold]
grip_now = float(state.gripper_positions["primary"])
live_hist = []
last_pts = []
if gate is not None:
    obs = get_observation()
    last_pts = live_points(obs)
    centers, counts = track_step(last_pts, centers, gate, min_pts, 4)
    window = [list(centers)]
    settled_at = None
    for tick in range(240):
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=hold_list, gripper_position=grip_now,
            embodiment=state.embodiment)}))
        try:
            if reply.terminated or reply.truncated:
                terminated = True
                ev("terminated during settle tick %d" % tick)
                break
        except Exception:
            pass
        try:
            obs = reply.observation
            if obs is None:
                obs = get_observation()
        except Exception:
            obs = get_observation()
        last_pts = live_points(obs)
        centers, counts = track_step(last_pts, centers, gate, min_pts, 4)
        centers = recover_lost(centers, counts, gate, min_pts, rad, 10000 + tick)
        window.append(list(centers))
        if len(window) > 12:
            window = window[-12:]
        if tick % 10 == 0:
            live_hist.append([tick, [[round(c[0], 3), round(c[1], 3)] for c in centers], counts])
        if tick >= 20 and len(window) >= 12:
            mv = 0.0
            for i in range(len(centers)):
                mv = max(mv, math.hypot(window[-1][i][0] - window[0][i][0], window[-1][i][1] - window[0][i][1]))
            if mv < 0.1 * rad:
                settled_at = tick
                break
    note("settled_at", settled_at)
note("live_track", live_hist)
note("recoveries", RECOVER[:60])
try:
    note("dump_live_last", [[int(round(p[0] * 1000)), int(round(p[1] * 1000)), int(round(p[2] * 1000)), 0 if p[3] == "front" else 1] for p in last_pts[::max(1, len(last_pts) // 300)]])
except Exception as exc:
    note("dump_live_err", str(exc))
note("final_centers", [{"color": tracks[i]["color"], "xy": [round(centers[i][0], 4), round(centers[i][1], 4)]} for i in range(len(centers))])

# ---------------------------------------------------------------- container geometry from final points
def container_geom(i):
    if gate is None:
        return None
    groups = assign_points(last_pts, centers, gate)
    g = groups[i]
    if len(g) < 3:
        return None
    zs = [p[2] for p in g]
    top = quantile(zs, 0.9)
    topg = [p for p in g if p[2] > top - 0.012]
    if len(topg) < 3:
        topg = g
    cx = median([p[0] for p in topg])
    cy = median([p[1] for p in topg])
    best = None
    for k in range(0, 90, 3):
        a = math.radians(k)
        ca = math.cos(a)
        sa = math.sin(a)
        u = [(p[0] - cx) * ca + (p[1] - cy) * sa for p in topg]
        v = [-(p[0] - cx) * sa + (p[1] - cy) * ca for p in topg]
        eu = quantile(u, 0.95) - quantile(u, 0.05)
        evv = quantile(v, 0.95) - quantile(v, 0.05)
        area = eu * evv
        if best is None or area < best[0]:
            best = (area, a, eu, evv)
    return {"xy": (cx, cy), "top": top, "yaw": best[1], "ext": (best[2], best[3]), "n": len(g), "ntop": len(topg)}


# ---------------------------------------------------------------- choose pick order
used = set()
order = []
for spec in plan_spec:
    cw = spec[0]
    cand = [i for i in range(len(tracks)) if tracks[i]["color"] == cw and i not in used]
    if len(cand) == 0:
        cand = [i for i in range(len(tracks)) if tracks[i]["color"] == "unknown" and i not in used]
        ev("no track for %s; fallback unknown %s" % (cw, str(cand)))
    if len(cand) == 0:
        ev("no candidate for " + cw)
        continue
    if spec[1]:
        rest = list(cand)
        while len(rest) > 0:
            ref = centers[order[-1][0]] if len(order) > 0 else centers[rest[0]]
            rest = sorted(rest, key=lambda j: math.hypot(centers[j][0] - ref[0], centers[j][1] - ref[1]))
            order.append((rest[0], cw))
            used.add(rest[0])
            rest = rest[1:]
    else:
        order.append((cand[0], cw))
        used.add(cand[0])
note("pick_order", [[o[0], o[1]] for o in order])

# ---------------------------------------------------------------- projection cross-check (diagnostic)
def world_to_px(p, cal):
    R = cal["R"]
    t = cal["t"]
    dx = p[0] - t[0]
    dy = p[1] - t[1]
    dz = p[2] - t[2]
    xc = R[0][0] * dx + R[1][0] * dy + R[2][0] * dz
    yc = R[0][1] * dx + R[1][1] * dy + R[2][1] * dz
    zc = R[0][2] * dx + R[1][2] * dy + R[2][2] * dz
    if conv == 1:
        yc = -yc
        zc = -zc
    if zc <= 0:
        return None
    return (cal["fx"] * xc / zc + cal["cx"], cal["fy"] * yc / zc + cal["cy"])


if len(order) > 0 and gate is not None and not terminated:
    try:
        g0 = assign_points(last_pts, centers, gate)[order[0][0]]
        zt = median([p[2] for p in g0]) if len(g0) > 0 else table_z + 0.02
        calf = calib_of(get_observation().cameras["front"])
        px = world_to_px((centers[order[0][0]][0], centers[order[0][0]][1], zt), calf)
        chk = {"px": [round(px[0], 1), round(px[1], 1)] if px is not None else None, "mine": [round(centers[order[0][0]][0], 4), round(centers[order[0][0]][1], 4), round(zt, 4)]}
        if px is not None:
            seg = segment_points("front", [[int(round(px[0])), int(round(px[1]))]])
            chk["seg_ok"] = bool(seg.ok)
            if seg.ok and len(seg.segmentations) > 0:
                cand = seg.segmentations[0]
                try:
                    chk["box"] = [round(float(v), 1) for v in cand.box_xyxy]
                except Exception:
                    pass
                pc = mask_to_point_cloud(cand, "front", target_frame=get_robot_state().base_frame)
                pp = pc.points
                n = len(pp)
                chk["pc_n"] = n
                if n > 0:
                    chk["pc_med"] = [round(median([float(q[0]) for q in pp]), 4), round(median([float(q[1]) for q in pp]), 4), round(median([float(q[2]) for q in pp]), 4)]
        note("proj_check", chk)
    except Exception as exc:
        note("proj_check_err", str(exc))

# ---------------------------------------------------------------- picking
ee0 = get_robot_state().end_effector_poses["primary"]
q0 = tuple(float(v) for v in ee0.quaternion_wxyz)
R0 = quat_to_rot(q0)
note("ee0", {"pos": [round(float(v), 3) for v in ee0.position], "quat": [round(v, 3) for v in q0], "z_axis_z": round(R0[2][2], 3)})
if R0[2][2] > -0.8:
    q0 = (0.0, 1.0, 0.0, 0.0)
    R0 = quat_to_rot(q0)
    ev("ee not pointing down; using (0,1,0,0)")
yaw_fing0 = math.atan2(R0[1][1], R0[0][1])   # hand y-axis (finger closing axis) heading
base_frame = get_robot_state().base_frame


def wrap90(a):
    while a > math.pi / 4:
        a -= math.pi / 2
    while a < -math.pi / 4:
        a += math.pi / 2
    return a


def grasp_quat(geom, idx):
    d0 = wrap90(geom["yaw"] - yaw_fing0)
    best = None
    for d in (d0, d0 + math.pi / 2 if d0 < 0 else d0 - math.pi / 2):
        fa = yaw_fing0 + d
        clear = 1.0
        for j in range(len(centers)):
            if j == idx:
                continue
            for sgn in (1.0, -1.0):
                fx = geom["xy"][0] + sgn * math.cos(fa) * (0.5 * max(geom["ext"]) + 0.015)
                fy = geom["xy"][1] + sgn * math.sin(fa) * (0.5 * max(geom["ext"]) + 0.015)
                clear = min(clear, math.hypot(fx - centers[j][0], fy - centers[j][1]))
        for o in obst_xy:
            for sgn in (1.0, -1.0):
                fx = geom["xy"][0] + sgn * math.cos(fa) * (0.5 * max(geom["ext"]) + 0.015)
                fy = geom["xy"][1] + sgn * math.sin(fa) * (0.5 * max(geom["ext"]) + 0.015)
                clear = min(clear, math.hypot(fx - o[0], fy - o[1]) + 0.02)
        if best is None or clear > best[0] + 0.005:
            best = (clear, d)
    d = best[1]
    qz = (math.cos(d / 2), 0.0, 0.0, math.sin(d / 2))
    return quat_mul(qz, q0), d, best[0]


def is_term(res):
    try:
        return bool(res.terminated) or bool(res.truncated)
    except Exception:
        return False


def mv(pos, quat, label, tol=0.01, steps=150):
    res = move_to_pose(Pose(pos, quat, base_frame), tolerance=tol, max_steps=steps)
    try:
        ev("%s ok=%s st=%s n=%s" % (label, str(res.ok), str(res.status), str(res.steps_executed)))
    except Exception:
        ev(label + " done")
    return res


picks = []
if not terminated:
    for oi in range(len(order)):
        idx = order[oi][0]
        geom = container_geom(idx)
        if geom is None:
            ev("no geometry for track %d" % idx)
            continue
        quat, dyaw, clear = grasp_quat(geom, idx)
        h = geom["top"] - table_z
        gz = geom["top"] - min(0.02, 0.45 * h)
        rec = {"track": idx, "color": order[oi][1], "xy": [round(geom["xy"][0], 4), round(geom["xy"][1], 4)],
               "top": round(geom["top"], 4), "h": round(h, 4), "yaw": round(geom["yaw"], 3), "ext": [round(geom["ext"][0], 4), round(geom["ext"][1], 4)],
               "dyaw": round(dyaw, 3), "clear": round(clear, 4), "n": geom["n"]}
        picks.append(rec)
        r = open_gripper()
        if is_term(r):
            terminated = True
            break
        if obstacle_top is not None:
            tz = min(0.65, max(geom["top"] + 0.10, obstacle_top + 0.08))
            cur = get_robot_state().end_effector_poses["primary"]
            cq = tuple(float(v) for v in cur.quaternion_wxyz)
            if float(cur.position[2]) < tz - 0.02:
                r = mv((float(cur.position[0]), float(cur.position[1]), tz), cq, "rise%d" % oi, 0.02, 200)
                if is_term(r):
                    terminated = True
                    break
            r = mv((geom["xy"][0], geom["xy"][1], tz), quat, "over%d" % oi, 0.02, 200)
            if is_term(r):
                terminated = True
                break
        r = mv((geom["xy"][0], geom["xy"][1], geom["top"] + 0.10), quat, "pre%d" % oi, 0.01, 200)
        if is_term(r):
            terminated = True
            break
        r = mv((geom["xy"][0], geom["xy"][1], gz), quat, "down%d" % oi, 0.008, 120)
        if is_term(r):
            terminated = True
            break
        r = close_gripper()
        try:
            rec["grip_after_close"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
        except Exception:
            pass
        if is_term(r):
            terminated = True
            break
        r = mv((geom["xy"][0], geom["xy"][1], geom["top"] + 0.12), quat, "lift%d" % oi, 0.01, 150)
        if is_term(r):
            terminated = True
            rec["terminated_on_lift"] = True
            break
        try:
            rec["grip_after_lift"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
            rec["ee_after_lift"] = [round(float(v), 3) for v in get_robot_state().end_effector_poses["primary"].position]
        except Exception:
            pass
        r = mv((geom["xy"][0], geom["xy"][1], gz + 0.005), quat, "place%d" % oi, 0.01, 150)
        if is_term(r):
            terminated = True
            break
        r = open_gripper()
        if is_term(r):
            terminated = True
            break
        r = mv((geom["xy"][0], geom["xy"][1], geom["top"] + 0.10), quat, "up%d" % oi, 0.015, 120)
        if is_term(r):
            terminated = True
            break
note("picks", picks)
note("terminated", terminated)
try:
    note("final_t", float(get_robot_state().timestamp_s))
except Exception:
    pass
result = report
