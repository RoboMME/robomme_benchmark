# VideoRepick v00_initial -- ASPIRE CaP program (robomme_plus/task_13).
#
# Goal (from get_task_context().language at runtime): watch the conditioning video,
# identify the object the robot picked up, then pick up and place down the SAME object
# twice, finally press the button.
#
# Strategy (all per-episode quantities perceived at runtime):
#  1. Self-calibrate the depth->robot_base projection convention on the live front camera
#     (compare against mask_to_point_cloud and table planarity).
#  2. For every conditioning-video frame, project front+wrist depth into robot_base and
#     keep points in a height band above the table (colour-agnostic: appearance varies).
#     Exclude points near the recorded end effector (robot fingers / carried object).
#  3. Cells occupied through most of the shuffle are static fixtures (button, occluders).
#  4. Count objects in the first frames, then track all of them with gated nearest-centre
#     assignment + constant-velocity prediction. The picked object is the track under the
#     recorded end effector when the recorded gripper starts closing; while carried, the
#     track is pinned to the end-effector xy; at release it resumes from there.
#  5. Match tracks to the live scene, then pick/lift/place the target twice, press button.
#
# Heavy instrumentation: everything lands in `result` even if a phase fails.

import math

result = {"version": "v00_initial", "errors": [], "phase": "start", "motions": []}
REPORT = result
MOTIONS = result["motions"]


def err(tag, exc):
    REPORT["errors"].append(str(tag) + ": " + str(exc)[:400])


def sstr(value, limit=300):
    try:
        return str(value)[:limit]
    except Exception:
        return "?"


def r4(v):
    return round(float(v), 4)


TOLIST = [0]


def rows2d(arr):
    try:
        return arr.tolist()
    except Exception as exc:
        TOLIST[0] = TOLIST[0] + 1
        if TOLIST[0] <= 2:
            err("tolist", exc)
    return [[float(x) for x in row] for row in arr]


def quat_to_mat(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        n = 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def argbest(items, scores, largest):
    """Return the item with the smallest (or largest) score; None if empty."""
    best = None
    bs = None
    for k in range(len(items)):
        sc = scores[k]
        if bs is None or (sc > bs if largest else sc < bs):
            bs = sc
            best = items[k]
    return best


def quantile(values, frac):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * frac)]


# ---------------------------------------------------------------------------
# Phase A: context, metadata, live observation
# ---------------------------------------------------------------------------
ctx = get_task_context()
REPORT["goal"] = sstr(ctx.language, 400)
MAX_STEPS = None
try:
    md = robomme_plus.get_task_metadata()
    REPORT["task_metadata"] = {str(k): sstr(md[k]) for k in md}
    try:
        MAX_STEPS = int(md["max_steps"])
    except Exception as exc:
        err("max_steps", exc)
except Exception as exc:
    err("task_metadata", exc)
try:
    cmd = robomme_plus.get_controller_metadata()
    REPORT["controller_metadata"] = {str(k): sstr(cmd[k]) for k in cmd}
except Exception as exc:
    err("controller_metadata", exc)

obs0 = get_observation()
state0 = get_robot_state()
BASE = state0.base_frame
CAMS = [name for name in obs0.cameras]
REPORT["cameras"] = CAMS
REPORT["base_frame"] = sstr(BASE)
REPORT["embodiment"] = sstr(state0.embodiment)
EE0 = state0.end_effector_poses["primary"]
EE0_POS = [float(v) for v in EE0.position]
EE0_Q = [float(v) for v in EE0.quaternion_wxyz]
REPORT["ee0"] = {"pos": [r4(v) for v in EE0_POS], "q": [r4(v) for v in EE0_Q],
                 "gripper": r4(state0.gripper_positions["primary"]),
                 "t": r4(state0.timestamp_s)}
FRONT = "front" if "front" in CAMS else CAMS[0]
WRIST = "wrist" if "wrist" in CAMS else None
try:
    REPORT["front_shape"] = [int(v) for v in obs0.cameras[FRONT].depth_m.shape]
    REPORT["front_K"] = [[r4(v) for v in row] for row in rows2d(obs0.cameras[FRONT].intrinsics)]
    REPORT["front_pose"] = [r4(v) for v in obs0.cameras[FRONT].camera_pose.position] + \
        [r4(v) for v in obs0.cameras[FRONT].camera_pose.quaternion_wxyz]
    REPORT["front_pose_frame"] = sstr(obs0.cameras[FRONT].camera_pose.frame)
except Exception as exc:
    err("front_info", exc)

# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------
CONVS = ["cv", "gl", "ros", "cvr", "glr", "rosr"]
RAYCACHE = {}


def base_vec(conv, xn, yn):
    if conv[0:2] == "cv":
        b = (xn, yn, 1.0)
    elif conv[0:2] == "gl":
        b = (xn, -yn, -1.0)
    else:
        b = (1.0, -xn, -yn)
    if len(conv) == 4 or conv == "cvr" or conv == "glr":
        s = 1.0 / math.sqrt(1.0 + xn * xn + yn * yn)
        b = (b[0] * s, b[1] * s, b[2] * s)
    return b


def cam_geom(cam):
    K = rows2d(cam.intrinsics)
    pose = cam.camera_pose
    pos = [float(v) for v in pose.position]
    q = [float(v) for v in pose.quaternion_wxyz]
    shape = cam.depth_m.shape
    return K, pos, q, int(shape[0]), int(shape[1])


def rays_for(cam_name, cam, conv, stride):
    K, pos, q, H, W = cam_geom(cam)
    key = conv + "|" + str(stride) + "|" + ",".join([str(round(v, 5)) for v in pos + q]) + \
        "|" + ",".join([str(round(K[0][0], 3)), str(round(K[1][1], 3)), str(round(K[0][2], 3)),
                        str(round(K[1][2], 3))])
    slot = cam_name + "|" + conv + "|" + str(stride)
    hit = RAYCACHE.get(slot)
    if hit is not None and hit[0] == key:
        return hit[1], pos
    Rm = quat_to_mat(q)
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    rows = []
    for v in range(stride // 2, H, stride):
        us = []
        wx = []
        wy = []
        wz = []
        yn = (v - cy) / fy
        for u in range(stride // 2, W, stride):
            xn = (u - cx) / fx
            b = base_vec(conv, xn, yn)
            us.append(u)
            wx.append(Rm[0][0] * b[0] + Rm[0][1] * b[1] + Rm[0][2] * b[2])
            wy.append(Rm[1][0] * b[0] + Rm[1][1] * b[1] + Rm[1][2] * b[2])
            wz.append(Rm[2][0] * b[0] + Rm[2][1] * b[1] + Rm[2][2] * b[2])
        rows.append((v, us, wx, wy, wz))
    RAYCACHE[slot] = (key, rows)
    return rows, pos


def pixel_of(P, cam, conv):
    K, pos, q, H, W = cam_geom(cam)
    Rm = quat_to_mat(q)
    d0 = P[0] - pos[0]
    d1 = P[1] - pos[1]
    d2 = P[2] - pos[2]
    c = [Rm[0][i] * d0 + Rm[1][i] * d1 + Rm[2][i] * d2 for i in range(3)]
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    if conv[0:2] == "cv":
        if c[2] <= 1e-6:
            return None
        xn, yn = c[0] / c[2], c[1] / c[2]
    elif conv[0:2] == "gl":
        if c[2] >= -1e-6:
            return None
        d = -c[2]
        xn, yn = c[0] / d, -c[1] / d
    else:
        if c[0] <= 1e-6:
            return None
        d = c[0]
        xn, yn = -c[1] / d, -c[2] / d
    return [round(fx * xn + cx, 1), round(fy * yn + cy, 1)]


def all_z(cam_name, cam, conv, stride):
    rows, pos = rays_for(cam_name, cam, conv, stride)
    D = rows2d(cam.depth_m)
    cz = pos[2]
    zs = []
    for (v, us, wx, wy, wz) in rows:
        drow = D[v]
        for k in range(len(us)):
            d = drow[us[k]]
            if d > 0.02 and d < 4.0:
                zs.append(cz + d * wz[k])
    return zs


def plane_peak(zs):
    if len(zs) < 20:
        return None, 0.0
    bins = {}
    for z in zs:
        b = int(math.floor(z / 0.002))
        bins[b] = bins.get(b, 0) + 1
    best = None
    bestc = -1
    for b in bins:
        c = bins.get(b - 1, 0) + bins[b] + bins.get(b + 1, 0)
        if c > bestc:
            bestc = c
            best = b
    center = (best + 0.5) * 0.002
    near = [z for z in zs if abs(z - center) < 0.004]
    return quantile(near, 0.5), len(near) / float(len(zs))


def band_points(cam_name, cam, conv, stride, zlo, zhi, excl, out):
    rows, pos = rays_for(cam_name, cam, conv, stride)
    D = rows2d(cam.depth_m)
    px, py, pz = pos[0], pos[1], pos[2]
    for (v, us, wx, wy, wz) in rows:
        drow = D[v]
        for k in range(len(us)):
            d = drow[us[k]]
            if d > 0.02 and d < 4.0:
                z = pz + d * wz[k]
                if z > zlo and z < zhi:
                    x = px + d * wx[k]
                    y = py + d * wy[k]
                    keep = True
                    for ex in excl:
                        dx = x - ex[0]
                        dy = y - ex[1]
                        if z > ex[2] and dx * dx + dy * dy < ex[3]:
                            keep = False
                            break
                    if keep:
                        out.append((x, y, z))
    return out


# ---------------------------------------------------------------------------
# Phase B: projection convention self-calibration on the live front camera
# ---------------------------------------------------------------------------
REPORT["phase"] = "calibrate"
CONV = "cv"
TABLE_Z = None
calib = {"planarity": {}, "maskcheck": {}}
REPORT["calibration"] = calib
try:
    fcam = obs0.cameras[FRONT]
    K, cpos, cq, H, W = cam_geom(fcam)
    plan = {}
    for conv in CONVS:
        zs = all_z(FRONT, fcam, conv, 4)
        peak, frac = plane_peak(zs)
        plan[conv] = (peak, frac)
        calib["planarity"][conv] = [None if peak is None else r4(peak), r4(frac)]
    # mask check: sparse grid mask through the public projection
    ref = None
    try:
        grid = [[(v % 16 == 8 and u % 16 == 8) for u in range(W)] for v in range(H)]
        pc = mask_to_point_cloud(grid, FRONT, target_frame=BASE)
        pts = rows2d(pc.points)
        calib["mask_points"] = len(pts)
        if len(pts) > 10:
            ref = [sum(p[i] for p in pts) / len(pts) for i in range(3)]
            calib["mask_centroid"] = [r4(v) for v in ref]
    except Exception as exc:
        err("maskcheck", exc)
    D = rows2d(fcam.depth_m)
    if ref is not None:
        Rm = quat_to_mat(cq)
        for conv in CONVS:
            acc = [0.0, 0.0, 0.0]
            n = 0
            for v in range(8, H, 16):
                for u in range(8, W, 16):
                    d = D[v][u]
                    if d > 0.02 and d < 4.0:
                        b = base_vec(conv, (u - K[0][2]) / K[0][0], (v - K[1][2]) / K[1][1])
                        for i in range(3):
                            acc[i] += cpos[i] + d * (Rm[i][0] * b[0] + Rm[i][1] * b[1] + Rm[i][2] * b[2])
                        n += 1
            if n > 0:
                cen = [a / n for a in acc]
                e = math.sqrt(sum((cen[i] - ref[i]) ** 2 for i in range(3)))
                calib["maskcheck"][conv] = r4(e)
    best = None
    if calib["maskcheck"]:
        mk = [c for c in calib["maskcheck"]]
        best = argbest(mk, [calib["maskcheck"][c] for c in mk], False)
        if calib["maskcheck"][best] > 0.03:
            best = None
    if best is None:
        # a 180-degree flip about the camera x axis (cv <-> gl) maps the table to another
        # horizontal plane ABOVE the camera; only a plane below the camera is the table
        def plan_score(c):
            pk, fr = plan[c]
            if pk is None or pk > cpos[2] - 0.1:
                return 0.0
            return fr
        pk = [c for c in plan]
        best = argbest(pk, [plan_score(c) for c in pk], True)
        calib["method"] = "planarity"
    else:
        calib["method"] = "maskcheck"
    CONV = best
    TABLE_Z = plan[CONV][0]
    calib["chosen"] = CONV
    calib["table_z_live_front"] = None if TABLE_Z is None else r4(TABLE_Z)
    if WRIST is not None:
        zs = all_z(WRIST, obs0.cameras[WRIST], CONV, 4)
        wp, wf = plane_peak(zs)
        calib["table_z_live_wrist"] = [None if wp is None else r4(wp), r4(wf)]
except Exception as exc:
    err("calibrate", exc)
if TABLE_Z is None:
    TABLE_Z = 0.0
    err("table_z", "fallback 0.0")

BAND_LO = TABLE_Z + 0.007
BAND_HI = TABLE_Z + 0.12
STRIDE = {FRONT: 2}
if WRIST is not None:
    STRIDE[WRIST] = 3


def frame_points(obs_like, ee_xyz, carrying):
    excl = []
    if ee_xyz is not None:
        excl.append((ee_xyz[0], ee_xyz[1], ee_xyz[2] - 0.015, 0.12 * 0.12))
        if carrying:
            excl.append((ee_xyz[0], ee_xyz[1], ee_xyz[2] - 0.09, 0.06 * 0.06))
    out = []
    for name in STRIDE:
        try:
            cam = obs_like.cameras[name]
        except Exception:
            continue
        band_points(name, cam, CONV, STRIDE[name], BAND_LO, BAND_HI, excl, out)
    return out


CELL = 0.01


def ckey(x, y, cell):
    return (int(math.floor(x / cell)), int(math.floor(y / cell)))


def components(pts, cell):
    cells = {}
    for p in pts:
        k = ckey(p[0], p[1], cell)
        if k in cells:
            cells[k].append(p)
        else:
            cells[k] = [p]
    seen = set()
    comps = []
    for k in cells:
        if k in seen:
            continue
        stack = [k]
        seen.add(k)
        members = []
        while stack:
            c = stack.pop()
            members.extend(cells[c])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (c[0] + dx, c[1] + dy)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        comps.append(members)
    return comps


def centroid(pts):
    n = float(len(pts))
    return [sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n]


def extent(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def kmeans2(pts):
    # split one merged component into two along its widest axis
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if max(xs) - min(xs) >= max(ys) - min(ys):
        a = [min(xs), sum(ys) / len(ys)]
        b = [max(xs), sum(ys) / len(ys)]
    else:
        a = [sum(xs) / len(xs), min(ys)]
        b = [sum(xs) / len(xs), max(ys)]
    for it in range(8):
        ga = []
        gb = []
        for p in pts:
            da = (p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2
            db = (p[0] - b[0]) ** 2 + (p[1] - b[1]) ** 2
            if da <= db:
                ga.append(p)
            else:
                gb.append(p)
        if not ga or not gb:
            return [pts]
        a = centroid(ga)
        b = centroid(gb)
    return [ga, gb]


GATE = 0.045
MINPTS = 3


def track_update(tracks, pts, dt, pinned_idx, pinned_xy, iters, gate, lost_gate_max):
    """Gated nearest-centre assignment. Returns per-track assigned point lists."""
    preds = []
    for i in range(len(tracks)):
        t = tracks[i]
        if pinned_idx is not None and i == pinned_idx:
            preds.append([pinned_xy[0], pinned_xy[1]])
        else:
            preds.append([t["p"][0] + t["v"][0] * dt, t["p"][1] + t["v"][1] * dt])
    centers = [list(c) for c in preds]
    groups = [[] for i in range(len(tracks))]
    active = [i for i in range(len(tracks)) if not (pinned_idx is not None and i == pinned_idx)]
    g2 = gate * gate
    for it in range(iters):
        groups = [[] for i in range(len(tracks))]
        leftovers = []
        for p in pts:
            best = -1
            bd = g2
            for i in active:
                dx = p[0] - centers[i][0]
                dy = p[1] - centers[i][1]
                d = dx * dx + dy * dy
                if d < bd:
                    bd = d
                    best = i
            if best >= 0:
                groups[best].append(p)
            else:
                leftovers.append(p)
        # lost tracks may reacquire leftover points within an expanded gate
        for i in active:
            if len(groups[i]) < MINPTS and tracks[i]["lost"] > 0 and leftovers:
                lg = min(lost_gate_max, gate + 0.004 * tracks[i]["lost"])
                lg2 = lg * lg
                cand = [p for p in leftovers if (p[0] - centers[i][0]) ** 2 + (p[1] - centers[i][1]) ** 2 < lg2]
                if len(cand) >= MINPTS:
                    cc = centroid(cand)
                    near = [p for p in cand if (p[0] - cc[0]) ** 2 + (p[1] - cc[1]) ** 2 < g2]
                    if len(near) >= MINPTS:
                        groups[i] = near
                        leftovers = [p for p in leftovers if not ((p[0] - cc[0]) ** 2 + (p[1] - cc[1]) ** 2 < g2)]
        for i in active:
            if len(groups[i]) >= MINPTS:
                centers[i] = centroid(groups[i])
    for i in range(len(tracks)):
        t = tracks[i]
        old = t["p"]
        if pinned_idx is not None and i == pinned_idx:
            t["p"] = [pinned_xy[0], pinned_xy[1]]
            t["v"] = [0.0, 0.0]
            t["n"] = 0
            continue
        if len(groups[i]) >= MINPTS:
            new = centers[i]
            if dt > 1e-4:
                vx = (new[0] - old[0]) / dt
                vy = (new[1] - old[1]) / dt
                sp = math.sqrt(vx * vx + vy * vy)
                if sp > 0.4:
                    vx, vy = vx * 0.4 / sp, vy * 0.4 / sp
                t["v"] = [0.5 * t["v"][0] + 0.5 * vx, 0.5 * t["v"][1] + 0.5 * vy]
            t["p"] = new
            t["n"] = len(groups[i])
            t["lost"] = 0
        else:
            t["p"] = preds[i]
            t["v"] = [t["v"][0] * 0.6, t["v"][1] * 0.6]
            t["n"] = len(groups[i])
            t["lost"] = t["lost"] + 1
    return groups


# ---------------------------------------------------------------------------
# Phase C: conditioning video -> identify and track the picked object
# ---------------------------------------------------------------------------
REPORT["phase"] = "video"
TRACKS = None
TARGET = None
STATIC = set()
vid = {}
REPORT["video"] = vid
try:
    video = robomme_plus.get_conditioning_video()
    frames = [f for f in video]
    nf = len(frames)
    vid["frames"] = nf
    ts = []
    grip = []
    ees = []
    for f in frames:
        ts.append(float(f.timestamp_s))
        try:
            rs = f.robot_state
            grip.append(float(rs.gripper_positions["primary"]))
            ees.append([float(v) for v in rs.end_effector_poses["primary"].position])
        except Exception as exc:
            grip.append(None)
            ees.append(None)
    if nf > 0:
        vid["t_first_last"] = [r4(ts[0]), r4(ts[-1])]
        vid["grip_sub"] = [None if g is None else round(g, 3) for g in grip[::8]]
        vid["ee_z_sub"] = [None if e is None else round(e[2], 3) for e in ees[::8]]
        vid["ee_xy_sub"] = [None if e is None else [round(e[0], 3), round(e[1], 3)] for e in ees[::16]]
        # video table height cross-check
        try:
            vz, vfrac = plane_peak(all_z(FRONT, frames[0].cameras[FRONT], CONV, 4))
            vid["table_z_video_front"] = [None if vz is None else r4(vz), r4(vfrac)]
        except Exception as exc:
            err("video_table", exc)
    # --- pick event from the recorded gripper trace
    T_CLOSE = None
    T_REL = None
    gvals = [g for g in grip if g is not None]
    if len(gvals) > 5:
        head = [g for g in grip[: min(20, nf)] if g is not None]
        g_open = max(head) if head else max(gvals)
        for i in range(nf):
            if grip[i] is not None and grip[i] < g_open - 0.08:
                T_CLOSE = i
                break
        if T_CLOSE is not None:
            for i in range(T_CLOSE + 1, nf):
                if grip[i] is not None and grip[i] >= g_open - 0.05:
                    T_REL = i
                    break
        vid["g_open"] = r4(g_open)
    vid["t_close"] = T_CLOSE
    vid["t_release"] = T_REL
    if T_CLOSE is not None:
        vid["ee_at_close"] = None if ees[T_CLOSE] is None else [r4(v) for v in ees[T_CLOSE]]
    if T_REL is not None:
        vid["ee_at_release"] = None if ees[T_REL] is None else [r4(v) for v in ees[T_REL]]

    # --- pass 1: band points per frame
    PTS = []
    for i in range(nf):
        carrying = T_CLOSE is not None and i >= T_CLOSE and (T_REL is None or i <= T_REL + 3)
        try:
            PTS.append(frame_points(frames[i], ees[i], carrying))
        except Exception as exc:
            PTS.append([])
            if len(REPORT["errors"]) < 30:
                err("frame_points_" + str(i), exc)
    vid["pts_count_sub"] = [len(p) for p in PTS[::16]]

    # --- static fixtures over the post-release window
    ws = 0 if T_REL is None else min(nf - 1, T_REL + 10)
    window = list(range(ws, nf))
    occ = {}
    for i in window:
        seen = set()
        for p in PTS[i]:
            k = ckey(p[0], p[1], CELL)
            if k not in seen:
                seen.add(k)
                occ[k] = occ.get(k, 0) + 1
    if len(window) >= 50:
        thr = 0.85 * len(window)
        STATIC = set([k for k in occ if occ[k] >= thr])
    vid["static_cells"] = len(STATIC)
    vid["static_cells_list"] = [[k[0], k[1]] for k in sorted(STATIC)][:80]

    def dyn(pts):
        return [p for p in pts if ckey(p[0], p[1], CELL) not in STATIC]

    # --- object count / init from the first frames
    best_i = 0
    best_comps = []
    last_init = min(nf, 12) if T_CLOSE is None else max(1, min(nf, 12, T_CLOSE - 2))
    counts = []
    for i in range(last_init):
        comps = [c for c in components(dyn(PTS[i]), 0.008) if len(c) >= 5]
        counts.append(len(comps))
        if len(comps) > len(best_comps):
            best_comps = comps
            best_i = i
    vid["init_counts"] = counts
    # split merged components (extent much larger than median)
    if best_comps:
        exts = sorted([extent(c) for c in best_comps])
        med = exts[len(exts) // 2]
        split = []
        for c in best_comps:
            if len(best_comps) >= 3 and extent(c) > 1.8 * med and len(c) >= 10:
                split.extend(kmeans2(c))
            else:
                split.append(c)
        best_comps = split
    TRACKS = []
    for c in best_comps:
        cc = centroid(c)
        TRACKS.append({"p": cc, "v": [0.0, 0.0], "n": len(c), "lost": 0})
    vid["init_frame"] = best_i
    vid["n_objects"] = len(TRACKS)
    vid["init_positions"] = [[r4(t["p"][0]), r4(t["p"][1]), t["n"]] for t in TRACKS]
    vid["init_extents"] = [r4(extent(c)) for c in best_comps]

    # --- tracking
    enc = []
    tgt_path = []
    lost_hist = []
    leftover_max = 0
    for i in range(best_i, nf):
        dt = 0.0 if i == best_i else max(0.0, ts[i] - ts[i - 1])
        if T_CLOSE is not None and i == T_CLOSE and TARGET is None and ees[i] is not None and TRACKS:
            dmin = None
            for j in range(len(TRACKS)):
                d = math.sqrt((TRACKS[j]["p"][0] - ees[i][0]) ** 2 + (TRACKS[j]["p"][1] - ees[i][1]) ** 2)
                if dmin is None or d < dmin:
                    dmin = d
                    TARGET = j
            vid["target_pick_dist"] = r4(dmin)
            vid["target_init_pos"] = [r4(v) for v in TRACKS[TARGET]["p"]]
        pinned = None
        pxy = None
        if TARGET is not None and T_CLOSE is not None and i >= T_CLOSE and (T_REL is None or i < T_REL):
            if ees[i] is not None:
                pinned = TARGET
                pxy = ees[i]
        if TARGET is not None and T_REL is not None and i == T_REL and ees[i] is not None:
            TRACKS[TARGET]["p"] = [ees[i][0], ees[i][1]]
            TRACKS[TARGET]["v"] = [0.0, 0.0]
            TRACKS[TARGET]["lost"] = 1
        pts = dyn(PTS[i])
        groups = track_update(TRACKS, pts, dt, pinned, pxy, 2, GATE, 0.10)
        if TARGET is not None:
            tp = TRACKS[TARGET]["p"]
            dmin = None
            for j in range(len(TRACKS)):
                if j != TARGET:
                    d = math.sqrt((TRACKS[j]["p"][0] - tp[0]) ** 2 + (TRACKS[j]["p"][1] - tp[1]) ** 2)
                    if dmin is None or d < dmin:
                        dmin = d
            if dmin is not None and dmin < 0.035:
                enc.append([i, r4(dmin)])
            if i % 8 == 0:
                tgt_path.append([i, r4(tp[0]), r4(tp[1]), TRACKS[TARGET]["n"]])
        if i % 16 == 0:
            lost_hist.append([i, sum(1 for t in TRACKS if t["lost"] > 0)])
    if TARGET is None and TRACKS:
        # no gripper-close event: use the track under the lowest recorded end effector
        lows = [i for i in range(min(nf, int(0.6 * nf) + 1)) if ees[i] is not None]
        if lows:
            ilow = argbest(lows, [ees[i][2] for i in lows], False)
            jj = list(range(len(TRACKS)))
            TARGET = argbest(jj, [(TRACKS[j]["p"][0] - ees[ilow][0]) ** 2 + (TRACKS[j]["p"][1] - ees[ilow][1]) ** 2 for j in jj], False)
        else:
            TARGET = 0
        vid["target_fallback"] = True
        err("target", "no pick event detected; fallback target " + str(TARGET))
    vid["target_index"] = TARGET
    vid["close_encounters"] = enc[:120]
    vid["n_close_encounters"] = len(enc)
    vid["target_path"] = tgt_path
    vid["lost_hist"] = lost_hist
    if TRACKS:
        vid["final_positions"] = [[r4(t["p"][0]), r4(t["p"][1]), t["n"], t["lost"]] for t in TRACKS]
        try:
            vid["final_pixels_front"] = [pixel_of([t["p"][0], t["p"][1], TABLE_Z + 0.02],
                                                  frames[nf - 1].cameras[FRONT], CONV) for t in TRACKS]
        except Exception as exc:
            err("final_pixels", exc)
    frames = None
    PTS = None
except Exception as exc:
    err("video", exc)

# ---------------------------------------------------------------------------
# Phase D: live matching
# ---------------------------------------------------------------------------
REPORT["phase"] = "live_match"
live = {}
REPORT["live"] = live


def live_points():
    st = get_robot_state()
    ee = [float(v) for v in st.end_effector_poses["primary"].position]
    ob = get_observation()
    pts = frame_points(ob, ee, False)
    return [p for p in pts if ckey(p[0], p[1], CELL) not in STATIC], ob


def target_geometry(tag):
    """Re-associate all tracks with the live scene; return target grasp info."""
    pts, ob = live_points()
    for t in TRACKS:
        t["v"] = [0.0, 0.0]
        if t["lost"] == 0:
            t["lost"] = 1
    groups = track_update(TRACKS, pts, 0.0, None, None, 3, 0.05, 0.10)
    info = {"tag": tag, "n_live_points": len(pts),
            "tracks": [[r4(t["p"][0]), r4(t["p"][1]), len(groups[i])] for i, t in enumerate(TRACKS)]}
    g = groups[TARGET]
    if len(g) >= MINPTS:
        xs = [p[0] for p in g]
        ys = [p[1] for p in g]
        zs = [p[2] for p in g]
        cx = 0.5 * (quantile(xs, 0.05) + quantile(xs, 0.95))
        cy = 0.5 * (quantile(ys, 0.05) + quantile(ys, 0.95))
        top = quantile(zs, 0.9)
        info["bbox_xy"] = [r4(quantile(xs, 0.05)), r4(quantile(xs, 0.95)), r4(quantile(ys, 0.05)), r4(quantile(ys, 0.95))]
        info["source"] = "points"
    else:
        cx, cy = TRACKS[TARGET]["p"][0], TRACKS[TARGET]["p"][1]
        top = TABLE_Z + 0.03
        info["source"] = "track_fallback"
    nb = None
    nbd = None
    for j in range(len(TRACKS)):
        if j != TARGET:
            d = math.sqrt((TRACKS[j]["p"][0] - cx) ** 2 + (TRACKS[j]["p"][1] - cy) ** 2)
            if nbd is None or d < nbd:
                nbd = d
                nb = [TRACKS[j]["p"][0] - cx, TRACKS[j]["p"][1] - cy]
    info["target"] = [r4(cx), r4(cy), r4(top)]
    info["neighbor_dist"] = None if nbd is None else r4(nbd)
    try:
        info["target_pixel_front"] = pixel_of([cx, cy, top], ob.cameras[FRONT], CONV)
        if WRIST is not None:
            info["target_pixel_wrist"] = pixel_of([cx, cy, top], ob.cameras[WRIST], CONV)
    except Exception as exc:
        err("target_pixel", exc)
    live[tag] = info
    return cx, cy, top, nb


BUTTON = None
try:
    if TRACKS and TARGET is not None:
        TX, TY, TTOP, NB = target_geometry("start")
        # overlay of the chosen target in media/overlays for visual verification
        try:
            pix = live["start"].get("target_pixel_front")
            if pix is not None:
                sp = segment_points(FRONT, [[int(pix[0]), int(pix[1])]])
                live["segment_points_ok"] = bool(sp.ok)
        except Exception as exc:
            err("segment_points", exc)
    else:
        err("live_match", "no tracks/target from video")
except Exception as exc:
    err("live_match", exc)

# --- button
REPORT["phase"] = "button_localize"
btn = {}
REPORT["button"] = btn
try:
    cands = []
    for prompt in ("button", "white button", "round button"):
        seg = segment_text(FRONT, prompt)
        if not seg.ok:
            continue
        for cand in seg.segmentations[:5]:
            try:
                pc = mask_to_point_cloud(cand, FRONT, target_frame=BASE)
                pts = rows2d(pc.points)
            except Exception as exc:
                err("button_pc", exc)
                continue
            if len(pts) < 8:
                continue
            zs = [p[2] for p in pts]
            top = quantile(zs, 0.95)
            capp = [p for p in pts if p[2] > top - 0.008]
            cc = centroid(capp)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ext = max(quantile(xs, 0.95) - quantile(xs, 0.05), quantile(ys, 0.95) - quantile(ys, 0.05))
            plaus = (top > TABLE_Z + 0.004) and (top < TABLE_Z + 0.08) and (ext < 0.12)
            cands.append({"prompt": prompt, "score": r4(cand.score), "n": len(pts),
                          "center": [r4(cc[0]), r4(cc[1])], "top": r4(top), "ext": r4(ext),
                          "plausible": plaus})
        if any(c["plausible"] for c in cands):
            break
    btn["candidates"] = cands
    good = [c for c in cands if c["plausible"]]
    if good:
        bestc = argbest(good, [c["score"] for c in good], True)
        BUTTON = [bestc["center"][0], bestc["center"][1], bestc["top"]]
        btn["source"] = "segment_text"
    elif STATIC:
        # fallback: static fixture component nearest to the object tracks
        sc = components([((k[0] + 0.5) * CELL, (k[1] + 0.5) * CELL, 0.0) for k in STATIC], CELL)
        if sc and TRACKS:
            ocx = sum(t["p"][0] for t in TRACKS) / len(TRACKS)
            ocy = sum(t["p"][1] for t in TRACKS) / len(TRACKS)
            sc = [c for c in sc if extent(c) < 0.1]
            if sc:
                c = argbest(sc, [(centroid(c)[0] - ocx) ** 2 + (centroid(c)[1] - ocy) ** 2 for c in sc], False)
                cc = centroid(c)
                BUTTON = [cc[0], cc[1], TABLE_Z + 0.02]
                btn["source"] = "static_fallback"
    btn["chosen"] = None if BUTTON is None else [r4(v) for v in BUTTON]
except Exception as exc:
    err("button", exc)

# ---------------------------------------------------------------------------
# Phase E: manipulation
# ---------------------------------------------------------------------------
REPORT["phase"] = "manipulate"


def is_terminal(r):
    flag = False
    for name in ("terminated", "truncated"):
        try:
            if name == "terminated" and bool(r.terminated):
                flag = True
            if name == "truncated" and bool(r.truncated):
                flag = True
        except Exception:
            pass
    return flag


def record(tag, r):
    rec = {"tag": tag}
    try:
        rec["ok"] = bool(r.ok)
    except Exception:
        rec["ok"] = None
    try:
        rec["status"] = sstr(r.status, 80)
    except Exception:
        pass
    try:
        rec["steps"] = int(r.steps_executed)
    except Exception:
        pass
    try:
        rec["error"] = sstr(r.error, 200) if r.error is not None else None
    except Exception:
        pass
    rec["terminal"] = is_terminal(r)
    try:
        st = get_robot_state()
        rec["t"] = r4(st.timestamp_s)
        rec["ee"] = [r4(v) for v in st.end_effector_poses["primary"].position]
        rec["grip"] = r4(st.gripper_positions["primary"])
    except Exception:
        pass
    MOTIONS.append(rec)
    if rec["terminal"]:
        raise ValueError("EPISODE_TERMINAL after " + tag)
    return rec


# top-down orientation: home orientation if it already points down
Rh = quat_to_mat(EE0_Q)
if Rh[2][2] < -0.94:
    Q_DOWN = list(EE0_Q)
    REPORT["q_down_source"] = "home"
else:
    Q_DOWN = [0.0, 1.0, 0.0, 0.0]
    REPORT["q_down_source"] = "constructed"


def grasp_quat(nb):
    """Yaw the top-down hand so the finger closing axis is perpendicular to the nearest neighbour."""
    if nb is None:
        return list(Q_DOWN)
    Rm = quat_to_mat(Q_DOWN)
    hx, hy = Rm[0][1], Rm[1][1]
    want = math.atan2(nb[1], nb[0]) + math.pi / 2.0
    ang = want - math.atan2(hy, hx)
    while ang > math.pi / 2.0:
        ang -= math.pi
    while ang < -math.pi / 2.0:
        ang += math.pi
    qy = [math.cos(ang / 2.0), 0.0, 0.0, math.sin(ang / 2.0)]
    return quat_mul(qy, Q_DOWN)


def mv(tag, x, y, z, q, tol=0.01, ms=250):
    r = move_to_pose(Pose([x, y, z], q, BASE), tolerance=tol, max_steps=ms)
    return record(tag, r)


def pick_place(k, tx, ty, ttop, nb):
    q = grasp_quat(nb)
    grasp_z = max(TABLE_Z + 0.008, ttop - 0.02)
    hover_z = max(ttop + 0.09, TABLE_Z + 0.12)
    lift_z = grasp_z + 0.13
    cyc = {"k": k, "target": [r4(tx), r4(ty), r4(ttop)], "grasp_z": r4(grasp_z), "q": [r4(v) for v in q]}
    REPORT["cycle_" + str(k)] = cyc
    record("open_" + str(k), open_gripper())
    mv("hover_" + str(k), tx, ty, hover_z, q, 0.01, 250)
    mv("descend_" + str(k), tx, ty, grasp_z, q, 0.006, 200)
    rc = record("close_" + str(k), close_gripper())
    held = rc.get("grip") is not None and rc["grip"] > 0.03
    cyc["held_after_close"] = held
    if not held:
        # verified miss: re-perceive once and retry the grasp
        mv("reup_" + str(k), tx, ty, hover_z, q, 0.01, 200)
        record("reopen_" + str(k), open_gripper())
        tx, ty, ttop, nb = target_geometry("regrasp_" + str(k))
        q = grasp_quat(nb)
        grasp_z = max(TABLE_Z + 0.008, ttop - 0.02)
        lift_z = grasp_z + 0.13
        mv("rehover_" + str(k), tx, ty, hover_z, q, 0.01, 250)
        mv("redescend_" + str(k), tx, ty, grasp_z, q, 0.006, 200)
        rc = record("reclose_" + str(k), close_gripper())
        held = rc.get("grip") is not None and rc["grip"] > 0.03
        cyc["held_after_regrasp"] = held
    rl = mv("lift_" + str(k), tx, ty, lift_z, q, 0.015, 250)
    cyc["held_after_lift"] = rl.get("grip") is not None and rl["grip"] > 0.03
    mv("place_" + str(k), tx, ty, grasp_z + 0.008, q, 0.008, 250)
    record("release_" + str(k), open_gripper())
    mv("retreat_" + str(k), tx, ty, hover_z, q, 0.015, 200)
    TRACKS[TARGET]["p"] = [tx, ty]


def press_button():
    bx, by, btop = BUTTON
    q = list(Q_DOWN)
    mv("btn_hover", bx, by, btop + 0.08, q, 0.01, 250)
    record("btn_close", close_gripper())
    mv("btn_press", bx, by, btop - 0.012, q, 0.004, 80)
    mv("btn_up", bx, by, btop + 0.06, q, 0.015, 150)
    REPORT["button_pressed_once"] = True
    mv("btn_press2", bx, by, btop - 0.022, q, 0.004, 80)
    mv("btn_up2", bx, by, btop + 0.06, q, 0.015, 150)


try:
    if TRACKS and TARGET is not None:
        pick_place(1, TX, TY, TTOP, NB)
        TX, TY, TTOP, NB = target_geometry("after_cycle1")
        pick_place(2, TX, TY, TTOP, NB)
        REPORT["phase"] = "button"
        if BUTTON is not None:
            press_button()
        else:
            err("button", "no button location")
    else:
        err("manipulate", "skipped: no target")
    REPORT["phase"] = "done"
except Exception as exc:
    msg = str(exc)
    if msg.find("EPISODE_TERMINAL") >= 0:
        REPORT["terminal_stop"] = msg[:200]
    else:
        err("manipulate", exc)

try:
    stf = get_robot_state()
    REPORT["final_t"] = r4(stf.timestamp_s)
except Exception as exc:
    err("final_state", exc)
REPORT["max_steps"] = MAX_STEPS
REPORT["tolist_failures"] = TOLIST[0]
result = REPORT
