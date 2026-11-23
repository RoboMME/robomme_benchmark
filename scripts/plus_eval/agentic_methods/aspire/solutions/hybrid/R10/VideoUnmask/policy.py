import math

# VideoUnmask v00 -- memory of which container hides which coloured cube.
# 1. Parse the goal into an ordered list of (colour, all?) steps.
# 2. Self-calibrate the depth->robot-base projection convention on the live
#    front/wrist cameras (table flatness), then scan the conditioning video
#    (front + wrist, RGB-D, per-frame calibration) for raised, saturated pixels
#    -> coloured cube tracks in robot_base XY.
# 3. In the live scene, find the raised container cloud over each target cube
#    track, estimate its top/centre/yaw, refine from the wrist camera above it,
#    grasp the sides top-down, lift, put it back, release.

COLOR_WORDS = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink"]

report = {"version": "v01"}
ctx = get_task_context()
goal = str(ctx.language)
report["goal"] = goal
try:
    meta = robomme_plus.get_task_metadata()
    report["max_steps"] = meta["max_steps"]
    try:
        report["video_sampling"] = meta["conditioning_video_sampling"]
    except Exception:
        pass
except Exception as exc:
    report["meta_error"] = str(exc)[:200]


# ---------------------------------------------------------------- goal parse
def parse_goal(text):
    low = text.lower()
    k = low.find(" then ")
    if k >= 0:
        low = low[k + 6:]
    low = low.replace("followed by", "|").replace("finally", "|").replace(", then", "|")
    steps = []
    for seg in low.split("|"):
        best = None
        best_pos = None
        for word in COLOR_WORDS:
            pos = seg.find(word)
            if pos >= 0 and (best_pos is None or pos < best_pos):
                best = word
                best_pos = pos
        if best is None:
            continue
        plural = ("all " in seg) or ((best + " cubes") in seg) or ("containers" in seg)
        steps.append([best, plural])
    return steps


plan_steps = parse_goal(goal)
report["plan"] = plan_steps


# ---------------------------------------------------------------- colour
def hue_class(r, g, b):
    r = int(r)
    g = int(g)
    b = int(b)
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 40:
        return None
    s = (mx - mn) / float(mx)
    if s < 0.45:
        return None
    d = float(mx - mn)
    if mx == r:
        h = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        h = 60.0 * ((b - r) / d + 2.0)
    else:
        h = 60.0 * ((r - g) / d + 4.0)
    if h < 10.0 or h >= 335.0:
        return "red"
    if h < 42.0:
        return "orange"
    if h < 75.0:
        return "yellow"
    if h < 165.0:
        return "green"
    if h < 200.0:
        return "cyan"
    if h < 262.0:
        return "blue"
    if h < 300.0:
        return "purple"
    return "pink"


# ---------------------------------------------------------------- geometry
def quat_to_mat(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def matmul(a, b):
    out = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j])
        out.append(row)
    return out


# camera-frame conventions: maps OpenCV camera coords (x right, y down, z fwd)
# into the camera_pose frame.
CONVS = {
    "opencv": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "opengl": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
    "ros": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
}


def as_list(arr):
    try:
        return arr.tolist()
    except Exception:
        return arr


def cam_model(cam, conv, ranged):
    K = cam.intrinsics
    fx = float(K[0][0])
    fy = float(K[1][1])
    cx = float(K[0][2])
    cy = float(K[1][2])
    pose = cam.camera_pose
    R = quat_to_mat(pose.quaternion_wxyz)
    t = [float(pose.position[0]), float(pose.position[1]), float(pose.position[2])]
    M = matmul(R, CONVS[conv])
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "M": M, "t": t, "ranged": ranged}


def ray(model, u, v):
    xn = (u - model["cx"]) / model["fx"]
    yn = (v - model["cy"]) / model["fy"]
    if model["ranged"]:
        n = math.sqrt(xn * xn + yn * yn + 1.0)
        return (xn / n, yn / n, 1.0 / n)
    return (xn, yn, 1.0)


def point(model, u, v, d):
    rx, ry, rz = ray(model, u, v)
    M = model["M"]
    t = model["t"]
    px, py, pz = rx * d, ry * d, rz * d
    return (
        M[0][0] * px + M[0][1] * py + M[0][2] * pz + t[0],
        M[1][0] * px + M[1][1] * py + M[1][2] * pz + t[1],
        M[2][0] * px + M[2][1] * py + M[2][2] * pz + t[2],
    )


def z_table(model, H, W):
    # per-pixel coefficient so that world z = d * A[v][u] + t_z
    M = model["M"]
    A = []
    for v in range(H):
        row = []
        for u in range(W):
            rx, ry, rz = ray(model, u, v)
            row.append(M[2][0] * rx + M[2][1] * ry + M[2][2] * rz)
        A.append(row)
    return A


def valid_depth(d):
    return d > 0.05 and d < 5.0


def mode_z(zs, binw=0.004):
    counts = {}
    for z in zs:
        k = int(math.floor(z / binw))
        counts[k] = counts.get(k, 0) + 1
    best = None
    bc = -1
    for k in counts:
        c = counts[k] + counts.get(k - 1, 0) + counts.get(k + 1, 0)
        if c > bc:
            bc = c
            best = k
    return (best + 0.5) * binw, bc


def calib_score(cam, conv, ranged):
    model = cam_model(cam, conv, ranged)
    depth = as_list(cam.depth_m)
    H = len(depth)
    W = len(depth[0])
    zs = []
    for v in range(4, H, 8):
        for u in range(4, W, 8):
            d = float(depth[v][u])
            if valid_depth(d):
                zs.append(point(model, u, v, d)[2])
    if len(zs) < 20:
        return -1.0, 0.0
    zm, c = mode_z(zs)
    inl = 0
    for z in zs:
        if abs(z - zm) < 0.008:
            inl += 1
    frac = inl / float(len(zs))
    if zm > model["t"][2] - 0.05:
        frac = frac - 1.0
    return frac, zm


obs0 = get_observation()
state0 = get_robot_state()
base_frame = state0.base_frame
embodiment = state0.embodiment
cams0 = obs0.cameras
cam_names = [n for n in cams0]
report["cameras"] = cam_names

best_conv = None
best_score = -9.0
calib = []
for conv in ["opencv", "opengl", "ros"]:
    for ranged in [False, True]:
        total = 0.0
        zms = []
        for cn in ["front", "wrist"]:
            if cn in cam_names:
                sc, zm = calib_score(cams0[cn], conv, ranged)
                total += sc
                zms.append(round(zm, 4))
        calib.append([conv, ranged, round(total, 3), zms])
        if total > best_score:
            best_score = total
            best_conv = [conv, ranged]
report["calib"] = calib
report["conv"] = best_conv
CONV = best_conv[0]
RANGED = best_conv[1]

front_model0 = cam_model(cams0["front"], CONV, RANGED)
sc, TABLE_Z = calib_score(cams0["front"], CONV, RANGED)
report["table_z"] = round(TABLE_Z, 4)

# ---------------------------------------------------------------- video scan
A_cache = {}


def pose_key(cam):
    p = cam.camera_pose
    vals = [float(x) for x in p.position] + [float(x) for x in p.quaternion_wxyz]
    return ",".join([str(round(x, 4)) for x in vals])


def scan_frame(cam, cam_label, frame_index, tstamp, out):
    model = cam_model(cam, CONV, RANGED)
    depth = as_list(cam.depth_m)
    rgb = as_list(cam.rgb)
    H = len(depth)
    W = len(depth[0])
    key = cam_label + "|" + pose_key(cam)
    if key not in A_cache:
        A_cache[key] = z_table(model, H, W)
    A = A_cache[key]
    tz = model["t"][2] - TABLE_Z
    groups = {}
    for v in range(H):
        drow = depth[v]
        arow = A[v]
        for u in range(W):
            d = drow[u]
            h = d * arow[u] + tz
            if h > 0.006 and h < 0.10 and d > 0.05:
                px = rgb[v][u]
                cls = hue_class(px[0], px[1], px[2])
                if cls is not None:
                    if cls not in groups:
                        groups[cls] = []
                    groups[cls].append(point(model, u, v, float(d)))
    for cls in groups:
        clusters = []
        for p in groups[cls]:
            placed = False
            for c in clusters:
                mx = c[0] / c[3]
                my = c[1] / c[3]
                if abs(p[0] - mx) < 0.025 and abs(p[1] - my) < 0.025:
                    c[0] += p[0]
                    c[1] += p[1]
                    c[2] += p[2]
                    c[3] += 1
                    placed = True
                    break
            if not placed:
                clusters.append([p[0], p[1], p[2], 1])
        for c in clusters:
            if c[3] >= 4:
                out.append([cls, c[0] / c[3], c[1] / c[3], c[2] / c[3] - TABLE_Z, c[3], cam_label, frame_index, tstamp])


video = robomme_plus.get_conditioning_video()
nvid = len(video)
report["video_frames"] = nvid
detections = []
stride = 1
if nvid > 120:
    stride = 2
if nvid > 260:
    stride = 4
frame_ids = list(range(0, nvid, stride))
if nvid > 0 and frame_ids[-1] != nvid - 1:
    frame_ids.append(nvid - 1)
scan_errors = []
for fi in frame_ids:
    fr = video[fi]
    for cn in ["front", "wrist"]:
        try:
            scan_frame(fr.cameras[cn], cn, fi, float(fr.timestamp_s), detections)
        except Exception as exc:
            if len(scan_errors) < 3:
                scan_errors.append(cn + ":" + str(exc)[:160])
report["scan_errors"] = scan_errors
report["n_detections"] = len(detections)

# tracks: cluster detections across frames/cameras by colour + XY
tracks = []
for det in detections:
    placed = False
    for tr in tracks:
        if tr["cls"] != det[0]:
            continue
        mx = tr["sx"] / tr["w"]
        my = tr["sy"] / tr["w"]
        if abs(det[1] - mx) < 0.03 and abs(det[2] - my) < 0.03:
            wgt = float(det[4])
            tr["sx"] += det[1] * wgt
            tr["sy"] += det[2] * wgt
            tr["sz"] += det[3] * wgt
            tr["w"] += wgt
            tr["frames"][det[6]] = 1
            tr["cams"][det[5]] = 1
            tr["last"] = max(tr["last"], det[7])
            tr["first"] = min(tr["first"], det[7])
            placed = True
            break
    if not placed:
        wgt = float(det[4])
        tracks.append({"cls": det[0], "sx": det[1] * wgt, "sy": det[2] * wgt, "sz": det[3] * wgt,
                       "w": wgt, "frames": {det[6]: 1}, "cams": {det[5]: 1},
                       "first": det[7], "last": det[7]})
cubes = []
for tr in tracks:
    nf = len(tr["frames"])
    if nf >= 2 or tr["w"] >= 15:
        cubes.append({"cls": tr["cls"], "x": tr["sx"] / tr["w"], "y": tr["sy"] / tr["w"],
                      "h": tr["sz"] / tr["w"], "nf": nf, "px": int(tr["w"]),
                      "cams": sorted([c for c in tr["cams"]]),
                      "t0": round(tr["first"], 2), "t1": round(tr["last"], 2)})
report["cube_tracks"] = [[c["cls"], round(c["x"], 3), round(c["y"], 3), round(c["h"], 3), c["nf"], c["px"], c["cams"], c["t0"], c["t1"]] for c in cubes]
report["raw_tracks"] = len(tracks)


# ---------------------------------------------------------------- live cloud
def raised_points(obs, cam_names_use, hmin, hmax):
    pts = []
    for cn in cam_names_use:
        cam = obs.cameras[cn]
        model = cam_model(cam, CONV, RANGED)
        depth = as_list(cam.depth_m)
        H = len(depth)
        W = len(depth[0])
        key = cn + "|" + pose_key(cam)
        if key not in A_cache:
            A_cache[key] = z_table(model, H, W)
        A = A_cache[key]
        tz = model["t"][2] - TABLE_Z
        for v in range(H):
            drow = depth[v]
            arow = A[v]
            for u in range(W):
                d = drow[u]
                h = d * arow[u] + tz
                if h > hmin and h < hmax and d > 0.05:
                    p = point(model, u, v, float(d))
                    pts.append((p[0], p[1], p[2] - TABLE_Z))
    return pts


def quantile(vals, q):
    s = sorted(vals)
    if len(s) == 0:
        return None
    return s[int((len(s) - 1) * q)]


def container_at(pts, x, y, radius=0.035, hcap=None):
    cx = x
    cy = y
    info = None
    for it in range(3):
        sel = []
        for p in pts:
            if abs(p[0] - cx) < radius and abs(p[1] - cy) < radius:
                if hcap is None or p[2] < hcap:
                    sel.append(p)
        if len(sel) < 15:
            return info
        top = quantile([p[2] for p in sel], 0.9)
        band = [p for p in sel if p[2] > top - 0.012]
        if len(band) < 6:
            return info
        cx = sum([p[0] for p in band]) / len(band)
        cy = sum([p[1] for p in band]) / len(band)
        info = {"x": cx, "y": cy, "top": top, "n": len(sel), "nb": len(band), "band": band}
    return info


def box_yaw(band):
    # minimum-area rectangle orientation of the top-face points, in [0, 90)
    best = None
    best_area = None
    best_ext = None
    for k in range(30):
        a = math.radians(k * 3.0)
        ca = math.cos(a)
        sa = math.sin(a)
        us = [p[0] * ca + p[1] * sa for p in band]
        vs = [-p[0] * sa + p[1] * ca for p in band]
        e1 = quantile(us, 0.97) - quantile(us, 0.03)
        e2 = quantile(vs, 0.97) - quantile(vs, 0.03)
        area = e1 * e2
        if best_area is None or area < best_area:
            best_area = area
            best = a
            best_ext = [e1, e2]
    return best, best_ext


live_pts = raised_points(obs0, ["front", "wrist"], 0.008, 0.30)
report["live_raised_points"] = len(live_pts)

# ---------------------------------------------------------------- orientation
home_pose = state0.end_effector_poses["primary"]
q_home = [float(v) for v in home_pose.quaternion_wxyz]
R_home = quat_to_mat(q_home)
report["home_ee"] = [round(float(v), 4) for v in home_pose.position] + [round(v, 4) for v in q_home]
if R_home[2][2] < -0.9:
    q_down = q_home
else:
    q_down = [0.0, 1.0, 0.0, 0.0]
Rd = quat_to_mat(q_down)
finger_yaw = math.atan2(Rd[1][1], Rd[0][1])


def finger_hits(x, y, axis_angle, h_floor):
    # raised points inside the two finger footprints (open jaw) for this finger axis
    ca = math.cos(axis_angle)
    sa = math.sin(axis_angle)
    n = 0
    for p in live_pts:
        if p[2] < h_floor:
            continue
        dx = p[0] - x
        dy = p[1] - y
        along = abs(dx * ca + dy * sa)
        across = abs(-dx * sa + dy * ca)
        if along > 0.032 and along < 0.058 and across < 0.013:
            n += 1
    return n


def grasp_quat(box_angle, x=None, y=None, h_floor=None, info=None):
    delta = box_angle - finger_yaw
    while delta > math.pi / 4:
        delta -= math.pi / 2
    while delta <= -math.pi / 4:
        delta += math.pi / 2
    if x is not None:
        # of the two face-aligned finger axes, take the one whose fingers land on free table
        if delta <= 0:
            alt = delta + math.pi / 2
        else:
            alt = delta - math.pi / 2
        n1 = finger_hits(x, y, finger_yaw + delta, h_floor)
        n2 = finger_hits(x, y, finger_yaw + alt, h_floor)
        if info is not None:
            info.append([round(math.degrees(delta), 1), n1, round(math.degrees(alt), 1), n2])
        if n2 < n1:
            delta = alt
    qz = [math.cos(delta / 2), 0.0, 0.0, math.sin(delta / 2)]
    return quat_mul(qz, q_down), delta


# ---------------------------------------------------------------- targets
order = []
used_containers = []
missing = []
for st in plan_steps:
    cls = st[0]
    cand = [c for c in cubes if c["cls"] == cls]
    if not st[1] and len(cand) > 1:
        bestc = cand[0]
        for c in cand:
            if c["px"] * c["nf"] > bestc["px"] * bestc["nf"]:
                bestc = c
        cand = [bestc]
    if len(cand) == 0:
        missing.append(cls)
    for c in cand:
        info = container_at(live_pts, c["x"], c["y"])
        if info is None:
            missing.append(cls + "@nocontainer")
            continue
        dup = False
        for u in used_containers:
            if abs(u[0] - info["x"]) < 0.03 and abs(u[1] - info["y"]) < 0.03:
                dup = True
        if dup:
            continue
        used_containers.append([info["x"], info["y"]])
        order.append({"cls": cls, "cube": [c["x"], c["y"]], "x": info["x"], "y": info["y"], "top": info["top"], "band": info["band"]})
report["missing"] = missing
report["targets"] = [[o["cls"], round(o["x"], 3), round(o["y"], 3), round(o["top"], 3)] for o in order]


# ---------------------------------------------------------------- motion
ticks = {"n": 0}
ended = {"v": False}


def track(res, label, log):
    entry = [label]
    try:
        entry.append(bool(res.ok))
    except Exception:
        entry.append(None)
    try:
        n = int(res.steps_executed)
        ticks["n"] += n
        entry.append(n)
    except Exception:
        entry.append(-1)
    try:
        entry.append(str(res.status))
    except Exception:
        pass
    term = False
    try:
        term = bool(res.terminated) or bool(res.truncated)
    except Exception:
        term = False
    if term:
        ended["v"] = True
        entry.append("TERMINAL")
    log.append(entry)
    return res


def go(x, y, z, q, label, log, tol=0.01, steps=200):
    pose = Pose([x, y, z + TABLE_Z], q, base_frame)
    return track(move_to_pose(pose, tolerance=tol, max_steps=steps), label, log)


picks = []
for idx in range(len(order)):
    if ended["v"]:
        break
    tgt = order[idx]
    log = []
    rec = {"cls": tgt["cls"], "log": log}
    yaw, ext = box_yaw(tgt["band"])
    q, delta = grasp_quat(yaw)
    top = tgt["top"]
    hover = top + 0.13
    go(tgt["x"], tgt["y"], hover, q, "hover", log, tol=0.015, steps=250)
    if ended["v"]:
        picks.append(rec)
        break
    # refine with the wrist camera from above
    try:
        obs = get_observation()
        wpts = raised_points(obs, ["wrist"], 0.008, top + 0.03)
        info = container_at(wpts, tgt["x"], tgt["y"], radius=0.04, hcap=top + 0.03)
        if info is not None and info["nb"] >= 30:
            if abs(info["x"] - tgt["x"]) < 0.03 and abs(info["y"] - tgt["y"]) < 0.03:
                rec["refine"] = [round(info["x"] - tgt["x"], 4), round(info["y"] - tgt["y"], 4), round(info["top"] - top, 4), info["nb"]]
                tgt["x"] = info["x"]
                tgt["y"] = info["y"]
                top = info["top"]
                yaw, ext = box_yaw(info["band"])
                q, delta = grasp_quat(yaw)
    except Exception as exc:
        rec["refine_error"] = str(exc)[:160]
    grasp_h = max(0.012, top - min(0.02, 0.4 * top))
    clr = []
    q, delta = grasp_quat(yaw, tgt["x"], tgt["y"], grasp_h - 0.005, clr)
    rec["clearance"] = clr
    rec["yaw_deg"] = round(math.degrees(yaw), 1)
    rec["delta_deg"] = round(math.degrees(delta), 1)
    rec["ext"] = [round(e, 3) for e in ext]
    rec["top"] = round(top, 4)
    rec["xy"] = [round(tgt["x"], 4), round(tgt["y"], 4)]
    grasp_h = max(0.012, top - min(0.02, 0.4 * top))
    track(open_gripper(), "open", log)
    go(tgt["x"], tgt["y"], top + 0.06, q, "pre", log, tol=0.01, steps=200)
    go(tgt["x"], tgt["y"], grasp_h, q, "descend", log, tol=0.008, steps=200)
    if ended["v"]:
        picks.append(rec)
        break
    track(close_gripper(), "close", log)
    try:
        rec["grip_after_close"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
    except Exception:
        pass
    go(tgt["x"], tgt["y"], grasp_h + 0.12, q, "lift", log, tol=0.015, steps=200)
    if ended["v"]:
        picks.append(rec)
        break
    try:
        rec["grip_after_lift"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
    except Exception:
        pass
    go(tgt["x"], tgt["y"], grasp_h + 0.01, q, "lower", log, tol=0.015, steps=200)
    if ended["v"]:
        picks.append(rec)
        break
    track(open_gripper(), "release", log)
    go(tgt["x"], tgt["y"], top + 0.10, q, "retreat", log, tol=0.02, steps=200)
    picks.append(rec)

report["picks"] = picks
report["ticks"] = ticks["n"]
report["ended"] = ended["v"]
result = report
