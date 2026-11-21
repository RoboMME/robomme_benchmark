import math

# VideoRepick v04 -- identity: track the video-picked object through the post-release shuffle
# in robot_base xy, fusing colour-agnostic depth object points (0.008-0.08 m above the table)
# from BOTH front and wrist cameras of every conditioning-video frame (5 mm cells, warm-started
# weighted k-means with capped constant-velocity prediction; a centre with no support stays at
# its last supported position). Continue into the live frame (front+wrist), take the target's
# 3D points, then: pick + place down twice, then press the button.

report = {"version": "v04", "errors": [], "phases": []}
STATE = {"ended": False}


def note_error(where, err):
    report["errors"].append(where + ": " + str(err))


def fl(v):
    return float(v)


def vec(p):
    return [float(p[0]), float(p[1]), float(p[2])]


def quat_to_rot(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
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


def cam_params(cam):
    pose = cam.camera_pose
    k = cam.intrinsics
    return {"rot": quat_to_rot(pose.quaternion_wxyz), "t": vec(pose.position),
            "fx": fl(k[0][0]), "fy": fl(k[1][1]), "cx": fl(k[0][2]), "cy": fl(k[1][2])}


def project(point, cp):
    """OpenCV-convention projection (verified on v00 ep0: 6/6 live SAM boxes hit)."""
    rot = cp["rot"]
    t = cp["t"]
    d = [point[0] - t[0], point[1] - t[1], point[2] - t[2]]
    xc = rot[0][0] * d[0] + rot[1][0] * d[1] + rot[2][0] * d[2]
    yc = rot[0][1] * d[0] + rot[1][1] * d[1] + rot[2][1] * d[2]
    zc = rot[0][2] * d[0] + rot[1][2] * d[1] + rot[2][2] * d[2]
    if zc <= 1e-6:
        return None
    return [cp["fx"] * xc / zc + cp["cx"], cp["fy"] * yc / zc + cp["cy"]]


def deproject(u, v, depth, cp):
    xc = (u - cp["cx"]) / cp["fx"] * depth
    yc = (v - cp["cy"]) / cp["fy"] * depth
    zc = depth
    rot = cp["rot"]
    t = cp["t"]
    return [rot[0][0] * xc + rot[0][1] * yc + rot[0][2] * zc + t[0],
            rot[1][0] * xc + rot[1][1] * yc + rot[1][2] * zc + t[1],
            rot[2][0] * xc + rot[2][1] * yc + rot[2][2] * zc + t[2]]


def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    return ordered[int((len(ordered) - 1) * fraction)]


MODE = {"pixels": None}
GRID = {}


def object_points(cam):
    """v04: colour-agnostic object points in robot_base from one camera's depth: points
    0.008-0.08 m above the table plane (table z = median of a sparse front-camera grid) and
    more than 0.25 m from the robot base axis. Returns (xs, ys, zs) lists."""
    d = cam.depth_m
    cp = cam_params(cam)
    key = str(len(d)) + "x" + str(len(d[0]))
    if key not in GRID:
        h = len(d)
        w = len(d[0])
        GRID[key] = [d * 0.0 + [float(u) for u in range(w)], d * 0.0 + [[float(v)] for v in range(h)]]
    gu = GRID[key][0]
    gv = GRID[key][1]
    xc = (gu - cp["cx"]) / cp["fx"] * d
    yc = (gv - cp["cy"]) / cp["fy"] * d
    rot = cp["rot"]
    t = cp["t"]
    xw = rot[0][0] * xc + rot[0][1] * yc + rot[0][2] * d + t[0]
    yw = rot[1][0] * xc + rot[1][1] * yc + rot[1][2] * d + t[1]
    zw = rot[2][0] * xc + rot[2][1] * yc + rot[2][2] * d + t[2]
    if "zt" not in GRID:
        vals = []
        for row in zw[::8, ::8].tolist():
            for val in row:
                vals.append(float(val))
        GRID["zt"] = quantile(vals, 0.5)
        report["table_z"] = round(GRID["zt"], 4)
    zt = GRID["zt"]
    m = (zw > zt + 0.008) & (zw < zt + 0.08) & ((xw * xw + yw * yw) > 0.0625) & (d > 0.05)
    MODE["pixels"] = "depth-vector-world"
    return xw[m].tolist(), yw[m].tolist(), zw[m].tolist()


CELL = 0.005


def frame_cells(frame):
    """Fuse front + wrist object points into 5 mm robot_base xy cells: list of [x, y, weight]."""
    counts = {}
    for name in CAMS:
        try:
            xs, ys, zs = object_points(frame.cameras[name])
        except Exception as err:
            note_error("points_" + name, err)
            continue
        for k in range(len(xs)):
            ck = (int(math.floor(xs[k] / CELL)), int(math.floor(ys[k] / CELL)))
            counts[ck] = counts.get(ck, 0) + 1
    out = []
    for ck in counts:
        out.append([(ck[0] + 0.5) * CELL, (ck[1] + 0.5) * CELL, counts[ck]])
    return out, counts


def kmeans_cells(cells, centers, radius, iters):
    cur = [[c[0], c[1]] for c in centers]
    weights = [0 for c in cur]
    r2 = radius * radius
    for it in range(iters):
        acc = [[0.0, 0.0, 0.0] for c in cur]
        for cell in cells:
            best = -1
            bd = r2
            for k in range(len(cur)):
                dx = cell[0] - cur[k][0]
                dy = cell[1] - cur[k][1]
                dd = dx * dx + dy * dy
                if dd < bd:
                    bd = dd
                    best = k
            if best >= 0:
                acc[best][0] += cell[0] * cell[2]
                acc[best][1] += cell[1] * cell[2]
                acc[best][2] += cell[2]
        new = []
        for k in range(len(cur)):
            if acc[k][2] > 0:
                new.append([acc[k][0] / acc[k][2], acc[k][1] / acc[k][2]])
            else:
                new.append(centers[k])   # no support: stay at the last supported position (no drift)
        cur = new
        weights = [a[2] for a in acc]
    return cur, weights


def components(pix):
    remaining = set(pix)
    comps = []
    while len(remaining) > 0:
        p = remaining.pop()
        stack = [p]
        comp = [p]
        while len(stack) > 0:
            q = stack.pop()
            for du in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    r = (q[0] + du, q[1] + dv)
                    if r in remaining:
                        remaining.remove(r)
                        stack.append(r)
                        comp.append(r)
        comps.append(comp)
    return comps


def centroid(pts):
    return [sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)]


def kmeans_step(pix, centers, radius, iters):
    """Warm-started k-means; returns new centers, counts and per-center pixel lists (last iter)."""
    cur = [[c[0], c[1]] for c in centers]
    members = [[] for c in cur]
    counts = [0 for c in cur]
    r2 = radius * radius
    for it in range(iters):
        members = [[] for c in cur]
        for p in pix:
            best = -1
            bd = r2
            for k in range(len(cur)):
                dx = p[0] - cur[k][0]
                dy = p[1] - cur[k][1]
                d = dx * dx + dy * dy
                if d < bd:
                    bd = d
                    best = k
            if best >= 0:
                members[best].append(p)
        new = []
        for k in range(len(cur)):
            if len(members[k]) > 0:
                new.append(centroid(members[k]))
            else:
                new.append(cur[k])
        cur = new
        counts = [len(m) for m in members]
    return cur, counts, members


def hold(ticks, gripper):
    state = get_robot_state()
    for tick in range(ticks):
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position",
            target=[float(v) for v in state.joint_positions["primary"]],
            gripper_position=gripper,
            embodiment=state.embodiment)}))
        try:
            if reply.terminated or reply.truncated:
                STATE["ended"] = True
                return "ended"
        except Exception:
            pass
    return "ok"


def run_motion(name, res):
    entry = [name]
    try:
        entry.append(bool(res.ok))
        entry.append(str(res.status))
        entry.append(int(res.steps_executed))
    except Exception as err:
        entry.append("?" + str(err))
    try:
        if res.terminated or res.truncated:
            STATE["ended"] = True
            entry.append("ENDED")
    except Exception:
        pass
    report["phases"].append(entry)
    return res


def move(name, pos, q, tol, steps):
    if STATE["ended"]:
        return None
    res = run_motion(name, move_to_pose(Pose([pos[0], pos[1], pos[2]], q, base_frame),
                                        tolerance=tol, max_steps=steps))
    # v02: a plan/IK failure executes 0 steps; retry once through a raised midpoint waypoint
    try:
        failed_unmoved = (not bool(res.ok)) and int(res.steps_executed) == 0
    except Exception:
        failed_unmoved = False
    if failed_unmoved and not STATE["ended"]:
        cur = vec(get_robot_state().end_effector_poses["primary"].position)
        mid = [(cur[0] + pos[0]) / 2.0, (cur[1] + pos[1]) / 2.0, max(cur[2], pos[2]) + 0.03]
        run_motion(name + "_via", move_to_pose(Pose(mid, q, base_frame), tolerance=0.02, max_steps=steps))
        if not STATE["ended"]:
            res = run_motion(name + "_retry", move_to_pose(Pose([pos[0], pos[1], pos[2]], q, base_frame),
                                                           tolerance=tol, max_steps=steps))
    return res


# ---------------------------------------------------------------- metadata
context = get_task_context()
report["goal"] = context.language
try:
    report["max_steps"] = robomme_plus.get_task_metadata()["max_steps"]
except Exception as err:
    note_error("task_metadata", err)

state0 = get_robot_state()
base_frame = state0.base_frame
home_pose = state0.end_effector_poses["primary"]
home_q = [float(v) for v in home_pose.quaternion_wxyz]

# ---------------------------------------------------------------- video: events
video = robomme_plus.get_conditioning_video()
nframes = len(video)
report["video_frames"] = nframes
series = []
for i in range(nframes):
    rs = video[i].robot_state
    series.append([float(rs.gripper_positions["primary"]),
                   vec(rs.end_effector_poses["primary"].position)])
events = []
closed = None
for i in range(nframes):
    g = series[i][0]
    if closed is None:
        closed = g < 0.7
        continue
    if (not closed) and g < 0.7:
        closed = True
        events.append(["close", i])
    elif closed and g > 0.85:
        closed = False
        events.append(["open", i])
report["video_events"] = [[e[0], e[1], [round(v, 4) for v in series[e[1]][1]]] for e in events]

opens = [e[1] for e in events if e[0] == "open"]
closes = [e[1] for e in events if e[0] == "close"]
if len(opens) == 0 or len(closes) == 0:
    raise ValueError("no grasp/release event in conditioning video")
release_i = opens[-1]
release_pt = series[release_i][1]
# lowest EE z near release = object resting height reference
for j in range(max(0, release_i - 6), min(nframes, release_i + 4)):
    if series[j][1][2] < release_pt[2]:
        release_pt = series[j][1]
# tracking starts once the arm has risen well clear of the released object
start_i = None
for j in range(release_i, nframes):
    dxy = math.sqrt((series[j][1][0] - release_pt[0]) ** 2 + (series[j][1][1] - release_pt[1]) ** 2)
    if series[j][1][2] > release_pt[2] + 0.12 and dxy > 0.05:
        start_i = j
        break
if start_i is None:
    start_i = min(nframes - 1, release_i + 20)
start_i = min(nframes - 1, start_i + 3)
report["release_frame"] = release_i
report["track_start_frame"] = start_i

# ---------------------------------------------------------------- video: tracking (world xy)
CAMS = ["front", "wrist"]
try:
    names = [n for n in video[start_i].cameras]
    CAMS = [n for n in ("front", "wrist") if n in names]
except Exception as err:
    note_error("cams", err)
cells0, counts0 = frame_cells(video[start_i])
# connected components over occupied cells (8-neighbourhood)
remaining = set([ck for ck in counts0])
comps = []
while len(remaining) > 0:
    p = remaining.pop()
    stack = [p]
    comp = [p]
    while len(stack) > 0:
        q = stack.pop()
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                r = (q[0] + du, q[1] + dv)
                if r in remaining:
                    remaining.remove(r)
                    stack.append(r)
                    comp.append(r)
    if len(comp) >= 4:
        comps.append(comp)
if len(comps) == 0:
    raise ValueError("no objects at tracking start")
sizes = sorted(len(c) for c in comps)
median_size = sizes[len(sizes) // 2]
spans = []
for c in comps:
    if int(round(len(c) / float(median_size))) <= 1:
        spans.append((max(max(q[0] for q in c) - min(q[0] for q in c),
                          max(q[1] for q in c) - min(q[1] for q in c)) + 1) * CELL)
spans = sorted(spans)
diam = spans[len(spans) // 2] if spans else 0.04
RADIUS = max(0.015, 0.75 * diam)
VCAP = 0.5 * diam
report["object_diam_m"] = round(diam, 4)
centers = []
for c in comps:
    w = 0.0
    sx = 0.0
    sy = 0.0
    for q in c:
        cw = counts0[q]
        sx += (q[0] + 0.5) * CELL * cw
        sy += (q[1] + 0.5) * CELL * cw
        w += cw
    mx = sx / w
    my = sy / w
    n_obj = int(round(len(c) / float(median_size)))
    if n_obj <= 1:
        centers.append([mx, my])
    else:
        pts = [[(q[0] + 0.5) * CELL, (q[1] + 0.5) * CELL] for q in c]
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        syy = sum((p[1] - my) ** 2 for p in pts)
        sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
        ang = 0.5 * math.atan2(2 * sxy, sxx - syy)
        spread = math.sqrt((sxx + syy) / len(pts))
        for k in range(n_obj):
            off = (k - (n_obj - 1) / 2.0) * spread
            centers.append([mx + off * math.cos(ang), my + off * math.sin(ang)])
report["init_components_cells"] = [len(c) for c in comps]
report["init_centers"] = [[round(c[0], 4), round(c[1], 4)] for c in centers]
target = 0
bd = None
for k in range(len(centers)):
    dd = (centers[k][0] - release_pt[0]) ** 2 + (centers[k][1] - release_pt[1]) ** 2
    if bd is None or dd < bd:
        bd = dd
        target = k
report["init_target_dist_m"] = round(math.sqrt(bd), 4)

prev = [[c[0], c[1]] for c in centers]
cur = [[c[0], c[1]] for c in centers]
frame_cells_list = []
track = []
min_sep = []
lost = 0
for i in range(start_i + 1, nframes):
    pred = []
    for k in range(len(cur)):
        vx = cur[k][0] - prev[k][0]
        vy = cur[k][1] - prev[k][1]
        sp = math.sqrt(vx * vx + vy * vy)
        if sp > VCAP:
            vx, vy = vx * VCAP / sp, vy * VCAP / sp
        pred.append([cur[k][0] + vx, cur[k][1] + vy])
    cells, cnt = frame_cells(video[i])
    frame_cells_list.append(cells)
    new, weights = kmeans_cells(cells, pred, RADIUS, 3)
    for k in range(len(new)):
        if weights[k] == 0:
            new[k] = [cur[k][0], cur[k][1]]
    if weights[target] == 0:
        lost += 1
    prev = cur
    cur = new
    sep = None
    for k in range(len(cur)):
        if k == target:
            continue
        dd = math.sqrt((cur[k][0] - cur[target][0]) ** 2 + (cur[k][1] - cur[target][1]) ** 2)
        if sep is None or dd < sep:
            sep = dd
    min_sep.append([i, round(sep, 4) if sep is not None else None])
    if (i - start_i) % 10 == 0 or i == nframes - 1:
        track.append([i, round(cur[target][0], 4), round(cur[target][1], 4), weights[target]])
report["track"] = track
report["track_lost_frames"] = lost
close_calls = [m for m in min_sep if m[1] is not None and m[1] < 0.8 * diam]
report["track_close_calls"] = close_calls[:40]
report["track_close_call_count"] = len(close_calls)
report["video_final_centers"] = [[round(c[0], 4), round(c[1], 4)] for c in cur]

# diagnostic only: backward tracking from the final configuration
try:
    bprev = [[c[0], c[1]] for c in cur]
    bcur = [[c[0], c[1]] for c in cur]
    for j in range(len(frame_cells_list) - 2, -2, -1):
        cells_j = frame_cells_list[j] if j >= 0 else cells0
        bpred = []
        for k in range(len(bcur)):
            vx = bcur[k][0] - bprev[k][0]
            vy = bcur[k][1] - bprev[k][1]
            sp = math.sqrt(vx * vx + vy * vy)
            if sp > VCAP:
                vx, vy = vx * VCAP / sp, vy * VCAP / sp
            bpred.append([bcur[k][0] + vx, bcur[k][1] + vy])
        bnew, bw = kmeans_cells(cells_j, bpred, RADIUS, 3)
        for k in range(len(bnew)):
            if bw[k] == 0:
                bnew[k] = [bcur[k][0], bcur[k][1]]
        bprev = bcur
        bcur = bnew
    nearest_k = 0
    nd = None
    for k in range(len(centers)):
        dd = (bcur[target][0] - centers[k][0]) ** 2 + (bcur[target][1] - centers[k][1]) ** 2
        if nd is None or dd < nd:
            nd = dd
            nearest_k = k
    report["backward_check_m"] = round(math.sqrt((bcur[target][0] - centers[target][0]) ** 2
                                                 + (bcur[target][1] - centers[target][1]) ** 2), 4)
    report["backward_consistent"] = nearest_k == target
except Exception as err:
    note_error("backward", err)

# ---------------------------------------------------------------- live: continue tracking
live = get_observation()
front = live.cameras["front"]
cpl = cam_params(front)
live_cells, live_counts = frame_cells(live)
live_centers, live_w = kmeans_cells(live_cells, cur, RADIUS, 3)
report["live_shift_m"] = round(math.sqrt((live_centers[target][0] - cur[target][0]) ** 2
                                         + (live_centers[target][1] - cur[target][1]) ** 2), 4)
report["live_target_weight"] = live_w[target]


def object_near(obs, cx, cy, radius):
    xs = []
    ys = []
    zs = []
    for name in CAMS:
        try:
            a, b2, c = object_points(obs.cameras[name])
        except Exception as err:
            note_error("near_" + name, err)
            continue
        for k in range(len(a)):
            if (a[k] - cx) ** 2 + (b2[k] - cy) ** 2 < radius * radius:
                xs.append(a[k])
                ys.append(b2[k])
                zs.append(c[k])
    if len(xs) < 10:
        return None
    top = quantile(zs, 0.95)
    return {"center": [quantile(xs, 0.5), quantile(ys, 0.5), quantile(zs, 0.5)], "top": top, "n": len(xs),
            "top_xy": [[xs[k], ys[k]] for k in range(len(xs)) if zs[k] > top - 0.008]}


obj = object_near(live, live_centers[target][0], live_centers[target][1], RADIUS)
if obj is None:
    raise ValueError("target has no live points")
report["target_live"] = {"center": obj["center"], "top": obj["top"], "n": obj["n"]}
try:
    uvt = project(obj["center"], cpl)
    seg = segment_points("front", [[int(round(uvt[0])), int(round(uvt[1]))]])
    if seg.ok and len(seg.segmentations) > 0:
        cloud = mask_to_point_cloud(seg.segmentations[0], "front", target_frame=base_frame)
        sp = cloud.points
        if len(sp) >= 10:
            sc = [quantile([float(p[0]) for p in sp], 0.5), quantile([float(p[1]) for p in sp], 0.5),
                  quantile([float(p[2]) for p in sp], 0.5)]
            report["target_sam_center"] = sc
            report["target_sam_vs_live_m"] = round(math.sqrt((sc[0] - obj["center"][0]) ** 2
                                                             + (sc[1] - obj["center"][1]) ** 2), 4)
except Exception as err:
    note_error("segment_points", err)

# ---------------------------------------------------------------- live: button
button = None
try:
    for prompt in ("button", "white button"):
        seg = segment_text("front", prompt)
        if not seg.ok:
            continue
        best = None
        for cand in seg.segmentations[:5]:
            cloud = mask_to_point_cloud(cand, "front", target_frame=base_frame)
            pts = cloud.points
            if len(pts) < 15:
                continue
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            zs = [float(p[2]) for p in pts]
            ex = quantile(xs, 0.95) - quantile(xs, 0.05)
            ey = quantile(ys, 0.95) - quantile(ys, 0.05)
            top = quantile(zs, 0.95)
            c = [quantile(xs, 0.5), quantile(ys, 0.5), quantile(zs, 0.5)]
            dist_obj = math.sqrt((c[0] - obj["center"][0]) ** 2 + (c[1] - obj["center"][1]) ** 2)
            ok_geo = ex < 0.12 and ey < 0.12 and ex > 0.02 and ey > 0.02 and top < 0.08 and dist_obj > 0.03
            report.setdefault("button_cands", []).append(
                {"prompt": prompt, "score": round(float(cand.score), 3), "center": c, "top": top,
                 "ext": [ex, ey], "ok": ok_geo})
            col = None
            try:
                uvb = project(c, cpl)
                col = [float(x) for x in front.rgb[int(round(uvb[1]))][int(round(uvb[0]))]]
            except Exception:
                col = None
            grey = col is not None and (max(col) - min(col)) < 45
            report["button_cands"][-1]["grey"] = grey
            rank = float(cand.score) + (1.0 if grey else 0.0)
            if ok_geo and (best is None or rank > best[0]):
                best = [rank, c, top]
        if best is not None:
            button = {"center": best[1], "top": best[2]}
            break
except Exception as err:
    note_error("button", err)
report["button"] = button

# ---------------------------------------------------------------- act
def grasp_quat(top_xy):
    yaw = None
    if len(top_xy) >= 8:
        mx = sum(p[0] for p in top_xy) / len(top_xy)
        my = sum(p[1] for p in top_xy) / len(top_xy)
        sxx = sum((p[0] - mx) ** 2 for p in top_xy)
        syy = sum((p[1] - my) ** 2 for p in top_xy)
        sxy = sum((p[0] - mx) * (p[1] - my) for p in top_xy)
        yaw = 0.5 * math.atan2(2 * sxy, sxx - syy)
    rot_home = quat_to_rot(home_q)
    finger_yaw = math.atan2(rot_home[1][1], rot_home[0][1])
    delta = 0.0
    if yaw is not None:
        delta = yaw - finger_yaw
        while delta > math.pi / 4:
            delta -= math.pi / 2
        while delta < -math.pi / 4:
            delta += math.pi / 2
    return quat_mul([math.cos(delta / 2), 0.0, 0.0, math.sin(delta / 2)], home_q), delta


def pick_and_place(tag, center, top, q):
    cx, cy = center[0], center[1]
    grasp_z = max(top - 0.02, GRID["zt"] + 0.006)
    run_motion(tag + "_open", open_gripper())
    move(tag + "_pre", [cx, cy, top + 0.10], q, 0.01, 200)
    move(tag + "_descend", [cx, cy, grasp_z], q, 0.008, 150)
    if STATE["ended"]:
        return
    run_motion(tag + "_close", close_gripper())
    try:
        report[tag + "_gripper_closed"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
    except Exception as err:
        note_error(tag + "_grip", err)
    move(tag + "_lift", [cx, cy, top + 0.15], q, 0.01, 200)
    move(tag + "_lower", [cx, cy, grasp_z + 0.005], q, 0.008, 200)
    if STATE["ended"]:
        return
    run_motion(tag + "_release", open_gripper())
    move(tag + "_retreat", [cx, cy, top + 0.10], q, 0.01, 200)


q1, d1 = grasp_quat(obj["top_xy"])
report["grasp_yaw_delta"] = round(d1, 3)
pick_and_place("pick1", obj["center"], obj["top"], q1)

# re-localize the placed object near where it was put down (both cameras)
center2 = obj["center"]
top2 = obj["top"]
q2 = q1
if not STATE["ended"]:
    try:
        obj2 = object_near(get_observation(), center2[0], center2[1], RADIUS)
        if obj2 is not None:
            shift = math.sqrt((obj2["center"][0] - center2[0]) ** 2 + (obj2["center"][1] - center2[1]) ** 2)
            report["relocalize_shift_m"] = round(shift, 4)
            report["relocalize_n"] = obj2["n"]
            if shift < 0.04:
                center2 = obj2["center"]
                top2 = obj2["top"]
                q2, d2 = grasp_quat(obj2["top_xy"])
        else:
            report["relocalize"] = "too few points"
    except Exception as err:
        note_error("relocalize", err)
if not STATE["ended"]:
    pick_and_place("pick2", center2, top2, q2)

# press the button
if not STATE["ended"] and button is not None:
    bx, by = button["center"][0], button["center"][1]
    btop = button["top"]
    run_motion("btn_close", close_gripper())
    move("btn_above", [bx, by, btop + 0.08], home_q, 0.01, 200)
    move("btn_press", [bx, by, max(0.004, btop - 0.012)], home_q, 0.005, 60)
    if not STATE["ended"]:
        move("btn_press_deeper", [bx, by, max(0.004, btop - 0.02)], home_q, 0.005, 40)
    move("btn_up", [bx, by, btop + 0.08], home_q, 0.01, 120)
if not STATE["ended"]:
    report["final_hold"] = hold(10, 0.0)
try:
    st = get_robot_state()
    report["final_t"] = float(st.timestamp_s)
except Exception as err:
    note_error("final_state", err)
report["ended_by_env"] = STATE["ended"]
result = report
