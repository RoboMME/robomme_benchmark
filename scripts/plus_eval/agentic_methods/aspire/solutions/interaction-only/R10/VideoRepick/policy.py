# v01 — VideoRepick (robomme_plus task 13).
# Change vs v00 (perception/tracking mode): track every tabletop object in WORLD XY using
# depth points fused from BOTH cameras (front + wrist) in every conditioning-video frame, with
# constant-velocity local k-means and spawning of newly visible objects.  The wrist camera sees
# objects that a tall occluder hides from the front camera, and at higher resolution.
# Live: move to the video's final recorded joint configuration (the viewpoint the tracker ended
# in), re-detect objects with both cameras, take the cluster nearest the tracked target.
# All quantities derived at runtime from public observations; no recorded constants.
import math

report = {"phase": "start", "errors": [], "markers": [], "motions": []}


def note(msg):
    report["errors"].append(str(msg))


def marker(label):
    # A cheap public call whose trace timestamp records wall-clock progress of the program.
    get_robot_state()
    report["markers"].append(label)


# ---------------------------------------------------------------- geometry helpers
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


def make_cam(cam_obs):
    k = cam_obs.intrinsics
    pose = cam_obs.camera_pose
    return {
        "fx": float(k[0][0]),
        "fy": float(k[1][1]),
        "cx": float(k[0][2]),
        "cy": float(k[1][2]),
        "R": quat_to_rot(pose.quaternion_wxyz),
        "t": [float(v) for v in pose.position],
        "rows": len(cam_obs.depth_m),
        "cols": len(cam_obs.depth_m[0]),
    }


def ray_dir(cam, u, v):
    a = (u - cam["cx"]) / cam["fx"]
    b = (v - cam["cy"]) / cam["fy"]
    R = cam["R"]
    return (
        R[0][0] * a + R[0][1] * b + R[0][2],
        R[1][0] * a + R[1][1] * b + R[1][2],
        R[2][0] * a + R[2][1] * b + R[2][2],
    )


def pixel_to_world(cam, u, v, d):
    dx, dy, dz = ray_dir(cam, u, v)
    t = cam["t"]
    return (t[0] + d * dx, t[1] + d * dy, t[2] + d * dz)


def pixel_on_plane(cam, u, v, h):
    # Intersect the pixel ray with the horizontal plane z = h (returns None if parallel/up).
    dx, dy, dz = ray_dir(cam, u, v)
    t = cam["t"]
    if dz > -1e-6:
        return None
    d = (h - t[2]) / dz
    return (t[0] + d * dx, t[1] + d * dy)


def world_to_pixel(cam, p):
    R = cam["R"]
    t = cam["t"]
    q = (p[0] - t[0], p[1] - t[1], p[2] - t[2])
    xc = R[0][0] * q[0] + R[1][0] * q[1] + R[2][0] * q[2]
    yc = R[0][1] * q[0] + R[1][1] * q[1] + R[2][1] * q[2]
    zc = R[0][2] * q[0] + R[1][2] * q[1] + R[2][2] * q[2]
    if zc <= 1e-6:
        return None
    return (cam["fx"] * xc / zc + cam["cx"], cam["fy"] * yc / zc + cam["cy"])


def quantile(values, frac):
    ordered = sorted(values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * frac)]


# ---------------------------------------------------------------- world-point extraction
STRIDES = {"front": 2, "wrist": 4}
TRACK_CAMERAS = ("front", "wrist")
BASE_EXCLUSION_M = 0.12
SPAWN_MIN_POINTS = 8
MODEL_CACHE = {}


def cam_model(name, cam_obs):
    cam = make_cam(cam_obs)
    key = name + ":" + ",".join([str(round(v, 4)) for v in cam["t"]]) + ":" + ",".join(
        [str(round(cam["R"][a][b], 4)) for a in range(3) for b in range(3)])
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]
    stride = STRIDES[name]
    dirs = {}
    for v in range(0, cam["rows"], stride):
        row = {}
        for u in range(0, cam["cols"], stride):
            row[u] = ray_dir(cam, u, v)
        dirs[v] = row
    cam["dirs"] = dirs
    cam["stride"] = stride
    cam["name"] = name
    MODEL_CACHE[key] = cam
    return cam


def scan(cam, depth, z_lo, z_hi, u0, u1, v0, v1, with_z):
    # World points (x, y[, z]) of grid pixels in [u0,u1)x[v0,v1) whose height is in (z_lo, z_hi).
    s = cam["stride"]
    out = []
    t = cam["t"]
    tx = t[0]
    ty = t[1]
    tz = t[2]
    u0 = u0 - (u0 % s)
    v0 = v0 - (v0 % s)
    lim2 = BASE_EXCLUSION_M * BASE_EXCLUSION_M
    dirs = cam["dirs"]
    for v in range(max(0, v0), min(cam["rows"], v1), s):
        drow = dirs[v]
        row = depth[v]
        for u in range(max(0, u0), min(cam["cols"], u1), s):
            dv = drow[u]
            if dv[2] > -1e-6:
                continue
            d = float(row[u])
            z = tz + d * dv[2]
            if z <= z_lo or z >= z_hi:
                continue
            x = tx + d * dv[0]
            y = ty + d * dv[1]
            if x * x + y * y < lim2:
                continue
            if with_z:
                out.append((x, y, z))
            else:
                out.append((x, y))
    return out


def scan_full(cam, depth, z_lo, z_hi, with_z):
    return scan(cam, depth, z_lo, z_hi, 0, cam["cols"], 0, cam["rows"], with_z)


def window_for(cam, p, radius):
    px = world_to_pixel(cam, p)
    if px is None:
        return None
    R = cam["R"]
    t = cam["t"]
    q = (p[0] - t[0], p[1] - t[1], p[2] - t[2])
    zc = R[0][2] * q[0] + R[1][2] * q[1] + R[2][2] * q[2]
    half = int(radius * cam["fx"] / max(zc, 0.05)) + cam["stride"]
    u = int(px[0])
    v = int(px[1])
    if u + half < 0 or v + half < 0 or u - half >= cam["cols"] or v - half >= cam["rows"]:
        return None
    return (u - half, u + half + 1, v - half, v + half + 1)


def grid_clusters(points, cell):
    # Single-linkage clustering of 2-D points via an occupancy grid (8-connected cells).
    cells = {}
    for p in points:
        key = (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)))
        if key in cells:
            cells[key].append(p)
        else:
            cells[key] = [p]
    seen = {}
    out = []
    for key in cells:
        if key in seen:
            continue
        seen[key] = True
        stack = [key]
        members = []
        while len(stack) > 0:
            k = stack.pop()
            for p in cells[k]:
                members.append(p)
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    nk = (k[0] + da, k[1] + db)
                    if nk in cells and nk not in seen:
                        seen[nk] = True
                        stack.append(nk)
        out.append(members)
    return out


def mean_xy(points):
    sx = 0.0
    sy = 0.0
    for p in points:
        sx += p[0]
        sy += p[1]
    n = float(len(points))
    return (sx / n, sy / n)


def local_kmeans(groups, centers, iters, gate):
    # groups: list of (points, neighbour_indices); each point only competes among neighbours.
    cur = [c for c in centers]
    gate2 = gate * gate
    k = len(cur)
    cnt = [0] * k
    assigned = {}
    for it in range(iters):
        su = [0.0] * k
        sv = [0.0] * k
        cnt = [0] * k
        for grp in groups:
            nb = grp[1]
            for p in grp[0]:
                best = -1
                bd = gate2
                for j in nb:
                    c = cur[j]
                    du = p[0] - c[0]
                    dv = p[1] - c[1]
                    dd = du * du + dv * dv
                    if dd < bd:
                        bd = dd
                        best = j
                if best >= 0:
                    su[best] += p[0]
                    sv[best] += p[1]
                    cnt[best] += 1
        nxt = []
        for j in range(k):
            if cnt[j] >= 2:
                nxt.append((su[j] / cnt[j], sv[j] / cnt[j]))
            else:
                nxt.append(cur[j])
        cur = nxt
    return cur, cnt


def frame_points(frame, z_lo, z_hi, with_z):
    pts = []
    for name in TRACK_CAMERAS:
        try:
            co = frame.cameras[name]
        except Exception:
            continue
        cam = cam_model(name, co)
        for p in scan_full(cam, co.depth_m, z_lo, z_hi, with_z):
            pts.append(p)
    return pts


# ---------------------------------------------------------------- video analysis
GATE = 0.025          # m: a point belongs to a track only within this radius
NEIGH = 0.10          # m: tracks closer than this compete for the same points
WIN_R = 0.045         # m: per-track window radius projected into each camera
SPAWN_EVERY = 6       # frames between full scans for newly visible objects
CELL = 0.01           # m: clustering grid cell


def analyse_video(video):
    info = {"ok": False}
    n = len(video)
    info["frames"] = n
    if n < 3:
        info["why"] = "no conditioning video"
        return info
    grips = []
    eefs = []
    for i in range(n):
        rs = video[i].robot_state
        grips.append(float(rs.gripper_positions["primary"]))
        eefs.append([float(v) for v in rs.end_effector_poses["primary"].position])
    grasp_i = None
    release_i = None
    for i in range(1, n):
        if grasp_i is None and grips[i - 1] > 0.5 and grips[i] <= 0.5:
            grasp_i = i
        elif grasp_i is not None and release_i is None and grips[i - 1] <= 0.5 and grips[i] > 0.5:
            release_i = i
    info["grasp_index"] = grasp_i
    info["release_index"] = release_i
    if grasp_i is None or release_i is None:
        info["why"] = "no grasp/release event in video"
        return info
    rel = eefs[release_i]
    info["release_eef"] = rel
    info["grasp_eef"] = eefs[grasp_i]
    f0 = None
    for i in range(release_i, n):
        if eefs[i][2] > rel[2] + 0.08:
            f0 = i
            break
    if f0 is None:
        f0 = min(n - 1, release_i + 10)
    info["track_start"] = f0
    frame = video[f0]
    # Table height: median height of a sparse front-camera sample (the table dominates).
    fc = make_cam(frame.cameras["front"])
    depth = frame.cameras["front"].depth_m
    zs = []
    for v in range(4, fc["rows"], 8):
        row = depth[v]
        for u in range(4, fc["cols"], 8):
            zs.append(pixel_to_world(fc, u, v, float(row[u]))[2])
    table_z = quantile(zs, 0.5)
    info["table_z"] = table_z
    # Picked object's height: highest points near the release point, both cameras.
    near = []
    for p in frame_points(frame, table_z + 0.004, table_z + 0.15, True):
        if (p[0] - rel[0]) ** 2 + (p[1] - rel[1]) ** 2 < 0.03 * 0.03:
            near.append(p[2] - table_z)
    info["release_points"] = len(near)
    if len(near) < 5:
        info["why"] = "no object points at the release point"
        return info
    height = quantile(near, 0.95)
    info["object_height"] = height
    z_lo = table_z + 0.5 * height
    z_hi = table_z + 1.5 * height
    z_mid = 0.5 * (z_lo + z_hi)
    info["band"] = [z_lo, z_hi]
    # Initial detection (both cameras).
    pts = frame_points(frame, z_lo, z_hi, False)
    cl = [c for c in grid_clusters(pts, CELL) if len(c) >= 4]
    if len(cl) == 0:
        info["why"] = "no foreground clusters"
        return info
    tracks = [mean_xy(c) for c in cl]
    best = None
    bd = None
    for j in range(len(tracks)):
        dd = (tracks[j][0] - rel[0]) ** 2 + (tracks[j][1] - rel[1]) ** 2
        if bd is None or dd < bd:
            bd = dd
            best = j
    info["target_index"] = best
    info["target_init_dist_m"] = math.sqrt(bd)
    info["initial_tracks"] = [[round(c[0], 4), round(c[1], 4)] for c in tracks]
    vel = [(0.0, 0.0) for c in tracks]
    path = [0.0 for c in tracks]
    born = [f0 for c in tracks]
    target_path = []
    spawned = 0
    for i in range(f0 + 1, n):
        fr = video[i]
        cams = []
        for name in TRACK_CAMERAS:
            try:
                co = fr.cameras[name]
            except Exception:
                continue
            cams.append((cam_model(name, co), co.depth_m))
        preds = []
        for j in range(len(tracks)):
            preds.append((tracks[j][0] + 0.7 * vel[j][0], tracks[j][1] + 0.7 * vel[j][1]))
        groups = []
        for j in range(len(preds)):
            p = preds[j]
            nb = []
            for k in range(len(preds)):
                if (preds[k][0] - p[0]) ** 2 + (preds[k][1] - p[1]) ** 2 < NEIGH * NEIGH:
                    nb.append(k)
            gp = []
            for cd in cams:
                w = window_for(cd[0], (p[0], p[1], z_mid), WIN_R)
                if w is None:
                    continue
                for q in scan(cd[0], cd[1], z_lo, z_hi, w[0], w[1], w[2], w[3], False):
                    gp.append(q)
            groups.append((gp, nb))
        new, cnt = local_kmeans(groups, preds, 2, GATE)
        for j in range(len(tracks)):
            vel[j] = (new[j][0] - tracks[j][0], new[j][1] - tracks[j][1])
            path[j] += math.sqrt(vel[j][0] ** 2 + vel[j][1] ** 2)
            tracks[j] = new[j]
        if (i - f0) % SPAWN_EVERY == 0:
            allp = []
            for cd in cams:
                for q in scan_full(cd[0], cd[1], z_lo, z_hi, False):
                    far = True
                    for c in tracks:
                        if (q[0] - c[0]) ** 2 + (q[1] - c[1]) ** 2 < (GATE + 0.01) ** 2:
                            far = False
                            break
                    if far:
                        allp.append(q)
            for c in grid_clusters(allp, CELL):
                if len(c) >= SPAWN_MIN_POINTS:
                    tracks.append(mean_xy(c))
                    vel.append((0.0, 0.0))
                    path.append(0.0)
                    born.append(i)
                    spawned += 1
        if (i - f0) % 8 == 0:
            target_path.append([i, round(tracks[best][0], 4), round(tracks[best][1], 4)])
    info["spawned"] = spawned
    info["target_path"] = target_path
    info["final_tracks"] = [[round(c[0], 4), round(c[1], 4)] for c in tracks]
    info["path_m"] = [round(p, 3) for p in path]
    info["born"] = born
    info["target_final"] = tracks[best]
    info["final_joints"] = [float(v) for v in video[n - 1].robot_state.joint_positions["primary"]]
    info["ok"] = True
    return info


# ---------------------------------------------------------------- live perception
def live_clusters(info):
    obs = get_observation()
    z_lo = info["band"][0]
    z_hi = info["band"][1]
    pts = frame_points(obs, z_lo, z_hi, True)
    out = []
    for c in grid_clusters(pts, CELL):
        if len(c) >= 4:
            out.append({"xy": mean_xy(c), "n": len(c), "pts": c})
    return out


def z_hi_of(info):
    return info["band"][1]


def live_points(z_lo, z_hi):
    return frame_points(get_observation(), z_lo, z_hi, True)


# ---------------------------------------------------------------- motion helpers
def ex_ok(res, label):
    try:
        ok = bool(res.ok)
    except Exception:
        ok = False
    entry = {"label": label, "ok": ok}
    try:
        entry["status"] = str(res.status)
        entry["steps"] = int(res.steps_executed)
    except Exception:
        pass
    term = False
    try:
        term = bool(res.terminated) or bool(res.truncated)
    except Exception:
        term = False
    entry["terminal"] = term
    report["motions"].append(entry)
    return ok, term


def pose_at(xy, z, quat, frame):
    return Pose((float(xy[0]), float(xy[1]), float(z)), quat, frame)


def choose_yaw(home_q, target_xy, others):
    # Closing axis that keeps both fingers farthest from neighbouring object points.
    best_q = home_q
    best_clear = -1.0
    best_deg = 0
    for deg in range(-90, 90, 15):
        th = math.radians(deg)
        qz = [math.cos(th / 2.0), 0.0, 0.0, math.sin(th / 2.0)]
        q = quat_mul(qz, home_q)
        R = quat_to_rot(q)
        ax = (R[0][1], R[1][1])
        clear = 1.0
        for s in (-1.0, 1.0):
            fx = target_xy[0] + s * 0.05 * ax[0]
            fy = target_xy[1] + s * 0.05 * ax[1]
            for p in others:
                d = math.sqrt((p[0] - fx) ** 2 + (p[1] - fy) ** 2)
                if d < clear:
                    clear = d
        # mild preference for small wrist rotations
        score = clear - 0.0001 * abs(deg)
        if score > best_clear:
            best_clear = score
            best_q = q
            best_deg = deg
    return best_q, best_deg, best_clear


def gripper_now():
    return float(get_robot_state().gripper_positions["primary"])


def pick_and_place(target_xy, grasp_z, quat, frame, label):
    # Returns (ok, terminal)
    lift_z = grasp_z + 0.12
    ok, term = ex_ok(open_gripper(), label + ":open")
    if term:
        return False, True
    ok, term = ex_ok(move_to_pose(pose_at(target_xy, lift_z, quat, frame), tolerance=0.01, max_steps=200), label + ":pregrasp")
    if term:
        return False, True
    ok, term = ex_ok(move_to_pose(pose_at(target_xy, grasp_z, quat, frame), tolerance=0.008, max_steps=120), label + ":descend")
    if term:
        return False, True
    ok, term = ex_ok(close_gripper(), label + ":close")
    if term:
        return False, True
    g = gripper_now()
    report[label + "_grip_closed"] = g
    ok, term = ex_ok(move_to_pose(pose_at(target_xy, lift_z, quat, frame), tolerance=0.01, max_steps=120), label + ":lift")
    if term:
        return False, True
    report[label + "_grip_lifted"] = gripper_now()
    ok, term = ex_ok(move_to_pose(pose_at(target_xy, grasp_z + 0.004, quat, frame), tolerance=0.008, max_steps=120), label + ":lower")
    if term:
        return False, True
    ok, term = ex_ok(open_gripper(), label + ":release")
    if term:
        return False, True
    ok, term = ex_ok(move_to_pose(pose_at(target_xy, lift_z, quat, frame), tolerance=0.01, max_steps=120), label + ":retreat")
    if term:
        return False, True
    return True, False


# ---------------------------------------------------------------- main
context = get_task_context()
report["language"] = context.language
state0 = get_robot_state()
base_frame = state0.base_frame
home_q = [float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz]
marker("before_video")
video = robomme_plus.get_conditioning_video()
marker("video_loaded")
info = analyse_video(video)
marker("video_tracked")
for key in ("frames", "grasp_index", "release_index", "release_eef", "grasp_eef", "track_start",
            "table_z", "release_points", "object_height", "band", "initial_tracks", "target_index",
            "target_init_dist_m", "target_path", "final_tracks", "path_m", "born", "spawned", "ok", "why"):
    if key in info:
        report["video_" + key] = info[key]

done = False
if info["ok"]:
    # Return to the viewpoint the video ended in (recorded joint configuration of the last frame).
    ok, term = ex_ok(move_to_joints(info["final_joints"], tolerance=0.01, max_steps=150), "view:video_final_pose")
    clusters = live_clusters(info)
    marker("live_clusters")
    tf = info["target_final"]
    best = None
    bd = None
    for c in clusters:
        dd = math.sqrt((c["xy"][0] - tf[0]) ** 2 + (c["xy"][1] - tf[1]) ** 2)
        if bd is None or dd < bd:
            bd = dd
            best = c
    report["live_clusters"] = [[round(c["xy"][0], 4), round(c["xy"][1], 4), c["n"]] for c in clusters]
    if best is None:
        note("no live cluster")
    else:
        report["live_match_dist_m"] = round(bd, 4)
        table_z = info["table_z"]
        height = info["object_height"]
        txy = best["xy"]
        report["target_xy"] = [round(txy[0], 4), round(txy[1], 4)]
        others = []
        for c in clusters:
            if c is best:
                continue
            for p in c["pts"]:
                others.append(p)
        quat, deg, clear = choose_yaw(home_q, txy, others)
        report["grasp_yaw_deg"] = deg
        report["grasp_clearance_m"] = round(clear, 4)
        grasp_z = table_z + max(0.55 * height, 0.012)
        report["grasp_z"] = round(grasp_z, 4)
        # Button: a fixture that stayed put while the objects were shuffled in the video and is
        # not part of a tall structure (points above the band right next to it).
        paths = info["path_m"]
        finals = info["final_tracks"]
        tidx = info["target_index"]
        tall = live_points(z_hi_of(info), table_z + 0.15)
        cand = [j for j in range(len(paths)) if j != tidx and info["born"][j] == info["track_start"]]
        statics = []
        if len(cand) > 1:
            med_path = quantile([paths[j] for j in cand], 0.5)
            for j in cand:
                if paths[j] >= 0.25 * med_path:
                    continue
                sc = None
                bdd = None
                for c in clusters:
                    dd = (c["xy"][0] - finals[j][0]) ** 2 + (c["xy"][1] - finals[j][1]) ** 2
                    if dd < 0.04 * 0.04 and (bdd is None or dd < bdd):
                        bdd = dd
                        sc = c
                if sc is None or sc is best:
                    continue
                ntall = 0
                for q in tall:
                    if (q[0] - sc["xy"][0]) ** 2 + (q[1] - sc["xy"][1]) ** 2 < 0.04 * 0.04:
                        ntall += 1
                report["static_candidates"] = report.get("static_candidates", []) + [
                    [j, paths[j], round(sc["xy"][0], 4), round(sc["xy"][1], 4), ntall]]
                if ntall < 5:
                    statics.append((paths[j], sc))
        sam_xy = []
        for prompt in ("button", "push button"):
            seg = segment_text("front", prompt)
            if not seg.ok:
                continue
            for cnd in seg.segmentations[:5]:
                try:
                    pc = mask_to_point_cloud(cnd, "front", target_frame=base_frame)
                except Exception as exc:
                    note("button pc: " + str(exc))
                    continue
                pts = pc.points
                if len(pts) < 10:
                    continue
                zsb = [float(p[2]) for p in pts]
                top = quantile(zsb, 0.9)
                if top - table_z > 0.15:
                    continue
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
                if max(xs) - min(xs) > 0.2 or max(ys) - min(ys) > 0.2:
                    continue
                bxy = mean_xy([p for p in pts if float(p[2]) >= top - 0.008])
                if bxy[0] * bxy[0] + bxy[1] * bxy[1] < BASE_EXCLUSION_M * BASE_EXCLUSION_M:
                    continue
                sam_xy.append({"xy": bxy, "top": top, "prompt": prompt})
        report["sam_button_candidates"] = [[round(c["xy"][0], 4), round(c["xy"][1], 4), round(c["top"], 4)] for c in sam_xy]
        button = None
        chosen = None
        for st in statics:
            for c in sam_xy:
                if math.sqrt((c["xy"][0] - st[1]["xy"][0]) ** 2 + (c["xy"][1] - st[1]["xy"][1]) ** 2) < 0.04:
                    chosen = st[1]
                    break
            if chosen is not None:
                break
        if chosen is None and len(statics) > 0:
            bp = None
            for st in statics:
                if bp is None or st[0] < bp:
                    bp = st[0]
                    chosen = st[1]
        if chosen is not None:
            zsb = [p[2] for p in chosen["pts"]]
            top = quantile(zsb, 0.9)
            capp = [p for p in chosen["pts"] if p[2] >= top - 0.008]
            button = {"xy": mean_xy(capp), "top": top, "source": "static_track"}
        else:
            for c in sam_xy:
                near_obj = False
                for cl in clusters:
                    if math.sqrt((c["xy"][0] - cl["xy"][0]) ** 2 + (c["xy"][1] - cl["xy"][1]) ** 2) < 0.04:
                        near_obj = True
                ntall = 0
                for q in tall:
                    if (q[0] - c["xy"][0]) ** 2 + (q[1] - c["xy"][1]) ** 2 < 0.04 * 0.04:
                        ntall += 1
                if not near_obj and ntall < 5:
                    button = {"xy": c["xy"], "top": c["top"], "source": "sam:" + c["prompt"]}
                    break
        if button is not None:
            report["button"] = {"xy": [round(button["xy"][0], 4), round(button["xy"][1], 4)],
                                "top": round(button["top"], 4), "source": button["source"]}
        else:
            note("button not found")
        ok1, term = pick_and_place(txy, grasp_z, quat, base_frame, "pick1")
        report["pick1_done"] = ok1
        if not term:
            txy2 = txy
            try:
                clusters2 = live_clusters(info)
                bd2 = None
                for c in clusters2:
                    dd = math.sqrt((c["xy"][0] - txy[0]) ** 2 + (c["xy"][1] - txy[1]) ** 2)
                    if bd2 is None or dd < bd2:
                        bd2 = dd
                        txy2 = c["xy"]
                if bd2 is None or bd2 > 0.03:
                    txy2 = txy
                report["pick2_relocalize_shift_m"] = bd2
            except Exception as exc:
                note("relocalize: " + str(exc))
            report["target_xy2"] = [round(txy2[0], 4), round(txy2[1], 4)]
            ok2, term = pick_and_place(txy2, grasp_z, quat, base_frame, "pick2")
            report["pick2_done"] = ok2
        if not term and button is not None:
            ok, term = ex_ok(close_gripper(), "button:close")
            if not term:
                above = pose_at(button["xy"], button["top"] + 0.10, home_q, base_frame)
                ok, term = ex_ok(move_to_pose(above, tolerance=0.01, max_steps=200), "button:above")
            if not term:
                press = pose_at(button["xy"], button["top"] - 0.012, home_q, base_frame)
                ok, term = ex_ok(move_to_pose(press, tolerance=0.006, max_steps=80), "button:press")
            if not term:
                up = pose_at(button["xy"], button["top"] + 0.08, home_q, base_frame)
                ok, term = ex_ok(move_to_pose(up, tolerance=0.01, max_steps=80), "button:up")
        report["terminal_seen"] = term
        done = True

report["phase"] = "done" if done else "aborted"
result = report
