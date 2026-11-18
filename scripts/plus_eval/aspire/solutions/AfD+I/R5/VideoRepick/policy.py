# VideoRepick v01 — (v00 + button localised at home before picking) track the video-picked object through the shuffle, pick/place it twice, press button.
import math

report = {"version": "v01"}
notes = []

# ---------------------------------------------------------------- utilities
def to_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        return arr

def quat_to_R(q):
    w = float(q[0]); x = float(q[1]); y = float(q[2]); z = float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w = w / n; x = x / n; y = y / n; z = z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]

def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)

def quantile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]

def cam_params(cam):
    K = to_rows(cam.intrinsics)
    pose = cam.camera_pose
    R = quat_to_R(pose.quaternion_wxyz)
    t = [float(v) for v in pose.position]
    return (float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2]), R, t)

def project_grid(cam, conv, stride, roi=None):
    """All valid depth pixels on a grid -> list of (x, y, z) in robot_base."""
    fx, fy, cx, cy, R, t = cam_params(cam)
    dep = to_rows(cam.depth_m)
    H = len(dep); W = len(dep[0])
    sy = 1.0 if conv == "opencv" else -1.0
    sz = 1.0 if conv == "opencv" else -1.0
    out = []
    u0, v0, u1, v1 = (0, 0, W, H) if roi is None else roi
    for v in range(v0, v1, stride):
        drow = dep[v]
        yn = sy * (v - cy) / fy
        for u in range(u0, u1, stride):
            d = float(drow[u])
            if d <= 0.0 or d > 10.0:
                continue
            xc = (u - cx) / fx * d; yc = yn * d; zc = sz * d
            out.append((R[0][0] * xc + R[0][1] * yc + R[0][2] * zc + t[0],
                        R[1][0] * xc + R[1][1] * yc + R[1][2] * zc + t[1],
                        R[2][0] * xc + R[2][1] * yc + R[2][2] * zc + t[2]))
    return out

def object_points(cam, conv, zlo, zhi, sat_min, stride, roi=None, excl=None, keep_uv=False):
    """Pixels raised above the table and coloured (not grey) -> (x, y, z[, u, v])."""
    fx, fy, cx, cy, R, t = cam_params(cam)
    dep = to_rows(cam.depth_m)
    rgb = to_rows(cam.rgb)
    H = len(dep); W = len(dep[0])
    sy = 1.0 if conv == "opencv" else -1.0
    sz = 1.0 if conv == "opencv" else -1.0
    r00, r01, r02 = R[0]; r10, r11, r12 = R[1]; r20, r21, r22 = R[2]
    t0, t1, t2 = t
    u0, v0, u1, v1 = (0, 0, W, H) if roi is None else roi
    ex = None
    if excl is not None:
        ex, ey, er2 = excl[0], excl[1], excl[2] * excl[2]
    out = []
    for v in range(v0, v1, stride):
        drow = dep[v]; crow = rgb[v]
        yn = sy * (v - cy) / fy
        for u in range(u0, u1, stride):
            d = float(drow[u])
            if d <= 0.0 or d > 10.0:
                continue
            xc = (u - cx) / fx * d; yc = yn * d; zc = sz * d
            z = r20 * xc + r21 * yc + r22 * zc + t2
            if z < zlo or z > zhi:
                continue
            px = crow[u]
            r = px[0]; g = px[1]; b = px[2]
            if max(r, g, b) - min(r, g, b) < sat_min:
                continue
            x = r00 * xc + r01 * yc + r02 * zc + t0
            y = r10 * xc + r11 * yc + r12 * zc + t1
            if ex is not None and (x - ex) * (x - ex) + (y - ey) * (y - ey) < er2:
                continue
            if keep_uv:
                out.append((x, y, z, u, v))
            else:
                out.append((x, y, z))
    return out

def grid_clusters(pts, cell=0.01, min_pts=4):
    grid = {}
    for i in range(len(pts)):
        p = pts[i]
        key = (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)))
        if key in grid:
            grid[key].append(i)
        else:
            grid[key] = [i]
    seen = set()
    clusters = []
    for key in grid:
        if key in seen:
            continue
        seen.add(key)
        stack = [key]; members = []
        while len(stack) > 0:
            k = stack.pop()
            members.extend(grid[k])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nk = (k[0] + dx, k[1] + dy)
                    if nk in grid and nk not in seen:
                        seen.add(nk); stack.append(nk)
        if len(members) >= min_pts:
            clusters.append([pts[i] for i in members])
    return clusters

def centroid_xy(pts):
    n = float(len(pts))
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)

def track_update(tracks, vel, pts, gate=0.035, iters=3, min_pts=3):
    """Gated nearest-centre assignment (k-means style) with velocity prediction."""
    K = len(tracks)
    pred = [(tracks[k][0] + vel[k][0], tracks[k][1] + vel[k][1]) for k in range(K)]
    cen = list(pred)
    g2 = gate * gate
    groups = [[] for k in range(K)]
    for it in range(iters):
        groups = [[] for k in range(K)]
        for p in pts:
            best = -1; bd = g2
            for k in range(K):
                dx = p[0] - cen[k][0]; dy = p[1] - cen[k][1]
                dd = dx * dx + dy * dy
                if dd < bd:
                    bd = dd; best = k
            if best >= 0:
                groups[best].append(p)
        newc = []
        for k in range(K):
            if len(groups[k]) >= min_pts:
                newc.append(centroid_xy(groups[k]))
            else:
                newc.append(pred[k])
        cen = newc
    newvel = []
    for k in range(K):
        if len(groups[k]) >= min_pts:
            newvel.append((0.5 * vel[k][0] + 0.5 * (cen[k][0] - tracks[k][0]),
                           0.5 * vel[k][1] + 0.5 * (cen[k][1] - tracks[k][1])))
        else:
            newvel.append((0.5 * vel[k][0], 0.5 * vel[k][1]))
    return cen, newvel, groups

# ---------------------------------------------------------------- context
ctx = get_task_context()
report["goal"] = ctx.language
try:
    meta = robomme_plus.get_task_metadata()
    report["max_steps"] = meta["max_steps"]
    try:
        report["video_sampling"] = meta["conditioning_video_sampling"]
    except Exception:
        pass
except Exception as exc:
    notes.append("meta: " + str(exc))

state0 = get_robot_state()
base = state0.base_frame
q_home = tuple(float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz)
report["home_ee"] = [round(float(v), 4) for v in state0.end_effector_poses["primary"].position]
report["home_q"] = [round(v, 4) for v in q_home]
obs0 = get_observation()

# ---------------------------------------------------------------- camera convention + table plane
def plane_stats(cam, conv):
    """Score a pose convention: the table must lie BELOW the camera, then be flat (small z IQR)."""
    pts = project_grid(cam, conv, 8)
    zs = [p[2] for p in pts]
    camz = float(cam.camera_pose.position[2])
    below = len([z for z in zs if z < camz]) / float(max(1, len(zs)))
    iqr = quantile(zs, 0.75) - quantile(zs, 0.25)
    return (1.0 - below) + iqr, zs

conv = "opencv"
table_z = None
try:
    iqr_cv, zs_cv = plane_stats(obs0.cameras["front"], "opencv")
    iqr_gl, zs_gl = plane_stats(obs0.cameras["front"], "opengl")
    report["conv_iqr"] = [round(iqr_cv, 4), round(iqr_gl, 4)]
    if iqr_gl < iqr_cv:
        conv = "opengl"; zs = zs_gl
    else:
        zs = zs_cv
    # table height = most populated 5 mm z bin
    bins = {}
    for z in zs:
        kb = int(math.floor(z / 0.005))
        bins[kb] = bins.get(kb, 0) + 1
    bestb = None
    for kb in bins:
        if bestb is None or bins[kb] > bins[bestb]:
            bestb = kb
    near = [z for z in zs if abs(z - (bestb + 0.5) * 0.005) < 0.006]
    table_z = quantile(near, 0.5)
except Exception as exc:
    notes.append("plane: " + str(exc))
report["conv"] = conv
report["table_z"] = table_z
if table_z is None:
    table_z = 0.0

ZLO = table_z + 0.005
ZHI = table_z + 0.06
SAT = 14

# ---------------------------------------------------------------- button (localised at home, BEFORE picking:
# the arm hovering over the target occludes it later — ep1 v00 failure)
def button_geo_clusters():
    """Raised, grey/white, compact clusters (not the robot's own base) from the front camera."""
    cam = obs0.cameras["front"]
    fx, fy, cx, cy, R, t = cam_params(cam)
    dep = to_rows(cam.depth_m); rgb = to_rows(cam.rgb)
    sy = 1.0 if conv == "opencv" else -1.0
    pts = []
    for v in range(0, len(dep)):
        drow = dep[v]; crow = rgb[v]
        yn = sy * (v - cy) / fy
        for u in range(0, len(drow)):
            d = float(drow[u])
            if d <= 0.0 or d > 10.0:
                continue
            px = crow[u]
            sat = max(px[0], px[1], px[2]) - min(px[0], px[1], px[2])
            if not (sat < 22 or min(px[0], px[1], px[2]) > 170):
                continue
            xc = (u - cx) / fx * d; yc = yn * d; zc = sy * d
            z = R[2][0] * xc + R[2][1] * yc + R[2][2] * zc + t[2]
            if z < table_z + 0.006 or z > table_z + 0.06:
                continue
            x = R[0][0] * xc + R[0][1] * yc + R[0][2] * zc + t[0]
            y = R[1][0] * xc + R[1][1] * yc + R[1][2] * zc + t[1]
            if x * x + y * y < 0.2 * 0.2:
                continue
            pts.append((x, y, z))
    out = []
    for c in grid_clusters(pts, 0.01, 8):
        xs = [p[0] for p in c]; ys = [p[1] for p in c]
        ext = max(max(xs) - min(xs), max(ys) - min(ys))
        if 0.012 < ext < 0.08:
            out.append({"xy": [round(quantile(xs, 0.5), 4), round(quantile(ys, 0.5), 4)],
                        "ztop": round(quantile([p[2] for p in c], 0.95), 4), "n": len(c), "ext": round(ext, 4)})
    return out

button = None
cands = []
geo = []
try:
    geo = button_geo_clusters()
except Exception as exc:
    notes.append("geo button: " + str(exc))
report["button_geo"] = geo[:8]
for prompt in ("button", "white button", "push button"):
    try:
        seg = segment_text("front", prompt)
    except Exception as exc:
        notes.append("seg " + prompt + ": " + str(exc))
        continue
    if not seg.ok:
        continue
    for c in seg.segmentations[:5]:
        try:
            pc = mask_to_point_cloud(c, "front", target_frame=base)
            P = [(float(p[0]), float(p[1]), float(p[2])) for p in pc.points]
        except Exception as exc:
            notes.append("pc: " + str(exc))
            continue
        if len(P) < 5:
            continue
        xs = [p[0] for p in P]; ys = [p[1] for p in P]; zz = [p[2] for p in P]
        ext = max(quantile(xs, 0.95) - quantile(xs, 0.05), quantile(ys, 0.95) - quantile(ys, 0.05))
        zmed = quantile(zz, 0.5)
        rec = {"prompt": prompt, "score": round(float(c.score), 4), "n": len(P),
               "xy": [round(quantile(xs, 0.5), 4), round(quantile(ys, 0.5), 4)],
               "ztop": round(quantile(zz, 0.95), 4), "ext": round(ext, 4), "zmed": round(zmed, 4)}
        rec["valid"] = ext < 0.10 and zmed < table_z + 0.08 and zmed > table_z - 0.01
        gd = None
        for g in geo:
            dd = math.sqrt((g["xy"][0] - rec["xy"][0]) ** 2 + (g["xy"][1] - rec["xy"][1]) ** 2)
            if gd is None or dd < gd:
                gd = dd
        rec["geo_dist"] = None if gd is None else round(gd, 4)
        cands.append(rec)
    if len([c for c in cands if c["valid"]]) > 0:
        break
report["button_cands"] = cands
valid = [c for c in cands if c["valid"]]
agreeing = [c for c in valid if c["geo_dist"] is not None and c["geo_dist"] < 0.03]
pool = agreeing if len(agreeing) > 0 else valid
for c in pool:
    if button is None or c["score"] > button["score"]:
        button = c
if button is None and len(geo) > 0:
    for g in geo:
        if button is None or g["n"] > button["n"]:
            button = g
    button = dict(button); button["source"] = "geo"
report["button"] = button

# ---------------------------------------------------------------- conditioning video: pick event + tracking
target_xy = None
tracks = []
vel = []
target_k = -1
try:
    video = robomme_plus.get_conditioning_video()
    n = len(video)
    report["video_frames"] = n
    grip = []
    eep = []
    for i in range(n):
        rs = video[i].robot_state
        grip.append(float(rs.gripper_positions["primary"]))
        p = rs.end_effector_poses["primary"].position
        eep.append((float(p[0]), float(p[1]), float(p[2])))
    gmax = max(grip)
    close_i = -1
    for i in range(n):
        if grip[i] < gmax - 0.2:
            close_i = i
            break
    release_i = -1
    if close_i >= 0:
        for i in range(close_i, n):
            if grip[i] > gmax - 0.08:
                release_i = i
                break
    report["grip_events"] = {"gmax": round(gmax, 3), "close": close_i, "release": release_i,
                             "ee_close": [round(v, 4) for v in eep[close_i]] if close_i >= 0 else None,
                             "ee_release": [round(v, 4) for v in eep[release_i]] if release_i >= 0 else None}
    report["grip_trace"] = [round(grip[i], 2) for i in range(0, n, 8)]
    report["ee_z_trace"] = [round(eep[i][2], 3) for i in range(0, n, 8)]
    if release_i < 0:
        # fallback: lowest EE frame in the first half
        lo = 0
        for i in range(n // 2):
            if eep[i][2] < eep[lo][2]:
                lo = i
        release_i = lo
        notes.append("no grip release event; using lowest-EE frame")

    # frame 0: count objects, front ROI
    f0 = video[0]
    p_front0 = object_points(f0.cameras["front"], conv, ZLO, ZHI, SAT, 1, keep_uv=True)
    p_wrist0 = object_points(f0.cameras["wrist"], conv, ZLO, ZHI, SAT, 2)
    us = [p[3] for p in p_front0]; vs = [p[4] for p in p_front0]
    if len(us) > 0:
        roi = (max(0, min(us) - 50), max(0, min(vs) - 50), min(256, max(us) + 51), min(256, max(vs) + 51))
    else:
        roi = None
    report["front_roi"] = roi
    init_pts = [(p[0], p[1], p[2]) for p in p_front0] + p_wrist0
    cl = grid_clusters(init_pts, 0.01, 6)
    tracks = [centroid_xy(c) for c in cl]
    vel = [(0.0, 0.0) for c in cl]
    report["K"] = len(tracks)
    report["init_tracks"] = [[round(c[0], 3), round(c[1], 3)] for c in tracks]
    report["init_cluster_sizes"] = [len(c) for c in cl]

    stride_f = 1
    while n // stride_f > 260:
        stride_f += 1
    frames = list(range(1, n, stride_f))
    if frames[-1] != n - 1:
        frames.append(n - 1)
    hist = []
    spawns = []
    target_set = False
    counts_log = []
    for i in frames:
        fr = video[i]
        ex = None
        if eep[i][2] < table_z + 0.2:
            ex = (eep[i][0], eep[i][1], 0.06)
        pts = object_points(fr.cameras["front"], conv, ZLO, ZHI, SAT, 2, roi=roi, excl=ex)
        pts = pts + object_points(fr.cameras["wrist"], conv, ZLO, ZHI, SAT, 4, excl=ex)
        if (not target_set) and i >= release_i:
            rx, ry = eep[release_i][0], eep[release_i][1]
            bk = 0; bd = None
            for k in range(len(tracks)):
                dd = (tracks[k][0] - rx) ** 2 + (tracks[k][1] - ry) ** 2
                if bd is None or dd < bd:
                    bd = dd; bk = k
            if len(tracks) == 0:
                tracks.append((rx, ry)); vel.append((0.0, 0.0)); bd = 0.0
            target_k = bk
            report["target_init_dist"] = round(math.sqrt(bd), 4) if bd is not None else None
            tracks[bk] = (rx, ry); vel[bk] = (0.0, 0.0)
            target_set = True
            ex = None
        tracks, vel, groups = track_update(tracks, vel, pts)
        if ex is None and len(tracks) < 14 and len(pts) > 0:
            # spawn tracks for coloured raised clusters no track explains (undercount at frame 0)
            far = []
            for p in pts:
                ok_far = True
                for c in tracks:
                    if (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 < 0.045 * 0.045:
                        ok_far = False
                        break
                if ok_far:
                    far.append(p)
            if len(far) >= 10:
                for c in grid_clusters(far, 0.01, 10):
                    tracks.append(centroid_xy(c)); vel.append((0.0, 0.0)); groups.append(c)
                    spawns.append((i, round(tracks[-1][0], 3), round(tracks[-1][1], 3), len(c)))
        counts_log.append(len(pts))
        if target_k >= 0:
            hist.append((i, round(tracks[target_k][0], 3), round(tracks[target_k][1], 3), len(groups[target_k])))
    report["spawns"] = spawns[:20]
    report["n_tracked_frames"] = len(frames)
    report["pts_per_frame"] = [counts_log[j] for j in range(0, len(counts_log), 10)]
    report["target_hist"] = [hist[j] for j in range(0, len(hist), 6)]
    report["final_tracks"] = [[round(c[0], 3), round(c[1], 3)] for c in tracks]
    report["target_k"] = target_k
    if target_k >= 0:
        target_xy = tracks[target_k]
except Exception as exc:
    notes.append("video: " + str(exc))

report["target_xy_end_video"] = None if target_xy is None else [round(target_xy[0], 4), round(target_xy[1], 4)]

# ---------------------------------------------------------------- live localisation of the target
def live_points():
    o = get_observation()
    pts = object_points(o.cameras["front"], conv, ZLO, ZHI, SAT, 1)
    pts = pts + object_points(o.cameras["wrist"], conv, ZLO, ZHI, SAT, 2)
    return pts

def locate_target(prior_xy):
    """Assign live points to the tracks (target prior overrides) and return target cloud."""
    pts = live_points()
    if len(tracks) == 0:
        cl = grid_clusters(pts, 0.01, 6)
        if len(cl) == 0:
            return None, pts
        best = None; bd = None
        for c in cl:
            cxy = centroid_xy(c)
            dd = (cxy[0] - prior_xy[0]) ** 2 + (cxy[1] - prior_xy[1]) ** 2
            if bd is None or dd < bd:
                bd = dd; best = c
        return best, pts
    tr = list(tracks)
    tr[target_k] = prior_xy
    zero = [(0.0, 0.0) for k in range(len(tr))]
    cen, v2, groups = track_update(tr, zero, pts, gate=0.03)
    for k in range(len(cen)):
        tracks[k] = cen[k]
    return groups[target_k], pts

def neighbour_yaw_offset(txy, others):
    """Yaw offset so the finger closing axis is perpendicular to the nearest neighbour."""
    best = None; bd = None
    for o in others:
        dd = (o[0] - txy[0]) ** 2 + (o[1] - txy[1]) ** 2
        if dd > 1e-6 and (bd is None or dd < bd):
            bd = dd; best = o
    if best is None or bd > 0.09 * 0.09:
        return 0.0
    theta = math.atan2(best[1] - txy[1], best[0] - txy[0])
    Rh = quat_to_R(q_home)
    phi = math.atan2(Rh[1][1], Rh[0][1])     # current finger (hand y) axis direction
    delta = theta + math.pi / 2 - phi
    while delta > math.pi / 2:
        delta -= math.pi
    while delta < -math.pi / 2:
        delta += math.pi
    return delta

Rh0 = quat_to_R(q_home)
if Rh0[2][2] > -0.9:
    q_down = (0.0, 1.0, 0.0, 0.0)
    notes.append("home not pointing down; using (0,1,0,0)")
else:
    q_down = q_home

def yawed(delta):
    return qmul((math.cos(delta / 2), 0.0, 0.0, math.sin(delta / 2)), q_down)

ended = [False]

def mv(xyz, q, tol=0.01, steps=150, label=""):
    if ended[0]:
        return None
    res = move_to_pose(Pose((xyz[0], xyz[1], xyz[2]), q, base), tolerance=tol, max_steps=steps)
    rec = {"label": label, "ok": bool(res.ok), "steps": int(res.steps_executed)}
    try:
        rec["status"] = str(res.status)
    except Exception:
        pass
    try:
        if res.terminated or res.truncated:
            ended[0] = True
            rec["terminal"] = [bool(res.terminated), bool(res.truncated)]
    except Exception:
        pass
    motions.append(rec)
    return res

def grip(close):
    if ended[0]:
        return None
    res = close_gripper() if close else open_gripper()
    rec = {"label": "close" if close else "open", "ok": bool(res.ok), "steps": int(res.steps_executed)}
    try:
        if res.terminated or res.truncated:
            ended[0] = True
            rec["terminal"] = [bool(res.terminated), bool(res.truncated)]
    except Exception:
        pass
    motions.append(rec)
    return res

motions = []
picks = []
place_xy = None
place_z = None
yaw_d = 0.0
for cycle in range(2):
    if ended[0]:
        break
    prior = place_xy if place_xy is not None else target_xy
    if prior is None:
        notes.append("no target prior; aborting picks")
        break
    tgt, livepts = locate_target(prior)
    info = {"cycle": cycle, "prior": [round(prior[0], 4), round(prior[1], 4)], "n_live": len(livepts)}
    if tgt is None or len(tgt) < 3:
        info["fallback"] = True
        gx, gy = prior
        top = table_z + 0.02
    else:
        gx, gy = centroid_xy(tgt)
        top = quantile([p[2] for p in tgt], 0.9)
        info["n_target_pts"] = len(tgt)
        if place_xy is not None and ((gx - place_xy[0]) ** 2 + (gy - place_xy[1]) ** 2) > 0.03 ** 2:
            info["redetect_far"] = True
            gx, gy = place_xy
    h = top - table_z
    gz = table_z + min(max(0.5 * h, 0.008), 0.03)
    others = [tracks[k] for k in range(len(tracks)) if k != target_k]
    yaw_d = neighbour_yaw_offset((gx, gy), others)
    q = yawed(yaw_d)
    info.update({"grasp": [round(gx, 4), round(gy, 4), round(gz, 4)], "top": round(top, 4),
                 "yaw_deg": round(yaw_d * 180 / math.pi, 1)})
    picks.append(info)
    grip(False)
    mv((gx, gy, gz + 0.10), q, 0.015, 200, "pre%d" % cycle)
    mv((gx, gy, gz), q, 0.008, 120, "grasp%d" % cycle)
    grip(True)
    mv((gx, gy, gz + 0.10), q, 0.015, 120, "lift%d" % cycle)
    try:
        info["grip_after_lift"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
    except Exception:
        pass
    mv((gx, gy, gz + 0.004), q, 0.01, 120, "place%d" % cycle)
    grip(False)
    mv((gx, gy, gz + 0.10), q, 0.015, 120, "retreat%d" % cycle)
    place_xy = (gx, gy)

report["picks"] = picks

# ---------------------------------------------------------------- button press
if not ended[0] and button is not None:
    bx, by = button["xy"]
    bz = button["ztop"]
    grip(True)
    mv((bx, by, bz + 0.08), q_down, 0.015, 200, "btn_pre")
    mv((bx, by, bz - 0.015), q_down, 0.01, 80, "btn_press")
    mv((bx, by, bz + 0.06), q_down, 0.015, 80, "btn_up")

report["motions"] = motions
report["ended"] = ended[0]
report["notes"] = notes
result = report
