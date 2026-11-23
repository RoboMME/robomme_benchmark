# VideoRepick v03 -- button role: among "button" segmentation candidates that are low and compact,
# keep those whose location was occupied in >=80% of the video frames (static), never the target;
# highest SAM score wins (replaces v02's fragile moved-track rule).
# VideoRepick v02 -- button: segment "button" in wrist+front at the start (arm at home), keep only
# candidates whose projected cloud is a low compact object (top < table+0.07, xy extent < 0.09);
# press, and press once more deeper if the episode did not end.
# VideoRepick v01 -- grasp: centre on the object's xy mid-range from a low wrist hover and
# yaw the jaw so it closes perpendicular to the largest empty angular sector (C-shell gap);
# verify each grasp by gripper reading and retry (re-observe, +45 deg yaw) up to 3 times.
# (v00 header follows)
# VideoRepick v00 -- initial program from offline analysis of the released recordings.
# Strategy (all quantities derived at runtime):
#  1. From the conditioning video's robot_state, find when the gripper closed / reopened
#     (the demonstrated pick) and where the end effector released the object.
#  2. Back-project depth of BOTH cameras (front + wrist) of every video frame after the
#     release into robot_base; keep points in a height band above the table plane.
#  3. Initialise one track per object cluster; the target is the cluster nearest the
#     release point. Track all objects jointly (k-means seeded by predicted positions)
#     through the shuffle to the last video frame.
#  4. In the live scene choose the object cluster nearest the target's final position,
#     pick it up and place it down twice (re-localising from the wrist camera each time),
#     then press the button (segmented by the goal noun "button").
import math

report = {"version": "v03"}
state0 = get_robot_state()
base_frame = state0.base_frame
embodiment = state0.embodiment
goal = get_task_context().language
report["goal"] = goal
try:
    meta = robomme_plus.get_task_metadata()
    report["max_steps"] = meta["max_steps"]
    try:
        report["sampling"] = meta["conditioning_video_sampling"]
    except Exception:
        pass
except Exception as exc:
    report["meta_error"] = str(exc)

HMIN = 0.007     # generic: min height above table for an object point
HMAX = 0.09      # generic: max height above table for an object point
STRIDE = 2       # pixel stride for depth scans
GATE = 0.05      # tracker association gate (m)
CELL = 0.01      # clustering grid (m)


def to_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        return [list(r) for r in arr]


def quat_to_r(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def cam_dir(conv, a, b):
    # a=(u-cx)/fx, b=(v-cy)/fy ; direction for unit depth in camera axes
    if conv == "cv":
        return (a, b, 1.0)
    if conv == "gl":
        return (a, -b, -1.0)
    return (1.0, -a, -b)  # "ros": x forward, y left, z up


def make_grid(cam, conv, stride):
    K = to_rows(cam.intrinsics)
    fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
    pose = cam.camera_pose
    R = quat_to_r(pose.quaternion_wxyz)
    t = [float(v) for v in pose.position]
    H = len(cam.depth_m)
    W = len(cam.depth_m[0])
    gx, gy, gz = [], [], []
    for r in range(0, H, stride):
        b = (r - cy) / fy
        rx, ry, rz = [], [], []
        for c in range(0, W, stride):
            a = (c - cx) / fx
            d = cam_dir(conv, a, b)
            rx.append(R[0][0] * d[0] + R[0][1] * d[1] + R[0][2] * d[2])
            ry.append(R[1][0] * d[0] + R[1][1] * d[1] + R[1][2] * d[2])
            rz.append(R[2][0] * d[0] + R[2][1] * d[1] + R[2][2] * d[2])
        gx.append(rx)
        gy.append(ry)
        gz.append(rz)
    key = tuple(round(v, 4) for v in t) + tuple(round(float(v), 4) for v in pose.quaternion_wxyz)
    return {"x": gx, "y": gy, "z": gz, "t": t, "s": stride, "key": key}


def pose_key(cam):
    pose = cam.camera_pose
    return tuple(round(float(v), 4) for v in pose.position) + tuple(
        round(float(v), 4) for v in pose.quaternion_wxyz)


def quantile(vals, f):
    o = sorted(vals)
    if len(o) == 0:
        return None
    return o[int((len(o) - 1) * f)]


def table_stats(cam, conv):
    g = make_grid(cam, conv, 8)
    D = to_rows(cam.depth_m)
    zs = []
    for ri in range(len(g["z"])):
        row = D[ri * 8]
        for ci in range(len(g["z"][ri])):
            Z = float(row[ci * 8])
            if Z > 0.01 and Z < 20.0:
                zs.append(g["t"][2] + Z * g["z"][ri][ci])
    if len(zs) < 50:
        return None
    q25 = quantile(zs, 0.25)
    q75 = quantile(zs, 0.75)
    med = quantile(zs, 0.5)
    if med > g["t"][2]:
        return None  # table above the camera: impossible convention
    return {"iqr": q75 - q25, "median": med, "n": len(zs)}


GRIDS = {}


def cam_points(obs, name, conv, table_z, zcap, stride=STRIDE):
    cam = obs.cameras[name]
    key = pose_key(cam)
    g = GRIDS.get(name)
    if g is None or g["key"] != key or g["s"] != stride:
        g = make_grid(cam, conv, stride)
        g["key"] = key
        GRIDS[name] = g
    D = to_rows(cam.depth_m)
    tx, ty, tz = g["t"]
    lo = table_z + HMIN
    hi = min(table_z + HMAX, zcap)
    pts = []
    gzr = g["z"]
    for ri in range(len(gzr)):
        row = D[ri * stride]
        rz = gzr[ri]
        rxr = g["x"][ri]
        ryr = g["y"][ri]
        for ci in range(len(rz)):
            Z = row[ci * stride]
            if Z > 0.01 and Z < 20.0:
                z = tz + Z * rz[ci]
                if z > lo and z < hi:
                    pts.append((tx + Z * rxr[ci], ty + Z * ryr[ci], 1.0, z))
    return pts


def cluster(pts, cell=CELL, minw=8):
    cells = {}
    for p in pts:
        k = (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)))
        e = cells.get(k)
        if e is None:
            e = [0.0, 0.0, 0.0, []]
            cells[k] = e
        e[0] += p[0] * p[2]
        e[1] += p[1] * p[2]
        e[2] += p[2]
        e[3].append(p[3])
    seen = set()
    out = []
    for k in cells:
        if k in seen:
            continue
        stack = [k]
        seen.add(k)
        sx, sy, sw, zl = 0.0, 0.0, 0.0, []
        while len(stack) > 0:
            cur = stack.pop()
            e = cells[cur]
            sx += e[0]
            sy += e[1]
            sw += e[2]
            zl.extend(e[3])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cur[0] + dx, cur[1] + dy)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        if sw >= minw:
            out.append({"x": sx / sw, "y": sy / sw, "w": sw, "ztop": quantile(zl, 0.9)})
    return out


def kmeans_track(tracks, points, gate, iters=3):
    pred = [(t["x"] + t["vx"], t["y"] + t["vy"]) for t in tracks]
    cen = [list(p) for p in pred]
    sums = [[0.0, 0.0, 0.0] for t in tracks]
    unassigned = []
    for it in range(iters):
        sums = [[0.0, 0.0, 0.0] for t in tracks]
        unassigned = []
        for p in points:
            best = -1
            bd = gate * gate
            for k in range(len(cen)):
                dx = p[0] - cen[k][0]
                dy = p[1] - cen[k][1]
                d = dx * dx + dy * dy
                if d < bd:
                    bd = d
                    best = k
            if best >= 0:
                s = sums[best]
                s[0] += p[0] * p[2]
                s[1] += p[1] * p[2]
                s[2] += p[2]
            else:
                unassigned.append(p)
        for k in range(len(cen)):
            if sums[k][2] > 0:
                cen[k] = [sums[k][0] / sums[k][2], sums[k][1] / sums[k][2]]
            else:
                cen[k] = list(pred[k])
    for k in range(len(tracks)):
        t = tracks[k]
        if sums[k][2] >= 5:
            nvx = cen[k][0] - t["x"]
            nvy = cen[k][1] - t["y"]
            t["vx"] = 0.5 * t["vx"] + 0.5 * nvx
            t["vy"] = 0.5 * t["vy"] + 0.5 * nvy
            t["x"] = cen[k][0]
            t["y"] = cen[k][1]
            t["lost"] = 0
        else:
            t["vx"] *= 0.5
            t["vy"] *= 0.5
            t["lost"] += 1
            t["maxlost"] = max(t["maxlost"], t["lost"])
        t["w"] = sums[k][2]
    return unassigned


def r3(v):
    return round(float(v), 3)


# ---------------------------------------------------------------- live calibration
live = get_observation()
convs = {}
for conv in ("cv", "gl", "ros"):
    st = table_stats(live.cameras["front"], conv)
    convs[conv] = st
best_conv = None
for conv in convs:
    st = convs[conv]
    if st is None:
        continue
    if best_conv is None or st["iqr"] < convs[best_conv]["iqr"]:
        best_conv = conv
report["conv_stats"] = {k: (None if convs[k] is None else {"iqr": r3(convs[k]["iqr"]), "median": r3(convs[k]["median"])}) for k in convs}
if best_conv is None:
    raise ValueError("no valid depth for convention check")
CONV = best_conv
report["conv"] = CONV
wst = table_stats(live.cameras["wrist"], CONV)
report["wrist_table"] = None if wst is None else {"iqr": r3(wst["iqr"]), "median": r3(wst["median"])}
TABLE_Z = convs[CONV]["median"]
report["table_z"] = r3(TABLE_Z)

home_pose = state0.end_effector_poses["primary"]
QDOWN = [float(v) for v in home_pose.quaternion_wxyz]
report["home_ee"] = [r3(v) for v in home_pose.position]
report["home_q"] = [r3(v) for v in QDOWN]

# ---------------------------------------------------------------- button (goal noun)
def find_button():
    found = []
    for camn in ("wrist", "front"):
        try:
            seg = segment_text(camn, "button")
        except Exception as exc:
            found.append({"cam": camn, "err": str(exc)})
            continue
        if not seg.ok:
            continue
        idx = 0
        for cand in seg.segmentations[:5]:
            idx += 1
            try:
                sc = float(cand.score)
            except Exception:
                sc = 0.0
            try:
                pc = mask_to_point_cloud(cand, camn, target_frame=base_frame)
                P = [p for p in pc.points if float(p[2]) > TABLE_Z - 0.01 and float(p[2]) < TABLE_Z + 0.15]
            except Exception:
                P = []
            if len(P) < 10:
                found.append({"cam": camn, "i": idx, "score": round(sc, 3), "n": len(P), "ok": False})
                continue
            xs = [float(p[0]) for p in P]
            ys = [float(p[1]) for p in P]
            zs = [float(p[2]) for p in P]
            top = quantile(zs, 0.9)
            ext = max(quantile(xs, 0.95) - quantile(xs, 0.05), quantile(ys, 0.95) - quantile(ys, 0.05))
            good = (top - TABLE_Z) < 0.07 and (top - TABLE_Z) > 0.008 and ext < 0.09
            found.append({"cam": camn, "i": idx, "score": round(sc, 3), "n": len(P), "x": quantile(xs, 0.5),
                          "y": quantile(ys, 0.5), "top": top, "ext": round(ext, 3), "ok": good})
    report["button_candidates"] = [[f.get("cam"), f.get("i"), f.get("score"), f.get("n"),
                                    r3(f["x"]) if "x" in f else None, r3(f["y"]) if "y" in f else None,
                                    r3(f["top"]) if "top" in f else None, f.get("ext"), f.get("ok")] for f in found]
    return found


BUTTON_CANDS = find_button()

# ---------------------------------------------------------------- video analysis
video = robomme_plus.get_conditioning_video()
NV = len(video)
report["video_len"] = NV
grip = []
ee = []
for fr in video:
    rs = fr.robot_state
    grip.append(float(rs.gripper_positions["primary"]))
    ee.append([float(v) for v in rs.end_effector_poses["primary"].position])
report["grip_samples"] = [round(grip[i], 2) for i in range(0, NV, max(1, NV // 40))]

closed_thr = 0.85 * max(max(grip), 0.5) if NV > 0 else 0.85
c_idx = None
seen_open = False
for i in range(NV):
    if grip[i] >= closed_thr:
        seen_open = True
    elif seen_open:
        c_idx = i
        break
r_idx = None
if c_idx is not None:
    for i in range(c_idx, NV):
        if grip[i] >= closed_thr:
            r_idx = i
            break
report["close_idx"] = c_idx
report["release_idx"] = r_idx
if c_idx is None or r_idx is None:
    raise ValueError("could not find pick in conditioning video")
pick_xy = (ee[c_idx][0], ee[c_idx][1])
rel_xy = (ee[r_idx][0], ee[r_idx][1])
report["pick_xy"] = [r3(pick_xy[0]), r3(pick_xy[1])]
report["release_xy"] = [r3(rel_xy[0]), r3(rel_xy[1])]
report["release_ee_z"] = r3(ee[r_idx][2])

t_init = None
for i in range(r_idx, NV):
    if ee[i][2] > ee[r_idx][2] + 0.06:
        t_init = i
        break
if t_init is None:
    t_init = min(NV - 1, r_idx + 5)
report["t_init"] = t_init


def frame_points(obs, eez):
    pts = cam_points(obs, "front", CONV, TABLE_Z, eez - 0.01)
    pts.extend(cam_points(obs, "wrist", CONV, TABLE_Z, eez - 0.01))
    return pts


init_pts = frame_points(video[t_init], ee[t_init][2])
init_cl = cluster(init_pts)
tracks = []
for cl in init_cl:
    tracks.append({"x": cl["x"], "y": cl["y"], "vx": 0.0, "vy": 0.0, "lost": 0, "maxlost": 0,
                   "w": cl["w"], "born": t_init, "ox": cl["x"], "oy": cl["y"], "maxd": 0.0})
report["init_clusters"] = [[r3(c["x"]), r3(c["y"]), int(c["w"])] for c in init_cl]
if len(tracks) == 0:
    raise ValueError("no object clusters at t_init")
target = None
bd = None
for k in range(len(tracks)):
    d = math.hypot(tracks[k]["x"] - rel_xy[0], tracks[k]["y"] - rel_xy[1])
    if bd is None or d < bd:
        bd = d
        target = k
report["target_init"] = target
report["target_init_dist"] = r3(bd)

hist = []
OKC = [f for f in BUTTON_CANDS if f.get("ok")]
for f in OKC:
    f["occ"] = 0
    f["cx"] = int(math.floor(f["x"] / CELL))
    f["cy"] = int(math.floor(f["y"] / CELL))
n_frames = 0
for i in range(t_init + 1, NV):
    pts = frame_points(video[i], ee[i][2])
    n_frames += 1
    if len(OKC) > 0:
        occ = set()
        for p in pts:
            occ.add((int(math.floor(p[0] / CELL)), int(math.floor(p[1] / CELL))))
        for f in OKC:
            hit = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if (f["cx"] + dx, f["cy"] + dy) in occ:
                        hit = True
            if hit:
                f["occ"] += 1
    un = kmeans_track(tracks, pts, GATE)
    if len(un) >= 15:
        for cl in cluster(un, minw=15):
            far = True
            for t in tracks:
                if math.hypot(t["x"] - cl["x"], t["y"] - cl["y"]) < GATE:
                    far = False
            if far:
                tracks.append({"x": cl["x"], "y": cl["y"], "vx": 0.0, "vy": 0.0, "lost": 0,
                               "maxlost": 0, "w": cl["w"], "born": i, "ox": cl["x"], "oy": cl["y"], "maxd": 0.0})
    for t in tracks:
        t["maxd"] = max(t["maxd"], math.hypot(t["x"] - t["ox"], t["y"] - t["oy"]))
    if (i - t_init) % 16 == 0 or i == NV - 1:
        hist.append([i] + [[round(t["x"], 3), round(t["y"], 3), t["lost"]] for t in tracks])
report["track_hist"] = hist
tgt = tracks[target]
report["target_final"] = [r3(tgt["x"]), r3(tgt["y"]), tgt["lost"], tgt["maxlost"]]
report["n_tracks"] = len(tracks)
report["tracks_final"] = [[r3(t["x"]), r3(t["y"]), t["lost"], t["maxlost"], t["born"], r3(t["maxd"])] for t in tracks]

chosen_xy_hint = (tracks[target]["x"], tracks[target]["y"])
# button role: segmentation of the goal noun decides; geometry (low, compact) filters; a candidate
# must be static (its cell occupied in >= 80% of video frames) and must not be the target object.
BUTTON = None
for f in OKC:
    f["static"] = n_frames > 0 and f["occ"] >= 0.8 * n_frames
    f["on_target"] = math.hypot(f["x"] - chosen_xy_hint[0], f["y"] - chosen_xy_hint[1]) < 0.04
for f in OKC:
    if f["static"] and not f["on_target"]:
        if BUTTON is None or f["score"] > BUTTON["score"]:
            BUTTON = f
if BUTTON is None:
    for f in OKC:
        if not f["on_target"]:
            if BUTTON is None or f["score"] > BUTTON["score"]:
                BUTTON = f
report["button_static"] = [[f.get("cam"), f.get("i"), f["occ"], n_frames, f["on_target"]] for f in OKC]
report["button"] = None if BUTTON is None else [r3(BUTTON["x"]), r3(BUTTON["y"]), r3(BUTTON["top"]), BUTTON["cam"], BUTTON["score"]]

# ---------------------------------------------------------------- live selection
GRIDS = {}
live = get_observation()
eez_live = float(live.robot_state.end_effector_poses["primary"].position[2])
live_pts = frame_points(live, eez_live)
live_cl = cluster(live_pts)
report["live_clusters"] = [[r3(c["x"]), r3(c["y"]), int(c["w"]), r3(c["ztop"])] for c in live_cl]
chosen = None
bd = None
for cl in live_cl:
    d = math.hypot(cl["x"] - tgt["x"], cl["y"] - tgt["y"])
    if bd is None or d < bd:
        bd = d
        chosen = cl
if chosen is None:
    raise ValueError("no live object clusters")
report["chosen"] = [r3(chosen["x"]), r3(chosen["y"]), r3(chosen["ztop"]), r3(bd)]

# ---------------------------------------------------------------- manipulation
log = []


def ended(res):
    try:
        if res.terminated or res.truncated:
            return True
    except Exception:
        pass
    return False


def move(x, y, z, tol=0.01, steps=150, tag=""):
    res = move_to_pose(Pose((x, y, z), QDOWN, base_frame), tolerance=tol, max_steps=steps)
    entry = [tag, bool(res.ok)]
    try:
        entry.append(int(res.steps_executed))
    except Exception:
        pass
    try:
        entry.append(str(res.status))
    except Exception:
        pass
    log.append(entry)
    return res


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def yaw_quat(psi):
    return qmul([math.cos(psi / 2.0), 0.0, 0.0, math.sin(psi / 2.0)], QDOWN)


def move_q(x, y, z, q, tol=0.01, steps=150, tag=""):
    res = move_to_pose(Pose((x, y, z), q, base_frame), tolerance=tol, max_steps=steps)
    entry = [tag, bool(res.ok)]
    try:
        entry.append(int(res.steps_executed))
    except Exception:
        pass
    log.append(entry)
    return res


def observe_object(xy, radius):
    # hover above xy, observe with the wrist camera; return refined geometry of the
    # cluster nearest xy (centre = x/y mid-range, gap direction from angular histogram)
    res = move(xy[0], xy[1], TABLE_Z + 0.17, tol=0.012, tag="hover")
    if ended(res):
        return None, True
    obs = get_observation()
    eez = float(obs.robot_state.end_effector_poses["primary"].position[2])
    pts = cam_points(obs, "wrist", CONV, TABLE_Z, eez - 0.03)
    cls = cluster(pts, minw=15)
    best = None
    bdist = None
    for cl in cls:
        d = math.hypot(cl["x"] - xy[0], cl["y"] - xy[1])
        if d < radius and (bdist is None or d < bdist):
            bdist = d
            best = cl
    if best is None:
        log.append(["observe", len(cls), None])
        return None, False
    near = [p for p in pts if math.hypot(p[0] - best["x"], p[1] - best["y"]) < 0.035]
    if len(near) < 15:
        log.append(["observe", len(cls), "few"])
        return None, False
    xs = [p[0] for p in near]
    ys = [p[1] for p in near]
    zs = [p[3] for p in near]
    cx = 0.5 * (quantile(xs, 0.05) + quantile(xs, 0.95))
    cy = 0.5 * (quantile(ys, 0.05) + quantile(ys, 0.95))
    ztop = quantile(zs, 0.9)
    nb = 12
    hist = [0] * nb
    for p in near:
        a = math.atan2(p[1] - cy, p[0] - cx)
        k = int(math.floor((a + math.pi) / (2 * math.pi) * nb)) % nb
        hist[k] += 1
    hmax = max(hist)
    low = [1 if h <= 0.12 * hmax else 0 for h in hist]
    best_run = 0
    best_mid = None
    for k0 in range(nb):
        if low[k0] == 1 and low[(k0 - 1) % nb] == 0:
            n = 0
            while n < nb and low[(k0 + n) % nb] == 1:
                n += 1
            if n > best_run:
                best_run = n
                best_mid = k0 + 0.5 * n
    gap = None
    if best_run >= 2 and best_run <= 6:
        gap = -math.pi + (best_mid / nb) * 2 * math.pi
    out = {"x": cx, "y": cy, "ztop": ztop, "gap": gap, "n": len(near)}
    log.append(["observe", len(cls), [r3(cx), r3(cy), r3(ztop), None if gap is None else round(math.degrees(gap)), hist]])
    return out, False


def wrap_yaw(psi):
    while psi > math.pi / 2:
        psi -= math.pi
    while psi <= -math.pi / 2:
        psi += math.pi
    return psi


target_xy = (chosen["x"], chosen["y"])
target_top = chosen["ztop"]
done = False
picks = []
for rep in range(2):
    if done:
        break
    held = False
    for attempt in range(3):
        ob, done = observe_object(target_xy, 0.05 if attempt == 0 else 0.08)
        if done:
            break
        psi = 0.0
        if ob is not None:
            target_xy = (ob["x"], ob["y"])
            target_top = ob["ztop"]
            if ob["gap"] is not None:
                psi = ob["gap"]
        psi = wrap_yaw(psi + attempt * math.pi / 4)
        q = yaw_quat(psi)
        height = max(0.02, target_top - TABLE_Z)
        gz = TABLE_Z + max(0.012, 0.45 * height)
        open_gripper()
        res = move_q(target_xy[0], target_xy[1], gz + 0.10, q, tol=0.01, tag="pre")
        if ended(res):
            done = True
            break
        res = move_q(target_xy[0], target_xy[1], gz, q, tol=0.006, tag="grasp")
        if ended(res):
            done = True
            break
        cres = close_gripper()
        if ended(cres):
            done = True
            break
        g_closed = float(get_robot_state().gripper_positions["primary"])
        res = move_q(target_xy[0], target_xy[1], gz + 0.14, q, tol=0.015, tag="lift")
        if ended(res):
            done = True
            break
        g_lift = float(get_robot_state().gripper_positions["primary"])
        picks.append({"rep": rep, "att": attempt, "xy": [r3(target_xy[0]), r3(target_xy[1])], "gz": r3(gz),
                      "yaw": round(math.degrees(psi)), "g_closed": round(g_closed, 3), "g_lift": round(g_lift, 3)})
        if g_lift > 0.04:
            held = True
            break
        open_gripper()
    if done or not held:
        break
    res = move_q(target_xy[0], target_xy[1], gz + 0.008, q, tol=0.01, tag="place")
    if ended(res):
        done = True
        break
    ores = open_gripper()
    if ended(ores):
        done = True
        break
    res = move_q(target_xy[0], target_xy[1], gz + 0.12, q, tol=0.015, tag="retreat")
    if ended(res):
        done = True
        break
report["picks"] = picks

if not done and BUTTON is not None:
    bx, by, btop = BUTTON["x"], BUTTON["y"], BUTTON["top"]
    close_gripper()
    res = move(bx, by, btop + 0.10, tol=0.015, tag="btn_above")
    for depth in (0.015, 0.025):
        if ended(res):
            break
        res = move(bx, by, btop - depth, tol=0.01, steps=100, tag="btn_press")
        if ended(res):
            break
        res = move(bx, by, btop + 0.06, tol=0.02, tag="btn_up")
        if ended(res):
            break
        state = get_robot_state()
        stop = False
        for tick in range(10):
            sr = step(RobotAction(arms={"primary": ArmCommand(
                "joint_position", state.joint_positions["primary"],
                gripper_position=0.0, embodiment=state.embodiment)}))
            if not sr.ok:
                log.append(["hold_reject", str(sr.error)])
                break
            try:
                if sr.terminated or sr.truncated:
                    stop = True
                    break
            except Exception:
                pass
        if stop:
            log.append(["ended_after_hold", depth])
            break
elif BUTTON is None:
    log.append(["no_button"])
report["log"] = log
result = report
