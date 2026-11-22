# v02 — VideoPlaceButton. Change vs v01 (target-semantics mode): the target rings SWAP positions
# in pairs throughout the conditioning video (point-symmetric motion about the pair midpoint).
# The target identity the cube was placed on after the press is tracked through every later
# swap to the end of the video; the live placement goes to that target's final position, not to
# the remembered release position (v00/v01 seeds 0 and 1 terminated as failures at the release
# position).
#
# Goal (from get_task_context().language): "place the <color> cube on the target where it was
# first placed after the button was pressed".
# Video events come from the RECORDED ROBOT STATE (EE pose + gripper reading per frame):
#   - holding segment ending (hold -> open) at low Z is a PLACEMENT; colour of the placed cube
#     checked by projecting the release point into a later video frame.
#   - press: gripper fully closed (empty jaw) and the EE dips.
# Target rings: pure-Python colour blobs per video frame -> table plane -> stable positions ->
# swap events -> identity tracking.

import math


def ring_pixel(r, g, b):
    # target rings: saturated pink/purple. Excludes wood (b << r), grey/white robot and button
    # (r ~ g ~ b), red cube (b ~ 0), green cube, blue cube (r small).
    return r > 120 and b > 0.6 * r and b > g + 10 and r > g + 20


def ring_blobs(rows, v0, min_area):
    # rows: list of pixel rows (lists of [r,g,b]) starting at image row v0.
    # returns list of (u, v, area) with concentric components merged.
    pos = set()
    h = len(rows)
    for vv in range(h):
        row = rows[vv]
        for uu in range(len(row)):
            px = row[uu]
            if ring_pixel(px[0], px[1], px[2]):
                pos.add((uu, vv))
    comps = []
    seen = set()
    for start in pos:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        su = 0.0
        sv = 0.0
        n = 0
        while len(stack) > 0:
            cur = stack.pop()
            su += cur[0]
            sv += cur[1]
            n += 1
            for du in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    nb = (cur[0] + du, cur[1] + dv)
                    if nb in pos and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        comps.append([su / n, sv / n + v0, n])
    # merge concentric rings / fragments of one target (centroids within 5 px)
    comps = [[t[1], t[2], -t[0]] for t in sorted([(-c[2], c[0], c[1]) for c in comps])]
    merged = []
    for c in comps:
        hit = None
        for m in merged:
            if abs(m[0] - c[0]) < 5.0 and abs(m[1] - c[1]) < 5.0:
                hit = m
                break
        if hit is None:
            merged.append([c[0], c[1], c[2]])
        else:
            tot = hit[2] + c[2]
            hit[0] = (hit[0] * hit[2] + c[0] * c[2]) / tot
            hit[1] = (hit[1] * hit[2] + c[1] * c[2]) / tot
            hit[2] = tot
    return [m for m in merged if m[2] >= min_area]


def cam_model(cam):
    q = cam.camera_pose.quaternion_wxyz
    w = float(q[0])
    x = float(q[1])
    y = float(q[2])
    z = float(q[3])
    rot = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    c = [float(v) for v in cam.camera_pose.position]
    k = cam.intrinsics
    return {"rot": rot, "c": c, "fx": float(k[0][0]), "fy": float(k[1][1]),
            "cx": float(k[0][2]), "cy": float(k[1][2])}


def pixel_to_plane(model, u, v, zp):
    dc = [(u + 0.5 - model["cx"]) / model["fx"], (v + 0.5 - model["cy"]) / model["fy"], 1.0]
    rot = model["rot"]
    dw = [rot[i][0] * dc[0] + rot[i][1] * dc[1] + rot[i][2] * dc[2] for i in range(3)]
    if abs(dw[2]) < 1e-9:
        return None
    s = (zp - model["c"][2]) / dw[2]
    return (model["c"][0] + s * dw[0], model["c"][1] + s * dw[1])


def plane_to_pixel(model, xy, zp):
    rot = model["rot"]
    d = [xy[0] - model["c"][0], xy[1] - model["c"][1], zp - model["c"][2]]
    pc = [rot[0][j] * d[0] + rot[1][j] * d[1] + rot[2][j] * d[2] for j in range(3)]
    if pc[2] <= 1e-6:
        return None
    return (model["fx"] * pc[0] / pc[2] + model["cx"], model["fy"] * pc[1] / pc[2] + model["cy"])


def dist2(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def cluster_stable(frame_blobs, tol, min_frames):
    # frame_blobs: list (per frame) of lists of (x, y). Positions seen in >= min_frames frames.
    clusters = []
    for blobs in frame_blobs:
        for p in blobs:
            hit = None
            for cl in clusters:
                if dist2(cl, p) < tol:
                    hit = cl
                    break
            if hit is None:
                clusters.append([p[0], p[1], 1])
            else:
                hit[0] = (hit[0] * hit[2] + p[0]) / (hit[2] + 1)
                hit[1] = (hit[1] * hit[2] + p[1]) / (hit[2] + 1)
                hit[2] += 1
    return [(cl[0], cl[1], cl[2]) for cl in clusters if cl[2] >= min_frames]


def find_swaps(frame_blobs, stable, off_tol, ring_tol, min_support):
    # Returns list of swap events [first_frame, last_frame, (i, j), support, n_off_blobs]: two stable
    # target positions i, j exchanged; the two moving rings stay point-symmetric about the
    # midpoint of i and j (measured on robomme_plus VideoPlaceButton eps 0-1), so support counts
    # frames whose pair of off-position blobs has its midpoint within ring_tol of that midpoint.
    off = []
    for f in range(len(frame_blobs)):
        row = []
        for p in frame_blobs[f]:
            if min(dist2(p, s) for s in stable) > off_tol:
                row.append(p)
        off.append(row)
    events = []
    f = 0
    n = len(off)
    while f < n:
        if len(off[f]) == 0:
            f += 1
            continue
        a = f
        b = f
        gap = 0
        g = f + 1
        while g < n and gap <= 2:
            if len(off[g]) > 0:
                b = g
                gap = 0
            else:
                gap += 1
            g += 1
        events.append([a, b])
        f = b + 1
    out = []
    for ev in events:
        support = {}
        nblob = 0
        for f in range(ev[0], ev[1] + 1):
            row = off[f]
            nblob += len(row)
            for p in range(len(row)):
                for q in range(p + 1, len(row)):
                    m = ((row[p][0] + row[q][0]) / 2.0, (row[p][1] + row[q][1]) / 2.0)
                    for i in range(len(stable)):
                        for j in range(i + 1, len(stable)):
                            mij = ((stable[i][0] + stable[j][0]) / 2.0, (stable[i][1] + stable[j][1]) / 2.0)
                            if dist2(m, mij) < ring_tol:
                                key = (i, j)
                                support[key] = support.get(key, 0) + 1
        scored = [(-t[0], t[1], t[2]) for t in sorted([(-support[k], k[0], k[1]) for k in support])]
        used = set()
        for sup, i, j in scored:
            if sup < min_support:
                break
            if i in used or j in used:
                continue
            used.add(i)
            used.add(j)
            out.append([ev[0], ev[1], (i, j), sup, nblob])
    return out


OPEN = 0.9          # normalized gripper reading above which the jaw counts as open
EMPTY = 0.2         # below this the jaw is closed on nothing
MIN_RUN = 3         # frames a holding / closed segment must last
SNAP_RADIUS = 0.04  # m: live ring detection accepted around the tracked target position
LIFT = 0.10         # m above the cube top for pre-grasp / transit

context = get_task_context()
goal = str(context.language)
base_frame = get_robot_state().base_frame
report = {"goal": goal, "stage": "start"}

COLOR_WORDS = ["red", "green", "blue", "yellow", "purple", "orange", "white", "black", "pink"]
goal_color = None
words = goal.lower().replace(",", " ").split()
for k in range(len(words) - 1):
    if words[k] in COLOR_WORDS and words[k + 1] in ("cube", "block"):
        goal_color = words[k]
        break
report["goal_color"] = goal_color


def color_match(rgb, name):
    r = float(rgb[0])
    g = float(rgb[1])
    b = float(rgb[2])
    if name == "red":
        return r > 90 and g < 0.45 * r and b < 0.45 * r
    if name == "green":
        return g > 80 and r < 0.6 * g and b < 0.6 * g
    if name == "blue":
        return b > 80 and r < 0.6 * b and g < 0.75 * b
    if name == "yellow":
        return r > 120 and g > 120 and b < 0.5 * min(r, g)
    return False


def quat_to_rot(q):
    w = float(q[0])
    x = float(q[1])
    y = float(q[2])
    z = float(q[3])
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def project(point, cam):
    rot = quat_to_rot(cam.camera_pose.quaternion_wxyz)
    c = cam.camera_pose.position
    d = [float(point[i]) - float(c[i]) for i in range(3)]
    pc = [rot[0][j] * d[0] + rot[1][j] * d[1] + rot[2][j] * d[2] for j in range(3)]
    if pc[2] <= 1e-6:
        return None
    k = cam.intrinsics
    u = float(k[0][0]) * pc[0] / pc[2] + float(k[0][2])
    v = float(k[1][1]) * pc[1] / pc[2] + float(k[1][2])
    return (u, v)


def color_fraction(cam, uv, name, radius=3):
    rgb = cam.rgb
    h = len(rgb)
    w = len(rgb[0])
    hits = 0
    total = 0
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            uu = int(round(uv[0])) + du
            vv = int(round(uv[1])) + dv
            if uu < 0 or vv < 0 or uu >= w or vv >= h:
                continue
            total += 1
            if color_match(rgb[vv][uu], name):
                hits += 1
    if total == 0:
        return 0.0
    return hits / total


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


# ------------------------------------------------------------------ video parsing
video = robomme_plus.get_conditioning_video()
n_frames = len(video)
report["video_frames"] = n_frames
tracks = []
for i in range(n_frames):
    rs = video[i].robot_state
    p = rs.end_effector_poses["primary"].position
    tracks.append((float(video[i].timestamp_s), float(p[0]), float(p[1]), float(p[2]),
                   float(rs.gripper_positions["primary"])))

z_floor = min(row[3] for row in tracks) if tracks else 0.0
labels = []
for row in tracks:
    if row[4] >= OPEN:
        labels.append("open")
    elif row[4] < EMPTY:
        labels.append("empty")
    else:
        labels.append("hold")

# contiguous segments
segments = []
start = 0
for i in range(1, n_frames + 1):
    if i == n_frames or labels[i] != labels[start]:
        segments.append((labels[start], start, i - 1))
        start = i

placements = []
presses = []
grasps = []
for s in range(len(segments)):
    lab, a, b = segments[s]
    if lab == "hold" and b - a + 1 >= MIN_RUN:
        g_vals = [tracks[i][4] for i in range(a, b + 1)]
        g_med = quantile(g_vals, 0.5)
        stable = len([v for v in g_vals if abs(v - g_med) < 0.05]) >= 0.6 * len(g_vals)
        grasps.append(a)
        nxt = segments[s + 1] if s + 1 < len(segments) else None
        if stable and nxt is not None and nxt[0] == "open":
            j = nxt[1]
            if tracks[j][3] < z_floor + 0.06:
                placements.append({"frame": j, "t": tracks[j][0], "x": tracks[j][1],
                                   "y": tracks[j][2], "z": tracks[j][3]})
    if lab == "empty" and b - a + 1 >= MIN_RUN:
        zs = [tracks[i][3] for i in range(a, b + 1)]
        zmin = min(zs)
        imin = a + zs.index(zmin)
        if zmin < tracks[a][3] - 0.05 or zmin < z_floor + 0.06:
            presses.append({"frame": imin, "t": tracks[imin][0], "x": tracks[imin][1],
                            "y": tracks[imin][2], "z": zmin})

# colour of the object at each placement
for pl in placements:
    pl["color_frac"] = None
    if goal_color is None:
        continue
    j0 = pl["frame"]
    chosen = None
    for j in range(j0 + 1, n_frames):
        dz = tracks[j][3] - pl["z"]
        dxy = math.sqrt((tracks[j][1] - pl["x"]) ** 2 + (tracks[j][2] - pl["y"]) ** 2)
        if dz > 0.08 or dxy > 0.10:
            chosen = j
            break
        if labels[j] != "open":
            break
    if chosen is None:
        continue
    cam = video[chosen].cameras["front"]
    uv = project((pl["x"], pl["y"], pl["z"]), cam)
    if uv is None:
        continue
    pl["check_frame"] = chosen
    pl["uv"] = [round(uv[0], 1), round(uv[1], 1)]
    pl["color_frac"] = round(color_fraction(cam, uv, goal_color), 3)

report["placements"] = placements
report["presses"] = presses
report["n_grasps"] = len(grasps)

# ------------------------------------------------------------------ live scene
obs = get_observation()
front = obs.cameras["front"]


def candidate_cloud(candidate):
    pc = mask_to_point_cloud(candidate, "front", target_frame=base_frame)
    pts = []
    for p in pc.points:
        pts.append((float(p[0]), float(p[1]), float(p[2])))
    return pts


press_t = None
if len(presses) > 0:
    press_t = presses[0]["t"]
else:
    # fallback: live button xy; first low, non-holding visit of the EE near it
    btn = segment_text("front", "button")
    if btn.ok and len(btn.segmentations) > 0:
        bpts = candidate_cloud(btn.segmentations[0])
        if len(bpts) > 5:
            bx = quantile([p[0] for p in bpts], 0.5)
            by = quantile([p[1] for p in bpts], 0.5)
            for i in range(n_frames):
                row = tracks[i]
                if labels[i] != "hold" and row[3] < z_floor + 0.06 and \
                        math.sqrt((row[1] - bx) ** 2 + (row[2] - by) ** 2) < 0.05:
                    press_t = row[0]
                    break
report["press_t"] = press_t

remembered = None
if press_t is not None:
    after = [pl for pl in placements if pl["t"] > press_t]
    colored = [pl for pl in after if pl["color_frac"] is not None and pl["color_frac"] >= 0.2]
    pick_list = colored if len(colored) > 0 else after
    if len(pick_list) > 0:
        remembered = pick_list[0]
report["remembered_target"] = remembered
if remembered is None:
    raise ValueError("could not determine the placement after the button press from the video")

# ---- target plane height from live "target" segmentation clouds
target_z = None
tz = []
tg = segment_text("front", "target")
if tg.ok:
    for cand in tg.segmentations[:5]:
        pts = candidate_cloud(cand)
        if len(pts) >= 10:
            tz.append(quantile([p[2] for p in pts], 0.5))
if len(tz) > 0:
    target_z = quantile(tz, 0.5)
zp = target_z if target_z is not None else 0.0

# ---- ring tracking through the conditioning video
vcam0 = video[0].cameras["front"]
vmodel = cam_model(vcam0)
img_h = len(vcam0.rgb)
img_w = len(vcam0.rgb[0])
min_area = max(8, int(40.0 * img_h * img_w / 65536.0))
frame_blobs = []
for i in range(n_frames):
    rows = video[i].cameras["front"].rgb.tolist()
    plane_pts = []
    for bl in ring_blobs(rows, 0, min_area):
        xy = pixel_to_plane(vmodel, bl[0], bl[1], zp)
        if xy is not None:
            plane_pts.append(xy)
    frame_blobs.append(plane_pts)
stable_raw = cluster_stable(frame_blobs, 0.02, max(10, n_frames // 12))
stable = [(sr[0], sr[1]) for sr in stable_raw]
swaps = find_swaps(frame_blobs, stable, 0.02, 0.025, 3) if len(stable) >= 2 else []
report["stable_targets"] = [[round(sr[0], 4), round(sr[1], 4), sr[2]] for sr in stable_raw]
report["swaps"] = [[e[0], e[1], [e[2][0], e[2][1]], e[3], e[4]] for e in swaps]

target_xy = (remembered["x"], remembered["y"])
track = {"tracked": False}
if len(stable) > 0:
    cur = None
    best_d = None
    for i in range(len(stable)):
        d = dist2(stable[i], target_xy)
        if best_d is None or d < best_d:
            best_d = d
            cur = i
    if best_d is not None and best_d < 0.04:
        hist = [cur]
        for e in swaps:
            if e[0] > remembered["frame"] and cur in e[2]:
                cur = e[2][1] if e[2][0] == cur else e[2][0]
                hist.append(cur)
        target_xy = stable[cur]
        track = {"tracked": True, "start_dist": round(best_d, 4), "history": hist,
                 "final_xy": [round(target_xy[0], 4), round(target_xy[1], 4)]}
report["track"] = track

# snap to a live ring detection
snap = {"snapped": False}
live_rows = front.rgb.tolist()
lmodel = cam_model(front)
bestl = None
for bl in ring_blobs(live_rows, 0, min_area):
    xy = pixel_to_plane(lmodel, bl[0], bl[1], zp)
    if xy is None:
        continue
    d = dist2(xy, target_xy)
    if d < SNAP_RADIUS and (bestl is None or d < bestl[0]):
        bestl = (d, xy)
if bestl is not None:
    snap = {"snapped": True, "dist": round(bestl[0], 4), "xy": [round(bestl[1][0], 4), round(bestl[1][1], 4)]}
    target_xy = bestl[1]
report["target_snap"] = snap
report["target_xy"] = [round(target_xy[0], 4), round(target_xy[1], 4)]

# locate the goal-coloured cube
cube_prompt = (goal_color + " cube") if goal_color is not None else "cube"
cube_pts = None
cube_info = None
for prompt in (cube_prompt, cube_prompt.replace("cube", "block")):
    seg = segment_text("front", prompt)
    if not seg.ok:
        continue
    for cand in seg.segmentations[:5]:
        box = cand.box_xyxy
        if box is None:
            continue
        uc = (float(box[0]) + float(box[2])) / 2.0
        vc = (float(box[1]) + float(box[3])) / 2.0
        frac = color_fraction(front, (uc, vc), goal_color, 2) if goal_color is not None else 1.0
        if frac < 0.3:
            continue
        pts = candidate_cloud(cand)
        if len(pts) < 20:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        span = max(quantile(xs, 0.95) - quantile(xs, 0.05), quantile(ys, 0.95) - quantile(ys, 0.05))
        if span < 0.015 or span > 0.09:
            continue
        cube_pts = pts
        cube_info = {"prompt": prompt, "score": float(cand.score), "box": [float(v) for v in box],
                     "color_frac": round(frac, 3), "n": len(pts), "span": round(span, 4)}
        break
    if cube_pts is not None:
        break
report["cube"] = cube_info
if cube_pts is None:
    raise ValueError("goal cube not found")

top_z = quantile([p[2] for p in cube_pts], 0.95)
bottom_z = quantile([p[2] for p in cube_pts], 0.03)
top_pts = [p for p in cube_pts if p[2] > top_z - 0.008]
cube_x = quantile([p[0] for p in top_pts], 0.5)
cube_y = quantile([p[1] for p in top_pts], 0.5)
cube_h = top_z - bottom_z

# yaw from minimum-area rectangle of the top face
best_yaw = 0.0
best_area = None
for deg in range(0, 90, 3):
    a = math.radians(deg)
    ca = math.cos(a)
    sa = math.sin(a)
    us = [p[0] * ca + p[1] * sa for p in top_pts]
    vs = [-p[0] * sa + p[1] * ca for p in top_pts]
    area = (quantile(us, 0.95) - quantile(us, 0.05)) * (quantile(vs, 0.95) - quantile(vs, 0.05))
    if best_area is None or area < best_area:
        best_area = area
        best_yaw = a
if best_yaw > math.pi / 4:
    best_yaw -= math.pi / 2
report["cube_geom"] = {"x": round(cube_x, 4), "y": round(cube_y, 4), "top_z": round(top_z, 4),
                       "bottom_z": round(bottom_z, 4), "yaw_deg": round(math.degrees(best_yaw), 1),
                       "n_top": len(top_pts)}

if target_z is None:
    target_z = bottom_z


def down_quat(yaw):
    # home orientation [0,-1,0,0] rotated about world z by yaw
    return (0.0, -math.cos(yaw / 2.0), -math.sin(yaw / 2.0), 0.0)


grasp_q = down_quat(best_yaw)
grasp_z = bottom_z + 0.5 * cube_h
transit_z = top_z + LIFT
# release at the grasp height (+3 mm) so the cube is set down on the table, not dropped
place_z = max(grasp_z + 0.003, target_z + 0.5 * cube_h - 0.005)

moves = []


def go(label, xyz, q, tol=0.01, steps=120):
    res = move_to_pose(Pose(xyz, q, base_frame), tolerance=tol, max_steps=steps)
    entry = {"label": label, "ok": bool(res.ok), "status": str(res.status), "steps": int(res.steps_executed)}
    try:
        entry["terminated"] = bool(res.terminated)
        entry["truncated"] = bool(res.truncated)
    except Exception:
        pass
    moves.append(entry)
    report["moves"] = moves
    return res


def ended(res):
    try:
        return bool(res.terminated) or bool(res.truncated)
    except Exception:
        return False


report["stage"] = "motion"
r = open_gripper()
r = go("pre_grasp", (cube_x, cube_y, transit_z), grasp_q, 0.015)
if not ended(r):
    r = go("grasp", (cube_x, cube_y, grasp_z), grasp_q, 0.01)
if not ended(r):
    r = close_gripper()
    moves.append({"label": "close", "ok": bool(r.ok), "status": str(r.status), "steps": int(r.steps_executed)})
if not ended(r):
    try:
        report["grip_after_close"] = float(get_robot_state().gripper_positions["primary"])
    except Exception:
        pass
    r = go("lift", (cube_x, cube_y, transit_z), grasp_q, 0.015)
if not ended(r):
    r = go("above_target", (target_xy[0], target_xy[1], transit_z), grasp_q, 0.015)
if not ended(r):
    r = go("lower", (target_xy[0], target_xy[1], place_z), grasp_q, 0.01)
if not ended(r):
    r = open_gripper()
    moves.append({"label": "release", "ok": bool(r.ok), "status": str(r.status), "steps": int(r.steps_executed)})
if not ended(r):
    r = go("retreat", (target_xy[0], target_xy[1], transit_z), grasp_q, 0.02)
report["stage"] = "done"
report["final_ended"] = ended(r)

result = report
