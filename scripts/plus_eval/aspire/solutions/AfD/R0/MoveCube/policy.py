# MoveCube (robomme_plus task 4) -- v04: reachability (choose the rod heading relative to the push
# direction by zero-tick solve_ik checks of the push start/end poses).
# Infer the demonstrated "manner" (which tool) from the episode's own conditioning video, then
# use that tool in the live scene to move the red cube onto the target.
import math

report = {"version": "v05", "events": []}


def note(msg):
    report["events"].append(str(msg))


# ---------------------------------------------------------------- small math helpers
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def qrot(q, v):
    w, x, y, z = q
    vx, vy, vz = v
    # R(q) v
    return (
        (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y - w * z) * vy + 2 * (x * z + w * y) * vz,
        2 * (x * y + w * z) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z - w * x) * vz,
        2 * (x * z - w * y) * vx + 2 * (y * z + w * x) * vy + (1 - 2 * (x * x + y * y)) * vz,
    )


def qconj(q):
    return (q[0], -q[1], -q[2], -q[3])


def qz(theta):
    return (math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0))


def wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def quant(values, frac):
    vals = sorted(float(v) for v in values)
    if len(vals) == 0:
        return None
    return vals[int((len(vals) - 1) * frac)]


def as_list(arr):
    try:
        return arr.tolist()
    except Exception:
        return [list(row) for row in arr]


def vec3(p):
    return (float(p[0]), float(p[1]), float(p[2]))


# ---------------------------------------------------------------- colour classes (offline-derived)
def is_teal(r, g, b):
    return g > 105 and b > 105 and r < 120 and g - r > 35 and abs(g - b) < 30


def is_yellow(r, g, b):
    return r > 170 and g > 140 and b < 120 and r - g < 60


def is_slate(r, g, b):
    return 45 < r < 115 and b > r + 8 and g > r and b < 140 and g - r < 30


def is_pink(r, g, b):
    return r > 195 and 120 < g < 195 and 110 < b < 175 and r - g > 35


def is_red(r, g, b):
    return r > 140 and g < 60 and b < 60


# ---------------------------------------------------------------- camera geometry
def cam_params(cam):
    K = as_list(cam.intrinsics)
    pose = cam.camera_pose
    return (float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2]),
            vec3(pose.position), tuple(float(v) for v in pose.quaternion_wxyz))


def cv_to_local(c, conv):
    if conv == "cv":
        return c
    if conv == "gl":
        return (c[0], -c[1], -c[2])
    return (c[2], -c[0], -c[1])          # ros: x forward, y left, z up


def local_to_cv(p, conv):
    if conv == "cv":
        return p
    if conv == "gl":
        return (p[0], -p[1], -p[2])
    return (-p[1], -p[2], p[0])


def pixel_to_world(u, v, depth, params, conv):
    fx, fy, cx, cy, t, q = params
    c = ((u - cx) / fx * depth, (v - cy) / fy * depth, depth)
    p = qrot(q, cv_to_local(c, conv))
    return (p[0] + t[0], p[1] + t[1], p[2] + t[2])


def world_to_pixel(w, params, conv):
    fx, fy, cx, cy, t, q = params
    p = qrot(qconj(q), (w[0] - t[0], w[1] - t[1], w[2] - t[2]))
    c = local_to_cv(p, conv)
    if c[2] <= 1e-6:
        return None
    return (fx * c[0] / c[2] + cx, fy * c[1] / c[2] + cy)


def red_pixels(rgb_rows, stride):
    out = []
    h = len(rgb_rows)
    w = len(rgb_rows[0])
    for v in range(0, h, stride):
        row = rgb_rows[v]
        for u in range(0, w, stride):
            px = row[u]
            if is_red(px[0], px[1], px[2]):
                out.append((u, v))
    return out


# ---------------------------------------------------------------- execution bookkeeping
state0 = get_robot_state()
BASE = state0.base_frame
EMB = state0.embodiment
try:
    META = robomme_plus.get_task_metadata()
    MAX_STEPS = int(META["max_steps"])
except Exception as exc:
    META = {}
    MAX_STEPS = 1000
    note("metadata max_steps unavailable: " + str(exc))
report["max_steps"] = MAX_STEPS
report["goal"] = get_task_context().language
budget = {"used": 0, "terminal": False}


def account(res, label):
    steps = 0
    try:
        steps = int(res.steps_executed)
    except Exception:
        steps = 0
    budget["used"] += steps
    term = False
    try:
        term = bool(res.terminated)
    except Exception:
        term = False
    try:
        term = term or bool(res.truncated)
    except Exception:
        pass
    if term:
        budget["terminal"] = True
    ok = False
    try:
        ok = bool(res.ok)
    except Exception:
        ok = False
    note(label + " ok=" + str(ok) + " steps=" + str(steps) + " used=" + str(budget["used"]))
    return ok


def can_act(reserve):
    return (not budget["terminal"]) and budget["used"] + reserve < MAX_STEPS


def move(pos, quat, label, max_steps=200, tol=0.01):
    if not can_act(10):
        note("skip " + label + " (budget/terminal)")
        return False
    target = Pose((float(pos[0]), float(pos[1]), float(pos[2])), tuple(quat), BASE)
    res = move_to_pose(target, tolerance=tol, max_steps=max_steps)
    return account(res, label)


def hold(ticks, grip):
    st = get_robot_state()
    for tick in range(ticks):
        if not can_act(1):
            return
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=st.joint_positions["primary"],
            gripper_position=grip, embodiment=st.embodiment)}))
        budget["used"] += 1
        try:
            if bool(reply.terminated) or bool(reply.truncated):
                budget["terminal"] = True
                return
        except Exception:
            pass


def ee_pose():
    p = get_robot_state().end_effector_poses["primary"]
    return vec3(p.position), tuple(float(v) for v in p.quaternion_wxyz)


# ---------------------------------------------------------------- stage 1: manner from video
def classify_wrist(rgb_rows):
    h = len(rgb_rows)
    w = len(rgb_rows[0])
    counts = {"T": 0, "Y": 0, "S": 0, "R": 0}
    n = 0
    for v in range(int(0.6 * h), h, 2):
        row = rgb_rows[v]
        for u in range(int(0.3 * w), int(0.7 * w), 2):
            r, g, b = row[u][0], row[u][1], row[u][2]
            n += 1
            if is_teal(r, g, b):
                counts["T"] += 1
            elif is_yellow(r, g, b):
                counts["Y"] += 1
            elif is_slate(r, g, b):
                counts["S"] += 1
            elif is_red(r, g, b):
                counts["R"] += 1
    pink = 0
    m = 0
    for v in range(0, int(0.5 * h), 2):
        row = rgb_rows[v]
        for u in range(int(0.25 * w), int(0.75 * w), 2):
            m += 1
            if is_pink(row[u][0], row[u][1], row[u][2]):
                pink += 1
    best = None
    for key in ("T", "Y", "S", "R"):
        frac = counts[key] / float(max(1, n))
        if frac > 0.15 and (best is None or frac > best[1]):
            best = (key, frac)
    label = "none" if best is None else best[0]
    if label == "T" and pink / float(max(1, m)) > 0.08:
        label = "P"
    return label


video = ()
try:
    video = robomme_plus.get_conditioning_video()
except Exception as exc:
    note("conditioning video unavailable: " + str(exc))
report["video_frames"] = len(video)

grips = []
for frame in video:
    try:
        grips.append(float(frame.robot_state.gripper_positions["primary"]))
    except Exception:
        grips.append(1.0)

hold_run = None
if len(grips) > 0:
    open_level = max(grips)
    thr = open_level - 0.15
    best_run = None
    start = None
    for i in range(len(grips) + 1):
        closed = i < len(grips) and grips[i] < thr
        if closed and start is None:
            start = i
        if (not closed) and start is not None:
            if best_run is None or (i - start) > (best_run[1] - best_run[0]):
                best_run = (start, i)
            start = None
    hold_run = best_run
report["video_hold_run"] = hold_run

manner = "unknown"
labels = []
if hold_run is not None and hold_run[1] - hold_run[0] >= 3:
    a, b = hold_run
    for k in range(5):
        idx = a + int((b - a) * (0.3 + 0.1 * k))
        idx = min(b - 1, max(a, idx))
        try:
            labels.append(classify_wrist(as_list(video[idx].cameras["wrist"].rgb)))
        except Exception as exc:
            note("wrist classify failed: " + str(exc))
    votes = {}
    for lab in labels:
        votes[lab] = votes.get(lab, 0) + 1
    top = None
    for lab in votes:
        if top is None or votes[lab] > votes[top]:
            top = lab
    manner = {"T": "rod", "P": "spatula", "Y": "ball", "S": "hammer", "R": "pick",
              "none": "unknown"}.get(top, "unknown")
elif len(video) > 0:
    manner = "gripper_push"
report["wrist_labels"] = labels

# Camera axis convention, validated on the LIVE front camera with the API point cloud.
live = get_observation()
front = live.cameras["front"]
front_params = cam_params(front)
front_rows = as_list(front.rgb)
front_depth = as_list(front.depth_m)
live_red = red_pixels(front_rows, 1)
report["live_red_pixels"] = len(live_red)


def seg_candidates(prompt, camera="front", limit=6):
    out = []
    try:
        seg = segment_text(camera, prompt)
    except Exception as exc:
        note("segment_text failed " + prompt + ": " + str(exc))
        return out
    try:
        if not seg.ok:
            return out
    except Exception:
        pass
    cands = []
    try:
        cands = list(seg.segmentations)
    except Exception:
        try:
            cands = list(seg.masks)
        except Exception:
            cands = []
    for cand in cands[:limit]:
        score = 0.0
        try:
            score = float(cand.score)
        except Exception:
            score = 0.0
        try:
            pc = mask_to_point_cloud(cand, camera, target_frame=BASE)
            pts = [vec3(p) for p in as_list(pc.points)]
        except Exception as exc:
            note("mask_to_point_cloud failed " + prompt + ": " + str(exc))
            continue
        if len(pts) >= 15:
            out.append({"score": score, "points": pts, "prompt": prompt})
    return out


def cloud_stats(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    # midrange of robust extremes: the visible surface biases a median toward the camera
    c = (0.5 * (quant(xs, 0.05) + quant(xs, 0.95)), 0.5 * (quant(ys, 0.05) + quant(ys, 0.95)),
         quant(zs, 0.5))
    ext = (quant(xs, 0.95) - quant(xs, 0.05), quant(ys, 0.95) - quant(ys, 0.05),
           quant(zs, 0.95) - quant(zs, 0.05))
    return c, ext, quant(zs, 0.05), quant(zs, 0.95)


def pick_cloud(prompts, min_span, max_span):
    """First prompt with a candidate whose xy span lies in [min_span, max_span]; best score.
    If no candidate passes the span gate, fall back to the best-scored candidate of any prompt."""
    fallback = None
    for prompt in prompts:
        best = None
        for cand in seg_candidates(prompt):
            c, ext, zlo, zhi = cloud_stats(cand["points"])
            span = max(ext[0], ext[1])
            if span <= 2.0 * max_span and (fallback is None or cand["score"] > fallback["score"]):
                fallback = cand
            if span < min_span or span > max_span:
                continue
            if best is None or cand["score"] > best["score"]:
                best = cand
        if best is not None:
            note("picked " + prompt + " score=" + str(round(best["score"], 3)))
            return best["points"]
    if fallback is not None:
        note("span gate failed; fallback " + fallback["prompt"] + " score=" + str(round(fallback["score"], 3)))
        return fallback["points"]
    return None


cube_pts = pick_cloud(("red cube", "red block", "small red cube"), 0.005, 0.10)
conv = "cv"
if cube_pts is not None and len(live_red) > 0:
    red_set = set(live_red)
    best_hits = -1
    for cand_conv in ("cv", "gl", "ros"):
        hits = 0
        for p in cube_pts[::3]:
            px = world_to_pixel(p, front_params, cand_conv)
            if px is None:
                continue
            u = int(round(px[0]))
            v = int(round(px[1]))
            for du in (-1, 0, 1):
                if (u + du, v) in red_set:
                    hits += 1
                    break
        if hits > best_hits:
            best_hits = hits
            conv = cand_conv
    report["conv_hits"] = best_hits
report["camera_convention"] = conv


def video_pull_score(frames, run):
    """>0: cube moved away from gripper (push); <0: toward gripper (pull)."""
    a, b = run
    n = b - a
    stride = max(1, n // 40)
    prev = None
    score = 0.0
    used = 0
    for idx in range(a, b, stride):
        fr = frames[idx]
        try:
            cam = fr.cameras["front"]
            rows = as_list(cam.rgb)
            dep = as_list(cam.depth_m)
            params = cam_params(cam)
        except Exception:
            continue
        pix = red_pixels(rows, 2)
        if len(pix) < 2:
            prev = None
            continue
        us = [p[0] for p in pix]
        vs = [p[1] for p in pix]
        u = quant(us, 0.5)
        v = quant(vs, 0.5)
        d = quant([dep[p[1]][p[0]] for p in pix], 0.5)
        if d is None or d <= 0:
            continue
        cube = pixel_to_world(u, v, float(d), params, conv)
        ee = vec3(fr.robot_state.end_effector_poses["primary"].position)
        if prev is not None:
            dc = (cube[0] - prev[0], cube[1] - prev[1])
            if math.hypot(dc[0], dc[1]) > 0.003:
                rel = (cube[0] - ee[0], cube[1] - ee[1])
                nrm = math.hypot(rel[0], rel[1])
                if nrm > 1e-6:
                    score += (dc[0] * rel[0] + dc[1] * rel[1]) / nrm
                    used += 1
        prev = cube
    return score, used


if manner == "rod":
    try:
        sc, used = video_pull_score(video, hold_run)
        report["pull_score"] = (round(sc, 4), used)
        if used >= 2 and sc < 0:
            manner = "hook"
        elif used >= 2:
            manner = "stick"
    except Exception as exc:
        note("pull analysis failed: " + str(exc))
report["manner"] = manner

# ---------------------------------------------------------------- stage 2: live perception
if cube_pts is None and len(live_red) > 0:
    us = [p[0] for p in live_red]
    vs = [p[1] for p in live_red]
    u = quant(us, 0.5)
    v = quant(vs, 0.5)
    try:
        seg = segment_points("front", [[int(u), int(v)]])
        cands = list(seg.segmentations)
        pc = mask_to_point_cloud(cands[0], "front", target_frame=BASE)
        cube_pts = [vec3(p) for p in as_list(pc.points)]
        note("cube from red-pixel point prompt")
    except Exception as exc:
        note("cube point-prompt fallback failed: " + str(exc))
if cube_pts is None:
    raise ValueError("cube not localized")
cube_c, cube_ext, cube_zlo, cube_zhi = cloud_stats(cube_pts)
cube_half = 0.5 * max(cube_ext[0], cube_ext[1])
cube_half = min(0.04, max(0.01, cube_half))

target_pts = pick_cloud(("purple target", "purple and white target", "bullseye target",
                         "purple circle"), 0.01, 0.15)
if target_pts is None:
    raise ValueError("target not localized")
target_c, target_ext, target_zlo, target_zhi = cloud_stats(target_pts)
table_z = min(target_zlo, cube_zlo)
# a cube's edge equals its height; the oblique visible-surface xy extent underestimates it
cube_half = min(0.04, max(cube_half, 0.5 * (cube_zhi - table_z)))
report["cube_half_m"] = round(cube_half, 4)
report["cube"] = [round(v, 4) for v in cube_c]
report["target"] = [round(v, 4) for v in target_c]

TOOL_PROMPTS = {
    "hook": ("hook", "L-shaped hook", "metal hook tool"),
    "stick": ("stick with a ball on the end", "rod with a ball tip", "stick"),
    "rod": ("hook", "stick with a ball on the end", "stick"),
    "spatula": ("spatula", "pink square spatula", "paddle"),
    "hammer": ("hammer", "small hammer"),
    "ball": ("yellow ball", "ball"),
}


def pixel_rgb(p, rows=None, params=None):
    rows = front_rows if rows is None else rows
    params = front_params if params is None else params
    px = world_to_pixel(p, params, conv)
    if px is None:
        return None
    u = int(round(px[0]))
    v = int(round(px[1]))
    if v < 0 or u < 0 or v >= len(rows) or u >= len(rows[0]):
        return None
    return rows[v][u]


def tool_ends(pts, grasp_wide_end, grasp_colour, rows=None, params=None):
    """Return (grasp_xyz, tip_xyz, axis_angle) for an elongated tool cloud."""
    n = float(len(pts))
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts) / n
    syy = sum((p[1] - my) ** 2 for p in pts) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts) / n
    ang = 0.5 * math.atan2(2 * sxy, sxx - syy)
    ax = (math.cos(ang), math.sin(ang))
    proj = [((p[0] - mx) * ax[0] + (p[1] - my) * ax[1], p) for p in pts]
    s_vals = [s for s, p in proj]
    s_lo = quant(s_vals, 0.03)
    s_hi = quant(s_vals, 0.97)
    lo_end = [p for s, p in proj if s <= s_lo + 0.03]
    hi_end = [p for s, p in proj if s >= s_hi - 0.03]

    def lateral(ps):
        ls = [-(p[0] - mx) * ax[1] + (p[1] - my) * ax[0] for p in ps]
        return (quant(ls, 0.95) - quant(ls, 0.05)) if len(ls) > 2 else 0.0

    def colour_hits(ps):
        hits = 0
        for p in ps[::2]:
            c = pixel_rgb(p, rows, params)
            if c is not None and grasp_colour(c[0], c[1], c[2]):
                hits += 1
        return hits

    c_lo = colour_hits(lo_end)
    c_hi = colour_hits(hi_end)
    if c_lo != c_hi and max(c_lo, c_hi) >= 2:
        grasp_is_lo = c_lo > c_hi
        note("tool ends by colour lo=" + str(c_lo) + " hi=" + str(c_hi))
    else:
        w_lo = lateral(lo_end)
        w_hi = lateral(hi_end)
        grasp_is_lo = (w_lo > w_hi) if grasp_wide_end else (w_lo < w_hi)
        note("tool ends by width lo=" + str(round(w_lo, 3)) + " hi=" + str(round(w_hi, 3)))
    g_end = lo_end if grasp_is_lo else hi_end
    t_end = hi_end if grasp_is_lo else lo_end
    g_s = s_lo if grasp_is_lo else s_hi
    t_s = quant(s_vals, 0.995) if grasp_is_lo else quant(s_vals, 0.005)
    inward = 1.0 if grasp_is_lo else -1.0
    # grasp ~2 cm in from the handle end, on the axis; tip at the far end
    gs = g_s + inward * 0.02
    grasp = (mx + ax[0] * gs, my + ax[1] * gs, quant([p[2] for p in g_end], 0.5))
    # tip = leading surface along the axis (contact face), shifted laterally to the far-end
    # region's mean lateral offset (hook foot centre; ~0 for stick/paddle/shaft)
    lat_mean = sum(-(p[0] - mx) * ax[1] + (p[1] - my) * ax[0] for p in t_end) / len(t_end)
    tip = (mx + ax[0] * t_s - ax[1] * lat_mean, my + ax[1] * t_s + ax[0] * lat_mean,
           quant([p[2] for p in t_end], 0.5))
    return grasp, tip, ang


# ---------------------------------------------------------------- top-down orientation helpers
home_pos, home_q = ee_pose()
approach_axis = qrot(home_q, (0.0, 0.0, 1.0))
if approach_axis[2] > -0.9:
    home_q = (0.0, 1.0, 0.0, 0.0)
    note("home EE not top-down; using (0,1,0,0)")
finger = qrot(home_q, (0.0, 1.0, 0.0))
finger_ang = math.atan2(finger[1], finger[0])


def yaw_for_axis(tool_ang):
    """Rotation about world z that makes the finger axis perpendicular to the tool axis."""
    want = tool_ang + math.pi / 2.0
    d = wrap(want - finger_ang)
    if d > math.pi / 2.0:
        d -= math.pi
    if d < -math.pi / 2.0:
        d += math.pi
    return d


SAFE_DZ = 0.15


def grasp_at(point, yaw, label):
    q = qmul(qz(yaw), home_q)
    gz = max(point[2], table_z + 0.012)
    open_gripper()
    ok = move((point[0], point[1], gz + SAFE_DZ), q, label + " pre-grasp", 250, 0.015)
    ok = move((point[0], point[1], gz + 0.04), q, label + " descend", 150, 0.01) and ok
    ok = move((point[0], point[1], gz), q, label + " grasp", 120, 0.008) and ok
    res = close_gripper()
    account(res, label + " close")
    try:
        gpos = float(get_robot_state().gripper_positions["primary"])
    except Exception:
        gpos = -1.0
    report[label + "_gripper_after_close"] = round(gpos, 3)
    return q, gz, ok


# ---------------------------------------------------------------- stage 3: execute manner
dx = target_c[0] - cube_c[0]
dy = target_c[1] - cube_c[1]
dist = math.hypot(dx, dy)
push_dir = (dx / max(dist, 1e-6), dy / max(dist, 1e-6))
report["cube_to_target_m"] = round(dist, 4)

if manner == "ball":
    ball_pts = pick_cloud(TOOL_PROMPTS["ball"], 0.015, 0.10)
    ramp_pts = pick_cloud(("blue ramp", "blue slope", "blue wedge"), 0.08, 0.6)
    if ball_pts is None or ramp_pts is None:
        raise ValueError("ball or ramp not localized")
    ball_c, ball_ext, bzlo, bzhi = cloud_stats(ball_pts)
    ball_r = 0.5 * max(ball_ext[0], ball_ext[1])
    ball_center = (ball_c[0], ball_c[1], max(table_z + ball_r * 0.8, 0.5 * (bzlo + bzhi)))
    zs = [p[2] for p in ramp_pts]
    z_hi = quant(zs, 0.85)
    z_lo = quant(zs, 0.15)
    high = [p for p in ramp_pts if p[2] >= z_hi]
    low = [p for p in ramp_pts if p[2] <= z_lo]
    hx = sum(p[0] for p in high) / len(high)
    hy = sum(p[1] for p in high) / len(high)
    hz = quant([p[2] for p in high], 0.9)
    lx = sum(p[0] for p in low) / len(low)
    ly = sum(p[1] for p in low) / len(low)
    down = (lx - hx, ly - hy)
    dn = max(1e-6, math.hypot(down[0], down[1]))
    down = (down[0] / dn, down[1] / dn)
    report["ramp_high"] = [round(hx, 3), round(hy, 3), round(hz, 3)]
    report["ramp_down_dot_cube"] = round(
        down[0] * (cube_c[0] - hx) + down[1] * (cube_c[1] - hy), 3)
    q, gz, ok = grasp_at(ball_center, 0.0, "ball")
    move((ball_center[0], ball_center[1], gz + SAFE_DZ), q, "ball lift", 200, 0.015)
    rel_xy = (hx + down[0] * 0.02, hy + down[1] * 0.02)
    move((rel_xy[0], rel_xy[1], hz + ball_r + SAFE_DZ), q, "above ramp top", 300, 0.015)
    move((rel_xy[0], rel_xy[1], hz + ball_r + 0.015), q, "lower onto ramp", 200, 0.01)
    res = open_gripper()
    account(res, "release ball")
    move((rel_xy[0] - down[0] * 0.05, rel_xy[1] - down[1] * 0.05, hz + ball_r + 0.12), q,
         "retreat", 150, 0.02)
    hold(min(120, max(0, MAX_STEPS - budget["used"] - 5)), 1.0)
elif manner in ("hook", "stick", "rod", "spatula", "hammer"):
    tool_pts = pick_cloud(TOOL_PROMPTS[manner], 0.06, 0.6)
    if tool_pts is None:
        raise ValueError("tool not localized for manner " + manner)
    # Re-localise from the wrist camera directly above the front-camera estimate: a close,
    # top-down view that is far less occluded than the oblique front view.
    t_c, t_ext, t_zlo, t_zhi = cloud_stats(tool_pts)
    view_rows = None
    view_params = None
    if move((t_c[0], t_c[1], table_z + 0.38), home_q, "wrist view above tool", 250, 0.02):
        wobs = get_observation()
        try:
            wcam = wobs.cameras["wrist"]
            w_rows = as_list(wcam.rgb)
            w_params = cam_params(wcam)
            best = None
            for prompt in TOOL_PROMPTS[manner]:
                for cand in seg_candidates(prompt, camera="wrist"):
                    cc, ce, czl, czh = cloud_stats(cand["points"])
                    span = max(ce[0], ce[1])
                    off = math.hypot(cc[0] - t_c[0], cc[1] - t_c[1])
                    if span < 0.06 or span > 0.6 or off > 0.12:
                        continue
                    if best is None or span > best[0] + 0.02 or (
                            abs(span - best[0]) <= 0.02 and cand["score"] > best[1]["score"]):
                        best = (span, cand)
                if best is not None:
                    break
            if best is not None:
                note("wrist re-localisation span=" + str(round(best[0], 3)) + " (front " +
                     str(round(max(t_ext[0], t_ext[1]), 3)) + ")")
                if best[0] >= 0.8 * max(t_ext[0], t_ext[1]):
                    tool_pts = best[1]["points"]
                    view_rows = w_rows
                    view_params = w_params
        except Exception as exc:
            note("wrist re-localisation failed: " + str(exc))
    report["tool_view"] = "front" if view_rows is None else "wrist"
    if manner == "hammer":
        grasp_pt, tip_pt, tool_ang = tool_ends(tool_pts, True, is_slate, view_rows, view_params)
    else:
        grasp_pt, tip_pt, tool_ang = tool_ends(tool_pts, False, is_teal, view_rows, view_params)
    report["tool_grasp"] = [round(v, 3) for v in grasp_pt]
    report["tool_tip"] = [round(v, 3) for v in tip_pt]
    g_yaw = yaw_for_axis(tool_ang)
    q, gz, ok = grasp_at(grasp_pt, g_yaw, "tool")
    # tool offset (grasp -> tip) in xy, carried rigidly by the hand
    o = (tip_pt[0] - grasp_pt[0], tip_pt[1] - grasp_pt[1])
    L = math.hypot(o[0], o[1])
    tip_dz = tip_pt[2] - grasp_pt[2]
    move((grasp_pt[0], grasp_pt[1], gz + SAFE_DZ), q, "tool lift", 200, 0.015)
    push_z = gz + 0.01
    gap = 0.03
    travel = dist + gap
    tip_start = (cube_c[0] - push_dir[0] * (cube_half + gap),
                 cube_c[1] - push_dir[1] * (cube_half + gap))

    def ik_ok(pos, quat):
        try:
            return bool(solve_ik(Pose((float(pos[0]), float(pos[1]), float(pos[2])), tuple(quat), BASE)).ok)
        except Exception as exc:
            note("solve_ik unavailable: " + str(exc))
            return True

    # Rod heading candidates relative to the push direction (tip leading first). Zero ticks:
    # every candidate is checked with solve_ik at the above/start/end poses before moving.
    base_heading = math.atan2(push_dir[1], push_dir[0])
    chosen = None
    for off_deg in (0, 30, -30, 60, -60, 90, -90):
        h = base_heading + off_deg * math.pi / 180.0
        th = wrap(h - math.atan2(o[1], o[0]))
        q_try = qmul(qz(th), q)
        e_s = (tip_start[0] - math.cos(h) * L, tip_start[1] - math.sin(h) * L)
        e_e = (e_s[0] + push_dir[0] * travel, e_s[1] + push_dir[1] * travel)
        if ik_ok((e_s[0], e_s[1], push_z), q_try) and ik_ok((e_e[0], e_e[1], push_z), q_try) and \
                ik_ok((e_s[0], e_s[1], push_z + SAFE_DZ), q_try):
            chosen = (off_deg, th, e_s, h)
            break
    if chosen is None:
        note("no heading passed solve_ik; using tip-leading heading")
        th = wrap(base_heading - math.atan2(o[1], o[0]))
        chosen = (0, th, (tip_start[0] - push_dir[0] * L, tip_start[1] - push_dir[1] * L), base_heading)
    report["push_heading_offset_deg"] = chosen[0]
    theta = chosen[1]
    report["tool_rotation_deg"] = round(theta * 180.0 / math.pi, 1)
    pos_now, q_now = ee_pose()
    steps_rot = 3
    for k in range(1, steps_rot + 1):
        qk = qmul(qz(theta * k / float(steps_rot)), q)
        move((pos_now[0], pos_now[1], gz + SAFE_DZ), qk, "rotate " + str(k), 150, 0.02)
    q_push = qmul(qz(theta), q)
    ee_start = chosen[2]
    move((ee_start[0], ee_start[1], push_z + SAFE_DZ), q_push, "above push start", 300, 0.015)
    move((ee_start[0], ee_start[1], push_z), q_push, "lower to push start", 200, 0.01)
    n_seg = max(1, int(travel / 0.02) + 1)
    for k in range(1, n_seg + 1):
        f = travel * k / float(n_seg)
        if not move((ee_start[0] + push_dir[0] * f, ee_start[1] + push_dir[1] * f, push_z),
                    q_push, "push " + str(k), 80, 0.01):
            pass
        if budget["terminal"]:
            break
    pos_now, q_now = ee_pose()
    move((pos_now[0], pos_now[1], push_z + SAFE_DZ), q_push, "lift after push", 150, 0.02)
    hold(10, 0.0)
    # Closed loop: re-observe the cube once; if it stopped short (or slid sideways) of the
    # target while still roughly ahead along the push, push the remainder from behind it.
    try:
        c2_pts = None
        if MAX_STEPS - budget["used"] > 250 and not budget["terminal"]:
            c2_pts = pick_cloud(("red cube", "red block", "small red cube"), 0.005, 0.10)
        if c2_pts is not None:
            c2, e2, z2lo, z2hi = cloud_stats(c2_pts)
            rx = target_c[0] - c2[0]
            ry = target_c[1] - c2[1]
            r2 = math.hypot(rx, ry)
            report["residual_after_push_m"] = round(r2, 4)
            d2 = (rx / max(r2, 1e-6), ry / max(r2, 1e-6))
            if r2 > 0.015 and d2[0] * push_dir[0] + d2[1] * push_dir[1] > 0.3:
                h2 = math.atan2(d2[1], d2[0]) + (chosen[3] - base_heading)
                th2 = wrap(h2 - math.atan2(o[1], o[0]))
                q2 = qmul(qz(th2), q)
                ts2 = (c2[0] - d2[0] * (cube_half + gap), c2[1] - d2[1] * (cube_half + gap))
                es2 = (ts2[0] - math.cos(h2) * L, ts2[1] - math.sin(h2) * L)
                pos_now, q_now = ee_pose()
                move((pos_now[0], pos_now[1], push_z + SAFE_DZ), q2, "correction rotate", 150, 0.02)
                move((es2[0], es2[1], push_z + SAFE_DZ), q2, "above correction start", 300, 0.015)
                move((es2[0], es2[1], push_z), q2, "lower to correction start", 200, 0.01)
                tr2 = r2 + gap
                n2 = max(1, int(tr2 / 0.02) + 1)
                for k in range(1, n2 + 1):
                    f = tr2 * k / float(n2)
                    move((es2[0] + d2[0] * f, es2[1] + d2[1] * f, push_z), q2,
                         "correction push " + str(k), 80, 0.01)
                    if budget["terminal"]:
                        break
                pos_now, q_now = ee_pose()
                move((pos_now[0], pos_now[1], push_z + SAFE_DZ), q2, "lift after correction", 150, 0.02)
                report["correction_pushes"] = 1
    except Exception as exc:
        note("correction skipped: " + str(exc))
elif manner == "pick":
    q, gz, ok = grasp_at((cube_c[0], cube_c[1], 0.5 * (cube_zlo + cube_zhi)), 0.0, "cube")
    move((cube_c[0], cube_c[1], gz + SAFE_DZ), q, "cube lift", 200, 0.015)
    move((target_c[0], target_c[1], gz + SAFE_DZ), q, "above target", 300, 0.015)
    move((target_c[0], target_c[1], gz + 0.005), q, "lower cube", 200, 0.01)
    account(open_gripper(), "release cube")
    move((target_c[0], target_c[1], gz + 0.10), q, "retreat", 120, 0.02)
    hold(10, 1.0)
else:
    # gripper push (no tool held in the video) or unknown: push with closed fingers.
    account(close_gripper(), "close for push")
    q = home_q
    push_z = table_z + 0.015
    start = (cube_c[0] - push_dir[0] * (cube_half + 0.03), cube_c[1] - push_dir[1] * (cube_half + 0.03))
    move((start[0], start[1], push_z + SAFE_DZ), q, "above push start", 300, 0.015)
    move((start[0], start[1], push_z), q, "lower", 200, 0.01)
    travel = dist + 0.03
    n_seg = max(1, int(travel / 0.02) + 1)
    for k in range(1, n_seg + 1):
        f = travel * k / float(n_seg)
        move((start[0] + push_dir[0] * f, start[1] + push_dir[1] * f, push_z), q,
             "push " + str(k), 80, 0.01)
        if budget["terminal"]:
            break
    pos_now, q_now = ee_pose()
    move((pos_now[0], pos_now[1], push_z + 0.08), q, "lift", 120, 0.02)

report["steps_used"] = budget["used"]
report["terminal"] = budget["terminal"]
result = report
