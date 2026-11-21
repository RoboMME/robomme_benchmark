# RouteStick v05 (v04 + bar points selected by height above the table plane only, not by colour:
# a yellow/orange bar is table-coloured and was dropped by v04) -- reproduce the drawn stick route shown in the episode's conditioning video.
#
# Strategy (everything episode-dependent is derived at runtime):
#   1. Read the recorded robot state of every conditioning-video frame (end-effector pose +
#      joints): that IS the demonstrated stick route, including which side of each bar it passes.
#   2. Detect the target discs (small white/grey blobs lying on the table) in the video's first
#      front frame and in the live front frame (pure-Python pixel test + public depth/intrinsics/
#      camera pose deprojection, convention chosen by table-plane flatness).
#   3. Pair video discs with live discs in left-to-right order and map every recorded EE position
#      through the 2-D similarity of the disc pair it lies between (piecewise retargeting).
#   4. If the live scene equals the video scene, replay the recorded joints; otherwise solve IK for
#      the retargeted poses (seeded for continuity) and replay those at 20 Hz with step().
import math

report = {"version": "v05"}
ctx = get_task_context()
goal = str(ctx.language)
report["goal"] = goal
meta = robomme_plus.get_task_metadata()
max_steps = int(meta["max_steps"])
report["max_steps"] = max_steps
try:
    report["conditioning_video_sampling"] = str(meta["conditioning_video_sampling"])
except Exception:
    report["conditioning_video_sampling"] = "n/a"
state0 = get_robot_state()
base_frame = state0.base_frame
embodiment = state0.embodiment
grip_hold = float(state0.gripper_positions["primary"])
DT = 0.05


# ---------------------------------------------------------------- small math helpers
def quat_to_mat(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 0.0:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def median(values):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[len(ordered) // 2]


def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[int((len(ordered) - 1) * fraction)]


def to_rows(array):
    try:
        return array.tolist()
    except Exception:
        return [[array[v][u] for u in range(len(array[0]))] for v in range(len(array))]


# Camera-frame conventions a public camera_pose may use. Each maps an OpenCV ray point
# (x right, y down, z forward) into that convention's camera axes.
CONVENTIONS = ("opencv", "opengl", "ros")


def cv_to_conv(p, conv):
    if conv == "opencv":
        return p
    if conv == "opengl":
        return (p[0], -p[1], -p[2])
    return (p[2], -p[0], -p[1])


def camera_model(cam):
    k = to_rows(cam.intrinsics)
    pose = cam.camera_pose
    rgb = to_rows(cam.rgb)
    return {
        "fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]), "cy": float(k[1][2]),
        "rot": quat_to_mat(pose.quaternion_wxyz),
        "pos": [float(v) for v in pose.position],
        "rgb": rgb,
        "depth": to_rows(cam.depth_m),
        "height": len(rgb),
        "width": len(rgb[0]),
    }


def deproject(model, u, v, conv):
    d = float(model["depth"][v][u])
    if not (d > 0.02 and d < 5.0):
        return None
    pc = ((u - model["cx"]) * d / model["fx"], (v - model["cy"]) * d / model["fy"], d)
    pk = cv_to_conv(pc, conv)
    r = model["rot"]
    t = model["pos"]
    return (
        r[0][0] * pk[0] + r[0][1] * pk[1] + r[0][2] * pk[2] + t[0],
        r[1][0] * pk[0] + r[1][1] * pk[1] + r[1][2] * pk[2] + t[1],
        r[2][0] * pk[0] + r[2][1] * pk[1] + r[2][2] * pk[2] + t[2],
    )


def is_table_px(px):
    r, g, b = int(px[0]), int(px[1]), int(px[2])
    return r > 90 and r - b > 50 and r >= g and g >= b


def is_disc_px(px):
    r, g, b = int(px[0]), int(px[1]), int(px[2])
    hi = max(r, g, b)
    lo = min(r, g, b)
    return lo >= 150 and hi - lo <= 40


def choose_convention(model):
    """Pick the camera-pose convention under which table pixels form a horizontal plane."""
    best = None
    scores = {}
    for conv in CONVENTIONS:
        zs = []
        for v in range(model["height"] // 4, model["height"], 4):
            for u in range(0, model["width"], 4):
                if is_table_px(model["rgb"][v][u]):
                    p = deproject(model, u, v, conv)
                    if p is not None:
                        zs.append(p[2])
        if len(zs) < 50:
            scores[conv] = None
            continue
        spread = quantile(zs, 0.9) - quantile(zs, 0.1)
        below_camera = median(zs) < model["pos"][2]
        scores[conv] = [round(spread, 4), round(median(zs), 4), below_camera]
        if below_camera and (best is None or spread < best[0]):
            best = (spread, conv, median(zs))
    if best is None:
        return None, None, scores
    return best[1], best[2], scores


def find_discs(model, conv, table_z):
    """Small white/grey blobs whose 3-D points lie on the table plane, in robot_base."""
    H, W = model["height"], model["width"]
    mask = [[is_disc_px(model["rgb"][v][u]) for u in range(W)] for v in range(H)]
    seen = [[False] * W for v in range(H)]
    discs = []
    for v0 in range(H):
        for u0 in range(W):
            if not mask[v0][u0] or seen[v0][u0]:
                continue
            stack = [(v0, u0)]
            seen[v0][u0] = True
            pixels = []
            while stack and len(pixels) < 2000:
                a, b = stack.pop()
                pixels.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    c, d = a + da, b + db
                    if c >= 0 and c < H and d >= 0 and d < W and mask[c][d] and not seen[c][d]:
                        seen[c][d] = True
                        stack.append((c, d))
            n = len(pixels)
            if n < 6 or n > 400:
                continue
            vs = [p[0] for p in pixels]
            us = [p[1] for p in pixels]
            bw = max(us) - min(us) + 1
            bh = max(vs) - min(vs) + 1
            if bw > 30 or bh > 24:
                continue
            pts = []
            for a, b in pixels:
                p = deproject(model, b, a, conv)
                if p is not None:
                    pts.append(p)
            if len(pts) < 4:
                continue
            cx = median([p[0] for p in pts])
            cy = median([p[1] for p in pts])
            cz = median([p[2] for p in pts])
            radius = max(math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2) for p in pts)
            if cz - table_z > 0.03 or radius > 0.05:
                continue
            discs.append({"xyz": (cx, cy, cz), "uv": (sum(us) / n, sum(vs) / n), "n": n})
    return discs


ROBOT_BASE_CLEARANCE_M = 0.20   # the robot's own pedestal footprint around the base-frame origin


def filter_discs(discs):
    """Keep disc candidates that are (a) away from the robot's own base and (b) on the common
    disc line: iteratively drop blobs whose perpendicular distance from the principal xy axis
    exceeds half the median nearest-neighbour spacing."""
    kept = [d for d in discs
            if math.sqrt(d["xyz"][0] ** 2 + d["xyz"][1] ** 2) > ROBOT_BASE_CLEARANCE_M]
    for iteration in range(4):
        if len(kept) < 3:
            return kept
        mx = sum(d["xyz"][0] for d in kept) / len(kept)
        my = sum(d["xyz"][1] for d in kept) / len(kept)
        sxx = sum((d["xyz"][0] - mx) ** 2 for d in kept)
        syy = sum((d["xyz"][1] - my) ** 2 for d in kept)
        sxy = sum((d["xyz"][0] - mx) * (d["xyz"][1] - my) for d in kept)
        angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
        nx, ny = -math.sin(angle), math.cos(angle)
        nn = []
        for d in kept:
            nn.append(min(math.sqrt((d["xyz"][0] - e["xyz"][0]) ** 2 + (d["xyz"][1] - e["xyz"][1]) ** 2)
                          for e in kept if e is not d))
        limit = 0.5 * median(nn)
        offs = [abs((d["xyz"][0] - mx) * nx + (d["xyz"][1] - my) * ny) for d in kept]
        worst = max(range(len(kept)), key=lambda i: offs[i])
        if offs[worst] <= limit:
            return kept
        kept = [d for i, d in enumerate(kept) if i != worst]
    return kept


def obstacle_points(model, conv, table_z):
    """3-D points 1.5-8 cm above the table plane, whatever their colour: the slice of each
    standing bar the stick routes around (flat discs and the table itself lie below it)."""
    pts = []
    for v in range(model["height"]):
        for u in range(model["width"]):
            p = deproject(model, u, v, conv)
            if p is None:
                continue
            if p[2] > table_z + 0.015 and p[2] < table_z + 0.08:
                pts.append(p)
    return pts


def bar_centres(pts, discs_xyz):
    """Median xy of obstacle points in the middle band between each consecutive disc pair."""
    out = []
    for k in range(len(discs_xyz) - 1):
        a, b = discs_xyz[k], discs_xyz[k + 1]
        vx, vy = b[0] - a[0], b[1] - a[1]
        length = math.sqrt(vx * vx + vy * vy)
        if length <= 1e-6:
            out.append(None)
            continue
        ux, uy = vx / length, vy / length
        xs = []
        ys = []
        for p in pts:
            dx, dy = p[0] - a[0], p[1] - a[1]
            t = (dx * ux + dy * uy) / length
            n = (-dx * uy + dy * ux) / length
            if t > 0.2 and t < 0.8 and abs(n) < 0.6:
                xs.append(p[0])
                ys.append(p[1])
        out.append((median(xs), median(ys), len(xs)) if len(xs) >= 15 else None)
    return out


def order_discs(discs, ref_axis=None):
    """Sort discs along their principal xy axis. The axis is oriented to agree with ref_axis (the
    video scene's axis) when given, otherwise toward +y of the base frame."""
    if len(discs) < 2:
        return discs, (0.0, 1.0)
    mx = sum(d["xyz"][0] for d in discs) / len(discs)
    my = sum(d["xyz"][1] for d in discs) / len(discs)
    sxx = sum((d["xyz"][0] - mx) ** 2 for d in discs)
    syy = sum((d["xyz"][1] - my) ** 2 for d in discs)
    sxy = sum((d["xyz"][0] - mx) * (d["xyz"][1] - my) for d in discs)
    angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    ax, ay = math.cos(angle), math.sin(angle)
    if ref_axis is not None:
        if ax * ref_axis[0] + ay * ref_axis[1] < 0.0:
            ax, ay = -ax, -ay
    elif ay < 0.0:
        ax, ay = -ax, -ay
    ordered = sorted(discs, key=lambda d: (d["xyz"][0] - mx) * ax + (d["xyz"][1] - my) * ay)
    return ordered, (ax, ay)


def similarity(a0, a1, b0, b1):
    """2-D similarity mapping segment a0->a1 onto b0->b1: returns (scale*cos, scale*sin, a0, b0)."""
    vx, vy = a1[0] - a0[0], a1[1] - a0[1]
    wx, wy = b1[0] - b0[0], b1[1] - b0[1]
    nv = vx * vx + vy * vy
    if nv <= 1e-9:
        return (1.0, 0.0, a0, b0)
    c = (vx * wx + vy * wy) / nv
    s = (vx * wy - vy * wx) / nv
    return (c, s, a0, b0)


def apply_sim(sim, p):
    c, s, a0, b0 = sim
    dx, dy = p[0] - a0[0], p[1] - a0[1]
    return (b0[0] + c * dx - s * dy, b0[1] + s * dx + c * dy)


def segment_index(points, p):
    best = 0
    best_d = None
    for k in range(len(points) - 1):
        a, b = points[k], points[k + 1]
        vx, vy = b[0] - a[0], b[1] - a[1]
        nv = vx * vx + vy * vy
        t = 0.0 if nv <= 1e-12 else ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / nv
        t = max(0.0, min(1.0, t))
        qx, qy = a[0] + t * vx, a[1] + t * vy
        d = (p[0] - qx) ** 2 + (p[1] - qy) ** 2
        if best_d is None or d < best_d:
            best_d = d
            best = k
    return best


def is_terminal(reply):
    flag = False
    for name in ("terminated", "truncated"):
        try:
            if name == "terminated" and bool(reply.terminated):
                flag = True
            if name == "truncated" and bool(reply.truncated):
                flag = True
        except Exception:
            pass
    return flag


def reply_error(reply):
    try:
        err = reply.error
        if err is not None:
            return str(err)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------- 1. the demonstrated route
video = robomme_plus.get_conditioning_video()
report["video_frames"] = len(video)
track = []
for frame in video:
    rs = frame.robot_state
    pose = rs.end_effector_poses["primary"]
    track.append({
        "t": float(frame.timestamp_s),
        "q": [float(v) for v in rs.joint_positions["primary"]],
        "p": [float(v) for v in pose.position],
        "quat": [float(v) for v in pose.quaternion_wxyz],
    })

status = "ok"
if len(track) < 2:
    status = "no_conditioning_video"
else:
    xs = [f["p"][0] for f in track]
    ys = [f["p"][1] for f in track]
    zs = [f["p"][2] for f in track]
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    report["video_ee_span_m"] = round(span, 4)
    report["video_duration_s"] = round(track[-1]["t"] - track[0]["t"], 3)
    if span < 0.02:
        status = "video_robot_state_static"

# ---------------------------------------------------------------- 2. discs in video and live scene
live_obs = get_observation()
live_model = camera_model(live_obs.cameras["front"])
conv, live_table_z, conv_scores = choose_convention(live_model)
report["camera_convention"] = conv
report["convention_scores"] = conv_scores
live_discs = []
video_discs = []
if status == "ok" and conv is None:
    status = "camera_convention_unresolved"
if status == "ok":
    video_model = camera_model(video[0].cameras["front"])
    vconv, video_table_z, vscores = choose_convention(video_model)
    report["video_convention"] = vconv
    if vconv is None:
        vconv = conv
        video_table_z = live_table_z
    video_raw = find_discs(video_model, vconv, video_table_z)
    video_discs, video_axis = order_discs(filter_discs(video_raw))
    report["video_disc_candidates"] = len(video_raw)
    live_raw = find_discs(live_model, conv, live_table_z)
    live_discs, live_axis = order_discs(filter_discs(live_raw), video_axis)
    report["live_disc_candidates"] = len(live_raw)
    report["disc_axes"] = [[round(c, 3) for c in video_axis], [round(c, 3) for c in live_axis]]
    report["live_table_z"] = round(live_table_z, 4)
    report["video_table_z"] = round(video_table_z, 4)
report["live_discs"] = [[round(c, 4) for c in d["xyz"]] for d in live_discs]
report["video_discs"] = [[round(c, 4) for c in d["xyz"]] for d in video_discs]
report["live_disc_px"] = [[round(c, 1) for c in d["uv"]] for d in live_discs]
report["video_disc_px"] = [[round(c, 1) for c in d["uv"]] for d in video_discs]

# Does the demonstrated low part of the route actually pass over the video discs?
if status == "ok" and video_discs:
    zmin = min(f["p"][2] for f in track)
    low = [f["p"] for f in track if f["p"][2] < zmin + 0.03]
    report["video_disc_to_route_xy_m"] = [
        round(min(math.sqrt((p[0] - d["xyz"][0]) ** 2 + (p[1] - d["xyz"][1]) ** 2) for p in low), 4)
        for d in video_discs
    ]

# ---------------------------------------------------------------- 3. pair discs, build retarget map
pairs = []
if status == "ok":
    nv, nl = len(video_discs), len(live_discs)
    if nv >= 2 and nv == nl:
        pairs = [(k, k) for k in range(nv)]
    elif nv >= 2 and nl >= 2:
        best = None
        for off in range(-(nv - 1), nl):
            cand = [(i, i + off) for i in range(nv) if i + off >= 0 and i + off < nl]
            if len(cand) < 2:
                continue
            cost = sum(
                math.sqrt((video_discs[i]["xyz"][0] - live_discs[j]["xyz"][0]) ** 2
                          + (video_discs[i]["xyz"][1] - live_discs[j]["xyz"][1]) ** 2)
                for i, j in cand) / len(cand) - 0.01 * len(cand)
            if best is None or cost < best[0]:
                best = (cost, cand)
        pairs = best[1] if best is not None else []
    if len(pairs) < 2:
        status = "disc_pairing_failed"
report["disc_pairs"] = pairs
report["disc_counts"] = [len(video_discs), len(live_discs)]

replay_mode = "none"
segments = []
bar_corrections = []
if status == "ok":
    vpts = [video_discs[i]["xyz"] for i, j in pairs]
    lpts = [live_discs[j]["xyz"] for i, j in pairs]
    shifts = [math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) for a, b in zip(vpts, lpts)]
    report["disc_shift_m"] = [round(s, 4) for s in shifts]
    for k in range(len(vpts) - 1):
        segments.append(similarity(vpts[k], vpts[k + 1], lpts[k], lpts[k + 1]))
    video_bars = bar_centres(obstacle_points(video_model, vconv, video_table_z), vpts)
    live_bars = bar_centres(obstacle_points(live_model, conv, live_table_z), lpts)
    bar_corrections = []
    bar_report = []
    for k in range(len(vpts) - 1):
        vb, lb = video_bars[k], live_bars[k]
        if vb is None or lb is None:
            bar_corrections.append(None)
            bar_report.append(None)
            continue
        mapped = apply_sim(segments[k], vb)
        dx, dy = lb[0] - mapped[0], lb[1] - mapped[1]
        spacing = math.sqrt((lpts[k + 1][0] - lpts[k][0]) ** 2 + (lpts[k + 1][1] - lpts[k][1]) ** 2)
        mag = math.sqrt(dx * dx + dy * dy)
        if mag > 0.35 * spacing:
            bar_corrections.append(None)          # implausible: treat as a detection error
            bar_report.append(["rejected", round(mag, 4)])
            continue
        bar_corrections.append((dx, dy))
        bar_report.append([round(dx, 4), round(dy, 4), vb[2], lb[2]])
    report["bar_corrections"] = bar_report
    bar_mag = max([0.0] + [math.sqrt(c[0] ** 2 + c[1] ** 2) for c in bar_corrections if c is not None])
    replay_mode = "joints" if max(shifts) < 0.006 and bar_mag < 0.006 else "retarget_ik"
report["replay_mode"] = replay_mode


def segment_param(points, p, k):
    a, b = points[k], points[k + 1]
    vx, vy = b[0] - a[0], b[1] - a[1]
    nv = vx * vx + vy * vy
    if nv <= 1e-12:
        return 0.0
    return ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / nv


def retarget(p):
    k = segment_index(vpts, p)
    xy = apply_sim(segments[k], p)
    corr = bar_corrections[k]
    if corr is not None:
        t = segment_param(vpts, p, k)
        if t > 0.0 and t < 1.0:
            w = math.sin(math.pi * t)
            xy = (xy[0] + w * corr[0], xy[1] + w * corr[1])
    return (xy[0], xy[1], p[2])


# ---------------------------------------------------------------- 4. keyframes (time, joints), lazily
# Retargeted IK is solved chunk by chunk while executing, so the controller never waits for
# hundreds of IK calls before its first control step (RUNTIME.md: <=600 s before the first step,
# <=180 s without advancing control).
selected = []
if status == "ok":
    t0 = track[0]["t"]
    last_t = None
    for idx, f in enumerate(track):
        if replay_mode == "retarget_ik" and last_t is not None and f["t"] - last_t < 0.099 and idx != len(track) - 1:
            continue
        last_t = f["t"]
        selected.append(f)
plan = {"next": 0, "prev_q": None, "fail": 0, "tried": 0, "keys": []}
IK_CHUNK = 15


def extend_keys():
    """Append the next chunk of keyframes; returns False when nothing is left."""
    if plan["next"] >= len(selected):
        return False
    stop = min(len(selected), plan["next"] + IK_CHUNK)
    for f in selected[plan["next"]:stop]:
        if replay_mode == "joints":
            plan["keys"].append((f["t"] - t0, f["q"]))
            continue
        target = retarget(f["p"])
        pose = Pose(target, f["quat"], base_frame)
        prev_q = plan["prev_q"]
        seeds = [prev_q, f["q"]] if prev_q is not None else [f["q"]]
        chosen = None
        for seed in seeds:
            ik = solve_ik(pose, seed_joints=seed)
            if not ik.ok:
                continue
            q = [float(v) for v in ik.joint_positions]
            ref = prev_q if prev_q is not None else f["q"]
            if max(abs(x - y) for x, y in zip(q, ref)) <= 0.35:
                chosen = q
                break
        plan["tried"] += 1
        if chosen is None:
            plan["fail"] += 1
            continue
        plan["prev_q"] = chosen
        plan["keys"].append((f["t"] - t0, chosen))
    plan["next"] = stop
    return True


def ik_unhealthy():
    return plan["tried"] >= 20 and plan["fail"] > 0.2 * plan["tried"]


def interp(keys, t):
    if t <= keys[0][0]:
        return keys[0][1]
    if t >= keys[-1][0]:
        return keys[-1][1]
    lo, hi = 0, len(keys) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if keys[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    a, b = keys[lo], keys[hi]
    w = (t - a[0]) / max(1e-9, b[0] - a[0])
    return [x + w * (y - x) for x, y in zip(a[1], b[1])]


# ---------------------------------------------------------------- 5. execute at 20 Hz
run = {"ticks": 0, "terminal": False, "error": None}


def send(q):
    reply = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=list(q), gripper_position=grip_hold,
        embodiment=embodiment)}))
    run["ticks"] += 1
    if is_terminal(reply):
        run["terminal"] = True
    err = reply_error(reply)
    if err is not None:
        run["error"] = err
    return reply


def stopped():
    return run["terminal"] or run["error"] is not None or run["ticks"] >= max_steps - 1


if status == "ok":
    while not plan["keys"] and extend_keys():
        pass
    if not plan["keys"]:
        status = "retarget_ik_failed"
if status == "ok":
    duration = selected[-1]["t"] - t0
    first_q = plan["keys"][0][1]
    start_q = [float(v) for v in get_robot_state().joint_positions["primary"]]
    gap = max(abs(a - b) for a, b in zip(start_q, first_q))
    approach_ticks = 0 if gap < 0.01 else max(10, int(math.ceil(gap / 0.03)))
    hold_ticks = 20
    budget = max_steps - approach_ticks - hold_ticks - 5
    natural = int(math.ceil(duration / DT))
    stretch = 1.25
    if natural * stretch > budget:
        stretch = max(0.5, float(budget) / max(1, natural))
    total = int(math.ceil(natural * stretch))
    report["stretch"] = round(stretch, 3)
    report["planned_ticks"] = [approach_ticks, total, hold_ticks]
    for k in range(1, approach_ticks + 1):
        if stopped():
            break
        w = float(k) / approach_ticks
        send([a + w * (b - a) for a, b in zip(start_q, first_q)])
    for k in range(1, total + 1):
        if stopped():
            break
        t_video = k * DT / stretch
        while plan["keys"][-1][0] < t_video + 0.3 and extend_keys():
            pass
        if ik_unhealthy():
            status = "retarget_ik_failed"
            break
        send(interp(plan["keys"], t_video))
    for k in range(hold_ticks):
        if stopped() or status != "ok":
            break
        send(plan["keys"][-1][1])
report["ik_failures"] = plan["fail"]
report["ik_attempts"] = plan["tried"]
report["keyframes"] = len(plan["keys"])

final_state = get_robot_state()
report["status"] = status
report["ticks_sent"] = run["ticks"]
report["terminal"] = run["terminal"]
report["step_error"] = run["error"]
report["elapsed_s_from_state"] = round(float(final_state.timestamp_s) - float(state0.timestamp_s), 3)
result = report
