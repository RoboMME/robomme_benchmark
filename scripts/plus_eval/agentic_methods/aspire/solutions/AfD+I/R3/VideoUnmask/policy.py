# VideoUnmask v00 -- initial program from offline demonstration analysis.
#
# Strategy (all task-dependent values derived at runtime):
#  1. Parse the goal into an ordered list of (color, quantifier) clauses.
#  2. Scan the episode's conditioning video (front + wrist RGB-D). Saturated colored
#     pixels are projected to robot_base with each frame's own calibration; per-frame
#     3D blobs are tracked over time. The last-seen 3D position of each colored cube
#     is where a container later covered it.
#  3. From the clean initial live observation (arm at its start pose), project live
#     depth and, for each target cube XY, extract the covering container's top patch
#     (region growing at the top height) -> rectangle center, yaw and size.
#  4. For each target in goal order: top-down approach, grasp across a face pair,
#     lift, lower back onto the same place, release, rise.
import math

report = {"version": "v00", "events": [], "errors": []}

COLOR_WORDS = ("red", "green", "blue", "yellow", "orange", "purple")


def log(msg):
    report["events"].append(msg)


def rnd(values, digits=4):
    return [round(float(value), digits) for value in values]


def quat_to_mat(q):
    w, x, y, z = (float(value) for value in q)
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


def column(mat, index, sign=1.0):
    return [sign * mat[0][index], sign * mat[1][index], sign * mat[2][index]]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


# Camera conventions: (forward axis, image-right axis, image-down axis) in camera-local terms
# expressed as (column index, sign).
CONVENTIONS = {
    "opencv": ((2, 1.0), (0, 1.0), (1, 1.0)),
    "opengl": ((2, -1.0), (0, 1.0), (1, -1.0)),
    "sapien": ((0, 1.0), (1, -1.0), (2, -1.0)),
}


def camera_axes(camera, convention):
    pose = camera.camera_pose
    rot = quat_to_mat(pose.quaternion_wxyz)
    spec = CONVENTIONS[convention]
    fwd = column(rot, spec[0][0], spec[0][1])
    right = column(rot, spec[1][0], spec[1][1])
    down = column(rot, spec[2][0], spec[2][1])
    pos = [float(value) for value in pose.position]
    return fwd, right, down, pos


def intrinsics_of(camera):
    k = camera.intrinsics
    return float(k[0][0]), float(k[1][1]), float(k[0][2]), float(k[1][2])


def to_list(array):
    try:
        return array.tolist()
    except Exception:
        return [list(row) for row in array]


def make_projector(camera, convention):
    fx, fy, cx, cy = intrinsics_of(camera)
    fwd, right, down, pos = camera_axes(camera, convention)

    def project(u, v, d):
        a = (u - cx) / fx * d
        b = (v - cy) / fy * d
        return (
            pos[0] + right[0] * a + down[0] * b + fwd[0] * d,
            pos[1] + right[1] * a + down[1] * b + fwd[1] * d,
            pos[2] + right[2] * a + down[2] * b + fwd[2] * d,
        )

    def to_pixel(point):
        rel = [point[0] - pos[0], point[1] - pos[1], point[2] - pos[2]]
        d = dot(rel, fwd)
        if d <= 1e-6:
            return None
        return (cx + fx * dot(rel, right) / d, cy + fy * dot(rel, down) / d, d)

    return project, to_pixel


def classify(r, g, b):
    r = int(r)
    g = int(g)
    b = int(b)
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 64:
        return None
    s = (mx - mn) / float(mx)
    if s < 0.55:
        return None
    span = float(mx - mn)
    if mx == r:
        h = (60.0 * ((g - b) / span)) % 360.0
    elif mx == g:
        h = 60.0 * ((b - r) / span) + 120.0
    else:
        h = 60.0 * ((r - g) / span) + 240.0
    if (h < 8.0 or h > 340.0) and s > 0.7:
        return "red"
    if 90.0 <= h <= 160.0:
        return "green"
    if 200.0 <= h <= 260.0:
        return "blue"
    if 48.0 <= h <= 70.0 and s > 0.6:
        return "yellow"
    if 33.0 <= h < 48.0 and s > 0.85:
        return "orange"
    if 270.0 <= h <= 330.0:
        return "purple"
    return None


def frame_blobs(camera, convention, stride, wanted):
    """Colored-pixel 3D blobs in one camera image: list of (color, xyz, count)."""
    project, to_pixel = make_projector(camera, convention)
    rgb = to_list(camera.rgb)
    depth = to_list(camera.depth_m)
    height = len(rgb)
    width = len(rgb[0])
    pts = {}
    for v in range(0, height, stride):
        row = rgb[v]
        drow = depth[v]
        for u in range(0, width, stride):
            px = row[u]
            c = classify(px[0], px[1], px[2])
            if c is None or c not in wanted:
                continue
            d = float(drow[u])
            if not (d > 0.05 and d < 5.0):
                continue
            pts.setdefault(c, []).append(project(u, v, d))
    blobs = []
    for c in pts:
        clusters = []
        for p in pts[c]:
            placed = False
            for cl in clusters:
                m = cl["sum"]
                n = cl["n"]
                if (p[0] - m[0] / n) ** 2 + (p[1] - m[1] / n) ** 2 + (p[2] - m[2] / n) ** 2 < 0.03 ** 2:
                    m[0] += p[0]
                    m[1] += p[1]
                    m[2] += p[2]
                    cl["n"] += 1
                    cl["zmin"] = min(cl["zmin"], p[2])
                    placed = True
                    break
            if not placed:
                clusters.append({"sum": [p[0], p[1], p[2]], "n": 1, "zmin": p[2]})
        for cl in clusters:
            if cl["n"] >= 3:
                n = float(cl["n"])
                blobs.append((c, (cl["sum"][0] / n, cl["sum"][1] / n, cl["sum"][2] / n), cl["n"], cl["zmin"]))
    return blobs


def parse_goal(text):
    """Ordered (color, quantifier) clauses from the instruction."""
    lowered = text.lower().replace(",", " , ")
    for punct in (".", ";"):
        lowered = lowered.replace(punct, " , ")
    words = lowered.split()
    clauses = []
    last_break = 0
    for index, word in enumerate(words):
        if word in (",", "then", "followed", "finally", "and"):
            last_break = index
        base = word
        if base in COLOR_WORDS:
            window = words[last_break:index + 2]
            quant = "one"
            if "all" in window or (index + 1 < len(words) and words[index + 1] == "cubes"):
                quant = "all"
            clauses.append((base, quant))
    return clauses


# ---------------------------------------------------------------- context
context = get_task_context()
goal = str(context.language)
report["goal"] = goal
try:
    meta = robomme_plus.get_task_metadata()
    report["task_metadata"] = {str(k): str(meta[k])[:200] for k in meta}
except Exception as exc:
    report["errors"].append("metadata: " + str(exc))
try:
    cmeta = robomme_plus.get_controller_metadata()
    report["controller_metadata"] = {str(k): str(cmeta[k])[:300] for k in cmeta}
except Exception as exc:
    report["errors"].append("controller metadata: " + str(exc))

clauses = parse_goal(goal)
report["clauses"] = [[c, q] for c, q in clauses]
wanted = set(c for c, q in clauses)

state0 = get_robot_state()
base_frame = state0.base_frame
ee0 = state0.end_effector_poses["primary"]
report["ee0"] = {"position": rnd(ee0.position), "quat": rnd(ee0.quaternion_wxyz)}
report["t0"] = float(state0.timestamp_s)
live = get_observation()

# ---------------------------------------------------------------- camera convention
# The wrist camera looks along the hand's approach axis (hand local +z).
hand_rot = quat_to_mat(ee0.quaternion_wxyz)
approach = column(hand_rot, 2)
conv_scores = {}
for name in CONVENTIONS:
    fwd, right, down, pos = camera_axes(live.cameras["wrist"], name)
    conv_scores[name] = round(dot(fwd, approach), 3)
convention = None
for name in CONVENTIONS:
    if convention is None or conv_scores[name] > conv_scores[convention]:
        convention = name
report["convention_scores_wrist_vs_approach"] = conv_scores
front_fwd = camera_axes(live.cameras["front"], convention)[0]
report["front_forward"] = rnd(front_fwd)
report["convention"] = convention

# API cross-check: project a SAM mask ourselves and compare with mask_to_point_cloud.
api_check = {}
try:
    seg = segment_text("wrist", "white box")
    if seg.ok and len(seg.segmentations) > 0:
        cand = seg.segmentations[0]
        mask = None
        try:
            mask = cand.mask
        except Exception:
            mask = None
        cloud = mask_to_point_cloud(cand, "wrist", target_frame=base_frame)
        api_pts = cloud.points
        n_api = len(api_pts)
        if n_api > 0 and mask is not None:
            api_mean = [sum(float(p[i]) for p in api_pts) / n_api for i in range(3)]
            mlist = to_list(mask)
            dlist = to_list(live.cameras["wrist"].depth_m)
            pix = []
            for v in range(len(mlist)):
                for u in range(len(mlist[v])):
                    if mlist[v][u]:
                        pix.append((u, v, float(dlist[v][u])))
            for name in CONVENTIONS:
                project, to_pixel = make_projector(live.cameras["wrist"], name)
                acc = [0.0, 0.0, 0.0]
                cnt = 0
                for u, v, d in pix:
                    if d > 0.05 and d < 5.0:
                        p = project(u, v, d)
                        acc[0] += p[0]
                        acc[1] += p[1]
                        acc[2] += p[2]
                        cnt += 1
                if cnt > 0:
                    mine = [acc[0] / cnt, acc[1] / cnt, acc[2] / cnt]
                    api_check[name] = round(math.sqrt(sum((mine[i] - api_mean[i]) ** 2 for i in range(3))), 4)
            api_check["api_mean"] = rnd(api_mean)
            api_check["n_api"] = n_api
            api_check["n_mask"] = len(pix)
except Exception as exc:
    api_check["error"] = str(exc)[:300]
report["api_projection_check"] = api_check
errs = [(api_check[name], name) for name in CONVENTIONS if name in api_check]
if errs:
    best_err, best_name = min(errs)
    if best_name != convention and best_err < 0.02:
        report["convention_override"] = [convention, best_name]
        convention = best_name

# ---------------------------------------------------------------- conditioning video
video = robomme_plus.get_conditioning_video()
report["video_frames"] = len(video)
tracks = []
frames_used = 0
last_t = -1.0
for frame in video:
    t = float(frame.timestamp_s)
    if t - last_t < 0.12:
        continue
    last_t = t
    frames_used += 1
    for cam_name in ("front", "wrist"):
        try:
            blobs = frame_blobs(frame.cameras[cam_name], convention, 2, wanted)
        except Exception as exc:
            report["errors"].append("blob " + cam_name + ": " + str(exc)[:200])
            blobs = []
        for color, xyz, count, zmin in blobs:
            best = None
            best_d = 0.045
            for tr in tracks:
                if tr["color"] != color:
                    continue
                dxy = math.sqrt((tr["last"][0] - xyz[0]) ** 2 + (tr["last"][1] - xyz[1]) ** 2)
                if dxy < best_d:
                    best_d = dxy
                    best = tr
            if best is None:
                best = {"color": color, "obs": [], "last": xyz, "first_t": t}
                tracks.append(best)
            best["obs"].append((t, cam_name, xyz, count, zmin))
            best["last"] = xyz
            best["last_t"] = t
report["video_frames_used"] = frames_used


def track_position(tr):
    """Mean of the observations in the last 0.4 s of visibility (settled position)."""
    tl = tr["last_t"]
    sel = [o for o in tr["obs"] if o[0] >= tl - 0.4]
    n = float(len(sel))
    return (
        sum(o[2][0] for o in sel) / n,
        sum(o[2][1] for o in sel) / n,
        sum(o[2][2] for o in sel) / n,
        min(o[4] for o in sel),
    )


cubes = []
for tr in tracks:
    nframes = len(set(o[0] for o in tr["obs"]))
    pos = track_position(tr)
    cubes.append({
        "color": tr["color"], "xyz": pos[:3], "zmin": pos[3], "frames": nframes,
        "first_t": tr["first_t"], "last_t": tr["last_t"],
        "cams": sorted(set(o[1] for o in tr["obs"])),
    })
# Merge tracks of the same color whose settled positions coincide (rolling / two cameras).
merged = []
def by_frames(items):
    out = list(items)
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if out[j]["frames"] > out[i]["frames"]:
                out[i], out[j] = out[j], out[i]
    return out


for cube in by_frames(cubes):
    dup = None
    for other in merged:
        if other["color"] == cube["color"] and math.sqrt(
                (other["xyz"][0] - cube["xyz"][0]) ** 2 + (other["xyz"][1] - cube["xyz"][1]) ** 2) < 0.04:
            dup = other
            break
    if dup is None:
        merged.append(cube)
    else:
        dup["frames"] += cube["frames"]
cubes = [c for c in merged if c["frames"] >= 2]
report["cubes"] = [{"color": c["color"], "xyz": rnd(c["xyz"]), "zmin": round(c["zmin"], 4),
                    "frames": c["frames"], "t": [round(c["first_t"], 2), round(c["last_t"], 2)],
                    "cams": c["cams"]} for c in cubes]

# ---------------------------------------------------------------- live geometry
live_points = []
ee0_pos = [float(value) for value in ee0.position]
for cam_name in ("wrist", "front"):
    cam = live.cameras[cam_name]
    project, to_pixel = make_projector(cam, convention)
    depth = to_list(cam.depth_m)
    for v in range(0, len(depth), 2):
        drow = depth[v]
        for u in range(0, len(drow), 2):
            d = float(drow[u])
            if d > 0.05 and d < 5.0:
                p = project(u, v, d)
                # Exclude the robot's own hand/fingers and anything at or above the start EE height.
                if (p[0] - ee0_pos[0]) ** 2 + (p[1] - ee0_pos[1]) ** 2 + (p[2] - ee0_pos[2]) ** 2 < 0.15 ** 2:
                    continue
                if p[2] > ee0_pos[2] - 0.10:
                    continue
                live_points.append(p)
report["live_points"] = len(live_points)


def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def container_at(xy):
    near = [p for p in live_points if (p[0] - xy[0]) ** 2 + (p[1] - xy[1]) ** 2 < 0.15 ** 2]
    if len(near) < 30:
        return None, "few points near cube: " + str(len(near))
    # Table = lowest well-populated 5 mm height bin.
    bins = {}
    for p in near:
        key = int(math.floor(p[2] / 0.005))
        bins[key] = bins.get(key, 0) + 1
    thresh = max(5, int(0.05 * len(near)))
    table_key = None
    for key in sorted(bins):
        if bins[key] >= thresh:
            table_key = key
            break
    table_z = (table_key + 0.5) * 0.005
    core = [p[2] for p in near if (p[0] - xy[0]) ** 2 + (p[1] - xy[1]) ** 2 < 0.03 ** 2]
    if len(core) < 5:
        return None, "no core points"
    top_z = quantile(core, 0.9)
    if top_z - table_z < 0.015:
        return None, "no raised container over cube (top-table=" + str(round(top_z - table_z, 4)) + ")"
    cand = [p for p in near if abs(p[2] - top_z) < 0.01 and
            (p[0] - xy[0]) ** 2 + (p[1] - xy[1]) ** 2 < 0.07 ** 2]
    if not cand:
        return None, "no top candidates"
    seed_index = 0
    seed_d = None
    for index, p in enumerate(cand):
        dd = (p[0] - xy[0]) ** 2 + (p[1] - xy[1]) ** 2
        if seed_d is None or dd < seed_d:
            seed_d = dd
            seed_index = index
    cell = 0.008
    grid = {}
    for index, p in enumerate(cand):
        key = (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)))
        grid.setdefault(key, []).append(index)
    seen = set()
    stack = [seed_index]
    seen.add(seed_index)
    while stack:
        index = stack.pop()
        p = cand[index]
        kx = int(math.floor(p[0] / cell))
        ky = int(math.floor(p[1] / cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((kx + dx, ky + dy), []):
                    if j in seen:
                        continue
                    q = cand[j]
                    if (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 < cell ** 2 and abs(q[2] - p[2]) < 0.006:
                        seen.add(j)
                        stack.append(j)
    patch = [cand[j] for j in seen]
    if len(patch) < 10:
        return None, "top patch too small: " + str(len(patch))
    best = None
    for step_index in range(30):
        ang = math.radians(3.0 * step_index)
        ca = math.cos(ang)
        sa = math.sin(ang)
        a_vals = [p[0] * ca + p[1] * sa for p in patch]
        b_vals = [-p[0] * sa + p[1] * ca for p in patch]
        a_lo = quantile(a_vals, 0.02)
        a_hi = quantile(a_vals, 0.98)
        b_lo = quantile(b_vals, 0.02)
        b_hi = quantile(b_vals, 0.98)
        area = (a_hi - a_lo) * (b_hi - b_lo)
        if best is None or area < best[0]:
            best = (area, ang, a_lo, a_hi, b_lo, b_hi)
    area, ang, a_lo, a_hi, b_lo, b_hi = best
    am = 0.5 * (a_lo + a_hi)
    bm = 0.5 * (b_lo + b_hi)
    cxw = am * math.cos(ang) - bm * math.sin(ang)
    cyw = am * math.sin(ang) + bm * math.cos(ang)
    info = {
        "center": [cxw, cyw], "yaw": ang, "size": [a_hi - a_lo, b_hi - b_lo],
        "top_z": top_z, "table_z": table_z, "patch": len(patch),
        "offset_from_cube": math.sqrt((cxw - xy[0]) ** 2 + (cyw - xy[1]) ** 2),
    }
    return info, "ok"


# ---------------------------------------------------------------- target list
# A cube counts only if a raised container covers it in the live scene (rejects table speckle).
container_cache = {}
for ci, c in enumerate(cubes):
    info, why = container_at(c["xyz"])
    container_cache[ci] = (info, why)
    c["container_ok"] = info is not None
    c["why"] = why
report["cube_container_check"] = [[c["color"], c["why"]] for c in cubes]
targets = []
plans = []
used = set()
for color, quant in clauses:
    same = []
    for ci, c in enumerate(cubes):
        if c["color"] == color and c["container_ok"] and ci not in used:
            same.append(ci)
    if not same:
        report["errors"].append("no covered cube of color " + color + " found")
        continue
    if quant == "one":
        bestci = same[0]
        for ci in same:
            if cubes[ci]["frames"] > cubes[bestci]["frames"]:
                bestci = ci
        same = [bestci]
    for ci in same:
        used.add(ci)
        targets.append(cubes[ci])
        plans.append(container_cache[ci][0])
report["targets"] = [[t["color"], rnd(t["xyz"])] for t in targets]
report["containers"] = [None if p is None else {
    "center": rnd(p["center"]), "yaw_deg": round(math.degrees(p["yaw"]), 1), "size": rnd(p["size"]),
    "top_z": round(p["top_z"], 4), "table_z": round(p["table_z"], 4), "patch": p["patch"],
    "offset": round(p["offset_from_cube"], 4)} for p in plans]

# Order "all" groups greedily by proximity, keeping clause order.
# (targets are already grouped by clause; within a color keep nearest-first from previous)

# ---------------------------------------------------------------- motion
q0 = [float(value) for value in ee0.quaternion_wxyz]
close_axis = column(hand_rot, 1)
phi0 = math.atan2(close_axis[1], close_axis[0])


def finger_clearance(info, closing_yaw, grasp_z):
    """Live points (non-table) in the two finger sweep zones beyond the container faces."""
    ca = math.cos(closing_yaw)
    sa = math.sin(closing_yaw)
    cx, cy = info["center"]
    # half extent of the container along this closing direction
    rel = (closing_yaw - info["yaw"]) % math.pi
    if abs(math.sin(rel)) < 0.7071:
        half = 0.5 * info["size"][0]
    else:
        half = 0.5 * info["size"][1]
    count = 0
    for p in live_points:
        if p[2] < grasp_z - 0.005 or p[2] > info["top_z"] + 0.02:
            continue
        dx = p[0] - cx
        dy = p[1] - cy
        along = abs(dx * ca + dy * sa)
        lateral = abs(-dx * sa + dy * ca)
        if along > half + 0.004 and along < 0.05 and lateral < 0.015:
            count += 1
    return count


def grasp_quat(info, grasp_z):
    options = []
    for k in range(4):
        delta = info["yaw"] + k * math.pi / 2.0 - phi0
        while delta > math.pi:
            delta -= 2 * math.pi
        while delta < -math.pi:
            delta += 2 * math.pi
        if abs(delta) <= math.pi / 2.0 + 1e-6:
            blocked = finger_clearance(info, phi0 + delta, grasp_z)
            options.append((blocked, abs(delta), delta))
    best = min(options)
    half = best[2] / 2.0
    qz = [math.cos(half), 0.0, 0.0, math.sin(half)]
    return quat_mul(qz, q0), best[2], [[o[0], round(math.degrees(o[2]), 1)] for o in options]


ended = {"flag": False}


def note(label, res):
    entry = {"call": label}
    try:
        entry["ok"] = bool(res.ok)
        entry["status"] = str(res.status)
        entry["steps"] = int(res.steps_executed)
    except Exception as exc:
        entry["err"] = str(exc)[:120]
    try:
        if res.terminated or res.truncated:
            entry["terminal"] = [bool(res.terminated), bool(res.truncated)]
            ended["flag"] = True
    except Exception:
        pass
    if not ended["flag"]:
        try:
            st = get_robot_state()
            entry["t"] = round(float(st.timestamp_s), 2)
            entry["grip"] = round(float(st.gripper_positions["primary"]), 3)
            entry["ee"] = rnd(st.end_effector_poses["primary"].position, 3)
        except Exception as exc:
            entry["state_err"] = str(exc)[:120]
    report["events"].append(entry)
    return res


def goto(label, xyz, quat, tol=0.01, steps=150):
    pose = Pose([float(xyz[0]), float(xyz[1]), float(xyz[2])], quat, base_frame)
    return note(label, move_to_pose(pose, tolerance=tol, max_steps=steps))


picks = []
for index, tgt in enumerate(targets):
    if ended["flag"]:
        break
    info = plans[index]
    if info is None:
        picks.append({"color": tgt["color"], "skipped": True})
        continue
    cx, cy = info["center"]
    top = info["top_z"]
    height = top - info["table_z"]
    grasp_z = max(top - min(0.03, 0.5 * height), info["table_z"] + 0.012)
    hover_z = top + 0.10
    quat, delta, options = grasp_quat(info, grasp_z)
    rec = {"color": tgt["color"], "delta_yaw_deg": round(math.degrees(delta), 1), "yaw_options": options,
           "grasp_z": round(grasp_z, 4), "hover_z": round(hover_z, 4)}
    picks.append(rec)
    label = tgt["color"] + str(index)
    note(label + ":open", open_gripper())
    if ended["flag"]:
        break
    goto(label + ":hover", (cx, cy, hover_z), quat, tol=0.015, steps=300)
    if ended["flag"]:
        break
    attempt = 0
    while attempt < 2 and not ended["flag"]:
        z_here = grasp_z - 0.008 * attempt
        if z_here < info["table_z"] + 0.01:
            z_here = info["table_z"] + 0.01
        goto(label + ":descend" + str(attempt), (cx, cy, z_here), quat, tol=0.008, steps=200)
        if ended["flag"]:
            break
        note(label + ":close" + str(attempt), close_gripper())
        if ended["flag"]:
            break
        grip = None
        try:
            grip = float(get_robot_state().gripper_positions["primary"])
        except Exception:
            grip = None
        rec["grip_after_close" + str(attempt)] = grip
        if grip is not None and grip < 0.05 and attempt == 0:
            # Jaw closed on nothing: reopen and retry once slightly lower.
            note(label + ":reopen", open_gripper())
            attempt += 1
            continue
        goto(label + ":lift", (cx, cy, z_here + 0.10), quat, tol=0.015, steps=200)
        if ended["flag"]:
            break
        try:
            rec["grip_after_lift"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
        except Exception:
            pass
        goto(label + ":lower", (cx, cy, z_here + 0.008), quat, tol=0.01, steps=200)
        if ended["flag"]:
            break
        note(label + ":release", open_gripper())
        break
    if ended["flag"]:
        break
    goto(label + ":rise", (cx, cy, hover_z), quat, tol=0.02, steps=200)

report["picks"] = picks
report["ended"] = ended["flag"]
result = report
