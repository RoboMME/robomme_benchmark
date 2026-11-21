# ButtonUnmask (RoboMME-Plus, task 1) -- v01: grasp-geometry fix (finer container yaw fit, re-localize + retry).
#
# Strategy (derived from the released recordings' task STRUCTURE, never their coordinates):
#   1. Parse the goal at runtime: ordered list of (cube colour, one|all).
#   2. Memory phase: coloured cubes appear on the table in waves and are then covered by
#      identical white containers. Every control tick we scan the live front AND wrist RGB
#      for saturated cube colours (pure Python), project blob pixels through the live depth,
#      intrinsics and camera_pose into robot_base, and cluster detections per colour.
#      The conditioning video (if the episode has one) is scanned the same way.
#   3. Press the button (localized at runtime) before the published press deadline,
#      while still observing every tick.
#   4. Return to the start pose and keep observing until no cube colour has been visible
#      for QUIET_TICKS ticks (later waves can appear after the press).
#   5. For each goal target in order: hover above the remembered cube XY, re-measure the
#      covering container from wrist depth (top face, centre, yaw), grasp top-down, lift,
#      then lower it back and release (the last one stays lifted).
#
# Only public APIs; only `import math`. All episode-dependent quantities are derived at runtime.

import math

# ---- generic tolerances / limits (not episode-specific) ----
PRESS_DEADLINE_STEP = 90       # PUBLIC_RULES.md: ButtonUnmask press must register before step 90
PRESS_MARGIN_TICKS = 14        # arrive at the button this many ticks before the deadline
STAY_AFTER_FIRST_SEEN = 15     # ticks to keep the start pose after a wave first appears
QUIET_TICKS = 25               # memory phase ends after this many ticks without any cube colour
MEMORY_HARD_CAP_STEP = 330     # never observe past this step
PIXEL_STRIDE = 2
MIN_BLOB_PIXELS = 3
CLUSTER_RADIUS = 0.04
MERGE_RADIUS = 0.045
MIN_CLUSTER_OBS = 3
MAX_JOINT_DELTA = 0.05         # rad per tick for free motion
SLOW_JOINT_DELTA = 0.02        # rad per tick for contact motion
PRESS_ABOVE = 0.08
PRESS_BELOW_TOP = 0.02
HOVER_ABOVE_TABLE = 0.28
PREGRASP_ABOVE_TOP = 0.07
LIFT_HEIGHT = 0.10
HOLD_LIFT_TICKS = 10
MAX_VIDEO_FRAMES = 120
MAX_GRASP_ATTEMPTS = 3
RELOCALIZE_ABOVE_TOP = 0.16    # second look from directly above the estimate
EMPTY_GRIPPER = 0.03           # normalized width below which the jaw closed on nothing

COLOURS = ("red", "green", "blue", "yellow", "orange", "purple")
COLOUR_ALIASES = {"red": "red", "green": "green", "blue": "blue", "yellow": "yellow",
                  "orange": "orange", "purple": "purple", "violet": "purple", "magenta": "purple"}
HIDE_WORDS = ("hiding", "hides", "hide", "covering", "covers", "cover", "concealing",
              "conceals", "containing", "contains", "over")
ALL_WORDS = ("all", "every", "each", "both")

report = {"version": "v01", "events": [], "warnings": []}
S = {"done": False, "ticks": 0, "last_step_reply": None, "last_target": None}


def note(msg):
    if len(report["events"]) < 400:
        report["events"].append(msg)


def warn(msg):
    if len(report["warnings"]) < 200:
        report["warnings"].append(msg)


# ------------------------------------------------------------------ small math helpers
def pct(values, q):
    vals = sorted(values)
    if not vals:
        return 0.0
    idx = int(round((len(vals) - 1) * q))
    return vals[max(0, min(len(vals) - 1, idx))]


def median(values):
    return pct(values, 0.5)


def quat_to_rot(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def yaw_quat(angle):
    return [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]


def wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def dist_xy(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ------------------------------------------------------------------ goal parsing
def parse_goal(text):
    t = str(text).lower()
    for ch in ",.;:!?()\"'":
        t = t.replace(ch, " ")
    words = t.split()
    targets = []
    for i in range(len(words)):
        if words[i] not in HIDE_WORDS:
            continue
        colour = None
        plural = False
        for j in range(i + 1, min(len(words), i + 5)):
            if words[j] in COLOUR_ALIASES:
                colour = COLOUR_ALIASES[words[j]]
                if j + 1 < len(words) and words[j + 1] in ("cubes", "blocks", "ones"):
                    plural = True
                break
        if colour is None:
            continue
        for j in range(max(0, i - 4), i):
            if words[j] in ALL_WORDS or words[j] in ("containers", "bins", "boxes", "cups", "ones"):
                plural = True
        targets.append({"colour": colour, "mode": "all" if plural else "one"})
    if not targets:
        start = 0
        if "then" in words:
            start = words.index("then")
        for j in range(start, len(words)):
            if words[j] in COLOUR_ALIASES:
                targets.append({"colour": COLOUR_ALIASES[words[j]], "mode": "one"})
    return targets


# ------------------------------------------------------------------ pixel colour classification
def classify_rgb(r, g, b):
    mx = r if r > g else g
    mx = mx if mx > b else b
    mn = r if r < g else g
    mn = mn if mn < b else b
    if mx < 50:
        return None
    d = mx - mn
    if d * 100 < 35 * mx:
        return None
    s = d / mx
    if mx == r:
        h = (60.0 * (g - b) / d) % 360.0
    elif mx == g:
        h = 60.0 * (b - r) / d + 120.0
    else:
        h = 60.0 * (r - g) / d + 240.0
    if (h >= 345.0 or h < 10.0) and s >= 0.75:
        return "red"
    if 85.0 <= h < 160.0 and s >= 0.6:
        return "green"
    if 215.0 <= h < 262.0 and s >= 0.7:
        return "blue"
    if 48.0 <= h < 72.0 and s >= 0.7:
        return "yellow"
    if 30.0 <= h < 48.0 and s >= 0.88:
        return "orange"
    if 265.0 <= h < 320.0 and s >= 0.35:
        return "purple"
    return None


def is_table_rgb(r, g, b):
    # warm wood colour, moderately saturated (used only to pick calibration pixels)
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 60 or mx != r:
        return False
    s = (mx - mn) / mx
    return 0.3 <= s <= 0.8 and g > b


def find_blobs(rows, wanted, stride, min_pixels):
    height = len(rows)
    width = len(rows[0])
    labels = {}
    for v in range(0, height, stride):
        row = rows[v]
        for u in range(0, width, stride):
            px = row[u]
            c = classify_rgb(int(px[0]), int(px[1]), int(px[2]))
            if c is not None and c in wanted:
                labels[(u, v)] = c
    seen = set()
    blobs = []
    for key in labels:
        if key in seen:
            continue
        colour = labels[key]
        stack = [key]
        seen.add(key)
        pix = []
        while stack:
            cur = stack.pop()
            pix.append(cur)
            cu, cv = cur
            for nb in ((cu + stride, cv), (cu - stride, cv), (cu, cv + stride), (cu, cv - stride)):
                if nb not in seen and labels.get(nb) == colour:
                    seen.add(nb)
                    stack.append(nb)
        if len(pix) >= min_pixels:
            su = sum(p[0] for p in pix) / len(pix)
            sv = sum(p[1] for p in pix) / len(pix)
            blobs.append({"colour": colour, "n": len(pix), "u": su, "v": sv, "pix": pix})
    return blobs


def to_rows(array):
    try:
        return array.tolist()
    except Exception:
        return [[array[v][u] for u in range(len(array[0]))] for v in range(len(array))]


# ------------------------------------------------------------------ camera geometry
CONVENTIONS = (("cv", "z"), ("cv", "range"), ("gl", "z"), ("gl", "range"), ("link", "z"), ("link", "range"))


def camera_model(cam):
    K = cam.intrinsics
    pose = cam.camera_pose
    return {"fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]), "cy": float(K[1][2]),
            "R": quat_to_rot(pose.quaternion_wxyz), "t": [float(v) for v in pose.position]}


def project(model, u, v, depth, conv):
    xn = (u - model["cx"]) / model["fx"]
    yn = (v - model["cy"]) / model["fy"]
    if conv[0] == "cv":
        d = (xn, yn, 1.0)
    elif conv[0] == "gl":
        d = (xn, -yn, -1.0)
    else:
        d = (1.0, -xn, -yn)
    scale = depth
    if conv[1] == "range":
        scale = depth / math.sqrt(1.0 + xn * xn + yn * yn)
    p = (d[0] * scale, d[1] * scale, d[2] * scale)
    R = model["R"]
    t = model["t"]
    return (R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2] + t[0],
            R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2] + t[1],
            R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2] + t[2])


def calibrate_camera(cam, name):
    """Pick the camera-frame convention under which table pixels form a horizontal plane."""
    rows = to_rows(cam.rgb)
    depth = to_rows(cam.depth_m)
    model = camera_model(cam)
    h = len(rows)
    w = len(rows[0])
    samples = []
    fallback = []
    step_px = max(4, w // 32)
    for v in range(step_px // 2, h, step_px):
        for u in range(step_px // 2, w, step_px):
            d = float(depth[v][u])
            if not (d > 0.02 and d < 10.0):
                continue
            px = rows[v][u]
            fallback.append((u, v, d))
            if is_table_rgb(int(px[0]), int(px[1]), int(px[2])):
                samples.append((u, v, d))
    if len(samples) < 30:
        samples = fallback
    best = None
    scores = {}
    for conv in CONVENTIONS:
        zs = [project(model, u, v, d, conv)[2] for (u, v, d) in samples]
        if len(zs) < 10:
            continue
        zm = median(zs)
        mad = median([abs(z - zm) for z in zs])
        score = mad
        if zm > model["t"][2]:
            score += 10.0
        scores[conv[0] + "/" + conv[1]] = round(score, 5)
        if best is None or score < best[0]:
            best = (score, conv, zm)
    if best is None:
        return None
    note("calib " + name + " conv=" + best[1][0] + "/" + best[1][1] + " table_z=" + str(round(best[2], 4))
         + " mad=" + str(round(best[0], 5)) + " n=" + str(len(samples)))
    report["calibration_" + name] = scores
    return {"conv": best[1], "table_z": best[2], "mad": best[0]}


# ------------------------------------------------------------------ robot helpers
def now_step():
    st = get_robot_state()
    ts = float(st.timestamp_s)
    if ts > 0.0:
        return int(round(ts * HZ))
    return S["ticks"]


def current_joints():
    return [float(v) for v in get_robot_state().joint_positions["primary"]]


def check_terminal(reply):
    flags = []
    for name in ("terminated", "truncated"):
        try:
            if name == "terminated" and bool(reply.terminated):
                flags.append(name)
            if name == "truncated" and bool(reply.truncated):
                flags.append(name)
        except Exception:
            pass
    if flags:
        S["done"] = True
        note("terminal flags " + str(flags) + " at step " + str(S["ticks"]))
    return S["done"]


def send(target, grip):
    if S["done"]:
        return False
    reply = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=[float(v) for v in target],
        gripper_position=float(grip), embodiment=EMBODIMENT)}))
    S["ticks"] += 1
    S["last_step_reply"] = reply
    S["last_target"] = [float(v) for v in target]
    check_terminal(reply)
    return not S["done"]


def go_joints(target, grip, observer=None, max_delta=MAX_JOINT_DELTA, settle_tol=0.01, settle_max=12):
    start = current_joints()
    dist = max(abs(target[i] - start[i]) for i in range(7))
    n = max(1, int(math.ceil(1.3 * dist / max_delta)))
    for k in range(1, n + 1):
        if S["done"]:
            return False
        a = 0.5 - 0.5 * math.cos(math.pi * k / n)
        send([start[i] + a * (target[i] - start[i]) for i in range(7)], grip)
        if observer is not None:
            observer()
    cur = current_joints()
    err = max(abs(target[i] - cur[i]) for i in range(7))
    for settle_tick in range(settle_max):
        if err < settle_tol or S["done"]:
            break
        send(target, grip)
        if observer is not None:
            observer()
        cur = current_joints()
        err = max(abs(target[i] - cur[i]) for i in range(7))
    return err < 0.05


def hold(ticks, grip, observer=None):
    target = S["last_target"] if S["last_target"] is not None else current_joints()
    for tick in range(ticks):
        if S["done"]:
            return
        send(target, grip)
        if observer is not None:
            observer()


def ik(position, quat, seed):
    try:
        res = solve_ik(Pose([float(v) for v in position], [float(v) for v in quat], BASE_FRAME),
                       seed_joints=[float(v) for v in seed])
    except Exception as exc:
        warn("solve_ik raised " + str(exc)[:120])
        return None
    if not res.ok:
        return None
    try:
        return [float(v) for v in res.joint_positions]
    except Exception:
        try:
            return [float(v) for v in res.solution]
        except Exception:
            return None


def ee_position():
    p = get_robot_state().end_effector_poses["primary"].position
    return [float(p[0]), float(p[1]), float(p[2])]


def go_cartesian(goal, quat, grip, observer=None, seg=0.06, max_delta=MAX_JOINT_DELTA):
    """Straight-ish Cartesian move through IK waypoints spaced <= seg metres."""
    start = ee_position()
    d = math.sqrt(sum((goal[i] - start[i]) ** 2 for i in range(3)))
    n = max(1, int(math.ceil(d / seg)))
    seed = current_joints()
    plan = []
    for k in range(1, n + 1):
        a = k / n
        wp = [start[i] + a * (goal[i] - start[i]) for i in range(3)]
        sol = ik(wp, quat, seed)
        if sol is None:
            warn("IK failed on cartesian waypoint " + str([round(x, 3) for x in wp]))
            return False
        plan.append(sol)
        seed = sol
    for sol in plan[:-1]:
        go_joints(sol, grip, observer, max_delta=max_delta, settle_max=0)
        if S["done"]:
            return False
    return go_joints(plan[-1], grip, observer, max_delta=max_delta, settle_max=12)


# ------------------------------------------------------------------ memory: cube clusters
CLUSTERS = {}      # colour -> list of clusters
ACTIVITY = {"last_seen_step": -1, "first_seen_step": -1, "frames": 0}


def add_detection(colour, p, step_idx, cam_name):
    lst = CLUSTERS.setdefault(colour, [])
    best = None
    for cl in lst:
        d = dist_xy(p, (median(cl["xs"]), median(cl["ys"])))
        if d < CLUSTER_RADIUS and (best is None or d < best[0]):
            best = (d, cl)
    if best is None:
        cl = {"colour": colour, "xs": [], "ys": [], "zs": [], "wx": [], "wy": [], "cams": {},
              "first": step_idx, "last": step_idx}
        lst.append(cl)
    else:
        cl = best[1]
    if len(cl["xs"]) < 300:
        cl["xs"].append(p[0])
        cl["ys"].append(p[1])
        cl["zs"].append(p[2])
        if cam_name == "wrist":
            cl["wx"].append(p[0])
            cl["wy"].append(p[1])
    cl["cams"][cam_name] = cl["cams"].get(cam_name, 0) + 1
    cl["last"] = step_idx


def scan_observation(obs, step_idx, wanted_goal):
    seen_any = False
    for cam_name in ("front", "wrist"):
        if cam_name not in CALIB or CALIB[cam_name] is None:
            continue
        try:
            cam = obs.cameras[cam_name]
        except Exception:
            continue
        rows = to_rows(cam.rgb)
        blobs = find_blobs(rows, COLOURS, PIXEL_STRIDE, MIN_BLOB_PIXELS)
        if not blobs:
            continue
        seen_any = True
        goal_blobs = [b for b in blobs if b["colour"] in wanted_goal]
        if not goal_blobs:
            continue
        depth = to_rows(cam.depth_m)
        model = camera_model(cam)
        conv = CALIB[cam_name]["conv"]
        for b in goal_blobs:
            pts = []
            for (u, v) in b["pix"]:
                d = float(depth[v][u])
                if d > 0.02 and d < 10.0:
                    pts.append(project(model, u, v, d, conv))
            if len(pts) < 2:
                continue
            p = (median([q[0] for q in pts]), median([q[1] for q in pts]), median([q[2] for q in pts]))
            if p[2] < TABLE_Z - 0.03 or p[2] > TABLE_Z + 0.10:
                continue
            add_detection(b["colour"], p, step_idx, cam_name)
    if seen_any:
        ACTIVITY["last_seen_step"] = step_idx
        ACTIVITY["frames"] += 1
        if ACTIVITY["first_seen_step"] < 0:
            ACTIVITY["first_seen_step"] = step_idx
    return seen_any


def live_observer():
    if S["done"]:
        return
    try:
        scan_observation(get_observation(), now_step(), GOAL_COLOURS)
    except Exception as exc:
        warn("scan failed: " + str(exc)[:160])


def merged_clusters(colour):
    lst = [cl for cl in CLUSTERS.get(colour, []) if len(cl["xs"]) >= MIN_CLUSTER_OBS]
    out = []
    for cl in lst:
        if len(cl["wx"]) >= 2:
            p = (median(cl["wx"]), median(cl["wy"]))
        else:
            p = (median(cl["xs"]), median(cl["ys"]))
        merged = False
        for o in out:
            if dist_xy(o["p"], p) < MERGE_RADIUS:
                o["n"] += len(cl["xs"])
                merged = True
                break
        if not merged:
            out.append({"colour": colour, "p": p, "n": len(cl["xs"]), "z": median(cl["zs"]),
                        "cams": dict(cl["cams"]), "first": cl["first"], "last": cl["last"], "used": False})
    return out


# ------------------------------------------------------------------ button localization
def cloud_points(cloud):
    try:
        pts = cloud.points.tolist()
    except Exception:
        pts = [list(p) for p in cloud.points]
    return [(float(p[0]), float(p[1]), float(p[2])) for p in pts]


def locate_button_sam():
    best = None
    for cam_name in ("wrist", "front"):
        for prompt in ("button", "round push button", "push button"):
            try:
                found = localize_object(prompt, camera_name=cam_name, target_frame=BASE_FRAME)
            except Exception as exc:
                warn("localize_object raised " + str(exc)[:100])
                continue
            if not found.ok:
                continue
            pts = cloud_points(found.point_cloud)
            if len(pts) < 15:
                continue
            zs = [p[2] for p in pts]
            top = pct(zs, 0.95)
            span_x = pct([p[0] for p in pts], 0.95) - pct([p[0] for p in pts], 0.05)
            span_y = pct([p[1] for p in pts], 0.95) - pct([p[1] for p in pts], 0.05)
            if top - TABLE_Z < -0.01 or top - TABLE_Z > 0.09 or span_x > 0.15 or span_y > 0.15:
                note("button candidate rejected cam=" + cam_name + " prompt=" + prompt + " top-table="
                     + str(round(top - TABLE_Z, 3)) + " span=" + str(round(span_x, 3)) + "," + str(round(span_y, 3)))
                continue
            cap = [p for p in pts if p[2] > top - 0.01]
            cx = sum(p[0] for p in cap) / len(cap)
            cy = sum(p[1] for p in cap) / len(cap)
            note("button via SAM cam=" + cam_name + " prompt=" + prompt + " xy=" + str([round(cx, 3), round(cy, 3)])
                 + " top=" + str(round(top, 3)))
            return {"x": cx, "y": cy, "top": top, "source": "sam/" + cam_name + "/" + prompt}
    return best


def locate_button_geometric(obs):
    """Fallback: smallest low compact cluster standing on the table in the start scene."""
    cands = []
    for cam_name in ("wrist", "front"):
        if CALIB.get(cam_name) is None:
            continue
        cam = obs.cameras[cam_name]
        depth = to_rows(cam.depth_m)
        model = camera_model(cam)
        conv = CALIB[cam_name]["conv"]
        pts = []
        for v in range(0, len(depth), 2):
            for u in range(0, len(depth[0]), 2):
                d = float(depth[v][u])
                if d > 0.02 and d < 10.0:
                    p = project(model, u, v, d, conv)
                    if p[2] > TABLE_Z + 0.006:
                        pts.append(p)
        # greedy xy clustering
        clusters = []
        for p in pts:
            placed = False
            for cl in clusters:
                if dist_xy(p, cl["c"]) < 0.05:
                    cl["pts"].append(p)
                    placed = True
                    break
            if not placed:
                clusters.append({"c": (p[0], p[1]), "pts": [p]})
        for cl in clusters:
            zs = [p[2] for p in cl["pts"]]
            if len(zs) < 8:
                continue
            top = pct(zs, 0.95)
            if top - TABLE_Z > 0.08:
                continue
            cap = [p for p in cl["pts"] if p[2] > top - 0.01]
            cands.append({"x": sum(p[0] for p in cap) / len(cap), "y": sum(p[1] for p in cap) / len(cap),
                          "top": top, "n": len(zs), "source": "geometric/" + cam_name})
        if cands:
            break
    if not cands:
        return None
    best = None
    for c in cands:
        if best is None or c["n"] > best["n"]:
            best = c
    note("button via geometry " + str([round(best["x"], 3), round(best["y"], 3), round(best["top"], 3)]))
    return best


# ------------------------------------------------------------------ container localization
def camera_points(cam, conv, centre, radius, zmin, zmax, stride=2):
    depth = to_rows(cam.depth_m)
    model = camera_model(cam)
    pts = []
    for v in range(0, len(depth), stride):
        row = depth[v]
        for u in range(0, len(row), stride):
            d = float(row[u])
            if not (d > 0.02 and d < 10.0):
                continue
            p = project(model, u, v, d, conv)
            if zmin < p[2] < zmax and dist_xy(p, centre) < radius:
                pts.append(p)
    return pts


def rect_area_at(pts, cx, cy, th):
    c = math.cos(th)
    s = math.sin(th)
    a = [(p[0] - cx) * c + (p[1] - cy) * s for p in pts]
    b = [-(p[0] - cx) * s + (p[1] - cy) * c for p in pts]
    a0, a1 = pct(a, 0.02), pct(a, 0.98)
    b0, b1 = pct(b, 0.02), pct(b, 0.98)
    ma = 0.5 * (a0 + a1)
    mb = 0.5 * (b0 + b1)
    return ((a1 - a0) * (b1 - b0), th, a1 - a0, b1 - b0, cx + ma * c - mb * s, cy + ma * s + mb * c)


def fit_rect(pts, cx, cy):
    """Minimum-area bounding rectangle: coarse 1.5 deg search over [0, 90), then 0.25 deg refinement."""
    best = None
    for k in range(60):
        cand = rect_area_at(pts, cx, cy, math.radians(1.5 * k))
        if best is None or cand[0] < best[0]:
            best = cand
    centre_th = best[1]
    for k in range(-6, 7):
        cand = rect_area_at(pts, cx, cy, centre_th + math.radians(0.25 * k))
        if cand[0] < best[0]:
            best = cand
    th = best[1] % (math.pi / 2.0)
    return {"theta": th, "ea": best[2], "eb": best[3], "cx": best[4], "cy": best[5]}


def locate_container(obs, guess, cam_name):
    if CALIB.get(cam_name) is None:
        return None
    cam = obs.cameras[cam_name]
    conv = CALIB[cam_name]["conv"]
    pts = camera_points(cam, conv, guess, 0.07, TABLE_Z + 0.012, TABLE_Z + 0.14)
    near = [p for p in pts if dist_xy(p, guess) < 0.04]
    if len(near) < 12:
        note("container " + cam_name + ": too few points near guess (" + str(len(near)) + ")")
        return None
    top = pct([p[2] for p in near], 0.9)
    band = [p for p in pts if p[2] > top - 0.012]
    c = (guess[0], guess[1])
    sel = []
    for it in range(6):
        sel = [p for p in band if dist_xy(p, c) < 0.035]
        if len(sel) < 6:
            break
        c = (sum(p[0] for p in sel) / len(sel), sum(p[1] for p in sel) / len(sel))
    if len(sel) < 6 or dist_xy(c, guess) > 0.05:
        note("container " + cam_name + ": mean-shift failed")
        return None
    rect = fit_rect(sel, c[0], c[1])
    body = [p for p in pts if dist_xy(p, (rect["cx"], rect["cy"])) < 0.05]
    cth = math.cos(rect["theta"])
    sth = math.sin(rect["theta"])
    ba = [(p[0] - rect["cx"]) * cth + (p[1] - rect["cy"]) * sth for p in body]
    bb = [-(p[0] - rect["cx"]) * sth + (p[1] - rect["cy"]) * cth for p in body]
    body_a = pct(ba, 0.95) - pct(ba, 0.05)
    body_b = pct(bb, 0.95) - pct(bb, 0.05)
    out = {"cx": rect["cx"], "cy": rect["cy"], "top": top, "theta": rect["theta"],
           "top_a": rect["ea"], "top_b": rect["eb"], "body_a": body_a, "body_b": body_b,
           "n_top": len(sel), "n_body": len(body), "cam": cam_name}
    note("container " + cam_name + " " + str({k: (round(v, 3) if k != "cam" else v) for k, v in out.items()}))
    return out


# ------------------------------------------------------------------ tall obstacles (e.g. occluding pillar)
def tall_obstacle_points(obs):
    pts = []
    home_r = math.sqrt(HOME_EE[0] ** 2 + HOME_EE[1] ** 2)
    for cam_name in ("front",):
        if CALIB.get(cam_name) is None:
            continue
        cam = obs.cameras[cam_name]
        depth = to_rows(cam.depth_m)
        model = camera_model(cam)
        conv = CALIB[cam_name]["conv"]
        for v in range(0, len(depth), 4):
            for u in range(0, len(depth[0]), 4):
                d = float(depth[v][u])
                if d > 0.02 and d < 10.0:
                    p = project(model, u, v, d, conv)
                    if p[2] > TABLE_Z + 0.15 and math.sqrt(p[0] ** 2 + p[1] ** 2) > home_r + 0.08:
                        pts.append(p)
    return pts


def seg_point_dist(p, a, b):
    vx, vy = b[0] - a[0], b[1] - a[1]
    L = vx * vx + vy * vy
    if L < 1e-9:
        return dist_xy(p, a)
    t = max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / L))
    return dist_xy(p, (a[0] + t * vx, a[1] + t * vy))


def transit(goal_xy, quat, grip, hover_z=None):
    """Rise, move horizontally at a safe height, return when above goal_xy."""
    cur = ee_position()
    z = max(TABLE_Z + HOVER_ABOVE_TABLE, cur[2] if cur[2] < TABLE_Z + 0.45 else TABLE_Z + HOVER_ABOVE_TABLE)
    if hover_z is not None:
        z = hover_z
    blocking = [p for p in OBSTACLES if seg_point_dist(p, cur, goal_xy) < 0.10]
    via = None
    if blocking:
        top = max(p[2] for p in blocking)
        if top + 0.08 <= TABLE_Z + 0.45:
            z = max(z, top + 0.08)
        else:
            via = (HOME_EE[0], HOME_EE[1])
        note("transit: " + str(len(blocking)) + " tall obstacle points near path; z=" + str(round(z, 3))
             + " via_home=" + str(via is not None))
    if abs(cur[2] - z) > 0.01:
        if not go_cartesian([cur[0], cur[1], z], quat, grip):
            return False
    if via is not None:
        if not go_cartesian([via[0], via[1], z], quat, grip):
            return False
    return go_cartesian([goal_xy[0], goal_xy[1], z], quat, grip)


# ------------------------------------------------------------------ grasp one container
def grasp_quat_for(theta_axes):
    """Choose a gripper yaw so the fingers close along one of the given world axes (radians)."""
    R = quat_to_rot(DOWN_Q)
    phi = math.atan2(R[1][1], R[0][1])  # current closing (hand y) direction
    best = None
    for ax in theta_axes:
        for flip in (0.0, math.pi):
            dlt = wrap_pi(ax + flip - phi)
            if best is None or abs(dlt) < abs(best):
                best = dlt
    return quat_mul(yaw_quat(best), DOWN_Q), best


def observe_container(guess, hover_z):
    """Hover above guess (wrist camera looking down), re-measure the container; front camera as fallback."""
    if not transit(guess, DOWN_Q, 1.0, hover_z):
        return None, "transit_failed"
    if S["done"]:
        return None, "terminal"
    hold(3, 1.0)
    obs = get_observation()
    geo = locate_container(obs, guess, "wrist")
    if geo is None:
        geo = locate_container(obs, guess, "front")
    return geo, "ok"


def pick_container(target, is_last):
    guess = target["p"]
    attempts = []
    target["attempts"] = attempts
    geo, status = observe_container(guess, None)
    if status != "ok":
        return status
    if geo is None:
        warn("container not re-localized; using remembered cube xy")
        attempts.append("no_relocalization")
        geo = {"cx": guess[0], "cy": guess[1], "top": TABLE_Z + 0.06, "theta": 0.0,
               "top_a": 0.05, "top_b": 0.05, "body_a": 0.06, "body_b": 0.06, "cam": "none"}
    target["geometry"] = {k: (round(v, 4) if k != "cam" else v) for k, v in geo.items()}
    for attempt in range(MAX_GRASP_ATTEMPTS):
        if S["done"]:
            return "terminal"
        if attempt > 0:
            # re-measure from directly above the last estimate (centred, closer view)
            geo2, status = observe_container((geo["cx"], geo["cy"]), geo["top"] + RELOCALIZE_ABOVE_TOP)
            if status == "terminal":
                return "terminal"
            if geo2 is not None and dist_xy((geo2["cx"], geo2["cy"]), guess) < 0.05:
                geo = geo2
                attempts.append("relocalized")
        th = geo["theta"]
        axes = [th, th + math.pi / 2.0]
        if geo["body_a"] + 0.01 < geo["body_b"]:
            axes = [th]
        elif geo["body_b"] + 0.01 < geo["body_a"]:
            axes = [th + math.pi / 2.0]
        gq, dyaw = grasp_quat_for(axes)
        height = geo["top"] - TABLE_Z
        body_min = min(geo["body_a"], geo["body_b"])
        depth_below = max(0.012, min(0.03, 0.35 * height))
        if body_min > 0.074 or attempt == MAX_GRASP_ATTEMPTS - 1 and body_min > 0.068:
            depth_below = 0.008
        pre = [geo["cx"], geo["cy"], geo["top"] + PREGRASP_ABOVE_TOP]
        grasp_z = geo["top"] - depth_below
        if not go_cartesian(pre, gq, 1.0):
            attempts.append("pregrasp_ik_failed")
            continue
        hold(4, 1.0)
        if not go_cartesian([geo["cx"], geo["cy"], grasp_z], gq, 1.0, max_delta=SLOW_JOINT_DELTA):
            attempts.append("descend_not_converged")
        if S["done"]:
            return "terminal"
        try:
            close_gripper()
        except Exception as exc:
            warn("close_gripper raised " + str(exc)[:100])
            hold(10, 0.0)
        hold(3, 0.0)
        g = float(get_robot_state().gripper_positions["primary"])
        attempts.append("closed g=" + str(round(g, 3)) + " yaw=" + str(round(math.degrees(th), 1)))
        if g < EMPTY_GRIPPER:
            hold(8, 1.0)
            go_cartesian(pre, gq, 1.0)
            continue
        lift = [geo["cx"], geo["cy"], grasp_z + LIFT_HEIGHT]
        go_cartesian(lift, gq, 0.0, max_delta=SLOW_JOINT_DELTA * 1.5)
        hold(HOLD_LIFT_TICKS, 0.0)
        g2 = float(get_robot_state().gripper_positions["primary"])
        attempts.append("lifted g=" + str(round(g2, 3)))
        target["geometry"] = {k: (round(v, 4) if k != "cam" else v) for k, v in geo.items()}
        if S["done"]:
            return "terminal"
        if g2 < EMPTY_GRIPPER:
            attempts.append("slipped")
            hold(6, 1.0)
            continue
        if is_last:
            return "lifted"
        go_cartesian([geo["cx"], geo["cy"], grasp_z + 0.004], gq, 0.0, max_delta=SLOW_JOINT_DELTA * 1.5)
        hold(8, 1.0)
        go_cartesian(pre, gq, 1.0)
        return "lifted_and_released"
    return "grasp_failed"


# ================================================================== main program
context = get_task_context()
GOAL = str(context.language)
report["goal"] = GOAL
META = {}
try:
    META = dict(robomme_plus.get_task_metadata())
except Exception as exc:
    warn("get_task_metadata failed: " + str(exc)[:100])
HZ = 20.0
for key in ("control_hz", "control_frequency", "control_freq", "frequency_hz", "frequency"):
    if key in META:
        try:
            HZ = float(META[key])
            break
        except Exception:
            pass
MAX_STEPS = None
try:
    MAX_STEPS = int(META["max_steps"])
except Exception:
    warn("max_steps not in metadata")
report["max_steps"] = MAX_STEPS
report["hz"] = HZ

TARGETS = parse_goal(GOAL)
report["targets"] = [dict(t) for t in TARGETS]
GOAL_COLOURS = set(t["colour"] for t in TARGETS)
WANTS_PRESS = ("button" in GOAL.lower())

init_state = get_robot_state()
EMBODIMENT = init_state.embodiment
BASE_FRAME = init_state.base_frame
HOME_JOINTS = [float(v) for v in init_state.joint_positions["primary"]]
home_pose = init_state.end_effector_poses["primary"]
HOME_EE = [float(v) for v in home_pose.position]
DOWN_Q = [float(v) for v in home_pose.quaternion_wxyz]
Rh = quat_to_rot(DOWN_Q)
if Rh[2][2] > -0.9:
    warn("start tool axis not pointing down (z=" + str(round(Rh[2][2], 3)) + "); using canonical down")
    DOWN_Q = [0.0, 1.0, 0.0, 0.0]

obs0 = get_observation()
CALIB = {}
for cam_name in ("front", "wrist"):
    try:
        CALIB[cam_name] = calibrate_camera(obs0.cameras[cam_name], cam_name)
    except Exception as exc:
        warn("calibration failed for " + cam_name + ": " + str(exc)[:120])
        CALIB[cam_name] = None
if CALIB.get("front") is not None:
    TABLE_Z = CALIB["front"]["table_z"]
elif CALIB.get("wrist") is not None:
    TABLE_Z = CALIB["wrist"]["table_z"]
else:
    TABLE_Z = 0.0
    warn("no camera calibration; table z assumed 0")
report["table_z"] = round(TABLE_Z, 4)

OBSTACLES = []
try:
    OBSTACLES = tall_obstacle_points(obs0)
except Exception as exc:
    warn("obstacle scan failed: " + str(exc)[:100])
report["tall_obstacle_points"] = len(OBSTACLES)

BUTTON = None
if WANTS_PRESS:
    BUTTON = locate_button_sam()
    if BUTTON is None:
        try:
            BUTTON = locate_button_geometric(obs0)
        except Exception as exc:
            warn("geometric button search failed: " + str(exc)[:100])
report["button"] = None if BUTTON is None else {k: (round(v, 4) if k != "source" else v) for k, v in BUTTON.items()}

PRE_PRESS_J = None
PRESS_J = None
if BUTTON is not None:
    PRE_PRESS_J = ik([BUTTON["x"], BUTTON["y"], BUTTON["top"] + PRESS_ABOVE], DOWN_Q, HOME_JOINTS)
    if PRE_PRESS_J is not None:
        PRESS_J = ik([BUTTON["x"], BUTTON["y"], BUTTON["top"] - PRESS_BELOW_TOP], DOWN_Q, PRE_PRESS_J)
    if PRESS_J is None:
        warn("IK failed for button press poses")

# scan the conditioning video (if the episode has one)
try:
    video = robomme_plus.get_conditioning_video()
    nvid = len(video)
except Exception as exc:
    video = ()
    nvid = 0
    warn("conditioning video unavailable: " + str(exc)[:100])
report["conditioning_video_frames"] = nvid
if nvid > 0:
    stride_v = max(1, int(math.ceil(nvid / float(MAX_VIDEO_FRAMES))))
    for vi in range(0, nvid, stride_v):
        try:
            scan_observation(video[vi], -1, GOAL_COLOURS)
        except Exception as exc:
            warn("video frame scan failed: " + str(exc)[:100])
    note("video scanned; clusters " + str({c: len(v) for c, v in CLUSTERS.items()}))

# ---- phase A: watch from the start pose, then press the button before the deadline
scan_observation(obs0, 0, GOAL_COLOURS)
press_report = {"pressed": False}
if PRESS_J is not None:
    travel = int(math.ceil(1.3 * max(abs(PRE_PRESS_J[i] - HOME_JOINTS[i]) for i in range(7)) / MAX_JOINT_DELTA))
    travel += int(math.ceil(1.3 * max(abs(PRESS_J[i] - PRE_PRESS_J[i]) for i in range(7)) / SLOW_JOINT_DELTA))
    travel += 10
    latest_leave = PRESS_DEADLINE_STEP - 1 - PRESS_MARGIN_TICKS - travel
    press_report["travel_ticks_est"] = travel
    press_report["latest_leave"] = latest_leave
    while not S["done"]:
        st = now_step()
        leave = latest_leave
        if ACTIVITY["first_seen_step"] >= 0:
            leave = min(leave, ACTIVITY["first_seen_step"] + STAY_AFTER_FIRST_SEEN)
        if st >= leave:
            break
        send(HOME_JOINTS, 1.0)
        live_observer()
    press_report["left_home_step"] = now_step()
    go_joints(PRE_PRESS_J, 0.0, live_observer)
    go_joints(PRESS_J, 0.0, live_observer, max_delta=SLOW_JOINT_DELTA, settle_tol=0.006, settle_max=6)
    press_report["bottom_step"] = now_step()
    hold(4, 0.0, live_observer)
    press_report["pressed"] = True
    press_report["late"] = press_report["bottom_step"] >= PRESS_DEADLINE_STEP
    if press_report["late"]:
        warn("button press reached bottom at/after the published deadline")
    go_joints(PRE_PRESS_J, 0.0, live_observer)
    go_joints(HOME_JOINTS, 1.0, live_observer)
elif WANTS_PRESS:
    warn("no button press possible (button not localized or IK failed)")
report["press"] = press_report

# ---- phase C: keep watching until no cube colour has been visible for QUIET_TICKS
while not S["done"]:
    st = now_step()
    if st >= MEMORY_HARD_CAP_STEP:
        note("memory phase hit hard cap")
        break
    last = ACTIVITY["last_seen_step"]
    if last >= 0 and st - last >= QUIET_TICKS and st >= PRESS_DEADLINE_STEP:
        break
    if last < 0 and st >= PRESS_DEADLINE_STEP + 3 * QUIET_TICKS:
        break
    send(HOME_JOINTS, 1.0)
    live_observer()
report["memory_end_step"] = now_step()
report["activity"] = dict(ACTIVITY)
summary = {}
for colour in CLUSTERS:
    summary[colour] = [{"xy": [round(median(cl["xs"]), 3), round(median(cl["ys"]), 3)], "n": len(cl["xs"]),
                        "cams": cl["cams"], "first": cl["first"], "last": cl["last"]} for cl in CLUSTERS[colour]]
report["raw_clusters"] = summary

# ---- phase D: pick containers in goal order
PLAN = []
for t in TARGETS:
    cands = [c for c in merged_clusters(t["colour"])]
    PLAN.append({"colour": t["colour"], "mode": t["mode"], "cands": cands})

picks = []
for idx, item in enumerate(PLAN):
    if S["done"]:
        break
    avail = [c for c in item["cands"] if not c["used"]]
    if item["mode"] == "one":
        best = None
        for c in avail:
            if best is None or c["n"] > best["n"]:
                best = c
        chosen = [] if best is None else [best]
    else:
        chosen = []
        remaining = list(avail)
        here = ee_position()
        while remaining:
            nxt = None
            for c in remaining:
                if nxt is None or dist_xy(c["p"], here) < dist_xy(nxt["p"], here):
                    nxt = c
            chosen.append(nxt)
            remaining = [c for c in remaining if c is not nxt]
            here = nxt["p"]
    if not chosen:
        warn("no remembered " + item["colour"] + " cube for target " + str(idx))
        picks.append({"colour": item["colour"], "status": "not_remembered"})
        continue
    for k, c in enumerate(chosen):
        c["used"] = True
        is_last = (idx == len(PLAN) - 1) and (k == len(chosen) - 1)
        status = pick_container(c, is_last)
        picks.append({"colour": item["colour"], "xy": [round(c["p"][0], 3), round(c["p"][1], 3)],
                      "status": status, "geometry": c.get("geometry"), "attempts": c.get("attempts"),
                      "step": now_step()})
        if S["done"]:
            break

report["picks"] = picks
report["final_step"] = now_step()
report["done_flag"] = S["done"]
result = report
