# v03 — memory/perception: detect cubes in BOTH front and wrist cameras and keep a list of
# distinct cube objects per colour (3D merge); ignore fixtures visible before any cube
# appears (e.g. a coloured button base); language parser handles "all ... <colour> cubes"
# (unmask every remembered cube of that colour); container localization picks the first
# camera whose view of the target is not occluded (front, then wrist); button from either camera.
# v02 — perception: return to the start (home) joints before localizing each container so the
# arm does not occlude the front view; reject implausible container rectangles (robot / slivers);
# ignore colour clusters lying on the table plane (glints).
# v01 — perception: hue-band cube colours; grasp: container yaw from min-area rectangle,
# mid-body grasp height, longer close, grip verification with one re-localized retry.
# (derived from v00 — ButtonUnmask initial program.)
# Strategy (all hypotheses from one snapshot; the scene at step 0 shows only a button):
#   1. Parse the colour order from get_task_context().language.
#   2. Press the button quickly (deadline: before control step 90) with closed fingers,
#      using per-tick joint-interpolated step() commands from public IK solutions.
#   3. Retreat to the start joints and WATCH the front camera tick by tick, remembering where
#      each coloured cube was last seen (pixel colour classes + public depth/calibration).
#   4. For each colour in order: segment the covering container at the remembered pixel,
#      grasp it top-down, lift it, scan (diagnostic), lower it back and release.
import math

report = {"events": [], "scans": []}
ctx = get_task_context()
lang = str(ctx.language)
report["language"] = lang
state0 = get_robot_state()
base_frame = state0.base_frame
emb = state0.embodiment
home_q = [float(v) for v in state0.joint_positions["primary"]]
down_q = [float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz]

COLOR_WORDS = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "white", "black", "cyan"]

# ---------------- language ----------------
words = lang.lower().replace(",", " ").replace(".", " ").split()
order = []
goals = []
for i, word in enumerate(words):
    if word in COLOR_WORDS and i + 1 < len(words) and words[i + 1].startswith("cube"):
        plural = words[i + 1] == "cubes" or ("all" in words[max(0, i - 4):i])
        order.append(word)
        goals.append({"color": word, "all": plural})
report["order"] = order
report["goals"] = goals

# ---------------- low-level stepping ----------------
tick = {"n": 0, "done": False, "term": None}


def send(q, grip):
    cmd = None
    try:
        cmd = ArmCommand("joint_position", [float(v) for v in q], gripper_position=float(grip), embodiment=emb)
    except TypeError:
        cmd = ArmCommand("joint_position", [float(v) for v in q], gripper_position=float(grip))
    res = step(RobotAction(arms={"primary": cmd}))
    tick["n"] = tick["n"] + 1
    term = False
    trunc = False
    try:
        term = bool(res.terminated)
    except Exception:
        term = False
    try:
        trunc = bool(res.truncated)
    except Exception:
        trunc = False
    if term or trunc:
        tick["done"] = True
        tick["term"] = {"terminated": term, "truncated": trunc, "tick": tick["n"]}
    return res


def current_q():
    return [float(v) for v in get_robot_state().joint_positions["primary"]]


def servo(q_target, ticks, grip, settle=0, scan_every=0):
    q0 = current_q()
    for k in range(1, ticks + 1):
        if tick["done"]:
            return False
        a = float(k) / float(ticks)
        q = [q0[j] + a * (q_target[j] - q0[j]) for j in range(7)]
        send(q, grip)
        if scan_every > 0 and k % scan_every == 0 and not tick["done"]:
            scan("servo")
    for k in range(settle):
        if tick["done"]:
            return False
        send(q_target, grip)
    return not tick["done"]


def qmul(a, b):
    return [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]


def yaw_quat(phi):
    return qmul([math.cos(phi / 2.0), 0.0, 0.0, math.sin(phi / 2.0)], down_q)


def ik(pos, seed=None, quat=None):
    if seed is None:
        seed = current_q()
    if quat is None:
        quat = down_q
    res = solve_ik(Pose((float(pos[0]), float(pos[1]), float(pos[2])), quat, base_frame), seed_joints=seed)
    if not res.ok:
        return None
    return [float(v) for v in res.joint_positions]


def ee_pos():
    return [round(float(v), 4) for v in get_robot_state().end_effector_poses["primary"].position]


def log(name, extra=None):
    entry = {"ev": name, "tick": tick["n"]}
    try:
        entry["ee"] = ee_pos()
    except Exception:
        pass
    if extra is not None:
        entry["x"] = extra
    report["events"].append(entry)


# ---------------- geometry ----------------
def quat_to_rot(qw, qx, qy, qz):
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ]


def cam_model(cam):
    K = cam.intrinsics
    pose = cam.camera_pose
    q = [float(v) for v in pose.quaternion_wxyz]
    return {
        "fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]), "cy": float(K[1][2]),
        "R": quat_to_rot(q[0], q[1], q[2], q[3]),
        "t": [float(v) for v in pose.position],
    }


def pix_to_base(model, u, v, d):
    xc = (u - model["cx"]) / model["fx"] * d
    yc = (v - model["cy"]) / model["fy"] * d
    zc = d
    R = model["R"]
    t = model["t"]
    return [R[i][0] * xc + R[i][1] * yc + R[i][2] * zc + t[i] for i in range(3)]


def classify(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = mx - mn
    if mx < 50 or d < 0.45 * mx:
        return None
    if mx == r:
        hue = (60.0 * (g - b) / d) % 360.0
    elif mx == g:
        hue = 60.0 * (b - r) / d + 120.0
    else:
        hue = 60.0 * (r - g) / d + 240.0
    if hue >= 340.0 or hue < 12.0:
        return "red"
    if hue < 28.0:
        return None  # wooden table band
    if hue < 50.0:
        return "orange"
    if hue < 75.0:
        return "yellow"
    if hue < 165.0:
        return "green"
    if hue < 200.0:
        return "cyan"
    if hue < 255.0:
        return "blue"
    if hue < 320.0:
        return "purple"
    return "pink"


memory = {}
fixtures = []
seen_any = {"v": False, "last_tick": -1, "first_tick": -1}


def scan_camera(obs, cam_name):
    cam = obs.cameras[cam_name]
    try:
        rgb = cam.rgb.tolist()
        dep = cam.depth_m.tolist()
    except Exception:
        rgb = cam.rgb
        dep = cam.depth_m
    model = cam_model(cam)
    h = len(rgb)
    w = len(rgb[0])
    clusters = []
    for v in range(0, h, 2):
        row = rgb[v]
        drow = dep[v]
        for u in range(0, w, 2):
            px = row[u]
            c = classify(int(px[0]), int(px[1]), int(px[2]))
            if c is None:
                continue
            d = float(drow[u])
            if d <= 0.12 or d > 3.0:
                continue
            p = pix_to_base(model, u, v, d)
            if p[2] > 0.12 or p[2] < -0.02:
                continue
            placed = False
            for cl in clusters:
                if cl["c"] == c and abs(cl["su"] / cl["n"] - u) < 12 and abs(cl["sv"] / cl["n"] - v) < 12:
                    cl["su"] += u
                    cl["sv"] += v
                    cl["sx"] += p[0]
                    cl["sy"] += p[1]
                    cl["sz"] += p[2]
                    cl["zmax"] = max(cl["zmax"], p[2])
                    cl["n"] += 1
                    placed = True
                    break
            if not placed:
                clusters.append({"c": c, "su": u, "sv": v, "sx": p[0], "sy": p[1], "sz": p[2], "zmax": p[2], "n": 1})
    found = []
    for cl in clusters:
        if cl["n"] < 5 or cl["zmax"] < 0.01:
            continue
        n = float(cl["n"])
        found.append({
            "c": cl["c"], "n": cl["n"], "cam": cam_name,
            "uv": [round(cl["su"] / n, 1), round(cl["sv"] / n, 1)],
            "xyz": [round(cl["sx"] / n, 4), round(cl["sy"] / n, 4), round(cl["sz"] / n, 4)],
            "zmax": round(cl["zmax"], 4),
        })
    return found


def near_xy(a, b, r):
    return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) < r * r


def is_fixture(f):
    for fx in fixtures:
        if fx["c"] == f["c"] and near_xy(fx["xyz"], f["xyz"], 0.05):
            return True
    if button is not None and near_xy([button["x"], button["y"]], f["xyz"], 0.045):
        return True
    return False


def scan(tag, learn=True):
    obs = get_observation()
    found = []
    for cam_name in ("front", "wrist"):
        try:
            found = found + scan_camera(obs, cam_name)
        except Exception as exc:
            report["events"].append({"ev": "scan_error", "cam": cam_name, "err": str(exc)})
    if tag == "start":
        for f in found:
            fixtures.append(f)
        report["fixtures"] = found
        return []
    found = [f for f in found if not is_fixture(f)]
    summary = sorted([(f["c"], f["cam"][0], f["n"], f["xyz"][0], f["xyz"][1]) for f in found])
    sig = [(x[0], x[1], x[2] // 6) for x in summary]
    lastsig = report["scans"][-1]["sig"] if report["scans"] else None
    if sig != lastsig and len(report["scans"]) < 120:
        report["scans"].append({"tick": tick["n"], "tag": tag, "found": summary, "sig": sig})
    if found:
        if not seen_any["v"]:
            seen_any["first_tick"] = tick["n"]
        seen_any["v"] = True
        seen_any["last_tick"] = tick["n"]
    if not learn:
        return found
    for f in found:
        objs = memory.get(f["c"])
        if objs is None:
            objs = []
            memory[f["c"]] = objs
        match = None
        for o in objs:
            if near_xy(o["xyz"], f["xyz"], 0.04):
                match = o
                break
        if match is None:
            objs.append({"xyz": f["xyz"], "n": f["n"], "cam": f["cam"], "first": tick["n"], "last": tick["n"], "hits": 1, "zmax": f["zmax"]})
        else:
            match["last"] = tick["n"]
            match["hits"] += 1
            if f["n"] > match["n"] or (f["cam"] == "front" and match["cam"] != "front" and f["n"] >= 10):
                match["xyz"] = f["xyz"]
                match["n"] = f["n"]
                match["cam"] = f["cam"]
                match["zmax"] = f["zmax"]
    return found


def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


# ---------------- 1. button ----------------
button = None
best = None
for bcam in ("front", "wrist"):
    seg = segment_text(bcam, "button")
    if not seg.ok:
        continue
    for cand in seg.segmentations[:6]:
        try:
            cloud = mask_to_point_cloud(cand, bcam, target_frame=base_frame)
            pts = cloud.points
            if len(pts) < 10:
                continue
            zs = [float(p[2]) for p in pts]
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            span = max(max(xs) - min(xs), max(ys) - min(ys))
            if span > 0.12 or quantile(zs, 0.5) > 0.15 or quantile(xs, 0.5) < 0.15:
                continue
            score = float(cand.score)
            if best is None or score > best["score"]:
                best = {"score": score, "x": quantile(xs, 0.5), "y": quantile(ys, 0.5),
                        "top": quantile(zs, 0.95), "n": len(pts)}
        except Exception as exc:
            continue
button = best
report["button"] = button
scan("start")

pressed = False
if button is not None:
    above = ik((button["x"], button["y"], button["top"] + 0.06), home_q)
    press = ik((button["x"], button["y"], button["top"] - 0.02), above if above is not None else home_q)
    report["ik_button"] = {"above": above is not None, "press": press is not None}
    if above is not None and press is not None:
        servo(above, 28, 0.0, settle=2, scan_every=4)
        log("above_button")
        servo(press, 10, 0.0, settle=6)
        log("pressed_bottom")
        servo(above, 8, 0.0, settle=1)
        log("button_released")
        pressed = True
report["pressed"] = pressed

# ---------------- 2. watch ----------------
servo(home_q, 20, 0.0, settle=2, scan_every=2)
log("home_after_press")
hidden_streak = 0
watch_ticks = 0
while watch_ticks < 260 and not tick["done"]:
    send(home_q, 0.0)
    watch_ticks += 1
    found = scan("watch")
    visible_targets = [f for f in found if f["c"] in order]
    if all([c in memory for c in order]) or watch_ticks > 150:
        if visible_targets:
            hidden_streak = 0
        else:
            hidden_streak += 1
        if hidden_streak >= 20:
            break
log("watch_end", {"watch_ticks": watch_ticks, "seen": seen_any})
report["memory"] = memory

# ---------------- 3. unmask in order ----------------
def base_to_pix(model, p):
    R = model["R"]
    t = model["t"]
    d = [p[0] - t[0], p[1] - t[1], p[2] - t[2]]
    xc = R[0][0] * d[0] + R[1][0] * d[1] + R[2][0] * d[2]
    yc = R[0][1] * d[0] + R[1][1] * d[1] + R[2][1] * d[2]
    zc = R[0][2] * d[0] + R[1][2] * d[1] + R[2][2] * d[2]
    if zc <= 0.01:
        return None
    return [model["fx"] * xc / zc + model["cx"], model["fy"] * yc / zc + model["cy"]]


def fit_rect(pts):
    """Minimum-area rectangle (3-degree search, 3/97 % quantiles) over XY of points."""
    best = None
    for k in range(30):
        th = math.radians(3.0 * k)
        c = math.cos(th)
        sn = math.sin(th)
        a = [p[0] * c + p[1] * sn for p in pts]
        b = [-p[0] * sn + p[1] * c for p in pts]
        alo = quantile(a, 0.03)
        ahi = quantile(a, 0.97)
        blo = quantile(b, 0.03)
        bhi = quantile(b, 0.97)
        area = (ahi - alo) * (bhi - blo)
        if best is None or area < best["area"]:
            am = 0.5 * (alo + ahi)
            bm = 0.5 * (blo + bhi)
            best = {"area": area, "th": th, "da": ahi - alo, "db": bhi - blo,
                    "cx": am * c - bm * sn, "cy": am * sn + bm * c}
    return best


def container_from_cloud(cloud, gx, gy, radius):
    pts = []
    for p in cloud.points:
        x = float(p[0])
        y = float(p[1])
        z = float(p[2])
        if z > 0.012 and (x - gx) * (x - gx) + (y - gy) * (y - gy) < radius * radius:
            pts.append([x, y, z])
    if len(pts) < 30:
        return None
    rect = fit_rect(pts)
    rect["top"] = quantile([p[2] for p in pts], 0.97)
    rect["n"] = len(pts)
    return rect


def view_clear(cam, p):
    model = cam_model(cam)
    uv = base_to_pix(model, p)
    if uv is None:
        return None
    u = int(round(uv[0]))
    v = int(round(uv[1]))
    if u < 2 or v < 2 or u > 253 or v > 253:
        return None
    R = model["R"]
    t = model["t"]
    d = [p[0] - t[0], p[1] - t[1], p[2] - t[2]]
    zc = R[0][2] * d[0] + R[1][2] * d[1] + R[2][2] * d[2]
    seen = float(cam.depth_m[v][u])
    return {"uv": [uv[0], uv[1]], "clear": seen > zc - 0.07, "seen": round(seen, 3), "expect": round(zc, 3)}


def locate_container(gx, gy, gz):
    obs = get_observation()
    target = [gx, gy, gz + 0.02]
    tried = []
    for cam_name in ("front", "wrist"):
        vc = view_clear(obs.cameras[cam_name], target)
        tried.append({"cam": cam_name, "view": vc})
        if vc is None or not vc["clear"]:
            continue
        found = []
        try:
            segp = segment_points(cam_name, [[vc["uv"][0], vc["uv"][1]]])
            if segp.ok:
                for cand in segp.segmentations[:3]:
                    cloud = mask_to_point_cloud(cand, cam_name, target_frame=base_frame)
                    rect = container_from_cloud(cloud, gx, gy, 0.06)
                    if rect is not None:
                        rect["src"] = cam_name
                        found.append(rect)
        except Exception as exc:
            tried.append({"cam": cam_name, "err": str(exc)})
        best = None
        for rect in found:
            if min(rect["da"], rect["db"]) > 0.075 or max(rect["da"], rect["db"]) > 0.11:
                continue
            if min(rect["da"], rect["db"]) < 0.03 or rect["top"] > 0.15 or rect["top"] < 0.03:
                continue
            dist = math.sqrt((rect["cx"] - gx) ** 2 + (rect["cy"] - gy) ** 2)
            rect["dist"] = dist
            if dist > 0.04:
                continue
            if best is None or dist < best["dist"]:
                best = rect
        if best is not None:
            best["tried"] = len(tried)
            return best, tried
    return None, tried


def gripper_yaw(rect):
    th = rect["th"]
    phi = th if rect["db"] <= rect["da"] else th + math.pi / 2.0
    square = abs(rect["da"] - rect["db"]) < 0.012
    options = [phi + k * math.pi / 2.0 for k in range(-4, 5)] if square else [phi + k * math.pi for k in range(-2, 3)]
    best = None
    for o in options:
        if o > -math.pi / 2.0 - 1e-6 and o <= math.pi / 2.0 + 1e-6:
            if best is None or abs(o) < abs(best):
                best = o
    return best


picks = []
targets = []
for goal in goals:
    objs = memory.get(goal["color"])
    if objs is None or len(objs) == 0:
        targets.append({"color": goal["color"], "obj": None})
        continue
    ranked = sorted(objs, key=lambda o: -(o["hits"] * 1000 + o["n"]))
    chosen = ranked if goal["all"] else ranked[:1]
    for o in chosen:
        targets.append({"color": goal["color"], "obj": o})
report["targets"] = targets
for tgt in targets:
    if tick["done"]:
        break
    color = tgt["color"]
    info = {"color": color, "attempts": []}
    item = tgt["obj"]
    if item is None:
        info["error"] = "colour never seen"
        picks.append(info)
        continue
    info["remembered"] = item
    gx = item["xyz"][0]
    gy = item["xyz"][1]
    gz = item["xyz"][2]
    for attempt in range(2):
        if tick["done"]:
            break
        att = {}
        servo(home_q, 20, 1.0, settle=2)
        if tick["done"]:
            break
        rect, tried = locate_container(gx, gy, gz)
        att["tried"] = tried
        if rect is None:
            att["error"] = "container not localized"
            cx = gx
            cy = gy
            top = item["zmax"] + 0.035
            phi = 0.0
        else:
            att["rect"] = {k: (round(v, 4) if k not in ("src", "tried") else v) for k, v in rect.items()}
            cx = rect["cx"]
            cy = rect["cy"]
            top = rect["top"]
            phi = gripper_yaw(rect)
        quat = yaw_quat(phi)
        att["phi_deg"] = round(math.degrees(phi), 1)
        grasp_z = max(0.5 * top, 0.02)
        q_pre = ik((cx, cy, top + 0.07), None, quat)
        q_grasp = ik((cx, cy, grasp_z), q_pre, quat)
        q_lift = ik((cx, cy, top + 0.12), q_grasp, quat)
        if q_pre is None or q_grasp is None or q_lift is None:
            att["error"] = "ik failed"
            info["attempts"].append(att)
            continue
        servo(q_pre, 25, 1.0, settle=4)
        servo(q_grasp, 14, 1.0, settle=5)
        log("at_grasp_" + color)
        for k in range(18):
            if tick["done"]:
                break
            send(q_grasp, 0.0)
        if tick["done"]:
            info["attempts"].append(att)
            break
        att["grip_closed"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
        servo(q_lift, 20, 0.0, settle=6)
        log("lifted_" + color)
        if tick["done"]:
            info["attempts"].append(att)
            break
        grip = float(get_robot_state().gripper_positions["primary"])
        att["grip_lifted"] = round(grip, 3)
        att["scan_lifted"] = [f for f in scan("lifted_" + color, False) if f["c"] == color]
        held = grip > 0.03
        att["held"] = held
        q_down = ik((cx, cy, grasp_z + 0.008), q_lift, quat)
        if held and q_down is not None:
            servo(q_down, 18, 0.0, settle=3)
        for k in range(8):
            if tick["done"]:
                break
            send(current_q(), 1.0)
        servo(q_pre, 14, 1.0, settle=2)
        log("released_" + color)
        info["attempts"].append(att)
        if held:
            break
    picks.append(info)

report["picks"] = picks
report["term"] = tick["term"]
report["ticks"] = tick["n"]
try:
    report["final_meta"] = dict(get_task_context().metadata)
except Exception:
    pass
result = report
