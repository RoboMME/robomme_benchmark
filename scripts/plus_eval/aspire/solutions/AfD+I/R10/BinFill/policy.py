# BinFill v03 -- v02 + jaw yaw aligned with the measured cube faces (top-face min-area orientation),
# scarcity-first / downstream-aware target choice against a running belt-end estimate.
# Closed loop: parse colour counts from the goal, repeatedly localize a cube of a still-needed
# colour on the front camera (hue-verified), estimate belt motion from timestamps, pick it with a
# predicted top-down grasp (wrist re-localization at hover), drop it into the localized bin,
# count it from the program's own memory (cubes inside the bin are not visible), then press the
# localized button. Every scene quantity is derived at runtime from public perception.
import math

report = {"version": "v03", "events": [], "picks": []}


def note(kind, data=None):
    entry = {"k": kind}
    if data is not None:
        for key in data:
            entry[key] = data[key]
    if len(report["events"]) < 400:
        report["events"].append(entry)


def now_s():
    return float(get_robot_state().timestamp_s)


# ---------------------------------------------------------------- goal parsing
NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
COLOR_WORDS = ["red", "green", "blue", "yellow", "orange", "cyan", "magenta", "purple", "pink"]


def parse_counts(text):
    cleaned = text.lower()
    for ch in ",.;:":
        cleaned = cleaned.replace(ch, " ")
    tokens = [tok for tok in cleaned.split(" ") if tok]
    counts = {}
    for index in range(1, len(tokens)):
        word = tokens[index]
        if word in COLOR_WORDS:
            prev = tokens[index - 1]
            number = None
            if prev in NUMBER_WORDS:
                number = NUMBER_WORDS[prev]
            elif prev.isdigit():
                number = int(prev)
            if number is not None:
                counts[word] = counts.get(word, 0) + number
    return counts


# ---------------------------------------------------------------- colour
def hue_of(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx <= 0:
        return None, 0.0, 0.0
    sat = (mx - mn) / mx
    val = mx / 255.0
    if mx == mn:
        return None, sat, val
    if mx == r:
        h = 60.0 * (((g - b) / (mx - mn)) % 6.0)
    elif mx == g:
        h = 60.0 * ((b - r) / (mx - mn) + 2.0)
    else:
        h = 60.0 * ((r - g) / (mx - mn) + 4.0)
    return h, sat, val


def color_name(h):
    # Bands measured on released demonstrations (cube pixels have saturation > 0.85;
    # the wooden table is hue 10-30 with saturation ~0.6 and is excluded by saturation).
    if h is None:
        return None
    if h >= 335.0 or h < 15.0:
        return "red"
    if 30.0 <= h < 50.0:
        return "orange"
    if 50.0 <= h < 78.0:
        return "yellow"
    if 85.0 <= h < 150.0:
        return "green"
    if 160.0 <= h < 205.0:
        return "cyan"
    if 210.0 <= h < 268.0:
        return "blue"
    if 275.0 <= h < 332.0:
        return "magenta"
    return None


def cand_mask(cand):
    try:
        return cand.mask
    except Exception:
        return None


def mask_color_votes(rgb, mask, box):
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(len(rgb[0]) - 1, int(box[2]) + 1)
    y2 = min(len(rgb) - 1, int(box[3]) + 1)
    votes = {}
    total = 0
    stride = 1
    if (x2 - x1) * (y2 - y1) > 1600:
        stride = 2
    for y in range(y1, y2, stride):
        for x in range(x1, x2, stride):
            if mask is not None:
                if not bool(mask[y][x]):
                    continue
            elif (x - x1) * 4 < (x2 - x1) or (x2 - x) * 4 < (x2 - x1) or (y - y1) * 4 < (y2 - y1) or (y2 - y) * 4 < (y2 - y1):
                continue
            px = rgb[y][x]
            h, s, v = hue_of(float(px[0]), float(px[1]), float(px[2]))
            total += 1
            if s < 0.75 or v < 0.12:
                continue
            name = color_name(h)
            if name is not None:
                votes[name] = votes.get(name, 0) + 1
    return votes, total


def dominant(votes, total):
    best = None
    best_n = 0
    for key in votes:
        if votes[key] > best_n:
            best_n = votes[key]
            best = key
    frac = 0.0
    if total > 0:
        frac = best_n / float(total)
    return best, best_n, frac


# ---------------------------------------------------------------- geometry
def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def cloud_summary(points):
    xs = []
    ys = []
    zs = []
    n = len(points)
    stride = 1
    if n > 1500:
        stride = n // 1500 + 1
    for index in range(0, n, stride):
        p = points[index]
        xs.append(float(p[0]))
        ys.append(float(p[1]))
        zs.append(float(p[2]))
    if len(xs) < 5:
        return None
    ztop = quantile(zs, 0.92)
    # Top-face centre: the side faces a camera sees bias the plain median toward that camera.
    txs = [xs[i] for i in range(len(xs)) if zs[i] >= ztop - 0.006]
    tys = [ys[i] for i in range(len(ys)) if zs[i] >= ztop - 0.006]
    tx = quantile(xs, 0.5)
    ty = quantile(ys, 0.5)
    if len(txs) >= 5:
        tx = quantile(txs, 0.5)
        ty = quantile(tys, 0.5)
    # Cube yaw (mod 90 deg) from the top face: orientation of the minimum-area bounding box.
    cyaw = None
    if len(txs) >= 20:
        best_area = None
        best_deg = 0
        degs = list(range(0, 90, 6))
        for stage in range(2):
            for step_deg in degs:
                ang = step_deg * math.pi / 180.0
                ca = math.cos(ang)
                sa = math.sin(ang)
                us = [txs[i] * ca + tys[i] * sa for i in range(len(txs))]
                vs = [-txs[i] * sa + tys[i] * ca for i in range(len(txs))]
                area = (quantile(us, 0.98) - quantile(us, 0.02)) * (quantile(vs, 0.98) - quantile(vs, 0.02))
                if best_area is None or area < best_area:
                    best_area = area
                    best_deg = step_deg
            degs = [best_deg + k for k in range(-5, 6) if k != 0]
        cyaw = (best_deg % 90) * math.pi / 180.0
    return {
        "x": quantile(xs, 0.5), "y": quantile(ys, 0.5), "tx": tx, "ty": ty, "cyaw": cyaw,
        "ztop": quantile(zs, 0.92), "zlow": quantile(zs, 0.05),
        "xspan": quantile(xs, 0.95) - quantile(xs, 0.05),
        "yspan": quantile(ys, 0.95) - quantile(ys, 0.05),
        "n": len(points),
    }


def project(cand, camera):
    base = get_robot_state().base_frame
    cloud = None
    mask = cand_mask(cand)
    if mask is not None:
        try:
            cloud = mask_to_point_cloud(mask, camera, target_frame=base)
        except Exception as exc:
            note("project_error", {"camera": camera, "err": str(exc)[:120]})
    if cloud is None:
        try:
            cloud = mask_to_point_cloud(cand, camera, target_frame=base)
        except Exception as exc:
            note("project_error2", {"camera": camera, "err": str(exc)[:120]})
            return None
    try:
        pts = cloud.points
    except Exception:
        return None
    if pts is None or len(pts) < 8:
        return None
    return cloud_summary(pts)


def detect_cubes(camera, colors, prompt_suffix="cube"):
    """Return hue-verified cube detections [{color,x,y,ztop,zlow,score}] on one camera."""
    obs = get_observation()
    rgb = obs.cameras[camera].rgb
    stamp = float(obs.timestamp_s)
    found = []
    seen_boxes = []
    for color in colors:
        seg = segment_text(camera, color + " " + prompt_suffix)
        if not seg.ok:
            continue
        for cand in seg.segmentations[:12]:
            if cand.box_xyxy is None:
                continue
            box = [float(v) for v in cand.box_xyxy]
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            if bw < 2 or bh < 2:
                continue
            if camera == "front" and (bw > 45 or bh > 45):
                continue
            duplicate = False
            for other in seen_boxes:
                if abs(other[0] - box[0]) < 3 and abs(other[1] - box[1]) < 3 and abs(other[2] - box[2]) < 3:
                    duplicate = True
            if duplicate:
                continue
            votes, total = mask_color_votes(rgb, cand_mask(cand), box)
            name, count, frac = dominant(votes, total)
            if name != color or frac < 0.3:
                continue
            geo = project(cand, camera)
            if geo is None:
                continue
            if geo["xspan"] > 0.09 or geo["yspan"] > 0.09:
                continue
            seen_boxes.append(box)
            found.append({"color": color, "x": geo["tx"], "y": geo["ty"], "ztop": geo["ztop"],
                          "zlow": geo["zlow"], "cyaw": geo["cyaw"], "score": round(float(cand.score), 3),
                          "box": [round(v, 1) for v in box], "frac": round(frac, 2), "t": stamp})
    return found, stamp


def locate_fixture(prompts, max_span=0.30):
    for prompt in prompts:
        seg = segment_text("front", prompt)
        if not seg.ok:
            note("fixture_miss", {"prompt": prompt})
            continue
        for cand in seg.segmentations[:5]:
            geo = project(cand, "front")
            if geo is None:
                continue
            if geo["xspan"] > max_span or geo["yspan"] > max_span:
                continue
            geo["prompt"] = prompt
            geo["score"] = round(float(cand.score), 3)
            geo["box"] = [round(float(v), 1) for v in cand.box_xyxy]
            return geo
    return None


# ---------------------------------------------------------------- quaternions
def qmul(a, b):
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]


def qrot(q, v):
    p = [0.0, v[0], v[1], v[2]]
    qc = [q[0], -q[1], -q[2], -q[3]]
    r = qmul(qmul(q, p), qc)
    return [r[1], r[2], r[3]]


def yaw_about_z(q, angle):
    rz = [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]
    return qmul(rz, q)


# ---------------------------------------------------------------- motion helpers
def timed(label, fn):
    t_start = now_s()
    res = fn()
    try:
        t_end = now_s()
    except Exception:
        t_end = t_start
        report["ended_during"] = label
    entry = {"label": label, "dt": round(t_end - t_start, 3)}
    try:
        entry["ok"] = bool(res.ok)
        entry["steps"] = int(res.steps_executed)
        entry["status"] = str(res.status)
    except Exception:
        pass
    try:
        if res.terminated or res.truncated:
            entry["terminal"] = True
            report["terminal"] = label
    except Exception:
        pass
    note("move", entry)
    return res, t_end - t_start


def terminal(res):
    try:
        return bool(res.terminated) or bool(res.truncated)
    except Exception:
        return False


def go_pose(label, xyz, quat, tol=0.01, steps=150):
    base = get_robot_state().base_frame
    target = Pose((float(xyz[0]), float(xyz[1]), float(xyz[2])), (quat[0], quat[1], quat[2], quat[3]), base)
    return timed(label, lambda: move_to_pose(target, tolerance=tol, max_steps=steps))


def hold_ticks(count):
    state = get_robot_state()
    last = None
    for tick in range(count):
        last = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=state.joint_positions["primary"],
            gripper_position=state.gripper_positions["primary"],
            embodiment=state.embodiment)}))
        if terminal(last):
            report["terminal"] = "hold"
            return last
    return last


# ---------------------------------------------------------------- main
context = get_task_context()
goal = context.language
report["goal"] = goal
needed = parse_counts(goal)
report["needed"] = dict(needed)
max_steps = None
try:
    meta = robomme_plus.get_task_metadata()
    max_steps = int(meta["max_steps"])
    report["max_steps"] = max_steps
except Exception as exc:
    report["meta_error"] = str(exc)[:200]
try:
    video = robomme_plus.get_conditioning_video()
    report["video_len"] = len(video)
except Exception as exc:
    report["video_error"] = str(exc)[:200]

hz = 20.0
horizon_s = 1e9
if max_steps is not None:
    horizon_s = max_steps / hz
PRESS_RESERVE_S = 4.0


def safe_now():
    try:
        return now_s()
    except Exception:
        return None


state0 = get_robot_state()
home_pose = state0.end_effector_poses["primary"]
home_q = [float(v) for v in home_pose.quaternion_wxyz]
home_finger = qrot(home_q, [0.0, 1.0, 0.0])
report["home_ee"] = [round(float(v), 4) for v in home_pose.position] + [round(v, 4) for v in home_q]

bin_geo = locate_fixture(["bin", "open box", "box", "container"])
button_geo = locate_fixture(["button", "white button", "push button"], max_span=0.15)
report["bin"] = bin_geo
report["button"] = button_geo

# All cube colours are scanned: needed ones are targets, the rest are obstacles for the jaws.
ALL_COLORS = ["red", "green", "blue", "yellow", "orange", "cyan", "magenta"]
scan_colors = [c for c in needed if needed[c] > 0]
for c in ALL_COLORS:
    if c not in scan_colors:
        scan_colors.append(c)

vel = [0.0, 0.0]
vel_samples = []
VEL_SAMPLES_WANTED = 8
TIP_OFFSET = 0.048     # fingertip centre distance from grasp centre with the jaw open (Panda 8 cm)
CLEAR_OK = 0.04        # neighbour-centre to fingertip-centre distance considered free


def match_velocity(first, t_a, second, t_b):
    dt = t_b - t_a
    if dt <= 0.05:
        return 0
    added = 0
    for a in first:
        px = a["x"] + vel[0] * dt
        py = a["y"] + vel[1] * dt
        best = None
        best_d = 0.03
        for b in second:
            if b["color"] != a["color"]:
                continue
            d = math.sqrt((px - b["x"]) ** 2 + (py - b["y"]) ** 2)
            if d < best_d:
                best_d = d
                best = b
        if best is not None:
            vel_samples.append([(best["x"] - a["x"]) / dt, (best["y"] - a["y"]) / dt])
            added += 1
    return added


def refresh_velocity():
    if vel_samples:
        vel[0] = quantile([s[0] for s in vel_samples], 0.5)
        vel[1] = quantile([s[1] for s in vel_samples], 0.5)


def finger_axis(yaw):
    return [home_finger[0] * math.cos(yaw) - home_finger[1] * math.sin(yaw),
            home_finger[0] * math.sin(yaw) + home_finger[1] * math.cos(yaw)]


def clearance(target, others, yaw):
    a = finger_axis(yaw)
    worst = 1.0
    for o in others:
        if o is target:
            continue
        if abs(o["x"] - target["x"]) < 0.012 and abs(o["y"] - target["y"]) < 0.012:
            continue
        for sign in (1.0, -1.0):
            tx = target["x"] + sign * TIP_OFFSET * a[0]
            ty = target["y"] + sign * TIP_OFFSET * a[1]
            d = math.sqrt((o["x"] - tx) ** 2 + (o["y"] - ty) ** 2)
            if d < worst:
                worst = d
    return worst


def wrap_half(angle):
    while angle > math.pi / 2:
        angle -= math.pi
    while angle <= -math.pi / 2:
        angle += math.pi
    return angle


def face_yaws(cyaw, prefer):
    """Two jaw yaws (relative to home) whose finger axis is normal to a cube face; nearest to prefer first."""
    base = wrap_half(cyaw - math.atan2(home_finger[1], home_finger[0]))
    other = wrap_half(base + math.pi / 2.0)
    if abs(wrap_half(base - prefer)) <= abs(wrap_half(other - prefer)):
        return [base, other]
    return [other, base]


s_end = None
placed = {}
for c in needed:
    placed[c] = 0
t_descend = 0.5
t_close = 0.3
t_hover = 1.0
attempts = 0
waits = 0
ended = False
report["scans"] = []

while not ended and attempts < 30:
    remaining = [c for c in needed if placed[c] < needed[c]]
    if not remaining:
        break
    t_now = now_s()
    if t_now > horizon_s - PRESS_RESERVE_S:
        note("out_of_time", {"t": t_now})
        break
    if bin_geo is None:
        report["abort"] = "bin not localized"
        break
    detections, t_det = detect_cubes("front", scan_colors)
    if len(vel_samples) < VEL_SAMPLES_WANTED and detections:
        res_hold = hold_ticks(8)
        if terminal(res_hold):
            ended = True
            break
        second, t_second = detect_cubes("front", scan_colors)
        n_new = match_velocity(detections, t_det, second, t_second)
        refresh_velocity()
        detections, t_det = second, t_second
    speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
    if len(report["scans"]) < 40:
        report["scans"].append({"t": round(t_det, 2), "n": len(detections),
                                "needed_n": len([d for d in detections if d["color"] in remaining]),
                                "vel": [round(vel[0], 4), round(vel[1], 4)], "vs": len(vel_samples)})
    pool = [d for d in detections if d["color"] in remaining]
    vh = [0.0, 0.0]
    if speed > 0.005:
        vh = [vel[0] / speed, vel[1] / speed]
        for d in detections:
            s_d = d["x"] * vh[0] + d["y"] * vh[1]
            if s_end is None or s_d > s_end:
                s_end = s_d
    if speed > 0.005 and pool and s_end is not None:
        # Reachable: predicted position at the close stays clear of the belt end seen so far.
        lead_s = (now_s() - t_det) + t_hover + t_descend + t_close + 0.4
        ok_pool = [d for d in pool if (d["x"] * vh[0] + d["y"] * vh[1]) + speed * lead_s < s_end - 0.03]
        if ok_pool:
            pool = ok_pool
    visible = {}
    for d in detections:
        visible[d["color"]] = visible.get(d["color"], 0) + 1
    if not pool:
        waits += 1
        res_hold = hold_ticks(15)
        if terminal(res_hold):
            ended = True
            break
        continue
    attempts += 1
    # Choose target and jaw yaw: free fingertip landing first, then nearest to the bin.
    along = 0.0
    if speed > 0.005:
        along = wrap_half(math.atan2(vel[1], vel[0]) - math.atan2(home_finger[1], home_finger[0]))
    best = None
    best_key = None
    for d in pool:
        dist_bin = math.sqrt((d["x"] - bin_geo["x"]) ** 2 + (d["y"] - bin_geo["y"]) ** 2)
        urgent = 1 if visible.get(d["color"], 0) <= needed[d["color"]] - placed[d["color"]] else 0
        s_d = d["x"] * vh[0] + d["y"] * vh[1]
        yaw_options = [along, wrap_half(along + math.pi / 2.0)]
        if d["cyaw"] is not None:
            yaw_options = face_yaws(d["cyaw"], along)
        for index in range(len(yaw_options)):
            yaw = yaw_options[index]
            clear = clearance(d, detections, yaw)
            free = clear >= CLEAR_OK
            rank = -dist_bin - 0.02 * index
            if urgent:
                rank = s_d - 0.02 * index
            key = (1 if free else 0, urgent, rank if free else clear)
            if best_key is None or key > best_key:
                best_key = key
                best = (d, yaw, clear)
    target, yaw, clear = best
    grasp_q = yaw_about_z(home_q, yaw)
    pick = {"color": target["color"], "urgent": visible.get(target["color"], 0) <= needed[target["color"]] - placed[target["color"]], "obs": [round(target["x"], 3), round(target["y"], 3), round(target["ztop"], 3)],
            "yaw_deg": round(yaw * 180.0 / math.pi, 1), "clear": round(clear, 3), "t_obs": round(t_det, 2)}
    size = max(0.02, min(0.06, target["ztop"] - target["zlow"]))
    hover_z = target["ztop"] + 0.10
    dt_pred = (now_s() - t_det) + t_hover
    hx = target["x"] + vel[0] * dt_pred
    hy = target["y"] + vel[1] * dt_pred
    res, dt = go_pose("hover", (hx, hy, hover_z), grasp_q, tol=0.01, steps=120)
    if terminal(res):
        ended = True
        break
    t_hover = min(1.5, max(0.3, 0.5 * t_hover + 0.5 * dt))
    wrist, t_w = detect_cubes("wrist", scan_colors)
    px = target["x"] + vel[0] * (t_w - t_det)
    py = target["y"] + vel[1] * (t_w - t_det)
    chosen = None
    chosen_d = 0.05
    for d in wrist:
        if d["color"] != target["color"]:
            continue
        dd = math.sqrt((d["x"] - px) ** 2 + (d["y"] - py) ** 2)
        if dd < chosen_d:
            chosen_d = dd
            chosen = d
    if chosen is not None:
        cx, cy, ctop = chosen["x"], chosen["y"], chosen["ztop"]
        pick["wrist"] = [round(cx, 3), round(cy, 3), round(ctop, 3), round(chosen_d, 3)]
    else:
        cx, cy, ctop = px, py, target["ztop"]
        pick["wrist"] = None
    # Re-choose the jaw yaw from the close-range wrist view of the neighbours.
    if chosen is not None:
        w_options = [yaw, wrap_half(yaw + math.pi / 2.0)]
        if chosen["cyaw"] is not None:
            w_options = face_yaws(chosen["cyaw"], along)
            pick["cube_yaw_deg"] = round(chosen["cyaw"] * 180.0 / math.pi, 1)
        wbest = None
        for yaw_try in w_options:
            c_try = clearance(chosen, wrist, yaw_try)
            if wbest is None or (wbest[1] < CLEAR_OK and c_try > wbest[1] + 0.005):
                wbest = (yaw_try, c_try)
        pick["wrist_clear"] = round(wbest[1], 3)
        if wbest[1] < CLEAR_OK:
            pick["skipped"] = "wrist clearance"
            report["picks"].append(pick)
            continue
        if wbest[0] != yaw:
            yaw = wbest[0]
            grasp_q = yaw_about_z(home_q, yaw)
            pick["yaw_deg"] = round(yaw * 180.0 / math.pi, 1)
    grasp_z = min(ctop - 0.5 * size, ctop - 0.015)
    grasp_z = max(grasp_z, target["zlow"] + 0.008)
    dt_g = (now_s() - t_w) + t_descend + 0.5 * t_close
    gx = cx + vel[0] * dt_g
    gy = cy + vel[1] * dt_g
    pick["grasp"] = [round(gx, 3), round(gy, 3), round(grasp_z, 3)]
    res, dt = go_pose("descend", (gx, gy, grasp_z), grasp_q, tol=0.008, steps=40)
    if terminal(res):
        ended = True
        break
    ee_now = get_robot_state().end_effector_poses["primary"].position
    pick["ee_after_descend"] = [round(float(v), 3) for v in ee_now]
    if float(ee_now[2]) > grasp_z + 0.012:
        # A finger is resting on something: back off instead of pushing.
        pick["blocked"] = True
        res, dt = go_pose("unblock", (float(ee_now[0]), float(ee_now[1]), hover_z), grasp_q, tol=0.02, steps=40)
        report["picks"].append(pick)
        if terminal(res):
            ended = True
            break
        continue
    t_descend = min(1.0, max(0.3, 0.5 * t_descend + 0.5 * dt))
    res, dt = timed("close", lambda: close_gripper())
    if terminal(res):
        ended = True
        break
    t_close = min(0.4, max(0.1, 0.5 * t_close + 0.5 * dt))
    g_closed = float(get_robot_state().gripper_positions["primary"])
    pick["g_closed"] = round(g_closed, 3)
    lift_z = max(grasp_z + 0.15, bin_geo["ztop"] + 0.14)
    res, dt = go_pose("lift", (gx, gy, lift_z), grasp_q, tol=0.02, steps=80)
    if terminal(res):
        ended = True
        break
    g_lift = float(get_robot_state().gripper_positions["primary"])
    pick["g_lift"] = round(g_lift, 3)
    held = g_lift > 0.06
    pick["held"] = held
    if not held:
        timed("open_after_miss", lambda: open_gripper())
        report["picks"].append(pick)
        continue
    res, dt = go_pose("over_bin", (bin_geo["x"], bin_geo["y"], bin_geo["ztop"] + 0.12), grasp_q, tol=0.02, steps=150)
    if terminal(res):
        ended = True
        break
    res, dt = go_pose("lower_bin", (bin_geo["x"], bin_geo["y"], bin_geo["ztop"] + 0.07), grasp_q, tol=0.02, steps=60)
    if terminal(res):
        ended = True
        break
    g_before_release = float(get_robot_state().gripper_positions["primary"])
    pick["g_before_release"] = round(g_before_release, 3)
    res, dt = timed("release", lambda: open_gripper())
    if g_before_release > 0.06:
        placed[target["color"]] += 1
        pick["counted"] = True
    report["picks"].append(pick)
    if terminal(res):
        ended = True
        break
    res, dt = go_pose("rise_bin", (bin_geo["x"], bin_geo["y"], bin_geo["ztop"] + 0.16), grasp_q, tol=0.03, steps=60)
    if terminal(res):
        ended = True
        break

report["placed"] = dict(placed)
report["attempts"] = attempts
report["waits"] = waits
report["vel_final"] = [round(vel[0], 4), round(vel[1], 4)]
report["vel_samples"] = len(vel_samples)
report["t_before_press"] = safe_now()

# Press the button with a closed hand, top-down.
if not ended and button_geo is not None:
    try:
        timed("close_for_press", lambda: close_gripper())
        bx, by, btop = button_geo["x"], button_geo["y"], button_geo["ztop"]
        res, dt = go_pose("over_button", (bx, by, btop + 0.08), home_q, tol=0.01, steps=150)
        if not terminal(res):
            res, dt = go_pose("press", (bx, by, btop - 0.012), home_q, tol=0.005, steps=40)
        if not terminal(res):
            res, dt = go_pose("unpress", (bx, by, btop + 0.06), home_q, tol=0.02, steps=40)
        if not terminal(res):
            hold_ticks(5)
    except Exception as exc:
        report["press_error"] = str(exc)[:200]

report["t_end"] = safe_now()
result = report
