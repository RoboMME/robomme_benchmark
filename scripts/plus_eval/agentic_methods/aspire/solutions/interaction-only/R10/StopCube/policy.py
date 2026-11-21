# StopCube v02 — perception: geometry + motion object identification (flat disc = target,
# static raised object = button, moving raised object = cube); timing: faster approach,
# latency calibrated above the button without contact (= ticks to first contact), and the
# calibration is skipped when fewer than two target visits remain.
#
# Evidence: v00/v01 ep0 — the goal reads "press the button to stop the cube just as it
# reaches the target for the <N>-th time"; a visit is the cube centre entering the observed
# target radius (3D); the episode ends ~3-4 ticks after the N-th visit if not stopped; the
# button registers on first contact. v01 ep1/ep2 — `button` prompt top-1 was the target disc
# (both failed with no plan); ep1's 5th visit was at tick 83 (period ~19 ticks).
# All positions, radii, colours and latencies are measured at runtime.
# Timing: closest approach predicted by constant-acceleration extrapolation of the last three
# per-tick cube positions (offline replay on the recorded v00/v01 tracks: rms 0.41 tick at a
# 3-tick horizon, best of six predictors tried); fire when it is <= latency + 0.5 ticks.
import math

report = {"version": "v02", "events": []}
FRONT = "front"


def note(msg):
    report["events"].append(str(msg))


def safe_dict(mapping):
    out = {}
    try:
        for key in mapping:
            try:
                text = str(mapping[key])
                out[str(key)] = text if len(text) < 300 else text[:300]
            except Exception as exc:
                out[str(key)] = "err " + str(exc)
    except Exception as exc:
        out["error"] = str(exc)
    return out


# ---------------------------------------------------------------- goal / metadata
ctx = get_task_context()
lang = str(ctx.language)
report["language"] = lang
meta = {}
try:
    meta = robomme_plus.get_task_metadata()
    report["task_metadata"] = safe_dict(meta)
except Exception as exc:
    report["task_metadata_error"] = str(exc)
max_steps = 0
try:
    max_steps = int(meta["max_steps"])
except Exception:
    max_steps = 0
report["max_steps"] = max_steps
try:
    report["conditioning_video_frames"] = len(robomme_plus.get_conditioning_video())
except Exception as exc:
    report["conditioning_video_error"] = str(exc)

WORDS = {"once": 1, "twice": 2, "thrice": 3, "first": 1, "second": 2, "third": 3,
         "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
         "tenth": 10, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "1st": 1, "2nd": 2, "3rd": 3,
         "4th": 4, "5th": 5, "6th": 6, "7th": 7, "8th": 8, "9th": 9, "10th": 10}
clean = ""
for ch in lang.lower():
    clean = clean + (ch if (ch.isalnum() or ch == " ") else " ")
count_needed = None
for tok in clean.split():
    if tok in WORDS:
        count_needed = WORDS[tok]
        break
    if tok.isdigit():
        count_needed = int(tok)
        break
report["count_parsed"] = count_needed
if count_needed is None:
    count_needed = 1

# ---------------------------------------------------------------- perception
state0 = get_robot_state()
EMB = state0.embodiment
BASE = state0.base_frame
home_joints = [float(v) for v in state0.joint_positions["primary"]]
ee0 = state0.end_effector_poses["primary"]
down_q = [float(v) for v in ee0.quaternion_wxyz]

obs0 = get_observation()
cam0 = obs0.cameras[FRONT]
img0 = cam0.rgb
H = len(img0)
W = len(img0[0])
K = cam0.intrinsics
FX = float(K[0][0])
FY = float(K[1][1])
CX = float(K[0][2])
CY = float(K[1][2])
cpos = [float(v) for v in cam0.camera_pose.position]
qw, qx, qy, qz = (float(v) for v in cam0.camera_pose.quaternion_wxyz)
ROT = [[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
       [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
       [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]]
report["camera"] = {"pos": [round(v, 4) for v in cpos], "f": [FX, FY], "c": [CX, CY]}


def pixel_to_plane(u, v, z_plane):
    """Ray through pixel (OpenCV camera convention) intersected with z = z_plane in BASE."""
    dc = [(u - CX) / FX, (v - CY) / FY, 1.0]
    d = [ROT[i][0] * dc[0] + ROT[i][1] * dc[1] + ROT[i][2] * dc[2] for i in range(3)]
    if abs(d[2]) < 1e-6:
        return None
    s = (z_plane - cpos[2]) / d[2]
    return (cpos[0] + s * d[0], cpos[1] + s * d[1])


def seg_list(prompt, camera=FRONT):
    entry = {"prompt": prompt, "ok": False, "cands": []}
    out = []
    try:
        found = segment_text(camera, prompt)
        entry["ok"] = bool(found.ok)
        if found.ok:
            for cand in found.segmentations[:5]:
                if cand.box_xyxy is None:
                    continue
                box = [float(v) for v in cand.box_xyxy]
                entry["cands"].append({"score": round(float(cand.score), 3), "box": [round(v, 1) for v in box]})
                out.append(cand)
    except Exception as exc:
        entry["error"] = str(exc)
    report.setdefault("probes", []).append(entry)
    return out


def box_of(cand):
    return [float(v) for v in cand.box_xyxy]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def center_in(box, other, pad):
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return other[0] - pad <= cx <= other[2] + pad and other[1] - pad <= cy <= other[3] + pad


def quant(values, frac):
    ordered = sorted(float(v) for v in values)
    return ordered[int((len(ordered) - 1) * frac)]


def cloud_summary(cand):
    try:
        pc = mask_to_point_cloud(cand.mask, FRONT, target_frame=BASE)
        pts = pc.points
        if len(pts) < 5:
            return None
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        zs = [float(p[2]) for p in pts]
        return {"n": len(pts), "x50": quant(xs, 0.5), "y50": quant(ys, 0.5),
                "z10": quant(zs, 0.1), "z50": quant(zs, 0.5), "z90": quant(zs, 0.9),
                "xr": [quant(xs, 0.05), quant(xs, 0.95)], "yr": [quant(ys, 0.05), quant(ys, 0.95)]}
    except Exception as exc:
        note("cloud error " + str(exc))
        return None


# ---- 1. collect distinct candidate objects from several prompts (dedupe by IoU)
PROMPTS = ("target", "circular target", "button", "cube")
objs = []
for prompt in PROMPTS:
    for cand in seg_list(prompt):
        box = box_of(cand)
        if box[2] - box[0] < 3 or box[3] - box[1] < 3 or box[2] - box[0] > 70 or box[3] - box[1] > 70:
            continue
        match = None
        for o in objs:
            if iou(o["box"], box) > 0.5:
                match = o
                break
        if match is None:
            objs.append({"box": box, "cand": cand, "scores": {prompt: float(cand.score)}})
        else:
            match["scores"][prompt] = max(match["scores"].get(prompt, 0.0), float(cand.score))

# ---- 2. 3D geometry per object; keep table-top objects only
table_objs = []
for o in objs:
    cl = cloud_summary(o["cand"])
    o["cloud"] = cl
    o["table"] = False
    if cl is None or cl["n"] < 10:
        continue
    if cl["z90"] > 0.08 or cl["xr"][1] - cl["xr"][0] > 0.12 or cl["yr"][1] - cl["yr"][0] > 0.12:
        continue
    o["table"] = True
    table_objs.append(o)
FLAT_Z = 0.024  # top below this = flat disc lying on the table (disc top ~0.016, cube/button top >= 0.03)


def sc(o, names):
    best = 0.0
    for name in names:
        best = max(best, o["scores"].get(name, 0.0))
    return best


target_obj = None
for o in table_objs:
    if o["cloud"]["z90"] < FLAT_Z:
        if target_obj is None or sc(o, ("target", "circular target")) > sc(target_obj, ("target", "circular target")):
            target_obj = o
raised = []
for o in table_objs:
    if o is target_obj:
        continue
    if o["cloud"]["z90"] >= FLAT_Z:
        raised.append(o)
report["objects"] = [{"box": [round(v, 1) for v in o["box"]], "scores": o["scores"],
                      "z90": None if o.get("cloud") is None else round(o["cloud"]["z90"], 4),
                      "xy": None if o.get("cloud") is None else [round(o["cloud"]["x50"], 4), round(o["cloud"]["y50"], 4)],
                      "table": o["table"]} for o in objs]


def rows_of(img, y0, y1):
    rows = {}
    for y in range(y0, y1):
        rows[y] = img[y].tolist()
    return rows


def clip_box(box, pad):
    return (max(0, int(box[0]) - pad), max(0, int(box[1]) - pad), min(W - 1, int(box[2]) + 1 + pad), min(H - 1, int(box[3]) + 1 + pad))


def mode_colour(img, cand):
    x1, y1, x2, y2 = clip_box(box_of(cand), 0)
    counts = {}
    mask = cand.mask
    for y in range(y1, y2 + 1):
        mrow = mask[y].tolist()
        row = img[y].tolist()
        for x in range(x1, x2 + 1):
            if mrow[x]:
                px = row[x]
                key = (int(px[0]), int(px[1]), int(px[2]))
                counts[key] = counts.get(key, 0) + 1
    best = None
    for key in counts:
        if best is None or counts[key] > counts[best]:
            best = key
    return best, counts.get(best, 0) if best is not None else 0


def changed_fraction(img_a, img_b, cand):
    x1, y1, x2, y2 = clip_box(box_of(cand), 0)
    mask = cand.mask
    n = 0
    ch = 0
    for y in range(y1, y2 + 1):
        mrow = mask[y].tolist()
        ra = img_a[y].tolist()
        rb = img_b[y].tolist()
        for x in range(x1, x2 + 1):
            if mrow[x]:
                n += 1
                pa = ra[x]
                pb = rb[x]
                if abs(pa[0] - pb[0]) + abs(pa[1] - pb[1]) + abs(pa[2] - pb[2]) > 30:
                    ch += 1
    return float(ch) / n if n > 0 else 0.0


# ---------------------------------------------------------------- stepping helpers
series = []
tick = [0]
ended = [False]
last_obs = [obs0]
hist = []
visit = {"count": 0, "inside": False, "log": []}


def do_step(joints, grip):
    if ended[0]:
        return None
    reply = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=[float(v) for v in joints],
        gripper_position=float(grip), embodiment=EMB)}))
    tick[0] += 1
    term = False
    trunc = False
    try:
        term = bool(reply.terminated)
    except Exception:
        term = False
    try:
        trunc = bool(reply.truncated)
    except Exception:
        trunc = False
    if term or trunc:
        ended[0] = True
        note("episode ended at tick " + str(tick[0]) + " term=" + str(term) + " trunc=" + str(trunc))
    try:
        last_obs[0] = reply.observation
    except Exception:
        last_obs[0] = None
    return reply


def ee_z():
    try:
        return float(last_obs[0].robot_state.end_effector_poses["primary"].position[2])
    except Exception:
        return None


# ---- 3. motion test: the cube moves, the button does not (3 hold ticks at home)
HOLD = 4
hold_obs = []
for k in range(HOLD):
    do_step(home_joints, 1.0)
    hold_obs.append(last_obs[0])
img_t = last_obs[0].cameras[FRONT].rgb if last_obs[0] is not None else get_observation().cameras[FRONT].rgb
motion = []
for o in raised:
    o["moved"] = changed_fraction(img0, img_t, o["cand"])
    motion.append([[round(v, 1) for v in o["box"]], round(o["moved"], 3)])
report["motion_test"] = motion
cube_obj = None
for o in raised:
    if o["moved"] > 0.3 and (cube_obj is None or o["moved"] > cube_obj["moved"]):
        cube_obj = o
if cube_obj is None:
    note("no raised object moved; falling back to prompt scores for the cube")
    for o in raised:
        if cube_obj is None or sc(o, ("cube",)) - sc(o, ("button",)) > sc(cube_obj, ("cube",)) - sc(cube_obj, ("button",)):
            cube_obj = o
button_obj = None
for o in raised:
    if o is cube_obj or o["moved"] > 0.3:
        continue
    if button_obj is None or sc(o, ("button",)) > sc(button_obj, ("button",)):
        button_obj = o
btop = None
if button_obj is not None:
    btop = button_obj["cloud"]["z90"]
    for o in raised:
        if o is cube_obj or o is button_obj or o["moved"] > 0.3:
            continue
        if center_in(o["box"], button_obj["box"], 2.0) or center_in(button_obj["box"], o["box"], 2.0):
            btop = max(btop, o["cloud"]["z90"])
report["identified"] = {
    "target": None if target_obj is None else [round(v, 1) for v in target_obj["box"]],
    "cube": None if cube_obj is None else [round(v, 1) for v in cube_obj["box"]],
    "button": None if button_obj is None else [round(v, 1) for v in button_obj["box"]],
    "btop": btop}

# ---- 4. tracker + 3D target model
tracker = None
cube_top = None
if cube_obj is not None:
    col, ncol = mode_colour(img0, cube_obj["cand"])
    if col is not None:
        b = cube_obj["box"]
        tracker = {"col": col, "cx": (b[0] + b[2]) / 2.0, "cy": (b[1] + b[3]) / 2.0, "lost": 0}
        report["tracker_init"] = {"col": list(col), "n_mode": ncol}
    cube_top = cube_obj["cloud"]["z90"]
target_xy = None
target_r = None
if target_obj is not None:
    tc = target_obj["cloud"]
    target_xy = (tc["x50"], tc["y50"])
    target_r = 0.25 * ((tc["xr"][1] - tc["xr"][0]) + (tc["yr"][1] - tc["yr"][0]))
report["target_xy_r"] = [target_xy, target_r]

COLTHR = 24


def scan(img, col, x0, x1, y0, y1):
    rows = rows_of(img, y0, y1 + 1)
    sx = 0.0
    sy = 0.0
    n = 0
    c0 = col[0]
    c1 = col[1]
    c2 = col[2]
    for y in range(y0, y1 + 1):
        row = rows[y]
        for x in range(x0, x1 + 1):
            px = row[x]
            if abs(px[0] - c0) + abs(px[1] - c1) + abs(px[2] - c2) < COLTHR:
                sx += x
                sy += y
                n += 1
    if n == 0:
        return None
    return (sx / n, sy / n, n)


def observe(obs=None, at_tick=None):
    """Track the cube top face, update 3D history and the visit counter."""
    if tracker is None or cube_top is None:
        return None
    if obs is None:
        obs = last_obs[0]
    if obs is None:
        try:
            obs = get_observation()
        except Exception:
            return None
    now = tick[0] if at_tick is None else at_tick
    img = obs.cameras[FRONT].rgb
    half = 30
    hit = scan(img, tracker["col"], max(0, int(tracker["cx"]) - half), min(W - 1, int(tracker["cx"]) + half),
               max(0, int(tracker["cy"]) - half), min(H - 1, int(tracker["cy"]) + half))
    if hit is None or hit[2] < 3:
        hit = scan(img, tracker["col"], 0, W - 1, 0, H - 1)
        if hit is not None and hit[2] >= 3:
            note("tracker reacquired by full scan at tick " + str(now))
    if hit is None or hit[2] < 3:
        tracker["lost"] += 1
        series.append([now, None, None, 0, None])
        return None
    tracker["cx"] = hit[0]
    tracker["cy"] = hit[1]
    p = pixel_to_plane(hit[0], hit[1], cube_top)
    if p is None:
        return None
    hist.append((now, p[0], p[1]))
    if len(hist) > 6:
        hist.pop(0)
    d = None
    if target_xy is not None:
        d = math.sqrt((p[0] - target_xy[0]) ** 2 + (p[1] - target_xy[1]) ** 2)
        if not visit["inside"] and d < target_r:
            visit["inside"] = True
            visit["count"] += 1
            visit["log"].append(["enter", now, visit["count"], round(d, 4)])
        elif visit["inside"] and d > target_r + 0.01:
            visit["inside"] = False
            visit["log"].append(["exit", now, visit["count"], round(d, 4)])
    series.append([now, round(hit[0], 2), round(hit[1], 2), hit[2],
                   round(p[0], 4), round(p[1], 4), None if d is None else round(d, 4)])
    return d


def closest_approach():
    """Constant-acceleration extrapolation of the last three consecutive cube positions:
    ticks until the closest approach to the target centre, and the predicted miss distance."""
    if len(hist) < 3 or target_xy is None:
        return None
    t0, x0, y0 = hist[-3]
    t1, x1, y1 = hist[-2]
    t2, x2, y2 = hist[-1]
    if t2 - t1 != 1 or t1 - t0 != 1 or tick[0] != t2:
        return None
    ax = x2 - 2.0 * x1 + x0
    ay = y2 - 2.0 * y1 + y0
    vx = (x2 - x1) + 0.5 * ax
    vy = (y2 - y1) + 0.5 * ay
    best_s = 0.0
    best_d = None
    s = 0.0
    while s <= 8.0:
        px = x2 + vx * s + 0.5 * ax * s * s - target_xy[0]
        py = y2 + vy * s + 0.5 * ay * s * s - target_xy[1]
        dd = math.sqrt(px * px + py * py)
        if best_d is None or dd < best_d:
            best_d = dd
            best_s = s
        s += 0.05
    return {"t": best_s, "miss": best_d, "speed": math.sqrt(vx * vx + vy * vy)}


# replay the start and hold frames through the tracker so no visit is missed
observe(obs0, 0)
for k in range(len(hold_obs)):
    if hold_obs[k] is not None:
        observe(hold_obs[k], k + 1)
report["inside_after_hold"] = visit["inside"]

# ---------------------------------------------------------------- motion plan
plan = {}
press_ok = False
if button_obj is not None and btop is not None and target_xy is not None and tracker is not None:
    bx = button_obj["cloud"]["x50"]
    by = button_obj["cloud"]["y50"]
    HOVER = 0.04      # contact travel of the press (m)
    PRESS = -0.012    # commanded press depth below the observed button top (m)
    PRE = 0.08        # pre-hover height used for the no-contact latency calibration (m)
    ik_pre = solve_ik(Pose((bx, by, btop + PRE), down_q, BASE), seed_joints=home_joints)
    if ik_pre.ok:
        pre_j = [float(v) for v in ik_pre.joint_positions]
        ik_h = solve_ik(Pose((bx, by, btop + HOVER), down_q, BASE), seed_joints=pre_j)
        ik_c = solve_ik(Pose((bx, by, btop + PRE - (HOVER - PRESS)), down_q, BASE), seed_joints=pre_j)
        plan = {"button": [round(bx, 4), round(by, 4), round(btop, 4)], "pre_ok": True,
                "hover_ok": bool(ik_h.ok), "calib_ok": bool(ik_c.ok)}
        if ik_h.ok:
            hover_j = [float(v) for v in ik_h.joint_positions]
            ik_p = solve_ik(Pose((bx, by, btop + PRESS), down_q, BASE), seed_joints=hover_j)
            plan["press_ok"] = bool(ik_p.ok)
            if ik_p.ok:
                press_ok = True
                press_j = [float(v) for v in ik_p.joint_positions]
    else:
        plan = {"pre_ok": False}
report["plan"] = plan


budget = max_steps - 3 if max_steps > 0 else 500
GRIP = 0.0  # closed fingers: a solid pusher
fired = None
if press_ok:
    # a) home -> pre-hover (PRE above the button) in 20 interpolated ticks, tracking throughout
    for k in range(1, 21):
        a = float(k) / 20.0
        do_step([home_joints[i] + a * (pre_j[i] - home_joints[i]) for i in range(7)], GRIP)
        observe()
        if ended[0]:
            break
    for k in range(6):
        if ended[0]:
            break
        do_step(pre_j, GRIP)
        observe()
    report["pre_ready_tick"] = tick[0]
    # b) latency calibration without contact: same commanded step (HOVER-PRESS) as the press,
    #    count ticks until the tool has travelled HOVER (the press's contact distance).
    latency = 3
    lat_src = "default"
    left = count_needed - visit["count"]   # visits still to come, including the one to stop on
    if ik_c.ok and left >= 2 and not ended[0]:
        c_j = [float(v) for v in ik_c.joint_positions]
        z_start = ee_z()
        zs = []
        t_travel = None
        for k in range(1, 9):
            do_step(c_j, GRIP)
            observe()
            z = ee_z()
            zs.append(None if z is None else round(z, 4))
            if z is not None and z_start is not None and t_travel is None and z_start - z >= HOVER:
                t_travel = k
                break
            if ended[0]:
                break
        report["calibration"] = {"z_start": z_start, "zs": zs, "t_travel": t_travel}
        if t_travel is not None:
            latency = t_travel
            lat_src = "calibrated"
    else:
        report["calibration"] = {"skipped": True, "visits_left": left}
    report["latency"] = [latency, lat_src]
    # c) settle at hover (HOVER above the button)
    for k in range(8):
        if ended[0]:
            break
        do_step(hover_j, GRIP)
        observe()
    report["hover_ready_tick"] = tick[0]
    # d) wait for the N-th visit; fire so that first contact lands at the closest approach
    approach_log = []
    while not ended[0] and tick[0] < budget:
        do_step(hover_j, GRIP)
        observe()
        ca = closest_approach()
        upcoming = visit["count"] + (0 if visit["inside"] else 1)
        if ca is not None and upcoming == count_needed and ca["miss"] < target_r:
            approach_log.append([tick[0], round(ca["t"], 2), round(ca["miss"], 4)])
            late = visit["inside"] and ca["t"] <= 0.0
            if (ca["t"] > 0.0 and ca["t"] <= latency + 0.5) or late:
                fired = {"tick": tick[0], "t_closest": round(ca["t"], 2), "miss": round(ca["miss"], 4),
                         "speed": round(ca["speed"], 4), "visits": visit["count"], "inside": visit["inside"],
                         "late": late}
                break
    report["approach_log"] = approach_log[-12:]
    report["fired"] = fired
    if fired is not None:
        press_trace = []
        for k in range(40):
            if ended[0]:
                break
            do_step(press_j, GRIP)
            d = observe()
            press_trace.append([tick[0], ee_z(), None if d is None else round(d, 4)])
        report["press_trace"] = press_trace
else:
    note("no press plan; observing only")
    while not ended[0] and tick[0] < min(budget, 300):
        do_step(home_joints, 1.0)
        observe()

report["ticks"] = tick[0]
report["visits"] = visit["log"]
report["tracker_lost_ticks"] = tracker["lost"] if tracker is not None else None
report["series"] = series
result = report
