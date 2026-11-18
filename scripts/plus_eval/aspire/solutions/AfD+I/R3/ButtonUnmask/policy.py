# ButtonUnmask v00 -- initial program from offline scene analysis (released recordings).
#
# Strategy (all task-dependent values are measured at runtime):
#   1. Parse the ordered colour targets from get_task_context().language.
#   2. Calibrate the depth->robot_base deprojection convention per camera from the table plane.
#   3. Localize the button (SAM3 text prompt, size-bounded) and solve IK for hover/press poses.
#   4. Watch the scene tick by tick (joint hold) with BOTH cameras, detect saturated cube colours,
#      deproject each blob to robot_base and cluster sightings into cube positions.
#      Once the first reveal is captured, press the button (stepped joint interpolation, still
#      observing), return to the home pose and keep observing until the scene is quiet.
#   5. For each goal colour in order: move above the remembered cube position, measure the
#      container covering it from wrist depth (top face, yaw by min-area rectangle), top-down
#      grasp, lift, put it back, release.
import math

report = {"version": "v00", "events": [], "errors": [], "frames": []}


def note(kind, data):
    try:
        tick_now = current_tick()
    except Exception:
        tick_now = -1
    report["events"].append({"tick": tick_now, "kind": kind, "data": data})


# ---------------------------------------------------------------- basic state / time
STATE = {"ended": False, "steps": 0, "hz": 20.0, "t0": None}


def current_tick():
    st = get_robot_state()
    ts = float(st.timestamp_s)
    if STATE["t0"] is None:
        STATE["t0"] = ts
    return int(round((ts - STATE["t0"]) * STATE["hz"]))


def flag_end(res, where):
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
        STATE["ended"] = True
        note("episode_end", {"where": where, "terminated": term, "truncated": trunc})
    return STATE["ended"]


# ---------------------------------------------------------------- math helpers
def quat_to_mat(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
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


def quantile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def wrap90(angle_deg):
    a = angle_deg
    while a > 45.0:
        a -= 90.0
    while a <= -45.0:
        a += 90.0
    return a


# ---------------------------------------------------------------- camera geometry
# Three candidate camera-axis conventions; the right one is chosen at runtime from the table plane.
CONVENTIONS = ("opencv", "opengl", "xforward")


def cam_point(conv, u, v, d, K):
    fx, fy, cx, cy = K
    a = (u - cx) / fx * d
    b = (v - cy) / fy * d
    if conv == "opencv":
        return (a, b, d)
    if conv == "opengl":
        return (a, -b, -d)
    return (d, -a, -b)


def as_lists(arr, channels):
    """Read-only array -> nested lists; .tolist() when available, element indexing otherwise."""
    try:
        return arr.tolist()
    except Exception:
        pass
    rows = []
    for v in range(len(arr)):
        src = arr[v]
        row = []
        for u in range(len(src)):
            if channels > 0:
                px = src[u]
                row.append([int(px[0]), int(px[1]), int(px[2])])
            else:
                row.append(float(src[u]))
        rows.append(row)
    return rows


def make_frame(cam, conv):
    k = as_lists(cam.intrinsics, 0)
    pose = cam.camera_pose
    rgb = as_lists(cam.rgb, 3)
    return {"rgb": rgb, "depth": as_lists(cam.depth_m, 0),
            "K": (float(k[0][0]), float(k[1][1]), float(k[0][2]), float(k[1][2])),
            "pos": [float(p) for p in pose.position], "R": quat_to_mat(pose.quaternion_wxyz),
            "conv": conv, "h": len(rgb), "w": len(rgb[0])}


def world(frame, u, v, conv=None):
    d = float(frame["depth"][v][u])
    if not (d > 0.02 and d < 5.0):
        return None
    c = conv if conv is not None else frame["conv"]
    p = cam_point(c, u + 0.5, v + 0.5, d, frame["K"])
    R = frame["R"]
    t = frame["pos"]
    return (
        R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2] + t[0],
        R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2] + t[1],
        R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2] + t[2],
    )


def calibrate_convention(cam):
    """Pick the axis convention under which a coarse pixel grid forms a horizontal plane below
    the camera (the table dominates both views at the start)."""
    frame = make_frame(cam, "opencv")
    scores = {}
    best = None
    for conv in CONVENTIONS:
        zs = []
        for v in range(8, frame["h"], 16):
            for u in range(8, frame["w"], 16):
                p = world(frame, u, v, conv)
                if p is not None:
                    zs.append(p[2])
        if len(zs) < 20:
            continue
        q25 = quantile(zs, 0.25)
        q75 = quantile(zs, 0.75)
        med = quantile(zs, 0.5)
        spread = q75 - q25
        penalty = 0.0
        if med > frame["pos"][2]:
            penalty = 10.0
        score = spread + penalty
        scores[conv] = {"iqr": round(spread, 4), "median_z": round(med, 4), "score": round(score, 4)}
        if best is None or score < scores[best]["score"]:
            best = conv
    return best, scores


# ---------------------------------------------------------------- colour detection
def classify(r, g, b):
    mx = r
    if g > mx:
        mx = g
    if b > mx:
        mx = b
    mn = r
    if g < mn:
        mn = g
    if b < mn:
        mn = b
    if mx < 90:
        return None
    s = (mx - mn) / mx
    if s < 0.72:
        if s > 0.45 and b == mx and r > g + 40 and b > 120:
            return "purple"
        return None
    if mx == r:
        h = 60.0 * (g - b) / (mx - mn)
        if h < 0:
            h += 360.0
    elif mx == g:
        h = 60.0 * (b - r) / (mx - mn) + 120.0
    else:
        h = 60.0 * (r - g) / (mx - mn) + 240.0
    if h < 12 or h > 340:
        return "red"
    if 28 <= h < 47:
        return "orange"
    if 47 <= h < 72:
        return "yellow"
    if 80 <= h < 165:
        return "green"
    if 200 <= h < 258:
        if mx < 110:
            return None
        return "blue"
    if 262 <= h < 320:
        return "purple"
    return None


def detect_blobs(frame, stride, min_cells):
    """Connected components of saturated cube colours on a stride grid. Returns blob dicts with a
    robot_base point (per-axis median of the deprojected blob pixels)."""
    labels = {}
    rgb = frame["rgb"]
    for v in range(0, frame["h"], stride):
        row = rgb[v]
        for u in range(0, frame["w"], stride):
            px = row[u]
            c = classify(px[0], px[1], px[2])
            if c is not None:
                labels[(u, v)] = c
    blobs = []
    seen = set()
    for key in labels:
        if key in seen:
            continue
        color = labels[key]
        stack = [key]
        seen.add(key)
        cells = []
        while len(stack) > 0:
            cur = stack.pop()
            cells.append(cur)
            cu, cv = cur
            for nb in ((cu + stride, cv), (cu - stride, cv), (cu, cv + stride), (cu, cv - stride)):
                if nb in labels and nb not in seen and labels[nb] == color:
                    seen.add(nb)
                    stack.append(nb)
        if len(cells) < min_cells:
            continue
        xs = []
        ys = []
        zs = []
        su = 0
        sv = 0
        for cell in cells:
            su += cell[0]
            sv += cell[1]
            p = world(frame, cell[0], cell[1])
            if p is not None:
                xs.append(p[0])
                ys.append(p[1])
                zs.append(p[2])
        if len(xs) < max(2, min_cells // 2):
            continue
        blobs.append({
            "color": color,
            "cells": len(cells),
            "px": [round(su / len(cells), 1), round(sv / len(cells), 1)],
            "xyz": [quantile(xs, 0.5), quantile(ys, 0.5), quantile(zs, 0.5)],
        })
    return blobs


CLUSTERS = []
STATIC = []
MERGE_R = 0.04


def add_sighting(blob, cam_name, tick):
    x, y, z = blob["xyz"]
    for s in STATIC:
        if s["color"] == blob["color"] and math.hypot(s["xyz"][0] - x, s["xyz"][1] - y) < 0.03:
            return
    best = None
    best_d = None
    for c in CLUSTERS:
        if c["color"] != blob["color"]:
            continue
        d = math.hypot(c["x"] - x, c["y"] - y)
        if d < MERGE_R and (best_d is None or d < best_d):
            best = c
            best_d = d
    if best is None:
        best = {"color": blob["color"], "xs": [], "ys": [], "zs": [], "n": 0, "cams": [],
                "first_tick": tick, "last_tick": tick, "x": x, "y": y, "z": z}
        CLUSTERS.append(best)
    best["n"] += 1
    if len(best["xs"]) < 60:
        best["xs"].append(x)
        best["ys"].append(y)
        best["zs"].append(z)
    best["x"] = quantile(best["xs"], 0.5)
    best["y"] = quantile(best["ys"], 0.5)
    best["z"] = quantile(best["zs"], 0.5)
    best["last_tick"] = tick
    if cam_name not in best["cams"]:
        best["cams"].append(cam_name)


OBS = {"conv": {}, "table_z": None, "last_seen_tick": None, "reveal_frames": 0,
       "first_reveal_tick": None, "processed": 0}


def process_observation(obs, tick, static_pass=False, record=True):
    counts = {}
    total = 0
    for cam_name in ("front", "wrist"):
        try:
            cam = obs.cameras[cam_name]
        except Exception:
            continue
        conv = OBS["conv"].get(cam_name)
        if conv is None:
            continue
        frame = make_frame(cam, conv)
        blobs = detect_blobs(frame, 2, 3)
        for blob in blobs:
            z = blob["xyz"][2]
            tz = OBS["table_z"]
            if tz is not None and (z < tz - 0.03 or z > tz + 0.10):
                continue
            key = cam_name[0] + ":" + blob["color"]
            counts[key] = counts.get(key, 0) + 1
            total += 1
            if static_pass:
                STATIC.append({"color": blob["color"], "xyz": blob["xyz"], "cam": cam_name})
            else:
                add_sighting(blob, cam_name, tick)
    OBS["processed"] += 1
    if not static_pass and total > 0:
        OBS["last_seen_tick"] = tick
        OBS["reveal_frames"] += 1
        if OBS["first_reveal_tick"] is None:
            OBS["first_reveal_tick"] = tick
    if record and len(report["frames"]) < 400:
        report["frames"].append([tick, counts])
    return total


# ---------------------------------------------------------------- motion helpers
def hold_tick(q, grip):
    if STATE.get("emb") is None:
        STATE["emb"] = get_robot_state().embodiment
    res = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=[float(a) for a in q], gripper_position=float(grip),
        embodiment=STATE["emb"])}))
    STATE["steps"] += 1
    flag_end(res, "step")
    return res


def joints_now():
    return [float(a) for a in get_robot_state().joint_positions["primary"]]


def servo_to(q_target, grip, max_dq, observe_every, settle_max):
    q0 = joints_now()
    dmax = 0.0
    for i in range(7):
        dmax = max(dmax, abs(q_target[i] - q0[i]))
    n = int(math.ceil(dmax / max_dq))
    if n < 1:
        n = 1
    for i in range(1, n + 1):
        if STATE["ended"]:
            return False
        frac = float(i) / n
        q = [q0[j] + (q_target[j] - q0[j]) * frac for j in range(7)]
        hold_tick(q, grip)
        if observe_every > 0 and i % observe_every == 0 and not STATE["ended"]:
            process_observation(get_observation(), current_tick())
    for k in range(settle_max):
        if STATE["ended"]:
            return False
        qn = joints_now()
        err = 0.0
        for j in range(7):
            err = max(err, abs(q_target[j] - qn[j]))
        if err < 0.015:
            break
        hold_tick(q_target, grip)
        if observe_every > 0 and k % observe_every == 0 and not STATE["ended"]:
            process_observation(get_observation(), current_tick())
    return not STATE["ended"]


def plan_ticks(q_from, q_to, max_dq):
    dmax = 0.0
    for i in range(7):
        dmax = max(dmax, abs(q_to[i] - q_from[i]))
    return int(math.ceil(dmax / max_dq))


def move_pose(pos, quat, tol, label):
    if STATE["ended"]:
        return False
    res = move_to_pose(Pose(position=[float(pos[0]), float(pos[1]), float(pos[2])],
                            quaternion_wxyz=[float(q) for q in quat], frame=BASE),
                       tolerance=tol, max_steps=300)
    ok = False
    status = "?"
    steps_exec = -1
    try:
        ok = bool(res.ok)
        status = str(res.status)
        steps_exec = int(res.steps_executed)
    except Exception:
        pass
    note("move", {"label": label, "ok": ok, "status": status, "steps": steps_exec,
                  "target": [round(float(a), 4) for a in pos]})
    flag_end(res, label)
    return ok


def gripper_call(open_it, label):
    if STATE["ended"]:
        return None
    if open_it:
        res = open_gripper()
    else:
        res = close_gripper()
    flag_end(res, label)
    g = None
    try:
        g = float(get_robot_state().gripper_positions["primary"])
    except Exception:
        g = None
    note("gripper", {"label": label, "pos": g})
    return g


# ================================================================ main program
ctx = get_task_context()
goal = str(ctx.language)
report["goal"] = goal
BASE = get_robot_state().base_frame

meta = {}
try:
    raw = robomme_plus.get_task_metadata()
    for k in raw:
        val = raw[k]
        sval = str(val)
        meta[str(k)] = val if len(sval) < 200 else sval[:200]
except Exception as exc:
    report["errors"].append("metadata: " + str(exc))
report["task_metadata"] = {k: str(meta[k]) for k in meta}
for k in meta:
    lk = k.lower()
    if ("hz" in lk or "freq" in lk) and "video" not in lk:
        try:
            STATE["hz"] = float(meta[k])
        except Exception:
            pass
max_steps = None
try:
    max_steps = int(meta.get("max_steps"))
except Exception:
    max_steps = None
report["max_steps"] = max_steps
try:
    cm = robomme_plus.get_controller_metadata()
    report["controller_metadata"] = {str(k): str(cm[k])[:200] for k in cm}
except Exception as exc:
    report["errors"].append("controller metadata: " + str(exc))

# ---- parse goal: ordered list of (colour, all_flag)
COLOR_WORDS = ("red", "green", "blue", "yellow", "orange", "purple")
tokens = goal.lower().replace(",", " ").replace(".", " ").replace(";", " ").split()
targets = []
for i in range(len(tokens)):
    if tokens[i] != "hiding":
        continue
    color = None
    plural = False
    for j in range(i + 1, min(len(tokens), i + 5)):
        if tokens[j] in COLOR_WORDS and color is None:
            color = tokens[j]
        if tokens[j] == "cubes":
            plural = True
        if tokens[j] in ("cube", "cubes"):
            break
    for j in range(max(0, i - 4), i):
        if tokens[j] == "all":
            plural = True
    if color is not None:
        targets.append({"color": color, "all": plural})
report["targets"] = targets

STATE["t0"] = float(get_robot_state().timestamp_s)
obs0 = get_observation()
home_q = joints_now()
home_pose = get_robot_state().end_effector_poses["primary"]
home_quat = [float(a) for a in home_pose.quaternion_wxyz]
Rh = quat_to_mat(home_quat)
report["home"] = {"joints": [round(a, 4) for a in home_q],
                  "ee_pos": [round(float(a), 4) for a in home_pose.position],
                  "ee_quat": [round(a, 4) for a in home_quat],
                  "ee_z_axis": [round(Rh[0][2], 3), round(Rh[1][2], 3), round(Rh[2][2], 3)]}
if Rh[2][2] < -0.9:
    down_quat = home_quat
else:
    down_quat = [0.0, 1.0, 0.0, 0.0]
Rd = quat_to_mat(down_quat)
finger_angle0 = math.degrees(math.atan2(Rd[1][1], Rd[0][1]))
report["down_quat"] = down_quat

for cam_name in ("front", "wrist"):
    try:
        cam = obs0.cameras[cam_name]
        report["camera_" + cam_name] = {"K": [[round(float(a), 3) for a in row] for row in as_lists(cam.intrinsics, 0)],
                                        "pos": [round(float(a), 4) for a in cam.camera_pose.position],
                                        "quat": [round(float(a), 4) for a in cam.camera_pose.quaternion_wxyz],
                                        "shape": [len(cam.rgb), len(cam.rgb[0])]}
    except Exception as exc:
        report["errors"].append("camera info: " + str(exc))

# ---- conditioning video length (expected empty for this task; processed if present)
try:
    video = robomme_plus.get_conditioning_video()
    report["video_frames"] = len(video)
except Exception as exc:
    video = []
    report["errors"].append("video: " + str(exc))

# ---- calibrate deprojection per camera
for cam_name in ("front", "wrist"):
    try:
        conv, scores = calibrate_convention(obs0.cameras[cam_name])
        OBS["conv"][cam_name] = conv
        report["calib_" + cam_name] = {"chosen": conv, "scores": scores}
    except Exception as exc:
        report["errors"].append("calib " + cam_name + ": " + str(exc))
try:
    fr0 = make_frame(obs0.cameras["front"], OBS["conv"]["front"])
    zs0 = []
    for v in range(4, fr0["h"], 8):
        for u in range(4, fr0["w"], 8):
            p = world(fr0, u, v)
            if p is not None:
                zs0.append(p[2])
    OBS["table_z"] = quantile(zs0, 0.5)
    report["table_z"] = OBS["table_z"]
except Exception as exc:
    report["errors"].append("table_z: " + str(exc))

# static coloured scene content at the start is not a cube reveal
process_observation(obs0, 0, static_pass=True)
report["static_colored"] = STATIC

if len(video) > 0:
    nvid = len(video)
    stride_v = max(1, nvid // 48)
    for vi in range(0, nvid, stride_v):
        process_observation(video[vi], -1000 + vi)
    note("video_processed", {"clusters": len(CLUSTERS)})

def ik_joints(ik):
    try:
        return [float(a) for a in ik.joint_positions]
    except Exception:
        pass
    try:
        return [float(a) for a in ik.solution]
    except Exception:
        pass
    return [float(a) for a in ik.joints]


# ---- button localization
button = None
cands = []
for cam_name in ("front", "wrist"):
    if button is not None:
        break
    for prompt in ("button", "round button", "push button"):
        try:
            seg = segment_text(cam_name, prompt)
        except Exception as exc:
            report["errors"].append("seg button: " + str(exc))
            continue
        if not seg.ok:
            continue
        for cand in seg.segmentations[:5]:
            try:
                pc = mask_to_point_cloud(cand, cam_name, target_frame=BASE)
                pts = pc.points
                if len(pts) < 10:
                    continue
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
                zs = [float(p[2]) for p in pts]
                ext = max(quantile(xs, 0.95) - quantile(xs, 0.05), quantile(ys, 0.95) - quantile(ys, 0.05))
                info = {"cam": cam_name, "prompt": prompt, "score": round(float(cand.score), 3),
                        "n": len(pts), "ext": round(ext, 4),
                        "center": [quantile(xs, 0.5), quantile(ys, 0.5)],
                        "top": quantile(zs, 0.9), "zmed": quantile(zs, 0.5)}
                cands.append(info)
                tz = OBS["table_z"]
                ok_size = ext > 0.015 and ext < 0.14
                ok_z = tz is None or (info["top"] > tz - 0.01 and info["top"] < tz + 0.10)
                if ok_size and ok_z and (button is None or info["score"] > button["score"]):
                    button = info
            except Exception as exc:
                report["errors"].append("button cand: " + str(exc))
        if button is not None:
            break
report["button_candidates"] = cands[:12]
report["button"] = button

press_plan = None
if button is not None:
    bx, by = button["center"]
    btop = button["top"]
    hover = solve_ik(Pose(position=[bx, by, btop + 0.05], quaternion_wxyz=down_quat, frame=BASE),
                     seed_joints=home_q)
    press = solve_ik(Pose(position=[bx, by, btop - 0.01], quaternion_wxyz=down_quat, frame=BASE),
                     seed_joints=home_q)
    if hover.ok and press.ok:
        qh = ik_joints(hover)
        qp = ik_joints(press)
        n1 = plan_ticks(home_q, qh, 0.06)
        n2 = plan_ticks(qh, qp, 0.02)
        press_plan = {"qh": qh, "qp": qp, "n1": n1, "n2": n2}
        report["press_plan"] = {"n1": n1, "n2": n2}
    else:
        report["errors"].append("press IK failed hover=%s press=%s" % (str(hover.ok), str(press.ok)))

# ---- phase 1: observe at home, then press once the first reveal is captured
PRESS_DEADLINE = 89     # PUBLIC_RULES.md: last allowed press step for ButtonUnmask
pressed = False
if press_plan is not None:
    leave_by = PRESS_DEADLINE - (press_plan["n1"] + press_plan["n2"] + 10) - 12
else:
    leave_by = PRESS_DEADLINE
report["leave_by"] = leave_by
tick = 0
report["hz"] = STATE["hz"]
while not STATE["ended"]:
    tick = max(current_tick(), STATE["steps"])
    if OBS["first_reveal_tick"] is not None and OBS["reveal_frames"] >= 4:
        note("leave_home", {"reason": "reveal_captured"})
        break
    if tick >= leave_by:
        note("leave_home", {"reason": "deadline"})
        break
    hold_tick(home_q, 0.0)
    if STATE["steps"] % 2 == 0 and not STATE["ended"]:
        process_observation(get_observation(), current_tick())

if press_plan is not None and not STATE["ended"]:
    servo_to(press_plan["qh"], 0.0, 0.06, 2, 6)
    servo_to(press_plan["qp"], 0.0, 0.02, 2, 6)
    for k in range(4):
        if STATE["ended"]:
            break
        hold_tick(press_plan["qp"], 0.0)
    pressed = True
    try:
        ee = get_robot_state().end_effector_poses["primary"].position
        note("pressed", {"ee": [round(float(a), 4) for a in ee]})
    except Exception:
        pass
    servo_to(press_plan["qh"], 0.0, 0.03, 2, 4)
    servo_to(home_q, 0.0, 0.06, 2, 6)
if press_plan is None and button is not None and not STATE["ended"]:
    # IK pre-solve failed: blind press with the planner (no mid-motion observation)
    bx, by = button["center"]
    move_pose((bx, by, button["top"] + 0.05), down_quat, 0.01, "press:hover")
    move_pose((bx, by, button["top"] - 0.01), down_quat, 0.01, "press:down")
    pressed = True
    move_pose((bx, by, button["top"] + 0.05), down_quat, 0.01, "press:up")
    servo_to(home_q, 0.0, 0.06, 2, 6)
report["pressed"] = pressed

# ---- phase 2: observe at home until the scene is quiet
QUIET_TICKS = 40
press_done_tick = current_tick() if not STATE["ended"] else -1
cap_tick = press_done_tick + 220
if max_steps is not None:
    cap_tick = min(cap_tick, max_steps - 170 * max(1, len(targets)))
while not STATE["ended"]:
    tick = current_tick()
    last = OBS["last_seen_tick"]
    if last is not None and tick - last >= QUIET_TICKS:
        note("quiet", {"last_seen": last})
        break
    if last is None and tick - press_done_tick >= QUIET_TICKS * 2:
        note("quiet_never_seen", {})
        break
    if tick >= cap_tick:
        note("observe_cap", {"cap": cap_tick})
        break
    hold_tick(home_q, 0.0)
    if STATE["steps"] % 2 == 0 and not STATE["ended"]:
        process_observation(get_observation(), current_tick())

cubes = []
for c in CLUSTERS:
    cubes.append({"color": c["color"], "x": round(c["x"], 4), "y": round(c["y"], 4),
                  "z": round(c["z"], 4), "n": c["n"], "cams": c["cams"],
                  "first": c["first_tick"], "last": c["last_tick"]})
report["clusters"] = cubes


# ---- phase 3: lift the containers in goal order
def container_at(x, y):
    """Measure the container over (x, y) from the current wrist (then front) depth."""
    obs = get_observation()
    tz = OBS["table_z"]
    for cam_name in ("wrist", "front"):
        conv = OBS["conv"].get(cam_name)
        if conv is None:
            continue
        frame = make_frame(obs.cameras[cam_name], conv)
        pts = []
        for v in range(0, frame["h"], 2):
            for u in range(0, frame["w"], 2):
                p = world(frame, u, v)
                if p is None:
                    continue
                if p[2] < tz + 0.012 or p[2] > tz + 0.15:
                    continue
                if math.hypot(p[0] - x, p[1] - y) > 0.06:
                    continue
                pts.append(p)
        if len(pts) < 15:
            continue
        top = quantile([p[2] for p in pts], 0.9)
        cx, cy = x, y
        face = []
        for it in range(2):
            face = [p for p in pts if p[2] > top - 0.012 and math.hypot(p[0] - cx, p[1] - cy) < 0.035]
            if len(face) < 8:
                break
            cx = sum(p[0] for p in face) / len(face)
            cy = sum(p[1] for p in face) / len(face)
        if len(face) < 8:
            continue
        best = None
        for deg in range(0, 90, 5):
            th = math.radians(deg)
            ca = math.cos(th)
            sa = math.sin(th)
            a1 = [p[0] * ca + p[1] * sa for p in face]
            a2 = [-p[0] * sa + p[1] * ca for p in face]
            lo1 = quantile(a1, 0.03)
            hi1 = quantile(a1, 0.97)
            lo2 = quantile(a2, 0.03)
            hi2 = quantile(a2, 0.97)
            area = (hi1 - lo1) * (hi2 - lo2)
            if best is None or area < best["area"]:
                m1 = 0.5 * (lo1 + hi1)
                m2 = 0.5 * (lo2 + hi2)
                best = {"area": area, "deg": deg, "w1": hi1 - lo1, "w2": hi2 - lo2,
                        "cx": m1 * ca - m2 * sa, "cy": m1 * sa + m2 * ca}
        return {"cam": cam_name, "n": len(pts), "face_n": len(face), "top": top,
                "cx": best["cx"], "cy": best["cy"], "yaw_deg": best["deg"],
                "w1": best["w1"], "w2": best["w2"]}
    return None


def lift_container(cube, label):
    tz = OBS["table_z"]
    x = cube["x"]
    y = cube["y"]
    if not move_pose((x, y, tz + 0.30), down_quat, 0.015, label + ":observe"):
        if STATE["ended"]:
            return {"ok": False, "why": "ended"}
    cont = container_at(x, y)
    note("container", {"label": label, "cont": cont})
    if cont is None:
        return {"ok": False, "why": "no container measured"}
    # A partially seen top face biases the face centre; then trust the remembered cube position.
    if min(cont["w1"], cont["w2"]) < 0.7 * max(cont["w1"], cont["w2"]):
        cont["center_source"] = "cube"
        cont["cx"] = x
        cont["cy"] = y
    else:
        cont["center_source"] = "face"
    delta = wrap90(cont["yaw_deg"] - finger_angle0)
    q = quat_mul([math.cos(math.radians(delta) / 2), 0.0, 0.0, math.sin(math.radians(delta) / 2)], down_quat)
    height = cont["top"] - tz
    depth = min(0.025, max(0.01, 0.45 * height))
    gz = cont["top"] - depth
    cx = cont["cx"]
    cy = cont["cy"]
    gripper_call(True, label + ":open")
    if not move_pose((cx, cy, cont["top"] + 0.07), q, 0.01, label + ":pregrasp"):
        if STATE["ended"]:
            return {"ok": False, "why": "ended"}
    if not move_pose((cx, cy, gz), q, 0.008, label + ":grasp"):
        if STATE["ended"]:
            return {"ok": False, "why": "ended"}
    g = gripper_call(False, label + ":close")
    held = g is not None and g > 0.03
    if not held and not STATE["ended"]:
        # missed: open, back off, re-measure once from above and retry
        gripper_call(True, label + ":reopen")
        move_pose((cx, cy, cont["top"] + 0.07), q, 0.01, label + ":backoff")
        move_pose((x, y, tz + 0.30), down_quat, 0.015, label + ":reobserve")
        cont2 = container_at(x, y)
        note("container_retry", {"label": label, "cont": cont2})
        if cont2 is not None and not STATE["ended"]:
            cx = cont2["cx"]
            cy = cont2["cy"]
            gz = cont2["top"] - depth
            move_pose((cx, cy, cont2["top"] + 0.07), q, 0.01, label + ":pregrasp2")
            move_pose((cx, cy, gz), q, 0.008, label + ":grasp2")
            g = gripper_call(False, label + ":close2")
            held = g is not None and g > 0.03
    move_pose((cx, cy, gz + 0.10), q, 0.012, label + ":lift")
    for k in range(8):
        if STATE["ended"]:
            break
        hold_tick(joints_now(), 0.0)
    if STATE["ended"]:
        return {"ok": True, "held": held, "why": "ended after lift", "cont": cont}
    # diagnostic: is the hidden colour now visible near the remembered cube position?
    revealed = []
    try:
        ob = get_observation()
        for cam_name in ("front", "wrist"):
            fr = make_frame(ob.cameras[cam_name], OBS["conv"][cam_name])
            for blob in detect_blobs(fr, 2, 3):
                dxy = math.hypot(blob["xyz"][0] - x, blob["xyz"][1] - y)
                if dxy < 0.08:
                    revealed.append([cam_name, blob["color"], round(dxy, 3), blob["cells"]])
    except Exception as exc:
        revealed.append("err " + str(exc))
    note("reveal_check", {"label": label, "near": revealed})
    # put it back where it was and release
    move_pose((cx, cy, gz + 0.006), q, 0.01, label + ":lower")
    gripper_call(True, label + ":release")
    move_pose((cx, cy, cont["top"] + 0.08), q, 0.015, label + ":retreat")
    return {"ok": True, "held": held, "grip": g, "cont": cont}


def neg_count(c):
    return -c["n"]


picks = []
for ti in range(len(targets)):
    if STATE["ended"]:
        break
    spec = targets[ti]
    pool = [c for c in cubes if c["color"] == spec["color"] and c["n"] >= 2]
    pool = sorted(pool, key=neg_count)
    if not spec["all"]:
        pool = pool[:1]
    else:
        distinct = []
        for c in pool:
            dup = False
            for d in distinct:
                if math.hypot(c["x"] - d["x"], c["y"] - d["y"]) < 0.06:
                    dup = True
            if not dup:
                distinct.append(c)
        pool = distinct
    if len(pool) == 0:
        picks.append({"target": spec, "ok": False, "why": "colour never observed"})
        continue
    for cube in pool:
        if STATE["ended"]:
            break
        r = lift_container(cube, spec["color"] + str(ti))
        r["target"] = spec
        r["cube"] = cube
        picks.append(r)
report["picks"] = picks
report["steps_by_program"] = STATE["steps"]
try:
    if not STATE["ended"]:
        report["final_tick"] = current_tick()
except Exception:
    pass
result = report
