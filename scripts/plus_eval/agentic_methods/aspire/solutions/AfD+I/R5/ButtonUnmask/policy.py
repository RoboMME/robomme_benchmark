# ButtonUnmask v00 -- initial program from offline analysis of the released recordings.
# Strategy (all task-dependent quantities derived at runtime):
#   1. parse the ordered colour list from get_task_context().language
#   2. hold the start pose, watch front+wrist RGB-D every tick, track coloured cubes in 3D
#      (pure-Python colour blobs + depth back-projection with the published calibration)
#   3. press the button (located with SAM + depth) with a closed gripper before step 90
#   4. return to the start pose and keep watching until the reveals stop
#   5. for each goal item: find the container sitting on the remembered cube position from
#      depth, top-down grasp it, lift it, put it back, release.
import math

S = {"tick": 0, "done": False, "term": None, "log": []}
REPORT = {"version": "v02"}

CTX = get_task_context()
GOAL = str(CTX.language)
REPORT["goal"] = GOAL
try:
    META = robomme_plus.get_task_metadata()
    MAX_STEPS = int(META["max_steps"])
    REPORT["meta"] = {"max_steps": MAX_STEPS, "control_hz": META.get("control_hz", None),
                      "sampling": str(META.get("conditioning_video_sampling", None))}
except Exception as exc:
    MAX_STEPS = 1000
    REPORT["meta_error"] = str(exc)

ST0 = get_robot_state()
EMB = ST0.embodiment
BASE = ST0.base_frame
Q0 = [float(x) for x in ST0.joint_positions["primary"]]
EE0 = ST0.end_effector_poses["primary"]
P0 = [float(x) for x in EE0.position]
QUAT0 = [float(x) for x in EE0.quaternion_wxyz]
REPORT["start_ee"] = [round(x, 4) for x in P0]
REPORT["start_quat"] = [round(x, 4) for x in QUAT0]

# ---------------------------------------------------------------- math helpers
def quat_to_r(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]

def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]

def quat_z(angle):
    return [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]

def wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

def quantile(vals, f):
    o = sorted(vals)
    if len(o) == 0:
        return None
    return o[int((len(o) - 1) * f)]

def dist2(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def to_list(arr):
    try:
        return arr.tolist()
    except Exception:
        return [[[int(c) for c in px] for px in row] for row in arr]

def to_list2(arr):
    try:
        return arr.tolist()
    except Exception:
        return [[float(c) for c in row] for row in arr]

# ---------------------------------------------------------------- camera projection
CONV = {"name": "cv"}

def cam_params(cam):
    k = cam.intrinsics
    pose = cam.camera_pose
    return {"fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]), "cy": float(k[1][2]),
            "r": quat_to_r([float(x) for x in pose.quaternion_wxyz]),
            "t": [float(x) for x in pose.position]}

def project(u, v, d, p, conv=None):
    if conv is None:
        conv = CONV["name"]
    xc = (u - p["cx"]) * d / p["fx"]
    yc = (v - p["cy"]) * d / p["fy"]
    if conv == "cv":
        c = (xc, yc, d)
    elif conv == "gl":
        c = (xc, -yc, -d)
    else:
        c = (d, -xc, -yc)
    r = p["r"]
    t = p["t"]
    return [r[0][0] * c[0] + r[0][1] * c[1] + r[0][2] * c[2] + t[0],
            r[1][0] * c[0] + r[1][1] * c[1] + r[1][2] * c[2] + t[1],
            r[2][0] * c[0] + r[2][1] * c[1] + r[2][2] * c[2] + t[2]]

def world_to_pix(pt, p, conv=None):
    if conv is None:
        conv = CONV["name"]
    r = p["r"]
    t = p["t"]
    d = [pt[0] - t[0], pt[1] - t[1], pt[2] - t[2]]
    c = [r[0][0] * d[0] + r[1][0] * d[1] + r[2][0] * d[2],
         r[0][1] * d[0] + r[1][1] * d[1] + r[2][1] * d[2],
         r[0][2] * d[0] + r[1][2] * d[1] + r[2][2] * d[2]]
    if conv == "cv":
        xc, yc, z = c[0], c[1], c[2]
    elif conv == "gl":
        xc, yc, z = c[0], -c[1], -c[2]
    else:
        xc, yc, z = -c[1], -c[2], c[0]
    if z <= 1e-6:
        return None
    return [p["fx"] * xc / z + p["cx"], p["fy"] * yc / z + p["cy"]]

def depth_points(cam, stride, conv=None):
    p = cam_params(cam)
    dep = to_list2(cam.depth_m)
    pts = []
    for v in range(0, len(dep), stride):
        row = dep[v]
        for u in range(0, len(row), stride):
            d = row[u]
            if d is None or d != d or d <= 0.03 or d > 4.0:
                continue
            pts.append(project(u, v, d, p, conv))
    return pts, p

# choose the camera-frame convention from front-camera table planarity (oblique view)
OBS0 = get_observation()
CAMS = list(OBS0.cameras.keys())
REPORT["cameras"] = CAMS
conv_scores = {}
for conv in ("cv", "gl", "ros"):
    try:
        cam = OBS0.cameras["front"]
        p = cam_params(cam)
        dep = to_list2(cam.depth_m)
        h = len(dep)
        zs = []
        for v in range(int(h * 0.45), h, 8):
            for u in range(0, len(dep[v]), 8):
                d = dep[v][u]
                if d > 0.03 and d < 4.0:
                    zs.append(project(u, v, d, p, conv)[2])
        spread = quantile(zs, 0.9) - quantile(zs, 0.1)
        # forward axis must point down
        fwd = {"cv": p["r"][2][2], "gl": -p["r"][2][2], "ros": p["r"][2][0]}[conv]
        conv_scores[conv] = [round(spread, 4), round(fwd, 3), round(quantile(zs, 0.5), 4)]
    except Exception as exc:
        conv_scores[conv] = [99.0, 0.0, str(exc)]
best_conv = None
for conv in conv_scores:
    sc = conv_scores[conv]
    if sc[1] < 0 and (best_conv is None or sc[0] < conv_scores[best_conv][0]):
        best_conv = conv
if best_conv is None:
    best_conv = "cv"
CONV["name"] = best_conv
REPORT["conv_scores"] = conv_scores
REPORT["conv"] = best_conv

# table height from the start wrist + front views (table dominates)
wpts, wp0 = depth_points(OBS0.cameras["wrist"], 6)
fpts, fp0 = depth_points(OBS0.cameras["front"], 6)
TABLE_Z = quantile([q[2] for q in wpts], 0.5)
REPORT["table_z_wrist"] = round(TABLE_Z, 4)
REPORT["table_z_front"] = round(quantile([q[2] for q in fpts], 0.5), 4)

# ---------------------------------------------------------------- goal parsing
COLOR_WORDS = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "white", "black",
               "cyan", "brown", "gray", "grey"]
DETECTABLE = ["red", "green", "blue", "yellow", "purple", "orange"]
text = GOAL.lower()
bi = text.find("button")
rest = text[bi:] if bi >= 0 else text
words = rest.replace(",", " ").replace(".", " ").replace(";", " ").split()
ITEMS = []
last_i = 0
for i in range(len(words)):
    w = words[i]
    if w in COLOR_WORDS:
        window = words[last_i:i]
        nxt = words[i + 1] if i + 1 < len(words) else ""
        all_tag = ("all" in window) or (nxt == "cubes")
        ITEMS.append({"color": w, "all": all_tag})
        last_i = i + 1
REPORT["items"] = ITEMS

# ---------------------------------------------------------------- colour blobs
def classify_px(r, g, b):
    if b > 110 and b > 2 * g and b > 2 * r:
        return "blue"
    if g > 100 and g > r + 50 and g > b + 50:
        return "green"
    if r > 100 and g < 0.3 * r and b < 0.3 * r:
        return "red"
    if r > 120 and g > 0.82 * r and b < 0.5 * g:
        return "yellow"
    if b > 100 and r > g + 40 and b > g + 40:
        return "purple"
    if r > 160 and b < 0.12 * r and g > 0.5 * r and g < 0.78 * r:
        return "orange"
    return None

def detect_blobs(rgb, stride, min_cells):
    h = len(rgb)
    w = len(rgb[0])
    grid = {}
    for v in range(0, h, stride):
        row = rgb[v]
        for u in range(0, w, stride):
            px = row[u]
            c = classify_px(px[0], px[1], px[2])
            if c is not None:
                grid[(v // stride, u // stride)] = c
    seen = set()
    blobs = []
    for key in list(grid.keys()):
        if key in seen:
            continue
        color = grid[key]
        stack = [key]
        seen.add(key)
        cells = []
        while len(stack) > 0:
            cur = stack.pop()
            cells.append(cur)
            for dv in (-1, 0, 1):
                for du in (-1, 0, 1):
                    nb = (cur[0] + dv, cur[1] + du)
                    if nb not in seen and nb in grid and grid[nb] == color:
                        seen.add(nb)
                        stack.append(nb)
        if len(cells) >= min_cells:
            blobs.append({"color": color, "cells": [(c[0] * stride, c[1] * stride) for c in cells]})
    return blobs

TRACKS = []
VIS = {"last_visible": -1, "first_visible": -1, "frames": 0, "empty_streak": 0, "ever": False,
       "summary": []}

def add_detection(color, pt, tick, cam_name, n):
    best = None
    bd = 0.035
    for tr in TRACKS:
        if tr["color"] != color:
            continue
        d = dist2(tr["pos"], pt)
        if d < bd:
            bd = d
            best = tr
    if best is None:
        best = {"color": color, "sx": 0.0, "sy": 0.0, "sz": 0.0, "w": 0.0, "pos": pt, "obs": 0,
                "first": tick, "last": tick, "cams": [], "cells": 0}
        TRACKS.append(best)
    wgt = float(n)
    best["sx"] += pt[0] * wgt
    best["sy"] += pt[1] * wgt
    best["sz"] += pt[2] * wgt
    best["w"] += wgt
    best["pos"] = [best["sx"] / best["w"], best["sy"] / best["w"], best["sz"] / best["w"]]
    best["obs"] += 1
    best["last"] = tick
    best["cells"] = max(best["cells"], n)
    if cam_name not in best["cams"]:
        best["cams"].append(cam_name)

def process_obs(obs, tick):
    total = 0
    per = {}
    for cam_name, stride, minc in (("front", 2, 4), ("wrist", 3, 4)):
        try:
            cam = obs.cameras[cam_name]
        except Exception:
            continue
        rgb = to_list(cam.rgb)
        blobs = detect_blobs(rgb, stride, minc)
        if len(blobs) == 0:
            per[cam_name] = 0
            continue
        p = cam_params(cam)
        dep = to_list2(cam.depth_m)
        cnt = 0
        for b in blobs:
            xs = []
            ys = []
            zs = []
            for c in b["cells"]:
                d = dep[c[0]][c[1]]
                if d is None or d != d or d <= 0.03 or d > 4.0:
                    continue
                q = project(c[1], c[0], d, p)
                xs.append(q[0])
                ys.append(q[1])
                zs.append(q[2])
            if len(xs) < 3:
                continue
            pt = [quantile(xs, 0.5), quantile(ys, 0.5), quantile(zs, 0.5)]
            if pt[2] < TABLE_Z - 0.03 or pt[2] > TABLE_Z + 0.08:
                continue
            add_detection(b["color"], pt, tick, cam_name, len(b["cells"]))
            cnt += 1
        per[cam_name] = cnt
        total += cnt
    VIS["frames"] += 1
    if total > 0:
        VIS["last_visible"] = tick
        if VIS["first_visible"] < 0:
            VIS["first_visible"] = tick
        VIS["ever"] = True
        VIS["empty_streak"] = 0
    else:
        VIS["empty_streak"] += 1
    if len(VIS["summary"]) < 400:
        VIS["summary"].append([tick, per.get("front", -1), per.get("wrist", -1)])
    return total

# conditioning video (may be empty for this task)
try:
    VIDEO = robomme_plus.get_conditioning_video()
    REPORT["video_frames"] = len(VIDEO)
    vi = 0
    for fr in VIDEO:
        if vi % 2 == 0:
            process_obs(fr, -1000 + vi)
        vi += 1
except Exception as exc:
    REPORT["video_error"] = str(exc)

# ---------------------------------------------------------------- stepping helpers
def obs_after(res):
    try:
        o = res.observation
        if o is not None:
            return o
    except Exception:
        pass
    return None

def do_tick(q, grip, watch):
    if S["done"]:
        return None
    act = RobotAction(arms={"primary": ArmCommand(mode="joint_position", target=[float(x) for x in q],
                                                  gripper_position=float(grip), embodiment=EMB)})
    res = step(act)
    S["tick"] += 1
    term = False
    trunc = False
    try:
        term = bool(res.terminated)
        trunc = bool(res.truncated)
    except Exception:
        pass
    if term or trunc:
        S["done"] = True
        S["term"] = {"tick": S["tick"], "terminated": term, "truncated": trunc}
        return None
    o = obs_after(res)
    if watch and o is not None and S["tick"] % 2 == 0:
        process_obs(o, S["tick"])
    return o

def cur_q():
    return [float(x) for x in get_robot_state().joint_positions["primary"]]

DQ_MAX = 0.04

def run_joint_path(qs, grip, watch, lin_speed=None, pts=None):
    """Step through joint waypoints with per-tick interpolation."""
    qa = cur_q()
    for k in range(len(qs)):
        qb = qs[k]
        dq = max(abs(qb[j] - qa[j]) for j in range(7))
        n = int(math.ceil(dq / DQ_MAX))
        if lin_speed is not None and pts is not None and len(pts) == len(qs) + 1:
            n = max(n, int(math.ceil(dist3(pts[k], pts[k + 1]) / lin_speed)))
        n = max(n, 1)
        for i in range(1, n + 1):
            f = float(i) / n
            do_tick([qa[j] + (qb[j] - qa[j]) * f for j in range(7)], grip, watch)
            if S["done"]:
                return False
        qa = qb
    return True

def settle(q, grip, watch, max_ticks, tol):
    for i in range(max_ticks):
        if S["done"]:
            return
        cq = cur_q()
        if max(abs(cq[j] - q[j]) for j in range(7)) < tol:
            return
        do_tick(q, grip, watch)

def dist3(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

def ik(pos, quat, seed):
    r = solve_ik(Pose([float(x) for x in pos], [float(x) for x in quat], BASE), seed_joints=[float(x) for x in seed])
    if not r.ok:
        return None
    return [float(x) for x in r.joint_positions]

def plan_line(p_from, p_to, quat_from, quat_to, seed, step_len):
    """IK chain along a straight Cartesian line (orientation nlerp)."""
    dd = dist3(p_from, p_to)
    n = max(1, int(math.ceil(dd / step_len)))
    qs = []
    pts = [list(p_from)]
    q = seed
    dot = sum(quat_from[i] * quat_to[i] for i in range(4))
    qt = quat_to if dot >= 0 else [-c for c in quat_to]
    for k in range(1, n + 1):
        f = float(k) / n
        p = [p_from[i] + (p_to[i] - p_from[i]) * f for i in range(3)]
        qq = [quat_from[i] + (qt[i] - quat_from[i]) * f for i in range(4)]
        nq = math.sqrt(sum(c * c for c in qq))
        qq = [c / nq for c in qq]
        sol = ik(p, qq, q)
        if sol is None:
            return None, None
        qs.append(sol)
        pts.append(p)
        q = sol
    return qs, pts

def ee_now():
    st = get_robot_state()
    return [float(x) for x in st.end_effector_poses["primary"].position]

def ee_quat_now():
    st = get_robot_state()
    return [float(x) for x in st.end_effector_poses["primary"].quaternion_wxyz]

def grip_now():
    return float(get_robot_state().gripper_positions["primary"])

# ---------------------------------------------------------------- button localization
def cloud_stats(points):
    xs = [float(q[0]) for q in points]
    ys = [float(q[1]) for q in points]
    zs = [float(q[2]) for q in points]
    return {"x": quantile(xs, 0.5), "y": quantile(ys, 0.5), "ztop": quantile(zs, 0.9),
            "zlow": quantile(zs, 0.1), "ex": quantile(xs, 0.95) - quantile(xs, 0.05),
            "ey": quantile(ys, 0.95) - quantile(ys, 0.05), "n": len(xs)}

BUTTON = None
btn_log = []
for cam_name in ("wrist", "front"):
    if BUTTON is not None:
        break
    for prompt in ("button", "round button", "push button"):
        if BUTTON is not None:
            break
        try:
            seg = segment_text(cam_name, prompt)
        except Exception as exc:
            btn_log.append([cam_name, prompt, "err " + str(exc)])
            continue
        if not seg.ok:
            btn_log.append([cam_name, prompt, "not ok"])
            continue
        best = None
        for cand in seg.segmentations[:5]:
            try:
                pc = mask_to_point_cloud(cand, cam_name, target_frame=BASE)
                pts = pc.points
                if len(pts) < 15:
                    btn_log.append([cam_name, prompt, "few", len(pts)])
                    continue
                stt = cloud_stats(pts)
                hgt = stt["ztop"] - TABLE_Z
                entry = [cam_name, prompt, round(float(cand.score), 4), round(stt["x"], 4), round(stt["y"], 4),
                         round(stt["ztop"], 4), round(stt["ex"], 3), round(stt["ey"], 3), stt["n"]]
                try:
                    bx = [float(c) for c in cand.box_xyxy]
                    uv = [(bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0]
                    cam = OBS0.cameras[cam_name]
                    dd = float(cam.depth_m[int(uv[1])][int(uv[0])])
                    mine = project(uv[0], uv[1], dd, cam_params(cam))
                    entry.append("mine")
                    entry.append([round(c, 4) for c in mine])
                    for conv in ("cv", "gl", "ros"):
                        alt = project(uv[0], uv[1], dd, cam_params(cam), conv)
                        entry.append(conv + ":" + str(round(dist2(alt, [stt["x"], stt["y"]]), 4)))
                except Exception as exc:
                    entry.append("cmp err " + str(exc))
                btn_log.append(entry)
                if stt["ex"] < 0.12 and stt["ey"] < 0.12 and hgt > -0.01 and hgt < 0.08:
                    if best is None or float(cand.score) > best[0]:
                        best = [float(cand.score), stt]
            except Exception as exc:
                btn_log.append([cam_name, prompt, "cand err " + str(exc)])
        if best is not None:
            BUTTON = best[1]
            BUTTON["cam"] = cam_name
REPORT["button_log"] = btn_log
REPORT["button"] = None if BUTTON is None else {k: (round(BUTTON[k], 4) if k != "cam" and k != "n" else BUTTON[k]) for k in BUTTON}

# fallback: raised compact cluster from wrist depth
if BUTTON is None:
    raised = [q for q in wpts if q[2] > TABLE_Z + 0.004 and q[2] < TABLE_Z + 0.06]
    if len(raised) > 5:
        cx = quantile([q[0] for q in raised], 0.5)
        cy = quantile([q[1] for q in raised], 0.5)
        near = [q for q in raised if dist2(q, [cx, cy]) < 0.05]
        if len(near) > 3:
            BUTTON = cloud_stats(near)
            BUTTON["cam"] = "depth"
    REPORT["button_fallback"] = BUTTON is not None

# ---------------------------------------------------------------- phase A: watch + press
DOWN_QUAT = QUAT0
press_plan = None
if BUTTON is not None:
    bz = BUTTON["ztop"]
    pre = [BUTTON["x"], BUTTON["y"], bz + 0.05]
    prs = [BUTTON["x"], BUTTON["y"], bz - 0.016]
    qs1, pts1 = plan_line(P0, pre, QUAT0, DOWN_QUAT, Q0, 0.04)
    if qs1 is not None:
        qs2, pts2 = plan_line(pre, prs, DOWN_QUAT, DOWN_QUAT, qs1[-1], 0.02)
        if qs2 is not None:
            press_plan = {"down": qs1, "down_pts": pts1, "press": qs2, "press_pts": pts2}
REPORT["press_planned"] = press_plan is not None

def ticks_for(qs, pts, speed):
    qa = Q0
    tot = 0
    for k in range(len(qs)):
        dq = max(abs(qs[k][j] - qa[j]) for j in range(7))
        n = int(math.ceil(dq / DQ_MAX))
        n = max(n, int(math.ceil(dist3(pts[k], pts[k + 1]) / speed)), 1)
        tot += n
        qa = qs[k]
    return tot

GRIP_CLOSED = 0.0
GRIP_OPEN = 1.0
press_info = {}
if press_plan is not None:
    n_down = ticks_for(press_plan["down"], press_plan["down_pts"], 0.015)
    n_press = 10
    latest_start = max(0, 76 - n_down - n_press)
    press_info["n_down"] = n_down
    press_info["latest_start"] = latest_start
    # hold & watch round 1 from the start pose
    while S["tick"] < latest_start and not S["done"]:
        do_tick(Q0, GRIP_CLOSED, True)
        if VIS["ever"] and VIS["empty_streak"] >= 2 and S["tick"] >= 20:
            break
    press_info["descent_start"] = S["tick"]
    run_joint_path(press_plan["down"], GRIP_CLOSED, True, 0.015, press_plan["down_pts"])
    # slow press
    qa = press_plan["down"][-1]
    qb = press_plan["press"][-1]
    for i in range(1, n_press + 1):
        f = float(i) / n_press
        do_tick([qa[j] + (qb[j] - qa[j]) * f for j in range(7)], GRIP_CLOSED, True)
    for i in range(6):
        do_tick(qb, GRIP_CLOSED, True)
    press_info["press_tick"] = S["tick"]
    try:
        press_info["ee_at_press"] = [round(c, 4) for c in ee_now()]
    except Exception:
        pass
    # retreat to the start pose along the same path
    back = list(reversed(press_plan["down"]))[1:] + [Q0]
    back_pts = list(reversed(press_plan["down_pts"]))
    run_joint_path([qa] + back, GRIP_CLOSED, True, 0.02, [prs] + back_pts)
    settle(Q0, GRIP_CLOSED, True, 6, 0.01)
    press_info["back_tick"] = S["tick"]
REPORT["press"] = press_info

# keep watching until the reveals stop
while not S["done"]:
    t = S["tick"]
    if t >= 260:
        break
    if t >= 112 and t >= VIS["last_visible"] + 26 and VIS["empty_streak"] >= 3:
        break
    do_tick(Q0, GRIP_CLOSED, True)
REPORT["watch_end_tick"] = S["tick"]
REPORT["vis"] = {"first": VIS["first_visible"], "last": VIS["last_visible"], "frames": VIS["frames"]}
REPORT["vis_summary"] = VIS["summary"]

# merge duplicate tracks (same colour within 4 cm)
merged = []
def neg_w(t):
    return -t["w"]

for tr in sorted(TRACKS, key=neg_w):
    dup = None
    for m in merged:
        if m["color"] == tr["color"] and dist2(m["pos"], tr["pos"]) < 0.04:
            dup = m
            break
    if dup is None:
        merged.append(tr)
    else:
        tw = dup["w"] + tr["w"]
        dup["pos"] = [(dup["pos"][i] * dup["w"] + tr["pos"][i] * tr["w"]) / tw for i in range(3)]
        dup["w"] = tw
        dup["obs"] += tr["obs"]
        dup["first"] = min(dup["first"], tr["first"])
        dup["last"] = max(dup["last"], tr["last"])
        for c in tr["cams"]:
            if c not in dup["cams"]:
                dup["cams"].append(c)
REPORT["tracks"] = [{"color": t["color"], "pos": [round(c, 4) for c in t["pos"]], "obs": t["obs"],
                     "first": t["first"], "last": t["last"], "cams": t["cams"], "cells": t["cells"]}
                    for t in merged]

# ---------------------------------------------------------------- container localization
def bin_at(xy, cams_obs, radius):
    """Container top geometry around xy from the wrist depth of the current view."""
    pts = []
    for cam_name, stride in cams_obs:
        try:
            cam = get_observation().cameras[cam_name]
        except Exception:
            continue
        p = cam_params(cam)
        cam_pos = p["t"]
        dep = to_list2(cam.depth_m)
        for v in range(0, len(dep), stride):
            row = dep[v]
            for u in range(0, len(row), stride):
                d = row[u]
                if d <= 0.03 or d > 4.0:
                    continue
                q = project(u, v, d, p)
                if q[2] < TABLE_Z + 0.012 or q[2] > TABLE_Z + 0.14:
                    continue
                if dist3(q, cam_pos) < 0.12:
                    continue
                if dist2(q, xy) < radius:
                    pts.append(q)
    if len(pts) < 8:
        return None
    ztop = quantile([q[2] for q in pts], 0.92)
    top = [q for q in pts if q[2] > ztop - 0.012]
    cx = sum(q[0] for q in top) / len(top)
    cy = sum(q[1] for q in top) / len(top)
    top2 = [q for q in top if dist2(q, [cx, cy]) < 0.035]
    if len(top2) >= 5:
        top = top2
    best = None
    for k in range(0, 90, 3):
        a = math.radians(k)
        ca = math.cos(a)
        sa = math.sin(a)
        us = [q[0] * ca + q[1] * sa for q in top]
        vs = [-q[0] * sa + q[1] * ca for q in top]
        u0 = quantile(us, 0.03)
        u1 = quantile(us, 0.97)
        v0 = quantile(vs, 0.03)
        v1 = quantile(vs, 0.97)
        area = (u1 - u0) * (v1 - v0)
        if best is None or area < best[0]:
            um = (u0 + u1) / 2.0
            vm = (v0 + v1) / 2.0
            best = [area, a, um * ca - vm * sa, um * sa + vm * ca, u1 - u0, v1 - v0]
    return {"x": best[2], "y": best[3], "ztop": ztop, "yaw": best[1], "sx": best[4], "sy": best[5],
            "n": len(pts), "ntop": len(top)}

# hand closing axis yaw at the start orientation
R0 = quat_to_r(QUAT0)
PHI0 = math.atan2(R0[1][1], R0[0][1])

def grasp_quat(bin_yaw):
    best = None
    for k in range(-2, 3):
        cand = bin_yaw + k * math.pi / 2.0
        dlt = wrap_pi(cand - PHI0)
        if best is None or abs(dlt) < abs(best):
            best = dlt
    return quat_mul(quat_z(best), QUAT0), best

# ---------------------------------------------------------------- phase B: pick containers
picks = []
used = []
used_bins = []

def choose_targets(item):
    cands = [t for t in merged if t["color"] == item["color"] and t["obs"] >= 2]
    cands = [t for t in cands if all(dist2(t["pos"], u) > 0.03 for u in used)]
    if len(cands) == 0:
        return []
    if item["all"]:
        return cands
    cands = sorted(cands, key=neg_w)
    return [cands[0]]

LIFT_H = 0.16
LIFT_HOLD = 4
AT_HIGH = {"v": True}
LAST_ZTOP = {"v": None}

def pick_container(track):
    info = {"color": track["color"], "track": [round(c, 4) for c in track["pos"]], "tick0": S["tick"]}
    picks.append(info)
    xy = [track["pos"][0], track["pos"][1]]
    # v02: only trust the wide-view localisation from the high start pose
    b = bin_at(xy, (("wrist", 2),), 0.045) if AT_HIGH["v"] else None
    info["bin_high"] = None if b is None else {k: round(b[k], 4) for k in b}
    if b is None:
        zt = LAST_ZTOP["v"] if LAST_ZTOP["v"] is not None else TABLE_Z + 0.05
        b = {"x": xy[0], "y": xy[1], "ztop": zt, "yaw": 0.0}
    for ub in used_bins:
        if dist2(ub, [b["x"], b["y"]]) < 0.03:
            info["err"] = "container already lifted"
            return True
    gq, dyaw = grasp_quat(b["yaw"])
    hover = [b["x"], b["y"], b["ztop"] + 0.14]
    here = ee_now()
    qstart = QUAT0
    if not AT_HIGH["v"]:
        try:
            qstart = ee_quat_now()
        except Exception:
            qstart = QUAT0
    qs, pts = plan_line(here, hover, qstart, gq, cur_q(), 0.04)
    if qs is None:
        info["err"] = "ik hover"
        return False
    run_joint_path(qs, GRIP_OPEN, False, 0.02, pts)
    settle(qs[-1], GRIP_OPEN, False, 8, 0.01)
    if S["done"]:
        return False
    b2 = bin_at([b["x"], b["y"]], (("wrist", 2),), 0.04)
    info["bin_hover"] = None if b2 is None else {k: round(b2[k], 4) for k in b2}
    if b2 is not None and dist2([b2["x"], b2["y"]], [b["x"], b["y"]]) < 0.03:
        b = b2
        gq, dyaw = grasp_quat(b["yaw"])
        LAST_ZTOP["v"] = b["ztop"]
    info["dyaw"] = round(dyaw, 3)
    hgt = b["ztop"] - TABLE_Z
    gz = max(TABLE_Z + 0.012, b["ztop"] - min(0.025, hgt * 0.5))
    hover = [b["x"], b["y"], b["ztop"] + 0.10]
    grasp = [b["x"], b["y"], gz]
    info["grasp"] = [round(c, 4) for c in grasp]
    qs, pts = plan_line(ee_now(), hover, quat_mul([1, 0, 0, 0], gq), gq, cur_q(), 0.04)
    if qs is None:
        info["err"] = "ik hover2"
        return False
    run_joint_path(qs, GRIP_OPEN, False, 0.02, pts)
    qs, pts = plan_line(hover, grasp, gq, gq, qs[-1], 0.02)
    if qs is None:
        info["err"] = "ik grasp"
        return False
    run_joint_path(qs, GRIP_OPEN, False, 0.008, pts)
    settle(qs[-1], GRIP_OPEN, False, 6, 0.008)
    qg = qs[-1]
    for i in range(14):
        do_tick(qg, GRIP_CLOSED, False)
    if S["done"]:
        return False
    info["grip_after_close"] = round(grip_now(), 3)
    # v01: lift high (expert lifts ~0.15-0.2 m) and hold it there before putting back
    lift = [grasp[0], grasp[1], grasp[2] + LIFT_H]
    qs, pts = plan_line(grasp, lift, gq, gq, qg, 0.02)
    if qs is None:
        lift = [grasp[0], grasp[1], grasp[2] + 0.14]
        qs, pts = plan_line(grasp, lift, gq, gq, qg, 0.02)
    if qs is None:
        info["err"] = "ik lift"
        return False
    run_joint_path(qs, GRIP_CLOSED, False, 0.012, pts)
    for i in range(LIFT_HOLD):
        do_tick(qs[-1], GRIP_CLOSED, False)
        if S["done"]:
            break
    if S["done"]:
        return False
    info["ee_lift"] = [round(c, 4) for c in ee_now()]
    info["grip_lift"] = round(grip_now(), 3)
    # put it back where it was
    down = [grasp[0], grasp[1], grasp[2] + 0.004]
    qs, pts = plan_line(lift, down, gq, gq, qs[-1], 0.02)
    if qs is None:
        info["err"] = "ik lower"
        return False
    run_joint_path(qs, GRIP_CLOSED, False, 0.015, pts)
    settle(qs[-1], GRIP_CLOSED, False, 4, 0.006)
    for i in range(10):
        do_tick(qs[-1], GRIP_OPEN, False)
    up = [grasp[0], grasp[1], b["ztop"] + 0.12]
    qs, pts = plan_line(down, up, gq, gq, qs[-1], 0.03)
    if qs is not None:
        run_joint_path(qs, GRIP_OPEN, False, 0.02, pts)
    info["tick1"] = S["tick"]
    used.append([track["pos"][0], track["pos"][1]])
    used_bins.append([b["x"], b["y"]])
    return True

ok_all = True
for item in ITEMS:
    if S["done"]:
        break
    targets = choose_targets(item)
    if len(targets) == 0:
        picks.append({"color": item["color"], "err": "no track"})
        ok_all = False
        continue
    remaining = list(targets)
    while len(remaining) > 0 and not S["done"]:
        here = ee_now()
        dists = [[dist2(t["pos"], here), k] for k, t in enumerate(remaining)]
        dists = sorted(dists)
        remaining = [remaining[d[1]] for d in dists]
        tr = remaining.pop(0)
        okp = pick_container(tr)
        AT_HIGH["v"] = False
        if not okp:
            ok_all = False
        # v02: chain directly to the next container; return to the start pose only after a failed pick
        if not okp and not S["done"]:
            qs, pts = plan_line(ee_now(), P0, QUAT0, QUAT0, cur_q(), 0.05)
            if qs is not None:
                run_joint_path(qs, GRIP_OPEN, False, 0.025, pts)
                AT_HIGH["v"] = True

REPORT["picks"] = picks
REPORT["ok_all"] = ok_all
REPORT["ticks"] = S["tick"]
REPORT["terminal"] = S["term"]
result = REPORT
