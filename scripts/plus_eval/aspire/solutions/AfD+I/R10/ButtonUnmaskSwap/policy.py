# ButtonUnmaskSwap v00 -- press both buttons, track every container in robot_base XY
# (fused front + wrist RGB-D, tick by tick), then lift the containers hiding the goal
# colours in the order the goal names them.
# All positions are derived at runtime from depth/RGB, calibration and robot state.
import math

report = {"version": "v03", "phases": [], "errors": []}
S = {"tick": 0, "ended": False, "conv": {}, "tracks": [], "ztop": None, "table_z": None,
     "static": set(), "frames": 0, "hist": [], "snap": [], "last_pts": {}, "excl": None,
     "frozen": False}

PROCESS_EVERY = 1
FRONT_STRIDE = 2
WRIST_STRIDE = 3
GRID = 0.01
MAX_DQ = 0.05


def note(msg):
    report["phases"].append([S["tick"], msg])


# ---------------------------------------------------------------- geometry helpers
def quat_to_mat(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def cam_params(cam, conv):
    K = cam.intrinsics
    pose = cam.camera_pose
    R = quat_to_mat(pose.quaternion_wxyz)
    t = [float(v) for v in pose.position]
    sy = 1.0
    sz = 1.0
    if conv == "gl":
        sy = -1.0
        sz = -1.0
    return {"fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]), "cy": float(K[1][2]),
            "R": R, "t": t, "sy": sy, "sz": sz}


def pix_to_world(cp, u, v, d):
    xc = (u - cp["cx"]) / cp["fx"] * d
    yc = (v - cp["cy"]) / cp["fy"] * d * cp["sy"]
    zc = d * cp["sz"]
    R = cp["R"]
    t = cp["t"]
    return (R[0][0] * xc + R[0][1] * yc + R[0][2] * zc + t[0],
            R[1][0] * xc + R[1][1] * yc + R[1][2] * zc + t[1],
            R[2][0] * xc + R[2][1] * yc + R[2][2] * zc + t[2])


def row_list(arr, v):
    try:
        return arr[v].tolist()
    except Exception:
        return [value for value in arr[v]]


def scan_points(cam, cp, stride, zlo, zhi, with_rgb=False):
    depth = cam.depth_m
    rgb = cam.rgb if with_rgb else None
    h = len(depth)
    w = len(depth[0])
    out = []
    fx = cp["fx"]
    fy = cp["fy"]
    cx = cp["cx"]
    cy = cp["cy"]
    sy = cp["sy"]
    sz = cp["sz"]
    R = cp["R"]
    t = cp["t"]
    r20 = R[2][0]
    r21 = R[2][1]
    r22 = R[2][2]
    for v in range(0, h, stride):
        drow = row_list(depth, v)
        crow = row_list(rgb, v) if with_rgb else None
        ky = (v - cy) / fy * sy
        for u in range(0, w, stride):
            d = drow[u]
            if d < 0.15 or d > 3.0:
                continue
            xc = (u - cx) / fx * d
            yc = ky * d
            zc = d * sz
            z = r20 * xc + r21 * yc + r22 * zc + t[2]
            if z < zlo or z > zhi:
                continue
            x = R[0][0] * xc + R[0][1] * yc + R[0][2] * zc + t[0]
            y = R[1][0] * xc + R[1][1] * yc + R[1][2] * zc + t[1]
            if with_rgb:
                out.append((x, y, z, crow[u]))
            else:
                out.append((x, y, z))
    return out


def cell(x, y):
    return (int(math.floor(x / GRID)), int(math.floor(y / GRID)))


def components(points, min_n):
    cells = {}
    for p in points:
        key = cell(p[0], p[1])
        if key in cells:
            cells[key].append(p)
        else:
            cells[key] = [p]
    seen = set()
    comps = []
    for key in cells:
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        members = []
        while stack:
            cur = stack.pop()
            members.extend(cells[cur])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cur[0] + dx, cur[1] + dy)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        if len(members) >= min_n:
            sx = sum(p[0] for p in members) / len(members)
            sy2 = sum(p[1] for p in members) / len(members)
            comps.append({"x": sx, "y": sy2, "n": len(members)})
    return comps


def quantile(values, frac):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * frac)]


# ---------------------------------------------------------------- task language
ctx = get_task_context()
goal = ctx.language
report["goal"] = goal
COLORS = ("blue", "green", "red", "yellow", "purple", "orange", "black", "white", "pink", "cyan")
order = []
for seg in goal.lower().split("hiding the ")[1:]:
    word = seg.strip().split(" ")[0]
    if word in COLORS:
        order.append(word)
report["order"] = order

try:
    meta = robomme_plus.get_task_metadata()
    report["meta"] = {k: str(meta[k])[:200] for k in meta}
    MAX_STEPS = int(meta["max_steps"])
except Exception as exc:
    report["errors"].append("meta: " + str(exc)[:200])
    MAX_STEPS = 1300
try:
    report["controller"] = {k: str(v)[:200] for k, v in robomme_plus.get_controller_metadata().items()}
except Exception as exc:
    report["errors"].append("ctrl meta: " + str(exc)[:200])

video = []
try:
    video = list(robomme_plus.get_conditioning_video())
except Exception as exc:
    report["errors"].append("video: " + str(exc)[:200])
report["video_frames"] = len(video)

obs0 = get_observation()
state0 = obs0.robot_state
base = state0.base_frame
embodiment = state0.embodiment
home_joints = [float(v) for v in state0.joint_positions["primary"]]
home_pose = state0.end_effector_poses["primary"]
home_q = tuple(float(v) for v in home_pose.quaternion_wxyz)
report["home_ee"] = [round(float(v), 4) for v in home_pose.position]
report["home_q"] = [round(v, 4) for v in home_q]
report["cameras"] = [name for name in obs0.cameras]

# ---------------------------------------------------------------- projection convention
buttons = []
for cam_name in ("front", "wrist"):
    best_conv = "cv"
    try:
        seg = segment_text(cam_name, "button")
        cands = []
        if seg.ok:
            cands = list(seg.segmentations)[:5]
        errs = {"cv": [], "gl": []}
        for cand in cands:
            pc = mask_to_point_cloud(cand, cam_name, target_frame=base)
            pts = pc.points
            if len(pts) < 5:
                continue
            mx = quantile([float(p[0]) for p in pts], 0.5)
            my = quantile([float(p[1]) for p in pts], 0.5)
            mz = quantile([float(p[2]) for p in pts], 0.5)
            ztop = quantile([float(p[2]) for p in pts], 0.9)
            zbot = quantile([float(p[2]) for p in pts], 0.05)
            b = [float(v) for v in cand.box_xyxy]
            u = int((b[0] + b[2]) / 2)
            v = int((b[1] + b[3]) / 2)
            d = float(obs0.cameras[cam_name].depth_m[v][u])
            for conv in ("cv", "gl"):
                cp = cam_params(obs0.cameras[cam_name], conv)
                wpt = pix_to_world(cp, u, v, d)
                errs[conv].append(math.sqrt((wpt[0] - mx) ** 2 + (wpt[1] - my) ** 2 + (wpt[2] - mz) ** 2))
            if cam_name == "front":
                buttons.append({"x": mx, "y": my, "z": mz, "top": ztop, "bot": zbot,
                                "n": len(pts), "score": float(cand.score)})
        if errs["cv"] and errs["gl"]:
            if sum(errs["gl"]) < sum(errs["cv"]):
                best_conv = "gl"
        report["conv_err_" + cam_name] = {k: [round(e, 4) for e in errs[k]] for k in errs}
    except Exception as exc:
        report["errors"].append("conv " + cam_name + ": " + str(exc)[:200])
    S["conv"][cam_name] = best_conv
report["conv"] = S["conv"]

# keep the two highest-scoring "button" candidates that form a plausible pair
# (v00 took every candidate and pressed a cube): distinct (>5 cm), within 35 cm,
# and with tops at the same height (+-1.5 cm).
sel_buttons = []
ranked = sorted([b for b in buttons if b["top"] is not None], key=lambda b: -b["score"])
if ranked:
    sel_buttons.append(ranked[0])
    for b in ranked[1:]:
        dd = math.sqrt((b["x"] - ranked[0]["x"]) ** 2 + (b["y"] - ranked[0]["y"]) ** 2)
        if dd > 0.05 and dd < 0.35 and abs(b["top"] - ranked[0]["top"]) < 0.015:
            sel_buttons.append(b)
            break
report["buttons_sel"] = [{k: round(v, 4) for k, v in b.items()} for b in sel_buttons]
report["buttons_all"] = [{k: round(v, 4) for k, v in b.items()} for b in buttons]


# ---------------------------------------------------------------- first frame analysis
def analyse_first(obs):
    allz = []
    pts_by_cam = {}
    for cam_name in ("front", "wrist"):
        if cam_name not in obs.cameras:
            continue
        cam = obs.cameras[cam_name]
        cp = cam_params(cam, S["conv"].get(cam_name, "cv"))
        pts = scan_points(cam, cp, 2, -1.0, 1.0, with_rgb=True)
        pts_by_cam[cam_name] = pts
        if cam_name == "front":
            allz.extend(p[2] for p in pts)
    table_z = quantile(allz, 0.5)
    if sel_buttons:
        table_z = min(table_z, min(b["bot"] for b in sel_buttons) + 0.005) if table_z is not None else sel_buttons[0]["bot"]
    S["table_z"] = table_z
    ee = obs.robot_state.end_effector_poses["primary"].position
    exs = float(ee[0])
    eys = float(ee[1])
    static_pts = []
    cube_pts = []
    color_pts = {}
    for cam_name in pts_by_cam:
        for p in pts_by_cam[cam_name]:
            dz = p[2] - table_z
            near_button = False
            for b in sel_buttons:
                if (p[0] - b["x"]) ** 2 + (p[1] - b["y"]) ** 2 < 0.035 ** 2:
                    near_button = True
            if dz > 0.045 and dz < 0.30:
                if (p[0] - exs) ** 2 + (p[1] - eys) ** 2 > 0.06 ** 2:
                    static_pts.append(p)
            elif dz > 0.006 and dz <= 0.045 and not near_button:
                cube_pts.append(p)
                r = float(p[3][0])
                g = float(p[3][1])
                bl = float(p[3][2])
                name = None
                if r > 110 and g < 0.4 * r and bl < 0.4 * r:
                    name = "red"
                elif g > 90 and r < 0.6 * g and bl < 0.6 * g:
                    name = "green"
                elif bl > 90 and r < 0.5 * bl and g < 0.6 * bl:
                    name = "blue"
                if name is not None:
                    if name in color_pts:
                        color_pts[name].append(p)
                    else:
                        color_pts[name] = [p]
    for p in static_pts:
        c = cell(p[0], p[1])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                S["static"].add((c[0] + dx, c[1] + dy))
    # v02: drop blobs that are the buttons themselves (<5 cm from a selected button),
    # the robot base (<25 cm from the base origin) or noise (<10 points).
    seeds = []
    for comp in components(cube_pts, 10):
        if math.sqrt(comp["x"] ** 2 + comp["y"] ** 2) < 0.25:
            continue
        near_b = False
        for b in sel_buttons:
            if (comp["x"] - b["x"]) ** 2 + (comp["y"] - b["y"]) ** 2 < 0.05 ** 2:
                near_b = True
        if not near_b:
            seeds.append(comp)
    colors = {}
    for name in color_pts:
        pts = color_pts[name]
        if len(pts) >= 3:
            colors[name] = {"x": quantile([p[0] for p in pts], 0.5), "y": quantile([p[1] for p in pts], 0.5),
                            "n": len(pts)}
    report["first"] = {"table_z": round(table_z, 4), "n_static_cells": len(S["static"]),
                       "seeds": [[round(s["x"], 3), round(s["y"], 3), s["n"]] for s in seeds],
                       "colors": {k: [round(colors[k]["x"], 3), round(colors[k]["y"], 3), colors[k]["n"]] for k in colors}}
    for s in seeds:
        S["tracks"].append({"x": s["x"], "y": s["y"], "vx": 0.0, "vy": 0.0, "miss": 0, "n": 0, "held": False})
    return colors


# ---------------------------------------------------------------- tracker
def frame_points(obs, zlo, zhi):
    ee = obs.robot_state.end_effector_poses["primary"].position
    ex = float(ee[0])
    ey = float(ee[1])
    ez = float(ee[2])
    # v03: exclude the hand only when its fingertips can reach the container-top band;
    # a wide disc while hovering biased the centroid of a partly excluded target.
    arm_r = 0.0
    if S["ztop"] is not None and ez < S["ztop"] + 0.06:
        arm_r = 0.09
    pts = []
    for cam_name in ("front", "wrist"):
        if cam_name not in obs.cameras:
            continue
        cam = obs.cameras[cam_name]
        cp = cam_params(cam, S["conv"].get(cam_name, "cv"))
        stride = FRONT_STRIDE if cam_name == "front" else WRIST_STRIDE
        for p in scan_points(cam, cp, stride, zlo, zhi):
            if cell(p[0], p[1]) in S["static"]:
                continue
            if arm_r > 0 and (p[0] - ex) ** 2 + (p[1] - ey) ** 2 < arm_r * arm_r:
                continue
            excl = S["excl"]
            if excl is not None and (p[0] - excl[0]) ** 2 + (p[1] - excl[1]) ** 2 < excl[2] ** 2:
                continue
            pts.append(p)
    return pts


def update_tracks(obs):
    if S["frozen"]:
        return
    tz = S["table_z"]
    if S["ztop"] is None:
        pts = frame_points(obs, tz + 0.045, tz + 0.20)
        if len(pts) < 60:
            return
        S["ztop"] = quantile([p[2] for p in pts], 0.85)
        note("containers detected ztop=%.4f n=%d" % (S["ztop"] - tz, len(pts)))
    pts = frame_points(obs, S["ztop"] - 0.02, S["ztop"] + 0.012)
    tracks = S["tracks"]
    if not tracks:
        return
    pred = []
    for tr in tracks:
        pred.append([tr["x"] + tr["vx"], tr["y"] + tr["vy"]])
    gate2 = 0.035 * 0.035
    sums = None
    for it in range(2):
        sums = [[0.0, 0.0, 0, []] for tr in tracks]
        for p in pts:
            best = -1
            bd = gate2
            for i in range(len(pred)):
                if tracks[i]["held"]:
                    continue
                dd = (p[0] - pred[i][0]) ** 2 + (p[1] - pred[i][1]) ** 2
                if dd < bd:
                    bd = dd
                    best = i
            if best >= 0:
                s = sums[best]
                s[0] += p[0]
                s[1] += p[1]
                s[2] += 1
                if it == 1:
                    s[3].append(p)
        for i in range(len(tracks)):
            if sums[i][2] >= 4:
                pred[i] = [sums[i][0] / sums[i][2], sums[i][1] / sums[i][2]]
    for i in range(len(tracks)):
        tr = tracks[i]
        if tr["held"]:
            continue
        n = sums[i][2]
        tr["n"] = n
        if n >= 4:
            nx = pred[i][0]
            ny = pred[i][1]
            tr["vx"] = 0.5 * tr["vx"] + 0.5 * (nx - tr["x"])
            tr["vy"] = 0.5 * tr["vy"] + 0.5 * (ny - tr["y"])
            sp = math.sqrt(tr["vx"] ** 2 + tr["vy"] ** 2)
            if sp > 0.01:
                tr["vx"] *= 0.01 / sp
                tr["vy"] *= 0.01 / sp
            tr["x"] = nx
            tr["y"] = ny
            tr["miss"] = 0
            S["last_pts"][i] = sums[i][3]
        else:
            tr["x"] += 0.5 * tr["vx"]
            tr["y"] += 0.5 * tr["vy"]
            tr["vx"] *= 0.5
            tr["vy"] *= 0.5
            tr["miss"] += 1
    S["hist"].append([(tr["x"], tr["y"]) for tr in tracks])
    if len(S["hist"]) > 40:
        S["hist"] = S["hist"][-40:]


def process(obs):
    S["frames"] += 1
    try:
        update_tracks(obs)
    except Exception as exc:
        if len(report["errors"]) < 20:
            report["errors"].append("track: " + str(exc)[:200])
    if S["tick"] % 10 == 0 and S["tracks"]:
        S["snap"].append([S["tick"]] + [[round(tr["x"], 3), round(tr["y"], 3), tr["n"]] for tr in S["tracks"]])


def moving(window):
    h = S["hist"]
    if len(h) < window:
        return True
    a = h[-window]
    b = h[-1]
    for i in range(len(b)):
        if (a[i][0] - b[i][0]) ** 2 + (a[i][1] - b[i][1]) ** 2 > 0.004 ** 2:
            return True
    return False


# ---------------------------------------------------------------- control primitives
def step_to(joints, grip):
    state = S["state"]
    S["cmd"] = [float(v) for v in joints]
    try:
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=list(joints), gripper_position=grip,
            embodiment=state.embodiment)}))
    except Exception as exc:
        S["ended"] = True
        note("step raised: " + str(exc)[:120])
        return False
    S["tick"] += 1
    term = False
    try:
        term = bool(reply.terminated) or bool(reply.truncated)
    except Exception:
        term = False
    if term:
        S["ended"] = True
        note("episode ended")
        return False
    obs = None
    try:
        obs = reply.observation
    except Exception:
        obs = None
    if obs is None:
        try:
            obs = get_observation()
        except Exception as exc:
            S["ended"] = True
            note("obs raised: " + str(exc)[:120])
            return False
    S["obs"] = obs
    S["state"] = obs.robot_state
    if S["tick"] % PROCESS_EVERY == 0:
        process(obs)
    if S["tick"] >= MAX_STEPS - 1:
        S["ended"] = True
    return True


def servo(target, grip, max_ticks, tol=0.012):
    # commanded set-point ramps from the previous command toward target (MAX_DQ/tick);
    # done when the measured joints are within tol of target.
    target = [float(v) for v in target]
    for k in range(max_ticks):
        if S["ended"]:
            return False
        cur = [float(v) for v in S["state"].joint_positions["primary"]]
        err = max(abs(target[j] - cur[j]) for j in range(7))
        if err < tol:
            return True
        prev = S["cmd"]
        cmd = []
        for j in range(7):
            dq = target[j] - prev[j]
            if dq > MAX_DQ:
                dq = MAX_DQ
            if dq < -MAX_DQ:
                dq = -MAX_DQ
            cmd.append(prev[j] + dq)
        if not step_to(cmd, grip):
            return False
    return False


def hold(ticks, grip):
    joints = list(S["cmd"])
    for k in range(ticks):
        if S["ended"]:
            return False
        if not step_to(joints, grip):
            return False
    return True


def ik(x, y, z, q):
    seed = [float(v) for v in S["state"].joint_positions["primary"]]
    res = solve_ik(Pose((x, y, z), q, base), seed_joints=seed)
    if not res.ok:
        res = solve_ik(Pose((x, y, z), q, base), seed_joints=home_joints)
    if not res.ok:
        return None
    return [float(v) for v in res.joint_positions]


def go_xyz(x, y, z, q, grip, max_ticks, tol=0.012):
    j = ik(x, y, z, q)
    if j is None:
        report["errors"].append("ik fail %.3f %.3f %.3f" % (x, y, z))
        return False
    return servo(j, grip, max_ticks, tol)


def yaw_q(delta):
    qz = (math.cos(delta / 2), 0.0, 0.0, math.sin(delta / 2))
    return quat_mul(qz, home_q)


Rh = quat_to_mat(home_q)
home_close_axis = math.atan2(Rh[1][1], Rh[0][1])


def box_yaw(pts):
    if pts is None or len(pts) < 12:
        return None
    best = None
    for k in range(18):
        th = math.radians(5 * k)
        c = math.cos(th)
        s = math.sin(th)
        a = [p[0] * c + p[1] * s for p in pts]
        b = [-p[0] * s + p[1] * c for p in pts]
        area = (quantile(a, 0.95) - quantile(a, 0.05)) * (quantile(b, 0.95) - quantile(b, 0.05))
        if best is None or area < best[0]:
            best = (area, th)
    return best[1]


# ---------------------------------------------------------------- run
S["state"] = state0
S["obs"] = obs0
S["cmd"] = list(home_joints)
first_obs = obs0
if len(video) > 0:
    first_obs = video[0]
colors = analyse_first(first_obs)
if len(video) > 1:
    for vf in video[1:]:
        process(vf)
    note("video processed")
process(obs0)

targets = []
for name in order:
    if name in colors and S["tracks"]:
        c = colors[name]
        bi = 0
        bd = None
        for i in range(len(S["tracks"])):
            tr = S["tracks"][i]
            dd = (tr["x"] - c["x"]) ** 2 + (tr["y"] - c["y"]) ** 2
            if bd is None or dd < bd:
                bd = dd
                bi = i
        targets.append({"color": name, "track": bi, "seed_dist": round(math.sqrt(bd), 4)})
    else:
        targets.append({"color": name, "track": None})
report["targets"] = targets

# ---- press both buttons (closed gripper), nearest first; descend only when no tracked
# container centre is within CLEAR_START of the button, abort upward below CLEAR_ABORT.
CLEAR_START = 0.075
CLEAR_ABORT = 0.055


def clearance(x, y):
    if not S["tracks"] or S["ztop"] is None:
        return 1.0
    best = 1.0
    for tr in S["tracks"]:
        if tr["held"]:
            continue
        dd = math.sqrt((tr["x"] - x) ** 2 + (tr["y"] - y) ** 2)
        if dd < best:
            best = dd
    return best


def servo_tick(target, grip):
    prev = S["cmd"]
    cmd = []
    for j in range(7):
        dq = target[j] - prev[j]
        if dq > MAX_DQ:
            dq = MAX_DQ
        if dq < -MAX_DQ:
            dq = -MAX_DQ
        cmd.append(prev[j] + dq)
    return step_to(cmd, grip)


def press(b, start_cutoff):
    rec = {"button": [round(b["x"], 3), round(b["y"], 3), round(b["top"], 3)], "aborts": 0, "waits": 0}
    hover = b["top"] + 0.10
    if S["ztop"] is not None:
        hover = max(hover, S["ztop"] + 0.06)
    rec["ok_hover"] = go_xyz(b["x"], b["y"], hover, home_q, 0.0, 60)
    jdown = ik(b["x"], b["y"], b["top"] - 0.03, home_q)
    jup = ik(b["x"], b["y"], hover, home_q)
    if jdown is None or jup is None:
        rec["error"] = "ik"
        return rec
    while not S["ended"]:
        if clearance(b["x"], b["y"]) < CLEAR_START and S["tick"] < start_cutoff:
            rec["waits"] += 1
            hold(1, 0.0)
            continue
        aborted = False
        rec["descent_tick"] = S["tick"]
        rec["clear_at_start"] = round(clearance(b["x"], b["y"]), 4)
        for k in range(30):
            if S["ended"]:
                break
            if clearance(b["x"], b["y"]) < CLEAR_ABORT and S["tick"] < start_cutoff:
                aborted = True
                break
            ez = float(S["state"].end_effector_poses["primary"].position[2])
            cur = [float(v) for v in S["state"].joint_positions["primary"]]
            if ez <= b["top"] - 0.015 or max(abs(jdown[j] - cur[j]) for j in range(7)) < 0.008:
                break
            servo_tick(jdown, 0.0)
        if aborted:
            rec["aborts"] += 1
            servo(jup, 0.0, 20)
            continue
        hold(5, 0.0)
        rec["reached"] = [round(float(v), 4) for v in S["state"].end_effector_poses["primary"].position]
        rec["press_tick"] = S["tick"]
        rec["ok_up"] = servo(jup, 0.0, 30)
        break
    return rec


press_log = []
if sel_buttons:
    hx = report["home_ee"][0]
    hy = report["home_ee"][1]
    sel_buttons.sort(key=lambda b: (b["x"] - hx) ** 2 + (b["y"] - hy) ** 2)
    cutoffs = (90, 150)
    for bi in range(min(2, len(sel_buttons))):
        if S["ended"]:
            break
        press_log.append(press(sel_buttons[bi], cutoffs[bi]))
        note("pressed button")
report["presses"] = press_log
if not S["ended"]:
    servo(home_joints, 0.0, 60)
note("home after presses")

# ---- wait until containers stop moving
wait_limit = min(MAX_STEPS - 350, 520)
while not S["ended"] and S["tick"] < wait_limit:
    hold(1, 1.0)
    if S["frames"] > 30 and not moving(25):
        break
settled = not moving(25)
note("wait done moving=%s" % str(not settled))
# v03: once the swap has settled the containers are static; freeze the tracks so arm
# occlusion cannot drag them, and correct each target locally from above before grasping.
if settled:
    S["frozen"] = True


def local_points(x, y, radius):
    obs = S["obs"]
    if obs is None or S["ztop"] is None:
        return []
    # top surface only (side faces seen by the front camera would bias the centroid)
    pts = frame_points(obs, S["ztop"] - 0.01, S["ztop"] + 0.012)
    return [p for p in pts if (p[0] - x) ** 2 + (p[1] - y) ** 2 < radius * radius]


def refine(x, y, frames):
    # centroid of the container-top band around (x, y), shrinking radius, over a few frames
    xs = []
    ys = []
    last = []
    for f in range(frames):
        if S["ended"]:
            break
        hold(1, 1.0)
        cx = x
        cy = y
        sel = []
        for radius in (0.05, 0.038, 0.032):
            sel = local_points(cx, cy, radius)
            if len(sel) < 8:
                break
            cx = sum(p[0] for p in sel) / len(sel)
            cy = sum(p[1] for p in sel) / len(sel)
        if len(sel) >= 8:
            xs.append(cx)
            ys.append(cy)
            last = sel
    if not xs:
        return x, y, [], 0
    xs.sort()
    ys.sort()
    return xs[len(xs) // 2], ys[len(ys) // 2], last, len(xs)


def grasp_q(pts, fallback_pts):
    yaw = box_yaw(pts) if len(pts) >= 12 else box_yaw(fallback_pts)
    delta = 0.0
    if yaw is not None:
        delta = yaw - home_close_axis
        while delta > math.pi / 4:
            delta -= math.pi / 2
        while delta <= -math.pi / 4:
            delta += math.pi / 2
    return yaw, delta


def grip_now():
    return float(S["state"].gripper_positions["primary"])


def attempt(tr, rec, ztop):
    # hover, refine, vertical descent, close, vertical lift and hold. Returns grip readings.
    x = tr["x"]
    y = tr["y"]
    yaw, delta = grasp_q([], S["last_pts"].get(rec["track"]))
    q = yaw_q(delta)
    a1 = go_xyz(x, y, ztop + 0.10, q, 1.0, 80)
    rx, ry, pts, nf = refine(x, y, 3)
    shift = math.sqrt((rx - x) ** 2 + (ry - y) ** 2)
    if shift > 0.008 and not S["ended"]:
        go_xyz(rx, ry, ztop + 0.10, q, 1.0, 40)
        rx2, ry2, pts2, nf2 = refine(rx, ry, 3)
        if nf2 > 0:
            rx = rx2
            ry = ry2
            pts = pts2
    yaw2, delta2 = grasp_q(pts, S["last_pts"].get(rec["track"]))
    if yaw2 is not None:
        yaw = yaw2
        delta = delta2
    q = yaw_q(delta)
    att = {"xy0": [round(x, 4), round(y, 4)], "xy": [round(rx, 4), round(ry, 4)], "shift": round(shift, 4),
           "n_pts": len(pts), "yaw_deg": None if yaw is None else round(math.degrees(yaw), 1),
           "delta_deg": round(math.degrees(delta), 1), "a1": a1}
    a2 = go_xyz(rx, ry, ztop + 0.10, q, 1.0, 40)
    a3 = go_xyz(rx, ry, ztop + 0.03, q, 1.0, 40, tol=0.01)
    a4 = go_xyz(rx, ry, ztop - 0.03, q, 1.0, 50, tol=0.008)
    att["ee_low"] = [round(float(v), 4) for v in S["state"].end_effector_poses["primary"].position] if not S["ended"] else None
    att["ok"] = [a2, a3, a4]
    tr["held"] = True
    hold(14, 0.0)
    att["grip"] = round(grip_now(), 3) if not S["ended"] else None
    go_xyz(rx, ry, ztop + 0.03, q, 0.0, 40, tol=0.012)
    go_xyz(rx, ry, ztop + 0.12, q, 0.0, 50)
    hold(8, 0.0)
    att["grip_lift"] = round(grip_now(), 3) if not S["ended"] else None
    att["tick_lift"] = S["tick"]
    att["ended"] = S["ended"]
    tr["x"] = rx
    tr["y"] = ry
    return att, q


def put_back(tr, q, ztop):
    go_xyz(tr["x"], tr["y"], ztop + 0.03, q, 0.0, 50, tol=0.01)
    go_xyz(tr["x"], tr["y"], ztop - 0.02, q, 0.0, 40, tol=0.01)
    hold(10, 1.0)
    go_xyz(tr["x"], tr["y"], ztop + 0.10, q, 1.0, 40)
    tr["held"] = False
    tr["vx"] = 0.0
    tr["vy"] = 0.0


# ---- pick in order
picks = []
for ti in range(len(targets)):
    tg = targets[ti]
    if S["ended"]:
        break
    if tg["track"] is None:
        picks.append({"color": tg["color"], "skipped": "no track"})
        continue
    tr = S["tracks"][tg["track"]]
    ztop = S["ztop"] if S["ztop"] is not None else S["table_z"] + 0.08
    rec = {"color": tg["color"], "track": tg["track"], "tick0": S["tick"], "attempts": []}
    for k in range(2):
        if S["ended"]:
            break
        att, q = attempt(tr, rec, ztop)
        rec["attempts"].append(att)
        if S["ended"]:
            break
        g0 = att["grip"]
        g1 = att["grip_lift"]
        # a slipping grasp (fingers closing further while lifted) is treated as not picked
        secure = g0 is not None and g1 is not None and g1 > 0.05 and g0 - g1 < 0.05
        att["secure"] = secure
        put_back(tr, q, ztop)
        if secure:
            break
    rec["ended_after"] = S["ended"]
    picks.append(rec)
report["picks"] = picks
report["ticks"] = S["tick"]
report["frames_processed"] = S["frames"]
report["ztop_rel"] = None if S["ztop"] is None else round(S["ztop"] - S["table_z"], 4)
report["final_tracks"] = [[round(tr["x"], 3), round(tr["y"], 3), tr["n"], tr["miss"]] for tr in S["tracks"]]
report["snap"] = S["snap"]
report["ended"] = S["ended"]
result = report
