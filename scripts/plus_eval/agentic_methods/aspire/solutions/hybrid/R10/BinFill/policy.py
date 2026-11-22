# BinFill v04 -- (v03 + belt-exit edge from the raised conveyor surface (fixes early "late" starvation), move-duration model incl. wrist yaw, yaw chosen near current yaw) (v02 + faster moving-belt cycle: no home/observe hold between cubes, release from transit height, shorter hover margin and closing, relaxed exit margin) (v01 + spawn-zone clearance for the hand on moving belts; across-belt closing preferred) (v00 + tracking descent, finger-sweep clearance filter) initial program from offline demo analysis (task_analysis.md).
# Strategy: parse goal counts per colour; repeatedly detect cubes of a needed colour in the
# front camera by runtime hue classification (goal colour words -> generic hue classes),
# project to 3D, estimate belt velocity from two observations, intercept/grasp, drop into
# the bin (located at runtime: dark interior + rim), then press the button.
# Only public APIs; no recorded coordinates; every geometric quantity is derived at runtime.
import math

report = {"version": "v04", "log": [], "errors": [], "placed": {}, "attempts": []}
ENDED = {"flag": False, "why": ""}


def log(msg):
    if len(report["log"]) < 400:
        report["log"].append(msg)


def check_end(res, tag):
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
        ENDED["flag"] = True
        ENDED["why"] = tag + (" terminated" if term else " truncated")
        raise ValueError("EPISODE_ENDED " + ENDED["why"])


ctx = get_task_context()
goal = str(ctx.language).lower()
report["goal"] = goal
state0 = get_robot_state()
BASE = state0.base_frame
EMB = state0.embodiment
try:
    META = dict(robomme_plus.get_task_metadata())
except Exception as exc:
    META = {}
    report["errors"].append("meta: " + str(exc))
report["meta"] = {k: str(META[k])[:200] for k in META}
try:
    CMETA = dict(robomme_plus.get_controller_metadata())
    report["controller_meta"] = {k: str(CMETA[k])[:300] for k in CMETA}
except Exception as exc:
    report["errors"].append("cmeta: " + str(exc))
try:
    video = robomme_plus.get_conditioning_video()
    report["video_frames"] = len(video)
except Exception as exc:
    report["errors"].append("video: " + str(exc))

HZ = 20.0
try:
    HZ = float(META.get("control_hz", META.get("control_frequency", 20)))
except Exception:
    HZ = 20.0
MAX_STEPS = 1000
try:
    MAX_STEPS = int(META.get("max_steps", 1000))
except Exception:
    MAX_STEPS = 1000
T0 = float(state0.timestamp_s)


def now_tick():
    return int(round((float(get_robot_state().timestamp_s) - T0) * HZ))


# ---------------- goal parsing ----------------
NUMS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10}
HUES = {"red": [(345.0, 360.1), (0.0, 15.0)], "orange": [(25.0, 50.0)], "yellow": [(50.0, 72.0)],
        "green": [(90.0, 150.0)], "cyan": [(165.0, 200.0)], "blue": [(210.0, 265.0)],
        "purple": [(265.0, 335.0)], "magenta": [(280.0, 335.0)], "pink": [(280.0, 345.0)]}
words = goal.replace(",", " ").replace(".", " ").split()
need = {}
for wi in range(len(words) - 1):
    word = words[wi]
    count = None
    if word in NUMS:
        count = NUMS[word]
    elif word.isdigit():
        count = int(word)
    if count is None:
        continue
    colour = words[wi + 1]
    if colour in HUES:
        need[colour] = need.get(colour, 0) + count
report["need"] = dict(need)
placed = {}
for colour in need:
    placed[colour] = 0


def hue_class(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 30:
        return None
    d = mx - mn
    if d < 0.7 * mx:
        return None
    if mx == r:
        h = ((g - b) / d) % 6.0
    elif mx == g:
        h = (b - r) / d + 2.0
    else:
        h = (r - g) / d + 4.0
    h = h * 60.0
    for name in ("red", "orange", "yellow", "green", "cyan", "blue", "magenta"):
        for lo, hi in HUES[name]:
            if lo <= h < hi:
                return name
    return None


def colour_matches(cls, wanted):
    if cls is None:
        return False
    if wanted in ("purple", "magenta", "pink"):
        return cls == "magenta"
    return cls == wanted


# ---------------- geometry helpers ----------------
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def qrot(q, v):
    w, x, y, z = q
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    return (r00 * v[0] + r01 * v[1] + r02 * v[2],
            r10 * v[0] + r11 * v[1] + r12 * v[2],
            r20 * v[0] + r21 * v[1] + r22 * v[2])


def q_down(theta):
    # top-down hand (z axis to -Z) with hand x axis at yaw theta; closing axis (hand y) at theta-90deg
    return qmul((math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)), (0.0, 1.0, 0.0, 0.0))


HOME_Q = tuple(float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz)
HOME_P = tuple(float(v) for v in state0.end_effector_poses["primary"].position)
hx = qrot(HOME_Q, (1.0, 0.0, 0.0))
hz = qrot(HOME_Q, (0.0, 0.0, 1.0))
THETA_HOME = math.atan2(hx[1], hx[0])
report["home"] = {"p": [round(v, 4) for v in HOME_P], "q": [round(v, 4) for v in HOME_Q],
                  "z_axis": [round(v, 3) for v in hz], "theta_deg": round(math.degrees(THETA_HOME), 1)}


def wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def theta_for_closing_axis(phi, period):
    # closing axis angle phi (rad); symmetric under rotation by `period`; choose nearest home yaw
    base_theta = phi + math.pi / 2.0
    best = None
    k = -8
    while k <= 8:
        cand = base_theta + k * period
        dist = abs(wrap_pi(cand - THETA_HOME))
        if best is None or dist < best[0]:
            best = (dist, cand)
        k += 1
    return THETA_HOME + wrap_pi(best[1] - THETA_HOME)


def ee_yaw():
    q = tuple(float(v) for v in get_robot_state().end_effector_poses["primary"].quaternion_wxyz)
    hxv = qrot(q, (1.0, 0.0, 0.0))
    return math.atan2(hxv[1], hxv[0])


def theta_near(phi, ref):
    # closing axis phi (period pi); hand yaw nearest `ref`, kept within ~100 deg of the home yaw
    best = None
    k = -8
    while k <= 8:
        cand = phi + math.pi / 2.0 + k * math.pi
        cand = THETA_HOME + wrap_pi(cand - THETA_HOME)
        if abs(cand - THETA_HOME) <= 1.75:
            dist = abs(wrap_pi(cand - ref))
            if best is None or dist < best[0]:
                best = (dist, cand)
        k += 1
    if best is None:
        return theta_for_closing_axis(phi, math.pi)
    return best[1]


# move_to_pose duration model fitted on this task's own runs (controller dynamics, not scene data):
# ticks ~= k * (8 + max(38 * metres, 0.38 * yaw_degrees)); k adapted online from measured hover moves
MOVE = {"k": 1.0}


def move_ticks_base(p_from, yaw_from, p_to, yaw_to):
    d = math.sqrt(sum((p_to[i] - p_from[i]) ** 2 for i in range(3)))
    dy = abs(math.degrees(wrap_pi(yaw_to - yaw_from)))
    return 8.0 + max(38.0 * d, 0.38 * dy)


def quantile(vals, frac):
    s = sorted(vals)
    if not s:
        return None
    return s[int((len(s) - 1) * frac)]


# ---------------- camera projection ----------------
CONV = {"mode": None}


def cam_params(cam):
    K = cam.intrinsics.tolist()
    pose = cam.camera_pose
    return (float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2]),
            tuple(float(v) for v in pose.position), tuple(float(v) for v in pose.quaternion_wxyz))


def project(params, u, v, d, mode):
    fx, fy, cx, cy, pos, q = params
    xn = (u - cx) / fx
    yn = (v - cy) / fy
    if mode == "cv":
        pc = (xn * d, yn * d, d)
    else:
        pc = (xn * d, -yn * d, -d)
    w = qrot(q, pc)
    return (pos[0] + w[0], pos[1] + w[1], pos[2] + w[2])


SUB = 2


def snapshot():
    obs = get_observation()
    cam = obs.cameras["front"]
    rgb = cam.rgb.tolist()
    dep = cam.depth_m.tolist()
    params = cam_params(cam)
    tick = int(round((float(obs.robot_state.timestamp_s) - T0) * HZ))
    return {"rgb": rgb, "dep": dep, "params": params, "tick": tick, "H": len(rgb), "W": len(rgb[0])}


def find_blobs(snap):
    rgb = snap["rgb"]
    H = snap["H"]
    W = snap["W"]
    labels = {}
    dark = {}
    for y in range(0, H, SUB):
        row = rgb[y]
        for x in range(0, W, SUB):
            px = row[x]
            r = px[0]
            g = px[1]
            b = px[2]
            c = hue_class(r, g, b)
            if c is not None:
                labels[(x, y)] = c
            elif max(r, g, b) < 22:
                dark[(x, y)] = True
    blobs = []
    seen = {}
    for key in labels:
        if key in seen:
            continue
        c = labels[key]
        stack = [key]
        seen[key] = True
        pix = []
        while stack:
            p = stack.pop()
            pix.append(p)
            for dx, dy in ((SUB, 0), (-SUB, 0), (0, SUB), (0, -SUB)):
                nb = (p[0] + dx, p[1] + dy)
                if nb in labels and nb not in seen and labels[nb] == c:
                    seen[nb] = True
                    stack.append(nb)
        if len(pix) >= 5:
            blobs.append({"colour": c, "pix": pix})
    darkblobs = []
    seen = {}
    for key in dark:
        if key in seen:
            continue
        stack = [key]
        seen[key] = True
        pix = []
        while stack:
            p = stack.pop()
            pix.append(p)
            for dx, dy in ((SUB, 0), (-SUB, 0), (0, SUB), (0, -SUB)):
                nb = (p[0] + dx, p[1] + dy)
                if nb in dark and nb not in seen:
                    seen[nb] = True
                    stack.append(nb)
        if len(pix) >= 6:
            darkblobs.append(pix)
    return blobs, darkblobs, labels


def blob_points(snap, pix, mode):
    out = []
    dep = snap["dep"]
    for (x, y) in pix:
        d = float(dep[y][x])
        if d > 0.05 and d < 5.0:
            out.append(project(snap["params"], x + 0.5, y + 0.5, d, mode))
    return out


def ring_pixels(pix, labels, pad, H, W):
    xs = [p[0] for p in pix]
    ys = [p[1] for p in pix]
    x0 = max(0, min(xs) - pad)
    x1 = min(W - 1, max(xs) + pad)
    y0 = max(0, min(ys) - pad)
    y1 = min(H - 1, max(ys) + pad)
    own = {}
    for p in pix:
        own[p] = True
    ring = []
    y = y0 - (y0 % SUB)
    while y <= y1:
        x = x0 - (x0 % SUB)
        while x <= x1:
            if (x, y) not in own and (x, y) not in labels:
                ring.append((x, y))
            x += SUB
        y += SUB
    return ring


def calibrate(snap, blobs):
    # decide pixel->3D convention by comparing with the public mask_to_point_cloud
    if CONV["mode"] is not None:
        return
    info = {}
    target = None
    for b in blobs:
        if target is None or len(b["pix"]) > len(target["pix"]):
            target = b
    if target is not None:
        try:
            H = snap["H"]
            W = snap["W"]
            mask = [[False] * W for yy in range(H)]
            for (x, y) in target["pix"]:
                for dy in range(SUB):
                    for dx in range(SUB):
                        if y + dy < H and x + dx < W:
                            mask[y + dy][x + dx] = True
            pc = mask_to_point_cloud(mask, "front", target_frame=BASE)
            pts = [tuple(float(c) for c in p) for p in pc.points]
            info["mask_cloud_n"] = len(pts)
            if len(pts) > 3:
                ref = (quantile([p[0] for p in pts], 0.5), quantile([p[1] for p in pts], 0.5),
                       quantile([p[2] for p in pts], 0.5))
                info["ref"] = [round(v, 4) for v in ref]
                best = None
                for mode in ("cv", "gl"):
                    mp = blob_points(snap, target["pix"], mode)
                    if not mp:
                        continue
                    med = (quantile([p[0] for p in mp], 0.5), quantile([p[1] for p in mp], 0.5),
                           quantile([p[2] for p in mp], 0.5))
                    dist = math.sqrt(sum((med[i] - ref[i]) ** 2 for i in range(3)))
                    info[mode] = [round(v, 4) for v in med] + [round(dist, 4)]
                    if best is None or dist < best[0]:
                        best = (dist, mode)
                if best is not None and best[0] < 0.03:
                    CONV["mode"] = best[1]
        except Exception as exc:
            info["mask_error"] = str(exc)[:300]
    if CONV["mode"] is None:
        # fallback heuristic: the correct convention puts table points below the camera
        params = snap["params"]
        cz = params[4][2]
        votes = {"cv": 0, "gl": 0}
        H = snap["H"]
        W = snap["W"]
        for y in range(H // 4, H, H // 8):
            for x in range(W // 8, W, W // 8):
                d = float(snap["dep"][y][x])
                if d <= 0.05 or d > 5.0:
                    continue
                for mode in ("cv", "gl"):
                    if project(params, x, y, d, mode)[2] < cz - 0.05:
                        votes[mode] += 1
        info["votes"] = votes
        CONV["mode"] = "cv" if votes["cv"] >= votes["gl"] else "gl"
    info["mode"] = CONV["mode"]
    report["calibration"] = info


def describe_cubes(snap, blobs, labels):
    cubes = []
    for b in blobs:
        pts = blob_points(snap, b["pix"], CONV["mode"])
        if len(pts) < 4:
            continue
        zs = [p[2] for p in pts]
        top = quantile(zs, 0.9)
        toppts = [p for p in pts if p[2] >= top - 0.008]
        cx = sum(p[0] for p in toppts) / len(toppts)
        cy = sum(p[1] for p in toppts) / len(toppts)
        ring = ring_pixels(b["pix"], labels, 6, snap["H"], snap["W"])
        rpts = blob_points(snap, ring, CONV["mode"])
        support = quantile([p[2] for p in rpts], 0.3) if rpts else min(zs)
        h = top - support
        cubes.append({"colour": b["colour"], "c": (cx, cy, top), "n": len(b["pix"]),
                      "support": support, "h": h, "toppts": toppts,
                      "uv": (sum(p[0] for p in b["pix"]) / len(b["pix"]),
                             sum(p[1] for p in b["pix"]) / len(b["pix"]))})
    return cubes


def find_bin(snap, darkblobs, labels):
    best = None
    for pix in darkblobs:
        pts = blob_points(snap, pix, CONV["mode"])
        if len(pts) < 5:
            continue
        ring = ring_pixels(pix, labels, 8, snap["H"], snap["W"])
        rpts = blob_points(snap, ring, CONV["mode"])
        if len(rpts) < 5:
            continue
        rim = quantile([p[2] for p in rpts], 0.95)
        floor = quantile([p[2] for p in pts], 0.5)
        depth = rim - floor
        score = len(pix) * (1.0 if depth > 0.01 else 0.2)
        if best is None or score > best[0]:
            rimpts = [p for p in rpts if p[2] >= rim - 0.015]
            xs = [p[0] for p in rimpts]
            ys = [p[1] for p in rimpts]
            dxs = [p[0] for p in pts]
            dys = [p[1] for p in pts]
            best = (score, {"rim_z": rim, "floor_z": floor, "n": len(pix),
                            "rim_center": ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0),
                            "dark_center": (sum(dxs) / len(dxs), sum(dys) / len(dys)),
                            "rim_span": (max(xs) - min(xs), max(ys) - min(ys))})
    return None if best is None else best[1]


def sam_locate(prompts):
    for prompt in prompts:
        try:
            found = localize_object(prompt, camera_name="front", target_frame=BASE)
        except Exception as exc:
            report["errors"].append("localize " + prompt + ": " + str(exc)[:200])
            continue
        if found.ok and len(found.point_cloud.points) >= 15:
            pts = [tuple(float(c) for c in p) for p in found.point_cloud.points]
            return prompt, pts
    return None, None


# ---------------- motion helpers ----------------
MOVES = []


def goto(pos, theta, tol=0.01, ms=300, tag=""):
    t_before = now_tick()
    pose = Pose((float(pos[0]), float(pos[1]), float(pos[2])), q_down(theta), BASE)
    res = move_to_pose(pose, tolerance=tol, max_steps=ms)
    steps = 0
    try:
        steps = int(res.steps_executed)
    except Exception:
        steps = -1
    status = ""
    try:
        status = str(res.status)
    except Exception:
        status = "?"
    MOVES.append([tag, bool(res.ok), status, steps, t_before])
    check_end(res, "move " + tag)
    return res


def hold(n, grip):
    st = get_robot_state()
    q = [float(v) for v in st.joint_positions["primary"]]
    for i in range(n):
        sr = step(RobotAction(arms={"primary": ArmCommand("joint_position", q, gripper_position=grip, embodiment=EMB)}))
        check_end(sr, "hold")


def grip_reading():
    return float(get_robot_state().gripper_positions["primary"])


def ik_joints(res):
    try:
        return [float(v) for v in res.joint_positions]
    except Exception:
        pass
    try:
        return [float(v) for v in res.solution]
    except Exception:
        return None


def track_close(vel, ticks):
    # close the gripper while translating with the belt so the relative motion is ~0
    st = get_robot_state()
    q0 = [float(v) for v in st.joint_positions["primary"]]
    ee = st.end_effector_poses["primary"]
    p = [float(v) for v in ee.position]
    q1 = None
    if abs(vel[0]) + abs(vel[1]) > 1e-6:
        tgt = Pose((p[0] + vel[0] * ticks, p[1] + vel[1] * ticks, p[2]),
                   tuple(float(v) for v in ee.quaternion_wxyz), BASE)
        try:
            ik = solve_ik(tgt, seed_joints=q0)
            if ik.ok:
                q1 = ik_joints(ik)
        except Exception as exc:
            report["errors"].append("track ik: " + str(exc)[:200])
    if q1 is None:
        q1 = q0
    for k in range(1, ticks + 1):
        f = float(k) / ticks
        qk = [q0[i] + (q1[i] - q0[i]) * f for i in range(len(q0))]
        sr = step(RobotAction(arms={"primary": ArmCommand("joint_position", qk, gripper_position=0.0, embodiment=EMB)}))
        check_end(sr, "track_close")


# ---------------- scene setup ----------------
VEL = {"v": (0.0, 0.0, 0.0), "known": False, "smax": None, "smin": None}


def match_velocity(c_a, c_b, dt):
    if dt <= 0:
        return None
    disps = []
    for a in c_a:
        best = None
        for b in c_b:
            if b["colour"] != a["colour"]:
                continue
            d = math.sqrt((a["c"][0] - b["c"][0]) ** 2 + (a["c"][1] - b["c"][1]) ** 2)
            if best is None or d < best[0]:
                best = (d, b)
        if best is not None and best[0] < 0.006 * dt + 0.01:
            disps.append(((best[1]["c"][0] - a["c"][0]) / dt, (best[1]["c"][1] - a["c"][1]) / dt))
    if not disps:
        return None
    vx = quantile([d[0] for d in disps], 0.5)
    vy = quantile([d[1] for d in disps], 0.5)
    return (vx, vy, len(disps))


SCENE = {"bin": None, "button": None}


def setup_scene():
    snap = snapshot()
    blobs, darkblobs, labels = find_blobs(snap)
    calibrate(snap, blobs)
    b = find_bin(snap, darkblobs, labels)
    prompt, pts = sam_locate(("bin", "box", "square box", "container"))
    sam_bin = None
    if pts:
        zs = [p[2] for p in pts]
        top = quantile(zs, 0.95)
        sam_bin = {"prompt": prompt, "center": (quantile([p[0] for p in pts], 0.5), quantile([p[1] for p in pts], 0.5)),
                   "top": top, "n": len(pts),
                   "span": (max(p[0] for p in pts) - min(p[0] for p in pts), max(p[1] for p in pts) - min(p[1] for p in pts))}
    report["bin_pixel"] = None if b is None else {k: (str(b[k]) if k != "rim_z" else round(b[k], 4)) for k in b}
    report["bin_sam"] = None if sam_bin is None else {k: str(sam_bin[k]) for k in sam_bin}
    if b is not None:
        SCENE["bin"] = {"xy": b["rim_center"], "rim": b["rim_z"]}
        if sam_bin is not None:
            dd = math.sqrt((b["rim_center"][0] - sam_bin["center"][0]) ** 2 + (b["rim_center"][1] - sam_bin["center"][1]) ** 2)
            report["bin_pixel_vs_sam"] = round(dd, 4)
    elif sam_bin is not None:
        SCENE["bin"] = {"xy": sam_bin["center"], "rim": sam_bin["top"]}
    prompt, pts = sam_locate(("button", "white button", "push button", "round button"))
    if pts:
        zs = [p[2] for p in pts]
        top = quantile(zs, 0.95)
        toppts = [p for p in pts if p[2] >= top - 0.012]
        SCENE["button"] = {"xy": (sum(p[0] for p in toppts) / len(toppts), sum(p[1] for p in toppts) / len(toppts)),
                  "top": top, "prompt": prompt, "n": len(pts),
                  "span": (max(p[0] for p in pts) - min(p[0] for p in pts), max(p[1] for p in pts) - min(p[1] for p in pts))}
    BIN = SCENE["bin"]
    BUTTON = SCENE["button"]
    report["bin"] = None if BIN is None else {"xy": [round(v, 4) for v in BIN["xy"]], "rim": round(BIN["rim"], 4)}
    report["button"] = None if BUTTON is None else {"xy": [round(v, 4) for v in BUTTON["xy"]], "top": round(BUTTON["top"], 4),
                                                    "prompt": BUTTON["prompt"], "n": BUTTON["n"],
                                                    "span": [round(v, 4) for v in BUTTON["span"]]}
    return snap, blobs, labels


def observe_pair(gap):
    s1 = snapshot()
    b1, d1, l1 = find_blobs(s1)
    calibrate(s1, b1)
    c1 = describe_cubes(s1, b1, l1)
    hold(gap, 1.0)
    s2 = snapshot()
    b2, d2, l2 = find_blobs(s2)
    c2 = describe_cubes(s2, b2, l2)
    dt = s2["tick"] - s1["tick"]
    mv = match_velocity(c1, c2, dt)
    if mv is not None:
        if VEL["known"]:
            VEL["v"] = (0.5 * VEL["v"][0] + 0.5 * mv[0], 0.5 * VEL["v"][1] + 0.5 * mv[1], 0.0)
        else:
            VEL["v"] = (mv[0], mv[1], 0.0)
            VEL["known"] = True
    return s2, c2


def speed():
    return math.sqrt(VEL["v"][0] ** 2 + VEL["v"][1] ** 2)


EST = {"pre": 42}
# Generic gripper-sweep tolerances (not scene-specific): half-extent of the open-finger sweep along
# the closing axis and across it, measured from the target centre, including a neighbour cube's
# half-diagonal. A neighbour inside this box would be touched by a finger during the descent.
SWEEP = {"along_close": 0.085, "across": 0.045}
ND = 10      # descent ticks (tracking)
# Generic hand-size tolerance: on a moving belt new cubes appear at the upstream edge (running min of
# observed along-belt positions); keep the hovering hand at least this far downstream of it.
SPAWN_CLEAR = 0.16
HOVER_MARGIN = 3   # ticks the cube should still travel when the hand reaches the hover point
NC = 7       # closing ticks (tracking, gripper commanded closed)
LAG = 2      # joint-position tracking lag compensation (ticks)


def clearance(c, cubes, axis):
    u0 = math.cos(axis)
    u1 = math.sin(axis)
    worst = 9.0
    for o in cubes:
        if o is c:
            continue
        dx = o["c"][0] - c["c"][0]
        dy = o["c"][1] - c["c"][1]
        a = abs(dx * u0 + dy * u1)
        b = abs(-dx * u1 + dy * u0)
        m = max(a / SWEEP["along_close"], b / SWEEP["across"])
        if m < worst:
            worst = m
    return worst


def grasp_axes(c):
    fa = face_angle(c["toppts"]) if len(c["toppts"]) >= 4 else 0.0
    # face-aligned closing axes first, diagonal (still graspable: cube diagonal < opening) last
    return [(fa, 0), (fa + math.pi / 2.0, 0), (fa + math.pi / 4.0, 1), (fa + 3.0 * math.pi / 4.0, 1)]


def best_axis(c, cubes):
    best = None
    for ax, diag in grasp_axes(c):
        clr = clearance(c, cubes, ax)
        th = theta_near(ax, YAW_NOW["v"])
        rot = abs(wrap_pi(th - YAW_NOW["v"]))
        key = (0 if clr > 1.0 else 1, diag, rot, -clr)
        if best is None or key < best[0]:
            best = (key, ax, clr, th)
    c["theta"] = best[3]
    return best[1], best[2]


YAW_NOW = {"v": 0.0}
BELT = {"lo": None, "hi": None, "n": 0, "done": False}


def belt_extent(snap, cubes):
    # Conveyor exit edge: low-saturation, non-dark pixels lying on the dominant raised surface
    # (z histogram peak of such pixels near table height), projected on the measured belt direction.
    sp = speed()
    if sp <= 0.0003 or len(cubes) < 1:
        return
    vhat = (VEL["v"][0] / sp, VEL["v"][1] / sp)
    rgb = snap["rgb"]
    pix = []
    for y in range(0, snap["H"], SUB):
        row = rgb[y]
        for x in range(0, snap["W"], SUB):
            px = row[x]
            mx = max(px[0], px[1], px[2])
            mn = min(px[0], px[1], px[2])
            if mx >= 22 and (mx - mn) < 0.3 * mx:
                pix.append((x, y))
    pts = blob_points(snap, pix, CONV["mode"])
    zmax = quantile([c["c"][2] for c in cubes], 0.5)
    hist = {}
    for p in pts:
        if -0.01 < p[2] < zmax - 0.01:
            k = int(round(p[2] / 0.002))
            hist[k] = hist.get(k, 0) + 1
    if not hist:
        BELT["done"] = True
        return
    kbest = sorted([(-hist[k], k) for k in hist])[0][1]
    zb = kbest * 0.002
    ss = [p[0] * vhat[0] + p[1] * vhat[1] for p in pts if abs(p[2] - zb) < 0.004]
    BELT["n"] = len(ss)
    BELT["done"] = True
    if len(ss) >= 50:
        BELT["lo"] = quantile(ss, 0.01)
        BELT["hi"] = quantile(ss, 0.99)
    report["belt"] = {"n": len(ss), "zb": round(zb, 4), "lo": None if BELT["lo"] is None else round(BELT["lo"], 4),
                      "hi": None if BELT["hi"] is None else round(BELT["hi"], 4)}


def exit_s():
    # cube centres must be grasped before this along-belt coordinate
    smax = VEL["smax"]
    if BELT["hi"] is not None:
        hi = BELT["hi"] - 0.02
        if smax is not None and smax > hi:
            hi = smax
        return hi
    if smax is not None and VEL["smin"] is not None and smax - VEL["smin"] > 0.3:
        return smax
    return None


def hover_lead(c, theta):
    # predicted ticks of the hover move from the current pose (two fixed-point iterations)
    st = get_robot_state()
    ee = [float(v) for v in st.end_effector_poses["primary"].position]
    dt0 = now_tick() - c["tick"]
    z_h = c["c"][2] + 0.06
    T = 25.0
    for it in range(2):
        hx_ = c["c"][0] + VEL["v"][0] * (dt0 + T + HOVER_MARGIN)
        hy_ = c["c"][1] + VEL["v"][1] * (dt0 + T + HOVER_MARGIN)
        T = MOVE["k"] * move_ticks_base(ee, YAW_NOW["v"], (hx_, hy_, z_h), theta)
    return int(round(T))


FAILED = []
AT_HOME = {"v": True}
SNAPREF = {"s": None}


def failed_here(c):
    # static scene: skip a cube location that already produced two failed grasps
    n = 0
    for f in FAILED:
        if math.sqrt((f[0] - c["c"][0]) ** 2 + (f[1] - c["c"][1]) ** 2) < 0.02:
            n += 1
    return n >= 2


def extra_lead(c):
    # extra ticks so that the hover/intercept point is SPAWN_CLEAR downstream of the spawn edge
    sp = speed()
    if sp <= 0.0003 or VEL["smin"] is None:
        return 0
    vhat = (VEL["v"][0] / sp, VEL["v"][1] / sp)
    s_h = c["c"][0] * vhat[0] + c["c"][1] * vhat[1] + sp * (c.get("T", EST["pre"]) + HOVER_MARGIN)
    need_s = VEL["smin"] + SPAWN_CLEAR
    if s_h >= need_s:
        return 0
    return int((need_s - s_h) / sp) + 1


def pick_one(cubes, tick_now):
    sp = speed()
    moving = sp > 0.0003
    vhat = (VEL["v"][0] / sp, VEL["v"][1] / sp) if moving else (1.0, 0.0)
    for c in cubes:
        s = c["c"][0] * vhat[0] + c["c"][1] * vhat[1]
        if moving and (VEL["smax"] is None or s > VEL["smax"]):
            VEL["smax"] = s
        if moving and (VEL["smin"] is None or s < VEL["smin"]):
            VEL["smin"] = s
    YAW_NOW["v"] = ee_yaw()
    if moving and not BELT["done"]:
        belt_extent(SNAPREF["s"], cubes)
    xs = exit_s() if moving else None
    cand = []
    keep = []
    for c in cubes:
        wanted = None
        for colour in need:
            if placed[colour] < need[colour] and colour_matches(c["colour"], colour):
                wanted = colour
        if wanted is None:
            continue
        if c["h"] > 0.09 or c["h"] < 0.008:
            continue
        upstream = 0.0
        if not moving and failed_here(c):
            continue
        if moving:
            best_axis(c, cubes)
            c["T"] = hover_lead(c, c["theta"])
            lead = (now_tick() - c["tick"]) + c["T"] + HOVER_MARGIN + extra_lead(c) + ND + NC + 2
            s_end = c["c"][0] * vhat[0] + c["c"][1] * vhat[1] + sp * lead
            if xs is not None and s_end > xs:
                continue
            upstream = -(c["c"][0] * vhat[0] + c["c"][1] * vhat[1])
        ax, clr = best_axis(c, cubes)
        merged = 1 if c["n"] > 2.2 * quantile([o["n"] for o in cubes], 0.5) else 0
        cand.append((0 if clr > 1.0 else 1, merged, upstream, len(cand), wanted, ax, clr))
        keep.append(c)
    if not cand:
        return None, None, None
    best = sorted(cand)[0]
    if best[0] == 1 and moving:
        # every candidate has a neighbour inside the finger sweep: wait for a clean one
        log("no clear candidate (best clearance %.2f)" % best[6])
        return None, None, None
    c = keep[best[3]]
    c["axis"] = best[5]
    c["clear"] = best[6]
    return c, best[4], best[5]


def face_angle(toppts):
    best = None
    a = 0
    while a < 90:
        th = math.radians(a)
        ca = math.cos(th)
        sa = math.sin(th)
        us = [p[0] * ca + p[1] * sa for p in toppts]
        vs = [-p[0] * sa + p[1] * ca for p in toppts]
        area = (max(us) - min(us)) * (max(vs) - min(vs))
        if best is None or area < best[0]:
            best = (area, th)
        a += 5
    return best[1]


def reobserve(cube):
    snap = snapshot()
    blobs, darkb, labels = find_blobs(snap)
    cubes = describe_cubes(snap, blobs, labels)
    dt = snap["tick"] - cube["tick"]
    pred = (cube["c"][0] + VEL["v"][0] * dt, cube["c"][1] + VEL["v"][1] * dt)
    best = None
    for c in cubes:
        if c["colour"] != cube["colour"]:
            continue
        d = math.sqrt((c["c"][0] - pred[0]) ** 2 + (c["c"][1] - pred[1]) ** 2)
        if best is None or d < best[0]:
            best = (d, c)
    if best is not None and best[0] < 0.03 and best[1]["n"] >= 0.6 * cube["n"]:
        return (best[1]["c"][0], best[1]["c"][1]), snap["tick"], round(best[0], 4)
    return pred, snap["tick"], None


def track_grasp(cxy, t_c, theta, z_from, z_to):
    # Joint-space stepping along a Cartesian path that follows the (moving) cube: descend in ND
    # ticks while matching its predicted position, then keep tracking for NC ticks with the
    # gripper commanded closed. Static scene: VEL is zero and this is a straight descent.
    st = get_robot_state()
    q = [float(v) for v in st.joint_positions["primary"]]
    ee = [float(v) for v in st.end_effector_poses["primary"].position]
    t0 = now_tick()
    vx = VEL["v"][0]
    vy = VEL["v"][1]
    c0 = (cxy[0] + vx * (t0 - t_c), cxy[1] + vy * (t0 - t_c))
    off = (ee[0] - c0[0], ee[1] - c0[1])
    quat = q_down(theta)
    plan = []
    seed = q
    fails = 0
    for k in range(1, ND + NC + 1):
        f = min(1.0, float(k) / ND)
        g = min(1.0, float(k) / (0.7 * ND))
        lag = LAG if k < ND + NC else 0
        px = c0[0] + vx * (k + lag) + off[0] * (1.0 - g)
        py = c0[1] + vy * (k + lag) + off[1] * (1.0 - g)
        pz = z_from + (z_to - z_from) * f
        qk = None
        try:
            ik = solve_ik(Pose((px, py, pz), quat, BASE), seed_joints=seed)
            if ik.ok:
                qk = ik_joints(ik)
        except Exception as exc:
            report["errors"].append("track ik: " + str(exc)[:200])
        if qk is None:
            fails += 1
            qk = seed
        plan.append((qk, 1.0 if k <= ND else 0.0))
        seed = qk
    for qk, gp in plan:
        sr = step(RobotAction(arms={"primary": ArmCommand("joint_position", qk, gripper_position=gp, embodiment=EMB)}))
        check_end(sr, "track_grasp")
    return {"off": [round(off[0], 4), round(off[1], 4)], "ik_fail": fails}


def attempt(cube, wanted, axis):
    BIN = SCENE["bin"]
    att = {"colour": wanted, "cube": [round(v, 4) for v in cube["c"]], "h": round(cube["h"], 4),
           "v": [round(VEL["v"][0], 5), round(VEL["v"][1], 5)], "t0": now_tick(),
           "clear": round(cube.get("clear", 0.0), 2)}
    report["attempts"].append(att)
    sp = speed()
    moving = sp > 0.0003
    h = min(max(cube["h"], 0.02), 0.07)
    grasp_z = cube["c"][2] - 0.55 * h
    grasp_z = max(grasp_z, cube["support"] + 0.008)
    z_hover = cube["c"][2] + 0.06
    att["grasp_z"] = round(grasp_z, 4)
    theta = cube.get("theta", theta_for_closing_axis(axis, math.pi))
    att["theta_deg"] = round(math.degrees(theta), 1)
    t_obs = cube["tick"]
    if moving:
        cube["T"] = hover_lead(cube, theta)
        att["T_pred"] = cube["T"]
    lead = cube["T"] + HOVER_MARGIN + extra_lead(cube) if moving else 0
    att["extra_lead"] = extra_lead(cube) if moving else 0
    dt = now_tick() - t_obs + lead
    hov = (cube["c"][0] + VEL["v"][0] * dt, cube["c"][1] + VEL["v"][1] * dt)
    att["hover"] = [round(hov[0], 4), round(hov[1], 4)]
    t1 = now_tick()
    r = goto((hov[0], hov[1], z_hover), theta, tol=0.01, tag="hover")
    att["pre"] = [bool(r.ok), now_tick() - t1]
    if not r.ok:
        return False
    if moving:
        EST["pre"] = int(0.5 * EST["pre"] + 0.5 * (now_tick() - t1))
        ratio = float(now_tick() - t1) / max(1.0, cube["T"] / MOVE["k"])
        MOVE["k"] = min(1.6, max(0.7, 0.5 * MOVE["k"] + 0.5 * ratio))
    cxy, t_c, err = reobserve(cube)
    att["reobs_err"] = err
    ee = get_robot_state().end_effector_poses["primary"].position
    if moving:
        vhat = (VEL["v"][0] / sp, VEL["v"][1] / sp)
        dtc = now_tick() - t_c
        cur = (cxy[0] + VEL["v"][0] * dtc, cxy[1] + VEL["v"][1] * dtc)
        along = (float(ee[0]) - cur[0]) * vhat[0] + (float(ee[1]) - cur[1]) * vhat[1]
        att["along"] = round(along, 4)
        wait = int(round(along / sp)) - 1
        att["wait"] = wait
        if wait > 0:
            hold(min(wait, 90), 1.0)
        elif along < -0.03:
            att["result"] = "passed"
            return False
    tg = track_grasp(cxy, t_c, theta, z_hover, grasp_z)
    att["track"] = tg
    r = close_gripper()
    check_end(r, "close")
    g_closed = grip_reading()
    att["grip_closed"] = round(g_closed, 3)
    ee = get_robot_state().end_effector_poses["primary"].position
    safe_z = max(BIN["rim"] + 0.16, grasp_z + 0.15)
    if moving:
        safe_z = max(BIN["rim"] + 0.13, grasp_z + 0.12)
    r = goto((float(ee[0]), float(ee[1]), safe_z), theta, tol=0.02, tag="lift")
    g_lift = grip_reading()
    att["grip_lift"] = round(g_lift, 3)
    if g_lift < 0.08:
        att["result"] = "empty"
        open_gripper()
        return False
    r = goto((BIN["xy"][0], BIN["xy"][1], safe_z), theta, tol=0.02, tag="to_bin")
    if not moving:
        r = goto((BIN["xy"][0], BIN["xy"][1], BIN["rim"] + 0.09), theta, tol=0.015, tag="bin_low")
    g_before = grip_reading()
    att["grip_before_release"] = round(g_before, 3)
    r = open_gripper()
    check_end(r, "open")
    if not moving:
        hold(4, 1.0)
    att["result"] = "released" if g_before > 0.08 else "lost_in_transit"
    att["t_end"] = now_tick()
    return g_before > 0.08


def press_button():
    BUTTON = SCENE["button"]
    if BUTTON is None:
        report["errors"].append("no button")
        return
    r = close_gripper()
    check_end(r, "close_for_press")
    theta = THETA_HOME
    bx, by = BUTTON["xy"]
    top = BUTTON["top"]
    goto((bx, by, top + 0.10), theta, tol=0.015, tag="press_above")
    goto((bx, by, top + 0.03), theta, tol=0.01, tag="press_near")
    goto((bx, by, top - 0.015), theta, tol=0.01, ms=80, tag="press")
    hold(6, 0.0)
    goto((bx, by, top + 0.10), theta, tol=0.02, tag="press_up")
    hold(10, 0.0)


try:
    snap0, blobs0, labels0 = setup_scene()
    report["max_steps"] = MAX_STEPS
    if SCENE["bin"] is None:
        raise ValueError("bin not found")
    open_gripper()
    total_needed = sum(need[c] for c in need)
    fails = 0
    cycles = 0
    while sum(placed[c] for c in placed) < total_needed and cycles < 400:
        cycles += 1
        if now_tick() > MAX_STEPS - 120:
            log("tick budget exhausted for cubes at %d" % now_tick())
            break
        if speed() > 0.0003 and VEL["known"]:
            snap = snapshot()
            b1, d1, l1 = find_blobs(snap)
            cubes = describe_cubes(snap, b1, l1)
        else:
            snap, cubes = observe_pair(6)
        SNAPREF["s"] = snap
        for c in cubes:
            c["tick"] = snap["tick"]
        log("cycle %d tick %d v=(%.5f,%.5f) cubes=%s" % (cycles, snap["tick"], VEL["v"][0], VEL["v"][1],
            ",".join("%s@%.3f,%.3f,%.3f" % (c["colour"][:3], c["c"][0], c["c"][1], c["c"][2]) for c in cubes)))
        cube, wanted, axis = pick_one(cubes, snap["tick"])
        if cube is None:
            if not AT_HOME["v"]:
                gh = go_home()
                check_end(gh, "home")
                AT_HOME["v"] = True
            else:
                hold(8 if speed() <= 0.0003 else 4, 1.0)
            continue
        AT_HOME["v"] = False
        ok = attempt(cube, wanted, axis)
        if ok:
            placed[wanted] += 1
        else:
            fails += 1
            FAILED.append((cube["c"][0], cube["c"][1]))
        report["placed"] = dict(placed)
        if speed() <= 0.0003 or not ok:
            gh = go_home()
            check_end(gh, "home")
            AT_HOME["v"] = True
            open_gripper()
    report["placed"] = dict(placed)
    report["tick_before_press"] = now_tick()
    press_button()
    report["tick_after_press"] = now_tick()
except Exception as exc:
    if ENDED["flag"]:
        report["ended"] = str(exc)
    else:
        report["errors"].append("fatal: " + str(exc)[:500])

report["moves"] = MOVES[:200]
report["placed"] = dict(placed)
try:
    report["final_tick"] = now_tick()
except Exception:
    report["final_tick"] = None
result = report
