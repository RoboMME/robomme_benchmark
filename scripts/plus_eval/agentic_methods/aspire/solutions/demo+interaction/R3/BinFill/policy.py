import math

# BinFill v00 — put N cubes of each named colour into the bin, then press the button.
# Every object choice and pose is derived at runtime from the goal text and live RGB-D.

report = {"version": "v01", "events": [], "picks": [], "errors": []}
S = {"q": None, "g": 1.0, "terminal": False, "ticks": 0, "emb": None, "last_reply": None}


def note(kind, data):
    if len(report["events"]) < 400:
        report["events"].append([kind, S["ticks"], data])


def err(where, exc):
    if len(report["errors"]) < 60:
        report["errors"].append([where, S["ticks"], str(exc)[:300]])


# ---------------------------------------------------------------- goal parsing
NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "a": 1, "an": 1, "single": 1}
COLOR_ALIASES = {"red": "red", "green": "green", "blue": "blue", "yellow": "yellow",
                 "cyan": "cyan", "orange": "orange", "magenta": "magenta", "purple": "magenta",
                 "pink": "magenta", "violet": "magenta"}
HUE_CENTERS = [["red", 0.0], ["orange", 40.0], ["yellow", 60.0], ["green", 120.0],
               ["cyan", 180.0], ["blue", 240.0], ["magenta", 300.0], ["red", 360.0]]


def parse_goal(text):
    words = []
    cur = ""
    for ch in text.lower():
        if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            cur += ch
        else:
            if cur:
                words.append(cur)
            cur = ""
    if cur:
        words.append(cur)
    need = {}
    for i in range(len(words) - 1):
        w = words[i]
        n = None
        if w in NUMBERS:
            n = NUMBERS[w]
        elif w.isdigit():
            n = int(w)
        if n is None:
            continue
        c = words[i + 1]
        if c in COLOR_ALIASES and i + 2 < len(words) and words[i + 2].startswith("cube"):
            key = COLOR_ALIASES[c]
            need[key] = need.get(key, 0) + n
    return need


# ---------------------------------------------------------------- math helpers
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def qnorm(q):
    n = math.sqrt(sum(v * v for v in q))
    return [v / n for v in q]


def qcol(q, j):
    w, x, y, z = q
    if j == 0:
        return [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)]
    if j == 1:
        return [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)]
    return [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]


def qaxis(axis, ang):
    n = math.sqrt(sum(v * v for v in axis))
    if n < 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    s = math.sin(ang / 2.0) / n
    return [math.cos(ang / 2.0), axis[0] * s, axis[1] * s, axis[2] * s]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def quant(vals, f):
    s = sorted(vals)
    if not s:
        return 0.0
    return s[int((len(s) - 1) * f)]


def wrap_half_pi(t):
    while t > math.pi / 2:
        t -= math.pi
    while t <= -math.pi / 2:
        t += math.pi
    return t


# ---------------------------------------------------------------- robot I/O
def is_terminal(reply):
    flag = False
    try:
        if reply.terminated:
            flag = True
    except Exception:
        pass
    try:
        if reply.truncated:
            flag = True
    except Exception:
        pass
    return flag


def tick(q, g):
    if S["terminal"]:
        return False
    try:
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=[float(v) for v in q],
            gripper_position=float(g), embodiment=S["emb"])}))
    except Exception as exc:
        err("step", exc)
        S["terminal"] = True
        return False
    S["ticks"] += 1
    S["q"] = [float(v) for v in q]
    S["g"] = float(g)
    S["last_reply"] = reply
    if is_terminal(reply):
        S["terminal"] = True
        note("terminal", {"ticks": S["ticks"]})
        return False
    return True


def now_s():
    try:
        return float(get_robot_state().timestamp_s)
    except Exception:
        return S["ticks"] / HZ


def measured_q():
    st = get_robot_state()
    return [float(v) for v in st.joint_positions["primary"]]


def grip_reading():
    try:
        return float(get_robot_state().gripper_positions["primary"])
    except Exception:
        return -1.0


def hold(n, g):
    for k in range(n):
        if not tick(S["q"], g):
            return False
    return True


def drive(qt, rate, g, settle):
    """Smooth (cosine) joint interpolation from the last command to qt at ~rate rad/tick avg."""
    q0 = S["q"]
    d = max(abs(a - b) for a, b in zip(q0, qt))
    n = max(1, int(math.ceil(d / rate)))
    for i in range(1, n + 1):
        s = 0.5 - 0.5 * math.cos(math.pi * i / n)
        if not tick([a + (b - a) * s for a, b in zip(q0, qt)], g):
            return False
    for k in range(settle):
        try:
            e = max(abs(a - b) for a, b in zip(measured_q(), qt))
        except Exception:
            e = 0.0
        if e < 0.008:
            break
        if not tick(qt, g):
            return False
    return True


def ik(pos, quat):
    try:
        res = solve_ik(Pose(position=[float(v) for v in pos], quaternion_wxyz=[float(v) for v in quat],
                            frame=BASE), seed_joints=S["q"])
        if res.ok:
            return [float(v) for v in res.joint_positions]
        note("ik_fail", {"pos": [round(v, 3) for v in pos]})
    except Exception as exc:
        err("ik", exc)
    return None


def ticks_for(qt, rate):
    return int(math.ceil(max(abs(a - b) for a, b in zip(S["q"], qt)) / rate))


def goto(pos, quat, rate, g, settle):
    qt = ik(pos, quat)
    if qt is None:
        return False
    return drive(qt, rate, g, settle)


# ---------------------------------------------------------------- perception
def to_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        rows = []
        for y in range(len(arr)):
            rows.append([[int(c) for c in px] for px in arr[y]])
        return rows


SAT_PCT = 86  # cube saturation gate (cubes >= 0.95; wood table <= 0.84 in recordings)


def hue_class(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = float(mx - mn)
    if mx == r:
        h = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        h = 60.0 * ((b - r) / d + 2.0)
    else:
        h = 60.0 * ((r - g) / d + 4.0)
    best = None
    bd = 999.0
    for name, c in HUE_CENTERS:
        if abs(h - c) < bd:
            bd = abs(h - c)
            best = name
    return best


def sat_ok(p):
    r = p[0]
    g = p[1]
    b = p[2]
    mx = max(r, g, b)
    if mx < 35:
        return False
    return (mx - min(r, g, b)) * 100 >= SAT_PCT * mx


def scan_blobs(rows, stride, x0, y0, x1, y1, min_px):
    H = len(rows)
    W = len(rows[0])
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W, x1)
    y1 = min(H, y1)
    grid = {}
    for y in range(y0, y1, stride):
        row = rows[y]
        for x in range(x0, x1, stride):
            p = row[x]
            if sat_ok(p):
                grid[(x, y)] = hue_class(p[0], p[1], p[2])
    seen = set()
    blobs = []
    for key in grid:
        if key in seen:
            continue
        col = grid[key]
        stack = [key]
        seen.add(key)
        comp = []
        while stack:
            k = stack.pop()
            comp.append(k)
            for dx in (-stride, 0, stride):
                for dy in (-stride, 0, stride):
                    nk = (k[0] + dx, k[1] + dy)
                    if nk in grid and nk not in seen and grid[nk] == col:
                        seen.add(nk)
                        stack.append(nk)
        xs = [k[0] for k in comp]
        ys = [k[1] for k in comp]
        bx0 = max(0, min(xs) - stride)
        bx1 = min(W, max(xs) + stride + 1)
        by0 = max(0, min(ys) - stride)
        by1 = min(H, max(ys) + stride + 1)
        px = []
        for yy in range(by0, by1):
            row = rows[yy]
            for xx in range(bx0, bx1):
                p = row[xx]
                if sat_ok(p) and hue_class(p[0], p[1], p[2]) == col:
                    px.append((xx, yy))
        if len(px) >= min_px:
            blobs.append({"color": col, "px": px,
                          "cx": sum(p[0] for p in px) / float(len(px)),
                          "cy": sum(p[1] for p in px) / float(len(px)), "n": len(px)})
    return blobs


def cloud_list(pc):
    try:
        pts = pc.points.tolist()
    except Exception:
        pts = [[float(p[0]), float(p[1]), float(p[2])] for p in pc.points]
    out = []
    for p in pts:
        if p[0] == p[0] and p[1] == p[1] and p[2] == p[2] and abs(p[0]) < 5 and abs(p[2]) < 5:
            out.append(p)
    return out


def project(cam, pixels, H, W):
    """Pixel list -> robot_base points via the pixel mask; fallback to a SAM point prompt."""
    if S.get("mask_mode", "list") == "list":
        try:
            mask = [[False] * W for y in range(H)]
            for p in pixels:
                mask[p[1]][p[0]] = True
            pc = mask_to_point_cloud(mask, cam, target_frame=BASE)
            pts = cloud_list(pc)
            if "mask_mode_ok" not in S:
                S["mask_mode_ok"] = True
                note("mask_mode", "list mask accepted")
            return pts
        except Exception as exc:
            err("mask_list", exc)
            S["mask_mode"] = "sam"
    try:
        cx = int(sum(p[0] for p in pixels) / len(pixels))
        cy = int(sum(p[1] for p in pixels) / len(pixels))
        seg = segment_points(cam, [[cx, cy]])
        if seg.ok and len(seg.segmentations) > 0:
            pc = mask_to_point_cloud(seg.segmentations[0], cam, target_frame=BASE)
            return cloud_list(pc)
    except Exception as exc:
        err("sam_points", exc)
    return []


def cube_geom(pts):
    if len(pts) < 4:
        return None
    zs = [p[2] for p in pts]
    zt = quant(zs, 0.92)
    zb = quant(zs, 0.05)
    top = [p for p in pts if p[2] >= zt - 0.007]
    if len(top) < 3:
        top = pts
    x = quant([p[0] for p in top], 0.5)
    y = quant([p[1] for p in top], 0.5)
    return {"x": x, "y": y, "zt": zt, "zb": zb, "top": top, "n": len(pts)}


def observe(cam, colors=None, nproj=12):
    """Scan colour blobs and project them to 3D IMMEDIATELY (before any tick advances the scene:
    the projection reads the camera's current depth, so a stale mask would land on the belt)."""
    obs = get_observation()
    t = float(obs.timestamp_s)
    c = obs.cameras[cam]
    rows = to_rows(c.rgb)
    H = len(rows)
    W = len(rows[0])
    blobs = scan_blobs(rows, 2, 0, 0, W, H, 6)
    order = sorted([[-x["n"], i] for i, x in enumerate(blobs)])
    k = 0
    for item in order:
        b = blobs[item[1]]
        if colors is not None and b["color"] not in colors:
            b["g"] = None
            continue
        if k >= nproj:
            b["g"] = None
            continue
        b["g"] = cube_geom(project(cam, b["px"], H, W))
        k += 1
    return {"t": t, "blobs": blobs, "H": H, "W": W, "cam": cam}


def blob3d(snap, b):
    return b.get("g")


# ---------------------------------------------------------------- setup
BASE = "robot_base"
HZ = 20.0
ctx = get_task_context()
goal = str(ctx.language)
need = parse_goal(goal)
report["goal"] = goal
report["need"] = dict(need)
max_steps = 1500
try:
    meta = robomme_plus.get_task_metadata()
    report["meta"] = {str(k): str(meta[k])[:120] for k in meta}
    try:
        max_steps = int(meta["max_steps"])
    except Exception:
        pass
    try:
        HZ = float(meta["control_hz"])
    except Exception:
        pass
except Exception as exc:
    err("meta", exc)
try:
    cmeta = robomme_plus.get_controller_metadata()
    report["controller"] = {str(k): str(cmeta[k])[:160] for k in cmeta}
except Exception as exc:
    err("cmeta", exc)
try:
    report["video_len"] = len(robomme_plus.get_conditioning_video())
except Exception as exc:
    err("video", exc)
report["max_steps"] = max_steps

st0 = get_robot_state()
S["emb"] = st0.embodiment
BASE = str(st0.base_frame)
S["q"] = [float(v) for v in st0.joint_positions["primary"]]
home_q = list(S["q"])
ee0 = st0.end_effector_poses["primary"]
q_home = [float(v) for v in ee0.quaternion_wxyz]
report["home_ee"] = [[round(float(v), 4) for v in ee0.position], [round(v, 4) for v in q_home]]
report["grip0"] = float(st0.gripper_positions["primary"])

# Top-down orientation: rotate the home orientation minimally so the tool z points straight down.
zt0 = qcol(q_home, 2)
down = [0.0, 0.0, -1.0]
ax = cross(zt0, down)
ang = math.acos(max(-1.0, min(1.0, dot(zt0, down))))
q_down = qnorm(qmul(qaxis(ax, ang), q_home)) if ang > 1e-3 else q_home
report["tool_z_home"] = [round(v, 3) for v in zt0]

# Closing axis of the fingers: the tool axis (x or y) parallel to the wrist image's horizontal
# axis (fingers are left/right in the wrist image). Camera convention detected from its pose.
close_axis_idx = 1
try:
    wc = get_observation().cameras["wrist"].camera_pose
    qc = [float(v) for v in wc.quaternion_wxyz]
    tz = qcol(q_home, 2)
    if abs(dot(qcol(qc, 2), tz)) >= abs(dot(qcol(qc, 0), tz)):
        img_right = qcol(qc, 0)          # OpenCV / OpenGL: x is image-right
        conv = "z-optical"
    else:
        img_right = [-v for v in qcol(qc, 1)]  # x-forward convention: -y is image-right
        conv = "x-optical"
    dx = abs(dot(img_right, qcol(q_home, 0)))
    dy = abs(dot(img_right, qcol(q_home, 1)))
    close_axis_idx = 0 if dx > dy else 1
    report["close_axis"] = {"conv": conv, "dot_x": round(dx, 3), "dot_y": round(dy, 3),
                            "idx": close_axis_idx}
except Exception as exc:
    err("close_axis", exc)


def grasp_quat(axis_xy):
    """Top-down quaternion whose finger closing axis is parallel to axis_xy (min yaw change)."""
    ca = qcol(q_down, close_axis_idx)
    th = wrap_half_pi(math.atan2(axis_xy[1], axis_xy[0]) - math.atan2(ca[1], ca[0]))
    return qnorm(qmul(qaxis([0.0, 0.0, 1.0], th), q_down))


def cube_grasp_yaw(axis_xy):
    """Yaw (about base z, applied to q_down) putting the closing axis on a cube face normal; |yaw|<=45 deg."""
    ca = qcol(q_down, close_axis_idx)
    t = math.atan2(axis_xy[1], axis_xy[0]) - math.atan2(ca[1], ca[0])
    while t > math.pi / 4:
        t -= math.pi / 2
    while t <= -math.pi / 4:
        t += math.pi / 2
    return t


def quat_yaw(th):
    return qnorm(qmul(qaxis([0.0, 0.0, 1.0], th), q_down))


def ee_pos():
    p = get_robot_state().end_effector_poses["primary"].position
    return [float(p[0]), float(p[1]), float(p[2])]


# ---------------------------------------------------------------- bin and button
def locate_bin():
    obs = get_observation()
    rows = to_rows(obs.cameras["front"].rgb)
    H = len(rows)
    W = len(rows[0])
    dark = set()
    for y in range(0, H, 2):
        for x in range(0, W, 2):
            p = rows[y][x]
            if max(p[0], p[1], p[2]) < 35:
                dark.add((x, y))
    best = None
    seen = set()
    for k in dark:
        if k in seen:
            continue
        comp = []
        stack = [k]
        seen.add(k)
        while stack:
            c = stack.pop()
            comp.append(c)
            for dx in (-2, 0, 2):
                for dy in (-2, 0, 2):
                    nk = (c[0] + dx, c[1] + dy)
                    if nk in dark and nk not in seen:
                        seen.add(nk)
                        stack.append(nk)
        if best is None or len(comp) > len(best):
            best = comp
    if best is None or len(best) < 6:
        return locate_bin_sam()
    xs = [c[0] for c in best]
    ys = [c[1] for c in best]
    bx0 = min(xs)
    bx1 = max(xs)
    by0 = min(ys)
    by1 = max(ys)
    m = max(4, int(0.5 * max(bx1 - bx0, by1 - by0)))
    inner = []
    ring = []
    for y in range(max(0, by0 - m), min(H, by1 + m + 1)):
        for x in range(max(0, bx0 - m), min(W, bx1 + m + 1)):
            p = rows[y][x]
            if max(p[0], p[1], p[2]) < 35:
                if bx0 <= x <= bx1 and by0 <= y <= by1:
                    inner.append((x, y))
            else:
                ring.append((x, y))
    rpts = project("front", ring, H, W)
    ipts = project("front", inner, H, W)
    if len(rpts) < 6:
        return locate_bin_sam()
    rim_z = quant([p[2] for p in rpts], 0.9)
    rim = [p for p in rpts if p[2] >= rim_z - 0.012]
    cx = 0.5 * (quant([p[0] for p in rim], 0.08) + quant([p[0] for p in rim], 0.92))
    cy = 0.5 * (quant([p[1] for p in rim], 0.08) + quant([p[1] for p in rim], 0.92))
    span = min(quant([p[0] for p in rim], 0.92) - quant([p[0] for p in rim], 0.08),
               quant([p[1] for p in rim], 0.92) - quant([p[1] for p in rim], 0.08))
    floor_z = quant([p[2] for p in ipts], 0.1) if ipts else rim_z - 0.05
    table_z = quant([p[2] for p in rpts], 0.1)
    return {"x": cx, "y": cy, "rim_z": rim_z, "floor_z": floor_z, "table_z": table_z,
            "span": span, "px": [int(sum(xs) / len(xs)), int(sum(ys) / len(ys))],
            "n_ring": len(rpts), "n_inner": len(ipts)}


def locate_bin_sam():
    for prompt in ("bin", "box", "open box"):
        try:
            found = localize_object(prompt, camera_name="front", target_frame=BASE)
        except Exception as exc:
            err("bin_sam", exc)
            continue
        if not found.ok:
            continue
        pts = cloud_list(found.point_cloud)
        if len(pts) < 10:
            continue
        rim_z = quant([p[2] for p in pts], 0.9)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return {"x": 0.5 * (quant(xs, 0.08) + quant(xs, 0.92)),
                "y": 0.5 * (quant(ys, 0.08) + quant(ys, 0.92)), "rim_z": rim_z,
                "floor_z": quant([p[2] for p in pts], 0.1), "table_z": quant([p[2] for p in pts], 0.05),
                "span": quant(xs, 0.92) - quant(xs, 0.08), "px": [0, 0], "n_ring": len(pts),
                "n_inner": 0, "src": prompt}
    return None


def locate_button(binfo):
    cands = []
    for prompt in ("button", "white button", "push button"):
        try:
            seg = segment_text("front", prompt)
        except Exception as exc:
            err("seg_button", exc)
            continue
        if not seg.ok:
            continue
        k = 0
        for cand in seg.segmentations:
            k += 1
            if k > 6:
                break
            try:
                pts = cloud_list(mask_to_point_cloud(cand, "front", target_frame=BASE))
            except Exception as exc:
                err("button_pc", exc)
                continue
            if len(pts) < 5:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            zs = [p[2] for p in pts]
            ext = max(quant(xs, 0.95) - quant(xs, 0.05), quant(ys, 0.95) - quant(ys, 0.05))
            zr = quant(zs, 0.95) - quant(zs, 0.05)
            bx = quant(xs, 0.5)
            by = quant(ys, 0.5)
            dbase = math.sqrt(bx * bx + by * by)
            dbin = 1.0
            if binfo is not None:
                dbin = math.sqrt((bx - binfo["x"]) ** 2 + (by - binfo["y"]) ** 2)
            score = 0.0
            try:
                score = float(cand.score)
            except Exception:
                pass
            ok = ext < 0.12 and zr < 0.08 and dbase > 0.15 and dbin > 0.06
            cands.append({"prompt": prompt, "x": bx, "y": by, "z": quant(zs, 0.95),
                          "ext": ext, "zr": zr, "score": score, "ok": ok})
        if any(c["ok"] for c in cands):
            break
    best = None
    for c in cands:
        if c["ok"] and (best is None or c["score"] > best["score"]):
            best = c
    report["button_cands"] = [{k: (round(v, 3) if k not in ("prompt", "ok") else v)
                               for k, v in c.items()} for c in cands[:8]]
    return best


binfo = None
button = None
try:
    binfo = locate_bin()
except Exception as exc:
    err("bin", exc)
report["bin"] = None if binfo is None else {k: (round(v, 4) if k != "px" else v)
                                            for k, v in binfo.items()}
try:
    button = locate_button(binfo)
except Exception as exc:
    err("button", exc)
report["button"] = button

# ---------------------------------------------------------------- belt velocity
V = {"v": None, "speed": 0.0, "mode": None, "pv": [0.0, 0.0]}


def match(b, blobs, maxd):
    best = None
    bd = maxd
    for o in blobs:
        if o["color"] != b["color"]:
            continue
        d = math.sqrt((o["cx"] - b["cx"]) ** 2 + (o["cy"] - b["cy"]) ** 2)
        if d < bd:
            bd = d
            best = o
    return best


def estimate_velocity(a, b):
    dt = b["t"] - a["t"]
    if dt <= 0.05:
        return None
    big = sorted([[-x["n"], i] for i, x in enumerate(a["blobs"])])
    vs = []
    pvs = []
    prior = V["v"] is not None
    for item in big[:4]:
        ba = a["blobs"][item[1]]
        if prior:
            pred = {"color": ba["color"], "cx": ba["cx"] + V["pv"][0] * dt,
                    "cy": ba["cy"] + V["pv"][1] * dt}
            bb = match(pred, b["blobs"], 7.0)
        else:
            bb = match(ba, b["blobs"], 4.0 + 30.0 * dt)
        if bb is None:
            continue
        ga = blob3d(a, ba)
        gb = blob3d(b, bb)
        if ga is None or gb is None:
            continue
        vs.append([(gb["x"] - ga["x"]) / dt, (gb["y"] - ga["y"]) / dt])
        pvs.append([(bb["cx"] - ba["cx"]) / dt, (bb["cy"] - ba["cy"]) / dt])
    if not vs:
        return None
    vx = quant([v[0] for v in vs], 0.5)
    vy = quant([v[1] for v in vs], 0.5)
    pvx = quant([v[0] for v in pvs], 0.5)
    pvy = quant([v[1] for v in pvs], 0.5)
    return [vx, vy, pvx, pvy, len(vs)]


def update_velocity(est):
    if est is None:
        return
    sp = math.sqrt(est[0] ** 2 + est[1] ** 2)
    if V["v"] is not None and V["mode"] == "belt" and V.get("n", 0) >= 3:
        # established belt: reject outliers (a mismatched blob pair once tilted the direction by 30 deg)
        cosang = (est[0] * V["v"][0] + est[1] * V["v"][1]) / max(1e-6, sp * V["speed"])
        if sp < 0.7 * V["speed"] or sp > 1.4 * V["speed"] or cosang < 0.966:
            note("velocity_reject", {"est": [round(est[0], 4), round(est[1], 4)]})
            return
    V["n"] = V.get("n", 0) + 1
    if V["v"] is None:
        V["v"] = [est[0], est[1]]
        V["pv"] = [est[2], est[3]]
    else:
        V["v"] = [0.5 * V["v"][0] + 0.5 * est[0], 0.5 * V["v"][1] + 0.5 * est[1]]
        V["pv"] = [0.5 * V["pv"][0] + 0.5 * est[2], 0.5 * V["pv"][1] + 0.5 * est[3]]
    V["speed"] = math.sqrt(V["v"][0] ** 2 + V["v"][1] ** 2)
    V["mode"] = "belt" if V["speed"] > 0.008 else "static"
    note("velocity", {"est": [round(est[0], 4), round(est[1], 4), est[4]],
                      "v": [round(V["v"][0], 4), round(V["v"][1], 4)], "mode": V["mode"]})


# ---------------------------------------------------------------- pick helpers
counts = {}
for c in need:
    counts[c] = 0


def remaining():
    return [c for c in need if counts[c] < need[c]]


def ticks_left():
    return max_steps - S["ticks"]


def held_color():
    """Colour of the object between the fingers: saturated wrist pixels at the closest depth."""
    try:
        obs = get_observation()
        cam = obs.cameras["wrist"]
        rows = to_rows(cam.rgb)
        dep = to_rows(cam.depth_m)
    except Exception as exc:
        err("held_color", exc)
        return None, 0
    H = len(rows)
    W = len(rows[0])
    sat = []
    for y in range(0, H, 3):
        for x in range(0, W, 3):
            p = rows[y][x]
            if sat_ok(p):
                d = float(dep[y][x])
                if d == d and d > 0:
                    sat.append([d, hue_class(p[0], p[1], p[2])])
    if not sat:
        return None, 0
    dmin = min(s[0] for s in sat)
    near = [s[1] for s in sat if s[0] <= dmin + 0.03 and s[0] < 0.25]
    if len(near) < 4:
        return None, len(near)
    tally = {}
    for c in near:
        tally[c] = tally.get(c, 0) + 1
    best = None
    for c in tally:
        if best is None or tally[c] > tally[best]:
            best = c
    return best, len(near)


def cube_yaw_axis(top):
    """Horizontal face-normal direction of a cube from its top-face points (min-area box)."""
    if len(top) < 5:
        return [1.0, 0.0]
    best = None
    ba = None
    for k in range(30):
        th = k * math.pi / 60.0
        c = math.cos(th)
        s = math.sin(th)
        u = [p[0] * c + p[1] * s for p in top]
        w = [-p[0] * s + p[1] * c for p in top]
        area = (max(u) - min(u)) * (max(w) - min(w))
        if ba is None or area < ba:
            ba = area
            best = th
    return [math.cos(best), math.sin(best)]


def lane_conflicts(g, others, u, n, a_park):
    k = 0
    for o in others:
        rx = o["x"] - g["x"]
        ry = o["y"] - g["y"]
        a = rx * u[0] + ry * u[1]
        l = rx * n[0] + ry * n[1]
        if 0.0 < a < a_park + 0.05 and abs(l) < 0.075:
            k += 1
    return k


def drop_to_bin(color):
    hover_z = binfo["rim_z"] + 0.12
    ok = goto([binfo["x"], binfo["y"], hover_z], q_down, 0.045, 0.0, 4)
    if not ok:
        return False
    g_before = grip_reading()
    hold(7, 1.0)
    counts[color] = counts[color] + 1
    note("drop", {"color": color, "grip_before": round(g_before, 3), "counts": dict(counts)})
    return True


def try_pick():
    """One pick attempt. Returns 'placed', 'retry', 'wait' or 'stop'."""
    rem = remaining()
    if V["mode"] == "belt":
        snap = observe("front", None, 16)
    else:
        snap = observe("front", rem, 8)   # static: only target colours need 3D (bounded RPC cost)
    cands = [b for b in snap["blobs"] if b["color"] in rem]
    if V["v"] is None or len(snap["blobs"]) == 0:
        return "wait"
    if not cands:
        return "wait"
    # 3D for candidates and neighbours (bounded)
    allg = []
    for bb in snap["blobs"]:
        g = blob3d(snap, bb)
        if g is not None:
            allg.append([bb, g])
    ztops = [g["zt"] for bb, g in allg]
    zmed = quant(ztops, 0.5) if ztops else 0.0
    usable = []
    for bb, g in allg:
        if bb["color"] not in rem:
            continue
        if g["zt"] > zmed + 0.04:      # lifted / held object
            continue
        r = math.sqrt(g["x"] ** 2 + g["y"] ** 2)
        if r < 0.2 or r > 0.85:
            continue
        if binfo is not None and math.sqrt((g["x"] - binfo["x"]) ** 2 + (g["y"] - binfo["y"]) ** 2) < 0.06:
            continue
        usable.append([bb, g])
    if not usable:
        return "wait"
    belt = V["mode"] == "belt"
    if belt:
        # v01: velocity-matched descent ("chase") yaw-aligned to the cube faces. v00 parked the open
        # fingers on the path: yawed cubes jammed against the fingers and were squeezed out (ep1).
        u = [V["v"][0] / V["speed"], V["v"][1] / V["speed"]]
        best = None
        for bb, g in usable:
            if abs(g["zt"] - zmed) > 0.009:       # not riding on the belt (fallen onto the table)
                continue
            if binfo is not None and math.sqrt((g["x"] - binfo["x"]) ** 2 + (g["y"] - binfo["y"]) ** 2) < 0.13:
                continue
            s_up = g["x"] * u[0] + g["y"] * u[1]
            conf = 0
            for o in allg:
                if o[1] is g:
                    continue
                if math.sqrt((o[1]["x"] - g["x"]) ** 2 + (o[1]["y"] - g["y"]) ** 2) < 0.065:
                    conf += 1
            key = [conf, s_up]
            if best is None or key < best[0]:
                best = [key, bb, g]
        if best is None:
            return "wait"
        bb, g = best[1], best[2]
        color = bb["color"]
        zg = max(g["zt"] - 0.018, g["zb"] + 0.008)
        th0 = cube_grasp_yaw(cube_yaw_axis(g["top"]))
        qg = quat_yaw(th0)
        hz = zg + 0.09
        t0 = snap["t"]
        lead = 2.0
        qh = None
        P = None
        for it in range(3):
            P = [g["x"] + V["v"][0] * lead, g["y"] + V["v"][1] * lead]
            qh = ik([P[0], P[1], hz], qg)
            if qh is None:
                break
            lead = (now_s() - t0) + (ticks_for(qh, 0.045) + 3) / HZ + 0.05
        rec = {"mode": "belt-chase", "color": color, "cube": [round(g["x"], 3), round(g["y"], 3),
               round(g["zt"], 3), round(g["zb"], 3)], "conf": best[0][0], "t_snap": t0,
               "yaw0": round(th0, 3)}
        if qh is None:
            note("pick_skip", {"why": "ik hover", "color": color})
            report["picks"].append(rec)
            hold(2, 1.0)
            return "retry"
        rec["P"] = [round(P[0], 3), round(P[1], 3)]
        rec["lead"] = round(lead, 2)
        if not drive(qh, 0.045, 1.0, 3):
            report["picks"].append(rec)
            return "stop"
        cx, cy, tob = g["x"], g["y"], t0
        th1 = th0
        # refine position and yaw from the (near top-down) wrist camera at the hover
        try:
            sn = observe("wrist", [color], 4)
            tn = sn["t"]
            pcx = cx + V["v"][0] * (tn - tob)
            pcy = cy + V["v"][1] * (tn - tob)
            found = None
            fd = 0.05
            for o in sn["blobs"]:
                go = o.get("g")
                if o["color"] != color or go is None or o["n"] < 12:
                    continue
                d = math.sqrt((go["x"] - pcx) ** 2 + (go["y"] - pcy) ** 2)
                if d < fd:
                    fd = d
                    found = go
            if found is not None:
                cx, cy, tob = found["x"], found["y"], tn
                rec["wrist_fix"] = [round(fd, 3), len(found["top"])]
                if len(found["top"]) >= 15:
                    th1 = cube_grasp_yaw(cube_yaw_axis(found["top"]))
                    rec["yaw1"] = round(th1, 3)
        except Exception as exc:
            err("wrist_fix", exc)
        if th1 - th0 > math.pi / 4:
            th1 -= math.pi / 2
        if th1 - th0 < -math.pi / 4:
            th1 += math.pi / 2
        z0 = ee_pos()[2]
        bias = [0.0, 0.0]
        N = 24
        LEADT = 0.10
        k = 0
        closing = 0
        last_err = None
        aborted = False
        while k < N + 22:
            tn = now_s()
            sdesc = min(1.0, k / float(N))
            sdesc = 0.5 - 0.5 * math.cos(math.pi * sdesc)
            zc = z0 + (zg - z0) * sdesc
            tc = tn + LEADT
            th = th0 + (th1 - th0) * min(1.0, k / (0.6 * N))
            qt = ik([cx + V["v"][0] * (tc - tob) + bias[0], cy + V["v"][1] * (tc - tob) + bias[1], zc],
                    quat_yaw(th))
            if qt is None:
                aborted = True
                break
            if not tick(qt, 0.0 if closing > 0 else 1.0):
                report["picks"].append(rec)
                return "stop"
            k += 1
            ep = ee_pos()
            tn2 = now_s()
            ex = cx + V["v"][0] * (tn2 - tob) - ep[0]
            ey = cy + V["v"][1] * (tn2 - tob) - ep[1]
            bias = [max(-0.025, min(0.025, bias[0] + 0.25 * ex)), max(-0.025, min(0.025, bias[1] + 0.25 * ey))]
            last_err = math.sqrt(ex * ex + ey * ey)
            if closing > 0:
                closing += 1
                if closing > 4:
                    break
                continue
            if k % 3 == 0 and ep[2] > zg + 0.05:
                try:
                    sn = observe("wrist", [color], 3)
                    pcx = cx + V["v"][0] * (sn["t"] - tob)
                    pcy = cy + V["v"][1] * (sn["t"] - tob)
                    found = None
                    fd = 0.03
                    for o in sn["blobs"]:
                        go = o.get("g")
                        if o["color"] != color or go is None or o["n"] < 12:
                            continue
                        d = math.sqrt((go["x"] - pcx) ** 2 + (go["y"] - pcy) ** 2)
                        if d < fd:
                            fd = d
                            found = go
                    if found is not None:
                        cx = 0.5 * pcx + 0.5 * found["x"]
                        cy = 0.5 * pcy + 0.5 * found["y"]
                        tob = sn["t"]
                        rec["track_wrist"] = rec.get("track_wrist", 0) + 1
                except Exception as exc:
                    err("track", exc)
            if k >= N and abs(ep[2] - zg) < 0.008 and last_err < 0.008:
                closing = 1
            elif k >= N + 12:
                closing = 1
                rec["forced_close"] = True
        rec["chase_ticks"] = k
        rec["close_err"] = None if last_err is None else round(last_err, 4)
        rec["bias"] = [round(bias[0], 4), round(bias[1], 4)]
        if aborted or closing == 0:
            note("chase_abort", {"color": color, "k": k})
            ep = ee_pos()
            goto([ep[0], ep[1], zg + 0.10], quat_yaw(th1), 0.04, 1.0, 2)
            report["picks"].append(rec)
            return "retry" if not S["terminal"] else "stop"
        qg = quat_yaw(th1)
        ep = ee_pos()
        lift_xy = [ep[0], ep[1]]
    else:
        best = None
        ee = get_robot_state().end_effector_poses["primary"].position
        for bb, g in usable:
            iso = 1.0
            for o in allg:
                if o[1] is g:
                    continue
                d = math.sqrt((o[1]["x"] - g["x"]) ** 2 + (o[1]["y"] - g["y"]) ** 2)
                iso = min(iso, d)
            dist = math.sqrt((g["x"] - float(ee[0])) ** 2 + (g["y"] - float(ee[1])) ** 2)
            key = [0 if iso > 0.06 else 1, dist]
            if best is None or key < best[0]:
                best = [key, bb, g]
        bb, g = best[1], best[2]
        color = bb["color"]
        axis = cube_yaw_axis(g["top"])
        qa = grasp_quat(axis)
        qb = grasp_quat([-axis[1], axis[0]])
        qg = qa if abs(dot(qa, q_down)) >= abs(dot(qb, q_down)) else qb
        zg = max(g["zt"] - 0.018, g["zb"] + 0.008)
        rec = {"mode": "static", "color": color, "cube": [round(g["x"], 3), round(g["y"], 3),
               round(g["zt"], 3), round(g["zb"], 3)], "iso": round(best[0][0], 3)}
        if not goto([g["x"], g["y"], zg + 0.10], qg, 0.045, 1.0, 4):
            report["picks"].append(rec)
            return "retry" if not S["terminal"] else "stop"
        if not goto([g["x"], g["y"], zg], qg, 0.03, 1.0, 4):
            report["picks"].append(rec)
            return "retry" if not S["terminal"] else "stop"
        lift_xy = [g["x"], g["y"]]
    # close (arm held), wait until the fingers stop moving
    prev = grip_reading()
    for k in range(14):
        if not tick(S["q"], 0.0):
            report["picks"].append(rec)
            return "stop"
        cur = grip_reading()
        if k >= 4 and abs(cur - prev) < 0.004:
            break
        prev = cur
    gclose = grip_reading()
    rec["grip_closed"] = round(gclose, 3)
    if not goto([lift_xy[0], lift_xy[1], zg + 0.13], qg, 0.035, 0.0, 3):
        report["picks"].append(rec)
        return "stop" if S["terminal"] else "retry"
    glift = grip_reading()
    hc, hn = held_color()
    rec["grip_lift"] = round(glift, 3)
    rec["held_color"] = [hc, hn]
    holding = glift > 0.06
    if not holding:
        note("pick_miss", {"color": color})
        hold(3, 1.0)
        report["picks"].append(rec)
        return "retry"
    if hc is not None and hc != color:
        # wrong cube between the fingers: put it back down beside the lift point
        note("wrong_color", {"want": color, "held": hc})
        if hc in need and counts[hc] < need[hc]:
            color = hc
        else:
            goto([lift_xy[0], lift_xy[1], zg + 0.03], qg, 0.03, 0.0, 2)
            hold(6, 1.0)
            goto([lift_xy[0], lift_xy[1], zg + 0.13], qg, 0.04, 1.0, 2)
            report["picks"].append(rec)
            return "retry"
    if binfo is None:
        report["picks"].append(rec)
        return "stop"
    ok = drop_to_bin(color)
    rec["placed"] = ok
    report["picks"].append(rec)
    return "placed" if ok else ("stop" if S["terminal"] else "retry")


def press_button():
    if button is None:
        note("button", "not found")
        return False
    bx, by, bz = button["x"], button["y"], button["z"]
    hold(8, 0.0)
    if not goto([bx, by, bz + 0.10], q_down, 0.045, 0.0, 4):
        return False
    for depth in (0.012, 0.025):
        if not goto([bx, by, bz + 0.03], q_down, 0.03, 0.0, 2):
            return False
        if not goto([bx, by, bz - depth], q_down, 0.015, 0.0, 6):
            note("button_press", {"depth": depth, "terminal": S["terminal"]})
            return S["terminal"]
        hold(4, 0.0)
        note("button_press", {"depth": depth, "terminal": S["terminal"]})
        if S["terminal"]:
            return True
        goto([bx, by, bz + 0.06], q_down, 0.03, 0.0, 2)
        if S["terminal"]:
            return True
    return S["terminal"]


# ---------------------------------------------------------------- main
try:
    hold(2, 1.0)
    # velocity / scene-mode estimate (also waits for the first cubes to appear)
    snapA = observe("front", None, 5)
    tries = 0
    while V["v"] is None and tries < 40 and not S["terminal"]:
        hold(6, 1.0)
        snapB = observe("front", None, 8)
        if snapA["blobs"] and snapB["blobs"]:
            update_velocity(estimate_velocity(snapA, snapB))
        snapA = snapB
        tries += 1
    if V["v"] is not None and not S["terminal"]:
        snapA = observe("front", None, 5)
        hold(16, 1.0)
        snapB = observe("front", None, 8)
        update_velocity(estimate_velocity(snapA, snapB))
    report["velocity_init"] = {"v": V["v"], "mode": V["mode"], "tries": tries}
    attempts = 0
    waits = 0
    while remaining() and not S["terminal"] and attempts < 25:
        if ticks_left() < 140:
            note("budget", "stop picking to keep ticks for the button")
            break
        res = try_pick()
        note("pick_result", res)
        if res == "stop":
            break
        if res == "wait":
            waits += 1
            if waits % 3 == 1:
                snapA = observe("front", None, 5)
                hold(8, S["g"])
                snapB = observe("front", None, 8)
                if snapA["blobs"] and snapB["blobs"]:
                    update_velocity(estimate_velocity(snapA, snapB))
            else:
                hold(8, S["g"])
            continue
        attempts += 1
    report["counts"] = dict(counts)
    report["waits"] = waits
    if not S["terminal"]:
        report["button_pressed"] = press_button()
except Exception as exc:
    err("main", exc)

report["ticks"] = S["ticks"]
report["terminal"] = S["terminal"]
report["counts"] = dict(counts)
result = report
