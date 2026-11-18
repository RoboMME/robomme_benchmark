# ButtonUnmask v01 -- memory: track coloured cubes over time and remember where each was covered.
# Diagnosis of v00 (ep 0): nothing coloured is visible at t=0; cubes appear later during
# execution and containers drop onto them later still, so a t=0 snapshot remembers nothing.
# v00 also classified the wooden table as one big "orange" blob and a gold cube as "yellow".
# v01 changes:
#   - background-adaptive colour classifier (table hue band needs high saturation),
#   - per-tick tracker of colour blobs during the press and while parked; a track that
#     disappears while the arm is parked and never reappears = cube now covered at that pixel,
#   - wait for the scene to become stable, then for each colour named in the goal (in order):
#     segment the container at the remembered pixel, min-area-rectangle yaw, top-down grasp of
#     its top, lift, verify the revealed colour, put it back down (except the last one).
# The press itself is unchanged from v00 (closed-gripper joint servo; v00 contact-stalled on the cap).
import math

report = {"version": "v01", "phase": "start", "errors": [], "events": []}
result = report
alive = {"ok": True}
COLOR_WORDS = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "magenta"]
BG = {"lo": 999.0, "hi": -999.0, "smin": 0.8}
TRACKS = []
STATE = {"sig": "", "last_change": 0.0, "obs": 0}


def note_err(where, exc):
    report["errors"].append(where + ": " + str(exc)[:400])


def ev(msg):
    report["events"].append(msg)


def plain(x, depth=0):
    if x is None or x is True or x is False:
        return x
    try:
        return (x + "")[:400]
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        pass
    if depth > 4:
        return str(x)[:200]
    try:
        out = {}
        for k, v in x.items():
            out[str(k)] = plain(v, depth + 1)
        return out
    except Exception:
        pass
    try:
        seq = list(x)
        return [plain(v, depth + 1) for v in seq[:48]]
    except Exception:
        pass
    return str(x)[:200]


def to_list(a):
    try:
        return a.tolist()
    except Exception:
        return [list(r) for r in a]


def quantile(vals, f):
    s = sorted(vals)
    if len(s) == 0:
        return None
    return s[int((len(s) - 1) * f)]


def is_terminal(res):
    t = False
    try:
        t = t or bool(res.terminated)
    except Exception:
        pass
    try:
        t = t or bool(res.truncated)
    except Exception:
        pass
    return t


def now_tick():
    try:
        return round(float(get_robot_state().timestamp_s) * 20.0, 2)
    except Exception:
        return None


# ---------------- colour classification ----------------
def hsv(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = mx - mn
    if mx <= 0:
        return 0.0, 0.0, 0
    s = d / float(mx)
    if d == 0:
        return 0.0, s, mx
    if mx == r:
        h = 60.0 * (((g - b) / float(d)) % 6.0)
    elif mx == g:
        h = 60.0 * ((b - r) / float(d) + 2.0)
    else:
        h = 60.0 * ((r - g) / float(d) + 4.0)
    return h, s, mx


def calibrate_background(rows):
    """Dominant saturated hue band of the empty scene (e.g. a wooden table) and the
    saturation a pixel inside that band needs before it counts as a coloured object."""
    hues = []
    for i in range(0, len(rows), 4):
        row = rows[i]
        for j in range(0, len(row), 4):
            p = row[j]
            h, s, v = hsv(p[0], p[1], p[2])
            if s > 0.3 and v >= 60:
                hues.append((h, s))
    if len(hues) < 200:
        return
    hs = [x[0] for x in hues]
    # dominant band: 20-degree window holding the most saturated pixels
    best = None
    for start in range(0, 360, 5):
        cnt = 0
        for h in hs:
            if (h - start) % 360.0 < 20.0:
                cnt += 1
        if best is None or cnt > best[0]:
            best = (cnt, start)
    if best[0] < 0.3 * len(hs):
        return
    lo = best[1] - 10.0
    hi = best[1] + 20.0 + 15.0
    ins = [x[1] for x in hues if (x[0] - lo) % 360.0 < (hi - lo)]
    BG["lo"] = lo
    BG["hi"] = hi
    BG["smin"] = max(0.8, quantile(ins, 0.999) + 0.08)


def in_bg_band(h):
    if BG["hi"] < BG["lo"]:
        return False
    return (h - BG["lo"]) % 360.0 < (BG["hi"] - BG["lo"])


def color_of(r, g, b):
    h, s, v = hsv(r, g, b)
    if v < 60 or s < 0.4:
        return None
    if in_bg_band(h) and s < BG["smin"]:
        return None
    if h < 15 or h >= 340:
        return "red"
    if h < 48:
        return "orange"
    if h < 75:
        return "yellow"
    if h < 165:
        return "green"
    if h < 195:
        return "cyan"
    if h < 255:
        return "blue"
    if h < 300:
        return "purple"
    return "pink"


def blob_key(b):
    return -b["n"]


def blobs(rows, stride, min_cells):
    H = len(rows)
    W = len(rows[0])
    lab = {}
    for i in range(H // stride):
        row = rows[i * stride]
        for j in range(W // stride):
            p = row[j * stride]
            c = color_of(p[0], p[1], p[2])
            if c is not None:
                lab[(i, j)] = c
    seen = set()
    out = []
    for key in lab:
        if key in seen:
            continue
        c = lab[key]
        stack = [key]
        seen.add(key)
        cells = []
        while stack:
            k = stack.pop()
            cells.append(k)
            for nb in ((k[0] + 1, k[1]), (k[0] - 1, k[1]), (k[0], k[1] + 1), (k[0], k[1] - 1)):
                if nb in lab and nb not in seen and lab[nb] == c:
                    seen.add(nb)
                    stack.append(nb)
        if len(cells) >= min_cells:
            us = [cc[1] * stride for cc in cells]
            vs = [cc[0] * stride for cc in cells]
            out.append({"color": c, "n": len(cells),
                        "u": round(sum(us) / len(us), 1), "v": round(sum(vs) / len(vs), 1)})
    out.sort(key=blob_key)
    return out[:24]


# ---------------- tracker ----------------
def observe(moving):
    """One front-camera look: update colour tracks; returns the blob list."""
    try:
        obs = get_observation()
        tick = round(float(obs.timestamp_s) * 20.0, 2)
        rows = to_list(obs.cameras["front"].rgb)
    except Exception as exc:
        note_err("observe", exc)
        return None
    bl = blobs(rows, 2, 6)
    STATE["obs"] += 1
    matched = set()
    for b in bl:
        best = None
        for k in range(len(TRACKS)):
            t = TRACKS[k]
            if t["color"] != b["color"] or k in matched:
                continue
            d = abs(t["u"] - b["u"]) + abs(t["v"] - b["v"])
            if d < 14 and (best is None or d < best[0]):
                best = (d, k)
        if best is None:
            TRACKS.append({"color": b["color"], "u": b["u"], "v": b["v"], "n": b["n"], "first": tick,
                           "last": tick, "seen": 1, "active": True, "lost": None, "lost_moving": None})
            matched.add(len(TRACKS) - 1)
            ev("appear " + b["color"] + " @" + str(tick) + " px(" + str(b["u"]) + "," + str(b["v"]) + ")")
        else:
            t = TRACKS[best[1]]
            if not t["active"]:
                ev("reappear " + t["color"] + " @" + str(tick))
            t["u"] = b["u"]
            t["v"] = b["v"]
            t["n"] = b["n"]
            t["last"] = tick
            t["seen"] += 1
            t["active"] = True
            t["lost"] = None
            matched.add(best[1])
    for k in range(len(TRACKS)):
        t = TRACKS[k]
        if k not in matched and t["active"]:
            t["active"] = False
            t["lost"] = tick
            t["lost_moving"] = moving
            ev("lost " + t["color"] + " @" + str(tick) + (" (arm moving)" if moving else ""))
    sig = []
    for t in TRACKS:
        if t["active"]:
            sig.append(t["color"] + str(int(t["u"] // 8)) + "_" + str(int(t["v"] // 8)))
    sig = ",".join(sorted(sig))
    if sig != STATE["sig"]:
        STATE["sig"] = sig
        STATE["last_change"] = tick
    report["tracks"] = TRACKS
    return bl


def hidden_location(c):
    best = None
    for t in TRACKS:
        if t["color"] == c and (not t["active"]) and t["seen"] >= 3:
            if best is None or t["lost"] > best["lost"]:
                best = t
    return best


def visible_track(c):
    best = None
    for t in TRACKS:
        if t["color"] == c and t["active"]:
            if best is None or t["n"] > best["n"]:
                best = t
    return best


# ---------------- geometry helpers ----------------
def cloud_points(pc):
    try:
        pts = to_list(pc.points)
    except Exception:
        pts = [list(p) for p in pc.points]
    return pts


def cloud_stats(pts):
    n = len(pts)
    if n < 8:
        return {"n": n}
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    zs = [float(p[2]) for p in pts]
    base = quantile(zs, 0.03)
    top = quantile(zs, 0.97)
    return {"n": n, "base_z": round(base, 4), "top_z": round(top, 4), "height": round(top - base, 4),
            "x05": round(quantile(xs, 0.05), 4), "x95": round(quantile(xs, 0.95), 4),
            "y05": round(quantile(ys, 0.05), 4), "y95": round(quantile(ys, 0.95), 4),
            "med": [round(quantile(xs, 0.5), 4), round(quantile(ys, 0.5), 4), round(quantile(zs, 0.5), 4)]}


def top_rectangle(pts, top, band):
    """Min-area rectangle (5-95% spans) of the points within `band` m of the top.
    Returns centre xy, narrow-axis angle, narrow and wide spans."""
    tx = []
    ty = []
    for p in pts:
        if float(p[2]) >= top - band:
            tx.append(float(p[0]))
            ty.append(float(p[1]))
    if len(tx) < 8:
        return None
    best = None
    for k in range(30):
        th = k * (math.pi / 2.0) / 30.0
        c = math.cos(th)
        s = math.sin(th)
        a = [tx[i] * c + ty[i] * s for i in range(len(tx))]
        b = [-tx[i] * s + ty[i] * c for i in range(len(tx))]
        a0 = quantile(a, 0.05)
        a1 = quantile(a, 0.95)
        b0 = quantile(b, 0.05)
        b1 = quantile(b, 0.95)
        area = (a1 - a0) * (b1 - b0)
        if best is None or area < best[0]:
            best = (area, th, a0, a1, b0, b1)
    area, th, a0, a1, b0, b1 = best
    am = (a0 + a1) / 2.0
    bm = (b0 + b1) / 2.0
    c = math.cos(th)
    s = math.sin(th)
    cx = am * c - bm * s
    cy = am * s + bm * c
    if (a1 - a0) <= (b1 - b0):
        narrow_angle = th
        narrow = a1 - a0
        wide = b1 - b0
    else:
        narrow_angle = th + math.pi / 2.0
        narrow = b1 - b0
        wide = a1 - a0
    return {"xy": [round(cx, 4), round(cy, 4)], "narrow_angle": round(narrow_angle, 4),
            "narrow": round(narrow, 4), "wide": round(wide, 4), "n": len(tx)}


def segment_at(u, v):
    """Segment the object under pixel (u=col, v=row) of the live front image -> (points, info)."""
    cand = None
    info = {"ok": False}
    for order_name in ("uv", "vu"):
        pt = [float(u), float(v)] if order_name == "uv" else [float(v), float(u)]
        seg = segment_points("front", [pt])
        if not seg.ok or len(seg.segmentations) == 0:
            continue
        cand = seg.segmentations[0]
        info = {"ok": True, "order": order_name, "score": round(float(cand.score), 4)}
        try:
            box = [float(x) for x in cand.box_xyxy]
            info["box"] = [round(x, 1) for x in box]
            info["box_contains_point"] = box[0] - 2 <= u <= box[2] + 2 and box[1] - 2 <= v <= box[3] + 2
        except Exception:
            info["box"] = None
            info["box_contains_point"] = None
        if info["box_contains_point"] is not False:
            break
    if cand is None:
        return None, info
    pts = cloud_points(mask_to_point_cloud(cand, "front", target_frame=BASE))
    info["stats"] = cloud_stats(pts)
    return pts, info


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def rotmat(q):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def yawed(q, target_axis_angle):
    """Rotate q about world z so the finger-closing axis (hand y) lies along target_axis_angle (mod pi)."""
    R = rotmat(q)
    phi = math.atan2(R[1][1], R[0][1])
    a = target_axis_angle - phi
    while a > math.pi / 2:
        a -= math.pi
    while a <= -math.pi / 2:
        a += math.pi
    qz = [math.cos(a / 2.0), 0.0, 0.0, math.sin(a / 2.0)]
    return qmul(qz, q), a


def ik_joints(pose, seed):
    ik = solve_ik(pose, seed_joints=seed)
    if not ik.ok:
        return None, str(ik.error)[:200]
    jp = ik.joint_positions
    try:
        jp = jp["primary"]
    except Exception:
        pass
    return [float(v) for v in jp], None


def hold_ticks(n, gripper, watch_every):
    st = get_robot_state()
    q = [float(v) for v in st.joint_positions["primary"]]
    for k in range(n):
        if not alive["ok"]:
            return
        r = step(RobotAction(arms={"primary": ArmCommand(
            "joint_position", q, gripper_position=gripper, embodiment=st.embodiment)}))
        if is_terminal(r):
            alive["ok"] = False
            ev("terminal during hold at tick " + str(now_tick()))
            return
        if watch_every and (k + 1) % watch_every == 0:
            observe(False)


def servo(target, gripper, max_delta, settle, watch):
    st = get_robot_state()
    cur = [float(v) for v in st.joint_positions["primary"]]
    span = max(abs(target[k] - cur[k]) for k in range(7))
    n = max(1, int(math.ceil(span / max_delta)))
    for i in range(1, n + settle + 1):
        if not alive["ok"]:
            return
        f = min(1.0, i / float(n))
        cmd = [cur[k] + (target[k] - cur[k]) * f for k in range(7)]
        r = step(RobotAction(arms={"primary": ArmCommand(
            "joint_position", cmd, gripper_position=gripper, embodiment=st.embodiment)}))
        if is_terminal(r):
            alive["ok"] = False
            ev("terminal during servo at tick " + str(now_tick()))
            return
        if watch:
            observe(True)


def ee_pos():
    p = get_robot_state().end_effector_poses["primary"].position
    return [round(float(v), 4) for v in p]


def gripper_now():
    return round(float(get_robot_state().gripper_positions["primary"]), 4)


def checked(r, label):
    """Record a motion result; returns False when the episode ended."""
    try:
        report.setdefault("motions", []).append(
            {"label": label, "ok": bool(r.ok), "status": str(r.status)[:60],
             "steps": int(r.steps_executed), "tick": now_tick()})
    except Exception:
        pass
    if is_terminal(r):
        alive["ok"] = False
        ev("terminal after " + label + " at tick " + str(now_tick()))
        return False
    return True


# =====================================================================
# Phase A: read everything before moving
# =====================================================================
ctx = get_task_context()
LANG = str(ctx.language)
report["language"] = LANG
st0 = get_robot_state()
BASE = st0.base_frame
ee0 = st0.end_effector_poses["primary"]
Q0 = [float(v) for v in ee0.quaternion_wxyz]
R0 = rotmat(Q0)
QDOWN = Q0
if R0[2][2] > -0.9:
    QDOWN = [0.0, 1.0, 0.0, 0.0]
    ev("home orientation not downward; using [0,1,0,0]")
try:
    report["task_metadata"] = plain(robomme_plus.get_task_metadata())
except Exception as exc:
    note_err("task_metadata", exc)
try:
    report["video_len"] = len(robomme_plus.get_conditioning_video())
except Exception as exc:
    note_err("video", exc)

lang_low = LANG.lower()
order = []
for word in COLOR_WORDS:
    idx = lang_low.find(word)
    while idx >= 0:
        order.append((idx, word))
        idx = lang_low.find(word, idx + 1)
order.sort()
TARGET_COLORS = []
for item in order:
    TARGET_COLORS.append("pink" if item[1] == "magenta" else item[1])
report["target_colors"] = TARGET_COLORS

obs0 = get_observation()
FRONT_ROWS0 = to_list(obs0.cameras["front"].rgb)
calibrate_background(FRONT_ROWS0)
report["background"] = dict(BG)
observe(False)
result = report

# ---- button localisation (unchanged from v00) ----
cube_pixels = [(t["u"], t["v"]) for t in TRACKS]
button_cands = []
for prompt in ("button", "round button", "push button"):
    try:
        seg = segment_text("front", prompt)
        if not seg.ok:
            continue
        for cand in seg.segmentations[:5]:
            item = {"prompt": prompt, "score": round(float(cand.score), 4)}
            try:
                box = [float(x) for x in cand.box_xyxy]
            except Exception:
                box = None
            item["box"] = box
            covers_cube = False
            if box is not None:
                for pix in cube_pixels:
                    if box[0] <= pix[0] <= box[2] and box[1] <= pix[1] <= box[3]:
                        covers_cube = True
            item["covers_cube"] = covers_cube
            pts = cloud_points(mask_to_point_cloud(cand, "front", target_frame=BASE))
            s = cloud_stats(pts)
            if s.get("n", 0) >= 8:
                rect = top_rectangle(pts, s["top_z"], 0.012)
                s["top_mid"] = rect["xy"] if rect is not None else [s["med"][0], s["med"][1]]
            item["stats"] = s
            ok = s.get("n", 0) >= 15 and s.get("height", 1.0) <= 0.07 and \
                s.get("x95", 0) - s.get("x05", 0) <= 0.16 and s.get("y95", 0) - s.get("y05", 0) <= 0.16
            item["geom_ok"] = ok
            button_cands.append(item)
    except Exception as exc:
        note_err("button prompt " + prompt, exc)
report["button_cands"] = button_cands
BUTTON = None
for pref in (True, False):
    for item in button_cands:
        if item["geom_ok"] and ((not item["covers_cube"]) or not pref):
            if BUTTON is None or item["score"] > BUTTON["score"]:
                BUTTON = item
    if BUTTON is not None:
        break
if BUTTON is None:
    for item in button_cands:
        if item["stats"].get("n", 0) >= 15 and not item["covers_cube"]:
            if BUTTON is None or item["score"] > BUTTON["score"]:
                BUTTON = item
    if BUTTON is not None:
        ev("button chosen without geometry bounds")
report["button"] = BUTTON
report["phase"] = "planned"
result = report

# =====================================================================
# Phase B: press the button (deadline: before control step 90), watching the scene every tick
# =====================================================================
if BUTTON is not None:
    try:
        s = BUTTON["stats"]
        bx = s["top_mid"][0]
        by = s["top_mid"][1]
        top = s["top_z"]
        press_z = max(top - 0.018, s["base_z"] + 0.002)
        pre = Pose([bx, by, top + 0.06], QDOWN, BASE)
        down = Pose([bx, by, press_z], QDOWN, BASE)
        q_pre, e1 = ik_joints(pre, [float(v) for v in st0.joint_positions["primary"]])
        q_down = None
        e2 = None
        if q_pre is not None:
            q_down, e2 = ik_joints(down, q_pre)
        report["press_plan"] = {"xy": [bx, by], "top": top, "press_z": press_z, "ik_pre_err": e1, "ik_down_err": e2}
        if q_pre is not None and q_down is not None:
            servo(q_pre, 0.0, 0.06, 2, True)
            report["press_plan"]["tick_at_pre"] = now_tick()
            servo(q_down, 0.0, 0.025, 5, True)
            report["press_plan"]["tick_at_bottom"] = now_tick()
            report["press_plan"]["ee_at_bottom"] = ee_pos()
            servo(q_pre, 0.0, 0.04, 1, True)
            report["press_plan"]["tick_after_rise"] = now_tick()
            ev("pressed")
    except Exception as exc:
        note_err("press", exc)
else:
    ev("no button found; skipping press")
report["phase"] = "pressed"
result = report

# =====================================================================
# Phase C: park and watch until the scene is stable (targets covered)
# =====================================================================
T_MIN = 130.0
STABLE_TICKS = 40.0
T_MAX = 320.0
if alive["ok"]:
    try:
        r = go_home()
        checked(r, "go_home_after_press")
    except Exception as exc:
        note_err("go_home", exc)
if alive["ok"]:
    try:
        observe(False)
        while alive["ok"]:
            hold_ticks(2, 1.0, 0)
            observe(False)
            tick = now_tick()
            if tick is None or tick >= T_MAX:
                ev("watch timeout at " + str(tick))
                break
            stable_for = tick - STATE["last_change"]
            all_hidden = len(TARGET_COLORS) > 0
            for c in TARGET_COLORS:
                if hidden_location(c) is None or visible_track(c) is not None:
                    all_hidden = False
            if tick >= T_MIN and stable_for >= STABLE_TICKS and all_hidden:
                ev("scene stable, all targets hidden, at " + str(tick))
                break
            if tick >= T_MIN and stable_for >= 2 * STABLE_TICKS:
                ev("scene stable (targets not all hidden) at " + str(tick))
                break
    except Exception as exc:
        note_err("watch", exc)
report["decision_tick"] = now_tick()
result = report

# =====================================================================
# Phase D: pick the container hiding each target colour, in order
# =====================================================================
picks = []


def find_container_for(c):
    t = hidden_location(c)
    if t is not None:
        pts, info = segment_at(t["u"], t["v"])
        info["source"] = "hidden_track"
        info["pixel"] = [t["u"], t["v"]]
        return pts, info
    return None, {"ok": False, "source": "none", "visible": visible_track(c) is not None}


def grasp_container(c, pts, info, pick, final):
    s = info["stats"]
    top = s["top_z"]
    rect = top_rectangle(pts, top, 0.008)
    if rect is None:
        rect = top_rectangle(pts, top, 0.015)
    if rect is None:
        pick["err"] = "no top points"
        return False
    pick["rect"] = rect
    qg, yaw = yawed(QDOWN, rect["narrow_angle"])
    cx = rect["xy"][0]
    cy = rect["xy"][1]
    grasp_z = max(top - 0.018, s["base_z"] + 0.01)
    pick["grasp"] = {"xy": [cx, cy], "top": top, "grasp_z": round(grasp_z, 4), "yaw": round(yaw, 3)}
    if not checked(open_gripper(), "open_" + c):
        return False
    if not checked(move_to_pose(Pose([cx, cy, top + 0.10], qg, BASE), tolerance=0.015, max_steps=150), "pre_" + c):
        return False
    if not checked(move_to_pose(Pose([cx, cy, grasp_z], qg, BASE), tolerance=0.01, max_steps=120), "descend_" + c):
        return False
    pick["ee_at_grasp"] = ee_pos()
    if not checked(close_gripper(), "close_" + c):
        return False
    pick["gripper_closed"] = gripper_now()
    if not checked(move_to_pose(Pose([cx, cy, top + 0.13], qg, BASE), tolerance=0.02, max_steps=150), "lift_" + c):
        return False
    pick["ee_after_lift"] = ee_pos()
    pick["gripper_after_lift"] = gripper_now()
    hold_ticks(10, 0.0, 0)
    if not alive["ok"]:
        return False
    # verify: is the target colour now visible near the remembered pixel?
    bl = observe(False)
    pix = info.get("pixel")
    seen = []
    if bl is not None:
        for b in bl:
            if pix is not None and abs(b["u"] - pix[0]) < 20 and abs(b["v"] - pix[1]) < 20:
                seen.append(b["color"])
    pick["revealed_colors"] = seen
    pick["held"] = pick["gripper_after_lift"] > 0.03
    if final:
        return True
    # put it back down where it was, then clear the gripper
    if not checked(move_to_pose(Pose([cx, cy, grasp_z + 0.008], qg, BASE), tolerance=0.012, max_steps=150), "lower_" + c):
        return False
    if not checked(open_gripper(), "release_" + c):
        return False
    if not checked(move_to_pose(Pose([cx, cy, top + 0.10], qg, BASE), tolerance=0.02, max_steps=120), "retreat_" + c):
        return False
    return True


for ci in range(len(TARGET_COLORS)):
    c = TARGET_COLORS[ci]
    if not alive["ok"]:
        break
    pick = {"color": c, "tick_start": now_tick()}
    picks.append(pick)
    report["picks"] = picks
    try:
        if ci > 0:
            if not checked(go_home(), "home_before_" + c):
                break
        pts, info = find_container_for(c)
        pick["container"] = info
        if pts is None or info["stats"].get("n", 0) < 15:
            pick["err"] = "container not found"
            continue
        ok = grasp_container(c, pts, info, pick, ci == len(TARGET_COLORS) - 1)
        if alive["ok"] and pick.get("gripper_closed") is not None and pick["gripper_closed"] <= 0.03:
            # closed on nothing: one retry from a fresh segmentation, slightly lower
            ev("empty close on " + c + "; retrying")
            checked(open_gripper(), "reopen_" + c)
            pts2, info2 = find_container_for(c)
            if pts2 is not None and info2["stats"].get("n", 0) >= 15:
                info2["stats"]["top_z"] = info2["stats"]["top_z"] - 0.01
                retry = {"color": c, "retry": True}
                picks.append(retry)
                grasp_container(c, pts2, info2, retry, ci == len(TARGET_COLORS) - 1)
    except Exception as exc:
        note_err("pick " + c, exc)
report["picks"] = picks
if alive["ok"]:
    try:
        hold_ticks(10, 0.0, 0)
    except Exception as exc:
        note_err("final hold", exc)
report["final_tick"] = now_tick() if alive["ok"] else None
report["alive_at_end"] = alive["ok"]
report["phase"] = "done"
result = report
