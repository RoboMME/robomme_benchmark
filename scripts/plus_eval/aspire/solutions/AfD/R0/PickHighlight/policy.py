# PickHighlight (robomme_plus task 6) -- v02: wrist-primary highlight set; front-only additions must be
# strong rings outside the region the wrist saw cubes in. Fragment filter uses the 75th-percentile size.
# (v01 header follows)
# PickHighlight (robomme_plus task 6) -- v01: perception/selection robustness + one grasp retry.
# Changes vs v00: drop fragment masks (< half the median cube size) before ring scoring; front-only
# additions limited to the count surplus over the wrist; one re-grasp when the jaw closes empty.
# Goal (parsed at runtime): "first press the button, then pick up all highlighted cubes one by one,
# finally press the button to stop".
# Strategy (derived from released recordings, see task_analysis.md):
#   1. Hold the arm still at home; record a baseline front+wrist RGB at tick 0.
#   2. Step-hold tick by tick and watch for NEW white pixels (highlight discs appear under a subset
#      of cubes shortly after the start and vanish again after a variable time).
#   3. On first detection, snapshot both cameras, segment cubes ("cube"...), and mark a cube as
#      highlighted when the new-white pixels nearest to it surround it (Voronoi + angular coverage).
#   4. Remember highlighted cube 3D centres; press the button; pick up each remembered cube
#      (top-down grasp, lift, put back, release); press the button again.
# Everything episode-dependent (which cubes, how many, where, button location) is derived at runtime.
import math

report = {"version": "v02", "events": []}
ctx = get_task_context()
goal = str(ctx.language)
report["goal"] = goal
state0 = get_robot_state()
BASE = state0.base_frame
EMB = state0.embodiment
meta = {}
try:
    meta = robomme_plus.get_task_metadata()
except Exception as exc:
    report["events"].append("metadata error " + str(exc))
HZ = 20.0
MAX_STEPS = None
try:
    HZ = float(meta["control_hz"])
except Exception:
    try:
        HZ = float(meta["control_frequency"])
    except Exception:
        HZ = 20.0
try:
    MAX_STEPS = int(meta["max_steps"])
except Exception:
    MAX_STEPS = None
report["max_steps"] = MAX_STEPS
low_goal = goal.lower()
WANT_PRESS_FIRST = ("press" in low_goal) and ("button" in low_goal)
WANT_PRESS_LAST = WANT_PRESS_FIRST and (("finally" in low_goal) or ("stop" in low_goal) or low_goal.count("press") >= 2)
WANT_HIGHLIGHT = "highlight" in low_goal

# ---------------------------------------------------------------- helpers
def elapsed_ticks():
    try:
        return int(round(float(get_robot_state().timestamp_s) * HZ))
    except Exception:
        return 0

def is_terminal(res):
    flags = False
    try:
        if res.terminated:
            flags = True
    except Exception:
        pass
    try:
        if res.truncated:
            flags = True
    except Exception:
        pass
    return flags

TERMINATED = [False]

def note(msg):
    report["events"].append(str(elapsed_ticks()) + ": " + str(msg))

def hold_tick():
    st = get_robot_state()
    reply = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=st.joint_positions["primary"],
        gripper_position=st.gripper_positions["primary"], embodiment=st.embodiment)}))
    if is_terminal(reply):
        TERMINATED[0] = True
    return reply

def rgb_rows(obs, cam):
    arr = obs.cameras[cam].rgb
    try:
        return arr.tolist()
    except Exception:
        rows = []
        for y in range(len(arr)):
            row = []
            for x in range(len(arr[y])):
                px = arr[y][x]
                row.append([int(px[0]), int(px[1]), int(px[2])])
            rows.append(row)
        return rows

WHITE_THR = 200
WHITE_RISE = 40

def new_white_pixels(now, base, stride):
    pts = []
    H = len(now)
    W = len(now[0])
    for y in range(0, H, stride):
        rn = now[y]
        rb = base[y]
        for x in range(0, W, stride):
            a = rn[x]
            mn = min(a[0], a[1], a[2])
            if mn > WHITE_THR:
                b = rb[x]
                mb = min(b[0], b[1], b[2])
                if mb <= WHITE_THR and mn - mb >= WHITE_RISE:
                    pts.append((x, y))
    return pts

def cand_box(c):
    try:
        b = c.box_xyxy
        if b is None:
            return None
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    except Exception:
        return None

def cand_score(c):
    try:
        return float(c.score)
    except Exception:
        return 0.0

def segment_all(cam, prompts, max_half):
    out = []
    for p in prompts:
        try:
            seg = segment_text(cam, p)
        except Exception as exc:
            note("segment_text error " + cam + " " + p + " " + str(exc))
            continue
        if not seg.ok:
            continue
        for c in seg.segmentations:
            b = cand_box(c)
            if b is None:
                continue
            w = b[2] - b[0]
            h = b[3] - b[1]
            if w < 2 or h < 2 or w > 2 * max_half or h > 2 * max_half:
                continue
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            half = max(w, h) / 2.0
            dup = False
            for o in out:
                if abs(o["cx"] - cx) < 0.6 * max(half, o["half"]) and abs(o["cy"] - cy) < 0.6 * max(half, o["half"]):
                    dup = True
                    break
            if not dup:
                out.append({"cand": c, "box": b, "cx": cx, "cy": cy, "half": max(2.0, half), "score": cand_score(c), "prompt": p})
    # fragment filter: a cube mask far smaller than the typical cube is a shard of a face or shadow
    if len(out) >= 3:
        q75 = quantile([o["half"] for o in out], 0.75)
        kept = [o for o in out if o["half"] >= 0.45 * q75]
        out = kept
    return out

def ring_scores(white_pts, cubes, outer):
    n = len(cubes)
    counts = [0] * n
    bins = []
    for i in range(n):
        bins.append([0] * 8)
    unassigned = []
    for pt in white_pts:
        x = pt[0]
        y = pt[1]
        best = -1
        bd = None
        for i in range(n):
            c = cubes[i]
            dd = math.sqrt((x - c["cx"]) ** 2 + (y - c["cy"]) ** 2) / c["half"]
            if bd is None or dd < bd:
                bd = dd
                best = i
        if best < 0 or bd > outer:
            unassigned.append(pt)
            continue
        counts[best] += 1
        ang = math.atan2(y - cubes[best]["cy"], x - cubes[best]["cx"])
        k = int(((ang + math.pi) / (2.0 * math.pi)) * 8) % 8
        bins[best][k] += 1
    for i in range(n):
        cubes[i]["ring_norm"] = counts[i] / (cubes[i]["half"] ** 2)
        cubes[i]["ring_cover"] = sum(1 for v in bins[i] if v >= 2)
        cubes[i]["ring_count"] = counts[i]
    return unassigned

def classify(cubes, min_cover):
    mx = 0.0
    for c in cubes:
        if c["ring_norm"] > mx:
            mx = c["ring_norm"]
    thr = max(0.6, 0.3 * mx)
    hl = []
    for c in cubes:
        if c["ring_norm"] >= thr and c["ring_cover"] >= min_cover:
            hl.append(c)
    return hl

def quantile(vals, f):
    s = sorted(vals)
    if len(s) == 0:
        return None
    return s[int((len(s) - 1) * f)]

def cube_from_cloud(cand, cam):
    try:
        pc = mask_to_point_cloud(cand, cam, target_frame=BASE)
        pts = [(float(p[0]), float(p[1]), float(p[2])) for p in pc.points]
    except Exception as exc:
        note("cloud error " + str(exc))
        return None
    if len(pts) < 8:
        return None
    zs = [p[2] for p in pts]
    top = quantile(zs, 0.9)
    lowz = quantile(zs, 0.05)
    topf = [p for p in pts if p[2] >= top - 0.008]
    if len(topf) < 5:
        topf = pts
    cx = quantile([p[0] for p in topf], 0.5)
    cy = quantile([p[1] for p in topf], 0.5)
    best_a = 0.0
    best_e = None
    for k in range(18):
        a = math.radians(5.0 * k)
        ca = math.cos(a)
        sa = math.sin(a)
        us = [(p[0] - cx) * ca + (p[1] - cy) * sa for p in topf]
        vs = [-(p[0] - cx) * sa + (p[1] - cy) * ca for p in topf]
        eu = quantile(us, 0.95) - quantile(us, 0.05)
        ev = quantile(vs, 0.95) - quantile(vs, 0.05)
        e = max(eu, ev)
        if best_e is None or e < best_e:
            best_e = e
            best_a = a
    side = best_e / 0.9 if best_e is not None else 0.04
    side = min(0.07, max(0.02, side))
    return {"x": cx, "y": cy, "top": top, "low": lowz, "yaw": best_a, "side": side, "n": len(pts)}

# --- quaternion helpers (wxyz)
def qmul(a, b):
    return (a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])

def qrot(q, v):
    p = (0.0, v[0], v[1], v[2])
    qc = (q[0], -q[1], -q[2], -q[3])
    r = qmul(qmul(q, p), qc)
    return (r[1], r[2], r[3])

def qnorm(q):
    n = math.sqrt(sum(c * c for c in q))
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)

def qyaw(psi):
    return (math.cos(psi / 2.0), 0.0, 0.0, math.sin(psi / 2.0))

def straight_down(q):
    # minimal rotation taking the tool z axis onto world -z
    z = qrot(q, (0.0, 0.0, 1.0))
    t = (0.0, 0.0, -1.0)
    ax = (z[1] * t[2] - z[2] * t[1], z[2] * t[0] - z[0] * t[2], z[0] * t[1] - z[1] * t[0])
    s = math.sqrt(ax[0] ** 2 + ax[1] ** 2 + ax[2] ** 2)
    d = z[0] * t[0] + z[1] * t[1] + z[2] * t[2]
    if s < 1e-6:
        return q
    ang = math.atan2(s, d)
    h = ang / 2.0
    r = (math.cos(h), ax[0] / s * math.sin(h), ax[1] / s * math.sin(h), ax[2] / s * math.sin(h))
    return qnorm(qmul(r, q))

def closing_yaw(q):
    yv = qrot(q, (0.0, 1.0, 0.0))
    return math.atan2(yv[1], yv[0])

def wrap(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

HOME_Q = tuple(float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz)
DOWN_Q = straight_down(HOME_Q)
DOWN_YAW = closing_yaw(DOWN_Q)

def grasp_quat(cube_yaw, others, cx, cy, side):
    # two closing-axis options (cube faces); pick the one with more clearance at the finger tips,
    # each realised with the smallest wrist rotation from the straight-down home orientation.
    best = None
    for k in range(2):
        axis = cube_yaw + k * math.pi / 2.0
        d = wrap(axis - DOWN_YAW)
        while d > math.pi / 2.0:
            d -= math.pi
        while d < -math.pi / 2.0:
            d += math.pi
        reach = 0.5 * side + 0.025
        fx = math.cos(axis) * reach
        fy = math.sin(axis) * reach
        clear = 1.0
        for o in others:
            for sgn in (1.0, -1.0):
                dd = math.sqrt((cx + sgn * fx - o[0]) ** 2 + (cy + sgn * fy - o[1]) ** 2)
                if dd < clear:
                    clear = dd
        cost = -clear + 0.01 * abs(d)
        if best is None or cost < best[0]:
            best = (cost, d)
    return qnorm(qmul(qyaw(best[1]), DOWN_Q))

def move(pos, quat, tol, steps):
    if TERMINATED[0]:
        return None
    try:
        res = move_to_pose(Pose(position=[pos[0], pos[1], pos[2]], quaternion_wxyz=list(quat), frame=BASE),
                           tolerance=tol, max_steps=steps)
    except Exception as exc:
        note("move error " + str(exc))
        return None
    if is_terminal(res):
        TERMINATED[0] = True
    try:
        ok = bool(res.ok)
    except Exception:
        ok = False
    if not ok:
        try:
            note("move not ok " + str(res.status) + " to " + str([round(v, 3) for v in pos]))
        except Exception:
            note("move not ok")
    return res

def grip(close):
    if TERMINATED[0]:
        return None
    try:
        res = close_gripper() if close else open_gripper()
    except Exception as exc:
        note("gripper error " + str(exc))
        return None
    if is_terminal(res):
        TERMINATED[0] = True
    return res

def ticks_left():
    if MAX_STEPS is None:
        return 100000
    return MAX_STEPS - elapsed_ticks()

# ---------------------------------------------------------------- 1-2. watch for the highlight
obs_base = get_observation()
cams = list(obs_base.cameras)
report["cameras"] = cams
FRONT = "front" if "front" in cams else cams[0]
WRIST = "wrist" if "wrist" in cams else None
base_front = rgb_rows(obs_base, FRONT)
base_wrist = rgb_rows(obs_base, WRIST) if WRIST is not None else None

WAIT_LIMIT_S = 4.0
hl_obs = None
detected_tick = None
quick_counts = []
while not TERMINATED[0]:
    hold_tick()
    o = get_observation()
    now_f = rgb_rows(o, FRONT)
    nq = len(new_white_pixels(now_f, base_front, 4))
    quick_counts.append(nq)
    if nq >= 6:
        hold_tick()   # one more tick so the highlight is fully rendered
        hl_obs = get_observation()
        detected_tick = elapsed_ticks()
        break
    if float(o.timestamp_s) > WAIT_LIMIT_S:
        break
report["quick_counts"] = quick_counts[-12:]
report["highlight_detected_tick"] = detected_tick

# ---------------------------------------------------------------- 3. classify cubes
CUBE_PROMPTS = ["cube", "block", "small cube"]
targets = []          # list of dicts with x, y, top, yaw, side
all_cubes_xy = []
if hl_obs is not None:
    views = []
    if WRIST is not None:
        views.append((WRIST, rgb_rows(hl_obs, WRIST), base_wrist, 5))
    views.append((FRONT, rgb_rows(hl_obs, FRONT), base_front, 4))
    per_cam = {}
    for cam, now_rows, base_rows, min_cover in views:
        wp = new_white_pixels(now_rows, base_rows, 1)
        cubes = segment_all(cam, CUBE_PROMPTS, len(now_rows) / 6.0)
        unassigned = ring_scores(wp, cubes, 2.6)
        hl = classify(cubes, min_cover)
        found = []
        for c in hl:
            g = cube_from_cloud(c["cand"], cam)
            if g is not None:
                g["cam"] = cam
                g["ring_norm"] = round(c["ring_norm"], 2)
                g["ring_cover"] = c["ring_cover"]
                found.append(g)
        if cam == WRIST or WRIST is None:
            for c in cubes:
                if c["ring_norm"] < 0.3 and len(all_cubes_xy) < 25:
                    g = cube_from_cloud(c["cand"], cam)
                    if g is not None:
                        all_cubes_xy.append((g["x"], g["y"]))
        # de-duplicate within the camera (two masks of one cube), keeping the stronger ring
        ranked_f = sorted([(-g["ring_norm"], i) for i, g in enumerate(found)])
        uniq = []
        for item in ranked_f:
            g = found[item[1]]
            same = False
            for u in uniq:
                if math.sqrt((u["x"] - g["x"]) ** 2 + (u["y"] - g["y"]) ** 2) < 0.03:
                    same = True
                    break
            if not same:
                uniq.append(g)
        found = uniq
        per_cam[cam] = found
        hold_tick()   # keep the control clock advancing between long perception blocks
        report["cam_" + cam] = {"white_px": len(wp), "cubes": len(cubes), "highlighted": len(hl),
                                "unassigned_white": len(unassigned),
                                "norms": sorted([round(c["ring_norm"], 2) for c in cubes], reverse=True)[:12]}
    # merge: wrist first (closer, top-down); front-only items only when the front camera sees
    # more highlighted cubes than the wrist (a highlighted cube outside the wrist field of view)
    n_wrist = len(per_cam[WRIST]) if (WRIST is not None and WRIST in per_cam) else 0
    for cam in (WRIST, FRONT):
        if cam is None or cam not in per_cam:
            continue
        n_wrist = len(targets)   # wrist targets after 3D de-duplication
        if cam == FRONT and WRIST is not None and len(per_cam[FRONT]) <= n_wrist:
            continue
        # region in which the wrist camera saw cubes (any cube, highlighted or not), +2 cm
        seen = [(t["x"], t["y"]) for t in targets] + all_cubes_xy
        if len(seen) > 0:
            sx0 = min(v[0] for v in seen) - 0.02
            sx1 = max(v[0] for v in seen) + 0.02
            sy0 = min(v[1] for v in seen) - 0.02
            sy1 = max(v[1] for v in seen) + 0.02
        front_max = 0.0
        for g in per_cam[cam]:
            if g["ring_norm"] > front_max:
                front_max = g["ring_norm"]
        budget_extra = len(per_cam[FRONT]) - n_wrist if (cam == FRONT and WRIST is not None) else 1000
        ranked = sorted([(-g["ring_norm"], i) for i, g in enumerate(per_cam[cam])])
        for item in ranked:
            g = per_cam[cam][item[1]]
            if budget_extra <= 0:
                break
            if cam == FRONT and WRIST is not None:
                strong = g["ring_norm"] >= 0.5 * front_max and g["ring_cover"] >= 6
                inside = len(seen) > 0 and sx0 <= g["x"] <= sx1 and sy0 <= g["y"] <= sy1
                if (not strong) or inside:
                    continue
            dup = False
            for t in targets:
                if math.sqrt((t["x"] - g["x"]) ** 2 + (t["y"] - g["y"]) ** 2) < 0.03:
                    dup = True
                    break
            if not dup:
                targets.append(g)
                if cam == FRONT and WRIST is not None:
                    budget_extra -= 1
report["targets"] = [{"x": round(t["x"], 3), "y": round(t["y"], 3), "top": round(t["top"], 3),
                      "side": round(t["side"], 3), "cam": t["cam"]} for t in targets]

# ---------------------------------------------------------------- button
def locate_button():
    best = None
    for cam in (FRONT, WRIST):
        if cam is None:
            continue
        for p in ("button", "round button", "gray button"):
            try:
                seg = segment_text(cam, p)
            except Exception:
                continue
            if not seg.ok:
                continue
            for c in seg.segmentations[:3]:
                g = cube_from_cloud(c, cam)
                if g is None:
                    continue
                s = cand_score(c)
                if best is None or s > best[0]:
                    best = (s, g, cam, p)
            if best is not None:
                break
        if best is not None:
            break
    return best

btn = locate_button()
if btn is not None:
    report["button"] = {"x": round(btn[1]["x"], 3), "y": round(btn[1]["y"], 3), "top": round(btn[1]["top"], 3),
                        "cam": btn[2], "prompt": btn[3], "score": round(btn[0], 3)}
else:
    note("button not found")

PRESS_DEPTH = 0.015
HOVER = 0.10

def press_button():
    if btn is None or TERMINATED[0]:
        return False
    g = btn[1]
    grip(True)
    above = (g["x"], g["y"], g["top"] + HOVER)
    move(above, DOWN_Q, 0.015, 150)
    move((g["x"], g["y"], g["top"] - PRESS_DEPTH), DOWN_Q, 0.01, 40)
    move(above, DOWN_Q, 0.02, 100)
    note("button press done")
    return True

# ---------------------------------------------------------------- refine with wrist from the hover pose
def refine(t):
    if WRIST is None:
        return t
    cubes = segment_all(WRIST, CUBE_PROMPTS[:1], 256 / 2.5)
    W2 = 128.0
    order = sorted([(math.sqrt((c["cx"] - W2) ** 2 + (c["cy"] - W2) ** 2), i) for i, c in enumerate(cubes)])
    best = None
    for item in order[:5]:
        c = cubes[item[1]]
        g = cube_from_cloud(c["cand"], WRIST)
        if g is None:
            continue
        d = math.sqrt((g["x"] - t["x"]) ** 2 + (g["y"] - t["y"]) ** 2)
        if d < 0.03 and (best is None or d < best[0]):
            best = (d, g)
    if best is None:
        return t
    g = best[1]
    g["cam"] = "wrist-refined"
    return g

# ---------------------------------------------------------------- 4. execute
LIFT = 0.08
picked = 0
if WANT_PRESS_FIRST:
    press_button()

remaining = list(targets)
cur = get_robot_state().end_effector_poses["primary"].position
while len(remaining) > 0 and not TERMINATED[0]:
    if ticks_left() < 180:
        note("tick budget low; skipping remaining picks")
        break
    bi = 0
    bd = None
    for i in range(len(remaining)):
        d = math.sqrt((remaining[i]["x"] - cur[0]) ** 2 + (remaining[i]["y"] - cur[1]) ** 2)
        if bd is None or d < bd:
            bd = d
            bi = i
    t = remaining.pop(bi)
    others = [(o["x"], o["y"]) for o in remaining] + all_cubes_xy
    grip(False)
    hover = (t["x"], t["y"], t["top"] + HOVER)
    q = grasp_quat(t["yaw"], others, t["x"], t["y"], t["side"])
    move(hover, q, 0.015, 150)
    t2 = refine(t)
    q = grasp_quat(t2["yaw"], others, t2["x"], t2["y"], t2["side"])
    gz = t2["top"] - 0.5 * t2["side"]
    move((t2["x"], t2["y"], t2["top"] + 0.04), q, 0.015, 80)
    move((t2["x"], t2["y"], gz), q, 0.01, 80)
    grip(True)
    move((t2["x"], t2["y"], gz + LIFT), q, 0.015, 80)
    gp = float(get_robot_state().gripper_positions["primary"])
    held = gp > 0.05
    note("pick lifted gripper=" + str(round(gp, 3)) + " held=" + str(held))
    if (not held) and (not TERMINATED[0]) and ticks_left() > 300:
        # jaw closed on nothing: re-observe from the lifted pose and try once more
        grip(False)
        move((t2["x"], t2["y"], t2["top"] + HOVER), q, 0.015, 80)
        t3 = refine(t2)
        if t3["cam"] == "wrist-refined":
            t2 = t3
            q = grasp_quat(t2["yaw"], others, t2["x"], t2["y"], t2["side"])
            gz = t2["top"] - 0.5 * t2["side"]
            move((t2["x"], t2["y"], t2["top"] + 0.04), q, 0.015, 80)
            move((t2["x"], t2["y"], gz), q, 0.01, 80)
            grip(True)
            move((t2["x"], t2["y"], gz + LIFT), q, 0.015, 80)
            gp = float(get_robot_state().gripper_positions["primary"])
            note("retry lifted gripper=" + str(round(gp, 3)))
    move((t2["x"], t2["y"], gz + 0.008), q, 0.015, 80)
    grip(False)
    move((t2["x"], t2["y"], t2["top"] + HOVER), q, 0.02, 80)
    picked += 1
    cur = (t2["x"], t2["y"], t2["top"] + HOVER)

if WANT_PRESS_LAST:
    press_button()

report["picked"] = picked
report["n_targets"] = len(targets)
report["terminated"] = TERMINATED[0]
report["final_tick"] = elapsed_ticks()
result = report
