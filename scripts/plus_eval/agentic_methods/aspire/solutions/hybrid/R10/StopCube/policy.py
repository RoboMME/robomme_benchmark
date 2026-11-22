# StopCube v04 — target class excludes only clearly moving candidates (cube may overlap target at t0).
# StopCube v03 — role-class margins centred between measured target/button/robot values.
# StopCube v02 — press timing aimed at the closest approach; periodic fallback only as a backstop.
# StopCube v01 — roles by segmentation + motion/flatness evidence; robust raised threshold.
# StopCube v00 — count cube arrivals at the target from the front camera (depth-raised pixels),
# hover over the button, and time a press so it registers at the N-th arrival.
import math

CAM = "front"
ENTER = 1.0        # normalised (target half-size) distance counted as "on the target"
EXIT = 1.4         # hysteresis for leaving
HOVER_CLEAR = 0.03  # fingertip clearance above button top while waiting (m)
PRESS_DEPTH = 0.02  # press joint target below the button top (m)
REG_DEPTH = -0.008  # v02: press registered 2 ticks after the trigger on ep0/1/2 while the stroke had
                    # descended 1.2-1.8 cm of the 3 cm clearance after tick 1 (top estimate/compliance)
AIM_EARLY = 0.5     # v02: aim registration at the predicted closest approach (fires 0..1 tick early)

report = {"version": "v04"}
ctx = get_task_context()
goal = str(ctx.language)
report["goal"] = goal
meta = {}
try:
    meta = dict(robomme_plus.get_task_metadata())
except Exception as exc:
    report["meta_error"] = str(exc)[:200]
max_steps = 0
try:
    max_steps = int(meta.get("max_steps", 0) or 0)
except Exception:
    max_steps = 0
report["max_steps"] = max_steps
if max_steps <= 0:
    max_steps = 1000

ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
       "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
       "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16,
       "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20,
       "once": 1, "twice": 2, "thrice": 3}
WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def parse_count(text):
    low = text.lower()
    clean = ""
    for ch in low:
        clean += ch if (ch.isalnum()) else " "
    words = clean.split()
    for i, w in enumerate(words):
        if w in ORD:
            return ORD[w]
        digits = ""
        for ch in w:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits and w[len(digits):] in ("st", "nd", "rd", "th"):
            return int(digits)
        if w == "times" and i > 0:
            prev = words[i - 1]
            if prev.isdigit():
                return int(prev)
            if prev in WORDNUM:
                return WORDNUM[prev]
    return None


N = parse_count(goal)
report["n_target"] = N
if N is None:
    N = 1
    report["n_parse_failed"] = True

state0 = get_robot_state()
EMB = state0.embodiment
base = state0.base_frame
home = [float(v) for v in state0.joint_positions["primary"]]
down_q = [float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz]
report["home_ee"] = [round(float(v), 3) for v in state0.end_effector_poses["primary"].position]


def q(values, frac):
    s = sorted(values)
    if not s:
        return None
    return s[int((len(s) - 1) * frac)]


# ---------------- perception: roles from segmentation + runtime physical evidence ----------------
# Candidates for the goal nouns "button" and "target" are pooled; each is classified by runtime
# evidence (moves between frames -> the cube; static & flat -> target; static & raised -> button)
# and the noun's segmentation score then picks among physically consistent candidates.
obs0 = get_observation()
depth0 = obs0.cameras[CAM].depth_m
H = len(depth0)
W = len(depth0[0])
grip0 = 1.0
try:
    grip0 = float(state0.gripper_positions["primary"])
except Exception:
    grip0 = 1.0
PRE_TICKS = 3
pre_ok = True
for k in range(PRE_TICKS):
    sr0 = step(RobotAction({"primary": ArmCommand("joint_position", home, gripper_position=grip0,
                                                  embodiment=EMB)}))
    try:
        pre_ok = pre_ok and bool(sr0.ok)
    except Exception:
        pass
report["pre_ticks_ok"] = pre_ok
obs1 = get_observation()
depth1 = obs1.cameras[CAM].depth_m


def box_stats(box):
    bx0 = int(max(0, math.floor(box[0])))
    bx1 = int(min(W - 1, math.ceil(box[2])))
    by0 = int(max(0, math.floor(box[1])))
    by1 = int(min(H - 1, math.ceil(box[3])))
    tot = 0
    moved = 0
    for v in range(by0, by1 + 1):
        r0 = depth0[v]
        r1 = depth1[v]
        for u in range(bx0, bx1 + 1):
            tot += 1
            if abs(float(r0[u]) - float(r1[u])) > 0.005:
                moved += 1
    return float(moved) / max(1, tot)


def iou(a1, b1):
    ix = max(0.0, min(a1[2], b1[2]) - max(a1[0], b1[0]))
    iy = max(0.0, min(a1[3], b1[3]) - max(a1[1], b1[1]))
    inter = ix * iy
    ua = (a1[2] - a1[0]) * (a1[3] - a1[1]) + (b1[2] - b1[0]) * (b1[3] - b1[1]) - inter
    return inter / max(1e-6, ua)


objs = []
for noun in ("button", "target"):
    seg = segment_text(CAM, noun)
    if not seg.ok:
        continue
    for cand in seg.segmentations[:6]:
        if cand.box_xyxy is None:
            continue
        box = [float(v) for v in cand.box_xyxy]
        match = None
        for o in objs:
            if iou(o["box"], box) > 0.6:
                match = o
                break
        if match is None:
            try:
                pc = mask_to_point_cloud(cand, CAM, target_frame=base)
                pts = [[float(c) for c in pt] for pt in pc.points]
            except Exception:
                pts = []
            if len(pts) < 10:
                continue
            zs = [pt[2] for pt in pts]
            match = {"box": box, "pts": pts, "zr": q(zs, 0.95) - q(zs, 0.05), "ztop": q(zs, 0.95),
                     "moved": box_stats(box), "score": {"button": 0.0, "target": 0.0}}
            objs.append(match)
        match["score"][noun] = max(match["score"][noun], float(cand.score))
report["objects"] = [{"box": [round(v, 1) for v in o["box"]], "zr": round(o["zr"], 4),
                      "ztop": round(o["ztop"], 4), "moved": round(o["moved"], 3),
                      "s_button": round(o["score"]["button"], 3), "s_target": round(o["score"]["target"], 3),
                      "n": len(o["pts"])} for o in objs]
FLAT_ZR = 0.011      # v03: target z-range measured 0.0008/0.0053/0.0073, button >= 0.0157 (ep0/1/2)
MOVED_FRAC = 0.05    # fraction of box pixels whose depth changed over PRE_TICKS ticks (button: static)
CUBE_MOVED = 0.3     # v04: at or above this a flat candidate is the moving cube, not the target
target = None
target_key = None
button = None
for o in objs:
    if o["zr"] < FLAT_ZR:
        # flat: the target disc. Exclude only clearly moving candidates (the cube: 0.57-0.66 of its
        # box changed); the real target box changed 0.15 on ep0 because the cube overlapped it at
        # t0, and v03's static-first preference then picked a tiny static inner-ring fragment.
        if o["score"]["target"] > 0 and o["moved"] < CUBE_MOVED:
            if target is None or o["score"]["target"] > target["score"]["target"]:
                target = o
    elif o["moved"] < MOVED_FRAC and o["zr"] < 0.06 and o["ztop"] < 0.08:
        # static & raised & bounded (excludes the moving cube and the tall robot arm): button
        if o["score"]["button"] > 0 and (button is None or o["score"]["button"] > button["score"]["button"]):
            button = o
if target is None:
    # fallback: the cube may cover most of the target box at t0 -> best flat candidate at all
    for o in objs:
        if o["zr"] < FLAT_ZR and o["score"]["target"] > 0:
            if target is None or o["score"]["target"] > target["score"]["target"]:
                target = o
    report["target_fallback_any_motion"] = target is not None
if target is None:
    raise ValueError("no flat 'target' candidate")
if button is None:
    raise ValueError("no static raised 'button' candidate")
bpts = button["pts"]
bz = [pt[2] for pt in bpts]
btop = q(bz, 0.95)
zcut = q(bz, 0.7)
capp = [pt for pt in bpts if pt[2] >= zcut]
bx = sum(pt[0] for pt in capp) / len(capp)
by = sum(pt[1] for pt in capp) / len(capp)
bbox = button["box"]
report["button"] = {"xy": [round(bx, 4), round(by, 4)], "top": round(btop, 4),
                    "zlow": round(q(bz, 0.05), 4), "n": len(bpts), "box": [round(v, 1) for v in bbox]}
tb = target["box"]
tx = 0.5 * (tb[0] + tb[2])
ty = 0.5 * (tb[1] + tb[3])
thw = max(2.0, 0.5 * (tb[2] - tb[0]))
thh = max(2.0, 0.5 * (tb[3] - tb[1]))
report["target_px"] = [round(tx, 1), round(ty, 1), round(thw, 1), round(thh, 1)]

# ---------------- ROI + table plane from the first frame ----------------
cam0 = obs1.cameras[CAM]
x0 = int(max(0, tx - max(60.0, 6.0 * thw)))
x1 = int(min(W, tx + max(60.0, 6.0 * thw)))
y0 = int(max(0, ty - max(45.0, 6.0 * thh)))
y1 = int(min(H, ty + max(45.0, 6.0 * thh)))
if bbox is not None:
    bcy = 0.5 * (bbox[1] + bbox[3])
    if bcy < ty:
        y0 = int(max(y0, bbox[3] + 4))
    else:
        y1 = int(min(y1, bbox[1] - 4))
report["roi"] = [x0, y0, x1, y1]


def in_button(u, v):
    if bbox is None:
        return False
    return bbox[0] - 6 <= u <= bbox[2] + 6 and bbox[1] - 6 <= v <= bbox[3] + 6


def read_rows(arr):
    try:
        return arr[y0:y1, x0:x1].tolist()
    except Exception:
        rows = []
        for v in range(y0, y1):
            row = arr[v]
            rows.append([float(row[u]) for u in range(x0, x1)])
        return rows


def solve3(m, r):
    def det(a):
        return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    d = det(m)
    if abs(d) < 1e-18:
        return None
    out = []
    for col in range(3):
        mm = [[(r[i] if j == col else m[i][j]) for j in range(3)] for i in range(3)]
        out.append(det(mm) / d)
    return out


# Cube pixels = pixels closer to the camera than a running per-pixel MAX-depth background.
# The cube is the only moving object in the ROI and always lies above whatever it covers, so the
# background converges to the static scene once the cube has moved off a pixel. Height- and
# appearance-free (v00's table-plane threshold sat 1 mm above a raised target disc on ep0).
BG_THR = 0.006
rows0 = read_rows(cam0.depth_m)
rowsA = read_rows(depth0)
bgrows = []
for j in range(len(rows0)):
    bgrows.append([max(float(rows0[j][i]), float(rowsA[j][i])) for i in range(len(rows0[j]))])
report["bg_thr"] = BG_THR


def detect(obs, prev):
    rows = read_rows(obs.cameras[CAM].depth_m)
    us = []
    vs = []
    for j in range(len(rows)):
        rowd = rows[j]
        rowb = bgrows[j]
        v = y0 + j
        for i in range(len(rowd)):
            dv = float(rowd[i])
            if rowb[i] - dv > BG_THR:
                if not in_button(x0 + i, v):
                    us.append(x0 + i)
                    vs.append(v)
            elif dv > rowb[i]:
                rowb[i] = dv
    n = len(us)
    if n < 6:
        return None
    cx = sum(us) / n
    cy = sum(vs) / n
    if prev is not None:
        gx, gy = prev
    else:
        gx, gy = cx, cy
    rad = max(14.0, 2.5 * math.sqrt(n))
    ku = []
    kv = []
    for k in range(n):
        if abs(us[k] - gx) <= rad and abs(vs[k] - gy) <= rad:
            ku.append(us[k])
            kv.append(vs[k])
    if len(ku) < 6:
        ku = us
        kv = vs
    return (sum(ku) / len(ku), sum(kv) / len(kv), len(ku), n)


# ---------------- motion helpers ----------------
tick = [PRE_TICKS]
ended = [False]
end_info = {}
track = []        # compact per-tick log
cube_state = {"inside": None, "count": 0, "arrivals": [], "prev_px": None, "hist": [], "min_ticks": [], "cur_min": None}


def observe_and_track():
    try:
        obs = get_observation()
    except Exception as exc:
        end_info["observe_error"] = str(exc)[:120]
        return None
    det = detect(obs, cube_state["prev_px"])
    if det is None:
        dist = 99.0
        p = None
    else:
        cube_state["prev_px"] = (det[0], det[1])
        p = ((det[0] - tx) / thw, (det[1] - ty) / thh)
        dist = math.sqrt(p[0] * p[0] + p[1] * p[1])
    if cube_state["inside"] is None:
        cube_state["inside"] = dist < ENTER
    elif not cube_state["inside"] and dist < ENTER:
        cube_state["inside"] = True
        cube_state["count"] += 1
        cube_state["arrivals"].append(tick[0])
        cube_state["cur_min"] = (dist, tick[0])
    elif cube_state["inside"] and dist > EXIT:
        cube_state["inside"] = False
        if cube_state["cur_min"] is not None:
            cube_state["min_ticks"].append(cube_state["cur_min"][1])
            cube_state["cur_min"] = None
    if cube_state["inside"] and cube_state["cur_min"] is not None and dist < cube_state["cur_min"][0]:
        cube_state["cur_min"] = (dist, tick[0])
    cube_state["hist"].append(p)
    if len(cube_state["hist"]) > 6:
        cube_state["hist"] = cube_state["hist"][-6:]
    track.append([tick[0], round(dist, 2), 0 if det is None else det[2]])
    return obs


def do_step(joints, grip=0.0):
    if ended[0]:
        return False
    sr = step(RobotAction({"primary": ArmCommand("joint_position", [float(v) for v in joints],
                                                 gripper_position=grip, embodiment=EMB)}))
    tick[0] += 1
    ok = True
    try:
        ok = bool(sr.ok)
    except Exception:
        ok = True
    term = False
    trunc = False
    try:
        term = bool(sr.terminated)
    except Exception:
        term = False
    try:
        trunc = bool(sr.truncated)
    except Exception:
        trunc = False
    if not ok:
        end_info["step_not_ok_tick"] = tick[0]
        try:
            end_info["step_error"] = str(sr.error)[:200]
        except Exception:
            pass
    if term or trunc:
        ended[0] = True
        end_info["terminated"] = term
        end_info["truncated"] = trunc
        end_info["end_tick"] = tick[0]
        return False
    observe_and_track()
    if tick[0] >= max_steps - 1:
        ended[0] = True
        end_info["horizon_tick"] = tick[0]
        return False
    return ok


def ee_z():
    return float(get_robot_state().end_effector_poses["primary"].position[2])


def ee_err(pos):
    p = get_robot_state().end_effector_poses["primary"].position
    return math.sqrt(sum((float(p[k]) - pos[k]) ** 2 for k in range(3)))


def ik(pos, seed):
    res = solve_ik(Pose(list(pos), down_q, base), seed_joints=list(seed))
    if not res.ok:
        return None
    return [float(v) for v in res.joint_positions]


def servo(start, goal_j, max_delta=0.05):
    span = max(abs(goal_j[k] - start[k]) for k in range(7))
    n = max(1, int(math.ceil(span / max_delta)))
    for s in range(1, n + 1):
        f = float(s) / n
        if not do_step([start[k] + f * (goal_j[k] - start[k]) for k in range(7)]):
            return False
    return True


def hold_until(goal_j, pos, tol, max_ticks):
    for s in range(max_ticks):
        if ee_err(pos) < tol:
            return True
        if not do_step(goal_j):
            return False
    return ee_err(pos) < tol


# ---------------- plan poses ----------------
hover_pos = (bx, by, btop + HOVER_CLEAR)
press_pos = (bx, by, btop - PRESS_DEPTH)
stroke = HOVER_CLEAR + PRESS_DEPTH
pre_pos = (bx, by, btop + HOVER_CLEAR + stroke)
j_pre = ik(pre_pos, home)
if j_pre is None:
    raise ValueError("IK pre-hover failed")
j_hover = ik(hover_pos, j_pre)
j_press = ik(press_pos, j_hover)
if j_hover is None or j_press is None:
    raise ValueError("IK hover/press failed")

observe_and_track()   # tick-0 sample
report["phase_ticks"] = {}
servo(home, j_pre)
hold_until(j_pre, pre_pos, 0.006, 25)
report["phase_ticks"]["at_pre"] = tick[0]

# ---- calibration stroke: same joint-target jump (pre -> hover) as the press (hover -> press)
L_reg = 6
calib = []
if not ended[0]:
    z_start = ee_z()
    need = HOVER_CLEAR + REG_DEPTH
    reached = None
    for s in range(1, 25):
        if not do_step(j_hover):
            break
        z = ee_z()
        calib.append(round(z_start - z, 4))
        if reached is None and z_start - z >= need:
            reached = s
        if reached is not None and s >= reached + 2 and ee_err(hover_pos) < 0.004:
            break
    if reached is not None:
        L_reg = reached
    hold_until(j_hover, hover_pos, 0.004, 20)
report["calib_descent"] = calib
report["L_reg"] = L_reg
report["phase_ticks"]["at_hover"] = tick[0]
report["count_at_hover"] = cube_state["count"]

# ---------------- wait for the N-th arrival ----------------
def quad_predict(hist):
    """Least-squares quadratic fit of the last <=5 normalised positions; return (tau, dmin)."""
    pts = [p for p in hist[-5:]]
    if len(pts) < 4 or any(p is None for p in pts):
        return None
    n = len(pts)
    ts = [float(k - (n - 1)) for k in range(n)]
    m = [[0.0] * 3 for k in range(3)]
    for t in ts:
        f = (1.0, t, t * t)
        for a in range(3):
            for b in range(3):
                m[a][b] += f[a] * f[b]
    cs = []
    for axis in range(2):
        r = [0.0] * 3
        for k in range(n):
            t = ts[k]
            f = (1.0, t, t * t)
            for a in range(3):
                r[a] += f[a] * pts[k][axis]
        c = solve3(m, r)
        if c is None:
            return None
        cs.append(c)
    best = None
    for k in range(0, 61):
        tau = k * 0.25
        x = cs[0][0] + cs[0][1] * tau + cs[0][2] * tau * tau
        y = cs[1][0] + cs[1][1] * tau + cs[1][2] * tau * tau
        dd = math.sqrt(x * x + y * y)
        if best is None or dd < best[1]:
            best = (tau, dd)
    return best


trigger = None
pred_log = []
while not ended[0]:
    c = cube_state["count"]
    hist = cube_state["hist"]
    fire = False
    why = ""
    if c >= N:
        fire = True
        why = "late: already counted (inside=%s)" % str(cube_state["inside"])
    elif c == N - 1 and not cube_state["inside"]:
        pr = quad_predict(hist)
        quad_pass = False
        if pr is not None:
            pred_log.append([tick[0], pr[0], round(pr[1], 2)])
            quad_pass = pr[1] < 1.0
            if quad_pass and pr[0] <= L_reg + AIM_EARLY:
                fire = True
                why = "quad tau=%.2f dmin=%.2f" % (pr[0], pr[1])
        mins = cube_state["min_ticks"]
        # periodic backstop only when the kinematic predictor does not see a pass (v01 ep2: the
        # period estimate fired 3 ticks early while the quadratic fit correctly said 5 ticks)
        if not fire and not quad_pass and len(mins) >= 2 and hist and hist[-1] is not None:
            period = float(mins[-1] - mins[-2])
            if len(mins) >= 3:
                period = 0.5 * (mins[-1] - mins[-3])
            t_c = mins[-1] + period
            dnow = math.sqrt(hist[-1][0] ** 2 + hist[-1][1] ** 2)
            dprev = None
            if len(hist) >= 2 and hist[-2] is not None:
                dprev = math.sqrt(hist[-2][0] ** 2 + hist[-2][1] ** 2)
            if (t_c - tick[0]) <= L_reg and dnow < 2.0 and dprev is not None and dnow < dprev:
                fire = True
                why = "periodic t_c=%.1f dnow=%.2f" % (t_c, dnow)
    if fire:
        trigger = {"tick": tick[0], "why": why, "count": c}
        break
    if not do_step(j_hover):
        break
report["pred_log_tail"] = pred_log[-12:]

press_log = []
if trigger is not None and not ended[0]:
    for s in range(30):
        if not do_step(j_press):
            break
        press_log.append(round(ee_z() - btop, 4))
    if not ended[0]:
        report["press_no_termination"] = True
        servo(j_press, j_hover, 0.05)

report["trigger"] = trigger
report["press_z_rel_top"] = press_log[:30]
report["arrivals"] = cube_state["arrivals"]
report["min_ticks"] = cube_state["min_ticks"]
report["count"] = cube_state["count"]
report["end"] = end_info
report["ticks"] = tick[0]
report["track_tail"] = track[-40:]
report["track"] = [t[1] for t in track]
result = report
