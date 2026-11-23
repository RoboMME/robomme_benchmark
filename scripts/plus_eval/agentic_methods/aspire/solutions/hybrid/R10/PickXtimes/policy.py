# v03 — PickXtimes: v02 + cube perception by colour-filtered segment_text candidates
# (top-1 of localize_object can be a distractor cube of another colour, ep2), localize_object
# kept only as fallback.
import math

report = {"version": "v03", "calls": [], "cycles": [], "events": []}
STATE = {"terminated": False, "ac_mode": 0}
MOVE_DT = []
TICK_DT = []
REL_LEAD = 2.0
TARGET_OBS = []   # (t, x, y, free_phase_id)  free_phase_id=-1 when the cube rests on it


def note(msg):
    if len(report["events"]) < 400:
        report["events"].append(msg)


def now_s():
    try:
        return float(get_robot_state().timestamp_s)
    except Exception:
        return -1.0


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return 0.0
    return ordered[int((len(ordered) - 1) * fraction)]


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def qrot(q, v):
    p = (0.0, v[0], v[1], v[2])
    qc = (q[0], -q[1], -q[2], -q[3])
    r = qmul(qmul(q, p), qc)
    return (r[1], r[2], r[3])


def qz(angle):
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def wrap(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def check_terminal(res):
    try:
        if bool(res.terminated) or bool(res.truncated):
            STATE["terminated"] = True
    except Exception:
        pass


def log_call(name, res):
    entry = [name]
    try:
        entry.append(bool(res.ok))
    except Exception:
        entry.append(None)
    try:
        entry.append(int(res.steps_executed))
    except Exception:
        entry.append(None)
    try:
        entry.append(str(res.status))
    except Exception:
        entry.append(None)
    entry.append(round(now_s(), 2))
    if len(report["calls"]) < 600:
        report["calls"].append(entry)
    check_terminal(res)


def move(name, pos, quat, tol=0.012, steps=200):
    if STATE["terminated"]:
        return None
    t_a = now_s()
    try:
        res = move_to_pose(Pose((float(pos[0]), float(pos[1]), float(pos[2])), quat, BASE),
                           tolerance=tol, max_steps=steps)
    except Exception as exc:
        note("move " + name + " exc " + str(exc))
        return None
    log_call(name, res)
    t_b = now_s()
    if t_a >= 0 and t_b >= t_a and len(MOVE_DT) < 200:
        MOVE_DT.append(t_b - t_a)
    return res


def gripper(name, open_it):
    if STATE["terminated"]:
        return None
    try:
        res = open_gripper() if open_it else close_gripper()
    except Exception as exc:
        note("gripper exc " + str(exc))
        return None
    log_call(name, res)
    return res


def make_action(joints, grip):
    cmd = None
    if STATE["ac_mode"] == 0:
        try:
            cmd = ArmCommand("joint_position", joints, gripper_position=grip, embodiment=EMB)
        except Exception:
            STATE["ac_mode"] = 1
    if cmd is None:
        cmd = ArmCommand("joint_position", joints, gripper_position=grip)
    return RobotAction(arms={"primary": cmd})


def hold(n, grip):
    if STATE["terminated"] or n <= 0:
        return 0
    st = get_robot_state()
    joints = [float(v) for v in st.joint_positions["primary"]]
    t_a = now_s()
    done = hold_ticks(joints, n, grip)
    t_b = now_s()
    if done > 0 and t_a >= 0 and t_b > t_a and len(TICK_DT) < 400:
        TICK_DT.append((t_b - t_a) / done)
    return done


def hold_ticks(joints, n, grip):
    done = 0
    for tick in range(n):
        try:
            res = step(make_action(joints, grip))
        except Exception as exc:
            note("step exc mode" + str(STATE["ac_mode"]) + " " + str(exc))
            if STATE["ac_mode"] == 0:
                STATE["ac_mode"] = 1
                try:
                    res = step(make_action(joints, grip))
                except Exception as exc2:
                    note("step exc2 " + str(exc2))
                    return done
            else:
                return done
        done += 1
        check_terminal(res)
        if STATE["terminated"]:
            break
    return done


def cloud_colors(pc):
    try:
        cols = pc.colors
        if cols is None:
            return None
        n = len(cols)
        if n == 0:
            return None
        sr = 0.0
        sg = 0.0
        sb = 0.0
        for c in cols:
            sr += float(c[0])
            sg += float(c[1])
            sb += float(c[2])
        m = (sr / n, sg / n, sb / n)
        if max(m) <= 1.0:
            m = (m[0] * 255.0, m[1] * 255.0, m[2] * 255.0)
        return m
    except Exception:
        return None


def summarize(points):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    zs = [float(p[2]) for p in points]
    return {
        "n": len(xs),
        "x": quantile(xs, 0.5), "y": quantile(ys, 0.5),
        "z_lo": quantile(zs, 0.03), "z_hi": quantile(zs, 0.97), "z_med": quantile(zs, 0.5),
        "dx": quantile(xs, 0.95) - quantile(xs, 0.05),
        "dy": quantile(ys, 0.95) - quantile(ys, 0.05),
    }


def is_pink(col):
    if col is None:
        return True
    r, g, b = col
    return b > g + 15 and r > g + 10 and r > 90


def is_green(col):
    if col is None:
        return True
    r, g, b = col
    return g > r + 30 and g > b + 30


def is_grey(col):
    if col is None:
        return True
    r, g, b = col
    return abs(r - g) < 40 and abs(g - b) < 40 and g > 80


def accept_cube(s, col):
    if not is_green(col):
        note("cube cand reject color " + str(s["col"]))
        return False
    return s["dx"] <= 0.10 and s["dy"] <= 0.10


def find_cube(camera="front", min_pts=8):
    s = seg_candidates("cube", CUBE_PROMPTS, camera, accept_cube)
    if s is not None:
        return s
    for prompt in CUBE_PROMPTS:
        try:
            loc = localize_object(prompt, camera_name=camera, target_frame=BASE)
        except Exception as exc:
            note("loc exc " + prompt + " " + str(exc))
            continue
        try:
            ok = bool(loc.ok)
        except Exception:
            ok = False
        if not ok:
            continue
        pts = loc.point_cloud.points
        if len(pts) < min_pts:
            continue
        s = summarize(pts)
        col = cloud_colors(loc.point_cloud)
        s["col"] = None if col is None else [round(c) for c in col]
        s["prompt"] = prompt
        if s["dx"] > 0.10 or s["dy"] > 0.10:
            note("cube reject size " + camera + " " + str(round(s["dx"], 3)) + "," + str(round(s["dy"], 3)))
            continue
        if not is_green(col):
            note("cube reject color " + camera + " " + str(s["col"]))
            continue
        s["points"] = pts
        return s
    return None


def seg_candidates(kind, prompts, camera, accept, max_cands=4):
    """segment_text, then project the top candidates and return the first accepted one."""
    for prompt in prompts:
        try:
            seg = segment_text(camera, prompt)
        except Exception as exc:
            note(kind + " seg exc " + str(exc))
            continue
        try:
            if not bool(seg.ok):
                continue
            cands = sorted([c for c in seg.segmentations], key=lambda c: -float(c.score))
        except Exception as exc:
            note(kind + " seg read exc " + str(exc))
            continue
        for cand in cands[:max_cands]:
            try:
                pc = mask_to_point_cloud(cand, camera, target_frame=BASE)
                pts = pc.points
            except Exception as exc:
                note(kind + " proj exc " + str(exc))
                continue
            if len(pts) < 12:
                continue
            s = summarize(pts)
            col = cloud_colors(pc)
            s["col"] = None if col is None else [round(c) for c in col]
            s["prompt"] = prompt
            s["cam"] = camera
            try:
                s["score"] = round(float(cand.score), 3)
            except Exception:
                s["score"] = None
            if accept(s, col):
                s["points"] = pts
                return s
    return None


def bbox_center(s):
    xs = [float(p[0]) for p in s["points"]]
    ys = [float(p[1]) for p in s["points"]]
    return (0.5 * (quantile(xs, 0.05) + quantile(xs, 0.95)), 0.5 * (quantile(ys, 0.05) + quantile(ys, 0.95)))


def accept_target(s, col):
    if not is_pink(col):
        return False
    if s["z_hi"] > GEOM["table_z"] + 0.03:
        return False
    lo = min(s["dx"], s["dy"])
    hi = max(s["dx"], s["dy"])
    dref = GEOM["tgt_d"]
    if dref is None:
        return 0.035 <= lo and hi <= 0.12
    return lo >= 0.75 * dref and hi <= 1.4 * dref


def accept_button(s, col):
    if not is_grey(col):
        return False
    if s["z_hi"] > GEOM["table_z"] + 0.07:
        return False
    return 0.015 <= max(s["dx"], s["dy"]) <= 0.09


def observe_target(phase_id, cams=("front", "wrist")):
    for cam in cams:
        s = seg_candidates("target", TARGET_PROMPTS, cam, accept_target)
        if s is not None:
            c = bbox_center(s)
            s["cx"] = c[0]
            s["cy"] = c[1]
            t = now_s()
            TARGET_OBS.append((t, c[0], c[1], phase_id))
            if len(OBS_LOG) < 400:
                OBS_LOG.append([round(t, 2), round(c[0], 4), round(c[1], 4), phase_id, cam,
                                round(s["dx"], 3), round(s["dy"], 3), s["n"]])
            return s
    if len(OBS_LOG) < 400:
        OBS_LOG.append([round(now_s(), 2), None, None, phase_id, "miss"])
    return None


def classify():
    """'static', 'moving' or None from free-phase target observations."""
    free = [o for o in TARGET_OBS if o[3] >= 0]
    phases = {}
    for o in free:
        phases.setdefault(o[3], []).append(o)
    verdict = None
    for pid in phases:
        obs = phases[pid]
        for i in range(len(obs)):
            for j in range(i + 1, len(obs)):
                d = math.sqrt((obs[i][1] - obs[j][1]) ** 2 + (obs[i][2] - obs[j][2]) ** 2)
                if d > 0.02:
                    return "moving"
        if len(obs) >= 2 and obs[-1][0] - obs[0][0] >= 0.6:
            spread = max(math.sqrt((o[1] - obs[-1][1]) ** 2 + (o[2] - obs[-1][2]) ** 2) for o in obs)
            if spread < 0.012:
                verdict = "static"
    return verdict


def solve_lin(m):
    n = len(m)
    for col in range(n):
        piv = col
        for row in range(col + 1, n):
            if abs(m[row][col]) > abs(m[piv][col]):
                piv = row
        if abs(m[piv][col]) < 1e-15:
            return None
        m[col], m[piv] = m[piv], m[col]
        for row in range(n):
            if row != col:
                f = m[row][col] / m[col][col]
                for k in range(n + 1):
                    m[row][k] -= f * m[col][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def ell_theta(e, x, y):
    dx = x - e["cx"]
    dy = y - e["cy"]
    u1 = e["l11"] * dx + e["l21"] * dy
    u2 = e["l22"] * dy
    return math.atan2(u2, u1), math.sqrt(u1 * u1 + u2 * u2)


def ell_point(e, th):
    dy = math.sin(th) / e["l22"]
    dx = (math.cos(th) - e["l21"] * dy) / e["l11"]
    return (e["cx"] + dx, e["cy"] + dy)


def ell_resid(e, x, y):
    th, rad = ell_theta(e, x, y)
    if rad <= 1e-9:
        return 1.0
    d = math.sqrt((x - e["cx"]) ** 2 + (y - e["cy"]) ** 2)
    return abs(d * (1.0 - 1.0 / rad))


def fit_ellipse():
    """Conic least squares A x'^2 + B x'y' + C y'^2 + D x' + E y' = 1 on all target observations
    (x' = x - mean); returns centre + Cholesky map to the unit circle, shape and coverage stats."""
    pts = [(o[1], o[2]) for o in TARGET_OBS]
    if len(pts) < 8:
        return None
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    rows = []
    for (x, y) in pts:
        a = x - mx
        b = y - my
        rows.append((a * a, a * b, b * b, a, b))
    m = []
    for i in range(5):
        line = [sum(r[i] * r[j] for r in rows) for j in range(5)]
        line.append(sum(r[i] for r in rows))
        m.append(line)
    sol = solve_lin(m)
    if sol is None:
        return None
    A, B, C, D, E = sol
    det = A * C - B * B / 4.0
    if A <= 0 or det <= 0:
        return None
    # centre of the conic: M c = -g/2
    cxp = (-D / 2.0 * C + E / 2.0 * B / 2.0) / det
    cyp = (-E / 2.0 * A + D / 2.0 * B / 2.0) / det
    k = 1.0 + A * cxp * cxp + B * cxp * cyp + C * cyp * cyp
    if k <= 0:
        return None
    m11 = A / k
    m12 = B / 2.0 / k
    m22 = C / k
    l11 = math.sqrt(m11)
    l21 = m12 / l11
    rest = m22 - l21 * l21
    if rest <= 0:
        return None
    e = {"cx": mx + cxp, "cy": my + cyp, "l11": l11, "l21": l21, "l22": math.sqrt(rest)}
    tr = m11 + m22
    dd = math.sqrt(max(0.0, (m11 - m22) ** 2 / 4.0 + m12 * m12))
    lam_hi = tr / 2.0 + dd
    lam_lo = tr / 2.0 - dd
    if lam_lo <= 0:
        return None
    e["semi_major"] = 1.0 / math.sqrt(lam_lo)
    e["semi_minor"] = 1.0 / math.sqrt(lam_hi)
    res = [ell_resid(e, x, y) for (x, y) in pts]
    e["resid"] = quantile(res, 0.9)
    bins = set()
    for (x, y) in pts:
        th, rad = ell_theta(e, x, y)
        bins.add(int(((th + math.pi) / (2.0 * math.pi)) * 12.0) % 12)
    e["coverage"] = len(bins)
    e["n"] = len(pts)
    return e


def ell_omega(e):
    """Common angular rate (rad/s) of the parametric angle, pooled within free-motion runs
    (same free phase id >= 0, consecutive gaps <= 1.0 s, residual < 0.012 m)."""
    runs = []
    cur = []
    last = None
    for o in TARGET_OBS:
        ok = o[3] >= 0 and ell_resid(e, o[1], o[2]) < 0.012
        if not ok:
            continue
        th = ell_theta(e, o[1], o[2])[0]
        if last is not None and last[3] == o[3] and 0.0 < o[0] - last[0] <= 1.0:
            cur.append((o[0], cur[-1][1] + wrap(th - cur[-1][2]), th))
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = [(o[0], th, th)]
        last = o
    if len(cur) >= 2:
        runs.append(cur)
    num = 0.0
    den = 0.0
    npts = 0
    for r in runs:
        tm = sum(p[0] for p in r) / len(r)
        am = sum(p[1] for p in r) / len(r)
        for p in r:
            num += (p[0] - tm) * (p[1] - am)
            den += (p[0] - tm) ** 2
        npts += len(r)
    if den <= 1e-9:
        return None, npts
    return num / den, npts


def ell_phase(e, w, phase_id):
    """phi such that theta(t) = phi + w t, from the latest (<=5) observations of this free phase."""
    obs = [o for o in TARGET_OBS if o[3] == phase_id and o[3] >= 0 and ell_resid(e, o[1], o[2]) < 0.012]
    if len(obs) == 0:
        return None
    obs = obs[-5:]
    sx = 0.0
    sy = 0.0
    for o in obs:
        ph = ell_theta(e, o[1], o[2])[0] - w * o[0]
        sx += math.cos(ph)
        sy += math.sin(ph)
    return math.atan2(sy, sx)


def good_ellipse(e, w, nw):
    return (e is not None and w is not None and e["coverage"] >= 8 and e["resid"] < 0.006
            and 0.015 < e["semi_minor"] and e["semi_major"] < 0.35 and nw >= 6 and abs(w) > 0.1)


def parse_count(text):
    words = {"one": 1, "once": 1, "two": 2, "twice": 2, "three": 3, "thrice": 3, "four": 4,
             "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
             "eleven": 11, "twelve": 12}
    cleaned = ""
    for ch in text.lower():
        cleaned += ch if (ch.isalnum() or ch == " ") else " "
    toks = cleaned.split()
    for i, tok in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if tok in ("once", "twice", "thrice"):
            return words[tok]
        if nxt == "times" or nxt == "time":
            if tok.isdigit():
                return int(tok)
            if tok in words:
                return words[tok]
    for tok in toks:
        if tok.isdigit():
            return int(tok)
    return 1


CUBE_PROMPTS = ("green cube", "green block")
TARGET_PROMPTS = ("pink and white target", "purple bullseye target")
BUTTON_PROMPTS = ("button", "white button")
OBS_LOG = []
GEOM = {"table_z": 0.01, "tgt_d": None}

# ------------------------------------------------------------------ setup
ctx = get_task_context()
GOAL = ctx.language
N = parse_count(GOAL)
report["goal"] = GOAL
report["N"] = N
MAX_STEPS = 0
try:
    meta = robomme_plus.get_task_metadata()
    MAX_STEPS = int(meta["max_steps"])
except Exception as exc:
    note("meta exc " + str(exc))
report["max_steps"] = MAX_STEPS

st0 = get_robot_state()
BASE = st0.base_frame
EMB = st0.embodiment
ee0 = st0.end_effector_poses["primary"]
Q0 = tuple(float(v) for v in ee0.quaternion_wxyz)
P0 = tuple(float(v) for v in ee0.position)
zaxis = qrot(Q0, (0.0, 0.0, 1.0))
QDOWN = Q0 if zaxis[2] < -0.9 else (0.0, 1.0, 0.0, 0.0)
yax = qrot(QDOWN, (0.0, 1.0, 0.0))
PHI = math.atan2(yax[1], yax[0])


def brief(s):
    if s is None:
        return None
    out = {}
    for k in s:
        if k == "points":
            continue
        v = s[k]
        if k in ("x", "y", "z_lo", "z_hi", "z_med", "dx", "dy", "cx", "cy"):
            v = round(v, 4)
        out[k] = v
    return out


cube0 = find_cube()
if cube0 is not None:
    GEOM["table_z"] = cube0["z_lo"]
tgt0 = observe_target(0, cams=("front",))
if tgt0 is None:
    tgt0 = observe_target(0, cams=("wrist",))
if tgt0 is not None:
    GEOM["table_z"] = tgt0["z_med"]
    GEOM["tgt_d"] = max(0.04, min(0.11, max(tgt0["dx"], tgt0["dy"])))
btn0 = seg_candidates("button", BUTTON_PROMPTS, "front", accept_button)
TABLE_Z = GEOM["table_z"]
CUBE_H = 0.035
if cube0 is not None:
    CUBE_H = min(0.06, max(0.02, cube0["z_hi"] - TABLE_Z))
TGT_TOP = TABLE_Z if tgt0 is None else tgt0["z_hi"]
HOVER_Z = TABLE_Z + 0.17
report["initial"] = {"cube": brief(cube0), "target": brief(tgt0), "button": brief(btn0)}
report["geom"] = {"table_z": round(TABLE_Z, 4), "cube_h": round(CUBE_H, 4), "tgt_top": round(TGT_TOP, 4),
                  "tgt_d": GEOM["tgt_d"], "phi_deg": round(math.degrees(PHI), 1),
                  "q0": [round(v, 4) for v in Q0], "p0": [round(v, 4) for v in P0]}
LAST_T = {"xy": None if tgt0 is None else (tgt0["cx"], tgt0["cy"])}
MODE = {"v": None, "ell": None, "w": None}
def cube_yaw(points, top_z):
    top = [(float(p[0]), float(p[1])) for p in points if float(p[2]) > top_z - 0.008]
    if len(top) < 15:
        return None, len(top)
    best = None
    best_a = 0.0
    for k in range(30):
        a = math.radians(3.0 * k)
        ca = math.cos(a)
        sa = math.sin(a)
        us = [x * ca + y * sa for (x, y) in top]
        vs = [-x * sa + y * ca for (x, y) in top]
        area = (quantile(us, 0.95) - quantile(us, 0.05)) * (quantile(vs, 0.95) - quantile(vs, 0.05))
        if best is None or area < best:
            best = area
            best_a = a
    return best_a, len(top)


def grasp_quat(yaw):
    if yaw is None:
        return QDOWN
    delta = yaw - PHI
    while delta > math.pi / 4.0:
        delta -= math.pi / 2.0
    while delta < -math.pi / 4.0:
        delta += math.pi / 2.0
    return qmul(qz(delta), QDOWN)


def grip_reading():
    try:
        return float(get_robot_state().gripper_positions["primary"])
    except Exception:
        return -1.0




# ------------------------------------------------------------------ one cycle
def pick(phase_id):
    info = {}
    cube = find_cube()
    if cube is None:
        move("obs_pose", P0, QDOWN, tol=0.02)
        cube = find_cube()
    if cube is None:
        info["fail"] = "cube not found"
        return False, info, QDOWN
    if phase_id >= 0:
        observe_target(phase_id, cams=("front",))
    cx, cy, top = cube["x"], cube["y"], cube["z_hi"]
    info["cube_front"] = [round(cx, 4), round(cy, 4), round(top, 4)]
    gripper("open", True)
    move("pregrasp", (cx, cy, top + 0.10), QDOWN, tol=0.015)
    yaw = None
    w = find_cube(camera="wrist", min_pts=20)
    if w is not None:
        info["cube_wrist"] = [round(w["x"], 4), round(w["y"], 4), round(w["z_hi"], 4), w["n"]]
        if abs(w["x"] - cx) < 0.04 and abs(w["y"] - cy) < 0.04:
            cx, cy = w["x"], w["y"]
            if abs(w["z_hi"] - top) < 0.03:
                top = w["z_hi"]
            yaw, ntop = cube_yaw(w["points"], w["z_hi"])
    info["yaw_deg"] = None if yaw is None else round(math.degrees(yaw), 1)
    q = grasp_quat(yaw)
    gz = max(top - 0.6 * CUBE_H, TABLE_Z + 0.008)
    move("above_grasp", (cx, cy, gz + 0.05), q, tol=0.012)
    move("grasp", (cx, cy, gz), q, tol=0.008, steps=120)
    gripper("close", False)
    move("lift", (cx, cy, HOVER_Z), q, tol=0.015)
    g = grip_reading()
    info["grip"] = round(g, 3)
    held = g > 0.05
    info["held"] = held
    info["picked_from"] = [round(cx, 4), round(cy, 4)]
    return held, info, q


def tick_s():
    if len(TICK_DT) == 0:
        return 0.05
    return quantile(TICK_DT, 0.5)


def ell_eta(e, w, phi, th_goal, t):
    """seconds until the parametric angle reaches th_goal (next arrival, in [0, period))."""
    th_now = phi + w * t
    d = (th_goal - th_now) if w > 0 else (th_now - th_goal)
    two = 2.0 * math.pi
    while d < 0:
        d += two
    while d >= two:
        d -= two
    return d / abs(w)


def choose_wait(e, w, phi, lead):
    """sample the path; among slow points (speed <= 1.35 x min speed) take the earliest arrival
    at least `lead` seconds ahead."""
    samples = []
    eps = 0.01
    for j in range(72):
        th = -math.pi + j * (2.0 * math.pi / 72.0)
        a = ell_point(e, th - eps)
        b = ell_point(e, th + eps)
        v = math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) / (2.0 * eps) * abs(w)
        samples.append((th, v))
    vmin = min(s[1] for s in samples)
    t = now_s()
    period = 2.0 * math.pi / abs(w)
    best = None
    for (th, v) in samples:
        if v > 1.35 * vmin:
            continue
        eta = ell_eta(e, w, phi, th, t)
        if eta < lead:
            eta += period
        if best is None or eta < best[1]:
            best = (th, eta, v)
    return best, vmin


def ensure_model(phase_id, max_iter):
    """refit the ellipse from all observations; gather more observations (hold 5 ticks +
    observe) until the model is good or max_iter is exhausted."""
    e = fit_ellipse()
    w, nw = (None, 0) if e is None else ell_omega(e)
    k = 0
    while not good_ellipse(e, w, nw) and k < max_iter and not STATE["terminated"]:
        k += 1
        hold(5, 0.0)
        observe_target(phase_id)
        e = fit_ellipse()
        w, nw = (None, 0) if e is None else ell_omega(e)
    return e, w, nw, k


def place(phase_id, q):
    info = {}
    guess = LAST_T["xy"]
    if guess is None:
        guess = (P0[0], P0[1])
    view = guess
    if MODE["ell"] is not None:
        view = (MODE["ell"]["cx"], MODE["ell"]["cy"])
    # observe from above (the arm does not occlude the target from there; the wrist camera
    # sees the whole path from above its centre)
    move("obs_hover", (view[0], view[1], HOVER_Z), q, tol=0.015)
    s = observe_target(phase_id)
    if s is None:
        move("obs_pose", (P0[0], P0[1], max(P0[2], HOVER_Z)), q, tol=0.02)
        s = observe_target(phase_id)
    if s is None:
        info["fail"] = "target not found"
        return False, info
    LAST_T["xy"] = (s["cx"], s["cy"])
    if MODE["v"] != "moving":
        hold(4, 0.0)
        s2 = observe_target(phase_id)
        if s2 is not None:
            LAST_T["xy"] = (s2["cx"], s2["cy"])
        cls = classify()
        if cls is not None:
            MODE["v"] = cls
    info["mode_before"] = MODE["v"]
    if MODE["v"] != "moving":
        tx, ty = LAST_T["xy"]
        move("over_target", (tx, ty, HOVER_Z - 0.05), q, tol=0.012)
        rel_z = TGT_TOP + 0.4 * CUBE_H + 0.006
        move("place_low", (tx, ty, rel_z), q, tol=0.008)
        gripper("release", True)
        info["mode"] = "static"
        info["place_xy"] = [round(tx, 4), round(ty, 4)]
        return True, info
    # moving target: path model = ellipse with uniform parametric angle (refit every cycle)
    e, w, nw, k = ensure_model(phase_id, 40 if MODE["ell"] is None else 12)
    info["gather_iters"] = k
    if good_ellipse(e, w, nw):
        MODE["ell"] = e
        MODE["w"] = w
    e = MODE["ell"]
    w = MODE["w"]
    drop_z = TGT_TOP + 0.4 * CUBE_H + 0.016
    if e is None:
        tx, ty = LAST_T["xy"]
        for o in TARGET_OBS:
            if o[3] >= 0:
                tx, ty = o[1], o[2]
        move("chase_low", (tx, ty, drop_z), q, tol=0.01)
        gripper("release", True)
        info["mode"] = "chase"
        return True, info
    info["ellipse"] = {kk: round(e[kk], 4) for kk in e}
    info["w"] = round(w, 4)
    # current-phase observation needed for the phase estimate
    phi = ell_phase(e, w, phase_id)
    tries = 0
    while phi is None and tries < 6 and not STATE["terminated"]:
        tries += 1
        hold(3, 0.0)
        observe_target(phase_id)
        phi = ell_phase(e, w, phase_id)
    if phi is None:
        info["fail"] = "no phase observation"
        tx, ty = LAST_T["xy"]
        move("chase_low", (tx, ty, drop_z), q, tol=0.01)
        gripper("release", True)
        info["mode"] = "chase"
        return True, info
    lead = 1.2
    if len(MOVE_DT) >= 3:
        lead = quantile(MOVE_DT, 0.75) + 0.5
    best, vmin = choose_wait(e, w, phi, lead)
    th_w = best[0]
    wx, wy = ell_point(e, th_w)
    info["wait"] = {"xy": [round(wx, 4), round(wy, 4)], "eta_s": round(best[1], 2),
                    "speed": round(best[2], 4), "vmin": round(vmin, 4), "lead": round(lead, 2)}
    move("wait_low", (wx, wy, drop_z), q, tol=0.008)
    tk = tick_s()
    log = []
    released = False
    for it in range(120):
        if STATE["terminated"]:
            break
        eta_t = ell_eta(e, w, phi, th_w, now_s()) / tk
        period_t = 2.0 * math.pi / abs(w) / tk
        if eta_t > period_t - 3.0:
            eta_t = 0.0   # just passed (within 3 ticks): release now
        if len(log) < 40:
            log.append(round(eta_t, 1))
        rem = eta_t - REL_LEAD
        if rem <= 0.5:
            gripper("release", True)
            released = True
            break
        if rem <= 8:
            hold(int(round(rem)), 0.0)
            continue
        hold(min(5, int(rem - 6)), 0.0)
        o = observe_target(phase_id, cams=("front",))
        if o is not None:
            r = ell_resid(e, o["cx"], o["cy"])
            pred = ell_point(e, phi + w * now_s())
            dpred = math.sqrt((pred[0] - o["cx"]) ** 2 + (pred[1] - o["cy"]) ** 2)
            if r < 0.012 and dpred < 0.04:
                phi2 = ell_phase(e, w, phase_id)
                if phi2 is not None:
                    phi = phi2
            if len(log) < 40:
                log.append("d" + str(round(dpred, 3)))
    if not released:
        gripper("release", True)
    info["mode"] = "intercept_wait"
    info["eta_log"] = log
    info["place_xy"] = [round(wx, 4), round(wy, 4)]
    LAST_T["xy"] = (wx, wy)
    return True, info


# ------------------------------------------------------------------ main loop
LAST_OK = {"v": False}
count = 0
attempts = 0
phase = 0
try:
    cyc_ticks = []
    while count < N and attempts < 3 * N + 4 and not STATE["terminated"]:
        if MAX_STEPS > 0 and len(cyc_ticks) > 0:
            used = now_s() / tick_s()
            need = max(cyc_ticks) + 160
            if used + need > MAX_STEPS:
                note("stop cycling: step budget used=" + str(round(used)) + " need=" + str(round(need)))
                break
        elif MAX_STEPS <= 0 and attempts >= N + 4:
            break
        attempts += 1
        cyc = {"i": attempts, "t_start": round(now_s(), 2)}
        on_target = LAST_OK["v"]
        held, pinfo, q = pick(-1 if on_target else phase)
        cyc["pick"] = pinfo
        if not held:
            cyc["result"] = "grasp failed"
            report["cycles"].append(cyc)
            move("retreat", (P0[0], P0[1], max(P0[2], HOVER_Z)), QDOWN, tol=0.02)
            continue
        if on_target:
            phase += 1
        ok, linfo = place(phase, q)
        cyc["place"] = linfo
        try:
            ep = get_robot_state().end_effector_poses["primary"].position
            move("post_lift", (float(ep[0]), float(ep[1]), TABLE_Z + 0.12), q, tol=0.02)
        except Exception as exc:
            note("post_lift exc " + str(exc))
        c_after = find_cube()
        t_a = observe_target(-1, cams=("front",))
        hold(10 if MODE["v"] == "moving" else 3, 1.0)
        t_b = observe_target(-1, cams=("front",))
        verdict = {}
        ref = LAST_T["xy"]
        if c_after is not None:
            verdict["cube_xy"] = [round(c_after["x"], 4), round(c_after["y"], 4)]
            verdict["cube_z"] = round(c_after["z_med"], 4)
            if ref is not None:
                verdict["cube_ref_d"] = round(math.sqrt((c_after["x"] - ref[0]) ** 2 + (c_after["y"] - ref[1]) ** 2), 4)
        if t_a is not None and t_b is not None:
            verdict["tgt_drift"] = round(math.sqrt((t_a["cx"] - t_b["cx"]) ** 2 + (t_a["cy"] - t_b["cy"]) ** 2), 4)
        success = ok
        if "cube_ref_d" in verdict and verdict["cube_ref_d"] > 0.035:
            success = False
        if "tgt_drift" in verdict and verdict["tgt_drift"] > 0.012:
            success = False
        if c_after is not None and success:
            LAST_T["xy"] = (c_after["x"], c_after["y"])
        cyc["verdict"] = verdict
        cyc["counted"] = success
        LAST_OK["v"] = success
        if success:
            count += 1
        else:
            phase += 1
        cyc["count"] = count
        cyc["t_end"] = round(now_s(), 2)
        cyc_ticks.append((cyc["t_end"] - cyc["t_start"]) / tick_s())
        report["cycles"].append(cyc)
    # ------------------------------------------------------------ button
    report["placements"] = count
    if not STATE["terminated"]:
        btn = btn0
        if btn is None:
            move("obs_pose_b", (P0[0], P0[1], max(P0[2], HOVER_Z)), QDOWN, tol=0.02)
            btn = seg_candidates("button", BUTTON_PROMPTS, "front", accept_button)
        report["button"] = brief(btn)
        if btn is not None:
            bx, by, btop = btn["x"], btn["y"], btn["z_hi"]
            move("btn_lift", (bx, by, HOVER_Z), QDOWN, tol=0.02)
            gripper("btn_close", False)
            move("btn_above", (bx, by, btop + 0.05), QDOWN, tol=0.012)
            move("btn_press", (bx, by, btop - 0.012), QDOWN, tol=0.01, steps=80)
            hold(4, 0.0)
            if not STATE["terminated"]:
                move("btn_up", (bx, by, btop + 0.05), QDOWN, tol=0.02)
            if not STATE["terminated"]:
                move("btn_press2", (bx, by, btop - 0.022), QDOWN, tol=0.01, steps=80)
                hold(6, 0.0)
            if not STATE["terminated"]:
                move("btn_up2", (bx, by, btop + 0.06), QDOWN, tol=0.02)
except Exception as exc:
    report["error"] = str(exc)

report["terminated"] = STATE["terminated"]
report["obs_log"] = OBS_LOG
report["tick_s"] = tick_s()
report["model"] = None if MODE["ell"] is None else {k: round(MODE["ell"][k], 4) for k in MODE["ell"]}
report["w"] = MODE["w"]
report["t_final"] = now_s()
result = report
