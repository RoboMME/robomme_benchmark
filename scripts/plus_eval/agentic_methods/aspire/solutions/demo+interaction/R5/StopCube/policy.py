# StopCube v00 — count cube arrivals at the target per tick, press the button on arrival N.
# Allowed public APIs only. All scene quantities are derived at runtime.
import math

report = {"version": "v01", "events": []}

def note(msg):
    if len(report["events"]) < 400:
        report["events"].append(msg)

# ---------------------------------------------------------------- goal / metadata
context = get_task_context()
goal = str(context.language)
report["goal"] = goal

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
            "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
            "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
            "twentieth": 20, "once": 1, "twice": 2, "thrice": 3}

def parse_count(text):
    words = text.lower().replace(",", " ").replace(".", " ").split()
    for w in words:
        if w in ORDINALS:
            return ORDINALS[w]
    for w in words:
        digits = ""
        for ch in w:
            if ch in "0123456789":
                digits += ch
            else:
                break
        if digits != "" and len(digits) < 3:
            return int(digits)
    return None

N_TARGET = parse_count(goal)
if N_TARGET is None:
    N_TARGET = 1
    note("could not parse count; default 1")
report["n_target"] = N_TARGET

max_steps = None
try:
    meta = robomme_plus.get_task_metadata()
    report["task_metadata"] = {}
    for key in meta:
        try:
            report["task_metadata"][str(key)] = str(meta[key])[:200]
        except Exception:
            pass
    max_steps = int(meta["max_steps"])
except Exception as exc:
    note("metadata error: " + str(exc)[:200])
report["max_steps"] = max_steps

try:
    video = robomme_plus.get_conditioning_video()
    report["conditioning_video_len"] = len(video)
except Exception as exc:
    report["conditioning_video_len"] = "error: " + str(exc)[:120]

state0 = get_robot_state()
EMB = state0.embodiment
BASE = state0.base_frame
q_home = [float(v) for v in state0.joint_positions["primary"]]
ee0 = state0.end_effector_poses["primary"]
DOWN = [float(v) for v in ee0.quaternion_wxyz]
w, x, y, z = DOWN
zaxis_z = 1.0 - 2.0 * (x * x + y * y)
if zaxis_z > -0.9:
    DOWN = [0.0, 1.0, 0.0, 0.0]
    note("home EE not pointing down (z_z=%.2f); using [0,1,0,0]" % zaxis_z)
report["ee_home"] = [round(float(v), 4) for v in ee0.position]

# ---------------------------------------------------------------- image helpers
def grab(rgb, u0, u1, v0, v1):
    try:
        return rgb[v0:v1, u0:u1].tolist()
    except Exception:
        out = []
        for v in range(v0, v1):
            row = rgb[v]
            r = []
            for u in range(u0, u1):
                px = row[u]
                r.append([int(px[0]), int(px[1]), int(px[2])])
            out.append(r)
        return out

def box_of(cand):
    b = cand.box_xyxy
    return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]

def box_lowsat_fraction(rgb, b, img_w, img_h):
    u0 = max(0, int(b[0])); u1 = min(img_w, int(b[2]) + 1)
    v0 = max(0, int(b[1])); v1 = min(img_h, int(b[3]) + 1)
    if u1 <= u0 or v1 <= v0:
        return 0.0
    sub = grab(rgb, u0, u1, v0, v1)
    n = 0; low = 0
    for row in sub:
        for px in row:
            n += 1
            mx = max(px[0], px[1], px[2]); mn = min(px[0], px[1], px[2])
            if mx - mn < 35 and mx > 90:
                low += 1
    if n == 0:
        return 0.0
    return low / float(n)

def candidates(camera, prompts, rgb, img_w, img_h):
    out = []
    for prompt in prompts:
        try:
            seg = segment_text(camera, prompt)
        except Exception as exc:
            note("segment_text %s/%s error %s" % (camera, prompt, str(exc)[:100]))
            continue
        if not seg.ok:
            continue
        for cand in seg.segmentations[:5]:
            if cand.box_xyxy is None:
                continue
            b = box_of(cand)
            area = (b[2] - b[0]) * (b[3] - b[1])
            out.append({"prompt": prompt, "cand": cand, "box": b, "score": float(cand.score),
                        "area": area, "cx": 0.5 * (b[0] + b[2]), "cy": 0.5 * (b[1] + b[3]),
                        "lowsat": box_lowsat_fraction(rgb, b, img_w, img_h)})
    return out

def summarize(cands):
    return [{"p": c["prompt"], "s": round(c["score"], 3),
             "box": [round(v, 1) for v in c["box"]], "ls": round(c["lowsat"], 2)}
            for c in cands[:20]]

def quantile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]

def cloud_of(cand, camera):
    try:
        pc = mask_to_point_cloud(cand, camera, target_frame=BASE)
        pts = pc.points
        xs = []; ys = []; zs = []
        for p in pts:
            xs.append(float(p[0])); ys.append(float(p[1])); zs.append(float(p[2]))
        return xs, ys, zs
    except Exception as exc:
        note("cloud error " + str(exc)[:120])
        return [], [], []

# ---------------------------------------------------------------- initial perception (front)
obs = get_observation()
front = obs.cameras["front"]
rgb0 = front.rgb
IMG_H = len(rgb0)
IMG_W = len(rgb0[0])

tgt_c = candidates("front", ["target", "bullseye target", "round flat target", "disc"], rgb0, IMG_W, IMG_H)
cube_c = candidates("front", ["cube", "small cube", "block"], rgb0, IMG_W, IMG_H)
btn_c = candidates("front", ["button", "white button", "push button"], rgb0, IMG_W, IMG_H)
report["front_candidates"] = {"target": summarize(tgt_c), "cube": summarize(cube_c),
                              "button": summarize(btn_c)}

MAX_SMALL_AREA = 0.03 * IMG_W * IMG_H

def best_of(cands, avoid, min_sep, prefer_lowsat, prefer_colour):
    best = None; best_key = None
    for c in cands:
        if c["area"] > MAX_SMALL_AREA or c["area"] < 4:
            continue
        bad = False
        for a in avoid:
            if a is not None and abs(c["cx"] - a["cx"]) + abs(c["cy"] - a["cy"]) < min_sep:
                bad = True
        if bad:
            continue
        key = c["score"]
        if prefer_lowsat:
            key = key + (1.0 if c["lowsat"] >= 0.4 else 0.0)
        if prefer_colour:
            key = key + (1.0 if c["lowsat"] < 0.4 else 0.0)
        if best_key is None or key > best_key:
            best = c; best_key = key
    return best

button = best_of(btn_c, [], 0, True, False)
target = best_of(tgt_c, [button], 4.0, False, True)
cube = best_of(cube_c, [button, target], 3.0, False, True)
if target is None:
    raise ValueError("target not found in front view")
if button is None:
    raise ValueError("button not found in front view")
if cube is None:
    note("cube not found by SAM; will initialise from motion")

TU = target["cx"]; TV = target["cy"]
T_R = 0.25 * ((target["box"][2] - target["box"][0]) + (target["box"][3] - target["box"][1]))
THR_IN = max(3.5, 0.7 * T_R)
THR_OUT = max(THR_IN + 3.0, 1.1 * T_R)
report["target_px"] = [round(TU, 1), round(TV, 1), round(T_R, 1), round(THR_IN, 1)]

# ---------------------------------------------------------------- tracker state
WIN = 60
U0 = max(0, int(TU) - WIN); U1 = min(IMG_W, int(TU) + WIN + 1)
V0 = max(0, int(TV) - WIN); V1 = min(IMG_H, int(TV) + WIN + 1)
bg = grab(rgb0, U0, U1, V0, V1)
stale = []
for v in range(V0, V1):
    stale.append([False] * (U1 - U0))
DIFF_T = 60

trk = {"p": None, "prev": None, "hist": [], "lost": 0, "count": 0, "inside": False,
       "arrivals": [], "tick": 0, "stale_box": None, "p0": None, "cube_size": 6.0,
       "dists": [], "pos": [], "exact": []}

if cube is not None:
    cb = cube["box"]
    trk["p"] = [cube["cx"], cube["cy"]]
    trk["p0"] = [cube["cx"], cube["cy"]]
    trk["cube_size"] = max(3.0, 0.5 * ((cb[2] - cb[0]) + (cb[3] - cb[1])))
    sb = [int(cb[0]) - 2, int(cb[1]) - 2, int(cb[2]) + 3, int(cb[3]) + 3]
    trk["stale_box"] = sb
    for v in range(max(V0, sb[1]), min(V1, sb[3])):
        for u in range(max(U0, sb[0]), min(U1, sb[2])):
            stale[v - V0][u - U0] = True
    d0 = math.hypot(cube["cx"] - TU, cube["cy"] - TV)
    trk["inside"] = d0 < THR_IN
report["cube_px0"] = trk["p"]

def foreground_centroid(cur, cu, cv, rad):
    sx = 0.0; sy = 0.0; n = 0
    va = max(V0, int(cv - rad)); vb = min(V1, int(cv + rad) + 1)
    ua = max(U0, int(cu - rad)); ub = min(U1, int(cu + rad) + 1)
    r2 = rad * rad
    for v in range(va, vb):
        rowc = cur[v - V0]; rowb = bg[v - V0]; rows = stale[v - V0]
        dv2 = (v - cv) * (v - cv)
        for u in range(ua, ub):
            if rows[u - U0]:
                continue
            if (u - cu) * (u - cu) + dv2 > r2:
                continue
            a = rowc[u - U0]; b = rowb[u - U0]
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]) > DIFF_T:
                sx += u; sy += v; n += 1
    if n == 0:
        return None, 0
    return [sx / n, sy / n], n

def seg_dist(a, b):
    ax = a[0] - TU; ay = a[1] - TV
    dx = b[0] - a[0]; dy = b[1] - a[1]
    dd = dx * dx + dy * dy
    f = 0.0
    if dd > 1e-9:
        f = -(ax * dx + ay * dy) / dd
        f = max(0.0, min(1.0, f))
    px = ax + f * dx; py = ay + f * dy
    return math.hypot(px, py), f

def track(observation):
    """Update tracker with one front observation (one control tick)."""
    trk["tick"] += 1
    t = trk["tick"]
    rgb = observation.cameras["front"].rgb
    cur = grab(rgb, U0, U1, V0, V1)
    min_px = max(3, int(0.15 * trk["cube_size"] * trk["cube_size"]))
    if trk["p"] is None:
        c, n = foreground_centroid(cur, TU, TV, WIN)
        if c is not None and n >= min_px:
            trk["p"] = c
        trk["prev"] = trk["p"]
        return
    p = trk["p"]
    vel = [0.0, 0.0]
    if trk["prev"] is not None:
        vel = [p[0] - trk["prev"][0], p[1] - trk["prev"][1]]
    pred = [p[0] + vel[0], p[1] + vel[1]]
    gate = max(10.0, 2.0 * trk["cube_size"]) + 3.0 * min(trk["lost"], 4)
    c, n = foreground_centroid(cur, pred[0], pred[1], gate)
    if c is None or n < min_px:
        trk["lost"] += 1
        c = pred
    else:
        trk["lost"] = 0
    # refresh the stale background patch once the cube has left it
    if trk["stale_box"] is not None and trk["p0"] is not None:
        sb = trk["stale_box"]
        hm = 0.5 * trk["cube_size"] + 1.0
        if c[0] < sb[0] - hm or c[0] > sb[2] + hm or c[1] < sb[1] - hm or c[1] > sb[3] + hm:
            for v in range(max(V0, sb[1]), min(V1, sb[3])):
                for u in range(max(U0, sb[0]), min(U1, sb[2])):
                    bg[v - V0][u - U0] = cur[v - V0][u - U0]
                    stale[v - V0][u - U0] = False
            trk["stale_box"] = None
            note("t%d stale background refreshed" % t)
    d, f = seg_dist(p, c)
    if not trk["inside"] and d < THR_IN:
        trk["count"] += 1
        trk["inside"] = True
        trk["arrivals"].append(round(t - 1 + f, 2))
        note("t%d arrival %d (closest %.1fpx)" % (t, trk["count"], d))
    elif trk["inside"] and math.hypot(c[0] - TU, c[1] - TV) > THR_OUT:
        trk["inside"] = False
        refine_last_arrival()
    trk["prev"] = p
    trk["p"] = c
    trk["dists"].append(round(math.hypot(c[0] - TU, c[1] - TV), 1))
    if len(trk["pos"]) < 700:
        trk["pos"].append([round(c[0], 1), round(c[1], 1)])
    if len(trk["hist"]) > 6:
        trk["hist"] = trk["hist"][1:]
    trk["hist"].append(c)

def refine_last_arrival():
    """Sub-tick closest-approach time of the arrival that just ended (parabola on dists)."""
    ds = trk["dists"]
    if len(ds) < 3:
        return
    start = max(1, int(trk["arrivals"][-1]) - 3)
    best = None
    for i in range(start, len(ds) - 1):
        if best is None or ds[i] < ds[best]:
            best = i
    if best is None:
        return
    a = ds[best - 1]; b = ds[best]; c = ds[min(best + 1, len(ds) - 1)]
    den = a - 2.0 * b + c
    off = 0.0
    if den > 1e-6:
        off = max(-0.5, min(0.5, 0.5 * (a - c) / den))
    # dists[i] belongs to tick i + 1
    trk["exact"].append(round(best + 1 + off, 2))

def phase_match():
    """Ticks to the next arrival, from the past tick whose position+velocity best matches now."""
    pos = trk["pos"]
    ex = trk["exact"]
    n = len(pos)
    if n < 8 or len(ex) == 0:
        return None, None
    cx = pos[-1][0]; cy = pos[-1][1]
    sp_now = 0.5 * math.hypot(pos[-1][0] - pos[-3][0], pos[-1][1] - pos[-3][1])
    best = None; best_cost = None
    for i in range(2, n - 6):
        ts = i + 1
        nxt = None
        for a in ex:
            if a > ts:
                nxt = a
                break
        if nxt is None:
            break
        if nxt - ts > 30:
            continue
        cost = 0.0
        for j in range(3):
            cost += math.hypot(pos[i - j][0] - pos[n - 1 - j][0], pos[i - j][1] - pos[n - 1 - j][1])
        cost = cost / 3.0
        if best_cost is None or cost < best_cost:
            sp_past = 0.5 * math.hypot(pos[i][0] - pos[i - 2][0], pos[i][1] - pos[i - 2][1])
            ratio = 1.0
            if sp_now > 0.3:
                ratio = max(0.5, min(2.0, sp_past / sp_now))
            best_cost = cost; best = (nxt - ts) * ratio
    return best, best_cost

def ls_quadratic_weights(npts):
    """Weights W[r][j]: coefficient r of x(t)=c0+c1 t+c2 t^2 fitted to samples at t=0,-1,..."""
    ts = [-float(j) for j in range(npts)]
    S = [[0.0] * 3 for r in range(3)]
    for t in ts:
        phi = [1.0, t, t * t]
        for r in range(3):
            for c in range(3):
                S[r][c] += phi[r] * phi[c]
    # invert 3x3 by adjugate
    a, b, c = S[0]; d, e, f = S[1]; g, h, i = S[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    inv = [[(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
           [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
           [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det]]
    W = []
    for r in range(3):
        row = []
        for j in range(npts):
            t = ts[j]
            phi = [1.0, t, t * t]
            row.append(sum(inv[r][k] * phi[k] for k in range(3)))
        W.append(row)
    return W

QW = ls_quadratic_weights(5)

def ticks_to_arrival_quadratic(kmax=15.0):
    """Closest approach of a least-squares quadratic fit to the last 5 cube positions."""
    pos = trk["pos"]
    if len(pos) < 5:
        return None, None
    pts = [pos[-1 - j] for j in range(5)]
    cx = [sum(QW[r][j] * pts[j][0] for j in range(5)) for r in range(3)]
    cy = [sum(QW[r][j] * pts[j][1] for j in range(5)) for r in range(3)]
    best_k = None; best_d = None
    k = 0.0
    while k <= kmax:
        xk = cx[0] + cx[1] * k + cx[2] * k * k
        yk = cy[0] + cy[1] * k + cy[2] * k * k
        dk = math.hypot(xk - TU, yk - TV)
        if best_d is None or dk < best_d:
            best_d = dk; best_k = k
        k += 0.1
    return best_k, best_d

def ticks_to_arrival():
    """Predicted ticks until closest approach, and predicted miss distance."""
    h = trk["hist"]
    if len(h) < 3:
        return None, None
    p = h[-1]
    vx = 0.5 * (h[-1][0] - h[-3][0]); vy = 0.5 * (h[-1][1] - h[-3][1])
    vv = vx * vx + vy * vy
    if vv < 0.25:
        return None, None
    rx = p[0] - TU; ry = p[1] - TV
    k = -(rx * vx + ry * vy) / vv
    dmin = math.hypot(rx + k * vx, ry + k * vy)
    return k, dmin

# ---------------------------------------------------------------- stepping
status = {"terminated": False, "truncated": False, "last": None, "steps": 0}

def hold(target_joints, gripper=0.0):
    act = RobotAction(arms={"primary": ArmCommand(mode="joint_position",
                                                  target=list(target_joints),
                                                  gripper_position=gripper,
                                                  embodiment=EMB)})
    res = step(act)
    status["steps"] += 1
    try:
        if res.terminated:
            status["terminated"] = True
    except Exception:
        pass
    try:
        if res.truncated:
            status["truncated"] = True
    except Exception:
        pass
    o = None
    try:
        o = res.observation
    except Exception:
        o = None
    if o is not None and not status["terminated"] and not status["truncated"]:
        status["last"] = o
        try:
            track(o)
        except Exception as exc:
            note("track error " + str(exc)[:150])
    return o

def done():
    return status["terminated"] or status["truncated"]

def ee_z(o):
    return float(o.robot_state.end_effector_poses["primary"].position[2])

def current_joints():
    if status["last"] is not None:
        return [float(v) for v in status["last"].robot_state.joint_positions["primary"]]
    return [float(v) for v in get_robot_state().joint_positions["primary"]]

def ik_for(pos, seed):
    target_pose = Pose(list(pos), list(DOWN), BASE)
    res = solve_ik(target_pose, seed_joints=list(seed))
    if not res.ok:
        note("solve_ik failed at %s: %s" % (str([round(v, 3) for v in pos]), str(res.error)[:160]))
        return None
    sol = None
    try:
        sol = res.joint_positions
    except Exception:
        sol = None
    if sol is None:
        try:
            sol = res.solution
        except Exception:
            sol = None
    if sol is None:
        try:
            sol = res.joints
        except Exception:
            sol = None
    if sol is None:
        plan = plan_motion(target_pose, start_joints=list(seed))
        if not plan.ok:
            return None
        sol = plan.trajectory.joint_positions[-1]
    sol = [float(v) for v in sol]
    if len(sol) > 7:
        sol = sol[:7]
    return sol

def interp_move(goal, max_delta=0.05, settle=8):
    start = current_joints()
    span = max(abs(goal[i] - start[i]) for i in range(7))
    n = max(4, int(math.ceil(span / max_delta)))
    for k in range(1, n + 1):
        if done():
            return
        a = k / float(n)
        hold([start[i] + a * (goal[i] - start[i]) for i in range(7)])
    for k in range(settle):
        if done():
            return
        hold(goal)
        cj = current_joints()
        if max(abs(goal[i] - cj[i]) for i in range(7)) < 0.01:
            return

# ---------------------------------------------------------------- button in 3D (front)
bx, by, bz = cloud_of(button["cand"], "front")
if len(bx) < 3:
    raise ValueError("button point cloud empty")
BTN = [quantile(bx, 0.5), quantile(by, 0.5), quantile(bz, 0.9)]
report["button_front_xyz"] = [round(v, 4) for v in BTN]

# ---------------------------------------------------------------- approach (tracked per tick)
pressed = False
try:
    high = [BTN[0], BTN[1], BTN[2] + 0.10]
    q_high = ik_for(high, q_home)
    if q_high is None:
        raise ValueError("IK failed for high hover")
    interp_move(q_high)
    report["ticks_at_high"] = status["steps"]

    # wrist refinement of the button cap
    if not done():
        wo = get_observation()
        wr = wo.cameras["wrist"].rgb
        ww = len(wr[0]); wh = len(wr)
        wc = candidates("wrist", ["button", "white button", "round grey button"], wr, ww, wh)
        report["wrist_candidates"] = summarize(wc)
        best = None; bestd = None
        for c in wc:
            if c["area"] > 0.5 * ww * wh or c["lowsat"] < 0.6:
                continue
            dpx = math.hypot(c["cx"] - 0.5 * ww, c["cy"] - 0.5 * wh) - 20.0 * c["score"]
            if bestd is None or dpx < bestd:
                best = c; bestd = dpx
        if best is not None:
            report["wrist_choice"] = {"p": best["prompt"], "box": [round(v, 1) for v in best["box"]]}
            wx, wy, wz = cloud_of(best["cand"], "wrist")
            if len(wx) >= 10:
                cand_btn = [quantile(wx, 0.5), quantile(wy, 0.5), quantile(wz, 0.95)]
                shift = math.hypot(cand_btn[0] - BTN[0], cand_btn[1] - BTN[1])
                report["button_wrist_xyz"] = [round(v, 4) for v in cand_btn]
                if shift < 0.03:
                    # xy from the close-up wrist view; height never below the front estimate:
                    # a wrist mask can cover only the white base plate (v00 ep1: plate top
                    # 1.6 cm vs cap top 3.4 cm -> the calibration stroke pressed the cap).
                    BTN = [cand_btn[0], cand_btn[1], max(BTN[2], cand_btn[2])]
                else:
                    note("wrist button estimate rejected (shift %.3f)" % shift)

    GAP = 0.015
    STROKE = 0.045
    # v00 ep0: the press registered after ONE tick of the stroke, when the EE reference had
    # travelled 1.2 cm of the 1.5 cm gap -> contact/registration ~4 mm before the reference
    # reaches the depth-estimated cap top (fingertips sit below the reference point).
    TIP_LEAD = 0.004
    p_cal = [BTN[0], BTN[1], BTN[2] + GAP + STROKE]
    p_hov = [BTN[0], BTN[1], BTN[2] + GAP]
    p_prs = [BTN[0], BTN[1], BTN[2] + GAP - STROKE]
    q_cal = ik_for(p_cal, current_joints())
    q_hov = ik_for(p_hov, q_cal if q_cal is not None else current_joints())
    q_prs = ik_for(p_prs, q_hov if q_hov is not None else current_joints())
    if q_cal is None or q_hov is None or q_prs is None:
        raise ValueError("IK failed for press poses")
    report["button_final_xyz"] = [round(v, 4) for v in BTN]

    LAT = 4
    if not done():
        interp_move(q_cal)
        # latency calibration: identical-size free-air stroke cal -> hover
        if not done() and status["last"] is not None:
            z_start = ee_z(status["last"])
            zs = []
            lat = None
            for k in range(1, 16):
                o = hold(q_hov)
                if done() or o is None:
                    break
                zk = ee_z(o)
                zs.append(round(zk, 4))
                if lat is None and z_start - zk >= GAP - TIP_LEAD:
                    lat = k
            report["calibration_z"] = zs
            report["calibration_z_start"] = round(z_start, 4)
            if lat is not None:
                LAT = lat
            else:
                note("calibration did not reach threshold; LAT default")
    report["latency_ticks"] = LAT
    report["ticks_ready"] = status["steps"]
    report["count_when_ready"] = trk["count"]

    # wait at hover, track, fire
    budget = 2000
    if max_steps is not None:
        budget = max_steps
    fire_log = []
    while not done() and status["steps"] < budget - 1:
        fire = False
        if len(trk["exact"]) >= N_TARGET - 1 and trk["lost"] == 0:
            tau, cost = phase_match()
            if tau is not None and cost < 5.0:
                if tau > -0.5 and tau <= LAT + 0.5:
                    fire = True
                    fire_log.append([status["steps"], "phase", round(tau, 2), round(cost, 2)])
            else:
                k, dmin = ticks_to_arrival_quadratic()
                if k is not None and k > 0.05 and dmin < 1.3 * THR_IN and k <= LAT + 0.5:
                    fire = True
                    fire_log.append([status["steps"], "quadratic", round(k, 2), round(dmin, 1)])
        if fire:
            pressed = True
            report["fire_tick"] = status["steps"]
            report["fire_count"] = trk["count"]
            report["fire_prediction"] = fire_log[-1]
            for j in range(30):
                if done():
                    break
                hold(q_prs)
            break
        hold(q_hov)
except Exception as exc:
    report["error"] = str(exc)[:300]

report["pressed"] = pressed
report["steps"] = status["steps"]
report["terminated"] = status["terminated"]
report["truncated"] = status["truncated"]
report["arrivals"] = trk["arrivals"]
report["arrivals_exact"] = trk["exact"]
report["count"] = trk["count"]
report["dists_tail"] = trk["dists"][-60:]
report["dists_all"] = trk["dists"][:600]
report["pos_all"] = trk["pos"]
result = report
