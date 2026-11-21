# PickXtimes v03 -- v02 + target detection-confidence gate, moving/static decision locked after an
#   unoccluded 3-frame home-pose observation, static aim from those home-pose frames only.
#   Built from v02 evidence on training episode 0.
# v02 header: v01 + ellipse (conic) target-path model with uniform eccentric-anomaly rate,
#   segment phase from recent observations, release consistency guard on measured distance.
#   Built from v01 evidence on training episode 2 (and v00 evidence on episode 1).
# v01 header: v00 + freeze-based placement verification, segment-pooled angular speed,
#   occlusion-excluded target fits, static-target aim from unoccluded obs, cube colour check,
#   button candidate filtering.  Built from v00 evidence on training episode 1.
#
# Strategy (all task-dependent quantities derived at runtime):
#   * N = repetition count parsed from get_task_context().language ("... eight times ...").
#     Offline demo analysis: the count INCLUDES the first pick-and-place (8 lifts for "eight").
#   * green cube / target / button localized with SAM3 on the front camera (+ wrist refine).
#   * The target can MOVE along a closed periodic path while the cube is not on it (demo ep1/ep2)
#     and freezes once the cube rests on it.  We track it with repeated observations, fit a
#     circle (Kasa) to every observed target position and estimate angular speed, then
#     intercept: hover the held cube low over a predicted path point and release when the
#     target is about to arrive.  Static targets are handled as the degenerate case.
#   * After each release: verify placement (cube near target and, for a moving target, target
#     frozen).  Only verified placements are counted.  After N, press the button.

import math

BASE = get_robot_state().base_frame
HZ = 20.0
LOG = {"moves": [], "target_obs": [], "events": [], "cycles": []}
STATE = {"ended": False, "end_reason": None}

# ----------------------------------------------------------------------------- utilities

def qtile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def ev(name, data):
    LOG["events"].append([name, data])


def check_end(res, label):
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
        STATE["end_reason"] = [label, term, trunc]
    return term or trunc


def now_s():
    try:
        return float(get_robot_state().timestamp_s)
    except Exception:
        return -1.0


def parse_count(text):
    words = {"one": 1, "once": 1, "two": 2, "twice": 2, "three": 3, "thrice": 3, "four": 4,
             "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
             "eleven": 11, "twelve": 12}
    clean = ""
    for ch in text.lower():
        if ch.isalnum():
            clean += ch
        else:
            clean += " "
    toks = clean.split()
    for i, tok in enumerate(toks):
        if tok in ("times", "time") and i > 0:
            prev = toks[i - 1]
            if prev.isdigit():
                return int(prev)
            if prev in words:
                return words[prev]
    for tok in toks:
        if tok in ("once", "twice", "thrice"):
            return words[tok]
    return None


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def qaxes(q):
    w, x, y, z = q
    xaxis = (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
    zaxis = (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
    return xaxis, zaxis


def rot_z(q, delta):
    return qmul((math.cos(delta / 2.0), 0.0, 0.0, math.sin(delta / 2.0)), q)


def wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def wrap_quarter(a):
    # wrap to [-pi/4, pi/4] (square symmetry)
    while a > math.pi / 4:
        a -= math.pi / 2
    while a < -math.pi / 4:
        a += math.pi / 2
    return a


# ----------------------------------------------------------------------------- motion

def goto(pos, quat, label, tol=0.01, max_steps=200):
    if STATE["ended"]:
        return None
    t0 = now_s()
    res = move_to_pose(Pose((float(pos[0]), float(pos[1]), float(pos[2])), tuple(quat), BASE),
                       tolerance=tol, max_steps=max_steps)
    rec = {"label": label, "target": [round(float(v), 4) for v in pos], "ok": bool(res.ok)}
    try:
        rec["steps"] = int(res.steps_executed)
    except Exception:
        rec["steps"] = None
    try:
        rec["status"] = str(res.status)
    except Exception:
        rec["status"] = None
    try:
        if res.error is not None:
            rec["error"] = str(res.error)[:300]
    except Exception:
        pass
    rec["t0"] = round(t0, 3)
    rec["t1"] = round(now_s(), 3) if not check_end(res, label) else None
    LOG["moves"].append(rec)
    return res


def gripper(open_it, label):
    if STATE["ended"]:
        return None
    res = open_gripper() if open_it else close_gripper()
    rec = {"label": label, "ok": bool(res.ok)}
    try:
        rec["steps"] = int(res.steps_executed)
    except Exception:
        rec["steps"] = None
    check_end(res, label)
    LOG["moves"].append(rec)
    return res


def hold(n, grip):
    if STATE["ended"] or n <= 0:
        return
    st = get_robot_state()
    joints = [float(v) for v in st.joint_positions["primary"]]
    for k in range(int(n)):
        r = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=joints, gripper_position=float(grip),
            embodiment=st.embodiment)}))
        if check_end(r, "hold"):
            return


def ee_pose():
    p = get_robot_state().end_effector_poses["primary"]
    return [float(v) for v in p.position], tuple(float(v) for v in p.quaternion_wxyz)


def grip_pos():
    try:
        return float(get_robot_state().gripper_positions["primary"])
    except Exception:
        return None


# ----------------------------------------------------------------------------- perception

def cloud_of(cand, camera):
    try:
        pc = mask_to_point_cloud(cand, camera, target_frame=BASE)
    except Exception:
        pc = mask_to_point_cloud(cand.mask, camera, target_frame=BASE)
    return pc.points


def summarize(pts):
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    zs = [float(p[2]) for p in pts]
    return {
        "n": len(xs),
        "x": qtile(xs, 0.5), "y": qtile(ys, 0.5),
        "z10": qtile(zs, 0.1), "z50": qtile(zs, 0.5), "z90": qtile(zs, 0.9), "z95": qtile(zs, 0.95),
        "xspan": qtile(xs, 0.9) - qtile(xs, 0.1), "yspan": qtile(ys, 0.9) - qtile(ys, 0.1),
        "xs": xs, "ys": ys, "zs": zs,
    }


def valid_for(kind, s, table_z):
    if s["n"] < 5:
        return False
    zspan = s["z90"] - s["z10"]
    span = max(s["xspan"], s["yspan"])
    if kind == "target":
        if zspan > 0.025 or span > 0.16 or span < 0.01:
            return False
        if table_z is not None and abs(s["z50"] - table_z) > 0.03:
            return False
        return True
    if kind == "cube":
        if span > 0.09:
            return False
        if table_z is not None and (s["z10"] < table_z - 0.03 or s["z10"] > table_z + 0.08):
            return False
        return True
    if kind == "button":
        if span > 0.14:
            return False
        if table_z is not None and (s["z50"] < table_z - 0.03 or s["z50"] > table_z + 0.12):
            return False
        if table_z is not None and (s["z95"] - table_z < 0.008 or s["z95"] - table_z > 0.08):
            return False
        if math.sqrt(s["x"] ** 2 + s["y"] ** 2) < 0.2:
            return False
        return True
    return True


def greenness(cand, camera):
    # mean (g - max(r, b)) over the candidate mask, sampled inside its box
    try:
        rgb = get_observation(camera_names=[camera]).cameras[camera].rgb
        box = [int(v) for v in cand.box_xyxy]
        mask = cand.mask
    except Exception:
        return None
    x1, y1, x2, y2 = box
    stride = 1
    if (x2 - x1) * (y2 - y1) > 2500:
        stride = 3
    tot = 0.0
    cnt = 0
    for yy in range(max(0, y1), y2 + 1, stride):
        for xx in range(max(0, x1), x2 + 1, stride):
            try:
                if not bool(mask[yy][xx]):
                    continue
                px = rgb[yy][xx]
            except Exception:
                continue
            r, g, b = float(px[0]), float(px[1]), float(px[2])
            tot += g - max(r, b)
            cnt += 1
    if cnt == 0:
        return None
    return tot / cnt


def detect(camera, prompts, kind, table_z=None, max_cands=5):
    tried = []
    for prompt in prompts:
        try:
            seg = segment_text(camera, prompt)
        except Exception as exc:
            tried.append([prompt, "exc", str(exc)[:120]])
            continue
        if not seg.ok:
            tried.append([prompt, "not ok"])
            continue
        best = None
        k = 0
        for cand in seg.segmentations[:max_cands]:
            k += 1
            try:
                pts = cloud_of(cand, camera)
            except Exception:
                continue
            if len(pts) < 5:
                continue
            s = summarize(pts)
            ok = valid_for(kind, s, table_z)
            gr = None
            if ok and kind == "cube":
                gr = greenness(cand, camera)
                if gr is not None and gr < 40.0:
                    ok = False
            # v03: genuine target detections scored 0.04-0.95 (median 0.92) in episodes 1/2;
            # occluded-target fragments in episode 0 scored 0.003-0.009
            if ok and kind == "target" and float(cand.score) < 0.03:
                ok = False
            tried.append([prompt, round(float(cand.score), 3), s["n"], round(s["x"], 3), round(s["y"], 3),
                          round(s["z50"], 3), round(s["xspan"], 3), round(s["yspan"], 3), ok, gr])
            if ok and (best is None or float(cand.score) > best["score"]):
                s["score"] = float(cand.score)
                s["prompt"] = prompt
                best = s
        if best is not None:
            best["tried"] = tried
            return best
    return {"none": True, "tried": tried}


TARGET_PROMPTS = ("purple target", "purple and white bullseye target", "target")
CUBE_PROMPTS = ("green cube", "green block")
BUTTON_PROMPTS = ("button", "white button", "push button")

TRACK = {"obs": [], "table_z": None}
SEG = [0]


def observe_target(tag, moving_phase=True):
    # moving_phase False: the cube may be resting on the target (target possibly frozen);
    # such samples are used for geometry but never for angular-speed estimation.
    if STATE["ended"]:
        return None
    t = now_s()
    s = detect("front", TARGET_PROMPTS, "target", TRACK["table_z"])
    if "none" in s:
        LOG["target_obs"].append([tag, round(t, 3), None])
        return None
    try:
        eep = get_robot_state().end_effector_poses["primary"].position
        dee = math.sqrt((float(eep[0]) - s["x"]) ** 2 + (float(eep[1]) - s["y"]) ** 2)
        eez = float(eep[2])
    except Exception:
        dee = 1.0
        eez = 1.0
    tz = TRACK["table_z"]
    seg = SEG[0] if moving_phase else None
    # samples taken with the gripper low (hover/grasp) or right above the target are biased by
    # arm/cube occlusion in the front view (v00 episode 1: poll samples pulled the angular-speed
    # median from 0.80 to 0.58 rad/s); they drive release timing but not the model fit.
    occl = False
    if tz is not None and (eez < tz + 0.06 or (dee < 0.05 and eez < tz + 0.15)):
        occl = True
    rec = (t, s["x"], s["y"], s["z50"], seg, occl)
    TRACK["obs"].append(rec)
    LOG["target_obs"].append([tag, round(t, 3), round(s["x"], 4), round(s["y"], 4), round(s["z50"], 4), s["n"],
                              round(s["score"], 3), seg, occl])
    return rec


def solve3(m, v):
    # Gaussian elimination for 3x3
    a = [list(m[0]) + [v[0]], list(m[1]) + [v[1]], list(m[2]) + [v[2]]]
    for c in range(3):
        piv = c
        for r in range(c + 1, 3):
            if abs(a[r][c]) > abs(a[piv][c]):
                piv = r
        if abs(a[piv][c]) < 1e-12:
            return None
        a[c], a[piv] = a[piv], a[c]
        for r in range(3):
            if r != c:
                f = a[r][c] / a[c][c]
                for k in range(c, 4):
                    a[r][k] -= f * a[c][k]
    return [a[i][3] / a[i][i] for i in range(3)]


def fit_circle(points):
    if len(points) < 4:
        return None
    sxx = sxy = syy = sx = sy = n = 0.0
    sxz = syz = sz = 0.0
    for p in points:
        x, y = p[0], p[1]
        z = x * x + y * y
        sxx += x * x; sxy += x * y; syy += y * y; sx += x; sy += y; n += 1
        sxz += x * z; syz += y * z; sz += z
    sol = solve3([[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]], [-sxz, -syz, -sz])
    if sol is None:
        return None
    cx, cy = -sol[0] / 2.0, -sol[1] / 2.0
    r2 = cx * cx + cy * cy - sol[2]
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    res = [abs(math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2) - r) for p in points]
    angs = sorted(math.atan2(p[1] - cy, p[0] - cx) for p in points)
    gaps = [angs[i + 1] - angs[i] for i in range(len(angs) - 1)] + [angs[0] + 2 * math.pi - angs[-1]]
    coverage = 2 * math.pi - max(gaps)
    return {"cx": cx, "cy": cy, "r": r, "rms": math.sqrt(sum(e * e for e in res) / len(res)),
            "coverage": coverage}


def solve_n(m, v):
    # Gaussian elimination with partial pivoting, n x n
    n = len(v)
    a = [list(m[i]) + [v[i]] for i in range(n)]
    for c in range(n):
        piv = c
        for r in range(c + 1, n):
            if abs(a[r][c]) > abs(a[piv][c]):
                piv = r
        if abs(a[piv][c]) < 1e-14:
            return None
        a[c], a[piv] = a[piv], a[c]
        for r in range(n):
            if r != c:
                f = a[r][c] / a[c][c]
                for k in range(c, n + 1):
                    a[r][k] -= f * a[c][k]
    return [a[i][n] / a[i][i] for i in range(n)]


def ell_param(e, x, y):
    dx = x - e["cx"]
    dy = y - e["cy"]
    u = (dx * e["ux"] + dy * e["uy"]) / e["a"]
    v = (-dx * e["uy"] + dy * e["ux"]) / e["b"]
    return math.atan2(v, u), math.sqrt(u * u + v * v)


def ell_point(e, phi):
    ca = e["a"] * math.cos(phi)
    sb = e["b"] * math.sin(phi)
    return (e["cx"] + ca * e["ux"] - sb * e["uy"], e["cy"] + ca * e["uy"] + sb * e["ux"])


def fit_ellipse(points):
    # conic A x^2 + B xy + C y^2 + D x + E y + F = 0 with A + C = 1 (linear least squares),
    # in coordinates centred on the point mean for conditioning
    n = len(points)
    if n < 8:
        return None
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    rows = []
    rhs = []
    for p in points:
        x = p[0] - mx
        y = p[1] - my
        rows.append([y * y - x * x, x * y, x, y, 1.0])
        rhs.append(-x * x)
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(5)] for i in range(5)]
    atb = [sum(rows[k][i] * rhs[k] for k in range(n)) for i in range(5)]
    sol = solve_n(ata, atb)
    if sol is None:
        return None
    C, B, D, E, F = sol
    A = 1.0 - C
    if B * B - 4.0 * A * C >= 0:
        return None
    cen = solve_n([[2.0 * A, B], [B, 2.0 * C]], [-D, -E])
    if cen is None:
        return None
    x0, y0 = cen
    Fc = A * x0 * x0 + B * x0 * y0 + C * y0 * y0 + D * x0 + E * y0 + F
    th = 0.5 * math.atan2(B, A - C)
    c = math.cos(th)
    sn = math.sin(th)
    l1 = A * c * c + B * c * sn + C * sn * sn
    l2 = A * sn * sn - B * c * sn + C * c * c
    if l1 * Fc >= 0 or l2 * Fc >= 0:
        return None
    e = {"cx": x0 + mx, "cy": y0 + my, "a": math.sqrt(-Fc / l1), "b": math.sqrt(-Fc / l2),
         "ux": c, "uy": sn}
    res = []
    phis = []
    for p in points:
        ph, rr = ell_param(e, p[0], p[1])
        res.append(abs(rr - 1.0))
        phis.append(ph)
    phis.sort()
    gaps = [phis[i + 1] - phis[i] for i in range(len(phis) - 1)] + [phis[0] + 2 * math.pi - phis[-1]]
    e["coverage"] = 2 * math.pi - max(gaps)
    e["rres"] = qtile(res, 0.5)
    e["n"] = n
    return e


MODEL = {"moving": None, "circle": None, "omega": None, "v": None, "ellipse": None, "e_omega": None}


def update_model():
    obs = TRACK["obs"]
    if len(obs) < 2:
        return
    clean = [o for o in obs if not o[5]]
    # moving?  any same-segment pair displaced > 1.2 cm at > 3 cm/s (decided during the home-pose
    # observation only; v03 locks it afterwards)
    for i in range(1, len(obs)):
        if MODEL.get("locked"):
            break
        a = obs[i - 1]
        b = obs[i]
        if a[4] is None or b[4] is None or a[4] != b[4] or a[5] or b[5]:
            continue
        dt = b[0] - a[0]
        d = math.sqrt((b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)
        if dt > 0.04 and d > 0.012 and d / dt > 0.03:
            MODEL["moving"] = True
    if MODEL["moving"] is None and obs[-1][0] - obs[0][0] > 0.3 and not MODEL.get("locked"):
        MODEL["moving"] = False
    if not MODEL["moving"]:
        return
    pts = []
    for o in clean:
        dup = False
        for q in pts:
            if (o[1] - q[0]) ** 2 + (o[2] - q[1]) ** 2 < 0.004 ** 2:
                dup = True
                break
        if not dup:
            pts.append((o[1], o[2]))
    circ = fit_circle(pts)
    if circ is not None and circ["r"] > 0.02 and circ["r"] < 0.4 and circ["rms"] < 0.015 and circ["coverage"] > 1.0:
        MODEL["circle"] = circ
    circ = MODEL["circle"]
    if circ is not None:
        segs = {}
        for o in clean:
            if o[4] is None:
                continue
            if o[4] not in segs:
                segs[o[4]] = []
            segs[o[4]].append(o)
        rates = []
        for key in segs:
            lst = sorted(segs[key])
            if len(lst) < 2:
                continue
            ths = []
            prev = None
            for o in lst:
                th = math.atan2(o[2] - circ["cy"], o[1] - circ["cx"])
                if prev is not None:
                    th = prev + wrap_pi(th - prev)
                ths.append(th)
                prev = th
            for i in range(len(lst)):
                for k in range(i + 1, len(lst)):
                    dt = lst[k][0] - lst[i][0]
                    if dt >= 0.3 and dt <= 4.0:
                        rates.append((ths[k] - ths[i]) / dt)
        if len(rates) > 0:
            MODEL["omega"] = qtile(rates, 0.5)
            MODEL["omega_n"] = len(rates)
    # ellipse model (v02): the path is an ellipse at uniform eccentric-anomaly rate
    ell = fit_ellipse(pts)
    if ell is not None:
        LOG["events"].append(["ellipse_fit", len(pts), round(ell["a"], 4), round(ell["b"], 4),
                              round(ell["rres"], 3), round(ell["coverage"], 2)])
    if (ell is not None and ell["a"] > 0.02 and ell["b"] > 0.02 and ell["a"] < 0.4 and ell["b"] < 0.4
            and ell["a"] < 5.0 * ell["b"] and ell["b"] < 5.0 * ell["a"]
            and ell["rres"] < 0.1 and ell["coverage"] > 3.5):
        MODEL["ellipse"] = ell
    ell = MODEL["ellipse"]
    if ell is not None:
        segs = {}
        for o in clean:
            if o[4] is None:
                continue
            if o[4] not in segs:
                segs[o[4]] = []
            segs[o[4]].append(o)
        rates = []
        for key in segs:
            lst = sorted(segs[key])
            if len(lst) < 2:
                continue
            ths = []
            prev = None
            for o in lst:
                th = ell_param(ell, o[1], o[2])[0]
                if prev is not None:
                    th = prev + wrap_pi(th - prev)
                ths.append(th)
                prev = th
            for i in range(len(lst)):
                for k in range(i + 1, len(lst)):
                    dt = lst[k][0] - lst[i][0]
                    if dt >= 0.3 and dt <= 4.0:
                        rates.append((ths[k] - ths[i]) / dt)
        if len(rates) > 0:
            MODEL["e_omega"] = qtile(rates, 0.5)
            MODEL["e_omega_n"] = len(rates)
    # linear fallback velocity from the last same-segment clean pair
    for i in range(len(clean) - 1, 0, -1):
        a = clean[i - 1]
        b = clean[i]
        if a[4] is not None and a[4] == b[4] and b[0] - a[0] > 0.04:
            MODEL["v"] = ((b[1] - a[1]) / (b[0] - a[0]), (b[2] - a[2]) / (b[0] - a[0]))
            break


def predict(t_future):
    obs = TRACK["obs"]
    last = obs[-1]
    ell = MODEL["ellipse"]
    if ell is not None and MODEL["e_omega"] is not None:
        om = MODEL["e_omega"]
        # phase offset of the current motion segment from its most recent observations
        # (the target clock pauses while frozen, so older segments carry other offsets)
        recent = [o for o in obs[-4:] if o[4] is not None and o[4] == last[4] and last[0] - o[0] < 1.5]
        if len(recent) == 0:
            recent = [last]
        ref = ell_param(ell, last[1], last[2])[0] - om * last[0]
        sc = 0.0
        ss = 0.0
        for o in recent:
            off = ell_param(ell, o[1], o[2])[0] - om * o[0]
            off = ref + wrap_pi(off - ref)
            sc += math.cos(off)
            ss += math.sin(off)
        off = math.atan2(ss, sc)
        return ell_point(ell, off + om * t_future), "ellipse"
    circ = MODEL["circle"]
    if circ is not None and MODEL["omega"] is not None:
        om = MODEL["omega"]
        ref = last
        th = math.atan2(ref[2] - circ["cy"], ref[1] - circ["cx"]) + om * (t_future - ref[0])
        return (circ["cx"] + circ["r"] * math.cos(th), circ["cy"] + circ["r"] * math.sin(th)), "circle"
    if MODEL["v"] is not None:
        return (last[1] + MODEL["v"][0] * (t_future - last[0]), last[2] + MODEL["v"][1] * (t_future - last[0])), "linear"
    return (last[1], last[2]), "static"


def closest_approach(px, py, horizon_s, within=0.015):
    # first predicted tick at which the target comes within `within` m of (px, py);
    # falls back to the overall closest approach when it never does.
    best_k = 0
    best_d = None
    k = 0
    t0 = now_s()
    while k <= int(horizon_s * HZ):
        pos, how = predict(t0 + k / HZ)
        d = math.sqrt((pos[0] - px) ** 2 + (pos[1] - py) ** 2)
        if d < within:
            return k, d
        if best_d is None or d < best_d:
            best_d = d
            best_k = k
        k += 1
    return best_k, best_d


# ----------------------------------------------------------------------------- task setup

context = get_task_context()
goal = context.language
report = {"goal": goal}
try:
    meta = dict(robomme_plus.get_task_metadata())
    report["metadata"] = {k: str(meta[k])[:200] for k in meta}
except Exception as exc:
    meta = {}
    report["metadata_error"] = str(exc)[:200]
try:
    report["controller"] = {k: str(v)[:200] for k, v in dict(robomme_plus.get_controller_metadata()).items()}
except Exception as exc:
    report["controller_error"] = str(exc)[:200]
try:
    HZ = float(meta.get("control_frequency", meta.get("control_hz", 20.0)))
except Exception:
    HZ = 20.0
try:
    video = robomme_plus.get_conditioning_video()
    report["video_len"] = len(video)
except Exception as exc:
    report["video_error"] = str(exc)[:200]

N = parse_count(goal)
report["N"] = N
if N is None:
    N = 1

pos0, quat0 = ee_pose()
xa, za = qaxes(quat0)
report["home_ee"] = [pos0, list(quat0), list(za)]
if za[2] > -0.9:
    quat0 = (0.0, 1.0, 0.0, 0.0)
    report["quat_override"] = True
QUAT = [quat0]

target0 = observe_target("init")
tz0 = None
if target0 is not None:
    tz0 = target0[3]
cube0 = detect("front", CUBE_PROMPTS, "cube", tz0)
button0 = detect("front", BUTTON_PROMPTS, "button", tz0)
report["init_detect"] = {
    "cube": cube0.get("tried"), "button": button0.get("tried"),
}
if target0 is not None:
    TRACK["table_z"] = target0[3]
    table_z = target0[3]
elif "none" not in cube0:
    table_z = cube0["z10"]
    TRACK["table_z"] = table_z
else:
    table_z = pos0[2] - 0.3
cube_h = 0.04
if "none" not in cube0:
    cube_h = min(0.06, max(0.02, cube0["z95"] - table_z))
report["table_z"] = table_z
report["cube_h"] = cube_h

CARRY_Z = table_z + cube_h * 0.5 + 0.09      # EE z while transporting (cube bottom ~6-7 cm up)
HOVER_Z = table_z + cube_h * 0.5 + 0.018     # EE z for release (cube bottom ~1.8 cm up)
PRE_Z = table_z + cube_h + 0.11              # EE z for observation above the cube

# second target observation for motion detection
hold(8, 1.0)
observe_target("init2")
update_model()
hold(12, 1.0)
observe_target("init3")
update_model()
if MODEL["moving"] is None:
    MODEL["moving"] = False
MODEL["locked"] = True
report["moving_decision"] = [MODEL["moving"], len(TRACK["obs"])]
STATIC_T = [None]


def static_center():
    if STATIC_T[0] is not None:
        return STATIC_T[0]
    clean = [o for o in TRACK["obs"] if not o[5]]
    if len(clean) == 0:
        return None
    STATIC_T[0] = (qtile([o[1] for o in clean], 0.5), qtile([o[2] for o in clean], 0.5))
    return STATIC_T[0]


if not MODEL["moving"]:
    report["static_center"] = static_center()   # frozen from the unoccluded home-pose frames

EST = {"lateral": 25.0, "descend": 12.0}


def cube_from_wrist(prev_xy):
    s = detect("wrist", CUBE_PROMPTS, "cube", table_z)
    if "none" in s:
        return None
    if prev_xy is not None and math.sqrt((s["x"] - prev_xy[0]) ** 2 + (s["y"] - prev_xy[1]) ** 2) > 0.08:
        return None
    # top face points for centre and yaw
    ztop = s["z90"]
    tx = []
    ty = []
    for i in range(len(s["zs"])):
        if s["zs"][i] > ztop - 0.008:
            tx.append(s["xs"][i])
            ty.append(s["ys"][i])
    yaw = None
    if len(tx) >= 20:
        best_area = None
        for step_deg in range(0, 90, 3):
            a = math.radians(step_deg)
            ca, sa = math.cos(a), math.sin(a)
            us = [ca * tx[i] + sa * ty[i] for i in range(len(tx))]
            vs = [-sa * tx[i] + ca * ty[i] for i in range(len(tx))]
            area = (qtile(us, 0.97) - qtile(us, 0.03)) * (qtile(vs, 0.97) - qtile(vs, 0.03))
            if best_area is None or area < best_area:
                best_area = area
                yaw = a
        cx = (qtile(tx, 0.03) + qtile(tx, 0.97)) / 2.0
        cy = (qtile(ty, 0.03) + qtile(ty, 0.97)) / 2.0
    else:
        cx, cy = s["x"], s["y"]
    return {"x": cx, "y": cy, "top": ztop, "yaw": yaw, "n": s["n"], "ntop": len(tx)}


def grasp_cube(cyc, hint_xy):
    info = {}
    xy = None
    top = None
    f = detect("front", CUBE_PROMPTS, "cube", table_z)
    if "none" not in f:
        xy = (f["x"], f["y"])
        top = f["z95"]
        info["front"] = [round(f["x"], 4), round(f["y"], 4), round(f["z95"], 4), f["n"]]
    else:
        info["front"] = f.get("tried")
    if xy is None:
        xy = hint_xy
    if xy is None:
        info["fail"] = "no cube"
        return False, info
    goto((xy[0], xy[1], PRE_Z), QUAT[0], "pre_grasp_%d" % cyc, tol=0.015)
    if cyc == 0 and MODEL["moving"]:
        observe_target("pregrasp_%d" % cyc)
    w = cube_from_wrist(xy)
    q = QUAT[0]
    if w is not None:
        info["wrist"] = [round(w["x"], 4), round(w["y"], 4), round(w["top"], 4), w["yaw"], w["n"], w["ntop"]]
        xy = (w["x"], w["y"])
        top = w["top"]
        if w["yaw"] is not None:
            xaxis, zaxis = qaxes(q)
            gyaw = math.atan2(xaxis[1], xaxis[0])
            delta = wrap_quarter(w["yaw"] - gyaw)
            info["yaw_delta_deg"] = round(math.degrees(delta), 1)
            q = rot_z(q, delta)
    if top is None:
        top = table_z + cube_h
    gz = max(top - cube_h * 0.5, table_z + 0.012)
    info["grasp"] = [round(xy[0], 4), round(xy[1], 4), round(gz, 4)]
    goto((xy[0], xy[1], gz + 0.04), q, "above_grasp_%d" % cyc, tol=0.01)
    goto((xy[0], xy[1], gz), q, "grasp_%d" % cyc, tol=0.008)
    gripper(False, "close_%d" % cyc)
    if cyc == 0 and MODEL["moving"]:
        observe_target("closed_%d" % cyc)
    goto((xy[0], xy[1], CARRY_Z), q, "lift_%d" % cyc, tol=0.015)
    g = grip_pos()
    info["grip_after_lift"] = g
    QUAT[0] = q
    held = g is not None and g > 0.04
    return held, info


def place_cube(cyc):
    info = {"polls": []}
    q = QUAT[0]
    a = observe_target("carry_a_%d" % cyc)
    hold(4, 0.0)
    b = observe_target("carry_b_%d" % cyc)
    update_model()
    info["model"] = {"moving": MODEL["moving"], "circle": MODEL["circle"], "omega": MODEL["omega"], "v": MODEL["v"],
                     "ellipse": MODEL["ellipse"], "e_omega": MODEL["e_omega"]}
    if len(TRACK["obs"]) == 0:
        info["fail"] = "no target"
        return False, info
    if not MODEL["moving"]:
        sc = static_center()
        if sc is None:
            sc = (TRACK["obs"][-1][1], TRACK["obs"][-1][2])
        px, py = sc[0], sc[1]
        info["aim"] = [round(px, 4), round(py, 4), "static"]
        goto((px, py, CARRY_Z), q, "transit_%d" % cyc, tol=0.012)
        goto((px, py, HOVER_Z), q, "hover_%d" % cyc, tol=0.006)
        eep, eeq = ee_pose()
        info["hover_err"] = round(math.sqrt((eep[0] - px) ** 2 + (eep[1] - py) ** 2), 4)
        gripper(True, "release_%d" % cyc)
        SEG[0] += 1
        return True, info
    # moving target: intercept
    lead = (EST["lateral"] + EST["descend"] + 8.0) / HZ
    pt, how = predict(now_s() + lead)
    info["aim"] = [round(pt[0], 4), round(pt[1], 4), how, round(lead, 2)]
    t0 = now_s()
    goto((pt[0], pt[1], CARRY_Z), q, "transit_%d" % cyc, tol=0.015)
    t1 = now_s()
    goto((pt[0], pt[1], HOVER_Z), q, "hover_%d" % cyc, tol=0.01)
    t2 = now_s()
    if t1 > t0:
        EST["lateral"] = 0.5 * EST["lateral"] + 0.5 * (t1 - t0) * HZ
    if t2 > t1:
        EST["descend"] = 0.5 * EST["descend"] + 0.5 * (t2 - t1) * HZ
    released = False
    replans = 0
    for poll in range(14):
        if STATE["ended"]:
            break
        o = observe_target("poll_%d_%d" % (cyc, poll))
        update_model()
        eep, eeq = ee_pose()
        if o is None:
            hold(2, 0.0)
            info["polls"].append([poll, "noobs"])
            continue
        dnow = math.sqrt((o[1] - eep[0]) ** 2 + (o[2] - eep[1]) ** 2)
        k, dmin = closest_approach(eep[0], eep[1], 9.0)
        info["polls"].append([poll, round(now_s(), 3), round(dnow, 4), k, round(dmin, 4)])
        # v02 guard: successful v01 releases had measured dnow <= 0.031 m; the misses 0.047 / 0.086
        if dnow < 0.012 or (dmin < 0.015 and k <= 2 and dnow < 0.042):
            gripper(True, "release_%d" % cyc)
            released = True
            SEG[0] += 1
            break
        if dmin > 0.025 or k > 60 or (k <= 2 and dnow >= 0.042 and replans < 3):
            if replans >= 3:
                # give up waiting: put it where the target is predicted soonest
                pass
            replans += 1
            lead2 = (EST["lateral"] * 0.7 + 6.0) / HZ
            pt, how = predict(now_s() + lead2)
            info["polls"].append([poll, "replan", round(pt[0], 4), round(pt[1], 4), how])
            goto((pt[0], pt[1], HOVER_Z), q, "rehover_%d_%d" % (cyc, poll), tol=0.01)
            continue
        hold(min(max(k - 2, 1), 4), 0.0)
    if not released and not STATE["ended"]:
        o = TRACK["obs"][-1]
        goto((o[1], o[2], HOVER_Z), q, "fallback_hover_%d" % cyc, tol=0.01)
        gripper(True, "release_fallback_%d" % cyc)
        SEG[0] += 1
        info["fallback"] = True
    return True, info


def verify_place(cyc, held_xy):
    info = {}
    eep, eeq = ee_pose()
    goto((eep[0], eep[1], PRE_Z), QUAT[0], "retreat_%d" % cyc, tol=0.015)
    a = observe_target("verify_a_%d" % cyc, False)
    hold(5, 1.0)
    b = observe_target("verify_b_%d" % cyc, False)
    c = detect("front", CUBE_PROMPTS, "cube", table_z)
    cube_xy = None
    if "none" not in c:
        cube_xy = (c["x"], c["y"])
    w = cube_from_wrist(None)
    if w is not None:
        cube_xy = (w["x"], w["y"])
        info["cube_wrist"] = [round(w["x"], 4), round(w["y"], 4)]
    info["cube_xy"] = cube_xy
    if not MODEL["moving"]:
        sc = static_center()
        if cube_xy is None or sc is None:
            info["verdict"] = "unknown"
            return True, cube_xy, info
        d = math.sqrt((sc[0] - cube_xy[0]) ** 2 + (sc[1] - cube_xy[1]) ** 2)
        info["cube_static_target_dist"] = round(d, 4)
        info["verdict"] = d < 0.025
        return d < 0.025, cube_xy, info
    # moving target: it FREEZES while the cube rests on it (v00 evidence, episode 1)
    if a is None or b is None:
        info["verdict"] = "unknown"
        return True, cube_xy, info
    frozen = math.sqrt((b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2) < 0.01
    info["frozen"] = frozen
    if cube_xy is not None:
        d = math.sqrt((b[1] - cube_xy[0]) ** 2 + (b[2] - cube_xy[1]) ** 2)
        info["cube_target_dist"] = round(d, 4)
        if d > 0.07:
            frozen = False
    info["verdict"] = frozen
    return frozen, cube_xy, info


def late_freeze_check(cube_xy):
    # after a missed release the target keeps moving and may pass under the resting cube;
    # if it is now frozen next to the cube the simulator counted it.
    a = observe_target("late_a", False)
    hold(4, 1.0)
    b = observe_target("late_b", False)
    if a is None or b is None or cube_xy is None:
        return False
    frozen = math.sqrt((b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2) < 0.01
    d = math.sqrt((b[1] - cube_xy[0]) ** 2 + (b[2] - cube_xy[1]) ** 2)
    LOG["events"].append(["late_freeze", frozen, round(d, 4)])
    return frozen and d < 0.07


# ----------------------------------------------------------------------------- main loop

def main_task():
    placed_count = 0
    attempts = 0
    hint = None
    if "none" not in cube0:
        hint = (cube0["x"], cube0["y"])
    # v02: observe a moving target from the home pose (unoccluded) until the ellipse model is
    # valid, so that the first placements do not rely on the circle fallback (v01/ep2 replay:
    # circle error ~7 cm at 1-2.5 s lead, ellipse ~0.3 cm).
    if MODEL["moving"]:
        t_scan = now_s()
        for i in range(30):
            if STATE["ended"] or now_s() - t_scan > 10.0:
                break
            if MODEL["ellipse"] is not None and MODEL["e_omega"] is not None and MODEL.get("e_omega_n", 0) >= 6:
                break
            hold(6, 1.0)
            observe_target("scan_%d" % i)
            update_model()
        report["scan"] = {"s": round(now_s() - t_scan, 2), "ellipse": MODEL["ellipse"] is not None,
                          "e_omega": MODEL["e_omega"]}
    while placed_count < N and attempts < N + 3 and not STATE["ended"]:
        cyc = attempts
        attempts += 1
        rec = {"cycle": cyc, "t_start": round(now_s(), 3)}
        held, ginfo = grasp_cube(cyc, hint)
        rec["grasp"] = ginfo
        rec["held"] = held
        if not held:
            gripper(True, "regrasp_open_%d" % cyc)
            eep, eeq = ee_pose()
            goto((eep[0], eep[1], PRE_Z), QUAT[0], "regrasp_up_%d" % cyc, tol=0.02)
            LOG["cycles"].append(rec)
            continue
        ok, pinfo = place_cube(cyc)
        rec["place"] = pinfo
        placed, cube_xy, vinfo = verify_place(cyc, None)
        rec["verify"] = vinfo
        if cube_xy is not None:
            hint = cube_xy
        if not placed and MODEL["moving"] and not STATE["ended"]:
            if late_freeze_check(cube_xy):
                placed = True
                rec["late_freeze"] = True
        if placed:
            placed_count += 1
        rec["placed_count"] = placed_count
        rec["t_end"] = round(now_s(), 3)
        LOG["cycles"].append(rec)

    report["placed_count"] = placed_count
    report["attempts"] = attempts

    # ----------------------------------------------------------------------------- button
    if not STATE["ended"] and placed_count >= N:
        btn = button0
        if "none" in btn:
            btn = detect("front", BUTTON_PROMPTS, "button", table_z)
        report["button_pick"] = [btn.get("prompt"), btn.get("score")]
        report["button"] = btn.get("tried")
        if "none" not in btn:
            bx, by = btn["x"], btn["y"]
            btop = btn["z95"]
            report["button_xyz"] = [bx, by, btop]
            eep, eeq = ee_pose()
            goto((eep[0], eep[1], max(eep[2], btop + 0.10)), QUAT[0], "button_up", tol=0.02)
            goto((bx, by, btop + 0.08), QUAT[0], "button_above", tol=0.01)
            gripper(False, "button_close")
            for depth in (0.015, 0.03):
                if STATE["ended"]:
                    break
                goto((bx, by, btop - depth), QUAT[0], "button_press_%d" % int(depth * 1000), tol=0.005, max_steps=60)
                if STATE["ended"]:
                    break
                goto((bx, by, btop + 0.05), QUAT[0], "button_release_%d" % int(depth * 1000), tol=0.01, max_steps=60)
                if not STATE["ended"]:
                    hold(3, 0.0)


try:
    main_task()
except Exception as exc:
    report["program_exception"] = str(exc)[:500]

report["ended"] = STATE["ended"]
report["end_reason"] = STATE["end_reason"]
report["final_t"] = now_s() if not STATE["ended"] else None
report["model"] = {"moving": MODEL["moving"], "circle": MODEL["circle"], "omega": MODEL["omega"],
                   "ellipse": MODEL["ellipse"], "e_omega": MODEL["e_omega"]}
report["log"] = LOG
result = report
