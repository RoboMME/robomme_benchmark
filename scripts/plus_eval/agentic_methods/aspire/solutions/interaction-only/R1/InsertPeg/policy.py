# InsertPeg v01_staged_insert_search -- UNTESTED trace-guided repair of v00 (R=1, interaction-only).
#
# v00 run (episode 0): replay_full reproduced grasp/transport/approach of the conditioning video
# to < 1 mm, then the peg jammed at the hole entrance (EE stalled 3.4 cm short of the demonstrated
# insertion end while the joint target ran ahead). v01 changes ONLY the insertion phase:
#   - replay stops a few frames before the demonstrated slow-down (contact onset) on the final
#     straight approach (k_pre, derived from the video EE path, no coordinates hard-coded)
#   - staged Cartesian insertion along the demonstrated direction via solve_ik (delta-joint
#     method), commanded lead capped at 5 mm, stall detection, back-off and a perpendicular
#     offset search (+-1.5 / +-3 mm) ordered by the measured contact deflection
#   - clause bookkeeping fix (combined grasp+insert clause; insert executed = depth reached)
#   - the v00 end-of-run push (never fired) is replaced by the staged insertion
#
# Phase 0  context, task/controller metadata, cameras, robot state
# Phase 1  SAM3 prompt probes on both cameras (recorded only)
# Phase 2  live peg / box localisation with geometric bounds, camera-convention self-calibration
#          (projection residual against live depth), table height, peg end colours
# Phase 3  conditioning-video summary: timestamps, joints, EE path, gripper events, image stats
# Phase 4  video-vs-live scene comparison (depth agreement at the live peg/box points) and the
#          video peg pose near the demonstrated grasp
# Phase 5  attempt:
#          replay_full        - live peg (and box) where the video had them: replay the
#                               demonstrated joint path + gripper events from the episode's own
#                               conditioning video, one control tick per recorded tick
#          transformed_grasp  - peg moved: grasp at the demonstrated peg-relative grasp, carried
#                               through the observed peg pose change, then replay post-lift path
#          perception_only    - no demonstrated grasp in the video: grasp + side insertion from
#                               live geometry
# Phase 6  bounded hold until a terminal status, clause self-check, report.
# Never raises: every phase is wrapped; errors are appended to result["errors"].

import math

report = {
    "program": "InsertPeg v01_staged_insert_search",
    "phase_log": [],
    "errors": [],
    "clauses": [],
    "decision": {},
    "terminal": False,
}
result = report

STATE = {
    "ticks": 0,
    "terminal": False,
    "stop": False,
    "max_steps": 0,
    "hz": 20.0,
    "step_errors": 0,
    "first_step_repr": None,
    "last_target": None,
    "last_grip": 1.0,
    "emb": "robomme_plus",
}


def note(msg):
    text = str(msg)[:400]
    print("[InsertPeg] " + text)
    report["phase_log"].append(text)


def fail(where, exc):
    text = where + ": " + str(exc)[:500]
    print("[InsertPeg][ERR] " + text)
    report["errors"].append(text)


def fl(x, nd=4):
    return round(float(x), nd)


def vec(a):
    return [float(v) for v in a]


def rvec(a, nd=4):
    return [round(float(v), nd) for v in a]


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def rows2d(arr):
    try:
        return arr.tolist()
    except Exception:
        pass
    return [[float(v) for v in row] for row in arr]


def rows3d(arr):
    try:
        return arr.tolist()
    except Exception:
        pass
    return [[[int(c) for c in px] for px in row] for row in arr]


def points_of(pc):
    try:
        raw = pc.points.tolist()
        return [[float(p[0]), float(p[1]), float(p[2])] for p in raw]
    except Exception:
        pass
    return [[float(p[0]), float(p[1]), float(p[2])] for p in pc.points]


def valid_depth(d):
    try:
        d = float(d)
    except Exception:
        return False
    return d == d and d > 0.01 and d < 20.0


def qmat(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        n = 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def qz(angle):
    return [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]


def mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def wrap_half(a):
    # wrap to [-pi/2, pi/2] (symmetric gripper / undirected axis)
    a = wrap_pi(a)
    if a > math.pi / 2:
        a -= math.pi
    if a < -math.pi / 2:
        a += math.pi
    return a


def color_dist(a, b):
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def is_terminal(res):
    t = False
    try:
        if res.terminated:
            t = True
    except Exception:
        pass
    try:
        if res.truncated:
            t = True
    except Exception:
        pass
    return t


# ---------------------------------------------------------------- camera geometry
CONV = {
    "cv": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "gl": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    "sapien": [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
}


def cam_model(cam, conv, ray):
    try:
        k = cam.intrinsics.tolist()
    except Exception:
        k = [[float(v) for v in row] for row in cam.intrinsics]
    pose = cam.camera_pose
    rot = qmat(vec(pose.quaternion_wxyz))
    rm = mat_mul(rot, CONV[conv])
    return {
        "fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]), "cy": float(k[1][2]),
        "rm": rm, "t": vec(pose.position), "conv": conv, "ray": ray,
    }


def backproject(m, u, v, d):
    xn = (u - m["cx"]) / m["fx"]
    yn = (v - m["cy"]) / m["fy"]
    if m["ray"]:
        s = d / math.sqrt(1.0 + xn * xn + yn * yn)
    else:
        s = d
    p = (xn * s, yn * s, s)
    rm = m["rm"]
    t = m["t"]
    return [
        rm[0][0] * p[0] + rm[0][1] * p[1] + rm[0][2] * p[2] + t[0],
        rm[1][0] * p[0] + rm[1][1] * p[1] + rm[1][2] * p[2] + t[1],
        rm[2][0] * p[0] + rm[2][1] * p[1] + rm[2][2] * p[2] + t[2],
    ]


def project(m, w):
    rm = m["rm"]
    t = m["t"]
    dx = [w[0] - t[0], w[1] - t[1], w[2] - t[2]]
    p0 = rm[0][0] * dx[0] + rm[1][0] * dx[1] + rm[2][0] * dx[2]
    p1 = rm[0][1] * dx[0] + rm[1][1] * dx[1] + rm[2][1] * dx[2]
    p2 = rm[0][2] * dx[0] + rm[1][2] * dx[1] + rm[2][2] * dx[2]
    if p2 <= 1e-6:
        return None
    u = m["fx"] * p0 / p2 + m["cx"]
    v = m["fy"] * p1 / p2 + m["cy"]
    rng = math.sqrt(p0 * p0 + p1 * p1 + p2 * p2)
    return (u, v, p2, rng)


def depth_agreement(m, depth_rows, pts, tol):
    # fraction of points whose projected pixel in this camera has depth within tol
    h = len(depth_rows)
    w = len(depth_rows[0]) if h > 0 else 0
    good = 0
    seen = 0
    residuals = []
    for p in pts:
        pr = project(m, p)
        if pr is None:
            continue
        ui = int(pr[0] + 0.5)
        vi = int(pr[1] + 0.5)
        if ui < 0 or vi < 0 or ui >= w or vi >= h:
            continue
        d = depth_rows[vi][ui]
        if not valid_depth(d):
            continue
        expect = pr[3] if m["ray"] else pr[2]
        seen += 1
        r = abs(float(d) - expect)
        residuals.append(r)
        if r <= tol:
            good += 1
    if seen == 0:
        return {"seen": 0, "frac": 0.0, "median": None}
    return {"seen": seen, "frac": round(good / float(seen), 3), "median": fl(quantile(residuals, 0.5), 4)}


def sample_list(items, cap):
    if len(items) <= cap:
        return list(items)
    stride = max(1, len(items) // cap)
    return [items[i] for i in range(0, len(items), stride)]


def cloud_geom(pts, table_z):
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts) / n
    syy = sum((p[1] - my) ** 2 for p in pts) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts) / n
    ang = 0.5 * math.atan2(2 * sxy, sxx - syy)
    ax = (math.cos(ang), math.sin(ang))
    ay = (-ax[1], ax[0])
    ss = [(p[0] - mx) * ax[0] + (p[1] - my) * ax[1] for p in pts]
    tt = [(p[0] - mx) * ay[0] + (p[1] - my) * ay[1] for p in pts]
    zs = [p[2] for p in pts]
    s_lo, s_hi = quantile(ss, 0.02), quantile(ss, 0.98)
    t_lo, t_hi = quantile(tt, 0.02), quantile(tt, 0.98)
    smid = 0.5 * (s_lo + s_hi)
    tmid = 0.5 * (t_lo + t_hi)
    cx = mx + smid * ax[0] + tmid * ay[0]
    cy = my + smid * ax[1] + tmid * ay[1]
    zlo, zhi = quantile(zs, 0.02), quantile(zs, 0.98)
    g = {
        "n": n,
        "center": [cx, cy, 0.5 * (zlo + zhi)],
        "axis_angle": ang,
        "length": s_hi - s_lo,
        "width": t_hi - t_lo,
        "zlo": zlo,
        "zhi": zhi,
        "height_above_table": None if table_z is None else zhi - table_z,
        "bottom_above_table": None if table_z is None else zlo - table_z,
    }
    return g


def geom_summary(g):
    return {
        "n": g["n"],
        "center": rvec(g["center"]),
        "axis_deg": fl(math.degrees(g["axis_angle"]), 1),
        "length": fl(g["length"]),
        "width": fl(g["width"]),
        "zlo": fl(g["zlo"]),
        "zhi": fl(g["zhi"]),
        "h_above_table": None if g["height_above_table"] is None else fl(g["height_above_table"]),
        "bottom_above_table": None if g["bottom_above_table"] is None else fl(g["bottom_above_table"]),
    }


# ---------------------------------------------------------------- control helpers
def tick(target, grip):
    if STATE["terminal"] or STATE["stop"]:
        return False
    if STATE["max_steps"] and STATE["ticks"] >= STATE["max_steps"]:
        STATE["stop"] = True
        note("tick budget reached " + str(STATE["ticks"]))
        return False
    try:
        r = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position",
            target=[float(v) for v in target],
            gripper_position=float(grip),
            embodiment=STATE["emb"])}))
    except Exception as exc:
        fail("step", exc)
        STATE["step_errors"] += 1
        if STATE["step_errors"] >= 3:
            STATE["stop"] = True
        return False
    STATE["ticks"] += 1
    STATE["last_target"] = [float(v) for v in target]
    STATE["last_grip"] = float(grip)
    if STATE["first_step_repr"] is None:
        STATE["first_step_repr"] = str(r)[:800]
        report["first_step_repr"] = STATE["first_step_repr"]
    if is_terminal(r):
        STATE["terminal"] = True
        report["terminal"] = True
        report["terminal_at_tick"] = STATE["ticks"]
        note("terminal status at tick " + str(STATE["ticks"]))
        return False
    try:
        e = r.error
        if e is not None and str(e) not in ("", "None"):
            fail("step.error", e)
            STATE["step_errors"] += 1
            if STATE["step_errors"] >= 3:
                STATE["stop"] = True
            return False
    except Exception:
        pass
    return True


def after_motion(name, res):
    entry = {"call": name}
    try:
        entry["ok"] = bool(res.ok)
    except Exception:
        entry["ok"] = None
    try:
        entry["status"] = str(res.status)[:80]
    except Exception:
        pass
    try:
        n = int(res.steps_executed)
        entry["steps"] = n
        STATE["ticks"] += n
    except Exception:
        pass
    try:
        entry["error"] = str(res.error)[:200]
    except Exception:
        pass
    if is_terminal(res):
        STATE["terminal"] = True
        report["terminal"] = True
        report["terminal_at_tick"] = STATE["ticks"]
        entry["terminal"] = True
        note("terminal status after " + name)
    report.setdefault("motions", []).append(entry)
    return entry.get("ok") is True


def current_joints():
    st = get_robot_state()
    return vec(st.joint_positions["primary"])


def current_grip():
    st = get_robot_state()
    return float(st.gripper_positions["primary"])


def hold(ticks, grip):
    tgt = STATE["last_target"]
    if tgt is None:
        tgt = current_joints()
    done = 0
    for i in range(ticks):
        if not tick(tgt, grip):
            break
        done += 1
    return done


def interp_to(target, grip, max_rate):
    # joint-space interpolation by single ticks, at most max_rate rad per tick
    start = current_joints()
    diff = max(abs(target[i] - start[i]) for i in range(7))
    n = max(2, int(math.ceil(diff / max_rate)))
    for j in range(1, n + 1):
        a = j / float(n)
        if not tick([start[i] + a * (target[i] - start[i]) for i in range(7)], grip):
            return False
    return True


# ================================================================ Phase 0
goal = ""
base_frame = "robot_base"
try:
    ctx = get_task_context()
    goal = str(ctx.language)
    report["language"] = goal
    try:
        report["task_name"] = str(ctx.task_name)
        report["task_id"] = str(ctx.task_id)
        report["suite"] = str(ctx.suite)
    except Exception as exc:
        fail("context.fields", exc)
    note("goal: " + goal)
except Exception as exc:
    fail("context", exc)

md = None
try:
    md = robomme_plus.get_task_metadata()
    mdd = {}
    for key in md:
        mdd[str(key)] = str(md[key])[:400]
    report["task_metadata"] = mdd
    try:
        STATE["max_steps"] = int(md["max_steps"])
    except Exception as exc:
        fail("max_steps", exc)
    for key in ("control_hz", "control_frequency", "control_freq", "frequency_hz", "hz"):
        try:
            STATE["hz"] = float(md[key])
            break
        except Exception:
            pass
except Exception as exc:
    fail("task_metadata", exc)

try:
    cmd_md = robomme_plus.get_controller_metadata()
    cdd = {}
    for key in cmd_md:
        cdd[str(key)] = str(cmd_md[key])[:400]
    report["controller_metadata"] = cdd
except Exception as exc:
    fail("controller_metadata", exc)

report["max_steps"] = STATE["max_steps"]
report["hz"] = STATE["hz"]

live_obs = None
try:
    state0 = get_robot_state()
    base_frame = str(state0.base_frame)
    STATE["emb"] = str(state0.embodiment)
    ee0 = state0.end_effector_poses["primary"]
    report["robot_state0"] = {
        "base_frame": base_frame,
        "embodiment": STATE["emb"],
        "joints": rvec(state0.joint_positions["primary"]),
        "gripper": fl(state0.gripper_positions["primary"], 3),
        "ee_pos": rvec(ee0.position),
        "ee_quat": rvec(ee0.quaternion_wxyz),
        "timestamp_s": fl(state0.timestamp_s, 3),
    }
except Exception as exc:
    fail("robot_state0", exc)

cams_report = {}
try:
    live_obs = get_observation()
    for name in live_obs.cameras:
        cam = live_obs.cameras[name]
        info = {}
        try:
            info["rgb_shape"] = [int(v) for v in cam.rgb.shape]
            info["depth_shape"] = [int(v) for v in cam.depth_m.shape]
        except Exception as exc:
            info["shape_err"] = str(exc)[:100]
        try:
            info["intrinsics"] = [[fl(v, 3) for v in row] for row in rows2d(cam.intrinsics)]
        except Exception as exc:
            info["intr_err"] = str(exc)[:100]
        try:
            info["pose_pos"] = rvec(cam.camera_pose.position)
            info["pose_quat"] = rvec(cam.camera_pose.quaternion_wxyz)
            info["pose_frame"] = str(cam.camera_pose.frame)
        except Exception as exc:
            info["pose_err"] = str(exc)[:100]
        cams_report[str(name)] = info
    report["cameras"] = cams_report
except Exception as exc:
    fail("observation", exc)

cam_names = list(cams_report.keys())
FRONT = "front" if "front" in cam_names else (cam_names[0] if cam_names else "front")
WRIST = "wrist" if "wrist" in cam_names else (cam_names[-1] if cam_names else "wrist")

# clauses of the goal (recorded; executed flags updated as phases run)
try:
    text = goal.lower().replace(";", ",").replace(" then ", ",").replace(" and then ", ",")
    text = text.replace(" finally ", ",").replace(". ", ",")
    for part in text.split(","):
        part = part.strip(" .")
        if len(part) == 0:
            continue
        kinds = []
        if "button" in part:
            kinds.append("button")
        if "grasp" in part or "pick" in part or "grab" in part or "hold" in part:
            kinds.append("grasp")
        if "insert" in part or "hole" in part or "slot" in part:
            kinds.append("insert")
        if len(kinds) == 0:
            kinds.append("other")
        report["clauses"].append({"text": part, "kind": "+".join(kinds), "kinds": kinds,
                                  "executed": False, "done": []})
except Exception as exc:
    fail("clauses", exc)


def mark_clause(kind):
    # a clause is executed once every action kind it names has been executed
    for c in report["clauses"]:
        if kind in c["kinds"]:
            if kind not in c["done"]:
                c["done"].append(kind)
            c["executed"] = all(k in c["done"] for k in c["kinds"])


# ================================================================ Phase 1: SAM probes
PROBES_FRONT = ["peg", "box with a hole", "box", "hole", "stick", "rod", "block"]
PROBES_WRIST = ["peg", "box with a hole", "hole"]
probe_log = []
seg_cache = {}
for cam_name, prompts in ((FRONT, PROBES_FRONT), (WRIST, PROBES_WRIST)):
    for prompt in prompts:
        entry = {"camera": cam_name, "prompt": prompt, "ok": False, "n": 0, "scores": [], "boxes": []}
        try:
            seg = segment_text(cam_name, prompt)
            if seg.ok:
                entry["ok"] = True
                cands = list(seg.segmentations)
                entry["n"] = len(cands)
                for cand in cands[:5]:
                    entry["scores"].append(fl(cand.score, 3))
                    try:
                        entry["boxes"].append([fl(v, 1) for v in cand.box_xyxy])
                    except Exception:
                        entry["boxes"].append(None)
                seg_cache[cam_name + "|" + prompt] = cands
        except Exception as exc:
            entry["err"] = str(exc)[:200]
        probe_log.append(entry)
report["prompt_probes"] = probe_log
note("probes done: " + str(len(probe_log)))

# ================================================================ Phase 2: live geometry
live_depth = None
live_rgb = None
try:
    live_depth = rows2d(live_obs.cameras[FRONT].depth_m)
    live_rgb = rows3d(live_obs.cameras[FRONT].rgb)
except Exception as exc:
    fail("live images", exc)

candidates = []   # (record, pts)
for prompt in ("peg", "stick", "rod", "box with a hole", "box", "block"):
    cands = seg_cache.get(FRONT + "|" + prompt, [])
    for idx in range(min(3, len(cands))):
        cand = cands[idx]
        rec = {"prompt": prompt, "rank": idx, "score": fl(cand.score, 3)}
        try:
            rec["box"] = [fl(v, 1) for v in cand.box_xyxy]
        except Exception:
            pass
        try:
            pc = mask_to_point_cloud(cand, FRONT, target_frame=base_frame)
            pts = points_of(pc)
            rec["npts"] = len(pts)
            if len(pts) >= 15:
                candidates.append((rec, pts))
        except Exception as exc:
            rec["err"] = str(exc)[:150]
        report.setdefault("candidates_raw", []).append(rec)

# camera convention self-calibration on the live front camera
calib = None
try:
    calib_pts = []
    for rec, pts in candidates:
        calib_pts.extend(sample_list(pts, 120))
    calib_pts = sample_list(calib_pts, 400)
    best = None
    trials = []
    if live_depth is not None and len(calib_pts) >= 20:
        for conv in ("cv", "gl", "sapien"):
            for ray in (False, True):
                m = cam_model(live_obs.cameras[FRONT], conv, ray)
                agr = depth_agreement(m, live_depth, calib_pts, 0.006)
                trials.append({"conv": conv, "ray": ray, "seen": agr["seen"], "frac": agr["frac"],
                               "median": agr["median"]})
                if agr["seen"] >= 0.5 * len(calib_pts) and agr["median"] is not None:
                    if best is None or agr["median"] < best[0]:
                        best = (agr["median"], conv, ray, agr["frac"])
    report["calibration_trials"] = trials
    if best is not None and best[0] < 0.01:
        calib = {"conv": best[1], "ray": best[2]}
    report["calibration"] = None if best is None else {
        "conv": best[1], "ray": best[2], "median": best[0], "frac": best[3], "accepted": calib is not None}
except Exception as exc:
    fail("calibration", exc)

# table height from the live front depth (mode of back-projected z)
table_z = None
try:
    if calib is not None:
        m = cam_model(live_obs.cameras[FRONT], calib["conv"], calib["ray"])
        hist = {}
        h = len(live_depth)
        w = len(live_depth[0])
        for v in range(0, h, 4):
            for u in range(0, w, 4):
                d = live_depth[v][u]
                if not valid_depth(d):
                    continue
                p = backproject(m, u, v, float(d))
                if p[2] < -0.5 or p[2] > 0.6 or abs(p[0]) > 1.5 or abs(p[1]) > 1.5:
                    continue
                b = int(math.floor(p[2] / 0.004))
                hist[b] = hist.get(b, 0) + 1
        bestb = None
        for b in hist:
            if bestb is None or hist[b] > hist[bestb]:
                bestb = b
        if bestb is not None:
            table_z = (bestb + 0.5) * 0.004
            report["table_z_support"] = hist[bestb]
except Exception as exc:
    fail("table_z", exc)
if table_z is None:
    try:
        lows = [quantile([p[2] for p in pts], 0.02) for rec, pts in candidates]
        if lows:
            table_z = min(lows)
            report["table_z_source"] = "min candidate q02"
    except Exception as exc:
        fail("table_z fallback", exc)
report["table_z"] = None if table_z is None else fl(table_z)

peg = None
box = None
try:
    peg_rank = []
    box_rank = []
    for rec, pts in candidates:
        g = cloud_geom(pts, table_z)
        rec["geom"] = geom_summary(g)
        L, W = g["length"], g["width"]
        H = g["height_above_table"]
        B = g["bottom_above_table"]
        is_peg = (0.06 <= L <= 0.45 and W <= 0.09 and L >= 1.8 * max(W, 1e-3)
                  and (H is None or H <= 0.10) and (B is None or B <= 0.04))
        rec["peg_like"] = bool(is_peg)
        if is_peg and rec["prompt"] in ("peg", "stick", "rod"):
            peg_rank.append((rec["prompt"] == "peg", float(rec["score"]), rec, pts, g))
        is_box = (0.04 <= max(L, W) <= 0.6 and (H is None or H >= 0.03) and not is_peg)
        rec["box_like"] = bool(is_box)
        if is_box and rec["prompt"] in ("box with a hole", "box", "block"):
            box_rank.append((rec["prompt"] != "block", float(rec["score"]), rec, pts, g))
    best = None
    for item in peg_rank:
        if best is None or (item[0], item[1]) > (best[0], best[1]):
            best = item
    if best is None:
        # relaxed fallback: elongated, low object from a peg-class prompt (table height ignored)
        for rec, pts in candidates:
            gg = cloud_geom(pts, table_z)
            if (rec["prompt"] in ("peg", "stick", "rod") and 0.05 <= gg["length"] <= 0.5
                    and gg["length"] >= 1.5 * max(gg["width"], 1e-3)):
                if best is None or float(rec["score"]) > best[1]:
                    best = (False, float(rec["score"]), rec, pts, gg)
        if best is not None:
            report["peg_relaxed_fallback"] = True
    if best is not None:
        peg = {"rec": best[2], "pts": best[3], "g": best[4]}
    best = None
    for item in box_rank:
        if best is None or (item[0], item[1]) > (best[0], best[1]):
            best = item
    if best is not None:
        box = {"rec": best[2], "pts": best[3], "g": best[4]}
    report["candidates"] = [rec for rec, pts in candidates]
    report["peg"] = None if peg is None else {"prompt": peg["rec"]["prompt"], "rank": peg["rec"]["rank"],
                                                "score": peg["rec"]["score"], "geom": peg["rec"]["geom"]}
    report["box"] = None if box is None else {"prompt": box["rec"]["prompt"], "rank": box["rec"]["rank"],
                                                "score": box["rec"]["score"], "geom": box["rec"]["geom"]}
    note("peg " + str(report["peg"]))
    note("box " + str(report["box"]))
except Exception as exc:
    fail("peg/box selection", exc)

# peg end colours (sampled from the live front RGB at projected cloud points)
peg_colors = None
try:
    if peg is not None and calib is not None and live_rgb is not None:
        m = cam_model(live_obs.cameras[FRONT], calib["conv"], calib["ray"])
        g = peg["g"]
        ax = (math.cos(g["axis_angle"]), math.sin(g["axis_angle"]))
        c = g["center"]
        plus = []
        minus = []
        allc = []
        h = len(live_rgb)
        w = len(live_rgb[0])
        for p in sample_list(peg["pts"], 600):
            pr = project(m, p)
            if pr is None:
                continue
            ui = int(pr[0] + 0.5)
            vi = int(pr[1] + 0.5)
            if ui < 0 or vi < 0 or ui >= w or vi >= h:
                continue
            col = live_rgb[vi][ui]
            s = (p[0] - c[0]) * ax[0] + (p[1] - c[1]) * ax[1]
            allc.append(col)
            if s > g["length"] / 6.0:
                plus.append(col)
            elif s < -g["length"] / 6.0:
                minus.append(col)

        def mean_col(cols):
            if len(cols) == 0:
                return None
            return [sum(float(cc[i]) for cc in cols) / len(cols) for i in range(3)]

        peg_colors = {"plus": mean_col(plus), "minus": mean_col(minus), "all": mean_col(allc),
                      "n_plus": len(plus), "n_minus": len(minus)}
        if peg_colors["plus"] is not None and peg_colors["minus"] is not None:
            peg_colors["end_contrast"] = color_dist(peg_colors["plus"], peg_colors["minus"])
        report["peg_colors"] = {
            "plus_end_rgb": None if peg_colors["plus"] is None else rvec(peg_colors["plus"], 1),
            "minus_end_rgb": None if peg_colors["minus"] is None else rvec(peg_colors["minus"], 1),
            "all_rgb": None if peg_colors["all"] is None else rvec(peg_colors["all"], 1),
            "end_contrast": None if "end_contrast" not in peg_colors else fl(peg_colors["end_contrast"], 1),
            "n_plus": len(plus), "n_minus": len(minus),
            "plus_axis_deg": fl(math.degrees(g["axis_angle"]), 1),
        }
except Exception as exc:
    fail("peg colours", exc)

# ================================================================ Phase 3: conditioning video
frames = []
TS = []
Q = []
G = []
P = []
QU = []
GC = []
vid = {"n": 0}
try:
    video = robomme_plus.get_conditioning_video()
    try:
        nvid = len(video)
        for k in range(nvid):
            frames.append(video[k])
    except Exception:
        frames = list(video)
    vid["n"] = len(frames)
    note("video frames: " + str(len(frames)))
    for fr in frames:
        rs = fr.robot_state
        TS.append(float(fr.timestamp_s))
        Q.append(vec(rs.joint_positions["primary"]))
        G.append(float(rs.gripper_positions["primary"]))
        eep = rs.end_effector_poses["primary"]
        P.append(vec(eep.position))
        QU.append(vec(eep.quaternion_wxyz))
except Exception as exc:
    fail("video load", exc)

n = min(len(TS), len(Q), len(G), len(P), len(QU))
kg = None
kl = None
kr = None
k_ins = None
k_pre = None
k_start = 0
k_end = 0
discont = []
try:
    if n >= 2:
        vid["t0"] = fl(TS[0], 3)
        vid["t_end"] = fl(TS[n - 1], 3)
        dts = [TS[k + 1] - TS[k] for k in range(n - 1)]
        vid["dt_min"] = fl(min(dts), 4)
        vid["dt_max"] = fl(max(dts), 4)
        vid["dt_median"] = fl(quantile(dts, 0.5), 4)
        # gripper command reconstruction (lookahead >= 2 ticks)
        # an event needs a 2-tick lookahead change AND a large travel within 0.5 s, so contact
        # jitter while holding (fingers pushed slightly apart) never reads as an opening
        cur = 1.0 if G[0] > 0.5 else 0.0
        for k in range(n):
            j = k + 1
            if j < n and TS[j] - TS[k] < 0.075:
                j = k + 2
            j = min(j, n - 1)
            delta = G[j] - G[k]
            win = [G[i] for i in range(k, n) if TS[i] - TS[k] <= 0.5]
            if delta < -0.02 and min(win) < G[k] - 0.1:
                cur = 0.0
            elif delta > 0.02 and max(win) > G[k] + 0.15:
                cur = 1.0
            GC.append(cur)
        closes = [k for k in range(1, n) if GC[k - 1] > 0.5 and GC[k] < 0.5]
        opens = [k for k in range(1, n) if GC[k - 1] < 0.5 and GC[k] > 0.5]
        vid["close_frames"] = closes[:10]
        vid["open_frames"] = opens[:10]
        hz = STATE["hz"]
        for k in range(n - 1):
            nt = max(1, int(round((TS[k + 1] - TS[k]) * hz)))
            jump = max(abs(Q[k + 1][i] - Q[k][i]) for i in range(7)) / nt
            if jump > 0.2:
                discont.append(k)
        vid["discontinuities"] = discont[:20]
        moving = [k for k in range(n) if max(abs(Q[k][i] - Q[0][i]) for i in range(7)) > 0.005]
        k_move = moving[0] if moving else None
        vid["first_motion_frame"] = k_move
        if closes:
            kg = closes[0]
            gz = P[kg][2]
            for k in opens:
                if k > kg:
                    kr = k
                    break
            kstop = kr if kr is not None else n
            for j in range(kg + 1, kstop):
                if P[j][2] - gz >= 0.05:
                    kl = j
                    break
            if kl is None:
                for j in range(kg + 1, kstop):
                    if P[j][2] - gz >= 0.03 and (kl is None or P[j][2] > P[kl][2]):
                        kl = j
                vid["lift_rule"] = "max z >= 3 cm"
            vid["grasp_frame"] = kg
            vid["grasp_ee_pos"] = rvec(P[kg])
            vid["grasp_ee_quat"] = rvec(QU[kg])
            settle = min(n - 1, kg + 10)
            vid["grip_after_close"] = fl(G[settle], 3)
            vid["lift_frame"] = kl
            vid["release_frame"] = kr
            k_ins = (kr - 1) if kr is not None else (n - 1)
            for d in discont:
                if d > kg and d < k_ins:
                    k_ins = d
                    break
            vid["insert_end_frame"] = k_ins
            vid["insert_end_ee_pos"] = rvec(P[k_ins])
            vid["insert_end_ee_quat"] = rvec(QU[k_ins])
            if kl is not None:
                vid["lift_ee_pos"] = rvec(P[kl])
            # final straight approach: frames before k_ins at the insertion height (+-4 mm);
            # contact onset = first frame on it whose speed drops below 35 % of its initial speed
            if kl is not None and k_ins - kl > 12:
                ka = k_ins
                while ka - 1 > kl and abs(P[ka - 1][2] - P[k_ins][2]) < 0.004:
                    ka -= 1
                sp = [math.sqrt(sum((P[j + 1][i] - P[j][i]) ** 2 for i in range(3))) for j in range(n - 1)]
                vid["approach_start_frame"] = ka
                if k_ins - ka >= 12:
                    # reference = peak speed of the first 25 approach frames; onset = first
                    # frame after that peak below 35 % of it (the demonstrator's contact slow-down)
                    kpk = ka
                    for j in range(ka, min(ka + 25, k_ins)):
                        if sp[j] > sp[kpk]:
                            kpk = j
                    vref = sp[kpk]
                    ks = None
                    for j in range(kpk + 1, k_ins):
                        if sp[j] < 0.35 * vref:
                            ks = j
                            break
                    vid["approach_speed"] = fl(vref, 5)
                    vid["approach_peak_frame"] = kpk
                    vid["slow_onset_frame"] = ks
                    if ks is not None and vref > 0.0005:
                        k_pre = max(ka, ks - 3)
                if k_pre is None:
                    k_pre = max(kl, k_ins - 40)
                    vid["k_pre_rule"] = "fallback k_ins - 40"
                vid["pre_insert_frame"] = k_pre
                vid["pre_insert_ee_pos"] = rvec(P[k_pre])
                vid["insert_length"] = fl(math.sqrt(sum((P[k_ins][i] - P[k_pre][i]) ** 2 for i in range(3))), 4)
        k_start = max(0, (k_move if k_move is not None else 0) - 1)
        k_end = n - 1
        for d in discont:
            if d >= k_start:
                k_end = d
                break
        vid["replay_start"] = k_start
        vid["replay_end"] = k_end
        # compact series for offline analysis
        stride = max(1, n // 256)
        vid["series_stride"] = stride
        vid["series"] = [[fl(TS[k], 3)] + rvec(P[k], 4) + [fl(G[k], 3), GC[k]] for k in range(0, n, stride)]
        vid["joints_series"] = [rvec(Q[k], 3) for k in range(0, n, max(1, n // 64))]
        vid["ee_z_min"] = fl(min(p[2] for p in P), 4)
except Exception as exc:
    fail("video analysis", exc)
report["video"] = vid

try:
    md_sampling = None
    if md is not None:
        try:
            md_sampling = str(md["conditioning_video_sampling"])[:300]
        except Exception:
            md_sampling = None
    vid["sampling_md"] = md_sampling
except Exception as exc:
    fail("video sampling md", exc)

# ================================================================ Phase 4: video vs live
video_peg = None
scene = {}
try:
    if len(frames) > 0 and live_depth is not None:
        checks = {"first": 0, "last": len(frames) - 1}
        if kg is not None:
            checks["grasp"] = kg
        if k_ins is not None:
            checks["insert_end"] = k_ins
        for label in checks:
            k = checks[label]
            fr = frames[k]
            fd = rows2d(fr.cameras[FRONT].depth_m)
            entry = {"frame": k}
            changed = 0
            seen = 0
            for v in range(0, len(fd), 4):
                for u in range(0, len(fd[0]), 4):
                    a = fd[v][u]
                    b = live_depth[v][u]
                    if valid_depth(a) and valid_depth(b):
                        seen += 1
                        if abs(float(a) - float(b)) > 0.01:
                            changed += 1
            entry["depth_changed_frac"] = round(changed / float(max(1, seen)), 4)
            try:
                entry["cam_pose_same"] = (max(abs(x - y) for x, y in zip(
                    vec(fr.cameras[FRONT].camera_pose.position),
                    vec(live_obs.cameras[FRONT].camera_pose.position))) < 1e-3)
            except Exception:
                pass
            if calib is not None:
                mv = cam_model(fr.cameras[FRONT], calib["conv"], calib["ray"])
                if peg is not None:
                    entry["peg_agree"] = depth_agreement(mv, fd, sample_list(peg["pts"], 300), 0.006)
                if box is not None:
                    entry["box_agree"] = depth_agreement(mv, fd, sample_list(box["pts"], 300), 0.006)
            try:
                entry["joints_vs_live"] = fl(max(abs(Q[k][i] - report["robot_state0"]["joints"][i])
                                                 for i in range(7)), 4)
            except Exception:
                pass
            scene[label] = entry
    report["scene_compare"] = scene
except Exception as exc:
    fail("scene compare", exc)

# video peg pose from frame 0 near the demonstrated grasp (colour + height band)
try:
    if (kg is not None and calib is not None and peg is not None and peg_colors is not None
            and table_z is not None and len(frames) > 0):
        fr = frames[0]
        fd = rows2d(fr.cameras[FRONT].depth_m)
        fc = rows3d(fr.cameras[FRONT].rgb)
        mv = cam_model(fr.cameras[FRONT], calib["conv"], calib["ray"])
        gpos = P[kg]
        top = (peg["g"]["zhi"] - table_z) + 0.015
        ref_cols = [c for c in (peg_colors["plus"], peg_colors["minus"], peg_colors["all"]) if c is not None]
        sel = []
        for v in range(0, len(fd), 2):
            for u in range(0, len(fd[0]), 2):
                d = fd[v][u]
                if not valid_depth(d):
                    continue
                p = backproject(mv, u, v, float(d))
                if math.hypot(p[0] - gpos[0], p[1] - gpos[1]) > 0.25:
                    continue
                hband = p[2] - table_z
                if hband < 0.004 or hband > top:
                    continue
                col = fc[v][u]
                if min(color_dist(col, rc) for rc in ref_cols) > 55.0:
                    continue
                sel.append([p[0], p[1], p[2], col])
        info = {"n_points": len(sel)}
        if len(sel) >= 15:
            vg = cloud_geom([[s[0], s[1], s[2]] for s in sel], table_z)
            info["geom"] = geom_summary(vg)
            ang = vg["axis_angle"]
            ax = (math.cos(ang), math.sin(ang))
            c = vg["center"]
            plus = []
            minus = []
            for s in sel:
                ss = (s[0] - c[0]) * ax[0] + (s[1] - c[1]) * ax[1]
                if ss > vg["length"] / 6.0:
                    plus.append(s[3])
                elif ss < -vg["length"] / 6.0:
                    minus.append(s[3])
            flip = None
            if plus and minus and peg_colors.get("plus") is not None and peg_colors.get("minus") is not None \
                    and peg_colors.get("end_contrast", 0.0) > 25.0:
                mp = [sum(float(cc[i]) for cc in plus) / len(plus) for i in range(3)]
                mm = [sum(float(cc[i]) for cc in minus) / len(minus) for i in range(3)]
                same_cost = color_dist(mp, peg_colors["plus"]) + color_dist(mm, peg_colors["minus"])
                flip_cost = color_dist(mp, peg_colors["minus"]) + color_dist(mm, peg_colors["plus"])
                flip = flip_cost < same_cost
                info["same_cost"] = fl(same_cost, 1)
                info["flip_cost"] = fl(flip_cost, 1)
            # video axis angle expressed in the live "plus" direction convention
            if flip is True:
                ang = wrap_pi(ang + math.pi)
            live_ang = peg["g"]["axis_angle"]
            if flip is None:
                dth = wrap_half(live_ang - ang)
            else:
                dth = wrap_pi(live_ang - ang)
            info["flip"] = flip
            info["dtheta_deg"] = fl(math.degrees(dth), 2)
            dc = [peg["g"]["center"][0] - c[0], peg["g"]["center"][1] - c[1]]
            info["dcenter"] = rvec(dc)
            # grasp offset along the peg (signed in live plus direction) from the video
            gs = (gpos[0] - c[0]) * math.cos(ang) + (gpos[1] - c[1]) * math.sin(ang)
            info["grasp_offset_along_axis"] = fl(gs)
            info["grasp_offset_frac_of_half"] = fl(gs / max(1e-3, 0.5 * vg["length"]), 3)
            video_peg = {"center": c, "angle": ang, "dtheta": dth, "dc": dc, "flip": flip}
        report["video_peg"] = info
except Exception as exc:
    fail("video peg", exc)

# ================================================================ Phase 5: decide + act
dec = report["decision"]
mode = "none"
try:
    peg_same = None
    first = scene.get("first", {})
    pa = first.get("peg_agree")
    if pa is not None and pa.get("seen", 0) >= 20:
        peg_same = pa["frac"] >= 0.7
    if peg_same is None and video_peg is not None:
        peg_same = (math.hypot(video_peg["dc"][0], video_peg["dc"][1]) < 0.015
                    and abs(video_peg["dtheta"]) < math.radians(8.0))
    if peg_same is None and first.get("depth_changed_frac") is not None:
        peg_same = first["depth_changed_frac"] < 0.01
    dec["peg_same_as_video_start"] = peg_same
    if kg is None or n < 2:
        mode = "perception_only"
    elif peg_same is False and video_peg is not None and kl is not None:
        mode = "transformed_grasp"
    else:
        mode = "replay_full"
    dec["mode"] = mode
    note("mode: " + mode)
except Exception as exc:
    fail("decide", exc)
    mode = "replay_full" if kg is not None else "perception_only"
    dec["mode"] = mode


def build_plan(k_from, k_to):
    plan = []
    hz = STATE["hz"]
    for k in range(k_from, k_to):
        nt = max(1, int(round((TS[k + 1] - TS[k]) * hz)))
        q0 = Q[k]
        q1 = Q[k + 1]
        for j in range(1, nt + 1):
            a = j / float(nt)
            plan.append(([q0[i] + a * (q1[i] - q0[i]) for i in range(7)], GC[k], k))
    return plan


def compress_plan(plan, budget):
    if len(plan) <= budget or budget <= 10:
        return plan
    s = int(math.ceil(len(plan) / float(budget)))
    keep = []
    for i in range(len(plan)):
        near_change = False
        for j in range(max(0, i - 8), min(len(plan), i + 9)):
            if plan[j][1] != plan[i][1]:
                near_change = True
                break
        if near_change or i % s == 0 or i == len(plan) - 1:
            keep.append(plan[i])
    return keep


def run_plan(plan, label):
    done = 0
    last_k = None
    for item in plan:
        if not tick(item[0], item[1]):
            break
        done += 1
        last_k = item[2]
        if kg is not None and last_k is not None and last_k >= kg + 3:
            mark_clause("grasp")
    report.setdefault("plans", []).append({"label": label, "planned": len(plan), "executed": done,
                                           "last_frame": last_k, "ticks_total": STATE["ticks"]})
    note(label + " executed " + str(done) + "/" + str(len(plan)) + " last frame " + str(last_k))
    return done


def remaining_budget():
    if not STATE["max_steps"]:
        return 100000
    return STATE["max_steps"] - STATE["ticks"]


# ---------------------------------------------------------------- replay_full
if mode == "replay_full":
    try:
        live_q = current_joints()
        dq = max(abs(live_q[i] - Q[k_start][i]) for i in range(7))
        dec["start_joint_diff"] = fl(dq)
        if dq > 0.01:
            interp_to(Q[k_start], GC[k_start], 0.05)
        k_stop = k_pre if (k_pre is not None and k_pre > k_start) else k_end
        dec["replay_to_frame"] = k_stop
        plan = build_plan(k_start, k_stop)
        dec["plan_ticks_raw"] = len(plan)
        plan = compress_plan(plan, remaining_budget() - 200)
        dec["plan_ticks"] = len(plan)
        run_plan(plan, "replay_full")
    except Exception as exc:
        fail("replay_full", exc)

# ---------------------------------------------------------------- transformed_grasp
if mode == "transformed_grasp":
    try:
        dth = video_peg["dtheta"]
        cv = video_peg["center"]
        cl = peg["g"]["center"]
        gp = P[kg]
        rx = gp[0] - cv[0]
        ry = gp[1] - cv[1]
        gx = cl[0] + math.cos(dth) * rx - math.sin(dth) * ry
        gy = cl[1] + math.sin(dth) * rx + math.cos(dth) * ry
        gz = gp[2]
        gq = qmul(qz(dth), QU[kg])
        dec["live_grasp_pos"] = rvec([gx, gy, gz])
        dec["live_grasp_quat"] = rvec(gq)
        # free-space part of the video approach is skipped; open first
        if current_grip() < 0.9:
            after_motion("open_gripper", open_gripper())
        ok1 = after_motion("move_to_pose(pre_grasp)", move_to_pose(
            Pose([gx, gy, gz + 0.10], gq, base_frame), tolerance=0.01, max_steps=200))
        ok2 = False
        if not STATE["terminal"]:
            ok2 = after_motion("move_to_pose(grasp)", move_to_pose(
                Pose([gx, gy, gz], gq, base_frame), tolerance=0.006, max_steps=150))
        if not STATE["terminal"]:
            after_motion("close_gripper", close_gripper())
            STATE["last_grip"] = 0.0
            mark_clause("grasp")
        try:
            dec["grip_after_close"] = fl(current_grip(), 3)
        except Exception:
            pass
        if not STATE["terminal"]:
            lift_z = gz + max(0.05, P[kl][2] - P[kg][2])
            after_motion("move_to_pose(lift)", move_to_pose(
                Pose([gx, gy, lift_z], gq, base_frame), tolerance=0.01, max_steps=150))
        if not STATE["terminal"]:
            STATE["last_target"] = None
            interp_to(Q[kl], 0.0, 0.04)
            k_stop = k_pre if (k_pre is not None and k_pre > kl) else k_end
            dec["replay_to_frame"] = k_stop
            plan = build_plan(kl, k_stop)
            dec["plan_ticks_raw"] = len(plan)
            plan = compress_plan(plan, remaining_budget() - 200)
            dec["plan_ticks"] = len(plan)
            run_plan(plan, "replay_after_lift")
    except Exception as exc:
        fail("transformed_grasp", exc)

# ---------------------------------------------------------------- staged insertion (v01)
def dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross3(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def unit3(a):
    nn = math.sqrt(dot3(a, a))
    if nn < 1e-9:
        return None
    return [a[0] / nn, a[1] / nn, a[2] / nn]


def nlerp(qa, qb, t):
    if dot3(qa[1:], qb[1:]) + qa[0] * qb[0] < 0:
        qb = [-v for v in qb]
    q = [qa[i] + t * (qb[i] - qa[i]) for i in range(4)]
    nn = math.sqrt(sum(v * v for v in q))
    return [v / nn for v in q]


def ik_joints(res):
    if res is None:
        return None
    try:
        if not bool(res.ok):
            return None
    except Exception:
        return None
    jp = None
    try:
        jp = res.joint_positions
    except Exception:
        jp = None
    if jp is None:
        try:
            jp = res.solution
        except Exception:
            jp = None
    if jp is None:
        return None
    try:
        q = vec(jp["primary"])
        if len(q) >= 7:
            return q[:7]
    except Exception:
        pass
    try:
        q = vec(jp)
        if len(q) >= 7:
            return q[:7]
    except Exception:
        pass
    return None


def live_ee():
    st = get_robot_state()
    ee = st.end_effector_poses["primary"]
    return vec(ee.position), vec(ee.quaternion_wxyz), vec(st.joint_positions["primary"])


INS = {"attempts": [], "ik_calls": 0, "ik_fail": 0, "best_depth": 0.0, "reached": False}


def staged_insert(grip):
    if k_pre is None or k_ins is None:
        INS["skipped"] = "no pre-insert frame"
        return
    u = unit3([P[k_ins][i] - P[k_pre][i] for i in range(3)])
    L = math.sqrt(sum((P[k_ins][i] - P[k_pre][i]) ** 2 for i in range(3)))
    if u is None or L < 0.01:
        INS["skipped"] = "insertion length " + str(fl(L))
        return
    # perpendicular axes: e1 = vertical component orthogonal to u, e2 = u x e1
    zup = [0.0, 0.0, 1.0]
    e1 = unit3([zup[i] - dot3(zup, u) * u[i] for i in range(3)])
    if e1 is None:
        e1 = unit3(cross3(u, [1.0, 0.0, 0.0]))
    e2 = unit3(cross3(u, e1))
    B, qB, qjB = live_ee()
    INS["u"] = rvec(u, 3)
    INS["e1"] = rvec(e1, 3)
    INS["e2"] = rvec(e2, 3)
    INS["L"] = fl(L)
    INS["B_live"] = rvec(B)
    INS["B_video"] = rvec(P[k_pre])
    qEnd = QU[k_ins]
    # delta-joint reference: IK of the live pose (cancels any constant TCP / solver bias)
    ref = None
    try:
        INS["ik_calls"] += 1
        ref = ik_joints(solve_ik(Pose(B, qB, base_frame), seed_joints=qjB))
    except Exception as exc:
        fail("ik ref", exc)
        ref = None
    if ref is None:
        INS["ik_fail"] += 1
        INS["ref_mode"] = "absolute"
    else:
        INS["ref_mode"] = "delta"
        INS["ref_residual"] = fl(max(abs(ref[i] - qjB[i]) for i in range(7)))
    seed = {"q": ref if ref is not None else qjB}

    def joints_for(sv, off):
        tgt = [B[i] + sv * u[i] + off[0] * e1[i] + off[1] * e2[i] for i in range(3)]
        q = nlerp(qB, qEnd, max(0.0, min(1.0, sv / L)))
        INS["ik_calls"] += 1
        try:
            sol = ik_joints(solve_ik(Pose(tgt, q, base_frame), seed_joints=seed["q"]))
        except Exception as exc:
            fail("ik", exc)
            sol = None
        if sol is None:
            INS["ik_fail"] += 1
            return None
        if max(abs(sol[i] - seed["q"][i]) for i in range(7)) > 0.25:
            INS["ik_fail"] += 1
            return None
        seed["q"] = sol
        if ref is None:
            return sol
        return [qjB[i] + sol[i] - ref[i] for i in range(7)]

    def progress():
        pos, qq, qj = live_ee()
        d = [pos[i] - B[i] for i in range(3)]
        return dot3(d, u), dot3(d, e1), dot3(d, e2), qj

    def go_joints(qt, nt):
        start = current_joints()
        for j in range(1, nt + 1):
            a = j / float(nt)
            if not tick([start[i] + a * (qt[i] - start[i]) for i in range(7)], grip):
                return False
        return True

    lead_cap = 0.005
    seg = 0.002
    ST = {"s_cmd": 0.0, "moved": False}

    def push_line(off, label):
        # advance along u at 0.5 mm/tick (1 mm/tick past 3 cm), never more than lead_cap ahead
        rec = {"phase": label, "offset_mm": [fl(off[0] * 1000, 2), fl(off[1] * 1000, 2)], "tick0": STATE["ticks"]}
        s0 = progress()[0]
        rec["s_start"] = fl(s0)
        ST["s_cmd"] = max(ST["s_cmd"], s0) if label == "push" else s0
        stall = 0
        last_s = s0
        res_s = "stopped"
        while not STATE["terminal"] and not STATE["stop"]:
            if remaining_budget() < 45:
                res_s = "budget"
                break
            s_act = progress()[0]
            if s_act > INS["best_depth"]:
                INS["best_depth"] = s_act
            if s_act >= L - 0.002:
                res_s = "reached"
                INS["reached"] = True
                break
            if ST["s_cmd"] - s_act < lead_cap:
                speed = 0.0005 if s_act < 0.03 else 0.001
                s_new = min(L + 0.002, ST["s_cmd"] + seg)
                qt = joints_for(s_new, off)
                if qt is None:
                    res_s = "ik_fail"
                    break
                nt = max(2, int(math.ceil((s_new - ST["s_cmd"]) / speed)))
                ST["s_cmd"] = s_new
                ST["moved"] = True
                if not go_joints(qt, nt):
                    break
            else:
                hold(2, grip)
            s_now = progress()[0]
            if s_now - last_s < 0.0003:
                stall += 1
            else:
                stall = 0
                last_s = s_now
            if stall >= 4:
                res_s = "stall"
                break
        s_act, d1, d2, qj = progress()
        rec["result"] = res_s
        rec["depth"] = fl(s_act)
        rec["deflect_mm"] = [fl((d1 - off[0]) * 1000, 2), fl((d2 - off[1]) * 1000, 2)]
        rec["ticks"] = STATE["ticks"] - rec["tick0"]
        INS["attempts"].append(rec)
        note("insert " + label + " offset " + str(rec["offset_mm"]) + " -> " + res_s + " depth " + str(rec["depth"]))
        return res_s, [d1 - off[0], d2 - off[1]]

    def spiral(off0, theta0, turn):
        # in-contact Archimedean spiral (pitch 1 mm, radius <= 3.5 mm), light push 2 mm past
        # the contact depth; an advance of > 1.2 mm means the tip found the hole
        rec = {"phase": "spiral", "start_offset_mm": [fl(off0[0] * 1000, 2), fl(off0[1] * 1000, 2)],
               "tick0": STATE["ticks"], "turn": turn}
        s_c = progress()[0]
        rec["s_contact"] = fl(s_c)
        pitch = 0.001
        th = 0.0
        found = None
        while not STATE["terminal"] and not STATE["stop"]:
            if remaining_budget() < 60:
                rec["result"] = "budget"
                break
            r = pitch * th / (2.0 * math.pi)
            if r > 0.0035:
                rec["result"] = "exhausted"
                break
            th += 0.0004 / max(r, 0.0006)
            r = pitch * th / (2.0 * math.pi)
            a = theta0 + turn * th
            off = [off0[0] + r * math.cos(a), off0[1] + r * math.sin(a)]
            qt = joints_for(s_c + 0.002, off)
            if qt is None:
                continue
            if not tick(qt, grip):
                break
            s_now = progress()[0]
            if s_now > s_c + 0.0012:
                found = off
                rec["result"] = "found"
                rec["found_offset_mm"] = [fl(off[0] * 1000, 2), fl(off[1] * 1000, 2)]
                break
        if "result" not in rec:
            rec["result"] = "stopped"
        rec["ticks"] = STATE["ticks"] - rec["tick0"]
        INS["attempts"].append(rec)
        note("insert spiral -> " + rec["result"])
        return found

    def back_off(off, dist):
        s_now = progress()[0]
        sb = max(0.0, s_now - dist)
        qb = joints_for(sb, off)
        if qb is None:
            return False
        ST["s_cmd"] = sb
        if not go_joints(qb, max(4, int(math.ceil(dist / 0.0008)))):
            return False
        hold(4, grip)
        return True

    off = [0.0, 0.0]
    res_s, dd = push_line(off, "push")
    rounds = 0
    turn = 1.0
    while res_s == "stall" and rounds < 3 and not STATE["terminal"] and not STATE["stop"]:
        rounds += 1
        dn = math.hypot(dd[0], dd[1])
        theta0 = math.atan2(dd[1], dd[0]) if dn > 0.0005 else 0.5 * math.pi
        INS.setdefault("deflections_mm", []).append([fl(dd[0] * 1000, 2), fl(dd[1] * 1000, 2)])
        found = spiral(off, theta0, turn)
        if found is not None:
            off = found
            res_s, dd = push_line(off, "push")
            continue
        # spiral exhausted: re-seat on the nominal line and spiral the other way
        turn = -turn
        off = [0.0, 0.0]
        if not back_off(off, 0.006):
            res_s = "ik_fail"
            break
        res_s, dd = push_line(off, "reseat")
    INS["final_result"] = res_s
    # no usable IK at all: fall back to the v00 behaviour (replay of the demonstrated joints)
    if res_s == "ik_fail" and not ST["moved"] and not STATE["terminal"] and not STATE["stop"]:
        INS["fallback"] = "video joint replay from pre-insert frame"
        plan = build_plan(k_pre, k_end)
        plan = compress_plan(plan, remaining_budget() - 45)
        run_plan(plan, "replay_insert_fallback")
    if INS["reached"]:
        mark_clause("insert")
    report["insertion"] = INS


if mode in ("replay_full", "transformed_grasp") and not STATE["terminal"] and not STATE["stop"]:
    try:
        staged_insert(0.0 if STATE["last_grip"] < 0.5 else STATE["last_grip"])
    except Exception as exc:
        fail("staged_insert", exc)
        report["insertion"] = INS


# ---------------------------------------------------------------- perception_only
if mode == "perception_only":
    try:
        if peg is None:
            raise ValueError("no peg localised")
        g = peg["g"]
        ang = g["axis_angle"]
        ax = [math.cos(ang), math.sin(ang)]
        c = g["center"]
        # insert the end nearer the box; grasp on the far half
        sign = 1.0
        if box is not None:
            bc = box["g"]["center"]
            if (bc[0] - c[0]) * ax[0] + (bc[1] - c[1]) * ax[1] < 0:
                sign = -1.0
        gx = c[0] - sign * 0.25 * g["length"] * ax[0]
        gy = c[1] - sign * 0.25 * g["length"] * ax[1]
        gz = 0.5 * (g["zlo"] + g["zhi"])
        st = get_robot_state()
        q_home = vec(st.end_effector_poses["primary"].quaternion_wxyz)
        rh = qmat(q_home)
        close_yaw = math.atan2(rh[1][1], rh[0][1])
        want = ang + math.pi / 2.0
        dyaw = wrap_half(want - close_yaw)
        gq = qmul(qz(dyaw), q_home)
        dec["perception_grasp"] = rvec([gx, gy, gz])
        after_motion("open_gripper", open_gripper())
        after_motion("move_to_pose(pre_grasp)", move_to_pose(
            Pose([gx, gy, gz + 0.10], gq, base_frame), tolerance=0.01, max_steps=200))
        if not STATE["terminal"]:
            after_motion("move_to_pose(grasp)", move_to_pose(
                Pose([gx, gy, gz], gq, base_frame), tolerance=0.006, max_steps=150))
        if not STATE["terminal"]:
            after_motion("close_gripper", close_gripper())
            STATE["last_grip"] = 0.0
            mark_clause("grasp")
        if not STATE["terminal"]:
            after_motion("move_to_pose(lift)", move_to_pose(
                Pose([gx, gy, gz + 0.10], gq, base_frame), tolerance=0.01, max_steps=150))
        if box is not None and not STATE["terminal"]:
            bg = box["g"]
            bc = bg["center"]
            # hole axis: box horizontal axis most aligned with peg-centre -> box direction
            to_box = [bc[0] - c[0], bc[1] - c[1]]
            best_axis = None
            for cand_ang in (bg["axis_angle"], bg["axis_angle"] + math.pi / 2.0):
                a2 = [math.cos(cand_ang), math.sin(cand_ang)]
                dot = a2[0] * to_box[0] + a2[1] * to_box[1]
                if best_axis is None or abs(dot) > abs(best_axis[0]):
                    best_axis = (dot, cand_ang)
            nang = best_axis[1] if best_axis[0] >= 0 else best_axis[1] + math.pi
            nx, ny = math.cos(nang), math.sin(nang)
            half_box = 0.5 * (bg["length"] if abs(wrap_half(nang - bg["axis_angle"])) < 0.5 else bg["width"])
            tip = 0.75 * g["length"]
            hole_z = 0.5 * (bg["zlo"] + bg["zhi"])
            # rotate held peg so its insertion end points along +n
            peg_dir = math.atan2(sign * ax[1], sign * ax[0])
            rot = wrap_pi(nang - peg_dir)
            iq = qmul(qz(rot), gq)
            stand = half_box + tip + 0.03
            pre = [bc[0] - nx * stand, bc[1] - ny * stand, hole_z]
            dec["perception_pre_insert"] = rvec(pre)
            after_motion("move_to_pose(pre_insert_high)", move_to_pose(
                Pose([pre[0], pre[1], hole_z + 0.08], iq, base_frame), tolerance=0.01, max_steps=200))
            if not STATE["terminal"]:
                after_motion("move_to_pose(pre_insert)", move_to_pose(
                    Pose(pre, iq, base_frame), tolerance=0.006, max_steps=150))
            depth_in = half_box + 0.04
            for frac in (0.35, 0.7, 1.0):
                if STATE["terminal"]:
                    break
                tgt = [pre[0] + nx * depth_in * frac, pre[1] + ny * depth_in * frac, hole_z]
                after_motion("move_to_pose(insert)", move_to_pose(
                    Pose(tgt, iq, base_frame), tolerance=0.006, max_steps=80))
            mark_clause("insert")
    except Exception as exc:
        fail("perception_only", exc)

# ================================================================ Phase 6: hold, self-check
try:
    if not STATE["terminal"] and not STATE["stop"] and STATE["ticks"] > 0:
        grip = STATE["last_grip"]
        held = hold(min(40, max(0, remaining_budget() - 1)), grip)
        dec["final_hold_ticks"] = held
except Exception as exc:
    fail("final hold", exc)

# clause self-check: any button clause still pending
try:
    pending = [c for c in report["clauses"] if not c["executed"]]
    report["pending_clauses_before_selfcheck"] = [c["text"] for c in pending]
    if (not STATE["terminal"] and not STATE["stop"]
            and any(c["kind"] == "button" for c in pending)):
        seg = segment_text(FRONT, "button")
        target = None
        if seg.ok:
            for cand in list(seg.segmentations)[:5]:
                pc = mask_to_point_cloud(cand, FRONT, target_frame=base_frame)
                pts = points_of(pc)
                if len(pts) < 10:
                    continue
                bg = cloud_geom(pts, table_z)
                if bg["length"] <= 0.12 and (bg["height_above_table"] is None or bg["height_above_table"] <= 0.08):
                    target = bg
                    break
        if target is not None:
            st = get_robot_state()
            qd = vec(st.end_effector_poses["primary"].quaternion_wxyz)
            if current_grip() < 0.9:
                after_motion("open_gripper", open_gripper())
            bx, by, bz = target["center"][0], target["center"][1], target["zhi"]
            after_motion("move_to_pose(button_above)", move_to_pose(
                Pose([bx, by, bz + 0.08], qd, base_frame), tolerance=0.01, max_steps=150))
            if not STATE["terminal"]:
                after_motion("move_to_pose(button_press)", move_to_pose(
                    Pose([bx, by, bz - 0.01], qd, base_frame), tolerance=0.01, max_steps=60))
            if not STATE["terminal"]:
                after_motion("move_to_pose(button_up)", move_to_pose(
                    Pose([bx, by, bz + 0.08], qd, base_frame), tolerance=0.01, max_steps=80))
            mark_clause("button")
except Exception as exc:
    fail("selfcheck", exc)

try:
    report["ticks_used"] = STATE["ticks"]
    report["stop"] = STATE["stop"]
    report["terminal"] = STATE["terminal"]
    st = get_robot_state()
    report["final_state"] = {
        "joints": rvec(st.joint_positions["primary"]),
        "gripper": fl(st.gripper_positions["primary"], 3),
        "ee_pos": rvec(st.end_effector_poses["primary"].position),
        "ee_quat": rvec(st.end_effector_poses["primary"].quaternion_wxyz),
        "timestamp_s": fl(st.timestamp_s, 3),
    }
except Exception as exc:
    fail("final state", exc)

note("done; ticks " + str(STATE["ticks"]) + " terminal " + str(STATE["terminal"]))
result = report
