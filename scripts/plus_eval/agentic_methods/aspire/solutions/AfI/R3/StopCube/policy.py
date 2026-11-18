# StopCube v01 — v00 plus button/cube disambiguation (AfI, RoboMME-Plus).
# v01 changes (each from the ep-1 v00 trace, where "button" top-1 was the target disc and the arm
# hovered over the target, in the cube's path):
#   * button must lie outside the selected target (centre distance > target radius + 0.02 m) and
#     stand taller than it (top >= target top + 0.008 m); press XY = median of its top-surface points;
#   * cube vs button by motion: hold 3 ticks, re-segment, the candidate that moved is the cube,
#     static tall candidates are button candidates (fallback: v00 score rule);
#   * arrival zone r_in = max(target radius, 0.02 m), r_out = target radius + 0.03 m.
#
# Records: task language, metadata, controller metadata, conditioning-video summary,
# SAM3 prompt probes with per-candidate cloud stats, camera-convention self-calibration,
# a per-tick cube timeline, and an in-air press-latency calibration. Then attempts the task:
# hover above the button, count the cube's arrivals at the target, and press so the press
# registers when the cube is at its closest approach on the N-th arrival.
#
# Everything task-dependent (button, cube, target positions, table height, N) is derived at
# runtime from the language, RGB-D and public geometry tools. Constants are generic offsets,
# tolerances and tick limits.

import math

report = {"version": "v01_button_disambig", "phase": "start", "notes": []}
result = report

S = {"ticks": 0, "terminal": False, "pressed": False, "press_tick": None,
     "visits": 0, "in_zone": False, "visit_ticks": [], "visit_dmin": [],
     "armed": False, "last": None, "lost": 0, "sam_checks": 0}
TL = []
POS = []
report["timeline"] = TL

HOVER = 0.025        # hover height of the tool point above the observed button top (m)
PRESS_DEPTH = 0.02   # press target below the observed button top (m)
REG_DEPTH = 0.005    # assumed travel below the top at which a press registers (m)
TRANSIT_H = 0.15     # transit waypoint height above the button top (m)
# The pre-hover sits HOVER + (HOVER + PRESS_DEPTH) above the top, so the pre-hover -> hover
# jump has exactly the dz of the hover -> press jump and doubles as the latency calibration.

ORD = {"first": 1, "once": 1, "one": 1, "1st": 1, "second": 2, "twice": 2, "two": 2,
       "2nd": 2, "third": 3, "thrice": 3, "three": 3, "3rd": 3, "fourth": 4, "four": 4,
       "4th": 4, "fifth": 5, "five": 5, "5th": 5, "sixth": 6, "six": 6, "6th": 6,
       "seventh": 7, "seven": 7, "7th": 7, "eighth": 8, "eight": 8, "8th": 8,
       "ninth": 9, "nine": 9, "9th": 9, "tenth": 10, "ten": 10, "10th": 10}
TARGET_WORDS = ["target", "disc", "circle", "goal", "zone", "region", "area", "marker",
                "spot", "pad", "plate", "line"]
COLORS = ["red", "green", "blue", "yellow", "purple", "orange", "white", "black",
          "pink", "cyan", "gray", "grey", "brown", "magenta"]


def quant(vals, f):
    o = sorted(vals)
    if len(o) == 0:
        return None
    return o[int((len(o) - 1) * f)]


def r4(x):
    if x is None:
        return None
    return round(float(x), 4)


def quat_to_rot(qq):
    w = float(qq[0]); x = float(qq[1]); y = float(qq[2]); z = float(qq[3])
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def cloud_stats(pc):
    xs = []; ys = []; zs = []
    for p in pc.points:
        xs.append(float(p[0])); ys.append(float(p[1])); zs.append(float(p[2]))
    if len(xs) == 0:
        return None
    ztop = quant(zs, 0.95)
    tx = []; ty = []
    for i in range(len(zs)):
        if zs[i] >= ztop - 0.006:
            tx.append(xs[i]); ty.append(ys[i])
    if len(tx) == 0:
        tx = xs; ty = ys
    return {"n": len(xs), "topxy": [quant(tx, 0.5), quant(ty, 0.5)],
            "med": [quant(xs, 0.5), quant(ys, 0.5), quant(zs, 0.5)],
            "lo": [quant(xs, 0.05), quant(ys, 0.05), quant(zs, 0.05)],
            "hi": [quant(xs, 0.95), quant(ys, 0.95), quant(zs, 0.95)]}


def round_stats(st):
    if st is None:
        return None
    return {"n": st["n"], "med": [r4(v) for v in st["med"]], "topxy": [r4(v) for v in st["topxy"]],
            "lo": [r4(v) for v in st["lo"]], "hi": [r4(v) for v in st["hi"]]}


def parse_count(text):
    t = text.lower()
    for ch in ".,;:!?()\"'":
        t = t.replace(ch, " ")
    words = t.split()
    with_times = []
    plain = []
    for i in range(len(words)):
        w = words[i]
        val = None
        if w in ORD:
            val = ORD[w]
        elif w.isdigit():
            val = int(w)
        if val is None:
            continue
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if nxt.startswith("time"):
            with_times.append(val)
        else:
            plain.append(val)
    return with_times, plain


def lang_prompts(text, noun):
    t = text.lower()
    for ch in ".,;:!?()\"'":
        t = t.replace(ch, " ")
    words = t.split()
    out = []
    for i in range(1, len(words)):
        if words[i].startswith(noun) and words[i - 1] in COLORS:
            out.append(words[i - 1] + " " + noun)
    return out


def is_terminal(res):
    term = False
    try:
        term = term or bool(res.terminated)
    except Exception:
        pass
    try:
        term = term or bool(res.truncated)
    except Exception:
        pass
    return term


def ee_pos(obs):
    try:
        p = obs.robot_state.end_effector_poses["primary"].position
        return [float(p[0]), float(p[1]), float(p[2])]
    except Exception:
        return None


G = {}   # geometry / calibration shared state


def build_grid(stride):
    fx = G["fx"]; fy = G["fy"]; cx = G["cx"]; cy = G["cy"]
    R = G["R"]; mode = G["mode"]
    rows = []
    for v in range(0, G["H"], stride):
        items = []
        for u in range(0, G["W"], stride):
            a = (u - cx) / fx
            b = (v - cy) / fy
            if mode == "cv":
                c = (a, b, 1.0)
            elif mode == "gl":
                c = (a, -b, -1.0)
            else:
                c = (1.0, -a, -b)
            rx = R[0][0] * c[0] + R[0][1] * c[1] + R[0][2] * c[2]
            ry = R[1][0] * c[0] + R[1][1] * c[1] + R[1][2] * c[2]
            rz = R[2][0] * c[0] + R[2][1] * c[1] + R[2][2] * c[2]
            items.append((u, rx, ry, rz))
        rows.append((v, items))
    return rows


def deproject(u, v, dep, mode):
    fx = G["fx"]; fy = G["fy"]; cx = G["cx"]; cy = G["cy"]; R = G["R"]; t = G["t"]
    a = (u - cx) / fx
    b = (v - cy) / fy
    if mode == "cv":
        c = (a, b, 1.0)
    elif mode == "gl":
        c = (a, -b, -1.0)
    else:
        c = (1.0, -a, -b)
    return [t[0] + dep * (R[0][0] * c[0] + R[0][1] * c[1] + R[0][2] * c[2]),
            t[1] + dep * (R[1][0] * c[0] + R[1][1] * c[1] + R[1][2] * c[2]),
            t[2] + dep * (R[2][0] * c[0] + R[2][1] * c[1] + R[2][2] * c[2])]


def band_points(depth):
    tx = G["t"][0]; ty = G["t"][1]; tz = G["t"][2]
    zlo = G["zlo"]; zhi = G["zhi"]
    bx = G["button"][0]; by = G["button"][1]; rex2 = G["rex"] * G["rex"]
    gx = G["target"][0]; gy = G["target"][1]; rmax2 = G["rmax"] * G["rmax"]
    rb2 = G["rbase"] * G["rbase"]
    xs = []; ys = []
    for row in G["grid"]:
        drow = depth[row[0]]
        try:
            drow = drow.tolist()
        except Exception:
            pass
        for it in row[1]:
            dep = float(drow[it[0]])
            if not (dep > 0.0 and dep < 10.0):
                continue
            z = tz + dep * it[3]
            if z < zlo or z > zhi:
                continue
            x = tx + dep * it[1]
            y = ty + dep * it[2]
            if (x - bx) * (x - bx) + (y - by) * (y - by) < rex2:
                continue
            if x * x + y * y < rb2:
                continue
            if (x - gx) * (x - gx) + (y - gy) * (y - gy) > rmax2:
                continue
            xs.append(x); ys.append(y)
    return xs, ys


def track_depth(obs):
    depth = obs.cameras["front"].depth_m
    xs, ys = band_points(depth)
    if len(xs) < 3:
        return None
    ref = S["last"]
    if ref is not None:
        gx = []; gy = []
        for i in range(len(xs)):
            if (xs[i] - ref[0]) ** 2 + (ys[i] - ref[1]) ** 2 < 0.08 * 0.08:
                gx.append(xs[i]); gy.append(ys[i])
        if len(gx) >= 3:
            return (quant(gx, 0.5), quant(gy, 0.5), len(gx), len(xs))
    return (quant(xs, 0.5), quant(ys, 0.5), len(xs), len(xs))


def update(obs, tick=None):
    if tick is None:
        tick = S["ticks"]
    p = None
    try:
        p = track_depth(obs)
    except Exception as e:
        if "track_err" not in report:
            report["track_err"] = str(e)[:300]
    ee = ee_pos(obs)
    gx = G["target"][0]; gy = G["target"][1]
    if p is None:
        S["lost"] += 1
        TL.append([tick, None, None, 0, None, S["visits"], r4(ee[2]) if ee else None])
        return
    S["last"] = (p[0], p[1])
    d = math.sqrt((p[0] - gx) ** 2 + (p[1] - gy) ** 2)
    POS.append((tick, p[0], p[1], d))
    if not S["in_zone"] and d < G["r_in"]:
        S["in_zone"] = True
        S["visits"] += 1
        S["visit_ticks"].append(tick)
        S["visit_dmin"].append(r4(d))
    elif S["in_zone"]:
        k = len(S["visit_dmin"]) - 1
        if k >= 0 and d < S["visit_dmin"][k]:
            S["visit_dmin"][k] = r4(d)
        if d > G["r_out"]:
            S["in_zone"] = False
    TL.append([tick, r4(p[0]), r4(p[1]), p[2], r4(d), S["visits"], r4(ee[2]) if ee else None])


def advance(q, track=True):
    if S["terminal"]:
        return None
    res = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=[float(x) for x in q], gripper_position=0.0,
        embodiment=G["emb"])}))
    S["ticks"] += 1
    if S["ticks"] == 1:
        try:
            report["first_step_repr"] = str(res)[:400]
        except Exception:
            pass
    if is_terminal(res):
        S["terminal"] = True
        report["terminal_tick"] = S["ticks"]
        return None
    obs = None
    try:
        obs = res.observation
        if obs is None or not ("front" in obs.cameras):
            obs = None
        elif obs.cameras["front"].depth_m is None:
            obs = None
    except Exception:
        obs = None
    if obs is None:
        obs = get_observation(camera_names=["front"])
    if track:
        update(obs)
    return obs


def move_interp(q_from, q_to, max_delta):
    dmax = 0.0
    for i in range(len(q_from)):
        dmax = max(dmax, abs(q_to[i] - q_from[i]))
    n = max(1, int(math.ceil(dmax / max_delta)))
    obs = None
    for k in range(1, n + 1):
        s = k / float(n)
        q = [q_from[i] + (q_to[i] - q_from[i]) * s for i in range(len(q_from))]
        obs = advance(q)
        if S["terminal"]:
            return None
    return obs


def velocity():
    if len(POS) < 2:
        return None
    a = POS[len(POS) - 1]
    b = None
    for j in range(len(POS) - 2, -1, -1):
        if a[0] - POS[j][0] >= 3:
            b = POS[j]
            break
    if b is None:
        b = POS[0]
    dt = a[0] - b[0]
    if dt <= 0 or dt > 8:
        return None
    return ((a[1] - b[1]) / dt, (a[2] - b[2]) / dt)


def closest_approach():
    if len(POS) == 0:
        return None
    a = POS[len(POS) - 1]
    if a[0] != S["ticks"]:
        return None
    v = velocity()
    if v is None:
        return None
    rx = a[1] - G["target"][0]; ry = a[2] - G["target"][1]
    vv = v[0] * v[0] + v[1] * v[1]
    if vv < 1e-8:
        return None
    tca = -(rx * v[0] + ry * v[1]) / vv
    dca = math.sqrt(max(0.0, (rx + v[0] * tca) ** 2 + (ry + v[1] * tca) ** 2))
    return (tca, dca, math.sqrt(vv))


def observe_only(cube, q0, max_steps):
    # Perception could not assign every role: hold the arm still and record a dynamics
    # timeline (no press) so the next version can be designed from evidence.
    report["phase"] = "observe_only"
    if cube is None:
        return
    c0 = cube["st"]["med"]
    if not ("target" in G):
        G["target"] = [c0[0], c0[1]]
    if not ("button" in G):
        G["button"] = [100.0, 100.0]
    G["r_in"] = 0.0; G["r_out"] = 0.0; G["rex"] = 0.0; G["rmax"] = 0.6; G["rbase"] = 0.2
    G["zlo"] = G["table_z"] + 0.012; G["zhi"] = G["table_z"] + 0.10
    G["grid"] = build_grid(2)
    S["last"] = (c0[0], c0[1])
    n = 200
    if max_steps is not None:
        n = min(n, max_steps - 5)
    for k in range(n):
        advance(q0)
        if S["terminal"]:
            return


def run_probes(prompts, base):
    probes = []
    cands = []
    for pr in prompts:
        entry = {"prompt": pr, "ok": False, "cands": []}
        try:
            seg = segment_text("front", pr)
            entry["ok"] = bool(seg.ok)
            if seg.ok:
                k = 0
                for c in seg.segmentations:
                    if k >= 5:
                        break
                    k += 1
                    info = {"score": r4(c.score)}
                    box = None
                    try:
                        box = [float(b) for b in c.box_xyxy]
                    except Exception:
                        box = None
                    info["box"] = [r4(b) for b in box] if box else None
                    st = None
                    try:
                        pc = mask_to_point_cloud(c, "front", target_frame=base)
                        st = cloud_stats(pc)
                    except Exception as e:
                        info["pc_err"] = str(e)[:200]
                    info["stats"] = round_stats(st)
                    entry["cands"].append(info)
                    if st is not None and box is not None:
                        cands.append({"prompt": pr, "score": float(c.score), "box": box, "st": st})
        except Exception as e:
            entry["err"] = str(e)[:200]
        probes.append(entry)
    return probes, cands


def main():
    ctx = get_task_context()
    lang = str(ctx.language)
    report["language"] = lang
    try:
        report["task_name"] = str(ctx.task_name)
    except Exception:
        pass
    max_steps = None
    try:
        meta = robomme_plus.get_task_metadata()
        md = {}
        for k in meta:
            md[str(k)] = str(meta[k])[:300]
        report["task_metadata"] = md
        max_steps = int(meta["max_steps"])
    except Exception as e:
        report["notes"].append("metadata: " + str(e)[:200])
    report["max_steps"] = max_steps
    try:
        cm = robomme_plus.get_controller_metadata()
        cmd = {}
        for k in cm:
            cmd[str(k)] = str(cm[k])[:300]
        report["controller_metadata"] = cmd
    except Exception as e:
        report["notes"].append("controller metadata: " + str(e)[:200])
    try:
        video = robomme_plus.get_conditioning_video()
        report["video_len"] = len(video)
        if len(video) > 0:
            report["video_t"] = [r4(video[0].timestamp_s), r4(video[len(video) - 1].timestamp_s)]
    except Exception as e:
        report["notes"].append("video: " + str(e)[:200])

    with_times, plain = parse_count(lang)
    report["count_words"] = {"with_times": with_times, "plain": plain}
    if len(with_times) > 0:
        N = with_times[0]
    elif len(plain) > 0:
        N = plain[0]
    else:
        N = 1
    report["N"] = N

    obs0 = get_observation()
    state0 = obs0.robot_state
    base = state0.base_frame
    G["emb"] = state0.embodiment
    report["cameras"] = [str(c) for c in obs0.cameras]
    report["embodiment"] = str(state0.embodiment)
    report["base_frame"] = str(base)
    cam = obs0.cameras["front"]
    K = cam.intrinsics
    G["fx"] = float(K[0][0]); G["fy"] = float(K[1][1])
    G["cx"] = float(K[0][2]); G["cy"] = float(K[1][2])
    cp = cam.camera_pose
    G["R"] = quat_to_rot(cp.quaternion_wxyz)
    G["t"] = [float(cp.position[0]), float(cp.position[1]), float(cp.position[2])]
    depth0 = cam.depth_m
    G["H"] = len(depth0); G["W"] = len(depth0[0])
    report["front_cam"] = {"K": [r4(G["fx"]), r4(G["fy"]), r4(G["cx"]), r4(G["cy"])],
                           "pos": [r4(v) for v in G["t"]],
                           "quat": [r4(v) for v in cp.quaternion_wxyz],
                           "frame": str(cp.frame), "H": G["H"], "W": G["W"]}
    ee0 = state0.end_effector_poses["primary"]
    report["ee0"] = {"pos": [r4(v) for v in ee0.position], "quat": [r4(v) for v in ee0.quaternion_wxyz]}
    q0 = [float(x) for x in state0.joint_positions["primary"]]
    report["q0"] = [r4(v) for v in q0]

    # ---- prompt probes ----
    prompts = ["cube", "block"] + lang_prompts(lang, "cube")
    prompts += ["target", "disc", "circle"]
    lw = lang.lower()
    for w in TARGET_WORDS:
        if w in lw and not (w in prompts):
            prompts.append(w)
        for lp in lang_prompts(lang, w):
            if not (lp in prompts):
                prompts.append(lp)
    prompts += ["button"] + lang_prompts(lang, "button")
    probes, cands = run_probes(prompts, base)
    report["probes"] = probes

    # ---- camera convention self-calibration ----
    scale_u = 1.0; scale_v = 1.0
    allsmall = True
    for c in cands:
        if max(c["box"]) > 1.5:
            allsmall = False
    if allsmall and len(cands) > 0:
        scale_u = float(G["W"]); scale_v = float(G["H"])
    errs = {"cv": [], "gl": [], "xf": []}
    for c in cands:
        b = c["box"]
        u = 0.5 * (b[0] + b[2]) * scale_u
        v = 0.5 * (b[1] + b[3]) * scale_v
        ui = min(G["W"] - 1, max(0, int(u))); vi = min(G["H"] - 1, max(0, int(v)))
        dep = float(depth0[vi][ui])
        if not (dep > 0.0 and dep < 10.0):
            continue
        for m in ["cv", "gl", "xf"]:
            p = deproject(u, v, dep, m)
            med = c["st"]["med"]
            errs[m].append(math.sqrt((p[0] - med[0]) ** 2 + (p[1] - med[1]) ** 2 + (p[2] - med[2]) ** 2))
    best = None
    besterr = None
    for m in ["cv", "gl", "xf"]:
        if len(errs[m]) == 0:
            continue
        e = quant(errs[m], 0.5)
        if besterr is None or e < besterr:
            besterr = e; best = m
    errs_out = {}
    for m in errs:
        errs_out[m] = [r4(x) for x in errs[m][:12]]
    report["calib"] = {"errs": errs_out,
                       "mode": best, "median_err": r4(besterr), "box_scale": [scale_u, scale_v]}
    if best is None:
        raise ValueError("camera calibration failed: no candidate with depth")
    G["mode"] = best

    # ---- table height from a coarse deprojected grid (mode of 1 cm bins) ----
    hist = {}
    zl = []
    for v in range(0, G["H"], 6):
        for u in range(0, G["W"], 6):
            dep = float(depth0[v][u])
            if not (dep > 0.0 and dep < 10.0):
                continue
            p = deproject(u, v, dep, best)
            if p[0] * p[0] + p[1] * p[1] < 0.2 * 0.2:
                continue
            key = int(math.floor(p[2] / 0.01))
            hist[key] = hist.get(key, 0) + 1
            zl.append(p[2])
    bestkey = None
    for key in hist:
        if bestkey is None or hist[key] > hist[bestkey]:
            bestkey = key
    near = [z for z in zl if abs(z - (bestkey + 0.5) * 0.01) < 0.01]
    table_z = quant(near, 0.5)
    report["table_z"] = r4(table_z)
    G["table_z"] = table_z

    # ---- role selection by geometry ----
    def span_xy(st):
        return max(st["hi"][0] - st["lo"][0], st["hi"][1] - st["lo"][1])

    def far_from_base(st):
        return st["med"][0] ** 2 + st["med"][1] ** 2 > 0.2 * 0.2

    def dist_xy(p, q):
        return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

    def is_cube_prompt(c):
        return "cube" in c["prompt"] or "block" in c["prompt"]

    target = None
    for c in cands:
        istarget = False
        for w in TARGET_WORDS:
            if w in c["prompt"]:
                istarget = True
        if not istarget:
            continue
        st = c["st"]
        top = st["hi"][2] - table_z
        zspan = st["hi"][2] - st["lo"][2]
        if st["n"] >= 15 and zspan < 0.015 and top < 0.02 and 0.03 <= span_xy(st) <= 0.40 and far_from_base(st):
            if target is None or c["score"] > target["score"]:
                target = c
    ttop = table_z if target is None else target["st"]["hi"][2]
    tr = 0.0
    if target is not None:
        tst = target["st"]
        tr = 0.25 * ((tst["hi"][0] - tst["lo"][0]) + (tst["hi"][1] - tst["lo"][1]))

    def tall(st):
        top = st["hi"][2] - table_z
        return (st["n"] >= 8 and 0.012 <= top <= 0.10 and st["hi"][2] >= ttop + 0.008
                and span_xy(st) <= 0.10 and far_from_base(st))

    # motion check: hold the arm still, re-segment; the candidate that moved is the cube
    HOLD = 4
    hold_obs = []
    for k in range(HOLD):
        o = advance(q0, track=False)
        if o is None:
            return
        hold_obs.append((S["ticks"], o))
    cprompts = ["cube", "block"] + lang_prompts(lang, "cube") + ["button"]
    probes2, cands2 = run_probes(cprompts, base)
    tall2 = [c2 for c2 in cands2 if tall(c2["st"])]
    motion = []
    for c in cands:
        c["moved"] = None
        if not tall(c["st"]) or len(tall2) == 0:
            continue
        dmin = None
        for c2 in tall2:
            dd = dist_xy(c["st"]["topxy"], c2["st"]["topxy"])
            if dmin is None or dd < dmin:
                dmin = dd
        c["moved"] = dmin > 0.008
        motion.append([c["prompt"], r4(c["score"]), r4(dmin)])
    report["probes2"] = probes2
    report["motion_check"] = {"hold_ticks": HOLD, "pairs": motion,
                              "t2": [[c2["prompt"], r4(c2["score"]), [r4(v) for v in c2["st"]["topxy"]]] for c2 in tall2]}
    cube = None
    for c in cands:
        if is_cube_prompt(c) and tall(c["st"]) and c["moved"] == True:
            if cube is None or c["score"] > cube["score"]:
                cube = c
    cube_by = "motion"
    if cube is None:
        cube_by = "score_fallback"
        for c in cands:
            if is_cube_prompt(c) and tall(c["st"]):
                if cube is None or c["score"] > cube["score"]:
                    cube = c
    report["cube_selected_by"] = cube_by
    # the cube's current (post-hold) position, for excluding it from button candidates
    cube_now = None
    if cube is not None:
        cube_now = cube["st"]["topxy"]
        if cube_by == "motion":
            best2 = None
            for c2 in tall2:
                if is_cube_prompt(c2) and (best2 is None or c2["score"] > best2["score"]):
                    moved2 = True
                    for c in cands:
                        if c["moved"] == False and dist_xy(c["st"]["topxy"], c2["st"]["topxy"]) <= 0.008:
                            moved2 = False
                    if moved2:
                        best2 = c2
            if best2 is not None:
                cube_now = best2["st"]["topxy"]

    def button_ok(c):
        st = c["st"]
        if not tall(st) or c["moved"] == True:
            return False
        if target is not None and dist_xy(st["topxy"], target["st"]["med"]) < tr + 0.02:
            return False
        if cube is not None:
            if dist_xy(st["topxy"], cube["st"]["topxy"]) < 0.04:
                return False
            if cube_now is not None and dist_xy(st["topxy"], cube_now) < 0.04:
                return False
        return True

    button = None
    button_by = "button_prompt"
    for c in cands:
        if "button" in c["prompt"] and button_ok(c):
            if button is None or c["score"] > button["score"]:
                button = c
    if button is None:
        button_by = "static_tall_fallback"
        for c in cands:
            if c["moved"] == False and button_ok(c):
                if button is None or c["score"] > button["score"]:
                    button = c
    report["button_selected_by"] = button_by
    sel = {}
    for name, c in [["cube", cube], ["target", target], ["button", button]]:
        sel[name] = None if c is None else {"prompt": c["prompt"], "score": r4(c["score"]),
                                             "box": [r4(b) for b in c["box"]], "stats": round_stats(c["st"])}
    report["selected"] = sel
    if cube is None or target is None or button is None:
        report["notes"].append("role selection incomplete (cube,target,button missing): " +
                               str([cube is None, target is None, button is None]))
        observe_only(cube, q0, max_steps)
        return

    G["button"] = [button["st"]["topxy"][0], button["st"]["topxy"][1]]
    btop = button["st"]["hi"][2]
    G["target"] = [target["st"]["med"][0], target["st"]["med"][1]]
    G["target_r"] = tr
    G["r_in"] = max(tr, 0.02)
    G["r_out"] = tr + 0.03
    c0 = [cube_now[0], cube_now[1], cube["st"]["med"][2]]
    dbc = math.sqrt((G["button"][0] - c0[0]) ** 2 + (G["button"][1] - c0[1]) ** 2)
    dbt = math.sqrt((G["button"][0] - G["target"][0]) ** 2 + (G["button"][1] - G["target"][1]) ** 2)
    G["rex"] = max(0.05, min(0.12, 0.6 * min(dbc, dbt)))
    G["rmax"] = 0.5
    G["rbase"] = 0.2
    G["zlo"] = max(table_z + 0.012, target["st"]["hi"][2] + 0.008)
    G["zhi"] = table_z + 0.10
    report["geom"] = {"button_xy": [r4(v) for v in G["button"]], "button_top": r4(btop),
                      "target_xy": [r4(v) for v in G["target"]], "target_r": r4(tr),
                      "r_in": r4(G["r_in"]), "r_out": r4(G["r_out"]), "rex": r4(G["rex"]),
                      "zlo": r4(G["zlo"]), "zhi": r4(G["zhi"]),
                      "cube0": [r4(v) for v in c0], "d_cube0_target": r4(math.sqrt((c0[0] - G["target"][0]) ** 2 + (c0[1] - G["target"][1]) ** 2))}

    G["grid"] = build_grid(2)
    S["last"] = (cube["st"]["topxy"][0], cube["st"]["topxy"][1])
    p0 = track_depth(obs0)
    report["track0"] = None if p0 is None else [r4(p0[0]), r4(p0[1]), p0[2], p0[3]]
    S["start_in_zone"] = False
    if p0 is not None:
        d0 = math.sqrt((p0[0] - G["target"][0]) ** 2 + (p0[1] - G["target"][1]) ** 2)
        if d0 < G["r_in"]:
            S["in_zone"] = True
            S["start_in_zone"] = True
    report["start_in_zone"] = S["start_in_zone"]
    # replay the held ticks through the tracker so no arrival during the hold is missed
    for item in hold_obs:
        update(item[1], item[0])

    # ---- IK for pre-hover, calibration, hover, press ----
    quat = [float(v) for v in ee0.quaternion_wxyz]
    rq = quat_to_rot(quat)
    if rq[2][2] > -0.9:
        quat = [0.0, 1.0, 0.0, 0.0]
        report["notes"].append("home EE not downward; using canonical top-down quaternion")
    bx = G["button"][0]; by = G["button"][1]

    def ik(z, seed):
        r = solve_ik(Pose([bx, by, z], quat, base), seed_joints=seed)
        if not r.ok:
            report["notes"].append("ik failed at z=" + str(r4(z)))
            return None
        return [float(x) for x in r.joint_positions]

    q_tr = ik(btop + TRANSIT_H, q0)
    if q_tr is None:
        raise ValueError("ik transit failed")
    q_pre = ik(btop + HOVER + HOVER + PRESS_DEPTH, q_tr)
    if q_pre is None:
        raise ValueError("ik pre-hover failed")
    q_hover = ik(btop + HOVER, q_pre)
    q_press = ik(btop - PRESS_DEPTH, q_hover if q_hover is not None else q_pre)
    if q_hover is None or q_press is None:
        raise ValueError("ik hover/press failed")

    if max_steps is None:
        max_steps = 600
    horizon = max_steps - 2
    report["phase"] = "approach"
    move_interp(q0, q_tr, 0.06)
    move_interp(q_tr, q_pre, 0.06)
    for k in range(3):
        advance(q_pre)
    if S["terminal"]:
        return

    # ---- press-latency calibration: pre-hover -> hover jump (same dz as the press) ----
    report["phase"] = "latency_calibration"
    obs = get_observation(camera_names=["front"])
    z_start = ee_pos(obs)[2]
    zs = []
    L = None
    for k in range(10):
        o = advance(q_hover)
        if o is None:
            return
        z = ee_pos(o)[2]
        zs.append(r4(z))
        if L is None and z <= z_start - (HOVER + REG_DEPTH):
            L = k + 1
        if L is not None and k >= L + 3:
            break
    if L is None:
        L = 6
        report["notes"].append("latency calibration did not reach threshold; L=6 assumed")
    report["latency"] = {"L": L, "z_start": r4(z_start), "zs": zs, "ticks_at_end": S["ticks"]}
    if S["terminal"]:
        return
    S["armed"] = True
    report["armed_tick"] = S["ticks"]
    report["visits_at_armed"] = S["visits"]
    report["phase"] = "watch"

    # ---- watch, count, trigger ----
    trig = None
    while not S["terminal"] and S["ticks"] < horizon:
        ca = closest_approach()
        left = horizon - S["ticks"]
        nth = (S["visits"] == N - 1 and not S["in_zone"]) or (S["visits"] == N and S["in_zone"])
        if ca is not None and nth:
            if ca[0] > 0 and ca[0] <= L + 0.5 and ca[1] < G["r_in"]:
                trig = {"why": "lead", "tca": r4(ca[0]), "dca": r4(ca[1]), "speed": r4(ca[2])}
        if trig is None and S["visits"] == N and S["in_zone"] and (ca is None or ca[0] <= 0):
            trig = {"why": "in_zone_fallback"}
            if ca is not None:
                trig["tca"] = r4(ca[0]); trig["dca"] = r4(ca[1]); trig["speed"] = r4(ca[2])
        if trig is None and left < 8:
            trig = {"why": "horizon_fallback"}
        if trig is not None:
            break
        if S["sam_checks"] < 8 and S["ticks"] % 50 == 0 and S["visits"] != N - 1:
            S["sam_checks"] += 1
            try:
                lo = localize_object(cube["prompt"], camera_name="front", target_frame=base)
                if lo.ok:
                    gp = lo.geometry.pose.position
                    report.setdefault("sam_checks", []).append(
                        [S["ticks"], r4(gp[0]), r4(gp[1]), r4(gp[2]), S["last"][0] if S["last"] else None])
            except Exception as e:
                report["notes"].append("sam check: " + str(e)[:120])
        advance(q_hover)
    if S["terminal"]:
        report["phase"] = "terminated_before_press"
        return
    trig["tick"] = S["ticks"]
    trig["visits"] = S["visits"]
    report["trigger"] = trig
    S["pressed"] = True
    S["press_tick"] = S["ticks"]
    report["phase"] = "press"
    for k in range(40):
        if S["terminal"] or S["ticks"] >= max_steps - 1:
            break
        advance(q_press)
    report["phase"] = "done"


try:
    main()
except Exception as e:
    report["error"] = str(e)[:500]

report["ticks"] = S["ticks"]
report["terminal"] = S["terminal"]
report["visits"] = S["visits"]
report["visit_ticks"] = S["visit_ticks"]
report["visit_dmin"] = S["visit_dmin"]
report["lost_ticks"] = S["lost"]
report["press_tick"] = S["press_tick"]
result = report
