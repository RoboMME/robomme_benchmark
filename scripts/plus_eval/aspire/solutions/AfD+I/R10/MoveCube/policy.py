# MoveCube v06 -- final-placement accuracy (closed-loop finish).
# v05 (ep0): route planner avoided all distractors (no termination): pull +y, then shaft-side push
# along -x; the cube passed within 4 mm of the target centre but came to rest 1.3 cm beyond it
# (overshoot along the push direction), not success; a 1.3 cm correction route was infeasible.
# v06 (one change): the final leg stops 2.5 cm short of the target, the cube is re-observed, and the
# finish is a short push with the SAME tool face and orientation as that leg (hand stays next to its
# already-verified poses), aimed short by the overshoot measured on the stop-short leg, repeated until
# the cube centre is within 6 mm; never execute an infeasible plan for a < 3 cm correction. Planner otherwise as v05:
import math

report = {"version": "v06", "errors": [], "motions": []}
ctx = get_task_context()
report["language"] = ctx.language
max_steps = 1339
try:
    meta = robomme_plus.get_task_metadata()
    max_steps = int(meta["max_steps"])
except Exception as exc:
    report["errors"].append("task_metadata: " + str(exc)[:200])
report["max_steps"] = max_steps

state0 = get_robot_state()
base_frame = state0.base_frame
home_pose = state0.end_effector_poses["primary"]
report["home_ee"] = [round(float(v), 4) for v in home_pose.position]


# ---------------------------------------------------------------- small math
def q(vals, f):
    s = sorted(vals)
    return s[int((len(s) - 1) * f)]


def quat_to_matrix(qq):
    w, x, y, z = float(qq[0]), float(qq[1]), float(qq[2]), float(qq[3])
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def yaw_quat(beta):
    # top-down tool frame: Rz(beta) * Rx(pi); finger axis (ee y) points at angle beta - 90 deg
    return quat_mul([math.cos(beta / 2), 0.0, 0.0, math.sin(beta / 2)], [0.0, 1.0, 0.0, 0.0])


def wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


Rh = quat_to_matrix(home_pose.quaternion_wxyz)
beta_home = math.atan2(Rh[1][1], Rh[0][1]) + math.pi / 2
report["beta_home_deg"] = round(math.degrees(beta_home), 1)


# ---------------------------------------------------------------- pixels
def to_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        return [[[int(c) for c in px] for px in row] for row in arr]


def colour_class(r, g, b):
    if r > 150 and g > 120 and b < 95 and g - b > 60 and r - g < 60:
        return "Y"   # yellow ball
    if r > 140 and b > 80 and 40 <= r - g <= 80 and 8 <= g - b <= 35:
        return "P"   # salmon/pink tool end (lit and shaded faces)
    if 50 < r < 115 and g >= r and b >= g and 5 < b - r < 30:
        return "S"   # dark slate (hammer head)
    if g > 70 and g - r > 30 and b - r > 30 and abs(g - b) < 12:
        return "T"   # teal tool handle
    if r > 150 and g < 60 and b < 60:
        return "R"   # red cube
    if b > 150 and r > 120 and g < 0.8 * r and b > g + 50:
        return "V"   # violet target ring
    if r > 170 and 22 <= r - g <= 50 and g - b > 30 and b > 90:
        return "W"   # tan wooden hammer handle
    if r > 150 and g > 150 and b > 150 and b >= r - 5:
        return "G"   # light grey shaft
    return None


def class_counts(rows, step):
    counts = {}
    for v in range(step // 2, len(rows), step):
        row = rows[v]
        for u in range(step // 2, len(row), step):
            px = row[u]
            k = colour_class(px[0], px[1], px[2])
            if k is not None:
                counts[k] = counts.get(k, 0) + 1
    return counts


def label_image(rows):
    labels = {}
    for v in range(len(rows)):
        row = rows[v]
        for u in range(len(row)):
            px = row[u]
            k = colour_class(px[0], px[1], px[2])
            if k is not None:
                labels[(u, v)] = k
    return labels


def clusters_of(labels, key, reach):
    pts = [p for p in labels if labels[p] == key]
    member = {p: True for p in pts}
    seen = {}
    out = []
    for start in pts:
        if start in seen:
            continue
        seen[start] = True
        stack = [start]
        group = []
        while stack:
            cur = stack.pop()
            group.append(cur)
            for du in range(-reach, reach + 1):
                for dv in range(-reach, reach + 1):
                    nb = (cur[0] + du, cur[1] + dv)
                    if nb in member and nb not in seen:
                        seen[nb] = True
                        stack.append(nb)
        out.append(group)
    return sorted(out, key=lambda g: -len(g))


def camera_model(cam):
    K = cam.intrinsics
    R = quat_to_matrix(cam.camera_pose.quaternion_wxyz)
    t = [float(c) for c in cam.camera_pose.position]
    return {"fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]), "cy": float(K[1][2]),
            "R": R, "t": t, "depth": to_rows(cam.depth_m)}


def backproject(model, u, v):
    d = float(model["depth"][v][u])
    xc = (u + 0.5 - model["cx"]) * d / model["fx"]
    yc = (v + 0.5 - model["cy"]) * d / model["fy"]
    R = model["R"]
    t = model["t"]
    return [R[i][0] * xc + R[i][1] * yc + R[i][2] * d + t[i] for i in range(3)]


def summarize3d(model, pixels):
    pts = [backproject(model, p[0], p[1]) for p in pixels]
    pts = [p for p in pts if -0.1 < p[2] < 0.4]
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    c = [q(xs, 0.5), q(ys, 0.5), q(zs, 0.5)]
    # principal extents in xy
    best = None
    for deg in range(0, 180, 10):
        a = math.radians(deg)
        ca, sa = math.cos(a), math.sin(a)
        us = [p[0] * ca + p[1] * sa for p in pts]
        span = q(us, 0.97) - q(us, 0.03)
        if best is None or span > best[0]:
            best = (span, a)
    a = best[1] + math.pi / 2
    ws = [p[0] * math.cos(a) + p[1] * math.sin(a) for p in pts]
    width = q(ws, 0.97) - q(ws, 0.03)
    return {"c": c, "z_hi": q(zs, 0.9), "z_lo": q(zs, 0.05), "len": best[0], "wid": width,
            "axis": best[1], "n": len(pts), "pts": pts}


def line_coverage(labels, a, b):
    hits = 0
    total = 0
    for k in range(1, 14):
        f = k / 14.0
        u = int(round(a[0] + (b[0] - a[0]) * f))
        v = int(round(a[1] + (b[1] - a[1]) * f))
        total += 1
        found = False
        for du in (-3, -2, -1, 0, 1, 2, 3):
            for dv in (-3, -2, -1, 0, 1, 2, 3):
                if labels.get((u + du, v + dv)) in ("G", "T", "P", "W"):
                    found = True
        if found:
            hits += 1
    return hits / float(total)


def pix_center(group):
    us = sorted(p[0] for p in group)
    vs = sorted(p[1] for p in group)
    return [us[len(us) // 2], vs[len(vs) // 2]]


def scene_objects(cam):
    rows = to_rows(cam.rgb)
    model = camera_model(cam)
    labels = label_image(rows)
    objs = {"tools": [], "hammer": None, "ball": None}
    teal = [g for g in clusters_of(labels, "T", 1) if len(g) >= 6]
    pink = [g for g in clusters_of(labels, "P", 2) if len(g) >= 4]
    pinfo = []
    for g in pink:
        s = summarize3d(model, g)
        if s is not None:
            pinfo.append((g, s))
    for tg in teal[:6]:
        hs = summarize3d(model, tg)
        if hs is None:
            continue
        hp = pix_center(tg)
        best = None
        for g, s in pinfo:
            dist = math.sqrt((s["c"][0] - hs["c"][0]) ** 2 + (s["c"][1] - hs["c"][1]) ** 2)
            if dist < 0.08 or dist > 0.5:
                continue
            cov = line_coverage(labels, hp, pix_center(g))
            if best is None or cov > best[0]:
                best = (cov, g, s, dist)
        if best is None or best[0] < 0.6:
            continue
        s = best[2]
        objs["tools"].append({"handle": hs["c"], "handle_z_hi": hs["z_hi"], "handle_axis": hs["axis"],
                              "end": s["c"], "end_len": s["len"], "end_wid": s["wid"],
                              "length": best[3], "coverage": best[0], "handle_px": hp,
                              "end_px": pix_center(best[1]), "end_z_hi": s["z_hi"], "pts": s["pts"]})
    slate = [g for g in clusters_of(labels, "S", 1) if len(g) >= 8]
    tan = [g for g in clusters_of(labels, "W", 1) if len(g) >= 4]
    for sg in slate[:4]:
        hs = summarize3d(model, sg)
        if hs is None or hs["c"][2] > 0.15 or hs["c"][2] < -0.03 or hs["len"] > 0.2:
            continue   # skip floor / background beyond the table
        far = None
        for g in tan:
            ts = summarize3d(model, g)
            if ts is None:
                continue
            for p in ts["pts"]:
                dd = math.sqrt((p[0] - hs["c"][0]) ** 2 + (p[1] - hs["c"][1]) ** 2)
                if dd < 0.4 and (far is None or dd > far[0]):
                    far = (dd, p)
        if far is not None and far[0] > 0.08:
            objs["hammer"] = {"handle": hs["c"], "handle_z_hi": hs["z_hi"], "handle_axis": hs["axis"],
                              "end": far[1], "end_len": 0.02, "end_wid": 0.02, "length": far[0],
                              "end_z_hi": far[1][2]}
            break
    yel = [g for g in clusters_of(labels, "Y", 1) if len(g) >= 4]
    if yel:
        ys = summarize3d(model, yel[0])
        if ys is not None:
            objs["ball"] = {"c": ys["c"], "z_hi": ys["z_hi"]}
    for key in ("R", "V"):
        cl = clusters_of(labels, key, 1)
        if cl and len(cl[0]) >= 3:
            s = summarize3d(model, cl[0])
            if s is not None:
                objs[key] = {"c": s["c"], "z_hi": s["z_hi"], "z_lo": s["z_lo"], "px": pix_center(cl[0]),
                             "len": s["len"], "wid": s["wid"]}
    objs["counts"] = {k: 0 for k in "YPSTRVWG"}
    for p in labels:
        objs["counts"][labels[p]] += 1
    return objs


def brief_tool(t):
    out = {}
    for k in t:
        v = t[k]
        if k == "pts":
            continue
        if isinstance_number(v):
            out[k] = round(v, 4)
        elif type_list(v):
            out[k] = [round(float(c), 4) for c in v]
        else:
            out[k] = v
    return out


def isinstance_number(v):
    try:
        return float(v) == v and not (v is True or v is False)
    except Exception:
        return False


def type_list(v):
    try:
        return len(v) > 0 and float(v[0]) == float(v[0])
    except Exception:
        return False


# ---------------------------------------------------------------- 1. conditioning video
video_info = {}
wrist_manner = "unknown"
video_tool = None
video_kind = "unknown"
try:
    video = robomme_plus.get_conditioning_video()
    n = len(video)
    video_info["frames"] = n
    stride = max(1, n // 200)
    idxs = list(range(0, n, stride))
    grips = []
    ees = []
    for i in idxs:
        rs = video[i].robot_state
        grips.append(float(rs.gripper_positions["primary"]))
        p = rs.end_effector_poses["primary"].position
        ees.append([float(p[0]), float(p[1]), float(p[2])])
    gmax = max(grips) if grips else 1.0
    closed = [g < 0.6 * gmax for g in grips]
    hold = [k for k in range(len(idxs)) if closed[k]]
    video_info["grip_series"] = [round(g, 2) for g in grips[::3]]
    video_info["hold_samples"] = len(hold)
    if hold:
        first = hold[0]
        last = hold[-1]
        video_info["hold_frames"] = [idxs[first], idxs[last]]
        gpos = ees[first]
        video_info["ee_at_grasp"] = [round(c, 3) for c in gpos]
        tallies = {}
        span = max(1, (last - first) // 6)
        for k in range(first + span, last + 1, span):
            if closed[k]:
                c = class_counts(to_rows(video[idxs[k]].cameras["wrist"].rgb), 8)
                for key in c:
                    tallies.setdefault(key, []).append(c[key])

        def med(key):
            vals = sorted(tallies.get(key, []))
            return vals[len(vals) // 2] if vals else 0

        wm = {key: med(key) for key in ("Y", "P", "S", "T")}
        video_info["wrist_median"] = wm
        if wm["Y"] > 40:
            wrist_manner = "ball"
        elif wm["P"] > 60:
            wrist_manner = "square"
        elif wm["S"] > 50:
            wrist_manner = "hammer"
        else:
            wrist_manner = "stick"
        # identity of the grasped object in video frame 0
        vobj = scene_objects(video[0].cameras["front"])
        cands = []
        for t in vobj["tools"]:
            dist = math.sqrt((t["handle"][0] - gpos[0]) ** 2 + (t["handle"][1] - gpos[1]) ** 2)
            cands.append((dist, "tool", t))
        if vobj["hammer"] is not None:
            t = vobj["hammer"]
            dist = math.sqrt((t["handle"][0] - gpos[0]) ** 2 + (t["handle"][1] - gpos[1]) ** 2)
            cands.append((dist, "hammer", t))
        if vobj["ball"] is not None:
            b = vobj["ball"]["c"]
            cands.append((math.sqrt((b[0] - gpos[0]) ** 2 + (b[1] - gpos[1]) ** 2), "ball", vobj["ball"]))
        cands = sorted(cands, key=lambda c: c[0])
        video_info["frame0_candidates"] = [[round(c[0], 3), c[1]] for c in cands[:6]]
        video_info["frame0_tools"] = [brief_tool(t) for t in vobj["tools"]]
        if cands:
            video_kind = cands[0][1]
            video_tool = cands[0][2]
            video_info["grasped_distance"] = round(cands[0][0], 3)
except Exception as exc:
    report["errors"].append("video: " + str(exc)[:300])
report["video"] = video_info
report["wrist_manner"] = wrist_manner
report["video_kind"] = video_kind
if video_kind == "tool" and video_tool is not None:
    report["video_tool"] = brief_tool(video_tool)

# ---------------------------------------------------------------- 2. live scene
obs = get_observation()
front = obs.cameras["front"]
live = scene_objects(front)
report["live_counts"] = live["counts"]
report["live_tools"] = [brief_tool(t) for t in live["tools"]]
if live["hammer"] is not None:
    report["live_hammer"] = brief_tool(live["hammer"])
if live["ball"] is not None:
    report["live_ball"] = [round(c, 4) for c in live["ball"]["c"]]


def seg_box_pick(prompts, seed_px):
    for prompt in prompts:
        try:
            s = segment_text("front", prompt)
        except Exception as exc:
            report["errors"].append("seg " + prompt + ": " + str(exc)[:100])
            continue
        if not s.ok:
            continue
        for c in s.segmentations[:5]:
            if c.box_xyxy is None:
                continue
            b = [float(x) for x in c.box_xyxy]
            if b[2] - b[0] > 60 or b[3] - b[1] > 60:
                continue
            if seed_px is None or (b[0] - 3 <= seed_px[0] <= b[2] + 3 and b[1] - 3 <= seed_px[1] <= b[3] + 3):
                return c
    if seed_px is not None:
        s = segment_points("front", [[seed_px[0], seed_px[1]]])
        if s.ok and len(s.segmentations) > 0:
            return s.segmentations[0]
    return None


def localize(prompts, key):
    seed = live.get(key)
    seed_px = seed["px"] if seed is not None else None
    c = seg_box_pick(prompts, seed_px)
    if c is None:
        return None
    cloud = mask_to_point_cloud(c, "front", target_frame=base_frame)
    pts = [[float(p[0]), float(p[1]), float(p[2])] for p in cloud.points]
    if len(pts) < 5:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    return {"c": [0.5 * (q(xs, 0.05) + q(xs, 0.95)), 0.5 * (q(ys, 0.05) + q(ys, 0.95)), q(zs, 0.5)],
            "z_hi": q(zs, 0.95), "z_lo": q(zs, 0.05), "n": len(pts)}


cube = localize(["red cube", "small red block"], "R")
target = localize(["purple and white target", "purple bullseye", "purple circle"], "V")
if cube is not None:
    report["cube"] = {"c": [round(v, 4) for v in cube["c"]], "z_hi": round(cube["z_hi"], 4),
                      "z_lo": round(cube["z_lo"], 4)}
    if live.get("R") is not None:
        report["cube_backprojected"] = [round(v, 4) for v in live["R"]["c"]]
if target is not None:
    report["target"] = {"c": [round(v, 4) for v in target["c"]], "z_hi": round(target["z_hi"], 4)}
    if live.get("V") is not None:
        report["target_backprojected"] = [round(v, 4) for v in live["V"]["c"]]

# ---------------------------------------------------------------- 3. choose the live tool
kind = video_kind
if video_info.get("grasped_distance", 1.0) > 0.08:
    kind = "unknown"
if kind == "unknown":
    kind = {"ball": "ball", "hammer": "hammer"}.get(wrist_manner, "tool")
tool = None
if kind == "tool" and live["tools"]:
    if video_tool is not None:
        best = None
        for t in live["tools"]:
            dd = abs(t["end_len"] - video_tool["end_len"]) + abs(t["end_wid"] - video_tool["end_wid"]) + \
                0.5 * abs(t["length"] - video_tool["length"])
            if best is None or dd < best[0]:
                best = (dd, t)
        tool = best[1]
        report["tool_match_cost"] = round(best[0], 4)
    else:
        tool = live["tools"][0]
elif kind == "hammer" and live["hammer"] is not None:
    tool = live["hammer"]
elif live["tools"]:
    tool = live["tools"][0]
    report["errors"].append("manner " + kind + " not implemented; falling back to a teal-handled tool")
report["chosen_kind"] = kind
if tool is not None:
    report["chosen_tool"] = brief_tool(tool)

# ---------------------------------------------------------------- 4. manipulation
ticks = {"used": 0}
ended = {"flag": False}


def move(pos, quat, label, tol=0.01, steps=150):
    if ended["flag"]:
        return False
    r = move_to_pose(Pose([float(pos[0]), float(pos[1]), float(pos[2])], quat, base_frame),
                     tolerance=tol, max_steps=steps)
    rec = {"label": label, "ok": bool(r.ok), "steps": int(r.steps_executed),
           "pos": [round(float(c), 3) for c in pos]}
    ticks["used"] += int(r.steps_executed)
    try:
        if bool(r.terminated) or bool(r.truncated):
            ended["flag"] = True
            rec["ended"] = True
    except Exception:
        pass
    if not r.ok:
        try:
            rec["error"] = str(r.error.message)[:150]
        except Exception:
            pass
    report["motions"].append(rec)
    return bool(r.ok)


def gripper_call(fn, label):
    if ended["flag"]:
        return None
    r = fn()
    rec = {"label": label, "ok": bool(r.ok), "steps": int(r.steps_executed)}
    ticks["used"] += int(r.steps_executed)
    try:
        if bool(r.terminated) or bool(r.truncated):
            ended["flag"] = True
            rec["ended"] = True
    except Exception:
        pass
    try:
        rec["grip"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
    except Exception:
        pass
    report["motions"].append(rec)
    return rec


def rot(vec, a):
    ca, sa = math.cos(a), math.sin(a)
    return [vec[0] * ca - vec[1] * sa, vec[0] * sa + vec[1] * ca]


def plan_push(C, T, off, beta, tip_half_len, tip_half_wid, cube_half):
    dx, dy = T[0] - C[0], T[1] - C[1]
    D = math.sqrt(dx * dx + dy * dy)
    if D < 1e-6:
        return None
    u = [dx / D, dy / D]
    phi_u = math.atan2(u[1], u[0])
    phi_o = math.atan2(off[1], off[0])
    options = []
    for extra, half in ((0.0, tip_half_len), (math.pi / 2, tip_half_wid), (-math.pi / 2, tip_half_wid)):
        delta = wrap(phi_u + extra - phi_o)
        for turn in (0.0, 2 * math.pi, -2 * math.pi):
            nb = beta + delta + turn
            if abs(nb - beta_home) > math.radians(150):
                continue
            o2 = rot(off, delta)
            clear = cube_half + half + 0.012
            p0 = [C[0] - u[0] * (clear + 0.02), C[1] - u[1] * (clear + 0.02)]
            p1 = [T[0] - u[0] * (clear - 0.006), T[1] - u[1] * (clear - 0.006)]
            s0 = [p0[0] - o2[0], p0[1] - o2[1]]
            s1 = [p1[0] - o2[0], p1[1] - o2[1]]
            worst = max(math.sqrt(s0[0] ** 2 + s0[1] ** 2), math.sqrt(s1[0] ** 2 + s1[1] ** 2))
            if worst < 0.68 and min(s0[0], s1[0]) > 0.15:
                options.append({"beta": nb, "delta": delta, "s0": s0, "s1": s1, "off": o2, "extra": extra,
                                "D": D, "worst_r": worst})
                break
    options = sorted(options, key=lambda o: o["worst_r"])
    return options[0] if options else None


def hook_geometry(tool):
    """Shaft direction, prong side/length and end extent of a teal-handled tool, from its pink points
    (live front RGB-D), in a frame anchored at the handle (grasp point). Tool frame: s along the shaft
    (handle -> end), l lateral (left of s)."""
    H = tool["handle"]
    E = tool["end"]
    pts = tool.get("pts") or []
    a0 = math.atan2(E[1] - H[1], E[0] - H[0])
    if len(pts) < 6:
        return None
    s0 = [(p[0] - H[0]) * math.cos(a0) + (p[1] - H[1]) * math.sin(a0) for p in pts]
    lim = q(s0, 0.25)
    near = [pts[i] for i in range(len(pts)) if s0[i] <= lim]
    nx = q([p[0] for p in near], 0.5)
    ny = q([p[1] for p in near], 0.5)
    a = math.atan2(ny - H[1], nx - H[0])
    ca, sa = math.cos(a), math.sin(a)
    ss = [(p[0] - H[0]) * ca + (p[1] - H[1]) * sa for p in pts]
    ls = [-(p[0] - H[0]) * sa + (p[1] - H[1]) * ca for p in pts]
    lnear = [abs(ls[i]) for i in range(len(pts)) if s0[i] <= lim]
    w = min(0.025, max(0.008, 2.0 * q(lnear, 0.9)))
    hi = q(ls, 0.95)
    lo = q(ls, 0.05)
    sigma = 1.0 if hi > -lo else -1.0
    Lp = hi if sigma > 0 else -lo
    s_far = q(ss, 0.97)
    shape = [[ss[i], ls[i]] for i in range(0, len(pts), 3)]
    sv = 0.06
    while sv < s_far - 0.02:
        shape.append([sv, 0.0])
        sv += 0.025
    return {"axis": a, "sigma": sigma, "prong": Lp, "s_far": s_far, "w": w, "shape": shape,
            "hook": Lp > 0.5 * w + 0.02}


def pull_plan(C, T, geo, beta_now, axis_now, cube_half):
    """Hook pull (as in the video): shaft pointing away from the target, prong beyond the cube, pull the
    tool toward the target so the prong's inner face drags the cube. Returns ee waypoints."""
    dx, dy = T[0] - C[0], T[1] - C[1]
    D = math.sqrt(dx * dx + dy * dy)
    if D < 1e-6:
        return None
    u = [dx / D, dy / D]
    phi_u = math.atan2(u[1], u[0])
    sig = geo["sigma"]
    # cube centre in the pocket: against the prong inner face (prong thickness <= 12 mm), laterally
    # centred between the shaft edge and the prong extent
    s_c = geo["s_far"] - min(geo["w"], 0.012) - cube_half - 0.002
    l_c = max(0.5 * geo["w"] + cube_half + 0.002, 0.5 * (0.5 * geo["w"] + geo["prong"]))
    best = None
    # tilt keeps the tool velocity component toward the prong side (cube slides into the corner)
    for tilt_deg in (0, -5, 5, -10, 10, -15, 15, -20, 20, -25, 25, -30, 30):
        tilt = math.radians(tilt_deg)
        shaft = phi_u + math.pi + sig * tilt
        delta = wrap(shaft - axis_now)
        for turn in (0.0, 2 * math.pi, -2 * math.pi):
            nb = beta_now + delta + turn
            if abs(nb - beta_home) > math.radians(150):
                continue
            ca, sa = math.cos(shaft), math.sin(shaft)

            def world(sv, lv):
                return [sv * ca - lv * sa, sv * sa + lv * ca]
            o_start = world(s_c - 0.035, sig * l_c)
            o_end = world(s_c - 0.004, sig * l_c)
            e0 = [C[0] - o_start[0] + 0.0, C[1] - o_start[1]]
            e1 = [T[0] - o_end[0], T[1] - o_end[1]]
            radii = [math.sqrt(e[0] ** 2 + e[1] ** 2) for e in (e0, e1)]
            mids = [[e0[0] + (e1[0] - e0[0]) * f, e0[1] + (e1[1] - e0[1]) * f] for f in (0.25, 0.5, 0.75)]
            radii = radii + [math.sqrt(e[0] ** 2 + e[1] ** 2) for e in mids]
            if max(radii) > 0.66 or min(radii) < 0.29 or min(e0[0], e1[0]) < 0.15:
                continue
            tool_clear = 1.0
            hand_clear = 1.0
            for f in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
                ex = e0[0] + (e1[0] - e0[0]) * f
                ey = e0[1] + (e1[1] - e0[1]) * f
                for ob in obstacles:
                    dh = math.sqrt((ob[0] - ex) ** 2 + (ob[1] - ey) ** 2)
                    if dh < hand_clear:
                        hand_clear = dh
                    if dh > 0.45:
                        continue
                    for sl in geo["shape"]:
                        w2 = world(sl[0], sl[1])
                        dt = math.sqrt((ob[0] - ex - w2[0]) ** 2 + (ob[1] - ey - w2[1]) ** 2)
                        if dt < tool_clear:
                            tool_clear = dt
            feasible = tool_clear >= 0.015 and hand_clear >= 0.045
            cost = abs(nb - beta_home) + 0.5 * abs(tilt) - 3.0 * min(hand_clear, 0.1)
            if not feasible:
                cost = 100.0 - 10.0 * min(tool_clear / 0.015, hand_clear / 0.045, 1.0)
            if best is None or cost < best["cost"]:
                best = {"beta": nb, "e0": e0, "e1": e1, "D": D, "tilt_deg": tilt_deg, "cost": cost,
                        "radii": [round(r, 3) for r in radii], "tool_clear": round(tool_clear, 3),
                        "hand_clear": round(hand_clear, 3), "feasible": feasible}
            break
    return best


if tool is not None and cube is not None and target is not None:
    table_z = min(target["c"][2], cube["z_lo"])
    cube_h = max(0.02, cube["z_hi"] - table_z)
    cube_half = 0.5 * cube_h
    H = tool["handle"]
    E = tool["end"]
    C = cube["c"]
    T = target["c"]
    phi_d = math.atan2(E[1] - H[1], E[0] - H[0])
    geo = None
    if kind == "tool":
        geo = hook_geometry(tool)
        vgeo = hook_geometry(video_tool) if video_tool is not None and video_kind == "tool" else None
        if geo is not None:
            report["live_tool_geometry"] = {k: (round(geo[k], 4) if k != "hook" else geo[k]) for k in geo if k != "shape"}
        if vgeo is not None:
            report["video_tool_geometry"] = {k: (round(vgeo[k], 4) if k != "hook" else vgeo[k]) for k in vgeo if k != "shape"}
        # pull only when the video's tool is a hook (and the live match is one too)
        if geo is not None and not (geo["hook"] and (vgeo is None or vgeo["hook"])):
            geo = None
    obstacles = []
    if geo is not None:
        try:
            om = camera_model(front)
            ex_pts = [p for p in tool["pts"]]
            grid = {}
            for v in range(0, len(om["depth"]), 2):
                for uu in range(0, len(om["depth"][0]), 2):
                    dval = float(om["depth"][v][uu])
                    if not (0.05 < dval < 3.0):
                        continue
                    p = backproject(om, uu, v)
                    if p[2] < table_z + 0.012 or p[2] > 0.3:
                        continue
                    if p[0] * p[0] + p[1] * p[1] < 0.18 * 0.18:
                        continue
                    if (p[0] - C[0]) ** 2 + (p[1] - C[1]) ** 2 < 0.035 ** 2:
                        continue
                    # points on the chosen tool (handle-to-end segment or near its pink end)
                    hx, hy = p[0] - H[0], p[1] - H[1]
                    ex_, ey_ = E[0] - H[0], E[1] - H[1]
                    L2 = ex_ * ex_ + ey_ * ey_ + 1e-9
                    tt = max(-0.1, min(1.2, (hx * ex_ + hy * ey_) / L2))
                    if (hx - tt * ex_) ** 2 + (hy - tt * ey_) ** 2 < 0.035 ** 2:
                        continue
                    near_tool = False
                    for tp in ex_pts:
                        if (p[0] - tp[0]) ** 2 + (p[1] - tp[1]) ** 2 < 0.03 ** 2:
                            near_tool = True
                            break
                    if near_tool:
                        continue
                    key = (int(round(p[0] / 0.01)), int(round(p[1] / 0.01)))
                    grid[key] = True
            obstacles = [[k[0] * 0.01, k[1] * 0.01] for k in grid]
        except Exception as exc:
            report["errors"].append("obstacles: " + str(exc)[:150])
        report["n_obstacle_cells"] = len(obstacles)
        if live["ball"] is not None:
            report["ball_obstacle_cells"] = len([o for o in obstacles if (o[0] - live["ball"]["c"][0]) ** 2 +
                                                 (o[1] - live["ball"]["c"][1]) ** 2 < 0.04 ** 2])
    # ---- v05 route planner for hook-shaped tools (pull / tip push / side push, 1-2 legs) ----
    obs2 = {}
    for ob in obstacles:
        obs2[(int(round(ob[0] / 0.02)), int(round(ob[1] / 0.02)))] = True
    plan_obs = [[k[0] * 0.02, k[1] * 0.02] for k in obs2]
    report["n_plan_obstacles"] = len(plan_obs)
    shape_pts = []
    if geo is not None:
        shape_pts = [geo["shape"][i] for i in range(0, len(geo["shape"]), 2)]
    hand_reach = (0.29, 0.70)
    need = {"tool": 0.015, "hand": 0.035, "cube": cube_half + 0.02}
    finger_reach = 0.08

    def clear_of(px, py, limit):
        best = 1.0
        for ob in plan_obs:
            d = math.sqrt((ob[0] - px) ** 2 + (ob[1] - py) ** 2)
            if d < best:
                best = d
                if best < limit:
                    return best
        return best

    def leg_options(P, Q, beta_now, axis_now, soft, force=None):
        dx, dy = Q[0] - P[0], Q[1] - P[1]
        D = math.sqrt(dx * dx + dy * dy)
        if D < 0.005:
            return None
        dv = [dx / D, dy / D]
        phi = math.atan2(dv[1], dv[0])
        sig = geo["sigma"]
        w = geo["w"]
        s_pull = geo["s_far"] - min(w, 0.012) - cube_half - 0.002
        l_pull = sig * max(0.5 * w + cube_half + 0.002, 0.5 * (0.5 * w + geo["prong"]))
        s_tip = geo["s_far"] + cube_half + 0.004
        l_tip = sig * 0.5 * (geo["prong"] - 0.5 * w)
        s_side = geo["s_far"] - 0.06
        l_side = -sig * (0.5 * w + cube_half + 0.004)
        variants = []
        for t in (0, 10, -10, 20, -20):
            variants.append(("pull", phi + math.pi + sig * math.radians(t), s_pull, l_pull, t, 0.0))
        for t in (0, 10, -10):
            variants.append(("push_tip", phi + math.radians(t), s_tip, l_tip, t, 0.15))
        variants.append(("push_side", phi + sig * math.pi / 2, s_side, l_side, 0, 0.25))
        if force is not None:
            variants = [force]
        best = None
        for mode_name, psi, sk, lk, t, pen in variants:
            delta = wrap(psi - axis_now)
            nb = None
            for turn in (0.0, 2 * math.pi, -2 * math.pi):
                if abs(beta_now + delta + turn - beta_home) <= math.radians(150):
                    nb = beta_now + delta + turn
                    break
            if nb is None:
                continue
            ca, sa = math.cos(psi), math.sin(psi)
            ox = sk * ca - lk * sa
            oy = sk * sa + lk * ca
            e0 = [P[0] - ox - 0.03 * dv[0], P[1] - oy - 0.03 * dv[1]]
            e1 = [Q[0] - ox, Q[1] - oy]
            samples = [[e0[0] + (e1[0] - e0[0]) * f, e0[1] + (e1[1] - e0[1]) * f] for f in (0.0, 0.33, 0.67, 1.0)]
            okr = True
            for e in samples:
                r = math.sqrt(e[0] ** 2 + e[1] ** 2)
                if r < hand_reach[0] or r > hand_reach[1] or e[0] < 0.15:
                    okr = False
            if not okr:
                continue
            fa = nb - math.pi / 2
            tool_c = 1.0
            hand_c = 1.0
            cube_c = 1.0
            for k in range(len(samples)):
                e = samples[k]
                for tf in (-finger_reach, -0.5 * finger_reach, 0.0, 0.5 * finger_reach, finger_reach):
                    hc = clear_of(e[0] + tf * math.cos(fa), e[1] + tf * math.sin(fa), 0.0 if soft else need["hand"])
                    hand_c = min(hand_c, hc)
                for sl in shape_pts:
                    px = e[0] + sl[0] * ca - sl[1] * sa
                    py = e[1] + sl[0] * sa + sl[1] * ca
                    tool_c = min(tool_c, clear_of(px, py, 0.0 if soft else need["tool"]))
                    if k == 0:
                        dc = math.sqrt((px - P[0]) ** 2 + (py - P[1]) ** 2)
                        if dc < cube_half + 0.006:
                            tool_c = min(tool_c, 0.0)
                if not soft and (tool_c < need["tool"] or hand_c < need["hand"]):
                    break
            for f in (0.0, 0.5, 1.0):
                cube_c = min(cube_c, clear_of(P[0] + dx * f, P[1] + dy * f, 0.0 if soft else need["cube"]))
            feasible = tool_c >= need["tool"] and hand_c >= need["hand"] and cube_c >= need["cube"]
            margin = min(tool_c / need["tool"], hand_c / need["hand"], cube_c / need["cube"])
            cost = pen + 0.3 * abs(math.radians(t)) + 0.1 * abs(nb - beta_home)
            rank = cost if feasible else 100.0 - min(margin, 1.0) * 10.0
            if best is None or rank < best["rank"]:
                best = {"mode": mode_name, "beta": nb, "psi": psi, "e0": e0, "e1": e1, "D": D, "tilt": t,
                        "sk": sk, "lk": lk, "dir": dv,
                        "feasible": feasible, "rank": rank, "cost": cost, "P": P, "Q": Q,
                        "clear": [round(tool_c, 3), round(hand_c, 3), round(cube_c, 3)]}
        return best

    def plan_route(Cc, Tt, beta_now, axis_now):
        direct = leg_options(Cc, Tt, beta_now, axis_now, False)
        if direct is not None and direct["feasible"]:
            return {"legs": [direct], "cost": 1.0 + direct["cost"], "feasible": True}
        best = None
        for ang in range(0, 360, 30):
            for rad_m in (0.05, 0.09, 0.13, 0.17):
                M = [Cc[0] + rad_m * math.cos(math.radians(ang)), Cc[1] + rad_m * math.sin(math.radians(ang))]
                rm = math.sqrt(M[0] ** 2 + M[1] ** 2)
                if rm < 0.3 or rm > 0.62:
                    continue
                l1 = leg_options(Cc, M, beta_now, axis_now, False)
                if l1 is None or not l1["feasible"]:
                    continue
                l2 = leg_options(M, Tt, l1["beta"], axis_now + (l1["beta"] - beta_now), False)
                if l2 is None or not l2["feasible"]:
                    continue
                dm = math.sqrt((Tt[0] - M[0]) ** 2 + (Tt[1] - M[1]) ** 2)
                cost = 2.0 + l1["cost"] + l2["cost"] + 2.0 * (rad_m + dm)
                if best is None or cost < best["cost"]:
                    best = {"legs": [l1, l2], "cost": cost, "feasible": True}
        if best is not None:
            return best
        soft = leg_options(Cc, Tt, beta_now, axis_now, True)
        if soft is None:
            return None
        return {"legs": [soft], "cost": 100.0, "feasible": False}

    def brief_leg(lg):
        return {"mode": lg["mode"], "beta_deg": round(math.degrees(lg["beta"]), 1), "tilt": lg["tilt"],
                "e0": [round(c, 3) for c in lg["e0"]], "e1": [round(c, 3) for c in lg["e1"]],
                "Q": [round(c, 3) for c in lg["Q"]], "clear": lg["clear"], "feasible": lg["feasible"]}

    mode = "route" if geo is not None else "push"
    report["mode"] = mode
    if kind == "hammer":
        b0 = phi_d + math.pi / 2
    else:
        b0 = phi_d
    choices = [b0, b0 + math.pi, b0 - math.pi]
    choices = [b for b in choices if abs(b - beta_home) <= math.radians(150)]
    if mode == "route":
        scored = []
        for b in choices:
            rt = plan_route(C, T, b, geo["axis"])
            if rt is not None:
                scored.append((rt["cost"], abs(b - beta_home), b, rt))
        report["grasp_yaw_options"] = [[round(math.degrees(s[2]), 1), round(s[0], 3),
                                        [brief_leg(lg) for lg in s[3]["legs"]]] for s in scored]
        if scored:
            beta = sorted(scored, key=lambda s: (s[0], s[1]))[0][2]
        else:
            beta = sorted(choices, key=lambda b: abs(b - beta_home))[0]
            mode = "push"
            report["mode"] = "push (no route)"
    else:
        beta = sorted(choices, key=lambda b: abs(b - beta_home))[0]
    z_grasp = max(table_z + 0.015, tool["handle_z_hi"] - 0.01)
    report["grasp_plan"] = {"handle": [round(c, 4) for c in H], "z_grasp": round(z_grasp, 4),
                            "beta_deg": round(math.degrees(beta), 1), "table_z": round(table_z, 4),
                            "cube_h": round(cube_h, 4)}
    gq = yaw_quat(beta)
    gripper_call(open_gripper, "open")
    move([H[0], H[1], z_grasp + 0.10], gq, "pre_grasp_tool")
    move([H[0], H[1], z_grasp], gq, "grasp_tool", tol=0.006)
    g = gripper_call(close_gripper, "close_on_tool")
    lift_z = z_grasp + 0.03
    move([H[0], H[1], lift_z], gq, "lift_tool", tol=0.01)
    transit_z = max(z_grasp + 0.13, table_z + 0.20)
    attempts = []
    if mode == "route":
        work_z = z_grasp + 0.012
        report["route_params"] = {"work_z": round(work_z, 4), "transit_z": round(transit_z, 4)}
        beta0 = beta
        last_final = {"leg": None, "Qs": None}
        rec_ov = {"v": None}
        for attempt in range(5):
            if ended["flag"] or ticks["used"] > max_steps - 200:
                break
            D_now = math.sqrt((C[0] - T[0]) ** 2 + (C[1] - T[1]) ** 2)
            if D_now < 0.006:
                break
            rt = None
            if last_final["leg"] is not None and D_now < 0.05:
                # closed-loop finish: same tool face and orientation as the last final leg, re-aimed at
                # the target from the re-observed cube (hand stays near the last, reachable, poses)
                pl = last_final["leg"]
                dxx, dyy = T[0] - C[0], T[1] - C[1]
                cosang = (dxx * pl["dir"][0] + dyy * pl["dir"][1]) / (D_now + 1e-9)
                ov = 0.0
                if last_final["Qs"] is not None:
                    # measured overshoot of the stop-short leg along its push direction (runtime-derived)
                    ov = (C[0] - last_final["Qs"][0]) * pl["dir"][0] + (C[1] - last_final["Qs"][1]) * pl["dir"][1]
                    ov = max(0.0, min(0.02, ov))
                rec_ov["v"] = ov
                Q_aim = [T[0] - ov * pl["dir"][0], T[1] - ov * pl["dir"][1], T[2]]
                if cosang > 0.8:
                    nl = leg_options(C, Q_aim, beta, geo["axis"] + (beta - beta0), False,
                                     (pl["mode"], pl["psi"], pl["sk"], pl["lk"], 0, 0.0))
                    if nl is not None and nl["feasible"]:
                        rt = {"legs": [nl], "cost": 0.0, "feasible": True, "nudge": True}
            if rt is None:
                rt = plan_route(C, T, beta, geo["axis"] + (beta - beta0))
            if rt is None:
                attempts.append({"attempt": attempt, "plan": None})
                break
            lg = rt["legs"][0]
            if not rt["feasible"] and D_now < 0.03:
                attempts.append({"attempt": attempt, "C": [round(c, 4) for c in C], "D": round(D_now, 4),
                                 "stop": "no feasible correction; keep the cube where it is"})
                break
            is_final = rt.get("nudge") or abs(lg["Q"][0] - T[0]) + abs(lg["Q"][1] - T[1]) < 1e-9
            stop_short = 0.0
            if is_final and not rt.get("nudge") and lg["D"] > 0.05:
                # stop short of the target, re-observe, then finish with a short nudge
                stop_short = 0.025
                lg = dict(lg)
                lg["e1"] = [lg["e1"][0] - stop_short * lg["dir"][0], lg["e1"][1] - stop_short * lg["dir"][1]]
            last_final["leg"] = lg if is_final else None
            last_final["Qs"] = None
            if is_final:
                last_final["Qs"] = [lg["Q"][0] - stop_short * lg["dir"][0], lg["Q"][1] - stop_short * lg["dir"][1]]
            rec = {"attempt": attempt, "C": [round(c, 4) for c in C], "D": round(D_now, 4),
                   "n_legs": len(rt["legs"]), "route_feasible": rt["feasible"],
                   "nudge": bool(rt.get("nudge")), "stop_short": stop_short, "overshoot_used": rec_ov["v"],
                   "legs": [brief_leg(x) for x in rt["legs"]]}
            attempts.append(rec)
            pq = yaw_quat(lg["beta"])
            cur = get_robot_state().end_effector_poses["primary"].position
            move([cur[0], cur[1], transit_z], yaw_quat(beta), "raise", steps=100)
            ok = move([lg["e0"][0], lg["e0"][1], transit_z], pq, "above_leg_start", tol=0.012, steps=160)
            if not ok:
                rec["abort"] = "above_leg_start failed"
                break
            beta = lg["beta"]
            move([lg["e0"][0], lg["e0"][1], work_z + 0.04], pq, "lower_tool_a", tol=0.01, steps=100)
            move([lg["e0"][0], lg["e0"][1], work_z], pq, "lower_tool_b", tol=0.008, steps=80)
            fails = 0
            for k in range(1, 6):
                f = k / 5.0
                ex = lg["e0"][0] + (lg["e1"][0] - lg["e0"][0]) * f
                ey = lg["e0"][1] + (lg["e1"][1] - lg["e0"][1]) * f
                if not move([ex, ey, work_z], pq, "%s_%d_%d" % (lg["mode"], attempt, k), tol=0.01, steps=70):
                    fails += 1
                if ended["flag"] or fails >= 1:
                    break
            if ended["flag"]:
                break
            cur = get_robot_state().end_effector_poses["primary"].position
            move([cur[0], cur[1], transit_z], pq, "lift_after", steps=100)
            if ended["flag"]:
                break
            try:
                newcube = localize(["red cube", "small red block"], None)
            except Exception as exc:
                newcube = None
                report["errors"].append("relocalize: " + str(exc)[:100])
            if newcube is None:
                rec["reobserved"] = None
                break
            C = newcube["c"]
            rec["reobserved"] = [round(C[0], 4), round(C[1], 4)]
            rec["residual"] = round(math.sqrt((C[0] - T[0]) ** 2 + (C[1] - T[1]) ** 2), 4)
        report["route_attempts"] = attempts
    else:
        off = [E[0] - H[0], E[1] - H[1]]
        end_z_rel = E[2] - z_grasp
        tip_half_len = min(0.035, max(0.008, 0.5 * tool["end_len"]))
        tip_half_wid = min(0.035, max(0.008, 0.5 * tool["end_wid"]))
        cur_beta = beta
        push_z = max(z_grasp + 0.006, table_z + 0.6 * cube_h - end_z_rel)
        report["push_params"] = {"off": [round(c, 4) for c in off], "end_z_rel": round(end_z_rel, 4),
                                 "push_z": round(push_z, 4)}
        for attempt in range(3):
            if ended["flag"] or ticks["used"] > max_steps - 200:
                break
            plan = plan_push(C, T, off, cur_beta, tip_half_len, tip_half_wid, cube_half)
            if plan is None:
                attempts.append({"attempt": attempt, "plan": None})
                break
            rec = {"attempt": attempt, "C": [round(c, 4) for c in C], "D": round(plan["D"], 4),
                   "beta_deg": round(math.degrees(plan["beta"]), 1), "worst_r": round(plan["worst_r"], 3)}
            attempts.append(rec)
            if plan["D"] < 0.015:
                break
            pq = yaw_quat(plan["beta"])
            cur = get_robot_state().end_effector_poses["primary"].position
            move([cur[0], cur[1], transit_z], yaw_quat(cur_beta), "raise", steps=100)
            if not move([plan["s0"][0], plan["s0"][1], transit_z], pq, "above_push_start", tol=0.012, steps=160):
                rec["abort"] = "above_push_start failed"
                break
            off = plan["off"]
            cur_beta = plan["beta"]
            move([plan["s0"][0], plan["s0"][1], push_z], pq, "lower_tool", tol=0.008, steps=100)
            fails = 0
            for k in range(1, 5):
                f = k / 4.0
                sx = plan["s0"][0] + (plan["s1"][0] - plan["s0"][0]) * f
                sy = plan["s0"][1] + (plan["s1"][1] - plan["s0"][1]) * f
                if not move([sx, sy, push_z], pq, "push_%d_%d" % (attempt, k), tol=0.008, steps=70):
                    fails += 1
                if ended["flag"] or fails >= 2:
                    break
            if ended["flag"]:
                break
            cur = get_robot_state().end_effector_poses["primary"].position
            move([cur[0], cur[1], transit_z], pq, "lift_after", steps=100)
            if ended["flag"]:
                break
            try:
                newcube = localize(["red cube", "small red block"], None)
            except Exception as exc:
                newcube = None
                report["errors"].append("relocalize: " + str(exc)[:100])
            if newcube is None:
                break
            C = newcube["c"]
            dd = math.sqrt((C[0] - T[0]) ** 2 + (C[1] - T[1]) ** 2)
            rec["reobserved"] = [round(C[0], 4), round(C[1], 4)]
            rec["residual"] = round(dd, 4)
            if dd < 0.012:
                break
        report["push_attempts"] = attempts
report["ticks_used"] = ticks["used"]
report["ended"] = ended["flag"]
result = report
