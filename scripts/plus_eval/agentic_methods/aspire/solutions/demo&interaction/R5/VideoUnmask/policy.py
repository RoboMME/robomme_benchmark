# VideoUnmask v03 -- v02 + grasp axis chosen so the hand's long (finger) axis avoids tall
# obstacles (points above container top + 2 cm inside the hand footprint), switch to the other
# container-aligned axis when a descent does not converge, held check at grip > 0.1.
# Strategy (all task-dependent quantities derived at runtime):
#  1. Parse the goal text into an ordered list of (colour, all-or-one) picks.
#  2. Scan the conditioning video (front + wrist RGB-D, per-frame calibration) for
#     saturated cube-coloured blobs of the goal colours; back-project each blob to
#     robot_base and cluster the observations per colour -> hidden-cube positions.
#  3. For each pick: hover above the hidden cube, read the wrist depth to find the
#     container covering it (top face centre, yaw, top height), grasp top-down,
#     lift, then put it back down and release.
import math

report = {"version": "v03", "errors": [], "picks": []}
COLOUR_WORDS = ["red", "green", "blue", "yellow", "purple", "orange"]


def note(msg):
    if len(report["errors"]) < 40:
        report["errors"].append(str(msg)[:300])


def to_list(arr):
    try:
        return arr.tolist()
    except Exception:
        return [list(row) for row in arr]


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def qrot(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def cam_model(cam):
    k = to_list(cam.intrinsics)
    pose = cam.camera_pose
    pos = [float(v) for v in pose.position]
    rot = qrot([float(v) for v in pose.quaternion_wxyz])
    return {"fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]),
            "cy": float(k[1][2]), "t": pos, "R": rot}


def backproject(model, u, v, d, conv):
    xc = (u - model["cx"]) / model["fx"] * d
    yc = (v - model["cy"]) / model["fy"] * d
    if conv == "cv":
        p = (xc, yc, d)
    elif conv == "gl":
        p = (xc, -yc, -d)
    else:
        p = (d, -xc, -yc)
    rot = model["R"]
    t = model["t"]
    return (rot[0][0] * p[0] + rot[0][1] * p[1] + rot[0][2] * p[2] + t[0],
            rot[1][0] * p[0] + rot[1][1] * p[1] + rot[1][2] * p[2] + t[1],
            rot[2][0] * p[0] + rot[2][1] * p[1] + rot[2][2] * p[2] + t[2])


def quantile(values, frac):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * frac)]


def classify(r, g, b, wanted):
    if "red" in wanted and r > 60 and g < 0.33 * r and b < 0.38 * r:
        return "red"
    if "green" in wanted and g > 60 and g > 1.45 * r and g > 1.3 * b:
        return "green"
    if "blue" in wanted and b > 60 and b > 1.6 * r and b > 1.35 * g:
        return "blue"
    if "yellow" in wanted and r > 120 and g > 0.8 * r and b < 0.35 * g:
        return "yellow"
    if "purple" in wanted and r > 60 and b > 60 and g < 0.6 * r and g < 0.6 * b:
        return "purple"
    return None


def blobs_in_frame(rgb, depth, wanted, stride, min_samples):
    """Return [(colour, mean_u, mean_v, median_depth, n)] for connected colour blobs."""
    hits = {}
    height = len(rgb)
    width = len(rgb[0])
    for v in range(0, height, stride):
        row = rgb[v]
        for u in range(0, width, stride):
            px = row[u]
            r = px[0]
            g = px[1]
            b = px[2]
            if r < 60 and g < 60 and b < 60:
                continue
            c = classify(r, g, b, wanted)
            if c is not None:
                hits[(u // stride, v // stride)] = c
    out = []
    seen = set()
    for key in hits:
        if key in seen:
            continue
        colour = hits[key]
        stack = [key]
        seen.add(key)
        members = []
        while stack:
            cur = stack.pop()
            members.append(cur)
            for du in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    nb = (cur[0] + du, cur[1] + dv)
                    if nb in hits and nb not in seen and hits[nb] == colour:
                        seen.add(nb)
                        stack.append(nb)
        if len(members) < min_samples:
            continue
        us = [m[0] * stride for m in members]
        vs = [m[1] * stride for m in members]
        ds = []
        for m in members:
            dval = float(depth[m[1] * stride][m[0] * stride])
            if dval > 0.01 and dval < 5.0:
                ds.append(dval)
        if not ds:
            continue
        edge = min(us) <= 2 * stride or min(vs) <= 2 * stride or max(us) >= width - 1 - 2 * stride or max(vs) >= height - 1 - 2 * stride
        out.append((colour, sum(us) / len(us), sum(vs) / len(vs), quantile(ds, 0.5), len(members), edge))
    return out


def parse_goal(text):
    low = text.lower()
    start = low.find("then")
    body = low[start:] if start >= 0 else low
    found = []
    for word in COLOUR_WORDS:
        idx = body.find(word)
        while idx >= 0:
            found.append((idx, word))
            idx = body.find(word, idx + 1)
    found = sorted(found)
    plan = []
    prev = 0
    for item in found:
        clause = body[prev:item[0]]
        want_all = (" all " in (" " + clause + " ")) or (item[1] + "s" in body[item[0]:item[0] + len(item[1]) + 2] and "cubes" in body[item[0]:item[0] + 14])
        plan.append({"colour": item[1], "all": bool(want_all)})
        prev = item[0] + len(item[1])
    return plan


def choose_convention(name, live_cam, live_depth, base_frame):
    model = cam_model(live_cam)
    convs = ["cv", "gl", "body"]
    info = {"camera": name}
    # (a) calibration through the public mask_to_point_cloud tool, hand-built mask
    h = len(live_depth)
    w = len(live_depth[0])
    best = None
    for (uc, vc) in ((w // 2, (3 * h) // 4), (w // 2, h // 2), (w // 4, (3 * h) // 4)):
        try:
            mask = []
            pix = []
            for v in range(h):
                row = []
                for u in range(w):
                    inside = abs(u - uc) <= 3 and abs(v - vc) <= 3
                    row.append(inside)
                    if inside and live_depth[v][u] > 0.01:
                        pix.append((u, v, float(live_depth[v][u])))
                mask.append(row)
            if len(pix) < 10:
                continue
            cloud = mask_to_point_cloud(mask, name, target_frame=base_frame)
            pts = to_list(cloud.points)
            if len(pts) < 5:
                continue
            ref = [sum(p[i] for p in pts) / len(pts) for i in range(3)]
            errs = {}
            for conv in convs:
                acc = [0.0, 0.0, 0.0]
                for (u, v, d) in pix:
                    q = backproject(model, u, v, d, conv)
                    acc = [acc[0] + q[0], acc[1] + q[1], acc[2] + q[2]]
                mine = [a / len(pix) for a in acc]
                errs[conv] = math.sqrt(sum((mine[i] - ref[i]) ** 2 for i in range(3)))
            info["mask_errs"] = errs
            info["mask_ref"] = ref
            best = sorted([(errs[c], c) for c in convs])[0]
            break
        except Exception as err:
            info["mask_error"] = str(err)[:200]
    # (b) horizontal-table test: the table is a horizontal plane below the camera
    plane = {}
    for conv in convs:
        zs = []
        for v in range(8, h, 16):
            for u in range(8, w, 16):
                d = float(live_depth[v][u])
                if d > 0.01 and d < 5.0:
                    zs.append(backproject(model, u, v, d, conv)[2])
        if len(zs) < 10:
            continue
        below = sum(1 for z in zs if z < model["t"][2] - 0.05) / float(len(zs))
        spread = quantile(zs, 0.6) - quantile(zs, 0.25)
        plane[conv] = (round(spread, 4), round(below, 3), round(quantile(zs, 0.5), 4))
    info["plane"] = plane
    choice = None
    if best is not None and best[0] < 0.03:
        choice = best[1]
        info["method"] = "mask_to_point_cloud"
    else:
        ranked = sorted([(plane[c][0], c) for c in plane if plane[c][1] > 0.8])
        if ranked:
            choice = ranked[0][1]
            info["method"] = "plane"
    if choice is None:
        choice = "gl"
        info["method"] = "default"
    info["conv"] = choice
    return choice, info


def table_height(model, depth, conv):
    zs = []
    for v in range(4, len(depth), 8):
        for u in range(4, len(depth[0]), 8):
            d = float(depth[v][u])
            if d > 0.01 and d < 5.0:
                zs.append(backproject(model, u, v, d, conv)[2])
    bins = {}
    for z in zs:
        key = int(math.floor(z / 0.01))
        bins[key] = bins.get(key, 0) + 1
    if not bins:
        return 0.0
    peak = sorted([(bins[k], k) for k in bins])[-1][1]
    near = [z for z in zs if abs(z - (peak + 0.5) * 0.01) < 0.012]
    return quantile(near, 0.5)


def cluster_observations(obs, radius):
    """Greedy per-colour clustering; centre from blobs not touching the image border when any."""
    clusters = []
    for o in obs:
        placed = None
        for c in clusters:
            if c["colour"] != o["colour"]:
                continue
            if math.sqrt((c["x"] - o["x"]) ** 2 + (c["y"] - o["y"]) ** 2) < radius:
                placed = c
                break
        if placed is None:
            placed = {"colour": o["colour"], "members": [], "t0": o["t"], "t1": o["t"], "cams": []}
            clusters.append(placed)
        placed["members"].append(o)
        placed["t1"] = max(placed["t1"], o["t"])
        placed["t0"] = min(placed["t0"], o["t"])
        if o["cam"] not in placed["cams"]:
            placed["cams"].append(o["cam"])
        good = [m for m in placed["members"] if not m["edge"]]
        use = good if good else placed["members"]
        placed["x"] = sum(m["x"] for m in use) / len(use)
        placed["y"] = sum(m["y"] for m in use) / len(use)
        placed["z"] = sum(m["z"] for m in use) / len(use)
        placed["n"] = len(placed["members"])
        placed["edge_only"] = not good
    return clusters


def ex_info(res):
    out = {}
    try:
        out["ok"] = bool(res.ok)
    except Exception:
        pass
    try:
        out["steps"] = int(res.steps_executed)
    except Exception:
        pass
    try:
        out["status"] = str(res.status)
    except Exception:
        pass
    for key in ("terminated", "truncated"):
        try:
            if key == "terminated":
                out[key] = bool(res.terminated)
            else:
                out[key] = bool(res.truncated)
        except Exception:
            pass
    return out


state_flags = {"ended": False, "ticks": 0}


def track(label, res, log):
    info = ex_info(res)
    info["what"] = label
    log.append(info)
    state_flags["ticks"] = state_flags["ticks"] + info.get("steps", 0)
    if info.get("terminated") or info.get("truncated"):
        state_flags["ended"] = True
    return info


def yaw_quat(q0, delta):
    return qmul((math.cos(delta / 2.0), 0.0, 0.0, math.sin(delta / 2.0)), q0)


def perceive_container(target, table_z, base_frame, conv_wrist, hover_z):
    """Find the container nearest target xy as a connected object component in wrist depth."""
    obs = get_observation()
    cam = obs.cameras["wrist"]
    model = cam_model(cam)
    depth = to_list(cam.depth_m)
    pts = []
    zcap = hover_z - 0.05          # everything higher is the robot's own hand/fingers
    for v in range(0, len(depth), 2):
        row = depth[v]
        for u in range(0, len(row), 2):
            d = float(row[u])
            if d < 0.01 or d > 3.0:
                continue
            p = backproject(model, u, v, d, conv_wrist)
            if p[2] > zcap:
                continue
            if abs(p[0] - target[0]) < 0.11 and abs(p[1] - target[1]) < 0.11:
                pts.append(p)
    info = {"n_local": len(pts)}
    cell = 0.006
    cells = {}
    for p in pts:
        if p[2] < table_z + 0.02:
            continue
        key = (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)))
        if key in cells:
            cells[key].append(p)
        else:
            cells[key] = [p]
    seen = set()
    comps = []
    for key in cells:
        if key in seen:
            continue
        seen.add(key)
        stack = [key]
        members = []
        while stack:
            cur = stack.pop()
            members.extend(cells[cur])
            for du in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    nb = (cur[0] + du, cur[1] + dv)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        if len(members) < 15:
            continue
        top = quantile([p[2] for p in members], 0.85)
        face = [p for p in members if abs(p[2] - top) < 0.012]
        if len(face) < 8:
            continue
        fx = sum(p[0] for p in face) / len(face)
        fy = sum(p[1] for p in face) / len(face)
        dist = math.sqrt((fx - target[0]) ** 2 + (fy - target[1]) ** 2)
        comps.append((dist, fx, fy, top, face, len(members)))
    comps = sorted(comps)
    info["comps"] = [[round(c[0], 3), round(c[1], 3), round(c[2], 3), round(c[3], 3), len(c[4]), c[5]] for c in comps[:5]]
    if not comps or comps[0][0] > 0.05:
        info["found"] = False
        return None, info
    best_c = comps[0]
    cx, cy, top, face = best_c[1], best_c[2], best_c[3], best_c[4]
    best = None
    for step_deg in range(0, 90, 3):
        a = math.radians(step_deg)
        ca, sa = math.cos(a), math.sin(a)
        pa = [(p[0] - cx) * ca + (p[1] - cy) * sa for p in face]
        pb = [-(p[0] - cx) * sa + (p[1] - cy) * ca for p in face]
        wa = quantile(pa, 0.97) - quantile(pa, 0.03)
        wb = quantile(pb, 0.97) - quantile(pb, 0.03)
        if best is None or wa * wb < best[0]:
            best = (wa * wb, a, wa, wb)
    info.update({"found": True, "cx": round(cx, 4), "cy": round(cy, 4), "top": round(top, 4),
                 "yaw_deg": round(math.degrees(best[1]), 1), "wa": round(best[2], 4),
                 "wb": round(best[3], 4), "n_face": len(face)})
    clear = []
    for k in range(2):
        a = best[1] + k * math.pi / 2.0
        half = (best[2] if k == 0 else best[3]) / 2.0
        ca, sa = math.cos(a), math.sin(a)
        count = 0
        for p in pts:
            if p[2] < table_z + 0.015 or p[2] > top + 0.04:
                continue
            da = (p[0] - cx) * ca + (p[1] - cy) * sa
            db = -(p[0] - cx) * sa + (p[1] - cy) * ca
            if abs(db) < 0.015 and abs(da) > half + 0.006 and abs(da) < 0.065:
                count += 1
        clear.append(count)
    info["clear"] = clear
    # hand footprint: the Panda hand is long along its closing axis; count points of anything
    # standing above the container top inside an 11 cm x 3.5 cm half-size box along each axis
    tall = []
    for k in range(2):
        a = best[1] + k * math.pi / 2.0
        ca, sa = math.cos(a), math.sin(a)
        count = 0
        for p in pts:
            if p[2] < top + 0.02:
                continue
            da = (p[0] - cx) * ca + (p[1] - cy) * sa
            db = -(p[0] - cx) * sa + (p[1] - cy) * ca
            if abs(da) < 0.11 and abs(db) < 0.035:
                count += 1
        tall.append(count)
    info["tall"] = tall
    return {"x": cx, "y": cy, "top": top, "yaw": best[1], "wa": best[2], "wb": best[3], "clear": clear,
            "tall": tall}, info


def grasp_quat(cont, k, phi0, q0):
    """Top-down grasp quaternion whose closing axis is the container's k-th edge normal."""
    dl = cont["yaw"] + k * math.pi / 2.0 - phi0
    while dl > math.pi / 2.0:
        dl -= math.pi
    while dl < -math.pi / 2.0:
        dl += math.pi
    return yaw_quat(q0, dl), dl


def front_top(target, table_z, live_front_pts):
    near = [p[2] for p in live_front_pts if (p[0] - target[0]) ** 2 + (p[1] - target[1]) ** 2 < 0.02 ** 2 and p[2] > table_z + 0.01]
    if len(near) < 3:
        return None
    return quantile(near, 0.9)


try:
    context = get_task_context()
    goal = context.language
    report["goal"] = goal
    try:
        meta = robomme_plus.get_task_metadata()
        report["meta"] = {"max_steps": meta.get("max_steps"), "control_hz": meta.get("control_hz"),
                          "sampling": meta.get("conditioning_video_sampling")}
    except Exception as err:
        note("meta: " + str(err))
    plan = parse_goal(goal)
    report["plan"] = plan
    wanted = set(p["colour"] for p in plan)

    state0 = get_robot_state()
    base_frame = state0.base_frame
    ee0 = state0.end_effector_poses["primary"]
    q0 = tuple(float(v) for v in ee0.quaternion_wxyz)
    rot0 = qrot(q0)
    if rot0[2][2] > -0.9:
        note("home EE not pointing down; using (0,1,0,0)")
        q0 = (0.0, 1.0, 0.0, 0.0)
        rot0 = qrot(q0)
    phi0 = math.atan2(rot0[1][1], rot0[0][1])
    report["ee0"] = [round(float(v), 4) for v in ee0.position] + [round(v, 4) for v in q0]

    live = get_observation()
    convs = {}
    calib = {}
    live_depth = {}
    for name in ("front", "wrist"):
        live_depth[name] = to_list(live.cameras[name].depth_m)
        conv, info = choose_convention(name, live.cameras[name], live_depth[name], base_frame)
        convs[name] = conv
        calib[name] = info
    report["calibration"] = calib
    front_model = cam_model(live.cameras["front"])
    table_z = table_height(front_model, live_depth["front"], convs["front"])
    report["table_z"] = round(table_z, 4)
    live_front_pts = []
    for v in range(0, 256, 2):
        for u in range(0, 256, 2):
            d = float(live_depth["front"][v][u])
            if d > 0.01 and d < 5.0:
                live_front_pts.append(backproject(front_model, u, v, d, convs["front"]))

    # ---- scan the conditioning video ----
    video = robomme_plus.get_conditioning_video()
    report["video_frames"] = len(video)
    obs_list = []
    last_t = -10.0
    used = 0
    for frame in video:
        t = float(frame.timestamp_s)
        if t - last_t < 0.14:
            continue
        last_t = t
        used += 1
        for name in ("front", "wrist"):
            cam = frame.cameras[name]
            rgb = to_list(cam.rgb)
            depth = to_list(cam.depth_m)
            model = cam_model(cam)
            min_samples = 3 if name == "front" else 10
            for blob in blobs_in_frame(rgb, depth, wanted, 2, min_samples):
                p = backproject(model, blob[1], blob[2], blob[3], convs[name])
                if p[2] < table_z - 0.04 or p[2] > table_z + 0.09:
                    continue
                obs_list.append({"colour": blob[0], "cam": name, "t": round(t, 2), "x": p[0], "y": p[1],
                                 "z": p[2], "u": round(blob[1], 1), "v": round(blob[2], 1), "n": blob[4], "edge": blob[5]})
    report["video_frames_used"] = used
    report["n_obs"] = len(obs_list)
    clusters = cluster_observations(obs_list, 0.03)
    clusters = [c for c in clusters if c["n"] >= 2]
    # same cube seen by both cameras: same colour, same visibility interval, nearby
    merged = []
    for c in clusters:
        dup = None
        for m in merged:
            if m["colour"] != c["colour"] or m["cams"] == c["cams"]:
                continue
            same_time = abs(m["t0"] - c["t0"]) < 0.35 and abs(m["t1"] - c["t1"]) < 0.35
            dist = math.sqrt((m["x"] - c["x"]) ** 2 + (m["y"] - c["y"]) ** 2)
            if same_time and dist < 0.12:
                dup = m
                break
        if dup is None:
            merged.append(c)
        else:
            report.setdefault("merged_pairs", []).append([dup["colour"], round(dup["x"], 4), round(dup["y"], 4),
                                                          round(c["x"], 4), round(c["y"], 4)])
            if "front" in c["cams"]:
                dup["x"], dup["y"], dup["z"] = c["x"], c["y"], c["z"]
            dup["n"] = dup["n"] + c["n"]
            dup["cams"] = dup["cams"] + c["cams"]
            dup["edge_only"] = dup["edge_only"] and c["edge_only"]
    clusters = merged
    report["clusters"] = [{"colour": c["colour"], "x": round(c["x"], 4), "y": round(c["y"], 4),
                           "z": round(c["z"], 4), "n": c["n"], "t": [c["t0"], c["t1"]], "cams": c["cams"],
                           "edge_only": c["edge_only"]}
                          for c in clusters]
    report["obs_sample"] = obs_list[:60]

    # ---- build the ordered pick list ----
    picks = []
    for item in plan:
        cands = [c for c in clusters if c["colour"] == item["colour"]]
        if not cands:
            note("no hidden cube found for " + item["colour"])
            continue
        if item["all"]:
            chosen = cands
        else:
            chosen = [sorted([(c["n"], i) for i, c in enumerate(cands)])[-1]]
            chosen = [cands[chosen[0][1]]]
        for c in chosen:
            picks.append({"colour": c["colour"], "x": c["x"], "y": c["y"]})
    report["pick_order"] = [[p["colour"], round(p["x"], 4), round(p["y"], 4)] for p in picks]

    # ---- execute ----
    log = []
    report["motion_log"] = log
    track("open", open_gripper(), log)
    hover_dz = 0.22
    for index, pk in enumerate(picks):
        if state_flags["ended"]:
            break
        rec = {"colour": pk["colour"], "cube_xy": [round(pk["x"], 4), round(pk["y"], 4)]}
        report["picks"].append(rec)
        hover = Pose((pk["x"], pk["y"], table_z + hover_dz), q0, base_frame)
        track("hover", move_to_pose(hover, tolerance=0.015, max_steps=200), log)
        if state_flags["ended"]:
            break
        cont, pinfo = perceive_container((pk["x"], pk["y"]), table_z, base_frame, convs["wrist"], table_z + hover_dz)
        rec["perceive"] = pinfo
        if cont is not None and math.sqrt((cont["x"] - pk["x"]) ** 2 + (cont["y"] - pk["y"]) ** 2) > 0.05:
            note("container centre far from cube; using cube xy")
            cont = None
        if cont is None:
            ft = front_top((pk["x"], pk["y"]), table_z, live_front_pts)
            top = ft if ft is not None else table_z + 0.05
            cont = {"x": pk["x"], "y": pk["y"], "top": top, "yaw": phi0, "wa": 0.05, "wb": 0.05,
                    "clear": [0, 0], "tall": [0, 0]}
        widths = [cont["wa"], cont["wb"]]
        options = [k for k in (0, 1) if widths[k] < 0.075]
        if not options:
            options = [0, 1]
        ranked = sorted([(cont["tall"][k], cont["clear"][k], k) for k in options])
        axis_order = [r[2] for r in ranked] + [k for k in (0, 1) if k not in options]
        qg, delta = grasp_quat(cont, axis_order[0], phi0, q0)
        axis_i = 0
        height = cont["top"] - table_z
        gz = cont["top"] - min(0.02, 0.45 * height)
        gz = max(gz, table_z + 0.012)
        rec["grasp"] = {"x": round(cont["x"], 4), "y": round(cont["y"], 4), "z": round(gz, 4),
                        "delta_deg": round(math.degrees(delta), 1), "height": round(height, 4),
                        "axis_order": axis_order}
        attempts = 0
        held = False
        while attempts < 3 and not held and not state_flags["ended"]:
            attempts += 1
            track("pre", move_to_pose(Pose((cont["x"], cont["y"], cont["top"] + 0.07), qg, base_frame),
                                      tolerance=0.01, max_steps=200), log)
            if state_flags["ended"]:
                break
            dres = track("descend", move_to_pose(Pose((cont["x"], cont["y"], gz), qg, base_frame),
                                                 tolerance=0.008, max_steps=150), log)
            if state_flags["ended"]:
                break
            if not dres.get("ok", True) and attempts < 3:
                note("descend did not converge at pick " + str(index))
                axis_i = (axis_i + 1) % len(axis_order)
                track("retreat", move_to_pose(Pose((cont["x"], cont["y"], table_z + hover_dz), qg, base_frame),
                                              tolerance=0.015, max_steps=150), log)
                if state_flags["ended"]:
                    break
                c2, pinfo2 = perceive_container((cont["x"], cont["y"]), table_z, base_frame, convs["wrist"], table_z + hover_dz)
                rec["perceive_retry"] = pinfo2
                if c2 is not None and math.sqrt((c2["x"] - pk["x"]) ** 2 + (c2["y"] - pk["y"]) ** 2) < 0.05:
                    cont["x"], cont["y"], cont["top"] = c2["x"], c2["y"], c2["top"]
                    gz = max(cont["top"] - min(0.02, 0.45 * (cont["top"] - table_z)), table_z + 0.012)
                qg, delta = grasp_quat(cont, axis_order[axis_i], phi0, q0)
                rec["retry_delta_deg_" + str(attempts)] = round(math.degrees(delta), 1)
                continue
            track("close", close_gripper(), log)
            if state_flags["ended"]:
                break
            grip = float(get_robot_state().gripper_positions["primary"])
            rec["grip_after_close_" + str(attempts)] = round(grip, 3)
            if grip < 0.1:
                note("gripper closed on nothing at pick " + str(index))
                track("reopen", open_gripper(), log)
                track("retreat", move_to_pose(Pose((cont["x"], cont["y"], table_z + hover_dz), qg, base_frame),
                                              tolerance=0.015, max_steps=150), log)
                if state_flags["ended"]:
                    break
                c2, pinfo2 = perceive_container((cont["x"], cont["y"]), table_z, base_frame, convs["wrist"], table_z + hover_dz)
                rec["perceive_retry"] = pinfo2
                if c2 is not None and math.sqrt((c2["x"] - pk["x"]) ** 2 + (c2["y"] - pk["y"]) ** 2) < 0.05:
                    cont["x"], cont["y"], cont["top"] = c2["x"], c2["y"], c2["top"]
                    gz = max(cont["top"] - min(0.02, 0.45 * (cont["top"] - table_z)), table_z + 0.012)
                qg, delta = grasp_quat(cont, axis_order[axis_i], phi0, q0)
                rec["retry_delta_deg_" + str(attempts)] = round(math.degrees(delta), 1)
                continue
            held = True
        if state_flags["ended"] or not held:
            rec["held"] = held
            continue
        lift = track("lift", move_to_pose(Pose((cont["x"], cont["y"], gz + 0.13), qg, base_frame),
                                          tolerance=0.015, max_steps=150), log)
        rec["held"] = True
        try:
            rec["grip_after_lift"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
        except Exception:
            pass
        if state_flags["ended"]:
            break
        track("lower", move_to_pose(Pose((cont["x"], cont["y"], gz + 0.008), qg, base_frame),
                                    tolerance=0.012, max_steps=150), log)
        if state_flags["ended"]:
            break
        track("release", open_gripper(), log)
        if state_flags["ended"]:
            break
        track("up", move_to_pose(Pose((cont["x"], cont["y"], cont["top"] + 0.08), qg, base_frame),
                                 tolerance=0.015, max_steps=150), log)
    report["ended"] = state_flags["ended"]
    report["ticks"] = state_flags["ticks"]
except Exception as err:
    note("fatal: " + str(err))

result = report
