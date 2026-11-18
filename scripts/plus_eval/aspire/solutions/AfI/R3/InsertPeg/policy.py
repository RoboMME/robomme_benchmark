# InsertPeg v02 -- start 3 of 3 (final).
#
# v01 (ep 0): the scene-equality test passed (box 0.96, pegs 0.94-0.99 vs video frame 0)
# and the joint replay followed the expert within ~1-2 ticks (EE within 0.5-5 mm near
# the box), grasped the right peg at the right (yellow) end, but the insertion stalled
# at EE y 0.050 vs the expert's 0.014: the wrist camera showed the hole opening to the
# image-left (world +x) of the peg tip, i.e. the in-hand peg pose differed from the
# expert's. The 1.5 cm move_to_pose nudge timed out after 120 ticks (max_steps=40).
#
# v02: 2-frame lead in the replay; an in-hand correction measured from WRIST DEPTH
# (held-peg points in the tool frame, live vs the video frame at the same tick, while the
# expert is stationary after the lift; colour is unusable there because a spotlight
# saturates the peg) applied to the rest of the expert path by IK;
# and, if still short of the expert's final point, a bounded lateral/vertical probe
# search with explicit per-tick joint servoing (8 + 22 + 4 ticks per probe).

import math

report = {"version": "v02", "errors": [], "motions": []}
result = report


def note_error(tag, exc):
    report["errors"].append(tag + ": " + str(exc))


def r4(x):
    return round(float(x), 4)


def vec(values):
    return [float(v) for v in values]


# ---------------------------------------------------------------- quaternions
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def qnorm(q):
    n = math.sqrt(sum(c * c for c in q))
    return [c / n for c in q]


def qz(theta):
    return [math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)]


def qmat(q):
    w, x, y, z = qnorm(q)
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def qangle(a, b):
    d = abs(sum(a[i] * b[i] for i in range(4)))
    d = min(1.0, d)
    return 2.0 * math.acos(d)


def wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a <= -math.pi:
        a += 2 * math.pi
    return a


# ---------------------------------------------------------------- arrays
def rgb_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        return [[[int(c) for c in px] for px in row] for row in arr]


def depth_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        return [[float(x) for x in row] for row in arr]


HEXD = "0123456789abcdef"


def hex_image(rows, stride):
    out = []
    for v in range(0, len(rows), stride):
        row = rows[v]
        parts = []
        for u in range(0, len(row), stride):
            px = row[u]
            for c in range(3):
                val = int(px[c])
                parts.append(HEXD[val // 16] + HEXD[val % 16])
        out.append("".join(parts))
    return out


def quantile(values, frac):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * frac)]


# ---------------------------------------------------------------- camera model
def cam_model(cam):
    k = cam.intrinsics
    pose = cam.camera_pose
    return {"fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]),
            "cy": float(k[1][2]), "p": vec(pose.position),
            "R": qmat(vec(pose.quaternion_wxyz))}


def project(m, P, conv):
    R = m["R"]
    d = [P[0] - m["p"][0], P[1] - m["p"][1], P[2] - m["p"][2]]
    xc = R[0][0] * d[0] + R[1][0] * d[1] + R[2][0] * d[2]
    yc = R[0][1] * d[0] + R[1][1] * d[1] + R[2][1] * d[2]
    zc = R[0][2] * d[0] + R[1][2] * d[1] + R[2][2] * d[2]
    if conv == 0:
        depth = zc
        if depth <= 0.01:
            return None
        return (m["cx"] + m["fx"] * xc / depth, m["cy"] + m["fy"] * yc / depth, depth)
    if conv == 1:
        depth = -zc
        if depth <= 0.01:
            return None
        return (m["cx"] + m["fx"] * xc / depth, m["cy"] - m["fy"] * yc / depth, depth)
    depth = xc
    if depth <= 0.01:
        return None
    return (m["cx"] - m["fx"] * yc / depth, m["cy"] - m["fy"] * zc / depth, depth)


def deproject(m, u, v, d, conv):
    a = (u - m["cx"]) / m["fx"] * d
    b = (v - m["cy"]) / m["fy"] * d
    if conv == 0:
        c = [a, b, d]
    elif conv == 1:
        c = [a, -b, -d]
    else:
        c = [d, -a, -b]
    R = m["R"]
    return [R[i][0] * c[0] + R[i][1] * c[1] + R[i][2] * c[2] + m["p"][i] for i in range(3)]


def pixel_of(m, P, conv, size):
    pr = project(m, P, conv)
    if pr is None:
        return None
    u = int(pr[0])
    v = int(pr[1])
    if u < 0 or v < 0 or u >= size or v >= size:
        return None
    return (u, v, pr[2])


# ---------------------------------------------------------------- planar shape
def planar_shape(points):
    """Centre (axis-end midpoint), unit axis, length, width, z quantiles."""
    n = len(points)
    if n < 10:
        return None
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points) / n
    syy = sum((p[1] - my) ** 2 for p in points) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points) / n
    ang = 0.5 * math.atan2(2 * sxy, sxx - syy)
    ax = [math.cos(ang), math.sin(ang)]
    ts = [(p[0] - mx) * ax[0] + (p[1] - my) * ax[1] for p in points]
    cs = [-(p[0] - mx) * ax[1] + (p[1] - my) * ax[0] for p in points]
    t0 = quantile(ts, 0.03)
    t1 = quantile(ts, 0.97)
    c0 = quantile(cs, 0.03)
    c1 = quantile(cs, 0.97)
    tc = 0.5 * (t0 + t1)
    cc = quantile(cs, 0.5)
    centre = [mx + tc * ax[0] - cc * ax[1], my + tc * ax[1] + cc * ax[0]]
    zs = [p[2] for p in points]
    return {"centre": centre, "axis": ax, "length": t1 - t0, "width": c1 - c0,
            "z05": quantile(zs, 0.05), "z50": quantile(zs, 0.5), "z95": quantile(zs, 0.95),
            "n": n}


def half_colours(points, colours, shape):
    """Mean colour of the +axis and -axis halves (relative to the shape centre)."""
    pos = [0.0, 0.0, 0.0, 0]
    neg = [0.0, 0.0, 0.0, 0]
    cx, cy = shape["centre"]
    ax = shape["axis"]
    for i in range(len(points)):
        t = (points[i][0] - cx) * ax[0] + (points[i][1] - cy) * ax[1]
        tgt = pos if t > 0 else neg
        for c in range(3):
            tgt[c] += float(colours[i][c])
        tgt[3] += 1
    out = []
    for tgt in (pos, neg):
        if tgt[3] == 0:
            out.append(None)
        else:
            out.append([tgt[0] / tgt[3], tgt[1] / tgt[3], tgt[2] / tgt[3]])
    return out


def cdist(a, b):
    if a is None or b is None:
        return 999.0
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def shape_summary(shape):
    if shape is None:
        return None
    return {"centre": [r4(v) for v in shape["centre"]], "axis": [r4(v) for v in shape["axis"]],
            "length": r4(shape["length"]), "width": r4(shape["width"]),
            "z05": r4(shape["z05"]), "z50": r4(shape["z50"]), "z95": r4(shape["z95"]),
            "n": shape["n"]}


try:
    # ============================================================ context
    context = get_task_context()
    state0 = get_robot_state()
    base = state0.base_frame
    report["goal"] = context.language
    report["task_name"] = context.task_name
    report["base_frame"] = base
    report["embodiment"] = state0.embodiment
    try:
        meta = robomme_plus.get_task_metadata()
        report["task_metadata"] = {str(k): str(meta[k]) for k in meta}
    except Exception as exc:
        meta = {}
        note_error("task_metadata", exc)
    try:
        cmeta = robomme_plus.get_controller_metadata()
        report["controller_metadata"] = {str(k): str(cmeta[k]) for k in cmeta}
    except Exception as exc:
        note_error("controller_metadata", exc)
    hz = 20.0
    max_steps = 1000
    try:
        max_steps = int(meta["max_steps"])
    except Exception as exc:
        note_error("max_steps", exc)
    try:
        hz = float(meta["control_hz"])
    except Exception:
        pass
    report["max_steps"] = max_steps
    ee0 = state0.end_effector_poses["primary"]
    report["live_start"] = {"ee_pos": [r4(v) for v in ee0.position],
                            "ee_quat": [r4(v) for v in ee0.quaternion_wxyz],
                            "gripper": r4(state0.gripper_positions["primary"]),
                            "joints": [r4(v) for v in state0.joint_positions["primary"]],
                            "t": r4(state0.timestamp_s)}

    # ============================================================ video timeline
    video = robomme_plus.get_conditioning_video()
    nv = len(video)
    report["video_frames"] = nv
    timeline = []
    for i in range(nv):
        rs = video[i].robot_state
        pe = rs.end_effector_poses["primary"]
        timeline.append({"t": r4(video[i].timestamp_s), "p": [r4(v) for v in pe.position],
                         "q": [r4(v) for v in pe.quaternion_wxyz],
                         "g": r4(rs.gripper_positions["primary"]),
                         "j": [r4(v) for v in rs.joint_positions["primary"]]})
    report["video_timeline"] = timeline
    result = report

    # gripper events
    events = []
    grasp_idx = None
    release_idx = None
    if nv > 0:
        g_open = max(f["g"] for f in timeline)
        closed_flag = timeline[0]["g"] < 0.5 * g_open
        for i in range(1, nv):
            g = timeline[i]["g"]
            if not closed_flag and g < timeline[i - 1]["g"] - 0.02 and g < 0.9 * g_open:
                # walk back to the start of the closing motion
                k = i
                while k > 0 and timeline[k - 1]["g"] > timeline[k]["g"] + 0.005:
                    k -= 1
                events.append(["close", k])
                closed_flag = True
            elif closed_flag and g > timeline[i - 1]["g"] + 0.02 and g > 0.5 * g_open:
                k = i
                while k > 0 and timeline[k - 1]["g"] < timeline[k]["g"] - 0.005:
                    k -= 1
                events.append(["open", k])
                closed_flag = False
        for ev in events:
            if ev[0] == "close" and grasp_idx is None:
                grasp_idx = ev[1]
            elif ev[0] == "open" and grasp_idx is not None and release_idx is None and ev[1] > grasp_idx:
                release_idx = ev[1]
    report["video_events"] = events
    report["grasp_idx"] = grasp_idx
    report["release_idx"] = release_idx

    # small video thumbnails (first/last front frame) for scene comparison offline
    vframes = {}
    for i in (0, nv - 1):
        if 0 <= i < nv:
            try:
                vframes[str(i)] = {"front": hex_image(rgb_rows(video[i].cameras["front"].rgb), 2)}
            except Exception as exc:
                note_error("vframe" + str(i), exc)
    report["video_images_128"] = vframes
    result = report

    # ============================================================ live observation
    obs = get_observation()
    fcam = obs.cameras["front"]
    live_rgb = rgb_rows(fcam.rgb)
    live_depth = depth_rows(fcam.depth_m)
    size = len(live_depth)
    lm = cam_model(fcam)

    # ============================================================ prompt probes
    probes = []
    cands = []
    for prompt in ("peg", "block"):
        entry = {"prompt": prompt, "ok": False, "cands": []}
        try:
            seg = segment_text("front", prompt)
            entry["ok"] = bool(seg.ok)
            if seg.ok:
                for ci in range(min(5, len(seg.segmentations))):
                    cand = seg.segmentations[ci]
                    ce = {"score": r4(cand.score)}
                    try:
                        ce["box"] = [r4(v) for v in cand.box_xyxy]
                    except Exception:
                        pass
                    try:
                        pc = mask_to_point_cloud(cand, "front", target_frame=base)
                        raw = pc.points
                        npts = len(raw)
                        stride = max(1, npts // 800)
                        pts = [vec(raw[k]) for k in range(0, npts, stride)]
                        shape = planar_shape(pts)
                        ce["npts"] = npts
                        ce["shape"] = shape_summary(shape)
                        if shape is not None:
                            cands.append({"prompt": prompt, "rank": ci, "score": float(cand.score),
                                          "pts": pts, "shape": shape})
                    except Exception as exc:
                        ce["cloud_error"] = str(exc)
                    entry["cands"].append(ce)
        except Exception as exc:
            entry["error"] = str(exc)
        probes.append(entry)
    report["probes"] = probes
    result = report

    # ---------------- objects: pegs (elongated, low) and the box (largest non-peg block)
    def dist2(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    pegs = []
    for c in cands:
        s = c["shape"]
        if s["n"] < 30 or s["length"] < 0.06 or s["length"] > 0.40 or s["width"] > 0.09:
            continue
        if s["length"] < 1.6 * max(0.005, s["width"]) or s["z95"] - s["z05"] > 0.09:
            continue
        dup = False
        for p in pegs:
            if dist2(p["shape"]["centre"], s["centre"]) < 0.03:
                dup = True
        if not dup:
            pegs.append(c)
    box = None
    for c in cands:
        s = c["shape"]
        if s["n"] < 100 or s["length"] > 0.40 or s["z95"] < 0.04:
            continue
        near_peg = False
        for p in pegs:
            if dist2(p["shape"]["centre"], s["centre"]) < 0.04:
                near_peg = True
        if near_peg:
            continue
        if box is None or s["n"] > box["shape"]["n"]:
            box = c
    report["pegs"] = [shape_summary(p["shape"]) for p in pegs]
    report["box"] = None if box is None else shape_summary(box["shape"])

    # ---------------- camera convention self-calibration
    conv = 0
    calib = []
    ref = box if box is not None else (pegs[0] if pegs else None)
    if ref is not None:
        best = None
        for cv in (0, 1, 2):
            res = []
            inb = 0
            for P in ref["pts"][:300]:
                px = pixel_of(lm, P, cv, size)
                if px is None:
                    continue
                inb += 1
                res.append(abs(float(live_depth[px[1]][px[0]]) - px[2]))
            med = quantile(res, 0.5) if res else 9.0
            calib.append({"conv": cv, "inbounds": inb, "median_resid": r4(med)})
            if inb > 0.6 * min(300, len(ref["pts"])) and (best is None or med < best):
                best = med
                conv = cv
    report["camera_convention"] = {"chosen": conv, "trials": calib}

    def static_frac(obj, vidx):
        """Fraction of the object's live points whose projection into video frame vidx
        has the same depth (within 1 cm). Depth only: the scene lights move."""
        vc = video[vidx].cameras["front"]
        vm = cam_model(vc)
        vd = depth_rows(vc.depth_m)
        ok_d = 0
        tot = 0
        for P in obj["pts"][:400]:
            px = pixel_of(vm, P, conv, size)
            if px is None:
                continue
            tot += 1
            if abs(float(vd[px[1]][px[0]]) - px[2]) < 0.01:
                ok_d += 1
        if tot == 0:
            return None
        return ok_d / tot

    # ---------------- is the live scene the video's initial scene?
    checks = []
    scene_static = nv > 0 and box is not None and len(pegs) > 0
    if nv > 0:
        objs = [("box", box)] + [("peg" + str(i), pegs[i]) for i in range(len(pegs))]
        for name, obj in objs:
            if obj is None:
                continue
            f = static_frac(obj, 0)
            checks.append([name, None if f is None else r4(f)])
            if f is None or f < 0.85:
                scene_static = False
    j0v = [float(v) for v in video[0].robot_state.joint_positions["primary"]] if nv > 0 else None
    jl = [float(v) for v in state0.joint_positions["primary"]]
    joint_gap = max(abs(j0v[i] - jl[i]) for i in range(7)) if j0v is not None else None
    report["static_checks_vs_video0"] = checks
    report["scene_static"] = scene_static
    report["start_joint_gap"] = None if joint_gap is None else r4(joint_gap)
    result = report

    # ============================================================ control helpers
    ended = [False]
    emb = state0.embodiment

    def tick_now():
        return int(round(float(get_robot_state().timestamp_s) * hz))

    def flags_of(res):
        out = {}
        try:
            out["terminated"] = bool(res.terminated)
        except Exception:
            pass
        try:
            out["truncated"] = bool(res.truncated)
        except Exception:
            pass
        if out.get("terminated") or out.get("truncated"):
            ended[0] = True
        return out

    def record_motion(label, res):
        entry = {"label": label}
        try:
            entry["ok"] = bool(res.ok)
            entry["status"] = str(res.status)
            entry["steps"] = int(res.steps_executed)
        except Exception as exc:
            entry["read_error"] = str(exc)
        entry.update(flags_of(res))
        try:
            st = get_robot_state()
            entry["ee"] = [r4(v) for v in st.end_effector_poses["primary"].position]
            entry["g"] = r4(st.gripper_positions["primary"])
            entry["tick"] = int(round(float(st.timestamp_s) * hz))
        except Exception as exc:
            entry["state_error"] = str(exc)
            ended[0] = True
        report["motions"].append(entry)
        return entry

    def hold(nticks, gcmd):
        st = get_robot_state()
        for k in range(nticks):
            if ended[0]:
                break
            try:
                sr = step(RobotAction(arms={"primary": ArmCommand(
                    mode="joint_position", target=st.joint_positions["primary"],
                    gripper_position=gcmd, embodiment=emb)}))
            except Exception as exc:
                note_error("hold_step", exc)
                ended[0] = True
                break
            flags_of(sr)

    attempt = {"mode": None}
    report["attempt"] = attempt

    if scene_static and grasp_idx is not None:
        # ======================================================== joint replay
        # Live scene == video's initial scene: re-issue the expert's measured joint
        # path with a 2-frame lead (target at tick k = video joints at frame k+2; v01's
        # 1-frame lead lagged ~1.5-2 ticks) and the expert's gripper state.
        # v02 additions: (1) in-hand peg correction measured from the wrist camera while
        # the expert is stationary after the lift; (2) a bounded lateral/vertical search
        # if the insertion still stalls.
        attempt["mode"] = "joint_replay"
        if joint_gap is not None and joint_gap > 0.02:
            record_motion("to_video_start", move_to_joints(j0v, tolerance=0.01, max_steps=120))
        VJ = [[float(v) for v in video[i].robot_state.joint_positions["primary"]] for i in range(nv)]
        gfinal = 0.0 if (release_idx is None) else 1.0

        def hand_of(rs):
            pe = rs.end_effector_poses["primary"]
            return vec(pe.position), qnorm(vec(pe.quaternion_wxyz))

        def blue_inhand(cam, rs):
            """Held-peg points from WRIST DEPTH (colour is unusable: moving spotlights
            saturate the peg), expressed in the tool frame. Keeps points beside the
            fingers along tool x (0.025 < |x| < 0.12 m), between the jaws (|y| < 0.035 m)
            and near the tool plane (|z| < 0.035 m); uses the side with more points.
            Returns (centroid xyz, axis angle in tool xy pointing away from the grasp, n)."""
            m = cam_model(cam)
            dep = depth_rows(cam.depth_m)
            hp, hq = hand_of(rs)
            R = qmat(hq)
            side_pos = []
            side_neg = []
            for v in range(0, len(dep), 2):
                for u in range(0, len(dep[0]), 2):
                    d = float(dep[v][u])
                    if d <= 0.005 or d > 0.30:
                        continue
                    P = deproject(m, u, v, d, conv)
                    w = [P[0] - hp[0], P[1] - hp[1], P[2] - hp[2]]
                    h = [R[0][i] * w[0] + R[1][i] * w[1] + R[2][i] * w[2] for i in range(3)]
                    if abs(h[1]) < 0.035 and abs(h[2]) < 0.035 and 0.025 < abs(h[0]) < 0.12:
                        if h[0] > 0:
                            side_pos.append(h)
                        else:
                            side_neg.append(h)
            pts = side_pos if len(side_pos) >= len(side_neg) else side_neg
            n = len(pts)
            if n < 30:
                return None
            mx = sum(p[0] for p in pts) / n
            my = sum(p[1] for p in pts) / n
            mz = sum(p[2] for p in pts) / n
            sxx = sum((p[0] - mx) ** 2 for p in pts) / n
            syy = sum((p[1] - my) ** 2 for p in pts) / n
            sxy = sum((p[0] - mx) * (p[1] - my) for p in pts) / n
            ang = 0.5 * math.atan2(2 * sxy, sxx - syy)
            if math.cos(ang) * mx + math.sin(ang) * my < 0:
                ang = wrap(ang + math.pi)
            return {"c": [mx, my, mz], "ang": ang, "n": n}

        def step_joints(target, gcmd):
            sr = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=target, gripper_position=gcmd, embodiment=emb)}))
            return flags_of(sr)

        trace = []
        corr = {"applied": False}
        attempt["inhand_correction"] = corr
        MEAS_K = 200
        k = 0
        plan = None  # corrected joint targets by frame index
        while k < nv - 1:
            if ended[0]:
                break
            f = min(nv - 1, k + 2)
            target = VJ[f] if plan is None or f not in plan else plan[f]
            closed = k >= grasp_idx and (release_idx is None or k < release_idx)
            gcmd = 0.0 if closed else 1.0
            try:
                fl = step_joints(target, gcmd)
            except Exception as exc:
                note_error("replay_step" + str(k), exc)
                ended[0] = True
                break
            if k % 10 == 0 or ended[0]:
                row = {"k": k, "flags": fl}
                if not ended[0]:
                    st = get_robot_state()
                    pe = st.end_effector_poses["primary"].position
                    vp = video[k + 1].robot_state.end_effector_poses["primary"].position
                    row["ee"] = [r4(v) for v in pe]
                    row["dev_mm"] = round(1000.0 * math.sqrt(sum((float(pe[i]) - float(vp[i])) ** 2 for i in range(3))), 1)
                    row["g"] = r4(st.gripper_positions["primary"])
                trace.append(row)
            # ---- in-hand correction, measured while the expert is stationary after lift
            if k == MEAS_K and k > grasp_idx and not ended[0]:
                try:
                    lobs = get_observation()
                    live_m = blue_inhand(lobs.cameras["wrist"], get_robot_state())
                    vid_m = blue_inhand(video[k + 1].cameras["wrist"], video[k + 1].robot_state)
                    corr["live"] = None if live_m is None else {"c": [r4(x) for x in live_m["c"]], "ang_deg": r4(live_m["ang"] * 180 / math.pi), "n": live_m["n"]}
                    corr["video"] = None if vid_m is None else {"c": [r4(x) for x in vid_m["c"]], "ang_deg": r4(vid_m["ang"] * 180 / math.pi), "n": vid_m["n"]}
                    sane = live_m is not None and vid_m is not None
                    for mm in (live_m, vid_m):
                        if mm is not None and (abs(mm["c"][2]) > 0.04 or
                                               math.sqrt(mm["c"][0] ** 2 + mm["c"][1] ** 2) > 0.11 or
                                               abs(math.cos(mm["ang"])) < 0.8 or mm["n"] < 60):
                            sane = False
                    corr["sane"] = sane
                    if sane:
                        dl = wrap(vid_m["ang"] - live_m["ang"])
                        cz = math.cos(dl)
                        sz = math.sin(dl)
                        pl = live_m["c"]
                        rot_pl = [cz * pl[0] - sz * pl[1], sz * pl[0] + cz * pl[1], pl[2]]
                        t = [vid_m["c"][i] - rot_pl[i] for i in range(3)]
                        tn = math.sqrt(sum(x * x for x in t))
                        corr["delta_deg"] = r4(dl * 180 / math.pi)
                        corr["t_hand"] = [r4(x) for x in t]
                        if abs(dl) <= 0.2 and tn <= 0.025 and (abs(dl) > 0.005 or tn > 0.0015):
                            # corrected hand pose H'(f) = H_exp(f) * D, D = (Rz(dl), t) in hand frame
                            newplan = {}
                            seed = [float(v) for v in get_robot_state().joint_positions["primary"]]
                            fails = 0
                            worst = 0.0
                            frames = list(range(k + 3, nv, 2))
                            if frames[-1] != nv - 1:
                                frames.append(nv - 1)
                            for fi in frames:
                                hp, hq = hand_of(video[fi].robot_state)
                                R = qmat(hq)
                                np2 = [hp[i] + R[i][0] * t[0] + R[i][1] * t[1] + R[i][2] * t[2] for i in range(3)]
                                nq = qnorm(qmul(hq, qz(dl)))
                                ik = solve_ik(Pose(tuple(np2), tuple(nq), base), seed_joints=seed)
                                if not ik.ok:
                                    fails += 1
                                    continue
                                sol = [float(v) for v in ik.joint_positions]
                                gap = max(abs(sol[i] - VJ[fi][i]) for i in range(7))
                                worst = max(worst, gap)
                                newplan[fi] = sol
                                seed = sol
                            corr["ik_fails"] = fails
                            corr["worst_joint_gap"] = r4(worst)
                            if fails <= 3 and worst < 0.6 and (nv - 1) in newplan:
                                # fill odd frames by interpolation, keep frames up to k+2 as video
                                keys = sorted(newplan.keys())
                                full = {}
                                for a in range(len(keys) - 1):
                                    fa = keys[a]
                                    fb = keys[a + 1]
                                    for fi in range(fa, fb):
                                        w = (fi - fa) / float(fb - fa)
                                        full[fi] = [newplan[fa][i] * (1 - w) + newplan[fb][i] * w for i in range(7)]
                                full[keys[-1]] = newplan[keys[-1]]
                                # blend in over the first 10 frames from the video targets
                                f0 = keys[0]
                                for fi in range(f0, f0 + 10):
                                    if fi in full:
                                        w = (fi - f0 + 1) / 10.0
                                        full[fi] = [VJ[fi][i] * (1 - w) + full[fi][i] * w for i in range(7)]
                                plan = full
                                corr["applied"] = True
                                corr["t_vec"] = t
                                corr["dl"] = dl
                except Exception as exc:
                    note_error("inhand_correction", exc)
            k += 1
        attempt["replay_trace"] = trace
        result = report

        if not ended[0]:
            hold(10, gfinal)

        # ---- stall check and bounded search along the expert's final insertion direction
        search = {"ran": False, "probes": []}
        attempt["search"] = search
        if not ended[0] and release_idx is None and nv > 80:
            pa = vec(video[nv - 81].robot_state.end_effector_poses["primary"].position)
            pb = vec(video[nv - 1].robot_state.end_effector_poses["primary"].position)
            dvec = [pb[i] - pa[i] for i in range(3)]
            ln = math.sqrt(sum(x * x for x in dvec))
            st = get_robot_state()
            pe = vec(st.end_effector_poses["primary"].position)
            qe = qnorm(vec(st.end_effector_poses["primary"].quaternion_wxyz))
            if ln > 0.02:
                dn = [x / ln for x in dvec]
                # how far short of the expert's final point along the insertion direction
                goal_end = pb
                if corr.get("applied"):
                    Rf = qmat(qnorm(vec(video[nv - 1].robot_state.end_effector_poses["primary"].quaternion_wxyz)))
                    tv = corr["t_vec"]
                    goal_end = [pb[i] + Rf[i][0] * tv[0] + Rf[i][1] * tv[1] + Rf[i][2] * tv[2] for i in range(3)]
                short = sum((goal_end[i] - pe[i]) * dn[i] for i in range(3))
                search["short_m"] = r4(short)
                search["dir"] = [r4(x) for x in dn]
                if short > 0.008:
                    search["ran"] = True
                    lat = [-dn[1], dn[0], 0.0]
                    ll = math.sqrt(lat[0] ** 2 + lat[1] ** 2)
                    lat = [lat[0] / ll, lat[1] / ll, 0.0]
                    if lat[0] < 0:
                        lat = [-lat[0], -lat[1], 0.0]   # +lat ~ world +x (v01 wrist evidence)
                    stall = pe
                    offsets = [(0.006, 0.0), (0.010, 0.0), (0.003, 0.003), (0.006, 0.004),
                               (0.006, -0.003), (-0.005, 0.0), (0.014, 0.0), (0.0, 0.005)]

                    def servo(pos, quat, nticks):
                        cur = [float(v) for v in get_robot_state().joint_positions["primary"]]
                        ik = solve_ik(Pose(tuple(pos), tuple(quat), base), seed_joints=cur)
                        if not ik.ok:
                            return False
                        sol = [float(v) for v in ik.joint_positions]
                        for s in range(1, nticks + 1):
                            if ended[0]:
                                return True
                            w = s / float(nticks)
                            step_joints([cur[i] * (1 - w) + sol[i] * w for i in range(7)], 0.0)
                        return True

                    for off in offsets:
                        if ended[0] or tick_now() > max_steps - 45:
                            break
                        base_pt = [stall[i] - 0.010 * dn[i] + off[0] * lat[i] for i in range(3)]
                        base_pt[2] += off[1]
                        push_pt = [stall[i] + (short + 0.005) * dn[i] + off[0] * lat[i] for i in range(3)]
                        push_pt[2] += off[1]
                        okb = servo(base_pt, qe, 8)
                        okp = servo(push_pt, qe, 22)
                        if not ended[0]:
                            hold(4, 0.0)
                        pr = {"off": [off[0], off[1]], "ik": [okb, okp]}
                        if not ended[0]:
                            now = vec(get_robot_state().end_effector_poses["primary"].position)
                            gain = sum((now[i] - stall[i]) * dn[i] for i in range(3))
                            pr["gain_m"] = r4(gain)
                            pr["tick"] = tick_now()
                        else:
                            pr["ended"] = True
                        search["probes"].append(pr)
                        if ended[0]:
                            break
                        if pr.get("gain_m", 0.0) > 0.012:
                            search["entered"] = pr["off"]
                            hold(10, 0.0)
                            break
        if not ended[0]:
            try:
                st = get_robot_state()
                report["motions"].append({"label": "final",
                                          "ee": [r4(v) for v in st.end_effector_poses["primary"].position],
                                          "g": r4(st.gripper_positions["primary"]),
                                          "tick": int(round(float(st.timestamp_s) * hz))})
            except Exception as exc:
                note_error("final_state", exc)
    elif grasp_idx is not None and len(pegs) > 0:
        # ======================================================== fallback
        # Scene differs from the video: Cartesian replay. Grasp the live peg nearest
        # the expert's grasp point at the same offset/orientation relative to that peg
        # (translation only), then replay the expert's post-grasp EE path.
        attempt["mode"] = "cartesian_fallback"
        gpv = timeline[grasp_idx]["p"]
        gqv = qnorm(vec(video[grasp_idx].robot_state.end_effector_poses["primary"].quaternion_wxyz))
        live_peg = None
        for p in pegs:
            if live_peg is None or dist2(p["shape"]["centre"], gpv) < dist2(live_peg["shape"]["centre"], gpv):
                live_peg = p
        attempt["live_peg"] = shape_summary(live_peg["shape"])
        # expert's peg in video frame 0: depth points near the grasp point at peg height
        vc = video[0].cameras["front"]
        vm = cam_model(vc)
        vd = depth_rows(vc.depth_m)
        vpts = []
        for v in range(0, size, 2):
            for u in range(0, size, 2):
                d = float(vd[v][u])
                if d <= 0.0:
                    continue
                P = deproject(vm, u, v, d, conv)
                if P[2] < 0.008 or P[2] > live_peg["shape"]["z95"] + 0.01:
                    continue
                if dist2(P, gpv) < 0.06:
                    vpts.append(P)
        vshape = planar_shape(vpts)
        attempt["video_peg"] = shape_summary(vshape)
        off = [0.0, 0.0]
        if vshape is not None:
            off = [live_peg["shape"]["centre"][0] - vshape["centre"][0],
                   live_peg["shape"]["centre"][1] - vshape["centre"][1]]
        gp = [gpv[0] + off[0], gpv[1] + off[1], gpv[2]]
        attempt["grasp_offset"] = [r4(x) for x in off]

        def go(label, p, q, tol, steps):
            if ended[0]:
                return None
            return record_motion(label, move_to_pose(Pose(tuple(p), tuple(qnorm(q)), base),
                                                     tolerance=tol, max_steps=steps))

        open_gripper()
        go("hover", [gp[0], gp[1], gp[2] + 0.10], gqv, 0.01, 200)
        go("pregrasp", [gp[0], gp[1], gp[2] + 0.04], gqv, 0.008, 120)
        go("grasp", gp, gqv, 0.005, 120)
        if not ended[0]:
            record_motion("close", close_gripper())
        go("lift", [gp[0], gp[1], gp[2] + 0.12], gqv, 0.01, 150)
        end_idx = release_idx if release_idx is not None else nv - 1
        last_p = None
        for i in range(grasp_idx + 1, end_idx + 1):
            if ended[0] or tick_now() > max_steps - 15:
                break
            p = timeline[i]["p"]
            if p[2] < gpv[2] + 0.08 and last_p is None:
                continue
            if last_p is not None and i != end_idx and \
                    math.sqrt(sum((p[k] - last_p[k]) ** 2 for k in range(3))) < 0.02:
                continue
            q = vec(video[i].robot_state.end_effector_poses["primary"].quaternion_wxyz)
            go("wp" + str(i), p, q, 0.005, 120)
            last_p = p
        if release_idx is not None and not ended[0]:
            record_motion("release", open_gripper())
        hold(10, 0.0 if release_idx is None else 1.0)
    else:
        attempt["mode"] = "none"
        open_gripper()
    try:
        report["final_tick"] = tick_now()
    except Exception as exc:
        note_error("final_tick", exc)
except Exception as exc:
    note_error("top", exc)

result = report
