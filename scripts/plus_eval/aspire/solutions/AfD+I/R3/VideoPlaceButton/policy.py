import math

# ---- vision core (pure Python; sandbox-safe: no imports beyond math, no underscores) ----
PIX_STRIDE = 2


def rgb_rows(camera):
    rgb = camera.rgb
    try:
        return rgb.tolist()
    except Exception:
        return rgb


def scan_frame(rows, cube_color, stride):
    """Return (target_pixels, cube_pixels) as lists of (x, y) on a stride grid (full-res coords)."""
    tpix = []
    cpix = []
    height = len(rows)
    for y in range(6, height - 2, stride):
        row = rows[y]
        width = len(row)
        for x in range(0, width, stride):
            p = row[x]
            r = int(p[0])
            g = int(p[1])
            b = int(p[2])
            if r > 140 and b - g > 5 and r - g > 12:
                tpix.append((x, y))
            elif cube_color == "red":
                if r > 140 and g < 60 and b < 60:
                    cpix.append((x, y))
            elif cube_color == "blue":
                if b > 140 and r < 60 and g < 60:
                    cpix.append((x, y))
            elif cube_color == "green":
                if g > 140 and r < 80 and b < 80:
                    cpix.append((x, y))
    return tpix, cpix


def blobs_of(pixels, stride):
    """4/8-connected components on the stride grid; returns list of [cx, cy, count]."""
    remaining = set(pixels)
    out = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        sx = 0.0
        sy = 0.0
        count = 0
        while stack:
            q = stack.pop()
            sx += q[0]
            sy += q[1]
            count += 1
            for dx in (-stride, 0, stride):
                for dy in (-stride, 0, stride):
                    nb = (q[0] + dx, q[1] + dy)
                    if nb in remaining:
                        remaining.discard(nb)
                        stack.append(nb)
        out.append([sx / count, sy / count, count])
    return out


def dist2d(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[len(ordered) // 2]


def find_slots(target_blobs, min_count):
    return [[b[0], b[1], b[2]] for b in target_blobs if b[2] >= min_count]


def detect_swaps(frames, slots):
    """frames: list of dict(t, tblobs). Returns (swaps, runs_diag)."""
    if len(slots) < 2:
        return [], []
    spacing = None
    for i in range(len(slots)):
        for j in range(i + 1, len(slots)):
            d = dist2d(slots[i], slots[j])
            if spacing is None or d < spacing:
                spacing = d
    sizes = sorted(s[2] for s in slots)
    typical = sizes[len(sizes) // 2]
    occ_radius = max(6.0, 0.3 * spacing)
    off_radius = max(9.0, 0.4 * spacing)
    min_off = max(3, int(0.35 * typical))
    per = []
    for fr in frames:
        occ = []
        for s in slots:
            hit = False
            for b in fr["tblobs"]:
                if dist2d(b, s) <= occ_radius:
                    hit = True
                    break
            occ.append(hit)
        off = []
        for b in fr["tblobs"]:
            if b[2] < min_off:
                continue
            near = min(dist2d(b, s) for s in slots)
            if near > off_radius:
                off.append(b)
        per.append({"t": fr["t"], "occ": occ, "off": off})
    runs = []
    cur = None
    gap = 0
    for idx in range(len(per)):
        if per[idx]["off"]:
            if cur is None:
                cur = [idx, idx]
            else:
                cur[1] = idx
            gap = 0
        elif cur is not None:
            gap += 1
            if gap > 2:
                runs.append(cur)
                cur = None
                gap = 0
    if cur is not None:
        runs.append(cur)
    swaps = []
    diag = []
    for run in runs:
        idxs = list(range(run[0], run[1] + 1))
        best = None
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                a = slots[i]
                b = slots[j]
                mx = 0.5 * (a[0] + b[0])
                my = 0.5 * (a[1] + b[1])
                rad = 0.5 * dist2d(a, b)
                if rad < 3:
                    continue
                errs = []
                empty = 0
                for k in idxs:
                    fr = per[k]
                    if not fr["occ"][i]:
                        empty += 1
                    if not fr["occ"][j]:
                        empty += 1
                    offs = fr["off"]
                    scored = sorted([[abs(dist2d(p, (mx, my)) - rad) / max(rad, 15.0), p] for p in offs])
                    scored = scored[:2]
                    if len(scored) == 2 and scored[1][0] > 0.35:
                        scored = scored[:1]
                    for item in scored:
                        errs.append(item[0])
                    if len(scored) == 2:
                        p1 = scored[0][1]
                        p2 = scored[1][1]
                        sx = p1[0] + p2[0] - (a[0] + b[0])
                        sy = p1[1] + p2[1] - (a[1] + b[1])
                        errs.append(math.sqrt(sx * sx + sy * sy) / (2.0 * max(rad, 15.0)))
                if not errs:
                    continue
                err = sum(errs) / len(errs)
                empty_frac = empty / (2.0 * len(idxs))
                score = err + 0.5 * (1.0 - empty_frac)
                if best is None or score < best[0]:
                    best = (score, i, j, err, empty_frac)
        n_off = sum(len(per[k]["off"]) for k in idxs)
        entry = {"t0": round(per[run[0]]["t"], 2), "t1": round(per[run[1]]["t"], 2),
                 "frames": len(idxs), "n_off": n_off}
        if best is not None:
            entry["pair"] = [best[1], best[2]]
            entry["err"] = round(best[3], 3)
            entry["empty"] = round(best[4], 3)
            entry["score"] = round(best[0], 3)
        accepted = (best is not None and n_off >= 2 and
                    ((best[3] < 0.3 and best[4] >= 0.5) or (best[3] < 0.45 and best[4] >= 0.9)))
        entry["accepted"] = accepted
        diag.append(entry)
        if accepted:
            swaps.append({"t0": per[run[0]]["t"], "t1": per[run[1]]["t"], "a": best[1], "b": best[2]})
    return swaps, diag


def cube_rests(frames, slots, min_duration):
    """Stationary periods of the cube centroid. Returns list of dict(t0, t1, x, y, slot, slot_dist)."""
    rests = []
    anchor = None
    members = []
    for fr in frames:
        c = fr["cube"]
        if c is None:
            continue
        if anchor is not None and dist2d(c, anchor) <= 2.5:
            members.append(fr)
            continue
        if anchor is not None and members[-1]["t"] - members[0]["t"] >= min_duration:
            rests.append(members)
        anchor = c
        members = [fr]
    if anchor is not None and members and members[-1]["t"] - members[0]["t"] >= min_duration:
        rests.append(members)
    out = []
    for mem in rests:
        x = median([m["cube"][0] for m in mem])
        y = median([m["cube"][1] for m in mem])
        best = None
        for si in range(len(slots)):
            d = dist2d((x, y), slots[si])
            if best is None or d < best[1]:
                best = (si, d)
        out.append({"t0": mem[0]["t"], "t1": mem[-1]["t"], "x": x, "y": y,
                    "slot": best[0] if best else None, "slot_dist": best[1] if best else None})
    # merge consecutive rests at the same place (occlusion splits)
    merged = []
    for r in out:
        if merged and dist2d((r["x"], r["y"]), (merged[-1]["x"], merged[-1]["y"])) <= 6.0:
            merged[-1]["t1"] = r["t1"]
        else:
            merged.append(r)
    return merged


def follow_swaps(slot, swaps, after_t):
    cur = slot
    applied = []
    for sw in swaps:
        if sw["t0"] < after_t:
            continue
        if cur == sw["a"]:
            cur = sw["b"]
            applied.append([sw["a"], sw["b"], round(sw["t0"], 2)])
        elif cur == sw["b"]:
            cur = sw["a"]
            applied.append([sw["b"], sw["a"], round(sw["t0"], 2)])
    return cur, applied

# ---- task program: VideoPlaceButton ----
report = {"version": "v01"}
log = []


def note(key, value):
    report[key] = value


def is_terminal(res):
    try:
        if bool(res.terminated) or bool(res.truncated):
            return True
    except Exception:
        pass
    return False


def res_info(res):
    info = {}
    try:
        info["ok"] = bool(res.ok)
    except Exception:
        pass
    try:
        info["status"] = str(res.status)
    except Exception:
        pass
    try:
        info["steps"] = int(res.steps_executed)
    except Exception:
        pass
    try:
        info["term"] = [bool(res.terminated), bool(res.truncated)]
    except Exception:
        pass
    try:
        if res.error is not None:
            info["error"] = str(res.error)[:300]
    except Exception:
        pass
    return info


def cloud_points(mask, camera_name, base_frame):
    try:
        pc = mask_to_point_cloud(mask, camera_name, target_frame=base_frame)
        return [[float(p[0]), float(p[1]), float(p[2])] for p in pc.points]
    except Exception as exc:
        log.append("cloud error " + str(exc)[:200])
        return []


def quant(values, frac):
    ordered = sorted(values)
    if not ordered:
        return None
    k = int((len(ordered) - 1) * frac)
    return ordered[k]


def cube_color_from_goal(goal):
    text = goal.lower()
    for c in ("red", "blue", "green"):
        if c in text:
            return c
    return None


def white_blob(rows, stride):
    pix = []
    for y in range(0, len(rows), stride):
        row = rows[y]
        for x in range(0, len(row), stride):
            p = row[x]
            lo = min(p[0], p[1], p[2])
            hi = max(p[0], p[1], p[2])
            if lo > 205 and hi - lo < 25:
                pix.append((x, y))
    bl = blobs_of(pix, stride)
    best = None
    for b in bl:
        if best is None or b[2] > best[2]:
            best = b
    return best, bl


def pose_xyz(pose):
    return [float(pose.position[0]), float(pose.position[1]), float(pose.position[2])]


stop = False
done_success_guess = False
try:
    ctx = get_task_context()
    goal = str(ctx.language)
    note("goal", goal)
    try:
        meta = robomme_plus.get_task_metadata()
        md = {}
        for k in meta:
            v = meta[k]
            md[str(k)] = str(v)[:200]
        note("task_metadata", md)
    except Exception as exc:
        note("task_metadata_error", str(exc)[:200])
    color = cube_color_from_goal(goal)
    note("cube_color", color)
    state0 = get_robot_state()
    base_frame = state0.base_frame
    home_pose = state0.end_effector_poses["primary"]
    down_q = [float(v) for v in home_pose.quaternion_wxyz]
    note("home_ee", pose_xyz(home_pose))
    note("home_q", down_q)
    note("t_start", float(state0.timestamp_s))

    live = get_observation()
    front = live.cameras["front"]
    live_rows = rgb_rows(front)
    stride = PIX_STRIDE
    ltp, lcp = scan_frame(live_rows, color, stride)
    lblobs = blobs_of(ltp, stride)
    maxc = max([b[2] for b in lblobs] + [1])
    slots = find_slots(lblobs, max(6, int(0.2 * maxc)))
    note("slots_px", [[round(s[0], 1), round(s[1], 1), s[2]] for s in slots])
    cblobs = blobs_of(lcp, stride)
    live_cube_px = None
    for b in cblobs:
        if live_cube_px is None or b[2] > live_cube_px[2]:
            live_cube_px = b
    note("live_cube_px", live_cube_px)
    try:
        note("live_cam", {"pos": [float(v) for v in front.camera_pose.position],
                          "q": [float(v) for v in front.camera_pose.quaternion_wxyz]})
    except Exception:
        pass

    # ---- conditioning video ----
    video = robomme_plus.get_conditioning_video()
    nvid = len(video)
    note("video_frames", nvid)
    frames = []
    ee_track = []
    vid_cam = None
    for vi in range(nvid):
        fr = video[vi]
        cam = fr.cameras["front"]
        rows = rgb_rows(cam)
        if rows is None:
            continue
        tp, cp = scan_frame(rows, color, stride)
        tb = blobs_of(tp, stride)
        cb = blobs_of(cp, stride)
        cube = None
        for b in cb:
            if b[2] >= 3 and (cube is None or b[2] > cube[2]):
                cube = b
        t = float(fr.timestamp_s)
        frames.append({"t": t, "tblobs": tb, "cube": [cube[0], cube[1]] if cube else None})
        try:
            rs = fr.robot_state
            ee = rs.end_effector_poses["primary"].position
            ee_track.append([t, float(ee[0]), float(ee[1]), float(ee[2]),
                             float(rs.gripper_positions["primary"])])
        except Exception:
            pass
        if vi == 0:
            try:
                vid_cam = {"pos": [float(v) for v in cam.camera_pose.position],
                           "q": [float(v) for v in cam.camera_pose.quaternion_wxyz]}
            except Exception:
                pass
    note("video_cam0", vid_cam)
    note("video_scanned", len(frames))
    if frames:
        note("video_t_range", [frames[0]["t"], frames[-1]["t"]])
        last_tb = frames[-1]["tblobs"]
        match = 0
        for s in slots:
            for b in last_tb:
                if dist2d(b, s) <= 5:
                    match += 1
                    break
        note("slots_matched_in_last_video_frame", match)
    step_ee = max(1, len(ee_track) // 150)
    note("ee_track_sub", [[round(v, 3) for v in e] for e in ee_track[::step_ee]])

    swaps, sdiag = detect_swaps(frames, slots)
    note("swap_runs", sdiag)
    rests = cube_rests(frames, slots, 0.4)
    note("cube_rests", [{"t0": round(r["t0"], 2), "t1": round(r["t1"], 2), "x": round(r["x"], 1),
                         "y": round(r["y"], 1), "slot": r["slot"],
                         "d": round(r["slot_dist"], 1) if r["slot_dist"] is not None else None}
                        for r in rests])

    # ---- button location (live) ----
    button_xyz = None
    wb, wbl = white_blob(live_rows, stride)
    note("white_blob", wb)
    bdiag = []
    for prompt in ("button", "white button", "gray button"):
        try:
            seg = segment_text("front", prompt)
        except Exception as exc:
            bdiag.append([prompt, "error " + str(exc)[:120]])
            continue
        if not seg.ok or len(seg.segmentations) == 0:
            bdiag.append([prompt, "none"])
            continue
        chosen = None
        cands = []
        for ci in range(len(seg.segmentations)):
            cand = seg.segmentations[ci]
            box = None
            try:
                box = [float(v) for v in cand.box_xyxy]
            except Exception:
                pass
            sc = None
            try:
                sc = float(cand.score)
            except Exception:
                pass
            cands.append([box, sc])
            if chosen is None and box is not None and wb is not None:
                if box[0] - 3 <= wb[0] <= box[2] + 3 and box[1] - 3 <= wb[1] <= box[3] + 3:
                    chosen = cand
        bdiag.append([prompt, cands])
        if chosen is None and wb is None:
            chosen = seg.segmentations[0]
        if chosen is not None:
            pts = cloud_points(chosen, "front", base_frame)
            if len(pts) >= 5:
                button_xyz = [median([p[0] for p in pts]), median([p[1] for p in pts]),
                              quant([p[2] for p in pts], 0.9)]
                break
    if button_xyz is None and wb is not None:
        try:
            seg = segment_points("front", [[int(wb[0]), int(wb[1])]])
            if seg.ok and len(seg.segmentations) > 0:
                pts = cloud_points(seg.segmentations[0], "front", base_frame)
                if len(pts) >= 5:
                    button_xyz = [median([p[0] for p in pts]), median([p[1] for p in pts]),
                                  quant([p[2] for p in pts], 0.9)]
                    bdiag.append(["points", "ok"])
        except Exception as exc:
            bdiag.append(["points", str(exc)[:120]])
    note("button_diag", bdiag)
    note("button_xyz", button_xyz)

    # ---- press time from recorded robot state ----
    # The press is the low point of a closed, EMPTY gripper (holding a cube leaves it part-open).
    press_t = None
    press_method = None
    if ee_track:
        gmax = max(e[4] for e in ee_track)
        closed = [e for e in ee_track if e[4] < 0.15 * gmax]
        note("gripper_max", gmax)
        note("closed_frames", len(closed))
        if button_xyz is not None and closed:
            nearb = [e for e in closed
                     if math.sqrt((e[1] - button_xyz[0]) ** 2 + (e[2] - button_xyz[1]) ** 2) < 0.06]
            note("closed_near_button", len(nearb))
            if nearb:
                closed = nearb
                press_method = "closed_gripper_low_point_near_button"
        if closed:
            zmin = min(e[3] for e in closed)
            for e in closed:
                if e[3] <= zmin + 0.01:
                    press_t = e[0]
                    break
            if press_method is None:
                press_method = "closed_gripper_low_point"
            note("press_zmin", zmin)
        if press_t is None and button_xyz is not None:
            near = [e for e in ee_track
                    if math.sqrt((e[1] - button_xyz[0]) ** 2 + (e[2] - button_xyz[1]) ** 2) < 0.04]
            if near:
                zmin = min(e[3] for e in near)
                for e in near:
                    if e[3] <= zmin + 0.01:
                        press_t = e[0]
                        break
                press_method = "ee_near_button"
    note("press_t", press_t)

    placements = [r for r in rests if r["slot_dist"] is not None and r["slot_dist"] <= 14.0
                  and r["t1"] - r["t0"] >= 1.0]
    chosen_rest = None
    if press_t is not None:
        for r in placements:
            if r["t0"] > press_t:
                chosen_rest = r
                break
    if chosen_rest is None and press_t is None and len(placements) >= 2:
        # fail-safe when the press could not be located: report it explicitly
        press_method = "fallback_second_on_target_rest"
        chosen_rest = placements[1]
    note("press_method", press_method)
    if chosen_rest is None:
        raise ValueError("could not determine the post-press placement")
    final_slot, applied = follow_swaps(chosen_rest["slot"], swaps, chosen_rest["t0"])
    note("placed_slot", chosen_rest["slot"])
    note("placed_t", chosen_rest["t0"])
    note("swaps_applied", applied)
    note("final_slot", final_slot)
    goal_px = slots[final_slot]
    note("goal_px", [round(goal_px[0], 1), round(goal_px[1], 1)])

    # ---- 3D target and cube in the live scene ----
    tseg = segment_points("front", [[int(round(goal_px[0])), int(round(goal_px[1]))]])
    tpts = []
    if tseg.ok and len(tseg.segmentations) > 0:
        tpts = cloud_points(tseg.segmentations[0], "front", base_frame)
    note("target_points", len(tpts))
    if len(tpts) < 5:
        raise ValueError("target point cloud empty")
    tx = median([p[0] for p in tpts])
    ty = median([p[1] for p in tpts])
    table_z = median([p[2] for p in tpts])
    note("target_xyz", [tx, ty, table_z])

    cpts = []
    cdiag = []
    for prompt in (color + " cube", color + " block"):
        seg = segment_text("front", prompt)
        if not seg.ok or len(seg.segmentations) == 0:
            cdiag.append([prompt, "none"])
            continue
        chosen = None
        for ci in range(len(seg.segmentations)):
            cand = seg.segmentations[ci]
            try:
                box = [float(v) for v in cand.box_xyxy]
            except Exception:
                continue
            if live_cube_px is not None and box[0] - 2 <= live_cube_px[0] <= box[2] + 2 and box[1] - 2 <= live_cube_px[1] <= box[3] + 2:
                chosen = cand
                break
        cdiag.append([prompt, "matched" if chosen is not None else "nomatch"])
        if chosen is not None:
            cpts = cloud_points(chosen, "front", base_frame)
            if len(cpts) >= 5:
                break
    if len(cpts) < 5 and live_cube_px is not None:
        seg = segment_points("front", [[int(round(live_cube_px[0])), int(round(live_cube_px[1]))]])
        if seg.ok and len(seg.segmentations) > 0:
            cpts = cloud_points(seg.segmentations[0], "front", base_frame)
            cdiag.append(["points", len(cpts)])
    note("cube_diag", cdiag)
    if len(cpts) < 5:
        raise ValueError("cube point cloud empty")
    cx = median([p[0] for p in cpts])
    cy = median([p[1] for p in cpts])
    ctop = quant([p[2] for p in cpts], 0.9)
    cube_h = ctop - table_z
    note("cube_xyz_top", [cx, cy, ctop])
    note("cube_height", cube_h)
    if cube_h <= 0.005:
        raise ValueError("implausible cube height")

    # ---- cube yaw from top-face points (min-area rectangle over 0..90 deg) ----
    top_pts = [p for p in cpts if p[2] >= ctop - 0.006]
    yaw = 0.0
    if len(top_pts) >= 8:
        best_area = None
        for k in range(18):
            ang = math.radians(5.0 * k)
            ca = math.cos(ang)
            sa = math.sin(ang)
            us = [p[0] * ca + p[1] * sa for p in top_pts]
            vs = [-p[0] * sa + p[1] * ca for p in top_pts]
            area = (quant(us, 0.95) - quant(us, 0.05)) * (quant(vs, 0.95) - quant(vs, 0.05))
            if best_area is None or area < best_area:
                best_area = area
                yaw = ang
        if yaw > math.radians(45.0):
            yaw = yaw - math.radians(90.0)
    note("cube_yaw_deg", math.degrees(yaw))
    note("top_points", len(top_pts))
    hz = math.cos(0.5 * yaw)
    sz = math.sin(0.5 * yaw)
    w0 = down_q[0]
    x0 = down_q[1]
    y0 = down_q[2]
    z0 = down_q[3]
    grasp_q = [hz * w0 - sz * z0, hz * x0 - sz * y0, hz * y0 + sz * x0, hz * z0 + sz * w0]
    note("grasp_q", grasp_q)

    # ---- pick ----
    hover = max(0.08, 3.0 * cube_h)
    grasp_z = table_z + 0.5 * cube_h
    motions = []

    def go(label, xyz, quat, steps):
        res = move_to_pose(Pose((xyz[0], xyz[1], xyz[2]), quat, base_frame), tolerance=0.01, max_steps=steps)
        info = res_info(res)
        info["label"] = label
        info["cap"] = steps
        if (not info.get("ok", False)) and info.get("steps", 0) == 0 and "max_steps" in info.get("error", ""):
            res = move_to_pose(Pose((xyz[0], xyz[1], xyz[2]), quat, base_frame), tolerance=0.01, max_steps=2 * steps)
            info["retry"] = res_info(res)
        try:
            info["t"] = float(get_robot_state().timestamp_s)
            info["ee"] = pose_xyz(get_robot_state().end_effector_poses["primary"])
        except Exception:
            pass
        motions.append(info)
        return res

    r = open_gripper()
    motions.append({"label": "open", "info": res_info(r)})
    stop = is_terminal(r)
    if not stop:
        r = go("pre_grasp", (cx, cy, table_z + cube_h + hover), grasp_q, 100)
        stop = is_terminal(r)
    if not stop:
        r = go("grasp", (cx, cy, grasp_z), grasp_q, 60)
        stop = is_terminal(r)
    if not stop:
        r = close_gripper()
        info = res_info(r)
        info["label"] = "close"
        try:
            info["grip"] = float(get_robot_state().gripper_positions["primary"])
        except Exception:
            pass
        motions.append(info)
        stop = is_terminal(r)
    if not stop:
        r = go("lift", (cx, cy, table_z + cube_h + hover), grasp_q, 50)
        stop = is_terminal(r)
    if not stop:
        try:
            note("grip_after_lift", float(get_robot_state().gripper_positions["primary"]))
        except Exception:
            pass
        r = go("above_target", (tx, ty, table_z + cube_h + hover), grasp_q, 100)
        stop = is_terminal(r)
    if not stop:
        r = go("lower", (tx, ty, table_z + 0.5 * cube_h + 0.012), grasp_q, 60)
        stop = is_terminal(r)
    if not stop:
        r = open_gripper()
        info = res_info(r)
        info["label"] = "release"
        motions.append(info)
        stop = is_terminal(r)
    tick = 0
    if not stop:
        st = get_robot_state()
        for tick in range(10):
            rep = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=st.joint_positions["primary"],
                gripper_position=1.0, embodiment=st.embodiment)}))
            if is_terminal(rep):
                stop = True
                break
        motions.append({"label": "settle", "ticks": tick + 1, "stop": stop})
    if not stop:
        r = go("retreat", (tx, ty, table_z + cube_h + hover), grasp_q, 40)
        stop = is_terminal(r)
    note("motions", motions)
    note("stopped_on_terminal", stop)
except Exception as exc:
    note("error", str(exc)[:500])
note("log", log)
result = report
