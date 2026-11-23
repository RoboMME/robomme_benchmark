# v03_fast_pick_rest_model — robomme_plus PickHighlight (from v02)
# v02 failed seed 2 (short static gaps): actions overran into shuffles and one target track went
# stale. v03: (1) shuffle timing model = REST interval (shuffle end -> next start) + duration,
# learned only from shuffles whose start was watched from home ("clean"); anchored on the last
# observed shuffle end, so a late-detected start after an overrun no longer skews predictions;
# (2) faster pick: lower stand-off/lift, no wrist refine, release at lift height, straight home;
# press returns home directly; (3) track repair: tracks without pixel support are re-matched
# one-to-one to unclaimed cube blobs; (4) geometry fallback (snap to blob / segment_text box).
# No probe-measured coordinate/size/colour/timing is used as a constant.
import math

CAM = "wrist"
RING_ON = 0.5            # ring white fraction that counts as "lit"
HIGHLIGHT_DELTA = 0.45   # max-min ring fraction for a cube to count as highlighted
OFF_TICKS = 30           # ticks with no lit ring before we accept "highlight over"
STATIC_WINDOW = 10       # ticks over which motion is measured
STATIC_PX = 0.5          # max centroid motion (px) over STATIC_WINDOW to count as static
PRE_Z = 0.03             # pre-grasp stand-off above the cube top (m)
LIFT_Z = 0.04            # lift height above the cube top (m)
PRESS_ABOVE = 0.08       # stand-off above the button top (m)
PRESS_DEPTH = 0.02       # commanded depth below the measured button top (m)


# ---------------------------------------------------------------- pure-python vision helpers
def classify_frame(rows, table_chroma):
    tr = table_chroma[0]
    tg = table_chroma[1]
    cube_px = []
    white_px = []
    y = 0
    for row in rows:
        x = 0
        for px in row:
            r = px[0]
            g = px[1]
            b = px[2]
            hi = max(r, g, b)
            lo = min(r, g, b)
            if lo > 185 and hi - lo < 35:
                white_px.append((x, y))
            elif hi - lo >= 25:
                s = r + g + b
                cr = r / s - tr
                cg = g / s - tg
                if cr * cr + cg * cg > 0.0009:
                    cube_px.append((x, y))
            x += 1
        y += 1
    return cube_px, white_px


def median_chroma(rows, stride):
    crs = []
    cgs = []
    for y in range(0, len(rows), stride):
        row = rows[y]
        for x in range(0, len(row), stride):
            px = row[x]
            s = px[0] + px[1] + px[2] + 1
            crs.append(px[0] / s)
            cgs.append(px[1] / s)
    crs.sort()
    cgs.sort()
    return (crs[len(crs) // 2], cgs[len(cgs) // 2])


def components(pixels, width):
    occupied = set()
    for p in pixels:
        occupied.add(p[1] * width + p[0])
    seen = set()
    comps = []
    for p in pixels:
        key = p[1] * width + p[0]
        if key in seen:
            continue
        seen.add(key)
        stack = [key]
        comp = []
        while len(stack) > 0:
            k = stack.pop()
            cx = k % width
            cy = k // width
            comp.append((cx, cy))
            for nk in (k - 1, k + 1, k - width, k + width):
                if nk in occupied and nk not in seen:
                    nx = nk % width
                    if abs(nx - cx) <= 1:
                        seen.add(nk)
                        stack.append(nk)
        comps.append(comp)
    return comps


def centroid(pts):
    sx = 0.0
    sy = 0.0
    for p in pts:
        sx += p[0]
        sy += p[1]
    n = max(1, len(pts))
    return (sx / n, sy / n)


def touches_border(comp, width, height):
    for p in comp:
        if p[0] <= 0 or p[1] <= 0 or p[0] >= width - 1 or p[1] >= height - 1:
            return True
    return False


def kmeans_split(pts, k):
    c0 = centroid(pts)
    far = pts[0]
    best = -1.0
    for p in pts:
        d = (p[0] - c0[0]) ** 2 + (p[1] - c0[1]) ** 2
        if d > best:
            best = d
            far = p
    centers = [(far[0] * 1.0, far[1] * 1.0)]
    while len(centers) < k:
        best = -1.0
        far = pts[0]
        for p in pts:
            d = min((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 for c in centers)
            if d > best:
                best = d
                far = p
        centers.append((far[0] * 1.0, far[1] * 1.0))
    for it in range(8):
        groups = [[] for c in centers]
        for p in pts:
            bi = 0
            bd = None
            for i in range(len(centers)):
                d = (p[0] - centers[i][0]) ** 2 + (p[1] - centers[i][1]) ** 2
                if bd is None or d < bd:
                    bd = d
                    bi = i
            groups[bi].append(p)
        new_centers = []
        for i in range(len(centers)):
            if len(groups[i]) > 0:
                new_centers.append(centroid(groups[i]))
            else:
                new_centers.append(centers[i])
        centers = new_centers
    return centers


def init_tracks(cube_px, width, height):
    comps = []
    for c in components(cube_px, width):
        if len(c) >= 8 and not touches_border(c, width, height):
            comps.append(c)
    if len(comps) == 0:
        return [], 0.0
    areas = sorted(len(c) for c in comps)
    med = areas[len(areas) // 2]
    tracks = []
    for c in comps:
        if len(c) < 0.35 * med:
            continue
        k = int(round(len(c) / med))
        if k <= 1:
            tracks.append(centroid(c))
        else:
            for cen in kmeans_split(c, k):
                tracks.append(cen)
    return tracks, float(med)


def update_tracks(tracks, cube_px, gate, min_count):
    groups = [[] for t in tracks]
    g2 = gate * gate
    for p in cube_px:
        bi = -1
        bd = g2
        for i in range(len(tracks)):
            dx = p[0] - tracks[i][0]
            dy = p[1] - tracks[i][1]
            d = dx * dx + dy * dy
            if d < bd:
                bd = d
                bi = i
        if bi >= 0:
            groups[bi].append(p)
    out = []
    counts = []
    for i in range(len(tracks)):
        counts.append(len(groups[i]))
        if len(groups[i]) >= min_count:
            out.append(centroid(groups[i]))
        else:
            out.append(tracks[i])
    return out, counts


def ring_fractions(tracks, white_px, r_in, r_out):
    counts = [0 for t in tracks]
    a2 = r_in * r_in
    b2 = r_out * r_out
    for p in white_px:
        for i in range(len(tracks)):
            dx = p[0] - tracks[i][0]
            dy = p[1] - tracks[i][1]
            d = dx * dx + dy * dy
            if d >= a2 and d <= b2:
                counts[i] += 1
    area = 3.14159 * (b2 - a2)
    return [c / area for c in counts]


# ---------------------------------------------------------------- geometry helpers
def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def qmul(a, b):
    return (a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])


def yaw_quat(base_quat, yaw):
    qz = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
    return qmul(qz, base_quat)


def box_fit(points_xy):
    """Min-area rectangle by brute force over 2-degree steps (trimmed extents).
    Returns (cx, cy, ex, ey, angle_rad) with angle in [0, pi/2)."""
    best = None
    for k in range(45):
        a = k * math.pi / 90.0
        ca = math.cos(a)
        sa = math.sin(a)
        us = []
        vs = []
        for p in points_xy:
            us.append(p[0] * ca + p[1] * sa)
            vs.append(-p[0] * sa + p[1] * ca)
        u0 = quantile(us, 0.03)
        u1 = quantile(us, 0.97)
        v0 = quantile(vs, 0.03)
        v1 = quantile(vs, 0.97)
        area = (u1 - u0) * (v1 - v0)
        if best is None or area < best[0]:
            uc = (u0 + u1) / 2.0
            vc = (v0 + v1) / 2.0
            best = (area, uc * ca - vc * sa, uc * sa + vc * ca, u1 - u0, v1 - v0, a)
    return (best[1], best[2], best[3], best[4], best[5])


def cube_from_cloud(points):
    zs = [float(p[2]) for p in points]
    z_top = quantile(zs, 0.95)
    top = []
    for p in points:
        if float(p[2]) > z_top - 0.012:
            top.append((float(p[0]), float(p[1])))
    if len(top) < 6:
        return None
    cx, cy, ex, ey, ang = box_fit(top)
    yaw = ang
    if yaw > math.pi / 4.0:
        yaw = yaw - math.pi / 2.0
    side = (ex + ey) / 2.0
    return {"x": cx, "y": cy, "z_top": z_top, "side": side, "ex": ex, "ey": ey,
            "yaw": yaw, "n": len(points)}


# ---------------------------------------------------------------- robot helpers
context = get_task_context()
state0 = get_robot_state()
base = state0.base_frame
emb = state0.embodiment
home_joints = [float(v) for v in state0.joint_positions["primary"]]
q_down = tuple(float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz)
meta = robomme_plus.get_task_metadata()
max_steps = int(meta["max_steps"])

report = {"task": context.language, "max_steps": max_steps, "events": [], "errors": [],
          "track_log": [], "picks": [], "presses": []}
clock = {"ticks": 0, "over": False}


def note(msg):
    report["events"].append([clock["ticks"], msg])


def check_end(res):
    try:
        if bool(res.terminated) or bool(res.truncated):
            clock["over"] = True
            note("episode ended (terminated=" + str(res.terminated) + ")")
    except Exception:
        pass


def count_steps(res):
    try:
        clock["ticks"] += int(res.steps_executed)
    except Exception:
        pass


def hold_tick(gripper):
    res = step(RobotAction(arms={"primary": ArmCommand(
        "joint_position", home_joints, gripper_position=gripper, embodiment=emb)}))
    clock["ticks"] += 1
    check_end(res)
    return res


def move(position, quat, tol, steps, label):
    if clock["over"]:
        return None
    res = move_to_pose(Pose(tuple(position), tuple(quat), base), tolerance=tol, max_steps=steps)
    count_steps(res)
    check_end(res)
    entry = {"label": label, "ok": bool(res.ok), "steps": 0}
    try:
        entry["steps"] = int(res.steps_executed)
        entry["status"] = str(res.status)
    except Exception:
        pass
    report["events"].append([clock["ticks"], entry])
    return res


def gripper_now():
    try:
        return float(get_robot_state().gripper_positions["primary"])
    except Exception:
        return None


def cloud_at_pixel(px, py):
    """segment_points at a wrist pixel -> base-frame cloud of the top mask (bounded)."""
    seg = segment_points(CAM, [[int(round(px)), int(round(py))]])
    if not seg.ok or len(seg.segmentations) == 0:
        return None
    for cand in seg.segmentations[:3]:
        pc = mask_to_point_cloud(cand, CAM, target_frame=base)
        pts = pc.points
        if len(pts) < 6:
            continue
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        span = max(quantile(xs, 0.95) - quantile(xs, 0.05), quantile(ys, 0.95) - quantile(ys, 0.05))
        if span < 0.12:
            return pts
    return None


# ---------------------------------------------------------------- tracking state
MOVE_PX = 0.3        # per-tick centroid motion that counts as "moving"
END_STILL = 3        # consecutive still ticks that end a shuffle
MARGIN = 2           # ticks of slack kept before a predicted shuffle start
PRESS_TICKS = 10     # joint-target ticks spent pushing the button
S = {"tracks": [], "vel": [], "episodes": [], "lit_active": False, "off_run": 0,
     "baseline_white": set(), "width": 0, "height": 0, "white_px": [],
     "shuffles": [], "moving": False, "still": 0, "cur_start": None, "watch_since": 0,
     "repairs": []}


def observe_tick(res, label):
    rows = res.observation.cameras[CAM].rgb.tolist()
    cube_px, white_px = classify_frame(rows, table_chroma)
    bw = S["baseline_white"]
    w = S["width"]
    new_white = []
    for p in white_px:
        if (p[1] * w + p[0]) not in bw:
            new_white.append(p)
    S["white_px"] = white_px
    old = S["tracks"]
    pred = []
    for i in range(len(old)):
        pred.append((old[i][0] + S["vel"][i][0], old[i][1] + S["vel"][i][1]))
    new, counts = update_tracks(pred, cube_px, gate, min_count)
    speed = 0.0
    vel = []
    for i in range(len(old)):
        dx = new[i][0] - old[i][0]
        dy = new[i][1] - old[i][1]
        vel.append((0.7 * dx, 0.7 * dy))
        speed = max(speed, math.sqrt(dx * dx + dy * dy))
    S["tracks"] = new
    S["vel"] = vel
    fr = ring_fractions(new, new_white, 1.4 * rad, 1.9 * rad)
    t = clock["ticks"]
    if max(fr) > RING_ON:
        S["off_run"] = 0
        if not S["lit_active"]:
            S["lit_active"] = True
            S["episodes"].append({"start": t, "end": t, "frames": 0,
                                  "lit_counts": [0 for k in new], "label": label})
            note("highlight on (" + label + ")")
        ep = S["episodes"][len(S["episodes"]) - 1]
        ep["frames"] += 1
        ep["end"] = t
        for i in range(len(new)):
            if fr[i] > RING_ON:
                ep["lit_counts"][i] += 1
    else:
        S["off_run"] += 1
        if S["off_run"] >= 3:
            S["lit_active"] = False
    if speed > MOVE_PX:
        S["still"] = 0
        if not S["moving"]:
            S["moving"] = True
            S["cur_start"] = t
            note("shuffle start (" + label + ")")
    else:
        S["still"] += 1
        if S["moving"] and S["still"] >= END_STILL:
            S["moving"] = False
            dur = t - END_STILL - S["cur_start"]
            known = 0
            for s in S["shuffles"]:
                if s[2]:
                    known = max(known, s[1] - s[0])
            if dur >= max(5, 0.5 * known):
                clean = (S["cur_start"] - S["watch_since"]) >= 3
                S["shuffles"].append([S["cur_start"], t - END_STILL, clean])
                note("shuffle end (" + label + ", clean=" + str(clean) + ")")
                repair_tracks(cube_px, "shuffle end")
            else:
                note("short motion ignored (" + label + ", " + str(dur) + " ticks)")
    if t % 5 == 0:
        report["track_log"].append([t, label, round(speed, 2), [round(f, 2) for f in fr],
                                    [[round(q[0], 1), round(q[1], 1)] for q in new]])


def watch_until(stop_fn, max_ticks, label):
    start = clock["ticks"]
    while clock["ticks"] - start < max_ticks and clock["ticks"] < hard_limit and not clock["over"]:
        res = hold_tick(1.0)
        if clock["over"]:
            break
        observe_tick(res, label)
        if stop_fn():
            return True
    return False


def repair_tracks(cube_px, label):
    """Tracks with no blob nearby are re-matched (one-to-one) to blobs no track claims."""
    blobs, area = init_tracks(cube_px, S["width"], S["height"])
    tr = S["tracks"]
    pairs = []
    for i in range(len(tr)):
        for j in range(len(blobs)):
            d = math.sqrt((tr[i][0] - blobs[j][0]) ** 2 + (tr[i][1] - blobs[j][1]) ** 2)
            pairs.append((d, i, j))
    pairs.sort()
    t_used = set()
    b_used = set()
    for pr in pairs:
        if pr[0] > gate:
            break
        if pr[1] in t_used or pr[2] in b_used:
            continue
        t_used.add(pr[1])
        b_used.add(pr[2])
    lost = [i for i in range(len(tr)) if i not in t_used]
    free = [j for j in range(len(blobs)) if j not in b_used]
    if len(lost) == 0 or len(free) == 0:
        return
    moves = []
    for pr in pairs:
        if pr[1] in lost and pr[2] in free and pr[1] not in t_used and pr[2] not in b_used:
            t_used.add(pr[1])
            b_used.add(pr[2])
            tr[pr[1]] = blobs[pr[2]]
            S["vel"][pr[1]] = (0.0, 0.0)
            moves.append([pr[1], round(pr[0], 1)])
    S["tracks"] = tr
    S["repairs"].append([clock["ticks"], label, moves, len(lost), len(free)])


def reassociate(label):
    if clock["over"]:
        return
    res = hold_tick(1.0)
    if clock["over"]:
        return
    rows = res.observation.cameras[CAM].rgb.tolist()
    cube_px, white_px = classify_frame(rows, table_chroma)
    old = S["tracks"]
    new, counts = update_tracks(old, cube_px, 3.0 * rad, min_count)
    shift = 0.0
    for i in range(len(old)):
        shift = max(shift, math.sqrt((new[i][0] - old[i][0]) ** 2 + (new[i][1] - old[i][1]) ** 2))
    S["tracks"] = new
    S["vel"] = [(0.0, 0.0) for t in new]
    S["white_px"] = white_px
    S["still"] = 0
    S["watch_since"] = clock["ticks"]
    repair_tracks(cube_px, label)
    report["events"].append([clock["ticks"], {"reassociate": label, "max_shift_px": round(shift, 1)}])


def go_joints(joints, label):
    if clock["over"]:
        return False
    for attempt in range(2):
        r = move_to_joints(joints, tolerance=0.01, max_steps=120)
        count_steps(r)
        check_end(r)
        if clock["over"]:
            return False
        if bool(r.ok):
            report["events"].append([clock["ticks"], {"joints": label, "ok": True}])
            return True
    report["events"].append([clock["ticks"], {"joints": label, "ok": False}])
    return False


def transit(position, quat, label):
    """Fast free-space transit: IK then joint servo; falls back to move_to_pose."""
    if clock["over"]:
        return
    ik = solve_ik(Pose(tuple(position), tuple(quat), base))
    if bool(ik.ok):
        if go_joints([float(v) for v in ik.joint_positions], label):
            return
    move(position, quat, 0.01, 150, label + " (pose fallback)")


def current_targets():
    if len(S["episodes"]) == 0:
        return []
    ep = S["episodes"][len(S["episodes"]) - 1]
    out = []
    for i in range(len(ep["lit_counts"])):
        if ep["lit_counts"][i] >= 0.5 * max(1, ep["frames"]):
            out.append(i)
    return out


def locate_track(i):
    px = S["tracks"][i][0]
    py = S["tracks"][i][1]
    entry = {"track": i, "px": [round(px, 1), round(py, 1)]}
    try:
        pts = cloud_at_pixel(px, py)
        if pts is not None:
            geo = cube_from_cloud(pts)
            if geo is not None:
                entry["geo"] = geo
    except Exception as exc:
        entry["error"] = str(exc)
    if "geo" in entry:
        return entry
    # fallback: segment_text candidates whose box contains the track pixel
    try:
        seg = segment_text(CAM, "cube")
        if seg.ok:
            for cand in seg.segmentations[:5]:
                b = cand.box_xyxy
                if b is None:
                    continue
                if px >= b[0] - 1 and px <= b[2] + 1 and py >= b[1] - 1 and py <= b[3] + 1:
                    pc = mask_to_point_cloud(cand, CAM, target_frame=base)
                    if len(pc.points) >= 6:
                        geo = cube_from_cloud(pc.points)
                        if geo is not None and geo["side"] < 0.12:
                            entry["geo"] = geo
                            entry["source"] = "segment_text box"
                            break
    except Exception as exc:
        entry["error2"] = str(exc)
    return entry


def locate_button():
    found = None
    try:
        best = None
        for c in components(S["white_px"], S["width"]):
            if len(c) >= 0.5 * med_area and not touches_border(c, S["width"], S["height"]):
                if best is None or len(c) > len(best):
                    best = c
        if best is not None:
            bc = centroid(best)
            report["button_px"] = [round(bc[0], 1), round(bc[1], 1), len(best)]
            pts = cloud_at_pixel(bc[0], bc[1])
            if pts is not None:
                found = {"x": quantile([float(p[0]) for p in pts], 0.5),
                         "y": quantile([float(p[1]) for p in pts], 0.5),
                         "z_top": quantile([float(p[2]) for p in pts], 0.95), "n": len(pts)}
    except Exception as exc:
        report["errors"].append("button: " + str(exc))
    if found is None:
        try:
            loc = localize_object("button", camera_name=CAM, target_frame=base)
            if loc.ok and len(loc.point_cloud.points) >= 6:
                pts = loc.point_cloud.points
                found = {"x": quantile([float(p[0]) for p in pts], 0.5),
                         "y": quantile([float(p[1]) for p in pts], 0.5),
                         "z_top": quantile([float(p[2]) for p in pts], 0.95), "n": len(pts),
                         "source": "localize_object"}
        except Exception as exc:
            report["errors"].append("button fallback: " + str(exc))
    return found


def press_button(button, label, return_home):
    if button is None or clock["over"]:
        return
    t0 = clock["ticks"]
    r = close_gripper()
    count_steps(r)
    check_end(r)
    above = (button["x"], button["y"], button["z_top"] + PRESS_ABOVE)
    transit(above, q_down, label + " above")
    ik = solve_ik(Pose((button["x"], button["y"], button["z_top"] - PRESS_DEPTH), q_down, base))
    if bool(ik.ok):
        target = [float(v) for v in ik.joint_positions]
        for k in range(PRESS_TICKS):
            if clock["over"]:
                break
            rs = step(RobotAction(arms={"primary": ArmCommand(
                "joint_position", target, gripper_position=0.0, embodiment=emb)}))
            clock["ticks"] += 1
            check_end(rs)
    else:
        move((button["x"], button["y"], button["z_top"] - 0.005), q_down, 0.004, 25, label + " press")
    report["events"].append([clock["ticks"], {"press": label, "ik_ok": bool(ik.ok)}])
    if return_home:
        go_joints(home_joints, label + " home")
    else:
        move(above, q_down, 0.015, 60, label + " retreat")
    report["presses"].append([label, t0, clock["ticks"]])
    return clock["ticks"] - t0


def pick_cube(entry, return_home):
    t0 = clock["ticks"]
    pick = {"track": entry["track"], "px": entry["px"], "t0": t0}
    report["picks"].append(pick)
    if "geo" not in entry:
        pick["skipped"] = "no geometry"
        return 0
    geo = entry["geo"]
    pick["geo_home"] = geo
    r = open_gripper()
    count_steps(r)
    check_end(r)
    q = yaw_quat(q_down, geo["yaw"])
    transit((geo["x"], geo["y"], geo["z_top"] + PRE_Z), q, "pre")
    if clock["over"]:
        return clock["ticks"] - t0
    grasp_z = geo["z_top"] - 0.5 * geo["side"]
    move((geo["x"], geo["y"], grasp_z), q, 0.008, 100, "descend")
    if clock["over"]:
        return clock["ticks"] - t0
    r = close_gripper()
    count_steps(r)
    check_end(r)
    pick["gripper_closed"] = gripper_now()
    move((geo["x"], geo["y"], geo["z_top"] + LIFT_Z), q, 0.015, 120, "lift")
    if clock["over"]:
        return clock["ticks"] - t0
    pick["gripper_lifted"] = gripper_now()
    r = open_gripper()          # release at lift height: the cube drops back onto its spot
    count_steps(r)
    check_end(r)
    if return_home:
        go_joints(home_joints, "pick home")
    pick["ticks"] = clock["ticks"] - t0
    return clock["ticks"] - t0


def shuffle_model():
    """REST (end -> next start) and duration, learned from shuffles whose start was watched."""
    sh = S["shuffles"]
    if len(sh) < 2:
        return None
    rests = []
    for k in range(1, len(sh)):
        if sh[k][2]:
            rests.append(sh[k][0] - sh[k - 1][1])
    durs = [s[1] - s[0] for s in sh if s[2]]
    if len(rests) == 0 or len(durs) == 0:
        return None
    return {"rest": min(rests), "duration": max(durs), "last_end": sh[len(sh) - 1][1],
            "period": min(rests) + max(durs)}


def next_shuffle_start(model, t):
    if model is None:
        return None
    n = model["last_end"] + model["rest"]
    while n <= t:
        n += model["period"]
    return n


def wait_next_shuffle(model, label):
    """Hold at home until a shuffle has started and ended (or a generous timeout)."""
    count0 = len(S["shuffles"])
    limit = 0.25 * max_steps
    if model is not None:
        limit = next_shuffle_start(model, clock["ticks"]) - clock["ticks"] + model["duration"] + 30

    def done():
        return len(S["shuffles"]) > count0
    return watch_until(done, int(limit), label)


# ---------------------------------------------------------------- main
hard_limit = max_steps - 5
try:
    obs = get_observation()
    rows = obs.cameras[CAM].rgb.tolist()
    S["height"] = len(rows)
    S["width"] = len(rows[0])
    table_chroma = median_chroma(rows, 4)
    cube_px, white_px = classify_frame(rows, table_chroma)
    for p in white_px:
        S["baseline_white"].add(p[1] * S["width"] + p[0])
    S["white_px"] = white_px
    tracks, med_area = init_tracks(cube_px, S["width"], S["height"])
    report["n_tracks"] = len(tracks)
    report["median_cube_area_px"] = med_area
    if len(tracks) == 0:
        raise ValueError("no cubes found in the wrist view")
    S["tracks"] = tracks
    S["vel"] = [(0.0, 0.0) for t in tracks]
    rad = math.sqrt(med_area / math.pi)
    gate = 2.2 * rad
    min_count = max(3, int(0.3 * med_area))

    # 1) watch: highlight, then two shuffles (learn period and duration at runtime)
    def two_shuffles():
        return len(S["shuffles"]) >= 2 and not S["moving"]
    watch_until(two_shuffles, int(0.3 * max_steps), "learn")
    model = shuffle_model()
    report["shuffle_model"] = model
    targets = current_targets()
    report["targets"] = targets
    button = locate_button()
    report["button"] = button

    # 2) schedule: press1, picks, press2 — one action per static gap
    actions = [["press", None]]
    for i in targets:
        actions.append(["pick", i])
    est = {"press": None, "pick": None}
    report["schedule"] = []
    k = 0
    guard = 0
    while k < len(actions) and not clock["over"] and clock["ticks"] < hard_limit and guard < 40:
        guard += 1
        kind = actions[k][0]
        last = (k == len(actions) - 1)
        t = clock["ticks"]
        nxt = next_shuffle_start(model, t)
        avail = None
        if nxt is not None:
            avail = nxt - t - MARGIN
        need = est[kind]
        fits = True
        if avail is not None:
            gap_start = (t - model["last_end"]) <= END_STILL + 3
            fits = gap_start or (need is not None and need <= avail)
        report["schedule"].append([t, kind, actions[k][1], avail, need, fits])
        if not fits:
            wait_next_shuffle(model, "wait")
            model2 = shuffle_model()
            if model2 is not None:
                model = model2
            continue
        if kind == "press":
            used = press_button(button, "press1", True)
        else:
            entry = locate_track(actions[k][1])
            used = pick_cube(entry, not last)
        if used is not None and not last and (est[kind] is None or used > est[kind]):
            est[kind] = used
        k += 1
        if last or clock["over"]:
            break
        reassociate("after " + kind)
        wait_next_shuffle(model, "after " + kind)
        model2 = shuffle_model()
        if model2 is not None:
            model = model2
    report["estimates"] = est

    # 3) final press (no tracking needed afterwards)
    press_button(button, "press2", False)
    for t in range(10):
        if clock["over"]:
            break
        st = get_robot_state()
        rs = step(RobotAction(arms={"primary": ArmCommand(
            "joint_position", st.joint_positions["primary"], gripper_position=0.0, embodiment=emb)}))
        clock["ticks"] += 1
        check_end(rs)
except Exception as exc:
    report["errors"].append("main: " + str(exc))
report["episodes"] = S["episodes"]
report["shuffles"] = S["shuffles"]
report["repairs"] = S["repairs"]
report["final_tracks"] = [[round(t[0], 1), round(t[1], 1)] for t in S["tracks"]]
report["ticks"] = clock["ticks"]
result = report
