# ButtonUnmask v00 -- initial program from demonstration + scene analysis.
# Strategy:
#   1. Read the goal, localize the button (front camera), self-calibrate the camera
#      deprojection convention against mask_to_point_cloud.
#   2. Press the button with a stepped joint trajectory (deadline: before step 90) while
#      watching both cameras every tick; coloured cubes appear in batches and are then
#      covered by white containers. Every coloured blob is deprojected and tracked in 3D.
#   3. Return home, keep watching until no cube has been visible for a while.
#   4. For each goal colour (in goal order) locate the container over the remembered cube
#      (segment_points at the projected memory point), grasp it with yaw alignment, lift it,
#      then lower it back and release.
import math

report = {"version": "v00", "events": [], "errors": []}
COLORS = ("red", "green", "blue", "yellow", "orange", "purple")
TICK = {"n": 0, "terminated": False, "truncated": False, "last_obs": None}
TRACKS = []
VIS = {"last_visible": -1, "visible_ticks": 0}
CAL = {}


def note(msg):
    report["events"].append(str(TICK["n"]) + ": " + str(msg))


def as_list(arr):
    try:
        return arr.tolist()
    except Exception:
        return [list(row) for row in arr]


def quat_to_mat(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def cam_local(u, v, d, K, conv):
    fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
    a = (u - cx) * d / fx
    b = (v - cy) * d / fy
    if conv == "cv":
        return [a, b, d]
    if conv == "gl":
        return [a, -b, -d]
    return [d, -a, -b]  # "ros": x forward, y left, z up


def deproject(u, v, d, K, R, t, conv):
    p = cam_local(u, v, d, K, conv)
    return [R[i][0] * p[0] + R[i][1] * p[1] + R[i][2] * p[2] + t[i] for i in range(3)]


def project(pw, K, R, t, conv):
    q = [pw[i] - t[i] for i in range(3)]
    pc = [R[0][j] * q[0] + R[1][j] * q[1] + R[2][j] * q[2] for j in range(3)]
    if conv == "cv":
        a, b, d = pc[0], pc[1], pc[2]
    elif conv == "gl":
        a, b, d = pc[0], -pc[1], -pc[2]
    else:
        d, a, b = pc[0], -pc[1], -pc[2]
    if d <= 1e-4:
        return None
    fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
    return [a * fx / d + cx, b * fy / d + cy]


def cam_geom(cam):
    pose = cam.camera_pose
    R = quat_to_mat(pose.quaternion_wxyz)
    t = [float(value) for value in pose.position]
    K = as_list(cam.intrinsics)
    return K, R, t


def quantile(values, frac):
    s = sorted(values)
    if not s:
        return None
    return s[int((len(s) - 1) * frac)]


def classify(r, g, b):
    if r > 110 and g < 0.2 * r and b < 0.2 * r:
        return "red"
    if g > 100 and r < 0.45 * g and b < 0.45 * g:
        return "green"
    if b > 100 and r < 0.3 * b and g < 0.3 * b:
        return "blue"
    if r > 150 and g > 150 and b < 0.3 * g and abs(r - g) < 50:
        return "yellow"
    if r > 180 and g > 0.6 * r and g < 0.8 * r and b < 0.15 * r:
        return "orange"
    if r > 90 and b > 90 and g < 0.6 * min(r, b) and b > 0.7 * r:
        return "purple"
    return None


def detect(cam, cam_name, stride):
    """Return coloured blobs [(colour, xyz, npix, u, v)] in one camera image."""
    rgb = as_list(cam.rgb)
    H = len(rgb)
    W = len(rgb[0])
    hits = {}
    for v in range(0, H, stride):
        row = rgb[v]
        for u in range(0, W, stride):
            px = row[u]
            r, g, b = px[0], px[1], px[2]
            if max(r, g, b) - min(r, g, b) < 70:
                continue
            c = classify(r, g, b)
            if c is not None:
                hits[(u, v)] = c
    if not hits:
        return []
    depth = as_list(cam.depth_m)
    K, R, t = cam_geom(cam)
    conv = CAL.get(cam_name, CAL.get("front", "cv"))
    blobs = []
    seen = set()
    min_px = 4 if cam_name == "front" else 6
    for key in hits:
        if key in seen:
            continue
        color = hits[key]
        comp = []
        stack = [key]
        seen.add(key)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for du in (-2 * stride, -stride, 0, stride, 2 * stride):
                for dv in (-2 * stride, -stride, 0, stride, 2 * stride):
                    nb = (cur[0] + du, cur[1] + dv)
                    if nb in hits and nb not in seen and hits[nb] == color:
                        seen.add(nb)
                        stack.append(nb)
        if len(comp) < min_px:
            continue
        xs, ys, zs = [], [], []
        for (u, v) in comp:
            d = float(depth[v][u])
            if d <= 0.01 or d > 5.0:
                continue
            p = deproject(u, v, d, K, R, t, conv)
            xs.append(p[0])
            ys.append(p[1])
            zs.append(p[2])
        if len(xs) < 3:
            continue
        us = [c[0] for c in comp]
        vs = [c[1] for c in comp]
        blobs.append((color, [quantile(xs, 0.5), quantile(ys, 0.5), quantile(zs, 0.5)],
                      len(comp), sum(us) / len(us), sum(vs) / len(vs)))
    return blobs


def update_tracks(blobs, cam_name):
    for (color, xyz, npix, u, v) in blobs:
        best = None
        best_d = None
        for tr in TRACKS:
            if tr["color"] != color:
                continue
            d = math.hypot(tr["xyz"][0] - xyz[0], tr["xyz"][1] - xyz[1])
            if d < 0.035 and (best_d is None or d < best_d):
                best, best_d = tr, d
        if best is None:
            TRACKS.append({"color": color, "xyz": list(xyz), "n": 1, "first": TICK["n"],
                           "last": TICK["n"], "cams": {cam_name: 1}, "npix": npix})
        else:
            n = best["n"]
            w = 1.0 / (n + 1)
            best["xyz"] = [best["xyz"][i] * (1 - w) + xyz[i] * w for i in range(3)]
            best["n"] = n + 1
            best["last"] = TICK["n"]
            best["cams"][cam_name] = best["cams"].get(cam_name, 0) + 1
            best["npix"] = max(best["npix"], npix)


def observe(obs, watch):
    TICK["last_obs"] = obs
    if not watch or obs is None:
        return
    any_blob = False
    for cam_name in ("front", "wrist"):
        try:
            cam = obs.cameras[cam_name]
        except Exception:
            continue
        try:
            blobs = detect(cam, cam_name, 2)
        except Exception as exc:
            if len(report["errors"]) < 20:
                report["errors"].append("detect " + cam_name + ": " + str(exc))
            continue
        if blobs:
            any_blob = True
            update_tracks(blobs, cam_name)
    if any_blob:
        VIS["last_visible"] = TICK["n"]
        VIS["visible_ticks"] += 1


def ended():
    return TICK["terminated"] or TICK["truncated"]


def tick(joints, gripper, watch):
    if ended():
        return None
    reply = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=[float(j) for j in joints],
        gripper_position=float(gripper), embodiment=EMB)}))
    TICK["n"] += 1
    try:
        if bool(reply.terminated):
            TICK["terminated"] = True
        if bool(reply.truncated):
            TICK["truncated"] = True
    except Exception:
        pass
    if ended():
        note("episode ended terminated=" + str(TICK["terminated"]) + " truncated=" + str(TICK["truncated"]))
        return reply
    obs = None
    try:
        obs = reply.observation
    except Exception:
        obs = None
    if obs is None:
        obs = get_observation()
    observe(obs, watch)
    return reply


def current_joints():
    return [float(j) for j in get_robot_state().joint_positions["primary"]]


def drive(target, gripper, watch, max_delta=0.05, tol=0.012, settle=15):
    """Stepped linear joint interpolation then hold until converged."""
    start = current_joints()
    span = max(abs(target[i] - start[i]) for i in range(7))
    n = max(1, int(math.ceil(span / max_delta)))
    for i in range(1, n + 1):
        if ended():
            return False
        f = float(i) / n
        tick([start[j] + (target[j] - start[j]) * f for j in range(7)], gripper, watch)
    for k in range(settle):
        if ended():
            return False
        cur = current_joints()
        err = max(abs(target[i] - cur[i]) for i in range(7))
        if err < tol:
            return True
        tick(target, gripper, watch)
    return False


def ik(position, quat, seed):
    res = solve_ik(Pose([float(p) for p in position], [float(q) for q in quat], BASE), seed_joints=seed)
    if not res.ok:
        return None
    return [float(j) for j in res.joint_positions]


def parse_goal(text):
    """Return ordered [(colour, all_flag)] from the goal sentence."""
    low = text.lower()
    idx = low.find("button")
    tail = low[idx + 6:] if idx >= 0 else low
    clauses = []
    for part in tail.replace("followed by", ",").replace("and finally", ",").replace(" then ", ",").split(","):
        clauses.append(part)
    out = []
    for cl in clauses:
        best = None
        for c in COLORS:
            p = cl.find(c)
            if p >= 0 and (best is None or p < best[0]):
                best = (p, c)
        if best is not None:
            out.append((best[1], ("all " in cl) or ("containers" in cl) or ("cubes" in cl)))
    return out


def points_of(pc):
    try:
        return as_list(pc.points)
    except Exception:
        return [[float(p[0]), float(p[1]), float(p[2])] for p in pc.points]


def calibrate(cam_name, seg_cand, cam):
    """Pick the deprojection convention that matches mask_to_point_cloud."""
    pc = mask_to_point_cloud(seg_cand, cam_name, target_frame=BASE)
    pts = points_of(pc)
    if len(pts) < 10:
        return None
    med = [quantile([p[i] for p in pts], 0.5) for i in range(3)]
    box = [float(value) for value in seg_cand.box_xyxy]
    u = int((box[0] + box[2]) / 2.0)
    v = int((box[1] + box[3]) / 2.0)
    depth = as_list(cam.depth_m)
    d = float(depth[v][u])
    K, R, t = cam_geom(cam)
    best = None
    scores = {}
    for conv in ("cv", "gl", "ros"):
        p = deproject(u, v, d, K, R, t, conv)
        e = math.sqrt(sum((p[i] - med[i]) ** 2 for i in range(3)))
        scores[conv] = round(e, 4)
        if best is None or e < best[1]:
            best = (conv, e)
    report["calibration_" + cam_name] = {"scores": scores, "cloud_median": med, "pixel": [u, v]}
    if best[1] < 0.05:
        return best[0]
    return None


def min_area_yaw(pts):
    best = None
    for k in range(45):
        a = math.radians(2.0 * k)
        ca, sa = math.cos(a), math.sin(a)
        xs = [p[0] * ca + p[1] * sa for p in pts]
        ys = [-p[0] * sa + p[1] * ca for p in pts]
        ex = quantile(xs, 0.97) - quantile(xs, 0.03)
        ey = quantile(ys, 0.97) - quantile(ys, 0.03)
        area = ex * ey
        if best is None or area < best[0]:
            best = (area, a, ex, ey)
    return best[1], best[2], best[3]


def locate_container(track):
    """segment_points at the projected memory point; accept a cloud centred near it."""
    attempts = []
    for cam_name in ("front", "wrist"):
        try:
            obs = get_observation()
            cam = obs.cameras[cam_name]
            K, R, t = cam_geom(cam)
            conv = CAL.get(cam_name, CAL.get("front", "cv"))
            target = [track["xyz"][0], track["xyz"][1], track["xyz"][2] + 0.02]
            uv = project(target, K, R, t, conv)
            if uv is None or uv[0] < 2 or uv[1] < 2 or uv[0] > 253 or uv[1] > 253:
                attempts.append({"cam": cam_name, "uv": uv, "why": "off-image"})
                continue
            seg = segment_points(cam_name, [[int(uv[0]), int(uv[1])]])
            if not seg.ok or len(seg.segmentations) == 0:
                attempts.append({"cam": cam_name, "uv": uv, "why": "no-seg"})
                continue
            pc = mask_to_point_cloud(seg.segmentations[0], cam_name, target_frame=BASE)
            pts = points_of(pc)
            if len(pts) < 15:
                attempts.append({"cam": cam_name, "uv": uv, "why": "few-points", "n": len(pts)})
                continue
            top = quantile([p[2] for p in pts], 0.95)
            upper = [p for p in pts if p[2] > top - 0.03]
            cx = quantile([p[0] for p in upper], 0.5)
            cy = quantile([p[1] for p in upper], 0.5)
            near = [p for p in upper if math.hypot(p[0] - cx, p[1] - cy) < 0.07]
            yaw, ex, ey = min_area_yaw(near)
            dist = math.hypot(cx - track["xyz"][0], cy - track["xyz"][1])
            info = {"cam": cam_name, "uv": [round(uv[0], 1), round(uv[1], 1)], "n": len(pts),
                    "center": [round(cx, 4), round(cy, 4)], "top": round(top, 4),
                    "yaw_deg": round(math.degrees(yaw), 1), "ext": [round(ex, 4), round(ey, 4)],
                    "dist": round(dist, 4), "zmin": round(quantile([p[2] for p in pts], 0.03), 4)}
            attempts.append(info)
            if dist < 0.045 and max(ex, ey) < 0.11 and top > track["xyz"][2] + 0.005:
                info["accepted"] = True
                return info, attempts
        except Exception as exc:
            attempts.append({"cam": cam_name, "error": str(exc)})
    return None, attempts


def pick_container(info, track, home_quat):
    """Grasp the container from above, lift it, put it back. Returns lifted flag."""
    rec = {"color": track["color"], "track": [round(v, 4) for v in track["xyz"]]}
    cx, cy, top = info["center"][0], info["center"][1], info["top"]
    yaw = math.radians(info["yaw_deg"])
    Rh = quat_to_mat(home_quat)
    phi = math.atan2(Rh[1][1], Rh[0][1])  # closing axis (hand y) in world
    d = (yaw - phi) % (math.pi / 2.0)
    if d > math.pi / 4.0:
        d -= math.pi / 2.0
    rz = [math.cos(d / 2.0), 0.0, 0.0, math.sin(d / 2.0)]
    quat = quat_mul(rz, home_quat)
    rec["dyaw_deg"] = round(math.degrees(d), 1)
    width = min(info["ext"][0], info["ext"][1])
    open_frac = min(1.0, max(0.6, (width + 0.03) / 0.08))
    rec["open_frac"] = round(open_frac, 3)
    hover = [cx, cy, top + 0.10]
    grasp = [cx, cy, top - 0.022]
    lift = [cx, cy, top + 0.13]
    seed = current_joints()
    jh = ik(hover, quat, seed)
    if jh is None:
        rec["fail"] = "ik hover"
        return False, rec
    jg = ik(grasp, quat, jh)
    jl = ik(lift, quat, jh)
    if jg is None or jl is None:
        rec["fail"] = "ik grasp/lift"
        return False, rec
    drive(jh, open_frac, False)
    drive(jg, open_frac, False, max_delta=0.03)
    if ended():
        return False, rec
    st = get_robot_state()
    rec["ee_at_grasp"] = [round(float(v), 4) for v in st.end_effector_poses["primary"].position]
    cg = close_gripper()
    try:
        if bool(cg.terminated) or bool(cg.truncated):
            TICK["terminated"] = True
            note("ended at close")
            return False, rec
    except Exception:
        pass
    try:
        TICK["n"] += int(cg.steps_executed)
    except Exception:
        pass
    g = float(get_robot_state().gripper_positions["primary"])
    rec["grip_after_close"] = round(g, 4)
    drive(jl, 0.0, False, max_delta=0.03)
    if ended():
        rec["ended_on_lift"] = True
        return True, rec
    st = get_robot_state()
    rec["ee_lifted"] = [round(float(v), 4) for v in st.end_effector_poses["primary"].position]
    rec["grip_lifted"] = round(float(st.gripper_positions["primary"]), 4)
    for k in range(6):
        tick(jl, 0.0, False)
        if ended():
            rec["ended_on_hold"] = True
            return True, rec
    # put it back where it was
    jp = ik([cx, cy, top - 0.018], quat, jl)
    if jp is None:
        jp = jg
    drive(jp, 0.0, False, max_delta=0.03)
    if ended():
        return True, rec
    for k in range(12):
        tick(jp, 1.0, False)
        if ended():
            return True, rec
    drive(jh, 1.0, False)
    return True, rec


def track_rank(i):
    return -TRACKS[i]["n"]


def main():
    ctx = get_task_context()
    goal = ctx.language
    report["goal"] = goal
    try:
        meta = robomme_plus.get_task_metadata()
        report["metadata"] = dict(meta)
    except Exception as exc:
        report["errors"].append("metadata: " + str(exc))
    try:
        cmeta = robomme_plus.get_controller_metadata()
        report["controller_keys"] = [str(k) for k in cmeta]
    except Exception as exc:
        report["errors"].append("controller metadata: " + str(exc))
    targets = parse_goal(goal)
    report["targets"] = [[c, a] for (c, a) in targets]
    # conditioning video (may be empty)
    try:
        video = robomme_plus.get_conditioning_video()
        report["video_len"] = len(video)
    except Exception as exc:
        video = []
        report["errors"].append("video: " + str(exc))
    state = get_robot_state()
    home = [float(j) for j in state.joint_positions["primary"]]
    home_pose = state.end_effector_poses["primary"]
    home_quat = [float(q) for q in home_pose.quaternion_wxyz]
    report["home_ee"] = [round(float(v), 4) for v in home_pose.position]
    report["home_quat"] = [round(q, 4) for q in home_quat]
    report["home_joints"] = [round(j, 4) for j in home]
    report["home_grip"] = float(state.gripper_positions["primary"])
    obs0 = get_observation()
    TICK["last_obs"] = obs0
    # button localization + calibration
    seg = segment_text("front", "button")
    button = None
    cands = []
    if seg.ok:
        for cand in seg.segmentations[:5]:
            try:
                pc = mask_to_point_cloud(cand, "front", target_frame=BASE)
                pts = points_of(pc)
                if len(pts) < 10:
                    continue
                zs = [p[2] for p in pts]
                top = quantile(zs, 0.95)
                low = quantile(zs, 0.05)
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ext = max(quantile(xs, 0.95) - quantile(xs, 0.05), quantile(ys, 0.95) - quantile(ys, 0.05))
                capp = [p for p in pts if p[2] > top - 0.008]
                c = {"score": round(float(cand.score), 3), "n": len(pts), "top": round(top, 4),
                     "low": round(low, 4), "ext": round(ext, 4),
                     "center": [quantile([p[0] for p in capp], 0.5), quantile([p[1] for p in capp], 0.5)]}
                cands.append(c)
                if button is None and ext < 0.12 and top - low < 0.08:
                    button = c
                    button["cand"] = cand
            except Exception as exc:
                report["errors"].append("button cand: " + str(exc))
    report["button_candidates"] = [{k: c[k] for k in c if k != "cand"} for c in cands]
    if button is None:
        raise ValueError("button not found")
    conv = calibrate("front", button["cand"], obs0.cameras["front"])
    CAL["front"] = conv if conv is not None else "cv"
    try:
        wseg = segment_text("wrist", "button")
        if wseg.ok and len(wseg.segmentations) > 0:
            wc = calibrate("wrist", wseg.segmentations[0], obs0.cameras["wrist"])
            if wc is not None:
                CAL["wrist"] = wc
    except Exception as exc:
        report["errors"].append("wrist calib: " + str(exc))
    report["conventions"] = dict(CAL)
    # scan conditioning video, if any
    if len(video) > 0:
        for i in range(0, len(video), 2):
            observe(video[i], True)
        report["tracks_from_video"] = len(TRACKS)
    # first look at the live scene
    observe(obs0, True)
    # --- press the button ---
    bx, by = button["center"][0], button["center"][1]
    btop = button["top"]
    pre = ik([bx, by, btop + 0.06], home_quat, home)
    press = ik([bx, by, btop - 0.015], home_quat, pre if pre is not None else home)
    if pre is None or press is None:
        raise ValueError("button IK failed")
    drive(pre, 0.0, True, max_delta=0.06, settle=6)
    report["tick_pre_press"] = TICK["n"]
    drive(press, 0.0, True, max_delta=0.04, settle=8)
    st = get_robot_state()
    report["ee_at_press"] = [round(float(v), 4) for v in st.end_effector_poses["primary"].position]
    for k in range(4):
        tick(press, 0.0, True)
    report["tick_pressed"] = TICK["n"]
    drive(pre, 0.0, True, max_delta=0.05, settle=4)
    drive(home, 0.0, True, max_delta=0.06, settle=8)
    report["tick_home"] = TICK["n"]
    # --- keep watching until the reveal sequence is over ---
    while not ended() and TICK["n"] < 320:
        quiet = TICK["n"] - max(VIS["last_visible"], 0)
        if len(TRACKS) > 0 and quiet >= 35 and TICK["n"] >= 100:
            break
        tick(home, 0.0, True)
    report["tick_reveal_done"] = TICK["n"]
    report["tracks"] = [{"color": t["color"], "xyz": [round(v, 4) for v in t["xyz"]], "n": t["n"],
                         "first": t["first"], "last": t["last"], "cams": t["cams"], "npix": t["npix"]}
                        for t in TRACKS]
    if ended():
        return
    # --- choose containers in goal order ---
    plan = []
    used = set()
    for (color, all_flag) in targets:
        cand = [i for i in range(len(TRACKS)) if TRACKS[i]["color"] == color and i not in used]
        cand = sorted(cand, key=track_rank)
        if not cand:
            note("no track for " + color)
            continue
        chosen = cand if all_flag else cand[:1]
        for i in chosen:
            used.add(i)
            plan.append(i)
    report["plan"] = [TRACKS[i]["color"] for i in plan]
    located = []
    report["locate"] = []
    for i in plan:
        info, attempts = locate_container(TRACKS[i])
        report["locate"].append({"color": TRACKS[i]["color"], "attempts": attempts})
        located.append(info)
    report["picks"] = []
    tick(home, 1.0, False)
    for k in range(len(plan)):
        if ended():
            break
        info = located[k]
        tr = TRACKS[plan[k]]
        if info is None:
            info2, attempts = locate_container(tr)
            report["locate"].append({"color": tr["color"], "retry": True, "attempts": attempts})
            if info2 is None:
                note("container not located for " + tr["color"] + "; using memory point")
                info = {"center": [tr["xyz"][0], tr["xyz"][1]], "top": tr["xyz"][2] + 0.04,
                        "yaw_deg": 0.0, "ext": [0.05, 0.05]}
            else:
                info = info2
        ok, rec = pick_container(info, tr, home_quat)
        rec["tick_end"] = TICK["n"]
        report["picks"].append(rec)


BASE = get_robot_state().base_frame
EMB = get_robot_state().embodiment
try:
    main()
except Exception as exc:
    report["errors"].append("main: " + str(exc))
report["ticks"] = TICK["n"]
report["terminated"] = TICK["terminated"]
report["truncated"] = TICK["truncated"]
if "tracks" not in report:
    report["tracks"] = [{"color": t["color"], "xyz": [round(v, 4) for v in t["xyz"]], "n": t["n"],
                         "first": t["first"], "last": t["last"]} for t in TRACKS]
result = report
