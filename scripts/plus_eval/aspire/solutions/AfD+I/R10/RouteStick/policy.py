# RouteStick v00 -- reproduce the conditioning video's stick route in the live scene.
#
# Strategy (all episode-dependent quantities derived at runtime):
#   1. Read the conditioning video (robomme_plus.get_conditioning_video()).  Each frame carries
#      the recorded robot state, so the demonstrated end-effector path is available in robot_base.
#   2. Detect the white target discs in the video's first front frame and in the live front frame
#      (white / low-saturation pixels lying on the table plane, projected with the frame's own
#      depth + intrinsics + camera pose).  Order them along the row.
#   3. Fit a 2-D rigid transform (yaw + translation) video-targets -> live-targets and apply it to
#      the demonstrated end-effector poses (z unchanged).
#   4. If the transform is ~identity, replay the recorded joints; otherwise solve IK for the
#      mapped poses (seeded by the previous solution) and stream joint targets tick by tick.
import math

CAM = "front"
WHITE_MIN = 150          # min RGB channel for a target-disc pixel
WHITE_SPREAD = 40        # max channel spread (low saturation)
DISC_Z_BAND = 0.015      # disc pixels lie within this height above the table plane
MIN_BLOB = 8             # min pixels in a disc blob
ROBOT_FOOTPRINT = 0.20   # ignore white pixels within this xy radius of the robot base (own body)
IDENTITY_POS = 0.004     # m: treat transform as identity below this translation...
IDENTITY_YAW = 0.006     # rad: ...and this yaw
MAX_IK = 300
TICK = 0.05

diag = {"version": "v00"}
result = diag


def to_list(a):
    try:
        return a.tolist()
    except Exception:
        return [list(r) for r in a]


def quat_to_mat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def cam_ray(conv, xn, yn):
    if conv == "cv":
        return [xn, yn, 1.0]
    if conv == "gl":
        return [xn, -yn, -1.0]
    return [1.0, -xn, -yn]  # "ros": x forward, y left, z up


def make_projector(cam, conv, ray_depth):
    K = to_list(cam.intrinsics)
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    pose = cam.camera_pose
    R = quat_to_mat(list(pose.quaternion_wxyz))
    P = list(pose.position)

    def proj(u, v, d):
        xn = (u - cx) / fx
        yn = (v - cy) / fy
        r = cam_ray(conv, xn, yn)
        if ray_depth:
            n = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
            s = d / n
        else:
            s = d  # depth along the optical axis (axis component of r is +-1)
        c = [r[0] * s, r[1] * s, r[2] * s]
        return [P[i] + R[i][0] * c[0] + R[i][1] * c[1] + R[i][2] * c[2] for i in range(3)]
    return proj


def choose_projection(cam, depth):
    """Pick the camera convention / depth type under which the table is a flat plane below the camera."""
    H = len(depth)
    W = len(depth[0])
    camz = list(cam.camera_pose.position)[2]
    best = None
    report = []
    for conv in ("cv", "gl", "ros"):
        for ray_depth in (False, True):
            proj = make_projector(cam, conv, ray_depth)
            zs = []
            for v in range(H // 2, H, 8):
                for u in range(0, W, 8):
                    d = depth[v][u]
                    if d > 0.05 and d < 5.0:
                        zs.append(proj(u, v, d)[2])
            if len(zs) < 10:
                continue
            zs.sort()
            med = zs[len(zs) // 2]
            iqr = zs[(3 * len(zs)) // 4] - zs[len(zs) // 4]
            below = med < camz
            report.append([conv, ray_depth, round(med, 4), round(iqr, 4), below])
            score = iqr + (0.0 if below else 10.0)
            if best is None or score < best[0]:
                best = [score, conv, ray_depth, med]
    return best, report


def detect_targets(obs, label):
    cam = obs.cameras[CAM]
    rgb = to_list(cam.rgb)
    depth = to_list(cam.depth_m)
    best, rep = choose_projection(cam, depth)
    info = {"projection_candidates": rep}
    if best is None:
        info["error"] = "no projection"
        diag[label] = info
        return None
    conv, ray_depth, table_z = best[1], best[2], best[3]
    info["conv"] = conv
    info["ray_depth"] = ray_depth
    info["table_z"] = round(table_z, 4)
    proj = make_projector(cam, conv, ray_depth)
    H = len(rgb)
    W = len(rgb[0])
    on = {}
    for v in range(H):
        row = rgb[v]
        drow = depth[v]
        for u in range(W):
            p = row[u]
            r, g, b = p[0], p[1], p[2]
            lo = min(r, g, b)
            if lo < WHITE_MIN or max(r, g, b) - lo > WHITE_SPREAD:
                continue
            d = drow[u]
            if not (d > 0.05 and d < 5.0):
                continue
            xyz = proj(u, v, d)
            if xyz[2] > table_z + DISC_Z_BAND or xyz[2] < table_z - 0.02:
                continue
            if math.sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1]) < ROBOT_FOOTPRINT:
                continue
            on[(u, v)] = xyz
    seen = set()
    blobs = []
    for key in sorted(on.keys()):
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        comp = []
        while stack:
            q = stack.pop()
            comp.append(q)
            for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                n = (q[0] + du, q[1] + dv)
                if n in on and n not in seen:
                    seen.add(n)
                    stack.append(n)
        if len(comp) >= MIN_BLOB:
            pts = [on[c] for c in comp]
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            cz = sum(p[2] for p in pts) / len(pts)
            pu = sum(c[0] for c in comp) / len(comp)
            pv = sum(c[1] for c in comp) / len(comp)
            blobs.append({"xyz": [cx, cy, cz], "px": [round(pu, 1), round(pv, 1)], "n": len(comp)})
    # merge blobs closer than 2.5 cm (rings of one disc split by grey pattern)
    merged = []
    def blob_size_key(b):
        return -b["n"]
    for b in sorted(blobs, key=blob_size_key):
        hit = None
        for m in merged:
            dx = m["xyz"][0] - b["xyz"][0]
            dy = m["xyz"][1] - b["xyz"][1]
            if math.sqrt(dx * dx + dy * dy) < 0.025:
                hit = m
                break
        if hit is None:
            merged.append(b)
        else:
            tot = hit["n"] + b["n"]
            hit["xyz"] = [(hit["xyz"][i] * hit["n"] + b["xyz"][i] * b["n"]) / tot for i in range(3)]
            hit["n"] = tot
    info["blobs"] = [{"xyz": [round(c, 4) for c in m["xyz"]], "px": m["px"], "n": m["n"]} for m in merged]
    diag[label] = info
    return merged


def order_row(targets):
    n = len(targets)
    mx = sum(t["xyz"][0] for t in targets) / n
    my = sum(t["xyz"][1] for t in targets) / n
    sxx = sum((t["xyz"][0] - mx) ** 2 for t in targets)
    syy = sum((t["xyz"][1] - my) ** 2 for t in targets)
    sxy = sum((t["xyz"][0] - mx) * (t["xyz"][1] - my) for t in targets)
    ang = 0.5 * math.atan2(2 * sxy, sxx - syy)
    dx, dy = math.cos(ang), math.sin(ang)
    if dy < 0 or (abs(dy) < 1e-6 and dx < 0):
        dx, dy = -dx, -dy
    def along_key(t):
        return (t["xyz"][0] - mx) * dx + (t["xyz"][1] - my) * dy
    return sorted(targets, key=along_key)


def fit_rigid(src, dst):
    n = len(src)
    sx = sum(p[0] for p in src) / n
    sy = sum(p[1] for p in src) / n
    dx = sum(p[0] for p in dst) / n
    dy = sum(p[1] for p in dst) / n
    a = 0.0
    b = 0.0
    for p, q in zip(src, dst):
        px, py = p[0] - sx, p[1] - sy
        qx, qy = q[0] - dx, q[1] - dy
        a += px * qx + py * qy
        b += px * qy - py * qx
    th = math.atan2(b, a)
    c, s = math.cos(th), math.sin(th)
    tx = dx - (c * sx - s * sy)
    ty = dy - (s * sx + c * sy)
    res = 0.0
    for p, q in zip(src, dst):
        mx = c * p[0] - s * p[1] + tx
        my = s * p[0] + c * p[1] + ty
        res = max(res, math.sqrt((mx - q[0]) ** 2 + (my - q[1]) ** 2))
    # scale diagnostic
    ssrc = sum((p[0] - sx) ** 2 + (p[1] - sy) ** 2 for p in src)
    sdst = sum((q[0] - dx) ** 2 + (q[1] - dy) ** 2 for q in dst)
    scale = math.sqrt(sdst / ssrc) if ssrc > 1e-9 else 1.0
    return th, tx, ty, res, scale


def lerp(a, b, f):
    return [a[i] + (b[i] - a[i]) * f for i in range(len(a))]


ended = [False]


def is_end(res):
    try:
        if bool(res.terminated) or bool(res.truncated):
            return True
    except Exception:
        pass
    return False


# ------------------------------------------------------------------ main
ctx = get_task_context()
diag["language"] = ctx.language
meta = robomme_plus.get_task_metadata()
max_steps = 1000
try:
    max_steps = int(meta["max_steps"])
except Exception:
    pass
diag["max_steps"] = max_steps
try:
    diag["video_sampling"] = dict(meta["conditioning_video_sampling"])
except Exception:
    pass

video = robomme_plus.get_conditioning_video()
nvid = len(video)
diag["video_frames"] = nvid
state0 = get_robot_state()
emb = state0.embodiment
q_live0 = [float(v) for v in state0.joint_positions["primary"]]
grip0 = float(state0.gripper_positions["primary"])
live_obs = get_observation()

frames = []
for fr in video:
    rs = fr.robot_state
    ee = rs.end_effector_poses["primary"]
    frames.append({
        "t": float(fr.timestamp_s),
        "q": [float(v) for v in rs.joint_positions["primary"]],
        "p": [float(v) for v in ee.position],
        "quat": [float(v) for v in ee.quaternion_wxyz],
    })
if nvid > 0:
    t0 = frames[0]["t"]
    for f in frames:
        f["t"] = f["t"] - t0
    diag["video_duration_s"] = round(frames[-1]["t"], 3)
    zmin = min(f["p"][2] for f in frames)
    diag["video_ee_zmin"] = round(zmin, 4)
    diag["video_ee_z0"] = round(frames[0]["p"][2], 4)
    diag["video_q0_vs_live_maxdiff"] = round(max(abs(a - b) for a, b in zip(frames[0]["q"], q_live0)), 4)

vt = None
lt = None
if nvid > 0:
    try:
        vt = detect_targets(video[0], "video_targets")
    except Exception as exc:
        diag["video_detect_error"] = str(exc)[:300]
try:
    lt = detect_targets(live_obs, "live_targets")
except Exception as exc:
    diag["live_detect_error"] = str(exc)[:300]

def keep_largest(blobs, n):
    def size_key(b):
        return -b["n"]
    return sorted(blobs, key=size_key)[:n]


if vt is not None and lt is not None and len(vt) != len(lt) and min(len(vt), len(lt)) >= 3:
    nkeep = min(len(vt), len(lt))
    diag["count_mismatch"] = [len(vt), len(lt)]
    vt = keep_largest(vt, nkeep)
    lt = keep_largest(lt, nkeep)

# cross-check of the hand-written projection against the public localization tool (diagnostic)
try:
    loc = localize_object("white round target disc", camera_name=CAM, target_frame=state0.base_frame)
    if loc.ok:
        cpos = [float(v) for v in loc.geometry.pose.position]
        dmin = None
        for t in (lt or []):
            d = math.sqrt((t["xyz"][0] - cpos[0]) ** 2 + (t["xyz"][1] - cpos[1]) ** 2)
            if dmin is None or d < dmin:
                dmin = d
        diag["localize_check"] = {"center": [round(v, 4) for v in cpos],
                                  "nearest_detected_xy_m": None if dmin is None else round(dmin, 4)}
    else:
        diag["localize_check"] = "not ok"
except Exception as exc:
    diag["localize_check"] = "error " + str(exc)[:150]

if nvid > 0:
    diag["video_ee_path_every8"] = [[round(f["t"], 2)] + [round(v, 3) for v in f["p"]] for f in frames[::8]]

th, tx, ty = 0.0, 0.0, 0.0
mode = "replay"
if vt is not None and lt is not None and len(vt) >= 2 and len(vt) == len(lt):
    vo = order_row(vt)
    lo = order_row(lt)
    th, tx, ty, res, scale = fit_rigid([t["xyz"] for t in vo], [t["xyz"] for t in lo])
    diag["fit"] = {"yaw_deg": round(math.degrees(th), 3), "tx": round(tx, 4), "ty": round(ty, 4),
                   "max_residual": round(res, 4), "scale": round(scale, 4)}
    if abs(th) > IDENTITY_YAW or math.sqrt(tx * tx + ty * ty) > IDENTITY_POS:
        mode = "ik"
    # demonstrated route summary: nearest video target whenever the EE is near its lowest height
    route = []
    for f in frames:
        if f["p"][2] < diag["video_ee_zmin"] + 0.01:
            bi = None
            bd = 1e9
            for k, t in enumerate(vo):
                d = math.sqrt((f["p"][0] - t["xyz"][0]) ** 2 + (f["p"][1] - t["xyz"][1]) ** 2)
                if d < bd:
                    bd, bi = d, k
            if bd < 0.03 and (not route or route[-1][0] != bi):
                route.append([bi, round(f["t"], 2), round(bd, 3)])
    diag["video_route_low_ee"] = route
else:
    diag["fit"] = "unavailable (counts video=%s live=%s)" % (
        None if vt is None else len(vt), None if lt is None else len(lt))
diag["mode"] = mode

# ------------------------------------------------------------------ build joint waypoints
waypoints = []   # [t, q]
ik_fail = 0
ik_calls = 0
big_jumps = 0
if nvid > 0:
    if mode == "replay":
        for f in frames:
            waypoints.append([f["t"], f["q"]])
    else:
        c, s = math.cos(th), math.sin(th)
        qz = [math.cos(th / 2), 0.0, 0.0, math.sin(th / 2)]
        stride = max(1, int(math.ceil(nvid / float(MAX_IK))))
        idx = list(range(0, nvid, stride))
        if idx[-1] != nvid - 1:
            idx.append(nvid - 1)
        prev = None
        for i in idx:
            f = frames[i]
            p = f["p"]
            mp = [c * p[0] - s * p[1] + tx, s * p[0] + c * p[1] + ty, p[2]]
            mq = quat_mul(qz, f["quat"])
            target = Pose(position=mp, quaternion_wxyz=mq, frame=state0.base_frame)
            vseed = list(f["q"])
            vseed[0] = vseed[0] + th
            seed = prev if prev is not None else vseed
            ik_calls += 1
            sol = None
            try:
                ik = solve_ik(target, seed_joints=seed)
                if ik.ok:
                    sol = [float(v) for v in ik.joint_positions]
            except Exception as exc:
                diag["ik_exc"] = str(exc)[:200]
            if prev is not None and (sol is None or max(abs(a - b) for a, b in zip(sol, prev)) > 0.35):
                ik_calls += 1
                try:
                    ik2 = solve_ik(target, seed_joints=vseed)
                    if ik2.ok:
                        sol2 = [float(v) for v in ik2.joint_positions]
                        if sol is None or max(abs(a - b) for a, b in zip(sol2, prev)) < max(abs(a - b) for a, b in zip(sol, prev)):
                            sol = sol2
                except Exception:
                    pass
                if sol is not None and max(abs(a - b) for a, b in zip(sol, prev)) > 0.35:
                    big_jumps += 1
            if sol is None:
                ik_fail += 1
                continue
            waypoints.append([f["t"], sol])
            prev = sol
diag["ik_calls"] = ik_calls
diag["ik_fail"] = ik_fail
diag["ik_big_jumps"] = big_jumps
diag["n_waypoints"] = len(waypoints)

# ------------------------------------------------------------------ execute
steps_count = [0]
status = "not_started"


def send(q):
    r = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=list(q), gripper_position=grip0, embodiment=emb)}))
    steps_count[0] += 1
    if is_end(r):
        ended[0] = True
    return r


if len(waypoints) >= 2 and not ended[0]:
    budget = max_steps - 40
    # approach from the live start to the first waypoint
    q_first = waypoints[0][1]
    gap = max(abs(a - b) for a, b in zip(q_first, q_live0))
    n_app = 0
    if gap > 0.01:
        n_app = min(80, max(5, int(math.ceil(gap / 0.03))))
        for k in range(1, n_app + 1):
            send(lerp(q_live0, q_first, k / float(n_app)))
            if ended[0]:
                break
    diag["approach_ticks"] = n_app
    total_t = waypoints[-1][0] - waypoints[0][0]
    n_ticks = int(math.ceil(total_t / TICK))
    speed = 1.0
    if n_ticks > budget - n_app - 20:
        speed = n_ticks / float(max(1, budget - n_app - 20))
        n_ticks = budget - n_app - 20
    diag["path_ticks_planned"] = n_ticks
    diag["time_scale"] = round(speed, 3)
    j = 0
    tstart = waypoints[0][0]
    for k in range(1, n_ticks + 1):
        if ended[0]:
            break
        tt = tstart + k * TICK * speed
        while j < len(waypoints) - 2 and waypoints[j + 1][0] < tt:
            j += 1
        a = waypoints[j]
        b = waypoints[j + 1]
        span = b[0] - a[0]
        fr = 1.0 if span <= 1e-9 else max(0.0, min(1.0, (tt - a[0]) / span))
        send(lerp(a[1], b[1], fr))
    # hold the final pose briefly so the last contact registers
    hold = 0
    while not ended[0] and hold < 30 and steps_count[0] < max_steps - 5:
        send(waypoints[-1][1])
        hold += 1
    diag["hold_ticks"] = hold
    status = "terminated" if ended[0] else "path_done"
else:
    status = "no_waypoints"
diag["steps_sent"] = steps_count[0]
diag["status"] = status
if not ended[0]:
    try:
        st = get_robot_state()
        diag["final_ee"] = [round(float(v), 4) for v in st.end_effector_poses["primary"].position]
        diag["final_t"] = float(st.timestamp_s)
    except Exception as exc:
        diag["final_state_error"] = str(exc)[:200]
result = diag
