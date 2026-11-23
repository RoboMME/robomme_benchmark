# v04 -- RouteStick (from v03): lift until the stick TIP (not only the hand body) clears the obstacle
#        tops before re-orienting; keep the hand centred in free space (soft clearance target 5 cm);
#        use more of the episode budget for a slower replay (x2.5 max).
# v03 -- RouteStick (from v02): when the (shorter) live tool forces the hand body below the obstacle
#        tops, schedule the wrist yaw so the hand slab threads between the obstacles detected in the
#        live front RGB-D (dynamic programming over yaw offsets, joint-7-limit aware, 180-degree
#        flips done as lift-rotate-lower at target dwells), and slow the replay down.
# v02 -- RouteStick (from v01): measure the stick tip in the wrist RGB-D of video frame 0 and of the
#        live start (same home pose) and replay the TIP path, not the end-effector path.
# v01 -- RouteStick (from v00): drop detector clusters at the robot base, warp residuals.
# v00 -- RouteStick: layout-registered replay of the conditioning video's end-effector path.
#
# Idea (from offline analysis of the released recordings): the conditioning video shows the
# robot routing its drawing stick between white target discs and around the obstacles that
# separate them; the live scene contains the same target/obstacle row, possibly shifted/rotated
# as a whole.  The conditioning video carries the recorded robot state per frame, so the expert
# end-effector path is known in robot_base.  We
#   1. detect the white target discs in the video's first front frame and in the live front
#      frame with the SAME pixel+depth detector (projection convention self-calibrated by
#      requiring the table to be a horizontal plane),
#   2. fit a 2-D rigid transform (yaw + xy translation) video->live from index-ordered targets,
#   3. replay the video EE path through that transform (IK per frame, seeded by the previous
#      solution) with one joint-position command per control tick.
# Nothing is copied from the recordings: every position comes from the live episode's own
# conditioning video and live observation.
import math

result = {"version": "v04", "stage": "start"}

ctx = get_task_context()
goal = str(ctx.language)
result["goal"] = goal

meta = {}
try:
    meta = robomme_plus.get_task_metadata()
except Exception as exc:
    result["meta_error"] = str(exc)
meta_out = {}
try:
    for key in meta:
        meta_out[str(key)] = str(meta[key])[:300]
except Exception as exc:
    meta_out["error"] = str(exc)
result["task_metadata"] = meta_out
max_steps = None
try:
    max_steps = int(meta["max_steps"])
except Exception:
    max_steps = None
hz = 20.0
try:
    hz = float(meta["control_hz"])
except Exception:
    try:
        hz = float(meta["control_frequency"])
    except Exception:
        hz = 20.0
result["max_steps"] = max_steps
result["hz"] = hz
try:
    cm = robomme_plus.get_controller_metadata()
    result["controller_metadata"] = {str(k): str(cm[k])[:300] for k in cm}
except Exception as exc:
    result["controller_metadata_error"] = str(exc)


# ---------------------------------------------------------------- small math helpers
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def qmat(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def to_list(arr):
    try:
        return arr.tolist()
    except Exception:
        return [list(row) for row in arr]


def median(vals):
    s = sorted(vals)
    if len(s) == 0:
        return None
    return s[len(s) // 2]


# ---------------------------------------------------------------- camera projection
CONVENTIONS = ("cv", "gl", "sapien")


def cam_model(cam):
    K = to_list(cam.intrinsics)
    pose = cam.camera_pose
    R = qmat([float(v) for v in pose.quaternion_wxyz])
    t = [float(v) for v in pose.position]
    return {"fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]),
            "cy": float(K[1][2]), "R": R, "t": t}


def project(model, conv, u, v, d):
    a = (u + 0.5 - model["cx"]) / model["fx"] * d
    b = (v + 0.5 - model["cy"]) / model["fy"] * d
    if conv == "cv":
        pc = (a, b, d)
    elif conv == "gl":
        pc = (a, -b, -d)
    else:
        pc = (d, -a, -b)
    R = model["R"]
    t = model["t"]
    return [R[0][0] * pc[0] + R[0][1] * pc[1] + R[0][2] * pc[2] + t[0],
            R[1][0] * pc[0] + R[1][1] * pc[1] + R[1][2] * pc[2] + t[1],
            R[2][0] * pc[0] + R[2][1] * pc[1] + R[2][2] * pc[2] + t[2]]


def is_table(px):
    r, g, b = px[0], px[1], px[2]
    return (max(r, g, b) - min(r, g, b)) > 60 and r > g and g > b


def calibrate(rgb, depth, model):
    """Pick the projection convention under which table pixels form the flattest horizontal plane."""
    H = len(rgb)
    W = len(rgb[0])
    samples = []
    for v in range(0, H, 6):
        for u in range(0, W, 6):
            d = float(depth[v][u])
            if d > 0.05 and d < 5.0 and is_table(rgb[v][u]):
                samples.append((u, v, d))
    best = None
    report = {}
    for conv in CONVENTIONS:
        zs = [project(model, conv, s[0], s[1], s[2])[2] for s in samples]
        if len(zs) < 20:
            continue
        mz = sum(zs) / len(zs)
        sd = math.sqrt(sum((z - mz) ** 2 for z in zs) / len(zs))
        report[conv] = {"z_mean": round(mz, 4), "z_std": round(sd, 4), "n": len(zs)}
        if mz > model["t"][2] - 0.05:
            report[conv]["rejected"] = "table above camera"
            continue
        if best is None or sd < report[best]["z_std"]:
            best = conv
    return best, report


def detect_targets(cam, conv_hint=None):
    """White target discs lying flat on the table: low-saturation bright pixels at table height."""
    rgb = to_list(cam.rgb)
    depth = to_list(cam.depth_m)
    model = cam_model(cam)
    conv, report = calibrate(rgb, depth, model)
    if conv_hint is not None:
        conv = conv_hint
    if conv is None:
        return {"ok": False, "why": "calibration failed", "calib": report}
    table_z = report[conv]["z_mean"]
    H = len(rgb)
    W = len(rgb[0])
    pts = []
    for v in range(H):
        row = rgb[v]
        for u in range(W):
            px = row[u]
            r, g, b = px[0], px[1], px[2]
            mx = max(r, g, b)
            if mx - min(r, g, b) > 40 or (r + g + b) < 390:
                continue
            d = float(depth[v][u])
            if d <= 0.05 or d >= 5.0:
                continue
            p = project(model, conv, u, v, d)
            if abs(p[2] - table_z) > 0.02:
                continue
            pts.append((p, u, v, (r + g + b) / 3.0))
    clusters = []
    for item in pts:
        p = item[0]
        placed = False
        for c in clusters:
            cx = c["sx"] / c["n"]
            cy = c["sy"] / c["n"]
            if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 < 0.025 ** 2:
                c["sx"] += p[0]
                c["sy"] += p[1]
                c["sz"] += p[2]
                c["su"] += item[1]
                c["sv"] += item[2]
                c["n"] += 1
                if item[3] > 175:
                    c["bright"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"sx": p[0], "sy": p[1], "sz": p[2], "su": item[1], "sv": item[2],
                             "n": 1, "bright": 1 if item[3] > 175 else 0})
    out = []
    for c in clusters:
        cxy = [c["sx"] / c["n"], c["sy"] / c["n"]]
        # the robot base sits at the robot_base origin; nothing on the task row lies within 0.2 m
        if math.sqrt(cxy[0] ** 2 + cxy[1] ** 2) < 0.2:
            continue
        if c["n"] >= 12 and c["bright"] >= 3:
            out.append({"xyz": [c["sx"] / c["n"], c["sy"] / c["n"], c["sz"] / c["n"]],
                        "uv": [c["su"] / c["n"], c["sv"] / c["n"]], "n": c["n"]})
    out = sorted(out, key=lambda c: c["uv"][0])
    return {"ok": len(out) >= 2, "targets": out, "conv": conv, "calib": report,
            "table_z": table_z, "candidate_pixels": len(pts), "raw_clusters": len(clusters)}


def fit_rigid(src, dst):
    n = len(src)
    sx = sum(p[0] for p in src) / n
    sy = sum(p[1] for p in src) / n
    dx = sum(p[0] for p in dst) / n
    dy = sum(p[1] for p in dst) / n
    num = 0.0
    den = 0.0
    for a, b in zip(src, dst):
        ax, ay = a[0] - sx, a[1] - sy
        bx, by = b[0] - dx, b[1] - dy
        num += ax * by - ay * bx
        den += ax * bx + ay * by
    th = math.atan2(num, den)
    c, s = math.cos(th), math.sin(th)
    res = []
    for a, b in zip(src, dst):
        ax, ay = a[0] - sx, a[1] - sy
        px = c * ax - s * ay + dx
        py = s * ax + c * ay + dy
        res.append(math.sqrt((px - b[0]) ** 2 + (py - b[1]) ** 2))
    return {"theta": th, "src_c": [sx, sy], "dst_c": [dx, dy], "residuals": res,
            "rms": math.sqrt(sum(r * r for r in res) / n)}


def apply_rigid(tf, p):
    c, s = math.cos(tf["theta"]), math.sin(tf["theta"])
    ax, ay = p[0] - tf["src_c"][0], p[1] - tf["src_c"][1]
    out = [c * ax - s * ay + tf["dst_c"][0], s * ax + c * ay + tf["dst_c"][1]]
    # smooth residual warp: each matched video target lands exactly on its live target,
    # blended with a Gaussian kernel (sigma 4 cm) so far-away points keep the rigid map
    warp = tf.get("warp", [])
    if len(warp) > 0:
        wsum = 0.0
        ox = 0.0
        oy = 0.0
        for w in warp:
            d2 = (p[0] - w[0]) ** 2 + (p[1] - w[1]) ** 2
            k = math.exp(-d2 / (2 * 0.04 ** 2))
            wsum += k
            ox += k * w[2]
            oy += k * w[3]
        if wsum > 1e-9:
            k = min(1.0, wsum)
            out = [out[0] + k * ox / wsum, out[1] + k * oy / wsum]
    return out


def matvec(R, v):
    return [R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
            R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
            R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]]


def mattvec(R, v):
    return [R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2],
            R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2],
            R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2]]


def tool_tip(cam, ee_pos, ee_quat):
    """Lowest robot-attached point seen by the wrist camera = stick tip.

    Robot points: projected wrist pixels more than 3 cm above the table plane and within
    0.30 m of the end-effector reference.  Returns the tip offset in the EE-local frame."""
    rgb = to_list(cam.rgb)
    depth = to_list(cam.depth_m)
    model = cam_model(cam)
    conv, report = calibrate(rgb, depth, model)
    if conv is None:
        return {"ok": False, "why": "wrist calibration failed", "calib": report}
    table_z = report[conv]["z_mean"]
    pts = []
    H = len(depth)
    W = len(depth[0])
    for v in range(H):
        for u in range(W):
            d = float(depth[v][u])
            if d <= 0.005 or d > 0.8:
                continue
            p = project(model, conv, u, v, d)
            if p[2] < table_z + 0.03:
                continue
            dd = math.sqrt((p[0] - ee_pos[0]) ** 2 + (p[1] - ee_pos[1]) ** 2 + (p[2] - ee_pos[2]) ** 2)
            if dd > 0.30:
                continue
            pts.append(p)
    if len(pts) < 10:
        return {"ok": False, "why": "too few robot points", "n": len(pts), "conv": conv}
    pts = sorted(pts, key=lambda q: q[2])
    low = pts[:5]
    tip = [sum(q[i] for q in low) / len(low) for i in range(3)]
    R = qmat(ee_quat)
    local = mattvec(R, [tip[0] - ee_pos[0], tip[1] - ee_pos[1], tip[2] - ee_pos[2]])
    return {"ok": True, "tip": tip, "local": local, "n": len(pts), "conv": conv,
            "table_z": table_z, "highest_robot_z": pts[-1][2]}


# ---------------------------------------------------------------- 1. conditioning video
video = robomme_plus.get_conditioning_video()
nv = len(video)
result["video_frames"] = nv
if nv < 2:
    raise ValueError("no conditioning video")
traj = []
for fr in video:
    rs = fr.robot_state
    pose = rs.end_effector_poses["primary"]
    traj.append({"t": float(fr.timestamp_s),
                 "p": [float(c) for c in pose.position],
                 "q": [float(c) for c in pose.quaternion_wxyz],
                 "j": [float(c) for c in rs.joint_positions["primary"]],
                 "g": float(rs.gripper_positions["primary"])})
result["video_t_range"] = [traj[0]["t"], traj[-1]["t"]]
result["video_ee_path"] = [[round(f["t"], 3)] + [round(c, 4) for c in f["p"]] for f in traj]
result["video_gripper_first_last"] = [traj[0]["g"], traj[-1]["g"]]
result["video_q0"] = traj[0]["q"]

# motion start: first frame whose joints leave the initial configuration
k0 = 0
for k in range(nv):
    dj = max(abs(a - b) for a, b in zip(traj[k]["j"], traj[0]["j"]))
    if dj > 0.002:
        k0 = max(0, k - 1)
        break
result["motion_start_frame"] = k0

vcam0 = video[0].cameras["front"]
vdet = detect_targets(vcam0)
result["video_detection"] = {"conv": vdet.get("conv"), "calib": vdet.get("calib"),
                             "table_z": vdet.get("table_z"),
                             "targets": [{"xyz": [round(c, 4) for c in t["xyz"]],
                                          "uv": [round(c, 1) for c in t["uv"]], "n": t["n"]}
                                         for t in vdet.get("targets", [])]}
try:
    result["video_front_pose"] = [float(c) for c in vcam0.camera_pose.position] + \
        [float(c) for c in vcam0.camera_pose.quaternion_wxyz]
except Exception as exc:
    result["video_front_pose"] = str(exc)

# ---------------------------------------------------------------- 2. live scene
obs = get_observation()
lcam = obs.cameras["front"]
ldet = detect_targets(lcam)
result["live_detection"] = {"conv": ldet.get("conv"), "calib": ldet.get("calib"),
                            "table_z": ldet.get("table_z"),
                            "targets": [{"xyz": [round(c, 4) for c in t["xyz"]],
                                         "uv": [round(c, 1) for c in t["uv"]], "n": t["n"]}
                                        for t in ldet.get("targets", [])]}
try:
    result["live_front_pose"] = [float(c) for c in lcam.camera_pose.position] + \
        [float(c) for c in lcam.camera_pose.quaternion_wxyz]
except Exception as exc:
    result["live_front_pose"] = str(exc)

# diagnostic SAM view of the targets (not used for control)
try:
    seg = segment_text("front", "white round target disc")
    result["sam_targets"] = {"ok": bool(seg.ok), "n": len(seg.segmentations),
                             "boxes": [[round(float(c), 1) for c in s.box_xyxy] for s in seg.segmentations[:8]],
                             }
except Exception as exc:
    result["sam_targets"] = {"error": str(exc)}

state0 = get_robot_state()
emb = state0.embodiment
base = state0.base_frame
g_hold = float(state0.gripper_positions["primary"])
live_j0 = [float(c) for c in state0.joint_positions["primary"]]
result["live_j0"] = [round(c, 4) for c in live_j0]
result["video_j0"] = [round(c, 4) for c in traj[0]["j"]]
result["live_gripper"] = g_hold
result["live_ee0"] = [float(c) for c in state0.end_effector_poses["primary"].position]
live_q0 = [float(c) for c in state0.end_effector_poses["primary"].quaternion_wxyz]

vtip = tool_tip(video[0].cameras["wrist"], traj[0]["p"], traj[0]["q"])
ltip = tool_tip(obs.cameras["wrist"], result["live_ee0"], live_q0)
result["video_tool_tip"] = {k: (([round(c, 4) for c in vtip[k]]) if k in ("tip", "local") else vtip[k]) for k in vtip}
result["live_tool_tip"] = {k: (([round(c, 4) for c in ltip[k]]) if k in ("tip", "local") else ltip[k]) for k in ltip}
tool_local_v = [0.0, 0.0, 0.0]
tool_local_l = [0.0, 0.0, 0.0]
if vtip["ok"] and ltip["ok"]:
    tool_local_v = vtip["local"]
    tool_local_l = ltip["local"]
    # sanity bound: a tool-offset difference above 25 cm means a measurement failure
    if math.sqrt(sum((a - b) ** 2 for a, b in zip(tool_local_v, tool_local_l))) > 0.25:
        tool_local_v = [0.0, 0.0, 0.0]
        tool_local_l = [0.0, 0.0, 0.0]
        result["tool_offset_rejected"] = True
result["tool_local_delta"] = [round(a - b, 4) for a, b in zip(tool_local_v, tool_local_l)]

# ---------------------------------------------------------------- 3. layout transform
tf = None
vt = vdet.get("targets", [])
lt = ldet.get("targets", [])
if len(vt) >= 2 and len(lt) >= 2:
    best = None
    m = min(len(vt), len(lt))
    for ov in range(len(vt) - m + 1):
        for ol in range(len(lt) - m + 1):
            cand = fit_rigid([t["xyz"] for t in vt[ov:ov + m]], [t["xyz"] for t in lt[ol:ol + m]])
            if best is None or cand["rms"] < best["rms"]:
                best = cand
                best_ov = ov
                best_ol = ol
    tf = best
    # residual warp terms, only when the rigid model is already a good description (< 3 cm)
    if tf is not None and max(tf["residuals"]) < 0.03:
        src = [t["xyz"] for t in vt[best_ov:best_ov + m]]
        dst = [t["xyz"] for t in lt[best_ol:best_ol + m]]
        warp = []
        for a, b in zip(src, dst):
            pr = apply_rigid(tf, a)
            warp.append([a[0], a[1], b[0] - pr[0], b[1] - pr[1]])
        tf["warp"] = warp
if tf is None:
    tf = {"theta": 0.0, "src_c": [0.0, 0.0], "dst_c": [0.0, 0.0], "residuals": [], "rms": None}
    result["transform_fallback"] = "identity (target detection failed)"
dz = 0.0
if vdet.get("table_z") is not None and ldet.get("table_z") is not None:
    dz = float(ldet["table_z"]) - float(vdet["table_z"])
    if abs(dz) > 0.02:
        dz = 0.0
shift = apply_rigid(tf, [0.0, 0.0])
result["transform"] = {"theta_deg": round(math.degrees(tf["theta"]), 3),
                       "warp": [[round(c, 4) for c in w] for w in tf.get("warp", [])],
                       "origin_shift": [round(c, 4) for c in shift],
                       "rms": tf["rms"], "residuals": [round(r, 4) for r in tf["residuals"]],
                       "dz": round(dz, 4)}
identity = abs(tf["theta"]) < 0.003 and \
    math.sqrt((tf["dst_c"][0] - tf["src_c"][0]) ** 2 + (tf["dst_c"][1] - tf["src_c"][1]) ** 2) < 0.003 \
    and abs(dz) < 0.003 and \
    math.sqrt(sum((a - b) ** 2 for a, b in zip(tool_local_v, tool_local_l))) < 0.003
result["identity_transform"] = identity
half = tf["theta"] / 2.0
qyaw = [math.cos(half), 0.0, 0.0, math.sin(half)]


def mapped_pose(f):
    Rv = qmat(f["q"])
    ov = matvec(Rv, tool_local_v)
    tip_v = [f["p"][0] + ov[0], f["p"][1] + ov[1], f["p"][2] + ov[2]]
    xy = apply_rigid(tf, tip_v)
    q = qmul(qyaw, f["q"])
    ol = matvec(qmat(q), tool_local_l)
    return Pose([xy[0] - ol[0], xy[1] - ol[1], tip_v[2] + dz - ol[2]], q, base)



# ---------------------------------------------------------------- 3b. live obstacles
def seg_dist(px, py, ax, ay, bx, by):
    vx = bx - ax
    vy = by - ay
    L2 = vx * vx + vy * vy
    if L2 < 1e-12:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    t = ((px - ax) * vx + (py - ay) * vy) / L2
    t = max(0.0, min(1.0, t))
    return math.sqrt((px - ax - t * vx) ** 2 + (py - ay - t * vy) ** 2)


def seg_seg(a, b):
    """2-D distance between segments a=[x0,y0,x1,y1] and b."""
    def cross(ox, oy, px, py, qx, qy):
        return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)
    d1 = cross(a[0], a[1], a[2], a[3], b[0], b[1])
    d2 = cross(a[0], a[1], a[2], a[3], b[2], b[3])
    d3 = cross(b[0], b[1], b[2], b[3], a[0], a[1])
    d4 = cross(b[0], b[1], b[2], b[3], a[2], a[3])
    if d1 * d2 < 0 and d3 * d4 < 0:
        return 0.0
    return min(seg_dist(a[0], a[1], b[0], b[1], b[2], b[3]), seg_dist(a[2], a[3], b[0], b[1], b[2], b[3]),
               seg_dist(b[0], b[1], a[0], a[1], a[2], a[3]), seg_dist(b[2], b[3], a[0], a[1], a[2], a[3]))


def detect_obstacles(cam, conv, table_z, tpts):
    """Tall objects standing on the table near the target row (front RGB-D, same projection)."""
    depth = to_list(cam.depth_m)
    model = cam_model(cam)
    H = len(depth)
    W = len(depth[0])
    clusters = []
    for v in range(H):
        for u in range(W):
            d = float(depth[v][u])
            if d <= 0.05 or d >= 5.0:
                continue
            p = project(model, conv, u, v, d)
            if p[2] < table_z + 0.04 or p[2] > table_z + 0.35:
                continue
            if math.sqrt(p[0] ** 2 + p[1] ** 2) < 0.2:
                continue
            if len(tpts) >= 2:
                dmin = min(seg_dist(p[0], p[1], tpts[i][0], tpts[i][1], tpts[i + 1][0], tpts[i + 1][1])
                           for i in range(len(tpts) - 1))
                if dmin > 0.12:
                    continue
            placed = False
            for c in clusters:
                if (p[0] - c["sx"] / c["n"]) ** 2 + (p[1] - c["sy"] / c["n"]) ** 2 < 0.05 ** 2:
                    c["sx"] += p[0]
                    c["sy"] += p[1]
                    c["n"] += 1
                    c["pts"].append(p)
                    placed = True
                    break
            if not placed:
                clusters.append({"sx": p[0], "sy": p[1], "n": 1, "pts": [p]})
    out = []
    for c in clusters:
        if c["n"] < 30:
            continue
        zmax = max(q[2] for q in c["pts"])
        top = [q for q in c["pts"] if q[2] > zmax - 0.004]
        if len(top) < 12:
            top = [q for q in c["pts"] if q[2] > zmax - 0.015]
        cx = sum(q[0] for q in top) / len(top)
        cy = sum(q[1] for q in top) / len(top)
        sxx = sum((q[0] - cx) ** 2 for q in top) / len(top)
        syy = sum((q[1] - cy) ** 2 for q in top) / len(top)
        sxy = sum((q[0] - cx) * (q[1] - cy) for q in top) / len(top)
        ang = 0.5 * math.atan2(2 * sxy, sxx - syy)
        ux, uy = math.cos(ang), math.sin(ang)
        pa = [(q[0] - cx) * ux + (q[1] - cy) * uy for q in top]
        pb = [-(q[0] - cx) * uy + (q[1] - cy) * ux for q in top]
        ha = (max(pa) - min(pa)) / 2.0
        hb = (max(pb) - min(pb)) / 2.0
        ha = max(0.01, min(0.06, ha))
        hb = max(0.01, min(ha, hb))
        # capsule model of the top face: segment of half-length (ha-hb) along its long axis, radius hb
        e = ha - hb
        out.append({"xy": [cx, cy], "top": zmax, "r": hb, "ha": ha,
                    "seg": [cx - e * ux, cy - e * uy, cx + e * ux, cy + e * uy],
                    "ang_deg": math.degrees(ang), "n": c["n"]})
    return out


obstacles = []
obst_src = "none"
try:
    if ldet.get("conv") is not None:
        obstacles = detect_obstacles(lcam, ldet["conv"], float(ldet["table_z"]), [t["xyz"] for t in lt])
        obst_src = "front_depth"
except Exception as exc:
    result["obstacle_error"] = str(exc)[:200]
result["obstacles"] = [{"xy": [round(c, 4) for c in o["xy"]], "top": round(o["top"], 4), "r": round(o["r"], 4),
                        "ha": round(o["ha"], 4), "ang_deg": round(o["ang_deg"], 1), "n": o["n"]} for o in obstacles]
result["obstacle_source"] = obst_src

# generic hand-slab tolerances (robot geometry, measured on live training rollouts, not recordings)
HB = 0.033   # hand-body bottom above the EE reference point
HL = 0.11    # hand half-length along EE local y (finger-opening axis)
HW = 0.03    # hand half-thickness along EE local x
LIFT_CLEAR = 0.02

# ---------------------------------------------------------------- 4. plan
ticks = []
for k in range(k0 + 1, nv):
    ticks.append(max(1, int(round((traj[k]["t"] - traj[k - 1]["t"]) * hz))))
total = sum(ticks)
plan = []
for k in range(k0, nv):
    f = traj[k]
    Rv = qmat(f["q"])
    ov = matvec(Rv, tool_local_v)
    tip_v = [f["p"][0] + ov[0], f["p"][1] + ov[1], f["p"][2] + ov[2]]
    xy = apply_rigid(tf, tip_v)
    q0 = qmul(qyaw, f["q"])
    R0 = qmat(q0)
    ol = matvec(R0, tool_local_l)
    tipz = tip_v[2] + dz
    plan.append({"k": k, "tip": [xy[0], xy[1], tipz], "eez": tipz - ol[2], "ol": ol,
                 "psi0": math.atan2(R0[1][1], R0[0][1]), "j7": f["j"][6]})
npl = len(plan)
top_max = None
if len(obstacles) > 0:
    top_max = max(o["top"] for o in obstacles)
for pl in plan:
    pl["low"] = top_max is not None and (pl["eez"] + HB) < top_max + 0.01
nlow = sum(1 for pl in plan if pl["low"])
manage = (not identity) and nlow > 0
result["yaw_management"] = manage
result["low_frames"] = nlow

# time scaling: slower replay when there is room in the episode budget
scale = 1.0
if max_steps is not None:
    room = max_steps - 150 - (600 if manage else 0)
    scale = min(2.5 if manage else 1.0, float(room) / max(1, total))
    if manage and scale < 2.0:
        scale = min(2.0, float(max_steps - 150 - 400) / max(1, total))
    if scale < 0.3:
        scale = 0.3
# cumulative rounding so fractional scales are honoured
acc = 0.0
emitted = 0
scaled = []
for t in ticks:
    acc += t * scale
    n = max(1, int(round(acc)) - emitted)
    emitted += n
    scaled.append(n)
ticks = scaled
result["time_scale"] = round(scale, 3)

STEP = math.radians(5.0)
NB = 72
NS = 144
phis = [-2 * math.pi + j * STEP for j in range(NS)]
s7 = -1.0


def pose_at(i, phi, lift):
    pl = plan[i]
    f = traj[pl["k"]]
    th = tf["theta"] + phi
    q = qmul([math.cos(th / 2.0), 0.0, 0.0, math.sin(th / 2.0)], f["q"])
    ol = matvec(qmat(q), tool_local_l)
    t = pl["tip"]
    return Pose([t[0] - ol[0], t[1] - ol[1], t[2] + lift - ol[2]], q, base)


def ik_sol(pose, seed):
    try:
        ik = solve_ik(pose, seed_joints=seed)
        if ik.ok:
            return [float(c) for c in ik.joint_positions]
    except Exception:
        return None
    return None


phi_sched = [0.0] * npl
lift_after = []
if manage:
    # joint-7 response to a world-yaw offset, probed once by IK
    try:
        a = ik_sol(pose_at(0, 0.0, 0.0), traj[plan[0]["k"]]["j"])
        b = ik_sol(pose_at(0, 0.3, 0.0), traj[plan[0]["k"]]["j"])
        if a is not None and b is not None:
            ratio = (b[6] - a[6]) / 0.3
            if abs(ratio) > 0.5 and abs(ratio) < 1.5:
                s7 = ratio
            result["j7_probe"] = round(ratio, 3)
    except Exception as exc:
        result["j7_probe_error"] = str(exc)[:200]

    def clearance(i, psi):
        pl = plan[i]
        c = math.cos(psi)
        s = math.sin(psi)
        # hand centre = EE reference = tip minus the (yaw-rotated) EE-local tool offset
        dp = psi - pl["psi0"]
        cd = math.cos(dp)
        sd = math.sin(dp)
        hx = pl["tip"][0] - (cd * pl["ol"][0] - sd * pl["ol"][1])
        hy = pl["tip"][1] - (sd * pl["ol"][0] + cd * pl["ol"][1])
        best = 1.0
        for o in obstacles:
            if pl["eez"] + HB > o["top"] + 0.01:
                continue
            d = seg_seg([hx - HL * c, hy - HL * s, hx + HL * c, hy + HL * s], o["seg"])
            best = min(best, d - HW - o["r"])
        return best

    def ccost(cl):
        if cl >= 0.05:
            return 0.0
        v = 2000.0 * (0.05 - cl) ** 2
        if cl < 0.015:
            v += 40000.0 * (0.015 - cl) ** 2
        if cl < 0.0:
            v += 5.0
        return v

    # dwell frames: closest approach to a live target within 2 cm (lift-rotate-lower allowed there)
    tl = [t["xyz"] for t in lt]
    dwell = [False] * npl
    run_t = None
    run_best = None
    for i in range(npl):
        pl = plan[i]
        dd = None
        ti = None
        for j in range(len(tl)):
            dj = math.sqrt((pl["tip"][0] - tl[j][0]) ** 2 + (pl["tip"][1] - tl[j][1]) ** 2)
            if dd is None or dj < dd:
                dd = dj
                ti = j
        inside = pl["low"] and dd is not None and dd < 0.02
        if inside and run_t == ti:
            if dd < run_best[1]:
                run_best = [i, dd]
        else:
            if run_best is not None:
                dwell[run_best[0]] = True
            run_best = None
            run_t = None
            if inside:
                run_t = ti
                run_best = [i, dd]
    if run_best is not None:
        dwell[run_best[0]] = True
    result["dwell_frames"] = [plan[i]["k"] for i in range(npl) if dwell[i]]

    INF = 1e18
    cost_prev = [INF] * NS
    back = []
    min_cl = []
    for j in range(NS):
        if abs(plan[0]["j7"] + s7 * phis[j]) <= 2.65:
            cost_prev[j] = 0.001 * abs(phis[j]) / STEP
    back.append([-1] * NS)
    for i in range(1, npl):
        pl = plan[i]
        cc = [0.0] * NB
        if pl["low"]:
            for m in range(NB):
                cc[m] = ccost(clearance(i, pl["psi0"] - 2 * math.pi + m * STEP))
        R = max(1, ticks[i - 1])
        can_lift = dwell[i - 1]
        lift_best = INF
        lift_arg = -1
        if can_lift:
            for j in range(NS):
                if cost_prev[j] < lift_best:
                    lift_best = cost_prev[j]
                    lift_arg = j
        cur = [INF] * NS
        bk = [-1] * NS
        for j in range(NS):
            if abs(pl["j7"] + s7 * phis[j]) > 2.65:
                continue
            bv = INF
            ba = -1
            for jj in range(max(0, j - R), min(NS, j + R + 1)):
                v = cost_prev[jj] + 0.02 * abs(j - jj)
                if v < bv:
                    bv = v
                    ba = jj
            if can_lift and lift_best + 30.0 < bv:
                bv = lift_best + 30.0
                ba = lift_arg
            cur[j] = bv + cc[j % NB]
            bk[j] = ba
        back.append(bk)
        cost_prev = cur
    jbest = 0
    for j in range(NS):
        if cost_prev[j] < cost_prev[jbest]:
            jbest = j
    result["dp_cost"] = round(cost_prev[jbest], 3)
    idx = [0] * npl
    idx[npl - 1] = jbest
    for i in range(npl - 1, 0, -1):
        idx[i - 1] = back[i][idx[i]]
        if idx[i - 1] < 0:
            idx[i - 1] = idx[i]
    for i in range(npl):
        phi_sched[i] = phis[idx[i]]
    for i in range(1, npl):
        if abs(idx[i] - idx[i - 1]) > max(1, ticks[i - 1]):
            lift_after.append(i - 1)
    cls = []
    for i in range(npl):
        if plan[i]["low"]:
            cls.append(clearance(i, plan[i]["psi0"] + phi_sched[i]))
    if len(cls) > 0:
        result["clearance_min"] = round(min(cls), 4)
        result["clearance_neg_frames"] = sum(1 for c in cls if c < 0)
        result["clearance_below_1cm"] = sum(1 for c in cls if c < 0.01)
    result["lifts_at_frames"] = [plan[i]["k"] for i in lift_after]
    result["phi_sched_deg"] = [round(math.degrees(phi_sched[i]), 1) for i in range(0, npl, 5)]

# ---------------------------------------------------------------- 5. waypoints + IK
wps = []   # (pose or joints, ticks)
ik_fail = 0
prev = None
lift_set = set(lift_after)


def solve_wp(pose, phi, kk):
    global_prev = prev
    cands = []
    seeds = []
    if global_prev is not None:
        seeds.append(global_prev)
    vj = list(traj[kk]["j"])
    vj[6] = vj[6] + s7 * phi
    seeds.append(vj)
    for sd in seeds:
        sol = ik_sol(pose, sd)
        if sol is not None:
            jump = 0.0
            if global_prev is not None:
                jump = max(abs(a - b) for a, b in zip(sol, global_prev))
            cands.append((jump, sol))
    if len(cands) == 0:
        return None
    cands = sorted(cands, key=lambda c: c[0])
    if global_prev is not None and cands[0][0] > 0.6:
        return None
    return cands[0][1]


targets_j = []
tick_list = []
j7_max = 0.0
for i in range(npl):
    if identity:
        targets_j.append(traj[plan[i]["k"]]["j"])
        if i > 0:
            tick_list.append(ticks[i - 1])
        continue
    sol = solve_wp(pose_at(i, phi_sched[i], 0.0), phi_sched[i], plan[i]["k"])
    if sol is None:
        ik_fail += 1
        sol = prev if prev is not None else traj[plan[i]["k"]]["j"]
    targets_j.append(sol)
    if i > 0:
        tick_list.append(ticks[i - 1])
    prev = sol
    j7_max = max(j7_max, abs(sol[6]))
    if i in lift_set and i + 1 < npl:
        lift = 0.0
        if top_max is not None:
            lift = max(0.0, top_max + LIFT_CLEAR - HB - plan[i]["eez"], top_max + LIFT_CLEAR - plan[i]["tip"][2])
        a = phi_sched[i]
        b = phi_sched[i + 1]
        seq = [(a, lift, 12)]
        nrot = max(1, int(math.ceil(abs(b - a) / 0.12)))
        for r in range(1, nrot + 1):
            seq.append((a + (b - a) * r / nrot, lift, 3))
        seq.append((b, 0.0, 12))
        for ph, lf, nt in seq:
            s2 = solve_wp(pose_at(i, ph, lf), ph, plan[i]["k"])
            if s2 is None:
                ik_fail += 1
                continue
            targets_j.append(s2)
            tick_list.append(nt)
            prev = s2
            j7_max = max(j7_max, abs(s2[6]))
result["ik_failures"] = ik_fail
result["j7_abs_max"] = round(j7_max, 3)
result["stage"] = "planned"
result["replay_ticks_planned"] = sum(tick_list)
if max_steps is not None and sum(tick_list) > max_steps - 60:
    result["warning"] = "plan longer than episode budget"

mv = move_to_joints(targets_j[0], tolerance=0.01, max_steps=150)
result["approach"] = {"ok": bool(mv.ok), "steps": int(mv.steps_executed), "status": str(mv.status)}

steps_done = int(mv.steps_executed)
ended = False
step_errors = 0
last_err = None
prevj = targets_j[0]
for i in range(len(tick_list)):
    cur = targets_j[i + 1]
    n = tick_list[i]
    for s in range(n):
        a = float(s + 1) / n
        jt = [p + (c - p) * a for p, c in zip(prevj, cur)]
        sr = step(RobotAction(arms={"primary": ArmCommand("joint_position", jt, gripper_position=g_hold,
                                                         embodiment=emb)}))
        if not sr.ok:
            step_errors += 1
            last_err = str(sr.error)[:200]
            if step_errors > 3:
                ended = True
                break
        else:
            steps_done += 1
        term = False
        try:
            term = bool(sr.terminated) or bool(sr.truncated)
        except Exception:
            term = False
        if term:
            ended = True
            result["terminated_at_waypoint"] = i + 1
            result["terminated_of_waypoints"] = len(tick_list)
            break
    prevj = cur
    if ended:
        break
result["replay_steps"] = steps_done
result["step_errors"] = step_errors
result["last_step_error"] = last_err
result["ended_during_replay"] = ended

if not ended:
    for s in range(20):
        sr = step(RobotAction(arms={"primary": ArmCommand("joint_position", prevj, gripper_position=g_hold,
                                                         embodiment=emb)}))
        try:
            if bool(sr.terminated) or bool(sr.truncated):
                result["terminated_in_hold"] = s
                break
        except Exception:
            break
try:
    fs = get_robot_state()
    result["final_ee"] = [round(float(c), 4) for c in fs.end_effector_poses["primary"].position]
except Exception as exc:
    result["final_state_error"] = str(exc)[:200]
result["stage"] = "done"
