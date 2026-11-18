# RouteStick v01_stub_tip_dips -- ASPIRE task worker, robomme_plus/task_8.
# Derived from v00_initial (phases 0-3 unchanged); see the v01 block below for the repair.
#
# Strategy (hypothesis from the released recordings, Stage 0):
#   The conditioning video shows the robot's stick travelling between white
#   bullseye targets laid out in a line between tall pillars, passing each pillar
#   on a particular side.  The goal asks to reproduce the same drawn path.  The
#   video retains the recorded robot state at every sampled timestamp, so the
#   demonstrated end-effector path is available directly from this episode's own
#   input.  The execution layout can be translated/rotated relative to the video
#   layout, so the program:
#     1. reads the video's per-frame end-effector poses and joints;
#     2. detects the target discs (whitish, table-height blobs) in the video's first
#        front frame and in the live front frame, using each frame's own depth and
#        calibration (projection convention chosen by a table-planarity test);
#     3. fits a planar rigid transform video->live from the ordered targets
#        (plus a small per-target residual correction along the line);
#     4. maps every kept video end-effector pose through that transform, solves IK
#        (seeded for continuity), and tracks the result with one-tick joint
#        commands at the video's own timing (compressed if it would exceed the
#        episode's max_steps).
#   If the transform is ~identity the recorded joints are replayed directly.
#   Every perception failure falls back to identity replay so control always runs.
#
# Nothing task-dependent is hardcoded: targets, layout and path come from this
# episode's conditioning video and live observation.
import math

result = {"version": "v00_initial", "errors": [], "phases": {}}
STATE = {"ended": False, "ticks": 0, "first_reply": None, "end_reason": None}


def note(where, err):
    result["errors"].append(where + ": " + str(err)[:400])


def rnd(x, n=4):
    try:
        return round(float(x), n)
    except Exception:
        return None


def vec(x):
    out = []
    for i in range(len(x)):
        out.append(float(x[i]))
    return out


def rowlist(row):
    try:
        return row.tolist()
    except Exception:
        out = []
        for i in range(len(row)):
            out.append(row[i])
        return out


def quat_to_mat(q):
    w = float(q[0]); x = float(q[1]); y = float(q[2]); z = float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    w = w / n; x = x / n; y = y / n; z = z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def quat_mul(a, b):
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def quat_angle(a, b):
    d = abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
    if d > 1.0:
        d = 1.0
    return 2.0 * math.acos(d)


def median(vals):
    s = sorted(vals)
    if len(s) == 0:
        return None
    return s[len(s) // 2]


def cam_params(cam):
    K = cam.intrinsics
    fx = float(K[0][0]); fy = float(K[1][1]); cx = float(K[0][2]); cy = float(K[1][2])
    pose = cam.camera_pose
    R = quat_to_mat(vec(pose.quaternion_wxyz))
    t = vec(pose.position)
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "R": R, "t": t}


def project(u, v, d, cp, conv):
    xc = (u - cp["cx"]) * d / cp["fx"]
    yc = (v - cp["cy"]) * d / cp["fy"]
    zc = d
    if conv == 0:      # OpenCV optical (x right, y down, z forward)
        c0, c1, c2 = xc, yc, zc
    elif conv == 1:    # OpenGL (x right, y up, z backward)
        c0, c1, c2 = xc, -yc, -zc
    else:              # x forward, y left, z up (ROS / ManiSkill body)
        c0, c1, c2 = zc, -xc, -yc
    R = cp["R"]; t = cp["t"]
    return [R[0][0] * c0 + R[0][1] * c1 + R[0][2] * c2 + t[0],
            R[1][0] * c0 + R[1][1] * c1 + R[1][2] * c2 + t[1],
            R[2][0] * c0 + R[2][1] * c1 + R[2][2] * c2 + t[2]]


def scan_pixels(cam):
    """Collect whitish (low-saturation bright) pixels and a wood subsample."""
    rgb = cam.rgb
    dep = cam.depth_m
    H = len(rgb)
    W = len(rgb[0])
    white = []
    wood = []
    for v in range(H):
        row = rowlist(rgb[v])
        drow = rowlist(dep[v])
        for u in range(W):
            p = row[u]
            r = int(p[0]); g = int(p[1]); b = int(p[2])
            mn = min(r, g, b); mx = max(r, g, b)
            d = float(drow[u])
            if not (d > 0.01 and d < 10.0):
                continue
            if mn >= 140 and mx - mn <= 40:
                white.append((u, v, d))
            elif (u % 4 == 0) and (v % 4 == 0) and r > 90 and r > g and g > b and r - b > 40:
                wood.append((u, v, d))
    return {"H": H, "W": W, "white": white, "wood": wood}


def choose_convention(scan, cp):
    """Pick the camera-axis convention under which table pixels form a horizontal plane."""
    report = []
    best = None
    for conv in (0, 1, 2):
        zs = []
        for item in scan["wood"]:
            zs.append(project(item[0], item[1], item[2], cp, conv)[2])
        if len(zs) < 20:
            report.append({"conv": conv, "n": len(zs)})
            continue
        zmed = median(zs)
        dev = []
        for z in zs:
            dev.append(abs(z - zmed))
        mad = median(dev)
        report.append({"conv": conv, "n": len(zs), "z_med": rnd(zmed), "mad": rnd(mad, 5)})
        # the table must lie below the camera (rules out the mirrored solution)
        if abs(zmed) < 1.5 and zmed < cp["t"][2] - 0.05 and (best is None or mad < best[1]):
            best = (conv, mad, zmed)
    return best, report


def detect_targets(scan, cp, conv, table_z):
    W = scan["W"]
    cand = {}
    for item in scan["white"]:
        pt = project(item[0], item[1], item[2], cp, conv)
        if pt[2] > table_z - 0.012 and pt[2] < table_z + 0.03:
            cand[item[1] * W + item[0]] = pt
    seen = set()
    comps = []
    for key in cand:
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        members = []
        while len(stack) > 0:
            k = stack.pop()
            members.append(cand[k])
            ku = k % W; kv = k // W
            for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                nu = ku + du; nv = kv + dv
                if nu < 0 or nu >= W or nv < 0:
                    continue
                nk = nv * W + nu
                if nk in cand and nk not in seen:
                    seen.add(nk)
                    stack.append(nk)
        sx = 0.0; sy = 0.0; sz = 0.0
        xmin = 1e9; xmax = -1e9; ymin = 1e9; ymax = -1e9
        for m in members:
            sx += m[0]; sy += m[1]; sz += m[2]
            xmin = min(xmin, m[0]); xmax = max(xmax, m[0])
            ymin = min(ymin, m[1]); ymax = max(ymax, m[1])
        n = len(members)
        comps.append({"c": [sx / n, sy / n, sz / n], "n": n,
                      "ext": max(xmax - xmin, ymax - ymin)})
    # merge fragments of the same disc (rings split the disc into components)
    merged = []
    for c in sorted(comps, key=comp_size, reverse=True):
        hit = None
        for m in merged:
            dx = m["c"][0] - c["c"][0]; dy = m["c"][1] - c["c"][1]
            if math.sqrt(dx * dx + dy * dy) < 0.02:
                hit = m
                break
        if hit is None:
            merged.append({"c": list(c["c"]), "n": c["n"], "ext": c["ext"]})
        else:
            tot = hit["n"] + c["n"]
            for i in range(3):
                hit["c"][i] = (hit["c"][i] * hit["n"] + c["c"][i] * c["n"]) / tot
            hit["n"] = tot
            hit["ext"] = max(hit["ext"], c["ext"])
    maxn = 0
    for m in merged:
        maxn = max(maxn, m["n"])
    keep = []
    for m in merged:
        if m["n"] >= max(4, 0.2 * maxn) and m["ext"] < 0.08:
            keep.append(m)
    return keep, merged


def comp_size(c):
    return c["n"]


def line_axis(points):
    n = len(points)
    mx = 0.0; my = 0.0
    for p in points:
        mx += p[0]; my += p[1]
    mx /= n; my /= n
    sxx = 0.0; syy = 0.0; sxy = 0.0
    for p in points:
        dx = p[0] - mx; dy = p[1] - my
        sxx += dx * dx; syy += dy * dy; sxy += dx * dy
    ang = 0.5 * math.atan2(2 * sxy, sxx - syy)
    return [math.cos(ang), math.sin(ang)], [mx, my]


def sort_along(points, axis, center):
    keyed = []
    for i in range(len(points)):
        p = points[i]
        a = (p[0] - center[0]) * axis[0] + (p[1] - center[1]) * axis[1]
        keyed.append((a, i))
    keyed = sorted(keyed)
    out = []
    for item in keyed:
        out.append(points[item[1]])
    return out


def procrustes(src, dst):
    n = len(src)
    sx = 0.0; sy = 0.0; dx = 0.0; dy = 0.0
    for i in range(n):
        sx += src[i][0]; sy += src[i][1]; dx += dst[i][0]; dy += dst[i][1]
    sx /= n; sy /= n; dx /= n; dy /= n
    num = 0.0; den = 0.0
    for i in range(n):
        ax = src[i][0] - sx; ay = src[i][1] - sy
        bx = dst[i][0] - dx; by = dst[i][1] - dy
        num += ax * by - ay * bx
        den += ax * bx + ay * by
    th = math.atan2(num, den) if n >= 2 else 0.0
    c = math.cos(th); s = math.sin(th)
    tx = dx - (c * sx - s * sy)
    ty = dy - (s * sx + c * sy)
    res = []
    tot = 0.0
    for i in range(n):
        px = c * src[i][0] - s * src[i][1] + tx
        py = s * src[i][0] + c * src[i][1] + ty
        rx = dst[i][0] - px; ry = dst[i][1] - py
        res.append([rx, ry])
        tot += rx * rx + ry * ry
    rms = math.sqrt(tot / max(1, n))
    return {"theta": th, "tx": tx, "ty": ty, "res": res, "rms": rms}


def apply_xy(T, x, y):
    c = math.cos(T["theta"]); s = math.sin(T["theta"])
    return c * x - s * y + T["tx"], s * x + c * y + T["ty"]


def residual_at(T, x, y):
    """Piecewise-linear residual correction along the live target line."""
    pts = T.get("res_pts")
    if not pts:
        return 0.0, 0.0
    ax = T["res_axis"]; cen = T["res_center"]
    a = (x - cen[0]) * ax[0] + (y - cen[1]) * ax[1]
    if a <= pts[0][0]:
        return pts[0][1], pts[0][2]
    if a >= pts[-1][0]:
        return pts[-1][1], pts[-1][2]
    for i in range(len(pts) - 1):
        a0 = pts[i][0]; a1 = pts[i + 1][0]
        if a >= a0 and a <= a1:
            f = (a - a0) / max(1e-6, a1 - a0)
            return (pts[i][1] * (1 - f) + pts[i + 1][1] * f,
                    pts[i][2] * (1 - f) + pts[i + 1][2] * f)
    return 0.0, 0.0


def reply_flags(reply):
    term = False; trunc = False
    try:
        term = bool(reply.terminated)
    except Exception:
        pass
    try:
        trunc = bool(reply.truncated)
    except Exception:
        pass
    return term, trunc


def do_step(target, grip, embodiment):
    if STATE["ended"]:
        return False
    try:
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=list(target),
            gripper_position=grip, embodiment=embodiment)}))
    except Exception as e:
        note("step@" + str(STATE["ticks"]), e)
        STATE["ended"] = True
        STATE["end_reason"] = "step_exception"
        return False
    STATE["ticks"] += 1
    if STATE["first_reply"] is None:
        STATE["first_reply"] = str(reply)[:600]
    term, trunc = reply_flags(reply)
    if term or trunc:
        STATE["ended"] = True
        STATE["end_reason"] = "terminated" if term else "truncated"
        result["terminal_at_tick"] = STATE["ticks"]
        result["terminal_flags"] = {"terminated": term, "truncated": trunc}
        return False
    try:
        if reply.ok is False:
            result.setdefault("step_not_ok", []).append([STATE["ticks"], str(reply.error)[:200]])
    except Exception:
        pass
    return True


# ---------------------------------------------------------------- phase 0: context
ctx = None
state0 = None
embodiment = "robomme_plus"
base_frame = "robot_base"
grip0 = 1.0
hz = 20.0
max_steps = 1000
try:
    ctx = get_task_context()
    result["goal"] = str(ctx.language)
    result["task_name"] = str(ctx.task_name)
except Exception as e:
    note("context", e)
try:
    meta = robomme_plus.get_task_metadata()
    md = {}
    for k in meta:
        md[str(k)] = str(meta[k])[:300]
    result["task_metadata"] = md
    try:
        max_steps = int(meta["max_steps"])
    except Exception as e:
        note("max_steps", e)
    try:
        hz = float(meta["control_hz"])
    except Exception:
        try:
            hz = float(meta["control_frequency"])
        except Exception:
            pass
except Exception as e:
    note("task_metadata", e)
try:
    cmeta = robomme_plus.get_controller_metadata()
    cm = {}
    for k in cmeta:
        cm[str(k)] = str(cmeta[k])[:300]
    result["controller_metadata"] = cm
except Exception as e:
    note("controller_metadata", e)
try:
    state0 = get_robot_state()
    embodiment = state0.embodiment
    base_frame = state0.base_frame
    try:
        grip0 = float(state0.gripper_positions["primary"])
    except Exception:
        grip0 = 1.0
    ee0 = state0.end_effector_poses["primary"]
    result["live_start"] = {
        "joints": [rnd(x) for x in vec(state0.joint_positions["primary"])],
        "ee_pos": [rnd(x) for x in vec(ee0.position)],
        "ee_quat": [rnd(x) for x in vec(ee0.quaternion_wxyz)],
        "gripper": rnd(grip0), "embodiment": str(embodiment),
        "base_frame": str(base_frame), "timestamp_s": rnd(state0.timestamp_s)}
except Exception as e:
    note("robot_state", e)
result["max_steps"] = max_steps
result["hz"] = hz

live_obs = None
try:
    live_obs = get_observation()
    cams = []
    for name in live_obs.cameras:
        cams.append(str(name))
    result["cameras"] = cams
    fc = live_obs.cameras["front"]
    result["live_front_cam"] = {
        "shape": [len(fc.rgb), len(fc.rgb[0])],
        "pose_pos": [rnd(x) for x in vec(fc.camera_pose.position)],
        "pose_quat": [rnd(x) for x in vec(fc.camera_pose.quaternion_wxyz)],
        "pose_frame": str(fc.camera_pose.frame),
        "K": [[rnd(fc.intrinsics[i][j], 3) for j in range(3)] for i in range(3)]}
except Exception as e:
    note("observation", e)

# initial-scene segmentation probe (instrumentation only; leaves an overlay)
try:
    seg = segment_text("front", "white round target disc")
    info = {"ok": bool(seg.ok), "n": 0, "scores": [], "boxes": []}
    if seg.ok:
        info["n"] = len(seg.segmentations)
        for cnd in seg.segmentations[:6]:
            info["scores"].append(rnd(cnd.score, 3))
            try:
                info["boxes"].append([rnd(x, 1) for x in cnd.box_xyxy])
            except Exception:
                pass
    result["initial_segment_targets"] = info
except Exception as e:
    note("initial_segment", e)

# ---------------------------------------------------------------- phase 1: video
vt = []; vp = []; vq = []; vj = []
video = []
try:
    raw_video = robomme_plus.get_conditioning_video()
    video = []
    for fr in raw_video:
        video.append(fr)
    nvid = len(video)
    result["video_frames"] = nvid
    for i in range(nvid):
        fr = video[i]
        try:
            rs = fr.robot_state
            pose = rs.end_effector_poses["primary"]
            vt.append(float(fr.timestamp_s))
            vp.append(vec(pose.position))
            vq.append(vec(pose.quaternion_wxyz))
            vj.append(vec(rs.joint_positions["primary"]))
        except Exception as e:
            if len(result["errors"]) < 30:
                note("video_frame_" + str(i), e)
    if len(vt) > 0:
        result["video_t_first_last"] = [rnd(vt[0], 3), rnd(vt[-1], 3)]
        zs = [p[2] for p in vp]
        result["video_ee_z_min_max"] = [rnd(min(zs)), rnd(max(zs))]
        result["video_j0"] = [rnd(x) for x in vj[0]]
        result["video_ee0"] = [rnd(x) for x in vp[0]]
        result["video_q0"] = [rnd(x) for x in vq[0]]
        comp = []
        for i in range(0, len(vp), 2):
            comp.append([rnd(vt[i], 2), rnd(vp[i][0], 3), rnd(vp[i][1], 3), rnd(vp[i][2], 3)])
        result["video_ee_path"] = comp
        try:
            vf = video[0].cameras["front"]
            result["video_front_cam"] = {
                "pose_pos": [rnd(x) for x in vec(vf.camera_pose.position)],
                "pose_quat": [rnd(x) for x in vec(vf.camera_pose.quaternion_wxyz)],
                "pose_frame": str(vf.camera_pose.frame)}
        except Exception as e:
            note("video_cam", e)
except Exception as e:
    note("conditioning_video", e)
result["video_states"] = len(vt)

# ---------------------------------------------------------------- phase 2: targets
T = {"theta": 0.0, "tx": 0.0, "ty": 0.0, "dz": 0.0, "mode": "identity_fallback"}
vid_targets = []
live_targets = []
conv_used = None
vid_table_z = None
live_table_z = None
try:
    if live_obs is not None:
        lcam = live_obs.cameras["front"]
        lcp = cam_params(lcam)
        lscan = scan_pixels(lcam)
        best, rep = choose_convention(lscan, lcp)
        result["live_conv_report"] = rep
        result["live_scan_counts"] = [len(lscan["white"]), len(lscan["wood"])]
        # cross-check with the public projection on one table pixel
        try:
            if len(lscan["wood"]) > 0:
                pick = lscan["wood"][len(lscan["wood"]) // 2]
                mask = []
                for v in range(lscan["H"]):
                    mrow = [False] * lscan["W"]
                    if v == pick[1]:
                        mrow[pick[0]] = True
                    mask.append(mrow)
                pc = mask_to_point_cloud(mask, "front", target_frame=base_frame)
                pts = pc.points
                ref = vec(pts[0])
                chk = {"pixel": [pick[0], pick[1]], "api": [rnd(x) for x in ref]}
                bestd = None
                for conv in (0, 1, 2):
                    mine = project(pick[0], pick[1], pick[2], lcp, conv)
                    dd = math.sqrt((mine[0] - ref[0]) ** 2 + (mine[1] - ref[1]) ** 2 + (mine[2] - ref[2]) ** 2)
                    chk["conv" + str(conv)] = rnd(dd)
                    if bestd is None or dd < bestd[1]:
                        bestd = (conv, dd)
                result["projection_check"] = chk
                if bestd is not None and bestd[1] < 0.02:
                    if best is None or best[0] != bestd[0]:
                        note("conv", "planarity chose " + str(best) + " api chose " + str(bestd))
                    zs2 = []
                    for item in lscan["wood"]:
                        zs2.append(project(item[0], item[1], item[2], lcp, bestd[0])[2])
                    best = (bestd[0], 0.0, median(zs2))
        except Exception as e:
            note("projection_check", e)
        if best is not None:
            conv_used = best[0]
            live_table_z = best[2]
            keep, allc = detect_targets(lscan, lcp, conv_used, live_table_z)
            live_targets = [k["c"] for k in keep]
            result["live_targets"] = [[rnd(c[0]), rnd(c[1]), rnd(c[2]), k["n"]] for c, k in zip(live_targets, keep)]
            result["live_components"] = [[rnd(m["c"][0]), rnd(m["c"][1]), rnd(m["c"][2]), m["n"], rnd(m["ext"])] for m in allc[:15]]
    result["conv_used"] = conv_used
    result["live_table_z"] = rnd(live_table_z)
except Exception as e:
    note("live_targets", e)

try:
    if conv_used is not None and len(video) > 0:
        for fidx in (0, len(video) - 1):
            vcam = video[fidx].cameras["front"]
            vcp = cam_params(vcam)
            vscan = scan_pixels(vcam)
            zs = []
            for item in vscan["wood"]:
                zs.append(project(item[0], item[1], item[2], vcp, conv_used)[2])
            vtz = median(zs)
            keep, allc = detect_targets(vscan, vcp, conv_used, vtz)
            result["video_targets_frame" + str(fidx)] = [[rnd(k["c"][0]), rnd(k["c"][1]), rnd(k["c"][2]), k["n"]] for k in keep]
            if len(keep) >= 2:
                vid_targets = [k["c"] for k in keep]
                vid_table_z = vtz
                result["video_target_frame_used"] = fidx
                break
    result["video_table_z"] = rnd(vid_table_z)
except Exception as e:
    note("video_targets", e)

# ---------------------------------------------------------------- phase 3: transform
try:
    if len(vid_targets) >= 2 and len(live_targets) >= 2:
        vax, vcen = line_axis(vid_targets)
        lax, lcen = line_axis(live_targets)
        if vax[0] * lax[0] + vax[1] * lax[1] < 0:
            lax = [-lax[0], -lax[1]]
        vs = sort_along(vid_targets, vax, vcen)
        ls = sort_along(live_targets, lax, lcen)
        cand = None
        if len(vs) == len(ls):
            fwd = procrustes(vs, ls)
            rev = procrustes(vs, list(reversed(ls)))
            result["procrustes_fwd_rev_rms"] = [rnd(fwd["rms"]), rnd(rev["rms"])]
            cand = fwd
            if rev["rms"] < fwd["rms"] * 0.5:
                cand = rev
                ls = list(reversed(ls))
                note("transform", "reversed order matched better")
            matched_v = vs; matched_l = ls
        else:
            # unequal counts: iterative nearest-neighbour rigid fit from centroid alignment
            cur = {"theta": 0.0, "tx": lcen[0] - vcen[0], "ty": lcen[1] - vcen[1]}
            for it in range(6):
                mv = []; ml = []
                for p in vs:
                    px, py = apply_xy(cur, p[0], p[1])
                    bestl = None
                    for q in ls:
                        d = math.sqrt((q[0] - px) ** 2 + (q[1] - py) ** 2)
                        if bestl is None or d < bestl[0]:
                            bestl = (d, q)
                    if bestl is not None and bestl[0] < 0.05:
                        mv.append(p); ml.append(bestl[1])
                if len(mv) < 2:
                    break
                cur = procrustes(mv, ml)
            cand = cur if "rms" in cur else None
            matched_v = mv if cand is not None else []
            matched_l = ml if cand is not None else []
            note("transform", "unequal target counts video=" + str(len(vs)) + " live=" + str(len(ls)))
        if cand is not None and cand["rms"] < 0.05:
            T = {"theta": cand["theta"], "tx": cand["tx"], "ty": cand["ty"], "dz": 0.0,
                 "mode": "rigid", "rms": cand["rms"]}
            # residual correction table along the live line
            maxres = 0.0
            pts = []
            for i in range(len(matched_l)):
                r = cand["res"][i]
                maxres = max(maxres, math.sqrt(r[0] * r[0] + r[1] * r[1]))
                a = (matched_l[i][0] - lcen[0]) * lax[0] + (matched_l[i][1] - lcen[1]) * lax[1]
                pts.append((a, r[0], r[1]))
            if maxres < 0.03:
                T["res_pts"] = sorted(pts)
                T["res_axis"] = lax
                T["res_center"] = lcen
            T["max_res"] = maxres
            if vid_table_z is not None and live_table_z is not None:
                dz = live_table_z - vid_table_z
                if abs(dz) < 0.02:
                    T["dz"] = dz
                result["table_dz"] = rnd(dz)
        elif cand is not None:
            T = {"theta": 0.0, "tx": lcen[0] - vcen[0], "ty": lcen[1] - vcen[1], "dz": 0.0,
                 "mode": "centroid_translation", "rms": cand["rms"]}
            note("transform", "rigid rms too large; centroid translation only")
except Exception as e:
    note("transform", e)
    T = {"theta": 0.0, "tx": 0.0, "ty": 0.0, "dz": 0.0, "mode": "identity_fallback"}

shift = math.sqrt(T["tx"] ** 2 + T["ty"] ** 2)
# translation of the video target centroid (more meaningful than tx,ty about the origin)
cen_shift = None
try:
    if len(vid_targets) >= 2:
        cx0 = sum([p[0] for p in vid_targets]) / len(vid_targets)
        cy0 = sum([p[1] for p in vid_targets]) / len(vid_targets)
        mx0, my0 = apply_xy(T, cx0, cy0)
        cen_shift = math.sqrt((mx0 - cx0) ** 2 + (my0 - cy0) ** 2)
except Exception as e:
    note("cen_shift", e)
identity = (cen_shift is not None and cen_shift < 0.003 and abs(T["theta"]) < 0.006
            and T.get("max_res", 0.0) < 0.005) or T["mode"] == "identity_fallback"
result["transform"] = {"mode": T["mode"], "theta_deg": rnd(T["theta"] * 180.0 / math.pi, 3),
                       "tx": rnd(T["tx"]), "ty": rnd(T["ty"]), "dz": rnd(T["dz"]),
                       "rms": rnd(T.get("rms")), "max_res": rnd(T.get("max_res")),
                       "centroid_shift": rnd(cen_shift), "identity_replay": identity}


def map_pose(p, q):
    x, y = apply_xy(T, p[0], p[1])
    rx, ry = residual_at(T, x, y)
    qz = [math.cos(T["theta"] / 2.0), 0.0, 0.0, math.sin(T["theta"] / 2.0)]
    return [x + rx, y + ry, p[2] + T["dz"]], quat_mul(qz, q)



# =============================================================================
# v01 changes (trace-guided repair of v00, episode 1 run):
#   * v00 replayed the video EE height (z=0.12).  The video robot's stick reaches
#     the targets from that height, but the live tool is a short stub whose end
#     sits ~0.010 m above the EE reference (front-camera frames of the v00 run),
#     so the live tip stayed ~0.11 m above the targets and nothing was touched.
#   * v01 keeps the video's xy route and the v00 video->live mapping, but runs
#     the low (drawing) part of the route with the stub tip just above the disc
#     tops, dips onto each target the video passes over (contact-sensed from the
#     observed EE height), and calibrates the tool offset from the first contact.
#   * With the tip that low, the hand body is below the pillar tops.  The hand's
#     long axis is held perpendicular to the target line for the whole low route
#     (fits between pillars at the targets), and route points are pushed away from
#     the line wherever the hand would overlap a detected pillar, keeping the
#     video's side of every pillar.  Hover/rotation happens above the pillar tops.
# Robot-geometry constants below (tool offset, hand footprint) are properties of
# the robot measured from this task's own v00 run, not per-episode scene data.
# =============================================================================
result["version"] = "v01_stub_tip_dips"
TOOL_L_NOM = 0.010        # stub end height above the EE reference (v00 run frames)
TOOL_L_TOL = 0.008        # accepted calibration window around the nominal
HAND_HALF_LONG = 0.105    # panda hand half length along local y
HAND_HALF_THICK = 0.035   # panda hand half thickness along local x
HAND_BODY_ABOVE = 0.035   # hand body bottom above the EE reference
CLEAR_MARGIN = 0.012
TRAVEL_ABOVE = 0.006      # tip clearance above the disc tops while travelling
PRESS = 0.008             # deepest dip below the expected contact height
DIP_STEP = 0.0012         # m per tick while dipping
SLOW = 1.5                # time scale applied to the video timing
VMAX = 0.10               # m/s cap for inserted / pushed motion
TOUCH_R = 0.025           # video pass within this xy distance of a target = touch
LOW_BAND = 0.01           # video EE z within this of its minimum = drawing
TOOL = {"L": TOOL_L_NOM, "contacts": []}


def wrap_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


def qz_of(a):
    return [math.cos(a / 2.0), 0.0, 0.0, math.sin(a / 2.0)]


def hand_axis_angle(q):
    R = quat_to_mat(q)
    return math.atan2(R[1][1], R[0][1])


def slerp(a, b, f):
    d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    bb = list(b)
    if d < 0:
        bb = [-x for x in b]
        d = -d
    if d > 0.9995:
        out = [a[i] + (bb[i] - a[i]) * f for i in range(4)]
    else:
        th = math.acos(d)
        s0 = math.sin((1 - f) * th) / math.sin(th)
        s1 = math.sin(f * th) / math.sin(th)
        out = [a[i] * s0 + bb[i] * s1 for i in range(4)]
    n = math.sqrt(sum([x * x for x in out]))
    return [x / n for x in out]


def dist2(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ---------------------------------------------------------------- phase 3b: line frame + pillars
LINE = None
pillars = []
disc_top = None
try:
    if len(live_targets) >= 2:
        lax2, lcen2 = line_axis(live_targets)
        LINE = {"u": lax2, "n": [-lax2[1], lax2[0]], "o": lcen2}
        tg = []
        for c in live_targets:
            s = (c[0] - lcen2[0]) * lax2[0] + (c[1] - lcen2[1]) * lax2[1]
            d = (c[0] - lcen2[0]) * LINE["n"][0] + (c[1] - lcen2[1]) * LINE["n"][1]
            tg.append([s, d, c[0], c[1], c[2]])
        LINE["targets"] = sorted(tg)
        disc_top = median([c[2] for c in live_targets])
except Exception as e:
    note("line_frame", e)
    LINE = None


def to_sd(x, y):
    o = LINE["o"]; u = LINE["u"]; n = LINE["n"]
    return ((x - o[0]) * u[0] + (y - o[1]) * u[1], (x - o[0]) * n[0] + (y - o[1]) * n[1])


def from_sd(s, d):
    o = LINE["o"]; u = LINE["u"]; n = LINE["n"]
    return [o[0] + s * u[0] + d * n[0], o[1] + s * u[1] + d * n[1]]


def comp_s(p):
    return p[0]


try:
    if LINE is not None and live_obs is not None and conv_used is not None:
        lcam = live_obs.cameras["front"]
        lcp = cam_params(lcam)
        dep = lcam.depth_m
        H = len(dep); W = len(dep[0])
        smin = LINE["targets"][0][0]; smax = LINE["targets"][-1][0]
        tall = []
        for v in range(H):
            drow = rowlist(dep[v])
            for u in range(W):
                dd = float(drow[u])
                if not (dd > 0.01 and dd < 10.0):
                    continue
                p = project(u, v, dd, lcp, conv_used)
                if p[2] < live_table_z + 0.04 or p[2] > live_table_z + 0.4:
                    continue
                s, d = to_sd(p[0], p[1])
                if abs(d) > 0.08 or s < smin - 0.03 or s > smax + 0.03:
                    continue
                tall.append((s, d, p[0], p[1], p[2]))
        tall = sorted(tall, key=comp_s)
        clusters = []
        cur = []
        for p in tall:
            if len(cur) > 0 and p[0] - cur[-1][0] > 0.015:
                clusters.append(cur)
                cur = []
            cur.append(p)
        if len(cur) > 0:
            clusters.append(cur)
        for cl in clusters:
            if len(cl) < 6:
                continue
            zs = sorted([p[4] for p in cl])
            ztop = zs[int(0.95 * (len(zs) - 1))]
            top = [p for p in cl if p[4] > ztop - 0.004]
            if len(top) < 3:
                top = cl
            ss = sorted([p[0] for p in top]); dl = sorted([p[1] for p in top])
            sc = median(ss)
            dc = median(dl)
            lo = int(0.05 * (len(ss) - 1)); hi = int(0.95 * (len(ss) - 1))
            hs = min(0.04, max(0.010, (ss[hi] - ss[lo]) / 2.0 + 0.003))
            hd = min(0.06, max(0.010, (dl[hi] - dl[lo]) / 2.0 + 0.003))
            pillars.append({"s": sc, "d": dc, "hs": hs, "hd": hd, "top": ztop, "n": len(cl), "assumed": False})
        # every gap between consecutive targets must have an obstacle model
        tl = LINE["targets"]
        for i in range(len(tl) - 1):
            a = tl[i][0]; b = tl[i + 1][0]
            found = False
            for P in pillars:
                if P["s"] > a + 0.02 and P["s"] < b - 0.02:
                    found = True
            if not found:
                pillars.append({"s": 0.5 * (a + b), "d": 0.5 * (tl[i][1] + tl[i + 1][1]), "hs": 0.02, "hd": 0.04,
                                "top": live_table_z + 0.25, "n": 0, "assumed": True})
        for P in pillars:
            near = 1.0
            for t in tl:
                near = min(near, abs(t[0] - P["s"]))
            P["near_target_ds"] = near
            margin = CLEAR_MARGIN
            if HAND_HALF_THICK + P["hs"] + margin > near - 0.006:
                margin = max(0.003, near - 0.006 - HAND_HALF_THICK - P["hs"])
                note("pillar_fit", "reduced s-margin to " + str(rnd(margin, 4)))
            S1 = HAND_HALF_THICK + P["hs"] + margin
            P["S1"] = S1
            P["S2"] = max(S1 + 0.002, near - 0.003)
            if P["S2"] >= near:
                note("pillar_fit", "hand cannot fit between pillar and target at s=" + str(rnd(P["s"], 3)))
            P["R"] = HAND_HALF_LONG + P["hd"] + CLEAR_MARGIN
        res_p = []
        for P in pillars:
            xy = from_sd(P["s"], P["d"])
            res_p.append([rnd(xy[0], 3), rnd(xy[1], 3), rnd(P["hs"], 3), rnd(P["hd"], 3), rnd(P["top"], 3), P["n"], P["assumed"],
                          rnd(P["S1"], 3), rnd(P["S2"], 3), rnd(P["near_target_ds"], 3)])
        result["pillars"] = res_p
except Exception as e:
    note("pillars", e)

# ---------------------------------------------------------------- phase 4: low route
plan = {"ok": False}
try:
    n = len(vt)
    if n > 0 and LINE is not None and disc_top is not None:
        zall = [p[2] for p in vp]
        zmin = min(zall)
        low = [vp[k][2] <= zmin + LOW_BAND for k in range(n)]
        runs = []
        k = 0
        while k < n:
            if low[k]:
                j = k
                while j + 1 < n and low[j + 1]:
                    j += 1
                if j - k >= 2:
                    runs.append([k, j])
                k = j + 1
            else:
                k += 1
        result["video_low_runs"] = [[a, b, rnd(vt[a], 2), rnd(vt[b], 2)] for a, b in runs]
        routes = []
        for a, b in runs:
            # mapped xy and time of every video sample in the low run (thinned to >=3 mm)
            pts = []
            for k in range(a, b + 1):
                mp, mq = map_pose(vp[k], vq[k])
                if len(pts) == 0 or dist2(mp, pts[-1]) >= 0.003 or k == b:
                    pts.append([mp[0], mp[1], vt[k]])
            # touches: passes within TOUCH_R of a live target
            ctr = [[t[2], t[3]] for t in LINE["targets"]]
            touches = []
            i = 0
            while i < len(pts):
                best = None
                for ti in range(len(ctr)):
                    d = dist2(pts[i], ctr[ti])
                    if best is None or d < best[0]:
                        best = (d, ti)
                if best[0] < 0.04:
                    ti = best[1]
                    j = i
                    kmin = i; dmin = best[0]
                    while j < len(pts) and dist2(pts[j], ctr[ti]) < 0.04:
                        dj = dist2(pts[j], ctr[ti])
                        if dj < dmin:
                            dmin = dj; kmin = j
                        j += 1
                    if dmin < TOUCH_R:
                        touches.append([kmin, ti, dmin])
                    i = max(j, i + 1)
                else:
                    i += 1
            # snap each touched pass onto the target centre (cosine window, 5 cm of path)
            arc = [0.0]
            for i in range(1, len(pts)):
                arc.append(arc[-1] + dist2(pts[i], pts[i - 1]))
            dx = [0.0] * len(pts); dy = [0.0] * len(pts)
            for kmin, ti, dmin in touches:
                ex = ctr[ti][0] - pts[kmin][0]; ey = ctr[ti][1] - pts[kmin][1]
                for i in range(len(pts)):
                    aa = abs(arc[i] - arc[kmin])
                    if aa < 0.05:
                        w = 0.5 * (1.0 + math.cos(math.pi * aa / 0.05))
                        dx[i] = dx[i] + w * ex
                        dy[i] = dy[i] + w * ey
            for i in range(len(pts)):
                pts[i][0] += dx[i]; pts[i][1] += dy[i]
            for kmin, ti, dmin in touches:
                pts[kmin][0] = ctr[ti][0]; pts[kmin][1] = ctr[ti][1]
            # push away from pillars (hand long axis perpendicular to the line)
            sd = [list(to_sd(p[0], p[1])) for p in pts]
            pushes = 0; max_push = 0.0
            for P in pillars:
                i = 0
                while i < len(sd):
                    if abs(sd[i][0] - P["s"]) < P["S2"]:
                        j = i
                        apex = i
                        while j < len(sd) and abs(sd[j][0] - P["s"]) < P["S2"]:
                            if abs(sd[j][0] - P["s"]) < abs(sd[apex][0] - P["s"]):
                                apex = j
                            j += 1
                        rel_a = sd[apex][1] - P["d"]
                        if abs(rel_a) < 0.004:
                            tot = 0.0
                            for m in range(i, j):
                                tot += sd[m][1] - P["d"]
                            rel_a = tot
                        side = 1.0 if rel_a >= 0 else -1.0
                        for m in range(i, j):
                            ds = abs(sd[m][0] - P["s"])
                            if ds <= P["S1"]:
                                req = P["R"]
                            else:
                                req = P["R"] * (P["S2"] - ds) / (P["S2"] - P["S1"])
                            rel = sd[m][1] - P["d"]
                            if rel * side < req:
                                newd = P["d"] + side * req
                                max_push = max(max_push, abs(newd - sd[m][1]))
                                sd[m][1] = newd
                                pushes += 1
                        i = j
                    else:
                        i += 1
            for i in range(len(pts)):
                xy = from_sd(sd[i][0], sd[i][1])
                pts[i][0] = xy[0]; pts[i][1] = xy[1]
            # densify to <= 8 mm spacing, carry dip markers
            dipset = {}
            for kmin, ti, dmin in touches:
                dipset[kmin] = ti
            dense = []
            for i in range(len(pts)):
                if i > 0:
                    L = dist2(pts[i], pts[i - 1])
                    m = int(L / 0.008)
                    for q in range(1, m + 1):
                        f = float(q) / float(m + 1)
                        dense.append({"x": pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * f,
                                      "y": pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * f,
                                      "t": pts[i - 1][2] + (pts[i][2] - pts[i - 1][2]) * f, "dip": None})
                dense.append({"x": pts[i][0], "y": pts[i][1], "t": pts[i][2],
                              "dip": dipset.get(i)})
            routes.append({"pts": dense, "touches": touches, "pushes": pushes, "max_push": max_push,
                           "end_high": b < n - 1 and max([vp[k][2] for k in range(b, n)]) > zmin + 0.05})
        # orientation: video orientation of the low part, re-yawed so the hand long axis is
        # perpendicular to the live target line
        k0 = runs[0][0] if len(runs) > 0 else 0
        mp0, q_base = map_pose(vp[k0], vq[k0])
        qvar = 0.0
        for a, b in runs:
            for k in range(a, b + 1):
                qvar = max(qvar, quat_angle(vq[k], vq[k0]))
        a0 = hand_axis_angle(q_base)
        an = math.atan2(LINE["n"][1], LINE["n"][0])
        psi1 = wrap_pi(an - a0)
        if psi1 > math.pi / 2:
            psi1 -= math.pi
        if psi1 <= -math.pi / 2:
            psi1 += math.pi
        psi2 = psi1 - math.pi if psi1 > 0 else psi1 + math.pi
        top_max = max([P["top"] for P in pillars]) if len(pillars) > 0 else live_table_z + 0.25
        plan = {"ok": len(routes) > 0 and sum([len(r["pts"]) for r in routes]) > 1,
                "routes": routes, "q_base": q_base, "psi": [psi1, psi2], "top_max": top_max}
        result["plan"] = {"n_routes": len(routes), "n_pts": [len(r["pts"]) for r in routes],
                          "touches": [[[t[1], rnd(t[2], 4)] for t in r["touches"]] for r in routes],
                          "pushes": [r["pushes"] for r in routes],
                          "max_push": [rnd(r["max_push"], 4) for r in routes],
                          "psi_deg": [rnd(psi1 * 180 / math.pi, 2), rnd(psi2 * 180 / math.pi, 2)],
                          "video_quat_var": rnd(qvar, 4), "top_max": rnd(top_max, 4),
                          "disc_top": rnd(disc_top, 4)}
        # hand-footprint clearance of the planned low route against the pillar models
        minclr = 1.0
        for r in routes:
            for p in r["pts"]:
                s, d = to_sd(p["x"], p["y"])
                for P in pillars:
                    cs = abs(s - P["s"]) - (HAND_HALF_THICK + P["hs"])
                    cd = abs(d - P["d"]) - (HAND_HALF_LONG + P["hd"])
                    minclr = min(minclr, max(cs, cd))
        result["plan"]["min_footprint_clearance"] = rnd(minclr, 4)
        comp = []
        for r in routes:
            for p in r["pts"][::3]:
                comp.append([rnd(p["x"], 3), rnd(p["y"], 3)])
        result["plan_xy"] = comp
except Exception as e:
    note("plan", e)
    plan = {"ok": False}

# ---------------------------------------------------------------- phase 5: execute
ik_stats = {"calls": 0, "ok": 0, "retry": 0, "fallback": 0, "max_jump": 0.0, "skipped": 0}
track = []
dips = []


def ik_solve(pos, quat, seed):
    ik_stats["calls"] += 1
    r = solve_ik(Pose(list(pos), list(quat), base_frame), seed_joints=list(seed))
    if not bool(r.ok):
        return None
    return vec(r.joint_positions)


def max_abs_diff(a, b):
    m = 0.0
    for i in range(min(len(a), len(b))):
        m = max(m, abs(a[i] - b[i]))
    return m


def ik_cont(pos, quat, seed, limit):
    """IK seeded for continuity; None if no solution close to the seed."""
    sol = None
    try:
        sol = ik_solve(pos, quat, seed)
    except Exception as e:
        if ik_stats["calls"] < 4:
            note("ik", e)
    if sol is not None and max_abs_diff(sol, seed) > limit:
        ik_stats["retry"] += 1
        sol = None
    if sol is None:
        return None
    ik_stats["ok"] += 1
    ik_stats["max_jump"] = max(ik_stats["max_jump"], max_abs_diff(sol, seed))
    return sol


def ee_now():
    rs = get_robot_state()
    return vec(rs.end_effector_poses["primary"].position)


EX = {"cmd": None}


def move_joints(target, ticks):
    cmd = EX["cmd"]
    nt = max(1, int(ticks))
    for tick in range(1, nt + 1):
        f = float(tick) / float(nt)
        q = [cmd[j] + (target[j] - cmd[j]) * f for j in range(len(target))]
        if not do_step(q, grip0, embodiment):
            break
    EX["cmd"] = list(target)


def z_travel():
    return disc_top + TRAVEL_ABOVE - TOOL["L"]


def ticks_for(dist, dt):
    a = int(round(dt * hz * SLOW_EFF[0]))
    b = int(math.ceil(dist / (VMAX / hz)))
    return max(1, a, b)


SLOW_EFF = [SLOW]


def do_dip(ti, xy, quat):
    """Settle over the target, dip until contact (observed EE z stalls), hold, rise."""
    info = {"target": ti}
    jt = EX["cmd"]
    # settle in xy
    for s in range(8):
        try:
            ep = ee_now()
        except Exception as e:
            note("dip_state", e)
            return info
        if dist2(ep, xy) < 0.004:
            break
        if not do_step(jt, grip0, embodiment):
            return info
    ztop = z_travel()
    zbot = disc_top - TOOL["L"] - PRESS
    jb = ik_cont([xy[0], xy[1], zbot], quat, jt, 0.4)
    if jb is None:
        zbot = disc_top - TOOL["L"] - 0.5 * PRESS
        jb = ik_cont([xy[0], xy[1], zbot], quat, jt, 0.4)
    if jb is None:
        info["ik_fail"] = True
        dips.append(info)
        return info
    N = max(4, int(math.ceil((ztop - zbot) / DIP_STEP)))
    errs = []
    contact = None
    zlast = None
    nstall = 0
    alpha = 1.0
    for i in range(1, N + 1):
        alpha = float(i) / float(N)
        q = [jt[j] + (jb[j] - jt[j]) * alpha for j in range(len(jt))]
        if not do_step(q, grip0, embodiment):
            break
        try:
            ep = ee_now()
        except Exception as e:
            note("dip_state", e)
            break
        zc = ztop + (zbot - ztop) * alpha
        errs.append(rnd(ep[2] - zc, 4))
        if zlast is not None and (ep[2] - zlast) > -0.0003:
            nstall += 1
        else:
            nstall = 0
        zlast = ep[2]
        # contact: the observed EE height stops following the commanded descent
        near_expected = ep[2] < disc_top - TOOL_L_NOM + TOOL_L_TOL + 0.004
        if i >= 4 and nstall >= 2 and errs[-1] > 0.003 and near_expected:
            contact = {"i": i, "alpha": rnd(alpha, 3), "z": rnd(ep[2], 4), "lag": errs[-1]}
            break
    info["errs"] = errs[:24]
    if STATE["ended"]:
        info["ended"] = True
        dips.append(info)
        return info
    if contact is not None:
        extra = 0.003 / max(1e-4, ztop - zbot)
        alpha = min(1.0, alpha + extra)
        Lm = disc_top - contact["z"]
        if abs(Lm - TOOL_L_NOM) <= TOOL_L_TOL:
            TOOL["contacts"].append(Lm)
            TOOL["L"] = median(TOOL["contacts"])
        contact["L_meas"] = rnd(Lm, 4)
    info["contact"] = contact
    info["N"] = N
    hold = [jt[j] + (jb[j] - jt[j]) * alpha for j in range(len(jt))]
    for i in range(5):
        if not do_step(hold, grip0, embodiment):
            break
    try:
        info["z_hold"] = rnd(ee_now()[2], 4)
    except Exception:
        pass
    EX["cmd"] = hold
    # rise back to the (possibly re-calibrated) travel height
    jt2 = ik_cont([xy[0], xy[1], z_travel()], quat, jt, 0.4)
    if jt2 is None:
        jt2 = jt
    move_joints(jt2, max(4, int(N * 0.7)))
    info["L_after"] = rnd(TOOL["L"], 4)
    dips.append(info)
    return info


try:
    if state0 is None:
        raise ValueError("no robot state")
    start_j = vec(state0.joint_positions["primary"])
    start_p = vec(state0.end_effector_poses["primary"].position)
    start_q = vec(state0.end_effector_poses["primary"].quaternion_wxyz)
    EX["cmd"] = list(start_j)
    if not plan.get("ok"):
        raise ValueError("no low-route plan")
    routes = plan["routes"]
    q_base = plan["q_base"]
    first = routes[0]["pts"][0]
    z_hover = max(plan["top_max"] - HAND_BODY_ABOVE + 0.06, z_travel() + 0.10)
    result["z_hover"] = rnd(z_hover, 4)
    # choose the yaw candidate by an IK sweep over the whole low route: the hand yaw is held
    # constant while the base joint turns along the line, so one candidate can run the last
    # joint into its limit (Panda q7 limit +-2.8973 rad) while the other stays well inside.
    choice = None
    sweep = []
    for psi in plan["psi"]:
        qd = quat_mul(qz_of(psi), q_base)
        nfail = 0
        margin = 9.0
        jump = 0.0
        sol = None
        try:
            sol = ik_solve([first["x"], first["y"], z_hover], qd, start_j)
        except Exception as e:
            note("ik_hover", e)
        if sol is None:
            nfail += 5
            seedc = list(start_j)
        else:
            margin = min(margin, 2.8973 - abs(sol[6]))
            jump = max_abs_diff(sol, start_j)
            seedc = sol
        for r in routes:
            pts = r["pts"]
            for i in range(0, len(pts), 6):
                try:
                    sl = ik_solve([pts[i]["x"], pts[i]["y"], z_travel()], qd, seedc)
                except Exception:
                    sl = None
                if sl is None or max_abs_diff(sl, seedc) > 0.8:
                    nfail += 1
                    continue
                margin = min(margin, 2.8973 - abs(sl[6]))
                seedc = sl
        score = (nfail, -round(min(margin, 0.6), 2), jump)
        sweep.append([rnd(psi * 180 / math.pi, 2), nfail, rnd(margin, 3), rnd(jump, 3)])
        if choice is None or score < choice[0]:
            choice = (score, psi, qd)
    result["yaw_sweep"] = sweep
    if choice is None:
        psi = plan["psi"][0]
        qd = quat_mul(qz_of(psi), q_base)
        note("yaw", "no yaw candidate evaluated")
    else:
        psi = choice[1]; qd = choice[2]
    result["yaw_used_deg"] = rnd(psi * 180 / math.pi, 2)
    # estimated budget; slow factor reduced if needed
    est_draw = 0.0
    ndip = 0
    for r in routes:
        pts = r["pts"]
        for i in range(1, len(pts)):
            est_draw += max((pts[i]["t"] - pts[i - 1]["t"]) * hz, dist2([pts[i]["x"], pts[i]["y"]], [pts[i - 1]["x"], pts[i - 1]["y"]]) / (VMAX / hz))
            if pts[i]["dip"] is not None:
                ndip += 1
        if pts[0]["dip"] is not None:
            ndip += 1
    fixed = 40.0 * ndip + 90.0 * len(routes) + 80.0
    avail = max_steps - 40 - fixed
    if est_draw * SLOW > avail and est_draw > 0:
        SLOW_EFF[0] = max(0.6, avail / est_draw)
        note("timing", "slow factor reduced to " + str(rnd(SLOW_EFF[0], 3)))
    result["timing_plan"] = {"est_draw_ticks_at_1x": rnd(est_draw, 1), "dips": ndip,
                             "slow": rnd(SLOW_EFF[0], 3), "fixed": fixed}
    seed = list(start_j)
    cur_p = list(start_p); cur_q = list(start_q)
    for ri in range(len(routes)):
        if STATE["ended"]:
            break
        pts = routes[ri]["pts"]
        p0 = pts[0]
        # 1. move (above the pillar tops) to hover over the run start, rotating on the way
        hp = [p0["x"], p0["y"], z_hover]
        if ri == 0:
            L = math.sqrt(sum([(hp[i] - cur_p[i]) ** 2 for i in range(3)]))
            m = max(6, int(L / 0.02))
            for q in range(1, m + 1):
                f = float(q) / float(m)
                pos = [cur_p[i] + (hp[i] - cur_p[i]) * f for i in range(3)]
                if pos[2] < z_hover:
                    pos[2] = z_hover
                qq = slerp(cur_q, qd, f)
                sol = ik_cont(pos, qq, seed, 0.6)
                if sol is None:
                    ik_stats["skipped"] += 1
                    continue
                move_joints(sol, ticks_for(L / m, 0.0) + 1)
                seed = sol
                if STATE["ended"]:
                    break
        else:
            # lift vertically, travel at hover height, then descend
            up = [cur_p[0], cur_p[1], z_hover]
            for target_p in (up, hp):
                L = dist2(cur_p, target_p) + abs(target_p[2] - cur_p[2])
                m = max(2, int(L / 0.02))
                base_p = list(cur_p)
                for q in range(1, m + 1):
                    f = float(q) / float(m)
                    pos = [base_p[i] + (target_p[i] - base_p[i]) * f for i in range(3)]
                    sol = ik_cont(pos, qd, seed, 0.6)
                    if sol is None:
                        ik_stats["skipped"] += 1
                        continue
                    move_joints(sol, ticks_for(L / m, 0.0) + 1)
                    seed = sol
                cur_p = list(target_p)
        cur_p = list(hp)
        # 2. vertical descent to travel height (slower over the last 3 cm)
        zt = z_travel()
        z = z_hover
        while z > zt + 1e-6 and not STATE["ended"]:
            stepz = 0.01 if z - zt > 0.03 else 0.005
            z = max(zt, z - stepz)
            sol = ik_cont([p0["x"], p0["y"], z], qd, seed, 0.6)
            if sol is None:
                ik_stats["skipped"] += 1
                continue
            move_joints(sol, max(2, int(math.ceil(stepz / (0.05 / hz)))))
            seed = sol
        cur_p = [p0["x"], p0["y"], zt]
        if p0["dip"] is not None and not STATE["ended"]:
            do_dip(p0["dip"], [p0["x"], p0["y"]], qd)
            seed = list(EX["cmd"])
        # 3. low route at the video's (scaled) timing
        for i in range(1, len(pts)):
            if STATE["ended"]:
                break
            p = pts[i]
            pos = [p["x"], p["y"], z_travel()]
            sol = ik_cont(pos, qd, seed, 0.5)
            if sol is None:
                ik_stats["skipped"] += 1
                continue
            dist = dist2(pos, cur_p)
            move_joints(sol, ticks_for(dist, p["t"] - pts[i - 1]["t"]))
            seed = sol
            cur_p = pos
            if p["dip"] is not None and not STATE["ended"]:
                do_dip(p["dip"], [p["x"], p["y"]], qd)
                seed = list(EX["cmd"])
            if i % 10 == 0 and not STATE["ended"]:
                try:
                    ep = ee_now()
                    track.append([ri, i, STATE["ticks"], rnd(ep[0], 3), rnd(ep[1], 3), rnd(ep[2], 4),
                                  rnd(math.sqrt((ep[0] - pos[0]) ** 2 + (ep[1] - pos[1]) ** 2 + (ep[2] - pos[2]) ** 2), 4)])
                except Exception as e:
                    note("track", e)
        if routes[ri]["end_high"] and ri == len(routes) - 1 and not STATE["ended"]:
            sol = ik_cont([cur_p[0], cur_p[1], z_hover], qd, seed, 0.6)
            if sol is not None:
                move_joints(sol, 40)
                seed = sol
    if not STATE["ended"]:
        for tick in range(15):
            if not do_step(EX["cmd"], grip0, embodiment):
                break
except Exception as e:
    note("execute", e)
    # always reach control: hold the last command for a few ticks
    try:
        if not STATE["ended"]:
            hold = EX["cmd"] if EX["cmd"] is not None else vec(state0.joint_positions["primary"])
            for tick in range(5):
                if not do_step(hold, grip0, embodiment):
                    break
    except Exception as e2:
        note("execute_hold", e2)

result["dips"] = dips
result["tool_L_final"] = rnd(TOOL["L"], 4)
result["tool_contacts"] = [rnd(x, 4) for x in TOOL["contacts"]]
result["ik_stats"] = {"calls": ik_stats["calls"], "ok": ik_stats["ok"], "retry": ik_stats["retry"],
                      "skipped": ik_stats["skipped"], "max_jump": rnd(ik_stats["max_jump"])}
result["tracking"] = track
result["ticks_executed"] = STATE["ticks"]
result["ended"] = STATE["ended"]
result["end_reason"] = STATE["end_reason"]
result["first_step_reply"] = STATE["first_reply"]

# ---------------------------------------------------------------- phase 6: final
if not STATE["ended"]:
    try:
        rs = get_robot_state()
        result["final_ee"] = [rnd(x) for x in vec(rs.end_effector_poses["primary"].position)]
        result["final_timestamp_s"] = rnd(rs.timestamp_s)
    except Exception as e:
        note("final_state", e)
