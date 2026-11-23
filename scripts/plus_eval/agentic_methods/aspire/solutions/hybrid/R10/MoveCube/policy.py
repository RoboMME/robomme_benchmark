# MoveCube v00 — retarget the conditioning video's end-effector trajectory onto the live scene.
#
# Strategy (all quantities derived at runtime):
#  1. Read the conditioning video (Observations with RGB-D, calibration, robot state).
#  2. Find the grasp event in the video from the recorded gripper reading and the EE pose there.
#  3. In video frame 0, project RGB-D to robot_base, extract elevated connected components and
#     identify the grasped object (tool / ball / cube) as the component under the grasp point.
#  4. In the live frame, extract the same components, find cube (red), target (purple),
#     ramp (blue) and the live component matching the grasped video object by colour/shape features.
#  5. Grasp the live object at the same relative point / relative yaw, then replay the video's
#     post-grasp EE trajectory transformed by the 2-D rigid map (cube->target frame, or ramp frame
#     for the ball), with a stretch along the push axis for differing cube-target distance.
import math

report = {"version": "v04"}
events = []
report["events"] = events


def note(key, value):
    report[key] = value


# ------------------------------------------------------------------ small math helpers
def quat_to_mat(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def rz_quat(angle):
    return [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]


def yaw_of(q):
    m = quat_to_mat(q)
    return math.atan2(m[1][0], m[0][0])


def wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def median(vals):
    s = sorted(vals)
    if len(s) == 0:
        return 0.0
    return s[len(s) // 2]


def quantile(vals, f):
    s = sorted(vals)
    if len(s) == 0:
        return 0.0
    return s[int((len(s) - 1) * f)]


def as_list(arr):
    try:
        return arr.tolist()
    except Exception:
        return [list(row) for row in arr]


# ------------------------------------------------------------------ camera projection
def cam_params(cam):
    K = as_list(cam.intrinsics)
    pose = cam.camera_pose
    R = quat_to_mat(list(pose.quaternion_wxyz))
    t = [float(v) for v in pose.position]
    return float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2]), R, t


def pix_to_world(params, conv, u, v, d):
    fx, fy, cx, cy, R, t = params
    a = (u - cx) / fx * d
    b = (v - cy) / fy * d
    if conv == 0:      # OpenCV: x right, y down, z forward
        pc = (a, b, d)
    elif conv == 1:    # OpenGL: x right, y up, z backward
        pc = (a, -b, -d)
    else:              # x forward, y left, z up
        pc = (d, -a, -b)
    return (
        R[0][0] * pc[0] + R[0][1] * pc[1] + R[0][2] * pc[2] + t[0],
        R[1][0] * pc[0] + R[1][1] * pc[1] + R[1][2] * pc[2] + t[1],
        R[2][0] * pc[0] + R[2][1] * pc[1] + R[2][2] * pc[2] + t[2],
    )


def pick_convention(params, depth, H, W):
    t = params[5]
    # The table dominates the front view: the correct convention makes it horizontal.
    best = None
    for conv in (0, 1, 2):
        zs = []
        for v in range(H // 2, H, 8):
            for u in range(8, W - 8, 8):
                d = float(depth[v][u])
                if d > 0.05 and d < 5.0:
                    zs.append(pix_to_world(params, conv, u, v, d)[2])
        if len(zs) < 20:
            continue
        spread = quantile(zs, 0.75) - quantile(zs, 0.25)
        mz = median(zs)
        if mz > t[2] - 0.05:
            spread += 10.0
        if best is None or spread < best[0]:
            best = (spread, conv, mz)
    return best


# ------------------------------------------------------------------ colour classes
def color_class(r, g, b):
    if r > 90 and g < 0.35 * r and b < 0.35 * r:
        return "red"
    if r > g + 30 and b > g + 30:
        return "purple"
    if r > 110 and g > 0.75 * r and b < 0.55 * g:
        return "yellow"
    if b > r + 40 and b > g + 15:
        return "blue"
    if g > r + 25 and b > r + 15:
        return "teal"
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx - mn < 28:
        if mx > 125:
            return "lgrey"
        return "dgrey"
    if r > 170 and g > 105 and b > 85 and r - g < 110:
        return "peach"
    if r > 120 and g > 80 and b < 0.8 * g:
        return "tan"
    return "other"


CLASSES = ["red", "purple", "yellow", "blue", "teal", "lgrey", "dgrey", "peach", "tan", "other"]


# ------------------------------------------------------------------ scene parsing
def parse_scene(obs, conv_hint):
    cam = obs.cameras["front"]
    rgb = as_list(cam.rgb)
    depth = as_list(cam.depth_m)
    H = len(depth)
    W = len(depth[0])
    params = cam_params(cam)
    if conv_hint is None:
        best = pick_convention(params, depth, H, W)
        conv = best[1]
        table_z = best[2]
    else:
        conv = conv_hint[0]
        table_z = None
    pts = [[None] * W for row in range(H)]
    cls = [[None] * W for row in range(H)]
    allz = []
    for v in range(H):
        drow = depth[v]
        crow = rgb[v]
        for u in range(W):
            d = float(drow[u])
            if d <= 0.05 or d > 5.0:
                continue
            p = pix_to_world(params, conv, u, v, d)
            pts[v][u] = p
            c = crow[u]
            cls[v][u] = color_class(int(c[0]), int(c[1]), int(c[2]))
            if (u + v) % 7 == 0:
                allz.append(p[2])
    if table_z is None:
        table_z = median(allz)
    # refine table height as the median of the lower half heights near the mode
    near = [z for z in allz if abs(z - table_z) < 0.03]
    if len(near) > 50:
        table_z = median(near)
    purple = []
    blue = []
    elev = [[False] * W for row in range(H)]
    for v in range(H):
        for u in range(W):
            p = pts[v][u]
            if p is None:
                continue
            if cls[v][u] == "purple" and p[2] < table_z + 0.02:
                purple.append(p)
            if cls[v][u] == "blue" and p[2] > table_z + 0.004:
                blue.append(p)
            h = p[2] - table_z
            if h > 0.004 and h < 0.25 and cls[v][u] != "blue" and p[0] * p[0] + p[1] * p[1] > 0.18 * 0.18:
                elev[v][u] = True
    # connected components on elevated pixels (4-neighbourhood, depth-continuity)
    seen = [[False] * W for row in range(H)]
    comps = []
    for v0 in range(H):
        for u0 in range(W):
            if not elev[v0][u0] or seen[v0][u0]:
                continue
            queue = [(v0, u0)]
            seen[v0][u0] = True
            qi = 0
            while qi < len(queue):
                v, u = queue[qi]
                qi += 1
                p = pts[v][u]
                for dv, du in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    vv = v + dv
                    uu = u + du
                    if vv < 0 or vv >= H or uu < 0 or uu >= W:
                        continue
                    if seen[vv][uu] or not elev[vv][uu]:
                        continue
                    q = pts[vv][uu]
                    if abs(q[0] - p[0]) + abs(q[1] - p[1]) + abs(q[2] - p[2]) > 0.03:
                        continue
                    seen[vv][uu] = True
                    queue.append((vv, uu))
            if len(queue) < 6:
                continue
            cpts = []
            counts = {}
            for k in CLASSES:
                counts[k] = 0
            for (v, u) in queue:
                p = pts[v][u]
                c = cls[v][u]
                counts[c] += 1
                col = rgb[v][u]
                cpts.append((p[0], p[1], p[2], c, int(col[0]), int(col[1]), int(col[2])))
            comps.append({"pts": cpts, "counts": counts, "n": len(cpts)})
    return {
        "conv": conv,
        "table_z": table_z,
        "purple": purple,
        "blue": blue,
        "comps": comps,
        "touches_top": None,
    }


def comp_summary(comp, table_z):
    pts = comp["pts"]
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts) / n
    syy = sum((p[1] - my) ** 2 for p in pts) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts) / n
    ang = 0.5 * math.atan2(2 * sxy, sxx - syy)
    ax = (math.cos(ang), math.sin(ang))
    proj = [(p[0] - mx) * ax[0] + (p[1] - my) * ax[1] for p in pts]
    lat = [-(p[0] - mx) * ax[1] + (p[1] - my) * ax[0] for p in pts]
    lo = quantile(proj, 0.01)
    hi = quantile(proj, 0.99)
    length = max(hi - lo, 1e-3)
    fr = {}
    for k in CLASSES:
        fr[k] = comp["counts"][k] / float(n)
    zmax = quantile([p[2] for p in pts], 0.95) - table_z
    width = quantile(lat, 0.97) - quantile(lat, 0.03)
    # end colours
    end1 = [0.0, 0.0, 0.0, 0]
    end2 = [0.0, 0.0, 0.0, 0]
    for i in range(n):
        f = (proj[i] - lo) / length
        if f < 0.2:
            end1 = [end1[0] + pts[i][4], end1[1] + pts[i][5], end1[2] + pts[i][6], end1[3] + 1]
        elif f > 0.8:
            end2 = [end2[0] + pts[i][4], end2[1] + pts[i][5], end2[2] + pts[i][6], end2[3] + 1]
    c1 = [end1[j] / max(end1[3], 1) for j in range(3)]
    c2 = [end2[j] / max(end2[3], 1) for j in range(3)]
    # per-end lateral spread (shape cue: hook foot / ball / paddle / hammer head)
    w1 = [lat[i] for i in range(n) if (proj[i] - lo) / length < 0.2]
    w2 = [lat[i] for i in range(n) if (proj[i] - lo) / length > 0.8]
    sp1 = (quantile(w1, 0.95) - quantile(w1, 0.05)) if len(w1) > 3 else 0.0
    sp2 = (quantile(w2, 0.95) - quantile(w2, 0.05)) if len(w2) > 3 else 0.0
    return {
        "center": (mx, my),
        "axis": ax,
        "axis_yaw": ang,
        "lo": lo,
        "hi": hi,
        "length": length,
        "width": width,
        "zmax": zmax,
        "frac": fr,
        "end1": c1,
        "end2": c2,
        "sp1": sp1,
        "sp2": sp2,
        "n": n,
        "proj": proj,
        "lat": lat,
    }


def dominant(s):
    best = None
    for k in CLASSES:
        if best is None or s["frac"][k] > s["frac"][best]:
            best = k
    return best


def feature_dist(a, b):
    d = 0.0
    for k in CLASSES:
        d += abs(a["frac"][k] - b["frac"][k])
    d += abs(a["length"] - b["length"]) / 0.05
    d += abs(a["width"] - b["width"]) / 0.03
    d += abs(a["zmax"] - b["zmax"]) / 0.02
    # end shape compared orientation-free
    e_same = abs(a["sp1"] - b["sp1"]) + abs(a["sp2"] - b["sp2"])
    e_swap = abs(a["sp1"] - b["sp2"]) + abs(a["sp2"] - b["sp1"])
    d += min(e_same, e_swap) / 0.02
    return d


def col_dist(c1, c2):
    return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])


def mean_xyz(pts):
    n = float(len(pts))
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n, sum(p[2] for p in pts) / n)


def find_cube(scene):
    best = None
    for c in scene["comps"]:
        nred = c["counts"]["red"]
        if nred >= 5 and (best is None or nred > best["counts"]["red"]):
            best = c
    if best is None:
        return None
    reds = [p for p in best["pts"] if p[3] == "red"]
    return mean_xyz(reds)


def find_target(scene):
    pur = scene["purple"]
    if len(pur) < 5:
        return None
    # cluster: keep points near the median
    mx = median([p[0] for p in pur])
    my = median([p[1] for p in pur])
    keep = [p for p in pur if abs(p[0] - mx) < 0.06 and abs(p[1] - my) < 0.06]
    if len(keep) < 3:
        keep = pur
    return mean_xyz(keep)


def find_ramp(scene):
    bl = scene["blue"]
    if len(bl) < 30:
        return None
    mx0 = median([p[0] for p in bl])
    my0 = median([p[1] for p in bl])
    bl = [p for p in bl if abs(p[0] - mx0) < 0.2 and abs(p[1] - my0) < 0.2]
    if len(bl) < 30:
        return None
    n = float(len(bl))
    mx = sum(p[0] for p in bl) / n
    my = sum(p[1] for p in bl) / n
    mz = sum(p[2] for p in bl) / n
    sxx = sum((p[0] - mx) ** 2 for p in bl)
    syy = sum((p[1] - my) ** 2 for p in bl)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in bl)
    sxz = sum((p[0] - mx) * (p[2] - mz) for p in bl)
    syz = sum((p[1] - my) * (p[2] - mz) for p in bl)
    det = sxx * syy - sxy * sxy
    if abs(det) < 1e-12:
        return None
    a = (sxz * syy - syz * sxy) / det
    b = (syz * sxx - sxz * sxy) / det
    g = math.sqrt(a * a + b * b)
    if g < 1e-6:
        return None
    down = (-a / g, -b / g)
    top = max(p[2] for p in bl)
    return {"center": (mx, my, mz), "down": down, "slope": g, "top_z": top}


# ------------------------------------------------------------------ helpers for motion
def ended(res):
    try:
        if res.terminated or res.truncated:
            return True
    except Exception:
        pass
    return False


def mk_pose(p, q, frame):
    return Pose((float(p[0]), float(p[1]), float(p[2])), [float(v) for v in q], frame)


flags = {"over": False}


def go(p, q, frame, tol, steps, label):
    if flags["over"]:
        return None
    res = move_to_pose(mk_pose(p, q, frame), tolerance=tol, max_steps=steps)
    rec = {"label": label, "target": [round(float(v), 4) for v in p], "ok": False}
    try:
        rec["ok"] = bool(res.ok)
        rec["steps"] = int(res.steps_executed)
        rec["status"] = str(res.status)
    except Exception:
        pass
    if ended(res):
        rec["ended"] = True
        flags["over"] = True
    events.append(rec)
    return res


def grip(open_it, label):
    if flags["over"]:
        return None
    res = open_gripper() if open_it else close_gripper()
    rec = {"label": label, "open": open_it}
    try:
        rec["ok"] = bool(res.ok)
        rec["steps"] = int(res.steps_executed)
    except Exception:
        pass
    if ended(res):
        rec["ended"] = True
        flags["over"] = True
    events.append(rec)
    return res


# ================================================================== main
ctx = get_task_context()
note("goal", ctx.language)
try:
    meta = robomme_plus.get_task_metadata()
    note("max_steps", meta["max_steps"])
except Exception as exc:
    note("meta_error", str(exc))

state0 = get_robot_state()
frame = state0.base_frame
home_q = list(state0.end_effector_poses["primary"].quaternion_wxyz)
home_yaw = yaw_of(home_q)
home_j = [float(v) for v in state0.joint_positions["primary"]]
sig_h = -1.0
try:
    hp = [float(v) for v in state0.end_effector_poses["primary"].position]
    th = solve_ik(Pose((hp[0], hp[1], hp[2]), quat_mul(rz_quat(0.3), home_q), state0.base_frame), seed_joints=home_j)
    sig_h = 1.0 if float(th.joint_positions[6]) - home_j[6] > 0 else -1.0
except Exception as exc:
    report["sig_error"] = str(exc)
report["sig_h"] = sig_h


def j7_overflow(jg, D0, cum):
    best = None
    for D in (D0, D0 - 2 * math.pi, D0 + 2 * math.pi):
        j7s = [jg + sig_h * (D + c) for c in cum] + [jg + sig_h * D, jg]
        over = max(0.0, max(j7s) - 2.8) + max(0.0, -2.8 - min(j7s))
        if best is None or over < best:
            best = over
    return best

video = robomme_plus.get_conditioning_video()
nvid = len(video)
note("video_frames", nvid)

live_obs = get_observation()
live = parse_scene(live_obs, None)
note("live_conv", live["conv"])
note("live_table_z", round(live["table_z"], 4))
c_l = find_cube(live)
t_l = find_target(live)
ramp_l = find_ramp(live)
note("live_cube", c_l)
note("live_target", t_l)

mode = "fallback_push"
plan_ok = False
if nvid > 2 and c_l is not None and t_l is not None:
    vframes = [video[i] for i in range(nvid)]
    vp = []
    vq = []
    vg = []
    for fobs in vframes:
        rs = fobs.robot_state
        pz = rs.end_effector_poses["primary"]
        vp.append([float(v) for v in pz.position])
        vq.append([float(v) for v in pz.quaternion_wxyz])
        vg.append(float(rs.gripper_positions["primary"]))
    note("video_grip_range", [round(min(vg), 3), round(max(vg), 3)])
    # grasp event: gripper reading drops 0.25 below its running maximum
    g_idx = None
    run_max = vg[0]
    for i in range(nvid):
        run_max = max(run_max, vg[i])
        if vg[i] < run_max - 0.25:
            g_idx = i
            break
    rel_idx = None
    if g_idx is not None:
        hold = vg[min(g_idx + 3, nvid - 1)]
        for i in range(g_idx + 3, nvid):
            if vg[i] > hold + 0.25:
                rel_idx = i
                break
    note("video_grasp_idx", g_idx)
    note("video_traj", [[i, round(vp[i][0], 3), round(vp[i][1], 3), round(vp[i][2], 3),
                         round(math.degrees(yaw_of(vq[i])), 1), round(vg[i], 2)] for i in range(0, nvid, 6)])
    note("video_release_idx", rel_idx)
    vscene = parse_scene(vframes[0], None)
    note("video_conv", vscene["conv"])
    c_v = find_cube(vscene)
    t_v = find_target(vscene)
    ramp_v = find_ramp(vscene)
    note("video_cube", c_v)
    note("video_target", t_v)
    grasped = None
    gsum = None
    if g_idx is not None:
        pg = vp[g_idx]
        best_n = 0
        for c in vscene["comps"]:
            nn = 0
            for p in c["pts"]:
                if abs(p[0] - pg[0]) < 0.025 and abs(p[1] - pg[1]) < 0.025:
                    nn += 1
            if nn > best_n:
                best_n = nn
                grasped = c
        vdiag = []
        for c in vscene["comps"]:
            if c["n"] < 15:
                continue
            md = min(math.sqrt((p[0] - pg[0]) ** 2 + (p[1] - pg[1]) ** 2) for p in c["pts"])
            cc = mean_xyz(c["pts"])
            top = None
            for kk in CLASSES:
                if top is None or c["counts"][kk] > c["counts"][top]:
                    top = kk
            vdiag.append([round(cc[0], 3), round(cc[1], 3), round(cc[2], 3), c["n"], top, round(md, 3)])
        note("video_comps", vdiag)
        if grasped is None:
            best_d = 0.08
            for c in vscene["comps"]:
                for p in c["pts"]:
                    dd = math.sqrt((p[0] - pg[0]) ** 2 + (p[1] - pg[1]) ** 2)
                    if dd < best_d:
                        best_d = dd
                        grasped = c
            note("grasp_comp_fallback_dist", round(best_d, 3))
        if grasped is not None:
            gsum = comp_summary(grasped, vscene["table_z"])
            note("video_grasped", {"dominant": dominant(gsum), "n": gsum["n"],
                                   "length": round(gsum["length"], 3),
                                   "width": round(gsum["width"], 3)})
    kind = "none"
    if gsum is not None:
        dom = dominant(gsum)
        if gsum["frac"]["red"] > 0.4:
            kind = "cube"
        elif gsum["frac"]["yellow"] > 0.4 and gsum["length"] < 0.08:
            kind = "ball"
        else:
            kind = "tool"
    note("video_kind", kind)

    if c_v is not None and t_v is not None:
        # --- anchor frames
        use_ramp = kind == "ball" and ramp_v is not None and ramp_l is not None
        if use_ramp:
            A_v = ramp_v["center"]
            d_v = ramp_v["down"]
            A_l = ramp_l["center"]
            d_l = ramp_l["down"]
        else:
            A_v = c_v
            A_l = c_l
            dv = (t_v[0] - c_v[0], t_v[1] - c_v[1])
            dl = (t_l[0] - c_l[0], t_l[1] - c_l[1])
            nv = math.sqrt(dv[0] ** 2 + dv[1] ** 2)
            nl = math.sqrt(dl[0] ** 2 + dl[1] ** 2)
            d_v = (dv[0] / max(nv, 1e-6), dv[1] / max(nv, 1e-6))
            d_l = (dl[0] / max(nl, 1e-6), dl[1] / max(nl, 1e-6))
        phi = wrap(math.atan2(d_l[1], d_l[0]) - math.atan2(d_v[1], d_v[0]))
        note("phi_deg", round(math.degrees(phi), 1))
        D_v = math.sqrt((t_v[0] - c_v[0]) ** 2 + (t_v[1] - c_v[1]) ** 2)
        D_l = math.sqrt((t_l[0] - c_l[0]) ** 2 + (t_l[1] - c_l[1]) ** 2)
        note("D_v", round(D_v, 3))
        note("D_l", round(D_l, 3))

        # --- work-phase start k
        start = 0 if g_idx is None else g_idx
        end = nvid - 1
        z_g = vp[start][2]
        f1 = None
        lifted = False
        stop_i = end if rel_idx is None else rel_idx
        for i in range(start, stop_i + 1):
            if vp[i][2] > z_g + 0.05:
                lifted = True
            if lifted and vp[i][2] < z_g + 0.03:
                f1 = i
                break
        if f1 is None:
            f1 = stop_i
        k = start
        for i in range(start, nvid):
            k = i
            if math.sqrt((vp[i][0] - vp[start][0]) ** 2 + (vp[i][1] - vp[start][1]) ** 2) > 0.03:
                break
        lift_z = max(vp[i][2] for i in range(start, k + 1))
        if kind == "cube" or kind == "none":
            k = 0
        note("work_start_idx", k)

        def to_uw(p):
            rx = p[0] - A_v[0]
            ry = p[1] - A_v[1]
            return (rx * d_v[0] + ry * d_v[1], -rx * d_v[1] + ry * d_v[0])

        # stretch along push axis (cube anchor only)
        u_end = None
        zmin_work = min(vp[i][2] for i in range(k, stop_i + 1))
        note("zmin_work", round(zmin_work, 3))
        if not use_ramp:
            for i in range(k, stop_i + 1):
                if vp[i][2] > zmin_work + 0.02:
                    continue
                uu = to_uw(vp[i])[0]
                if u_end is None or uu > u_end:
                    u_end = uu
        scale = 1.0
        if u_end is not None and D_v > 0.02:
            scale = D_l / D_v
            if scale < 0.4:
                scale = 0.4
            if scale > 2.5:
                scale = 2.5
        u_c = None if u_end is None else u_end - D_v
        note("u_end", None if u_end is None else round(u_end, 3))

        def map_p(p):
            u, w = to_uw(p)
            if u_c is not None:
                if u > u_end:
                    u = u_c + D_v * scale + (u - u_end)
                elif u > u_c:
                    u = u_c + (u - u_c) * scale
            if xf["refl"]:
                w = -w
            return (A_l[0] + u * d_l[0] - w * d_l[1], A_l[1] + u * d_l[1] + w * d_l[0], p[2])

        a_v = math.atan2(d_v[1], d_v[0])
        a_l = math.atan2(d_l[1], d_l[0])
        # xf: refl = mirror the video across the push axis (sideways sweeps are mirror-symmetric),
        #     flip = rotate the hand 180 deg about its approach axis (parallel gripper symmetry)
        xf = {"refl": 0, "flip": 0}

        def map_q(q):
            if xf["refl"]:
                ps = yaw_of(q)
                out = quat_mul(rz_quat(wrap(a_l + a_v - 2.0 * ps)), q)
            else:
                out = quat_mul(rz_quat(phi), q)
            if xf["flip"]:
                out = quat_mul(out, rz_quat(math.pi))
            return out

        def reach_pen(p):
            r = math.sqrt(p[0] * p[0] + p[1] * p[1])
            return max(0.0, 0.33 - r) * 20.0 + max(0.0, r - 0.78) * 20.0

        # --- subsample waypoints from k
        wps = []
        last = None
        last_yaw = None
        prev_open = vg[k] > (vg[g_idx] + 0.25 if g_idx is not None else 0.5)
        for i in range(k, nvid):
            yw = yaw_of(vq[i])
            is_open = vg[i] > 0.5 * (max(vg) + min(vg))
            grip_change = is_open != prev_open
            take = last is None or grip_change or i == nvid - 1
            if not take:
                dd = math.sqrt(sum((vp[i][j] - last[j]) ** 2 for j in range(3)))
                if dd > 0.025 or abs(wrap(yw - last_yaw)) > 0.2:
                    take = True
            if u_end is not None and not take and abs(to_uw(vp[i])[0] - u_end) < 1e-9:
                take = True
            if take:
                wps.append((i, grip_change, is_open))
                last = vp[i]
                last_yaw = yw
            prev_open = is_open
        note("n_waypoints", len(wps))

        # --- live grasp computation for tool / ball
        grasp_ok = True
        live_grasp_xy = None
        q_grasp_live = None
        if kind == "tool" or kind == "ball":
            cands = []
            for c in live["comps"]:
                s = comp_summary(c, live["table_z"])
                if s["n"] < 8:
                    continue
                if s["frac"]["red"] > 0.3 or s["frac"]["blue"] > 0.3:
                    continue
                cands.append((feature_dist(gsum, s), s, c))
            note("video_grasped_feat", {"frac": dict([(kk, round(gsum["frac"][kk], 3)) for kk in CLASSES]),
                                        "len": round(gsum["length"], 3), "w": round(gsum["width"], 3),
                                        "zmax": round(gsum["zmax"], 3), "sp": [round(gsum["sp1"], 3), round(gsum["sp2"], 3)]})
            note("live_cand_feat", [{"d": round(e[0], 2), "c": [round(e[1]["center"][0], 3), round(e[1]["center"][1], 3)],
                                     "frac": dict([(kk, round(e[1]["frac"][kk], 2)) for kk in CLASSES if e[1]["frac"][kk] > 0.02]),
                                     "len": round(e[1]["length"], 3), "w": round(e[1]["width"], 3),
                                     "zmax": round(e[1]["zmax"], 3)} for e in cands])
            order = sorted([(cands[i][0], i) for i in range(len(cands))])
            cands = [cands[o[1]] for o in order]
            note("live_candidates", [[round(e[0], 2), dominant(e[1]), e[1]["n"],
                                      round(e[1]["length"], 3)] for e in cands[:6]])
            if len(cands) == 0:
                grasp_ok = False
                note("no_live_candidate", True)
            else:
                ls = cands[0][1]
                pg = vp[g_idx]
                if kind == "ball":
                    yel = [p for p in cands[0][2]["pts"]]
                    mxyz = mean_xyz(yel)
                    live_grasp_xy = (mxyz[0], mxyz[1])
                    vb = [p for p in grasped["pts"]]
                    vb_c = mean_xyz(vb)
                    # keep the same small offset between grasp point and ball centre
                    live_grasp_xy = (live_grasp_xy[0] + (pg[0] - vb_c[0]),
                                     live_grasp_xy[1] + (pg[1] - vb_c[1]))
                    alpha_v = a_v
                    alpha_l = a_l
                else:
                    # align end orientation by end colours
                    same = col_dist(gsum["end1"], ls["end1"]) + col_dist(gsum["end2"], ls["end2"])
                    swap = col_dist(gsum["end1"], ls["end2"]) + col_dist(gsum["end2"], ls["end1"])
                    ax_l = ls["axis"]
                    lo_l = ls["lo"]
                    hi_l = ls["hi"]
                    proj_l = ls["proj"]
                    lat_l = ls["lat"]
                    if swap < same:
                        ax_l = (-ax_l[0], -ax_l[1])
                        lo_l, hi_l = -hi_l, -lo_l
                        proj_l = [-v for v in proj_l]
                        lat_l = [-v for v in lat_l]
                    ax_v = gsum["axis"]
                    sv = ((pg[0] - gsum["center"][0]) * ax_v[0] + (pg[1] - gsum["center"][1]) * ax_v[1])
                    frac = (sv - gsum["lo"]) / gsum["length"]
                    note("grasp_frac", round(frac, 3))
                    s_l = lo_l + frac * (hi_l - lo_l)
                    near_lat = [lat_l[i] for i in range(len(proj_l)) if abs(proj_l[i] - s_l) < 0.012]
                    w_l = median(near_lat) if len(near_lat) > 0 else 0.0
                    cx, cy = ls["center"]
                    perp = (-ax_l[1], ax_l[0])
                    live_grasp_xy = (cx + s_l * ax_l[0] + w_l * perp[0], cy + s_l * ax_l[1] + w_l * perp[1])
                    alpha_v = math.atan2(ax_v[1], ax_v[0])
                    alpha_l = math.atan2(ax_l[1], ax_l[0])
                    note("alpha_v_deg", round(math.degrees(alpha_v), 1))
                    note("alpha_l_deg", round(math.degrees(alpha_l), 1))
                # choose mirror / wrist-flip variant: reachability first, then wrist yaw near home
                best_cost = None
                costs = []
                psi_vg = yaw_of(vq[g_idx])
                for rv in (0, 1):
                    for fv in (0, 1):
                        xf["refl"] = rv
                        xf["flip"] = fv
                        if rv:
                            psi_lg = alpha_l - (psi_vg - alpha_v)
                            qg = quat_mul(rz_quat(wrap(psi_lg - psi_vg)), vq[g_idx])
                        else:
                            qg = quat_mul(rz_quat(wrap(alpha_l - alpha_v)), vq[g_idx])
                        if fv:
                            qg = quat_mul(qg, rz_quat(math.pi))
                        ymax = abs(wrap(yaw_of(qg) - home_yaw))
                        rp = 0.0
                        for (i, gc, io) in wps:
                            ymax = max(ymax, abs(wrap(yaw_of(map_q(vq[i])) - home_yaw)))
                            rp += reach_pen(map_p(vp[i]))
                        # predicted joint-7 travel over the whole sweep (best rotation direction)
                        jg_pred = home_j[6] + sig_h * wrap(yaw_of(qg) - home_yaw)
                        sgn = -1.0 if rv else 1.0
                        cumv = []
                        acc = 0.0
                        prev = yaw_of(vq[k])
                        for (i, gc, io) in wps:
                            yi = yaw_of(vq[i])
                            acc += wrap(yi - prev)
                            prev = yi
                            cumv.append(sgn * acc)
                        ov = j7_overflow(jg_pred, wrap(yaw_of(map_q(vq[k])) - yaw_of(qg)), cumv)
                        cost = rp + ov * 20.0 + 0.1 * ymax
                        costs.append([rv, fv, round(rp, 3), round(ymax, 2), round(ov, 3)])
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                            best_xf = (rv, fv)
                            q_grasp_live = qg
                xf["refl"] = best_xf[0]
                xf["flip"] = best_xf[1]
                note("xf_costs", costs)
                note("xf", [xf["refl"], xf["flip"]])
                pgv = vp[g_idx]
                vz = [p[2] for p in grasped["pts"] if abs(p[0] - pgv[0]) < 0.02 and abs(p[1] - pgv[1]) < 0.02]
                lz = [p[2] for p in cands[0][2]["pts"] if abs(p[0] - live_grasp_xy[0]) < 0.02 and abs(p[1] - live_grasp_xy[1]) < 0.02]
                z_off = 0.0
                if len(vz) > 2 and len(lz) > 2:
                    z_off = quantile(lz, 0.9) - quantile(vz, 0.9)
                if z_off > 0.15:
                    z_off = 0.15
                if z_off < -0.15:
                    z_off = -0.15
                note("grasp_z_offset", round(z_off, 4))
                note("live_grasp_xy", [round(live_grasp_xy[0], 4), round(live_grasp_xy[1], 4)])
        if kind == "cube" or kind == "none":
            bestc = None
            for rv in (0, 1):
                xf["refl"] = rv
                rp = sum(reach_pen(map_p(vp[i])) for (i, gc, io) in wps)
                if bestc is None or rp < bestc[0]:
                    bestc = (rp, rv)
            xf["refl"] = bestc[1]
            note("xf", [xf["refl"], 0])
        note("mode_kind", kind)
        plan_ok = grasp_ok
        mode = "retarget_" + kind

        if plan_ok:
            grip(True, "open_start")
            if kind == "tool" or kind == "ball":
                zg = vp[g_idx][2] + z_off
                gx, gy = live_grasp_xy
                go((gx, gy, zg + 0.10), q_grasp_live, frame, 0.015, 200, "pregrasp")
                go((gx, gy, zg + 0.03), q_grasp_live, frame, 0.01, 150, "descend")
                go((gx, gy, zg), q_grasp_live, frame, 0.008, 120, "grasp_pose")
                grip(False, "close_tool")
                try:
                    report["grip_after_close"] = float(get_robot_state().gripper_positions["primary"])
                except Exception:
                    pass
                h0 = max(lift_z + z_off + 0.02, zg + 0.06)
                go((gx, gy, h0), q_grasp_live, frame, 0.015, 150, "lift")
                pk = map_p(vp[k])
                # --- wrist unwinding: pick the rotation direction that keeps joint 7 inside its
                # limits for the whole remaining sweep, then pre-rotate in place via seeded IK.
                try:
                    stg = get_robot_state()
                    jg = [float(v) for v in stg.joint_positions["primary"]]
                    qcur = list(stg.end_effector_poses["primary"].quaternion_wxyz)
                    pcur = [float(v) for v in stg.end_effector_poses["primary"].position]
                    ycur = yaw_of(qcur)
                    test = solve_ik(mk_pose(pcur, quat_mul(rz_quat(0.3), qcur), frame), seed_joints=jg)
                    sig = (float(test.joint_positions[6]) - jg[6]) / 0.3
                    sig = 1.0 if sig > 0 else -1.0
                    yk = yaw_of(map_q(vq[k]))
                    D0 = wrap(yk - ycur)
                    sgn = -1.0 if xf["refl"] else 1.0
                    cum = []
                    acc = 0.0
                    prev = yaw_of(vq[k])
                    for (i, gc, io) in wps:
                        yi = yaw_of(vq[i])
                        acc += wrap(yi - prev)
                        prev = yi
                        cum.append(sgn * acc)
                    best = None
                    for D in (D0, D0 - 2 * math.pi, D0 + 2 * math.pi):
                        j7s = [jg[6] + sig * (D + c) for c in cum] + [jg[6] + sig * D]
                        over = max(0.0, max(j7s) - 2.8) + max(0.0, -2.8 - min(j7s))
                        score = over * 100.0 + abs(D)
                        if best is None or score < best[0]:
                            best = (score, D)
                    D = best[1]
                    note("unwind", {"sig": sig, "D0": round(D0, 3), "D": round(D, 3), "j7g": round(jg[6], 3),
                                    "cum_range": [round(min(cum), 3), round(max(cum), 3)] if len(cum) > 0 else None})
                    for part in (1, 2, 3):
                        qd = quat_mul(rz_quat(D * part / 3.0), qcur)
                        seed = [v for v in get_robot_state().joint_positions["primary"]]
                        seed = [float(v) for v in seed]
                        seed[6] = jg[6] + sig * D * part / 3.0
                        sol = solve_ik(mk_pose(pcur, qd, frame), seed_joints=seed)
                        if sol.ok:
                            mr = move_to_joints([float(v) for v in sol.joint_positions], tolerance=0.02, max_steps=150)
                            events.append({"label": "unwind" + str(part), "ok": bool(mr.ok),
                                           "j7": round(float(sol.joint_positions[6]), 3)})
                            if ended(mr):
                                flags["over"] = True
                                break
                        else:
                            events.append({"label": "unwind" + str(part), "ok": False, "ik": "failed"})
                except Exception as exc:
                    note("unwind_error", str(exc))
                go((pk[0], pk[1], max(h0, pk[2] + 0.02)), map_q(vq[k]), frame, 0.02, 250, "transit")
            for (i, gc, io) in wps:
                if flags["over"]:
                    break
                p = map_p(vp[i])
                if p[2] > zmin_work + 0.03:
                    go(p, map_q(vq[i]), frame, 0.018, 60, "wp" + str(i))
                else:
                    go(p, map_q(vq[i]), frame, 0.01, 80, "wp" + str(i))
                if gc:
                    grip(io, "grip_wp" + str(i))
            # final settle: hold a few ticks with current joints
            if not flags["over"]:
                try:
                    st = get_robot_state()
                    gpos = 1.0 if st.gripper_positions["primary"] > 0.5 else 0.0
                    for tick in range(10):
                        sr = step(RobotAction(arms={"primary": ArmCommand(
                            "joint_position", st.joint_positions["primary"],
                            gripper_position=gpos, embodiment=st.embodiment)}))
                        try:
                            if sr.terminated or sr.truncated:
                                flags["over"] = True
                                break
                        except Exception:
                            pass
                except Exception as exc:
                    note("settle_error", str(exc))

if not plan_ok and not flags["over"]:
    # fallback: push the cube with the closed gripper from behind toward the target
    mode = "fallback_push"
    if c_l is not None and t_l is not None:
        dx = t_l[0] - c_l[0]
        dy = t_l[1] - c_l[1]
        n = max(math.sqrt(dx * dx + dy * dy), 1e-6)
        dx, dy = dx / n, dy / n
        z = live["table_z"] + 0.02
        grip(False, "close_fb")
        go((c_l[0] - dx * 0.07, c_l[1] - dy * 0.07, z + 0.10), home_q, frame, 0.015, 200, "fb_above")
        go((c_l[0] - dx * 0.07, c_l[1] - dy * 0.07, z), home_q, frame, 0.012, 150, "fb_down")
        go((t_l[0] - dx * 0.03, t_l[1] - dy * 0.03, z), home_q, frame, 0.012, 250, "fb_push")

note("mode", mode)
note("episode_over", flags["over"])
result = report
