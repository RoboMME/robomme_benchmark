# MoveCube v03 -- UNTESTED (0 simulator runs) minimal repair of v02 for the two failures in v02's trace:
# (1) the planned branch's ascent ends at the joint-4 limit just below +6 cm -> transfer at the highest
# continuous level (one step lower as margin); (2) the strict residual filter stopped the fallback at
# waypoint 3 -> fallback uses v01's tolerance (public residual limits).
# v02: repair of v01 (backward joint-space drag plan + self-motion transfer) -- classify the manner shown in the conditioning video, then execute it on the
# live scene (start 2 of 3).
#
# v00 diagnosis (ep 0): the conditioning video's scene layout differs from the execution scene,
# so replaying the video's joints moved the arm through empty space (cube untouched).
# The video's robot-state timeline still tells HOW the cube was moved:
#   - gripper closes and holds something (width >> 0) -> grasped object = the object whose colour
#     lies at the video grasp point (projected into the video's own first front frame);
#     grasped object == cube -> "pick"; anything else -> "tool" (hook pull);
#   - no hold -> "push" with the bare gripper.
# Live scene geometry comes from front RGB-D (deprojected with the published calibration):
# above-table connected components, SAM3 "cube" (size-bounded) and "disc"/"target" (flat).
# Hook pull: grasp the tool at its handle (colour from the video grasp point), hand x along
# handle->head; measure the head bar/lip in the tool frame from the live cloud; place the
# head behind the cube (bar beyond it by CLEARANCE), pull the handle along cube->target.
import math

CLEARANCE = 0.02          # bar lowered this far beyond the cube before pulling (m)
DRAG_STEP = 0.01          # Cartesian drag waypoint spacing (m)
TICKS_PER_STEP = 6        # control ticks per drag waypoint (about 3 cm/s)
TRANSIT_Z = 0.28
PRE_DZ = 0.06             # pre-drag height above the drag height (m)
DESCEND_STEPS = 6
MIN_ASCEND_STEPS = 2      # transfer at least 2 cm above the drag height
MAX_CHAIN_JUMP = 0.25     # max joint change between consecutive Cartesian waypoints (rad)
MAX_TRANSFER_JUMP = 0.12  # max joint change between consecutive self-motion samples (rad)
TRANSFER_SAMPLES = 40
TRANSFER_TICKS = 2
IK_POS_TOL = 0.003
IK_ORI_TOL = 0.02
PULL_EXTRA = 0.005        # pull this much past the computed end (v01: cube lagged by about 2 mm)
ROT_INCREMENTS = 4        # in-place tool re-orientation split into this many pose moves
ABOVE_TABLE = 0.004
COLOR_TOL = 45
MAX_PITCH_DEG = 8.0
N_VIDEO_IMAGES = 16
GRASP_TOP_MAX = 0.045      # grasped part's top at most this far above the video grasp TCP height

HEXD = "0123456789abcdef"
HEX4 = [HEXD[i // 16] for i in range(256)]

report = {"version": "v03", "stage": "start", "errors": [], "motions": [], "ticks": 0}
result = report


def note_error(where, exc):
    report["errors"].append(where + ": " + str(exc)[:400])


def flist(values, nd):
    return [round(float(v), nd) for v in values]


def rows_of(arr, stride):
    try:
        return arr[::stride, ::stride].tolist()
    except Exception:
        pass
    full = arr.tolist()
    return [row[::stride] for row in full[::stride]]


def encode(rows):
    parts = []
    for row in rows:
        for px in row:
            parts.append(HEX4[int(px[0])] + HEX4[int(px[1])] + HEX4[int(px[2])])
    return {"h": len(rows), "w": len(rows[0]), "bits": 4, "hex": "".join(parts)}


def quant(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def term_flags(res):
    flags = {}
    try:
        flags["terminated"] = bool(res.terminated)
    except Exception:
        pass
    try:
        flags["truncated"] = bool(res.truncated)
    except Exception:
        pass
    return flags


def ended(res):
    flags = term_flags(res)
    return bool(flags.get("terminated")) or bool(flags.get("truncated"))


def quat_to_mat(q):
    w = float(q[0])
    x = float(q[1])
    y = float(q[2])
    z = float(q[3])
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def mat_to_quat(m):
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return [w / n, x / n, y / n, z / n]


def down_quat(yaw, pitch):
    # Hand z points down, hand x horizontal at `yaw`; then rotate about hand y by `pitch`
    # (positive pitch tilts hand +x downwards).
    c = math.cos(yaw)
    s = math.sin(yaw)
    xa = [c, s, 0.0]
    ya = [s, -c, 0.0]          # y = z cross x with z = (0, 0, -1)
    za = [0.0, 0.0, -1.0]
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    # R = [xa ya za] @ Ry(-pitch): hand x gets z component -sin(pitch) (tilts down for pitch > 0)
    col0 = [xa[k] * cp + za[k] * sp for k in range(3)]
    col1 = ya
    col2 = [-xa[k] * sp + za[k] * cp for k in range(3)]
    m = [[col0[r], col1[r], col2[r]] for r in range(3)]
    return mat_to_quat(m)


def cam_model(cam):
    k = cam.intrinsics
    pose = cam.camera_pose
    return {
        "fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]), "cy": float(k[1][2]),
        "R": quat_to_mat(pose.quaternion_wxyz),
        "p": [float(v) for v in pose.position],
    }


def project(model, point):
    rr = model["R"]
    d = [point[k] - model["p"][k] for k in range(3)]
    cxyz = [rr[0][k] * d[0] + rr[1][k] * d[1] + rr[2][k] * d[2] for k in range(3)]
    if cxyz[2] <= 1e-6:
        return None
    return (model["cx"] + model["fx"] * cxyz[0] / cxyz[2], model["cy"] + model["fy"] * cxyz[1] / cxyz[2])


def deproject_grid(model, depth_rows):
    rr = model["R"]
    p = model["p"]
    grid = []
    for v in range(len(depth_rows)):
        row = []
        drow = depth_rows[v]
        yc = (v - model["cy"]) / model["fy"]
        for u in range(len(drow)):
            d = float(drow[u])
            if d <= 0.0 or d > 5.0:
                row.append(None)
                continue
            xc = (u - model["cx"]) / model["fx"] * d
            ycd = yc * d
            row.append((rr[0][0] * xc + rr[0][1] * ycd + rr[0][2] * d + p[0],
                        rr[1][0] * xc + rr[1][1] * ycd + rr[1][2] * d + p[1],
                        rr[2][0] * xc + rr[2][1] * ycd + rr[2][2] * d + p[2]))
        grid.append(row)
    return grid


def components(grid, rgb_rows, zmin, zmax):
    h = len(grid)
    w = len(grid[0])
    label = [[-1] * w for v in range(h)]
    comps = []
    for v0 in range(h):
        for u0 in range(w):
            pt = grid[v0][u0]
            if pt is None or label[v0][u0] >= 0 or pt[2] <= zmin or pt[2] >= zmax:
                continue
            cid = len(comps)
            stack = [(v0, u0)]
            label[v0][u0] = cid
            pix = []
            while stack:
                cur = stack.pop()
                pix.append(cur)
                for dv, du in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    vv = cur[0] + dv
                    uu = cur[1] + du
                    if 0 <= vv < h and 0 <= uu < w and label[vv][uu] < 0:
                        q = grid[vv][uu]
                        if q is not None and zmin < q[2] < zmax:
                            label[vv][uu] = cid
                            stack.append((vv, uu))
            pts = [grid[a][b] for a, b in pix]
            cols = [rgb_rows[a][b] for a, b in pix]
            comps.append({"pix": pix, "pts": pts, "cols": cols})
    return comps


def comp_summary(c):
    xs = [p[0] for p in c["pts"]]
    ys = [p[1] for p in c["pts"]]
    zs = [p[2] for p in c["pts"]]
    return {
        "n": len(c["pts"]),
        "x": [round(min(xs), 3), round(quant(xs, 0.5), 3), round(max(xs), 3)],
        "y": [round(min(ys), 3), round(quant(ys, 0.5), 3), round(max(ys), 3)],
        "z95": round(quant(zs, 0.95), 4),
        "rgb_med": [quant([float(col[k]) for col in c["cols"]], 0.5) for k in range(3)],
    }


def color_dist(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) + abs(int(a[2]) - int(b[2]))


def cloud_points(cand, base_frame):
    pc = None
    try:
        pc = mask_to_point_cloud(cand, "front", target_frame=base_frame)
    except Exception:
        pc = mask_to_point_cloud(cand.mask, "front", target_frame=base_frame)
    try:
        return pc.points.tolist()
    except Exception:
        return [list(p) for p in pc.points]


def seg_cands(prompt, base_frame):
    out = []
    seg = segment_text("front", prompt)
    if not seg.ok:
        return out
    for cand in seg.segmentations[:5]:
        entry = {"prompt": prompt, "score": round(float(cand.score), 3)}
        try:
            entry["box"] = [round(float(v), 1) for v in cand.box_xyxy]
        except Exception:
            entry["box"] = None
        try:
            pts = cloud_points(cand, base_frame)
        except Exception as exc:
            entry["pc_error"] = str(exc)[:120]
            pts = []
        entry["n"] = len(pts)
        if len(pts) >= 10:
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            zs = [float(p[2]) for p in pts]
            entry["cx"] = quant(xs, 0.5)
            entry["cy"] = quant(ys, 0.5)
            entry["span"] = max(quant(xs, 0.95) - quant(xs, 0.05), quant(ys, 0.95) - quant(ys, 0.05))
            entry["z95"] = quant(zs, 0.95)
            entry["z05"] = quant(zs, 0.05)
        out.append(entry)
    return out


def record_motion(name, res):
    entry = {"name": name}
    try:
        entry["ok"] = bool(res.ok)
    except Exception:
        pass
    try:
        entry["status"] = str(res.status)
    except Exception:
        pass
    try:
        entry["steps"] = int(res.steps_executed)
        report["ticks"] += entry["steps"]
    except Exception:
        pass
    try:
        if res.error is not None:
            entry["error"] = str(res.error)[:300]
    except Exception:
        pass
    try:
        st = get_robot_state()
        entry["ee"] = flist(st.end_effector_poses["primary"].position, 4)
        entry["g"] = round(float(st.gripper_positions["primary"]), 4)
    except Exception:
        pass
    report["motions"].append(entry)
    return entry


END_TAG = "EPISODE_ENDED:"


def episode_ended(name):
    return ValueError(END_TAG + name)


def go_pose(name, pos, quat, tol, steps):
    res = move_to_pose(Pose([float(v) for v in pos], quat, BASE), tolerance=tol, max_steps=steps)
    rec = record_motion(name, res)
    if not rec.get("ok", True) and rec.get("steps", 1) == 0 and not ended(res):
        # plan longer than max_steps is rejected without moving: retry once with a larger cap
        res = move_to_pose(Pose([float(v) for v in pos], quat, BASE), tolerance=tol, max_steps=2 * steps)
        record_motion(name + "_retry", res)
    if ended(res):
        raise episode_ended(name)
    return res


def hold_ticks(n, gripper):
    st = get_robot_state()
    for tick in range(n):
        res = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=st.joint_positions["primary"],
            gripper_position=gripper, embodiment=st.embodiment)}))
        report["ticks"] += 1
        if ended(res):
            raise episode_ended("hold")


def ik_good(pos, quat, seed, ptol=None, otol=None):
    try:
        if seed is None:
            r = solve_ik(Pose([float(v) for v in pos], quat, BASE))
        else:
            r = solve_ik(Pose([float(v) for v in pos], quat, BASE), seed_joints=seed)
    except Exception:
        return None
    if not r.ok:
        return None
    try:
        dg = r.diagnostics
        pt = IK_POS_TOL if ptol is None else ptol
        ot = IK_ORI_TOL if otol is None else otol
        if float(dg["position_error_m"]) > pt or float(dg["orientation_error_rad"]) > ot:
            return None
    except Exception:
        pass
    return [float(v) for v in r.joint_positions]


def max_jump(qa, qb):
    return max(abs(float(qa[m]) - float(qb[m])) for m in range(7))


def step_chain(chain, last, ticks_each, gripper, emb, name):
    log = []
    for k in range(len(chain)):
        qk = chain[k]
        for j in range(1, ticks_each + 1):
            a = j / ticks_each
            tgt = [last[m] + a * (qk[m] - last[m]) for m in range(7)]
            sres = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=tgt, gripper_position=gripper, embodiment=emb)}))
            report["ticks"] += 1
            if ended(sres):
                report["ended_during"] = [name, k]
                raise episode_ended(name)
        last = qk
        if k % 4 == 0 or k == len(chain) - 1:
            st = get_robot_state()
            log.append({"k": k, "ee": flist(st.end_effector_poses["primary"].position, 4),
                        "g": round(float(st.gripper_positions["primary"]), 3)})
    report[name + "_log"] = log
    return last


BASE = "robot_base"

try:
    # ------------------------------------------------------------------ observe
    ctx = get_task_context()
    report["language"] = str(ctx.language)
    md = robomme_plus.get_task_metadata()
    try:
        report["max_steps"] = int(md["max_steps"])
    except Exception as exc:
        note_error("max_steps", exc)
    state0 = get_robot_state()
    BASE = state0.base_frame
    report["live_state"] = {
        "joints": flist(state0.joint_positions["primary"], 4),
        "ee": flist(state0.end_effector_poses["primary"].position, 4),
        "g": round(float(state0.gripper_positions["primary"]), 4),
    }
    obs0 = get_observation()
    fcam = obs0.cameras["front"]
    fmodel = cam_model(fcam)
    report["front_model"] = {"fx": fmodel["fx"], "cx": fmodel["cx"], "p": fmodel["p"]}
    live_rgb = rows_of(fcam.rgb, 1)
    live_depth = rows_of(fcam.depth_m, 1)
    images = {"live_front": encode([row[::2] for row in live_rgb[::2]])}
    report["images"] = images

    # --- video timeline (joints, gripper, ee pose incl. orientation)
    video = robomme_plus.get_conditioning_video()
    nvid = len(video)
    report["video_len"] = nvid
    ts = []
    gs = []
    ees = []
    eqs = []
    qs = []
    for i in range(nvid):
        fr = video[i]
        rs = fr.robot_state
        ts.append(round(float(fr.timestamp_s), 4))
        gs.append(round(float(rs.gripper_positions["primary"]), 4))
        ees.append(flist(rs.end_effector_poses["primary"].position, 4))
        eqs.append(flist(rs.end_effector_poses["primary"].quaternion_wxyz, 4))
        qs.append(flist(rs.joint_positions["primary"], 4))
    report["video_timeline"] = {"t": ts, "g": gs, "ee": ees, "quat": eqs, "q": qs}
    vimgs = []
    sel = []
    for k in range(N_VIDEO_IMAGES):
        idx = int(round(k * (nvid - 1) / (N_VIDEO_IMAGES - 1))) if nvid > 1 else 0
        if nvid > 0 and idx not in sel:
            sel.append(idx)
    for idx in sel:
        fr = video[idx]
        vimgs.append({"idx": idx, "t": ts[idx],
                      "front": encode(rows_of(fr.cameras["front"].rgb, 4)),
                      "wrist": encode(rows_of(fr.cameras["wrist"].rgb, 4))})
    images["video"] = vimgs
    report["stage"] = "video_read"

    # --- classify the manner from the video's robot state
    vid = {}
    report["video_analysis"] = vid
    close_i = None
    for i in range(1, nvid):
        if gs[i - 1] > 0.6 and gs[i] < gs[i - 1] - 0.05:
            close_i = i - 1
            break
    held = None
    if close_i is not None:
        after = [gs[j] for j in range(close_i + 3, min(nvid, close_i + 23))]
        held = quant(after, 0.5) if after else None
    vid["close_i"] = close_i
    vid["held_width"] = held
    grasped = close_i is not None and held is not None and held > 0.08
    vid["grasped"] = grasped
    grasp_color = None
    if grasped:
        gp = ees[close_i]
        vid["grasp_pos"] = gp
        vid["grasp_quat"] = eqs[close_i]
        f0 = video[0].cameras["front"]
        vmodel = cam_model(f0)
        vid["video_front_p"] = vmodel["p"]
        uv = project(vmodel, gp)
        vid["grasp_uv_video0"] = [round(uv[0], 1), round(uv[1], 1)] if uv is not None else None
        if uv is not None:
            v0rgb = rows_of(f0.rgb, 1)
            samples = []
            for dv in (-1, 0, 1):
                for du in (-1, 0, 1):
                    vv = int(round(uv[1])) + dv
                    uu = int(round(uv[0])) + du
                    if 0 <= vv < len(v0rgb) and 0 <= uu < len(v0rgb[0]):
                        samples.append(v0rgb[vv][uu])
            if samples:
                grasp_color = [quant([float(s[k]) for s in samples], 0.5) for k in range(3)]
        vid["grasp_color"] = grasp_color
        # drag phase: still holding, low, after having lifted at least 5 cm
        lifted = None
        for j in range(close_i, nvid):
            if ees[j][2] > gp[2] + 0.05:
                lifted = j
                break
        drag = []
        if lifted is not None:
            for j in range(lifted, nvid):
                if gs[j] > 0.08 and ees[j][2] < gp[2] + 0.02:
                    drag.append(j)
        vid["drag_n"] = len(drag)
        if len(drag) >= 2:
            a = ees[drag[0]]
            b = ees[drag[-1]]
            vid["drag_len"] = round(math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2), 4)
            pitches = []
            pulls = []
            for j in drag:
                m = quat_to_mat(eqs[j])
                xz = m[2][0]
                pitches.append(math.degrees(math.asin(max(-1.0, min(1.0, abs(xz))))))
            vid["drag_pitch_deg"] = round(quant(pitches, 0.5), 2)
            vid["drag_dz"] = round(quant([ees[j][2] for j in drag], 0.5) - gp[2], 4)
            # motion direction relative to hand x (pull: moving against the tool direction)
            m0 = quat_to_mat(eqs[drag[0]])
            hx = [m0[0][0], m0[1][0]]
            if m0[2][0] > 0:
                hx = [-hx[0], -hx[1]]     # make hx point to the tipped-down (tool) side
            mv = [b[0] - a[0], b[1] - a[1]]
            vid["drag_dot_toolaxis"] = round((mv[0] * hx[0] + mv[1] * hx[1]) / max(1e-6, vid["drag_len"]), 3)
    report["stage"] = "video_classified"

    # --- live perception
    grid = deproject_grid(fmodel, live_depth)
    comps = components(grid, live_rgb, ABOVE_TABLE, 0.35)
    keep = []
    for c in comps:
        if len(c["pts"]) < 12:
            continue
        sm = comp_summary(c)
        if sm["x"][0] < 0.08:      # robot base / arm
            continue
        c["sum"] = sm
        keep.append(c)
    report["components"] = [c["sum"] for c in keep]

    cube_cands = seg_cands("cube", BASE)
    cube = None
    for e in cube_cands:
        if e.get("n", 0) < 20:
            continue
        if 0.015 <= e["span"] <= 0.08 and 0.02 <= e["z95"] <= 0.09:
            if cube is None or e["score"] > cube["score"]:
                cube = e
    report["cube_cands"] = cube_cands
    if cube is None:
        raise ValueError("cube not found")
    cube_comp = None
    for c in keep:
        d = math.sqrt((c["sum"]["x"][1] - cube["cx"]) ** 2 + (c["sum"]["y"][1] - cube["cy"]) ** 2)
        if d < 0.04 and (cube_comp is None or len(c["pts"]) > len(cube_comp["pts"])):
            cube_comp = c
    if cube_comp is not None:
        cxs = [p[0] for p in cube_comp["pts"]]
        cys = [p[1] for p in cube_comp["pts"]]
        czs = [p[2] for p in cube_comp["pts"]]
        cube_top = quant(czs, 0.95)
        cube_xy = [(quant(cxs, 0.05) + quant(cxs, 0.95)) / 2.0, (quant(cys, 0.05) + quant(cys, 0.95)) / 2.0]
    else:
        cube_top = cube["z95"]
        cube_xy = [cube["cx"], cube["cy"]]
    cube_w = max(0.02, min(0.08, cube_top))
    # Front camera sees the camera-facing faces: shift the centre half a width away from it.
    away = [cube_xy[0] - fmodel["p"][0], cube_xy[1] - fmodel["p"][1]]
    an = math.sqrt(away[0] ** 2 + away[1] ** 2)
    if cube_comp is None and an > 1e-6:
        cube_xy = [cube_xy[0] + 0.5 * cube_w * away[0] / an * 0.5, cube_xy[1] + 0.5 * cube_w * away[1] / an * 0.5]
    report["cube"] = {"xy": [round(cube_xy[0], 4), round(cube_xy[1], 4)], "top": round(cube_top, 4), "w": round(cube_w, 4)}

    target = None
    target_cands = seg_cands("disc", BASE) + seg_cands("target", BASE)
    for e in target_cands:
        if e.get("n", 0) < 10:
            continue
        dc = math.sqrt((e["cx"] - cube_xy[0]) ** 2 + (e["cy"] - cube_xy[1]) ** 2)
        if e["z95"] <= 0.02 and 0.015 <= e["span"] <= 0.12 and dc > 0.03:
            if target is None or e["score"] > target["score"]:
                target = e
    report["target_cands"] = target_cands
    if target is None:
        raise ValueError("target not found")
    target_comp = None
    for c in keep:
        d = math.sqrt((c["sum"]["x"][1] - target["cx"]) ** 2 + (c["sum"]["y"][1] - target["cy"]) ** 2)
        if d < 0.04 and c["sum"]["z95"] < 0.025 and (target_comp is None or len(c["pts"]) > len(target_comp["pts"])):
            target_comp = c
    if target_comp is not None:
        txs = [p[0] for p in target_comp["pts"]]
        tys = [p[1] for p in target_comp["pts"]]
        target_xy = [(min(txs) + max(txs)) / 2.0, (min(tys) + max(tys)) / 2.0]
    else:
        target_xy = [target["cx"], target["cy"]]
    report["target"] = {"xy": [round(target_xy[0], 4), round(target_xy[1], 4)], "score": target["score"]}

    dxy = [target_xy[0] - cube_xy[0], target_xy[1] - cube_xy[1]]
    dist_ct = math.sqrt(dxy[0] ** 2 + dxy[1] ** 2)
    u_dir = [dxy[0] / dist_ct, dxy[1] / dist_ct]
    report["pull"] = {"dist": round(dist_ct, 4), "u": [round(u_dir[0], 4), round(u_dir[1], 4)]}

    manner = "push"
    tool = None
    if grasped:
        manner = "tool"
        if grasp_color is not None:
            # The grasped part's top must lie just above the video grasp height (fingers
            # close around it); among such components, most pixels of the grasp-point colour.
            gz_v = vid["grasp_pos"][2]
            best = None
            cand_log = []
            for c in keep:
                if not (gz_v - 0.005 <= c["sum"]["z95"] <= gz_v + GRASP_TOP_MAX):
                    continue
                cnt = 0
                for col in c["cols"]:
                    if color_dist(col, grasp_color) < COLOR_TOL:
                        cnt += 1
                cand_log.append([c["sum"]["x"][1], c["sum"]["y"][1], cnt])
                if cnt >= 5 and (best is None or cnt > best[0]):
                    best = (cnt, c)
            report["grasp_match_log"] = cand_log
            if best is not None:
                tool = best[1]
                report["grasped_comp"] = {"match": best[0], "sum": tool["sum"]}
                if tool is cube_comp or tool["sum"]["x"][2] - tool["sum"]["x"][0] < 0.09 and tool["sum"]["y"][2] - tool["sum"]["y"][0] < 0.09:
                    manner = "pick"
    report["manner"] = manner
    report["stage"] = "perceived"

    # ------------------------------------------------------------------ act
    if manner == "tool":
        if tool is None:
            raise ValueError("tool component not found")
        hp = []
        for k in range(len(tool["pts"])):
            if color_dist(tool["cols"][k], grasp_color) < COLOR_TOL:
                hp.append(tool["pts"][k])
        hx = quant([p[0] for p in hp], 0.5)
        hy = quant([p[1] for p in hp], 0.5)
        htop = quant([p[2] for p in hp], 0.95)
        rel = [(p[0] - hx, p[1] - hy) for p in tool["pts"]]
        dists = [math.sqrt(a * a + b * b) for a, b in rel]
        length = max(dists)
        mid = [rel[k] for k in range(len(rel)) if 0.25 * length < dists[k] < 0.75 * length]
        sx = sum(a for a, b in mid) / len(mid)
        sy = sum(b for a, b in mid) / len(mid)
        sn = math.sqrt(sx * sx + sy * sy)
        s_ax = [sx / sn, sy / sn]
        n_ax = [-s_ax[1], s_ax[0]]
        tn = [(a * s_ax[0] + b * s_ax[1], a * n_ax[0] + b * n_ax[1]) for a, b in rel]
        shaft_end = max(t for t, n in tn if abs(n) < 0.012)
        head = [(t, n) for t, n in tn if t > shaft_end - 0.06 and abs(n) > 0.015]
        if len(head) < 5:
            raise ValueError("tool head not found")
        sigma = 1.0 if sum(n for t, n in head) > 0 else -1.0
        head = [(t, sigma * n) for t, n in head]
        n_max = max(n for t, n in head)
        bar = [t for t, n in head if 0.35 * n_max < n < 0.8 * n_max]
        bar_inner = quant(bar, 0.1) if bar else shaft_end - 0.005
        lip = [n for t, n in head if t < bar_inner - 0.012 and n > 0.5 * n_max]
        lip_inner = quant(lip, 0.1) if lip else n_max - 0.005
        lip_ts = [t for t, n in head if n > 0.5 * n_max]
        lip_start = quant(lip_ts, 0.05) if lip_ts else bar_inner - 0.03
        inner_side = [n for t, n in head if bar_inner - 0.04 < t < bar_inner - 0.008 and n < 0.5 * n_max]
        shaft_side = quant(inner_side, 0.9) if inner_side else 0.01
        n_c = sigma * 0.5 * (shaft_side + lip_inner)
        t_contact = bar_inner - 0.5 * cube_w
        geo = {"handle": [round(hx, 4), round(hy, 4), round(htop, 4)], "s": [round(s_ax[0], 4), round(s_ax[1], 4)],
               "length": round(length, 4), "shaft_end": round(shaft_end, 4), "sigma": sigma,
               "n_max": round(n_max, 4), "bar_inner": round(bar_inner, 4), "lip_inner": round(lip_inner, 4),
               "shaft_side": round(shaft_side, 4), "n_c": round(n_c, 4), "t_contact": round(t_contact, 4),
               "lip_start": round(lip_start, 4),
               "n_handle_pts": len(hp)}
        report["tool_geo"] = geo

        vgz = vid["grasp_pos"][2]
        grasp_z = min(max(vgz, htop - 0.012), htop - 0.003)
        pitch = math.radians(min(MAX_PITCH_DEG, vid.get("drag_pitch_deg", 0.0)))
        drag_z = grasp_z + max(0.0, min(0.015, vid.get("drag_dz", 0.0)))
        sd = [-u_dir[0], -u_dir[1]]                 # tool axis during the pull
        nd = [-sd[1], sd[0]]
        # descend with the cube's far face clear of the bar AND of the lip's open end
        # (extra 3 cm: a tool hanging slightly in the grip reaches less far when it touches down)
        start_t = min(t_contact - CLEARANCE, lip_start - 0.5 * cube_w - 0.03)
        g0 = [cube_xy[0] - start_t * sd[0] - n_c * nd[0], cube_xy[1] - start_t * sd[1] - n_c * nd[1]]
        g1 = [target_xy[0] - t_contact * sd[0] - n_c * nd[0] + PULL_EXTRA * u_dir[0],
              target_xy[1] - t_contact * sd[1] - n_c * nd[1] + PULL_EXTRA * u_dir[1]]
        pull_len = math.sqrt((g1[0] - g0[0]) ** 2 + (g1[1] - g0[1]) ** 2)
        geo["g0"] = [round(g0[0], 4), round(g0[1], 4)]
        geo["g1"] = [round(g1[0], 4), round(g1[1], 4)]
        geo["grasp_z"] = round(grasp_z, 4)
        geo["drag_z"] = round(drag_z, 4)
        geo["pitch_deg"] = round(math.degrees(pitch), 2)
        geo["pull_len"] = round(pull_len, 4)

        # Choose the grasp yaw (hand x = +s or -s); the pull yaw follows the tool.
        options = []
        for flip in (1.0, -1.0):
            gyaw = math.atan2(flip * s_ax[1], flip * s_ax[0])
            dyaw = math.atan2(flip * sd[1], flip * sd[0])
            gq = down_quat(gyaw, 0.0)
            dq = down_quat(dyaw, flip * pitch)
            hq = down_quat(dyaw, 0.0)
            ok_all = True
            info = {"flip": flip}
            ik_g = solve_ik(Pose([hx, hy, grasp_z], gq, BASE))
            info["grasp"] = bool(ik_g.ok)
            seed = ik_g.joint_positions if ik_g.ok else None
            ik_0 = solve_ik(Pose([g0[0], g0[1], drag_z], dq, BASE), seed_joints=seed)
            info["g0"] = bool(ik_0.ok)
            ik_1 = solve_ik(Pose([g1[0], g1[1], drag_z], dq, BASE),
                            seed_joints=ik_0.joint_positions if ik_0.ok else None)
            info["g1"] = bool(ik_1.ok)
            if ik_g.ok:
                info["qg"] = [float(v) for v in ik_g.joint_positions]
            if ik_1.ok:
                info["q1"] = [float(v) for v in ik_1.joint_positions]
            ok_all = bool(ik_g.ok) and bool(ik_0.ok) and bool(ik_1.ok)
            info["ok"] = ok_all
            options.append(info)
            if ok_all:
                break
        report["yaw_options"] = [{k: v for k, v in o.items() if k not in ("qg", "q1")} for o in options]
        chosen = None
        for info in options:
            if info["ok"]:
                chosen = info
                break
        if chosen is None:
            chosen = options[0]
            note_error("ik", "no yaw option solved all poses; trying the first")
        flip = chosen["flip"]
        gyaw = math.atan2(flip * s_ax[1], flip * s_ax[0])
        dyaw = math.atan2(flip * sd[1], flip * sd[0])
        gq = down_quat(gyaw, 0.0)
        dq = down_quat(dyaw, flip * pitch)
        hq = down_quat(dyaw, 0.0)
        report["stage"] = "tool_planned"

        # grasp the handle
        res = open_gripper()
        record_motion("open", res)
        go_pose("pre_grasp", [hx, hy, grasp_z + 0.10], gq, 0.01, 200)
        go_pose("grasp_descend", [hx, hy, grasp_z], gq, 0.006, 150)
        res = close_gripper()
        rec = record_motion("close", res)
        if ended(res):
            raise episode_ended("close")
        hold_ticks(5, 0.0)
        width = float(get_robot_state().gripper_positions["primary"])
        report["held_width_live"] = round(width, 4)
        if width < 0.05:
            raise ValueError("tool grasp missed, width " + str(round(width, 4)))
        go_pose("lift", [hx, hy, TRANSIT_Z], gq, 0.015, 250)
        go_pose("transit", [g0[0], g0[1], TRANSIT_Z], gq, 0.015, 300)
        # Re-orient the held tool in place above g0. The head sweeps a circle of radius `length`
        # around the hand: turn in the direction that keeps it farthest from the robot base.
        dyaw_total = dyaw - gyaw
        while dyaw_total > math.pi:
            dyaw_total -= 2 * math.pi
        while dyaw_total < -math.pi:
            dyaw_total += 2 * math.pi
        alt = dyaw_total - 2 * math.pi if dyaw_total > 0 else dyaw_total + 2 * math.pi
        sweep = []
        tool_ang0 = math.atan2(s_ax[1], s_ax[0])
        for total in (dyaw_total, alt):
            clear = None
            for k in range(25):
                ang = tool_ang0 + total * k / 24.0
                hxp = g0[0] + length * math.cos(ang)
                hyp = g0[1] + length * math.sin(ang)
                dd = math.sqrt(hxp * hxp + hyp * hyp)
                clear = dd if clear is None or dd < clear else clear
            sweep.append([total, clear])
        turn = sweep[0][0] if sweep[0][1] >= sweep[1][1] else sweep[1][0]
        report["turn"] = {"deg": round(math.degrees(turn), 1),
                          "options": [[round(math.degrees(a), 1), round(b, 3)] for a, b in sweep]}
        for k in range(1, ROT_INCREMENTS + 1):
            yk = gyaw + turn * k / ROT_INCREMENTS
            go_pose("turn", [g0[0], g0[1], TRANSIT_Z], down_quat(yk, 0.0), 0.02, 200)
        go_pose("turn_final", [g0[0], g0[1], TRANSIT_Z], hq, 0.012, 150)
        pre_z = drag_z + PRE_DZ
        report["stage"] = "planning_drag"

        # v02: plan the drag in joint space BACKWARD from the drag end, so the whole pull stays on
        # one continuous IK branch that reaches g1 (v01's forward chain from the configuration
        # left by move_to_pose ran joint 7 into its limit two thirds of the way).
        nsteps = max(1, int(math.ceil(pull_len / DRAG_STEP)))
        drag_wps = []
        for k in range(nsteps + 1):
            a = k / nsteps
            drag_wps.append([g0[0] + a * (g1[0] - g0[0]), g0[1] + a * (g1[1] - g0[1]), drag_z])
        q_now = [float(v) for v in get_robot_state().joint_positions["primary"]]
        anchors = [["precheck_g1", chosen.get("q1")], ["current", q_now], ["grasp", chosen.get("qg")], ["none", None]]
        plan = None
        plan_log = []
        for name, anchor in anchors:
            if anchor is None and name != "none":
                continue
            q_end = ik_good(drag_wps[-1], dq, anchor)
            if q_end is None:
                plan_log.append([name, "end_fail"])
                continue
            back = [q_end]
            fail = None
            for k in range(nsteps - 1, -1, -1):
                q = ik_good(drag_wps[k], dq, back[-1])
                if q is None or max_jump(q, back[-1]) > MAX_CHAIN_JUMP:
                    fail = ["drag", k]
                    break
                back.append(q)
            asc = []
            asc_z = []
            if fail is None:
                prevq = back[-1]
                for k in range(1, DESCEND_STEPS + 1):
                    z = drag_z + PRE_DZ * k / DESCEND_STEPS
                    q = ik_good([g0[0], g0[1], z], dq, prevq)
                    if q is None or max_jump(q, prevq) > MAX_CHAIN_JUMP:
                        break
                    asc.append(q)
                    asc_z.append(z)
                    prevq = q
                # v02 failed here: the planned branch ends (joint-4 limit) just below +6 cm. Use the
                # highest continuous level, one step lower as a margin if the ascent was cut short.
                use_n = len(asc) if len(asc) == DESCEND_STEPS else len(asc) - 1
                if use_n < MIN_ASCEND_STEPS:
                    fail = ["ascend", len(asc) + 1]
                else:
                    asc = asc[:use_n]
                    asc_z = asc_z[:use_n]
            if fail is not None:
                plan_log.append([name, fail])
                continue
            plan_log.append([name, "ok"])
            descend_chain = list(reversed(asc[:-1])) + [back[-1]]
            plan = {"anchor": name, "q_pre": asc[-1], "descend": descend_chain,
                    "drag": list(reversed(back))[1:], "pre_z": asc_z[-1]}
            break
        report["plan_log"] = plan_log
        transfer = None
        if plan is not None:
            pre_z = plan["pre_z"]
            go_pose("pre_drag", [g0[0], g0[1], pre_z], dq, 0.01, 200)
            q_now = [float(v) for v in get_robot_state().joint_positions["primary"]]
            report["plan"] = {"anchor": plan["anchor"], "q_pre": flist(plan["q_pre"], 3), "pre_z": round(pre_z, 4),
                              "q_end": flist(plan["drag"][-1], 3), "q_now": flist(q_now, 3)}
            # self-motion transfer at the fixed pre-drag pose: IK seeded with joint-space blends
            if max_jump(q_now, plan["q_pre"]) < 0.15:
                transfer = [plan["q_pre"]]
            else:
                fam = []
                prevq = q_now
                for k in range(1, TRANSFER_SAMPLES + 1):
                    a = k / TRANSFER_SAMPLES
                    seedq = [(1 - a) * q_now[m] + a * plan["q_pre"][m] for m in range(7)]
                    q = ik_good([g0[0], g0[1], pre_z], dq, seedq)
                    if q is None or max_jump(q, prevq) > MAX_TRANSFER_JUMP:
                        report["transfer_fail"] = [k, None if q is None else round(max_jump(q, prevq), 3)]
                        fam = None
                        break
                    fam.append(q)
                    prevq = q
                if fam is not None:
                    if max_jump(fam[-1], plan["q_pre"]) > 0.05:
                        fam.append(plan["q_pre"])
                    transfer = fam
            report["transfer_len"] = None if transfer is None else len(transfer)
        emb = get_robot_state().embodiment
        if plan is not None and transfer is not None:
            report["stage"] = "transfer"
            last = q_now
            last = step_chain(transfer, last, TRANSFER_TICKS, 0.0, emb, "transfer")
            hold_ticks(4, 0.0)
            last = step_chain(plan["descend"], last, 4, 0.0, emb, "descend")
            hold_ticks(4, 0.0)
            report["width_before_pull"] = round(float(get_robot_state().gripper_positions["primary"]), 4)
            report["stage"] = "pulling"
            budget = report.get("max_steps", 1339) - report["ticks"] - 60
            tps = max(2, min(TICKS_PER_STEP, int(budget / max(1, len(plan["drag"])))))
            report["pull_ticks_per_step"] = tps
            last = step_chain(plan["drag"], last, tps, 0.0, emb, "pull")
            report["chain_len"] = len(plan["drag"])
        else:
            # fallback (v01 behaviour): forward chain from wherever the arm is
            note_error("plan", "no continuous backward drag plan or transfer; forward chain fallback")
            if plan is None:
                go_pose("pre_drag", [g0[0], g0[1], pre_z], dq, 0.01, 200)
            go_pose("drag_descend", [g0[0], g0[1], drag_z], dq, 0.006, 150)
            report["stage"] = "pulling"
            prevq = [float(v) for v in get_robot_state().joint_positions["primary"]]
            chain = []
            for k in range(1, nsteps + 1):
                # v01 tolerance (public residual limits): v02's strict filter stopped this at k=3
                q = ik_good(drag_wps[k], dq, prevq, 0.02, 0.15)
                if q is None or max_jump(q, prevq) > MAX_CHAIN_JUMP:
                    report["chain_fail"] = k
                    break
                chain.append(q)
                prevq = q
            report["chain_len"] = len(chain)
            last = [float(v) for v in get_robot_state().joint_positions["primary"]]
            last = step_chain(chain, last, TICKS_PER_STEP, 0.0, emb, "pull")
        hold_ticks(8, 0.0)
        report["stage"] = "pulled"
        st = get_robot_state()
        ee = [float(v) for v in st.end_effector_poses["primary"].position]
        report["ee_after_pull"] = flist(ee, 4)
        # lift the tool straight up on the same branch
        prevq = [float(v) for v in st.joint_positions["primary"]]
        lift_chain = []
        for k in range(1, 6):
            q = ik_good([ee[0], ee[1], ee[2] + 0.01 * k], dq, prevq)
            if q is None or max_jump(q, prevq) > MAX_CHAIN_JUMP:
                break
            lift_chain.append(q)
            prevq = q
        report["lift_len"] = len(lift_chain)
        step_chain(lift_chain, [float(v) for v in st.joint_positions["primary"]], 4, 0.0, emb, "lift")
        hold_ticks(10, 0.0)

    elif manner == "pick":
        # yaw from the cube component (minimum-area rectangle over 5-degree steps)
        yaw = 0.0
        if cube_comp is not None:
            best = None
            for k in range(18):
                ang = math.radians(5 * k)
                c = math.cos(ang)
                s = math.sin(ang)
                a_ = [p[0] * c + p[1] * s for p in cube_comp["pts"]]
                b_ = [-p[0] * s + p[1] * c for p in cube_comp["pts"]]
                area = (max(a_) - min(a_)) * (max(b_) - min(b_))
                if best is None or area < best[0]:
                    best = (area, ang)
            yaw = best[1]
        q = down_quat(yaw, 0.0)
        gz = cube_top - 0.5 * cube_w
        res = open_gripper()
        record_motion("open", res)
        go_pose("pre_grasp", [cube_xy[0], cube_xy[1], gz + 0.10], q, 0.01, 200)
        go_pose("grasp_descend", [cube_xy[0], cube_xy[1], gz], q, 0.006, 150)
        res = close_gripper()
        record_motion("close", res)
        if ended(res):
            raise episode_ended("close")
        hold_ticks(5, 0.0)
        go_pose("lift", [cube_xy[0], cube_xy[1], gz + 0.12], q, 0.015, 200)
        go_pose("above_target", [target_xy[0], target_xy[1], gz + 0.12], q, 0.015, 250)
        go_pose("lower", [target_xy[0], target_xy[1], gz + 0.012], q, 0.008, 150)
        res = open_gripper()
        record_motion("release", res)
        if ended(res):
            raise episode_ended("release")
        hold_ticks(8, 1.0)
        go_pose("retreat", [target_xy[0], target_xy[1], gz + 0.12], q, 0.015, 150)
        hold_ticks(10, 1.0)

    else:
        # push with the bare gripper along cube -> target
        gval = 0.0
        if close_i is None and nvid > 0 and gs[0] > 0.5:
            gval = 1.0
        yaw = math.atan2(u_dir[1], u_dir[0]) + math.pi / 2.0
        q = down_quat(yaw, 0.0)
        pz = max(0.012, 0.5 * cube_top)
        back = 0.5 * cube_w + 0.04
        start = [cube_xy[0] - back * u_dir[0], cube_xy[1] - back * u_dir[1]]
        stop_d = 0.5 * cube_w + 0.012
        finish = [target_xy[0] - stop_d * u_dir[0], target_xy[1] - stop_d * u_dir[1]]
        if gval < 0.5:
            res = close_gripper()
            record_motion("close_empty", res)
        go_pose("pre_push", [start[0], start[1], pz + 0.10], q, 0.01, 200)
        go_pose("push_descend", [start[0], start[1], pz], q, 0.006, 150)
        plen = math.sqrt((finish[0] - start[0]) ** 2 + (finish[1] - start[1]) ** 2)
        nsteps = max(1, int(math.ceil(plen / 0.02)))
        for k in range(1, nsteps + 1):
            a = k / nsteps
            go_pose("push_wp", [start[0] + a * (finish[0] - start[0]), start[1] + a * (finish[1] - start[1]), pz], q, 0.006, 40)
        hold_ticks(8, gval)
        go_pose("push_retreat", [finish[0] - 0.03 * u_dir[0], finish[1] - 0.03 * u_dir[1], pz + 0.10], q, 0.015, 150)
        hold_ticks(10, gval)
    report["stage"] = "acted"

    # ------------------------------------------------------------------ re-observe
    fobs = get_observation()
    images["final_front"] = encode(rows_of(fobs.cameras["front"].rgb, 2))
    final = seg_cands("cube", BASE)
    report["final_cube_cands"] = final
    report["stage"] = "done"
except Exception as exc:
    msg = str(exc)
    if msg.startswith(END_TAG):
        report["episode_ended_at"] = msg[len(END_TAG):]
    else:
        note_error("top", exc)

result = report
