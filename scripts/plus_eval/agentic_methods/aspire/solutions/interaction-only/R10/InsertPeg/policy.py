import math

# v04 — InsertPeg: v03 (video imitation + in-hand peg alignment) with a wider insertion-probe grid.
# The conditioning video (robomme_plus.get_conditioning_video(), live episode input) carries the
# demonstrating robot's per-frame joint state, end-effector pose and gripper reading.
# 1. Compare the video's first front RGB-D frame with the live front RGB-D frame around the
#    video's grasp point (peg) and final insertion point (box), using depth only.
# 2. If both objects are where the video had them, replay the video's joint trajectory tick by
#    tick (joint_position steps), closing/opening the gripper when the video's gripper does.
# 3. Otherwise replay the video's end-effector waypoints transformed rigidly: pre-grasp/grasp/lift
#    by the peg's displacement, transport/insertion by the box's displacement.
# 4. v02 (failure in v01 ep0: the replayed EE matched the video within 1 mm, but the peg jammed
#    on the box face 3.4 cm short of the video's final pose): after the lift, measure the held peg's
#    yaw relative to the hand from WRIST depth (live vs the same video frame), follow the video's
#    insertion path with the hand yaw corrected by that difference (IK + joint interpolation),
#    and, if still short, retract 12 mm and re-push with small lateral/vertical/yaw offsets.
# 6. v04 (failure in v03 ep0: in-hand pose matched the video within 0.2 mm / 0.2 deg, yet every
#    +-8 mm ring probe met a flat wall at the box face plane): probe a wider/lower/higher grid.
# 5. v03 (failure in v02 ep0: every re-push stopped at the box face at the same depth; the wrist
#    view showed the hole opening beside the peg tip): measure the held peg's full in-hand pose
#    (yaw, lateral offset of its axis, height) from wrist depth at the INSERTION START frame, live
#    vs the same video frame, and shift/rotate the insertion path so the live peg follows the
#    video peg. If still short, press lightly against the face and scan lateral/vertical offsets
#    in growing rings (then with +-3 deg yaw); when the peg advances, push to the video's depth.

report = {"stage": "start", "notes": []}
result = report


def note(msg):
    report["notes"].append(str(msg))


def r4(values):
    return [round(float(v), 4) for v in values]


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def qz(angle):
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def rotmat(q):
    w, x, y, z = (float(v) for v in q)
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


def backproject(cam, stride):
    """Front-camera depth -> robot_base points (OpenCV pinhole: z forward, y down)."""
    depth = cam.depth_m
    k = cam.intrinsics
    fx = float(k[0][0])
    fy = float(k[1][1])
    cx = float(k[0][2])
    cy = float(k[1][2])
    rm = rotmat(cam.camera_pose.quaternion_wxyz)
    p = [float(v) for v in cam.camera_pose.position]
    h = len(depth)
    w = len(depth[0])
    pts = []
    for vv in range(0, h, stride):
        row = depth[vv]
        for uu in range(0, w, stride):
            d = float(row[uu])
            if d <= 0.0 or d > 3.0:
                continue
            a = (uu - cx) / fx * d
            b = (vv - cy) / fy * d
            x = rm[0][0] * a + rm[0][1] * b + rm[0][2] * d + p[0]
            y = rm[1][0] * a + rm[1][1] * b + rm[1][2] * d + p[1]
            z = rm[2][0] * a + rm[2][1] * b + rm[2][2] * d + p[2]
            pts.append((x, y, z))
    return pts


def region(pts, cxy, radius, zlo, zhi):
    out = []
    r2 = radius * radius
    for q in pts:
        if q[2] < zlo or q[2] > zhi:
            continue
        dx = q[0] - cxy[0]
        dy = q[1] - cxy[1]
        if dx * dx + dy * dy <= r2:
            out.append(q)
    return out


def centroid_axis(pts):
    n = len(pts)
    if n < 8:
        return None
    mx = sum(q[0] for q in pts) / n
    my = sum(q[1] for q in pts) / n
    mz = sum(q[2] for q in pts) / n
    sxx = sum((q[0] - mx) * (q[0] - mx) for q in pts) / n
    syy = sum((q[1] - my) * (q[1] - my) for q in pts) / n
    sxy = sum((q[0] - mx) * (q[1] - my) for q in pts) / n
    ang = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    return {"c": (mx, my, mz), "ang": ang, "n": n}


def track(pts, start_xy, radius, zlo, zhi):
    """Re-centre a radius search a few times so a displaced object is followed."""
    cur = start_xy
    info = None
    for it in range(3):
        sel = region(pts, cur, radius, zlo, zhi)
        info = centroid_axis(sel)
        if info is None:
            return None
        cur = (info["c"][0], info["c"][1])
    return info


def wrap_half(a):
    while a > math.pi / 2.0:
        a -= math.pi
    while a < -math.pi / 2.0:
        a += math.pi
    return a


def transform_pose(pos, quat, center, dyaw, shift):
    c = math.cos(dyaw)
    s = math.sin(dyaw)
    rx = pos[0] - center[0]
    ry = pos[1] - center[1]
    nx = center[0] + c * rx - s * ry + shift[0]
    ny = center[1] + s * rx + c * ry + shift[1]
    nz = pos[2] + shift[2]
    nq = qmul(qz(dyaw), quat)
    return (nx, ny, nz), nq


def ended(res):
    try:
        return bool(res.terminated) or bool(res.truncated)
    except Exception:
        return False



def hand_yaw(q):
    hx = qrot(q, (1.0, 0.0, 0.0))
    return math.atan2(hx[1], hx[0])


def qrot(q, v):
    w, x, y, z = q
    r = qmul(qmul((w, x, y, z), (0.0, v[0], v[1], v[2])), (w, -x, -y, -z))
    return (r[1], r[2], r[3])


def wrap_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def held_peg(cam, ee_pos, ee_q):
    """Held peg's long side seen by the wrist camera: points 3-13 cm from the tool point
    horizontally, within 3 cm of its height. Returns yaw relative to the hand x-axis."""
    pts = backproject(cam, 2)
    sx = 0.0
    sy = 0.0
    sz = 0.0
    cnt = 0
    for q in pts:
        dz = q[2] - ee_pos[2]
        if dz < -0.03 or dz > 0.03:
            continue
        dx = q[0] - ee_pos[0]
        dy = q[1] - ee_pos[1]
        rr = math.sqrt(dx * dx + dy * dy)
        if rr < 0.03 or rr > 0.13:
            continue
        sx += dx / rr
        sy += dy / rr
        sz += dz
        cnt += 1
    if cnt < 12:
        return {"n": cnt}
    yaw = math.atan2(sy, sx)
    return {"n": cnt, "yaw": yaw, "rel": wrap_pi(yaw - hand_yaw(ee_q)), "dz": sz / cnt,
            "spread": math.sqrt(sx * sx + sy * sy) / cnt}


def peg_pose(cam, ee_pos, ee_q):
    """Held peg in hand-relative terms from wrist depth: yaw of its axis relative to the hand
    x-axis, signed perpendicular distance of the axis from the tool point (left normal), height of
    its visible surface above the tool point, and the far-end projection along the axis."""
    pts = backproject(cam, 1)
    sel = []
    sx = 0.0
    sy = 0.0
    for q in pts:
        dz = q[2] - ee_pos[2]
        if dz < -0.03 or dz > 0.03:
            continue
        dx = q[0] - ee_pos[0]
        dy = q[1] - ee_pos[1]
        rr = math.sqrt(dx * dx + dy * dy)
        if rr < 0.025 or rr > 0.11:
            continue
        sel.append((dx, dy, dz))
        sx += dx / rr
        sy += dy / rr
    if len(sel) < 30:
        return None
    ang = math.atan2(sy, sx)
    core = []
    mx = 0.0
    my = 0.0
    for it in range(3):
        # straight band along the current axis estimate (no radial clipping bias), then PCA
        ca = math.cos(ang)
        sa = math.sin(ang)
        core = []
        for p in sel:
            al = p[0] * ca + p[1] * sa
            ac = -p[0] * sa + p[1] * ca
            if al > 0.03 and al < 0.09 and abs(ac) < 0.018:
                core.append(p)
        if len(core) < 30:
            return None
        m = len(core)
        mx = sum(p[0] for p in core) / m
        my = sum(p[1] for p in core) / m
        sxx = sum((p[0] - mx) ** 2 for p in core) / m
        syy = sum((p[1] - my) ** 2 for p in core) / m
        sxy = sum((p[0] - mx) * (p[1] - my) for p in core) / m
        a1 = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
        if math.cos(a1 - ang) < 0.0:
            a1 += math.pi
        ang = a1
    m = len(core)
    ca = math.cos(ang)
    sa = math.sin(ang)
    lat = -mx * sa + my * ca
    along = sorted(p[0] * ca + p[1] * sa for p in core)
    zs = sorted(p[2] for p in core)
    return {"n": m, "yaw": ang, "rel": wrap_pi(ang - hand_yaw(ee_q)), "lat": lat,
            "ztop": zs[int(0.8 * (m - 1))], "zmed": zs[int(0.5 * (m - 1))],
            "tip": along[int(0.98 * (m - 1))]}


def step_to(jt, grip, embodiment):
    return step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=jt, gripper_position=grip, embodiment=embodiment)}))


def drive(jfrom, jto, ticks, grip, embodiment):
    last = None
    for t in range(1, ticks + 1):
        a = t / float(ticks)
        last = step_to([jfrom[j] + (jto[j] - jfrom[j]) * a for j in range(7)], grip, embodiment)
        if ended(last):
            return last, True
    return last, False


try:
    context = get_task_context()
    report["language"] = context.language
    meta = robomme_plus.get_task_metadata()
    max_steps = int(meta["max_steps"])
    report["max_steps"] = max_steps
    state0 = get_robot_state()
    base_frame = state0.base_frame
    embodiment = state0.embodiment

    # ---------------- conditioning video ----------------
    video = robomme_plus.get_conditioning_video()
    n = len(video)
    report["video_frames"] = n
    if n < 10:
        raise ValueError("conditioning video too short")
    vj = []
    vpos = []
    vq = []
    vg = []
    vt = []
    for i in range(n):
        rs = video[i].robot_state
        vj.append([float(v) for v in rs.joint_positions["primary"]])
        ee = rs.end_effector_poses["primary"]
        vpos.append([float(v) for v in ee.position])
        vq.append(tuple(float(v) for v in ee.quaternion_wxyz))
        vg.append(float(rs.gripper_positions["primary"]))
        vt.append(float(video[i].timestamp_s))
    report["video_dt"] = round((vt[n - 1] - vt[0]) / (n - 1), 4)
    grasp_idx = None
    for i in range(1, n):
        if vg[i] < 0.6 and vg[i - 1] >= 0.6:
            grasp_idx = i
            break
    if grasp_idx is None:
        raise ValueError("no gripper close in video")
    release_idx = None
    for i in range(grasp_idx + 1, n):
        if vg[i] > 0.6:
            release_idx = i
            break
    report["video_grasp_idx"] = grasp_idx
    report["video_release_idx"] = release_idx
    G = vpos[grasp_idx]
    F = vpos[n - 1]
    report["video_grasp_ee"] = r4(G)
    report["video_final_ee"] = r4(F)

    # ---------------- object displacement video[0] -> live ----------------
    vpts = backproject(video[0].cameras["front"], 2)
    obs = get_observation()
    lpts = backproject(obs.cameras["front"], 2)
    peg_v = track(vpts, (G[0], G[1]), 0.06, 0.004, 0.035)
    box_v = track(vpts, (F[0], F[1]), 0.12, 0.03, 0.20)
    peg_l = None
    box_l = None
    if peg_v is not None:
        peg_l = track(lpts, (peg_v["c"][0], peg_v["c"][1]), 0.06, 0.004, 0.035)
    if box_v is not None:
        box_l = track(lpts, (box_v["c"][0], box_v["c"][1]), 0.12, 0.03, 0.20)
    if peg_v is None or box_v is None or peg_l is None or box_l is None:
        raise ValueError("could not find peg/box near the video grasp/insert points")
    dpeg = [peg_l["c"][k] - peg_v["c"][k] for k in range(3)]
    dbox = [box_l["c"][k] - box_v["c"][k] for k in range(3)]
    dyaw_peg = wrap_half(peg_l["ang"] - peg_v["ang"])
    report["peg_video"] = {"c": r4(peg_v["c"]), "ang": round(peg_v["ang"], 4), "n": peg_v["n"]}
    report["peg_live"] = {"c": r4(peg_l["c"]), "ang": round(peg_l["ang"], 4), "n": peg_l["n"]}
    report["box_video"] = {"c": r4(box_v["c"]), "n": box_v["n"]}
    report["box_live"] = {"c": r4(box_l["c"]), "n": box_l["n"]}
    report["dpeg"] = r4(dpeg)
    report["dbox"] = r4(dbox)
    report["dyaw_peg_deg"] = round(dyaw_peg * 180.0 / math.pi, 2)
    same_layout = (math.sqrt(dpeg[0] ** 2 + dpeg[1] ** 2) < 0.01 and
                   math.sqrt(dbox[0] ** 2 + dbox[1] ** 2) < 0.01 and
                   abs(dyaw_peg) < 4.0 * math.pi / 180.0)
    report["same_layout"] = same_layout


    # insertion start: last video frame more than 7 cm from the final pose
    k_ins = n - 1
    while k_ins > grasp_idx and math.sqrt(sum((vpos[k_ins][j] - F[j]) ** 2 for j in range(3))) < 0.07:
        k_ins -= 1
    T = None
    for i in range(grasp_idx + 1, k_ins):
        if vpos[i][2] - G[2] > 0.10:
            T = i
            break
    report["insert_start_idx"] = k_ins
    shift = [0.0, 0.0, 0.0]
    report["inhand_check_idx"] = T
    delta = 0.0
    if same_layout:
        # ---------------- joint replay up to the insertion start ----------------
        report["stage"] = "joint_replay"
        last = None
        for i in range(1, k_ins + 1):
            look = min(i + 1, n - 1)
            grip = 0.0 if vg[look] < 0.9 and look >= grasp_idx and (release_idx is None or look < release_idx) else 1.0
            last = step_to(vj[i], grip, embodiment)
            if ended(last):
                report["ended_at_replay_frame"] = i
                report["episode_ended"] = True
                break
            if T is not None and i == T:
                try:
                    vv = held_peg(video[T].cameras["wrist"], vpos[T], vq[T])
                    so = get_observation()
                    ls = so.robot_state
                    lv = held_peg(so.cameras["wrist"], [float(v) for v in ls.end_effector_poses["primary"].position],
                                  tuple(float(v) for v in ls.end_effector_poses["primary"].quaternion_wxyz))
                    rv = {}
                    for kk in vv:
                        rv[kk] = round(float(vv[kk]), 4)
                    rl = {}
                    for kk in lv:
                        rl[kk] = round(float(lv[kk]), 4)
                    report["inhand_video"] = rv
                    report["inhand_live"] = rl
                    if "rel" in vv and "rel" in lv and vv["spread"] > 0.8 and lv["spread"] > 0.8:
                        dl = wrap_pi(lv["rel"] - vv["rel"])
                        if abs(dl) < 12.0 * math.pi / 180.0:
                            delta = dl
                except Exception as exc:
                    note("inhand " + str(exc))
        report["inhand_delta_deg_lift"] = round(delta * 180.0 / math.pi, 2)
        delta = 0.0
        shift = [0.0, 0.0, 0.0]
        if not report.get("episode_ended", False):
            # ---------- in-hand peg pose at the insertion start: live vs video ----------
            try:
                pv = peg_pose(video[k_ins].cameras["wrist"], vpos[k_ins], vq[k_ins])
                so = get_observation()
                ls = so.robot_state
                le = [float(v) for v in ls.end_effector_poses["primary"].position]
                lq = tuple(float(v) for v in ls.end_effector_poses["primary"].quaternion_wxyz)
                pl = peg_pose(so.cameras["wrist"], le, lq)
                report["ee_err_at_kins"] = r4([le[j] - vpos[k_ins][j] for j in range(3)])
                if pv is not None:
                    rv = {}
                    for kk in pv:
                        rv[kk] = round(float(pv[kk]), 4)
                    report["peg_kins_video"] = rv
                if pl is not None:
                    rl = {}
                    for kk in pl:
                        rl[kk] = round(float(pl[kk]), 4)
                    report["peg_kins_live"] = rl
                if pv is not None and pl is not None:
                    th = wrap_pi(pv["rel"] - pl["rel"])
                    dlat = pv["lat"] - pl["lat"]
                    dzc = pv["ztop"] - pl["ztop"]
                    ok_th = abs(th) < 10.0 * math.pi / 180.0
                    ok_lat = abs(dlat) < 0.015
                    ok_z = abs(dzc) < 0.015
                    report["align_raw"] = {"dyaw_deg": round(th * 180.0 / math.pi, 2), "dlat": round(dlat, 4),
                                           "dz": round(dzc, 4)}
                    if ok_th:
                        delta = -th
                    nvx = -math.sin(pv["yaw"])
                    nvy = math.cos(pv["yaw"])
                    if ok_lat:
                        shift[0] = dlat * nvx
                        shift[1] = dlat * nvy
                    if ok_z:
                        shift[2] = dzc
            except Exception as exc:
                note("peg_pose " + str(exc))
        report["inhand_delta_deg"] = round(delta * 180.0 / math.pi, 2)
        report["align_shift"] = r4(shift)
        if not report.get("episode_ended", False):
            # ---------------- corrected insertion along the video path ----------------
            report["stage"] = "corrected_insert"
            s = get_robot_state()
            jcur = [float(v) for v in s.joint_positions["primary"]]
            corr = qz(-delta)
            idxs = list(range(k_ins, n, 3))
            if idxs[len(idxs) - 1] != n - 1:
                idxs.append(n - 1)
            first = True
            for i in idxs:
                tp = [vpos[i][j] + shift[j] for j in range(3)]
                ik = solve_ik(Pose(tp, qmul(corr, vq[i]), base_frame), seed_joints=jcur)
                if not ik.ok:
                    note("ik fail at video frame " + str(i))
                    continue
                jn = [float(v) for v in ik.joint_positions]
                last, stop = drive(jcur, jn, 10 if first else 3, 0.0, embodiment)
                first = False
                jcur = jn
                if stop:
                    report["ended_at_insert_frame"] = i
                    report["episode_ended"] = True
                    break
            if not report.get("episode_ended", False):
                for tick in range(5):
                    last = step_to(jcur, 0.0, embodiment)
                    if ended(last):
                        report["episode_ended"] = True
                        break
    else:
        # ---------------- transformed Cartesian replay ----------------
        report["stage"] = "cartesian_replay"
        lift_idx = n - 1
        for i in range(grasp_idx, n):
            if vpos[i][2] > G[2] + 0.15:
                lift_idx = i
                break
        report["video_lift_idx"] = lift_idx
        dyaw_box = 0.0
        open_gripper()
        idxs = list(range(5, grasp_idx, 8)) + [grasp_idx]
        idxs += list(range(grasp_idx + 8, n, 8)) + [n - 1]
        closed = False
        stopped = False
        for i in idxs:
            if i <= lift_idx:
                pos, q = transform_pose(vpos[i], vq[i], peg_v["c"], dyaw_peg, dpeg)
            else:
                pos, q = transform_pose(vpos[i], vq[i], box_v["c"], dyaw_box, dbox)
            tol = 0.004 if (i >= grasp_idx - 10 and i <= grasp_idx) or i > lift_idx else 0.01
            res = move_to_pose(Pose(pos, q, base_frame), tolerance=tol, max_steps=60)
            if ended(res):
                report["ended_at_waypoint"] = i
                report["episode_ended"] = True
                stopped = True
                break
            if i == grasp_idx and not closed:
                cg = close_gripper()
                closed = True
                report["close_status"] = str(cg.status)
                if ended(cg):
                    report["episode_ended"] = True
                    stopped = True
                    break
        report["cartesian_done"] = not stopped
    report["stage"] = "done"
except Exception as exc:
    report["error"] = str(exc)

def along(pos, origin, d):
    return sum((pos[j] - origin[j]) * d[j] for j in range(3))


try:
    # ---------------- press-and-scan insertion search ----------------
    if report.get("stage") == "done" and not report.get("episode_ended", False):
        report["stage"] = "search"
        if same_layout:
            F_t = [F[j] + shift[j] for j in range(3)]
        else:
            F_b, unused_q = transform_pose(F, vq[n - 1], box_v["c"], 0.0, dbox)
            F_t = [F_b[j] + shift[j] for j in range(3)]
        k = k_ins
        dv = [F[j] - vpos[k][j] for j in range(3)]
        dn = math.sqrt(sum(v * v for v in dv))
        d = [v / dn for v in dv]
        hn = math.sqrt(d[0] * d[0] + d[1] * d[1])
        perp = [-d[1] / hn, d[0] / hn, 0.0]
        report["insert_dir"] = r4(d)
        deg = math.pi / 180.0
        # v04: v03's +-8 mm rings all met a flat wall at the face plane, so probe farther out:
        # lower first (towards the box's mid-height), then wider sideways, then higher.
        offsets = [(0.0, 0.0, 0.0)]
        for zz in (-0.011, -0.014, -0.017, -0.020, -0.024):
            for ll in (0.0, 0.004, -0.004):
                offsets.append((ll, zz, 0.0))
        for zz in (0.0, -0.005):
            for ll in (0.011, -0.011, 0.015, -0.015):
                offsets.append((ll, zz, 0.0))
        for zz in (0.011, 0.014):
            for ll in (0.0, 0.004, -0.004):
                offsets.append((ll, zz, 0.0))
        s = get_robot_state()
        cur = [float(v) for v in s.end_effector_poses["primary"].position]
        remaining = along(F_t, cur, d)
        contact = along(cur, F_t, d)
        report["remaining_after_insert"] = round(remaining, 4)
        tries = []
        found = remaining < 0.004
        for off in offsets:
            if found:
                break
            s = get_robot_state()
            ticks_used = int(round(float(s.timestamp_s) * 20.0))
            if ticks_used + 16 > max_steps:
                tries.append({"out_of_ticks": True, "ticks": ticks_used})
                break
            q_t = qmul(qz(-delta + off[2]), vq[n - 1])
            back = [F_t[j] + d[j] * (contact - 0.004) + perp[j] * off[0] for j in range(3)]
            back[2] += off[1]
            push = [F_t[j] + d[j] * (contact + 0.005) + perp[j] * off[0] for j in range(3)]
            push[2] += off[1]
            jcur = [float(v) for v in s.joint_positions["primary"]]
            ik_b = solve_ik(Pose(back, q_t, base_frame), seed_joints=jcur)
            if not ik_b.ok:
                tries.append({"off": r4(off), "ik": "fail"})
                continue
            jb = [float(v) for v in ik_b.joint_positions]
            ik_p = solve_ik(Pose(push, q_t, base_frame), seed_joints=jb)
            if not ik_p.ok:
                tries.append({"off": r4(off), "ik": "fail"})
                continue
            jp = [float(v) for v in ik_p.joint_positions]
            last, stop = drive(jcur, jb, 5, 0.0, embodiment)
            if not stop:
                last, stop = drive(jb, jp, 6, 0.0, embodiment)
            if not stop:
                last, stop = drive(jp, jp, 1, 0.0, embodiment)
            if stop:
                report["episode_ended"] = True
                tries.append({"off": r4(off), "episode_ended": True})
                break
            s2 = get_robot_state()
            cur2 = [float(v) for v in s2.end_effector_poses["primary"].position]
            adv = along(cur2, F_t, d) - contact
            entry = {"off": r4(off), "adv": round(adv, 4)}
            if adv > 0.004:
                # the peg entered: push on to the video's insertion depth along the same offset
                deep = [F_t[j] + d[j] * 0.004 + perp[j] * off[0] for j in range(3)]
                deep[2] += off[1]
                ik_d = solve_ik(Pose(deep, q_t, base_frame), seed_joints=jp)
                if ik_d.ok:
                    jd = [float(v) for v in ik_d.joint_positions]
                    last, stop = drive(jp, jd, 20, 0.0, embodiment)
                    if not stop:
                        last, stop = drive(jd, jd, 5, 0.0, embodiment)
                    if stop:
                        report["episode_ended"] = True
                        entry["episode_ended"] = True
                        tries.append(entry)
                        break
                    s3 = get_robot_state()
                    cur3 = [float(v) for v in s3.end_effector_poses["primary"].position]
                    entry["remaining_after_deep"] = round(along(F_t, cur3, d), 4)
                    if along(F_t, cur3, d) < 0.006:
                        found = True
                    else:
                        contact = along(cur3, F_t, d)
            tries.append(entry)
        report["search"] = tries
        report["inserted"] = found
        if found and not report.get("episode_ended", False):
            s = get_robot_state()
            jh = [float(v) for v in s.joint_positions["primary"]]
            for tick in range(10):
                last = step_to(jh, 0.0, embodiment)
                if ended(last):
                    report["episode_ended"] = True
                    break
        if not report.get("episode_ended", False):
            report["stage"] = "search_done"
except Exception as exc:
    report["error_search"] = str(exc)

try:
    s = get_robot_state()
    report["final_ee"] = r4(s.end_effector_poses["primary"].position)
    report["final_gripper"] = round(float(s.gripper_positions["primary"]), 3)
    report["final_t"] = float(s.timestamp_s)
except Exception as exc:
    note("final state " + str(exc))

result = report
