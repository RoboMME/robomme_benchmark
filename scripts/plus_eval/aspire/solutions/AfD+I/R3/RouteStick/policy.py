import math

# RouteStick v00 -- reproduce the conditioning video's stick route in the live scene.
# Strategy (all task-dependent values derived at runtime):
#  1. Read the episode's conditioning video: per-frame robot joints + end-effector pose
#     (this is the route the video drew) and the first frame's front RGB-D.
#  2. Detect the waypoint discs (small light blobs at table height, surrounded by table colour)
#     in the video's first frame and in the live front image, project them to robot_base with depth.
#  3. Order discs along the row, build a smooth local similarity map video-scene -> live-scene
#     anchored on corresponding discs.
#  4. Map the video's end-effector path into the live scene; solve IK at keypoints (seeded with
#     the video's own joints) and stream joint targets at 20 Hz = video joints + IK correction.

LEAD_S = 0.05            # command slightly ahead of the recorded (measured) state
KEY_SPACING_S = 0.30     # IK keypoint spacing along the video time axis
NEGLIGIBLE_SHIFT_M = 0.004
HOLD_EVERY_IK = 30

report = {"version": "v00_initial", "stage": "start"}
base_frame = get_robot_state().base_frame
report["goal"] = get_task_context().language

try:
    meta = robomme_plus.get_task_metadata()
    report["task_metadata"] = {}
    for key in meta:
        try:
            value = meta[key]
            report["task_metadata"][str(key)] = value if (value is None or str(value) == value or float(value) == value) else str(value)
        except Exception:
            report["task_metadata"][str(key)] = str(meta[key])
except Exception as exc:
    meta = {}
    report["task_metadata_error"] = str(exc)

max_steps = None
try:
    max_steps = int(meta["max_steps"])
except Exception:
    max_steps = None
report["max_steps"] = max_steps


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


def to_rows(array):
    try:
        return array.tolist()
    except Exception:
        return [list(row) for row in array]


def camera_model(cam):
    k = to_rows(cam.intrinsics)
    pose = cam.camera_pose
    return {
        "fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]), "cy": float(k[1][2]),
        "R": quat_to_mat(pose.quaternion_wxyz),
        "t": [float(v) for v in pose.position],
    }


def project(model, u, v, depth, convention):
    xc = (u - model["cx"]) / model["fx"] * depth
    yc = (v - model["cy"]) / model["fy"] * depth
    zc = depth
    if convention == "opengl":
        yc = -yc
        zc = -zc
    R = model["R"]
    t = model["t"]
    return [
        R[0][0] * xc + R[0][1] * yc + R[0][2] * zc + t[0],
        R[1][0] * xc + R[1][1] * yc + R[1][2] * zc + t[1],
        R[2][0] * xc + R[2][1] * yc + R[2][2] * zc + t[2],
    ]


def median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[len(ordered) // 2]


def is_table(p):
    return p[0] > p[1] + 25 and p[1] > p[2] + 5


def light_components(rgb, height, width):
    cand = [[False] * width for r in range(height)]
    for v in range(height):
        row = rgb[v]
        for u in range(width):
            p = row[u]
            lo = min(p[0], p[1], p[2])
            hi = max(p[0], p[1], p[2])
            if lo >= 125 and hi - lo <= 45:
                cand[v][u] = True
    seen = [[False] * width for r in range(height)]
    comps = []
    for v in range(height):
        for u in range(width):
            if cand[v][u] and not seen[v][u]:
                seen[v][u] = True
                stack = [(v, u)]
                pix = []
                while stack:
                    a, b = stack.pop()
                    pix.append((a, b))
                    if len(pix) > 400:
                        continue
                    for da in (-1, 0, 1):
                        for db in (-1, 0, 1):
                            c = a + da
                            d = b + db
                            if 0 <= c < height and 0 <= d < width and cand[c][d] and not seen[c][d]:
                                seen[c][d] = True
                                stack.append((c, d))
                comps.append(pix)
    return comps


def detect_discs(cam, label):
    """Small light blobs surrounded by table colour at table height -> 3D centres (both conventions)."""
    rgb = to_rows(cam.rgb)
    depth = to_rows(cam.depth_m)
    height = len(rgb)
    width = len(rgb[0])
    model = camera_model(cam)
    info = {"label": label, "size": [height, width]}
    out = {}
    for convention in ("opencv", "opengl"):
        table_z = []
        for v in range(0, height, 4):
            for u in range(0, width, 4):
                d = float(depth[v][u])
                if d > 0.05 and d < 5.0 and is_table(rgb[v][u]):
                    table_z.append(project(model, u, v, d, convention)[2])
        out[convention] = {"table_z": median(table_z), "table_n": len(table_z)}
    info["camera_z"] = model["t"][2]
    # choose convention: camera must look DOWN onto the table
    chosen = None
    for convention in ("opencv", "opengl"):
        tz = out[convention]["table_z"]
        if tz is not None and tz < model["t"][2] - 0.1:
            chosen = convention if chosen is None else chosen
    info["convention_by_geometry"] = chosen
    info["table_z_by_convention"] = {"opencv": out["opencv"]["table_z"], "opengl": out["opengl"]["table_z"]}
    comps = light_components(rgb, height, width)
    blobs = []
    for pix in comps:
        n = len(pix)
        if n < 4 or n > 300:
            continue
        vs = [p[0] for p in pix]
        us = [p[1] for p in pix]
        v0, v1, u0, u1 = min(vs), max(vs), min(us), max(us)
        if v1 - v0 > 25 or u1 - u0 > 25:
            continue
        ring = 0
        ring_table = 0
        for v in range(max(0, v0 - 3), min(height, v1 + 4)):
            for u in range(max(0, u0 - 3), min(width, u1 + 4)):
                if v0 <= v <= v1 and u0 <= u <= u1:
                    continue
                ring += 1
                if is_table(rgb[v][u]):
                    ring_table += 1
        frac = ring_table / float(max(1, ring))
        blobs.append({"u": sum(us) / float(n), "v": sum(vs) / float(n), "n": n,
                      "ring_table": round(frac, 2), "pix": pix})
    info["blobs_raw"] = len(blobs)
    return info, blobs, model, depth


def blob_points(blob, model, depth, convention):
    pts = []
    for (v, u) in blob["pix"]:
        d = float(depth[v][u])
        if d > 0.05 and d < 5.0:
            pts.append(project(model, u, v, d, convention))
    if not pts:
        return None
    return [median([p[0] for p in pts]), median([p[1] for p in pts]), median([p[2] for p in pts])]


def finalize_discs(info, blobs, model, depth, convention):
    table_z = info["table_z_by_convention"][convention]
    discs = []
    rejected = []
    for blob in blobs:
        c = blob_points(blob, model, depth, convention)
        entry = {"u": round(blob["u"], 1), "v": round(blob["v"], 1), "n": blob["n"], "ring_table": blob["ring_table"]}
        if c is None:
            entry["why"] = "no depth"
            rejected.append(entry)
            continue
        entry["xyz"] = [round(c[0], 4), round(c[1], 4), round(c[2], 4)]
        if table_z is None or abs(c[2] - table_z) > 0.03:
            entry["why"] = "height"
            rejected.append(entry)
            continue
        if blob["ring_table"] < 0.4:
            entry["why"] = "surround"
            rejected.append(entry)
            continue
        discs.append({"xyz": c, "entry": entry})
    # merge fragments closer than 2.5 cm
    merged = []
    for disc in discs:
        joined = False
        for m in merged:
            dx = m["xyz"][0] - disc["xyz"][0]
            dy = m["xyz"][1] - disc["xyz"][1]
            if math.sqrt(dx * dx + dy * dy) < 0.025:
                w0 = m["n"]
                w1 = disc["entry"]["n"]
                m["xyz"] = [(m["xyz"][i] * w0 + disc["xyz"][i] * w1) / float(w0 + w1) for i in range(3)]
                m["n"] = w0 + w1
                joined = True
                break
        if not joined:
            merged.append({"xyz": list(disc["xyz"]), "n": disc["entry"]["n"], "u": disc["entry"]["u"], "v": disc["entry"]["v"]})
    info["rejected"] = rejected[:20]
    return merged


def row_axis(points):
    mx = sum(p[0] for p in points) / float(len(points))
    my = sum(p[1] for p in points) / float(len(points))
    sxx = sum((p[0] - mx) ** 2 for p in points)
    syy = sum((p[1] - my) ** 2 for p in points)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    angle = 0.5 * math.atan2(2 * sxy, sxx - syy)
    ax = [math.cos(angle), math.sin(angle)]
    if ax[1] < 0:
        ax = [-ax[0], -ax[1]]
    return ax, [mx, my]


def order_discs(discs):
    pts = [d["xyz"] for d in discs]
    ax, center = row_axis(pts)
    keyed = []
    for i in range(len(discs)):
        s = (pts[i][0] - center[0]) * ax[0] + (pts[i][1] - center[1]) * ax[1]
        keyed.append((s, i))
    keyed.sort()
    return [discs[i] for (s, i) in keyed], ax


def wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


# ---------------------------------------------------------------- conditioning video
video = robomme_plus.get_conditioning_video()
report["video_frames"] = len(video)
if len(video) < 2:
    report["stage"] = "no_video"
    result = report
else:
    times = []
    joints = []
    ee_pos = []
    ee_quat = []
    for frame in video:
        rs = frame.robot_state
        times.append(float(frame.timestamp_s))
        joints.append([float(v) for v in rs.joint_positions["primary"]])
        pose = rs.end_effector_poses["primary"]
        ee_pos.append([float(v) for v in pose.position])
        ee_quat.append([float(v) for v in pose.quaternion_wxyz])
    t0 = times[0]
    times = [t - t0 for t in times]
    ranges = [max(p[i] for p in ee_pos) - min(p[i] for p in ee_pos) for i in range(3)]
    report["video_duration_s"] = round(times[-1], 3)
    report["video_ee_range_m"] = [round(r, 4) for r in ranges]
    report["video_ee_z_min"] = round(min(p[2] for p in ee_pos), 4)
    report["video_first_joints"] = [round(v, 4) for v in joints[0]]
    report["video_first_ee"] = [round(v, 4) for v in ee_pos[0]] + [round(v, 4) for v in ee_quat[0]]

    # trim stationary start / end
    first_move = 0
    for i in range(len(joints)):
        if max(abs(joints[i][j] - joints[0][j]) for j in range(7)) > 0.002:
            first_move = max(0, i - 1)
            break
    last_move = len(joints) - 1
    for i in range(len(joints) - 1, -1, -1):
        if max(abs(joints[i][j] - joints[-1][j]) for j in range(7)) > 0.002:
            last_move = min(len(joints) - 1, i + 1)
            break
    report["video_active_window"] = [first_move, last_move, round(times[first_move], 2), round(times[last_move], 2)]

    live_state = get_robot_state()
    live_q = [float(v) for v in live_state.joint_positions["primary"]]
    report["live_start_joints"] = [round(v, 4) for v in live_q]
    report["live_vs_video_start_joint_diff"] = round(max(abs(live_q[j] - joints[0][j]) for j in range(7)), 4)
    lee = live_state.end_effector_poses["primary"]
    report["live_start_ee"] = [round(float(v), 4) for v in lee.position] + [round(float(v), 4) for v in lee.quaternion_wxyz]

    # ------------------------------------------------------------ disc detection
    def hold_tick():
        st = get_robot_state()
        return step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=st.joint_positions["primary"],
            gripper_position=st.gripper_positions["primary"], embodiment=st.embodiment)}))

    def ik_solution(ik):
        try:
            return [float(v) for v in ik.joint_positions]
        except Exception:
            return [float(v) for v in ik.solution]

    use_ik = False
    key_idx = [first_move]
    key_dq = [None]
    hold_ticks = 0
    vaxis = [0.0, 1.0]
    try:
        report["stage"] = "detect"
        vinfo, vblobs, vmodel, vdepth = detect_discs(video[0].cameras["front"], "video0")
        live_obs = get_observation()
        linfo, lblobs, lmodel, ldepth = detect_discs(live_obs.cameras["front"], "live")
        convention = linfo["convention_by_geometry"] or vinfo["convention_by_geometry"] or "opengl"
        report["convention"] = convention
        vdiscs = finalize_discs(vinfo, vblobs, vmodel, vdepth, convention)
        ldiscs = finalize_discs(linfo, lblobs, lmodel, ldepth, convention)
        report["video_detect"] = vinfo
        report["live_detect"] = linfo
        for info in (vinfo, linfo):
            info["rejected_n"] = len(info["rejected"])

        # cross-check convention with the public projection tool on one live disc
        try:
            if ldiscs:
                probe = segment_points("front", [[int(round(ldiscs[0]["u"])), int(round(ldiscs[0]["v"]))]])
                if probe.ok and len(probe.segmentations) > 0:
                    cloud = mask_to_point_cloud(probe.segmentations[0], "front", target_frame=base_frame)
                    pts = cloud.points
                    report["convention_check"] = {
                        "tool_median": [round(median([float(p[i]) for p in pts]), 4) for i in range(3)],
                        "ours": [round(v, 4) for v in ldiscs[0]["xyz"]],
                        "n": len(pts),
                    }
        except Exception as exc:
            report["convention_check_error"] = str(exc)

        vdiscs, vaxis = order_discs(vdiscs) if len(vdiscs) >= 2 else (vdiscs, [0.0, 1.0])
        ldiscs, laxis = order_discs(ldiscs) if len(ldiscs) >= 2 else (ldiscs, [0.0, 1.0])
        report["video_discs"] = [[round(v, 4) for v in d["xyz"]] + [d["u"], d["v"]] for d in vdiscs]
        report["live_discs"] = [[round(v, 4) for v in d["xyz"]] + [d["u"], d["v"]] for d in ldiscs]

        # route decoding (diagnostic): nearest video disc to the EE path, in xy
        visits = []
        if vdiscs:
            for i in range(len(ee_pos)):
                best = None
                for k in range(len(vdiscs)):
                    dx = ee_pos[i][0] - vdiscs[k]["xyz"][0]
                    dy = ee_pos[i][1] - vdiscs[k]["xyz"][1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if best is None or dist < best[0]:
                        best = (dist, k)
                if best[0] < 0.03 and (not visits or visits[-1][0] != best[1]):
                    visits.append([best[1], round(times[i], 2), round(best[0], 3)])
        report["video_visits_xy3cm"] = visits

        # ------------------------------------------------------------ scene map
        mapping_ok = len(vdiscs) >= 2 and len(vdiscs) == len(ldiscs)
        report["mapping_ok"] = mapping_ok
        nd = len(vdiscs)
        disc_tf = []
        max_shift = 0.0
        dz = 0.0
        if mapping_ok:
            seg = []
            for k in range(nd - 1):
                a = vdiscs[k]["xyz"]
                b = vdiscs[k + 1]["xyz"]
                c = ldiscs[k]["xyz"]
                d = ldiscs[k + 1]["xyz"]
                ang = wrap(math.atan2(d[1] - c[1], d[0] - c[0]) - math.atan2(b[1] - a[1], b[0] - a[0]))
                lv = math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
                ll = math.sqrt((d[0] - c[0]) ** 2 + (d[1] - c[1]) ** 2)
                seg.append((ang, ll / max(1e-6, lv)))
            for k in range(nd):
                parts = []
                if k > 0:
                    parts.append(seg[k - 1])
                if k < nd - 1:
                    parts.append(seg[k])
                ang = sum(p[0] for p in parts) / float(len(parts))
                scale = sum(p[1] for p in parts) / float(len(parts))
                disc_tf.append({"ang": ang, "scale": scale, "src": vdiscs[k]["xyz"], "dst": ldiscs[k]["xyz"]})
                sx = ldiscs[k]["xyz"][0] - vdiscs[k]["xyz"][0]
                sy = ldiscs[k]["xyz"][1] - vdiscs[k]["xyz"][1]
                max_shift = max(max_shift, math.sqrt(sx * sx + sy * sy))
            dz = sum(ldiscs[k]["xyz"][2] - vdiscs[k]["xyz"][2] for k in range(nd)) / float(nd)
            report["disc_transforms"] = [[round(tf["ang"] * 180 / math.pi, 2), round(tf["scale"], 3)] for tf in disc_tf]
            report["max_disc_shift_m"] = round(max_shift, 4)
            report["mean_dz_m"] = round(dz, 4)
        elif len(vdiscs) >= 1 and len(ldiscs) >= 1:
            # degraded: centroid translation only
            vc = [sum(d["xyz"][i] for d in vdiscs) / float(len(vdiscs)) for i in range(3)]
            lc = [sum(d["xyz"][i] for d in ldiscs) / float(len(ldiscs)) for i in range(3)]
            disc_tf = [{"ang": 0.0, "scale": 1.0, "src": vc, "dst": lc}]
            max_shift = math.sqrt((lc[0] - vc[0]) ** 2 + (lc[1] - vc[1]) ** 2)
            dz = lc[2] - vc[2]
            report["degraded_mapping"] = "centroid translation"
            report["max_disc_shift_m"] = round(max_shift, 4)

        def apply_tf(tf, p):
            ca = math.cos(tf["ang"]) * tf["scale"]
            sa = math.sin(tf["ang"]) * tf["scale"]
            x = p[0] - tf["src"][0]
            y = p[1] - tf["src"][1]
            return [tf["dst"][0] + ca * x - sa * y, tf["dst"][1] + sa * x + ca * y]

        def map_point(p):
            if not disc_tf:
                return [p[0], p[1], p[2]], 0.0
            if len(disc_tf) == 1:
                q = apply_tf(disc_tf[0], p)
                return [q[0], q[1], p[2] + dz], disc_tf[0]["ang"]
            ax = vaxis
            s = [(tf["src"][0] * ax[0] + tf["src"][1] * ax[1]) for tf in disc_tf]
            sp = p[0] * ax[0] + p[1] * ax[1]
            k = 0
            while k < len(s) - 2 and sp > s[k + 1]:
                k += 1
            f = (sp - s[k]) / max(1e-6, s[k + 1] - s[k])
            f = min(1.0, max(0.0, f))
            qa = apply_tf(disc_tf[k], p)
            qb = apply_tf(disc_tf[k + 1], p)
            ang = (1 - f) * disc_tf[k]["ang"] + f * disc_tf[k + 1]["ang"]
            return [(1 - f) * qa[0] + f * qb[0], (1 - f) * qa[1] + f * qb[1], p[2] + dz], ang

        # ------------------------------------------------------------ IK corrections at keypoints
        report["stage"] = "ik"
        dt_v = times[-1] / float(len(times) - 1)
        stride = max(1, int(round(KEY_SPACING_S / max(1e-3, dt_v))))
        key_idx = list(range(first_move, last_move + 1, stride))
        if key_idx[-1] != last_move:
            key_idx.append(last_move)
        use_ik = mapping_ok and max_shift >= NEGLIGIBLE_SHIFT_M
        if not mapping_ok and disc_tf and max_shift >= NEGLIGIBLE_SHIFT_M:
            use_ik = True
        report["use_ik"] = use_ik
        report["ik_keypoints"] = len(key_idx)
        key_dq = []
        ik_fail = 0
        ik_calls = 0
        jumps = 0
        selfcheck = []
        hold_ticks = 0


        diff_method = False
        if use_ik:
            for i in (key_idx[0], key_idx[len(key_idx) // 2], key_idx[-1]):
                ik = solve_ik(Pose(ee_pos[i], ee_quat[i], base_frame), seed_joints=joints[i])
                ik_calls += 1
                if ik.ok:
                    sol = ik_solution(ik)
                    selfcheck.append(round(max(abs(sol[j] - joints[i][j]) for j in range(7)), 4))
                else:
                    selfcheck.append(None)
            report["ik_selfcheck_max_rad"] = selfcheck
            for value in selfcheck:
                if value is None or value > 0.03:
                    diff_method = True
            report["ik_diff_method"] = diff_method
            prev = [0.0] * 7
            for n_key in range(len(key_idx)):
                i = key_idx[n_key]
                if ik_calls > 0 and ik_calls % HOLD_EVERY_IK == 0:
                    hold_tick()
                    hold_ticks += 1
                mapped, ang = map_point(ee_pos[i])
                qz = [math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2)]
                quat = quat_mul(qz, ee_quat[i])
                seed = [joints[i][j] + prev[j] for j in range(7)]
                ik = solve_ik(Pose(mapped, quat, base_frame), seed_joints=seed)
                ik_calls += 1
                if not ik.ok:
                    ik_fail += 1
                    key_dq.append(None)
                    continue
                sol = ik_solution(ik)
                ref = joints[i]
                if diff_method:
                    ik0 = solve_ik(Pose(ee_pos[i], ee_quat[i], base_frame), seed_joints=joints[i])
                    ik_calls += 1
                    if ik0.ok:
                        ref = ik_solution(ik0)
                dq = [sol[j] - ref[j] for j in range(7)]
                if max(abs(dq[j] - prev[j]) for j in range(7)) > 0.3:
                    jumps += 1
                key_dq.append(dq)
                prev = dq
        report["ik_calls"] = ik_calls
        report["ik_fail"] = ik_fail
        report["ik_jumps_gt_0p3"] = jumps
        report["hold_ticks"] = hold_ticks

    except Exception as exc:
        report["map_or_ik_error"] = str(exc)[:300]
        report["fallback"] = "direct replay of video joints (identity map)"
        use_ik = False
        key_idx = [first_move]
        key_dq = [None]

    # fill failed keypoints by neighbours
    kts = []
    kdq = []
    for n_key in range(len(key_idx)):
        if use_ik and key_dq[n_key] is not None:
            kts.append(times[key_idx[n_key]])
            kdq.append(key_dq[n_key])
    if not kts:
        kts = [times[first_move]]
        kdq = [[0.0] * 7]
    if kdq:
        report["dq_max_abs_rad"] = round(max(max(abs(v) for v in dq) for dq in kdq), 4)

    def interp_joints(t):
        if t <= times[0]:
            return list(joints[0])
        if t >= times[-1]:
            return list(joints[-1])
        lo = 0
        hi = len(times) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if times[mid] <= t:
                lo = mid
            else:
                hi = mid
        f = (t - times[lo]) / max(1e-9, times[hi] - times[lo])
        return [joints[lo][j] + f * (joints[hi][j] - joints[lo][j]) for j in range(7)]

    def interp_dq(t):
        if t <= kts[0]:
            return kdq[0]
        if t >= kts[-1]:
            return kdq[-1]
        k = 0
        while k < len(kts) - 2 and t > kts[k + 1]:
            k += 1
        f = (t - kts[k]) / max(1e-9, kts[k + 1] - kts[k])
        return [kdq[k][j] + f * (kdq[k + 1][j] - kdq[k][j]) for j in range(7)]

    # ------------------------------------------------------------ execute
    report["stage"] = "execute"
    t_start = times[first_move]
    t_end = times[last_move]
    span = t_end - t_start
    n_ticks = int(math.ceil(span / 0.05)) + 1
    time_scale = 1.0
    budget = None
    if max_steps is not None:
        budget = max_steps - hold_ticks - 40
        if n_ticks > budget and budget > 50:
            time_scale = n_ticks / float(budget)
            n_ticks = budget
    report["exec_ticks_planned"] = n_ticks
    report["time_scale"] = round(time_scale, 3)

    first_target = [interp_joints(t_start)[j] + interp_dq(t_start)[j] for j in range(7)]
    cur = [float(v) for v in get_robot_state().joint_positions["primary"]]
    report["first_target_gap_rad"] = round(max(abs(first_target[j] - cur[j]) for j in range(7)), 4)
    terminated = False
    truncated = False
    status_counts = {}
    errors = []
    if report["first_target_gap_rad"] > 0.05:
        pre = move_to_joints(first_target, tolerance=0.02, max_steps=150)
        try:
            report["pre_move"] = {"ok": bool(pre.ok), "steps": int(pre.steps_executed), "status": str(pre.status)}
            if pre.terminated or pre.truncated:
                terminated = bool(pre.terminated)
                truncated = bool(pre.truncated)
        except Exception as exc:
            report["pre_move_error"] = str(exc)
    executed = 0
    track_err = []
    state = get_robot_state()
    gripper = state.gripper_positions["primary"]
    embodiment = state.embodiment
    for tick in range(n_ticks):
        if terminated or truncated:
            break
        t = t_start + tick * 0.05 * time_scale
        tv = min(t_end, t + LEAD_S * time_scale)
        qv = interp_joints(tv)
        dq = interp_dq(t)
        target = [qv[j] + dq[j] for j in range(7)]
        try:
            reply = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=target, gripper_position=gripper, embodiment=embodiment)}))
        except Exception as exc:
            errors.append(str(exc)[:200])
            break
        executed += 1
        try:
            terminated = bool(reply.terminated)
            truncated = bool(reply.truncated)
        except Exception as exc:
            if len(errors) < 3:
                errors.append("flags:" + str(exc)[:120])
        if tick % 20 == 0 and not (terminated or truncated):
            try:
                mq = [float(v) for v in reply.observation.robot_state.joint_positions["primary"]]
                ref = [interp_joints(t)[j] + dq[j] for j in range(7)]
                track_err.append(round(max(abs(mq[j] - ref[j]) for j in range(7)), 3))
            except Exception:
                pass
    report["exec_ticks_done"] = executed
    report["track_err_sample"] = track_err[:60]
    # settle a few ticks at the final pose if still running
    settle = 0
    while not (terminated or truncated) and settle < 15:
        try:
            reply = hold_tick()
            settle += 1
            terminated = bool(reply.terminated)
            truncated = bool(reply.truncated)
        except Exception as exc:
            errors.append("settle:" + str(exc)[:120])
            break
    report["settle_ticks"] = settle
    report["terminated"] = terminated
    report["truncated"] = truncated
    report["errors"] = errors
    if not (terminated or truncated):
        try:
            fin = get_robot_state()
            report["final_ee"] = [round(float(v), 4) for v in fin.end_effector_poses["primary"].position]
            report["final_time_s"] = float(fin.timestamp_s)
        except Exception as exc:
            report["final_error"] = str(exc)[:120]
    report["stage"] = "done"
    result = report
