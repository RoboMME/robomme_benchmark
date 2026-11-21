import math

# InsertPeg v02 -- imitate this episode's own conditioning video, and LOCATE THE HOLE live.
# v01 evidence (ep0): with joints tracking the video and the peg sitting in the hand as in the
# video (~1 mm), the peg tip stopped on the box face at the same depth for all 12 small offsets.
# The wrist views at the same hand pose show the hole opening displaced sideways relative to the
# video. v02 therefore:
#  (a) replays the video joint path (lag-compensated, LEAD frames ahead) up to the video's own
#      pre-contact frame (the first frame after the fast approach where the hand slows down),
#  (b) holds there, so that the live wrist camera sits where the video's wrist camera was, and
#      registers the live wrist depth against the video's wrist depth in a window just beyond the
#      peg tip (the peg footprint itself is masked out, measured from a quiet transport frame),
#  (c) converts the image shift into a metric shift of the hole (per-pixel back-projection with
#      each frame's own camera pose) and inserts along the video's own final insertion line,
#      offset by that shift, with stall detection and a local search; if registration is not
#      trustworthy it falls back to a lateral sweep.
# Every target comes from this episode's conditioning video and live sensing; no coordinates,
# dimensions, pixel locations or seed identities are hard-coded.

LEAD = 2
report = {"version": "v02", "phase": "start"}
TERMINAL = {"ended": False, "how": None}
TICKS = {"n": 0}
EMB = {"name": None}
LIMIT = {"max": None}


def fl(values):
    return [float(v) for v in values]


def vadd(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def vsub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def vscale(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vcross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def vnorm(a):
    return math.sqrt(vdot(a, a))


def vunit(a):
    n = vnorm(a)
    if n < 1e-9:
        return [0.0, 0.0, 0.0]
    return [a[0] / n, a[1] / n, a[2] / n]


def qrot(q, v):
    w, x, y, z = q[0], q[1], q[2], q[3]
    r = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    return [r[i][0] * v[0] + r[i][1] * v[1] + r[i][2] * v[2] for i in range(3)]


def rnd(v, k=4):
    return [round(float(x), k) for x in v]


def check_terminal(reply, label):
    term = False
    trunc = False
    try:
        term = bool(reply.terminated)
    except Exception:
        term = False
    try:
        trunc = bool(reply.truncated)
    except Exception:
        trunc = False
    if term or trunc:
        TERMINAL["ended"] = True
        TERMINAL["how"] = {"label": label, "terminated": term, "truncated": trunc, "tick": TICKS["n"]}
    return term or trunc


def send(q, grip, label):
    if TERMINAL["ended"]:
        return None
    if LIMIT["max"] is not None and TICKS["n"] >= LIMIT["max"] - 1:
        TERMINAL["ended"] = True
        TERMINAL["how"] = {"label": "own_tick_guard", "tick": TICKS["n"]}
        return None
    reply = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=fl(q), gripper_position=float(grip),
        embodiment=EMB["name"])}))
    TICKS["n"] += 1
    check_terminal(reply, label)
    return reply


def max_err(a, b):
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def live_state():
    return get_robot_state()


def live_joints():
    return fl(get_robot_state().joint_positions["primary"])


def hold_until(q, grip, tol, max_ticks, label):
    used = 0
    err = max_err(live_joints(), q)
    while err > tol and used < max_ticks and not TERMINAL["ended"]:
        send(q, grip, label)
        used += 1
        if TERMINAL["ended"]:
            break
        err = max_err(live_joints(), q)
    return {"ticks": used, "final_err": round(err, 5)}


def peg_rows(depth, intr, row_lo, row_hi):
    """Per image row, the longest run of 'near' pixels (closer than 0.6 x the image median depth):
    returns {row: (u_center, z_median, width)}. Measures the held peg in a wrist view."""
    rows = len(depth)
    cols = len(depth[0])
    sample = []
    for r in range(0, rows, 8):
        drow = depth[r]
        for c in range(0, cols, 8):
            z = float(drow[c])
            if z > 0.0:
                sample.append(z)
    sample = sorted(sample)
    if len(sample) < 10:
        return {}, None
    cut = 0.6 * sample[len(sample) // 2]
    out = {}
    for r in range(row_lo, row_hi, 2):
        drow = depth[r]
        best_len = 0
        best_a = -1
        run_a = -1
        for c in range(cols):
            z = float(drow[c])
            near = z > 0.0 and z < cut
            if near:
                if run_a < 0:
                    run_a = c
                length = c - run_a + 1
                if length > best_len:
                    best_len = length
                    best_a = run_a
            else:
                run_a = -1
        if best_len >= 3:
            zs = sorted(float(drow[c]) for c in range(best_a, best_a + best_len))
            out[r] = (best_a + 0.5 * (best_len - 1), zs[len(zs) // 2], best_len)
    return out, cut


def cam_point(u, v, z, intr):
    fx = float(intr[0][0])
    fy = float(intr[1][1])
    cx = float(intr[0][2])
    cy = float(intr[1][2])
    return [(u - cx) / fx * z, (v - cy) / fy * z, z]




def depth_grid(depth):
    return [[float(z) for z in row] for row in depth]


def register_window(vd, ld, rows, cols, mgrid, du_rng, dv_rng, stride_px, stride_sh):
    """Best image shift (du, dv) such that live(u+du, v+dv) matches video(u, v) over the window.
    Cost: mean truncated |depth difference|; requires at least half the window to stay valid."""
    nrow = len(vd)
    ncol = len(vd[0])
    pts = []
    for v in range(rows[0], rows[1], stride_px):
        for u in range(cols[0], cols[1], stride_px):
            if mgrid[v][u]:
                continue
            z = vd[v][u]
            if z > 0.0:
                pts.append((u, v, z))
    need = max(10, len(pts) // 2)
    best = None
    table = {}
    for dv in range(dv_rng[0], dv_rng[1] + 1, stride_sh):
        for du in range(du_rng[0], du_rng[1] + 1, stride_sh):
            tot = 0.0
            cnt = 0
            for (u, v, z) in pts:
                uu = u + du
                vv = v + dv
                if uu < 0 or vv < 0 or uu >= ncol or vv >= nrow:
                    continue
                if mgrid[vv][uu]:
                    continue
                zl = ld[vv][uu]
                if zl <= 0.0:
                    continue
                e = abs(zl - z)
                if e > 0.02:
                    e = 0.02
                tot += e
                cnt += 1
            if cnt < need:
                continue
            c = tot / cnt
            table[(du, dv)] = c
            if best is None or c < best[0]:
                best = (c, du, dv, cnt)
    return best, table, len(pts)


def cam_to_world(cam_pos, cam_q, pc, flip):
    if flip:
        pc = [-pc[0], -pc[1], pc[2]]
    return vadd(cam_pos, qrot(cam_q, pc))


try:
    ctx = get_task_context()
    report["goal"] = str(ctx.language)
    try:
        meta = robomme_plus.get_task_metadata()
        report["meta"] = {str(k): str(meta[k])[:300] for k in meta}
    except Exception as exc:
        meta = {}
        report["meta_error"] = str(exc)[:300]
    hz = 20.0
    for key in ("control_frequency_hz", "control_hz"):
        try:
            if key in meta:
                hz = float(meta[key])
                break
        except Exception:
            pass
    try:
        LIMIT["max"] = int(meta["max_steps"])
    except Exception:
        LIMIT["max"] = None
    report["hz"] = hz
    report["max_steps"] = LIMIT["max"]

    live = get_observation()
    state0 = live.robot_state
    EMB["name"] = state0.embodiment
    live_q0 = fl(state0.joint_positions["primary"])

    video = list(robomme_plus.get_conditioning_video())
    nframes = len(video)
    report["video_frames"] = nframes
    if nframes < 10:
        raise ValueError("conditioning video missing or too short: %d" % nframes)
    vq = []
    vg = []
    vt = []
    vee = []
    vquat = []
    for frame in video:
        rs = frame.robot_state
        vq.append(fl(rs.joint_positions["primary"]))
        vg.append(float(rs.gripper_positions["primary"]))
        vt.append(float(frame.timestamp_s))
        pose = rs.end_effector_poses["primary"]
        vee.append(fl(pose.position))
        vquat.append(fl(pose.quaternion_wxyz))
    report["live_vs_video_q0_err"] = round(max_err(live_q0, vq[0]), 5)

    # ---- scene check (live front depth vs video first frame)
    scene = {}
    try:
        ld0 = live.cameras["front"].depth_m
        vd0 = video[0].cameras["front"].depth_m
        total = 0
        differ = 0
        for r in range(0, len(ld0), 4):
            lrow = ld0[r]
            vrow = vd0[r]
            for c in range(0, len(ld0[0]), 4):
                a = float(lrow[c])
                b = float(vrow[c])
                if a > 0.0 and b > 0.0 and a < 10.0 and b < 10.0:
                    total += 1
                    if abs(a - b) > 0.01:
                        differ += 1
        scene = {"grid_valid": total, "frac_differ_1cm": round(differ / max(1, total), 5),
                 "cam_pose_shift_m": round(max_err(fl(live.cameras["front"].camera_pose.position),
                                                   fl(video[0].cameras["front"].camera_pose.position)), 5)}
    except Exception as exc:
        scene = {"error": str(exc)[:300]}
    report["scene_check"] = scene

    # ---- gripper events (binarized command with hysteresis)
    closed_cmd = []
    is_closed = False
    events = []
    for i in range(nframes):
        if not is_closed:
            if i + 1 < nframes and vg[i + 1] < vg[i] - 0.003:
                horizon = vg[i + 1:min(nframes, i + 25)]
                if len(horizon) > 0 and min(horizon) < vg[i] - 0.1:
                    is_closed = True
                    events.append({"type": "close", "frame": i})
        else:
            if i + 1 < nframes and vg[i + 1] > vg[i] + 0.003:
                horizon = vg[i + 1:min(nframes, i + 25)]
                if len(horizon) > 0 and max(horizon) > vg[i] + 0.1:
                    is_closed = False
                    events.append({"type": "open", "frame": i})
        closed_cmd.append(is_closed)
    report["gripper_events"] = events
    last_event = events[-1]["frame"] if len(events) > 0 else 0
    final_grip = 0.0 if closed_cmd[-1] else 1.0

    # ---- the video's final approach: direction, speed profile, pre-contact frame
    far = None
    for k in range(nframes - 1, last_event, -1):
        if vnorm(vsub(vee[-1], vee[k])) > 0.03:
            far = k
            break
    if far is None:
        raise ValueError("video has no final approach segment")
    d0 = vunit(vsub(vee[-1], vee[far]))
    sv = [vdot(vsub(vee[k], vee[-1]), d0) for k in range(nframes)]
    lo = last_event + 10
    peak_k = None
    for k in range(lo, nframes - 1):
        if sv[k] > -0.25:
            spd = sv[k + 1] - sv[k]
            if peak_k is None or spd > sv[peak_k + 1] - sv[peak_k]:
                peak_k = k
    f_pre = None
    if peak_k is not None:
        pk = sv[peak_k + 1] - sv[peak_k]
        for k in range(peak_k, nframes - 1):
            if sv[k + 1] - sv[k] < 0.35 * pk:
                f_pre = k
                break
    if f_pre is None or f_pre >= nframes - 5:
        f_pre = far
    # insertion line: from the pre-contact pose to the video's final pose
    d = vunit(vsub(vee[-1], vee[f_pre]))
    s_pre = vdot(vsub(vee[f_pre], vee[-1]), d)
    approach = qrot(vquat[-1], [0.0, 0.0, 1.0])
    e2 = vunit(vsub(approach, vscale(d, vdot(approach, d))))
    e1 = vunit(vcross(d, e2))
    dev = 0.0
    for k in range(f_pre, nframes):
        rel = vsub(vee[k], vee[-1])
        dev = max(dev, vnorm(vsub(rel, vscale(d, vdot(rel, d)))))
    report["insert"] = {"far": far, "peak_k": peak_k, "f_pre": f_pre, "s_pre": round(s_pre, 4),
                        "d": rnd(d), "e1": rnd(e1), "e2": rnd(e2), "path_dev_from_line_m": round(dev, 4),
                        "video_ee_pre": rnd(vee[f_pre]), "video_ee_end": rnd(vee[-1])}

    # ---- quiet transport frame (peg footprint in the wrist view, background far away)
    f_ref = None
    if len(events) > 0 and events[-1]["type"] == "close":
        best = None
        for f in range(last_event + 30, f_pre - 10):
            if f - 2 < 0 or f + 2 >= nframes:
                continue
            speed = vnorm(vsub(vee[f + 2], vee[f - 2]))
            if best is None or speed < best:
                best = speed
                f_ref = f
    report["f_ref"] = f_ref

    # ---- replay helper (lag-compensated)
    def replay(i_from, i_to, label):
        worst = 0.0
        for i in range(i_from, i_to):
            if TERMINAL["ended"]:
                break
            grip = 0.0 if closed_cmd[i] else 1.0
            ticks = int(round((vt[i + 1] - vt[i]) * hz))
            if ticks < 1:
                ticks = 1
            j0 = min(i_to, i + LEAD)
            j1 = min(i_to, i + 1 + LEAD)
            for k in range(1, ticks + 1):
                a = k / float(ticks)
                target = [vq[j0][j] + a * (vq[j1][j] - vq[j0][j]) for j in range(7)]
                send(target, grip, label)
                if TERMINAL["ended"]:
                    break
            if i % 10 == 0 and not TERMINAL["ended"]:
                err = max_err(live_joints(), vq[i + 1])
                if err > worst:
                    worst = err
        return round(worst, 4)

    report["phase"] = "replay"
    if max_err(live_joints(), vq[0]) > 0.02:
        move_to_joints(vq[0], tolerance=0.01, max_steps=150)
    report["replay_worst_err"] = replay(0, f_pre, "replay")
    report["hold_pre"] = hold_until(vq[f_pre], final_grip, 0.002, 20, "hold_pre")
    report["ticks_at_pre"] = TICKS["n"]
    if not TERMINAL["ended"]:
        rs = get_robot_state()
        report["live_ee_minus_video_at_pre"] = rnd(vsub(fl(rs.end_effector_poses["primary"].position), vee[f_pre]), 5)

    # ---- hole registration: live wrist depth vs video wrist depth at the pre-contact frame
    est = None
    reg = {}
    if not TERMINAL["ended"] and f_ref is not None:
        try:
            vw_ref = video[f_ref].cameras["wrist"]
            intr = vw_ref.intrinsics
            nrows = len(vw_ref.depth_m)
            ncols = len(vw_ref.depth_m[0])
            foot, cut = peg_rows(vw_ref.depth_m, intr, 0, int(nrows * 0.72))
            if len(foot) < 6:
                raise ValueError("peg footprint not found (%d rows)" % len(foot))
            tip = min(foot)
            upper = sorted(r for r in foot if r <= tip + 40)
            ucs = sorted(foot[r][0] for r in upper)
            wds = sorted(foot[r][2] for r in upper)
            uc = ucs[len(ucs) // 2]
            w = wds[len(wds) // 2]
            reg["peg_tip_row"] = tip
            reg["peg_u"] = round(uc, 1)
            reg["peg_width_px"] = w

            def masked(u, v):
                if v < tip - 3:
                    return False
                cu = uc
                if v in foot:
                    cu = foot[v][0]
                elif (v - 1) in foot:
                    cu = foot[v - 1][0]
                return abs(u - cu) <= 0.5 * w + 3

            mgrid = [[masked(u, v) for u in range(ncols)] for v in range(nrows)]
            vw = video[f_pre].cameras["wrist"]
            lw = get_observation().cameras["wrist"]
            vd = depth_grid(vw.depth_m)
            ldg = depth_grid(lw.depth_m)
            r0 = max(0, tip - int(1.2 * w))
            r1 = min(nrows, tip + 3)
            c0 = max(0, int(uc - 1.1 * w))
            c1 = min(ncols, int(uc + 1.1 * w))
            reg["window"] = [r0, r1, c0, c1]
            if r1 - r0 < 6 or c1 - c0 < 10:
                raise ValueError("registration window too small")
            sh_u = int(w)
            sh_v = max(2, int(0.5 * w))
            coarse, table, npts = register_window(vd, ldg, (r0, r1), (c0, c1), mgrid,
                                                  (-sh_u, sh_u), (-sh_v, sh_v), 2, 2)
            reg["window_pts"] = npts
            if coarse is None:
                raise ValueError("no valid coarse registration")
            fine, table2, npts2 = register_window(vd, ldg, (r0, r1), (c0, c1), mgrid,
                                                  (coarse[1] - 2, coarse[1] + 2), (coarse[2] - 2, coarse[2] + 2), 1, 1)
            zero, t0, n0 = register_window(vd, ldg, (r0, r1), (c0, c1), mgrid, (0, 0), (0, 0), 1, 1)
            best = fine if fine is not None else coarse
            du = best[1]
            dv = best[2]
            reg["best_shift_px"] = [du, dv]
            reg["best_cost_m"] = round(best[0], 5)
            reg["zero_cost_m"] = round(zero[0], 5) if zero is not None else None
            # second-best distinct minimum (ambiguity check) from the coarse table
            alt = None
            for key in table:
                if abs(key[0] - coarse[1]) > 0.25 * w or abs(key[1] - coarse[2]) > 0.25 * w:
                    if alt is None or table[key] < alt[0]:
                        alt = (table[key], key[0], key[1])
            reg["alt_min"] = [round(alt[0], 5), alt[1], alt[2]] if alt is not None else None
            # metric hole shift: per-pixel back-projection with each frame's own camera pose
            vpos = fl(vw.camera_pose.position)
            vqt = fl(vw.camera_pose.quaternion_wxyz)
            lpos = fl(lw.camera_pose.position)
            lqt = fl(lw.camera_pose.quaternion_wxyz)
            up_dot = vdot(qrot(vqt, [0.0, -1.0, 0.0]), d)
            reg["image_up_dot_insert_dir"] = round(up_dot, 3)
            reg["camz_dot_approach"] = round(vdot(qrot(vqt, [0.0, 0.0, 1.0]), approach), 3)
            flip = up_dot < -0.3
            dx = []
            dy = []
            dz = []
            for v in range(r0, r1, 1):
                for u in range(c0, c1, 1):
                    if mgrid[v][u]:
                        continue
                    uu = u + du
                    vv = v + dv
                    if uu < 0 or vv < 0 or uu >= ncols or vv >= nrows or mgrid[vv][uu]:
                        continue
                    zv = vd[v][u]
                    zl = ldg[vv][uu]
                    if zv <= 0.0 or zl <= 0.0 or abs(zv - zl) > 0.004:
                        continue
                    pv = cam_to_world(vpos, vqt, cam_point(u, v, zv, intr), flip)
                    pl = cam_to_world(lpos, lqt, cam_point(uu, vv, zl, lw.intrinsics), flip)
                    dlt = vsub(pl, pv)
                    dx.append(dlt[0])
                    dy.append(dlt[1])
                    dz.append(dlt[2])
            reg["matched_pts"] = len(dx)
            if len(dx) >= 20:
                dx = sorted(dx)
                dy = sorted(dy)
                dz = sorted(dz)
                m = len(dx) // 2
                delta = [dx[m], dy[m], dz[m]]
                reg["hole_shift_world_m"] = rnd(delta, 5)
                a_est = vdot(delta, e1)
                b_est = vdot(delta, e2)
                reg["hole_shift_e1_e2_m"] = [round(a_est, 5), round(b_est, 5)]
                gain = (zero[0] - best[0]) if zero is not None else 0.0
                reg["cost_gain_m"] = round(gain, 5)
                trusted = (abs(up_dot) > 0.3 and abs(a_est) < 0.025 and abs(b_est) < 0.02
                           and zero is not None and best[0] < 0.8 * zero[0])
                reg["trusted"] = trusted
                if trusted:
                    est = [a_est, b_est]
        except Exception as exc:
            reg["error"] = str(exc)[:300]
    report["registration"] = reg

    # ---- insertion along the video's own final line (offset by the estimated hole shift)
    attempts = []
    report["phase"] = "insert"
    qref = vq[-1]
    ik_ref = solve_ik(Pose(vee[-1], vquat[-1], "robot_base"), seed_joints=qref)
    if not ik_ref.ok:
        raise ValueError("IK failed at the video's final pose")
    try:
        q_ik_ref = fl(ik_ref.joint_positions)
    except Exception:
        q_ik_ref = fl(ik_ref.solution)
    report["ik_ref_vs_video_q"] = round(max_err(q_ik_ref, qref), 5)

    def joints_for(s, a, b):
        p = vadd(vadd(vee[-1], vscale(d, s)), vadd(vscale(e1, a), vscale(e2, b)))
        ik = solve_ik(Pose(p, vquat[-1], "robot_base"), seed_joints=q_ik_ref)
        if not ik.ok:
            return None
        try:
            qs = fl(ik.joint_positions)
        except Exception:
            qs = fl(ik.solution)
        if max_err(qs, q_ik_ref) > 0.8:
            return None
        return [qref[j] + qs[j] - q_ik_ref[j] for j in range(7)]

    def glide(q_from, q_to, ticks, label):
        for k in range(1, ticks + 1):
            a = k / float(ticks)
            send([q_from[j] + a * (q_to[j] - q_from[j]) for j in range(7)], final_grip, label)
            if TERMINAL["ended"]:
                return

    def s_live():
        p = fl(get_robot_state().end_effector_poses["primary"].position)
        rel = vsub(p, vee[-1])
        return vdot(rel, d), [vdot(rel, e1), vdot(rel, e2)]

    offsets = []
    if est is not None:
        ea = est[0]
        eb = est[1]
        offsets.append(("estimate", ea, eb))
        for (da, db) in ((0.002, 0.0), (-0.002, 0.0), (0.0, 0.002), (0.0, -0.002),
                         (0.004, 0.0), (-0.004, 0.0)):
            offsets.append(("local", ea + da, eb + db))
    sweep = [("zero", 0.0, 0.0)]
    signs = (1.0, -1.0)
    if est is not None and est[0] < 0.0:
        signs = (-1.0, 1.0)
    for k in range(1, 6):
        for sg in signs:
            sweep.append(("sweep", sg * 0.004 * k, 0.0))
    offsets = offsets + sweep
    s_contact = None
    prev = None
    for (kind, a, b) in offsets:
        if TERMINAL["ended"]:
            break
        if LIMIT["max"] is not None and TICKS["n"] > LIMIT["max"] - 70:
            attempts.append({"kind": "stop_budget"})
            break
        entry = {"kind": kind, "a": round(a, 4), "b": round(b, 4)}
        s_back = s_pre if s_contact is None else min(s_pre, s_contact - 0.01)
        q_back = joints_for(s_back, a, b)
        q_end = joints_for(0.006, a, b)
        if q_back is None or q_end is None:
            entry["ik"] = "failed"
            attempts.append(entry)
            continue
        if prev is not None:
            q_ret = joints_for(s_back, prev[0], prev[1])
            if q_ret is not None:
                glide(live_joints(), q_ret, 8, "retreat")
        shift_len = vnorm([a - (prev[0] if prev is not None else 0.0), b - (prev[1] if prev is not None else 0.0), 0.0])
        glide(live_joints(), q_back, max(5, int(shift_len / 0.001) + 1), "shift")
        hold_until(q_back, final_grip, 0.003, 8, "shift_hold")
        prev = (a, b)
        dist = 0.006 - s_back
        nt = max(6, int(dist / 0.001))
        stalled = False
        s_now = s_back
        for k in range(1, nt + 1):
            fr = k / float(nt)
            send([q_back[j] + fr * (q_end[j] - q_back[j]) for j in range(7)], final_grip, "insert")
            if TERMINAL["ended"]:
                break
            if k % 2 == 0:
                s_now, lat = s_live()
                s_cmd = s_back + fr * dist
                if s_cmd - s_now > 0.008:
                    stalled = True
                    break
        entry["stalled"] = stalled
        entry["s_reached"] = round(s_now, 4)
        entry["tick"] = TICKS["n"]
        if TERMINAL["ended"]:
            attempts.append(entry)
            break
        if stalled:
            s_contact = s_now
            attempts.append(entry)
            continue
        hold_until(q_end, final_grip, 0.003, 20, "insert_hold")
        if not TERMINAL["ended"]:
            q_deep = joints_for(0.012, a, b)
            if q_deep is not None:
                glide(live_joints(), q_deep, 6, "deeper")
                hold_until(q_deep, final_grip, 0.003, 15, "deeper_hold")
            if not TERMINAL["ended"]:
                s_fin, lat = s_live()
                entry["s_final"] = round(s_fin, 4)
        attempts.append(entry)
        break
    report["attempts"] = attempts
    report["phase"] = "done"
except Exception as exc:
    report["error"] = str(exc)[:500]

report["ticks"] = TICKS["n"]
report["terminal"] = TERMINAL["how"]
try:
    if not TERMINAL["ended"]:
        rs = get_robot_state()
        report["final_ee"] = rnd(fl(rs.end_effector_poses["primary"].position))
        report["final_g"] = round(float(rs.gripper_positions["primary"]), 4)
except Exception as exc:
    report["final_read_error"] = str(exc)[:200]

result = report
