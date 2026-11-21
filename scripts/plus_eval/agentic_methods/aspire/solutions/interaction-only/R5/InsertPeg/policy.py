# v04 -- InsertPeg: replay the conditioning video's recorded joint trajectory (peg identity,
# grasp end and box side come from the video) up to its final straight insertion segment,
# push along the video's axis until the face stops the peg, back off, locate the opening in
# the box face from the wrist depth image (ray/face-plane test), then insert closed-loop:
# 1 mm grid around the detected opening first, generic 2 mm grid as fallback.
import math

report = {"version": "v04"}
flags = {"ended": False}
HEXD = "0123456789abcdef"


def mark_end(res):
    try:
        if res.terminated:
            flags["ended"] = True
    except Exception:
        pass
    try:
        if res.truncated:
            flags["ended"] = True
    except Exception:
        pass


def r4(v):
    return round(float(v), 4)


def tolist2(arr):
    try:
        return arr.tolist()
    except Exception:
        return [list(row) for row in arr]


def hexrgb(rgb, stride):
    try:
        sub = rgb[::stride, ::stride].tolist()
    except Exception:
        full = tolist2(rgb)
        sub = [row[::stride] for row in full[::stride]]
    parts = []
    for row in sub:
        for px in row:
            for c in px[:3]:
                v = int(c)
                parts.append(HEXD[v // 16] + HEXD[v % 16])
    return "".join(parts)


def frac_diff(a, b, stride, thr):
    try:
        A = a[::stride, ::stride].tolist()
        B = b[::stride, ::stride].tolist()
    except Exception:
        return None
    cnt = 0
    tot = 0
    for r in range(min(len(A), len(B))):
        ra = A[r]
        rb = B[r]
        for c in range(min(len(ra), len(rb))):
            d = abs(int(ra[c][0]) - int(rb[c][0])) + abs(int(ra[c][1]) - int(rb[c][1])) + abs(int(ra[c][2]) - int(rb[c][2]))
            tot += 1
            if d > thr:
                cnt += 1
    if tot == 0:
        return None
    return round(cnt / float(tot), 4)


def quat_to_xaxis(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))


def quat_to_zaxis(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))


def hand_yaw(q):
    ax = quat_to_xaxis(q)
    return math.atan2(ax[1], ax[0])


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def rotz_quat(q, ang):
    qz = (math.cos(ang / 2.0), 0.0, 0.0, math.sin(ang / 2.0))
    out = qmul(qz, (float(q[0]), float(q[1]), float(q[2]), float(q[3])))
    nrm = math.sqrt(sum(v * v for v in out))
    return [v / nrm for v in out]


def wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def quant(vals, f):
    s = sorted(vals)
    if not s:
        return None
    return s[int((len(s) - 1) * f)]


def cloud_stats(pc):
    try:
        pts = pc.points.tolist()
    except Exception:
        pts = [list(p) for p in pc.points]
    n = len(pts)
    out = {"n": n}
    if n < 5:
        return out, pts
    stride = max(1, n // 3000)
    sp = pts[::stride]
    xs = [float(p[0]) for p in sp]
    ys = [float(p[1]) for p in sp]
    zs = [float(p[2]) for p in sp]
    m = len(sp)
    mx = sum(xs) / m
    my = sum(ys) / m
    sxx = sum((x - mx) ** 2 for x in xs) / m
    syy = sum((y - my) ** 2 for y in ys) / m
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(m)) / m
    ang = 0.5 * math.atan2(2 * sxy, sxx - syy)
    ux, uy = math.cos(ang), math.sin(ang)
    along = [(xs[i] - mx) * ux + (ys[i] - my) * uy for i in range(m)]
    perp = [-(xs[i] - mx) * uy + (ys[i] - my) * ux for i in range(m)]
    out.update({
        "mean": [r4(mx), r4(my), r4(sum(zs) / m)],
        "xmin": r4(min(xs)), "xmax": r4(max(xs)), "ymin": r4(min(ys)), "ymax": r4(max(ys)),
        "z05": r4(quant(zs, 0.05)), "z50": r4(quant(zs, 0.5)), "z95": r4(quant(zs, 0.95)),
        "axis_deg": round(math.degrees(ang), 2),
        "len": r4(quant(along, 0.98) - quant(along, 0.02)),
        "wid": r4(quant(perp, 0.98) - quant(perp, 0.02)),
        "along_lo": r4(quant(along, 0.02)), "along_hi": r4(quant(along, 0.98)),
    })
    return out, sp


# ---------------------------------------------------------------- context / metadata
ctx = get_task_context()
report["goal"] = ctx.language
meta = {}
try:
    meta = dict(robomme_plus.get_task_metadata())
    report["task_metadata"] = {k: str(meta[k])[:300] for k in meta}
except Exception as exc:
    report["task_metadata_error"] = str(exc)[:300]
hz = 20.0
try:
    hz = float(meta["control_frequency_hz"])
except Exception:
    hz = 20.0
max_steps = 0
try:
    max_steps = int(meta["max_steps"])
except Exception:
    max_steps = 0

state0 = get_robot_state()
emb = state0.embodiment
base_frame = state0.base_frame
home_q = [float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz]
home_j = [float(v) for v in state0.joint_positions["primary"]]

# ---------------------------------------------------------------- conditioning video
video = []
try:
    video = robomme_plus.get_conditioning_video()
except Exception as exc:
    report["video_error"] = str(exc)[:300]
nv = len(video)
report["video_len"] = nv
traj = []
for i in range(nv):
    rs = video[i].robot_state
    pose = rs.end_effector_poses["primary"]
    traj.append({
        "t": float(video[i].timestamp_s),
        "p": [float(v) for v in pose.position],
        "q": [float(v) for v in pose.quaternion_wxyz],
        "g": float(rs.gripper_positions["primary"]),
        "j": [float(v) for v in rs.joint_positions["primary"]],
    })
if nv > 0:
    report["video_traj"] = [[r4(e["t"]), r4(e["p"][0]), r4(e["p"][1]), r4(e["p"][2]),
                             round(math.degrees(hand_yaw(e["q"])), 1), round(e["g"], 3)] for e in traj[::2]]

# gripper events: closing onset (grasp) and re-opening (release)
open_level = 1.0
close_idx = None
release_idx = None
if nv > 0:
    open_level = max(e["g"] for e in traj[: max(1, nv // 10)])
    for i in range(nv - 3):
        if traj[i]["g"] < open_level - 0.004 and traj[i + 1]["g"] < traj[i]["g"] + 0.002 and traj[i + 3]["g"] < open_level - 0.05:
            close_idx = i
            break
    if close_idx is not None:
        low = min(e["g"] for e in traj[close_idx:])
        settled = traj[min(nv - 1, close_idx + 10)]["g"]
        for i in range(close_idx + 10, nv - 1):
            if traj[i]["g"] > settled + 0.05 and traj[i + 1]["g"] >= traj[i]["g"]:
                release_idx = i
                break
report["video_open_level"] = r4(open_level)
report["video_close_idx"] = close_idx
report["video_release_idx"] = release_idx
if close_idx is not None:
    gi = min(nv - 1, close_idx + 4)
    report["video_grasp_pose"] = {"idx": gi, "p": [r4(v) for v in traj[gi]["p"]],
                                  "yaw_deg": round(math.degrees(hand_yaw(traj[gi]["q"])), 2)}
if nv > 0:
    report["video_final_pose"] = {"p": [r4(v) for v in traj[-1]["p"]], "q": [r4(v) for v in traj[-1]["q"]],
                                  "g": r4(traj[-1]["g"])}

thumbs = {}
if nv > 0:
    idxs = [0, nv - 1]
    if close_idx is not None:
        idxs.append(min(nv - 1, close_idx + 4))
        idxs.append((close_idx + nv) // 2)
    for i in sorted(set(idxs)):
        try:
            thumbs["front_%d" % i] = hexrgb(video[i].cameras["front"].rgb, 4)
        except Exception as exc:
            thumbs["front_%d_err" % i] = str(exc)[:200]
    if close_idx is not None:
        try:
            thumbs["wrist_%d" % min(nv - 1, close_idx + 4)] = hexrgb(video[min(nv - 1, close_idx + 4)].cameras["wrist"].rgb, 4)
        except Exception as exc:
            thumbs["wrist_err"] = str(exc)[:200]
report["thumbs"] = thumbs

# ---------------------------------------------------------------- layout check (depth: lighting-invariant)
live = get_observation()


def depth_diff(da, db, stride, thr):
    try:
        A = da[::stride, ::stride].tolist()
        B = db[::stride, ::stride].tolist()
    except Exception:
        return None
    cnt = 0
    tot = 0
    for r in range(min(len(A), len(B))):
        ra = A[r]
        rb = B[r]
        for c in range(min(len(ra), len(rb))):
            tot += 1
            if abs(float(ra[c]) - float(rb[c])) > thr:
                cnt += 1
    if tot == 0:
        return None
    return round(cnt / float(tot), 5)


layout_frac = None
if nv > 0:
    try:
        layout_frac = depth_diff(video[0].cameras["front"].depth_m, live.cameras["front"].depth_m, 2, 0.004)
        report["depth_diff_video0_live"] = layout_frac
        report["depth_diff_videoN_live"] = depth_diff(video[nv - 1].cameras["front"].depth_m, live.cameras["front"].depth_m, 2, 0.004)
        jd = max(abs(traj[0]["j"][k] - home_j[k]) for k in range(7))
        report["joint_diff_video0_live"] = r4(jd)
    except Exception as exc:
        report["layout_err"] = str(exc)[:200]

# ---------------------------------------------------------------- live peg perception (diagnostic + fallback)
pegs = []
try:
    seg = segment_text("front", "peg")
    if seg.ok:
        for ci, cand in enumerate(seg.segmentations[:6]):
            try:
                pc = mask_to_point_cloud(cand, "front", target_frame=base_frame)
                st, sp = cloud_stats(pc)
                st["score"] = round(float(cand.score), 4)
                try:
                    st["box"] = [round(float(v), 1) for v in cand.box_xyxy]
                except Exception:
                    pass
                if st.get("n", 0) >= 20 and "len" in st and st["wid"] < 0.06 and (st["z95"] - st["z05"]) < 0.06:
                    pegs.append(st)
            except Exception as exc:
                report["peg_cloud_err"] = str(exc)[:200]
except Exception as exc:
    report["peg_seg_err"] = str(exc)[:200]
report["live_pegs"] = pegs

grasp_match = None
if close_idx is not None and pegs:
    gp = traj[min(nv - 1, close_idx + 4)]["p"]
    best = None
    for st in pegs:
        ax = math.radians(st["axis_deg"])
        dx = gp[0] - st["mean"][0]
        dy = gp[1] - st["mean"][1]
        along = dx * math.cos(ax) + dy * math.sin(ax)
        perp = -dx * math.sin(ax) + dy * math.cos(ax)
        over = max(0.0, st["along_lo"] - along, along - st["along_hi"])
        dist = math.sqrt(perp * perp + over * over)
        if best is None or dist < best[0]:
            best = (dist, st, along, perp)
    grasp_match = {"dist": r4(best[0]), "along": r4(best[2]), "perp": r4(best[3]), "peg_mean": best[1]["mean"],
                   "peg_axis_deg": best[1]["axis_deg"]}
report["video_grasp_on_live_peg"] = grasp_match

# ---------------------------------------------------------------- attempt
attempt = {"mode": None}
report["attempt"] = attempt


def exec_info(res):
    d = {}
    try:
        d["ok"] = bool(res.ok)
    except Exception:
        pass
    try:
        d["status"] = str(res.status)
    except Exception:
        pass
    try:
        d["steps"] = int(res.steps_executed)
    except Exception:
        pass
    mark_end(res)
    d["ended"] = flags["ended"]
    return d


def grip_cmd_for(idx):
    if close_idx is None:
        return 1.0
    if idx >= close_idx and (release_idx is None or idx < release_idx):
        return 0.0
    return 1.0


def replay_joints(i0, i1, log):
    k = i0
    t0 = traj[i0]["t"]
    nt = int(round((traj[i1]["t"] - t0) * hz))
    log["ticks_planned"] = nt
    samples = []
    sent = 0
    for tick in range(1, nt + 1):
        if flags["ended"]:
            break
        tt = t0 + tick / hz
        while k + 1 < i1 and traj[k + 1]["t"] <= tt + 1e-6:
            k += 1
        a = traj[k]
        b = traj[min(k + 1, i1)]
        span = b["t"] - a["t"]
        f = 1.0 if span <= 1e-6 else min(1.0, max(0.0, (tt - a["t"]) / span))
        jt = [a["j"][m] + f * (b["j"][m] - a["j"][m]) for m in range(7)]
        gcmd = grip_cmd_for(min(k + 1, i1) if f > 0.5 else k)
        res = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=jt, gripper_position=gcmd, embodiment=emb)}))
        sent += 1
        mark_end(res)
        if tick % 20 == 0 and not flags["ended"]:
            try:
                s = get_robot_state()
                ep = s.end_effector_poses["primary"].position
                vb = b["p"]
                samples.append([tick, r4(ep[0]), r4(ep[1]), r4(ep[2]), r4(s.gripper_positions["primary"]),
                                r4(math.sqrt(sum((float(ep[m]) - vb[m]) ** 2 for m in range(3))))])
            except Exception:
                pass
    log["ticks_sent"] = sent
    log["samples"] = samples


def hold(ticks, gcmd, log, key):
    s = get_robot_state()
    jt = [float(v) for v in s.joint_positions["primary"]]
    n = 0
    for tick in range(ticks):
        if flags["ended"]:
            break
        res = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=jt, gripper_position=gcmd, embodiment=emb)}))
        n += 1
        mark_end(res)
    log[key] = n


def final_segment():
    # Video's final straight approach: walk back from the last frame while the EE stays on
    # the line through the final point (perpendicular < 3 mm, height within 3 mm).
    pe = traj[nv - 1]["p"]
    ia = None
    for i in range(nv - 1, 0, -1):
        d = math.sqrt(sum((traj[i]["p"][m] - pe[m]) ** 2 for m in range(3)))
        if d > 0.04:
            ia = i
            break
    if ia is None:
        return None
    dv = [pe[m] - traj[ia]["p"][m] for m in range(3)]
    nd = math.sqrt(sum(v * v for v in dv))
    a = [v / nd for v in dv]
    s0 = ia
    for i in range(ia, 0, -1):
        p = traj[i]["p"]
        rel = [p[m] - pe[m] for m in range(3)]
        along = sum(rel[m] * a[m] for m in range(3))
        perp = math.sqrt(max(0.0, sum(v * v for v in rel) - along * along))
        if perp < 0.003 and abs(p[2] - pe[2]) < 0.003 and along <= 0.0:
            s0 = i
        else:
            break
    ps = traj[s0]["p"]
    length = sum((pe[m] - ps[m]) * a[m] for m in range(3))
    return {"s0": s0, "a": a, "length": length, "ps": ps}


def ee_now():
    s = get_robot_state()
    return s, [float(v) for v in s.end_effector_poses["primary"].position]


def quat_to_R(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def depth_val(d):
    try:
        return float(d)
    except Exception:
        return float(d[0])


def find_hole(cam, tcp, a, lat, stride):
    # Locate the opening in the box face ahead of the peg from the wrist depth image.
    # Face plane = dominant constant-y surface ahead of the peg; opening = pixels whose
    # ray crosses that plane below the box top but whose depth lies behind the plane.
    out = {}
    K = tolist2(cam.intrinsics)
    fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
    R = quat_to_R(cam.camera_pose.quaternion_wxyz)
    t = [float(v) for v in cam.camera_pose.position]
    out["cam_frame"] = str(cam.camera_pose.frame)
    D = cam.depth_m[::stride, ::stride].tolist()
    rays = []
    for r in range(len(D)):
        row = D[r]
        vpx = r * stride
        for c in range(len(row)):
            dv = depth_val(row[c])
            if not (dv > 0.01 and dv < 1.5):
                continue
            upx = c * stride
            xc = (upx - cx) / fx
            yc = (vpx - cy) / fy
            db = [R[m][0] * xc + R[m][1] * yc + R[m][2] for m in range(3)]
            P = [t[m] + dv * db[m] for m in range(3)]
            rays.append((P, db, upx, vpx))
    out["n_rays"] = len(rays)
    # face plane: mode of y over off-peg points above the table and ahead of the hand
    hist = {}
    for e in rays:
        P = e[0]
        if P[2] < 0.012 or P[1] > tcp[1] - 0.02:
            continue
        if abs((P[0] - tcp[0]) * lat[0] + (P[1] - tcp[1]) * lat[1]) < 0.013:
            continue
        k = int(math.floor(P[1] / 0.002))
        hist[k] = hist.get(k, 0) + 1
    if not hist:
        out["err"] = "no face points"
        return out
    kb = None
    for k in hist:
        if kb is None or hist[k] > hist[kb]:
            kb = k
    out["face_bin_n"] = hist[kb]
    face = [e[0] for e in rays if abs(e[0][1] - (kb + 0.5) * 0.002) < 0.003 and e[0][2] >= 0.012]
    yf = quant([P[1] for P in face], 0.5)
    out["face_y"] = r4(yf)
    fxs = [P[0] for P in face]
    fzs = [P[2] for P in face]
    out["face_x"] = [r4(quant(fxs, 0.03)), r4(quant(fxs, 0.97))]
    out["face_z"] = [r4(quant(fzs, 0.03)), r4(quant(fzs, 0.97))]
    # upper bound of the face actually seen (box top, or the image border)
    ztop = quant(fzs, 0.99)
    out["z_top"] = r4(ztop)
    xlo = quant(fxs, 0.03)
    xhi = quant(fxs, 0.97)
    qs = []
    for e in rays:
        P, db = e[0], e[1]
        if P[1] > yf - 0.003 or P[2] < 0.004 or abs(db[1]) < 1e-6:
            continue
        tq = (yf - t[1]) / db[1]
        if tq <= 0:
            continue
        Q = [t[m] + tq * db[m] for m in range(3)]
        if Q[2] < 0.004 or Q[2] > ztop - 0.002 or Q[0] < xlo + 0.001 or Q[0] > xhi - 0.001:
            continue
        qs.append(Q)
    out["n_hole"] = len(qs)
    if len(qs) < 12:
        out["err"] = "no opening"
        return out
    qx = [Q[0] for Q in qs]
    qz = [Q[2] for Q in qs]
    xl = quant(qx, 0.02)
    xh = quant(qx, 0.98)
    zl = quant(qz, 0.02)
    zh = quant(qz, 0.98)
    hx = 0.5 * (xl + xh)
    # the lower rim is hidden (downward view, peg in front): square opening, top rim + half width
    hz = zh - 0.5 * (xh - xl)
    if zh > ztop - 0.003:
        # top rim cut by the image/face border: blend with the lower-rim estimate
        hz = 0.5 * (hz + zl + 0.5 * (xh - xl))
    if zh - zl > xh - xl:
        hz = 0.5 * (zl + zh)
    out["hole_x"] = [r4(xl), r4(xh)]
    out["hole_z"] = [r4(zl), r4(zh)]
    out["center"] = [r4(hx), r4(yf), r4(hz)]
    out["ok"] = True
    return out


def search_insert(seg, log):
    a = seg["a"]
    pe = traj[nv - 1]["p"]
    ps = [pe[m] - seg["length"] * a[m] for m in range(3)]
    hn = math.sqrt(a[0] * a[0] + a[1] * a[1])
    lat = [-a[1] / hn, a[0] / hn, 0.0]
    up = [0.0, 0.0, 1.0]
    qf = traj[nv - 1]["q"]
    goal = seg["length"] + 0.005
    tick_cap = max_steps - 4 if max_steps > 0 else 800
    tries = []
    cache = {"key": None, "j": None}

    def command(s_cmd, ol, ov):
        tgt = [ps[m] + s_cmd * a[m] + ol * lat[m] + ov * up[m] for m in range(3)]
        key = (round(s_cmd, 4), round(ol, 4), round(ov, 4))
        if cache["key"] != key:
            st = get_robot_state()
            ik = solve_ik(Pose(tgt, qf, base_frame), seed_joints=[float(v) for v in st.joint_positions["primary"]])
            if not ik.ok:
                return False
            jt = [float(v) for v in ik.joint_positions]
            cur = [float(v) for v in st.joint_positions["primary"]]
            if max(abs(jt[m] - cur[m]) for m in range(7)) > 0.35:
                return False
            cache["key"] = key
            cache["j"] = jt
        res = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=cache["j"], gripper_position=0.0, embodiment=emb)}))
        mark_end(res)
        return True

    def depth_of(p):
        return sum((p[m] - ps[m]) * a[m] for m in range(3))

    def ticks_now():
        st = get_robot_state()
        return int(round(float(st.timestamp_s) * hz))

    def push(s_cmd, ol, ov):
        hist = []
        ticks = 0
        ik_fail = 0
        s_meas = s_cmd
        while not flags["ended"]:
            st, p = ee_now()
            if int(round(float(st.timestamp_s) * hz)) >= tick_cap:
                log["tick_cap_hit"] = True
                return s_meas, False, ticks, ik_fail, s_cmd
            s_meas = depth_of(p)
            hist.append(s_meas)
            if s_meas >= goal - 0.004:
                return s_meas, False, ticks, ik_fail, s_cmd
            s_cmd = min(goal, s_cmd + 0.0015, s_meas + 0.004)
            if not command(s_cmd, ol, ov):
                ik_fail += 1
                if ik_fail > 3:
                    return s_meas, False, ticks, ik_fail, s_cmd
            ticks += 1
            if len(hist) >= 5 and hist[-1] - hist[-5] < 0.0008 and s_cmd - s_meas > 0.003:
                return s_meas, True, ticks, ik_fail, s_cmd
        return s_meas, False, ticks, ik_fail, s_cmd

    def move_at(s_val, ol, ov, n):
        for k in range(n):
            if flags["ended"]:
                break
            command(s_val, ol, ov)

    # 1) find the face: push straight along the video's insertion line
    st, p = ee_now()
    s_cmd = depth_of(p)
    s_meas, stalled, ticks, ik_fail, s_cmd = push(s_cmd, 0.0, 0.0)
    tries.append([0, 0, r4(s_meas), ticks, stalled, ik_fail])
    best_depth = s_meas
    if stalled and not flags["ended"]:
        face = s_meas
        log["face_depth"] = r4(face)
        # 2) back off so the face and its opening are in the wrist camera's view
        view_s = face - 0.022
        move_at(view_s, 0.0, 0.0, 14)
        cl = None
        cv = None
        if not flags["ended"]:
            try:
                obs = get_observation()
                cam = obs.cameras["wrist"]
                st, p = ee_now()
                hl = find_hole(cam, p, a, lat, 2)
                log["hole_live"] = hl
                try:
                    log["wrist_view"] = hexrgb(cam.rgb, 4)
                except Exception as exc:
                    log["wrist_err"] = str(exc)[:200]
                if hl.get("ok"):
                    c = hl["center"]
                    s_face_pt = sum((c[m] - ps[m]) * a[m] for m in range(3))
                    cl = sum((c[m] - ps[m]) * lat[m] for m in range(3))
                    cv = c[2] - (ps[2] + s_face_pt * a[2])
                    log["hole_offset_mm"] = [round(cl * 1000.0, 1), round(cv * 1000.0, 1)]
            except Exception as exc:
                log["hole_err"] = str(exc)[:300]
            # diagnostic: the same detector on the conditioning video at the matching depth
            try:
                bi = None
                for i in range(seg["s0"], nv):
                    dd = abs(depth_of(traj[i]["p"]) - view_s)
                    if bi is None or dd < bi[1]:
                        bi = (i, dd)
                if bi is not None:
                    hv = find_hole(video[bi[0]].cameras["wrist"], traj[bi[0]]["p"], a, lat, 2)
                    hv["frame"] = bi[0]
                    log["hole_video"] = hv
            except Exception as exc:
                log["hole_video_err"] = str(exc)[:300]
        # 3) offsets to try (mm, relative to the video line): around the detected opening first
        offs = []
        seen = {}
        if cl is not None and abs(cl) < 0.03 and abs(cv) < 0.03:
            c0 = (round(cl * 2000.0) / 2.0, round(cv * 2000.0) / 2.0)
            ring = []
            for i in range(-3, 4):
                for j in range(-3, 4):
                    ring.append((i * i + j * j, c0[0] + i, c0[1] + j))
            ring.sort()
            for e in ring:
                offs.append((e[1], e[2]))
        grid = []
        for i in range(-7, 8):
            for j in range(-4, 5):
                grid.append((4 * i * i + 4 * j * j, 2.0 * i, 2.0 * j))
        grid.sort()
        for e in grid:
            if e[1] == 0 and e[2] == 0:
                continue
            offs.append((e[1], e[2]))
        log["n_offsets"] = len(offs)
        local = [(1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)]
        local_left = 0
        li = 0
        base = (0.0, 0.0)
        oi = 0
        first = True
        while not flags["ended"] and not log.get("tick_cap_hit"):
            if local_left > 0:
                off = (base[0] + local[li % len(local)][0], base[1] + local[li % len(local)][1])
                li += 1
                local_left -= 1
            else:
                if oi >= len(offs):
                    break
                off = offs[oi]
                oi += 1
                key = (off[0], off[1])
                if key in seen:
                    continue
                seen[key] = 1
            ol = off[0] * 0.001
            ov = off[1] * 0.001
            if first:
                move_at(view_s, ol, ov, 4)
                s_cmd = view_s
                first = False
            else:
                move_at(s_cmd, ol, ov, 3)
            s_meas, stalled, ticks, ik_fail, s_cmd = push(s_cmd, ol, ov)
            if s_meas > best_depth:
                best_depth = s_meas
            tries.append([off[0], off[1], r4(s_meas), ticks, stalled, ik_fail])
            if not stalled:
                break
            if s_meas > face + 0.004:
                if local_left == 0 and li == 0:
                    base = (off[0], off[1])
                    local_left = len(local)
                back = s_meas - 0.002
            else:
                back = s_meas - 0.005
            move_at(back, ol, ov, 4)
            s_cmd = back
    log["tries"] = tries
    log["best_depth"] = r4(best_depth)
    log["goal_depth"] = r4(goal)


try:
    on_peg = grasp_match is not None and grasp_match["dist"] < 0.025
    same_layout = layout_frac is not None and layout_frac < 0.02
    attempt["same_layout"] = same_layout
    attempt["grasp_on_live_peg"] = on_peg
    seg = final_segment() if nv > 0 else None
    if seg is not None:
        attempt["segment"] = {"s0": seg["s0"], "a": [r4(v) for v in seg["a"]], "length": r4(seg["length"]),
                              "ps": [r4(v) for v in seg["ps"]]}
    if nv > 0 and close_idx is not None and seg is not None and (same_layout or on_peg):
        attempt["mode"] = "video_joint_replay_then_search_insert"
        jd0 = max(abs(traj[0]["j"][k] - home_j[k]) for k in range(7))
        if jd0 > 0.01:
            attempt["to_start"] = exec_info(move_to_joints(traj[0]["j"], tolerance=0.01, max_steps=80))
        if not flags["ended"]:
            replay_joints(0, seg["s0"], attempt)
        if not flags["ended"]:
            hold(3, grip_cmd_for(seg["s0"]), attempt, "hold_pre")
        if not flags["ended"] and grip_cmd_for(seg["s0"]) < 0.5:
            ins = {}
            attempt["insert"] = ins
            search_insert(seg, ins)
        if not flags["ended"]:
            hold(10, 0.0, attempt, "hold_end")
    else:
        attempt["mode"] = "none"
except Exception as exc:
    attempt["error"] = str(exc)[:400]

if not flags["ended"]:
    try:
        sN = get_robot_state()
        report["final_state"] = {"p": [r4(v) for v in sN.end_effector_poses["primary"].position],
                                 "g": r4(sN.gripper_positions["primary"]), "t": r4(sN.timestamp_s)}
    except Exception as exc:
        report["final_state_err"] = str(exc)[:200]
report["ended"] = flags["ended"]
result = report
