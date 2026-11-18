# RouteStick v00 -- replay the conditioning video's stick route, re-anchored to the live targets.
#
# Strategy (all quantities derived at runtime):
#   1. Read the conditioning video (Observations with front RGB-D + recorded robot state).
#   2. Detect the white target discs in the first video frame and in the live front image
#      (pure-python colour/shape test), order both sets left-to-right in the image.
#   3. Detect each touch in the video as the onset of red pixels on a disc (targets turn red
#      when the stick touches them).  The touch order = the route to reproduce.
#   4. Re-anchor the recorded end-effector path: each recorded touch pose is moved onto the
#      corresponding live disc; the path between two touches is mapped with the 2-D similarity
#      transform defined by its two anchors (keeps the side on which it passes the obstacle).
#   5. IK every waypoint (seeded from the recorded joints / previous solution) and stream the
#      joint targets tick by tick with step().

import math

MIN_WAYPOINT_DT = 0.1       # s of video time between IK waypoints
SLOW = 1.25                 # execution time / video time
TOUCH_HOLD_TICKS = 4        # extra ticks holding each touch pose
RED_ON = 8                  # red pixels above baseline in a disc box = touched
RED_OFF = 4
DESCEND_STEP = 0.006        # m, adaptive extra descent when a touch is not registered
DESCEND_MAX = 2
MAX_JOINT_STEP = 0.06       # rad per tick rate limit on streamed joint targets

log = {"version": "v00", "notes": []}


def note(msg):
    log["notes"].append(str(msg))


def to_list(arr):
    try:
        return arr.tolist()
    except Exception:
        return [list(x) for x in arr]


def fl(vec):
    return [float(x) for x in vec]


def disk_u(d):
    return d["u"]


def find_disks(rows):
    H = len(rows)
    W = len(rows[0])
    cand = [[False] * W for v in range(H)]
    for v in range(H):
        row = rows[v]
        crow = cand[v]
        for u in range(W):
            px = row[u]
            lo = min(px[0], px[1], px[2])
            hi = max(px[0], px[1], px[2])
            if lo >= 135 and hi - lo <= 45:
                crow[u] = True
    seen = [[False] * W for v in range(H)]
    disks = []
    for v in range(H):
        for u in range(W):
            if cand[v][u] and not seen[v][u]:
                stack = [(u, v)]
                seen[v][u] = True
                pix = []
                while stack:
                    cur = stack.pop()
                    pix.append(cur)
                    for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nu = cur[0] + d[0]
                        nv = cur[1] + d[1]
                        if 0 <= nu < W and 0 <= nv < H and cand[nv][nu] and not seen[nv][nu]:
                            seen[nv][nu] = True
                            stack.append((nu, nv))
                n = len(pix)
                if n < 6 or n > 200:
                    continue
                us = [p[0] for p in pix]
                vs = [p[1] for p in pix]
                w = max(us) - min(us) + 1
                h = max(vs) - min(vs) + 1
                if w > 20 or h > 20:
                    continue
                nwhite = 0
                for p in pix:
                    px = rows[p[1]][p[0]]
                    if min(px[0], px[1], px[2]) >= 200:
                        nwhite += 1
                fill = n / float(w * h)
                if nwhite < 8 or fill < 0.45:
                    continue
                disks.append({"u": sum(us) / float(n), "v": sum(vs) / float(n), "area": n,
                              "w": w, "h": h, "pixels": pix})
    disks.sort(key=disk_u)
    return disks


def red_count(rows, u, v, rad=7):
    H = len(rows)
    W = len(rows[0])
    c = 0
    for vv in range(max(0, int(v) - rad), min(H, int(v) + rad + 1)):
        row = rows[vv]
        for uu in range(max(0, int(u) - rad), min(W, int(u) + rad + 1)):
            px = row[uu]
            if px[0] >= 140 and px[1] <= 90 and px[2] <= 90 and px[0] - px[1] >= 80:
                c += 1
    return c


def disc_pixels(rows, u, v, rad=7):
    H = len(rows)
    W = len(rows[0])
    c = 0
    for vv in range(max(0, int(v) - rad), min(H, int(v) + rad + 1)):
        row = rows[vv]
        for uu in range(max(0, int(u) - rad), min(W, int(u) + rad + 1)):
            px = row[uu]
            lo = min(px[0], px[1], px[2])
            hi = max(px[0], px[1], px[2])
            if lo >= 135 and hi - lo <= 45:
                c += 1
    return c


def quat_to_R(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def deproject(u, v, d, K, R, t, conv, ray):
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    xn = (u - cx) / fx
    yn = (v - cy) / fy
    z = d
    if ray:
        z = d / math.sqrt(1.0 + xn * xn + yn * yn)
    pc = [xn * z, yn * z, z]
    if conv == "gl":
        pc = [pc[0], -pc[1], -pc[2]]
    return [R[i][0] * pc[0] + R[i][1] * pc[1] + R[i][2] * pc[2] + t[i] for i in range(3)]


def median(vals):
    s = sorted(vals)
    if not s:
        return None
    return s[len(s) // 2]


def disk_depth(depth_rows, disk):
    vals = []
    for p in disk["pixels"]:
        dv = depth_rows[p[1]][p[0]]
        if dv > 0.05 and dv < 5.0:
            vals.append(dv)
    return median(vals)


def disk_live_3d(disk, H, W, base_frame):
    """3-D centre of one disc from the public depth projection of its own pixel mask."""
    errors = []
    mask = [[False] * W for v in range(H)]
    for p in disk["pixels"]:
        mask[p[1]][p[0]] = True
    pc = None
    try:
        pc = mask_to_point_cloud(mask, "front", target_frame=base_frame)
    except Exception as exc:
        errors.append("list-mask: " + str(exc)[:160])
        pc = None
    if pc is None:
        try:
            seg = segment_points("front", [[int(round(disk["u"])), int(round(disk["v"]))]])
            best = None
            if seg.ok:
                for cand in seg.segmentations:
                    if cand.box_xyxy is None:
                        continue
                    b = fl(cand.box_xyxy)
                    if (b[2] - b[0]) <= 3 * disk["w"] + 4 and (b[3] - b[1]) <= 3 * disk["h"] + 4:
                        best = cand
                        break
            if best is not None:
                pc = mask_to_point_cloud(best, "front", target_frame=base_frame)
        except Exception as exc:
            errors.append("segment_points: " + str(exc)[:160])
            pc = None
    if pc is None:
        return None, errors
    pts = to_list(pc.points)
    if len(pts) < 3:
        return None, errors + ["few points"]
    return [median([p[0] for p in pts]), median([p[1] for p in pts]), median([p[2] for p in pts])], errors


def ik_list(ik):
    jp = ik.joint_positions
    try:
        return fl(jp)
    except Exception:
        return fl(jp["primary"])


def is_terminal(reply):
    term = False
    for name in ("terminated", "truncated"):
        try:
            if name == "terminated" and bool(reply.terminated):
                term = True
            if name == "truncated" and bool(reply.truncated):
                term = True
        except Exception:
            pass
    return term


# ---------------------------------------------------------------- task context
ctx = get_task_context()
log["language"] = ctx.language
meta = {}
try:
    meta = dict(robomme_plus.get_task_metadata())
except Exception as exc:
    note("metadata error " + str(exc)[:200])
log["metadata"] = {k: str(meta[k])[:200] for k in meta}
max_steps = 1000
try:
    max_steps = int(meta["max_steps"])
except Exception:
    note("max_steps missing; using 1000")
try:
    log["controller"] = {k: str(v)[:200] for k, v in dict(robomme_plus.get_controller_metadata()).items()}
except Exception as exc:
    note("controller metadata error " + str(exc)[:200])

state0 = get_robot_state()
base_frame = state0.base_frame
emb = state0.embodiment
grip0 = float(state0.gripper_positions["primary"])
live_j0 = fl(state0.joint_positions["primary"])
log["live_j0"] = [round(x, 4) for x in live_j0]
ee0 = state0.end_effector_poses["primary"]
log["live_ee0"] = {"p": [round(x, 4) for x in fl(ee0.position)], "q": [round(x, 4) for x in fl(ee0.quaternion_wxyz)]}

# ---------------------------------------------------------------- conditioning video
video = robomme_plus.get_conditioning_video()
nv = len(video)
log["video_frames"] = nv
vt = []
vp = []
vq = []
vj = []
for fr in video:
    rs = fr.robot_state
    vt.append(float(fr.timestamp_s))
    pose = rs.end_effector_poses["primary"]
    vp.append(fl(pose.position))
    vq.append(fl(pose.quaternion_wxyz))
    vj.append(fl(rs.joint_positions["primary"]))
log["video_t_range"] = [vt[0], vt[-1]] if nv else None
log["video_ee"] = [[round(vt[i], 3)] + [round(x, 4) for x in vp[i]] for i in range(nv)]
if nv:
    log["video_j0"] = [round(x, 4) for x in vj[0]]
    log["video_q0"] = [round(x, 4) for x in vq[0]]

# disks in first video frame
f0 = video[0].cameras["front"]
rows0 = to_list(f0.rgb)
vdisks = find_disks(rows0)
log["video_disks"] = [[round(d["u"], 1), round(d["v"], 1), d["area"]] for d in vdisks]
base_red = [red_count(rows0, d["u"], d["v"]) for d in vdisks]

# touches: onset of red on a disc
touches = []          # (frame index, disk index)
prev_on = [c >= RED_ON for c in base_red]
red_series = []
band_v0 = 0
band_v1 = len(rows0)
if vdisks:
    band_v0 = max(0, int(min(d["v"] for d in vdisks)) - 9)
    band_v1 = min(len(rows0), int(max(d["v"] for d in vdisks)) + 10)
for i in range(nv):
    if i == 0:
        rows = rows0[band_v0:band_v1]
    else:
        rgb_i = video[i].cameras["front"].rgb
        try:
            rows = to_list(rgb_i[band_v0:band_v1])
        except Exception:
            rows = to_list(rgb_i)[band_v0:band_v1]
    cnt = [red_count(rows, d["u"], d["v"] - band_v0) - b for d, b in zip(vdisks, base_red)]
    red_series.append(cnt)
    for k in range(len(vdisks)):
        if (not prev_on[k]) and cnt[k] >= RED_ON:
            prev_on[k] = True
            if not touches or touches[-1][1] != k:
                touches.append((i, k))
        elif prev_on[k] and cnt[k] < RED_OFF:
            prev_on[k] = False
log["touch_source"] = "red"
if len(touches) < 2:
    note("red touch detection found %d touches; trying kinematic z-minima fallback" % len(touches))
log["touches"] = [[t[0], round(vt[t[0]], 3), t[1]] for t in touches]
log["red_series_sparse"] = [[i] + red_series[i] for i in range(0, nv, 4)]

# ---------------------------------------------------------------- live discs
obs = get_observation()
lf = obs.cameras["front"]
lrows = to_list(lf.rgb)
ldepth = to_list(lf.depth_m)
H = len(lrows)
W = len(lrows[0])
ldisks = find_disks(lrows)
log["live_disks"] = [[round(d["u"], 1), round(d["v"], 1), d["area"]] for d in ldisks]
live3d = []
for d in ldisks:
    p3, errs = disk_live_3d(d, H, W, base_frame)
    if errs:
        note("disk3d errors " + "; ".join(errs))
    live3d.append(p3)
log["live_disks_3d"] = [None if p is None else [round(x, 4) for x in p] for p in live3d]

# projection convention check (live pixels+depth vs public point cloud), then video disc 3-D
Kl = to_list(lf.intrinsics)
Rl = quat_to_R(fl(lf.camera_pose.quaternion_wxyz))
tl = fl(lf.camera_pose.position)
best_conv = None
best_err = None
for conv in ("cv", "gl"):
    for ray in (False, True):
        errs = []
        for d, p3 in zip(ldisks, live3d):
            if p3 is None:
                continue
            dd = disk_depth(ldepth, d)
            if dd is None:
                continue
            q3 = deproject(d["u"], d["v"], dd, Kl, Rl, tl, conv, ray)
            errs.append(math.sqrt(sum((q3[i] - p3[i]) ** 2 for i in range(3))))
        if errs:
            e = sum(errs) / len(errs)
            if best_err is None or e < best_err:
                best_err = e
                best_conv = (conv, ray)
log["deproject_check"] = {"conv": best_conv, "mean_err_m": best_err}
vdisk3d = [None] * len(vdisks)
if best_conv is not None and best_err is not None and best_err < 0.02:
    vdepth = to_list(f0.depth_m)
    Kv = to_list(f0.intrinsics)
    Rv = quat_to_R(fl(f0.camera_pose.quaternion_wxyz))
    tv = fl(f0.camera_pose.position)
    for k in range(len(vdisks)):
        dd = disk_depth(vdepth, vdisks[k])
        if dd is not None:
            vdisk3d[k] = deproject(vdisks[k]["u"], vdisks[k]["v"], dd, Kv, Rv, tv, best_conv[0], best_conv[1])
log["video_disks_3d"] = [None if p is None else [round(x, 4) for x in p] for p in vdisk3d]

if len(touches) < 2 and nv > 20 and all(p is not None for p in vdisk3d) and vdisk3d:
    zs = [p[2] for p in vp]
    zmin = min(zs)
    kt = []
    for i in range(nv):
        lo = max(0, i - 10)
        hi = min(nv, i + 11)
        if zs[i] <= zmin + 0.015 and zs[i] == min(zs[lo:hi]):
            if kt and vt[i] - vt[kt[-1][0]] < 1.0:
                continue
            best = None
            bd = None
            for k in range(len(vdisk3d)):
                dd = (vp[i][0] - vdisk3d[k][0]) ** 2 + (vp[i][1] - vdisk3d[k][1]) ** 2
                if bd is None or dd < bd:
                    bd = dd
                    best = k
            if bd is not None and bd < 0.03 ** 2 and (not kt or kt[-1][1] != best):
                kt.append((i, best))
    touches = kt
    log["touch_source"] = "kinematic"
    log["touches"] = [[t[0], round(vt[t[0]], 3), t[1]] for t in touches]

# live <-> video disk correspondence (left-to-right order in the fixed front camera)
corr = {}
if len(vdisks) == len(ldisks):
    for k in range(len(vdisks)):
        corr[k] = k
else:
    note("disk count mismatch video %d live %d; nearest-pixel matching" % (len(vdisks), len(ldisks)))
    for k in range(len(vdisks)):
        best = None
        bd = None
        for j in range(len(ldisks)):
            dd = (vdisks[k]["u"] - ldisks[j]["u"]) ** 2 + (vdisks[k]["v"] - ldisks[j]["v"]) ** 2
            if bd is None or dd < bd:
                bd = dd
                best = j
        corr[k] = best
log["corr"] = [[k, corr[k]] for k in corr]

# ---------------------------------------------------------------- anchors
anchors = []   # (frame, video EE pos, live goal pos, live disk idx)
for (fi, k) in touches:
    P = vp[fi]
    j = corr.get(k)
    if j is None or live3d[j] is None:
        note("touch at frame %d disk %d has no live 3d" % (fi, k))
        continue
    L = live3d[j]
    if vdisk3d[k] is not None:
        D = vdisk3d[k]
        dz = L[2] - D[2]
        if abs(dz) < 0.015:
            dz = 0.0
        G = [P[0] + L[0] - D[0], P[1] + L[1] - D[1], P[2] + dz]
        off = [P[0] - D[0], P[1] - D[1], P[2] - D[2]]
    else:
        G = [L[0], L[1], P[2]]
        off = None
    anchors.append((fi, P, G, j, off))
log["anchors"] = [[a[0], [round(x, 4) for x in a[1]], [round(x, 4) for x in a[2]], a[3],
                   None if a[4] is None else [round(x, 4) for x in a[4]]] for a in anchors]


def map_point(i, seg_a, seg_b):
    """Map video EE position of frame i, lying between anchors seg_a and seg_b (seg_a may be None)."""
    p = vp[i]
    if seg_a is None:
        fb, Pb, Gb = seg_b[0], seg_b[1], seg_b[2]
        span = vt[fb] - vt[0]
        s = 1.0 if span <= 1e-6 else max(0.0, min(1.0, (vt[i] - vt[0]) / span))
        return [p[0] + s * (Gb[0] - Pb[0]), p[1] + s * (Gb[1] - Pb[1]), p[2] + s * (Gb[2] - Pb[2])]
    fa, Pa, Ga = seg_a[0], seg_a[1], seg_a[2]
    fb, Pb, Gb = seg_b[0], seg_b[1], seg_b[2]
    span = vt[fb] - vt[fa]
    s = 1.0 if span <= 1e-6 else max(0.0, min(1.0, (vt[i] - vt[fa]) / span))
    vx = Pb[0] - Pa[0]
    vy = Pb[1] - Pa[1]
    gx = Gb[0] - Ga[0]
    gy = Gb[1] - Ga[1]
    nv2 = vx * vx + vy * vy
    dx = p[0] - Pa[0]
    dy = p[1] - Pa[1]
    if nv2 < 1e-6:
        x = p[0] + (Ga[0] - Pa[0])
        y = p[1] + (Ga[1] - Pa[1])
    else:
        # complex ratio (gx+i gy)/(vx+i vy) = scale*rotation
        cr = (gx * vx + gy * vy) / nv2
        ci = (gy * vx - gx * vy) / nv2
        x = Ga[0] + cr * dx - ci * dy
        y = Ga[1] + ci * dx + cr * dy
    z = p[2] + (1 - s) * (Ga[2] - Pa[2]) + s * (Gb[2] - Pb[2])
    return [x, y, z]


# ---------------------------------------------------------------- execution
S = {"ticks": 0, "terminal": False, "ik_fail": 0, "ik_calls": 0}
exec_log = []


def step_to(target):
    if S["terminal"] or S["ticks"] >= max_steps - 1:
        S["terminal"] = True
        return None
    reply = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=list(target), gripper_position=grip0, embodiment=emb)}))
    S["ticks"] += 1
    if is_terminal(reply):
        S["terminal"] = True
    return reply


def solve(pos, quat, seed):
    S["ik_calls"] += 1
    try:
        ik = solve_ik(Pose(list(pos), list(quat), base_frame), seed_joints=list(seed))
        if ik.ok:
            return ik_list(ik)
    except Exception as exc:
        note("ik exc " + str(exc)[:120])
    S["ik_fail"] += 1
    return None


def stream(waypoints, start_joints):
    """waypoints: list of (dt_video, joints). Linear joint interpolation, one step per tick."""
    cur = start_joints
    for wp in waypoints:
        dt, jt = wp[0], wp[1]
        n = max(1, int(round(dt * SLOW / 0.05)))
        jump = max(abs(jt[m] - cur[m]) for m in range(len(jt)))
        n = max(n, int(math.ceil(jump / MAX_JOINT_STEP)))
        for s in range(1, n + 1):
            a = s / float(n)
            step_to([cur[m] + a * (jt[m] - cur[m]) for m in range(len(jt))])
            if S["terminal"]:
                return jt
        cur = jt
    return cur


def disk_red_live(j):
    """(red pixels above baseline, visible disc pixels) in the live front view of disc j."""
    try:
        o = get_observation(camera_names=["front"])
        rows = to_list(o.cameras["front"].rgb)
        d = ldisks[j]
        return red_count(rows, d["u"], d["v"]) - live_base_red[j], disc_pixels(rows, d["u"], d["v"])
    except Exception:
        return -1, -1


live_base_red = [red_count(lrows, d["u"], d["v"]) for d in ldisks]

seg_prev = None
cur_j = live_j0
prev_frame = 0
for ai in range(len(anchors)):
    if S["terminal"]:
        break
    seg_b = anchors[ai]
    fa = 0 if seg_prev is None else seg_prev[0]
    fb = seg_b[0]
    # choose waypoint frames
    frames = []
    last_t = vt[fa]
    for i in range(fa + 1, fb):
        if vt[i] - last_t >= MIN_WAYPOINT_DT:
            frames.append(i)
            last_t = vt[i]
    frames.append(fb)
    wps = []
    seed = vj[frames[0]] if seg_prev is None else cur_j
    t_prev = vt[fa]
    acc_dt = 0.0
    for i in frames:
        pos = map_point(i, seg_prev, seg_b) if i != fb else seg_b[2]
        sol = solve(pos, vq[i], seed)
        if sol is None:
            sol = solve(pos, vq[i], vj[i])
        acc_dt += vt[i] - t_prev
        t_prev = vt[i]
        if sol is None:
            continue
        wps.append((acc_dt, sol))
        acc_dt = 0.0
        seed = sol
    if not wps:
        note("segment %d: no IK waypoints" % ai)
        seg_prev = seg_b
        continue
    cur_j = stream(wps, cur_j)
    for h in range(TOUCH_HOLD_TICKS):
        if S["terminal"]:
            break
        step_to(cur_j)
    entry = {"anchor": ai, "frame": fb, "disk": seg_b[3], "ticks": S["ticks"], "n_wp": len(wps)}
    if not S["terminal"]:
        try:
            st = get_robot_state()
            ee = fl(st.end_effector_poses["primary"].position)
            entry["ee"] = [round(x, 4) for x in ee]
            entry["err"] = round(math.sqrt(sum((ee[m] - seg_b[2][m]) ** 2 for m in range(3))), 4)
        except Exception as exc:
            entry["ee_err"] = str(exc)[:100]
        rv = disk_red_live(seg_b[3])
        red = rv[0]
        entry["red"] = red
        entry["disc_px"] = rv[1]
        nd = 0
        goal = list(seg_b[2])
        visible = (rv[0] + rv[1]) >= 8
        while (not S["terminal"]) and visible and red < RED_ON and nd < DESCEND_MAX:
            nd += 1
            goal = [goal[0], goal[1], goal[2] - DESCEND_STEP]
            sol = solve(goal, vq[fb], cur_j)
            if sol is None:
                break
            cur_j = stream([(0.25, sol)], cur_j)
            for h in range(TOUCH_HOLD_TICKS):
                if S["terminal"]:
                    break
                step_to(cur_j)
            if S["terminal"]:
                break
            rv = disk_red_live(seg_b[3])
            red = rv[0]
        entry["descend"] = nd
        entry["red_after"] = red
        if nd > 0 and not S["terminal"]:
            # come back up to the recorded touch height before the next segment
            sol = solve(seg_b[2], vq[fb], cur_j)
            if sol is not None:
                cur_j = stream([(0.2, sol)], cur_j)
    entry["terminal"] = S["terminal"]
    exec_log.append(entry)
    seg_prev = seg_b

log["exec"] = exec_log
log["ticks"] = S["ticks"]
log["terminal"] = S["terminal"]
log["ik_calls"] = S["ik_calls"]
log["ik_fail"] = S["ik_fail"]
result = log
