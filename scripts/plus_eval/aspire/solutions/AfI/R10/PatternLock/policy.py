import math

# PatternLock v06 -- v05 with the wrist tool-point filter fixed.
# v05 evidence (ep1): every wrist-depth point between table+4 cm and the EE was taken as "tool", which
# included the robot base at the image edge; the fitted stick axis was the base, the search start pose
# was unreachable and the robot never moved. v06 keeps only points closer than 0.30 m to the wrist
# camera and within 8 cm (EE-frame xy) / 0-25 cm (EE-frame z) of the EE, falls back to the tool z axis
# for an implausible fit, and retries the search start with shorter offsets if IK fails.
#
# (v05 header follows)
# PatternLock v05 -- contact-calibrated tool offset.
# Evidence (ep1, v01-v04): the grid transform is right (rms 0.5 mm) but no button ever lit. Offline
# analysis of the ep1 conditioning video (red-onset frame of each button vs recorded EE pose) shows the
# VIDEO tool tip sits a constant (0.005, -0.002, -0.1035) m from the EE at every one of 7 touches, i.e.
# exactly on the 1.56 cm button tops; the depth-based tip measurements of v02-v04 (4.9 cm) only saw the
# upper part of the stick, so all compensations were wrong. The live stick is shorter and tilted
# (wrist view), and its length cannot be read from depth either. v05:
#   1. grid similarity transform video->live from front depth (as v01-v04);
#   2. tool gate: if the grid is identical AND the wrist-depth tool signature matches the video's,
#      replay the recorded joints exactly at recorded speed (the v00 behaviour that solved ep0);
#   3. otherwise: (a) finds, in the conditioning video, the frame each button turns red and from it the
#      video tip offset o_v (median over touches) and the first touched button; (b) fits the visible
#      live stick axis from wrist depth; (c) puts the EE so that this axis passes through the mapped
#      first button and moves along the axis in 4 mm steps until that button turns red (front or wrist
#      RGB) -- this measures the live tip offset o_l by contact; (d) replays the video TIP path from the
#      first touch on, mapped through the grid transform, with the live EE placed by o_l (per-frame IK),
#      at 1.5x recorded speed.

CONTROL_HZ = 20.0
MIN_H = 0.004      # button top must rise this far above the table plane (m)
MAX_H = 0.06       # ... and no further (excludes the arm / tall structures)
MIN_BASE_R = 0.20  # ignore anything within this xy radius of the robot base origin
MIN_PX = 12
MAX_PX = 400
ID_ANGLE_DEG = 0.3
ID_SHIFT_M = 0.002
ID_SCALE = 0.01
ID_TOOL_M = 0.008      # wrist-depth tool signature difference treated as "same tool"
TIME_SCALE = 1.5       # video seconds replayed per real second (contact mode)
TIP_MIN_ABOVE_TABLE = 0.04
RED_R = 140
RED_GB = 90
RED_MIN_PX = 4
S_START = 0.09         # start this far (along the stick axis) beyond the lowest visible tool point
S_STEP = 0.004
S_MIN = -0.02
TICKS_PER_STEP = 3
FLOOR_ABOVE_TOP = 0.008  # never put the EE (finger level) lower than this above the button tops
TOOL_MAX_DEPTH = 0.30
TOOL_MAX_XY = 0.08
TOOL_MAX_Z = 0.25

def jl(state):
    return [float(v) for v in state.joint_positions["primary"]]


def ee_pose(state):
    p = state.end_effector_poses["primary"]
    return [float(v) for v in p.position], [float(v) for v in p.quaternion_wxyz]


def quat_to_rot(q):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def quat_mul(a, b):
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]


def detect_cells(cam):
    """Raised buttons on the table from one RGB-D camera: list of [x, y, ztop, npx] in robot_base."""
    depth = cam.depth_m.tolist()
    K = [[float(v) for v in row] for row in cam.intrinsics]
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    Rm = quat_to_rot([float(v) for v in cam.camera_pose.quaternion_wxyz])
    tc = [float(v) for v in cam.camera_pose.position]
    h = len(depth)
    w = len(depth[0])
    world = []
    zs = []
    for v in range(h):
        row = depth[v]
        out = []
        yn = (v + 0.5 - cy) / fy
        for u in range(w):
            d = float(row[u])
            if d <= 0.0 or d > 5.0:
                out.append(None)
                continue
            xc = (u + 0.5 - cx) / fx * d
            yc = yn * d
            X = Rm[0][0] * xc + Rm[0][1] * yc + Rm[0][2] * d + tc[0]
            Y = Rm[1][0] * xc + Rm[1][1] * yc + Rm[1][2] * d + tc[1]
            Z = Rm[2][0] * xc + Rm[2][1] * yc + Rm[2][2] * d + tc[2]
            out.append((X, Y, Z))
            if v % 4 == 0 and u % 4 == 0:
                zs.append(Z)
        world.append(out)
    zs.sort()
    table_z = zs[len(zs) // 2]
    mask = []
    for v in range(h):
        mrow = []
        for u in range(w):
            p = world[v][u]
            ok = False
            if p is not None:
                hz = p[2] - table_z
                if hz > MIN_H and hz < MAX_H and p[0] * p[0] + p[1] * p[1] > MIN_BASE_R * MIN_BASE_R:
                    ok = True
            mrow.append(ok)
        mask.append(mrow)
    seen = [[False] * w for r in range(h)]
    cells = []
    for v in range(h):
        for u in range(w):
            if not mask[v][u] or seen[v][u]:
                continue
            stack = [(v, u)]
            seen[v][u] = True
            sx = 0.0
            sy = 0.0
            zt = -1.0
            n = 0
            while stack:
                cur = stack.pop()
                p = world[cur[0]][cur[1]]
                sx += p[0]
                sy += p[1]
                zt = max(zt, p[2])
                n += 1
                for dv, du in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a = cur[0] + dv
                    b = cur[1] + du
                    if a >= 0 and a < h and b >= 0 and b < w and mask[a][b] and not seen[a][b]:
                        seen[a][b] = True
                        stack.append((a, b))
            if n >= MIN_PX and n <= MAX_PX:
                cells.append([sx / n, sy / n, zt, n])
    return cells, table_z


def grid_angle(pts):
    c4 = 0.0
    s4 = 0.0
    spacing = []
    for i in range(len(pts)):
        best = None
        for j in range(len(pts)):
            if i == j:
                continue
            dx = pts[j][0] - pts[i][0]
            dy = pts[j][1] - pts[i][1]
            dd = dx * dx + dy * dy
            if best is None or dd < best[0]:
                best = (dd, dx, dy)
        if best is not None:
            th = math.atan2(best[2], best[1])
            c4 += math.cos(4 * th)
            s4 += math.sin(4 * th)
            spacing.append(math.sqrt(best[0]))
    spacing.sort()
    sp = spacing[len(spacing) // 2] if spacing else 0.0
    return math.atan2(s4, c4) / 4.0, sp


def wrap45(a):
    q = math.pi / 2.0
    while a > q / 2.0:
        a -= q
    while a <= -q / 2.0:
        a += q
    return a


def fit_transform(vid, liv):
    av, spv = grid_angle(vid)
    al, spl = grid_angle(liv)
    th = wrap45(al - av)
    cvx = sum(p[0] for p in vid) / len(vid)
    cvy = sum(p[1] for p in vid) / len(vid)
    clx = sum(p[0] for p in liv) / len(liv)
    cly = sum(p[1] for p in liv) / len(liv)
    sc = spl / spv if spv > 0 and spl > 0 else 1.0
    c = sc * math.cos(th)
    s = sc * math.sin(th)
    tx = clx - (c * cvx - s * cvy)
    ty = cly - (s * cvx + c * cvy)
    gate = 0.5 * min(spv, spl) if min(spv, spl) > 0 else 0.03
    matched = 0
    rms = 0.0
    for it in range(5):
        c = sc * math.cos(th)
        s = sc * math.sin(th)
        pairs = []
        for p in vid:
            qx = c * p[0] - s * p[1] + tx
            qy = s * p[0] + c * p[1] + ty
            best = None
            for l in liv:
                dd = (l[0] - qx) ** 2 + (l[1] - qy) ** 2
                if best is None or dd < best[0]:
                    best = (dd, l)
            if best is not None and best[0] < gate * gate:
                pairs.append((p, best[1]))
        if len(pairs) < 3:
            break
        ax = sum(pp[0][0] for pp in pairs) / len(pairs)
        ay = sum(pp[0][1] for pp in pairs) / len(pairs)
        bx = sum(pp[1][0] for pp in pairs) / len(pairs)
        by = sum(pp[1][1] for pp in pairs) / len(pairs)
        sxx = 0.0
        sxy = 0.0
        saa = 0.0
        for pp in pairs:
            a0 = pp[0][0] - ax
            a1 = pp[0][1] - ay
            b0 = pp[1][0] - bx
            b1 = pp[1][1] - by
            sxx += a0 * b0 + a1 * b1
            sxy += a0 * b1 - a1 * b0
            saa += a0 * a0 + a1 * a1
        th = math.atan2(sxy, sxx)
        sc = math.sqrt(sxx * sxx + sxy * sxy) / saa if saa > 0 else 1.0
        c = sc * math.cos(th)
        s = sc * math.sin(th)
        tx = bx - (c * ax - s * ay)
        ty = by - (s * ax + c * ay)
        matched = len(pairs)
        err = 0.0
        for pp in pairs:
            qx = c * pp[0][0] - s * pp[0][1] + tx
            qy = s * pp[0][0] + c * pp[0][1] + ty
            err += (qx - pp[1][0]) ** 2 + (qy - pp[1][1]) ** 2
        rms = math.sqrt(err / len(pairs))
    return {"theta": th, "scale": sc, "tx": tx, "ty": ty, "matched": matched, "rms": rms,
            "spacing_video": spv, "spacing_live": spl, "angle_video": av, "angle_live": al}


def check_end(res):
    term = False
    trunc = False
    try:
        term = bool(res.terminated)
    except Exception:
        pass
    try:
        trunc = bool(res.truncated)
    except Exception:
        pass
    return term, trunc





def resample_run(targets, times, i0, i1, ticks, max_steps, grip, emb, log):
    """Step joint targets of frames i0..i1 (inclusive) at TIME_SCALE x recorded speed. Returns (ticks, ended)."""
    k = 0
    seg = i0
    t_end = times[i1]
    while True:
        k += 1
        vt = times[i0] + k * TIME_SCALE / CONTROL_HZ
        if vt > t_end:
            vt = t_end
        while seg < i1 - 1 and times[seg + 1] < vt:
            seg += 1
        span = times[seg + 1] - times[seg]
        f = (vt - times[seg]) / span if span > 0 else 1.0
        f = max(0.0, min(1.0, f))
        a = targets[seg]
        b = targets[seg + 1]
        tgt = [a[j] + (b[j] - a[j]) * f for j in range(7)]
        if ticks >= max_steps - 5:
            log["status"] = "step_budget"
            return ticks, True
        res = step(RobotAction(arms={"primary": ArmCommand(mode="joint_position", target=tgt,
                                                            gripper_position=grip, embodiment=emb)}))
        ticks += 1
        t1, t2 = check_end(res)
        if t1 or t2:
            log["terminated"] = t1
            log["truncated"] = t2
            log["end_video_t"] = round(vt, 3)
            return ticks, True
        if k % 30 == 0:
            try:
                pe, qe = ee_pose(get_robot_state())
                log["ee_samples"].append([round(vt, 2)] + [round(v, 4) for v in pe])
            except Exception as exc:
                log["ee_samples"].append([round(vt, 2), str(exc)])
        if vt >= t_end:
            return ticks, False


def hold(n, grip, emb, log):
    target = jl(get_robot_state())
    used = 0
    for m in range(n):
        res = step(RobotAction(arms={"primary": ArmCommand(mode="joint_position", target=target,
                                                            gripper_position=grip, embodiment=emb)}))
        used += 1
        t1, t2 = check_end(res)
        if t1 or t2:
            log["terminated"] = t1
            log["truncated"] = t2
            return used, True
    return used, False



def rot_t(R, v):
    return [R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2],
            R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2],
            R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2]]


def rot_v(R, v):
    return [R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
            R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
            R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]]


def project(cam, p):
    K = [[float(v) for v in row] for row in cam.intrinsics]
    Rm = quat_to_rot([float(v) for v in cam.camera_pose.quaternion_wxyz])
    tc = [float(v) for v in cam.camera_pose.position]
    pc = rot_t(Rm, [p[0] - tc[0], p[1] - tc[1], p[2] - tc[2]])
    if pc[2] <= 1e-6:
        return None
    return [int(round(pc[0] / pc[2] * K[0][0] + K[0][2] - 0.5)), int(round(pc[1] / pc[2] * K[1][1] + K[1][2] - 0.5))]


def patch_red(rows, uv, r):
    """Red pixels in a (2r+1)^2 patch of an RGB image given as nested lists."""
    u = uv[0]
    v = uv[1]
    h = len(rows)
    w = len(rows[0])
    n = 0
    for y in range(max(0, v - r), min(h, v + r + 1)):
        row = rows[y]
        for x in range(max(0, u - r), min(w, u + r + 1)):
            px = row[x]
            if int(px[0]) > RED_R and int(px[1]) < RED_GB and int(px[2]) < RED_GB:
                n += 1
    return n


def image_red(rows):
    n = 0
    for y in range(0, len(rows), 2):
        row = rows[y]
        for x in range(0, len(row), 2):
            px = row[x]
            if int(px[0]) > RED_R and int(px[1]) < RED_GB and int(px[2]) < RED_GB:
                n += 1
    return n


def tool_points(cam, ee_p, ee_q, table_z):
    """Tool points seen in wrist depth (world z above table+4 cm and below the EE), in the EE frame."""
    depth = cam.depth_m.tolist()
    K = [[float(v) for v in row] for row in cam.intrinsics]
    Rm = quat_to_rot([float(v) for v in cam.camera_pose.quaternion_wxyz])
    tc = [float(v) for v in cam.camera_pose.position]
    Re = quat_to_rot(ee_q)
    pts = []
    for v in range(len(depth)):
        row = depth[v]
        yn = (v + 0.5 - K[1][2]) / K[1][1]
        for u in range(len(row)):
            d = float(row[u])
            if d <= 0.0 or d > TOOL_MAX_DEPTH:
                continue
            pc = [(u + 0.5 - K[0][2]) / K[0][0] * d, yn * d, d]
            pw = rot_v(Rm, pc)
            pw = [pw[0] + tc[0], pw[1] + tc[1], pw[2] + tc[2]]
            if pw[2] > table_z + TIP_MIN_ABOVE_TABLE and pw[2] < ee_p[2]:
                pl = rot_t(Re, [pw[0] - ee_p[0], pw[1] - ee_p[1], pw[2] - ee_p[2]])
                if abs(pl[0]) < TOOL_MAX_XY and abs(pl[1]) < TOOL_MAX_XY and pl[2] > 0.0 and pl[2] < TOOL_MAX_Z:
                    pts.append(pl)
    return pts


def lowest_signature(pts):
    if len(pts) < 5:
        return None
    s = [t[1] for t in sorted([[-p[2], p] for p in pts])[:5]]
    return [sum(p[k] for p in s) / 5.0 for k in range(3)]


def fit_axis(pts):
    n = len(pts)
    m = [sum(p[k] for p in pts) / n for k in range(3)]
    C = [[0.0] * 3 for k in range(3)]
    for p in pts:
        d = [p[0] - m[0], p[1] - m[1], p[2] - m[2]]
        for a in range(3):
            for b in range(3):
                C[a][b] += d[a] * d[b]
    a = [0.0, 0.0, 1.0]
    for it in range(50):
        b = rot_v(C, a)
        nb = math.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])
        if nb <= 0:
            break
        a = [b[0] / nb, b[1] / nb, b[2] / nb]
    if a[2] < 0:
        a = [-a[0], -a[1], -a[2]]
    proj = [t[1] for t in sorted([[-((p[0] - m[0]) * a[0] + (p[1] - m[1]) * a[1] + (p[2] - m[2]) * a[2]), p] for p in pts])[:5]]
    L = [sum(p[k] for p in proj) / len(proj) for k in range(3)]
    return m, a, L


def median(vals):
    s = sorted(vals)
    return s[len(s) // 2]


def run_joint_replay(video, nv, grip, emb, max_steps, log):
    """v00 behaviour: recorded joints at recorded timestamps."""
    first = jl(video[0].robot_state)
    mv = move_to_joints(first, tolerance=0.01, max_steps=150)
    ticks = int(mv.steps_executed)
    t1, t2 = check_end(mv)
    if t1 or t2:
        return ticks, True
    prev = first
    prev_t = float(video[0].timestamp_s)
    for i in range(1, nv):
        cur = jl(video[i].robot_state)
        cur_t = float(video[i].timestamp_s)
        n = int(round((cur_t - prev_t) * CONTROL_HZ))
        if n < 1:
            n = 1
        for k in range(1, n + 1):
            if ticks >= max_steps - 5:
                log["status"] = "step_budget"
                return ticks, True
            f = float(k) / float(n)
            tgt = [prev[j] + (cur[j] - prev[j]) * f for j in range(7)]
            res = step(RobotAction(arms={"primary": ArmCommand(mode="joint_position", target=tgt,
                                                                gripper_position=grip, embodiment=emb)}))
            ticks += 1
            t1, t2 = check_end(res)
            if t1 or t2:
                log["terminated"] = t1
                log["truncated"] = t2
                log["end_frame"] = i
                return ticks, True
        prev = cur
        prev_t = cur_t
    return ticks, False


report = {"version": "v06"}
context = get_task_context()
report["language"] = context.language
meta = dict(robomme_plus.get_task_metadata())
max_steps = int(meta.get("max_steps", 1000))
report["max_steps"] = max_steps

state0 = get_robot_state()
base = state0.base_frame
emb = state0.embodiment
grip = float(state0.gripper_positions["primary"])
live = get_observation()
video = robomme_plus.get_conditioning_video()
nv = len(video)
report["video_len"] = nv
if nv < 3:
    raise ValueError("no conditioning video to reproduce")

vid_cells, vid_table = detect_cells(video[0].cameras["front"])
liv_cells, liv_table = detect_cells(live.cameras["front"])
report["n_cells"] = [len(vid_cells), len(liv_cells)]
report["table_z"] = [round(vid_table, 4), round(liv_table, 4)]

theta = 0.0
scale = 1.0
tx = 0.0
ty = 0.0
dz = 0.0
grid_identity = True
grid_ok = False
if len(vid_cells) >= 4 and len(liv_cells) >= 4:
    tf = fit_transform(vid_cells, liv_cells)
    zv = sorted(c[2] for c in vid_cells)[len(vid_cells) // 2]
    zl = sorted(c[2] for c in liv_cells)[len(liv_cells) // 2]
    cvx = sum(p[0] for p in vid_cells) / len(vid_cells)
    cvy = sum(p[1] for p in vid_cells) / len(vid_cells)
    c = tf["scale"] * math.cos(tf["theta"])
    s = tf["scale"] * math.sin(tf["theta"])
    shift = math.sqrt((c * cvx - s * cvy + tf["tx"] - cvx) ** 2 + (s * cvx + c * cvy + tf["ty"] - cvy) ** 2)
    usable = tf["matched"] >= max(4, (2 * min(len(vid_cells), len(liv_cells))) // 3)
    report["transform"] = {"theta_deg": round(tf["theta"] * 180.0 / math.pi, 3), "scale": round(tf["scale"], 4),
                           "tx": round(tf["tx"], 4), "ty": round(tf["ty"], 4), "dz": round(zl - zv, 4),
                           "matched": tf["matched"], "rms_m": round(tf["rms"], 4),
                           "spacing": [round(tf["spacing_video"], 4), round(tf["spacing_live"], 4)],
                           "centre_shift_m": round(shift, 4), "usable": usable}
    if usable:
        grid_ok = True
        theta = tf["theta"]
        scale = tf["scale"]
        tx = tf["tx"]
        ty = tf["ty"]
        dz = zl - zv
        grid_identity = (abs(theta) * 180.0 / math.pi <= ID_ANGLE_DEG and shift <= ID_SHIFT_M
                         and abs(scale - 1.0) <= ID_SCALE)
else:
    report["transform"] = {"error": "grid not detected", "n_video": len(vid_cells), "n_live": len(liv_cells)}
report["grid_identity"] = grid_identity

c_s = scale * math.cos(theta)
s_s = scale * math.sin(theta)
qz = [math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)]
times = [float(video[i].timestamp_s) for i in range(nv)]
vposes = [ee_pose(video[i].robot_state) for i in range(nv)]


def sim2(p):
    return [c_s * p[0] - s_s * p[1] + tx, s_s * p[0] + c_s * p[1] + ty, p[2] + dz]


# ---- tool signature (wrist depth), live vs video frame 0 ----
p_l0, q_l0 = ee_pose(state0)
p_v0, q_v0 = vposes[0]
pts_l = tool_points(live.cameras["wrist"], p_l0, q_l0, liv_table)
pts_v = tool_points(video[0].cameras["wrist"], p_v0, q_v0, vid_table)
sig_l = lowest_signature(pts_l)
sig_v = lowest_signature(pts_v)
tool_same = False
toolinfo = {"n_live": len(pts_l), "n_video": len(pts_v)}
if sig_l is not None and sig_v is not None:
    dtool = math.sqrt(sum((sig_l[k] - sig_v[k]) ** 2 for k in range(3)))
    toolinfo.update({"sig_live": [round(v, 4) for v in sig_l], "sig_video": [round(v, 4) for v in sig_v],
                     "diff_m": round(dtool, 4)})
    tool_same = dtool <= ID_TOOL_M
elif sig_l is None and sig_v is None:
    tool_same = True
toolinfo["same"] = tool_same
report["tool"] = toolinfo

replay = {"ticks": 0, "terminated": False, "truncated": False, "status": "running", "ee_samples": []}
ticks = 0
ended = False
mode = "joint_replay" if (grid_identity and tool_same) else "contact_calibrated"
if mode == "contact_calibrated" and not grid_ok:
    mode = "joint_replay"
    report["mode_note"] = "grid not usable; falling back to joint replay"
report["mode"] = mode

if mode == "joint_replay":
    ticks, ended = run_joint_replay(video, nv, grip, emb, max_steps, replay)
else:
    # (a) button red-onset frames in the conditioning video
    cam_v = video[0].cameras["front"]
    vuv = [project(cam_v, [c[0], c[1], c[2]]) for c in vid_cells]
    onset = [None] * len(vid_cells)
    for i in range(nv):
        rgb = video[i].cameras["front"].rgb.tolist()
        for ci in range(len(vid_cells)):
            if onset[ci] is None and vuv[ci] is not None and patch_red(rgb, vuv[ci], 2) >= RED_MIN_PX:
                onset[ci] = i
    touches = sorted([[onset[ci], ci] for ci in range(len(vid_cells)) if onset[ci] is not None and onset[ci] > 0])
    report["video_touches"] = [[t[0], t[1], [round(v, 3) for v in vid_cells[t[1]][:3]]] for t in touches]
    offs = []
    for t in touches:
        cl = vid_cells[t[1]]
        p, q = vposes[t[0]]
        offs.append(rot_t(quat_to_rot(q), [cl[0] - p[0], cl[1] - p[1], cl[2] - p[2]]))
    if len(offs) >= 1:
        o_v = [median([o[k] for o in offs]) for k in range(3)]
        spread = max(math.sqrt(sum((o[k] - o_v[k]) ** 2 for k in range(3))) for o in offs)
        report["o_v_local"] = [round(v, 4) for v in o_v]
        report["o_v_spread_m"] = round(spread, 4)
        k1 = touches[0][0]
        c1 = vid_cells[touches[0][1]]
    else:
        o_v = sig_v if sig_v is not None else [0.0, 0.0, 0.1]
        report["o_v_local"] = [round(v, 4) for v in o_v]
        report["o_v_note"] = "no red onsets found in the video; fallback to wrist signature"
        zmin = min(p[0][2] for p in vposes)
        k1 = 0
        for i in range(nv):
            if vposes[i][0][2] < zmin + 0.003:
                k1 = i
                break
        tipw = rot_v(quat_to_rot(vposes[k1][1]), o_v)
        tp = [vposes[k1][0][k] + tipw[k] for k in range(3)]
        c1 = sorted([[(c[0] - tp[0]) ** 2 + (c[1] - tp[1]) ** 2, c] for c in vid_cells])[0][1]
    # first touched button in the live grid
    m1 = sim2([c1[0], c1[1], c1[2]])
    lc = sorted([[(c[0] - m1[0]) ** 2 + (c[1] - m1[1]) ** 2, c] for c in liv_cells])[0][1]
    P = [lc[0], lc[1], lc[2]]
    top_live = median([c[2] for c in liv_cells])
    report["first_touch"] = {"video_frame": k1, "video_cell": [round(v, 4) for v in c1[:3]],
                             "live_cell": [round(v, 4) for v in P]}
    # (b) live stick axis (EE frame) from wrist depth
    if len(pts_l) >= 10:
        m_ax, a_ax, L_ax = fit_axis(pts_l)
        if a_ax[2] < 0.5:
            a_ax = [0.0, 0.0, 1.0]
            report["axis_note"] = "implausible axis; using tool z"
        if abs(L_ax[0]) > TOOL_MAX_XY or abs(L_ax[1]) > TOOL_MAX_XY or L_ax[2] < 0.0 or L_ax[2] > TOOL_MAX_Z:
            L_ax = [0.0, 0.0, 0.0]
            report["axis_note"] = "implausible lowest point; using the EE reference"
    else:
        a_ax = [0.0, 0.0, 1.0]
        L_ax = sig_l if sig_l is not None else [0.0, 0.0, 0.0]
    report["live_axis"] = {"dir": [round(v, 4) for v in a_ax], "lowest_visible": [round(v, 4) for v in L_ax],
                           "n": len(pts_l)}
    # (c) contact search along the axis above the first button
    q_t = quat_mul(qz, vposes[k1][1])
    R_t = quat_to_rot(q_t)
    # buttons already red before the search are ignored
    cam_l = live.cameras["front"]
    luv = [project(cam_l, [c[0], c[1], c[2]]) for c in liv_cells]
    live_front_rows = cam_l.rgb.tolist()
    red0 = [luv[ci] is not None and patch_red(live_front_rows, luv[ci], 2) >= RED_MIN_PX for ci in range(len(liv_cells))]
    wred0 = image_red(live.cameras["wrist"].rgb.tolist())

    def ee_for(s):
        tl = [L_ax[k] + s * a_ax[k] for k in range(3)]
        tw = rot_v(R_t, tl)
        return [P[0] - tw[0], P[1] - tw[1], P[2] - tw[2]]

    search = {"steps": 0, "contact": False}
    s = S_START
    seed = jl(state0)
    ik0 = solve_ik(Pose(ee_for(s), q_t, base), seed_joints=seed)
    while not ik0.ok and s > 0.03:
        s -= 0.02
        ik0 = solve_ik(Pose(ee_for(s), q_t, base), seed_joints=seed)
    search["s_start"] = round(s, 4)
    if ik0.ok:
        seed = [float(v) for v in ik0.joint_positions]
        mv = move_to_joints(seed, tolerance=0.005, max_steps=120)
        ticks += int(mv.steps_executed)
        t1, t2 = check_end(mv)
        ended = t1 or t2
    else:
        search["error"] = "ik failed at search start"
    hit_cell = None
    ee_c = None
    while not ended and ik0.ok and s > S_MIN:
        s_new = s - S_STEP
        tgt_p = ee_for(s_new)
        if tgt_p[2] < top_live + FLOOR_ABOVE_TOP:
            search["floor"] = True
            break
        ik = solve_ik(Pose(tgt_p, q_t, base), seed_joints=seed)
        if not ik.ok:
            search["ik_fail_at"] = round(s_new, 4)
            break
        nxt = [float(v) for v in ik.joint_positions]
        for k in range(1, TICKS_PER_STEP + 1):
            f = float(k) / float(TICKS_PER_STEP)
            tgt = [seed[j] + (nxt[j] - seed[j]) * f for j in range(7)]
            res = step(RobotAction(arms={"primary": ArmCommand(mode="joint_position", target=tgt,
                                                                gripper_position=grip, embodiment=emb)}))
            ticks += 1
            t1, t2 = check_end(res)
            if t1 or t2:
                replay["terminated"] = t1
                replay["truncated"] = t2
                ended = True
                break
        seed = nxt
        s = s_new
        search["steps"] += 1
        if ended:
            break
        ob = get_observation()
        fr = ob.cameras["front"].rgb.tolist()
        for ci in range(len(liv_cells)):
            if not red0[ci] and luv[ci] is not None and patch_red(fr, luv[ci], 2) >= RED_MIN_PX:
                hit_cell = ci
                break
        wr = image_red(ob.cameras["wrist"].rgb.tolist())
        if hit_cell is not None or wr >= wred0 + 8:
            search["contact"] = True
            search["by"] = "front" if hit_cell is not None else "wrist"
            search["wrist_red"] = wr
            ee_c = ee_pose(ob.robot_state)
            break
    search["s_final"] = round(s, 4)
    if ee_c is None:
        ee_c = ee_pose(get_robot_state())
    cell_c = P
    if hit_cell is not None:
        cell_c = [liv_cells[hit_cell][0], liv_cells[hit_cell][1], liv_cells[hit_cell][2]]
        search["hit_cell"] = [round(v, 4) for v in cell_c]
        search["hit_is_target"] = abs(cell_c[0] - P[0]) + abs(cell_c[1] - P[1]) < 1e-6
    search["ee_contact"] = [round(v, 4) for v in ee_c[0]]
    if search["contact"]:
        o_l = rot_t(R_t, [cell_c[0] - ee_c[0][0], cell_c[1] - ee_c[0][1], cell_c[2] - ee_c[0][2]])
    else:
        o_l = [L_ax[k] + s * a_ax[k] for k in range(3)]
    report["o_l_local"] = [round(v, 4) for v in o_l]
    report["search"] = search

    # (d) mapped tip path from the first touch on
    def map_pose(i):
        p, q = vposes[i]
        tw = rot_v(quat_to_rot(q), o_v)
        tip = sim2([p[0] + tw[0], p[1] + tw[1], p[2] + tw[2]])
        nq = quat_mul(qz, q)
        lw = rot_v(quat_to_rot(nq), o_l)
        return [tip[0] - lw[0], tip[1] - lw[1], tip[2] - lw[2]], nq

    targets = [None] * nv
    prev = jl(get_robot_state())
    last_p = None
    fails = 0
    for i in range(k1, nv):
        np_, nq = map_pose(i)
        if last_p is None or abs(np_[0] - last_p[0]) + abs(np_[1] - last_p[1]) + abs(np_[2] - last_p[2]) >= 0.0005:
            ik = solve_ik(Pose(np_, nq, base), seed_joints=prev)
            if ik.ok:
                prev = [float(v) for v in ik.joint_positions]
                last_p = np_
            else:
                fails += 1
        targets[i] = list(prev)
    replay["ik_fail"] = fails
    report["ee_targets"] = [[i] + [round(v, 4) for v in map_pose(i)[0]] for i in range(k1, nv, 50)]
    if not ended:
        mv2 = move_to_joints(targets[k1], tolerance=0.005, max_steps=30)
        ticks += int(mv2.steps_executed)
        t1, t2 = check_end(mv2)
        ended = t1 or t2
    if not ended:
        ticks, ended = resample_run(targets, times, k1, nv - 1, ticks, max_steps, grip, emb, replay)
if not ended:
    used, ended = hold(20, grip, emb, replay)
    ticks += used
replay["ticks"] = ticks
replay["time_scale"] = TIME_SCALE if mode != "joint_replay" else 1.0
if replay["status"] == "running":
    replay["status"] = "ended" if ended else "completed"
report["replay"] = replay
result = report
