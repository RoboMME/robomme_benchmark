# PatternLock v00 -- reproduce the conditioning-video stick pattern on the live grid.
#
# Strategy (all quantities derived at runtime):
#  1. Read the conditioning video (robot state per frame) -> recorded EE pose trajectory.
#  2. Detect the 5x5 target grid in the video's first front frame and in the live front frame
#     from depth (targets are raised discs above the fitted table plane).
#  3. Fit lattice pose (centre, yaw mod 90, spacing) for both -> rigid xy/yaw transform T.
#  4. Estimate the stick-tip offset in the EE frame from the wrist depth (video frame 0 and live),
#     because tool geometry may differ between the video and the live episode.
#  5. Live tip path = T(video tip path); EE pose = tip - R' * live_offset; IK per frame.
#     Identity transform + same tool -> replay the recorded joints directly.
#  6. At the first contact frame, verify a target turned red; otherwise descend in small steps.
import math

LOG = {"version": "v00"}
CTRL_HZ = 20.0


def fl(v):
    return [float(x) for x in v]


def quat_to_R(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def mv(R, v):
    return [R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
            R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
            R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]]


def mtv(R, v):
    return [R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2],
            R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2],
            R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2]]


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def to_list2(arr):
    try:
        return arr.tolist()
    except Exception:
        return [[v for v in row] for row in arr]


def median(vals):
    s = sorted(vals)
    if not s:
        return None
    return s[len(s) // 2]


def wrap90(a):
    # wrap angle to (-pi/4, pi/4]
    while a > math.pi / 4:
        a -= math.pi / 2
    while a <= -math.pi / 4:
        a += math.pi / 2
    return a


def cam_setup(cam):
    K = to_list2(cam.intrinsics)
    pose = cam.camera_pose
    m = {"fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]), "cy": float(K[1][2]),
         "R": quat_to_R(fl(pose.quaternion_wxyz)), "t": fl(pose.position)}
    return m


def deproj(m, u, v, d, conv):
    # conv: (gl, inverse, ray)
    gl, inv, ray = conv
    x = (u - m["cx"]) / m["fx"]
    y = (v - m["cy"]) / m["fy"]
    if ray:
        d = d / math.sqrt(1.0 + x * x + y * y)
    pc = [x * d, y * d, d]
    if gl:
        pc = [pc[0], -pc[1], -pc[2]]
    if inv:
        R = m["R"]
        t = m["t"]
        q = [pc[0] - t[0], pc[1] - t[1], pc[2] - t[2]]
        return mtv(R, q)
    p = mv(m["R"], pc)
    return [p[0] + m["t"][0], p[1] + m["t"][1], p[2] + m["t"][2]]


CONVS = []
for g in (False, True):
    for inv in (False, True):
        for ray in (False, True):
            CONVS.append((g, inv, ray))


def fit_plane_z(pts):
    # least squares z = a x + b y + c with two trimming passes
    use = pts
    coef = None
    for it in range(3):
        n = len(use)
        if n < 10:
            break
        sx = sy = sz = sxx = syy = sxy = sxz = syz = 0.0
        for p in use:
            x, y, z = p
            sx += x; sy += y; sz += z
            sxx += x * x; syy += y * y; sxy += x * y; sxz += x * z; syz += y * z
        A = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, float(n)]]
        bvec = [sxz, syz, sz]
        coef = solve3(A, bvec)
        if coef is None:
            break
        res = [abs(p[2] - (coef[0] * p[0] + coef[1] * p[1] + coef[2])) for p in use]
        thr = max(sorted(res)[int(0.6 * (len(res) - 1))] * 2.0, 0.0005)
        use = [p for p, r in zip(use, res) if r <= thr]
    if coef is None:
        return None, None
    res = sorted([abs(p[2] - (coef[0] * p[0] + coef[1] * p[1] + coef[2])) for p in use])
    return coef, res[len(res) // 2]


def solve3(A, b):
    M = [A[0][:] + [b[0]], A[1][:] + [b[1]], A[2][:] + [b[2]]]
    for c in range(3):
        piv = c
        for r in range(c + 1, 3):
            if abs(M[r][c]) > abs(M[piv][c]):
                piv = r
        if abs(M[piv][c]) < 1e-12:
            return None
        M[c], M[piv] = M[piv], M[c]
        for r in range(3):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, 4):
                    M[r][k] -= f * M[c][k]
    return [M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2]]


def choose_conv(m, depth, H, W):
    # pick the deprojection convention for which the table (lower image) is flattest & horizontal
    samples = []
    for v in range(int(H * 0.45), H, 6):
        for u in range(0, W, 6):
            d = float(depth[v][u])
            if d > 0.05 and d < 5.0:
                samples.append((u, v, d))
    best = None
    diag = []
    for conv in CONVS:
        pts = [deproj(m, u, v, d, conv) for (u, v, d) in samples]
        coef, res = fit_plane_z(pts)
        if coef is None:
            continue
        tilt = math.sqrt(coef[0] ** 2 + coef[1] ** 2)
        score = res + 0.05 * tilt
        # the table must lie below the camera
        tz = coef[0] * m["t"][0] + coef[1] * m["t"][1] + coef[2]
        if tz > m["t"][2] - 0.05:
            score += 10.0
        diag.append([str(conv), round(res, 5), round(tilt, 4), round(coef[2], 4)])
        if best is None or score < best[0]:
            best = (score, conv, coef)
    return best, diag


def detect_grid(obs_cam, conv=None, label="", hmin=0.002, hmax=0.04):
    info = {"label": label}
    m = cam_setup(obs_cam)
    depth = to_list2(obs_cam.depth_m)
    H = len(depth)
    W = len(depth[0])
    if conv is None:
        best, diag = choose_conv(m, depth, H, W)
        info["conv_diag"] = diag
        conv = best[1]
    info["conv"] = str(conv)
    # deproject everything, fit table plane on all points
    P = [[None] * W for r in range(H)]
    allpts = []
    for v in range(H):
        row = depth[v]
        for u in range(W):
            d = float(row[u])
            if d > 0.05 and d < 5.0:
                p = deproj(m, u, v, d, conv)
                P[v][u] = p
                if (u % 3 == 0) and (v % 3 == 0):
                    allpts.append(p)
    coef, res = fit_plane_z(allpts)
    info["plane"] = [round(c, 5) for c in coef]
    info["plane_res"] = res
    cand = set()
    hvals = []
    for v in range(H):
        for u in range(W):
            p = P[v][u]
            if p is None:
                continue
            h = p[2] - (coef[0] * p[0] + coef[1] * p[1] + coef[2])
            if h > hmin and h < hmax:
                cand.add((u, v))
                hvals.append(h)
    info["n_cand_px"] = len(cand)
    # connected components
    comps = []
    seen = set()
    for start in list(cand):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        pix = []
        while stack:
            cu, cv = stack.pop()
            pix.append((cu, cv))
            for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (cu + du, cv + dv)
                if q in cand and q not in seen:
                    seen.add(q)
                    stack.append(q)
        if len(pix) >= 4:
            xs = [P[b][a] for (a, b) in pix]
            cx = sum(p[0] for p in xs) / len(xs)
            cy = sum(p[1] for p in xs) / len(xs)
            cz = max(p[2] for p in xs)
            hs = [p[2] - (coef[0] * p[0] + coef[1] * p[1] + coef[2]) for p in xs]
            comps.append({"n": len(pix), "xyz": [cx, cy, cz],
                          "px": [sum(a for a, b in pix) / len(pix), sum(b for a, b in pix) / len(pix)],
                          "hmax": max(hs)})
    info["n_comps"] = len(comps)
    # keep similar-size components
    if comps:
        med = median([c["n"] for c in comps if c["n"] >= 8] or [c["n"] for c in comps])
        comps2 = [c for c in comps if c["n"] >= 0.25 * med and c["n"] <= 4.0 * med]
    else:
        comps2 = []
    info["n_comps_sized"] = len(comps2)
    info["comps"] = [[round(c["xyz"][0], 4), round(c["xyz"][1], 4), round(c["xyz"][2], 4), c["n"],
                      round(c["px"][0], 1), round(c["px"][1], 1), round(c["hmax"], 4)] for c in comps2]
    return comps2, info, conv, coef


def fit_lattice(points, label=""):
    # points: list of [x, y]; returns centre, theta (wrapped to +-45deg), spacing, indices
    info = {"label": label, "n": len(points)}
    n = len(points)
    if n < 6:
        info["error"] = "too few points"
        return None, info
    nn = []
    for i in range(n):
        best = None
        for j in range(n):
            if i == j:
                continue
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            d = math.sqrt(dx * dx + dy * dy)
            if best is None or d < best[0]:
                best = (d, dx, dy)
        nn.append(best)
    s = median([b[0] for b in nn])
    # keep only lattice-spaced NN vectors
    c4 = 0.0
    s4 = 0.0
    for b in nn:
        if abs(b[0] - s) < 0.25 * s:
            a = math.atan2(b[2], b[1])
            c4 += math.cos(4 * a)
            s4 += math.sin(4 * a)
    th = wrap90(math.atan2(s4, c4) / 4.0)
    ct = math.cos(th)
    st = math.sin(th)
    # lattice coords relative to the first point
    p0 = points[0]
    q = []
    for p in points:
        dx = p[0] - p0[0]
        dy = p[1] - p0[1]
        q.append([(ct * dx + st * dy) / s, (-st * dx + ct * dy) / s])
    iq = [[int(round(a)), int(round(b))] for a, b in q]
    resid = median([abs(a - ia) + abs(b - ib) for (a, b), (ia, ib) in zip(q, iq)])
    mi = min(a for a, b in iq)
    mj = min(b for a, b in iq)
    iq = [[a - mi, b - mj] for a, b in iq]
    span = [max(a for a, b in iq), max(b for a, b in iq)]
    # refine spacing & centre by least squares on index (centre = index 2,2)
    sx = sy = 0.0
    for p, (a, b) in zip(points, iq):
        ga = (a - 2) * s
        gb = (b - 2) * s
        sx += p[0] - (ct * ga - st * gb)
        sy += p[1] - (st * ga + ct * gb)
    centre = [sx / n, sy / n]
    info.update({"spacing": round(s, 5), "theta_deg": round(math.degrees(th), 3), "centre": [round(centre[0], 5), round(centre[1], 5)],
                 "span": span, "resid": round(resid, 4), "unique": len(set((a, b) for a, b in iq))})
    return {"c": centre, "th": th, "s": s, "idx": iq}, info


def red_count(obs):
    total = 0
    for name in ("front", "wrist"):
        try:
            rgb = to_list2(obs.cameras[name].rgb)
        except Exception:
            continue
        for row in rgb:
            for px in row:
                if px[0] > 150 and px[1] < 80 and px[2] < 80:
                    total += 1
    return total


def tip_offset(obs, ee_pose, label):
    # stick tip from wrist depth: lowest point well above the table, expressed in EE frame
    info = {"label": label}
    cam = obs.cameras["wrist"]
    m = cam_setup(cam)
    depth = to_list2(cam.depth_m)
    H = len(depth)
    W = len(depth[0])
    best, diag = choose_conv(m, depth, H, W)
    conv = best[1]
    coef = best[2]
    info["conv"] = str(conv)
    info["conv_diag"] = diag
    pts = []
    for v in range(0, H, 1):
        row = depth[v]
        for u in range(0, W, 1):
            d = float(row[u])
            if d <= 0.0 or d > 5.0:
                continue
            p = deproj(m, u, v, d, conv)
            h = p[2] - (coef[0] * p[0] + coef[1] * p[1] + coef[2])
            if h > 0.03:
                pts.append((p[2], p, u, v, d))
    info["n_tool_px"] = len(pts)
    if len(pts) < 5:
        info["error"] = "no tool pixels"
        return None, info
    pts = sorted(pts)
    low = pts[:max(3, len(pts) // 50)]
    tip = [sum(e[1][k] for e in low) / len(low) for k in range(3)]
    info["tip_base"] = [round(x, 4) for x in tip]
    info["tip_px"] = [low[0][2], low[0][3]]
    info["table_z_under_tip"] = round(coef[0] * tip[0] + coef[1] * tip[1] + coef[2], 4)
    R = quat_to_R(fl(ee_pose.quaternion_wxyz))
    pe = fl(ee_pose.position)
    off = mtv(R, [tip[0] - pe[0], tip[1] - pe[1], tip[2] - pe[2]])
    info["offset_ee"] = [round(x, 4) for x in off]
    return off, info


def is_terminal(res):
    t = False
    try:
        t = bool(res.terminated) or bool(res.truncated)
    except Exception:
        t = False
    return t


# ------------------------------------------------------------------ main
ctx = get_task_context()
LOG["goal"] = ctx.language
try:
    meta = robomme_plus.get_task_metadata()
    LOG["meta"] = {str(k): str(meta[k])[:200] for k in meta}
except Exception as e:
    meta = {}
    LOG["meta_error"] = str(e)
max_steps = None
try:
    max_steps = int(meta["max_steps"])
except Exception:
    max_steps = None
try:
    cm = robomme_plus.get_controller_metadata()
    LOG["ctrl_meta"] = {str(k): str(cm[k])[:200] for k in cm}
except Exception as e:
    LOG["ctrl_meta_error"] = str(e)

live = get_observation()
state0 = live.robot_state
emb = state0.embodiment
q_live0 = fl(state0.joint_positions["primary"])
ee_live0 = state0.end_effector_poses["primary"]
grip0 = float(state0.gripper_positions["primary"])
LOG["live_q0"] = [round(x, 4) for x in q_live0]
LOG["live_ee0"] = fl(ee_live0.position) + fl(ee_live0.quaternion_wxyz)

video = robomme_plus.get_conditioning_video()
nv = len(video)
LOG["video_len"] = nv
VQ = []
VP = []
VQUAT = []
VT = []
for k in range(nv):
    rs = video[k].robot_state
    VQ.append(fl(rs.joint_positions["primary"]))
    e = rs.end_effector_poses["primary"]
    VP.append(fl(e.position))
    VQUAT.append(fl(e.quaternion_wxyz))
    VT.append(float(video[k].timestamp_s))
LOG["video_t"] = [VT[0], VT[-1]] if nv else None
LOG["video_q0"] = [round(x, 4) for x in VQ[0]] if nv else None

# ---------- grid detection
comps_v, info_v, conv_f, plane_v = detect_grid(video[0].cameras["front"], None, "video0_front")
comps_l, info_l, conv_l, plane_l = detect_grid(live.cameras["front"], None, "live_front")
LOG["grid_det_video"] = info_v
LOG["grid_det_live"] = info_l
lat_v, linfo_v = fit_lattice([[c["xyz"][0], c["xyz"][1]] for c in comps_v], "video")
lat_l, linfo_l = fit_lattice([[c["xyz"][0], c["xyz"][1]] for c in comps_l], "live")
LOG["lattice_video"] = linfo_v
LOG["lattice_live"] = linfo_l

dth = 0.0
scale = 1.0
cv_ = [0.0, 0.0]
cl_ = [0.0, 0.0]
transform_ok = lat_v is not None and lat_l is not None
if transform_ok:
    dth = wrap90(lat_l["th"] - lat_v["th"])
    scale = lat_l["s"] / lat_v["s"]
    if abs(scale - 1.0) > 0.15:
        LOG["scale_warning"] = scale
        scale = 1.0
    cv_ = lat_v["c"]
    cl_ = lat_l["c"]
shift = math.sqrt((cl_[0] - cv_[0]) ** 2 + (cl_[1] - cv_[1]) ** 2)
LOG["transform"] = {"ok": transform_ok, "dtheta_deg": round(math.degrees(dth), 3), "scale": round(scale, 4),
                    "shift_m": round(shift, 4), "cv": cv_, "cl": cl_}

# ---------- tool tip
off_v, tinfo_v = tip_offset(video[0], video[0].robot_state.end_effector_poses["primary"], "video0")
off_l, tinfo_l = tip_offset(live, ee_live0, "live")
LOG["tip_video"] = tinfo_v
LOG["tip_live"] = tinfo_l
if off_v is None or off_l is None:
    off_v = [0.0, 0.0, 0.0]
    off_l = [0.0, 0.0, 0.0]
    LOG["tip_fallback"] = True
doff = math.sqrt(sum((a - b) ** 2 for a, b in zip(off_v, off_l)))
LOG["tip_offset_diff_m"] = round(doff, 4)

identity = transform_ok and abs(math.degrees(dth)) < 1.0 and shift < 0.004 and abs(scale - 1.0) < 0.02 and doff < 0.006
if not transform_ok:
    identity = True  # nothing better available: replay
LOG["identity_replay"] = identity

cz = math.cos(dth)
sz = math.sin(dth)
qz = [math.cos(dth / 2.0), 0.0, 0.0, math.sin(dth / 2.0)]

# video tip path
VTIP = []
for k in range(nv):
    R = quat_to_R(VQUAT[k])
    o = mv(R, off_v)
    VTIP.append([VP[k][0] + o[0], VP[k][1] + o[1], VP[k][2] + o[2]])
zs = [p[2] for p in VTIP]
zmin = min(zs)
LOG["video_tip_z"] = {"min": round(zmin, 4), "max": round(max(zs), 4)}
LOG["video_ee_z_min"] = round(min(p[2] for p in VP), 4)

# first moving frame, first contact frame, hover frame
k_move = 0
for k in range(nv):
    if max(abs(a - b) for a, b in zip(VQ[k], VQ[0])) > 0.002:
        k_move = max(0, k - 1)
        break
k_contact = None
for k in range(nv):
    if VTIP[k][2] < zmin + 0.004:
        k_contact = k
        break
k_hover = k_move
for k in range(k_contact, -1, -1):
    if VTIP[k][2] > zmin + 0.03:
        k_hover = k
        break
LOG["frames"] = {"k_move": k_move, "k_hover": k_hover, "k_contact": k_contact, "n": nv}
# compact log of the low (drawing) part of the video tip path
low_log = []
for k in range(nv):
    if k % 4 == 0:
        low_log.append([k, round(VT[k], 2), round(VTIP[k][0], 4), round(VTIP[k][1], 4), round(VTIP[k][2], 4)])
LOG["video_tip_path"] = low_log


def live_pose_for(k, dz):
    t = VTIP[k]
    dx = (t[0] - cv_[0]) * scale
    dy = (t[1] - cv_[1]) * scale
    tx = cl_[0] + cz * dx - sz * dy
    ty = cl_[1] + sz * dx + cz * dy
    tzz = t[2] - dz
    qn = qmul(qz, VQUAT[k])
    R = quat_to_R(qn)
    o = mv(R, off_l)
    return [tx - o[0], ty - o[1], tzz - o[2]], qn


ik_fail = [0]
last_q = [None]


def joints_for(k, dz):
    if identity and dz == 0.0:
        return VQ[k]
    pos, qn = live_pose_for(k, dz)
    seed = last_q[0]
    if seed is None:
        seed = VQ[k][:]
        seed[0] = seed[0] + dth
    ik = solve_ik(Pose(position=pos, quaternion_wxyz=qn, frame="robot_base"), seed_joints=seed)
    if ik.ok:
        qq = fl(ik.joint_positions)
        last_q[0] = qq
        return qq
    ik_fail[0] += 1
    return last_q[0] if last_q[0] is not None else seed


# sanity: IK of the current EE pose reproduces current joints?
try:
    ik0 = solve_ik(Pose(position=fl(ee_live0.position), quaternion_wxyz=fl(ee_live0.quaternion_wxyz), frame="robot_base"), seed_joints=q_live0)
    LOG["ik_self_check"] = {"ok": bool(ik0.ok), "maxdiff": round(max(abs(a - b) for a, b in zip(fl(ik0.joint_positions), q_live0)), 4) if ik0.ok else None}
except Exception as e:
    LOG["ik_self_check"] = str(e)

ST = {"steps": 0, "terminal": False, "prev_t": 0.0, "prev_q": None, "speed": 1.0}
red0 = red_count(live)
LOG["red_baseline"] = red0


def do_step(q):
    r = step(RobotAction(arms={"primary": ArmCommand(mode="joint_position", target=list(q), gripper_position=grip0, embodiment=emb)}))
    ST["steps"] += 1
    if is_terminal(r):
        ST["terminal"] = True
    return r


# ---------- approach: move to the hover frame
q_h = joints_for(k_hover, -0.012 if doff > 0.006 else 0.0)
mres = move_to_joints(list(q_h), tolerance=0.01, max_steps=150)
try:
    ST["steps"] += int(mres.steps_executed)
except Exception:
    pass
if is_terminal(mres):
    ST["terminal"] = True
LOG["approach"] = {"ok": bool(mres.ok), "status": str(mres.status)}

# ---------- replay from hover to contact
# with an estimated (changed) tool, start 1.2 cm high and find contact by descending
dz_start = 0.0
if doff > 0.006:
    dz_start = -0.012
dz = dz_start
LOG["dz_start"] = dz_start
k = k_hover + 1
ST["prev_t"] = VT[k_hover]
ST["prev_q"] = q_h


def replay_frame(kk, dzv):
    qt = joints_for(min(kk + 1, nv - 1), dzv)  # one-frame lead to offset controller lag
    nt = max(1, int(round((VT[kk] - ST["prev_t"]) * CTRL_HZ / ST["speed"])))
    pq = ST["prev_q"]
    for j in range(nt):
        a = (j + 1.0) / nt
        qi = [p + a * (c - p) for p, c in zip(pq, qt)]
        do_step(qi)
        if ST["terminal"]:
            break
    ST["prev_t"] = VT[kk]
    ST["prev_q"] = qt


while k <= k_contact and not ST["terminal"]:
    replay_frame(k, dz)
    k += 1

# settle & check contact (red target)
contact_log = []
if not ST["terminal"]:
    for j in range(3):
        do_step(ST["prev_q"])
    obs = get_observation()
    rc = red_count(obs)
    contact_log.append([0.0, rc])
    found = rc > red0 + 12
    step_dz = 0.003
    tries = 0
    while not found and tries < 16 and not ST["terminal"]:
        tries += 1
        dz = dz_start + step_dz * tries
        qd = joints_for(k_contact, dz)
        pq = ST["prev_q"]
        for j in range(4):
            a = (j + 1.0) / 4
            do_step([p + a * (c - p) for p, c in zip(pq, qd)])
        ST["prev_q"] = qd
        obs = get_observation()
        rc = red_count(obs)
        contact_log.append([round(dz, 4), rc])
        found = rc > red0 + 12
    if not found:
        dz = 0.0
    elif dz != dz_start:
        dz = dz + 0.002
    LOG["contact_found"] = found
LOG["contact_log"] = contact_log
LOG["dz"] = dz

# ---------- replay the rest of the pattern
budget_left = None
if max_steps is not None:
    budget_left = max_steps - ST["steps"]
    need = int(round((VT[-1] - VT[k_contact]) * CTRL_HZ))
    LOG["ticks_needed"] = need
    LOG["budget_left"] = budget_left
    if need > budget_left - 30 and budget_left > 60:
        ST["speed"] = float(need) / float(budget_left - 30)
    LOG["speed"] = ST["speed"]
red_trace = []
while k < nv and not ST["terminal"]:
    replay_frame(k, dz)
    if k % 40 == 0 and not ST["terminal"]:
        try:
            red_trace.append([k, red_count(get_observation())])
        except Exception:
            pass
    k += 1
# hold at the end
if not ST["terminal"]:
    for j in range(10):
        do_step(ST["prev_q"])
        if ST["terminal"]:
            break
LOG["red_trace"] = red_trace
LOG["steps_used"] = ST["steps"]
LOG["terminal"] = ST["terminal"]
LOG["ik_fail"] = ik_fail[0]
try:
    if not ST["terminal"]:
        st = get_robot_state()
        LOG["final_ee"] = fl(st.end_effector_poses["primary"].position)
        LOG["final_t"] = float(st.timestamp_s)
except Exception as e:
    LOG["final_read_error"] = str(e)
result = LOG
