# InsertPeg v09 -- v08 with the opening's size and centre measured (top rim + floor -> square opening of
# that side) instead of assuming opening = peg width and resting the peg on the floor.
# v08 was: v07 plus live hole localisation: when the peg is stopped on the box face, the wrist
# depth image is searched for the box opening (a flat patch deeper than the surrounding face, bounded by
# face/peg on both sides); the peg is moved so its cross-section lands inside the opening, then the
# slow raster runs around that new line. Falls back to v07 when no opening is seen.
# v07 was: v06 with the fast wiggle replaced by a slow raster under a gentle constant push
# (a tight hole needs dwell time: the peg must sit inside the clearance long enough to advance).
# v06 was: v05 with a compliant wiggle insertion (small lateral Lissajous + orientation
# wobble while gently pushing along the insertion direction) replacing the retract/probe spiral.
# v05 was: in-hand peg offset correction. At the insertion gate the wrist depth image is
# back-projected (camera pose + intrinsics) to locate the grasped peg's top face in the base frame,
# both live and in the conditioning video's frame at the same point of the demonstration. The
# difference of the peg-in-hand offsets (orthogonal to the demonstrated insertion direction) is
# added to every demonstrated EE target of the insertion (per-tick IK), so the live peg follows the
# video's peg line rather than the video's hand line. Fallback: fine 1 mm spiral around that line.
# v04 was: v03 with a radius-ordered spiral search over both lateral axes (+-7.5 mm) and
# compact wrist-depth diagnostics at the insertion gate (live and video). v03 was: v02 with a lateral-first, finer, gentler search that only stops when the peg
# reaches the demonstrated insertion depth. v02 was: v01 without time-stretch, plus: stall detection during the insertion segment
# and a small lateral search (offsets orthogonal to the demonstrated insertion direction) when the
# peg tip is blocked by the box face. Base behaviour: reproduce the conditioning video's demonstrated arm motion.
#
# Goal (read at runtime): "Watch the video carefully, then grasp the same peg at the same end
# and insert it into the same side of the box as in the video."
# The episode's conditioning video (robomme_plus.get_conditioning_video()) carries the recorded
# robot state for every frame.  This program re-executes that demonstrated joint trajectory
# tick by tick (gripper schedule derived from the recorded gripper readings), after checking
# that the live scene matches the video's first frame.  Nothing is hardcoded: joints, timing
# and gripper events all come from the current episode's video.
import math

report = {"version": "v09"}
ctx = get_task_context()
report["goal"] = ctx.language
meta = robomme_plus.get_task_metadata()
try:
    max_steps = int(meta["max_steps"])
except Exception:
    max_steps = 1000
report["max_steps"] = max_steps
try:
    hz = float(meta["control_hz"])
except Exception:
    hz = 20.0
try:
    report["video_sampling"] = str(meta["conditioning_video_sampling"])[:300]
except Exception:
    report["video_sampling"] = None
try:
    cm = robomme_plus.get_controller_metadata()
    report["controller_keys"] = sorted([str(k) for k in cm])[:40]
except Exception as exc:
    report["controller_keys"] = "err " + str(exc)[:100]

state0 = get_robot_state()
emb = state0.embodiment
q_live0 = [float(v) for v in state0.joint_positions["primary"]]

video = robomme_plus.get_conditioning_video()
report["video_len"] = len(video)
if len(video) < 2:
    raise ValueError("no conditioning video")

vq = []
vg = []
vt = []
vee = []
vquat = []
for fr in video:
    rs = fr.robot_state
    try:
        vee.append([float(v) for v in rs.end_effector_poses["primary"].position])
        vquat.append([float(v) for v in rs.end_effector_poses["primary"].quaternion_wxyz])
    except Exception:
        vee.append(None)
        vquat.append(None)
    vq.append([float(v) for v in rs.joint_positions["primary"]])
    vg.append(float(rs.gripper_positions["primary"]))
    vt.append(float(fr.timestamp_s))
report["video_t"] = [round(vt[0], 3), round(vt[-1], 3)]

# ---- resample to one target per control tick (handles a uniformly downsampled video) ----
dt = 1.0 / hz
total_ticks = int(round((vt[-1] - vt[0]) / dt))
tq = []
tg = []
tee = []
tj = []
j = 0
for k in range(total_ticks + 1):
    t = vt[0] + k * dt
    while j < len(vt) - 2 and vt[j + 1] < t - 1e-9:
        j += 1
    span = vt[j + 1] - vt[j]
    a = 0.0 if span <= 0 else min(1.0, max(0.0, (t - vt[j]) / span))
    tq.append([vq[j][i] + a * (vq[j + 1][i] - vq[j][i]) for i in range(7)])
    tg.append(vg[j] + a * (vg[j + 1] - vg[j]))
    tj.append(j if a < 0.5 else j + 1)
    if vee[j] is not None and vee[j + 1] is not None:
        tee.append([vee[j][i] + a * (vee[j + 1][i] - vee[j][i]) for i in range(3)])
    else:
        tee.append(None)
report["ticks_in_video"] = total_ticks

# ---- gripper schedule from recorded readings (0 closed, 1 open) ----
g_open = max(tg[0], 0.5)
lead = 2
cmd = []
closed = False
g_min = 1.0
close_idx = None
open_idx = None
for k in range(len(tg)):
    g = tg[k]
    if not closed:
        if g < g_open - 0.04:
            closed = True
            g_min = g
            if close_idx is None:
                close_idx = k
    else:
        g_min = min(g_min, g)
        if g > g_min + 0.08:
            closed = False
            g_open = max(g, 0.5)
            if open_idx is None:
                open_idx = k
    cmd.append(0.0 if closed else 1.0)
# command leads the measured reading by a couple of ticks
gcmd = []
for k in range(len(cmd)):
    gcmd.append(cmd[min(len(cmd) - 1, k + lead)])
report["close_idx"] = close_idx
report["open_idx"] = open_idx
report["g_min_video"] = round(min(tg), 3)
report["g_end_video"] = round(tg[-1], 3)
try:
    pe = video[-1].robot_state.end_effector_poses["primary"]
    report["video_end_ee"] = [round(float(v), 4) for v in pe.position]
    if close_idx is not None:
        ci = min(len(video) - 1, close_idx)
        pg = video[ci].robot_state.end_effector_poses["primary"]
        report["video_grasp_ee"] = [round(float(v), 4) for v in pg.position]
except Exception as exc:
    report["ee_err"] = str(exc)[:120]

# ---- scene consistency: live front depth vs video frame 0 (diagnostic only) ----
try:
    live = get_observation()
    d_live = live.cameras["front"].depth_m
    d_vid = video[0].cameras["front"].depth_m
    h = len(d_live)
    w = len(d_live[0])
    n = 0
    big = 0
    tot = 0.0
    for r in range(0, h, 8):
        for c in range(0, w, 8):
            a = float(d_live[r][c])
            b = float(d_vid[r][c])
            if a > 0 and b > 0 and a < 10 and b < 10:
                dd = abs(a - b)
                n += 1
                tot += dd
                if dd > 0.01:
                    big += 1
    report["scene_depth_check"] = {"n": n, "mean_abs_m": round(tot / max(1, n), 5), "frac_gt_1cm": round(big / max(1, n), 4)}
except Exception as exc:
    report["scene_depth_check"] = "err " + str(exc)[:120]

# ---- insertion segment: the final approach within INSERT_RADIUS of the video's final EE pose ----
INSERT_RADIUS = 0.10   # m; generic, measured from the video's own EE track at runtime
STRETCH_MAX = 1.0
k_star = None
ee_end = vee[-1]
if ee_end is not None:
    for kk in range(len(vee) - 1, -1, -1):
        e = vee[kk]
        if e is None:
            continue
        d = math.sqrt(sum((e[i] - ee_end[i]) ** 2 for i in range(3)))
        if d > INSERT_RADIUS:
            k_star = kk
            break
gate_tick = None
if k_star is not None and vt[-1] > vt[0]:
    gate_tick = int(round((vt[k_star] - vt[0]) / dt))
    if close_idx is not None and gate_tick <= close_idx + 5:
        gate_tick = None
report["gate_tick"] = gate_tick
u_dir = None
if gate_tick is not None and tee[gate_tick] is not None and ee_end is not None:
    dv = [ee_end[i] - tee[gate_tick][i] for i in range(3)]
    nrm = math.sqrt(sum(v * v for v in dv))
    if nrm > 1e-6:
        u_dir = [v / nrm for v in dv]
report["u_dir"] = None if u_dir is None else [round(v, 3) for v in u_dir]
STALL_LAG = 0.008   # m behind the video's progress along the insertion direction -> blocked
blocked_at = None
seg_len = 0 if gate_tick is None else (len(tq) - 1 - gate_tick)
stretch = 1.0
if seg_len > 0:
    spare = max_steps - 120 - len(tq)
    stretch = max(1.0, min(STRETCH_MAX, 1.0 + spare / float(seg_len)))
report["stretch"] = round(stretch, 2)
try:
    report["video_ee_gate"] = [round(v, 4) for v in vee[k_star]]
    report["video_ee_track"] = {str(kk): [round(v, 4) for v in vee[kk]] for kk in range(k_star, len(vee), 10)}
except Exception:
    pass

err0 = max(abs(q_live0[i] - tq[0][i]) for i in range(7))
report["start_joint_err"] = round(err0, 4)
ticks = 0
done = False
term_info = None
if err0 > 0.02:
    mv = move_to_joints(tq[0], tolerance=0.01, max_steps=120)
    try:
        ticks += int(mv.steps_executed)
    except Exception:
        pass
    report["pre_align_ok"] = bool(mv.ok)


def depth_grid(d):
    """Every 4th pixel of a depth image, in millimetres (compact diagnostic)."""
    out = []
    for r in range(0, len(d), 4):
        row = d[r]
        out.append([int(round(1000.0 * float(row[c]))) for c in range(0, len(row), 4)])
    return out


def send(target, g):
    sr = step(RobotAction(arms={"primary": ArmCommand("joint_position", target, gripper_position=g, embodiment=emb)}))
    ok = True
    try:
        ok = bool(sr.ok)
    except Exception:
        ok = True
    term = False
    try:
        term = bool(sr.terminated) or bool(sr.truncated)
    except Exception:
        term = False
    q = None
    try:
        q = [float(v) for v in sr.observation.robot_state.joint_positions["primary"]]
    except Exception:
        q = None
    return ok, term, q, sr


def qmat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


# wrist image band holding the grasped peg: below the far field, above the finger pads
BAND = (0.30, 0.74)
PEG_REL = 0.006    # m: pixels within this of the row's nearest depth belong to the peg's top face
PEG_SEP = 0.020    # m: a row counts only if its nearest surface stands this far above the row median


def peg_centroid(cam):
    """Back-project the grasped peg's top face (nearest surface per row of the wrist band) to the base frame."""
    d = cam.depth_m
    K = cam.intrinsics
    fx = float(K[0][0])
    fy = float(K[1][1])
    cx = float(K[0][2])
    cy = float(K[1][2])
    R = qmat([float(v) for v in cam.camera_pose.quaternion_wxyz])
    p = [float(v) for v in cam.camera_pose.position]
    h = len(d)
    sx = 0.0
    sy = 0.0
    sz = 0.0
    n = 0
    for r in range(int(BAND[0] * h), int(BAND[1] * h) + 1, 2):
        row = [float(v) for v in d[r]]
        good = [v for v in row if v > 0.0 and v < 5.0]
        if len(good) < 10:
            continue
        m = min(good)
        srt = sorted(good)
        med = srt[len(srt) // 2]
        if med - m < PEG_SEP:
            continue
        for c in range(0, len(row), 2):
            z = row[c]
            if z > 0.0 and z <= m + PEG_REL:
                X = (c - cx) * z / fx
                Y = (r - cy) * z / fy
                sx += p[0] + R[0][0] * X + R[0][1] * Y + R[0][2] * z
                sy += p[1] + R[1][0] * X + R[1][1] * Y + R[1][2] * z
                sz += p[2] + R[2][0] * X + R[2][1] * Y + R[2][2] * z
                n += 1
    if n == 0:
        return 0, None
    return n, [sx / n, sy / n, sz / n]


HOLE_SEP = 0.008     # m deeper than the row's face median -> looking into an opening
TABLE_GAP = 0.015    # m: pixels within this of the row's farthest depth are table/background
HOLE_SIDE = 24       # px: an opening must have face or peg (shallower) on both sides within this
MIN_HOLE = 12        # pixels (at stride 2) for a valid opening
MAX_SHIFT = 0.015    # m: larger computed shifts are treated as a detection failure


hole_rim = []


def peg_and_hole(cam, step):
    d = cam.depth_m
    K = cam.intrinsics
    fx = float(K[0][0])
    fy = float(K[1][1])
    cx = float(K[0][2])
    cy = float(K[1][2])
    R = qmat([float(v) for v in cam.camera_pose.quaternion_wxyz])
    p = [float(v) for v in cam.camera_pose.position]
    h = len(d)
    w = len(d[0])

    def bp(r, c, z):
        X = (c - cx) * z / fx
        Y = (r - cy) * z / fy
        return [p[0] + R[0][0] * X + R[0][1] * Y + R[0][2] * z,
                p[1] + R[1][0] * X + R[1][1] * Y + R[1][2] * z,
                p[2] + R[2][0] * X + R[2][1] * Y + R[2][2] * z]
    peg_pts = []
    peg_depths = []
    for r in range(int(BAND[0] * h), int(BAND[1] * h) + 1, step):
        row = [float(v) for v in d[r]]
        good = [v for v in row if v > 0.0 and v < 5.0]
        if len(good) < 4:
            continue
        m = min(good)
        srt = sorted(good)
        if srt[len(srt) // 2] - m < PEG_SEP:
            continue
        peg_depths.append(m)
        for c in range(0, w, step):
            if row[c] > 0.0 and row[c] <= m + PEG_REL:
                peg_pts.append(bp(r, c, row[c]))
    if len(peg_pts) < MIN_PIX // 4:
        return None, None
    peg_d = sorted(peg_depths)[len(peg_depths) // 2]
    hole_pts = []
    top_row = {}
    for r in range(0, int(BAND[1] * h) + 1, step):
        row = [float(v) for v in d[r]]
        far = max(row)
        cand = [v for v in row if v > peg_d + 0.003 and v < far - TABLE_GAP]
        if len(cand) < 4:
            continue
        srt = sorted(cand)
        med = srt[len(srt) // 2]
        for c in range(0, w, step):
            z = row[c]
            if z > med + HOLE_SEP and z < far - TABLE_GAP:
                okl = False
                okr = False
                for k2 in range(1, HOLE_SIDE + 1):
                    if c - k2 >= 0 and not okl:
                        zz = float(row[c - k2])
                        if zz < z - HOLE_SEP:
                            okl = True
                        elif zz > far - TABLE_GAP:
                            break
                for k2 in range(1, HOLE_SIDE + 1):
                    if c + k2 < w and not okr:
                        zz = float(row[c + k2])
                        if zz < z - HOLE_SEP:
                            okr = True
                        elif zz > far - TABLE_GAP:
                            break
                if okl and okr:
                    hole_pts.append(bp(r, c, z))
                    if c not in top_row:
                        top_row[c] = [r, z]
    # top rim of the opening: first shallower face pixel above the topmost opening pixel of each column
    rim_pts = []
    for c in top_row:
        r0 = top_row[c][0]
        z0 = top_row[c][1]
        for k2 in range(1, 5):
            rr = r0 - k2 * step
            if rr < 0:
                break
            zz = float(d[rr][c])
            if zz < z0 - HOLE_SEP:
                rim_pts.append(bp(rr, c, zz))
                break
    hole_rim.append(rim_pts)
    return peg_pts, hole_pts


def ee_now():
    return [float(v) for v in get_robot_state().end_effector_poses["primary"].position]


def ik_joints(pos, quat, seed):
    try:
        ik = solve_ik(Pose(pos, quat, frame), seed_joints=seed)
        if not ik.ok:
            return None
        return [float(v) for v in ik.joint_positions]
    except Exception:
        return None


frame = get_robot_state().base_frame
cur_g = 1.0


def cart_move(pos, quat, n_ticks, settle):
    """IK to pos/quat, then joint-interpolate over n_ticks and hold settle ticks. Returns (ok, used, term, goal)."""
    q_now = [float(v) for v in get_robot_state().joint_positions["primary"]]
    q_goal = ik_joints(pos, quat, q_now)
    if q_goal is None:
        return False, 0, False, None
    used = 0
    for t in range(1, n_ticks + settle + 1):
        a = min(1.0, t / float(n_ticks))
        tgt = [q_now[i] + a * (q_goal[i] - q_now[i]) for i in range(7)]
        ok, term, q, sr = send(tgt, cur_g)
        used += 1
        if term:
            return True, used, True, q_goal
    return True, used, False, q_goal


# ---- scene identity gate: replaying the video is only meaningful if the live layout is the video's ----
SCENE_MISMATCH = 0.20
try:
    if report["scene_depth_check"]["frac_gt_1cm"] > SCENE_MISMATCH:
        report["abort"] = "live scene differs from the conditioning video's first frame"
except Exception:
    pass

MAX_CORR = 0.015    # m: larger estimated in-hand offsets are treated as a measurement failure
MIN_PIX = 50
RESID_OK = 0.0007   # m: re-measure once and refine if the residual peg offset is larger than this
corr = [0.0, 0.0, 0.0]
dq = [0.0] * 7
prev_sol = None
ik_fallbacks = 0
last_target = None
live_track = {}
track_err_max = 0.0
track_err_at = {}
rejected = 0
k = 1
while k < len(tq) and "abort" not in report:
    if ticks >= max_steps - 2:
        report["stopped_for_budget_at"] = k
        break
    # converge before the grasp closes so the jaw meets the peg where the video did
    if close_idx is not None and k == max(1, close_idx - lead):
        st = get_robot_state()
        q = [float(v) for v in st.joint_positions["primary"]]
        hold = 0
        while max(abs(q[i] - tq[k][i]) for i in range(7)) > 0.004 and hold < 25 and not done:
            ok, term, qq, sr = send(tq[k], 1.0)
            ticks += 1
            hold += 1
            if not ok:
                rejected += 1
            if term:
                done = True
                term_info = "during pre-grasp hold"
            if qq is not None:
                q = qq
        report["pregrasp_hold_ticks"] = hold
        report["pregrasp_err"] = round(max(abs(q[i] - tq[k][i]) for i in range(7)), 4)
        # appearance cross-check of the object between the jaws: live wrist depth vs the video's
        try:
            dl = get_observation().cameras["wrist"].depth_m
            dv = video[min(len(video) - 1, tj[k])].cameras["wrist"].depth_m
            nn = 0
            bad = 0
            for r in range(0, len(dl), 8):
                for c in range(0, len(dl[0]), 8):
                    nn += 1
                    if abs(float(dl[r][c]) - float(dv[r][c])) > 0.01:
                        bad += 1
            report["grasp_view_mismatch"] = round(bad / max(1, nn), 3)
            if bad / max(1, nn) > SCENE_MISMATCH:
                report["abort"] = "object between the jaws differs from the video's at grasp"
                break
        except Exception as exc:
            report["grasp_view_mismatch"] = "err " + str(exc)[:80]
        if done:
            break
    if gate_tick is not None and k == gate_tick and u_dir is not None:
        st = get_robot_state()
        q = [float(v) for v in st.joint_positions["primary"]]
        hold = 0
        while max(abs(q[i] - tq[k][i]) for i in range(7)) > 0.002 and hold < 40 and not done:
            ok, term, qq, sr = send(tq[k], gcmd[k])
            ticks += 1
            hold += 1
            if term:
                done = True
                term_info = "during insertion gate"
            if qq is not None:
                q = qq
        report["gate_hold_ticks"] = hold
        if done:
            break
        cur_g = gcmd[k]
        # in-hand peg offset: (peg - hand) live minus (peg - hand) in the video at the same point
        try:
            nv, pv = peg_centroid(video[k_star].cameras["wrist"])
            nl, pl_ = peg_centroid(get_observation().cameras["wrist"])
            el = ee_now()
            ev = vee[k_star]
            report["peg_pix"] = [nv, nl]
            if nv >= MIN_PIX and nl >= MIN_PIX:
                off = [(pl_[i] - el[i]) - (pv[i] - ev[i]) for i in range(3)]
                along = sum(off[i] * u_dir[i] for i in range(3))
                c0 = [-(off[i] - along * u_dir[i]) for i in range(3)]
                report["inhand_offset_mm"] = [round(1000 * v, 2) for v in off]
                if math.sqrt(sum(v * v for v in c0)) <= MAX_CORR:
                    corr = c0
        except Exception as exc:
            report["peg_est_err"] = str(exc)[:120]
        report["corr_mm"] = [round(1000 * v, 2) for v in corr]
        qg = vquat[tj[k]]
        if max(abs(v) for v in corr) > 0.0 and qg is not None:
            okm, used, term, q_goal = cart_move([tee[k][i] + corr[i] for i in range(3)], qg, 8, 4)
            ticks += used
            if term:
                done = True
                term_info = "gate correction"
                break
            if q_goal is not None:
                # verify: re-measure the peg; refine once if it is still off the video's peg line
                try:
                    nl2, pl2 = peg_centroid(get_observation().cameras["wrist"])
                    if nl2 >= MIN_PIX:
                        res = [pl2[i] - (pv[i] + tee[k][i] - ev[i]) for i in range(3)]
                        al2 = sum(res[i] * u_dir[i] for i in range(3))
                        rp = [res[i] - al2 * u_dir[i] for i in range(3)]
                        report["resid_mm"] = [round(1000 * v, 2) for v in rp]
                        if math.sqrt(sum(v * v for v in rp)) > RESID_OK:
                            c2 = [corr[i] - rp[i] for i in range(3)]
                            if math.sqrt(sum(v * v for v in c2)) <= MAX_CORR:
                                corr = c2
                                okm, used, term, q_goal2 = cart_move([tee[k][i] + corr[i] for i in range(3)], qg, 4, 4)
                                ticks += used
                                if term:
                                    done = True
                                    term_info = "gate refinement"
                                    break
                                if q_goal2 is not None:
                                    q_goal = q_goal2
                                report["corr2_mm"] = [round(1000 * v, 2) for v in corr]
                except Exception as exc:
                    report["resid_err"] = str(exc)[:120]
                dq = [q_goal[i] - tq[k][i] for i in range(7)]
                prev_sol = q_goal
            else:
                report["corr_ik_failed"] = True
                corr = [0.0, 0.0, 0.0]
        try:
            report["diag_gate_live"] = depth_grid(get_observation().cameras["wrist"].depth_m)
            report["live_ee_gate"] = [round(v, 4) for v in ee_now()]
        except Exception:
            pass
    target = tq[k]
    if gate_tick is not None and k > gate_tick and prev_sol is not None:
        guess = [tq[k][i] + dq[i] for i in range(7)]
        qk = vquat[tj[k]]
        sol = None
        if tee[k] is not None and qk is not None:
            sol = ik_joints([tee[k][i] + corr[i] for i in range(3)], qk, prev_sol)
        if sol is None or max(abs(sol[i] - guess[i]) for i in range(7)) > 0.1:
            sol = guess
            ik_fallbacks += 1
        prev_sol = sol
        target = sol
    last_target = target
    ok, term, q, sr = send(target, gcmd[k])
    ticks += 1
    if gate_tick is not None and k > gate_tick and u_dir is not None and tee[k] is not None:
        try:
            pl = [float(v) for v in sr.observation.robot_state.end_effector_poses["primary"].position]
        except Exception:
            pl = None
        if pl is not None:
            if k % 10 == 0:
                live_track[str(k)] = [round(v, 4) for v in pl]
            lag_along = sum((tee[k][i] - pl[i]) * u_dir[i] for i in range(3))
            if lag_along > STALL_LAG:
                blocked_at = k
                report["blocked_live_ee"] = [round(v, 4) for v in pl]
                report["blocked_video_ee"] = [round(v, 4) for v in tee[k]]
                k += 1
                break
    if not ok:
        rejected += 1
        if rejected >= 3:
            try:
                report["reject_error"] = str(sr.error)[:200]
            except Exception:
                report["reject_error"] = "rejected"
            break
    if q is not None:
        e = max(abs(q[i] - target[i]) for i in range(7))
        if e > track_err_max:
            track_err_max = e
        if k % 50 == 0:
            track_err_at[str(k)] = round(e, 4)
    if term:
        done = True
        term_info = "replay tick " + str(k)
        break
    k += 1

report["replayed_to"] = k
report["ik_fallbacks"] = ik_fallbacks
report["track_err_max"] = round(track_err_max, 4)
report["track_err_at"] = track_err_at
report["rejected_steps"] = rejected
report["live_ee_track"] = live_track

# ---- hold the final (corrected) demonstrated pose so the insertion can settle ----
final_g = gcmd[-1]
cur_g = final_g
if not done and blocked_at is None and "abort" not in report and last_target is not None:
    hold = 0
    while hold < 20 and ticks < max_steps - 1:
        ok, term, q, sr = send(last_target, final_g)
        ticks += 1
        hold += 1
        if term:
            done = True
            term_info = "final hold tick " + str(hold)
            break
    report["final_hold_ticks"] = hold
    if not done and u_dir is not None:
        try:
            pl = ee_now()
            if sum((ee_end[i] - pl[i]) * u_dir[i] for i in range(3)) > 0.004:
                blocked_at = len(tq) - 1
                report["blocked_live_ee"] = [round(v, 4) for v in pl]
        except Exception:
            pass

# ---- still blocked: slow raster under a gentle push around the corrected peg line ----
RASTER_HALF = 0.0016  # m half-width of the raster square around the corrected peg line
RASTER_PITCH = 0.0004 # m between raster rows (below the clearance a probe grid could miss)
RASTER_SPEED = 0.00008  # m per tick along the raster path (slow enough to drop into a tight hole)
PUSH_AHEAD = 0.002    # m commanded ahead of the current depth (bounded contact force)
ENTRY_GAIN = 0.0015   # m of progress beyond the contact depth -> entered: stop sweeping, push straight
DEPTH_OK = 0.006    # m: within this of the demonstrated final depth counts as inserted
search_log = []
inserted = False


def qmul(a, b):
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]


def qaxis(ax, ang):
    s = math.sin(ang / 2.0)
    return [math.cos(ang / 2.0), ax[0] * s, ax[1] * s, ax[2] * s]


if not done and blocked_at is not None and u_dir is not None:
    quat = vquat[-1]
    if quat is None:
        quat = [float(v) for v in get_robot_state().end_effector_poses["primary"].quaternion_wxyz]
    p_block = ee_now()
    try:
        report["diag_block_live"] = depth_grid(get_observation().cameras["wrist"].depth_m)
    except Exception as exc:
        report["diag_block_live"] = "err " + str(exc)[:100]
    e1 = [-u_dir[1], u_dir[0], 0.0]
    n1 = math.sqrt(e1[0] ** 2 + e1[1] ** 2)
    if n1 < 1e-6:
        e1 = [1.0, 0.0, 0.0]
        n1 = 1.0
    e1 = [v / n1 for v in e1]
    e2 = [u_dir[1] * e1[2] - u_dir[2] * e1[1], u_dir[2] * e1[0] - u_dir[0] * e1[2], u_dir[0] * e1[1] - u_dir[1] * e1[0]]
    if e2[2] < 0:
        e2 = [-v for v in e2]
    ref = [ee_end[i] + corr[i] for i in range(3)]
    d_block = sum((p_block[i] - ref[i]) * u_dir[i] for i in range(3))
    report["d_block"] = round(d_block, 4)
    # ---- look for the box opening next to the stopped peg and re-aim the peg line at it ----
    okm, used, term, g_ = cart_move([ref[i] + (d_block - 0.003) * u_dir[i] for i in range(3)], quat, 5, 4)
    ticks += used
    if term:
        done = True
        term_info = "hole look retract"
    hole_info = {}
    try:
        peg_pts, hole_pts = peg_and_hole(get_observation().cameras["wrist"], 2)
        hole_info["n_peg"] = 0 if peg_pts is None else len(peg_pts)
        hole_info["n_hole"] = 0 if hole_pts is None else len(hole_pts)
        if peg_pts is not None and hole_pts is not None and len(hole_pts) >= MIN_HOLE:
            s1p = sorted([sum((q[i] - ref[i]) * e1[i] for i in range(3)) for q in peg_pts])
            s2p = sorted([sum((q[i] - ref[i]) * e2[i] for i in range(3)) for q in peg_pts])
            s1h = sorted([sum((q[i] - ref[i]) * e1[i] for i in range(3)) for q in hole_pts])
            s2h = sorted([sum((q[i] - ref[i]) * e2[i] for i in range(3)) for q in hole_pts])
            lo = s1p[int(0.03 * len(s1p))]
            hi = s1p[int(0.97 * len(s1p))]
            wpeg = hi - lo
            pc1 = 0.5 * (lo + hi)
            ptop = s2p[len(s2p) // 2]
            hlo = s1h[int(0.05 * len(s1h))]
            hhi = s1h[int(0.95 * len(s1h))]
            hfloor = s2h[len(s2h) // 2]
            hside = wpeg + 0.002
            rim = hole_rim[-1] if len(hole_rim) > 0 else []
            hole_info["n_rim"] = len(rim)
            if len(rim) >= 3:
                s2r = sorted([sum((q[i] - ref[i]) * e2[i] for i in range(3)) for q in rim])
                rim2 = s2r[len(s2r) // 2]
                hole_info["hole_rim_mm"] = round(1000 * rim2, 2)
                if rim2 - hfloor > wpeg and rim2 - hfloor < wpeg + 0.010:
                    hside = rim2 - hfloor
                hc2 = 0.5 * (rim2 + hfloor)
            else:
                hc2 = hfloor + 0.5 * hside
            hside = max(hside, hhi - hlo)
            if hhi - hlo >= hside - 0.0005:
                hc1 = 0.5 * (hlo + hhi)
            elif 0.5 * (hlo + hhi) < pc1:
                hc1 = hlo + 0.5 * hside
            else:
                hc1 = hhi - 0.5 * hside
            sh1 = hc1 - pc1
            sh2 = hc2 - (ptop - 0.5 * wpeg)   # square peg section: centre = top - width/2
            hole_info["hole_side_mm"] = round(1000 * hside, 2)
            hole_info["peg_w_mm"] = round(1000 * wpeg, 2)
            hole_info["peg_c1_mm"] = round(1000 * pc1, 2)
            hole_info["peg_top_mm"] = round(1000 * ptop, 2)
            hole_info["hole_e1_mm"] = [round(1000 * hlo, 2), round(1000 * hhi, 2)]
            hole_info["hole_floor_mm"] = round(1000 * hfloor, 2)
            hole_info["shift_mm"] = [round(1000 * sh1, 2), round(1000 * sh2, 2)]
            if abs(sh1) <= MAX_SHIFT and abs(sh2) <= MAX_SHIFT and not done:
                ref = [ref[i] + sh1 * e1[i] + sh2 * e2[i] for i in range(3)]
                okm, used, term, g_ = cart_move([ref[i] + (d_block - 0.010) * u_dir[i] - (sh1 * e1[i] + sh2 * e2[i]) for i in range(3)], quat, 6, 2)
                ticks += used
                okm, used, term, g_ = cart_move([ref[i] + (d_block - 0.010) * u_dir[i] for i in range(3)], quat, 12, 6)
                ticks += used
                if term:
                    done = True
                    term_info = "hole re-aim"
                hole_info["reaimed"] = True
    except Exception as exc:
        hole_info["err"] = str(exc)[:120]
    report["hole"] = hole_info
    q_seed = [float(v) for v in get_robot_state().joint_positions["primary"]]
    nhalf = int(round(RASTER_HALF / RASTER_PITCH))
    rows = [0.0]
    for i in range(1, nhalf + 1):
        rows.append(i * RASTER_PITCH)
        rows.append(-i * RASTER_PITCH)
    path = [[0.0, 0.0] for i in range(30)]
    cur1 = 0.0
    cur2 = 0.0
    for a2 in rows:
        if abs(cur1 + RASTER_HALF) < abs(cur1 - RASTER_HALF):
            ends = [-RASTER_HALF, RASTER_HALF]
        else:
            ends = [RASTER_HALF, -RASTER_HALF]
        for seg in ([cur1, cur2, ends[0], a2], [ends[0], a2, ends[1], a2]):
            ln = math.sqrt((seg[2] - seg[0]) ** 2 + (seg[3] - seg[1]) ** 2)
            ns = max(1, int(ln / RASTER_SPEED))
            for j in range(1, ns + 1):
                f = j / float(ns)
                path.append([seg[0] + f * (seg[2] - seg[0]), seg[1] + f * (seg[3] - seg[1])])
        cur1 = ends[1]
        cur2 = a2
    report["raster_len"] = len(path)
    best = d_block
    t = 0
    n_ik_fail = 0
    entered_at = None
    o_fix = None
    while not done and ticks < max_steps - 25:
        pl = ee_now()
        dn = sum((pl[i] - ref[i]) * u_dir[i] for i in range(3))
        if dn > best:
            best = dn
        if t % 20 == 0:
            search_log.append([t, round(dn, 4)])
        if dn > -DEPTH_OK:
            inserted = True
            break
        if entered_at is None and dn - d_block > ENTRY_GAIN:
            entered_at = t
            o_fix = [sum((pl[i] - ref[i]) * e1[i] for i in range(3)), sum((pl[i] - ref[i]) * e2[i] for i in range(3))]
            report["entered_at"] = [t, [round(1000 * v, 2) for v in o_fix]]
        if o_fix is not None:
            o1 = o_fix[0]
            o2 = o_fix[1]
        elif t < len(path):
            o1 = path[t][0]
            o2 = path[t][1]
        else:
            o1 = path[-1][0]
            o2 = path[-1][1]
        dcmd = min(0.004, dn + PUSH_AHEAD)
        pos = [ref[i] + dcmd * u_dir[i] + o1 * e1[i] + o2 * e2[i] for i in range(3)]
        sol = ik_joints(pos, quat, q_seed)
        if sol is None:
            n_ik_fail += 1
            sol = q_seed
        q_seed = sol
        ok, term, q, sr = send(sol, final_g)
        ticks += 1
        t += 1
        if term:
            done = True
            inserted = True
            term_info = "raster tick " + str(t)
            break
    report["wiggle_ticks"] = t
    report["wiggle_best_depth"] = round(best, 4)
    report["wiggle_ik_fail"] = n_ik_fail
    if inserted and not done:
        okm, used, term, g_ = cart_move([ref[i] + 0.002 * u_dir[i] for i in range(3)], quat, 10, 10)
        ticks += used
        if term:
            done = True
            term_info = "after wiggle full push"
report["search_log"] = search_log
report["search_inserted"] = inserted
report["blocked_at"] = blocked_at

report["ticks"] = ticks
report["terminated"] = done
report["term_info"] = term_info
try:
    st = get_robot_state()
    report["final_gripper"] = round(float(st.gripper_positions["primary"]), 3)
    report["final_ee"] = [round(float(v), 4) for v in st.end_effector_poses["primary"].position]
except Exception as exc:
    report["final_read"] = "err " + str(exc)[:80]

result = report
