# InsertPeg (robomme_plus task 3) -- v05 (v04 + Mode B peg yaw / held-end colour check, tail guard) (demo-only, written offline, 0 starts).
#
# Strategy: imitate the episode's OWN conditioning video. Every conditioning frame carries
# the recorded robot_state (joints, gripper, EE pose). The released training recordings show
# the execution starting from the same scene as conditioning frame 0, so:
#   Mode A (scene unchanged, checked from depth): replay the video's joint trajectory
#           tick-by-tick, closing the gripper where the video's measured gripper closes.
#   Mode B (scene changed): estimate the grasped peg's and the box's planar displacement
#           from video-frame-0 depth vs live depth (no hardcoded geometry), shift the
#           video's EE poses accordingly, IK them, and execute at the video's pacing.
# Everything episode-dependent (which peg, which end, which side, timing) comes from the
# current episode's conditioning video and live observation.
import math

GRIP_DROP = 0.04          # measured-gripper drop that marks the start of a close
DEPTH_DIFF_M = 0.01       # per-pixel depth change counted as "scene changed"
SCENE_SAME_FRAC = 0.002   # fraction of changed pixels tolerated for Mode A
PIX_STRIDE = 2            # pixel stride for pure-Python depth scans
SHIFT_SAME_M = 0.006      # displacement below which Mode A is still used
CONVERGE_RAD = 0.004      # joint convergence before closing the gripper
CONVERGE_MAX_TICKS = 10
FINAL_HOLD_TICKS = 20
KEY_STRIDE = 4            # Mode B IK keyframe stride (video frames)

report = {"version": "v05", "mode": None, "events": []}
context = get_task_context()
report["goal"] = context.language
state0 = get_robot_state()
EMB = state0.embodiment
BASE = state0.base_frame

meta = robomme_plus.get_task_metadata()
max_steps = None
try:
    max_steps = int(meta["max_steps"])
except Exception:
    max_steps = None
hz = 20.0
for key in ("control_hz", "control_frequency", "control_freq", "frequency_hz"):
    try:
        hz = float(meta[key])
        break
    except Exception:
        pass
report["max_steps"] = max_steps
report["hz"] = hz

video = robomme_plus.get_conditioning_video()
report["video_frames"] = len(video)

ABORT = None
if len(video) < 10:
    ABORT = "no usable conditioning video: cannot tell which peg/end/side"
VQ = []
VG = []
VT = []
VP = []
VQUAT = []
for frame in video:
    rs = frame.robot_state
    VQ.append([float(v) for v in rs.joint_positions["primary"]])
    VG.append(float(rs.gripper_positions["primary"]))
    VT.append(float(frame.timestamp_s))
    pose = rs.end_effector_poses["primary"]
    VP.append([float(pose.position[0]), float(pose.position[1]), float(pose.position[2])])
    VQUAT.append([float(v) for v in pose.quaternion_wxyz])
N = len(VQ)

# ---- gripper events from the video's measured gripper signal ----
open_ref = max(VG[0:min(5, N)]) if N > 0 else 1.0
close_idx = None
for k in range(N):
    if VG[k] < open_ref - GRIP_DROP:
        close_idx = k
        break
if close_idx is None:
    ABORT = "video gripper never closes: no grasp to imitate"
    close_idx = N
settle_idx = close_idx
for k in range(close_idx + 1, N - 2):
    if abs(VG[k] - VG[k - 1]) < 0.003 and abs(VG[k + 1] - VG[k]) < 0.003:
        settle_idx = k
        break
closed_level = VG[min(settle_idx, N - 1)] if N > 0 else 0.0
release_idx = N + 1
for k in range(settle_idx + 1, N):
    if VG[k] > closed_level + 0.05 and VG[k] > VG[k - 1]:
        release_idx = k
        break
report["close_idx"] = close_idx
report["settle_idx"] = settle_idx
report["release_idx"] = release_idx if release_idx <= N else None
report["video_closed_level"] = round(closed_level, 4)
report["grasp_point_video"] = [round(v, 4) for v in VP[close_idx]] if close_idx < N else None


def grip_for(j):
    if close_idx <= j and j < release_idx:
        return 0.0
    return 1.0


def ticks_between(j):
    # control ticks the video spent between frame j-1 and frame j
    dt = VT[j] - VT[j - 1]
    return max(1, int(round(dt * hz)))


# ---- terminal-aware single tick ----
RUN = {"ticks": 0, "terminal": False, "last_q": None, "step_errors": 0}


def flag(reply, name):
    try:
        if name == "terminated":
            return bool(reply.terminated)
        if name == "truncated":
            return bool(reply.truncated)
        if name == "done":
            return bool(reply.done)
        if name == "terminal":
            return bool(reply.terminal)
    except Exception:
        return False
    return False


def tick(q, grip):
    if RUN["terminal"]:
        return False
    try:
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=[float(v) for v in q],
            gripper_position=float(grip), embodiment=EMB)}))
    except Exception:
        # uncertain action: never retry; physics may have advanced (RUNTIME.md).
        # The provider span in trace/events.jsonl records the error text.
        RUN["terminal"] = True
        report["events"].append(["step_exception", RUN["ticks"]])
        return False
    RUN["ticks"] += 1
    if flag(reply, "terminated") or flag(reply, "truncated") or flag(reply, "done") or flag(reply, "terminal"):
        RUN["terminal"] = True
        report["events"].append(["terminal", RUN["ticks"]])
        return False
    ok = True
    try:
        ok = bool(reply.ok)
    except Exception:
        ok = True
    if not ok:
        RUN["step_errors"] += 1
        try:
            report["events"].append(["step_not_ok", RUN["ticks"], str(reply.error)])
        except Exception:
            report["events"].append(["step_not_ok", RUN["ticks"]])
        if RUN["step_errors"] >= 3:
            RUN["terminal"] = True
            return False
    if max_steps is not None and RUN["ticks"] >= max_steps:
        RUN["terminal"] = True
        return False
    return True


def measured_q():
    return [float(v) for v in get_robot_state().joint_positions["primary"]]


def max_err(a, b):
    return max(abs(a[i] - b[i]) for i in range(len(a)))


# ---- depth helpers (pure Python) ----
def as_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        return arr


def quat_to_rot(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def cam_vec(conv, xn, yn, d):
    if conv == "opencv":
        return (xn, yn, d)
    if conv == "opengl":
        return (xn, -yn, -d)
    return (d, -xn, -yn)          # "ros" x-forward camera link


def project(cam, rows, conv, row_lo, row_hi, stride, rgb_rows=None):
    K = as_rows(cam.intrinsics)
    fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
    R = quat_to_rot([float(v) for v in cam.camera_pose.quaternion_wxyz])
    t = [float(v) for v in cam.camera_pose.position]
    pts = []
    H = len(rows)
    W = len(rows[0])
    for r in range(max(0, row_lo), min(H, row_hi), stride):
        row = rows[r]
        for c in range(0, W, stride):
            d = float(row[c])
            if d != d or d <= 0.0 or d > 5.0:
                continue
            v = cam_vec(conv, (c - cx) / fx * d, (r - cy) / fy * d, d)
            X = R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2] + t[0]
            Y = R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2] + t[1]
            Z = R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2] + t[2]
            if rgb_rows is None:
                pts.append((X, Y, Z))
            else:
                px = rgb_rows[r][c]
                pts.append((X, Y, Z, float(px[0]), float(px[1]), float(px[2])))
    return pts


def quantile(vals, f):
    s = sorted(vals)
    return s[int((len(s) - 1) * f)]


def choose_convention(cam, rows):
    # The table fills the lower image; under the right convention it projects to a
    # horizontal plane below the camera.
    H = len(rows)
    best = None
    camz = float(cam.camera_pose.position[2])
    for conv in ("opencv", "opengl", "ros"):
        pts = project(cam, rows, conv, H // 2, H, PIX_STRIDE)
        if len(pts) < 50:
            continue
        zs = [p[2] for p in pts]
        spread = quantile(zs, 0.9) - quantile(zs, 0.1)
        if quantile(zs, 0.5) >= camz:
            continue
        if best is None or spread < best[1]:
            best = (conv, spread)
    return best


def centroid_near(pts, center, radius, zmin, zmax):
    sx = sy = sz = 0.0
    n = 0
    for p in pts:
        if p[2] < zmin or p[2] > zmax:
            continue
        if (p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2 <= radius * radius:
            sx += p[0]
            sy += p[1]
            sz += p[2]
            n += 1
    if n < 4:
        return None
    return [sx / n, sy / n, sz / n, n]


def mean_shift(pts, start, radius, zmin, zmax, iters):
    c = [start[0], start[1], start[2]]
    last = None
    for it in range(iters):
        got = centroid_near(pts, c, radius, zmin, zmax)
        if got is None:
            return last
        c = [got[0], got[1], got[2]]
        last = got
    return last


CELL_M = 0.008              # Mode B grid cell for connected-cluster extraction
MODEB_STRIDE = 1            # full-resolution projection for Mode B estimates


def grid_of(pts, zmin, zmax):
    grid = {}
    for p in pts:
        if p[2] < zmin or p[2] > zmax:
            continue
        key = (int(math.floor(p[0] / CELL_M)), int(math.floor(p[1] / CELL_M)))
        if key in grid:
            grid[key].append(p)
        else:
            grid[key] = [p]
    return grid


def cluster_from(grid, seed, max_seed_dist, exclude):
    """Connected cluster (8-neighbour cells) nearest to `seed`; returns
    [cx, cy, cz, n_points, yaw_of_major_axis] or None. `exclude` = set of cells to skip."""
    best = None
    best_d = None
    for key in grid:
        if key in exclude:
            continue
        cxk = (key[0] + 0.5) * CELL_M
        cyk = (key[1] + 0.5) * CELL_M
        d = math.sqrt((cxk - seed[0]) ** 2 + (cyk - seed[1]) ** 2)
        if d <= max_seed_dist and (best_d is None or d < best_d):
            best_d = d
            best = key
    if best is None:
        return None
    seen = set([best])
    todo = [best]
    while len(todo) > 0:
        key = todo.pop()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nb = (key[0] + dx, key[1] + dy)
                if nb in grid and nb not in seen and nb not in exclude:
                    seen.add(nb)
                    todo.append(nb)
    pts = []
    for key in seen:
        pts.extend(grid[key])
    n = len(pts)
    if n < 6:
        return None
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    mz = sum(p[2] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts) / n
    syy = sum((p[1] - my) ** 2 for p in pts) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts) / n
    yaw = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    tr = sxx + syy
    disc = math.sqrt(max(0.0, (sxx - syy) ** 2 / 4.0 + sxy * sxy))
    lam1 = tr / 2.0 + disc
    lam2 = max(1e-12, tr / 2.0 - disc)
    return [mx, my, mz, n, yaw, seen, pts, lam1 / lam2]


YAW_MIN_RAD = math.radians(5.0)   # smaller peg rotations are treated as PCA noise
ANISO_MIN = 3.0                   # eigenvalue ratio below which the peg axis is unreliable
CHROMA_MARGIN = 0.02              # colour evidence needed to prefer one end over the other


def end_chroma(cluster, direction):
    """Mean chromaticity (r/s, g/s) of cluster points more than 1 cm from the centre
    toward `direction` (unit xy). None without colour or with < 3 points."""
    sr = 0.0
    sg = 0.0
    n = 0
    for p in cluster[6]:
        if len(p) < 6:
            return None
        if (p[0] - cluster[0]) * direction[0] + (p[1] - cluster[1]) * direction[1] > 0.01:
            tot = p[3] + p[4] + p[5] + 1.0
            sr += p[3] / tot
            sg += p[4] / tot
            n += 1
    if n < 3:
        return None
    return [sr / n, sg / n]


def rot2(v, a):
    return [math.cos(a) * v[0] - math.sin(a) * v[1], math.sin(a) * v[0] + math.cos(a) * v[1]]


def quat_mul(a, b):
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]


def peg_rotation(vpeg, lpeg, grasp_point):
    """Planar rotation (rad) taking the video peg onto the live peg, with the 180-degree
    ambiguity of the major axis resolved by the colour of the HELD end."""
    gvec = [grasp_point[0] - vpeg[0], grasp_point[1] - vpeg[1]]
    axv = [math.cos(vpeg[4]), math.sin(vpeg[4])]
    sgn = 1.0 if gvec[0] * axv[0] + gvec[1] * axv[1] >= 0 else -1.0
    held_v = [sgn * axv[0], sgn * axv[1]]
    if vpeg[7] >= ANISO_MIN and lpeg[7] >= ANISO_MIN:
        d0 = lpeg[4] - vpeg[4]
        while d0 > math.pi / 2:
            d0 -= math.pi
        while d0 <= -math.pi / 2:
            d0 += math.pi
    else:
        d0 = 0.0
    if abs(d0) < YAW_MIN_RAD:
        d0 = 0.0
    alt = d0 - math.pi if d0 > 0 else d0 + math.pi
    cv = end_chroma(vpeg, held_v)
    if cv is None:
        return d0, "no_colour"
    scores = []
    for cand in (d0, alt):
        cl = end_chroma(lpeg, rot2(held_v, cand))
        if cl is None:
            scores.append(None)
        else:
            scores.append(math.sqrt((cl[0] - cv[0]) ** 2 + (cl[1] - cv[1]) ** 2))
    if scores[0] is None or scores[1] is None:
        return d0, "one_end_unseen"
    if scores[1] + CHROMA_MARGIN < scores[0]:
        return alt, "colour_flip"
    return d0, "colour_keep"


def dilate(cells):
    out = set()
    for key in cells:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                out.add((key[0] + dx, key[1] + dy))
    return out


def estimate_modeb(cam_live, cam_vid, rows_live, rows_vid):
    conv = choose_convention(cam_live, rows_live)
    report["projection"] = conv
    if conv is None:
        report["events"].append(["modeB_no_projection_fallback_A"])
        return None
    rgb_v = None
    rgb_l = None
    try:
        rgb_v = as_rows(cam_vid.rgb)
        rgb_l = as_rows(cam_live.rgb)
    except Exception:
        rgb_v = None
        rgb_l = None
    if rgb_v is None or rgb_l is None:
        rgb_v = None
        rgb_l = None
    pv = project(cam_vid, rows_vid, conv[0], 0, len(rows_vid), MODEB_STRIDE, rgb_v)
    pl = project(cam_live, rows_live, conv[0], 0, len(rows_live), MODEB_STRIDE, rgb_l)
    table_z = quantile([p[2] for p in pl], 0.5)
    report["table_z"] = round(table_z, 4)
    # Same estimator on both clouds: box = connected cluster of the high layer nearest the
    # video's final (inserted) EE position; peg = connected cluster of the low layer nearest
    # the video's grasp point, with the box footprint excluded.
    gv_hi = grid_of(pv, table_z + 0.035, table_z + 0.25)
    gl_hi = grid_of(pl, table_z + 0.035, table_z + 0.25)
    vbox = cluster_from(gv_hi, VP[N - 1], 0.12, set())
    lbox = None
    if vbox is not None:
        lbox = cluster_from(gl_hi, vbox, 0.12, set())
    vpeg = None
    lpeg = None
    if vbox is not None and lbox is not None:
        gv_lo = grid_of(pv, table_z + 0.005, table_z + 0.06)
        gl_lo = grid_of(pl, table_z + 0.005, table_z + 0.06)
        vpeg = cluster_from(gv_lo, VP[close_idx], 0.04, dilate(vbox[5]))
        if vpeg is not None:
            lpeg = cluster_from(gl_lo, vpeg, 0.08, dilate(lbox[5]))
    if vbox is not None and lbox is not None:
        report["box_points"] = [vbox[3], lbox[3]]
    if vpeg is None or lpeg is None or vbox is None or lbox is None:
        report["events"].append(["modeB_estimation_failed_fallback_A"])
        return None
    report["peg_points"] = [vpeg[3], lpeg[3]]
    report["peg_yaw_deg"] = [round(math.degrees(vpeg[4]), 1), round(math.degrees(lpeg[4]), 1)]
    report["peg_anisotropy"] = [round(vpeg[7], 1), round(lpeg[7], 1)]
    rot = peg_rotation(vpeg, lpeg, VP[close_idx])
    report["peg_dtheta_deg"] = [round(math.degrees(rot[0]), 1), rot[1]]
    d_peg = [lpeg[0] - vpeg[0], lpeg[1] - vpeg[1], 0.0]
    d_box = [lbox[0] - vbox[0], lbox[1] - vbox[1], 0.0]
    report["d_peg"] = [round(v, 4) for v in d_peg]
    report["d_box"] = [round(v, 4) for v in d_box]
    return {"d_peg": d_peg, "d_box": d_box, "dtheta": rot[0],
            "c_v": [vpeg[0], vpeg[1]], "c_l": [lpeg[0], lpeg[1]]}


def run_policy():
    # ---- scene-consistency check: video frame 0 vs live (depth is lighting-invariant) ----
    live = get_observation()
    cam_live = live.cameras["front"]
    cam_vid = video[0].cameras["front"]
    rows_live = as_rows(cam_live.depth_m)
    rows_vid = as_rows(cam_vid.depth_m)
    same_pose = max(abs(float(cam_live.camera_pose.position[i]) - float(cam_vid.camera_pose.position[i]))
                    for i in range(3)) < 1e-3
    changed = 0
    total = 0
    if same_pose and len(rows_live) == len(rows_vid):
        for r in range(0, len(rows_live), PIX_STRIDE):
            a = rows_live[r]
            b = rows_vid[r]
            for c in range(0, len(a), PIX_STRIDE):
                da = float(a[c])
                db = float(b[c])
                if da != da or db != db:
                    continue
                total += 1
                if abs(da - db) > DEPTH_DIFF_M:
                    changed += 1
    frac = (float(changed) / total) if total > 0 else 1.0
    report["scene_check"] = {"same_camera_pose": same_pose, "changed_frac": round(frac, 5),
                             "pixels": total}
    mode = "A" if (same_pose and frac <= SCENE_SAME_FRAC) else "B"

    # ---- Mode B estimation (falls back to A if the scene turns out effectively unchanged) ----
    d_peg = [0.0, 0.0, 0.0]
    d_box = [0.0, 0.0, 0.0]
    dtheta = 0.0
    c_v = [0.0, 0.0]
    c_l = [0.0, 0.0]
    if mode == "B":
        est = None
        try:
            est = estimate_modeb(cam_live, cam_vid, rows_live, rows_vid)
        except Exception:
            report["events"].append(["modeB_estimation_exception_fallback_A"])
            est = None
        if est is None:
            mode = "A"
        else:
            d_peg = est["d_peg"]
            d_box = est["d_box"]
            dtheta = est["dtheta"]
            c_v = est["c_v"]
            c_l = est["c_l"]
            if math.sqrt(d_peg[0] ** 2 + d_peg[1] ** 2) < SHIFT_SAME_M and \
                    math.sqrt(d_box[0] ** 2 + d_box[1] ** 2) < SHIFT_SAME_M and dtheta == 0.0:
                mode = "A"
    report["mode"] = mode

    # ---- build the per-frame joint targets ----
    TARGET = [VQ[j] for j in range(N)]
    if mode == "B":
        approach_idx = N - 1
        for j in range(settle_idx, N):
            if math.sqrt((VP[j][0] - VP[N - 1][0]) ** 2 + (VP[j][1] - VP[N - 1][1]) ** 2) < 0.10:
                approach_idx = j
                break
        keys = list(range(0, N, KEY_STRIDE))
        for extra in (close_idx - 1, close_idx, settle_idx, approach_idx, N - 1):
            if 0 <= extra < N and extra not in keys:
                keys.append(extra)
        keys = sorted(keys)
        sol = {}
        prev = VQ[0]
        prev_key = 0
        ik_fail = 0
        for j in keys:
            if j <= settle_idx:
                w = 0.0
            elif j >= approach_idx:
                w = 1.0
            else:
                w = float(j - settle_idx) / float(max(1, approach_idx - settle_idx))
            # grasp-side transform: rotate about the video peg centre by dtheta, move onto
            # the live peg centre; box-side transform: translate by d_box; blend in between.
            rel = rot2([VP[j][0] - c_v[0], VP[j][1] - c_v[1]], dtheta)
            pg = [c_l[0] + rel[0], c_l[1] + rel[1]]
            if dtheta == 0.0:
                pg = [VP[j][0] + d_peg[0], VP[j][1] + d_peg[1]]
            pb = [VP[j][0] + d_box[0], VP[j][1] + d_box[1]]
            pos = [(1 - w) * pg[0] + w * pb[0], (1 - w) * pg[1] + w * pb[1], VP[j][2]]
            a = (1 - w) * dtheta
            quat = quat_mul([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)], VQUAT[j])
            qj = None
            try:
                ik = solve_ik(Pose(pos, quat, BASE), seed_joints=prev)
                if ik.ok:
                    qj = [float(v) for v in ik.joint_positions]
            except Exception:
                qj = None
            if qj is None:
                ik_fail += 1
                qj = [prev[i] + VQ[j][i] - VQ[prev_key][i] for i in range(len(prev))]
            sol[j] = qj
            prev = qj
            prev_key = j
        report["modeB_ik_fail"] = ik_fail
        # interpolate joint targets between keyframes
        for a, b in zip(keys[:-1], keys[1:]):
            for j in range(a, b + 1):
                s = float(j - a) / float(max(1, b - a))
                TARGET[j] = [sol[a][i] + s * (sol[b][i] - sol[a][i]) for i in range(len(sol[a]))]

    # ---- per-tick schedule (video pacing), then budget-aware resampling of the free-space
    #      pre-grasp segment under a per-tick joint-step bound ----
    PRE_MARGIN_FRAMES = 20     # frames before the close kept at original pacing
    MAX_DQ_TICK = 0.08         # rad per tick bound when time-compressing (Panda ~2 rad/s at 20 Hz)
    RESERVE_TICKS = CONVERGE_MAX_TICKS + FINAL_HOLD_TICKS


    def interp(a, b, f):
        return [a[i] + f * (b[i] - a[i]) for i in range(len(a))]


    pre_end = max(1, close_idx - PRE_MARGIN_FRAMES)
    pre_path = [TARGET[0]]                 # one entry per original tick, frames 0..pre_end
    for j in range(1, pre_end + 1):
        nt = ticks_between(j)
        for s in range(1, nt + 1):
            pre_path.append(interp(TARGET[j - 1], TARGET[j], float(s) / float(nt)))
    post_ticks = 0
    for j in range(pre_end + 1, N):
        post_ticks += ticks_between(j)
    pre_ticks = len(pre_path) - 1
    needed = pre_ticks + post_ticks + RESERVE_TICKS
    report["needed_ticks"] = needed
    factor = 1.0
    if max_steps is not None and needed > max_steps and pre_ticks > 0:
        allowed_pre = max_steps - post_ticks - RESERVE_TICKS
        factor = float(pre_ticks) / float(max(1, allowed_pre))
        # per-tick joint-step bound caps the compression
        peak = 0.0
        for i in range(1, len(pre_path)):
            peak = max(peak, max_err(pre_path[i], pre_path[i - 1]))
        if peak > 0.0:
            factor = min(factor, MAX_DQ_TICK / peak)
        factor = max(1.0, factor)
        if allowed_pre < 1 or float(pre_ticks) / factor > allowed_pre:
            report["events"].append(["budget_short", needed, max_steps])
    report["pre_grasp_time_factor"] = round(factor, 3)

    SCHED = []                              # (q, grip, tag)
    if factor > 1.0:
        m = int(math.ceil(pre_ticks / factor))
        for i in range(1, m + 1):
            x = min(float(pre_ticks), i * factor)
            lo = int(math.floor(x))
            hi = min(pre_ticks, lo + 1)
            SCHED.append((interp(pre_path[lo], pre_path[hi], x - lo), grip_for(0), "pre"))
    else:
        for i in range(1, len(pre_path)):
            SCHED.append((pre_path[i], grip_for(0), "pre"))
    for j in range(pre_end + 1, N):
        nt = ticks_between(j)
        for s in range(1, nt + 1):
            tag = "close" if (j == close_idx and s == 1) else "main"
            SCHED.append((interp(TARGET[j - 1], TARGET[j], float(s) / float(nt)), grip_for(j), tag))
    report["schedule_ticks"] = len(SCHED)

    # ---- execute ----
    q_now = measured_q()
    if max_err(q_now, TARGET[0]) > 0.01:
        mv = move_to_joints(TARGET[0], tolerance=0.01, max_steps=60)
        report["events"].append(["align_start", bool(mv.ok)])

    prev_q = TARGET[0]
    for entry in SCHED:
        if RUN["terminal"]:
            break
        if entry[2] == "close":
            # make sure the arm sits on the grasp configuration before closing
            for hold in range(CONVERGE_MAX_TICKS):
                if max_err(measured_q(), prev_q) < CONVERGE_RAD:
                    break
                if not tick(prev_q, 1.0):
                    break
            report["events"].append(["close_at_tick", RUN["ticks"]])
            if RUN["terminal"]:
                break
        if not tick(entry[0], entry[1]):
            break
        prev_q = entry[0]

    if not RUN["terminal"]:
        for hold in range(FINAL_HOLD_TICKS):
            if not tick(TARGET[N - 1], grip_for(N - 1)):
                break



if ABORT is None:
    run_policy()
else:
    report["aborted"] = ABORT

final_state = get_robot_state()
report["final_gripper"] = round(float(final_state.gripper_positions["primary"]), 4)
fp = final_state.end_effector_poses["primary"].position
report["final_ee"] = [round(float(v), 4) for v in fp]
report["video_final_ee"] = [round(v, 4) for v in VP[N - 1]] if N > 0 else None
report["ticks"] = RUN["ticks"]
report["terminal"] = RUN["terminal"]
result = report
