# v01_phase_clock -- RoboMME-Plus PickXtimes (task 7). Repair of v00 from the run-1 trace
# (episode 1). UNTESTED IN SIMULATOR (0 starts remaining for this task).
#
# Run-1 trace findings that drive this repair (all re-measured online by this program):
#   * the target follows a deterministic, time-parameterised closed path: after a pause it
#     resumes exactly where it stopped and repeats the free-run positions tick for tick;
#   * hover occlusion biases the colour centroid of the target by ~2 cm, so the v00 release
#     trigger (predicted closest approach < 2 cm) never fired in cycles 4-5 (420-tick timeouts);
#   * the episode truncated at max_steps while the arm was descending to the button.
# v01 therefore estimates the period and path from unoccluded free-run detections, re-acquires
# the phase after every lift, schedules each release by this phase clock, verifies placements
# strictly, budgets steps per cycle, and presses the button faster.
#
# --- original v00 header ---
# Strategy (hypotheses from the released recordings, all quantities perceived at runtime):
#   * goal text gives the cube colour and the repetition count X;
#   * the target disc may MOVE along a closed path while the cube is not resting on it, and
#     stops while the cube rests on it (seen in 2 of 3 recordings; the third was static);
#   * so: pick the cube, lift it, hover it just above a point P that lies on the target's
#     observed path, watch the target in the front camera every control tick, and release
#     when the target arrives under the cube (or has stopped under it);
#   * verify each placement (target stopped / cube near target), repeat until X verified
#     placements, then press the button with a closed gripper.
# Low-level control is our own joint interpolation through step() so that every control tick
# is observed. The program is heavily instrumented: this single run is also the scene probe.
import math

R = {"version": "v01_phase_clock", "errors": [], "phases": [], "notes": [], "seg": {},
     "cycles": [], "conv": {}, "ik_fail": 0}
result = R

LIFT_CLEAR = 0.15      # cube bottom above table while lifted (m)
HOVER_GAP = 0.02       # cube bottom above table while waiting for the target (m)
PREGRASP_UP = 0.12     # EE height above cube top for the wrist look before the first grasp
REGRASP_UP = 0.05      # rise after a release before re-grasping (m)
V_FAST = 0.018         # Cartesian speed limit per tick in free space (m/tick)
V_SLOW = 0.008         # Cartesian speed limit per tick near contact (m/tick)
V_CARRY = 0.012        # Cartesian speed limit per tick while holding the cube (m/tick)
DQ_MAX = 0.04          # max joint change per tick (rad)
CLOSE_TICKS = 10
OPEN_TICKS = 6
RELEASE_LAT = 3        # ticks between release command and the cube landing (estimate)
CLK_LAT = 6            # ticks from first open command until the target is stopped (run 1: the
                       # stop phase equalled the phase at the last of the 6 open ticks, 3 cycles)
CYCLE_EST0 = 230       # default ticks per pick-place cycle until measured
COLOR_TOL = 60.0       # RGB distance for target-colour pixels
BUTTON_RESERVE = 170   # ticks reserved for the button press (run 1: 96 were not enough)

COLORS = {"red": (200, 30, 30), "green": (30, 200, 30), "blue": (30, 30, 200),
          "yellow": (210, 210, 30), "purple": (150, 60, 200), "orange": (230, 130, 30),
          "white": (235, 235, 235), "black": (20, 20, 20), "pink": (240, 130, 180),
          "cyan": (30, 200, 200), "brown": (130, 80, 40), "gray": (128, 128, 128),
          "grey": (128, 128, 128), "violet": (150, 60, 200), "magenta": (220, 40, 220)}
NUMBERS = {"one": 1, "once": 1, "two": 2, "twice": 2, "three": 3, "thrice": 3, "four": 4,
           "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
           "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15}

ST = {"tick": 0, "terminal": False, "term_info": None, "cmd_q": None, "cmd_x": None,
      "grip": 1.0, "last_obs": None, "track_on": False, "rep_sample": None, "step_errors": 0,
      "obs_from_step": None}
CLK = {"T": None, "path": None, "nthr": None, "o": 0, "running": True, "m_stop": None,
       "t_stop": None, "D": [], "events": [], "build": None, "free_end": None, "tries": 0}
TR = {"color": None, "tol": COLOR_TOL, "last_uv": None, "hist": [], "table_z": None,
      "wrist_last_uv": None, "whist": []}


def log_err(where, exc):
    if len(R["errors"]) < 80:
        R["errors"].append([where, ST["tick"], str(exc)[:300]])


def phase(name, info=None):
    entry = {"name": name, "tick": ST["tick"]}
    if info is not None:
        for key in info:
            entry[key] = info[key]
    R["phases"].append(entry)


# ---------------------------------------------------------------- small vector / quaternion math
def vsub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def vadd(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def vscale(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def vnorm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def dist2(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def q2m(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def m2q(m):
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return [(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s]
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s]
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s]


def mv(m, v):
    return [m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]]


def down_quat(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return m2q([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def wrap_quarter(a):
    q = math.pi / 2
    while a >= q / 2:
        a -= q
    while a < -q / 2:
        a += q
    return a


def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def r4(v):
    return round(float(v), 4)


# ---------------------------------------------------------------- robot state / stepping
def as_list(arr):
    return [float(v) for v in arr]


def cur_q():
    obs = ST["obs_from_step"]
    if obs is not None:
        try:
            return as_list(obs.robot_state.joint_positions["primary"])
        except Exception:
            pass
    return as_list(get_robot_state().joint_positions["primary"])


def cur_grip():
    return float(get_robot_state().gripper_positions["primary"])


def cur_ee():
    p = get_robot_state().end_effector_poses["primary"]
    return as_list(p.position), as_list(p.quaternion_wxyz)


def check_terminal(rep):
    flags = {}
    for name in ("terminated", "truncated"):
        try:
            if name == "terminated":
                val = rep.terminated
            else:
                val = rep.truncated
            flags[name] = bool(val)
        except Exception:
            flags[name] = None
    if flags.get("terminated") or flags.get("truncated"):
        ST["terminal"] = True
        ST["term_info"] = {"tick": ST["tick"], "flags": flags}
        try:
            ST["term_info"]["status"] = str(rep.status)[:200]
        except Exception:
            pass
    return flags


def do_step(q, g):
    if ST["terminal"]:
        return None
    try:
        rep = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=[float(v) for v in q],
            gripper_position=float(g), embodiment=ST["emb"])}))
    except Exception as exc:
        ST["step_errors"] += 1
        log_err("step", exc)
        if ST["step_errors"] > 5:
            ST["terminal"] = True
            ST["term_info"] = {"tick": ST["tick"], "reason": "step exceptions"}
        return None
    ST["tick"] += 1
    ST["grip"] = g
    if ST["rep_sample"] is None:
        ST["rep_sample"] = str(rep)[:1500]
    check_terminal(rep)
    obs = None
    try:
        obs = rep.observation
    except Exception:
        obs = None
    ST["obs_from_step"] = obs
    if ST["track_on"] and not ST["terminal"]:
        try:
            track_tick(obs)
        except Exception as exc:
            log_err("track", exc)
    return rep


def drive(q_goal, g, min_ticks=1, settle_max=0, tol=0.01, max_dq=DQ_MAX):
    q0 = ST["cmd_q"] if ST["cmd_q"] is not None else cur_q()
    dmax = 0.0
    for i in range(7):
        dmax = max(dmax, abs(q_goal[i] - q0[i]))
    n = max(int(min_ticks), int(math.ceil(dmax / max_dq)), 1)
    for k in range(1, n + 1):
        f = float(k) / n
        do_step([q0[i] + (q_goal[i] - q0[i]) * f for i in range(7)], g)
        if ST["terminal"]:
            return False
    ST["cmd_q"] = list(q_goal)
    for k in range(settle_max):
        qm = cur_q()
        err = max(abs(qm[i] - q_goal[i]) for i in range(7))
        if err < tol:
            break
        do_step(q_goal, g)
        if ST["terminal"]:
            return False
    return True


def hold(g, ticks):
    q = ST["cmd_q"] if ST["cmd_q"] is not None else cur_q()
    for k in range(ticks):
        do_step(q, g)
        if ST["terminal"]:
            return False
    ST["cmd_q"] = q
    return True


def ik(pos, quat, seed):
    for attempt in range(2):
        try:
            if attempt == 0 and seed is not None:
                res = solve_ik(Pose(list(pos), list(quat), ST["bf"]), seed_joints=list(seed))
            else:
                res = solve_ik(Pose(list(pos), list(quat), ST["bf"]))
            if res.ok:
                sol = res.joint_positions
                try:
                    sol = sol["primary"]
                except Exception:
                    pass
                return as_list(sol)
        except Exception as exc:
            log_err("ik", exc)
    R["ik_fail"] += 1
    return None


def move_cart(target, quat, g, vmax=V_FAST, settle=8, seg=0.03):
    start = ST["cmd_x"]
    if start is None:
        start = cur_ee()[0]
    d = vnorm(vsub(target, start))
    nseg = max(1, int(math.ceil(d / seg)))
    seed = ST["cmd_q"] if ST["cmd_q"] is not None else cur_q()
    for k in range(1, nseg + 1):
        wp = vadd(start, vscale(vsub(target, start), float(k) / nseg))
        q = ik(wp, quat, seed)
        if q is None:
            phase("ik_failed", {"target": [r4(v) for v in wp]})
            return False
        ticks = max(1, int(math.ceil((d / nseg) / vmax)))
        last = k == nseg
        if not drive(q, g, min_ticks=ticks, settle_max=(settle if last else 0)):
            return False
        seed = q
        ST["cmd_x"] = wp
    return True


# ---------------------------------------------------------------- images and geometry
def roi_rows(arr, y0, y1, x0, x1):
    try:
        return arr[y0:y1, x0:x1].tolist()
    except Exception:
        rows = []
        for y in range(y0, y1):
            row = []
            for x in range(x0, x1):
                px = arr[y][x]
                try:
                    row.append([int(px[0]), int(px[1]), int(px[2])])
                except Exception:
                    row.append(float(px))
            rows.append(row)
        return rows


def img_size(arr):
    try:
        return int(arr.shape[0]), int(arr.shape[1])
    except Exception:
        return len(arr), len(arr[0])


def mean_color(rgb, box, shrink=0.3):
    h, w = img_size(rgb)
    x1, y1, x2, y2 = box
    cx0 = int(max(0, x1 + (x2 - x1) * shrink))
    cx1 = int(min(w, max(cx0 + 1, x2 - (x2 - x1) * shrink)))
    cy0 = int(max(0, y1 + (y2 - y1) * shrink))
    cy1 = int(min(h, max(cy0 + 1, y2 - (y2 - y1) * shrink)))
    rows = roi_rows(rgb, cy0, cy1, cx0, cx1)
    acc = [0.0, 0.0, 0.0]
    n = 0
    for row in rows:
        for px in row:
            acc[0] += px[0]
            acc[1] += px[1]
            acc[2] += px[2]
            n += 1
    if n == 0:
        return None
    return [acc[0] / n, acc[1] / n, acc[2] / n]


def nearest_color_name(c):
    best = None
    bd = None
    for name in COLORS:
        p = COLORS[name]
        d = (c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2 + (c[2] - p[2]) ** 2
        if bd is None or d < bd:
            bd = d
            best = name
    return best


def cam_params(obs, cam):
    co = obs.cameras[cam]
    K = [[float(v) for v in row] for row in co.intrinsics]
    pose = co.camera_pose
    return K, q2m(pose.quaternion_wxyz), as_list(pose.position)


def ray_cam(u, v, K, conv):
    x = (u - K[0][2]) / K[0][0]
    y = (v - K[1][2]) / K[1][1]
    if conv == "cv":
        return [x, y, 1.0]
    if conv == "gl":
        return [x, -y, -1.0]
    return [1.0, -x, -y]


def deproject(u, v, depth, K, Rm, t, conv):
    rc = ray_cam(u, v, K, conv)
    if conv == "ros":
        pc = vscale(rc, depth)
    elif conv == "gl":
        pc = vscale(rc, depth)
    else:
        pc = vscale(rc, depth)
    return vadd(mv(Rm, pc), t)


def pix_to_plane(u, v, K, Rm, t, conv, zp):
    d = mv(Rm, ray_cam(u, v, K, conv))
    if abs(d[2]) < 1e-6:
        return None
    s = (zp - t[2]) / d[2]
    if s <= 0:
        return None
    return vadd(t, vscale(d, s))


def depth_at(obs, cam, u, v):
    dep = obs.cameras[cam].depth_m
    h, w = img_size(dep)
    x0 = int(max(0, u - 2))
    x1 = int(min(w, u + 3))
    y0 = int(max(0, v - 2))
    y1 = int(min(h, v + 3))
    rows = roi_rows(dep, y0, y1, x0, x1)
    vals = []
    for row in rows:
        for d in row:
            try:
                fd = float(d)
            except Exception:
                fd = float(d[0])
            if fd > 0.01 and fd < 10.0:
                vals.append(fd)
    return quantile(vals, 0.5)


def validate_conv(cam, obs, box, pts, label):
    if not pts or box is None:
        return
    u = 0.5 * (box[0] + box[2])
    v = 0.5 * (box[1] + box[3])
    d = depth_at(obs, cam, int(u), int(v))
    if d is None:
        return
    K, Rm, t = cam_params(obs, cam)
    med = [quantile([p[0] for p in pts], 0.5), quantile([p[1] for p in pts], 0.5),
           quantile([p[2] for p in pts], 0.5)]
    errs = {}
    for conv in ("cv", "gl", "ros"):
        p = deproject(u, v, d, K, Rm, t, conv)
        errs[conv] = r4(vnorm(vsub(p, med)))
    entry = R["conv"].get(cam, {"checks": []})
    entry["checks"].append({"label": label, "errs": errs, "depth": r4(d)})
    best = None
    for conv in errs:
        if best is None or errs[conv] < errs[best]:
            best = conv
    # accumulate votes
    votes = entry.get("votes", {})
    votes[best] = votes.get(best, 0) + 1
    entry["votes"] = votes
    if errs[best] < 0.05:
        if entry.get("chosen") is None:
            entry["chosen"] = best
        elif entry["chosen"] != best and votes[best] > votes.get(entry["chosen"], 0):
            entry["chosen"] = best
    R["conv"][cam] = entry


def conv_of(cam):
    entry = R["conv"].get(cam)
    if entry is None:
        return None
    return entry.get("chosen")


# ---------------------------------------------------------------- perception (SAM)
def seg_best(cam, prompts, want=None, min_score=0.15):
    obs = get_observation()
    rgb = obs.cameras[cam].rgb
    for prompt in prompts:
        try:
            s = segment_text(cam, prompt)
        except Exception as exc:
            log_err("segment_text " + prompt, exc)
            continue
        rec = []
        best = None
        best_val = None
        ok = False
        try:
            ok = bool(s.ok)
        except Exception:
            ok = False
        if ok:
            for cand in list(s.segmentations)[:8]:
                try:
                    score = float(cand.score)
                except Exception:
                    score = 0.0
                box = None
                try:
                    if cand.box_xyxy is not None:
                        box = [float(v) for v in cand.box_xyxy]
                except Exception:
                    box = None
                col = None
                cname = None
                if box is not None:
                    try:
                        col = mean_color(rgb, box)
                        cname = nearest_color_name(col)
                    except Exception as exc:
                        log_err("mean_color", exc)
                val = score
                if want is not None and cname is not None and cname != want:
                    val = score * 0.3
                rec.append([round(score, 3), [round(v, 1) for v in box] if box else None,
                            [int(c) for c in col] if col else None, cname])
                if box is not None and score >= min_score and (best_val is None or val > best_val):
                    best_val = val
                    best = [cand, box, score, cname]
        key = cam + ":" + prompt + "@" + str(ST["tick"])
        R["seg"][key] = rec
        if best is not None:
            return best[0], best[1], best[2], prompt, obs
    return None, None, None, None, obs


def cloud(cand, cam):
    try:
        pc = mask_to_point_cloud(cand, cam, target_frame=ST["bf"])
        pts = []
        for p in pc.points:
            pts.append([float(p[0]), float(p[1]), float(p[2])])
        return pts
    except Exception as exc:
        log_err("mask_to_point_cloud", exc)
        return []


def robust_filter(pts, rad=0.06):
    if len(pts) < 5:
        return pts
    mx = quantile([p[0] for p in pts], 0.5)
    my = quantile([p[1] for p in pts], 0.5)
    mz = quantile([p[2] for p in pts], 0.5)
    out = []
    for p in pts:
        if abs(p[0] - mx) < rad and abs(p[1] - my) < rad and abs(p[2] - mz) < rad:
            out.append(p)
    return out


def cube_from_pts(pts, table_z):
    pts = robust_filter(pts)
    if len(pts) < 8:
        return None
    zs = [p[2] for p in pts]
    top = quantile(zs, 0.92)
    topp = [p for p in pts if p[2] > top - 0.008]
    if len(topp) < 6:
        topp = pts
    if len(topp) > 400:
        stride = int(len(topp) / 400) + 1
        topp = topp[::stride]
    best = None
    for k in range(30):
        a = math.radians(3.0 * k)
        c, s = math.cos(a), math.sin(a)
        p1 = [p[0] * c + p[1] * s for p in topp]
        p2 = [-p[0] * s + p[1] * c for p in topp]
        lo1, hi1 = quantile(p1, 0.05), quantile(p1, 0.95)
        lo2, hi2 = quantile(p2, 0.05), quantile(p2, 0.95)
        area = (hi1 - lo1) * (hi2 - lo2)
        if best is None or area < best[0]:
            m1 = 0.5 * (lo1 + hi1)
            m2 = 0.5 * (lo2 + hi2)
            best = [area, a, m1 * c - m2 * s, m1 * s + m2 * c, hi1 - lo1, hi2 - lo2]
    edge = None
    if table_z is not None:
        edge = top - table_z
    if edge is None or edge < 0.015 or edge > 0.09:
        edge = max(0.02, min(0.07, 0.5 * (best[4] + best[5]) / 0.9))
    return {"x": best[2], "y": best[3], "top": top, "yaw": best[1], "edge": edge,
            "n": len(pts), "ntop": len(topp), "span": [r4(best[4]), r4(best[5])]}


def pts_summary(pts):
    if not pts:
        return None
    return {"n": len(pts),
            "x": [r4(quantile([p[0] for p in pts], f)) for f in (0.05, 0.5, 0.95)],
            "y": [r4(quantile([p[1] for p in pts], f)) for f in (0.05, 0.5, 0.95)],
            "z": [r4(quantile([p[2] for p in pts], f)) for f in (0.05, 0.5, 0.95)]}


# ---------------------------------------------------------------- target colour tracking
def estimate_target_color(rgb, box):
    h, w = img_size(rgb)
    x0 = int(max(0, box[0] - 4))
    x1 = int(min(w, box[2] + 5))
    y0 = int(max(0, box[1] - 4))
    y1 = int(min(h, box[3] + 5))
    rows = roi_rows(rgb, y0, y1, x0, x1)
    border = []
    inner = []
    for j in range(len(rows)):
        for i in range(len(rows[j])):
            px = rows[j][i]
            if j < 2 or j >= len(rows) - 2 or i < 2 or i >= len(rows[j]) - 2:
                border.append(px)
            else:
                inner.append(px)
    if not border or not inner:
        return None
    tab = [quantile([p[0] for p in border], 0.5), quantile([p[1] for p in border], 0.5),
           quantile([p[2] for p in border], 0.5)]
    sel = []
    for p in inner:
        dt = (p[0] - tab[0]) ** 2 + (p[1] - tab[1]) ** 2 + (p[2] - tab[2]) ** 2
        sat = max(p[0], p[1], p[2]) - min(p[0], p[1], p[2])
        if dt > 3600 and sat > 45:
            sel.append(p)
    if len(sel) < 4:
        return None
    return {"color": [sum(p[0] for p in sel) / len(sel), sum(p[1] for p in sel) / len(sel),
                      sum(p[2] for p in sel) / len(sel)], "n": len(sel), "table": tab}


def detect_color(rgb, color, tol, roi, stride):
    h, w = img_size(rgb)
    x0 = int(max(0, roi[0]))
    y0 = int(max(0, roi[1]))
    x1 = int(min(w, roi[2]))
    y1 = int(min(h, roi[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    rows = roi_rows(rgb, y0, y1, x0, x1)
    tol2 = tol * tol
    su = 0.0
    sv = 0.0
    n = 0
    cr, cg, cb = color[0], color[1], color[2]
    for j in range(0, len(rows), stride):
        row = rows[j]
        for i in range(0, len(row), stride):
            p = row[i]
            dr = p[0] - cr
            dg = p[1] - cg
            db = p[2] - cb
            if dr * dr + dg * dg + db * db < tol2:
                if max(p[0], p[1], p[2]) - min(p[0], p[1], p[2]) > 40:
                    su += i
                    sv += j
                    n += 1
    if n == 0:
        return None
    return [x0 + su / n + 0.5, y0 + sv / n + 0.5, n * stride * stride]


def locate_target_in(obs, cam, last_key, full_stride=2, win=30, min_n=6):
    co = obs.cameras[cam]
    rgb = co.rgb
    h, w = img_size(rgb)
    det = None
    last = TR[last_key]
    if last is not None:
        det = detect_color(rgb, TR["color"], TR["tol"],
                           [last[0] - win, last[1] - win, last[0] + win, last[1] + win], 1)
        if det is not None and det[2] < min_n:
            det = None
    if det is None:
        det = detect_color(rgb, TR["color"], TR["tol"], [0, 0, w, h], full_stride)
        if det is not None and det[2] < min_n:
            det = None
    if det is None:
        return None
    TR[last_key] = [det[0], det[1]]
    conv = conv_of(cam)
    if conv is None or TR["table_z"] is None:
        return [None, None, det[2], det[0], det[1]]
    K, Rm, t = cam_params(obs, cam)
    p = pix_to_plane(det[0], det[1], K, Rm, t, conv, TR["table_z"])
    if p is None:
        return [None, None, det[2], det[0], det[1]]
    return [p[0], p[1], det[2], det[0], det[1]]


def track_tick(obs):
    if TR["color"] is None:
        return
    if obs is None:
        obs = get_observation()
    det = locate_target_in(obs, "front", "last_uv")
    if det is not None and det[0] is not None:
        TR["hist"].append([ST["tick"], r4(det[0]), r4(det[1]), int(det[2])])
    elif det is not None:
        TR["hist"].append([ST["tick"], None, None, int(det[2])])
    if ST.get("wrist_track") and conv_of("wrist") is not None:
        try:
            wd = locate_target_in(obs, "wrist", "wrist_last_uv", full_stride=3, win=45, min_n=20)
            if wd is not None and wd[0] is not None:
                TR["whist"].append([ST["tick"], r4(wd[0]), r4(wd[1]), int(wd[2])])
        except Exception as exc:
            log_err("wrist track", exc)


def last_detection(max_age=0, src="hist", min_n=6):
    hist = TR[src]
    for k in range(len(hist) - 1, -1, -1):
        e = hist[k]
        if ST["tick"] - e[0] > max_age:
            return None
        if e[1] is not None and e[3] >= min_n:
            return e
    return None


def velocity(src="hist", window=4, min_n=6):
    hist = TR[src]
    pts = []
    for k in range(len(hist) - 1, -1, -1):
        e = hist[k]
        if ST["tick"] - e[0] > window + 2:
            break
        if e[1] is not None and e[3] >= min_n:
            pts.append(e)
    if len(pts) < 2:
        return None
    a = pts[-1]
    b = pts[0]
    dt = b[0] - a[0]
    if dt <= 0:
        return None
    return [(b[1] - a[1]) / dt, (b[2] - a[2]) / dt]


def target_moving(window=10, thresh=0.015):
    hist = TR["hist"]
    pts = [e for e in hist if e[1] is not None and e[3] >= 6 and ST["tick"] - e[0] <= window]
    if len(pts) < 3 or pts[-1][0] - pts[0][0] < window // 2:
        return None
    return dist2([pts[0][1], pts[0][2]], [pts[-1][1], pts[-1][2]]) > thresh


def snap_to_path(xy, maxd=0.05):
    best = None
    bd = None
    for e in TR["hist"]:
        if e[1] is None or e[3] < 10:
            continue
        d = dist2([e[1], e[2]], xy)
        if bd is None or d < bd:
            bd = d
            best = [e[1], e[2]]
    if best is not None and bd <= maxd:
        return best, bd
    return None, bd


# ---------------------------------------------------------------- phase clock (v01)
def n_threshold():
    if CLK["nthr"] is None:
        ns = [e[3] for e in TR["hist"] if e[1] is not None and e[0] <= 40]
        if len(ns) < 8:
            return None
        CLK["nthr"] = 0.65 * quantile(ns, 0.5)
    return CLK["nthr"]


def good_pts(t0, t1):
    thr = n_threshold()
    if thr is None:
        return []
    return [e for e in TR["hist"] if e[1] is not None and e[3] >= thr and t0 <= e[0] <= t1]


def clock_build(t_end):
    """Period and time-parameterised path from free-run (uninterrupted) good detections."""
    CLK["tries"] += 1
    pts = good_pts(0, t_end)
    if len(pts) < 40:
        return False
    by_t = {}
    for e in pts:
        by_t[e[0]] = e
    ticks = sorted(by_t)
    span = ticks[-1] - ticks[0]
    xs = [e[1] for e in pts]
    ys = [e[2] for e in pts]
    if max(max(xs) - min(xs), max(ys) - min(ys)) < 0.03:
        return False
    costs = []
    for lag in range(30, span - 8):
        tot = 0.0
        c = 0
        for t in ticks:
            e2 = by_t.get(t + lag)
            if e2 is not None:
                e1 = by_t[t]
                tot += dist2([e1[1], e1[2]], [e2[1], e2[2]])
                c += 1
        if c >= 12:
            costs.append([tot / c, lag, c])
    if not costs:
        return False
    best = min(costs)
    if best[0] > 0.006:
        return False
    # smallest lag that is a near-best local minimum (avoid choosing a multiple of the period)
    lim = max(0.004, 1.5 * best[0])
    pick = None
    for k in range(len(costs)):
        cst = costs[k]
        lo = costs[k - 1][0] if k > 0 else 1e9
        hi = costs[k + 1][0] if k + 1 < len(costs) else 1e9
        if cst[0] <= lim and cst[0] <= lo and cst[0] <= hi:
            pick = cst
            break
    if pick is None:
        pick = best
    T = pick[1]
    acc = {}
    for e in pts:
        m = e[0] % T
        if m not in acc:
            acc[m] = [0.0, 0.0, 0]
        acc[m][0] += e[1]
        acc[m][1] += e[2]
        acc[m][2] += 1
    known = sorted(acc)
    if len(known) < 0.6 * T:
        return False
    path = []
    for m in range(T):
        if m in acc:
            a = acc[m]
            path.append([a[0] / a[2], a[1] / a[2]])
            continue
        prev = None
        nxt = None
        for d in range(1, T):
            if prev is None and ((m - d) % T) in acc:
                prev = [(m - d) % T, d]
            if nxt is None and ((m + d) % T) in acc:
                nxt = [(m + d) % T, d]
            if prev is not None and nxt is not None:
                break
        a = acc[prev[0]]
        b = acc[nxt[0]]
        f = float(prev[1]) / (prev[1] + nxt[1])
        path.append([(1 - f) * a[0] / a[2] + f * b[0] / b[2], (1 - f) * a[1] / a[2] + f * b[1] / b[2]])
    res = [dist2([e[1], e[2]], path[e[0] % T]) for e in pts]
    CLK["T"] = T
    CLK["path"] = path
    CLK["o"] = 0
    CLK["build"] = {"tick": ST["tick"], "t_end": t_end, "T": T, "cost": r4(pick[0]), "pairs": pick[2],
                    "best_lag": best[1], "n_pts": len(pts), "phases_seen": len(known),
                    "fit_mean": r4(sum(res) / len(res)), "fit_max": r4(max(res)), "nthr": round(CLK["nthr"], 1)}
    return True


def clock_pos(t):
    return CLK["path"][(t - CLK["o"]) % CLK["T"]]


def clock_match(t_from, max_cost=0.008):
    """Offset o such that the target at tick t is path[(t - o) mod T], from recent good detections."""
    T = CLK["T"]
    allp = good_pts(t_from, ST["tick"])
    path = CLK["path"]
    out = None
    for keep in (10, 7, 5):
        pts = allp[-keep:]
        if len(pts) < 4:
            break
        if dist2([pts[0][1], pts[0][2]], [pts[-1][1], pts[-1][2]]) < 0.01:
            continue
        best = None
        for o in range(T):
            tot = 0.0
            for e in pts:
                p = path[(e[0] - o) % T]
                tot += math.sqrt((e[1] - p[0]) ** 2 + (e[2] - p[1]) ** 2)
            tot = tot / len(pts)
            if best is None or tot < best[0]:
                best = [tot, o]
        if best[0] <= max_cost and (out is None or best[0] < out[0]):
            out = [best[0], best[1], len(pts)]
        if out is not None and out[0] <= 0.5 * max_cost:
            break
    return out


def clock_residual(t0, t1):
    if CLK["T"] is None:
        return None
    res = [dist2([e[1], e[2]], clock_pos(e[0])) for e in good_pts(t0, t1)]
    if not res:
        return None
    return {"n": len(res), "mean": r4(sum(res) / len(res)), "max": r4(max(res))}


def est_ticks(d, v):
    return int(math.ceil(abs(d) / v)) + 1


def plan_intercept(t_now, here, lift_z, hover_z, quat):
    T = CLK["T"]
    m_now = (t_now - CLK["o"]) % T
    for delta in range(CLK_LAT + 4, T + CLK_LAT + 4):
        m = (m_now + delta) % T
        P = CLK["path"][m]
        lateral = dist2(P, [here[0], here[1]])
        est = est_ticks(lateral, V_CARRY) + est_ticks(lift_z - hover_z, V_CARRY) + 6
        need = int(1.35 * est) + 10
        if delta - CLK_LAT >= need:
            if ik([P[0], P[1], hover_z], quat, ST["cmd_q"]) is None:
                continue
            return {"m_now": m_now, "delta": delta, "m": m, "P": [P[0], P[1]],
                    "t_open": t_now + delta - CLK_LAT, "est": est, "need": need}
    return None


def reacquire(lift_start, cyc):
    """After a lift the target resumes; find its new clock offset (or predict it)."""
    got = clock_match(lift_start)
    if got is None:
        for k in range(25):
            hold(0.0, 1)
            if ST["terminal"]:
                return False
            got = clock_match(lift_start)
            if got is not None:
                break
    if got is not None:
        CLK["o"] = got[1]
        if not CLK["running"] and CLK["m_stop"] is not None:
            t_res = CLK["o"] + CLK["m_stop"]
            while t_res < CLK["t_stop"]:
                t_res += CLK["T"]
            while t_res > ST["tick"]:
                t_res -= CLK["T"]
            CLK["D"].append(t_res - lift_start)
            cyc["resume_delay"] = t_res - lift_start
        CLK["running"] = True
        cyc["clock_src"] = "reacquired"
        cyc["clock_fit"] = [r4(got[0]), got[2]]
        return True
    if not CLK["running"] and CLK["m_stop"] is not None and CLK["D"]:
        d = int(round(quantile(CLK["D"], 0.5)))
        CLK["o"] = lift_start + d - CLK["m_stop"]
        CLK["running"] = True
        cyc["clock_src"] = "predicted_D" + str(d)
        return True
    if CLK["running"]:
        cyc["clock_src"] = "unchanged_running"
        return True
    cyc["clock_src"] = "lost"
    return False


# ---------------------------------------------------------------- task logic
def build_loop(t0, t1):
    """Timed target path between the tick it resumed (t0) and the tick it was caught (t1)."""
    loop = []
    for e in TR["hist"]:
        if e[0] < t0 or e[0] > t1 or e[1] is None or e[3] < 10:
            continue
        loop.append([e[0] - t0, e[1], e[2]])
    return loop


def choose_intercept(loop, lift_start, here_xy, rise_ticks, descend_ticks, quat, hover_z):
    """Earliest point of the previous timed path the arm can reach before the target does."""
    if not loop or loop[-1][0] < 40:
        return None
    period = loop[-1][0]
    for e in loop:
        tau = e[0]
        if tau > period - 15:
            break
        lateral = dist2([e[1], e[2]], here_xy)
        travel = rise_ticks + lateral / V_CARRY + descend_ticks + 14
        if ST["tick"] - lift_start + travel <= tau:
            if ik([e[1], e[2], hover_z], quat, ST["cmd_q"]) is None:
                continue
            return [e[1], e[2], tau, period]
    return None


def release_decision(P):
    """Decide whether to open now so the cube lands as the target centre passes P."""
    lastg = last_detection(max_age=6)
    vel = velocity(window=6)
    if lastg is None:
        return None
    age = ST["tick"] - lastg[0]
    if vel is None:
        d = dist2([lastg[1], lastg[2]], P)
        if age == 0 and d <= 0.006:
            return ["arrived", {"d": r4(d)}]
        return None
    rel = [lastg[1] + vel[0] * age - P[0], lastg[2] + vel[1] * age - P[1]]
    d = math.sqrt(rel[0] ** 2 + rel[1] ** 2)
    sp = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
    info = {"d": r4(d), "sp": r4(sp), "age": age, "n": lastg[3]}
    if d <= 0.006:
        return ["arrived", info]
    if sp < 0.0012:
        return None
    tstar = -(rel[0] * vel[0] + rel[1] * vel[1]) / (sp * sp)
    dmin = math.sqrt((rel[0] + vel[0] * tstar) ** 2 + (rel[1] + vel[1] * tstar) ** 2)
    info["tstar"] = round(tstar, 2)
    info["dmin"] = r4(dmin)
    if 0 <= tstar <= RELEASE_LAT + 0.5 and dmin < 0.02:
        return ["closest_approach", info]
    if tstar < 0:
        late = math.sqrt((rel[0] + vel[0] * RELEASE_LAT) ** 2 + (rel[1] + vel[1] * RELEASE_LAT) ** 2)
        if late < 0.015 and d < 0.02:
            info["late"] = r4(late)
            return ["late_release", info]
    return None


def parse_goal(text):
    toks = []
    cur = ""
    for ch in text.lower():
        if ch.isalnum():
            cur += ch
        else:
            if cur:
                toks.append(cur)
            cur = ""
    if cur:
        toks.append(cur)
    color = None
    for i in range(1, len(toks)):
        if toks[i] in ("cube", "block") and toks[i - 1] in COLORS:
            color = toks[i - 1]
            break
    count = None
    for i in range(len(toks)):
        t = toks[i]
        if t in ("once", "twice", "thrice"):
            count = NUMBERS[t]
            break
        if t in ("time", "times") and i > 0:
            p = toks[i - 1]
            if p in NUMBERS:
                count = NUMBERS[p]
                break
            if p.isdigit():
                count = int(p)
                break
    return color, count, toks


def wrist_cube(prompts, want):
    cand, box, score, prompt, obs = seg_best("wrist", prompts, want=want)
    if cand is None:
        return None
    pts = cloud(cand, "wrist")
    validate_conv("wrist", obs, box, robust_filter(pts), "cube")
    cube = cube_from_pts(pts, TR["table_z"])
    if cube is not None:
        cube["src"] = "wrist"
        cube["score"] = round(score, 3)
    return cube


def front_cube(prompts, want):
    cand, box, score, prompt, obs = seg_best("front", prompts, want=want)
    if cand is None:
        return None
    pts = cloud(cand, "front")
    validate_conv("front", obs, box, robust_filter(pts), "cube")
    cube = cube_from_pts(pts, TR["table_z"])
    if cube is not None:
        cube["src"] = "front"
        cube["score"] = round(score, 3)
        cube["box"] = [round(v, 1) for v in box]
    return cube


def cube_record(c):
    if c is None:
        return None
    return {"x": r4(c["x"]), "y": r4(c["y"]), "top": r4(c["top"]), "yaw_deg": round(math.degrees(c["yaw"]), 1),
            "edge": r4(c["edge"]), "n": c["n"], "src": c.get("src"), "span": c.get("span")}


def grasp_quat(cube_yaw):
    psi = ST["yaw0"] + wrap_quarter(cube_yaw - ST["yaw0"])
    return down_quat(psi)


def grasp_cube(cube, cyc):
    """Descend on the cube from above (gripper open), close, verify width."""
    quat = grasp_quat(cube["yaw"])
    gz = cube["top"] - 0.55 * cube["edge"]
    if TR["table_z"] is not None:
        gz = max(gz, TR["table_z"] + 0.012)
    above = [cube["x"], cube["y"], cube["top"] + 0.06]
    if not move_cart(above, quat, 1.0, vmax=V_FAST):
        return False, quat, gz
    if not move_cart([cube["x"], cube["y"], gz], quat, 1.0, vmax=V_SLOW, settle=10):
        return False, quat, gz
    hold(0.0, CLOSE_TICKS)
    width = cur_grip()
    cyc["grip_width"] = round(width, 3)
    cyc["grasp_z"] = r4(gz)
    return width > 0.08, quat, gz


def main():
    state = get_robot_state()
    ST["bf"] = state.base_frame
    ST["emb"] = state.embodiment
    R["base_frame"] = str(state.base_frame)
    R["embodiment"] = str(state.embodiment)
    ctx = get_task_context()
    R["goal"] = ctx.language
    try:
        R["task_meta"] = dict(robomme_plus.get_task_metadata())
    except Exception as exc:
        log_err("task_meta", exc)
        R["task_meta"] = {}
    try:
        R["ctrl_meta"] = dict(robomme_plus.get_controller_metadata())
    except Exception as exc:
        log_err("ctrl_meta", exc)
    try:
        video = robomme_plus.get_conditioning_video()
        R["video_frames"] = len(video)
    except Exception as exc:
        log_err("video", exc)
    max_steps = None
    for key in ("max_steps", "max_episode_steps"):
        try:
            if R["task_meta"].get(key) is not None:
                max_steps = int(R["task_meta"][key])
                break
        except Exception:
            pass
    R["max_steps"] = max_steps
    if max_steps is None:
        max_steps = 2000
    color, count, toks = parse_goal(str(ctx.language))
    R["parsed"] = {"color": color, "count": count}
    if count is None:
        count = 1
        R["notes"].append("count not parsed; defaulting to 1")
    cube_prompts = []
    if color is not None:
        cube_prompts.append(color + " cube")
    cube_prompts.append("cube")

    q_home = cur_q()
    ee_pos, ee_quat = cur_ee()
    m = q2m(ee_quat)
    R["home"] = {"q": [r4(v) for v in q_home], "ee": [r4(v) for v in ee_pos],
                 "quat": [r4(v) for v in ee_quat], "z_axis": [r4(m[0][2]), r4(m[1][2]), r4(m[2][2])],
                 "grip": round(cur_grip(), 3), "ts": float(state.timestamp_s)}
    ST["yaw0"] = math.atan2(m[1][0], m[0][0])
    ST["cmd_q"] = q_home
    ST["cmd_x"] = ee_pos

    obs = get_observation()
    cams = {}
    for name in obs.cameras:
        co = obs.cameras[name]
        try:
            cams[name] = {"shape": list(co.rgb.shape), "K": [[r4(v) for v in row] for row in co.intrinsics],
                          "pos": [r4(v) for v in co.camera_pose.position],
                          "quat": [r4(v) for v in co.camera_pose.quaternion_wxyz],
                          "frame": str(co.camera_pose.frame)}
        except Exception as exc:
            log_err("cam " + str(name), exc)
    R["cameras"] = cams

    # --- initial front perception: target, button, cube
    phase("perceive_front")
    tcand, tbox, tscore, tprompt, tobs = seg_best(
        "front", ["target", "purple target", "bullseye target", "round target on the table"])
    target_pts = []
    if tcand is not None:
        target_pts = robust_filter(cloud(tcand, "front"), rad=0.08)
        R["target_front"] = {"prompt": tprompt, "score": round(tscore, 3), "box": tbox,
                             "cloud": pts_summary(target_pts)}
        if target_pts:
            TR["table_z"] = quantile([p[2] for p in target_pts], 0.5)
            ST["target_sam_xy"] = [quantile([p[0] for p in target_pts], 0.5),
                                   quantile([p[1] for p in target_pts], 0.5)]
        validate_conv("front", tobs, tbox, target_pts, "target")
        tc = estimate_target_color(tobs.cameras["front"].rgb, tbox)
        R["target_color_est"] = tc
        if tc is not None:
            TR["color"] = tc["color"]
            TR["last_uv"] = [0.5 * (tbox[0] + tbox[2]), 0.5 * (tbox[1] + tbox[3])]
    if TR["color"] is None:
        TR["color"] = list(COLORS["purple"])
        R["notes"].append("target colour fallback to generic purple")

    bcand, bbox, bscore, bprompt, bobs = seg_best("front", ["button", "white button", "round button"])
    button = None
    if bcand is not None:
        bpts = robust_filter(cloud(bcand, "front"), rad=0.06)
        validate_conv("front", bobs, bbox, bpts, "button")
        if bpts:
            ztop = quantile([p[2] for p in bpts], 0.9)
            cap = [p for p in bpts if p[2] > ztop - 0.006]
            button = {"x": quantile([p[0] for p in cap], 0.5), "y": quantile([p[1] for p in cap], 0.5),
                      "top": ztop, "zmin": quantile([p[2] for p in bpts], 0.05)}
        R["button_front"] = {"prompt": bprompt, "score": round(bscore, 3), "box": bbox,
                             "cloud": pts_summary(bpts),
                             "press": None if button is None else [r4(button["x"]), r4(button["y"]), r4(button["top"])]}

    cube = front_cube(cube_prompts, color)
    R["cube_front0"] = cube_record(cube)
    if TR["table_z"] is None and cube is not None:
        TR["table_z"] = cube["top"] - 0.04
        R["notes"].append("table z from cube top minus 4 cm (no target cloud)")
    R["table_z"] = None if TR["table_z"] is None else r4(TR["table_z"])
    R["target_color"] = [int(c) for c in TR["color"]]

    # initial target detection in the front image
    d0 = locate_target_in(get_observation(), "front", "last_uv")
    R["target_det0"] = d0
    ST["track_on"] = True
    if cube is None:
        R["notes"].append("cube not found in front view")
        return

    # --- approach: above the cube, look with the wrist camera
    phase("approach")
    quat0 = grasp_quat(cube["yaw"])
    if not move_cart([cube["x"], cube["y"], cube["top"] + PREGRASP_UP], quat0, 1.0, vmax=V_FAST):
        R["notes"].append("approach failed")
        return
    ST["wrist_track"] = False
    wc = wrist_cube(cube_prompts, color)
    R["cube_wrist0"] = cube_record(wc)
    if wc is not None and dist2([wc["x"], wc["y"]], [cube["x"], cube["y"]]) < 0.06:
        cube = wc
    moving_initial = target_moving(window=60, thresh=0.02)
    R["target_moving_initial"] = moving_initial

    placed = 0
    attempts = 0
    have_cube_above = False
    while placed < count and attempts < count + 4 and not ST["terminal"]:
        attempts += 1
        cyc = {"k": attempts, "start_tick": ST["tick"], "cube": cube_record(cube)}
        R["cycles"].append(cyc)
        durs = [c["end_tick"] - c["start_tick"] for c in R["cycles"] if c.get("end_tick") is not None]
        cyc_est = CYCLE_EST0 if not durs else max(150, int(quantile(durs[-3:], 0.5)))
        cyc["cyc_est"] = cyc_est
        if ST["tick"] + BUTTON_RESERVE + min(cyc_est, 200) > max_steps:
            cyc["abort"] = "step budget"
            break
        ok, quat, gz = grasp_cube(cube, cyc)
        cyc["grasp_ok"] = ok
        if ST["terminal"]:
            break
        if not ok:
            hold(1.0, OPEN_TICKS)
            move_cart([cube["x"], cube["y"], cube["top"] + PREGRASP_UP], quat, 1.0, vmax=V_FAST)
            nc = wrist_cube(cube_prompts, color)
            if nc is None:
                nc = front_cube(cube_prompts, color)
            cyc["relocalized"] = cube_record(nc)
            if nc is not None:
                cube = nc
            continue
        # lift
        tz = TR["table_z"] if TR["table_z"] is not None else cube["top"] - cube["edge"]
        off = gz - (cube["top"] - cube["edge"])          # EE height above cube bottom
        lift_z = tz + LIFT_CLEAR + off
        hover_z = tz + HOVER_GAP + off
        cyc["lift_start_tick"] = ST["tick"]
        if not move_cart([cube["x"], cube["y"], lift_z], quat, 0.0, vmax=V_CARRY):
            cyc["abort"] = "lift ik"
            break
        cyc["lift_tick"] = ST["tick"]
        cyc["grip_after_lift"] = round(cur_grip(), 3)
        if cyc["grip_after_lift"] < 0.05:
            cyc["note"] = "cube lost during lift"
            hold(1.0, OPEN_TICKS)
            nc = front_cube(cube_prompts, color)
            if nc is None:
                break
            cube = nc
            move_cart([cube["x"], cube["y"], cube["top"] + PREGRASP_UP], quat, 1.0, vmax=V_FAST)
            continue
        moving = moving_initial is not False
        plan = None
        t_P = None
        if moving and CLK["T"] is None and CLK["free_end"] is None:
            clock_build(ST["tick"])
        if moving and CLK["T"] is not None:
            if reacquire(cyc["lift_start_tick"], cyc):
                plan = plan_intercept(ST["tick"], ST["cmd_x"], lift_z, hover_z, quat)
        if ST["terminal"]:
            break
        if plan is not None:
            P = plan["P"]
            cyc["P_src"] = "clock"
            cyc["plan"] = {"m_now": plan["m_now"], "delta": plan["delta"], "m": plan["m"],
                           "t_open": plan["t_open"], "est": plan["est"], "need": plan["need"],
                           "T": CLK["T"], "o": CLK["o"]}
        else:
            # v00 choice: a recent, well-seen target position (its phase is known later)
            thr = n_threshold()
            latest = last_detection(max_age=3, min_n=(6 if thr is None else int(thr)))
            if latest is None:
                latest = last_detection(max_age=3)
            snapped, sd = snap_to_path([cube["x"], cube["y"]], maxd=0.05)
            if latest is not None:
                P = [latest[1], latest[2]]
                t_P = latest[0]
                cyc["P_src"] = "latest_target"
            elif snapped is not None:
                P = snapped
                cyc["P_src"] = "path_near_cube"
            elif ST.get("target_sam_xy") is not None:
                P = list(ST["target_sam_xy"])
                cyc["P_src"] = "sam_target_initial"
            else:
                P = [cube["x"], cube["y"]]
                cyc["P_src"] = "cube_xy_fallback"
        cyc["P"] = [r4(P[0]), r4(P[1])]
        cyc["t_P"] = t_P
        if not move_cart([P[0], P[1], lift_z], quat, 0.0, vmax=V_CARRY, settle=0):
            cyc["abort"] = "transfer ik"
            break
        if not move_cart([P[0], P[1], hover_z], quat, 0.0, vmax=V_CARRY, settle=6):
            cyc["abort"] = "hover ik"
            break
        q_hover = ST["cmd_q"]
        cyc["hover_tick"] = ST["tick"]
        t_open = None
        if plan is not None:
            t_open = plan["t_open"]
            while t_open < ST["tick"]:
                t_open += CLK["T"]
                cyc["late_laps"] = cyc.get("late_laps", 0) + 1
        # wait for the target to arrive under the cube
        ST["wrist_track"] = True
        budget_left = max_steps - BUTTON_RESERVE - ST["tick"]
        cap = 420 if CLK["T"] is None else CLK["T"] + 40
        max_wait = max(20, min(cap, budget_left - 30))
        reason = "timeout"
        stopped_run = 0
        info = {}
        if not moving:
            max_wait = 0
            reason = "static_target"
        for w in range(max_wait):
            if t_open is None and t_P is not None and CLK["T"] is None and CLK["free_end"] is None \
                    and w % 8 == 0:
                if clock_build(ST["tick"]):
                    m_P = (t_P - CLK["o"]) % CLK["T"]
                    t_open = ST["tick"] + ((m_P - (ST["tick"] + CLK_LAT - CLK["o"])) % CLK["T"])
                    cyc["P_src"] = "latest_target_clock"
                    cyc["plan"] = {"m": m_P, "t_open": t_open, "T": CLK["T"], "o": CLK["o"], "built_tick": ST["tick"]}
            if t_open is not None:
                if ST["tick"] >= t_open:
                    reason = "clock"
                    info = {"w": w, "t_open": t_open}
                    break
                do_step(q_hover, 0.0)
                if ST["terminal"]:
                    reason = "terminal"
                    break
                continue
            do_step(q_hover, 0.0)
            if ST["terminal"]:
                reason = "terminal"
                break
            fresh = last_detection(max_age=0)
            if fresh is not None:
                dfresh = dist2([fresh[1], fresh[2]], P)
                vfresh = velocity(window=4)
                if vfresh is not None and math.sqrt(vfresh[0] ** 2 + vfresh[1] ** 2) < 0.0012 and dfresh < 0.04:
                    stopped_run += 1
                else:
                    stopped_run = 0
                if stopped_run >= 3:
                    reason = "target_stopped_near"
                    info = {"d": r4(dfresh), "w": w}
                    break
            dec = release_decision(P)
            if dec is not None:
                reason = dec[0]
                info = dec[1]
                info["w"] = w
                break
        ST["wrist_track"] = False
        cyc["wait_ticks"] = ST["tick"] - cyc["hover_tick"]
        cyc["release_reason"] = reason
        cyc["release_info"] = info
        if CLK["T"] is not None:
            cyc["clock_resid_wait"] = clock_residual(cyc["lift_start_tick"], ST["tick"])
            cyc["clock_pred_at_open"] = [r4(v) for v in clock_pos(ST["tick"] + CLK_LAT)]
        if ST["terminal"]:
            break
        t_open_actual = ST["tick"]
        hold(1.0, OPEN_TICKS)
        cyc["release_tick"] = ST["tick"]
        cyc["open_tick"] = t_open_actual
        # rise a little and verify the placement
        move_cart([P[0], P[1], hover_z + REGRASP_UP], quat, 1.0, vmax=V_SLOW, settle=4)
        hold(1.0, 6)
        mv_after = target_moving(window=14, thresh=0.025)
        cyc["target_moving_after"] = mv_after
        ST["wrist_track"] = False
        wobs = get_observation()
        ring = None
        try:
            ring = locate_target_in(wobs, "wrist", "wrist_last_uv", full_stride=3, win=60, min_n=20)
        except Exception as exc:
            log_err("ring", exc)
        nc = wrist_cube(cube_prompts, color)
        if nc is None:
            nc = front_cube(cube_prompts, color)
        cyc["cube_after"] = cube_record(nc)
        ring_near = None
        if ring is not None and ring[0] is not None and nc is not None:
            rd = dist2([ring[0], ring[1]], [nc["x"], nc["y"]])
            ring_near = rd < 0.04 and ring[2] >= 150
            cyc["wrist_ring"] = {"d": r4(rd), "n": int(ring[2])}
        cyc["ring_near"] = ring_near
        lastt = last_detection(max_age=12)
        if nc is not None and lastt is not None:
            cyc["cube_target_d"] = r4(dist2([nc["x"], nc["y"]], [lastt[1], lastt[2]]))
        dct = cyc.get("cube_target_d")
        far = dct is not None and dct > 0.03
        if not moving:
            verified = not far
        elif mv_after is True:
            verified = False
        elif mv_after is False:
            verified = True
        else:
            verified = ring_near is True
        cyc["verified"] = verified
        if moving:
            if CLK["T"] is None and verified and CLK["free_end"] is None:
                CLK["free_end"] = t_open_actual
                clock_build(t_open_actual)
            if CLK["T"] is not None:
                if CLK["free_end"] is None:
                    CLK["free_end"] = t_open_actual
                if verified:
                    CLK["running"] = False
                    CLK["t_stop"] = t_open_actual + CLK_LAT
                    CLK["m_stop"] = (CLK["t_stop"] - CLK["o"]) % CLK["T"]
                    cyc["m_stop"] = CLK["m_stop"]
            CLK["events"].append([ST["tick"], "verified" if verified else "unverified", CLK["o"], CLK["m_stop"]])
            if CLK["free_end"] is None and not verified:
                pass
        if verified:
            placed += 1
        cyc["placed_total"] = placed
        cyc["end_tick"] = ST["tick"]
        if nc is not None:
            cube = nc
        if placed >= count:
            break
    R["placed"] = placed
    R["attempts"] = attempts
    if ST["terminal"]:
        return

    # --- press the button
    phase("button")
    for ms in (0.15, 0.03):
        if button is not None:
            break
        bc, bb, bs, bp, bo = seg_best("front", ["button", "white button", "round button"], min_score=ms)
        if bc is not None:
            bpts = robust_filter(cloud(bc, "front"), rad=0.06)
            if bpts:
                ztop = quantile([p[2] for p in bpts], 0.9)
                cap = [p for p in bpts if p[2] > ztop - 0.006]
                button = {"x": quantile([p[0] for p in cap], 0.5), "y": quantile([p[1] for p in cap], 0.5),
                          "top": ztop}
                R["button_late"] = {"score": round(bs, 3), "box": bb, "min_score": ms,
                                    "press": [r4(button["x"]), r4(button["y"]), r4(button["top"])]}
    if button is None:
        R["notes"].append("button not found")
        return
    here = ST["cmd_x"] if ST["cmd_x"] is not None else cur_ee()[0]
    quat_b = down_quat(ST["yaw0"])
    R["button_start_tick"] = ST["tick"]
    move_cart([here[0], here[1], max(here[2] + 0.04, button["top"] + 0.08)], quat_b, 0.0, vmax=V_FAST, settle=0)
    if not move_cart([button["x"], button["y"], button["top"] + 0.06], quat_b, 0.0, vmax=V_FAST, settle=3):
        R["notes"].append("button approach ik failed")
        return
    move_cart([button["x"], button["y"], button["top"] + 0.012], quat_b, 0.0, vmax=V_CARRY, settle=2)
    phase("press_start")
    for k in range(10):
        if ST["terminal"]:
            break
        z = button["top"] + 0.012 - 0.005 * (k + 1)
        move_cart([button["x"], button["y"], z], quat_b, 0.0, vmax=0.005, settle=0)
    if not ST["terminal"]:
        hold(0.0, 10)
    phase("press_end", {"terminal": ST["terminal"]})


try:
    main()
except Exception as exc:
    log_err("main", exc)

R["clock"] = {"T": CLK["T"], "build": CLK["build"], "o": CLK["o"], "D": CLK["D"],
              "events": CLK["events"], "free_end": CLK["free_end"], "tries": CLK["tries"],
              "path": None if CLK["path"] is None else [[r4(p[0]), r4(p[1])] for p in CLK["path"]]}
R["ticks"] = ST["tick"]
R["terminal"] = ST["terminal"]
R["term_info"] = ST["term_info"]
R["step_reply_sample"] = ST["rep_sample"]
R["target_hist"] = TR["hist"][-2500:]
R["target_whist"] = TR["whist"][-800:]
try:
    fin = get_robot_state()
    R["final_ts"] = float(fin.timestamp_s)
    R["final_ee"] = [r4(v) for v in fin.end_effector_poses["primary"].position]
except Exception as exc:
    log_err("final", exc)
result = R
