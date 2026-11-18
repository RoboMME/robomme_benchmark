# MoveCube v02_narrow_open -- ASPIRE task worker, robomme_plus/robomme_plus/task_4
# v02 = v01 + pre-narrowed gripper opening before the grasp descent (UNTESTED in simulator):
#  offline replay of v01 on the run's real step-0 RGB-D found no fully-open top-down grasp of the
#  dustpan handle with >=1.5 cm clearance (ball on one side, ramp on the other). The opening is set
#  from the gripper reading seen in the conditioning video while the tool was held (+0.25), with
#  set_gripper(); if the reading does not follow, the full-open footprint is used.
# v01 (trace-guided repair of v00, episode 2 run attempt_001_25da95df; UNTESTED in simulator):
#  - tool/ball grasp chooses grasp point (along handle/shaft) and yaw by clearance of the open
#    gripper footprint from every other perceived object (v00's open finger landed on the yellow
#    ball beside the dustpan handle, stalled 3 cm above the grasp height and the episode ended);
#  - descent stall check: a grasp descent that stops >1.2 cm high is aborted and the next candidate tried;
#  - camera convention tie-break when table flatness is equal (v00 chose OpenGL; SAM fixed it);
#  - stricter hammer detection (v00 paired grey stick-shaft fragments with stray beige pixels).
# Goal (from get_task_context): watch the conditioning video, then move the cube to the
# target "in the same manner shown in the video".
# Strategy (hypothesis from the released recordings, all values perceived at runtime):
#   1. classify the manner from the conditioning video (wrist-camera colour signature of the
#      held object + which front-camera tool component vanished after the grasp);
#   2. parse the live front RGB-D image by colour into cube / target / ramp / tools;
#   3. execute: tool push (stick, dustpan/scraper, hammer), hook pull, ball-down-ramp,
#      or closed-gripper push (fallback for an unknown manner).
# Every phase is guarded: an empty or unexpected perception result is recorded and the
# program falls back to the next strategy so that control phases always run.
import math

result = {"version": "v02_narrow_open", "errors": [], "events": [], "phase_log": []}
R = result
S = {"ended": False, "base": "robot_base", "emb": "robomme_plus", "hz": 20.0,
     "max_steps": 0, "t0": 0.0, "nres": 0}


def err(where, e):
    msg = str(e)
    if len(R["errors"]) < 60:
        R["errors"].append(where + ": " + msg[:300])
    if "Episode has ended" in msg or "execution is closing" in msg:
        S["ended"] = True


def phase(name, info=None):
    entry = {"phase": name, "t": tick_now()}
    if info is not None:
        entry["info"] = info
    R["phase_log"].append(entry)


def tick_now():
    if S["ended"]:
        return -1
    try:
        st = get_robot_state()
        return int(round((float(st.timestamp_s) - S["t0"]) * S["hz"]))
    except Exception as e:
        err("tick_now", e)
        return -1


def remaining_ticks():
    t = tick_now()
    if t < 0:
        return 0
    return S["max_steps"] - t


def is_terminal(res):
    flag = False
    try:
        if res.terminated:
            flag = True
    except Exception:
        pass
    try:
        if res.truncated:
            flag = True
    except Exception:
        pass
    return flag


def rec(label, res):
    entry = {"label": label}
    try:
        entry["ok"] = bool(res.ok)
    except Exception:
        entry["ok"] = None
    try:
        entry["status"] = str(res.status)
    except Exception:
        pass
    try:
        entry["steps"] = int(res.steps_executed)
    except Exception:
        pass
    try:
        if res.error is not None:
            entry["error"] = str(res.error)[:200]
    except Exception:
        pass
    if S["nres"] < 3:
        try:
            entry["repr"] = str(res)[:600]
        except Exception:
            pass
        S["nres"] = S["nres"] + 1
    if is_terminal(res):
        entry["terminal"] = True
        S["ended"] = True
    if len(R["events"]) < 220:
        R["events"].append(entry)
    return entry


# ---------------------------------------------------------------- math helpers
def wrap(a):
    while a > math.pi:
        a = a - 2.0 * math.pi
    while a < -math.pi:
        a = a + 2.0 * math.pi
    return a


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def qmat(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        n = 1.0
    w = w / n
    x = x / n
    y = y / n
    z = z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def q_down(psi):
    # hand z-axis pointing down (-z of base); hand x-axis = (cos psi, sin psi, 0);
    # Panda fingers close along hand y = (sin psi, -cos psi, 0).
    return qmul([math.cos(psi / 2.0), 0.0, 0.0, math.sin(psi / 2.0)], [0.0, 1.0, 0.0, 0.0])


def median(vals):
    s = sorted(vals)
    if len(s) == 0:
        return 0.0
    return s[len(s) // 2]


def quant(vals, f):
    s = sorted(vals)
    if len(s) == 0:
        return 0.0
    return s[int((len(s) - 1) * f)]


def norm2(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1])


def unit2(v):
    n = norm2(v)
    if n < 1e-9:
        return [1.0, 0.0]
    return [v[0] / n, v[1] / n]


def rot2(v, a):
    c = math.cos(a)
    s = math.sin(a)
    return [c * v[0] - s * v[1], s * v[0] + c * v[1]]


def r3(p):
    return [round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)]


def seg_dist(p, a, b):
    # distance in xy from point p to segment a-b
    ab = [b[0] - a[0], b[1] - a[1]]
    ap = [p[0] - a[0], p[1] - a[1]]
    den = ab[0] * ab[0] + ab[1] * ab[1]
    t = 0.0
    if den > 1e-12:
        t = (ap[0] * ab[0] + ap[1] * ab[1]) / den
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    q = [a[0] + t * ab[0] - p[0], a[1] + t * ab[1] - p[1]]
    return norm2(q)


# ---------------------------------------------------------------- image helpers
def pixclass(r, g, b):
    if r > 150 and g < 70 and b < 70:
        return "R"
    if b > 120 and b > g + 30 and r > g + 20:
        return "U"
    if r < 120 and g > r + 25 and b > r + 25 and -12 < g - b < 12:
        return "T"
    if b > r + 35 and b - g >= 12:
        return "B"
    if r > 170 and g > 0.75 * r and b < 0.36 * r:
        return "Y"
    if r > 180 and g > 130 and g < 0.8 * r and b > 0.5 * r and b < 0.72 * r:
        return "P"
    if r > 150 and 0.78 * r <= g <= 0.92 * r and 0.5 * r <= b <= 0.75 * r:
        return "H"
    if 45 < r < 140 and g < 145 and b < 150 and g >= r - 5 and b >= g - 3 and b - r < 35 and b - g < 15:
        return "S"
    if r > 140 and g < 0.66 * r and b < 0.45 * r:
        return "W"
    return None


def to_rows(arr):
    try:
        rows = arr.tolist()
        return rows
    except Exception as e:
        err("tolist", e)
    rows = []
    h = len(arr)
    for y in range(h):
        row = arr[y]
        rows.append([row[x] for x in range(len(row))])
    return rows


def classify(rows):
    h = len(rows)
    w = len(rows[0])
    cm = [None] * (h * w)
    for y in range(h):
        row = rows[y]
        base = y * w
        for x in range(w):
            p = row[x]
            cm[base + x] = pixclass(int(p[0]), int(p[1]), int(p[2]))
    return cm, h, w


def components(cm, h, w, wanted, min_px, max_px):
    seen = [False] * (h * w)
    comps = []
    for idx in range(h * w):
        c = cm[idx]
        if c is None or seen[idx] or c not in wanted:
            continue
        stack = [idx]
        seen[idx] = True
        pix = []
        while len(stack) > 0:
            k = stack.pop()
            pix.append(k)
            x = k % w
            y = k // w
            for dy in (-1, 0, 1):
                yy = y + dy
                if yy < 0 or yy >= h:
                    continue
                for dx in (-1, 0, 1):
                    xx = x + dx
                    if xx < 0 or xx >= w:
                        continue
                    nb = yy * w + xx
                    if not seen[nb] and cm[nb] == c:
                        seen[nb] = True
                        stack.append(nb)
        if min_px <= len(pix) <= max_px:
            us = [k % w for k in pix]
            vs = [k // w for k in pix]
            comps.append({"cls": c, "pix": pix, "n": len(pix),
                          "u": sum(us) / float(len(us)), "v": sum(vs) / float(len(vs)),
                          "box": [min(us), min(vs), max(us), max(vs)]})
    return comps


def cam_model(camobs):
    k = camobs.intrinsics
    pose = camobs.camera_pose
    m = {"fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]), "cy": float(k[1][2]),
         "t": [float(v) for v in pose.position],
         "R": qmat([float(v) for v in pose.quaternion_wxyz])}
    return m


def pix_cam(m, u, v, d, conv):
    x = (u - m["cx"]) / m["fx"] * d
    y = (v - m["cy"]) / m["fy"] * d
    if conv == "opencv":
        return [x, y, d]
    if conv == "opengl":
        return [x, -y, -d]
    return [d, -x, -y]


def cam_world(m, pc):
    rm = m["R"]
    t = m["t"]
    return [rm[0][0] * pc[0] + rm[0][1] * pc[1] + rm[0][2] * pc[2] + t[0],
            rm[1][0] * pc[0] + rm[1][1] * pc[1] + rm[1][2] * pc[2] + t[1],
            rm[2][0] * pc[0] + rm[2][1] * pc[1] + rm[2][2] * pc[2] + t[2]]


def depth_at(drows, k, w):
    try:
        d = float(drows[k // w][k % w])
    except Exception:
        return -1.0
    if d != d or d < 0.05 or d > 4.0:
        return -1.0
    return d


def comp_points(m, drows, comp, w, conv, cap):
    pts = []
    pix = comp["pix"]
    stride = 1
    if len(pix) > cap:
        stride = len(pix) // cap + 1
    for i in range(0, len(pix), stride):
        k = pix[i]
        d = depth_at(drows, k, w)
        if d < 0:
            continue
        pts.append(cam_world(m, pix_cam(m, float(k % w), float(k // w), d, conv)))
    return pts


def centroid(pts):
    if len(pts) == 0:
        return None
    return [median([p[0] for p in pts]), median([p[1] for p in pts]), median([p[2] for p in pts])]


def xy_axis(pts):
    if len(pts) < 3:
        return None, 0.0
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    syy = sum((p[1] - my) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    ang = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    lam1 = 0.5 * (sxx + syy) + math.sqrt(0.25 * (sxx - syy) ** 2 + sxy * sxy)
    lam2 = 0.5 * (sxx + syy) - math.sqrt(0.25 * (sxx - syy) ** 2 + sxy * sxy)
    elong = 0.0
    if lam1 > 1e-12:
        elong = 1.0 - max(lam2, 0.0) / lam1
    return ang, elong


def max_extent(pts):
    sub = pts
    if len(sub) > 60:
        st = len(sub) // 60 + 1
        sub = [sub[i] for i in range(0, len(sub), st)]
    best = 0.0
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            dx = sub[i][0] - sub[j][0]
            dy = sub[i][1] - sub[j][1]
            dz = sub[i][2] - sub[j][2]
            dd = dx * dx + dy * dy + dz * dz
            if dd > best:
                best = dd
    return math.sqrt(best)


def comp_area_m2(m, drows, comp, w):
    # pixel footprint area, ignoring foreshortening (relative measure only)
    ds = []
    for i in range(0, len(comp["pix"]), max(1, len(comp["pix"]) // 40)):
        d = depth_at(drows, comp["pix"][i], w)
        if d > 0:
            ds.append(d)
    if len(ds) == 0:
        return 0.0
    d = median(ds)
    return comp["n"] * (d / m["fx"]) * (d / m["fy"])


# ---------------------------------------------------------------- API wrappers
def mv(pos, quat, label, tol=0.01, ms=150):
    if S["ended"]:
        return False
    try:
        res = move_to_pose(Pose([float(pos[0]), float(pos[1]), float(pos[2])],
                                [float(v) for v in quat], S["base"]),
                           tolerance=tol, max_steps=ms)
    except Exception as e:
        err("move_to_pose " + label, e)
        return False
    entry = rec(label, res)
    if not entry.get("ok") and entry.get("steps", -1) == 0 and "exceeds" in str(entry.get("error", "")) \
            and not S["ended"] and ms < 600:
        # the plan was longer than the cap and the arm did not move: safe to re-plan once
        try:
            res = move_to_pose(Pose([float(pos[0]), float(pos[1]), float(pos[2])],
                                    [float(v) for v in quat], S["base"]),
                               tolerance=tol, max_steps=int(ms * 3))
            entry = rec(label + "_replan", res)
        except Exception as e:
            err("move_to_pose " + label + "_replan", e)
            return False
    return bool(entry.get("ok"))


def ik_ok(pos, quat):
    if S["ended"]:
        return False
    try:
        res = solve_ik(Pose([float(pos[0]), float(pos[1]), float(pos[2])],
                            [float(v) for v in quat], S["base"]))
        return bool(res.ok)
    except Exception as e:
        err("solve_ik", e)
        return False


def grip(open_it, label):
    if S["ended"]:
        return None
    S["grip_cmd"] = 1.0 if open_it else 0.0
    try:
        if open_it:
            res = open_gripper()
        else:
            res = close_gripper()
    except Exception as e:
        err("gripper " + label, e)
        return None
    rec(label, res)
    return gripper_value()


def gripper_value():
    if S["ended"]:
        return -1.0
    try:
        return float(get_robot_state().gripper_positions["primary"])
    except Exception as e:
        err("gripper_value", e)
        return -1.0


def hold(n, gpos, label):
    if S["ended"]:
        return 0
    done = 0
    try:
        st = get_robot_state()
        target = [float(v) for v in st.joint_positions["primary"]]
        for k in range(n):
            res = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=target, gripper_position=gpos,
                embodiment=st.embodiment)}))
            done = done + 1
            if is_terminal(res):
                S["ended"] = True
                break
    except Exception as e:
        err("hold " + label, e)
    R["events"].append({"label": label, "hold_ticks": done})
    return done


def ee_pos():
    try:
        p = get_robot_state().end_effector_poses["primary"].position
        return [float(p[0]), float(p[1]), float(p[2])]
    except Exception as e:
        err("ee_pos", e)
        return None


# ---------------------------------------------------------------- stage A: metadata
try:
    ctx = get_task_context()
    R["goal"] = str(ctx.language)
    try:
        R["task_name"] = str(ctx.task_name)
    except Exception:
        pass
except Exception as e:
    err("context", e)

try:
    md = robomme_plus.get_task_metadata()
    mdd = {}
    for key in md:
        mdd[str(key)] = str(md[key])[:300]
    R["task_metadata"] = mdd
    try:
        S["max_steps"] = int(md["max_steps"])
    except Exception as e:
        err("max_steps", e)
    try:
        S["hz"] = float(md["control_hz"])
    except Exception:
        try:
            S["hz"] = float(md["control_frequency_hz"])
        except Exception:
            pass
except Exception as e:
    err("task_metadata", e)
if S["max_steps"] <= 0:
    S["max_steps"] = 1000
R["max_steps_used"] = S["max_steps"]

try:
    cmd = robomme_plus.get_controller_metadata()
    cdd = {}
    for key in cmd:
        cdd[str(key)] = str(cmd[key])[:300]
    R["controller_metadata"] = cdd
except Exception as e:
    err("controller_metadata", e)

home_quat = [0.0, 1.0, 0.0, 0.0]
home_pos = [0.4, 0.0, 0.3]
home_psi = 0.0
try:
    st0 = get_robot_state()
    S["base"] = str(st0.base_frame)
    S["emb"] = str(st0.embodiment)
    S["t0"] = float(st0.timestamp_s)
    hp = st0.end_effector_poses["primary"]
    home_pos = [float(v) for v in hp.position]
    home_quat = [float(v) for v in hp.quaternion_wxyz]
    hm = qmat(home_quat)
    home_psi = math.atan2(hm[1][0], hm[0][0])
    R["robot0"] = {"base": S["base"], "emb": S["emb"], "t0": S["t0"], "ee": r3(home_pos),
                   "quat": [round(v, 4) for v in home_quat],
                   "hand_z": [round(hm[0][2], 3), round(hm[1][2], 3), round(hm[2][2], 3)],
                   "psi_home": round(home_psi, 3),
                   "grip": float(st0.gripper_positions["primary"]),
                   "joints": [round(float(v), 3) for v in st0.joint_positions["primary"]]}
except Exception as e:
    err("robot_state0", e)

# ---------------------------------------------------------------- stage B: live front parse
LIVE = {"ok": False}
try:
    obs = get_observation()
    R["cameras"] = [str(c) for c in obs.cameras]
    fcam = obs.cameras["front"]
    FM = cam_model(fcam)
    R["front_model"] = {"fx": round(FM["fx"], 2), "fy": round(FM["fy"], 2), "cx": round(FM["cx"], 2),
                        "cy": round(FM["cy"], 2), "t": r3(FM["t"]),
                        "quat": [round(float(v), 4) for v in fcam.camera_pose.quaternion_wxyz],
                        "frame": str(fcam.camera_pose.frame)}
    frows = to_rows(fcam.rgb)
    fdrows = to_rows(fcam.depth_m)
    fcm, FH, FW = classify(frows)
    LIVE["cm"] = fcm
    LIVE["drows"] = fdrows
    counts = {}
    for c in fcm:
        if c is not None:
            counts[c] = counts.get(c, 0) + 1
    R["live_class_counts"] = counts
    # convention by table flatness (wood pixels)
    wood = []
    for y in range(FH // 6, FH, 6):
        for x in range(4, FW - 4, 6):
            k = y * FW + x
            if fcm[k] == "W":
                d = depth_at(fdrows, k, FW)
                if d > 0:
                    wood.append((x, y, d))
    conv_stats = {}
    best_conv = "opencv"
    best_std = 1e9
    for conv in ("opencv", "opengl", "ros"):
        zs = [cam_world(FM, pix_cam(FM, float(p[0]), float(p[1]), p[2], conv))[2] for p in wood]
        if len(zs) < 10:
            continue
        mz = sum(zs) / len(zs)
        sd = math.sqrt(sum((z - mz) ** 2 for z in zs) / len(zs))
        conv_stats[conv] = {"z_mean": round(mz, 4), "z_std": round(sd, 4), "n": len(zs)}
        if mz > home_pos[2] - 0.1:
            # the table must lie well below the gripper's start pose
            sd = sd + 0.5
        if sd < best_std:
            best_std = sd
            best_conv = conv
    S["conv"] = best_conv
    R["conv_stats"] = conv_stats
    R["conv"] = best_conv
    wz = [cam_world(FM, pix_cam(FM, float(p[0]), float(p[1]), p[2], best_conv))[2] for p in wood]
    table_z = median(wz) if len(wz) > 0 else 0.0
    S["table_z"] = table_z
    R["table_z"] = round(table_z, 4)
    LIVE["comps"] = components(fcm, FH, FW, ("R", "U", "T", "P", "S", "H", "Y", "B"), 2, 20000)
    LIVE["FW"] = FW
    LIVE["FH"] = FH
    LIVE["FM"] = FM
    LIVE["wood"] = wood
    for comp in LIVE["comps"]:
        pts = comp_points(FM, fdrows, comp, FW, best_conv, 300)
        comp["pts"] = pts
        comp["c3"] = centroid(pts)
    LIVE["ok"] = True
except Exception as e:
    err("live_parse", e)


def comp_summary(comp):
    out = {"cls": comp["cls"], "n": comp["n"], "uv": [round(comp["u"], 1), round(comp["v"], 1)],
           "box": comp["box"]}
    if comp.get("c3") is not None:
        out["c3"] = r3(comp["c3"])
    return out


def plausible(comp):
    c3 = comp.get("c3")
    if c3 is None:
        return False
    tz = S.get("table_z", 0.0)
    if c3[2] < tz - 0.04 or c3[2] > tz + 0.30:
        return False
    if norm2(c3) > 1.3:
        return False
    return True


def neg_n(c):
    return -c["n"]


def neg_first(p):
    return -p[0]


# SAM cross-check (recorded; also used as a fallback for cube / target)
SAMX = {}
for prompt in ("red cube", "purple target", "dustpan", "yellow ball"):
    try:
        seg = segment_text("front", prompt)
        info = {"ok": bool(seg.ok), "n": 0, "top": []}
        if seg.ok:
            info["n"] = len(seg.segmentations)
            for cand in seg.segmentations[:4]:
                bx = None
                if cand.box_xyxy is not None:
                    bx = [round(float(v), 1) for v in cand.box_xyxy]
                info["top"].append({"score": round(float(cand.score), 3), "box": bx})
        SAMX[prompt] = info
    except Exception as e:
        err("segment_text " + prompt, e)
R["sam_probe"] = SAMX

sam_cube = None
sam_target = None
try:
    loc = localize_object("red cube", camera_name="front", target_frame=S["base"])
    if loc.ok and len(loc.point_cloud.points) >= 10:
        gp = loc.geometry.pose.position
        sam_cube = [float(gp[0]), float(gp[1]), float(gp[2])]
        R["sam_cube"] = {"c": r3(sam_cube), "n": len(loc.point_cloud.points),
                         "extents": [round(float(v), 4) for v in loc.geometry.extents]}
except Exception as e:
    err("localize cube", e)
try:
    loc = localize_object("purple target", camera_name="front", target_frame=S["base"])
    if loc.ok and len(loc.point_cloud.points) >= 10:
        gp = loc.geometry.pose.position
        sam_target = [float(gp[0]), float(gp[1]), float(gp[2])]
        R["sam_target"] = {"c": r3(sam_target), "n": len(loc.point_cloud.points)}
except Exception as e:
    err("localize target", e)

# Cross-check the self-chosen camera convention against SAM's harness projection of the
# cube (largest red component, before any plausibility filter). Switch only on clear evidence.
if LIVE["ok"] and sam_cube is not None:
    try:
        allred = sorted([c for c in LIVE["comps"] if c["cls"] == "R"], key=neg_n)
        if len(allred) > 0:
            best = None
            agree = {}
            for conv in ("opencv", "opengl", "ros"):
                pts = comp_points(LIVE["FM"], LIVE["drows"], allred[0], LIVE["FW"], conv, 200)
                c3 = centroid(pts)
                if c3 is None:
                    continue
                dd = math.sqrt(sum([(c3[i] - sam_cube[i]) ** 2 for i in range(3)]))
                agree[conv] = round(dd, 4)
                if best is None or dd < best[0]:
                    best = (dd, conv)
            R["conv_agreement_with_sam"] = agree
            if best is not None and best[1] != S["conv"] and best[0] < 0.05 and agree.get(S["conv"], 9.0) > 0.05:
                R["conv_switched"] = [S["conv"], best[1]]
                S["conv"] = best[1]
                for comp in LIVE["comps"]:
                    comp["pts"] = comp_points(LIVE["FM"], LIVE["drows"], comp, LIVE["FW"], S["conv"], 300)
                    comp["c3"] = centroid(comp["pts"])
                wz2 = [cam_world(LIVE["FM"], pix_cam(LIVE["FM"], float(p[0]), float(p[1]), p[2], S["conv"]))[2]
                       for p in LIVE["wood"]]
                if len(wz2) > 0:
                    S["table_z"] = median(wz2)
                    R["table_z"] = round(S["table_z"], 4)
    except Exception as e:
        err("conv_check", e)

cube = None
target = None
ramp = None
if LIVE["ok"]:
    try:
        good = [c for c in LIVE["comps"] if plausible(c)]
        LIVE["good"] = good
        R["live_comps"] = [comp_summary(c) for c in sorted(good, key=neg_n)[:40]]
        reds = sorted([c for c in good if c["cls"] == "R"], key=neg_n)
        if len(reds) > 0:
            cube = reds[0]
        purples = sorted([c for c in good if c["cls"] == "U"], key=neg_n)
        if len(purples) > 0:
            target = purples[0]
        blues = sorted([c for c in good if c["cls"] == "B" and c["n"] >= 60], key=neg_n)
        if len(blues) > 0:
            ramp = blues[0]
    except Exception as e:
        err("live_select", e)
if cube is None:
    R["errors"].append("perception: no plausible red component; using SAM cube if any")
if target is None:
    R["errors"].append("perception: no plausible purple component; using SAM target if any")

cube_c = None
rc = 0.02
if cube is not None and cube.get("c3") is not None:
    cube_c = cube["c3"]
    try:
        ext = max_extent(cube["pts"])
        rc = min(0.03, max(0.012, 0.5 * ext / 1.5))
    except Exception:
        pass
elif sam_cube is not None:
    cube_c = sam_cube
target_c = None
if target is not None and target.get("c3") is not None:
    target_c = target["c3"]
elif sam_target is not None:
    target_c = sam_target
R["cube_c"] = r3(cube_c) if cube_c is not None else None
R["cube_r"] = round(rc, 4)
R["target_c"] = r3(target_c) if target_c is not None else None
if cube is not None:
    R["cube_uv"] = [round(cube["u"], 1), round(cube["v"], 1)]


# ---------------------------------------------------------------- live tool candidates
def teal_tools(good):
    handles = [c for c in good if c["cls"] == "T" and 3 <= c["n"] <= 400 and c.get("c3") is not None]
    ends = [c for c in good if c["cls"] == "P" and c["n"] >= 2 and c.get("c3") is not None]
    pairs = []
    for hi in range(len(handles)):
        h = handles[hi]
        ang, elong = xy_axis(h["pts"])
        for ei in range(len(ends)):
            e = ends[ei]
            dv = [e["c3"][0] - h["c3"][0], e["c3"][1] - h["c3"][1]]
            dist = norm2(dv)
            if dist < 0.03 or dist > 0.45:
                continue
            align = 0.5
            if ang is not None and elong > 0.3:
                align = abs(math.cos(math.atan2(dv[1], dv[0]) - ang))
            if align < 0.75:
                continue
            score = align - 0.4 * dist / 0.45 + 0.1 * min(1.0, e["n"] / 30.0)
            pairs.append((score, hi, ei, dist, align))
    pairs = sorted(pairs, key=neg_first)
    used_h = set()
    used_e = set()
    tools = []
    for p in pairs:
        if p[1] in used_h or p[2] in used_e:
            continue
        used_h.add(p[1])
        used_e.add(p[2])
        h = handles[p[1]]
        e = ends[p[2]]
        area = comp_area_m2(LIVE["FM"], LIVE["drows"], e, LIVE["FW"])
        extent = max_extent(e["pts"])
        kind = end_kind(area, extent)
        tools.append({"kind": kind, "grip": h["c3"], "grip_pts": h["pts"], "end": e["c3"], "own": [h, e],
                      "end_pts": e["pts"], "end_area": area, "end_extent": extent,
                      "dist": p[3], "align": p[4], "handle_uv": [h["u"], h["v"]],
                      "end_uv": [e["u"], e["v"]], "end_n": e["n"]})
    return tools


def end_kind(area, extent):
    if area > 0.0010 or extent > 0.065:
        return "scraper"
    if extent > 0.028:
        return "hook"
    return "stick"


def hammer_tools(good):
    heads = [c for c in good if c["cls"] == "S" and 15 <= c["n"] <= 900 and c.get("c3") is not None]
    hs = [c for c in good if c["cls"] == "H" and c["n"] >= 2 and c.get("c3") is not None]
    tools = []
    for h in heads:
        hp = []
        hcomps = []
        for c in hs:
            if norm2([c["c3"][0] - h["c3"][0], c["c3"][1] - h["c3"][1]]) < 0.35:
                hp = hp + c["pts"]
                hcomps.append(c)
        if len(hp) < 8:
            continue
        far = None
        fd = 0.0
        for p in hp:
            dd = norm2([p[0] - h["c3"][0], p[1] - h["c3"][1]])
            if dd > fd:
                fd = dd
                far = p
        if far is None or fd < 0.08:
            continue
        tools.append({"kind": "hammer", "grip": h["c3"], "grip_pts": h["pts"], "end": far, "own": [h] + hcomps,
                      "end_pts": [p for p in hp if norm2([p[0] - far[0], p[1] - far[1]]) < 0.03],
                      "len": fd, "handle_uv": [h["u"], h["v"]]})
    return tools


TOOLS = []
if LIVE["ok"]:
    try:
        TOOLS = teal_tools(LIVE["good"]) + hammer_tools(LIVE["good"])
    except Exception as e:
        err("tools", e)
R["live_tools"] = []
for tl in TOOLS:
    entry = {"kind": tl["kind"], "grip": r3(tl["grip"]), "end": r3(tl["end"])}
    for key in ("end_area", "end_extent", "dist", "align", "len"):
        if key in tl:
            entry[key] = round(float(tl[key]), 5)
    for key in ("handle_uv", "end_uv"):
        if key in tl:
            entry[key] = [round(float(tl[key][0]), 1), round(float(tl[key][1]), 1)]
    R["live_tools"].append(entry)


# ---------------------------------------------------------------- stage C: conditioning video
VID = {"n": 0}
manner = "unknown"
try:
    video = robomme_plus.get_conditioning_video()
    nv = len(video)
    VID["n"] = nv
    if nv > 0:
        VID["t_first"] = float(video[0].timestamp_s)
        VID["t_last"] = float(video[nv - 1].timestamp_s)
        nsamp = min(nv, 60)
        idxs = sorted(set([int(round(i * (nv - 1) / max(1, nsamp - 1))) for i in range(nsamp)]))
        votes = {"ball": 0, "hammer": 0, "scraper": 0, "tealshaft": 0}
        feats = []
        grips = []
        first_vote = -1
        for i in idxs:
            fr = video[i]
            try:
                grips.append(round(float(fr.robot_state.gripper_positions["primary"]), 3))
            except Exception:
                pass
            wrows = to_rows(fr.cameras["wrist"].rgb)
            wh = len(wrows)
            ww = len(wrows[0])
            low = {"Y": 0, "T": 0, "P": 0, "S": 0}
            up = {"Y": 0, "T": 0, "P": 0, "S": 0}
            nlow = 0
            nup = 0
            for y in range(0, wh, 4):
                row = wrows[y]
                for x in range(ww // 4, (3 * ww) // 4, 4):
                    p = row[x]
                    c = pixclass(int(p[0]), int(p[1]), int(p[2]))
                    if y >= (150 * wh) // 256:
                        if (96 * ww) // 256 <= x < (160 * ww) // 256:
                            nlow = nlow + 1
                            if c in low:
                                low[c] = low[c] + 1
                    else:
                        nup = nup + 1
                        if c in up:
                            up[c] = up[c] + 1
            fl = {}
            for key in low:
                fl["l" + key] = round(low[key] / float(max(1, nlow)), 3)
                fl["u" + key] = round(up[key] / float(max(1, nup)), 3)
            vote = None
            if fl["lY"] >= 0.15:
                vote = "ball"
            elif fl["lS"] >= 0.20 and fl["lT"] < 0.08:
                vote = "hammer"
            elif fl["lT"] >= 0.12:
                if fl["uP"] >= 0.12:
                    vote = "scraper"
                else:
                    vote = "tealshaft"
            if vote is not None:
                votes[vote] = votes[vote] + 1
                if first_vote < 0:
                    first_vote = i
            fl["i"] = i
            fl["vote"] = vote
            feats.append(fl)
        VID["votes"] = votes
        VID["grips"] = grips
        late = [gv for gv in grips[len(grips) // 2:] if 0.05 < gv < 0.9]
        if len(late) >= 3:
            S["open_frac"] = min(1.0, median(late) + 0.25)
        VID["open_frac"] = S.get("open_frac", 1.0)
        VID["feats"] = feats[:60]
        VID["first_vote_index"] = first_vote
        bestv = None
        for key in votes:
            if votes[key] >= 2 and (bestv is None or votes[key] > votes[bestv]):
                bestv = key
        if bestv is not None:
            manner = bestv
        # vanish analysis on front camera: which teal handle disappeared after the grasp
        try:
            f0 = video[0].cameras["front"]
            v0rows = to_rows(f0.rgb)
            v0d = to_rows(f0.depth_m)
            V0M = cam_model(f0)
            v0cm, vh, vw = classify(v0rows)
            v0comps = components(v0cm, vh, vw, ("T", "P", "S", "Y"), 3, 900)
            later = [i for i in idxs if i >= max(first_vote, (nv * 5) // 10)]
            if len(later) > 8:
                st2 = len(later) // 8 + 1
                later = [later[j] for j in range(0, len(later), st2)]
            lcms = []
            for i in later:
                lr = to_rows(video[i].cameras["front"].rgb)
                lcms.append(lr)
            pres = []
            for ci in range(len(v0comps)):
                comp = v0comps[ci]
                if comp["cls"] not in ("T", "S", "Y"):
                    continue
                sample = comp["pix"][::max(1, len(comp["pix"]) // 30)]
                tot = 0
                hit = 0
                for lr in lcms:
                    for k in sample:
                        p = lr[k // vw][k % vw]
                        tot = tot + 1
                        if pixclass(int(p[0]), int(p[1]), int(p[2])) == comp["cls"]:
                            hit = hit + 1
                frac = hit / float(max(1, tot))
                pres.append((frac, ci))
            pres = sorted(pres)
            VID["vanish"] = [{"cls": v0comps[p[1]]["cls"], "n": v0comps[p[1]]["n"],
                              "uv": [round(v0comps[p[1]]["u"], 1), round(v0comps[p[1]]["v"], 1)],
                              "presence": round(p[0], 3)} for p in pres[:8]]
            if manner == "tealshaft":
                gone = None
                for p in pres:
                    if v0comps[p[1]]["cls"] == "T" and p[0] < 0.5:
                        gone = v0comps[p[1]]
                        break
                if gone is not None:
                    gpts = comp_points(V0M, v0d, gone, vw, S.get("conv", "opencv"), 200)
                    gc = centroid(gpts)
                    ang, elong = xy_axis(gpts)
                    bestend = None
                    for e in v0comps:
                        if e["cls"] != "P":
                            continue
                        epts = comp_points(V0M, v0d, e, vw, S.get("conv", "opencv"), 200)
                        ec = centroid(epts)
                        if gc is None or ec is None:
                            continue
                        dv = [ec[0] - gc[0], ec[1] - gc[1]]
                        dist = norm2(dv)
                        if dist < 0.03 or dist > 0.45:
                            continue
                        align = 0.5
                        if ang is not None and elong > 0.3:
                            align = abs(math.cos(math.atan2(dv[1], dv[0]) - ang))
                        if align < 0.75:
                            continue
                        score = align - 0.4 * dist / 0.45
                        if bestend is None or score > bestend[0]:
                            bestend = (score, e, epts, dist)
                    if bestend is not None:
                        area = comp_area_m2(V0M, v0d, bestend[1], vw)
                        extent = max_extent(bestend[2])
                        kind = end_kind(area, extent)
                        VID["vanished_end"] = {"area": round(area, 6), "extent": round(extent, 4),
                                               "dist": round(bestend[3], 4), "kind": kind}
                        if kind == "hook":
                            manner = "hook"
                        else:
                            manner = "stick"
                if manner == "tealshaft":
                    manner = "stick"
        except Exception as e:
            err("vanish", e)
            if manner == "tealshaft":
                manner = "stick"
except Exception as e:
    err("video", e)
VID["manner"] = manner
R["video"] = VID
R["manner"] = manner


# ---------------------------------------------------------------- stage D: execution helpers
def neg_area(t):
    return -t.get("end_area", 0.0)


def neg_extent(t):
    return -t.get("end_extent", 0.0)


def pos_extent(t):
    return t.get("end_extent", 0.0)


def pick_tool(kind):
    if cube_c is None:
        return None
    cands = [t for t in TOOLS if t["kind"] == kind]
    if len(cands) == 0 and kind in ("stick", "hook"):
        cands = [t for t in TOOLS if t["kind"] in ("stick", "hook")]
    if len(cands) == 0:
        return None
    if kind == "scraper":
        cands = sorted(cands, key=neg_area)
    elif kind == "hook":
        cands = sorted(cands, key=neg_extent)
    elif kind == "stick":
        cands = sorted(cands, key=pos_extent)
    return cands[0]


def observe_cube(prev_uv):
    # re-detect the red cube in the live front image near its previous pixel location
    if S["ended"]:
        return None
    try:
        o = get_observation()
        fc = o.cameras["front"]
        m = cam_model(fc)
        rows = to_rows(fc.rgb)
        drows = to_rows(fc.depth_m)
        h = len(rows)
        w = len(rows[0])
        u0 = int(prev_uv[0])
        v0 = int(prev_uv[1])
        rad = 45
        pts = []
        us = []
        vs = []
        for y in range(max(0, v0 - rad), min(h, v0 + rad)):
            row = rows[y]
            for x in range(max(0, u0 - rad), min(w, u0 + rad)):
                p = row[x]
                if int(p[0]) > 150 and int(p[1]) < 70 and int(p[2]) < 70:
                    d = depth_at(drows, y * w + x, w)
                    if d > 0:
                        pts.append(cam_world(m, pix_cam(m, float(x), float(y), d, S["conv"])))
                        us.append(x)
                        vs.append(y)
        if len(pts) < 4:
            return None
        return {"c": centroid(pts), "uv": [sum(us) / len(us), sum(vs) / len(vs)], "n": len(pts)}
    except Exception as e:
        err("observe_cube", e)
        return None


def ramp_points():
    if ramp is None:
        return []
    pts = ramp["pts"]
    if len(pts) > 150:
        st = len(pts) // 150 + 1
        pts = [pts[i] for i in range(0, len(pts), st)]
    return pts


def path_clear(e0, e1, contact0, contact1, half_width):
    # xy clearance between the arm/tool sweep and the observed ramp footprint
    rp = ramp_points()
    if len(rp) == 0:
        return 1.0
    best = 9.0
    for p in rp:
        d1 = seg_dist(p, e0, e1) - 0.05
        d2 = seg_dist(p, e0, contact0) - half_width
        d3 = seg_dist(p, e1, contact1) - half_width
        best = min(best, d1, d2, d3)
    return best


def plan_push(v, psi_g, push_dir_sign, lead, phis, align):
    # v: contact offset from grasp point (xy) at grasp yaw psi_g.
    # returns the best (phi, delta, v_rot, e0, contact0) for pushing the cube to the target
    d = unit2([target_c[0] - cube_c[0], target_c[1] - cube_c[1]])
    dist = norm2([target_c[0] - cube_c[0], target_c[1] - cube_c[1]])
    va = math.atan2(align[1], align[0])
    da = math.atan2(d[1], d[0])
    if push_dir_sign < 0:
        da = da + math.pi
    contact0 = [cube_c[0] - d[0] * (rc + lead + 0.03), cube_c[1] - d[1] * (rc + lead + 0.03)]
    contact1 = [target_c[0] - d[0] * rc, target_c[1] - d[1] * rc]
    plans = []
    for phi in phis:
        delta = wrap(da + phi - va)
        vr = rot2(v, delta)
        e0 = [contact0[0] - vr[0], contact0[1] - vr[1]]
        e1 = [contact1[0] - vr[0], contact1[1] - vr[1]]
        clear = path_clear(e0, e1, contact0, contact1, 0.03)
        reach = norm2(e0)
        plans.append({"phi": phi, "delta": delta, "vr": vr, "e0": e0, "e1": e1,
                      "clear": clear, "reach0": reach, "reach1": norm2(e1)})
    return d, dist, contact0, plans


def choose_plan(plans, psi_g, z_push):
    tried = []
    chosen = None
    for pl in plans:
        q = q_down(psi_g + pl["delta"])
        ok0 = ik_ok([pl["e0"][0], pl["e0"][1], z_push], q)
        ok1 = ik_ok([pl["e1"][0], pl["e1"][1], z_push], q)
        tried.append({"phi": round(pl["phi"], 3), "delta": round(pl["delta"], 3),
                      "clear": round(pl["clear"], 3), "ik0": ok0, "ik1": ok1,
                      "e0": [round(pl["e0"][0], 3), round(pl["e0"][1], 3)],
                      "e1": [round(pl["e1"][0], 3), round(pl["e1"][1], 3)]})
        if ok0 and ok1 and pl["clear"] > 0.0:
            chosen = pl
            break
    if chosen is None:
        for pl in plans:
            q = q_down(psi_g + pl["delta"])
            if ik_ok([pl["e0"][0], pl["e0"][1], z_push], q):
                chosen = pl
                break
    if chosen is None and len(plans) > 0:
        chosen = plans[0]
    return chosen, tried


def narrow(open_frac, label):
    # set a partial opening; returns the measured reading (or -1)
    if S["ended"]:
        return -1.0
    S["grip_cmd"] = open_frac
    try:
        res = set_gripper(open_frac)
        rec(label, res)
    except Exception as e:
        err("set_gripper " + label, e)
        return -1.0
    return gripper_value()


def held_end_offset(tool, rel_expected):
    # find the tool's working end (P class) in the live front image after the lift; compare its
    # bearing from the gripper with the bearing measured before the grasp
    try:
        now = ee_pos()
        if now is None:
            return None
        o = get_observation()
        fc = o.cameras["front"]
        m = cam_model(fc)
        rows = to_rows(fc.rgb)
        drows = to_rows(fc.depth_m)
        cmh, hh, ww = classify(rows)
        comps = components(cmh, hh, ww, ("P",), 3, 20000)
        exp = [now[0] + rel_expected[0], now[1] + rel_expected[1]]
        best = None
        for c in comps:
            pts = comp_points(m, drows, c, ww, S["conv"], 200)
            cc = centroid(pts)
            if cc is None or cc[2] < S["table_z"] + 0.015:
                continue
            dd = norm2([cc[0] - exp[0], cc[1] - exp[1]])
            if dd > 0.12:
                continue
            if best is None or c["n"] > best[0]:
                best = (c["n"], cc, dd)
        if best is None:
            return {"found": False}
        rel = [best[1][0] - now[0], best[1][1] - now[1]]
        dth = wrap(math.atan2(rel[1], rel[0]) - math.atan2(rel_expected[1], rel_expected[0]))
        ratio = norm2(rel) / max(1e-6, norm2(rel_expected))
        use = 0.7 < ratio < 1.3 and abs(dth) < 1.0
        return {"found": True, "n": best[0], "end": r3(best[1]), "dist_from_expected": round(best[2], 4),
                "dtheta": round(dth, 3), "len_ratio": round(ratio, 3), "use": use}
    except Exception as e:
        err("held_end_offset", e)
        return None


def test_narrow(of, info):
    # one check (in place, before any approach) that a partial opening is held; else full open
    if of >= 0.99 or S["ended"]:
        return 1.0
    reading = narrow(of, "narrow_test")
    info["narrow_test"] = {"cmd": round(of, 3), "reading": round(reading, 3)}
    if reading < 0 or abs(reading - of) > 0.12:
        grip(True, "narrow_test_reopen")
        return 1.0
    return of


def grasp_at(gp, psi, gz, label, open_frac):
    # returns (gripper reading after close or None, stalled flag)
    q = q_down(psi)
    hover = S["table_z"] + 0.16
    if open_frac < 0.99:
        reading = narrow(open_frac, label + "_narrow")
        R["events"].append({"label": label + "_narrow_reading", "cmd": round(open_frac, 3), "reading": round(reading, 3)})
    else:
        grip(True, label + "_open")
    mv([gp[0], gp[1], hover], q, label + "_hover", 0.015, 200)
    mv([gp[0], gp[1], gz + 0.05], q, label + "_pre", 0.01, 120)
    mv([gp[0], gp[1], gz], q, label + "_down", 0.006, 60)
    now = ee_pos()
    if now is not None and now[2] > gz + 0.012:
        # blocked on the way down (v00: finger on a neighbouring object); back off, do not close
        R["events"].append({"label": label + "_stalled", "ee_z": round(now[2], 4), "target_z": round(gz, 4)})
        mv([gp[0], gp[1], gz + 0.06], q, label + "_backoff", 0.01, 80)
        return None, True
    g = grip(False, label + "_close")
    mv([gp[0], gp[1], gz + 0.04], q, label + "_lift", 0.01, 100)
    return g, False


def obstacle_pts(own, tool_a, tool_b):
    # 3D points of every perceived object that is not part of the tool being grasped
    pts = []
    for c in LIVE.get("good", []):
        mine = False
        for o in own:
            if c is o:
                mine = True
        if mine:
            continue
        cp = c.get("pts", [])
        st = 1
        if len(cp) > 80:
            st = len(cp) // 80 + 1
        for i in range(0, len(cp), st):
            p = cp[i]
            if p[2] < S["table_z"] + 0.010:
                continue
            if tool_a is not None and seg_dist(p, tool_a, tool_b) < 0.015 and p[2] < S["table_z"] + 0.06:
                continue
            pts.append(p)
    return pts


def gripper_clear(g, psi, gz, obs):
    return gripper_clear_open(g, psi, gz, obs, 1.0)


def gripper_clear_open(g, psi, gz, obs, open_frac):
    # xy clearance of the open Panda gripper footprint (finger span and, higher up, the palm)
    f = [math.sin(psi), -math.cos(psi)]
    hx = [math.cos(psi), math.sin(psi)]
    best = 9.0
    for p in obs:
        dx = p[0] - g[0]
        dy = p[1] - g[1]
        a = abs(dx * f[0] + dy * f[1])
        b = abs(dx * hx[0] + dy * hx[1])
        ha = 0.04 * open_frac + 0.026
        hb = 0.013
        if p[2] > gz + 0.05:
            ha = 0.11
            hb = 0.035
        da = max(a - ha, 0.0)
        db = max(b - hb, 0.0)
        if da == 0.0 and db == 0.0:
            d = -min(ha - a, hb - b)
        else:
            d = math.sqrt(da * da + db * db)
        if d < best:
            best = d
    return best


def cand_cost(c):
    return c["cost"]


def neg_clear(c):
    return -c["clear"]


def grasp_candidates(gp, u, psi_g, gz, s_lo, s_hi, obs, thetas, open_frac):
    ss = [0.0]
    for k in range(1, 9):
        for sgn in (1.0, -1.0):
            sv = sgn * 0.01 * k
            if s_lo <= sv <= s_hi:
                ss.append(sv)
    cands = []
    for sv in ss:
        g = [gp[0] + u[0] * sv, gp[1] + u[1] * sv, gp[2]]
        for th in thetas:
            psi = psi_g + th
            clr = gripper_clear_open(g, psi, gz, obs, open_frac)
            cands.append({"g": g, "psi": psi, "s": sv, "th": th, "clear": clr,
                          "cost": abs(sv) + 0.06 * abs(th)})
    ok = sorted([c for c in cands if c["clear"] >= 0.012], key=cand_cost)
    rest = sorted([c for c in cands if c["clear"] < 0.012], key=neg_clear)
    return ok + rest


def cand_summary(c):
    return {"s": round(c["s"], 3), "th": round(c["th"], 2), "clear": round(c["clear"], 4),
            "g": [round(c["g"][0], 4), round(c["g"][1], 4)]}


def grasp_yaw(axis_angle):
    best = None
    for cand in (axis_angle, axis_angle + math.pi, axis_angle - math.pi):
        dd = abs(wrap(cand - home_psi))
        if best is None or dd < best[0]:
            best = (dd, wrap(home_psi + wrap(cand - home_psi)))
    return best[1]


def closed_loop_push(vr, psi_p, z_push, d0, label, max_iter, lead):
    # advance the contact toward the target in small steps, re-aiming from the observed cube
    q = q_down(psi_p)
    log = []
    cur = ee_pos()
    if cur is None:
        return log
    contact = [cur[0] + vr[0], cur[1] + vr[1]]
    c = [cube_c[0], cube_c[1]]
    uv = R.get("cube_uv", [128.0, 128.0])
    travelled = 0.0
    budget_m = norm2([target_c[0] - cube_c[0], target_c[1] - cube_c[1]]) + rc + lead + 0.25
    stall = 0
    for it in range(max_iter):
        if S["ended"] or remaining_ticks() < 25:
            log.append({"it": it, "stop": "budget_or_end"})
            break
        ob = observe_cube(uv)
        if ob is not None:
            c = [ob["c"][0], ob["c"][1]]
            uv = ob["uv"]
        rem = norm2([target_c[0] - c[0], target_c[1] - c[1]])
        entry = {"it": it, "cube": [round(c[0], 4), round(c[1], 4)], "rem": round(rem, 4),
                 "seen": ob is not None}
        if rem < 0.012:
            entry["stop"] = "at_target"
            log.append(entry)
            break
        if travelled > budget_m:
            entry["stop"] = "travel_budget"
            log.append(entry)
            break
        d = unit2([target_c[0] - c[0], target_c[1] - c[1]])
        if d[0] * d0[0] + d[1] * d0[1] < 0.5:
            entry["stop"] = "direction_changed"
            log.append(entry)
            break
        stepm = min(0.025, rem + 0.004)
        lat = [c[0] - contact[0], c[1] - contact[1]]
        along = lat[0] * d[0] + lat[1] * d[1]
        perp = [lat[0] - along * d[0], lat[1] - along * d[1]]
        pn = norm2(perp)
        if pn > 0.012:
            perp = [perp[0] * 0.012 / pn, perp[1] * 0.012 / pn]
        nxt = [contact[0] + d[0] * stepm + perp[0], contact[1] + d[1] * stepm + perp[1]]
        ok = mv([nxt[0] - vr[0], nxt[1] - vr[1], z_push], q, label + "_step", 0.008, 60)
        now = ee_pos()
        if now is not None:
            newc = [now[0] + vr[0], now[1] + vr[1]]
            moved = norm2([newc[0] - contact[0], newc[1] - contact[1]])
            travelled = travelled + moved
            contact = newc
            entry["moved"] = round(moved, 4)
            if moved < 0.003:
                stall = stall + 1
            else:
                stall = 0
        entry["ok"] = ok
        log.append(entry)
        if stall >= 3:
            log[-1]["stop"] = "stalled"
            break
    return log


def run_tool(tool, mode):
    # mode: "push" (stick, scraper, hammer) or "pull" (hook)
    info = {"kind": tool["kind"], "mode": mode}
    gp = tool["grip"]
    ep = tool["end"]
    u = unit2([ep[0] - gp[0], ep[1] - gp[1]])
    axis_angle = math.atan2(u[1], u[0])
    psi_g = grasp_yaw(axis_angle)
    gtop = quant([p[2] for p in tool["grip_pts"]], 0.9) if len(tool["grip_pts"]) > 0 else gp[2]
    gz = max(S["table_z"] + 0.012, gtop - 0.012)
    # contact point on the tool, as offset from the grasp point
    sproj = [(p[0] - gp[0]) * u[0] + (p[1] - gp[1]) * u[1] for p in tool["end_pts"]]
    if len(sproj) == 0:
        sproj = [norm2([ep[0] - gp[0], ep[1] - gp[1]])]
    s_c = (ep[0] - gp[0]) * u[0] + (ep[1] - gp[1]) * u[1]
    s_max = max(sproj)
    s_min = min(sproj)
    lead = 0.0
    if tool["kind"] == "scraper":
        s_contact = s_min + 0.006
        lead = max(0.0, s_max - s_min)
    elif tool["kind"] == "hook":
        s_contact = s_max - 0.012
    else:
        s_contact = s_max
    off = [ep[0] - gp[0] - u[0] * s_c, ep[1] - gp[1] - u[1] * s_c]
    v = [u[0] * s_contact + off[0], u[1] * s_contact + off[1]]
    info.update({"grip": r3(gp), "end": r3(ep), "psi_g": round(psi_g, 3), "gz": round(gz, 4),
                 "v": [round(v[0], 4), round(v[1], 4)], "lead": round(lead, 4),
                 "s": [round(s_min, 4), round(s_c, 4), round(s_max, 4)]})
    # grasp point/yaw by clearance from neighbouring objects
    hproj = [(p[0] - gp[0]) * u[0] + (p[1] - gp[1]) * u[1] for p in tool["grip_pts"]]
    s_lo = min(0.0, (min(hproj) + 0.006) if len(hproj) > 0 else 0.0)
    s_hi = max(0.0, (max(hproj) - 0.006) if len(hproj) > 0 else 0.0)
    if tool["kind"] in ("stick", "scraper", "hook"):
        # the rigid shaft between handle and working end is graspable too
        s_hi = max(s_hi, min(s_min - 0.02, s_hi + 0.05))
    obs = obstacle_pts(tool.get("own", []), gp, ep)
    of = test_narrow(S.get("open_frac", 1.0), info)
    cands = grasp_candidates(gp, u, psi_g, gz, s_lo, s_hi, obs, [0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75], of)
    info["open_frac_planned"] = round(of, 3)
    info["grasp_range"] = [round(s_lo, 3), round(s_hi, 3)]
    info["n_obstacle_pts"] = len(obs)
    info["grasp_cands"] = [cand_summary(c) for c in cands[:6]]
    phase("tool_grasp", {"kind": tool["kind"]})
    g = None
    held = False
    tries = 0
    for cand in cands[:3]:
        if S["ended"] or tries >= 2 or remaining_ticks() < 300:
            break
        psi_c = cand["psi"]
        gc = cand["g"]
        if not ik_ok([gc[0], gc[1], gz], q_down(psi_c)):
            alt = wrap(psi_c + math.pi)
            if ik_ok([gc[0], gc[1], gz], q_down(alt)):
                psi_c = alt
            else:
                continue
        tries = tries + 1
        g, stalled = grasp_at(gc, psi_c, gz, "tool" + str(tries), of)
        info["try" + str(tries)] = {"cand": cand_summary(cand), "psi": round(psi_c, 3),
                                    "stalled": stalled, "grip": g}
        if stalled:
            continue
        held = g is not None and g > 0.02
        # the contact offset is measured from the grasp point actually used
        v = [v[0] - (gc[0] - gp[0]), v[1] - (gc[1] - gp[1])]
        psi_g = psi_c
        gp = gc
        info["v_used"] = [round(v[0], 4), round(v[1], 4)]
        break
    info["grip_after_close"] = g
    info["held"] = held
    if not held and not S["ended"]:
        # v00 trace: gripper reads ~0.3 when closed on a handle (conditioning video) and ~0 on air
        info["abort"] = "tool not held"
        return info
    gnow = gripper_value()
    info["grip_after_lift"] = gnow
    if tool["kind"] in ("scraper", "stick", "hook") and not S["ended"]:
        # the jaws may turn the tool while closing (skewed grasp): re-measure the working end
        meas = held_end_offset(tool, [ep[0] - gp[0], ep[1] - gp[1]])
        info["in_hand"] = meas
        if meas is not None and meas.get("use"):
            v = rot2(v, meas["dtheta"])
            info["v_in_hand"] = [round(v[0], 4), round(v[1], 4)]
    z_push = gz + 0.004
    hover = S["table_z"] + 0.17
    try:
        if ramp is not None:
            rtop = quant([p[2] for p in ramp["pts"]], 0.95)
            hover = max(hover, min(rtop + 0.08, S["table_z"] + 0.32))
    except Exception:
        pass
    info["hover"] = round(hover, 4)
    if target_c is None or cube_c is None:
        info["abort"] = "no cube/target"
        return info
    sign = 1
    if mode == "pull":
        sign = -1
    phis = [0.0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9]
    if tool["kind"] == "scraper":
        phis = [0.0, 0.2, -0.2, 0.35, -0.35]
    align = v
    if mode == "pull":
        # hook: keep the shaft parallel to the pull line so it passes beside the cube and the
        # bracket (lateral offset) closes behind it
        align = u
        info["hook_lateral_offset"] = round(norm2(off), 4)
    d0, dist0, contact0, plans = plan_push(v, psi_g, sign, lead, phis, align)
    chosen, tried = choose_plan(plans, psi_g, z_push)
    info["plans"] = tried
    if chosen is None:
        info["abort"] = "no plan"
        return info
    info["chosen_phi"] = round(chosen["phi"], 3)
    psi_p = psi_g + chosen["delta"]
    q = q_down(psi_p)
    phase("tool_transport", {"psi_p": round(psi_p, 3)})
    cur = ee_pos()
    if cur is not None:
        mv([cur[0], cur[1], hover], q_down(psi_g), "tool_up", 0.015, 150)
        mv([cur[0], cur[1], hover], q, "tool_rotate", 0.02, 200)
    mv([chosen["e0"][0], chosen["e0"][1], hover], q, "tool_above_start", 0.015, 250)
    mv([chosen["e0"][0], chosen["e0"][1], z_push + 0.04], q, "tool_pre_down", 0.01, 150)
    mv([chosen["e0"][0], chosen["e0"][1], z_push], q, "tool_down", 0.008, 120)
    phase("tool_push")
    info["push_log"] = closed_loop_push(chosen["vr"], psi_p, z_push, d0, "push", 45, lead)
    return info


def run_gripper_push():
    info = {"kind": "gripper"}
    if target_c is None or cube_c is None:
        info["abort"] = "no cube/target"
        return info
    d0 = unit2([target_c[0] - cube_c[0], target_c[1] - cube_c[1]])
    psi = grasp_yaw(math.atan2(d0[1], d0[0]) + math.pi / 2.0)
    q = q_down(psi)
    z_push = S["table_z"] + max(0.012, 0.8 * rc)
    start = [cube_c[0] - d0[0] * (rc + 0.04), cube_c[1] - d0[1] * (rc + 0.04)]
    grip(False, "gp_close")
    hover = S["table_z"] + 0.15
    mv([start[0], start[1], hover], q, "gp_hover", 0.015, 200)
    mv([start[0], start[1], z_push + 0.03], q, "gp_pre", 0.01, 120)
    mv([start[0], start[1], z_push], q, "gp_down", 0.008, 100)
    info["push_log"] = closed_loop_push([0.0, 0.0], psi, z_push, d0, "gp", 45, 0.0)
    return info


def run_ball():
    info = {"kind": "ball"}
    balls = []
    if LIVE["ok"]:
        balls = sorted([c for c in LIVE["good"] if c["cls"] == "Y" and c["n"] >= 3],
                       key=neg_n)
    bc = None
    bpts = []
    if len(balls) > 0:
        bc = balls[0]["c3"]
        bpts = balls[0]["pts"]
    if bc is None:
        try:
            loc = localize_object("yellow ball", camera_name="front", target_frame=S["base"])
            if loc.ok and len(loc.point_cloud.points) >= 5:
                gp = loc.geometry.pose.position
                bc = [float(gp[0]), float(gp[1]), float(gp[2])]
                info["ball_from_sam"] = True
        except Exception as e:
            err("localize ball", e)
    if bc is None or ramp is None:
        info["abort"] = "no ball or ramp"
        return info
    rb = 0.02
    if len(bpts) > 3:
        rb = min(0.035, max(0.012, 0.5 * max_extent(bpts)))
    rpts = ramp["pts"]
    zs = [p[2] for p in rpts]
    zhi = quant(zs, 0.92)
    zlo = quant(zs, 0.08)
    hi = [p for p in rpts if p[2] >= zhi - 0.01]
    lo = [p for p in rpts if p[2] <= zlo + 0.01]
    hic = centroid(hi)
    loc2 = centroid(lo)
    sdir = unit2([loc2[0] - hic[0], loc2[1] - hic[1]])
    rel = [hic[0] + sdir[0] * 0.03, hic[1] + sdir[1] * 0.03]
    info.update({"ball": r3(bc), "rb": round(rb, 4), "ramp_hi": r3(hic), "ramp_lo": r3(loc2),
                 "release_xy": [round(rel[0], 4), round(rel[1], 4)]})
    gz = max(S["table_z"] + 0.012, S["table_z"] + rb)
    bof = test_narrow(max(S.get("open_frac", 1.0), min(1.0, 2.0 * rb / 0.08 + 0.25)), info)
    own = []
    if len(balls) > 0:
        own = [balls[0]]
    obs = obstacle_pts(own, None, None)
    best = None
    for k in range(6):
        psi_k = wrap(home_psi + k * math.pi / 6.0)
        clr = gripper_clear_open(bc, psi_k, gz, obs, bof)
        if best is None or clr > best[0] + 0.005:
            best = (clr, psi_k)
    psi = best[1]
    info["ball_grasp_clear"] = round(best[0], 4)
    phase("ball_grasp")
    g, stalled = grasp_at(bc, psi, gz, "ball", bof)
    info["ball_stalled"] = stalled
    info["grip_after_close"] = g
    q = q_down(psi)
    hover = max(S["table_z"] + 0.16, zhi + rb + 0.10)
    cur = ee_pos()
    if cur is not None:
        mv([cur[0], cur[1], hover], q, "ball_up", 0.015, 150)
    phase("ball_transport")
    mv([rel[0], rel[1], hover], q, "ball_above_ramp", 0.015, 250)
    mv([rel[0], rel[1], zhi + rb + 0.02], q, "ball_release_pose", 0.01, 150)
    grip(True, "ball_release")
    cur = ee_pos()
    if cur is not None:
        mv([cur[0], cur[1], hover], q, "ball_retreat", 0.02, 150)
    hold(min(80, max(0, remaining_ticks() - 5)), 1.0, "ball_wait")
    return info


# ---------------------------------------------------------------- stage E: execute
EXEC = {"manner": manner}
try:
    phase("execute", {"manner": manner, "n_tools": len(TOOLS)})
    if cube_c is None or target_c is None:
        EXEC["warning"] = "cube or target not perceived; executing fallback with whatever exists"
    done = False
    if manner == "ball" and not S["ended"]:
        EXEC["ball"] = run_ball()
        done = "abort" not in EXEC["ball"]
    elif manner in ("stick", "hook", "scraper", "hammer") and not S["ended"]:
        tool = pick_tool(manner)
        if tool is None:
            EXEC["tool_missing"] = manner
        else:
            mode = "push"
            if manner == "hook":
                mode = "pull"
            EXEC["tool"] = run_tool(tool, mode)
            done = "abort" not in EXEC["tool"]
    if not done and not S["ended"] and cube_c is not None and target_c is not None:
        phase("fallback_gripper_push")
        EXEC["gripper_push"] = run_gripper_push()
except Exception as e:
    err("execute", e)
R["exec"] = EXEC

# ---------------------------------------------------------------- stage F: final observation
try:
    if not S["ended"]:
        hold(5, S.get("grip_cmd", 1.0), "settle")
    if not S["ended"]:
        ob = observe_cube(R.get("cube_uv", [128.0, 128.0]))
        if ob is not None and target_c is not None:
            R["final_cube"] = r3(ob["c"])
            R["final_err_xy"] = round(norm2([ob["c"][0] - target_c[0], ob["c"][1] - target_c[1]]), 4)
    R["final_tick"] = tick_now()
except Exception as e:
    err("final", e)
R["ended_flag"] = S["ended"]
result = R
