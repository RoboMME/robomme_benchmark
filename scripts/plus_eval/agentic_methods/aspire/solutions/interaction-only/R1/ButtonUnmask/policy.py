# v00_initial — ButtonUnmask (robomme_plus task 1), ASPIRE Stage 0 program.
#
# This single program is BOTH the scene snapshot and the attempt (R=1, one start).
# Phase 0  snapshot: language, task/controller metadata, cameras, robot state.
# Phase 1  conditioning-video summary: per-sampled-frame colour-pixel sets (front camera).
# Phase 2  SAM3 prompt probes on both cameras (no control ticks).
# Phase 3  live t0 colour set + button localisation (front, 3D via mask_to_point_cloud).
# Phase 4  press the button with a closed gripper via per-tick joint steps (deadline: step 89).
# Phase 5  return to the start joints, wait for the scene to settle, keep monitoring colours.
# Phase 6  choose the container covering the goal-colour cube: the cube's pixels that
#          VANISHED (seen earlier, not in the current front frame) must lie inside the
#          covering container's current box/mask (fixed front camera).
# Phase 7  top-down grasp of that container (wrist refinement), lift, hold.
# Every phase is wrapped in try/except; the program must never raise.

import math

report = {"version": "v00_initial", "errors": [], "notes": [], "phases": {}}
result = report

S = {
    "tick": 0, "terminal": False, "terminal_info": None, "emb": "robomme_plus",
    "base": "robot_base", "max_steps": 300, "hz": 20.0, "press_deadline": 89,
    "step_fail": 0, "monitor": False, "frames": [], "tick_counts": [],
    "front": "front", "wrist": "wrist", "colors": [], "stride": 2,
    "ik_fail": [], "W": 256, "H": 256, "home": None, "qdown": [0.0, 1.0, 0.0, 0.0],
    "last_gray": None,
}

COLOR_HUES = {
    "red": [(0.0, 15.0), (335.0, 361.0)], "orange": [(15.0, 40.0)],
    "yellow": [(40.0, 70.0)], "green": [(70.0, 170.0)], "cyan": [(170.0, 200.0)],
    "blue": [(200.0, 260.0)], "purple": [(260.0, 300.0)], "violet": [(260.0, 300.0)],
    "magenta": [(300.0, 335.0)], "pink": [(300.0, 345.0)],
}
ACHROMATIC = ("white", "black", "gray", "grey")
COVER_WORDS = ("container", "containers", "bin", "bins", "box", "boxes", "cup", "cups",
               "bowl", "bowls", "lid", "lids", "basket", "baskets", "bucket", "buckets",
               "cover", "covers", "mug", "mugs", "can", "cans", "jar", "jars")
CUBE_WORDS = ("cube", "cubes", "block", "blocks")


def say(msg):
    print("[v00 t=" + str(S["tick"]) + "] " + msg)


def err(where, exc):
    msg = where + ": " + str(exc)[:400]
    report["errors"].append(msg)
    print("[v00 ERR] " + msg)


def quantile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def fl(v, nd=4):
    try:
        return round(float(v), nd)
    except Exception:
        return None


def vec(v, nd=4):
    out = []
    try:
        for x in v:
            out.append(fl(x, nd))
    except Exception:
        return None
    return out


def qmul(a, b):
    w1, x1, y1, z1 = a[0], a[1], a[2], a[3]
    w2, x2, y2, z2 = b[0], b[1], b[2], b[3]
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def qaxes(qq):
    w, x, y, z = qq[0], qq[1], qq[2], qq[3]
    xa = [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)]
    ya = [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)]
    za = [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]
    return xa, ya, za


def hsv(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = mx - mn
    if mx <= 0:
        return 0.0, 0.0, 0.0, 0
    s = d / float(mx)
    if d == 0:
        h = 0.0
    elif mx == r:
        h = (60.0 * (g - b) / d) % 360.0
    elif mx == g:
        h = 60.0 * (b - r) / d + 120.0
    else:
        h = 60.0 * (r - g) / d + 240.0
    return h, s, mx, d


def is_color(color, r, g, b):
    h, s, v, d = hsv(r, g, b)
    if color == "white":
        return s < 0.18 and v > 185
    if color == "black":
        return v < 45
    if color == "gray" or color == "grey":
        return s < 0.15 and v > 70 and v < 185
    if color == "any":
        return s > 0.45 and v > 70 and d > 40
    ranges = COLOR_HUES.get(color)
    if ranges is None:
        return False
    smin = 0.45
    if color == "pink":
        smin = 0.30
    if s < smin or v < 60 or d < 35:
        return False
    for rng in ranges:
        if h >= rng[0] and h < rng[1]:
            return True
    return False


def rgb_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        return None


def frame_color_set(arr, color, stride):
    """Set of (u, v) grid pixels of `color` in an RGB array (pure Python, subsampled)."""
    pix = set()
    rows = rgb_rows(arr)
    try:
        H = int(arr.shape[0])
        W = int(arr.shape[1])
    except Exception:
        H = S["H"]
        W = S["W"]
    for v in range(0, H, stride):
        if rows is not None:
            row = rows[v]
        else:
            row = arr[v]
        for u in range(0, W, stride):
            px = row[u]
            if is_color(color, int(px[0]), int(px[1]), int(px[2])):
                pix.add((u, v))
    return pix


def gray_grid(arr, stride=8):
    out = []
    rows = rgb_rows(arr)
    try:
        H = int(arr.shape[0])
        W = int(arr.shape[1])
    except Exception:
        H = S["H"]
        W = S["W"]
    for v in range(0, H, stride):
        row = rows[v] if rows is not None else arr[v]
        for u in range(0, W, stride):
            px = row[u]
            out.append(int(px[0]) + int(px[1]) + int(px[2]))
    return out


def all_color_counts(arr, stride=4):
    counts = {}
    for c in ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "white", "black"):
        counts[c] = 0
    rows = rgb_rows(arr)
    try:
        H = int(arr.shape[0])
        W = int(arr.shape[1])
    except Exception:
        H = S["H"]
        W = S["W"]
    for v in range(0, H, stride):
        row = rows[v] if rows is not None else arr[v]
        for u in range(0, W, stride):
            px = row[u]
            r, g, b = int(px[0]), int(px[1]), int(px[2])
            for c in counts:
                if is_color(c, r, g, b):
                    counts[c] += 1
    return counts


def box_iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    if ua <= 0:
        return 0.0
    return inter / ua


def seg(camera, prompt, limit=8):
    out = []
    try:
        found = segment_text(camera, prompt)
    except Exception as exc:
        err("segment_text " + camera + " '" + prompt + "'", exc)
        return out
    try:
        if not found.ok:
            return out
        cands = found.segmentations
    except Exception as exc:
        err("segment_text result " + prompt, exc)
        return out
    k = 0
    for cand in cands:
        if k >= limit:
            break
        k += 1
        score = 0.0
        try:
            score = float(cand.score)
        except Exception:
            score = 0.0
        box = None
        try:
            if cand.box_xyxy is not None:
                box = [float(x) for x in cand.box_xyxy]
        except Exception:
            box = None
        out.append({"cand": cand, "score": score, "box": box, "prompt": prompt, "camera": camera})
    return out


def cand_mask_rows(c):
    if "mask_rows" in c:
        return c["mask_rows"]
    rows = None
    try:
        rows = c["cand"].mask.tolist()
    except Exception:
        rows = None
    c["mask_rows"] = rows
    return rows


def cand_cloud(c):
    if "cloud" in c:
        return c["cloud"]
    pts = []
    try:
        pc = mask_to_point_cloud(c["cand"], c["camera"], target_frame=S["base"])
        raw = pc.points
        try:
            raw = raw.tolist()
        except Exception:
            raw = [[p[0], p[1], p[2]] for p in raw]
        stepn = 1
        if len(raw) > 1500:
            stepn = len(raw) // 1500 + 1
        i = 0
        while i < len(raw):
            p = raw[i]
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                pts.append((x, y, z))
            i += stepn
    except Exception as exc:
        err("mask_to_point_cloud " + c["prompt"], exc)
    c["cloud"] = pts
    return pts


def cloud_stats(pts):
    if len(pts) < 5:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    top = quantile(zs, 0.95)
    low = quantile(zs, 0.05)
    topp = [p for p in pts if p[2] >= top - 0.015]
    if len(topp) < 3:
        topp = pts
    txs = [p[0] for p in topp]
    tys = [p[1] for p in topp]
    cx = 0.5 * (quantile(txs, 0.05) + quantile(txs, 0.95))
    cy = 0.5 * (quantile(tys, 0.05) + quantile(tys, 0.95))
    return {"n": len(pts), "cx": cx, "cy": cy, "top": top, "low": low,
            "mx": quantile(xs, 0.5), "my": quantile(ys, 0.5),
            "ext_x": quantile(xs, 0.95) - quantile(xs, 0.05),
            "ext_y": quantile(ys, 0.95) - quantile(ys, 0.05),
            "ext_z": top - low, "top_pts": [(p[0], p[1]) for p in topp]}


def stats_brief(st):
    if st is None:
        return None
    return {"n": st["n"], "center_xy": [fl(st["cx"]), fl(st["cy"])], "median_xy": [fl(st["mx"]), fl(st["my"])],
            "top_z": fl(st["top"]), "low_z": fl(st["low"]), "ext": [fl(st["ext_x"]), fl(st["ext_y"]), fl(st["ext_z"])]}


def min_area_axes(pts2):
    """Brute-force minimum-area rectangle over 5-degree yaw steps (robust 5-95% extents)."""
    if len(pts2) < 8:
        return None
    sub = pts2
    if len(sub) > 400:
        k = len(sub) // 400 + 1
        sub = sub[::k]
    best = None
    for k in range(18):
        a = math.radians(5.0 * k)
        c = math.cos(a)
        s = math.sin(a)
        us = [p[0] * c + p[1] * s for p in sub]
        vs = [-p[0] * s + p[1] * c for p in sub]
        wu = quantile(us, 0.95) - quantile(us, 0.05)
        wv = quantile(vs, 0.95) - quantile(vs, 0.05)
        area = wu * wv
        if best is None or area < best[0]:
            best = (area, a, wu, wv)
    return {"yaw": best[1], "wu": best[2], "wv": best[3]}


# ---------------------------------------------------------------- robot helpers
def cur_state():
    try:
        return get_robot_state()
    except Exception as exc:
        err("get_robot_state", exc)
        return None


def cur_joints():
    st = cur_state()
    if st is None:
        return None
    try:
        return [float(v) for v in st.joint_positions["primary"]]
    except Exception as exc:
        err("joints", exc)
        return None


def cur_ee():
    st = cur_state()
    try:
        return [float(v) for v in st.end_effector_poses["primary"].position]
    except Exception as exc:
        err("ee", exc)
        return None


def cur_grip():
    st = cur_state()
    try:
        return float(st.gripper_positions["primary"])
    except Exception:
        return None


def reply_terminal(reply):
    term = False
    info = {}
    try:
        info["terminated"] = bool(reply.terminated)
        term = term or info["terminated"]
    except Exception:
        pass
    try:
        info["truncated"] = bool(reply.truncated)
        term = term or info["truncated"]
    except Exception:
        pass
    try:
        info["done"] = bool(reply.done)
        term = term or info["done"]
    except Exception:
        pass
    return term, info


def monitor_frame(obs):
    if not S["monitor"] or len(S["colors"]) == 0:
        return
    try:
        cam = None
        if obs is not None:
            try:
                cam = obs.cameras[S["front"]]
            except Exception:
                cam = None
        if cam is None:
            cam = get_observation(camera_names=[S["front"]]).cameras[S["front"]]
        sets = {}
        counts = {}
        for color in S["colors"]:
            sset = frame_color_set(cam.rgb, color, S["stride"])
            sets[color] = sset
            counts[color] = len(sset)
        S["frames"].append({"src": "tick", "idx": S["tick"], "sets": sets})
        S["tick_counts"].append([S["tick"]] + [counts[c] for c in S["colors"]])
        S["last_gray"] = gray_grid(cam.rgb)
    except Exception as exc:
        if len(report["errors"]) < 60:
            err("monitor", exc)


def do_step(joints, grip):
    if S["terminal"]:
        return False
    if S["tick"] >= S["max_steps"] - 1:
        S["terminal"] = True
        S["terminal_info"] = "own step budget reached"
        return False
    try:
        action = RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=[float(v) for v in joints],
            gripper_position=float(grip), embodiment=S["emb"])})
        reply = step(action)
    except Exception as exc:
        err("step", exc)
        S["step_fail"] += 1
        if S["step_fail"] >= 3:
            S["terminal"] = True
            S["terminal_info"] = "repeated step exceptions"
        return False
    S["tick"] += 1
    ok = True
    try:
        ok = bool(reply.ok)
    except Exception:
        ok = True
    if ok:
        S["step_fail"] = 0
    else:
        S["step_fail"] += 1
        try:
            err("step not ok", reply.error)
        except Exception:
            err("step not ok", "unknown")
        if S["step_fail"] >= 3:
            S["terminal"] = True
            S["terminal_info"] = "repeated step failures"
    term, info = reply_terminal(reply)
    if term:
        S["terminal"] = True
        S["terminal_info"] = {"at_tick": S["tick"], "flags": info}
        say("episode terminal: " + str(info))
        return False
    obs = None
    try:
        obs = reply.observation
    except Exception:
        obs = None
    monitor_frame(obs)
    return not S["terminal"]


def hold(grip, ticks):
    q = cur_joints()
    if q is None:
        return False
    for k in range(ticks):
        if not do_step(q, grip):
            return False
    return True


def move_joints(target, grip, max_delta=0.06, settle=3, tol=0.012):
    if S["terminal"] or target is None:
        return False
    start = cur_joints()
    if start is None:
        return False
    n = len(target)
    diff = max(abs(target[j] - start[j]) for j in range(n))
    steps_n = max(1, int(math.ceil(diff / max_delta)))
    for i in range(1, steps_n + 1):
        frac = float(i) / steps_n
        qq = [start[j] + (target[j] - start[j]) * frac for j in range(n)]
        if not do_step(qq, grip):
            return False
    for k in range(settle):
        now = cur_joints()
        if now is None:
            break
        e = max(abs(target[j] - now[j]) for j in range(n))
        if e < tol:
            break
        if not do_step(target, grip):
            return False
    return True


def ik(pos, quat, seed):
    try:
        res = solve_ik(Pose(position=[float(pos[0]), float(pos[1]), float(pos[2])],
                            quaternion_wxyz=[float(v) for v in quat], frame=S["base"]),
                       seed_joints=seed)
        if res.ok:
            sol = [float(v) for v in res.joint_positions]
            if S["home"] is not None and len(sol) > len(S["home"]):
                sol = sol[:len(S["home"])]
            return sol
        try:
            S["ik_fail"].append(str(res.error)[:200])
        except Exception:
            S["ik_fail"].append("ik not ok")
    except Exception as exc:
        err("solve_ik", exc)
    return None


def move_free(pos, quat, grip, max_delta=0.07, settle=3):
    seed = cur_joints()
    sol = ik(pos, quat, seed)
    if sol is None:
        sol = ik(pos, quat, None)
    if sol is None:
        say("move_free IK failed for " + str(vec(pos)))
        return False
    return move_joints(sol, grip, max_delta=max_delta, settle=settle)


def move_line(pos, quat, grip, max_delta=0.04, seg_len=0.03, settle=3):
    ee = cur_ee()
    if ee is None:
        return False
    d = math.sqrt(sum((pos[i] - ee[i]) ** 2 for i in range(3)))
    k = max(1, int(math.ceil(d / seg_len)))
    seed = cur_joints()
    sols = []
    for i in range(1, k + 1):
        f = float(i) / k
        p = [ee[j] + (pos[j] - ee[j]) * f for j in range(3)]
        sol = ik(p, quat, seed)
        if sol is None:
            if i == k:
                sol = ik(p, quat, None)
            if sol is None:
                continue
        if seed is not None:
            jump = max(abs(sol[j] - seed[j]) for j in range(len(sol)))
            if jump > 1.0:
                report["notes"].append("large IK jump " + str(fl(jump)) + " at waypoint " + str(vec(p)))
        sols.append(sol)
        seed = sol
    if len(sols) == 0:
        say("move_line: no IK solution toward " + str(vec(pos)))
        return False
    for idx in range(len(sols)):
        last = idx == len(sols) - 1
        if not move_joints(sols[idx], grip, max_delta=max_delta, settle=(settle if last else 0)):
            return False
    return True


def exec_info(res):
    info = {}
    try:
        info["ok"] = bool(res.ok)
    except Exception:
        pass
    try:
        info["status"] = str(res.status)
    except Exception:
        pass
    try:
        info["steps_executed"] = int(res.steps_executed)
    except Exception:
        pass
    term, flags = reply_terminal(res)
    info["terminal_flags"] = flags
    try:
        if res.error is not None:
            info["error"] = str(res.error)[:300]
    except Exception:
        pass
    return info, term


def high_level(name, fn_result):
    info, term = exec_info(fn_result)
    try:
        S["tick"] += int(info.get("steps_executed", 0))
    except Exception:
        pass
    if term:
        S["terminal"] = True
        S["terminal_info"] = {"at": name, "flags": info.get("terminal_flags")}
    say(name + " -> " + str(info))
    return info


# ================================================================ Phase 0: snapshot
lang = ""
try:
    ctx = get_task_context()
    lang = str(ctx.language)
    report["task_language"] = lang
    for key in ("suite", "task_id", "task_name", "family"):
        try:
            if key == "suite":
                report["ctx_suite"] = str(ctx.suite)
            elif key == "task_id":
                report["ctx_task_id"] = str(ctx.task_id)
            elif key == "task_name":
                report["ctx_task_name"] = str(ctx.task_name)
            else:
                report["ctx_family"] = str(ctx.family)
        except Exception as exc:
            err("ctx." + key, exc)
    try:
        md = ctx.metadata
        report["ctx_metadata"] = {str(k): str(md[k])[:300] for k in md}
    except Exception as exc:
        err("ctx.metadata", exc)
    say("language: " + lang)
except Exception as exc:
    err("get_task_context", exc)

try:
    tmeta = robomme_plus.get_task_metadata()
    report["task_metadata"] = {str(k): str(tmeta[k])[:400] for k in tmeta}
    try:
        S["max_steps"] = int(tmeta["max_steps"])
    except Exception as exc:
        err("max_steps", exc)
    for key in ("control_hz", "control_frequency", "control_freq", "frequency_hz", "hz"):
        try:
            if key in tmeta:
                S["hz"] = float(tmeta[key])
                break
        except Exception:
            pass
except Exception as exc:
    err("get_task_metadata", exc)

try:
    cmeta = robomme_plus.get_controller_metadata()
    report["controller_metadata"] = {str(k): str(cmeta[k])[:400] for k in cmeta}
except Exception as exc:
    err("get_controller_metadata", exc)

report["max_steps"] = S["max_steps"]
report["control_hz"] = S["hz"]

obs0 = None
try:
    obs0 = get_observation()
    cams = {}
    names = []
    for name in obs0.cameras:
        names.append(str(name))
        cam = obs0.cameras[name]
        info = {}
        try:
            info["rgb_shape"] = [int(v) for v in cam.rgb.shape]
        except Exception as exc:
            err("rgb shape " + str(name), exc)
        try:
            info["depth_shape"] = [int(v) for v in cam.depth_m.shape]
        except Exception as exc:
            err("depth shape " + str(name), exc)
        try:
            info["intrinsics"] = [vec(row) for row in cam.intrinsics]
        except Exception as exc:
            err("intrinsics " + str(name), exc)
        try:
            info["pose_position"] = vec(cam.camera_pose.position)
            info["pose_quat_wxyz"] = vec(cam.camera_pose.quaternion_wxyz)
            info["pose_frame"] = str(cam.camera_pose.frame)
        except Exception as exc:
            err("camera_pose " + str(name), exc)
        try:
            info["frame"] = str(cam.frame)
            info["timestamp_s"] = fl(cam.timestamp_s)
        except Exception:
            pass
        cams[str(name)] = info
    report["cameras"] = cams
    report["camera_names"] = names
    if "front" not in names and len(names) > 0:
        S["front"] = names[0]
    if "wrist" not in names:
        for nm in names:
            if nm != S["front"]:
                S["wrist"] = nm
    try:
        S["H"] = int(obs0.cameras[S["front"]].rgb.shape[0])
        S["W"] = int(obs0.cameras[S["front"]].rgb.shape[1])
    except Exception:
        pass
except Exception as exc:
    err("get_observation", exc)

try:
    st0 = get_robot_state()
    S["base"] = str(st0.base_frame)
    try:
        S["emb"] = str(st0.embodiment)
    except Exception as exc:
        err("embodiment", exc)
    S["home"] = [float(v) for v in st0.joint_positions["primary"]]
    eep = st0.end_effector_poses["primary"]
    q0 = [float(v) for v in eep.quaternion_wxyz]
    xa, ya, za = qaxes(q0)
    if za[2] < -0.85:
        S["qdown"] = q0
        report["notes"].append("using start EE orientation as top-down (z-axis " + str(vec(za, 3)) + ")")
    else:
        report["notes"].append("start EE not pointing down (z-axis " + str(vec(za, 3)) + "); using [0,1,0,0]")
    report["robot_state0"] = {
        "base_frame": S["base"], "embodiment": S["emb"], "joints": vec(S["home"]),
        "gripper": fl(st0.gripper_positions["primary"]),
        "ee_position": vec(eep.position), "ee_quat_wxyz": vec(q0), "ee_frame": str(eep.frame),
        "ee_axes": {"x": vec(xa, 3), "y": vec(ya, 3), "z": vec(za, 3)},
    }
    try:
        report["robot_state0"]["timestamp_s"] = fl(st0.timestamp_s)
    except Exception:
        pass
    try:
        report["robot_state0"]["joint_names"] = [str(v) for v in st0.joint_names["primary"]]
    except Exception:
        pass
except Exception as exc:
    err("robot state", exc)

# ---------------------------------------------------------------- language parsing
tokens = []
try:
    low = lang.lower()
    for ch in ",.;:!?()'\"":
        low = low.replace(ch, " ")
    tokens = low.split()
except Exception as exc:
    err("tokenize", exc)

cube_colors = []
container_colors = []
button_colors = []
cover_noun = None
for i in range(len(tokens)):
    t = tokens[i]
    if cover_noun is None and t in COVER_WORDS:
        cover_noun = t
    if t in COLOR_HUES or t in ACHROMATIC:
        noun = None
        for j in range(i + 1, min(i + 4, len(tokens))):
            if tokens[j] in COVER_WORDS or tokens[j] in CUBE_WORDS or tokens[j] == "button":
                noun = tokens[j]
                break
        if noun == "button":
            button_colors.append(t)
        elif noun is not None and noun in COVER_WORDS:
            container_colors.append(t)
        else:
            cube_colors.append(t)
lift_cube = False
if cover_noun is None and any(t in CUBE_WORDS for t in tokens):
    lift_cube = True
report["parsed_goal"] = {"tokens": tokens, "cube_colors": cube_colors, "container_colors": container_colors,
                         "button_colors": button_colors, "cover_noun": cover_noun,
                         "mentions_button": "button" in tokens, "pick_cube_variant": lift_cube}
monitor_colors = []
for c in cube_colors:
    if c not in monitor_colors:
        monitor_colors.append(c)
if len(monitor_colors) == 0:
    monitor_colors.append("any")
S["colors"] = monitor_colors
say("parsed goal: " + str(report["parsed_goal"]))

cover_prompts = []
if cover_noun is not None:
    singular = cover_noun
    if singular.endswith("es") and singular[:-2] in ("box",):
        singular = singular[:-2]
    elif singular.endswith("s"):
        singular = singular[:-1]
    cover_prompts.append(singular)
for p in ("container", "bin", "box", "cup"):
    if p not in cover_prompts:
        cover_prompts.append(p)
button_prompts = []
for c in button_colors:
    button_prompts.append(c + " button")
for p in ("button", "push button", "round button"):
    if p not in button_prompts:
        button_prompts.append(p)
report["prompts"] = {"cover": cover_prompts, "button": button_prompts}

# ================================================================ Phase 1: conditioning video
try:
    video = robomme_plus.get_conditioning_video()
    nvid = len(video)
    vinfo = {"frames": nvid}
    if nvid > 0:
        idxs = []
        if nvid <= 64:
            idxs = list(range(nvid))
        else:
            for k in range(48):
                idxs.append(int(round(k * (nvid - 1) / 47.0)))
            for k in range(16):
                idxs.append(nvid - 16 + k)
        seen = set()
        uniq = []
        for i in idxs:
            if i not in seen and i >= 0 and i < nvid:
                seen.add(i)
                uniq.append(i)
        vinfo["sampled"] = len(uniq)
        try:
            vinfo["t_first"] = fl(video[0].timestamp_s)
            vinfo["t_last"] = fl(video[nvid - 1].timestamp_s)
        except Exception as exc:
            err("video timestamps", exc)
        per = []
        live_pose = None
        try:
            live_pose = vec(obs0.cameras[S["front"]].camera_pose.position)
        except Exception:
            live_pose = None
        pose_shift = 0.0
        for n_i in range(len(uniq)):
            i = uniq[n_i]
            fr = video[i]
            cam = fr.cameras[S["front"]]
            sets = {}
            counts = []
            for color in S["colors"]:
                sset = frame_color_set(cam.rgb, color, S["stride"])
                sets[color] = sset
                counts.append(len(sset))
            S["frames"].append({"src": "video", "idx": i, "sets": sets})
            row = {"i": i, "t": fl(fr.timestamp_s, 3), "target_counts": counts}
            if n_i % 8 == 0 or n_i == len(uniq) - 1:
                row["all_colors_stride4"] = all_color_counts(cam.rgb)
                try:
                    vp = vec(cam.camera_pose.position)
                    row["front_pose"] = vp
                    if live_pose is not None:
                        dd = math.sqrt(sum((vp[j] - live_pose[j]) ** 2 for j in range(3)))
                        pose_shift = max(pose_shift, dd)
                except Exception:
                    pass
            per.append(row)
        vinfo["per_frame"] = per
        vinfo["max_front_pose_shift_vs_live_m"] = fl(pose_shift)
        try:
            ws = video[0].cameras[S["wrist"]].rgb.shape
            vinfo["wrist_shape"] = [int(v) for v in ws]
        except Exception:
            pass
    report["conditioning_video"] = vinfo
    say("conditioning video frames: " + str(nvid))
except Exception as exc:
    err("conditioning video", exc)

# ================================================================ Phase 2: SAM probes (both cameras)
probes = []
probe_prompts = []
for p in button_prompts[:2] + cover_prompts[:3] + ["cube"]:
    if p not in probe_prompts:
        probe_prompts.append(p)
for c in cube_colors:
    if (c + " cube") not in probe_prompts:
        probe_prompts.append(c + " cube")
for camname in (S["front"], S["wrist"]):
    for p in probe_prompts:
        found = seg(camname, p, 5)
        probes.append({"camera": camname, "prompt": p, "n": len(found),
                       "scores": [fl(c["score"], 3) for c in found],
                       "boxes": [vec(c["box"], 1) if c["box"] is not None else None for c in found]})
report["prompt_probes"] = probes

# ================================================================ Phase 3: live t0 colours + button
try:
    if obs0 is not None:
        cam = obs0.cameras[S["front"]]
        sets = {}
        for color in S["colors"]:
            sets[color] = frame_color_set(cam.rgb, color, S["stride"])
        S["frames"].append({"src": "live0", "idx": 0, "sets": sets})
        report["live0_counts"] = {c: len(sets[c]) for c in sets}
        report["live0_all_colors_stride4"] = all_color_counts(cam.rgb)
        S["last_gray"] = gray_grid(cam.rgb)
except Exception as exc:
    err("live0 colours", exc)

button = None
button_box = None
try:
    choices = []
    for p in button_prompts:
        cands = seg(S["front"], p, 4)
        for c in cands:
            if c["box"] is None:
                continue
            st = cloud_stats(cand_cloud(c))
            if st is None:
                continue
            plausible = st["ext_x"] < 0.15 and st["ext_y"] < 0.15 and st["ext_z"] < 0.12
            choices.append({"prompt": p, "score": fl(c["score"], 3), "box": vec(c["box"], 1),
                            "stats": stats_brief(st), "plausible": plausible})
            if plausible and button is None:
                button = st
                button_box = c["box"]
        if button is not None:
            break
    report["button_candidates"] = choices
    if button is not None:
        report["button_front"] = stats_brief(button)
        say("button at " + str(stats_brief(button)))
    else:
        say("no button localised")
except Exception as exc:
    err("button localisation", exc)

# ================================================================ Phase 4: press the button
press = {"attempted": False}
report["phases"]["press"] = press
S["monitor"] = True
try:
    if button is not None and S["home"] is not None:
        press["attempted"] = True
        bx, by, btop = button["cx"], button["cy"], button["top"]
        hover = [bx, by, btop + 0.08]
        ok = move_free(hover, S["qdown"], 0.0, max_delta=0.07, settle=3)
        press["hover_ok"] = ok
        press["tick_at_hover"] = S["tick"]
        # wrist refinement (no control ticks)
        try:
            best = None
            for p in button_prompts[:2]:
                for c in seg(S["wrist"], p, 4):
                    st = cloud_stats(cand_cloud(c))
                    if st is None:
                        continue
                    d = math.sqrt((st["cx"] - bx) ** 2 + (st["cy"] - by) ** 2)
                    if d < 0.04 and abs(st["top"] - btop) < 0.04 and st["ext_x"] < 0.15 and st["ext_y"] < 0.15:
                        if best is None or d < best[0]:
                            best = (d, st)
                if best is not None:
                    break
            if best is not None:
                press["wrist_button"] = stats_brief(best[1])
                bx = 0.5 * (bx + best[1]["cx"])
                by = 0.5 * (by + best[1]["cy"])
                btop = max(btop, best[1]["top"])
        except Exception as exc:
            err("wrist button refine", exc)
        press["target_xy"] = [fl(bx), fl(by)]
        press["top_z"] = fl(btop)
        table_guess = button["low"]
        z1 = max(btop - 0.008, table_guess - 0.005)
        z2 = max(btop - 0.02, table_guess - 0.005)
        move_line([bx, by, btop + 0.03], S["qdown"], 0.0, max_delta=0.06, seg_len=0.05, settle=0)
        move_line([bx, by, z1], S["qdown"], 0.0, max_delta=0.035, seg_len=0.02, settle=2)
        press["ee_at_z1"] = vec(cur_ee())
        press["tick_at_z1"] = S["tick"]
        move_line([bx, by, z2], S["qdown"], 0.0, max_delta=0.03, seg_len=0.02, settle=3)
        press["ee_at_z2"] = vec(cur_ee())
        press["tick_at_z2"] = S["tick"]
        try:
            press["state_ts_at_z2"] = fl(get_robot_state().timestamp_s, 3)
        except Exception:
            pass
        press["z_targets"] = [fl(z1), fl(z2)]
        move_line([bx, by, btop + 0.07], S["qdown"], 0.0, max_delta=0.06, seg_len=0.05, settle=0)
        press["tick_after_rise"] = S["tick"]
        press["before_deadline"] = press["tick_at_z2"] <= S["press_deadline"]
        say("press done: " + str(press))
except Exception as exc:
    err("press phase", exc)

# ================================================================ Phase 5: clear the view, settle
settle_info = {}
report["phases"]["settle"] = settle_info
try:
    if not S["terminal"] and S["home"] is not None:
        move_joints(S["home"], 1.0, max_delta=0.07, settle=4)
        settle_info["tick_home"] = S["tick"]
        quiet = 0
        waited = 0
        prev = S["last_gray"]
        while waited < 40 and not S["terminal"]:
            if not do_step(S["home"], 1.0):
                break
            waited += 1
            cur = S["last_gray"]
            if prev is not None and cur is not None and len(prev) == len(cur):
                diff = sum(abs(cur[i] - prev[i]) for i in range(len(cur))) / float(len(cur))
            else:
                diff = 999.0
            prev = cur
            if diff < 3.0:
                quiet += 1
            else:
                quiet = 0
            if quiet >= 3 and waited >= 4:
                break
        settle_info["waited_ticks"] = waited
        # If a goal-colour cube is still as visible as it ever was, the covering event has
        # not happened yet: keep holding (bounded) until its pixel count collapses.
        extra = 0
        still = True
        while still and not S["terminal"] and extra < 60 and S["tick"] < S["max_steps"] - 150:
            still = False
            if len(S["frames"]) == 0:
                break
            lastf = S["frames"][-1]
            for color in S["colors"]:
                mx = 0
                for fr in S["frames"]:
                    sset = fr["sets"].get(color)
                    if sset is not None and len(sset) > mx:
                        mx = len(sset)
                nowc = len(lastf["sets"].get(color, set()))
                if mx >= 6 and nowc >= 0.5 * mx:
                    still = True
            if still:
                if not do_step(S["home"], 1.0):
                    break
                extra += 1
        settle_info["extra_wait_for_cover_ticks"] = extra
        settle_info["target_still_visible"] = still
        settle_info["tick_ready"] = S["tick"]
except Exception as exc:
    err("settle phase", exc)
S["monitor"] = False


# ================================================================ Phase 6: choose container(s)
def clusters_of(pix, stride):
    """Connected components on the subsampled grid (8-neighbourhood)."""
    remaining = set(pix)
    comps = []
    while len(remaining) > 0:
        seed = None
        for p in remaining:
            seed = p
            break
        remaining.discard(seed)
        stack = [seed]
        comp = [seed]
        while len(stack) > 0:
            u, v = stack.pop()
            for du in (-stride, 0, stride):
                for dv in (-stride, 0, stride):
                    nb = (u + du, v + dv)
                    if nb in remaining:
                        remaining.discard(nb)
                        stack.append(nb)
                        comp.append(nb)
        comps.append(comp)
    return comps


def inside_box(p, box, pad=2.0):
    return box[0] - pad <= p[0] <= box[2] + pad and box[1] - pad <= p[1] <= box[3] + pad


def collect_covers():
    out = []
    for p in cover_prompts:
        for c in seg(S["front"], p, 8):
            b = c["box"]
            if b is None:
                continue
            area = (b[2] - b[0]) * (b[3] - b[1])
            if area < 20 or area > S["W"] * S["H"] / 5.0:
                continue
            if button_box is not None and box_iou(b, button_box) > 0.3:
                continue
            dup = False
            for e in out:
                if box_iou(e["box"], b) > 0.5:
                    dup = True
                    break
            if not dup:
                out.append(c)
    keep = []
    for c in out:
        b = c["box"]
        inner = 0
        for e in out:
            if e is c:
                continue
            eu = 0.5 * (e["box"][0] + e["box"][2])
            ev = 0.5 * (e["box"][1] + e["box"][3])
            if b[0] < eu < b[2] and b[1] < ev < b[3]:
                inner += 1
        if inner >= 2:
            report["notes"].append("dropped group box " + str(vec(b, 1)) + " (" + c["prompt"] + ")")
            continue
        keep.append(c)
    return keep


decisions = []
report["decisions"] = decisions
covers = []
ref_sets = {}
try:
    ref_obs = get_observation()
    rcam = ref_obs.cameras[S["front"]]
    for color in S["colors"]:
        ref_sets[color] = frame_color_set(rcam.rgb, color, S["stride"])
    report["ref_counts"] = {c: len(ref_sets[c]) for c in ref_sets}
    report["ref_all_colors_stride4"] = all_color_counts(rcam.rgb)
    covers = collect_covers()
    cov_rep = []
    for c in covers:
        st = cloud_stats(cand_cloud(c))
        c["stats"] = st
        cov_rep.append({"prompt": c["prompt"], "score": fl(c["score"], 3), "box": vec(c["box"], 1),
                        "has_mask": cand_mask_rows(c) is not None, "stats": stats_brief(st)})
    report["cover_candidates"] = cov_rep
    say("cover candidates: " + str(len(covers)))
except Exception as exc:
    err("cover collection", exc)

targets = []
try:
    goal_list = list(cube_colors)
    if len(goal_list) == 0:
        goal_list = ["any"]
    used = set()
    for color in goal_list:
        if color not in S["colors"]:
            continue
        dec = {"color": color}
        ref = ref_sets.get(color, set())
        best_frame = None
        best_trans = None
        for fr in S["frames"]:
            sset = fr["sets"].get(color)
            if sset is None:
                continue
            trans = sset - ref
            if button_box is not None:
                trans = set(p for p in trans if not inside_box(p, button_box, 3.0))
            if best_trans is None or len(trans) > len(best_trans):
                best_trans = trans
                best_frame = fr
        if best_frame is not None:
            dec["best_frame"] = {"src": best_frame["src"], "idx": best_frame["idx"],
                                 "transient_px": len(best_trans)}
        evidence = None
        if best_trans is not None and len(best_trans) >= 3:
            comps = clusters_of(best_trans, S["stride"])
            big = None
            for comp in comps:
                if big is None or len(comp) > len(big):
                    big = comp
            evidence = big
            dec["evidence_kind"] = "vanished_pixels"
        else:
            # fallback: target-coloured pixels anywhere in the max-count frame, minus the button box
            maxf = None
            maxset = None
            for fr in S["frames"]:
                sset = fr["sets"].get(color)
                if sset is None:
                    continue
                if button_box is not None:
                    sset = set(p for p in sset if not inside_box(p, button_box, 3.0))
                if maxset is None or len(sset) > len(maxset):
                    maxset = sset
                    maxf = fr
            if maxset is not None and len(maxset) >= 3:
                comps = clusters_of(maxset, S["stride"])
                big = None
                for comp in comps:
                    if big is None or len(comp) > len(big):
                        big = comp
                evidence = big
                dec["evidence_kind"] = "visible_pixels"
                dec["visible_frame"] = {"src": maxf["src"], "idx": maxf["idx"]}
        chosen = None
        scores = []
        if evidence is not None and len(evidence) > 0:
            cu = sum(p[0] for p in evidence) / float(len(evidence))
            cv = sum(p[1] for p in evidence) / float(len(evidence))
            dec["evidence_centroid_px"] = [fl(cu, 1), fl(cv, 1)]
            dec["evidence_n"] = len(evidence)
            for ci in range(len(covers)):
                c = covers[ci]
                b = c["box"]
                mrows = cand_mask_rows(c)
                n_in = 0
                for p in evidence:
                    if mrows is not None:
                        try:
                            if mrows[p[1]][p[0]]:
                                n_in += 1
                            continue
                        except Exception:
                            pass
                    if inside_box(p, b):
                        n_in += 1
                bcu = 0.5 * (b[0] + b[2])
                bcv = 0.5 * (b[1] + b[3])
                dist = math.sqrt((bcu - cu) ** 2 + (bcv - cv) ** 2)
                scores.append({"i": ci, "inside": n_in, "dist_px": fl(dist, 1)})
                key = (n_in, -dist)
                if ci in used:
                    continue
                if chosen is None or key > chosen[0]:
                    chosen = (key, ci)
        elif len(covers) > 0:
            dec["evidence_kind"] = "none: fell back to first unused cover candidate"
            for ci in range(len(covers)):
                if ci not in used:
                    chosen = ((0, 0.0), ci)
                    break
        dec["cover_scores"] = scores
        if chosen is not None:
            used.add(chosen[1])
            dec["chosen_cover"] = chosen[1]
            targets.append(covers[chosen[1]])
        decisions.append(dec)
        say("decision " + str(dec))
    if len(container_colors) > 0 and len(targets) == 0:
        # instruction names the container colour directly
        for color in container_colors:
            best = None
            for ci in range(len(covers)):
                b = covers[ci]["box"]
                n = 0
                for p in ref_sets.get(color, set()):
                    if inside_box(p, b, 0.0):
                        n += 1
                if best is None or n > best[0]:
                    best = (n, ci)
            if best is not None:
                targets.append(covers[best[1]])
                decisions.append({"container_color": color, "chosen_cover": best[1], "n": best[0]})
except Exception as exc:
    err("decision", exc)


# ================================================================ Phase 7: grasp and lift
def refine_with_wrist(cx, cy, top):
    best = None
    for p in cover_prompts[:3]:
        for c in seg(S["wrist"], p, 6):
            st = cloud_stats(cand_cloud(c))
            if st is None:
                continue
            d = math.sqrt((st["cx"] - cx) ** 2 + (st["cy"] - cy) ** 2)
            if d < 0.05 and abs(st["top"] - top) < 0.05 and st["ext_x"] < 0.2 and st["ext_y"] < 0.2:
                if best is None or d < best[0]:
                    best = (d, st, p)
        if best is not None:
            break
    return best


def grasp_lift(cover, label, keep_lifted):
    g = {"label": label}
    report["phases"][label] = g
    st = cover.get("stats")
    if st is None:
        st = cloud_stats(cand_cloud(cover))
    if st is None:
        g["error"] = "no cloud for chosen cover"
        return False
    g["front_stats"] = stats_brief(st)
    cx, cy, top, lowz = st["cx"], st["cy"], st["top"], st["low"]
    top_pts = st["top_pts"]
    if not move_free([cx, cy, top + 0.12], S["qdown"], 1.0, max_delta=0.07, settle=3):
        g["hover_failed"] = True
    g["tick_hover"] = S["tick"]
    wr = None
    try:
        wr = refine_with_wrist(cx, cy, top)
    except Exception as exc:
        err("wrist refine", exc)
    if wr is not None:
        g["wrist_stats"] = stats_brief(wr[1])
        g["wrist_prompt"] = wr[2]
        cx, cy = wr[1]["cx"], wr[1]["cy"]
        top = wr[1]["top"]
        top_pts = wr[1]["top_pts"]
    quat = S["qdown"]
    try:
        rect = min_area_axes(top_pts)
        if rect is not None:
            axis = rect["yaw"]
            narrow = rect["wu"]
            if rect["wv"] < rect["wu"]:
                axis = rect["yaw"] + math.pi / 2.0
                narrow = rect["wv"]
            xa, ya, za = qaxes(S["qdown"])
            phi = math.atan2(ya[1], ya[0])
            delta = axis - phi
            while delta > math.pi / 2.0:
                delta -= math.pi
            while delta <= -math.pi / 2.0:
                delta += math.pi
            quat = qmul([math.cos(delta / 2.0), 0.0, 0.0, math.sin(delta / 2.0)], S["qdown"])
            g["rect"] = {"yaw_deg": fl(math.degrees(rect["yaw"]), 1), "wu": fl(rect["wu"]), "wv": fl(rect["wv"]),
                         "narrow": fl(narrow), "rotate_deg": fl(math.degrees(delta), 1)}
    except Exception as exc:
        err("yaw estimate", exc)
    height = max(0.01, top - lowz)
    gz = max(top - min(0.025, 0.5 * height), lowz + 0.012)
    g["grasp_xyz"] = [fl(cx), fl(cy), fl(gz)]
    options = [quat, qmul([0.0, 0.0, 0.0, 1.0], quat), S["qdown"]]
    chosen_q = None
    for oi in range(len(options)):
        if ik([cx, cy, gz], options[oi], cur_joints()) is not None:
            chosen_q = options[oi]
            g["orientation_option"] = oi
            break
    if chosen_q is not None:
        quat = chosen_q
    else:
        g["orientation_option"] = "none reachable; using yaw-adjusted"
    g["top_z"] = fl(top)
    move_line([cx, cy, top + 0.10], quat, 1.0, max_delta=0.06, seg_len=0.05, settle=2)
    move_line([cx, cy, gz], quat, 1.0, max_delta=0.035, seg_len=0.025, settle=4)
    g["ee_at_grasp"] = vec(cur_ee())
    g["tick_grasp"] = S["tick"]
    if S["terminal"]:
        return False
    try:
        g["close"] = high_level("close_gripper", close_gripper())
    except Exception as exc:
        err("close_gripper", exc)
    g["grip_after_close"] = fl(cur_grip())
    move_line([cx, cy, gz + 0.15], quat, 0.0, max_delta=0.03, seg_len=0.03, settle=3)
    g["ee_after_lift"] = vec(cur_ee())
    g["grip_after_lift"] = fl(cur_grip())
    g["tick_lift"] = S["tick"]
    if not S["terminal"]:
        hold(0.0, 6)
    try:
        lifted = None
        for c in collect_covers():
            s2 = cloud_stats(cand_cloud(c))
            if s2 is None:
                continue
            d = math.sqrt((s2["cx"] - cx) ** 2 + (s2["cy"] - cy) ** 2)
            if d < 0.06:
                if lifted is None or s2["top"] > lifted:
                    lifted = s2["top"]
        g["front_top_after_lift"] = fl(lifted)
        g["lift_seen"] = lifted is not None and lifted > top + 0.06
    except Exception as exc:
        err("lift check", exc)
    if not keep_lifted and not S["terminal"]:
        move_line([cx, cy, gz + 0.005], quat, 0.0, max_delta=0.03, seg_len=0.03, settle=2)
        try:
            high_level("open_gripper", open_gripper())
        except Exception as exc:
            err("open_gripper", exc)
        move_line([cx, cy, top + 0.12], quat, 1.0, max_delta=0.06, seg_len=0.05, settle=1)
    return True


try:
    report["n_targets"] = len(targets)
    for ti in range(len(targets)):
        if S["terminal"]:
            break
        last = ti == len(targets) - 1
        grasp_lift(targets[ti], "grasp_" + str(ti), keep_lifted=(last and not lift_cube))
        g = report["phases"].get("grasp_" + str(ti), {})
        gp = g.get("grip_after_close")
        if not S["terminal"] and gp is not None and gp < 0.02 and S["tick"] < S["max_steps"] - 80:
            report["notes"].append("grasp " + str(ti) + " closed on nothing; one retry")
            try:
                high_level("open_gripper", open_gripper())
            except Exception as exc:
                err("open_gripper retry", exc)
            covers_now = collect_covers()
            best = None
            for c in covers_now:
                s2 = cloud_stats(cand_cloud(c))
                c["stats"] = s2
                if s2 is None:
                    continue
                d = math.sqrt((s2["cx"] - g["front_stats"]["center_xy"][0]) ** 2 +
                              (s2["cy"] - g["front_stats"]["center_xy"][1]) ** 2)
                if best is None or d < best[0]:
                    best = (d, c)
            if best is not None and best[0] < 0.08:
                grasp_lift(best[1], "grasp_" + str(ti) + "_retry", keep_lifted=(last and not lift_cube))
    if lift_cube and not S["terminal"] and len(cube_colors) > 0:
        report["notes"].append("pick-cube variant: container lifted aside not implemented beyond lift; trying cube pick")
    if not S["terminal"]:
        hold(0.0, 10)
except Exception as exc:
    err("grasp phase", exc)

# ================================================================ wrap up
try:
    report["final"] = {"tick": S["tick"], "terminal": S["terminal"], "terminal_info": S["terminal_info"],
                       "ee": vec(cur_ee()), "gripper": fl(cur_grip()), "ik_failures": S["ik_fail"][:20]}
    tc = S["tick_counts"]
    if len(tc) > 120:
        k = len(tc) // 120 + 1
        tc = tc[::k]
    report["tick_color_counts"] = {"columns": ["tick"] + S["colors"], "rows": tc}
except Exception as exc:
    err("final report", exc)
say("done; ticks=" + str(S["tick"]) + " terminal=" + str(S["terminal_info"]))
result = report
