# v01_split_rim_grasp_button -- VideoRepick (robomme_plus task 13), AfI.
# UNTESTED trace-guided repair of v00_initial (v00: 0/1 on episode 0; the task's single
# simulator start is spent, so this version has never been run in the simulator).
#
# Phases A-F are v00's (context, SAM probes, self-calibration, video summary, depth tracking,
# live selection).  Repairs, each tied to the episode-0 trace:
#   R1 merged-blob split: at video frame 448 two touching crescents fused into one depth blob
#      (n=24, live n=20) and v00 grasped the pair's centroid -> fingers closed on empty space.
#      Oversize coloured blobs are now split with 2-means so each crescent keeps its own track.
#   R2 material grasp: the crescent is concave/thin (video hold reads gripper 0.13), so the grasp
#      point is the densest material point of the target seen in WRIST depth from a look pose,
#      with yaw chosen so the fingers close across the rim and land on free table.
#   R3 empty-close retry changes the pose (next grasp candidate) instead of repeating it; the
#      held test uses the post-lift gripper reading (v00 attempt 2 read 0.0301 then 0.0).
#   R4 the goal "... then press the button to stop" is parsed; after the two pick/place-down
#      cycles the program presses the button (white low-saturation static blob / SAM "button").
# Every phase is wrapped; the program never raises. Only `import math` is used.


import math

report = {"version": "v01_split_rim_grasp_button", "phase": "init", "errors": [], "log": []}
result = report
FLAGS = {"terminal": False, "terminal_label": None, "ticks": 0}


def say(msg):
    text = "[v01] " + str(msg)
    print(text)
    if len(report["log"]) < 400:
        report["log"].append(str(msg)[:300])


def err(where, exc):
    msg = str(where) + ": " + str(exc)
    print("[v01] ERROR " + msg[:400])
    if len(report["errors"]) < 100:
        report["errors"].append(msg[:500])


def r4(v):
    try:
        return round(float(v), 4)
    except Exception:
        return None


def vec3(p):
    return [float(p[0]), float(p[1]), float(p[2])]


def quat4(q):
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def quat_to_mat(q):
    w, x, y, z = quat4(q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        n = 1.0
    w = w / n
    x = x / n
    y = y / n
    z = z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def quat_mul(a, b):
    aw, ax, ay, az = quat4(a)
    bw, bx, by, bz = quat4(b)
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def quat_about_z(theta):
    return [math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)]


def median(values):
    vals = sorted(values)
    if len(vals) == 0:
        return None
    return vals[len(vals) // 2]


def quantile(values, frac):
    vals = sorted(values)
    if len(vals) == 0:
        return None
    idx = int((len(vals) - 1) * frac)
    return vals[idx]


def dist2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def row_list(arr, v):
    try:
        return arr[v].tolist()
    except Exception:
        row = arr[v]
        return [row[u] for u in range(len(row))]


def is_terminal(res):
    term = False
    try:
        if bool(res.terminated):
            term = True
    except Exception:
        pass
    try:
        if bool(res.truncated):
            term = True
    except Exception:
        pass
    try:
        st = str(res.status).lower()
        if ("terminat" in st) or ("truncat" in st) or ("episode_end" in st):
            term = True
    except Exception:
        pass
    return term


def res_summary(res):
    out = {}
    try:
        out["ok"] = bool(res.ok)
    except Exception:
        out["ok"] = None
    try:
        out["status"] = str(res.status)
    except Exception:
        pass
    try:
        out["steps"] = int(res.steps_executed)
    except Exception:
        pass
    try:
        if res.error is not None:
            out["error"] = str(res.error)[:300]
    except Exception:
        pass
    out["terminal"] = is_terminal(res)
    return out


def note_motion(label, res):
    s = res_summary(res)
    s["label"] = label
    try:
        FLAGS["ticks"] = FLAGS["ticks"] + int(s.get("steps", 0) or 0)
    except Exception:
        pass
    if s["terminal"] and not FLAGS["terminal"]:
        FLAGS["terminal"] = True
        FLAGS["terminal_label"] = label
        say("episode terminal after " + label)
    report.setdefault("motions", []).append(s)
    return s


# ---------------------------------------------------------------- Phase A: context
BASE = "robot_base"
EMB = "robomme_plus"
LANG = ""
MAX_STEPS = 400
FRONT = "front"
WRIST = "wrist"
HOME_POS = None
HOME_QUAT = None
try:
    report["phase"] = "A_context"
    ctx = get_task_context()
    LANG = str(ctx.language)
    report["language"] = LANG
    say("language: " + LANG)
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
except Exception as exc:
    err("task_context", exc)

try:
    meta = robomme_plus.get_task_metadata()
    md = {}
    for key in meta:
        md[str(key)] = str(meta[key])[:400]
    report["task_metadata"] = md
    try:
        MAX_STEPS = int(meta["max_steps"])
    except Exception as exc:
        err("max_steps", exc)
    if LANG == "":
        try:
            LANG = str(meta["language"])
            report["language"] = LANG
        except Exception:
            pass
except Exception as exc:
    err("task_metadata", exc)
report["max_steps"] = MAX_STEPS

try:
    cmeta = robomme_plus.get_controller_metadata()
    cm = {}
    for key in cmeta:
        cm[str(key)] = str(cmeta[key])[:400]
    report["controller_metadata"] = cm
except Exception as exc:
    err("controller_metadata", exc)

OBS0 = None
try:
    OBS0 = get_observation()
    names = [str(n) for n in OBS0.cameras]
    report["cameras"] = names
    if "front" not in names and len(names) > 0:
        FRONT = names[0]
    if "wrist" not in names:
        for n in names:
            if n != FRONT:
                WRIST = n
    caminfo = {}
    for n in names:
        c = OBS0.cameras[n]
        info = {}
        try:
            info["rgb_shape"] = [int(s) for s in c.rgb.shape]
            info["depth_shape"] = [int(s) for s in c.depth_m.shape]
            info["intrinsics"] = [[r4(c.intrinsics[i][j]) for j in range(3)] for i in range(3)]
            info["pose_pos"] = [r4(v) for v in c.camera_pose.position]
            info["pose_quat"] = [r4(v) for v in c.camera_pose.quaternion_wxyz]
            info["pose_frame"] = str(c.camera_pose.frame)
        except Exception as exc:
            err("caminfo " + n, exc)
        caminfo[n] = info
    report["camera_info"] = caminfo
    try:
        report["obs0_timestamp"] = r4(OBS0.timestamp_s)
    except Exception:
        pass
except Exception as exc:
    err("observation", exc)

try:
    st = get_robot_state()
    BASE = str(st.base_frame)
    EMB = st.embodiment
    HOME_POS = vec3(st.end_effector_poses["primary"].position)
    HOME_QUAT = quat4(st.end_effector_poses["primary"].quaternion_wxyz)
    report["robot_state0"] = {
        "base_frame": BASE,
        "embodiment": str(EMB),
        "joints": [r4(v) for v in st.joint_positions["primary"]],
        "gripper": r4(st.gripper_positions["primary"]),
        "ee_pos": [r4(v) for v in HOME_POS],
        "ee_quat": [r4(v) for v in HOME_QUAT],
        "timestamp": r4(st.timestamp_s),
    }
except Exception as exc:
    err("robot_state", exc)

# ------------------------------------------------------------- language parsing
LOW = LANG.lower().replace(",", " ").replace(".", " ").replace(";", " ")
WORDS = LOW.split()
NUMWORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10}
REPEATS = 1
try:
    for i in range(len(WORDS)):
        w = WORDS[i]
        nxt = WORDS[i + 1] if i + 1 < len(WORDS) else ""
        if w == "once":
            REPEATS = 1
        elif w == "twice":
            REPEATS = 2
        elif w == "thrice":
            REPEATS = 3
        elif nxt in ("times", "time"):
            if w in NUMWORDS:
                REPEATS = NUMWORDS[w]
            else:
                try:
                    REPEATS = int(w)
                except Exception:
                    pass
except Exception as exc:
    err("parse repeats", exc)
SAME_AS_VIDEO_COUNT = ("as many times" in LOW) or ("same number of times" in LOW)
PUT_DOWN = ("put it down" in LOW) or ("put down" in LOW) or ("place it back" in LOW) or \
    ("put it back" in LOW) or ("then place" in LOW) or ("and place" in LOW) or ("place down" in LOW)
PRESS_BUTTON = ("button" in WORDS) and (("press" in WORDS) or ("push" in WORDS) or ("hit" in WORDS))
COLOR_WORDS = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "white", "black",
               "gray", "grey", "brown", "cyan"]
LANG_COLOR = None
for cw in COLOR_WORDS:
    if cw in WORDS:
        LANG_COLOR = "gray" if cw == "grey" else cw
ORDINAL = "first"
for ow in ("second", "third", "last", "final"):
    if ow in WORDS:
        ORDINAL = ow
report["parsed"] = {"repeats": REPEATS, "same_as_video_count": SAME_AS_VIDEO_COUNT,
                    "put_down": PUT_DOWN, "lang_color": LANG_COLOR, "ordinal": ORDINAL,
                    "press_button": PRESS_BUTTON}
say("parsed " + str(report["parsed"]))


def color_name(rgbv):
    r = float(rgbv[0])
    g = float(rgbv[1])
    b = float(rgbv[2])
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 50:
        return "black"
    sat = (mx - mn) / mx if mx > 0 else 0.0
    if sat < 0.22:
        if mx > 190:
            return "white"
        return "gray"
    if mx == r:
        hue = 60.0 * (((g - b) / (mx - mn)) % 6.0)
    elif mx == g:
        hue = 60.0 * ((b - r) / (mx - mn) + 2.0)
    else:
        hue = 60.0 * ((r - g) / (mx - mn) + 4.0)
    if hue < 15 or hue >= 340:
        return "red"
    if hue < 40:
        return "orange"
    if hue < 70:
        return "yellow"
    if hue < 165:
        return "green"
    if hue < 195:
        return "cyan"
    if hue < 255:
        return "blue"
    if hue < 290:
        return "purple"
    return "pink"


# ---------------------------------------------------------------- Phase B: SAM probes
SEG_CACHE = {}


def probe(camera, prompt, keep=False):
    entry = {"camera": camera, "prompt": prompt, "ok": False, "n": 0, "scores": [], "boxes": []}
    try:
        seg = segment_text(camera, prompt)
        entry["ok"] = bool(seg.ok)
        if seg.ok:
            entry["n"] = len(seg.segmentations)
            for cand in seg.segmentations[:6]:
                entry["scores"].append(r4(cand.score))
                try:
                    entry["boxes"].append([round(float(v), 1) for v in cand.box_xyxy])
                except Exception:
                    entry["boxes"].append(None)
            if keep:
                SEG_CACHE[camera + "|" + prompt] = seg
    except Exception as exc:
        entry["error"] = str(exc)[:200]
    report.setdefault("sam_probes", []).append(entry)
    return entry


try:
    report["phase"] = "B_probes"
    probe(FRONT, "cube", keep=True)
    probe(FRONT, "block", keep=True)
    probe(FRONT, "small cube")
    if LANG_COLOR is not None:
        probe(FRONT, LANG_COLOR + " cube", keep=True)
    else:
        probe(FRONT, "red cube")
    probe(WRIST, "cube")
    say("front probes: " + str([(p["prompt"], p["n"], p["scores"][:3]) for p in report["sam_probes"]]))
except Exception as exc:
    err("probes", exc)


# ---------------------------------------------------------------- Phase C: calibration
def cam_model(cam):
    K = cam.intrinsics
    pose = cam.camera_pose
    return {
        "fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]), "cy": float(K[1][2]),
        "R": quat_to_mat(pose.quaternion_wxyz), "t": vec3(pose.position),
        "H": int(cam.depth_m.shape[0]), "W": int(cam.depth_m.shape[1]),
    }


CONVS = [
    {"name": "cv_z", "gl": False, "ray": False},
    {"name": "gl_z", "gl": True, "ray": False},
    {"name": "cv_ray", "gl": False, "ray": True},
    {"name": "gl_ray", "gl": True, "ray": True},
]


def ray_dir(m, conv, u, v):
    xn = (u - m["cx"]) / m["fx"]
    yn = (v - m["cy"]) / m["fy"]
    if conv["gl"]:
        c0, c1, c2 = xn, -yn, -1.0
    else:
        c0, c1, c2 = xn, yn, 1.0
    s = 1.0
    if conv["ray"]:
        s = 1.0 / math.sqrt(1.0 + xn * xn + yn * yn)
    R = m["R"]
    return (s * (R[0][0] * c0 + R[0][1] * c1 + R[0][2] * c2),
            s * (R[1][0] * c0 + R[1][1] * c1 + R[1][2] * c2),
            s * (R[2][0] * c0 + R[2][1] * c1 + R[2][2] * c2))


def project(m, conv, p):
    t = m["t"]
    R = m["R"]
    d0 = p[0] - t[0]
    d1 = p[1] - t[1]
    d2 = p[2] - t[2]
    pc0 = R[0][0] * d0 + R[1][0] * d1 + R[2][0] * d2
    pc1 = R[0][1] * d0 + R[1][1] * d1 + R[2][1] * d2
    pc2 = R[0][2] * d0 + R[1][2] * d1 + R[2][2] * d2
    if conv["gl"]:
        zc = -pc2
        yv = -pc1
    else:
        zc = pc2
        yv = pc1
    if zc <= 1e-6:
        return None
    return (m["cx"] + m["fx"] * pc0 / zc, m["cy"] + m["fy"] * yv / zc)


def cloud_points(pc, limit=600):
    pts = pc.points
    n = len(pts)
    stepn = max(1, n // limit)
    out = []
    for i in range(0, n, stepn):
        p = pts[i]
        out.append((float(p[0]), float(p[1]), float(p[2])))
    return out


def cloud_median(points):
    return (median([p[0] for p in points]), median([p[1] for p in points]),
            median([p[2] for p in points]))


def flatness(obs, conv):
    """Largest 5 mm horizontal-height bin over a stride-4 full-frame back-projection."""
    depth = obs.cameras[FRONT].depth_m
    t = FM["t"]
    bins = {}
    for v in range(0, FM["H"], 4):
        drow = row_list(depth, v)
        for u in range(0, FM["W"], 4):
            d = float(drow[u])
            if d <= 0.01 or d > 8.0:
                continue
            rd = ray_dir(FM, conv, u + 0.5, v + 0.5)
            z = t[2] + d * rd[2]
            k = int(math.floor(z / 0.005))
            bins[k] = bins.get(k, 0) + 1
    bestk = None
    for k in bins:
        if bestk is None or bins[k] > bins[bestk]:
            bestk = k
    if bestk is None:
        return (0, None)
    return (bins[bestk], (bestk + 0.5) * 0.005)


FM = None
CONV = CONVS[0]
CALIB = {"chosen": "cv_z(default)", "evidence": []}
try:
    report["phase"] = "C_calibration"
    FM = cam_model(OBS0.cameras[FRONT])
    seg = None
    for key in (FRONT + "|cube", FRONT + "|block"):
        if key in SEG_CACHE and len(SEG_CACHE[key].segmentations) > 0:
            seg = SEG_CACHE[key]
            break
    depth0 = OBS0.cameras[FRONT].depth_m
    totals = [0.0, 0.0, 0.0, 0.0]
    used = 0
    if seg is not None:
        for cand in seg.segmentations[:3]:
            try:
                pc = mask_to_point_cloud(cand, FRONT, target_frame=BASE)
                pts = cloud_points(pc, 400)
                if len(pts) < 10:
                    continue
                ref = cloud_median(pts)
                x1, y1, x2, y2 = [float(v) for v in cand.box_xyxy]
                pix = []
                mask_used = False
                try:
                    mk = cand.mask
                    for v in range(max(0, int(y1)), min(FM["H"], int(y2) + 1)):
                        mrow = row_list(mk, v)
                        for u in range(max(0, int(x1)), min(FM["W"], int(x2) + 1)):
                            if mrow[u]:
                                pix.append((u, v))
                    mask_used = len(pix) > 0
                except Exception:
                    pix = []
                if len(pix) == 0:
                    cxb = (x1 + x2) / 2.0
                    cyb = (y1 + y2) / 2.0
                    hw = max(1.0, (x2 - x1) * 0.2)
                    hh = max(1.0, (y2 - y1) * 0.2)
                    for v in range(int(cyb - hh), int(cyb + hh) + 1):
                        for u in range(int(cxb - hw), int(cxb + hw) + 1):
                            pix.append((u, v))
                stepn = max(1, len(pix) // 300)
                pix = pix[::stepn]
                errs = []
                for ci in range(4):
                    conv = CONVS[ci]
                    xs = []
                    ys = []
                    zs = []
                    for (u, v) in pix:
                        if u < 0 or v < 0 or u >= FM["W"] or v >= FM["H"]:
                            continue
                        d = float(depth0[v][u])
                        if d <= 0.01 or d > 8.0:
                            continue
                        rd = ray_dir(FM, conv, u + 0.5, v + 0.5)
                        xs.append(FM["t"][0] + d * rd[0])
                        ys.append(FM["t"][1] + d * rd[1])
                        zs.append(FM["t"][2] + d * rd[2])
                    if len(xs) < 5:
                        errs.append(9.0)
                        continue
                    mine = (median(xs), median(ys), median(zs))
                    e = math.sqrt((mine[0] - ref[0]) ** 2 + (mine[1] - ref[1]) ** 2 + (mine[2] - ref[2]) ** 2)
                    errs.append(e)
                for ci in range(4):
                    totals[ci] = totals[ci] + errs[ci]
                used = used + 1
                pj = [project(FM, CONVS[ci], ref) for ci in range(2)]
                CALIB["evidence"].append({
                    "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "cloud_median": [r4(v) for v in ref], "n_cloud": len(pc.points),
                    "mask_used": mask_used, "n_pix": len(pix),
                    "errs": [r4(e) for e in errs],
                    "proj_cv": None if pj[0] is None else [round(pj[0][0], 1), round(pj[0][1], 1)],
                    "proj_gl": None if pj[1] is None else [round(pj[1][0], 1), round(pj[1][1], 1)],
                })
            except Exception as exc:
                err("calib cand", exc)
    # Table flatness separates z-depth from ray-depth (a plane stays a plane only under the
    # right depth model); the SAM-cloud error separates the camera-axis convention.
    flat = []
    for ci in range(4):
        flat.append(flatness(OBS0, CONVS[ci]))
    CALIB["flatness"] = [[f[0], r4(f[1])] for f in flat]
    pool = []
    if used > 0:
        bestE = 0
        for ci in range(4):
            if totals[ci] < totals[bestE]:
                bestE = ci
        CALIB["mean_err"] = [r4(t / used) for t in totals]
        for ci in range(4):
            if totals[ci] / used < 0.05:
                pool.append(ci)
        if len(pool) == 0:
            pool = [bestE]
        CALIB["ok"] = (totals[bestE] / used) < 0.05
    else:
        for ci in range(4):
            if flat[ci][1] is not None and flat[ci][1] < FM["t"][2] - 0.10:
                pool.append(ci)
        CALIB["ok"] = len(pool) > 0
        CALIB["no_sam_reference"] = True
    best = None
    for ci in pool:
        if best is None or flat[ci][0] > flat[best][0]:
            best = ci
    if best is not None:
        CONV = CONVS[best]
        CALIB["chosen"] = CONV["name"]
    say("calibration " + str(CALIB.get("chosen")) + " errs " + str(CALIB.get("mean_err")))
except Exception as exc:
    err("calibration", exc)
    CALIB["ok"] = False
report["calibration"] = CALIB

# ----------------------------------------------------- Phase D: conditioning video summary
VIDEO = []
VN = 0
VG = []
VZ = []
VXY = []
VQ = []
VT = []
EVENTS = []
try:
    report["phase"] = "D_video"
    VIDEO = robomme_plus.get_conditioning_video()
    VN = len(VIDEO)
    vs = {"n_frames": VN}
    for i in range(VN):
        fr = VIDEO[i]
        try:
            rs = fr.robot_state
            VG.append(float(rs.gripper_positions["primary"]))
            p = rs.end_effector_poses["primary"]
            pp = vec3(p.position)
            VZ.append(pp[2])
            VXY.append((pp[0], pp[1]))
            VQ.append(quat4(p.quaternion_wxyz))
        except Exception as exc:
            if len(VG) == 0:
                err("video robot_state", exc)
            VG.append(VG[-1] if len(VG) > 0 else 1.0)
            VZ.append(VZ[-1] if len(VZ) > 0 else 1.0)
            VXY.append(VXY[-1] if len(VXY) > 0 else (0.0, 0.0))
            VQ.append(VQ[-1] if len(VQ) > 0 else [0.0, 1.0, 0.0, 0.0])
        try:
            VT.append(float(fr.timestamp_s))
        except Exception:
            VT.append(float(i))
    if VN > 0:
        vs["t_first"] = r4(VT[0])
        vs["t_last"] = r4(VT[-1])
        vs["mean_dt"] = r4((VT[-1] - VT[0]) / max(1, VN - 1))
        stride = max(1, VN // 128)
        vs["trace_stride"] = stride
        vs["trace"] = [[i, r4(VT[i]), r4(VG[i]), r4(VXY[i][0]), r4(VXY[i][1]), r4(VZ[i])]
                       for i in range(0, VN, stride)]
        vs["gripper_min"] = r4(min(VG))
        vs["gripper_max"] = r4(max(VG))
        vs["ee_z_min"] = r4(min(VZ))
    report["video"] = vs
    say("video frames " + str(VN))
except Exception as exc:
    err("video", exc)


def find_held_segments(table_z):
    """Gripper segments that are (a) partially closed and stable = holding something,
    (b) began near table height, (c) actually lifted. Empty / in-air closes are rejected."""
    segs = []
    rejected = []
    if VN < 3:
        return segs, rejected
    gopen = quantile(VG, 0.9)
    if gopen is None or gopen < 0.2:
        gopen = 1.0
    hi = 0.85 * gopen
    lo = 0.05 * gopen
    zmin = min(VZ)
    i = 0
    while i < VN:
        if lo < VG[i] < hi:
            j = i
            while j + 1 < VN and lo < VG[j + 1] < hi:
                j = j + 1
            n = j - i + 1
            seg = {"f0": i, "f1": j, "n": n}
            mid = VG[i + n // 4: j + 1 - n // 4] if n >= 4 else VG[i:j + 1]
            grange = max(mid) - min(mid)
            z0 = VZ[i]
            zmax = max(VZ[i:j + 1])
            seg["g_med"] = r4(median(VG[i:j + 1]))
            seg["g_range_mid"] = r4(grange)
            seg["z_start"] = r4(z0)
            seg["lift"] = r4(zmax - z0)
            seg["grasp_xy"] = [r4(VXY[i][0]), r4(VXY[i][1])]
            seg["release_xy"] = [r4(VXY[j][0]), r4(VXY[j][1])]
            seg["release_z"] = r4(VZ[j])
            seg["t0"] = r4(VT[i])
            seg["t1"] = r4(VT[j])
            low_ok = (z0 < table_z + 0.10) if table_z is not None else (z0 < zmin + 0.06)
            why = []
            if n < 3:
                why.append("short")
            if grange > 0.15:
                why.append("unstable")
            if not low_ok:
                why.append("not_low")
            if zmax - z0 < 0.02:
                why.append("not_lifted")
            if len(why) == 0:
                segs.append(seg)
            else:
                seg["why"] = why
                rejected.append(seg)
            i = j + 1
        else:
            i = i + 1
    return segs, rejected


SEGS = []
try:
    SEGS, rej = find_held_segments(None)
    report["video_held_segments_pre"] = SEGS
    report["video_rejected_segments_pre"] = rej[:12]
    say("held segments (pre table): " + str([(s["f0"], s["f1"], s["grasp_xy"], s["release_xy"]) for s in SEGS]))
except Exception as exc:
    err("segments", exc)

# --------------------------------------------- Phase E: depth-based cube detection + tracking
TABLE_Z = None
RAYC = {}


def ray_row(v):
    if v in RAYC:
        return RAYC[v]
    row = [ray_dir(FM, CONV, u + 0.5, v + 0.5) for u in range(FM["W"])]
    RAYC[v] = row
    return row


def estimate_table_z(obs, hint_hi):
    depth = obs.cameras[FRONT].depth_m
    t = FM["t"]
    bins = {}
    for v in range(0, FM["H"], 4):
        drow = row_list(depth, v)
        rr = ray_row(v)
        for u in range(0, FM["W"], 4):
            d = float(drow[u])
            if d <= 0.01 or d > 8.0:
                continue
            rd = rr[u]
            x = t[0] + d * rd[0]
            y = t[1] + d * rd[1]
            z = t[2] + d * rd[2]
            if x * x + y * y > 1.5 * 1.5:
                continue
            if hint_hi is not None and (z > hint_hi or z < hint_hi - 0.10):
                continue
            k = int(math.floor(z / 0.005))
            bins[k] = bins.get(k, 0) + 1
    if len(bins) == 0:
        return None, 0
    bestk = None
    for k in bins:
        if bestk is None or bins[k] > bins[bestk]:
            bestk = k
    return (bestk + 0.5) * 0.005, bins[bestk]


try:
    report["phase"] = "E_table"
    hint = None
    if len(SEGS) > 0:
        hint = float(SEGS[0]["z_start"]) - 0.003
    tz, cnt = estimate_table_z(OBS0, hint)
    if tz is None and hint is not None:
        tz, cnt = estimate_table_z(OBS0, None)
    TABLE_Z = tz
    report["table_z"] = r4(TABLE_Z)
    report["table_bin_count"] = cnt
    say("table z " + str(r4(TABLE_Z)) + " count " + str(cnt))
    if TABLE_Z is not None:
        SEGS, rej = find_held_segments(TABLE_Z)
        report["video_held_segments"] = SEGS
        report["video_rejected_segments"] = rej[:12]
        say("held segments: " + str([(s["f0"], s["f1"], s["grasp_xy"], s["release_xy"]) for s in SEGS]))
except Exception as exc:
    err("table", exc)

BAND_LO = 0.012
BAND_HI = 0.13
ROI = [0, 0, 256, 256]
STRIDE = 3


def detect(obs, roi, stride, excl):
    """Cube-like blobs: depth points 1.2-13 cm above the table, grid connected components.
    excl = (x, y, radius) region (gripper vicinity) whose blobs are dropped, or None."""
    cam = obs.cameras[FRONT]
    depth = cam.depth_m
    rgb = cam.rgb
    t = FM["t"]
    u0, v0, u1, v1 = roi
    pts = {}
    lo = TABLE_Z + BAND_LO
    hi = TABLE_Z + BAND_HI
    for v in range(v0, v1, stride):
        drow = row_list(depth, v)
        rr = ray_row(v)
        crow = None
        for u in range(u0, u1, stride):
            d = drow[u]
            if d <= 0.01 or d > 8.0:
                continue
            rd = rr[u]
            z = t[2] + d * rd[2]
            if z < lo or z > hi:
                continue
            x = t[0] + d * rd[0]
            y = t[1] + d * rd[1]
            if crow is None:
                crow = row_list(rgb, v)
            c = crow[u]
            pts[((u - u0) // stride, (v - v0) // stride)] = (x, y, z, c[0], c[1], c[2], u, v)
    seen = set()
    blobs = []
    for key in pts:
        if key in seen:
            continue
        seen.add(key)
        stack = [key]
        members = []
        while len(stack) > 0:
            k = stack.pop()
            p = pts[k]
            members.append(p)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nk = (k[0] + di, k[1] + dj)
                    if nk in seen or nk not in pts:
                        continue
                    q = pts[nk]
                    if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2 < 0.03 * 0.03:
                        seen.add(nk)
                        stack.append(nk)
        n = len(members)
        if n < 4:
            continue
        xs = [m[0] for m in members]
        ys = [m[1] for m in members]
        ext = max(max(xs) - min(xs), max(ys) - min(ys))
        if ext > 0.11:
            continue
        bx = sum(xs) / n
        by = sum(ys) / n
        if excl is not None and dist2((bx, by), excl) < excl[2]:
            continue
        parts = split_members(members)
        for part in parts:
            blobs.append(make_blob(part, len(parts) > 1))
    return blobs


def xy_extent(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return max(max(xs) - min(xs), max(ys) - min(ys))


SPLIT = {"single_n": None, "single_ext": None, "count": 0, "examples": []}


def make_blob(members, was_split):
    n = len(members)
    xs = [m[0] for m in members]
    ys = [m[1] for m in members]
    return {
        "x": sum(xs) / n, "y": sum(ys) / n, "zmax": max([m[2] for m in members]), "n": n,
        "ext": max(max(xs) - min(xs), max(ys) - min(ys)),
        "col": (sum([float(m[3]) for m in members]) / n, sum([float(m[4]) for m in members]) / n,
                sum([float(m[5]) for m in members]) / n),
        "u": sum([m[6] for m in members]) / n, "v": sum([m[7] for m in members]) / n,
        "umin": min([m[6] for m in members]), "umax": max([m[6] for m in members]),
        "vmin": min([m[7] for m in members]), "vmax": max([m[7] for m in members]),
        "pts": [(m[0], m[1], m[2]) for m in members], "split": was_split,
    }


def saturation(col):
    return max(col[0], col[1], col[2]) - min(col[0], col[1], col[2])


def two_means(items):
    """2-means on xy, initialised at the extremes of the principal axis. Returns (A, B, sep)."""
    n = len(items)
    mx = sum([m[0] for m in items]) / n
    my = sum([m[1] for m in items]) / n
    sxx = sum([(m[0] - mx) ** 2 for m in items])
    syy = sum([(m[1] - my) ** 2 for m in items])
    sxy = sum([(m[0] - mx) * (m[1] - my) for m in items])
    ang = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    ca = math.cos(ang)
    sa = math.sin(ang)
    proj = [(m[0] - mx) * ca + (m[1] - my) * sa for m in items]
    lo = min(proj)
    hi = max(proj)
    c1 = (mx + lo * 0.5 * ca, my + lo * 0.5 * sa)
    c2 = (mx + hi * 0.5 * ca, my + hi * 0.5 * sa)
    A = []
    B = []
    for it in range(8):
        A = []
        B = []
        for m in items:
            p = m
            if (p[0] - c1[0]) ** 2 + (p[1] - c1[1]) ** 2 <= (p[0] - c2[0]) ** 2 + (p[1] - c2[1]) ** 2:
                A.append(m)
            else:
                B.append(m)
        if len(A) == 0 or len(B) == 0:
            return items, [], 0.0
        c1 = (sum([m[0] for m in A]) / len(A), sum([m[1] for m in A]) / len(A))
        c2 = (sum([m[0] for m in B]) / len(B), sum([m[1] for m in B]) / len(B))
    return A, B, dist2(c1, c2)


def neck_ratio(members, A, B):
    """Members in the middle fifth between the two 2-means centres, relative to a uniform
    spread (1.0). Two touching objects have a narrow contact (<0.7); one convex object does not."""
    c1 = (sum([m[0] for m in A]) / len(A), sum([m[1] for m in A]) / len(A))
    c2 = (sum([m[0] for m in B]) / len(B), sum([m[1] for m in B]) / len(B))
    L = dist2(c1, c2)
    if L < 1e-6:
        return 1.0
    ux = (c2[0] - c1[0]) / L
    uy = (c2[1] - c1[1]) / L
    mx = 0.5 * (c1[0] + c2[0])
    my = 0.5 * (c1[1] + c2[1])
    proj = [(m[0] - mx) * ux + (m[1] - my) * uy for m in members]
    span = max(proj) - min(proj)
    if span < 1e-6:
        return 1.0
    mid = len([v for v in proj if abs(v) < span / 10.0])
    return mid / (len(members) / 5.0)


def split_members(members):
    """R1: split an oversize *coloured* blob (two touching objects) into two parts.
    Only applies once the single-object blob size is known from the first video frame;
    white/gray low-saturation blobs (button, robot) are never split."""
    sn = SPLIT["single_n"]
    se = SPLIT["single_ext"]
    n = len(members)
    if sn is None or se is None or n < 1.7 * sn:
        return [members]
    if xy_extent(members) < 1.6 * se:
        return [members]
    col = (sum([float(m[3]) for m in members]) / n, sum([float(m[4]) for m in members]) / n,
           sum([float(m[5]) for m in members]) / n)
    if saturation(col) < 40.0:
        return [members]
    A, B, sep = two_means(members)
    if len(B) == 0 or sep < max(0.022, 0.55 * se) or min(len(A), len(B)) < 0.35 * sn:
        return [members]
    neck = neck_ratio(members, A, B)
    if neck > 0.7:
        return [members]
    SPLIT["count"] = SPLIT["count"] + 1
    if len(SPLIT["examples"]) < 6:
        SPLIT["examples"].append([n, len(A), len(B), r4(sep), r4(neck)])
    out = []
    for part in (A, B):
        if len(part) >= 2.6 * sn:
            out = out + split_members(part)
        else:
            out.append(part)
    return out


def blob_brief(b):
    return [r4(b["x"]), r4(b["y"]), b["n"], color_name(b["col"])]


TRACKS = []
TARGET = {"track": None, "seg": None}
TRACK_OK = False


def new_track(x, y, col, f):
    tr = {"id": len(TRACKS), "x": x, "y": y, "vx": 0.0, "vy": 0.0, "col": col, "miss": 0,
          "born": f, "hist": []}
    TRACKS.append(tr)
    return tr


def update_tracks(blobs, f, ee, carried_id):
    """Greedy global-nearest association with colour tie-break and growing gates."""
    excl_r = 0.07
    ee_low = ee is not None and TABLE_Z is not None and ee[2] < TABLE_Z + 0.20
    active = []
    for tr in TRACKS:
        if carried_id is not None and tr["id"] == carried_id:
            continue
        if ee_low and dist2((tr["x"], tr["y"]), ee) < excl_r:
            continue  # frozen next to the gripper
        active.append(tr)
    pairs = []
    for ti in range(len(active)):
        tr = active[ti]
        px = tr["x"] + tr["vx"]
        py = tr["y"] + tr["vy"]
        gate = min(0.35, 0.08 + 0.03 * tr["miss"] + 2.0 * math.sqrt(tr["vx"] ** 2 + tr["vy"] ** 2))
        for bi in range(len(blobs)):
            b = blobs[bi]
            dd = math.sqrt((px - b["x"]) ** 2 + (py - b["y"]) ** 2)
            if dd > gate:
                continue
            cd = math.sqrt((tr["col"][0] - b["col"][0]) ** 2 + (tr["col"][1] - b["col"][1]) ** 2 +
                           (tr["col"][2] - b["col"][2]) ** 2)
            pairs.append((dd + 0.0002 * cd + 0.01 * tr["miss"], ti, bi))
    pairs = sorted(pairs)
    used_t = set()
    used_b = set()
    for (cost, ti, bi) in pairs:
        if ti in used_t or bi in used_b:
            continue
        used_t.add(ti)
        used_b.add(bi)
        tr = active[ti]
        b = blobs[bi]
        nvx = b["x"] - tr["x"]
        nvy = b["y"] - tr["y"]
        tr["vx"] = 0.5 * tr["vx"] + 0.5 * nvx
        tr["vy"] = 0.5 * tr["vy"] + 0.5 * nvy
        tr["path"] = tr.get("path", 0.0) + math.sqrt(nvx * nvx + nvy * nvy)
        tr["x"] = b["x"]
        tr["y"] = b["y"]
        tr["col"] = (0.8 * tr["col"][0] + 0.2 * b["col"][0], 0.8 * tr["col"][1] + 0.2 * b["col"][1],
                     0.8 * tr["col"][2] + 0.2 * b["col"][2])
        tr["miss"] = 0
        tr["last_blob"] = b
    for ti in range(len(active)):
        if ti in used_t:
            continue
        tr = active[ti]
        tr["miss"] = tr["miss"] + 1
        if tr["miss"] <= 4:
            tr["x"] = tr["x"] + 0.5 * tr["vx"]
            tr["y"] = tr["y"] + 0.5 * tr["vy"]
        tr["vx"] = 0.5 * tr["vx"]
        tr["vy"] = 0.5 * tr["vy"]
    for bi in range(len(blobs)):
        if bi in used_b:
            continue
        b = blobs[bi]
        near = False
        for tr in TRACKS:
            if dist2((tr["x"], tr["y"]), (b["x"], b["y"])) < 0.05:
                near = True
        if (not near) and b["n"] >= 6:
            new_track(b["x"], b["y"], b["col"], f)
    return len(used_b)


def sig_of(obs, roi):
    depth = obs.cameras[FRONT].depth_m
    out = []
    u0, v0, u1, v1 = roi
    for v in range(v0, v1, 6):
        drow = row_list(depth, v)
        for u in range(u0, u1, 6):
            out.append(float(drow[u]))
    return out


def sig_changed(a, b):
    if a is None or b is None or len(a) != len(b):
        return True
    for i in range(len(a)):
        if abs(a[i] - b[i]) > 0.008:
            return True
    return False


TRACKLOG = []
try:
    report["phase"] = "E_tracking"
    if FM is not None and TABLE_Z is not None and CALIB.get("ok", False):
        TRACK_OK = True
        # ROI from blobs in the live frame and the first video frame, plus video grasp/release points.
        full = [0, 0, FM["W"], FM["H"]]
        seeds = []
        b_live = detect(OBS0, full, 3, None)
        seeds = seeds + b_live
        if VN > 0:
            seeds = seeds + detect(VIDEO[0], full, 3, None)
        report["live_blobs_initial_fullscan"] = [blob_brief(b) for b in b_live]
        us = []
        vs2 = []
        for b in seeds:
            us = us + [b["umin"], b["umax"]]
            vs2 = vs2 + [b["vmin"], b["vmax"]]
        for s in SEGS:
            for key in ("grasp_xy", "release_xy"):
                pj = project(FM, CONV, (s[key][0], s[key][1], TABLE_Z + 0.02))
                if pj is not None:
                    us.append(pj[0])
                    vs2.append(pj[1])
        if len(us) > 0:
            m = 44
            ROI = [max(0, int(min(us)) - m), max(0, int(min(vs2)) - m),
                   min(FM["W"], int(max(us)) + m + 1), min(FM["H"], int(max(vs2)) + m + 1)]
        else:
            ROI = full
        area = (ROI[2] - ROI[0]) * (ROI[3] - ROI[1])
        STRIDE = 2 if area / 4.0 <= 5000 else 3
        report["roi"] = ROI
        report["stride"] = STRIDE
        # R1: single-object blob size from the first video frame (objects are apart at the start)
        try:
            ref_obs = VIDEO[0] if VN > 0 else OBS0
            ref_blobs = detect(ref_obs, ROI, STRIDE, None)
            ns = [b["n"] for b in ref_blobs if saturation(b["col"]) >= 40.0]
            es = [b["ext"] for b in ref_blobs if saturation(b["col"]) >= 40.0]
            if len(ns) >= 2:
                SPLIT["single_n"] = median(ns)
                SPLIT["single_ext"] = median(es)
            report["split_single_n"] = SPLIT["single_n"]
            report["split_single_ext"] = r4(SPLIT["single_ext"])
            say("single-object blob size " + str(SPLIT["single_n"]) + " from " + str(ns))
        except Exception as exc:
            err("single_n", exc)
        say("roi " + str(ROI) + " stride " + str(STRIDE))

        seg_at = {}
        for si in range(len(SEGS)):
            for f in range(SEGS[si]["f0"], SEGS[si]["f1"] + 1):
                seg_at[f] = si
        seg_track = {}
        chosen_seg = None
        if len(SEGS) > 0:
            if ORDINAL in ("last", "final"):
                chosen_seg = len(SEGS) - 1
            elif ORDINAL == "second" and len(SEGS) > 1:
                chosen_seg = 1
            elif ORDINAL == "third" and len(SEGS) > 2:
                chosen_seg = 2
            else:
                chosen_seg = 0
        TARGET["seg"] = chosen_seg
        last_sig = None
        last_blobs = []
        n_detect = 0
        frames = []
        for i in range(VN):
            frames.append(("v", i))
        for i in range(VN):
            kind, idx = frames[i]
            fr = VIDEO[idx]
            ee = (VXY[idx][0], VXY[idx][1], VZ[idx])
            si = seg_at.get(idx, None)
            carried = None
            if si is not None:
                if si not in seg_track:
                    gx, gy = SEGS[si]["grasp_xy"]
                    best = None
                    for tr in TRACKS:
                        dd = dist2((tr["x"], tr["y"]), (gx, gy))
                        if dd < 0.06 and (best is None or dd < best[0]):
                            best = (dd, tr)
                    if best is None:
                        trn = new_track(gx, gy, (128.0, 128.0, 128.0), idx)
                        seg_track[si] = trn["id"]
                    else:
                        seg_track[si] = best[1]["id"]
                    if si == chosen_seg:
                        TARGET["track"] = seg_track[si]
                carried = seg_track[si]
                ctr = TRACKS[carried]
                ctr["x"] = ee[0]
                ctr["y"] = ee[1]
                ctr["vx"] = 0.0
                ctr["vy"] = 0.0
                ctr["miss"] = 0
            sig = sig_of(fr, ROI)
            ee_low = ee[2] < TABLE_Z + 0.20
            excl = (ee[0], ee[1], 0.07) if ee_low else None
            if sig_changed(sig, last_sig) or i == 0 or si is not None:
                blobs = detect(fr, ROI, STRIDE, excl)
                n_detect = n_detect + 1
                last_sig = sig
                update_tracks(blobs, idx, ee if ee_low else None, carried)
                last_blobs = blobs
            if i % max(1, VN // 64) == 0 or i == VN - 1:
                TRACKLOG.append([idx, [[tr["id"], r4(tr["x"]), r4(tr["y"]), tr["miss"]] for tr in TRACKS],
                                 [blob_brief(b) for b in last_blobs]])
        report["video_detect_frames"] = n_detect
        report["video_tracks_final"] = [[tr["id"], r4(tr["x"]), r4(tr["y"]), tr["miss"], tr["born"],
                                         color_name(tr["col"]), r4(tr.get("path", 0.0))] for tr in TRACKS]
        report["target_track"] = TARGET["track"]
        say("video tracking done; detections " + str(n_detect) + " tracks " +
            str(report["video_tracks_final"]) + " target " + str(TARGET["track"]))
except Exception as exc:
    err("video tracking", exc)
    TRACK_OK = False
report["tracklog"] = TRACKLOG

# fallback target position from robot events only (no visual tracking)
EVENT_POS = None
try:
    if TARGET["seg"] is None and len(SEGS) > 0:
        TARGET["seg"] = 0
    if TARGET["seg"] is not None:
        s = SEGS[TARGET["seg"]]
        EVENT_POS = (float(s["release_xy"][0]), float(s["release_xy"][1]))
        later = 0
        for s2 in SEGS[TARGET["seg"] + 1:]:
            if dist2((float(s2["grasp_xy"][0]), float(s2["grasp_xy"][1])), EVENT_POS) < 0.04:
                EVENT_POS = (float(s2["release_xy"][0]), float(s2["release_xy"][1]))
                later = later + 1
        report["event_target_pos"] = [r4(EVENT_POS[0]), r4(EVENT_POS[1])]
        report["event_later_moves"] = later
except Exception as exc:
    err("event pos", exc)


# ------------------------------------------- live: continue tracking until the scene is still
def hold_ticks(n, grip):
    k = 0
    while k < n and not FLAGS["terminal"]:
        try:
            st = get_robot_state()
            rep = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=st.joint_positions["primary"],
                gripper_position=grip, embodiment=st.embodiment)}))
            FLAGS["ticks"] = FLAGS["ticks"] + 1
            if is_terminal(rep):
                FLAGS["terminal"] = True
                FLAGS["terminal_label"] = "hold_tick"
                say("terminal during hold tick")
        except Exception as exc:
            err("hold tick", exc)
            return k
        k = k + 1
    return k


def track_target_pos():
    if TARGET["track"] is None:
        return None
    tr = TRACKS[TARGET["track"]]
    return (tr["x"], tr["y"])


LIVE = {"checks": []}
try:
    report["phase"] = "E_live_tracking"
    if TRACK_OK:
        st = get_robot_state()
        ee = vec3(st.end_effector_poses["primary"].position)
        ee_low = ee[2] < TABLE_Z + 0.20
        blobs = detect(OBS0, ROI, STRIDE, (ee[0], ee[1], 0.07) if ee_low else None)
        update_tracks(blobs, VN, ee if ee_low else None, None)
        prev = {}
        for tr in TRACKS:
            prev[tr["id"]] = (tr["x"], tr["y"])
        LIVE["obs0_blobs"] = [blob_brief(b) for b in blobs]
        still_count = 0
        budget = min(150, int(MAX_STEPS * 0.3))
        used = 0
        while used < budget and not FLAGS["terminal"]:
            used = used + hold_ticks(4 if used == 0 else 3, 1.0)
            obs = get_observation()
            blobs = detect(obs, ROI, STRIDE, None)
            update_tracks(blobs, VN + used, None, None)
            mv = 0.0
            for tr in TRACKS:
                if tr["id"] in prev and tr["miss"] == 0:
                    mv = max(mv, dist2(prev[tr["id"]], (tr["x"], tr["y"])))
                prev[tr["id"]] = (tr["x"], tr["y"])
            tp = track_target_pos()
            LIVE["checks"].append([used, r4(mv), len(blobs),
                                   None if tp is None else [r4(tp[0]), r4(tp[1])]])
            if mv < 0.004:
                still_count = still_count + 1
            else:
                still_count = 0
            if still_count >= 2 or (used <= 4 and mv < 0.002 and still_count >= 1):
                break
        LIVE["ticks_waited"] = used
        LIVE["tracks"] = [[tr["id"], r4(tr["x"]), r4(tr["y"]), tr["miss"], color_name(tr["col"])] for tr in TRACKS]
        say("live tracking: waited " + str(used) + " ticks; checks " + str(LIVE["checks"][-3:]))
    else:
        hold_ticks(2, 1.0)
except Exception as exc:
    err("live tracking", exc)
report["live_tracking"] = LIVE

# ---------------------------------------------------------------- Phase F: select the cube
TARGET_XY = None
TARGET_INFO = {}
try:
    report["phase"] = "F_select"
    TARGET_XY = track_target_pos()
    TARGET_INFO["source"] = "visual_track" if TARGET_XY is not None else None
    if TARGET_XY is None and EVENT_POS is not None:
        TARGET_XY = EVENT_POS
        TARGET_INFO["source"] = "video_release_event"
    if TARGET_XY is not None:
        TARGET_INFO["expected_xy"] = [r4(TARGET_XY[0]), r4(TARGET_XY[1])]
    if EVENT_POS is not None and TARGET_XY is not None:
        TARGET_INFO["track_vs_event_dist"] = r4(dist2(TARGET_XY, EVENT_POS))
    obs = get_observation()
    cands = []
    for prompt in ("cube", "block"):
        seg = None
        try:
            seg = segment_text(FRONT, prompt)
        except Exception as exc:
            err("segment live " + prompt, exc)
        if seg is None or not seg.ok:
            continue
        for ci in range(min(8, len(seg.segmentations))):
            cand = seg.segmentations[ci]
            try:
                pc = mask_to_point_cloud(cand, FRONT, target_frame=BASE)
                pts = cloud_points(pc, 500)
                if len(pts) < 15:
                    continue
                med = cloud_median(pts)
                zs = [p[2] for p in pts]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ext = max(quantile(xs, 0.95) - quantile(xs, 0.05), quantile(ys, 0.95) - quantile(ys, 0.05))
                ztop = quantile(zs, 0.95)
                col = None
                pj = project(FM, CONV, med) if FM is not None else None
                if pj is not None:
                    rgbimg = obs.cameras[FRONT].rgb
                    acc = [0.0, 0.0, 0.0]
                    cn = 0
                    for dv in (-1, 0, 1):
                        for du in (-1, 0, 1):
                            uu = int(pj[0]) + du
                            vv = int(pj[1]) + dv
                            if 0 <= uu < FM["W"] and 0 <= vv < FM["H"]:
                                px = rgbimg[vv][uu]
                                acc[0] = acc[0] + float(px[0])
                                acc[1] = acc[1] + float(px[1])
                                acc[2] = acc[2] + float(px[2])
                                cn = cn + 1
                    if cn > 0:
                        col = (acc[0] / cn, acc[1] / cn, acc[2] / cn)
                height = (ztop - TABLE_Z) if TABLE_Z is not None else None
                plausible = ext < 0.10 and (height is None or (0.01 < height < 0.12))
                cands.append({"prompt": prompt, "rank": ci, "score": r4(cand.score), "x": med[0],
                              "y": med[1], "z": med[2], "ztop": ztop, "ext": ext, "n": len(pc.points),
                              "color": None if col is None else color_name(col),
                              "height": height, "plausible": plausible, "pts": pts})
            except Exception as exc:
                err("cand cloud", exc)
        if len(cands) > 0 and prompt == "cube":
            break
    pool = [c for c in cands if c["plausible"]]
    if len(pool) == 0:
        pool = cands
    TARGET_INFO["n_candidates"] = len(cands)
    if LANG_COLOR is not None:
        cpool = [c for c in pool if c["color"] == LANG_COLOR]
        TARGET_INFO["color_filter"] = [LANG_COLOR, len(cpool)]
        if len(cpool) > 0:
            pool = cpool
            TARGET_INFO["color_decides"] = True
    best = None
    for c in pool:
        if TARGET_XY is not None:
            c["d"] = dist2((c["x"], c["y"]), TARGET_XY)
        else:
            c["d"] = 0.0 - float(c["score"] or 0.0)
        if best is None or c["d"] < best["d"]:
            best = c
    TARGET_INFO["candidates"] = [[c["prompt"], c["rank"], c["score"], r4(c["x"]), r4(c["y"]), r4(c["ztop"]),
                                  r4(c["ext"]), c["color"], c["plausible"], r4(c.get("d"))] for c in cands]
    CHOSEN = None
    # v01: identical neighbours can touch the target (episode 0: 4 cm apart), so a SAM mask only
    # replaces the tracked depth blob when it is within 2 cm of the track (v00 used 5 cm).
    if best is not None and (TARGET_XY is None or best["d"] < 0.02 or TARGET_INFO.get("color_decides", False)):
        CHOSEN = {"x": best["x"], "y": best["y"], "ztop": best["ztop"], "pts": best["pts"],
                  "src": "sam_front"}
    elif TARGET_XY is not None and TRACK_OK and TARGET["track"] is not None and \
            TRACKS[TARGET["track"]].get("last_blob") is not None:
        lb = TRACKS[TARGET["track"]]["last_blob"]
        CHOSEN = {"x": lb["x"], "y": lb["y"], "ztop": lb["zmax"], "pts": None, "src": "depth_blob"}
        TARGET_INFO["blob_was_split"] = bool(lb.get("split", False))
        TARGET_INFO["blob_n"] = lb.get("n")
    elif best is not None:
        CHOSEN = {"x": best["x"], "y": best["y"], "ztop": best["ztop"], "pts": best["pts"],
                  "src": "sam_front_far"}
    elif TARGET_XY is not None:
        CHOSEN = {"x": TARGET_XY[0], "y": TARGET_XY[1], "ztop": None, "pts": None, "src": "expected_only"}
    if CHOSEN is not None:
        TARGET_INFO["chosen"] = [r4(CHOSEN["x"]), r4(CHOSEN["y"]), r4(CHOSEN["ztop"]), CHOSEN["src"]]
    say("selection " + str(TARGET_INFO.get("chosen")) + " expected " + str(TARGET_INFO.get("expected_xy")))
except Exception as exc:
    err("select", exc)
    CHOSEN = None
report["target"] = TARGET_INFO
report["split"] = SPLIT

# ---------------------------------------------------------------- Phase G: pick
DOWN_Q = [0.0, 1.0, 0.0, 0.0]
try:
    if HOME_QUAT is not None and quat_to_mat(HOME_QUAT)[2][2] < -0.9:
        DOWN_Q = HOME_QUAT
    elif TARGET["seg"] is not None and len(SEGS) > 0:
        qv = VQ[SEGS[TARGET["seg"]]["f0"]]
        if quat_to_mat(qv)[2][2] < -0.9:
            DOWN_Q = qv
except Exception as exc:
    err("down quat", exc)
report["down_quat"] = [r4(v) for v in DOWN_Q]


def yaw_from_points(pts):
    """Min-area-rectangle yaw (mod 90 deg) of the top-face points; None if ambiguous."""
    if pts is None or len(pts) < 20:
        return None
    zt = quantile([p[2] for p in pts], 0.9)
    top = [p for p in pts if p[2] > zt - 0.012]
    if len(top) < 15:
        return None
    mx = sum([p[0] for p in top]) / len(top)
    my = sum([p[1] for p in top]) / len(top)
    best = None
    worst = None
    for k in range(18):
        a = k * math.pi / 36.0
        ca = math.cos(a)
        sa = math.sin(a)
        us = [(p[0] - mx) * ca + (p[1] - my) * sa for p in top]
        vs3 = [-(p[0] - mx) * sa + (p[1] - my) * ca for p in top]
        area = (quantile(us, 0.97) - quantile(us, 0.03)) * (quantile(vs3, 0.97) - quantile(vs3, 0.03))
        if best is None or area < best[0]:
            best = (area, a)
        if worst is None or area > worst:
            worst = area
    if worst is None or worst <= 0 or best[0] / worst > 0.93:
        return None
    return best[1]


def grasp_quat(yaw_cube):
    if yaw_cube is None:
        return DOWN_Q
    Rb = quat_to_mat(DOWN_Q)
    phi = math.atan2(Rb[1][0], Rb[0][0])
    d = yaw_cube - phi
    while d > math.pi / 4:
        d = d - math.pi / 2
    while d < -math.pi / 4:
        d = d + math.pi / 2
    return quat_mul(quat_about_z(d), DOWN_Q)


def move(label, pos, quat, tol=0.012, max_steps=200):
    if FLAGS["terminal"]:
        return None
    try:
        pose = Pose(position=[float(pos[0]), float(pos[1]), float(pos[2])],
                    quaternion_wxyz=[float(q) for q in quat], frame=BASE)
        res = move_to_pose(pose, tolerance=tol, max_steps=max_steps)
        s = note_motion(label, res)
        if (not s["ok"]) and int(s.get("steps", 0) or 0) == 0 and not s["terminal"]:
            res = move_to_pose(pose, tolerance=tol, max_steps=max_steps * 2)
            note_motion(label + "_retry", res)
        return res
    except Exception as exc:
        err("move " + label, exc)
        return None


def ee_now():
    st = get_robot_state()
    return vec3(st.end_effector_poses["primary"].position), float(st.gripper_positions["primary"])


def wrist_cloud(center, radius, z_lo, z_hi):
    """R2: base-frame points from the WRIST depth image (same calibrated convention as the front
    camera). Returns (points within `radius` of `center` and z_lo<z<z_hi, near-camera points,
    camera xy)."""
    obs = get_observation()
    cam = obs.cameras[WRIST]
    m = cam_model(cam)
    depth = cam.depth_m
    t = m["t"]
    grid = {}
    near = []
    for v in range(0, m["H"], 2):
        drow = row_list(depth, v)
        for u in range(0, m["W"], 2):
            d = float(drow[u])
            if d <= 0.005 or d > 3.0:
                continue
            rd = ray_dir(m, CONV, u + 0.5, v + 0.5)
            x = t[0] + d * rd[0]
            y = t[1] + d * rd[1]
            z = t[2] + d * rd[2]
            if d < 0.10:
                near.append((x, y, z))
                continue
            if z < z_lo or z > z_hi:
                continue
            if center is not None and dist2((x, y), center) > radius:
                continue
            grid[(u // 2, v // 2)] = (x, y, z)
    return grid, near, (t[0], t[1])


def components(grid, link):
    seen = set()
    comps = []
    for key in grid:
        if key in seen:
            continue
        seen.add(key)
        stack = [key]
        mem = []
        while len(stack) > 0:
            k = stack.pop()
            p = grid[k]
            mem.append(p)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nk = (k[0] + di, k[1] + dj)
                    if nk in seen or nk not in grid:
                        continue
                    q = grid[nk]
                    if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2 < link * link:
                        seen.add(nk)
                        stack.append(nk)
        if len(mem) >= 8:
            comps.append(mem)
    return comps


def split_wide(comp, depth_left):
    """R1 in the wrist view: a component wider than one object is split with 2-means."""
    lim = 0.055
    if SPLIT["single_ext"] is not None:
        lim = max(lim, 1.4 * SPLIT["single_ext"])
    if depth_left <= 0 or len(comp) < 16 or xy_extent(comp) < lim:
        return [comp]
    A, B, sep = two_means(comp)
    if len(B) == 0 or sep < 0.022 or min(len(A), len(B)) < 0.25 * len(comp):
        return [comp]
    return split_wide(A, depth_left - 1) + split_wide(B, depth_left - 1)


def centroid(pts):
    n = len(pts)
    return (sum([p[0] for p in pts]) / n, sum([p[1] for p in pts]) / n)


def principal_angle(pts):
    mx, my = centroid(pts)
    sxx = sum([(p[0] - mx) ** 2 for p in pts])
    syy = sum([(p[1] - my) ** 2 for p in pts])
    sxy = sum([(p[0] - mx) * (p[1] - my) for p in pts])
    return 0.5 * math.atan2(2.0 * sxy, sxx - syy)


def wrap_half_pi(a):
    while a > math.pi / 2:
        a = a - math.pi
    while a <= -math.pi / 2:
        a = a + math.pi
    return a


FINGER = {"axis": None, "src": None}


def finger_axis(near, ee_xy):
    """Closing axis of the open fingers, measured from the wrist depth (points < 10 cm from the
    camera split into the two fingers). Falls back to the hand y axis of DOWN_Q."""
    R0 = quat_to_mat(DOWN_Q)
    hand = wrap_half_pi(math.atan2(R0[1][1], R0[0][1]))
    try:
        if len(near) >= 12:
            A, B, sep = two_means(near)
            if len(B) >= 4 and sep > 0.03:
                ca = centroid(A)
                cb = centroid(B)
                meas = wrap_half_pi(math.atan2(cb[1] - ca[1], cb[0] - ca[0]))
                # episode-0 wrist keyframes show the fingers left/right of the image, i.e. along
                # the hand y axis; a measurement is only trusted when it agrees within 25 deg.
                if abs(wrap_half_pi(meas - hand)) < 25.0 * math.pi / 180.0:
                    return meas, "wrist_depth_fingers"
                return hand, "hand_y_axis(meas_disagreed:" + str(r4(meas * 180.0 / math.pi)) + ")"
    except Exception as exc:
        err("finger axis", exc)
    return hand, "hand_y_axis"


def quat_for_closing(angle):
    """Top-down quaternion whose finger closing axis points along `angle` (mod pi)."""
    phi0 = FINGER["axis"]
    if phi0 is None:
        R0 = quat_to_mat(DOWN_Q)
        phi0 = math.atan2(R0[1][1], R0[0][1])
    d = wrap_half_pi(angle - phi0)
    return quat_mul(quat_about_z(d), DOWN_Q), d


def grasp_candidates(target, others, gz):
    """R2: ranked (point, closing angle) grasps on the target's own material.
    Grasp points are local density maxima (the thick back of a crescent, never its hollow);
    the closing axis is across the local rim (minor principal axis), adjusted so the finger
    landing zones are free of material higher than the grasp height."""
    P = target
    dens = []
    for i in range(len(P)):
        p = P[i]
        c = 0
        for q in P:
            if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < 0.010 * 0.010:
                c = c + 1
        dens.append((c, i))
    dens = sorted(dens, reverse=True)
    picks = []
    for (c, i) in dens:
        p = P[i]
        ok = True
        for (pp, cc) in picks:
            if dist2(pp, p) < 0.012:
                ok = False
        if ok:
            picks.append((p, c))
        if len(picks) >= 3:
            break
    allpts = [(q, True) for q in P] + [(q, False) for q in others]
    out = []
    for (g, c) in picks:
        local = [q for q in P if dist2(q, g) < 0.02]
        if len(local) < 5:
            continue
        major = principal_angle(local)
        for dk in (0, 1, -1, 2, -2, 3):
            delta = dk * math.pi / 6.0
            ang = major + math.pi / 2.0 + delta
            ca = math.cos(ang)
            sa = math.sin(ang)
            span_s = []
            obst = 0
            for (q, own) in allpts:
                dx = q[0] - g[0]
                dy = q[1] - g[1]
                s = dx * ca + dy * sa
                w = -dx * sa + dy * ca
                if abs(w) < 0.005 and abs(s) < 0.035 and own:
                    span_s.append(s)
                if abs(w) < 0.012 and 0.035 <= abs(s) <= 0.062 and q[2] > gz + 0.004:
                    obst = obst + 1
            span = (max(span_s) - min(span_s)) if len(span_s) > 0 else 0.0
            score = 1.0 * obst + 20.0 * span + 0.3 * abs(delta) - 0.01 * c
            if span > 0.06:
                score = score + 50.0
            out.append({"xy": (g[0], g[1]), "zmax": max([q[2] for q in local]), "angle": ang,
                        "obst": obst, "span": span, "dens": c, "score": score})
    out = sorted(out, key=score_of)
    return out


def score_of(c):
    return c["score"]


def wrist_look(xy, gz, exclude):
    """R2: move so the wrist camera is straight above `xy`, segment the target from wrist depth
    and return ranked grasp candidates (excluding failed ones)."""
    out = {"used": False}
    try:
        st = get_robot_state()
        ee = vec3(st.end_effector_poses["primary"].position)
        obs = get_observation()
        cpos = vec3(obs.cameras[WRIST].camera_pose.position)
        off = (cpos[0] - ee[0], cpos[1] - ee[1])
        out["cam_offset"] = [r4(off[0]), r4(off[1])]
        move("wrist_look", (xy[0] - off[0], xy[1] - off[1], gz + 0.17), DOWN_Q, tol=0.015, max_steps=250)
        if FLAGS["terminal"]:
            return None, out
        hold_ticks(2, 1.0)
        grid, near, cxy = wrist_cloud(xy, 0.11, TABLE_Z + 0.006, TABLE_Z + 0.075)
        st = get_robot_state()
        ee = vec3(st.end_effector_poses["primary"].position)
        ax, src = finger_axis(near, (ee[0], ee[1]))
        FINGER["axis"] = ax
        FINGER["src"] = src
        out["finger_axis_deg"] = r4(ax * 180.0 / math.pi)
        out["finger_src"] = src
        out["n_pts"] = len(grid)
        comps = []
        for cpt in components(grid, 0.008):
            comps = comps + split_wide(cpt, 2)
        out["comps"] = [[r4(centroid(c)[0]), r4(centroid(c)[1]), len(c), r4(xy_extent(c))] for c in comps]
        best = None
        for ci in range(len(comps)):
            dd = dist2(centroid(comps[ci]), xy)
            if dd < 0.03 and (best is None or dd < best[0]):
                best = (dd, ci)
        if best is None:
            return None, out
        target = comps[best[1]]
        others = []
        for ci in range(len(comps)):
            if ci != best[1]:
                others = others + comps[ci]
        out["target_comp"] = [r4(centroid(target)[0]), r4(centroid(target)[1]), len(target), r4(best[0])]
        cands = grasp_candidates(target, others, gz)
        keep = []
        for c in cands:
            bad = False
            for (fxy, fang) in exclude:
                if dist2(c["xy"], fxy) < 0.008 and abs(wrap_half_pi(c["angle"] - fang)) < 0.35:
                    bad = True
            if not bad:
                keep.append(c)
        out["cands"] = [[r4(c["xy"][0]), r4(c["xy"][1]), r4(c["angle"] * 180.0 / math.pi), c["obst"],
                         r4(c["span"]), c["dens"], r4(c["score"])] for c in keep[:6]]
        out["used"] = len(keep) > 0
        out["centroid"] = [r4(v) for v in centroid(target)]
        return keep, out
    except Exception as exc:
        err("wrist look", exc)
        return None, out


def grasp_height(ztop):
    gz = None
    try:
        if TARGET["seg"] is not None:
            vz = float(SEGS[TARGET["seg"]]["z_start"])
            if TABLE_Z is None or (TABLE_Z - 0.005 < vz < TABLE_Z + 0.07):
                gz = vz
    except Exception:
        gz = None
    if gz is None and ztop is not None:
        gz = ztop - 0.02
        if TABLE_Z is not None:
            gz = max(gz, TABLE_Z + 0.012)
    if gz is None and TABLE_Z is not None:
        gz = TABLE_Z + 0.02
    if gz is None:
        gz = 0.02
    if TABLE_Z is not None:
        gz = max(gz, TABLE_Z + 0.008)
    return gz


PICKS = []
FAILED = []
HELD_MIN = 0.05   # v01: the video hold reads 0.1327; v00's 0.03 accepted an empty 0.0301 close


def pick_once(xy, ztop, attempt):
    info = {"attempt": attempt, "xy_in": [r4(xy[0]), r4(xy[1])]}
    PICKS.append(info)
    gz = grasp_height(ztop)
    info["grasp_z"] = r4(gz)
    try:
        note_motion("open_pre", open_gripper())
    except Exception as exc:
        err("open_pre", exc)
    cands, wl = wrist_look(xy, gz, FAILED)
    info["wrist"] = wl
    if FLAGS["terminal"]:
        return info
    x = xy[0]
    y = xy[1]
    q = DOWN_Q
    ang = None
    if cands is not None and len(cands) > 0:
        c = cands[0]
        x = c["xy"][0]
        y = c["xy"][1]
        ang = c["angle"]
        q, d = quat_for_closing(ang)
        info["grasp_src"] = "wrist_material"
        info["closing_deg"] = r4(ang * 180.0 / math.pi)
        info["yaw_delta_deg"] = r4(d * 180.0 / math.pi)
        info["cand"] = [c["obst"], r4(c["span"]), c["dens"]]
        if ztop is None:
            gz = grasp_height(c["zmax"])
            info["grasp_z"] = r4(gz)
    else:
        # no wrist evidence: fall back to v00's front estimate, rotated on each retry
        ang = (attempt - 1) * math.pi / 4.0
        q, d = quat_for_closing(ang + math.pi / 2.0)
        info["grasp_src"] = "front_fallback"
    info["xy_final"] = [r4(x), r4(y)]
    info["quat"] = [r4(v) for v in q]
    move("pregrasp", (x, y, gz + 0.10), q, tol=0.012, max_steps=250)
    move("descend", (x, y, gz + 0.04), q, tol=0.01, max_steps=120)
    move("grasp", (x, y, gz), q, tol=0.008, max_steps=120)
    if FLAGS["terminal"]:
        return info
    try:
        pos, g = ee_now()
        info["ee_at_grasp"] = [r4(v) for v in pos]
    except Exception as exc:
        err("ee at grasp", exc)
    try:
        note_motion("close", close_gripper())
    except Exception as exc:
        err("close", exc)
    if FLAGS["terminal"]:
        return info
    g = 0.0
    try:
        pos, g = ee_now()
        info["gripper_after_close"] = r4(g)
    except Exception as exc:
        err("gripper read", exc)
    move("lift", (x, y, gz + 0.15), q, tol=0.02, max_steps=150)
    info["xyz"] = (x, y, gz)
    info["q"] = q
    info["held"] = g > HELD_MIN
    if not FLAGS["terminal"]:
        try:
            pos, g2 = ee_now()
            info["gripper_after_lift"] = r4(g2)
            info["ee_after_lift"] = [r4(v) for v in pos]
            info["held"] = g2 > HELD_MIN
        except Exception as exc:
            err("after lift", exc)
    if not info["held"] and ang is not None:
        FAILED.append(((x, y), ang))
    return info


def put_down(info):
    x, y, gz = info["xyz"]
    q = info["q"]
    move("place_descend", (x, y, gz + 0.012), q, tol=0.01, max_steps=150)
    try:
        note_motion("place_open", open_gripper())
    except Exception as exc:
        err("place open", exc)
    move("place_retreat", (x, y, gz + 0.13), q, tol=0.015, max_steps=150)


BUTTON = {}


def find_button():
    """R4: the button is the static, low-saturation (white/gray) object tracked from the first
    video frame; a SAM 'button' mask near it refines the cap centre and top height."""
    best = None
    for tr in TRACKS:
        if TARGET["track"] is not None and tr["id"] == TARGET["track"]:
            continue
        lb = tr.get("last_blob")
        if lb is None or tr["born"] != 0 or tr["miss"] > 2:
            continue
        col = lb["col"]
        if saturation(col) >= 40.0 or (col[0] + col[1] + col[2]) / 3.0 < 120.0:
            continue
        if tr.get("path", 0.0) > 0.15:
            continue
        sc = lb["n"] - 50.0 * tr.get("path", 0.0)
        if best is None or sc > best[0]:
            best = (sc, tr, lb)
    if best is not None:
        lb = best[2]
        top = [p for p in lb["pts"] if p[2] > lb["zmax"] - 0.008]
        if len(top) == 0:
            top = lb["pts"]
        cx, cy = centroid(top)
        BUTTON["blob"] = [r4(cx), r4(cy), r4(lb["zmax"]), best[1]["id"]]
        BUTTON["xy"] = (cx, cy)
        BUTTON["ztop"] = lb["zmax"]
        BUTTON["src"] = "depth_blob"
    try:
        seg = segment_text(FRONT, "button")
        if seg is not None and seg.ok:
            sams = []
            for ci in range(min(5, len(seg.segmentations))):
                cand = seg.segmentations[ci]
                pts = cloud_points(mask_to_point_cloud(cand, FRONT, target_frame=BASE), 400)
                if len(pts) < 8:
                    continue
                med = cloud_median(pts)
                zt = quantile([p[2] for p in pts], 0.95)
                ext = xy_extent(pts)
                sams.append([ci, r4(cand.score), r4(med[0]), r4(med[1]), r4(zt), r4(ext)])
                if TABLE_Z is not None and not (0.005 < zt - TABLE_Z < 0.10 and ext < 0.10):
                    continue
                if "xy" in BUTTON and dist2(med, BUTTON["xy"]) < 0.04:
                    if BUTTON["src"] == "depth_blob":
                        BUTTON["xy"] = (med[0], med[1])
                        BUTTON["ztop"] = max(zt, BUTTON["ztop"])
                        BUTTON["src"] = "sam_button+depth_blob"
                elif "xy" not in BUTTON:
                    BUTTON["xy"] = (med[0], med[1])
                    BUTTON["ztop"] = zt
                    BUTTON["src"] = "sam_button_only"
            BUTTON["sam"] = sams
    except Exception as exc:
        err("button sam", exc)
    if "xy" in BUTTON:
        BUTTON["chosen"] = [r4(BUTTON["xy"][0]), r4(BUTTON["xy"][1]), r4(BUTTON["ztop"]), BUTTON["src"]]
    return "xy" in BUTTON


def press_button():
    if not find_button():
        say("no button found; not pressing")
        return False
    bx, by = BUTTON["xy"]
    zt = BUTTON["ztop"]
    try:
        note_motion("press_close", close_gripper())
    except Exception as exc:
        err("press close", exc)
    move("press_above", (bx, by, zt + 0.10), DOWN_Q, tol=0.015, max_steps=250)
    move("press_approach", (bx, by, zt + 0.03), DOWN_Q, tol=0.01, max_steps=120)
    move("press_push", (bx, by, zt - 0.012), DOWN_Q, tol=0.006, max_steps=60)
    if not FLAGS["terminal"]:
        hold_ticks(4, 0.0)
    move("press_retreat", (bx, by, zt + 0.10), DOWN_Q, tol=0.02, max_steps=120)
    BUTTON["pressed"] = True
    return True


try:
    report["phase"] = "G_pick"
    n_picks = REPEATS
    if SAME_AS_VIDEO_COUNT and TARGET["track"] is not None:
        n_picks = max(1, len(SEGS))
    n_picks = max(1, min(5, n_picks))
    report["n_picks_planned"] = n_picks
    if CHOSEN is None:
        say("no target chosen; nothing to pick")
    else:
        xy = (CHOSEN["x"], CHOSEN["y"])
        ztop = CHOSEN["ztop"]
        done = 0
        attempts = 0
        while done < n_picks and attempts < n_picks + 3 and not FLAGS["terminal"]:
            attempts = attempts + 1
            info = pick_once(xy, ztop, attempts)
            if FLAGS["terminal"]:
                break
            if not info.get("held", False):
                say("grasp looks empty (gripper " + str(info.get("gripper_after_close")) +
                    "); retrying with a different grasp")
                try:
                    note_motion("open_after_miss", open_gripper())
                except Exception as exc:
                    err("open after miss", exc)
                if "xyz" in info:
                    move("retreat_after_miss", (info["xyz"][0], info["xyz"][1], info["xyz"][2] + 0.13),
                         info["q"], tol=0.02, max_steps=150)
                wl = info.get("wrist", {})
                if wl.get("centroid") is not None:
                    xy = (float(wl["centroid"][0]), float(wl["centroid"][1]))
                continue
            done = done + 1
            hold_ticks(8, 0.0)
            if FLAGS["terminal"]:
                break
            if done < n_picks or PUT_DOWN:
                put_down(info)
                wl = info.get("wrist", {})
                if wl.get("centroid") is not None:
                    xy = (float(wl["centroid"][0]), float(wl["centroid"][1]))
                ztop = None
                FAILED = []
                hold_ticks(4, 1.0)
        report["picks_done"] = done
    if PRESS_BUTTON and not FLAGS["terminal"]:
        report["phase"] = "H_button"
        press_button()
    if not FLAGS["terminal"]:
        hold_ticks(10, 1.0)
except Exception as exc:
    err("pick phase", exc)
report["button"] = BUTTON
for p in PICKS:
    if "xyz" in p:
        p["xyz"] = [r4(v) for v in p["xyz"]]
    if "q" in p:
        p["q"] = [r4(v) for v in p["q"]]
report["picks"] = PICKS
report["terminal"] = FLAGS["terminal"]
report["terminal_label"] = FLAGS["terminal_label"]
report["ticks_counted"] = FLAGS["ticks"]
report["phase"] = "done"
say("done; terminal=" + str(FLAGS["terminal"]) + " at " + str(FLAGS["terminal_label"]))
result = report
