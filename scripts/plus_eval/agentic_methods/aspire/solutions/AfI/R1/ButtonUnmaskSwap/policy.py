# v04_button_containment -- ButtonUnmaskSwap (robomme_plus task 2), AfI.
# UNTESTED. v03 PLUS a second perception repair: v02/v03 compared a button-prompt mask with
# cube-prompt masks by IoU > 0.5. Replaying that rule on the episode-0 recorded candidates
# (offline_test/replay_inventory_ep0.py) leaves a partial "red button" sliver of the red cube
# filed as a button, which still blocks the red target cube watch and its tracker seed.
# v04 uses the overlap over the SMALLER box (containment) instead of IoU; on the replay only
# the two real buttons remain buttons.
# --- v03 header follows ---
# v03_horizon480 -- ButtonUnmaskSwap (robomme_plus task 2), AfI.
# UNTESTED. v02 PLUS mode "timing/horizon": docs/robomme-protocol-v2/PUBLIC_RULES.md states a
# separate 480-step motion limit for ButtonUnmaskSwap, but v00 watched until MAX_STEPS-170
# (1312-170 = 1142 on episode 0) and reserved only 170 steps for three pick-and-put-back
# cycles (about 200 steps in the offline mock). v03 plans against min(max_steps, 480) and
# reserves 20 + 90 steps per target container; a pick retry keeps the put-back flag.
# --- v02 header follows ---
# v02_buttons_vs_cubes -- ButtonUnmaskSwap (robomme_plus task 2), AfI.
# UNTESTED. v01 (press both buttons once, clean press) PLUS mode "perception: button/cube
# confusion": in v00 the "button"/"red button" prompts also returned the black and the red
# cube (low scores, 0.06), these were filed as buttons, and every cube/seed within a button
# box was then discarded, so the red target cube (third in the language order) was never
# watched or tracked. v02 files a button-prompt candidate as a cube when a cube-prompt mask
# on the same box scores higher than its button score (relative, episode-derived test).
# --- v01 header follows ---
# v01_press_both_clean -- ButtonUnmaskSwap (robomme_plus task 2), AfI.
# UNTESTED trace-guided repair of v00 (v00: 0/1 on episode 0; episode terminated at step 65,
# right after the first and only button press was released; the language asked for BOTH buttons).
# Changes vs v00 (mode "press protocol"):
#   * parse the required number of presses ("both"/"two"/"each"/... buttons) from the language;
#   * press every required button exactly once, nearest first, inside the step-180 deadline;
#   * one clean press per button: keep commanding the depressed target during the hold (v00
#     relaxed to the measured joints, so the cap pushed the finger back up 2.2 mm while still
#     in contact), then retract straight up once and never re-contact the cap.
# Everything else is unchanged from v00.
#
# This program is BOTH the scene snapshot and the attempt (R=1, no separate probe start).
#   Phase A  record language, task/controller metadata, cameras, robot state (no ticks).
#   Phase B  summarise the conditioning video (length, timestamps, colour masses).
#   Phase C  SAM3 prompt probes on front + wrist; calibrate pixel<->robot_base projection
#            against mask_to_point_cloud; fit the table plane in front depth.
#   Phase D  parse the goal language (press button? which cube colour(s)?).
#   Phase E  find the target cube (live if visible, else from the video) and seed a
#            per-tick front-camera depth tracker over the containers.
#   Phase F  press the button early (public deadline: before control step 180), with
#            per-tick joint stepping so the tracker sees every frame.
#   Phase G  return to the start pose and watch until the containers stop moving.
#   Phase H  pick up the container that the tracker says hides the target cube.
# Every phase is wrapped in try/except; the program must never raise.
# Only public APIs; no hardcoded coordinates, pixels or seed-specific branches.

import math

result = {"version": "v04_button_containment", "phase": "start", "errors": [], "events": []}
R = result
S = {"tick": 0, "done": False, "step_fail": 0, "obs": None, "term": None,
     "step_errors": 0, "track_on": False}


def log_err(where, e):
    msg = str(where) + ": " + str(e)
    if len(R["errors"]) < 80:
        R["errors"].append(msg[:500])
    print("[v04][ERR] t=" + str(S["tick"]) + " " + msg[:300])


def ev(msg):
    m = "t=" + str(S["tick"]) + " " + str(msg)
    if len(R["events"]) < 400:
        R["events"].append(m[:500])
    print("[v04] " + m[:300])


def r4(x):
    try:
        return round(float(x), 4)
    except Exception:
        return None


def quant(vals, f):
    o = sorted(vals)
    if len(o) == 0:
        return None
    return o[int((len(o) - 1) * f)]


def jsonable(x, depth=0):
    if x is None or x is True or x is False:
        return x
    try:
        if str(x) == x:
            return x[:400]
    except Exception:
        pass
    try:
        f = float(x)
        if f != f:
            return None
        s = str(x)
        if ("." not in s) and ("e" not in s) and abs(f) < 1e15:
            try:
                return int(x)
            except Exception:
                return f
        return f
    except Exception:
        pass
    if depth > 3:
        return str(x)[:200]
    try:
        keys = list(x.keys())
        out = {}
        for k in keys[:60]:
            out[str(k)] = jsonable(x[k], depth + 1)
        return out
    except Exception:
        pass
    try:
        n = len(x)
        out = []
        for i in range(min(n, 40)):
            out.append(jsonable(x[i], depth + 1))
        return out
    except Exception:
        pass
    return str(x)[:200]


def vec(a):
    return [float(v) for v in a]


def ok_of(x):
    try:
        return bool(x.ok)
    except Exception:
        return False


def diag_of(x):
    out = {}
    try:
        out["status"] = str(x.status)
    except Exception:
        pass
    try:
        out["error"] = str(x.error)[:300]
    except Exception:
        pass
    try:
        out["steps"] = int(x.steps_executed)
    except Exception:
        pass
    try:
        out["diag"] = str(x.diagnostics)[:300]
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- geometry helpers
def qmat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        n = 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def qyaw(yaw):
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def cam_model(c):
    K = c.intrinsics
    p = c.camera_pose
    return {"fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]),
            "cy": float(K[1][2]), "t": vec(p.position), "R": qmat(vec(p.quaternion_wxyz))}


def world_to_px(cm, p, conv):
    d = [p[0] - cm["t"][0], p[1] - cm["t"][1], p[2] - cm["t"][2]]
    Rm = cm["R"]
    pc = [Rm[0][0] * d[0] + Rm[1][0] * d[1] + Rm[2][0] * d[2],
          Rm[0][1] * d[0] + Rm[1][1] * d[1] + Rm[2][1] * d[2],
          Rm[0][2] * d[0] + Rm[1][2] * d[1] + Rm[2][2] * d[2]]
    if conv == "cv":
        x, y, z = pc[0], pc[1], pc[2]
    elif conv == "gl":
        x, y, z = pc[0], -pc[1], -pc[2]
    else:
        x, y, z = -pc[1], -pc[2], pc[0]
    if z <= 1e-4:
        return None
    return [cm["fx"] * x / z + cm["cx"], cm["fy"] * y / z + cm["cy"], z]


def px_to_world(cm, u, v, dep, conv):
    x = (u - cm["cx"]) / cm["fx"] * dep
    y = (v - cm["cy"]) / cm["fy"] * dep
    z = dep
    if conv == "cv":
        pc = [x, y, z]
    elif conv == "gl":
        pc = [x, -y, -z]
    else:
        pc = [z, -x, -y]
    Rm = cm["R"]
    return [Rm[0][0] * pc[0] + Rm[0][1] * pc[1] + Rm[0][2] * pc[2] + cm["t"][0],
            Rm[1][0] * pc[0] + Rm[1][1] * pc[1] + Rm[1][2] * pc[2] + cm["t"][1],
            Rm[2][0] * pc[0] + Rm[2][1] * pc[1] + Rm[2][2] * pc[2] + cm["t"][2]]


# ---------------------------------------------------------------- colour helpers
def hsv(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx <= 0:
        return 0.0, 0.0, 0.0
    s = (mx - mn) / float(mx)
    if mx == mn:
        h = 0.0
    elif mx == r:
        h = (60.0 * (g - b) / (mx - mn)) % 360.0
    elif mx == g:
        h = 60.0 * (b - r) / (mx - mn) + 120.0
    else:
        h = 60.0 * (r - g) / (mx - mn) + 240.0
    return h, s, mx / 255.0


def color_class(r, g, b):
    h, s, v = hsv(r, g, b)
    if v < 0.18:
        return "black"
    if s < 0.28:
        if v > 0.75:
            return "white"
        return "gray"
    if h < 14 or h >= 340:
        return "red"
    if h < 40:
        if v < 0.45:
            return "brown"
        return "orange"
    if h < 70:
        return "yellow"
    if h < 165:
        return "green"
    if h < 195:
        return "cyan"
    if h < 258:
        return "blue"
    if h < 292:
        return "purple"
    return "pink"


COLOR_WORDS = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "cyan",
               "white", "black", "gray", "grey", "brown", "magenta", "violet"]
COLOR_ALIAS = {"grey": "gray", "magenta": "pink", "violet": "purple"}


NOMINAL_HUE = {"red": 0.0, "orange": 28.0, "yellow": 55.0, "green": 120.0, "cyan": 180.0,
               "blue": 225.0, "purple": 275.0, "pink": 320.0}


def hue_close(rgb, cname, tol=35.0):
    if cname not in NOMINAL_HUE:
        return False
    h, s, v = hsv(rgb[0], rgb[1], rgb[2])
    if s < 0.22 or v < 0.15:
        return False
    d = abs(h - NOMINAL_HUE[cname]) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return d < tol


def color_match(px, cname, model):
    r, g, b = px[0], px[1], px[2]
    if model is not None:
        dist = abs(r - model[0]) + abs(g - model[1]) + abs(b - model[2])
        if dist < 75:
            return True
        return False
    return color_class(r, g, b) == cname


HEXD = "0123456789abcdef"
SNAPS = []


def snapshot(tag, obs=None, camera=None, with_depth=False):
    """Store a 64x64 hex-encoded RGB thumbnail (and optional depth in mm) in result."""
    if len(SNAPS) >= 16:
        return
    try:
        if obs is None:
            obs = S["obs"]
        if obs is None:
            obs = get_observation()
        if camera is None:
            camera = FRONT
        cam = obs.cameras[camera]
        img = cam.rgb[0:IMG_H:4, 0:IMG_W:4].tolist()
        rows = []
        for row in img:
            chars = []
            for px in row:
                for ch in px[:3]:
                    vi = int(ch)
                    chars.append(HEXD[vi // 16] + HEXD[vi % 16])
            rows.append("".join(chars))
        rec = {"tag": tag, "tick": S["tick"], "camera": camera, "step": 4, "rgb_hex_rows": rows}
        if with_depth:
            dep = cam.depth_m[0:IMG_H:4, 0:IMG_W:4].tolist()
            drows = []
            for row in dep:
                vals = []
                for d in row:
                    try:
                        vals.append(int(float(d) * 1000))
                    except Exception:
                        vals.append(int(float(d[0]) * 1000))
                drows.append(vals)
            rec["depth_mm"] = drows
        SNAPS.append(rec)
        R["snapshots"] = SNAPS
    except Exception as e:
        log_err("snapshot " + str(tag), e)


def dep_at(dep_rows, u, v):
    try:
        vi = int(min(len(dep_rows) - 1, max(0, round(v))))
        ui = int(min(len(dep_rows[0]) - 1, max(0, round(u))))
        return float(dep_rows[vi][ui])
    except Exception:
        return 1.0


# ---------------------------------------------------------------- phase A: metadata
LANG = ""
MAX_STEPS = 480
EMB = "robomme_plus"
BASE = "robot_base"
FRONT = "front"
WRIST = "wrist"
Q_HOME = None
QUAT_DOWN = [0.0, 1.0, 0.0, 0.0]
HOME_EE = None
try:
    R["phase"] = "A_metadata"
    ctx = get_task_context()
    LANG = str(ctx.language)
    R["language"] = LANG
    try:
        R["task_name"] = str(ctx.task_name)
    except Exception:
        pass
    try:
        R["context_metadata"] = jsonable(ctx.metadata)
    except Exception:
        pass
    ev("language: " + LANG)
except Exception as e:
    log_err("context", e)
try:
    md = robomme_plus.get_task_metadata()
    R["task_metadata"] = jsonable(md)
    try:
        MAX_STEPS = int(md["max_steps"])
    except Exception as e:
        log_err("max_steps", e)
    try:
        if len(LANG) == 0:
            LANG = str(md["language"])
            R["language"] = LANG
    except Exception:
        pass
except Exception as e:
    log_err("task_metadata", e)
try:
    R["controller_metadata"] = jsonable(robomme_plus.get_controller_metadata())
except Exception as e:
    log_err("controller_metadata", e)
R["max_steps"] = MAX_STEPS

OBS0 = None
try:
    st0 = get_robot_state()
    BASE = str(st0.base_frame)
    EMB = str(st0.embodiment)
    Q_HOME = vec(st0.joint_positions["primary"])[:7]
    eep = st0.end_effector_poses["primary"]
    HOME_EE = vec(eep.position)
    hq = vec(eep.quaternion_wxyz)
    Rh = qmat(hq)
    R["robot_state0"] = {"base_frame": BASE, "embodiment": EMB, "q": [r4(v) for v in Q_HOME],
                         "ee_pos": [r4(v) for v in HOME_EE], "ee_quat": [r4(v) for v in hq],
                         "gripper": r4(st0.gripper_positions["primary"]),
                         "timestamp_s": r4(st0.timestamp_s),
                         "ee_z_axis": [r4(Rh[0][2]), r4(Rh[1][2]), r4(Rh[2][2])]}
    if Rh[2][2] < -0.9:
        QUAT_DOWN = hq
        ev("home EE points down; using its quaternion")
    else:
        ev("home EE not pointing down (z-axis z=" + str(r4(Rh[2][2])) + "); using [0,1,0,0]")
except Exception as e:
    log_err("robot_state0", e)

CAMS = []
try:
    OBS0 = get_observation()
    for name in OBS0.cameras:
        CAMS.append(str(name))
    R["cameras"] = {}
    for name in CAMS:
        c = OBS0.cameras[name]
        info = {}
        try:
            info["rgb_shape"] = jsonable(c.rgb.shape)
            info["depth_shape"] = jsonable(c.depth_m.shape)
            info["intrinsics"] = jsonable(c.intrinsics)
            info["pose_pos"] = [r4(v) for v in c.camera_pose.position]
            info["pose_quat"] = [r4(v) for v in c.camera_pose.quaternion_wxyz]
            info["pose_frame"] = str(c.camera_pose.frame)
        except Exception as e:
            log_err("caminfo " + name, e)
        R["cameras"][name] = info
    if FRONT not in CAMS and len(CAMS) > 0:
        FRONT = CAMS[0]
    if WRIST not in CAMS and len(CAMS) > 1:
        WRIST = CAMS[1]
    R["obs0_timestamp"] = r4(OBS0.timestamp_s)
except Exception as e:
    log_err("observation0", e)

IMG_H = 256
IMG_W = 256
try:
    shp = OBS0.cameras[FRONT].rgb.shape
    IMG_H = int(shp[0])
    IMG_W = int(shp[1])
except Exception as e:
    log_err("img shape", e)


def cur_state():
    try:
        return S["obs"].robot_state
    except Exception:
        pass
    try:
        return get_robot_state()
    except Exception:
        return None


def cur_q():
    st = cur_state()
    try:
        return vec(st.joint_positions["primary"])[:7]
    except Exception:
        try:
            return vec(get_robot_state().joint_positions["primary"])[:7]
        except Exception:
            return list(Q_HOME)


def cur_ee():
    st = cur_state()
    try:
        return vec(st.end_effector_poses["primary"].position)
    except Exception:
        return None


def cur_grip():
    st = cur_state()
    try:
        return float(st.gripper_positions["primary"])
    except Exception:
        return None


# ---------------------------------------------------------------- phase D: language
TARGET_COLORS = []
BUTTON_COLOR = None
PRESS_NEEDED = True
PRESS_COUNT = 1
try:
    R["phase"] = "D_language"
    low = LANG.lower()
    for ch in [",", ".", ";", ":", "!", "?", "(", ")", "'", '"']:
        low = low.replace(ch, " ")
    words = [w for w in low.split(" ") if len(w) > 0]
    for i, w in enumerate(words):
        if w in COLOR_WORDS:
            cname = w
            if cname in COLOR_ALIAS:
                cname = COLOR_ALIAS[cname]
            nxt = " ".join(words[i + 1:i + 3])
            if "button" in nxt:
                BUTTON_COLOR = cname
            elif cname not in TARGET_COLORS:
                TARGET_COLORS.append(cname)
    PRESS_NEEDED = ("button" in low) or ("press" in low)
    NUMW = {"one": 1, "once": 1, "single": 1, "both": 2, "two": 2, "twice": 2, "three": 3,
            "four": 4, "each": 0, "every": 0, "all": 0}
    for i, w in enumerate(words):
        if w in ["buttons", "button"]:
            for j in range(max(0, i - 3), i):
                if words[j] in NUMW and words[j] not in ["once", "twice"]:
                    PRESS_COUNT = NUMW[words[j]]
            if w == "buttons" and PRESS_COUNT == 1:
                PRESS_COUNT = 2
    R["parsed"] = {"target_colors": TARGET_COLORS, "button_color": BUTTON_COLOR,
                   "press_needed": PRESS_NEEDED, "press_count": PRESS_COUNT, "words": words[:60]}
    ev("parsed targets=" + str(TARGET_COLORS) + " button_color=" + str(BUTTON_COLOR) +
       " press=" + str(PRESS_NEEDED) + " press_count=" + str(PRESS_COUNT))
except Exception as e:
    log_err("language parse", e)


# ---------------------------------------------------------------- perception helpers
SEGS = {}


def seg_text(camera, prompt, keep=6):
    out = []
    try:
        s = segment_text(camera, prompt)
    except Exception as e:
        log_err("segment_text " + camera + " " + prompt, e)
        return out
    if not ok_of(s):
        return out
    try:
        cands = list(s.segmentations)
    except Exception as e:
        log_err("segmentations", e)
        return out
    for i, c in enumerate(cands[:keep]):
        item = {"i": i, "cand": c, "score": None, "box": None}
        try:
            item["score"] = float(c.score)
        except Exception:
            pass
        try:
            if c.box_xyxy is not None:
                item["box"] = vec(c.box_xyxy)
        except Exception:
            pass
        out.append(item)
    return out


def seg_points(camera, u, v):
    try:
        s = segment_points(camera, [[float(u), float(v)]])
    except Exception as e:
        log_err("segment_points " + camera, e)
        return None
    if not ok_of(s):
        return None
    try:
        cands = list(s.segmentations)
        if len(cands) == 0:
            return None
        item = {"i": 0, "cand": cands[0], "score": None, "box": None}
        try:
            item["score"] = float(cands[0].score)
        except Exception:
            pass
        try:
            item["box"] = vec(cands[0].box_xyxy)
        except Exception:
            pass
        return item
    except Exception as e:
        log_err("segment_points parse", e)
        return None


def cloud_of(item, camera):
    """Project a candidate mask to robot_base; returns list of [x,y,z] (subsampled) or None."""
    pc = None
    try:
        pc = mask_to_point_cloud(item["cand"].mask, camera, target_frame=BASE)
    except Exception:
        pc = None
    if pc is None:
        try:
            pc = mask_to_point_cloud(item["cand"], camera, target_frame=BASE)
        except Exception as e:
            log_err("mask_to_point_cloud", e)
            return None
    try:
        pts = pc.points.tolist()
    except Exception:
        try:
            pts = [vec(p) for p in pc.points]
        except Exception as e:
            log_err("cloud points", e)
            return None
    if len(pts) > 1500:
        stp = int(len(pts) / 1500) + 1
        pts = pts[::stp]
    good = []
    for p in pts:
        if p[0] == p[0] and abs(p[0]) < 5 and abs(p[1]) < 5 and abs(p[2]) < 5:
            good.append(p)
    return good


def cloud_stats(pts):
    if pts is None or len(pts) == 0:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    return {"n": len(pts), "med": [r4(quant(xs, 0.5)), r4(quant(ys, 0.5)), r4(quant(zs, 0.5))],
            "z05": r4(quant(zs, 0.05)), "z90": r4(quant(zs, 0.9)),
            "xr": r4(quant(xs, 0.95) - quant(xs, 0.05)), "yr": r4(quant(ys, 0.95) - quant(ys, 0.05)),
            "zr": r4(quant(zs, 0.95) - quant(zs, 0.05))}


def mask_info(item, img_rows, stp=2):
    """Pixel centroid / count / median colour of a candidate mask (pure python, subsampled)."""
    try:
        m = item["cand"].mask.tolist()
    except Exception:
        return None
    su = 0.0
    sv = 0.0
    n = 0
    rs = []
    gs = []
    bs = []
    try:
        H = len(m)
        for v in range(0, H, stp):
            row = m[v]
            W = len(row)
            for u in range(0, W, stp):
                if row[u]:
                    su += u
                    sv += v
                    n += 1
                    if img_rows is not None:
                        px = img_rows[v][u]
                        rs.append(px[0])
                        gs.append(px[1])
                        bs.append(px[2])
    except Exception as e:
        log_err("mask_info", e)
        return None
    if n == 0:
        return {"n": 0}
    out = {"n": n * stp * stp, "u": su / n, "v": sv / n}
    if len(rs) > 0:
        out["rgb"] = [quant(rs, 0.5), quant(gs, 0.5), quant(bs, 0.5)]
        out["cls"] = color_class(out["rgb"][0], out["rgb"][1], out["rgb"][2])
    return out


def box_center(b):
    return [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]


# ---------------------------------------------------------------- phase B: conditioning video
VIDEO = []
VID_N = 0
try:
    R["phase"] = "B_video"
    VIDEO = robomme_plus.get_conditioning_video()
    VID_N = len(VIDEO)
    vs = {"n": VID_N}
    if VID_N > 0:
        vs["t_first"] = r4(VIDEO[0].timestamp_s)
        vs["t_last"] = r4(VIDEO[VID_N - 1].timestamp_s)
        try:
            vs["cams"] = [str(k) for k in VIDEO[0].cameras]
            vs["front_pose0"] = [r4(v) for v in VIDEO[0].cameras[FRONT].camera_pose.position]
        except Exception as e:
            log_err("video cams", e)
        samples = []
        nsamp = min(VID_N, 40)
        for k in range(nsamp):
            idx = int(round(k * (VID_N - 1) / max(1, nsamp - 1)))
            fr = VIDEO[idx]
            rec = {"i": idx, "t": r4(fr.timestamp_s)}
            try:
                img = fr.cameras[FRONT].rgb[0:IMG_H:4, 0:IMG_W:4].tolist()
                counts = {}
                cu = {}
                cv = {}
                for iy, row in enumerate(img):
                    for ix, px in enumerate(row):
                        cl = color_class(px[0], px[1], px[2])
                        if cl in ("red", "green", "blue", "yellow", "purple", "pink", "cyan", "orange"):
                            counts[cl] = counts.get(cl, 0) + 1
                            cu[cl] = cu.get(cl, 0) + ix * 4
                            cv[cl] = cv.get(cl, 0) + iy * 4
                rec["colors"] = {}
                for cl in counts:
                    rec["colors"][cl] = [counts[cl], round(cu[cl] / counts[cl], 1),
                                         round(cv[cl] / counts[cl], 1)]
            except Exception as e:
                log_err("video frame " + str(idx), e)
            try:
                rec["ee"] = [r4(v) for v in fr.robot_state.end_effector_poses["primary"].position]
            except Exception:
                pass
            samples.append(rec)
        vs["samples"] = samples
    R["video"] = vs
    ev("conditioning video frames=" + str(VID_N))
except Exception as e:
    log_err("video", e)


# ---------------------------------------------------------------- phase C: probes + calibration
IMG0 = None
DEP0 = None
try:
    IMG0 = OBS0.cameras[FRONT].rgb.tolist()
    DEP0 = OBS0.cameras[FRONT].depth_m.tolist()
    try:
        float(DEP0[0][0])
    except Exception:
        DEP0 = [[d[0] for d in row] for row in DEP0]
except Exception as e:
    log_err("front image0", e)

snapshot("t0_front", OBS0, FRONT, True)
snapshot("t0_wrist", OBS0, WRIST, False)
CALIB = {"conv": None}
PROBE_RECORD = {}
OBJS = {"cubes": [], "containers": [], "buttons": []}
try:
    R["phase"] = "C_probes"
    front_prompts = ["button", "red button", "cube", "container", "bin", "cup", "box", "bowl"]
    if BUTTON_COLOR is not None:
        front_prompts.append(BUTTON_COLOR + " button")
    for cname in TARGET_COLORS + ["red", "green", "blue"]:
        p = cname + " cube"
        if p not in front_prompts:
            front_prompts.append(p)
    for prompt in front_prompts:
        items = seg_text(FRONT, prompt)
        SEGS[prompt] = items
        PROBE_RECORD["front|" + prompt] = [[r4(it["score"]), [round(b, 1) for b in it["box"]] if it["box"] else None]
                                           for it in items[:5]]
    for prompt in ["button", "cube", "container"]:
        items = seg_text(WRIST, prompt)
        PROBE_RECORD["wrist|" + prompt] = [[r4(it["score"]), [round(b, 1) for b in it["box"]] if it["box"] else None]
                                           for it in items[:5]]
    R["probes"] = PROBE_RECORD
    ev("probes done: " + str(len(PROBE_RECORD)))
except Exception as e:
    log_err("probes", e)

CAND_INFO = []
try:
    R["phase"] = "C_clouds"
    wanted = []
    for prompt in SEGS:
        kind = "cube"
        if "button" in prompt:
            kind = "button"
        elif "cube" not in prompt:
            kind = "container"
        for it in SEGS[prompt][:3]:
            wanted.append([prompt, kind, it])
    for rec in wanted[:40]:
        prompt, kind, it = rec[0], rec[1], rec[2]
        pts = cloud_of(it, FRONT)
        cs = cloud_stats(pts)
        mi = mask_info(it, IMG0)
        info = {"prompt": prompt, "kind": kind, "i": it["i"], "score": r4(it["score"]),
                "box": [round(b, 1) for b in it["box"]] if it["box"] else None,
                "cloud": cs, "mask": None}
        if mi is not None:
            info["mask"] = {"n": mi.get("n"), "u": r4(mi.get("u")), "v": r4(mi.get("v")),
                            "rgb": mi.get("rgb"), "cls": mi.get("cls")}
        it["pts"] = pts
        it["cs"] = cs
        it["mi"] = mi
        it["kind"] = kind
        it["prompt"] = prompt
        CAND_INFO.append(info)
    R["candidates"] = CAND_INFO
except Exception as e:
    log_err("candidate clouds", e)

# projection-convention self-calibration against mask_to_point_cloud
CM0 = None
try:
    R["phase"] = "C_calib"
    CM0 = cam_model(OBS0.cameras[FRONT])
    errs = {"cv": [], "gl": [], "ros": []}
    for prompt in SEGS:
        for it in SEGS[prompt][:3]:
            cs = it.get("cs")
            mi = it.get("mi")
            if cs is None or cs["n"] < 20:
                continue
            if mi is not None and mi.get("n", 0) > 0:
                uv = [mi["u"], mi["v"]]
            elif it["box"] is not None:
                uv = box_center(it["box"])
            else:
                continue
            for conv in ["cv", "gl", "ros"]:
                pp = world_to_px(CM0, cs["med"], conv)
                if pp is None:
                    errs[conv].append(999.0)
                else:
                    errs[conv].append(math.hypot(pp[0] - uv[0], pp[1] - uv[1]))
    best = None
    bestv = None
    summ = {}
    for conv in errs:
        if len(errs[conv]) > 0:
            m = quant(errs[conv], 0.5)
            summ[conv] = r4(m)
            if bestv is None or m < bestv:
                bestv = m
                best = conv
    CALIB["conv"] = best
    CALIB["median_px_err"] = r4(bestv)
    CALIB["all"] = summ
    if bestv is None or bestv > 25:
        ev("calibration weak: " + str(summ))
        if bestv is None:
            CALIB["conv"] = None
    R["calibration"] = CALIB
    ev("calibration conv=" + str(CALIB["conv"]) + " err=" + str(CALIB.get("median_px_err")))
except Exception as e:
    log_err("calibration", e)


def proj(cm, p):
    if CALIB["conv"] is None or cm is None:
        return None
    return world_to_px(cm, p, CALIB["conv"])


def unproj(cm, u, v, d):
    if CALIB["conv"] is None or cm is None:
        return None
    return px_to_world(cm, u, v, d, CALIB["conv"])


# robot exclusion helpers (keep robot base / arm out of object candidates)
def near_robot(p):
    try:
        if math.hypot(p[0], p[1]) < 0.22:
            return True
        ee = cur_ee()
        if ee is not None:
            if math.sqrt((p[0] - ee[0]) ** 2 + (p[1] - ee[1]) ** 2 + (p[2] - ee[2]) ** 2) < 0.09:
                return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------- object inventory
TABLE_Z = None
try:
    R["phase"] = "C_inventory"
    lows = []
    for prompt in SEGS:
        for it in SEGS[prompt][:3]:
            cs = it.get("cs")
            if cs is None or cs["n"] < 15:
                continue
            if near_robot(cs["med"]):
                continue
            lows.append(cs["z05"])
    if len(lows) > 0:
        TABLE_Z = quant(lows, 0.3)
    R["table_z_est_from_objects"] = r4(TABLE_Z)
except Exception as e:
    log_err("table z", e)


def plausible_object(it, max_xy=0.16, max_h=0.2):
    cs = it.get("cs")
    if cs is None or cs["n"] < 12:
        return False
    if near_robot(cs["med"]):
        return False
    if cs["xr"] > max_xy or cs["yr"] > max_xy:
        return False
    if TABLE_Z is not None and cs["z90"] - TABLE_Z > max_h:
        return False
    b = it.get("box")
    if b is not None:
        if (b[2] - b[0]) > 0.4 * IMG_W or (b[3] - b[1]) > 0.4 * IMG_H:
            return False
    return True


def dedupe(items, min_px=8.0):
    out = []
    for it in items:
        c = it.get("uv")
        dup = False
        for o in out:
            if math.hypot(o["uv"][0] - c[0], o["uv"][1] - c[1]) < min_px:
                dup = True
                break
        if not dup:
            out.append(it)
    return out


def item_uv(it):
    mi = it.get("mi")
    if mi is not None and mi.get("n", 0) > 0:
        return [mi["u"], mi["v"]]
    if it.get("box") is not None:
        return box_center(it["box"])
    return None


def box_iou(a, b):
    try:
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        if ua <= 0:
            return 0.0
        return inter / ua
    except Exception:
        return 0.0


def box_contain(a, b):
    """intersection over the smaller of the two boxes (1.0 when one box lies inside the other)"""
    try:
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        sm = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
        if sm <= 0:
            return 0.0
        return ix * iy / sm
    except Exception:
        return 0.0


def cube_score_at(box):
    best = 0.0
    for prompt in SEGS:
        if "cube" not in prompt:
            continue
        for c in SEGS[prompt]:
            if c.get("box") is None or c.get("score") is None:
                continue
            if box_contain(box, c["box"]) > 0.5 and c["score"] > best:
                best = c["score"]
    return best


RECLASS = []
try:
    for prompt in SEGS:
        for it in SEGS[prompt][:3]:
            if not plausible_object(it):
                continue
            uv = item_uv(it)
            if uv is None:
                continue
            it["uv"] = uv
            if it["kind"] == "button" and it.get("box") is not None:
                cscore = cube_score_at(it["box"])
                if cscore > (it["score"] or 0.0):
                    it["kind"] = "cube"
                    RECLASS.append([prompt, [round(b, 1) for b in it["box"]], r4(it["score"]), r4(cscore)])
            if it["kind"] == "button":
                OBJS["buttons"].append(it)
            elif it["kind"] == "cube":
                if it["cs"]["xr"] < 0.09 and it["cs"]["yr"] < 0.09:
                    OBJS["cubes"].append(it)
            else:
                OBJS["containers"].append(it)
    R["inventory"] = {"buttons": len(OBJS["buttons"]), "cubes": len(OBJS["cubes"]),
                      "containers": len(OBJS["containers"]), "button_to_cube": RECLASS}
    ev("inventory " + str(R["inventory"]))
except Exception as e:
    log_err("inventory", e)

# ---------------------------------------------------------------- ROI and table plane (front)
ROI = [16, 40, IMG_W - 16, IMG_H - 8]
try:
    boxes = []
    for key in ["cubes", "containers", "buttons"]:
        for it in OBJS[key]:
            if it.get("box") is not None:
                boxes.append(it["box"])
    if len(boxes) > 0:
        u0 = min(b[0] for b in boxes) - 28
        v0 = min(b[1] for b in boxes) - 28
        u1 = max(b[2] for b in boxes) + 28
        v1 = max(b[3] for b in boxes) + 28
        ROI = [int(max(0, u0)), int(max(0, v0)), int(min(IMG_W, u1)), int(min(IMG_H, v1))]
    R["roi"] = ROI
except Exception as e:
    log_err("roi", e)

PLANE = None
try:
    R["phase"] = "C_plane"
    rows = []
    gu0 = max(0, ROI[0] - 30)
    gv0 = max(0, ROI[1] - 30)
    gu1 = min(IMG_W, ROI[2] + 30)
    gv1 = min(IMG_H, ROI[3] + 30)
    for v in range(gv0, gv1, 4):
        for u in range(gu0, gu1, 4):
            d = DEP0[v][u]
            if d > 0.05 and d < 10:
                rows.append([float(u), float(v), 1.0 / d])

    def fit_plane(pts):
        s = [[0.0] * 3 for k in range(3)]
        t = [0.0, 0.0, 0.0]
        for p in pts:
            a = [p[0], p[1], 1.0]
            for i in range(3):
                t[i] += a[i] * p[2]
                for j in range(3):
                    s[i][j] += a[i] * a[j]

        def det3(m):
            return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
        D = det3(s)
        if abs(D) < 1e-12:
            return None
        sol = []
        for k in range(3):
            m = [list(r) for r in s]
            for i in range(3):
                m[i][k] = t[i]
            sol.append(det3(m) / D)
        return sol

    # robust init: per-row medians -> Theil-Sen slope in v; per-column residual medians -> slope in u
    byrow = {}
    for p in rows:
        byrow.setdefault(p[1], []).append(p)
    rmed = []
    for vv in sorted(byrow.keys()):
        if len(byrow[vv]) >= 5:
            rmed.append([vv, quant([p[2] for p in byrow[vv]], 0.5)])
    slopes = []
    for i in range(len(rmed)):
        for j in range(i + 1, len(rmed)):
            if rmed[j][0] - rmed[i][0] >= 8:
                slopes.append((rmed[j][1] - rmed[i][1]) / (rmed[j][0] - rmed[i][0]))
    pb = quant(slopes, 0.5) if len(slopes) > 0 else 0.0
    pc = quant([m[1] - pb * m[0] for m in rmed], 0.5) if len(rmed) > 0 else 1.0
    bycol = {}
    for p in rows:
        bycol.setdefault(p[0], []).append(p[2] - pb * p[1] - pc)
    cmed = []
    for uu in sorted(bycol.keys()):
        if len(bycol[uu]) >= 5:
            cmed.append([uu, quant(bycol[uu], 0.5)])
    slopes = []
    for i in range(len(cmed)):
        for j in range(i + 1, len(cmed)):
            if cmed[j][0] - cmed[i][0] >= 8:
                slopes.append((cmed[j][1] - cmed[i][1]) / (cmed[j][0] - cmed[i][0]))
    pa = quant(slopes, 0.5) if len(slopes) > 0 else 0.0
    pc = pc + (quant([m[1] - pa * m[0] for m in cmed], 0.5) if len(cmed) > 0 else 0.0)
    pl = [pa, pb, pc]
    for it_n in range(3):
        if pl is None:
            break
        keep = []
        for p in rows:
            w = pl[0] * p[0] + pl[1] * p[1] + pl[2]
            if w > 1e-6 and abs(1.0 / p[2] - 1.0 / w) < 0.006:
                keep.append(p)
        if len(keep) < 30:
            break
        pl2 = fit_plane(keep)
        if pl2 is None:
            break
        pl = pl2
    R["plane_inliers"] = len(keep) if pl is not None else 0
    PLANE = pl
    if PLANE is not None:
        res = [abs(p[2] - (PLANE[0] * p[0] + PLANE[1] * p[1] + PLANE[2])) for p in rows]
        R["plane"] = {"coef": [r4(c * 1000) for c in PLANE], "n": len(rows),
                      "res_med_x1e3": r4(quant(res, 0.5) * 1000), "res_q60_x1e3": r4(quant(res, 0.6) * 1000)}
        # table height from plane inliers via calibration
        if CALIB["conv"] is not None:
            zs = []
            for k in range(0, len(rows), 7):
                if res[k] < quant(res, 0.4):
                    wp = unproj(CM0, rows[k][0], rows[k][1], 1.0 / rows[k][2])
                    if wp is not None:
                        zs.append(wp[2])
            if len(zs) > 5:
                tz = quant(zs, 0.5)
                R["table_z_from_plane"] = r4(tz)
                if TABLE_Z is None or abs(tz - TABLE_Z) < 0.05:
                    TABLE_Z = tz
    R["table_z"] = r4(TABLE_Z)
    ev("plane fitted; table_z=" + str(r4(TABLE_Z)))
except Exception as e:
    log_err("plane", e)


def plane_depth(u, v):
    if PLANE is None:
        return None
    w = PLANE[0] * u + PLANE[1] * v + PLANE[2]
    if w <= 1e-6:
        return None
    return 1.0 / w


# ---------------------------------------------------------------- tracker
ST = 2
FG_T = 0.010
T = {"tracks": [], "gate": 20.0, "log": [], "grid": None, "minn": 3, "target": None,
     "covered": False, "cover_tick": None, "motion_seen": False, "quiet": 0, "moving_ticks": 0,
     "min_sep": None, "frames": 0, "cube_watch": [], "ready_reason": None}


def build_grid():
    g = []
    for v in range(ROI[1], ROI[3], ST):
        row = []
        for u in range(ROI[0], ROI[2], ST):
            row.append(plane_depth(u, v))
        g.append(row)
    T["grid"] = g


def depth_rows(cam):
    d = cam.depth_m[ROI[1]:ROI[3]:ST, ROI[0]:ROI[2]:ST].tolist()
    try:
        float(d[0][0])
    except Exception:
        d = [[x[0] for x in row] for row in d]
    return d


def fg_points(cam, excl):
    d = depth_rows(cam)
    g = T["grid"]
    pts = []
    for iy in range(min(len(d), len(g))):
        row = d[iy]
        grow = g[iy]
        v = ROI[1] + iy * ST
        for ix in range(min(len(row), len(grow))):
            pd = grow[ix]
            dd = row[ix]
            if pd is None or dd <= 0.01:
                continue
            if pd - dd > FG_T:
                u = ROI[0] + ix * ST
                skip = False
                for c in excl:
                    if (u - c[0]) * (u - c[0]) + (v - c[1]) * (v - c[1]) < c[2]:
                        skip = True
                        break
                if not skip:
                    pts.append([u, v, dd])
    return pts


def arm_exclusion(st, cm):
    out = []
    if CALIB["conv"] is None or cm is None or st is None:
        return out
    try:
        ee = vec(st.end_effector_poses["primary"].position)
    except Exception:
        return out
    anchors = [[0.0, 0.0, 0.33]]
    pts3 = []
    for k in range(6):
        f = k / 5.0
        pts3.append([anchors[0][0] + (ee[0] - anchors[0][0]) * f,
                     anchors[0][1] + (ee[1] - anchors[0][1]) * f,
                     anchors[0][2] + (ee[2] + 0.08 - anchors[0][2]) * f])
    pts3.append(ee)
    pts3.append([ee[0], ee[1], ee[2] + 0.12])
    for p in pts3:
        pp = proj(cm, p)
        if pp is None:
            continue
        rad = cm["fx"] * 0.075 / max(0.1, pp[2])
        out.append([pp[0], pp[1], rad * rad])
    return out


def track_update(pts, excl):
    tr = T["tracks"]
    if len(tr) == 0:
        return
    g2 = (T["gate"] * 1.4) ** 2
    preds = []
    frozen = []
    for t in tr:
        pu = t["u"] + t["vu"]
        pv = t["v"] + t["vv"]
        if t.get("dead"):
            preds.append([-1000.0, -1000.0, t["d"]])
        else:
            preds.append([pu, pv, t["d"]])
        fz = t.get("anchored", False) or t.get("dead", False)
        for c in excl:
            if (pu - c[0]) ** 2 + (pv - c[1]) ** 2 < c[2]:
                fz = True
                break
        frozen.append(fz)
    acc = [[0.0, 0.0, 0.0, 0] for t in tr]
    for p in pts:
        best = -1
        bd = g2
        for k in range(len(preds)):
            du = p[0] - preds[k][0]
            dv = p[1] - preds[k][1]
            dd = (p[2] - preds[k][2]) * 150.0
            dist = du * du + dv * dv + dd * dd
            if dist < bd:
                bd = dist
                best = k
        if best >= 0:
            a = acc[best]
            a[0] += p[0]
            a[1] += p[1]
            a[2] += p[2]
            a[3] += 1
    for k in range(len(tr)):
        t = tr[k]
        a = acc[k]
        t["n"] = a[3]
        if frozen[k]:
            t["vu"] = 0.0
            t["vv"] = 0.0
            t["frozen"] = True
            continue
        t["frozen"] = False
        if a[3] >= T["minn"]:
            nu = a[0] / a[3]
            nv = a[1] / a[3]
            t["d"] = a[2] / a[3]
            t["vu"] = 0.6 * t["vu"] + 0.4 * (nu - t["u"])
            t["vv"] = 0.6 * t["vv"] + 0.4 * (nv - t["v"])
            t["u"] = nu
            t["v"] = nv
        else:
            t["u"] = t["u"] + t["vu"]
            t["v"] = t["v"] + t["vv"]
            t["vu"] *= 0.7
            t["vv"] *= 0.7


def window_color_mass(cam, cu, cv, rad, cname, model):
    u0 = int(max(0, cu - rad))
    u1 = int(min(IMG_W, cu + rad + 1))
    v0 = int(max(0, cv - rad))
    v1 = int(min(IMG_H, cv + rad + 1))
    if u1 <= u0 or v1 <= v0:
        return 0
    img = cam.rgb[v0:v1:2, u0:u1:2].tolist()
    n = 0
    for row in img:
        for px in row:
            if color_match(px, cname, model):
                n += 1
    return n


def fg_components(pts, min_n=6):
    """Connected components over the subsampled fg grid; returns list of [u,v,d,n]."""
    cells = {}
    for p in pts:
        cells[(int(p[0]), int(p[1]))] = p[2]
    seen = set()
    comps = []
    for key in list(cells.keys()):
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        su = 0.0
        sv = 0.0
        sd = 0.0
        n = 0
        while len(stack) > 0:
            cur = stack.pop()
            su += cur[0]
            sv += cur[1]
            sd += cells[cur]
            n += 1
            for nb in [(cur[0] + ST, cur[1]), (cur[0] - ST, cur[1]), (cur[0], cur[1] + ST),
                       (cur[0], cur[1] - ST)]:
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if n >= min_n:
            comps.append([su / n, sv / n, sd / n, n])
    return comps


def new_track(u, v, d, src, anchored=False):
    return {"u": float(u), "v": float(v), "d": float(d), "vu": 0.0, "vv": 0.0, "n": 0,
            "src": src, "anchored": anchored, "id": len(T["tracks"]), "path_len": 0.0}


def log_tracks(extra=None):
    rec = [S["tick"]]
    for t in T["tracks"]:
        rec.append([round(t["u"], 1), round(t["v"], 1), t["n"], 1 if t.get("frozen") else 0])
    if extra is not None:
        rec.append(extra)
    if len(T["log"]) < 700:
        T["log"].append(rec)


def on_frame(obs):
    if not S["track_on"] or obs is None:
        return
    try:
        cam = obs.cameras[FRONT]
        st = obs.robot_state
        excl = arm_exclusion(st, CM0)
        pts = fg_points(cam, excl)
        prev = [[t["u"], t["v"]] for t in T["tracks"]]
        arm_moving = False
        try:
            ee = vec(st.end_effector_poses["primary"].position)
            pe = T.get("prev_ee")
            if pe is not None and math.sqrt(sum((ee[j] - pe[j]) ** 2 for j in range(3))) > 0.002:
                arm_moving = True
            T["prev_ee"] = ee
        except Exception:
            pass
        track_update(pts, excl)
        T["frames"] += 1
        # cube watch: colour mass at each seeded cube location (anchored until covered)
        tm = None
        for w in T["cube_watch"]:
            if w["covered"]:
                if w["target"] or (w["trk"].get("tcolor") is not None) or w.get("reseen"):
                    rewatch(w, cam, excl)
                continue
            arm_over = False
            for c in excl:
                rr = math.sqrt(c[2]) + w["rad"]
                if (w["u"] - c[0]) ** 2 + (w["v"] - c[1]) ** 2 < rr * rr:
                    arm_over = True
                    break
            if arm_over:
                w["low"] = 0
                continue
            m = window_color_mass(cam, w["u"], w["v"], w["rad"], w["cname"], w["model"])
            w["last"] = m
            if m <= max(1, int(0.3 * w["m0"])):
                w["low"] += 1
            else:
                w["low"] = 0
            if w["target"]:
                tm = m
            if w["low"] >= 3:
                w["covered"] = True
                w["cover_tick"] = S["tick"]
                recenter(w)
        # motion statistics
        moving = False
        for k in range(min(len(prev), len(T["tracks"]))):
            t = T["tracks"][k]
            dlt = math.hypot(t["u"] - prev[k][0], t["v"] - prev[k][1])
            t["path_len"] += dlt
            if (not t.get("frozen")) and (not t.get("anchored")) and (not t.get("dead")) and dlt > 0.6:
                moving = True
        if T["covered"] and not arm_moving:
            if moving:
                T["moving_ticks"] += 1
                if T["moving_ticks"] >= 6:
                    T["motion_seen"] = True
                T["quiet"] = 0
            else:
                T["quiet"] += 1
            # min separation between target and others after cover
            ti = target_index()
            if ti is not None:
                for k in range(len(T["tracks"])):
                    if k == ti or T["tracks"][k].get("dead"):
                        continue
                    sep = math.hypot(T["tracks"][k]["u"] - T["tracks"][ti]["u"],
                                     T["tracks"][k]["v"] - T["tracks"][ti]["v"])
                    if T["min_sep"] is None or sep < T["min_sep"]:
                        T["min_sep"] = sep
        if S["tick"] % 2 == 0 or moving:
            log_tracks([tm, len(pts), 1 if moving else 0])
    except Exception as e:
        T["err_count"] = T.get("err_count", 0) + 1
        if T["err_count"] < 6:
            log_err("on_frame", e)


def rewatch(w, cam, excl):
    """After covering, keep watching the cube's own location. If the cube shows again (its
    container was lifted away rather than slid with it) and is then covered again, the
    container that hides it is whichever track now sits on that location."""
    for c in excl:
        rr = math.sqrt(c[2]) + w["rad"]
        if (w["u"] - c[0]) ** 2 + (w["v"] - c[1]) ** 2 < rr * rr:
            return
    m = window_color_mass(cam, w["u"], w["v"], w["rad"], w["cname"], w["model"])
    w["last"] = m
    vis = m >= max(3, int(0.4 * w["m0"]))
    if not w.get("reseen"):
        if vis:
            w["vis_run"] = w.get("vis_run", 0) + 1
        else:
            w["vis_run"] = 0
        if w.get("vis_run", 0) >= 2:
            w["reseen"] = True
            w["low2"] = 0
            T["reseen_events"] = T.get("reseen_events", 0) + 1
            ev("cube " + w["cname"] + " visible again at its own location (containers lifted?)")
        return
    if m <= max(1, int(0.3 * w["m0"])):
        w["low2"] = w.get("low2", 0) + 1
    else:
        w["low2"] = 0
    if w["low2"] >= 3:
        w["reseen"] = False
        w["vis_run"] = 0
        best = None
        bd = None
        for t in T["tracks"]:
            if t.get("dead"):
                continue
            d = math.hypot(t["u"] - w["u"], t["v"] - w["v"])
            if bd is None or d < bd:
                bd = d
                best = t
        if best is not None and bd < 1.5 * T["gate"]:
            if w["cname"] in TARGET_COLORS:
                for t in T["tracks"]:
                    if t.get("tcolor") == w["cname"]:
                        t["tcolor"] = None
                best["tcolor"] = w["cname"]
            w["trk"] = best
            ev("cube " + w["cname"] + " re-covered; now under track at (" + str(round(best["u"], 1)) +
               "," + str(round(best["v"], 1)) + ") d=" + str(round(bd, 1)))
            T["quiet"] = 0


def recenter(w):
    """A seeded cube just got covered: snap its track to the covering blob, merge duplicates."""
    try:
        t = w["trk"]
        k = t["id"]
        cam = S["obs"].cameras[FRONT]
        pts = fg_points(cam, [])
        su = 0.0
        sv = 0.0
        sd = 0.0
        n = 0
        g2 = (T["gate"] * 1.5) ** 2
        for p in pts:
            if (p[0] - w["u"]) ** 2 + (p[1] - w["v"]) ** 2 < g2:
                su += p[0]
                sv += p[1]
                sd += p[2]
                n += 1
        if n >= T["minn"]:
            t["u"] = su / n
            t["v"] = sv / n
            t["d"] = sd / n
        t["anchored"] = False
        t["vu"] = 0.0
        t["vv"] = 0.0
        # merge other (non-cube) tracks sitting on the same blob
        for j in range(len(T["tracks"])):
            o = T["tracks"][j]
            if (o is not t) and (not o.get("dead")) and o["src"] != "cube":
                if math.hypot(o["u"] - t["u"], o["v"] - t["v"]) < 0.8 * T["gate"]:
                    o["dead"] = True
        if w["target"]:
            T["covered"] = True
            T["cover_tick"] = S["tick"]
            snapshot("target_covered_front", None, FRONT, False)
        ev("cube " + w["cname"] + " covered; track " + str(k) + " -> (" + str(round(t["u"], 1)) +
           "," + str(round(t["v"], 1)) + ") n=" + str(n))
    except Exception as e:
        log_err("recenter", e)


def prune_dead():
    keep = []
    for t in T["tracks"]:
        if not t.get("dead"):
            keep.append(t)
    T["tracks"] = keep
    T["target"] = target_index()


def target_index(cname=None):
    if cname is None:
        if len(TARGET_COLORS) > 0:
            cname = TARGET_COLORS[0]
    for k in range(len(T["tracks"])):
        t = T["tracks"][k]
        if t.get("dead"):
            continue
        if t.get("tcolor") is not None and (cname is None or t.get("tcolor") == cname):
            return k
    return None


# ---------------------------------------------------------------- phase E: target cube + seeds
TARGET_INFO = {"mode": None}
try:
    R["phase"] = "E_target"
    build_grid()
    fg0 = fg_points(OBS0.cameras[FRONT], arm_exclusion(OBS0.robot_state, CM0))
    comps0 = fg_components(fg0)
    R["fg0"] = {"n": len(fg0), "components": [[round(c[0], 1), round(c[1], 1), r4(c[2]), c[3]] for c in comps0[:20]]}
    # gate from container / cube sizes
    sizes = []
    for it in OBJS["containers"]:
        b = it.get("box")
        if b is not None:
            sizes.append(0.5 * math.hypot(b[2] - b[0], b[3] - b[1]))
    if len(sizes) > 0:
        T["gate"] = max(10.0, min(32.0, quant(sizes, 0.5)))
    # visible cubes by colour (language colours first)
    cube_seen = {}
    for cname in TARGET_COLORS + ["red", "green", "blue"]:
        if cname in cube_seen:
            continue
        best = None
        loose = None
        for it in SEGS.get(cname + " cube", []) + SEGS.get("cube", []):
            if not plausible_object(it, 0.09, 0.12):
                continue
            mi = it.get("mi")
            if mi is None or mi.get("n", 0) == 0 or mi.get("rgb") is None:
                continue
            inbtn = False
            for bt in OBJS["buttons"]:
                if bt.get("uv") is not None and math.hypot(bt["uv"][0] - mi["u"], bt["uv"][1] - mi["v"]) < 5:
                    inbtn = True
            if inbtn:
                continue
            if mi.get("cls") == cname:
                if best is None or (it["score"] or 0) > (best["score"] or 0):
                    best = it
            elif hue_close(mi["rgb"], cname):
                if loose is None or (it["score"] or 0) > (loose["score"] or 0):
                    loose = it
        if best is None and loose is not None:
            best = loose
            ev("cube " + cname + " accepted by hue tolerance; mask rgb=" + str(loose["mi"]["rgb"]))
        if best is not None:
            cube_seen[cname] = best
    # colour-blob fallback for target colours SAM missed
    for cname in TARGET_COLORS:
        if cname in cube_seen:
            continue
        cnt = 0
        su = 0.0
        sv = 0.0
        for v in range(ROI[1], ROI[3], 2):
            row = IMG0[v]
            for u in range(ROI[0], ROI[2], 2):
                px = row[u]
                if color_class(px[0], px[1], px[2]) == cname:
                    skip = False
                    for bt in OBJS["buttons"]:
                        b = bt.get("box")
                        if b is not None and b[0] - 3 <= u <= b[2] + 3 and b[1] - 3 <= v <= b[3] + 3:
                            skip = True
                    if not skip:
                        cnt += 1
                        su += u
                        sv += v
        if cnt >= 6:
            it = seg_points(FRONT, su / cnt, sv / cnt)
            if it is not None:
                it["pts"] = cloud_of(it, FRONT)
                it["cs"] = cloud_stats(it["pts"])
                it["mi"] = mask_info(it, IMG0)
                it["uv"] = item_uv(it) or [su / cnt, sv / cnt]
                it["kind"] = "cube"
                it["prompt"] = "colorblob:" + cname
                cube_seen[cname] = it
                ev("colour-blob cube " + cname + " at " + str([round(su / cnt, 1), round(sv / cnt, 1)]))
    R["cubes_seen"] = {}
    for cname in cube_seen:
        it = cube_seen[cname]
        R["cubes_seen"][cname] = {"uv": [r4(x) for x in it["uv"]], "cloud": it.get("cs"),
                                  "prompt": it.get("prompt"), "score": r4(it.get("score"))}
    # seed tracks: cubes first (anchored until covered), then containers / fg blobs
    for cname in cube_seen:
        it = cube_seen[cname]
        uv = it["uv"]
        dd = dep_at(DEP0, uv[0], uv[1])
        k = len(T["tracks"])
        trk = new_track(uv[0], uv[1], dd, "cube", True)
        trk["cube"] = cname
        if cname in TARGET_COLORS:
            trk["tcolor"] = cname
        T["tracks"].append(trk)
        b = it.get("box")
        rad = 8
        if b is not None:
            rad = int(max(4, min(16, 0.5 * max(b[2] - b[0], b[3] - b[1]))))
        model = None
        mi = it.get("mi")
        if mi is not None and mi.get("rgb") is not None:
            model = mi["rgb"]
        w = {"cname": cname, "u": uv[0], "v": uv[1], "rad": rad, "model": model,
             "trk": trk, "covered": False, "low": 0, "m0": 0, "last": None,
             "target": (len(TARGET_COLORS) > 0 and cname == TARGET_COLORS[0])}
        w["m0"] = window_color_mass(OBS0.cameras[FRONT], w["u"], w["v"], w["rad"], cname, model)
        if w["m0"] < 2:
            # model too strict; fall back to the hue class
            w["model"] = None
            w["m0"] = window_color_mass(OBS0.cameras[FRONT], w["u"], w["v"], w["rad"], cname, None)
        T["cube_watch"].append(w)
    seeds = []
    for it in OBJS["containers"]:
        seeds.append([it["uv"][0], it["uv"][1], "sam:" + it["prompt"]])
    for c in comps0:
        seeds.append([c[0], c[1], "fg"])
    for sd in seeds:
        dup = False
        for t in T["tracks"]:
            if math.hypot(t["u"] - sd[0], t["v"] - sd[1]) < 0.9 * T["gate"]:
                dup = True
                break
        inbtn = False
        for bt in OBJS["buttons"]:
            b = bt.get("box")
            if b is not None and b[0] - 4 <= sd[0] <= b[2] + 4 and b[1] - 4 <= sd[1] <= b[3] + 4:
                inbtn = True
        if dup or inbtn:
            continue
        if len(T["tracks"]) >= 8:
            break
        dd = dep_at(DEP0, sd[0], sd[1])
        T["tracks"].append(new_track(sd[0], sd[1], dd, sd[2], False))
    # refine non-anchored seeds with a few k-means passes on frame 0
    for it_n in range(3):
        track_update(fg0, [])
    for t in T["tracks"]:
        t["vu"] = 0.0
        t["vv"] = 0.0
    # drop non-cube tracks with no mass (empty seeds)
    for t in T["tracks"]:
        if t["src"] != "cube" and t["n"] < T["minn"]:
            t["dead"] = True
    prune_dead()
    for t in T["tracks"]:
        t["id"] = T["tracks"].index(t)
    if T["target"] is not None:
        TARGET_INFO["mode"] = "live_visible"
    elif VID_N > 0 and len(TARGET_COLORS) > 0:
        TARGET_INFO["mode"] = "video"
    else:
        TARGET_INFO["mode"] = "unknown"
    R["seed_tracks"] = [[round(t["u"], 1), round(t["v"], 1), t["src"], t["n"], t["anchored"]] for t in T["tracks"]]
    ev("tracks=" + str(len(T["tracks"])) + " gate=" + str(round(T["gate"], 1)) + " target=" +
       str(T["target"]) + " mode=" + str(TARGET_INFO["mode"]))
except Exception as e:
    log_err("target/seeds", e)


# ---------------------------------------------------------------- video backward pass (target hidden at t0)
try:
    if TARGET_INFO["mode"] == "video" and len(T["tracks"]) > 0:
        R["phase"] = "E_video_backward"
        cname = TARGET_COLORS[0]
        same_cam = True
        try:
            p0 = vec(VIDEO[0].cameras[FRONT].camera_pose.position)
            pl = vec(OBS0.cameras[FRONT].camera_pose.position)
            if math.sqrt(sum((p0[k] - pl[k]) ** 2 for k in range(3))) > 0.01:
                same_cam = False
        except Exception:
            pass
        TARGET_INFO["same_cam"] = same_cam
        hist = []
        BTN_BOXES = [bt["box"] for bt in OBJS["buttons"] if bt.get("box") is not None]
        saved = [[t["u"], t["v"], t["d"]] for t in T["tracks"]]
        found = None
        for idx in range(VID_N - 1, -1, -1):
            fr = VIDEO[idx]
            cam = fr.cameras[FRONT]
            pts = fg_points(cam, arm_exclusion(fr.robot_state, cam_model(cam)))
            track_update(pts, [])
            img = cam.rgb[ROI[1]:ROI[3]:2, ROI[0]:ROI[2]:2].tolist()
            cnt = 0
            su = 0.0
            sv = 0.0
            for iy, row in enumerate(img):
                for ix, px in enumerate(row):
                    if color_class(px[0], px[1], px[2]) == cname:
                        uu = ROI[0] + ix * 2
                        vv = ROI[1] + iy * 2
                        inb = False
                        for bb in BTN_BOXES:
                            if bb[0] - 3 <= uu <= bb[2] + 3 and bb[1] - 3 <= vv <= bb[3] + 3:
                                inb = True
                        if not inb:
                            cnt += 1
                            su += uu
                            sv += vv
            pos = [[round(t["u"], 1), round(t["v"], 1)] for t in T["tracks"]]
            hist.append([idx, cnt, pos])
            if cnt >= 5:
                cu = su / cnt
                cv = sv / cnt
                # the container covering it is the track nearest in the next (later) frames
                best = None
                bestd = None
                look = hist[-6:-1] if len(hist) > 1 else hist[-1:]
                for hrec in look:
                    for k in range(len(hrec[2])):
                        dist = math.hypot(hrec[2][k][0] - cu, hrec[2][k][1] - cv)
                        if bestd is None or dist < bestd:
                            bestd = dist
                            best = k
                found = [idx, cu, cv, best, bestd]
                break
        for k in range(len(T["tracks"])):
            T["tracks"][k]["u"] = saved[k][0]
            T["tracks"][k]["v"] = saved[k][1]
            T["tracks"][k]["d"] = saved[k][2]
            T["tracks"][k]["vu"] = 0.0
            T["tracks"][k]["vv"] = 0.0
        TARGET_INFO["video_hist_tail"] = hist[-12:]
        TARGET_INFO["video_found"] = found
        if found is not None and found[3] is not None:
            T["tracks"][found[3]]["tcolor"] = cname
            T["target"] = found[3]
            T["covered"] = True
            T["cover_tick"] = 0
            ev("video: target cube last visible at frame " + str(found[0]) + " -> track " + str(found[3]))
        else:
            ev("video: target cube never found in video")
except Exception as e:
    log_err("video backward", e)

if T["target"] is None and len(T["tracks"]) > 0 and TARGET_INFO["mode"] != "live_visible":
    # no evidence: pick the largest container blob, record that it is a guess
    try:
        bi = 0
        for k in range(len(T["tracks"])):
            if T["tracks"][k]["n"] > T["tracks"][bi]["n"]:
                bi = k
        if len(TARGET_COLORS) > 0:
            T["tracks"][bi]["tcolor"] = TARGET_COLORS[0]
        else:
            T["tracks"][bi]["tcolor"] = "any"
        T["target"] = bi
        T["covered"] = True
        T["cover_tick"] = 0
        TARGET_INFO["guess"] = True
        ev("NO target evidence; guessing track " + str(bi))
    except Exception as e:
        log_err("guess", e)
R["target_info"] = TARGET_INFO


# ---------------------------------------------------------------- motion primitives (per tick)
def tick(q, grip):
    if S["done"]:
        return False
    try:
        act = RobotAction(arms={"primary": ArmCommand(mode="joint_position", target=[float(v) for v in q],
                                                      gripper_position=float(grip), embodiment=EMB)})
        r = step(act)
    except Exception as e:
        log_err("step", e)
        S["step_fail"] += 1
        if S["step_fail"] >= 3:
            S["done"] = True
        return False
    S["tick"] += 1
    term = False
    try:
        if bool(r.terminated):
            term = True
    except Exception:
        pass
    try:
        if bool(r.truncated):
            term = True
    except Exception:
        pass
    try:
        if r.error is not None and S["step_errors"] < 5:
            S["step_errors"] += 1
            ev("step error: " + str(r.error)[:200])
    except Exception:
        pass
    obs = None
    try:
        obs = r.observation
    except Exception:
        obs = None
    if obs is None:
        try:
            obs = get_observation()
        except Exception:
            obs = None
    S["obs"] = obs
    if term:
        S["done"] = True
        S["term"] = S["tick"]
        ev("episode reported terminal at tick " + str(S["tick"]))
    on_frame(obs)
    return not S["done"]


def hold(n, grip, q=None):
    if q is None:
        q = cur_q()
    for k in range(n):
        if not tick(q, grip):
            return False
    return True


def ik(pos, quat, seed=None):
    if seed is None:
        seed = cur_q()
    try:
        res = solve_ik(Pose([float(v) for v in pos], [float(v) for v in quat], BASE),
                       seed_joints=[float(v) for v in seed])
    except Exception as e:
        log_err("solve_ik", e)
        return None
    if not ok_of(res):
        ev("ik failed at " + str([r4(v) for v in pos]) + " " + str(diag_of(res))[:200])
        return None
    try:
        q = res.joint_positions
        try:
            q = q["primary"]
        except Exception:
            pass
        return vec(q)[:7]
    except Exception as e:
        log_err("ik parse", e)
        return None


def move_joints(qg, grip, max_dq=0.05, settle=10, tol=0.012):
    q0 = cur_q()
    dmax = max(abs(qg[k] - q0[k]) for k in range(7))
    n = max(1, int(math.ceil(dmax / max_dq)))
    for i in range(1, n + 1):
        f = i / float(n)
        q = [q0[k] + (qg[k] - q0[k]) * f for k in range(7)]
        if not tick(q, grip):
            return False
    for k in range(settle):
        qc = cur_q()
        if max(abs(qg[j] - qc[j]) for j in range(7)) < tol:
            break
        if not tick(qg, grip):
            return False
    return True


def move_ee(pos, quat, grip, max_dq=0.05, settle=10):
    qg = ik(pos, quat)
    if qg is None and Q_HOME is not None:
        qg = ik(pos, quat, Q_HOME)
    if qg is None:
        try:
            ev("fallback move_to_pose " + str([r4(v) for v in pos]))
            res = move_to_pose(Pose([float(v) for v in pos], [float(v) for v in quat], BASE),
                               tolerance=0.015, max_steps=150)
            S["tick"] += 0
            try:
                S["tick"] += int(res.steps_executed)
            except Exception:
                pass
            try:
                S["obs"] = get_observation()
            except Exception:
                pass
            return ok_of(res)
        except Exception as e:
            log_err("move_to_pose", e)
            return False
    return move_joints(qg, grip, max_dq, settle)


def move_line(p_to, quat, grip, step_m=0.02, max_dq=0.03, settle=8):
    p_from = cur_ee()
    if p_from is None:
        return move_ee(p_to, quat, grip, max_dq, settle)
    dist = math.sqrt(sum((p_to[k] - p_from[k]) ** 2 for k in range(3)))
    n = max(1, int(math.ceil(dist / step_m)))
    seed = cur_q()
    qs = []
    for i in range(1, n + 1):
        f = i / float(n)
        p = [p_from[k] + (p_to[k] - p_from[k]) * f for k in range(3)]
        q = ik(p, quat, seed)
        if q is None:
            break
        qs.append(q)
        seed = q
    if len(qs) == 0:
        return move_ee(p_to, quat, grip, max_dq, settle)
    for i in range(len(qs)):
        st_n = settle if i == len(qs) - 1 else 0
        if not move_joints(qs[i], grip, max_dq, st_n):
            return False
    return True


def ee_report(tag):
    ee = cur_ee()
    rec = {"tag": tag, "tick": S["tick"], "ee": [r4(v) for v in ee] if ee else None,
           "grip": r4(cur_grip())}
    R.setdefault("ee_log", []).append(rec)
    return rec


# ---------------------------------------------------------------- object 3D refinement
def refine_with_camera(camera, guess, kind_prompts, max_dist=0.06):
    """Look for the object near `guess` (robot_base xyz) in `camera`; return cloud stats or None."""
    try:
        obs = get_observation()
        cam = obs.cameras[camera]
        cm = cam_model(cam)
    except Exception as e:
        log_err("refine obs", e)
        return None
    cands = []
    pp = proj(cm, guess)
    if pp is not None and 2 <= pp[0] < IMG_W - 2 and 2 <= pp[1] < IMG_H - 2:
        it = seg_points(camera, pp[0], pp[1])
        if it is not None:
            it["src"] = "point"
            cands.append(it)
    for prompt in kind_prompts:
        for it in seg_text(camera, prompt, 4):
            it["src"] = prompt
            cands.append(it)
    best = None
    bestd = None
    rec = []
    for it in cands:
        pts = cloud_of(it, camera)
        cs = cloud_stats(pts)
        if cs is None or cs["n"] < 10:
            continue
        d = math.hypot(cs["med"][0] - guess[0], cs["med"][1] - guess[1])
        rec.append([it["src"], r4(it.get("score")), r4(d), cs["med"], cs["xr"], cs["yr"]])
        if cs["xr"] > 0.2 or cs["yr"] > 0.2:
            continue
        if d < max_dist and (bestd is None or d < bestd):
            bestd = d
            best = [cs, pts, it["src"]]
    R.setdefault("refine_log", []).append({"camera": camera, "guess": [r4(v) for v in guess],
                                            "cands": rec[:10], "tick": S["tick"]})
    return best


def top_center(pts, frac=0.6):
    zs = [p[2] for p in pts]
    ztop = quant(zs, 0.92)
    zlo = quant(zs, 0.05)
    thr = zlo + (ztop - zlo) * frac
    top = [p for p in pts if p[2] >= thr]
    if len(top) < 5:
        top = pts
    return [quant([p[0] for p in top], 0.5), quant([p[1] for p in top], 0.5), ztop]


# ---------------------------------------------------------------- phase F: press the button
BUTTON = {"pressed": False}


def button_ok(it):
    cs = it["cs"]
    for w in T["cube_watch"]:
        if it.get("uv") is not None and math.hypot(it["uv"][0] - w["u"], it["uv"][1] - w["v"]) < 6:
            return False
    if TABLE_Z is not None and cs["z90"] - TABLE_Z > 0.08:
        return False
    if cs["xr"] > 0.12 or cs["yr"] > 0.12:
        return False
    return True


def locate_buttons(n_wanted):
    """Distinct plausible buttons (>= 5 cm apart), best score first; n_wanted 0 means all."""
    cands = []
    for it in OBJS["buttons"]:
        if not button_ok(it):
            continue
        sc = it["score"] or 0.0
        if BUTTON_COLOR is not None and it["prompt"] == BUTTON_COLOR + " button":
            sc += 0.05
        cands.append([sc, it])
    cands = sort_desc(cands)
    out = []
    for c in cands:
        med = c[1]["cs"]["med"]
        dup = False
        for o in out:
            om = o["cs"]["med"]
            if math.hypot(med[0] - om[0], med[1] - om[1]) < 0.05:
                dup = True
        if not dup:
            out.append(c[1])
    if len(out) == 0:
        for it in SEGS.get("button", []):
            cs = it.get("cs")
            if cs is None or cs["n"] < 10 or near_robot(cs["med"]):
                continue
            if cs["xr"] > 0.2 or cs["yr"] > 0.2:
                continue
            ev("button fallback: implausible-but-closest candidate from 'button'")
            out.append(it)
            break
    if n_wanted > 0:
        out = out[:n_wanted]
    return out


def sort_desc(cands):
    res = []
    for c in cands:
        k = 0
        while k < len(res) and res[k][0] >= c[0]:
            k += 1
        res.insert(k, c)
    return res


PRESS_DEADLINE = 179


def press_once(bt, rec):
    tc = top_center(bt["pts"], 0.5)
    rec["front_est"] = [r4(v) for v in tc]
    rec["prompt"] = bt["prompt"]
    ev("button front estimate " + str(rec["front_est"]) + " via '" + bt["prompt"] + "'")
    hover_z = tc[2] + 0.10
    move_ee([tc[0], tc[1], hover_z], QUAT_DOWN, 0.0, 0.06, 6)
    ee_report("button_hover")
    snapshot("button_hover_wrist", None, WRIST, False)
    if S["tick"] < PRESS_DEADLINE - 70:
        ref = refine_with_camera(WRIST, tc, ["button"], 0.05)
        if ref is not None:
            tc2 = top_center(ref[1], 0.5)
            rec["wrist_est"] = [r4(v) for v in tc2]
            if math.hypot(tc2[0] - tc[0], tc2[1] - tc[1]) < 0.04 and abs(tc2[2] - tc[2]) < 0.04:
                tc = [tc2[0], tc2[1], 0.5 * (tc[2] + tc2[2])]
                move_ee([tc[0], tc[1], hover_z], QUAT_DOWN, 0.0, 0.04, 4)
    rec["used"] = [r4(v) for v in tc]
    move_line([tc[0], tc[1], tc[2] + 0.025], QUAT_DOWN, 0.0, 0.03, 0.05, 0)
    # slow final approach to a shallow target just past the cap top
    q_press = ik([tc[0], tc[1], tc[2] - 0.008], QUAT_DOWN)
    move_line([tc[0], tc[1], tc[2] - 0.008], QUAT_DOWN, 0.0, 0.004, 0.015, 0)
    zmin = None
    for k in range(6):
        if not tick(q_press if q_press is not None else cur_q(), 0.0):
            break
        ee = cur_ee()
        if ee is not None and (zmin is None or ee[2] < zmin):
            zmin = ee[2]
    rec["press_tick"] = S["tick"]
    rec["pressed"] = True
    rec["ee_z_min"] = r4(zmin)
    if zmin is not None:
        rec["press_depth_reached"] = r4(tc[2] - zmin)
    snapshot("button_pressed_front", None, FRONT, False)
    # retract straight up once; never re-contact the cap
    move_line([tc[0], tc[1], hover_z], QUAT_DOWN, 0.0, 0.03, 0.05, 2)
    ee_report("button_lift")
    ev("button press done at tick " + str(rec["press_tick"]))


try:
    R["phase"] = "F_press"
    S["track_on"] = True
    log_tracks(["start"])
    bts = locate_buttons(PRESS_COUNT)
    BUTTON["n_found"] = len(bts)
    BUTTON["n_wanted"] = PRESS_COUNT
    if len(bts) == 0:
        ev("no plausible button candidate; skip press")
    if PRESS_COUNT > 0 and len(bts) < PRESS_COUNT:
        ev("only " + str(len(bts)) + " of " + str(PRESS_COUNT) + " required buttons located")
    # nearest to the current end effector first
    ee0 = cur_ee()
    if ee0 is not None and len(bts) > 1:
        bts = [c[1] for c in sort_desc([[-math.hypot(b["cs"]["med"][0] - ee0[0], b["cs"]["med"][1] - ee0[1]), b]
                                        for b in bts])]
    BUTTON["presses"] = []
    for bt in bts:
        if S["done"]:
            break
        if S["tick"] >= PRESS_DEADLINE - 12:
            ev("press deadline reached before next button; tick " + str(S["tick"]))
            break
        rec = {"pressed": False}
        BUTTON["presses"].append(rec)
        try:
            press_once(bt, rec)
        except Exception as e:
            log_err("press_once", e)
    BUTTON["pressed"] = len([p for p in BUTTON["presses"] if p.get("pressed")]) > 0
except Exception as e:
    log_err("press", e)
R["button"] = BUTTON

# ---------------------------------------------------------------- phase G: watch the containers
WATCH = {}
try:
    R["phase"] = "G_watch"
    if Q_HOME is not None and not S["done"]:
        move_joints(Q_HOME, 1.0, 0.06, 12)
    ee_report("watch_pose")
    WATCH["start_tick"] = S["tick"]
    HORIZON = min(MAX_STEPS, 480)
    reserve = 20 + 90 * max(1, len(TARGET_COLORS))
    limit = max(S["tick"] + 20, HORIZON - reserve)
    WATCH["horizon"] = HORIZON
    WATCH["limit"] = limit
    reason = "limit"
    for k in range(MAX_STEPS):
        if S["done"]:
            reason = "done"
            break
        if S["tick"] >= limit:
            reason = "tick_limit"
            break
        if T["covered"] and T["motion_seen"] and T["quiet"] >= 50:
            reason = "motion_then_quiet"
            break
        if not tick(Q_HOME if Q_HOME is not None else cur_q(), 1.0):
            break
    prune_dead()
    snapshot("watch_end_front", None, FRONT, False)
    WATCH["end_tick"] = S["tick"]
    WATCH["reason"] = reason
    WATCH["covered"] = T["covered"]
    WATCH["cover_tick"] = T["cover_tick"]
    WATCH["motion_seen"] = T["motion_seen"]
    WATCH["moving_ticks"] = T["moving_ticks"]
    WATCH["min_sep_px"] = r4(T["min_sep"])
    WATCH["tracks"] = [[round(t["u"], 1), round(t["v"], 1), r4(t["d"]), t["n"], t["src"],
                        round(t["path_len"], 1)] for t in T["tracks"]]
    WATCH["target"] = T["target"]
    WATCH["cube_watch"] = [[w["cname"], w["covered"], w.get("cover_tick"), w["m0"], w["last"], w["target"]]
                           for w in T["cube_watch"]]
    ev("watch end reason=" + reason + " target=" + str(T["target"]) + " tracks=" + str(WATCH["tracks"]))
except Exception as e:
    log_err("watch", e)
R["watch"] = WATCH
S["track_on"] = False
R["track_log"] = T["log"]


# ---------------------------------------------------------------- phase H: pick up the container
def grasp_geometry(pts, ztable, neighbor_dir=None):
    """Top centre, height, narrowest horizontal width and its axis angle."""
    tc = top_center(pts, 0.6)
    zt = tc[2]
    body = [p for p in pts if ztable is None or p[2] > ztable + 0.008]
    if len(body) < 8:
        body = pts
    best = None
    widths = []
    for k in range(36):
        th = k * math.pi / 36.0
        c = math.cos(th)
        s = math.sin(th)
        proj1 = [p[0] * c + p[1] * s for p in body]
        w = quant(proj1, 0.97) - quant(proj1, 0.03)
        widths.append([w, th])
        if best is None or w < best[0]:
            best = [w, th]
    if neighbor_dir is not None:
        # among axes within 8 mm of the narrowest, prefer the one most perpendicular to the
        # direction of the nearest other container, so the open fingers do not land on it
        pick = None
        for wt in widths:
            if wt[0] <= best[0] + 0.008:
                align = abs(math.cos(wt[1]) * neighbor_dir[0] + math.sin(wt[1]) * neighbor_dir[1])
                if pick is None or align < pick[0]:
                    pick = [align, wt]
        if pick is not None:
            best = pick[1]
    h = None
    if ztable is not None:
        h = zt - ztable
    return {"top": tc, "height": h, "width": best[0], "axis": best[1]}


PICK = {"attempts": []}


def pick_container(track_idx, put_back):
    att = {"track": track_idx, "tick0": S["tick"]}
    PICK["attempts"].append(att)
    t = T["tracks"][track_idx]
    u = t["u"]
    v = t["v"]
    obs = get_observation()
    cam = obs.cameras[FRONT]
    cm = cam_model(cam)
    guess = None
    it = seg_points(FRONT, u, v)
    if it is not None:
        pts = cloud_of(it, FRONT)
        cs = cloud_stats(pts)
        att["front_point_seg"] = cs
        if cs is not None and cs["n"] >= 10 and cs["xr"] < 0.2 and cs["yr"] < 0.2:
            guess = top_center(pts, 0.5)
    if guess is None:
        dd = t["d"]
        wp = unproj(cm, u, v, dd)
        if wp is not None:
            guess = [wp[0], wp[1], wp[2]]
    if guess is None:
        ev("pick: cannot localize track " + str(track_idx))
        att["result"] = "no_localization"
        return False
    att["front_guess"] = [r4(x) for x in guess]
    hover = [guess[0], guess[1], guess[2] + 0.16]
    move_ee(hover, QUAT_DOWN, 1.0, 0.06, 8)
    ee_report("container_hover")
    snapshot("container_hover_wrist", None, WRIST, False)
    ref = refine_with_camera(WRIST, guess, ["container", "cup", "bin"], 0.06)
    pts = None
    if ref is not None:
        pts = ref[1]
        att["wrist_ref"] = ref[0]
        att["wrist_src"] = ref[2]
    elif it is not None:
        pts = cloud_of(it, FRONT)
    if pts is None or len(pts) < 8:
        ev("pick: no cloud for grasp geometry")
        att["result"] = "no_cloud"
        return False
    ndir = None
    try:
        tcen = top_center(pts, 0.6)
        bestd = None
        for j in range(len(T["tracks"])):
            if j == track_idx or T["tracks"][j].get("dead"):
                continue
            o = T["tracks"][j]
            wp = unproj(CM0, o["u"], o["v"], o["d"])
            if wp is None:
                continue
            dx = wp[0] - tcen[0]
            dy = wp[1] - tcen[1]
            dn = math.hypot(dx, dy)
            if dn > 0.01 and (bestd is None or dn < bestd):
                bestd = dn
                ndir = [dx / dn, dy / dn]
        att["neighbor_dist"] = r4(bestd)
    except Exception as e:
        log_err("neighbor dir", e)
    gg = grasp_geometry(pts, TABLE_Z, ndir)
    att["geom"] = {"top": [r4(x) for x in gg["top"]], "height": r4(gg["height"]),
                   "width": r4(gg["width"]), "axis_deg": r4(gg["axis"] * 180 / math.pi)}
    top = gg["top"]
    # finger closing axis = hand y axis; rotate about world z so it matches the narrow axis
    Rd = qmat(QUAT_DOWN)
    phi0 = math.atan2(Rd[1][1], Rd[0][1])
    yaw = gg["axis"] - phi0
    while yaw > math.pi / 2:
        yaw -= math.pi
    while yaw < -math.pi / 2:
        yaw += math.pi
    quat = qmul(qyaw(yaw), QUAT_DOWN)
    att["yaw_deg"] = r4(yaw * 180 / math.pi)
    hgt = gg["height"]
    if hgt is None or hgt <= 0:
        hgt = 0.06
    depth_in = min(0.03, max(0.012, 0.45 * hgt))
    gz = top[2] - depth_in
    if TABLE_Z is not None:
        gz = max(gz, TABLE_Z + 0.012)
    att["grasp_z"] = r4(gz)
    if gg["width"] > 0.078:
        ev("pick: container width " + str(r4(gg["width"])) + " exceeds gripper; trying grasp anyway")
    move_ee([top[0], top[1], top[2] + 0.12], quat, 1.0, 0.05, 8)
    move_line([top[0], top[1], top[2] + 0.04], quat, 1.0, 0.02, 0.04, 2)
    move_line([top[0], top[1], gz], quat, 1.0, 0.01, 0.02, 6)
    ee_report("container_grasp_pose")
    hold(14, 0.0)
    snapshot("grasp_closed_front", None, FRONT, False)
    g_after = cur_grip()
    att["grip_after_close"] = r4(g_after)
    ev("grip after close " + str(r4(g_after)))
    move_line([top[0], top[1], gz + 0.18], quat, 0.0, 0.02, 0.03, 6)
    hold(8, 0.0)
    rec = ee_report("container_lifted")
    snapshot("container_lifted_front", None, FRONT, False)
    att["grip_after_lift"] = r4(cur_grip())
    att["lift_ee"] = rec["ee"]
    holding = g_after is not None and g_after > 0.02
    att["holding_signal"] = holding
    if put_back and not S["done"]:
        move_line([top[0], top[1], gz + 0.01], quat, 0.0, 0.02, 0.03, 4)
        hold(8, 1.0)
        move_line([top[0], top[1], top[2] + 0.15], quat, 1.0, 0.03, 0.05, 2)
    att["result"] = "lifted" if holding else "closed_empty"
    att["tick1"] = S["tick"]
    return holding


try:
    R["phase"] = "H_pick"
    order = []
    for cname in TARGET_COLORS:
        k = target_index(cname)
        if k is not None and k not in order:
            order.append(k)
    if len(order) == 0 and T["target"] is not None:
        order.append(T["target"])
    PICK["order"] = order
    PICK["target_mode"] = TARGET_INFO.get("mode")
    for n_i in range(len(order)):
        if S["done"]:
            break
        more = n_i < len(order) - 1
        ok_pick = pick_container(order[n_i], more)
        if not ok_pick and not S["done"] and S["tick"] < min(MAX_STEPS, 480) - 90:
            ev("pick retry after empty close")
            hold(1, 1.0)
            pick_container(order[n_i], more)
    if not S["done"]:
        hold(10, 0.0)
except Exception as e:
    log_err("pick", e)
R["pick"] = PICK

R["phase"] = "end"
R["ticks"] = S["tick"]
R["terminal_tick"] = S["term"]
R["final_ee"] = ee_report("final")
snapshot("final_front", None, FRONT, False)
ev("program end ticks=" + str(S["tick"]))
result = R
