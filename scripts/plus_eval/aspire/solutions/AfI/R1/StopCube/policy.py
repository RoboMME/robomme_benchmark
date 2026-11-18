# StopCube v00 -- scene snapshot AND full attempt (R=1: one start, episode 0).
# Strategy (hypothesis from the task name; the runtime goal language decides):
#   a cube moves live; the robot must press a button so that the cube stops ON the target
#   at the k-th time it reaches it (k parsed from the goal).
#   1. Snapshot: language, task/controller metadata, cameras, robot state, video summary,
#      SAM3 prompt probes on both cameras (no ticks spent).
#   2. Classify objects by measured geometry (flat target, small elevated objects), identify
#      the cube as the small elevated object that MOVES across live ticks.
#   3. Track the cube every tick (colour + depth elevation in a window, own back-projection
#      calibrated against the harness point cloud), count target entries per target.
#   4. Move (per-tick joint interpolation, tracking continues) to hover above the button,
#      refine with the wrist camera, calibrate the press latency with an identical 4 cm jump.
#   5. Fire the press at the tick that puts the cube closest to the target centre at the
#      predicted registration time, on the k-th approach only. Hold, retreat once, hold until
#      a terminal status.
# Only public APIs; no simulator state; no hardcoded coordinates.
import math

R = {}
result = R
R["version"] = "v00_initial"
R["errors"] = []
R["events"] = []
R["phase"] = "init"
S = {"ticks": 0, "done": False, "grip": 0.0, "emb": "robomme_plus", "max_steps": 1000,
     "hz": 20.0, "table_z": 0.0, "bp_ok": False, "step_fail": 0, "pressed": False,
     "trackers": [], "targets": [], "cube": None, "active_target": None, "k": 1,
     "cmd": None, "ee": None, "ee_hist": [], "sam_calls": 0, "log": [], "front": "front",
     "wrist": "wrist", "base": "robot_base", "bias": [0.0, 0.0, 0.0], "tz_bp": 0.0, "phase": "init", "cube_prompts": ["cube", "block"],
     "L": 4.0, "ready": False, "missed_before_ready": [], "deadline_fallback": False}


def err(where, exc):
    msg = where + ": " + str(exc)
    R["errors"].append(msg[:400])
    print("ERR " + msg[:300])


def ev(msg):
    R["events"].append("t" + str(S["ticks"]) + " " + msg)
    print("EV t" + str(S["ticks"]) + " " + msg)


def rnd(v, n=4):
    try:
        return round(float(v), n)
    except Exception:
        return None


def vec(v, n=4):
    out = []
    try:
        for a in v:
            out.append(rnd(a, n))
    except Exception:
        return None
    return out


def qtl(sorted_vals, f):
    if len(sorted_vals) == 0:
        return None
    return sorted_vals[int((len(sorted_vals) - 1) * f)]


def dist2(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def quat_to_mat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    nrm = math.sqrt(w * w + x * x + y * y + z * z)
    if nrm < 1e-9:
        nrm = 1.0
    w, x, y, z = w / nrm, x / nrm, y / nrm, z / nrm
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


# ---------------------------------------------------------------- phase 0: snapshot
try:
    ctx = get_task_context()
    R["task_language"] = str(ctx.language)
    try:
        R["task_name"] = str(ctx.task_name)
        R["task_id"] = str(ctx.task_id)
        R["suite"] = str(ctx.suite)
        R["family"] = str(ctx.family)
    except Exception as exc:
        err("ctx fields", exc)
except Exception as exc:
    err("get_task_context", exc)
    R["task_language"] = ""

try:
    tmeta = robomme_plus.get_task_metadata()
    md = {}
    for key in tmeta:
        md[str(key)] = str(tmeta[key])[:600]
    R["task_metadata"] = md
    try:
        S["max_steps"] = int(tmeta["max_steps"])
    except Exception as exc:
        err("max_steps", exc)
    for key in ("control_hz", "control_frequency", "control_freq", "hz"):
        try:
            S["hz"] = float(tmeta[key])
            break
        except Exception:
            pass
    if len(str(R.get("task_language", ""))) == 0:
        try:
            R["task_language"] = str(tmeta["language"])
        except Exception:
            pass
except Exception as exc:
    err("get_task_metadata", exc)
R["max_steps"] = S["max_steps"]
R["control_hz"] = S["hz"]
print("max_steps " + str(S["max_steps"]) + " hz " + str(S["hz"]))

try:
    cmeta = robomme_plus.get_controller_metadata()
    cm = {}
    for key in cmeta:
        cm[str(key)] = str(cmeta[key])[:600]
    R["controller_metadata"] = cm
except Exception as exc:
    err("get_controller_metadata", exc)

obs0 = None
try:
    obs0 = get_observation()
    cams = {}
    for name in obs0.cameras:
        c = obs0.cameras[name]
        info = {}
        try:
            info["rgb_shape"] = [int(a) for a in c.rgb.shape]
            info["depth_shape"] = [int(a) for a in c.depth_m.shape]
            info["intrinsics"] = [vec(row, 3) for row in c.intrinsics]
            info["pose_position"] = vec(c.camera_pose.position)
            info["pose_quat_wxyz"] = vec(c.camera_pose.quaternion_wxyz)
            info["pose_frame"] = str(c.camera_pose.frame)
        except Exception as exc:
            err("camera info " + str(name), exc)
        cams[str(name)] = info
    R["cameras"] = cams
    names = list(cams.keys())
    if "front" not in names and len(names) > 0:
        S["front"] = names[0]
    if "wrist" not in names:
        for nm in names:
            if nm != S["front"]:
                S["wrist"] = nm
except Exception as exc:
    err("get_observation", exc)

try:
    st = get_robot_state()
    S["base"] = str(st.base_frame)
    S["emb"] = str(st.embodiment)
    S["cmd"] = [float(a) for a in st.joint_positions["primary"]]
    ee = st.end_effector_poses["primary"]
    S["ee"] = [float(a) for a in ee.position]
    S["quat0"] = [float(a) for a in ee.quaternion_wxyz]
    R["robot_state"] = {"base_frame": S["base"], "embodiment": S["emb"],
                        "joints": vec(S["cmd"]), "ee_pos": vec(S["ee"]),
                        "ee_quat": vec(S["quat0"]),
                        "gripper": rnd(st.gripper_positions["primary"]),
                        "timestamp_s": rnd(st.timestamp_s)}
except Exception as exc:
    err("get_robot_state", exc)

# top-down orientation: keep the home orientation if its tool z axis points down
S["down_quat"] = [0.0, 1.0, 0.0, 0.0]
try:
    m = quat_to_mat(S["quat0"])
    R["home_tool_z_axis"] = [rnd(m[0][2]), rnd(m[1][2]), rnd(m[2][2])]
    if m[2][2] < -0.9:
        S["down_quat"] = list(S["quat0"])
        R["orientation_choice"] = "home quaternion (tool z points down)"
    else:
        R["orientation_choice"] = "fixed [0,1,0,0] (home not top-down)"
except Exception as exc:
    err("orientation", exc)

# conditioning video summary (cheap)
VIDEO = []
try:
    VIDEO = robomme_plus.get_conditioning_video()
    nv = len(VIDEO)
    vs = {"frames": nv}
    if nv > 0:
        vs["t_first"] = rnd(VIDEO[0].timestamp_s)
        vs["t_last"] = rnd(VIDEO[nv - 1].timestamp_s)
        try:
            j0 = [float(a) for a in VIDEO[0].robot_state.joint_positions["primary"]]
            j1 = [float(a) for a in VIDEO[nv - 1].robot_state.joint_positions["primary"]]
            vs["joints_first"] = vec(j0)
            vs["joints_last"] = vec(j1)
            if S["cmd"] is not None:
                vs["live_minus_last_maxabs"] = rnd(max(abs(a - b) for a, b in zip(S["cmd"], j1)))
        except Exception as exc:
            err("video joints", exc)
        # coarse front-image change between sampled frames (16x16 grid)
        try:
            idxs = []
            nsamp = min(nv, 24)
            for i in range(nsamp):
                idxs.append(int(round(i * (nv - 1) / max(1, nsamp - 1))))
            prev = None
            diffs = []
            for i in idxs:
                img = VIDEO[i].cameras[S["front"]].rgb
                hh = int(img.shape[0])
                ww = int(img.shape[1])
                grid = []
                for gy in range(8, hh, 16):
                    row = img[gy].tolist()
                    for gx in range(8, ww, 16):
                        px = row[gx]
                        grid.append(px[0] + px[1] + px[2])
                if prev is not None:
                    diffs.append([i, rnd(sum(abs(a - b) for a, b in zip(grid, prev)) / len(grid), 2)])
                prev = grid
            vs["front_grid_diff"] = diffs
        except Exception as exc:
            err("video diff", exc)
    R["video"] = vs
    print("video frames " + str(nv))
except Exception as exc:
    err("get_conditioning_video", exc)

# ---------------------------------------------------------------- goal parsing
ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
       "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
CARD = {"once": 1, "twice": 2, "thrice": 3}
NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10}
COLORS = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "white", "black",
          "brown", "gray", "grey", "cyan", "magenta"]
goal = {"k": None, "rule": None, "cube_color": None, "clauses": []}
try:
    lang = str(R.get("task_language", "")).lower()
    clean = lang
    for ch in ",.;:!?()\"'":
        clean = clean.replace(ch, " ")
    words = clean.split()
    for i in range(len(words)):
        w = words[i]
        nxt = words[i + 1] if i + 1 < len(words) else ""
        val = None
        rule = None
        if w in ORD:
            val = ORD[w]
            rule = "ordinal word '" + w + "'"
        elif len(w) > 2 and w[:-2].isdigit() and w[-2:] in ("st", "nd", "rd", "th"):
            val = int(w[:-2])
            rule = "ordinal digits '" + w + "'"
        elif w in CARD:
            val = CARD[w]
            rule = "count word '" + w + "'"
        elif w in NUMW and nxt.startswith("time"):
            val = NUMW[w]
            rule = "count '" + w + " " + nxt + "'"
        elif w.isdigit() and nxt.startswith("time"):
            val = int(w)
            rule = "count '" + w + " " + nxt + "'"
        if val is not None and goal["k"] is None:
            goal["k"] = val
            goal["rule"] = rule
        if w in COLORS and nxt in ("cube", "block", "box") and goal["cube_color"] is None:
            goal["cube_color"] = w
    parts = [lang]
    for sep in [" and then ", ", then ", " then ", " and ", ";", ". "]:
        np2 = []
        for p in parts:
            for q in p.split(sep):
                if len(q.strip()) > 0:
                    np2.append(q.strip())
        parts = np2
    for p in parts:
        goal["clauses"].append({"text": p, "executed": False, "note": ""})
except Exception as exc:
    err("goal parse", exc)
if goal["cube_color"] is not None:
    S["cube_prompts"] = [goal["cube_color"] + " cube", "cube", "block"]
R["goal"] = goal
print("goal " + str(goal))

# ---------------------------------------------------------------- phase 1: SAM probes
FRONT_PROMPTS = ["cube", "block", "button", "red button", "target", "disc", "circle",
                 "bullseye", "goal region"]
if goal["cube_color"] is not None:
    FRONT_PROMPTS = [goal["cube_color"] + " cube"] + FRONT_PROMPTS
WRIST_PROMPTS = ["cube", "button", "target"]
TARGET_PROMPTS = ["target", "disc", "circle", "bullseye", "goal region"]
BUTTON_PROMPTS = ["button", "red button"]


def cloud_stats(pc):
    xs = []
    ys = []
    zs = []
    pts = pc.points
    try:
        pts = pts.tolist()
    except Exception:
        pass
    for p in pts:
        xs.append(float(p[0]))
        ys.append(float(p[1]))
        zs.append(float(p[2]))
    n = len(zs)
    if n == 0:
        return None
    sx = sorted(xs)
    sy = sorted(ys)
    sz = sorted(zs)
    ztop = qtl(sz, 0.95)
    tx = 0.0
    ty = 0.0
    tn = 0
    for i in range(n):
        if zs[i] >= ztop - 0.008:
            tx += xs[i]
            ty += ys[i]
            tn += 1
    if tn == 0:
        tn = 1
        tx = qtl(sx, 0.5)
        ty = qtl(sy, 0.5)
    return {"n": n, "med": [qtl(sx, 0.5), qtl(sy, 0.5), qtl(sz, 0.5)],
            "q05": [qtl(sx, 0.05), qtl(sy, 0.05), qtl(sz, 0.05)],
            "q95": [qtl(sx, 0.95), qtl(sy, 0.95), ztop],
            "top_xy": [tx / tn, ty / tn]}


def stats_public(s):
    if s is None:
        return None
    return {"n": s["n"], "med": vec(s["med"]), "q05": vec(s["q05"]), "q95": vec(s["q95"]),
            "top_xy": vec(s["top_xy"])}


def seg_probe(camera, prompt, ncloud):
    S["sam_calls"] += 1
    out = {"camera": camera, "prompt": prompt, "ok": False, "n": 0, "cands": []}
    objs = []
    try:
        sg = segment_text(camera, prompt)
        out["ok"] = bool(sg.ok)
        if sg.ok:
            cands = sg.segmentations
            out["n"] = len(cands)
            idx = 0
            for c in cands[:5]:
                e = {"score": rnd(c.score, 3), "box": None, "stats": None}
                try:
                    if c.box_xyxy is not None:
                        e["box"] = [float(a) for a in c.box_xyxy]
                except Exception:
                    pass
                if idx < ncloud:
                    try:
                        pc = mask_to_point_cloud(c, camera, target_frame=S["base"])
                        e["stats_raw"] = cloud_stats(pc)
                    except Exception as exc:
                        err("cloud " + prompt, exc)
                objs.append({"prompt": prompt, "score": float(c.score), "box": e["box"],
                             "stats": e.get("stats_raw"), "cand": c, "camera": camera})
                e["stats"] = stats_public(e.get("stats_raw"))
                if "stats_raw" in e:
                    e.pop("stats_raw")
                out["cands"].append(e)
                idx += 1
    except Exception as exc:
        err("segment " + camera + " " + prompt, exc)
    return out, objs


R["phase"] = "probes"
POOL = []
R["probes"] = []
for pr in FRONT_PROMPTS:
    o, objs = seg_probe(S["front"], pr, 3)
    R["probes"].append(o)
    for ob in objs:
        if ob["stats"] is not None:
            POOL.append(ob)
for pr in WRIST_PROMPTS:
    o, objs = seg_probe(S["wrist"], pr, 2)
    R["probes"].append(o)
print("probes done, pool " + str(len(POOL)))

# ---------------------------------------------------------------- phase 2: classify
OBJS = []
try:
    q05s = sorted(ob["stats"]["q05"][2] for ob in POOL)
    if len(q05s) > 0:
        S["table_z"] = qtl(q05s, 0.5)
    # merge entries describing the same physical object
    for ob in POOL:
        s = ob["stats"]
        hit = None
        for o2 in OBJS:
            if dist2(o2["stats"]["med"], s["med"]) < 0.02 and abs(o2["stats"]["med"][2] - s["med"][2]) < 0.03:
                hit = o2
                break
        if hit is None:
            OBJS.append({"stats": s, "box": ob["box"], "cand": ob["cand"], "hits": {ob["prompt"]: ob["score"]},
                         "best": ob["score"]})
        else:
            if ob["prompt"] not in hit["hits"] or hit["hits"][ob["prompt"]] < ob["score"]:
                hit["hits"][ob["prompt"]] = ob["score"]
            if ob["score"] > hit["best"]:
                hit["best"] = ob["score"]
                hit["stats"] = s
                hit["box"] = ob["box"]
                hit["cand"] = ob["cand"]
    # flat table estimate refinement: flat target-like objects define the table height
    flat_z = []
    for o2 in OBJS:
        s = o2["stats"]
        if s["q95"][2] - s["q05"][2] < 0.01:
            flat_z.append(s["med"][2])
    if len(flat_z) > 0:
        flat_z = sorted(flat_z)
        tz = qtl(flat_z, 0.5)
        if abs(tz - S["table_z"]) < 0.03:
            S["table_z"] = tz
    tz = S["table_z"]
    for o2 in OBJS:
        s = o2["stats"]
        h = s["q95"][2] - tz
        zext = s["q95"][2] - s["q05"][2]
        fp = max(s["q95"][0] - s["q05"][0], s["q95"][1] - s["q05"][1])
        o2["h"] = h
        o2["fp"] = fp
        cls = "other"
        if h <= 0.012 and zext <= 0.012 and fp >= 0.015 and fp <= 0.3:
            cls = "flat"
        elif h > 0.012 and h <= 0.10 and fp >= 0.012 and fp <= 0.12:
            cls = "small"
        o2["cls"] = cls
        o2["tscore"] = max([o2["hits"].get(p, 0.0) for p in TARGET_PROMPTS])
        o2["bscore"] = max([o2["hits"].get(p, 0.0) for p in BUTTON_PROMPTS])
        o2["cscore"] = max([o2["hits"].get(p, 0.0) for p in S["cube_prompts"]])
    R["objects"] = []
    for o2 in OBJS:
        R["objects"].append({"cls": o2["cls"], "h": rnd(o2["h"]), "fp": rnd(o2["fp"]),
                             "hits": dict((k2, rnd(v2, 3)) for k2, v2 in o2["hits"].items()),
                             "med": vec(o2["stats"]["med"]), "top_xy": vec(o2["stats"]["top_xy"]),
                             "box": vec(o2["box"], 1)})
    R["table_z"] = rnd(S["table_z"])
except Exception as exc:
    err("classify", exc)

# ---------------------------------------------------------------- phase 3: back-projection model
CAM = None
try:
    fc = obs0.cameras[S["front"]]
    K = fc.intrinsics
    CAM = {"fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]), "cy": float(K[1][2]),
           "R": quat_to_mat([float(a) for a in fc.camera_pose.quaternion_wxyz]),
           "t": [float(a) for a in fc.camera_pose.position], "conv": "cv"}
    S["H"] = int(fc.rgb.shape[0])
    S["W"] = int(fc.rgb.shape[1])
except Exception as exc:
    err("camera model", exc)
    S["H"] = 256
    S["W"] = 256


def bp(cam, u, v, d):
    a = (u - cam["cx"]) / cam["fx"] * d
    b = (v - cam["cy"]) / cam["fy"] * d
    c = cam["conv"]
    if c == "cv":
        p0, p1, p2 = a, b, d
    elif c == "gl":
        p0, p1, p2 = a, -b, -d
    else:
        p0, p1, p2 = d, -a, -b
    m = cam["R"]
    t = cam["t"]
    return (m[0][0] * p0 + m[0][1] * p1 + m[0][2] * p2 + t[0],
            m[1][0] * p0 + m[1][1] * p1 + m[1][2] * p2 + t[1],
            m[2][0] * p0 + m[2][1] * p1 + m[2][2] * p2 + t[2])


def new_frame(obs):
    c = obs.cameras[S["front"]]
    return {"rgb": c.rgb, "dep": c.depth_m, "rc": {}, "dc": {}}


def rgb_row(fr, y):
    r = fr["rc"].get(y)
    if r is None:
        try:
            r = fr["rgb"][y].tolist()
        except Exception:
            r = [list(px) for px in fr["rgb"][y]]
        fr["rc"][y] = r
    return r


def dep_row(fr, y):
    r = fr["dc"].get(y)
    if r is None:
        try:
            r = fr["dep"][y].tolist()
        except Exception:
            r = [float(a) for a in fr["dep"][y]]
        fr["dc"][y] = r
    return r


def box_pixels(fr, box, shrink, cam):
    """Back-projected pixels inside a shrunken box: list of (x, y, z, u, v, rgb)."""
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    cxp = 0.5 * (x1 + x2)
    cyp = 0.5 * (y1 + y2)
    hw = 0.5 * (x2 - x1) * shrink
    hh = 0.5 * (y2 - y1) * shrink
    out = []
    u0 = max(0, int(math.ceil(cxp - hw)))
    u1 = min(S["W"] - 1, int(math.floor(cxp + hw)))
    v0 = max(0, int(math.ceil(cyp - hh)))
    v1 = min(S["H"] - 1, int(math.floor(cyp + hh)))
    for v in range(v0, v1 + 1):
        rr = rgb_row(fr, v)
        dr = dep_row(fr, v)
        for u in range(u0, u1 + 1):
            d = dr[u]
            if d > 0.05 and d < 5.0:
                P = bp(cam, u, v, d)
                out.append((P[0], P[1], P[2], u, v, rr[u]))
    return out


FR0 = None
try:
    FR0 = new_frame(obs0)
    calib = {}
    best = None
    offs_by_conv = {}
    for conv in ("cv", "gl", "ros"):
        CAM["conv"] = conv
        errs = []
        offs = []
        for o2 in OBJS[:8]:
            if o2["box"] is None or o2["cls"] == "other":
                continue
            pix = box_pixels(FR0, o2["box"], 0.4, CAM)
            if len(pix) < 3:
                continue
            mx = qtl(sorted(p[0] for p in pix), 0.5)
            my = qtl(sorted(p[1] for p in pix), 0.5)
            mz = qtl(sorted(p[2] for p in pix), 0.5)
            m2 = o2["stats"]["med"]
            errs.append(math.sqrt((mx - m2[0]) ** 2 + (my - m2[1]) ** 2 + (mz - m2[2]) ** 2))
            offs.append([mx - m2[0], my - m2[1], mz - m2[2]])
        offs_by_conv[conv] = offs
        if len(errs) > 0:
            e = sorted(errs)[len(errs) // 2]
            calib[conv] = [rnd(e), len(errs)]
            if best is None or e < best[1]:
                best = [conv, e]
    R["bp_calibration"] = calib
    CAM["conv"] = "cv"
    if best is not None:
        CAM["conv"] = best[0]
        S["bp_ok"] = best[1] < 0.025
        offs = offs_by_conv[best[0]]
        if len(offs) > 0 and S["bp_ok"]:
            S["bias"] = [qtl(sorted(o[0] for o in offs), 0.5), qtl(sorted(o[1] for o in offs), 0.5),
                         qtl(sorted(o[2] for o in offs), 0.5)]
    R["bp_bias_vs_harness_cloud"] = vec(S["bias"])
    # table height in the back-projection frame: near-table pixels of flat objects, else bias-shifted
    S["tz_bp"] = S["table_z"] + S["bias"][2]
    if S["bp_ok"]:
        zf = []
        for o2 in OBJS:
            if o2["cls"] == "flat" and o2["box"] is not None:
                for p in box_pixels(FR0, o2["box"], 0.6, CAM):
                    zf.append(p[2])
        if len(zf) >= 5:
            tzb = qtl(sorted(zf), 0.5)
            if abs(tzb - S["tz_bp"]) < 0.02:
                S["tz_bp"] = tzb
    R["table_z_bp"] = rnd(S["tz_bp"])
    R["bp_conv"] = CAM["conv"]
    R["bp_ok"] = S["bp_ok"]
    print("bp calib " + str(calib) + " ok " + str(S["bp_ok"]))
except Exception as exc:
    err("bp calibration", exc)

# ---------------------------------------------------------------- trackers and targets


def tracker_from_obj(name, o2):
    s = o2["stats"]
    box = o2["box"]
    tr = {"name": name, "obj": o2, "u": 0.5 * (box[0] + box[2]), "v": 0.5 * (box[1] + box[3]),
          "pu": 0.0, "pv": 0.0, "half": max(4.0, 0.5 * max(box[2] - box[0], box[3] - box[1])),
          "xy": [s["top_xy"][0] + S["bias"][0], s["top_xy"][1] + S["bias"][1]], "z": s["med"][2], "h": o2["h"],
          "origin": None, "vel": [0.0, 0.0], "lost": 0,
          "hist": [], "maxdisp": 0.0, "color": [128, 128, 128], "thr": 999.0, "n": 0}
    tr["win"] = tr["half"] + 6
    tr["u0"] = tr["u"]
    tr["v0"] = tr["v"]
    try:
        pix = box_pixels(FR0, box, 1.0, CAM) if S["bp_ok"] else []
        el = [p[5] for p in pix if p[2] > S["tz_bp"] + 0.008]
        bg = [p[5] for p in pix if p[2] <= S["tz_bp"] + 0.004]
        if len(el) >= 3:
            col = [qtl(sorted(c[0] for c in el), 0.5), qtl(sorted(c[1] for c in el), 0.5),
                   qtl(sorted(c[2] for c in el), 0.5)]
            tr["color"] = col
            if len(bg) >= 3:
                bcol = [qtl(sorted(c[0] for c in bg), 0.5), qtl(sorted(c[1] for c in bg), 0.5),
                        qtl(sorted(c[2] for c in bg), 0.5)]
                contrast = abs(col[0] - bcol[0]) + abs(col[1] - bcol[1]) + abs(col[2] - bcol[2])
                tr["bg"] = bcol
                tr["contrast"] = contrast
                if contrast >= 40:
                    tr["thr"] = max(30.0, min(120.0, 0.5 * contrast))
    except Exception as exc:
        err("tracker color " + name, exc)
    return tr


def tracker_step(tr, fr, tick):
    W = int(tr["win"])
    uc = int(round(tr["u"] + tr["pu"]))
    vc = int(round(tr["v"] + tr["pv"]))
    y0 = max(0, vc - W)
    y1 = min(S["H"] - 1, vc + W)
    x0 = max(0, uc - W)
    x1 = min(S["W"] - 1, uc + W)
    stride = 1 if W <= 18 else 2
    cr, cg, cb = tr["color"][0], tr["color"][1], tr["color"][2]
    thr = tr["thr"]
    zmin = S["tz_bp"] + 0.008
    zmax = S["tz_bp"] + max(0.10, 2.5 * tr["h"])
    pts = []
    for y in range(y0, y1 + 1, stride):
        rr = rgb_row(fr, y)
        dr = dep_row(fr, y)
        for x in range(x0, x1 + 1, stride):
            px = rr[x]
            if abs(px[0] - cr) + abs(px[1] - cg) + abs(px[2] - cb) > thr:
                continue
            d = dr[x]
            if not (d > 0.05 and d < 5.0):
                continue
            P = bp(CAM, x, y, d)
            if P[2] < zmin or P[2] > zmax:
                continue
            pts.append((P[2], P[0], P[1], x, y))
    need = 3 if stride == 1 else 2
    tr["n"] = len(pts)
    if len(pts) >= need:
        zs = sorted(p[0] for p in pts)
        ztop = qtl(zs, 0.9)
        sx = 0.0
        sy = 0.0
        nt = 0
        su = 0.0
        sv = 0.0
        for p in pts:
            su += p[3]
            sv += p[4]
            if p[0] >= ztop - 0.008:
                sx += p[1]
                sy += p[2]
                nt += 1
        nu = su / len(pts)
        nv = sv / len(pts)
        tr["pu"] = 0.5 * tr["pu"] + 0.5 * (nu - tr["u"])
        tr["pv"] = 0.5 * tr["pv"] + 0.5 * (nv - tr["v"])
        tr["u"] = nu
        tr["v"] = nv
        tr["xy"] = [sx / nt, sy / nt]
        tr["lost"] = 0
        tr["hist"].append([tick, tr["xy"][0], tr["xy"][1], 1])
        sp = math.sqrt(tr["pu"] ** 2 + tr["pv"] ** 2)
        tr["win"] = max(tr["half"] + 6, min(40.0, tr["half"] + 6 + 2.5 * sp))
    else:
        tr["lost"] += 1
        tr["xy"] = [tr["xy"][0] + tr["vel"][0], tr["xy"][1] + tr["vel"][1]]
        tr["u"] = tr["u"] + tr["pu"]
        tr["v"] = tr["v"] + tr["pv"]
        tr["win"] = min(48.0, tr["win"] + 6)
        tr["hist"].append([tick, tr["xy"][0], tr["xy"][1], 0])
    update_velocity(tr, tick)
    if tr["lost"] == 0:
        if tr["origin"] is None:
            tr["origin"] = [tr["xy"][0], tr["xy"][1]]
        dsp = dist2(tr["xy"], tr["origin"])
        if dsp > tr["maxdisp"]:
            tr["maxdisp"] = dsp


def update_velocity(tr, tick):
    hist = tr["hist"]
    if len(hist) > 60:
        tr["hist"] = hist[-60:]
        hist = tr["hist"]
    cur = None
    for e in reversed(hist):
        if e[3] == 1:
            cur = e
            break
    if cur is None:
        return
    old = None
    for e in reversed(hist):
        if e[3] == 1 and e[0] <= cur[0] - 3:
            old = e
            break
    if old is None:
        for e in reversed(hist):
            if e[3] == 1 and e[0] < cur[0]:
                old = e
                break
    if old is None or cur[0] - old[0] > 8:
        return
    dt = float(cur[0] - old[0])
    tr["vel"] = [(cur[1] - old[1]) / dt, (cur[2] - old[2]) / dt]


def sam_reanchor(tr, why, min_disagree):
    """Re-locate one tracker with SAM3 in the current front image (no ticks).
    Commits the SAM position only when it disagrees with the tracker by >= min_disagree."""
    best = None
    for pr in S["cube_prompts"]:
        S["sam_calls"] += 1
        try:
            sg = segment_text(S["front"], pr)
            if not sg.ok:
                continue
            for c in sg.segmentations[:5]:
                try:
                    pc = mask_to_point_cloud(c, S["front"], target_frame=S["base"])
                    s = cloud_stats(pc)
                except Exception:
                    continue
                if s is None:
                    continue
                h = s["q95"][2] - S["table_z"]
                fp = max(s["q95"][0] - s["q05"][0], s["q95"][1] - s["q05"][1])
                if h < 0.012 or h > 0.10 or fp > 0.12:
                    continue
                if S.get("button_xy") is not None and dist2(s["top_xy"], S["button_xy"]) < 0.03:
                    continue
                sxy = [s["top_xy"][0] + S["bias"][0], s["top_xy"][1] + S["bias"][1]]
                dd = dist2(sxy, tr["xy"])
                if best is None or dd < best[0]:
                    best = [dd, sxy, c]
        except Exception as exc:
            err("reanchor " + pr, exc)
        if best is not None and best[0] < 0.05:
            break
    if best is None:
        return False
    s = best[1]
    c = best[2]
    disagreement = best[0]
    lg = R.setdefault("reanchors", [])
    if len(lg) < 80:
        lg.append([S["ticks"], why, rnd(disagreement), tr["name"]])
    if disagreement > 0.12:
        R["reanchor_rejects"] = R.get("reanchor_rejects", 0) + 1
        return False
    if disagreement < min_disagree:
        return False
    tr["xy"] = [s[0], s[1]]
    try:
        b = [float(a) for a in c.box_xyxy]
        tr["u"] = 0.5 * (b[0] + b[2])
        tr["v"] = 0.5 * (b[1] + b[3])
    except Exception:
        pass
    tr["lost"] = 0
    if tr["origin"] is None:
        tr["origin"] = [tr["xy"][0], tr["xy"][1]]
    if len(tr["hist"]) > 0 and tr["hist"][-1][0] == S["ticks"]:
        tr["hist"][-1] = [S["ticks"], tr["xy"][0], tr["xy"][1], 1]
    else:
        tr["hist"].append([S["ticks"], tr["xy"][0], tr["xy"][1], 1])
    dsp = dist2(tr["xy"], tr["origin"])
    if dsp > tr["maxdisp"]:
        tr["maxdisp"] = dsp
    return True


def target_new(o2):
    s = o2["stats"]
    r = 0.5 * 0.5 * ((s["q95"][0] - s["q05"][0]) + (s["q95"][1] - s["q05"][1]))
    rin = max(0.015, min(0.05, r))
    txy = [s["med"][0] + S["bias"][0], s["med"][1] + S["bias"][1]]
    src = "harness cloud + bias"
    try:
        if S["bp_ok"] and o2.get("box") is not None and FR0 is not None:
            pix = [p for p in box_pixels(FR0, o2["box"], 0.6, CAM) if p[2] < S["tz_bp"] + 0.006]
            if len(pix) >= 5:
                bx = qtl(sorted(p[0] for p in pix), 0.5)
                by = qtl(sorted(p[1] for p in pix), 0.5)
                if dist2([bx, by], txy) < 0.03:
                    txy = [bx, by]
                    src = "back-projected box pixels"
    except Exception as exc:
        err("target bp", exc)
    return {"xy": txy, "src": src, "harness_xy": [s["med"][0], s["med"][1]], "r": r, "rin": rin, "inside": False, "count": 0,
            "entries": [], "exits": [], "mind": 9.0, "d": 9.0, "cur_min": 9.0, "pass_min": [],
            "start_inside": False, "score": o2["tscore"], "hits": o2["hits"]}


def target_update(tg, xy, tick):
    d = dist2(xy, tg["xy"])
    tg["d"] = d
    if d < tg["mind"]:
        tg["mind"] = d
    if tg["inside"]:
        if d < tg["cur_min"]:
            tg["cur_min"] = d
        if d > tg["rin"] + 0.01:
            tg["inside"] = False
            tg["exits"].append(tick)
            tg["pass_min"].append(rnd(tg["cur_min"]))
    else:
        if d < tg["rin"]:
            tg["inside"] = True
            tg["cur_min"] = d
            if tick == 0:
                tg["start_inside"] = True
            else:
                tg["count"] += 1
                tg["entries"].append(tick)


def video_visits(tr):
    """Track the identified cube BACKWARDS through the conditioning video, starting from its
    live tick-0 pixel. If the video shows the cube moving and its last frame is contiguous with
    the live start, the video's target entries belong to this episode's count."""
    nv = len(VIDEO)
    out = {"frames": nv}
    R["video_track"] = out
    if nv < 2 or not S["bp_ok"]:
        out["skipped"] = "no video" if nv < 2 else "no back-projection"
        return
    try:
        vp = VIDEO[nv - 1].cameras[S["front"]].camera_pose.position
        out["cam_shift_vs_live"] = rnd(math.sqrt(sum((float(a) - b) ** 2 for a, b in zip(vp, CAM["t"]))))
    except Exception as exc:
        err("video cam pose", exc)
    if tr["origin"] is None:
        out["skipped"] = "tracker has no origin"
        return
    tmp = {"name": "video", "u": tr["u0"], "v": tr["v0"], "pu": 0.0, "pv": 0.0, "half": tr["half"],
           "xy": [tr["origin"][0], tr["origin"][1]], "h": tr["h"], "origin": [tr["origin"][0], tr["origin"][1]],
           "vel": [0.0, 0.0], "lost": 0, "hist": [], "maxdisp": 0.0, "color": tr["color"],
           "thr": tr["thr"], "n": 0, "win": 30.0}
    seq = []
    for i in range(nv - 1, -1, -1):
        try:
            fr = new_frame(VIDEO[i])
        except Exception as exc:
            err("video frame", exc)
            break
        tmp["win"] = 30.0
        tmp["vel"] = [0.0, 0.0]
        tmp["pu"] = 0.0
        tmp["pv"] = 0.0
        tracker_step(tmp, fr, -i)
        seq.append([i, tmp["xy"][0], tmp["xy"][1], 1 if tmp["lost"] == 0 else 0,
                    rnd(VIDEO[i].timestamp_s, 3)])
    seq = list(reversed(seq))
    valid = [e for e in seq if e[3] == 1]
    out["valid_frames"] = len(valid)
    out["path"] = [[e[0], rnd(e[1]), rnd(e[2]), e[3], e[4]] for e in seq[::max(1, len(seq) // 64)]]
    if len(valid) < 2:
        out["decision"] = "too few valid detections; video visits ignored"
        return
    moved = 0.0
    for e in valid:
        dd = dist2([e[1], e[2]], [valid[0][1], valid[0][2]])
        if dd > moved:
            moved = dd
    last = valid[-1]
    contiguous = last[0] >= nv - 3 and dist2([last[1], last[2]], tr["origin"]) < 0.03
    out["moved"] = rnd(moved)
    out["contiguous"] = contiguous
    counts = []
    for tg in TGT:
        tmpg = {"xy": tg["xy"], "rin": tg["rin"], "inside": False, "count": 0, "entries": [], "exits": [],
                "mind": 9.0, "d": 9.0, "cur_min": 9.0, "pass_min": [], "start_inside": False}
        first = True
        for e in valid:
            target_update(tmpg, [e[1], e[2]], 0 if first else e[0] + 1)
            first = False
        counts.append([tmpg["count"], tmpg["inside"], tmpg["start_inside"]])
    out["target_counts"] = counts
    lost_frac = 1.0 - float(len(valid)) / nv
    out["lost_frac"] = rnd(lost_frac, 3)
    if moved > 0.03 and contiguous and lost_frac < 0.25:
        for j in range(len(TGT)):
            tg = TGT[j]
            add = counts[j][0]
            tg["count"] += add
            tg["video_added"] = add
            if counts[j][1] and not tg["inside"]:
                tg["inside"] = True
        out["decision"] = "video visits ADDED to live counts (cube moved and video is contiguous)"
    else:
        out["decision"] = "video visits NOT added (moved=" + str(rnd(moved)) + ", contiguous=" + str(contiguous) + ")"
    ev("video: " + out["decision"])


TRK = []
TGT = []
try:
    smalls = [o2 for o2 in OBJS if o2["cls"] == "small" and o2["box"] is not None]
    # order: cube-prompt score first
    ordered = []
    for o2 in smalls:
        ordered.append([-o2["cscore"], len(ordered), o2])
    ordered = sorted(ordered)
    for ent in ordered[:4]:
        TRK.append(tracker_from_obj("obj" + str(len(TRK)), ent[2]))
    flats = []
    for o2 in OBJS:
        if o2["cls"] == "flat" and o2["tscore"] > 0.0:
            flats.append([-o2["tscore"], len(flats), o2])
    flats = sorted(flats)
    for ent in flats[:3]:
        TGT.append(target_new(ent[2]))
    R["trackers_init"] = [{"name": t["name"], "xy": vec(t["xy"]), "uv": [rnd(t["u"], 1), rnd(t["v"], 1)],
                           "color": vec(t["color"], 1), "thr": rnd(t["thr"], 1),
                           "contrast": rnd(t.get("contrast", -1), 1), "h": rnd(t["h"]),
                           "cscore": rnd(t["obj"]["cscore"], 3), "bscore": rnd(t["obj"]["bscore"], 3)}
                          for t in TRK]
    R["targets_init"] = [{"xy": vec(t["xy"]), "src": t["src"], "harness_xy": vec(t["harness_xy"]), "r": rnd(t["r"]), "rin": rnd(t["rin"]),
                          "score": rnd(t["score"], 3)} for t in TGT]
    print("trackers " + str(len(TRK)) + " targets " + str(len(TGT)))
except Exception as exc:
    err("tracker init", exc)


# ---------------------------------------------------------------- perception per tick
def cube_tracker():
    if S["cube"] is not None:
        return S["cube"]
    return None


def choose_cube():
    if S["cube"] is not None:
        return
    best = None
    for tr in TRK:
        if S.get("button_tracker") is tr:
            continue
        if S["ee"] is not None and dist2(tr["xy"], S["ee"]) < 0.06:
            continue
        if tr["maxdisp"] > 0.012 and (best is None or tr["maxdisp"] > best["maxdisp"]):
            best = tr
    if best is not None:
        S["cube"] = best
        ev("cube identified by motion: " + best["name"] + " disp " + str(rnd(best["maxdisp"])))
        # replay this tracker's history through every target so early visits are counted
        for e in best["hist"]:
            if e[3] == 1:
                for tg in TGT:
                    target_update(tg, [e[1], e[2]], e[0])
        try:
            video_visits(best)
        except Exception as exc:
            err("video visits", exc)


def active_target():
    if len(TGT) == 0:
        return None
    if len(TGT) == 1:
        return TGT[0]
    tr = S["cube"]
    if tr is None or tr["maxdisp"] < 0.05:
        return TGT[0]
    best = TGT[0]
    for tg in TGT:
        if tg["mind"] < best["mind"] - 0.01:
            best = tg
    return best


def perceive(obs):
    tick = S["ticks"]
    try:
        st = obs.robot_state
        S["ee"] = [float(a) for a in st.end_effector_poses["primary"].position]
        S["ee_hist"].append([tick, S["ee"][2]])
        if len(S["ee_hist"]) > 40:
            S["ee_hist"] = S["ee_hist"][-40:]
        S["joints"] = [float(a) for a in st.joint_positions["primary"]]
    except Exception as exc:
        err("perceive state", exc)
    if not S["bp_ok"]:
        if S["cube"] is not None:
            # SAM-only tracking: every tick once waiting, but thin out after 900 SAM calls (wall clock)
            if (S["ready"] or tick % 2 == 0) and (S["sam_calls"] < 900 or tick % 3 == 0):
                sam_reanchor(S["cube"], "nobp", 0.0)
                update_velocity(S["cube"], tick)
        elif tick % 2 == 0:
            for tr in TRK:
                if S.get("button_tracker") is not tr:
                    sam_reanchor(tr, "nobp-search", 0.0)
                    update_velocity(tr, tick)
            choose_cube()
        cube_bookkeeping(tick)
        return
    try:
        fr = new_frame(obs)
    except Exception as exc:
        err("frame", exc)
        return
    active = TRK if S["cube"] is None else [S["cube"]]
    for tr in active:
        try:
            tracker_step(tr, fr, tick)
        except Exception as exc:
            err("track " + tr["name"], exc)
    choose_cube()
    tr = S["cube"]
    if tr is not None:
        if tr["lost"] >= 3 or (tick % 20 == 0 and tick > 0):
            why = "lost" if tr["lost"] >= 3 else "periodic"
            if sam_reanchor(tr, why, 0.0 if why == "lost" else 0.02):
                update_velocity(tr, tick)
    cube_bookkeeping(tick)


def cube_bookkeeping(tick):
    tr = S["cube"]
    if tr is None:
        if len(S["log"]) < 1500:
            S["log"].append([tick, None])
        return
    for tg in TGT:
        target_update(tg, tr["xy"], tick)
    tg = active_target()
    S["active_target"] = tg
    if len(S["log"]) < 1500:
        S["log"].append([tick, rnd(tr["xy"][0]), rnd(tr["xy"][1]), tr["n"], tr["lost"],
                         rnd(tg["d"]) if tg is not None else None,
                         tg["count"] if tg is not None else None,
                         rnd(S["ee"][2]) if S["ee"] is not None else None])


# ---------------------------------------------------------------- control
def advance(target):
    if S["done"]:
        return False
    if S["ticks"] >= S["max_steps"] - 1:
        S["done"] = True
        ev("max_steps guard reached")
        return False
    reply = None
    try:
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=[float(a) for a in target],
            gripper_position=S["grip"], embodiment=S["emb"])}))
    except Exception as exc:
        err("step", exc)
        S["step_fail"] += 1
        S["ticks"] += 1
        if S["step_fail"] >= 3:
            S["done"] = True
        return not S["done"]
    S["ticks"] += 1
    S["cmd"] = [float(a) for a in target]
    term = False
    for fld in ("terminated", "truncated", "done"):
        try:
            if fld == "terminated" and bool(reply.terminated):
                term = True
                R["terminal_flag"] = "terminated"
            if fld == "truncated" and bool(reply.truncated):
                term = True
                R["terminal_flag"] = R.get("terminal_flag", "truncated")
            if fld == "done" and bool(reply.done):
                term = True
                R["terminal_flag"] = R.get("terminal_flag", "done")
        except Exception:
            pass
    try:
        if reply.error is not None:
            R.setdefault("step_errors", []).append([S["ticks"], str(reply.error)[:200]])
    except Exception:
        pass
    if term:
        S["done"] = True
        R["terminal_tick"] = S["ticks"]
        ev("episode terminal")
        return False
    obs = None
    try:
        obs = reply.observation
    except Exception:
        obs = None
    if obs is None:
        try:
            obs = get_observation()
        except Exception as exc:
            err("get_observation after step", exc)
            return True
    perceive(obs)
    return not S["done"]


def solve(pos, seed):
    try:
        ik = solve_ik(Pose([float(a) for a in pos], [float(a) for a in S["down_quat"]], S["base"]),
                      seed_joints=[float(a) for a in seed])
        if ik.ok:
            return [float(a) for a in ik.joint_positions]
        R.setdefault("ik_fail", []).append([vec(pos), str(ik.error)[:200]])
    except Exception as exc:
        err("solve_ik", exc)
    return None


def move_interp(goal_j, max_delta, settle_max, check):
    start = list(S["cmd"])
    dmax = max(abs(a - b) for a, b in zip(goal_j, start))
    n = max(1, int(math.ceil(dmax / max_delta)))
    for i in range(1, n + 1):
        tgt = [a + (b - a) * i / n for a, b in zip(start, goal_j)]
        if not advance(tgt):
            return False
        if check is not None and check():
            return True
    for i in range(settle_max):
        try:
            jerr = max(abs(a - b) for a, b in zip(goal_j, S["joints"]))
        except Exception:
            jerr = 0.0
        if jerr < 0.01:
            break
        if not advance(goal_j):
            return False
        if check is not None and check():
            return True
    return True


def joint_err(goal_j):
    try:
        return max(abs(a - b) for a, b in zip(goal_j, S["joints"]))
    except Exception:
        return -1.0


# ---------------------------------------------------------------- trigger logic
def trigger_check():
    """True when this tick is the press tick for the k-th visit."""
    if S["pressed"]:
        return False
    tr = S["cube"]
    tg = S["active_target"]
    if tr is None or tg is None:
        return False
    k = S["k"]
    L = S["L"]
    remaining = S["max_steps"] - S["ticks"]
    gate = (tg["count"] == k - 1 and not tg["inside"]) or (tg["count"] == k and tg["inside"])
    if not gate and not S["deadline_fallback"]:
        ents = tg["entries"]
        interval = 60.0
        if len(ents) >= 2:
            interval = float(ents[-1] - ents[0]) / (len(ents) - 1)
        if remaining < 1.2 * interval + L + 5 and tg["count"] >= k and not tg["inside"]:
            S["deadline_fallback"] = True
            ev("deadline fallback: count " + str(tg["count"]) + " >= k " + str(k) + ", relaxing gate")
    if S["deadline_fallback"]:
        gate = True
    if not gate:
        return False
    px = tr["xy"][0]
    py = tr["xy"][1]
    vx = tr["vel"][0]
    vy = tr["vel"][1]
    pd0 = math.sqrt((px + vx * L - tg["xy"][0]) ** 2 + (py + vy * L - tg["xy"][1]) ** 2)
    pd1 = math.sqrt((px + vx * (L + 1) - tg["xy"][0]) ** 2 + (py + vy * (L + 1) - tg["xy"][1]) ** 2)
    rec = R.setdefault("trigger_trace", [])
    if len(rec) < 400:
        rec.append([S["ticks"], tg["count"], int(tg["inside"]), rnd(tg["d"]), rnd(pd0), rnd(pd1),
                    rnd(math.sqrt(vx * vx + vy * vy), 5)])
    if pd0 < 0.9 * tg["rin"] and pd0 <= pd1:
        R["trigger"] = {"tick": S["ticks"], "count": tg["count"], "inside": tg["inside"],
                        "d_now": rnd(tg["d"]), "pred_d_at_reg": rnd(pd0), "L": rnd(L, 2),
                        "vel": vec(tr["vel"], 5), "cube_xy": vec(tr["xy"]), "target_xy": vec(tg["xy"]),
                        "deadline_fallback": S["deadline_fallback"]}
        return True
    if tg["count"] == k and tg["inside"] and pd0 >= 0.9 * tg["rin"]:
        R["trigger"] = {"tick": S["ticks"], "count": tg["count"], "inside": True, "late": True,
                        "d_now": rnd(tg["d"]), "pred_d_at_reg": rnd(pd0), "L": rnd(L, 2)}
        ev("late trigger: cube inside on k-th visit but predicted to leave")
        return True
    return False


def note_missed_before_ready():
    tg = S["active_target"]
    if tg is not None and tg["count"] >= S["k"] and not S["ready"]:
        if len(S["missed_before_ready"]) == 0:
            S["missed_before_ready"].append(S["ticks"])
            ev("k-th visit happened before the hover was ready")
    return False


# ---------------------------------------------------------------- run
try:
    if goal["k"] is not None:
        S["k"] = int(goal["k"])
    R["k_used"] = S["k"]
    R["k_rule"] = goal["rule"] if goal["k"] is not None else "default 1 (no count in goal)"
    R["phase"] = "observe"
    if obs0 is not None:
        perceive(obs0)
    # A. hold a few ticks, detect which small object moves
    if S["cmd"] is not None:
        for i in range(10):
            if not advance(S["cmd"]):
                break
            if S["cube"] is not None and i >= 3:
                break
        R["hold_ticks"] = S["ticks"]
        R["tracker_motion"] = [[t["name"], rnd(t["maxdisp"]), t["n"]] for t in TRK]
except Exception as exc:
    err("phase observe", exc)

BTN = None
try:
    R["phase"] = "button"
    cands = []
    tgt_xy = [tg["xy"] for tg in TGT]
    for o2 in OBJS:
        if o2["cls"] == "other" or o2["h"] < 0.006 or o2["fp"] > 0.12:
            continue
        moving = False
        for tr in TRK:
            if tr["obj"] is o2 and (S["cube"] is tr or tr["maxdisp"] > 0.012):
                moving = True
        if moving:
            continue
        if S["cube"] is not None and dist2(o2["stats"]["top_xy"], [S["cube"]["xy"][0] - S["bias"][0], S["cube"]["xy"][1] - S["bias"][1]]) < 0.03:
            continue
        near_tgt = False
        for txy in tgt_xy:
            if dist2(o2["stats"]["med"], txy) < 0.02:
                near_tgt = True
        if near_tgt:
            continue
        # rank: button-prompt score minus cube-prompt score (both logged); geometry already bounded
        cands.append([-(o2["bscore"] - o2["cscore"]), len(cands), o2])
    cands = sorted(cands)
    R["button_candidates"] = [{"med": vec(c[2]["stats"]["med"]), "h": rnd(c[2]["h"]), "fp": rnd(c[2]["fp"]),
                               "bscore": rnd(c[2]["bscore"], 3), "cscore": rnd(c[2]["cscore"], 3)}
                              for c in cands]
    chosen = None
    for c in cands:
        if c[2]["bscore"] > 0.0:
            chosen = c[2]
            R["button_rule"] = "highest (button - cube) score among static elevated non-target objects"
            break
    if chosen is None and len(cands) > 0:
        chosen = cands[0][2]
        R["button_rule"] = "FALLBACK: no button-prompt hit; least cube-like static elevated object"
    if chosen is not None:
        o2 = chosen
        BTN = {"xy": [o2["stats"]["top_xy"][0], o2["stats"]["top_xy"][1]], "top": o2["stats"]["q95"][2],
               "src": "front", "bscore": o2["bscore"], "h": o2["h"]}
        S["button_xy"] = BTN["xy"]
        for tr in TRK:
            if tr["obj"] is o2:
                S["button_tracker"] = tr
    R["button_front"] = None if BTN is None else {"xy": vec(BTN["xy"]), "top": rnd(BTN["top"]),
                                                  "bscore": rnd(BTN["bscore"], 3), "h": rnd(BTN["h"])}
    if BTN is None:
        ev("no button candidate found")
except Exception as exc:
    err("button select", exc)

J10 = None
J06 = None
J02 = None
JP = None
JP2 = None


def ik_chain():
    top = BTN["top"]
    x = BTN["xy"][0]
    y = BTN["xy"][1]
    j10 = solve([x, y, top + 0.10], S["cmd"])
    if j10 is None:
        return None
    j06 = solve([x, y, top + 0.06], j10)
    j02 = solve([x, y, top + 0.02], j06 if j06 is not None else j10)
    jp = solve([x, y, top - 0.02], j02 if j02 is not None else j10)
    jp2 = solve([x, y, top - 0.035], jp if jp is not None else j10)
    return [j10, j06, j02, jp, jp2]


try:
    if BTN is not None and S["cmd"] is not None and not S["done"]:
        R["phase"] = "approach"
        ch = ik_chain()
        if ch is None:
            ev("IK failed for hover above button")
        else:
            J10 = ch[0]
            ok = move_interp(J10, 0.08, 8, note_missed_before_ready)
            R["hover10_ee"] = vec(S["ee"])
            R["hover10_tick"] = S["ticks"]
            # wrist refinement of the button
            try:
                o, objs = seg_probe(S["wrist"], "button", 3)
                R["wrist_button_probe"] = o
                best = None
                for ob in objs:
                    s = ob["stats"]
                    if s is None:
                        continue
                    dd = dist2(s["top_xy"], BTN["xy"])
                    h = s["q95"][2] - S["table_z"]
                    if dd < 0.04 and abs(s["q95"][2] - BTN["top"]) < 0.03 and h > 0.008:
                        if best is None or ob["score"] > best[0]:
                            best = [ob["score"], s]
                if best is not None:
                    s = best[1]
                    R["button_wrist"] = {"xy": vec(s["top_xy"]), "top": rnd(s["q95"][2]),
                                         "shift": rnd(dist2(s["top_xy"], BTN["xy"])),
                                         "dtop": rnd(s["q95"][2] - BTN["top"])}
                    BTN["xy"] = [s["top_xy"][0], s["top_xy"][1]]
                    BTN["top"] = s["q95"][2]
                    BTN["src"] = "wrist"
                    S["button_xy"] = BTN["xy"]
                    ch2 = ik_chain()
                    if ch2 is not None:
                        ch = ch2
                else:
                    ev("wrist button refinement rejected; keeping front estimate")
            except Exception as exc:
                err("wrist refine", exc)
            J10, J06, J02, JP, JP2 = ch[0], ch[1], ch[2], ch[3], ch[4]
            R["button_final"] = {"xy": vec(BTN["xy"]), "top": rnd(BTN["top"]), "src": BTN["src"]}
            R["ik_ok"] = [J10 is not None, J06 is not None, J02 is not None, JP is not None, JP2 is not None]
            if J06 is not None and J02 is not None and JP is not None and not S["done"]:
                move_interp(J10, 0.08, 4, note_missed_before_ready)
                move_interp(J06, 0.08, 6, note_missed_before_ready)
                # press-latency calibration: identical 4 cm single-command jump J06 -> J02
                z0 = S["ee"][2]
                prof = []
                for i in range(10):
                    if not advance(J02):
                        break
                    prof.append(rnd(z0 - S["ee"][2]))
                    note_missed_before_ready()
                    if i >= 3 and prof[-1] is not None and prof[-1] >= 0.036 and joint_err(J02) < 0.01:
                        break
                R["descent_profile_4cm"] = prof
                Lc = None
                prev = 0.0
                for i in range(len(prof)):
                    dcur = prof[i] if prof[i] is not None else 0.0
                    if dcur >= 0.027:
                        frac = 0.0
                        if dcur - prev > 1e-6:
                            frac = (0.027 - prev) / (dcur - prev)
                        Lc = i + frac
                        break
                    prev = dcur
                if Lc is not None:
                    S["L"] = max(1.0, min(8.0, Lc + 0.25))
                R["L_ticks"] = rnd(S["L"], 2)
                R["L_rule"] = "ticks to descend 0.027 m in the 4 cm calibration jump + 0.25" if Lc is not None else "default 4"
                for i in range(6):
                    if joint_err(J02) < 0.005:
                        break
                    if not advance(J02):
                        break
                R["hover02_ee"] = vec(S["ee"])
                R["hover02_tick"] = S["ticks"]
                S["ready"] = True
                ev("ready at hover top+0.02, L=" + str(rnd(S["L"], 2)))
            else:
                ev("IK chain incomplete; cannot press")
except Exception as exc:
    err("approach", exc)

# ---------------------------------------------------------------- wait, press
try:
    if S["ready"] and not S["done"]:
        R["phase"] = "wait"
        if len(TGT) == 0:
            # no flat target in the first probes: probe again with wider prompts (no ticks)
            for pr in TARGET_PROMPTS + ["red circle", "white circle", "marker", "goal", "zone"]:
                o, objs = seg_probe(S["front"], pr, 3)
                R.setdefault("target_reprobe", []).append(o)
                for ob in objs:
                    s2 = ob["stats"]
                    if s2 is None:
                        continue
                    h2 = s2["q95"][2] - S["table_z"]
                    fp2 = max(s2["q95"][0] - s2["q05"][0], s2["q95"][1] - s2["q05"][1])
                    if h2 <= 0.012 and fp2 >= 0.015 and fp2 <= 0.3 and len(TGT) < 3:
                        TGT.append(target_new({"stats": s2, "box": ob["box"], "tscore": ob["score"], "hits": {pr: ob["score"]}}))
                        if S["cube"] is not None:
                            for e in S["cube"]["hist"]:
                                if e[3] == 1:
                                    target_update(TGT[-1], [e[1], e[2]], e[0])
                if len(TGT) > 0:
                    break
            ev("target re-probe found " + str(len(TGT)))
        static_waited = 0
        while not S["done"]:
            if S["cube"] is None:
                static_waited += 1
                if static_waited % 15 == 0 and S["bp_ok"]:
                    for tr in TRK:
                        if S.get("button_tracker") is not tr:
                            sam_reanchor(tr, "search", 0.0)
                    choose_cube()
                if static_waited == 100 and not S.get("static_press_done"):
                    # nothing has moved: the cube may start only on a press. Press once, return.
                    ev("no object moved in 100 waiting ticks: one exploratory press")
                    R["static_press"] = {"tick": S["ticks"]}
                    S["static_press_done"] = True
                    for i in range(8):
                        if not advance(JP):
                            break
                    for i in range(8):
                        if not advance(J02):
                            break
                    R["static_press"]["end_tick"] = S["ticks"]
            if trigger_check():
                break
            if not advance(J02):
                break
        if R.get("trigger") is not None and not S["done"]:
            R["phase"] = "press"
            S["pressed"] = True
            ev("PRESS command")
            press_log = []
            moving_after = None
            for i in range(30):
                if not advance(JP):
                    break
                tr = S["cube"]
                sp = None
                if tr is not None:
                    sp = math.sqrt(tr["vel"][0] ** 2 + tr["vel"][1] ** 2)
                press_log.append([S["ticks"], rnd(S["ee"][2]), rnd(sp, 5),
                                  rnd(S["active_target"]["d"]) if S["active_target"] is not None else None])
                if i == 9 and sp is not None:
                    moving_after = sp
                    if sp > 0.002 and JP2 is not None:
                        ev("cube still moving 10 ticks after press: deeper press")
                        R["deeper_press"] = True
                        for j in range(10):
                            if not advance(JP2):
                                break
                        break
            R["press_log"] = press_log
            R["speed_10_ticks_after_press"] = rnd(moving_after, 5)
            for cl in goal["clauses"]:
                if "press" in cl["text"] or "button" in cl["text"] or "stop" in cl["text"]:
                    cl["executed"] = True
                    cl["note"] = "pressed at tick " + str(R["trigger"]["tick"])
            if not S["done"]:
                R["phase"] = "retreat"
                ev("retreat once")
                for i in range(40):
                    if not advance(J10):
                        break
except Exception as exc:
    err("wait/press", exc)

# ---------------------------------------------------------------- final self-check
try:
    R["phase"] = "final"
    R["ticks_used"] = S["ticks"]
    R["sam_calls"] = S["sam_calls"]
    R["pressed"] = S["pressed"]
    R["ready"] = S["ready"]
    R["missed_before_ready"] = S["missed_before_ready"]
    R["cube_tracker"] = None if S["cube"] is None else S["cube"]["name"]
    R["targets_final"] = [{"xy": vec(t["xy"]), "rin": rnd(t["rin"]), "count": t["count"],
                           "entries": t["entries"], "exits": t["exits"], "pass_min": t["pass_min"],
                           "mind": rnd(t["mind"]), "start_inside": t["start_inside"],
                           "d_final": rnd(t["d"])} for t in TGT]
    R["track_log"] = S["log"]
    for cl in goal["clauses"]:
        if not cl["executed"]:
            cl["note"] = cl["note"] + " NOT executed (pressed=" + str(S["pressed"]) + ")"
    R["clauses"] = goal["clauses"]
    if not S["pressed"]:
        ev("self-check: press clause not executed; no valid press moment observed")
    print("final: pressed " + str(S["pressed"]) + " ticks " + str(S["ticks"]) + " errors " + str(len(R["errors"])))
except Exception as exc:
    err("final", exc)
result = R
