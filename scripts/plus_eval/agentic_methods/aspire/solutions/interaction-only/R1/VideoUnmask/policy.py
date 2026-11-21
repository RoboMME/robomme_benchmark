# VideoUnmask v00 -- snapshot + full attempt in ONE start (R=1, interaction-only).
#
# Hypothesis (from the task name only; verify in the run):
#   the conditioning video shows coloured cubes that end up hidden under containers;
#   the language names the cube colour(s); the robot must lift the container that
#   hides that cube.  Memory cue = where the named cube was last seen in the video.
#
# Phases (each wrapped so the program never raises):
#   0 snapshot   language, task/controller metadata, cameras, robot state
#   1 video      length/timestamps, per-frame colour-label counts (front, stride 2)
#   2 probes     SAM3 prompt probes on front + wrist
#   3 calib      which camera convention reproduces mask_to_point_cloud (front)
#   4 target     named cube's last-seen 3-D position -> container top from live depth
#   5 motion     hover, wrist re-localise, top-down grasp, lift, hold
# Every task-dependent number below is derived at runtime from public observations;
# the constants are generic offsets / tolerances / thresholds only.

import math

R = {"version": "v00_initial", "phase_log": [], "errors": [], "motions": []}
result = R
ST = {"terminal": False, "steps": 0, "max_steps": 1000, "first_step_done": False}


def log(msg):
    text = str(msg)
    print("[VU] " + text)
    if len(R["phase_log"]) < 200:
        R["phase_log"].append(text[:400])


def err(phase, exc):
    text = phase + ": " + str(exc)[:400]
    print("[VU][ERR] " + text)
    if len(R["errors"]) < 80:
        R["errors"].append(text)


def short(value):
    try:
        return str(value)[:400]
    except Exception:
        return "<unprintable>"


def to_list(arr):
    try:
        return arr.tolist()
    except Exception:
        pass
    try:
        return float(arr)
    except Exception:
        pass
    return [to_list(item) for item in arr]


def flist(values, nd=4):
    return [round(float(item), nd) for item in values]


def quantile(values, frac):
    ordered = sorted(values)
    if len(ordered) == 0:
        return None
    idx = int((len(ordered) - 1) * frac)
    return ordered[idx]


def median(values):
    return quantile(values, 0.5)


# ---------------------------------------------------------------- quaternions
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def quat_norm(q):
    n = math.sqrt(sum(item * item for item in q))
    if n <= 0:
        return [1.0, 0.0, 0.0, 0.0]
    return [item / n for item in q]


def quat_to_rot(q):
    w, x, y, z = quat_norm(q)
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def wrap_half_pi(angle):
    while angle > math.pi / 2:
        angle -= math.pi
    while angle < -math.pi / 2:
        angle += math.pi
    return angle


# ---------------------------------------------------------------- terminal / motion bookkeeping
def flag_terminal(res):
    term = False
    try:
        if res.terminated:
            term = True
    except Exception:
        pass
    try:
        if res.truncated:
            term = True
    except Exception:
        pass
    try:
        status = str(res.status).lower()
        if "terminat" in status or "truncat" in status or "episode_end" in status:
            term = True
    except Exception:
        pass
    return term


def note_motion(name, res):
    rec = {"name": name}
    try:
        rec["ok"] = bool(res.ok)
    except Exception:
        rec["ok"] = None
    try:
        rec["status"] = str(res.status)
    except Exception:
        pass
    try:
        rec["steps"] = int(res.steps_executed)
        ST["steps"] += rec["steps"]
    except Exception:
        pass
    try:
        if res.error is not None:
            rec["error"] = short(res.error)
    except Exception:
        pass
    rec["terminal"] = flag_terminal(res)
    if rec["terminal"]:
        ST["terminal"] = True
    try:
        st = get_robot_state()
        rec["t"] = round(float(st.timestamp_s), 3)
        rec["ee"] = flist(st.end_effector_poses["primary"].position)
        rec["grip"] = round(float(st.gripper_positions["primary"]), 4)
    except Exception:
        pass
    rec["cum_steps"] = ST["steps"]
    R["motions"].append(rec)
    ST["first_step_done"] = True
    log("motion " + name + " -> " + short(rec))
    return rec


def budget(want):
    remaining = ST["max_steps"] - ST["steps"]
    return max(0, min(want, remaining - 5))


def can_move(want_min=15):
    if ST["terminal"]:
        return False
    return budget(1000) >= want_min


# ---------------------------------------------------------------- colour labels
LABELS = ["none", "red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink",
          "white", "gray", "black", "brown"]
COLOR_WORDS = {"red": [1], "orange": [2], "yellow": [3], "green": [4], "cyan": [5],
               "teal": [5], "blue": [6], "purple": [7], "violet": [7], "pink": [8],
               "magenta": [8], "white": [9], "gray": [10], "grey": [10], "black": [11],
               "brown": [12]}
NEIGHBOR_LABELS = {1: [8, 2], 2: [12, 1, 3], 3: [2], 4: [5], 5: [4, 6], 6: [5, 7],
                   7: [6, 8], 8: [1, 7], 9: [10], 10: [9], 11: [10], 12: [2]}
CONTAINER_WORDS = ["container", "containers", "bin", "bins", "cup", "cups", "box", "boxes",
                   "bowl", "bowls", "lid", "lids", "cover", "covers", "mug", "mugs", "jar",
                   "can", "cans", "bucket", "buckets"]


def label_of(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 40:
        return 11
    delta = mx - mn
    sat = delta / mx
    if sat < 0.18:
        if mx > 200:
            return 9
        if mx < 70:
            return 11
        return 10
    if sat < 0.40 or mx < 60:
        return 0
    if mx == r:
        hue = 60.0 * (((g - b) / delta) % 6.0)
    elif mx == g:
        hue = 60.0 * ((b - r) / delta + 2.0)
    else:
        hue = 60.0 * ((r - g) / delta + 4.0)
    if hue < 12 or hue >= 345:
        return 1
    if hue < 40:
        if mx < 140:
            return 12
        return 2
    if hue < 70:
        if mx < 120:
            return 0
        return 3
    if hue < 165:
        return 4
    if hue < 195:
        return 5
    if hue < 255:
        return 6
    if hue < 290:
        return 7
    return 8


def label_grid(rgb_list, stride):
    h = len(rgb_list)
    w = len(rgb_list[0])
    out = []
    for v in range(0, h, stride):
        row = rgb_list[v]
        for u in range(0, w, stride):
            px = row[u]
            out.append(label_of(px[0], px[1], px[2]))
    return out


# ---------------------------------------------------------------- camera projection
VARIANTS = [("opencv", False), ("opencv", True), ("opengl", False), ("opengl", True),
            ("robotic", False), ("robotic", True)]
CALIB = {"variant": ("opencv", False), "source": "default"}


def cam_params(cam):
    K = to_list(cam.intrinsics)
    pose = cam.camera_pose
    pos = [float(item) for item in pose.position]
    quat = [float(item) for item in pose.quaternion_wxyz]
    return {"fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]),
            "cy": float(K[1][2]), "t": pos, "q": quat, "R": quat_to_rot(quat),
            "frame": str(pose.frame)}


def backproject(u, v, d, cp, variant):
    xn = (u - cp["cx"]) / cp["fx"]
    yn = (v - cp["cy"]) / cp["fy"]
    conv = variant[0]
    if conv == "opencv":
        ray = (xn, yn, 1.0)
    elif conv == "opengl":
        ray = (xn, -yn, -1.0)
    else:
        ray = (1.0, -xn, -yn)
    scale = d
    if variant[1]:
        scale = d / math.sqrt(1.0 + xn * xn + yn * yn)
    px = ray[0] * scale
    py = ray[1] * scale
    pz = ray[2] * scale
    Rm = cp["R"]
    t = cp["t"]
    return (Rm[0][0] * px + Rm[0][1] * py + Rm[0][2] * pz + t[0],
            Rm[1][0] * px + Rm[1][1] * py + Rm[1][2] * pz + t[1],
            Rm[2][0] * px + Rm[2][1] * py + Rm[2][2] * pz + t[2])


def valid_depth(d):
    return d > 0.02 and d < 8.0


def cloud_points(found_cloud):
    pts = to_list(found_cloud.points)
    out = []
    for p in pts:
        out.append((float(p[0]), float(p[1]), float(p[2])))
    return out


def mask_pixels(cand, stride):
    src = cand["mask"]
    if src is None:
        src = cand["obj"]
    m = to_list(src)
    out = []
    for v in range(0, len(m), stride):
        row = m[v]
        for u in range(0, len(row), stride):
            if row[u]:
                out.append((u, v))
    return out


def cloud_of(cand, camera, base_frame):
    try:
        pc = mask_to_point_cloud(cand["obj"], camera, target_frame=base_frame)
    except Exception as exc:
        if cand["mask"] is None:
            err("mask_to_point_cloud", exc)
            return []
        try:
            pc = mask_to_point_cloud(cand["mask"], camera, target_frame=base_frame)
        except Exception as exc2:
            err("mask_to_point_cloud", exc2)
            return []
    try:
        return cloud_points(pc)
    except Exception as exc:
        err("cloud_points", exc)
        return []


def cloud_stats(pts):
    if len(pts) == 0:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    return {"n": len(pts), "cx": median(xs), "cy": median(ys), "cz": median(zs),
            "z05": quantile(zs, 0.05), "z95": quantile(zs, 0.95),
            "xspan": quantile(xs, 0.95) - quantile(xs, 0.05),
            "yspan": quantile(ys, 0.95) - quantile(ys, 0.05),
            "mean": (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))}


def round_stats(s):
    if s is None:
        return None
    out = {}
    for key in s:
        if key == "mean":
            out[key] = flist(s[key])
        elif key == "n":
            out[key] = s[key]
        else:
            out[key] = round(float(s[key]), 4)
    return out


def seg_candidates(seg, limit):
    """Uniform candidate dicts for either SegmentationSet shape:
    .segmentations[i].{score, box_xyxy, mask}  or parallel .masks/.scores/.boxes."""
    out = []
    try:
        if not seg.ok:
            return out
    except Exception:
        pass
    try:
        for cand in list(seg.segmentations)[:limit]:
            d = {"obj": cand, "score": None, "box": None, "mask": None}
            try:
                d["score"] = float(cand.score)
            except Exception:
                pass
            try:
                d["box"] = [float(item) for item in cand.box_xyxy]
            except Exception:
                pass
            try:
                d["mask"] = cand.mask
            except Exception:
                pass
            out.append(d)
        return out
    except Exception:
        pass
    try:
        masks = list(seg.masks)[:limit]
        scores = []
        boxes = []
        try:
            scores = list(seg.scores)
        except Exception:
            pass
        try:
            boxes = list(seg.boxes)
        except Exception:
            pass
        for i, m in enumerate(masks):
            d = {"obj": m, "score": None, "box": None, "mask": m}
            if i < len(scores):
                d["score"] = float(scores[i])
            if i < len(boxes):
                try:
                    d["box"] = [float(item) for item in boxes[i]]
                except Exception:
                    pass
            out.append(d)
    except Exception as exc:
        err("seg_candidates", exc)
    return out


def seg_summary(seg):
    out = {"ok": False, "n": 0, "scores": [], "boxes": []}
    try:
        out["ok"] = bool(seg.ok)
    except Exception as exc:
        out["error"] = short(exc)
    cands = seg_candidates(seg, 50)
    out["n"] = len(cands)
    for d in cands[:5]:
        out["scores"].append(None if d["score"] is None else round(d["score"], 4))
        out["boxes"].append(None if d["box"] is None else flist(d["box"], 1))
    if len(cands) == 0:
        try:
            out["seg_error"] = short(seg.error)
        except Exception:
            pass
    return out


def box_of(cand):
    return cand["box"]


def rscore(cand):
    if cand["score"] is None:
        return None
    return round(cand["score"], 4)


# ================================================================ PHASE 0: snapshot
base_frame = "robot_base"
cam_front = "front"
cam_wrist = "wrist"
language = ""
try:
    ctx = get_task_context()
    language = str(ctx.language)
    R["language"] = language
    try:
        R["task_name"] = str(ctx.task_name)
        R["task_id"] = short(ctx.task_id)
        R["suite"] = short(ctx.suite)
        R["family"] = short(ctx.family)
    except Exception as exc:
        err("context-fields", exc)
    try:
        meta = {}
        for key in ctx.metadata:
            meta[str(key)] = short(ctx.metadata[key])
        R["context_metadata"] = meta
    except Exception as exc:
        err("context-metadata", exc)
    log("language: " + language)
except Exception as exc:
    err("context", exc)

try:
    md = robomme_plus.get_task_metadata()
    tm = {}
    for key in md:
        tm[str(key)] = short(md[key])
    R["task_metadata"] = tm
    try:
        ST["max_steps"] = int(md["max_steps"])
    except Exception as exc:
        err("max_steps", exc)
    if language == "":
        try:
            language = str(md["language"])
            R["language"] = language
        except Exception:
            pass
except Exception as exc:
    err("task_metadata", exc)
R["max_steps"] = ST["max_steps"]
log("max_steps " + str(ST["max_steps"]))

try:
    cmeta = robomme_plus.get_controller_metadata()
    cm = {}
    for key in cmeta:
        cm[str(key)] = short(cmeta[key])
    R["controller_metadata"] = cm
except Exception as exc:
    err("controller_metadata", exc)

home_state = None
try:
    home_state = get_robot_state()
    base_frame = str(home_state.base_frame)
    eep = home_state.end_effector_poses["primary"]
    R["robot_state"] = {
        "base_frame": base_frame,
        "embodiment": short(home_state.embodiment),
        "joints": flist(home_state.joint_positions["primary"]),
        "ee_pos": flist(eep.position),
        "ee_quat": flist(eep.quaternion_wxyz),
        "ee_frame": short(eep.frame),
        "gripper": round(float(home_state.gripper_positions["primary"]), 4),
        "timestamp_s": round(float(home_state.timestamp_s), 4),
    }
except Exception as exc:
    err("robot_state", exc)

live_obs = None
try:
    live_obs = get_observation()
    names = [str(name) for name in live_obs.cameras]
    R["cameras"] = names
    if "front" not in names and len(names) > 0:
        cam_front = names[0]
    if "wrist" not in names and len(names) > 1:
        cam_wrist = names[1]
    cinfo = {}
    for name in names:
        cam = live_obs.cameras[name]
        info = {}
        try:
            info["rgb_shape"] = [int(item) for item in cam.rgb.shape]
            info["depth_shape"] = [int(item) for item in cam.depth_m.shape]
            cp = cam_params(cam)
            info["K"] = [round(cp["fx"], 3), round(cp["fy"], 3), round(cp["cx"], 3),
                         round(cp["cy"], 3)]
            info["pose_pos"] = flist(cp["t"])
            info["pose_quat"] = flist(cp["q"])
            info["pose_frame"] = cp["frame"]
        except Exception as exc:
            info["error"] = short(exc)
        cinfo[name] = info
    R["camera_info"] = cinfo
    R["obs_timestamp_s"] = round(float(live_obs.timestamp_s), 4)
except Exception as exc:
    err("observation", exc)

# ---------------------------------------------------------------- language parse
cube_targets = []
container_colors = []
try:
    clean = language.lower()
    for ch in ",.;:!?()'\"":
        clean = clean.replace(ch, " ")
    tokens = clean.split()
    for i, tok in enumerate(tokens):
        if tok in COLOR_WORDS:
            nxt = tokens[i + 1:i + 3]
            is_container = False
            for word in nxt:
                if word in CONTAINER_WORDS:
                    is_container = True
            if is_container:
                container_colors.append(tok)
            elif tok not in cube_targets:
                cube_targets.append(tok)
    R["parsed"] = {"cube_colors": cube_targets, "container_colors": container_colors,
                   "tokens": tokens[:60]}
    log("parsed cube colours " + str(cube_targets) + " container colours " +
        str(container_colors))
except Exception as exc:
    err("parse", exc)

# ================================================================ PHASE 1: conditioning video
video = []
VID = {"frames": [], "grids": [], "counts": [], "times": []}
GRID_STRIDE = 2
try:
    video = robomme_plus.get_conditioning_video()
    nvid = len(video)
    vs = {"length": nvid}
    if nvid > 0:
        vs["t_first"] = round(float(video[0].timestamp_s), 4)
        vs["t_last"] = round(float(video[nvid - 1].timestamp_s), 4)
        try:
            vs["cameras"] = [str(name) for name in video[0].cameras]
            fc = cam_params(video[0].cameras[cam_front])
            vs["front_pose_first"] = flist(fc["t"]) + flist(fc["q"])
            if live_obs is not None:
                lc = cam_params(live_obs.cameras[cam_front])
                vs["front_pose_live"] = flist(lc["t"]) + flist(lc["q"])
                vs["front_pose_delta"] = round(sum(abs(fc["t"][k] - lc["t"][k])
                                                   for k in range(3)), 5)
        except Exception as exc:
            err("video-pose", exc)
    R["video"] = vs
    log("video frames " + str(nvid))
except Exception as exc:
    err("video", exc)
    video = []

try:
    nvid = len(video)
    if nvid > 0:
        ksamp = min(nvid, 32)
        idxs = []
        for i in range(ksamp):
            if ksamp == 1:
                idx = 0
            else:
                idx = int(round(i * (nvid - 1) / (ksamp - 1)))
            if idx not in idxs:
                idxs.append(idx)
        gw = None
        for idx in idxs:
            frame = video[idx]
            rgb = to_list(frame.cameras[cam_front].rgb)
            if gw is None:
                VID["H"] = len(rgb)
                VID["W"] = len(rgb[0])
                gw = (VID["W"] + GRID_STRIDE - 1) // GRID_STRIDE
                VID["GW"] = gw
                VID["GH"] = (VID["H"] + GRID_STRIDE - 1) // GRID_STRIDE
            grid = label_grid(rgb, GRID_STRIDE)
            counts = [0] * len(LABELS)
            for lab in grid:
                counts[lab] += 1
            VID["frames"].append(idx)
            VID["grids"].append(grid)
            VID["counts"].append(counts)
            VID["times"].append(round(float(frame.timestamp_s), 3))
        R["video"]["sampled_idx"] = VID["frames"]
        R["video"]["sampled_t"] = VID["times"]
        table = {}
        for li in range(1, len(LABELS)):
            table[LABELS[li]] = [counts[li] for counts in VID["counts"]]
        R["video"]["label_counts"] = table
        log("video sampled " + str(len(VID["frames"])) + " frames")
except Exception as exc:
    err("video-labels", exc)


def find_clusters(grid, labset, gw, gh):
    seen = {}
    clusters = []
    for start in range(len(grid)):
        if grid[start] not in labset or start in seen:
            continue
        stack = [start]
        seen[start] = True
        cells = []
        while len(stack) > 0:
            cur = stack.pop()
            cells.append(cur)
            cy = cur // gw
            cx = cur % gw
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny = cy + dy
                    nx = cx + dx
                    if ny < 0 or nx < 0 or ny >= gh or nx >= gw:
                        continue
                    nid = ny * gw + nx
                    if nid in seen or grid[nid] not in labset:
                        continue
                    seen[nid] = True
                    stack.append(nid)
        clusters.append(cells)
    return clusters


FRAME_CACHE = {}


def frame_geom(sample_i):
    if sample_i in FRAME_CACHE:
        return FRAME_CACHE[sample_i]
    cam = video[VID["frames"][sample_i]].cameras[cam_front]
    g = {"dep": to_list(cam.depth_m), "cp": cam_params(cam)}
    FRAME_CACHE[sample_i] = g
    return g


def cluster_geom(cells, sample_i, gw):
    """3-D summary of a video blob from that frame's own depth and camera pose."""
    g = frame_geom(sample_i)
    pts = []
    stride = max(1, len(cells) // 60)
    for k in range(0, len(cells), stride):
        c = cells[k]
        u = GRID_STRIDE * (c % gw)
        v = GRID_STRIDE * (c // gw)
        d = float(g["dep"][v][u])
        if valid_depth(d):
            pts.append(backproject(u, v, d, g["cp"], CALIB["variant"]))
    if len(pts) < 3:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    tz = TABLE["z"]
    if tz is None:
        tz = min(zs)
    return {"x": round(median(xs), 4), "y": round(median(ys), 4),
            "z95_rel": round(quantile(zs, 0.95) - tz, 4),
            "span": round(max(quantile(xs, 0.95) - quantile(xs, 0.05),
                              quantile(ys, 0.95) - quantile(ys, 0.05)), 4)}


def analyse_colour(labset):
    """Find the named cube in the video: a small blob of labset resting on the table,
    preferably one that is covered (disappears) later in the video."""
    out = {"labels": [LABELS[li] for li in labset]}
    if len(VID["grids"]) == 0:
        out["error"] = "no video"
        return out
    gw = VID["GW"]
    gh = VID["GH"]
    totals = []
    for counts in VID["counts"]:
        totals.append(sum(counts[li] for li in labset))
    fbest = 0
    for i in range(len(totals)):
        if totals[i] > totals[fbest]:
            fbest = i
    out["fmax"] = fbest
    out["total_series"] = totals
    scan = [0]
    for extra in (fbest, len(VID["grids"]) // 4):
        if extra not in scan:
            scan.append(extra)
    cand = []
    for fs in scan:
        clusters = find_clusters(VID["grids"][fs], labset, gw, gh)
        for cells in clusters:
            if len(cells) < 3 or len(cells) > 600:
                continue
            gxs = [c % gw for c in cells]
            gys = [c // gw for c in cells]
            x0 = max(0, min(gxs) - 1)
            x1 = min(gw - 1, max(gxs) + 1)
            y0 = max(0, min(gys) - 1)
            y1 = min(gh - 1, max(gys) + 1)
            series = []
            for grid in VID["grids"]:
                cnt = 0
                for yy in range(y0, y1 + 1):
                    base = yy * gw
                    for xx in range(x0, x1 + 1):
                        if grid[base + xx] in labset:
                            cnt += 1
                series.append(cnt)
            mx = max(series)
            drop = 0.0
            if mx > 0:
                drop = 1.0 - series[len(series) - 1] / mx
            geom = None
            try:
                geom = cluster_geom(cells, fs, gw)
            except Exception as exc:
                err("cluster_geom", exc)
            low = geom is not None and geom["z95_rel"] < 0.07 and geom["span"] < 0.09
            cand.append({"cells": len(cells), "bbox_grid": [x0, y0, x1, y1],
                         "series": series, "drop": round(drop, 3), "found_in_sample": fs,
                         "geom": geom, "low": low,
                         "centroid_px": [round(GRID_STRIDE * sum(gxs) / len(gxs), 1),
                                         round(GRID_STRIDE * sum(gys) / len(gys), 1)]})
    best = None
    for c in cand:
        zrel = 0.0
        if c["geom"] is not None:
            zrel = round(c["geom"]["z95_rel"] / 0.02) * 0.02
        key = (1 if c["low"] else 0, 1 if c["drop"] >= 0.5 else 0, -zrel, c["cells"])
        if best is None or key > best[0]:
            best = (key, c)
    out["clusters"] = cand[:10]
    if best is None:
        out["error"] = "no compact cluster"
        return out
    chosen = best[1]
    out["chosen"] = chosen
    series = chosen["series"]
    mx = max(series)
    tmax = 0
    for i in range(len(series)):
        if series[i] == mx:
            tmax = i
            break
    tlast = tmax
    while tlast + 1 < len(series) and series[tlast + 1] >= 0.6 * mx:
        tlast += 1
    tcover = None
    for i in range(tlast + 1, len(series)):
        if series[i] < 0.3 * mx:
            tcover = i
            break
    out["t_max_sample"] = tmax
    out["t_last_visible_sample"] = tlast
    out["t_cover_sample"] = tcover
    out["covered_at_end"] = series[len(series) - 1] < 0.3 * mx
    return out


def refine_cube_pixels(info, labset, sample_i):
    """Full-resolution pixels of the chosen blob in one sampled frame, with depth."""
    idx = VID["frames"][sample_i]
    frame = video[idx]
    cam = frame.cameras[cam_front]
    rgb = to_list(cam.rgb)
    dep = to_list(cam.depth_m)
    x0, y0, x1, y1 = info["chosen"]["bbox_grid"]
    u0 = max(0, GRID_STRIDE * x0 - 3)
    u1 = min(VID["W"] - 1, GRID_STRIDE * x1 + 4)
    v0 = max(0, GRID_STRIDE * y0 - 3)
    v1 = min(VID["H"] - 1, GRID_STRIDE * y1 + 4)
    pix = []
    for v in range(v0, v1 + 1):
        for u in range(u0, u1 + 1):
            px = rgb[v][u]
            if label_of(px[0], px[1], px[2]) in labset:
                d = float(dep[v][u])
                if valid_depth(d):
                    pix.append((u, v, d))
    return {"sample": sample_i, "frame_idx": idx, "pix": pix, "cp": cam_params(cam),
            "t": round(float(frame.timestamp_s), 3)}


# ================================================================ PHASE 2: SAM probes
FRONT_PROMPTS = ["container", "bin", "cup", "box", "cube", "table"]
for cname in cube_targets[:2]:
    FRONT_PROMPTS.append(cname + " cube")
WRIST_PROMPTS = ["container", "cube"]
SEGS = {}
try:
    probes = []
    for prompt in FRONT_PROMPTS:
        try:
            seg = segment_text(cam_front, prompt)
            SEGS[prompt] = seg
            s = seg_summary(seg)
            s["camera"] = cam_front
            s["prompt"] = prompt
            probes.append(s)
        except Exception as exc:
            probes.append({"camera": cam_front, "prompt": prompt, "error": short(exc)})
    for prompt in WRIST_PROMPTS:
        try:
            seg = segment_text(cam_wrist, prompt)
            s = seg_summary(seg)
            s["camera"] = cam_wrist
            s["prompt"] = prompt
            probes.append(s)
        except Exception as exc:
            probes.append({"camera": cam_wrist, "prompt": prompt, "error": short(exc)})
    R["prompt_probes"] = probes
    log("probes done: " + str(len(probes)))
except Exception as exc:
    err("probes", exc)

# ---------------------------------------------------------------- front container candidates
CONTAINER_PROMPTS = ["container", "bin", "cup", "box"]
FRONT_CANDS = []
calib_samples = []
try:
    for prompt in CONTAINER_PROMPTS:
        seg = SEGS.get(prompt)
        if seg is None:
            continue
        for rank, cand in enumerate(seg_candidates(seg, 5)):
            pts = cloud_of(cand, cam_front, base_frame)
            st = cloud_stats(pts)
            rec = {"prompt": prompt, "rank": rank, "score": rscore(cand),
                   "box": box_of(cand), "stats": st, "cand": cand}
            FRONT_CANDS.append(rec)
            if st is not None and st["n"] >= 30 and len(calib_samples) < 4:
                try:
                    mp = mask_pixels(cand, 2)
                    if len(mp) >= 10:
                        calib_samples.append((mp, st))
                except Exception as exc:
                    err("mask-pixels", exc)
    R["front_candidates"] = [{"prompt": c["prompt"], "rank": c["rank"], "score": c["score"],
                              "box": c["box"], "stats": round_stats(c["stats"])}
                             for c in FRONT_CANDS]
    log("front container candidates " + str(len(FRONT_CANDS)))
except Exception as exc:
    err("front-candidates", exc)

# ================================================================ PHASE 3: calibration
live_front_cp = None
live_front_depth = None
TABLE = {"z": None}
try:
    live_front_cp = cam_params(live_obs.cameras[cam_front])
    live_front_depth = to_list(live_obs.cameras[cam_front].depth_m)
    cal = {"cloud": [], "plane": []}
    if len(calib_samples) > 0:
        best = None
        for variant in VARIANTS:
            total = 0.0
            for sample in calib_samples:
                mp = sample[0]
                stv = sample[1]
                acc = [0.0, 0.0, 0.0]
                cnt = 0
                for (u, v) in mp:
                    d = float(live_front_depth[v][u])
                    if not valid_depth(d):
                        continue
                    p = backproject(u, v, d, live_front_cp, variant)
                    acc[0] += p[0]
                    acc[1] += p[1]
                    acc[2] += p[2]
                    cnt += 1
                if cnt == 0:
                    total += 9.0
                    continue
                m = stv["mean"]
                total += math.sqrt((acc[0] / cnt - m[0]) ** 2 + (acc[1] / cnt - m[1]) ** 2 +
                                   (acc[2] / cnt - m[2]) ** 2)
            total = total / len(calib_samples)
            cal["cloud"].append([variant[0], variant[1], round(total, 4)])
            if best is None or total < best[0]:
                best = (total, variant)
        if best is not None and best[0] < 0.03:
            CALIB["variant"] = best[1]
            CALIB["source"] = "cloud"
            CALIB["cloud_err"] = round(best[0], 4)
    H = len(live_front_depth)
    W = len(live_front_depth[0])
    plane_best = None
    for variant in VARIANTS:
        zs = []
        for v in range(0, H, 8):
            for u in range(0, W, 8):
                d = float(live_front_depth[v][u])
                if valid_depth(d):
                    zs.append(backproject(u, v, d, live_front_cp, variant)[2])
        if len(zs) == 0:
            continue
        bins = {}
        for z in zs:
            key = int(math.floor(z / 0.01))
            bins[key] = bins.get(key, 0) + 1
        bestkey = None
        bestcnt = -1
        for key in bins:
            cnt = bins[key] + bins.get(key + 1, 0)
            if cnt > bestcnt:
                bestcnt = cnt
                bestkey = key
        sel = [z for z in zs if z >= bestkey * 0.01 and z < (bestkey + 2) * 0.01]
        zt = median(sel)
        frac = bestcnt / len(zs)
        plaus = zt < live_front_cp["t"][2] - 0.1
        cal["plane"].append([variant[0], variant[1], round(frac, 3), round(zt, 4), plaus])
        if variant == CALIB["variant"]:
            TABLE["z"] = zt
            TABLE["frac"] = frac
        if plaus and (plane_best is None or frac > plane_best[0]):
            plane_best = (frac, variant, zt)
    if CALIB["source"] != "cloud" and plane_best is not None and plane_best[0] > 0.25:
        CALIB["variant"] = plane_best[1]
        CALIB["source"] = "plane"
        TABLE["z"] = plane_best[2]
        TABLE["frac"] = plane_best[0]
    R["calibration"] = {"chosen": [CALIB["variant"][0], CALIB["variant"][1]],
                        "source": CALIB["source"], "detail": cal,
                        "table_z": None if TABLE["z"] is None else round(TABLE["z"], 4)}
    log("calibration " + short(R["calibration"]["chosen"]) + " via " + CALIB["source"] +
        " table_z " + short(TABLE["z"]))
except Exception as exc:
    err("calibration", exc)

# table height fallback from container candidates
try:
    if TABLE["z"] is None:
        lows = [c["stats"]["z05"] for c in FRONT_CANDS if c["stats"] is not None]
        if len(lows) > 0:
            TABLE["z"] = min(lows)
            log("table z from candidate lows " + short(TABLE["z"]))
except Exception as exc:
    err("table-fallback", exc)

# ---------------------------------------------------------------- video target (needs calibration)
TARGETS = []
try:
    names_to_try = list(cube_targets)
    if len(names_to_try) == 0:
        names_to_try = ["*any*"]
    for cname in names_to_try:
        if cname == "*any*":
            labset = [1, 2, 3, 4, 5, 6, 7, 8]
        else:
            labset = list(COLOR_WORDS[cname])
        info = analyse_colour(labset)
        if "chosen" not in info or not info["chosen"]["low"]:
            wider = list(labset)
            for li in labset:
                for nb in NEIGHBOR_LABELS.get(li, []):
                    if nb not in wider:
                        wider.append(nb)
            info2 = analyse_colour(wider)
            info2["widened_from"] = [LABELS[li] for li in labset]
            if "chosen" in info2 and info2["chosen"]["low"] and ("chosen" not in info or
                                                                 not info["chosen"]["low"]):
                info = info2
                labset = wider
        if ("chosen" not in info or not info["chosen"]["low"]) and cname != "*any*":
            # last resort, logged as a guess: any low blob that gets covered in the video
            info3 = analyse_colour([1, 2, 3, 4, 5, 6, 7, 8, 11, 12])
            if "chosen" in info3 and info3["chosen"]["low"]:
                info3["fallback_any_from"] = cname
                info = info3
                labset = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
                log("named colour " + cname + " not found as a low blob; GUESS from any colour")
        rec = {"color": cname, "labset": labset, "video": info}
        if "chosen" in info:
            ra = refine_cube_pixels(info, labset, info["t_max_sample"])
            rb = ra
            if info["t_last_visible_sample"] != info["t_max_sample"]:
                rb = refine_cube_pixels(info, labset, info["t_last_visible_sample"])
            use = ra
            if len(ra["pix"]) > 0 and len(rb["pix"]) > 0:
                ua = median([p[0] for p in ra["pix"]])
                va = median([p[1] for p in ra["pix"]])
                ub = median([p[0] for p in rb["pix"]])
                vb = median([p[1] for p in rb["pix"]])
                if abs(ua - ub) + abs(va - vb) > 4:
                    use = rb
            elif len(ra["pix"]) == 0:
                use = rb
            rec["ref"] = use
            if len(use["pix"]) > 0:
                rec["pixel"] = [median([p[0] for p in use["pix"]]),
                                median([p[1] for p in use["pix"]])]
                rec["pixel_depth"] = round(median([p[2] for p in use["pix"]]), 4)
                rec["npix"] = len(use["pix"])
        TARGETS.append(rec)
        summary = {"color": cname, "labset": [LABELS[li] for li in labset]}
        for key in ("fmax", "t_max_sample", "t_last_visible_sample", "t_cover_sample",
                    "covered_at_end", "error", "widened_from", "total_series",
                    "fallback_any_from"):
            if key in info:
                summary[key] = info[key]
        if "chosen" in info:
            summary["chosen"] = info["chosen"]
            summary["n_clusters"] = len(info.get("clusters", []))
            summary["other_clusters"] = [[c["cells"], c["drop"], c["low"], c["centroid_px"],
                                          c["geom"], c["found_in_sample"]]
                                         for c in info.get("clusters", [])][:8]
        for key in ("pixel", "pixel_depth", "npix"):
            if key in rec:
                summary[key] = rec[key]
        if "ref" in rec:
            summary["ref_frame_idx"] = rec["ref"]["frame_idx"]
            summary["ref_t"] = rec["ref"]["t"]
        if "video_targets" not in R:
            R["video_targets"] = []
        R["video_targets"].append(summary)
        log("video target " + cname + ": " + short(summary.get("pixel")) + " drop " +
            short(info.get("chosen", {}).get("drop")))
except Exception as exc:
    err("video-target", exc)

# ================================================================ PHASE 4: target -> container
PLAN = []


def first_item(item):
    return item[0]


def project_video_cube(rec):
    ref = rec["ref"]
    pts = []
    for (u, v, d) in ref["pix"]:
        pts.append(backproject(u, v, d, ref["cp"], CALIB["variant"]))
    if len(pts) == 0:
        return None
    return top_band_center(pts, 0.006)


def top_band_center(pts, band):
    """XY of the upward-facing surface (a raised camera sees it whole), median z."""
    zs = [p[2] for p in pts]
    ztop = quantile(zs, 0.95)
    sel = [p for p in pts if p[2] >= ztop - band]
    if len(sel) < 5:
        sel = pts
    return (sum(p[0] for p in sel) / len(sel), sum(p[1] for p in sel) / len(sel), median(zs))


def live_column_top(cx, cy, radius, upix, vpix, win):
    """Top of whatever stands at (cx, cy) in the live front depth (own projection)."""
    H = len(live_front_depth)
    W = len(live_front_depth[0])
    zs = []
    u0 = max(0, int(upix) - win)
    u1 = min(W - 1, int(upix) + win)
    v0 = max(0, int(vpix) - win)
    v1 = min(H - 1, int(vpix) + win)
    for v in range(v0, v1 + 1):
        for u in range(u0, u1 + 1):
            d = float(live_front_depth[v][u])
            if not valid_depth(d):
                continue
            p = backproject(u, v, d, live_front_cp, CALIB["variant"])
            if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 <= radius * radius:
                zs.append(p)
    return zs


try:
    for rec in TARGETS:
        plan = {"color": rec["color"]}
        if "ref" not in rec or len(rec["ref"]["pix"]) == 0:
            plan["error"] = "cube not found in video"
            PLAN.append(plan)
            continue
        cube = project_video_cube(rec)
        plan["cube_xyz_video"] = flist(cube)
        # SAM front candidates: nearest in XY to the remembered cube
        ranked = []
        for c in FRONT_CANDS:
            st = c["stats"]
            if st is None or st["n"] < 15:
                continue
            height = st["z95"] - (TABLE["z"] if TABLE["z"] is not None else st["z05"])
            if st["xspan"] > 0.25 or st["yspan"] > 0.25 or height > 0.30:
                continue
            dxy = math.sqrt((st["cx"] - cube[0]) ** 2 + (st["cy"] - cube[1]) ** 2)
            ranked.append((dxy, c))
        ranked = sorted(ranked, key=first_item)
        plan["sam_ranked"] = [[round(item[0], 4), item[1]["prompt"], item[1]["rank"],
                               round_stats(item[1]["stats"])] for item in ranked[:6]]
        # pixel-space cross-check: which SAM box contains the remembered pixel
        upix = rec["pixel"][0]
        vpix = rec["pixel"][1]
        inbox = []
        for c in FRONT_CANDS:
            b = c["box"]
            if b is not None and b[0] <= upix <= b[2] and b[1] <= vpix <= b[3]:
                inbox.append([c["prompt"], c["rank"], flist(b, 1)])
        plan["boxes_containing_pixel"] = inbox[:6]
        colpts = live_column_top(cube[0], cube[1], 0.045, upix, vpix, 30)
        plan["live_column_n"] = len(colpts)
        top = None
        col_xy = None
        if len(colpts) >= 8:
            top = quantile([p[2] for p in colpts], 0.9)
            plan["live_column_top"] = round(top, 4)
            ctr = top_band_center(colpts, 0.008)
            col_xy = (ctr[0], ctr[1])
            plan["live_column_top_xy"] = [round(ctr[0], 4), round(ctr[1], 4)]
        sam_pick = None
        if len(ranked) > 0 and ranked[0][0] < 0.06:
            sam_pick = ranked[0][1]["stats"]
            plan["sam_pick_dxy"] = round(ranked[0][0], 4)
        cx = cube[0]
        cy = cube[1]
        if col_xy is not None and math.sqrt((col_xy[0] - cx) ** 2 +
                                            (col_xy[1] - cy) ** 2) < 0.03:
            # container top surface seen live, consistent with the remembered cube
            cx = col_xy[0]
            cy = col_xy[1]
            plan["xy_source"] = "live_column_top"
        else:
            plan["xy_source"] = "video_cube"
        if top is None and sam_pick is not None:
            top = sam_pick["z95"]
            plan["top_source"] = "sam"
        elif top is not None:
            plan["top_source"] = "live_depth"
        if top is None:
            top = cube[2] + 0.04
            plan["top_source"] = "cube+0.04"
        if CALIB["source"] == "default" and sam_pick is None and len(inbox) > 0:
            # projection unverified: trust the harness cloud of the box holding the pixel
            for c in FRONT_CANDS:
                if c["prompt"] == inbox[0][0] and c["rank"] == inbox[0][1] and \
                        c["stats"] is not None:
                    cx = c["stats"]["cx"]
                    cy = c["stats"]["cy"]
                    top = c["stats"]["z95"]
                    plan["xy_source"] = "pixel-box"
        tz = TABLE["z"]
        if tz is None:
            tz = cube[2] - 0.02
        plan["table_z"] = round(tz, 4)
        plan["center"] = [round(cx, 4), round(cy, 4)]
        plan["top"] = round(top, 4)
        plan["height"] = round(top - tz, 4)
        plan["ok"] = True
        PLAN.append(plan)
        log("plan " + rec["color"] + ": " + short({"center": plan["center"],
                                                    "top": plan["top"],
                                                    "h": plan["height"]}))
    R["plan"] = PLAN
except Exception as exc:
    err("plan", exc)
    R["plan"] = PLAN

# ================================================================ PHASE 5: motion
DOWN = [0.0, 1.0, 0.0, 0.0]
try:
    q0 = [float(item) for item in home_state.end_effector_poses["primary"].quaternion_wxyz]
    Rq = quat_to_rot(q0)
    if Rq[2][2] < -0.9:
        DOWN = quat_norm(q0)
        R["down_source"] = "home"
    else:
        R["down_source"] = "fixed-x180"
except Exception as exc:
    err("down-quat", exc)
    R["down_source"] = "fixed-x180"


def yaw_quat(yaw):
    qz = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
    return quat_norm(quat_mul(qz, DOWN))


def finger_axis_angle(q):
    Rm = quat_to_rot(q)
    return math.atan2(Rm[1][1], Rm[0][1])


def go(name, pos, yaw, tol, steps):
    if not can_move():
        log("skip " + name + " (terminal or no budget)")
        return {"ok": False, "skipped": True}
    try:
        pose = Pose([float(pos[0]), float(pos[1]), float(pos[2])], yaw_quat(yaw), base_frame)
        res = move_to_pose(pose, tolerance=tol, max_steps=budget(steps))
        rec = note_motion(name, res)
        text = (short(rec.get("status")) + " " + short(rec.get("error"))).lower()
        if rec.get("ok") is False and rec.get("steps", 0) == 0 and "exceeds max_steps" in text \
                and can_move() and budget(int(steps * 2.5)) > budget(steps):
            res = move_to_pose(pose, tolerance=tol, max_steps=budget(int(steps * 2.5)))
            rec = note_motion(name + "_longer", res)
        return rec
    except Exception as exc:
        err("move " + name, exc)
        return {"ok": False, "exception": short(exc)}


def gripper_now():
    try:
        return float(get_robot_state().gripper_positions["primary"])
    except Exception:
        return None


def rect_fit(pts):
    """Minimum-area rectangle over XY points: (angle, extent_a, extent_b, cx, cy)."""
    best = None
    for step_i in range(18):
        ang = step_i * math.pi / 36.0
        ca = math.cos(ang)
        sa = math.sin(ang)
        a = [p[0] * ca + p[1] * sa for p in pts]
        b = [-p[0] * sa + p[1] * ca for p in pts]
        ea = quantile(a, 0.97) - quantile(a, 0.03)
        eb = quantile(b, 0.97) - quantile(b, 0.03)
        if best is None or ea * eb < best[0]:
            ma = (quantile(a, 0.97) + quantile(a, 0.03)) / 2.0
            mb = (quantile(b, 0.97) + quantile(b, 0.03)) / 2.0
            best = (ea * eb, ang, ea, eb, ma * ca - mb * sa, ma * sa + mb * ca)
    return best


def wrist_refine(cx, cy, top):
    """Re-localise the container from the hover pose (wrist camera, harness clouds)."""
    info = {"expected": [round(cx, 4), round(cy, 4), round(top, 4)], "cands": []}
    best = None
    for prompt in CONTAINER_PROMPTS:
        try:
            seg = segment_text(cam_wrist, prompt)
        except Exception as exc:
            info["cands"].append({"prompt": prompt, "error": short(exc)})
            continue
        for rank, cand in enumerate(seg_candidates(seg, 5)):
            pts = cloud_of(cand, cam_wrist, base_frame)
            st = cloud_stats(pts)
            if st is None or st["n"] < 15:
                continue
            dxy = math.sqrt((st["cx"] - cx) ** 2 + (st["cy"] - cy) ** 2)
            dz = st["z95"] - top
            info["cands"].append({"prompt": prompt, "rank": rank,
                                  "score": rscore(cand),
                                  "dxy": round(dxy, 4), "dz": round(dz, 4),
                                  "stats": round_stats(st)})
            if dxy < 0.05 and abs(dz) < 0.04 and st["xspan"] < 0.2 and st["yspan"] < 0.2:
                if best is None or dxy < best[0]:
                    best = (dxy, pts, st, prompt, rank)
        if best is not None and best[0] < 0.025:
            break
    if best is None:
        # SAM-free fallback: top surface around the expected centre in the wrist depth
        try:
            obs = get_observation()
            cam = obs.cameras[cam_wrist]
            cp = cam_params(cam)
            dep = to_list(cam.depth_m)
            pts = []
            for v in range(0, len(dep), 2):
                row = dep[v]
                for u in range(0, len(row), 2):
                    d = float(row[u])
                    if not valid_depth(d):
                        continue
                    p = backproject(u, v, d, cp, CALIB["variant"])
                    if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 < 0.05 * 0.05 and \
                            top - 0.03 < p[2] < top + 0.03:
                        pts.append(p)
            info["depth_fallback_n"] = len(pts)
            if len(pts) >= 15:
                best = (0.0, pts, None, "wrist-depth", -1)
        except Exception as exc:
            err("wrist-depth-fallback", exc)
    if best is None:
        info["result"] = "no consistent wrist candidate"
        return None, info
    pts = best[1]
    zt = quantile([p[2] for p in pts], 0.95)
    toppts = [p for p in pts if p[2] >= zt - 0.012]
    if len(toppts) < 8:
        toppts = pts
    fit = rect_fit(toppts)
    out = {"cx": fit[4], "cy": fit[5], "top": zt, "angle": fit[1], "ea": fit[2], "eb": fit[3],
           "n_top": len(toppts), "prompt": best[3], "rank": best[4]}
    info["result"] = {"cx": round(fit[4], 4), "cy": round(fit[5], 4), "top": round(zt, 4),
                      "angle_deg": round(fit[1] * 180 / math.pi, 1), "ea": round(fit[2], 4),
                      "eb": round(fit[3], 4), "n_top": len(toppts), "prompt": best[3],
                      "rank": best[4]}
    return out, info


def front_reveal_count(rec):
    """How many target-colour pixels are visible near the remembered pixel now."""
    try:
        obs = get_observation()
        rgb = to_list(obs.cameras[cam_front].rgb)
        upix = int(rec["pixel"][0])
        vpix = int(rec["pixel"][1])
        cnt = 0
        for v in range(max(0, vpix - 14), min(len(rgb), vpix + 15)):
            for u in range(max(0, upix - 14), min(len(rgb[0]), upix + 15)):
                px = rgb[v][u]
                if label_of(px[0], px[1], px[2]) in rec["labset"]:
                    cnt += 1
        return cnt
    except Exception as exc:
        err("reveal", exc)
        return None


def hold(ticks, grip):
    done = 0
    for tick in range(ticks):
        if not can_move(2):
            break
        try:
            st = get_robot_state()
            reply = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=[float(x) for x in st.joint_positions["primary"]],
                gripper_position=grip, embodiment=st.embodiment)}))
            ST["steps"] += 1
            done += 1
            if flag_terminal(reply):
                ST["terminal"] = True
                log("terminal during hold at tick " + str(tick))
                break
        except Exception as exc:
            err("hold", exc)
            break
    R["motions"].append({"name": "hold", "ticks": done, "terminal": ST["terminal"],
                         "cum_steps": ST["steps"]})
    return done


ATTEMPTS = []
try:
    rec_by_color = {}
    for rec in TARGETS:
        rec_by_color[rec["color"]] = rec
    usable = [p for p in PLAN if p.get("ok")]
    if len(usable) == 0:
        log("no usable plan; snapshot only")
    try:
        if can_move():
            note_motion("open_gripper", open_gripper())
    except Exception as exc:
        err("open", exc)
    for pi, plan in enumerate(usable):
        if not can_move():
            break
        att = {"color": plan["color"]}
        ATTEMPTS.append(att)
        rec = rec_by_color.get(plan["color"])
        cx = plan["center"][0]
        cy = plan["center"][1]
        top = plan["top"]
        tz = plan["table_z"]
        yaw = 0.0
        hover_z = top + 0.15
        att["hover"] = go("hover", (cx, cy, hover_z), yaw, 0.015, 160)
        refined, winfo = None, {}
        try:
            refined, winfo = wrist_refine(cx, cy, top)
        except Exception as exc:
            err("wrist_refine", exc)
        att["wrist"] = winfo
        width = None
        if refined is not None and math.sqrt((refined["cx"] - cx) ** 2 +
                                             (refined["cy"] - cy) ** 2) < 0.05:
            cx = refined["cx"]
            cy = refined["cy"]
            if abs(refined["top"] - top) < 0.03:
                top = refined["top"]
            narrow = refined["angle"]
            width = refined["ea"]
            if refined["eb"] < refined["ea"]:
                narrow = refined["angle"] + math.pi / 2.0
                width = refined["eb"]
            yaw = wrap_half_pi(narrow - finger_axis_angle(DOWN))
        att["final_target"] = {"cx": round(cx, 4), "cy": round(cy, 4), "top": round(top, 4),
                               "yaw_deg": round(yaw * 180 / math.pi, 1),
                               "finger_width_needed": None if width is None else round(width, 4)}
        height = top - tz
        depth_in = min(0.03, max(0.01, 0.5 * height))
        grasp_z = max(top - depth_in, tz + 0.012)
        att["grasp_z"] = round(grasp_z, 4)
        grasped = False
        for trial in range(2):
            if not can_move():
                break
            pre = go("pregrasp" + str(trial), (cx, cy, top + 0.06), yaw, 0.01, 120)
            if pre.get("ok") is False and not ST["terminal"]:
                yaw = wrap_half_pi(yaw + math.pi / 2.0)
                pre = go("pregrasp_rot" + str(trial), (cx, cy, top + 0.06), yaw, 0.01, 120)
            down = go("descend" + str(trial), (cx, cy, grasp_z), yaw, 0.008, 80)
            if not can_move():
                break
            try:
                note_motion("close" + str(trial), close_gripper())
            except Exception as exc:
                err("close", exc)
            g = gripper_now()
            att["grip_after_close" + str(trial)] = g
            if g is not None and g > 0.03:
                grasped = True
                break
            log("gripper closed fully (" + short(g) + "): miss, retry rotated & deeper")
            if not can_move():
                break
            try:
                note_motion("reopen" + str(trial), open_gripper())
            except Exception as exc:
                err("reopen", exc)
            go("retreat" + str(trial), (cx, cy, top + 0.06), yaw, 0.015, 80)
            yaw = wrap_half_pi(yaw + math.pi / 2.0)
            grasp_z = max(grasp_z - 0.01, tz + 0.012)
        att["grasped"] = grasped
        lift_z = grasp_z + 0.15
        if can_move():
            att["lift"] = go("lift", (cx, cy, lift_z), yaw, 0.015, 120)
        if rec is not None and "pixel" in rec:
            att["reveal_count_after_lift"] = front_reveal_count(rec)
        last = (pi == len(usable) - 1)
        if last:
            hold(20, 0.0)
        elif can_move():
            # more cubes named: put this container back where it was before the next one
            go("putback", (cx, cy, grasp_z + 0.005), yaw, 0.01, 120)
            try:
                if can_move():
                    note_motion("release", open_gripper())
            except Exception as exc:
                err("release", exc)
            go("retreat_up", (cx, cy, top + 0.15), yaw, 0.02, 100)
    R["attempts"] = ATTEMPTS
except Exception as exc:
    err("motion", exc)
    R["attempts"] = ATTEMPTS

R["terminal"] = ST["terminal"]
R["steps_used"] = ST["steps"]
try:
    st = get_robot_state()
    R["final_state"] = {"ee": flist(st.end_effector_poses["primary"].position),
                        "grip": round(float(st.gripper_positions["primary"]), 4),
                        "t": round(float(st.timestamp_s), 3)}
except Exception as exc:
    err("final", exc)
log("done; steps " + str(ST["steps"]) + " terminal " + str(ST["terminal"]))
result = R
