# v01_track_target_swaps.py -- VideoPlaceButton (robomme_plus task 11), Stage 1 repair of v00.
#
# v00 (0/1, episode 0): picked the red cube and placed it exactly where the video released it
# first after the button press, and the episode terminated with failure at the release.
# The trace shows why: the four identical targets swap places pairwise 12 times during the video,
# so the release *position* is stale; the target *object* moved.  v01 adds Phase C2, which tracks
# the chosen target's identity through every later swap from the video's front frames,
# and a colour check of the cube named in the language.  All v00 motion phases are kept unchanged.
#
# Phase A: snapshot (language, metadata, cameras, robot state, SAM3 prompt probes on both cameras).
# Phase B: conditioning video -> per-frame robot-state series -> EE low-point events
#          (grasp / release / contact) -> button-press time and the ordered placement list.
# Phase C: parse the runtime language (before/after/ordinal) and choose the release event.
# Phase C2: track that target through the video's swaps to its live slot (v01).
# Phase D: pick the cube (if not already held) and place it on the chosen target.
# Every phase is wrapped; the program never raises.  No world coordinates are hard-coded:
# all positions come from live RGB-D segmentation or from the episode's own conditioning video.

import math

report = {"version": "v01_track_target_swaps", "phases": {}, "errors": [], "log": []}
result = report


def note(msg):
    text = str(msg)
    report["log"].append(text)
    print("[v01] " + text)


def err(phase, exc):
    text = phase + ": " + str(exc)
    report["errors"].append(text)
    print("[v01][ERR] " + text)


def fl(v):
    return float(v)


def xyz(p):
    return [fl(p[0]), fl(p[1]), fl(p[2])]


def quantile(values, frac):
    ordered = sorted(values)
    if len(ordered) == 0:
        return None
    idx = int((len(ordered) - 1) * frac)
    return ordered[idx]


def dist2(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def qaxis(q, col):
    w, x, y, z = q
    if col == 0:
        return [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)]
    if col == 1:
        return [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)]
    return [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]


def neg_score(item):
    return -item["score"]


def is_terminal(res):
    flag = False
    try:
        flag = flag or bool(res.terminated)
    except Exception:
        pass
    try:
        flag = flag or bool(res.truncated)
    except Exception:
        pass
    return flag


ended = {"value": False}


def summarize_exec(name, res):
    entry = {"name": name}
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
        entry["error"] = str(res.error)
    except Exception:
        pass
    entry["terminal"] = is_terminal(res)
    try:
        st = get_robot_state()
        entry["t"] = round(fl(st.timestamp_s), 3)
        entry["ee"] = [round(v, 4) for v in xyz(st.end_effector_poses["primary"].position)]
        entry["g"] = round(fl(st.gripper_positions["primary"]), 3)
    except Exception:
        pass
    report["motions"].append(entry)
    note("motion " + name + " -> " + str(entry))
    if entry["terminal"]:
        ended["value"] = True
    return entry


report["motions"] = []

# ---------------------------------------------------------------- Phase A: snapshot
base = "robot_base"
lang = ""
max_steps = None
hz = 20.0
cams = []
try:
    ctx = get_task_context()
    lang = str(ctx.language)
    report["language"] = lang
    try:
        report["task_name"] = str(ctx.task_name)
        report["task_id"] = str(ctx.task_id)
    except Exception:
        pass
    note("language: " + lang)
except Exception as exc:
    err("context", exc)

try:
    meta = robomme_plus.get_task_metadata()
    md = {}
    for k in meta:
        try:
            md[str(k)] = str(meta[k])[:400]
        except Exception:
            md[str(k)] = "?"
    report["task_metadata"] = md
    try:
        max_steps = int(meta["max_steps"])
    except Exception:
        max_steps = None
    try:
        if len(lang) == 0:
            lang = str(meta["language"])
    except Exception:
        pass
except Exception as exc:
    err("task_metadata", exc)

try:
    cmeta = robomme_plus.get_controller_metadata()
    cm = {}
    for k in cmeta:
        try:
            cm[str(k)] = str(cmeta[k])[:400]
        except Exception:
            cm[str(k)] = "?"
    report["controller_metadata"] = cm
    for key in ("control_frequency_hz", "frequency_hz", "control_hz", "frequency"):
        try:
            hz = fl(cmeta[key])
            break
        except Exception:
            pass
except Exception as exc:
    err("controller_metadata", exc)
report["max_steps"] = max_steps
report["hz"] = hz

state0 = None
down_q = [0.0, 1.0, 0.0, 0.0]
try:
    state0 = get_robot_state()
    base = str(state0.base_frame)
    ee0 = state0.end_effector_poses["primary"]
    q0 = [fl(v) for v in ee0.quaternion_wxyz]
    report["robot_state0"] = {
        "base_frame": base,
        "embodiment": str(state0.embodiment),
        "timestamp_s": fl(state0.timestamp_s),
        "joints": [round(fl(v), 4) for v in state0.joint_positions["primary"]],
        "ee_pos": xyz(ee0.position),
        "ee_quat": q0,
        "gripper": fl(state0.gripper_positions["primary"]),
    }
    zax = qaxis(q0, 2)
    report["robot_state0"]["ee_z_axis"] = zax
    if zax[2] < -0.9:
        down_q = q0
    note("state0 " + str(report["robot_state0"]))
except Exception as exc:
    err("robot_state", exc)
report["down_q"] = down_q

try:
    obs0 = get_observation()
    for name in obs0.cameras:
        cams.append(str(name))
        cam = obs0.cameras[name]
        info = {"name": str(name)}
        try:
            info["rgb_shape"] = [int(v) for v in cam.rgb.shape]
            info["depth_shape"] = [int(v) for v in cam.depth_m.shape]
            info["intrinsics"] = [[round(fl(cam.intrinsics[r][c]), 3) for c in range(3)] for r in range(3)]
            info["pose_pos"] = xyz(cam.camera_pose.position)
            info["pose_quat"] = [fl(v) for v in cam.camera_pose.quaternion_wxyz]
            info["pose_frame"] = str(cam.camera_pose.frame)
        except Exception as exc2:
            info["error"] = str(exc2)
        report.setdefault("cameras", []).append(info)
    note("cameras " + str(cams))
except Exception as exc:
    err("observation", exc)

front = "front" if "front" in cams else (cams[0] if len(cams) > 0 else "front")
wrist = "wrist" if "wrist" in cams else front


def cloud_info(cand, cam):
    pc = mask_to_point_cloud(cand, cam, target_frame=base)
    pts = pc.points
    n = len(pts)
    stride = 1
    if n > 1500:
        stride = int(n / 1500) + 1
    xs = []
    ys = []
    zs = []
    i = 0
    while i < n:
        p = pts[i]
        xs.append(fl(p[0]))
        ys.append(fl(p[1]))
        zs.append(fl(p[2]))
        i += stride
    if len(xs) < 5:
        return {"n": n}
    return {
        "n": n,
        "center": [quantile(xs, 0.5), quantile(ys, 0.5), quantile(zs, 0.5)],
        "z10": quantile(zs, 0.1), "z90": quantile(zs, 0.9),
        "z02": quantile(zs, 0.02), "z98": quantile(zs, 0.98),
        "span_x": quantile(xs, 0.9) - quantile(xs, 0.1),
        "span_y": quantile(ys, 0.9) - quantile(ys, 0.1),
        "xs": xs, "ys": ys, "zs": zs,
    }


def seg_candidates(cam, prompt, limit, with_cloud):
    out = []
    probe = {"camera": cam, "prompt": prompt, "ok": False, "cands": []}
    try:
        found = segment_text(cam, prompt)
        probe["ok"] = bool(found.ok)
        if found.ok:
            k = 0
            for cand in found.segmentations:
                if k >= limit:
                    break
                k += 1
                item = {"prompt": prompt, "cam": cam, "cand": cand}
                try:
                    item["score"] = fl(cand.score)
                except Exception:
                    item["score"] = 0.0
                try:
                    item["box"] = [round(fl(v), 1) for v in cand.box_xyxy]
                except Exception:
                    item["box"] = None
                rec = {"score": round(item["score"], 3), "box": item["box"]}
                if with_cloud:
                    try:
                        ci = cloud_info(cand, cam)
                        item["cloud"] = ci
                        if "center" in ci:
                            rec["center"] = [round(v, 4) for v in ci["center"]]
                            rec["z90"] = round(ci["z90"], 4)
                            rec["z10"] = round(ci["z10"], 4)
                            rec["span"] = [round(ci["span_x"], 4), round(ci["span_y"], 4)]
                        rec["n"] = ci["n"]
                    except Exception as exc3:
                        rec["cloud_error"] = str(exc3)
                probe["cands"].append(rec)
                out.append(item)
    except Exception as exc:
        probe["error"] = str(exc)
    report.setdefault("probes", []).append(probe)
    return out


# Prompt probes (front: with clouds, wrist: scores/boxes only, cheap).
front_sets = {}
try:
    for prompt in ("cube", "block", "button", "red button", "target", "disc",
                   "red and white target", "circular target", "table"):
        front_sets[prompt] = seg_candidates(front, prompt, 6, True)
    for prompt in ("cube", "button", "target"):
        seg_candidates(wrist, prompt, 4, False)
except Exception as exc:
    err("probes", exc)

# Table height from the "table" mask; fallback below from the video.
table_z = None
try:
    tbl = front_sets.get("table", [])
    if len(tbl) > 0 and "cloud" in tbl[0] and "center" in tbl[0]["cloud"]:
        table_z = tbl[0]["cloud"]["center"][2]
except Exception as exc:
    err("table", exc)
report["table_z_seg"] = table_z

# ---------------------------------------------------------------- Phase B: conditioning video
series = []
events = []
video = []
try:
    video = robomme_plus.get_conditioning_video()
    nvid = len(video)
    report["video_len"] = nvid
    note("video frames: " + str(nvid))
    idx = 0
    for frame in video:
        try:
            rs = frame.robot_state
            p = rs.end_effector_poses["primary"].position
            series.append([fl(frame.timestamp_s), fl(p[0]), fl(p[1]), fl(p[2]),
                           fl(rs.gripper_positions["primary"])])
        except Exception as exc4:
            if idx < 3:
                err("video_frame_" + str(idx), exc4)
        idx += 1
    if len(series) > 0:
        report["video_t_range"] = [series[0][0], series[-1][0]]
    report["video_series"] = [[round(v, 4) for v in row] for row in series]
except Exception as exc:
    err("video", exc)

try:
    n = len(series)
    if n >= 5:
        zs = [row[3] for row in series]
        zmin = min(zs)
        report["video_zmin"] = zmin
        prom_req = 0.03
        gmax = max(row[4] for row in series)
        open_thr = min(0.85, gmax - 0.1)
        report["video_gmax"] = gmax
        report["video_open_thr"] = open_thr
        cands = []
        for i in range(n):
            zi = zs[i]
            if zi > zmin + 0.12:
                continue
            if i > 0 and zs[i - 1] < zi:
                continue
            if i < n - 1 and zs[i + 1] < zi:
                continue
            lmax = zi
            j = i - 1
            while j >= 0 and zs[j] >= zi:
                if zs[j] > lmax:
                    lmax = zs[j]
                j -= 1
            if j < 0:
                lmax = max(lmax, zs[0])
            rmax = zi
            j = i + 1
            while j < n and zs[j] >= zi:
                if zs[j] > rmax:
                    rmax = zs[j]
                j += 1
            prom = min(lmax, rmax) - zi
            # an event at the very start/end of the video counts with one-sided prominence
            if i == 0 or i == n - 1:
                prom = max(lmax, rmax) - zi
            if prom >= prom_req:
                cands.append(i)
        # merge minima not separated by a rise of prom_req
        merged = []
        for i in cands:
            if len(merged) > 0:
                last = merged[-1]
                peak = max(zs[last:i + 1])
                if peak < min(zs[last], zs[i]) + prom_req:
                    if zs[i] < zs[last]:
                        merged[-1] = i
                    continue
            merged.append(i)
        for i in merged:
            zi = zs[i]
            a = i
            while a > 0 and zs[a] < zi + prom_req:
                a -= 1
            b = i
            while b < n - 1 and zs[b] < zi + prom_req:
                b += 1
            g_pre = series[a][4]
            g_post = series[b][4]
            gin = [series[k][4] for k in range(a, b + 1)]
            kind = "contact"
            if g_pre > open_thr and 0.1 < g_post < open_thr:
                kind = "grasp"
            elif 0.1 < g_pre < open_thr and g_post > open_thr:
                kind = "release"
            events.append({"i": i, "t": series[i][0], "xy": [series[i][1], series[i][2]],
                           "z": zi, "g_pre": g_pre, "g_post": g_post,
                           "g_min": min(gin), "g_max": max(gin), "kind": kind,
                           "win": [a, b]})
    report["video_events"] = [{"i": e["i"], "t": round(e["t"], 3),
                               "xy": [round(e["xy"][0], 4), round(e["xy"][1], 4)],
                               "z": round(e["z"], 4), "g_pre": round(e["g_pre"], 3),
                               "g_post": round(e["g_post"], 3), "kind": e["kind"]}
                              for e in events]
    note("video events: " + str(report["video_events"]))
except Exception as exc:
    err("events", exc)

try:
    if len(series) > 0:
        tz_video = min(row[3] for row in series) - 0.02
        report["table_z_from_video"] = tz_video
        if table_z is None or abs(table_z - tz_video) > 0.04:
            table_z = tz_video
            report["table_z_source"] = "video_min_ee_minus_0.02"
except Exception as exc:
    err("table_fallback", exc)
if table_z is None:
    table_z = 0.0
    report["table_z_default"] = True
report["table_z"] = table_z

# ---------------------------------------------------------------- live object classification
button_xy = None
button_box = None
targets = []
cube_cands = []
try:
    btn = []
    for prompt in ("button", "red button"):
        for item in front_sets.get(prompt, []):
            ci = item.get("cloud", {})
            if "center" not in ci:
                continue
            h = ci["z90"] - table_z
            if h < 0.003 or h > 0.10 or ci["span_x"] > 0.15 or ci["span_y"] > 0.15:
                continue
            btn.append(item)
    cub = []
    for prompt in ("cube", "block"):
        for item in front_sets.get(prompt, []):
            ci = item.get("cloud", {})
            if "center" not in ci:
                continue
            h = ci["z90"] - table_z
            if h < 0.012 or h > 0.09 or ci["span_x"] > 0.10 or ci["span_y"] > 0.10:
                continue
            cub.append(item)
    tgt = []
    for prompt in ("target", "disc", "red and white target", "circular target"):
        for item in front_sets.get(prompt, []):
            ci = item.get("cloud", {})
            if "center" not in ci:
                continue
            h = ci["z90"] - table_z
            if h > 0.03 or ci["span_x"] > 0.25 or ci["span_y"] > 0.25:
                continue
            tgt.append(item)

    def same_obj(a, b):
        return dist2(a["cloud"]["center"], b["cloud"]["center"]) < 0.025

    # a box claimed by both "cube" and "button" goes to the class that scores it higher
    cube_clean = []
    for c in cub:
        beaten = False
        for b in btn:
            if same_obj(c, b) and b["score"] > c["score"]:
                beaten = True
        if not beaten:
            cube_clean.append(c)
    btn_clean = []
    for b in btn:
        beaten = False
        for c in cub:
            if same_obj(c, b) and c["score"] > b["score"]:
                beaten = True
        if not beaten:
            btn_clean.append(b)
    # dedupe targets by position, keep best score
    tsorted = sorted(tgt, key=neg_score)
    for t in tsorted:
        dup = False
        for u in targets:
            if dist2(t["cloud"]["center"], u["cloud"]["center"]) < 0.03:
                dup = True
        if not dup:
            targets.append(t)
    cube_cands = sorted(cube_clean, key=neg_score)
    btn_clean = sorted(btn_clean, key=neg_score)
    if len(btn_clean) > 0:
        button_xy = btn_clean[0]["cloud"]["center"][:2]
        button_box = btn_clean[0]["box"]
    report["live"] = {
        "button": [{"score": round(b["score"], 3), "center": [round(v, 4) for v in b["cloud"]["center"]],
                    "z90": round(b["cloud"]["z90"], 4), "box": b["box"], "prompt": b["prompt"]}
                   for b in btn_clean],
        "cubes": [{"score": round(c["score"], 3), "center": [round(v, 4) for v in c["cloud"]["center"]],
                   "z90": round(c["cloud"]["z90"], 4), "box": c["box"], "prompt": c["prompt"]}
                  for c in cube_cands],
        "targets": [{"score": round(t["score"], 3), "center": [round(v, 4) for v in t["cloud"]["center"]],
                     "z90": round(t["cloud"]["z90"], 4), "box": t["box"], "prompt": t["prompt"]}
                    for t in targets],
    }
    note("live " + str(report["live"]))
except Exception as exc:
    err("classify", exc)

# Button-region brightness in the conditioning video (fallback press timing, cheap summary).
try:
    video2 = video
    nv = len(video2)
    stride = 1
    if nv > 128:
        stride = int(nv / 128) + 1
    bseries = []
    gstats = []
    k = 0
    for frame in video2:
        if k % stride == 0 or k == nv - 1:
            rgb = frame.cameras[front].rgb
            hgt = int(rgb.shape[0])
            wid = int(rgb.shape[1])
            if button_box is not None:
                x1, y1, x2, y2 = button_box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                hw = max(1, int((x2 - x1) / 4))
                hh = max(1, int((y2 - y1) / 4))
                acc = [0, 0, 0]
                cnt = 0
                for yy in (cy - hh, cy, cy + hh):
                    for xx in (cx - hw, cx, cx + hw):
                        if 0 <= yy < hgt and 0 <= xx < wid:
                            px = rgb[yy][xx]
                            acc[0] += int(px[0])
                            acc[1] += int(px[1])
                            acc[2] += int(px[2])
                            cnt += 1
                if cnt > 0:
                    bseries.append([round(fl(frame.timestamp_s), 3), round(acc[0] / cnt, 1),
                                    round(acc[1] / cnt, 1), round(acc[2] / cnt, 1)])
            if k % (stride * 8) == 0:
                tot = 0
                cnt = 0
                for yy in range(8, hgt, 32):
                    for xx in range(8, wid, 32):
                        px = rgb[yy][xx]
                        tot += int(px[0]) + int(px[1]) + int(px[2])
                        cnt += 1
                gstats.append([round(fl(frame.timestamp_s), 3), round(tot / (3.0 * cnt), 1)])
        k += 1
    report["video_button_rgb"] = bseries
    report["video_front_mean"] = gstats
except Exception as exc:
    err("video_images", exc)


# ---------------------------------------------------------------- projection / colour helpers
def project(pt, cam, conv):
    """Project a robot_base point into a camera image under one of three axis conventions."""
    cp = cam.camera_pose
    t = xyz(cp.position)
    q = [fl(v) for v in cp.quaternion_wxyz]
    d = [pt[0] - t[0], pt[1] - t[1], pt[2] - t[2]]
    ax = [qaxis(q, 0), qaxis(q, 1), qaxis(q, 2)]
    c = [ax[k][0] * d[0] + ax[k][1] * d[1] + ax[k][2] * d[2] for k in range(3)]
    k00 = fl(cam.intrinsics[0][0])
    k11 = fl(cam.intrinsics[1][1])
    k02 = fl(cam.intrinsics[0][2])
    k12 = fl(cam.intrinsics[1][2])
    if conv == "cv":
        if c[2] <= 1e-6:
            return None
        return [k00 * c[0] / c[2] + k02, k11 * c[1] / c[2] + k12]
    if conv == "gl":
        if c[2] >= -1e-6:
            return None
        return [k00 * c[0] / (-c[2]) + k02, k12 - k11 * c[1] / (-c[2])]
    if c[0] <= 1e-6:
        return None
    return [k02 - k00 * c[1] / c[0], k12 - k11 * c[2] / c[0]]


def color_at(rgb, u, v):
    hgt = int(rgb.shape[0])
    wid = int(rgb.shape[1])
    acc = [0, 0, 0]
    cnt = 0
    for dv in (-1, 0, 1):
        for du in (-1, 0, 1):
            x = int(round(u)) + du
            y = int(round(v)) + dv
            if 0 <= x < wid and 0 <= y < hgt:
                px = rgb[y][x]
                acc[0] += int(px[0])
                acc[1] += int(px[1])
                acc[2] += int(px[2])
                cnt += 1
    if cnt == 0:
        return None
    return [acc[0] / cnt, acc[1] / cnt, acc[2] / cnt]


calib = {"conv": None}
try:
    live_cam = get_observation().cameras[front]
    best_conv = None
    best_err = 1e9
    per = {}
    for conv in ("cv", "gl", "ros"):
        tot = 0.0
        cnt = 0
        for group in (front_sets.get("cube", []), front_sets.get("button", []), front_sets.get("target", [])):
            for item in group[:2]:
                ci = item.get("cloud", {})
                if "center" not in ci or item.get("box") is None:
                    continue
                uv = project(ci["center"], live_cam, conv)
                if uv is None:
                    tot += 1000.0
                else:
                    bx = item["box"]
                    tot += math.sqrt((uv[0] - (bx[0] + bx[2]) / 2.0) ** 2 + (uv[1] - (bx[1] + bx[3]) / 2.0) ** 2)
                cnt += 1
        if cnt > 0:
            per[conv] = round(tot / cnt, 2)
            if tot / cnt < best_err:
                best_err = tot / cnt
                best_conv = conv
    if best_conv is not None and best_err < 12.0:
        calib["conv"] = best_conv
    report["camera_convention"] = {"chosen": calib["conv"], "mean_px_err": per}
except Exception as exc:
    err("calibration", exc)


def color_match_target(event):
    """Scene changed: colour of the table under the release point in the first video frame,
    matched against the live target candidates' colours."""
    if calib["conv"] is None or len(targets) == 0:
        return None
    first = video[0]
    vcam = first.cameras[front]
    uv = project([event["xy"][0], event["xy"][1], table_z + 0.002], vcam, calib["conv"])
    if uv is None:
        return None
    ref = color_at(vcam.rgb, uv[0], uv[1])
    live_cam2 = get_observation().cameras[front]
    best = None
    best_d = 1e9
    rows = []
    for t in targets:
        bx = t["box"]
        if bx is None:
            continue
        col = color_at(live_cam2.rgb, (bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0)
        if col is None or ref is None:
            continue
        d = abs(col[0] - ref[0]) + abs(col[1] - ref[1]) + abs(col[2] - ref[2])
        rows.append({"center": [round(v, 4) for v in t["cloud"]["center"]], "rgb": col, "d": d})
        if d < best_d:
            best_d = d
            best = t
    if best is None:
        return None
    second = 1e9
    skipped = False
    for row in sorted([rw["d"] for rw in rows]):
        if (not skipped) and row == best_d:
            skipped = True
            continue
        second = min(second, row)
    # identical-looking targets cannot be told apart by colour: require a unique match
    if second - best_d < 60:
        return None
    return {"xy": best["cloud"]["center"][:2], "best_d": best_d,
            "info": {"ref_rgb": ref, "uv": uv, "cands": rows}}

# ---------------------------------------------------------------- Phase C: goal parsing + selection
choice = {}
try:
    low = lang.lower().replace(",", " ").replace(".", " ")
    words = low.split()
    relation = None
    rel_pos = -1
    for w_i in range(len(words)):
        w = words[w_i]
        if w in ("before", "prior"):
            relation = "before"
            rel_pos = w_i
            break
        if w in ("after", "following"):
            relation = "after"
            rel_pos = w_i
            break
        if w in ("when", "while", "during"):
            relation = "during"
            rel_pos = w_i
            break
    ordmap = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
              "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "last": -1, "final": -1}
    ordinal = None
    ord_pos = -1
    for w_i in range(len(words)):
        if words[w_i] in ordmap:
            ordinal = ordmap[words[w_i]]
            ord_pos = w_i
            break
    press_pos = -1
    for w_i in range(len(words)):
        if words[w_i].startswith("press") or words[w_i].startswith("button"):
            press_pos = w_i
            break
    ord_on_press = False
    if ordinal is not None and press_pos >= 0 and rel_pos >= 0 and ord_pos > rel_pos:
        ord_on_press = True  # e.g. "after the button was pressed the second time"
    choice.update({"relation": relation, "ordinal": ordinal, "ord_on_press": ord_on_press})

    releases = [e for e in events if e["kind"] == "release"]
    contacts = [e for e in events if e["kind"] == "contact"]
    if len(releases) == 0:
        # fall back: non-grasp low points that are not at the button
        for e in events:
            if e["kind"] != "grasp" and (button_xy is None or dist2(e["xy"], button_xy) > 0.05):
                releases.append(e)
        choice["release_fallback"] = True
    presses = []
    if button_xy is not None:
        presses = [e for e in contacts if dist2(e["xy"], button_xy) < 0.06]
        if len(presses) == 0:
            presses = [e for e in events if dist2(e["xy"], button_xy) < 0.05 and e["kind"] != "release"]
    if len(presses) == 0:
        presses = contacts
        choice["press_source"] = "contact_events"
    else:
        choice["press_source"] = "contact_near_button"
    press_t = None
    if len(presses) > 0:
        pk = 0
        if ord_on_press and ordinal is not None:
            pk = ordinal - 1 if ordinal > 0 else len(presses) - 1
            pk = max(0, min(pk, len(presses) - 1))
        press_t = presses[pk]["t"]
    elif len(report.get("video_button_rgb", [])) > 2:
        bs = report["video_button_rgb"]
        ref = bs[0]
        best = -1.0
        for row in bs:
            d = abs(row[1] - ref[1]) + abs(row[2] - ref[2]) + abs(row[3] - ref[3])
            if d > best:
                best = d
                press_t = row[0]
        choice["press_source"] = "button_pixels"
    choice["press_t"] = press_t
    choice["n_releases"] = len(releases)
    chosen = None
    if len(releases) > 0:
        if press_t is None:
            pool = releases
            k = -1
        elif relation == "after":
            pool = [e for e in releases if e["t"] > press_t]
            k = 0
        else:
            pool = [e for e in releases if e["t"] < press_t]
            k = -1
        if (not ord_on_press) and ordinal is not None and len(pool) > 0:
            if ordinal > 0:
                k = min(ordinal - 1, len(pool) - 1)
            else:
                k = -1
        if len(pool) == 0:
            pool = releases
            choice["pool_fallback"] = True
        chosen = pool[k]
    if chosen is not None:
        choice["chosen_event"] = {"t": chosen["t"], "xy": chosen["xy"], "z": chosen["z"]}
        place_xy = [chosen["xy"][0], chosen["xy"][1]]
        place_src = "video_release"
        # Is the live scene the one in the video?  Count release points with a live target nearby.
        consistent = 0
        for e in releases:
            dmin = 1e9
            for t in targets:
                dmin = min(dmin, dist2(t["cloud"]["center"], e["xy"]))
            e["live_target_dist"] = dmin
            if dmin < 0.04:
                consistent += 1
        choice["release_live_dists"] = [round(e["live_target_dist"], 4) for e in releases]
        scene_same = len(targets) == 0 or 2 * consistent >= len(releases)
        choice["scene_same"] = scene_same
        best_t = None
        best_d = 1e9
        for t in targets:
            d = dist2(t["cloud"]["center"], place_xy)
            if d < best_d:
                best_d = d
                best_t = t
        if scene_same:
            if best_t is not None and best_d < 0.03:
                lc = best_t["cloud"]["center"]
                place_xy = [0.5 * (place_xy[0] + lc[0]), 0.5 * (place_xy[1] + lc[1])]
                place_src = "video_release_avg_live_target"
                choice["snap_dist"] = best_d
            # appearance check: same colour at that spot in the first video frame and now?
            try:
                cube_near = False
                for c in cube_cands:
                    if dist2(c["cloud"]["center"], place_xy) < 0.04:
                        cube_near = True
                if calib["conv"] is not None and not cube_near and len(targets) > 1:
                    pt = [place_xy[0], place_xy[1], table_z + 0.002]
                    vcam0 = video[0].cameras[front]
                    lcam0 = get_observation().cameras[front]
                    uv0 = project(pt, vcam0, calib["conv"])
                    uv1 = project(pt, lcam0, calib["conv"])
                    if uv0 is not None and uv1 is not None:
                        c0 = color_at(vcam0.rgb, uv0[0], uv0[1])
                        c1 = color_at(lcam0.rgb, uv1[0], uv1[1])
                        diff = abs(c0[0] - c1[0]) + abs(c0[1] - c1[1]) + abs(c0[2] - c1[2])
                        choice["appearance_check"] = {"video_rgb": c0, "live_rgb": c1, "l1": round(diff, 1)}
                        if diff > 90:
                            cm = color_match_target(chosen)
                            if cm is not None and cm["best_d"] < 45:
                                place_xy = cm["xy"]
                                place_src = "color_match_permuted"
                                choice["color_match"] = cm["info"]
            except Exception as exc6:
                err("appearance_check", exc6)
        else:
            try:
                cm = color_match_target(chosen)
                if cm is not None:
                    place_xy = cm["xy"]
                    place_src = "color_match"
                    choice["color_match"] = cm["info"]
            except Exception as exc5:
                err("color_match", exc5)
        choice["place_xy"] = place_xy
        choice["place_src"] = place_src
        choice["place_release_z"] = chosen["z"]
    note("choice " + str(choice))
except Exception as exc:
    err("selection", exc)
report["choice"] = choice

# ---------------------------------------------------------------- Phase C2 (v01): target identity tracking
# Episode-0 evidence (v00 run): the four identical targets swap places pairwise many times during the
# conditioning video (each swap: both slots empty for ~1 s while the two targets orbit their midpoint).
# The target the cube was released on is an OBJECT; where it is now must be tracked through every
# later swap.  Slots = live target centres (the scene at program start equals the end of the video).
# Per video frame we count target-coloured pixels inside each slot window and on the swap orbits
# outside all slots.  A swap = two slots that were occupied go empty together while target-coloured
# pixels appear on the orbits (arm occlusion empties slots without that off-slot mass).
def rgb_rows(rgb):
    try:
        return rgb.tolist()
    except Exception:
        return rgb


def px_at(rows, u, v):
    p = rows[v][u]
    return [int(p[0]), int(p[1]), int(p[2])]


def disc_pixels(cam, cx, cy, r_in, r_out, plane_z, step, hgt, wid, conv):
    """Unique image pixels whose table-plane point lies in r_in <= d < r_out of (cx, cy)."""
    out = set()
    n = int(r_out / step) + 1
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            dx = i * step
            dy = j * step
            d = math.sqrt(dx * dx + dy * dy)
            if d < r_in or d >= r_out:
                continue
            uv = project([cx + dx, cy + dy, plane_z], cam, conv)
            if uv is None:
                continue
            u = int(round(uv[0]))
            v = int(round(uv[1]))
            if 0 <= u < wid and 0 <= v < hgt:
                out.add((u, v))
    return out


def neg_excess_key(ev):
    return -ev["excess"] * (1 + ev["f"][1] - ev["f"][0])


def start_key(ev):
    return ev["f"][0]


track = {"ok": False}
try:
    conv = calib["conv"]
    slot_list = []
    for t in targets:
        c0 = t["cloud"]["center"]
        # a target with a cube on it stays a slot (its window reads empty only while covered;
        # a swap also needs both slots full just before/after and target mass on the orbits)
        under_cube = False
        for c in cube_cands:
            if dist2(c["cloud"]["center"], c0) < 0.03:
                under_cube = True
        slot_list.append({"xy": [c0[0], c0[1]], "z": c0[2], "under_cube": under_cube,
                          "span": max(t["cloud"]["span_x"], t["cloud"]["span_y"])})
    ns = len(slot_list)
    track["n_slots"] = ns
    if conv is not None and ns >= 2 and len(video) >= 10:
        vcam = video[0].cameras[front]
        try:
            lcp = get_observation().cameras[front].camera_pose
            track["cam_pose_delta"] = round(dist2(xyz(lcp.position), xyz(vcam.camera_pose.position)), 4)
        except Exception:
            pass
        hgt = int(vcam.rgb.shape[0])
        wid = int(vcam.rgb.shape[1])
        min_sep = 1e9
        for a in range(ns):
            for b in range(a + 1, ns):
                min_sep = min(min_sep, dist2(slot_list[a]["xy"], slot_list[b]["xy"]))
        spans = sorted([s["span"] for s in slot_list])
        tspan = spans[int(len(spans) / 2)]
        r_win = max(0.012, min(0.03, 0.3 * tspan, 0.2 * min_sep))
        r_excl = max(r_win + 0.005, min(0.6 * tspan, 0.45 * min_sep))
        plane_z = quantile([s["z"] for s in slot_list], 0.5)
        track["r_win"] = round(r_win, 4)
        track["r_excl"] = round(r_excl, 4)
        win = []
        excl = set()
        for s in slot_list:
            win.append(sorted(disc_pixels(vcam, s["xy"][0], s["xy"][1], 0.0, r_win, plane_z, 0.002, hgt, wid, conv)))
            excl = excl | disc_pixels(vcam, s["xy"][0], s["xy"][1], 0.0, r_excl, plane_z, 0.003, hgt, wid, conv)
        # swap orbits: circles about every slot pair's midpoint through both slots
        orbit = set()
        for a in range(ns):
            for b in range(a + 1, ns):
                pa = slot_list[a]["xy"]
                pb = slot_list[b]["xy"]
                mx = 0.5 * (pa[0] + pb[0])
                my = 0.5 * (pa[1] + pb[1])
                rad = 0.5 * dist2(pa, pb)
                nang = int(2.0 * math.pi * rad / 0.004) + 8
                for k in range(nang):
                    ang = 2.0 * math.pi * k / nang
                    for dr in (-0.03, -0.015, 0.0, 0.015, 0.03):
                        uv = project([mx + (rad + dr) * math.cos(ang), my + (rad + dr) * math.sin(ang), plane_z],
                                     vcam, conv)
                        if uv is None:
                            continue
                        u = int(round(uv[0]))
                        v = int(round(uv[1]))
                        if 0 <= u < wid and 0 <= v < hgt and (u, v) not in excl:
                            orbit.add((u, v))
        orbit = sorted(orbit)
        track["n_win_px"] = [len(w) for w in win]
        track["n_orbit_px"] = len(orbit)
        # colour model from the live image: table colour around the slots, target chroma inside them
        lrows = rgb_rows(get_observation().cameras[front].rgb)
        tab = [[], [], []]
        for s in slot_list:
            for (u, v) in disc_pixels(vcam, s["xy"][0], s["xy"][1], r_excl + 0.01, r_excl + 0.03, plane_z,
                                      0.004, hgt, wid, conv):
                p = px_at(lrows, u, v)
                tab[0].append(p[0])
                tab[1].append(p[1])
                tab[2].append(p[2])
        table_rgb = [quantile(tab[0], 0.5), quantile(tab[1], 0.5), quantile(tab[2], 0.5)]
        c1s = []
        c2s = []
        for w in win:
            for (u, v) in w:
                p = px_at(lrows, u, v)
                if abs(p[0] - table_rgb[0]) + abs(p[1] - table_rgb[1]) + abs(p[2] - table_rgb[2]) > 90 \
                        and max(p) - min(p) > 20:
                    c1s.append(p[0] - p[1])
                    c2s.append(p[2] - p[1])
        proto = None
        if len(c1s) >= 10:
            proto = [quantile(c1s, 0.5), quantile(c2s, 0.5)]
        track["table_rgb"] = [round(v, 1) for v in table_rgb]
        track["target_chroma"] = None if proto is None else [round(v, 1) for v in proto]
        track["n_chroma_px"] = len(c1s)

        def is_tgt(p):
            r = int(p[0])
            g = int(p[1])
            b = int(p[2])
            if abs(r - table_rgb[0]) + abs(g - table_rgb[1]) + abs(b - table_rgb[2]) <= 90:
                return False
            if max(r, g, b) - min(r, g, b) <= 20:
                return False
            return abs(r - g - proto[0]) + abs(b - g - proto[1]) < 35

        if proto is not None:
            counts = []
            offc = []
            ts = []
            for frame in video:
                try:
                    rows = rgb_rows(frame.cameras[front].rgb)
                    ts.append(fl(frame.timestamp_s))
                    row = []
                    for w in win:
                        n_in = 0
                        for (u, v) in w:
                            if is_tgt(rows[v][u]):
                                n_in += 1
                        row.append(n_in)
                    counts.append(row)
                    n_off = 0
                    for (u, v) in orbit:
                        if is_tgt(rows[v][u]):
                            n_off += 1
                    offc.append(n_off)
                except Exception as exc7:
                    if len(counts) < 3:
                        err("track_frame", exc7)
            nf = len(counts)
            track["n_frames"] = nf
            refs = [quantile([counts[f][k] for f in range(nf)], 0.8) for k in range(ns)]
            track["ref_counts"] = refs
            usable = nf >= 10 and min(refs) >= 6
            track["usable"] = usable
            if usable:
                vac = [[counts[f][k] < 0.3 * refs[k] for k in range(ns)] for f in range(nf)]
                full = [[counts[f][k] >= 0.5 * refs[k] for k in range(ns)] for f in range(nf)]
                allfull = [offc[f] for f in range(nf) if all(full[f])]
                off_base = quantile(allfull if len(allfull) > 0 else offc, 0.5)
                track["off_base"] = off_base
                cands_ev = []
                rejected = []
                for a in range(ns):
                    for b in range(a + 1, ns):
                        f = 0
                        while f < nf:
                            if not (vac[f][a] and vac[f][b]):
                                f += 1
                                continue
                            s0 = f
                            e0 = f
                            g = f + 1
                            while g < nf and g <= e0 + 3:
                                if vac[g][a] and vac[g][b]:
                                    e0 = g
                                g += 1
                            f = e0 + 1
                            pre_a = any(full[h][a] for h in range(max(0, s0 - 8), s0))
                            pre_b = any(full[h][b] for h in range(max(0, s0 - 8), s0))
                            post_a = any(full[h][a] for h in range(e0 + 1, min(nf, e0 + 9)))
                            post_b = any(full[h][b] for h in range(e0 + 1, min(nf, e0 + 9)))
                            seg = offc[s0:e0 + 1]
                            excess = sum(seg) / float(len(seg)) - off_base
                            need = 0.6 * (refs[a] + refs[b])
                            ev = {"pair": [a, b], "f": [s0, e0], "t": [round(ts[s0], 2), round(ts[e0], 2)],
                                  "excess": round(excess, 1), "need": round(need, 1),
                                  "pre": [pre_a, pre_b], "post": [post_a, post_b]}
                            okv = pre_a and pre_b and post_a and post_b and excess >= need and e0 > s0
                            if okv:
                                cands_ev.append(ev)
                            else:
                                rejected.append(ev)
                accepted = []
                for ev in sorted(cands_ev, key=neg_excess_key):
                    clash = False
                    for acc in accepted:
                        if ev["f"][0] <= acc["f"][1] + 2 and acc["f"][0] <= ev["f"][1] + 2:
                            clash = True
                    if not clash:
                        accepted.append(ev)
                    else:
                        rejected.append(ev)
                accepted = sorted(accepted, key=start_key)
                track["swaps"] = accepted
                track["rejected"] = rejected[:40]
                ce = choice.get("chosen_event", None)
                if ce is not None:
                    rel_slot = None
                    best_d = 1e9
                    for k in range(ns):
                        d = dist2(slot_list[k]["xy"], ce["xy"])
                        if d < best_d:
                            best_d = d
                            rel_slot = k
                    track["release_slot"] = rel_slot
                    track["release_slot_dist"] = round(best_d, 4)
                    if rel_slot is not None and best_d < 0.05:
                        where = rel_slot
                        path = [rel_slot]
                        straddle = False
                        for ev in accepted:
                            if ts[ev["f"][0]] <= ce["t"] <= ts[ev["f"][1]]:
                                straddle = True
                            if ts[ev["f"][0]] > ce["t"]:
                                a, b = ev["pair"]
                                if where == a:
                                    where = b
                                elif where == b:
                                    where = a
                                path.append(where)
                        track["path"] = path
                        track["straddle"] = straddle
                        track["final_slot"] = where
                        track["final_xy"] = [round(v, 4) for v in slot_list[where]["xy"]]
                        track["ok"] = True
                        if not straddle:
                            choice["place_xy"] = [slot_list[where]["xy"][0], slot_list[where]["xy"][1]]
                            choice["place_src"] = "tracked_target_identity_through_" + str(len(path) - 1) + "_swaps"
        track["slots"] = [[round(s["xy"][0], 4), round(s["xy"][1], 4)] for s in slot_list]
        track["slots_under_cube"] = [s["under_cube"] for s in slot_list]
    note("track " + str({k: track[k] for k in track if k not in ("rejected",)}))
except Exception as exc:
    err("track", exc)
report["track"] = track
report["choice"] = choice

# Where is the cube now?  Last video event tells; live detection confirms.
cube = None
try:
    pred = None
    held_at_start = False
    if len(events) > 0:
        last = events[-1]
        if last["kind"] in ("release", "contact", "grasp"):
            for e in reversed(events):
                if e["kind"] == "release":
                    pred = e["xy"]
                    break
    g0 = fl(get_robot_state().gripper_positions["primary"])
    if 0.1 < g0 < 0.85:
        held_at_start = True
    report["cube_pred_xy"] = pred
    report["held_at_start"] = held_at_start
    best = None
    best_d = 1e9
    for c in cube_cands:
        d = 0.0 if pred is None else dist2(c["cloud"]["center"], pred)
        if pred is None:
            d = -c["score"]
        if d < best_d:
            best_d = d
            best = c
    # v01: the language names the cube's colour ("red cube"); check the candidate's live colour.
    try:
        cname = None
        for cw in ("red", "green", "blue", "yellow"):
            if (cw + " cube") in lang.lower() or (cw + " block") in lang.lower():
                cname = cw
        if cname is not None and len(cube_cands) > 0:
            lrows2 = rgb_rows(get_observation().cameras[front].rgb)
            scored = []
            for c in cube_cands:
                bx = c["box"]
                if bx is None:
                    continue
                cu = int((bx[0] + bx[2]) / 2.0)
                cvv = int((bx[1] + bx[3]) / 2.0)
                acc = [0, 0, 0]
                n_px = 0
                for dv in (-2, 0, 2):
                    for du in (-2, 0, 2):
                        try:
                            p = lrows2[cvv + dv][cu + du]
                            acc[0] += int(p[0])
                            acc[1] += int(p[1])
                            acc[2] += int(p[2])
                            n_px += 1
                        except Exception:
                            pass
                if n_px == 0:
                    continue
                r = acc[0] / n_px
                g = acc[1] / n_px
                b = acc[2] / n_px
                sc = {"red": r - max(g, b), "green": g - max(r, b), "blue": b - max(r, g),
                      "yellow": min(r, g) - b}[cname]
                scored.append([sc, c, [round(r), round(g), round(b)]])
            if len(scored) > 0:
                top_sc = None
                for row in scored:
                    if top_sc is None or row[0] > top_sc[0]:
                        top_sc = row
                best_sc = None
                for row in scored:
                    if best is not None and dist2(row[1]["cloud"]["center"], best["cloud"]["center"]) < 1e-6:
                        best_sc = row[0]
                report["cube_colour_check"] = {"colour": cname, "video_pick_score": best_sc,
                                               "best_score": round(top_sc[0], 1), "best_rgb": top_sc[2],
                                               "scores": [[round(rw[0], 1), rw[2]] for rw in scored]}
                if top_sc[0] > 40 and (best is None or best_sc is None or top_sc[0] - best_sc > 30):
                    report["cube_colour_check"]["override"] = True
                    best = top_sc[1]
    except Exception as exc:
        err("cube_colour", exc)
    if best is not None:
        ci = best["cloud"]
        top = ci["z90"]
        size = max(0.02, min(0.07, top - table_z))
        # top-face points for the xy centre (less biased toward the camera-facing side)
        tx = []
        ty = []
        for m in range(len(ci["zs"])):
            if ci["zs"][m] > top - 0.008:
                tx.append(ci["xs"][m])
                ty.append(ci["ys"][m])
        cxy = ci["center"][:2]
        if len(tx) >= 5:
            cxy = [quantile(tx, 0.5), quantile(ty, 0.5)]
        cube = {"xy": cxy, "top": top, "size": size, "score": best["score"], "src": "front"}
    elif pred is not None:
        gz = None
        for e in events:
            if e["kind"] == "grasp":
                gz = e["z"]
        cube = {"xy": pred, "top": (gz if gz is not None else table_z + 0.02) + 0.02,
                "size": 0.04, "src": "video_pred"}
    report["cube"] = cube
    note("cube " + str(cube))
except Exception as exc:
    err("cube", exc)

# Last resort (no usable video events): a live target that is not under the cube, labelled a guess.
try:
    if choice.get("place_xy", None) is None and len(targets) > 0:
        for t in targets:
            if cube is None or dist2(t["cloud"]["center"], cube["xy"]) > 0.05:
                choice["place_xy"] = t["cloud"]["center"][:2]
                choice["place_src"] = "GUESS_live_target_not_under_cube"
                break
        report["choice"] = choice
except Exception as exc:
    err("guess", exc)

# ---------------------------------------------------------------- Phase D: act
def ticks_used():
    try:
        return int(round(fl(get_robot_state().timestamp_s) * hz))
    except Exception:
        return -1


def go(name, pos, quat, tol, steps):
    if ended["value"]:
        return None
    try:
        target = Pose([fl(pos[0]), fl(pos[1]), fl(pos[2])], quat, base)
        res = move_to_pose(target, tolerance=tol, max_steps=steps)
        entry = summarize_exec(name, res)
        if (not entry.get("ok", False)) and entry.get("steps", -1) == 0 and not ended["value"]:
            res = move_to_pose(target, tolerance=tol, max_steps=steps * 2)
            entry = summarize_exec(name + "_retry", res)
        return entry
    except Exception as exc:
        err("move_" + name, exc)
        return None


def grip(name, close):
    if ended["value"]:
        return None
    try:
        if close:
            res = close_gripper()
        else:
            res = open_gripper()
        return summarize_exec(name, res)
    except Exception as exc:
        err("grip_" + name, exc)
        return None


def gripper_now():
    try:
        return fl(get_robot_state().gripper_positions["primary"])
    except Exception:
        return -1.0


def ee_now():
    try:
        return xyz(get_robot_state().end_effector_poses["primary"].position)
    except Exception:
        return None


def yaw_quat(yaw):
    return qmul([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], down_q)


def wrist_refine(guess_xy, guess_top):
    """Segment the cube in the wrist camera from above; return (xy, top, yaw_delta) or None."""
    best = None
    best_d = 1e9
    for prompt in ("cube", "block"):
        items = seg_candidates(wrist, prompt, 5, True)
        for it in items:
            ci = it.get("cloud", {})
            if "center" not in ci:
                continue
            d = dist2(ci["center"], guess_xy)
            if d > 0.04 or abs(ci["z98"] - guess_top) > 0.02:
                continue
            if d < best_d:
                best_d = d
                best = ci
        if best is not None:
            break
    if best is None:
        return None
    top = best["z98"]
    px = []
    py = []
    for m in range(len(best["zs"])):
        if best["zs"][m] > top - 0.008:
            px.append(best["xs"][m])
            py.append(best["ys"][m])
    if len(px) < 8:
        return None
    mx = quantile(px, 0.5)
    my = quantile(py, 0.5)
    best_ang = 0.0
    best_area = 1e9
    ang = 0
    while ang < 90:
        a = ang * math.pi / 180.0
        ca = math.cos(a)
        sa = math.sin(a)
        us = [(px[m] - mx) * ca + (py[m] - my) * sa for m in range(len(px))]
        vs = [-(px[m] - mx) * sa + (py[m] - my) * ca for m in range(len(px))]
        area = (quantile(us, 0.95) - quantile(us, 0.05)) * (quantile(vs, 0.95) - quantile(vs, 0.05))
        if area < best_area:
            best_area = area
            best_ang = a
        ang += 3
    yax = qaxis(down_q, 1)
    a0 = math.atan2(yax[1], yax[0])
    delta = best_ang - a0
    while delta > math.pi / 4:
        delta -= math.pi / 2
    while delta < -math.pi / 4:
        delta += math.pi / 2
    us = [(px[m] - mx) * math.cos(best_ang) + (py[m] - my) * math.sin(best_ang) for m in range(len(px))]
    vs = [-(px[m] - mx) * math.sin(best_ang) + (py[m] - my) * math.cos(best_ang) for m in range(len(px))]
    cu = (quantile(us, 0.95) + quantile(us, 0.05)) / 2.0
    cv = (quantile(vs, 0.95) + quantile(vs, 0.05)) / 2.0
    cx = mx + cu * math.cos(best_ang) - cv * math.sin(best_ang)
    cy = my + cu * math.sin(best_ang) + cv * math.cos(best_ang)
    return {"xy": [cx, cy], "top": top, "yaw_delta": delta, "n_top": len(px),
            "span": [quantile(us, 0.95) - quantile(us, 0.05), quantile(vs, 0.95) - quantile(vs, 0.05)]}


try:
    report["ticks_before_motion"] = ticks_used()
    held = bool(report.get("held_at_start", False))
    place_xy = choice.get("place_xy", None)
    if place_xy is None:
        note("no placement target derived; stopping without motion")
    else:
        hover_z = table_z + 0.16
        # 0. if the hand starts low, rise straight up first
        e0 = ee_now()
        if e0 is not None and e0[2] < table_z + 0.12:
            go("rise", [e0[0], e0[1], hover_z], down_q, 0.015, 80)
        grasp_q = down_q
        attempts = 0
        while (not held) and cube is not None and attempts < 2 and not ended["value"]:
            attempts += 1
            cxy = cube["xy"]
            grasp_z = max(table_z + 0.012, cube["top"] - 0.5 * cube["size"])
            pre_z = cube["top"] + 0.12
            go("pregrasp_" + str(attempts), [cxy[0], cxy[1], pre_z], down_q, 0.012, 150)
            grip("open_" + str(attempts), False)
            ref = None
            try:
                ref = wrist_refine(cxy, cube["top"])
            except Exception as exc:
                err("wrist_refine", exc)
            report["wrist_refine_" + str(attempts)] = None if ref is None else {
                "xy": [round(v, 4) for v in ref["xy"]], "top": round(ref["top"], 4),
                "yaw_delta_deg": round(ref["yaw_delta"] * 180.0 / math.pi, 1),
                "n_top": ref["n_top"], "span": [round(v, 4) for v in ref["span"]]}
            if ref is not None:
                cxy = ref["xy"]
                grasp_q = yaw_quat(ref["yaw_delta"])
                go("pregrasp_aligned_" + str(attempts), [cxy[0], cxy[1], pre_z], grasp_q, 0.012, 100)
            go("descend_" + str(attempts), [cxy[0], cxy[1], grasp_z + 0.04], grasp_q, 0.01, 80)
            go("grasp_" + str(attempts), [cxy[0], cxy[1], grasp_z], grasp_q, 0.008, 60)
            grip("close_" + str(attempts), True)
            go("lift_" + str(attempts), [cxy[0], cxy[1], hover_z], grasp_q, 0.015, 100)
            g = gripper_now()
            report["grip_after_lift_" + str(attempts)] = g
            if 0.08 < g < 0.9:
                held = True
            else:
                note("grasp looks empty (g=" + str(g) + "); re-localizing")
                grip("reopen_" + str(attempts), False)
                # re-localize from the front camera with the arm lifted
                try:
                    best = None
                    best_d = 1e9
                    for it in seg_candidates(front, "cube", 6, True):
                        ci = it.get("cloud", {})
                        if "center" not in ci:
                            continue
                        h = ci["z90"] - table_z
                        if h < 0.012 or h > 0.09:
                            continue
                        d = dist2(ci["center"], cxy)
                        if d < best_d:
                            best_d = d
                            best = ci
                    if best is not None and best_d < 0.10:
                        cube = {"xy": best["center"][:2], "top": best["z90"],
                                "size": cube["size"], "src": "front_retry"}
                except Exception as exc:
                    err("relocalize", exc)
        report["held"] = held
        if held and not ended["value"]:
            rz = choice.get("place_release_z", None)
            size = 0.04 if cube is None else cube["size"]
            place_z = table_z + 0.5 * size + 0.012
            if rz is not None and table_z + 0.01 < rz < table_z + 0.08:
                place_z = max(place_z, rz)
            report["place_z"] = place_z
            go("transit", [place_xy[0], place_xy[1], hover_z], grasp_q, 0.015, 150)
            go("place_descend", [place_xy[0], place_xy[1], place_z + 0.04], grasp_q, 0.01, 80)
            go("place", [place_xy[0], place_xy[1], place_z], grasp_q, 0.008, 60)
            grip("release", False)
            go("retreat", [place_xy[0], place_xy[1], place_z + 0.12], grasp_q, 0.015, 80)
            # a few hold ticks so the cube settles before the episode is judged
            try:
                st = None
                if not ended["value"]:
                    st = get_robot_state()
                for tick in range(10 if st is not None else 0):
                    if ended["value"]:
                        break
                    r = step(RobotAction(arms={"primary": ArmCommand(
                        mode="joint_position", target=st.joint_positions["primary"],
                        gripper_position=1.0, embodiment=st.embodiment)}))
                    if is_terminal(r):
                        ended["value"] = True
                        note("episode terminal during settle at tick " + str(tick))
            except Exception as exc:
                err("settle", exc)
    report["ticks_after"] = ticks_used()
except Exception as exc:
    err("act", exc)

# final observation of the cube (record only)
try:
    if not ended["value"]:
        fin = []
        for it in seg_candidates(front, "cube", 4, True):
            ci = it.get("cloud", {})
            if "center" in ci:
                fin.append({"score": round(it["score"], 3), "center": [round(v, 4) for v in ci["center"]]})
        report["final_cubes"] = fin
except Exception as exc:
    err("final", exc)

report["ended"] = ended["value"]
note("done; errors=" + str(len(report["errors"])))
result = report
