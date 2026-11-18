# v01_goal_colour_cube_select — robomme_plus task_12 VideoPlaceOrder (AfI, single start, episode 0)
# UNTESTED trace-guided repair of v00 (v00: 0/1 on episode 0). Changes vs v00, nothing else:
#  (a) video grasp/release events are kept only if made near the table (EE z < video min z + 0.06)
#      and, for a grasp, only if the gripper actually holds something (not fully closed); a
#      release counts only after a kept grasp. v00 took an empty in-air closure/re-open at the
#      end of the video as the "last release" and used it as the cube anchor.
#  (b) when the goal names the cube colour, the cube is chosen by a live RGB check of that colour
#      inside each geometry-valid candidate box plus the colour-prompt SAM score; the episode
#      anchor (last kept video release) is only a cross-check / tie-break. v00 chose the generic
#      "cube" candidate nearest the (spurious) anchor and grasped a green cube.
#
# Phase 0  record context / metadata / cameras / robot state            (no control)
# Phase 1  conditioning-video robot-state trace -> grasp/release events   (no control)
# Phase 2  SAM3 prompt probes, front + wrist                               (no control)
# Phase 3  camera back-projection calibration, flat-pixel colour model,
#          target tracking through every video frame                       (no control)
# Phase 4  live stillness check (hold ticks, keep tracking)                (control)
# Phase 5  parse the goal's order position -> placement event -> target
# Phase 6  pick the cube   Phase 7 place on the chosen target   Phase 8 settle
#
# Every phase is wrapped; the program must never raise. Only public APIs are used and no
# coordinates, dimensions or pixel locations are hard-coded: everything is derived from the
# episode's own video robot states, live RGB-D and segmentation.

import math

VERSION = "v01_goal_colour_cube_select"
report = {"version": VERSION, "errors": [], "log": [], "phases": {}}
result = report
STATE = {"terminal": False, "grip_cmd": 1.0, "ticks_used_calls": 0}


def note(msg):
    text = str(msg)
    print("[" + VERSION + "] " + text[:300])
    if len(report["log"]) < 400:
        report["log"].append(text[:400])


def err(phase, exc):
    text = phase + ": " + str(exc)
    print("[" + VERSION + "][ERR] " + text[:500])
    report["errors"].append(text[:700])


def jsonable(value, depth=0):
    if value is None or value is True or value is False:
        return value
    try:
        text = value + ""
        return text[:300]
    except Exception:
        pass
    try:
        number = float(value)
        if math.isfinite(number) and number == int(number) and abs(number) < 1e12:
            return int(number)
        if math.isfinite(number):
            return round(number, 5)
        return str(number)
    except Exception:
        pass
    if depth > 4:
        return str(value)[:200]
    try:
        items = list(value.items())
        out = {}
        for key, val in items[:80]:
            out[str(key)] = jsonable(val, depth + 1)
        return out
    except Exception:
        pass
    try:
        seq = list(value)
        if len(seq) > 40:
            return {"len": len(seq), "head": [jsonable(x, depth + 1) for x in seq[:8]]}
        return [jsonable(x, depth + 1) for x in seq]
    except Exception:
        pass
    return str(value)[:200]


def to_list(arr):
    try:
        return arr.tolist()
    except Exception:
        pass
    try:
        return [to_list(x) for x in arr]
    except Exception:
        return arr


def fl(values):
    return [float(v) for v in values]


def r3(values):
    return [round(float(v), 4) for v in values]


def quantile(values, frac):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    idx = int((len(ordered) - 1) * frac)
    return ordered[idx]


def median(values):
    return quantile(values, 0.5)


def dist2(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def quat_to_mat(q):
    w, x, y, z = fl(q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        n = 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def quat_mul(a, b):
    aw, ax, ay, az = fl(a)
    bw, bx, by, bz = fl(b)
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def yaw_quat(delta):
    return [math.cos(delta / 2.0), 0.0, 0.0, math.sin(delta / 2.0)]


# ---------------------------------------------------------------- Phase 0: context
base_frame = "robot_base"
lang = ""
hz = 20.0
max_steps = None
try:
    ctx = get_task_context()
    lang = str(ctx.language)
    report["language"] = lang
    try:
        report["task_name"] = str(ctx.task_name)
        report["task_id"] = jsonable(ctx.task_id)
    except Exception as exc:
        err("ctx-ids", exc)
    try:
        report["context_metadata"] = jsonable(ctx.metadata)
    except Exception as exc:
        err("ctx-meta", exc)
    note("language: " + lang)
except Exception as exc:
    err("context", exc)

try:
    tmeta = robomme_plus.get_task_metadata()
    report["task_metadata"] = jsonable(tmeta)
    try:
        max_steps = int(tmeta["max_steps"])
    except Exception as exc:
        err("max_steps", exc)
    for key in ("control_hz", "control_frequency_hz", "frequency_hz", "control_frequency"):
        try:
            hz = float(tmeta[key])
            break
        except Exception:
            pass
    if not lang:
        try:
            lang = str(tmeta["language"])
        except Exception:
            pass
except Exception as exc:
    err("task-metadata", exc)
try:
    report["controller_metadata"] = jsonable(robomme_plus.get_controller_metadata())
except Exception as exc:
    err("controller-metadata", exc)
report["hz"] = hz
report["max_steps"] = max_steps
note("max_steps=" + str(max_steps) + " hz=" + str(hz))

live0 = None
embodiment = "robomme_plus"
try:
    live0 = get_observation()
    st0 = live0.robot_state
    base_frame = str(st0.base_frame)
    embodiment = str(st0.embodiment)
    cams = {}
    for name in live0.cameras:
        cam = live0.cameras[name]
        info = {}
        try:
            info["shape"] = jsonable(cam.rgb.shape)
            info["depth_shape"] = jsonable(cam.depth_m.shape)
            info["intrinsics"] = jsonable(to_list(cam.intrinsics))
            info["pose_position"] = r3(cam.camera_pose.position)
            info["pose_quat"] = r3(cam.camera_pose.quaternion_wxyz)
            info["pose_frame"] = str(cam.camera_pose.frame)
        except Exception as exc:
            err("camera-info-" + str(name), exc)
        cams[str(name)] = info
    report["cameras"] = cams
    ee0 = st0.end_effector_poses["primary"]
    report["robot_state0"] = {
        "joints": r3(st0.joint_positions["primary"]),
        "ee_pos": r3(ee0.position),
        "ee_quat": r3(ee0.quaternion_wxyz),
        "gripper": round(float(st0.gripper_positions["primary"]), 4),
        "timestamp_s": float(st0.timestamp_s),
        "base_frame": base_frame,
        "embodiment": embodiment,
    }
    STATE["grip_cmd"] = 1.0 if float(st0.gripper_positions["primary"]) > 0.85 else 0.0
    note("robot0 " + str(report["robot_state0"]))
except Exception as exc:
    err("live0", exc)


def tick_now():
    try:
        return int(round(float(get_robot_state().timestamp_s) * hz))
    except Exception:
        return None


def ticks_left():
    t = tick_now()
    if max_steps is None or t is None:
        return 100000
    return max_steps - t


# ---------------------------------------------------------------- Phase 1: video trace
video = []
vtrace = []
events = {"picks": [], "releases": [], "dwells": []}
try:
    raw_video = robomme_plus.get_conditioning_video()
    try:
        nvid = len(raw_video)
        video = raw_video
    except Exception:
        video = list(raw_video)
        nvid = len(video)
    report["video_len"] = nvid
    note("conditioning video frames: " + str(nvid))
    for idx in range(nvid):
        try:
            frame = video[idx]
            rs = frame.robot_state
            ee = rs.end_effector_poses["primary"]
            vtrace.append({
                "i": idx,
                "t": float(frame.timestamp_s),
                "p": fl(ee.position),
                "q": fl(ee.quaternion_wxyz),
                "g": float(rs.gripper_positions["primary"]),
            })
        except Exception as exc:
            if len(report["errors"]) < 50:
                err("video-frame-" + str(idx), exc)
    if len(vtrace) > 0:
        report["video_summary"] = {
            "t0": vtrace[0]["t"], "t1": vtrace[-1]["t"],
            "g_min": min(v["g"] for v in vtrace), "g_max": max(v["g"] for v in vtrace),
            "z_min": min(v["p"][2] for v in vtrace), "z_max": max(v["p"][2] for v in vtrace),
        }
        stride = max(1, len(vtrace) // 128)
        report["video_trace_sub"] = [
            [round(v["t"], 2)] + r3(v["p"]) + [round(v["g"], 3)] for v in vtrace[::stride]
        ]
        try:
            f0 = video[0].cameras["front"].camera_pose
            f1 = video[len(video) - 1].cameras["front"].camera_pose
            report["video_front_pose_first"] = r3(f0.position) + r3(f0.quaternion_wxyz)
            report["video_front_pose_last"] = r3(f1.position) + r3(f1.quaternion_wxyz)
        except Exception as exc:
            err("video-campose", exc)
except Exception as exc:
    err("video", exc)


def frame_joints(idx):
    try:
        return fl(video[idx].robot_state.joint_positions["primary"])
    except Exception:
        return None


def make_event(kind, idx):
    v = vtrace[idx]
    return {"kind": kind, "vi": idx, "frame": v["i"], "t": round(v["t"], 3), "p": v["p"],
            "q": v["q"], "g": round(v["g"], 3)}


try:
    if len(vtrace) >= 3:
        gs = [v["g"] for v in vtrace]
        gmax = max(gs)
        gmin = min(gs)
        span = gmax - gmin
        report["video_gripper_span"] = round(span, 4)
        if span > 0.15:
            open_t = gmax - 0.2 * span
            close_t = gmax - 0.35 * span
            state = "open" if gs[0] >= close_t else "closed"
            last_change = 0
            for k in range(1, len(gs)):
                if state == "open" and gs[k] < close_t:
                    # grasp: lowest EE within the few frames around the close
                    lo = max(0, k - 4)
                    best = lo
                    for m in range(lo, min(len(gs), k + 2)):
                        if vtrace[m]["p"][2] < vtrace[best]["p"][2]:
                            best = m
                    events["picks"].append(make_event("pick", best))
                    state = "closed"
                    last_change = k
                elif state == "closed" and gs[k] > open_t:
                    lo = max(0, k - 4)
                    best = lo
                    for m in range(lo, min(len(gs), k + 1)):
                        if vtrace[m]["p"][2] < vtrace[best]["p"][2]:
                            best = m
                    events["releases"].append(make_event("release", best))
                    state = "open"
                    last_change = k
            report["video_final_gripper_state"] = state
        # v01: keep only events made near the table while actually holding something
        rejected_spans = []
        if span > 0.15 and (events["picks"] or events["releases"]):
            zgate = min(v["p"][2] for v in vtrace) + 0.06
            raw_ev = []
            for ev in events["picks"] + events["releases"]:
                at = len(raw_ev)
                while at > 0 and raw_ev[at - 1]["vi"] > ev["vi"]:
                    at -= 1
                raw_ev.insert(at, ev)
            kept_p = []
            kept_r = []
            rejected = []
            holding_valid = False
            for n_e in range(len(raw_ev)):
                ev = raw_ev[n_e]
                low = ev["p"][2] < zgate
                if ev["kind"] == "pick":
                    end_vi = len(gs)
                    for later in raw_ev[n_e + 1:]:
                        if later["kind"] == "release":
                            end_vi = later["vi"]
                            break
                    a_vi = min(ev["vi"] + 3, end_vi)
                    seg = gs[a_vi:end_vi]
                    hold_g = median(seg) if len(seg) > 0 else ev["g"]
                    empty = hold_g < 0.05  # documented normalisation: 0 = fully closed (nothing between the fingers)
                    ev["hold_g"] = round(hold_g, 3)
                    if low and not empty:
                        kept_p.append(ev)
                        holding_valid = True
                    else:
                        holding_valid = False
                        rejected_spans.append((ev["vi"], end_vi))
                        rejected.append({"kind": "pick", "frame": ev["frame"], "z": round(ev["p"][2], 4),
                                         "hold_g": round(hold_g, 3), "low": low, "empty": empty})
                else:
                    if low and holding_valid:
                        kept_r.append(ev)
                    else:
                        rejected.append({"kind": "release", "frame": ev["frame"], "z": round(ev["p"][2], 4),
                                         "low": low, "after_kept_pick": holding_valid})
                    holding_valid = False
            events["picks"] = kept_p
            events["releases"] = kept_r
            report["video_events_rejected"] = rejected
            report["video_event_zgate"] = round(zgate, 4)
            report["video_final_gripper_state_raw"] = report.get("video_final_gripper_state")
            last_kind = None
            for ev in raw_ev:
                if (ev["kind"] == "pick" and ev in kept_p) or (ev["kind"] == "release" and ev in kept_r):
                    last_kind = ev["kind"]
            report["video_final_gripper_state"] = "closed" if last_kind == "pick" else "open"
        # dwell minima of EE height (gripper-independent)
        zs = [v["p"][2] for v in vtrace]
        zmax = max(zs)
        zmin = min(zs)
        if zmax - zmin > 0.04:
            low_t = zmin + 0.35 * (zmax - zmin)
            inside = False
            seg_best = None
            for k in range(len(zs)):
                if zs[k] < low_t:
                    if not inside:
                        inside = True
                        seg_best = k
                    elif zs[k] < zs[seg_best]:
                        seg_best = k
                else:
                    if inside:
                        events["dwells"].append(make_event("dwell", seg_best))
                        inside = False
            if inside and seg_best is not None:
                events["dwells"].append(make_event("dwell", seg_best))
            # v01: a dwell inside a rejected (empty / in-air) closure is not a placement
            events["dwells"] = [d for d in events["dwells"]
                                if not any(sp[0] <= d["vi"] <= sp[1] for sp in rejected_spans)]
    for kind in ("picks", "releases", "dwells"):
        for ev in events[kind]:
            ev["joints"] = frame_joints(ev["frame"])
    report["video_events"] = {
        kind: [{"frame": e["frame"], "t": e["t"], "p": r3(e["p"]), "q": r3(e["q"]), "g": e["g"]}
               for e in events[kind]] for kind in events
    }
    note("events picks=" + str(len(events["picks"])) + " releases=" + str(len(events["releases"]))
         + " dwells=" + str(len(events["dwells"])))
except Exception as exc:
    err("events", exc)


# ---------------------------------------------------------------- Phase 2: SAM probes
def seg_candidates(camera, prompt, want_cloud=True, limit=5):
    out = []
    try:
        found = segment_text(camera, prompt)
        if not found.ok:
            return out
        k = 0
        for cand in found.segmentations[:limit]:
            item = {"prompt": prompt, "camera": camera, "score": round(float(cand.score), 4),
                    "box": None, "cand": cand, "pts": None}
            try:
                if cand.box_xyxy is not None:
                    item["box"] = [float(v) for v in cand.box_xyxy]
            except Exception:
                pass
            if want_cloud:
                try:
                    cloud = mask_to_point_cloud(cand, camera, target_frame=base_frame)
                    pts = to_list(cloud.points)
                    item["pts"] = pts
                except Exception as exc:
                    item["cloud_error"] = str(exc)[:200]
            out.append(item)
            k += 1
    except Exception as exc:
        err("segment " + camera + " " + prompt, exc)
    return out


def cloud_stats(pts):
    if pts is None or len(pts) < 5:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    zs = [float(p[2]) for p in pts]
    return {
        "n": len(pts),
        "c": [median(xs), median(ys), median(zs)],
        "z10": quantile(zs, 0.1), "z90": quantile(zs, 0.9), "z98": quantile(zs, 0.98),
        "ex": quantile(xs, 0.95) - quantile(xs, 0.05),
        "ey": quantile(ys, 0.95) - quantile(ys, 0.05),
    }


def summarize_item(item):
    s = {"prompt": item["prompt"], "camera": item["camera"], "score": item["score"],
         "box": [round(v, 1) for v in item["box"]] if item["box"] else None}
    cs = item.get("stats")
    if cs:
        s["n"] = cs["n"]
        s["c"] = r3(cs["c"])
        s["z10"] = round(cs["z10"], 4)
        s["z90"] = round(cs["z90"], 4)
        s["ex"] = round(cs["ex"], 4)
        s["ey"] = round(cs["ey"], 4)
    if "cloud_error" in item:
        s["cloud_error"] = item["cloud_error"]
    return s


GOAL_WORDS = []
COLOR_WORDS = ("red", "green", "blue", "yellow", "purple", "orange", "pink", "white", "black",
               "brown", "gray", "grey", "cyan", "magenta")
low_lang = lang.lower()
clean = ""
for ch in low_lang:
    if ch.isalnum() or ch == "-":
        clean += ch
    else:
        clean += " "
tokens = [t for t in clean.replace("-", " ").split(" ") if t]
cube_color = None
for i, tok in enumerate(tokens):
    if tok in COLOR_WORDS and i + 1 < len(tokens) and tokens[i + 1] in ("cube", "block", "box"):
        cube_color = tok
report["cube_color_from_goal"] = cube_color

cube_prompts = []
if cube_color:
    cube_prompts.append(cube_color + " cube")
cube_prompts += ["cube", "block"]
target_prompts = ["target", "red and white target", "bullseye", "disc", "round target"]

front_items = []
wrist_items = []
probe_summary = []
try:
    for prompt in target_prompts + cube_prompts + ["table"]:
        items = seg_candidates("front", prompt, want_cloud=True, limit=6)
        for it in items:
            it["stats"] = cloud_stats(it["pts"])
            front_items.append(it)
        probe_summary.append({"camera": "front", "prompt": prompt, "n": len(items),
                              "cands": [summarize_item(it) for it in items]})
    for prompt in ["cube", "target", "disc"]:
        items = seg_candidates("wrist", prompt, want_cloud=True, limit=4)
        for it in items:
            it["stats"] = cloud_stats(it["pts"])
            wrist_items.append(it)
        probe_summary.append({"camera": "wrist", "prompt": prompt, "n": len(items),
                              "cands": [summarize_item(it) for it in items]})
    report["prompt_probes"] = probe_summary
    note("probes done: front cands=" + str(len(front_items)) + " wrist cands=" + str(len(wrist_items)))
except Exception as exc:
    err("probes", exc)


# ---------------------------------------------------------------- Phase 3: calibration + tracking
def cam_params(cam):
    K = to_list(cam.intrinsics)
    pose = cam.camera_pose
    shape = cam.rgb.shape
    return {"fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]), "cy": float(K[1][2]),
            "p": fl(pose.position), "R": quat_to_mat(pose.quaternion_wxyz),
            "key": r3(pose.position) + r3(pose.quaternion_wxyz),
            "H": int(shape[0]), "W": int(shape[1])}


def ray_dir(par, conv, u, v):
    a = (u - par["cx"]) / par["fx"]
    b = (v - par["cy"]) / par["fy"]
    if conv == 0:
        dc = (a, b, 1.0)
    elif conv == 1:
        dc = (a, -b, -1.0)
    else:
        dc = (1.0, -a, -b)
    R = par["R"]
    return (R[0][0] * dc[0] + R[0][1] * dc[1] + R[0][2] * dc[2],
            R[1][0] * dc[0] + R[1][1] * dc[1] + R[1][2] * dc[2],
            R[2][0] * dc[0] + R[2][1] * dc[1] + R[2][2] * dc[2])


def make_grid(par, conv, stride):
    grid = []
    i = 0
    for v in range(0, par["H"], stride):
        j = 0
        for u in range(0, par["W"], stride):
            k = ray_dir(par, conv, float(u), float(v))
            grid.append((i, j, k[0], k[1], k[2], u, v))
            j += 1
        i += 1
    return grid


def sub_arrays(cam, stride):
    try:
        rgb = cam.rgb[::stride, ::stride].tolist()
        dep = cam.depth_m[::stride, ::stride].tolist()
        return rgb, dep
    except Exception:
        pass
    rgb_full = to_list(cam.rgb)
    dep_full = to_list(cam.depth_m)
    rgb = [row[::stride] for row in rgb_full[::stride]]
    dep = [row[::stride] for row in dep_full[::stride]]
    return rgb, dep


def color_bin(px):
    r = int(px[0])
    g = int(px[1])
    b = int(px[2])
    return (r // 32) * 64 + (g // 32) * 8 + (b // 32)


calib = {"conv": None}
table_z = None
TRACK = {"ok": False}
try:
    if live0 is not None:
        fpar = cam_params(live0.cameras["front"])
        frgb_full = to_list(live0.cameras["front"].rgb)
        fdep_full = to_list(live0.cameras["front"].depth_m)
        errs = [[], [], []]
        for it in front_items:
            cs = it.get("stats")
            if not cs or not it["box"] or cs["n"] < 20:
                continue
            u = int((it["box"][0] + it["box"][2]) / 2.0)
            v = int((it["box"][1] + it["box"][3]) / 2.0)
            ds = []
            for dv in (-1, 0, 1):
                for du in (-1, 0, 1):
                    vv = min(max(v + dv, 0), fpar["H"] - 1)
                    uu = min(max(u + du, 0), fpar["W"] - 1)
                    d = float(fdep_full[vv][uu])
                    if d > 0.0 and d < 10.0:
                        ds.append(d)
            if not ds:
                continue
            d = median(ds)
            for conv in (0, 1, 2):
                k = ray_dir(fpar, conv, float(u), float(v))
                pt = (fpar["p"][0] + d * k[0], fpar["p"][1] + d * k[1], fpar["p"][2] + d * k[2])
                e = math.sqrt((pt[0] - cs["c"][0]) ** 2 + (pt[1] - cs["c"][1]) ** 2 + (pt[2] - cs["c"][2]) ** 2)
                errs[conv].append(e)
        med_err = [median(e) if e else None for e in errs]
        calib["median_err"] = [round(e, 4) if e is not None else None for e in med_err]
        calib["n_pairs"] = len(errs[0])
        best = None
        for conv in (0, 1, 2):
            if med_err[conv] is not None and (best is None or med_err[conv] < med_err[best]):
                best = conv
        if best is not None and med_err[best] < 0.04:
            calib["conv"] = best
        note("calibration " + str(calib))
except Exception as exc:
    err("calibration", exc)
report["calibration"] = calib

# table plane
try:
    tz_sam = []
    for it in front_items:
        if it["prompt"] == "table" and it.get("stats"):
            tz_sam.append(it["stats"]["c"][2])
    if calib["conv"] is not None:
        grid4 = make_grid(fpar, calib["conv"], 4)
        rgb4, dep4 = sub_arrays(live0.cameras["front"], 4)
        hist = {}
        zs_all = []
        for cell in grid4:
            d = float(dep4[cell[0]][cell[1]])
            if d <= 0.0 or d > 10.0:
                continue
            z = fpar["p"][2] + d * cell[4]
            zs_all.append(z)
            key = int(math.floor(z / 0.005))
            hist[key] = hist.get(key, 0) + 1
        best_key = None
        for key in hist:
            if best_key is None or hist[key] > hist[best_key]:
                best_key = key
        if best_key is not None:
            near = [z for z in zs_all if abs(z - (best_key + 0.5) * 0.005) < 0.01]
            table_z = median(near)
    if table_z is None and tz_sam:
        table_z = median(tz_sam)
    report["table_z"] = table_z
    report["table_z_sam"] = [round(z, 4) for z in tz_sam]
    note("table_z=" + str(table_z))
except Exception as exc:
    err("table", exc)


def flat_target_items(items):
    found = []
    for it in items:
        cs = it.get("stats")
        if not cs or cs["n"] < 15 or it["prompt"] not in target_prompts:
            continue
        if table_z is not None:
            if cs["z90"] - table_z > 0.02 or cs["z10"] - table_z < -0.03:
                continue
        if cs["ex"] < 0.015 or cs["ey"] < 0.015 or cs["ex"] > 0.3 or cs["ey"] > 0.3:
            continue
        dup = False
        for f in found:
            if dist2(f["stats"]["c"], cs["c"]) < 0.03:
                dup = True
                if it["score"] > f["score"]:
                    f["stats"] = cs
                    f["score"] = it["score"]
                    f["box"] = it["box"]
                break
        if not dup:
            found.append({"stats": cs, "score": it["score"], "box": it["box"], "prompt": it["prompt"]})
    return found


sam_targets = []
try:
    sam_targets = flat_target_items(front_items)
    report["sam_targets"] = [{"c": r3(t["stats"]["c"]), "score": t["score"], "prompt": t["prompt"],
                              "box": t["box"]} for t in sam_targets]
    note("sam flat targets: " + str(len(sam_targets)))
except Exception as exc:
    err("sam-targets", exc)

FLAT_LO = -0.012
FLAT_HI = 0.015


def frame_points(cam, par_cache, bins, stride):
    # returns list of (x, y) base-frame points of flat, target-coloured pixels
    par = cam_params(cam)
    key = str(par["key"]) + "|" + str(stride)
    if key not in par_cache:
        par_cache[key] = (par, make_grid(par, calib["conv"], stride))
    par, grid = par_cache[key]
    rgb, dep = sub_arrays(cam, stride)
    px0 = par["p"][0]
    py0 = par["p"][1]
    pz0 = par["p"][2]
    lo = table_z + FLAT_LO
    hi = table_z + FLAT_HI
    pts = []
    for cell in grid:
        d = dep[cell[0]][cell[1]]
        if d <= 0.0 or d > 10.0:
            continue
        z = pz0 + d * cell[4]
        if z < lo or z > hi:
            continue
        if color_bin(rgb[cell[0]][cell[1]]) in bins:
            pts.append((px0 + d * cell[2], py0 + d * cell[3]))
    return pts


def cluster_xy(pts, radius):
    clusters = []
    for p in pts:
        best = None
        bd = radius
        for c in clusters:
            d = math.sqrt((c[0] / c[2] - p[0]) ** 2 + (c[1] / c[2] - p[1]) ** 2)
            if d < bd:
                bd = d
                best = c
        if best is None:
            clusters.append([p[0], p[1], 1])
        else:
            best[0] += p[0]
            best[1] += p[1]
            best[2] += 1
    merged = True
    while merged:
        merged = False
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                ca = clusters[a]
                cb = clusters[b]
                d = math.sqrt((ca[0] / ca[2] - cb[0] / cb[2]) ** 2 + (ca[1] / ca[2] - cb[1] / cb[2]) ** 2)
                if d < radius:
                    ca[0] += cb[0]
                    ca[1] += cb[1]
                    ca[2] += cb[2]
                    clusters = clusters[:b] + clusters[b + 1:]
                    merged = True
                    break
            if merged:
                break
    return [(c[0] / c[2], c[1] / c[2], c[2]) for c in clusters]


bins = set()
min_pts = 3
n_targets = 0
par_cache = {}
TS = {"stride": 4, "fstep": 1}


def accumulate_bins(cam, centers, inside, outside, stride, elevated=None):
    # centers: list of (x, y, rim_radius). Flat pixels well inside a rim count as target colour,
    # flat pixels well outside every rim as background; the band in between is ignored.
    par = cam_params(cam)
    grid = make_grid(par, calib["conv"], stride)
    rgb, dep = sub_arrays(cam, stride)
    n_in = 0
    n_out = 0
    for cell in grid:
        d = float(dep[cell[0]][cell[1]])
        if d <= 0.0 or d > 10.0:
            continue
        z = par["p"][2] + d * cell[4]
        if z < table_z + FLAT_LO or z > table_z + 0.12:
            continue
        x = par["p"][0] + d * cell[2]
        y = par["p"][1] + d * cell[3]
        b = color_bin(rgb[cell[0]][cell[1]])
        if z > table_z + 0.02:
            # elevated colour near a target centre = the cube (or the hand), never the target
            if elevated is not None:
                for c in centers:
                    if math.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2) < 1.5 * c[2]:
                        elevated[b] = elevated.get(b, 0) + 1
                        break
            continue
        if z > table_z + FLAT_HI:
            continue
        rel = None
        for c in centers:
            dn = math.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2) / c[2]
            if rel is None or dn < rel:
                rel = dn
        if rel is not None and rel < 0.75:
            inside[b] = inside.get(b, 0) + 1
            n_in += 1
        elif rel is None or rel > 1.4:
            outside[b] = outside.get(b, 0) + 1
            n_out += 1
    return n_in, n_out


def select_bins(inside, outside, n_in, n_out, elevated):
    chosen_bins = set()
    for b in inside:
        if elevated.get(b, 0) > 0.5 * inside[b]:
            continue
        fi = inside[b] / max(1.0, float(n_in))
        fo = outside.get(b, 0) / max(1.0, float(n_out))
        if inside[b] >= 3 and fi >= 0.04 and (fo <= 0.0 or fi / fo >= 4.0):
            chosen_bins.add(b)
    return chosen_bins


try:
    if calib["conv"] is not None and table_z is not None:
        inside = {}
        outside = {}
        elevated = {}
        n_in = 0
        n_out = 0
        source_cm = None
        if len(sam_targets) > 0:
            centers = [(t["stats"]["c"][0], t["stats"]["c"][1],
                        0.25 * (t["stats"]["ex"] + t["stats"]["ey"]) + 0.002) for t in sam_targets]
            n_in, n_out = accumulate_bins(live0.cameras["front"], centers, inside, outside, 2, elevated)
            source_cm = "live SAM flat targets"
        elif len(events["releases"]) > 0 and len(video) > 0:
            # fallback: the video frame just after each release, once the hand has risen >= 6 cm
            for ev in events["releases"]:
                fi = None
                for m in range(ev["frame"], min(len(vtrace), ev["frame"] + 40)):
                    if vtrace[m]["p"][2] > ev["p"][2] + 0.06:
                        fi = m
                        break
                if fi is None:
                    continue
                a, b2 = accumulate_bins(video[fi].cameras["front"], [(ev["p"][0], ev["p"][1], 0.035)],
                                        inside, outside, 1, elevated)
                n_in += a
                n_out += b2
            source_cm = "video frames after each release (no SAM target)"
        bins = select_bins(inside, outside, n_in, n_out, elevated)
        covered = sum(inside[b] for b in bins)
        report["color_model"] = {
            "source": source_cm, "n_in": n_in, "n_out": n_out, "covered": covered,
            "bins": sorted(list(bins)),
            "inside_top": sorted([[inside[b], b, outside.get(b, 0), elevated.get(b, 0)] for b in inside],
                                 reverse=True)[:12],
        }
        if len(bins) > 0:
            live_pts = frame_points(live0.cameras["front"], par_cache, bins, 4)
            live_clusters = cluster_xy(live_pts, 0.05)
            sizes = sorted([c[2] for c in live_clusters], reverse=True)
            if sizes:
                min_pts = max(3, int(0.2 * sizes[0]))
            good = [c for c in live_clusters if c[2] >= min_pts]
            if good and min([c[2] for c in good]) < 8:
                TS["stride"] = 2
                live_pts = frame_points(live0.cameras["front"], par_cache, bins, 2)
                live_clusters = cluster_xy(live_pts, 0.05)
                sizes = sorted([c[2] for c in live_clusters], reverse=True)
                if sizes:
                    min_pts = max(3, int(0.2 * sizes[0]))
                good = [c for c in live_clusters if c[2] >= min_pts]
            n_targets = len(good)
            report["live_color_clusters"] = [[round(c[0], 4), round(c[1], 4), c[2]] for c in live_clusters]
            note("colour bins=" + str(len(bins)) + " covered=" + str(covered) + "/" + str(n_in)
                 + " live clusters=" + str(len(good)) + " min_pts=" + str(min_pts) + " src=" + str(source_cm))
            if len(sam_targets) > 0 and n_targets != len(sam_targets):
                note("WARN live colour clusters " + str(n_targets) + " != sam targets " + str(len(sam_targets)))
except Exception as exc:
    err("color-model", exc)


def tracker_init(positions):
    return [{"x": p[0], "y": p[1], "vx": 0.0, "vy": 0.0, "unseen": 0, "x0": p[0], "y0": p[1],
             "maxd": 0.0, "path": 0.0} for p in positions]


def tracker_update(tracks, pts):
    n = len(tracks)
    assign = [[] for k in range(n)]
    preds = []
    for t in tracks:
        preds.append((t["x"] + t["vx"], t["y"] + t["vy"]))
    for p in pts:
        best = None
        bd = None
        for k in range(n):
            gate = min(0.15, 0.06 + 0.01 * tracks[k]["unseen"])
            d = math.sqrt((preds[k][0] - p[0]) ** 2 + (preds[k][1] - p[1]) ** 2)
            if d < gate and (bd is None or d < bd):
                bd = d
                best = k
        if best is not None:
            assign[best].append(p)
    newpos = []
    for k in range(n):
        if len(assign[k]) >= max(2, int(0.5 * min_pts)):
            newpos.append((median([p[0] for p in assign[k]]), median([p[1] for p in assign[k]]),
                           len(assign[k])))
        else:
            newpos.append(None)
    # conflict: two tracks collapsing onto one blob -> keep the one with more support
    for a in range(n):
        for b in range(a + 1, n):
            if newpos[a] is not None and newpos[b] is not None:
                if math.sqrt((newpos[a][0] - newpos[b][0]) ** 2 + (newpos[a][1] - newpos[b][1]) ** 2) < 0.02:
                    if newpos[a][2] >= newpos[b][2]:
                        newpos[b] = None
                    else:
                        newpos[a] = None
    for k in range(n):
        t = tracks[k]
        if newpos[k] is None:
            t["unseen"] += 1
            t["vx"] *= 0.5
            t["vy"] *= 0.5
        else:
            dx = newpos[k][0] - t["x"]
            dy = newpos[k][1] - t["y"]
            t["path"] += math.sqrt(dx * dx + dy * dy)
            t["vx"] = 0.5 * t["vx"] + 0.5 * dx
            t["vy"] = 0.5 * t["vy"] + 0.5 * dy
            t["x"] = newpos[k][0]
            t["y"] = newpos[k][1]
            t["unseen"] = 0
            t["maxd"] = max(t["maxd"], math.sqrt((t["x"] - t["x0"]) ** 2 + (t["y"] - t["y0"]) ** 2))
    return [(t["x"], t["y"]) for t in tracks]


tracks = []
track_hist = []
try:
    if len(bins) > 0 and n_targets >= 1 and len(video) > 0:
        nv = len(video)
        init_at = None
        init_pos = None
        for idx in range(0, max(1, nv // 2)):
            pts = frame_points(video[idx].cameras["front"], par_cache, bins, TS["stride"])
            cl = [c for c in cluster_xy(pts, 0.05) if c[2] >= min_pts]
            if len(cl) == n_targets:
                init_at = idx
                init_pos = [(c[0], c[1]) for c in cl]
                break
        if init_pos is None:
            init_at = 0
            init_pos = [(c[0], c[1]) for c in report["live_color_clusters"] if c[2] >= min_pts]
            note("WARN tracker init fell back to live clusters (assumes no motion before first frame)")
        tracks = tracker_init(init_pos)
        for idx in range(init_at):
            track_hist.append([(t["x"], t["y"]) for t in tracks])
        npts = []
        cells = (256 // TS["stride"]) ** 2
        TS["fstep"] = max(1, int(math.ceil(nv * cells / 4200000.0)))
        for idx in range(init_at, nv):
            if (idx - init_at) % TS["fstep"] != 0 and idx != nv - 1:
                track_hist.append([(t["x"], t["y"]) for t in tracks])
                continue
            pts = frame_points(video[idx].cameras["front"], par_cache, bins, TS["stride"])
            npts.append(len(pts))
            track_hist.append(tracker_update(tracks, pts))
        TRACK["ok"] = True
        TRACK["init_at"] = init_at
        TRACK["video_end"] = [(t["x"], t["y"]) for t in tracks]
        report["tracking"] = {
            "init_at": init_at, "n": len(tracks), "stride": TS["stride"], "fstep": TS["fstep"],
            "init": [r3(p) for p in init_pos],
            "video_end": [r3(p) for p in TRACK["video_end"]],
            "maxd": [round(t["maxd"], 4) for t in tracks],
            "path": [round(t["path"], 4) for t in tracks],
            "hist_sub": [[r3(p) for p in h] for h in track_hist[::max(1, len(track_hist) // 48)]],
            "npts_sub": npts[::max(1, len(npts) // 64)],
        }
        note("tracking ok n=" + str(len(tracks)) + " maxd=" + str(report["tracking"]["maxd"]))
except Exception as exc:
    err("tracking", exc)
    TRACK["ok"] = False


def match_event_to_track(ev):
    if not TRACK["ok"] or len(track_hist) == 0:
        return None, None
    fi = min(ev["frame"], len(track_hist) - 1)
    best = None
    bd = None
    # use a few frames before the event as well (gripper occludes the target at release)
    for f in (fi, max(0, fi - 3), max(0, fi - 8)):
        for k in range(len(track_hist[f])):
            d = dist2(track_hist[f][k], ev["p"])
            if bd is None or d < bd:
                bd = d
                best = k
        if bd is not None and bd < 0.04:
            break
    return best, bd


try:
    for kind in ("releases", "dwells", "picks"):
        for ev in events[kind]:
            k, d = match_event_to_track(ev)
            ev["track"] = k
            ev["track_dist"] = round(d, 4) if d is not None else None
    report["event_tracks"] = {kind: [[e["frame"], e.get("track"), e.get("track_dist")] for e in events[kind]]
                              for kind in events}
except Exception as exc:
    err("event-match", exc)


# ---------------------------------------------------------------- control helpers
def is_terminal(res, label):
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
        if bool(res.terminal):
            term = True
    except Exception:
        pass
    try:
        if bool(res.done):
            term = True
    except Exception:
        pass
    try:
        sval = str(res.status).lower()
        if "terminat" in sval or "truncat" in sval or "episode_end" in sval or "done" == sval:
            term = True
    except Exception:
        pass
    if term and not STATE["terminal"]:
        STATE["terminal"] = True
        note("TERMINAL after " + label + " at tick " + str(tick_now()))
    return term


def summarize_res(res, label):
    s = {"label": label}
    try:
        s["ok"] = bool(res.ok)
    except Exception:
        pass
    try:
        s["status"] = str(res.status)
    except Exception:
        pass
    try:
        s["steps"] = int(res.steps_executed)
    except Exception:
        pass
    try:
        if res.error is not None:
            s["error"] = str(res.error)[:300]
    except Exception:
        pass
    s["tick"] = tick_now()
    try:
        s["ee"] = r3(get_robot_state().end_effector_poses["primary"].position)
        s["g"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
    except Exception:
        pass
    is_terminal(res, label)
    s["terminal"] = STATE["terminal"]
    if "motions" not in report:
        report["motions"] = []
    report["motions"].append(s)
    print("[" + VERSION + "] motion " + str(s))
    return s


def hold(n, label="hold"):
    done = 0
    try:
        st = get_robot_state()
        q = fl(st.joint_positions["primary"])
        while done < n and not STATE["terminal"]:
            r = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=q, gripper_position=STATE["grip_cmd"],
                embodiment=st.embodiment)}))
            done += 1
            if is_terminal(r, label):
                break
    except Exception as exc:
        err(label, exc)
    return done


def goto(pos, quat, label, tol=0.01, steps=200, seed=None):
    if STATE["terminal"]:
        return False
    pose = Pose([float(pos[0]), float(pos[1]), float(pos[2])], fl(quat), base_frame)
    ok = False
    try:
        res = move_to_pose(pose, tolerance=tol, max_steps=steps)
        s = summarize_res(res, label)
        ok = bool(s.get("ok", False))
        if not ok and not STATE["terminal"]:
            msg = str(s.get("error", "")) + str(s.get("status", ""))
            if "max_steps" in msg and steps < 400:
                res = move_to_pose(pose, tolerance=tol, max_steps=400)
                s = summarize_res(res, label + "-long")
                ok = bool(s.get("ok", False))
        if not ok and not STATE["terminal"]:
            ik = solve_ik(pose, seed_joints=seed) if seed is not None else solve_ik(pose)
            if ik.ok:
                res = move_to_joints(fl(ik.joint_positions), tolerance=0.01, max_steps=steps)
                s = summarize_res(res, label + "-ik")
                ok = bool(s.get("ok", False))
            else:
                note(label + " ik failed: " + str(ik.error)[:200])
    except Exception as exc:
        err("goto " + label, exc)
    try:
        ee = get_robot_state().end_effector_poses["primary"].position
        e = math.sqrt((float(ee[0]) - pos[0]) ** 2 + (float(ee[1]) - pos[1]) ** 2 + (float(ee[2]) - pos[2]) ** 2)
        if e < tol * 2.5:
            ok = True
        if "motion_err" not in report:
            report["motion_err"] = []
        report["motion_err"].append([label, round(e, 4)])
    except Exception:
        pass
    return ok


def grip(open_it, label):
    if STATE["terminal"]:
        return None
    try:
        if open_it:
            res = open_gripper()
            STATE["grip_cmd"] = 1.0
        else:
            res = close_gripper()
            STATE["grip_cmd"] = 0.0
        summarize_res(res, label)
    except Exception as exc:
        err(label, exc)
    try:
        return float(get_robot_state().gripper_positions["primary"])
    except Exception:
        return None


def track_live(n_ticks, label):
    # hold n ticks, updating target tracks from each live front frame
    moves = []
    if not TRACK["ok"]:
        return moves
    for k in range(n_ticks):
        if STATE["terminal"]:
            break
        before = [(t["x"], t["y"]) for t in tracks]
        hold(1, label)
        try:
            obs = get_observation()
            pts = frame_points(obs.cameras["front"], par_cache, bins, TS["stride"])
            after = tracker_update(tracks, pts)
            mv = max([dist2(before[i], after[i]) for i in range(len(after))] + [0.0])
            moves.append(round(mv, 4))
        except Exception as exc:
            err("track_live", exc)
            break
    return moves


# ---------------------------------------------------------------- Phase 4: live stillness
try:
    if TRACK["ok"]:
        obs = get_observation()
        before = [(t["x"], t["y"]) for t in tracks]
        tracker_update(tracks, frame_points(obs.cameras["front"], par_cache, bins, TS["stride"]))
        jump = [round(dist2(before[i], (tracks[i]["x"], tracks[i]["y"])), 4) for i in range(len(tracks))]
        moves = track_live(4, "still-check")
        still_moves = list(moves)
        quiet = 0
        waited = 0
        if max(moves + [0.0]) > 0.004:
            note("targets moving live; tracking until still")
            while waited < 160 and quiet < 8 and not STATE["terminal"] and ticks_left() > 260:
                mv = track_live(1, "wait-still")
                waited += 1
                if mv and mv[0] < 0.002:
                    quiet += 1
                else:
                    quiet = 0
                if mv:
                    still_moves.append(mv[0])
        report["live_still"] = {"jump_from_video_end": jump, "moves": still_moves[:200],
                                "waited": waited,
                                "live_pos": [r3((t["x"], t["y"])) for t in tracks]}
        note("live stillness " + str(report["live_still"]["jump_from_video_end"]) + " waited=" + str(waited))
    else:
        hold(2, "first-hold")
except Exception as exc:
    err("live-still", exc)


# ---------------------------------------------------------------- Phase 5: goal parsing
ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
       "eighth": 8, "ninth": 9, "tenth": 10, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
       "6th": 6, "7th": 7, "8th": 8, "earliest": 1, "initial": 1, "last": -1, "final": -1,
       "latest": -1, "penultimate": -2}
NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
VERBS = ("pick", "grab", "grasp", "take", "lift", "move", "go", "put", "place")
plan = {"mode": None, "k": None, "cands": []}
try:
    cands = []
    for i, tok in enumerate(tokens):
        val = None
        if tok in ORD:
            val = ORD[tok]
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            nxt2 = tokens[i + 2] if i + 2 < len(tokens) else ""
            if val > 0 and (nxt == "last" or (nxt == "to" and nxt2 == "last")):
                val = -val  # "second to last" -> -2, "third last" -> -3
            if tok == "last" and i > 0 and (tokens[i - 1] == "to" or
                                            (tokens[i - 1] in ORD and ORD[tokens[i - 1]] > 0)):
                continue
            if tok in ("first",) and (i == 0 or tokens[i - 1] in ("and", "then")) and nxt in VERBS:
                cands.append({"tok": tok, "i": i, "val": val, "skipped": "sequencing word"})
                continue
        elif tok.isdigit() and i > 0 and tokens[i - 1] in ("number", "no", "target", "position", "step", "time"):
            val = int(tok)
        elif tok in NUMW and i > 0 and tokens[i - 1] in ("number", "target", "position", "step"):
            val = NUMW[tok]
        if val is not None:
            cands.append({"tok": tok, "i": i, "val": val})
    plan["cands"] = cands
    usable = [c for c in cands if "skipped" not in c]
    if usable:
        plan["mode"] = "single"
        plan["k"] = usable[-1]["val"]
    elif "order" in tokens or "sequence" in tokens or "same" in tokens:
        plan["mode"] = "sequence"
    else:
        plan["mode"] = "single"
        plan["k"] = -1
        plan["default"] = "no ordinal found; defaulted to the last placement"
    plan["reverse"] = "reverse" in tokens or "reversed" in tokens
    note("plan " + str(plan))
except Exception as exc:
    err("parse", exc)
    plan["mode"] = "single"
    plan["k"] = -1
report["plan"] = plan


def merged_events(lst):
    out = []
    for ev in lst:
        if out and dist2(out[-1]["p"], ev["p"]) < 0.02 and (
                ev.get("track") is None or ev.get("track") == out[-1].get("track")):
            continue
        out.append(ev)
    return out


placements = merged_events(events["releases"])
source = "releases"
try:
    need = plan["k"] if plan["k"] is not None else 1
    need_n = need if need > 0 else -need
    if len(placements) < max(1, need_n):
        dw = []
        # dwells while holding that are not at a pick location
        for ev in events["dwells"]:
            near_pick = False
            for pk in events["picks"]:
                if abs(pk["frame"] - ev["frame"]) <= 3:
                    near_pick = True
            if not near_pick:
                dw.append(ev)
        dw = merged_events(dw)
        if len(dw) >= max(1, need_n) and len(dw) > len(placements):
            placements = dw
            source = "dwells"
    report["placements"] = {"source": source, "list": [
        {"frame": e["frame"], "t": e["t"], "p": r3(e["p"]), "track": e.get("track"),
         "track_dist": e.get("track_dist")} for e in placements]}
    note("placements (" + source + "): " + str(len(placements)))
except Exception as exc:
    err("placements", exc)

chosen = []
try:
    if plan["mode"] == "sequence":
        chosen = list(placements)
        if plan.get("reverse"):
            chosen = list(reversed(chosen))
    else:
        k = plan["k"]
        if k is not None and len(placements) > 0:
            if k > 0 and k <= len(placements):
                chosen = [placements[k - 1]]
            elif k < 0 and -k <= len(placements):
                chosen = [placements[len(placements) + k]]
            else:
                note("WARN ordinal " + str(k) + " outside " + str(len(placements)) + " placements")
                chosen = [placements[min(len(placements) - 1, max(0, k - 1))]]
                plan["clamped"] = True
except Exception as exc:
    err("choose", exc)


def live_target_xy(ev):
    # identity through the tracker if possible, otherwise the EE release xy snapped to a SAM target
    info = {"event_frame": ev["frame"], "ee_xy": r3(ev["p"][:2])}
    xy = None
    if TRACK["ok"] and ev.get("track") is not None and ev.get("track_dist") is not None \
            and ev["track_dist"] < 0.06:
        t = tracks[ev["track"]]
        xy = (t["x"], t["y"])
        info["via"] = "track " + str(ev["track"])
    else:
        xy = (ev["p"][0], ev["p"][1])
        info["via"] = "ee-release-xy"
    best = None
    bd = None
    for t in sam_targets:
        d = dist2(t["stats"]["c"], xy)
        if bd is None or d < bd:
            bd = d
            best = t
    if best is not None and bd < 0.04:
        info["snap_sam_d"] = round(bd, 4)
        xy = (0.5 * (xy[0] + best["stats"]["c"][0]), 0.5 * (xy[1] + best["stats"]["c"][1]))
        info["target_surface_z"] = round(best["stats"]["z90"], 4)
    info["xy"] = r3(xy)
    return xy, info


# ---------------------------------------------------------------- Phase 6/7: pick and place
def pointing_down(q):
    R = quat_to_mat(q)
    return R[2][2] < -0.85


def grasp_quat_default():
    for ev in reversed(events["releases"]):
        if pointing_down(ev["q"]):
            return ev["q"]
    for ev in events["picks"]:
        if pointing_down(ev["q"]):
            return ev["q"]
    try:
        q = fl(get_robot_state().end_effector_poses["primary"].quaternion_wxyz)
        if pointing_down(q):
            return q
    except Exception:
        pass
    return [0.0, 1.0, 0.0, 0.0]


def cube_candidates(items):
    out = []
    for it in items:
        cs = it.get("stats")
        if not cs or cs["n"] < 10 or it["prompt"] not in cube_prompts:
            continue
        if table_z is not None:
            h = cs["z98"] - table_z
            if h < 0.012 or h > 0.15:
                continue
        if cs["ex"] > 0.12 or cs["ey"] > 0.12:
            continue
        out.append(it)
    return out


def expected_cube_xy():
    # where the cube was last released: by this program, else by the video robot
    if STATE.get("last_release") is not None:
        return STATE["last_release"][:2], "own-last-release"
    if report.get("video_final_gripper_state") == "open" and events["releases"]:
        return events["releases"][-1]["p"][:2], "last-video-release"
    if events["picks"]:
        return events["picks"][0]["p"][:2], "first-video-pick"
    return None, "none"


def top_face_fit(pts, z_top):
    top = [p for p in pts if float(p[2]) > z_top - 0.008]
    if len(top) < 12:
        return None
    cx = median([float(p[0]) for p in top])
    cy = median([float(p[1]) for p in top])
    best = None
    for step_deg in range(0, 90, 3):
        a = math.radians(step_deg)
        ca = math.cos(a)
        sa = math.sin(a)
        us = [(float(p[0]) - cx) * ca + (float(p[1]) - cy) * sa for p in top]
        vs = [-(float(p[0]) - cx) * sa + (float(p[1]) - cy) * ca for p in top]
        area = (quantile(us, 0.95) - quantile(us, 0.05)) * (quantile(vs, 0.95) - quantile(vs, 0.05))
        if best is None or area < best[0]:
            best = (area, a)
    return {"c": (cx, cy), "yaw": best[1], "n": len(top)}


def align_yaw(q, face_yaw):
    R = quat_to_mat(q)
    gy = math.atan2(R[1][1], R[0][1])  # closing axis (tool y) heading
    best = None
    for m in range(-4, 5):
        target = face_yaw + m * math.pi / 2.0
        delta = target - gy
        while delta > math.pi:
            delta -= 2 * math.pi
        while delta < -math.pi:
            delta += 2 * math.pi
        if best is None or abs(delta) < abs(best):
            best = delta
    return quat_mul(yaw_quat(best), q), best


def pix_has_colour(colour, px):
    r = float(px[0])
    g = float(px[1])
    b = float(px[2])
    if colour == "red":
        return r >= 90 and g <= 0.45 * r and b <= 0.45 * r
    if colour == "green":
        return g >= 80 and r <= 0.6 * g and b <= 0.6 * g
    if colour == "blue":
        return b >= 80 and r <= 0.6 * b and g <= 0.75 * b
    if colour == "yellow":
        return r >= 120 and g >= 120 and b <= 0.5 * min(r, g)
    return None


def colour_fraction(cam, box, colour):
    # fraction of pixels of the named colour in the central half of a candidate box
    if box is None or colour is None or pix_has_colour(colour, (0, 0, 0)) is None:
        return None
    w = box[2] - box[0]
    h = box[3] - box[1]
    x0 = int(box[0] + 0.25 * w)
    x1 = max(x0 + 1, int(math.ceil(box[2] - 0.25 * w)))
    y0 = int(box[1] + 0.25 * h)
    y1 = max(y0 + 1, int(math.ceil(box[3] - 0.25 * h)))
    try:
        crop = to_list(cam.rgb[y0:y1, x0:x1])
    except Exception:
        full = to_list(cam.rgb)
        crop = [row[x0:x1] for row in full[y0:y1]]
    n = 0
    hit = 0
    for row in crop:
        for px in row:
            n += 1
            if pix_has_colour(colour, px):
                hit += 1
    if n == 0:
        return None
    return hit / float(n)


def select_cube(cands, exp_xy, exp_src, info):
    # group duplicate detections of one object (largest cloud first)
    order = []
    for c in cands:
        at = len(order)
        while at > 0 and order[at - 1]["stats"]["n"] < c["stats"]["n"]:
            at -= 1
        order.insert(at, c)
    groups = []
    for c in order:
        home = None
        for g in groups:
            if dist2(g["rep"]["stats"]["c"], c["stats"]["c"]) < 0.02:
                home = g
                break
        if home is None:
            groups.append({"rep": c, "items": [c]})
        else:
            home["items"].append(c)
    cam = None
    if cube_color:
        try:
            cam = get_observation().cameras["front"]
        except Exception as exc:
            err("pick-front-rgb", exc)
    col_prompt = (cube_color + " cube") if cube_color else None
    rows = []
    for g in groups:
        g["col_score"] = max([0.0] + [it["score"] for it in g["items"] if it["prompt"] == col_prompt])
        g["frac"] = None
        if cam is not None:
            try:
                g["frac"] = colour_fraction(cam, g["rep"]["box"], cube_color)
            except Exception as exc:
                err("colour-fraction", exc)
        g["anchor_d"] = dist2(g["rep"]["stats"]["c"], exp_xy) if exp_xy is not None else None
        rows.append({"c": r3(g["rep"]["stats"]["c"]), "n": g["rep"]["stats"]["n"],
                     "prompts": sorted(set(it["prompt"] for it in g["items"])),
                     "col_score": round(g["col_score"], 4),
                     "frac": None if g["frac"] is None else round(g["frac"], 3),
                     "anchor_d": None if g["anchor_d"] is None else round(g["anchor_d"], 4)})
    info["cube_groups"] = rows
    if cube_color:
        pool = []
        for g in groups:
            if g["frac"] is not None:
                ok = g["frac"] >= 0.15
            else:
                ok = g["col_score"] > 0.0
            if ok:
                g["key"] = (g["frac"] if g["frac"] is not None else 0.0) + g["col_score"]
                pool.append(g)
        if pool:
            best = pool[0]
            for g in pool:
                if g["key"] > best["key"]:
                    best = g
            how = "goal colour '" + cube_color + "' (frac+colour-prompt score " + str(round(best["key"], 3)) + ")"
            if exp_xy is not None and best["anchor_d"] is not None and best["anchor_d"] > 0.08:
                alt = [g for g in pool if g["anchor_d"] is not None and g["anchor_d"] < 0.04
                       and g["key"] >= 0.5 * best["key"]]
                if alt:
                    best = alt[0]
                    for g in alt:
                        if g["anchor_d"] < best["anchor_d"]:
                            best = g
                    how += "; tie-break to " + exp_src
            info["anchor_check"] = None if best["anchor_d"] is None else round(best["anchor_d"], 4)
            info["chosen_by"] = how
            return best["rep"]
        note("WARN no candidate confirmed as " + cube_color + "; falling back to episode anchor")
        info["colour_unconfirmed"] = True
    if exp_xy is not None:
        best = None
        for c in cands:
            d = dist2(c["stats"]["c"], exp_xy)
            if best is None or d < best[0]:
                best = (d, c)
        info["chosen_by"] = "distance to " + exp_src + " d=" + str(round(best[0], 4))
        return best[1]
    best = None
    for c in cands:
        if best is None or c["score"] > best["score"]:
            best = c
    info["chosen_by"] = "score (no episode-derived anchor)"
    return best


pick_info = {}


def pick_cube():
    info = {}
    pick_info.clear()
    st = get_robot_state()
    g0 = float(st.gripper_positions["primary"])
    eep = fl(st.end_effector_poses["primary"].position)
    exp_xy, exp_src = expected_cube_xy()
    info["expected"] = [r3(exp_xy) if exp_xy else None, exp_src]
    # live front detection
    items = []
    for prompt in cube_prompts:
        items += [dict(it, stats=cloud_stats(it["pts"])) for it in seg_candidates("front", prompt, True, 6)]
    cands = cube_candidates(items)
    info["front_cands"] = [summarize_item(c) for c in cands]
    if g0 < 0.85 and g0 > 0.03:
        for c in cands:
            if dist2(c["stats"]["c"], eep) < 0.05 and c["stats"]["c"][2] > eep[2] - 0.06:
                info["already_holding"] = True
                pick_info.update(info)
                return True, info
    chosen_c = None
    if cands:
        chosen_c = select_cube(cands, exp_xy, exp_src, info)
    if chosen_c is not None:
        cxy = (chosen_c["stats"]["c"][0], chosen_c["stats"]["c"][1])
        top_z = chosen_c["stats"]["z98"]
    elif exp_xy is not None:
        cxy = (exp_xy[0], exp_xy[1])
        top_z = None
        info["chosen_by"] = "no detection; using " + exp_src
    else:
        info["fail"] = "no cube candidate"
        pick_info.update(info)
        return False, info
    # grasp height: an exact video event at this spot, else top - video-derived offset
    grasp_z = None
    if STATE.get("last_release") is not None and dist2(STATE["last_release"], cxy) < 0.03:
        grasp_z = STATE["last_release"][2] - 0.006
        info["grasp_z_src"] = "own last release"
    for ev in ([] if grasp_z is not None else events["picks"] + events["releases"]):
        if dist2(ev["p"], cxy) < 0.03:
            grasp_z = ev["p"][2]
            info["grasp_z_src"] = ev["kind"] + "@" + str(ev["frame"])
            break
    if grasp_z is None:
        off = 0.015
        if events["picks"] and table_z is not None:
            off = min(0.03, max(0.006, median([e["p"][2] for e in events["picks"]]) - table_z))
        if top_z is not None:
            grasp_z = top_z - off
            info["grasp_z_src"] = "top-offset " + str(round(off, 4))
        elif table_z is not None:
            grasp_z = table_z + off
            info["grasp_z_src"] = "table+offset"
        else:
            info["fail"] = "no grasp height"
            pick_info.update(info)
            return False, info
    if table_z is not None:
        grasp_z = max(grasp_z, table_z + 0.006)
    q = grasp_quat_default()
    seed = None
    for ev in events["picks"] + events["releases"]:
        if dist2(ev["p"], cxy) < 0.03:
            seed = ev.get("joints")
            break
    if g0 < 0.9:
        grip(True, "open-before-pick")
    hover_z = grasp_z + 0.13
    goto((cxy[0], cxy[1], hover_z), q, "pick-hover", tol=0.012, steps=200, seed=seed)
    # wrist refinement
    need_realign = False
    shift = 0.0
    try:
        wit = []
        for prompt in cube_prompts:
            wit += [dict(it, stats=cloud_stats(it["pts"])) for it in seg_candidates("wrist", prompt, True, 5)]
        wc = cube_candidates(wit)
        best = None
        for c in wc:
            d = dist2(c["stats"]["c"], cxy)
            if d < 0.045 and (best is None or d < best[0]):
                best = (d, c)
        info["wrist_cands"] = [summarize_item(c) for c in wc]
        if best is not None:
            fit = top_face_fit(best[1]["pts"], best[1]["stats"]["z98"])
            if fit is not None:
                info["wrist_fit"] = {"c": r3(fit["c"]), "yaw_deg": round(math.degrees(fit["yaw"]), 1),
                                     "n": fit["n"], "shift": round(dist2(fit["c"], cxy), 4)}
                if dist2(fit["c"], cxy) < 0.035:
                    shift = dist2(fit["c"], cxy)
                    cxy = fit["c"]
                q, dy = align_yaw(q, fit["yaw"])
                info["yaw_delta_deg"] = round(math.degrees(dy), 1)
                need_realign = abs(dy) > math.radians(5.0) or shift > 0.006
    except Exception as exc:
        err("wrist-refine", exc)
    info["grasp"] = r3((cxy[0], cxy[1], grasp_z))
    info["quat"] = r3(q)
    if need_realign:
        goto((cxy[0], cxy[1], hover_z), q, "pick-hover-aligned", tol=0.012, steps=200)
    if ticks_left() > 450:
        goto((cxy[0], cxy[1], grasp_z + 0.05), q, "pick-pre", tol=0.01, steps=200)
    goto((cxy[0], cxy[1], grasp_z), q, "pick-descend", tol=0.008, steps=200)
    gclosed = grip(False, "close")
    info["g_after_close"] = gclosed
    goto((cxy[0], cxy[1], grasp_z + 0.13), q, "pick-lift", tol=0.015, steps=200)
    g_lift = None
    try:
        g_lift = float(get_robot_state().gripper_positions["primary"])
    except Exception:
        pass
    info["g_after_lift"] = g_lift
    held = g_lift is not None and g_lift > 0.03
    info["held_guess"] = held
    pick_info.update(info)
    pick_info["q"] = q
    pick_info["grasp_z"] = grasp_z
    pick_info["support_z"] = table_z
    return held, info


def place_on(xy, tinfo, ev):
    info = {"target": tinfo}
    q = pick_info.get("q", grasp_quat_default())
    rel_z = None
    if ev is not None and ev["kind"] == "release":
        rel_z = ev["p"][2]
        info["rel_z_src"] = "video release @" + str(ev["frame"])
    elif events["releases"]:
        rel_z = median([e["p"][2] for e in events["releases"]])
        info["rel_z_src"] = "median video release"
    calc = None
    try:
        surf = tinfo.get("target_surface_z", table_z)
        if surf is not None and pick_info.get("grasp_z") is not None:
            sup = pick_info.get("support_z")
            if sup is None:
                sup = table_z
            calc = surf + (pick_info["grasp_z"] - sup)
    except Exception:
        calc = None
    if rel_z is None and calc is None:
        rel_z = get_robot_state().end_effector_poses["primary"].position[2] - 0.08
        info["rel_z_src"] = "fallback current-8cm"
    elif rel_z is None:
        rel_z = calc
        info["rel_z_src"] = "computed"
    elif calc is not None:
        rel_z = max(rel_z, calc)
    rel_z = float(rel_z) + 0.006
    info["release_z"] = round(rel_z, 4)
    cur = fl(get_robot_state().end_effector_poses["primary"].position)
    transit_z = max(cur[2], rel_z + 0.12)
    if cur[2] < transit_z - 0.02:
        goto((cur[0], cur[1], transit_z), q, "place-raise", tol=0.015, steps=150)
    goto((xy[0], xy[1], transit_z), q, "place-transit", tol=0.012, steps=250)
    if ticks_left() > 260:
        goto((xy[0], xy[1], rel_z + 0.05), q, "place-pre", tol=0.01, steps=200)
    goto((xy[0], xy[1], rel_z), q, "place-descend", tol=0.008, steps=200)
    grip(True, "release")
    STATE["last_release"] = (xy[0], xy[1], rel_z)
    info["settle_ticks"] = hold(6, "settle")
    goto((xy[0], xy[1], rel_z + 0.10), q, "place-retreat", tol=0.015, steps=150)
    info["post_ticks"] = hold(10, "post-retreat")
    return info


try:
    report["chosen"] = [{"frame": e["frame"], "p": r3(e["p"]), "track": e.get("track")} for e in chosen]
    report["pick_place"] = []
    if not chosen:
        note("no placement event could be chosen; nothing to place")
    for ci, ev in enumerate(chosen):
        if STATE["terminal"]:
            break
        if ticks_left() < 150:
            note("tick budget exhausted before placement " + str(ci))
            break
        xy, tinfo = live_target_xy(ev)
        note("target for placement " + str(ci) + ": " + str(tinfo))
        held, pinfo = pick_cube()
        rec = {"pick": dict((k, pinfo[k]) for k in pinfo if k not in ("q",)), "held": held}
        if not held and not STATE["terminal"]:
            note("grasp looks empty; one re-try")
            grip(True, "reopen")
            held, pinfo = pick_cube()
            rec["retry"] = dict((k, pinfo[k]) for k in pinfo if k not in ("q",))
            rec["held"] = held
        if not STATE["terminal"]:
            rec["place"] = place_on(xy, tinfo, ev)
        report["pick_place"].append(jsonable(rec))
except Exception as exc:
    err("pick-place", exc)

try:
    report["final"] = {"tick": tick_now(), "terminal": STATE["terminal"],
                       "ee": r3(get_robot_state().end_effector_poses["primary"].position),
                       "g": round(float(get_robot_state().gripper_positions["primary"]), 3)}
except Exception as exc:
    err("final", exc)
note("done; errors=" + str(len(report["errors"])))
result = report
