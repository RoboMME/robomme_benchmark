# PatternLock v01 -- slide at one drawing height (from v00) (demonstrations + public docs).
# Strategy:
#   1. Detect the button grid (white discs) in the conditioning video's first front frame and
#      in the live front frame (RGB-D -> robot_base), fit a square lattice (spacing, yaw, centre).
#   2. Read the touched-button order from the conditioning video: a touched button turns red;
#      project red wrist/front pixels to robot_base and map them to video lattice indices.
#   3. Map lattice indices to the live grid (yaw alignment nearest to the video's).
#   4. Tool offset (hand reference -> stick tip) is estimated from the video: at each touch the
#      tip is on the touched button top.  Live: Cartesian IK waypoints, press each button in
#      order, check the live image turns it red, press deeper if not.
import math

result = {"version": "v01"}
diag = {}
result["diag"] = diag

# ---------------------------------------------------------------- small math helpers
def quat_to_mat(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 0:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def mat_vec(m, v):
    return [m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]]


def mat_t_vec(m, v):
    return [m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
            m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
            m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2]]


def median(vals):
    s = sorted(vals)
    if len(s) == 0:
        return None
    k = len(s) // 2
    if len(s) % 2 == 1:
        return s[k]
    return 0.5 * (s[k - 1] + s[k])


def quantile(vals, q):
    s = sorted(vals)
    if len(s) == 0:
        return None
    i = int(round(q * (len(s) - 1)))
    return s[max(0, min(len(s) - 1, i))]


def r3(v):
    return [round(float(c), 4) for c in v]


def as_list(arr):
    try:
        return arr.tolist()
    except Exception:
        return [list(row) for row in arr]


# ---------------------------------------------------------------- camera projection
def make_view(cam, conv):
    k = as_list(cam.intrinsics)
    pose = cam.camera_pose
    rgb = as_list(cam.rgb)
    return {"rgb": rgb, "depth": as_list(cam.depth_m),
            "fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]), "cy": float(k[1][2]),
            "rot": quat_to_mat(pose.quaternion_wxyz), "t": [float(c) for c in pose.position],
            "conv": conv, "h": len(rgb), "w": len(rgb[0])}


def project(view, u, v):
    d = float(view["depth"][v][u])
    if not (d > 0.02 and d < 6.0):
        return None
    x = (u - view["cx"]) / view["fx"] * d
    y = (v - view["cy"]) / view["fy"] * d
    if view["conv"] == "cv":
        pc = [x, y, d]
    elif view["conv"] == "gl":
        pc = [x, -y, -d]
    else:
        pc = [d, -x, -y]
    p = mat_vec(view["rot"], pc)
    t = view["t"]
    return [p[0] + t[0], p[1] + t[1], p[2] + t[2]]


def is_white(px):
    r, g, b = px[0], px[1], px[2]
    mn = min(r, g, b)
    mx = max(r, g, b)
    return mn >= 150 and mx - mn <= 55


def is_red(px):
    r, g, b = px[0], px[1], px[2]
    return r >= 110 and g * 10 <= r * 3 and b * 10 <= r * 4


def components(view, pred, stride, box=None):
    """8-connected components (on the stride lattice) of pixels satisfying pred."""
    u0, v0, u1, v1 = 0, 0, view["w"] - 1, view["h"] - 1
    if box is not None:
        u0 = max(0, box[0]); v0 = max(0, box[1]); u1 = min(view["w"] - 1, box[2]); v1 = min(view["h"] - 1, box[3])
    mark = {}
    for v in range(v0, v1 + 1, stride):
        row = view["rgb"][v]
        for u in range(u0, u1 + 1, stride):
            if pred(row[u]):
                mark[(u, v)] = 1
    comps = []
    seen = {}
    for key in mark:
        if key in seen:
            continue
        stack = [key]
        seen[key] = 1
        pix = []
        while len(stack) > 0:
            cur = stack.pop()
            pix.append(cur)
            for du in (-stride, 0, stride):
                for dv in (-stride, 0, stride):
                    nb = (cur[0] + du, cur[1] + dv)
                    if nb in mark and nb not in seen:
                        seen[nb] = 1
                        stack.append(nb)
        comps.append(pix)
    return comps


def comp_points(view, pix):
    pts = []
    for p in pix:
        q = project(view, p[0], p[1])
        if q is not None:
            pts.append(q)
    return pts


def summarize(pts, pix):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
    us = [p[0] for p in pix]; vs = [p[1] for p in pix]
    return {"x": median(xs), "y": median(ys), "ztop": quantile(zs, 0.85), "zmed": median(zs),
            "n": len(pts), "u": sum(us) / len(us), "v": sum(vs) / len(vs),
            "box": [min(us), min(vs), max(us), max(vs)]}


# ---------------------------------------------------------------- grid detection + lattice
def detect_dots(view):
    comps = components(view, is_white, 1)
    blobs = []
    for pix in comps:
        if len(pix) < 3 or len(pix) > 1500:
            continue
        pts = comp_points(view, pix)
        if len(pts) < 3:
            continue
        blobs.append(summarize(pts, pix))
    # merge blobs closer than 1.5 cm in xy (rings / side faces of one button)
    merged = []
    for b in sorted([(-bb["n"], i) for i, bb in enumerate(blobs)]):
        bb = blobs[b[1]]
        hit = None
        for m in merged:
            if math.hypot(m["x"] - bb["x"], m["y"] - bb["y"]) < 0.015:
                hit = m
                break
        if hit is None:
            merged.append(dict(bb))
        else:
            tot = hit["n"] + bb["n"]
            hit["x"] = (hit["x"] * hit["n"] + bb["x"] * bb["n"]) / tot
            hit["y"] = (hit["y"] * hit["n"] + bb["y"] * bb["n"]) / tot
            hit["ztop"] = max(hit["ztop"], bb["ztop"])
            hit["box"] = [min(hit["box"][0], bb["box"][0]), min(hit["box"][1], bb["box"][1]),
                          max(hit["box"][2], bb["box"][2]), max(hit["box"][3], bb["box"][3])]
            hit["n"] = tot
    if len(merged) == 0:
        return [], {"raw": len(blobs)}
    zmed = median([m["ztop"] for m in merged if m["n"] >= 4] or [m["ztop"] for m in merged])
    dots = [m for m in merged if abs(m["ztop"] - zmed) < 0.02 and m["n"] >= 3]
    return dots, {"raw": len(blobs), "merged": len(merged), "zmed": zmed}


def fit_lattice(dots, theta_ref=None):
    if len(dots) < 4:
        return None
    nn = []
    for i, a in enumerate(dots):
        best = None
        for j, b in enumerate(dots):
            if i == j:
                continue
            d = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
            if best is None or d < best:
                best = d
        nn.append(best)
    s = median(nn)
    sn = 0.0; cs = 0.0
    for i, a in enumerate(dots):
        for j, b in enumerate(dots):
            if i == j:
                continue
            dx = b["x"] - a["x"]; dy = b["y"] - a["y"]
            d = math.hypot(dx, dy)
            if 0.75 * s < d < 1.3 * s:
                ang = math.atan2(dy, dx)
                sn += math.sin(4 * ang); cs += math.cos(4 * ang)
    th = math.atan2(sn, cs) / 4.0
    if theta_ref is not None:
        best = None
        for k in range(-4, 5):
            cand = th + k * math.pi / 2
            diff = abs(math.atan2(math.sin(cand - theta_ref), math.cos(cand - theta_ref)))
            if best is None or diff < best[0]:
                best = (diff, cand)
        th = best[1]
    e1 = (math.cos(th), math.sin(th)); e2 = (-math.sin(th), math.cos(th))
    mx = sum(d["x"] for d in dots) / len(dots); my = sum(d["y"] for d in dots) / len(dots)
    org = None
    for d in dots:
        dd = math.hypot(d["x"] - mx, d["y"] - my)
        if org is None or dd < org[0]:
            org = (dd, d)
    ox, oy = org[1]["x"], org[1]["y"]
    idx = []
    for d in dots:
        a = ((d["x"] - ox) * e1[0] + (d["y"] - oy) * e1[1]) / s
        b = ((d["x"] - ox) * e2[0] + (d["y"] - oy) * e2[1]) / s
        ia = int(round(a)); ib = int(round(b))
        if math.hypot(a - ia, b - ib) < 0.3:
            idx.append((ia, ib, d))
    if len(idx) < 4:
        return None
    # refine origin
    ox = sum(d["x"] - ia * s * e1[0] - ib * s * e2[0] for ia, ib, d in idx) / len(idx)
    oy = sum(d["y"] - ia * s * e1[1] - ib * s * e2[1] for ia, ib, d in idx) / len(idx)
    amin = min(t[0] for t in idx); amax = max(t[0] for t in idx)
    bmin = min(t[1] for t in idx); bmax = max(t[1] for t in idx)
    ca = (amin + amax) / 2.0; cb = (bmin + bmax) / 2.0
    cx = ox + ca * s * e1[0] + cb * s * e2[0]
    cy = oy + ca * s * e1[1] + cb * s * e2[1]
    cells = {}
    for ia, ib, d in idx:
        cells[(ia - ca, ib - cb)] = d
    return {"s": s, "theta": th, "cx": cx, "cy": cy, "e1": e1, "e2": e2, "cells": cells,
            "span": [amax - amin + 1, bmax - bmin + 1], "z": median([d["ztop"] for d in dots]),
            "n": len(idx)}


def lattice_index(lat, x, y):
    a = ((x - lat["cx"]) * lat["e1"][0] + (y - lat["cy"]) * lat["e1"][1]) / lat["s"]
    b = ((x - lat["cx"]) * lat["e2"][0] + (y - lat["cy"]) * lat["e2"][1]) / lat["s"]
    best = None
    for key in lat["cells"]:
        d = math.hypot(a - key[0], b - key[1])
        if best is None or d < best[0]:
            best = (d, key)
    return best


def lattice_point(lat, key):
    if key in lat["cells"]:
        d = lat["cells"][key]
        return [d["x"], d["y"], d["ztop"]]
    x = lat["cx"] + key[0] * lat["s"] * lat["e1"][0] + key[1] * lat["s"] * lat["e2"][0]
    y = lat["cy"] + key[0] * lat["s"] * lat["e1"][1] + key[1] * lat["s"] * lat["e2"][1]
    return [x, y, lat["z"]]


def red_cells(view, lat, stride, box=None):
    comps = components(view, is_red, stride, box)
    found = []
    for pix in comps:
        if len(pix) < 2:
            continue
        pts = comp_points(view, pix)
        if len(pts) < 2:
            continue
        sm = summarize(pts, pix)
        li = lattice_index(lat, sm["x"], sm["y"])
        if li is not None and li[0] < 0.45:
            found.append((li[1], len(pix), round(li[0], 3)))
    return found


# ---------------------------------------------------------------- read inputs
ctx = get_task_context()
result["language"] = str(ctx.language)
meta = {}
try:
    md = robomme_plus.get_task_metadata()
    for k in md:
        try:
            meta[str(k)] = md[k] if k != "conditioning_video_sampling" else dict(md[k])
        except Exception:
            meta[str(k)] = str(md[k])
except Exception as exc:
    meta["error"] = str(exc)
result["task_metadata"] = {k: (meta[k] if k in ("max_steps", "control_hz", "language", "task") else str(meta[k])[:300]) for k in meta}
max_steps = 1000
try:
    max_steps = int(meta["max_steps"])
except Exception:
    pass
try:
    cm = robomme_plus.get_controller_metadata()
    result["controller_metadata"] = {str(k): str(cm[k])[:200] for k in cm}
except Exception as exc:
    result["controller_metadata"] = str(exc)

video = robomme_plus.get_conditioning_video()
nvid = len(video)
result["video_frames"] = nvid
obs0 = get_observation()
state0 = obs0.robot_state
emb = state0.embodiment
base = state0.base_frame
result["live_start"] = {"joints": r3(state0.joint_positions["primary"]),
                        "ee": r3(state0.end_effector_poses["primary"].position),
                        "ee_q": r3(state0.end_effector_poses["primary"].quaternion_wxyz),
                        "t": float(state0.timestamp_s)}

# ---------------------------------------------------------------- camera convention (table flatness)
conv_scores = {}
best_conv = None
for conv in ("cv", "gl", "ros"):
    vw = make_view(obs0.cameras["front"], conv)
    zs = []
    for v in range(int(vw["h"] * 0.45), vw["h"], 12):
        for u in range(8, vw["w"] - 8, 12):
            p = project(vw, u, v)
            if p is not None:
                zs.append(p[2])
    if len(zs) < 10:
        continue
    spread = quantile(zs, 0.9) - quantile(zs, 0.1)
    # the table must lie below the camera; a mirrored convention puts it above
    if median(zs) > vw["t"][2] - 0.05:
        spread += 10.0
    conv_scores[conv] = [round(spread, 4), round(median(zs), 4)]
    if best_conv is None or spread < best_conv[0]:
        best_conv = (spread, conv)
conv = best_conv[1]
result["camera_convention"] = {"chosen": conv, "scores": conv_scores}

# ---------------------------------------------------------------- live grid
live_front = make_view(obs0.cameras["front"], conv)
live_dots, live_dd = detect_dots(live_front)
diag["live_detect"] = live_dd

# SAM cross-check of the projection on one live dot (diagnostic only)
try:
    if len(live_dots) > 0:
        dd0 = live_dots[0]
        seg = segment_points("front", [[int(round(dd0["u"])), int(round(dd0["v"]))]])
        if seg.ok and len(seg.segmentations) > 0:
            pc = mask_to_point_cloud(seg.segmentations[0], "front", target_frame=base)
            pp = pc.points
            n = len(pp)
            if n > 0:
                sx = sum(float(p[0]) for p in pp) / n
                sy = sum(float(p[1]) for p in pp) / n
                sz = sum(float(p[2]) for p in pp) / n
                diag["sam_check"] = {"mine": r3([dd0["x"], dd0["y"], dd0["ztop"]]), "sam": r3([sx, sy, sz]), "n": n}
except Exception as exc:
    diag["sam_check_error"] = str(exc)[:200]

# ---------------------------------------------------------------- video grid + touch sequence
vid_lat = None
seq = []
events = []
ee_traj = []
if nvid > 0:
    vf0 = make_view(video[0].cameras["front"], conv)
    vid_dots, vid_dd = detect_dots(vf0)
    diag["video_detect"] = vid_dd
    vid_lat = fit_lattice(vid_dots)
if vid_lat is not None:
    diag["video_lattice"] = {"s": round(vid_lat["s"], 4), "theta_deg": round(math.degrees(vid_lat["theta"]), 2),
                             "c": r3([vid_lat["cx"], vid_lat["cy"], vid_lat["z"]]), "n": vid_lat["n"], "span": vid_lat["span"]}
    us = [d["box"][0] for d in vid_lat["cells"].values()] + [d["box"][2] for d in vid_lat["cells"].values()]
    vs = [d["box"][1] for d in vid_lat["cells"].values()] + [d["box"][3] for d in vid_lat["cells"].values()]
    fbox = [int(min(us)) - 8, int(min(vs)) - 8, int(max(us)) + 8, int(max(vs)) + 8]
    prev = set()
    pending = {}
    per_frame = []
    for fi in range(nvid):
        fr = video[fi]
        st = fr.robot_state
        eep = st.end_effector_poses["primary"]
        cur = set()
        info = []
        try:
            wv = make_view(fr.cameras["wrist"], conv)
            for c in red_cells(wv, vid_lat, 2):
                if c[1] >= 6:
                    cur.add(c[0]); info.append(("w",) + c)
        except Exception as exc:
            info.append(("werr", str(exc)[:80]))
        try:
            fv = make_view(fr.cameras["front"], conv)
            for c in red_cells(fv, vid_lat, 1, fbox):
                if c[1] >= 2:
                    cur.add(c[0]); info.append(("f",) + c)
        except Exception as exc:
            info.append(("ferr", str(exc)[:80]))
        if fi % 4 == 0:
            ee_traj.append([round(float(fr.timestamp_s), 2)] + r3(eep.position))
        if len(cur) > 0:
            per_frame.append([fi, sorted([list(k) for k in cur])])
        # confirm a cell when it is red in two consecutive analysed frames
        for key in cur:
            if key in prev and (len(seq) == 0 or seq[-1] != key) and key not in pending:
                pending[key] = 1
                seq.append(key)
                events.append({"frame": fi, "t": round(float(fr.timestamp_s), 2), "cell": list(key),
                               "ee": [float(c) for c in eep.position], "ee_q": [float(c) for c in eep.quaternion_wxyz]})
        for key in list(pending.keys()):
            if key not in cur:
                pending.pop(key)
        prev = cur
    diag["red_frames"] = per_frame[:400]
result["video_sequence"] = [list(k) for k in seq]
result["video_ee_traj"] = ee_traj

# ---------------------------------------------------------------- tool offset from video touches
offsets = []
for ev in events:
    tip = lattice_point(vid_lat, tuple(ev["cell"]))
    rot = quat_to_mat(ev["ee_q"])
    off = mat_t_vec(rot, [tip[0] - ev["ee"][0], tip[1] - ev["ee"][1], tip[2] - ev["ee"][2]])
    offsets.append(off)
if len(offsets) > 0:
    tool_off = [median([o[i] for o in offsets]) for i in range(3)]
    touch_q = events[0]["ee_q"]
else:
    tool_off = [0.0, 0.0, 0.0]
    touch_q = [float(c) for c in state0.end_effector_poses["primary"].quaternion_wxyz]
result["tool_offset_local"] = r3(tool_off)
result["tool_offset_all"] = [r3(o) for o in offsets]
result["touch_q"] = r3(touch_q)
result["events"] = [{"frame": e["frame"], "t": e["t"], "cell": e["cell"], "ee": r3(e["ee"])} for e in events]

live_lat = None
if len(live_dots) >= 4:
    live_lat = fit_lattice(live_dots, vid_lat["theta"] if vid_lat is not None else None)
if live_lat is not None:
    diag["live_lattice"] = {"s": round(live_lat["s"], 4), "theta_deg": round(math.degrees(live_lat["theta"]), 2),
                            "c": r3([live_lat["cx"], live_lat["cy"], live_lat["z"]]), "n": live_lat["n"], "span": live_lat["span"]}

# ---------------------------------------------------------------- live execution helpers
touch_rot = quat_to_mat(touch_q)
world_off = mat_vec(touch_rot, tool_off)
ticks = [0]
status = {"terminated": False, "truncated": False}


def ee_for_tip(tip):
    return [tip[0] - world_off[0], tip[1] - world_off[1], tip[2] - world_off[2]]


def check_end(res):
    try:
        if bool(res.terminated):
            status["terminated"] = True
        if bool(res.truncated):
            status["truncated"] = True
    except Exception:
        pass
    return status["terminated"] or status["truncated"]


last_joints = [list(state0.joint_positions["primary"])]


def step_joints(joints, n=1):
    for k in range(n):
        if ticks[0] >= max_steps - 2:
            return True
        st = get_robot_state()
        res = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=list(joints),
            gripper_position=st.gripper_positions["primary"], embodiment=emb)}))
        ticks[0] += 1
        if check_end(res):
            return True
    return False


ik_fail = [0]


def goto_tip(tip, n=1):
    ee = ee_for_tip(tip)
    ik = solve_ik(Pose(ee, touch_q, base), seed_joints=last_joints[0])
    if not ik.ok:
        ik_fail[0] += 1
        return False
    last_joints[0] = list(ik.joint_positions)
    return step_joints(last_joints[0], n)


def line_tip(a, b, step_m, n_end=1):
    d = math.sqrt(sum((b[i] - a[i]) ** 2 for i in range(3)))
    k = max(1, int(math.ceil(d / step_m)))
    for j in range(1, k + 1):
        f = j / float(k)
        p = [a[i] + f * (b[i] - a[i]) for i in range(3)]
        if goto_tip(p, 1 if j < k else n_end):
            return True
    return False


def tip_now():
    st = get_robot_state()
    pose = st.end_effector_poses["primary"]
    rot = quat_to_mat(pose.quaternion_wxyz)
    w = mat_vec(rot, tool_off)
    return [float(pose.position[0]) + w[0], float(pose.position[1]) + w[1], float(pose.position[2]) + w[2]]


def live_red():
    try:
        ob = get_observation()
        cells = set()
        wv = make_view(ob.cameras["wrist"], conv)
        for c in red_cells(wv, live_lat, 2):
            if c[1] >= 6:
                cells.add(c[0])
        return cells
    except Exception as exc:
        return set()


# ---------------------------------------------------------------- execute
log = []
result["log"] = log
# v01: the video keeps the hand at ONE drawing height for the whole pattern and dwells ~8-12
# ticks on each button (red appears ~6 ticks after arrival).  Lifting between buttons ended
# v00's episode.  So: descend once onto the first button, then slide at the drawing height.
HOVER = 0.05
DWELL = 10          # ticks on each button before moving on
draw_dz = [0.0]     # tip height relative to the observed button top (video: tip on top)
try:
    if live_lat is None or vid_lat is None or len(seq) == 0:
        raise ValueError("missing grid or sequence: live=%s video=%s seq=%d" % (live_lat is not None, vid_lat is not None, len(seq)))
    targets = [lattice_point(live_lat, k) for k in seq]
    result["live_targets"] = [r3(t) for t in targets]
    first = targets[0]
    hover = [first[0], first[1], first[2] + HOVER]
    mv = move_to_pose(Pose(ee_for_tip(hover), touch_q, base), tolerance=0.01, max_steps=200)
    try:
        ticks[0] += int(mv.steps_executed)
    except Exception:
        pass
    check_end(mv)
    log.append({"hover_ok": bool(mv.ok), "ticks": ticks[0]})
    last_joints[0] = list(get_robot_state().joint_positions["primary"])
    ended = status["terminated"] or status["truncated"]
    cur = hover
    for i, tgt in enumerate(targets):
        if ended:
            break
        dest = [tgt[0], tgt[1], tgt[2] + draw_dz[0]]
        ended = line_tip(cur, dest, 0.004 if i > 0 else 0.003, DWELL)
        if ended:
            log.append({"i": i, "cell": list(seq[i]), "ended_on_arrival": True, "ticks": ticks[0]})
            break
        red = live_red()
        extra = 0
        # first button only: if it does not light, the live tool reaches less far -> lower
        while i == 0 and (not ended) and (seq[i] not in red) and extra < 3:
            extra += 1
            draw_dz[0] -= 0.003
            dest = [tgt[0], tgt[1], tgt[2] + draw_dz[0]]
            ended = goto_tip(dest, 6)
            red = live_red() if not ended else set()
        tn = tip_now() if not ended else None
        log.append({"i": i, "cell": list(seq[i]), "red": sorted([list(k) for k in red]), "extra": extra,
                    "tip": r3(tn) if tn is not None else None, "tgt": r3(tgt), "ticks": ticks[0]})
        cur = dest
    hold = 0
    while (not ended) and hold < 20:
        hold += 1
        ended = step_joints(last_joints[0], 1)
    result["hold_ticks"] = hold
    result["ended_by_env"] = ended
except Exception as exc:
    result["error"] = str(exc)[:500]
result["ticks"] = ticks[0]
result["ik_fail"] = ik_fail[0]
result["status"] = status
