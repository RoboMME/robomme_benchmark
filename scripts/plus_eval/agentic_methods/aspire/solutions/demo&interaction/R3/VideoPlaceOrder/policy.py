import math

# VideoPlaceOrder v00 -- watch the conditioning video, track target identities through
# pairwise swaps, find the Nth target the cube was placed on, then pick-and-place.
# Every task-dependent quantity (colour, ordinal, slots, swaps, placements, 3D poses)
# is derived at runtime from the goal text, the conditioning video and live observations.

report = {"version": "v00", "stage": "start"}
result = report

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
            "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6, "7th": 7, "8th": 8}
COLOURS = ("red", "green", "blue", "yellow")


def is_purple(r, g, b):
    # saturated magenta/purple ring; gated on chroma so tinted lighting and the white arm do not pass
    if b - g <= 40 or r <= 90 or b <= 90:
        return False
    lo = g if g < r else r
    hi = b if b > r else r
    return hi - lo > 60


def is_colour(name, r, g, b):
    if name == "red":
        return r > 140 and g < 70 and b < 70
    if name == "green":
        return g > 140 and r < 90 and b < 90
    if name == "blue":
        return b > 120 and r < 50 and g < 50
    if name == "yellow":
        return r > 160 and g > 160 and b < 80
    return False


def rgb_rows(camera):
    rgb = camera.rgb
    try:
        return rgb.tolist()
    except Exception:
        return [[[int(v) for v in px] for px in row] for row in rgb]


def analyse_image(rows, colour, step):
    """Return (target blobs, cube stats) from one RGB image (nested lists)."""
    h = len(rows)
    w = len(rows[0])
    gh = (h + step - 1) // step
    gw = (w + step - 1) // step
    grid = [[0] * gw for gy in range(gh)]
    cx = 0.0
    cy = 0.0
    cn = 0
    cymax = -1
    for gy in range(gh):
        row = rows[gy * step]
        grow = grid[gy]
        for gx in range(gw):
            px = row[gx * step]
            r = px[0]
            g = px[1]
            b = px[2]
            if is_purple(r, g, b):
                grow[gx] = 1
            elif is_colour(colour, r, g, b):
                cx += gx * step
                cy += gy * step
                cn += 1
                if gy * step > cymax:
                    cymax = gy * step
    # dilate purple by one cell so ring pieces join, then label 8-connected components
    dil = [[0] * gw for gy in range(gh)]
    for gy in range(gh):
        for gx in range(gw):
            if grid[gy][gx]:
                for dy in (-1, 0, 1):
                    yy = gy + dy
                    if yy < 0 or yy >= gh:
                        continue
                    drow = dil[yy]
                    for dx in (-1, 0, 1):
                        xx = gx + dx
                        if 0 <= xx < gw:
                            drow[xx] = 1
    label = [[0] * gw for gy in range(gh)]
    blobs = []
    nlab = 0
    for gy in range(gh):
        for gx in range(gw):
            if dil[gy][gx] and not label[gy][gx]:
                nlab += 1
                stack = [(gy, gx)]
                label[gy][gx] = nlab
                sx = 0.0
                sy = 0.0
                n = 0
                x0 = gx
                x1 = gx
                y0 = gy
                y1 = gy
                while stack:
                    cyy, cxx = stack.pop()
                    if grid[cyy][cxx]:
                        sx += cxx
                        sy += cyy
                        n += 1
                        if cxx < x0:
                            x0 = cxx
                        if cxx > x1:
                            x1 = cxx
                        if cyy < y0:
                            y0 = cyy
                        if cyy > y1:
                            y1 = cyy
                    for dy in (-1, 0, 1):
                        yy = cyy + dy
                        if yy < 0 or yy >= gh:
                            continue
                        for dx in (-1, 0, 1):
                            xx = cxx + dx
                            if 0 <= xx < gw and dil[yy][xx] and not label[yy][xx]:
                                label[yy][xx] = nlab
                                stack.append((yy, xx))
                if n > 0:
                    blobs.append({"x": sx / n * step, "y": sy / n * step, "n": n * step * step,
                                  "w": (x1 - x0 + 1) * step, "h": (y1 - y0 + 1) * step})
    cube = None
    if cn > 0:
        cube = {"x": cx / cn, "y": cy / cn, "n": cn * step * step, "ymax": cymax}
    return blobs, cube


def dist(ax, ay, bx, by):
    return math.sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by))


def nearest_slot(slots, x, y):
    best = None
    bd = None
    for i, s in enumerate(slots):
        d = dist(s["x"], s["y"], x, y)
        if bd is None or d < bd:
            bd = d
            best = i
    return best, bd


def analyse_video(per):
    """per: list of (timestamp_s, blobs, cube) per video frame. Returns memory analysis dict."""
    out = {"frames": len(per)}
    if not per:
        return out
    # --- slots: target rest positions from the first frame, minimum size relative to the largest
    first = per[0][1]
    big = max([b["n"] for b in first]) if first else 0
    slots = [b for b in first if b["n"] >= 0.15 * big]
    for s in slots:
        s["r"] = 0.5 * max(s["w"], s["h"])
    out["slots"] = [[round(s["x"], 1), round(s["y"], 1), round(s["r"], 1)] for s in slots]
    if len(slots) < 2:
        out["error"] = "fewer than two target slots in first video frame"
        return out
    # --- cube reference size from frames where it is visible
    # reference = the cube's pixel area in the first frame (resting on the table, unoccluded)
    sizes = sorted([c["n"] for ts, b, c in per if c is not None])
    ref = per[0][2]["n"] if per[0][2] is not None else (sizes[len(sizes) // 2] if sizes else 0)
    out["cube_ref_px"] = ref
    # --- placements: cube at rest, fully visible, with its contact point inside a slot
    rests = []
    cur = None
    for k, item in enumerate(per):
        ts, blobs, c = item
        if c is None or c["n"] < 0.5 * ref:
            cur = None
            continue
        if cur is not None and dist(c["x"], c["y"], cur["x"], cur["y"]) < 2.5:
            cur["t1"] = ts
            cur["k1"] = k
        else:
            cur = {"x": c["x"], "y": c["y"], "yb": c["ymax"], "t0": ts, "t1": ts, "k0": k, "k1": k}
            rests.append(cur)
    places = []
    for rr in rests:
        if rr["t1"] - rr["t0"] < 0.6:
            continue
        px = rr["x"]
        py = 0.5 * (rr["y"] + rr["yb"])
        si, sd = nearest_slot(slots, px, py)
        if sd > 0.8 * slots[si]["r"]:
            continue
        places.append({"slot": si, "t0": rr["t0"], "t1": rr["t1"], "k0": rr["k0"], "k1": rr["k1"],
                       "cube": [round(rr["x"], 1), round(rr["y"], 1)]})
    # merge consecutive rests on the same slot if the cube was never seen far away in between
    merged = []
    for p in places:
        if merged and merged[-1]["slot"] == p["slot"]:
            moved = False
            s = slots[p["slot"]]
            for k in range(merged[-1]["k1"], p["k0"]):
                c = per[k][2]
                if c is not None and c["n"] >= 0.3 * ref and dist(c["x"], c["y"], s["x"], s["y"]) > 3.0 * s["r"]:
                    moved = True
                    break
            if not moved:
                merged[-1]["t1"] = p["t1"]
                merged[-1]["k1"] = p["k1"]
                continue
        merged.append(p)
    places = merged
    # --- swap evidence: two off-slot blobs whose midpoint equals a slot pair's midpoint
    votes = []
    settled = []
    offs = []
    for k, item in enumerate(per):
        ts, blobs, c = item
        off = []
        off_near = []
        covered = set()
        for b in blobs:
            si, sd = nearest_slot(slots, b["x"], b["y"])
            if sd > 0.6 * slots[si]["r"] and b["n"] >= 0.1 * big:
                off.append(b)
                off_near.append(si)
            elif sd <= 0.6 * slots[si]["r"]:
                covered.add(si)
        if len(off) == 0 and len(covered) == len(slots):
            settled.append(ts)
        if off:
            offs.append((ts, off_near))
        if len(off) < 2:
            continue
        for i in range(len(off)):
            for j in range(i + 1, len(off)):
                p = off[i]
                q = off[j]
                mx = p["x"] + q["x"]
                my = p["y"] + q["y"]
                pq = dist(p["x"], p["y"], q["x"], q["y"])
                best = None
                bd = None
                second = None
                for a in range(len(slots)):
                    for bb in range(a + 1, len(slots)):
                        A = slots[a]
                        B = slots[bb]
                        e = dist(mx, my, A["x"] + B["x"], A["y"] + B["y"])
                        ab = dist(A["x"], A["y"], B["x"], B["y"])
                        if pq < 0.35 * ab or pq > 5.0 * ab:
                            continue
                        tol = 0.6 * (A["r"] + B["r"])
                        if e < tol:
                            if bd is None or e < bd:
                                second = bd
                                bd = e
                                best = (a, bb)
                            elif second is None or e < second:
                                second = e
                if best is not None:
                    votes.append((ts, best[0], best[1], k))
    # cluster votes in time into swap episodes
    votes = sorted(votes)
    episodes = []
    si3 = 0
    for v in votes:
        # a settled frame (every visible target on a slot) between two votes ends an episode
        split = False
        if episodes:
            while si3 < len(settled) and settled[si3] <= episodes[-1]["t1"]:
                si3 += 1
            if si3 < len(settled) and settled[si3] < v[0]:
                split = True
        if episodes and not split and v[0] - episodes[-1]["t1"] <= 1.0:
            episodes[-1]["t1"] = v[0]
            episodes[-1]["votes"].append(v)
        else:
            episodes.append({"t0": v[0], "t1": v[0], "votes": [v]})
    swaps = []
    for ep in episodes:
        tally = {}
        for v in ep["votes"]:
            key = (v[1], v[2])
            tally[key] = tally.get(key, 0) + 1
        ranked = sorted([(-n, key) for key, n in tally.items()])
        # slots that targets leave from / arrive at: nearest slots of the first and last
        # off-slot detections around this episode
        window = [o for o in offs if ep["t0"] - 0.6 <= o[0] <= ep["t1"] + 0.6]
        moving = set()
        for o in window[:2] + window[-2:]:
            for si in o[1]:
                moving.add(si)
        restricted = [rk for rk in ranked if rk[1][0] in moving and rk[1][1] in moving and -rk[0] >= 2]
        if restricted:
            ranked = restricted
        ep["moving"] = sorted(moving)
        used = set()
        chosen = []
        for negn, key in ranked:
            if -negn < 2:
                continue
            if key[0] in used or key[1] in used:
                continue
            used.add(key[0])
            used.add(key[1])
            chosen.append([key[0], key[1], -negn])
        if chosen:
            swaps.append({"t0": ep["t0"], "t1": ep["t1"], "pairs": chosen, "moving": ep["moving"],
                          "tally": [[key[0], key[1], -negn] for negn, key in ranked[:6]]})
    # --- replay events in time order: occupant[slot] = target identity
    occupant = list(range(len(slots)))
    keyed = []
    for pi, p in enumerate(places):
        keyed.append((p["t0"], 1, pi))
    for si2, s in enumerate(swaps):
        keyed.append((0.5 * (s["t0"] + s["t1"]), 0, si2))
    keyed = sorted(keyed)
    seq = []
    log = []
    for t, order, ix in keyed:
        kind = "swap" if order == 0 else "place"
        data = swaps[ix] if order == 0 else places[ix]
        if kind == "swap":
            for pair in data["pairs"]:
                a = pair[0]
                b = pair[1]
                tmp = occupant[a]
                occupant[a] = occupant[b]
                occupant[b] = tmp
            log.append([round(t, 2), "swap", [[pp[0], pp[1]] for pp in data["pairs"]]])
        else:
            ident = occupant[data["slot"]]
            seq.append(ident)
            log.append([round(t, 2), "place", data["slot"], ident])
    final_slot_of = {}
    for si, ident in enumerate(occupant):
        final_slot_of[ident] = si
    timeline = []
    for k in range(0, len(per), 4):
        ts, blobs, c = per[k]
        timeline.append([round(ts, 2), len(blobs), None if c is None else [round(c["x"]), round(c["y"]), c["n"]]])
    out["timeline"] = timeline
    out["placements"] = [[round(p["t0"], 2), round(p["t1"], 2), p["slot"], p["cube"]] for p in places]
    out["swaps"] = [[round(s["t0"], 2), round(s["t1"], 2), s["pairs"], s["moving"], s["tally"]] for s in swaps]
    out["events"] = log
    out["identity_sequence"] = seq
    out["final_occupant"] = occupant
    out["final_slot_of_identity"] = [final_slot_of[i] for i in range(len(slots))]
    out["slots_full"] = slots
    return out


def parse_goal(goal):
    text = goal.lower().replace(",", " ").replace(".", " ")
    words = text.split()
    colour = None
    for c in COLOURS:
        if c in words:
            colour = c
            break
    ordinal = None
    for wd in words:
        if wd in ORDINALS:
            ordinal = ORDINALS[wd]
            break
    return colour, ordinal


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def rotate_vec(q, v):
    qv = [0.0, v[0], v[1], v[2]]
    qc = [q[0], -q[1], -q[2], -q[3]]
    r = quat_mul(quat_mul(q, qv), qc)
    return [r[1], r[2], r[3]]


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def cloud_from_pixel(u, v):
    """Segment the object under image pixel (u, v) in the front camera and project it."""
    info = {"pixel": [round(u, 1), round(v, 1)]}
    try:
        seg = segment_points("front", [[int(round(u)), int(round(v))]])
        info["seg_ok"] = bool(seg.ok)
        if seg.ok and len(seg.segmentations) > 0:
            cand = seg.segmentations[0]
            try:
                info["box"] = [round(float(z), 1) for z in cand.box_xyxy]
            except Exception:
                pass
            cloud = mask_to_point_cloud(cand, "front", target_frame=base_frame)
            pts = [[float(p[0]), float(p[1]), float(p[2])] for p in cloud.points]
            info["points"] = len(pts)
            return pts, info
    except Exception as exc:
        info["error"] = str(exc)[:200]
    return [], info


def cloud_from_text(prompts, u, v):
    """Fallback: text segmentation, candidate whose box contains / is nearest to (u, v)."""
    info = {}
    for prompt in prompts:
        try:
            seg = segment_text("front", prompt)
        except Exception as exc:
            info[prompt] = str(exc)[:120]
            continue
        if not seg.ok or len(seg.segmentations) == 0:
            info[prompt] = "none"
            continue
        best = None
        bd = None
        for cand in seg.segmentations:
            try:
                x1, y1, x2, y2 = (float(z) for z in cand.box_xyxy)
            except Exception:
                continue
            d = dist(0.5 * (x1 + x2), 0.5 * (y1 + y2), u, v)
            if bd is None or d < bd:
                bd = d
                best = cand
        if best is None:
            continue
        info[prompt] = round(bd, 1)
        cloud = mask_to_point_cloud(best, "front", target_frame=base_frame)
        pts = [[float(p[0]), float(p[1]), float(p[2])] for p in cloud.points]
        if len(pts) >= 10:
            return pts, info
    return [], info


def top_face_yaw(points, top_z, height):
    """Yaw (deg, 0..90) of the cube's top face from its upper points: min-area box search."""
    top = [p for p in points if p[2] >= top_z - 0.25 * height]
    if len(top) < 8:
        return None
    mx = sum(p[0] for p in top) / len(top)
    my = sum(p[1] for p in top) / len(top)
    best = None
    ba = None
    for deg in range(0, 90, 3):
        a = math.radians(deg)
        ca = math.cos(a)
        sa = math.sin(a)
        us = [(p[0] - mx) * ca + (p[1] - my) * sa for p in top]
        vs = [-(p[0] - mx) * sa + (p[1] - my) * ca for p in top]
        area = (quantile(us, 0.95) - quantile(us, 0.05)) * (quantile(vs, 0.95) - quantile(vs, 0.05))
        if ba is None or area < ba:
            ba = area
            best = deg
    return best


def ended(res):
    try:
        return bool(res.terminated) or bool(res.truncated)
    except Exception:
        return False


def summarize(res):
    out = {}
    try:
        out["ok"] = bool(res.ok)
    except Exception:
        pass
    try:
        out["status"] = str(res.status)
    except Exception:
        pass
    try:
        out["steps"] = int(res.steps_executed)
    except Exception:
        pass
    try:
        out["term"] = bool(res.terminated)
        out["trunc"] = bool(res.truncated)
    except Exception:
        pass
    try:
        if res.error is not None:
            out["error"] = str(res.error)[:200]
    except Exception:
        pass
    return out


motion_log = []
flags = {"over": False}


def go(name, pose_xyz, quat, max_steps=200, tol=0.01):
    if flags["over"]:
        return None
    res = move_to_pose(Pose(tuple(pose_xyz), tuple(quat), base_frame), tolerance=tol, max_steps=max_steps)
    s = summarize(res)
    s["name"] = name
    s["target"] = [round(v, 4) for v in pose_xyz]
    try:
        s["ee_after"] = [round(float(v), 4) for v in get_robot_state().end_effector_poses["primary"].position] if not ended(res) else None
    except Exception:
        pass
    motion_log.append(s)
    if ended(res):
        flags["over"] = True
    return res


def grip(name, closing):
    if flags["over"]:
        return None
    res = close_gripper() if closing else open_gripper()
    s = summarize(res)
    s["name"] = name
    motion_log.append(s)
    if ended(res):
        flags["over"] = True
    return res


# ------------------------------------------------------------------ main
ctx = get_task_context()
goal = str(ctx.language)
report["goal"] = goal
colour, ordinal = parse_goal(goal)
report["colour"] = colour
report["ordinal"] = ordinal
try:
    meta = robomme_plus.get_task_metadata()
    report["max_steps"] = meta["max_steps"]
    try:
        report["video_sampling"] = meta["conditioning_video_sampling"]
    except Exception:
        pass
except Exception as exc:
    report["meta_error"] = str(exc)[:200]

state0 = get_robot_state()
base_frame = state0.base_frame
home_pose = state0.end_effector_poses["primary"]
home_q = [float(v) for v in home_pose.quaternion_wxyz]
home_p = [float(v) for v in home_pose.position]
report["home_ee"] = [round(v, 4) for v in home_p]
report["home_q"] = [round(v, 4) for v in home_q]

video = robomme_plus.get_conditioning_video()
report["video_len"] = len(video)
per_frame = []
grip_trace = []
for k in range(len(video)):
    fr = video[k]
    if colour is not None:
        fb, fc = analyse_image(rgb_rows(fr.cameras["front"]), colour, 2)
        per_frame.append((float(fr.timestamp_s), fb, fc))
    try:
        st = fr.robot_state
        grip_trace.append([round(float(fr.timestamp_s), 2), round(float(st.gripper_positions["primary"]), 3),
                           [round(float(v), 3) for v in st.end_effector_poses["primary"].position]])
    except Exception:
        pass
report["stage"] = "video_loaded"
# keep a compact gripper/EE trace for diagnosis (every 4th sample)
report["video_grip_trace"] = grip_trace[::4]

memory = {}
if colour is not None and ordinal is not None and len(per_frame) > 0:
    memory = analyse_video(per_frame)
per_frame = []
slots = memory.get("slots_full", [])
report["memory"] = {k: v for k, v in memory.items() if k != "slots_full"}
report["stage"] = "video_analysed"

# live front image: slots now, cube pixel
live = get_observation()
live_rows = rgb_rows(live.cameras["front"])
live_blobs, live_cube = analyse_image(live_rows, colour if colour else "red", 2)
report["live_blobs"] = [[round(b["x"], 1), round(b["y"], 1), b["n"]] for b in live_blobs]
report["live_cube_px"] = None if live_cube is None else [round(live_cube["x"], 1), round(live_cube["y"], 1), live_cube["n"]]

target_px = None
seq = memory.get("identity_sequence", [])
if ordinal is not None and len(seq) >= ordinal and slots:
    ident = seq[ordinal - 1]
    fslot = memory["final_slot_of_identity"][ident]
    sx = slots[fslot]["x"]
    sy = slots[fslot]["y"]
    # snap to the nearest live target blob (scene at execution start)
    best = None
    bd = None
    for b in live_blobs:
        if b["n"] < 20:
            continue
        d = dist(b["x"], b["y"], sx, sy)
        if bd is None or d < bd:
            bd = d
            best = b
    if best is not None and bd < 2.0 * slots[fslot]["r"]:
        target_px = [best["x"], best["y"]]
    else:
        target_px = [sx, sy]
    report["chosen"] = {"identity": ident, "final_slot": fslot, "slot_px": [round(sx, 1), round(sy, 1)],
                        "target_px": [round(target_px[0], 1), round(target_px[1], 1)],
                        "snap_dist": None if bd is None else round(bd, 1)}
else:
    report["chosen"] = {"error": "not enough placements decoded", "n_placements": len(seq)}

# ------------------------------------------------------------------ act
if target_px is not None and live_cube is not None:
    cube_pts, cinfo = cloud_from_pixel(live_cube["x"], live_cube["y"])
    if len(cube_pts) < 10:
        cube_pts, cinfo2 = cloud_from_text([colour + " cube", colour + " block"], live_cube["x"], live_cube["y"])
        cinfo["fallback"] = cinfo2
    report["cube_seg"] = cinfo
    tgt_pts, tinfo = cloud_from_pixel(target_px[0], target_px[1])
    if len(tgt_pts) < 10:
        tgt_pts, tinfo2 = cloud_from_text(["purple target", "target", "bullseye"], target_px[0], target_px[1])
        tinfo["fallback"] = tinfo2
    report["target_seg"] = tinfo
    if len(cube_pts) >= 10 and len(tgt_pts) >= 10:
        cxw = quantile([p[0] for p in cube_pts], 0.5)
        cyw = quantile([p[1] for p in cube_pts], 0.5)
        ctop = quantile([p[2] for p in cube_pts], 0.95)
        cbot = quantile([p[2] for p in cube_pts], 0.05)
        cube_h = ctop - cbot
        txw = quantile([p[0] for p in tgt_pts], 0.5)
        tyw = quantile([p[1] for p in tgt_pts], 0.5)
        tz = quantile([p[2] for p in tgt_pts], 0.5)
        report["cube_xyz"] = [round(cxw, 4), round(cyw, 4), round(ctop, 4), round(cube_h, 4)]
        report["target_xyz"] = [round(txw, 4), round(tyw, 4), round(tz, 4)]
        # gripper yaw: align the finger axis with a cube face
        yaw = top_face_yaw(cube_pts, ctop, cube_h)
        finger_axis = rotate_vec(home_q, [0.0, 1.0, 0.0])
        phi = math.degrees(math.atan2(finger_axis[1], finger_axis[0]))
        grasp_q = home_q
        if yaw is not None:
            delta = (yaw - phi) % 90.0
            if delta > 45.0:
                delta -= 90.0
            half = math.radians(delta) * 0.5
            qz = [math.cos(half), 0.0, 0.0, math.sin(half)]
            grasp_q = quat_mul(qz, home_q)
            report["yaw"] = {"cube_deg": yaw, "finger_deg": round(phi, 1), "delta_deg": round(delta, 1)}
        grasp_z = min(ctop - 0.5 * cube_h, ctop - 0.015)
        hover = max(ctop, tz) + 0.12
        go("pre_grasp", [cxw, cyw, hover], grasp_q, 250, 0.015)
        grip("open", False)
        go("grasp", [cxw, cyw, grasp_z], grasp_q, 200, 0.01)
        grip("close", True)
        go("lift", [cxw, cyw, hover], grasp_q, 200, 0.015)
        go("above_target", [txw, tyw, hover], grasp_q, 250, 0.015)
        place_z = tz + cube_h * 0.5 + 0.015
        go("place", [txw, tyw, place_z], grasp_q, 200, 0.01)
        grip("release", False)
        if not flags["over"]:
            st = get_robot_state()
            for tick in range(10):
                rr = step(RobotAction(arms={"primary": ArmCommand(
                    mode="joint_position", target=st.joint_positions["primary"],
                    gripper_position=1.0, embodiment=st.embodiment)}))
                if ended(rr):
                    flags["over"] = True
                    break
        go("retreat", [txw, tyw, place_z + 0.08], grasp_q, 150, 0.02)
        report["stage"] = "placed"
    else:
        report["stage"] = "segmentation_failed"
else:
    report["stage"] = "no_target_decoded"

report["motions"] = motion_log
report["episode_over"] = flags["over"]
result = report
