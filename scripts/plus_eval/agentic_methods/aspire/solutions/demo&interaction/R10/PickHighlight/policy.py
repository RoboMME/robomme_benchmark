# PickHighlight v06 — v05 (seed 0: native success at step 970) plus end-of-episode guards only:
# v05 called get_robot_state() after the final press had already ended the episode, so the
# runtime raised and program_ok was false. No behaviour change before the episode ends.
# --- v05 notes:
# PickHighlight v05 — v04 plus fixes from its seed-0 rollout: hold_tick commanded the gripper
# with its own reading, which decayed geometrically and closed the open gripper during long
# watches (the descent then stalled on a cube top for 240 ticks: 3 retries x 80); now snapped
# to open/closed. Descents are single-shot and short; the pick-duration estimate uses the last
# measured pick (clamped 60-120) instead of the max; picks stop when the horizon is near.
# --- v04 notes:
# periodic rearrangements. v03's rollout (and the released ep-0 recording)
# show the cubes exchanging places in pairs in repeated phases (seed 0: ticks ~55-95, ~195-231,
# ~337-375: period ~142 ticks, ~40 ticks long), not once. v03 watched only one phase, so its
# second pick lifted a non-highlighted cube that had moved into a highlighted cube's place
# (native failure, as in v01). Its depth tracker also latched onto the closed fingers.
# v04: keep the gripper open at the home view; track every remaining highlighted cube through
# each rearrangement with a depth band (2-8 cm above the local table, fingers excluded);
# measure the rearrangement period online; press and pick only inside the still windows,
# one lean pick per window, always returning to the home view before the next predicted phase.
# Base: v03_track_shuffle.py (v02 IK execution, v01 disc detection).
import math

report = {"version": "v07", "events": []}
DONE = {"ended": False}


def log(msg):
    report["events"].append(str(msg))


def ended_from(res):
    try:
        if bool(res.terminated) or bool(res.truncated):
            DONE["ended"] = True
            return True
    except Exception:
        return False
    return False


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def qrot(q, v):
    p = (0.0, v[0], v[1], v[2])
    qc = (q[0], -q[1], -q[2], -q[3])
    r = qmul(qmul(q, p), qc)
    return (r[1], r[2], r[3])


def qz(angle):
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def wrap_half_pi(a):
    # gripper is symmetric under 180 deg: map to (-pi/2, pi/2]
    while a > math.pi / 2.0:
        a -= math.pi
    while a <= -math.pi / 2.0:
        a += math.pi
    return a


def to_list(arr):
    try:
        return arr.tolist()
    except Exception:
        return [list(row) for row in arr]


def is_white(p):
    r = int(p[0]); g = int(p[1]); b = int(p[2])
    mn = min(r, g, b); mx = max(r, g, b)
    return mn > 195 and mx - mn < 35


def white_count(img, stride):
    n = 0
    h = len(img); w = len(img[0])
    for v in range(0, h, stride):
        row = img[v]
        for u in range(0, w, stride):
            if is_white(row[u]):
                n += 1
    return n


def hold_tick():
    st = get_robot_state()
    res = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=[float(x) for x in st.joint_positions["primary"]],
        gripper_position=(1.0 if float(st.gripper_positions["primary"]) > 0.5 else 0.0),
        embodiment=st.embodiment)}))
    ended_from(res)
    return res


def points_of(cloud):
    try:
        return to_list(cloud.points)
    except Exception:
        return [list(p) for p in cloud.points]


def cube_from_points(pts):
    """Top-face centre / top z / yaw / edge from a cube's projected points (robot_base)."""
    if len(pts) < 8:
        return None
    zs = [float(p[2]) for p in pts]
    top = quantile(zs, 0.92)
    face = [p for p in pts if float(p[2]) > top - 0.012]
    if len(face) < 5:
        face = pts
    xs = [float(p[0]) for p in face]
    ys = [float(p[1]) for p in face]
    cx = quantile(xs, 0.5); cy = quantile(ys, 0.5)
    best = None
    for k in range(45):
        th = math.radians(2.0 * k)
        c = math.cos(th); s = math.sin(th)
        a = [(float(p[0]) - cx) * c + (float(p[1]) - cy) * s for p in face]
        b = [-(float(p[0]) - cx) * s + (float(p[1]) - cy) * c for p in face]
        ea = quantile(a, 0.97) - quantile(a, 0.03)
        eb = quantile(b, 0.97) - quantile(b, 0.03)
        area = ea * eb
        if best is None or area < best[0]:
            best = (area, th, ea, eb, 0.5 * (quantile(a, 0.97) + quantile(a, 0.03)),
                    0.5 * (quantile(b, 0.97) + quantile(b, 0.03)))
    th = best[1]
    # recentre on the box midpoint of the fitted rectangle
    mx = cx + best[4] * math.cos(th) - best[5] * math.sin(th)
    my = cy + best[4] * math.sin(th) + best[5] * math.cos(th)
    edge = max(best[2], best[3])
    return {"x": mx, "y": my, "top": top, "yaw": th, "edge": edge, "n": len(pts),
            "zmin": quantile(zs, 0.05)}


def cube_candidates(camera, prompt):
    seg = segment_text(camera, prompt)
    out = []
    if not seg.ok:
        return out, seg
    for cand in seg.segmentations:
        if cand.box_xyxy is None:
            continue
        x1, y1, x2, y2 = (float(v) for v in cand.box_xyxy)
        out.append({"cand": cand, "box": (x1, y1, x2, y2), "score": float(cand.score),
                    "cu": 0.5 * (x1 + x2), "cv": 0.5 * (y1 + y2),
                    "half": 0.5 * max(x2 - x1, y2 - y1)})
    return out, seg


def dedupe(cands):
    kept = []
    for c in sorted(cands, key=score_key, reverse=True):
        dup = False
        for k in kept:
            if abs(c["cu"] - k["cu"]) < 0.6 * k["half"] and abs(c["cv"] - k["cv"]) < 0.6 * k["half"]:
                dup = True
        if not dup:
            kept.append(c)
    return kept


def score_key(c):
    return c["score"]


def ring_fraction(cur, pre, cu, cv, radius):
    h = len(cur); w = len(cur[0])
    hits = 0; total = 0
    for k in range(24):
        a = 2.0 * math.pi * k / 24.0
        u = int(round(cu + radius * math.cos(a)))
        v = int(round(cv + radius * math.sin(a)))
        if u < 0 or v < 0 or u >= w or v >= h:
            continue
        total += 1
        if is_white(cur[v][u]) and (pre is None or not is_white(pre[v][u])):
            hits += 1
    if total == 0:
        return 0.0
    return hits / float(total)


def project(cand, camera):
    try:
        pc = mask_to_point_cloud(cand, camera, target_frame=base_frame)
        return cube_from_points(points_of(pc))
    except Exception as exc:
        log("project failed: " + str(exc))
        return None


IKSTAT = {"ik_ok": 0, "ik_fallback": 0, "insane": 0}


def ik_for(pos, quat):
    """IK from a canonical elbow-up seed; tries the 180-deg-equivalent gripper yaw as well."""
    base_yaw = math.atan2(pos[1], pos[0])
    seed = [float(v) for v in home_j]
    seed[0] = base_yaw
    best = None
    for flip in (0.0, math.pi):
        q = qmul(qz(flip), quat) if flip != 0.0 else quat
        try:
            ik = solve_ik(Pose(tuple(float(v) for v in pos), tuple(q), base_frame), seed_joints=seed)
        except Exception as exc:
            log("solve_ik raised " + str(exc))
            continue
        if not ik.ok:
            continue
        try:
            j = [float(v) for v in ik.joint_positions]
        except Exception as exc:
            log("ik joint read failed " + str(exc))
            continue
        if abs(j[0] - base_yaw) > 0.9 or abs(j[2]) > 0.9:
            IKSTAT["insane"] += 1
            continue
        cost = sum((j[k] - seed[k]) ** 2 for k in range(7))
        if best is None or cost < best[0]:
            best = (cost, j)
    if best is None:
        return None
    return best[1]


LAST = {"res": None}


def move(pos, quat, tol, steps, press=False):
    if DONE["ended"] and LAST["res"] is not None:
        return LAST["res"]
    res = move_inner(pos, quat, tol, steps, press)
    LAST["res"] = res
    return res


def move_inner(pos, quat, tol, steps, press):
    j = ik_for(pos, quat)
    if j is None:
        IKSTAT["ik_fallback"] += 1
        res = move_to_pose(Pose(tuple(float(v) for v in pos), tuple(quat), base_frame),
                           tolerance=tol, max_steps=steps)
        ended_from(res)
        return res
    IKSTAT["ik_ok"] += 1
    res = move_to_joints(j, tolerance=0.01, max_steps=steps)
    ended_from(res)
    tries = 1
    while (not press) and (not res.ok) and (not DONE["ended"]) and tries < 3:
        res = move_to_joints(j, tolerance=0.01, max_steps=steps)
        ended_from(res)
        tries += 1
    return res


def gripper_now():
    return float(get_robot_state().gripper_positions["primary"])


# ---------------------------------------------------------------------------
ctx = get_task_context()
report["language"] = ctx.language
try:
    meta = robomme_plus.get_task_metadata()
    report["max_steps"] = meta["max_steps"]
    try:
        report["metadata"] = dict(meta)
    except Exception:
        pass
except Exception as exc:
    report["metadata_error"] = str(exc)
try:
    report["conditioning_video_len"] = len(robomme_plus.get_conditioning_video())
except Exception as exc:
    report["conditioning_video_error"] = str(exc)

state0 = get_robot_state()
base_frame = state0.base_frame
home_q = tuple(float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz)
home_p = tuple(float(v) for v in state0.end_effector_poses["primary"].position)
home_j = [float(v) for v in state0.joint_positions["primary"]]
report["home_pose"] = [list(home_p), list(home_q)]
close_axis = qrot(home_q, (0.0, 1.0, 0.0))
phi0 = math.atan2(close_axis[1], close_axis[0])
report["phi0_deg"] = math.degrees(phi0)

obs0 = get_observation()
pre_wrist = to_list(obs0.cameras["wrist"].rgb)
pre_front = to_list(obs0.cameras["front"].rgb)
baseline = white_count(pre_wrist, 4)
report["white_baseline"] = baseline

# --- 1. wait for the highlight ------------------------------------------------
counts = []
onset = None
ticks = 0
cur_wrist = pre_wrist
while ticks < 90 and not DONE["ended"]:
    hold_tick()
    ticks += 1
    if DONE["ended"]:
        break
    obs = get_observation()
    cur_wrist = to_list(obs.cameras["wrist"].rgb)
    n = white_count(cur_wrist, 4)
    counts.append(n)
    if onset is None and n > baseline + 12:
        onset = ticks
    if onset is not None and ticks >= onset + 3:
        if len(counts) >= 2 and counts[-1] <= counts[-2] * 1.05:
            break
    if onset is not None and ticks >= onset + 8:
        break
report["white_counts"] = counts
report["highlight_onset_tick"] = onset
report["observe_tick"] = ticks
use_pre = pre_wrist if onset is not None else None

# --- 2. classify cubes while the highlight is visible ------------------------
cubes = []
cands, seg = cube_candidates("wrist", "cube")
report["wrist_cube_raw"] = len(cands)
cands = dedupe(cands)
for c in cands:
    cu = c["cu"]; cv = c["cv"]; half = c["half"]
    # skip white objects (the button) by the colour at the box centre
    centre_white = is_white(cur_wrist[int(cv)][int(cu)])
    f1 = max(ring_fraction(cur_wrist, use_pre, cu, cv, 1.05 * half),
             ring_fraction(cur_wrist, use_pre, cu, cv, 1.2 * half))
    f2 = ring_fraction(cur_wrist, use_pre, cu, cv, 1.35 * half)
    geo = project(c["cand"], "wrist")
    entry = {"box": [round(v, 1) for v in c["box"]], "score": round(c["score"], 3),
             "ring": round(max(f1, f2), 3), "centre_white": centre_white}
    if geo is not None:
        entry["geo"] = {k: round(geo[k], 4) for k in geo}
    c["geo"] = geo
    c["ring"] = max(f1, f2)
    c["centre_white"] = centre_white
    cubes.append(entry)
report["wrist_cubes"] = cubes

valid = [c for c in cands if c["geo"] is not None and not c["centre_white"]
         and 0.01 < c["geo"]["edge"] < 0.08]
# plausibility in height: cubes rest on the table -> tops within a band of the median top
med_top = None
if len(valid) > 0:
    med_top = quantile([c["geo"]["top"] for c in valid], 0.5)
    valid = [c for c in valid if abs(c["geo"]["top"] - med_top) < 0.03]
report["n_valid_cubes"] = len(valid)

# Highlight discs: SAM3 boxes, validated by pixels that turned white since tick 0 on a ring
# between the cube and the disc rim (cube half-width ~0.3 box, disc radius ~0.5 box).
discs = []
disc_log = []
for prompt in ("white disc", "white circle"):
    try:
        dseg = segment_text("wrist", prompt)
    except Exception as exc:
        log("disc seg failed " + str(exc))
        continue
    if not dseg.ok:
        continue
    for d in dseg.segmentations:
        if d.box_xyxy is None:
            continue
        x1, y1, x2, y2 = (float(v) for v in d.box_xyxy)
        size = max(x2 - x1, y2 - y1)
        cu = 0.5 * (x1 + x2); cv = 0.5 * (y1 + y2)
        newfrac = max(ring_fraction(cur_wrist, use_pre, cu, cv, 0.36 * size),
                      ring_fraction(cur_wrist, use_pre, cu, cv, 0.43 * size))
        disc_log.append([prompt, round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                         round(float(d.score), 3), round(newfrac, 2)])
        if newfrac < 0.3 or size < 8 or size > 70:
            continue
        dup = False
        for k in discs:
            if abs(k["cu"] - cu) < 0.5 * size and abs(k["cv"] - cv) < 0.5 * size:
                dup = True
        if not dup:
            discs.append({"cand": d, "cu": cu, "cv": cv, "size": size, "newfrac": newfrac})
report["disc_log"] = disc_log


def disc_cube(dk):
    """3D cube estimate for one disc: point-prompt the cube at the disc centre, else disc centre."""
    g = None
    try:
        sp = segment_points("wrist", [[dk["cu"], dk["cv"]]])
        if sp.ok and len(sp.segmentations) > 0:
            sc = sp.segmentations[0]
            ok_box = True
            if sc.box_xyxy is not None:
                bx1, by1, bx2, by2 = (float(v) for v in sc.box_xyxy)
                ok_box = max(bx2 - bx1, by2 - by1) < 0.9 * dk["size"]
            if ok_box:
                g = project(sc, "wrist")
    except Exception as exc:
        log("segment_points failed " + str(exc))
    dg = None
    try:
        dpc = mask_to_point_cloud(dk["cand"], "wrist", target_frame=base_frame)
        dpts = points_of(dpc)
        if len(dpts) >= 5:
            dg = (quantile([float(p[0]) for p in dpts], 0.5), quantile([float(p[1]) for p in dpts], 0.5))
    except Exception as exc:
        log("disc projection failed " + str(exc))
    src = "points"
    if g is not None and dg is not None:
        if math.sqrt((g["x"] - dg[0]) ** 2 + (g["y"] - dg[1]) ** 2) > 0.03:
            g = None
    if g is not None and not (0.015 < g["edge"] < 0.07):
        g = None
    if g is not None and med_top is not None and abs(g["top"] - med_top) > 0.025:
        g = None
    if g is None and dg is not None:
        src = "disc"
        top = med_top if med_top is not None else 0.04
        g = {"x": dg[0], "y": dg[1], "top": top, "yaw": 0.0, "edge": 0.04, "n": 0, "zmin": 0.0}
    if g is None:
        return None
    g["src"] = src
    return g


targets = []
for dk in discs:
    g = disc_cube(dk)
    if g is None:
        continue
    targets.append({"geo": g, "ring": dk["newfrac"], "src": g["src"], "uv": [dk["cu"], dk["cv"]]})
# union with v00's ring test on directly segmented cubes
for c in valid:
    if c["ring"] >= 0.4:
        dup = False
        for t in targets:
            if math.sqrt((t["geo"]["x"] - c["geo"]["x"]) ** 2 + (t["geo"]["y"] - c["geo"]["y"]) ** 2) < 0.03:
                dup = True
        if not dup:
            targets.append({"geo": c["geo"], "ring": c["ring"], "src": "cube-ring", "uv": [c["cu"], c["cv"]]})
report["targets"] = [[round(t["geo"]["x"], 4), round(t["geo"]["y"], 4), round(t["geo"]["top"], 4),
                      round(t["ring"], 2), t["src"]] for t in targets]

# --- 3. button ---------------------------------------------------------------
def find_button():
    for camera in ("front", "wrist"):
        bc, bseg = cube_candidates(camera, "button")
        best = None
        for c in bc[:5]:
            try:
                pc = mask_to_point_cloud(c["cand"], camera, target_frame=base_frame)
                pts = points_of(pc)
            except Exception:
                continue
            if len(pts) < 5:
                continue
            xs = [float(p[0]) for p in pts]; ys = [float(p[1]) for p in pts]
            zs = [float(p[2]) for p in pts]
            ext = max(quantile(xs, 0.95) - quantile(xs, 0.05), quantile(ys, 0.95) - quantile(ys, 0.05))
            top = quantile(zs, 0.95)
            info = {"camera": camera, "x": quantile(xs, 0.5), "y": quantile(ys, 0.5), "top": top,
                    "ext": ext, "score": c["score"]}
            log("button cand " + str({k: (round(info[k], 4) if k != "camera" else info[k]) for k in info}))
            if ext > 0.15 or top > 0.2:
                continue
            if best is None or c["score"] > best["score"]:
                best = info
        if best is not None:
            return best
    return None


button = find_button()
report["button"] = button
if button is not None:
    targets = [t for t in targets if math.sqrt((t["geo"]["x"] - button["x"]) ** 2 +
                                               (t["geo"]["y"] - button["y"]) ** 2) > 0.035]
report["n_targets"] = len(targets)



def cur_tick():
    if DONE["ended"]:
        return -1
    try:
        return int(round(float(get_robot_state().timestamp_s) * 20.0))
    except Exception:
        return -1


def press_button(tag):
    if button is None or DONE["ended"]:
        log(tag + ": no button / ended")
        return False
    r = close_gripper(); ended_from(r)
    if DONE["ended"]:
        return True
    bx = button["x"]; by = button["y"]; bt = button["top"]
    a = move((bx, by, bt + 0.08), home_q, 0.015, 200)
    log(tag + " above ok=" + str(a.ok) + " t=" + str(cur_tick()))
    if DONE["ended"]:
        return True
    d = move((bx, by, bt - 0.015), home_q, 0.006, 15, press=True)
    if DONE["ended"]:
        log(tag + " press: episode ended during the press")
        return True
    log(tag + " press ok=" + str(d.ok) + " z=" + str(round(float(get_robot_state().end_effector_poses["primary"].position[2]), 4)) + " t=" + str(cur_tick()))
    u = move((bx, by, bt + 0.08), home_q, 0.02, 120)
    return True


# --- 3b. tracking helpers (home view, wrist depth) ------------------------------------------
def go_home_view():
    res = None
    for k in range(3):
        res = move_to_joints(home_j, tolerance=0.005, max_steps=150)
        ended_from(res)
        if DONE["ended"] or res.ok:
            return res
    return res


def dval(x):
    try:
        return float(x)
    except Exception:
        return float(x[0])


def depth_rows(cam, v0, v1, u0, u1):
    d = cam.depth_m
    try:
        return d[v0:v1, u0:u1].tolist()
    except Exception:
        full = to_list(d)
        return [row[u0:u1] for row in full[v0:v1]]


def coarse_depth(cam):
    d = cam.depth_m
    try:
        rows = d[::4, ::4].tolist()
    except Exception:
        full = to_list(d)
        rows = [full[v][::4] for v in range(0, len(full), 4)]
    return [[dval(x) for x in r] for r in rows]


def wrist_obs():
    try:
        return get_observation(camera_names=["wrist"])
    except Exception:
        return get_observation()


IMG = {"h": len(pre_wrist), "w": len(pre_wrist[0])}
WIN = 14


def track_step(cam, t, radius, iters):
    """radius: max component extent from its seed; iters: max seed distance from prediction (px)."""
    pu = t["pos"][0]; pv = t["pos"][1]
    cu = pu + t["vel"][0]; cv = pv + t["vel"][1]
    u0 = max(0, int(cu) - WIN); u1 = min(IMG["w"], int(cu) + WIN + 1)
    v0 = max(0, int(cv) - WIN); v1 = min(IMG["h"], int(cv) + WIN + 1)
    if u1 - u0 < 4 or v1 - v0 < 4:
        t["lost"] += 1
        return 0
    rows = depth_rows(cam, v0, v1, u0, u1)
    vals = []
    for r in rows:
        for x in r:
            vals.append(dval(x))
    table = quantile(vals, 0.75)
    fg = []
    for i in range(len(rows)):
        r = rows[i]
        for j in range(len(r)):
            dj = dval(r[j])
            # cube tops/sides only: 2-8 cm nearer than the local table; fingers are far nearer
            if table - 0.08 < dj < table - 0.02:
                fg.append((u0 + j + 0.5, v0 + i + 0.5))
    fgset = set()
    for p in fg:
        fgset.add((int(p[0]), int(p[1])))
    seed = None
    for p in fgset:
        dd = (p[0] + 0.5 - cu) ** 2 + (p[1] + 0.5 - cv) ** 2
        if dd <= iters * iters and (seed is None or dd < seed[0]):
            seed = (dd, p)
    n = 0
    if seed is not None:
        comp = [seed[1]]
        seen = set([seed[1]])
        k = 0
        while k < len(comp):
            q = comp[k]
            k += 1
            for nb in ((q[0] + 1, q[1]), (q[0] - 1, q[1]), (q[0], q[1] + 1), (q[0], q[1] - 1)):
                if nb in fgset and nb not in seen:
                    if (nb[0] - seed[1][0]) ** 2 + (nb[1] - seed[1][1]) ** 2 <= radius * radius:
                        seen.add(nb)
                        comp.append(nb)
        n = len(comp)
        if n >= 6:
            cu = sum(q[0] + 0.5 for q in comp) / float(n)
            cv = sum(q[1] + 0.5 for q in comp) / float(n)
    if n < 6:
        t["lost"] += 1
        t["vel"] = [0.0, 0.0]
        return 0
    t["vel"] = [0.5 * t["vel"][0] + 0.5 * (cu - pu), 0.5 * t["vel"][1] + 0.5 * (cv - pv)]
    t["pos"] = [cu, cv]
    return n


SW = {"starts": [], "ends": [], "watches": [], "blips": []}


def watch(max_tick, tag):
    """Hold still at the home view, track every remaining target each tick. Return when one
    rearrangement has been seen to start and then settle, or at max_tick."""
    obsw = wrist_obs()
    camw = obsw.cameras["wrist"]
    for t in targets:
        t["vel"] = [0.0, 0.0]
        track_step(camw, t, 9.0, 6)
        t["vel"] = [0.0, 0.0]
    prev_c = coarse_depth(camw)
    wt = 0; quiet = 0; moving = 0; start = None; motion = []
    t0 = cur_tick()
    tick_now = t0
    while not DONE["ended"]:
        hold_tick()
        wt += 1
        if DONE["ended"]:
            break
        obsw = wrist_obs()
        camw = obsw.cameras["wrist"]
        for t in targets:
            track_step(camw, t, 9.0, 5)
        cur_c = coarse_depth(camw)
        m = 0
        for i in range(min(len(cur_c), len(prev_c))):
            a = cur_c[i]; b = prev_c[i]
            for j in range(min(len(a), len(b))):
                if abs(a[j] - b[j]) > 0.01:
                    m += 1
        prev_c = cur_c
        motion.append(m)
        try:
            tick_now = int(round(float(obsw.robot_state.timestamp_s) * 20.0))
        except Exception:
            tick_now = tick_now + 1
        if start is None:
            if wt > 6 and m >= 4 and motion[-2] >= 4:
                start = tick_now - 1
                moving = 0; quiet = 0
        else:
            if m <= 1:
                quiet += 1
            else:
                quiet = 0
            if m >= 4:
                moving += 1
            if quiet >= 4:
                if moving < 8:
                    SW["blips"].append([start, tick_now, moving])
                    start = None
                    quiet = 0
                else:
                    SW["starts"].append(start)
                    SW["ends"].append(tick_now - quiet)
                    break
        if tick_now >= max_tick and (start is None or tick_now - start > 120):
            break
    SW["watches"].append({"tag": tag, "from": t0, "to": tick_now, "start": start,
                          "motion": motion,
                          "tracks": [[round(t["pos"][0], 1), round(t["pos"][1], 1), t["lost"]] for t in targets]})
    return start is not None


def period():
    s = SW["starts"]
    best = None
    for i in range(1, len(s)):
        d = s[i] - s[i - 1]
        if d >= 60 and (best is None or d < best):
            best = d
    return best


def next_swap_start(now):
    s = SW["starts"]
    p = period()
    if len(s) == 0:
        return None
    if p is None:
        p = 140  # fallback before two rearrangements were observed (released recordings: ~142)
    n = s[-1] + p
    while n + 5 < now:
        n += p
        log("a rearrangement was probably missed (predicted start " + str(n - p) + ")")
    return n


# --- 3c. re-localize a tracked cube in 3D (home view) ------------------------------------
def scene_cubes():
    after, aseg = cube_candidates("wrist", "cube")
    after = dedupe(after)
    out = []
    for c in after:
        g = project(c["cand"], "wrist")
        if g is None or not (0.01 < g["edge"] < 0.08):
            continue
        if med_top is not None and abs(g["top"] - med_top) > 0.03:
            continue
        c["geo"] = g
        out.append(c)
    return out


def relocalize(t, cubes_now):
    u = t["pos"][0]; v = t["pos"][1]
    g = None; how = "none"
    try:
        sp = segment_points("wrist", [[u, v]])
        if sp.ok and len(sp.segmentations) > 0:
            sc = sp.segmentations[0]
            ok_box = True
            if sc.box_xyxy is not None:
                bx1, by1, bx2, by2 = (float(k) for k in sc.box_xyxy)
                ok_box = max(bx2 - bx1, by2 - by1) < 30.0 and bx1 - 2 <= u <= bx2 + 2 and by1 - 2 <= v <= by2 + 2
            if ok_box:
                g = project(sc, "wrist")
                how = "points"
    except Exception as exc:
        log("relocalize segment_points failed " + str(exc))
    if g is not None and not (0.015 < g["edge"] < 0.07):
        g = None
    if g is not None and med_top is not None and abs(g["top"] - med_top) > 0.025:
        g = None
    if g is None:
        best = None
        for c in cubes_now:
            dd = (c["cu"] - u) ** 2 + (c["cv"] - v) ** 2
            if dd < 64.0 and (best is None or dd < best[0]):
                best = (dd, c)
        if best is not None:
            g = dict(best[1]["geo"]); how = "cube-box"
    if g is None:
        how = "stale"
        g = dict(t["geo"])
    g["src"] = how
    t["geo"] = g
    t["src"] = how
    return g


def choose_yaw(cx, cy, cube_yaw, extra):
    best = None
    for alt in (0.0, math.pi / 2.0):
        alpha = cube_yaw + alt
        ux = math.cos(alpha); uy = math.sin(alpha)
        clear = 1.0
        for o in extra:
            if math.sqrt((o["x"] - cx) ** 2 + (o["y"] - cy) ** 2) < 0.02:
                continue
            for sgn in (1.0, -1.0):
                fx = cx + sgn * 0.05 * ux; fy = cy + sgn * 0.05 * uy
                dd = math.sqrt((o["x"] - fx) ** 2 + (o["y"] - fy) ** 2)
                clear = min(clear, dd)
        delta = wrap_half_pi(alpha - phi0)
        key = (clear, -abs(delta))
        if best is None or key > best[0]:
            best = (key, delta, clear)
    return best[1], best[2]


def refine(cx, cy):
    """Re-localize the cube nearest (cx, cy) from the wrist camera at the hover pose."""
    try:
        rc, rseg = cube_candidates("wrist", "cube")
    except Exception:
        return None, []
    best = None
    seen = []
    for c in rc[:12]:
        g = project(c["cand"], "wrist")
        if g is None or not (0.01 < g["edge"] < 0.08):
            continue
        seen.append(g)
        dd = math.sqrt((g["x"] - cx) ** 2 + (g["y"] - cy) ** 2)
        if dd < 0.03 and (best is None or dd < best[0]):
            best = (dd, g)
    return best, seen


def pick_once(g, rec, others_geo):
    """Lean grasp-lift-setdown from the home view, ending back at the home view.
    Returns True if the cube was held during the lift."""
    if gripper_now() < 0.9:
        r0 = open_gripper(); ended_from(r0)
    hover_z = g["top"] + 0.10
    delta, clear = choose_yaw(g["x"], g["y"], g["yaw"], others_geo)
    hv_delta = delta
    q = qmul(qz(delta), home_q)
    hv = move((g["x"], g["y"], hover_z), q, 0.015, 200)
    rec["hover_ok"] = bool(hv.ok)
    if DONE["ended"]:
        return False
    ref, seen = refine(g["x"], g["y"])
    cx = g["x"]; cy = g["y"]; top = g["top"]; yaw = g["yaw"]; edge = g["edge"]
    if ref is not None:
        rg = ref[1]
        rec["refined"] = [round(rg["x"], 4), round(rg["y"], 4), round(rg["top"], 4),
                          round(math.degrees(rg["yaw"]), 1), round(rg["edge"], 4)]
        cx = rg["x"]; cy = rg["y"]; top = rg["top"]; yaw = rg["yaw"]; edge = rg["edge"]
    else:
        rec["refined"] = None
    delta, clear = choose_yaw(cx, cy, yaw, seen + others_geo)
    q = qmul(qz(delta), home_q)
    rec["delta_deg"] = round(math.degrees(delta), 1)
    rec["clear"] = round(clear, 3)
    size = min(max(edge, 0.02), 0.06)
    grasp_z = top - 0.5 * size
    rec["yaw_change_deg"] = round(math.degrees(abs(wrap_half_pi(delta - hv_delta))), 1)
    if abs(wrap_half_pi(delta - hv_delta)) > math.radians(10.0):
        pre = move((cx, cy, top + 0.03), q, 0.01, 40, press=True)
    dn = move((cx, cy, grasp_z), q, 0.008, 40, press=True)
    if DONE["ended"]:
        return False
    cl = close_gripper(); ended_from(cl)
    if DONE["ended"]:
        return False
    gc = gripper_now()
    rec["grip_after_close"] = round(gc, 3)
    held = False
    if gc >= 0.08:
        # lean: lift 6 cm, release there (short drop), straight back to the home view
        lf = move((cx, cy, grasp_z + 0.06), q, 0.015, 40, press=True)
        if DONE["ended"]:
            return True
        gl = gripper_now()
        rec["grip_after_lift"] = round(gl, 3)
        held = gl >= 0.08
    op = open_gripper(); ended_from(op)
    if DONE["ended"]:
        return held
    if not held:
        up = move((cx, cy, top + 0.05), q, 0.02, 40, press=True)
        if DONE["ended"]:
            return held
    go_home_view()
    return held


# --- 4. schedule: watch rearrangements from home, act only in the still windows ----------
report["pick_log"] = []
for t in targets:
    t["pos"] = [float(t["uv"][0]), float(t["uv"][1])]
    t["vel"] = [0.0, 0.0]
    t["lost"] = 0
# (a) still at the home view since detection: watch the first rearrangement
seen1 = watch(170, "w1")
# (b) press in the still window, come back, watch the next rearrangement
press_button("press1")
if not DONE["ended"]:
    r = open_gripper(); ended_from(r)
if not DONE["ended"]:
    go_home_view()
if not DONE["ended"]:
    ns = next_swap_start(cur_tick())
    watch((ns + 90) if ns is not None else 450, "w2")
PICK_EST = {"ticks": 50}
MAXT = {"t": 2000}
try:
    MAXT["t"] = int(report["max_steps"])
except Exception:
    pass
attempts = {}
loops = 0
while not DONE["ended"] and len(targets) > 0 and loops < 30:
    loops += 1
    now = cur_tick()
    ns = next_swap_start(now)
    avail = (ns - now - 4) if ns is not None else 10000
    if now > 0 and MAXT["t"] - now < 90:
        log("stopping picks to leave time for the final press at t=" + str(now))
        break
    if avail < PICK_EST["ticks"]:
        watch(ns + 90 if ns is not None else now + 200, "w" + str(len(SW["watches"]) + 1))
        continue
    cubes_now = scene_cubes()
    for t in targets:
        relocalize(t, cubes_now)
    # nearest-to-home target first (shortest round trip)
    best = None
    for t in targets:
        dd = (t["geo"]["x"] - home_p[0]) ** 2 + (t["geo"]["y"] - home_p[1]) ** 2
        if best is None or dd < best[0]:
            best = (dd, t)
    t = best[1]
    others_geo = [c["geo"] for c in cubes_now]
    rec = {"t0": now, "avail": avail, "uv": [round(t["pos"][0], 1), round(t["pos"][1], 1)],
           "start": [round(t["geo"]["x"], 4), round(t["geo"]["y"], 4), round(t["geo"]["top"], 4)],
           "src": t["src"]}
    held = pick_once(t["geo"], rec, others_geo)
    t1 = cur_tick()
    rec["t1"] = t1
    rec["held"] = held
    report["pick_log"].append(rec)
    if t1 > 0 and now > 0:
        PICK_EST["ticks"] = min(100, max(42, t1 - now + 2))
    if held:
        targets = [k for k in targets if k is not t]
    else:
        t["fails"] = t.get("fails", 0) + 1
        if t["fails"] >= 2:
            log("giving up on a target after 2 empty grasps")
            targets = [k for k in targets if k is not t]
report["swaps"] = SW
report["targets_left"] = len(targets)

# --- 5. final button press ---------------------------------------------------
if not DONE["ended"]:
    press_button("press2")
    if not DONE["ended"]:
        for k in range(5):
            if not DONE["ended"]:
                hold_tick()
report["ik_stats"] = IKSTAT
report["ended"] = DONE["ended"]
try:
    if not DONE["ended"]:
        report["final_ts"] = float(get_robot_state().timestamp_s)
except Exception:
    pass
result = report
