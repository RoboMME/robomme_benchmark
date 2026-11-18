# v02_low_hover_fast_press -- ButtonUnmaskSwap (robomme_plus task 2).
# v02 vs v01 (diagnosed from v01 ep0's own result/timeline/keyframes): v01 pressed the right
# button at tick 82 without ending the episode, then re-rotated the hand 4 times over the left
# button; the yaw-change branch skipped the force check, the second press never happened and
# the episode ended at step 179 (press deadline). Containers circle the buttons at ~0.11 m and
# ~4 mm/tick, so a 22-tick prediction never cleared. v02: (a) force check first, bounded yaw
# changes; (b) hover LOW, with the hand body just above the container tops, so only the last
# few millimetres of the press (a few ticks) bring the hand body below the container tops, and
# predict only that short window; (c) occluded tracks get a growing uncertainty radius;
# (d) slightly deeper single press.
#
# v01 header:
# Changes from v00 (each addresses a failure diagnosed in v00 ep0's own trace/keyframes):
#  1. Task predicate / early termination: the episode terminated (no success) on the very first
#     tick the closed fingers depressed the left button cap (step 40) while a moving container
#     was ~5 cm in front of that button. v01 presses only when no tracked container is within a
#     clearance radius of the button (now and predicted 12 ticks ahead), descends vertically
#     from a hover pose, and retracts on the first tick the cap is depressed (single press).
#     Goal says "press BOTH buttons once": v01 presses the two "round button" detections.
#  2. Perception: SAM returns <= 5 segmentations, there are 8 containers, and the t=0
#     "white container" reference was the robot base/buttons (no containers exist before step
#     ~10). v01 tracks containers from the front DEPTH image instead: pixels deprojected to
#     robot_base, kept in a height band above cubes/buttons, clustered top-down, and tracked by
#     nearest-prediction point assignment with constant-velocity coasting.
#  3. Tracking under occlusion: the arm stays home (clean front view) except during the two
#     presses; observation comes from each step's own StepResult.
#  4. Goal order: blue, then green, then red containers are lifted in order; each is put back
#     down before the next pick; the gripper yaw is aligned to the container's observed sides.
# All positions/colours/heights come from the live observation; constants are generic
# tolerances, offsets and tick budgets only.

import math

CAM = "front"
R = {"version": "v02_low_hover_fast_press", "log": [], "errors": [], "motions": [],
     "timeline": [], "presses": [], "picks": []}
result = R
S = {"tick": 0, "terminal": False, "hz": 20.0, "emb": "robomme_plus", "base": "robot_base",
     "q_home": None, "down_q": [0.0, 1.0, 0.0, 0.0], "last_depth": None, "ee": None,
     "phase": "init"}
TR = {"tracks": [], "ctop": None, "init": False, "rays": None, "last_obs_tick": -1,
      "cand_hist": [], "last_motion": 0, "frames": 0}
COLOUR_WORDS = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink",
                "white", "gray", "black", "brown"]
SYN = {"grey": "gray", "violet": "purple", "magenta": "pink"}
HUES = [["red", 0.0], ["orange", 28.0], ["yellow", 58.0], ["green", 120.0],
        ["cyan", 185.0], ["blue", 225.0], ["purple", 275.0], ["pink", 322.0]]
PRESS_FORCE = [100, 165]      # force 1st/2nd press descent by these ticks (register < 180)
HAND_BOTTOM = 0.030           # generic Panda: hand body bottom ~0.034 m above the EE point

HAND_HL = 0.105               # generic Panda hand half-length along the finger axis (+margin)
HAND_HW = 0.040               # generic Panda hand half-width across the finger axis (+margin)
CLEAR_MARGIN = 0.008          # required gap between hand footprint and container footprint
PRESS_AHEAD = 8               # ticks the hand body spends below container tops (low hover)
MOTION_END = 480              # public rule: separate 480-step motion limit
STRIDE = 2                    # depth pixel stride


def note(msg):
    if len(R["log"]) < 400:
        R["log"].append([S["tick"], str(msg)])


def err(where, exc):
    if len(R["errors"]) < 60:
        R["errors"].append([S["tick"], where, str(exc)[:300]])
    if "ended" in str(exc) or "closing" in str(exc):
        S["terminal"] = True


def rnd(v, k):
    try:
        return round(float(v), k)
    except Exception:
        return None


def res_terminal(res):
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


def sync_tick():
    try:
        st = get_robot_state()
        S["tick"] = int(round(float(st.timestamp_s) * S["hz"]))
        S["ee"] = [float(v) for v in st.end_effector_poses["primary"].position]
        return st
    except Exception as exc:
        err("sync_tick", exc)
        return None


def record_exec(name, res):
    entry = {"call": name, "tick_before": S["tick"]}
    try:
        entry["ok"] = bool(res.ok)
    except Exception:
        entry["ok"] = None
    try:
        entry["status"] = str(res.status)
    except Exception:
        entry["status"] = "?"
    try:
        entry["steps"] = int(res.steps_executed)
    except Exception:
        entry["steps"] = None
    try:
        if res.error is not None:
            entry["error"] = str(res.error)[:200]
    except Exception:
        pass
    if res_terminal(res):
        S["terminal"] = True
        entry["terminal"] = True
    if not S["terminal"]:
        sync_tick()
    entry["tick_after"] = S["tick"]
    if len(R["motions"]) < 100:
        R["motions"].append(entry)
    return entry


def joints_now():
    st = get_robot_state()
    S["ee"] = [float(v) for v in st.end_effector_poses["primary"].position]
    return [float(v) for v in st.joint_positions["primary"]]


def as_joint_list(value):
    try:
        out = [float(v) for v in value]
        if len(out) >= 7:
            return out[:7]
    except Exception:
        pass
    try:
        return [float(v) for v in value["primary"]][:7]
    except Exception:
        return None


def step_joints(q, grip, observe):
    """One control tick; if observe, run the depth tracker on the step's own observation."""
    if S["terminal"]:
        return False
    try:
        res = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=[float(v) for v in q],
            gripper_position=float(grip), embodiment=S["emb"])}))
    except Exception as exc:
        err("step", exc)
        return False
    S["tick"] += 1
    try:
        S["ee"] = [float(v) for v in res.observation.robot_state.end_effector_poses["primary"].position]
    except Exception:
        pass
    if res_terminal(res):
        S["terminal"] = True
        note("terminal flag after step")
        return False
    if observe:
        depth = None
        try:
            depth = res.observation.cameras[CAM].depth_m
        except Exception:
            depth = None
        if depth is None:
            try:
                depth = get_observation(camera_names=[CAM]).cameras[CAM].depth_m
            except Exception as exc:
                err("get_observation", exc)
        if depth is not None:
            track_frame(depth)
    return True


# ---------------------------------------------------------------- geometry

def quat_to_mat(q):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def build_rays(cam):
    k = [[float(v) for v in row] for row in cam.intrinsics]
    fx = k[0][0]
    fy = k[1][1]
    cx = k[0][2]
    cy = k[1][2]
    pos = [float(v) for v in cam.camera_pose.position]
    rot = quat_to_mat([float(v) for v in cam.camera_pose.quaternion_wxyz])
    shape = [int(v) for v in cam.depth_m.shape]
    rays = []
    for v in range(0, shape[0], STRIDE):
        row = []
        for u in range(0, shape[1], STRIDE):
            a = (u + 0.5 - cx) / fx
            b = (v + 0.5 - cy) / fy
            row.append([u, rot[0][0] * a + rot[0][1] * b + rot[0][2],
                        rot[1][0] * a + rot[1][1] * b + rot[1][2],
                        rot[2][0] * a + rot[2][1] * b + rot[2][2]])
        rays.append([v, row])
    TR["rays"] = rays
    TR["cam_pos"] = pos
    R["camera_model"] = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "pos": [rnd(p, 4) for p in pos],
                         "rot": [[rnd(e, 4) for e in r] for r in rot], "shape": shape}


def band_points(depth, zlo, zhi):
    """Deproject strided depth pixels; keep robot_base points in a height band on the table."""
    rows = depth.tolist()
    px = TR["cam_pos"][0]
    py = TR["cam_pos"][1]
    pz = TR["cam_pos"][2]
    ee = S["ee"]
    excl = ee is not None and ee[2] < zhi + 0.10
    pts = []
    for item in TR["rays"]:
        drow = rows[item[0]]
        for ray in item[1]:
            d = drow[ray[0]]
            if d <= 0.05 or d > 2.5:
                continue
            z = pz + d * ray[3]
            if z < zlo or z > zhi:
                continue
            x = px + d * ray[1]
            y = py + d * ray[2]
            if x < 0.25 or x > 1.0 or y < -0.5 or y > 0.5:
                continue
            if excl and (x - ee[0]) ** 2 + (y - ee[1]) ** 2 < 0.04 * 0.04:
                continue
            pts.append([x, y, z])
    return pts


def components(pts, cell):
    grid = {}
    for i in range(len(pts)):
        key = (int(math.floor(pts[i][0] / cell)), int(math.floor(pts[i][1] / cell)))
        if key not in grid:
            grid[key] = []
        grid[key].append(i)
    seen = set()
    comps = []
    for key in grid:
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        members = []
        while len(stack) > 0:
            cur = stack.pop()
            members.extend(grid[cur])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cur[0] + dx, cur[1] + dy)
                    if nb in grid and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        comps.append(members)
    return comps


def centroid(pts, idx):
    n = len(idx)
    if n == 0:
        return None
    return [sum(pts[i][0] for i in idx) / n, sum(pts[i][1] for i in idx) / n,
            sum(pts[i][2] for i in idx) / n]


def quant(values, frac):
    ordered = sorted(values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * frac)]


# ---------------------------------------------------------------- tracker

def track_frame(depth):
    t = S["tick"]
    TR["frames"] += 1
    if not TR["init"]:
        pts = band_points(depth, 0.045, 0.16)
        comps = [c for c in components(pts, 0.012) if len(c) >= 12]
        tops = []
        for c in comps:
            tops.append(quant([pts[i][2] for i in c], 0.95))
        mtop = quant(tops, 0.5) if len(tops) > 0 else 0.0
        TR["cand_hist"].append([len(comps), mtop])
        h = TR["cand_hist"]
        stable = len(h) >= 3 and h[-2][0] == len(comps) and h[-3][0] == len(comps) and \
            abs(h[-2][1] - mtop) < 0.004 and abs(h[-3][1] - mtop) < 0.004
        if len(comps) >= 3 and stable:
            TR["ctop"] = mtop
            for c in comps:
                top_idx = [i for i in c if pts[i][2] >= TR["ctop"] - 0.02]
                cen = centroid(pts, top_idx if len(top_idx) >= 6 else c)
                TR["tracks"].append({"id": len(TR["tracks"]), "x": cen[0], "y": cen[1],
                                     "vx": 0.0, "vy": 0.0, "last": t, "n": len(c),
                                     "hist": [[t, cen[0], cen[1]]], "occl": 0, "pts": []})
            TR["init"] = True
            R["init_tick"] = t
            R["container_top_z"] = rnd(TR["ctop"], 4)
            R["init_tracks"] = [[tr["id"], rnd(tr["x"], 4), rnd(tr["y"], 4), tr["n"]] for tr in TR["tracks"]]
            note("tracker init with " + str(len(comps)) + " containers")
            assign_owners()
        return
    ctop = TR["ctop"]
    pts = band_points(depth, ctop - 0.02, ctop + 0.012)
    tracks = TR["tracks"]
    preds = []
    for tr in tracks:
        gap = max(1, t - tr["last"])
        g = min(gap, 40)
        preds.append([tr["x"] + tr["vx"] * g, tr["y"] + tr["vy"] * g])
    centres = [list(p) for p in preds]
    members = [[] for tr in tracks]
    for it in range(2):
        members = [[] for tr in tracks]
        for i in range(len(pts)):
            best = -1
            bestd = 0.045 * 0.045
            for k in range(len(centres)):
                d = (pts[i][0] - centres[k][0]) ** 2 + (pts[i][1] - centres[k][1]) ** 2
                if d < bestd:
                    bestd = d
                    best = k
            if best >= 0:
                members[best].append(i)
        for k in range(len(centres)):
            if len(members[k]) >= 6:
                c = centroid(pts, members[k])
                centres[k] = [c[0], c[1]]
    maxspeed = 0.0
    for k in range(len(tracks)):
        tr = tracks[k]
        if len(members[k]) >= 6:
            gap = max(1, t - tr["last"])
            ivx = (centres[k][0] - tr["x"]) / gap
            ivy = (centres[k][1] - tr["y"]) / gap
            a = 0.3 if gap <= 3 else 0.6
            tr["vx"] = (1 - a) * tr["vx"] + a * ivx
            tr["vy"] = (1 - a) * tr["vy"] + a * ivy
            tr["x"] = centres[k][0]
            tr["y"] = centres[k][1]
            tr["last"] = t
            tr["n"] = len(members[k])
            tr["occl"] = 0
            tr["pts"] = [pts[i] for i in members[k]]
        else:
            tr["occl"] += 1
        tr["hist"].append([t, tr["x"], tr["y"]])
        if len(tr["hist"]) > 60:
            tr["hist"] = tr["hist"][-60:]
        sp = math.sqrt(tr["vx"] ** 2 + tr["vy"] ** 2)
        if sp > maxspeed:
            maxspeed = sp
    if maxspeed > 0.0004:
        TR["last_motion"] = t
    TR["last_pts_count"] = len(pts)
    if t % 5 == 0 and len(R["timeline"]) < 400:
        R["timeline"].append([t, rnd(maxspeed * 1000, 2), [[tr["id"], rnd(tr["x"], 3), rnd(tr["y"], 3),
                                                         len(tr["pts"]) if tr["last"] == t else 0]
                                                        for tr in tracks]])


def assign_owners():
    R["owner_dist"] = {}
    used = set()
    pairs = []
    for name in CUBES:
        c = CUBES[name]
        for tr in TR["tracks"]:
            d = math.sqrt((tr["x"] - c[0]) ** 2 + (tr["y"] - c[1]) ** 2)
            pairs.append([d, name, tr["id"]])
    owner = {}
    for item in sorted(pairs):
        if item[1] in owner or item[2] in used:
            continue
        if item[0] > 0.06:
            continue
        owner[item[1]] = item[2]
        used.add(item[2])
        R["owner_dist"][item[1]] = rnd(item[0], 4)
    OWNER.clear()
    for name in owner:
        OWNER[name] = owner[name]
    R["owner"] = dict(owner)


def track_by_id(tid):
    for tr in TR["tracks"]:
        if tr["id"] == tid:
            return tr
    return None


def min_clearance(bx, by, ahead):
    best = 9.0
    t = S["tick"]
    for tr in TR["tracks"]:
        gap = max(0, t - tr["last"])
        for extra in (0, ahead):
            g = min(gap + extra, 60)
            x = tr["x"] + tr["vx"] * g
            y = tr["y"] + tr["vy"] * g
            d = math.sqrt((x - bx) ** 2 + (y - by) ** 2)
            if d < best:
                best = d
    return best


def static_for(window, tol):
    t = S["tick"]
    for tr in TR["tracks"]:
        old = None
        for h in tr["hist"]:
            if h[0] <= t - window:
                old = h
        if old is None:
            return False
        if math.sqrt((tr["x"] - old[1]) ** 2 + (tr["y"] - old[2]) ** 2) > tol:
            return False
    return True


# ---------------------------------------------------------------- colour helpers (t=0 cubes)

def rgb_to_hsv(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = mx - mn
    v = mx / 255.0
    s = 0.0 if mx <= 0 else d / mx
    if d <= 0:
        h = 0.0
    elif mx == r:
        h = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        h = 60.0 * ((b - r) / d + 2.0)
    else:
        h = 60.0 * ((r - g) / d + 4.0)
    return h, s, v


def hue_name(h):
    best = None
    bestd = 1000.0
    for item in HUES:
        d = abs(h - item[1])
        d = min(d, 360.0 - d)
        if d < bestd:
            bestd = d
            best = item[0]
    return best


def box_colour(rgb, box):
    x1, y1, x2, y2 = box
    hh = len(rgb)
    ww = len(rgb[0])
    sat = []
    vals = []
    for iy in range(5):
        for ix in range(5):
            px = min(max(int(x1 + (x2 - x1) * (0.3 + 0.1 * ix)), 0), ww - 1)
            py = min(max(int(y1 + (y2 - y1) * (0.3 + 0.1 * iy)), 0), hh - 1)
            pix = rgb[py][px]
            h, s, v = rgb_to_hsv(float(pix[0]), float(pix[1]), float(pix[2]))
            vals.append(v)
            if s > 0.3 and v > 0.15:
                sat.append(h)
    if len(sat) >= 4:
        cx = sum(math.cos(math.radians(h)) for h in sat)
        cy = sum(math.sin(math.radians(h)) for h in sat)
        return hue_name(math.degrees(math.atan2(cy, cx)) % 360.0)
    v = sorted(vals)[len(vals) // 2]
    return "black" if v < 0.2 else ("white" if v > 0.7 else "gray")


def parse_targets(language):
    text = language.lower()
    for ch in ",.;:!?()\"'":
        text = text.replace(ch, " ")
    words = [SYN[w] if w in SYN else w for w in text.split()]
    targets = []
    for i in range(len(words)):
        if words[i] in COLOUR_WORDS and "button" not in words[i + 1:i + 3] and words[i] not in targets:
            targets.append(words[i])
    return targets


def cloud_xy(cand):
    try:
        pc = mask_to_point_cloud(cand, CAM, target_frame=S["base"])
        pts = [[float(p[0]), float(p[1]), float(p[2])] for p in pc.points]
    except Exception as exc:
        err("mask_to_point_cloud", exc)
        return None
    if len(pts) < 10:
        return None
    zs = [p[2] for p in pts]
    top = quant(zs, 0.95)
    tp = [p for p in pts if p[2] >= top - 0.012]
    return {"x": quant([p[0] for p in tp], 0.5), "y": quant([p[1] for p in tp], 0.5), "top": top,
            "zspan": top - quant(zs, 0.05), "xspan": quant([p[0] for p in pts], 0.95) - quant([p[0] for p in pts], 0.05),
            "yspan": quant([p[1] for p in pts], 0.95) - quant([p[1] for p in pts], 0.05), "n": len(pts)}


CUBES = {}
OWNER = {}


def seg(prompt):
    try:
        found = segment_text(CAM, prompt)
        if not found.ok:
            return []
        out = []
        for cand in found.segmentations[:8]:
            try:
                out.append({"cand": cand, "box": [float(v) for v in cand.box_xyxy], "score": float(cand.score)})
            except Exception:
                pass
        return out
    except Exception as exc:
        err("segment_text " + prompt, exc)
        return []


def locate_cubes(targets, rgb):
    info = {}
    for name in targets:
        best = None
        for d in seg(name + " cube")[:5]:
            w = d["box"][2] - d["box"][0]
            h = d["box"][3] - d["box"][1]
            if d["score"] < 0.2 or w > 40 or h > 40:
                continue
            col = box_colour(rgb, d["box"])
            if col != name:
                continue
            if best is None or d["score"] > best["score"]:
                best = d
        if best is None:
            info[name] = None
            continue
        c = cloud_xy(best["cand"])
        info[name] = {"score": rnd(best["score"], 3), "box": [rnd(v, 1) for v in best["box"]],
                      "cloud": None if c is None else {"x": rnd(c["x"], 4), "y": rnd(c["y"], 4), "top": rnd(c["top"], 4)}}
        if c is not None:
            CUBES[name] = [c["x"], c["y"]]
    R["cubes_t0"] = info


def locate_buttons():
    found = []
    for d in seg("round button")[:6]:
        if d["score"] < 0.1:
            continue
        c = cloud_xy(d["cand"])
        if c is None or c["zspan"] > 0.07 or c["xspan"] > 0.16 or c["yspan"] > 0.16 or c["top"] > 0.08:
            continue
        dup = False
        for f in found:
            if (f["x"] - c["x"]) ** 2 + (f["y"] - c["y"]) ** 2 < 0.03 * 0.03:
                dup = True
        if not dup:
            found.append({"x": c["x"], "y": c["y"], "top": c["top"], "score": d["score"]})
    found = found[:2]
    R["buttons"] = [{"x": rnd(b["x"], 4), "y": rnd(b["y"], 4), "top": rnd(b["top"], 4),
                     "score": rnd(b["score"], 3)} for b in found]
    return found


# ---------------------------------------------------------------- motion

def ik(pos, quat, seed):
    try:
        res = solve_ik(Pose([float(pos[0]), float(pos[1]), float(pos[2])], quat, S["base"]), seed_joints=seed)
        if not res.ok:
            note("ik failed at " + str([rnd(v, 3) for v in pos]))
            return None
        return as_joint_list(res.joint_positions)
    except Exception as exc:
        err("solve_ik", exc)
        return None


def servo(goal, grip, max_delta, settle_max, tol, observe):
    if S["terminal"] or goal is None:
        return False
    start = joints_now()
    diff = max(abs(goal[j] - start[j]) for j in range(7))
    n = max(1, int(math.ceil(diff / max_delta)))
    for i in range(1, n + 1):
        if not step_joints([start[j] + (goal[j] - start[j]) * i / n for j in range(7)], grip, observe):
            return False
    for k in range(settle_max):
        cur = joints_now()
        if max(abs(goal[j] - cur[j]) for j in range(7)) < tol:
            break
        if not step_joints(goal, grip, observe):
            return False
    return True


def hold_at(q, nticks, grip, observe_every):
    for k in range(nticks):
        if not step_joints(q, grip, (S["tick"] % observe_every) == 0):
            return False
    return True


def track_radius(tr):
    pts = tr["pts"]
    if len(pts) < 8:
        return 0.035
    ds = [math.sqrt((p[0] - tr["x"]) ** 2 + (p[1] - tr["y"]) ** 2) for p in pts]
    return max(0.025, min(0.06, quant(ds, 0.95)))


def hand_clearance(bx, by, theta, ahead, skip):
    """Min gap between the hand footprint (rectangle, finger axis (sin t, -cos t)) centred at
    (bx, by) and every tracked container disk, now and predicted up to `ahead` ticks."""
    fx = math.sin(theta)
    fy = -math.cos(theta)
    best = 9.0
    t = S["tick"]
    for tr in TR["tracks"]:
        if tr["id"] == skip:
            continue
        gap = max(0, t - tr["last"])
        rc = track_radius(tr) + (min(0.05, 0.003 * gap) if gap > 2 else 0.0)
        for frac in (0.0, 0.5, 1.0):
            g = min(gap, 10) + ahead * frac
            dx = tr["x"] + tr["vx"] * g - bx
            dy = tr["y"] + tr["vy"] * g - by
            u = abs(dx * fx + dy * fy)
            w = abs(-dx * fy + dy * fx)
            eu = max(u - HAND_HL, 0.0)
            ew = max(w - HAND_HW, 0.0)
            d = math.sqrt(eu * eu + ew * ew) - rc
            if d < best:
                best = d
    return best


def best_press_yaw(bx, by, ahead):
    best = None
    for k in range(12):
        th = math.radians(-90.0 + 15.0 * k)
        c = hand_clearance(bx, by, th, ahead, -1)
        if best is None or c > best[0] + 0.002:
            best = [c, th]
    return best


def press_button(b, idx):
    rec = {"button": [rnd(b["x"], 4), rnd(b["y"], 4)], "tick_start": S["tick"], "yaw_changes": 0}
    R["presses"].append(rec)
    force = PRESS_FORCE[min(idx, len(PRESS_FORCE) - 1)]
    ctop = TR["ctop"] if TR["ctop"] is not None else b["top"] + 0.06
    z_low = max(b["top"] + 0.025, ctop - HAND_BOTTOM + 0.02)
    rec["z_low"] = rnd(z_low, 4)
    cand = best_press_yaw(b["x"], b["y"], PRESS_AHEAD)
    theta = cand[1]
    q_high = ik([b["x"], b["y"], z_low + 0.07], yaw_quat(theta), S["q_home"])
    if q_high is None:
        theta = 0.0
        q_high = ik([b["x"], b["y"], z_low + 0.07], S["down_q"], S["q_home"])
    if q_high is None:
        rec["error"] = "hover ik"
        return False
    servo(q_high, 0.0, 0.10, 4, 0.04, True)
    q_hover = ik([b["x"], b["y"], z_low], yaw_quat(theta), q_high)
    if q_hover is None:
        q_hover = q_high
    servo(q_hover, 0.0, 0.08, 4, 0.03, True)
    rec["tick_hover"] = S["tick"]
    last_change = S["tick"]
    while not S["terminal"]:
        cur = hand_clearance(b["x"], b["y"], theta, PRESS_AHEAD, -1)
        if cur >= CLEAR_MARGIN:
            rec["clear"] = rnd(cur, 4)
            break
        if S["tick"] >= force:
            rec["forced"] = True
            rec["clear"] = rnd(cur, 4)
            break
        if S["tick"] - last_change >= 10 and S["tick"] + 14 < force and rec["yaw_changes"] < 3:
            cand = best_press_yaw(b["x"], b["y"], PRESS_AHEAD + 8)
            if cand[0] > cur + 0.015 and abs(cand[1] - theta) > math.radians(20):
                qh = ik([b["x"], b["y"], z_low], yaw_quat(cand[1]), q_hover)
                if qh is not None:
                    theta = cand[1]
                    q_hover = qh
                    rec["yaw_changes"] += 1
                    last_change = S["tick"]
                    servo(q_hover, 0.0, 0.12, 3, 0.04, True)
                    continue
        if not step_joints(q_hover, 0.0, True):
            break
    if S["terminal"]:
        rec["terminal_before_press"] = True
        return False
    rec["yaw_deg"] = rnd(math.degrees(theta), 1)
    rec["tick_descend"] = S["tick"]
    rec["clear_now"] = rnd(hand_clearance(b["x"], b["y"], theta, 0, -1), 4)
    q_low = ik([b["x"], b["y"], b["top"] - 0.010], yaw_quat(theta), q_hover)
    if q_low is None:
        rec["error"] = "ik press"
        return False
    start = joints_now()
    n = 3
    depressed = False
    for i in range(1, n + 8):
        f = min(1.0, i / float(n))
        if not step_joints([start[j] + (q_low[j] - start[j]) * f for j in range(7)], 0.0, True):
            break
        if S["ee"] is not None and S["ee"][2] <= b["top"] - 0.006:
            depressed = True
            break
    rec["tick_pressed"] = S["tick"]
    rec["ee_low"] = None if S["ee"] is None else [rnd(v, 4) for v in S["ee"]]
    rec["depressed"] = depressed
    rec["clear_at_press"] = rnd(hand_clearance(b["x"], b["y"], theta, 0, -1), 4)
    if S["terminal"]:
        rec["terminal_at_press"] = True
        return False
    for k in range(4):
        if not step_joints(q_hover, 0.0, True):
            break
    servo(q_high, 0.0, 0.10, 3, 0.04, True)
    rec["tick_up"] = S["tick"]
    return True


def yaw_quat(theta):
    return [0.0, math.cos(theta / 2.0), math.sin(theta / 2.0), 0.0]


def side_angle(pts):
    """Angle (rad, in [-pi/4, pi/4)) of a side normal of the min-area box of the XY points."""
    if len(pts) < 8:
        return 0.0
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    best = None
    for k in range(18):
        a = math.radians(-45.0 + 5.0 * k)
        ca = math.cos(a)
        sa = math.sin(a)
        us = [(p[0] - mx) * ca + (p[1] - my) * sa for p in pts]
        vs = [-(p[0] - mx) * sa + (p[1] - my) * ca for p in pts]
        area = (quant(us, 0.97) - quant(us, 0.03)) * (quant(vs, 0.97) - quant(vs, 0.03))
        if best is None or area < best[0]:
            best = [area, a]
    return best[1]


def refresh_frames(n):
    q = joints_now()
    for k in range(n):
        if not step_joints(q, S["grip_hold"], True):
            return False
    return True


def pick(target, last):
    rec = {"target": target, "tick_start": S["tick"]}
    R["picks"].append(rec)
    tid = OWNER.get(target)
    rec["track"] = tid
    tr = track_by_id(tid) if tid is not None else None
    if tr is None:
        rec["error"] = "no owner track"
        return False
    x = tr["x"]
    y = tr["y"]
    a = side_angle(tr["pts"])
    opts = []
    for th in (a, a + math.pi / 2.0):
        if th >= math.pi / 2.0:
            th -= math.pi
        opts.append([hand_clearance(x, y, th, 0, tid), th])
    opts = sorted(opts)
    theta = opts[-1][1]
    rec["yaw_opts"] = [[rnd(o[0], 4), rnd(math.degrees(o[1]), 1)] for o in opts]
    quat = yaw_quat(theta)
    top = TR["ctop"]
    rec["pose"] = [rnd(x, 4), rnd(y, 4), rnd(top, 4), rnd(math.degrees(theta), 1), len(tr["pts"])]
    gz = top - 0.018
    record_exec("open", open_gripper())
    if S["terminal"]:
        return False
    record_exec("pre", move_to_pose(Pose([x, y, top + 0.10], quat, S["base"]), tolerance=0.015, max_steps=250))
    if S["terminal"]:
        return False
    record_exec("grasp", move_to_pose(Pose([x, y, gz], quat, S["base"]), tolerance=0.01, max_steps=200))
    if S["terminal"]:
        return False
    record_exec("close", close_gripper())
    if S["terminal"]:
        return False
    st = sync_tick()
    if st is not None:
        rec["grip_closed"] = rnd(st.gripper_positions["primary"], 3)
    record_exec("lift", move_to_pose(Pose([x, y, top + 0.12], quat, S["base"]), tolerance=0.02, max_steps=200))
    if S["terminal"]:
        rec["terminal_after_lift"] = True
        return False
    S["grip_hold"] = 0.0
    q = joints_now()
    hold_at(q, 8, 0.0, 1000)
    if S["terminal"]:
        rec["terminal_after_hold"] = True
        return False
    st = sync_tick()
    if st is not None:
        rec["grip_lifted"] = rnd(st.gripper_positions["primary"], 3)
    rec["tick_lifted"] = S["tick"]
    if last:
        hold_at(q, 20, 0.0, 1000)
        return True
    record_exec("down", move_to_pose(Pose([x, y, gz + 0.006], quat, S["base"]), tolerance=0.015, max_steps=200))
    if S["terminal"]:
        return False
    record_exec("release", open_gripper())
    if S["terminal"]:
        return False
    S["grip_hold"] = 1.0
    record_exec("retreat", move_to_pose(Pose([x, y, top + 0.10], quat, S["base"]), tolerance=0.02, max_steps=200))
    if S["terminal"]:
        return False
    hold_at(joints_now(), 5, 1.0, 1000)
    rec["tick_end"] = S["tick"]
    return True


# ---------------------------------------------------------------- main

def main():
    ctx = get_task_context()
    R["task_language"] = str(ctx.language)
    try:
        meta = robomme_plus.get_task_metadata()
        R["max_steps"] = int(meta["max_steps"])
    except Exception as exc:
        err("metadata", exc)
        R["max_steps"] = 1312
    targets = parse_targets(R["task_language"])
    R["targets"] = targets
    st = get_robot_state()
    S["emb"] = st.embodiment
    S["base"] = st.base_frame
    S["q_home"] = [float(v) for v in st.joint_positions["primary"]]
    S["ee"] = [float(v) for v in st.end_effector_poses["primary"].position]
    S["grip_hold"] = 0.0
    q = [float(v) for v in st.end_effector_poses["primary"].quaternion_wxyz]
    w, x, y, z = q
    if 1 - 2 * (x * x + y * y) < -0.9:
        S["down_q"] = q
    obs = get_observation(camera_names=[CAM])
    cam = obs.cameras[CAM]
    build_rays(cam)
    buttons = locate_buttons()
    locate_cubes(targets, cam.rgb)
    note("t0 analysed: buttons " + str(len(buttons)) + " cubes " + str(list(CUBES.keys())))
    # Phase A: watch the containers arrive and start moving, arm at home (clean view)
    S["phase"] = "watch"
    hold_at(S["q_home"], 30, 0.0, 1)
    R["tracks_at_30"] = [[tr["id"], rnd(tr["x"], 4), rnd(tr["y"], 4)] for tr in TR["tracks"]]
    # Phase B: press both buttons once each, only when the button neighbourhood is clear
    S["phase"] = "press"
    order = sorted([[-best_press_yaw(b["x"], b["y"], PRESS_AHEAD)[0], i] for i, b in enumerate(buttons)])
    for k in range(len(order)):
        if S["terminal"]:
            break
        press_button(buttons[order[k][1]], k)
    if not S["terminal"]:
        servo(S["q_home"], 0.0, 0.08, 10, 0.02, True)
    note("presses done")
    # Phase C: wait at home until the containers stop moving
    S["phase"] = "wait"
    cap = R["max_steps"] - 160 * max(1, len(targets)) - 20
    R["wait_cap"] = cap
    while not S["terminal"] and S["tick"] < cap:
        quiet = S["tick"] - TR["last_motion"]
        if static_for(30, 0.006) and quiet > 20 and (S["tick"] >= MOTION_END or quiet > 100):
            note("containers static")
            break
        if not step_joints(S["q_home"], 0.0, (S["tick"] % 2) == 0):
            break
    R["wait_end"] = S["tick"]
    if not S["terminal"]:
        refresh_frames(2)
    R["tracks_final"] = [[tr["id"], rnd(tr["x"], 4), rnd(tr["y"], 4), tr["n"], tr["occl"], tr["last"]]
                         for tr in TR["tracks"]]
    # Phase D: pick containers in the instructed order
    S["phase"] = "pick"
    for i in range(len(targets)):
        if S["terminal"]:
            break
        pick(targets[i], i == len(targets) - 1)
        if not S["terminal"] and i < len(targets) - 1:
            servo(S["q_home"], 1.0, 0.08, 10, 0.03, True)
    R["phase_end"] = S["phase"]


try:
    main()
except Exception as exc:
    err("main", exc)
R["terminal_seen"] = S["terminal"]
R["final_tick"] = S["tick"]
R["frames_tracked"] = TR["frames"]
result = R
