# v05 - PickHighlight: v04 + (a) lost-track re-acquisition: a cube blob not explained by any
# track is given to the nearest lost track (a pairwise swap hidden by the arm leaves exactly that
# signature), (b) leave home only right after a swap ended (or when the measured quiet gap still
# covers a full excursion), (c) faster excursions (press ~50, pick ~60 ticks) so an excursion fits
# the quiet gap (ep1: swaps every ~100 ticks, quiet gap ~64), (d) per-tick tracked gripper close/open,
# (e) skip/postpone a target whose track is lost or ambiguous at departure.
# v04 - PickHighlight: v03 + IK branch selection (cube 4-fold yaw symmetry) + robust pick-cycle /
# tick-budget scheduling. v03 succeeded on ep0, but one pick took 186 ticks (IK jumped branch on a
# 6 cm descent) and the inflated cycle estimate then cost a 400-tick wait.
# v03 - PickHighlight: watch-and-track program synchronised with the periodic swaps.
# v02 (ep0) showed: (a) its disc detector counted 0 white pixels although the observation RGB had
# the discs (depth/height gate too strict), (b) swaps recur periodically (~every 140 ticks,
# ~38 ticks long) for the whole episode. v03 gates disc pixels by height < 3 cm (invalid depth ->
# table plane), measures the swap period at home, and only leaves home for a press or one
# pick when the predicted quiet window is long enough.
# Evidence from v01 (ep0 video): white discs appear under some cubes ~0.5 s into the episode
# for ~2.2 s, then cubes swap positions pairwise in ~2 s events (two events seen, 7 s apart).
# Lifting a non-highlighted cube terminates the episode as failure.
# Strategy: hold still at home (the arm is outside the front view), track every cube each
# control tick from front RGB-D (height above the observed table plane), record which tracks
# get a white disc, keep tracking through swaps until the scene is quiet, then press the
# button, pick + replace every highlighted track (moving by IK + per-tick joint interpolation
# so tracking continues during motion), and press the button again.
# Public API only; no hardcoded world coordinates or pixels.
import math

CAM = "front"
report = {"stage": "start", "errors": []}


def note_error(where, err):
    if len(report["errors"]) < 40:
        report["errors"].append(where + ": " + str(err)[:300])


def to_list(arr):
    try:
        return arr.tolist()
    except Exception:
        out = []
        for row in arr:
            out.append([v for v in row])
        return out


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def mat_from_quat(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def cam_model(obs, cam):
    c = obs.cameras[cam]
    K = to_list(c.intrinsics)
    pose = c.camera_pose
    return {"fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]),
            "cy": float(K[1][2]), "R": mat_from_quat(pose.quaternion_wxyz),
            "t": [float(v) for v in pose.position]}


def project(m, u, v, d):
    xc = (u + 0.5 - m["cx"]) / m["fx"] * d
    yc = (v + 0.5 - m["cy"]) / m["fy"] * d
    R = m["R"]
    t = m["t"]
    return (R[0][0] * xc + R[0][1] * yc + R[0][2] * d + t[0],
            R[1][0] * xc + R[1][1] * yc + R[1][2] * d + t[1],
            R[2][0] * xc + R[2][1] * yc + R[2][2] * d + t[2])


def res_info(res):
    d = {}
    try:
        d["ok"] = bool(res.ok)
        d["status"] = str(res.status)
        d["steps"] = int(res.steps_executed)
    except Exception:
        pass
    return d


def is_terminal(res):
    try:
        return bool(res.terminated) or bool(res.truncated)
    except Exception:
        return False


# ---------------- metadata ----------------
context = get_task_context()
state0 = get_robot_state()
base_frame = state0.base_frame
embodiment = state0.embodiment
report["language"] = context.language
max_steps = 2000
try:
    max_steps = int(robomme_plus.get_task_metadata()["max_steps"])
except Exception as err:
    note_error("meta", err)
report["max_steps"] = max_steps
home_quat = [float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz]
home_joints = [float(v) for v in state0.joint_positions["primary"]]
flags = {"terminal": False, "tick": 0, "cycle": 80}


def yaw_quat(delta_deg):
    a = math.radians(delta_deg) / 2.0
    w1, x1, y1, z1 = math.cos(a), 0.0, 0.0, math.sin(a)
    w2, x2, y2, z2 = home_quat
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


# ---------------- initial full-frame detection ----------------
obs0 = get_observation()
M = cam_model(obs0, CAM)
dep0 = to_list(obs0.cameras[CAM].depth_m)
H = len(dep0)
W = len(dep0[0])
zs = []
for v in range(0, H, 8):
    for u in range(0, W, 8):
        d = float(dep0[v][u])
        if d > 0.05 and d < 3.0:
            zs.append(project(M, u, v, d)[2])
table_z = quantile(zs, 0.5)
report["table_z"] = table_z


def plane_point(u, v):
    """Intersect the pixel ray with the observed table plane (used when depth is invalid)."""
    r = project(M, u, v, 1.0)
    t0 = M["t"]
    dz = r[2] - t0[2]
    if abs(dz) < 1e-6:
        return None
    s = (table_z - t0[2]) / dz
    if s <= 0:
        return None
    return (t0[0] + (r[0] - t0[0]) * s, t0[1] + (r[1] - t0[1]) * s, table_z)


def height_points(dep, u0, v0, u1, v1, stride, hmin, hmax):
    out = []
    for v in range(max(0, v0), min(H, v1), stride):
        drow = dep[v]
        for u in range(max(0, u0), min(W, u1), stride):
            d = float(drow[u])
            if d <= 0.05 or d >= 3.0:
                continue
            p = project(M, u, v, d)
            h = p[2] - table_z
            if h < hmin or h > hmax:
                continue
            if p[0] < 0.05 or p[0] > 1.0 or abs(p[1]) > 0.6:
                continue
            out.append((p[0], p[1], p[2], u, v))
    return out


def components(pts, cell):
    cells = {}
    for i in range(len(pts)):
        key = (int(math.floor(pts[i][0] / cell)), int(math.floor(pts[i][1] / cell)))
        if key in cells:
            cells[key].append(i)
        else:
            cells[key] = [i]
    seen = set()
    comps = []
    for key in cells:
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        members = []
        while stack:
            k = stack.pop()
            members.extend(cells[k])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nk = (k[0] + dx, k[1] + dy)
                    if nk in cells and nk not in seen:
                        seen.add(nk)
                        stack.append(nk)
        comps.append(members)
    return comps


def extent(pts, members, axis):
    vals = [pts[i][axis] for i in members]
    return quantile(vals, 0.97) - quantile(vals, 0.03)


def top_center(pts, members):
    zmax = quantile([pts[i][2] for i in members], 0.97)
    top = [i for i in members if pts[i][2] >= zmax - 0.008]
    if len(top) < 3:
        top = members
    return [quantile([pts[i][0] for i in top], 0.5), quantile([pts[i][1] for i in top], 0.5), zmax, top]


def split_members(pts, members):
    xs = [pts[i][0] for i in members]
    ys = [pts[i][1] for i in members]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) * (x - mx) for x in xs)
    syy = sum((y - my) * (y - my) for y in ys)
    sxy = sum((xs[k] - mx) * (ys[k] - my) for k in range(len(xs)))
    ang = 0.5 * math.atan2(2 * sxy, sxx - syy)
    ax = math.cos(ang)
    ay = math.sin(ang)
    proj = sorted(((pts[i][0] - mx) * ax + (pts[i][1] - my) * ay, i) for i in members)
    span = proj[-1][0] - proj[0][0]
    k = int(round(span / 0.045))
    if k < 2:
        return [members]
    if k > 4:
        k = 4
    lo = proj[0][0]
    groups = [[] for g in range(k)]
    for val, i in proj:
        g = int((val - lo) / (span / k + 1e-9))
        if g >= k:
            g = k - 1
        groups[g].append(i)
    return [g for g in groups if g]


# button from SAM (prompt "button"), refined with height points near it
button = None
for prompt in ("button", "round button", "white button"):
    try:
        loc = localize_object(prompt, camera_name=CAM, target_frame=base_frame)
    except Exception as err:
        note_error("button_loc", err)
        continue
    if loc.ok and len(loc.point_cloud.points) >= 10:
        p = loc.geometry.pose.position
        button = {"prompt": prompt, "x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
        break
report["button_sam"] = button

full_pts = height_points(dep0, 0, 0, W, H, 1, 0.012, 0.15)
objects = []
for comp in components(full_pts, 0.01):
    if len(comp) < 12:
        continue
    parts = [comp]
    if extent(full_pts, comp, 0) > 0.065 or extent(full_pts, comp, 1) > 0.065:
        parts = split_members(full_pts, comp)
    for part in parts:
        if len(part) < 12:
            continue
        tc = top_center(full_pts, part)
        objects.append({"x": tc[0], "y": tc[1], "top": tc[2], "h": tc[2] - table_z,
                        "xe": extent(full_pts, part, 0), "ye": extent(full_pts, part, 1),
                        "n": len(part),
                        "u0": min(full_pts[i][3] for i in part), "u1": max(full_pts[i][3] for i in part),
                        "v0": min(full_pts[i][4] for i in part), "v1": max(full_pts[i][4] for i in part)})

# button refinement: the cap top = height points within 4 cm of SAM center
if button is not None:
    near = [p for p in full_pts if (p[0] - button["x"]) ** 2 + (p[1] - button["y"]) ** 2 < 0.04 ** 2]
    if len(near) >= 5:
        bz = quantile([p[2] for p in near], 0.97)
        cap = [p for p in near if p[2] >= bz - 0.006]
        button["top"] = bz
        button["cx"] = quantile([p[0] for p in cap], 0.5)
        button["cy"] = quantile([p[1] for p in cap], 0.5)
    else:
        button["top"] = button["z"] + 0.005
        button["cx"] = button["x"]
        button["cy"] = button["y"]

tracks = []
for o in objects:
    if button is not None and (o["x"] - button["x"]) ** 2 + (o["y"] - button["y"]) ** 2 < 0.045 ** 2:
        continue
    if o["h"] < 0.02 or o["h"] > 0.08:
        continue
    if o["xe"] > 0.09 or o["ye"] > 0.09 or o["xe"] < 0.015 or o["ye"] < 0.015:
        continue
    tracks.append({"id": len(tracks), "x": o["x"], "y": o["y"], "top": o["top"], "vx": 0.0, "vy": 0.0,
                   "white": 0, "lost": 0, "picked": False, "x0": o["x"], "y0": o["y"],
                   "u0": o["u0"], "u1": o["u1"], "v0": o["v0"], "v1": o["v1"]})
report["objects"] = [{k: (round(v, 4) if k in ("x", "y", "top", "h", "xe", "ye") else v) for k, v in o.items()}
                     for o in objects]
report["n_tracks"] = len(tracks)

roi = [W, H, 0, 0]
for t in tracks:
    roi[0] = min(roi[0], t["u0"])
    roi[1] = min(roi[1], t["v0"])
    roi[2] = max(roi[2], t["u1"])
    roi[3] = max(roi[3], t["v1"])
roi = [max(0, roi[0] - 30), max(0, roi[1] - 30), min(W, roi[2] + 31), min(H, roi[3] + 31)]
report["roi"] = roi

hl = {"peak": 0, "on_tick": None, "off_tick": None, "low_run": 0, "frames": 0}
motion = {"last_motion_tick": None, "events": [], "in_event": False, "n_motion_ticks": 0, "enabled": False,
          "enabled_since": None}
reacq_log = []
work = [0.0, 0.0, 0.0, 0.0]
if tracks:
    work = [min(t["x"] for t in tracks) - 0.08, max(t["x"] for t in tracks) + 0.08,
            min(t["y"] for t in tracks) - 0.08, max(t["y"] for t in tracks) + 0.08]


def track_frame(obs):
    """Update every track from one observation; also accumulate highlight-disc evidence."""
    tick = flags["tick"]
    dep = to_list(obs.cameras[CAM].depth_m)
    rgb = None
    need_rgb = hl["off_tick"] is None
    if need_rgb:
        rgb = to_list(obs.cameras[CAM].rgb)
    pts = height_points(dep, roi[0], roi[1], roi[2], roi[3], 2, 0.012, 0.09)
    keep = []
    for comp in components(pts, 0.01):
        if extent(pts, comp, 0) > 0.13 or extent(pts, comp, 1) > 0.13:
            continue  # the arm, not a cube
        keep.extend(comp)
    # point-level assignment to the nearest predicted track
    assigned = [[] for t in tracks]
    preds = [(t["x"] + t["vx"], t["y"] + t["vy"]) for t in tracks]
    for i in keep:
        p = pts[i]
        best = None
        bd = 0.04 * 0.04
        for k in range(len(tracks)):
            dx = p[0] - preds[k][0]
            dy = p[1] - preds[k][1]
            dd = dx * dx + dy * dy
            if dd < bd:
                bd = dd
                best = k
        if best is not None:
            assigned[best].append(i)
    moved = 0.0
    for k in range(len(tracks)):
        t = tracks[k]
        if len(assigned[k]) >= 4:
            tc = top_center(pts, assigned[k])
            nvx = tc[0] - t["x"]
            nvy = tc[1] - t["y"]
            step_len = math.sqrt(nvx * nvx + nvy * nvy)
            if step_len > moved and t["lost"] == 0:
                moved = step_len
            t["vx"] = 0.5 * t["vx"] + 0.5 * nvx
            t["vy"] = 0.5 * t["vy"] + 0.5 * nvy
            t["x"] = tc[0]
            t["y"] = tc[1]
            t["top"] = tc[2]
            t["lost"] = 0
        else:
            t["x"] = preds[k][0]
            t["y"] = preds[k][1]
            t["vx"] = 0.5 * t["vx"]
            t["vy"] = 0.5 * t["vy"]
            t["lost"] += 1
    # re-acquisition: cube-sized blobs of unassigned points go to the nearest lost track
    lost_ids = [k for k in range(len(tracks)) if tracks[k]["lost"] >= 2]
    # only with the arm parked at home (out of view) for a few ticks: no gripper blobs
    es = motion["enabled_since"]
    if lost_ids and motion["enabled"] and es is not None and tick - es >= 2:
        used = set()
        for k in range(len(tracks)):
            for i in assigned[k]:
                used.add(i)
        free = [pts[i] for i in keep if i not in used]
        if len(free) >= 6:
            for comp in components(free, 0.01):
                if len(comp) < 6 or not lost_ids:
                    continue
                ex = extent(free, comp, 0)
                ey = extent(free, comp, 1)
                if ex > 0.07 or ey > 0.07 or (ex < 0.012 and ey < 0.012):
                    continue
                tc = top_center(free, comp)
                if tc[2] - table_z < 0.025 or tc[2] - table_z > 0.06:
                    continue
                if tc[0] < work[0] or tc[0] > work[1] or tc[1] < work[2] or tc[1] > work[3]:
                    continue
                if button is not None and (tc[0] - button["x"]) ** 2 + (tc[1] - button["y"]) ** 2 < 0.05 ** 2:
                    continue
                near_track = False
                for t2 in tracks:
                    if t2["lost"] == 0 and (t2["x"] - tc[0]) ** 2 + (t2["y"] - tc[1]) ** 2 < 0.03 ** 2:
                        near_track = True
                if near_track:
                    continue
                best = None
                for k in lost_ids:
                    dd = (tracks[k]["x"] - tc[0]) ** 2 + (tracks[k]["y"] - tc[1]) ** 2
                    if dd < 0.4 * 0.4 and (best is None or dd < best[0]):
                        best = (dd, k)
                if best is None:
                    continue
                t = tracks[best[1]]
                reacq_log.append([tick, t["id"], round(t["x"], 3), round(t["y"], 3), round(tc[0], 3),
                                  round(tc[1], 3), t["lost"]])
                t["x"] = tc[0]
                t["y"] = tc[1]
                t["top"] = tc[2]
                t["vx"] = 0.0
                t["vy"] = 0.0
                t["lost"] = 0
                t["reacq"] = t.get("reacq", 0) + 1
                lost_ids = [k for k in lost_ids if k != best[1]]
    if moved > 0.003 and motion["enabled"]:
        motion["n_motion_ticks"] += 1
        motion["last_motion_tick"] = tick
        if not motion["in_event"]:
            motion["in_event"] = True
            es = motion["enabled_since"]
            clean = es is not None and tick - es >= 3
            motion["events"].append([tick, tick, clean, False])
        else:
            motion["events"][-1][1] = tick
    elif motion["in_event"] and motion["last_motion_tick"] is not None and tick - motion["last_motion_tick"] > 3:
        motion["in_event"] = False
        if motion["enabled"] and motion["events"]:
            motion["events"][-1][3] = True  # end observed from home
    # highlight discs: white, flat (table height) pixels near a track, away from the button
    if need_rgb:
        counts = [0 for t in tracks]
        total = 0
        for v in range(roi[1], roi[3], 2):
            crow = rgb[v]
            drow = dep[v]
            for u in range(roi[0], roi[2], 2):
                c = crow[u]
                lo = min(c[0], c[1], c[2])
                if lo < 200 or max(c[0], c[1], c[2]) - lo > 30:
                    continue
                d = float(drow[u])
                if d <= 0.05 or d >= 3.0:
                    p = plane_point(u, v)
                else:
                    p = project(M, u, v, d)
                if p is None or p[2] - table_z > 0.03:
                    continue
                if button is not None and (p[0] - button["x"]) ** 2 + (p[1] - button["y"]) ** 2 < 0.05 ** 2:
                    continue
                best = None
                bd = 0.06 * 0.06
                for k in range(len(tracks)):
                    dx = p[0] - tracks[k]["x"]
                    dy = p[1] - tracks[k]["y"]
                    dd = dx * dx + dy * dy
                    if dd < bd:
                        bd = dd
                        best = k
                if best is not None:
                    counts[best] += 1
                    total += 1
        if total >= 15:
            if hl["on_tick"] is None:
                hl["on_tick"] = tick
            hl["frames"] += 1
            for k in range(len(tracks)):
                tracks[k]["white"] += counts[k]
            if total > hl["peak"]:
                hl["peak"] = total
                hl["peak_counts"] = counts
            hl["low_run"] = 0
        elif hl["on_tick"] is not None:
            hl["low_run"] += 1
            if hl["low_run"] >= 3:
                hl["off_tick"] = tick


def obs_of(res):
    try:
        o = res.observation
        if o is not None:
            return o
    except Exception:
        pass
    return get_observation()


def tick_hold(grip):
    st = get_robot_state()
    r = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=[float(v) for v in st.joint_positions["primary"]],
        gripper_position=grip, embodiment=embodiment)}))
    flags["tick"] += 1
    if is_terminal(r):
        flags["terminal"] = True
        return r
    try:
        track_frame(obs_of(r))
    except Exception as err:
        note_error("track", err)
    return r


def step_joints(target, grip):
    r = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=target, gripper_position=grip, embodiment=embodiment)}))
    flags["tick"] += 1
    if is_terminal(r):
        flags["terminal"] = True
        return r
    try:
        track_frame(obs_of(r))
    except Exception as err:
        note_error("track", err)
    return r


motion_log = []


def move_joints_tracked(q1, grip, tag, max_dq=0.045, settle=4):
    st = get_robot_state()
    q0 = [float(v) for v in st.joint_positions["primary"]]
    dq = max(abs(q1[k] - q0[k]) for k in range(7))
    n = max(1, int(dq / max_dq) + 1)
    for i in range(1, n + settle + 1):
        a = min(1.0, float(i) / n)
        target = [q0[k] + (q1[k] - q0[k]) * a for k in range(7)]
        step_joints(target, grip)
        if flags["terminal"]:
            break
    motion_log.append({"tag": tag, "ticks": n + settle, "t": flags["tick"]})


def best_ik(pos, yaw_deg):
    """IK for a top-down pose; the cube is 4-fold symmetric, so try yaw, yaw+-90, yaw+180 and
    keep the solution with the smallest joint change (prefer the requested yaw if it is small)."""
    st = get_robot_state()
    q0 = [float(v) for v in st.joint_positions["primary"]]
    best = None
    for off in (0, 90, -90, 180):
        yaw = yaw_deg + off
        try:
            ik = solve_ik(Pose(pos, yaw_quat(yaw), base_frame), seed_joints=st.joint_positions["primary"])
        except Exception as err:
            note_error("ik", err)
            continue
        if not ik.ok:
            continue
        q1 = [float(v) for v in ik.joint_positions]
        dq = max(abs(q1[k] - q0[k]) for k in range(7))
        if off == 0 and dq < 0.6:
            return (dq, yaw, q1)
        if best is None or dq < best[0]:
            best = (dq, yaw, q1)
    return best


def move_top_down(pos, yaw_deg, grip, tag, max_dq=0.045, settle=4):
    """Top-down move with branch selection; returns the yaw actually used."""
    sol = best_ik(pos, yaw_deg)
    if sol is None:
        move_pose_tracked(pos, yaw_quat(yaw_deg), grip, tag, max_dq, settle)
        return yaw_deg
    move_joints_tracked(sol[2], grip, tag, max_dq, settle)
    motion_log[-1]["yaw"] = sol[1]
    motion_log[-1]["dq"] = round(sol[0], 3)
    try:
        ee = get_robot_state().end_effector_poses["primary"].position
        motion_log[-1]["err"] = round(math.sqrt(sum((float(ee[k]) - pos[k]) ** 2 for k in range(3))), 4)
    except Exception:
        pass
    return sol[1]


def move_pose_tracked(pos, quat, grip, tag, max_dq=0.045, settle=4):
    st = get_robot_state()
    ik = solve_ik(Pose(pos, quat, base_frame), seed_joints=st.joint_positions["primary"])
    if not ik.ok:
        note_error("ik_" + tag, "ik failed; fallback move_to_pose")
        r = move_to_pose(Pose(pos, quat, base_frame), tolerance=0.01, max_steps=150)
        flags["tick"] += 1
        if is_terminal(r):
            flags["terminal"] = True
        motion_log.append({"tag": tag, "fallback": res_info(r)})
        return
    q1 = [float(v) for v in ik.joint_positions]
    move_joints_tracked(q1, grip, tag, max_dq, settle)
    try:
        ee = get_robot_state().end_effector_poses["primary"].position
        motion_log[-1]["err"] = round(math.sqrt(sum((float(ee[k]) - pos[k]) ** 2 for k in range(3))), 4)
    except Exception:
        pass


def grip_ticks(g, tag, n_max=8):
    """Hold the arm, command the gripper for up to n_max ticks (tracked), stop once it settles."""
    st = get_robot_state()
    q = [float(v) for v in st.joint_positions["primary"]]
    prev = None
    first = None
    n = 0
    for k in range(n_max):
        step_joints(q, g)
        n += 1
        if flags["terminal"]:
            break
        try:
            cur = float(get_robot_state().gripper_positions["primary"])
        except Exception:
            cur = None
        if first is None:
            first = cur
        if cur is not None and g > 0.5 and cur >= 0.97:
            break
        if (k >= 3 and cur is not None and prev is not None and abs(cur - prev) < 0.01
                and first is not None and abs(cur - first) > 0.12):
            break
        prev = cur
    try:
        motion_log.append({"tag": tag, "ticks": n, "t": flags["tick"],
                           "grip": round(float(get_robot_state().gripper_positions["primary"]), 3)})
        return
    except Exception:
        pass
    motion_log.append({"tag": tag, "ticks": n, "t": flags["tick"]})


def gripper_call(fn, tag):
    r = fn()
    motion_log.append({"tag": tag, "info": res_info(r)})
    if is_terminal(r):
        flags["terminal"] = True
    try:
        flags["tick"] += int(r.steps_executed)
    except Exception:
        pass
    try:
        track_frame(get_observation())
    except Exception as err:
        note_error("track_g", err)


def enable_motion(on):
    if on and not motion["enabled"]:
        motion["enabled_since"] = flags["tick"]
    if not on:
        motion["enabled_since"] = None
        if motion["in_event"] and motion["events"]:
            motion["events"][-1][2] = False
        motion["in_event"] = False
    motion["enabled"] = on


def watch(quiet_needed, cap_ticks, need_highlight_end, grip):
    start = flags["tick"]
    enable_motion(True)
    while not flags["terminal"] and flags["tick"] - start < cap_ticks:
        tick_hold(grip)
        if need_highlight_end and hl["off_tick"] is None:
            # highlight not yet over (or not yet seen): keep watching; give up waiting for an
            # unseen highlight after 200 ticks
            if hl["on_tick"] is None and flags["tick"] - start > 200:
                need_highlight_end = False
            continue
        last = motion["last_motion_tick"]
        ref = last if last is not None else start
        if hl["off_tick"] is not None and hl["off_tick"] > ref:
            ref = hl["off_tick"]
        if flags["tick"] - ref >= quiet_needed and not motion["in_event"]:
            break
    return flags["tick"] - start


def swap_events():
    return [e for e in motion["events"] if e[1] - e[0] >= 12]


def swap_period():
    ev = [e for e in swap_events() if e[2]]
    if len(ev) < 2:
        return None
    diffs = sorted(ev[k + 1][0] - ev[k][0] for k in range(len(ev) - 1) if ev[k + 1][0] - ev[k][0] < 260)
    if not diffs:
        return None
    return diffs[len(diffs) // 2]


def quiet_gap():
    """Median quiet gap (end of a swap -> start of the next) over cleanly observed pairs."""
    ev = swap_events()
    gaps = []
    for k in range(len(ev) - 1):
        if ev[k][3] and ev[k + 1][2] and ev[k + 1][0] - ev[k][1] < 200:
            gaps.append(ev[k + 1][0] - ev[k][1])
    if not gaps:
        return None
    gaps = sorted(gaps)
    return gaps[0] if len(gaps) < 3 else gaps[len(gaps) // 2]


def quiet_left():
    """Predicted quiet ticks left, if the last swap end was observed from home."""
    ev = swap_events()
    gap = quiet_gap()
    if gap is None or not ev or motion["in_event"] or not motion["enabled"]:
        return None
    last = ev[-1]
    if not last[3]:
        return None
    return gap - (flags["tick"] - last[1])


def wait_for_window(need, cap_ticks):
    """At home, tracking: go now if the observed quiet time left covers `need`; otherwise wait
    for the next swap to end and go right after it (the longest available window)."""
    start = flags["tick"]
    enable_motion(True)
    seen_event = False
    while not flags["terminal"] and flags["tick"] - start < cap_ticks:
        left = quiet_left()
        if left is not None and left >= need:
            return [left, flags["tick"] - start, "fits"]
        if motion["in_event"]:
            seen_event = True
        elif seen_event:
            return [left, flags["tick"] - start, "after_swap"]
        tick_hold(get_grip())
    return [quiet_left(), flags["tick"] - start, "cap"]


def get_grip():
    try:
        g = float(get_robot_state().gripper_positions["primary"])
        return 1.0 if g > 0.5 else 0.0
    except Exception:
        return 1.0


def press_button(tag):
    if button is None:
        note_error("press", "no button")
        return
    bx, by, bt = button["cx"], button["cy"], button["top"]
    if get_grip() > 0.5:
        grip_ticks(0.0, tag + "_close", 6)
    if flags["terminal"]:
        return
    move_pose_tracked([bx, by, bt + 0.05], home_quat, 0.0, tag + "_above", max_dq=0.08, settle=2)
    if flags["terminal"]:
        return
    move_pose_tracked([bx, by, bt - 0.012], home_quat, 0.0, tag + "_down", max_dq=0.04, settle=5)
    if flags["terminal"]:
        return
    try:
        motion_log[-1]["ee_z"] = round(float(get_robot_state().end_effector_poses["primary"].position[2]), 4)
    except Exception:
        pass
    move_pose_tracked([bx, by, bt + 0.05], home_quat, 0.0, tag + "_up", max_dq=0.06, settle=1)
    if flags["terminal"]:
        return
    move_joints_tracked(home_joints, 0.0, tag + "_home", max_dq=0.08, settle=1)


def snapshot(tag):
    report.setdefault("snapshots", []).append(
        {"tag": tag, "tick": flags["tick"],
         "tracks": [[t["id"], round(t["x"], 3), round(t["y"], 3), t["lost"]] for t in tracks]})


# ---------------- 1) watch highlight + swaps at home ----------------
report["stage"] = "watch1"
try:
    snapshot("init")
    start1 = flags["tick"]
    watch(5, 260, True, 1.0)
    while not flags["terminal"] and len(swap_events()) < 2 and flags["tick"] - start1 < 420:
        watch(6, 60, False, 1.0)
    snapshot("after_watch1")
    report["period_after_watch1"] = swap_period()
except Exception as err:
    note_error("watch1", err)

highlighted = []
mx = max([t["white"] for t in tracks]) if tracks else 0
for t in tracks:
    if mx > 0 and t["white"] >= max(6, 0.35 * mx):
        highlighted.append(t)
report["highlight"] = {"on": hl["on_tick"], "off": hl["off_tick"], "peak": hl["peak"],
                       "frames": hl["frames"], "peak_counts": hl.get("peak_counts"),
                       "white": [t["white"] for t in tracks],
                       "ids": [t["id"] for t in highlighted]}

# ---------------- 2) first press ----------------
report["stage"] = "press1"
report["gap_after_watch1"] = quiet_gap()
try:
    if not flags["terminal"]:
        report["window_press1"] = wait_for_window(55, 260)
        enable_motion(False)
        t_before = flags["tick"]
        press_button("press1")
        report["press1_ticks"] = flags["tick"] - t_before
    if not flags["terminal"]:
        grip_ticks(1.0, "open_after_press1", 6)
    snapshot("after_press1")
    if not highlighted and hl["on_tick"] is not None and not flags["terminal"]:
        mx = max([t["white"] for t in tracks]) if tracks else 0
        for t in tracks:
            if mx > 0 and t["white"] >= max(6, 0.35 * mx):
                highlighted.append(t)
        report["highlight_late"] = [t["id"] for t in highlighted]
except Exception as err:
    note_error("press1", err)


def ambiguous(t):
    if t["lost"] > 0:
        return True
    for t2 in tracks:
        if t2 is not t and (t2["x"] - t["x"]) ** 2 + (t2["y"] - t["y"]) ** 2 < 0.025 ** 2:
            return True
    return False


# ---------------- 3) pick + replace each highlighted cube ----------------
report["stage"] = "picks"
pick_log = []
cycles = []
flags["cycle"] = 62
attempts = 0
try:
    while not flags["terminal"] and attempts < 3 * len(highlighted) + 2:
        todo = [h for h in highlighted if not h["picked"]]
        if not todo:
            break
        attempts += 1
        remaining = len(todo)
        per = swap_period()
        cap = 260 if per is None else per + 40
        spare = max_steps - flags["tick"] - remaining * (flags["cycle"] + 10) - 90
        if spare < 40:
            cap = 0
        entry_window = wait_for_window(flags["cycle"], cap)
        if flags["terminal"]:
            break
        # first unambiguous target (keep list order otherwise)
        t = None
        for h in todo:
            if not ambiguous(h):
                t = h
                break
        if t is None:
            report.setdefault("ambiguous_waits", []).append([flags["tick"], [h["id"] for h in todo]])
            if spare < 40:
                t = todo[0]
            else:
                tick_hold(get_grip())
                continue
        enable_motion(False)
        x, y, top = t["x"], t["y"], t["top"]
        t_start = flags["tick"]
        entry = {"id": t["id"], "xy": [round(x, 4), round(y, 4)], "top": round(top, 4),
                 "lost": t["lost"], "reacq": t.get("reacq", 0), "tick": flags["tick"], "window": entry_window}
        if get_grip() < 0.5:
            grip_ticks(1.0, "open_" + str(t["id"]), 6)
        if flags["terminal"]:
            break
        yaw = move_top_down([x, y, top + 0.05], 0.0, 1.0, "pre_" + str(t["id"]), max_dq=0.08, settle=1)
        if flags["terminal"]:
            break
        # re-read the target after the transit (it may have moved while we travelled)
        if t["lost"] == 0 and (t["x"] - x) ** 2 + (t["y"] - y) ** 2 > 0.01 ** 2:
            entry["moved_during_transit"] = [round(t["x"], 4), round(t["y"], 4)]
            x, y, top = t["x"], t["y"], t["top"]
            yaw = move_top_down([x, y, top + 0.05], yaw, 1.0, "pre2_" + str(t["id"]), max_dq=0.06, settle=1)
            if flags["terminal"]:
                break
        yaw = move_top_down([x, y, top - 0.02], yaw, 1.0, "desc_" + str(t["id"]), max_dq=0.05, settle=2)
        if flags["terminal"]:
            break
        grip_ticks(0.0, "close_" + str(t["id"]), 8)
        if flags["terminal"]:
            break
        try:
            entry["grip_closed"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
        except Exception:
            pass
        yaw = move_top_down([x, y, top + 0.03], yaw, 0.0, "lift_" + str(t["id"]), max_dq=0.05, settle=2)
        if flags["terminal"]:
            break
        try:
            entry["grip_lifted"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
        except Exception:
            pass
        yaw = move_top_down([x, y, top - 0.012], yaw, 0.0, "lower_" + str(t["id"]), max_dq=0.05, settle=1)
        if flags["terminal"]:
            break
        grip_ticks(1.0, "release_" + str(t["id"]), 6)
        if flags["terminal"]:
            break
        t["picked"] = True
        move_joints_tracked(home_joints, 1.0, "home_" + str(t["id"]), max_dq=0.08, settle=1)
        entry["done_tick"] = flags["tick"]
        entry["cycle_ticks"] = flags["tick"] - t_start
        cycles.append(flags["tick"] - t_start + 2)
        srt = sorted(cycles)
        flags["cycle"] = min(95, max(45, srt[len(srt) // 2]))
        pick_log.append(entry)
except Exception as err:
    note_error("picks", err)
report["picks"] = pick_log

# ---------------- 4) final press ----------------
report["stage"] = "press2"
try:
    if not flags["terminal"] and highlighted:
        report["window_press2"] = wait_for_window(55, 200)
        enable_motion(False)
        press_button("press2")
    k = 0
    while not flags["terminal"] and k < 20:
        tick_hold(0.0)
        k += 1
except Exception as err:
    note_error("press2", err)

report["motion_events"] = motion["events"]
report["quiet_gap"] = quiet_gap()
report["reacq"] = reacq_log
report["swap_period"] = swap_period()
report["n_motion_ticks"] = motion["n_motion_ticks"]
report["motions"] = motion_log
report["terminal"] = flags["terminal"]
report["ticks"] = flags["tick"]
snapshot("final")
report["stage"] = "done"
result = report
