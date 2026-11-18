# ButtonUnmask v01 -- memory: watch the LIVE front camera every control tick.
# v00 evidence (ep0): no conditioning video; cubes appear on the table during execution
# (~step 12) and are covered by white two-tier covers later (~step 44); the goal names THREE
# cubes in order ("...blue cube, followed by ... green cube, and finally ... red cube").
# Plan: press the button immediately (deadline < step 90) while a per-tick watcher records
# saturated colour blobs and back-projects them to robot_base with the frame's own depth and
# calibration; hold until the blobs vanish (covered); then for each goal colour in order, find
# the cover over the remembered cube from white pixels near it, grasp its top, lift +0.20 m,
# put it back down, release, and continue.
import math

report = {"version": "v02", "events": [], "motions": []}


def note(msg):
    report["events"].append(str(msg))


def fl(values):
    return [round(float(value), 4) for value in values]


def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def as_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        return arr


# ---------------------------------------------------------------- task / metadata
context = get_task_context()
goal = str(context.language)
report["goal"] = goal
max_steps = 0
try:
    meta = robomme_plus.get_task_metadata()
    max_steps = int(meta["max_steps"])
except Exception as exc:
    report["task_metadata_error"] = str(exc)
report["max_steps"] = max_steps
try:
    video_len = len(robomme_plus.get_conditioning_video())
except Exception:
    video_len = -1
report["video_len"] = video_len

state0 = get_robot_state()
base_frame = state0.base_frame
embodiment = state0.embodiment
home_pose = state0.end_effector_poses["primary"]
home_quat = [float(v) for v in home_pose.quaternion_wxyz]
home_joints = [float(v) for v in state0.joint_positions["primary"]]

CANON_HUE = {"red": 0.0, "orange": 30.0, "yellow": 55.0, "green": 120.0, "cyan": 180.0,
             "blue": 225.0, "purple": 280.0, "violet": 280.0, "magenta": 310.0, "pink": 320.0}
words = [w.strip(".,;:!?'\"()").lower() for w in goal.split()]
goal_colors = []
for word in words:
    if word in CANON_HUE and word not in goal_colors:
        goal_colors.append(word)
report["goal_colors"] = goal_colors


# ---------------------------------------------------------------- pixel / geometry helpers
def hue_sat(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    span = float(mx - mn)
    if mx < 50 or span < 40:
        return None
    sat = span / mx
    if mx == r:
        hue = 60.0 * (((g - b) / span) % 6.0)
    elif mx == g:
        hue = 60.0 * ((b - r) / span + 2.0)
    else:
        hue = 60.0 * ((r - g) / span + 4.0)
    return hue, sat


def cube_pixel(r, g, b):
    """Saturated object colour, excluding the wood table (orange hue, moderate saturation)."""
    hs = hue_sat(r, g, b)
    if hs is None:
        return None
    hue, sat = hs
    if sat < 0.45:
        return None
    if 8.0 <= hue < 40.0 and sat < 0.9:
        return None
    if 40.0 <= hue < 50.0 and sat < 0.8:
        return None
    return hue


def hue_dist(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def quat_matrix(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def camera_model(cam):
    K = as_rows(cam.intrinsics)
    pose = cam.camera_pose
    return {"fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]), "cy": float(K[1][2]),
            "R": quat_matrix(pose.quaternion_wxyz), "t": [float(v) for v in pose.position]}


def backproject(model, u, v, d):
    """OpenCV pinhole (z forward, y down) camera -> robot_base; verified on ep0 calibration."""
    xc = (u - model["cx"]) * d / model["fx"]
    yc = (v - model["cy"]) * d / model["fy"]
    R = model["R"]
    t = model["t"]
    return [R[0][0] * xc + R[0][1] * yc + R[0][2] * d + t[0],
            R[1][0] * xc + R[1][1] * yc + R[1][2] * d + t[1],
            R[2][0] * xc + R[2][1] * yc + R[2][2] * d + t[2]]


STRIDE = 2


def find_blobs(cam, want_3d):
    rows = as_rows(cam.rgb)
    height = len(rows)
    width = len(rows[0])
    gh = height // STRIDE
    gw = width // STRIDE
    hue_at = {}
    for gy in range(gh):
        row = rows[gy * STRIDE]
        for gx in range(gw):
            pix = row[gx * STRIDE]
            hue = cube_pixel(int(pix[0]), int(pix[1]), int(pix[2]))
            if hue is not None:
                hue_at[gy * gw + gx] = hue
    seen = set()
    blobs = []
    for start in hue_at:
        if start in seen:
            continue
        seen.add(start)
        stack = [start]
        cells = []
        while stack:
            cell = stack.pop()
            cells.append(cell)
            cy = cell // gw
            cx = cell % gw
            for other in (cell - gw, cell + gw, cell - 1 if cx > 0 else -1, cell + 1 if cx < gw - 1 else -1):
                if other >= 0 and other in hue_at and other not in seen and abs(hue_at[other] - hue_at[cell]) < 30.0:
                    seen.add(other)
                    stack.append(other)
        if len(cells) < 6 or len(cells) > 400:
            continue
        hues = [hue_at[c] for c in cells]
        # circular mean hue
        sx = sum(math.cos(math.radians(h)) for h in hues)
        sy = sum(math.sin(math.radians(h)) for h in hues)
        mean_hue = math.degrees(math.atan2(sy, sx)) % 360.0
        us = [(c % gw) * STRIDE for c in cells]
        vs = [(c // gw) * STRIDE for c in cells]
        blobs.append({"size": len(cells), "hue": round(mean_hue, 1),
                      "u": round(sum(us) / len(us), 1), "v": round(sum(vs) / len(vs), 1),
                      "cells": cells})
    if want_3d and blobs:
        depth = as_rows(cam.depth_m)
        model = camera_model(cam)
        for blob in blobs:
            pts = []
            for c in blob["cells"]:
                u = (c % gw) * STRIDE
                v = (c // gw) * STRIDE
                d = float(depth[v][u])
                if d > 0.05:
                    pts.append(backproject(model, u, v, d))
            if pts:
                blob["xyz"] = [round(quantile([p[0] for p in pts], 0.5), 4),
                               round(quantile([p[1] for p in pts], 0.5), 4),
                               round(quantile([p[2] for p in pts], 0.95), 4)]
    for blob in blobs:
        blob.pop("cells")
    return blobs


# ---------------------------------------------------------------- watcher (runs every tick)
tracks = []
watch = {"log": [], "now": 0}
status = {"terminated": False, "truncated": False}


def watcher(obs, tick):
    """Every tick: detect saturated cubes, back-project, and keep one track per cube."""
    watch["now"] = tick
    try:
        cam = obs.cameras["front"]
        blobs = find_blobs(cam, True)
    except Exception as exc:
        note("watcher error at %d: %s" % (tick, exc))
        return
    if len(watch["log"]) < 500:
        watch["log"].append([tick, len(blobs)])
    for blob in blobs:
        if "xyz" not in blob:
            continue
        match = None
        for track in tracks:
            if hue_dist(track["hue"], blob["hue"]) < 25.0 and abs(track["u"] - blob["u"]) < 12 and abs(track["v"] - blob["v"]) < 12:
                match = track
                break
        if match is None:
            tracks.append({"hue": blob["hue"], "u": blob["u"], "v": blob["v"], "xyz": blob["xyz"],
                           "size": blob["size"], "max_size": blob["size"], "first_tick": tick, "last_tick": tick})
            continue
        match["last_tick"] = tick
        if blob["size"] >= 0.9 * match["max_size"]:
            match["u"] = blob["u"]
            match["v"] = blob["v"]
            match["xyz"] = blob["xyz"]
            match["hue"] = blob["hue"]
            match["size"] = blob["size"]
        if blob["size"] > match["max_size"]:
            match["max_size"] = blob["size"]


def choose_tracks(strict_only):
    assignment = {}
    used = set()
    passes = (True,) if strict_only else (True, False)
    for strict in passes:
        for color in goal_colors:
            if color in assignment:
                continue
            best = None
            for index, track in enumerate(tracks):
                if index in used:
                    continue
                dist = hue_dist(track["hue"], CANON_HUE[color])
                if strict and dist > 25.0:
                    continue
                key = (dist if not strict else 0.0, -track["last_tick"])
                if best is None or key < best[0]:
                    best = (key, index)
            if best is not None:
                assignment[color] = best[1]
                used.add(best[1])
    return assignment


def flag_terminal(reply):
    term = False
    try:
        if reply.terminated:
            status["terminated"] = True
            term = True
    except Exception:
        pass
    try:
        if reply.truncated:
            status["truncated"] = True
            term = True
    except Exception:
        pass
    return term


def reply_obs(reply):
    try:
        return reply.observation
    except Exception:
        return get_observation()


def ended():
    return status["terminated"] or status["truncated"]


def servo(goal_joints, grip, max_ticks, tol=0.015, rate=0.08, label="", min_ticks=0):
    st = get_robot_state()
    cmd = [float(v) for v in st.joint_positions["primary"]]
    goal_list = [float(v) for v in goal_joints]
    ticks = 0
    err = None
    while ticks < max_ticks:
        for j in range(7):
            delta = goal_list[j] - cmd[j]
            if delta > rate:
                delta = rate
            if delta < -rate:
                delta = -rate
            cmd[j] = cmd[j] + delta
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=list(cmd), gripper_position=grip, embodiment=embodiment)}))
        ticks += 1
        if flag_terminal(reply):
            note("%s: terminal after %d ticks" % (label, ticks))
            return {"ticks": ticks, "terminal": True, "label": label}
        obs = reply_obs(reply)
        q = obs.robot_state.joint_positions["primary"]
        tick = int(round(float(obs.robot_state.timestamp_s) * 20.0))
        watcher(obs, tick)
        err = max(abs(float(q[j]) - goal_list[j]) for j in range(7))
        ramp_done = max(abs(goal_list[j] - cmd[j]) for j in range(7)) < 1e-9
        if ramp_done and err < tol and ticks >= min_ticks:
            break
    return {"ticks": ticks, "err": None if err is None else round(err, 4), "terminal": False, "label": label}


def ik_for(position, quat):
    pose = Pose([float(v) for v in position], [float(v) for v in quat], base_frame)
    seed = [float(v) for v in get_robot_state().joint_positions["primary"]]
    ik = solve_ik(pose, seed_joints=seed)
    if not ik.ok:
        return None
    return [float(v) for v in ik.joint_positions]


def go(position, quat, grip, max_ticks, label, tol=0.015, rate=0.08, min_ticks=0):
    joints = ik_for(position, quat)
    if joints is None:
        note("%s: IK failed for %s" % (label, fl(position)))
        return None
    out = servo(joints, grip, max_ticks, tol=tol, rate=rate, label=label, min_ticks=min_ticks)
    if not out["terminal"]:
        st = get_robot_state()
        out["ee"] = fl(st.end_effector_poses["primary"].position)
        out["gripper"] = round(float(st.gripper_positions["primary"]), 3)
        out["tick"] = round(float(st.timestamp_s) * 20.0, 1)
    out["target"] = fl(position)
    report["motions"].append(out)
    return out


def hold(ticks, grip, label):
    st = get_robot_state()
    return servo([float(v) for v in st.joint_positions["primary"]], grip, ticks, tol=-1.0, rate=0.08, label=label)


def tool_axis(q, col):
    w, x, y, z = q
    if col == 1:
        return [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)]
    return [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


down_quat = home_quat if tool_axis(home_quat, 2)[2] < -0.9 else [0.0, 1.0, 0.0, 0.0]

# ---------------------------------------------------------------- button (arm at home: clean view)
obs0 = get_observation()
watcher(obs0, 0)
button = None
button_log = []
found = segment_text("front", "button")
if found.ok:
    for cand in found.segmentations[:6]:
        try:
            cloud = mask_to_point_cloud(cand, "front", target_frame=base_frame)
            pts = cloud.points
        except Exception as exc:
            button_log.append({"error": str(exc)})
            continue
        if len(pts) < 10:
            continue
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        zs = [float(p[2]) for p in pts]
        top = quantile(zs, 0.95)
        height = top - quantile(zs, 0.05)
        span = max(quantile(xs, 0.95) - quantile(xs, 0.05), quantile(ys, 0.95) - quantile(ys, 0.05))
        top_pts = [(xs[i], ys[i]) for i in range(len(xs)) if zs[i] >= top - 0.01]
        entry = {"score": round(float(cand.score), 3), "top": round(top, 4), "height": round(height, 4),
                 "span": round(span, 4),
                 "top_xy": [round(quantile([p[0] for p in top_pts], 0.5), 4), round(quantile([p[1] for p in top_pts], 0.5), 4)]}
        entry["accepted"] = height <= 0.09 and span <= 0.16 and button is None
        button_log.append(entry)
        if entry["accepted"]:
            button = entry
report["button_candidates"] = button_log
report["button"] = button

# ---------------------------------------------------------------- 1. press the button
if button is not None:
    bx, by = button["top_xy"]
    btop = button["top"]
    above = go([bx, by, btop + 0.07], down_quat, 0.0, 30, "above_button")
    if above is not None and not above["terminal"]:
        press = go([bx, by, btop - 0.02], down_quat, 0.0, 20, "press_button", tol=0.01, rate=0.05)
        if press is not None and not press["terminal"]:
            hold(3, 0.0, "press_hold")
            go([bx, by, btop + 0.08], down_quat, 0.0, 20, "retreat_button")
            if not ended():
                report["motions"].append(servo(home_joints, 0.0, 40, tol=0.02, label="home_after_press"))
else:
    note("no button candidate accepted; skipping press")
report["tick_after_press"] = round(float(get_robot_state().timestamp_s) * 20.0, 1) if not ended() else None

# ---------------------------------------------------------------- 2. wait (arm home) until every goal cube was seen and is covered
# Cubes arrive in waves and are covered later (ep0: wave 1 steps 12-43, wave 2 steps 56-~88), so
# keep watching from the home pose until each goal colour has a strict-hue track that has been
# invisible for HIDDEN_TICKS, or until WAIT_LIMIT_TICK.
HIDDEN_TICKS = 10
WAIT_LIMIT_TICK = 400
waited = 0
ready_reason = None
while not ended():
    now = watch["now"]
    strict = choose_tracks(True)
    if len(strict) == len(goal_colors) and len(goal_colors) > 0:
        if all(now - tracks[strict[c]]["last_tick"] >= HIDDEN_TICKS for c in strict):
            ready_reason = "all goal cubes seen and hidden"
            break
    if now >= WAIT_LIMIT_TICK:
        ready_reason = "wait limit"
        break
    hold(1, 0.0, "wait_covers")
    waited += 1
report["wait_ticks"] = waited
report["ready_reason"] = ready_reason
report["tick_ready"] = watch["now"]
if not ended():
    hold(4, 0.0, "settle_covers")

memory_blobs = tracks
report["tracks"] = tracks
report["watch_log"] = watch["log"]
assignment = choose_tracks(False)
report["assignment"] = {c: {"track": assignment[c], "hue": tracks[assignment[c]]["hue"],
                            "xyz": tracks[assignment[c]]["xyz"]} for c in assignment}

# ---------------------------------------------------------------- 3. cover geometry from white pixels
def cover_cloud(cam, center_xyz, radius=0.04):
    rows = as_rows(cam.rgb)
    depth = as_rows(cam.depth_m)
    model = camera_model(cam)
    pts = []
    for v in range(0, len(rows)):
        row = rows[v]
        drow = depth[v]
        for u in range(0, len(row)):
            pix = row[u]
            r = int(pix[0])
            g = int(pix[1])
            b = int(pix[2])
            if min(r, g, b) < 120 or max(r, g, b) - min(r, g, b) > 35:
                continue
            d = float(drow[u])
            if d <= 0.05:
                continue
            p = backproject(model, u, v, d)
            if abs(p[0] - center_xyz[0]) > radius or abs(p[1] - center_xyz[1]) > radius:
                continue
            if (p[0] - center_xyz[0]) ** 2 + (p[1] - center_xyz[1]) ** 2 > radius * radius:
                continue
            if p[2] < 0.004:
                continue
            pts.append(p)
    return pts


def cover_geometry(pts):
    if len(pts) < 20:
        return None
    zs = [p[2] for p in pts]
    top = quantile(zs, 0.97)
    top_pts = [p for p in pts if p[2] >= top - 0.01]
    upper = [p for p in pts if p[2] >= top - 0.035]
    cx = quantile([p[0] for p in top_pts], 0.5)
    cy = quantile([p[1] for p in top_pts], 0.5)
    best = None
    for k in range(30):
        ang = math.radians(3.0 * k)
        ca = math.cos(ang)
        sa = math.sin(ang)
        a1 = [p[0] * ca + p[1] * sa for p in upper]
        a2 = [-p[0] * sa + p[1] * ca for p in upper]
        w1 = quantile(a1, 0.97) - quantile(a1, 0.03)
        w2 = quantile(a2, 0.97) - quantile(a2, 0.03)
        if best is None or w1 * w2 < best[0]:
            best = (w1 * w2, ang, w1, w2)
    return {"n": len(pts), "top": round(top, 4), "top_xy": [round(cx, 4), round(cy, 4)],
            "low": round(quantile(zs, 0.03), 4), "edge_angle": round(best[1], 4),
            "rect": [round(best[2], 4), round(best[3], 4)], "top_n": len(top_pts)}


covers = {}
if not ended() and assignment:
    obs_c = get_observation()
    cam_c = obs_c.cameras["front"]
    for color in goal_colors:
        if color not in assignment:
            continue
        cube = memory_blobs[assignment[color]]["xyz"]
        geom = cover_geometry(cover_cloud(cam_c, cube, 0.04))
        covers[color] = geom
report["covers"] = covers


def yaw_quat(edge_angle):
    finger = tool_axis(down_quat, 1)
    finger_angle = math.atan2(finger[1], finger[0])
    delta = edge_angle - finger_angle
    while delta > math.pi / 4.0:
        delta -= math.pi / 2.0
    while delta < -math.pi / 4.0:
        delta += math.pi / 2.0
    return qmul([math.cos(delta / 2.0), 0.0, 0.0, math.sin(delta / 2.0)], down_quat), delta


# ---------------------------------------------------------------- 4. lift covers in goal order
report["lifts"] = []
for color in goal_colors:
    if ended():
        break
    if color not in assignment:
        note("no remembered cube for %s" % color)
        continue
    cube = memory_blobs[assignment[color]]["xyz"]
    geom = covers.get(color)
    entry = {"color": color, "cube_xyz": cube}
    if geom is not None and (geom["top_xy"][0] - cube[0]) ** 2 + (geom["top_xy"][1] - cube[1]) ** 2 < 0.035 ** 2:
        gx, gy = geom["top_xy"]
        top = geom["top"]
        quat, delta = yaw_quat(geom["edge_angle"])
        entry["yaw_delta"] = round(delta, 4)
    else:
        gx, gy = cube[0], cube[1]
        top = (geom["top"] if geom is not None else cube[2] + 0.03)
        quat = down_quat
        entry["fallback"] = True
    grasp_z = top - 0.012
    entry["grasp"] = [round(gx, 4), round(gy, 4), round(grasp_z, 4)]
    pre = go([gx, gy, top + 0.10], quat, 1.0, 60, "pre_" + color, tol=0.02)
    if pre is None:
        quat = down_quat
        pre = go([gx, gy, top + 0.10], quat, 1.0, 60, "pre_homeyaw_" + color, tol=0.02)
    if pre is None or pre["terminal"]:
        report["lifts"].append(entry)
        continue
    hold(4, 1.0, "open_" + color)
    down = go([gx, gy, grasp_z], quat, 1.0, 40, "descend_" + color, tol=0.01, rate=0.04)
    if down is None or down["terminal"]:
        report["lifts"].append(entry)
        continue
    closed = close_gripper()
    try:
        entry["close"] = {"ok": bool(closed.ok), "steps": int(closed.steps_executed), "status": str(closed.status)}
        if closed.terminated or closed.truncated:
            status["terminated"] = True
    except Exception as exc:
        entry["close_error"] = str(exc)
    if ended():
        report["lifts"].append(entry)
        break
    lift = go([gx, gy, grasp_z + 0.20], quat, 0.0, 60, "lift_" + color, tol=0.015, rate=0.05)
    if lift is not None and not lift["terminal"]:
        hold(5, 0.0, "hold_" + color)
        if not ended():
            entry["held_gripper"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
            down2 = go([gx, gy, grasp_z + 0.008], quat, 0.0, 60, "replace_" + color, tol=0.015, rate=0.05)
            if down2 is not None and not down2["terminal"]:
                hold(6, 1.0, "release_" + color)
                if not ended():
                    go([gx, gy, top + 0.10], quat, 1.0, 40, "rise_" + color, tol=0.02)
    report["lifts"].append(entry)

report["status"] = status
try:
    if not ended():
        fin = get_robot_state()
        report["final_tick"] = round(float(fin.timestamp_s) * 20.0, 1)
except Exception as exc:
    report["final_error"] = str(exc)

result = report
