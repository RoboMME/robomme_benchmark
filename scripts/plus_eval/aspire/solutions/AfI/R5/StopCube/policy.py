# StopCube v03 -- v02 + prediction-gated, height-checked cube association (the static
# target answered 'cube' and hijacked v02's nearest-previous tracker on ep2), looser
# distance gate when a pass is detected after missing samples, trigger dmin gate r_hit+1cm,
# and a late-press safety net if the k-th arrival is counted without a trigger.
# StopCube v02 -- v01 + press the centroid of the button's TOP points (cap, not base
# centre) and seed the cube track with its t=0 position (passes from tick 2 countable).
# StopCube v01 -- v00 + measured press latency (2 ticks, centred trigger window),
# re-armed pass counting (actual distance gate), target-before-button disambiguation.
# (v00 header follows)
# StopCube v00 -- exploration folded into a first attempt (AfI, RoboMME-Plus).
# Records goal/metadata/video summary/segmentation probes/per-tick cube track, and
# attempts: hover a closed gripper above the button, track the cube on the front
# camera every tick, and press when the cube is predicted to arrive on the target
# for the k-th time (k parsed from the goal text; default 1).
import math

CAM = "front"
LAT_TICKS = 2.0          # press latency (ticks) from hover to button registration, measured v00 ep0
HOVER_DZ = 0.035         # tool hover height above observed button top (m)
PRESS_DZ = -0.02         # tool target below observed button top when pressing (m)
JOINT_STEP = 0.05        # max joint change per tick during transit (rad)

report = {"version": "v03", "events": [], "errors": []}
S = {"ended": False, "tick": 0, "sam": 0, "passes": 0, "last_tstar": None, "prev_px": None, "armed": True}
track = []

def note(msg):
    report["events"].append(str(msg))

def rflag(reply):
    term = False
    trunc = False
    try:
        term = bool(reply.terminated)
    except Exception:
        term = False
    try:
        trunc = bool(reply.truncated)
    except Exception:
        trunc = False
    return term, trunc

ctx = get_task_context()
goal = str(ctx.language)
report["goal"] = goal
try:
    tmeta = robomme_plus.get_task_metadata()
    report["task_metadata"] = dict(tmeta)
except Exception as exc:
    tmeta = {}
    report["errors"].append("task_metadata: " + str(exc))
try:
    report["controller_metadata"] = dict(robomme_plus.get_controller_metadata())
except Exception as exc:
    report["errors"].append("controller_metadata: " + str(exc))
max_steps = 300
try:
    max_steps = int(tmeta["max_steps"])
except Exception:
    pass
report["max_steps"] = max_steps

# ---- goal count parsing ----
WORDS = {"once": 1, "twice": 2, "thrice": 3, "one": 1, "two": 2, "three": 3, "four": 4,
         "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
         "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "1st": 1, "2nd": 2,
         "3rd": 3, "4th": 4, "5th": 5}
k_goal = None
clean = ""
for ch in goal.lower():
    if ch.isalnum() or ch == " ":
        clean = clean + ch
    else:
        clean = clean + " "
for word in clean.split():
    if k_goal is None:
        if word in WORDS:
            k_goal = WORDS[word]
        elif word.isdigit():
            k_goal = int(word)
report["k_goal_parsed"] = k_goal
if k_goal is None:
    k_goal = 1

state0 = get_robot_state()
emb = state0.embodiment
base = state0.base_frame
home_q = [float(v) for v in state0.joint_positions["primary"]]
ee0 = state0.end_effector_poses["primary"]
down_quat = [float(v) for v in ee0.quaternion_wxyz]
report["home_q"] = home_q
report["ee0"] = [float(v) for v in ee0.position]
report["ee0_quat"] = down_quat
report["t0"] = float(state0.timestamp_s)

obs0 = get_observation()
cams = []
for name in obs0.cameras:
    cams.append(str(name))
report["cameras"] = cams
front0 = obs0.cameras[CAM]
try:
    report["front_intrinsics"] = [[float(v) for v in row] for row in front0.intrinsics]
    report["front_pose"] = [float(v) for v in front0.camera_pose.position] + [float(v) for v in front0.camera_pose.quaternion_wxyz]
except Exception as exc:
    report["errors"].append("calib: " + str(exc))

def rgb_rows(cam_obs):
    img = cam_obs.rgb
    try:
        return img.tolist()
    except Exception:
        return img

def thumb(cam_obs, stride):
    rows = rgb_rows(cam_obs)
    out = []
    y = 0
    while y < len(rows):
        row = rows[y]
        line = []
        x = 0
        while x < len(row):
            px = row[x]
            line.append([int(px[0]), int(px[1]), int(px[2])])
            x += stride
        out.append(line)
        y += stride
    return out

report["front_thumb0"] = thumb(front0, 8)

# ---- segmentation probes with 3D geometry ----
def zq(points, q):
    zs = []
    for p in points:
        zs.append(float(p[2]))
    if len(zs) == 0:
        return None
    zs = sorted(zs)
    return zs[min(len(zs) - 1, int(q * len(zs)))]

def describe(candidate):
    d = {"score": round(float(candidate.score), 4)}
    try:
        d["box"] = [round(float(v), 1) for v in candidate.box_xyxy]
    except Exception:
        d["box"] = None
    try:
        pc = mask_to_point_cloud(candidate, CAM, target_frame=base)
        pts = pc.points
        d["n"] = len(pts)
        if len(pts) >= 5:
            g = estimate_geometry(pc)
            d["c"] = [round(float(v), 4) for v in g.pose.position]
            d["ext"] = [round(float(v), 4) for v in g.extents]
            d["z10"] = round(zq(pts, 0.10), 4)
            d["z95"] = round(zq(pts, 0.95), 4)
            top_x = 0.0
            top_y = 0.0
            top_n = 0
            for p in pts:
                if float(p[2]) >= d["z95"] - 0.006:
                    top_x += float(p[0])
                    top_y += float(p[1])
                    top_n += 1
            if top_n >= 3:
                d["top_c"] = [round(top_x / top_n, 4), round(top_y / top_n, 4)]
                d["top_n"] = top_n
    except Exception as exc:
        d["pc_err"] = str(exc)
    return d

PROMPTS = ["cube", "button", "target", "red cube", "small block", "round button",
           "target region", "bullseye", "disc", "goal marker"]
probes = {}
segs = {}
for prompt in PROMPTS:
    entry = {"ok": False, "cands": []}
    try:
        seg = segment_text(CAM, prompt)
        entry["ok"] = bool(seg.ok)
        if seg.ok:
            segs[prompt] = seg
            for cand in seg.segmentations[:5]:
                entry["cands"].append(describe(cand))
    except Exception as exc:
        entry["err"] = str(exc)
    probes[prompt] = entry
report["probes"] = probes

def box_center(box):
    return [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]

def box_area(box):
    return max(1.0, (box[2] - box[0]) * (box[3] - box[1]))

# table height estimate: low quantile of all candidate z10 values
z10s = []
for prompt in probes:
    for d in probes[prompt]["cands"]:
        if "z10" in d:
            z10s.append(d["z10"])
table_z = 0.0
if len(z10s) > 0:
    z10s = sorted(z10s)
    table_z = z10s[len(z10s) // 4]
report["table_z_est"] = table_z

def pick_cube():
    best = None
    for prompt in ("cube", "red cube", "small block"):
        for d in probes[prompt]["cands"]:
            if "ext" not in d or d["box"] is None:
                continue
            ex = d["ext"]
            if max(ex) > 0.09 or max(ex) < 0.01:
                continue
            rise = d["z95"] - table_z
            if rise < 0.015 or rise > 0.09:
                continue
            if best is None or d["score"] > best["score"]:
                best = d
        if best is not None:
            return best
    return best

def pick_button(cube, tgt):
    best = None
    for prompt in ("button", "round button"):
        for d in probes[prompt]["cands"]:
            if "ext" not in d or d["box"] is None:
                continue
            ex = d["ext"]
            if max(ex[0], ex[1]) > 0.2 or max(ex) < 0.01:
                continue
            if d["z95"] - table_z < 0.015:
                continue
            near = False
            for other, rad in ((cube, 0.02), (tgt, 0.03)):
                if other is not None and "c" in other:
                    dd = math.sqrt((d["c"][0] - other["c"][0]) ** 2 + (d["c"][1] - other["c"][1]) ** 2)
                    if dd < rad:
                        near = True
            if near:
                continue
            if best is None or d["score"] > best["score"]:
                best = d
        if best is not None:
            return best
    return best

def pick_target(cube, button):
    best = None
    for prompt in ("target", "target region", "bullseye", "disc", "goal marker"):
        for d in probes[prompt]["cands"]:
            if "ext" not in d or d["box"] is None:
                continue
            rise = d["z95"] - table_z
            max_rise = 0.03
            if cube is not None and "z95" in cube:
                max_rise = min(0.03, 0.6 * (cube["z95"] - table_z))
            if rise > max_rise:
                continue
            skip = False
            for other in (button,):
                if other is not None and "c" in other:
                    dd = math.sqrt((d["c"][0] - other["c"][0]) ** 2 + (d["c"][1] - other["c"][1]) ** 2)
                    if dd < 0.03:
                        skip = True
            if skip:
                continue
            if best is None or d["score"] > best["score"]:
                best = d
        if best is not None:
            return best
    return best

cube0 = pick_cube()
target = pick_target(cube0, None)
button = pick_button(cube0, target)
report["cube0"] = cube0
report["button"] = button
report["target"] = target

# cube colour from box-centre 5x5 of the live front image
cube_rgb = None
rows0 = rgb_rows(front0)
if cube0 is not None:
    cc = box_center(cube0["box"])
    acc = [0.0, 0.0, 0.0]
    n = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            yy = int(cc[1]) + dy
            xx = int(cc[0]) + dx
            if yy >= 0 and xx >= 0 and yy < len(rows0) and xx < len(rows0[0]):
                px = rows0[yy][xx]
                acc[0] += float(px[0])
                acc[1] += float(px[1])
                acc[2] += float(px[2])
                n += 1
    if n > 0:
        cube_rgb = [acc[0] / n, acc[1] / n, acc[2] / n]
report["cube_rgb"] = cube_rgb

def color_centroid(rows, stride, thr):
    if cube_rgb is None:
        return None
    sx = 0.0
    sy = 0.0
    n = 0
    y = 0
    H = len(rows)
    while y < H:
        row = rows[y]
        x = 0
        W = len(row)
        while x < W:
            px = row[x]
            d = abs(float(px[0]) - cube_rgb[0]) + abs(float(px[1]) - cube_rgb[1]) + abs(float(px[2]) - cube_rgb[2])
            if d < thr:
                sx += x
                sy += y
                n += 1
            x += stride
        y += stride
    if n == 0:
        return [None, None, 0]
    return [round(sx / n, 1), round(sy / n, 1), n]

# ---- conditioning video summary ----
vid = {"n": 0}
try:
    video = robomme_plus.get_conditioning_video()
    vid["n"] = len(video)
    if len(video) > 0:
        vid["t_first"] = float(video[0].timestamp_s)
        vid["t_last"] = float(video[len(video) - 1].timestamp_s)
        vid["thumb_first"] = thumb(video[0].cameras[CAM], 8)
        vid["thumb_mid"] = thumb(video[len(video) // 2].cameras[CAM], 8)
        vid["thumb_last"] = thumb(video[len(video) - 1].cameras[CAM], 8)
        cents = []
        stepv = max(1, len(video) // 128)
        i = 0
        while i < len(video):
            fr = video[i]
            cents.append([round(float(fr.timestamp_s), 3)] + color_centroid(rgb_rows(fr.cameras[CAM]), 4, 60.0))
            i += stepv
        vid["cube_color_centroids"] = cents
except Exception as exc:
    report["errors"].append("video: " + str(exc))
report["video"] = vid

# ---- motion helpers (per-tick, so tracking continues during transit) ----
def cmd(q, grip):
    return RobotAction(arms={"primary": ArmCommand(mode="joint_position", target=list(q),
                                                  gripper_position=grip, embodiment=emb)})

if cube0 is not None:
    S["prev_px"] = box_center(cube0["box"])
cube_area0 = 400.0
if cube0 is not None:
    cube_area0 = box_area(cube0["box"])

# Cube association (v03): predict the cube's pixel position from its pixel velocity, take
# SAM 'cube' candidates nearest the prediction inside a gate that widens with the number of
# ticks since the last fix, and reject any candidate whose 3D top (z95) is not cube-high:
# the flat target answers 'cube' too and, being static, hijacked a nearest-previous tracker.
height_thr = None
if cube0 is not None and target is not None and "z95" in cube0 and "z95" in target:
    if cube0["z95"] > target["z95"] + 0.006:
        height_thr = target["z95"] + 0.5 * (cube0["z95"] - target["z95"])
report["cube_height_thr"] = height_thr
S["vel_px"] = [0.0, 0.0]
S["fix_tick"] = 0
S["rejected_low"] = 0

def track_cube(reply_obs):
    """SAM 'cube' on the live front image; nearest-to-prediction candidate that is cube-high."""
    rec = {"px": None, "xyz": None}
    try:
        seg = segment_text(CAM, "cube")
        S["sam"] += 1
        if seg.ok:
            gap = max(1, S["tick"] - S["fix_tick"])
            pred = None
            if S["prev_px"] is not None:
                pred = [S["prev_px"][0] + S["vel_px"][0] * gap, S["prev_px"][1] + S["vel_px"][1] * gap]
            radius = min(60.0, 15.0 + 8.0 * gap)
            ranked = []
            for cand in seg.segmentations[:5]:
                try:
                    box = [float(v) for v in cand.box_xyxy]
                except Exception:
                    continue
                ar = box_area(box)
                if ar < 0.3 * cube_area0 or ar > 3.0 * cube_area0:
                    continue
                c = box_center(box)
                dpx = 0.0
                if pred is not None:
                    dpx = math.sqrt((c[0] - pred[0]) ** 2 + (c[1] - pred[1]) ** 2)
                if dpx > radius:
                    continue
                ranked.append([dpx, len(ranked), cand, c])
            ranked = sorted(ranked)
            tried = 0
            for item in ranked:
                if tried >= 3:
                    break
                tried += 1
                cand = item[2]
                c = item[3]
                pc = mask_to_point_cloud(cand, CAM, target_frame=base)
                if len(pc.points) < 5:
                    continue
                top = zq(pc.points, 0.95)
                if height_thr is not None and top < height_thr:
                    S["rejected_low"] += 1
                    continue
                g = estimate_geometry(pc)
                rec["xyz"] = [round(float(v), 4) for v in g.pose.position]
                rec["px"] = [round(c[0], 1), round(c[1], 1)]
                rec["top"] = round(top, 4)
                if S["prev_px"] is not None:
                    nv = [(c[0] - S["prev_px"][0]) / gap, (c[1] - S["prev_px"][1]) / gap]
                    S["vel_px"] = [0.5 * S["vel_px"][0] + 0.5 * nv[0], 0.5 * S["vel_px"][1] + 0.5 * nv[1]]
                S["prev_px"] = c
                S["fix_tick"] = S["tick"]
                break
    except Exception as exc:
        rec["err"] = str(exc)
    try:
        rec["col"] = color_centroid(rgb_rows(reply_obs.cameras[CAM]), 2, 60.0)
    except Exception:
        rec["col"] = None
    return rec

pressed = False
press_tick = None
last_rec_xyz = []
if cube0 is not None and "c" in cube0:
    last_rec_xyz.append([0, cube0["c"][0], cube0["c"][1]])

tgt_xy = None
r_hit = 0.03
if target is not None and "c" in target:
    tgt_xy = [target["c"][0], target["c"][1]]
    r_hit = max(0.02, min(0.08, 0.5 * max(target["ext"][0], target["ext"][1])))
report["r_hit"] = r_hit

def do_tick(q, grip, phase):
    """One control tick + tracking + pass counting. Returns False if episode ended."""
    if S["ended"]:
        return False
    reply = step(cmd(q, grip))
    S["tick"] += 1
    term, trunc = rflag(reply)
    entry = {"k": S["tick"], "ph": phase}
    try:
        st = reply.observation.robot_state
        entry["t"] = round(float(st.timestamp_s), 3)
        entry["ee"] = [round(float(v), 4) for v in st.end_effector_poses["primary"].position]
    except Exception:
        pass
    if term or trunc:
        S["ended"] = True
        entry["end"] = [term, trunc]
        track.append(entry)
        return False
    rec = track_cube(reply.observation)
    entry["px"] = rec["px"]
    entry["xyz"] = rec["xyz"]
    if "top" in rec:
        entry["top"] = rec["top"]
    entry["col"] = rec["col"]
    if "err" in rec:
        entry["err"] = rec["err"]
    # velocity / pass counting in xy
    if rec["xyz"] is not None:
        last_rec_xyz.append([S["tick"], rec["xyz"][0], rec["xyz"][1]])
    if tgt_xy is not None and len(last_rec_xyz) >= 3:
        a = last_rec_xyz[len(last_rec_xyz) - 3]
        b = last_rec_xyz[len(last_rec_xyz) - 1]
        dt = float(b[0] - a[0])
        if dt > 0 and dt <= 6:
            vx = (b[1] - a[1]) / dt
            vy = (b[2] - a[2]) / dt
            sp2 = vx * vx + vy * vy
            entry["v"] = [round(vx, 5), round(vy, 5)]
            if sp2 > 1e-7:
                px = b[1] - tgt_xy[0]
                py = b[2] - tgt_xy[1]
                ts = -(px * vx + py * vy) / sp2
                cx = px + vx * ts
                cy = py + vy * ts
                dmin = math.sqrt(cx * cx + cy * cy)
                entry["tstar"] = round(ts, 2)
                entry["dmin"] = round(dmin, 4)
                dnow = math.sqrt(px * px + py * py)
                entry["dnow"] = round(dnow, 4)
                if not S["armed"] and dnow > r_hit + 0.02:
                    S["armed"] = True
                if S["armed"] and S["last_tstar"] is not None and S["last_tstar"] > 0 and ts <= 0 and dmin < r_hit and dnow < r_hit + 0.04:
                    S["passes"] += 1
                    S["armed"] = False
                    note("pass " + str(S["passes"]) + " at tick " + str(S["tick"]))
                S["last_tstar"] = ts
    entry["passes"] = S["passes"]
    track.append(entry)
    return True

def ik_for(xyz, seed):
    ik = solve_ik(Pose(position=list(xyz), quaternion_wxyz=down_quat, frame=base), seed_joints=seed)
    if not ik.ok:
        return None
    return [float(v) for v in ik.joint_positions]

def travel(qgoal, grip, phase):
    st = get_robot_state()
    q0 = [float(v) for v in st.joint_positions["primary"]]
    dmax = 0.0
    for i in range(7):
        dmax = max(dmax, abs(qgoal[i] - q0[i]))
    n = max(1, int(math.ceil(dmax / JOINT_STEP)))
    for j in range(1, n + 1):
        f = float(j) / n
        q = [q0[i] + f * (qgoal[i] - q0[i]) for i in range(7)]
        if not do_tick(q, grip, phase):
            return False
    for settle in range(4):
        if not do_tick(qgoal, grip, phase):
            return False
    return True

# ---- attempt ----
hover_q = None
press_q = None
if button is not None and "c" in button:
    btop = button["z95"]
    bx = button["c"][0]
    by = button["c"][1]
    if "top_c" in button:
        bx = button["top_c"][0]
        by = button["top_c"][1]
    report["press_xy"] = [bx, by]
    report["button_top"] = btop
    hover_q = ik_for([bx, by, btop + HOVER_DZ], home_q)
    if hover_q is not None:
        press_q = ik_for([bx, by, btop + PRESS_DZ], hover_q)
    report["hover_ik_ok"] = hover_q is not None
    report["press_ik_ok"] = press_q is not None
else:
    note("no button found")

if hover_q is not None and press_q is not None:
    mid_q = ik_for([bx, by, button["z95"] + 0.12], home_q)
    if mid_q is not None:
        travel(mid_q, 0.0, "to_mid")
    travel(hover_q, 0.0, "to_hover")
    note("hover reached at tick " + str(S["tick"]))
    budget_ticks = max_steps - 15
    while not S["ended"] and S["tick"] < budget_ticks:
        trig = False
        if len(track) > 0:
            e = track[len(track) - 1]
            if S["armed"] and S["passes"] == k_goal - 1 and "tstar" in e and e["tstar"] >= 0 and e["tstar"] <= LAT_TICKS + 0.5 and e["dmin"] < r_hit + 0.01:
                trig = True
            if S["passes"] >= k_goal:
                # safety net: the k-th arrival was counted without a trigger -> press at once
                trig = True
                note("late press (trigger window missed)")
        if trig:
            pressed = True
            press_tick = S["tick"]
            note("press triggered at tick " + str(S["tick"]) + " passes=" + str(S["passes"]))
            for pk in range(8):
                if not do_tick(press_q, 0.0, "press"):
                    break
            for pk in range(6):
                if not do_tick(hover_q, 0.0, "lift"):
                    break
            break
        if not do_tick(hover_q, 0.0, "hover"):
            break
    # observe what happens afterwards
    for obs_tick in range(25):
        if S["ended"]:
            break
        if not do_tick(hover_q, 0.0, "after"):
            break
else:
    note("ik failed or no button; observing only")
    for obs_tick in range(min(120, max_steps - 5)):
        if not do_tick(home_q, 1.0, "observe"):
            break

report["pressed"] = pressed
report["press_tick"] = press_tick
report["passes"] = S["passes"]
report["k_goal"] = k_goal
report["ticks"] = S["tick"]
report["sam_calls"] = S["sam"]
report["rejected_low"] = S["rejected_low"]
report["ended"] = S["ended"]
report["track"] = track
result = report
