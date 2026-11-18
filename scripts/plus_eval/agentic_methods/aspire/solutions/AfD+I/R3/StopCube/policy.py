import math

# StopCube v01 (v00 + evidence-based object identity) -- press the button so that it registers just as the moving cube reaches the
# target for the N-th time (N parsed from the episode goal).
#
# Strategy (all quantities derived at runtime from this episode's goal and live RGB-D):
#  1. Segment button / target / cube in the front camera; project to robot_base.
#  2. Fit the table as an inverse-depth plane in image space; cube pixels are those raised above
#     it by more than the (thin) target and less than ~1.5x the cube's own height.
#  3. Track the cube EVERY control tick (all motion is issued with one-tick `step` calls), count
#     entries of the cube into the target zone (radius = target radius + 0.25 cube radius).
#  4. Move above the button (pre-hover = hover + D). The pre-hover -> hover descent is a step
#     command with the same magnitude/direction as the later press, so it measures the press
#     latency L (ticks until the fingertip has descended to the button top).
#  5. After N-1 arrivals, predict the time of the N-th closest approach by matching the current
#     cube position/velocity to the previous approaches (time-scaled by speed), and issue the
#     press so that it reaches the button about half a tick before that closest approach.

HOVER_ABOVE_TOP = 0.025      # m, fingertip clearance above the measured button top while waiting
PRESS_BELOW_TOP = 0.015      # m, commanded press depth below the measured button top
CONTACT_FRACTION_P = 0.25    # latency is measured to (top - 0.25 * press depth)
AIM_LEAD_TICKS = 0.5         # aim the press slightly before the predicted closest approach
MAX_JOINT_STEP = 0.05        # rad per tick for transit interpolation

ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
    "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "once": 1, "twice": 2, "thrice": 3,
}

report = {"version": "v01", "events": [], "errors": []}


def note(msg):
    report["events"].append(str(msg))


def parse_count(text):
    cleaned = text.lower()
    for ch in ",.;:!?":
        cleaned = cleaned.replace(ch, " ")
    for word in cleaned.split():
        if word in ORDINALS:
            return ORDINALS[word]
        digits = ""
        for ch in word:
            if ch in "0123456789":
                digits = digits + ch
            else:
                break
        if len(digits) > 0 and word[len(digits):] in ("st", "nd", "rd", "th"):
            return int(digits)
    return None


def quantile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def rnd(x, n=4):
    try:
        return round(float(x), n)
    except Exception:
        return None


def box_of(candidate):
    try:
        b = candidate.box_xyxy
        if b is None:
            return None
        return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    except Exception:
        return None


def iou(a, b):
    if a is None or b is None:
        return 0.0
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    if ua <= 0:
        return 0.0
    return inter / ua


def choose(prompts, avoid_boxes, max_area, max_iou, label):
    tried = []
    for prompt in prompts:
        try:
            found = segment_text("front", prompt)
        except Exception as exc:
            tried.append({"prompt": prompt, "error": str(exc)[:200]})
            continue
        entry = {"prompt": prompt, "ok": bool(found.ok), "cands": []}
        best = None
        if found.ok:
            for cand in found.segmentations[:5]:
                b = box_of(cand)
                sc = float(cand.score)
                entry["cands"].append([rnd(sc, 3), [rnd(v, 1) for v in b] if b is not None else None])
                if b is None:
                    continue
                area = (b[2] - b[0]) * (b[3] - b[1])
                if area > max_area or area <= 0:
                    continue
                bad = False
                for ab in avoid_boxes:
                    if iou(b, ab) > max_iou:
                        bad = True
                if bad:
                    continue
                if best is None or sc > best[0]:
                    best = (sc, b, cand)
        tried.append(entry)
        if best is not None:
            report["seg_" + label] = tried
            return best[1], best[2], prompt
    report["seg_" + label] = tried
    return None, None, None


def cloud_points(candidate, base_frame):
    try:
        cloud = mask_to_point_cloud(candidate, "front", target_frame=base_frame)
        pts = []
        for p in cloud.points:
            pts.append((float(p[0]), float(p[1]), float(p[2])))
        return pts
    except Exception as exc:
        report["errors"].append("cloud: " + str(exc)[:200])
        return []


def solve3(m, r):
    # Gaussian elimination for a 3x3 system m x = r
    a = [list(m[0]) + [r[0]], list(m[1]) + [r[1]], list(m[2]) + [r[2]]]
    for col in range(3):
        piv = col
        for row in range(col + 1, 3):
            if abs(a[row][col]) > abs(a[piv][col]):
                piv = row
        if abs(a[piv][col]) < 1e-18:
            return None
        tmp = a[col]
        a[col] = a[piv]
        a[piv] = tmp
        for row in range(3):
            if row != col:
                f = a[row][col] / a[col][col]
                for j in range(col, 4):
                    a[row][j] = a[row][j] - f * a[col][j]
    return [a[0][3] / a[0][0], a[1][3] / a[1][1], a[2][3] / a[2][2]]


def quat_matrix(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


# ---------------------------------------------------------------- task / metadata
ctx = get_task_context()
goal = ctx.language
report["goal"] = goal
target_count = parse_count(goal)
report["target_count"] = target_count
max_steps = None
try:
    meta = robomme_plus.get_task_metadata()
    try:
        max_steps = int(meta["max_steps"])
    except Exception:
        max_steps = None
    try:
        report["video_sampling"] = str(meta["conditioning_video_sampling"])[:300]
    except Exception:
        report["video_sampling"] = "absent"
    try:
        report["meta_keys"] = sorted([str(k) for k in meta])
    except Exception:
        pass
except Exception as exc:
    report["errors"].append("meta: " + str(exc)[:200])
report["max_steps"] = max_steps

state0 = get_robot_state()
base = state0.base_frame
emb = state0.embodiment
obs0 = get_observation()
front0 = obs0.cameras["front"]
depth0 = front0.depth_m
IMG_H = len(depth0)
IMG_W = len(depth0[0])
report["img"] = [IMG_H, IMG_W]

# ---------------------------------------------------------------- perception (v01: identity by evidence)
# v00 took the first-ranked candidate per prompt; in a far camera view the "button" scores were
# uninformative and the three roles got permuted. v01 pools all candidates of all prompts into
# distinct objects, identifies the CUBE as the object that moves while the arm holds still, the
# TARGET as the best "target"-scored remaining object, the BUTTON as the best "button"-scored
# other one, and checks with depth that the target is flatter than the button (else swaps).
img_area = float(IMG_H * IMG_W)
ROLE_PROMPTS = {"button": ("button", "push button", "round button on a white base"),
                "target": ("target", "bullseye target", "round flat target disc"),
                "cube": ("cube", "small cube", "block")}
CANDS = []


def collect(prompt, role):
    entry = {"prompt": prompt, "cands": []}
    try:
        found = segment_text("front", prompt)
    except Exception as exc:
        entry["error"] = str(exc)[:200]
        report.setdefault("seg_log", []).append(entry)
        return
    entry["ok"] = bool(found.ok)
    if found.ok:
        for cand in found.segmentations[:5]:
            b = box_of(cand)
            sc = float(cand.score)
            entry["cands"].append([rnd(sc, 3), [rnd(v, 1) for v in b] if b is not None else None])
            if b is None:
                continue
            area = (b[2] - b[0]) * (b[3] - b[1])
            if area <= 0 or area > 0.08 * img_area:
                continue
            CANDS.append({"role": role, "prompt": prompt, "score": sc, "box": b, "cand": cand, "area": area})
    report.setdefault("seg_log", []).append(entry)


def overlap_small(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    sa = (a[2] - a[0]) * (a[3] - a[1])
    sb = (b[2] - b[0]) * (b[3] - b[1])
    sm = min(sa, sb)
    return inter / sm if sm > 0 else 0.0


def build_objects(cands):
    objs = []
    dropped = []
    areas = [c["area"] for c in cands]
    asc = order_desc([-a for a in areas])
    for ci in asc:
        c = cands[ci]
        contains = [o for o in objs if overlap_small(c["box"], o["box"]) > 0.7 and c["area"] > 1.5 * o["area"]]
        if len(contains) >= 2:
            dropped.append([c["prompt"], [rnd(v, 1) for v in c["box"]]])
            continue
        home = None
        for o in objs:
            ratio = max(c["area"], o["area"]) / max(1e-9, min(c["area"], o["area"]))
            if iou(c["box"], o["box"]) > 0.5 or (overlap_small(c["box"], o["box"]) > 0.7 and ratio < 3.0):
                home = o
                break
        if home is None:
            objs.append({"box": list(c["box"]), "area": c["area"], "members": [c]})
        else:
            home["members"].append(c)
            if c["area"] > home["area"]:
                home["box"] = list(c["box"])
                home["area"] = c["area"]
    kept = []
    for o in objs:
        b = o["box"]
        if b[0] <= 1.0 or b[1] <= 1.0 or b[2] >= IMG_W - 2.0 or b[3] >= IMG_H - 2.0:
            dropped.append(["border", [rnd(v, 1) for v in b]])
            continue
        kept.append(o)
    return kept, dropped


def role_score(o, role):
    best = 0.0
    for m in o["members"]:
        if m["role"] == role and m["score"] > best:
            best = m["score"]
    return best


def role_member(o, role):
    best = None
    for m in o["members"]:
        if m["role"] == role and (best is None or m["score"] > best["score"]):
            best = m
    if best is None:
        for m in o["members"]:
            if best is None or m["area"] > best["area"]:
                best = m
    return best


def order_desc(vals):
    idx = list(range(len(vals)))
    out = []
    while len(idx) > 0:
        bi = idx[0]
        for i in idx:
            if vals[i] > vals[bi]:
                bi = i
        out.append(bi)
        idx = [i for i in idx if i != bi]
    return out


def best_by_role(objs, role):
    best = None
    for o in objs:
        if best is None or role_score(o, role) > role_score(best, role):
            best = o
    return best


for role in ("button", "target", "cube"):
    collect(ROLE_PROMPTS[role][0], role)
OBJS, DROPPED = build_objects(CANDS)
if len(OBJS) < 3:
    for role in ("button", "target", "cube"):
        for prompt in ROLE_PROMPTS[role][1:]:
            collect(prompt, role)
    OBJS, DROPPED = build_objects(CANDS)
report["objects"] = [[[rnd(v, 1) for v in o["box"]], rnd(role_score(o, "button"), 3), rnd(role_score(o, "target"), 3),
                      rnd(role_score(o, "cube"), 3), len(o["members"])] for o in OBJS]
report["dropped"] = DROPPED
if len(OBJS) < 3:
    raise ValueError("fewer than 3 distinct objects segmented: " + str(report["objects"]))


def fit_plane(depth_arr, excl, u_lo, u_hi, v_lo, v_hi):
    smp = []
    for v in range(int(max(0, v_lo)), int(min(IMG_H - 1, v_hi)) + 1, 4):
        row = depth_arr[v]
        for u in range(int(max(0, u_lo)), int(min(IMG_W - 1, u_hi)) + 1, 4):
            bad = False
            for b in excl:
                if b[0] - 6 <= u <= b[2] + 6 and b[1] - 6 <= v <= b[3] + 6:
                    bad = True
                    break
            if bad:
                continue
            d = float(row[u])
            if d > 0.05 and d < 10.0:
                smp.append((u, v, 1.0 / d))
    pl = None
    kp = smp
    for it in range(3):
        if len(kp) < 20:
            break
        m = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        r = [0.0, 0.0, 0.0]
        for s in kp:
            f = (float(s[0]), float(s[1]), 1.0)
            for i in range(3):
                r[i] = r[i] + f[i] * s[2]
                for j in range(3):
                    m[i][j] = m[i][j] + f[i] * f[j]
        sol = solve3(m, r)
        if sol is None:
            break
        pl = sol
        res = []
        for s in smp:
            w = pl[0] * s[0] + pl[1] * s[1] + pl[2]
            res.append(abs(1.0 / w - 1.0 / s[2]) if w > 0 else 1.0)
        med = quantile(res, 0.5)
        lim = 3.0 * med + 0.002
        kp = [smp[i] for i in range(len(smp)) if res[i] < lim]
    return pl, smp, kp


def box_height(depth_arr, pl, b, skip):
    hs = []
    for v in range(int(max(0, b[1])), int(min(IMG_H - 1, b[3])) + 1):
        for u in range(int(max(0, b[0])), int(min(IMG_W - 1, b[2])) + 1):
            if skip is not None and skip[0] - 2 <= u <= skip[2] + 2 and skip[1] - 2 <= v <= skip[3] + 2:
                continue
            w = pl[0] * u + pl[1] * v + pl[2]
            if w > 0:
                hs.append(1.0 / w - float(depth_arr[v][u]))
    return quantile(hs, 0.95) if len(hs) > 0 else None


all_boxes = [o["box"] for o in OBJS]
pre_plane, pre_smp, pre_keep = fit_plane(depth0, all_boxes, min([b[0] for b in all_boxes]) - 60,
                                         max([b[2] for b in all_boxes]) + 60, min([b[1] for b in all_boxes]) - 40,
                                         max([b[3] for b in all_boxes]) + 40)
if pre_plane is None:
    raise ValueError("preliminary table plane fit failed")
pre_res = []
for s in pre_keep:
    w = pre_plane[0] * s[0] + pre_plane[1] * s[1] + pre_plane[2]
    pre_res.append(abs(1.0 / w - 1.0 / s[2]))
NOISE = quantile(pre_res, 0.95)
for o in OBJS:
    o["hgt"] = box_height(depth0, pre_plane, o["box"], None)
report["object_heights"] = [rnd(o["hgt"]) for o in OBJS]
report["pre_plane_noise_q95"] = rnd(NOISE, 5)

# clouds are taken NOW (t=0): the cube moves during the hold, so a later mask_to_point_cloud of a
# t=0 mask could mix in table points
CLOUDS = []
for o in OBJS:
    for role in ("button", "target", "cube"):
        m = role_member(o, role)
        if "pts" not in m:
            m["pts"] = cloud_points(m["cand"], base)
            CLOUDS.append(len(m["pts"]))
report["t0_cloud_calls"] = CLOUDS

# ---- motion test: hold the arm still and see which object's pixels change
ended = {"flag": False, "why": None, "tick": None}
HOLD_OBS = []
hold_q = [float(x) for x in state0.joint_positions["primary"]]


def hold_step(q):
    cmd = ArmCommand(mode="joint_position", target=[float(x) for x in q], gripper_position=0.0, embodiment=emb)
    reply = step(RobotAction(arms={"primary": cmd}))
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
    obs = None
    try:
        obs = reply.observation
    except Exception:
        obs = None
    if term or trunc:
        ended["flag"] = True
        ended["why"] = "terminated" if term else "truncated"
        ended["tick"] = len(HOLD_OBS) + 1
    if obs is None and not ended["flag"]:
        obs = get_observation()
    HOLD_OBS.append(obs)
    return obs


def motion_fraction(dep, o):
    b = o["box"]
    thr = max(0.25 * o["hgt"] if o["hgt"] is not None else 0.0, 10.0 * NOISE)
    n = 0
    ch = 0
    nf = 0
    chf = 0
    others = [x["box"] for x in OBJS if x is not o]
    for v in range(int(max(0, b[1] - 2)), int(min(IMG_H - 1, b[3] + 2)) + 1):
        r0 = depth0[v]
        r1 = dep[v]
        for u in range(int(max(0, b[0] - 2)), int(min(IMG_W - 1, b[2] + 2)) + 1):
            changed = abs(float(r1[u]) - float(r0[u])) > thr
            nf += 1
            if changed:
                chf += 1
            shared = False
            for ob2 in others:
                if ob2[0] - 2 <= u <= ob2[2] + 2 and ob2[1] - 2 <= v <= ob2[3] + 2:
                    shared = True
                    break
            if shared:
                continue
            n += 1
            if changed:
                ch += 1
    # pixels shared with another object's box are ambiguous; use the full box only if few remain
    if n >= 0.3 * nf and n > 0:
        return ch / float(n)
    return chf / float(nf) if nf > 0 else 0.0


HOLD_MAX = 12
motion_log = []
cube_obj = None
for hk in range(HOLD_MAX):
    ob = hold_step(hold_q)
    if ended["flag"] or ob is None:
        break
    dep = ob.cameras["front"].depth_m
    fr = [motion_fraction(dep, o) for o in OBJS]
    motion_log.append([rnd(x, 3) for x in fr])
    order = order_desc(fr)
    top = fr[order[0]]
    second = fr[order[1]]
    if hk >= 1 and top >= 0.15 and top >= 3.0 * second:
        cube_obj = OBJS[order[0]]
        break
report["motion_log"] = motion_log
report["hold_ticks"] = len(HOLD_OBS)
if cube_obj is None and len(motion_log) > 0:
    fr = motion_log[-1]
    order = order_desc(fr)
    if fr[order[0]] >= 0.05 and fr[order[0]] >= 2.0 * fr[order[1]]:
        cube_obj = OBJS[order[0]]
        note("motion test weak; cube = most-changed object")
if cube_obj is None:
    cube_obj = best_by_role(OBJS, "cube")
    note("motion test inconclusive; cube = best cube-prompt object")
    report["cube_by"] = "score"
else:
    report["cube_by"] = "motion"
rest = [o for o in OBJS if o is not cube_obj]
for o in rest:
    o["hgt"] = box_height(depth0, pre_plane, o["box"], cube_obj["box"])
target_obj = best_by_role(rest, "target")
rest2 = [o for o in rest if o is not target_obj]
button_obj = best_by_role(rest2, "button")
if (target_obj["hgt"] is not None and button_obj["hgt"] is not None and target_obj["hgt"] > button_obj["hgt"]):
    note("target candidate taller than button candidate; swapping roles by flatness")
    tmp = target_obj
    target_obj = button_obj
    button_obj = tmp
bm = role_member(button_obj, "button")
tm = role_member(target_obj, "target")
cm = role_member(cube_obj, "cube")
button_box, button_cand, button_prompt = bm["box"], bm["cand"], bm["prompt"]
target_box, target_cand, target_prompt = tm["box"], tm["cand"], tm["prompt"]
cube_box, cube_cand, cube_prompt = cm["box"], cm["cand"], cm["prompt"]
if target_prompt not in ROLE_PROMPTS["target"]:
    target_prompt = ROLE_PROMPTS["target"][0]
report["boxes"] = {"button": button_box, "target": target_box, "cube": cube_box}
report["prompts"] = [button_prompt, target_prompt, cube_prompt]
report["role_heights"] = {"button": rnd(button_obj["hgt"]), "target": rnd(target_obj["hgt"]), "cube": rnd(cube_obj["hgt"])}
report["object_heights_excl_cube"] = [rnd(o["hgt"]) for o in rest]
if ended["flag"]:
    raise ValueError("episode ended during the motion-identification hold")

button_pts = bm["pts"]
target_pts = tm["pts"]
cube_pts = cm["pts"]
report["cloud_counts"] = [len(button_pts), len(target_pts), len(cube_pts)]
if len(button_pts) < 10 or len(target_pts) < 5:
    raise ValueError("too few cloud points")

button_top = quantile([p[2] for p in button_pts], 0.95)
cap = [p for p in button_pts if p[2] >= button_top - 0.006]
button_x = quantile([p[0] for p in cap], 0.5)
button_y = quantile([p[1] for p in cap], 0.5)
report["button_xyz_top"] = [rnd(button_x), rnd(button_y), rnd(button_top)]
target_cx = quantile([p[0] for p in target_pts], 0.5)
target_cy = quantile([p[1] for p in target_pts], 0.5)
target_r3 = quantile([math.hypot(p[0] - target_cx, p[1] - target_cy) for p in target_pts], 0.9)
cube_r3 = None
cube_c3 = None
if len(cube_pts) >= 5:
    ccx = quantile([p[0] for p in cube_pts], 0.5)
    ccy = quantile([p[1] for p in cube_pts], 0.5)
    ccz = quantile([p[2] for p in cube_pts], 0.5)
    cube_c3 = (ccx, ccy, ccz)
    cube_r3 = quantile([math.hypot(p[0] - ccx, p[1] - ccy) for p in cube_pts], 0.9)
report["target_xy_r"] = [rnd(target_cx), rnd(target_cy), rnd(target_r3)]
report["cube_c3_r"] = [[rnd(v) for v in cube_c3] if cube_c3 else None, rnd(cube_r3)]

# ---------------------------------------------------------------- table plane (inverse depth)
tu = 0.5 * (target_box[0] + target_box[2])
tv = 0.5 * (target_box[1] + target_box[3])
cu0 = 0.5 * (cube_box[0] + cube_box[2])
cv0 = 0.5 * (cube_box[1] + cube_box[3])


def inside(box, u, v, margin):
    return box[0] - margin <= u <= box[2] + margin and box[1] - margin <= v <= box[3] + margin


samples = []
reg_u0 = int(max(0, min(tu, cu0) - 90))
reg_u1 = int(min(IMG_W - 1, max(tu, cu0) + 90))
reg_v0 = int(max(0, min(tv, cv0) - 60))
reg_v1 = int(min(IMG_H - 1, max(tv, cv0) + 60))
for v in range(reg_v0, reg_v1 + 1, 4):
    row = depth0[v]
    for u in range(reg_u0, reg_u1 + 1, 4):
        if inside(button_box, u, v, 6) or inside(target_box, u, v, 6) or inside(cube_box, u, v, 6):
            continue
        d = float(row[u])
        if d > 0.05 and d < 10.0:
            samples.append((u, v, 1.0 / d))
plane = None
keep = samples
for it in range(3):
    if len(keep) < 20:
        break
    m = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    r = [0.0, 0.0, 0.0]
    for s in keep:
        f = (float(s[0]), float(s[1]), 1.0)
        for i in range(3):
            r[i] = r[i] + f[i] * s[2]
            for j in range(3):
                m[i][j] = m[i][j] + f[i] * f[j]
    sol = solve3(m, r)
    if sol is None:
        break
    plane = sol
    res = []
    for s in samples:
        w = plane[0] * s[0] + plane[1] * s[1] + plane[2]
        res.append(abs(1.0 / w - 1.0 / s[2]) if w > 0 else 1.0)
    med = quantile(res, 0.5)
    lim = 3.0 * med + 0.002
    keep = [samples[i] for i in range(len(samples)) if res[i] < lim]
if plane is None:
    raise ValueError("table plane fit failed")
PA, PB, PC = plane[0], plane[1], plane[2]
fit_res = []
for s in keep:
    w = PA * s[0] + PB * s[1] + PC
    fit_res.append(abs(1.0 / w - 1.0 / s[2]))
report["plane"] = {"n": len(samples), "kept": len(keep), "res_q50": rnd(quantile(fit_res, 0.5), 5),
                   "res_q95": rnd(quantile(fit_res, 0.95), 5)}


def height_at(depth_arr, u, v):
    w = PA * u + PB * v + PC
    if w <= 0:
        return -1.0
    return 1.0 / w - float(depth_arr[v][u])


cube_diffs = []
for v in range(int(max(0, cube_box[1] - 2)), int(min(IMG_H - 1, cube_box[3] + 2)) + 1):
    for u in range(int(max(0, cube_box[0] - 2)), int(min(IMG_W - 1, cube_box[2] + 2)) + 1):
        h = height_at(depth0, u, v)
        if h > 0.003:
            cube_diffs.append(h)
target_diffs = []
for v in range(int(max(0, target_box[1])), int(min(IMG_H - 1, target_box[3])) + 1):
    for u in range(int(max(0, target_box[0])), int(min(IMG_W - 1, target_box[2])) + 1):
        if inside(cube_box, u, v, 2):
            continue
        target_diffs.append(height_at(depth0, u, v))
cube_d = quantile(cube_diffs, 0.9) if len(cube_diffs) > 0 else None
target_d = quantile(target_diffs, 0.9) if len(target_diffs) > 0 else 0.0
if target_d is None or target_d < 0:
    target_d = 0.0
report["cube_d"] = rnd(cube_d)
report["target_d"] = rnd(target_d)
if cube_d is None or cube_d - target_d < 0.006:
    raise ValueError("cube not raised above target/table in depth: " + str([cube_d, target_d]))
THR_LO = target_d + 0.35 * (cube_d - target_d)
THR_HI = 1.5 * cube_d + 0.01
THR_TOP = target_d + 0.8 * (cube_d - target_d)
bw = button_box[2] - button_box[0]
EXCL = [button_box[0] - 0.5 * bw, button_box[1] - 0.5 * bw, button_box[2] + 0.5 * bw, button_box[3] + 0.5 * bw]
cube_w = max(cube_box[2] - cube_box[0], cube_box[3] - cube_box[1])
report["thr"] = [rnd(THR_LO), rnd(THR_HI)]

# ---------------------------------------------------------------- camera deprojection variants
Kmat = front0.intrinsics
FX = float(Kmat[0][0])
FY = float(Kmat[1][1])
CX = float(Kmat[0][2])
CY = float(Kmat[1][2])
cam_pose = front0.camera_pose
CR = quat_matrix(cam_pose.quaternion_wxyz)
CT = [float(cam_pose.position[0]), float(cam_pose.position[1]), float(cam_pose.position[2])]
VARIANT = None


def deproject(u, v, d, variant):
    xn = (u - CX) / FX
    yn = (v - CY) / FY
    if variant[0] == "cv":
        ray = (xn, yn, 1.0)
    else:
        ray = (xn, -yn, -1.0)
    if variant[1] == "r":
        nrm = math.sqrt(ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2])
        ray = (ray[0] / nrm, ray[1] / nrm, ray[2] / nrm)
    pc = (ray[0] * d, ray[1] * d, ray[2] * d)
    return (CR[0][0] * pc[0] + CR[0][1] * pc[1] + CR[0][2] * pc[2] + CT[0],
            CR[1][0] * pc[0] + CR[1][1] * pc[1] + CR[1][2] * pc[2] + CT[1],
            CR[2][0] * pc[0] + CR[2][1] * pc[1] + CR[2][2] * pc[2] + CT[2])


# ---------------------------------------------------------------- per-tick cube measurement
def measure(depth_arr, cu, cv, half, variant):
    u0 = int(max(0, cu - half))
    u1 = int(min(IMG_W - 1, cu + half))
    v0 = int(max(0, cv - half))
    v1 = int(min(IMG_H - 1, cv + half))
    n = 0
    su = 0.0
    sv = 0.0
    sx = 0.0
    sy = 0.0
    nt = 0
    tx = 0.0
    ty = 0.0
    for v in range(v0, v1 + 1):
        row = depth_arr[v]
        wv = PB * v + PC
        for u in range(u0, u1 + 1):
            if EXCL[0] <= u <= EXCL[2] and EXCL[1] <= v <= EXCL[3]:
                continue
            w = PA * u + wv
            if w <= 0:
                continue
            d = float(row[u])
            h = 1.0 / w - d
            if h > THR_LO and h < THR_HI:
                n += 1
                su += u
                sv += v
                if variant is not None:
                    p = deproject(u, v, d, variant)
                    sx += p[0]
                    sy += p[1]
                    if h > THR_TOP:
                        nt += 1
                        tx += p[0]
                        ty += p[1]
    if n == 0:
        return None
    if nt >= 3:
        return (n, su / n, sv / n, tx / nt, ty / nt)
    return (n, su / n, sv / n, sx / n, sy / n)


init_m = measure(depth0, cu0, cv0, 0.5 * cube_w + 4, None)
if init_m is None:
    raise ValueError("cube pixels not found by depth at t=0")
N_REF = init_m[0]
report["cube_init_px"] = [N_REF, rnd(init_m[1], 1), rnd(init_m[2], 1)]

# pick deprojection variant that reproduces the harness's own cube cloud
var_errs = {}
if cube_c3 is not None:
    best_err = None
    for variant in (("cv", "z"), ("cv", "r"), ("gl", "z"), ("gl", "r")):
        mm = measure(depth0, cu0, cv0, 0.5 * cube_w + 4, variant)
        if mm is None:
            continue
        err = math.hypot(mm[3] - cube_c3[0], mm[4] - cube_c3[1])
        var_errs[variant[0] + variant[1]] = rnd(err)
        if best_err is None or err < best_err:
            best_err = err
            VARIANT = variant
    flat = {}
    best_flat = None
    for variant in (("cv", "z"), ("cv", "r"), ("gl", "z"), ("gl", "r")):
        key = variant[0] + variant[1]
        if key not in var_errs or var_errs[key] is None or var_errs[key] > 0.03:
            continue
        zs = []
        for smp in keep[::3]:
            zs.append(deproject(smp[0], smp[1], 1.0 / smp[2], variant)[2])
        if len(zs) < 10:
            continue
        mz = sum(zs) / len(zs)
        sd = math.sqrt(sum([(z - mz) * (z - mz) for z in zs]) / len(zs))
        flat[key] = [rnd(mz), rnd(sd, 5)]
        if best_flat is None or sd < best_flat:
            best_flat = sd
            VARIANT = variant
    report["variant_flat"] = flat
    if best_flat is None or best_flat > 0.01:
        VARIANT = None
report["variant_errs"] = var_errs
report["variant"] = (VARIANT[0] + VARIANT[1]) if VARIANT is not None else "pixel"

if VARIANT is not None and cube_r3 is not None:
    ZONE = {"tx": target_cx, "ty": target_cy, "r": target_r3 + 0.25 * cube_r3}
else:
    VARIANT = None
    ZONE = {"tx": tu, "ty": tv,
            "r": 0.5 * max(target_box[2] - target_box[0], target_box[3] - target_box[1]) + 0.125 * cube_w}
ZONE["exit"] = 1.3 * ZONE["r"]
report["zone"] = [rnd(ZONE["tx"]), rnd(ZONE["ty"]), rnd(ZONE["r"]), rnd(ZONE["exit"])]
# a target partly hidden by the cube at t=0 gives a biased centre: re-segment once the cube is clear
ZONE["need_refine"] = iou(cube_box, [target_box[0] - 3, target_box[1] - 3, target_box[2] + 3,
                                     target_box[3] + 3]) > 0.0
report["target_refine_needed"] = ZONE["need_refine"]


def refine_target():
    ZONE["need_refine"] = False
    try:
        found = segment_text("front", target_prompt)
    except Exception as exc:
        report["errors"].append("refine seg: " + str(exc)[:200])
        return
    if not found.ok:
        return
    best = None
    for cand in found.segmentations[:5]:
        b = box_of(cand)
        ov = iou(b, target_box)
        if b is not None and ov > 0.3 and (best is None or ov > best[0]):
            best = (ov, b, cand)
    if best is None:
        return
    b = best[1]
    if VARIANT is not None:
        pts = cloud_points(best[2], base)
        if len(pts) < 5:
            return
        ncx = quantile([p[0] for p in pts], 0.5)
        ncy = quantile([p[1] for p in pts], 0.5)
        nr = quantile([math.hypot(p[0] - ncx, p[1] - ncy) for p in pts], 0.9)
        ZONE["tx"] = ncx
        ZONE["ty"] = ncy
        ZONE["r"] = nr + 0.25 * cube_r3
    else:
        ZONE["tx"] = 0.5 * (b[0] + b[2])
        ZONE["ty"] = 0.5 * (b[1] + b[3])
        ZONE["r"] = 0.5 * max(b[2] - b[0], b[3] - b[1]) + 0.125 * cube_w
    ZONE["exit"] = 1.3 * ZONE["r"]
    report["zone_refined"] = [len(P) - 1, rnd(ZONE["tx"]), rnd(ZONE["ty"]), rnd(ZONE["r"]), [rnd(x, 1) for x in b]]

# ---------------------------------------------------------------- tracking state
track = {"k": 0, "last": (init_m[1], init_m[2]), "vel": (0.0, 0.0), "count": 0, "in_zone": False,
         "zone_min": None, "lost": 0}
P = []        # per tick position (x, y) in the chosen space, or None
D = []        # per tick distance to target
TS = []       # per tick timestamps
arrivals = []  # refined arrival tick (float) of finalized zone visits
entries = []   # tick of zone entries
log_ticks = []
EE_Z = []


def pos_from(mm):
    if VARIANT is not None:
        return (mm[3], mm[4])
    return (mm[1], mm[2])


def refine_min(kmin):
    # parabolic refinement of squared distance around tick kmin
    if kmin <= 0 or kmin + 1 >= len(D) or D[kmin - 1] is None or D[kmin + 1] is None:
        return float(kmin)
    y0 = D[kmin - 1] ** 2
    y1 = D[kmin] ** 2
    y2 = D[kmin + 1] ** 2
    den = y0 - 2 * y1 + y2
    if den <= 0:
        return float(kmin)
    off = 0.5 * (y0 - y2) / den
    off = max(-1.0, min(1.0, off))
    return kmin + off


def update_track(obs):
    k = len(P)
    try:
        dep = obs.cameras["front"].depth_m
        ts = float(obs.timestamp_s)
    except Exception:
        dep = None
        ts = None
    TS.append(ts)
    mm = None
    if dep is not None:
        gu = track["last"][0] + track["vel"][0]
        gv = track["last"][1] + track["vel"][1]
        sp = math.hypot(track["vel"][0], track["vel"][1])
        mm = measure(dep, gu, gv, 0.65 * cube_w + sp + 3, VARIANT)
        if mm is None or mm[0] < 0.3 * N_REF:
            mm = measure(dep, gu, gv, 2.5 * cube_w + 2 * sp + 6, VARIANT)
            if mm is not None and mm[0] < 0.3 * N_REF:
                mm = None
    if mm is None:
        track["lost"] += 1
        P.append(None)
        D.append(None)
        log_ticks.append([k, rnd(ts, 3), None])
        return
    newpx = (mm[1], mm[2])
    track["vel"] = (newpx[0] - track["last"][0], newpx[1] - track["last"][1]) if k > 0 else (0.0, 0.0)
    track["last"] = newpx
    p = pos_from(mm)
    d = math.hypot(p[0] - ZONE["tx"], p[1] - ZONE["ty"])
    P.append(p)
    D.append(d)
    log_ticks.append([k, rnd(ts, 3), rnd(p[0], 4), rnd(p[1], 4), rnd(d, 4), mm[0]])
    if not track["in_zone"]:
        if d < ZONE["r"]:
            track["in_zone"] = True
            track["zone_min"] = (d, k)
            if k == 0:
                track["initial_in_zone"] = True
                note("cube starts inside target zone (not counted)")
            else:
                track["count"] += 1
                entries.append(k)
    else:
        if d < track["zone_min"][0]:
            track["zone_min"] = (d, k)
        if d > ZONE["exit"]:
            track["in_zone"] = False
            if k > 0 and not (len(entries) == 0 and track.get("initial_in_zone", False)):
                arrivals.append(refine_min(track["zone_min"][1]))
            track["initial_in_zone"] = False


def vel_now():
    k = len(P) - 1
    if k < 2 or P[k] is None or P[k - 2] is None:
        if k >= 1 and P[k] is not None and P[k - 1] is not None:
            return (P[k][0] - P[k - 1][0], P[k][1] - P[k - 1][1])
        return None
    return (0.5 * (P[k][0] - P[k - 2][0]), 0.5 * (P[k][1] - P[k - 2][1]))


def predict_remaining():
    k = len(P) - 1
    if k < 0 or P[k] is None:
        return None, None, None
    v = vel_now()
    if v is None:
        return None, None, None
    sp = math.hypot(v[0], v[1])
    if sp <= 1e-9:
        return None, None, None
    rx = P[k][0] - ZONE["tx"]
    ry = P[k][1] - ZONE["ty"]
    t_lin = -(rx * v[0] + ry * v[1]) / (sp * sp)
    best = None
    prev_a = -1.0
    for a in arrivals:
        ia = int(a)
        lo = max(0, ia - 60, int(prev_a) + 1)
        prev_a = a
        for i in range(lo, ia):
            if i + 1 >= len(P) or P[i] is None or P[i + 1] is None:
                continue
            sx = P[i + 1][0] - P[i][0]
            sy = P[i + 1][1] - P[i][1]
            sl = math.hypot(sx, sy)
            if sl <= 1e-9:
                continue
            if sx * v[0] + sy * v[1] < 0.5 * sl * sp:
                continue
            s = ((P[k][0] - P[i][0]) * sx + (P[k][1] - P[i][1]) * sy) / (sl * sl)
            s = max(0.0, min(1.0, s))
            dist = math.hypot(P[i][0] + s * sx - P[k][0], P[i][1] + s * sy - P[k][1])
            rem = a - (i + s)
            if rem < 0:
                continue
            ratio = max(0.5, min(2.0, sl / sp))
            if best is None or dist < best[0] - 1e-9 or (abs(dist - best[0]) < 1e-9 and a > best[2]):
                best = (dist, rem * ratio, a)
    t_match = None
    if best is not None and best[0] < 0.6 * ZONE["r"]:
        t_match = best[1]
    return t_match, t_lin, best[0] if best is not None else None


# ---------------------------------------------------------------- motion helpers
# (ended flag defined before the identification hold)



def do_step(q):
    cmd = ArmCommand(mode="joint_position", target=[float(x) for x in q], gripper_position=0.0,
                     embodiment=emb)
    reply = step(RobotAction(arms={"primary": cmd}))
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
    obs = None
    try:
        obs = reply.observation
    except Exception:
        obs = None
    if term or trunc:
        ended["flag"] = True
        ended["why"] = "terminated" if term else "truncated"
        ended["tick"] = len(P)
    if obs is None and not ended["flag"]:
        obs = get_observation()
    if obs is not None:
        update_track(obs)
        if ZONE["need_refine"] and not ended["flag"]:
            lp = track["last"]
            tcu = 0.5 * (target_box[0] + target_box[2])
            tcv = 0.5 * (target_box[1] + target_box[3])
            clear_u = 0.5 * (target_box[2] - target_box[0]) + 0.5 * cube_w + 2
            clear_v = 0.5 * (target_box[3] - target_box[1]) + 0.5 * cube_w + 2
            if abs(lp[0] - tcu) > clear_u or abs(lp[1] - tcv) > clear_v:
                refine_target()
        try:
            rs = obs.robot_state
            EE_Z.append(float(rs.end_effector_poses["primary"].position[2]))
        except Exception:
            EE_Z.append(None)
    return obs


def robot_from(obs):
    try:
        return obs.robot_state
    except Exception:
        return get_robot_state()


def ik_for(pos, quat, seed):
    res = solve_ik(Pose(position=[float(pos[0]), float(pos[1]), float(pos[2])], quaternion_wxyz=quat,
                        frame=base), seed_joints=seed)
    if not res.ok:
        try:
            why = str(res.error)[:200]
        except Exception:
            why = "unknown"
        raise ValueError("IK failed for " + str([rnd(x) for x in pos]) + ": " + why)
    try:
        sol = res.joint_positions
    except Exception:
        sol = res.solution
    return [float(x) for x in sol]


# initial observation counts as tick 0 of tracking
update_track(obs0)
EE_Z.append(float(state0.end_effector_poses["primary"].position[2]))
# replay the identification-hold observations so that every tick since t=0 is tracked
for hob in HOLD_OBS:
    if hob is None:
        continue
    update_track(hob)
    try:
        EE_Z.append(float(hob.robot_state.end_effector_poses["primary"].position[2]))
    except Exception:
        EE_Z.append(None)

home_q = [float(x) for x in state0.joint_positions["primary"]]
home_quat = [float(x) for x in state0.end_effector_poses["primary"].quaternion_wxyz]
HM = quat_matrix(home_quat)
report["home_tool_z_axis"] = [rnd(HM[0][2]), rnd(HM[1][2]), rnd(HM[2][2])]
press_quat = home_quat if HM[2][2] < -0.9 else [0.0, 1.0, 0.0, 0.0]

z_hover = button_top + HOVER_ABOVE_TOP
z_press = button_top - PRESS_BELOW_TOP
D_STEP = z_hover - z_press
z_pre = z_hover + D_STEP
q_pre = ik_for((button_x, button_y, z_pre), press_quat, home_q)
q_hover = ik_for((button_x, button_y, z_hover), press_quat, q_pre)
q_press = ik_for((button_x, button_y, z_press), press_quat, q_hover)
report["targets_z"] = [rnd(z_pre), rnd(z_hover), rnd(z_press)]

# ---------------------------------------------------------------- transit to pre-hover
maxdq = max([abs(q_pre[i] - home_q[i]) for i in range(7)])
nsteps = max(1, int(math.ceil(maxdq / MAX_JOINT_STEP)))
report["transit_ticks"] = nsteps
for s in range(1, nsteps + 1):
    f = s / float(nsteps)
    q = [home_q[i] + f * (q_pre[i] - home_q[i]) for i in range(7)]
    do_step(q)
    if ended["flag"]:
        break
settle_pre = 0
last_obs = None
while not ended["flag"] and settle_pre < 30:
    last_obs = do_step(q_pre)
    settle_pre += 1
    if len(EE_Z) >= 2 and EE_Z[-1] is not None and EE_Z[-2] is not None:
        if abs(EE_Z[-1] - z_pre) < 0.004 and abs(EE_Z[-1] - EE_Z[-2]) < 0.0008:
            break
report["settle_pre_ticks"] = settle_pre
try:
    eep = robot_from(last_obs).end_effector_poses["primary"].position
    report["pre_err"] = [rnd(eep[0] - button_x), rnd(eep[1] - button_y), rnd(eep[2] - z_pre)]
except Exception:
    pass

# ---------------------------------------------------------------- rehearsal descent = latency
L_OBS = None
if not ended["flag"]:
    k0 = len(EE_Z) - 1
    z_start = EE_Z[k0]
    rehearsal = []
    for s in range(30):
        do_step(q_hover)
        rehearsal.append(rnd(EE_Z[-1], 5))
        if ended["flag"]:
            break
        if s >= 4 and abs(EE_Z[-1] - EE_Z[-2]) < 0.0004 and abs(EE_Z[-1] - z_hover) < 0.004:
            break
    z_end = EE_Z[-1]
    frac = (HOVER_ABOVE_TOP + CONTACT_FRACTION_P * PRESS_BELOW_TOP) / D_STEP
    level = z_start - frac * (z_start - z_end)
    for j in range(k0 + 1, len(EE_Z)):
        if EE_Z[j] is not None and EE_Z[j - 1] is not None and EE_Z[j] <= level:
            dz = EE_Z[j - 1] - EE_Z[j]
            fr = (EE_Z[j - 1] - level) / dz if dz > 1e-9 else 1.0
            L_OBS = (j - 1 - k0) + fr
            break
    report["rehearsal"] = {"z_start": rnd(z_start, 5), "z_end": rnd(z_end, 5), "level": rnd(level, 5),
                           "zs": rehearsal, "L_obs": rnd(L_OBS, 3), "start_tick": k0}
L_TICKS = None
if L_OBS is not None:
    L_TICKS = int(math.ceil(L_OBS - 1e-6))
    report["L_ticks"] = L_TICKS
ready_tick = len(P) - 1
report["ready_tick"] = ready_tick
report["count_at_ready"] = track["count"]
if L_OBS is None and not ended["flag"]:
    raise ValueError("press latency could not be measured from the rehearsal descent")

# ---------------------------------------------------------------- wait for the N-th approach
press_tick = None
pred_log = []
if target_count is None:
    report["errors"].append("could not parse count from goal")
limit_ticks = (max_steps - 5) if max_steps is not None else 900
while not ended["flag"] and target_count is not None and len(P) < limit_ticks:
    kk = len(P) - 1
    approaching = track["count"] == target_count - 1 and not track["in_zone"]
    in_nth = track["count"] == target_count and track["in_zone"]
    if track["count"] > target_count or (track["count"] == target_count and not track["in_zone"]):
        note("missed: count passed N before press at tick " + str(kk))
        break
    if in_nth and D[kk] is not None and D[kk] > track["zone_min"][0] + 0.15 * ZONE["r"]:
        press_tick = kk
        report["press_prediction"] = {"tick": press_tick, "late": True, "L_obs": rnd(L_OBS, 3)}
        note("late press: closest approach already passed at tick " + str(kk))
        break
    if approaching or in_nth:
        t_match, t_lin, mdist = predict_remaining()
        t_use = t_match if t_match is not None else (t_lin if t_lin is not None and t_lin > 0 else None)
        pred_log.append([len(P) - 1, rnd(t_match, 2), rnd(t_lin, 2), rnd(mdist, 4)])
        if t_use is not None and t_use < 40 and t_use - L_TICKS - AIM_LEAD_TICKS <= 0.5:
            press_tick = len(P) - 1
            report["press_prediction"] = {"tick": press_tick, "t_match": rnd(t_match, 3),
                                          "t_lin": rnd(t_lin, 3), "L_obs": rnd(L_OBS, 3),
                                          "L_ticks": L_TICKS}
            break
    do_step(q_hover)

# ---------------------------------------------------------------- press
press_z = []
if press_tick is not None and not ended["flag"]:
    for s in range(25):
        do_step(q_press)
        press_z.append(rnd(EE_Z[-1], 5) if len(EE_Z) > 0 else None)
        if ended["flag"]:
            break
    if not ended["flag"]:
        note("press did not end the episode; lifting back to hover")
        for s in range(10):
            do_step(q_hover)
            if ended["flag"]:
                break

report["press_z"] = press_z
report["ended"] = ended
report["count"] = track["count"]
report["entries"] = entries
report["arrivals"] = [rnd(a, 2) for a in arrivals]
report["lost_ticks"] = track["lost"]
report["pred_log"] = pred_log[-40:]
report["ticks"] = log_ticks
report["ee_z"] = [rnd(z, 4) for z in EE_Z]
result = report
