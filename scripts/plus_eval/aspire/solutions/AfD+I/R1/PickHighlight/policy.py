# PickHighlight v01_track_swaps -- ASPIRE task worker, robomme_plus/robomme_plus/task_6.
# Goal: "first press the button, then pick up all highlighted cubes one by one, finally press the
# button to stop".
# v00 (run attempt_001, episode 0) failed: the cubes are permuted by timed swap events after the
# highlight ends, so the highlight-time positions went stale; v00's third pick lifted a
# non-highlighted cube and the episode terminated (task_success false) at step 470.
# v01 keeps v00's highlight detector, inventory and button logic, and adds:
#   * one control tick at a time for EVERY motion (solve_ik + joint interpolation via step()),
#     with a front observation and a template-tracker update after each tick;
#   * swap-event detection while the arm is parked out of the front view, an online estimate of
#     the swap period, and an action scheduler that only starts a press/pick when it fits
#     before the next predicted swap;
#   * the pick target is re-localised from the tracked pixel at pick time (segment_points).
# Nothing is copied from the recordings: periods, positions and counts are measured online.
import math

result = {"version": "v02_table_color_floor", "errors": [], "motions": [], "phases": {}, "notes": []}
R = result
STATE = {"ended": False, "end_reason": None, "t0": None, "hz": 20.0, "max_steps": None}

WHITE_MIN = 175        # min channel of a "white" pixel (highlight disc)
WHITE_SPREAD = 40      # max-min channel spread of a "white" pixel
DIFF_MIN = 60          # summed |F-B| needed for a pixel to count as newly white
ON_MIN_PIXELS = 40     # newly-white pixel count that marks a highlight frame
DISC_MIN_AREA = 12     # min pixels in one disc component
HOLD_MAX_TICKS = 80    # never watch for highlights longer than this
HOLD_ON_FRAMES = 3     # highlight frames to collect before moving on
FINAL_RESERVE = 60     # ticks kept for the final button press (a skipped pick fails anyway)
HOVER_DZ = 0.12        # hover above cube top
LIFT_DZ = 0.12         # lift above grasp height
GRIP_HELD_MIN = 0.08   # gripper reading above this after close = something held


def err(where, exc):
    R["errors"].append(str(where) + ": " + str(exc)[:300])
    text = str(exc).lower()
    if "episode has ended" in text or "execution is closing" in text:
        STATE["ended"] = True
        STATE["end_reason"] = "raised: " + str(exc)[:120]


def fl(v, nd=4):
    try:
        return round(float(v), nd)
    except Exception:
        return None


def vec(v, nd=4):
    try:
        return [round(float(x), nd) for x in v]
    except Exception:
        return None


def as_list(arr):
    try:
        return arr.tolist()
    except Exception:
        return [list(row) for row in arr]


def check_terminal(res, where):
    flags = {}
    for name in ("terminated", "truncated"):
        try:
            if name == "terminated":
                val = res.terminated
            else:
                val = res.truncated
            flags[name] = bool(val)
        except Exception:
            flags[name] = None
    if flags.get("terminated") or flags.get("truncated"):
        STATE["ended"] = True
        STATE["end_reason"] = str(where) + " " + str(flags)
    return flags


def now_tick():
    try:
        st = get_robot_state()
        ts = float(st.timestamp_s)
        if STATE["t0"] is None:
            STATE["t0"] = ts
        return int(round((ts - STATE["t0"]) * STATE["hz"]))
    except Exception as exc:
        err("now_tick", exc)
        return -1


def ee_pos():
    try:
        return vec(get_robot_state().end_effector_poses["primary"].position)
    except Exception:
        return None


def grip_reading():
    try:
        return float(get_robot_state().gripper_positions["primary"])
    except Exception:
        return None


# ---------------------------------------------------------------- quaternion helpers
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def qmat(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def qyaw(q, delta):
    return qmul([math.cos(delta / 2.0), 0.0, 0.0, math.sin(delta / 2.0)], q)


def wrap90(a):
    # wrap an angle (rad) into (-pi/4, pi/4]
    q = math.pi / 2.0
    while a > q / 2.0:
        a -= q
    while a <= -q / 2.0:
        a += q
    return a


# ---------------------------------------------------------------- image helpers (pure Python)
def white_flags(img):
    flags = []
    for row in img:
        for p in row:
            r, g, b = p[0], p[1], p[2]
            mn = min(r, g, b)
            flags.append(mn >= WHITE_MIN and max(r, g, b) - mn <= WHITE_SPREAD)
    return flags


def new_white(base_img, base_white, img):
    """Flat list of booleans: pixel is white now, was not white and differs from baseline."""
    out = []
    i = 0
    for y in range(len(img)):
        brow = base_img[y]
        row = img[y]
        for x in range(len(row)):
            p = row[x]
            hit = False
            if not base_white[i]:
                r, g, b = p[0], p[1], p[2]
                mn = min(r, g, b)
                if mn >= WHITE_MIN and max(r, g, b) - mn <= WHITE_SPREAD:
                    q = brow[x]
                    if abs(r - q[0]) + abs(g - q[1]) + abs(b - q[2]) >= DIFF_MIN:
                        hit = True
            out.append(hit)
            i += 1
    return out


def components(flags, width, height, min_area, eight=True):
    seen = set()
    comps = []
    for start in range(len(flags)):
        if not flags[start] or start in seen:
            continue
        stack = [start]
        seen.add(start)
        pix = []
        while stack:
            k = stack.pop()
            pix.append(k)
            ky = k // width
            kx = k - ky * width
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    if not eight and dx != 0 and dy != 0:
                        continue
                    nx = kx + dx
                    ny = ky + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        n = ny * width + nx
                        if flags[n] and n not in seen:
                            seen.add(n)
                            stack.append(n)
        if len(pix) >= min_area:
            comps.append(pix)
    return comps


def comp_summary(pix, width):
    xs = [k % width for k in pix]
    ys = [k // width for k in pix]
    return {"area": len(pix), "bbox": [min(xs), min(ys), max(xs), max(ys)],
            "centroid": [round(sum(xs) / float(len(xs)), 1), round(sum(ys) / float(len(ys)), 1)]}


HOLE_TABLE_L1 = 15.0   # v02: min mean L1 distance from the table colour for a highlighted-cube hole


def detect_highlights(base_img, base_white, frames, width, height, max_gap=18):
    """frames: images that show the highlight. A highlighted cube is a run of non-white,
    non-table pixels enclosed left and right (same row, run <= max_gap px) by newly-white
    disc pixels. The table colour is learned from the baseline colour of the disc pixels
    (they were bare table before the highlight appeared). Returns disc + cube records."""
    votes = [0] * (width * height)
    for img in frames:
        nw = new_white(base_img, base_white, img)
        for i in range(len(nw)):
            if nw[i]:
                votes[i] += 1
    need = max(1, (len(frames) + 1) // 2)
    flags = [v >= need for v in votes]
    comps = components(flags, width, height, DISC_MIN_AREA)
    mask = [False] * (width * height)
    for pix in comps:
        for k in pix:
            mask[k] = True
    # table colour model (robust: per-channel median, median L1 deviation) from the baseline
    # colour under the disc pixels -- mostly bare table before the highlight appeared
    samples = []
    for pix in comps:
        for k in pix:
            samples.append(base_img[k // width][k % width])
    table = {"mean": None, "mad": None, "thr": None}
    tm = None
    thr = 45.0
    if len(samples) >= 10:
        tm = [quant([q[0] for q in samples], 0.5), quant([q[1] for q in samples], 0.5),
              quant([q[2] for q in samples], 0.5)]
        mad = quant([abs(q[0] - tm[0]) + abs(q[1] - tm[1]) + abs(q[2] - tm[2]) for q in samples], 0.5)
        thr = max(45.0, 3.0 * mad)
        table = {"mean": [round(float(v), 1) for v in tm], "mad": round(float(mad), 1), "thr": round(thr, 1)}
    hole = [False] * (width * height)
    img = frames[len(frames) // 2]
    for y in range(height):
        last = -1
        for x in range(width):
            k = y * width + x
            if mask[k]:
                gap = x - last - 1
                if last >= 0 and 0 < gap <= max_gap:
                    for xx in range(last + 1, x):
                        hole[y * width + xx] = True
                last = x
    raw = components(hole, width, height, 4, eight=False)
    cubes = []
    for h in raw:
        hs = comp_summary(h, width)
        # colour test: mean colour of the enclosed run vs the table model
        cr = 0.0
        cg = 0.0
        cb = 0.0
        for k in h:
            q = img[k // width][k % width]
            cr += q[0]
            cg += q[1]
            cb += q[2]
        cm = [cr / len(h), cg / len(h), cb / len(h)]
        cdist = 0.0
        if tm is not None:
            cdist = abs(cm[0] - tm[0]) + abs(cm[1] - tm[1]) + abs(cm[2] - tm[2])
        # support test: disc pixels below the run (the front part of a cube's disc is
        # visible below the cube; a gap between two side-by-side discs has none)
        cols = sorted(set(k % width for k in h))
        ybot = hs["bbox"][3]
        sup = 0
        for x in cols:
            for yy in range(ybot + 1, min(height, ybot + 1 + max_gap)):
                if mask[yy * width + x]:
                    sup += 1
                    break
        hs["color_dist"] = round(cdist, 1)
        hs["support"] = round(sup / float(len(cols)), 2)
        hs["px"] = hs["centroid"]
        # v02: a hole whose mean colour is within HOLE_TABLE_L1 of the table median is bare table
        # (gap between two discs), not a cube, whatever its support (live run ep0: L1 3.9, support 1.0)
        hs["accept"] = ((hs["support"] >= 0.5) or (cdist > min(thr, 90.0))) and (tm is None or cdist >= HOLE_TABLE_L1)
        cubes.append(hs)
    accepted = [c for c in cubes if c["accept"]]
    if accepted:
        big = max(c["area"] for c in accepted)
        accepted = [c for c in accepted if c["area"] >= max(4, 0.2 * big)]
    rejected = [c for c in cubes if c not in accepted]
    discs = []
    for pix in comps:
        rec = comp_summary(pix, width)
        discs.append(rec)
    return {"discs": discs, "cubes": accepted, "rejected": rejected[:20], "table": table, "max_gap": max_gap}


# ---------------------------------------------------------------- geometry helpers
def cloud_points(cloud, cap=2500):
    try:
        pts = as_list(cloud.points)
    except Exception:
        return []
    if len(pts) > cap:
        stride = len(pts) // cap + 1
        pts = pts[::stride]
    out = []
    for p in pts:
        try:
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            if x == x and y == y and z == z:
                out.append([x, y, z])
        except Exception:
            pass
    return out


def quant(values, f):
    s = sorted(values)
    if not s:
        return None
    return s[int((len(s) - 1) * f)]


def cloud_stats(pts):
    if len(pts) < 3:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    return {"n": len(pts), "cx": quant(xs, 0.5), "cy": quant(ys, 0.5),
            "ztop": quant(zs, 0.95), "zlow": quant(zs, 0.05),
            "ext": [quant(xs, 0.95) - quant(xs, 0.05), quant(ys, 0.95) - quant(ys, 0.05),
                    quant(zs, 0.95) - quant(zs, 0.05)]}


def top_face_fit(pts):
    """Centre, yaw (mod 90 deg) and side of a cube from its top-face points."""
    zs = [p[2] for p in pts]
    if len(zs) < 10:
        return None
    zt = quant(zs, 0.95)
    top = [p for p in pts if p[2] >= zt - 0.008]
    if len(top) < 8:
        return None
    cx = sum(p[0] for p in top) / len(top)
    cy = sum(p[1] for p in top) / len(top)
    best = None
    for deg10 in range(0, 900, 30):
        a = math.radians(deg10 / 10.0)
        ca = math.cos(a)
        sa = math.sin(a)
        us = [(p[0] - cx) * ca + (p[1] - cy) * sa for p in top]
        vs = [-(p[0] - cx) * sa + (p[1] - cy) * ca for p in top]
        eu = quant(us, 0.97) - quant(us, 0.03)
        ev = quant(vs, 0.97) - quant(vs, 0.03)
        area = eu * ev
        if best is None or area < best[0]:
            best = (area, a, eu, ev)
    a0 = best[1]
    for step_deg in (-2, -1, 1, 2):
        a = a0 + math.radians(step_deg)
        ca = math.cos(a)
        sa = math.sin(a)
        us = [(p[0] - cx) * ca + (p[1] - cy) * sa for p in top]
        vs = [-(p[0] - cx) * sa + (p[1] - cy) * ca for p in top]
        eu = quant(us, 0.97) - quant(us, 0.03)
        ev = quant(vs, 0.97) - quant(vs, 0.03)
        if eu * ev < best[0]:
            best = (eu * ev, a, eu, ev)
    # centre of the fitted rectangle (robust to uneven sampling)
    a = best[1]
    ca = math.cos(a)
    sa = math.sin(a)
    us = [(p[0] - cx) * ca + (p[1] - cy) * sa for p in top]
    vs = [-(p[0] - cx) * sa + (p[1] - cy) * ca for p in top]
    um = 0.5 * (quant(us, 0.97) + quant(us, 0.03))
    vm = 0.5 * (quant(vs, 0.97) + quant(vs, 0.03))
    rx = cx + um * ca - vm * sa
    ry = cy + um * sa + vm * ca
    return {"cx": rx, "cy": ry, "ztop": sum(p[2] for p in top) / len(top), "yaw": a,
            "side": 0.5 * (best[2] + best[3]), "n_top": len(top)}


# ---------------------------------------------------------------- perception wrappers
def seg_candidates(camera, prompt, limit=30):
    out = []
    try:
        seg = segment_text(camera, prompt)
        if not seg.ok:
            return out, {"prompt": prompt, "ok": False, "error": str(seg.error)[:160]}
        cands = list(seg.segmentations)[:limit]
        info = {"prompt": prompt, "ok": True, "n": len(cands)}
        for c in cands:
            box = None
            try:
                if c.box_xyxy is not None:
                    box = [float(v) for v in c.box_xyxy]
            except Exception:
                box = None
            sc = None
            try:
                sc = float(c.score)
            except Exception:
                sc = None
            out.append({"cand": c, "box": box, "score": sc, "prompt": prompt})
        return out, info
    except Exception as exc:
        err("segment_text " + camera + " " + prompt, exc)
        return out, {"prompt": prompt, "ok": False, "exc": str(exc)[:160]}


def cand_cloud(cand, camera):
    try:
        pc = mask_to_point_cloud(cand, camera, target_frame=BASE)
        return cloud_points(pc)
    except Exception as exc:
        err("mask_to_point_cloud " + camera, exc)
        return []


def box_center(box):
    return [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]


def box_frac_white(box, wflags, width, height):
    if box is None:
        return 0.0
    x0 = max(0, int(box[0]))
    y0 = max(0, int(box[1]))
    x1 = min(width - 1, int(box[2]))
    y1 = min(height - 1, int(box[3]))
    tot = 0
    wh = 0
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            tot += 1
            if wflags[y * width + x]:
                wh += 1
    if tot == 0:
        return 0.0
    return wh / float(tot)


# ---------------------------------------------------------------- control wrappers
def motion_record(res, label, extra):
    rec = {"label": label}
    for k in extra:
        rec[k] = extra[k]
    try:
        rec["ok"] = bool(res.ok)
    except Exception:
        rec["ok"] = None
    try:
        rec["status"] = str(res.status)
    except Exception:
        pass
    try:
        rec["steps"] = int(res.steps_executed)
    except Exception:
        rec["steps"] = None
    try:
        if res.error is not None:
            rec["error"] = str(res.error)[:240]
    except Exception:
        pass
    rec["term"] = check_terminal(res, label)
    return rec


def move(xyz, quat, label, tol=0.015, max_steps=200):
    if STATE["ended"]:
        return None
    rec = None
    res = None
    try:
        pose = Pose([float(xyz[0]), float(xyz[1]), float(xyz[2])], [float(q) for q in quat], BASE)
        res = move_to_pose(pose, tolerance=tol, max_steps=max_steps)
        rec = motion_record(res, label, {"target": vec(xyz), "max_steps": max_steps})
        if (not rec["ok"]) and rec.get("steps") == 0 and not STATE["ended"]:
            # nothing executed (e.g. plan longer than max_steps): one larger-budget attempt
            res = move_to_pose(pose, tolerance=tol, max_steps=max_steps * 3)
            rec2 = motion_record(res, label + "_retry", {"max_steps": max_steps * 3})
            rec["retry"] = rec2
            rec["ok"] = rec2["ok"]
    except Exception as exc:
        err("move " + label, exc)
        rec = {"label": label, "exception": str(exc)[:200]}
    if not STATE["ended"]:
        rec["tick"] = now_tick()
        rec["ee"] = ee_pos()
    R["motions"].append(rec)
    return rec


def grip(open_it, label):
    if STATE["ended"]:
        return None
    rec = None
    try:
        if open_it:
            res = open_gripper()
        else:
            res = close_gripper()
        rec = motion_record(res, label, {})
    except Exception as exc:
        err("grip " + label, exc)
        rec = {"label": label, "exception": str(exc)[:200]}
    if not STATE["ended"]:
        rec["tick"] = now_tick()
        rec["grip"] = fl(grip_reading())
    R["motions"].append(rec)
    return rec


def hold_tick(joints, gpos):
    st = get_robot_state()
    res = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=joints, gripper_position=gpos,
        embodiment=st.embodiment)}))
    check_terminal(res, "hold_tick")
    return res



def finger_clearance(cx, cy, ang, half, neigh):
    best = 1.0
    for sgn in (-1.0, 1.0):
        fx = cx + sgn * (half + 0.012) * math.cos(ang)
        fy = cy + sgn * (half + 0.012) * math.sin(ang)
        for n in neigh:
            d = math.sqrt((fx - n[0]) ** 2 + (fy - n[1]) ** 2)
            if d < best:
                best = d
    return best


# ================================================================ v01: per-tick tracking + servo control
# Root cause of the v00 failure (run attempt_001, episode 0): after the highlight ends the cubes are
# permuted in timed "swap" events (observed onsets at ticks 56, ~196, ~336; ~40 ticks each). v00 used
# the highlight-time positions, so at pick time one of its targets held a non-highlighted cube and
# lifting it ended the episode. v01 observes the front camera on EVERY control tick, tracks each
# highlighted cube with an appearance template, schedules arm actions between swap events and keeps
# the arm parked out of the front view while a swap is in progress.
TRK_HALF = 4           # template half size (9x9 px core of a cube)
TRK_RADIUS = 4         # per-tick search radius around the predicted position (px)
TRK_STILL = 18.0       # mean L1 (3 channels) below which a track is considered unmoved
TRK_LOST = 70.0        # mean L1 above which the best match is rejected (occluded)
TRK_LAM = 1.5          # penalty per px of deviation from the constant-velocity prediction
TRK_SEP = 6.0          # two tracks may not claim positions closer than this (px)
TRK_OCC = 0.2          # fraction of grey (arm) pixels around a track that freezes it
TRK_COAST = 12         # ticks a lost track keeps coasting on its last velocity
MOT_DIFF = 45          # per-pixel L1 change that counts as motion (frame to frame)
MOT_ON = 4             # changed sampled pixels that mean "something is moving"
QUIET_TICKS = 8        # consecutive quiet ticks that end a swap event
PARK_Z = 0.40          # park height (checked online for visibility; falls back to home)
LIFT_DZ1 = 0.10        # v01 lift above grasp height
HOVER_DZ1 = 0.08       # v01 hover above cube top
SPEED = 0.05           # max joint change per tick for transit (move_to_pose used ~0.049)
SPEED_SLOW = 0.03      # descent / contact segments

TR = {"tracks": [], "prev": None, "free": True, "mc": 0, "n": 0, "gray0": None, "blind": 0,
      "q_hold": None, "g_cmd": 1.0, "emb": None}


def front_img():
    return as_list(get_observation(camera_names=[FRONT]).cameras[FRONT].rgb)


def motion_count(a, b):
    c = 0
    for y in range(0, len(a), 2):
        ra = a[y]
        rb = b[y]
        for x in range(0, len(ra), 2):
            p = ra[x]
            q = rb[x]
            if abs(p[0] - q[0]) + abs(p[1] - q[1]) + abs(p[2] - q[2]) > MOT_DIFF:
                c += 1
    return c


def gray_count(img):
    c = 0
    for y in range(0, len(img), 2):
        row = img[y]
        for x in range(0, len(row), 2):
            p = row[x]
            if max(p) - min(p) <= 22:
                c += 1
    return c


def tpatch(img, cx, cy, half):
    out = []
    h = len(img)
    w = len(img[0])
    for y in range(cy - half, cy + half + 1):
        row = img[min(h - 1, max(0, y))]
        for x in range(cx - half, cx + half + 1):
            out.append(row[min(w - 1, max(0, x))])
    return out


def tssd(img, cx, cy, ref, half):
    h = len(img)
    w = len(img[0])
    if cy - half < 0 or cy + half >= h or cx - half < 0 or cx + half >= w:
        return 1e9
    s = 0
    k = 0
    for y in range(cy - half, cy + half + 1):
        row = img[y]
        for x in range(cx - half, cx + half + 1):
            p = row[x]
            q = ref[k]
            k += 1
            s += abs(p[0] - q[0]) + abs(p[1] - q[1]) + abs(p[2] - q[2])
    return s / float(k)


def gray_frac(img, cx, cy, half):
    h = len(img)
    w = len(img[0])
    n = 0
    g = 0
    for y in range(max(0, cy - half - 2), min(h, cy + half + 3)):
        row = img[y]
        for x in range(max(0, cx - half - 2), min(w, cx + half + 3)):
            p = row[x]
            n += 1
            if max(p) - min(p) <= 22:
                g += 1
    return g / float(max(1, n))


def track_tick(tracks, img, free):
    order = sorted([(t.get("last", 0.0), i) for i, t in enumerate(tracks)])
    claimed = []
    for item in order:
        t = tracks[item[1]]
        cx = t["p"][0]
        cy = t["p"][1]
        if gray_frac(img, cx, cy, TRK_HALF) > TRK_OCC:
            t["occ"] = t.get("occ", 0) + 1
            t["v"] = [0.0, 0.0]
            claimed.append((cx, cy))
            continue
        s0 = tssd(img, cx, cy, t["ref"], TRK_HALF)
        if (not free) or (s0 <= TRK_STILL and math.sqrt(t["v"][0] ** 2 + t["v"][1] ** 2) < 0.75):
            t["v"] = [0.0, 0.0]
            t["last"] = s0
            claimed.append((cx, cy))
            continue
        px = int(round(cx + t["v"][0]))
        py = int(round(cy + t["v"][1]))
        rad = TRK_RADIUS + (2 if t.get("lost_run", 0) > 0 else 0)
        best = None
        for oy in range(-rad, rad + 1):
            for ox in range(-rad, rad + 1):
                qx = px + ox
                qy = py + oy
                bad = False
                for c in claimed:
                    if math.sqrt((qx - c[0]) ** 2 + (qy - c[1]) ** 2) < TRK_SEP:
                        bad = True
                        break
                if bad:
                    continue
                s = tssd(img, qx, qy, t["ref"], TRK_HALF) + TRK_LAM * math.sqrt(ox * ox + oy * oy)
                if best is None or s < best[0]:
                    best = (s, qx, qy)
        if best is None or best[0] > TRK_LOST:
            t["lost"] = t.get("lost", 0) + 1
            t["lost_run"] = t.get("lost_run", 0) + 1
            if t["lost_run"] <= TRK_COAST:
                t["p"] = [px, py]
            else:
                t["v"] = [0.0, 0.0]
            t["last"] = TRK_LOST
            continue
        t["lost_run"] = 0
        t["moved"] = t.get("moved", 0) + 1
        nv = [best[1] - cx, best[2] - cy]
        t["v"] = [0.7 * nv[0] + 0.3 * t["v"][0], 0.7 * nv[1] + 0.3 * t["v"][1]]
        t["p"] = [best[1], best[2]]
        t["last"] = best[0]
        claimed.append((best[1], best[2]))
    return tracks


def recenter_tracks(img, radius, refresh, only=None):
    # local re-search (static scene) for every track; optionally refresh its template
    out = []
    for i in range(len(TR["tracks"])):
        t = TR["tracks"][i]
        if only is not None and i not in only:
            continue
        others = [o["p"] for j, o in enumerate(TR["tracks"]) if j != i]
        best = None
        for oy in range(-radius, radius + 1):
            for ox in range(-radius, radius + 1):
                qx = t["p"][0] + ox
                qy = t["p"][1] + oy
                bad = False
                for o in others:
                    if math.sqrt((qx - o[0]) ** 2 + (qy - o[1]) ** 2) < TRK_SEP:
                        bad = True
                        break
                if bad:
                    continue
                s = tssd(img, qx, qy, t["ref"], TRK_HALF) + 0.5 * math.sqrt(ox * ox + oy * oy)
                if best is None or s < best[0]:
                    best = (s, qx, qy)
        rec = {"i": i, "from": list(t["p"])}
        if best is not None and best[0] <= TRK_LOST:
            t["p"] = [best[1], best[2]]
            t["last"] = best[0]
            t["v"] = [0.0, 0.0]
            t["lost_run"] = 0
            t["uncertain"] = False
            if refresh and gray_frac(img, best[1], best[2], TRK_HALF) <= TRK_OCC:
                t["ref"] = tpatch(img, best[1], best[2], TRK_HALF)
        else:
            t["uncertain"] = True
        rec["to"] = list(t["p"])
        rec["ssd"] = None if best is None else fl(best[0], 1)
        out.append(rec)
    return out


def track_snapshot():
    return [{"p": list(t["p"]), "done": t.get("done", False), "unc": t.get("uncertain", False),
             "lost": t.get("lost", 0), "occ": t.get("occ", 0), "moved": t.get("moved", 0)} for t in TR["tracks"]]


def joints_now():
    return [float(v) for v in get_robot_state().joint_positions["primary"]]


def tick(q, g):
    # exactly one control tick, then one front observation and one tracker update
    if STATE["ended"]:
        return False
    try:
        res = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=q, gripper_position=g, embodiment=TR["emb"])}))
        check_terminal(res, "tick")
    except Exception as exc:
        err("tick", exc)
        return False
    TR["n"] += 1
    TR["q_hold"] = q
    TR["g_cmd"] = g
    if STATE["ended"]:
        return False
    try:
        img = front_img()
        if TR["prev"] is not None:
            TR["mc"] = motion_count(TR["prev"], img)
        TR["prev"] = img
        if TR["tracks"]:
            track_tick(TR["tracks"], img, TR["free"])
    except Exception as exc:
        err("tick obs", exc)
    return True


def ik_joints(ik):
    v = None
    try:
        if not ik.ok:
            return None
    except Exception:
        return None
    try:
        v = ik.joint_positions
    except Exception:
        v = None
    if v is None:
        try:
            v = ik.solution
        except Exception:
            v = None
    if v is None:
        return None
    try:
        q = [float(x) for x in v]
        if len(q) >= 7:
            return q[:7]
    except Exception:
        pass
    try:
        q = [float(x) for x in v["primary"]]
        if len(q) >= 7:
            return q[:7]
    except Exception:
        pass
    return None


def servo_q(qt, g, label, speed=SPEED, settle=8, tol=0.012, cap=90):
    rec = {"label": label, "mode": "servo"}
    if STATE["ended"] or qt is None:
        rec["skipped"] = True
        R["motions"].append(rec)
        return rec
    t0 = TR["n"]
    try:
        q0 = joints_now()
        d = max(abs(a - b) for a, b in zip(q0, qt))
        n = max(1, min(cap, int(math.ceil(d / speed))))
        rec["n"] = n
        for i in range(1, n + 1):
            f = i / float(n)
            if not tick([a + (b - a) * f for a, b in zip(q0, qt)], g):
                break
        k = 0
        while k < settle and not STATE["ended"]:
            qm = joints_now()
            if max(abs(a - b) for a, b in zip(qm, qt)) < tol:
                break
            tick(qt, g)
            k += 1
        rec["settle"] = k
    except Exception as exc:
        err("servo " + label, exc)
        rec["exception"] = str(exc)[:200]
    rec["ticks"] = TR["n"] - t0
    if not STATE["ended"]:
        rec["ee"] = ee_pos()
    R["motions"].append(rec)
    return rec


def servo_pose(xyz, quat, g, label, speed=SPEED, settle=8, tol=0.012, allow_blind=True):
    if STATE["ended"]:
        return None
    q = None
    try:
        pose = Pose([float(xyz[0]), float(xyz[1]), float(xyz[2])], [float(v) for v in quat], BASE)
        ik = solve_ik(pose, seed_joints=joints_now())
        q = ik_joints(ik)
        if q is None:
            try:
                R["notes"].append("solve_ik failed for " + label + ": " + str(ik.error)[:120])
            except Exception:
                R["notes"].append("solve_ik failed for " + label)
    except Exception as exc:
        err("solve_ik " + label, exc)
        q = None
    if q is not None:
        rec = servo_q(q, g, label, speed=speed, settle=settle, tol=tol)
    elif allow_blind:
        # blind fallback: the tracker sees nothing during these ticks
        n0 = now_tick()
        rec = move(xyz, quat, label + "_blind", tol=0.015, max_steps=200)
        nb = max(0, now_tick() - n0)
        TR["blind"] += nb
        TR["n"] += nb
        try:
            TR["q_hold"] = joints_now()
        except Exception:
            pass
    else:
        rec = {"label": label, "ik_fail": True}
        R["motions"].append(rec)
    if rec is not None:
        rec["target"] = vec(xyz)
        e = rec.get("ee")
        if e is not None:
            rec["err"] = fl(math.sqrt(sum((a - b) ** 2 for a, b in zip(e, xyz))))
    return rec


def grip_ticks(g, label, nmin=3, nmax=12):
    rec = {"label": label, "g": g}
    if STATE["ended"]:
        return rec
    q = TR["q_hold"]
    if q is None:
        q = joints_now()
    prev = None
    stable = 0
    k = 0
    while k < nmax and not STATE["ended"]:
        tick(q, g)
        k += 1
        r = grip_reading()
        if prev is not None and r is not None and abs(r - prev) < 0.004:
            stable += 1
        else:
            stable = 0
        prev = r
        if k >= nmin and stable >= 2:
            break
    rec["ticks"] = k
    rec["grip"] = fl(grip_reading())
    R["motions"].append(rec)
    return rec

# ================================================================ PHASE 0: metadata
BASE = "robot_base"
CAMS = []
try:
    ctx = get_task_context()
    R["goal"] = str(ctx.language)
    try:
        R["task_name"] = str(ctx.task_name)
    except Exception:
        pass
except Exception as exc:
    err("get_task_context", exc)
    R["goal"] = None

try:
    md = robomme_plus.get_task_metadata()
    mdd = {}
    for k in md:
        try:
            v = md[k]
            mdd[str(k)] = v if (v is None or v is True or v is False) else str(v)[:300]
        except Exception:
            pass
    R["task_metadata"] = mdd
    for key in ("max_steps", "max_episode_steps"):
        try:
            if key in md and md[key] is not None:
                STATE["max_steps"] = int(md[key])
                break
        except Exception:
            pass
    for key in ("control_hz", "control_frequency_hz", "control_freq", "control_frequency", "hz"):
        try:
            if key in md and md[key] is not None:
                STATE["hz"] = float(md[key])
                break
        except Exception:
            pass
except Exception as exc:
    err("get_task_metadata", exc)

try:
    cm = robomme_plus.get_controller_metadata()
    cmd = {}
    for k in cm:
        try:
            cmd[str(k)] = str(cm[k])[:300]
        except Exception:
            pass
    R["controller_metadata"] = cmd
except Exception as exc:
    err("get_controller_metadata", exc)

try:
    video = robomme_plus.get_conditioning_video()
    nvid = len(video)
    R["conditioning_video"] = {"n": nvid}
    if nvid > 0:
        R["conditioning_video"]["t_first"] = fl(video[0].timestamp_s)
        R["conditioning_video"]["t_last"] = fl(video[nvid - 1].timestamp_s)
except Exception as exc:
    err("get_conditioning_video", exc)
    video = []

state0 = None
HOME_JOINTS = None
Q_DOWN = [0.0, 1.0, 0.0, 0.0]
PHI0 = 0.0
try:
    state0 = get_robot_state()
    BASE = str(state0.base_frame)
    STATE["t0"] = float(state0.timestamp_s)
    HOME_JOINTS = [float(v) for v in state0.joint_positions["primary"]]
    home_pose = state0.end_effector_poses["primary"]
    qh = [float(v) for v in home_pose.quaternion_wxyz]
    m = qmat(qh)
    R["home"] = {"ee": vec(home_pose.position), "quat": vec(qh), "z_axis": vec([m[0][2], m[1][2], m[2][2]]),
                 "joints": vec(HOME_JOINTS), "grip": fl(state0.gripper_positions["primary"]),
                 "embodiment": str(state0.embodiment), "base_frame": BASE}
    if m[2][2] < -0.9:
        Q_DOWN = qh
    else:
        R["notes"].append("home EE not top-down; using canonical [0,1,0,0]")
    md2 = qmat(Q_DOWN)
    PHI0 = math.atan2(md2[1][1], md2[0][1])
    R["home"]["finger_axis_deg"] = fl(math.degrees(PHI0), 2)
except Exception as exc:
    err("robot_state0", exc)

try:
    obs0 = get_observation()
    CAMS = [str(c) for c in obs0.cameras]
    R["cameras"] = CAMS
    cam_info = {}
    for c in CAMS:
        co = obs0.cameras[c]
        info = {}
        try:
            info["shape"] = [int(v) for v in co.rgb.shape]
        except Exception:
            pass
        try:
            info["intrinsics"] = [vec(row) for row in co.intrinsics]
        except Exception:
            pass
        try:
            info["pose"] = {"p": vec(co.camera_pose.position), "q": vec(co.camera_pose.quaternion_wxyz),
                            "frame": str(co.camera_pose.frame)}
        except Exception:
            pass
        cam_info[c] = info
    R["camera_info"] = cam_info
except Exception as exc:
    err("get_observation0", exc)
    obs0 = None

FRONT = "front"
if CAMS and FRONT not in CAMS:
    for c in CAMS:
        if c != "wrist":
            FRONT = c
            break
WRIST = "wrist"

# ================================================================ PHASE 1: tick-0 inventory
W = 256
H = 256
base_img = None
base_white = None
try:
    base_img = as_list(obs0.cameras[FRONT].rgb)
    H = len(base_img)
    W = len(base_img[0])
    base_white = white_flags(base_img)
    R["phases"]["baseline"] = {"w": W, "h": H, "white_px": sum(1 for v in base_white if v)}
except Exception as exc:
    err("baseline image", exc)

inventory = []        # SAM cube candidates on the front camera at tick 0
inv_info = []
try:
    for prompt in ("cube", "small cube", "block"):
        cands, info = seg_candidates(FRONT, prompt)
        inv_info.append(info)
        for c in cands:
            if c["box"] is None:
                continue
            bw = c["box"][2] - c["box"][0]
            bh = c["box"][3] - c["box"][1]
            cc = box_center(c["box"])
            dup = False
            for o in inventory:
                oc = box_center(o["box"])
                if abs(oc[0] - cc[0]) <= 3 and abs(oc[1] - cc[1]) <= 3:
                    dup = True
                    break
            if dup:
                continue
            if bw > 0.25 * W or bh > 0.25 * H:
                continue
            inventory.append(c)
        if len(inventory) >= 8:
            break
except Exception as exc:
    err("inventory", exc)

for c in inventory:
    pts = cand_cloud(c["cand"], FRONT)
    c["stats"] = cloud_stats(pts)
    if c["stats"] is not None and max(c["stats"]["ext"][0], c["stats"]["ext"][1]) > 0.09:
        c["too_big"] = True
        c["stats"] = None
R["phases"]["inventory"] = {"prompts": inv_info, "n": len(inventory),
                            "items": [{"box": vec(c["box"], 1), "score": fl(c["score"], 3), "prompt": c["prompt"],
                                       "stats": (None if c["stats"] is None else
                                                 {"n": c["stats"]["n"], "c": vec([c["stats"]["cx"], c["stats"]["cy"], c["stats"]["ztop"]]),
                                                  "ext": vec(c["stats"]["ext"])})} for c in inventory]}

# button
button = None
btn_info = []
inv_ztops = sorted(c["stats"]["ztop"] for c in inventory if c.get("stats") is not None)
z_ref = inv_ztops[len(inv_ztops) // 2] if inv_ztops else None
try:
    ranked = []
    for prompt in ("button", "white button", "push button"):
        cands, info = seg_candidates(FRONT, prompt, limit=8)
        btn_info.append(info)
        for c in cands:
            frac = box_frac_white(c["box"], base_white, W, H) if base_white is not None else 0.0
            info.setdefault("white_fracs", []).append(round(frac, 3))
            if frac >= 0.08 and c["box"] is not None and c["box"][1] > 2:
                ranked.append((frac + 0.1 * (c["score"] or 0.0), len(ranked), c))
    ranked = sorted(ranked, reverse=True)
    tried = []
    for item in ranked[:6]:
        c = item[2]
        st = cloud_stats(cand_cloud(c["cand"], FRONT))
        ok_ext = st is not None and max(st["ext"][0], st["ext"][1]) < 0.15
        ok_z = st is not None and (z_ref is None or abs(st["ztop"] - z_ref) < 0.06)
        tried.append({"box": vec(c["box"], 1), "key": fl(item[0], 3), "ok_ext": ok_ext, "ok_z": ok_z,
                      "c": None if st is None else vec([st["cx"], st["cy"], st["ztop"]])})
        if ok_ext and ok_z:
            button = {"cx": st["cx"], "cy": st["cy"], "ztop": st["ztop"], "src": "front_sam",
                      "box": c["box"], "ext": st["ext"]}
            break
    btn_info.append({"tried": tried, "z_ref": fl(z_ref)})
except Exception as exc:
    err("button", exc)
if button is None and base_white is not None:
    # fallback: the largest baseline white blob not touching the image border, point-prompted
    try:
        bcomps = components(base_white, W, H, 10)
        bestc = None
        for pix in bcomps:
            s = comp_summary(pix, W)
            b = s["bbox"]
            if b[0] <= 0 or b[1] <= 0 or b[2] >= W - 1 or b[3] >= H - 1:
                continue
            if bestc is None or s["area"] > bestc["area"]:
                bestc = s
        if bestc is not None:
            sp = segment_points(FRONT, [[float(bestc["centroid"][0]), float(bestc["centroid"][1])]])
            if sp.ok and len(sp.segmentations) > 0:
                st = cloud_stats(cand_cloud(sp.segmentations[0], FRONT))
                if st is not None and max(st["ext"][0], st["ext"][1]) < 0.15:
                    button = {"cx": st["cx"], "cy": st["cy"], "ztop": st["ztop"], "src": "white_blob_points",
                              "box": bestc["bbox"], "ext": st["ext"]}
    except Exception as exc:
        err("button fallback", exc)
R["phases"]["button"] ={"prompts": btn_info, "found": None if button is None else
                         {"c": vec([button["cx"], button["cy"], button["ztop"]]), "box": vec(button["box"], 1),
                          "ext": vec(button["ext"])}}


# ================================================================ PHASE 2: watch for highlights (per tick)
try:
    TR["emb"] = state0.embodiment
except Exception:
    TR["emb"] = None
try:
    TR["gray0"] = gray_count(base_img)
except Exception:
    TR["gray0"] = None
TR["prev"] = base_img
watch = {"ticks": [], "counts": [], "on_ticks": [], "disc_sam": None}
hl_frames = []
last_img = None
off_img = None
try:
    gpos0 = grip_reading()
    if gpos0 is None:
        gpos0 = 1.0
    hold_count = 0
    while hold_count < HOLD_MAX_TICKS and not STATE["ended"] and HOME_JOINTS is not None:
        tick(HOME_JOINTS, gpos0)
        hold_count += 1
        if STATE["ended"]:
            break
        img = TR["prev"]
        last_img = img
        nw = new_white(base_img, base_white, img)
        cnt = sum(1 for v in nw if v)
        tk = TR["n"]
        watch["ticks"].append(tk)
        watch["counts"].append(cnt)
        if cnt >= ON_MIN_PIXELS:
            watch["on_ticks"].append(tk)
            if len(hl_frames) < HOLD_ON_FRAMES:
                hl_frames.append(img)
            if watch["disc_sam"] is None:
                dc, dinfo = seg_candidates(FRONT, "white disc", limit=12)
                dinfo["boxes"] = [vec(c["box"], 1) for c in dc]
                watch["disc_sam"] = dinfo
        elif hl_frames:
            # highlight has ended; this frame (discs gone, cubes not yet moved) is the
            # appearance reference for the trackers
            watch["off_tick"] = tk
            off_img = img
            break
    watch["hold_ticks"] = hold_count
except Exception as exc:
    err("watch", exc)
R["phases"]["watch"] = watch


# ================================================================ PHASE 3: highlighted cubes
targets = []
det = {"discs": [], "cubes": [], "table": None}
max_gap = 18
try:
    widths = sorted(c["box"][2] - c["box"][0] for c in inventory)
    if widths:
        max_gap = int(min(30, max(8, 1.6 * widths[len(widths) // 2])))
except Exception as exc:
    err("max_gap", exc)
try:
    if hl_frames and base_img is not None:
        det = detect_highlights(base_img, base_white, hl_frames, W, H, max_gap)
    elif base_img is not None and last_img is not None:
        # no white appeared after tick 0: maybe the highlight was already on at tick 0 and
        # vanished while holding. Swap roles (latest frame = baseline, tick-0 frame = highlight).
        det = detect_highlights(last_img, white_flags(last_img), [base_img], W, H, max_gap)
        R["notes"].append("no highlight onset seen; reversed detection found " + str(len(det["cubes"])) + " cubes")
except Exception as exc:
    err("detect_highlights", exc)

used = set()
for cube in det["cubes"]:
    hx, hy = cube["px"][0], cube["px"][1]
    best = None
    for idx in range(len(inventory)):
        if idx in used:
            continue
        c = inventory[idx]
        b = c["box"]
        cc = box_center(b)
        dist = math.sqrt((cc[0] - hx) ** 2 + (cc[1] - hy) ** 2)
        inside = (b[0] - 2 <= hx <= b[2] + 2) and (b[1] - 2 <= hy <= b[3] + 2)
        if (inside or dist <= 8) and c.get("stats") is not None:
            key = dist - (5 if inside else 0)
            if best is None or key < best[0]:
                best = (key, idx)
    tgt = {"px": cube["px"], "hole_area": cube["area"]}
    if best is not None:
        st = inventory[best[1]]["stats"]
        tgt["src"] = "inventory"
        tgt["inv_idx"] = best[1]
        tgt["c"] = [st["cx"], st["cy"], st["ztop"]]
    else:
        # fallback: point prompt at the enclosed cube pixel (current image, cube unmoved)
        try:
            sp = segment_points(FRONT, [[float(hx), float(hy)]])
            if sp.ok and len(sp.segmentations) > 0:
                pts = cand_cloud(sp.segmentations[0], FRONT)
                st = cloud_stats(pts)
                if st is not None and max(st["ext"][0], st["ext"][1]) < 0.12:
                    tgt["src"] = "segment_points"
                    tgt["c"] = [st["cx"], st["cy"], st["ztop"]]
                    tgt["ext"] = vec(st["ext"])
        except Exception as exc:
            err("segment_points fallback", exc)
    if "c" not in tgt:
        R["notes"].append("highlight at px " + str(cube["px"]) + " has no 3D estimate")
        continue
    dup = False
    for o in targets:
        d3 = math.sqrt((o["c"][0] - tgt["c"][0]) ** 2 + (o["c"][1] - tgt["c"][1]) ** 2)
        d2 = math.sqrt((o["px"][0] - tgt["px"][0]) ** 2 + (o["px"][1] - tgt["px"][1]) ** 2)
        if d3 < 0.02 and d2 < 7:
            dup = True
    if dup:
        R["notes"].append("hole at px " + str(cube["px"]) + " duplicates a target in 3D")
        continue
    if tgt.get("inv_idx") is not None:
        used.add(tgt["inv_idx"])
    targets.append(tgt)

R["phases"]["highlights"] = {"n_frames": len(hl_frames), "max_gap": max_gap, "table": det.get("table"),
                             "discs": det["discs"], "cubes": det["cubes"],
                             "targets": [{"px": t["px"], "src": t["src"], "c": vec(t["c"])} for t in targets]}


# ================================================================ PHASE 3b: start one tracker per highlighted cube
ref_img = off_img if off_img is not None else last_img
try:
    if ref_img is not None:
        for i in range(len(targets)):
            px = targets[i]["px"]
            cx = int(round(px[0]))
            cy = int(round(px[1]))
            TR["tracks"].append({"p": [cx, cy], "v": [0.0, 0.0], "ref": tpatch(ref_img, cx, cy, TRK_HALF),
                                 "tgt": i, "done": False, "attempts": 0, "p0": [cx, cy]})
        TR["prev"] = ref_img
except Exception as exc:
    err("track init", exc)
R["phases"]["tracks_init"] = track_snapshot()

# pixel -> table-plane xy fallback from the tick-0 inventory (affine least squares)
AFF = None


def solve3(m, b):
    det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
           + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    if abs(det) < 1e-9:
        return None
    out = []
    for col in range(3):
        mm = [[(b[r] if c == col else m[r][c]) for c in range(3)] for r in range(3)]
        d = (mm[0][0] * (mm[1][1] * mm[2][2] - mm[1][2] * mm[2][1]) - mm[0][1] * (mm[1][0] * mm[2][2] - mm[1][2] * mm[2][0])
             + mm[0][2] * (mm[1][0] * mm[2][1] - mm[1][1] * mm[2][0]))
        out.append(d / det)
    return out


try:
    pairs = [(box_center(c["box"]), c["stats"]) for c in inventory if c.get("stats") is not None]
    if len(pairs) >= 3:
        m = [[0.0] * 3 for k in range(3)]
        bx = [0.0] * 3
        byy = [0.0] * 3
        for uv, st in pairs:
            row = [1.0, uv[0], uv[1]]
            for r in range(3):
                for c in range(3):
                    m[r][c] += row[r] * row[c]
                bx[r] += row[r] * st["cx"]
                byy[r] += row[r] * st["cy"]
        ax = solve3(m, bx)
        ay = solve3(m, byy)
        if ax is not None and ay is not None:
            AFF = {"x": ax, "y": ay}
except Exception as exc:
    err("affine", exc)
R["phases"]["affine"] = None if AFF is None else {"x": vec(AFF["x"], 5), "y": vec(AFF["y"], 5)}


def est3d(px):
    # current 3D estimate of the cube at a tracked front pixel (static scene; no ticks)
    out = {"px": list(px)}
    try:
        sp = segment_points(FRONT, [[float(px[0]), float(px[1])]])
        if sp.ok and len(sp.segmentations) > 0:
            st = cloud_stats(cand_cloud(sp.segmentations[0], FRONT))
            if st is not None:
                out["sam"] = {"c": vec([st["cx"], st["cy"], st["ztop"]]), "ext": vec(st["ext"])}
                okz = z_ref is None or abs(st["ztop"] - z_ref) < 0.03
                if max(st["ext"][0], st["ext"][1]) < 0.07 and okz:
                    out["c"] = [st["cx"], st["cy"], st["ztop"]]
                    out["src"] = "segment_points"
                    return out
    except Exception as exc:
        err("est3d", exc)
    if AFF is not None:
        u = float(px[0])
        v = float(px[1])
        zt = z_ref if z_ref is not None else 0.04
        out["c"] = [AFF["x"][0] + AFF["x"][1] * u + AFF["x"][2] * v, AFF["y"][0] + AFF["y"][1] * u + AFF["y"][2] * v, zt]
        out["src"] = "affine"
    return out


# ================================================================ PHASE 4: park pose (out of the front view)
PARK = {"q": HOME_JOINTS, "src": "home", "checked": False}
try:
    xs = [c["stats"]["cx"] for c in inventory if c.get("stats") is not None]
    ys = [c["stats"]["cy"] for c in inventory if c.get("stats") is not None]
    if xs:
        pxyz = [sum(xs) / len(xs), sum(ys) / len(ys), PARK_Z]
        ikp = solve_ik(Pose(pxyz, [float(v) for v in Q_DOWN], BASE), seed_joints=HOME_JOINTS)
        qp = ik_joints(ikp)
        if qp is not None:
            PARK["q"] = qp
            PARK["src"] = "ik"
            PARK["xyz"] = vec(pxyz)
except Exception as exc:
    err("park ik", exc)


def park(label):
    rec = servo_q(PARK["q"], TR["g_cmd"], label, speed=SPEED, settle=6)
    if not PARK["checked"] and not STATE["ended"] and TR["prev"] is not None and TR["gray0"] is not None:
        PARK["checked"] = True
        extra = gray_count(TR["prev"]) - TR["gray0"]
        PARK["gray_extra"] = extra
        if extra > 40 and PARK["src"] != "home":
            R["notes"].append("park pose visible in front view (grey px +" + str(extra) + "); using home")
            PARK["q"] = HOME_JOINTS
            PARK["src"] = "home"
            rec = servo_q(PARK["q"], TR["g_cmd"], label + "_home", speed=SPEED, settle=6)
    return rec


# ================================================================ PHASE 5: swap events
SCHED = {"onsets": [], "events": [], "P": None, "D": 0, "none": False, "misses": 0}


def wait_swap(max_wait, label):
    # hold the parked arm, one tick at a time, until a swap event starts and ends
    ev = {"label": label, "start": TR["n"], "max_wait": max_wait}
    TR["free"] = True
    q = TR["q_hold"] if TR["q_hold"] is not None else PARK["q"]
    g = TR["g_cmd"]
    onset = None
    end = None
    on_run = 0
    quiet = 0
    counts = []
    k = 0
    while not STATE["ended"]:
        if not tick(q, g):
            break
        k += 1
        c = TR["mc"]
        counts.append(min(9, c // 4))
        if onset is None:
            on_run = on_run + 1 if c >= MOT_ON else 0
            if on_run >= 2:
                onset = TR["n"] - 1
            elif k >= max_wait:
                break
        else:
            quiet = quiet + 1 if c < MOT_ON else 0
            if quiet >= QUIET_TICKS:
                end = TR["n"] - QUIET_TICKS
                break
            if TR["n"] - onset > 160:
                end = TR["n"]
                ev["forced_end"] = True
                break
    ev["onset"] = onset
    ev["end"] = end
    ev["ticks"] = k
    ev["counts"] = "".join(str(v) for v in counts[-120:])
    if onset is not None and not STATE["ended"] and TR["prev"] is not None:
        ev["recenter"] = recenter_tracks(TR["prev"], 3, True)
        for r in ev["recenter"]:
            if r["ssd"] is None or r["ssd"] > TRK_LOST:
                ev.setdefault("wide", []).append(recenter_tracks(TR["prev"], 12, True, only=[r["i"]]))
        ev["tracks"] = track_snapshot()
        SCHED["misses"] = 0
        if SCHED["onsets"]:
            d = onset - SCHED["onsets"][-1]
            if SCHED["P"] is None:
                SCHED["P"] = float(d)
            else:
                kk = max(1, int(round(d / SCHED["P"])))
                SCHED["P"] = 0.5 * SCHED["P"] + 0.5 * (d / float(kk))
        SCHED["onsets"].append(onset)
        if end is not None:
            SCHED["D"] = max(SCHED["D"], end - onset)
    elif onset is None and SCHED["P"] is not None:
        SCHED["misses"] += 1
    ev["P"] = fl(SCHED["P"], 1)
    SCHED["events"].append(ev)
    return ev


def next_onset(now):
    if SCHED["P"] is None or not SCHED["onsets"]:
        return None
    o = SCHED["onsets"][-1] + SCHED["P"]
    while o < now - 2:
        o += SCHED["P"]
    return int(o)


# ================================================================ PHASE 6: actions
def press_button1(label):
    rec = {"label": label, "tick_start": TR["n"]}
    if button is None:
        rec["skipped"] = "no button estimate"
        R["phases"][label] = rec
        return rec
    bx, by, bz = button["cx"], button["cy"], button["ztop"]
    servo_pose([bx, by, bz + 0.10], Q_DOWN, 0.0, label + "_hover")
    try:
        if not STATE["ended"]:
            cands, info = seg_candidates(WRIST, "button", limit=6)
            bestw = None
            for c in cands:
                st = cloud_stats(cand_cloud(c["cand"], WRIST))
                if st is None:
                    continue
                dxy = math.sqrt((st["cx"] - bx) ** 2 + (st["cy"] - by) ** 2)
                if dxy < 0.05 and max(st["ext"][0], st["ext"][1]) < 0.12 and (bestw is None or dxy < bestw[0]):
                    bestw = (dxy, st)
            if bestw is not None:
                st = bestw[1]
                rec["wrist_refine"] = {"c": vec([st["cx"], st["cy"], st["ztop"]]), "shift": fl(bestw[0])}
                bx, by = st["cx"], st["cy"]
                if abs(st["ztop"] - bz) < 0.03:
                    bz = st["ztop"]
                button["cx"], button["cy"], button["ztop"] = bx, by, bz
    except Exception as exc:
        err(label + " wrist", exc)
    servo_pose([bx, by, bz + 0.04], Q_DOWN, 0.0, label + "_pre", speed=SPEED_SLOW, settle=4)
    # press: command below the cap top; contact stalls the arm, so keep the settle short
    servo_pose([bx, by, bz - 0.012], Q_DOWN, 0.0, label + "_push", speed=SPEED_SLOW, settle=4, tol=0.02)
    rec["ee_at_push"] = ee_pos()
    for k in range(3):
        tick(TR["q_hold"], 0.0)
    servo_pose([bx, by, bz + 0.10], Q_DOWN, 0.0, label + "_rise", settle=3, tol=0.03)
    park(label + "_park")
    rec["tick_end"] = TR["n"]
    rec["target"] = vec([bx, by, bz])
    R["phases"][label] = rec
    return rec


def pick_tracked(ti):
    t = TR["tracks"][ti]
    t["attempts"] += 1
    rec = {"track": ti, "attempt": t["attempts"], "px": list(t["p"]), "p0": t["p0"], "tick_start": TR["n"],
           "unc": t.get("uncertain", False), "ssd": fl(t.get("last"), 1)}
    e3 = est3d(t["p"])
    rec["front"] = {"src": e3.get("src"), "c": vec(e3.get("c")), "sam": e3.get("sam")}
    if e3.get("c") is None:
        rec["skipped"] = "no 3D estimate"
        return rec
    cx, cy, ztop = e3["c"][0], e3["c"][1], e3["c"][2]
    servo_pose([cx, cy, ztop + HOVER_DZ1], Q_DOWN, 1.0, "t" + str(ti) + "_hover")
    fit = None
    neigh = []
    lim = 0.03 if e3.get("src") == "segment_points" else 0.045
    try:
        if not STATE["ended"]:
            cands, info = seg_candidates(WRIST, "cube", limit=15)
            rec["wrist_seg"] = {"n": len(cands), "ok": info.get("ok")}
            bestw = None
            for c in cands:
                pts = cand_cloud(c["cand"], WRIST)
                st = cloud_stats(pts)
                if st is None or max(st["ext"][0], st["ext"][1]) > 0.09:
                    continue
                dxy = math.sqrt((st["cx"] - cx) ** 2 + (st["cy"] - cy) ** 2)
                if dxy < lim and abs(st["ztop"] - ztop) < 0.04:
                    if bestw is None or dxy < bestw[0]:
                        bestw = (dxy, pts, st)
                else:
                    neigh.append([st["cx"], st["cy"]])
            if bestw is not None:
                fit = top_face_fit(bestw[1])
                rec["wrist_match"] = {"dxy": fl(bestw[0]), "n": bestw[2]["n"], "ext": vec(bestw[2]["ext"])}
    except Exception as exc:
        err("t" + str(ti) + " wrist", exc)
    side = 0.04
    delta = 0.0
    if fit is not None and 0.015 < fit["side"] < 0.08:
        side = fit["side"]
        if math.sqrt((fit["cx"] - cx) ** 2 + (fit["cy"] - cy) ** 2) < lim:
            cx, cy = fit["cx"], fit["cy"]
        if abs(fit["ztop"] - ztop) < 0.03:
            ztop = fit["ztop"]
        delta = wrap90(fit["yaw"] - PHI0)
        alt = delta - math.pi / 2.0 if delta > 0 else delta + math.pi / 2.0
        c1 = finger_clearance(cx, cy, PHI0 + delta, side / 2.0, neigh)
        c2 = finger_clearance(cx, cy, PHI0 + alt, side / 2.0, neigh)
        rec["clearance"] = [fl(c1), fl(c2)]
        if c2 > c1 + 0.01:
            delta = alt
        rec["fit"] = {"c": vec([fit["cx"], fit["cy"], fit["ztop"]]), "yaw_deg": fl(math.degrees(fit["yaw"]), 1),
                      "side": fl(fit["side"]), "n_top": fit["n_top"]}
    else:
        rec["fit"] = None
    rec["delta_deg"] = fl(math.degrees(delta), 1)
    q = qyaw(Q_DOWN, delta)
    grasp_z = max(ztop - 0.6 * side, ztop - side + 0.008)
    rec["grasp"] = vec([cx, cy, grasp_z])
    m1 = servo_pose([cx, cy, grasp_z], q, 1.0, "t" + str(ti) + "_down", speed=SPEED_SLOW, settle=6,
                    allow_blind=False)
    if m1 is not None and m1.get("ik_fail"):
        q = Q_DOWN
        rec["yaw_fallback"] = True
        servo_pose([cx, cy, grasp_z], q, 1.0, "t" + str(ti) + "_down_noyaw", speed=SPEED_SLOW, settle=6)
    g1 = grip_ticks(0.0, "t" + str(ti) + "_close", nmin=4, nmax=12)
    rec["grip_closed"] = g1.get("grip")
    if rec["grip_closed"] is not None and rec["grip_closed"] <= GRIP_HELD_MIN:
        rec["held"] = False
        grip_ticks(1.0, "t" + str(ti) + "_reopen", nmin=3, nmax=8)
        servo_pose([cx, cy, ztop + HOVER_DZ1], Q_DOWN, 1.0, "t" + str(ti) + "_back_up", settle=3, tol=0.03)
        park("t" + str(ti) + "_park")
        rec["tick_end"] = TR["n"]
        return rec
    servo_pose([cx, cy, grasp_z + LIFT_DZ1], q, 0.0, "t" + str(ti) + "_lift", settle=4, tol=0.02)
    rec["grip_lifted"] = fl(grip_reading())
    rec["held"] = rec["grip_lifted"] is not None and rec["grip_lifted"] > GRIP_HELD_MIN
    servo_pose([cx, cy, grasp_z + 0.015], q, 0.0, "t" + str(ti) + "_lower", settle=4, tol=0.015)
    grip_ticks(1.0, "t" + str(ti) + "_release", nmin=3, nmax=8)
    servo_pose([cx, cy, grasp_z + 0.07], q, 1.0, "t" + str(ti) + "_rise", settle=2, tol=0.03)
    park("t" + str(ti) + "_park")
    rec["tick_end"] = TR["n"]
    return rec


# ================================================================ PHASE 7: schedule
EST = {"press": 55, "pick": 80}
sched_log = []
picks = []
press1_done = False
press2_done = False
skips = 0
try:
    # the first swap starts right after the highlight ends; watch it from home
    wait_swap(40, "after_highlight")
    guard = 0
    while not STATE["ended"] and not press2_done and guard < 60:
        guard += 1
        now = TR["n"]
        pending = [i for i, t in enumerate(TR["tracks"]) if not t["done"] and t["attempts"] < 2]
        if not press1_done:
            kind = "press"
            act = "press1"
        elif pending:
            kind = "pick"
            act = "pick"
        else:
            kind = "press"
            act = "press2"
        if STATE["max_steps"] is not None and act != "press2":
            if STATE["max_steps"] - now < EST["press"] + 40:
                R["notes"].append("step budget: final press at tick " + str(now) + " with " +
                                  str(len(pending)) + " tracked cubes not picked")
                act = "press2"
                kind = "press"
        if SCHED["misses"] >= 2:
            SCHED["none"] = True
        if (not SCHED["none"]) and SCHED["P"] is None:
            ev = wait_swap(260, "learn_period")
            if ev.get("onset") is None:
                SCHED["none"] = True
                R["notes"].append("no second swap within 260 ticks; acting without a swap schedule")
            continue
        no = None if SCHED["none"] else next_onset(now)
        avail = 100000 if no is None else no - now - 4
        entry = {"act": act, "tick": now, "avail": avail, "est": EST[kind], "next_onset": no}
        if avail >= EST[kind] or skips >= 1 or act == "press2" and avail >= EST[kind] - 10:
            TR["free"] = no is None or avail < EST[kind]
            t0 = TR["n"]
            if act == "press1":
                press_button1("press1")
                press1_done = True
            elif act == "press2":
                press_button1("press2")
                press2_done = True
            else:
                order = sorted([(1 if TR["tracks"][i].get("uncertain") else 0, TR["tracks"][i]["attempts"],
                                 TR["tracks"][i].get("last", 0.0), i) for i in pending])
                ti = order[0][3]
                entry["track"] = ti
                prec = pick_tracked(ti)
                picks.append(prec)
                if prec.get("held"):
                    TR["tracks"][ti]["done"] = True
                entry["held"] = prec.get("held")
            dur = TR["n"] - t0
            entry["dur"] = dur
            EST[kind] = max(int(0.8 * EST[kind]), dur + 3)
            TR["free"] = True
            skips = 0
            if not STATE["ended"] and TR["prev"] is not None:
                entry["recenter"] = recenter_tracks(TR["prev"], 6, False)
        else:
            skips += 1
            entry["wait"] = True
            ev = wait_swap(max(10, no - now + 30), "window_" + str(len(sched_log)))
            entry["swap"] = [ev.get("onset"), ev.get("end")]
        sched_log.append(entry)
except Exception as exc:
    err("schedule", exc)
R["schedule"] = sched_log
R["picks"] = picks
R["swaps"] = SCHED["events"]
R["sched"] = {"P": fl(SCHED["P"], 1), "D": SCHED["D"], "none": SCHED["none"], "onsets": SCHED["onsets"], "est": EST}
R["park"] = {"src": PARK["src"], "xyz": PARK.get("xyz"), "gray_extra": PARK.get("gray_extra")}
R["tracks_final"] = track_snapshot()
R["blind_ticks"] = TR["blind"]

if not press2_done and not STATE["ended"]:
    try:
        press_button1("press2")
    except Exception as exc:
        err("press2", exc)

# a few settle ticks so a delayed success check can register
try:
    if not STATE["ended"]:
        jj = TR["q_hold"] if TR["q_hold"] is not None else joints_now()
        for settle in range(10):
            if not tick(jj, TR["g_cmd"]):
                break
except Exception as exc:
    err("settle", exc)

R["final"] = {"ended": STATE["ended"], "end_reason": STATE["end_reason"], "max_steps": STATE["max_steps"],
              "hz": STATE["hz"], "ticks_counted": TR["n"]}
if not STATE["ended"]:
    R["final"]["tick"] = now_tick()
    R["final"]["ee"] = ee_pos()
result = R

