# PickXtimes v02 — v01 with the placement check fixed: a cube resting on the target occludes up to
# half of it, so the visible-purple centroid is biased by up to ~5 cm (v01 ep1 trace). The target
# stillness test alone decides; the distance test is only a coarse sanity bound (7 cm).
# PickXtimes v01 — v00 + per-tick tracking/servoing onto a MOVING target (fix for v00 ep1).
# Goal (runtime): "pick up the <color> cube and place it on the target, repeating this
# pick-and-place action <N> times, then press the button to stop."
# Colour and N are parsed from get_task_context().language; every position comes from
# live RGB-D perception (SAM3 + depth projection). No recorded coordinates are used.
import math

R = {"version": "v02", "events": [], "errors": [], "moves": [], "perception": []}
STATE = {"ended": False, "placements": 0}

NUMBER_WORDS = {
    "once": 1, "one": 1, "twice": 2, "two": 2, "thrice": 3, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
}
COLOR_WORDS = ("green", "red", "blue", "yellow", "purple", "orange", "white", "black",
               "pink", "cyan", "gray", "grey", "brown", "magenta")


def clean_tokens(text):
    out = []
    for raw in str(text).lower().replace(",", " ").replace(".", " ").split():
        out.append(raw.strip())
    return out


def parse_count(text):
    toks = clean_tokens(text)
    for i, tok in enumerate(toks):
        if tok in ("once", "twice", "thrice"):
            return NUMBER_WORDS[tok]
        if tok in ("times", "time") and i > 0:
            prev = toks[i - 1]
            if prev in NUMBER_WORDS:
                return NUMBER_WORDS[prev]
            try:
                return int(prev)
            except Exception:
                pass
    return None


def parse_color(text):
    toks = clean_tokens(text)
    for i, tok in enumerate(toks):
        if tok in ("cube", "block") and i > 0 and toks[i - 1] in COLOR_WORDS:
            return toks[i - 1]
    for tok in toks:
        if tok in COLOR_WORDS:
            return tok
    return None


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def pix_is(color, r, g, b):
    r = float(r); g = float(g); b = float(b)
    if color == "green":
        return g > 80 and r < 0.45 * g and b < 0.45 * g
    if color == "red":
        return r > 80 and g < 0.3 * r and b < 0.3 * r
    if color == "blue":
        return b > 80 and r < 0.4 * b and g < 0.4 * b
    if color == "yellow":
        return r > 130 and g > 0.8 * r and b < 0.35 * r
    if color in ("purple", "magenta", "pink"):
        return b > 120 and r > 0.7 * b and g < 0.75 * b and g < 0.75 * r
    if color == "white":
        return r > 190 and g > 190 and b > 190
    if color in ("gray", "grey"):
        return abs(r - g) < 20 and abs(g - b) < 20 and 90 < r < 200
    if color == "orange":
        return r > 150 and 0.35 * r < g < 0.75 * r and b < 0.3 * r
    return False


def box_color_fraction(rgb, box, colors):
    try:
        h = int(rgb.shape[0]); w = int(rgb.shape[1])
        x1 = max(0, int(math.floor(float(box[0])))); y1 = max(0, int(math.floor(float(box[1]))))
        x2 = min(w - 1, int(math.ceil(float(box[2])))); y2 = min(h - 1, int(math.ceil(float(box[3]))))
        total = 0; hit = 0
        sy = max(1, (y2 - y1) // 24); sx = max(1, (x2 - x1) // 24)
        for y in range(y1, y2 + 1, sy):
            row = rgb[y]
            for x in range(x1, x2 + 1, sx):
                px = row[x]
                total += 1
                for c in colors:
                    if pix_is(c, px[0], px[1], px[2]):
                        hit += 1
                        break
        if total == 0:
            return 0.0
        return float(hit) / float(total)
    except Exception as exc:
        R["errors"].append("box_color_fraction: " + str(exc))
        return 0.0


def cloud_points(cand, cam, base_frame):
    attempts = []
    try:
        attempts.append(cand.mask)
    except Exception:
        pass
    attempts.append(cand)
    for m in attempts:
        try:
            pc = mask_to_point_cloud(m, cam, target_frame=base_frame)
            pts = []
            for p in pc.points:
                pts.append((float(p[0]), float(p[1]), float(p[2])))
            if len(pts) > 0:
                return pts
        except Exception as exc:
            R["errors"].append("mask_to_point_cloud(" + cam + "): " + str(exc)[:200])
    return []


def segment_candidates(cam, prompts, colors, min_frac, label):
    """All candidates over the prompts, with score, box and colour fraction of the box."""
    obs = get_observation()
    rgb = obs.cameras[cam].rgb
    out = []
    for prompt in prompts:
        try:
            seg = segment_text(cam, prompt)
        except Exception as exc:
            R["errors"].append("segment_text " + prompt + ": " + str(exc)[:200])
            continue
        rec = {"label": label, "cam": cam, "prompt": prompt, "ok": False, "cands": []}
        try:
            rec["ok"] = bool(seg.ok)
        except Exception:
            pass
        if not rec["ok"]:
            R["perception"].append(rec)
            continue
        for idx, cand in enumerate(seg.segmentations[:8]):
            try:
                box = [float(v) for v in cand.box_xyxy]
            except Exception:
                box = None
            score = 0.0
            try:
                score = float(cand.score)
            except Exception:
                pass
            frac = box_color_fraction(rgb, box, colors) if box is not None else 0.0
            rec["cands"].append({"i": idx, "score": round(score, 3),
                                 "box": [round(v, 1) for v in box] if box is not None else None,
                                 "frac": round(frac, 3)})
            out.append({"cand": cand, "score": score, "box": box, "frac": frac,
                        "prompt": prompt, "cam": cam})
        R["perception"].append(rec)
        good = [c for c in out if c["frac"] >= min_frac]
        if len(good) > 0:
            break
    return out


def rank_cands(cands, min_frac):
    keyed = []
    for i, c in enumerate(cands):
        keyed.append((0 if c["frac"] >= min_frac else 1, -c["score"], i))
    keyed = sorted(keyed)
    return [cands[k[2]] for k in keyed]


def color_blob_point(cam, colors):
    """Fallback: centroid of the largest blob of matching pixels (coarse 8 px grid)."""
    obs = get_observation()
    rgb = obs.cameras[cam].rgb
    h = int(rgb.shape[0]); w = int(rgb.shape[1])
    cells = {}
    for y in range(0, h, 2):
        row = rgb[y]
        for x in range(0, w, 2):
            px = row[x]
            ok = False
            for c in colors:
                if pix_is(c, px[0], px[1], px[2]):
                    ok = True
                    break
            if ok:
                key = (y // 8, x // 8)
                if key not in cells:
                    cells[key] = [0, 0.0, 0.0]
                cells[key][0] += 1
                cells[key][1] += x
                cells[key][2] += y
    if len(cells) == 0:
        return None
    # connected components over grid cells
    seen = set()
    best = None
    for key in cells:
        if key in seen:
            continue
        stack = [key]; seen.add(key); comp = []
        while len(stack) > 0:
            k = stack.pop()
            comp.append(k)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nb = (k[0] + dy, k[1] + dx)
                    if nb in cells and nb not in seen:
                        seen.add(nb); stack.append(nb)
        n = 0; sx = 0.0; sy = 0.0
        for k in comp:
            n += cells[k][0]; sx += cells[k][1]; sy += cells[k][2]
        if best is None or n > best[0]:
            best = (n, sx / n, sy / n)
    return best


def min_area_yaw(pts):
    """Face direction (radians, in [0, pi/2)) of a square footprint: the rotation minimising the
    4th moment sum(u^4 + v^4) (projections onto face axes are uniform, onto diagonals peaked)."""
    if len(pts) < 5:
        return None
    stride = max(1, len(pts) // 400)
    sub = pts[::stride]
    mx = sum(p[0] for p in sub) / len(sub)
    my = sum(p[1] for p in sub) / len(sub)
    best = None
    for k in range(0, 90, 2):
        a = math.radians(k)
        ca = math.cos(a); sa = math.sin(a)
        acc = 0.0
        for p in sub:
            dx = p[0] - mx; dy = p[1] - my
            u = dx * ca + dy * sa
            v = -dx * sa + dy * ca
            acc += u * u * u * u + v * v * v * v
        if best is None or acc < best[0]:
            best = (acc, a)
    a = best[1]
    ca = math.cos(a); sa = math.sin(a)
    us = [(p[0] - mx) * ca + (p[1] - my) * sa for p in sub]
    vs = [-(p[0] - mx) * sa + (p[1] - my) * ca for p in sub]
    su = quantile(us, 0.98) - quantile(us, 0.02)
    sv = quantile(vs, 0.98) - quantile(vs, 0.02)
    return (best[0], a, su, sv)


def cube_from_points(pts, table_z):
    zs = [p[2] for p in pts]
    top = quantile(zs, 0.9)
    lo = table_z + 0.004 if table_z is not None else None
    if lo is not None:
        pts = [p for p in pts if p[2] > lo and p[2] < top + 0.02]
    top_pts = [p for p in pts if p[2] > top - 0.008]
    if len(top_pts) < 8:
        top_pts = [p for p in pts if p[2] > top - 0.015]
    if len(top_pts) < 3:
        top_pts = pts
    if len(top_pts) == 0:
        return None
    cx = quantile([p[0] for p in top_pts], 0.5)
    cy = quantile([p[1] for p in top_pts], 0.5)
    yaw = min_area_yaw(top_pts)
    return {"x": cx, "y": cy, "top": top, "n": len(pts), "n_top": len(top_pts),
            "yaw": yaw[1] if yaw is not None else None,
            "span": [round(yaw[2], 4), round(yaw[3], 4)] if yaw is not None else None}


def localize_cube(cam, color, base_frame, table_z, expect_xy=None, max_dev=None):
    prompts = [color + " cube", color + " block", "small " + color + " cube", "cube"]
    cands = segment_candidates(cam, prompts, [color], 0.2, "cube")
    ranked = rank_cands(cands, 0.2)
    for c in ranked:
        if c["frac"] < 0.2:
            break
        pts = cloud_points(c["cand"], cam, base_frame)
        if len(pts) < 5:
            continue
        cube = cube_from_points(pts, table_z)
        if cube is None:
            continue
        if expect_xy is not None and max_dev is not None:
            d = math.hypot(cube["x"] - expect_xy[0], cube["y"] - expect_xy[1])
            if d > max_dev:
                R["events"].append({"cube_rejected_far": round(d, 3), "cam": cam})
                continue
        cube["src"] = cam + ":" + c["prompt"]
        cube["frac"] = round(c["frac"], 3)
        cube["score"] = round(c["score"], 3)
        return cube
    # fallback: colour blob -> point prompt
    blob = color_blob_point(cam, [color])
    if blob is not None:
        try:
            seg = segment_points(cam, [[blob[1], blob[2]]])
            if seg.ok and len(seg.segmentations) > 0:
                pts = cloud_points(seg.segmentations[0], cam, base_frame)
                if len(pts) >= 5:
                    cube = cube_from_points(pts, table_z)
                    if cube is not None:
                        cube["src"] = cam + ":blob_point"
                        R["events"].append({"cube_blob": [round(blob[1], 1), round(blob[2], 1), blob[0]]})
                        return cube
        except Exception as exc:
            R["errors"].append("segment_points cube: " + str(exc)[:200])
    return None


def localize_flat(cam, prompts, colors, min_frac, base_frame, label, max_height=None, ref_z=None):
    cands = segment_candidates(cam, prompts, colors, min_frac, label)
    ranked = rank_cands(cands, min_frac)
    for c in ranked:
        if c["frac"] < min_frac:
            break
        pts = cloud_points(c["cand"], cam, base_frame)
        if ref_z is not None and max_height is not None:
            pts = [p for p in pts if p[2] < ref_z + max_height]
        if len(pts) < 5:
            continue
        span_x = quantile([p[0] for p in pts], 0.95) - quantile([p[0] for p in pts], 0.05)
        span_y = quantile([p[1] for p in pts], 0.95) - quantile([p[1] for p in pts], 0.05)
        if max(span_x, span_y) > 0.12:
            R["events"].append({"flat_rejected_span": [label, c["prompt"], round(span_x, 3), round(span_y, 3)]})
            continue
        zs = [p[2] for p in pts]
        return {"span": [round(span_x, 4), round(span_y, 4)], "x": quantile([p[0] for p in pts], 0.5), "y": quantile([p[1] for p in pts], 0.5),
                "z": quantile(zs, 0.5), "top": quantile(zs, 0.95), "zlo": quantile(zs, 0.05),
                "n": len(pts), "src": cam + ":" + c["prompt"], "frac": round(c["frac"], 3),
                "score": round(c["score"], 3)}
    return None


# ---------------- quaternion helpers (wxyz) ----------------
def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def qnorm(q):
    n = math.sqrt(sum(v * v for v in q))
    return tuple(v / n for v in q)


def axis_y(q):
    w, x, y, z = q
    return (2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x))


def axis_z(q):
    w, x, y, z = q
    return (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))


def qz(angle):
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def grasp_quat(q_ref, face_yaw):
    """Rotate q_ref about world z so the finger (local y) axis is along a cube face normal."""
    if face_yaw is None:
        return q_ref
    ay = axis_y(q_ref)
    finger = math.atan2(ay[1], ay[0])
    d = face_yaw - finger
    quarter = math.pi / 2.0
    d = (d + quarter / 2.0) % quarter - quarter / 2.0
    return qnorm(qmul(qz(d), q_ref))


# ---------------- motion helpers ----------------
def now_ts():
    try:
        return round(float(get_robot_state().timestamp_s), 3)
    except Exception:
        return None


def check_end(res, label):
    term = False
    for name in ("terminated", "truncated"):
        try:
            if name == "terminated" and bool(res.terminated):
                term = True
            if name == "truncated" and bool(res.truncated):
                term = True
        except Exception:
            pass
    if term:
        STATE["ended"] = True
        R["events"].append({"episode_ended_after": label})
        raise Exception("EPISODE_ENDED")


def summarize(res, label, extra=None):
    rec = {"label": label}
    try:
        rec["ok"] = bool(res.ok)
    except Exception:
        pass
    try:
        rec["status"] = str(res.status)
    except Exception:
        pass
    try:
        rec["steps"] = int(res.steps_executed)
    except Exception:
        pass
    try:
        rec["term"] = [bool(res.terminated), bool(res.truncated)]
    except Exception:
        pass
    try:
        if not res.ok:
            rec["error"] = str(res.error)[:240]
    except Exception:
        pass
    if extra is not None:
        rec.update(extra)
    return rec


def move(pos, quat, label, tol=0.01, max_steps=200):
    pose = Pose([float(pos[0]), float(pos[1]), float(pos[2])], [float(v) for v in quat], BASE)
    res = move_to_pose(pose, tolerance=tol, max_steps=max_steps)
    rec = summarize(res, label, {"target": [round(float(v), 4) for v in pos]})
    try:
        st = get_robot_state()
        ee = st.end_effector_poses["primary"].position
        rec["ee"] = [round(float(v), 4) for v in ee]
        rec["grip"] = round(float(st.gripper_positions["primary"]), 3)
        rec["t"] = round(float(st.timestamp_s), 3)
    except Exception:
        pass
    R["moves"].append(rec)
    check_end(res, label)
    return res


def gripper(action, label):
    if action == "open":
        res = open_gripper()
    else:
        res = close_gripper()
    rec = summarize(res, label)
    try:
        st = get_robot_state()
        rec["grip"] = round(float(st.gripper_positions["primary"]), 3)
        rec["t"] = round(float(st.timestamp_s), 3)
    except Exception:
        pass
    R["moves"].append(rec)
    check_end(res, label)
    return res


def grip_width():
    return float(get_robot_state().gripper_positions["primary"])


# ---------------- v01: moving-target tracking (front camera, purple pixels) ----------------
# Failure mode fixed (v00 on episode 1): the target disc MOVES during the episode, and v00 placed
# the cube at the target's stale initial position. v01 re-measures the target every control tick
# from the live front RGB (purple pixel centroid back-projected onto the target's plane), fits a
# constant-velocity model to recent unoccluded measurements, and servos the carried cube onto
# the predicted target with per-tick IK + step(). Nothing here is read from the recordings.
TR = {"cam": None, "hist": [], "uv": None, "K": None, "off": [0.0, 0.0], "zp": None,
      "vel": [0.0, 0.0], "maxspeed": 0.0, "moving": False, "hz": 20.0, "lastq": None,
      "x0": None, "steps": 0}
LEAD = 2            # ticks of command lead (controller lag ~1 tick + this tick)
VMAX = 0.018        # max commanded xy change per tick (m)
ZMAX = 0.015        # max commanded z change per tick (m)
GOOD_RATIO = 0.70   # min purple count / expected count for an unoccluded measurement


def quat_rows(q):
    w, x, y, z = [float(v) for v in q]
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def set_front_model(cobs):
    kin = cobs.intrinsics
    TR["cam"] = {"t": [float(v) for v in cobs.camera_pose.position],
                 "R": quat_rows(cobs.camera_pose.quaternion_wxyz),
                 "fx": float(kin[0][0]), "fy": float(kin[1][1]),
                 "cx": float(kin[0][2]), "cy": float(kin[1][2])}


def backproject(u, v, zp):
    c = TR["cam"]
    dc = [(u - c["cx"]) / c["fx"], (v - c["cy"]) / c["fy"], 1.0]
    rm = c["R"]
    dw = [rm[i][0] * dc[0] + rm[i][1] * dc[1] + rm[i][2] * dc[2] for i in range(3)]
    if abs(dw[2]) < 1e-6:
        return None
    s = (zp - c["t"][2]) / dw[2]
    if s <= 0:
        return None
    return [c["t"][0] + s * dw[0], c["t"][1] + s * dw[1], zp]


def project(p):
    c = TR["cam"]
    rm = c["R"]
    d = [p[i] - c["t"][i] for i in range(3)]
    pc = [rm[0][i] * d[0] + rm[1][i] * d[1] + rm[2][i] * d[2] for i in range(3)]
    if pc[2] <= 1e-6:
        return None
    return [c["fx"] * pc[0] / pc[2] + c["cx"], c["fy"] * pc[1] / pc[2] + c["cy"]]


def purple_scan(rgb, uc, vc, half, stride):
    h = int(rgb.shape[0]); w = int(rgb.shape[1])
    if uc is None:
        u0 = 0; u1 = w - 1; v0 = 0; v1 = h - 1
    else:
        u0 = max(0, int(uc) - half); u1 = min(w - 1, int(uc) + half)
        v0 = max(0, int(vc) - half); v1 = min(h - 1, int(vc) + half)
    n = 0; su = 0.0; sv = 0.0
    for v in range(v0, v1 + 1, stride):
        try:
            row = rgb[v].tolist()
        except Exception:
            row = rgb[v]
        for u in range(u0, u1 + 1, stride):
            px = row[u]
            r = px[0]; g = px[1]; b = px[2]
            if b > 120 and r > 0.7 * b and g < 0.75 * b and g < 0.75 * r:
                n += 1; su += u; sv += v
    if n == 0:
        return None
    return [su / n, sv / n, n * stride * stride]


def expected_count(p):
    c = TR["cam"]
    d = math.sqrt(sum((p[i] - c["t"][i]) ** 2 for i in range(3)))
    return (c["t"][2] - p[2]) / (d * d * d)


def obs_tick(obs):
    try:
        return int(round(float(obs.timestamp_s) * TR["hz"]))
    except Exception:
        return int(round(float(obs.robot_state.timestamp_s) * TR["hz"]))


def fit_track(now_tick):
    """Constant-velocity fit over the last unoccluded measurements. Returns anchor + velocity."""
    good = [h for h in TR["hist"] if h[3] and now_tick - h[0] <= 12]
    good = good[-7:]
    if len(good) >= 3 and good[-1][0] - good[0][0] >= 3:
        tm = sum(h[0] for h in good) / len(good)
        xm = sum(h[1] for h in good) / len(good)
        ym = sum(h[2] for h in good) / len(good)
        stt = sum((h[0] - tm) ** 2 for h in good)
        vx = sum((h[0] - tm) * (h[1] - xm) for h in good) / stt
        vy = sum((h[0] - tm) * (h[2] - ym) for h in good) / stt
        return {"t": tm, "x": xm, "y": ym, "vx": vx, "vy": vy, "n": len(good), "fresh": True}
    allgood = [h for h in TR["hist"] if h[3]]
    if len(allgood) == 0:
        return None
    h = allgood[-1]
    return {"t": h[0], "x": h[1], "y": h[2], "vx": TR["vel"][0], "vy": TR["vel"][1],
            "n": len(good), "fresh": False}


def predict(fit, tick):
    if fit is None:
        return None
    dt = min(15.0, max(-15.0, tick - fit["t"]))
    return [fit["x"] + fit["vx"] * dt, fit["y"] + fit["vy"] * dt]


def target_measure(obs, use_for_fit=True):
    """Purple-centroid target measurement on the live front image (one per control tick)."""
    rgb = obs.cameras["front"].rgb
    tick = obs_tick(obs)
    sc = None
    if TR["uv"] is not None:
        sc = purple_scan(rgb, TR["uv"][0], TR["uv"][1], 32, 1)
    if sc is None or sc[2] < 12:
        full = purple_scan(rgb, None, None, 0, 2)
        if full is not None and full[2] >= 12:
            sc = purple_scan(rgb, full[0], full[1], 32, 1)
    m = {"tick": tick, "good": False, "n": 0}
    if sc is None or sc[2] < 12:
        return m
    p = backproject(sc[0], sc[1], TR["zp"])
    if p is None:
        return m
    TR["uv"] = [sc[0], sc[1]]
    x = p[0] + TR["off"][0]; y = p[1] + TR["off"][1]
    m["x"] = x; m["y"] = y; m["n"] = sc[2]; m["u"] = round(sc[0], 1); m["v"] = round(sc[1], 1)
    ratio = None
    if TR["K"] is not None:
        ratio = sc[2] / (TR["K"] * expected_count(p))
        m["ratio"] = round(ratio, 2)
    good = ratio is not None and ratio >= GOOD_RATIO
    if good:
        fit = fit_track(tick)
        if fit is not None and fit["fresh"]:
            pr = predict(fit, tick)
            if math.hypot(pr[0] - x, pr[1] - y) > 0.025:
                good = False
                m["jump"] = True
    m["good"] = good and use_for_fit
    if len(TR["hist"]) > 0 and TR["hist"][-1][0] == tick:
        TR["hist"][-1] = [tick, x, y, m["good"], sc[2]]
    else:
        TR["hist"].append([tick, x, y, m["good"], sc[2]])
    if len(TR["hist"]) > 80:
        TR["hist"] = TR["hist"][-80:]
    if m["good"]:
        fit = fit_track(tick)
        if fit is not None and fit["fresh"] and fit["n"] >= 4 and fit["n"] > 0:
            TR["vel"] = [fit["vx"], fit["vy"]]
            sp = math.hypot(fit["vx"], fit["vy"])
            if sp > TR["maxspeed"]:
                TR["maxspeed"] = sp
            if sp > 0.0012:
                TR["moving"] = True
        if TR["x0"] is not None and math.hypot(x - TR["x0"][0], y - TR["x0"][1]) > 0.015:
            TR["moving"] = True
    return m


def calibrate_tracker(target):
    """Front camera model + expected-count scale + offset to the SAM/depth target centre (same
    observation instant: perception does not advance simulation time)."""
    obs = get_observation()
    set_front_model(obs.cameras["front"])
    try:
        TR["hz"] = float(robomme_plus.get_task_metadata()["control_frequency_hz"])
    except Exception:
        TR["hz"] = 20.0
    TR["zp"] = float(target["top"])
    sc = purple_scan(obs.cameras["front"].rgb, None, None, 0, 1)
    rec = {"zp": round(TR["zp"], 4), "hz": TR["hz"]}
    if sc is None or sc[2] < 12:
        rec["error"] = "no purple pixels"
        R["tracker_calib"] = rec
        return False
    p = backproject(sc[0], sc[1], TR["zp"])
    TR["uv"] = [sc[0], sc[1]]
    TR["K"] = sc[2] / expected_count(p)
    off = [target["x"] - p[0], target["y"] - p[1]]
    rec["pix"] = [round(sc[0], 2), round(sc[1], 2), sc[2]]
    rec["pix_xy"] = [round(p[0], 4), round(p[1], 4)]
    rec["sam_xy"] = [round(target["x"], 4), round(target["y"], 4)]
    rec["offset"] = [round(off[0], 4), round(off[1], 4)]
    if math.hypot(off[0], off[1]) < 0.025:
        TR["off"] = off
        rec["offset_applied"] = True
    else:
        rec["offset_applied"] = False
    rec["K"] = round(TR["K"], 3)
    TR["x0"] = [target["x"], target["y"]]
    m = target_measure(obs)
    rec["first"] = {"x": round(m.get("x", 0.0), 4), "y": round(m.get("y", 0.0), 4),
                    "ratio": m.get("ratio"), "good": m["good"]}
    R["tracker_calib"] = rec
    return True


def ik_joints(ik):
    jp = None
    for name in ("joint_positions", "solution", "joints"):
        try:
            if name == "joint_positions":
                jp = ik.joint_positions
            elif name == "solution":
                jp = ik.solution
            else:
                jp = ik.joints
        except Exception:
            jp = None
        if jp is not None:
            break
    if jp is None:
        return None
    try:
        jp = jp["primary"]
    except Exception:
        pass
    try:
        return [float(v) for v in jp]
    except Exception:
        return None


def step_to(pos, quat, grip_cmd, label, st):
    """One control tick toward an EE pose via IK (seeded at the measured joints)."""
    q0 = [float(v) for v in st.joint_positions["primary"]]
    qcmd = None
    note = None
    try:
        ik = solve_ik(Pose([float(pos[0]), float(pos[1]), float(pos[2])], [float(v) for v in quat], BASE),
                      seed_joints=q0)
        if bool(ik.ok):
            qcmd = ik_joints(ik)
            if qcmd is not None and len(qcmd) == len(q0):
                jump = max([abs(a - b) for a, b in zip(qcmd, q0)])
                if jump > 0.5:
                    note = "ik_jump_" + str(round(jump, 2))
                    qcmd = None
            else:
                note = "ik_bad_shape"
                qcmd = None
        else:
            note = "ik_fail"
    except Exception as exc:
        note = "ik_exc:" + str(exc)[:120]
    if qcmd is None:
        qcmd = TR["lastq"] if TR["lastq"] is not None else q0
    else:
        TR["lastq"] = qcmd
    res = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=list(qcmd), gripper_position=float(grip_cmd),
        embodiment=st.embodiment)}))
    TR["steps"] += 1
    try:
        if not res.ok:
            R["errors"].append("step " + label + ": " + str(res.error)[:160])
    except Exception:
        pass
    check_end(res, label)
    return note


def servo_place(qg, label, wait_resume, track_z, place_z, max_ticks):
    """Carry the held cube onto the (possibly moving) target and release it.
    Phases: wait (target frozen after the previous pick; wait for it to resume) -> track at
    track_z -> descend -> release (gripper opens while xy is held)."""
    info = {"label": label, "wait_resume": wait_resume, "aborts": 0, "forced": False,
            "ik_notes": 0, "trace": []}
    phase = "wait" if wait_resume else "track"
    kp = 0
    frozen = None
    resume_tick = None
    cmd = None
    zc = None
    rel_xy = None
    ik_bad = 0
    grip_at_release = None
    tz = track_z
    zdesc = None
    for it in range(max_ticks):
        obs = get_observation()
        st = obs.robot_state
        m = target_measure(obs, use_for_fit=(phase in ("wait", "track")))
        tick = m["tick"]
        ee = [float(v) for v in st.end_effector_poses["primary"].position]
        grip = float(st.gripper_positions["primary"])
        if cmd is None:
            cmd = [ee[0], ee[1]]
            zc = ee[2]
            info["tick0"] = tick
        fit = fit_track(tick)
        now = predict(fit, tick)
        if now is None:
            now = [cmd[0], cmd[1]]
        ahead = predict(fit, tick + LEAD)
        if ahead is None:
            ahead = now
        err = math.hypot(ee[0] - now[0], ee[1] - now[1])
        ngood = len([h for h in TR["hist"] if h[3] and tick - h[0] <= 8])
        # ---- phase transitions ----
        if phase == "wait":
            if m["good"]:
                if frozen is None:
                    frozen = [m["x"], m["y"]]
                elif math.hypot(m["x"] - frozen[0], m["y"] - frozen[1]) > 0.006:
                    phase = "track"; kp = 0; resume_tick = tick
                    info["resumed_after"] = it
            if phase == "wait" and it >= 70:
                phase = "track"; kp = 0; info["wait_timeout"] = True
        elif phase == "track":
            if kp >= 10 and kp % 10 == 0 and ngood == 0 and tz < track_z + 0.06:
                tz = tz + 0.03  # target hidden at this height: track higher
                info["raised_track_z"] = round(tz, 4)
            ready = ngood >= 4 and err < 0.012 and abs(ee[2] - tz) < 0.012 and kp >= 4
            if resume_tick is not None and tick - resume_tick < 6:
                ready = False
            if ready or kp >= 120:
                if not ready:
                    info["forced"] = True
                info["descend_err"] = round(err, 4)
                info["descend_vel"] = [round(fit["vx"], 5), round(fit["vy"], 5)] if fit is not None else None
                phase = "descend"; kp = 0; zdesc = zc
        elif phase == "descend":
            if err > 0.03 and info["aborts"] < 2:
                info["aborts"] += 1
                phase = "track"; kp = 0
            elif ee[2] <= place_z + 0.006 or kp >= 4 + int((zdesc - place_z) / 0.012):
                phase = "release"; kp = 0
                rel_xy = [cmd[0], cmd[1]]
                grip_at_release = grip
                info["release_err"] = round(err, 4)
                info["release_tick"] = tick
                info["release_ee"] = [round(v, 4) for v in ee]
        elif phase == "release":
            if (kp >= 3 and grip >= 0.8) or kp >= 8:
                info["grip_after"] = round(grip, 3)
                break
        # ---- desired pose ----
        if phase == "release":
            des = rel_xy
            zt = place_z
            gcmd = 1.0
        else:
            cx = max(-0.01, min(0.01, 0.3 * (now[0] - ee[0])))
            cy = max(-0.01, min(0.01, 0.3 * (now[1] - ee[1])))
            des = [ahead[0] + cx, ahead[1] + cy]
            if phase == "descend":
                zt = max(place_z, zdesc - (kp + 1) * 0.012)
            else:
                zt = tz
            gcmd = 0.0
        dx = des[0] - cmd[0]; dy = des[1] - cmd[1]
        d = math.hypot(dx, dy)
        if d > VMAX:
            dx = dx * VMAX / d; dy = dy * VMAX / d
        cmd = [cmd[0] + dx, cmd[1] + dy]
        zc = zc + max(-ZMAX, min(ZMAX, zt - zc))
        note = step_to((cmd[0], cmd[1], zc), qg, gcmd, "servo_" + label + "_" + phase, st)
        if note is not None:
            info["ik_notes"] += 1
            ik_bad += 1
            if info.get("first_ik_note") is None:
                info["first_ik_note"] = note
        else:
            ik_bad = 0
        if len(info["trace"]) < 260:
            info["trace"].append([tick, phase[0], round(m.get("x", 0.0), 4), round(m.get("y", 0.0), 4),
                                  m.get("ratio"), 1 if m["good"] else 0,
                                  round(now[0], 4), round(now[1], 4),
                                  round(ee[0], 4), round(ee[1], 4), round(ee[2], 4), round(err, 4)])
        kp += 1
        if ik_bad >= 6:
            info["ik_abort"] = True
            break
    info["end_phase"] = phase
    info["rel_xy"] = rel_xy
    info["grip_at_release"] = grip_at_release
    info["maxspeed"] = round(TR["maxspeed"], 5)
    info["moving"] = TR["moving"]
    TR["vel"] = [0.0, 0.0]
    return info


def verify_on_target(rel_xy, label):
    """Moving-target episodes only: the target freezes while the cube rests on it. Hold 5 ticks
    at the retract pose and measure the raw purple centroid; it must be still and near the cube."""
    pts = []
    for k in range(5):
        obs = get_observation()
        st = obs.robot_state
        m = target_measure(obs, use_for_fit=False)
        if m.get("x") is not None:
            pts.append([m["tick"], m["x"], m["y"], m.get("ratio")])
        q0 = [float(v) for v in st.joint_positions["primary"]]
        res = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=q0, gripper_position=1.0, embodiment=st.embodiment)}))
        TR["steps"] += 1
        check_end(res, "verify_" + label)
    out = {"pts": [[p[0], round(p[1], 4), round(p[2], 4), p[3]] for p in pts]}
    if len(pts) < 2:
        out["ok"] = None
        return out
    disp = math.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2])
    near = math.hypot(pts[-1][1] - rel_xy[0], pts[-1][2] - rel_xy[1])
    out["disp"] = round(disp, 4)
    out["near"] = round(near, 4)
    out["ok"] = disp < 0.004 and near < 0.07
    return out


# ======================= main =======================
ctx = get_task_context()
GOAL = ctx.language
R["goal"] = GOAL
N = parse_count(GOAL)
COLOR = parse_color(GOAL)
R["parsed"] = {"count": N, "color": COLOR}
if COLOR is None:
    COLOR = "green"
if N is None:
    N = 1
    R["errors"].append("count not parsed; defaulting to 1")

try:
    meta = robomme_plus.get_task_metadata()
    R["meta"] = {}
    for key in meta:
        R["meta"][str(key)] = str(meta[key])[:200]
except Exception as exc:
    R["errors"].append("task_metadata: " + str(exc)[:200])
try:
    cmeta = robomme_plus.get_controller_metadata()
    R["controller"] = {}
    for key in cmeta:
        R["controller"][str(key)] = str(cmeta[key])[:300]
except Exception as exc:
    R["errors"].append("controller_metadata: " + str(exc)[:200])
try:
    R["conditioning_video_len"] = len(robomme_plus.get_conditioning_video())
except Exception as exc:
    R["errors"].append("conditioning_video: " + str(exc)[:200])

state0 = get_robot_state()
BASE = state0.base_frame
home_pose = state0.end_effector_poses["primary"]
q_home = tuple(float(v) for v in home_pose.quaternion_wxyz)
R["home"] = {"pos": [round(float(v), 4) for v in home_pose.position],
             "quat": [round(v, 4) for v in q_home], "frame": str(home_pose.frame),
             "grip": round(float(state0.gripper_positions["primary"]), 3),
             "t": round(float(state0.timestamp_s), 3), "base_frame": str(BASE)}
az = axis_z(q_home)
R["home"]["axis_z"] = [round(v, 3) for v in az]
if az[2] < -0.9:
    Q_DOWN = q_home
else:
    Q_DOWN = (0.0, 1.0, 0.0, 0.0)
    R["events"].append("home not top-down; using canonical down quaternion")
try:
    obs0 = get_observation()
    camrec = {}
    for name in obs0.cameras:
        cobs = obs0.cameras[name]
        camrec[str(name)] = {"shape": [int(v) for v in cobs.rgb.shape],
                             "pose": [round(float(v), 4) for v in cobs.camera_pose.position],
                             "quat": [round(float(v), 4) for v in cobs.camera_pose.quaternion_wxyz]}
    R["cameras"] = camrec
except Exception as exc:
    R["errors"].append("camera info: " + str(exc)[:200])

try:
    # ---------- scene perception from the front camera at home ----------
    target = localize_flat("front", ["purple and white target", "bullseye target", "target",
                                     "purple circle"], ["purple"], 0.1, BASE, "target")
    if target is None:
        target = localize_flat("wrist", ["purple and white target", "bullseye target", "target"],
                               ["purple"], 0.1, BASE, "target")
    R["target"] = target
    if target is None:
        raise ValueError("target not found")
    TABLE_Z = target["zlo"]
    R["table_z"] = TABLE_Z
    try:
        TRACK_OK = calibrate_tracker(target)
    except Exception as exc:
        R["errors"].append("calibrate_tracker: " + str(exc)[:200])
        TRACK_OK = False
    try:
        MAX_STEPS = int(float(robomme_plus.get_task_metadata()["max_steps"]))
    except Exception:
        MAX_STEPS = 1800
    R["max_steps"] = MAX_STEPS

    button = localize_flat("front", ["white button", "button", "white square button",
                                     "grey button"], ["white", "gray"], 0.2, BASE, "button",
                           max_height=0.08, ref_z=TABLE_Z)
    if button is None:
        button = localize_flat("wrist", ["white button", "button"], ["white", "gray"], 0.2, BASE,
                               "button", max_height=0.08, ref_z=TABLE_Z)
    R["button"] = button

    cube = localize_cube("front", COLOR, BASE, TABLE_Z)
    if cube is None:
        cube = localize_cube("wrist", COLOR, BASE, TABLE_Z)
    R["cube0"] = cube
    if cube is None:
        raise ValueError("cube not found")

    CUBE_H = cube["top"] - TABLE_Z
    if CUBE_H < 0.02 or CUBE_H > 0.07:
        R["events"].append({"cube_h_clamped_from": round(CUBE_H, 4)})
        CUBE_H = min(0.07, max(0.02, CUBE_H))
    R["cube_h"] = round(CUBE_H, 4)
    HOVER_Z = TABLE_Z + CUBE_H + 0.10
    GRASP_Z = TABLE_Z + max(0.012, 0.5 * CUBE_H - 0.002)
    PLACE_Z = GRASP_Z + 0.008
    TRACK_Z = PLACE_Z + 0.045
    R["heights"] = {"hover": round(HOVER_Z, 4), "grasp": round(GRASP_Z, 4), "place": round(PLACE_Z, 4),
                    "track": round(TRACK_Z, 4)}

    gripper("open", "open_initial")

    cur = {"x": cube["x"], "y": cube["y"], "yaw": cube["yaw"]}
    attempts = 0
    cycles = []
    on_target = False
    R["servo"] = []
    while STATE["placements"] < N and attempts < N + 4:
        t_now = now_ts()
        if t_now is not None and MAX_STEPS - t_now * TR["hz"] < 120:
            R["events"].append({"stop_cycles_low_budget_tick": round(t_now * TR["hz"])})
            break
        attempts += 1
        cyc = {"attempt": attempts, "t0": t_now}
        qg = grasp_quat(Q_DOWN, cur["yaw"])
        move((cur["x"], cur["y"], HOVER_Z), qg, "hover_cube_" + str(attempts))
        # refine with the wrist camera (close-up, top view)
        w = localize_cube("wrist", COLOR, BASE, TABLE_Z, expect_xy=(cur["x"], cur["y"]), max_dev=0.06)
        cyc["wrist"] = w
        if w is not None:
            dxy = math.hypot(w["x"] - cur["x"], w["y"] - cur["y"])
            cyc["wrist_shift"] = round(dxy, 4)
            cur = {"x": w["x"], "y": w["y"], "yaw": w["yaw"] if w["yaw"] is not None else cur["yaw"]}
            qg = grasp_quat(Q_DOWN, cur["yaw"])
        move((cur["x"], cur["y"], GRASP_Z), qg, "grasp_" + str(attempts), tol=0.006, max_steps=150)
        gripper("close", "close_" + str(attempts))
        g = grip_width()
        cyc["grip_closed"] = round(g, 3)
        if g < 0.04:
            cyc["result"] = "missed"
            on_target = False
            cycles.append(cyc)
            gripper("open", "reopen_" + str(attempts))
            move((cur["x"], cur["y"], HOVER_Z), qg, "retreat_" + str(attempts))
            again = localize_cube("front", COLOR, BASE, TABLE_Z)
            if again is not None:
                cur = {"x": again["x"], "y": again["y"], "yaw": again["yaw"]}
            continue
        move((cur["x"], cur["y"], HOVER_Z), qg, "lift_" + str(attempts))
        g2 = grip_width()
        cyc["grip_lifted"] = round(g2, 3)
        if g2 < 0.04:
            cyc["result"] = "dropped_on_lift"
            on_target = False
            cycles.append(cyc)
            gripper("open", "reopen_" + str(attempts))
            again = localize_cube("front", COLOR, BASE, TABLE_Z)
            if again is not None:
                cur = {"x": again["x"], "y": again["y"], "yaw": again["yaw"]}
            continue
        # ---- v01: servo onto the (possibly moving) target, release, verify ----
        wait = TR["moving"] and on_target
        if TRACK_OK:
            info = servo_place(qg, str(attempts), wait, TRACK_Z, PLACE_Z, 330)
        else:
            info = {"ik_abort": True, "no_tracker": True}
        R["servo"].append(info)
        if info.get("rel_xy") is None:
            # fallback (tracker or IK unavailable): v00-style move to the latest prediction
            fit = fit_track(int(round((now_ts() or 0.0) * TR["hz"])))
            pr = predict(fit, int(round((now_ts() or 0.0) * TR["hz"])) + 12) if fit is not None else None
            if pr is None:
                pr = [target["x"], target["y"]]
            move((pr[0], pr[1], HOVER_Z), qg, "carry_fallback_" + str(attempts))
            move((pr[0], pr[1], PLACE_Z), qg, "place_fallback_" + str(attempts), tol=0.006, max_steps=150)
            info["grip_at_release"] = grip_width()
            gripper("open", "release_" + str(attempts))
            info["rel_xy"] = [pr[0], pr[1]]
        g3 = info["grip_at_release"] if info.get("grip_at_release") is not None else grip_width()
        rel = info["rel_xy"]
        cyc["grip_before_release"] = round(g3, 3)
        cyc["rel_xy"] = [round(rel[0], 4), round(rel[1], 4)]
        try:
            ee_now = get_robot_state().end_effector_poses["primary"].position
            rx = float(ee_now[0]); ry = float(ee_now[1])
        except Exception:
            rx = rel[0]; ry = rel[1]
        move((rx, ry, HOVER_Z), qg, "retract_" + str(attempts))
        ver = {"ok": True, "skipped": True}
        if TR["moving"] and TRACK_OK:
            ver = verify_on_target(rel, str(attempts))
        cyc["verify"] = ver
        if g3 >= 0.04 and ver.get("ok") is not False:
            STATE["placements"] += 1
            cyc["result"] = "placed"
            on_target = True
        else:
            cyc["result"] = "lost_before_release" if g3 < 0.04 else "missed_target"
            on_target = False
        cyc["placements"] = STATE["placements"]
        cyc["t1"] = now_ts()
        cycles.append(cyc)
        R["cycles"] = cycles
        ay = axis_y(qg)
        cur = {"x": rel[0], "y": rel[1], "yaw": math.atan2(ay[1], ay[0]) % (math.pi / 2.0)}
        if not on_target:
            again = localize_cube("front", COLOR, BASE, TABLE_Z, expect_xy=(rel[0], rel[1]), max_dev=0.2)
            cyc["relocalized"] = again
            if again is not None:
                cur = {"x": again["x"], "y": again["y"], "yaw": again["yaw"] if again["yaw"] is not None else cur["yaw"]}
    R["cycles"] = cycles
    R["placements"] = STATE["placements"]

    # ---------- press the button ----------
    if button is None:
        button = localize_flat("front", ["white button", "button", "white square button"],
                               ["white", "gray"], 0.2, BASE, "button", max_height=0.08, ref_z=TABLE_Z)
        R["button_late"] = button
    if button is None:
        raise ValueError("button not found")
    gripper("close", "close_for_button")
    bz_hover = max(HOVER_Z, button["top"] + 0.10)
    move((button["x"], button["y"], bz_hover), Q_DOWN, "hover_button")
    move((button["x"], button["y"], button["top"] + 0.03), Q_DOWN, "prepress", tol=0.008)
    for depth in (0.012, 0.025):
        move((button["x"], button["y"], button["top"] - depth), Q_DOWN,
             "press_" + str(depth), tol=0.004, max_steps=60)
        move((button["x"], button["y"], button["top"] + 0.04), Q_DOWN, "unpress_" + str(depth),
             tol=0.01, max_steps=60)
    R["events"].append("button press sequence finished without termination")
except Exception as exc:
    msg = str(exc)
    if msg != "EPISODE_ENDED":
        R["errors"].append("main: " + msg[:300])
R["placements"] = STATE["placements"]
R["ended"] = STATE["ended"]
R["tracker"] = {"maxspeed": round(TR["maxspeed"], 5), "moving": TR["moving"], "servo_steps": TR["steps"],
                "hist_tail": [[h[0], round(h[1], 4), round(h[2], 4), h[3], h[4]] for h in TR["hist"][-20:]]}
R["t_end"] = None
if not STATE["ended"]:
    try:
        R["t_end"] = now_ts()
    except Exception:
        pass
result = R
