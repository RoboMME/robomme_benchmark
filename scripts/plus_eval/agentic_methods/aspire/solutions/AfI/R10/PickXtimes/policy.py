# v06 — pause-aware target phase. Keyframe analysis of v04/v05 ep1: the target PAUSES while the cube rests
#        on it (t 11.25-13.1 s after the first placement) and resumes on lift, so the fixed-time sinusoid lagged
#        by ~2 s and placements 2-6 were dropped ~10 cm from the real target. v06 keeps the ellipse shape/speed
#        fitted while observing, but re-estimates the phase (time lag) from a fresh, ungated pixel sample after
#        every lift and above the target; a moving-target placement is counted only if the target is seen to
#        STOP under the cube after release (otherwise the cycle is retried). Press immediately once N verified.
# v05 — v04 + timed button press (v04 placed 6/6 on the moving ep1 target but the episode failed at
#        the press: the target had moved away from the resting cube). With a moving target the arm
#        now hovers just above the button cap and presses when the fitted target path brings the
#        target back under the resting cube. Static targets: press immediately, as before.
# v04 — provider-independent tracking (v03 lost ep1 to a sustained SAM3 HTTP-500 outage).
# RoboMME-Plus PickXtimes (task 7), AfI.
# Goal parsed from get_task_context().language: colour, repetition count, target, button.
# 1. t=0, arm at home: SAM localizes the cube ("<colour> cube"), the target ("target", flat-disc
#    filter) and the button (unsaturated raised blob), with patient retries. Colour signatures of
#    target and cube are DERIVED from the SAM clouds' colours (target: its saturated ring pixels).
# 2. From then on, target and cube are tracked per pixel on the front RGB-D image (colour distance
#    to the signature, OpenCV pinhole back-projection with the published intrinsics/camera_pose),
#    so a SAM outage cannot stop the episode. Target samples taken while the arm holds still decide
#    static vs moving (spread > 8 mm); a moving target is fitted with
#    x(t), y(t) = c + a cos(wt) + b sin(wt) (grid search over w) and refitted all episode.
# 3. Each cycle: localize the cube (pixels -> SAM -> last release point), top-down grasp at its
#    top-face centre, verify hold by gripper width, carry at transit height, descend in two stages
#    re-predicting the target, release with a small lead, retreat. After N placements press the button.
# Every location is derived at runtime from images + depth.
import math

report = {"version": "v06", "log": [], "cycles": [], "tsamples": []}
STATE = {"ended": False}

NUMBER_WORDS = {"once": 1, "twice": 2, "thrice": 3, "one": 1, "two": 2, "three": 3,
                "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12}
COLORS = ("red", "green", "blue", "yellow", "purple", "orange", "pink", "white", "black",
          "brown", "gray", "grey", "cyan", "magenta")
HOLD_MIN_GRIP = 0.15        # measured width while holding the 4 cm cube was ~0.50; empty close ~0
TRANSIT_ABOVE = 0.11        # EE transit height above the highest relevant surface
STATIC_SPREAD = 0.008       # target samples within this radius of their median => static target
OBS_STATIC_TICKS = 24       # first observation window (ticks) before deciding static vs moving
OBS_MOVING_TICKS = 140      # total observation window when moving (7 s at 20 Hz)
OBS_STRIDE = 2              # ticks between target samples while holding still
RELEASE_LEAD_S = 0.2        # aim the landing this far ahead along the target's path
SEG_RETRIES = 6             # retries of a failed (provider_unavailable) segmentation call
INIT_SEG_RETRIES = 60       # patient retries for the t=0 localization (no sim time passes)
COLOR_TOL = 55.0            # RGB distance to a derived colour signature for a pixel match
MIN_SAT = 40                # pixel saturation (max-min channel) required for a signature match


def note(tag, data=None):
    report["log"].append([tag, data])


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def check_terminal(res, tag, info):
    term = False
    try:
        term = bool(res.terminated) or bool(res.truncated)
    except Exception:
        pass
    if term:
        info["terminal"] = True
        note("terminal", info)
        STATE["ended"] = True
        raise ValueError("EPISODE_ENDED at " + tag)


def exec_info(res, tag):
    info = {"tag": tag}
    try:
        info["ok"] = bool(res.ok)
    except Exception:
        info["ok"] = None
    try:
        info["steps"] = int(res.steps_executed)
    except Exception:
        pass
    try:
        if res.error is not None:
            info["error"] = str(res.error)[:200]
    except Exception:
        pass
    note("exec", info)
    check_terminal(res, tag, info)
    return info


def now():
    return float(get_robot_state().timestamp_s)


def ee_state():
    st = get_robot_state()
    pose = st.end_effector_poses["primary"]
    return {"pos": [round(float(v), 4) for v in pose.position],
            "grip": round(float(st.gripper_positions["primary"]), 4),
            "t": round(float(st.timestamp_s), 3)}


def hold_tick(grip):
    st = get_robot_state()
    res = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=[float(v) for v in st.joint_positions["primary"]],
        gripper_position=grip, embodiment=st.embodiment)}))
    check_terminal(res, "hold", {"tag": "hold"})
    return res


def cloud_stats(points):
    pts = [p for p in points]
    if len(pts) == 0:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    zs = [float(p[2]) for p in pts]
    return {"n": len(pts),
            "x": [round(quantile(xs, q), 4) for q in (0.05, 0.5, 0.95)],
            "y": [round(quantile(ys, q), 4) for q in (0.05, 0.5, 0.95)],
            "z": [round(quantile(zs, q), 4) for q in (0.05, 0.5, 0.95)]}


def cloud_color(cloud):
    try:
        cols = cloud.colors
        if cols is None:
            return None
        n = len(cols)
        if n == 0:
            return None
        acc = [0.0, 0.0, 0.0]
        stride = max(1, n // 200)
        cnt = 0
        for i in range(0, n, stride):
            c = cols[i]
            acc[0] += float(c[0]); acc[1] += float(c[1]); acc[2] += float(c[2])
            cnt += 1
        return [round(a / cnt, 3) for a in acc]
    except Exception:
        return None


def candidates(camera, prompt, base_frame, limit=5):
    """Segment with retries: a provider_unavailable (recoverable HTTP 500) answer is retried;
    observation calls do not advance simulation time."""
    out = []
    seg = None
    for attempt in range(SEG_RETRIES):
        seg = segment_text(camera, prompt)
        if seg.ok:
            break
        STATE["seg_failures"] = STATE.get("seg_failures", 0) + 1
        try:
            STATE["last_seg_error"] = str(seg.error.code) + ": " + str(seg.error.message)[:120]
        except Exception:
            pass
    if seg is None or not seg.ok:
        return out
    for idx, cand in enumerate(seg.segmentations[:limit]):
        item = {"prompt": prompt, "rank": idx, "score": round(float(cand.score), 3)}
        try:
            cloud = mask_to_point_cloud(cand, camera, target_frame=base_frame)
            item["points"] = cloud.points
            item["stats"] = cloud_stats(cloud.points)
            item["rgb"] = cloud_color(cloud)
            try:
                item["sig"] = signature_from_cloud(cloud.colors, prompt.find("target") >= 0)
            except Exception:
                item["sig"] = None
        except Exception as exc:
            item["points"] = []
            item["stats"] = None
            item["err"] = str(exc)[:200]
        out.append(item)
    return out


def public_item(item):
    return {k: item[k] for k in item if k != "points"}


def parse_goal(text):
    low = text.lower().replace(",", " ").replace(".", " ").replace("-", " ")
    words = low.split()
    count = None
    for w in words:
        if w.isdigit():
            count = int(w)
            break
        if w in NUMBER_WORDS:
            count = NUMBER_WORDS[w]
            break
    color = None
    for w in words:
        if w in COLORS:
            color = w
            break
    noun = "cube"
    for cand in ("cube", "block", "box"):
        if cand in words:
            noun = cand
            break
    return {"count": count, "color": color, "noun": noun,
            "button": "button" in low, "target": "target" in low}


def downward_quat(home_quat):
    w, x, y, z = (float(v) for v in home_quat)
    zz = 1.0 - 2.0 * (x * x + y * y)
    if zz < -0.9:
        return [w, x, y, z]
    return [0.0, 1.0, 0.0, 0.0]


def saturation(rgb):
    if rgb is None:
        return None
    return max(rgb) - min(rgb)


def pick_cube(cands, color):
    best = None
    for item in cands:
        st = item.get("stats")
        if st is None or st["n"] < 20:
            continue
        dx = st["x"][2] - st["x"][0]
        dy = st["y"][2] - st["y"][0]
        dz = st["z"][2] - st["z"][0]
        if dx > 0.10 or dy > 0.10 or dz > 0.10 or st["z"][2] > 0.15:
            continue
        rgb = item.get("rgb")
        if color in ("red", "green", "blue") and rgb is not None:
            ch = {"red": 0, "green": 1, "blue": 2}[color]
            others = [rgb[i] for i in range(3) if i != ch]
            if rgb[ch] < 1.5 * max(others) + 1e-6:
                continue
        if best is None or item["score"] > best["score"]:
            best = item
    return best


def object_frame(item):
    pts = item["points"]
    zs = [float(p[2]) for p in pts]
    top = quantile(zs, 0.95)
    bottom = quantile(zs, 0.03)
    topface = [p for p in pts if float(p[2]) > top - 0.008]
    if len(topface) < 10:
        topface = pts
    cx = quantile((float(p[0]) for p in topface), 0.5)
    cy = quantile((float(p[1]) for p in topface), 0.5)
    height = top - bottom
    grasp_z = top - 0.5 * height
    if grasp_z < bottom + 0.01:
        grasp_z = bottom + 0.01
    return {"cx": cx, "cy": cy, "top": top, "bottom": bottom, "height": height,
            "grasp_z": grasp_z}


def flat_target(cands, min_extent=0.02):
    best = None
    for item in cands:
        st = item.get("stats")
        if st is None or st["n"] < 20:
            continue
        dx = st["x"][2] - st["x"][0]
        dy = st["y"][2] - st["y"][0]
        dz = st["z"][2] - st["z"][0]
        if dz > 0.02 or dx > 0.15 or dy > 0.15 or dx < min_extent or dy < min_extent:
            continue
        if st["z"][2] > 0.03:
            continue
        if best is None or item["score"] > best["score"]:
            best = item
    return best


def far_from(st, points_xy, radius):
    cx, cy = st["x"][1], st["y"][1]
    for p in points_xy:
        if math.sqrt((cx - p[0]) ** 2 + (cy - p[1]) ** 2) < radius:
            return False
    return True


def pick_button(cands, avoid_xy):
    best = None
    for item in cands:
        st = item.get("stats")
        if st is None or st["n"] < 10:
            continue
        dx = st["x"][2] - st["x"][0]
        dy = st["y"][2] - st["y"][0]
        if dx > 0.10 or dy > 0.10:
            continue
        if st["z"][2] < 0.015 or st["z"][2] > 0.08:
            continue
        sat = saturation(item.get("rgb"))
        if sat is not None and sat > 60:
            continue
        if not far_from(st, avoid_xy, 0.04):
            continue
        if best is None or item["score"] > best["score"]:
            best = item
    return best


def go(pos, quat, base_frame, tag, tol=0.01, max_steps=250):
    res = move_to_pose(Pose(tuple(pos), tuple(quat), base_frame), tolerance=tol, max_steps=max_steps)
    return exec_info(res, tag)


def locate_cube(obj_prompts, color, base_frame):
    for prompt in obj_prompts:
        cs = candidates("front", prompt, base_frame)
        chosen = pick_cube(cs, color)
        if chosen is not None:
            return chosen
    return None


# ---------------- target motion model ----------------
MODEL = {"kind": None, "w": None, "ax": None, "ay": None, "static": None, "rms": None, "lag": 0.0}


def solve3(m, v):
    """Solve 3x3 linear system m x = v (Cramer); returns None when singular."""
    def det(a):
        return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    d = det(m)
    if abs(d) < 1e-12:
        return None
    out = []
    for col in range(3):
        a = [[(v[r] if c == col else m[r][c]) for c in range(3)] for r in range(3)]
        out.append(det(a) / d)
    return out


def fit_at(w, samples):
    s1 = sc = ss = scc = scs = sss = 0.0
    sx = sxc = sxs = sxx = 0.0
    sy = syc = syq = syy = 0.0
    for smp in samples:
        t, x, y = smp[0], smp[1], smp[2]
        c = math.cos(w * t)
        s = math.sin(w * t)
        s1 += 1.0; sc += c; ss += s; scc += c * c; scs += c * s; sss += s * s
        sx += x; sxc += x * c; sxs += x * s; sxx += x * x
        sy += y; syc += y * c; syq += y * s; syy += y * y
    m = [[s1, sc, ss], [sc, scc, scs], [ss, scs, sss]]
    ax = solve3(m, [sx, sxc, sxs])
    ay = solve3(m, [sy, syc, syq])
    if ax is None or ay is None:
        return None
    rss = (sxx - (ax[0] * sx + ax[1] * sxc + ax[2] * sxs)) + (syy - (ay[0] * sy + ay[1] * syc + ay[2] * syq))
    if rss < 0.0:
        rss = 0.0
    return {"w": w, "ax": ax, "ay": ay, "rss": rss}


def fit_motion(samples, w_lo, w_hi, n):
    best = None
    ratio = (w_hi / w_lo) ** (1.0 / (n - 1))
    w = w_lo
    for i in range(n):
        f = fit_at(w, samples)
        if f is not None and (best is None or f["rss"] < best["rss"]):
            best = f
        w = w * ratio
    return best


def median_xy(samples):
    return (quantile((s[1] for s in samples), 0.5), quantile((s[2] for s in samples), 0.5))


def update_model(full_search):
    samples = report["tsamples"]
    if len(samples) < 3:
        return
    mx, my = median_xy(samples)
    spread = max(math.sqrt((s[1] - mx) ** 2 + (s[2] - my) ** 2) for s in samples)
    if spread < STATIC_SPREAD:
        MODEL["kind"] = "static"
        MODEL["static"] = (mx, my)
        return
    if len(samples) < 4:
        return
    if full_search or MODEL["w"] is None:
        best = fit_motion(samples, 2.0 * math.pi / 30.0, 2.0 * math.pi / 1.2, 400)
    else:
        best = fit_motion(samples, MODEL["w"] * 0.9, MODEL["w"] * 1.1, 60)
    if best is None:
        return
    fine = fit_motion(samples, best["w"] * 0.985, best["w"] * 1.015, 31)
    if fine is not None and fine["rss"] <= best["rss"]:
        best = fine
    MODEL["kind"] = "moving"
    MODEL["w"] = best["w"]
    MODEL["ax"] = best["ax"]
    MODEL["ay"] = best["ay"]
    MODEL["rms"] = math.sqrt(best["rss"] / len(samples))


def predict(t):
    if MODEL["kind"] == "moving":
        w = MODEL["w"]
        t = t - MODEL["lag"]
        c = math.cos(w * t)
        s = math.sin(w * t)
        ax, ay = MODEL["ax"], MODEL["ay"]
        return (ax[0] + ax[1] * c + ax[2] * s, ay[0] + ay[1] * c + ay[2] * s)
    return MODEL["static"]




PRESS_HOVER = 0.014         # closed-gripper hover above the cap; the press registered at ~cap+0.005
PRESS_LATENCY_S = 0.2       # assumed time from issuing the final press move to the registered press
ALIGN_MAX = 0.03            # only wait for the target if its path passes this close to the cube


def press_time(cxy, t_now):
    """First time the predicted target centre is closest to the resting cube (within one period)."""
    if MODEL["kind"] != "moving" or MODEL["w"] is None:
        return None
    period = 2.0 * math.pi / MODEL["w"]
    n = int((period + 0.6) / 0.05) + 1
    ds = []
    for k in range(n):
        t = t_now + PRESS_LATENCY_S + 0.05 * k
        p = predict(t)
        ds.append(math.sqrt((p[0] - cxy[0]) ** 2 + (p[1] - cxy[1]) ** 2))
    dmin = min(ds)
    for k in range(n):
        if ds[k] <= dmin + 0.002:
            t = t_now + PRESS_LATENCY_S + 0.05 * k
            return {"t_target": round(t, 3), "t_press": round(t - PRESS_LATENCY_S, 3),
                    "dmin": round(dmin, 4), "period": round(period, 3)}
    return None


RESYNC_MAX_DIST = 0.015     # a sample must lie this close to the fitted path to re-estimate the phase
STOP_EPS = 0.005            # two samples 0.2 s apart closer than this => target stopped (placement seen)


def rephase(t, x, y):
    """Re-estimate the target's phase on its fitted closed path from one sample (shape/speed fixed)."""
    if MODEL["kind"] != "moving" or MODEL["w"] is None:
        return None
    w = MODEL["w"]
    period = 2.0 * math.pi / w
    saved = MODEL["lag"]
    MODEL["lag"] = 0.0
    best = None
    n = int(period / 0.02) + 1
    for k in range(n):
        tau = 0.02 * k
        p = predict(tau)
        d = (p[0] - x) ** 2 + (p[1] - y) ** 2
        if best is None or d < best[1]:
            best = (tau, d)
    lo = best[0] - 0.02
    for k in range(21):
        tau = lo + 0.002 * k
        p = predict(tau)
        d = (p[0] - x) ** 2 + (p[1] - y) ** 2
        if d < best[1]:
            best = (tau, d)
    dist = math.sqrt(best[1])
    if dist > RESYNC_MAX_DIST:
        MODEL["lag"] = saved
        return {"lag": None, "dist": round(dist, 4), "kept": round(saved, 3)}
    lag = t - best[0]
    lag = lag - period * math.floor(lag / period)
    MODEL["lag"] = lag
    return {"lag": round(lag, 3), "dist": round(dist, 4), "prev": round(saved, 3)}


def resync(tag, rec):
    """Ungated pixel sample of the target -> phase re-estimate. Returns the sample or None."""
    smp = pixel_target_sample(None)
    if smp is None:
        rec.setdefault("resync", []).append([tag, None])
        return None
    r = rephase(smp[0], smp[1], smp[2])
    rec.setdefault("resync", []).append([tag, smp[0], smp[1], smp[2], r])
    return smp


# ---------------- per-pixel RGB-D tracking on the front camera ----------------
def to_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        return [[px for px in row] for row in arr]


def quat_to_rot(q):
    w, x, y, z = (float(v) for v in q)
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def front_frame():
    obs = get_observation(camera_names=["front"])
    cam = obs.cameras["front"]
    k = cam.intrinsics
    try:
        t = float(obs.timestamp_s)
    except Exception:
        t = now()
    return {"t": t, "rgb": to_rows(cam.rgb), "dep": to_rows(cam.depth_m),
            "fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]), "cy": float(k[1][2]),
            "rot": quat_to_rot(cam.camera_pose.quaternion_wxyz),
            "pos": [float(v) for v in cam.camera_pose.position]}


def back_project(fr, u, v, d):
    xc = (u + 0.5 - fr["cx"]) / fr["fx"] * d
    yc = (v + 0.5 - fr["cy"]) / fr["fy"] * d
    r = fr["rot"]
    p = fr["pos"]
    return (r[0][0] * xc + r[0][1] * yc + r[0][2] * d + p[0],
            r[1][0] * xc + r[1][1] * yc + r[1][2] * d + p[1],
            r[2][0] * xc + r[2][1] * yc + r[2][2] * d + p[2])


def match_points(fr, sig, stride=1, zmax=0.2):
    """3D points of pixels whose colour is within COLOR_TOL of sig (and saturated)."""
    out = []
    tol2 = COLOR_TOL * COLOR_TOL
    s0, s1, s2 = sig[0], sig[1], sig[2]
    rgb = fr["rgb"]
    dep = fr["dep"]
    for v in range(0, len(rgb), stride):
        row = rgb[v]
        drow = dep[v]
        for u in range(0, len(row), stride):
            px = row[u]
            r, g, b = px[0], px[1], px[2]
            if max(r, g, b) - min(r, g, b) < MIN_SAT:
                continue
            if (r - s0) * (r - s0) + (g - s1) * (g - s1) + (b - s2) * (b - s2) > tol2:
                continue
            d = float(drow[u])
            if d <= 0.05 or d > 3.0:
                continue
            p = back_project(fr, u, v, d)
            if p[2] < -0.02 or p[2] > zmax:
                continue
            out.append(p)
    return out


def cluster_near(points, seed, radius):
    """Points within radius (xy) of seed, re-centred twice; seed None -> median."""
    if len(points) == 0:
        return []
    if seed is None:
        seed = (quantile((p[0] for p in points), 0.5), quantile((p[1] for p in points), 0.5))
    sel = points
    cx, cy = seed[0], seed[1]
    for it in range(3):
        sel = [p for p in points if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 < radius * radius]
        if len(sel) == 0:
            return []
        cx = quantile((p[0] for p in sel), 0.5)
        cy = quantile((p[1] for p in sel), 0.5)
    return sel


def signature_from_cloud(cloud_colors, saturated_only):
    cols = []
    n = len(cloud_colors)
    stride = max(1, n // 400)
    for i in range(0, n, stride):
        c = cloud_colors[i]
        cols.append((float(c[0]), float(c[1]), float(c[2])))
    if saturated_only:
        sat = [c for c in cols if max(c) - min(c) > 50]
        if len(sat) >= 10:
            cols = sat
    if len(cols) == 0:
        return None
    # mode seeking: most populous 32-level colour bin, then mean of colours near it (x2)
    bins = {}
    for c in cols:
        key = (int(c[0]) // 32, int(c[1]) // 32, int(c[2]) // 32)
        bins[key] = bins.get(key, 0) + 1
    best = None
    for key in bins:
        if best is None or bins[key] > bins[best]:
            best = key
    m = [best[0] * 32 + 16.0, best[1] * 32 + 16.0, best[2] * 32 + 16.0]
    for it in range(2):
        near = [c for c in cols if (c[0] - m[0]) ** 2 + (c[1] - m[1]) ** 2 + (c[2] - m[2]) ** 2 < 2500]
        if len(near) >= 5:
            m = [sum(c[i] for c in near) / len(near) for i in range(3)]
    return [round(v, 1) for v in m]


def color_prior(color):
    """Fallback cube signature from the language colour word (only red/green/blue)."""
    table = {"red": [190.0, 5.0, 5.0], "green": [5.0, 190.0, 5.0], "blue": [5.0, 5.0, 185.0]}
    return table.get(color)


def pixel_target_sample(seed, gate=None):
    fr = front_frame()
    pts = match_points(fr, SIG["target"], stride=1, zmax=0.03)
    sel = cluster_near(pts, seed, 0.06)
    if len(sel) < 15:
        return None
    xs = [p[0] for p in sel]
    ys = [p[1] for p in sel]
    x = 0.5 * (quantile(xs, 0.05) + quantile(xs, 0.95))
    y = 0.5 * (quantile(ys, 0.05) + quantile(ys, 0.95))
    ext = 0.5 * ((quantile(xs, 0.95) - quantile(xs, 0.05)) + (quantile(ys, 0.95) - quantile(ys, 0.05)))
    t = fr["t"]
    if gate is not None and MODEL["kind"] is not None:
        px, py = predict(t)
        if math.sqrt((px - x) ** 2 + (py - y) ** 2) > gate:
            return None
    smp = [round(t, 3), round(x, 4), round(y, 4), len(sel), round(ext, 4)]
    report["tsamples"].append(smp)
    return smp


def pixel_cube(seed):
    """Cube cloud from pixels of the cube signature, clustered near seed."""
    fr = front_frame()
    pts = match_points(fr, SIG["cube"], stride=1, zmax=0.15)
    sel = cluster_near(pts, seed, 0.04)
    if len(sel) < 25:
        return None
    return {"points": sel, "source": "pixels", "n": len(sel)}


SIG = {"target": None, "cube": None}


def patient_candidates(prompt, base_frame):
    for attempt in range(INIT_SEG_RETRIES // SEG_RETRIES):
        cs = candidates("front", prompt, base_frame)
        if len(cs) > 0:
            return cs
        get_observation()
    return []


def locate_cube_now(obj_prompts, color, base_frame, seed):
    """Pixels first (SAM-independent), then SAM, else None."""
    if SIG["cube"] is not None:
        found = pixel_cube(seed)
        if found is None and seed is not None:
            found = pixel_cube(None)
        if found is not None:
            return found
    for prompt in obj_prompts:
        cs = candidates("front", prompt, base_frame)
        chosen = pick_cube(cs, color)
        if chosen is not None:
            chosen["source"] = "sam"
            return chosen
    return None


try:
    context = get_task_context()
    goal = context.language
    report["language"] = goal
    try:
        report["task_metadata"] = dict(robomme_plus.get_task_metadata())
    except Exception as exc:
        report["task_metadata_err"] = str(exc)[:200]
    try:
        video = robomme_plus.get_conditioning_video()
        report["video_len"] = len(video)
    except Exception as exc:
        report["video_err"] = str(exc)[:200]

    state = get_robot_state()
    base_frame = state.base_frame
    report["home"] = ee_state()
    quat = downward_quat(state.end_effector_poses["primary"].quaternion_wxyz)

    parsed = parse_goal(goal)
    report["parsed"] = parsed
    count = parsed["count"] if parsed["count"] is not None else 1
    if count > 12:
        count = 12
    color = parsed["color"]
    obj_prompts = []
    if color is not None:
        obj_prompts.append(color + " " + parsed["noun"])
    obj_prompts.append(parsed["noun"])

    # ---- t=0, arm at home: target first (it may move), then cube, then button ----
    target = None
    tcloud_colors = None
    for prompt in ("target", "round target", "bullseye target"):
        tcs = patient_candidates(prompt, base_frame)
        target = flat_target(tcs)
        if target is not None:
            report["target_prompt"] = prompt
            break
    if target is None:
        raise ValueError("target not localized")
    tst = target["stats"]
    tz = tst["z"][2]
    t_radius = 0.25 * ((tst["x"][2] - tst["x"][0]) + (tst["y"][2] - tst["y"][0]))
    report["target0"] = public_item(target)
    tx0 = 0.5 * (tst["x"][0] + tst["x"][2])
    ty0 = 0.5 * (tst["y"][0] + tst["y"][2])
    SIG["target"] = target.get("sig")

    cube0 = None
    for prompt in obj_prompts:
        cs = patient_candidates(prompt, base_frame)
        cube0 = pick_cube(cs, color)
        if cube0 is not None:
            break
    if cube0 is not None:
        SIG["cube"] = cube0.get("sig")
        report["cube0"] = public_item(cube0)
    else:
        SIG["cube"] = color_prior(color)
        report["cube0_fallback"] = "language colour prior"
        cube0 = pixel_cube(None) if SIG["cube"] is not None else None
        if cube0 is None:
            raise ValueError("cube not localized")
    report["signatures"] = {"target": SIG["target"], "cube": SIG["cube"]}
    geo0 = object_frame(cube0)
    cube_xy = (geo0["cx"], geo0["cy"])

    button = None
    if parsed["button"]:
        for prompt in ("button", "white button", "round button", "push button", "gray button"):
            bcs = patient_candidates(prompt, base_frame)
            button = pick_button(bcs, [(tx0, ty0), cube_xy])
            if button is not None:
                break
        if button is None:
            report["button_error"] = "button not localized"
        else:
            bpts = button["points"]
            btop = quantile((float(p[2]) for p in bpts), 0.95)
            capface = [p for p in bpts if float(p[2]) > btop - 0.006]
            if len(capface) < 5:
                capface = bpts
            bx = quantile((float(p[0]) for p in capface), 0.5)
            by = quantile((float(p[1]) for p in capface), 0.5)
            report["button"] = {"x": round(bx, 4), "y": round(by, 4), "top": round(btop, 4),
                                "score": button["score"], "prompt": button["prompt"]}

    # ---- observe the target while holding still: static or moving? ----
    if SIG["target"] is None:
        raise ValueError("no target colour signature")
    seed = (tx0, ty0)
    last = pixel_target_sample(seed)
    if last is not None:
        seed = (last[1], last[2])
    ticks = 0
    limit = OBS_STATIC_TICKS
    while ticks < limit:
        for k in range(OBS_STRIDE):
            hold_tick(1.0)
        ticks += OBS_STRIDE
        last = pixel_target_sample(seed)
        if last is not None:
            seed = (last[1], last[2])
        if ticks >= OBS_STATIC_TICKS and limit == OBS_STATIC_TICKS:
            update_model(True)
            if MODEL["kind"] == "moving":
                limit = OBS_MOVING_TICKS
    update_model(True)
    report["model0"] = {"kind": MODEL["kind"], "w": MODEL["w"], "ax": MODEL["ax"], "ay": MODEL["ay"],
                        "static": MODEL["static"], "rms": MODEL["rms"], "n": len(report["tsamples"])}
    if MODEL["kind"] is None:
        MODEL["kind"] = "static"
        MODEL["static"] = (tx0, ty0)
        report["model0_fallback"] = "t=0 SAM target"

    # ---- N verified placements on the target ----
    placed = 0
    cycle = 0
    geo = geo0
    fresh = True
    d_desc = 0.45
    last_release = None
    while placed < count and cycle < count + 4:
        rec = {"cycle": cycle, "placed_before": placed}
        report["cycles"].append(rec)
        cycle += 1
        if not fresh:
            seed_xy = (geo["cx"], geo["cy"])
            found = locate_cube_now(obj_prompts, color, base_frame, seed_xy)
            if found is not None:
                geo = object_frame(found)
                rec["cube_source"] = found.get("source")
            elif last_release is not None:
                rec["cube_source"] = "last release point"
                geo = {"cx": last_release[0], "cy": last_release[1], "top": geo["top"],
                       "bottom": geo["bottom"], "height": geo["height"], "grasp_z": geo["grasp_z"]}
            else:
                rec["error"] = "cube not re-localized"
                break
        fresh = False
        rec["geo"] = {k: round(geo[k], 4) for k in geo}
        cx, cy = geo["cx"], geo["cy"]
        transit_z = max(geo["top"], tz + geo["height"]) + TRANSIT_ABOVE
        release_z = geo["grasp_z"] + (tz - geo["bottom"]) + 0.012
        exec_info(open_gripper(), "open")
        go((cx, cy, transit_z), quat, base_frame, "pre_grasp", tol=0.02)
        go((cx, cy, geo["grasp_z"]), quat, base_frame, "grasp", tol=0.01)
        exec_info(close_gripper(), "close")
        go((cx, cy, transit_z), quat, base_frame, "lift", tol=0.02)
        held = ee_state()
        rec["lifted"] = held
        if held["grip"] < HOLD_MIN_GRIP:
            rec["error"] = "empty grasp"
            exec_info(open_gripper(), "open_after_miss")
            continue
        if MODEL["kind"] == "moving":
            resync("after_lift", rec)
            t0 = now()
            p = predict(t0 + 0.6 + d_desc + RELEASE_LEAD_S)
            go((p[0], p[1], transit_z), quat, base_frame, "above_target", tol=0.02)
            resync("above_target", rec)
            t1 = now()
            p = predict(t1 + 0.5 * d_desc + RELEASE_LEAD_S)
            go((p[0], p[1], release_z + 0.04), quat, base_frame, "approach", tol=0.015)
            t2 = now()
            p = predict(t2 + 0.25 + RELEASE_LEAD_S)
            go((p[0], p[1], release_z), quat, base_frame, "lower", tol=0.01)
            t3 = now()
            d_desc = max(0.2, min(1.0, (t3 - t1)))
            rec["release_pred"] = [round(p[0], 4), round(p[1], 4)]
            rec["release_t"] = round(t3, 3)
        else:
            p = MODEL["static"]
            go((p[0], p[1], transit_z), quat, base_frame, "above_target", tol=0.02)
            go((p[0], p[1], release_z), quat, base_frame, "lower", tol=0.01)
        exec_info(open_gripper(), "release")
        t4 = now()
        last_release = (p[0], p[1])
        tp = predict(t4)
        rec["target_at_release"] = [round(tp[0], 4), round(tp[1], 4)]
        go((p[0], p[1], transit_z), quat, base_frame, "retreat", tol=0.02)
        stopped = None
        s1 = None
        if MODEL["kind"] == "moving":
            s1 = pixel_target_sample(None)
            for k in range(4):
                hold_tick(1.0)
            s2 = pixel_target_sample(None)
            if s1 is not None and s2 is not None:
                mv = math.sqrt((s1[1] - s2[1]) ** 2 + (s1[2] - s2[2]) ** 2)
                stopped = mv < STOP_EPS
                rec["after_release"] = {"s1": s1, "s2": s2, "moved": round(mv, 4), "stopped": stopped}
                tp = (s2[1], s2[2])
        placed += 1
        rec["placed_after"] = placed
        landed = locate_cube_now(obj_prompts, color, base_frame, (p[0], p[1]))
        if landed is not None:
            lg = object_frame(landed)
            off = math.sqrt((lg["cx"] - tp[0]) ** 2 + (lg["cy"] - tp[1]) ** 2)
            rec["landed"] = {"x": round(lg["cx"], 4), "y": round(lg["cy"], 4), "src": landed.get("source"),
                             "top": round(lg["top"], 4), "off_vs_target_at_release": round(off, 4)}
            geo = lg
            fresh = True
            if stopped is False:
                rec["warn"] = "target kept moving after release -> placement not registered; not counted"
                placed -= 1
            elif off > t_radius + 0.03:
                rec["warn"] = "cube landed off target; not counted"
                placed -= 1
        elif stopped is False:
            rec["warn"] = "target kept moving after release -> placement not registered; not counted"
            placed -= 1
    report["placed"] = placed
    report["model_final"] = {"kind": MODEL["kind"], "w": MODEL["w"], "ax": MODEL["ax"],
                             "ay": MODEL["ay"], "static": MODEL["static"], "rms": MODEL["rms"],
                             "lag": MODEL["lag"], "n": len(report["tsamples"])}

    if parsed["button"] and "button" not in report:
        for prompt in ("button", "white button", "round button"):
            bcs = patient_candidates(prompt, base_frame)
            button = pick_button(bcs, [cube_xy])
            if button is not None:
                st = button["stats"]
                report["button"] = {"x": st["x"][1], "y": st["y"][1], "top": st["z"][2],
                                    "score": button["score"], "prompt": prompt, "late": True}
                break
    if parsed["button"] and "button" in report and placed >= count:
        b = report["button"]
        exec_info(close_gripper(), "close_for_press")
        go((b["x"], b["y"], b["top"] + 0.08), quat, base_frame, "above_button", tol=0.02)
        go((b["x"], b["y"], b["top"] + 0.04), quat, base_frame, "near_button", tol=0.01)
        go((b["x"], b["y"], b["top"] + PRESS_HOVER), quat, base_frame, "hover_button", tol=0.004)
        cxy = (geo["cx"], geo["cy"])
        if len(report["cycles"]) > 0 and "landed" not in report["cycles"][-1] and last_release is not None:
            cxy = last_release
        pr = {"cube": [round(cxy[0], 4), round(cxy[1], 4)], "hover": ee_state()}
        report["press_timing"] = pr
        last_rec = report["cycles"][-1] if len(report["cycles"]) > 0 else {}
        pr["last_stopped"] = last_rec.get("after_release", {}).get("stopped")
        if MODEL["kind"] == "moving" and pr["last_stopped"] is not True:
            resync("before_press", pr)
            # v05: the episode is judged at the press -> press when the moving target is back
            # under the resting cube (the cube was dropped on the target's own periodic path).
            plan = press_time(cxy, now())
            pr["plan0"] = plan
            ticks = 0
            while plan is not None and plan["dmin"] < ALIGN_MAX and now() < plan["t_press"] - 0.026:
                hold_tick(0.0)
                ticks += 1
                if ticks % 4 == 0:
                    pt = predict(now())
                    if ticks % 12 == 0 and resync("press_wait", pr) is not None:
                        plan = press_time(cxy, now())
                if ticks > 400:
                    break
            pr["plan"] = plan
            pr["wait_ticks"] = ticks
            pr["t_issue"] = round(now(), 3)
            tp = predict(now() + PRESS_LATENCY_S)
            pr["pred_offset_at_press"] = round(math.sqrt((tp[0] - cxy[0]) ** 2 + (tp[1] - cxy[1]) ** 2), 4)
        go((b["x"], b["y"], b["top"] - 0.01), quat, base_frame, "press", tol=0.01, max_steps=120)
        go((b["x"], b["y"], b["top"] + 0.08), quat, base_frame, "unpress", tol=0.02)
    elif parsed["button"]:
        report["button_skipped"] = "placements incomplete or button unknown"
except Exception as exc:
    report["exception"] = str(exc)[:400]

try:
    if not STATE["ended"]:
        report["final"] = ee_state()
except Exception:
    pass
report["ended"] = STATE["ended"]
report["seg_failures"] = STATE.get("seg_failures", 0)
report["last_seg_error"] = STATE.get("last_seg_error")
result = report
