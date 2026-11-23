import math
# VideoPlaceOrder v04 — v03 with the target colour class derived at runtime from segment_text('front','target') (no recording-derived target hue).
# ---- CORE (pure Python; shared by the program and the offline replay harness) ----
COLOR_WORDS = ("red", "green", "blue")
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "1st": 1, "2nd": 2,
            "3rd": 3, "4th": 4, "5th": 5, "6th": 6}

def parse_goal(text):
    words = text.lower().replace(",", " ").replace(".", " ").split()
    color = None
    k = None
    last = False
    for i, w in enumerate(words):
        if w in COLOR_WORDS and color is None and i + 1 < len(words) and words[i + 1] in ("cube", "block"):
            color = w
        if w in ORDINALS and k is None:
            k = ORDINALS[w]
        if w == "last" and k is None:
            last = True
    if color is None:
        for w in words:
            if w in COLOR_WORDS:
                color = w
                break
    return color, k, last

TARGET_HUE = []   # [hi, mid, lo] channel order of the target discs, derived at runtime from segment_text("front", "target")

def is_target(p):
    # generic hue-class rule over the runtime-derived channel order: dominant channel clearly above the weakest,
    # bright enough, and not a dark saturated object (excludes dark cubes)
    if not TARGET_HUE:
        return False
    hi = TARGET_HUE[0]; mid = TARGET_HUE[1]; lo = TARGET_HUE[2]
    return p[hi] > p[lo] + 14 and p[hi] > 80 and (p[mid] >= 70 or p[lo] >= 70)

def is_cube(p, color):
    r = p[0]; g = p[1]; b = p[2]
    if color == "red":
        return r > 150 and g < 60 and b < 60
    if color == "green":
        return g > 150 and r < 100 and b < 100
    if color == "blue":
        return b > 120 and r < 60 and g < 60
    return False

def image_rows(rgb):
    try:
        return rgb.tolist()
    except Exception:
        rows = []
        for y in range(len(rgb)):
            rows.append([[int(v) for v in px] for px in rgb[y]])
        return rows

def scan_frame(rows, color, step=2):
    """Return (target_blobs, cube) in full-res pixel units. blob=(cx, cy, area_px)."""
    h = len(rows); w = len(rows[0])
    cells = {}
    cube_n = 0; cube_x = 0.0; cube_y = 0.0
    for y in range(0, h, step):
        row = rows[y]
        for x in range(0, w, step):
            p = row[x]
            if is_target(p):
                cells[(x // step, y // step)] = 1
            elif is_cube(p, color):
                cube_n += 1; cube_x += x; cube_y += y
    seen = {}
    blobs = []
    for key in cells:
        if key in seen:
            continue
        stack = [key]; seen[key] = 1
        n = 0; sx = 0.0; sy = 0.0
        while stack:
            c = stack.pop()
            n += 1; sx += c[0]; sy += c[1]
            for dx in (-2, -1, 0, 1, 2):
                for dy in (-2, -1, 0, 1, 2):
                    q = (c[0] + dx, c[1] + dy)
                    if q in cells and q not in seen:
                        seen[q] = 1
                        stack.append(q)
        area = n * step * step
        if area >= 12:
            blobs.append((sx / n * step + step * 0.5 - 0.5, sy / n * step + step * 0.5 - 0.5, area))
    cube = None
    area = cube_n * step * step
    if area >= 20:
        cube = (cube_x / cube_n, cube_y / cube_n, area)
    return blobs, cube

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest_slot(p, slots, tol):
    best = None; bd = None
    for j, s in enumerate(slots):
        d = dist(p, s)
        if bd is None or d < bd:
            bd = d; best = j
    if bd is not None and bd <= tol:
        return best
    return None

def cube_to_target_px(cube):
    # a resting cube's image centroid sits above/inside the target centre (generic perspective offset)
    return (cube[0] + (128.0 - cube[0]) * 0.055, cube[1] + 5.0)

def reflect_resid(slot_i, slot_j, b, blobs):
    rx = slot_i[0] + slot_j[0] - b[0]
    ry = slot_i[1] + slot_j[1] - b[1]
    best = None
    for q in blobs:
        d = math.hypot(rx - q[0], ry - q[1])
        if best is None or d < best:
            best = d
    for s in (slot_i, slot_j):
        d = math.hypot(rx - s[0], ry - s[1])
        if best is None or d < best:
            best = d
    return best

def matchings(pairs, max_pairs):
    out = [[]]
    def extend(start, current, used):
        for n in range(start, len(pairs)):
            p = pairs[n]
            if p[0] in used or p[1] in used:
                continue
            nxt = current + [p]
            out.append(nxt)
            if len(nxt) < max_pairs:
                u2 = dict(used); u2[p[0]] = 1; u2[p[1]] = 1
                extend(n + 1, nxt, u2)
    extend(0, [], {})
    return out

SWAP_PARAMS = {"cand_f": 0.5, "match_w": 1.0, "unm_w": 1.0, "support": 0.0, "pair_cost": 4.0, "max_event_s": 2.5}

def analyse_video(frames, slots, unit, slot_tol=7.0, place_tol=10.0, min_rest_s=0.8):
    """frames: list of (t_seconds, blobs, cube). Returns events, placements, final permutation."""
    S = len(slots)
    perm = list(range(S))  # perm[slot] = identity currently at slot
    def off_items(blobs):
        return [b for b in blobs if nearest_slot(b, slots, slot_tol) is None and 0.25 * unit <= b[2] <= 2.6 * unit]
    active = [len(off_items(fr[1])) > 0 for fr in frames]
    runs = []
    k = 0
    while k < len(frames):
        if active[k]:
            s = k
            while k < len(frames) and active[k]:
                k += 1
            runs.append([s, k - 1])
        else:
            k += 1
    merged = []
    for r in runs:
        if merged and frames[r[0]][0] - frames[merged[-1][1]][0] <= 0.3:
            merged[-1][1] = r[1]
        else:
            merged.append([r[0], r[1]])
    # split long activity runs (chained / overlapping swaps, arm occlusion) at the quietest samples
    def quiet(t):
        n = 0
        for j in range(S):
            if not any(dist(b, slots[j]) <= slot_tol for b in frames[t][1]):
                n += 1
        return n + len(off_items(frames[t][1]))
    LMAX = SWAP_PARAMS["max_event_s"]
    work = list(merged)
    merged = []
    while work:
        s, e = work.pop(0)
        if LMAX <= 0 or frames[e][0] - frames[s][0] <= LMAX or e - s < 4:
            merged.append([s, e])
            continue
        best = None
        for t in range(s + 2, e - 1):
            q = quiet(t)
            if best is None or q < best[0]:
                best = (q, t)
        work.insert(0, [best[1] + 1, e])
        work.insert(0, [s, best[1] - 1])
    events = []
    for s, e in merged:
        nsamp = e - s + 1
        items = []
        vacant = [0] * S
        for t in range(s, e + 1):
            blobs = frames[t][1]
            for j in range(S):
                if not any(dist(b, slots[j]) <= slot_tol for b in blobs):
                    vacant[j] += 1
            for b in off_items(blobs):
                items.append((t, b))
        def vac_at(t, j):
            if t < 0 or t >= len(frames):
                return False
            return not any(dist(b, slots[j]) <= slot_tol for b in frames[t][1])
        frac = [vacant[j] / float(nsamp) for j in range(S)]
        # a slot vacant both just before and just after the event is statically occluded (cube/arm), not swapping
        static = [vac_at(s - 1, j) and vac_at(e + 1, j) for j in range(S)]
        PRM = SWAP_PARAMS
        cand = [j for j in range(S) if frac[j] >= PRM["cand_f"] and not static[j]]
        pairs = []
        resid = {}
        for x in range(len(cand)):
            for y in range(x + 1, len(cand)):
                i = cand[x]; j = cand[y]
                scale = 5.0 + 0.12 * dist(slots[i], slots[j])
                rl = [min(1.0, reflect_resid(slots[i], slots[j], b, frames[t][1]) / scale) for t, b in items]
                if sum(1 for v in rl if v < 0.6) < PRM["support"] * nsamp:
                    continue
                resid[(i, j)] = rl
                pairs.append((i, j))
        best = None
        for M in matchings(pairs, 4):
            matched = {}
            for p in M:
                matched[p[0]] = 1; matched[p[1]] = 1
            cost = PRM["pair_cost"] * len(M)
            for j in cand:
                if j in matched:
                    cost += PRM["match_w"] * (1.0 - frac[j]) * nsamp
                else:
                    cost += PRM["unm_w"] * frac[j] * nsamp
            for n in range(len(items)):
                v = 1.0
                for p in M:
                    if resid[p][n] < v:
                        v = resid[p][n]
                cost += v
            if best is None or cost < best[0] - 1e-9:
                best = (cost, M)
        M = best[1] if best else []
        for p in M:
            perm[p[0]], perm[p[1]] = perm[p[1]], perm[p[0]]
        events.append({"t0": frames[s][0], "t1": frames[e][0], "pairs": [list(p) for p in M],
                       "items": len(items), "cost": round(best[0], 2) if best else None,
                       "perm_after": list(perm), "end_index": e,
                       "frac": [round(v, 2) for v in frac]})
    # --- placements: cube at rest near a slot for >= min_rest_s ---
    placements = []
    anchor = None; anchor_t = None; current = None
    for idx, fr in enumerate(frames):
        cube = fr[2]
        if cube is None:
            anchor = None
            continue
        if anchor is None or dist(cube, anchor) > 1.5:
            anchor = cube; anchor_t = fr[0]
        if current is not None and dist(cube, current["cube"]) > 6.0:
            current = None
        if current is None and fr[0] - anchor_t >= min_rest_s:
            j = nearest_slot(cube_to_target_px(cube), slots, place_tol)
            if j is not None:
                ident = list(range(S))
                for ev in events:
                    if ev["end_index"] < idx:
                        ident = ev["perm_after"]
                current = {"t": fr[0], "slot": j, "identity": ident[j], "cube": cube}
                placements.append(current)
    return {"events": events, "placements": placements, "final_perm": list(perm)}

def choose_slot(analysis, k, last):
    pl = analysis["placements"]
    if not pl:
        return None, "no placements detected"
    if last or k is None:
        chosen = pl[-1]
    elif k <= len(pl):
        chosen = pl[k - 1]
    else:
        chosen = pl[-1]
    ident = chosen["identity"]
    perm = analysis["final_perm"]
    for s in range(len(perm)):
        if perm[s] == ident:
            return s, "ok"
    return chosen["slot"], "identity not found; using placement slot"
# ---- END CORE ----
# ---- MAIN (live program) ----
def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]

def cloud_center(points):
    return (quantile((p[0] for p in points), 0.5), quantile((p[1] for p in points), 0.5),
            quantile((p[2] for p in points), 0.5))

def box_of(candidate):
    try:
        b = candidate.box_xyxy
        if b is None:
            return None
        return [float(v) for v in b]
    except Exception:
        return None

def pick_candidate_at(camera, prompts, px, max_side, report, tag):
    """Smallest segment_text box containing the tracked pixel (localize.md Disambiguation)."""
    tries = []
    for prompt in prompts:
        found = segment_text(camera, prompt)
        if not found.ok:
            tries.append([prompt, "fail"])
            continue
        best = None; best_area = None; nbox = 0
        for cand in found.segmentations[:12]:
            b = box_of(cand)
            if b is None:
                continue
            nbox += 1
            if not (b[0] - 3 <= px[0] <= b[2] + 3 and b[1] - 3 <= px[1] <= b[3] + 3):
                continue
            if (b[2] - b[0]) > max_side or (b[3] - b[1]) > max_side:
                continue
            area = (b[2] - b[0]) * (b[3] - b[1])
            if best_area is None or area < best_area:
                best_area = area; best = cand
        tries.append([prompt, nbox, best_area])
        if best is not None:
            report[tag + "_seg"] = {"prompt": prompt, "box": box_of(best), "tries": tries}
            return best
    pts = segment_points(camera, [[int(round(px[0])), int(round(px[1]))]])
    if pts.ok and len(pts.segmentations) > 0:
        report[tag + "_seg"] = {"prompt": "points", "box": box_of(pts.segmentations[0]), "tries": tries}
        return pts.segmentations[0]
    report[tag + "_seg"] = {"prompt": None, "tries": tries}
    return None

def project(candidate, camera, base_frame):
    cloud = mask_to_point_cloud(candidate, camera, target_frame=base_frame)
    try:
        pts = [list(p) for p in cloud.points]
    except Exception:
        pts = []
    return pts

def top_face_xy(pts, band=0.012):
    zt = quantile((p[2] for p in pts), 0.95)
    top = [p for p in pts if p[2] >= zt - band]
    if len(top) < 5:
        top = pts
    return (quantile((p[0] for p in top), 0.5), quantile((p[1] for p in top), 0.5)), len(top)

def color_loose(p, color):
    r = p[0]; g = p[1]; b = p[2]
    if color == "red":
        return r > g + 50 and r > b + 50
    if color == "green":
        return g > r + 40 and g > b + 40
    if color == "blue":
        return b > r + 50 and b > g + 50
    return False

def wrist_refine(color, guess_xy, base_frame, report):
    info = {"cands": []}
    report["wrist_refine"] = info
    found = segment_text("wrist", color + " cube")
    if not found.ok:
        info["seg"] = "fail"
        return None
    obs = get_observation()
    rows = image_rows(obs.cameras["wrist"].rgb)
    best = None; bestd = None
    for cand in found.segmentations[:8]:
        b = box_of(cand)
        if b is None:
            continue
        cx = int(min(len(rows[0]) - 1, max(0, (b[0] + b[2]) / 2.0)))
        cy = int(min(len(rows) - 1, max(0, (b[1] + b[3]) / 2.0)))
        colok = color_loose(rows[cy][cx], color)
        pts = project(cand, "wrist", base_frame)
        if len(pts) < 10:
            info["cands"].append([rnd(b, 1), colok, len(pts)])
            continue
        xy, ntop = top_face_xy(pts)
        zt = quantile((p[2] for p in pts), 0.95)
        d = math.hypot(xy[0] - guess_xy[0], xy[1] - guess_xy[1])
        info["cands"].append([rnd(b, 1), colok, len(pts), rnd(xy), round(d, 4), round(zt, 4)])
        if not colok or d > 0.06:
            continue
        if bestd is None or d < bestd:
            bestd = d; best = (xy[0], xy[1], zt)
    info["chosen"] = rnd(best) if best else None
    return best

def derive_target_hue(rows, report):
    """Channel order of the discs the goal calls 'target': chromatic pixels inside the central 80% ellipse of
    every segment_text("front", "target") box scoring >= half the best score."""
    found = segment_text("front", "target")
    if not found.ok:
        return None
    scored = []
    for cand in found.segmentations[:12]:
        b = box_of(cand)
        if b is None:
            continue
        try:
            sc = float(cand.score)
        except Exception:
            sc = 1.0
        scored.append((sc, b))
    if not scored:
        return None
    smax = max(s for s, b in scored)
    h = len(rows); w = len(rows[0])
    sums = [0.0, 0.0, 0.0]; n = 0; used = 0
    for sc, b in scored:
        if sc < 0.5 * smax:
            continue
        used += 1
        cx = (b[0] + b[2]) / 2.0; cy = (b[1] + b[3]) / 2.0
        rx = max(1.0, (b[2] - b[0]) / 2.0 * 0.8); ry = max(1.0, (b[3] - b[1]) / 2.0 * 0.8)
        for y in range(max(0, int(cy - ry)), min(h, int(cy + ry) + 1)):
            for x in range(max(0, int(cx - rx)), min(w, int(cx + rx) + 1)):
                if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 > 1.0:
                    continue
                p = rows[y][x]
                if max(p) - min(p) < 40:
                    continue
                sums[0] += p[0]; sums[1] += p[1]; sums[2] += p[2]; n += 1
    if n < 30:
        report["target_hue"] = {"pixels": n, "boxes": used}
        return None
    mean = [v / n for v in sums]
    order = []
    for k in range(3):
        pos = 0
        while pos < len(order) and mean[order[pos]] >= mean[k]:
            pos += 1
        order.insert(pos, k)
    report["target_hue"] = {"pixels": n, "boxes": used, "mean": rnd(mean, 1), "order": order}
    if mean[order[0]] - mean[order[2]] < 30:
        return None
    return order

def rnd(v, n=4):
    try:
        return [round(float(x), n) for x in v]
    except Exception:
        return None

def hold_ticks(n, gripper):
    st = get_robot_state()
    last = None
    for tick in range(n):
        last = step(RobotAction(arms={"primary": ArmCommand(
            "joint_position", st.joint_positions["primary"],
            gripper_position=gripper, embodiment=st.embodiment)}))
        if not last.ok:
            return last
        try:
            if last.terminated or last.truncated:
                return last
        except Exception:
            pass
    return last

def ended(res):
    try:
        return bool(res.terminated) or bool(res.truncated)
    except Exception:
        return False

report = {"version": "v04"}
done = False
try:
    goal = get_task_context().language
    report["goal"] = goal
    color, k, last = parse_goal(goal)
    report["parsed"] = [color, k, last]
    state0 = get_robot_state()
    base_frame = state0.base_frame
    live = get_observation()
    live_rows = image_rows(live.cameras["front"].rgb)
    hue = derive_target_hue(live_rows, report)
    if hue is None:
        raise ValueError("could not derive the target colour class from segment_text('front', 'target')")
    TARGET_HUE.extend(hue)
    liveb, livec = scan_frame(live_rows, color)
    big = sorted(a for x, y, a in liveb if a >= 40)
    if not big:
        raise ValueError("no target discs visible in live front image")
    unit = big[len(big) // 2]
    slots = [(x, y) for x, y, a in sorted(liveb) if a >= 0.4 * unit]
    report["slots_px"] = [[round(x, 1), round(y, 1)] for x, y in slots]
    report["live_cube_px"] = rnd(livec, 1) if livec else None
    video = robomme_plus.get_conditioning_video()
    frames = []
    for fr in video:
        b, c = scan_frame(image_rows(fr.cameras["front"].rgb), color)
        frames.append((float(fr.timestamp_s), b, c))
    report["video_frames"] = len(frames)
    report["frames_dump"] = [[round(f[0], 3), [[round(q[0], 1), round(q[1], 1), int(q[2])] for q in f[1]],
                              ([round(f[2][0], 1), round(f[2][1], 1), int(f[2][2])] if f[2] else None)] for f in frames]
    report["live_blobs"] = [[round(q[0], 1), round(q[1], 1), int(q[2])] for q in liveb]
    an = analyse_video(frames, slots, unit)
    report["events"] = [[e["t0"], e["t1"], e["pairs"], e["items"]] for e in an["events"]]
    report["placements"] = [[p["t"], p["slot"], p["identity"]] for p in an["placements"]]
    report["final_perm"] = an["final_perm"]
    sidx, why = choose_slot(an, k, last)
    report["chosen_slot"] = sidx
    report["chosen_why"] = why
    if sidx is None:
        raise ValueError("could not determine target slot: " + why)
    tpx = slots[sidx]
    report["chosen_px"] = [round(tpx[0], 1), round(tpx[1], 1)]

    # --- localize target (chosen disc) and cube in 3D ---
    tcand = pick_candidate_at("front", ("target", "round target disc"), tpx, 60, report, "target")
    if tcand is None:
        raise ValueError("target segmentation failed")
    tpts = project(tcand, "front", base_frame)
    if len(tpts) < 10:
        raise ValueError("target cloud too small")
    tcenter = cloud_center(tpts)
    surface_z = quantile((p[2] for p in tpts), 0.9)
    report["target_xyz"] = rnd(tcenter); report["target_surface_z"] = round(surface_z, 4)
    if livec is None:
        raise ValueError("goal cube not visible in live front image")
    ccand = pick_candidate_at("front", (color + " cube", color + " block", "cube"), livec, 40, report, "cube")
    if ccand is None:
        raise ValueError("cube segmentation failed")
    cpts = project(ccand, "front", base_frame)
    if len(cpts) < 10:
        raise ValueError("cube cloud too small")
    ccenter = cloud_center(cpts)
    report["cube_xyz"] = rnd(ccenter)
    ztop = quantile((p[2] for p in cpts), 0.95)
    report["cube_z_q"] = rnd([quantile((p[2] for p in cpts), q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)])
    report["cube_xy_spread"] = rnd([quantile((p[0] for p in cpts), 0.05), quantile((p[0] for p in cpts), 0.95),
                                    quantile((p[1] for p in cpts), 0.05), quantile((p[1] for p in cpts), 0.95)])
    report["cube_points"] = len(cpts)

    # --- pick (grasp.md standard skeleton) ---
    down = state0.end_effector_poses["primary"].quaternion_wxyz
    report["ee_start"] = rnd(state0.end_effector_poses["primary"].position)
    # grasp across the cube's mid-height: top of the observed cloud minus a generic 2.5 cm, never below 1.5 cm
    gz = max(0.015, ztop - 0.025)
    r = open_gripper()
    if ended(r):
        done = True
    motions = []
    def go(name, xyz, tol=0.02, ms=250):
        res = move_to_pose(Pose((xyz[0], xyz[1], xyz[2]), down, base_frame), tolerance=tol, max_steps=ms)
        try:
            st = str(res.status)
        except Exception:
            st = "?"
        motions.append([name, bool(res.ok), st, rnd(xyz)])
        return res
    held = False
    for attempt in range(2):
        if done:
            break
        if attempt > 0:
            # re-localize after a failed grasp (the cube may have moved); fresh colour centroid -> box -> cloud
            r = open_gripper()
            done = ended(r)
            if done:
                break
            r = go("retry_up", (ccenter[0], ccenter[1], gz + 0.15))
            done = ended(r)
            if done:
                break
            nb, nc = scan_frame(image_rows(get_observation().cameras["front"].rgb), color)
            report["retry_cube_px"] = rnd(nc, 1) if nc else None
            if nc is None:
                break
            cc2 = pick_candidate_at("front", (color + " cube", color + " block", "cube"), nc, 40, report, "cube_retry")
            if cc2 is None:
                break
            cp2 = project(cc2, "front", base_frame)
            if len(cp2) < 10:
                break
            ccenter = cloud_center(cp2)
            gz = max(0.015, quantile((p[2] for p in cp2), 0.95) - 0.025)
            report["retry_cube_xyz"] = rnd(ccenter)
        r = go("pregrasp", (ccenter[0], ccenter[1], gz + 0.12))
        done = ended(r)
        if done:
            break
        try:
            wxy = wrist_refine(color, ccenter, base_frame, report)
        except Exception as exc:
            wxy = None
            report["wrist_refine_error"] = str(exc)
        if wxy is not None and abs(wxy[2] - (gz + 0.025)) < 0.03:
            # wrist depth (closer, top-down) refines only the grasp height; xy stays the front-cloud median
            gz = max(0.015, wxy[2] - 0.025)
        report["gz_" + str(attempt)] = round(gz, 4)
        r = go("grasp", (ccenter[0], ccenter[1], gz), 0.015)
        done = ended(r)
        if done:
            break
        r = close_gripper()
        done = ended(r)
        if done:
            break
        r = go("lift", (ccenter[0], ccenter[1], gz + 0.15))
        done = ended(r)
        if done:
            break
        g = float(get_robot_state().gripper_positions["primary"])
        report["gripper_after_lift_" + str(attempt)] = round(g, 3)
        if g > 0.08:
            held = True
            break
    report["held"] = held
    if not done:
        r = go("above_target", (tcenter[0], tcenter[1], gz + 0.15), 0.02, 300)
        done = ended(r)
    if not done:
        r = go("release", (tcenter[0], tcenter[1], max(surface_z + 0.03, gz + 0.012)))
        done = ended(r)
    if not done:
        r = open_gripper()
        done = ended(r)
    if not done:
        r = hold_ticks(5, 1.0)
        done = ended(r)
    if not done:
        r = go("retreat", (tcenter[0], tcenter[1], surface_z + 0.15))
        done = ended(r)
    report["motions"] = motions
    report["episode_ended"] = done
    if not done:
        try:
            fin = get_observation()
            fb, fc = scan_frame(image_rows(fin.cameras["front"].rgb), color)
            report["final_cube_px"] = rnd(fc, 1) if fc else None
            if fc:
                report["final_cube_slot"] = nearest_slot(cube_to_target_px(fc), slots, 12)
        except Exception as exc:
            report["final_obs_error"] = str(exc)
except Exception as exc:
    report["error"] = str(exc)
result = report
