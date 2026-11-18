import math

# VideoPlaceButton v04 — the video moves SEVERAL cubes: follow each through grasp/release events and
# pick the referenced release among the goal cube's own releases; >2 simultaneously emptied slots are
# paired by point symmetry of moving ring centroids about the pair midpoint (table XY via homography).
# v03 — swap detection by per-slot occupancy (ring gone + wood visible),
# pair = the two emptied slots; >2 simultaneous = NN ring tracking; pixel-only slots via homography.
# v02 — memory: targets SWAP positions during the conditioning video. Track the
# identity of the referenced target (slot at the referenced release) through every later swap
# (pink-ring blobs in video front RGB) and place on the slot it occupies at the end of the video.
# (v01: select the goal-colored cube via prompt + RGB check.)
# Reads the episode goal, summarizes the conditioning video from its recorded robot-state
# timeline (gripper close/open events = pick/place, empty-hand EE descents = button presses),
# chooses the placement referenced by the goal, then picks the live cube (front-camera SAM3 +
# depth) and places it on the live target nearest the referenced video placement.

CAM = "front"
report = {"version": "v04", "stage": "start", "errors": []}


def qtl(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def r3(v):
    return round(float(v), 3)


def note(msg):
    report["errors"].append(str(msg))


def exec_info(res):
    info = {}
    try:
        info["ok"] = bool(res.ok)
    except Exception:
        info["ok"] = None
    try:
        info["status"] = str(res.status)
    except Exception:
        pass
    try:
        info["steps"] = int(res.steps_executed)
    except Exception:
        pass
    try:
        info["terminated"] = bool(res.terminated)
        info["truncated"] = bool(res.truncated)
    except Exception:
        pass
    try:
        if res.error is not None:
            info["error"] = str(res.error)[:300]
    except Exception:
        pass
    return info


def ended(info):
    return bool(info.get("terminated")) or bool(info.get("truncated"))


ctx = get_task_context()
goal = str(ctx.language)
report["goal"] = goal
def plain(m):
    out = {}
    for k in m:
        v = m[k]
        if v is None or v is True or v is False:
            out[str(k)] = v
        else:
            try:
                out[str(k)] = float(v) if str(float(v)) == str(v) else str(v)[:800]
            except Exception:
                out[str(k)] = str(v)[:800]
    return out


try:
    report["task_metadata"] = plain(robomme_plus.get_task_metadata())
except Exception as exc:
    note("task_metadata: " + str(exc))
try:
    report["controller_metadata"] = plain(robomme_plus.get_controller_metadata())
except Exception as exc:
    note("controller_metadata: " + str(exc))

state0 = get_robot_state()
base = state0.base_frame
home_pose = state0.end_effector_poses["primary"]
down_q = [float(v) for v in home_pose.quaternion_wxyz]
report["home_ee"] = [r3(v) for v in home_pose.position]
report["home_q"] = [r3(v) for v in down_q]
report["gripper0"] = r3(state0.gripper_positions["primary"])
report["t0"] = r3(state0.timestamp_s)

# ---------------- conditioning video timeline ----------------
video = robomme_plus.get_conditioning_video()
frames = []
front_frames = []
for idx, frame in enumerate(video):
    front_frames.append(frame.cameras["front"].rgb)
    rs = frame.robot_state
    p = rs.end_effector_poses["primary"].position
    frames.append((float(frame.timestamp_s), float(p[0]), float(p[1]), float(p[2]),
                   float(rs.gripper_positions["primary"])))
report["video_frames"] = len(frames)
report["video_timeline"] = [[r3(f[0]), r3(f[1]), r3(f[2]), r3(f[3]), r3(f[4])] for f in frames]

events = []
if len(frames) > 2:
    closed = frames[0][4] < 0.6
    for i in range(1, len(frames)):
        g = frames[i][4]
        if (not closed) and g < 0.6:
            closed = True
            events.append({"kind": "close", "i": i, "t": r3(frames[i][0]),
                           "xyz": [r3(frames[i][1]), r3(frames[i][2]), r3(frames[i][3])], "g": r3(g)})
        elif closed and g > 0.8:
            closed = False
            events.append({"kind": "open", "i": i, "t": r3(frames[i][0]),
                           "xyz": [r3(frames[i][1]), r3(frames[i][2]), r3(frames[i][3])], "g": r3(g)})
    # EE z local minima (descents) with prominence
    zs = [f[3] for f in frames]
    n = len(zs)
    win = 6
    zmins = []
    for i in range(1, n - 1):
        lo = max(0, i - win)
        hi = min(n, i + win + 1)
        if zs[i] <= min(zs[lo:hi]) and zs[i] < zs[i - 1]:
            left_max = max(zs[max(0, i - 40):i + 1])
            right_max = max(zs[i:min(n, i + 41)])
            prom = min(left_max, right_max) - zs[i]
            if prom > 0.03:
                if len(zmins) > 0 and i - zmins[-1]["i"] <= win:
                    continue
                zmins.append({"kind": "zmin", "i": i, "t": r3(frames[i][0]),
                              "xyz": [r3(frames[i][1]), r3(frames[i][2]), r3(frames[i][3])],
                              "g": r3(frames[i][4]), "prom": r3(prom)})
    events = events + zmins
events = sorted(events, key=lambda e: e["i"])
report["video_events"] = events

# classify descents: those far (in frames) from any close/open are press candidates
grip_idx = [e["i"] for e in events if e["kind"] in ("close", "open")]
presses = []
for e in events:
    if e["kind"] != "zmin":
        continue
    near = False
    for gi in grip_idx:
        if abs(gi - e["i"]) <= 12:
            near = True
    if (not near) or e["g"] < 0.05:
        presses.append(e)
releases = []
for e in events:
    if e["kind"] != "open":
        continue
    held = [f[4] for f in frames[max(0, e["i"] - 8):max(0, e["i"] - 2)]]
    e["held_g"] = r3(qtl(held, 0.5)) if len(held) > 0 else None
    if e["held_g"] is None or e["held_g"] > 0.05:
        releases.append(e)
grasps = [e for e in events if e["kind"] == "close"]
report["video_presses"] = presses
report["video_releases"] = releases

low = goal.lower()
mode = "before"
if "after" in low:
    mode = "after"
if "before" in low:
    mode = "before"
report["goal_mode"] = mode

chosen = None
if len(presses) > 0 and len(releases) > 0:
    press_i = presses[0]["i"]
    if mode == "before":
        for e in releases:
            if e["i"] < press_i:
                chosen = e
    else:
        for e in releases:
            if e["i"] > press_i and chosen is None:
                chosen = e
if chosen is None and len(releases) > 0:
    chosen = releases[-1]
    note("fallback: last video release")
report["chosen_release"] = chosen

# ---------------- live perception ----------------
def candidates(prompt, max_n=5):
    out = []
    seg = segment_text(CAM, prompt)
    if not seg.ok:
        return out
    for cand in seg.segmentations[:max_n]:
        item = {"prompt": prompt, "score": r3(cand.score), "cand": cand}
        try:
            item["box"] = [r3(v) for v in cand.box_xyxy]
        except Exception:
            item["box"] = None
        try:
            pc = mask_to_point_cloud(cand, CAM, target_frame=base)
            pts = pc.points
            item["n"] = len(pts)
            if len(pts) >= 15:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                zz = [p[2] for p in pts]
                item["c"] = [r3(qtl(xs, 0.5)), r3(qtl(ys, 0.5)), r3(qtl(zz, 0.5))]
                item["zlo"] = r3(qtl(zz, 0.05))
                item["zhi"] = r3(qtl(zz, 0.95))
                item["sx"] = r3(qtl(xs, 0.95) - qtl(xs, 0.05))
                item["sy"] = r3(qtl(ys, 0.95) - qtl(ys, 0.05))
        except Exception as exc:
            item["pc_error"] = str(exc)[:200]
        out.append(item)
    return out


def public(items):
    res = []
    for it in items:
        d = {}
        for k in it:
            if k != "cand":
                d[k] = it[k]
        res.append(d)
    return res


probe = {}
for prompt in ("cube", "red cube", "target", "round target", "button", "disc"):
    try:
        probe[prompt] = candidates(prompt)
    except Exception as exc:
        probe[prompt] = []
        note("seg " + prompt + ": " + str(exc)[:200])
report["probe"] = {}
for k in probe:
    report["probe"][k] = public(probe[k])

# table height estimate: low quantile of all candidate bottoms is noisy; use cube bottom.
def small_object(it, lo, hi):
    if "c" not in it:
        return False
    return lo <= it["sx"] <= hi and lo <= it["sy"] <= hi and it["c"][2] > -0.05 and it["c"][2] < 0.25


COLORS = {"red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0),
          "yellow": (0.5, 0.5, 0.0), "purple": (0.5, 0.0, 0.5), "orange": (0.7, 0.3, 0.0),
          "cyan": (0.0, 0.5, 0.5), "pink": (0.55, 0.15, 0.3)}
words = low.replace(",", " ").replace(".", " ").split()
cube_color = None
for wi in range(len(words) - 1):
    if words[wi] in COLORS and words[wi + 1] in ("cube", "block"):
        cube_color = words[wi]
        break
report["cube_color"] = cube_color

live_obs = get_observation()
front_rgb = live_obs.cameras[CAM].rgb


def box_rgb(box):
    # mean RGB of a 5x5 patch at the box center (cube top/side face)
    cx = int((box[0] + box[2]) / 2.0)
    cy = int((box[1] + box[3]) / 2.0)
    acc = [0.0, 0.0, 0.0]
    cnt = 0
    for yy in range(cy - 2, cy + 3):
        for xx in range(cx - 2, cx + 3):
            if 0 <= yy < 256 and 0 <= xx < 256:
                px = front_rgb[yy][xx]
                acc[0] += float(px[0])
                acc[1] += float(px[1])
                acc[2] += float(px[2])
                cnt += 1
    if cnt == 0:
        return None
    return [acc[0] / cnt, acc[1] / cnt, acc[2] / cnt]


def chroma_dist(rgb, name):
    tot = rgb[0] + rgb[1] + rgb[2] + 1e-6
    ref = COLORS[name]
    rt = ref[0] + ref[1] + ref[2]
    d = 0.0
    for k in range(3):
        d += (rgb[k] / tot - ref[k] / rt) ** 2
    return math.sqrt(d)


cube_pool = []
cube_prompts = ("cube",)
if cube_color is not None:
    cube_prompts = (cube_color + " cube", "cube")
for prompt in cube_prompts:
    if prompt not in probe:
        try:
            probe[prompt] = candidates(prompt)
        except Exception as exc:
            probe[prompt] = []
            note("seg " + prompt + ": " + str(exc)[:200])
    for it in probe.get(prompt, []):
        if small_object(it, 0.015, 0.07) and 0.015 < (it["zhi"] - it["zlo"]) < 0.08 and it.get("box") is not None:
            dup = False
            for c2 in cube_pool:
                if abs(c2["c"][0] - it["c"][0]) < 0.02 and abs(c2["c"][1] - it["c"][1]) < 0.02:
                    dup = True
            if not dup:
                it["rgb"] = box_rgb(it["box"])
                if cube_color is not None and it["rgb"] is not None:
                    it["cdist"] = r3(chroma_dist(it["rgb"], cube_color))
                it["rgb"] = [r3(v) for v in it["rgb"]] if it["rgb"] is not None else None
                cube_pool.append(it)
report["cube_pool"] = public(cube_pool)
cube = None
for it in cube_pool:
    if cube_color is not None:
        if it.get("cdist") is None or it["cdist"] > 0.35:
            continue
        if cube is None or it["cdist"] < cube["cdist"]:
            cube = it
    elif cube is None or it["score"] > cube["score"]:
        cube = it
if cube is None and len(cube_pool) > 0 and cube_color is None:
    cube = cube_pool[0]
report["cube"] = public([cube])[0] if cube is not None else None

# targets: flat things
targets = []
for prompt in ("target", "round target", "disc"):
    for it in probe.get(prompt, []):
        if "c" in it and (it["zhi"] - it["zlo"]) < 0.02 and 0.02 <= it["sx"] <= 0.2 and 0.02 <= it["sy"] <= 0.2:
            dup = False
            for t in targets:
                if abs(t["c"][0] - it["c"][0]) < 0.03 and abs(t["c"][1] - it["c"][1]) < 0.03:
                    dup = True
            if not dup:
                targets.append(it)
report["targets"] = public(targets)

def nearest_slot(slots, x, y):
    best = None
    best_d = None
    for si in range(len(slots)):
        d = ((slots[si][0] - x) ** 2 + (slots[si][1] - y) ** 2) ** 0.5
        if best_d is None or d < best_d:
            best_d = d
            best = si
    return best, best_d


# ---------------- which video cube is the goal cube (v04) ----------------
# The video moves several cubes. Follow each cube through grasp (close holding something) and
# release events; the goal cube is the video cube whose end-of-video position matches the live
# goal-colored cube; the referenced placement is chosen among that cube's releases only.
vid_cubes = []
holding_id = None
cube_releases = []
for e in events:
    if e["kind"] == "close":
        post = [f[4] for f in frames[e["i"] + 2:min(len(frames), e["i"] + 9)]]
        post_g = qtl(post, 0.5) if len(post) > 0 else None
        if post_g is None or post_g <= 0.05:
            continue
        best = None
        best_d = None
        for ci in range(len(vid_cubes)):
            d = math.hypot(vid_cubes[ci]["xy"][0] - e["xyz"][0], vid_cubes[ci]["xy"][1] - e["xyz"][1])
            if best_d is None or d < best_d:
                best_d = d
                best = ci
        if best is None or best_d > 0.05:
            vid_cubes.append({"start": [e["xyz"][0], e["xyz"][1]], "xy": [e["xyz"][0], e["xyz"][1]], "grasps": [], "releases": []})
            best = len(vid_cubes) - 1
        vid_cubes[best]["grasps"].append(e["i"])
        holding_id = best
    elif e["kind"] == "open" and e in releases:
        if holding_id is not None:
            vid_cubes[holding_id]["xy"] = [e["xyz"][0], e["xyz"][1]]
            vid_cubes[holding_id]["releases"].append(e["i"])
            e["cube_id"] = holding_id
        holding_id = None
report["video_cubes"] = vid_cubes
goal_cube_id = None
if cube is not None and len(vid_cubes) > 0:
    best_d = None
    for ci in range(len(vid_cubes)):
        d = math.hypot(vid_cubes[ci]["xy"][0] - cube["c"][0], vid_cubes[ci]["xy"][1] - cube["c"][1])
        if best_d is None or d < best_d:
            best_d = d
            goal_cube_id = ci
    report["goal_cube_match_d"] = r3(best_d)
    if best_d > 0.06:
        note("live goal cube does not match any video cube end position")
        goal_cube_id = None
report["goal_cube_id"] = goal_cube_id
if goal_cube_id is not None and len(presses) > 0:
    own = [e for e in releases if e.get("cube_id") == goal_cube_id]
    press_i = presses[0]["i"]
    pick = None
    if mode == "before":
        for e in own:
            if e["i"] < press_i:
                pick = e
    else:
        for e in own:
            if e["i"] > press_i and pick is None:
                pick = e
    if pick is not None:
        if chosen is None or pick["i"] != chosen["i"]:
            report["chosen_release_any_cube"] = chosen
        chosen = pick
    else:
        note("goal cube has no release on the referenced side of the press; keeping any-cube choice")
report["chosen_release"] = chosen

# ---------------- target identity through video swaps (v03: occupancy + tracking) ----------------
def is_pink(px):
    r = int(px[0])
    if r <= 170:
        return False
    g = int(px[1])
    b = int(px[2])
    return b > 120 and b > g + 15


def is_wood(px):
    r = int(px[0])
    g = int(px[1])
    b = int(px[2])
    return r > g + 25 and g > b + 5 and r < 240


def disk_fracs(rgb, x, y, rad):
    h = len(rgb)
    w = len(rgb[0])
    xi = int(round(x))
    yi = int(round(y))
    p = 0
    wd = 0
    n = 0
    for dy in range(-rad, rad + 1):
        yy = yi + dy
        if yy < 0 or yy >= h:
            continue
        row = rgb[yy]
        for dx in range(-rad, rad + 1):
            if dx * dx + dy * dy > rad * rad:
                continue
            xx = xi + dx
            if xx < 0 or xx >= w:
                continue
            px = row[xx]
            n += 1
            if is_pink(px):
                p += 1
            elif is_wood(px):
                wd += 1
    if n == 0:
        return 0.0, 0.0
    return p / n, wd / n


def ring_blobs(rgb, x0, y0, x1, y1, min_n=8):
    # full-resolution, 4-connected pink components; components with nearby centroids
    # (outer ring + inner dot of one target) are merged.
    pts = {}
    for yy in range(y0, y1 + 1):
        row = rgb[yy]
        for xx in range(x0, x1 + 1):
            if is_pink(row[xx]):
                pts[(yy, xx)] = True
    seen = {}
    comps = []
    for key in pts:
        if key in seen:
            continue
        stack = [key]
        seen[key] = True
        sx = 0.0
        sy = 0.0
        n = 0
        while len(stack) > 0:
            k = stack.pop()
            sx += k[1]
            sy += k[0]
            n += 1
            for nb in ((k[0] + 1, k[1]), (k[0] - 1, k[1]), (k[0], k[1] + 1), (k[0], k[1] - 1)):
                if nb in pts and nb not in seen:
                    seen[nb] = True
                    stack.append(nb)
        if n >= 3:
            comps.append([sx / n, sy / n, n])
    merged = []
    for c in sorted(comps, key=lambda q: -q[2]):
        hit = None
        for m in merged:
            if math.hypot(m[0] - c[0], m[1] - c[1]) < 4.0:
                hit = m
        if hit is None:
            merged.append([c[0], c[1], c[2]])
        else:
            tot = hit[2] + c[2]
            hit[0] = (hit[0] * hit[2] + c[0] * c[2]) / tot
            hit[1] = (hit[1] * hit[2] + c[1] * c[2]) / tot
            hit[2] = tot
    return [m for m in merged if m[2] >= min_n]


def occupancy_episodes(frames_rgb, slot_px, rad=4, gap=2):
    """Per frame, slots whose ring is gone (no pink, wood visible) = the swapping targets left.
    Returns episodes {start, end, members} (frame indices relative to frames_rgb)."""
    empties = []
    for idx in range(len(frames_rgb)):
        es = []
        for si in range(len(slot_px)):
            pf, wf = disk_fracs(frames_rgb[idx], slot_px[si][0], slot_px[si][1], rad)
            if pf < 0.08 and wf > 0.4:
                es.append(si)
        empties.append(es)
    episodes = []
    cur = None
    for idx in range(len(empties)):
        if len(empties[idx]) > 0:
            if cur is not None and idx - cur["end"] <= gap + 1:
                cur["end"] = idx
            else:
                cur = {"start": idx, "end": idx, "members": []}
                episodes.append(cur)
            for si in empties[idx]:
                if si not in cur["members"]:
                    cur["members"].append(si)
    for ep in episodes:
        ep["members"] = sorted(ep["members"])
    return episodes


def track_episode(frames_rgb, slot_px, members, s, e, margin=45, max_jump=18.0, lead=5, end_pool=None):
    """Nearest-neighbour tracking of the member rings from their slots (frame s-lead, before the
    rings visibly leave) through frame e+2. Returns {start_slot: end_slot} for members."""
    xs = [p[0] for p in slot_px]
    ys = [p[1] for p in slot_px]
    h = len(frames_rgb[0])
    w = len(frames_rgb[0][0])
    x0 = max(0, int(min(xs)) - margin)
    x1 = min(w - 1, int(max(xs)) + margin)
    y0 = max(0, int(min(ys)) - margin)
    y1 = min(h - 1, int(max(ys)) + margin)
    pos = {}
    vel = {}
    for m in members:
        pos[m] = (slot_px[m][0], slot_px[m][1])
        vel[m] = (0.0, 0.0)
    others = [si for si in range(len(slot_px)) if si not in members]
    for idx in range(max(0, s - lead), min(len(frames_rgb), e + 3)):
        bl = ring_blobs(frames_rgb[idx], x0, y0, x1, y1)
        pairs = []
        for m in members:
            px = pos[m][0] + vel[m][0]
            py = pos[m][1] + vel[m][1]
            for bi in range(len(bl)):
                d = math.hypot(bl[bi][0] - px, bl[bi][1] - py)
                if d < max_jump:
                    pairs.append((d, m, bi))
        pairs = sorted(pairs)
        used_m = {}
        used_b = {}
        newpos = {}
        for d, m, bi in pairs:
            if m in used_m or bi in used_b:
                continue
            used_m[m] = True
            used_b[bi] = True
            bx = bl[bi][0]
            by = bl[bi][1]
            near_static = False
            for o in others:
                if math.hypot(slot_px[o][0] - bx, slot_px[o][1] - by) < 9.0:
                    near_static = True
            if near_static:
                newpos[m] = (pos[m][0] + vel[m][0], pos[m][1] + vel[m][1])
            else:
                newpos[m] = (bx, by)
        for m in members:
            if m not in newpos:
                newpos[m] = (pos[m][0] + vel[m][0], pos[m][1] + vel[m][1])
            nv = (newpos[m][0] - pos[m][0], newpos[m][1] - pos[m][1])
            vel[m] = (0.5 * vel[m][0] + 0.5 * nv[0], 0.5 * vel[m][1] + 0.5 * nv[1])
            pos[m] = newpos[m]
    mapping = {}
    pool = members if end_pool is None else end_pool
    for m in members:
        best = None
        best_d = None
        for t in pool:
            d = math.hypot(slot_px[t][0] - pos[m][0], slot_px[t][1] - pos[m][1])
            if best_d is None or d < best_d:
                best_d = d
                best = t
        mapping[m] = best
    return mapping


def solve_linear(a, b):
    n = len(b)
    m = [list(a[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        piv = c
        for r in range(c + 1, n):
            if abs(m[r][c]) > abs(m[piv][c]):
                piv = r
        if abs(m[piv][c]) < 1e-12:
            return None
        tmp = m[c]
        m[c] = m[piv]
        m[piv] = tmp
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def fit_homography(px_list, xy_list):
    # least squares DLT with h33 = 1 (normal equations); table plane pixel -> base XY
    rows = []
    rhs = []
    for (u, v), (x, y) in zip(px_list, xy_list):
        rows.append([u, v, 1.0, 0.0, 0.0, 0.0, -u * x, -v * x])
        rhs.append(x)
        rows.append([0.0, 0.0, 0.0, u, v, 1.0, -u * y, -v * y])
        rhs.append(y)
    ata = [[sum(rows[k][i] * rows[k][j] for k in range(len(rows))) for j in range(8)] for i in range(8)]
    atb = [sum(rows[k][i] * rhs[k] for k in range(len(rows))) for i in range(8)]
    return solve_linear(ata, atb)


def apply_h(hm, u, v):
    den = hm[6] * u + hm[7] * v + 1.0
    return ((hm[0] * u + hm[1] * v + hm[2]) / den, (hm[3] * u + hm[4] * v + hm[5]) / den)


def to_xy(u, v):
    if hom is not None:
        return apply_h(hom, u, v)
    return (u, v)


def pair_scores(frames_rgb, slot_px, cand_pairs, s, e, tol):
    """Swapping rings move point-symmetrically about their pair midpoint. For each candidate pair,
    count frames (s-2..e+1) with two moving ring centroids (away from every slot) whose midpoint
    lies within tol of the pair midpoint (table XY when a homography exists)."""
    h = len(frames_rgb[0])
    w = len(frames_rgb[0][0])
    sxy = [to_xy(p[0], p[1]) for p in slot_px]
    cnt = {}
    for pr in cand_pairs:
        cnt[pr] = 0
    for idx in range(max(0, s - 2), min(len(frames_rgb), e + 2)):
        mv = []
        for bl in ring_blobs(frames_rgb[idx], 4, 4, w - 5, h - 5, 15):
            dmin = None
            for q in slot_px:
                d = math.hypot(bl[0] - q[0], bl[1] - q[1])
                if dmin is None or d < dmin:
                    dmin = d
            if dmin is not None and dmin > 6.0:
                mv.append(to_xy(bl[0], bl[1]))
        for pr in cand_pairs:
            mx = (sxy[pr[0]][0] + sxy[pr[1]][0]) / 2.0
            my = (sxy[pr[0]][1] + sxy[pr[1]][1]) / 2.0
            ok = False
            for i in range(len(mv)):
                for j in range(i + 1, len(mv)):
                    if math.hypot((mv[i][0] + mv[j][0]) / 2.0 - mx, (mv[i][1] + mv[j][1]) / 2.0 - my) < tol:
                        ok = True
            if ok:
                cnt[pr] += 1
    return cnt


def best_matching(items, cnt):
    # max-score set of disjoint pairs (at most one item left unmatched when the count is odd)
    if len(items) < 2:
        return 0, []
    first = items[0]
    rest = items[1:]
    best_sc = None
    best_m = []
    if len(items) % 2 == 1:
        sc, m = best_matching(rest, cnt)
        best_sc = sc
        best_m = m
    for k in range(len(rest)):
        pr = (first, rest[k]) if first < rest[k] else (rest[k], first)
        sc, m = best_matching(rest[:k] + rest[k + 1:], cnt)
        sc += cnt.get(pr, 0)
        if best_sc is None or sc > best_sc:
            best_sc = sc
            best_m = [pr] + m
    return best_sc, best_m


slots = []
for t in targets:
    if t.get("box") is None:
        continue
    slots.append({"px": ((t["box"][0] + t["box"][2]) / 2.0, (t["box"][1] + t["box"][3]) / 2.0),
                  "c": t["c"], "src": "sam"})
# add resting rings visible in the last video frame that SAM did not return (pixel only)
last_rgb = front_frames[-1] if len(front_frames) > 0 else None
if last_rgb is not None:
    try:
        hh = len(last_rgb)
        ww = len(last_rgb[0])
        for bl in ring_blobs(last_rgb, 4, 4, ww - 5, hh - 5, 15):
            dmin = None
            for sl in slots:
                d = math.hypot(sl["px"][0] - bl[0], sl["px"][1] - bl[1])
                if dmin is None or d < dmin:
                    dmin = d
            if dmin is None or dmin > 12.0:
                pf4, wf4 = disk_fracs(last_rgb, bl[0], bl[1], 4)
                if pf4 >= 0.3:
                    slots.append({"px": (bl[0], bl[1]), "c": None, "src": "pixel"})
                else:
                    report.setdefault("rejected_pixel_slots", []).append([r3(bl[0]), r3(bl[1]), bl[2], r3(pf4)])
    except Exception as exc:
        note("last-frame blobs: " + str(exc)[:200])
    # keep only slots that hold a pink ring at the end of the video (drops the button etc.)
    kept = []
    for sl in slots:
        pf, wf = disk_fracs(last_rgb, sl["px"][0], sl["px"][1], 6)
        sl["pink_end"] = r3(pf)
        if pf >= 0.08:
            kept.append(sl)
        else:
            report.setdefault("dropped_slots", []).append({"px": [r3(sl["px"][0]), r3(sl["px"][1])], "c": sl["c"], "pink": r3(pf)})
    slots = kept
# pixel -> table XY homography from SAM-localized rings (for pixel-only slots)
hom = None
sam_sl = [sl for sl in slots if sl["c"] is not None]
if len(sam_sl) >= 4:
    try:
        hom = fit_homography([sl["px"] for sl in sam_sl], [(sl["c"][0], sl["c"][1]) for sl in sam_sl])
        if hom is not None:
            res_err = []
            for sl in sam_sl:
                q = apply_h(hom, sl["px"][0], sl["px"][1])
                res_err.append(r3(math.hypot(q[0] - sl["c"][0], q[1] - sl["c"][1])))
            report["hom_residuals"] = res_err
    except Exception as exc:
        note("homography: " + str(exc)[:200])
for sl in slots:
    if sl["c"] is None:
        hq = apply_h(hom, sl["px"][0], sl["px"][1]) if hom is not None else None
        sq = None
        try:
            sp = segment_points(CAM, [[int(round(sl["px"][0])), int(round(sl["px"][1]))]])
            if sp.ok and len(sp.segmentations) > 0:
                pc = mask_to_point_cloud(sp.segmentations[0], CAM, target_frame=base)
                if len(pc.points) >= 10:
                    sq = (qtl([q[0] for q in pc.points], 0.5), qtl([q[1] for q in pc.points], 0.5),
                          qtl([q[2] for q in pc.points], 0.5))
        except Exception as exc:
            note("segment_points: " + str(exc)[:200])
        sl["h_xy"] = [r3(hq[0]), r3(hq[1])] if hq is not None else None
        sl["sp_c"] = [r3(v) for v in sq] if sq is not None else None
        if sq is not None and sq[2] < 0.03 and (hq is None or math.hypot(sq[0] - hq[0], sq[1] - hq[1]) < 0.04):
            sl["c"] = [r3(sq[0]), r3(sq[1]), r3(sq[2])]
        elif hq is not None:
            sl["c"] = [r3(hq[0]), r3(hq[1]), 0.009]
report["slots"] = [{"px": [r3(sl["px"][0]), r3(sl["px"][1])], "c": sl["c"], "src": sl["src"],
                    "pink_end": sl.get("pink_end"), "h_xy": sl.get("h_xy"), "sp_c": sl.get("sp_c")} for sl in slots]

goal_xy = None
ref_slot = None
if chosen is not None and len(slots) > 0:
    best_d = None
    for si in range(len(slots)):
        if slots[si]["c"] is None:
            continue
        d = math.hypot(slots[si]["c"][0] - chosen["xyz"][0], slots[si]["c"][1] - chosen["xyz"][1])
        if best_d is None or d < best_d:
            best_d = d
            ref_slot = si
    report["ref_slot"] = ref_slot
    report["ref_slot_d"] = r3(best_d) if best_d is not None else None
    if best_d is None or best_d > 0.06:
        ref_slot = None
if ref_slot is not None:
    cur = ref_slot
    swap_log = []
    try:
        start_i = chosen["i"] + 1
        seg_frames = front_frames[start_i:]
        slot_px = [sl["px"] for sl in slots]
        eps = occupancy_episodes(seg_frames, slot_px)
        for ep in eps:
            mem = ep["members"]
            item = {"start": ep["start"] + start_i, "end": ep["end"] + start_i, "members": mem}
            if len(mem) == 2:
                mp = {mem[0]: mem[1], mem[1]: mem[0]}
                item["how"] = "pair"
            elif len(mem) >= 3:
                tol = 0.015 if hom is not None else 5.0
                cps = []
                for a in range(len(mem)):
                    for b in range(a + 1, len(mem)):
                        cps.append((mem[a], mem[b]))
                cnt = pair_scores(seg_frames, slot_px, cps, ep["start"], ep["end"], tol)
                sc, matching = best_matching(mem, cnt)
                mp = {}
                for pr in matching:
                    mp[pr[0]] = pr[1]
                    mp[pr[1]] = pr[0]
                item["how"] = "symmetry"
                item["scores"] = [[pr[0], pr[1], cnt[pr]] for pr in cps]
                item["matching"] = [list(pr) for pr in matching]
            else:
                # one emptied slot: its partner may be occluded (arm/cube) -> best symmetric partner
                m0 = mem[0]
                tol = 0.015 if hom is not None else 5.0
                cps = []
                for o in range(len(slot_px)):
                    if o != m0:
                        cps.append((m0, o) if m0 < o else (o, m0))
                cnt = pair_scores(seg_frames, slot_px, cps, ep["start"], ep["end"], tol)
                bestp = None
                for pr in cps:
                    if bestp is None or cnt[pr] > cnt[bestp]:
                        bestp = pr
                item["scores"] = [[pr[0], pr[1], cnt[pr]] for pr in cps]
                if bestp is not None and cnt[bestp] >= 3:
                    mp = {bestp[0]: bestp[1], bestp[1]: bestp[0]}
                    item["how"] = "single_symmetry_pair"
                else:
                    mp = {}
                    item["how"] = "ignored_single"
            if cur in mp:
                cur = mp[cur]
            item["cur"] = cur
            swap_log.append(item)
    except Exception as exc:
        note("swap tracking: " + str(exc)[:300])
    report["swap_log"] = swap_log
    report["final_slot"] = cur
    if slots[cur]["c"] is not None:
        goal_xy = (slots[cur]["c"][0], slots[cur]["c"][1])
if goal_xy is None and chosen is not None:
    goal_xy = (chosen["xyz"][0], chosen["xyz"][1])
    note("fallback: raw video release xy")
report["goal_xy"] = [r3(v) for v in goal_xy] if goal_xy is not None else None


def live_extras(tag):
    # evidence: are targets static during execution? (rings vs slots)
    try:
        rgb = get_observation().cameras[CAM].rgb
        out = []
        for bl in ring_blobs(rgb, 4, 4, len(rgb[0]) - 5, len(rgb) - 5, 15):
            si, d = nearest_slot([sl["px"] for sl in slots], bl[0], bl[1])
            out.append([r3(bl[0]), r3(bl[1]), bl[2], si, r3(d)])
        report["live_blobs_" + tag] = out
    except Exception as exc:
        note("live_extras " + tag + ": " + str(exc)[:200])


live_extras("start")


# ---------------- act ----------------
report["moves"] = []


def go(xyz, tol=0.012, steps=150, label=""):
    pose = Pose((float(xyz[0]), float(xyz[1]), float(xyz[2])), down_q, base)
    res = move_to_pose(pose, tolerance=tol, max_steps=steps)
    info = exec_info(res)
    info["label"] = label
    info["target"] = [r3(v) for v in xyz]
    try:
        info["reached"] = [r3(v) for v in get_robot_state().end_effector_poses["primary"].position]
    except Exception:
        pass
    report["moves"].append(info)
    return info


done = False
try:
    if cube is None:
        raise ValueError("cube not found")
    if goal_xy is None:
        raise ValueError("no referenced placement from video")
    report["stage"] = "pick"
    cz_lo = cube["zlo"]
    cz_hi = cube["zhi"]
    h = max(0.02, cz_hi - cz_lo)
    grasp_z = cz_lo + 0.45 * h
    if len(grasps) > 0:
        vz = grasps[0]["xyz"][2]
        report["video_grasp_z"] = vz
        if cz_lo + 0.005 <= vz <= cz_hi:
            grasp_z = vz
    grasp_z = max(grasp_z, cz_lo + 0.012)
    report["grasp_z"] = r3(grasp_z)
    cx, cy = cube["c"][0], cube["c"][1]
    holding = 0.1 < report["gripper0"] < 0.8
    report["holding_at_start"] = holding
    if not holding:
        info = exec_info(open_gripper())
        report["moves"].append(dict(info, label="open"))
        if ended(info):
            raise ValueError("ended")
        info = go((cx, cy, grasp_z + 0.12), 0.015, 150, "pregrasp")
        if ended(info):
            raise ValueError("ended")
        info = go((cx, cy, grasp_z), 0.008, 120, "grasp")
        if ended(info):
            raise ValueError("ended")
        info = exec_info(close_gripper())
        report["moves"].append(dict(info, label="close"))
        report["grip_after_close"] = r3(get_robot_state().gripper_positions["primary"])
        if ended(info):
            raise ValueError("ended")
        info = go((cx, cy, grasp_z + 0.12), 0.015, 150, "lift")
        if ended(info):
            raise ValueError("ended")
        report["grip_after_lift"] = r3(get_robot_state().gripper_positions["primary"])
    live_extras("lifted")
    report["stage"] = "place"
    place_z = cz_lo + h + 0.005 + (grasp_z - cz_lo)
    if chosen is not None:
        vz = chosen["xyz"][2]
        if cz_lo <= vz <= cz_lo + 0.12:
            place_z = vz + 0.005
    report["place_z"] = r3(place_z)
    info = go((goal_xy[0], goal_xy[1], place_z + 0.10), 0.015, 180, "above_target")
    if ended(info):
        raise ValueError("ended")
    live_extras("above")
    info = go((goal_xy[0], goal_xy[1], place_z), 0.01, 120, "place")
    if ended(info):
        raise ValueError("ended")
    info = exec_info(open_gripper())
    report["moves"].append(dict(info, label="release"))
    if ended(info):
        raise ValueError("ended")
    info = go((goal_xy[0], goal_xy[1], place_z + 0.10), 0.02, 120, "retreat")
    if ended(info):
        raise ValueError("ended")
    # settle a few ticks
    st = get_robot_state()
    for tick in range(10):
        rep = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=st.joint_positions["primary"],
            gripper_position=1.0, embodiment=st.embodiment)}))
        try:
            if rep.terminated or rep.truncated:
                report["settle_end"] = {"tick": tick, "terminated": bool(rep.terminated),
                                        "truncated": bool(rep.truncated)}
                break
        except Exception:
            pass
    done = True
except Exception as exc:
    note("act: " + str(exc)[:300])

report["done"] = done
try:
    report["t_end"] = r3(get_robot_state().timestamp_s)
except Exception:
    pass
result = report
