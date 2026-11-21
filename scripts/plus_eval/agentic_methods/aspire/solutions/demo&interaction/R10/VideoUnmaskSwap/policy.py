# VideoUnmaskSwap v01 -- track containers through the conditioning video, then lift in goal order.
import math

VERSION = 'v01'
GATE = 11.0
VIS_FRAC = 0.5
SETTLE_MAX = 150
SETTLE_N = 15
GRASP_DEPTH = 0.02
LIFT = 0.12

# ---- tracker core (sandbox-compatible: no imports except math, no underscores) ----
# Images are nested lists rgb[y][x] = [r, g, b] (from ndarray.tolist()).

STRIDE = 2


def is_white(px):
    r = px[0]
    g = px[1]
    b = px[2]
    mn = min(r, g, b)
    mx = max(r, g, b)
    return mn >= 150 and (mx - mn) <= 40


def white_points(rgb, static, stride):
    pts = []
    h = len(rgb)
    w = len(rgb[0])
    y = 0
    while y < h:
        row = rgb[y]
        srow = static[y // stride]
        x = 0
        while x < w:
            if not srow[x // stride]:
                px = row[x]
                if px[0] >= 150 and px[1] >= 150 and px[2] >= 150:
                    mn = min(px[0], px[1], px[2])
                    mx = max(px[0], px[1], px[2])
                    if mx - mn <= 40:
                        pts.append((x, y))
            x += stride
        y += stride
    return pts


def static_mask(frames, stride, frac):
    h = len(frames[0])
    w = len(frames[0][0])
    gh = (h + stride - 1) // stride
    gw = (w + stride - 1) // stride
    counts = [[0] * gw for i in range(gh)]
    for rgb in frames:
        for gy in range(gh):
            row = rgb[gy * stride]
            crow = counts[gy]
            for gx in range(gw):
                if is_white(row[gx * stride]):
                    crow[gx] += 1
    need = frac * len(frames)
    return [[c >= need for c in crow] for crow in counts]


def diff_blobs(bg, img, thresh, min_area, static):
    """Connected components (4-neigh, stride 1) of pixels differing from bg."""
    h = len(img)
    w = len(img[0])
    on = [[False] * w for i in range(h)]
    for y in range(h):
        rb = bg[y]
        ri = img[y]
        for x in range(w):
            a = rb[x]
            c = ri[x]
            if abs(a[0] - c[0]) + abs(a[1] - c[1]) + abs(a[2] - c[2]) > thresh:
                if not static[y // STRIDE][x // STRIDE]:
                    on[y][x] = True
    seen = [[False] * w for i in range(h)]
    blobs = []
    for y in range(h):
        for x in range(w):
            if on[y][x] and not seen[y][x]:
                stack = [(x, y)]
                seen[y][x] = True
                pix = []
                while stack:
                    p = stack.pop()
                    pix.append(p)
                    px0 = p[0]
                    py0 = p[1]
                    for q in ((px0 + 1, py0), (px0 - 1, py0), (px0, py0 + 1), (px0, py0 - 1)):
                        if 0 <= q[0] < w and 0 <= q[1] < h and on[q[1]][q[0]] and not seen[q[1]][q[0]]:
                            seen[q[1]][q[0]] = True
                            stack.append(q)
                if len(pix) >= min_area:
                    sr = 0
                    sg = 0
                    sb = 0
                    sx = 0
                    sy = 0
                    for p in pix:
                        c = img[p[1]][p[0]]
                        sr += c[0]
                        sg += c[1]
                        sb += c[2]
                        sx += p[0]
                        sy += p[1]
                    n = float(len(pix))
                    blobs.append({"cx": sx / n, "cy": sy / n, "area": len(pix),
                                  "rgb": [sr / n, sg / n, sb / n]})
    return blobs


def color_name(rgb):
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 70:
        return "black"
    if mx - mn < 35:
        if mn > 150:
            return "white"
        return "gray"
    # hue
    if mx == r:
        hue = 60.0 * (((g - b) / float(mx - mn)) % 6.0)
    elif mx == g:
        hue = 60.0 * ((b - r) / float(mx - mn) + 2.0)
    else:
        hue = 60.0 * ((r - g) / float(mx - mn) + 4.0)
    if hue < 15 or hue >= 330:
        return "red"
    if hue < 45:
        return "orange"
    if hue < 70:
        return "yellow"
    if hue < 170:
        return "green"
    if hue < 200:
        return "cyan"
    if hue < 260:
        return "blue"
    if hue < 330:
        return "purple"
    return "red"


def track_step(tracks, pts, gate, min_frac, iters):
    """tracks: list of dict(cx, cy, n_ref). Nearest-centre assignment within gate."""
    g2 = gate * gate
    k = len(tracks)
    counts = [0] * k
    for it in range(iters):
        sx = [0.0] * k
        sy = [0.0] * k
        counts = [0] * k
        cxs = [t["cx"] for t in tracks]
        cys = [t["cy"] for t in tracks]
        for p in pts:
            best = -1
            bd = g2
            px0 = p[0]
            py0 = p[1]
            for j in range(k):
                dx = px0 - cxs[j]
                dy = py0 - cys[j]
                d = dx * dx + dy * dy
                if d < bd:
                    bd = d
                    best = j
            if best >= 0:
                sx[best] += px0
                sy[best] += py0
                counts[best] += 1
        for j in range(k):
            t = tracks[j]
            ref = t["n_ref"]
            if counts[j] > 0 and (ref <= 0 or counts[j] >= min_frac * ref):
                t["cx"] = sx[j] / counts[j]
                t["cy"] = sy[j] / counts[j]
    for j in range(k):
        t = tracks[j]
        t["last_n"] = counts[j]
        if t["n_ref"] <= 0:
            t["n_ref"] = float(counts[j])
        elif counts[j] >= 0.8 * t["n_ref"]:
            t["n_ref"] = 0.95 * t["n_ref"] + 0.05 * counts[j]
    return counts


def grid_components(pts, stride, min_size):
    """Connected components (8-neigh) of stride-grid points; returns list of point lists."""
    occ = {}
    for p in pts:
        occ[(p[0] // stride, p[1] // stride)] = p
    seen = {}
    comps = []
    for key in occ:
        if key in seen:
            continue
        seen[key] = True
        stack = [key]
        comp = []
        while stack:
            c = stack.pop()
            comp.append(occ[c])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = (c[0] + dx, c[1] + dy)
                    if q in occ and q not in seen:
                        seen[q] = True
                        stack.append(q)
        if len(comp) >= min_size:
            comps.append(comp)
    return comps


def track_step2(tracks, pts, gate, vis_frac, iters, stride):
    """Visible tracks: nearest-centre update within gate. Lost tracks are frozen and
    re-acquired only from foreground blobs no visible track claims."""
    g2 = gate * gate
    k = len(tracks)
    vis = [j for j in range(k) if not tracks[j].get("lost", False)]
    counts = [0] * k
    claimed = [False] * len(pts)
    for it in range(iters):
        sx = [0.0] * k
        sy = [0.0] * k
        counts = [0] * k
        cxs = [tracks[j]["cx"] for j in range(k)]
        cys = [tracks[j]["cy"] for j in range(k)]
        for pi in range(len(pts)):
            p = pts[pi]
            best = -1
            bd = g2
            for j in vis:
                dx = p[0] - cxs[j]
                dy = p[1] - cys[j]
                d = dx * dx + dy * dy
                if d < bd:
                    bd = d
                    best = j
            claimed[pi] = best >= 0
            if best >= 0:
                sx[best] += p[0]
                sy[best] += p[1]
                counts[best] += 1
        for j in vis:
            if counts[j] > 0:
                tracks[j]["cx"] = sx[j] / counts[j]
                tracks[j]["cy"] = sy[j] / counts[j]
    events = []
    for j in vis:
        t = tracks[j]
        t["last_n"] = counts[j]
        if t["n_ref"] > 0 and counts[j] < vis_frac * t["n_ref"]:
            t["lost"] = True
            events.append(("lost", j))
        elif t["n_ref"] <= 0:
            t["n_ref"] = float(counts[j])
        elif counts[j] >= 0.8 * t["n_ref"]:
            t["n_ref"] = 0.95 * t["n_ref"] + 0.05 * counts[j]
    lost = [j for j in range(k) if tracks[j].get("lost", False) and j not in [e[1] for e in events]]
    if lost:
        free = [pts[pi] for pi in range(len(pts)) if not claimed[pi]]
        comps = grid_components(free, stride, 3)
        cands = []
        for comp in comps:
            n = len(comp)
            cx = sum(p[0] for p in comp) / float(n)
            cy = sum(p[1] for p in comp) / float(n)
            for j in lost:
                if n >= vis_frac * tracks[j]["n_ref"]:
                    d = (cx - tracks[j]["cx"]) ** 2 + (cy - tracks[j]["cy"]) ** 2
                    cands.append((d, j, cx, cy, n))
        cands.sort()
        used_j = []
        used_c = []
        for c in cands:
            if c[1] in used_j or (c[2], c[3]) in used_c:
                continue
            used_j.append(c[1])
            used_c.append((c[2], c[3]))
            t = tracks[c[1]]
            t["cx"] = c[2]
            t["cy"] = c[3]
            t["lost"] = False
            t["last_n"] = c[4]
            events.append(("found", c[1]))
    return events


def add_border_blobs_to_static(rgb, static, stride, border):
    """Mark white components touching the top image border as static (robot base in the
    live view). Returns number of grid cells added."""
    pts = white_points(rgb, static, stride)
    comps = grid_components(pts, stride, 1)
    added = 0
    gh = len(static)
    gw = len(static[0])
    for comp in comps:
        if min(p[1] for p in comp) <= border:
            for p in comp:
                gx = p[0] // stride
                gy = p[1] // stride
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if 0 <= gy + dy < gh and 0 <= gx + dx < gw and not static[gy + dy][gx + dx]:
                            static[gy + dy][gx + dx] = True
                            added += 1
    return added


def is_table(px):
    r = px[0]
    g = px[1]
    b = px[2]
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 90 or mx == mn:
        return False
    s = (mx - mn) / float(mx)
    if s < 0.28 or s > 0.84:
        return False
    if mx != r:
        return False
    hue = 60.0 * (((g - b) / float(mx - mn)) % 6.0)
    return 8.0 <= hue <= 32.0


def exact_static(frames, ref, tol, frac):
    """Per-pixel (stride 1) flag: colour within tol (sum abs) of ref in >= frac of frames."""
    h = len(ref)
    w = len(ref[0])
    counts = [[0] * w for i in range(h)]
    for img in frames:
        for y in range(h):
            ri = img[y]
            rr = ref[y]
            cr = counts[y]
            for x in range(w):
                a = ri[x]
                c = rr[x]
                if abs(a[0] - c[0]) + abs(a[1] - c[1]) + abs(a[2] - c[2]) <= tol:
                    cr[x] += 1
    need = frac * len(frames)
    return [[c >= need for c in cr] for cr in counts]


def cube_blobs(img, xstatic, min_area):
    """Connected components (4-neigh) of non-table, non-static pixels at the pre-cover frame."""
    h = len(img)
    w = len(img[0])
    on = [[False] * w for i in range(h)]
    for y in range(h):
        ri = img[y]
        sr = xstatic[y]
        for x in range(w):
            if not sr[x] and not is_table(ri[x]):
                on[y][x] = True
    seen = [[False] * w for i in range(h)]
    blobs = []
    for y in range(h):
        for x in range(w):
            if on[y][x] and not seen[y][x]:
                stack = [(x, y)]
                seen[y][x] = True
                pix = []
                while stack:
                    p = stack.pop()
                    pix.append(p)
                    for q in ((p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)):
                        if 0 <= q[0] < w and 0 <= q[1] < h and on[q[1]][q[0]] and not seen[q[1]][q[0]]:
                            seen[q[1]][q[0]] = True
                            stack.append(q)
                if len(pix) >= min_area:
                    sr2 = 0
                    sg = 0
                    sb = 0
                    sx = 0
                    sy = 0
                    for p in pix:
                        c = img[p[1]][p[0]]
                        sr2 += c[0]
                        sg += c[1]
                        sb += c[2]
                        sx += p[0]
                        sy += p[1]
                    n = float(len(pix))
                    blobs.append({"cx": sx / n, "cy": sy / n, "area": len(pix),
                                  "rgb": [sr2 / n, sg / n, sb / n]})
    return blobs


def dilate(mask, r):
    h = len(mask)
    w = len(mask[0])
    out = [list(row) for row in mask]
    for y in range(h):
        row = mask[y]
        for x in range(w):
            if row[x]:
                for yy in range(max(0, y - r), min(h, y + r + 1)):
                    orow = out[yy]
                    for xx in range(max(0, x - r), min(w, x + r + 1)):
                        orow[xx] = True
    return out


# ---- runtime main ----

report = {"version": VERSION, "errors": [], "timing": {}}
ctx = get_task_context()
goal = str(ctx.language)
report["goal"] = goal
try:
    meta = robomme_plus.get_task_metadata()
    report["max_steps"] = meta["max_steps"]
    try:
        report["sampling"] = meta["conditioning_video_sampling"]
    except Exception:
        pass
except Exception as exc:
    report["errors"].append("meta: " + str(exc))
state0 = get_robot_state()
base_frame = state0.base_frame
embodiment = state0.embodiment
home_joints = [float(v) for v in state0.joint_positions["primary"]]
home_pose = state0.end_effector_poses["primary"]
report["home_ee"] = [round(float(v), 4) for v in home_pose.position]
report["home_q"] = [round(float(v), 4) for v in home_pose.quaternion_wxyz]
t_start = float(state0.timestamp_s)

COLOR_WORDS = ("red", "green", "blue", "yellow", "orange", "purple", "black", "white", "pink", "cyan", "brown", "gray", "grey")


def parse_goal(text):
    """Ordered list of (color, all_flag) from the instruction."""
    low = text.lower().replace(",", " ").replace(".", " ")
    words = low.split()
    out = []
    for i in range(len(words)):
        w = words[i]
        if w in COLOR_WORDS:
            nxt = words[i + 1] if i + 1 < len(words) else ""
            if nxt.startswith("cube"):
                plural = nxt.startswith("cubes")
                c = "gray" if w == "grey" else w
                out.append((c, plural))
    return out


def to_lists(rgb):
    try:
        return rgb.tolist()
    except Exception:
        return [[[int(v) for v in px] for px in row] for row in rgb]


order = parse_goal(goal)
report["order"] = order

# ---------------- conditioning video ----------------
clock = 0.0
video = robomme_plus.get_conditioning_video()
nvid = len(video)
report["video_len"] = nvid
S = STRIDE
tracks = []
static = None
if nvid > 0:
    probe = []
    step16 = max(1, nvid // 32)
    fi = 0
    while fi < nvid:
        probe.append(to_lists(video[fi].cameras["front"].rgb))
        fi += step16
    static = static_mask(probe, S, 0.8)
    report["static_cells"] = sum(sum(1 for c in row if c) for row in static)
    bg = to_lists(video[0].cameras["front"].rgb)
    prev_img = bg
    prev_count = len(white_points(bg, static, S))
    cover = None
    fi = 1
    while fi < nvid and cover is None:
        img = to_lists(video[fi].cameras["front"].rgb)
        cnt = len(white_points(img, static, S))
        if cnt > max(60, 4 * prev_count):
            cover = fi
        else:
            prev_img = img
            prev_count = cnt
            fi += 1
    report["cover_index"] = cover
    if cover is not None:
        xstatic = dilate(exact_static(probe, prev_img, 45, 0.9), 1)
        blobs = cube_blobs(prev_img, xstatic, 20)
        report["cubes"] = [[round(b["cx"], 1), round(b["cy"], 1), b["area"], color_name(b["rgb"])] for b in blobs]
        tracks = [{"cx": b["cx"], "cy": b["cy"], "n_ref": 0.0, "color": color_name(b["rgb"])} for b in blobs]
        img = to_lists(video[cover].cameras["front"].rgb)
        track_step(tracks, white_points(img, static, S), 14.0, 0.0, 4)
        nlost = 0
        fi = cover + 1
        while fi < nvid:
            img = to_lists(video[fi].cameras["front"].rgb)
            ev = track_step2(tracks, white_points(img, static, S), GATE, VIS_FRAC, 2, S)
            nlost += len(ev)
            fi += 1
        report["video_track_events"] = nlost
    bg = None
    prev_img = None
    probe = None

# ---------------- live: keep tracking until containers settle ----------------
def hold_tick():
    st = get_robot_state()
    res = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=st.joint_positions["primary"],
        gripper_position=st.gripper_positions["primary"], embodiment=st.embodiment)}))
    return res


def ended(res):
    try:
        return bool(res.terminated) or bool(res.truncated)
    except Exception:
        return False


episode_over = False
live_ticks = 0
if tracks:
    obs = get_observation()
    img = to_lists(obs.cameras["front"].rgb)
    report["border_cells"] = add_border_blobs_to_static(img, static, S, 6)
    track_step2(tracks, white_points(img, static, S), GATE, VIS_FRAC, 2, S)
    prev = [(t["cx"], t["cy"]) for t in tracks]
    still = 0
    while live_ticks < SETTLE_MAX and still < SETTLE_N:
        res = hold_tick()
        live_ticks += 1
        if ended(res):
            episode_over = True
            break
        img = to_lists(get_observation().cameras["front"].rgb)
        track_step2(tracks, white_points(img, static, S), GATE, VIS_FRAC, 2, S)
        cur = [(t["cx"], t["cy"]) for t in tracks]
        mv = 0.0
        for j in range(len(cur)):
            d = math.sqrt((cur[j][0] - prev[j][0]) ** 2 + (cur[j][1] - prev[j][1]) ** 2)
            if d > mv:
                mv = d
        prev = cur
        if mv < 0.3:
            still += 1
        else:
            still = 0
report["settle_ticks"] = live_ticks
report["tracks"] = [[j, t["color"], round(t["cx"], 1), round(t["cy"], 1), round(t["n_ref"]), t.get("lost", False)] for j, t in enumerate(tracks)]

# ---------------- choose containers ----------------
plan = []
for item in order:
    c = item[0]
    ids = [j for j in range(len(tracks)) if tracks[j]["color"] == c]
    if not item[1]:
        ids = ids[:1]
    for j in ids:
        if j not in plan:
            plan.append(j)
report["plan"] = plan


def quantile(values, frac):
    vs = sorted(values)
    if not vs:
        return None
    return vs[int((len(vs) - 1) * frac)]


def quat_mul(a, b):
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]


def quat_xaxis(q):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    return [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)]


def wrap90(a):
    while a > 45.0:
        a -= 90.0
    while a <= -45.0:
        a += 90.0
    return a


def square_yaw(pts):
    """Yaw (deg) of a square footprint: angle minimizing the rotated bounding-box area."""
    best = None
    besta = 0.0
    if len(pts) < 5:
        return 0.0
    for k in range(30):
        a = k * 3.0
        r = math.radians(a)
        c = math.cos(r)
        s = math.sin(r)
        us = [p[0] * c + p[1] * s for p in pts]
        vs = [-p[0] * s + p[1] * c for p in pts]
        area = (quantile(us, 0.97) - quantile(us, 0.03)) * (quantile(vs, 0.97) - quantile(vs, 0.03))
        if best is None or area < best:
            best = area
            besta = a
    return besta


def locate_container(t):
    """Point-prompt SAM at the tracked image centroid; return dict or None."""
    u = int(round(t["cx"]))
    v = int(round(t["cy"]))
    seg = segment_points("front", [[u, v]])
    if not seg.ok or len(seg.segmentations) == 0:
        return None
    cloud = mask_to_point_cloud(seg.segmentations[0], "front", target_frame=base_frame)
    pts = [(float(p[0]), float(p[1]), float(p[2])) for p in cloud.points]
    if len(pts) < 20:
        return None
    zs = [p[2] for p in pts]
    ztop = quantile(zs, 0.95)
    zlow = quantile(zs, 0.03)
    top = [p for p in pts if p[2] > ztop - 0.012]
    cx = quantile([p[0] for p in top], 0.5)
    cy = quantile([p[1] for p in top], 0.5)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    info = {"x": cx, "y": cy, "ztop": ztop, "zlow": zlow, "n": len(pts),
            "ext": [quantile(xs, 0.97) - quantile(xs, 0.03), quantile(ys, 0.97) - quantile(ys, 0.03)],
            "yaw": square_yaw([(p[0] - cx, p[1] - cy) for p in top])}
    return info


home_q = [float(v) for v in home_pose.quaternion_wxyz]
hx = quat_xaxis(home_q)
home_axis_yaw = math.degrees(math.atan2(hx[1], hx[0]))


def grasp_quat(yaw_deg):
    d = wrap90(yaw_deg - home_axis_yaw)
    r = math.radians(d) / 2.0
    qz = [math.cos(r), 0.0, 0.0, math.sin(r)]
    return quat_mul(qz, home_q)


def go(pos, q, tol, steps):
    res = move_to_pose(Pose(position=list(pos), quaternion_wxyz=list(q), frame=base_frame), tolerance=tol, max_steps=steps)
    return res


picks = []
if not episode_over:
    # localize all planned containers from the clean live view first
    infos = {}
    for j in plan:
        try:
            infos[j] = locate_container(tracks[j])
        except Exception as exc:
            report["errors"].append("locate %d: %s" % (j, str(exc)))
            infos[j] = None
    report["infos"] = {}
    for j in infos:
        report["infos"][str(j)] = infos[j]
    for j in plan:
        if episode_over:
            break
        info = infos.get(j)
        rec = {"track": j, "color": tracks[j]["color"]}
        if info is None:
            rec["status"] = "no_info"
            picks.append(rec)
            continue
        q = grasp_quat(info["yaw"])
        gz = info["ztop"] - GRASP_DEPTH
        above = (info["x"], info["y"], info["ztop"] + 0.10)
        r1 = go(above, q, 0.015, 200)
        rec["above_ok"] = bool(r1.ok)
        if ended(r1):
            episode_over = True
            rec["status"] = "ended_above"
            picks.append(rec)
            break
        og = open_gripper()
        if ended(og):
            episode_over = True
            break
        r2 = go((info["x"], info["y"], gz), q, 0.01, 120)
        rec["down_ok"] = bool(r2.ok)
        if ended(r2):
            episode_over = True
            rec["status"] = "ended_down"
            picks.append(rec)
            break
        cg = close_gripper()
        if ended(cg):
            episode_over = True
            rec["status"] = "ended_close"
            picks.append(rec)
            break
        st = get_robot_state()
        rec["grip"] = round(float(st.gripper_positions["primary"]), 3)
        r3 = go((info["x"], info["y"], gz + LIFT), q, 0.015, 120)
        rec["lift_ok"] = bool(r3.ok)
        if ended(r3):
            episode_over = True
            rec["status"] = "ended_lift"
            picks.append(rec)
            break
        st = get_robot_state()
        rec["grip_after_lift"] = round(float(st.gripper_positions["primary"]), 3)
        rec["t"] = round(float(st.timestamp_s) - t_start, 2)
        # put it back where it was
        r4 = go((info["x"], info["y"], gz + 0.005), q, 0.015, 120)
        if ended(r4):
            episode_over = True
            rec["status"] = "ended_lower"
            picks.append(rec)
            break
        og = open_gripper()
        if ended(og):
            episode_over = True
            rec["status"] = "ended_release"
            picks.append(rec)
            break
        r5 = go((info["x"], info["y"], info["ztop"] + 0.08), q, 0.02, 120)
        rec["status"] = "done"
        picks.append(rec)
        if ended(r5):
            episode_over = True
            break
report["picks"] = picks
report["episode_over"] = episode_over
try:
    if not episode_over:
        report["t_end"] = round(float(get_robot_state().timestamp_s) - t_start, 2)
except Exception:
    pass
result = report
