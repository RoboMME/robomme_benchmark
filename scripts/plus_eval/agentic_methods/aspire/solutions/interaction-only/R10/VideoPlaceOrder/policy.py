# v01 — VideoPlaceOrder (robomme_plus): track target-disc IDENTITY through the conditioning video.
# Changes vs v00:
#   (a) task predicate / memory: the target discs are identical and are shuffled in the video, so
#       the N-th placement's disc is followed through every later video frame (front camera) with a
#       blob tracker; the goal is where THAT disc is at the end of the video (live scene).
#       Disc colour model is built at runtime from the live SAM target masks (no fixed colour).
#   (b) execution: cube bottom / table plane from front depth around the cube box (v00 saw only the
#       cube top and pressed the cube into the disc for 124 ticks); grasp at cube mid-height.
import math

report = {"version": "v01", "errors": [], "moves": []}
context = get_task_context()
language = str(context.language)
report["language"] = language
state0 = get_robot_state()
base_frame = state0.base_frame
home_quat = [float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz]

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "1st": 1, "2nd": 2,
            "3rd": 3, "4th": 4, "5th": 5, "6th": 6, "7th": 7, "8th": 8, "9th": 9,
            "10th": 10, "last": -1, "final": -1}
COLORS = {"red": (0.80, 0.10, 0.10), "green": (0.10, 0.80, 0.10), "blue": (0.10, 0.10, 0.80),
          "yellow": (0.45, 0.45, 0.10), "purple": (0.45, 0.10, 0.45), "orange": (0.60, 0.30, 0.10),
          "cyan": (0.10, 0.45, 0.45), "pink": (0.50, 0.20, 0.30)}

words = language.lower().replace(",", " ").replace(".", " ").split()
ordinal = None
for w in words:
    if w in ORDINALS:
        ordinal = ORDINALS[w]
        break
color_word = None
for k in range(len(words)):
    if words[k] in COLORS and k + 1 < len(words) and words[k + 1] in ("cube", "block", "box"):
        color_word = words[k]
        break
if color_word is None:
    for w in words:
        if w in COLORS:
            color_word = w
            break
report["ordinal"] = ordinal
report["color_word"] = color_word


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def as_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        return arr


def quat_to_rot(q):
    w = float(q[0])
    x = float(q[1])
    y = float(q[2])
    z = float(q[3])
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def project(cam, point):
    # base-frame point -> pixel (u, v) for a camera observation (OpenCV convention)
    R = quat_to_rot(cam.camera_pose.quaternion_wxyz)
    t = [float(v) for v in cam.camera_pose.position]
    d = [point[0] - t[0], point[1] - t[1], point[2] - t[2]]
    c = [R[0][i] * d[0] + R[1][i] * d[1] + R[2][i] * d[2] for i in range(3)]
    K = cam.intrinsics
    if c[2] <= 1e-6:
        return None
    return [float(K[0][0]) * c[0] / c[2] + float(K[0][2]), float(K[1][1]) * c[1] / c[2] + float(K[1][2])]


def backproject(cam, u, v, depth):
    R = quat_to_rot(cam.camera_pose.quaternion_wxyz)
    t = [float(x) for x in cam.camera_pose.position]
    K = cam.intrinsics
    cx = (u - float(K[0][2])) * depth / float(K[0][0])
    cy = (v - float(K[1][2])) * depth / float(K[1][1])
    c = [cx, cy, depth]
    return [R[i][0] * c[0] + R[i][1] * c[1] + R[i][2] * c[2] + t[i] for i in range(3)]


def chroma(px):
    s = float(px[0]) + float(px[1]) + float(px[2]) + 1e-6
    return (float(px[0]) / s, float(px[1]) / s, float(px[2]) / s)


def saturation(px):
    mx = max(px[0], px[1], px[2])
    mn = min(px[0], px[1], px[2])
    if mx <= 0:
        return 0.0
    return float(mx - mn) / float(mx)


def box_pixels(rgb, box, frac):
    x1, y1, x2, y2 = (float(v) for v in box)
    out = []
    for yy in range(max(int(y1 + frac * (y2 - y1)), 0), min(int(y2 - frac * (y2 - y1)) + 1, len(rgb))):
        row = rgb[yy]
        for xx in range(max(int(x1 + frac * (x2 - x1)), 0), min(int(x2 - frac * (x2 - x1)) + 1, len(row))):
            out.append(row[xx])
    return out


def mean_rgb(pixels):
    if len(pixels) == 0:
        return None
    return (sum(float(p[0]) for p in pixels) / len(pixels), sum(float(p[1]) for p in pixels) / len(pixels),
            sum(float(p[2]) for p in pixels) / len(pixels))


def chroma_distance(rgb_mean, proto):
    c = chroma(rgb_mean)
    pt = proto[0] + proto[1] + proto[2]
    q = (proto[0] / pt, proto[1] / pt, proto[2] / pt)
    return math.sqrt((c[0] - q[0]) ** 2 + (c[1] - q[1]) ** 2 + (c[2] - q[2]) ** 2)


def cloud_pts(seg, camera):
    pc = mask_to_point_cloud(seg, camera, target_frame=base_frame)
    return [[float(p[0]), float(p[1]), float(p[2])] for p in pc.points]


# ---------------- Live perception first (colour model for the discs, target positions) -----------
obs = get_observation()
front = obs.cameras["front"]
front_rgb = as_rows(front.rgb)
front_depth = as_rows(front.depth_m)

targets = []
seg_t = segment_text("front", "flat circular target")
if seg_t.ok:
    top_score = None
    for cand in seg_t.segmentations[:5]:
        if top_score is None:
            top_score = float(cand.score)
        if cand.box_xyxy is None or float(cand.score) < 0.5 * top_score:
            continue
        try:
            tpts = cloud_pts(cand, "front")
        except Exception as exc:
            report["errors"].append("target cloud: " + str(exc))
            continue
        if len(tpts) < 10:
            continue
        b = [float(v) for v in cand.box_xyxy]
        targets.append({"xy": [quantile([p[0] for p in tpts], 0.5), quantile([p[1] for p in tpts], 0.5)],
                        "top": quantile([p[2] for p in tpts], 0.9), "score": float(cand.score),
                        "box": b, "px": [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0], "seg": cand})
report["live_targets"] = [{"xy": [round(t["xy"][0], 4), round(t["xy"][1], 4)], "top": round(t["top"], 4),
                           "score": round(t["score"], 3), "px": [round(t["px"][0], 1), round(t["px"][1], 1)]}
                          for t in targets]
if len(targets) == 0:
    raise ValueError("no live targets")

# disc ring colour model from the inner part of the live target boxes (saturated pixels only)
ring = []
for t in targets:
    for px in box_pixels(front_rgb, t["box"], 0.25):
        if saturation(px) > 0.25:
            ring.append(chroma(px))
if len(ring) < 10:
    raise ValueError("no saturated target pixels for colour model")
ring_c = (sum(c[0] for c in ring) / len(ring), sum(c[1] for c in ring) / len(ring), sum(c[2] for c in ring) / len(ring))
dev = sorted(math.sqrt((c[0] - ring_c[0]) ** 2 + (c[1] - ring_c[1]) ** 2 + (c[2] - ring_c[2]) ** 2) for c in ring)
RING_TOL = max(0.05, 1.5 * dev[int(0.8 * (len(dev) - 1))])
report["ring_model"] = {"chroma": [round(v, 3) for v in ring_c], "tol": round(RING_TOL, 3), "n": len(ring)}


def is_disc(px):
    if saturation(px) <= 0.25:
        return False
    s = float(px[0]) + float(px[1]) + float(px[2]) + 1e-6
    a = float(px[0]) / s - ring_c[0]
    b = float(px[1]) / s - ring_c[1]
    c = float(px[2]) / s - ring_c[2]
    return a * a + b * b + c * c < RING_TOL * RING_TOL


STRIDE = 2


def find_blobs(rows, min_cells):
    h = len(rows)
    w = len(rows[0])
    gh = (h + STRIDE - 1) // STRIDE
    gw = (w + STRIDE - 1) // STRIDE
    grid = []
    for gy in range(gh):
        row = rows[gy * STRIDE]
        line = []
        for gx in range(gw):
            line.append(is_disc(row[gx * STRIDE]))
        grid.append(line)
    seen = [[False] * gw for k in range(gh)]
    blobs = []
    for gy in range(gh):
        for gx in range(gw):
            if grid[gy][gx] and not seen[gy][gx]:
                stack = [(gy, gx)]
                seen[gy][gx] = True
                sx = 0.0
                sy = 0.0
                n = 0
                while len(stack) > 0:
                    cell = stack.pop()
                    cy = cell[0]
                    cx = cell[1]
                    sx += cx
                    sy += cy
                    n += 1
                    for dy in (-2, -1, 0, 1, 2):
                        for dx in (-2, -1, 0, 1, 2):
                            ny = cy + dy
                            nx = cx + dx
                            if 0 <= ny < gh and 0 <= nx < gw and grid[ny][nx] and not seen[ny][nx]:
                                seen[ny][nx] = True
                                stack.append((ny, nx))
                if n >= min_cells:
                    blobs.append([sx / n * STRIDE, sy / n * STRIDE, float(n)])
    return blobs


def d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def enumerate_assign(preds, obs, gate2, share_cost, allow_none, allow_share):
    # exhaustive odometer search over track->blob assignments (K small)
    K = len(preds)
    M = len(obs)
    options = []
    for k in range(K):
        opts = []
        if allow_none:
            opts.append((-1, gate2))
        for j in range(M):
            c = d2(preds[k], obs[j])
            if allow_none and c > gate2:
                continue
            opts.append((j, c))
        if len(opts) == 0:
            opts.append((-1, gate2))
        options.append(opts)
    total = 1
    for o in options:
        total *= len(o)
    best = None
    best_cost = None
    if total > 50000:
        # greedy fallback
        combo = []
        taken = set()
        for k in range(K):
            cand = sorted(options[k], key=None) if False else options[k]
            pick = None
            for opt in cand:
                if opt[0] >= 0 and opt[0] in taken and not allow_share:
                    continue
                if pick is None or opt[1] < pick[1]:
                    pick = opt
            if pick is None:
                pick = (-1, gate2)
            combo.append(pick[0])
            if pick[0] >= 0:
                taken.add(pick[0])
        return combo
    idx = [0] * K
    while True:
        cost = 0.0
        used = {}
        valid = True
        for k in range(K):
            j = options[k][idx[k]][0]
            cost += options[k][idx[k]][1]
            if j >= 0:
                if j in used:
                    if not allow_share:
                        valid = False
                        break
                    cost += share_cost
                used[j] = used.get(j, 0) + 1
        if valid and (best_cost is None or cost < best_cost):
            best_cost = cost
            best = [options[k][idx[k]][0] for k in range(K)]
        p = K - 1
        while p >= 0:
            idx[p] += 1
            if idx[p] < len(options[p]):
                break
            idx[p] = 0
            p -= 1
        if p < 0:
            break
    return best


def resync(pos, obs):
    # bijective assignment of tracks to static blobs, linear-distance cost
    K = len(pos)
    best = None
    best_cost = None
    idx = list(range(K))
    # permutations of obs indices taken K at a time (M >= K)
    M = len(obs)
    stack = [[]]
    while len(stack) > 0:
        cur = stack.pop()
        if len(cur) == K:
            c = sum(math.sqrt(d2(pos[k], obs[cur[k]])) for k in range(K))
            if best_cost is None or c < best_cost:
                best_cost = c
                best = cur
            continue
        for j in range(M):
            if j not in cur:
                stack.append(cur + [j])
    return best


GATE = 30.0          # px per frame a disc may move between retained frames
SHARE_COST = 300.0
MOVING_EPS = 1.5
STATIC_TOL = 2.0


def track_step(pos, vel, area, frozen, obs):
    K = len(pos)
    preds = [[pos[k][0] + vel[k][0], pos[k][1] + vel[k][1]] for k in range(K)]
    combo = enumerate_assign(preds, obs, GATE * GATE, SHARE_COST, True, True)
    groups = {}
    for k in range(K):
        if combo[k] >= 0:
            groups.setdefault(combo[k], []).append(k)
    newpos = [None] * K
    newvel = [None] * K
    for j in groups:
        ks = groups[j]
        o = obs[j]
        if len(ks) == 1:
            k = ks[0]
            newpos[k] = [o[0], o[1]]
            newvel[k] = [o[0] - pos[k][0], o[1] - pos[k][1]]
            frozen[k] = False
            area[k] = 0.8 * area[k] + 0.2 * o[2]
            continue
        moving = [k for k in ks if frozen[k] or vel[k][0] ** 2 + vel[k][1] ** 2 > MOVING_EPS ** 2]
        static = [k for k in ks if k not in moving]
        for s in static:
            newpos[s] = [pos[s][0], pos[s][1]]
            newvel[s] = [0.0, 0.0]
        if len(moving) == 1:
            k = moving[0]
            n_static = sum(area[s] for s in static)
            rem = o[2] - n_static
            est = None
            if rem > 0.3 * area[k]:
                est = [(o[2] * o[0] - sum(area[s] * pos[s][0] for s in static)) / rem,
                       (o[2] * o[1] - sum(area[s] * pos[s][1] for s in static)) / rem]
                if d2(est, preds[k]) > GATE * GATE:
                    est = None
            if est is None:
                est = preds[k]
            newpos[k] = est
            newvel[k] = [est[0] - pos[k][0], est[1] - pos[k][1]]
        else:
            # two or more moving discs overlap: sit on the blob centroid, keep pre-merge velocity
            for k in moving:
                newpos[k] = [o[0], o[1]]
                newvel[k] = [vel[k][0], vel[k][1]]
                frozen[k] = True
    for k in range(K):
        if newpos[k] is None:
            newpos[k] = [pos[k][0], pos[k][1]]   # unseen (occluded): hold
            newvel[k] = [0.0, 0.0]
            frozen[k] = False
    return newpos, newvel


# ---------------- Conditioning video: placement events + disc tracking -----------------
placements = []
track_hist = []
video = robomme_plus.get_conditioning_video()
n_frames = len(video)
report["video_frames"] = n_frames
grips = []
ees = []
for i in range(n_frames):
    rs = video[i].robot_state
    grips.append(float(rs.gripper_positions["primary"]))
    p = rs.end_effector_poses["primary"].position
    ees.append([float(p[0]), float(p[1]), float(p[2])])
OPEN_LEVEL = 0.9
CLOSED_LEVEL = 0.8
EMPTY_LEVEL = 0.15
i = 0
while i < n_frames:
    if grips[i] < CLOSED_LEVEL:
        start = i
        while i < n_frames and grips[i] < OPEN_LEVEL:
            i += 1
        end = i
        if end >= n_frames:
            break
        closed = grips[start:end]
        settled = closed[len(closed) // 3:]
        held_min = min(settled) if len(settled) > 0 else min(closed)
        rel = end - 1
        while rel > start and grips[rel - 1] > grips[rel] + 0.02:
            rel -= 1
        event = {"grasp_frame": start, "release_frame": rel, "held_min": round(held_min, 3),
                 "xyz": [round(v, 4) for v in ees[rel]]}
        if held_min > EMPTY_LEVEL:
            placements.append(event)
        else:
            report.setdefault("empty_closures", []).append(event)
    else:
        i += 1
report["placements"] = placements
if len(placements) == 0:
    raise ValueError("no placement events found in conditioning video")
goal_idx = len(placements) - 1
if ordinal is not None and ordinal != -1:
    if ordinal <= len(placements):
        goal_idx = ordinal - 1
    else:
        report["errors"].append("ordinal beyond placements; using last")
report["goal_index"] = goal_idx
goal_event = placements[goal_idx]

# disc blobs in every retained frame
frame_blobs = []
for i in range(n_frames):
    rows = as_rows(video[i].cameras["front"].rgb)
    frame_blobs.append(find_blobs(rows, 10))
counts = [len(b) for b in frame_blobs]
K = max(counts[0], counts[-1])
report["disc_count"] = K
report["blob_counts_summary"] = [min(counts), max(counts)]
start_f = 0
while start_f < n_frames and counts[start_f] != K:
    start_f += 1
if start_f >= n_frames or K == 0:
    raise ValueError("no frame with all discs visible")
pos = [[b[0], b[1]] for b in frame_blobs[start_f]]
vel = [[0.0, 0.0] for k in range(K)]
area = [b[2] for b in frame_blobs[start_f]]
frozen = [False] * K
hist = {start_f: [list(p) for p in pos]}
prev = frame_blobs[start_f]
resyncs = 0
for i in range(start_f + 1, n_frames):
    ob = frame_blobs[i]
    is_static = (len(ob) == K and len(prev) == K and
                 max(min(d2(o, q) for q in prev) for o in ob) < STATIC_TOL ** 2)
    all_still = True
    for k in range(K):
        if frozen[k] or vel[k][0] ** 2 + vel[k][1] ** 2 > MOVING_EPS ** 2:
            all_still = False
    if is_static and all_still and max(min(d2(o, q) for o in ob) for q in pos) < STATIC_TOL ** 2:
        hist[i] = [list(p) for p in pos]       # nothing changed
    elif is_static:
        combo = resync(pos, ob)
        pos = [[ob[combo[k]][0], ob[combo[k]][1]] for k in range(K)]
        vel = [[0.0, 0.0] for k in range(K)]
        frozen = [False] * K
        resyncs += 1
        hist[i] = [list(p) for p in pos]
    else:
        pos, vel = track_step(pos, vel, area, frozen, ob)
        hist[i] = [list(p) for p in pos]
    prev = ob
report["resyncs"] = resyncs

# which disc was under the cube at the goal placement?
rel_frame = goal_event["release_frame"]
plane_z = quantile([t["top"] for t in targets], 0.5)
gx = goal_event["xyz"][0]
gy = goal_event["xyz"][1]
puv = project(video[rel_frame].cameras["front"], [gx, gy, plane_z])
look = rel_frame if rel_frame in hist else start_f
best_k = None
best_d = None
for k in range(K):
    dd = math.sqrt(d2(hist[look][k], puv))
    if best_d is None or dd < best_d:
        best_d = dd
        best_k = k
final_px = hist[n_frames - 1][best_k]
report["goal_disc"] = {"track": best_k, "release_px": [round(puv[0], 1), round(puv[1], 1)],
                       "match_px_dist": round(best_d, 1), "final_px": [round(final_px[0], 1), round(final_px[1], 1)],
                       "moved_px": round(math.sqrt(d2(final_px, puv)), 1)}
report["final_tracks"] = [[round(p[0], 1), round(p[1], 1)] for p in hist[n_frames - 1]]

# live disc nearest to the final tracked pixel
goal_t = None
goal_td = None
for t in targets:
    dd = math.sqrt(d2(t["px"], final_px))
    if goal_td is None or dd < goal_td:
        goal_td = dd
        goal_t = t
if goal_t is not None and goal_td < 12.0:
    place_x = goal_t["xy"][0]
    place_y = goal_t["xy"][1]
    target_top = goal_t["top"]
    report["goal_source"] = "live_target_box"
else:
    u = int(round(final_px[0]))
    v = int(round(final_px[1]))
    pts = []
    for yy in range(v - 2, v + 3):
        for xx in range(u - 2, u + 3):
            if 0 <= yy < len(front_depth) and 0 <= xx < len(front_depth[0]):
                dval = float(front_depth[yy][xx])
                if dval > 0:
                    pts.append(backproject(front, xx, yy, dval))
    if len(pts) == 0:
        raise ValueError("no depth at tracked goal pixel")
    place_x = quantile([p[0] for p in pts], 0.5)
    place_y = quantile([p[1] for p in pts], 0.5)
    target_top = quantile([p[2] for p in pts], 0.5)
    report["goal_source"] = "depth_backproject"
report["goal_live_dist_px"] = None if goal_td is None else round(goal_td, 1)
report["place_xy"] = [round(place_x, 4), round(place_y, 4)]

# ---------------- cube selection by colour ----------------
cube_prompts = []
if color_word is not None:
    cube_prompts.append(color_word + " cube")
cube_prompts.append("cube")
cube_candidates = []
for prompt in cube_prompts:
    seg_set = segment_text("front", prompt)
    if not seg_set.ok:
        continue
    for cand in seg_set.segmentations[:5]:
        if cand.box_xyxy is None:
            continue
        col = mean_rgb(box_pixels(front_rgb, cand.box_xyxy, 0.3))
        if col is None:
            continue
        dist = 0.0 if color_word is None else chroma_distance(col, COLORS[color_word])
        b = [float(v) for v in cand.box_xyxy]
        cube_candidates.append({"seg": cand, "prompt": prompt, "score": float(cand.score), "dist": dist,
                                "area": (b[2] - b[0]) * (b[3] - b[1]), "box": b,
                                "color": [round(v, 1) for v in col]})
if len(cube_candidates) == 0:
    raise ValueError("cube not found")
best_dist = min(c["dist"] for c in cube_candidates)
best_cube = None
for c in cube_candidates:
    if c["dist"] <= best_dist + 0.1 and (best_cube is None or c["area"] > best_cube["area"]):
        best_cube = c
report["cube_candidates"] = [{"prompt": c["prompt"], "score": round(c["score"], 3), "color": c["color"],
                              "dist": round(c["dist"], 3), "box": [round(v, 1) for v in c["box"]]}
                             for c in cube_candidates]
cube_pts = cloud_pts(best_cube["seg"], "front")
if len(cube_pts) < 10:
    raise ValueError("cube cloud too small")
cube_x = quantile([p[0] for p in cube_pts], 0.5)
cube_y = quantile([p[1] for p in cube_pts], 0.5)
cube_top = quantile([p[2] for p in cube_pts], 0.95)

# table plane: depth just outside the cube box
bx1, by1, bx2, by2 = best_cube["box"]
mw = max(3.0, 0.4 * (bx2 - bx1))
ring_pts = []
for yy in range(int(by1 - mw), int(by2 + mw) + 1):
    for xx in range(int(bx1 - mw), int(bx2 + mw) + 1):
        if bx1 - 1 <= xx <= bx2 + 1 and by1 - 1 <= yy <= by2 + 1:
            continue
        if 0 <= yy < len(front_depth) and 0 <= xx < len(front_depth[0]):
            dval = float(front_depth[yy][xx])
            if dval > 0:
                ring_pts.append(backproject(front, xx, yy, dval))
table_z = quantile([p[2] for p in ring_pts], 0.5) if len(ring_pts) > 0 else min(p[2] for p in cube_pts)
report["cube_front"] = {"xy": [round(cube_x, 4), round(cube_y, 4)], "top": round(cube_top, 4),
                        "table_z": round(table_z, 4), "n": len(cube_pts)}


def yaw_quat(theta):
    qz = (math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0))
    q = home_quat
    return [qz[0] * q[0] - qz[3] * q[3], qz[0] * q[1] - qz[3] * q[2],
            qz[0] * q[2] + qz[3] * q[1], qz[0] * q[3] + qz[3] * q[0]]


def footprint_yaw(pts):
    best = 0.0
    best_area = None
    for k in range(30):
        a = math.radians(3.0 * k)
        ca = math.cos(a)
        sa = math.sin(a)
        us = [p[0] * ca + p[1] * sa for p in pts]
        vs = [-p[0] * sa + p[1] * ca for p in pts]
        ar = (quantile(us, 0.95) - quantile(us, 0.05)) * (quantile(vs, 0.95) - quantile(vs, 0.05))
        if best_area is None or ar < best_area:
            best_area = ar
            best = a
    if best >= math.radians(45.0):
        best -= math.radians(90.0)
    return best


def move(label, x, y, z, quat, max_steps=150, tol=0.01):
    entry = {"label": label, "target": [round(x, 4), round(y, 4), round(z, 4)]}
    try:
        res = move_to_pose(Pose([x, y, z], quat, base_frame), tolerance=tol, max_steps=max_steps)
    except Exception as exc:
        entry["exception"] = str(exc)
        report["moves"].append(entry)
        return None
    try:
        entry["ok"] = bool(res.ok)
        entry["status"] = str(res.status)
        entry["steps"] = int(res.steps_executed)
        entry["terminated"] = bool(res.terminated)
        entry["truncated"] = bool(res.truncated)
    except Exception as exc:
        entry["diag_error"] = str(exc)
    report["moves"].append(entry)
    return res


def ended(res):
    if res is None:
        return True
    try:
        return bool(res.terminated) or bool(res.truncated)
    except Exception:
        return False


hover_dz = 0.08
quat = home_quat
done = False
r = move("pregrasp", cube_x, cube_y, cube_top + hover_dz, quat, max_steps=200, tol=0.015)
if ended(r):
    done = True

yaw = 0.0
if not done:
    try:
        wobs = get_observation()
        wrgb = as_rows(wobs.cameras["wrist"].rgb)
        refined = None
        for prompt in cube_prompts:
            ws = segment_text("wrist", prompt)
            if not ws.ok:
                continue
            best_w = None
            for cand in ws.segmentations[:5]:
                if cand.box_xyxy is None:
                    continue
                col = mean_rgb(box_pixels(wrgb, cand.box_xyxy, 0.3))
                if col is None:
                    continue
                dist = 0.0 if color_word is None else chroma_distance(col, COLORS[color_word])
                wpts = cloud_pts(cand, "wrist")
                if len(wpts) < 20:
                    continue
                wx = quantile([p[0] for p in wpts], 0.5)
                wy = quantile([p[1] for p in wpts], 0.5)
                if math.sqrt((wx - cube_x) ** 2 + (wy - cube_y) ** 2) > 0.04:
                    continue
                if best_w is None or dist < best_w[0]:
                    best_w = (dist, wx, wy, wpts)
            if best_w is not None:
                refined = best_w
                break
        if refined is not None:
            wpts = refined[3]
            ztop = quantile([p[2] for p in wpts], 0.9)
            top_pts = [p for p in wpts if p[2] > ztop - 0.01]
            if len(top_pts) < 15:
                top_pts = wpts
            yaw = footprint_yaw(top_pts)
            report["wrist_refine"] = {"xy": [round(refined[1], 4), round(refined[2], 4)],
                                      "yaw_deg": round(math.degrees(yaw), 1), "n": len(wpts), "ztop": round(ztop, 4)}
            cube_x = refined[1]
            cube_y = refined[2]
            cube_top = max(cube_top, ztop) if abs(ztop - cube_top) < 0.02 else cube_top
    except Exception as exc:
        report["errors"].append("wrist refine: " + str(exc))
    quat = yaw_quat(yaw)
    r = move("pregrasp_refined", cube_x, cube_y, cube_top + hover_dz, quat, max_steps=80, tol=0.015)
    if ended(r):
        done = True

cube_h = max(cube_top - table_z, 0.01)
grasp_z = table_z + 0.5 * cube_h
report["grasp_z"] = round(grasp_z, 4)
if not done:
    try:
        open_gripper()
    except Exception as exc:
        report["errors"].append("open: " + str(exc))
    r = move("grasp", cube_x, cube_y, grasp_z, quat, max_steps=100, tol=0.01)
    if ended(r):
        done = True
if not done:
    try:
        cg = close_gripper()
        report["close"] = {"ok": bool(cg.ok), "steps": int(cg.steps_executed),
                           "grip": float(get_robot_state().gripper_positions["primary"])}
    except Exception as exc:
        report["errors"].append("close: " + str(exc))
        cg = None
    if ended(cg):
        done = True

release_z = target_top + (grasp_z - table_z) + 0.015
transit_z = max(grasp_z, release_z) + 0.10
report["release_z"] = round(release_z, 4)
if not done:
    r = move("lift", cube_x, cube_y, transit_z, quat, max_steps=100, tol=0.02)
    if ended(r):
        done = True
if not done:
    try:
        report["grip_after_lift"] = float(get_robot_state().gripper_positions["primary"])
    except Exception:
        pass
    r = move("above_goal", place_x, place_y, transit_z, quat, max_steps=200, tol=0.015)
    if ended(r):
        done = True
if not done:
    r = move("lower", place_x, place_y, release_z, quat, max_steps=80, tol=0.01)
    if ended(r):
        done = True
if not done:
    try:
        og = open_gripper()
        report["release"] = {"ok": bool(og.ok), "steps": int(og.steps_executed)}
    except Exception as exc:
        report["errors"].append("release: " + str(exc))
        og = None
    if ended(og):
        done = True
if not done:
    r = move("retreat", place_x, place_y, release_z + 0.10, quat, max_steps=80, tol=0.02)
    if ended(r):
        done = True
result = report
