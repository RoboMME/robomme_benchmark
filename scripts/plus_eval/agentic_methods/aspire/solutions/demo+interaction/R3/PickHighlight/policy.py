# PickHighlight v02 -- memory: continuous multi-camera cube tracking for the whole episode.
# v01 (episode 0) tracked the highlighted cubes correctly through the first shuffle, but the
# front video shows the cubes reshuffle periodically for the whole episode (ep0 ticks ~55, ~195,
# ~335, ~475; the released demonstrations show the same pattern), so positions captured after the
# first shuffle went stale and the third grasp took a non-highlighted cube (episode terminated).
# v02 never stops observing: every control tick is a single joint-position step issued by this
# program (IK per tick, no multi-tick helpers), and after every tick all cube top faces are
# detected in both the static front camera and the wrist camera (depth + per-frame published
# calibration -> robot-base xy), fused, and associated to persistent tracks (constant-velocity
# prediction, global nearest-neighbour assignment, merged-blob splitting). Highlighted cubes are
# flagged from the wrist highlight discs; each grasp follows the target's live track and aborts the
# descent if the target starts moving.
import math

report = {"version": "v02", "events": []}
ended = {"flag": False}
clock = {"tick": 0}


def log(name, **kw):
    entry = {"event": name}
    for key in kw:
        entry[key] = kw[key]
    entry["tick"] = clock["tick"]
    report["events"].append(entry)


def check_end(res):
    term = False
    try:
        if res.terminated:
            term = True
    except Exception:
        pass
    try:
        if res.truncated:
            term = True
    except Exception:
        pass
    if term:
        ended["flag"] = True
    return term


def to_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        out = []
        for y in range(len(arr)):
            row = []
            for x in range(len(arr[0])):
                p = arr[y][x]
                row.append([int(p[0]), int(p[1]), int(p[2])])
            out.append(row)
        return out


def is_white(p):
    return p[0] > 200 and p[1] > 200 and p[2] > 200 and max(p) - min(p) < 30


def grid_white(rows, stride):
    n = 0
    for y in range(0, len(rows), stride):
        row = rows[y]
        for x in range(0, len(row), stride):
            if is_white(row[x]):
                n += 1
    return n


def components(mask):
    H = len(mask)
    W = len(mask[0])
    seen = [[False] * W for y in range(H)]
    comps = []
    for y in range(H):
        for x in range(W):
            if mask[y][x] and not seen[y][x]:
                stack = [(x, y)]
                seen[y][x] = True
                pts = []
                while stack:
                    cur = stack.pop()
                    pts.append(cur)
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx = cur[0] + dx
                        ny = cur[1] + dy
                        if 0 <= nx < W and 0 <= ny < H and mask[ny][nx] and not seen[ny][nx]:
                            seen[ny][nx] = True
                            stack.append((nx, ny))
                comps.append(pts)
    return comps


def bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def kmeans_split(pts, k):
    xs = sorted(pts, key=lambda p: p[0])
    ys = sorted(pts, key=lambda p: p[1])
    order = xs if (xs[-1][0] - xs[0][0]) >= (ys[-1][1] - ys[0][1]) else ys
    centers = []
    for i in range(k):
        p = order[int((i + 0.5) * len(order) / k)]
        centers.append((float(p[0]), float(p[1])))
    groups = []
    for it in range(12):
        groups = [[] for i in range(k)]
        for p in pts:
            best = 0
            bd = None
            for i in range(k):
                d = (p[0] - centers[i][0]) ** 2 + (p[1] - centers[i][1]) ** 2
                if bd is None or d < bd:
                    bd = d
                    best = i
            groups[best].append(p)
        newc = []
        for i in range(k):
            g = groups[i]
            if g:
                newc.append((sum(p[0] for p in g) / float(len(g)), sum(p[1] for p in g) / float(len(g))))
            else:
                newc.append(centers[i])
        centers = newc
    return [g for g in groups if g]


def detect_discs(base_rows, hl_rows):
    H = len(hl_rows)
    W = len(hl_rows[0])
    mask = [[is_white(hl_rows[y][x]) and not is_white(base_rows[y][x]) for x in range(W)] for y in range(H)]
    comps = components(mask)
    base_mask = [[is_white(base_rows[y][x]) for x in range(W)] for y in range(H)]
    base_boxes = [bbox(c) for c in components(base_mask) if len(c) >= 30]
    kept = []
    rejected = []
    for c in comps:
        overl = False
        for bb in base_boxes:
            inside = 0
            for p in c:
                if bb[0] - 2 <= p[0] <= bb[2] + 2 and bb[1] - 2 <= p[1] <= bb[3] + 2:
                    inside += 1
            if inside > 0.5 * len(c):
                overl = True
        if overl:
            rejected.append({"area": len(c), "box": bbox(c), "why": "inside baseline-white box"})
        else:
            kept.append(c)
    info = {"n_components": len(comps), "base_white_boxes": base_boxes, "rejected": rejected}
    if not kept:
        return [], info
    squarish = []
    for c in kept:
        b = bbox(c)
        w = b[2] - b[0] + 1.0
        h = b[3] - b[1] + 1.0
        if max(w, h) / min(w, h) < 1.25 and len(c) >= 40:
            squarish.append(len(c))
    squarish.sort()
    all_areas = sorted(len(c) for c in kept)
    med_all = all_areas[len(all_areas) // 2]
    plausible = [a for a in squarish if a >= 0.4 * med_all]
    ref = plausible[0] if plausible else all_areas[-1]
    info["areas"] = all_areas
    info["ref_area"] = ref
    discs = []
    for c in kept:
        if len(c) < 0.35 * ref:
            rejected.append({"area": len(c), "box": bbox(c), "why": "small"})
            continue
        k = max(1, min(3, int(round(len(c) / float(ref)))))
        groups = kmeans_split(c, k) if k > 1 else [c]
        for g in groups:
            cx = sum(p[0] for p in g) / float(len(g))
            cy = sum(p[1] for p in g) / float(len(g))
            rmax = max(math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2) for p in g)
            discs.append({"cx": cx, "cy": cy, "area": len(g), "rmax": rmax, "k": k})
    for d in discs:
        r = 0.8 * d["rmax"]
        sx = 0.0
        sy = 0.0
        n = 0
        for y in range(max(0, int(d["cy"] - r)), min(H, int(d["cy"] + r) + 1)):
            for x in range(max(0, int(d["cx"] - r)), min(W, int(d["cx"] + r) + 1)):
                if (x - d["cx"]) ** 2 + (y - d["cy"]) ** 2 <= r * r and not mask[y][x] and not is_white(hl_rows[y][x]):
                    sx += x
                    sy += y
                    n += 1
        d["cube_px"] = [sx / n, sy / n] if n else [d["cx"], d["cy"]]
        d["cube_px_n"] = n
    return discs, info


def quantile(values, q):
    v = sorted(values)
    if not v:
        return None
    return v[int((len(v) - 1) * q)]


def top_face_yaw(pts):
    """Return (yaw_rad mod pi/2, side_m, cx, cy) from the cube's top-face points."""
    zs = [p[2] for p in pts]
    ztop = quantile(zs, 0.95)
    top = [p for p in pts if p[2] >= ztop - 0.008]
    if len(top) < 12:
        return None
    best = None
    for step_i in range(30):
        a = step_i * (math.pi / 2.0) / 30.0
        ca = math.cos(a)
        sa = math.sin(a)
        us = [p[0] * ca + p[1] * sa for p in top]
        vs = [-p[0] * sa + p[1] * ca for p in top]
        eu = quantile(us, 0.97) - quantile(us, 0.03)
        ev = quantile(vs, 0.97) - quantile(vs, 0.03)
        area = eu * ev
        if best is None or area < best[0]:
            best = (area, a, eu, ev)
    cx = quantile([p[0] for p in top], 0.5)
    cy = quantile([p[1] for p in top], 0.5)
    return (best[1], 0.5 * (best[2] + best[3]), cx, cy, len(top))


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def qrot(q, v):
    p = [0.0, v[0], v[1], v[2]]
    qc = [q[0], -q[1], -q[2], -q[3]]
    r = qmul(qmul(q, p), qc)
    return [r[1], r[2], r[3]]


def wrap45(a):
    quarter = math.pi / 2.0
    while a > quarter / 2.0:
        a -= quarter
    while a < -quarter / 2.0:
        a += quarter
    return a


def quat_matrix(q):
    w, x, y, z = [float(v) for v in q]
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def kmeans_seeded(pts, seeds, iters=8):
    centers = [(float(s[0]), float(s[1])) for s in seeds]
    groups = []
    for it in range(iters):
        groups = [[] for c in centers]
        for p in pts:
            best = 0
            bd = None
            for i in range(len(centers)):
                d = (p[0] - centers[i][0]) ** 2 + (p[1] - centers[i][1]) ** 2
                if bd is None or d < bd:
                    bd = d
                    best = i
            groups[best].append(p)
        newc = []
        for i in range(len(centers)):
            g = groups[i]
            if g:
                newc.append((sum(p[0] for p in g) / float(len(g)), sum(p[1] for p in g) / float(len(g))))
            else:
                newc.append(centers[i])
        centers = newc
    return centers, groups


def farthest_point(pts, seeds):
    best = None
    for p in pts:
        dmin = min((p[0] - s[0]) ** 2 + (p[1] - s[1]) ** 2 for s in seeds)
        if best is None or dmin > best[0]:
            best = (dmin, p)
    return best[1]


# ---------------------------------------------------------------- camera geometry (published per-frame calibration)
def cam_geom(cam):
    K = [[float(v) for v in row] for row in cam.intrinsics]
    pose = cam.camera_pose
    return {"R": quat_matrix(pose.quaternion_wxyz), "pos": [float(v) for v in pose.position],
            "fx": K[0][0], "fy": K[1][1], "cx": K[0][2], "cy": K[1][2]}


def pix_to_base(g, u, v, d):
    xc = (u - g["cx"]) / g["fx"]
    yc = (v - g["cy"]) / g["fy"]
    R = g["R"]
    p = g["pos"]
    return [p[0] + d * (R[0][0] * xc + R[0][1] * yc + R[0][2]),
            p[1] + d * (R[1][0] * xc + R[1][1] * yc + R[1][2]),
            p[2] + d * (R[2][0] * xc + R[2][1] * yc + R[2][2])]


def z_band_mask(g, depth, z_lo, z_hi):
    """Pixels whose deprojected height lies in [z_lo, z_hi] (OpenCV pinhole, z-depth)."""
    R2 = g["R"][2]
    pz = g["pos"][2]
    H = len(depth)
    W = len(depth[0])
    xcs = [R2[0] * (u - g["cx"]) / g["fx"] for u in range(W)]
    mask = []
    for v in range(H):
        c = R2[1] * (v - g["cy"]) / g["fy"] + R2[2]
        mask.append([(d > 0.0 and z_lo < pz + d * (a + c) < z_hi) for d, a in zip(depth[v], xcs)])
    return mask


def spread(pts):
    n = float(len(pts))
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts) / n
    syy = sum((p[1] - my) ** 2 for p in pts) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts) / n
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = math.sqrt(max(0.0, tr * tr / 4.0 - det))
    l1 = tr / 2.0 + disc
    l2 = max(0.0, tr / 2.0 - disc)
    return mx, my, math.sqrt(l1), math.sqrt(l2)


def camera_blobs(name, cam, z_lo, z_hi, min_px=6):
    """Top-face blobs (robot-base points) of everything whose surface lies in the cube-top height band."""
    g = cam_geom(cam)
    depth = to_rows(cam.depth_m)
    mask = z_band_mask(g, depth, z_lo, z_hi)
    H = len(mask)
    W = len(mask[0])
    out = []
    for c in components(mask):
        if len(c) < min_px:
            continue
        border = False
        for q in c:
            if q[0] <= 0 or q[1] <= 0 or q[0] >= W - 1 or q[1] >= H - 1:
                border = True
                break
        if border:
            continue
        stride = max(1, len(c) // 400)
        pts = [pix_to_base(g, q[0], q[1], depth[q[1]][q[0]]) for q in c[::stride]]
        mx, my, s1, s2 = spread(pts)
        out.append({"cam": name, "pts": pts, "x": mx, "y": my, "s1": s1, "s2": s2, "npx": len(c),
                    "upx": sum(q[0] for q in c) / float(len(c)), "vpx": sum(q[1] for q in c) / float(len(c))})
    return out, g, depth


# ---------------------------------------------------------------- tracking state
P = {"cube_top": None, "band": 0.004, "s_single": None, "s2_min": None, "grasp_depth": 0.019,
     "button": None, "ws": None}
tracks = []
T = {"active": None, "ee": None, "ee_low": False, "moving_ticks": 0, "quiet_ticks": 0, "last_motion": -1, "onsets": [], "ends": [],
     "motion_log": [], "n_det": [], "spawned": 0}


def blob_to_dets(b, preds):
    """Turn one blob into 1..k cube detections; split merged top faces using live predictions."""
    s_single = P["s_single"]
    claim = []
    sub = b["pts"][::max(1, len(b["pts"]) // 120)]
    for pr in preds:
        dmin = min((q[0] - pr[0]) ** 2 + (q[1] - pr[1]) ** 2 for q in sub)
        if dmin < 0.010 ** 2:
            claim.append(pr)
    k = len(claim)
    if b["s1"] > 1.6 * s_single:
        k = max(k, 2)
    if b["s1"] > 2.6 * s_single:
        k = max(k, 3)
    if k <= 1:
        return [{"x": b["x"], "y": b["y"], "pts": b["pts"], "cam": b["cam"], "split": 1}]
    seeds = [(c[0], c[1]) for c in claim]
    if not seeds:
        seeds.append((b["x"], b["y"]))
        seeds = [farthest_point(b["pts"], seeds)]
    while len(seeds) < k:
        seeds.append(farthest_point(b["pts"], seeds))
    centers, groups = kmeans_seeded(b["pts"], seeds[:k])
    out = []
    for ci in range(len(centers)):
        if len(groups[ci]) >= 3:
            out.append({"x": centers[ci][0], "y": centers[ci][1], "pts": groups[ci], "cam": b["cam"], "split": k})
    return out


def valid_blob(b):
    if P["s_single"] is None:
        return True
    if b["s2"] < 0.35 * P["s2_min"]:
        return False
    if b["s1"] > 4.0 * P["s_single"]:
        return False
    ws = P["ws"]
    if ws is not None and not (ws[0] <= b["x"] <= ws[2] and ws[1] <= b["y"] <= ws[3]):
        return False
    return True


def fuse(dets):
    """Merge detections of the same cube seen by both cameras."""
    fused = []
    wr = [d for d in dets if d["cam"] == "wrist"]
    fr = [d for d in dets if d["cam"] == "front"]
    used = set()
    for d in wr:
        best = None
        for j in range(len(fr)):
            if j in used:
                continue
            e = fr[j]
            dd = (d["x"] - e["x"]) ** 2 + (d["y"] - e["y"]) ** 2
            if dd < 0.015 ** 2 and (best is None or dd < best[0]):
                best = (dd, j)
        if best is not None:
            used.add(best[1])
            e = fr[best[1]]
            fused.append({"x": 0.5 * (d["x"] + e["x"]), "y": 0.5 * (d["y"] + e["y"]), "pts": d["pts"], "cams": "wf",
                          "wrist": d, "split": max(d["split"], e["split"])})
        else:
            fused.append({"x": d["x"], "y": d["y"], "pts": d["pts"], "cams": "w", "wrist": d, "split": d["split"]})
    for j in range(len(fr)):
        if j not in used:
            e = fr[j]
            fused.append({"x": e["x"], "y": e["y"], "pts": e["pts"], "cams": "f", "wrist": None, "split": e["split"]})
    return fused


def new_track(x, y, origin):
    t = {"id": len(tracks), "x": x, "y": y, "vx": 0.0, "vy": 0.0, "miss": 0, "hl": False, "done": False,
         "held": False, "seen": 1, "origin": origin, "wrist": None, "last_seen": clock["tick"]}
    tracks.append(t)
    T["spawned"] += 1
    return t


def perceive(obs, allow_spawn=True):
    """One perception + tracking update from both cameras of this tick's observation."""
    if P["cube_top"] is None:
        return
    z_lo = P["cube_top"] - P["band"]
    z_hi = P["cube_top"] + P["band"]
    blobs = []
    for name in ("front", "wrist"):
        try:
            bl, g, depth = camera_blobs(name, obs.cameras[name], z_lo, z_hi)
            blobs += bl
        except Exception as exc:
            log("perceive_error", cam=name, err=str(exc)[:200])
    # predictions
    preds = []
    for t in tracks:
        if t["held"]:
            continue
        t["px"] = t["x"] + t["vx"]
        t["py"] = t["y"] + t["vy"]
        preds.append((t["px"], t["py"]))
    dets = []
    for b in blobs:
        if not valid_blob(b):
            continue
        dets += blob_to_dets(b, preds)
    dets = fuse(dets)
    ee = T["ee"]
    act = T["active"]
    for d in dets:
        d["near_ee"] = False
        if ee is not None and T["ee_low"]:
            if (d["x"] - ee[0]) ** 2 + (d["y"] - ee[1]) ** 2 < 0.035 ** 2:
                d["near_ee"] = True
        btn = P["button"]
        d["near_btn"] = btn is not None and (d["x"] - btn["x"]) ** 2 + (d["y"] - btn["y"]) ** 2 < 0.03 ** 2
    T["n_det"].append(len(dets))
    pairs = []
    for ti in range(len(tracks)):
        t = tracks[ti]
        if t["held"]:
            continue
        gate = 0.015 + 0.010 * min(t["miss"], 4)
        for di in range(len(dets)):
            d = dets[di]
            if d["near_ee"] and (act is None or t["id"] != act["id"]):
                continue
            dd = math.sqrt((d["x"] - t["px"]) ** 2 + (d["y"] - t["py"]) ** 2)
            if dd < gate:
                pairs.append((dd, ti, di))
    pairs.sort()
    used_t = set()
    used_d = set()
    n_moving = 0
    for pr in pairs:
        if pr[1] in used_t or pr[2] in used_d:
            continue
        used_t.add(pr[1])
        used_d.add(pr[2])
        t = tracks[pr[1]]
        d = dets[pr[2]]
        span = float(t["miss"] + 1)
        nvx = (d["x"] - t["x"]) / span
        nvy = (d["y"] - t["y"]) / span
        if math.sqrt(nvx * nvx + nvy * nvy) > 0.0025 and not d["near_ee"]:
            n_moving += 1
        if d["split"] > 1:
            # merged / crossing cubes: the split centres are unreliable, keep the pre-merge velocity
            t["x"] = 0.5 * (t["px"] + d["x"])
            t["y"] = 0.5 * (t["py"] + d["y"])
            t["merged"] = clock["tick"]
        else:
            t["vx"] = 0.5 * t["vx"] + 0.5 * nvx
            t["vy"] = 0.5 * t["vy"] + 0.5 * nvy
            t["x"] = d["x"]
            t["y"] = d["y"]
        t["miss"] = 0
        t["seen"] += 1
        t["last_seen"] = clock["tick"]
        t["wrist"] = d["wrist"]
    for ti in range(len(tracks)):
        t = tracks[ti]
        if t["held"] or ti in used_t:
            continue
        t["x"] = t["px"]
        t["y"] = t["py"]
        t["vx"] *= 0.6
        t["vy"] *= 0.6
        t["miss"] += 1
        t["wrist"] = None
    if allow_spawn:
        for di in range(len(dets)):
            if di in used_d:
                continue
            d = dets[di]
            if d["near_ee"] or d["near_btn"]:
                continue
            far = True
            for t in tracks:
                if (d["x"] - t["x"]) ** 2 + (d["y"] - t["y"]) ** 2 < 0.03 ** 2:
                    far = False
                    break
            if far:
                new_track(d["x"], d["y"], "spawn@%d" % clock["tick"])
    # robust motion test: displacement over two consecutive 3-tick windows, same direction
    # (partial occlusion by the arm shifts a centroid once; a shuffle keeps moving cubes)
    mv = []
    for t in tracks:
        if t["held"]:
            t["hist"] = []
            continue
        h = (t.get("hist") or []) + [(t["x"], t["y"], t["miss"])]
        h = h[-8:]
        t["hist"] = h
        t["moving"] = False
        if len(h) >= 7 and h[-1][2] == 0 and h[-4][2] == 0:
            d1x = h[-1][0] - h[-4][0]
            d1y = h[-1][1] - h[-4][1]
            d2x = h[-4][0] - h[-7][0]
            d2y = h[-4][1] - h[-7][1]
            if (d1x * d1x + d1y * d1y > 0.008 ** 2 and d2x * d2x + d2y * d2y > 0.008 ** 2
                    and d1x * d2x + d1y * d2y > 0.0):
                t["moving"] = True
                mv.append(t["id"])
    moving = len(mv) >= 2
    T["mv"] = mv
    if moving and T["quiet_ticks"] >= 15:
        T["onsets"].append(clock["tick"])
        log("shuffle_onset", onsets=list(T["onsets"]))
    if moving:
        T["moving_ticks"] += 1
        T["quiet_ticks"] = 0
        T["last_motion"] = clock["tick"]
    else:
        T["quiet_ticks"] += 1
        T["moving_ticks"] = 0
        if T["quiet_ticks"] == 6 and len(T["onsets"]) > len(T["ends"]):
            T["ends"].append(clock["tick"] - 6)
    if clock["tick"] % 5 == 0:
        T["motion_log"].append([clock["tick"], 1 if moving else 0, len(dets)] +
                               [[t["id"], round(t["x"], 3), round(t["y"], 3), t["miss"]] for t in tracks if t["hl"]])


# ---------------------------------------------------------------- control: one joint-position step per tick
C = {"joints": None, "grip": 1.0, "pos": None, "yaw": 0.0, "emb": None, "ik_fail": 0, "obs": None, "state": None}


def read_state(obs):
    try:
        st = obs.robot_state
        st.end_effector_poses["primary"]
    except Exception:
        st = get_robot_state()
    C["state"] = st
    ee = st.end_effector_poses["primary"].position
    T["ee"] = [float(ee[0]), float(ee[1]), float(ee[2])]
    T["ee_low"] = T["ee"][2] < 0.16
    return st


def tick(do_perceive=True):
    if ended["flag"]:
        return None
    reply = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=[float(v) for v in C["joints"]], gripper_position=float(C["grip"]),
        embodiment=C["emb"])}))
    clock["tick"] += 1
    if check_end(reply):
        return None
    try:
        obs = reply.observation
        obs.cameras["front"]
    except Exception:
        obs = get_observation()
    C["obs"] = obs
    read_state(obs)
    held = [t for t in tracks if t["held"]]
    for t in held:
        t["x"] = T["ee"][0]
        t["y"] = T["ee"][1]
    if do_perceive:
        perceive(obs)
    return obs


def q_for_yaw(yaw):
    return qmul([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], q_down)


def ik_to(pos, yaw):
    target = Pose(position=[float(pos[0]), float(pos[1]), float(pos[2])],
                  quaternion_wxyz=[float(v) for v in q_for_yaw(yaw)], frame=base_frame)
    try:
        ik = solve_ik(target, seed_joints=[float(v) for v in C["joints"]])
    except Exception as exc:
        C["ik_fail"] += 1
        log("ik_error", err=str(exc)[:200])
        return False
    if not ik.ok:
        C["ik_fail"] += 1
        return False
    joints = [float(v) for v in ik.joint_positions]
    jump = max(abs(a - b) for a, b in zip(joints, C["joints"]))
    if jump > 0.6:
        C["ik_fail"] += 1
        log("ik_jump_rejected", jump=round(jump, 3), target=[round(float(v), 3) for v in pos])
        return False
    C["joints"] = joints
    C["pos"] = [float(pos[0]), float(pos[1]), float(pos[2])]
    C["yaw"] = yaw
    return True


def servo_step(goal, yaw, vmax=0.012, wmax=0.15):
    """Advance the commanded pose one bounded increment toward goal and execute one tick."""
    cur = C["pos"]
    d = [goal[i] - cur[i] for i in range(3)]
    n = math.sqrt(sum(v * v for v in d))
    if n > vmax:
        nxt = [cur[i] + d[i] * vmax / n for i in range(3)]
    else:
        nxt = [goal[0], goal[1], goal[2]]
    dy = yaw - C["yaw"]
    if dy > wmax:
        nyaw = C["yaw"] + wmax
    elif dy < -wmax:
        nyaw = C["yaw"] - wmax
    else:
        nyaw = yaw
    ik_to(nxt, nyaw)
    tick()
    return n


def ee_err(goal):
    ee = T["ee"]
    return math.sqrt(sum((ee[i] - goal[i]) ** 2 for i in range(3)))


def go_to(goal, yaw, vmax=0.012, tol=0.006, settle=12, max_ticks=150, name=""):
    k = 0
    while k < max_ticks and not ended["flag"]:
        n = servo_step(goal, yaw, vmax)
        k += 1
        if n < 1e-4 and abs(C["yaw"] - yaw) < 1e-6:
            break
    s = 0
    while s < settle and not ended["flag"] and ee_err(goal) > tol:
        tick()
        s += 1
    if name:
        log("go:" + name, goal=[round(v, 4) for v in goal], ticks=k + s,
            err=round(ee_err(goal), 4) if not ended["flag"] else None)
    return k + s


def gripper_reading():
    try:
        return float(C["state"].gripper_positions["primary"])
    except Exception:
        return None


def set_grip(value, ticks):
    C["grip"] = value
    for k in range(ticks):
        if ended["flag"]:
            break
        tick()
    return gripper_reading()


# ---------------------------------------------------------------- setup
ctx = get_task_context()
state0 = get_robot_state()
base_frame = state0.base_frame
C["emb"] = state0.embodiment
C["joints"] = [float(v) for v in state0.joint_positions["primary"]]
home_joints = list(C["joints"])
report["language"] = ctx.language
max_steps_total = None
try:
    meta = robomme_plus.get_task_metadata()
    report["task_metadata"] = {str(k): str(meta[k])[:200] for k in meta}
    max_steps_total = int(meta["max_steps"])
except Exception as exc:
    report["task_metadata_error"] = str(exc)[:200]
if max_steps_total is None:
    max_steps_total = 1500
home_pose = state0.end_effector_poses["primary"]
q_home = [float(v) for v in home_pose.quaternion_wxyz]
tool_z = qrot(q_home, [0.0, 0.0, 1.0])
q_down = q_home if tool_z[2] < -0.95 else [0.0, 1.0, 0.0, 0.0]
C["pos"] = [float(v) for v in home_pose.position]
home_pos = list(C["pos"])
fx_tool = qrot(q_down, [1.0, 0.0, 0.0])
phi_tool = math.atan2(fx_tool[1], fx_tool[0])
report["home"] = {"pos": home_pos, "quat": q_home}

obs0 = get_observation()
C["obs"] = obs0
read_state(obs0)
wcam0 = obs0.cameras["wrist"]
g0 = cam_geom(wcam0)
depth0 = to_rows(wcam0.depth_m)
zs0 = []
for v in range(0, len(depth0), 4):
    for u in range(0, len(depth0[0]), 4):
        d = depth0[v][u]
        if d > 0.0:
            zs0.append(pix_to_base(g0, u, v, d)[2])
table_z = quantile(zs0, 0.5)
# raised objects (everything 1.2..10 cm above the table) -> per-blob top height -> cube top height
coarse, gc, dc = camera_blobs("wrist", wcam0, table_z + 0.012, table_z + 0.10, min_px=20)
tops = []
for b in coarse:
    zz = sorted(q[2] for q in b["pts"])
    tops.append(zz[int(0.9 * (len(zz) - 1))])
cube_top = quantile(tops, 0.5)
P["cube_top"] = cube_top
report["table_z"] = table_z
report["coarse_tops"] = [round(v, 4) for v in sorted(tops)]

# button = baseline-white region in the wrist image; its cap height comes from depth
rows0 = to_rows(wcam0.rgb)
base_white = [bbox(c) for c in components([[is_white(p) for p in row] for row in rows0]) if len(c) >= 30]
button = None
best = None
for b in base_white:
    w_ = b[2] - b[0] + 1
    h_ = b[3] - b[1] + 1
    if b[3] >= len(rows0) - 2 or b[1] <= 1:
        continue
    if max(w_, h_) > 3.0 * min(w_, h_) or w_ * h_ > 3000:
        continue
    if best is None or w_ * h_ > best[0]:
        best = (w_ * h_, b)
if best is not None:
    b = best[1]
    pts = []
    for v in range(b[1], b[3] + 1):
        for u in range(b[0], b[2] + 1):
            d = depth0[v][u]
            if d > 0.0:
                pts.append(pix_to_base(g0, u, v, d))
    ztop_b = quantile([q[2] for q in pts], 0.95)
    capp = [q for q in pts if q[2] >= ztop_b - 0.006]
    button = {"x": quantile([q[0] for q in capp], 0.5), "y": quantile([q[1] for q in capp], 0.5), "z_top": ztop_b,
              "box": b}
P["button"] = button
report["button"] = button

# single-cube statistics from the initial top faces (both cameras)
init_blobs = []
for name in ("wrist", "front"):
    try:
        bl, g, dd = camera_blobs(name, obs0.cameras[name], cube_top - P["band"], cube_top + P["band"])
        init_blobs += bl
    except Exception as exc:
        log("init_blob_error", cam=name, err=str(exc)[:200])
s1s = sorted(b["s1"] for b in init_blobs if b["npx"] >= 20)
s2s = sorted(b["s2"] for b in init_blobs if b["npx"] >= 20)
P["s_single"] = quantile(s1s, 0.5) if s1s else 0.011
P["s2_min"] = quantile(s2s, 0.5) if s2s else 0.010
sides = []
for b in init_blobs:
    if b["cam"] == "wrist" and b["npx"] >= 20 and b["s1"] < 1.4 * P["s_single"]:
        r = top_face_yaw(b["pts"])
        if r is not None:
            sides.append(r[1])
side = quantile(sides, 0.5) if sides else 0.038
P["side"] = side
P["grasp_depth"] = min(0.022, 0.5 * side)
xs = [b["x"] for b in init_blobs]
ys = [b["y"] for b in init_blobs]
if xs:
    P["ws"] = [min(xs) - 0.15, min(ys) - 0.15, max(xs) + 0.15, max(ys) + 0.15]
P["s_single_raw"] = s1s
report["params"] = {k: (P[k] if k not in ("button",) else None) for k in P}
dets0 = []
for b in init_blobs:
    if valid_blob(b):
        dets0 += blob_to_dets(b, [])
for d in fuse(dets0):
    if button is not None and (d["x"] - button["x"]) ** 2 + (d["y"] - button["y"]) ** 2 < 0.03 ** 2:
        continue
    far = True
    for t in tracks:
        if (d["x"] - t["x"]) ** 2 + (d["y"] - t["y"]) ** 2 < 0.03 ** 2:
            far = False
    if far:
        new_track(d["x"], d["y"], "init:" + d["cams"])
log("init", n_tracks=len(tracks), cube_top=round(cube_top, 4), table_z=round(table_z, 4), side=round(side, 4),
    s_single=round(P["s_single"], 4), s2=round(P["s2_min"], 4),
    tracks=[[t["id"], round(t["x"], 3), round(t["y"], 3), t["origin"]] for t in tracks])

# ---------------------------------------------------------------- phase 1: hold still, find the highlight, track the first shuffle
w0 = grid_white(rows0, 2)
min_w = w0
white_trace = [w0]
highlight_tick = None
highlight_end = None
shuffle_seen = False
while clock["tick"] < 330 and not ended["flag"]:
    obs = tick()
    if obs is None:
        break
    rows = to_rows(obs.cameras["wrist"].rgb)
    w = grid_white(rows, 2)
    white_trace.append(w)
    if highlight_tick is None:
        if w < min_w:
            min_w = w
            rows0 = rows
        if w - min_w >= 20:
            highlight_tick = clock["tick"]
            discs, dinfo = detect_discs(rows0, rows)
            report["disc_info"] = dinfo
            wcam = obs.cameras["wrist"]
            g = cam_geom(wcam)
            depth = to_rows(wcam.depth_m)
            flagged = []
            for dsc in discs:
                u = int(round(dsc["cube_px"][0]))
                v = int(round(dsc["cube_px"][1]))
                # cube top-face pixels around the disc centre -> base xy
                pts = []
                for vv in range(max(0, v - 4), min(len(depth), v + 5)):
                    for uu in range(max(0, u - 4), min(len(depth[0]), u + 5)):
                        d = depth[vv][uu]
                        if d > 0.0:
                            q = pix_to_base(g, uu, vv, d)
                            if abs(q[2] - cube_top) < P["band"]:
                                pts.append(q)
                if not pts:
                    q = pix_to_base(g, u, v, depth[v][u])
                    pts = [q]
                cx = sum(q[0] for q in pts) / float(len(pts))
                cy = sum(q[1] for q in pts) / float(len(pts))
                bestt = None
                for t in tracks:
                    dd = math.sqrt((t["x"] - cx) ** 2 + (t["y"] - cy) ** 2)
                    if not t["hl"] and dd < 0.03 and (bestt is None or dd < bestt[0]):
                        bestt = (dd, t)
                if bestt is None:
                    t = new_track(cx, cy, "disc")
                    dist = None
                else:
                    t = bestt[1]
                    dist = round(bestt[0], 4)
                t["hl"] = True
                flagged.append([t["id"], round(cx, 3), round(cy, 3), dist, len(pts)])
            log("highlight", n_discs=len(discs), flagged=flagged)
        elif clock["tick"] > 200:
            log("no_highlight_seen")
            break
        continue
    if highlight_end is None and w - min_w < 10:
        highlight_end = clock["tick"]
    if T["moving_ticks"] >= 3:
        shuffle_seen = True
    if shuffle_seen and T["quiet_ticks"] >= 8:
        break
    if not shuffle_seen and highlight_end is not None and clock["tick"] - highlight_end > 60:
        log("no_shuffle_after_highlight")
        break
report["phase1"] = {"ticks": clock["tick"], "highlight_tick": highlight_tick, "highlight_end": highlight_end,
                    "shuffle_seen": shuffle_seen, "white_trace": white_trace[::2]}
log("phase1_done", hl=[[t["id"], round(t["x"], 3), round(t["y"], 3)] for t in tracks if t["hl"]])

# ---------------------------------------------------------------- task actions
Z_WATCH = cube_top + 0.17


def free_spot(x, y):
    """Nearest point to (x, y) at least 1.5 cube spreads clear of every other tracked cube and the button."""
    clear = 0.055
    cands = [(x, y)]
    for r in (0.03, 0.06, 0.09):
        for k in range(12):
            a = k * math.pi / 6.0
            cands.append((x + r * math.cos(a), y + r * math.sin(a)))
    for c in cands:
        ok = True
        for o in tracks:
            if o["held"]:
                continue
            if (o["x"] - c[0]) ** 2 + (o["y"] - c[1]) ** 2 < clear ** 2:
                ok = False
                break
        if ok and button is not None and (button["x"] - c[0]) ** 2 + (button["y"] - c[1]) ** 2 < 0.07 ** 2:
            ok = False
        if ok:
            return c
    return (x, y)


SCHED = {"n": -1, "fit": None}


def fit_schedule():
    """Fit a periodic shuffle schedule to the detected onsets.
    Score = onsets explained - predicted shuffles that were not observed (rejects sub-multiples
    of the true period and isolated spurious onsets). Returns (period, anchor_onset) or None."""
    on = T["onsets"]
    if SCHED["n"] == len(on):
        return SCHED["fit"]
    SCHED["n"] = len(on)
    SCHED["fit"] = None
    if len(on) < 2:
        return None
    best = None
    for per in range(60, 261, 2):
        for ref in on:
            mem = []
            for o in on:
                r = (o - ref + per / 2.0) % per - per / 2.0
                if abs(r) <= 7:
                    mem.append(o)
            if len(mem) < 2:
                continue
            expected = int(round((mem[-1] - mem[0]) / per)) + 1
            score = len(mem) - 1.5 * max(0, expected - len(mem))
            if best is None or score > best[0] + 1e-9:
                best = (score, per, mem)
    if best is None:
        return None
    mem = best[2]
    per = float(best[1])
    ks = [int(round((o - mem[0]) / per)) for o in mem]
    if ks[-1] > 0:
        n = len(mem)
        mk = sum(ks) / n
        mo = sum(mem) / n
        num = sum((ks[i] - mk) * (mem[i] - mo) for i in range(n))
        den = sum((ks[i] - mk) ** 2 for i in range(n))
        if den > 0:
            per = num / den
    SCHED["fit"] = (per, mem[-1], len(mem))
    return SCHED["fit"]


def period_estimate():
    f = fit_schedule()
    return None if f is None else round(f[0], 1)


def shuffle_duration():
    durs = sorted(T["ends"][i] - T["onsets"][i] for i in range(min(len(T["ends"]), len(T["onsets"]))))
    durs = [d for d in durs if d > 5]
    return durs[len(durs) // 2] if durs else 40


def ticks_to_onset():
    """Ticks until the next expected shuffle (onsets are detected ~6 ticks late; keep a margin)."""
    on = T["onsets"]
    if not on:
        return 10 ** 6
    now = clock["tick"]
    per = period_estimate()
    if per is None:
        # only one shuffle seen: assume the next one cannot come sooner than two shuffle durations later
        nxt = on[-1] + 2 * shuffle_duration()
        if nxt < now:
            return 10 ** 6
        return nxt - now - 8
    nxt = fit_schedule()[1] + per
    while nxt + shuffle_duration() < now:
        nxt += per
    return nxt - now - 8


def in_shuffle():
    if T["moving_ticks"] > 0:
        return True
    on = T["onsets"]
    return len(on) > len(T["ends"])


def wait_high(tag, follow=None):
    """Rise to the high watch pose (both cameras see the table) and wait until the shuffle has passed.
    When a target track is given (and nothing is held) hover above its live position."""
    t0 = clock["tick"]
    n_on = len(T["onsets"])
    x = C["pos"][0]
    y = C["pos"][1]
    seen = in_shuffle()
    while not ended["flag"] and clock["tick"] - t0 < 170:
        if follow is not None:
            x = follow["x"]
            y = follow["y"]
        servo_step((x, y, Z_HIGH), C["yaw"], vmax=0.016)
        if len(T["onsets"]) > n_on or in_shuffle():
            seen = True
        if seen and not in_shuffle() and T["quiet_ticks"] >= 6:
            break
        if not seen and ticks_to_onset() > 60 and clock["tick"] - t0 > 10:
            break
    log("wait_high:" + tag, ticks=clock["tick"] - t0, seen=seen, onsets=list(T["onsets"]), per=period_estimate())


def press(tag):
    if button is None or ended["flag"]:
        log("press_skipped:" + tag)
        return False
    bx = button["x"]
    by = button["y"]
    bz = button["z_top"]
    ok = False
    for attempt in range(3):
        if ended["flag"]:
            break
        if in_shuffle() or ticks_to_onset() < 45:
            wait_high("press_" + tag)
        C["grip"] = 0.0
        go_to((bx, by, bz + 0.05), 0.0, vmax=0.014, tol=0.008, name="press_above_" + tag)
        go_to((bx, by, bz - 0.015), 0.0, vmax=0.006, tol=0.002, settle=8, max_ticks=20, name="press_down_" + tag)
        zmin = T["ee"][2] if not ended["flag"] else None
        go_to((bx, by, Z_WATCH), 0.0, vmax=0.014, tol=0.02, settle=0, name="press_up_" + tag)
        ok = zmin is not None and zmin < bz - 0.002
        log("press:" + tag, z_min=round(zmin, 4) if zmin is not None else None, ok=ok, attempt=attempt)
        if ok or ended["flag"]:
            break
    C["grip"] = 1.0
    return ok


def target_yaw(t):
    w = t["wrist"]
    if w is None or len(w["pts"]) < 12:
        return None
    r = top_face_yaw(w["pts"])
    if r is None:
        return None
    return wrap45(r[0] - phi_tool)


GRASP_NEED = 34
RELEASE_NEED = 30


def pick(t):
    """Follow the live track, grasp in a quiet window, lift, set down (after any shuffle), release."""
    T["active"] = t
    attempts = 0
    result_state = "not_attempted"
    while attempts < 5 and not ended["flag"]:
        if clock["tick"] > max_steps_total - 150:
            result_state = "out_of_time"
            break
        if in_shuffle() or ticks_to_onset() < GRASP_NEED + 4:
            wait_high("before_grasp", follow=t)
            continue
        # approach and hover over the live target until it is centred and quiet
        k = 0
        ready = False
        while k < 60 and not ended["flag"]:
            servo_step((t["x"], t["y"], Z_WATCH), C["yaw"], vmax=0.016)
            k += 1
            if in_shuffle() or ticks_to_onset() < GRASP_NEED:
                break
            dxy = math.sqrt((T["ee"][0] - t["x"]) ** 2 + (T["ee"][1] - t["y"]) ** 2)
            if dxy < 0.008 and abs(T["ee"][2] - Z_WATCH) < 0.02 and T["quiet_ticks"] >= 4 and t["miss"] == 0:
                ready = True
                break
        if not ready:
            attempts += 1
            continue
        yaw = target_yaw(t)
        if yaw is None:
            yaw = C["yaw"]
        x0 = t["x"]
        y0 = t["y"]
        grasp_z = cube_top - P["grasp_depth"]
        aborted = False
        k = 0
        while k < 60 and not ended["flag"]:
            if (T["moving_ticks"] >= 1 or t.get("moving")) and C["pos"][2] > grasp_z + 0.025:
                aborted = True
                break
            vmax = 0.014 if C["pos"][2] > cube_top + 0.06 else 0.007
            n = servo_step((x0, y0, grasp_z), yaw, vmax=vmax)
            k += 1
            if n < 1e-4 and abs(T["ee"][2] - grasp_z) < 0.004:
                break
        if aborted:
            log("descent_aborted", id=t["id"], z=round(T["ee"][2], 3))
            go_to((t["x"], t["y"], Z_WATCH), C["yaw"], vmax=0.014, tol=0.03, settle=0)
            attempts += 1
            continue
        if ended["flag"]:
            break
        g_closed = set_grip(0.0, 8)
        if g_closed is None or g_closed < 0.05:
            log("grasp_empty", id=t["id"], g=g_closed, ee=[round(v, 4) for v in T["ee"]],
                track=[round(t["x"], 4), round(t["y"], 4)])
            set_grip(1.0, 6)
            go_to((t["x"], t["y"], Z_WATCH), C["yaw"], vmax=0.014, tol=0.03, settle=0)
            attempts += 1
            result_state = "empty"
            continue
        t["held"] = True
        gx = C["pos"][0]
        gy = C["pos"][1]
        go_to((gx, gy, grasp_z + 0.08), C["yaw"], vmax=0.012, tol=0.01, settle=4, name="lift")
        g_lift = gripper_reading()
        log("lifted", id=t["id"], g_closed=round(g_closed, 3), g_lift=round(g_lift, 3) if g_lift is not None else None,
            to_onset=ticks_to_onset())
        if in_shuffle() or ticks_to_onset() < RELEASE_NEED:
            wait_high("holding")
            gx, gy = free_spot(gx, gy)
            go_to((gx, gy, grasp_z + 0.08), C["yaw"], vmax=0.014, tol=0.01, settle=2)
        go_to((gx, gy, grasp_z + 0.004), C["yaw"], vmax=0.012, tol=0.004, settle=6, name="lower")
        set_grip(1.0, 6)
        t["held"] = False
        t["x"] = gx
        t["y"] = gy
        t["vx"] = 0.0
        t["vy"] = 0.0
        t["miss"] = 0
        t["hist"] = []
        go_to((gx, gy, Z_WATCH), C["yaw"], vmax=0.014, tol=0.03, settle=0)
        result_state = "lifted" if (g_lift is not None and g_lift > 0.05) else "dropped"
        break
    T["active"] = None
    return result_state, attempts


Z_HIGH = max(Z_WATCH + 0.12, min(home_pos[2] - 0.05, cube_top + 0.38))
report["z_watch"] = Z_WATCH
report["z_high"] = Z_HIGH


press_ok = press("start")
picks = []
while not ended["flag"]:
    todo = [t for t in tracks if t["hl"] and not t["done"]]
    if not todo:
        break
    if clock["tick"] > max_steps_total - 150:
        log("time_limit_before_picks_done", remaining=[t["id"] for t in todo])
        break
    ee = T["ee"]
    todo.sort(key=lambda t: (t["x"] - ee[0]) ** 2 + (t["y"] - ee[1]) ** 2)
    t = todo[0]
    st, att = pick(t)
    if st in ("lifted", "dropped") or att >= 5 or st == "out_of_time":
        t["done"] = True
    picks.append({"id": t["id"], "state": st, "attempts": att, "tick": clock["tick"]})
    log("pick_done", id=t["id"], state=st, attempts=att)
    if len(picks) > 3 * len([q for q in tracks if q["hl"]]) + 3:
        log("too_many_pick_rounds")
        break
report["picks"] = picks
report["press_start_ok"] = press_ok
report["press_stop_ok"] = press("stop")
report["onsets"] = T["onsets"]
report["ends"] = T["ends"]
for k in range(5):
    if ended["flag"]:
        break
    tick()
report["tracks"] = [{k: (round(t[k], 4) if k in ("x", "y", "vx", "vy") else t[k]) for k in
                     ("id", "x", "y", "hl", "done", "seen", "miss", "origin", "last_seen")} for t in tracks]
report["motion_log"] = T["motion_log"]
report["n_det"] = T["n_det"][::5]
report["ik_fail"] = C["ik_fail"]
report["ticks"] = clock["tick"]
report["ended_by_env"] = ended["flag"]
result = report
