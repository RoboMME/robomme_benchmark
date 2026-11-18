# ButtonUnmaskSwap v00_initial -- ASPIRE task worker, robomme_plus task_2.
#
# Strategy (from the released recordings; every task-dependent quantity is perceived live):
#   * At execution start the coloured cubes are visible; a few ticks later identical white
#     containers drop over them, then the containers keep moving/swapping for a long time.
#   * The program therefore runs its OWN per-tick control loop (step() one tick at a time)
#     and re-perceives every tick: front + wrist depth are back-projected into robot_base and
#     the container tops are tracked with a gated nearest-centre (warm-started k-means) tracker.
#   * Goal colours are parsed from get_task_context().language; each goal cube is bound to the
#     container that dropped onto it; buttons are found with SAM3 ("button") + depth.
#   * Press both buttons (closed gripper) early, park the wrist camera above the containers,
#     then lift the tracked containers in goal order (grasp, lift, put back).
# Heavily instrumented: this single run is also the scene probe.
import math

report = {"version": "v00_initial", "errors": [], "events": [], "phases": [], "track_log": []}
S = {
    "done": False, "ticks": 0, "phase": "pre", "cams": {}, "tracks": [], "held": -1,
    "table_z": None, "top_z": None, "cubes": {}, "bg": None, "cand_top": None,
    "box": None, "frames": 0, "frame_errors": 0, "last_ts": None, "goal_map": {},
    "q": None, "qcmd": None, "ee": None, "ee_q": None, "grip": None, "emb": None,
    "max_steps": 1300, "hz": 20.0, "ik_calls": 0, "ik_fail": 0, "last_assign": {},
    "near_events": 0, "width": 0.05,
}

DQ_FAST = 0.05
DQ_SLOW = 0.03
COLOR_WORDS = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "black",
               "white", "gray", "grey", "cyan", "brown", "magenta"]


def note(kind, data):
    if len(report["events"]) < 500:
        report["events"].append({"tick": S["ticks"], "kind": kind, "data": data})


def err(where, e):
    if len(report["errors"]) < 80:
        report["errors"].append(where + " @tick " + str(S["ticks"]) + ": " + str(e)[:400])


def r3(v):
    return round(float(v), 3)


def pct(vals, f):
    if len(vals) == 0:
        return None
    o = sorted(vals)
    i = int((len(o) - 1) * f)
    return o[i]


def to_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        pass
    rows = []
    for r in range(len(arr)):
        rows.append(list(arr[r]))
    return rows


def rotmat(q):
    w = float(q[0]); x = float(q[1]); y = float(q[2]); z = float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    w = w / n; x = x / n; y = y / n; z = z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def qz(angle):
    return [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]


def vec3(v):
    return [float(v[0]), float(v[1]), float(v[2])]


# ---------------------------------------------------------------- state / terminal helpers
def read_state(state):
    try:
        S["q"] = [float(v) for v in state.joint_positions["primary"]]
    except Exception as e:
        err("read_state.q", e)
    try:
        pose = state.end_effector_poses["primary"]
        S["ee"] = vec3(pose.position)
        S["ee_q"] = [float(v) for v in pose.quaternion_wxyz]
    except Exception as e:
        err("read_state.ee", e)
    try:
        S["grip"] = float(state.gripper_positions["primary"])
    except Exception as e:
        err("read_state.grip", e)


def terminal_flags(res):
    flags = {}
    try:
        flags["terminated"] = bool(res.terminated)
    except Exception:
        pass
    try:
        flags["truncated"] = bool(res.truncated)
    except Exception:
        pass
    try:
        flags["done"] = bool(res.done)
    except Exception:
        pass
    term = False
    for k in flags:
        if flags[k]:
            term = True
    return term, flags


def res_status(res):
    out = {}
    try:
        out["ok"] = bool(res.ok)
    except Exception:
        pass
    try:
        out["status"] = str(res.status)
    except Exception:
        pass
    try:
        if res.error is not None:
            out["error"] = str(res.error)[:200]
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- camera geometry
def cam_dirs(variant, a, b):
    conv = variant[0]
    if conv == "cv":
        d = (a, b, 1.0)
    elif conv == "gl":
        d = (a, -b, -1.0)
    else:
        d = (1.0, -a, -b)
    if variant[1] == "ray":
        n = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        d = (d[0] / n, d[1] / n, d[2] / n)
    return d


def cam_basic(cam):
    K = cam.intrinsics
    fx = float(K[0][0]); fy = float(K[1][1]); cx = float(K[0][2]); cy = float(K[1][2])
    p = vec3(cam.camera_pose.position)
    q = [float(v) for v in cam.camera_pose.quaternion_wxyz]
    return fx, fy, cx, cy, p, q


def calibrate(name, cam, D):
    fx, fy, cx, cy, p, q = cam_basic(cam)
    H = len(D); W = len(D[0])
    R = rotmat(q)
    best = None
    scores = []
    for variant in (("cv", "z"), ("gl", "z"), ("ros", "z"), ("cv", "ray"), ("gl", "ray"), ("ros", "ray")):
        zs = []
        for r in range(2, H, 6):
            row = D[r]
            for c in range(2, W, 6):
                d = float(row[c])
                if not (d > 0.02 and d < 8.0):
                    continue
                dc = cam_dirs(variant, (c - cx) / fx, (r - cy) / fy)
                wz = R[2][0] * dc[0] + R[2][1] * dc[1] + R[2][2] * dc[2]
                zs.append(p[2] + d * wz)
        if len(zs) < 50:
            scores.append([variant[0], variant[1], 0.0, None])
            continue
        hist = {}
        for z in zs:
            k = int(math.floor(z / 0.004))
            hist[k] = hist.get(k, 0) + 1
        bk = None; bn = -1
        for k in hist:
            n = hist[k] + hist.get(k - 1, 0) + hist.get(k + 1, 0)
            if n > bn:
                bn = n; bk = k
        tz_vals = [z for z in zs if abs(z - (bk + 0.5) * 0.004) < 0.006]
        tz = sum(tz_vals) / max(1, len(tz_vals))
        frac = float(bn) / len(zs)
        below = tz < p[2] - 0.05
        scores.append([variant[0], variant[1], round(frac, 3), r3(tz), below])
        key = frac if below else frac - 1.0
        if best is None or key > best[0]:
            best = [key, variant, tz, frac]
    info = {"camera": name, "scores": scores, "chosen": [best[1][0], best[1][1]],
            "table_z": r3(best[2]), "flat_frac": round(best[3], 3), "cam_pos": [r3(v) for v in p],
            "fx": r3(fx), "fy": r3(fy), "cx": r3(cx), "cy": r3(cy), "H": H, "W": W}
    note("calibrate", info)
    return best[1], best[2], info


def setup_cam(name, cam, D, stride):
    if name in S["cams"]:
        return S["cams"][name]
    variant, tz, info = calibrate(name, cam, D)
    fx, fy, cx, cy, p, q = cam_basic(cam)
    H = len(D); W = len(D[0])
    pix = []
    dirs = []
    for r in range(0, H, stride):
        for c in range(0, W, stride):
            pix.append((r, c))
            dirs.append(cam_dirs(variant, (c - cx) / fx, (r - cy) / fy))
    st = {"variant": variant, "table_z": tz, "pix": pix, "dirs": dirs, "stride": stride,
          "world_rays": None, "ray_pose": None, "sel": None, "H": H, "W": W}
    S["cams"][name] = st
    if name == "front" or S["table_z"] is None:
        S["table_z"] = tz
    return st


def world_rays(st, p, q):
    R = rotmat(q)
    rays = []
    for d in st["dirs"]:
        rays.append((R[0][0] * d[0] + R[0][1] * d[1] + R[0][2] * d[2],
                     R[1][0] * d[0] + R[1][1] * d[1] + R[1][2] * d[2],
                     R[2][0] * d[0] + R[2][1] * d[1] + R[2][2] * d[2]))
    return rays


def select_front_pixels(st, p):
    # keep sampled pixels whose ray meets the table plane inside the workspace box
    box = S["box"]
    tz = S["table_z"]
    sel = []
    rays = st["world_rays"]
    for i in range(len(rays)):
        w = rays[i]
        if abs(w[2]) < 1e-6:
            continue
        t = (tz - p[2]) / w[2]
        if t <= 0:
            continue
        x = p[0] + t * w[0]; y = p[1] + t * w[1]
        if box[0] - 0.05 <= x <= box[1] + 0.05 and box[2] - 0.05 <= y <= box[3] + 0.05:
            sel.append(i)
    st["sel"] = sel
    note("front_roi", {"selected": len(sel), "of": len(rays), "box": [r3(v) for v in box]})


def ee_box():
    if S["ee"] is None or S["ee_q"] is None:
        return None
    R = rotmat(S["ee_q"])
    ay = (R[0][1], R[1][1])
    n = math.sqrt(ay[0] * ay[0] + ay[1] * ay[1])
    if n < 0.2:
        ay = (0.0, 1.0)
    else:
        ay = (ay[0] / n, ay[1] / n)
    g = S["grip"] if S["grip"] is not None else 1.0
    g = max(0.0, min(1.0, g))
    return (S["ee"][0], S["ee"][1], S["ee"][2], ay[0], ay[1], 0.04 * g)


def in_hand(x, y, z, hb):
    # hand body: above EE+0.04 (EE = Panda hand + 0.1 m); fingers: two pads at +-offset along the
    # finger axis between EE-0.02 and EE+0.04.  The space between open fingers stays visible.
    if hb is None or z < hb[2] - 0.02:
        return False
    dx = x - hb[0]; dy = y - hb[1]
    along = dx * hb[3] + dy * hb[4]
    perp = -dx * hb[4] + dy * hb[3]
    if z > hb[2] + 0.04:
        return abs(along) < 0.115 and abs(perp) < 0.06
    a = abs(along)
    return abs(perp) < 0.022 and a > hb[5] - 0.012 and a < hb[5] + 0.028


def cam_points(name, cam, zlo, want_rgb, hb):
    """Back-project one camera into robot_base. Returns list of (x,y,z[,rgb])."""
    D = to_rows(cam.depth_m)
    stride = 2 if name == "front" else 4
    st = setup_cam(name, cam, D, stride)
    fx, fy, cx, cy, p, q = cam_basic(cam)
    if name == "front":
        if st["ray_pose"] is None or abs(st["ray_pose"][0] - p[0]) + abs(st["ray_pose"][1] - p[1]) + abs(st["ray_pose"][2] - p[2]) > 1e-4:
            st["world_rays"] = world_rays(st, p, q)
            st["ray_pose"] = p
            st["sel"] = None
            if S["box"] is not None:
                select_front_pixels(st, p)
        rays = st["world_rays"]
    else:
        rays = world_rays(st, p, q)
    idx = st["sel"]
    if idx is None:
        idx = range(len(rays))
    RGB = None
    if want_rgb:
        RGB = to_rows(cam.rgb)
    pix = st["pix"]
    px = p[0]; py = p[1]; pz = p[2]
    box = S["box"]
    out = []
    near_lim = 0.12 if name == "wrist" else 0.05
    for i in idx:
        rc = pix[i]
        d = D[rc[0]][rc[1]]
        if not (d > near_lim and d < 6.0):
            continue
        w = rays[i]
        z = pz + d * w[2]
        if z < zlo:
            continue
        x = px + d * w[0]
        y = py + d * w[1]
        if box is not None and (x < box[0] or x > box[1] or y < box[2] or y > box[3]):
            continue
        if hb is not None and in_hand(x, y, z, hb):
            continue
        if want_rgb:
            out.append((x, y, z, RGB[rc[0]][rc[1]]))
        else:
            out.append((x, y, z))
    return out


# ---------------------------------------------------------------- colour + clustering
def color_name(rgb):
    r = float(rgb[0]); g = float(rgb[1]); b = float(rgb[2])
    mx = max(r, g, b); mn = min(r, g, b)
    if mx < 70:
        return "black"
    s = (mx - mn) / mx
    if s < 0.16:
        if mx > 150:
            return "white"
        return "gray"
    if s < 0.42:
        return None
    if mx == r:
        h = 60.0 * (((g - b) / (mx - mn)) % 6.0)
    elif mx == g:
        h = 60.0 * ((b - r) / (mx - mn) + 2.0)
    else:
        h = 60.0 * ((r - g) / (mx - mn) + 4.0)
    if h < 12 or h >= 340:
        return "red"
    if h < 40:
        return "orange"
    if h < 72:
        return "yellow"
    if h < 165:
        return "green"
    if h < 195:
        return "cyan"
    if h < 255:
        return "blue"
    if h < 300:
        return "purple"
    return "pink"


def components(pts, cell):
    cells = {}
    for i in range(len(pts)):
        k = (int(math.floor(pts[i][0] / cell)), int(math.floor(pts[i][1] / cell)))
        if k in cells:
            cells[k].append(i)
        else:
            cells[k] = [i]
    seen = set()
    comps = []
    for k in cells:
        if k in seen:
            continue
        stack = [k]
        seen.add(k)
        members = []
        while len(stack) > 0:
            c = stack.pop()
            members.extend(cells[c])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nk = (c[0] + dx, c[1] + dy)
                    if nk in cells and nk not in seen:
                        seen.add(nk)
                        stack.append(nk)
        sx = 0.0; sy = 0.0
        for i in members:
            sx += pts[i][0]; sy += pts[i][1]
        comps.append({"x": sx / len(members), "y": sy / len(members), "n": len(members), "idx": members})
    return comps


def min_area_yaw(pts):
    best = None
    for k in range(18):
        a = math.radians(5.0 * k)
        ca = math.cos(a); sa = math.sin(a)
        us = []; vs = []
        for p in pts:
            us.append(p[0] * ca + p[1] * sa)
            vs.append(-p[0] * sa + p[1] * ca)
        eu = pct(us, 0.95) - pct(us, 0.05)
        ev = pct(vs, 0.95) - pct(vs, 0.05)
        if best is None or eu * ev < best[0]:
            best = [eu * ev, a, eu, ev]
    return best


# ---------------------------------------------------------------- per-frame perception
def detect_cubes(pts):
    tz = S["table_z"]
    by_color = {}
    for p in pts:
        if p[2] > tz + 0.07:
            continue
        cname = color_name(p[3])
        if cname is None:
            continue
        if cname in by_color:
            by_color[cname].append(p)
        else:
            by_color[cname] = [p]
    found = {}
    for cname in by_color:
        comps = components(by_color[cname], 0.01)
        good = [c for c in comps if c["n"] >= 3]
        if len(good) == 0:
            continue
        lst = []
        for c in good:
            zs = [by_color[cname][i][2] for i in c["idx"]]
            lst.append({"x": c["x"], "y": c["y"], "n": c["n"], "ztop": pct(zs, 0.9)})
        lst = sorted([(-c["n"], i, c) for i, c in enumerate(lst)])
        found[cname] = [t[2] for t in lst]
    return found


def bg_key(x, y):
    return (int(math.floor(x / 0.01)), int(math.floor(y / 0.01)))


def process_pre(front, wrist):
    tz = S["table_z"]
    allp = front + wrist
    cubes = detect_cubes(allp)
    for cname in cubes:
        S["cubes"][cname] = cubes[cname]
    if S["bg"] is None:
        bg = {}
        for p in front:
            if p[2] > tz + 0.012:
                k = bg_key(p[0], p[1])
                if p[2] > bg.get(k, -9.0):
                    bg[k] = p[2]
        S["bg"] = bg
        summary = {}
        for cname in cubes:
            summary[cname] = [[r3(c["x"]), r3(c["y"]), c["n"], r3(c["ztop"])] for c in cubes[cname][:3]]
        note("first_pre_frame", {"cubes": summary, "bg_cells": len(bg), "n_front": len(front), "n_wrist": len(wrist)})
        return
    bg = S["bg"]
    newp = []
    for p in allp:
        if p[2] < tz + 0.03 or p[2] > tz + 0.30:
            continue
        k = bg_key(p[0], p[1])
        if k in bg and p[2] < bg[k] + 0.02:
            continue
        newp.append(p)
    nfront = 0
    for p in front:
        if p[2] < tz + 0.03 or p[2] > tz + 0.30:
            continue
        k = bg_key(p[0], p[1])
        if k in bg and p[2] < bg[k] + 0.02:
            continue
        nfront += 1
    if nfront < 80:
        S["cand_top"] = None
        return
    top = pct([p[2] for p in newp], 0.92)
    if S["cand_top"] is None or abs(S["cand_top"] - top) > 0.006:
        S["cand_top"] = top
        note("container_candidate", {"top_z": r3(top), "n_new_front": nfront, "n_new": len(newp)})
        return
    init_tracks(newp, top)


def band_points(pts, hb):
    # per-camera tall-cell exclusion + band selection
    tall = set()
    thr = S["top_z"] + 0.035
    for p in pts:
        if p[2] > thr and p[2] < S["table_z"] + 0.8:
            tall.add((int(math.floor(p[0] / 0.02)), int(math.floor(p[1] / 0.02))))
    lo = S["band_lo"]; hi = S["band_hi"]
    out = []
    for p in pts:
        if p[2] < lo or p[2] > hi:
            continue
        kx = int(math.floor(p[0] / 0.02)); ky = int(math.floor(p[1] / 0.02))
        blocked = False
        if len(tall) > 0:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if (kx + dx, ky + dy) in tall:
                        blocked = True
        if not blocked:
            out.append((p[0], p[1], p[2]))
    return out, len(tall)


def init_tracks(newp, top):
    tz = S["table_z"]
    Hc = top - tz
    S["top_z"] = top
    S["band_lo"] = top - max(0.012, 0.3 * Hc)
    S["band_hi"] = top + 0.015
    band = [p for p in newp if S["band_lo"] <= p[2] <= S["band_hi"]]
    comps = components(band, 0.008)
    comps = [c for c in comps if c["n"] >= 5]
    # seeds: saturated-colour cubes first (containers drop onto them), then unexplained components
    seeds = []
    added = 0
    for cname in S["cubes"]:
        if cname in ("white", "gray"):
            continue
        for cube in S["cubes"][cname]:
            if cube["n"] < 4:
                continue
            dup = False
            for sd in seeds:
                if (sd[0] - cube["x"]) ** 2 + (sd[1] - cube["y"]) ** 2 < 0.03 ** 2:
                    dup = True
            if not dup:
                seeds.append([cube["x"], cube["y"]])
                added += 1
    order = sorted([(-c["n"], i) for i, c in enumerate(comps)])
    for negn, i in order:
        c = comps[i]
        dup = False
        for sd in seeds:
            if (sd[0] - c["x"]) ** 2 + (sd[1] - c["y"]) ** 2 < 0.035 ** 2:
                dup = True
        if not dup:
            seeds.append([c["x"], c["y"]])
    n_comp = len(comps)
    merged_pairs = 0
    for rnd in range(3):
        # refine with gated k-means
        for it in range(4):
            acc = [[0.0, 0.0, 0] for sd in seeds]
            for p in band:
                bi = -1; bd = 0.045 ** 2
                for k in range(len(seeds)):
                    d2 = (p[0] - seeds[k][0]) ** 2 + (p[1] - seeds[k][1]) ** 2
                    if d2 < bd:
                        bd = d2; bi = k
                if bi >= 0:
                    acc[bi][0] += p[0]; acc[bi][1] += p[1]; acc[bi][2] += 1
            keep = []
            for k in range(len(seeds)):
                if acc[k][2] >= 4:
                    keep.append([acc[k][0] / acc[k][2], acc[k][1] / acc[k][2]])
                elif rnd == 0:
                    keep.append(seeds[k])
            seeds = keep
        # merge duplicates (one container split into two seeds)
        out = []
        for sd in seeds:
            hit = -1
            for j in range(len(out)):
                if (out[j][0] - sd[0]) ** 2 + (out[j][1] - sd[1]) ** 2 < 0.03 ** 2:
                    hit = j
            if hit >= 0:
                out[hit] = [(out[hit][0] + sd[0]) / 2.0, (out[hit][1] + sd[1]) / 2.0]
                merged_pairs += 1
            else:
                out.append(sd)
        seeds = out
    tracks = []
    for k in range(len(seeds)):
        tracks.append({"id": k, "x": seeds[k][0], "y": seeds[k][1], "vx": 0.0, "vy": 0.0,
                       "n": 0, "miss": 0, "spd": 0.0, "x0": seeds[k][0], "y0": seeds[k][1]})
    S["tracks"] = tracks
    # container width from components
    widths = []
    for sd in seeds:
        own = [p for p in band if (p[0] - sd[0]) ** 2 + (p[1] - sd[1]) ** 2 < 0.04 ** 2]
        if len(own) >= 10:
            mm = min_area_yaw(own)
            widths.append(min(mm[2], mm[3]))
    if len(widths) > 0:
        S["width"] = pct(widths, 0.5)
    # workspace box
    xs = [t["x"] for t in tracks]; ys = [t["y"] for t in tracks]
    if len(xs) > 0:
        S["box"] = [min(xs) - 0.25, max(xs) + 0.25, min(ys) - 0.25, max(ys) + 0.25]
        if "front" in S["cams"] and S["cams"]["front"]["ray_pose"] is not None:
            select_front_pixels(S["cams"]["front"], S["cams"]["front"]["ray_pose"])
    # bind goal colours
    gm = {}
    for cname in S["goal_colors"]:
        if cname not in S["cubes"]:
            note("goal_cube_missing", {"color": cname})
            continue
        cube = S["cubes"][cname][0]
        bi = -1; bd = 0.06 ** 2
        for t in tracks:
            d2 = (t["x"] - cube["x"]) ** 2 + (t["y"] - cube["y"]) ** 2
            if d2 < bd:
                bd = d2; bi = t["id"]
        if bi >= 0:
            gm[cname] = bi
        note("bind_goal", {"color": cname, "cube": [r3(cube["x"]), r3(cube["y"])], "n_cube_clusters": len(S["cubes"][cname]),
                           "track": bi, "dist": r3(math.sqrt(bd)) if bi >= 0 else None})
    S["goal_map"] = gm
    S["phase"] = "tracking"
    S["init_tick"] = S["ticks"]
    cube_summary = {}
    for cname in S["cubes"]:
        cube_summary[cname] = [[r3(c["x"]), r3(c["y"]), c["n"]] for c in S["cubes"][cname][:3]]
    note("tracks_init", {"top_z": r3(top), "table_z": r3(tz), "height": r3(Hc), "band": [r3(S["band_lo"]), r3(S["band_hi"])],
                         "n_components": n_comp, "seeds_from_cubes": added, "merged": merged_pairs, "n_tracks": len(tracks),
                         "tracks": [[t["id"], r3(t["x"]), r3(t["y"])] for t in tracks],
                         "width": r3(S["width"]), "goal_map": gm, "cubes": cube_summary,
                         "n_band": len(band)})


def update_tracks(pts, dt):
    T = S["tracks"]
    if len(T) == 0:
        return
    gate = max(0.03, min(0.06, 0.75 * S["width"]))
    preds = []
    for t in T:
        if S["held"] == t["id"] and S["ee"] is not None:
            preds.append((S["ee"][0], S["ee"][1]))
        else:
            preds.append((t["x"] + t["vx"] * dt, t["y"] + t["vy"] * dt))
    assign = [[] for t in T]
    unassigned = []
    g2 = gate * gate
    nT = len(T)
    held_k = -1
    for k in range(nT):
        if S["held"] == T[k]["id"]:
            held_k = k
    for p in pts:
        bi = -1; bd = g2
        for k in range(nT):
            dx = p[0] - preds[k][0]; dy = p[1] - preds[k][1]
            d2 = dx * dx + dy * dy
            if d2 < bd:
                bd = d2; bi = k
        if bi < 0 and held_k >= 0:
            # anything near the gripper while holding belongs to the held container
            dx = p[0] - preds[held_k][0]; dy = p[1] - preds[held_k][1]
            if dx * dx + dy * dy < (2.0 * gate) ** 2:
                bi = held_k
        if bi >= 0:
            assign[bi].append(p)
        else:
            unassigned.append(p)
    # re-acquisition for lost tracks
    for k in range(nT):
        t = T[k]
        if len(assign[k]) >= 4 or t["miss"] == 0 or S["held"] == t["id"]:
            continue
        g2b = gate + min(0.06, 0.004 * t["miss"])
        cand = [p for p in unassigned if (p[0] - preds[k][0]) ** 2 + (p[1] - preds[k][1]) ** 2 < g2b * g2b]
        if len(cand) >= 6:
            mx = sum(p[0] for p in cand) / len(cand); my = sum(p[1] for p in cand) / len(cand)
            got = [p for p in unassigned if (p[0] - mx) ** 2 + (p[1] - my) ** 2 < g2]
            if len(got) >= 6:
                assign[k] = got
                unassigned = [p for p in unassigned if (p[0] - mx) ** 2 + (p[1] - my) ** 2 >= g2]
                note("reacquire", {"track": t["id"], "miss": t["miss"],
                                   "jump": r3(math.sqrt((mx - preds[k][0]) ** 2 + (my - preds[k][1]) ** 2))})
    S["last_assign"] = {}
    for k in range(nT):
        t = T[k]
        pts_k = assign[k]
        S["last_assign"][t["id"]] = pts_k
        if S["held"] == t["id"]:
            t["x"] = preds[k][0]; t["y"] = preds[k][1]; t["vx"] = 0.0; t["vy"] = 0.0
            t["n"] = len(pts_k)
            continue
        if len(pts_k) >= 4:
            mx = sum(p[0] for p in pts_k) / len(pts_k)
            my = sum(p[1] for p in pts_k) / len(pts_k)
            vx = (mx - t["x"]) / dt; vy = (my - t["y"]) / dt
            t["vx"] = 0.5 * t["vx"] + 0.5 * vx
            t["vy"] = 0.5 * t["vy"] + 0.5 * vy
            sp = math.sqrt(t["vx"] ** 2 + t["vy"] ** 2)
            if sp > 0.012:
                t["vx"] = t["vx"] * 0.012 / sp; t["vy"] = t["vy"] * 0.012 / sp
            t["spd"] = 0.6 * t["spd"] + 0.4 * math.sqrt(vx * vx + vy * vy)
            t["x"] = mx; t["y"] = my; t["n"] = len(pts_k); t["miss"] = 0
        else:
            t["x"] = preds[k][0]; t["y"] = preds[k][1]
            damp = 0.6 ** dt
            t["vx"] = t["vx"] * damp; t["vy"] = t["vy"] * damp
            t["n"] = len(pts_k); t["miss"] += 1
    # proximity (identity-risk) events
    for a in range(nT):
        for b in range(a + 1, nT):
            d = math.sqrt((T[a]["x"] - T[b]["x"]) ** 2 + (T[a]["y"] - T[b]["y"]) ** 2)
            if d < 0.6 * S["width"]:
                S["near_events"] += 1
                if S["near_events"] <= 40:
                    note("tracks_close", {"a": T[a]["id"], "b": T[b]["id"], "d": r3(d)})
    S["unassigned_last"] = len(unassigned)


def process_frame(obs, source):
    S["frames"] += 1
    try:
        state = obs.robot_state
        read_state(state)
        try:
            ts = float(obs.timestamp_s)
        except Exception:
            ts = None
        dt = 1.0
        if ts is not None and S["last_ts"] is not None:
            dt = (ts - S["last_ts"]) * S["hz"]
            if dt <= 0.05:
                dt = 1.0
        if ts is not None:
            S["last_ts"] = ts
        cams = obs.cameras
        hb = ee_box()
        if S["phase"] == "pre":
            zlo = (S["table_z"] if S["table_z"] is not None else -9.0) + 0.004
            front = cam_points("front", cams["front"], zlo, True, hb)
            try:
                wrist = cam_points("wrist", cams["wrist"], zlo, True, hb)
            except Exception as e:
                wrist = []
                err("wrist_pre", e)
            if S["frames"] == 1:
                # first call calibrated; re-run with the now-known table height
                zlo = S["table_z"] + 0.004
                front = cam_points("front", cams["front"], zlo, True, hb)
                try:
                    wrist = cam_points("wrist", cams["wrist"], zlo, True, hb)
                except Exception as e:
                    wrist = []
            process_pre(front, wrist)
        else:
            zlo = S["band_lo"]
            front = cam_points("front", cams["front"], zlo, False, hb)
            try:
                wrist = cam_points("wrist", cams["wrist"], zlo, False, hb)
            except Exception as e:
                wrist = []
                if S["frame_errors"] < 5:
                    err("wrist_track", e)
            bf, tf = band_points(front, hb)
            bw, tw = band_points(wrist, hb)
            update_tracks(bf + bw, dt)
            if S["ticks"] % 10 == 0 or source == "video":
                if len(report["track_log"]) < 260:
                    eez = r3(S["ee"][2]) if S["ee"] is not None else None
                    report["track_log"].append([S["ticks"], source, eez, len(bf), len(bw), tf, tw,
                                                [[t["id"], r3(t["x"]), r3(t["y"]), t["n"], t["miss"]] for t in S["tracks"]]])
    except Exception as e:
        S["frame_errors"] += 1
        if S["frame_errors"] <= 8:
            err("process_frame(" + source + ")", e)


# ---------------------------------------------------------------- control
def tick(q_cmd, grip):
    if S["done"]:
        return False
    if S["ticks"] >= S["max_steps"] - 2:
        S["done"] = True
        note("step_budget_exhausted", {"ticks": S["ticks"]})
        return False
    try:
        action = RobotAction(arms={"primary": ArmCommand(mode="joint_position", target=[float(v) for v in q_cmd],
                                                         gripper_position=float(grip), embodiment=S["emb"])})
        res = step(action)
    except Exception as e:
        err("step", e)
        S["done"] = True
        return False
    S["ticks"] += 1
    term, flags = terminal_flags(res)
    if S["ticks"] <= 2:
        note("step_result", {"flags": flags, "status": res_status(res)})
    if term:
        S["done"] = True
        S["terminal"] = {"tick": S["ticks"], "flags": flags, "status": res_status(res), "phase": S.get("ctl_phase")}
        note("terminal", S["terminal"])
        return False
    obs = None
    try:
        obs = res.observation
    except Exception:
        obs = None
    if obs is None:
        try:
            obs = get_observation()
        except Exception as e:
            err("get_observation", e)
            return True
    process_frame(obs, "live")
    return True


def ik_solve(pos, quat, seed):
    S["ik_calls"] += 1
    try:
        ik = solve_ik(Pose([float(pos[0]), float(pos[1]), float(pos[2])], [float(v) for v in quat], S["base"]),
                      seed_joints=[float(v) for v in seed])
    except Exception as e:
        err("solve_ik", e)
        S["ik_fail"] += 1
        return None
    ok = False
    try:
        ok = bool(ik.ok)
    except Exception:
        ok = False
    if not ok:
        S["ik_fail"] += 1
        if S["ik_fail"] <= 6:
            note("ik_fail", {"pos": [r3(v) for v in pos], "detail": str(ik)[:300]})
        return None
    sol = None
    try:
        sol = [float(v) for v in ik.joint_positions]
    except Exception:
        sol = None
    if sol is None:
        try:
            sol = [float(v) for v in ik.solution]
        except Exception:
            sol = None
    if sol is None:
        try:
            sol = [float(v) for v in ik.joints]
        except Exception:
            sol = None
    if sol is None and S["ik_calls"] <= 2:
        note("ik_unparsed", {"detail": str(ik)[:400]})
    return sol


def goto(pos, quat, grip, max_ticks, dq=DQ_FAST, tol=0.012, track=None, lead=3.0, min_ticks=0):
    if S["done"]:
        return "done"
    S["qcmd"] = list(S["q"])
    target = ik_solve(pos, quat, S["q"])
    if target is None:
        return "ik_fail"
    jump = max(abs(target[i] - S["q"][i]) for i in range(7))
    for t in range(max_ticks):
        if track is not None and t > 0 and t % 3 == 0:
            tr = S["tracks"][track]
            if tr["miss"] <= 2:
                npos = [tr["x"] + tr["vx"] * lead, tr["y"] + tr["vy"] * lead, pos[2]]
                q2 = ik_solve(npos, quat, target)
                if q2 is not None:
                    target = q2
                    pos = npos
        qc = S["qcmd"]
        m = max(abs(target[i] - qc[i]) for i in range(7))
        if m > dq:
            qc = [qc[i] + (target[i] - qc[i]) * dq / m for i in range(7)]
        else:
            qc = list(target)
        S["qcmd"] = qc
        if not tick(qc, grip):
            return "done"
        e = max(abs(S["q"][i] - target[i]) for i in range(7))
        if m <= dq and e < tol and t >= min_ticks:
            return "ok"
    return "timeout(jump=" + str(round(jump, 2)) + ")"


def hold(n, grip):
    if S["qcmd"] is None:
        S["qcmd"] = list(S["q"])
    for t in range(n):
        if not tick(S["qcmd"], grip):
            return False
    return True


def phase(name):
    S["ctl_phase"] = name
    report["phases"].append({"phase": name, "tick": S["ticks"], "sim_t": r3(S["last_ts"]) if S["last_ts"] is not None else None,
                             "ee": [r3(v) for v in S["ee"]] if S["ee"] is not None else None, "grip": S["grip"]})


# ---------------------------------------------------------------- buttons (SAM3 + depth)
def find_buttons(n_wanted):
    tz = S["table_z"]
    cands = []
    probes = []
    for cam in ("front", "wrist"):
        for prompt in ("button", "white button", "round button"):
            try:
                seg = segment_text(cam, prompt)
            except Exception as e:
                err("segment_text", e)
                continue
            pr = {"cam": cam, "prompt": prompt, "ok": False, "n": 0, "kept": []}
            try:
                pr["ok"] = bool(seg.ok)
                segs = seg.segmentations
                pr["n"] = len(segs)
            except Exception as e:
                err("seg_read", e)
                segs = []
            for cand in segs[:6]:
                try:
                    score = float(cand.score)
                except Exception:
                    score = 0.0
                try:
                    pc = mask_to_point_cloud(cand, cam, target_frame=S["base"])
                    pts = pc.points
                except Exception as e:
                    err("mask_to_point_cloud", e)
                    continue
                P = [(float(p[0]), float(p[1]), float(p[2])) for p in pts]
                if len(P) < 8:
                    continue
                above = [p for p in P if p[2] > tz + 0.004]
                if len(above) < 5:
                    above = P
                xs = [p[0] for p in above]; ys = [p[1] for p in above]; zs = [p[2] for p in above]
                cx = pct(xs, 0.5); cy = pct(ys, 0.5); top = pct(zs, 0.9)
                ext = max(pct(xs, 0.9) - pct(xs, 0.1), pct(ys, 0.9) - pct(ys, 0.1))
                box = None
                try:
                    box = [r3(v) for v in cand.box_xyxy]
                except Exception:
                    box = None
                c = {"cam": cam, "prompt": prompt, "score": round(score, 3), "x": cx, "y": cy, "top": top,
                     "ext": ext, "npts": len(P), "box": box}
                pr["kept"].append([round(score, 3), r3(cx), r3(cy), r3(top), r3(ext), len(P), box])
                ok_geom = (top > tz + 0.004 and top < tz + 0.07 and ext < 0.10 and ext > 0.008)
                # a saturated colour cube is not a button
                for cname in S["cubes"]:
                    if cname in ("white", "gray"):
                        continue
                    for cube in S["cubes"][cname]:
                        if (cube["x"] - cx) ** 2 + (cube["y"] - cy) ** 2 < 0.025 ** 2:
                            ok_geom = False
                if ok_geom:
                    cands.append(c)
            probes.append(pr)
    report["button_probes"] = probes
    # dedupe by xy, rank: score plus a bonus for button-sized footprint (>=3cm)
    merged = []
    for c in cands:
        hit = None
        for m in merged:
            if (m["x"] - c["x"]) ** 2 + (m["y"] - c["y"]) ** 2 < 0.03 ** 2:
                hit = m
        if hit is None:
            c["votes"] = 1
            c["best"] = c["score"]
            merged.append(c)
        else:
            hit["votes"] += 1
            if c["score"] > hit["best"]:
                hit["best"] = c["score"]
                if c["cam"] == "front" or hit["cam"] != "front":
                    hit["x"] = c["x"]; hit["y"] = c["y"]; hit["top"] = c["top"]
    ranked = sorted([(-(m["best"] + 0.1 * m["votes"] + (0.15 if m["ext"] >= 0.03 else 0.0)), i, m) for i, m in enumerate(merged)])
    chosen = [t[2] for t in ranked[:n_wanted]]
    report["button_candidates"] = [[r3(m["x"]), r3(m["y"]), r3(m["top"]), r3(m["ext"]), m["votes"], m["best"]] for t, i, m in ranked]
    return chosen


def press_button(btn, qdown, label):
    phase("press_" + label)
    top_c = S["top_z"] if S["top_z"] is not None else S["table_z"] + 0.08
    hover_z = max(btn["top"] + 0.10, top_c + 0.07)
    st = goto([btn["x"], btn["y"], hover_z], qdown, 0.0, 70)
    note("press_hover", {"button": label, "status": st, "ee": [r3(v) for v in S["ee"]]})
    if S["done"]:
        return
    # wait (briefly) until no tracked container sits on the button
    for w in range(12):
        close = False
        for t in S["tracks"]:
            if (t["x"] - btn["x"]) ** 2 + (t["y"] - btn["y"]) ** 2 < (0.6 * S["width"] + 0.025) ** 2:
                close = True
        if not close:
            break
        if not hold(1, 0.0):
            return
    st = goto([btn["x"], btn["y"], btn["top"] - 0.015], qdown, 0.0, 40, dq=DQ_SLOW, tol=0.02)
    zmin = S["ee"][2]
    hold(4, 0.0)
    if S["ee"] is not None:
        zmin = min(zmin, S["ee"][2])
    note("press_down", {"button": label, "status": st, "ee_z_min": r3(zmin), "button_top": r3(btn["top"]),
                        "tick": S["ticks"]})
    if S["done"]:
        return
    st = goto([btn["x"], btn["y"], hover_z], qdown, 0.0, 40)
    note("press_up", {"button": label, "status": st})


# ---------------------------------------------------------------- container pick
def grasp_quat(qdown, pts):
    R = rotmat(qdown)
    phi0 = math.atan2(R[1][1], R[0][1])
    if len(pts) >= 8:
        mm = min_area_yaw(pts)
        theta = mm[1]
        dims = [r3(mm[2]), r3(mm[3])]
    else:
        theta = phi0
        dims = None
    d = theta - phi0
    while d > math.pi / 4:
        d -= math.pi / 2
    while d < -math.pi / 4:
        d += math.pi / 2
    return qmul(qz(d), qdown), r3(math.degrees(d)), dims


def wait_still(tr, z, quat, max_wait):
    still = 0
    waited = 0
    for w in range(max_wait):
        if tr["miss"] == 0 and tr["spd"] < 0.0015:
            still += 1
        else:
            still = 0
        if still >= 4:
            break
        waited += 1
        goto([tr["x"] + tr["vx"] * 3.0, tr["y"] + tr["vy"] * 3.0, z], quat, 1.0, 3)
        if S["done"]:
            break
    return waited, still


def pick_container(color, qdown, order_idx):
    phase("pick_" + color)
    rec = {"color": color, "order": order_idx}
    report["picks"].append(rec)
    if color not in S["goal_map"]:
        rec["status"] = "no_track_bound"
        return
    tid = S["goal_map"][color]
    tr = S["tracks"][tid]
    top = S["top_z"]
    hover_z = top + 0.12
    rec["track"] = tid
    rec["start_xy"] = [r3(tr["x"]), r3(tr["y"])]
    st = goto([tr["x"], tr["y"], hover_z], qdown, 1.0, 80, track=tid)
    rec["hover"] = st
    if S["done"]:
        return
    remaining = S["max_steps"] - S["ticks"]
    max_wait = 120 if remaining > 400 else 40
    waited, still = wait_still(tr, hover_z, qdown, max_wait)
    rec["waited"] = waited
    rec["still"] = still
    if S["done"]:
        return
    pts = S["last_assign"].get(tid, [])
    qg, dyaw, dims = grasp_quat(qdown, pts)
    rec["yaw_deg"] = dyaw
    rec["top_dims"] = dims
    rec["n_pts"] = len(pts)
    st = goto([tr["x"], tr["y"], hover_z], qg, 1.0, 40, track=tid, lead=2.0)
    rec["rotate"] = st
    gz = top - min(0.025, 0.35 * (top - S["table_z"]))
    rec["attempts"] = []
    for attempt in range(3):
        st = goto([tr["x"], tr["y"], top + 0.04], qg, 1.0, 40, dq=DQ_SLOW, track=tid, lead=2.0)
        if S["done"]:
            return
        st2 = goto([tr["x"], tr["y"], gz], qg, 1.0, 40, dq=DQ_SLOW, tol=0.015, track=tid, lead=1.0)
        if S["done"]:
            return
        off = math.sqrt((tr["x"] - S["ee"][0]) ** 2 + (tr["y"] - S["ee"][1]) ** 2)
        rec["attempts"].append({"pre": st, "descend": st2, "offset": r3(off), "miss": tr["miss"], "n": tr["n"],
                                "ee": [r3(v) for v in S["ee"]]})
        if off <= 0.018 or tr["miss"] > 0 or attempt == 2:
            break
        # target moved under the open gripper: back off, wait for it to settle, retry
        goto([S["ee"][0], S["ee"][1], top + 0.06], qg, 1.0, 25, dq=DQ_SLOW)
        wait_still(tr, top + 0.06, qg, 40)
        if S["done"]:
            return
    rec["ee_at_grasp"] = [r3(v) for v in S["ee"]]
    if S["done"]:
        return
    # close while still following the target (containers may keep moving in this task)
    rec["close"] = goto([tr["x"], tr["y"], gz], qg, 0.0, 10, dq=DQ_SLOW, tol=0.0, track=tid, lead=1.0, min_ticks=10)
    rec["grip_after_close"] = S["grip"]
    if S["done"]:
        return
    S["held"] = tid
    st = goto([S["ee"][0], S["ee"][1], gz + 0.10], qg, 0.0, 40, dq=DQ_SLOW, tol=0.02)
    rec["lift"] = st
    rec["grip_after_lift"] = S["grip"]
    rec["ee_after_lift"] = [r3(v) for v in S["ee"]] if S["ee"] is not None else None
    if S["done"]:
        rec["terminal_during"] = "lift"
        return
    hold(3, 0.0)
    if S["done"]:
        rec["terminal_during"] = "lift_hold"
        return
    # put the container back where it was lifted from
    st = goto([S["ee"][0], S["ee"][1], gz + 0.006], qg, 0.0, 40, dq=DQ_SLOW, tol=0.02)
    rec["lower"] = st
    hold(8, 1.0)
    S["held"] = -1
    tr["x"] = S["ee"][0]; tr["y"] = S["ee"][1]; tr["vx"] = 0.0; tr["vy"] = 0.0; tr["miss"] = 0
    if S["done"]:
        rec["terminal_during"] = "release"
        return
    st = goto([S["ee"][0], S["ee"][1], hover_z], qg, 1.0, 40)
    rec["retreat"] = st


# ---------------------------------------------------------------- main
def parse_goal(goal):
    g = goal.lower()
    start = g.find("pick up")
    if start < 0:
        start = 0
    found = []
    letters = "abcdefghijklmnopqrstuvwxyz"
    for w in COLOR_WORDS:
        i = g.find(w, start)
        while i >= 0:
            before = g[i - 1] if i > 0 else " "
            j = i + len(w)
            after = g[j] if j < len(g) else " "
            if before not in letters and after not in letters:
                found.append((i, w))
            i = g.find(w, i + 1)
    found = sorted(found)
    colors = []
    for i, w in found:
        c = "gray" if w == "grey" else ("pink" if w == "magenta" else w)
        if c not in colors:
            colors.append(c)
    nb = 2 if ("both buttons" in g or "two buttons" in g) else (1 if "button" in g else 0)
    return colors, nb


def main():
    ctx = get_task_context()
    goal = str(ctx.language)
    report["goal"] = goal
    try:
        report["task_name"] = str(ctx.task_name)
    except Exception:
        pass
    try:
        md = robomme_plus.get_task_metadata()
        report["task_metadata"] = str(dict(md))[:4000]
        if "max_steps" in md:
            S["max_steps"] = int(md["max_steps"])
        for k in ("control_hz", "control_frequency", "frequency_hz"):
            if k in md:
                S["hz"] = float(md[k])
    except Exception as e:
        err("task_metadata", e)
    try:
        report["controller_metadata"] = str(dict(robomme_plus.get_controller_metadata()))[:4000]
    except Exception as e:
        err("controller_metadata", e)
    colors, nb = parse_goal(goal)
    S["goal_colors"] = colors
    report["goal_colors"] = colors
    report["n_buttons_goal"] = nb
    state = get_robot_state()
    S["base"] = state.base_frame
    S["emb"] = state.embodiment
    read_state(state)
    report["base_frame"] = str(S["base"])
    report["embodiment"] = str(S["emb"])
    report["home_q"] = [r3(v) for v in S["q"]]
    report["home_ee"] = [r3(v) for v in S["ee"]]
    report["home_ee_q"] = [r3(v) for v in S["ee_q"]]
    report["home_grip"] = S["grip"]
    qdown = list(S["ee_q"])
    Rh = rotmat(qdown)
    report["home_tool_z_axis"] = [r3(Rh[0][2]), r3(Rh[1][2]), r3(Rh[2][2])]
    if Rh[2][2] > -0.9:
        qdown = [0.0, 1.0, 0.0, 0.0]
        note("qdown_override", {"home_tool_z": report["home_tool_z_axis"]})
    report["qdown"] = [r3(v) for v in qdown]

    video = []
    try:
        video = robomme_plus.get_conditioning_video()
        report["video_len"] = len(video)
        stamps = []
        for i in range(len(video)):
            if i < 5 or i >= len(video) - 3:
                try:
                    stamps.append([i, r3(video[i].timestamp_s)])
                except Exception:
                    pass
        report["video_stamps"] = stamps
    except Exception as e:
        err("conditioning_video", e)
    obs0 = get_observation()
    try:
        report["cameras"] = [str(c) for c in obs0.cameras]
        report["obs0_t"] = float(obs0.timestamp_s)
        fc = obs0.cameras["front"]
        report["front_shape"] = [len(fc.rgb), len(fc.rgb[0])]
        report["front_pose"] = {"p": [r3(v) for v in fc.camera_pose.position], "q": [r3(v) for v in fc.camera_pose.quaternion_wxyz],
                                "frame": str(fc.camera_pose.frame)}
        wc = obs0.cameras["wrist"]
        report["wrist_pose0"] = {"p": [r3(v) for v in wc.camera_pose.position], "q": [r3(v) for v in wc.camera_pose.quaternion_wxyz]}
    except Exception as e:
        err("obs0_meta", e)

    phase("perceive_initial")
    for i in range(len(video)):
        process_frame(video[i], "video")
    S["last_ts"] = None
    process_frame(obs0, "live")
    report["phase_after_obs0"] = S["phase"]
    buttons = []
    if nb > 0:
        try:
            buttons = find_buttons(nb)
        except Exception as e:
            err("find_buttons", e)
    report["buttons"] = [[r3(b["x"]), r3(b["y"]), r3(b["top"]), r3(b["ext"]), b["cam"]] for b in buttons]

    # hold still (closing the gripper) until the containers have dropped and tracking starts
    phase("wait_containers")
    S["qcmd"] = list(S["q"])
    for w in range(40):
        if S["phase"] == "tracking" and w >= 6:
            break
        if not hold(1, 0.0):
            break
    report["tracking_started_tick"] = S.get("init_tick")
    if S["done"]:
        return

    # press buttons, nearest first
    order = []
    rest = list(buttons)
    while len(rest) > 0 and S["ee"] is not None:
        best = None
        for b in rest:
            d = (b["x"] - S["ee"][0]) ** 2 + (b["y"] - S["ee"][1]) ** 2
            if best is None or d < best[0]:
                best = [d, b]
        order.append(best[1])
        rest = [b for b in rest if b is not best[1]]
    for i in range(len(order)):
        press_button(order[i], qdown, str(i))
        if S["done"]:
            return
    report["presses_done_tick"] = S["ticks"]

    # observation pose: wrist camera above the container field
    phase("observe")
    if len(S["tracks"]) > 0:
        cxs = [t["x"] for t in S["tracks"]]; cys = [t["y"] for t in S["tracks"]]
        ox = sum(cxs) / len(cxs); oy = sum(cys) / len(cys)
        top = S["top_z"]
        st = "ik_fail"
        for dz in (0.30, 0.25, 0.20):
            st = goto([ox, oy, top + dz], qdown, 1.0, 60)
            if st != "ik_fail":
                break
        note("observe_pose", {"status": st, "xy": [r3(ox), r3(oy)]})
        quiet = 0
        for w in range(60):
            moving = 0
            for t in S["tracks"]:
                if t["spd"] > 0.0015:
                    moving += 1
            if moving == 0:
                quiet += 1
            else:
                quiet = 0
            if quiet >= 15:
                break
            if not hold(1, 1.0):
                return
        note("observe_done", {"quiet": quiet, "tick": S["ticks"],
                              "speeds": [[t["id"], round(t["spd"], 4), t["miss"]] for t in S["tracks"]]})

    report["picks"] = []
    for i in range(len(S["goal_colors"])):
        if S["done"]:
            break
        pick_container(S["goal_colors"][i], qdown, i)
    phase("end")


try:
    main()
except Exception as e:
    err("main", e)

report["ticks"] = S["ticks"]
report["frames_processed"] = S["frames"]
report["frame_errors"] = S["frame_errors"]
report["ik_calls"] = S["ik_calls"]
report["ik_fail"] = S["ik_fail"]
report["near_events"] = S["near_events"]
report["terminal"] = S.get("terminal")
report["final_phase"] = S.get("ctl_phase")
report["goal_map"] = S["goal_map"]
report["final_tracks"] = [[t["id"], r3(t["x"]), r3(t["y"]), t["n"], t["miss"], round(t["spd"], 4),
                           r3(t["x0"]), r3(t["y0"])] for t in S["tracks"]]
report["cubes_seen"] = {}
for cname in S["cubes"]:
    report["cubes_seen"][cname] = [[r3(c["x"]), r3(c["y"]), c["n"]] for c in S["cubes"][cname][:3]]
result = report
