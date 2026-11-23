# ButtonUnmaskSwap v04 -- v03 with: step-0 scene objects from the front AND the top-down wrist view
# (cubes hidden from the front camera by tall occluders get their own track from the start),
# wrist points used for the drop/cover check, wider button-base exclusion, strict anonymous tracks,
# short coasting of occluded tracks (6 ticks, then hold).
# v05 -- v04 without the per-tick point log (result size), behaviour otherwise identical.
# Strategy: (1) at step 0, project the front RGB-D view to robot_base, find the table plane,
# small objects and their colours; (2) step one tick at a time, detect the container drop,
# separate buttons (never covered) from cubes (covered); (3) track every container in 3D xy each
# tick (nearest-predicted-track assignment of white points in the container height band);
# (4) press both buttons before tick 179 while tracking; (5) wait for the shuffle to settle;
# (6) pick up the containers tracked from the goal's colours, in goal order, and put each back.
import math

R = {"version": "v05", "events": [], "snap": []}
S = {"t": 0, "done": False, "joints": None, "ee": None, "grip": 1.0}

COLOR_PROTOS = {
    "red": (190, 25, 25), "green": (30, 190, 30), "blue": (25, 25, 190),
    "yellow": (200, 200, 20), "purple": (150, 60, 190), "orange": (205, 130, 20),
    "black": (35, 35, 35), "white": (225, 225, 225), "gray": (128, 128, 128),
    "grey": (128, 128, 128), "pink": (230, 120, 170), "cyan": (30, 200, 200),
    "brown": (120, 70, 30),
}


def ev(msg):
    R["events"].append([S["t"], msg])


def tolist2(a):
    try:
        return a.tolist()
    except Exception:
        return [list(r) for r in a]


def qmat(q):
    w, x, y, z = [float(v) for v in q]
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def quant(vals, f):
    s = sorted(vals)
    if not s:
        return None
    return s[int((len(s) - 1) * f)]


# ---------------------------------------------------------------- task context
ctx = get_task_context()
goal = str(ctx.language)
R["goal"] = goal
meta = {}
try:
    meta = robomme_plus.get_task_metadata()
    R["meta"] = {str(k): str(meta[k])[:200] for k in meta}
except Exception as exc:
    R["meta_err"] = str(exc)
MAX_STEPS = 1000
try:
    MAX_STEPS = int(meta["max_steps"])
except Exception:
    pass
R["max_steps"] = MAX_STEPS
PRESS_DEADLINE = 179

words = goal.lower().replace(",", " ").replace(".", " ").split()
then_idx = 0
for i, w in enumerate(words):
    if w == "then":
        then_idx = i
        break
order = []
for w in words[then_idx:]:
    if w in COLOR_PROTOS and w not in order:
        order.append(w)
R["order"] = order

state0 = get_robot_state()
BASE = state0.base_frame
EMB = state0.embodiment
HOME = [float(v) for v in state0.joint_positions["primary"]]
S["joints"] = list(HOME)
ee0 = state0.end_effector_poses["primary"]
S["ee"] = [float(v) for v in ee0.position]
Q_DOWN = [float(v) for v in ee0.quaternion_wxyz]
try:
    S["grip"] = float(state0.gripper_positions["primary"])
except Exception:
    pass
R["home_ee"] = [round(v, 4) for v in S["ee"]]
R["home_q"] = [round(v, 4) for v in Q_DOWN]

# ---------------------------------------------------------------- camera geometry
obs0 = get_observation()
cam0 = obs0.cameras["front"]
K = tolist2(cam0.intrinsics)
FX, FY, CX, CY = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
CP = [float(v) for v in cam0.camera_pose.position]
CR = qmat(cam0.camera_pose.quaternion_wxyz)
rgb0 = tolist2(cam0.rgb)
dep0 = tolist2(cam0.depth_m)
IH = len(rgb0)
IW = len(rgb0[0])
R["img"] = [IH, IW, round(FX, 2), round(FY, 2), round(CX, 2), round(CY, 2)]
R["cam_pos"] = [round(v, 4) for v in CP]


def cam_vec(conv, a, b):
    if conv == "cv":
        return (a, b, 1.0)
    if conv == "gl":
        return (a, -b, -1.0)
    return (1.0, -a, -b)          # ros camera-link: x forward, y left, z up


def ray_world(conv, dmode, r, c):
    a = (c - CX) / FX
    b = (r - CY) / FY
    v = cam_vec(conv, a, b)
    sc = 1.0
    if dmode == "ray":
        sc = 1.0 / math.sqrt(1.0 + a * a + b * b)
    return [sc * (CR[0][0] * v[0] + CR[0][1] * v[1] + CR[0][2] * v[2]),
            sc * (CR[1][0] * v[0] + CR[1][1] * v[1] + CR[1][2] * v[2]),
            sc * (CR[2][0] * v[0] + CR[2][1] * v[1] + CR[2][2] * v[2])]


# choose the projection convention that makes the bottom (table-only) rows flattest
best = None
proj_scores = []
for conv in ("cv", "gl", "ros"):
    for dmode in ("z", "ray"):
        zs = []
        for r in range(int(IH * 0.78), IH, 6):
            for c in range(4, IW - 4, 10):
                d = float(dep0[r][c])
                if d <= 0.0 or d > 10.0:
                    continue
                rw = ray_world(conv, dmode, r, c)
                zs.append(CP[2] + d * rw[2])
        if len(zs) < 20:
            continue
        m = sum(zs) / len(zs)
        sd = math.sqrt(sum((z - m) ** 2 for z in zs) / len(zs))
        proj_scores.append([conv, dmode, round(m, 4), round(sd, 5)])
        if m < CP[2] - 0.05 and (best is None or sd < best[2]):
            best = (conv, dmode, sd, m)
R["proj_scores"] = proj_scores
if best is None:
    raise ValueError("no projection convention gives a table below the camera")
CONV, DMODE = best[0], best[1]
R["proj"] = [CONV, DMODE, round(best[2], 5)]

STRIDE = 2
GRID = []          # (r, c, ray)
for r in range(0, IH, STRIDE):
    for c in range(0, IW, STRIDE):
        GRID.append((r, c, ray_world(CONV, DMODE, r, c)))

# table height from bottom rows (robust median)
tzs = []
for r in range(int(IH * 0.78), IH, 4):
    for c in range(4, IW - 4, 6):
        d = float(dep0[r][c])
        if 0.0 < d < 10.0:
            rw = ray_world(CONV, DMODE, r, c)
            tzs.append(CP[2] + d * rw[2])
TZ = quant(tzs, 0.5)
R["table_z"] = round(TZ, 4)


def project_all(rgbl, depl, rmin, rmax, white_only):
    pts = []
    for g in GRID:
        r = g[0]
        if r < rmin or r > rmax:
            continue
        c = g[1]
        px = rgbl[r][c]
        if white_only:
            mn = min(px[0], px[1], px[2])
            mx = max(px[0], px[1], px[2])
            if mn < 120 or mx - mn > 45:
                continue
        d = float(depl[r][c])
        if d <= 0.0 or d > 10.0:
            continue
        rw = g[2]
        z = CP[2] + d * rw[2]
        if z < TZ + 0.005 or z > TZ + 0.16:
            continue
        pts.append((CP[0] + d * rw[0], CP[1] + d * rw[1], z, px[0], px[1], px[2], r, c))
    return pts


def clusters(pts, cell):
    cells = {}
    for i, p in enumerate(pts):
        k = (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)))
        if k in cells:
            cells[k].append(i)
        else:
            cells[k] = [i]
    seen = set()
    out = []
    for k in cells:
        if k in seen:
            continue
        stack = [k]
        seen.add(k)
        members = []
        while stack:
            cur = stack.pop()
            members.extend(cells[cur])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cur[0] + dx, cur[1] + dy)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        out.append(members)
    return out


# ---------------------------------------------------------------- step-0 scene objects
def wrist_scene_points(obs):
    # every wrist pixel (any colour) projected to robot_base; row index -1 marks wrist points
    cam = obs.cameras["wrist"]
    Kw = tolist2(cam.intrinsics)
    fxw, fyw, cxw, cyw = float(Kw[0][0]), float(Kw[1][1]), float(Kw[0][2]), float(Kw[1][2])
    rgbl = tolist2(cam.rgb)
    depl = tolist2(cam.depth_m)
    cp = [float(v) for v in cam.camera_pose.position]
    cr = qmat(cam.camera_pose.quaternion_wxyz)
    out = []
    zs = []
    for r in range(0, len(depl), STRIDE):
        for c in range(0, len(depl[0]), STRIDE):
            d = float(depl[r][c])
            if d <= 0.0 or d > 10.0:
                continue
            a = (c - cxw) / fxw
            b = (r - cyw) / fyw
            x = cp[0] + d * (cr[0][0] * a + cr[0][1] * b + cr[0][2])
            y = cp[1] + d * (cr[1][0] * a + cr[1][1] * b + cr[1][2])
            z = cp[2] + d * (cr[2][0] * a + cr[2][1] * b + cr[2][2])
            zs.append(z)
            px = rgbl[r][c]
            out.append((x, y, z, px[0], px[1], px[2], -1, c))
    if len(zs) < 50 or abs(quant(zs, 0.3) - TZ) > 0.01:
        return []
    return out


pts0 = [p for p in project_all(rgb0, dep0, 0, IH, False) if p[2] - TZ < 0.10]
try:
    wp0 = [p for p in wrist_scene_points(obs0) if p[2] - TZ > 0.005 and p[2] - TZ < 0.10]
    R["wrist0_pts"] = len(wp0)
    pts0 = pts0 + wp0
except Exception as exc:
    R["wrist0_err"] = str(exc)
objs = []
obstacle_cells = set()
for mem in clusters(pts0, 0.012):
    ps = [pts0[i] for i in mem]
    xs = [p[0] for p in ps]
    ys = [p[1] for p in ps]
    zs = [p[2] for p in ps]
    ztop = quant(zs, 0.95) - TZ
    ext = max(max(xs) - min(xs), max(ys) - min(ys))
    if len(ps) >= 4 and ztop < 0.06 and ext < 0.08:
        top = [p for p in ps if p[2] - TZ > ztop - 0.006]
        objs.append({
            "xy": [sum(p[0] for p in ps) / len(ps), sum(p[1] for p in ps) / len(ps)],
            "top_xy": [sum(p[0] for p in top) / len(top), sum(p[1] for p in top) / len(top)],
            "ztop": ztop, "n": len(ps), "ext": ext,
            "rgb": [sum(p[3] for p in ps) / len(ps), sum(p[4] for p in ps) / len(ps),
                    sum(p[5] for p in ps) / len(ps)],
            "rows": [min([p[6] for p in ps if p[6] >= 0] + [IH]), max([p[6] for p in ps if p[6] >= 0] + [0])],
        })
    elif len(ps) >= 4:
        for p in ps:
            if p[2] - TZ < 0.12:
                kx = int(math.floor(p[0] / 0.01))
                ky = int(math.floor(p[1] / 0.01))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        obstacle_cells.add((kx + dx, ky + dy))
# merge fragments of one object split by the sparse pixel grid
merged = True
while merged:
    merged = False
    for a in range(len(objs)):
        for b in range(a + 1, len(objs)):
            oa, ob = objs[a], objs[b]
            if (oa["xy"][0] - ob["xy"][0]) ** 2 + (oa["xy"][1] - ob["xy"][1]) ** 2 < 0.025 ** 2:
                na, nb = oa["n"], ob["n"]
                tot = float(na + nb)
                hi, lo = (oa, ob) if oa["ztop"] >= ob["ztop"] else (ob, oa)
                objs[a] = {
                    "xy": [(oa["xy"][0] * na + ob["xy"][0] * nb) / tot,
                           (oa["xy"][1] * na + ob["xy"][1] * nb) / tot],
                    "top_xy": list(hi["top_xy"]), "ztop": hi["ztop"], "n": na + nb,
                    "ext": max(oa["ext"], ob["ext"]),
                    "rgb": [(oa["rgb"][k] * na + ob["rgb"][k] * nb) / tot for k in range(3)],
                    "rows": [min(oa["rows"][0], ob["rows"][0]), max(oa["rows"][1], ob["rows"][1])],
                }
                objs.pop(b)
                merged = True
                break
        if merged:
            break
R["objects0"] = [[round(o["xy"][0], 3), round(o["xy"][1], 3), round(o["ztop"], 4), o["n"],
                  [int(v) for v in o["rgb"]]] for o in objs]
R["obstacle_cells"] = len(obstacle_cells)
if len(objs) < 3:
    raise ValueError("fewer than three small objects found at step 0")

ROW_MIN = max(0, min(o["rows"][0] for o in objs) - 45)
ROW_MAX = min(IH - 1, max(o["rows"][1] for o in objs) + 45)
R["rows"] = [ROW_MIN, ROW_MAX]

# goal colours -> step-0 objects (greedy unique nearest prototype)
target_obj = {}
used = set()
pairs = []
for cname in order:
    pr = COLOR_PROTOS[cname]
    for j, o in enumerate(objs):
        d = math.sqrt(sum((o["rgb"][k] - pr[k]) ** 2 for k in range(3)))
        pairs.append((d, cname, j))
for d, cname, j in sorted(pairs):
    if cname in target_obj or j in used:
        continue
    target_obj[cname] = j
    used.add(j)
R["target_obj"] = target_obj

# ---------------------------------------------------------------- tracker state
TR = {"tracks": [], "buttons": [], "hc": 0.06, "ra": 0.035, "init": False,
      "static": 0, "last_pts": []}


WR = {"ok": None, "vec": None, "tz": None}


def wrist_points(obs):
    # top-down wrist view: only used while the hand is high (e.g. at home), where it sees every
    # container from above and is not blocked by tall front-view occluders.
    cam = obs.cameras["wrist"]
    if WR["vec"] is None:
        Kw = tolist2(cam.intrinsics)
        fxw, fyw, cxw, cyw = float(Kw[0][0]), float(Kw[1][1]), float(Kw[0][2]), float(Kw[1][2])
        vec = []
        hw = len(tolist2(cam.depth_m))
        for r in range(0, hw, STRIDE):
            for c in range(0, hw, STRIDE):
                vec.append((r, c, (c - cxw) / fxw, (r - cyw) / fyw))
        WR["vec"] = vec
    rgbl = tolist2(cam.rgb)
    depl = tolist2(cam.depth_m)
    cp = [float(v) for v in cam.camera_pose.position]
    cr = qmat(cam.camera_pose.quaternion_wxyz)
    pts = []
    tzs = []
    for g in WR["vec"]:
        r = g[0]
        c = g[1]
        px = rgbl[r][c]
        mn = min(px[0], px[1], px[2])
        mx = max(px[0], px[1], px[2])
        white = mn >= 120 and mx - mn <= 45
        if not white and (r + c) % 16 != 0:
            continue
        d = float(depl[r][c])
        if d <= 0.0 or d > 10.0:
            continue
        a = g[2]
        b = g[3]
        x = cp[0] + d * (cr[0][0] * a + cr[0][1] * b + cr[0][2])
        y = cp[1] + d * (cr[1][0] * a + cr[1][1] * b + cr[1][2])
        z = cp[2] + d * (cr[2][0] * a + cr[2][1] * b + cr[2][2])
        if not white:
            tzs.append(z)
            continue
        if z < TZ + 0.005 or z > TZ + 0.16:
            continue
        pts.append((x, y, z, px[0], px[1], px[2], r, c))
    if WR["ok"] is None and len(tzs) > 50:
        WR["tz"] = quant(tzs, 0.3)
        WR["ok"] = abs(WR["tz"] - TZ) < 0.01
        R["wrist_table_z"] = round(WR["tz"], 4)
        R["wrist_ok"] = WR["ok"]
    if WR["ok"]:
        return pts
    return []


def perceive(obs):
    cam = obs.cameras["front"]
    rgbl = tolist2(cam.rgb)
    depl = tolist2(cam.depth_m)
    pts = project_all(rgbl, depl, ROW_MIN, ROW_MAX, True)
    ee = S["ee"]
    low_ee = ee[2] - TZ < TR["hc"] + 0.09
    keep = []
    if WR["ok"] is not False:
        try:
            ztop_w = TR["hc"] + 0.004 if TR["init"] else 0.10
            for p in wrist_points(obs):
                if p[2] - TZ < 0.012 or p[2] - TZ > ztop_w:
                    continue
                if low_ee and (p[0] - ee[0]) ** 2 + (p[1] - ee[1]) ** 2 < 0.06 ** 2:
                    continue
                k = (int(math.floor(p[0] / 0.01)), int(math.floor(p[1] / 0.01)))
                if k in obstacle_cells:
                    continue
                skip = False
                for b in TR["buttons"]:
                    if (p[0] - b["xy"][0]) ** 2 + (p[1] - b["xy"][1]) ** 2 < 0.04 ** 2 and \
                            p[2] - TZ < b["ztop"] + 0.012:
                        skip = True
                        break
                if not skip:
                    keep.append(p)
        except Exception as exc:
            WR["ok"] = False
            R["wrist_err"] = str(exc)
    for p in pts:
        if p[2] - TZ < 0.012:
            continue
        if TR["init"] and p[2] - TZ > TR["hc"] + 0.004:
            continue
        k = (int(math.floor(p[0] / 0.01)), int(math.floor(p[1] / 0.01)))
        if k in obstacle_cells:
            continue
        if low_ee and (p[0] - ee[0]) ** 2 + (p[1] - ee[1]) ** 2 < 0.14 ** 2:
            continue
        skip = False
        for b in TR["buttons"]:
            if (p[0] - b["xy"][0]) ** 2 + (p[1] - b["xy"][1]) ** 2 < 0.04 ** 2 and \
                    p[2] - TZ < b["ztop"] + 0.012:
                skip = True
                break
        if skip:
            continue
        keep.append(p)
    TR["last_pts"] = keep
    return keep


MIN_SEP = 0.05


def track_update(pts):
    tracks = TR["tracks"]
    if not tracks:
        return
    preds = [[t["xy"][0] + t["v"][0], t["xy"][1] + t["v"][1]] for t in tracks]
    sx = [0.0] * len(tracks)
    sy = [0.0] * len(tracks)
    cnt = [0] * len(tracks)
    own = [[] for t in tracks]
    orphans = []
    ra2 = TR["ra"] ** 2
    for p in pts:
        bi = -1
        bd = ra2
        for i, q in enumerate(preds):
            d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
            if d < bd:
                bd = d
                bi = i
        if bi >= 0:
            sx[bi] += p[0]
            sy[bi] += p[1]
            cnt[bi] += 1
            own[bi].append(p)
        else:
            orphans.append(p)
    newxy = []
    good = []
    for i, t in enumerate(tracks):
        if cnt[i] >= 5:
            newxy.append([sx[i] / cnt[i], sy[i] / cnt[i]])
            good.append(True)
        else:
            newxy.append(None)
            good.append(False)
    # two tracks may never sit on one container: the weaker one reverts and is marked lost
    changed = True
    while changed:
        changed = False
        for i in range(len(tracks)):
            if not good[i]:
                continue
            for j in range(i + 1, len(tracks)):
                if not good[j]:
                    continue
                dx = newxy[i][0] - newxy[j][0]
                dy = newxy[i][1] - newxy[j][1]
                if dx * dx + dy * dy < MIN_SEP ** 2:
                    weak = i if cnt[i] < cnt[j] else j
                    good[weak] = False
                    TR["merges"] = TR.get("merges", 0) + 1
                    changed = True
                    break
            if changed:
                break
    moving = 0
    for i, t in enumerate(tracks):
        if good[i]:
            nx, ny = newxy[i]
            vx = 0.6 * t["v"][0] + 0.4 * (nx - t["xy"][0])
            vy = 0.6 * t["v"][1] + 0.4 * (ny - t["xy"][1])
            sp = math.sqrt(vx * vx + vy * vy)
            if sp > 0.012:
                vx, vy = vx * 0.012 / sp, vy * 0.012 / sp
            t["v"] = [vx, vy]
            t["xy"] = [nx, ny]
            t["lost"] = 0
            t["typ"] = max(0.9 * t.get("typ", 0.0), float(cnt[i])) if cnt[i] > t.get("typ", 0.0) \
                else 0.98 * t.get("typ", 0.0) + 0.02 * cnt[i]
        else:
            # occluded: keep moving at constant velocity for a while, then slow down
            if t.get("lost", 0) < 6:
                t["xy"] = [t["xy"][0] + t["v"][0], t["xy"][1] + t["v"][1]]
            else:
                t["v"] = [0.5 * t["v"][0], 0.5 * t["v"][1]]
            t["lost"] = t.get("lost", 0) + 1
        t["n"] = cnt[i] if good[i] else 0
        t["pts"] = own[i] if good[i] else []
        if math.sqrt(t["v"][0] ** 2 + t["v"][1] ** 2) > 0.0008:
            moving += 1
    # unexplained point clusters: re-acquire lost tracks, or start anonymous container tracks
    cands = []
    if len(orphans) >= 15:
        ee = S["ee"]
        low_ee = ee[2] - TZ < TR["hc"] + 0.09
        for mem in clusters(orphans, 0.012):
            if len(mem) < 15:
                continue
            cx = sum(orphans[m][0] for m in mem) / len(mem)
            cy = sum(orphans[m][1] for m in mem) / len(mem)
            far = True
            for t in tracks:
                if t.get("lost", 0) == 0 and (t["xy"][0] - cx) ** 2 + (t["xy"][1] - cy) ** 2 < (1.25 * MIN_SEP) ** 2:
                    far = False
            if low_ee and (ee[0] - cx) ** 2 + (ee[1] - cy) ** 2 < 0.10 ** 2:
                far = False
            if far:
                cands.append([cx, cy, len(mem)])
    lost_ids = [i for i, t in enumerate(tracks) if t.get("lost", 0) >= 3]
    pairs = []
    for ci, cnd in enumerate(cands):
        for i in lost_ids:
            d = math.sqrt((tracks[i]["xy"][0] - cnd[0]) ** 2 + (tracks[i]["xy"][1] - cnd[1]) ** 2)
            if d < 0.15:
                pairs.append((d, ci, i))
    usedc = set()
    usedt = set()
    for d, ci, i in sorted(pairs):
        if ci in usedc or i in usedt:
            continue
        usedc.add(ci)
        usedt.add(i)
        tracks[i]["xy"] = [cands[ci][0], cands[ci][1]]
        tracks[i]["v"] = [0.0, 0.0]
        tracks[i]["lost"] = 0
        R["events"].append([S["t"], "reacquire %d d=%.3f" % (i, d)])
        moving += 1
    pend = TR.get("pending", [])
    newpend = []
    for ci, cnd in enumerate(cands):
        if ci in usedc:
            continue
        hits = 1
        for q in pend:
            if (q[0] - cnd[0]) ** 2 + (q[1] - cnd[1]) ** 2 < 0.03 ** 2:
                hits = max(hits, q[3] + 1)
        if hits >= 10 and cnd[2] >= 40 and len(tracks) < 14:
            tracks.append({"xy": [cnd[0], cnd[1]], "v": [0.0, 0.0], "n": cnd[2], "obj": -1, "pts": [],
                           "lost": 0, "anon": True})
            R["events"].append([S["t"], "anon track %d at %.3f,%.3f" % (len(tracks) - 1, cnd[0], cnd[1])])
            moving += 1
        else:
            newpend.append([cnd[0], cnd[1], cnd[2], hits])
    TR["pending"] = newpend
    if moving == 0:
        TR["static"] += 1
    else:
        TR["static"] = 0


def snapshot():
    R["snap"].append([S["t"], [[round(t["xy"][0], 3), round(t["xy"][1], 3), t["n"]]
                               for t in TR["tracks"]]])


def try_init_tracks(pts):
    # a step-0 object is covered when white points appear well above its own top
    covered = []
    for j, o in enumerate(objs):
        n = 0
        for p in pts:
            if (p[0] - o["xy"][0]) ** 2 + (p[1] - o["xy"][1]) ** 2 < 0.03 ** 2 and \
                    p[2] - TZ > o["ztop"] + 0.012:
                n += 1
        covered.append(n)
    return covered


# ---------------------------------------------------------------- control helpers
def after_step(res):
    term = False
    try:
        term = bool(res.terminated) or bool(res.truncated)
    except Exception:
        pass
    if term:
        S["done"] = True
        ev("terminal")
        return None
    obs = None
    try:
        obs = res.observation
    except Exception:
        obs = None
    if obs is None:
        obs = get_observation()
    st = obs.robot_state
    S["joints"] = [float(v) for v in st.joint_positions["primary"]]
    S["ee"] = [float(v) for v in st.end_effector_poses["primary"].position]
    try:
        S["grip_meas"] = float(st.gripper_positions["primary"])
    except Exception:
        pass
    return obs


def tick(cmd, grip):
    if S["done"]:
        return None
    res = step(RobotAction(arms={"primary": ArmCommand(
        "joint_position", [float(v) for v in cmd], gripper_position=float(grip), embodiment=EMB)}))
    S["t"] += 1
    obs = after_step(res)
    if obs is None:
        return None
    pts = perceive(obs)
    if TR["init"] and not TR.get("frozen", False):
        track_update(pts)
        if S["t"] % 5 == 0:
            snapshot()
    else:
        cov = try_init_tracks(pts)
        TR["cov"] = cov
    return pts


def goto(target, grip, max_d=0.04, settle=3):
    cur = list(S["joints"])
    dmax = max(abs(target[i] - cur[i]) for i in range(7))
    n = max(1, int(math.ceil(dmax / max_d)))
    for i in range(1, n + 1):
        if S["done"]:
            return
        cmd = [cur[k] + (target[k] - cur[k]) * i / n for k in range(7)]
        tick(cmd, grip)
    for s in range(settle):
        if S["done"]:
            return
        tick(target, grip)


def ik(pos, quat):
    res = solve_ik(Pose([float(v) for v in pos], [float(v) for v in quat], BASE),
                   seed_joints=list(S["joints"]))
    if not res.ok:
        ev("ik_fail %s" % str([round(v, 3) for v in pos]))
        return None
    return [float(v) for v in res.joint_positions]


def move_pose(pos, quat, grip, max_d=0.04, settle=3):
    j = ik(pos, quat)
    if j is None:
        return False
    goto(j, grip, max_d, settle)
    return True


def hold(n, grip):
    for s in range(n):
        if S["done"]:
            return
        tick(S["joints"], grip)


# ---------------------------------------------------------------- run
try:
    # 1. hold until the containers drop, then split cubes / buttons
    grip_now = 1.0 if S["grip"] > 0.5 else 0.0
    covered = None
    stable = 0
    for k in range(30):
        tick(HOME, grip_now)
        cov = TR.get("cov", [])
        nc = sum(1 for n in cov if n >= 8)
        if nc >= 2:
            stable += 1
            if stable >= 2:
                covered = cov
                break
    if covered is None:
        raise ValueError("container drop not detected")
    ev("drop detected")
    cubes = [j for j, n in enumerate(covered) if n >= 8]
    buttons = [j for j, n in enumerate(covered) if n < 8]
    R["covered"] = covered
    if len(buttons) > 2:
        # keep the two least saturated / brightest uncovered objects
        buttons = sorted(buttons, key=lambda j: -(min(objs[j]["rgb"]) - (max(objs[j]["rgb"]) - min(objs[j]["rgb"]))))[:2]
    TR["buttons"] = [{"xy": objs[j]["top_xy"], "ztop": objs[j]["ztop"]} for j in buttons]
    R["buttons"] = [[round(objs[j]["top_xy"][0], 3), round(objs[j]["top_xy"][1], 3),
                     round(objs[j]["ztop"], 4)] for j in buttons]
    # container height and tracks
    pts = TR["last_pts"]
    hz = []
    for j in cubes:
        o = objs[j]
        for p in pts:
            if (p[0] - o["xy"][0]) ** 2 + (p[1] - o["xy"][1]) ** 2 < 0.035 ** 2:
                hz.append(p[2] - TZ)
    TR["hc"] = quant(hz, 0.97) if hz else 0.06
    R["container_h"] = round(TR["hc"], 4)
    for j in cubes:
        TR["tracks"].append({"xy": list(objs[j]["xy"]), "v": [0.0, 0.0], "n": 0, "obj": j, "pts": []})
    TR["init"] = True
    track_update(TR["last_pts"])
    # assignment radius from the observed spread of points
    rad = []
    for t in TR["tracks"]:
        for p in t["pts"]:
            rad.append(math.sqrt((p[0] - t["xy"][0]) ** 2 + (p[1] - t["xy"][1]) ** 2))
    if rad:
        TR["ra"] = max(0.03, min(0.06, 1.6 * quant(rad, 0.9)))
    R["assign_radius"] = round(TR["ra"], 4)
    target_track = {}
    for cname in order:
        if cname in target_obj and target_obj[cname] in cubes:
            target_track[cname] = cubes.index(target_obj[cname])
    R["target_track"] = target_track
    snapshot()

    # 2. press both buttons (nearest first), gripper closed
    if len(TR["buttons"]) >= 2:
        bl = list(TR["buttons"])
        ee = S["ee"]
        bl = sorted(bl, key=lambda b: (b["xy"][0] - ee[0]) ** 2 + (b["xy"][1] - ee[1]) ** 2)
        for b in bl:
            above = [b["xy"][0], b["xy"][1], TZ + b["ztop"] + 0.07]
            press = [b["xy"][0], b["xy"][1], TZ + b["ztop"] - 0.015]
            move_pose(above, Q_DOWN, 0.0, 0.085, 2)
            move_pose(press, Q_DOWN, 0.0, 0.05, 4)
            ev("pressed ee=%s" % str([round(v, 3) for v in S["ee"]]))
            move_pose(above, Q_DOWN, 0.0, 0.07, 0)
            if S["done"]:
                break
    else:
        ev("buttons not found")
    R["t_after_press"] = S["t"]

    # 3. go back home and wait until the containers stop moving
    goto(HOME, 1.0, 0.08, 2)
    wait_cap = min(520, MAX_STEPS - 3 * 140)
    while not S["done"] and S["t"] < wait_cap and TR["static"] < 25:
        tick(HOME, 1.0)
    R["t_settled"] = S["t"]
    R["static_ticks"] = TR["static"]
    try:
        cons = []
        lp = TR["last_pts"]
        tr = TR["tracks"]
        blobs = []
        for mem in clusters(lp, 0.012):
            if len(mem) >= 15:
                blobs.append([sum(lp[m][0] for m in mem) / len(mem), sum(lp[m][1] for m in mem) / len(mem), len(mem)])
        owner = []
        for b in blobs:
            bi, bd = -1, None
            for i, t in enumerate(tr):
                d = (t["xy"][0] - b[0]) ** 2 + (t["xy"][1] - b[1]) ** 2
                if t.get("lost", 0) == 0 and d < MIN_SEP ** 2 and (bd is None or d < bd):
                    bi, bd = i, d
            owner.append(bi)
        free = [k for k in range(len(blobs)) if owner[k] < 0]
        held = set(o for o in owner if o >= 0)
        spare = [i for i, t in enumerate(tr) if i not in held]
        prs = []
        for k in free:
            for i in spare:
                prs.append(((tr[i]["xy"][0] - blobs[k][0]) ** 2 + (tr[i]["xy"][1] - blobs[k][1]) ** 2, k, i))
        uk, ui = set(), set()
        for d, k, i in sorted(prs):
            if k in uk or i in ui:
                continue
            uk.add(k)
            ui.add(i)
            cons.append([i, round(tr[i]["xy"][0], 3), round(tr[i]["xy"][1], 3),
                         round(blobs[k][0], 3), round(blobs[k][1], 3)])
            tr[i]["xy"] = [blobs[k][0], blobs[k][1]]
            tr[i]["v"] = [0.0, 0.0]
            tr[i]["lost"] = 0
            tr[i]["pts"] = []
            for m in range(len(lp)):
                if (lp[m][0] - blobs[k][0]) ** 2 + (lp[m][1] - blobs[k][1]) ** 2 < TR["ra"] ** 2:
                    tr[i]["pts"].append(lp[m])
            tr[i]["n"] = len(tr[i]["pts"])
        R["consistency"] = cons
        R["blobs"] = [[round(b[0], 3), round(b[1], 3), b[2]] for b in blobs]
    except Exception as exc:
        R["consistency_err"] = str(exc)
    if TR["static"] >= 25:
        TR["frozen"] = True       # settled: the arm must not drag tracks during the pickups
    R["frozen"] = TR.get("frozen", False)

    # 4. pick up each target container in goal order, then put it back
    picks = []
    for cname in order:
        if S["done"]:
            break
        if cname not in target_track:
            ev("no track for %s" % cname)
            continue
        t = TR["tracks"][target_track[cname]]
        tp = t.get("pts", [])
        zt = quant([p[2] for p in tp], 0.95) if len(tp) >= 5 else TZ + TR["hc"]
        top = [p for p in tp if p[2] > zt - 0.012]
        if len(top) >= 4:
            cx = sum(p[0] for p in top) / len(top)
            cy = sum(p[1] for p in top) / len(top)
        else:
            cx, cy = t["xy"][0], t["xy"][1]
        best_th, best_area = 0.0, None
        if len(top) >= 6:
            for kdeg in range(0, 90, 3):
                th = math.radians(kdeg)
                ca, sa = math.cos(th), math.sin(th)
                us = [(p[0] - cx) * ca + (p[1] - cy) * sa for p in top]
                vs = [-(p[0] - cx) * sa + (p[1] - cy) * ca for p in top]
                area = (max(us) - min(us)) * (max(vs) - min(vs))
                if best_area is None or area < best_area:
                    best_area, best_th = area, th
        Rm = qmat(Q_DOWN)
        fing = math.atan2(Rm[1][1], Rm[0][1])
        delta = best_th - fing
        while delta >= math.pi / 4:
            delta -= math.pi / 2
        while delta < -math.pi / 4:
            delta += math.pi / 2
        qg = qmul([math.cos(delta / 2), 0.0, 0.0, math.sin(delta / 2)], Q_DOWN)
        gz = zt - 0.02
        pick = {"color": cname, "track": target_track[cname], "xy": [round(cx, 3), round(cy, 3)],
                "ztop": round(zt - TZ, 4), "yaw_deg": round(math.degrees(delta), 1), "t0": S["t"]}
        pre = [cx, cy, zt + 0.10]
        if not move_pose(pre, qg, 1.0, 0.045, 2):
            picks.append(pick)
            continue
        # re-centre if the container drifted while we approached
        t2 = TR["tracks"][target_track[cname]]
        if t2["n"] >= 5:
            dx = t2["xy"][0] - t["xy"][0]
            dy = t2["xy"][1] - t["xy"][1]
            if dx * dx + dy * dy > 0.01 ** 2:
                cx += dx
                cy += dy
                pick["drift"] = [round(dx, 3), round(dy, 3)]
                move_pose([cx, cy, zt + 0.10], qg, 1.0, 0.04, 1)
        move_pose([cx, cy, gz], qg, 1.0, 0.025, 3)
        if S["done"]:
            picks.append(pick)
            break
        cg = close_gripper()
        try:
            S["t"] += int(cg.steps_executed)
            if bool(cg.terminated) or bool(cg.truncated):
                S["done"] = True
        except Exception:
            pass
        if S["done"]:
            picks.append(pick)
            break
        hold(1, 0.0)
        pick["grip_closed"] = round(S.get("grip_meas", -1.0), 3)
        move_pose([cx, cy, gz + 0.12], qg, 0.0, 0.03, 3)
        pick["grip_lifted"] = round(S.get("grip_meas", -1.0), 3)
        pick["ee_lifted"] = [round(v, 3) for v in S["ee"]]
        if S["done"]:
            picks.append(pick)
            break
        # put it back where it was
        move_pose([cx, cy, gz + 0.006], qg, 0.0, 0.03, 2)
        og = open_gripper()
        try:
            S["t"] += int(og.steps_executed)
            if bool(og.terminated) or bool(og.truncated):
                S["done"] = True
        except Exception:
            pass
        if S["done"]:
            picks.append(pick)
            break
        move_pose([cx, cy, zt + 0.10], qg, 1.0, 0.04, 1)
        t2 = TR["tracks"][target_track[cname]]
        t2["xy"] = [cx, cy]
        t2["v"] = [0.0, 0.0]
        pick["t1"] = S["t"]
        picks.append(pick)
    R["picks"] = picks
except Exception as exc:
    R["error"] = str(exc)

R["merges"] = TR.get("merges", 0)
R["final_t"] = S["t"]
R["done"] = S["done"]
result = R
