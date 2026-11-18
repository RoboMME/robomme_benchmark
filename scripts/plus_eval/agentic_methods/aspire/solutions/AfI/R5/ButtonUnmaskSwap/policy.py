# v03 — ButtonUnmaskSwap.
# v02 evidence (ep0): centred press (EE 1 mm from the cap centre) -> env terminated (failure) at
# control step 40-41, i.e. immediately when the first press registered. During that press the
# command kept driving 8 mm/tick into the cap: the fingertips slid ~9 mm sideways on the cap and
# dropped again (possible double press / knocked cap). v01's off-centre contact at step 62 did not
# terminate. v03: hover above the first button until step >= 62, then a gentle single press:
# 2 mm/tick commanded descent from just above the cap to 1.2 cm below its top, short hold, slow
# 3 mm/tick release, then rise; same for the second button; per-tick press trace recorded.
import math

R = {"errors": [], "events": [], "log": []}
S = {"ended": False, "step": 0, "tracks": None, "zhi": 0.12, "top": None,
     "carried": None, "quiet": 0, "band_hist": []}
COLOR_WORDS = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "cyan", "magenta",
               "black", "white", "brown", "gray", "grey"]
ZLO = 0.045          # above cube/button tops (measured 0.034-0.035 on ep0) -> cover bodies only
ASSOC_R = 0.045      # association radius around a predicted cover centre (m)
MINP = 6             # minimum sampled points for a track update


def err(where, e):
    R["errors"].append([S["step"], str(where) + ": " + str(e)[:300]])


def ev(msg):
    R["events"].append([S["step"], str(msg)[:300]])


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def hue_class(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 60:
        return None
    c = mx - mn
    if c < 0.42 * mx:
        return None
    if mx == r:
        h = 60.0 * (((g - b) / c) % 6.0)
    elif mx == g:
        h = 60.0 * ((b - r) / c + 2.0)
    else:
        h = 60.0 * ((r - g) / c + 4.0)
    if h < 15 or h >= 340:
        return "red"
    if h < 40:
        return "orange"
    if h < 70:
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


def quat_mul(a, b):
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]


def rotmat(q):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


# ------------------------------------------------------------------ episode info
ctx = get_task_context()
goal = str(ctx.language)
R["goal"] = goal
max_steps = 1312
hz = 20.0
try:
    meta = robomme_plus.get_task_metadata()
    max_steps = int(meta["max_steps"])
    hz = float(meta["control_frequency_hz"])
except Exception as e:
    err("meta", e)
R["max_steps"] = max_steps
words = goal.lower().replace(",", " ").replace(".", " ").split()
targets = [w for w in words if w in COLOR_WORDS]
n_buttons = 2 if ("both" in words or "two" in words) else 1
R["targets"] = targets
R["n_buttons"] = n_buttons

state0 = get_robot_state()
base = state0.base_frame
emb = state0.embodiment
home_q = [float(v) for v in state0.joint_positions["primary"]]
down_q = [float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz]

obs0 = get_observation()
cam0 = obs0.cameras["front"]
Kmat = cam0.intrinsics.tolist()
fx = float(Kmat[0][0])
fy = float(Kmat[1][1])
ppx = float(Kmat[0][2])
ppy = float(Kmat[1][2])
cpos = [float(v) for v in cam0.camera_pose.position]
Rc = rotmat([float(v) for v in cam0.camera_pose.quaternion_wxyz])
rows0 = cam0.rgb.tolist()
dep0 = cam0.depth_m.tolist()
H0 = len(rows0)
W0 = len(rows0[0])


def ray(u, v):
    a = (u - ppx) / fx
    b = (v - ppy) / fy
    return (Rc[0][0] * a + Rc[0][1] * b + Rc[0][2],
            Rc[1][0] * a + Rc[1][1] * b + Rc[1][2],
            Rc[2][0] * a + Rc[2][1] * b + Rc[2][2])


def pix3d(u, v, d):
    r = ray(u, v)
    return (cpos[0] + r[0] * d, cpos[1] + r[1] * d, cpos[2] + r[2] * d)


# calibration self-check against the public projection (OpenCV convention assumed)
STRIDE = 3
SAMP = []
for v in range(12, H0, STRIDE):
    for u in range(0, W0, STRIDE):
        r = ray(u + 0.5, v + 0.5)
        SAMP.append((u, v, r[0], r[1], r[2]))
R["n_samples"] = len(SAMP)

# ------------------------------------------------------------------ cubes at t0 (goal colours)
cube_xy = {}
for cname in targets:
    n = 0
    su = 0.0
    sv = 0.0
    for v in range(H0):
        row = rows0[v]
        for u in range(0, W0):
            p = row[u]
            if hue_class(p[0], p[1], p[2]) == cname:
                n += 1
                su += u
                sv += v
    if n >= 8:
        uc = su / n
        vc = sv / n
        # median 3D of the colour pixels near the centroid
        xs = []
        ys = []
        zs = []
        for v in range(max(0, int(vc) - 8), min(H0, int(vc) + 9)):
            for u in range(max(0, int(uc) - 8), min(W0, int(uc) + 9)):
                p = rows0[v][u]
                if hue_class(p[0], p[1], p[2]) == cname:
                    q = pix3d(u + 0.5, v + 0.5, float(dep0[v][u]))
                    xs.append(q[0])
                    ys.append(q[1])
                    zs.append(q[2])
        if len(xs) >= 5:
            cube_xy[cname] = (quantile(xs, 0.5), quantile(ys, 0.5), quantile(zs, 0.9), n)
R["cubes"] = {k: [round(v[0], 4), round(v[1], 4), round(v[2], 4), v[3]] for k, v in cube_xy.items()}
try:
    # cross-check the projection with the provider's point cloud for one SAM cube mask
    got = segment_text("front", "cube")
    if got.ok and len(got.segmentations) > 0:
        pc = mask_to_point_cloud(got.segmentations[0], "front", target_frame=base)
        pts = [[float(p[0]), float(p[1]), float(p[2])] for p in pc.points]
        m = got.segmentations[0].mask.tolist()
        mine = []
        for v in range(H0):
            for u in range(W0):
                if m[v][u]:
                    mine.append(pix3d(u + 0.5, v + 0.5, float(dep0[v][u])))
        if pts and mine:
            R["calib_check"] = [[round(quantile([p[i] for p in pts], 0.5), 4) for i in range(3)],
                                [round(quantile([p[i] for p in mine], 0.5), 4) for i in range(3)]]
except Exception as e:
    err("calib_check", e)

# ------------------------------------------------------------------ buttons
buttons = []
try:
    got = segment_text("front", "button")
    cands = []
    if got.ok:
        for cand in got.segmentations[:6]:
            try:
                pc = mask_to_point_cloud(cand, "front", target_frame=base)
                pts = [[float(p[0]), float(p[1]), float(p[2])] for p in pc.points]
                if len(pts) < 10:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                zs = [p[2] for p in pts]
                span = max(quantile(xs, 0.9) - quantile(xs, 0.1), quantile(ys, 0.9) - quantile(ys, 0.1))
                m = cand.mask.tolist()
                bright = 0
                dark = 0
                sat = 0
                tot = 0
                for v in range(H0):
                    mr = m[v]
                    for u in range(W0):
                        if mr[u]:
                            p = rows0[v][u]
                            tot += 1
                            if hue_class(p[0], p[1], p[2]) is not None:
                                sat += 1
                            elif min(p) > 150:
                                bright += 1
                            elif max(p) < 110:
                                dark += 1
                c = {"score": round(float(cand.score), 3), "span": round(span, 3),
                     "xy": [round(quantile(xs, 0.5), 4), round(quantile(ys, 0.5), 4)],
                     "top": round(quantile(zs, 0.9), 4), "bright": bright, "dark": dark,
                     "sat": sat, "tot": tot}
                c["ok"] = (span <= 0.08 and tot > 0 and sat < 0.3 * tot and bright >= 0.1 * tot
                           and dark >= 0.05 * tot)
                cands.append(c)
            except Exception as e:
                err("button cand", e)
    R["button_cands"] = cands
    ordered = [c for c in cands if c["ok"]] + [c for c in cands if (not c["ok"]) and c["span"] <= 0.08
                                                and c["tot"] > 0 and c["sat"] < 0.3 * c["tot"]]
    for c in ordered:
        if len(buttons) < n_buttons:
            dup = False
            for b in buttons:
                if (b["xy"][0] - c["xy"][0]) ** 2 + (b["xy"][1] - c["xy"][1]) ** 2 < 0.03 ** 2:
                    dup = True
            if not dup:
                buttons.append(c)
except Exception as e:
    err("buttons", e)
for b in buttons:
    try:
        bx0, by0, bt = b["xy"][0], b["xy"][1], b["top"]
        sx = 0.0
        sy = 0.0
        sz = []
        n = 0
        for v in range(H0):
            for u in range(W0):
                d = float(dep0[v][u])
                q = pix3d(u + 0.5, v + 0.5, d)
                if (q[0] - bx0) ** 2 + (q[1] - by0) ** 2 > 0.04 ** 2:
                    continue
                if q[2] < bt - 0.008 or q[2] > bt + 0.006:
                    continue
                p = rows0[v][u]
                if max(p) - min(p) > 30 or min(p) < 110 or max(p) > 215:
                    continue
                sx += q[0]
                sy += q[1]
                sz.append(q[2])
                n += 1
        b["cap_n"] = n
        if n >= 8:
            b["sam_xy"] = [bx0, by0]
            b["xy"] = [round(sx / n, 4), round(sy / n, 4)]
            b["top"] = round(quantile(sz, 0.5), 4)
    except Exception as e:
        err("cap", e)
R["buttons"] = buttons


# ------------------------------------------------------------------ perception per tick
def frame_points(obs):
    cam = obs.cameras["front"]
    rows = cam.rgb.tolist()
    dep = cam.depth_m.tolist()
    try:
        e = obs.robot_state.end_effector_poses["primary"].position
    except Exception:
        e = get_robot_state().end_effector_poses["primary"].position
    ex = float(e[0])
    ey = float(e[1])
    ez = float(e[2])
    zhi = S["zhi"]
    pts = []
    for s in SAMP:
        p = rows[s[1]][s[0]]
        mn = min(p)
        if mn < 90:
            continue
        if max(p) - mn > 50:
            continue
        d = dep[s[1]][s[0]]
        z = cpos[2] + s[4] * d
        if z < ZLO or z > zhi:
            continue
        x = cpos[0] + s[2] * d
        y = cpos[1] + s[3] * d
        if z > ez - 0.012 and abs(x - ex) < 0.05 and abs(y - ey) < 0.12:
            continue
        if x * x + y * y < 0.04:
            continue  # robot's own base/column (within 0.2 m of the base origin)
        pts.append((x, y, z))
    return pts


def init_tracks(pts):
    cells = {}
    for p in pts:
        key = (int(math.floor(p[0] / 0.01)), int(math.floor(p[1] / 0.01)))
        cells[key] = cells.get(key, 0) + 1
    remaining = list(pts)
    tracks = []
    for it in range(14):
        if len(remaining) < MINP:
            break
        cells = {}
        for p in remaining:
            key = (int(math.floor(p[0] / 0.01)), int(math.floor(p[1] / 0.01)))
            cells[key] = cells.get(key, 0) + 1
        best = None
        bestc = 0
        for key in cells:
            c = 0
            for dx in (-2, -1, 0, 1, 2):
                for dy in (-2, -1, 0, 1, 2):
                    c += cells.get((key[0] + dx, key[1] + dy), 0)
            if c > bestc:
                bestc = c
                best = key
        if best is None or bestc < 2 * MINP:
            break
        cx = (best[0] + 0.5) * 0.01
        cy = (best[1] + 0.5) * 0.01
        for k in range(4):
            sx = 0.0
            sy = 0.0
            n = 0
            for p in remaining:
                if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 < 0.035 ** 2:
                    sx += p[0]
                    sy += p[1]
                    n += 1
            if n == 0:
                break
            cx = sx / n
            cy = sy / n
        keep = []
        n = 0
        for p in remaining:
            if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 < 0.04 ** 2:
                n += 1
            else:
                keep.append(p)
        if n < 2 * MINP:
            break
        remaining = keep
        tracks.append({"x": cx, "y": cy, "vx": 0.0, "vy": 0.0, "n": n, "lost": 0,
                       "tx": cx, "ty": cy, "top": None, "n0": n})
    return tracks


def update_tracks(pts):
    T = S["tracks"]
    K = len(T)
    preds = []
    for t in T:
        preds.append([t["x"] + t["vx"], t["y"] + t["vy"]])
    r2 = ASSOC_R * ASSOC_R
    groups = None
    orphans = []
    for it in range(2):
        groups = [[] for k in range(K)]
        orphans = []
        for p in pts:
            best = -1
            bd = r2
            for k in range(K):
                if S["carried"] == k:
                    continue
                dx = p[0] - preds[k][0]
                dy = p[1] - preds[k][1]
                d = dx * dx + dy * dy
                if d < bd:
                    bd = d
                    best = k
            if best >= 0:
                groups[best].append(p)
            else:
                orphans.append(p)
        for k in range(K):
            g = groups[k]
            if len(g) >= MINP:
                preds[k] = [sum(p[0] for p in g) / len(g), sum(p[1] for p in g) / len(g)]
    for k in range(K):
        t = T[k]
        g = groups[k]
        if S["carried"] == k:
            continue
        if len(g) >= MINP:
            nx = preds[k][0]
            ny = preds[k][1]
            t["vx"] = 0.6 * t["vx"] + 0.4 * (nx - t["x"])
            t["vy"] = 0.6 * t["vy"] + 0.4 * (ny - t["y"])
            ox = t["tx"] - t["x"]
            oy = t["ty"] - t["y"]
            t["x"] = nx
            t["y"] = ny
            t["n"] = len(g)
            t["lost"] = 0
            top = quantile([p[2] for p in g], 0.9)
            tp = [p for p in g if p[2] >= top - 0.008]
            if len(tp) >= 4:
                t["tx"] = sum(p[0] for p in tp) / len(tp)
                t["ty"] = sum(p[1] for p in tp) / len(tp)
                t["top"] = top
                t["toppts"] = tp
            else:
                t["tx"] = nx + ox
                t["ty"] = ny + oy
        else:
            t["x"] = preds[k][0]
            t["y"] = preds[k][1]
            t["tx"] = t["tx"] + t["vx"]
            t["ty"] = t["ty"] + t["vy"]
            t["vx"] *= 0.7
            t["vy"] *= 0.7
            t["n"] = len(g)
            t["lost"] += 1
    respawn(orphans)


def respawn(orphans):
    """A track lost >= 8 ticks while exactly one dense untracked cluster exists -> move it there."""
    T = S["tracks"]
    lost = [k for k in range(len(T)) if T[k]["lost"] >= 8 and S["carried"] != k]
    if not lost or len(orphans) < 2 * MINP:
        return
    cl = init_tracks(orphans)
    if len(cl) != 1:
        return
    c = cl[0]
    best = lost[0]
    bd = None
    for k in lost:
        d = (T[k]["x"] - c["x"]) ** 2 + (T[k]["y"] - c["y"]) ** 2
        if bd is None or d < bd:
            bd = d
            best = k
    t = T[best]
    ev("respawn track %d lost %d -> (%.3f,%.3f) jump %.3f" % (best, t["lost"], c["x"], c["y"], math.sqrt(bd)))
    R.setdefault("respawns", []).append([S["step"], best, round(math.sqrt(bd), 3)])
    t["x"] = c["x"]
    t["y"] = c["y"]
    t["tx"] = c["x"]
    t["ty"] = c["y"]
    t["vx"] = 0.0
    t["vy"] = 0.0
    t["lost"] = 0
    t["n"] = c["n"]


def perceive(obs):
    try:
        S["step"] = int(round(float(obs.timestamp_s) * hz))
    except Exception:
        pass
    pts = frame_points(obs)
    S["band_hist"].append(len(pts))
    if S["tracks"] is None:
        # covers appear abruptly; initialise two ticks after the band fills
        h = S["band_hist"]
        if len(h) >= 3 and h[-1] >= 60 and h[-2] >= 60 and h[-3] >= 60:
            tracks = init_tracks(pts)
            if tracks:
                tops = quantile([p[2] for p in pts], 0.95)
                S["top"] = tops
                S["zhi"] = tops + 0.012
                S["tracks"] = tracks
                assign_targets()
                ev("tracks init: %d covers, top %.3f" % (len(tracks), tops))
    else:
        update_tracks(pts)
        sp = 0.0
        for t in S["tracks"]:
            v = math.sqrt(t["vx"] ** 2 + t["vy"] ** 2)
            if v > sp:
                sp = v
        if sp < 0.0008:
            S["quiet"] += 1
        else:
            S["quiet"] = 0
    if S["step"] % 4 == 0:
        tr = []
        if S["tracks"]:
            tr = [[round(t["x"] * 1000), round(t["y"] * 1000), t["n"]] for t in S["tracks"]]
        R["log"].append([S["step"], len(pts), tr])


def assign_targets():
    T = S["tracks"]
    S["target_idx"] = {}
    used = []
    for cname in targets:
        if cname not in cube_xy:
            continue
        cx, cy = cube_xy[cname][0], cube_xy[cname][1]
        best = None
        bd = 0.05 ** 2
        for k in range(len(T)):
            if k in used:
                continue
            d = (T[k]["x"] - cx) ** 2 + (T[k]["y"] - cy) ** 2
            if d < bd:
                bd = d
                best = k
        if best is not None:
            S["target_idx"][cname] = best
            used.append(best)
    R["target_idx"] = dict(S["target_idx"])


# ------------------------------------------------------------------ control helpers
def send(q, grip):
    S["cmd"] = [float(v) for v in q]
    reply = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=[float(v) for v in q],
        gripper_position=float(grip), embodiment=emb)}))
    try:
        if bool(reply.terminated) or bool(reply.truncated):
            S["ended"] = True
    except Exception:
        pass
    if not S["ended"]:
        try:
            obs = reply.observation
            cam = obs.cameras["front"]
        except Exception:
            obs = get_observation()
        try:
            perceive(obs)
        except Exception as e:
            err("perceive", e)
    return reply


def cur_state():
    st = get_robot_state()
    q = [float(v) for v in st.joint_positions["primary"]]
    p = [float(v) for v in st.end_effector_poses["primary"].position]
    return q, p


def hold(n, grip):
    q, p = cur_state()
    for k in range(n):
        if S["ended"]:
            return
        send(q, grip)


def servo(desired_fn, quat, grip, max_ticks, tol, max_dq=0.045, settle_ok=1, xy_only=False):
    """Per-tick IK servo toward desired_fn() (re-evaluated every tick). The *commanded* joints are
    rate-limited (max_dq per tick) starting from the previous command, so the PD lag stays small.
    Returns True when the measured end-effector is within tol of the desired position."""
    okc = 0
    q, p = cur_state()
    cmdq = S.get("cmd")
    if cmdq is None or max(abs(cmdq[i] - q[i]) for i in range(7)) > 0.15:
        cmdq = list(q)
    for k in range(max_ticks):
        if S["ended"]:
            return False
        if k > 0:
            q, p = cur_state()
        des = desired_fn()
        if xy_only:
            e = math.sqrt((p[0] - des[0]) ** 2 + (p[1] - des[1]) ** 2 + (0.3 * (p[2] - des[2])) ** 2)
        else:
            e = math.sqrt((p[0] - des[0]) ** 2 + (p[1] - des[1]) ** 2 + (p[2] - des[2]) ** 2)
        if e < tol:
            okc += 1
            if okc >= settle_ok:
                return True
        else:
            okc = 0
        ik = solve_ik(Pose(des, quat, base), seed_joints=cmdq)
        if not ik.ok:
            ev("ik failed at %s" % [round(v, 3) for v in des])
            send(cmdq, grip)
            continue
        tq = [float(v) for v in ik.joint_positions]
        cmd = []
        for i in range(7):
            d = tq[i] - cmdq[i]
            if d > max_dq:
                d = max_dq
            if d < -max_dq:
                d = -max_dq
            cmd.append(cmdq[i] + d)
        cmdq = cmd
        send(cmd, grip)
    return False


def const(p):
    def f():
        return p
    return f


def yaw_quat(theta):
    return quat_mul([math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)], down_q)


def cover_yaw(pts):
    if not pts or len(pts) < 5:
        return 0.0
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    best = 0.0
    bestv = None
    for k in range(18):
        a = k * math.pi / 36.0
        c = math.cos(a)
        s = math.sin(a)
        us = [(p[0] - mx) * c + (p[1] - my) * s for p in pts]
        vs = [-(p[0] - mx) * s + (p[1] - my) * c for p in pts]
        v = (max(us) - min(us)) * (max(vs) - min(vs))
        if bestv is None or v < bestv:
            bestv = v
            best = a
    if best > math.pi / 4.0:
        best -= math.pi / 2.0
    return best


# ------------------------------------------------------------------ 1) press the buttons now
R["press"] = []
PRESS_EARLIEST = 62   # v01: a contact at step 62 did not end the episode; v02: one at step 40 did
for bi in range(len(buttons)):
    b = buttons[bi]
    if S["ended"]:
        break
    try:
        bx, by, btop = b["xy"][0], b["xy"][1], b["top"]
        info = {"xy": [bx, by], "btop": btop, "step0": S["step"], "trace": []}
        hi = btop + 0.06
        info["above_ok"] = servo(const([bx, by, hi]), down_q, 0.0, 60, 0.004, max_dq=0.045,
                                 settle_ok=2, xy_only=True)
        if bi == 0:
            while not S["ended"] and S["step"] < PRESS_EARLIEST:
                servo(const([bx, by, hi]), down_q, 0.0, 1, 0.0, max_dq=0.045)
        q, p = cur_state()
        info["ee_above"] = [round(v, 4) for v in p]
        info["step_above"] = S["step"]
        # fast to just above the cap
        servo(const([bx, by, btop + 0.012]), down_q, 0.0, 14, 0.003, max_dq=0.04, settle_ok=1)
        zdeep = btop - 0.012
        zc = [btop + 0.012]

        def press_fn():
            zc[0] = max(zdeep, zc[0] - 0.002)
            return [bx, by, zc[0]]

        info["step_press"] = S["step"]
        for k in range(16):
            if S["ended"]:
                break
            servo(press_fn, down_q, 0.0, 1, 0.0, max_dq=0.03)
            q, p = cur_state()
            info["trace"].append([S["step"], round(zc[0], 4), round(p[0], 4), round(p[1], 4), round(p[2], 4)])
        # slow release
        zr = [zc[0]]

        def rel_fn():
            zr[0] = min(btop + 0.02, zr[0] + 0.003)
            return [bx, by, zr[0]]

        for k in range(9):
            if S["ended"]:
                break
            servo(rel_fn, down_q, 0.0, 1, 0.0, max_dq=0.03)
            q, p = cur_state()
            info["trace"].append([S["step"], round(zr[0], 4), round(p[0], 4), round(p[1], 4), round(p[2], 4)])
        info["step_done"] = S["step"]
        info["ended_after"] = S["ended"]
        R["press"].append(info)
        if not S["ended"]:
            info["retreat"] = servo(const([bx, by, hi + 0.01]), down_q, 0.0, 15, 0.012, max_dq=0.045)
    except Exception as e:
        err("press", e)
ev("presses done")

# back to an observation pose (home) while tracking; commanded joints rate-limited
try:
    q, p = cur_state()
    cmdq = S.get("cmd") or q
    for k in range(80):
        if S["ended"]:
            break
        q, p = cur_state()
        if max(abs(home_q[i] - q[i]) for i in range(7)) < 0.02:
            break
        cmd = []
        for i in range(7):
            d = home_q[i] - cmdq[i]
            if d > 0.05:
                d = 0.05
            if d < -0.05:
                d = -0.05
            cmd.append(cmdq[i] + d)
        cmdq = cmd
        send(cmd, 1.0)
except Exception as e:
    err("home", e)
ev("at home")

# ------------------------------------------------------------------ 2) wait for the shuffle to settle
PICK_RESERVE = 150 * max(1, len(targets)) + 40
WATCH_CAP = max_steps - PICK_RESERVE
while not S["ended"] and S["step"] < WATCH_CAP:
    hold(1, 1.0)
    if S["tracks"] is not None and S["quiet"] >= 20:
        R["quiet_at"] = S["step"]
        break
ev("watch end; quiet=%d" % S["quiet"])
R["tracks_before_pick"] = [[round(t["tx"], 4), round(t["ty"], 4), t["n"], t["lost"]] for t in (S["tracks"] or [])]

# ------------------------------------------------------------------ 3) pick each target in order
R["picks"] = []
if S["tracks"] is not None:
    for cname in targets:
        if S["ended"]:
            break
        if cname not in S.get("target_idx", {}):
            ev("no track for " + cname)
            continue
        k = S["target_idx"][cname]
        info = {"color": cname, "track": k}
        try:
            T = S["tracks"]
            top = T[k]["top"] if T[k]["top"] is not None else S["top"]
            info["top"] = round(top, 4)

            def above_fn():
                t = S["tracks"][k]
                return [t["tx"] + 2.0 * t["vx"], t["ty"] + 2.0 * t["vy"], top + 0.09]

            info["approach"] = servo(above_fn, down_q, 1.0, 70, 0.012, max_dq=0.06, settle_ok=2)
            t = S["tracks"][k]
            yaw = cover_yaw(t.get("toppts", []))
            info["yaw_deg"] = round(yaw * 180.0 / math.pi, 1)
            gq = yaw_quat(yaw)
            info["rot"] = servo(above_fn, gq, 1.0, 25, 0.012, max_dq=0.06, settle_ok=2)
            gz = max(0.02, top - 0.025)
            info["gz"] = round(gz, 4)
            zc = [top + 0.09]
            # freeze the target XY for the descent: once the hand is low it hides/excludes the
            # cover's own points, so extrapolate the last clean estimate with its velocity
            t = S["tracks"][k]
            frz = [t["tx"], t["ty"], t["vx"], t["vy"], S["step"]]
            info["frozen_xy"] = [round(frz[0], 4), round(frz[1], 4)]
            info["track_n_lost"] = [t["n"], t["lost"], t["n0"]]
            info["step_descend"] = S["step"]

            def descend_fn():
                zc[0] = max(gz, zc[0] - 0.012)
                el = S["step"] - frz[4]
                return [frz[0] + frz[2] * el, frz[1] + frz[3] * el, zc[0]]

            info["descend"] = servo(descend_fn, gq, 1.0, 30, 0.008, max_dq=0.05, settle_ok=1)
            q, p = cur_state()
            info["ee_at_grasp"] = [round(v, 4) for v in p]
            info["track_at_grasp"] = [round(S["tracks"][k]["tx"], 4), round(S["tracks"][k]["ty"], 4)]
            close_gripper()
            S["cmd"] = None
            S["carried"] = k
            lift = [p[0], p[1], gz + 0.20]
            info["lift"] = servo(const(lift), gq, 0.0, 40, 0.015, max_dq=0.06)
            hold(6, 0.0)
            try:
                info["grip"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
            except Exception as e:
                err("grip", e)
            if S["ended"]:
                R["picks"].append(info)
                break
            # put the cover back where it came from, then release
            servo(const([p[0], p[1], gz + 0.01]), gq, 0.0, 40, 0.01, max_dq=0.06)
            open_gripper()
            S["cmd"] = None
            S["carried"] = None
            T = S["tracks"]
            T[k]["x"] = p[0]
            T[k]["y"] = p[1]
            T[k]["tx"] = p[0]
            T[k]["ty"] = p[1]
            T[k]["vx"] = 0.0
            T[k]["vy"] = 0.0
            servo(const([p[0], p[1], top + 0.10]), gq, 1.0, 25, 0.015, max_dq=0.06)
        except Exception as e:
            err("pick " + cname, e)
        R["picks"].append(info)
R["final_step"] = S["step"]
R["ended"] = S["ended"]
result = R
