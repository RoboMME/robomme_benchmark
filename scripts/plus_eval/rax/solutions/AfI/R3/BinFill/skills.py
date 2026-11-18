"""Task-level skills: detect cubes/bin/button, pick, place, press."""
import numpy as np
from perception import world_points, color_masks, components
from motion import DOWN, rotvec_to_R, R_to_rotvec

R_DOWN = np.diag([1.0, -1.0, -1.0])
CUBE = 0.04          # approx cube edge (finger gap when closed ~0.036)


def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def yaw_rot(theta):
    return R_to_rotvec(Rz(theta) @ R_DOWN)


def square_yaw(xy):
    """Yaw (rad, in [-pi/4, pi/4)) minimising bounding-box area of xy points."""
    if len(xy) < 5:
        return 0.0
    xy = xy - xy.mean(0)
    best, bt = 1e9, 0.0
    for t in np.radians(np.arange(-45, 45, 2.0)):
        c, s = np.cos(t), np.sin(t)
        p = xy @ np.array([[c, -s], [s, c]])
        a = np.ptp(p[:, 0]) * np.ptp(p[:, 1])
        if a < best:
            best, bt = a, t
    return float(bt)


def detect_cubes(obs, cam, colors=("red", "blue", "green"), min_px=6):
    """List of dicts: color, xyz (top-surface median), n, yaw, uv."""
    rgb = obs[(cam, "rgb")]
    P = world_points(obs, cam)
    res = []
    for name, m in color_masks(rgb).items():
        if name not in colors:
            continue
        for c in components(m, min_px):
            pts = P[c[:, 0], c[:, 1]]
            ok = np.isfinite(pts).all(1)
            pts = pts[ok]
            if len(pts) < 3:
                continue
            ztop = np.percentile(pts[:, 2], 90)
            top = pts[pts[:, 2] > ztop - 0.006]
            xyz = np.median(top, 0)
            edge = bool((c.min() <= 1) or (c.max() >= rgb.shape[0] - 2))
            res.append({"color": name, "xyz": xyz, "n": int(len(c)), "zmax": float(ztop), "edge": edge,
                        "yaw": square_yaw(top[:, :2]), "uv": (float(c[:, 1].mean()), float(c[:, 0].mean()))})
    return res


def yaw_near(cube_yaw, ref):
    """Cube-aligned gripper yaw (cube_yaw + k*90deg) closest to ref, within [-110, 110] deg."""
    opts = [cube_yaw + k * np.pi / 2 for k in range(-3, 4)]
    opts = [t for t in opts if abs(t) <= np.radians(110)]
    return min(opts, key=lambda t: abs(t - ref))


def detect_bin(obs, cam="front"):
    """Bin interior: dark pixels near table height. Returns centre xyz or None."""
    rgb = obs[(cam, "rgb")].astype(int)
    P = world_points(obs, cam)
    dark = (rgb.max(-1) < 18)
    z = P[..., 2]
    dark &= np.isfinite(z) & (z < 0.06) & (z > -0.02)
    comps = components(dark, 20)
    if not comps:
        return None
    c = max(comps, key=len)
    pts = P[c[:, 0], c[:, 1]]
    lo, hi = np.percentile(pts[:, :2], 5, 0), np.percentile(pts[:, :2], 95, 0)
    return {"xyz": np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, float(np.median(pts[:, 2]))]),
            "lo": lo, "hi": hi, "n": len(c)}


def nearest(cands, xy):
    if not cands:
        return None
    return min(cands, key=lambda c: np.linalg.norm(c["xyz"][:2] - np.asarray(xy)[:2]))


def choose_yaw(cube_yaw, neighbour_dir=None):
    """Gripper yaw aligned to cube faces; if a neighbour is close, put the finger axis
    (yaw+90deg) perpendicular to the neighbour direction. Result in [-90, 90] deg."""
    opts = []
    for t in (cube_yaw, cube_yaw + np.pi / 2, cube_yaw - np.pi / 2):
        if -np.pi / 2 - 1e-6 <= t <= np.pi / 2 + 1e-6:
            opts.append(t)
    if neighbour_dir is None:
        return min(opts, key=abs)
    return min(opts, key=lambda t: (abs(np.cos(t + np.pi / 2 - neighbour_dir)), abs(t)))


def detect_button(obs, cam="front", bin_xy=None):
    """Low-saturation compact object with a raised cap (button). Returns dict or None.
    Rejects large flat gray regions (e.g. a conveyor mat) by world extent / cap height."""
    rgb = obs[(cam, "rgb")].astype(int)
    P = world_points(obs, cam)
    mx, mn = rgb.max(-1), rgb.min(-1)
    gray = ((mx - mn) < 10) & (mx > 35)
    z = P[..., 2]; x = P[..., 0]
    gray &= np.isfinite(z) & (z > 0.004) & (z < 0.06) & (x > -0.45) & (x < 0.3)
    if bin_xy is not None:
        gray &= np.abs(P[..., :2] - np.asarray(bin_xy)[:2]).max(-1) > 0.065
    best = None
    for c in components(gray, 12):
        pts = P[c[:, 0], c[:, 1]]
        ext = np.percentile(pts[:, :2], 95, 0) - np.percentile(pts[:, :2], 5, 0)
        if ext.max() > 0.10:
            continue
        zr = np.percentile(pts[:, 2], 95) - np.percentile(pts[:, 2], 5)
        cap = pts[pts[:, 2] > np.percentile(pts[:, 2], 95) - 0.006]
        score = len(c) * (1.0 if zr > 0.008 else 0.3)
        if best is None or score > best[0]:
            best = (score, pts, cap, c)
    if best is None:
        return None
    _, pts, cap, c = best
    return {"xyz": np.median(cap, 0), "n": len(c), "base_xy": np.median(pts[:, :2], 0)}


def wrist_bin_refine(obs, guess_xy):
    """Bin interior centre from the wrist camera (dark, z<0.05, near guess)."""
    rgb = obs[("wrist", "rgb")].astype(int)
    P = world_points(obs, "wrist")
    z = P[..., 2]
    dark = (rgb.max(-1) < 18) & np.isfinite(z) & (z < 0.05) & (z > -0.02)
    dark &= np.linalg.norm(P[..., :2] - np.asarray(guess_xy)[:2], axis=-1) < 0.08
    comps = components(dark, 50)
    if not comps:
        return None
    c = max(comps, key=len)
    pts = P[c[:, 0], c[:, 1]]
    lo, hi = np.percentile(pts[:, :2], 3, 0), np.percentile(pts[:, :2], 97, 0)
    return {"xy": (lo + hi) / 2, "lo": lo, "hi": hi, "n": len(c)}


# ---------------------------------------------------------------- moving scenes
TRACK = 0.38      # measured fraction of commanded delta achieved per step (ep1: 0.376-0.39)
V_MAX = 0.008     # m/step clamp on estimated conveyor velocity (observed 0.0025)


def estimate_velocity(prev, cur, dsteps, max_match=0.06):
    """Median xy displacement per step between matched same-colour detections.
    prev/cur: lists of detections (table cubes only). Returns (v or None, n_matches)."""
    if dsteps <= 0 or not prev or not cur:
        return None, 0
    disp = []
    used = set()
    for c in cur:
        best, bd = None, max_match
        for k, p in enumerate(prev):
            if p["color"] != c["color"] or k in used:
                continue
            d = np.linalg.norm(c["xyz"][:2] - p["xyz"][:2])
            if d < bd:
                best, bd = k, d
        if best is not None:
            used.add(best)
            disp.append((c["xyz"][:2] - prev[best]["xyz"][:2]) / dsteps)
    if len(disp) < 2:
        return None, len(disp)
    v = np.median(np.array(disp), 0)
    if np.linalg.norm(v) > V_MAX:
        v = v / np.linalg.norm(v) * V_MAX
    return v, len(disp)


def _ff_action(v):
    return v / (TRACK * 0.05)


def _xy_cmd(err, v, gain=2.2, vmax=1.0):
    a = gain * err / 0.05 + _ff_action(v)
    n = np.linalg.norm(a)
    return a * vmax / n if n > vmax else a


def find_target(obs, color, pred_xy, ztop, radius=0.035):
    best, bd = None, radius
    for c in detect_cubes(obs, "wrist", (color,)):
        if c["n"] < 60 or c.get("edge") or abs(c["zmax"] - ztop) > 0.015:
            continue
        d = np.linalg.norm(c["xyz"][:2] - pred_xy)
        if d < bd:
            best, bd = c, d
    return best


MEAS_SIGMA = 0.0015    # m, wrist top-face position noise (assumed)
PRIOR_SIGMA = 0.0005   # m/step, uncertainty of the scene-velocity prior


def fuse_track(hist, v_prior, s_now):
    """Line fit of (step, xy) history with a Gaussian prior on the slope.
    Returns (velocity, position at s_now)."""
    hist = hist[-12:]
    ts = np.array([h[0] for h in hist], float); xs = np.array([h[1] for h in hist])
    tm = ts.mean(); xm = xs.mean(0)
    stt = ((ts - tm) ** 2).sum()
    w_data = stt / MEAS_SIGMA ** 2
    w_prior = 1.0 / PRIOR_SIGMA ** 2
    slope_data = ((ts - tm)[:, None] * (xs - xm)).sum(0) / stt if stt > 0 else np.asarray(v_prior)
    v = (w_data * slope_data + w_prior * np.asarray(v_prior)) / (w_data + w_prior)
    if np.linalg.norm(v) > V_MAX:
        v = v / np.linalg.norm(v) * V_MAX
    return v, xm + v * (s_now - tm)


def servo_pick(R, det, v0, log=print, hover_dz=0.065, approach_steps=80):
    """Intercept a (possibly moving) cube: approach, wrist-servo at hover with velocity
    feedforward, descend while tracking, close while tracking, lift. Returns (held, info)."""
    color = det["color"]
    p = np.array(det["xyz"], float); t_meas = R.meta["step_id"]
    v = np.zeros(2) if v0 is None else np.array(v0, float)
    v_prior = v.copy()
    ztop = float(det["zmax"])
    hover_z = ztop + hover_dz
    grasp_z = max(0.012, ztop - 0.023)
    yaw = choose_yaw(det["yaw"], None)
    rot = yaw_rot(yaw)
    R.grip = -1.0

    def pred(step):
        q = p.copy(); q[:2] += v * (step - t_meas); return q

    def row(xy_a, z_a, rd, g):
        return list(xy_a) + [z_a] + list(np.clip(2.0 * rd / 0.5, -0.6, 0.6)) + [g]

    # ---- Phase A: approach intercept point at hover height
    for k in range(approach_steps):
        if R.ended:
            return False, {"why": "ended"}
        s = R.meta["step_id"]
        q = pred(s)
        dist = np.linalg.norm(q[:2] - R.pos[:2])
        lead = min(dist / 0.019, 15.0)
        aim = q[:2] + v * lead
        e = np.r_[aim - R.pos[:2], hover_z - R.pos[2]]
        rd = R.rot_delta(rot)
        if dist < 0.03 and abs(e[2]) < 0.02 and np.linalg.norm(rd) < 0.15:
            break
        a = 2.2 * e / 0.05
        a[:2] += _ff_action(v) * (dist < 0.06)
        n = np.linalg.norm(a)
        if n > 1.0:
            a = a / n
        R.step([row(a[:2], a[2], rd, -1)])
    # ---- Phase B: servo at hover using wrist detections
    hist = []          # (step, xy)
    miss = 0
    locked = False
    for k in range(45):
        if R.ended:
            return False, {"why": "ended"}
        s = R.meta["step_id"]
        c = find_target(R.obs, color, pred(s)[:2], ztop, radius=0.05 if not hist else 0.03)
        if c is not None:
            miss = 0
            hist.append((s, c["xyz"][:2].copy()))
            hist = hist[-10:]
            if len(hist) >= 3:
                v, p[:2] = fuse_track(hist, v_prior, s)
            else:
                p[:2] = c["xyz"][:2]
            p[2] = c["xyz"][2]; t_meas = s
            yaw = yaw_near(c["yaw"], yaw); rot = yaw_rot(yaw)
        else:
            miss += 1
            if miss >= 6:
                log(f"  servo lost target ({len(hist)} meas)")
                return False, {"why": "refine"}
        q = pred(s + 1)
        e = q[:2] - R.pos[:2]
        rd = R.rot_delta(rot)
        if len(hist) >= 5 and np.linalg.norm(pred(s)[:2] - R.pos[:2]) < 0.006 and np.linalg.norm(rd) < 0.06:
            locked = True
            break
        a = _xy_cmd(e, v)
        R.step([row(a, 2.2 * (hover_z - R.pos[2]) / 0.05, rd, -1)])
    if not locked and len(hist) < 2:
        log("  servo never locked")
        return False, {"why": "refine"}
    log(f"  servo {'locked' if locked else 'timeout'}: {color} p={np.round(pred(R.meta['step_id']),3)} "
        f"v={np.round(v*1000,2)}mm/step meas={len(hist)} yaw={np.degrees(yaw):.0f} grasp_z={grasp_z:.3f}")
    # ---- Phase C: descend while tracking
    for k in range(20):
        if R.ended:
            return False, {"why": "ended"}
        s = R.meta["step_id"]
        if R.pos[2] < grasp_z + 0.004:
            break
        if R.pos[2] > ztop + 0.04:
            c = find_target(R.obs, color, pred(s)[:2], ztop, radius=0.02)
            if c is not None:
                hist.append((s, c["xyz"][:2].copy())); hist = hist[-12:]
                v, p[:2] = fuse_track(hist, v_prior, s); t_meas = s
        e = pred(s + 1)[:2] - R.pos[:2]
        a = _xy_cmd(e, v)
        za = float(np.clip(2.2 * (grasp_z - R.pos[2]) / 0.05, -0.8, 0.8))
        R.step([row(a, za, R.rot_delta(rot), -1)])
    if R.pos[2] > grasp_z + 0.012:
        log(f"  descent blocked at z={R.pos[2]:.3f} -> abort")
        R.move_to(np.r_[R.pos[:2], hover_z], tol=0.01, max_steps=30)
        return False, {"why": "blocked"}
    # ---- close while tracking
    prev = R.state[6:8].sum()
    for k in range(12):
        if R.ended:
            return False, {"why": "ended"}
        s = R.meta["step_id"]
        e = pred(s + 1)[:2] - R.pos[:2]
        # follow the cube until the fingers touch it (gap stops near cube size)
        a = _xy_cmd(e, v) if prev > 0.045 else np.zeros(2)
        za = float(np.clip(2.2 * (grasp_z - R.pos[2]) / 0.05, -0.3, 0.3))
        R.step([list(a) + [za, 0, 0, 0, 1.0]])
        R.grip = 1.0
        cur = R.state[6:8].sum()
        if k >= 3 and abs(cur - prev) < 0.0005:
            break
        prev = cur
    fingers = R.state[6:8].sum()
    R.move_to(np.r_[R.pos[:2], 0.16], tol=0.012, max_steps=40)
    gap = R.state[6:8].sum()
    held = 0.025 < gap < 0.075
    log(f"  grasp fingers {fingers:.4f} after lift {gap:.4f} held={held}")
    return held, {"xyz": p, "v": v}
