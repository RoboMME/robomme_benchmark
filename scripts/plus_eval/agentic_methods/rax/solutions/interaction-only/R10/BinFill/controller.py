"""BinFill controller v1: count-and-place cubes of requested colors into the bin, then press button."""
import os, re, sys, time, json, math, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import Obs, Robot, masks, components, log

VERSION = "v6"
NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
       "a": 1, "an": 1}
COLORS = ("red", "blue", "green")
DETECT_COLORS = set(COLORS)
BIN_EXCLUDE_R = 0.065   # cubes whose centre is this close to the bin centre are in/on the bin (outer half-size ~0.05)
PICK_Z = 0.02          # TCP height for grasp (cube center); ep1 4/4 grasps
SAFE_Z = 0.13
RELEASE_Z = 0.12       # above bin rim (0.051)
BIN_OFFS = [(0.0, 0.0)]

def parse_instruction(text):
    t = text.lower()
    need = {}
    for m in re.finditer(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an)\s+(red|blue|green|yellow|orange|cyan|magenta|purple)\b", t):
        n = int(m.group(1)) if m.group(1).isdigit() else NUM[m.group(1)]
        need[m.group(2)] = need.get(m.group(2), 0) + n
    return need

def find_bin(o):
    rgb = o.rgb("front").astype(int)
    W = o.world_map("front")
    dark = (rgb.max(-1) < 12)
    dark[:20] = False
    zz = np.nan_to_num(W[..., 2], nan=-1.0)
    dark &= (zz > 0.003) & (zz < 0.06)
    comps = [cp for cp in components(dark, 15)]
    if not comps:
        return None
    big = max(comps, key=len)
    dark = np.zeros_like(dark); dark[big[:, 0], big[:, 1]] = True
    P = W[big[:, 0], big[:, 1]]
    P = P[np.isfinite(P).all(1)]
    if len(P) < 15:
        return None
    dc = np.median(P[:, :2], 0)
    zd = np.median(P[:, 2])
    cm = np.zeros(rgb.shape[:2], bool)
    for m in masks(o.rgb("front")).values():
        cm |= m
    Wf = W.reshape(-1, 3)
    sel = (~dark.reshape(-1)) & (~cm.reshape(-1)) & np.isfinite(Wf).all(1)
    Q = Wf[sel]
    Q = Q[(np.linalg.norm(Q[:, :2] - dc, axis=1) < 0.09) & (Q[:, 2] > zd + 0.008) & (Q[:, 2] < 0.08)]
    if len(Q) < 20:
        lo, hi = np.percentile(P[:, :2], 2, 0), np.percentile(P[:, :2], 98, 0)
        return (lo + hi) / 2, hi - lo, None
    zr = np.percentile(Q[:, 2], 90)
    Rr = Q[np.abs(Q[:, 2] - zr) < 0.006]
    lo, hi = np.percentile(Rr[:, :2], 3, 0), np.percentile(Rr[:, :2], 97, 0)
    plo, phi = np.percentile(P[:, :2], 2, 0), np.percentile(P[:, :2], 98, 0)
    c = 0.5 * ((lo + hi) / 2) + 0.5 * ((plo + phi) / 2)
    return c, hi - lo, float(zr)

def find_button(o):
    rgb = o.rgb("front").astype(int)
    W = o.world_map("front")
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = (abs(r - g) < 5) & (abs(g - b) < 5) & (r > 40) & (r < 140)
    gray[:30] = False
    z = W[..., 2]
    cap = gray & (z > 0.026) & (z < 0.045)
    base = gray & (z > 0.010) & (z < 0.022)
    ys, xs = np.nonzero(base)
    if len(ys) < 10:
        return None
    PB = W[ys, xs]
    cb = np.median(PB[:, :2], 0)
    ys2, xs2 = np.nonzero(cap)
    if len(ys2) >= 5:
        PC = W[ys2, xs2]
        PC = PC[np.linalg.norm(PC[:, :2] - cb, axis=1) < 0.05]
        if len(PC) >= 3:
            return np.median(PC[:, :2], 0), float(np.max(PC[:, 2]))
    return cb, 0.034

def cube_yaw(pts):
    """yaw of square footprint from top-face xy points (mod 90deg), in [-pi/4, pi/4)."""
    if len(pts) < 6:
        return 0.0
    q = pts - pts.mean(0)
    best, ba = 0.0, 1e9
    for a in np.linspace(0, math.pi / 2, 30, endpoint=False):
        c, s = math.cos(a), math.sin(a)
        u = q @ np.array([c, s]); v = q @ np.array([-s, c])
        area = (u.max() - u.min()) * (v.max() - v.min())
        if area < ba:
            ba, best = area, a
    if best >= math.pi / 4:
        best -= math.pi / 2
    return best

def other_mask(rgb):
    r, g, b = [rgb[..., i].astype(np.int32) for i in range(3)]
    mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b)
    return (mx - mn > 70) & (mx > 120)

def detect_cubes(o, binc, button):
    rgb = o.rgb("front")
    W = o.world_map("front")
    out = []
    for c, m in masks(rgb).items():
        if c not in DETECT_COLORS:
            continue
        for comp in components(m, 12):
            P = W[comp[:, 0], comp[:, 1]]
            P = P[np.isfinite(P).all(1)]
            if len(P) < 6:
                continue
            ztop = np.percentile(P[:, 2], 90)
            top = P[P[:, 2] > ztop - 0.006]
            xy = np.median(top[:, :2], 0)
            # cube center is ~half-size behind the visible top centroid? use top face points only
            d = {"color": c, "xy": xy, "ztop": float(ztop), "n": len(comp), "yaw": cube_yaw(top[:, :2])}
            d["in_bin"] = binc is not None and float(np.linalg.norm(xy - binc)) < BIN_EXCLUDE_R
            d["on_table"] = 0.03 < ztop < 0.075 and not d["in_bin"]
            out.append(d)
    return out

def holding(state):
    w = state[6] + state[7]
    return 0.028 < w < 0.075, w

def other_blobs(o):
    """xy of any saturated blob (cubes of any colour) for clearance."""
    W = o.world_map("front")
    out = []
    zm = np.nan_to_num(W[..., 2], nan=-1.0)
    for comp in components(other_mask(o.rgb("front")) & (zm > 0.025) & (zm < 0.08), 12):
        P = W[comp[:, 0], comp[:, 1]]
        P = P[np.isfinite(P).all(1)]
        if len(P) >= 6 and 0.02 < np.percentile(P[:, 2], 90) < 0.08:
            out.append(np.median(P[:, :2], 0))
    return out

def estimate_velocity(c1, c2, dt):
    """match cubes of same colour between two detections; return per-cube velocity list and belt median."""
    vs = []
    for a in c2:
        best, bd = None, 0.03 + 0.004 * dt
        for b in c1:
            if b["color"] != a["color"]:
                continue
            d = np.linalg.norm(a["xy"] - b["xy"])
            if d < bd:
                best, bd = b, d
        a["v"] = None
        a["matched"] = best is not None
        if best is not None:
            a["v"] = (a["xy"] - best["xy"]) / dt
            vs.append(a["v"])
    moving = [v for v in vs if np.linalg.norm(v) > 0.0008]
    vb = np.median(np.array(moving), 0) if len(moving) >= 1 else np.zeros(2)
    for a in c2:
        if a["v"] is None:
            a["v"] = np.zeros(2)
        elif np.linalg.norm(a["v"]) > 0.0008:
            a["v"] = vb if len(moving) >= 2 else a["v"]
        else:
            a["v"] = np.zeros(2)
    return vb

REACH_X = (-0.40, 0.20)
REACH_Y = (-0.36, 0.36)

def follow(R, p_ref, s_ref, v, z_of_k, nsteps, grip, chunk=3):
    """Track moving xy target p_ref + v*(step - s_ref) for nsteps; z given per step index."""
    k = 0
    while k < nsteps and not R.ended:
        n = min(chunk, nsteps - k)
        s0 = R.obs.step
        p = R.pos.copy()
        rc = R.rot_cmd(R.obs.state[3:6])
        rows = []
        for j in range(n):
            s = s0 + j + 1
            tgt = np.array([p_ref[0] + v[0] * (s - s_ref), p_ref[1] + v[1] * (s - s_ref), z_of_k(k + j)])
            d = np.clip((tgt - p) / 0.05, -1, 1)
            p = p + d * 0.05
            rows.append(list(d) + (list(rc) if j == 0 else [0, 0, 0]) + [grip])
        R.grip = grip
        R.step(rows)
        k += n
    return R.obs

def wrist_refine(o, color, pred_xy, tol=0.05):
    """Measure the target cube's top-face xy (and yaw) with the wrist camera near pred_xy."""
    try:
        W = o.world_map("wrist")
        m = masks(o.rgb("wrist"))[color]
    except Exception:
        return None
    best = None
    for comp in components(m, 20):
        P = W[comp[:, 0], comp[:, 1]]
        P = P[np.isfinite(P).all(1)]
        if len(P) < 15:
            continue
        zt = np.percentile(P[:, 2], 90)
        top = P[P[:, 2] > zt - 0.005]
        xy = np.median(top[:, :2], 0)
        d = np.linalg.norm(xy - pred_xy)
        if d < tol and (best is None or d < best[0]):
            best = (d, xy, float(zt), cube_yaw(top[:, :2]))
    return best

def main(ctx):
    t0 = time.time()
    R = Robot(ctx)
    o = R.observe()
    instr = o.meta.get("instruction", "")
    need = parse_instruction(instr)
    DETECT_COLORS.update(need.keys())
    log(VERSION, "instruction:", instr, "need:", need, "limits", o.meta.get("episode_limits"))
    home = o.state[:3].copy()
    home[2] = max(home[2], 0.45)
    fb = find_bin(o)
    bt = find_button(o)
    log("bin", None if fb is None else (fb[0].round(3).tolist(), fb[1].round(3).tolist(), fb[2]), "button",
        None if bt is None else (bt[0].round(3).tolist(), round(bt[1], 3)))
    binc = fb[0] if fb is not None else np.array([0.027, -0.147])
    rim_z = fb[2] if (fb is not None and fb[2] is not None) else 0.055
    try:
        if fb is not None:
            binc = locate_bin_wrist(R, binc)
    except Exception as e:
        log("ERR bin locate", repr(e))
    placed = {c: 0 for c in need}
    nplaced = 0
    idle_waits = 0
    vbelt = np.zeros(2)
    while not R.ended and any(placed[c] < need[c] for c in need):
        if time.time() - t0 > 1000 or (R.remaining is not None and R.remaining < 120):
            log("budget low; stop placing"); break
        try:
            R.yaw = 0.0
            if np.linalg.norm(R.pos - home) > 0.03:
                R.move_to(home, tol=0.03, max_iter=4, grip=-1)
            o1 = R.observe(); c1 = detect_cubes(o1, binc, bt)
            R.hold(4, grip=-1)
            o2 = R.observe(); c2 = detect_cubes(o2, binc, bt)
            dt = max(1, o2.step - o1.step)
            vbelt = estimate_velocity(c1, c2, dt)
            others = other_blobs(o2) + [cc["xy"] for cc in c2]
            s_now = o2.step
            log("step", s_now, "placed", placed, "vbelt", np.round(vbelt, 5).tolist(), "visible",
                [(c["color"], c["xy"].round(3).tolist(), round(c["ztop"], 3), np.round(c["v"], 4).tolist()) for c in c2 if c["on_table"]])
            cand = []
            for c in c2:
                if not (c["on_table"] and c["color"] in need and placed[c["color"]] < need[c["color"]]):
                    continue
                if not c.get("matched", False):
                    continue
                dist = np.linalg.norm(np.r_[c["xy"], SAFE_Z] - R.pos)
                T = int(dist / 0.04) + 20   # steps until grasp closes
                pg = c["xy"] + c["v"] * T
                pend = c["xy"] + c["v"] * (T + 25)
                if not (REACH_X[0] < pg[0] < REACH_X[1] and REACH_Y[0] < pg[1] < REACH_Y[1]):
                    continue
                if not (REACH_X[0] - 0.05 < pend[0] < REACH_X[1] + 0.05 and REACH_Y[0] - 0.05 < pend[1] < REACH_Y[1] + 0.05):
                    continue
                dd = [np.linalg.norm(c["xy"] - e) for e in others if np.linalg.norm(c["xy"] - e) > 0.012]
                clear = min(dd) if dd else 1.0
                cand.append((c, clear, T))
            if not cand:
                idle_waits += 1
                if idle_waits > 60:
                    log("no target cubes after long wait"); break
                R.hold(6, grip=-1)
                continue
            idle_waits = 0
            moving = np.linalg.norm(vbelt) > 0.0008
            cand.sort(key=lambda t: (-(min(t[1], 0.07)), -(need[t[0]["color"]] - placed[t[0]["color"]])))
            tgt, clear, T = cand[0]
            v = tgt["v"]
            log("pick", tgt["color"], tgt["xy"].round(3).tolist(), "v", np.round(v, 4).tolist(), "clear", round(clear, 3), "T", T)
            ok = pick_cube(R, tgt, v, s_now)
            if not ok:
                R.hold(3, grip=-1)
                continue
            off = BIN_OFFS[nplaced % len(BIN_OFFS)]
            R.yaw = 0.0
            R.move_to([binc[0], binc[1], rim_z + 0.09], tol=0.008, max_iter=8)
            R.move_to([binc[0], binc[1], rim_z + 0.055], tol=0.003, max_iter=5)
            ok, w = holding(R.obs.state)
            log("release at", R.pos.round(4).tolist(), "bin", np.round(binc, 4).tolist())
            if not ok:
                log("dropped before bin!", round(w, 4))
                continue
            R.hold(6, grip=-1)
            R.move_to([binc[0], binc[1], rim_z + 0.12], tol=0.02, max_iter=2)
            placed[tgt["color"]] += 1
            try:
                verify_drop(R, binc, rim_z, tgt["color"])
            except Exception as e:
                log("ERR verify", repr(e))
            nplaced += 1
            log("released", tgt["color"], "placed", placed, "step", R.obs.step, "t", round(time.time() - t0, 1))
        except Exception as e:
            log("ERR", repr(e), traceback.format_exc()[-800:])
            try:
                R.observe()
            except Exception:
                pass
    if not R.ended and bt is not None:
        try:
            press_button(R, bt)
        except Exception as e:
            log("ERR press", repr(e))
    log("END step", R.obs.step if R.obs else None, "ended", R.ended, "complete", R.complete, "placed", placed, "t", round(time.time() - t0, 1))

def locate_bin_wrist(R, approx):
    """Hover the wrist camera over the bin and take the centre of the black opening."""
    cam_off = np.array([0.046, 0.020])
    R.move_to([approx[0] - cam_off[0], approx[1] - cam_off[1], 0.26], tol=0.01, max_iter=6, grip=-1)
    R.hold(2)
    o = R.observe()
    rgb = o.rgb("wrist").astype(int)
    W = o.world_map("wrist")
    dark = rgb.max(-1) < 14
    best = None
    for comp in components(dark, 40):
        P = W[comp[:, 0], comp[:, 1]]
        P = P[np.isfinite(P).all(1)]
        if len(P) < 40:
            continue
        xy = np.median(P[:, :2], 0)
        d = np.linalg.norm(xy - approx)
        if d < 0.08 and (best is None or len(P) > best[0]):
            lo, hi = np.percentile(P[:, :2], 1, 0), np.percentile(P[:, :2], 99, 0)
            best = (len(P), (lo + hi) / 2, hi - lo, float(np.median(P[:, 2])))
    if best is None:
        log("wrist bin locate failed; keep", np.round(approx, 3).tolist())
        return approx
    log("wrist bin", "n", best[0], "center", best[1].round(4).tolist(), "size", best[2].round(3).tolist(), "z", round(best[3], 3), "front est", np.round(approx, 3).tolist())
    return best[1]

def pick_cube(R, tgt, v, s_ref):
    p_ref = tgt["xy"].copy()
    ztop = tgt["ztop"]
    v3 = np.r_[v, 0.0]
    cube = lambda s: p_ref + v * (s - s_ref)
    # 1. intercept: hover above the moving cube
    e = R.servo(lambda s: np.r_[cube(s), SAFE_Z], vff=v3, n_max=60, grip=-1, tol=0.012, chunk=3)
    R.servo(lambda s: np.r_[cube(s), SAFE_Z], vff=v3, n_max=4, grip=-1, tol=0.0, chunk=2)
    o = R.observe()
    pred = cube(o.step)
    ref = wrist_refine(o, tgt["color"], pred)
    if ref is not None:
        d, xy, zt, yaw = ref
        log("wrist refine", "err", round(d, 4), "xy", xy.round(3).tolist(), "ztop", round(zt, 3), "yaw", round(math.degrees(yaw), 1), "tcp", R.pos.round(3).tolist())
        p_ref, s_ref = xy, o.step
        if abs(zt - ztop) < 0.02:
            ztop = zt
    else:
        log("wrist refine failed; pred", pred.round(3).tolist(), "tcp", R.pos.round(3).tolist())
    cube = lambda s: p_ref + v * (s - s_ref)
    pz = min(max(ztop - 0.021, 0.012), 0.07)
    # 2. descend while tracking
    R.servo(lambda s: np.r_[cube(s), pz], vff=v3, n_max=25, grip=-1, tol=0.006, chunk=2, gain=0.6)
    log("at grasp", R.pos.round(4).tolist(), "cube pred", cube(R.obs.step).round(4).tolist(), "step", R.obs.step)
    # 3. close while tracking
    R.servo(lambda s: np.r_[cube(s), pz], vff=v3, n_max=7, grip=1, tol=0.0, chunk=7, gain=0.6)
    # 4. lift
    R.move_to(np.r_[R.pos[:2], SAFE_Z + 0.03], tol=0.02, max_iter=3, grip=1)
    ok, w = holding(R.obs.state)
    log("grasp", ok, round(w, 4), "step", R.obs.step)
    return ok

def stuck_cube(o, binc, rim_z, color):
    """A cube of `color` near the bin that sits on/over the rim (not inside)."""
    W = o.world_map("wrist")
    m = masks(o.rgb("wrist"))[color]
    for comp in components(m, 25):
        P = W[comp[:, 0], comp[:, 1]]
        P = P[np.isfinite(P).all(1)]
        if len(P) < 20:
            continue
        zt = np.percentile(P[:, 2], 90)
        top = P[P[:, 2] > zt - 0.005]
        xy = np.median(top[:, :2], 0)
        r = np.linalg.norm(xy - binc)
        if r < 0.09 and (zt > rim_z + 0.015 or (r > 0.03 and zt > rim_z - 0.01)):
            return xy, float(zt), r
    return None

def verify_drop(R, binc, rim_z, color, max_fix=2):
    cam_off = np.array([0.046, 0.020])
    for k in range(max_fix):
        R.move_to([binc[0] - cam_off[0], binc[1] - cam_off[1], rim_z + 0.20], tol=0.015, max_iter=4, grip=-1)
        R.hold(3)
        o = R.observe()
        s = stuck_cube(o, binc, rim_z, color)
        if s is None:
            log("drop verified: no", color, "cube on rim")
            return True
        xy, zt, r = s
        log("STUCK cube", color, "xy", xy.round(3).tolist(), "ztop", round(zt, 3), "r", round(r, 3), "-> regrasp")
        R.move_to([xy[0], xy[1], zt + 0.08], tol=0.006, grip=-1)
        R.move_to([xy[0], xy[1], zt - 0.02], tol=0.004, max_iter=5)
        R.hold(7, grip=1)
        R.move_to([xy[0], xy[1], rim_z + 0.09], tol=0.01, max_iter=4)
        ok, w = holding(R.obs.state)
        R.move_to([binc[0], binc[1], rim_z + 0.09], tol=0.008, max_iter=6)
        R.move_to([binc[0], binc[1], rim_z + 0.055], tol=0.003, max_iter=5)
        R.hold(6, grip=-1)
        R.move_to([binc[0], binc[1], rim_z + 0.12], tol=0.02, max_iter=2)
        log("refix", ok, round(w, 4))
    return False

def press_button(R, bt):
    (bx, by), ztop = bt
    for k, (ox, oy) in enumerate([(0, 0), (0.008, 0), (-0.008, 0), (0, 0.008), (0, -0.008)]):
        if R.ended:
            return
        if k:
            log("press retry", k)
        press_once(R, ((bx + ox, by + oy), ztop))

def press_once(R, bt):
    (bx, by), ztop = bt
    R.yaw = 0.0
    R.move_to([bx, by, 0.12], grip=1, tol=0.005)
    R.hold(3)
    z = ztop + 0.03
    while not R.ended and z > max(ztop - 0.025, 0.005):
        R.move_to([bx, by, z], max_iter=2, speed=0.3, tol=0.003)
        log("press z", round(z, 3), R.pos.round(4).tolist(), "step", R.obs.step, "ended", R.ended, "complete", R.complete)
        z -= 0.005
    if not R.ended:
        R.hold(5)
        R.move_to([bx, by, 0.12])
