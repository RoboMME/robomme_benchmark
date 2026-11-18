"""MoveCube controller v1: classify manner from conditioning video, then execute it.
Modes: hook (grasp L-hook, pull cube), pick (pick-and-place cube), push (gripper push)."""
import json, math, os, sys, time, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perc import parse_reply, load_frame, world_points
from scene import masks, cloud, pts_of, cluster_main
from motion import Robot, yaw_of, wrap, log, down_R, rotvec_to_R

VERSION = "v4"
ROOT = "/workspace"
CUBE_HALF = 0.02          # measured: top face z=0.041 (ep1)
HOOK_PULL_Z_OFF = 0.007   # EE z above grasp z while pulling (video: 0.017 -> 0.024)
HOOK_GRASP_DZ = -0.010    # TCP target below teal handle top (ep1: top .0255 -> tcp .017, ok)
LIFT_Z = 0.16
BALL_R = 0.02
ROT_Z = 0.20
HOOK_REF_LAT = 0.050     # cube centre this far from shaft line along the L leg (video: 0.056; leg spans 0.011..0.084)
ROLL = math.radians(22)
PRECLOSE = -0.3          # -0.2 -> 0.020 m per finger (ep3 test)

def R_axis(u, th):
    u = np.asarray(u, float); u = u / np.linalg.norm(u)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K

def Rz3(a):
    c, s_ = math.cos(a), math.sin(a); return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])

def Rz2(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, -s], [s, c]])

# ---------------------------------------------------------------- perception
def find_cube(cl):
    q = pts_of(cl, "red", zmin=0.003, zmax=0.08)
    if len(q) < 5: return None
    q = cluster_main(q, 0.05)
    top = q[q[:, 2] > q[:, 2].max() - 0.006]
    c = np.median(top, 0)
    return np.array([c[0], c[1], CUBE_HALF]), top

def find_target(cl):
    q = pts_of(cl, "purple", zmin=-0.01, zmax=0.03)
    if len(q) < 5: return None
    q = cluster_main(q, 0.05)
    return np.median(q, 0)

def cube_yaw(top):
    """face angle of square top face in [0, 90) deg"""
    if top is None or len(top) < 8: return 0.0
    xy = top[:, :2] - top[:, :2].mean(0); best = (1e9, 0.0)
    for deg in range(0, 90, 2):
        a = math.radians(deg); R = Rz2(-a); p = xy @ R.T
        area = np.ptp(p[:, 0]) * np.ptp(p[:, 1])
        if area < best[0]: best = (area, a)
    return best[1]

def clusters(q, rad=0.05, min_n=8, max_k=6):
    out = []; q = q.copy()
    for _ in range(max_k):
        if len(q) < min_n: break
        # seed = point with most neighbours (subsampled)
        sub = q[:: max(1, len(q) // 300)]
        dm = np.linalg.norm(sub[:, None, :2] - sub[None, :, :2], axis=2)
        seed = sub[np.argmax((dm < rad).sum(1))]
        c = seed
        for _ in range(3):
            sel = np.linalg.norm(q[:, :2] - c[:2], axis=1) < rad
            if sel.sum() == 0: break
            c = np.median(q[sel], 0)
        sel = np.linalg.norm(q[:, :2] - c[:2], axis=1) < rad
        if sel.sum() >= min_n: out.append(q[sel])
        q = q[~sel]
    return out

def find_tools(cl):
    """All long tools: teal handle + grey shaft (angular histogram) + pink tip at its end."""
    pink = pts_of(cl, "pink", zmin=0.005, zmax=0.06)
    teal = pts_of(cl, "teal", zmin=0.004, zmax=0.06)
    sh = pts_of(cl, "shaft", zmin=0.004, zmax=0.045)
    hands = clusters(teal, 0.045, 6)
    tools = []
    for hq in hands:
        h = np.median(hq, 0)
        rel = sh[:, :2] - h[:2]; dist = np.linalg.norm(rel, axis=1)
        sel = (dist > 0.05) & (dist < 0.42)
        if sel.sum() < 8: continue
        ang = np.degrees(np.arctan2(rel[sel, 1], rel[sel, 0]))
        hist = np.zeros(360)
        for a in ang: hist[int(round(a)) % 360] += 1
        sm = np.array([hist[(k + np.arange(-2, 3)) % 360].sum() for k in range(360)])
        k = int(np.argmax(sm)); 
        if sm[k] < 8: continue
        a0 = math.radians(k); u = np.array([math.cos(a0), math.sin(a0)]); nrm = np.array([-u[1], u[0]])
        tt = rel @ u; nn = rel @ nrm
        line = sh[(tt > 0.03) & (tt < 0.42) & (np.abs(nn) < 0.012)]
        if len(line) < 8: continue
        X = line[:, :2]; c = X.mean(0); w, V = np.linalg.eigh((X - c).T @ (X - c)); ax = V[:, 1]
        if ax @ u < 0: ax = -ax
        tend = float(np.percentile((line[:, :2] - h[:2]) @ ax, 98))
        # tip: pink points near the shaft end
        prel = pink[:, :2] - h[:2]; pt = prel @ ax; pn = prel @ np.array([-ax[1], ax[0]])
        tsel = (pt > tend - 0.04) & (pt < tend + 0.10) & (np.abs(pn) < 0.10)
        tq = pink[tsel]
        if len(tq) < 5: continue
        endp = h[:2] + ax * tend
        cands = clusters(tq, 0.035, 4)
        if not cands: continue
        tq = min(cands, key=lambda q: np.min(np.linalg.norm(q[:, :2] - endp, axis=1)))
        if np.min(np.linalg.norm(tq[:, :2] - endp, axis=1)) > 0.04: continue
        topz = float(np.percentile(tq[:, 2], 90))
        t = np.median(tq, 0)
        nax = np.array([-ax[1], ax[0]])
        lat_s = float((t[:2] - h[:2]) @ nax); lat = abs(lat_s)
        endp3 = h[:2] + ax * tend
        legdir = nax * (1.0 if lat_s >= 0 else -1.0)
        ref = endp3 + legdir * HOOK_REF_LAT   # point on the L leg where the cube should sit (clear of the shaft)
        ext = float(np.ptp(tq[:, :2] @ np.array([-ax[1], ax[0]])))
        tools.append({"handle": h, "tip": t, "leg": t, "ref": np.array([ref[0], ref[1], t[2]]), "s_ang": math.atan2(ax[1], ax[0]), "tip_z": topz,
                      "shaft_len": tend, "lat": lat, "ext": ext,
                      "type": "ball" if (topz > 0.02 and lat < 0.013 and ext < 0.045) else "flat",
                      "n_shaft": int(len(line)), "n_tip": int(len(tq))})
    return tools

def find_scoop(cl):
    """teal handle with a large flat pink pan 0.06-0.16 m away."""
    teal = pts_of(cl, "teal", zmin=0.004, zmax=0.06)
    pink = pts_of(cl, "pink", zmin=0.0, zmax=0.03)
    hands = clusters(teal, 0.045, 6); pans = [q for q in clusters(pink, 0.06, 40)
            if np.ptp(q[:, 0]) > 0.04 and np.ptp(q[:, 1]) > 0.04 and np.median(q[:, 2]) < 0.012]
    best = None
    for hq in hands:
        h = np.median(hq, 0)
        for pq in pans:
            p = np.median(pq, 0); L = np.linalg.norm(p[:2] - h[:2])
            if 0.06 < L < 0.16 and (best is None or len(pq) > best[2]):
                best = (h, p, len(pq))
    if best is None: return None
    h, p, n = best; v = p[:2] - h[:2]
    return {"handle": h, "pan": p, "s_ang": math.atan2(v[1], v[0]), "L": float(np.linalg.norm(v)), "n_pan": n, "type": "scoop"}

def find_hook(cl, want="flat"):
    ts = [t for t in find_tools(cl) if t["type"] == want]
    return ts[0] if ts else None

# ---------------------------------------------------------------- video
def classify_video(meta0):
    """Return (mode, info). mode in hook/peg/pick/push. Uses video frame states (gripper) + scene in first frame."""
    info = {}
    try:
        fs = meta0.get("frame_sequence") or {}
        if fs.get("manifest_path"):
            mp = Path(ROOT) / fs["manifest_path"]
        else:   # not at step 0: newest run's initial sequence
            c = sorted(Path(ROOT, "runtime/autonomous_controller/runs").glob("*/frames/sequence_000000/manifest.json"),
                       key=lambda p: p.stat().st_mtime)
            mp = c[-1]; info["manifest"] = str(mp)
        man = json.loads(mp.read_text())
        frames = [f for f in man["frames"] if f.get("phase") == "video_demo"]
        if not frames:
            alt = mp.parent.parent / "sequence_000000" / "manifest.json"
            if alt.exists():
                man = json.loads(alt.read_text()); info["manifest"] = str(alt)
                frames = [f for f in man["frames"] if f.get("phase") == "video_demo"]
        frames.sort(key=lambda f: f["frame_index"])
        info["n_video"] = len(frames)
        if not frames: return "pick", info
        f0 = frames[0]; im = load_frame(ROOT, f0)
        cl = {}
        for cam in ("front", "wrist"):
            P = world_points(im[(cam, "depth")], f0["camera_intrinsics"][cam], f0["camera_poses"][cam])
            cl[cam] = (im[(cam, "rgb")], P)
        cb = find_cube(cl)
        tools = find_tools(cl)
        sc = find_scoop(cl)
        if sc is not None:
            tools = [t for t in tools if np.linalg.norm(t["handle"][:2] - sc["handle"][:2]) > 0.04] + [sc]
        vball = find_ball(cl)
        info["video_ball"] = None if vball is None else vball.round(3).tolist()
        info["video_cube"] = None if cb is None else cb[0][:2].tolist()
        info["video_tools"] = [(t["type"], t["handle"][:2].round(3).tolist()) for t in tools]
        st = np.array([f["state"] for f in frames])
        fing = st[:, 6] + st[:, 7]
        closed = np.nonzero(fing < 0.07)[0]
        info["min_fing"] = float(fing.min())
        if len(closed) == 0:
            info["why"] = "never closed"; return "push_open", info
        k = closed[0]
        k2 = min(len(st) - 1, k + 15)
        fclosed = float(fing[k2]); pclose = st[k2, :3]
        info.update(close_frame=int(k), close_pos=pclose.tolist(), fing_after=fclosed)
        dcube = None if cb is None else float(np.linalg.norm(pclose[:2] - cb[0][:2]))
        dt = [float(np.linalg.norm(pclose[:2] - t["handle"][:2])) for t in tools]
        info.update(dcube=dcube, dtools=dt)
        if fclosed < 0.008:
            info["why"] = "closed empty"; return "push", info
        jt = int(np.argmin(dt)) if dt else -1
        dball = None if vball is None else float(np.linalg.norm(pclose[:2] - vball[:2]))
        k0 = int(closed[0]); pc0 = st[k0, :3]
        dball0 = None if vball is None else float(np.linalg.norm(pc0[:2] - vball[:2]))
        info.update(dball=dball, dball0=dball0)
        if dball0 is not None and dball0 < 0.04 and (dcube is None or dball0 < dcube - 0.01) \
                and (jt < 0 or dball0 < dt[jt]):
            return "ramp", info
        # backup: gripper re-opened low over the blue ramp (tools are never released in the videos)
        reop = np.nonzero(fing[k0:] > 0.07)[0]
        if len(reop):
            pr = st[k0 + reop[0], :3]; info["reopen_pos"] = pr.round(3).tolist()
            try:
                rp = find_ramp(cl)
                if rp is not None and pr[2] < 0.12:
                    dmin = float(np.min(np.linalg.norm(rp["pts"][:, :2] - pr[:2], axis=1)))
                    info["reopen_ramp_d"] = dmin
                    if dmin < 0.03 and (dcube is None or dcube > 0.05):
                        return "ramp", info
            except Exception:
                pass
        if dcube is not None and dcube < 0.05 and (jt < 0 or dcube < dt[jt]):
            return "pick", info
        if jt >= 0 and dt[jt] < 0.08:
            return {"ball": "peg", "scoop": "scoop"}.get(tools[jt]["type"], "hook"), info
        if dcube is not None and dcube < 0.08: return "pick", info
        if pclose[2] > 0.06:
            info["why"] = "closed high"; return "push", info
        return "hook", info
    except Exception as e:
        info["err"] = repr(e); log(traceback.format_exc())
        return "hook", info

# ---------------------------------------------------------------- skills
def rotate_yaw(r, target_yaw, xyz, direction, step_deg=25.0):
    """Rotate to target_yaw going in the given direction (+1/-1), via waypoints."""
    cur = yaw_of(r.state)
    total = wrap(target_yaw - cur)
    if direction > 0 and total < 0: total += 2 * math.pi
    if direction < 0 and total > 0: total -= 2 * math.pi
    n = max(1, int(math.ceil(abs(total) / math.radians(step_deg))))
    for i in range(1, n + 1):
        y = cur + total * i / n
        r.move_to(xyz, yaw=wrap(y), tol=0.01, yaw_tol=0.05, max_iter=3)
        if r.ended: return False
    got = yaw_of(r.state)
    log("rotate_yaw target", round(math.degrees(target_yaw), 1), "got", round(math.degrees(got), 1))
    return abs(wrap(got - target_yaw)) < math.radians(8)

def refine_handle(r, h_guess, rad=0.06):
    """re-detect the teal handle near h_guess in the current wrist (and front) view."""
    r.observe()
    cl = cloud(r.meta, r.imgs)
    q = pts_of(cl, "teal", cams=("wrist",), zmin=0.004, zmax=0.06)
    q = q[np.linalg.norm(q[:, :2] - h_guess[:2], axis=1) < rad] if len(q) else q
    if len(q) < 15:
        return h_guess
    hq = cluster_main(q, 0.04)
    h = np.median(hq, 0)
    top = float(np.percentile(hq[:, 2], 90))
    return np.array([h[0], h[1], top])

def grasp_shift(cl, h, s_ang):
    """choose a grasp point along the handle axis (shift t from the handle centre) and a finger pre-opening so the
    fingers do not land on neighbouring objects (ep3: a ball beside the handle blocked one finger)."""
    u = np.array([math.cos(s_ang), math.sin(s_ang)]); n = np.array([-u[1], u[0]])
    rgb, P = cl["front"]
    q = P.reshape(-1, 3); q = q[~np.isnan(q).any(1)]
    q = q[(q[:, 2] > h[2] + 0.006) & (q[:, 2] < 0.25)]
    rel = q[:, :2] - h[:2]; t = rel @ u; nn = rel @ n
    near = (np.abs(t) < 0.08) & (np.abs(nn) < 0.08)
    t, nn, z = t[near], nn[near], q[near, 2]
    tq = pts_of(cl, "teal", ("front",), 0.004, 0.06)
    tq = tq[np.linalg.norm(tq[:, :2] - h[:2], axis=1) < 0.06] if len(tq) else tq
    if len(tq) >= 10:
        tt = (tq[:, :2] - h[:2]) @ u
        tlo, thi = np.percentile(tt, 5) + 0.012, np.percentile(tt, 95) - 0.012
    else:
        tlo, thi = -0.01, 0.01
    if tlo > thi: tlo = thi = 0.5 * (tlo + thi)
    best = None
    for ts in np.linspace(tlo, thi, 7):
        for half, pc in ((0.040, -1.0), (0.022, -0.2)):
            fz = (np.abs(t - ts) < 0.013) & (np.abs(nn) > 0.010) & (np.abs(nn) < half + 0.012)
            cost = int(fz.sum()) * 10 + abs(ts) * 100 + (1 if pc > -1 else 0)
            if best is None or cost < best[0]: best = (cost, float(ts), pc, int(fz.sum()))
    log("grasp shift t %.3f preclose %.1f finger-zone pts %d (range %.3f..%.3f)" % (best[1], best[2], best[3], tlo, thi))
    return best[1], best[2]

def side_heights(cl, h, s_ang):
    """max obstacle height either side of the handle in the hand footprint (|n| 0.04-0.12 along finger axis)."""
    u = np.array([math.cos(s_ang), math.sin(s_ang)]); n = np.array([-u[1], u[0]])
    out = []
    for sign in (+1, -1):
        hmax = 0.0
        for cam in cl:
            rgb, P = cl[cam]; q = P.reshape(-1, 3); q = q[~np.isnan(q).any(1)]
            rel = q[:, :2] - h[:2]; t = rel @ u; nn = (rel @ n) * sign
            sel = (np.abs(t) < 0.05) & (nn > 0.045) & (nn < 0.12) & (q[:, 2] < 0.3)
            if sel.sum(): hmax = max(hmax, float(np.percentile(q[sel, 2], 99)))
        out.append(hmax)
    return out   # [+n side, -n side]

def grasp_R(g, s_ang, roll):
    """down gripper at yaw g, rolled about tool axis s by roll (+ lifts the +n side)."""
    u = np.array([math.cos(s_ang), math.sin(s_ang), 0.0])
    return R_axis(u, roll) @ down_R(g)

def grasp_handle(r, h, g, s_ang, cl=None, tries=3):
    """hover, refine with wrist camera, descend, close; verify finger gap; retry with roll if blocked."""
    roll = 0.0; tshift = 0.0; pre = PRECLOSE
    u2 = np.array([math.cos(s_ang), math.sin(s_ang)])
    if cl is not None:
        try:
            tshift, pre0 = grasp_shift(cl, h, s_ang)
            pre = min(PRECLOSE, pre0) if pre0 <= -0.99 else pre0
        except Exception:
            log("grasp_shift failed", traceback.format_exc())
        hs = h.copy(); hs[:2] += u2 * tshift
        hp, hn = side_heights(cl, hs, s_ang)
        log("side heights +n %.3f -n %.3f" % (hp, hn))
        if max(hp, hn) > 0.05: roll = ROLL if hp > hn else -ROLL
    for k in range(tries):
        Rg = grasp_R(g, s_ang, roll)
        r.grip = -1.0
        r.move_to([h[0], h[1], LIFT_Z], R_des=Rg, tol=0.006)
        r.move_to([h[0], h[1], 0.12], R_des=Rg, tol=0.004, vmax=0.5)
        h2 = refine_handle(r, h)
        log("handle refine", np.round(h, 3), "->", np.round(h2, 3), "roll", round(math.degrees(roll)))
        if np.linalg.norm(h2[:2] - h[:2]) < 0.05: h = h2
        hg = h.copy(); hg[:2] += u2 * tshift
        r.move_to([hg[0], hg[1], h[2] + 0.03], R_des=Rg, tol=0.003, vmax=0.4)
        r.gripper(pre, 8)
        tz = h[2] + HOOK_GRASP_DZ
        r.move_to([hg[0], hg[1], tz], R_des=Rg, tol=0.003, vmax=0.25)
        blocked = r.state[2] > tz + 0.008
        r.gripper(1.0, 15)
        fing = r.state[6] + r.state[7]
        log("grasp try", k, "fingers", round(fing, 4), "ee", np.round(r.state[:3], 3), "blocked", blocked)
        if fing > 0.008 or r.ended:
            return fing > 0.008
        r.gripper(-1.0, 8)
        p = r.state[:3]; r.move_to([p[0], p[1], 0.12], R_des=Rg, tol=0.006, vmax=0.4)
        if blocked:
            pre = -0.2
            if roll == 0:
                hp0, hn0 = side_heights(cl, h, s_ang) if cl is not None else (0.0, 0.0)
                roll = ROLL if (k == 0) == (hp0 >= hn0) else -ROLL
            else:
                roll = math.copysign(min(abs(roll) * 1.6, math.radians(36)), roll)   # roll further AWAY from the obstacle
    return False

def rotate_R(r, R0, total, xyz, step_deg=25.0):
    """rotate from R0 about world z by total (signed, may exceed pi) through waypoints."""
    n = max(1, int(math.ceil(abs(total) / math.radians(step_deg))))
    for i in range(1, n + 1):
        r.move_to(xyz, R_des=Rz3(total * i / n) @ R0, tol=0.01, yaw_tol=0.05, max_iter=3)
        if r.ended: return False
    return True

def in_ws(p):
    return p[0] < 0.19 and abs(p[1]) < 0.45 and np.linalg.norm(np.asarray(p[:2]) - np.array([-0.615, 0.0])) > 0.28

def peg_shaft_angle(tool, cube, d, standoff, D):
    """shaft angle for a sideways ball push: angle(d) - 90 (video) unless that leaves the workspace, then + 90."""
    L = float(np.linalg.norm(tool["tip"][:2] - tool["handle"][:2]))
    base = math.atan2(d[1], d[0])
    for sgn in (-1.0, 1.0):
        a = base + sgn * math.pi / 2
        sv = np.array([math.cos(a), math.sin(a)])
        tip0 = cube[:2] - d * standoff
        ee0 = tip0 - sv * L; ee1 = ee0 + d * D
        if in_ws(ee0) and in_ws(ee1):
            return a
    return base - math.pi / 2

def do_tool(r, cube, tgt, tool, mode, cl=None):
    """mode 'hook': pull with flat leg (handle leads toward target); 'peg': push with ball (tip leads)."""
    h = tool["handle"]; s_ang = tool["s_ang"]
    d = tgt[:2] - cube[:2]; dist = np.linalg.norm(d); d = d / dist
    if mode == "hook":
        want_s = math.atan2(-d[1], -d[0]); standoff = CUBE_HALF + 0.006 + 0.02; extra = 0.02 + 0.004
    else:
        # video: stick held PERPENDICULAR to the push direction (shaft at angle(d) - 90 deg), ball swept sideways into cube
        standoff = CUBE_HALF + BALL_R + 0.008; extra = 0.012
        want_s = peg_shaft_angle(tool, cube, d, standoff, dist + extra)
    delta = wrap(want_s - s_ang)
    opts = [wrap(s_ang), wrap(s_ang + math.pi)]
    g = min(opts, key=lambda g: max(abs(g), abs(wrap(g + delta))))
    log(mode, "plan: s_ang", round(math.degrees(s_ang)), "grasp yaw", round(math.degrees(g)),
        "delta", round(math.degrees(delta)), "dist", round(dist, 3))
    if not grasp_handle(r, h, g, s_ang, cl):
        log("tool grasp missed"); return False
    ee_g = r.state[:2].copy(); yaw_g = yaw_of(r.state); gz = r.state[2]
    R0 = rotvec_to_R(r.state[3:6])
    refp = tool["ref"] if mode == "hook" else tool["tip"]
    local = Rz2(-yaw_g) @ (refp[:2] - ee_g)
    r.move_to([ee_g[0], ee_g[1], LIFT_Z], R_des=R0, tol=0.01, vmax=0.4)
    tipT = cube[:2] - d * standoff
    ee_T = tipT - Rz2(yaw_g + delta) @ local
    if abs(delta) > math.radians(5):
        r.move_to([ee_g[0], ee_g[1], ROT_Z], R_des=R0, tol=0.01, vmax=0.4)
        mid = (ee_g + ee_T) / 2
        rotate_R(r, R0, delta, [mid[0], mid[1], ROT_Z])
    got = wrap(yaw_of(r.state) - yaw_g)
    if abs(wrap(got - delta)) > math.radians(10): log("rotation imprecise", round(math.degrees(got)), round(math.degrees(delta)))
    Rn = Rz3(delta) @ R0
    ee_T = tipT - Rz2(yaw_g + delta) @ local
    r.move_to([ee_T[0], ee_T[1], max(LIFT_Z, r.state[2] - 0.05)], R_des=Rn, tol=0.004)
    r.move_to([ee_T[0], ee_T[1], gz + 0.04], R_des=Rn, tol=0.004, vmax=0.5)
    r.move_to([ee_T[0], ee_T[1], gz + HOOK_PULL_Z_OFF], R_des=Rn, tol=0.003, vmax=0.25)
    log("tool placed ee", np.round(r.state[:3], 3), "yaw", round(math.degrees(yaw_of(r.state)), 1), "step", r.meta.get("step_id"))
    if r.ended: return False
    D = dist + extra
    start = r.state[:3].copy()
    goal = start.copy(); goal[:2] += d * D
    n_seg = 4
    for i in range(1, n_seg + 1):
        wp = start + (goal - start) * i / n_seg
        r.move_to(wp, R_des=Rn, tol=0.003, vmax=0.25, max_iter=4)
        if r.ended: return r.complete
    log("moved ee", np.round(r.state[:3], 3), "step", r.meta.get("step_id"))
    p = r.state[:3].copy()
    if mode == "peg":   # back off a little before lifting so the ball does not drag the cube
        p[:2] -= d * 0.015; r.move_to(p, R_des=Rn, tol=0.004, vmax=0.2, max_iter=3)
    r.move_to([p[0], p[1], gz + 0.08], R_des=Rn, tol=0.01, vmax=0.3)
    return True

def tilted_R(yaw_ee, s_ang, tilt):
    """down gripper at yaw_ee, then tip the tool direction s_ang downward by tilt (rad)."""
    u = np.array([-math.sin(s_ang), math.cos(s_ang), 0.0])
    return R_axis(u, tilt) @ down_R(yaw_ee)

SCOOP_START_BACK = 0.084   # pan centre behind cube centre at sweep start (video ep3)
SCOOP_SWEEP = 0.29         # sweep length (video ep3)
SCOOP_DZ = 0.021           # EE above grasp height while sweeping (video: 0.023 -> 0.044)
SCOOP_TILT = math.radians(7.3)
DUMP_BACK = 0.087          # pan centre behind target when dumping (video ep3)
DUMP_Z = 0.096
DUMP_TILT = math.radians(16.5)

def tall_points(cl, cube, zmin=0.05, zmax=0.25):
    """world points of tall obstacles (front camera), excluding the cube itself."""
    rgb, P = cl["front"]
    q = P.reshape(-1, 3)
    m = (q[:, 2] > zmin) & (q[:, 2] < zmax) & (q[:, 0] > -0.75) & (q[:, 0] < 0.25) & (np.abs(q[:, 1]) < 0.6)
    q = q[m]
    q = q[np.linalg.norm(q[:, :2] - cube[:2], axis=1) > 0.045]
    q = q[np.linalg.norm(q[:, :2] - np.array([-0.615, 0.0]), axis=1) > 0.2]   # robot base / arm
    return q[::3]

def sweep_obstacles(tall, cube, sc, a):
    u = np.array([math.cos(a), math.sin(a)]); L = sc["L"] + 0.02
    if len(tall) == 0: return 0
    rel = tall[:, :2] - cube[:2]
    t = rel @ u; n = rel @ np.array([-u[1], u[0]])
    m = (t > -(SCOOP_START_BACK + L)) & (t < SCOOP_SWEEP - SCOOP_START_BACK + 0.04) & (np.abs(n) < 0.09)
    return int(m.sum())

def do_scoop(r, cube, tgt, sc, cl=None):
    h = sc["handle"]; s_ang = sc["s_ang"]
    cands = [s_ang + math.radians(k) for k in (0, 30, -30, 60, -60, 90, -90, 120, -120, 180)]
    def ok_path(a):
        u = np.array([math.cos(a), math.sin(a)]); L = sc["L"] + 0.01
        pts = [cube[:2] - u * (SCOOP_START_BACK + L), cube[:2] + u * (SCOOP_SWEEP - SCOOP_START_BACK - L),
               tgt[:2] - u * (DUMP_BACK + L)]
        return all(-0.62 < p[0] < 0.06 and abs(p[1]) < 0.42 for p in pts)
    tall = tall_points(cl, cube) if cl is not None else np.zeros((0, 3))
    obst = lambda a: sweep_obstacles(tall, cube, sc, a)
    scored = [(obst(a), i, a) for i, a in enumerate(cands) if ok_path(a)]
    log("scoop dir costs", [(round(math.degrees(a)), c) for c, i, a in scored])
    sweep = min(scored)[2] if scored else s_ang
    delta = wrap(sweep - s_ang)
    opts = [wrap(s_ang), wrap(s_ang + math.pi)]
    g = min(opts, key=lambda g: max(abs(g), abs(wrap(g + delta))))
    log("scoop plan s_ang", round(math.degrees(s_ang)), "sweep", round(math.degrees(sweep)), "grasp yaw", round(math.degrees(g)))
    if not grasp_handle(r, h, g, s_ang, cl):
        log("scoop grasp missed"); return False
    ee_g = r.state[:2].copy(); yaw_g = yaw_of(r.state); gz = r.state[2]
    R0 = rotvec_to_R(r.state[3:6])
    local = Rz2(-yaw_g) @ (sc["pan"][:2] - ee_g)
    r.move_to([ee_g[0], ee_g[1], ROT_Z], R_des=R0, tol=0.01, vmax=0.4)
    u = np.array([math.cos(sweep), math.sin(sweep)]); u3 = np.array([-u[1], u[0], 0.0])
    panS = cube[:2] - u * SCOOP_START_BACK
    eeS = panS - Rz2(yaw_g + delta) @ local
    if abs(delta) > math.radians(5):
        rotate_R(r, R0, delta, [(ee_g[0] + eeS[0]) / 2, (ee_g[1] + eeS[1]) / 2, ROT_Z])
    Rn = Rz3(delta) @ R0
    r.move_to([eeS[0], eeS[1], ROT_Z], R_des=Rn, tol=0.005)
    Rt = R_axis(u3, SCOOP_TILT) @ Rn
    r.move_to([eeS[0], eeS[1], gz + 0.06], R_des=Rt, tol=0.004, vmax=0.5)
    r.move_to([eeS[0], eeS[1], gz + SCOOP_DZ], R_des=Rt, tol=0.003, vmax=0.25)
    log("scoop start ee", np.round(r.state[:3], 3), "step", r.meta.get("step_id"))
    if r.ended: return False
    st0 = r.state[:3].copy(); goal = st0.copy(); goal[:2] += u * SCOOP_SWEEP
    for i in range(1, 6):
        r.move_to(st0 + (goal - st0) * i / 5, R_des=Rt, tol=0.003, vmax=0.25, max_iter=4)
        if r.ended: return r.complete
    log("scoop swept ee", np.round(r.state[:3], 3), "step", r.meta.get("step_id"))
    p = r.state[:3].copy()
    r.move_to([p[0], p[1], p[2] + 0.02], R_des=Rn, tol=0.004, vmax=0.15)   # un-tilt, small lift
    r.move_to([p[0], p[1], 0.24], R_des=Rn, tol=0.006, vmax=0.25)
    panD = tgt[:2] - u * DUMP_BACK
    eeD = panD - Rz2(yaw_g + delta) @ local
    r.move_to([eeD[0], eeD[1], 0.24], R_des=Rn, tol=0.005, vmax=0.3)
    r.move_to([eeD[0], eeD[1], DUMP_Z], R_des=Rn, tol=0.004, vmax=0.25)
    log("dump pose ee", np.round(r.state[:3], 3), "step", r.meta.get("step_id"))
    if r.ended: return r.complete
    for k in range(1, 5):
        r.move_to([eeD[0], eeD[1], DUMP_Z], R_des=R_axis(u3, DUMP_TILT * k / 4) @ Rn, tol=0.004, vmax=0.2, max_iter=3)
        if r.ended: return r.complete
    r.hold(15)
    return True

BALL_RADIUS = 0.0195
RAMP_VIDEO_DROP_PER_M = 0.038 / 0.173   # video: ball centre released 3.8cm above rest height, cube->target 0.173 m
RAMP_VIDEO_COS2 = 0.64                  # video ramp slope 0.75 (37 deg)

def find_ball(cl):
    """yellow ball centre (x, y, z) or None."""
    q = pts_of(cl, "yellow", ("front",), 0.004, 0.10)
    if len(q) < 8: return None
    best = None
    for c in clusters(q, rad=0.035, min_n=8, max_k=6):
        ext = np.ptp(c[:, :2], 0).max()
        top = np.percentile(c[:, 2], 95)
        if ext > 0.07 or top < 0.025 or top > 0.06: continue
        if best is None or len(c) > len(best): best = c
    if best is None: return None
    ctr = np.median(best, 0); ctr[2] = np.percentile(best[:, 2], 95) - BALL_RADIUS
    # top-view centre: points are on the visible hemisphere; shift slightly away from front camera (+x side seen)
    ctr[0] -= 0.004
    return ctr

def find_ramp(cl):
    """plane z = a x + b y + c of the blue ramp top; returns dict or None."""
    q = pts_of(cl, "blue", ("front",), 0.004, 0.25)
    q = q[np.linalg.norm(q[:, :2] - np.array([-0.615, 0.0]), axis=1) > 0.12]
    if len(q) < 100: return None
    q = cluster_main(q, rad=0.25)
    keep = q
    for _ in range(3):
        A = np.c_[keep[:, 0], keep[:, 1], np.ones(len(keep))]
        coef = np.linalg.lstsq(A, keep[:, 2], rcond=None)[0]
        res = np.abs(q[:, 2] - np.c_[q[:, 0], q[:, 1], np.ones(len(q))] @ coef)
        keep = q[res < 0.004]
        if len(keep) < 50: return None
    g = coef[:2]; sl = float(np.linalg.norm(g))
    if sl < 0.08: return None
    u = -g / sl
    return {"coef": coef, "u": u, "slope": sl, "pts": keep}

def ramp_release(ramp, cube, tgt):
    """release xy and TCP z for the ball, on the ramp line through the cube."""
    a, b, c = ramp["coef"]; u = ramp["u"]; sl = ramp["slope"]
    v = np.array([-u[1], u[0]])
    dist = float(np.linalg.norm(tgt[:2] - cube[:2]))
    cos2 = 1.0 / (1.0 + sl * sl)
    drop = RAMP_VIDEO_DROP_PER_M * dist * math.sqrt(RAMP_VIDEO_COS2 / cos2)
    drop = float(np.clip(drop, 0.02, 0.09))
    zc = BALL_RADIUS + drop                       # ball centre height at release
    zs = zc - BALL_RADIUS * math.sqrt(1 + sl * sl)   # ramp surface height under ball centre
    plane = lambda p: a * p[0] + b * p[1] + c
    # lateral: same lateral coordinate as cube, clipped into ramp footprint
    P = ramp["pts"][:, :2]
    lat = (P - cube[:2]) @ v
    lo, hi = np.percentile(lat, 3) + 0.03, np.percentile(lat, 97) - 0.03
    off = float(np.clip(0.0, lo, hi)) if lo < hi else float(np.median(lat))
    base = cube[:2] + v * off
    t = (zs - plane(base)) / sl                   # distance uphill from base along -u
    p = base - u * t
    # stay inside ramp along its length
    along = (P - base) @ (-u)
    tmax = np.percentile(along, 97) - 0.03
    if t > tmax:
        p = base - u * tmax; zs = plane(p); zc = zs + BALL_RADIUS * math.sqrt(1 + sl * sl)
    return p, zc, {"dist": dist, "drop": drop, "slope": sl, "off": off, "t": float(t), "zs": float(zs)}

def do_ramp(r, cube, tgt, cl):
    ramp = find_ramp(cl); ball = find_ball(cl)
    if ramp is None or ball is None:
        log("ramp/ball not found", ramp is None, ball is None); return False
    p, zc, inf = ramp_release(ramp, cube, tgt)
    u = ramp["u"]
    yaw = wrap(math.atan2(u[1], u[0]))
    while yaw > math.pi / 2: yaw -= math.pi
    while yaw < -math.pi / 2: yaw += math.pi
    log("ramp plan ball", np.round(ball, 3), "u", np.round(u, 3), "release", np.round(p, 3), round(zc, 3), json.dumps(inf))
    ok = False
    for k in range(2):
        r.grip = -1.0
        r.move_to([ball[0], ball[1], LIFT_Z], yaw=yaw, tol=0.006)
        r.move_to([ball[0], ball[1], 0.08], yaw=yaw, tol=0.004, vmax=0.5)
        r.move_to([ball[0], ball[1], ball[2] + 0.001], yaw=yaw, tol=0.003, vmax=0.25)
        r.gripper(1.0, 15)
        fing = r.state[6] + r.state[7]
        log("ball grasp try", k, "fingers", round(fing, 4), "ee", np.round(r.state[:3], 3))
        if r.ended: return r.complete
        if fing > 0.025: ok = True; break
        r.gripper(-1.0, 8)
        r.move_to([ball[0], ball[1], LIFT_Z], yaw=yaw, tol=0.01)
        r.observe(); cl2 = cloud(r.meta, r.imgs); b2 = find_ball(cl2)
        if b2 is not None: ball = b2
    if not ok: log("ball grasp failed"); return False
    r.move_to([ball[0], ball[1], 0.22], yaw=yaw, tol=0.01, vmax=0.5)
    r.move_to([p[0], p[1], 0.22], yaw=yaw, tol=0.005, vmax=0.6)
    r.move_to([p[0], p[1], zc + 0.04], yaw=yaw, tol=0.004, vmax=0.4)
    r.move_to([p[0], p[1], zc + 0.002], yaw=yaw, tol=0.003, vmax=0.15)
    r.hold(8)
    log("release pose", np.round(r.state[:3], 3), "fing", round(r.state[6] + r.state[7], 4), "step", r.meta.get("step_id"))
    if r.ended: return r.complete
    r.gripper(-1.0, 4)
    q = r.state[:3].copy()
    r.move_to([q[0], q[1], q[2] + 0.12], yaw=yaw, tol=0.01, vmax=0.6, max_iter=3)
    r.hold(60)
    return True

def do_pick(r, cube, tgt, top):
    a = cube_yaw(top)            # face angle
    g = wrap(a - math.pi / 2)
    while g > math.pi / 4: g -= math.pi / 2
    while g < -math.pi / 4: g += math.pi / 2
    r.grip = -1.0
    r.move_to([cube[0], cube[1], LIFT_Z], yaw=g, tol=0.006)
    r.move_to([cube[0], cube[1], 0.07], yaw=g, tol=0.004, vmax=0.5)
    r.move_to([cube[0], cube[1], CUBE_HALF], yaw=g, tol=0.003, vmax=0.3)
    r.gripper(1.0, 15)
    fing = r.state[6] + r.state[7]; log("pick fingers", round(fing, 4))
    if fing < 0.01: log("pick missed"); return False
    r.move_to([cube[0], cube[1], 0.10], yaw=g, tol=0.006, vmax=0.4)
    r.move_to([tgt[0], tgt[1], 0.10], yaw=g, tol=0.004, vmax=0.5)
    r.move_to([tgt[0], tgt[1], CUBE_HALF + 0.008], yaw=g, tol=0.003, vmax=0.3)
    r.gripper(-1.0, 10)
    r.move_to([tgt[0], tgt[1], 0.12], yaw=g, tol=0.01, vmax=0.4)
    return True

def do_push(r, cube, tgt, closed=True):
    d = tgt[:2] - cube[:2]; dist = np.linalg.norm(d); d = d / dist
    yaw = wrap(math.atan2(d[1], d[0]))      # finger axis (yaw+90) perpendicular to d
    while yaw > math.pi / 2: yaw -= math.pi
    while yaw < -math.pi / 2: yaw += math.pi
    r.grip = 1.0 if closed else -1.0
    start = cube[:2] - d * (CUBE_HALF + 0.035)
    r.move_to([start[0], start[1], LIFT_Z], yaw=yaw, tol=0.006)
    r.hold(8)
    r.move_to([start[0], start[1], 0.06], yaw=yaw, tol=0.004, vmax=0.5)
    r.move_to([start[0], start[1], 0.018], yaw=yaw, tol=0.003, vmax=0.3)
    s0 = r.state[:3].copy(); goal = s0.copy(); goal[:2] += d * (dist + 0.035 - 0.012)
    for i in range(1, 5):
        r.move_to(s0 + (goal - s0) * i / 4, yaw=yaw, tol=0.003, vmax=0.2, max_iter=4)
        if r.ended: return r.complete
    p = r.state[:3].copy(); back = p.copy(); back[:2] -= d * 0.03
    r.move_to(back, yaw=yaw, tol=0.005, vmax=0.2)
    r.move_to([back[0], back[1], 0.12], yaw=yaw, tol=0.01, vmax=0.4)
    return True

def perceive(r):
    cl = cloud(r.meta, r.imgs)
    cb = find_cube(cl); tg = find_target(cl); hk = find_hook(cl)
    return cb, tg, hk

def main(ctx):
    t0 = time.time()
    os.makedirs("/workspace/output", exist_ok=True)
    r = Robot(ctx)
    try:
        r.observe()
        log("controller", VERSION, "step", r.meta.get("step_id"), "instr", r.meta.get("instruction"))
        mode, info = classify_video(r.meta)
        log("mode", mode, json.dumps(info, default=str))
        cl = cloud(r.meta, r.imgs)
        cb = find_cube(cl); tg = find_target(cl); tools = find_tools(cl)
        log("cube", None if cb is None else np.round(cb[0], 3), "target", None if tg is None else np.round(tg, 3))
        for t in tools: log(" tool", t["type"], np.round(t["handle"], 3), np.round(t["tip"], 3), round(math.degrees(t["s_ang"]), 1))
        if cb is None or tg is None:
            log("perception failed"); return
        cube, top = cb
        sc = find_scoop(cl)
        if sc is not None:
            log(" scoop", np.round(sc["handle"], 3), np.round(sc["pan"], 3), round(math.degrees(sc["s_ang"]), 1))
            tools = [t for t in tools if np.linalg.norm(t["handle"][:2] - sc["handle"][:2]) > 0.04]
        if mode == "scoop" and sc is None:
            log("no scoop found"); mode = "hook"
        if mode in ("hook", "peg"):
            want = "flat" if mode == "hook" else "ball"
            cand = [t for t in tools if t["type"] == want] or tools
            if not cand:
                log("no tool found -> push fallback"); mode = "push"
            else:
                tool = min(cand, key=lambda t: np.linalg.norm(t["handle"][:2] - cube[:2]))
        if mode == "ramp": ok = do_ramp(r, cube, tg, cl)
        elif mode == "scoop": ok = do_scoop(r, cube, tg, sc, cl)
        elif mode in ("hook", "peg"): ok = do_tool(r, cube, tg, tool, mode, cl)
        elif mode == "pick": ok = do_pick(r, cube, tg, top)
        else: ok = do_push(r, cube, tg, closed=(mode == "push"))
        log("skill done ok", ok, "ended", r.ended, "complete", r.complete, "step", r.meta.get("step_id"))
        if not r.ended:
            r.observe(); cb2 = find_cube(cloud(r.meta, r.imgs))
            if cb2 is not None:
                log("final cube", np.round(cb2[0], 3), "err to target", round(float(np.linalg.norm(cb2[0][:2] - tg[:2])), 3))
            r.hold(5)
    except Exception:
        log("EXC", traceback.format_exc())
    finally:
        log("elapsed", round(time.time() - t0, 1), "steps used", r.steps_used)
        try:
            Path("/workspace/output/controller_log.txt").write_text("\n".join(__import__("motion").LOG))
        except Exception:
            pass
