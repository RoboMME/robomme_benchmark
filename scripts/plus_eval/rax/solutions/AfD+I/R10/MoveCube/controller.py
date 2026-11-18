"""MoveCube controller. Classifies the demonstrated manner from the conditioning
video, then executes it. Implemented manner: ramp-ball (pick yellow ball, release
at top of blue ramp so it rolls into the cube)."""
import json, math, os, sys, time, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion import Robot
from percep import *

OUT = "/workspace/output"
LOGF = None

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    try:
        with open(os.path.join(OUT, "controller_log.txt"), "a") as f:
            f.write(s + "\n")
    except Exception:
        pass

def current_arrays(R):
    m = R.observe()
    seq = m.get("frame_sequence") or {}
    # current images: decode from attachments in raw reply
    import base64, io
    from PIL import Image
    arrays = {}
    for att in m.get("attachments", []):
        blk = R.last_raw[att["content_index"]]
        data = blk.get("data") if isinstance(blk, dict) else None
        if data is None:
            continue
        im = Image.open(io.BytesIO(base64.b64decode(data)))
        if att["kind"] == "depth":
            arrays[(att["camera"], "depth")] = np.array(im).astype(np.float64) * 0.001
        else:
            arrays[(att["camera"], "rgb")] = np.array(im.convert("RGB")).astype(np.int32)
    return m, arrays

def video_frames(m0):
    seq = m0.get("frame_sequence") or {}
    mp = seq.get("manifest_path")
    if not mp:
        return []
    man = json.load(open(os.path.join("/workspace", mp)))
    return man["frames"]

def classify(frames):
    """returns 'ramp' if the yellow ball moves in the conditioning video."""
    vid = [f for f in frames if f.get("phase") == "video_demo"]
    if not vid:
        return "unknown", {}
    from PIL import Image
    def ycent(f):
        p = [x for x in f["files"] if x["camera"] == "front" and x["kind"] == "rgb"][0]["path"]
        a = np.array(Image.open(os.path.join("/workspace", p)).convert("RGB")).astype(np.int32)
        bl = blobs(mask_yellow(a), min_px=6)
        return (bl[0][0], bl[0][1], bl[0][2]) if bl else None
    c0 = ycent(vid[0])
    info = {"c0": c0}
    if c0 is None:
        return "noball", info
    moved = 0; checked = 0
    idx = list(range(0, len(vid), max(1, len(vid) // 40))) + [len(vid) - 1]
    for i in idx:
        c = ycent(vid[i]); checked += 1
        if c is None or np.hypot(c[1] - c0[1], c[2] - c0[2]) > 12:
            moved += 1
    info.update(moved=moved, checked=checked)
    if moved >= 3:
        return "ramp", info
    return "other", info

def locate_ball(R, approx=None):
    m, A = current_arrays(R)
    out = {}
    for cam in ("wrist", "front"):
        P = backproject(A[(cam, "depth")], m["camera_intrinsics"][cam]["matrix"], m["camera_poses"][cam])
        bl = blobs(mask_yellow(A[(cam, "rgb")]), min_px=6)
        best = None
        for n, u, v, pix in bl:
            pts = P[pix[:, 0], pix[:, 1]]
            pts = pts[~np.isnan(pts).any(1)]
            if len(pts) < 4:
                continue
            c = np.median(pts, 0)
            if approx is not None and np.linalg.norm(c[:2] - approx[:2]) > 0.08:
                continue
            best = (n, c, pts)
            break
        if best:
            out[cam] = best
    return out

def ramp_geometry(P, rgb):
    bl = blobs(mask_blue(rgb), min_px=300)
    if not bl:
        return None
    n, u, v, pix = bl[0]
    pts = P[pix[:, 0], pix[:, 1]]
    pts = pts[~np.isnan(pts).any(1)]
    z = pts[:, 2]
    zt = np.percentile(z, 97)
    top = pts[z > zt - 0.012]
    low = pts[z < np.percentile(z, 15)]
    tc = np.median(top, 0)
    lc = np.median(low, 0)
    d = lc[:2] - tc[:2]
    d = d / (np.linalg.norm(d) + 1e-9)
    # downhill direction from a plane fit of the sloped surface (more robust than centroids)
    try:
        s = pts[pts[:, 2] > 0.012]
        G = np.c_[s[:, 0], s[:, 1], np.ones(len(s))]
        co = np.linalg.lstsq(G, s[:, 2], rcond=None)[0]
        g = -co[:2] / (np.linalg.norm(co[:2]) + 1e-9)
        if np.dot(g, d) > 0.8:
            d = g
    except Exception:
        pass
    # re-anchor top/low on the fitted axis through the surface centroid
    c = np.median(pts[:, :2], 0)
    pr = (pts[:, :2] - c) @ d
    lat = (pts[:, :2] - c) @ np.array([-d[1], d[0]])
    core = np.abs(lat - np.median(lat)) < 0.03
    smin, smax = np.percentile(pr[core], 2), np.percentile(pr[core], 98)
    cl = c + np.array([-d[1], d[0]]) * np.median(lat)
    tc = np.r_[cl + d * smin, zt]
    lc = np.r_[cl + d * smax, np.percentile(z, 5)]
    return {"top": tc, "low": lc, "down": d, "ztop": float(zt), "n": n}

# roll model calibrated on ep9 (central hit, ball centre drop h=0.079 -> cube slid 0.45 m,
# offset +0.006 deflected cube by -0.09 rad) and demo 3 (glancing hit near cube edge, top release)
ROLL_C = 6.0       # cube travel per metre of ball-centre drop for a central hit
ROLL_K = 12.0      # cube deflection (rad) per metre of lateral offset
def roll_f(b):
    return max(0.3, 1.0 - 0.62 * (abs(b) / 0.023) ** 2)

XMIN_REACH = -0.355
BASE_XY = np.array([-0.615, 0.0])

def plan_release(rg, rad, cb_, tg_):
    down = rg["down"]
    slope_len = max(np.linalg.norm(rg["low"][:2] - rg["top"][:2]), 1e-3)
    n_ = np.array([-down[1], down[0]])
    top0 = rg["top"][:2].copy()
    zs = lambda o: rg["top"][2] - (rg["top"][2] - rg["low"][2]) * (o / slope_len)
    h_need = None
    try:
        if cb_ is not None and tg_ is not None:
            v = tg_[:2] - cb_[:2]; dn = float(np.linalg.norm(v)); v = v / max(dn, 1e-6)
            th = math.atan2(down[0] * v[1] - down[1] * v[0], float(np.dot(down, v)))
            b = float(np.clip(-th / ROLL_K, -0.022, 0.022))
            lat = float(np.dot(cb_[:2] - top0, n_)) + b
            lat = float(np.clip(lat, -0.05, 0.05))
            h_need = dn / (ROLL_C * roll_f(b))
            log("aim th", round(th, 3), "b", round(b, 4), "lat", round(lat, 4), "dist", round(dn, 3), "h_need", round(h_need, 4))
            rg = dict(rg); rg["top"] = np.r_[top0 + n_ * lat, rg["top"][2]]
    except Exception:
        log(traceback.format_exc())
    from tool import reach_ok
    off_min = 0.02
    while not reach_ok(rg["top"][:2] + down * off_min) and off_min < slope_len - 0.04:
        off_min += 0.005
    off_max = max(off_min, slope_len - 0.04)
    clear = 0.012
    if h_need is None:
        off = off_min; zrel = zs(off) + rad + clear
    else:
        # ball-centre drop above table needed -> TCP release height rad + h_need
        zsurf_need = h_need - clear
        if zsurf_need >= zs(off_min):
            off = off_min
            zrel = min(rad + h_need, zs(off) + rad + 0.05)
        else:
            off = off_min
            while off < off_max and zs(off) > zsurf_need:
                off += 0.005
            zrel = max(zs(off) + rad + clear, rad + h_need)
    rel = rg["top"][:2] + down * off
    log("release plan off", round(off, 3), "rel", np.round(rel, 3).tolist(), "zrel", round(zrel, 4), "off_min", round(off_min, 3))
    return off, rel, zrel, rg

def ramp_once(R, rg):
    bl = locate_ball(R)
    if "front" not in bl and "wrist" not in bl:
        log("ball not found"); return "noball"
    c = (bl.get("front") or bl.get("wrist"))[1]
    log("ball seen", c.tolist())
    try:
        from tool import find_cube
        cb = find_cube(R)
        R.cube_now = cb
        if cb is not None and np.linalg.norm(cb[:2] - c[:2]) < 0.07:
            log("ball too close to cube; stop"); return "noball"
    except Exception:
        pass
    R.hold_yaw = -math.pi / 2
    R.grip = -1
    p = R.state()
    R.move_to([p[0], p[1], max(p[2], 0.2)], tol=0.01)
    R.move_to([c[0], c[1], 0.2], tol=0.006)
    R.orient(-math.pi / 2)
    bl2 = locate_ball(R, approx=c)
    if "wrist" in bl2:
        pts = bl2["wrist"][2]
        ztop = float(np.percentile(pts[:, 2], 95))
        cx, cy = float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
    else:
        cx, cy, ztop = c[0], c[1], c[2] + 0.008
    rad = max(0.015, min(0.03, ztop / 2.0))
    log("ball", cx, cy, ztop, "rad", rad)
    R.move_to([cx, cy, 0.10], tol=0.004)
    R.move_to([cx, cy, max(0.012, ztop - rad)], tol=0.003, speed=0.4)
    R.gripper(1, 18)
    st = R.state(); log("grip state", np.round(st, 4).tolist())
    if st[6] + st[7] < 0.01:
        log("ball grasp failed"); R.gripper(-1, 6); return "graspfail"
    zt = rg["ztop"] + rad + 0.04
    R.move_to([cx, cy, zt], tol=0.005, speed=0.4)
    st = R.state()
    if st[6] + st[7] < 0.01:
        log("ball lost on lift"); return "lost"
    down = rg["down"]
    slope_len = max(np.linalg.norm(rg["low"][:2] - rg["top"][:2]), 1e-3)
    cb_ = getattr(R, "cube_now", None)
    if cb_ is None:
        cb_ = getattr(R, "cube0", None)
    off, rel, zrel, rg = plan_release(rg, rad, cb_, getattr(R, "target0", None))
    zs = lambda o: rg["top"][2] - (rg["top"][2] - rg["low"][2]) * (o / slope_len)
    pre = rel + down * 0.08
    zpre = zs(off + 0.08) + rad + 0.04
    zpre = max(zpre, zrel + 0.02)
    fy = math.atan2(down[1], down[0]) + math.pi / 2
    fy = min([fy + k * math.pi for k in (-2, -1, 0, 1, 2)], key=lambda a: abs(a + math.pi / 2))
    R.move_to([pre[0], pre[1], max(zt, zpre)], tol=0.006, speed=0.5, max_steps=120)
    R.orient(fy)
    R.move_to([pre[0], pre[1], zpre], tol=0.004, speed=0.4)
    R.move_to([rel[0], rel[1], zrel], tol=0.003, speed=0.3)
    st = R.state()
    log("release at", np.round(st, 4).tolist(), "wanted", rel.tolist(), zrel, "off", off)
    if st[6] + st[7] < 0.01:
        log("ball lost before release")
    R.gripper(-1, 4)
    R.move_to([st[0], st[1], zrel + 0.10], tol=0.01, speed=0.6)
    for _ in range(10):
        if R.ended or R.complete:
            break
        R.step([[0, 0, 0, 0, 0, 0, -1]] * 8)
    log("after roll", R.last.get("step_id"), R.ended, R.complete)
    return "rolled"

def do_ramp(R):
    m, A = current_arrays(R)
    P = backproject(A[("front", "depth")], m["camera_intrinsics"]["front"]["matrix"], m["camera_poses"]["front"])
    rg = ramp_geometry(P, A[("front", "rgb")])
    log("ramp", {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in (rg or {}).items()})
    if rg is None:
        return False
    try:
        from tool import find_cube as _fc, find_target as _ft
        R.cube0 = _fc(R); R.target0 = _ft(R)
        log("cube0", R.cube0, "target0", R.target0)
    except Exception:
        log(traceback.format_exc())
    for attempt in range(3):
        if R.ended or R.complete or R.remaining() < 150:
            break
        r = ramp_once(R, rg)
        log("attempt", attempt, r)
        if r == "noball":
            break
    return True

def pick_tool(R, ttype):
    from tool import scene_now, detect_tools, tool_type
    S = scene_now(R)
    for cam in ("front", "wrist"):
        P, rgb = S[cam]
        ts = [t for t in detect_tools(P, rgb) if tool_type(t) == ttype]
        if ts:
            return ts[0], P, rgb, cam
    return None, None, None, None

def scene_now_front(R):
    from tool import scene_now
    S = scene_now(R)
    return S["front"]

def do_toolpush(R, ttype):
    from tool import find_cube, find_target, grasp_and_push, hook_geometry
    R.hold_yaw = -math.pi / 2
    c = find_cube(R); t = find_target(R)
    tool, P, rgb, cam = pick_tool(R, ttype)
    log("cube", c, "target", t, "tool", None if tool is None else {k: tool[k] for k in ("handle", "head", "L", "heading")})
    if c is None or t is None or tool is None:
        log("missing perception; abort")
        return False
    H = tool["handle"]; zg = max(0.012, H[2] - 0.010)
    if ttype == "ballstick":
        from tool import obstacle_points
        obs = obstacle_points(*[x for x in scene_now_front(R)])
        res = grasp_and_push(R, H[:2], zg, tool["heading"], tool["head"][:2], 0.013, c, t, log=log,
                             obstacles=obs, deltas=tuple(np.radians(np.arange(-100, 101, 10))))
    elif ttype == "hook":
        g = hook_geometry(P, rgb, tool)
        if g is None:
            log("hook geometry failed"); return False
        phi = math.atan2(g["s"][1], g["s"][0])
        contact = H[:2] + g["s"] * g["arm_al"] + g["side"] * g["n"] * (0.6 * g["la"])
        u = t[:2] - c[:2]
        res = grasp_and_push(R, H[:2], zg, phi, contact, 0.008, c, t, log=log,
                             push_heading=math.atan2(-u[1], -u[0]))
    else:
        return False
    log("push result", res)
    for _ in range(3):
        if R.ended:
            break
        R.step([[0, 0, 0, 0, 0, 0, R.grip]] * 5)
    return True

def main(ctx, force=None):
    os.makedirs(OUT, exist_ok=True)
    R = Robot(ctx, log=log)
    m0 = R.observe()
    log("instr", m0.get("instruction"), "limits", m0.get("episode_limits"))
    manner = "unknown"
    try:
        frames = video_frames(m0)
        manner, info = classify(frames)
        log("ball classifier", manner, info)
        if manner != "ramp":
            from tool import classify_tools
            tt, info2 = classify_tools(frames, log=log)
            log("tool classifier", tt, info2)
            manner = tt or "unknown"
    except Exception:
        log(traceback.format_exc())
    if force:
        manner = force
    log("manner", manner)
    try:
        if manner == "ramp":
            do_ramp(R)
        elif manner in ("ballstick", "hook"):
            do_toolpush(R, manner)
        else:
            log("no implemented manner; idle")
    except Exception:
        log(traceback.format_exc())
    log("done", R.last.get("step_id"), "ended", R.ended, "complete", R.complete)
