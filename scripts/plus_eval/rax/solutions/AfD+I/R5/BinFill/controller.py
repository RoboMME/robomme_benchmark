"""BinFill controller: put the instructed number of coloured cubes into the bin, then press the button.

Observation-driven loop: find bin/button once from the front RGB-D view, then repeatedly
detect a target-colour cube on the table, grasp it (wrist refinement + yaw alignment),
verify the grasp by finger opening, drop it over the bin and count it. Finally press button.
"""
import os, re, sys, time, json, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import (hsv, Robot, Cam, parse, log, LOG, color_masks, blobs, blob_world, find_fixtures,
                 front_points, rotvec_to_R, R_to_rotvec, rot_err)

NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "a": 1, "an": 1}
COLORS = ["red", "green", "blue", "yellow", "orange", "cyan", "magenta", "purple", "pink"]
T0 = time.time()
WALL_LIMIT = 1050  # seconds for the controller (runtime limit 1200)

CARRY_Z = 0.17
APPROACH_Z = 0.12


def parse_instruction(text):
    text = text.lower()
    want = {}
    for m in re.finditer(r"\b(one|two|three|four|five|six|a|an|\d+)\s+(red|green|blue|yellow|orange|cyan|magenta|purple|pink)\s+cubes?", text):
        n = m.group(1)
        n = int(n) if n.isdigit() else NUM[n]
        want[m.group(2)] = want.get(m.group(2), 0) + n
    return want


def remaining_time():
    return WALL_LIMIT - (time.time() - T0)


def detect_cubes(obs, color, binc, cam="front", min_area=8):
    out = []
    masks = color_masks(obs["rgb"][cam])
    if color not in masks:
        return out
    for b in blobs(masks[color], min_area):
        p = blob_world(obs, cam, b)
        if p is None:
            continue
        if not (0.02 < p[2] < 0.07):  # resting on table / mat (not lifted, not in gripper)
            continue
        if binc is not None and np.linalg.norm(p[:2] - binc["xy"]) < 0.075:
            continue
        if p[0] < -0.45 or abs(p[1]) > 0.45 or p[0] > 0.30:
            continue
        out.append({"p": p, "area": b["area"], "mask": b["mask"]})
    return out


def cube_yaw(obs, cam, mask):
    """Yaw (rad, mod pi/2 in [-pi/4, pi/4]) of a cube's top face from world points."""
    import cv2
    c = Cam(obs["meta"], cam)
    d = obs["depth"][cam]
    vs, us = np.nonzero(mask & (d > 0))
    if len(us) < 12:
        return None
    pts = c.to_world(us, vs, d[vs, us])
    zt = np.percentile(pts[:, 2], 90)
    top = pts[pts[:, 2] > zt - 0.005][:, :2]
    if len(top) < 12:
        return None
    rect = cv2.minAreaRect((top * 1000).astype(np.float32))
    ang = np.deg2rad(rect[2])
    ang = (ang + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return ang


def yaw_of(rv):
    R = rotvec_to_R(np.asarray(rv))
    # gripper x-axis projected to world xy
    x = R[:, 0]
    return np.arctan2(x[1], x[0])


def rotate_to_yaw(R0, rob, dyaw, grip):
    """Rotate about world z by dyaw relative to R0 (the downward reference orientation)."""
    Rz = np.array([[np.cos(dyaw), -np.sin(dyaw), 0], [np.sin(dyaw), np.cos(dyaw), 0], [0, 0, 1]])
    tgt = R_to_rotvec(Rz @ R0)
    for _ in range(12):
        e = rot_err(rob.state[3:6], tgt)
        if np.linalg.norm(e) < 0.03 or rob.ended:
            break
        a = np.clip(e / 0.5, -1, 1)
        rob.step([[0, 0, 0, a[0], a[1], a[2], grip]])
    return tgt


def move(rob, tgt, grip, rot=None, max_steps=60, tol=0.004):
    """Translate with optional orientation hold."""
    tgt = np.asarray(tgt, float)
    for _ in range(max_steps):
        if rob.ended:
            return False
        p = rob.state[:3]
        err = tgt - p
        dist = np.linalg.norm(err)
        rerr = rot_err(rob.state[3:6], rot) if rot is not None else np.zeros(3)
        if dist < tol and np.linalg.norm(rerr) < 0.05:
            return True
        n = 3 if dist > 0.08 else (2 if dist > 0.025 else 1)
        a = np.clip(err / (0.05 * n), -1, 1)
        r = np.clip(rerr / (0.5 * n), -1, 1)
        rob.step([[a[0], a[1], a[2], r[0], r[1], r[2], grip]] * n)
    return np.linalg.norm(tgt - rob.state[:3]) < 3 * tol


def all_cubes(obs, binc, cam="front"):
    pts = []
    for col in ["red", "green", "blue", "yellow", "orange", "cyan", "magenta"]:
        pts += [c["p"] for c in detect_cubes(obs, col, binc, cam=cam)]
    return pts


def clearance(p, others):
    ds = [np.linalg.norm(p[:2] - o[:2]) for o in others if np.linalg.norm(p[:2] - o[:2]) > 0.01]
    return min(ds) if ds else 1.0


def choose(cands, ref_xy, others):
    # prefer isolated cubes (fingers need ~5 cm either side), then nearest
    def key(c):
        cl = clearance(c["p"], others)
        return (0 if cl > 0.065 else 1, np.linalg.norm(c["p"][:2] - ref_xy))
    return min(cands, key=key)


def finger_yaw(p, yaw, others):
    """Choose yaw or yaw+90deg so finger pads (closing along gripper y) avoid neighbours."""
    best, bestd = yaw, -1
    for y in (yaw, yaw + np.pi / 2 if yaw < 0 else yaw - np.pi / 2):
        d = np.array([-np.sin(y), np.cos(y)])  # rotated world-y closing axis
        fing = [p[:2] + 0.05 * d, p[:2] - 0.05 * d]
        dm = min([np.linalg.norm(f - o[:2]) for f in fing for o in others
                  if np.linalg.norm(p[:2] - o[:2]) > 0.01] or [1.0])
        if dm > bestd + 0.005:
            best, bestd = y, dm
    return best


def belt_velocity(rob, binc, dn=4):
    """Estimate common cube velocity (m/step) from two front observations dn steps apart."""
    o1 = rob.observe(); s1 = rob.step_id
    a = all_cubes(o1, binc)
    rob.hold(-1, dn)
    o2 = rob.observe(); s2 = rob.step_id
    b = all_cubes(o2, binc)
    if not a or not b or s2 <= s1:
        return np.zeros(2), o2
    ds = []
    for p in b:
        d = [p[:2] - q[:2] for q in a]
        j = int(np.argmin([np.linalg.norm(x) for x in d]))
        if np.linalg.norm(d[j]) < 0.004 * (s2 - s1) + 0.01:
            ds.append(d[j] / (s2 - s1))
    if len(ds) < 1:
        return np.zeros(2), o2
    v = np.median(np.array(ds), 0)
    if np.linalg.norm(v) < 0.0004:
        v = np.zeros(2)
    return v, o2


BELT = {"v": None, "t": -999}
LEAD = 3  # steps of feed-forward lead for tracking


def track_to(rob, cube, v, z, grip, rot, steps, tol=0.004, settle=False):
    """Move toward the predicted cube xy at height z while matching belt velocity.
    cube: dict p (xyz at step cube['t'])."""
    for _ in range(steps):
        if rob.ended:
            return False
        dt = rob.step_id - cube["t"]
        pred = cube["p"][:2] + v * (dt + LEAD)
        tgt = np.array([pred[0], pred[1], z])
        err = tgt - rob.state[:3]
        rerr = rot_err(rob.state[3:6], rot)
        if not settle and np.linalg.norm(err - np.r_[v * LEAD, 0]) < tol and np.linalg.norm(rerr) < 0.05:
            return True
        a = np.clip(err / 0.05, -1, 1)
        r = np.clip(rerr / 0.5, -1, 1)
        rob.step([[a[0], a[1], a[2], r[0], r[1], r[2], grip]])
    return True


def refine_cube(rob, color, binc, cube, v, cam, radius):
    obs = rob.observe()
    cs = detect_cubes(obs, color, binc, cam=cam, min_area=40 if cam == "wrist" else 8)
    pred = cube["p"][:2] + v * (rob.step_id - cube["t"])
    if not cs:
        return None, obs
    w = min(cs, key=lambda q: np.linalg.norm(q["p"][:2] - pred))
    if np.linalg.norm(w["p"][:2] - pred) > radius:
        return None, obs
    return {"p": w["p"].copy(), "t": rob.step_id, "mask": w["mask"]}, obs


def belt_exit(obs):
    """Max extent of the low, large, low-saturation mat along +y and -y (None if no mat)."""
    try:
        P = front_points(obs)
        h = hsv(obs["rgb"]["front"])
        m = (P[..., 2] > 0.006) & (P[..., 2] < 0.025) & (h[..., 1] < 60) & (P[..., 0] > -0.5)
        bl = blobs(m, 500)
        if not bl:
            return None
        b = max(bl, key=lambda q: q["area"])
        pts = P[b["mask"]]
        return np.percentile(pts[:, :2], 2, axis=0), np.percentile(pts[:, :2], 98, axis=0)
    except Exception:
        return None


MAT = {"ext": None}


def pick_and_place(rob, colors, binc, R0, home):
    """One attempt on any cube of the given colours. Returns (result, colour)."""
    v, obs = belt_velocity(rob, binc)
    BELT["v"] = v
    t_obs = rob.step_id
    cands = []
    for col in colors:
        for c in detect_cubes(obs, col, binc):
            c["color"] = col
            cands.append(c)
    others = all_cubes(obs, binc)
    if np.linalg.norm(v) > 0:
        lead = 45
        vd = v / np.linalg.norm(v)
        if MAT["ext"] is not None:
            lo, hi = MAT["ext"]
            corners = [np.array([a, b]) for a in (lo[0], hi[0]) for b in (lo[1], hi[1])]
            exitp = max(float(np.dot(q, vd)) for q in corners)
        else:
            exitp = 0.30
        ok = [c for c in cands if np.dot(c["p"][:2] + v * lead, vd) < exitp - 0.04]
        if ok:
            urgent = sorted(ok, key=lambda c: -np.dot(c["p"][:2], vd))
            clear = [c for c in urgent if clearance(c["p"], others) > 0.055]
            c = clear[0] if clear else urgent[0]
        else:
            c = None
    else:
        c = choose(cands, rob.state[:2], others) if cands else None
    if c is None:
        return "none", None
    color = c["color"]
    cube = {"p": c["p"].copy(), "t": t_obs, "mask": c["mask"]}
    log(f"[{rob.step_id}] target {color} at {np.round(cube['p'], 3)} v={np.round(v*1000,2)}mm/step ({len(cands)} cands)")
    rot0 = R_to_rotvec(R0)
    # approach above (with feed-forward of belt motion)
    track_to(rob, cube, v, APPROACH_Z, -1, rot0, 40, tol=0.008)
    r, obs = refine_cube(rob, color, binc, cube, v, "wrist", 0.05)
    yaw = 0.0
    if r is not None:
        cube = r
        y = cube_yaw(obs, "wrist", r["mask"])
        yaw = y if y is not None else 0.0
    ow = all_cubes(obs, binc, cam="wrist") or others
    yaw = finger_yaw(cube["p"] + np.r_[v * 0, 0], yaw, ow)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    tgt_rot = R_to_rotvec(Rz @ R0)
    track_to(rob, cube, v, APPROACH_Z, -1, tgt_rot, 15, tol=0.005)
    r, obs = refine_cube(rob, color, binc, cube, v, "wrist", 0.03)
    if r is not None:
        cube = r
    log(f"[{rob.step_id}] refined {np.round(cube['p'], 3)} yaw {np.degrees(yaw):.1f}")
    gz = max(cube["p"][2] - 0.020, 0.012)
    track_to(rob, cube, v, gz + 0.03, -1, tgt_rot, 15, tol=0.005)
    r, obs = refine_cube(rob, color, binc, cube, v, "wrist", 0.02)
    if r is not None:
        cube = r
    track_to(rob, cube, v, gz, -1, tgt_rot, 20, tol=0.003)
    # close while following the belt
    if np.linalg.norm(v) > 0:
        track_to(rob, cube, v, gz, 1, tgt_rot, 8, settle=True)
    else:
        rob.hold(1, 8)
    rob.observe()
    move(rob, rob.state[:3] + np.array([0, 0, 0.05]), 1, rot=tgt_rot, max_steps=10, tol=0.01)
    move(rob, [rob.state[0], rob.state[1], CARRY_Z], 1, rot=tgt_rot, max_steps=30, tol=0.01)
    rob.observe()
    g = rob.grip_open()
    log(f"[{rob.step_id}] after lift finger sum {g:.4f}")
    if not (0.02 < g < 0.06):
        rob.hold(-1, 4)
        return "missed", color
    # carry to bin
    bx, by = binc["xy"]
    move(rob, [bx, by, CARRY_Z], 1, rot=tgt_rot, max_steps=50, tol=0.008)
    rob.observe()
    g2 = rob.grip_open()
    if not (0.02 < g2 < 0.06):
        log(f"[{rob.step_id}] cube lost during carry ({g2:.4f})")
        rob.hold(-1, 4)
        return "missed", color
    ref = bin_from_wrist(rob.observe(), binc)
    if ref is not None and np.linalg.norm(ref - binc["xy"]) < 0.06:
        log(f"[{rob.step_id}] bin refine {np.round(binc['xy'],3)} -> {np.round(ref,3)}")
        binc["xy"] = 0.5 * binc["xy"] + 0.5 * ref
        move(rob, [binc["xy"][0], binc["xy"][1], CARRY_Z], 1, rot=tgt_rot, max_steps=10, tol=0.006)
    rob.hold(-1, 8)
    rob.observe()
    log(f"[{rob.step_id}] released over bin")
    return "placed", color


def bin_from_wrist(obs, binc):
    """xy of the bin's dark interior seen from above by the wrist camera."""
    try:
        c = Cam(obs["meta"], "wrist")
        d = obs["depth"]["wrist"]
        h = hsv(obs["rgb"]["wrist"])
        vv, uu = np.mgrid[0:d.shape[0], 0:d.shape[1]]
        P = c.to_world(uu, vv, d)
        m = (h[..., 2] < 35) & (d > 0) & (P[..., 2] < 0.06) & (P[..., 2] > -0.02)
        bl = blobs(m, 40)
        if not bl:
            return None
        b = max(bl, key=lambda q: q["area"])
        pts = P[b["mask"]]
        return (pts[:, :2].min(0) + pts[:, :2].max(0)) / 2 * 0.5 + np.median(pts[:, :2], 0) * 0.5
    except Exception:
        return None


def press_button(rob, btn, R0):
    bx, by = btn["xy"]
    top = btn["ztop"]
    rot = R_to_rotvec(R0)
    log(f"[{rob.step_id}] pressing button at {np.round(btn['xy'], 3)} top {top:.3f}")
    rob.hold(1, 6)
    move(rob, [bx, by, top + 0.08], 1, rot=rot, max_steps=50, tol=0.005)
    rob.observe()
    move(rob, [bx, by, top + 0.02], 1, rot=rot, max_steps=20, tol=0.004)
    # push down
    for k in range(25):
        if rob.ended:
            break
        rob.step([[0, 0, -0.4, 0, 0, 0, 1]])
        rob.observe()
        if rob.state[2] < top - 0.02:
            break
    rob.hold(1, 3)
    if not rob.ended:
        move(rob, [bx, by, top + 0.08], 1, rot=rot, max_steps=15, tol=0.01)
    return rob.ended


def get_fixtures(rob):
    obs = rob.observe()
    binc, btn, cands = find_fixtures(obs)
    res = {}
    if binc is not None:
        res["bin"] = {"xy": (np.asarray(binc["center"]) + np.asarray(binc["rim_center"])) / 2,
                      "ztop": binc["zmax"]}
    if btn is not None:
        res["btn"] = {"xy": np.asarray(btn["center"]), "ztop": btn["zmax"]}
    return res


def main(ctx, want_override=None):
    os.makedirs("/workspace/output", exist_ok=True)
    rob = Robot(ctx)
    obs = rob.observe()
    instr = obs["meta"].get("instruction", "")
    want = parse_instruction(instr)
    if want_override:
        want = dict(want_override)
    log("instruction:", instr, "->", want)
    home = rob.state[:3].copy()
    R0 = rotvec_to_R(rob.state[3:6])
    fx = get_fixtures(rob)
    log("fixtures:", {k: {kk: np.round(vv, 3).tolist() for kk, vv in v.items()} for k, v in fx.items()})
    binc = fx.get("bin")
    btn = fx.get("btn")
    MAT["ext"] = belt_exit(rob.obs)
    log("mat extent", None if MAT["ext"] is None else [np.round(x, 3).tolist() for x in MAT["ext"]])
    done = {c: 0 for c in want}
    fails = 0
    order = list(want.keys())
    try:
        while not rob.ended and remaining_time() > 120:
            todo = [c for c in order if done[c] < want[c]]
            if not todo:
                break
            if rob.remaining is not None and rob.remaining < 150:
                log("low steps; going to button")
                break
            if binc is None:
                fx = get_fixtures(rob); binc = fx.get("bin"); btn = fx.get("btn") or btn
                if binc is None:
                    rob.hold(0, 10); continue
            progressed = False
            r, color = pick_and_place(rob, todo, binc, R0, home)
            log(f"[{rob.step_id}] {color}: {r}")
            if r == "placed":
                done[color] += 1
                progressed = True
            elif r == "missed":
                fails += 1
                progressed = True
            log("progress", done, "want", want)
            if not progressed:
                # no target cube visible: park out of the camera view and wait for spawns
                move(rob, home, -1, rot=R_to_rotvec(R0), max_steps=30, tol=0.02)
                rob.hold(-1, 6)
        if btn is None:
            move(rob, home, -1, rot=R_to_rotvec(R0), max_steps=30, tol=0.02)
            fx = get_fixtures(rob); btn = fx.get("btn")
        offsets = [(0, 0), (0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01), (0.015, 0.015), (-0.015, -0.015)]
        for k, off in enumerate(offsets):
            if rob.ended or btn is None:
                break
            b2 = dict(btn); b2["xy"] = np.asarray(btn["xy"]) + np.asarray(off)
            press_button(rob, b2, R0)
            if not rob.ended:
                rob.hold(1, 5)
            if not rob.ended and k == 0:
                move(rob, home, 1, rot=R_to_rotvec(R0), max_steps=30, tol=0.02)
                fx = get_fixtures(rob)
                if fx.get("btn") is not None and np.linalg.norm(fx["btn"]["xy"] - btn["xy"]) < 0.05:
                    btn = fx["btn"]
        log("final", done, "ended", rob.ended, "complete", rob.complete, "step", rob.step_id)
    except Exception:
        log("EXC", traceback.format_exc())
    finally:
        try:
            with open("/workspace/output/controller_log.txt", "w") as f:
                f.write("\n".join(LOG))
        except Exception:
            pass
