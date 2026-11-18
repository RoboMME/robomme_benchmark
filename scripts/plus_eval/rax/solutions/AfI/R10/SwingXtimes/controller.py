"""SwingXtimes controller v1.

Plan: perceive targets/button/cube at step 0 (unoccluded), parse X from the instruction,
grasp red cube (wrist-refined), alternate right(+y, camera-right) -> left(-y) X times,
put the cube back where it was picked, press the button with a closed gripper.
"""
import math, re, time, json, os, traceback
import numpy as np
import lib
from lib import log

WORDS = {"once": 1, "twice": 2, "thrice": 3, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}

def parse_count(instr):
    t = instr.lower()
    m = re.search(r"(\d+)\s*(?:times|x\b)", t)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(" + "|".join(WORDS) + r")\b(?:\s+times)?", t)
    for w in re.findall(r"\b(\w+)\s+times\b", t):
        if w in WORDS:
            return WORDS[w]
        if w.isdigit():
            return int(w)
    for w in ("once", "twice", "thrice"):
        if w in t:
            return WORDS[w]
    nums = [WORDS[w] for w in re.findall(r"\b\w+\b", t) if w in WORDS and w not in ("one",)]
    return nums[0] if nums else 1

def parse_first(instr):
    t = instr.lower()
    m = re.search(r"\b(left|right)\b", t)
    m2 = re.search(r"(left|right)-to-(left|right)", t)
    if m2:
        return m2.group(1)
    return m.group(1) if m else "right"

def assign_sides(tg, bxy, cxy):
    """H7: layout frame. f = direction (perp. to target line) from the cube side toward the button side;
    a viewer looking along f has 'right' = f x z = (f_y, -f_x)."""
    T0, T1 = np.array(tg[0]), np.array(tg[1])
    mid = (T0 + T1) / 2; t = (T1 - T0) / (np.linalg.norm(T1 - T0) + 1e-9)
    def perp(v):
        v = np.array(v, float) - mid
        return v - np.dot(v, t) * t
    votes = []
    if bxy is not None:
        f = perp(bxy)
        if np.linalg.norm(f) > 0.02:
            votes.append(("button", f / np.linalg.norm(f)))
    if cxy is not None:
        f = -perp(cxy)
        if np.linalg.norm(f) > 0.02:
            votes.append(("cube", f / np.linalg.norm(f)))
    if not votes:
        log("assign_sides: no reference; default camera view")
        return (T1, T0) if T1[1] > T0[1] else (T0, T1)
    if len(votes) == 2 and np.dot(votes[0][1], votes[1][1]) < 0:
        log("assign_sides: button and cube disagree; using button")
    f = votes[0][1]
    rd = np.array([f[1], -f[0]])
    right, left = (T0, T1) if np.dot(T0 - mid, rd) > np.dot(T1 - mid, rd) else (T1, T0)
    log("assign_sides f=", f.round(3), "via", votes[0][0], "right=", right.round(3), "left=", left.round(3))
    return right, left

def waypoints(p, g, avoid, clear=0.10, depth=0):
    """xy waypoints from p to g keeping >= clear from every avoid point (recursive detours)."""
    p, g = np.array(p[:2], float), np.array(g[:2], float)
    for a in avoid:
        if a is None:
            continue
        a = np.array(a, float)
        if np.linalg.norm(a - p) < 0.05 or np.linalg.norm(a - g) < 0.05:
            continue
        d = g - p; L2 = float(np.dot(d, d))
        if L2 < 1e-9:
            continue
        u = min(1.0, max(0.0, float(np.dot(a - p, d) / L2)))
        c = p + u * d
        if np.linalg.norm(c - a) < clear and 0.0 < u < 1.0:
            v = c - a
            if np.linalg.norm(v) < 1e-3:
                v = np.array([-d[1], d[0]])
            w = a + v / np.linalg.norm(v) * (clear + 0.03)
            if depth >= 3:
                return [w, g]
            return waypoints(p, w, avoid, clear, depth + 1) + waypoints(w, g, avoid, clear, depth + 1)
    return [g]

def safe_move(r, g, z, Rt, avoid, tol=0.008, max_steps=50):
    """Lift to z, then move in xy along a path that keeps clear of the avoid points."""
    s = r.state
    if s[2] < z - 0.01:
        r.move_to([s[0], s[1], z], Rt, tol=0.01, max_steps=25)
    wps = waypoints(r.state[:2], g, avoid)
    for w in wps[:-1]:
        log("  waypoint", np.round(w, 3))
        r.move_to([w[0], w[1], z], Rt, tol=0.015, max_steps=40)
    r.move_to([g[0], g[1], z], Rt, tol=tol, max_steps=max_steps)

def travel(r, g, Rt, avoid, z=0.09):
    s = r.state
    if s[2] < z - 0.01:
        r.move_to([s[0], s[1], z], Rt, tol=0.01, max_steps=25)
    for w in waypoints(r.state[:2], g, avoid):
        r.move_to([w[0], w[1], z], Rt, tol=0.01, max_steps=50)

def find_target(r, est, other, btn_xy, nref=None):
    """Front-camera disc nearest to est; only full (unoccluded) discs are trusted."""
    imgs = r.images()
    best = None
    try:
        cands = lib.flat_discs(r.meta, imgs, "front")
    except Exception as e:
        log("flat_discs fail", e); return None
    for xy, c in cands:
        d = float(np.linalg.norm(xy - est))
        if d > 0.15:
            continue
        if other is not None and np.linalg.norm(xy - other) < d:
            continue
        if btn_xy is not None and np.linalg.norm(xy - btn_xy) < 0.06:
            continue
        if nref is not None and c < 0.8 * nref:
            continue
        if best is None or d < best[0]:
            best = (d, xy, "front", c)
    return best

def highlighted(r, xy):
    n, col, grey = lib.disc_stats(r.meta, r.images(), xy)
    return n, col, grey

def look_for(r, est, other, btn_xy, nref, Rt):
    """Retreat high and toward the camera so the front view is clear; re-detect the target."""
    log("  target lost near", np.round(est, 3), "- retreat and look")
    r.move_to([est[0] + 0.10, est[1], 0.20], Rt, tol=0.02, max_steps=25)
    r.hold(1)
    b = find_target_wide(r, est, other, btn_xy, nref)
    if b is not None:
        log("  look found", np.round(b[1], 3), "n=", b[3])
        return np.array(b[1]), True
    return est, False

def find_target_wide(r, est, other, btn_xy, nref):
    imgs = r.images()
    best = None
    for cam in ("front", "wrist"):
        try:
            cands = lib.flat_discs(r.meta, imgs, cam)
        except Exception:
            continue
        for xy, c in cands:
            d = float(np.linalg.norm(xy - est))
            if d > 0.25 or (other is not None and np.linalg.norm(xy - other) < d):
                continue
            if btn_xy is not None and np.linalg.norm(xy - btn_xy) < 0.06:
                continue
            if cam == "front" and nref is not None and c < 0.8 * nref:
                continue
            if cam == "wrist" and c < 150:
                continue
            if best is None or d < best[0]:
                best = (d, xy, cam, c)
    return best

def tracked_visit(r, est, other, Rt, btn_xy, nref=None, z_travel=0.09, z_touch=0.032, dwell=2):
    est = np.array(est, float)
    s = r.state
    if s[2] < z_travel - 0.01:
        r.move_to([s[0], s[1], z_travel], Rt, tol=0.01, max_steps=20)
    for attempt in range(5):
        lost = 0; ready = False
        for it in range(40):
            if r.ended:
                return est
            b = find_target(r, est, other, btn_xy, nref)
            if b is not None and b[0] < 0.12:
                if b[0] > 0.01:
                    log(f"  target update {est.round(3)} -> {b[1].round(3)} (n={b[3]})")
                est = np.array(b[1]); lost = 0; seen = True
            else:
                lost += 1; seen = False
            if lost >= 3:
                est, ok = look_for(r, est, other, btn_xy, nref, Rt)
                lost = 0
                r.move_to([r.state[0], r.state[1], z_travel], Rt, tol=0.01, max_steps=10)
                continue
            s = r.state; d = est - s[:2]; dist = float(np.linalg.norm(d))
            if dist < 0.007 and abs(s[2] - z_travel) < 0.015:
                if seen:
                    ready = True; break
                r.hold(1)
                continue
            goal = est if dist < 0.12 else s[:2] + d / dist * 0.12
            r.move_to([goal[0], goal[1], z_travel], Rt, tol=0.006, max_steps=4)
        if not ready:
            log("  could not confirm target; descending at estimate")
        r.move_to([est[0], est[1], z_touch], Rt, tol=0.004, max_steps=30)
        if dwell:
            r.hold(dwell)
        if r.ended:
            return est
        r.move_to([est[0], est[1], 0.06], Rt, tol=0.01, max_steps=15)
        n, col, grey = highlighted(r, est)
        b = find_target(r, est, other, btn_xy, nref)
        log(f"  visit check {attempt}: disc px={n} coloured={col} grey={grey} found={b and (round(b[0],3), b[3])}")
        if col >= 8:
            break
        if b is not None and b[0] < 0.012:
            # disc still here but no highlight: highlights may be absent; accept after one retry
            if attempt >= 1:
                break
            continue
        # target moved or not visible: track again from travel height
        r.move_to([est[0], est[1], z_travel], Rt, tol=0.01, max_steps=15)
    r.move_to([est[0], est[1], z_travel], Rt, tol=0.01, max_steps=20)
    return est

def scene_from(meta, imgs):
    out = {}
    for cam in ("front", "wrist"):
        try:
            out[cam] = lib.perceive(meta, imgs, cam)
        except Exception as e:
            log("perceive fail", cam, e)
    return out

def merge_scene(sc):
    f, w = sc.get("front", {}), sc.get("wrist", {})
    tg = None
    for src in (f, w):
        if src and len(src.get("targets", [])) == 2:
            tg = [t["xy"] for t in src["targets"]]
            break
    # cross-check / average with wrist estimates when they agree
    if tg is not None and w and len(w.get("targets", [])) == 2:
        wt = [t["xy"] for t in w["targets"]]
        tg = [(a + b) / 2 if np.linalg.norm(a - b) < 0.02 else a for a, b in zip(tg, wt)]
    btn = None
    for src in (f, w):
        if src and src.get("button") is not None:
            btn = (src["button"]["xy"], src["button"]["ztop"]); break
    red = None
    for src in (w, f):
        if src and src.get("red") is not None:
            red = src["red"]["xy"]; break
    return tg, btn, red

AVOID = []

def grasp(r, guess_xy, attempts=3):
    """Grasp the cube; return (ok, xy, yaw)."""
    xy = np.array(guess_xy, float); yaw = 0.0
    for a in range(attempts):
        if r.ended:
            return False, xy, yaw
        r.grip = -1
        safe_move(r, xy, max(0.15, min(0.3, r.state[2])), lib.down_R(yaw), AVOID, tol=0.02)
        r.move_to([xy[0], xy[1], 0.15], lib.down_R(yaw), grip=-1, tol=0.01, max_steps=50)
        res = lib.red_top_points(r.meta, r.images(), "wrist")
        if res is None:
            res = lib.red_top_points(r.meta, r.images(), "front")
        if res is None:
            log("grasp: cube not seen"); r.move_to([xy[0], xy[1], 0.25], lib.down_R(yaw), tol=0.01, max_steps=30)
            continue
        pts, top = res
        yaw, ext, c = lib.square_yaw(top[:, :2]); zt = float(np.max(top[:, 2]))
        xy = np.array(c)
        Rt = lib.down_R(yaw)
        r.move_to([xy[0], xy[1], 0.10], Rt, tol=0.006, max_steps=40)
        res = lib.red_top_points(r.meta, r.images(), "wrist")
        if res is not None:
            _, top = res
            yaw2, ext, c = lib.square_yaw(top[:, :2])
            if np.linalg.norm(np.array(c) - xy) < 0.03:
                xy = np.array(c); zt = float(np.max(top[:, 2]))
        log(f"grasp attempt {a}: xy={xy.round(4)} yaw={math.degrees(yaw):.1f} zt={zt:.3f}")
        r.move_to([xy[0], xy[1], zt - 0.022], Rt, tol=0.003, max_steps=40)
        r.hold(8, grip=1)
        r.move_to([xy[0], xy[1], 0.12], Rt, tol=0.01, max_steps=30)
        fs = r.state[6] + r.state[7]
        log(f"after lift fingers={fs:.4f} z={r.state[2]:.3f}")
        if fs > 0.02:
            return True, xy, yaw
        r.hold(3, grip=-1)
    return False, xy, yaw

def holding(r):
    return (r.state[6] + r.state[7]) > 0.02

def main(ctx):
    try:
        _main(ctx)
    except Exception:
        log('EXCEPTION', traceback.format_exc())

def _main(ctx):
    t0 = time.time()
    r = lib.Robot(ctx)
    r.observe()
    instr = r.meta.get("instruction", "")
    X = parse_count(instr)
    mcol = re.search(r"pick up the (\w+) (?:cube|block)", instr.lower())
    lib.COLOR["name"] = mcol.group(1) if mcol else "red"
    log("cube color:", lib.COLOR["name"])
    log("instruction:", instr, "| X =", X)
    sc = scene_from(r.meta, r.images())
    tg, btn, red = merge_scene(sc)
    log("targets", tg, "button", btn, "red", red)
    if tg is None or red is None:
        log("perception failed at step 0; moving up for a wider look")
        r.move_to([r.state[0], r.state[1], r.state[2]], None, max_steps=1)
        sc = scene_from(r.meta, r.images()); tg2, btn2, red2 = merge_scene(sc)
        tg = tg or tg2; btn = btn or btn2; red = red if red is not None else red2
    if tg is None or red is None:
        log("ABORT: missing targets or cube"); return
    right, left = assign_sides(tg, btn[0] if btn else None, red)
    nref = {}
    try:
        cands = lib.flat_discs(r.meta, r.images(), "front")
        for nm, xy in (("R", right), ("L", left)):
            cs = [c for p_, c in cands if np.linalg.norm(p_ - xy) < 0.03]
            if cs:
                nref[nm] = max(cs)
    except Exception as e:
        log("nref fail", e)
    log("nref", nref)
    first = parse_first(instr)
    seq = (("R", right), ("L", left)) if first == "right" else (("L", left), ("R", right))
    log("first side:", first)
    AVOID[:] = [right, left]
    ok, pick_xy, yaw = grasp(r, red)
    if not ok:
        log("ABORT: grasp failed"); return
    Rt = lib.down_R(yaw)
    done = 0
    first_visit = True
    while done < X and not r.ended:
        for name in [n_ for n_, _ in seq]:
            xy = right if name == "R" else left
            if not holding(r):
                log("dropped cube before", name, "; regrasp")
                r.move_to([r.state[0], r.state[1], 0.2], Rt, tol=0.01, max_steps=20)
                r.observe()
                res = lib.red_top_points(r.meta, r.images(), "front")
                guess = np.nanmedian(res[1][:, :2], 0) if res is not None else pick_xy
                ok, _, yaw = grasp(r, guess); Rt = lib.down_R(yaw)
                if not ok:
                    log("ABORT regrasp"); return
            other = left if name == "R" else right
            if first_visit:
                s_ = r.state
                if s_[2] < 0.08:
                    r.move_to([s_[0], s_[1], 0.09], Rt, tol=0.01, max_steps=20)
                for w in waypoints(r.state[:2], xy, [other])[:-1]:
                    log("  waypoint", np.round(w, 3))
                    r.move_to([w[0], w[1], 0.09], Rt, tol=0.015, max_steps=40)
            first_visit = False
            newxy = tracked_visit(r, xy, other, Rt, btn[0] if btn else None, nref.get(name))
            if name == "R":
                right = newxy
            else:
                left = newxy
            log(f"swing {done+1}/{X} {name} step={r.meta['step_id']} hold={holding(r)}")
            if r.ended:
                break
        done += 1
    if r.ended:
        log("ended during swings", r.complete); return
    # put the cube down where it came from (or a safe spot), avoiding both targets
    drop = np.array(pick_xy, float)
    mid = (right + left) / 2
    if min(np.linalg.norm(drop - right), np.linalg.norm(drop - left)) < 0.12:
        tdir = (right - left) / np.linalg.norm(right - left)
        f = np.array([-tdir[1], tdir[0]])
        cand = [mid + f * 0.15, mid - f * 0.15]
        bx = btn[0] if btn else mid
        drop = max(cand, key=lambda c: np.linalg.norm(c - bx))
        log("drop spot moved to", drop.round(3))
    r.move_to([r.state[0], r.state[1], 0.12], Rt, tol=0.01, max_steps=25)
    safe_move(r, drop, 0.12, Rt, [right, left])
    r.move_to([drop[0], drop[1], 0.10], Rt, tol=0.008, max_steps=50)
    r.move_to([drop[0], drop[1], 0.03], Rt, tol=0.004, max_steps=30)
    r.hold(6, grip=-1)
    r.move_to([drop[0], drop[1], 0.12], Rt, tol=0.01, max_steps=30)
    log("cube released step", r.meta["step_id"])
    if btn is None:
        log("button not perceived; ABORT"); return
    bxy, btop = btn
    R0 = lib.down_R(0.0)
    r.grip = 1
    safe_move(r, bxy, 0.12, R0, [right, left], tol=0.006)
    r.hold(4, grip=1)
    for depth in (0.012, 0.02, 0.028):
        if r.ended:
            break
        r.move_to([bxy[0], bxy[1], btop + 0.02], R0, tol=0.004, max_steps=30)
        r.move_to([bxy[0], bxy[1], btop - depth], R0, tol=0.003, max_steps=20)
        r.hold(3)
        log(f"press depth {depth} z={r.state[2]:.3f} ended={r.ended} complete={r.complete}")
        if r.ended:
            break
        r.move_to([bxy[0], bxy[1], btop + 0.05], R0, tol=0.006, max_steps=20)
        r.hold(3)
        if r.ended:
            break
    log("done: ended", r.ended, "complete", r.complete, "step", r.meta.get("step_id"), "wall", round(time.time()-t0, 1))
