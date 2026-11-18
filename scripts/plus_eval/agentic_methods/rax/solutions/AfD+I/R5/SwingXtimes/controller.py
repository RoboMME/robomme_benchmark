"""SwingXtimes controller: pick cube, swing between targets N times, place, press button."""
import os, re, sys, time, json, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion import Robot
from scene import analyze, color_mask, white_mask, blobs, components
from perc import backproject

NUMW = {"once": 1, "twice": 2, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "thrice": 3}
LOGF = None

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    try:
        with open(LOGF, "a") as f: f.write(s + "\n")
    except Exception: pass

def parse_instruction(text):
    t = text.lower()
    m = re.search(r"pick up the (\w+) cube", t)
    color = m.group(1) if m else "red"
    m = re.search(r"move it to the (left|right)-side target", t)
    first = m.group(1) if m else "right"
    n = None
    m = re.search(r"(\w+) times", t)
    if m:
        w = m.group(1)
        n = int(w) if w.isdigit() else NUMW.get(w)
    if n is None:
        for w, k in NUMW.items():
            if re.search(r"\b%s\b" % w, t.split("swing")[-1]): n = k; break
    if n is None: n = 1
    return color, first, n

def find_scene(r, color):
    meta, imgs = r.meta, r.imgs
    rgb = imgs[("front", "rgb")][..., :3]
    pw = backproject(meta, imgs, "front")
    zt = 0.0
    valid = ~np.isnan(pw).any(-1)
    X, Y, Z = pw[..., 0], pw[..., 1], pw[..., 2]
    onw = valid & (np.nan_to_num(X, nan=-9) > -0.42) & (np.nan_to_num(Z, nan=9) < 0.08) & (np.nan_to_num(Z, nan=-9) > -0.01)
    # cube
    cm = color_mask(rgb, color) & onw
    cubes = []
    for pix in components(cm, 6):
        pts = pw[pix[:, 0], pix[:, 1]]; pts = pts[~np.isnan(pts).any(1)]
        if len(pts) < 6: continue
        zmax = np.percentile(pts[:, 2], 95)
        top = pts[pts[:, 2] > zmax - 0.008]
        cubes.append(dict(n=len(pix), xy=np.median(top[:, :2], 0), zmax=float(zmax), uv=pix.mean(0)[::-1]))
    cubes = [c for c in cubes if 0.015 < c["zmax"] < 0.07]
    cubes.sort(key=lambda c: -c["n"])
    # white/grey objects
    h_s_v = None
    wm = white_mask_loose(rgb) & onw
    objs = []
    for pix in components(wm, 8):
        pts = pw[pix[:, 0], pix[:, 1]]; pts = pts[~np.isnan(pts).any(1)]
        if len(pts) < 8: continue
        zmax = float(np.percentile(pts[:, 2], 95))
        ext = np.ptp(pts[:, :2], 0)
        objs.append(dict(n=len(pix), xy=np.median(pts[:, :2], 0), zmax=zmax, pix=pix, ext=ext))
    # merge objects closer than 3cm (target ring pieces)
    merged = []
    for o in sorted(objs, key=lambda o: -o["n"]):
        for m in merged:
            if np.linalg.norm(m["xy"] - o["xy"]) < 0.035:
                m["pix"] = np.concatenate([m["pix"], o["pix"]]); m["n"] += o["n"]; m["zmax"] = max(m["zmax"], o["zmax"]); break
        else:
            merged.append(dict(o))
    targets = [o for o in merged if o["zmax"] < 0.018 and o["n"] >= 30]
    buttons = [o for o in merged if 0.018 <= o["zmax"] < 0.07 and o["n"] >= 30]
    targets.sort(key=lambda o: -o["n"]); buttons.sort(key=lambda o: -o["n"])
    return cubes, targets[:2], buttons[:1]

def white_mask_loose(rgb):
    from scene import hsv
    h, s, v = hsv(rgb)
    return (s < 0.2) & (v > 0.35)

def target_lit(r, tgt):
    """Fraction of the target disc's (initially white) pixels now showing the lit red rings."""
    a = r.imgs[("front", "rgb")][..., :3].astype(int)
    pix = tgt["pix"]
    vals = a[pix[:, 0], pix[:, 1]]
    R, G, B = vals[:, 0], vals[:, 1], vals[:, 2]
    red = (R > 150) & (G < 70) & (B < 70) & (R - G > 110)
    white = (R > 170) & (G > 170) & (B > 170)
    # a lit disc shows red AND white rings; a red cube covering it shows no white
    if white.mean() < 0.05 and red.mean() > 0.5:
        return 0.5 * red.mean()
    return float(red.mean())

def wrist_cube(r, color, prior_xy):
    try:
        rgb = r.imgs[("wrist", "rgb")][..., :3]
        pw = backproject(r.meta, r.imgs, "wrist")
        m = color_mask(rgb, color) & ~np.isnan(pw).any(-1)
        best = None
        for pix in components(m, 15):
            pts = pw[pix[:, 0], pix[:, 1]]
            zmax = np.percentile(pts[:, 2], 95)
            top = pts[pts[:, 2] > zmax - 0.006]
            xy = np.median(top[:, :2], 0)
            if not (0.015 < zmax < 0.07): continue
            if np.linalg.norm(xy - prior_xy) > 0.06: continue
            if best is None or len(pix) > best[0]: best = (len(pix), xy)
        return None if best is None else best[1]
    except Exception as e:
        log("wrist_cube error", repr(e)); return None


def detect_discs(r):
    """Flat disc-like objects (targets) at table level in the current front image.
    Returns list of dict(xy, n, lit) where lit = strict-red fraction near the disc."""
    rgb = r.imgs[("front", "rgb")][..., :3]
    a = rgb.astype(int)
    pw = backproject(r.meta, r.imgs, "front")
    X = np.nan_to_num(pw[..., 0], nan=-9); Z = np.nan_to_num(pw[..., 2], nan=9)
    tbl = (X > -0.42) & (Z < 0.017) & (Z > -0.01)
    wm = white_mask_loose(rgb) & tbl
    red = (a[..., 0] > 150) & (a[..., 1] < 70) & (a[..., 2] < 70) & (a[..., 0] - a[..., 1] > 110) & tbl
    # high clusters (button) must be excluded: build white mask of everything above too
    hi = white_mask_loose(rgb) & (X > -0.42) & (Z >= 0.017) & (Z < 0.07)
    comps = components(wm | red, 6)
    out = []
    for pix in comps:
        pts = pw[pix[:, 0], pix[:, 1]]; pts = pts[~np.isnan(pts).any(1)]
        if len(pts) < 6: continue
        xy = np.median(pts[:, :2], 0)
        out.append(dict(xy=xy, n=len(pix), pix=pix, nred=int(red[pix[:, 0], pix[:, 1]].sum())))
    # merge fragments within 3.5cm
    merged = []
    for o in sorted(out, key=lambda o: -o["n"]):
        for m in merged:
            if np.linalg.norm(m["xy"] - o["xy"]) < 0.035:
                m["n"] += o["n"]; m["nred"] += o["nred"]; m["pix"] = np.concatenate([m["pix"], o["pix"]]); break
        else: merged.append(dict(o))
    # drop clusters near a raised white object (button base edges)
    hp = pw[hi]; hp = hp[~np.isnan(hp).any(1)]
    res = []
    for m in merged:
        if m["n"] < 12: continue
        if len(hp) and np.min(np.linalg.norm(hp[:, :2] - m["xy"], axis=1)) < 0.03: continue
        nwhite = m["n"] - m["nred"]
        if nwhite < 4: continue   # pure red blob = glow / cube, not a disc
        m["lit"] = m["nred"] / m["n"]
        res.append(m)
    return res

def track(r, tg, max_jump=0.15):
    """Update target estimate tg (dict with xy) from the current front image. Returns lit fraction or None."""
    ds = detect_discs(r)
    if not ds: return None
    d = min(ds, key=lambda d: np.linalg.norm(d["xy"] - tg["xy"]))
    if np.linalg.norm(d["xy"] - tg["xy"]) > max_jump: return None
    tg["xy"] = d["xy"].copy(); tg["pix"] = d["pix"]
    return d["lit"]

def approach_target(r, tg, hover, off, max_steps=140):
    """Move the held cube over a (possibly moving) target until it lights. Returns (lit, off)."""
    OFFS = [(0, 0), (0.012, 0), (-0.012, 0), (0, 0.012), (0, -0.012), (0.024, 0), (-0.024, 0)]
    used0 = r.remaining; k = 0; settle = 0; z = hover; lit = 0.0
    while not r.ended and used0 - r.remaining < max_steps:
        l = track(r, tg)
        if l is not None: lit = l
        if l is not None and l >= 0.15 and np.linalg.norm(r.pos[:2] - (tg["xy"] + off)) < 0.03:
            return True, off
        goal = np.array([tg["xy"][0] + off[0], tg["xy"][1] + off[1], z])
        dist = np.linalg.norm(goal - r.pos)
        if dist < 0.006:
            settle += 1
            if settle >= 2:   # arrived but not lit: try next offset / lower
                settle = 0; k += 1
                if k < len(OFFS):
                    off = off + np.array(OFFS[k]) - np.array(OFFS[k - 1])
                elif z > hover - 0.011:
                    z = hover - 0.01; k = 0; off = off - np.array(OFFS[-1])
                else:
                    return False, off
            r.hold(2)
        else:
            r.move_to(goal, vmax=0.04, tol=0.005, max_steps=3, batch=3)
    return False, off

def main(ctx):
    global LOGF
    os.makedirs("/workspace/output", exist_ok=True)
    LOGF = "/workspace/output/controller_log.txt"
    t0 = time.time()
    r = Robot(ctx, log=log)
    instr = r.meta.get("instruction", "")
    color, first, n = parse_instruction(instr)
    log("instr:", instr); log("parsed:", color, first, n, "limit", r.remaining)
    try:
        run(r, color, first, n)
    except Exception as e:
        log("ERROR", repr(e), traceback.format_exc())
        # fall through: try to at least press button
    log("done; complete=", r.complete, "ended=", r.ended, "steps_left", r.remaining, "wall", time.time() - t0)

def run(r, color, first, n):
    cubes, targets, buttons = find_scene(r, color)
    log("cubes", [(np.round(c["xy"], 3), c["n"], round(c["zmax"], 3)) for c in cubes])
    log("targets", [(np.round(t["xy"], 3), t["n"], round(t["zmax"], 3)) for t in targets])
    log("buttons", [(np.round(b["xy"], 3), b["n"], round(b["zmax"], 3)) for b in buttons])
    if len(targets) < 2 or not cubes:
        raise RuntimeError("perception failed")
    T = sorted(targets, key=lambda t: t["xy"][1])  # y ascending: right (-y) first
    right, left = T[0], T[1]
    seq_first, seq_second = (right, left) if first == "right" else (left, right)
    button = buttons[0] if buttons else None
    cube = cubes[0]
    cz_top = cube["zmax"]
    # --- pick ---
    cx, cy = cube["xy"]
    grasp_z = max(0.012, cz_top - 0.018)
    r.move_to([cx, cy, cz_top + 0.07], grip=-1, vmax=0.05, tol=0.006)
    # refine using a fresh observation
    w = wrist_cube(r, color, cube["xy"])
    if w is not None:
        log("wrist refined cube", np.round(w, 3), "front was", np.round(cube["xy"], 3))
        cx, cy = w
        r.move_to([cx, cy, cz_top + 0.07], vmax=0.03, tol=0.004)
        w2 = wrist_cube(r, color, w)
        if w2 is not None: cx, cy = w2; log("wrist refined 2", np.round(w2, 3))
    else:
        cubes2, _, _ = find_scene(r, color)
        near = [c for c in cubes2 if np.linalg.norm(c["xy"] - cube["xy"]) < 0.05]
        if near: cx, cy = near[0]["xy"]; log("refined cube (front)", np.round(near[0]["xy"], 3))
    r.move_to([cx, cy, cz_top + 0.03], vmax=0.03, tol=0.005)
    r.move_to([cx, cy, grasp_z], vmax=0.02, tol=0.004)
    r.hold(8, grip=1)
    fing = r.state[6:].sum(); log("grasp fingers", fing)
    r.move_to(r.pos + [0, 0, 0.05], vmax=0.02, tol=0.006)
    held = check_held(r, color)
    log("held after lift:", held, r.state[6:].sum())
    if not held:
        # one retry
        cubes3, _, _ = find_scene(r, color)
        if cubes3:
            c = cubes3[0]; log("retry pick at", c["xy"])
            r.move_to([c["xy"][0], c["xy"][1], c["zmax"] + 0.06], grip=-1, vmax=0.03)
            r.move_to([c["xy"][0], c["xy"][1], max(0.012, c["zmax"] - 0.018)], vmax=0.02)
            r.hold(8, grip=1)
            r.move_to(r.pos + [0, 0, 0.05], vmax=0.02)
    # --- travel high to first target ---
    hover = 0.056   # TCP z: cube centre ~ 4.8cm over table
    r.move_to([r.pos[0], r.pos[1], 0.13], vmax=0.04, tol=0.01)
    track(r, seq_first); f = seq_first["xy"]
    r.move_to([f[0], f[1], 0.13], vmax=0.04, tol=0.02)
    track(r, seq_first); f = seq_first["xy"]
    r.move_to([f[0], f[1], hover + 0.02], vmax=0.04, tol=0.01)
    visits = [seq_first, seq_second] * n
    off = np.zeros(2)   # learned TCP-vs-cube xy offset
    for i, tg in enumerate(visits):
        track(r, tg)
        x, y = tg["xy"]
        if i > 0:
            mid = (r.pos[:2] + np.array([x, y]) + off) / 2
            r.move_to([mid[0], mid[1], hover + 0.03], vmax=0.04, tol=0.025)
        lit, off2 = approach_target(r, tg, hover, off)
        if lit: off = off2
        log(f"visit {i+1}/{len(visits)} lit={lit} tgt={np.round(tg['xy'],3)} off={np.round(off,3)} pos={np.round(r.pos,3)} steps_left={r.remaining}")
        if r.ended: return
        if not lit: log("WARNING target never lit")
    # --- put down on last target ---
    x, y = r.pos[0], r.pos[1]
    r.move_to([x, y, 0.009 + 0.024], vmax=0.02, tol=0.004)
    r.hold(6, grip=-1)
    r.move_to(r.pos + [0, 0, 0.08], vmax=0.04, tol=0.01)
    log("placed; steps_left", r.remaining)
    # --- press button ---
    press(r, button)

def check_held(r, color):
    # cube visible in front image above table near gripper?
    s = r.state[6:].sum()
    if 0.02 < s < 0.07:   # closed on a ~4cm cube; ~0 means closed on nothing
        return True
    return False

def press(r, button):
    if button is None:
        _, _, buttons = find_scene(r, "red")
        if not buttons: log("no button found"); return
        button = buttons[0]
    try:
        _, _, bs = find_scene(r, "none")
        bs = [b for b in bs if np.linalg.norm(b["xy"] - button["xy"]) < 0.12]
        if bs: log("button re-detected", np.round(bs[0]["xy"], 3), "was", np.round(button["xy"], 3)); button = bs[0]
    except Exception as e: log("button redetect err", repr(e))
    bx, by = button["xy"]; top = button["zmax"]
    log("button at", np.round(button["xy"], 3), "top", round(top, 3))
    r.move_to([bx, by, top + 0.08], grip=1, vmax=0.05, tol=0.008)
    r.hold(6, grip=1)
    r.move_to([bx, by, top + 0.03], vmax=0.03, tol=0.005)
    z = top + 0.02
    while not r.complete and not r.ended and z > top - 0.03:
        z -= 0.005
        r.move_to([bx, by, z], vmax=0.012, tol=0.003, max_steps=10)
        log("press z", round(z, 3), "pos", np.round(r.pos, 3), "complete", r.complete, "ended", r.ended)
    if not r.ended:
        r.move_to([bx, by, top + 0.08], vmax=0.04, tol=0.01, max_steps=20)
    log("press done complete", r.complete, "ended", r.ended)
