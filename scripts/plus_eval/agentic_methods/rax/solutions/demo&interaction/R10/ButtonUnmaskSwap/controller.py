"""ButtonUnmaskSwap controller: remember cube colours, track the swapping containers
in front depth every control step, press both buttons, then lift the containers in order."""
import os, re, sys, time, json, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlib import backproject
from tracker import detect_cubes, top_mask, init_tracker, init_tracker_union
from robot import Robot, pick, place_back, ee_yaw

OUT = "/workspace/output"

def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    try:
        with open(os.path.join(OUT, "controller_log.txt"), "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass

def low_blobs(meta, depth, min_px=60):
    from scipy import ndimage
    P = backproject(meta, depth); z = P[..., 2]
    m = (z > 0.012) & (z < 0.048) & (P[..., 0] > -0.40) & (P[..., 0] < 0.3) & (np.abs(P[..., 1]) < 0.4)
    lab, n = ndimage.label(m); out = []
    for i in range(1, n + 1):
        sel = lab == i
        if sel.sum() >= min_px:
            out.append((P[sel][:, :2].mean(0), int(sel.sum())))
    return out

COLS = {"red": (200, 3, 3), "green": (4, 193, 4), "blue": (3, 3, 191)}

def revealed_ok(R, c, col):
    """True unless a clearly visible cube of another colour sits under the lifted container."""
    try:
        rgb, dep = R.front()
        P = backproject(R.meta, dep)
        sat = rgb.max(-1).astype(int) - rgb.min(-1).astype(int)
        m = (P[..., 2] > 0.008) & (P[..., 2] < 0.045) & (np.linalg.norm(P[..., :2] - c, axis=-1) < 0.025) & (sat > 80)
        n = int(m.sum())
        if n < 12 or col not in COLS:
            return True, f"unknown (n={n})"
        mc = rgb[m].astype(float).mean(0)
        d = float(np.linalg.norm(mc - np.array(COLS[col])))
        return d < 90, f"n={n} color={mc.round(0).tolist()} dist={d:.0f}"
    except Exception as e:
        return True, f"err {e}"

def parse_order(instr):
    cols = re.findall(r"hiding the (\w+) cube", instr)
    return cols or ["blue", "green", "red"]

def press(R, b, log=log):
    R.move_to([b[0], b[1], 0.10], grip=1, tol=0.012, vmax=1.0, max_steps=40)
    R.move_to([b[0], b[1], 0.012], grip=1, tol=0.004, vmax=0.5, max_steps=15)
    log(f"pressed {np.round(b,3)} at step {R.step_id} ee {R.ee.round(3)}")
    R.hold(2, grip=1)
    R.move_to([b[0], b[1], 0.10], grip=1, tol=0.015, vmax=1.0, max_steps=30)

def main(ctx):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    R = Robot(ctx, log=log)
    instr = R.meta.get("instruction", "")
    order = parse_order(instr)
    log("instruction:", instr, "order:", order, "step", R.step_id)
    rgb0, dep0 = R.front(); meta0 = R.meta
    cubes_f = detect_cubes(meta0, rgb0, dep0)
    cubes_w = {}
    try:
        cubes_w = detect_cubes(meta0, R.imgs[("wrist", "rgb")], R.imgs[("wrist", "depth")], "wrist")
    except Exception as e:
        log("wrist cube detect failed", e)
    cubes = {**cubes_w, **cubes_f}
    log("cubes front", {k: v.round(3).tolist() for k, v in cubes_f.items()}, "wrist", {k: v.round(3).tolist() for k, v in cubes_w.items()})
    blobs0 = low_blobs(meta0, dep0)
    log("cubes", {k: v.round(3).tolist() for k, v in cubes.items()}, "low blobs", [(c.round(3).tolist(), n) for c, n in blobs0])
    # wait for containers to spawn
    for _ in range(40):
        R.step([0, 0, 0, 0, 0, 0, 1])
        P, m = top_mask(R.meta, R.imgs[("front", "depth")])
        if m.sum() > 300:
            break
    R.tracker = init_tracker(R.meta, R.imgs[("front", "depth")], cubes)
    R.use_wrist = False
    T = R.tracker
    log("spawn step", R.step_id, "tracks", len(T.p), T.labels, T.p.round(3).tolist())
    # buttons: low blobs at step 0 not under a container
    btns = [c for c, n in blobs0 if min(np.linalg.norm(c - p) for p in T.p) > 0.035 and n >= 100]
    if len(btns) < 2:
        log("button fallback")
        btns = [c for c, n in sorted(blobs0, key=lambda cn: cn[0][0])[:2]]
    btns = sorted(btns, key=lambda c: -c[1])[:2] if len(btns) > 2 else btns
    btns = sorted(btns, key=lambda c: np.linalg.norm(c - R.ee[:2]))
    log("buttons", [b.round(3).tolist() for b in btns])
    for b in btns:
        press(R, b)
        if R.ended: break
    log("presses done step", R.step_id)
    R.move_to([-0.05, 0.0, 0.35], grip=1, tol=0.015, vmax=1.0, max_steps=30)
    still = 0
    while not R.ended and R.step_id < 520:
        prev = T.p.copy(); R.step([0, 0, 0, 0, 0, 0, 1])
        mv = np.linalg.norm(T.p - prev, axis=1).max()
        still = still + 1 if mv < 0.0006 else 0
        if still >= 15: break
    log("motion stopped step", R.step_id, "miss", T.miss.tolist(), "events", T.events)
    log("tracks", T.labels, T.p.round(3).tolist())
    np.save(os.path.join(OUT, "track_hist.npy"), np.array(T.hist))
    R.tracker = None
    targets = {l: T.p[i].copy() for i, l in enumerate(T.labels) if l}
    others = [T.p[i].copy() for i, l in enumerate(T.labels) if not l]
    for col in order:
        if R.ended: break
        if col not in targets:
            log("no track for", col)
            if not others: continue
            targets[col] = others.pop(0)
        cands = [targets[col]] + sorted(others, key=lambda q: np.linalg.norm(q - targets[col]))[:2]
        for ci, xy in enumerate(cands):
            if R.ended: break
            ok, c = pick(R, xy, log=log)
            if not ok and not R.ended:
                log("grasp failed, retry"); R.hold(4, grip=-1)
                ok, c = pick(R, c, log=log)
            if R.ended: break
            seen = detect_cubes(R.meta, *R.front())
            log(f"after lifting for {col} (cand {ci}): seen {[(k, v.round(3).tolist()) for k, v in seen.items()]}")
            good, info = revealed_ok(R, c, col)
            log(f"reveal check {col}: good={good} {info}")
            if R.complete: break
            place_back(R, c)
            if good:
                break
            log(f"{col} not revealed under candidate {ci}; trying next")
            others = [q for q in others if np.linalg.norm(q - xy) > 1e-6]
    log("done step", R.step_id, "ended", R.ended, "complete", R.complete, "wall", round(time.time() - t0, 1))
