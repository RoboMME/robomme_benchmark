"""ButtonUnmask controller v1.

Plan: press button early (before step 90), watch the cube reveals (front + wrist cameras,
every saved frame), remember each coloured cube's world xy, then after containers have
covered them, lift the containers hiding the instructed colours in order (place each back).
"""
import os, sys, re, json, math, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import lib
from lib import log

COLORS = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
HIGH = np.array([-0.06, 0.0, 0.40])   # observation pose (wrist sees table top-down)
OUT = "/workspace/output"


def parse_instruction(text):
    t = text.lower()
    # split after "press the button"
    idx = t.find("button")
    t2 = t[idx:] if idx >= 0 else t
    targets = []
    for m in re.finditer(r"(all containers hiding|the (?:container|one) hiding|containers hiding)\s+(?:the\s+|a\s+)?(\w+)\s+cube", t2):
        allf = m.group(1).startswith("all")
        col = m.group(2)
        if col in COLORS:
            targets.append((col, allf))
    if not targets:
        seen = []
        for w in re.findall(r"\w+", t2):
            if w in COLORS and w not in seen:
                seen.append(w)
        targets = [(c, False) for c in seen]
    return targets


def detect_cubes_in_frame(fmeta, files):
    """returns list of (color, xy, npix, cam)"""
    out = []
    for cam in ("front", "wrist"):
        try:
            rgb = lib.load_png(files[(cam, "rgb")])
            dep = lib.load_png(files[(cam, "depth")])
        except Exception:
            continue
        if cam == "wrist":
            # only trust the wrist when it is high above the table (top-down view)
            st = fmeta.get("state") or [0, 0, 0]
            if st[2] < 0.35:
                continue
        P = None
        for col, m in lib.color_masks(rgb).items():
            for comp in lib.components(m, 4 if cam == "front" else 6):
                if P is None:
                    P = lib.deproject_all(fmeta, cam, dep)
                pts = P[comp[:, 0], comp[:, 1]]
                pts = pts[np.isfinite(pts).all(-1)]
                if len(pts) == 0:
                    continue
                p = np.median(pts, 0)
                if not (-0.35 < p[0] < 0.4 and -0.45 < p[1] < 0.45 and -0.01 < p[2] < 0.08):
                    continue
                out.append((col, p[:2], len(comp), cam))
    return out


def cluster_cubes(dets, radius=0.035):
    """dets: list of (color, xy, npix, cam, step). -> dict color -> list of cluster dict"""
    res = {}
    for col in COLORS:
        ds = [d for d in dets if d[0] == col]
        cls = []
        for d in ds:
            for c in cls:
                if np.linalg.norm(c["xy"] - d[1]) < radius:
                    c["obs"].append(d)
                    c["xy"] = np.median(np.array([o[1] for o in c["obs"]]), 0)
                    break
            else:
                cls.append(dict(xy=np.array(d[1]), obs=[d]))
        single = [c for c in cls if len(c["obs"]) < 2]
        cls = [c for c in cls if len(c["obs"]) >= 2] or single
        for c in cls:
            # prefer wrist (top-down, unbiased) observations if present
            w = [o[1] for o in c["obs"] if o[3] == "wrist"]
            f = [o[1] for o in c["obs"] if o[3] == "front"]
            c["xy_w"] = np.median(np.array(w), 0) if w else None
            c["xy_f"] = np.median(np.array(f), 0) if f else None
            c["best"] = c["xy_w"] if c["xy_w"] is not None else (c["xy_f"] + np.array([-0.008, 0.0]))
        if cls:
            res[col] = cls
    return res


def containers_now(R, avoid_button=None, go_high=True):
    if go_high:
        R.move_to(HIGH, speed=1.0, tol=0.012, chunk=6)
        R.hold(2)
    meta, imgs = R.observe()
    views = []
    if ("front", "depth") in imgs:
        views.append((meta, "front", imgs[("front", "depth")]))
    if ("wrist", "depth") in imgs and meta["state"][2] > 0.3:
        views.append((meta, "wrist", imgs[("wrist", "depth")]))
    if not views:
        return []
    cls = lib.table_objects(meta, None, xlim=(-0.32, 0.40), ylim=(-0.45, 0.45), zmax=0.11, views=views)
    out = []
    for c in cls:
        if not (0.05 < c["zmax"] < 0.10):      # button / cube / tall pillar part
            continue
        if c["tall"] > 30:
            log("skip tall-adjacent cluster", c["center"].round(3), c["tall"])
            continue
        ctr, yaw, size, ntop = lib.top_center_yaw(c)
        if size.max() > 0.07 or size.min() < 0.02:
            log("skip odd-size cluster", ctr.round(3), size.round(3))
            continue
        if avoid_button is not None and np.linalg.norm(ctr - avoid_button) < 0.03:
            continue
        out.append(dict(xy=np.array(ctr), yaw=yaw, size=size, n=c["n"], ntop=ntop, zmax=c["zmax"]))
    return out


def grey_mask(rgb):
    a = rgb[..., :3].astype(int)
    return ((a.max(-1) - a.min(-1)) < 40) & (a.max(-1) > 70)


def find_button(R):
    meta, imgs = R.observe()
    views = []
    for cam in ("front", "wrist"):
        if (cam, "depth") in imgs and (cam, "rgb") in imgs:
            views.append((meta, cam, imgs[(cam, "depth")], grey_mask(imgs[(cam, "rgb")])))
    cls = lib.table_objects(meta, None, zmin=0.012, zmax=0.07, xlim=(-0.35, 0.40), ylim=(-0.45, 0.45), views=views)
    best = None
    for c in cls:
        ctr, yaw, size, ntop = lib.top_center_yaw(c, band=0.006)
        log("button cand", ctr.round(3), c["n"], round(c["zmax"], 3))
        if not (0.02 < c["zmax"] < 0.05):
            continue
        if best is None or c["n"] > best[1]:
            best = (ctr, c["n"], c["zmax"])
    if best is None:
        return np.array([-0.19, -0.016]), 0.034
    return best[0], best[2]


def press_button(R, bxy, btop):
    R.grip = 1.0
    R.move_to([bxy[0], bxy[1], btop + 0.07], speed=1.0, tol=0.006, chunk=6)
    log("above button", R.step_id, R.pos.round(3))
    # descend until stalled
    last = R.pos[2]
    stall = 0
    for i in range(14):
        if R.step_id >= 86 or R.ended:
            break
        z0 = R.pos[2]
        rate = -0.6 if z0 > btop + 0.03 else -0.3
        R.step([[0, 0, rate, 0, 0, 0, 1.0]] * 2)
        z = R.pos[2]
        if last - z < 0.004:
            stall += 1
        else:
            stall = 0
        last = z
        if stall >= 2 or z < btop - 0.025:
            break
    log("pressed", R.step_id, R.pos.round(4))
    R.step([[0, 0, -0.2, 0, 0, 0, 1.0]] * 2)
    R.step([[0, 0, 0.8, 0, 0, 0, 1.0]] * 3)


def collect(R, dets, upto):
    for sid, f, files in R.frames[collect.k:]:
        if sid is None:
            continue
        for d in detect_cubes_in_frame(f, files):
            dets.append(d + (sid,))
    collect.k = len(R.frames)
collect.k = 0


def match(cubes, conts):
    """assign each cube cluster to nearest container"""
    out = []
    for col, cls in cubes.items():
        for c in cls:
            if not conts:
                continue
            d = [np.linalg.norm(k["xy"] - c["best"]) for k in conts]
            j = int(np.argmin(d))
            out.append(dict(color=col, cube=c["best"], cont=j, dist=d[j]))
    return out


def make_plan(targets, cubes, conts):
    plan = []
    # containers not explained by any seen cube (fallback for unseen targets)
    def nearest(p):
        d = [np.linalg.norm(k["xy"] - p) for k in conts]
        return (int(np.argmin(d)), min(d)) if d else (None, 9)
    explained = set()
    for col2, cls2 in cubes.items():
        for c in cls2:
            j, dd = nearest(c["best"])
            if j is not None and dd < 0.03:
                explained.add(j)
    unexplained = [k for i, k in enumerate(conts) if i not in explained]
    log("unexplained containers", [k["xy"].round(3) for k in unexplained])
    for col, allf in targets:
        cls = cubes.get(col, [])
        if not cls:
            log("WARN no cube seen for", col)
            if unexplained and not allf:
                k = unexplained.pop(0)
                log("fallback: unexplained container for", col, k["xy"].round(3))
                plan.append((col, k["xy"]))
            continue
        if not allf:
            # single: choose the observation cluster with most evidence
            cls = sorted(cls, key=lambda c: -len(c["obs"]))[:1]
        for c in cls:
            plan.append((col, c["best"]))
    return plan


def run(ctx):
    R = lib.Robot(ctx)
    meta, imgs = R.observe()
    instr = meta.get("instruction", "")
    targets = parse_instruction(instr)
    log("instruction:", instr)
    log("targets:", targets)
    bxy, btop = find_button(R)
    log("button", bxy.round(3), round(btop, 3))
    dets = []
    # phase 1: watch batch-1 reveal from the high pose (wrist top-down + front)
    R.move_to(HIGH, speed=0.8, tol=0.01, chunk=6, grip=1.0)
    while R.step_id < 25 and not R.ended:
        R.hold(min(5, 25 - R.step_id), grip=1.0)
    # phase 2: press the button (deadline step 89)
    press_button(R, bxy, btop)
    # phase 3: back to the high pose and watch the remaining reveals
    R.move_to(HIGH, speed=1.0, tol=0.01, chunk=6)
    last_seen = 0
    while not R.ended and R.step_id < 260:
        R.hold(5, grip=1.0)
        collect(R, dets, None)
        if dets:
            last_seen = max(d[4] for d in dets)
        if R.step_id >= 140 and R.step_id > last_seen + 20:
            break
    log("observation ended at", R.step_id, "last cube seen", last_seen)
    cubes = cluster_cubes(dets)
    for col, cls in cubes.items():
        for c in cls:
            log("cube", col, c["best"].round(3), "nobs", len(c["obs"]),
                "w", None if c["xy_w"] is None else c["xy_w"].round(3),
                "f", None if c["xy_f"] is None else c["xy_f"].round(3))
    conts = containers_now(R, avoid_button=bxy)
    for i, k in enumerate(conts):
        log("container", i, k["xy"].round(3), k["yaw"], k["size"].round(3), k["ntop"])
    plan = make_plan(targets, cubes, conts)
    log("plan", [(c, p.round(3)) for c, p in plan])
    for col, p in plan:
        if conts:
            d = min(np.linalg.norm(k["xy"] - p) for k in conts)
            log("plan check", col, p.round(3), "nearest container dist", round(d, 3), "OK" if d < 0.03 else "UNCERTAIN")
    done_xy = []
    for col, cxy in plan:
        if R.ended:
            break
        ok = False
        for attempt in range(3):
            if R.ended:
                break
            conts = containers_now(R, avoid_button=bxy)
            if not conts:
                log("no containers"); break
            d = [np.linalg.norm(k["xy"] - cxy) for k in conts]
            j = int(np.argmin(d))
            k = conts[j]
            log("pick", col, "cube", cxy.round(3), "-> cont", k["xy"].round(3), "yaw", k["yaw"], "dist", round(d[j], 3))
            w = lib.pick_lift(R, k["xy"], k["yaw"])
            log("lifted width", round(w, 4), "step", R.step_id, "complete", R.complete, "ended", R.ended)
            if R.ended:
                ok = True; break
            if w > 0.01:
                ok = True
                lib.place_back(R, k["xy"])
                break
            R.grip = -1.0
            R.hold(3, grip=-1.0)
            R.move_to(R.pos + np.array([0, 0, 0.1]), speed=0.6)
    log("final step", R.step_id, "ended", R.ended, "complete", R.complete)
    return R


def main(ctx):
    t0 = time.time()
    try:
        R = run(ctx)
    except Exception:
        log("EXCEPTION", traceback.format_exc())
    try:
        os.makedirs(OUT, exist_ok=True)
        open(os.path.join(OUT, "controller_log.txt"), "w").write("\n".join(lib.LOG))
    except Exception:
        pass
    log("wall", round(time.time() - t0, 1))
