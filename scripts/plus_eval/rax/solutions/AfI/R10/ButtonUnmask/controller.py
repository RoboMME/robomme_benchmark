"""ButtonUnmask controller: remember cube colours before they are covered, press the button,
then pick the containers hiding the instructed colours in order."""
import os, sys, re, math, time, json, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from robot import Robot, decode_images, jtext
from percep import color_blobs, blob_world, backproject
from pick import pick_container, place_back, map_containers, free_spot, place_at

COLORS = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink"]
OUT = "/workspace/output"
HOVER_BTN_Z = 0.32     # wrist camera sees the whole table from here
PRESS_DONE_BY = 56     # press between set-1 masking (~45) and set-2 appearance (~57); deadline is step 90
WATCH_MIN_END = 100    # keep watching until at least this step ...
WATCH_QUIET = 12       # ... and until no coloured cube has been visible for this many steps
WATCH_MAX_END = 170

class Logger:
    def __init__(self):
        self.t0 = time.time()
        try:
            os.makedirs(OUT, exist_ok=True)
            self.f = open(os.path.join(OUT, "controller_log.txt"), "a")
        except Exception:
            self.f = None
    def __call__(self, *a):
        msg = "[%.1fs] " % (time.time() - self.t0) + " ".join(str(x) for x in a)
        print(msg, flush=True)
        if self.f:
            self.f.write(msg + "\n"); self.f.flush()

def parse_targets(instr):
    """Ordered list of (colour, all_flag)."""
    s = instr.lower()
    i = s.find("pick")
    if i >= 0:
        s = s[i:]
    found = []
    for m in re.finditer(r"\b(" + "|".join(COLORS) + r")\b(\s+cubes)?", s):
        pre = s[max(0, m.start() - 40):m.start()]
        allf = bool(m.group(2)) or ("all " in pre.split(",")[-1])
        found.append((m.group(1), allf))
    return found

class Tracker:
    """Records world positions of saturated coloured cubes seen by the front and wrist cameras."""
    def __init__(self, log):
        self.obs = {}          # colour -> list of (step, cam, xyz)
        self.log = log
        self.history = []      # (step, set of colours)
        self.exclude = []      # xy points (e.g. the button, which can be coloured) to ignore
    def __call__(self, res):
        meta, ims = decode_images(res)
        step = meta.get("step_id", -1)
        seen = set()
        for cam in ("wrist", "front"):
            if (cam, "rgb") not in ims or (cam, "depth") not in ims:
                continue
            rgb, dep = ims[(cam, "rgb")], ims[(cam, "depth")]
            K = meta["camera_intrinsics"][cam]["matrix"]; pose = meta["camera_poses"][cam]
            bl = color_blobs(rgb, min_area=8)
            for c, blobs in bl.items():
                for a, cx, cy, (xs, ys) in blobs[:6]:
                    w = blob_world(xs, ys, dep, K, pose)
                    if w is None or not (0.0 < w[2] < 0.07) or not (-0.45 < w[0] < 0.3) or abs(w[1]) > 0.45:
                        continue
                    if cam == "front":
                        w = w - np.array([0.008, 0.0, 0.0])  # front-face bias toward the camera
                    if any(np.hypot(w[0] - ex[0], w[1] - ex[1]) < 0.04 for ex in self.exclude):
                        continue
                    self.obs.setdefault(c, []).append((step, cam, w))
                    seen.add(c)
        self.history.append((step, seen))
    def clusters(self, c, rad=0.03):
        """Distinct cube locations of colour c: list of (count, xyz, n_wrist), most observed first.
        Wrist (top-down) observations are preferred for the position."""
        if c not in self.obs:
            return []
        cl = []
        for _, cam, p in self.obs[c]:
            for k in cl:
                if np.hypot(*(np.median(np.array([q for _, q in k]), axis=0)[:2] - p[:2])) < rad:
                    k.append((cam, p)); break
            else:
                cl.append([(cam, p)])
        out = []
        for k in cl:
            wr = [q for cam, q in k if cam == "wrist"]
            pts = wr if wr else [q for _, q in k]
            out.append((len(k), np.median(np.array(pts), axis=0), len(wr)))
        out.sort(key=lambda t: -t[0])
        return out
    def estimate(self, c):
        cl = self.clusters(c)
        return cl[0][1] if cl else None
    def visible_now(self):
        return self.history[-1][1] if self.history else set()

def find_button(meta, ims, log, cam="wrist", near=None):
    d = ims[(cam, "depth")]
    K = meta["camera_intrinsics"][cam]["matrix"]; pose = meta["camera_poses"][cam]
    from percep import quat_to_R
    H, W = d.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    fx, cx, fy, cy = K[0][0], K[0][2], K[1][1], K[1][2]
    pc = np.stack([(us - cx) * d / fx, -(vs - cy) * d / fy, -d], -1).reshape(-1, 3)
    P = pc @ quat_to_R(pose["quaternion_wxyz"]).T + np.array(pose["position_m"])
    P = P[d.reshape(-1) > 0]
    m = (P[:, 2] > 0.012) & (P[:, 2] < 0.06) & (P[:, 0] > -0.45) & (P[:, 0] < 0.3) & (np.abs(P[:, 1]) < 0.45)
    if near is not None:
        m &= np.hypot(P[:, 0] - near[0], P[:, 1] - near[1]) < 0.035
    pts = P[m]
    log("button candidate points", cam, len(pts))
    if len(pts) < 10:
        return None
    hi = pts[pts[:, 2] > np.percentile(pts[:, 2], 70)]  # the cap
    c = np.median(hi, axis=0)
    log("button (", cam, ") at", np.round(c, 4), "cap z", round(float(np.max(pts[:, 2])), 4))
    return c

def main(ctx):
    log = Logger()
    R = Robot(ctx, log=log)
    tr = Tracker(log)
    try:
        run(ctx, R, tr, log)
    except Exception as e:
        log("FATAL", repr(e), traceback.format_exc())
    log("done step", R.step_id, "ended", R.ended, "complete", R.complete)

def run(ctx, R, tr, log):
    res = ctx.call("robomme_observe")
    meta, ims = decode_images(res)
    R.last = meta
    R.step_id = meta.get("step_id", 0)
    instr = meta.get("instruction", "")
    targets = parse_targets(instr)
    log("instruction:", instr, "targets:", targets)
    tr(res)
    btn = None
    try:
        btn = find_button(meta, ims, log, "wrist")
    except Exception as e:
        log("wrist button error", repr(e))
    if btn is None:
        btn = find_button(meta, ims, log, "front")
        if btn is not None:
            btn = btn - np.array([0.015, 0.0, 0.0])
    if btn is None:
        btn = np.array([-0.19, -0.015, 0.03])
        log("button not found, default", btn)
    R.on_frame = tr
    bx, by = float(btn[0]), float(btn[1])
    tr.exclude = [(bx, by)]
    tr.obs = {}
    # Phase A: hover high above the button; wrist camera sees the whole table (front view may be occluded)
    R.move_to([bx, by, HOVER_BTN_Z], grip=1, tol=0.01, max_steps=40)
    log("hovering above button at step", R.step_id)
    # refine button centre from a closer top-down view
    try:
        m2, im2 = decode_images(ctx.call("robomme_observe"))
        b2 = find_button(m2, im2, log, "wrist", near=(bx, by))
        if b2 is not None and np.hypot(b2[0] - bx, b2[1] - by) < 0.04:
            bx, by = float(b2[0]), float(b2[1])
            tr.exclude.append((bx, by))
    except Exception as e:
        log("refine button error", repr(e))
    descent_steps = int((HOVER_BTN_Z - 0.03) / 0.02) + 6
    start_press = PRESS_DONE_BY - descent_steps
    while R.step_id < start_press and not R.ended:
        R.step([[0, 0, 0, 0, 0, 0, 1]] * 2)
    log("watch phase end at step", R.step_id, "colours so far", sorted(tr.obs.keys()))
    # Phase B: press (front camera keeps tracking during the descent)
    R.move_to([bx, by, 0.05], grip=1, tol=0.006, max_steps=30)
    prev = R.state()[2]; stall = 0
    for i in range(20):
        R.step([[0, 0, -1.0, 0, 0, 0, 1]])
        z = R.state()[2]
        if prev - z < 0.0015:
            stall += 1
        else:
            stall = 0
        prev = z
        if stall >= 2 or z < 0.012:
            break
    log("press bottom z %.4f at step %d" % (R.state()[2], R.step_id))
    for i in range(3):
        R.step([[0, 0, -1.0, 0, 0, 0, 1]])
        log("press push z %.4f step %d" % (R.state()[2], R.step_id))
    press_step = R.step_id
    R.move_to([bx, by, HOVER_BTN_Z], grip=1, tol=0.01, max_steps=30)
    last_seen = R.step_id
    while not R.ended:
        if tr.visible_now():
            last_seen = R.step_id
        if R.step_id >= WATCH_MAX_END:
            break
        if R.step_id >= WATCH_MIN_END and R.step_id - last_seen >= WATCH_QUIET:
            break
        R.step([[0, 0, 0, 0, 0, 0, 1]] * 2)
    log("set2 phase end at step", R.step_id, "colours", sorted(tr.obs.keys()))
    for c in tr.obs:
        log("colour", c, "n", len(tr.obs[c]), "clusters", [(n, list(np.round(p, 3)), nw) for n, p, nw in tr.clusters(c)], "steps", tr.obs[c][0][0], "-", tr.obs[c][-1][0])
    try:
        with open(os.path.join(OUT, "tracker.json"), "w") as f:
            json.dump({c: [(s, cam, list(map(float, p))) for s, cam, p in v] for c, v in tr.obs.items()}, f)
    except Exception:
        pass
    R.on_frame = None
    s0 = R.state()
    R.move_to([s0[0], s0[1], 0.42], grip=-1, tol=0.015, max_steps=20)
    R.hold(2, grip=-1)
    occupied = []
    try:
        occupied = map_containers(ctx, log=log)
    except Exception as e:
        log("map error", repr(e))
    # Phase D: picks
    plan = []
    for c, allf in targets:
        cl = tr.clusters(c)
        if not cl:
            log("target", c, "never seen; skipping")
            continue
        # drop spurious clusters (seen far less than the best one)
        cl = [k for k in cl if k[0] >= 2] or cl[:1]
        if allf:
            plan += [(c, k[1]) for k in cl]
        else:
            plan.append((c, cl[0][1]))
    log("plan", [(c, list(np.round(p[:2], 3))) for c, p in plan])
    for i, (c, est) in enumerate(plan):
        if R.ended:
            break
        xy = (float(est[0]), float(est[1]))
        log("pick", i, c, "est", np.round(xy, 4), "step", R.step_id)
        held, cxy = pick_container(R, xy, log=log)
        if not held:
            log("grasp failed; retry once")
            R.move_to([cxy[0], cxy[1], 0.16], grip=-1, tol=0.01, max_steps=30)
            held, cxy = pick_container(R, cxy, log=log)
        if R.ended:
            break
        occupied.append(cxy)
        if i < len(plan) - 1:
            spot = free_spot(occupied, cxy, (bx, by), log=log)
            if spot is None:
                log("no free spot; placing back")
                place_back(R, cxy, log=log)
            else:
                place_at(R, spot, log=log)
                occupied.append(spot)
        else:
            R.hold(10, grip=1)
    for _ in range(5):
        if R.ended:
            break
        R.hold(5, grip=R.grip)
