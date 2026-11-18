"""StopCube controller v1.

Observation-driven: segment target / button / cube from front RGB-D, track the cube
every control step, count arrivals at the target (local minima of cube-target
distance below a threshold), hover just above the button, and press so the press
registers at the predicted closest approach of the Nth arrival.
"""
import json, os, re, sys, time, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from perception import decode, backproject, hsv  # noqa: E402
from scipy import ndimage  # noqa: E402

OUT = "/workspace/output"
CUBE_ZMIN, CUBE_ZMAX = 0.022, 0.075
ROBOT_BASE_X = -0.45          # points with x below this belong to the robot base
BUTTON_EXCL_R = 0.075         # exclude points this close to the button when tracking cube
ARRIVE_IN = 0.030             # cube-target distance that opens an arrival window
ARRIVE_OUT = 0.040            # distance that closes it
HOVER_DZ = 0.0075             # hover TCP height above measured button top
APPROACH_DZ = 0.05
NUMWORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
            "once": 1, "twice": 2}


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    try:
        with open(os.path.join(OUT, "controller_log.txt"), "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def parse_count(instr):
    s = instr.lower()
    for w, n in NUMWORDS.items():
        if re.search(r"\b" + w + r"\b", s):
            return n
    m = re.search(r"(\d+)(st|nd|rd|th)?\s+time", s)
    if m:
        return int(m.group(1))
    return 1


class Scene:
    def __init__(self):
        self.target = None
        self.button = None
        self.button_top = None


def components(mask, minpix=5):
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) >= minpix:
            out.append((ys, xs))
    return out


def find_target(pw, rgb):
    z = pw[..., 2]
    h, s, v = hsv(rgb)
    flat = (z > 0.009) & (z < 0.021) & (pw[..., 0] > ROBOT_BASE_X)
    tablelike = (h < 45) & (s > 0.3)
    best = None
    for ys, xs in components(flat & ~tablelike, 15):
        sat = ((s[ys, xs] > 0.25) & ~((h[ys, xs] < 45))).sum()
        if best is None or sat > best[0]:
            best = (sat, ys, xs)
    if best is None or best[0] < 10:
        return None, 0
    _, ys, xs = best
    P = pw[ys, xs]
    return np.median(P[:, :2], 0), len(xs)


def high_components(pw):
    z = pw[..., 2]
    mk = (z > CUBE_ZMIN) & (z < CUBE_ZMAX) & np.isfinite(z) & (pw[..., 0] > ROBOT_BASE_X)
    out = []
    for ys, xs in components(mk, 6):
        P = pw[ys, xs]
        out.append(dict(xy=np.median(P[:, :2], 0), top=float(np.percentile(P[:, 2], 90)), n=len(xs)))
    return out


def cube_xy(pw, scene, prev):
    z = pw[..., 2]
    mk = (z > CUBE_ZMIN) & (z < CUBE_ZMAX) & np.isfinite(z) & (pw[..., 0] > ROBOT_BASE_X)
    P = pw[mk]
    if len(P) == 0:
        return None
    if scene.button is not None:
        d = np.hypot(P[:, 0] - scene.button[0], P[:, 1] - scene.button[1])
        P = P[d > BUTTON_EXCL_R]
    if scene.target is not None:
        d = np.hypot(P[:, 0] - scene.target[0], P[:, 1] - scene.target[1])
        P = P[d < 0.30]
    if len(P) < 4:
        return None
    ref = prev if prev is not None else np.median(P[:, :2], 0)
    for _ in range(2):
        d = np.hypot(P[:, 0] - ref[0], P[:, 1] - ref[1])
        Q = P[d < 0.045]
        if len(Q) < 4:
            if prev is not None:
                # lost near previous: fall back to global median cluster
                ref = np.median(P[:, :2], 0)
                continue
            return None
        ref = np.median(Q[:, :2], 0)
    return ref


class Tracker:
    def __init__(self, scene, n_needed):
        self.scene = scene
        self.N = n_needed
        self.t, self.p = [], []          # step ids and cube xy (None if lost)
        self.count = 0
        self.arrivals = []
        self.in_window = False
        self.ignore_window = False
        self.seen_valid = False
        self.win_min = (1e9, None)

    def dist(self, p):
        return float(np.hypot(*(p - self.scene.target)))

    def update(self, step, p):
        self.t.append(step)
        self.p.append(p)
        if p is None:
            return
        d = self.dist(p)
        first = not self.seen_valid
        self.seen_valid = True
        if not self.in_window and d < ARRIVE_IN:
            self.in_window = True
            self.ignore_window = first   # cube started inside the target zone: not an arrival
            self.win_min = (d, step)
        elif self.in_window:
            if d < self.win_min[0]:
                self.win_min = (d, step)
            if d > ARRIVE_OUT:
                self.in_window = False
                if self.ignore_window:
                    log("ignored initial window, min dist %.3f at %s" % self.win_min)
                else:
                    self.count += 1
                    self.arrivals.append(self.win_min[1])
                    log("ARRIVAL", self.count, "at step", self.win_min[1], "min dist %.3f" % self.win_min[0])

    def velocity(self):
        # last two valid consecutive samples
        if len(self.p) < 2 or self.p[-1] is None or self.p[-2] is None:
            return None
        dt = self.t[-1] - self.t[-2]
        if dt <= 0:
            return None
        return (self.p[-1] - self.p[-2]) / dt

    def time_to_closest(self):
        """Predicted steps until closest approach, constant velocity extrapolation."""
        v = self.velocity()
        if v is None or self.p[-1] is None:
            return None, None
        r = self.p[-1] - self.scene.target
        vv = float(v @ v)
        if vv < 1e-8:
            return None, None
        tau = -float(r @ v) / vv
        miss = float(np.hypot(*(r + tau * v)))
        return tau, miss


def step(ctx, row, scene, tracker, state_box, save=None):
    reply = ctx.call("robomme_step", actions=[[float(x) for x in row]])
    meta, im = decode(reply)
    state_box["meta"] = meta
    state_box["state"] = np.array(meta["state"], dtype=float)
    pw = backproject(meta, im[("front", "depth")])
    state_box["pw"] = pw
    state_box["rgb"] = im[("front", "rgb")]
    prev = None
    for q in reversed(tracker.p[-3:]):
        if q is not None:
            prev = q
            break
    p = cube_xy(pw, scene, prev)
    tracker.update(meta["step_id"], p)
    ended = bool(meta.get("episode_ended"))
    if ended:
        log("EPISODE ENDED at step", meta["step_id"], json.dumps(meta.get("transition", {}).get("steps", []))[:400])
    return ended


def main(ctx):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    try:
        run(ctx)
    except Exception:
        log("CONTROLLER EXCEPTION", traceback.format_exc())
    log("controller finished in %.1fs" % (time.time() - t0))


def run(ctx):
    meta, im = decode(ctx.call("robomme_observe"))
    instr = meta.get("instruction", "")
    N = parse_count(instr)
    log("instruction:", instr, "-> N =", N, "step", meta["step_id"], "limits", meta.get("episode_limits"))
    if meta.get("episode_ended"):
        log("episode already ended")
        return
    scene = Scene()
    pw = backproject(meta, im[("front", "depth")])
    rgb = im[("front", "rgb")]
    tgt, n = find_target(pw, rgb)
    log("target", tgt, n)
    comps0 = high_components(pw)
    tracker = Tracker(scene, N)
    box = {"state": np.array(meta["state"], float), "pw": pw, "rgb": rgb, "meta": meta}
    # provisional target needed for tracking; take it now
    scene.target = tgt if tgt is not None else np.array([0.05, -0.15])

    # --- identify static button blob vs. moving cube over 3 zero-action steps
    hold = [0, 0, 0, 0, 0, 0, 1]
    init_frames = [(meta["step_id"], pw)]
    for _ in range(3):
        if step(ctx, hold, scene, tracker, box):
            return
        init_frames.append((box["meta"]["step_id"], box["pw"]))
    comps3 = high_components(box["pw"])
    tg2, n2 = find_target(box["pw"], box["rgb"])
    if tg2 is not None and (tgt is None or n2 > n):
        scene.target = tg2
    best = None
    for c0 in comps0:
        for c3 in comps3:
            mv = float(np.hypot(*(c0["xy"] - c3["xy"])))
            dt = float(np.hypot(*(c3["xy"] - scene.target)))
            if mv < 0.004 and dt > 0.04:
                score = c3["n"]
                if best is None or score > best[0]:
                    best = (score, c3)
    if best is None:
        # fallback: largest high component farthest from target
        cands = [c for c in comps3 if np.hypot(*(c["xy"] - scene.target)) > 0.04]
        if not cands:
            log("no button found; abort")
            return
        best = (0, max(cands, key=lambda c: c["n"]))
    scene.button = best[1]["xy"]
    scene.button_top = best[1]["top"]
    log("button", scene.button, "top", scene.button_top, "target", scene.target)
    # re-run tracking of the first frames now that the button is known is unnecessary:
    # the cube was within search radius; reset tracker positions to avoid button confusion
    tracker = Tracker(scene, N)
    prev = None
    for sid, fpw in init_frames:
        q = cube_xy(fpw, scene, prev)
        tracker.update(sid, q)
        prev = q if q is not None else prev
        log("init cube", sid, None if q is None else np.round(q, 3).tolist())

    bx, by = scene.button
    top = scene.button_top
    hover = np.array([bx, by, top + HOVER_DZ])
    above = np.array([bx, by, top + APPROACH_DZ])

    def move_cmd(goal, vmax):
        st = box["state"][:3]
        d = (goal - st) / 0.05
        # command scaled up for the ~40% per-step tracking, limited
        a = np.clip(d * 1.0, -vmax, vmax)
        return [a[0], a[1], a[2], 0, 0, 0, 1]

    phase = "approach"
    limit = meta.get("episode_limits", {}).get("episode_step_limit", 578)
    pressed = False
    while True:
        st = box["state"]
        stepid = box["meta"]["step_id"]
        if stepid >= limit - 1:
            log("step limit reached")
            return
        pos = st[:3]
        # --- press decision (any phase, but only if we are low enough)
        if tracker.count == N - 1 and tracker.p[-1] is not None:
            tau, miss = tracker.time_to_closest()
            d = tracker.dist(tracker.p[-1])
            aligned = np.hypot(pos[0] - bx, pos[1] - by) < 0.012 and pos[2] < top + 0.09
            # steps needed for the press to register: ~2 cm of travel per full-speed step
            lat = max(1, int(np.ceil((pos[2] - (top + 0.003)) / 0.018)))
            if tau is not None and aligned and miss < ARRIVE_IN and d < 0.15 and -0.5 <= tau < lat + 0.5:
                log("PRESS at step", stepid, "tau %.2f lat %d miss %.3f dist %.3f z %.4f" % (tau, lat, miss, d, pos[2]))
                for k in range(lat + 4):
                    if step(ctx, [0, 0, -1, 0, 0, 0, 1], scene, tracker, box):
                        return
                    log("press step", k, "z %.4f" % box["state"][2])
                # not triggered: keep pushing gently, log
                pressed = True
        if tracker.count >= N and not pressed:
            log("WARNING: missed the Nth arrival (count %d); pressing on next arrival" % tracker.count)
            N = tracker.count + 1
        if pressed:
            # retreat to hover and wait (count may be off); try next arrival
            pressed = False
            N = tracker.count + 1
            phase = "descend"
        if phase == "approach":
            # move above the button at high speed
            row = move_cmd(above, 1.0)
            if np.linalg.norm(above - pos) < 0.01:
                phase = "descend"
        elif phase == "descend":
            row = move_cmd(hover, 0.6)
            # never command below the hover height
            if pos[2] + row[2] * 0.05 < hover[2] - 0.0005:
                row[2] = (hover[2] - pos[2]) / 0.05
            if abs(pos[2] - hover[2]) < 0.0015 and np.hypot(pos[0] - bx, pos[1] - by) < 0.004:
                phase = "hover"
                log("HOVER reached at step", stepid, "z %.4f" % pos[2])
        else:
            row = move_cmd(hover, 0.1)
            if pos[2] + row[2] * 0.05 < hover[2] - 0.0005:
                row[2] = (hover[2] - pos[2]) / 0.05
        row = [float(np.clip(x, -1, 1)) for x in row]
        if step(ctx, row, scene, tracker, box):
            return
        p = tracker.p[-1]
        if box["meta"]["step_id"] % 10 == 0 or (p is not None and tracker.dist(p) < 0.06):
            log("s", box["meta"]["step_id"], phase, "ee", np.round(box["state"][:3], 4).tolist(),
                "cube", None if p is None else np.round(p, 3).tolist(),
                "d", None if p is None else round(tracker.dist(p), 3), "count", tracker.count)
