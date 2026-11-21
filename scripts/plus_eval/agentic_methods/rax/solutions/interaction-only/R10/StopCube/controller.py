"""StopCube controller v1.

Count the cube's arrivals on the target (centre distance < ENTER_D) from execution step 0,
hover the closed gripper just above the button, and press so the press registers when the
cube is predicted closest to the target on the N-th arrival.
"""
import json, os, re, sys, time, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlib import parse, quat_to_R

ENTER_D = 0.030      # cube-target xy distance that counts as "reached"
EXIT_D = 0.045       # hysteresis to end a pass
HOVER_DZ = 0.008     # TCP height above button top while waiting
HIGH_DZ = 0.035      # approach point height above button top
ACQ_STEPS = 4       # zero-motion steps at start for static-scene acquisition
PRESS_LAT = 1        # steps from press command to registration (measured ep1: ~1 from top+0.008)
OUT = "/workspace/output"

ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
       "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "once": 1, "twice": 2}


def log(msg):
    print(msg, flush=True)
    try:
        with open(os.path.join(OUT, "controller.log"), "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def parse_count(instr):
    s = instr.lower()
    for w, n in ORD.items():
        if re.search(r"\b" + w + r"\b", s):
            return n
    m = re.search(r"(\d+)(st|nd|rd|th)?\s*time", s)
    if m:
        return int(m.group(1))
    return 1


def cloud(meta, depth, cam="front"):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    d = depth.astype(float) * 0.001
    pc = np.stack([(u - K[0, 2]) * d / K[0, 0], -(v - K[1, 2]) * d / K[1, 1], -d], -1)
    R = quat_to_R(pose["quaternion_wxyz"])
    P = pc @ R.T + np.array(pose["position_m"])
    P[d <= 0] = np.nan
    return P


def masks(rgb):
    a = rgb.astype(int)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    cube = (B > 110) & (R - B > 15) & (R - B < 90) & (G - B > 10) & (R > 150) & (np.abs(R - G) < 40)
    target = (B > R + 10) & (B > G + 60)
    gray = (np.abs(R - G) < 12) & (np.abs(G - B) < 12) & (R > 60) & (R < 235)
    return cube, target, gray


class Tracker:
    def __init__(self, target_xy, n_goal):
        self.t = np.asarray(target_xy)
        self.n_goal = n_goal
        self.passes = 0
        self.inside = False
        self.hist = []  # (step, xy or None)
        self.min_d_pass = 9.0
        self.min_step = None
        self.t_min = []  # step of closest approach of each completed pass

    def period(self):
        if len(self.t_min) < 2:
            return None
        return float(np.median(np.diff(self.t_min)))

    def update(self, step, xy):
        self.hist.append((step, xy))
        if xy is None:
            self._virtual(step)
            return
        d = float(np.linalg.norm(xy - self.t))
        if not self.inside and d < ENTER_D:
            self.inside = True
            self.passes += 1
            self.min_d_pass = d
            self.min_step = step
            log(f"PASS {self.passes} enter step={step} d={d:.3f}")
        elif self.inside:
            if d < self.min_d_pass:
                self.min_d_pass = d
                self.min_step = step
            if d > EXIT_D:
                self.inside = False
                self.t_min.append(self.min_step)
                log(f"PASS {self.passes} exit step={step} min_d={self.min_d_pass:.3f} at {self.min_step} period={self.period()}")
        else:
            self._virtual(step)

    def _virtual(self, step):
        # cube hidden through a whole expected pass: count it from the period
        T = self.period()
        if T is None or self.inside or not self.t_min:
            return
        exp = self.t_min[-1] + T
        if step < exp + 0.5 * T:
            return
        wide = [p for s, p in self.hist if exp - 0.4 * T <= s <= exp + 0.4 * T and p is not None]
        hit = any(np.linalg.norm(p - self.t) < ENTER_D for p in wide)
        hidden = [p for s, p in self.hist if exp - 3 <= s <= exp + 3 and p is None]
        if not hit and len(hidden) >= 2:
            self.passes += 1
            self.t_min.append(int(round(exp)))
            log(f"VIRTUAL PASS {self.passes} at {exp:.1f} (cube hidden)")

    def n_completed(self):
        return len(self.t_min)

    def unseen_recent(self, n=2):
        recent = self.hist[-n:]
        return len(recent) == n and all(p is None for s, p in recent)

    def predict(self, k):
        """Predicted xy k steps after the last observation (constant-accel fit on last 3)."""
        pts = [(s, p) for s, p in self.hist[-4:] if p is not None]
        if len(pts) < 2:
            return None
        (s1, p1), (s2, p2) = pts[-2], pts[-1]
        if s2 == s1:
            return p2
        v = (p2 - p1) / (s2 - s1)
        acc = np.zeros(2)
        if len(pts) >= 3:
            s0, p0 = pts[-3]
            if s1 != s0:
                v0 = (p1 - p0) / (s1 - s0)
                acc = (v - v0) / ((s2 - s0) / 2.0)
        return p2 + v * k + 0.5 * acc * k * k

    def upcoming_index(self):
        return self.passes if self.inside else self.passes + 1


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = None
        self.ended = False
        self.complete = None
        self.on_obs = None

    def observe(self):
        meta, imgs = parse(self.ctx.call("robomme_observe"))
        self.meta = meta
        return meta, imgs

    def step(self, row):
        row = [float(np.clip(x, -1, 1)) for x in row]
        meta, imgs = parse(self.ctx.call("robomme_step", actions=[row]))
        self.meta = meta
        tr = meta.get("transition") or {}
        if meta.get("episode_ended") or tr.get("episode_ended"):
            self.ended = True
            self.complete = tr.get("task_complete")
            for s in tr.get("steps", []) or []:
                if s.get("task_complete"):
                    self.complete = True
            log(f"EPISODE ENDED step={meta.get('step_id')} task_complete={self.complete}")
        return meta, imgs

    def pos(self):
        return np.array(self.meta["state"][:3])

    def remaining(self):
        lim = self.meta.get("episode_limits") or {}
        return lim.get("remaining_episode_steps", 999)


def locate_cube(meta, imgs, statics=(), prev=None):
    """Colour-agnostic: points 2.2-7 cm above the table, away from static objects/robot; top-face centroid."""
    P = cloud(meta, imgs[("front", "depth")])
    z = P[..., 2]
    m = (z > 0.022) & (z < 0.07) & (P[..., 0] > -0.45)
    for sxy in statics:
        m &= np.hypot(P[..., 0] - sxy[0], P[..., 1] - sxy[1]) > 0.05
    pts = P[m]
    if len(pts) < 8:
        return None
    if prev is not None:
        near = np.linalg.norm(pts[:, :2] - prev[:2], axis=1) < 0.08
        if near.sum() >= 8:
            pts = pts[near]
    med = np.median(pts, 0)
    pts = pts[np.linalg.norm(pts[:, :2] - med[:2], axis=1) < 0.04]
    if len(pts) < 8:
        return None
    top = np.percentile(pts[:, 2], 95)
    tp = pts[pts[:, 2] > top - 0.005]
    if len(tp) < 4:
        tp = pts
    c = tp.mean(0)
    return c


def cluster_xy(pts, radius=0.04):
    """Greedy xy clustering; returns list of point arrays, largest first."""
    pts = pts[~np.isnan(pts).any(1)]
    out = []
    rest = pts
    while len(rest) > 0 and len(out) < 6:
        # seed at the densest point (most neighbours)
        sub = rest[:: max(1, len(rest) // 300)]
        dd = np.linalg.norm(sub[:, None, :2] - sub[None, :, :2], axis=2)
        seed = sub[np.argmax((dd < radius).sum(1)), :2]
        for _ in range(3):
            near = np.linalg.norm(rest[:, :2] - seed, axis=1) < radius
            seed = np.median(rest[near, :2], 0)
        near = np.linalg.norm(rest[:, :2] - seed, axis=1) < radius
        out.append(rest[near])
        rest = rest[~near]
    out.sort(key=len, reverse=True)
    return out


def acquire_scene(metas, imgs_list):
    """Static low disk = target, static tall cluster = button (colour agnostic)."""
    Ps = [cloud(m, im[("front", "depth")]) for m, im in zip(metas, imgs_list)]
    n = len(Ps)
    tall = [(P[..., 2] > 0.022) & (P[..., 2] < 0.07) & (P[..., 0] > -0.45) for P in Ps]
    low = [(P[..., 2] > 0.005) & (P[..., 2] < 0.021) & (P[..., 0] > -0.45) for P in Ps]
    static_tall = np.logical_and.reduce(tall)
    P0 = Ps[0]
    rgb0 = imgs_list[0][("front", "rgb")].astype(float)
    mx, mn = rgb0.max(-1), rgb0.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1.0)
    P4 = np.concatenate([P0, sat[..., None]], -1)
    bcl = [c for c in cluster_xy(P4[static_tall], 0.05) if len(c) >= 10]
    if not bcl:
        raise RuntimeError("button not found")
    # the button is grey (low saturation); prefer it over any slow/static coloured object
    bcl.sort(key=lambda c: len(c) * (1.0 if np.median(c[:, 3]) < 0.2 else 0.25), reverse=True)
    g = bcl[0][:, :3]
    top = float(np.percentile(g[:, 2], 90))
    cap = g[g[:, 2] > top - 0.006]
    b_xy = np.median(cap[:, :2], 0) if len(cap) >= 5 else np.median(g[:, :2], 0)
    button = np.array([b_xy[0], b_xy[1], top])
    statics = [np.median(c[:, :2], 0) for c in bcl]
    lowcount = np.sum(low, 0)
    lm = lowcount >= max(1, n - 1)
    # median position over frames where pixel is low
    stack = np.stack(Ps, 0)
    pts = stack[0][lm]
    keep = np.ones(len(pts), bool)
    for sxy in statics:
        keep &= np.linalg.norm(pts[:, :2] - sxy, axis=1) > 0.055
    pts = pts[keep]
    tcl = cluster_xy(pts, 0.04)
    if not tcl or len(tcl[0]) < 15:
        raise RuntimeError("target not found")
    t = tcl[0]
    # disk centre: centre of xy extent (robust to partial occlusion) blended with median
    lo, hi = np.percentile(t[:, :2], 3, 0), np.percentile(t[:, :2], 97, 0)
    t_xy = 0.5 * (0.5 * (lo + hi) + np.median(t[:, :2], 0))
    info = {"target_px": int(len(t)), "target_extent": (hi - lo).round(3).tolist(),
            "button_px": int(len(g)), "n_static": len(statics),
            "target_clusters": [int(len(c)) for c in tcl]}
    return t_xy, button, statics, info


def locate_scene(meta, imgs):
    rgb = imgs[("front", "rgb")]
    P = cloud(meta, imgs[("front", "depth")])
    cube, target, gray = masks(rgb)
    tm = target & ~np.isnan(P[..., 2]) & (P[..., 2] < 0.06)
    if tm.sum() < 10:
        raise RuntimeError("target not found")
    tpts = P[tm]
    t_xy = np.median(tpts[:, :2], 0)
    gm = gray & (P[..., 2] > 0.012) & (P[..., 2] < 0.08) & (P[..., 0] > -0.45)
    gpts = P[gm]
    gpts = gpts[~np.isnan(gpts).any(1)]
    gpts = gpts[np.linalg.norm(gpts[:, :2] - t_xy, axis=1) > 0.05]
    if len(gpts) < 10:
        raise RuntimeError("button not found")
    med = np.median(gpts[:, :2], 0)
    gpts = gpts[np.linalg.norm(gpts[:, :2] - med, axis=1) < 0.06]
    top = np.percentile(gpts[:, 2], 90)
    # cap: the highest points
    cap = gpts[gpts[:, 2] > top - 0.006]
    b_xy = np.median(cap[:, :2], 0) if len(cap) >= 5 else np.median(gpts[:, :2], 0)
    return t_xy, np.array([b_xy[0], b_xy[1], float(top)]), len(tpts), len(gpts)


def main(ctx):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rb = Robot(ctx)
    meta, imgs = rb.observe()
    instr = meta.get("instruction", "")
    n_goal = parse_count(instr)
    log(f"instruction={instr!r} n_goal={n_goal} step={meta.get('step_id')}")
    # hold still a few steps: static low disk = target, static tall = button, moving tall = cube
    metas, frames = [meta], [imgs]
    for _ in range(ACQ_STEPS):
        m2, i2 = rb.step([0, 0, 0, 0, 0, 0, 1])
        metas.append(m2); frames.append(i2)
        if rb.ended:
            break
    try:
        t_xy, button, statics, info = acquire_scene(metas, frames)
    except Exception as e:
        log("acquire_scene failed: " + repr(e) + "; colour fallback")
        try:
            t_xy, button, nt, nb = locate_scene(meta, imgs)
            statics, info = [button[:2]], {"fallback": True}
        except Exception as e2:
            log("fallback failed too: " + repr(e2))
            return
    log(f"target_xy={np.round(t_xy,3).tolist()} button={np.round(button,3).tolist()} statics={len(statics)} info={info}")
    tr = Tracker(t_xy, n_goal)
    last_cube = [None]
    for m2, i2 in zip(metas, frames):
        c = locate_cube(m2, i2, statics, last_cube[0])
        if c is not None:
            last_cube[0] = c
        tr.update(m2.get("step_id", 0), None if c is None else c[:2])
        if m2 is meta and c is not None and np.linalg.norm(c[:2] - t_xy) < ENTER_D:
            log("cube starts inside target radius (counted as pass 1)")
    high = button + np.array([0, 0, HIGH_DZ])
    hover = button + np.array([0, 0, HOVER_DZ])
    phase = "transit"
    pressed_at = None
    trace = []

    def act(row):
        meta, imgs = rb.step(row)
        c = locate_cube(meta, imgs, statics, last_cube[0])
        if c is not None:
            last_cube[0] = c
        xy = None if c is None else c[:2]
        tr.update(meta.get("step_id"), xy)
        trace.append({"step": meta.get("step_id"), "ee": [round(x, 4) for x in meta["state"][:3]],
                      "cube": None if xy is None else [round(float(x), 4) for x in xy],
                      "passes": tr.passes, "inside": tr.inside, "phase": phase})
        return meta

    def servo_row(goal, vmax):
        err = goal - rb.pos()
        r = np.clip(err / 0.05, -vmax, vmax)
        return list(r) + [0, 0, 0, 1], float(np.linalg.norm(err))

    try:
        while not rb.ended and rb.remaining() > 0:
            p = rb.pos()
            if phase == "transit":
                # straight line from the (high) start to just above the button
                goal = high.copy()
                if p[2] < button[2] + 0.025 and np.linalg.norm(p[:2] - high[:2]) > 0.03:
                    goal = np.array([p[0], p[1], high[2]])  # too low far from the button: lift first
                row, e = servo_row(goal, 1.0 if np.linalg.norm(goal - p) > 0.08 else 0.4)
                if np.linalg.norm(high - p) < 0.008:
                    phase = "descend"
                    log(f"at approach point step={rb.meta.get('step_id')}")
                act(row)
            elif phase == "descend":
                row, e = servo_row(hover, 0.6 if np.linalg.norm(hover - p) > 0.03 else 0.25)
                if np.linalg.norm(hover - p) < 0.003:
                    phase = "wait"
                    log(f"hovering step={rb.meta.get('step_id')} ee={np.round(p,4).tolist()}")
                act(row)
            elif phase == "wait":
                k = tr.upcoming_index()
                p1, p2 = tr.predict(PRESS_LAT), tr.predict(PRESS_LAT + 1)
                fire = False
                if k == n_goal and p1 is not None:
                    d1 = float(np.linalg.norm(p1 - t_xy)); d2 = float(np.linalg.norm(p2 - t_xy))
                    if d1 < ENTER_D * 0.9 and d1 <= d2:
                        fire = True
                        log(f"FIRE step={rb.meta.get('step_id')} pass={k} pred_d1={d1:.4f} pred_d2={d2:.4f}")
                T = tr.period()
                if not fire and k == n_goal and tr.unseen_recent(2) and T is not None:
                    tk = tr.t_min[-1] + T * (k - tr.n_completed())
                    if rb.meta.get("step_id", 0) + PRESS_LAT >= tk - 0.5:
                        fire = True
                        log(f"FIRE(period) step={rb.meta.get('step_id')} pass={k} t_pred={tk:.1f} T={T:.1f}")
                if k > n_goal:
                    log("missed the goal pass; pressing on next arrival is wrong too -> waiting is pointless; press now")
                    fire = True
                if fire:
                    phase = "press"
                    pressed_at = rb.meta.get("step_id")
                    act([0, 0, -1, 0, 0, 0, 1])
                else:
                    row, e = servo_row(hover, 0.25)
                    act(row)
            elif phase == "press":
                act([0, 0, -1, 0, 0, 0, 1])
                if rb.meta.get("step_id", 0) - pressed_at > 8:
                    log("press did not end episode after 8 steps; retry hover+press")
                    phase = "descend"
            if time.time() - t0 > 1000:
                log("wall time guard")
                break
    except Exception as e:
        log("ERROR " + repr(e) + "\n" + traceback.format_exc())
    finally:
        try:
            with open(os.path.join(OUT, "trace.json"), "w") as f:
                json.dump({"target": t_xy.tolist(), "button": button.tolist(), "n_goal": n_goal,
                           "pressed_at": pressed_at, "complete": rb.complete, "trace": trace}, f)
        except Exception:
            pass
    log(f"done ended={rb.ended} complete={rb.complete} passes={tr.passes} wall={time.time()-t0:.1f}s")
