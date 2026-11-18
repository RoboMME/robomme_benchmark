"""StopCube controller v3.

Hover a closed gripper just above the button cap, track the cube in the front
camera (RGB-D back-projection), count its arrivals at the target (distance
minima below R_ARR) and start the press so it lands on the predicted closest
approach of the k-th arrival.
"""
import json, os, re, sys, time, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rax import parse  # noqa: E402
from percep import height_map, nontable_mask  # noqa: E402
from scipy import ndimage  # noqa: E402

CFG = dict(
    hover_gap=0.012,      # fingertip gap above cap at hover (m)
    finger_off=0.0065,    # fingertip bottom below TCP z
    press_latency=2.0,    # steps from descent command to registered press
    R_ARR=0.035,           # arrival distance threshold (m)
    safe_z=0.15,
)
ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
       "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
       "fourteenth": 14, "fifteenth": 15, "once": 1, "twice": 2, "thrice": 3}

OUT = "/workspace/output"
LOG = []


def log(msg, **kw):
    try:
        rec = dict(kw)
        rec["wall"] = round(time.time(), 3)
        rec["msg"] = msg
        LOG.append(rec)
        print(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), flush=True)
    except Exception:
        pass


def parse_k(instr):
    s = instr.lower()
    for w, k in ORD.items():
        if re.search(r"\b" + w + r"\b", s):
            return k
    m = re.search(r"(\d+)\s*(st|nd|rd|th)?\s*time", s)
    if m:
        return int(m.group(1))
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else 1


def sat(rgb):
    im = rgb.astype(int)
    return im.max(-1) - im.min(-1)


def find_button(P, rgb, tcp=None):
    z = P[..., 2]
    nt = nontable_mask(rgb)
    m = nt & (z > 0.024) & (z < 0.045) & (sat(rgb) < 45)
    m[:28] = False
    if tcp is not None and tcp[2] < 0.3:
        m &= np.hypot(P[..., 0] - tcp[0], P[..., 1] - tcp[1]) > 0.045
    lab, n = ndimage.label(m)
    best = None
    for k in range(1, n + 1):
        s = lab == k
        c = int(s.sum())
        if c < 6:
            continue
        if best is None or c > best[0]:
            best = (c, s)
    if best is None:
        return None
    s = best[1]
    zt = np.where(s, z, np.nan)
    top = s & (z > np.nanmax(zt) - 0.004)
    xy = np.array([np.nanmean(P[..., 0][top]), np.nanmean(P[..., 1][top])])
    ztop = float(np.nanmedian(z[top]))
    return dict(xy=xy, ztop=ztop, n=best[0])


def find_target(Pmin, rgb_list, button_xy, need_sat=True):
    z = Pmin[..., 2]
    m = (z > 0.008) & (z < 0.022)
    for rgb in rgb_list:
        m &= nontable_mask(rgb)
    if need_sat:
        m &= sat(rgb_list[0]) > 35
    m[:28] = False
    if button_xy is not None:
        d = np.hypot(Pmin[..., 0] - button_xy[0], Pmin[..., 1] - button_xy[1])
        m &= d > 0.07
    lab, n = ndimage.label(m)
    best = None
    for k in range(1, n + 1):
        s = lab == k
        c = int(s.sum())
        if c < 6:
            continue
        if best is None or c > best[0]:
            best = (c, s)
    if best is None:
        if need_sat:
            return find_target(Pmin, rgb_list, button_xy, need_sat=False)
        return None
    s = best[1]
    return dict(xy=np.array([np.nanmean(Pmin[..., 0][s]), np.nanmean(Pmin[..., 1][s])]),
                z=float(np.nanmedian(z[s])), n=best[0], sat=need_sat)


def _largest(m, minpx=4):
    lab, n = ndimage.label(m)
    best = None
    for k in range(1, n + 1):
        s = lab == k
        c = int(s.sum())
        if c < minpx:
            continue
        if best is None or c > best[0]:
            best = (c, s)
    return best


def find_cube(P, rgb, button_xy, tcp):
    z = P[..., 2]
    m = (z > 0.022) & (z < 0.06)
    m[:28] = False
    if button_xy is not None:
        m &= np.hypot(P[..., 0] - button_xy[0], P[..., 1] - button_xy[1]) > 0.06
    m &= np.hypot(P[..., 0] - tcp[0], P[..., 1] - tcp[1]) > 0.05
    best = _largest(m & nontable_mask(rgb))
    if best is None:
        best = _largest(m)
    if best is None:
        return None
    s = best[1]
    return np.array([float(np.nanmedian(P[..., 0][s])), float(np.nanmedian(P[..., 1][s]))])


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = parse(ctx.call("robomme_observe"))
        self.ended = False
        self.complete = None

    def step(self, row):
        row = [float(np.clip(v, -1, 1)) for v in row]
        self.meta = parse(self.ctx.call("robomme_step", actions=[row]))
        tr = self.meta.get("transition") or {}
        for s in tr.get("steps", []) or []:
            if s.get("episode_ended"):
                self.ended = True
                self.complete = s.get("task_complete")
        if self.meta.get("episode_ended"):
            self.ended = True
        return self.meta

    @property
    def tcp(self):
        return np.array(self.meta["state"][:3])

    @property
    def step_id(self):
        return int(self.meta.get("step_id", 0))

    @property
    def remaining(self):
        lim = self.meta.get("episode_limits") or {}
        return int(lim.get("remaining_episode_steps", 999))


def seg_closest(p0, p1, q):
    """Fraction f in [0,1] of the point on p0->p1 closest to q, and that distance."""
    d = p1 - p0
    L2 = float(d @ d)
    f = 0.0 if L2 < 1e-12 else float(np.clip((q - p0) @ d / L2, 0.0, 1.0))
    return f, float(np.linalg.norm(p0 + f * d - q))


class Tracker:
    def __init__(self, target_xy):
        self.target = target_xy
        self.hist = []  # (step, xy or None, dist)
        self.arrivals = []   # integer step of the sampled minimum
        self.arr_t = []      # refined continuous arrival time
        self.left = True  # cube outside the arrival zone since last arrival

    def _refine(self, i):
        """Continuous closest-approach time around hist index i."""
        best = (float(self.hist[i][0]), self.hist[i][2])
        for j in (i - 1, i):
            if j < 0 or j + 1 >= len(self.hist):
                continue
            (sa, pa, _), (sb, pb, _) = self.hist[j], self.hist[j + 1]
            if pa is None or pb is None or sb - sa != 1:
                continue
            f, d = seg_closest(pa, pb, self.target)
            if d < best[1]:
                best = (sa + f, d)
        return best[0]

    def update(self, step, xy):
        d = None if xy is None else float(np.hypot(*(xy - self.target)))
        self.hist.append((step, xy, d))
        idx = [i for i in range(max(0, len(self.hist) - 4), len(self.hist)) if self.hist[i][1] is not None]
        if len(idx) >= 3:
            i0, i1, i2 = idx[-3:]
            d0, d1, d2 = self.hist[i0][2], self.hist[i1][2], self.hist[i2][2]
            if d1 < CFG["R_ARR"] and d1 <= d0 and d1 < d2 and self.left:
                self.arrivals.append(self.hist[i1][0])
                self.arr_t.append(self._refine(i1))
                self.left = False
                log("arrival", n=len(self.arrivals), step=self.hist[i1][0], tarr=self.arr_t[-1], dist=d1)
        if d is not None and d > CFG["R_ARR"] * 1.3:
            self.left = True
        return d

    def predict_template(self):
        """Match the current position/velocity to an earlier lap; return steps to the next arrival."""
        if len(self.hist) < 2 or not self.arr_t:
            return None, None
        (st, pt, _), (sp, pp, _) = self.hist[-1], self.hist[-2]
        if pt is None or pp is None or st - sp != 1:
            return None, None
        v = pt - pp
        last_arr = self.arr_t[-1]
        best = None
        for j in range(len(self.hist) - 1):
            (sa, pa, _), (sb, pb, _) = self.hist[j], self.hist[j + 1]
            if pa is None or pb is None or sb - sa != 1 or sb > last_arr:
                continue
            nxt = [a for a in self.arr_t if a > sa + 1e-6]
            if not nxt:
                continue
            f, dpos = seg_closest(pa, pb, pt)
            dv = float(np.linalg.norm((pb - pa) - v))
            cost = dpos + 0.5 * dv
            if best is None or cost < best[0]:
                best = (cost, nxt[0] - (sa + f), dpos, dv)
        if best is None or best[2] > 0.015 or best[3] > 0.012:
            return None, best
        return best[1], best

    def predict(self):
        """Quadratic extrapolation: steps from the latest sample to the closest approach, and that distance."""
        pts = [(s, xy) for s, xy, _ in self.hist[-6:] if xy is not None]
        if len(pts) < 3:
            return None, None
        pts = pts[-5:]
        s_last = pts[-1][0]
        if s_last - pts[0][0] > 7:
            return None, None
        T = np.array([s - s_last for s, _ in pts], float)
        Y = np.array([xy for _, xy in pts], float)
        deg = 2 if len(pts) >= 4 else 1
        A = np.stack([T ** j for j in range(deg + 1)], 1)
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        ss = np.arange(-1.0, 8.01, 0.05)
        B = np.stack([ss ** j for j in range(deg + 1)], 1)
        pred = B @ coef
        dd = np.hypot(pred[:, 0] - self.target[0], pred[:, 1] - self.target[1])
        i = int(np.argmin(dd))
        return float(ss[i]), float(dd[i])


def move_to(rob, goal, trk, ctxinfo, max_steps=60, tol=0.004, gain=0.8, maxa=1.0):
    for _ in range(max_steps):
        if rob.ended:
            return
        err = goal - rob.tcp
        if np.linalg.norm(err) < tol:
            return
        a = np.clip(err / 0.05 * gain, -maxa, maxa)
        rob.step([a[0], a[1], a[2], 0, 0, 0, 1.0])
        observe_cube(rob, trk, ctxinfo)


def observe_cube(rob, trk, info):
    try:
        P = height_map(rob.meta)
        xy = find_cube(P, rob.meta["_imgs"][("front", "rgb")], info.get("button_xy"), rob.tcp)
    except Exception as e:
        log("cube_err", err=repr(e))
        xy = None
    if trk is None:
        info.setdefault("pre", []).append((rob.step_id, xy))
        return None
    d = trk.update(rob.step_id, xy)
    return d


def decide(trk, rob, k, n, pressed):
    """Return True when the press should start at this observation."""
    Lat = CFG["press_latency"]
    if pressed:
        return False
    if n >= k:
        # missed the predicted window: press at once while the count is still k
        log("trigger_late", step=rob.step_id, n=n)
        return True
    if n != k - 1 or not trk.left:
        return False
    s_tpl, info_tpl = trk.predict_template()
    s_q, dmin = trk.predict()
    vals = [h[2] for h in trk.hist[-3:] if h[2] is not None]
    approaching = len(vals) >= 2 and vals[-1] < vals[-2]
    if s_tpl is not None:
        if Lat - 0.5 < s_tpl <= Lat + 0.5 or (0.8 < s_tpl <= Lat - 0.5 and approaching):
            log("trigger", how="template", step=rob.step_id, s_tpl=s_tpl, s_q=s_q, dmin=dmin, n=n,
                d=vals[-1] if vals else None, match=info_tpl)
            return True
    elif (s_q is not None and dmin is not None and dmin < 0.045 and approaching
            and 0.8 < s_q <= Lat + 0.5):
        log("trigger", how="quadratic", step=rob.step_id, s_q=s_q, dmin=dmin, n=n, d=vals[-1],
            match=info_tpl)
        return True
    # fallback when the cube is not visible: periodic prediction from past arrivals
    real = [a for a in trk.arr_t if a >= 0]
    nvis = sum(1 for h in trk.hist[-3:] if h[1] is None)
    if nvis >= 2 and len(real) >= 2:
        pred = real[-1] + (real[-1] - real[-2])
        if rob.step_id >= pred - Lat - 0.5:
            log("trigger_fallback", step=rob.step_id, pred=pred)
            return True
    return False


def main(ctx):
    t0 = time.time()
    rob = Robot(ctx)
    instr = rob.meta.get("instruction", "")
    k = parse_k(instr)
    log("start", instr=instr, k=k, state=rob.meta["state"])
    info = {}
    P0 = height_map(rob.meta)
    rgb0 = rob.meta["_imgs"][("front", "rgb")]
    btn = find_button(P0, rgb0, rob.tcp)
    log("button", btn=None if btn is None else dict(xy=btn["xy"].tolist(), ztop=btn["ztop"], n=btn["n"]))
    if btn is None:
        btn = dict(xy=np.array([-0.14, 0.25]), ztop=0.034, n=0)
    if not (0.02 < btn["ztop"] < 0.045):
        log("button_ztop_suspicious", ztop=btn["ztop"])
        btn["ztop"] = 0.034
    info["button_xy"] = btn["xy"]
    # target from static min-height map over the first frames while rising/moving
    Ps, rgbs = [P0], [rgb0]
    trk = None
    hover = np.array([btn["xy"][0], btn["xy"][1], max(CFG["safe_z"], 0)])
    # first steps: move toward the safe point above the button, collecting frames
    for i in range(6):
        err = hover - rob.tcp
        a = np.clip(err / 0.05 * 0.8, -1, 1)
        rob.step([a[0], a[1], a[2], 0, 0, 0, 1.0])
        Ps.append(height_map(rob.meta)); rgbs.append(rob.meta["_imgs"][("front", "rgb")])
        observe_cube(rob, None, info)
    Pmin = Ps[0].copy()
    zst = np.stack([p[..., 2] for p in Ps])
    imin = np.nanargmin(np.where(np.isnan(zst), np.inf, zst), axis=0)
    Pmin = np.take_along_axis(np.stack(Ps), imin[None, ..., None].repeat(3, -1), 0)[0]
    tgt = find_target(Pmin, rgbs, btn["xy"])
    log("target", tgt=None if tgt is None else dict(xy=tgt["xy"].tolist(), z=tgt["z"], n=tgt["n"]))
    if tgt is None:
        tgt = dict(xy=np.array([0.05, -0.15]), z=0.016, n=0)
    trk = Tracker(tgt["xy"])
    trk.arrivals = [-1] * int(os.environ.get("STOPCUBE_PRIOR_ARRIVALS", "0"))
    trk.arr_t = [-1.0] * len(trk.arrivals)
    for s, xy in info.get("pre", []):
        trk.update(s, xy)
    # straight to a point just above the hover height, then a gentle final approach
    zh = btn["ztop"] + CFG["finger_off"] + CFG["hover_gap"]
    move_to(rob, np.array([hover[0], hover[1], max(zh + 0.015, 0.10)]), trk, info, max_steps=80, tol=0.01, gain=1.0)
    move_to(rob, np.array([hover[0], hover[1], zh + 0.015]), trk, info, max_steps=30, tol=0.004, gain=1.0)
    move_to(rob, np.array([hover[0], hover[1], zh]), trk, info, max_steps=30, tol=0.0015, gain=0.6, maxa=0.4)
    hover_pt = np.array([hover[0], hover[1], zh])
    log("hovering", step=rob.step_id, tcp=rob.tcp.tolist(), zh=zh, arrivals=len(trk.arrivals))
    # wait / track
    pressed = False
    while not rob.ended and rob.remaining > 2:
        n = len(trk.arrivals)
        trigger = False
        try:
            trigger = decide(trk, rob, k, n, pressed)
        except Exception as e:
            log("decide_err", err=repr(e), tb=traceback.format_exc()[-800:])
            trigger = False
        if trigger:
            pressed = True
            for j in range(8):
                rob.step([0, 0, -1.0, 0, 0, 0, 1.0])
                observe_cube(rob, trk, info)
                log("press_step", step=rob.step_id, tcp=rob.tcp.tolist(), ended=rob.ended, complete=rob.complete)
                if rob.ended:
                    break
            if not rob.ended:
                # retreat back to hover and keep trying
                move_to(rob, hover_pt + np.array([0, 0, 0.01]), trk, info, max_steps=10, tol=0.003)
            continue
        # hold hover (small correction)
        err = hover_pt - rob.tcp
        a = np.clip(err / 0.05 * 0.5, -0.3, 0.3)
        rob.step([a[0], a[1], a[2], 0, 0, 0, 1.0])
        observe_cube(rob, trk, info)
    log("end", step=rob.step_id, ended=rob.ended, complete=rob.complete, arrivals=trk.arrivals,
        pressed=pressed, wall=time.time() - t0)
    try:
        os.makedirs(OUT, exist_ok=True)
        hist = [(s, None if xy is None else [float(xy[0]), float(xy[1])], d) for s, xy, d in trk.hist]
        json.dump(dict(log=LOG, hist=hist), open(os.path.join(OUT, "stopcube_log.json"), "w"), default=float)
    except Exception:
        pass
