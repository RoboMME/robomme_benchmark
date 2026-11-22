"""StopCube controller: track cube vs target with front RGB-D, hover over button, press on N-th arrival."""
import json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib, percep

ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
       "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "once": 1, "twice": 2}
CFG = dict(R_IN=0.03, R_OUT=0.045, LAT=1, HOVER_DZ=0.008, PRESS_DZ=-0.035, BTN_EXCL=0.06,
           ROBOT_X=-0.45, VMAX=0.02)

def parse_n(instr):
    s = instr.lower()
    m = re.search(r"for the (\w+) time", s)
    if m and m.group(1) in ORD: return ORD[m.group(1)]
    for w, n in ORD.items():
        if re.search(r"\b%s\b" % w, s): return n
    m = re.search(r"(\d+)\s*(st|nd|rd|th)?\s*time", s)
    if m: return int(m.group(1))
    return 1

class Log:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a")
    def __call__(self, **kw):
        kw["t"] = time.time()
        self.f.write(json.dumps(kw, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        self.f.flush()

class Robot:
    def __init__(self, ctx, log):
        self.ctx, self.log = ctx, log
        self.meta, self.imgs = rlib.decode(ctx.call("robomme_observe"))
        self.ended = self.meta.get("episode_ended", False)
        self.complete = None
    @property
    def pos(self): return np.array(self.meta["state"][:3])
    @property
    def step_id(self): return self.meta["step_id"]
    @property
    def remaining(self): return self.meta["episode_limits"]["remaining_episode_steps"]
    def step(self, row):
        if self.ended: return False
        row = [float(np.clip(x, -1, 1)) for x in row]
        self.meta, self.imgs = rlib.decode(self.ctx.call("robomme_step", actions=[row]))
        tr = self.meta.get("transition") or {}
        for s in tr.get("steps", []):
            if s.get("episode_ended"):
                self.ended = True; self.complete = s.get("task_complete")
        if tr.get("episode_ended") or self.meta.get("episode_ended"):
            self.ended = True
            if tr.get("task_complete") is not None: self.complete = tr.get("task_complete")
        return not self.ended
    def move_row(self, target, gain=1.0, grip=1.0):
        d = (np.asarray(target) - self.pos) / 0.05 * gain
        return [d[0], d[1], d[2], 0, 0, 0, grip]

class Tracker:
    def __init__(self, log):
        self.log = log
        self.button = None; self.button_z = None
        self.cube_hist = []  # (step, xy)
        self.target_samples = []
        self.target = None
        self.inside = False; self.count = 0; self.events = []
        self.dist_hist = []
    def blobs(self, P):
        return [b for b in percep.high_blobs(P) if b["xy"][0] > CFG["ROBOT_X"]]
    def identify(self, frames):
        """frames: list of (step, blobs). Button = blob static since the first frame, cube = blob that moved."""
        first = frames[0][1]; last = frames[-1][1]
        if len(first) < 2 or len(last) < 2: return False
        def disp(b, ref):
            return min([np.hypot(*(b["xy"] - c["xy"])) for c in ref] + [1.0])
        static = [b for b in last if disp(b, first) < 0.004]
        moving = [b for b in last if disp(b, first) > 0.008]
        if not static or not moving: return False
        btn = max(static, key=lambda b: b["n"])
        cube = max(moving, key=lambda b: b["n"])
        if np.hypot(*(cube["xy"] - btn["xy"])) < 0.03: return False
        self.button = btn["xy"]; self.button_z = btn["z"]
        self.cube_hist.append((frames[-1][0], cube["xy"]))
        return True
    def update_target(self, P):
        z = P[..., 2]
        mask = np.isfinite(z) & (z > 0.009) & (z < 0.021)
        xy = P[..., :2]
        mask &= xy[..., 0] > CFG["ROBOT_X"]
        mask &= np.hypot(xy[..., 0] - self.button[0], xy[..., 1] - self.button[1]) > 0.06
        if self.cube_hist:
            c = self.cube_hist[-1][1]
            mask &= np.hypot(xy[..., 0] - c[0], xy[..., 1] - c[1]) > 0.045
        comps = percep.components(mask, 30)
        if not comps: return
        ys, xs = max(comps, key=lambda c: len(c[0]))
        self.target_samples.append((len(xs), np.median(P[ys, xs, :2], 0)))
        n = np.array([s[0] for s in self.target_samples]); pts = np.array([s[1] for s in self.target_samples])
        big = n >= 0.8 * n.max()
        self.target = np.median(pts[big], 0)
    def update_cube(self, step, P):
        bl = [b for b in self.blobs(P) if np.hypot(*(b["xy"] - self.button)) > CFG["BTN_EXCL"]]
        if not bl: return None
        prev = self.cube_hist[-1][1] if self.cube_hist else None
        if prev is not None:
            b = min(bl, key=lambda b: np.hypot(*(b["xy"] - prev)))
            if np.hypot(*(b["xy"] - prev)) > 0.03 * max(1, step - self.cube_hist[-1][0]) + 0.02: return None
        else:
            b = max(bl, key=lambda b: b["n"])
        self.cube_hist.append((step, b["xy"]))
        return b["xy"]
    def update_count(self, step):
        """Recount arrivals over the whole cube history with the current target estimate (hysteresis R_IN/R_OUT)."""
        if self.target is None or not self.cube_hist: return None
        inside = False; count = 0; events = []
        for s, c in self.cube_hist:
            d = float(np.hypot(*(c - self.target)))
            if not inside and d < CFG["R_IN"]:
                inside = True; count += 1; events.append(s)
            elif inside and d > CFG["R_OUT"]:
                inside = False
        self.inside, self.count, self.events = inside, count, events
        self.dist_hist.append((self.cube_hist[-1][0], d))
        return d
    def predict(self, horizon=8):
        """Quadratic extrapolation of the cube's xy over the next steps; returns list of (j, dist)."""
        h = self.cube_hist[-5:]
        if len(h) < 3: return []
        t = np.array([x[0] for x in h], float); P = np.array([x[1] for x in h])
        t0 = t[-1]; deg = 2 if len(h) >= 4 else 1
        cx = np.polyfit(t - t0, P[:, 0], deg); cy = np.polyfit(t - t0, P[:, 1], deg)
        out = []
        for j in range(0, horizon + 1):
            p = np.array([np.polyval(cx, j), np.polyval(cy, j)])
            out.append((j, float(np.hypot(*(p - self.target)))))
        return out

def decide(count, inside, d, prev_d, pred, lat, N):
    """Return a reason string to press now, else None. pred[j] = predicted cube-target distance j steps ahead."""
    if not pred or d is None: return None
    approaching = (count == N - 1 and not inside) or (count == N and inside)
    if not approaching: return None
    pl = pred[min(lat, len(pred) - 1)][1]; pl1 = pred[min(lat + 1, len(pred) - 1)][1]
    if pl < CFG["R_IN"] and pl <= pl1: return "predicted"
    if count == N and inside and prev_d is not None and d > prev_d: return "late"
    return None

def main(ctx):
    outdir = "/workspace/output"
    log = Log(os.path.join(outdir, "stopcube_log.jsonl"))
    rob = Robot(ctx, log)
    N = parse_n(rob.meta.get("instruction", ""))
    log(ev="start", instr=rob.meta.get("instruction"), N=N, cfg=CFG, limits=rob.meta.get("episode_limits"))
    print("N =", N, rob.meta.get("instruction"))
    trk = Tracker(log)
    home = rob.pos.copy()
    # Phase 1: identify button (static) vs cube (moving) with zero actions
    frames = [(rob.step_id, trk.blobs(rlib.world_map(rob.meta, rob.imgs)))]
    pre_cube = []  # cube positions before identification (for counting)
    for k in range(12):
        if not rob.step([0, 0, 0, 0, 0, 0, 1.0]): break
        frames.append((rob.step_id, trk.blobs(rlib.world_map(rob.meta, rob.imgs))))
        if k >= 1 and trk.identify(frames): break
    if trk.button is None:
        # fallback: assume the largest blob farthest from others... use first-frame blob with max n change? pick blob least moved
        log(ev="identify_fail", frames=[(s, [b["xy"] for b in bl]) for s, bl in frames])
        first = frames[0][1]; bl = frames[-1][1]
        if len(bl) >= 2 and first:
            dd = [min(np.hypot(*(b["xy"] - c["xy"])) for c in first) for b in bl]
            order = np.argsort(dd)
            trk.button = bl[order[0]]["xy"]; trk.button_z = bl[order[0]]["z"]
            trk.cube_hist.append((frames[-1][0], bl[order[-1]]["xy"]))
        else:
            print("cannot identify button"); return
    # back-fill cube history for earlier frames (counting from step 0)
    seed = trk.cube_hist[-1][1]; trk.cube_hist = []
    back = []
    prev = seed
    for s, bl in reversed(frames):
        cand = [b for b in bl if np.hypot(*(b["xy"] - trk.button)) > 0.03]
        if cand:
            b = min(cand, key=lambda b: np.hypot(*(b["xy"] - prev)))
            back.append((s, b["xy"])); prev = b["xy"]
    trk.cube_hist = list(reversed(back))
    log(ev="identified", button=trk.button, button_z=trk.button_z, cube=trk.cube_hist[-1])
    print("button", trk.button, trk.button_z, "cube", trk.cube_hist[-1])
    P = rlib.world_map(rob.meta, rob.imgs)
    trk.update_target(P)
    trk.update_count(rob.step_id)
    run_loop(rob, trk, N, log)

def run_loop(rob, trk, N, log):
    hover = np.array([trk.button[0], trk.button[1], trk.button_z + CFG["HOVER_DZ"]])
    fired = None; press_steps = 0; prev_d = None
    while not rob.ended:
        if rob.remaining <= 1: break
        if fired is None:
            row = rob.move_row(hover, gain=1.0)
        else:
            tgt = np.array([trk.button[0], trk.button[1], trk.button_z + CFG["PRESS_DZ"]])
            row = rob.move_row(tgt, gain=1.0)
            row[2] = -1.0
            press_steps += 1
        if not rob.step(row): break
        P = rlib.world_map(rob.meta, rob.imgs)
        c = trk.update_cube(rob.step_id, P)
        if len(trk.target_samples) < 25 or trk.target is None:
            trk.update_target(P)
        d = trk.update_count(rob.step_id)
        pred = trk.predict(horizon=10) if trk.target is not None else []
        log(ev="step", step=rob.step_id, pos=rob.pos, cube=c, d=d, count=trk.count, inside=trk.inside,
            pred=pred[:6], fired=fired)
        # steps needed to press from the current pose (2 from hover, measured ep1)
        excess = max(np.hypot(*(rob.pos[:2] - hover[:2])), rob.pos[2] - hover[2], 0.0)
        lat = CFG["LAT"] + int(np.ceil(max(0.0, excess - 0.005) / CFG["VMAX"]))
        if fired is None and trk.target is not None and lat <= CFG["LAT"] + 3:
            why = decide(trk.count, trk.inside, d, prev_d, pred, lat, N)
            if why:
                fired = rob.step_id
                log(ev="fire", why=why, step=rob.step_id, pred=pred, d=d, count=trk.count, lat=lat)
                print("FIRE", why, "at", rob.step_id, "d", d)
        prev_d = d
        if fired is not None and press_steps > 12:
            # press did not end the episode: retreat to hover and keep tracking
            log(ev="press_no_end", step=rob.step_id)
            print("press did not end the episode")
            fired = None; press_steps = 0
            N = trk.count + 1 if trk.count >= N else N
    log(ev="end", step=rob.step_id, ended=rob.ended, complete=rob.complete, events=trk.events, count=trk.count)
    print("END step", rob.step_id, "ended", rob.ended, "complete", rob.complete, "events", trk.events)
