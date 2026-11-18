"""StopCube controller v3: count cube arrivals at target, press button on the Nth."""
import json, os, re, sys, time, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rax import decode
from percep import find_button, find_cube_generic, target_from_static, target_position

ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
       "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "once": 1, "twice": 2,
       "thrice": 3, "1st": 1, "2nd": 2, "3rd": 3}
PRESS_LAT = 2.0     # rows from trigger until press registers (ep1: 2nd row)
HOVER_DZ = 0.018    # TCP height above cap top while waiting
R_IN, R_OUT = 0.025, 0.05
OUT = "/workspace/output"

def log(*a):
    print(*a, flush=True)

def parse_n(instr):
    s = instr.lower()
    m = re.search(r"for the (\w+) time", s)
    if m and m.group(1) in ORD: return ORD[m.group(1)]
    m = re.search(r"(\d+)(?:st|nd|rd|th)? time", s)
    if m: return int(m.group(1))
    for w, k in ORD.items():
        if re.search(r"\b%s\b" % w, s): return k
    return 1

def seg_dist(a, b, t):
    ab = b - a; L2 = float(ab @ ab)
    if L2 < 1e-12: return float(np.linalg.norm(a - t))
    s = np.clip(float((t - a) @ ab) / L2, 0, 1)
    return float(np.linalg.norm(a + s * ab - t))

class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        os.makedirs(OUT, exist_ok=True)
        self.logf = open(os.path.join(OUT, "ctrl_log.jsonl"), "a")
        self.ended = False; self.complete = None
        self.target = None; self.target_px = 0
        self.button = None
        self.arrivals = []; self.armed = True
        self.last_cube = None; self.prev_cube = None; self.last_cube_step = None
        self.cube_hist = []
        self.gripper = 1.0
        self.n_steps = 0
        self.tgt = None; self.buffer = []

    def process(self, meta, imgs):
        self.meta = meta
        self.state = np.array(meta["state"], float)
        self.step_id = meta.get("step_id")
        tr = meta.get("transition") or {}
        if meta.get("episode_ended") or tr.get("episode_ended"):
            self.ended = True
            self.complete = tr.get("task_complete")
            for s in tr.get("steps", []):
                if s.get("task_complete") is not None: self.complete = s.get("task_complete")
        rgb = imgs.get(("front", "rgb")); dep = imgs.get(("front", "depth"))
        if rgb is None or dep is None or "camera_poses" not in meta:
            self.logf.write(json.dumps(dict(step=self.step_id, note="no image")) + "\n"); return
        if self.tgt is None:
            self.buffer.append((meta, rgb, dep))
            if len(self.buffer) < self.TGT_FRAMES and not self.ended: return
            self.init_target()
            buf, self.buffer = self.buffer, []
            for m_, r_, d_ in buf:
                self.track(m_, r_, d_)
            return
        self.track(meta, rgb, dep)

    TGT_FRAMES = 9

    def init_target(self):
        try:
            tg = target_from_static([f[1] for f in self.buffer])
            if tg is not None:
                pos = target_position(tg, self.buffer)
                if pos is not None:
                    self.tgt = tg; self.target = pos[0]
                    log("target", self.target.round(4).tolist(), "chroma", tg["chroma"].round(3).tolist(), "px", pos[1])
        except Exception:
            log("target err", traceback.format_exc())
        if self.tgt is None:
            log("TARGET NOT FOUND; using empty footprint")
            self.tgt = False

    def track(self, meta, rgb, dep):
        step_id = meta.get("step_id")
        rec = dict(step=step_id, ee=[round(float(x), 4) for x in meta["state"][:3]])
        try:
            fc = find_cube_generic(meta, rgb, dep, self.tgt if self.tgt else None)
        except Exception as e:
            fc = None; log("cube err", e)
        if fc is not None and fc[0] is not None:
            cw = fc[0][:2]
            rec["cube"] = cw.round(4).tolist()
            if self.target is not None:
                t = self.target[:2]
                d = float(np.linalg.norm(cw - t)); rec["d"] = round(d, 4)
                ds = seg_dist(self.last_cube, cw, t) if self.last_cube is not None else d
                if self.armed and ds < R_IN:
                    self.arrivals.append(step_id); self.armed = False
                    log(f"ARRIVAL #{len(self.arrivals)} step {step_id} d={d:.3f} seg={ds:.3f}")
                elif not self.armed and d > R_OUT:
                    self.armed = True
            self.prev_cube = (self.last_cube, self.last_cube_step)
            self.last_cube, self.last_cube_step = cw, step_id
        rec["n"] = len(self.arrivals)
        self.logf.write(json.dumps(rec) + "\n"); self.logf.flush()

    def observe(self):
        meta, imgs = decode(self.ctx.call("robomme_observe"))
        self.process(meta, imgs)
        return meta, imgs

    def step(self, row):
        row = [float(np.clip(x, -1, 1)) for x in row]
        meta, imgs = decode(self.ctx.call("robomme_step", actions=[row]))
        self.n_steps += 1
        self.process(meta, imgs)
        return meta, imgs

    def hold(self):
        return self.step([0, 0, 0, 0, 0, 0, self.gripper])

    def remaining(self):
        lim = (self.meta or {}).get("episode_limits") or {}
        return lim.get("remaining_episode_steps", 999)

    def prediction(self):
        """(tau steps to closest approach, closest distance) from last two consecutive sightings."""
        if self.target is None or self.last_cube is None: return None
        prev, ps = self.prev_cube
        if prev is None or ps is None or self.last_cube_step != self.step_id: return None
        dt = self.step_id - ps
        if dt <= 0 or dt > 2: return None
        p = self.last_cube; v = (p - prev) / dt; t = self.target[:2]
        vv = float(v @ v)
        if vv < 1e-8: return None
        tau = -float((p - t) @ v) / vv
        dmin = float(np.linalg.norm(p - t + tau * v))
        return tau, dmin

    def should_fire(self):
        """Fire now if the linearly predicted cube distance to target LAT rows ahead is
        small and not larger than one step later (offline: matches oracle on ep1/ep2)."""
        if self.target is None or self.last_cube is None: return None
        prev, ps = self.prev_cube
        if prev is None or ps is None or self.last_cube_step != self.step_id: return None
        dt = self.step_id - ps
        if dt <= 0 or dt > 2: return None
        p = self.last_cube; v = (p - prev) / dt; t = self.target[:2]
        k = PRESS_LAT
        d2 = float(np.linalg.norm(p + k * v - t)); d3 = float(np.linalg.norm(p + (k + 1) * v - t))
        if d2 < 0.03 and d2 <= d3: return d2, d3
        return None

    def move_to(self, goal, vmax=0.4, tol=0.003, max_steps=60, ready=None):
        goal = np.array(goal, float)
        for _ in range(max_steps):
            if self.ended: return False
            if ready is not None and ready(): return False
            err = goal - self.state[:3]
            if np.linalg.norm(err) < tol: return True
            dv = err / 0.05
            n = np.max(np.abs(dv))
            if n > vmax: dv = dv * vmax / n
            self.step([dv[0], dv[1], dv[2], 0, 0, 0, self.gripper])
        return np.linalg.norm(goal - self.state[:3]) < 3 * tol

def run(ctx):
    R = Robot(ctx)
    meta, imgs = R.observe()
    N = parse_n(meta.get("instruction", ""))
    log("instruction:", meta.get("instruction"), "N=", N, "step", R.step_id)
    # --- locate button (retry on later frames if needed)
    btn = None
    for attempt in range(20):
        try:
            btn = find_button(meta, imgs[("front", "rgb")], imgs[("front", "depth")])
        except Exception as e:
            log("button err", e)
        if btn is not None or R.ended: break
        meta, imgs = R.hold()
    if btn is None:
        log("BUTTON NOT FOUND"); return R
    bxy, bz, bsz = btn
    log("button", bxy.round(4).tolist(), round(bz, 4), bsz, "target", None if R.target is None else R.target.round(4).tolist())
    hover = np.array([bxy[0], bxy[1], bz + HOVER_DZ])
    transit = np.array([bxy[0], bxy[1], bz + 0.05])
    # --- go above the button quickly, then descend to hover
    R.move_to(transit, vmax=1.0, tol=0.01, max_steps=60)
    R.move_to(hover, vmax=0.3, tol=0.002, max_steps=40)
    log("hover reached", R.step_id, R.state[:3].round(4).tolist(), "arrivals", R.arrivals)
    # --- wait and trigger
    pressed = False
    while not R.ended and R.remaining() > 1:
        if len(R.arrivals) == N - 1 and R.armed:
            fire = R.should_fire()
            if fire is not None:
                d2, d3 = fire
                log(f"TRIGGER step {R.step_id} pred_d2={d2:.3f} pred_d3={d3:.3f} d={np.linalg.norm(R.last_cube-R.target[:2]):.3f}")
                pressed = True
                break
        if len(R.arrivals) >= N:
            log("MISSED: arrivals already", len(R.arrivals)); pressed = True; break
        # keep hover position (small correction)
        err = hover - R.state[:3]
        dv = np.clip(err / 0.05, -0.2, 0.2) if np.linalg.norm(err) > 0.002 else np.zeros(3)
        R.step([dv[0], dv[1], dv[2], 0, 0, 0, R.gripper])
    for k in range(8):
        if R.ended: break
        R.step([0, 0, -1, 0, 0, 0, R.gripper])
        log("press row", k, R.step_id, round(R.state[2], 4), "d", None if R.last_cube is None else round(float(np.linalg.norm(R.last_cube - R.target[:2])), 4), R.ended, R.complete)
    for k in range(10):
        if R.ended: break
        R.hold()
    log("END step", R.step_id, "ended", R.ended, "task_complete", R.complete, "arrivals", R.arrivals)
    return R

def main(ctx):
    t0 = time.time()
    try:
        run(ctx)
    except Exception:
        log("CONTROLLER EXCEPTION\n" + traceback.format_exc())
    log("wall", round(time.time() - t0, 1))
