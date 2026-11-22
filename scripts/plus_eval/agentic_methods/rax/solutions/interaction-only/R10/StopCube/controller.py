"""StopCube controller v1.
Track the cube with the front RGB-D camera every control step, count its passes over the
target, hover a closed gripper just above the button cap, and press so the press registers
as the cube reaches the target on the N-th pass."""
import json, os, re, sys, time, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sclib, presser

VERSION = "v4"
LATENCY = 2.0          # steps from press decision to registration (ep1 measured)
HOVER_DZ = 0.024       # TCP height above cap top while waiting (press threshold ~cap_top+0.003; 1st press step -> ~+0.011, 2nd -> ~-0.005: latency 2)
HIGH_DZ = 0.045        # transit end height above cap top (then slow descent)
ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
       "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "once": 1, "twice": 2, "thrice": 3,
       "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}

def parse_n(instr):
    s = instr.lower()
    m = re.search(r"(\d+)(st|nd|rd|th)?\s+time", s)
    if m:
        return int(m.group(1))
    for w, n in ORD.items():
        if re.search(r"\b%s\b" % w, s):
            return n
    return 1

class Run:
    def __init__(self, ctx):
        self.ctx = ctx
        os.makedirs("/workspace/output", exist_ok=True)
        self.logf = open("/workspace/output/stopcube_log.jsonl", "a")
        self.t0 = time.time()
        self.ended = False
        self.complete = None
        self.meta = None; self.imgs = None
        self.best_target = None; self.best_tpx = -1

    def log(self, **kw):
        kw["t"] = round(time.time() - self.t0, 2)
        line = json.dumps(kw, default=lambda o: np.asarray(o).round(4).tolist())
        print(line); self.logf.write(line + "\n"); self.logf.flush()

    def observe(self):
        self.meta, self.imgs = sclib.decode(self.ctx.call("robomme_observe"))
        return self.meta

    def step(self, row):
        res = self.ctx.call("robomme_step", actions=[[float(x) for x in row]])
        self.meta, self.imgs = sclib.decode(res)
        steps = self.meta.get("transition", {}).get("steps", [])
        if self.meta.get("episode_ended") or any(s.get("episode_ended") for s in steps):
            self.ended = True
            self.complete = any(s.get("task_complete") for s in steps)
            self.log(ev="ENDED", step=self.meta.get("step_id"), last=steps[-1] if steps else None)
        return self.meta

    def remaining(self):
        return self.meta.get("episode_limits", {}).get("remaining_episode_steps", 999)

    def perceive(self):
        det = sclib.detect(self.meta, self.imgs, "front", ee=self.meta["state"][:3])
        if "target" in det and det["target_px"] > self.best_tpx:
            self.best_tpx = det["target_px"]; self.best_target = det["target"]
        return det

def find_static_cap(R, det0):
    """Fallback: the button is a static raised blob (top ~0.03 m) that is neither the cube nor the target.
    Watch a few steps, keep raised blobs that do not move."""
    from scipy import ndimage
    blobs = []
    for i in range(4):
        P, d = sclib.world_points(R.meta, R.imgs, "front")
        z = P[..., 2]
        m = (d > 0) & (P[..., 0] > -0.35) & (P[..., 0] < 0.45) & (np.abs(P[..., 1]) < 0.6) & (z > 0.024) & (z < 0.06)
        lab, n = ndimage.label(m)
        cur = []
        for k in range(1, n + 1):
            mk = lab == k
            if mk.sum() >= 4:
                cur.append((P[mk].mean(0), float(np.percentile(P[mk][:, 2], 95)), int(mk.sum())))
        blobs.append(cur)
        if i < 3:
            R.step([0, 0, 0, 0, 0, 0, 1])
    best = None
    for c, top, npx in blobs[-1]:
        static = all(any(np.linalg.norm(c[:2] - c0[:2]) < 0.01 for c0, _, _ in b) for b in blobs[:-1])
        if static and (best is None or npx > best[2]):
            best = (c, top, npx)
    if best is None:
        return np.array([-0.15, 0.26, 0.033]), 0.034
    return best[0], best[1]

def axis_rows(cur, target, grip=1.0):
    d = (np.asarray(target) - np.asarray(cur)) / 0.05
    d = np.clip(d * 1.0, -1, 1)
    return [d[0], d[1], d[2], 0, 0, 0, grip]

def main(ctx):
    R = Run(ctx)
    try:
        run(R)
    except Exception:
        R.log(ev="EXC", tb=traceback.format_exc())
        # fall back: keep holding pose until the episode ends or steps run out
        try:
            while not R.ended and R.remaining() > 0:
                R.step([0, 0, 0, 0, 0, 0, 1])
        except Exception:
            R.log(ev="EXC2", tb=traceback.format_exc())

def run(R):
    meta = R.observe()
    instr = meta.get("instruction", "")
    N = parse_n(instr)
    det = R.perceive()
    R.log(ev="start", version=VERSION, instr=instr, N=N, step=meta["step_id"],
          target=det.get("target"), cube=det.get("cube"), cap=det.get("cap"), cap_top=det.get("cap_top"),
          limit=meta.get("episode_limits"))
    # button cap from the first clear frame (arm far away)
    cap = det.get("cap"); cap_top = det.get("cap_top")
    if cap is None:
        R.log(ev="no_cap_detected")
        cap, cap_top = find_static_cap(R, det)
        R.log(ev="cap_fallback", cap=cap, cap_top=cap_top)
        det = R.perceive()
    cap_xy = np.asarray(cap[:2], float)
    pc = presser.PassCounter(r_in=0.03, r_out=0.042, latency=LATENCY, press_on=N, done=0, margin=0.5)
    high = np.array([cap_xy[0], cap_xy[1], cap_top + HIGH_DZ])
    hover = np.array([cap_xy[0], cap_xy[1], cap_top + HOVER_DZ])
    phase = "transit"; fired_at = None; press_rows = 0; lifted = 0
    n_missing = 0
    while not R.ended and R.remaining() > 0:
        meta = R.meta
        sid = meta["step_id"]
        T = R.best_target
        c = det.get("cube")
        if c is None:
            n_missing += 1
        ee = np.asarray(meta["state"][:3])
        # latency grows with height above the hover point (~0.018 m/step descent)
        pc.latency = LATENCY + max(0.0, float(ee[2] - hover[2])) / 0.018
        fire, info = (False, {"why": "no target"}) if T is None else pc.update(sid, None if c is None else c[:2], T[:2])
        ee = np.asarray(meta["state"][:3])
        # arm motion by phase
        if phase == "transit":
            row = axis_rows(ee, high)
            if np.linalg.norm(ee - high) < 0.02:
                phase = "descend"
        elif phase == "descend":
            row = sclib.move_rows(ee, hover, 1.0, 0.5)
            if np.linalg.norm(ee - hover) < 0.003:
                phase = "wait"
        elif phase == "wait":
            row = sclib.move_rows(ee, hover, 1.0, 0.3)
        elif phase == "press":
            row = [0, 0, -1, 0, 0, 0, 1]
            press_rows += 1
            if press_rows >= 6:
                phase = "lift"
        elif phase == "lift":
            row = sclib.move_rows(ee, hover, 1.0, 0.6); lifted += 1
            if lifted > 8:
                phase = "wait"; lifted = 0; fired_at = None; press_rows = 0
                pc.press_on = pc.done + 2   # missed; re-arm for a later pass (fallback only)
        near_xy = np.linalg.norm(ee[:2] - cap_xy) < 0.012
        if fire and fired_at is None and (phase in ("wait", "descend") or (phase == "transit" and near_xy)):
            fired_at = sid; phase = "press"; press_rows = 1
            row = [0, 0, -1, 0, 0, 0, 1]
            info["FIRE"] = True
        elif fire and phase == "transit":
            info["FIRE_BLOCKED_TRANSIT"] = True
        R.log(step=sid, phase=phase, ee=ee, cube=None if c is None else c[:2], **info)
        R.step(row)
        if R.ended:
            break
        det = R.perceive()
    R.log(ev="done", complete=R.complete, step=R.meta.get("step_id"), missing=n_missing,
          entries=pc.entries, done=pc.done, recovered=getattr(pc, "recovered", 0))
