"""BinFill controller v3.

Task: "put N1 <c1> cubes and N2 <c2> cubes into the bin, then press the button to stop".
Cubes keep appearing (ep1: batches every ~50 steps on a static table) or ride a conveyor
(ep2: +y at ~2.4 mm/step, vanish near the mat end).  Cubes in the bin are invisible, so the
controller counts its own verified drops.

Steps and transitions (each step returns ok/why; main loop never returns early):
  INIT     : parse goal; bin/button/mat from first frame; velocity from two frames 6 steps apart
  SELECT   : resting needed-colour cubes from latest front+wrist frame -> intercept plan
             none -> SURVEY (go high, hold 6, re-detect, update velocity) -> SELECT
  PICK     : servo to predicted cube xy at HOVER_Z, descend with feed-forward, close, lift
             verify finger sum; fail -> SELECT (per-cube attempt counter, blacklist after 3)
  CARRY    : (first time) refine bin from overhead wrist view; move over bin; verify still held
             lost -> SELECT
  DROP     : release at rim_z + DROP_CLEAR; count += 1 -> SELECT (until goal met)
  PRESS    : hover over button (wrist refine), descend closed gripper until episode ends
Budget guard: if remaining steps < PRESS_RESERVE (+pick estimate) -> PRESS.
"""
import math, re, time, traceback, os
import numpy as np
from motion import Robot, yaw_of, wrap, ALPHA
from perception import (detect_cubes, resting_cubes, find_bin, find_button, find_mat,
                        match_velocity)
from obs import world_map

VERSION = "v3"
HOVER_Z = 0.105         # TCP height for tracking above cube (wrist sees the cube)
GRASP_BELOW_TOP = 0.020 # grasp TCP = cube top - 0.02 (ep1: 0.021 on table, held 0.0365)
LIFT_Z = 0.17
DROP_CLEAR = 0.065      # TCP above rim top at release (ep1 0.063, all 5 in)
HOLD_MIN, HOLD_MAX = 0.02, 0.05
SURVEY_POS = [-0.06, 0.0, 0.50]
PRESS_RESERVE = 90
PICK_TIMEOUT = 170      # steps per pick attempt
V_STATIC = 0.0004       # |v| below this (m/step) -> static scene
ARM_SPEED = 0.013       # m/step planning speed for intercept (vmax 0.0185 with ALPHA)

NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "a": 1, "an": 1}
LOG_PATH = "/workspace/output/controller_log.txt"
BASE_XY = np.array([-0.615, 0.0])


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    try:
        os.makedirs("/workspace/output", exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(s + "\n")
    except Exception:
        pass


def parse_goal(instr):
    goal = {}
    for m in re.finditer(r"\b(\d+|one|two|three|four|five|six|seven|a|an)\s+(red|blue|green)\s+cubes?",
                         instr.lower()):
        n = m.group(1); n = int(n) if n.isdigit() else NUM[n]
        goal[m.group(2)] = goal.get(m.group(2), 0) + n
    return goal


def reachable(xy):
    return (-0.42 < xy[0] < 0.22 and abs(xy[1]) < 0.34 and np.linalg.norm(xy - BASE_XY) < 0.80)


class Ctl:
    def __init__(self, ctx):
        self.r = Robot(ctx, log)
        self.goal = {}; self.count = {}
        self.bin = None; self.bin_refined = False
        self.button = None; self.mat = None
        self.v = np.zeros(2); self.v_known = False
        self.attempts = {}; self.blacklist = []
        self.t0 = time.time()
        self.last_front = None   # (step, dets)
        self.dets = []           # latest merged resting detections (with 'step')
        self.r.on_frame = self.on_frame

    # ---------------- perception on every reply ----------------
    def on_frame(self, r):
        meta, imgs = r.meta, r.imgs
        if ("front", "depth") not in imgs:
            return
        Wf = world_map(meta, imgs, "front")
        Ww = world_map(meta, imgs, "wrist")
        bxy = None if self.bin is None else self.bin["center"]
        df = resting_cubes(detect_cubes(meta, imgs, "front", W=Wf), bxy)
        dw = resting_cubes(detect_cubes(meta, imgs, "wrist", W=Ww), bxy)
        # velocity from consecutive front frames
        if self.last_front is not None:
            dt = r.step_id - self.last_front[0]
            if 3 <= dt <= 15:
                v, n = match_velocity(self.last_front[1], df, dt)
                if v is not None and n >= 3:
                    if np.linalg.norm(v) > 0.012:
                        pass
                    elif not self.v_known:
                        self.v = v; self.v_known = True
                    else:
                        self.v = 0.7 * self.v + 0.3 * v
        if not self.last_front or r.step_id - self.last_front[0] >= 3:
            self.last_front = (r.step_id, df)
        merged = []
        for d in dw + df:   # wrist first: closer and more accurate
            if any(e["color"] == d["color"] and np.linalg.norm(e["top"][:2] - d["top"][:2]) < 0.03 for e in merged):
                continue
            d["step"] = r.step_id
            merged.append(d)
        self.dets = merged
        self.Wf, self.Ww = Wf, Ww

    def vel(self):
        return self.v if np.linalg.norm(self.v) > V_STATIC else np.zeros(2)

    def init_scene(self):
        r = self.r
        meta, imgs = r.meta, r.imgs
        cands = []
        for cam, W in (("front", self.Wf), ("wrist", self.Ww)):
            b = find_bin(meta, imgs, cam, W=W)
            if b is not None:
                cands.append(b)
        if cands:
            # prefer the camera with more rim evidence
            self.bin = max(cands, key=lambda b: b["nrim"])
            log(f"[init] bin cands {[ (c['cam'], np.round(c['center'],3).tolist(), round(c['rim_z'],3), c['nrim']) for c in cands]}")
        bxy = None if self.bin is None else self.bin["center"]
        for cam, W in (("front", self.Wf), ("wrist", self.Ww)):
            bt = find_button(meta, imgs, cam, bxy, W=W)
            if bt is not None:
                self.button = bt; break
        self.mat = find_mat(meta, imgs, "front", W=self.Wf)
        if self.mat is not None and bxy is not None:
            # the bin floor can look like a small mat; require a large region
            if self.mat["n"] < 1500:
                self.mat = None
        log(f"[init] bin={None if self.bin is None else np.round(self.bin['center'],3)} "
            f"rim_z={None if self.bin is None else round(self.bin['rim_z'],3)} button={self.button} "
            f"mat={None if self.mat is None else (np.round(self.mat['lo'],3).tolist(), np.round(self.mat['hi'],3).tolist())}")

    def survey(self, high=True):
        r = self.r
        if high:
            r.move_to(SURVEY_POS, yaw=0.0, grip=-1.0, tol=0.03, max_steps=40)
        r.hold(6, -1.0)
        if self.bin is None:
            self.init_scene()
        log(f"[survey] step={r.step_id} v={np.round(self.v,5)} known={self.v_known} "
            f"cubes={[ (d['color'], np.round(d['top'][:2],3).tolist()) for d in self.dets]}")

    # ---------------- decisions ----------------
    def need(self):
        return {c: n - self.count.get(c, 0) for c, n in self.goal.items() if n - self.count.get(c, 0) > 0}

    def conveyor_end_ok(self, xy):
        v = self.vel()
        if np.linalg.norm(v) == 0:
            return True
        u = v / np.linalg.norm(v)
        if self.mat is not None:
            corners = [self.mat["lo"], self.mat["hi"], np.array([self.mat["lo"][0], self.mat["hi"][1]]),
                       np.array([self.mat["hi"][0], self.mat["lo"][1]])]
            end = max(c @ u for c in corners) - 0.08
        else:
            end = 0.20
        return xy @ u < end

    def plan(self, d):
        """Intercept time (steps from now) and xy, or None."""
        r = self.r
        v = self.vel()
        age = r.step_id - d.get("step", r.step_id)
        p0 = d["top"][:2] + v * age
        arm = r.pos()
        for t in range(0, 260, 4):
            p = p0 + v * t
            travel = np.linalg.norm(np.array([p[0], p[1], HOVER_Z]) - arm)
            if t >= travel / ARM_SPEED + 6:
                done = p + v * 40
                if reachable(p) and reachable(done) and self.conveyor_end_ok(done):
                    return t, p
                return None
            if np.linalg.norm(v) == 0:
                continue
        return None

    def choose(self):
        need = self.need()
        best = None
        for d in self.dets:
            if d["color"] not in need:
                continue
            if any(np.linalg.norm(d["top"][:2] - b) < 0.03 for b in self.blacklist) and np.linalg.norm(self.vel()) == 0:
                continue
            pl = self.plan(d)
            if pl is None:
                continue
            t, p = pl
            if best is None or t < best[0]:
                best = (t, p, d)
        return best

    def grasp_yaw(self, d, others):
        best = None
        for yaw in (d["yaw"], d["yaw"] + math.pi / 2, d["yaw"] - math.pi / 2):
            if abs(yaw) > math.pi / 2 + 1e-6:
                continue
            ax = np.array([-math.sin(yaw), math.cos(yaw)])   # finger closing axis (gripper y)
            clear = 1.0
            for o in others:
                v = o["top"][:2] - d["top"][:2]
                if np.linalg.norm(v) < 0.015:
                    continue
                along = abs(v @ ax); perp = abs(v @ np.array([ax[1], -ax[0]]))
                if perp < 0.05:
                    clear = min(clear, along)
            key = (min(clear, 0.09), -abs(yaw))
            if best is None or key > best[0]:
                best = (key, yaw)
        return best[1]

    # ---------------- actions ----------------
    def pick(self, d):
        r = self.r
        color = d["color"]
        v = self.vel()
        est = d["top"][:2] + v * (r.step_id - d.get("step", r.step_id))   # xy at r.step_id
        est_step = r.step_id
        top_z = d["top"][2]
        yaw = self.grasp_yaw(d, [o for o in self.dets if o is not d])
        start = r.step_id
        phase = "approach"; seen = 0; desc_z = HOVER_Z; last_wrist = -999
        gz = max(0.012, top_z - GRASP_BELOW_TOP)

        def pred(step):
            return est + v * (step - est_step)

        while not r.ended and r.step_id - start < PICK_TIMEOUT:
            # update estimate from latest detections near the prediction
            now = r.step_id
            pn = pred(now)
            cand = [e for e in self.dets if e["color"] == color and np.linalg.norm(e["top"][:2] - pn) < 0.035]
            if cand:
                e = min(cand, key=lambda e: (e["cam"] != "wrist", np.linalg.norm(e["top"][:2] - pn)))
                est = e["top"][:2].copy(); est_step = now; top_z = e["top"][2]; seen += 1
                if e["cam"] == "wrist":
                    last_wrist = now
                gz = max(0.012, top_z - GRASP_BELOW_TOP)
                if e["cam"] == "wrist" and e["ntop"] >= 10 and phase == "approach":
                    yaw = self.grasp_yaw(e, [o for o in self.dets if o is not e])
                    yaw = wrap(yaw) if abs(wrap(yaw)) <= math.pi / 2 else yaw
            pn = pred(now)
            p = r.pos()
            exy = np.linalg.norm(pn - p[:2])
            if phase == "approach":
                if exy < 0.008 and abs(p[2] - HOVER_Z) < 0.02 and r.yaw_err(yaw) < 0.08 and seen >= 1 \
                        and (now - last_wrist <= 4 or now - start > 60):
                    phase = "descend"; desc_z = p[2]
                    log(f"[pick] descend {color} at {np.round(pn,3)} top_z={top_z:.3f} yaw={math.degrees(yaw):.0f} step={now}")
                    continue
                k = 4 if exy > 0.05 else 2
                rows = r.plan_rows(lambda i: np.array([*pred(now + i), HOVER_Z]), yaw, -1.0, k,
                                   ff=np.array([*v, 0.0]))
                r.step(rows)
            elif phase == "descend":
                if p[2] <= gz + 0.004:
                    phase = "close"; continue
                if exy > 0.02:
                    phase = "approach"; continue
                k = 2
                zs = [max(gz, p[2] - 0.012 * (i)) for i in range(1, k + 1)]
                rows = r.plan_rows(lambda i: np.array([*pred(now + i), zs[i - 1]]), yaw, -1.0, k,
                                   ff=np.array([*v, 0.0]))
                r.step(rows)
            elif phase == "close":
                rows = r.plan_rows(lambda i: np.array([*pred(now + i), gz]), yaw, 1.0, 10,
                                   ff=np.array([*v, 0.0]))
                r.step(rows)
                q = r.pos()
                rows = r.plan_rows(lambda i: np.array([q[0] + v[0] * i, q[1] + v[1] * i, gz + 0.03 * i]),
                                   None, 1.0, 3)
                r.step(rows)
                r.move_to([r.pos()[0], r.pos()[1], LIFT_Z], yaw=None, grip=1.0, tol=0.02, max_steps=16)
                gs = r.grip_open()
                ok = HOLD_MIN < gs < HOLD_MAX
                log(f"[pick] closed finger_sum={gs:.4f} ok={ok} step={r.step_id} took={r.step_id-start}")
                return ok, ("held" if ok else f"empty {gs:.4f}")
        log(f"[pick] timeout/ended phase={phase} seen={seen} step={r.step_id}")
        return False, "timeout"

    def refine_bin(self):
        r = self.r
        c = self.bin["center"]
        r.move_to([c[0], c[1], 0.26], yaw=None, grip=1.0, tol=0.01, max_steps=60)
        if r.ended:
            return
        b = find_bin(r.meta, r.imgs, "wrist", W=self.Ww, near=c)
        if b is not None and np.linalg.norm(b["center"] - c) < 0.06 and b["nrim"] >= 25:
            log(f"[bin] refined {np.round(c,3)} -> {np.round(b['center'],3)} rim_z={b['rim_z']:.3f} nrim={b['nrim']}")
            self.bin = b
            self.bin_refined = True
        else:
            log(f"[bin] refine failed ({None if b is None else (np.round(b['center'],3), b['nrim'])})")

    def drop(self):
        r = self.r
        if not self.bin_refined:
            self.refine_bin()
        bx, by = self.bin["center"]
        dz = self.bin["rim_z"] + DROP_CLEAR
        r.move_to([bx, by, max(dz + 0.05, r.pos()[2] if r.pos()[2] < 0.3 else 0.0)], yaw=None, grip=1.0,
                  tol=0.01, max_steps=60)
        r.move_to([bx, by, dz], yaw=None, grip=1.0, tol=0.004, max_steps=30)
        gs = r.grip_open()
        if not (HOLD_MIN < gs < HOLD_MAX):
            log(f"[drop] grasp lost before drop finger_sum={gs:.4f}")
            return False
        p = r.pos()
        r.hold(6, -1.0)
        log(f"[drop] released at {np.round(p,3)} bin={np.round(self.bin['center'],3)} off={np.linalg.norm(p[:2]-self.bin['center'])*1000:.0f}mm step={r.step_id}")
        r.move_to([bx, by, max(dz + 0.06, 0.15)], yaw=None, grip=-1.0, tol=0.02, max_steps=10)
        return True

    def press(self):
        r = self.r
        if self.button is None:
            r.move_to(SURVEY_POS, yaw=0.0, grip=1.0, tol=0.03, max_steps=40)
            bxy = None if self.bin is None else self.bin["center"]
            for cam, W in (("front", self.Wf), ("wrist", self.Ww)):
                bt = find_button(r.meta, r.imgs, cam, bxy, W=W)
                if bt is not None:
                    self.button = bt; break
        if self.button is None:
            log("[press] no button found"); return False
        bx, by, bz = self.button
        r.move_to([bx, by, 0.12], yaw=0.0, grip=1.0, tol=0.005, max_steps=60)
        bxy = None if self.bin is None else self.bin["center"]
        bt = find_button(r.meta, r.imgs, "wrist", bxy, W=self.Ww, near=self.button)
        if bt is not None:
            bx, by, bz = bt
            log(f"[press] refined button {np.round(bt,3)} (was {np.round(self.button,3)})")
            r.move_to([bx, by, 0.10], yaw=0.0, grip=1.0, tol=0.004, max_steps=20)
        for k, depth in enumerate((0.0, -0.008, -0.016)):
            offs = [(0, 0), (0.006, 0), (-0.006, 0)][k]
            z = bz - 0.02 + depth
            tx, ty = bx + offs[0], by + offs[1]
            for _ in range(20):
                p = r.pos()
                if r.ended or p[2] <= z + 0.003:
                    break
                zs = max(z, p[2] - 0.02)
                r.step(r.plan_rows(lambda i: np.array([tx, ty, zs]), 0.0, 1.0, 2))
            r.hold(4, 1.0)
            log(f"[press] at {np.round(r.pos(),3)} ended={r.ended} complete={r.complete} step={r.step_id}")
            if r.ended:
                return True
            r.move_to([tx, ty, 0.08], yaw=0.0, grip=1.0, tol=0.006, max_steps=15)
            if r.ended:
                return True
        return r.ended

    # ---------------- main loop ----------------
    def run(self):
        r = self.r
        meta = r.observe()
        self.goal = parse_goal(meta.get("instruction", ""))
        log(f"[{VERSION}] instruction={meta.get('instruction')!r} goal={self.goal} limit={meta.get('episode_limits')}")
        if not self.goal:
            self.goal = {"red": 3, "blue": 3}
        self.count = {c: 0 for c in self.goal}
        self.init_scene()
        r.hold(6, -1.0)   # second frame -> conveyor velocity
        log(f"[init] v={np.round(self.v,5)} known={self.v_known} cubes={[ (d['color'], np.round(d['top'][:2],3).tolist()) for d in self.dets]}")
        misses = 0
        while not r.ended and self.need():
            if r.remaining is not None and r.remaining < PRESS_RESERVE + 60:
                log("[main] step budget low; going to press"); break
            if time.time() - self.t0 > 1000:
                log("[main] wall budget low; going to press"); break
            try:
                ch = self.choose()
                if ch is None:
                    misses += 1
                    self.survey(high=(misses % 3 == 1))
                    continue
                misses = 0
                t, p, d = ch
                log(f"[main] target {d['color']} at {np.round(d['top'][:2],3)} intercept t={t} p={np.round(p,3)} need={self.need()} step={r.step_id}")
                ok, why = self.pick(d)
                if not ok:
                    key = (d["color"], round(float(d["top"][0]), 2), round(float(d["top"][1]), 2))
                    self.attempts[key] = self.attempts.get(key, 0) + 1
                    if self.attempts[key] >= 3:
                        self.blacklist.append(np.array(d["top"][:2]))
                    log(f"[main] pick failed ({why})")
                    r.move_to([r.pos()[0], r.pos()[1], LIFT_Z], yaw=None, grip=-1.0, tol=0.03, max_steps=10)
                    continue
                if self.drop():
                    self.count[d["color"]] += 1
                    log(f"[main] COUNT {self.count} need {self.need()} step={r.step_id} t={time.time()-self.t0:.0f}s")
                else:
                    r.hold(3, -1.0)
            except Exception:
                log("[main] exception", traceback.format_exc())
                if r.ended:
                    break
                try:
                    r.observe()
                except Exception:
                    log("[main] observe failed", traceback.format_exc())
        if not r.ended:
            try:
                self.press()
            except Exception:
                log("[press] exception", traceback.format_exc())
        log(f"[end] count={self.count} ended={r.ended} complete={r.complete} step={r.step_id} t={time.time()-self.t0:.0f}s")


def main(ctx):
    Ctl(ctx).run()
