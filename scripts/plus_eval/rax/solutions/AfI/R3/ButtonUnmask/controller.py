"""ButtonUnmask controller v2.

Task model (ep1 + rehearsal 1): cubes appear in waves (~step 12, 56, 100, ... period 44) and white bins
drop over them ~32 steps later. Lifts made while the hiding sequence is still running seem not to
count (reh1: red lifted at 130 during wave 3, then the correct blue lift at 208 failed). A wrong-bin
lift ends the episode. Instruction may say "all containers hiding green cubes" (several instances).

Transition table
  INIT    -> OBSERVE0
  OBSERVE0: at start pose (wrist sees whole table) zero steps until first cubes seen (>=2 sightings)
            or step >= PRESS_START_MAX -> PRESS
  PRESS   : rise/travel at TRAVEL_Z, descend on button, push, rise, back to PARK (start pose) -> WATCH
  WATCH   : zero steps at PARK, tracking with front+wrist, until no cube visible for QUIET_STEPS and
            every target colour has an instance (or WATCH_CAP) -> PICK list
  PICK    : per target instance: locate bins at PARK (wrist+front), nearest bin to instance xy,
            travel high, descend to APPROACH_Z, refine bin pose with wrist, grasp, verify finger gap
            (retry <=2), lift to LIFT_Z, hold, place back, release, rise, return to PARK.
  DONE    : hold & return.  EpisodeEnded anywhere -> return.
"""
import math, os, sys, time, traceback, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion import Robot, EpisodeEnded, BASE_ROT
from perception import cloud, find_bins, white_mask, rect_fit
from tracker import parse_targets, detect_cubes, detect_neutral, CubeMemory, NEUTRAL
from scipy import ndimage
from scipy.spatial.transform import Rotation as Rot

VERSION = "v2"
PRESS_DEPTH = 0.020        # TCP target below button top (ep1/reh1: TCP stalls ~0.009 below top)
PRESS_START_MIN = 20       # cubes appear ~step 12; give both cameras ~8 sightings of wave 1 first
PRESS_START_MAX = 30       # press must register before step 90 (PUBLIC_RULES)
QUIET_STEPS = 30           # gap between a wave's bins and next wave's cubes was 12 steps (ep1, reh1)
WATCH_CAP = 330
GRASP_BELOW_TOP = 0.015    # ep1: TCP .056-.058 held body (.0278/finger); reh1: .061 held knob (.0183)
APPROACH_Z = 0.20
LIFT_Z = 0.20              # reh1: correct-lift detection fired at TCP 0.164
FINGER_MIN_HELD = 0.008
OUT = "/workspace/output"

def log(*a):
    print(f"[{VERSION} {time.strftime('%H:%M:%S')}]", *a, flush=True)

class Ctl:
    def __init__(self, ctx):
        self.R = Robot(ctx, log)
        self.mem = CubeMemory()
        self.targets = []
        self.neutral_targets = []
        self.button = None
        self.park = None
        self.travel_z = 0.32
        self.errors = 0
        self.prev_rgb = None
        self.picked_xy = []
        self.t0 = time.time()

    # ---------------- perception ----------------
    def track(self, m, A):
        cubes = []
        for cam in ("front", "wrist"):
            if (cam, "rgb") not in A or (cam, "depth") not in A: continue
            P = cloud(m, A, cam)
            cubes += detect_cubes(A[(cam, "rgb")], P, cam)
            if self.neutral_targets:
                ex = [self.button[:2]] if self.button else []
                cubes += detect_neutral(A[(cam, "rgb")], P, cam, ex, self.neutral_targets)
        if self.button:
            cubes = [c for c in cubes if np.hypot(c["xy"][0] - self.button[0], c["xy"][1] - self.button[1]) > 0.03]
        self.mem.update(m["step_id"], cubes)
        rgb = A.get(("front", "rgb"))
        ch = 99999
        if rgb is not None and self.prev_rgb is not None:
            ch = int((np.abs(rgb.astype(int) - self.prev_rgb.astype(int)).sum(2) > 40).sum())
        self.prev_rgb = rgb
        return cubes, ch

    def step_track(self, rows):
        try:
            m = self.R.step(rows)
            A = self.R.last_arrays
        except EpisodeEnded:
            raise
        except Exception as e:
            self.errors += 1
            log("WARN step error", self.errors, repr(e)[:200])
            if self.errors > 30: raise
            for k in range(5):
                try:
                    m, A = self.R.observe(); break
                except Exception as e2:
                    self.errors += 1
                    log("WARN observe error", self.errors, repr(e2)[:200])
                    time.sleep(1.0 + k)
            else:
                raise RuntimeError("observe keeps failing")
            if self.R.ended:
                raise EpisodeEnded()
        return self.track(m, A)

    # ---------------- motion ----------------
    def goto(self, target, yaw=0.0, grip=1.0, tol=0.008, max_steps=40, vmax=1.0, rtol=0.04):
        R = self.R; n = 0
        tr = Rot.from_rotvec([0, 0, yaw]) * BASE_ROT
        while n < max_steps:
            p = np.array(R.state[:3]); e = np.array(target) - p
            re = R.rot_err(tr)
            if np.linalg.norm(e) < tol and np.linalg.norm(re) < rtol:
                break
            d = np.clip(e / 0.05, -vmax, vmax); r = np.clip(re / 0.5 * 0.8, -1, 1)
            k = 2 if np.linalg.norm(e) > 0.03 else 1
            self.step_track([[d[0], d[1], d[2], r[0], r[1], r[2], grip]] * k); n += k
        return float(np.linalg.norm(np.array(target) - np.array(R.state[:3])))

    def travel(self, xy, z_end, yaw=0.0, grip=1.0):
        """Rise to travel height, move horizontally, descend to z_end."""
        R = self.R
        p = np.array(R.state[:3])
        if np.linalg.norm(p[:2] - np.array(xy)) > 0.03:
            zt = max(self.travel_z, z_end)
            if p[2] < zt - 0.02:
                self.goto([p[0], p[1], zt], yaw=yaw, grip=grip, tol=0.02, max_steps=30)
            self.goto([xy[0], xy[1], zt], yaw=yaw, grip=grip, tol=0.015, max_steps=40)
        return self.goto([xy[0], xy[1], z_end], yaw=yaw, grip=grip, tol=0.008, max_steps=40)

    def to_park(self, grip=1.0):
        self.travel(self.park[:2], self.park[2], 0.0, grip)

    # ---------------- phases ----------------
    def init(self):
        m, A = self.R.observe()
        self.instruction = m.get("instruction", "")
        self.targets = parse_targets(self.instruction)
        self.neutral_targets = sorted({c for c, _ in self.targets if c in NEUTRAL})
        self.park = list(m["state"][:3])
        self.park[2] = min(max(self.park[2], 0.40), 0.55)
        self.button = self.find_button(m, A)
        self.travel_z = self.obstacle_travel_z(m, A)
        log("instruction:", self.instruction)
        log("targets:", self.targets, "park:", [round(v, 3) for v in self.park], "button:",
            [round(v, 3) for v in self.button], "travel_z", round(self.travel_z, 3), "limits", m.get("episode_limits"))
        self.track(m, A)

    def find_button(self, m, A):
        best = None
        for cam in ("front", "wrist"):
            P = cloud(m, A, cam); z = P[..., 2]
            msk = (z > 0.012) & (z < 0.06) & (P[..., 0] > -0.45) & (P[..., 0] < 0.3) & (np.abs(P[..., 1]) < 0.4)
            if cam == "wrist": msk &= A[("wrist", "depth")] > 0.15
            pts = P[msk]; pts = pts[~np.isnan(pts).any(1)]
            if len(pts) < 10: continue
            med = np.median(pts[:, :2], 0)
            pts = pts[np.linalg.norm(pts[:, :2] - med, axis=1) < 0.04]
            if len(pts) < 5: continue
            zt = np.percentile(pts[:, 2], 95)
            top = pts[pts[:, 2] > zt - 0.006]
            c = top[:, :2].mean(0)
            log("button from", cam, [round(float(c[0]), 4), round(float(c[1]), 4), round(float(zt), 4)], "n", len(pts))
            if best is None: best = [float(c[0]), float(c[1]), float(zt)]
            else: best = [(best[0] + c[0]) / 2, (best[1] + c[1]) / 2, max(best[2], float(zt))]
        return best or [-0.19, -0.015, 0.033]

    def obstacle_travel_z(self, m, A):
        zmax = 0.0
        for cam in ("front", "wrist"):
            P = cloud(m, A, cam); z = P[..., 2]
            msk = (z > 0.06) & (z < 0.5) & (P[..., 0] > -0.4) & (P[..., 0] < 0.4) & (np.abs(P[..., 1]) < 0.45)
            if cam == "wrist": msk &= A[("wrist", "depth")] > 0.15
            if cam == "front":   # exclude the arm near its start pose
                tcp = np.array(m["state"][:3])
                msk &= ~((np.abs(P[..., 0] - tcp[0]) < 0.2) & (np.abs(P[..., 1] - tcp[1]) < 0.2) & (z > 0.25))
            if msk.sum() > 20:
                zz = np.percentile(z[msk], 99)
                log("obstacle", cam, "n", int(msk.sum()), "z99", round(float(zz), 3))
                zmax = max(zmax, float(zz))
        return float(min(max(0.30, zmax + 0.10), 0.48))

    def observe0(self):
        while self.R.step_id < PRESS_START_MAX:
            self.step_track([[0, 0, 0, 0, 0, 0, 1.0]])
            good = any(it["n"] >= 2 for lst in self.mem.inst.values() for it in lst)
            if good and self.R.step_id >= PRESS_START_MIN:
                break
        log("observe0 done step", self.R.step_id, "mem", self.mem.summary())

    def press(self):
        bx, by, bz = self.button
        self.travel([bx, by], bz + 0.07, grip=1.0)
        log("above button step", self.R.step_id, [round(v, 4) for v in self.R.state[:3]])
        self.goto([bx, by, bz - PRESS_DEPTH], grip=1.0, tol=0.004, max_steps=8, vmax=0.8)
        for _ in range(3):
            self.step_track([[0, 0, -0.3, 0, 0, 0, 1.0]])
        log("pressed step", self.R.step_id, "tcp", [round(v, 4) for v in self.R.state[:3]])
        self.goto([bx, by, bz + 0.08], grip=1.0, tol=0.02, max_steps=8)
        self.to_park(grip=1.0)
        log("parked step", self.R.step_id)

    def watch(self):
        cap = WATCH_CAP
        if self.R.remaining is not None:
            cap = min(cap, self.R.step_id + max(0, self.R.remaining - 400))
        while self.R.step_id < cap:
            cubes, ch = self.step_track([[0, 0, 0, 0, 0, 0, 1.0]])
            quiet = self.R.step_id - self.mem.last_any_visible
            have = all(self.mem.instances(c) for c, _ in self.targets)
            if self.R.step_id % 10 == 0:
                log("watch step", self.R.step_id, "vis", self.mem.visible_colors(), "quiet", quiet, "ch", ch, "have", have)
            if have and quiet >= QUIET_STEPS and ch < 50:
                break
        log("watch done step", self.R.step_id, "mem", self.mem.summary())
        self.save_front("watch_done")

    def save_front(self, tag):
        try:
            from PIL import Image
            A = self.R.last_arrays
            for cam in ("front", "wrist"):
                Image.fromarray(A[(cam, "rgb")]).save(f"{OUT}/{self.R.step_id:04d}_{tag}_{cam}.png")
        except Exception as e:
            log("save fail", repr(e))

    def plan_targets(self):
        plan = []
        for col, allf in self.targets:
            inst = self.mem.instances(col)
            if not inst:
                inst = self.mem.instances(col, min_n=1)
            if not inst:
                log("WARN no instance for", col); plan.append((col, None)); continue
            if allf:
                for it in inst: plan.append((col, it["xy"]))
            else:
                it = max(inst, key=lambda t: t["n"]); plan.append((col, it["xy"]))
        # drop duplicates (same xy twice)
        out = []
        for col, xy in plan:
            if xy is not None and any(x is not None and np.linalg.norm(np.array(xy) - np.array(x)) < 0.03 for _, x in out):
                continue
            out.append((col, xy))
        return out

    def bins_at_park(self):
        m, A = self.R.observe()
        bins = []
        for cam in ("wrist", "front"):
            for b in find_bins(m, A, cam, min_px=15 if cam == "wrist" else 60):
                if not (0.055 < b["ztop"] < 0.09): continue
                if not (0.02 < min(b["ext"]) and max(b["ext"]) < 0.075): continue
                if not (-0.45 < b["xy"][0] < 0.35 and abs(b["xy"][1]) < 0.45): continue
                b["cam"] = cam
                if any(np.linalg.norm(np.array(b["xy"]) - np.array(o["xy"])) < 0.03 for o in bins): continue
                bins.append(b)
        return bins

    def refine_bin(self, xy):
        """Wrist view from above the bin: top-face rectangle nearest xy."""
        m, A = self.R.observe()
        P = cloud(m, A, "wrist"); z = P[..., 2]
        w = white_mask(A[("wrist", "rgb")]) & (z > 0.045) & (z < 0.095)
        lab, n = ndimage.label(w)
        best = None
        for i in range(1, n + 1):
            msk = lab == i
            if msk.sum() < 40: continue
            r = rect_fit(P[msk])
            if r is None: continue
            d = np.linalg.norm(np.array(r["xy"]) - np.array(xy))
            if d < 0.05 and 0.055 < r["ztop"] < 0.09 and (best is None or d < best[0]):
                best = (d, r)
        return best[1] if best else None

    def choose_yaw(self, b_xy, b_yaw, others):
        a = b_yaw
        alts = [a, a - math.pi / 2 if a > 0 else a + math.pi / 2]
        best = None
        for yaw in alts:
            axis = np.array([-math.sin(yaw), math.cos(yaw)])
            c = np.array(b_xy)
            fingers = [c + 0.045 * axis, c - 0.045 * axis]
            clear = min([np.linalg.norm(f - np.array(o["xy"])) for f in fingers for o in others] + [1.0])
            score = min(clear, 0.08) - 0.01 * abs(yaw)
            if best is None or score > best[0]:
                best = (score, yaw, clear)
        return best[1], best[2]

    def pick(self, color, cube_xy):
        R = self.R
        for attempt in range(3):
            if np.linalg.norm(np.array(R.state[:2]) - np.array(self.park[:2])) > 0.05 or R.state[2] < self.park[2] - 0.05:
                self.to_park(grip=-1.0)
            bins = self.bins_at_park()
            if not bins:
                log("no bins found"); self.step_track([[0, 0, 0, 0, 0, 0, -1]] * 3); continue
            if cube_xy is None:
                seen = [np.array(it["xy"]) for lst in self.mem.inst.values() for it in lst]
                cand = sorted(bins, key=lambda b: -min([np.linalg.norm(np.array(b["xy"]) - s) for s in seen] + [1.0]))
                cand = [b for b in cand if all(np.linalg.norm(np.array(b["xy"]) - np.array(p)) > 0.03 for p in self.picked_xy)]
                b = (cand or bins)[0]
                log("WARN guessing bin for", color, b["xy"])
            else:
                ds = [np.linalg.norm(np.array(b["xy"]) - np.array(cube_xy)) for b in bins]
                i = int(np.argmin(ds)); b = bins[i]
                log("target", color, "cube", [round(v, 3) for v in cube_xy], "bin", [round(v, 3) for v in b["xy"]],
                    b["cam"], "dist", round(ds[i], 3), "nbins", len(bins))
                if ds[i] > 0.05:
                    log("WARN no bin near cube; using cube xy")
                    b = dict(xy=list(cube_xy), yaw=0.0, ztop=0.073, ext=[0.04, 0.04], cam="cube")
            others = [o for o in bins if o is not b]
            # approach above the bin and refine with the wrist camera
            self.travel(b["xy"], APPROACH_Z, yaw=0.0, grip=-1.0)
            r = self.refine_bin(b["xy"])
            if r is not None and b["cam"] != "cube" and (np.linalg.norm(np.array(r["xy"]) - np.array(b["xy"])) > 0.02
                                                        or max(r["ext"]) > 0.055 or min(r["ext"]) < 0.025):
                log("refine rejected", [round(v, 3) for v in r["xy"]], [round(v, 3) for v in r["ext"]])
                r = None
            if r is not None:
                log("refined", [round(v, 3) for v in r["xy"]], "yaw", round(r["yaw"], 3), "ztop", round(r["ztop"], 3),
                    "ext", [round(v, 3) for v in r["ext"]], "shift", round(float(np.linalg.norm(np.array(r["xy"]) - np.array(b["xy"]))), 3))
                bx, by, yaw0, zt = r["xy"][0], r["xy"][1], r["yaw"], r["ztop"]
            else:
                log("refine failed; using park estimate")
                bx, by, yaw0, zt = b["xy"][0], b["xy"][1], b["yaw"], b["ztop"]
            yaw, clear = self.choose_yaw([bx, by], yaw0, others)
            gz = max(zt - GRASP_BELOW_TOP, 0.03)
            log("grasp plan", round(bx, 3), round(by, 3), "gz", round(gz, 3), "yaw", round(yaw, 3), "clear", round(clear, 3))
            self.goto([bx, by, APPROACH_Z - 0.04], yaw=yaw, grip=-1.0, tol=0.006, max_steps=40, rtol=0.03)
            self.goto([bx, by, gz], yaw=yaw, grip=-1.0, tol=0.004, max_steps=25, vmax=0.7, rtol=0.03)
            log("at grasp pose", [round(v, 4) for v in R.state])
            self.step_track([[0, 0, 0, 0, 0, 0, 1.0]] * 8)
            fing = R.state[6:8]
            log("closed fingers", [round(v, 4) for v in fing])
            if min(fing) < FINGER_MIN_HELD:
                log("grip miss; reopen, rise, retry")
                self.step_track([[0, 0, 0, 0, 0, 0, -1.0]] * 4)
                self.goto([bx, by, APPROACH_Z], yaw=yaw, grip=-1.0, tol=0.01, max_steps=20)
                continue
            self.goto([bx, by, LIFT_Z], yaw=yaw, grip=1.0, tol=0.01, max_steps=30, vmax=0.6)
            self.step_track([[0, 0, 0, 0, 0, 0, 1.0]] * 4)
            self.save_front("lifted_" + color)
            log("lifted", color, "fingers", [round(v, 4) for v in R.state[6:8]], "step", R.step_id, round(time.time() - self.t0, 1), "s")
            self.picked_xy.append([bx, by])
            # place back where it was
            self.goto([bx, by, gz + 0.012], yaw=yaw, grip=1.0, tol=0.006, max_steps=30, vmax=0.6)
            self.step_track([[0, 0, 0, 0, 0, 0, -1.0]] * 6)
            self.goto([bx, by, APPROACH_Z], yaw=yaw, grip=-1.0, tol=0.02, max_steps=20)
            return True
        return False

    def run(self):
        self.init()
        self.observe0()
        self.press()
        log("PHASE press done step", self.R.step_id, round(time.time() - self.t0, 1), "s")
        try:
            self.watch()
        except EpisodeEnded:
            raise
        except Exception as e:
            log("ERROR in watch", repr(e)); traceback.print_exc()
        log("PHASE watch done step", self.R.step_id, round(time.time() - self.t0, 1), "s")
        plan = self.plan_targets()
        log("plan", [(c, None if xy is None else [round(v, 3) for v in xy]) for c, xy in plan])
        for col, xy in plan:
            ok = False
            for tries in range(2):
                try:
                    ok = self.pick(col, xy); break
                except EpisodeEnded:
                    raise
                except Exception as e:
                    log("ERROR in pick", col, repr(e)); traceback.print_exc()
                    try: self.R.observe()
                    except Exception: pass
            log("PHASE pick", col, "ok", ok, "step", self.R.step_id, round(time.time() - self.t0, 1), "s")
        self.to_park(grip=-1.0)
        self.step_track([[0, 0, 0, 0, 0, 0, -1.0]] * 5)
        log("DONE step", self.R.step_id, "task_complete", self.R.task_complete)

def main(ctx):
    os.makedirs(OUT, exist_ok=True)
    c = Ctl(ctx)
    try:
        c.run()
    except EpisodeEnded:
        log("episode ended at step", c.R.step_id, "task_complete", c.R.task_complete)
    except Exception as e:
        log("FATAL", repr(e)); traceback.print_exc()
    try:
        log("main return; mem", c.mem.summary())
    except Exception:
        pass
