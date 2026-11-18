"""ButtonUnmaskSwap controller: press both buttons while tracking containers, then pick in order."""
import json, math, os, re, sys, time, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robo, percep, tracker, tracker2, grasp
from robo import log

BASE = np.array([-0.615, 0.0])
REACH_MAX, REACH_MIN = 0.78, 0.30
R0 = robo.rotvec_to_R(np.array([-math.pi, 0.0, 0.0]))
COLORS = ["red", "green", "blue", "yellow", "orange", "purple", "black", "white", "gray"]
OUT = "/workspace/output"

def clamp_target(p):
    p = np.array(p, float)
    p[2] = min(max(p[2], 0.012), 0.45)
    v = p[:2] - BASE
    d = np.linalg.norm(v)
    if d > REACH_MAX:
        p[:2] = BASE + v / d * REACH_MAX
    if d < REACH_MIN:
        p[:2] = BASE + v / max(d, 1e-6) * REACH_MIN
    return p

def parse_order(instr):
    s = instr.lower()
    i = s.find("pick")
    s = s[i:] if i >= 0 else s
    found = []
    for m in re.finditer(r"\b(" + "|".join(COLORS) + r")\b", s):
        if m.group(1) not in found:
            found.append(m.group(1))
    return found

XLIM = (-0.45, 0.18)
WRIST_MIN_Z = 0.30
OVER_Z = 0.40
PRESS_START = 58

def detect_initial(o):
    """Detect buttons/cubes in front and wrist views; merge duplicates by xy."""
    allb = []
    for cam in ("wrist", "front"):
        try:
            for b in percep.blobs(o, cam=cam, zmin=0.008, zmax=0.06, min_px=20, xlim=XLIM):
                b["cam"] = cam
                allb.append(b)
        except Exception as e:
            log("detect err", cam, repr(e))
    merged = []
    for b in allb:
        for m in merged:
            if np.linalg.norm(m["xyz"][:2] - b["xyz"][:2]) < 0.03:
                break
        else:
            merged.append(b)
    bs = merged
    buttons, cubes = [], []
    for b in bs:
        ext = b["ext"]
        name = percep.color_name(b["rgb"])
        if ext[0] > 0.058 and ext[1] > 0.058 and b["zmax"] < 0.045 and name in ("white", "gray") and b["n"] > 150:
            buttons.append(b)
        elif b["zmax"] < 0.05:
            cubes.append(dict(b, name=name))
    return buttons, cubes

class Ctl:
    def __init__(self, ctx):
        self.ctx = ctx
        self.r = robo.Robot(ctx)
        self.T = None
        self.t0 = time.time()
        self.grip = 1.0

    @property
    def obs(self):
        return self.r.obs

    def steps_left(self):
        return self.obs.limits.get("remaining_episode_steps", 0)

    def cap_pts(self, cams=("front", "wrist")):
        allp = []
        for cam in cams:
            try:
                pw, ok = self.obs.world_map(cam)
                if cam == "wrist":
                    if self.obs.pose["wrist"][0][2] < WRIST_MIN_Z:
                        continue
                    ok = ok & (self.obs.depth["wrist"] > 0.03)
                pts, _ = tracker.cap_points(pw, ok, ee=self.obs.state[:3], xlim=XLIM)
                if cam == "wrist" and len(pts) > 1500:
                    pts = pts[::max(1, len(pts) // 1500)]
                allp.append(pts)
            except Exception as e:
                log("cap_pts err", cam, repr(e))
        return np.concatenate(allp, 0) if allp else np.zeros((0, 2))

    def track(self):
        if self.T is None or self.obs.step_id < 6:
            return
        self.T.update(self.cap_pts(), self.obs.step_id)

    def rows_toward(self, target, n, vmax, Rt=None):
        target = clamp_target(target)
        p = self.r.pos.copy()
        rot = self.r.rot_cmd(R0 if Rt is None else Rt)
        rows = []
        for i in range(n):
            v = np.clip((target - p) / 0.05, -vmax, vmax)
            rows.append(list(v) + list(rot if i == 0 else [0, 0, 0]) + [self.r.grip])
            p = p + v * 0.05
        return rows

    def go_tracked(self, target, tol, vmax, chunk=2, max_steps=80, Rt=None, stall_ok=False):
        """Move while tracking every `chunk` steps."""
        target = clamp_target(target)
        start = self.obs.step_id
        last = []
        while not self.obs.ended and self.obs.step_id - start < max_steps:
            e = np.linalg.norm(target - self.r.pos)
            if e < tol:
                return True
            self.r.do(self.rows_toward(target, chunk, vmax, Rt))
            self.track()
            last.append(self.r.pos.copy())
            if stall_ok and len(last) >= 3 and np.linalg.norm(last[-1] - last[-3]) < 0.002:
                return True
        return False

    def wait_tracked(self, n, chunk=2):
        for _ in range(max(1, n // chunk)):
            if self.obs.ended:
                return
            self.r.do([[0, 0, 0, 0, 0, 0, self.r.grip]] * chunk)
            self.track()

    # ---------- phases ----------
    def press_buttons(self, buttons):
        self.r.grip = 1.0
        bl = sorted(buttons, key=lambda b: np.linalg.norm(b["xyz"][:2] - self.r.pos[:2]))
        for i, b in enumerate(bl):
            bx, by = b["xyz"][0], b["xyz"][1]
            # lateral move at overview height (wrist sees the table), then resync a few steps
            self.go_tracked([bx, by, OVER_Z], tol=0.02, vmax=1.0, chunk=1, max_steps=30)
            self.wait_tracked(2, chunk=1)
            # fast dive, press, pop up
            self.go_tracked([bx, by, 0.06], tol=0.012, vmax=1.0, chunk=1, max_steps=16)
            self.go_tracked([bx, by, 0.035], tol=0.006, vmax=0.5, chunk=1, max_steps=6)
            ok = self.go_tracked([bx, by, 0.006], tol=0.003, vmax=0.3, chunk=1, max_steps=7)
            log(f"press{i} at step {self.obs.step_id} ee={np.round(self.r.pos,3)} reached={ok}")
            self.go_tracked([bx, by, OVER_Z], tol=0.02, vmax=1.0, chunk=1, max_steps=16)
            self.wait_tracked(2, chunk=1)

    def wait_still(self, min_step=150):
        limit = self.obs.limits.get("episode_step_limit", 1300)
        max_step = max(300, limit - 480)
        hist = []
        while not self.obs.ended and self.obs.step_id < max_step:
            self.wait_tracked(1, chunk=1)
            hist.append((self.obs.step_id, self.T.c.copy()))
            if self.obs.step_id >= min_step and len(hist) >= 12:
                mv = np.linalg.norm(hist[-1][1] - hist[-12][1], axis=1)
                if np.percentile(mv, 80) < 0.003 and mv.max() < 0.012:
                    # confirm a few more steps
                    log(f"containers still at step {self.obs.step_id} (max move {mv.max():.4f})")
                    return True
        log(f"wait_still timeout at step {self.obs.step_id} (limit {limit})")
        return False

    def free_spot(self, exclude_xy=None):
        pw, ok = self.obs.world_map("front")
        z = pw[..., 2]
        m = ok & (z > 0.008) & (z < 0.095)
        pts = pw[m][:, :2]
        if exclude_xy is not None and len(pts):
            pts = pts[np.linalg.norm(pts - exclude_xy, axis=1) > 0.05]
        best, bestd = None, -1
        for x in np.arange(-0.30, 0.10, 0.02):
            for y in np.arange(-0.32, 0.33, 0.02):
                p = np.array([x, y])
                d = np.linalg.norm(p - BASE)
                if d > 0.64 or d < 0.36:
                    continue
                md = np.min(np.linalg.norm(pts - p, axis=1)) if len(pts) else 1.0
                md = min(md, 0.15)
                score = md - 0.05 * abs(y)
                if score > bestd:
                    bestd, best = score, p
        return best, bestd

    def revealed_color(self, xy):
        o = self.obs
        pw, ok = o.world_map("front")
        z = pw[..., 2]
        m = ok & (z > 0.008) & (z < 0.045) & (np.linalg.norm(pw[..., :2] - xy, axis=-1) < 0.045)
        if m.sum() < 10:
            return None, int(m.sum())
        rgb = o.rgb["front"][m].astype(float)
        sat = rgb.max(1) - rgb.min(1)
        dark = rgb.max(1) < 70
        sel = (sat > 60) | dark
        if sel.sum() < 8:
            return "none", int(sel.sum())
        return percep.color_name(rgb[sel].mean(0)), int(sel.sum())

    def pick(self, name):
        xy = self.T.pos(name).copy()
        log(f"pick {name}: tracked xy {np.round(xy,3)} step {self.obs.step_id}")
        res = None
        for attempt in range(3):
            # refine from current front view (arm parked high)
            res = grasp.refine_container(self.obs, xy, ee=None)
            if res is None:
                log("refine failed; widening")
                res = grasp.refine_container(self.obs, xy, r=0.05, ee=None)
            if res is None:
                log("refine failed again")
                c, yaw = xy, 0.0
            else:
                c, yaw, n = res
            log(f"  refine -> c={np.round(c,3)} yaw={math.degrees(yaw):.1f}")
            Rt = grasp.Rz(yaw) @ R0
            self.r.grip = -1.0
            self.r.move_to(clamp_target([c[0], c[1], 0.16]), tol=0.006, vmax=0.8, Rt=Rt, max_iter=8)
            self.r.move_to(clamp_target([c[0], c[1], 0.045]), tol=0.004, vmax=0.4, Rt=Rt, max_iter=8)
            self.r.set_grip(1.0, 10)
            w = self.obs.state[6] + self.obs.state[7]
            log(f"  closed width {w:.4f} at {np.round(self.r.pos,3)}")
            self.r.move_to(clamp_target([c[0], c[1], 0.20]), tol=0.008, vmax=0.5, Rt=Rt, max_iter=8)
            w = self.obs.state[6] + self.obs.state[7]
            col, npx = self.revealed_color(c)
            log(f"  lifted step {self.obs.step_id} width {w:.4f} revealed={col} ({npx}px) ended={self.obs.ended} complete={self.obs.complete}")
            if self.obs.ended:
                return True
            if 0.02 < w < 0.075:
                return True
            # missed: open and retry from a fresh view after moving up out of the way
            self.r.set_grip(-1.0, 6)
            self.r.move_to(clamp_target([-0.25, 0.0, 0.35]), tol=0.03, vmax=0.8, max_iter=6)
            xy = c
        return False

    def place_away(self, from_xy):
        spot, sc = self.free_spot(exclude_xy=from_xy)
        log(f"  place at {np.round(spot,3)} clearance {sc:.3f}")
        Rt = robo.rotvec_to_R(self.obs.state[3:6])
        self.r.move_to(clamp_target([spot[0], spot[1], 0.16]), tol=0.01, vmax=0.8, Rt=Rt, max_iter=8)
        self.r.move_to(clamp_target([spot[0], spot[1], 0.05]), tol=0.008, vmax=0.5, Rt=Rt, max_iter=6)
        self.r.set_grip(-1.0, 8)
        self.r.move_to(clamp_target([spot[0], spot[1], 0.18]), tol=0.01, vmax=0.8, Rt=Rt, max_iter=6)
        self.r.move_to(clamp_target([-0.15, 0.0, OVER_Z]), tol=0.03, vmax=0.8, max_iter=6)

def main(ctx):
    os.makedirs(OUT, exist_ok=True)
    c = Ctl(ctx)
    try:
        instr = c.obs.meta.get("instruction", "")
        order = parse_order(instr)
        log("instruction:", instr, "order:", order)
        buttons, cubes = detect_initial(c.obs)
        log("buttons:", [np.round(b["xyz"], 3).tolist() for b in buttons])
        log("cubes:", [(b["name"], np.round(b["xyz"], 3).tolist(), b["n"]) for b in cubes])
        c.T = tracker2.Tracker([b["xyz"] for b in cubes], [b["name"] for b in cubes])
        # Pressing very early (step ~42) ended an episode in failure; presses at steps 54-132 were safe.
        while not c.obs.ended and c.obs.step_id < PRESS_START:
            c.wait_tracked(1, chunk=1)
        c.press_buttons(buttons)
        log("after press step", c.obs.step_id, "lost", c.T.lost.tolist())
        c.wait_still()
        log("tracks:", {n: np.round(p, 3).tolist() for n, p in zip(c.T.names, c.T.c)}, "lost", c.T.lost.tolist(), "events", c.T.events)
        for name in order:
            if c.obs.ended:
                break
            if name not in c.T.names:
                log("target color not tracked:", name)
                continue
            from_xy = c.T.pos(name).copy()
            ok = c.pick(name)
            if c.obs.ended:
                break
            c.place_away(from_xy)
        log("final step", c.obs.step_id, "ended", c.obs.ended, "complete", c.obs.complete)
        if not c.obs.ended:
            c.r.hold(4)
            log("after hold ended", c.obs.ended, "complete", c.obs.complete)
    except Exception as e:
        log("EXC", repr(e), traceback.format_exc())
    finally:
        try:
            with open(os.path.join(OUT, "controller_log.txt"), "w") as f:
                f.write("\n".join(robo.LOG))
            if c.T is not None and c.T.hist:
                np.savez_compressed(os.path.join(OUT, "tracks.npz"), steps=np.array([h[0] for h in c.T.hist]),
                                    c=np.array([h[1] for h in c.T.hist]), cnt=np.array([h[2] for h in c.T.hist]), names=np.array(c.T.names))
        except Exception:
            pass
