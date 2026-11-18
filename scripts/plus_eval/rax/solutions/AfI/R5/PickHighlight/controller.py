"""PickHighlight controller v3.

Task facts (learned in exploration episodes 1-3):
- highlight = white discs under target cubes for ~25-40 steps starting ~step 10;
- cubes swap positions pairwise (sliding on table) periodically (~every 80 steps, ~30 steps long)
  through the whole episode -> identities must be tracked continuously;
- lifting a non-target cube (or picking after a bad press) ends the episode in failure;
- one gentle button press before picking, one after the last pick.

Every robot step goes through World.step(), which re-detects cubes (wrist + front, height based)
and associates them to tracks, so identities are maintained during all motion.
Picks are started right after a swap phase ends so that they complete in the quiet window.
"""
import os, sys, time, json, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib
from rlib import log

OUT = "/workspace/output"
OBS_POSE = [-0.13, 0.0, 0.30]
Z_HOVER = 0.075
Z_GRASP = 0.018
Z_LIFT = 0.10


class World:
    def __init__(self, ctx, o):
        self.ctx = ctx
        self.o = o
        cubes = [c for c in rlib.cube_tops2(o, "wrist") if c["n"] > 25 and max(c["size"]) < 0.06]
        self.cap = rlib.find_cap2(o, cubes, "wrist") or rlib.find_cap2(o, cubes, "front")
        self.tracks = [dict(xy=c["xy"].copy(), yaw=c["yaw"], last=o.step_id, moved=0) for c in cubes]
        self.score = np.zeros(len(cubes))
        self.front_base = None
        self.hl_first = None
        self.hl_last = None
        self.last_motion = None
        self.motion_steps = []
        self.held = None  # index of cube in the gripper (excluded from association)
        self.update(o)

    # ---------- perception ----------
    def detections(self, o):
        dets = []
        tcp = o.state[:3]
        if o.world_points is not None:
            for cam in ("wrist", "front"):
                if cam == "wrist" and tcp[2] < 0.06:
                    continue
                try:
                    ds = rlib.cube_tops2(o, cam)
                except Exception:
                    continue
                for d in ds:
                    if d["n"] < (25 if cam == "wrist" else 15) or max(d["size"]) > 0.055 or min(d["size"]) < 0.02:
                        continue
                    # ignore anything at/under the gripper when it is low (held cube, fingers)
                    if tcp[2] < 0.16 and np.hypot(*(d["xy"] - tcp[:2])) < 0.045:
                        continue
                    if any(np.hypot(*(d["xy"] - e["xy"])) < 0.02 for e in dets):
                        continue  # duplicate (wrist first, preferred)
                    d["cam"] = cam
                    dets.append(d)
        return dets

    def update_highlight(self, o):
        C = [t["xy"] for t in self.tracks]
        active = False
        for cam in ("front", "wrist"):
            if cam == "wrist" and o.state[2] < 0.25:
                continue
            try:
                r = rlib.ring_white(o, cam, C, exclude=(self.cap["xy"] if self.cap else None))
            except Exception:
                continue
            if cam == "front" and self.front_base is None:
                self.front_base = np.array([f if np.isfinite(f) else 0.0 for f, n in r])
            for i, (f, n) in enumerate(r):
                if not np.isfinite(f):
                    continue
                if cam == "front":
                    f = f - self.front_base[i]
                self.score[i] = max(self.score[i], f)
                if f > 0.35:
                    active = True
        if active:
            self.hl_last = o.step_id
            if self.hl_first is None:
                self.hl_first = o.step_id

    def update(self, o):
        from scipy.optimize import linear_sum_assignment
        self.o = o
        if self.hl_last is None or o.step_id - self.hl_last < 15:
            self.update_highlight(o)
        dets = self.detections(o)
        idx = [i for i in range(len(self.tracks)) if i != self.held]
        moved_any = False
        if dets and idx:
            T = np.array([self.tracks[i]["xy"] for i in idx]); D = np.array([d["xy"] for d in dets])
            Cm = np.linalg.norm(T[:, None, :] - D[None, :, :], axis=2)
            r, c = linear_sum_assignment(Cm)
            for a, b in zip(r, c):
                if Cm[a, b] < 0.03:
                    t = self.tracks[idx[a]]
                    if Cm[a, b] > 0.0045:
                        moved_any = True
                        t["moved"] += 1
                    # wrist detections more accurate; front ok when static
                    t["xy"] = D[b].copy(); t["yaw"] = dets[b]["yaw"]; t["last"] = o.step_id
        if moved_any:
            if self.last_motion is None or o.step_id - self.last_motion > 12:
                self.motion_steps.append(o.step_id)  # start of a swap phase
            self.last_motion = o.step_id
        return o

    # ---------- motion ----------
    def step(self, rows):
        o, res = rlib.step(self.ctx, rows)
        self.update(o)
        return o

    def move_to(self, target, grip, vmax=0.9, tol=0.006, max_steps=60, yaw=None, chunk=2, gain=1.8):
        """Closed loop: each row commands gain*(remaining error) (realized motion ~0.45 of command)."""
        target = np.array(target, float)
        used = 0
        while used < max_steps and not self.o.ended:
            pos = self.o.state[:3]
            e = target - pos
            if np.linalg.norm(e) < tol:
                break
            aa_t = rlib.DOWN_AA if yaw is None else rlib.yaw_aa(yaw)
            r = rlib.rot_err(self.o.state[3:6], aa_t)
            k = chunk if np.linalg.norm(e) > 0.03 else 1
            d = gain * e / 0.05 / k
            n = np.max(np.abs(d))
            if n > vmax:
                d = d * (vmax / n)
            rr = np.clip(1.5 * r / 0.5, -0.4, 0.4)
            rows = [list(d) + list(rr / k) + [grip]] * k
            self.step(rows)
            used += len(rows)
        return self.o

    def hold(self, n, grip):
        for _ in range(n):
            if self.o.ended:
                break
            self.step([[0, 0, 0, 0, 0, 0, grip]])
        return self.o

    def rotate_to(self, aa_t, grip, max_steps=12, tol=0.04):
        for _ in range(max_steps):
            r = rlib.rot_err(self.o.state[3:6], aa_t)
            if np.linalg.norm(r) < tol or self.o.ended:
                break
            self.step([[0, 0, 0] + list(np.clip(r / 0.5, -0.4, 0.4)) + [grip]])
        return self.o

    # ---------- skills ----------
    def press_button(self, grip=1.0):
        x, y = self.cap["xy"]; ztop = self.cap["z"]
        self.move_to([x, y, ztop + 0.06], grip, vmax=0.8, tol=0.006)
        self.move_to([x, y, ztop + 0.012], grip, vmax=0.3, tol=0.004, chunk=1)
        zs = [self.o.state[2]]
        for _ in range(25):
            self.step([[0, 0, -0.15, 0, 0, 0, grip]])
            zs.append(self.o.state[2])
            if self.o.ended:
                return self.o
            if len(zs) > 3 and zs[-3] - zs[-1] < 0.0015 and zs[-1] < ztop - 0.002:
                break
            if zs[-1] < ztop - 0.02:
                break
        log("press: min z", round(min(zs), 4), "cap top", round(ztop, 4), "step", self.o.step_id)
        self.hold(3, grip)
        self.move_to([x, y, ztop + 0.07], grip, vmax=0.5, tol=0.01)
        return self.o

    def wait_quiet(self, max_wait=140, quiet=4, need_motion=True):
        """Hold until a swap phase has ended (motion seen, then `quiet` steps without motion)."""
        start = self.o.step_id
        seen = False
        while not self.o.ended and self.o.step_id - start < max_wait:
            self.hold(1, -1.0)
            if self.last_motion is not None and self.last_motion > start:
                seen = True
            if seen and self.o.step_id - self.last_motion >= quiet:
                return True
            if not need_motion and (self.last_motion is None or self.o.step_id - self.last_motion >= quiet):
                return True
        return False

    def pick(self, i):
        t = self.tracks[i]
        xy = t["xy"].copy()
        cy = rlib.current_yaw(self.o)
        yaw_t = min([t["yaw"] + k * np.pi / 2 for k in range(-3, 4)],
                    key=lambda a: abs(((a - cy + np.pi) % (2 * np.pi)) - np.pi))
        self.move_to([xy[0], xy[1], Z_HOVER], -1.0, vmax=0.9, tol=0.008, yaw=yaw_t)
        xy = self.tracks[i]["xy"].copy()
        self.rotate_to(rlib.yaw_aa(yaw_t), -1.0)
        xy = self.tracks[i]["xy"].copy()
        self.move_to([xy[0], xy[1], Z_HOVER], -1.0, vmax=0.4, tol=0.004, yaw=yaw_t)
        xy = self.tracks[i]["xy"].copy()
        self.move_to([xy[0], xy[1], Z_GRASP], -1.0, vmax=0.5, tol=0.005, yaw=yaw_t, chunk=2)
        self.held = i
        self.hold(8, 1.0)
        w0 = float(self.o.state[6] + self.o.state[7])
        if w0 < 0.015:
            log("pick", i, "empty grasp width", w0)
            self.hold(4, -1.0)
            self.held = None
            self.move_to([xy[0], xy[1], Z_HOVER], -1.0, vmax=0.6, tol=0.01)
            return False
        self.move_to([xy[0], xy[1], Z_LIFT], 1.0, vmax=0.7, tol=0.01)
        self.hold(2, 1.0)
        w = float(self.o.state[6] + self.o.state[7])
        ok = w > 0.015
        log("pick", i, "at", np.round(xy, 3), "lift z", round(self.o.state[2], 3), "width", round(w, 4), "step", self.o.step_id)
        # put it back where it was
        self.move_to([xy[0], xy[1], Z_GRASP + 0.008], 1.0, vmax=0.6, tol=0.006)
        self.hold(4, -1.0)
        self.held = None
        self.tracks[i]["xy"] = xy.copy()
        self.move_to([xy[0], xy[1], Z_HOVER + 0.03], -1.0, vmax=0.7, tol=0.01)
        return ok


def main(ctx):
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    summary = {"version": "v3"}
    W = None
    try:
        o = rlib.observe(ctx)
        log("start step", o.step_id, "instr", o.meta.get("instruction"))
        if o.state[2] < 0.45:
            o = rlib.move_to(ctx, o, [-0.06, 0.0, 0.52], grip=-1.0)
        W = World(ctx, o)
        log("cubes", [t["xy"].round(3).tolist() for t in W.tracks], "cap", W.cap)
        # 1) watch highlight + first swap phase from home (best view)
        while not W.o.ended and W.o.step_id < 160:
            W.hold(1, -1.0)
            if W.hl_last is not None and W.o.step_id - W.hl_last > 5:
                break
        log("highlight", W.hl_first, W.hl_last, "scores", W.score.round(2).tolist())
        targets = [i for i in range(len(W.tracks)) if W.score[i] > 0.3]
        summary.update(hl_first=W.hl_first, hl_last=W.hl_last, scores=W.score.tolist(), targets=targets)
        W.wait_quiet(max_wait=120)
        log("swap1 done step", W.o.step_id, "motion starts", W.motion_steps)
        # 2) press button in the quiet window
        W.press_button()
        summary["press1_step"] = W.o.step_id
        W.move_to(OBS_POSE, -1.0, vmax=0.8, tol=0.02)
        # 3) picks, one per quiet window
        remaining = list(targets)
        picked = []
        fails = {}
        while remaining and not W.o.ended and W.o.step_id < 1900:
            W.move_to(OBS_POSE, -1.0, vmax=0.8, tol=0.02)
            W.wait_quiet(max_wait=110)
            if W.o.ended:
                break
            cur = W.o.state[:2]
            i = min(remaining, key=lambda k: np.hypot(*(W.tracks[k]["xy"] - cur)))
            ok = W.pick(i)
            log("picked", i, ok, "step", W.o.step_id, "t", round(time.time() - t0, 1), "motion", W.motion_steps[-3:])
            if ok:
                remaining.remove(i)
                picked.append((i, W.o.step_id))
            else:
                fails[i] = fails.get(i, 0) + 1
                if fails[i] >= 3:
                    remaining.remove(i)
        summary["picked"] = picked
        # 4) final press
        if not W.o.ended:
            W.press_button()
            summary["press2_step"] = W.o.step_id
            W.hold(6, 1.0)
        summary["ended"] = W.o.ended
        summary["final_step"] = W.o.step_id
        summary["motion_steps"] = W.motion_steps
        log("done step", W.o.step_id, "ended", W.o.ended, "t", round(time.time() - t0, 1))
    except Exception:
        summary["error"] = traceback.format_exc()
        log("ERROR", traceback.format_exc())
    finally:
        summary["wall_s"] = time.time() - t0
        try:
            with open(os.path.join(OUT, "summary.json"), "w") as f:
                json.dump(summary, f, default=str, indent=1)
            with open(os.path.join(OUT, "log.txt"), "w") as f:
                f.write("\n".join(rlib.LOG))
        except Exception:
            pass
