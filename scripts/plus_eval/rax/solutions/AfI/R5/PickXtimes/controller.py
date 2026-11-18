"""PickXtimes controller: pick the named cube, place it on the target N times, then press the button."""
import json
import os
import re
import time
import traceback

import numpy as np

import percep
from robo import Robot, log, axis_angle_to_R, BASE_XY, R_MAX, reach
R_DESC = 0.66
LOOKOUT = (-0.15, 0.0, 0.40)

OPEN, CLOSE = -1.0, 1.0
NUM_WORDS = {"once": 1, "one": 1, "twice": 2, "two": 2, "three": 3, "thrice": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
COLORS = ["red", "green", "blue", "yellow"]

SAFE_Z = 0.15
GRASP_Z = 0.018
CUBE = 0.04


def parse_instruction(text):
    t = text.lower()
    color = next((c for c in COLORS if re.search(r"\b%s\b" % c, t)), "green")
    n = None
    m = re.search(r"(\d+)\s*(?:times|x\b)", t)
    if m:
        n = int(m.group(1))
    if n is None:
        for w, k in NUM_WORDS.items():
            if re.search(r"\b%s\b\s*(?:times)?" % w, t) and ("times" in t or w in ("once", "twice", "thrice")):
                # prefer word directly preceding 'times'
                if re.search(r"\b%s\s+times\b" % w, t) or w in ("once", "twice", "thrice"):
                    n = k
                    break
    if n is None:
        for w, k in NUM_WORDS.items():
            if re.search(r"\b%s\b" % w, t):
                n = k
                break
    if n is None:
        n = 1
    button = "button" in t
    return color, n, button


def cube_center(obs, color, cam="front", near=None):
    bl = percep.find_cubes(obs, cam=cam, color=color)
    if not bl:
        return None
    if near is not None:
        bl.sort(key=lambda b: np.linalg.norm(b["xyz"][:2] - np.asarray(near)[:2]))
    b = bl[0]
    pw = obs.world_points(cam)
    m = percep.color_mask(obs.rgb[cam], color) & np.isfinite(pw[..., 2])
    # restrict to this blob's neighbourhood
    vs, us = np.nonzero(m)
    sel = (np.abs(us - b["u"]) < 15) & (np.abs(vs - b["v"]) < 15)
    P = pw[vs[sel], us[sel]]
    P = P[P[:, 2] > 0.004]
    if len(P) == 0:
        return None
    ztop = np.percentile(P[:, 2], 95)
    top = P[P[:, 2] > ztop - 0.006]
    xy = top[:, :2].mean(0) if len(top) >= 3 else P[:, :2].mean(0)
    return np.array([xy[0], xy[1], ztop])


def target_center(obs, cam="front"):
    bl = percep.find_target(obs, cam=cam)
    if not bl:
        return None
    pw = obs.world_points(cam)
    m = percep.color_mask(obs.rgb[cam], "purple", smin=0.3, vmin=0.3) & np.isfinite(pw[..., 2])
    m &= pw[..., 2] < 0.03
    b = bl[0]
    vs, us = np.nonzero(m)
    sel = (np.abs(us - b["u"]) < 20) & (np.abs(vs - b["v"]) < 20)
    P = pw[vs[sel], us[sel]]
    return np.array([P[:, 0].mean(), P[:, 1].mean(), np.median(P[:, 2])])


def button_center(obs, cam="front"):
    bl = percep.find_button(obs, cam=cam)
    bl = [b for b in bl if b["zmax"] > 0.015]
    if not bl:
        return None
    b = bl[0]
    return np.array([b["xyz"][0], b["xyz"][1], b["zmax"]])


class Tracker:
    """Keeps (step, xy) detections of a moving object and predicts by linear fit."""
    def __init__(self, window=10):
        self.hist = []
        self.window = window

    def add(self, step, xy):
        self.hist.append((step, np.asarray(xy[:2], dtype=float)))
        self.hist = self.hist[-40:]

    def velocity(self):
        if not self.hist:
            return np.zeros(2)
        last = self.hist[-1][0]
        pts = [(s, p) for s, p in self.hist if last - s <= self.window]
        if len(pts) < 2 or pts[-1][0] - pts[0][0] < 2:
            return np.zeros(2)
        S = np.array([s for s, _ in pts], float)
        P = np.array([p for _, p in pts])
        A = np.stack([S - S.mean(), np.ones_like(S)], 1)
        coef, *_ = np.linalg.lstsq(A, P, rcond=None)
        v = coef[0]
        sp = np.linalg.norm(v)
        if sp > 0.02:
            v = v / sp * 0.02
        return v

    def predict(self, step):
        if not self.hist:
            return None
        s0, p0 = self.hist[-1]
        return p0 + self.velocity() * (step - s0)

    def age(self, step):
        return step - self.hist[-1][0] if self.hist else 1e9


def _target_from(obs, cam, min_px):
    pw = obs.world_points(cam)
    m = percep.color_mask(obs.rgb[cam], "purple", smin=0.3, vmin=0.3) & np.isfinite(pw[..., 2])
    m &= (pw[..., 2] < 0.03) & (pw[..., 2] > -0.02)
    if m.sum() < min_px:
        return None, int(m.sum())
    lab, n = percep.label(m)
    sizes = [(lab == k).sum() for k in range(1, n + 1)]
    k = int(np.argmax(sizes)) + 1
    P0 = pw[lab == k]
    c0 = P0[:, :2].mean(0)
    P = pw[m]
    sel = np.linalg.norm(P[:, :2] - c0, axis=1) < 0.05
    P = P[sel]
    return np.array([P[:, 0].mean(), P[:, 1].mean(), np.median(P[:, 2])]), int(sel.sum())


def detect_target_xy(obs):
    """Moving bullseye: prefer the wrist view (unoccluded from above), else the front view."""
    det, n = None, 0
    if "wrist" in obs.depth:
        det, n = _target_from(obs, "wrist", 40 if obs.state[2] > 0.2 else 150)
        if det is not None:
            return det, n
    return _target_from(obs, "front", 40)


def _cube_from(obs, cam, color, min_px):
    pw = obs.world_points(cam)
    m = percep.color_mask(obs.rgb[cam], color) & np.isfinite(pw[..., 2])
    m &= (pw[..., 2] > 0.004) & (pw[..., 2] < 0.10)
    if m.sum() < min_px:
        return None, int(m.sum())
    lab, n = percep.label(m)
    sizes = [(lab == k).sum() for k in range(1, n + 1)]
    k = int(np.argmax(sizes)) + 1
    P = pw[lab == k]
    ztop = np.percentile(P[:, 2], 95)
    top = P[P[:, 2] > ztop - 0.004]
    xy = np.median(top[:, :2], axis=0) if len(top) >= 3 else np.median(P[:, :2], axis=0)
    return np.array([xy[0], xy[1], ztop]), int(len(P))


def detect_cube_xy(obs, color):
    if "wrist" in obs.depth:
        det, n = _cube_from(obs, "wrist", color, 120)
        if det is not None:
            return det, n
    return _cube_from(obs, "front", color, 20)


class Task:
    def __init__(self, ctx):
        self.r = Robot(ctx)
        self.r.observe()
        self.instruction = self.r.obs.instruction
        self.color, self.n, self.button = parse_instruction(self.instruction)
        log("instruction:", self.instruction, "->", self.color, self.n, self.button)
        self.done = 0
        self.target = None
        self.btn = None
        self.home = self.r.pos()
        self.verbose = False
        self.priors = {}
        try:
            self.see(self.r.obs)
        except Exception:
            pass
        self.r.R_des = axis_angle_to_R([np.pi, 0, 0])

    def see(self, o):
        """Update target / cube priors from an observation (used when tracking loses sight)."""
        t, n = detect_target_xy(o)
        if t is not None:
            self.priors["target"] = np.array(t[:2])
        c, n = detect_cube_xy(o, self.color)
        if c is not None:
            self.priors["cube"] = np.array(c[:2])
        log("see: priors", {k: np.round(v, 3).tolist() for k, v in self.priors.items()})

    def go_safe(self, grip):
        p = self.r.pos()
        if p[2] < SAFE_Z - 0.01:
            self.r.move_to([p[0], p[1], SAFE_Z], grip, tol=0.01, max_steps=30)

    def look(self):
        """Observe the scene from the front camera (arm raised)."""
        return self.r.observe()

    def pick(self, attempt=0):
        r = self.r
        o = self.look()
        c = cube_center(o, self.color, near=self.target)
        if c is None:
            log("cube not found")
            return False
        log("cube at", c.round(3))
        r.move_to([c[0], c[1], SAFE_Z], OPEN, tol=0.006, max_steps=60)
        r.move_to([c[0], c[1], max(GRASP_Z, c[2] - 0.022)], OPEN, tol=0.004, max_steps=40, speed=0.6)
        r.hold(CLOSE, 12)
        r.move_to([c[0], c[1], SAFE_Z], CLOSE, tol=0.01, max_steps=40)
        w = r.grip_open()
        log("after lift, grip opening %.4f pos %s" % (w, r.pos().round(3)))
        return w > 0.01

    def place(self):
        r = self.r
        t = self.target
        r.move_to([t[0], t[1], SAFE_Z], CLOSE, tol=0.006, max_steps=60)
        r.move_to([t[0], t[1], t[2] + CUBE / 2 + 0.008], CLOSE, tol=0.004, max_steps=40, speed=0.6)
        r.hold(OPEN, 8)
        r.move_to([t[0], t[1], SAFE_Z], OPEN, tol=0.01, max_steps=40)
        log("placed, pos", r.pos().round(3))
        return True

    def track_to(self, detect, z_hover, z_final, grip, max_steps=160, xy_tol=0.006, chunk=2,
                 desc_gate=0.015, desc_rate=0.015, name="obj", freeze_z=0.085, reach_gate=None):
        """Follow a (possibly moving) object seen by front/wrist RGB-D and descend onto it.
        Descends only when aligned (and, with reach_gate, when the object stays inside that
        reach radius from the base for the next ~10 steps). Returns (ok, tracker)."""
        r = self.r
        tr = Tracker()
        used = 0
        lost = 0
        rej = 0
        pen0 = r.penalties
        while used < max_steps and not r.ended:
            if r.penalties > pen0:
                # IK trouble: back off toward the base and up, then give up this attempt
                p = r.pos()
                d = BASE_XY - p[:2]
                d = d / (np.linalg.norm(d) + 1e-9) * 0.06
                r.move_to([p[0] + d[0], p[1] + d[1], max(p[2], 0.12)], grip, tol=0.01, max_steps=12)
                return False, tr
            o = r.obs
            step = o.step_id
            det, npx = detect(o)
            if name != "target" and r.pos()[2] >= freeze_z:
                try:
                    tt, _ = detect_target_xy(o)
                    if tt is not None:
                        self.priors["target"] = np.array(tt[:2])
                except Exception:
                    pass
            if r.pos()[2] < freeze_z and tr.hist:
                det = None  # close-range views are biased/occluded: coast on prediction
            if det is not None and len(tr.hist) >= 3:
                pr = tr.predict(step)
                if np.linalg.norm(det[:2] - pr) > 0.025 and rej < 3:
                    rej += 1
                    det = None
                else:
                    rej = 0
            if det is not None:
                tr.add(step, det)
                self.priors[name] = np.array(det[:2])
                lost = 0
            else:
                lost += 1
            if not tr.hist or (tr.age(step) > 12 and r.pos()[2] >= freeze_z):
                p = r.pos()
                prior = self.priors.get(name)
                if prior is not None and lost < 14:
                    goal = [prior[0], prior[1], max(z_hover, min(p[2], 0.30))]
                elif prior is not None and lost < 28:
                    goal = [prior[0], prior[1], 0.30]
                else:
                    goal = list(LOOKOUT)
                if self.verbose or lost % 6 == 1:
                    log("search", name, "lost", lost, "goal", np.round(goal, 3))
                r.step(r.plan_rows([goal] * chunk, grip))
                used += chunk
                if lost > 45:
                    return False, tr
                continue
            p = r.pos()
            now = tr.predict(step)
            herr = float(np.linalg.norm(now - p[:2]))
            reachable = True
            if reach_gate is not None:
                reachable = reach(now) < reach_gate and reach(tr.predict(step + 10)) < reach_gate
            if self.verbose:
                log("trk", step, "p", p.round(3), "now", now.round(3), "herr %.4f" % herr, "npx", npx,
                    "reach %.3f" % reach(now))
            if p[2] <= z_final + 0.004 and herr < xy_tol and reachable:
                return True, tr
            if herr < desc_gate and reachable:
                z_des = max(z_final, p[2] - desc_rate * chunk)
            elif (herr > 2 * desc_gate or not reachable) and p[2] < z_hover - 0.01:
                z_des = min(z_hover, p[2] + 0.02 * chunk)
            else:
                z_des = p[2]
            tg = []
            for k in range(chunk):
                xy = tr.predict(step + k + 1)
                zz = p[2] + (z_des - p[2]) * (k + 1) / chunk
                tg.append([xy[0], xy[1], zz])
            r.step(r.plan_rows(tg, grip))
            used += chunk
        return False, tr

    def follow_rows(self, tr, n, grip, dz=0.0):
        """Rows that keep moving with the tracked object (clamped to the workspace)."""
        p = self.r.pos()
        s = self.r.obs.step_id
        tg = []
        for k in range(n):
            xy = tr.predict(s + k + 1) if tr.hist else p[:2]
            tg.append([xy[0], xy[1], p[2] + dz * (k + 1)])
        return self.r.plan_rows(tg, grip)

    def pick_tracked(self):
        r = self.r
        p = r.pos()
        if p[2] < 0.10:
            r.move_to([p[0], p[1], 0.12], OPEN, tol=0.01, max_steps=20)
        c = cube_center(r.obs, self.color)
        if c is None:
            c, _ = detect_cube_xy(r.obs, self.color)
        if c is None:
            r.move_to(self.home, OPEN, tol=0.02, max_steps=30)
            c, _ = detect_cube_xy(r.obs, self.color)
            if c is None:
                log("pick: cube not visible")
                return False
        zg = max(GRASP_Z, c[2] - 0.022)
        det = lambda o: detect_cube_xy(o, self.color)
        ok, tr = self.track_to(det, z_hover=0.12, z_final=zg, grip=OPEN, xy_tol=0.006,
                               desc_rate=0.012, name="cube", reach_gate=R_MAX - 0.005)
        log("pick track ok", ok, "step", r.obs.step_id, "pos", r.pos().round(3), "v", tr.velocity().round(4))
        if not ok:
            p = r.pos()
            r.move_to([p[0], p[1], 0.12], OPEN, tol=0.01, max_steps=20)
            return False
        r.step(self.follow_rows(tr, 10, CLOSE))
        r.step(self.follow_rows(tr, 3, CLOSE, dz=0.01))
        p = r.pos()
        r.move_to([p[0], p[1], SAFE_Z], CLOSE, tol=0.01, max_steps=30)
        w = r.grip_open()
        log("after lift, grip opening %.4f pos %s" % (w, r.pos().round(3)))
        return w > 0.01

    def place_tracked(self, tries=4):
        """Carry the held cube over the moving target, descend while tracking, release.
        Returns True if released over the target, False if the cube was dropped/never released."""
        r = self.r
        for attempt in range(tries):
            if r.ended:
                return False
            p = r.pos()
            if p[2] < 0.10:
                r.move_to([p[0], p[1], 0.10], CLOSE, tol=0.01, max_steps=20)
            if r.grip_open() < 0.01:
                log("place: cube lost before release")
                return False
            ok, tr = self.track_to(detect_target_xy, z_hover=0.10, z_final=0.009 + CUBE / 2 + 0.012,
                                   grip=CLOSE, name="target", reach_gate=R_DESC, max_steps=200)
            log("place track ok", ok, "attempt", attempt, "step", r.obs.step_id, "pos", r.pos().round(3),
                "pred", None if not tr.hist else tr.predict(r.obs.step_id).round(3), "v", tr.velocity().round(4))
            if not ok:
                continue
            r.step(self.follow_rows(tr, 5, OPEN))
            r.step(self.follow_rows(tr, 4, OPEN, dz=0.005))
            p = r.pos()
            r.move_to([p[0], p[1], SAFE_Z], OPEN, tol=0.01, max_steps=30)
            return True
        # could not place: put it down where we are (still counts nothing)
        p = r.pos()
        log("place: giving up, releasing in place")
        r.move_to([p[0], p[1], 0.06], CLOSE, tol=0.01, max_steps=20)
        r.hold(OPEN, 6)
        r.move_to([p[0], p[1], SAFE_Z], OPEN, tol=0.01, max_steps=30)
        return False

    def check_on_target(self):
        """After release: is the cube resting on the target? (xy distance in current obs)"""
        o = self.r.obs
        c, _ = detect_cube_xy(o, self.color)
        t, n = detect_target_xy(o)
        if c is None or t is None:
            log("verify: cube", c, "target", t)
            return None
        d = float(np.linalg.norm(c[:2] - t[:2]))
        log("verify: cube-target dist %.3f cube z %.3f" % (d, c[2]))
        return d

    def find_button(self):
        """Refine the button cap position from a wrist view above it."""
        r = self.r
        b = self.btn
        r.move_to([b[0] - 0.04, b[1], 0.16], CLOSE, tol=0.01, max_steps=60)
        best = None
        for cam in ("wrist", "front"):
            o = r.obs
            pw = o.world_points(cam)
            h, s, v = percep.hsv(o.rgb[cam])
            m = (s < 0.2) & (v > 0.3) & (pw[..., 2] > 0.004) & (pw[..., 2] < 0.08)
            m &= (np.abs(pw[..., 0] - b[0]) < 0.06) & (np.abs(pw[..., 1] - b[1]) < 0.06)
            if m.sum() < 15:
                continue
            P = pw[m]
            zt = np.percentile(P[:, 2], 97)
            top = P[P[:, 2] > zt - 0.006]
            best = np.array([np.median(top[:, 0]), np.median(top[:, 1]), zt])
            log("button from", cam, best.round(3), "px", int(m.sum()))
            break
        if best is not None and np.linalg.norm(best[:2] - b[:2]) < 0.05:
            self.btn = best

    def press(self):
        r = self.r
        r.hold(CLOSE, 6)
        self.find_button()
        b = self.btn
        log("pressing button at", b.round(3))
        r.move_to([b[0], b[1], 0.12], CLOSE, tol=0.005, max_steps=40)
        r.move_to([b[0], b[1], b[2] + 0.02], CLOSE, tol=0.004, max_steps=30, speed=0.5)
        for depth in (0.012, 0.02, 0.028):
            if r.ended:
                break
            r.move_to([b[0], b[1], b[2] - depth], CLOSE, tol=0.003, max_steps=14, speed=0.3)
            r.hold(CLOSE, 4)
            log("press depth", depth, "pos", r.pos().round(3), "ended", r.ended, "success", r.success)
            if r.ended:
                break
            r.move_to([b[0], b[1], b[2] + 0.03], CLOSE, tol=0.005, max_steps=15)
            r.hold(CLOSE, 3)
            if r.ended:
                break
        if not r.ended:
            r.move_to([b[0], b[1], SAFE_Z], CLOSE, tol=0.01, max_steps=30)


def run(task, n_cycles):
    r = task.r
    done = 0
    fails = 0
    while done < n_cycles and not r.ended and fails < 8:
        rem = r.obs.remaining
        if rem is not None and rem < 260:
            log("low steps", rem)
            break
        ok = task.pick_tracked()
        if not ok:
            fails += 1
            log("pick failed", fails)
            continue
        okp = task.place_tracked()
        if okp:
            done += 1
        else:
            fails += 1
        log("cycle", done, "of", n_cycles, "placed_ok", okp, "step", r.obs.step_id,
            "penalties", r.penalties)
    return done


def main(ctx):
    t0 = time.time()
    task = Task(ctx)
    o = task.r.obs
    task.btn = button_center(o)
    log("button", None if task.btn is None else task.btn.round(3))
    done = 0
    try:
        done = run(task, task.n)
    except Exception:
        log("run error:\n" + traceback.format_exc())
        try:
            task.r.observe()
        except Exception:
            pass
    try:
        if task.button and not task.r.ended and task.btn is not None and done >= task.n:
            task.press()
        elif done < task.n:
            log("did not complete all cycles; not pressing")
    except Exception:
        log("press error:\n" + traceback.format_exc())
    log("finished: cycles", done, "of", task.n, "ended", task.r.ended, "success", task.r.success,
        "step", task.r.obs.step_id, "penalties", task.r.penalties, "wall %.1f" % (time.time() - t0))
    try:
        from robo import LOG
        os.makedirs("/workspace/output", exist_ok=True)
        with open("/workspace/output/controller_log.txt", "w") as f:
            f.write("\n".join(LOG[-3000:]))
    except Exception:
        pass
