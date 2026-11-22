"""ButtonUnmask controller v1: press button, track cubes until covered, lift containers in instructed order."""
import os, sys, time, json, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot import Robot
from perception import parse_order, find_cubes, BANDS, components

OUT = "/workspace/output"
PRESS_START = 22
QUIET = 15
NEWGAP = 75
WAIT_CAP = 450
ALL_COLORS = list(BANDS.keys())


class Logger:
    def __init__(self):
        os.makedirs(OUT, exist_ok=True)
        self.f = open(os.path.join(OUT, "controller_log.txt"), "a")
        self.t0 = time.time()

    def __call__(self, *a):
        s = "[%7.1fs] " % (time.time() - self.t0) + " ".join(str(x) for x in a)
        print(s)
        try:
            self.f.write(s + "\n"); self.f.flush()
        except Exception:
            pass


class Tracker:
    """Accumulate cube sightings per colour as 'sites' (xy clusters)."""
    def __init__(self, log):
        self.sites = {c: [] for c in ALL_COLORS}  # dict(xy, n, first, last, pts)
        self.visible_now = {c: [] for c in ALL_COLORS}
        self.last_any_visible = -1
        self.log = log
        self.exclude = []  # world xy points (button) to ignore

    def update(self, robot):
        res = {k: [] for k in ALL_COLORS}
        for cam in ("front", "wrist"):
            try:
                rr = find_cubes(robot, ALL_COLORS, cam)
                for k, v in rr.items():
                    res[k].extend(v)
            except Exception as e:
                self.log("perception error", cam, e)
        step = robot.step_id
        any_vis = False
        for c, blobs in res.items():
            self.visible_now[c] = []
            for xyz, uv, area in blobs:
                if any(np.linalg.norm(xyz[:2] - e) < 0.045 for e in self.exclude):
                    continue
                any_vis = True
                site = None
                for s in self.sites[c]:
                    if np.linalg.norm(s["xy"] - xyz[:2]) < 0.035:
                        site = s; break
                if site is None:
                    site = {"xy": xyz[:2].copy(), "pts": [], "first": step, "last": step, "area": area}
                    self.sites[c].append(site)
                    self.log("new site", c, np.round(xyz, 3), "uv", np.round(uv, 1), "area", area, "step", step)
                if any(s2 is site for s2 in self.visible_now[c]):
                    continue
                site["pts"].append(xyz[:2]); site["pts"] = site["pts"][-15:]
                site["xy"] = np.median(np.array(site["pts"]), axis=0)
                site["last"] = step; site["area"] = max(site["area"], area)
                self.visible_now[c].append(site)
        if any_vis:
            self.last_any_visible = step

    def covered_sites(self, color, step):
        return sorted([s for s in self.sites[color] if s["last"] < step - 3 and len(s["pts"]) >= 3], key=lambda s: s["last"])

    def covered_site(self, color, step):
        cand = [s for s in self.sites[color] if s["last"] < step - 3 and len(s["pts"]) >= 3]
        if not cand:
            cand = [s for s in self.sites[color] if s["last"] < step - 3]
        if not cand:
            return None
        return max(cand, key=lambda s: s["last"])


def make_robot(ctx, log, tracker):
    r = Robot(ctx, log)
    orig_step = r.step

    def step(rows):
        m = orig_step(rows)
        tracker.update(r)
        return m
    r.step = step
    return r


def _button_candidates(r, cam):
    D = r.depth(cam)
    vs, us = np.mgrid[0:256, 0:256]
    P = r.deproject(cam, us.ravel(), vs.ravel(), D.ravel()).reshape(256, 256, 3)
    m = (D > 0) & (P[..., 2] > 0.008) & (P[..., 2] < 0.06) & (P[..., 0] > -0.45) & (P[..., 0] < 0.3)
    out = []
    for cu, cv in components(m, 10):
        pts = P[cv, cu]
        ext = pts[:, :2].max(0) - pts[:, :2].min(0)
        if ext.max() > 0.08:
            continue
        # neighbourhood must not be part of a tall object: check ring points
        xy = np.median(pts[:, :2], axis=0)
        top = float(np.percentile(pts[:, 2], 90))
        out.append((len(cu), xy, top, ext))
    return out


def find_button(r, log):
    for cam in ("wrist", "front"):
        try:
            cands = _button_candidates(r, cam)
        except Exception as e:
            log("button detect error", cam, e); cands = []
        cands = [c for c in cands if 0.02 < c[2] < 0.05]
        if cands:
            n, xy, top, ext = max(cands, key=lambda c: c[0])
            log("button via", cam, np.round(xy, 3), "top", round(top, 3), "npix", n, "ext", np.round(ext, 3), "ncand", len(cands))
            return np.array([xy[0], xy[1], top])
    return None


def press_button(r, btn, log):
    r.move_to([btn[0], btn[1], btn[2] + 0.12], 1, tol=0.01, max_steps=40)
    b2 = find_button(r, log)
    if b2 is not None and np.linalg.norm(b2[:2] - btn[:2]) < 0.04:
        btn = b2
        r.move_to([btn[0], btn[1], btn[2] + 0.08], 1, tol=0.008, max_steps=20)
    r.move_to([btn[0], btn[1], btn[2] + 0.03], 1, tol=0.006, max_steps=15)
    # descend slowly until stall
    zs = []
    for i in range(25):
        if r.ended:
            break
        target = btn[2] - 0.02
        err = np.r_[btn[0] - r.state[0], btn[1] - r.state[1], target - r.state[2]]
        a = np.clip(err / 0.05, -0.5, 0.5)
        r.step([list(a) + r.rot_correction() + [1]])
        zs.append(r.state[2])
        if len(zs) >= 4 and abs(zs[-1] - zs[-4]) < 0.001:
            break
    log("press bottom z", round(r.state[2], 4), "step", r.step_id)
    r.hold(3, 1) if False else None
    for _ in range(3):
        r.step([[0, 0, -0.3] + r.rot_correction() + [1]])
    log("pressed; z", round(r.state[2], 4), "step", r.step_id)


def wrist_refine(r, exp_xy, log, radius=0.05):
    D = r.depth("wrist")
    vs, us = np.mgrid[0:256, 0:256]
    P = r.deproject("wrist", us.ravel(), vs.ravel(), D.ravel()).reshape(256, 256, 3)
    m = (D > 0) & (P[..., 2] > 0.062) & (P[..., 2] < 0.095)
    best = None
    for cu, cv in components(m, 15):
        pts = P[cv, cu]
        xy = np.median(pts[:, :2], axis=0)
        d = np.linalg.norm(xy - exp_xy)
        if d < radius and (best is None or d < best[0]):
            best = (d, xy, len(cu), float(np.median(pts[:, 2])))
    if best is None:
        log("wrist refine: no nub near", np.round(exp_xy, 3))
        return None
    log("wrist refine: nub", np.round(best[1], 3), "dist", round(best[0], 3), "npix", best[2], "ztop", round(best[3], 3))
    return best[1], best[3]


def pick_container(r, xy0, log, hover=0.17):
    xy = np.array(xy0, float)
    r.move_to([xy[0], xy[1], hover], -1, tol=0.008, max_steps=50)
    ztop = 0.073
    for it in range(2):
        ref = wrist_refine(r, xy, log)
        if ref is None:
            break
        xy, ztop = ref
        r.move_to([xy[0], xy[1], hover], -1, tol=0.005, max_steps=25)
    zg = ztop - 0.013
    r.move_to([xy[0], xy[1], zg + 0.03], -1, tol=0.006, max_steps=25)
    r.move_to([xy[0], xy[1], zg], -1, tol=0.004, max_steps=20)
    log("grasp pose", np.round(r.state[:3], 4), "step", r.step_id)
    for _ in range(8):
        r.step([[0, 0, 0] + r.rot_correction() + [1]])
    width = r.state[6] + r.state[7]
    log("closed width", round(width, 4))
    r.move_to([xy[0], xy[1], 0.20], 1, tol=0.01, max_steps=40)
    r.hold(4, 1) if not r.ended else None
    log("lifted", np.round(r.state[:3], 4), "width", round(r.state[6] + r.state[7], 4), "ended", r.ended, "complete", r.complete, "step", r.step_id)
    return xy, zg, width


def place_back(r, xy, zg, log):
    if r.ended:
        return
    r.move_to([xy[0], xy[1], zg + 0.01], 1, tol=0.006, max_steps=40)
    for _ in range(6):
        r.step([[0, 0, 0] + r.rot_correction() + [-1]])
    r.move_to([xy[0], xy[1], 0.17], -1, tol=0.01, max_steps=30)
    log("placed back", np.round(xy, 3), "step", r.step_id)


def main(ctx):
    log = Logger()
    tracker = Tracker(log)
    try:
        run(ctx, log, tracker)
    except Exception as e:
        log("FATAL", repr(e), traceback.format_exc())


def run(ctx, log, tracker):
    r = make_robot(ctx, log, tracker)
    instr = r.meta.get("instruction", "")
    order = parse_order(instr)
    log("instruction:", instr, "order:", order, "limits", r.meta.get("episode_limits"))
    park = r.state[:3].copy()
    btn = find_button(r, log)
    if btn is None:
        btn = np.array([-0.17, -0.0, 0.033]); log("button fallback")
    tracker.exclude.append(btn[:2].copy())
    tracker.update(r)
    # observe the first wave from the high start pose
    while not r.ended and r.step_id < PRESS_START:
        r.hold(1, -1)
    press_button(r, btn, log)
    r.move_to([park[0], park[1], 0.40], -1, tol=0.015, max_steps=40)
    log("parked step", r.step_id)
    # wait until waves stop: no new cube for NEWGAP steps and nothing visible for QUIET steps
    quiet = 0
    while not r.ended:
        s = r.step_id
        anyvis = any(len(v) > 0 for v in tracker.visible_now.values())
        quiet = 0 if anyvis else quiet + 1
        real = [st for v in tracker.sites.values() for st in v if len(st["pts"]) >= 3]
        last_new = max([st["first"] for st in real], default=0)
        if real and quiet >= QUIET and s - last_new >= NEWGAP:
            break
        if s > WAIT_CAP:
            log("wait timeout"); break
        r.hold(1, -1)
    log("waiting done step", r.step_id, "sites:", {c: [(np.round(st["xy"], 3).tolist(), st["first"], st["last"]) for st in v] for c, v in tracker.sites.items() if v})
    plan = []
    for c, allf in order:
        if allf:
            sites = tracker.covered_sites(c, r.step_id)
        else:
            s1 = tracker.covered_site(c, r.step_id)
            sites = [s1] if s1 is not None else []
        if not sites:
            log("no covered site for", c)
        plan.extend((c, s) for s in sites)
    log("plan", [(c, np.round(s["xy"], 3).tolist()) for c, s in plan])
    for c, site in plan:
        if r.ended:
            break
        exp = site["xy"]
        log("picking", c, "expected", np.round(exp, 3))
        xy, zg, w = pick_container(r, exp, log)
        if r.ended:
            break
        place_back(r, xy, zg, log)
    log("done; ended", r.ended, "complete", r.complete, "step", r.step_id)
