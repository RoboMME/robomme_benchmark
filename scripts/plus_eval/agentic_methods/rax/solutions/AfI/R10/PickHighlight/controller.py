"""PickHighlight controller: watch highlights, press button, pick each highlighted cube, press button."""
import os, sys, time, json, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion import Robot, hand_yaw
from percept import find_cubes, highlight_scores, find_button
from skills import pick_cube, press_button
from tracker import Tracker

OUT = "/workspace/output"
T0 = time.time()

def _log_factory():
    os.makedirs(OUT, exist_ok=True)
    f = open(os.path.join(OUT, "controller_log.txt"), "a")
    def log(*a):
        s = "[%6.1fs] " % (time.time() - T0) + " ".join(str(x) for x in a)
        print(s); f.write(s + "\n"); f.flush()
    return log

def merge_cubes(lists, tol=0.025):
    out = []
    for cs in lists:
        for c in cs:
            if not any(np.linalg.norm(o["xyz"][:2] - c["xyz"][:2]) < tol for o in out):
                out.append(dict(xyz=c["xyz"].copy()))
    return out

def scene_cubes(robot):
    m, im = robot.meta, robot.imgs
    return merge_cubes([find_cubes(m, im, "front"), find_cubes(m, im, "wrist")])

def score_frame(robot, cubes):
    m, im = robot.meta, robot.imgs
    sf = highlight_scores(m, im, cubes, "front")
    sw = highlight_scores(m, im, cubes, "wrist")
    return np.maximum(np.array(sf), np.array(sw))

def detect_xy(robot, cam):
    return np.array([c["xyz"][:2] for c in find_cubes(robot.meta, robot.imgs, cam)]).reshape(-1, 2)

HOME = np.array([-0.06, 0.0, 0.52])

class TrackState:
    def __init__(self, robot, log):
        self.log = log
        dw0, df0 = detect_xy(robot, "wrist"), detect_xy(robot, "front")
        init = dw0 if len(dw0) >= len(df0) else df0
        self.tr = Tracker(init)
        self.still = 0
        self.moving_seen = None
        self.last_disp = 0.0
        self.hist = []
        self.active = True

    def __call__(self, robot):
        if not self.active:
            return
        at_home = np.linalg.norm(robot.state[:3] - HOME) < 0.03
        df = detect_xy(robot, "front")
        dets = df
        if at_home:
            dw = detect_xy(robot, "wrist")
            if len(dw) >= len(df):
                dets = dw
        disp = self.tr.update(dets)
        # ignore the track being manipulated (near gripper) for motion statistics
        g = robot.state[:2]
        near = np.linalg.norm(self.tr.pos - g, axis=1) < 0.05
        low = robot.state[2] < 0.2
        d2 = disp[~(near & low)] if low else disp
        mv = float(d2.max()) if len(d2) else 0.0
        self.last_disp = mv
        st = robot.meta["step_id"]
        self.hist.append((st, len(dets), round(mv, 4), int((self.tr.miss > 0).sum())))
        if mv > 0.003:
            if self.still > 5:
                self.log("motion at step", st, "after still", self.still)
            self.still = 0
        else:
            self.still += 1

def wait_static(robot, ts, log, need, max_steps):
    n = 0
    while ts.still < need and n < max_steps and not robot.ended:
        robot.hold(1); n += 1
    return n

def watch_highlights(robot, ts, log, thr=0.3, max_steps=400):
    best = np.zeros(len(ts.tr.pos))
    seen = vanished = moving = None
    n = 0
    while n < max_steps and not robot.ended:
        robot.hold(1); n += 1
        st = robot.meta["step_id"]
        if moving is None:
            tmp = [dict(xyz=np.array([p[0], p[1], 0.04])) for p in ts.tr.pos]
            s = np.maximum(np.array(highlight_scores(robot.meta, robot.imgs, tmp, "front")),
                           np.array(highlight_scores(robot.meta, robot.imgs, tmp, "wrist")))
            best = np.maximum(best, s)
            if s.max() > thr and seen is None:
                seen = st; log("highlights seen at step", st, np.round(s, 2).tolist())
            if seen is not None and vanished is None and s.max() < 0.1:
                vanished = st; log("highlights vanished at step", st)
            if seen is not None and ts.last_disp > 0.003:
                moving = st; log("motion starts at step", st)
        if moving is not None and ts.still >= 30:
            log("static again at step", st); break
        if vanished is not None and moving is None and st - vanished > 70:
            log("no shuffle after vanish"); break
        if seen is None and n > 100:
            log("no highlights by step", st); break
    log("highlight best", np.round(best, 2).tolist())
    return best, seen is not None

def plan_order(start_xy, pts):
    order, cur, rem = [], np.asarray(start_xy), list(range(len(pts)))
    while rem:
        j = min(rem, key=lambda k: np.linalg.norm(pts[k] - cur))
        order.append(j); rem.remove(j); cur = pts[j]
    return order

def main(ctx):
    log = _log_factory()
    robot = Robot(ctx, log=log)
    try:
        run(robot, log)
    except Exception as e:
        log("EXCEPTION", repr(e)); log(traceback.format_exc())
        if not robot.ended:
            try:
                emergency_finish(robot, log)
            except Exception as e2:
                log("emergency failed", repr(e2))
    log("done: ended", robot.ended, "complete", robot.complete)

BUTTON = {}

def emergency_finish(robot, log):
    if "xy" in BUTTON:
        robot.observe()
        robot.set_down(hand_yaw(robot.state))
        robot.move_to(list(robot.state[:2]) + [0.15], grip=1.0)
        press_button(robot, BUTTON["xy"], BUTTON["ztop"], log=log)

def run(robot, log):
    robot.observe()
    log("instruction:", robot.meta.get("instruction"), "limit", robot.meta["episode_limits"])
    robot.set_down(hand_yaw(robot.state))
    bf = find_button(robot.meta, robot.imgs, "front")
    bw = find_button(robot.meta, robot.imgs, "wrist")
    cands = [b for b in (bf, bw) if b is not None]
    if not cands:
        raise RuntimeError("button not found")
    bxy = np.mean([b["xy"] for b in cands], 0); btop = float(np.mean([b["ztop"] for b in cands]))
    BUTTON.update(xy=bxy, ztop=btop)
    log("button", np.round(bxy, 3), round(btop, 3), [np.round(b["xy"], 3).tolist() for b in cands])
    ts = TrackState(robot, log)
    log("cubes at start:", len(ts.tr.pos), np.round(ts.tr.pos, 3).tolist())
    robot.on_frame = ts
    robot.track_chunk = 1
    best, seen = watch_highlights(robot, ts, log)
    hl = [i for i in range(len(best)) if best[i] > 0.3]
    log("highlighted idx", hl, "pos", np.round(ts.tr.pos[hl], 3).tolist())
    # first press (tracked every step)
    press_button(robot, bxy, btop, log=log)
    log("first press done at step", robot.meta["step_id"])
    robot.move_to(HOME, chunk=1)
    n = wait_static(robot, ts, log, need=40, max_steps=300)
    log("post-press wait", n, "steps; step", robot.meta["step_id"], "still", ts.still)
    log("highlighted now", np.round(ts.tr.pos[hl], 3).tolist(), "miss", ts.tr.miss.tolist())
    done = set()
    while len(done) < len(hl) and not robot.ended:
        if ts.still < 8:
            robot.move_to([robot.state[0], robot.state[1], 0.25], chunk=1)
            wait_static(robot, ts, log, need=15, max_steps=150)
        rem = [i for i in hl if i not in done]
        cur = robot.state[:2]
        i = min(rem, key=lambda k: np.linalg.norm(ts.tr.pos[k] - cur))
        others = [ts.tr.pos[k] for k in range(len(ts.tr.pos)) if k != i]
        ok = pick_cube(robot, lambda: ts.tr.pos[i].copy(), log=log, neighbors=others)
        if not ok and not robot.ended:
            log("retry pick", i)
            ok = pick_cube(robot, lambda: ts.tr.pos[i].copy(), log=log, neighbors=others)
        done.add(i)
        log("pick idx", i, "ok", ok, "step", robot.meta["step_id"])
    for attempt in range(3):
        if robot.ended:
            break
        press_button(robot, bxy, btop, log=log, depth=0.012 + 0.004 * attempt)
        robot.hold(4)
        log("final press attempt", attempt, "ended", robot.ended, "complete", robot.complete)
    log("track hist tail", ts.hist[-20:])
