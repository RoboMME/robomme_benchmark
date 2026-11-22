"""ButtonUnmaskSwap controller: remember cube colors at step 0, track the shuffling containers with
RGB-D point clouds, press both buttons, then lift the containers in instruction order."""
import os, sys, re, time, json, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np
import core
from core import observe, step, log
from percep import gather, clusters
from tracker import Tracker, init_cubes, split_buttons, CZ_LO, CZ_HI
from motion import move_to, hold
from pick import pick_container, DOWN

COLORS = ["blue", "green", "red", "yellow", "orange", "purple", "black", "white"]
OBS_POSE = [-0.06, 0.0, 0.45]


def parse_order(instr):
    found = []
    for m in re.finditer(r"\b(" + "|".join(COLORS) + r")\b", instr.lower()):
        found.append(m.group(1))
    return found


def press_once(ctx, st, b, trk):
    lastz = None
    stall = 0
    for _ in range(16):
        o = st["obs"]
        z = o.state[2]
        if z < 0.0265:
            break
        if lastz is not None and z < 0.04 and lastz - z < 0.002:
            stall += 1
            if stall >= 2:
                break
        else:
            stall = 0
        lastz = z
        dz = max(-0.02, 0.015 - z)
        dx, dy = b[0] - o.state[0], b[1] - o.state[1]
        row = [dx / 0.05, dy / 0.05, dz / 0.05, 0, 0, 0, 1]
        st["obs"] = step(ctx, [row])
        trk.update(st["obs"])
    return float(st["obs"].state[2])


def zone_clear(trk, b):
    for k in (0, 3, 6):
        p = trk.pos + 2.0 * trk.vel * k
        dx = np.abs(p[:, 0] - b[0]); dy = np.abs(p[:, 1] - b[1])
        if np.any((dx < 0.07) & (dy < 0.11)):
            return False
    return True


def press_button(ctx, st, b, trk, tag, deadline=135):
    move_to(ctx, st, [b[0], b[1], 0.30], 1, trk, tgt_rot=DOWN, maxstep=0.05, tol=0.015, chunk=1, max_calls=40)
    # wait until no container is (or will soon be) near the button/hand footprint
    waited = 0
    while st["obs"].step < deadline and not st["obs"].ended:
        if zone_clear(trk, b):
            break
        hold(ctx, st, 1, 1, trk, chunk=1)
        waited += 1
    log(tag, "zone wait", waited, "step", st["obs"].step)
    move_to(ctx, st, [b[0], b[1], 0.07], 1, trk, tgt_rot=DOWN, maxstep=0.05, tol=0.01, chunk=1, max_calls=20)
    log(tag, "above step", st["obs"].step, np.round(st["obs"].state[:3], 3))
    for attempt in range(4):
        z = press_once(ctx, st, b, trk)
        log(tag, "press attempt", attempt, "step", st["obs"].step, "z", round(z, 4))
        if z < 0.028 or st["obs"].step > 165:
            break
        # blocked (e.g. a container next to the button): back off, let it move away, retry
        move_to(ctx, st, [b[0], b[1], 0.10], 1, trk, tgt_rot=DOWN, maxstep=0.05, tol=0.01, chunk=1, max_calls=10)
        hold(ctx, st, 4, 1, trk, chunk=1)
        move_to(ctx, st, [b[0], b[1], 0.07], 1, trk, tgt_rot=DOWN, maxstep=0.05, tol=0.01, chunk=1, max_calls=10)
    move_to(ctx, st, [b[0], b[1], 0.30], 1, trk, tgt_rot=DOWN, maxstep=0.05, tol=0.03, chunk=1, max_calls=30)


def run(ctx):
    t0 = time.time()
    st = {"obs": observe(ctx)}
    o0 = st["obs"]
    instr = o0.meta.get("instruction", "")
    order = parse_order(instr)
    log("instruction:", instr, "order:", order, "step", o0.step)
    objs = init_cubes(o0)
    import percep
    log("static cells", int(percep.STATIC.sum()) if percep.STATIC is not None else None)
    log("step0 objects:", [(c, np.round(xy, 3).tolist(), n) for c, xy, n in objs])
    # wait for containers to cover cubes
    n_cubes = max(1, len(objs) - 2)
    cov = None
    for i in range(40):
        st["obs"] = step(ctx, [[0, 0, 0, 0, 0, 0, -1]])
        P, C = gather(st["obs"], CZ_LO, CZ_HI)
        cl = [c for c in clusters(P, C) if c["n"] > 60]
        if len(cl) >= n_cubes:
            cov = st["obs"]
            break
    log("covered at step", st["obs"].step, "containers", len(cl))
    cubes, buttons = split_buttons(objs, cov if cov is not None else st["obs"])
    log("cubes:", [(c, np.round(xy, 3).tolist()) for c, xy in cubes], "buttons:", [np.round(b, 3).tolist() for b in buttons])
    Pc, _ = gather(o0, 0.028, 0.045)
    refined = []
    for b in buttons:
        m = np.hypot(Pc[:, 0] - b[0], Pc[:, 1] - b[1]) < 0.035 if len(Pc) else np.zeros(0, bool)
        refined.append(Pc[m, :2].mean(0) if m.sum() > 20 else np.asarray(b))
    buttons = refined
    log("button caps:", [np.round(b, 3).tolist() for b in buttons])
    trk = Tracker(cubes)
    trk.update(st["obs"])
    # buttons: nearest to the arm first
    ee = st["obs"].state[:2]
    buttons = sorted(buttons, key=lambda b: np.linalg.norm(np.asarray(b) - ee))[:2]
    for i, b in enumerate(buttons):
        press_button(ctx, st, b, trk, "button%d" % i, deadline=(100 if i == 0 else 150))
    # observe from above until containers stop
    move_to(ctx, st, OBS_POSE, -1, trk, tgt_rot=DOWN, maxstep=0.045, tol=0.02, chunk=1, max_calls=60)
    while not st["obs"].ended and st["obs"].step < 520:
        hold(ctx, st, 1, -1, trk, chunk=1)
        if trk.still >= 30 and st["obs"].step > 200:
            break
    log("settled step", st["obs"].step, "still", trk.still, "pos", trk.pos.round(3).tolist())
    log("track modes", {m: sum(1 for h in trk.history if h[1] == m) for m in ("clusters", "kmeans")})
    targets = [c for c in order if c in trk.colors]
    log("targets", targets)
    for k, col in enumerate(targets):
        if st["obs"].ended:
            break
        last = (k == len(targets) - 1)
        for attempt in range(3):
            xy = trk.where(col)
            ok, ctr = pick_container(ctx, st, xy, None, put_back=not last)
            log("pick", col, "attempt", attempt, "ok", ok, "step", st["obs"].step, "ended", st["obs"].ended,
                "complete", st["obs"].task_complete)
            if ok or st["obs"].ended:
                break
            # failed grasp: re-acquire nearby container centre from current view
            move_to(ctx, st, [xy[0], xy[1], 0.30], -1, None, tgt_rot=DOWN)
            P, C = gather(st["obs"], CZ_LO, CZ_HI)
            cl = [c for c in clusters(P, C) if c["n"] > 40]
            if cl:
                X = np.array([c["xy"] for c in cl])
                j = np.linalg.norm(X - xy, axis=1).argmin()
                if np.linalg.norm(X[j] - xy) < 0.05:
                    trk.pos[trk.colors.index(col)] = X[j]
    try:
        json.dump(trk.history, open("/workspace/output/track_history.json", "w"))
    except Exception:
        pass
    log("done step", st["obs"].step, "ended", st["obs"].ended, "task_complete", st["obs"].task_complete,
        "wall", round(time.time() - t0, 1))


def main(ctx):
    os.makedirs("/workspace/output", exist_ok=True)
    core.LOG = open("/workspace/output/controller_log.txt", "a")
    try:
        run(ctx)
    except Exception:
        log("EXCEPTION", traceback.format_exc())
