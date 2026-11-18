"""PickXtimes controller: pick the named cube, place on target N times, press button."""
import math, os, re, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import robo, percep, skills, tracker, place
from robo import log

NUM = {"one": 1, "once": 1, "two": 2, "twice": 2, "three": 3, "thrice": 3, "four": 4, "five": 5,
       "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}

def parse_count(instr):
    s = instr.lower()
    m = re.search(r"(\d+)\s*(?:times|x\b)", s)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(" + "|".join(NUM) + r")\b\s*times", s)
    if m:
        return NUM[m.group(1)]
    for w in ("once", "twice", "thrice"):
        if re.search(r"\b" + w + r"\b", s):
            return NUM[w]
    m = re.search(r"repeat\w*[^.]*?\b(" + "|".join(NUM) + r")\b", s)
    if m:
        return NUM[m.group(1)]
    return None

def parse_color(instr):
    s = instr.lower()
    m = re.search(r"pick up the (\w+) cube", s)
    if m and m.group(1) in percep.COLORS:
        return m.group(1)
    for c in percep.COLORS:
        if c in s:
            return c
    return "green"

def survey(r, color):
    """Move to an overview pose and locate target, button and cube."""
    st = r.state
    poses = [[st[0], st[1], 0.45], [-0.06, 0.0, 0.45], [0.0, 0.0, 0.5], [-0.1, 0.0, 0.4]]
    tgt = btn = cube = None
    for p in poses:
        if r.ended:
            break
        r.move_to(p, grip=-1, max_steps=40)
        r.hold(2)
        m = r.observe()
        tgt = tgt or percep.find_target(m)
        cube = percep.find_cube(m, color=color) or cube
        excl = [cube["xy"]] if cube else []
        btn = btn or percep.find_button(m, exclude=excl)
        log(f"survey at {np.round(p,3)}: target={tgt and np.round(tgt['xy'],3)} button={btn and (np.round(btn['xy'],3), round(btn['ztop'],3))} cube={cube and np.round(cube['xy'],3)}")
        if tgt and btn and cube:
            break
    return tgt, btn, cube

USE_MODEL = True

def watch(r, trk, max_steps=230):
    """Hold still and record the target to learn whether/how it moves."""
    s0 = r.step_id
    model = None
    while r.step_id - s0 < max_steps and not r.ended:
        r.step([[0, 0, 0] + r.rot_fix(r.state) + [-1]] * 2)
        n = r.step_id - s0
        if n >= 30 and trk.moving() is False:
            log(f"watch: target static after {n} steps at {np.round(trk.hist[-1][1:],3)}")
            return None
        if n >= 70 and n % 10 == 0 and USE_MODEL:
            m = tracker.PathModel(trk)
            if m.build():
                model = m
                break
    log(f"watch: {r.step_id - s0} steps, {len(trk.hist)} measurements, moving={trk.moving()}, model={'yes' if model else 'no'}")
    return model

def on_target(cube, txy, tol=0.03):
    return cube is not None and np.linalg.norm(np.asarray(cube["xy"]) - np.asarray(txy)) < tol and cube["ztop"] < 0.07

def run(ctx):
    t_start = time.time()
    r = robo.Robot(ctx)
    m = r.observe()
    instr = m.get("instruction", "")
    n_req = parse_count(instr)
    color = parse_color(instr)
    log(f"instruction: {instr!r} -> count={n_req} color={color} limit={m.get('episode_limits')}")
    if n_req is None:
        n_req = 1
        log("WARNING: could not parse count, defaulting to 1")
    tgt, btn, cube = survey(r, color)
    if tgt is None:
        log("ERROR: target not found")
    if cube is None:
        log("ERROR: cube not found")
    if tgt is None or cube is None:
        return
    txy = np.asarray(tgt["xy"])
    trk = tracker.TargetTracker()
    r.on_obs = trk.measure
    trk.measure(r.meta)
    budget = (r.remaining or 2000) - n_req * 130 - 120
    model = watch(r, trk, max_steps=int(max(160, min(330, budget))))
    moving_target = bool(trk.moving())
    est = place.TargetEstimator(trk, model, static_mode=not moving_target)
    done = 0
    fails = 0
    while done < n_req and not r.ended:
        if r.remaining is not None and r.remaining < 150:
            log("step budget low; stopping cycles")
            break
        if time.time() - t_start > 1000:
            log("wall budget low; stopping cycles")
            break
        s0 = r.step_id
        ok = skills.pick(r, cube["xy"], color=color)
        if not ok:
            fails += 1
            log(f"pick failed ({fails})")
            m = r.observe()
            c = percep.find_cube(m, color=color)
            if c is None:
                r.move_to([r.state[0], r.state[1], 0.45], grip=-1, max_steps=30)
                m = r.observe(); c = percep.find_cube(m, color=color)
            if c is not None:
                cube = c
            if fails > 8:
                break
            continue
        released, info = place.place_tracked(r, est)
        m = r.observe()
        c = percep.find_cube(m, color=color)
        if c is None:
            r.move_to([r.state[0], r.state[1], 0.3], grip=-1, max_steps=30)
            m = r.observe(); c = percep.find_cube(m, color=color)
        tnow, tq = est.predict(r.step_id, 0)
        log(f"after place: cube={c and np.round(c['xy'],3)} target_est={tnow is not None and np.round(tnow,3)} ({tq})")
        if c is not None and tnow is not None:
            txy = tnow
        # target pauses while the cube rests on it: use that as confirmation
        for _ in range(5):
            r.hold(2)
        paused = est.static(r.step_id, window=10) if moving_target else None
        tnow, tq = est.predict(r.step_id, 0)
        near = c is not None and tnow is not None and np.linalg.norm(np.asarray(c["xy"]) - tnow) < 0.02
        log(f"verify: paused={paused} near={near} target={tnow is not None and np.round(tnow,3)}")
        if (moving_target and paused and near) or (not moving_target and released and on_target(c, tnow if tnow is not None else txy, tol=0.025)):
            done += 1
            log(f"cycle {done}/{n_req} done in {r.step_id - s0} steps; cube at {np.round(c['xy'],3)} (step {r.step_id}, remaining {r.remaining})")
        else:
            log(f"placement not verified: cube={c and np.round(c['xy'],3)}")
            fails += 1
        if c is not None:
            cube = c
    if r.ended:
        log(f"episode ended before press: complete={r.complete}")
        return
    if btn is None:
        r.move_to([-0.06, 0.0, 0.45], grip=-1, max_steps=40)
        m = r.observe()
        btn = percep.find_button(m, exclude=[txy])
        if btn is None:
            log("ERROR: button not found")
            return
    bxy, bz = btn["xy"], btn["ztop"]
    for k in range(3):
        if r.ended:
            break
        skills.press_button(r, bxy, bz, extra=0.01 * k)
        log(f"press {k}: ended={r.ended} complete={r.complete} step={r.step_id}")
        if not r.ended:
            r.move_to([bxy[0], bxy[1], 0.3], grip=1, max_steps=30)
            m = r.observe()
            b2 = percep.find_button(m, exclude=[txy])
            if b2 is not None:
                bxy, bz = b2["xy"], b2["ztop"]
    log(f"final: ended={r.ended} complete={r.complete} step={r.step_id} wall={time.time()-t_start:.1f}s")

def main(ctx):
    os.makedirs("/workspace/output", exist_ok=True)
    try:
        run(ctx)
    except Exception:
        log("EXCEPTION: " + traceback.format_exc())
    finally:
        try:
            with open("/workspace/output/controller_log.txt", "w") as f:
                f.write("\n".join(robo.LOG))
        except Exception:
            pass
