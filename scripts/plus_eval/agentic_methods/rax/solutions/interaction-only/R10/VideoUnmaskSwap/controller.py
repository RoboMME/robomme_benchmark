"""VideoUnmaskSwap controller: track containers through the conditioning video and the
continuing live shuffle, then lift the containers hiding the instructed colours in order."""
import json, math, os, re, sys, time, traceback
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import motion
from motion import Robot, move_to, rotvec_to_R, pick_container, place_down, revealed_color, yaw_of
from track import Tracker, track_video2, top_points, components

ROOT = '/workspace'
OUT = Path('/workspace/output'); 
COLOR_WORDS = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'white', 'black']

def make_log():
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        fh = open(OUT / 'controller_log.txt', 'a')
    except Exception:
        fh = None
    t0 = time.time()
    def log(*a):
        msg = f"[{time.time()-t0:7.1f}s] " + ' '.join(str(x) for x in a)
        print(msg, flush=True)
        if fh:
            fh.write(msg + '\n'); fh.flush()
    return log

def parse_plan(instruction):
    """ordered list of (color, all_flag) from the language goal"""
    text = instruction.lower()
    plan = []
    for m in re.finditer(r'\b(' + '|'.join(COLOR_WORDS) + r')\b', text):
        color = m.group(1)
        before = text[max(0, m.start() - 45):m.start()]
        after = text[m.end():m.end() + 8]
        allf = ('all ' in before.split(',')[-1]) or after.strip().startswith('cubes')
        if any(c == color for c, _ in plan):
            continue
        plan.append((color, allf))
    return plan

def live_track(robot, tr, log, static_needed=25, max_steps=400, thr=0.0025):
    static = 0; n = 0
    while n < max_steps and not robot.ended:
        P = robot.step_scene_points([0, 0, 0, 0, 0, 0, -1]); n += 1
        if P is None:
            log('live_track: no depth'); break
        mv = tr.update(P, dt=1.0)
        static = static + 1 if mv < thr else 0
        if n % 10 == 0:
            log(f'live_track step {n} mv={mv:.4f} static={static}')
        if static >= static_needed:
            break
    log(f'live_track done after {n} steps, static={static}')
    return n

def main(ctx):
    log = make_log()
    robot = Robot(ctx, log=log)
    meta = robot.observe()
    instr = meta.get('instruction', '')
    log('instruction:', instr)
    motion.R_DOWN = rotvec_to_R([-math.pi, 0.0, 0.0])
    plan = parse_plan(instr)
    log('plan:', plan)
    # --- video tracking
    try:
        mpath = meta['frame_sequence']['manifest_path']
        manifest = json.loads((Path(ROOT) / mpath).read_text())
        tr, cubes = track_video2(ROOT, manifest)
        log('cubes:', [(c['color'], np.round(c['xy'], 3).tolist(), c['n']) for c in cubes])
        log('after video:', [(l, np.round(c, 3).tolist()) for l, c in zip(tr.labels, tr.C)])
    except Exception as e:
        log('video tracking failed:', repr(e), traceback.format_exc())
        return
    # --- continue tracking during the live shuffle (arm parked high, out of view)
    live_track(robot, tr, log)
    final = [(l, np.round(c, 3).tolist()) for l, c in zip(tr.labels, tr.C)]
    log('final map:', final)
    try:
        json.dump({'instruction': instr, 'plan': plan, 'cubes': cubes, 'final': final}, open(OUT / 'map.json', 'w'))
    except Exception:
        pass
    # --- build pick sequence
    seq = []; used = set()
    for color, allf in plan:
        idx = [i for i, l in enumerate(tr.labels) if l == color]
        if not idx:
            # fallback: a container whose cube colour was not recognised
            unk = [i for i, l in enumerate(tr.labels) if l == 'unknown' and i not in used]
            log('WARNING: no container for color', color, 'fallback unknown', unk)
            if not unk: continue
            idx = unk[:1]
        used.update(idx)
        if not allf:
            idx = idx[:1]
        seq.append((color, idx))
    log('sequence:', seq)
    total = sum(len(i) for _, i in seq)
    done = 0
    for color, idx in seq:
        remaining = list(idx)
        while remaining and not robot.ended:
            p = robot.state()[:2]
            remaining.sort(key=lambda i: np.linalg.norm(tr.C[i] - p))
            i = remaining.pop(0)
            for attempt in range(3):
                if robot.ended: break
                ok, info = pick_container(robot, tr.C[i], log=log)
                if robot.ended: break
                try:
                    rc = revealed_color(robot, info['xy'])
                    log(f'revealed under container {i} (expected {color}):', rc)
                except Exception as e:
                    log('reveal check failed', repr(e))
                if ok:
                    break
                log('grasp failed, retry'); motion.gripper(robot, -1, 8)
            done += 1
            if robot.ended: break
            place_down(robot, info['xy'], info['yaw'])
            tr.C[i] = np.array(info['xy'])
    log(f'finished: ended={robot.ended} complete={robot.complete} picks={done}/{total} remaining_steps={robot.remaining()}')
