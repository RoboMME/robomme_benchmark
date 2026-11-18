"""VideoPlaceOrder controller: track target identities through swaps in the
conditioning video, find the k-th placement target of the instructed cube, then
pick the cube and place it on that target's final position. Falls back to other
targets if the episode has not ended after a placement."""
import sys, os, json, time, traceback
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from PIL import Image
from robot import Robot, backproject
from perception import solve_v3, slot_swap_solve, detect_targets, parse_instruction
from skills import pick_place

OUT = Path('/workspace/output'); 
def log(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True)
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / 'controller_log.txt', 'a') as f: f.write(msg + '\n')
    except Exception:
        pass

def load_video(meta):
    seq = meta.get('frame_sequence') or {}
    mp = seq.get('manifest_path')
    if not mp: return [], []
    root = Path('/workspace')
    man = json.loads((root / mp).read_text())
    frames, times = [], []
    for fr in sorted(man['frames'], key=lambda f: f['frame_index']):
        files = {(f['camera'], f['kind']): f['path'] for f in fr['files']}
        p = files.get(('front', 'rgb'))
        if not p: continue
        with Image.open(root / p) as im:
            frames.append(np.array(im.convert('RGB')))
        times.append(float(fr.get('timestamp_s', len(times) / 20.0)) if fr['phase'] == 'video_demo' else (times[-1] + 0.1 if times else 0.0))
    return frames, times

def main(ctx):
    t0 = time.time()
    robot = Robot(ctx, log=log)
    meta = robot.observe()
    instr = meta.get('instruction', '')
    log('instruction:', instr, 'limits', meta.get('episode_limits'))
    color, k = parse_instruction(instr)
    final_targets_px = None; order = []
    try:
        frames, times = load_video(meta)
        log(f'video frames {len(frames)} load {time.time()-t0:.1f}s')
        res = solve_v3(frames, times, instr, log=log)
        log('solve result', res)
        if res:
            order = list(res['order_px'])
    except Exception as e:
        log('perception error', repr(e), traceback.format_exc())
    if not order:
        robot.observe()
        order = [(d[0], d[1]) for d in detect_targets(robot.imgs[('front', 'rgb')][..., :3])]
    log('candidate target px order', [(round(u), round(v)) for u, v in order])
    # world positions of targets from current observation depth
    robot.observe()
    dep = robot.imgs[('front', 'depth')]
    cur = detect_targets(robot.imgs[('front', 'rgb')][..., :3])
    worlds = []
    for (u, v) in order:
        # snap to nearest detected target in the current frame
        if cur:
            dd = [np.hypot(c[0]-u, c[1]-v) for c in cur]; j = int(np.argmin(dd))
            if dd[j] < 12: u, v = cur[j][0], cur[j][1]
        P = backproject(robot.meta, dep, 'front', np.array([round(u)]), np.array([round(v)]))
        if len(P): worlds.append(P[0])
    log('target worlds', [np.round(w, 3).tolist() for w in worlds])
    for attempt, w in enumerate(worlds[:4]):
        if robot.ended: break
        if robot.remaining() < 90:
            log('not enough steps for another attempt'); break
        log(f'attempt {attempt}: place {color} at {np.round(w,3)}')
        try:
            pick_place(robot, color, (w[0], w[1]), log=log)
        except Exception as e:
            log('pick_place error', repr(e), traceback.format_exc())
            try: robot.observe()
            except Exception: pass
        log(f'after attempt {attempt}: ended={robot.ended} success={robot.success} step={robot.meta.get("step_id")}')
    log(f'done ended={robot.ended} success={robot.success} wall={time.time()-t0:.1f}s')
