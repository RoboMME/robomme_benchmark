"""VideoUnmaskSwap controller: track containers through the conditioning video (RGB-D, world XY),
wait for post-reset shuffling to stop, then lift containers in the instructed colour order."""
import os, sys, json, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)
import numpy as np
from PIL import Image
from mcpio import parse
from wtrack import WorldTracker
from colors import parse_instruction
from motion import Robot
from grasp import pick_container, place_back
from perceive import refine

OUT = '/workspace/output'

class Logger:
    def __init__(self):
        os.makedirs(OUT, exist_ok=True)
        self.f = open(os.path.join(OUT, 'controller_log.txt'), 'a')
        self.t0 = time.time()
    def __call__(self, *a):
        s = f'[{time.time() - self.t0:7.1f}s] ' + ' '.join(str(x) for x in a)
        print(s, flush=True)
        try:
            self.f.write(s + '\n'); self.f.flush()
        except Exception: pass

def load_frame_files(fr):
    out = {}
    for f in fr.get('files', []):
        p = f['path'] if f['path'].startswith('/') else os.path.join('/workspace', f['path'])
        out[(f['camera'], f['kind'])] = np.array(Image.open(p))
    return out

def track_video(meta0, log):
    wt = WorldTracker()
    seq = meta0.get('frame_sequence') or {}
    mp = seq.get('manifest_path')
    n = 0
    if mp:
        mpath = mp if mp.startswith('/') else os.path.join('/workspace', mp)
        man = json.load(open(mpath))
        frames = sorted(man['frames'], key=lambda f: f.get('frame_index', 0))
        for fr in frames:
            try:
                imgs = load_frame_files(fr)
                if 'camera_poses' not in fr:
                    fr = dict(fr, camera_poses=meta0['camera_poses'], camera_intrinsics=meta0['camera_intrinsics'])
                wt.feed(fr, imgs); n += 1
            except Exception as e:
                log('frame load error', fr.get('frame_index'), repr(e))
    log(f'video frames fed: {n}; tracker log: {wt.log[:3]}')
    return wt

def wait_still(rb, wt, log, max_steps=150, still_needed=8):
    prev = None; still = 0
    for k in range(max_steps):
        rb.step([[0, 0, 0, 0, 0, 0, -1]])
        if rb.ended: break
        try:
            wt.feed(rb.meta, rb.imgs)
        except Exception as e:
            log('feed error', repr(e))
        f = rb.imgs[('front', 'rgb')].astype(int)
        if prev is not None:
            ch = int((np.abs(f - prev).max(2) > 30).sum())
            still = still + 1 if ch < 5 else 0
        prev = f
        if still >= still_needed: break
    log(f'waited {k + 1} steps until still; step_id={rb.meta.get("step_id")}')

def main(ctx):
    log = Logger()
    try:
        _main(ctx, log)
    except Exception:
        log('FATAL', traceback.format_exc())

def _main(ctx, log):
    rb = Robot(ctx, log=log)
    meta0 = rb.meta
    instr = meta0.get('instruction', '')
    plan = parse_instruction(instr)
    log('instruction:', instr); log('plan:', plan)
    wt = track_video(meta0, log)
    if not wt.started:
        log('WARNING: containers never detected in video')
    wait_still(rb, wt, log)
    res = wt.result()
    for r in res:
        log('track', r['color'], r['rgb'], np.round(r['final'], 3).tolist(), 'min_cnt', r['min_cnt'])
    pos = {j: np.asarray(r['final'], float) for j, r in enumerate(res)}
    # refresh positions from the current still observation
    for j in pos:
        rr = refine(rb.meta, rb.imgs, pos[j])
        if rr is not None and np.linalg.norm(np.asarray(rr['xy']) - pos[j]) < 0.03:
            pos[j] = np.asarray(rr['xy'])
    targets = []
    used = set()
    for col, all_flag in plan:
        idx = [j for j, r in enumerate(res) if r['color'] == col and j not in used]
        if not idx:
            log(f'WARNING: no container tracked for colour {col}'); continue
        if not all_flag:
            idx = idx[:1]
        for j in idx:
            targets.append((col, j)); used.add(j)
    log('targets:', [(c, j, np.round(pos[j], 3).tolist()) for c, j in targets])
    for n, (col, j) in enumerate(targets):
        if rb.ended: break
        others = [pos[k] for k in pos if k != j]
        log(f'--- pick {n + 1}/{len(targets)}: {col} (track {j}) at {np.round(pos[j], 3).tolist()} remaining={rb.remaining()}')
        out = pick_container(rb, pos[j], others, log=log)
        log(f'    result held={out["held"]} ended={rb.ended} complete={rb.task_complete}')
        if rb.ended: break
        place_back(rb, out['xy'], out['yaw'])
        pos[j] = np.asarray(out['xy'])
        # refresh neighbours after placing
        for k in pos:
            rr = refine(rb.meta, rb.imgs, pos[k])
            if rr is not None and np.linalg.norm(np.asarray(rr['xy']) - pos[k]) < 0.03:
                pos[k] = np.asarray(rr['xy'])
    log(f'DONE ended={rb.ended} task_complete={rb.task_complete} step={rb.meta.get("step_id")}')
