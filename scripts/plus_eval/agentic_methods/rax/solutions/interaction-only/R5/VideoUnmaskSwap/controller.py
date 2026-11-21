"""VideoUnmaskSwap controller: track containers through the conditioning video and the
first execution steps (front RGB-D, top-down), then pick containers in instruction order."""
import json, os, re, sys, time, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from geom import deproject  # noqa: E402
from track import track_video, fused_points  # noqa: E402
from motion import Robot, observe_full, pick_container, cur_yaw  # noqa: E402

COLORS = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']
OUT = '/workspace/output'
_logf = None

def log(msg):
    global _logf
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    try:
        if _logf is None:
            os.makedirs(OUT, exist_ok=True)
            _logf = open(os.path.join(OUT, 'controller_log.txt'), 'a')
        _logf.write(line + '\n'); _logf.flush()
    except Exception:
        pass

def parse_order(instr):
    """Colors in the order the instruction mentions them (after 'then' if present)."""
    t = instr.lower()
    if 'then' in t:
        t = t[t.index('then'):]
    found = []
    for m in re.finditer(r'\b(' + '|'.join(COLORS) + r')\b', t):
        if m.group(1) not in found:
            found.append(m.group(1))
    return found

def front_world(p, imgs):
    return deproject(imgs[('front', 'depth')], p['camera_intrinsics']['front']['matrix'],
                     p['camera_poses']['front'])

def views_from(p, imgs):
    views = []
    for cam in ('front', 'wrist'):
        if (cam, 'depth') in imgs and (cam, 'rgb') in imgs:
            W = deproject(imgs[(cam, 'depth')], p['camera_intrinsics'][cam]['matrix'], p['camera_poses'][cam])
            views.append((W, imgs[(cam, 'rgb')]))
    return views

def exec_track(rb, ctx, tr, max_steps=150, still_needed=25, still_tol=0.0015):
    """Hold the arm still and keep tracking the containers until they stop moving."""
    still = 0
    for k in range(max_steps):
        if rb.ended: break
        C0 = tr.C.copy()
        rb.step([[0, 0, 0, 0, 0, 0, -1]])
        p, imgs = observe_full(ctx, rb.raw)
        tr.update(fused_points(views_from(p, imgs)), 1.0)
        d = np.linalg.norm(tr.C - C0, axis=1)
        good = tr.lost == 0
        mv = d[good].max() if good.any() else 1.0
        still = still + 1 if mv < still_tol else 0
        if k % 10 == 0:
            log(f'  exec track step {p.get("step_id")}: maxmove {mv:.4f} counts {tr.counts.tolist()}')
        if still >= still_needed:
            log(f'  containers stationary after {k+1} steps')
            break
    return tr.C.copy()

def main(ctx):
    t0 = time.time()
    rb = Robot(ctx, log=log)
    p = rb.observe()
    instr = p.get('instruction', '')
    order = parse_order(instr)
    log(f'instruction: {instr}')
    log(f'color order: {order}; limits {p.get("episode_limits")}')
    seq = p.get('frame_sequence') or {}
    mpath = seq.get('manifest_path')
    manifest = json.load(open(os.path.join('/workspace', mpath)))
    colors, seeds, hist, tr = track_video(manifest, log=log)
    C = hist[-1][1].copy()
    log(f'video tracked {len(hist)} frames in {time.time()-t0:.1f}s; final: ' +
        ', '.join(f'{c}@({x:.3f},{y:.3f})' for c, (x, y) in zip(colors, C)))
    n_targets = sum(1 for c in colors if c in order)
    budget = int(np.clip(rb.remaining() - 155 * n_targets - 20, 60, 300))
    log(f'{n_targets} target containers; exec tracking budget {budget} steps')
    C = exec_track(rb, ctx, tr, max_steps=budget)
    log('settled: ' + ', '.join(f'{c}@({x:.3f},{y:.3f})' for c, (x, y) in zip(colors, C)))
    # plan
    plan = []
    for col in order:
        idx = [i for i, c in enumerate(colors) if c == col]
        plan.append((col, idx))
    log(f'plan: {plan}')
    pos = rb.state()[:2]
    done = set()
    for col, idx in plan:
        rem = [i for i in idx if i not in done and -0.5 < C[i][0] < 0.3 and abs(C[i][1]) < 0.4]
        if len(rem) < len(idx):
            log(f'  WARNING: {len(idx)-len(rem)} {col} target(s) outside workspace bounds skipped')
        while rem and not rb.ended:
            i = min(rem, key=lambda j: np.linalg.norm(C[j] - pos))
            others = [C[j] for j in range(len(C)) if j != i]
            log(f'PICK {col} container {i} at ({C[i][0]:.3f},{C[i][1]:.3f}); remaining steps {rb.remaining()}')
            try:
                info = pick_container(rb, ctx, C[i], log=log, others=others)
            except Exception as e:
                log(f'pick error: {e!r}\n{traceback.format_exc()}')
                if rb.ended: break
                raise
            tries = 1
            while not info.get('held') and not rb.ended and tries < 3 and rb.remaining() > 150:
                log(f'  grasp missed (opening {info.get("finger")}); retry {tries}')
                tries += 1
                info = pick_container(rb, ctx, np.array(info.get('grasp_xy', C[i])), log=log, others=others)
            C[i] = np.array(info.get('grasp_xy', C[i]))
            pos = C[i]
            done.add(i); rem.remove(i)
            log(f'  info {json.dumps(info)}; complete={rb.complete} ended={rb.ended} t={time.time()-t0:.0f}s')
        if rb.ended: break
    log(f'finished: ended={rb.ended} complete={rb.complete} step={rb.last.get("step_id")} t={time.time()-t0:.0f}s')
