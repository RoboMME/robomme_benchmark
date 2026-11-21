import os, sys, json, time, re, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from robot import Robot, move_to, hold, log, LOG
from track import run_tracking
from pick import pick_and_show, wrist_components, SCAN_Z, s_yaw

COLORS = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan']
FRONT_BIAS = np.array([-0.009, 0.0])

def parse_order(instr):
    """list of (color, all_flag) in order of mention."""
    t = instr.lower()
    t = t.split('then', 1)[-1]
    out = []
    for mm in re.finditer(r'(all containers hiding|the (?:one|container) hiding)\s+(?:the\s+)?(\w+)\s+cube', t):
        out.append((mm.group(2), mm.group(1).startswith('all')))
    if not out:
        for c in COLORS:
            pass
        out = [(w, False) for w in re.findall(r'\b(' + '|'.join(COLORS) + r')\b', t)]
    return out

def hungarian(A, B):
    from scipy.optimize import linear_sum_assignment
    D = np.linalg.norm(A[:, None] - B[None], axis=-1)
    r, c = linear_sum_assignment(D)
    return dict(zip(r.tolist(), c.tolist())), D

def main(ctx):
    t0 = time.time()
    os.makedirs('/workspace/output', exist_ok=True)
    rb = Robot(ctx)
    try:
        m = rb.observe()
        instr = m['instruction']; log('instruction', instr)
        order = parse_order(instr); log('order', order)
        seq = m.get('frame_sequence') or {}
        mp = seq.get('manifest_path')
        if mp: mp = os.path.join('/workspace', mp)
        else:
            import glob
            cands = glob.glob('/workspace/runtime/autonomous_controller/runs/*/frames/sequence_000000/manifest.json')
            mp = max(cands, key=os.path.getmtime)
        log('manifest', mp)
        frames = json.load(open(mp))['frames']
        tr = run_tracking('/workspace', frames)
        cubes = tr['cubes']; final = tr['final'] + FRONT_BIAS
        log('tracking %.1fs' % (time.time() - t0), [(c['color'], np.round(f, 3).tolist()) for c, f in zip(cubes, final)])
        # targets in order
        targets = []
        for col, allf in order:
            idx = [i for i, c in enumerate(cubes) if c['color'] == col]
            if not idx: log('WARN no cube of colour', col); continue
            if not allf: idx = idx[:1]
            targets += idx
        log('targets', targets)
        # scan from above the cluster centre, match containers
        cen = final.mean(0)
        move_to(rb, [cen[0], cen[1], SCAN_Z + 0.05], 0.0, -1, max_steps=60, tol=0.01)
        comps, _ = wrist_components(rb)
        pos = final.copy()
        if len(comps) >= 1:
            B = np.array([c['xy'] for c in comps])
            asg, D = hungarian(final, B)
            for i, j in asg.items():
                if D[i, j] < 0.012: pos[i] = B[j]
            log('scan matched', {i: (round(float(D[i, j]), 3)) for i, j in asg.items()})
        done = []
        for i in targets:
            if rb.ended: break
            if rb.remaining() < 130: log('low steps'); break
            res = pick_and_show(rb, pos[i])
            log('pick', i, cubes[i]['color'], res, 'step', rb.last.get('step_id'), 'ended', rb.ended, 'complete', rb.complete)
            if not res.get('ok') and not rb.ended:
                # retry once from refreshed scan
                res = pick_and_show(rb, pos[i])
                log('retry', i, res)
            if res.get('xy') is not None: pos[i] = np.array(res['xy'])
            done.append(i)
        if not rb.ended:
            hold(rb, -0.45, 5)
        log('done', done, 'ended', rb.ended, 'complete', rb.complete, 'time %.1f' % (time.time() - t0))
    except Exception as e:
        log('ERROR', repr(e), traceback.format_exc())
    finally:
        try:
            open('/workspace/output/controller_log.txt', 'w').write('\n'.join(LOG))
        except Exception:
            pass
