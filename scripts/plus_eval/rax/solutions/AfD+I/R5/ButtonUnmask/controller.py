import sys, os, re, json, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from robo import Robot, jcall
from percep import detect_cubes, white_mask, find_container, cluster_xy
from robo import load_frame, backproject

OUT = Path('/workspace/output')
COLOR_WORDS = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']

def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / 'controller_log.txt', 'a') as f:
            f.write(s + '\n')
    except Exception:
        pass

def parse_instruction(instr):
    """returns list of (color, all_flag) in order after 'press the button'"""
    s = instr.lower()
    i = s.find('button')
    if i >= 0:
        s = s[i:]
    items = []
    # split into clauses by color mentions
    for m in re.finditer(r'(red|green|blue|yellow|orange|purple)', s):
        c = m.group(1)
        pre = s[max(0, m.start() - 45):m.start()]
        allf = ('all' in pre.split()[-5:]) or s[m.end():m.end() + 6].startswith(' cubes')
        items.append((c, allf))
    return items

class Tracker:
    def __init__(self):
        self.det = []  # (step, color, cam, xyz, n)
        self.seen_frames = set()
        self.last_cube_step = -1

    def ingest(self, R):
        for fr in R.frames:
            key = (fr['files'][0]['path'])
            if key in self.seen_frames:
                continue
            self.seen_frames.add(key)
            st = fr.get('step_id', 0)
            for cam in ('front', 'wrist'):
                try:
                    cs = detect_cubes(fr, cam, min_px=6 if cam == 'front' else 10)
                except Exception as e:
                    log('detect err', e); continue
                for c in cs:
                    x, y, z = c['xyz']
                    if not (-0.45 < x < 0.35 and -0.4 < y < 0.4 and 0.0 < z < 0.06):
                        continue
                    self.det.append((st, c['color'], cam, np.array(c['xyz']), c['n']))
                    self.last_cube_step = max(self.last_cube_step, st)

    def clusters(self, color):
        pts = [(d[3], d[0]) for d in self.det if d[1] == color]
        if not pts:
            return []
        P = np.array([p for p, _ in pts])
        cl = cluster_xy(P, 0.025)
        out = []
        for c, n, pp in cl:
            out.append((c, int(n)))
        out.sort(key=lambda t: -t[1])
        return out

def detect_button(fr):
    best = None
    for cam in ('front', 'wrist'):
        try:
            d = load_frame(fr)
            rgb = d[(cam, 'rgb')]; dep = d[(cam, 'depth')]
            a = rgb.astype(int)
            m = (a.min(2) > 110) & (a.max(2) - a.min(2) < 30)
            vs, us = np.nonzero(m)
            if len(us) == 0:
                continue
            P, dd = backproject(dep, fr['camera_intrinsics'][cam]['matrix'], fr['camera_poses'][cam], us, vs)
            ok = (dd > 0) & (P[:, 2] > 0.008) & (P[:, 2] < 0.05) & (P[:, 0] > -0.5) & (P[:, 0] < 0.3)
            P = P[ok]
            if len(P) < 5:
                continue
            for c, n, pp in cluster_xy(P, 0.03):
                ext = np.ptp(pp[:, :2], 0)
                if ext.max() > 0.06:
                    continue
                if best is None or n > best[1]:
                    best = (c, n, float(np.percentile(pp[:, 2], 95)), cam)
        except Exception as e:
            log('button det err', e)
    return best

def main(ctx):
    t0 = time.time()
    R = Robot(ctx, log=log)
    j = R.observe()
    instr = j.get('instruction', '')
    log('instruction:', instr)
    items = parse_instruction(instr)
    log('items:', items)
    home = R.pos().copy()
    tr = Tracker()
    btn = detect_button(R.frames[-1]) if R.frames else None
    log('button:', btn)
    if btn is None:
        bxy, btop = np.array([-0.19, -0.015]), 0.033
    else:
        bxy, btop = np.array(btn[0]), btn[2]
    # Phase 1: observe from home
    while R.step_id < 45 and not R.ended:
        R.hold(5, 1); tr.ingest(R)
    # Phase 2: go above button
    R.move_to([bxy[0], bxy[1], btop + 0.07], 1, max_steps=20, chunk=5, gain=0.7); tr.ingest(R)
    while R.step_id < 66 and not R.ended:
        R.move_to([bxy[0], bxy[1], btop + 0.07], 1, max_steps=5, chunk=5, gain=0.5); tr.ingest(R)
        if R.step_id < 66:
            R.hold(min(5, 66 - R.step_id), 1); tr.ingest(R)
    # Phase 3: press
    R.move_to([bxy[0], bxy[1], btop - 0.035], 1, max_steps=10, chunk=2, gain=1.0); tr.ingest(R)
    log('press bottom z', R.pos()[2], 'step', R.step_id)
    R.hold(2, 1)
    R.move_to([bxy[0], bxy[1], btop + 0.08], 1, max_steps=8, chunk=4, gain=1.0); tr.ingest(R)
    # Phase 4: return high and observe until stable
    R.move_to(home, 1, max_steps=30, chunk=5, gain=0.7); tr.ingest(R)
    while not R.ended and R.step_id < 260:
        R.hold(5, 1); tr.ingest(R)
        if R.step_id >= 110 and R.step_id - tr.last_cube_step >= 25:
            break
    log('observe done step', R.step_id, 'last cube step', tr.last_cube_step, 'ndet', len(tr.det))
    try:
        with open(OUT / 'detections.json', 'w') as f:
            json.dump([(d[0], d[1], d[2], d[3].tolist(), d[4]) for d in tr.det], f)
    except Exception:
        pass
    # targets
    targets = []
    for color, allf in items:
        cl = tr.clusters(color)
        log('clusters', color, [(np.round(c, 3).tolist(), n) for c, n in cl])
        if not cl:
            continue
        if allf:
            mx = cl[0][1]
            for c, n in cl:
                if n >= max(2, 0.25 * mx):
                    targets.append((color, c))
        else:
            targets.append((color, cl[0][0]))
    log('targets', [(c, np.round(p, 3).tolist()) for c, p in targets])
    for k, (color, xy) in enumerate(targets):
        if R.ended:
            break
        last = (k == len(targets) - 1)
        for attempt in range(3):
            if R.ended:
                break
            p0 = R.pos()
            if p0[2] < 0.13:
                R.move_to([p0[0], p0[1], 0.15], -1, max_steps=10, chunk=5, gain=1.0, tol=0.015)
            R.move_to([xy[0], xy[1], 0.23], -1, max_steps=40, chunk=5, gain=0.9, tol=0.008)
            fr = R.frames[-1]
            c = None
            for cam in ('wrist', 'front'):
                try:
                    c = find_container(fr, cam, xy)
                except Exception as e:
                    log('find err', e)
                if c is not None:
                    break
            if c is None:
                log('no container found, using cube xy')
                c = dict(xy=np.array(xy), yaw=0.0, top_z=0.073)
            log('container', color, np.round(c['xy'], 3).tolist(), 'yaw', round(c['yaw'], 3), 'top', round(c['top_z'], 3), 'step', R.step_id)
            gz = max(0.02, c['top_z'] - 0.038)
            R.move_to([c['xy'][0], c['xy'][1], 0.12], -1, max_steps=40, chunk=5, gain=0.9, tol=0.006, yaw_target=c['yaw'])
            R.move_to([c['xy'][0], c['xy'][1], gz], -1, max_steps=30, chunk=4, gain=0.6, yaw_target=c['yaw'])
            R.hold(7, 1)
            w = R.grip_open()
            log('grip after close', w, 'step', R.step_id)
            if w < 0.02:
                log('missed grasp, retry')
                R.hold(4, -1)
                continue
            R.move_to([c['xy'][0], c['xy'][1], 0.17], 1, max_steps=30, chunk=5, gain=1.0, tol=0.01)
            log('lifted', color, 'w', R.grip_open(), 'ended', R.ended, 'complete', R.complete, 'step', R.step_id)
            if R.ended:
                break
            if R.grip_open() < 0.02:
                log('dropped, retry'); continue
            # place back
            R.move_to([c['xy'][0], c['xy'][1], gz + 0.012], 1, max_steps=30, chunk=5, gain=0.8, tol=0.006)
            R.hold(5, -1)
            break
    log('final: ended', R.ended, 'complete', R.complete, 'step', R.step_id, 'elapsed', round(time.time() - t0, 1))
