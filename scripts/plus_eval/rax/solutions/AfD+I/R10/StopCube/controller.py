"""StopCube controller: hover over the button, count cube->target reaches, press on the Nth.

Perception: front-camera depth back-projected to world. Button = static raised blob,
cube = moving raised blob (top ~3 cm), target = flat raised disc (~1.5 cm).
"""
import json, os, re, sys, time, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pkt, percep

CFG = dict(
    hover_z=0.045,       # TCP height while waiting above the button dome
    press_z=0.000,       # TCP target during the press
    reach_r=0.030,       # closest approach below this counts as a reach (m)
    latency=1.0,         # steps from press row to registration (measured ep1: 1 step from hover 0.045)
    init_hold=3,         # zero steps used to separate static button from moving cube
    max_wait_steps=100000,  # bounded by remaining_episode_steps
)
ORDINALS = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6, 'seventh': 7,
            'eighth': 8, 'ninth': 9, 'tenth': 10, 'eleventh': 11, 'twelfth': 12, 'once': 1, 'twice': 2}
OUT = '/workspace/output'


def log(*a):
    print(*a, flush=True)


def parse_count(instr):
    s = instr.lower()
    for w, n in ORDINALS.items():
        if re.search(r'\b' + w + r'\b', s):
            return n
    m = re.search(r'(\d+)(st|nd|rd|th)?\s+time', s)
    if m:
        return int(m.group(1))
    return 1


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = None; self.imgs = None; self.P = None
        self.ended = False; self.complete = None

    def _ingest(self, r):
        m, im = pkt.parse(r)
        if not m:
            raise RuntimeError('empty packet')
        self.meta, self.imgs = m, im
        self.P = pkt.backproject(m, im[('front', 'depth')], 'front')
        if m.get('episode_ended'):
            self.ended = True
            tr = m.get('transition') or {}
            for s in tr.get('steps', []):
                if s.get('episode_ended'):
                    self.complete = s.get('task_complete')
        return m

    def observe(self):
        return self._ingest(self.ctx.call('robomme_observe'))

    def step(self, row):
        row = [float(np.clip(x, -1, 1)) for x in row]
        return self._ingest(self.ctx.call('robomme_step', actions=[row]))

    @property
    def state(self):
        return np.array(self.meta['state'], float)

    @property
    def step_id(self):
        return int(self.meta['step_id'])

    @property
    def remaining(self):
        return int(self.meta.get('episode_limits', {}).get('remaining_episode_steps', 0))


def move_row(cur, goal, grip=1.0, gain=1.0):
    err = (np.asarray(goal) - cur[:3]) / 0.05 * gain
    return list(np.clip(err, -1, 1)) + [0.0, 0.0, 0.0, grip]


def high_blobs(P):
    Q = P.copy(); Q[Q[..., 0] < -0.45] = np.nan  # robot base / far edge
    return [b for b in percep.find_high(Q) if b['n'] >= 8]


def dome_center(P, approx):
    z = P[..., 2]
    sel = (np.hypot(P[..., 0] - approx[0], P[..., 1] - approx[1]) < 0.05) & (z > 0.025) & (z < 0.06)
    if sel.sum() < 5:
        return np.array(approx[:2])
    pts = P[sel]
    return np.array([(pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 1].min() + pts[:, 1].max()) / 2])


def detect_cube(P, button):
    Q = P.copy(); Q[Q[..., 0] < -0.45] = np.nan
    bl = [b for b in percep.find_high(Q, exclude=[button], excl_r=0.06) if b['n'] >= 6]
    bl = [b for b in bl if b['xyz'][2] < 0.05]
    if not bl:
        return None
    b = max(bl, key=lambda b: b['n'])
    pts = P[b['ys'], b['xs']]
    top = pts[pts[:, 2] > np.nanmax(pts[:, 2]) - 0.006]
    return np.nanmedian(top[:, :2], 0)


def target_mask(P, button, cube):
    z = P[..., 2]
    m = (z > 0.012) & (z < 0.021)
    m &= np.hypot(P[..., 0] - button[0], P[..., 1] - button[1]) > 0.065
    if cube is not None:
        m &= np.hypot(P[..., 0] - cube[0], P[..., 1] - cube[1]) > 0.035
    m &= P[..., 0] > -0.45
    return m


class ReachCounter:
    """Counts cube->target reaches; predicts the closest approach of the current pass."""

    def __init__(self, target, r, count=0):
        self.t = np.asarray(target, float); self.r = r
        self.count = count; self.in_reach = False; self.min_d = 9.0
        self.hist = []  # (step, xy)

    def feed(self, step, p):
        p = np.asarray(p, float)
        d = float(np.hypot(*(p - self.t)))
        seg_d = d
        if self.hist and step - self.hist[-1][0] <= 3:
            a = self.hist[-1][1]; ab = p - a; L2 = float(ab @ ab)
            s = 0.0 if L2 < 1e-12 else float(np.clip((self.t - a) @ ab / L2, 0, 1))
            seg_d = float(np.hypot(*(a + s * ab - self.t)))
        self.hist.append((step, p))
        if self.in_reach:
            self.min_d = min(self.min_d, d)
            if d > self.r + 0.005:
                self.in_reach = False; self.count += 1
                print(f'reach #{self.count} completed (min d {self.min_d:.3f}) at step {step}', flush=True)
                self.min_d = 9.0
        elif d < self.r:
            self.in_reach = True; self.min_d = min(seg_d, d)
        elif seg_d < self.r and d > self.r + 0.005:
            self.count += 1
            print(f'reach #{self.count} completed between samples (seg d {seg_d:.3f}) at step {step}', flush=True)

    def velocity(self, step):
        h = [x for x in self.hist if step - x[0] <= 4]
        if len(h) < 2 or h[-1][0] != step:
            return None
        a = h[-3] if len(h) >= 3 and h[-1][0] - h[-3][0] <= 3 else h[-2]
        dt = h[-1][0] - a[0]
        if dt <= 0:
            return None
        return (h[-1][1] - a[1]) / dt

    def should_press(self, step, N, lat):
        if self.count != N - 1:
            return False, None, None
        v = self.velocity(step)
        if v is None:
            return False, None, None
        rel = self.hist[-1][1] - self.t
        vv = float(v @ v)
        if vv < 1e-8:
            return False, None, None
        tc = -float(rel @ v) / vv
        dmin = float(np.hypot(*(rel + v * max(tc, 0))))
        tc_r, dmin_r = round(tc, 2), round(dmin, 4)
        if tc > -0.75 and tc < lat + 0.5 and dmin < self.r:
            return True, tc_r, dmin_r
        if self.in_reach and tc < lat + 0.5:
            return True, tc_r, dmin_r  # fallback: Nth reach under way
        return False, tc_r, dmin_r


def main(ctx, prior_count=0, cfg=None):
    C = dict(CFG); C.update(cfg or {})
    os.makedirs(OUT, exist_ok=True)
    trace = []
    R = Robot(ctx)
    try:
        R.observe()
        N = parse_count(R.meta.get('instruction', ''))
        log('instruction:', R.meta.get('instruction'), '-> N =', N, 'step', R.step_id)
        frames_P = [R.P.copy()]
        # ---- phase 1: separate static button from moving cube
        b0 = high_blobs(R.P)
        for _ in range(C['init_hold']):
            R.step([0, 0, 0, 0, 0, 0, 1])
            frames_P.append(R.P.copy())
        b1 = high_blobs(R.P)
        static = []
        for a in b1:
            dmin = min([np.hypot(*(a['xyz'][:2] - b['xyz'][:2])) for b in b0] or [1])
            static.append((dmin, a))
        cand = [a for d, a in static if d < 0.006 and a['zmax'] > 0.025]
        if not cand:
            cand = [min(static, key=lambda t: t[0])[1]] if static else []
        if not cand:
            raise RuntimeError('no button found')
        bt = max(cand, key=lambda a: a['n'])
        button = dome_center(R.P, bt['xyz'])
        log('button blob', np.round(bt['xyz'], 3), 'dome center', np.round(button, 4))
        # ---- target from accumulated flat masks
        acc = None
        for Pf in frames_P:
            cb = detect_cube(Pf, button)
            tm = target_mask(Pf, button, cb).astype(int)
            acc = tm if acc is None else acc + tm
        from scipy import ndimage
        lab, n = ndimage.label(acc >= 1)
        best = None
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            if best is None or len(ys) > len(best[0]):
                best = (ys, xs)
        if best is None:
            raise RuntimeError('no target found')
        tp = R.P[best[0], best[1]]
        tp = tp[np.isfinite(tp[:, 0])]
        target = np.nanmedian(tp[:, :2], 0)
        log('target', np.round(target, 4), 'npx', len(best[0]))
        # ---- reach counting (segment-based, robust to fast cubes / dropped frames)
        RC = ReachCounter(target, C['reach_r'], prior_count)
        for k, Pf in enumerate(frames_P):
            cb = detect_cube(Pf, button)
            if cb is not None:
                RC.feed(R.step_id - (len(frames_P) - 1 - k), cb)
        hover = np.array([button[0], button[1], C['hover_z']])
        pressed = False; press_step = None
        while not R.ended and R.remaining > 0 and R.step_id < C['max_wait_steps']:
            cur = R.state
            cb = detect_cube(R.P, button)
            if cb is not None:
                RC.feed(R.step_id, cb)
            lat = max(C['latency'], np.ceil((cur[2] - 0.030) / 0.017))
            decide, tc, dmin = RC.should_press(R.step_id, N, lat)
            trace.append(dict(step=R.step_id, cube=None if cb is None else [round(float(x), 4) for x in cb],
                              d=None if cb is None else round(float(np.hypot(*(cb - target))), 4), tc=tc, dmin=dmin,
                              count=RC.count, in_reach=RC.in_reach, state=[round(float(x), 4) for x in cur[:3]]))
            if decide:
                log(f'PRESS at step {R.step_id}: count={RC.count} in_reach={RC.in_reach} tc={tc} dmin={dmin} lat={lat}')
                pressed = True; press_step = R.step_id
                break
            if RC.count >= N:
                log(f'WARNING: count {RC.count} >= N {N} without press at step {R.step_id}')
            # hover hold / approach
            R.step(move_row(cur, hover, 1.0))
        if pressed:
            for k in range(8):
                if R.ended:
                    break
                cur = R.state
                R.step(move_row(cur, [button[0], button[1], C['press_z']], 1.0))
                cb = detect_cube(R.P, button)
                trace.append(dict(step=R.step_id, phase='press', cube=None if cb is None else [round(float(x), 4) for x in cb],
                                  d=None if cb is None else round(float(np.hypot(*(cb - target))), 4),
                                  state=[round(float(x), 4) for x in R.state[:3]]))
            # hold until the episode reports its end (bounded)
            k = 0
            while not R.ended and R.remaining > 0 and k < 40:
                R.step([0, 0, 0, 0, 0, 0, 1]); k += 1
        log('done: ended', R.ended, 'complete', R.complete, 'step', R.step_id)
    except Exception as e:
        log('ERROR', repr(e)); log(traceback.format_exc())
    finally:
        try:
            with open(os.path.join(OUT, 'trace_%d.json' % int(time.time())), 'w') as f:
                json.dump(trace, f)
        except Exception:
            pass
