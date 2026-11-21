"""PatternLock controller: read conditioning video (TCP states + front depth), extract the
node sequence in grid-index space, detect the execution grid, and trace the same pattern."""
import json, math, os, sys, time, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perception import grid_from_frame, extract_pattern, video_offsets, red_mask, world_to_pixel
import base64, io
from PIL import Image

ROOT = '/workspace'
OUT = os.path.join(ROOT, 'output')
LOG = None
POS_SCALE = 0.05
MAX_STEP = 0.012      # metres per control step while drawing
MAX_STEP_FREE = 0.03
DWELL_MAX = 8  # metres per control step in free space


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    try:
        with open(os.path.join(OUT, 'controller_log.txt'), 'a') as fh:
            fh.write(s + '\n')
    except Exception:
        pass


def parse(reply):
    for blk in reply.get('content', reply) if isinstance(reply, dict) else reply:
        if isinstance(blk, dict) and blk.get('type') == 'text':
            return json.loads(blk['text'])
    raise RuntimeError('no json block')


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.ended = False
        self.complete = False
        self.state = None
        self.remaining = None
        self.last = None

    def observe(self):
        raw = self.ctx.call('robomme_observe')
        self.raw = raw
        p = parse(raw)
        self._upd(p)
        return p

    def _upd(self, p):
        if 'state' in p and p['state'] is not None:
            self.state = np.array(p['state'], float)
        lim = p.get('episode_limits') or {}
        if 'remaining_episode_steps' in lim:
            self.remaining = lim['remaining_episode_steps']
        if p.get('episode_ended'):
            self.ended = True
        for st in p.get('steps', []) or []:
            if isinstance(st, dict):
                if st.get('task_complete'):
                    self.complete = True
                if st.get('episode_ended'):
                    self.ended = True
        if p.get('task_complete'):
            self.complete = True
        self.last = p

    def step(self, rows):
        if self.ended:
            return None
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        if self.remaining is not None:
            rows = rows[:max(0, self.remaining)]
        if not rows:
            return None
        raw = self.ctx.call('robomme_step', actions=rows)
        p = parse(raw)
        self.raw = raw
        self._upd(p)
        return p

    def front_rgb(self):
        try:
            blocks = self.raw.get('content', []) if isinstance(self.raw, dict) else self.raw
            for a in self.last.get('attachments', []):
                if a.get('camera') == 'front' and a.get('kind') == 'rgb':
                    b = blocks[a['content_index']]
                    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b['data']))).convert('RGB')).astype(np.int32)
        except Exception as e:
            log('front_rgb fail', repr(e))
        return None

    def red_at(self, pw):
        img = self.front_rgb()
        if img is None:
            return -1
        K = self.last['camera_intrinsics']['front']['matrix']; pose = self.last['camera_poses']['front']
        u, v = world_to_pixel(pw, K, pose)
        m = red_mask(img)
        u0, v0 = int(round(u)), int(round(v))
        win = m[max(0, v0 - 7):v0 + 8, max(0, u0 - 8):u0 + 9]
        return int(win.sum())

    def rot_correction(self, target_rv):
        """small rotation correction toward target axis-angle (world)."""
        from scipy.spatial.transform import Rotation as Rr
        cur = Rr.from_rotvec(self.state[3:6])
        tgt = Rr.from_rotvec(target_rv)
        err = (tgt * cur.inv()).as_rotvec()
        return np.clip(err / 0.5 * 0.5, -0.2, 0.2)

    def move_to(self, target, max_step, tol=0.004, max_iters=12, chunk=8, rot_target=None):
        target = np.array(target, float)
        for _ in range(max_iters):
            if self.ended:
                return False
            d = target - self.state[:3]
            dist = float(np.linalg.norm(d))
            if dist < tol:
                return True
            n = int(min(chunk, max(1, math.ceil(dist / max_step))))
            rows = []
            # plan n rows of equal displacement along the line, open loop
            per = d / max(1, math.ceil(dist / max_step))
            rc = self.rot_correction(rot_target) if rot_target is not None else np.zeros(3)
            for k in range(n):
                r = list(per / POS_SCALE) + (list(rc) if k == 0 else [0, 0, 0]) + [0.0]
                rows.append(r)
            self.step(rows)
        return float(np.linalg.norm(target - self.state[:3])) < 3 * tol


def load_sequence(packet):
    seq = packet.get('frame_sequence') or {}
    mp = seq.get('manifest_path')
    if not mp:
        import glob
        c = sorted(glob.glob(os.path.join(ROOT, 'runtime/autonomous_controller/runs/*/frames/sequence_000000/manifest.json')),
                   key=os.path.getmtime)
        log('no manifest_path in packet; fallback candidates', c[-3:])
        mp = c[-1]
    m = json.load(open(os.path.join(ROOT, mp)))
    frames = sorted(m['frames'], key=lambda f: f['frame_index'])
    vid = [f for f in frames if f.get('phase') == 'video_demo']
    ex = [f for f in frames if f.get('phase') == 'execution']
    return vid, ex


def best_grid(frames, label):
    best = None
    cands = frames[:3] + frames[-3:] if len(frames) > 6 else frames
    for f in cands:
        try:
            g = grid_from_frame(f, ROOT)
        except Exception as e:
            log('grid fail', label, f.get('frame_index'), repr(e))
            continue
        score = g['n_inliers'] - 10 * g['resid_max']
        if best is None or score > best[0]:
            best = (score, g, f.get('frame_index'))
    if best is None:
        raise RuntimeError('no grid for ' + label)
    g = best[1]
    log(label, 'grid frame', best[2], 'inliers', g['n_inliers'], '/', g['n_total'], 'theta_deg',
        round(math.degrees(g['theta']), 2), 'spacing', g['spacing_rc'], 'resid', round(g['resid_max'], 4))
    return g


def main(ctx):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rb = Robot(ctx)
    pkt = rb.observe()
    log('instruction:', pkt.get('instruction'), 'step', pkt.get('step_id'), 'limits', pkt.get('episode_limits'))
    vid, ex = load_sequence(pkt)
    log('video frames', len(vid), 'exec frames', len(ex))
    gv = best_grid(vid, 'video')
    ge = best_grid(ex if ex else [], 'exec')
    pattern, zlow = extract_pattern(vid, gv)
    log('pattern', pattern, 'zlow', round(zlow, 4))
    if not pattern:
        log('EMPTY pattern; abort')
        return
    rot0 = rb.state[3:6].copy()
    z_draw = zlow + (ge['ztop'] - gv['ztop'])
    z_hover = z_draw + 0.04
    log('z_draw', round(z_draw, 4), 'ztop v/e', gv['ztop'], ge['ztop'])
    with open(os.path.join(OUT, 'plan.json'), 'w') as fh:
        json.dump(dict(pattern=pattern, zlow=zlow, z_draw=z_draw,
                       targets=[list(map(float, ge['node_world'](*n))) for n, _ in pattern]), fh)
    off, offs = video_offsets(vid, gv, pattern, zlow)
    log('video tcp offsets', np.round(offs, 4).tolist(), 'median', np.round(off, 4).tolist())
    ztop = ge['ztop']
    first = True
    results = []
    for (node, lift) in pattern:
        xy = ge['node_world'](*node) + off
        pw = [xy[0] - off[0], xy[1] - off[1], ztop]
        if first or lift:
            if not first:
                rb.move_to([rb.state[0], rb.state[1], z_hover], MAX_STEP_FREE, rot_target=rot0)
            rb.move_to([xy[0], xy[1], z_hover], MAX_STEP_FREE, rot_target=rot0, max_iters=20)
            rb.move_to([xy[0], xy[1], z_draw], 0.008, rot_target=rot0)
            first = False
        else:
            rb.move_to([xy[0], xy[1], z_draw], MAX_STEP, rot_target=rot0, max_iters=20)
            rb.move_to([xy[0], xy[1], z_draw], 0.004, tol=0.0015, max_iters=4, rot_target=rot0)
        red = 0
        k = 0
        for k in range(DWELL_MAX):
            if rb.ended:
                break
            rb.step([[0, 0, 0, 0, 0, 0, 0]])
            red = rb.red_at(pw)
            if red >= 6:
                break
        log('node', node, 'reached', np.round(rb.state[:3], 4).tolist(), 'target', np.round(xy, 4).tolist(),
            'red', red, 'dwell', k + 1, 'remaining', rb.remaining, 'ended', rb.ended, 'complete', rb.complete)
        results.append(dict(node=node, red=red, dwell=k + 1, state=rb.state[:3].tolist()))
        if rb.ended:
            break
    for _ in range(3):
        if rb.ended:
            break
        rb.step([[0, 0, 0, 0, 0, 0, 0]] * 3)
    log('done ended', rb.ended, 'complete', rb.complete, 'remaining', rb.remaining, 'wall', round(time.time() - t0, 1))
    return dict(results=results, ended=rb.ended, complete=rb.complete)
