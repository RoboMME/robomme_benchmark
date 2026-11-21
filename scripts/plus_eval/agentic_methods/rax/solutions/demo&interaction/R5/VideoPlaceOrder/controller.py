"""VideoPlaceOrder controller (v1).
1) parse instruction (cube colour, ordinal k)
2) analyse the conditioning video: track target identities through shuffles and the
   cube's rest events; answer = initial slot of the target under the k-th placement
3) pick the cube (front RGB-D localisation, top-down grasp) and place it on that target.
"""
import json, os, sys, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import video, percept
import robot as rb
from robot import log

OUT = Path('/workspace/output')


def load_video(meta):
    seq = meta.get('frame_sequence') or {}
    mp = seq.get('manifest_path')
    if not mp:
        return []
    man = json.loads((Path('/workspace') / mp).read_text())
    frames = []
    for f in sorted(man['frames'], key=lambda f: f.get('frame_index', 0)):
        if f.get('phase') != 'video_demo':
            continue
        files = {(x['camera'], x['kind']): x['path'] for x in f['files']}
        p = files.get(('front', 'rgb'))
        if not p:
            continue
        try:
            a = np.asarray(Image.open(Path('/workspace') / p).convert('RGB'))
        except Exception:
            continue
        frames.append((float(f.get('timestamp_s', 0.0)), a, f, files.get(('front', 'depth'))))
    return frames


def zmap(fmeta, depth_mm):
    H, W = depth_mm.shape
    vs, us = np.mgrid[0:H, 0:W]
    return rb.backproject(fmeta, depth_mm, 'front', us.ravel(), vs.ravel())[:, 2].reshape(H, W)


def with_depth(frames):
    """(t, rgb, meta, depth_path) -> (t, rgb, world_z); None if any depth frame is missing."""
    out = []
    for t, a, fm, dp in frames:
        if not dp:
            return None
        try:
            d = np.asarray(Image.open(Path('/workspace') / dp))
            out.append((t, a, zmap(fm, d)))
        except Exception:
            return None
    return out


def choose_target(frames, color, k, last):
    res = video.analyze(frames, color, log=log)
    slots = res['slots']
    ev = [e for e in res['placements'] if e['t0'] > 0.3]  # the start rest is not a placement
    log('placements slots', [e['slot'] for e in ev], 'ids', [e['target'] for e in ev])
    if not ev:
        return None, slots, 'none'
    if last:
        k = len(ev)
    if k is None:
        k = 1
    if len(ev) >= k:
        e = ev[k - 1]
    else:
        e = ev[-1]
    if e['target'] is not None and e['target'] < len(slots):
        return slots[e['target']], slots, 'identity'
    return slots[e['slot']], slots, 'slot'


CONFIGS = [
    dict(ADAPTIVE=False, GATE0=12.0, MERGE_RATIO=1.4),
    dict(ADAPTIVE=False, GATE0=15.0, MERGE_RATIO=1.3),
    dict(ADAPTIVE=False, GATE0=10.0, MERGE_RATIO=1.5),
    dict(ADAPTIVE=False, GATE0=13.0, MERGE_RATIO=1.6),
    dict(ADAPTIVE=False, GATE0=11.0, MERGE_RATIO=1.25),
]


NORMAL_TABLE = np.array([188.5, 112.4, 69.9])
DEPTH_CONFIGS = [
    dict(DEPTH_MODE=True, ADAPTIVE=False, GATE0=12.0, MERGE_RATIO=1.4),
    dict(DEPTH_MODE=True, ADAPTIVE=False, GATE0=15.0, MERGE_RATIO=1.3),
    dict(DEPTH_MODE=True, ADAPTIVE=False, GATE0=10.0, MERGE_RATIO=1.5),
    dict(DEPTH_MODE=True, ADAPTIVE=False, GATE0=13.0, MERGE_RATIO=1.6),
    dict(DEPTH_MODE=True, ADAPTIVE=False, GATE0=11.0, MERGE_RATIO=1.25),
]
LIGHT_CONFIGS = [
    dict(ADAPTIVE=True, GATE0=12.0, MERGE_RATIO=1.4),
    dict(ADAPTIVE=True, GATE0=15.0, MERGE_RATIO=1.3),
    dict(ADAPTIVE=True, GATE0=10.0, MERGE_RATIO=1.5),
    dict(ADAPTIVE=False, GATE0=12.0, MERGE_RATIO=1.4),
    dict(ADAPTIVE=True, GATE0=13.0, MERGE_RATIO=1.6),
]


def special_lighting(frames):
    M = np.array([fr[1][160:].reshape(-1, 3).astype(float).mean(0) for fr in frames[::10]])
    dev = float(np.linalg.norm(M.mean(0) - NORMAL_TABLE))
    sd = float(M.std(0).max())
    log('lighting check: mean dev', round(dev, 2), 'temporal sd', round(sd, 3))
    return dev > 10 or sd > 0.2


def choose_target_ensemble(frames, color, k, last, configs=None, budget_s=240.0):
    t0 = time.time()
    if configs is None:
        configs = CONFIGS
        if special_lighting(frames):
            fz = None
            try:
                fz = with_depth(frames)
            except Exception:
                log('depth load failed', traceback.format_exc())
            if fz is not None:
                log('special lighting: depth-band target detection', round(time.time() - t0, 1), 's')
                frames, configs = fz, DEPTH_CONFIGS
            else:
                configs = LIGHT_CONFIGS
    frames = [fr[:3] if (len(fr) > 2 and not isinstance(fr[2], dict)) else fr[:2] for fr in frames]
    answers = []
    for cfg in configs:
        if time.time() - t0 > budget_s:
            break
        for kk, vv in cfg.items():
            setattr(video, kk, vv)
        try:
            uv, slots, how = choose_target(frames, color, k, last)
        except Exception:
            log('config failed', cfg, traceback.format_exc())
            continue
        if uv is not None:
            answers.append((np.asarray(uv, float), how, cfg))
            log('config', cfg, '->', np.round(uv, 1).tolist(), how)
    for kk, vv in CONFIGS[0].items():
        setattr(video, kk, vv)
    video.DEPTH_MODE = False
    if not answers:
        return None, None, 'none'
    # vote: cluster answers within 6 px; identity answers weigh 1, slot fallbacks 0.5
    best, best_score = None, -1
    for i, (uv, how, cfg) in enumerate(answers):
        score = sum((1.0 if h == 'identity' else 0.5) for u, h, c in answers if np.linalg.norm(u - uv) < 6)
        score += 0.01 if i == 0 else 0.0
        if score > best_score:
            best, best_score = (uv, how), score
    log('vote', np.round(best[0], 1).tolist(), 'score', best_score, 'of', len(answers))
    return best[0], None, 'vote:' + best[1]


def main(ctx):
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    R = rb.Robot(ctx)
    meta = R.observe()
    instr = meta.get('instruction', '')
    color, k, last = video.parse_instruction(instr)
    log('instruction:', instr, '->', color, k, last)
    color = color or 'red'
    target_uv = None
    try:
        frames = load_video(meta)
        log('video frames', len(frames), 'load s', round(time.time() - t_start, 1))
        if frames:
            target_uv, slots, how = choose_target_ensemble(frames, color, k, last)
            log('target uv', None if target_uv is None else np.round(target_uv, 1).tolist(), 'via', how,
                'analysis s', round(time.time() - t_start, 1))
    except Exception:
        log('video analysis failed', traceback.format_exc())
    # live target blobs
    a = R.imgs[('front', 'rgb')]
    live = video.blobs(video.target_mask(a))
    log('live targets', [(round(x, 1), round(y, 1), n) for x, y, n in live])
    if target_uv is None:
        if not live:
            log('no targets visible; abort'); return
        target_uv = np.array(live[0][:2])
    if live:
        d = [np.hypot(x - target_uv[0], y - target_uv[1]) for x, y, n in live]
        j = int(np.argmin(d))
        if d[j] < 10:
            target_uv = np.array(live[j][:2])
    T = percept.pixel_world(R.meta, R.imgs, int(round(target_uv[0])), int(round(target_uv[1])), r=1)
    log('target world', np.round(T, 4).tolist())
    pick_and_place(R, color, T)
    log('done: ended', R.ended, 'complete', R.complete, 'total s', round(time.time() - t_start, 1))


def pick_and_place(R, color, T):
    for attempt in range(3):
        if R.ended:
            return
        R.observe()
        c = percept.cube_world(R.meta, R.imgs, color)
        log('attempt', attempt, 'cube', c)
        if c is None:
            # move up and look again
            p = R.state[:3]
            R.move_to([p[0], p[1], 0.25], -1)
            R.observe()
            c = percept.cube_world(R.meta, R.imgs, color)
            if c is None:
                log('cube not found'); return
        x, y = c['xy']
        R.move_to([x, y, 0.15], -1)
        # refine with the wrist camera (top-down)
        R.observe()
        cw = percept.cube_world(R.meta, R.imgs, color, cam='wrist')
        if cw is not None and np.linalg.norm(cw['xy'] - c['xy']) < 0.04:
            x, y = 0.5 * (x + cw['xy'][0]), 0.5 * (y + cw['xy'][1])
            log('wrist refine', np.round(cw['xy'], 4).tolist(), '->', round(x, 4), round(y, 4))
            R.move_to([x, y, 0.15], -1)
        zg = float(np.clip(c['ztop'] - 0.015, 0.012, 0.04))
        R.move_to([x, y, zg], -1, max_iter=8)
        R.gripper(1, 15)
        f = R.state[6:8].sum()
        log('closed width', round(f, 4))
        p = R.state[:3]
        R.move_to([p[0], p[1], 0.15], 1)
        if R.ended:
            return
        R.observe()
        c2 = percept.cube_world(R.meta, R.imgs, color)
        held = R.state[6:8].sum() > 0.02 and c2 is not None and c2['ztop'] > 0.09
        log('held?', held, 'width', round(R.state[6:8].sum(), 4), 'cube', c2)
        if not held:
            R.gripper(-1, 10)
            continue
        R.move_to([T[0], T[1], 0.15], 1)
        R.move_to([T[0], T[1], T[2] + 0.042], 1, max_iter=8)
        R.gripper(-1, 10)
        if R.ended:
            return
        p = R.state[:3]
        R.move_to([p[0], p[1], 0.15], -1)
        return
