"""VideoPlaceButton controller.

1. Read the conditioning video (front + wrist frames) from the reset frame_sequence.
2. Find the button press (wrist close-up of the white button base; fallback: front-view
   occlusion of the button), the cube's rest placements, the target slots and the
   pairwise target swaps (shell game).  The answer target is the target object the cube
   was first placed on after the press, followed through all later swaps.
3. Pick the instructed cube (pose from depth), place it on that target's current position.
"""
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from vision import cube_mask, detect_targets, target_mask  # noqa: E402
from analysis import analyze_video  # noqa: E402
from seqload import load_sequence  # noqa: E402
from motion import Robot  # noqa: E402
from live import decode, mask_world, cube_pose_from_points, backproject  # noqa: E402
from scipy import ndimage  # noqa: E402

OUT = '/workspace/output'
HOVER_Z = 0.12
GRASP_Z = 0.021
PLACE_Z = 0.037


class Logger:
    def __init__(self):
        os.makedirs(OUT, exist_ok=True)
        self.f = open(os.path.join(OUT, 'controller_log.txt'), 'a')
        self.t0 = time.time()

    def __call__(self, *a):
        s = '[%.1fs] ' % (time.time() - self.t0) + ' '.join(str(x) for x in a)
        print(s)
        try:
            self.f.write(s + '\n'); self.f.flush()
        except Exception:
            pass


def parse_color(instr):
    instr = (instr or '').lower()
    for c in ('red', 'blue', 'green'):
        if c + ' cube' in instr:
            return c
    for c in ('red', 'blue', 'green'):
        if c in instr:
            return c
    return 'red'


def run_analysis(meta, log):
    seqd = meta.get('frame_sequence') or {}
    mp = seqd.get('manifest_path')
    if not mp:
        log('no frame_sequence manifest in observation')
        return None
    _, seq = load_sequence(mp)
    vid = [r for r in seq if r['phase'] == 'video_demo']
    ex = [r for r in seq if r['phase'] != 'video_demo']
    log('sequence frames: video %d exec %d' % (len(vid), len(ex)))
    use = vid + ex[:1]
    times = []
    for r in vid:
        times.append(r['t'] if r['t'] is not None else (r['src'] or 0) / 20.0)
    times.append((times[-1] if times else 0.0) + 0.05)
    # guarantee monotonic times
    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            times[i] = times[i - 1] + 0.05
    color = parse_color(meta.get('instruction'))
    res = analyze_video([r['front'] for r in use], times, color, log=log, wrists=[r['wrist'] for r in use])
    res['color'] = color
    return res


def target_world(meta, imgs, px, log):
    """World xy of the target ring nearest to image pixel px in the current front view."""
    img = imgs[('front', 'rgb')]
    tm = ndimage.binary_closing(target_mask(img), iterations=1)
    lab, n = ndimage.label(tm)
    best = None
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        if len(xs) < 30:
            continue
        d = np.hypot(xs.mean() - px[0], ys.mean() - px[1])
        if best is None or d < best[0]:
            best = (d, k)
    if best is not None and best[0] < 12:
        m = lab == best[1]
        pts = mask_world(meta, imgs, 'front', m)
        if len(pts) > 10:
            c = np.median(pts, 0)
            log('target ring blob dist %.1fpx -> world %s (n=%d)' % (best[0], np.round(c, 4).tolist(), len(pts)))
            return c, best[0]
    pts, ok = backproject(meta, 'front', [px[0]], [px[1]], imgs[('front', 'depth')])
    log('target fallback pixel backprojection', np.round(pts[0], 4).tolist())
    return pts[0], 99.0


def cube_estimate(rb, color, cam, log):
    meta, imgs = decode(rb.ctx.call('robomme_observe'))
    rb.last = meta
    m = cube_mask(imgs[(cam, 'rgb')], color)
    # keep largest component
    lab, n = ndimage.label(m)
    if n == 0:
        log('cube not visible in', cam)
        return None, meta, imgs
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    m = lab == (int(np.argmax(sizes)) + 1)
    pts = mask_world(meta, imgs, cam, m)
    # discard points far from table height range
    pts = pts[(pts[:, 2] > -0.01) & (pts[:, 2] < 0.12)]
    pose = cube_pose_from_points(pts)
    if pose:
        log('cube(%s) center %s yaw %.3f size %s n=%d' % (cam, np.round(pose['center'], 4).tolist(), pose['yaw'],
                                                         np.round(pose['size'], 3).tolist(), pose['n']))
    return pose, meta, imgs


def grasp(rb, color, log, attempt=0):
    pose, meta, imgs = cube_estimate(rb, color, 'front', log)
    if pose is None:
        return False
    c = pose['center']
    rb.move_to([c[0], c[1], HOVER_Z], grip=-1.0)
    p2, meta, imgs = cube_estimate(rb, color, 'wrist', log)
    if p2 is not None and np.hypot(*(p2['center'][:2] - c[:2])) < 0.05:
        pose = p2
    c = pose['center']
    rb.move_to([c[0], c[1], 0.09], grip=-1.0)
    rb.align_yaw(pose['yaw'])
    p3, meta, imgs = cube_estimate(rb, color, 'wrist', log)
    if p3 is not None and np.hypot(*(p3['center'][:2] - c[:2])) < 0.04:
        pose = p3
        rb.align_yaw(pose['yaw'])
    c = pose['center']
    ztop = float(c[2])
    gz = float(np.clip(ztop - 0.019, 0.012, 0.03))
    rb.move_to([c[0], c[1], 0.06], grip=-1.0)
    rb.move_to([c[0], c[1], gz], grip=-1.0, tol=0.004)
    log('at grasp', np.round(rb.state(), 4).tolist())
    rb.gripper(1.0, n=10)
    st = rb.state()
    width = float(st[6] + st[7])
    log('gripper width after close %.4f' % width)
    rb.move_to([c[0], c[1], HOVER_Z], grip=1.0)
    st = rb.state()
    width = float(st[6] + st[7])
    held = width > 0.015
    # verify by vision: cube should be lifted (wrist view cube large / front cube z high)
    pose_after, _, _ = cube_estimate(rb, color, 'front', log)
    if pose_after is not None:
        log('cube top z after lift %.3f' % pose_after['center'][2])
        if pose_after['center'][2] < 0.06:
            held = False
    log('grasp held=%s width=%.4f' % (held, width))
    return held


def place(rb, tw, log):
    rb.move_to([tw[0], tw[1], HOVER_Z], grip=1.0)
    rb.move_to([tw[0], tw[1], 0.07], grip=1.0)
    rb.move_to([tw[0], tw[1], PLACE_Z], grip=1.0, tol=0.004)
    log('at place', np.round(rb.state(), 4).tolist())
    rb.gripper(-1.0, n=8)
    if not rb.ended:
        rb.move_to([tw[0], tw[1], HOVER_Z], grip=-1.0)


def main(ctx):
    log = Logger()
    rb = Robot(ctx, log=log)
    try:
        meta = rb.observe()
        log('instruction:', meta.get('instruction'), 'limits', meta.get('episode_limits'))
        try:
            res = run_analysis(meta, log)
        except Exception:
            log('analysis failed', traceback.format_exc())
            res = None
        color = parse_color(meta.get('instruction'))
        _, imgs = decode(ctx.call('robomme_observe'))
        meta0 = rb.observe()
        meta_full, imgs = decode(ctx.call('robomme_observe'))
        if res and res.get('final_px') is not None:
            px = res['final_px']
        else:
            # fallback: any target that the cube is not on
            dets = detect_targets(imgs[('front', 'rgb')])
            px = [dets[0][0], dets[0][1]] if dets else [128, 90]
            log('FALLBACK target px', px)
        tw, dpx = target_world(meta_full, imgs, px, log)
        log('answer target px %s world %s' % (np.round(px, 1).tolist(), np.round(tw, 4).tolist()))
        try:
            with open(os.path.join(OUT, 'analysis.json'), 'w') as f:
                json.dump(dict(final_px=list(map(float, px)), target_world=list(map(float, tw)),
                               press=res.get('press') if res else None,
                               events=[dict(t=e['t'], pair=e['pair']) for e in (res or {}).get('events', [])],
                               placements=[{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in p.items()}
                                           for p in (res or {}).get('placements', [])]), f, default=str)
        except Exception:
            log('could not save analysis', traceback.format_exc())
        for attempt in range(3):
            if rb.ended:
                break
            ok = grasp(rb, color, log, attempt)
            if rb.ended:
                break
            if not ok:
                log('grasp failed, retry', attempt)
                rb.gripper(-1.0, n=6)
                continue
            # re-acquire target position right before placing
            meta_full, imgs = decode(ctx.call('robomme_observe'))
            tw2, d2 = target_world(meta_full, imgs, px, log)
            if d2 < 12 and np.hypot(*(tw2[:2] - tw[:2])) < 0.03:
                tw = tw2
            place(rb, tw, log)
            log('after place: ended=%s complete=%s step=%s' % (rb.ended, rb.complete, (rb.last or {}).get('step_id')))
            if rb.ended:
                break
            # not ended: cube possibly off target -> retry
        log('FINAL ended=%s complete=%s' % (rb.ended, rb.complete))
    except Exception:
        log('controller exception', traceback.format_exc())
