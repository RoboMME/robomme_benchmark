"""VideoUnmaskSwap controller: track containers through the conditioning video and the
live shuffle (RGB-D, world frame), wait until motion stops, then lift containers in the
instructed colour order."""
import os, sys, json, time, re, base64, io, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from geom import backproject, load_frame
from robot import Robot, parse, yaw_of
from track3 import Tracker
from tracker import rect_fit, frame_points, cells, components
from skills import wrap90

ROOT = '/workspace'
COLORS = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'black', 'white']
T0 = time.time()
DRY = False

def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

def target_order(instr):
    s = instr.lower()
    pos = []
    for c in COLORS:
        for mm in re.finditer(r'\b' + c + r'\b', s):
            pos.append((mm.start(), c))
    pos.sort()
    order = []
    for _, c in pos:
        if c not in order: order.append(c)
    return order

def frame_from_reply(reply, meta):
    """Latest live frame (dict cam->(rgb, P)) plus its metadata."""
    fs = meta.get('frame_sequence') or {}
    mp = fs.get('manifest_path')
    if mp and os.path.exists(os.path.join(ROOT, mp)):
        try:
            man = json.load(open(os.path.join(ROOT, mp)))
            fr = sorted(man['frames'], key=lambda f: f['frame_index'])[-1]
            if fr.get('step_id') == meta.get('step_id') or True:
                return load_frame(ROOT, fr), fr
        except Exception as e:
            log('manifest read failed', e)
    # fallback: decode attachments
    blocks = reply['content'] if isinstance(reply, dict) and 'content' in reply else reply
    out = {}
    imgs = {}
    for att in meta.get('attachments', []):
        b = blocks[att['content_index']]
        data = base64.b64decode(b.get('data'))
        imgs[(att['camera'], att['kind'])] = np.asarray(Image.open(io.BytesIO(data)))
    for cam in ('front', 'wrist'):
        if (cam, 'rgb') in imgs and (cam, 'depth') in imgs:
            rgb = imgs[(cam, 'rgb')][..., :3].astype(np.uint8)
            dep = imgs[(cam, 'depth')].astype(np.float64) * meta['depth'][cam].get('meters_per_unit', 0.001)
            out[cam] = (rgb, backproject(dep, meta['camera_intrinsics'][cam]['matrix'], meta['camera_poses'][cam]))
    return out, meta

def container_yaw(fr, xy, ee=None):
    p, _ = frame_points(fr, 0.062, 0.13)
    c = cells(p)
    if len(c) == 0: return 0.0
    sel = np.linalg.norm(c - np.asarray(xy), axis=1) < 0.032
    if sel.sum() < 15: return 0.0
    t, a, b = rect_fit(c[sel])
    return t

def main(ctx):
    try:
        return _main(ctx)
    except Exception as e:
        import traceback; log('CONTROLLER ERROR', repr(e)); traceback.print_exc()
        return None

def _main(ctx):
    rb = Robot(ctx, log=log)
    reply = ctx.call('robomme_observe'); meta = parse(reply); rb.last = meta
    instr = meta.get('instruction', '')
    order = target_order(instr)
    log('instruction:', instr, '| order', order, '| limits', meta.get('episode_limits'))
    # ---- 1. track through the conditioning video
    fs = meta.get('frame_sequence') or {}
    man = json.load(open(os.path.join(ROOT, fs['manifest_path'])))
    vid = sorted([f for f in man['frames'] if f['phase'] == 'video_demo'], key=lambda f: f['frame_index'])
    ex0 = [f for f in man['frames'] if f['phase'] == 'execution']
    tr = Tracker(log=log)
    tr.init_from_video(ROOT, vid)
    base = float(vid[-1].get('source_frame_index', len(vid)))
    t_exec0 = float(ex0[0].get('source_frame_index', base + 1)) if ex0 else base + 1
    if ex0:
        tr.update(load_frame(ROOT, ex0[0]), t_exec0)
    log('after video:', [(np.round(x, 3).tolist(), c) for x, c in zip(tr.tracks, tr.content)])
    # ---- 2. wait for the live shuffle to stop, tracking every step
    still = 0; waited = 0; last_fr = None
    while waited < 300 and not rb.ended:
        reply = ctx.call('robomme_step', actions=[[0, 0, 0, 0, 0, 0, -1.0]])
        meta = parse(reply); rb.last = meta; waited += 1
        rb._check_end(meta)
        fr, fm = frame_from_reply(reply, meta)
        last_fr = fr
        stp, mt = tr.update(fr, t_exec0 + meta.get('step_id', waited), ee=meta['state'][:3])
        mx = float(stp.max())
        still = still + 1 if mx < 0.0015 else 0
        if waited % 10 == 0: log(f'wait step {meta.get("step_id")} max move {mx:.4f} still {still} unmatched {(~mt).sum()}')
        if still >= 12 and waited >= 25: break
    log('settled after', waited, 'steps:', [(np.round(x, 3).tolist(), c) for x, c in zip(tr.tracks, tr.content)])
    # ---- 3. pick in order
    plan = []
    for col in order:
        idx = [i for i, c in enumerate(tr.content) if c == col]
        plan += idx
    log('plan', [(tr.content[i], np.round(tr.tracks[i], 3).tolist()) for i in plan])
    if DRY:
        log('DRY run: stopping before picks'); return tr
    from skills import pick_container
    for n, i in enumerate(plan):
        if rb.ended: break
        xy = tr.tracks[i]
        yaw = container_yaw(last_fr, xy) if last_fr else 0.0
        log(f'pick {n}: {tr.content[i]} at {np.round(xy,3)} yaw {math.degrees(yaw):.0f}')
        held, w = pick_container(rb, xy, yaw, log=log, place_back=(n < len(plan) - 1),
                                 get_frame=lambda: frame_from_reply(None, rb.last)[0])
        log(f'  held={held} width={w:.4f} ended={rb.ended} complete={rb.complete} step={rb.last.get("step_id")}')
    if not rb.ended and not rb.complete:
        # hold still a few steps so the final state registers
        rb.step([[0, 0, 0, 0, 0, 0, rb.grip]] * 5)
    log('done: ended', rb.ended, 'complete', rb.complete)
