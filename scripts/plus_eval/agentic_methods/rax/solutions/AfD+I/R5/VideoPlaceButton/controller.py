"""VideoPlaceButton controller.
1) analyse conditioning video: button press, first placement of the instructed cube after the press,
   follow that target through pairwise swaps to its final slot (front-camera pixel);
2) localise cube and target with RGB-D, pick (yaw aligned) and place on the target."""
import os, sys, json, math, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from robot import Robot, log, LOG, axang_to_R, Rz, rotate_to, yaw_delta, closing_angle
from percept import cube_world, target_world_near
from video import analyze2
from frames import load_sequence

OUT = '/workspace/output'

def save(name, obj):
    try:
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, name), 'w') as f:
            json.dump(obj, f, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o), indent=1)
    except Exception:
        pass

def color_of(instr):
    s = instr.lower()
    for c in ('red', 'green', 'blue'):
        if c + ' cube' in s:
            return c
    for c in ('red', 'green', 'blue'):
        if c in s:
            return c
    return 'red'

def find_manifest(meta):
    fs = meta.get('frame_sequence') or {}
    return fs.get('manifest_path')

def pick(rb, color):
    rb.observe()
    c = cube_world(rb.meta, rb.imgs, color, 'front')
    log('cube front', c)
    if c is None:
        return False
    above = c['center'].copy(); above[2] = max(c['top_z'] + 0.08, 0.11)
    rb.target_R = axang_to_R(np.array([math.pi, 0, 0])) if False else None
    rb.target_R = axang_to_R(rb.state[3:6])
    rb.move_to(above, -1)
    rb.observe()
    cw = cube_world(rb.meta, rb.imgs, color, 'wrist')
    log('cube wrist', cw)
    if cw is not None and np.linalg.norm(cw['center'][:2] - c['center'][:2]) < 0.05:
        c = cw
    ca = closing_angle(rb)
    d = yaw_delta(c['yaw'], ca)
    Rt = Rz(d) @ axang_to_R(rb.state[3:6])
    rotate_to(rb, Rt)
    rb.target_R = Rt
    rb.observe()
    cw = cube_world(rb.meta, rb.imgs, color, 'wrist')
    if cw is not None and np.linalg.norm(cw['center'][:2] - c['center'][:2]) < 0.05:
        c = cw
    xy = c['center'][:2]
    rb.move_to([xy[0], xy[1], c['top_z'] + 0.03], -1, tol=0.004)
    g = [xy[0], xy[1], max(c['top_z'] - 0.018, 0.012)]
    rb.move_to(g, -1, tol=0.003)
    rb.gripper(1, 10)
    width = rb.state[6] + rb.state[7]
    p = rb.state[:3].copy(); p[2] = 0.13
    rb.move_to(p, 1)
    rb.observe()
    width = rb.state[6] + rb.state[7]
    cf = cube_world(rb.meta, rb.imgs, color, 'front')
    lifted = cf is not None and cf['top_z'] > 0.08
    log('grasp width', width, 'cube after lift', None if cf is None else cf['center'], 'lifted', lifted)
    return width > 0.01 and lifted

def place(rb, tgt):
    t = tgt
    rb.move_to([t[0], t[1], 0.13], 1)
    rb.move_to([t[0], t[1], t[2] + 0.024], 1, tol=0.003)
    rb.gripper(-1, 6)
    if not rb.ended:
        p = rb.state[:3].copy(); p[2] = 0.12
        rb.move_to(p, -1)

def main(ctx):
    t0 = time.time()
    rb = Robot(ctx)
    meta = rb.observe()
    instr = meta.get('instruction', '')
    color = color_of(instr)
    log('instruction', instr, 'color', color, 'step', meta.get('step_id'))
    first_meta, first_imgs = rb.meta, rb.imgs
    pixel = None; res = {}
    try:
        mp = find_manifest(meta)
        log('manifest', mp)
        frames, _ = load_sequence(mp)
        log('frames', len(frames), 'load s', round(time.time() - t0, 1))
        res = analyze2(frames, color, log=log)
        pixel = res.get('pixel')
        if pixel is None and res.get('slots'):
            # fallback: slot nearest the cube's last position in the video ... use first slot
            pixel = tuple(res['slots'][0])
            log('FALLBACK pixel', pixel)
    except Exception as e:
        log('video analysis error', repr(e), traceback.format_exc())
    save('analysis.json', {k: v for k, v in res.items()})
    if pixel is None:
        log('no target pixel; abort placement attempt with centre target guess')
        pixel = (128.0, 90.0)
    tw = target_world_near(first_meta, first_imgs, pixel)
    log('target world', tw, 'analysis s', round(time.time() - t0, 1))
    if tw is None:
        save('log.json', LOG); return
    tgt = tw['center']
    for attempt in range(3):
        if rb.ended:
            break
        ok = pick(rb, color)
        log('pick attempt', attempt, ok, 'step', rb.meta.get('step_id'))
        if ok:
            place(rb, tgt)
            break
        else:
            rb.gripper(-1, 6)
            p = rb.state[:3].copy(); p[2] = 0.13
            rb.move_to(p, -1)
    log('done ended', rb.ended, 'success', rb.success, 'step', rb.meta.get('step_id'), 'wall', round(time.time() - t0, 1))
    save('log.json', LOG)
