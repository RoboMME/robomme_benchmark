"""VideoPlaceOrder controller: analyse conditioning video, pick named cube, place on inferred target."""
import os, sys, json, time, traceback
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from motion import Robot, meta
from geom import Cam
from perception import read_rgb, read_depth, world_points, cube_mask, detect_targets, parse_instruction
from analyze import analyze

OUT = '/workspace/output'

def log(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True)
    try:
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, 'controller_log.txt'), 'a') as f:
            f.write(msg + '\n')
    except Exception:
        pass

def current_frame_files(m):
    """Files of the latest observation (last frame of its sequence manifest)."""
    seq = m.get('frame_sequence') or {}
    mp = seq.get('manifest_path')
    if not mp:
        return None, None
    man = json.load(open(os.path.join('/workspace', mp)))
    fr = man['frames'][-1]
    files = {(x['camera'], x['kind']): os.path.join('/workspace', x['path']) for x in fr['files']}
    return os.path.join('/workspace', mp), (fr, files)

def locate_cube(m, files, color):
    cam = Cam(m['camera_intrinsics']['front'], m['camera_poses']['front'])
    rgb = read_rgb(files[('front', 'rgb')])
    P = world_points(cam, read_depth(files[('front', 'depth')]))
    mk = cube_mask(rgb, color) & (P[..., 2] > 0.015) & (P[..., 2] < 0.08) & (P[..., 0] > -0.45)
    if mk.sum() < 5:
        return None
    pts = P[mk]
    zt = np.percentile(pts[:, 2], 90)
    top = pts[pts[:, 2] > zt - 0.006]
    xy = np.median(top[:, :2], axis=0)
    return np.array([xy[0], xy[1], zt])

def main(ctx):
    t0 = time.time()
    r = Robot(ctx, log)
    m = r.observe()
    instr = m.get('instruction', '')
    log('instruction:', instr, 'remaining', r.remaining)
    color, nth = parse_instruction(instr)
    mp, (fr, files) = current_frame_files(m)
    goal = None
    try:
        goal, info = analyze(mp, instr, log=log)
    except Exception:
        log('analyze failed', traceback.format_exc())
    if goal is None:
        # fallback: any target in current view
        cam = Cam(m['camera_intrinsics']['front'], m['camera_poses']['front'])
        P = world_points(cam, read_depth(files[('front', 'depth')]))
        D = detect_targets(read_rgb(files[('front', 'rgb')]), P, cam)
        goal = D[0][0] if D else np.array([-0.15, 0.0])
        log('FALLBACK goal', goal)
    goal = np.array(goal[:2], float)
    log('goal xy', goal.round(4).tolist(), 'analysis time', round(time.time() - t0, 1))
    cube = locate_cube(m, files, color or 'red')
    log('cube', None if cube is None else cube.round(4).tolist())
    if cube is None:
        log('cube not found'); return
    # refine the goal using current detections (snap to nearest live target)
    try:
        cam = Cam(m['camera_intrinsics']['front'], m['camera_poses']['front'])
        P = world_points(cam, read_depth(files[('front', 'depth')]))
        D = detect_targets(read_rgb(files[('front', 'rgb')]), P, cam)
        if D:
            dd = [np.linalg.norm(p - goal) for p, n in D]
            k = int(np.argmin(dd))
            if dd[k] < 0.03:
                goal = D[k][0]
            log('live targets', [np.round(p, 3).tolist() for p, n in D], 'snapped goal', np.round(goal, 4).tolist())
    except Exception:
        log('snap failed', traceback.format_exc())
    pick = cube[:2]
    gz = max(0.015, cube[2] - 0.02)
    r.move_to([*pick, 0.12], -1, tol=0.006, max_steps=60)
    r.move_to([*pick, gz], -1, tol=0.003, max_steps=50)
    r.grip(1, 12)
    log('closed fingers', r.state[6:8].round(4).tolist())
    r.move_to([*pick, 0.12], 1, tol=0.006, max_steps=50)
    held = (r.state[6] + r.state[7])
    log('lifted', r.state.round(4).tolist(), 'opening', round(held, 4))
    if held < 0.02:
        log('grasp seems to have missed; retrying once')
        m = r.observe()
        _, (fr, files) = current_frame_files(m)
        cube = locate_cube(m, files, color or 'red')
        if cube is not None:
            pick = cube[:2]
            r.grip(-1, 8)
            r.move_to([*pick, 0.12], -1, tol=0.006, max_steps=50)
            r.move_to([*pick, gz], -1, tol=0.003, max_steps=50)
            r.grip(1, 12)
            r.move_to([*pick, 0.12], 1, tol=0.006, max_steps=50)
    r.move_to([*goal, 0.12], 1, tol=0.004, max_steps=80)
    r.move_to([*goal, gz + 0.02], 1, tol=0.003, max_steps=50)
    r.grip(-1, 8)
    log('released at', r.state.round(4).tolist(), 'ended', r.ended)
    if not r.ended:
        r.move_to([*goal, 0.15], -1, tol=0.01, max_steps=20)
    log('done ended', r.ended, 'complete', r.complete, 'remaining', r.remaining, 'time', round(time.time() - t0, 1))
