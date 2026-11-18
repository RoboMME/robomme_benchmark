import sys, os, re, math, json, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np
from motion import Robot, parse
from perception import track_cubes, frame_points, white_points, locate_container
from skills import refine_container

ROOT = '/workspace'
COLOR_WORDS = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
LOGF = None

def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    try:
        with open(LOGF, 'a') as f:
            f.write(s + '\n')
    except Exception:
        pass

def parse_instruction(text):
    text = text.lower()
    after = text.split('then', 1)[-1]
    parts = re.split(r',|followed by|and finally|finally', after)
    goals = []
    for p in parts:
        for c in COLOR_WORDS:
            if re.search(r'\b' + c + r'\b', p):
                plural = bool(re.search(r'\ball\b', p)) or (c + ' cubes') in p
                goals.append((c, plural))
                break
    return goals

def find_containers(frames, cubes):
    ex = [f for f in frames if f['phase'] != 'video_demo']
    ex = ex[-1] if ex else frames[-1]
    clouds = {}
    for cam in ('wrist', 'front'):
        try:
            rgb, pts = frame_points(ROOT, ex, cam)
            clouds[cam] = white_points(rgb, pts, zmin=0.02, zmax=0.12)
        except Exception as e:
            log('cloud fail', cam, repr(e))
    for cb in cubes:
        cb['container'] = None
        for cam in ('wrist', 'front'):
            if cam not in clouds:
                continue
            c = locate_container(clouds[cam], cb['pos'][:2], radius=0.045)
            if c and c['n_top'] > 20:
                cb['container'] = c; cb['container_cam'] = cam
                break
    return cubes

def pick_targets(goals, cubes):
    plan = []
    used = []
    for color, plural in goals:
        cand = [c for c in cubes if c['color'] == color]
        cand.sort(key=lambda c: -c['n_frames'])
        if not plural:
            cand = cand[:1]
        for c in cand:
            xy = np.array(c['container']['center'] if c.get('container') else c['pos'][:2])
            if any(np.linalg.norm(xy - u) < 0.025 for u in used):
                continue
            used.append(xy)
            plan.append({'color': color, 'xy': xy.tolist(),
                         'ztop': (c['container'] or {}).get('ztop', 0.073),
                         'yaw_deg': (c['container'] or {}).get('yaw_deg', 0.0), 'cube': c['pos']})
    return plan

def choose_yaw(robot, yaw_deg):
    """Closing axis must be along a box edge: yaw_deg or yaw_deg+90; choose one closest to current."""
    cur = robot.closing_yaw()
    best = None
    for k in range(4):
        y = math.radians(yaw_deg + 90 * k)
        d = abs((y - cur + math.pi / 2) % math.pi - math.pi / 2)
        if best is None or d < best[0]:
            best = (d, y)
    return best[1]

def lift_container(ctx, robot, tgt):
    xy = np.array(tgt['xy'])
    ztop = tgt['ztop'] if 0.05 < tgt['ztop'] < 0.1 else 0.073
    yaw = choose_yaw(robot, tgt['yaw_deg'])
    robot.move([xy[0], xy[1], max(robot.state()[2], ztop + 0.08)], yaw=None, grip=-1, vmax=0.7)
    robot.move([xy[0], xy[1], ztop + 0.06], yaw=yaw, grip=-1, vmax=0.6)
    if robot.ended: return 'ended'
    # refine with close view
    c, cam = refine_container(ctx, robot, xy, log=log)
    if c is not None and np.linalg.norm(np.array(c['center']) - xy) < 0.03:
        xy = np.array(c['center'])
        if 0.05 < c['ztop'] < 0.1:
            ztop = c['ztop']
        yaw = choose_yaw(robot, c['yaw_deg'])
    log(f"grasp target {tgt['color']} xy={np.round(xy,4)} ztop={ztop:.3f} yaw={math.degrees(yaw):.1f}")
    for attempt in range(2):
        robot.move([xy[0], xy[1], ztop + 0.05], yaw=yaw, grip=-1, vmax=0.4)
        robot.move([xy[0], xy[1], ztop - 0.028], yaw=yaw, grip=-1, vmax=0.25)
        if robot.ended: return 'ended'
        w = robot.gripper(1.0, n=15)
        log(f'  attempt {attempt} grip width {w:.4f} state {np.round(robot.state(),3).tolist()}')
        if robot.ended: return 'ended'
        if w < 0.025:
            robot.gripper(-1.0, n=8)
            robot.move([xy[0], xy[1], ztop + 0.06], yaw=yaw, grip=-1, vmax=0.4)
            c, cam = refine_container(ctx, robot, xy, log=log)
            if c is not None and np.linalg.norm(np.array(c['center']) - xy) < 0.04:
                xy = np.array(c['center']); yaw = choose_yaw(robot, c['yaw_deg'])
            continue
        robot.move([xy[0], xy[1], ztop + 0.10], yaw=None, vmax=0.3)
        w2 = float(robot.state()[6:8].sum())
        log(f'  lifted z={robot.state()[2]:.3f} width {w2:.4f} step {robot.last.get("step_id")}')
        if robot.ended: return 'ended'
        # put it back down where it was
        robot.move([xy[0], xy[1], ztop - 0.024], yaw=None, vmax=0.3)
        robot.gripper(-1.0, n=10)
        robot.move([xy[0], xy[1], ztop + 0.07], yaw=None, grip=-1, vmax=0.4)
        return 'lifted' if w2 > 0.025 else 'dropped'
    return 'failed'

def get_manifest(robot):
    seq = (robot.last or {}).get('frame_sequence')
    if seq and seq.get('manifest_path'):
        return os.path.join(ROOT, seq['manifest_path'])
    import glob
    c = sorted(glob.glob(os.path.join(ROOT, 'runtime/autonomous_controller/runs/*/frames/sequence_000000/manifest.json')),
               key=os.path.getmtime)
    return c[-1] if c else None

def main(ctx, targets_override=None, manifest_override=None):
    try:
        _main(ctx, targets_override, manifest_override)
    except Exception as e:
        log('FATAL controller error', repr(e), traceback.format_exc())

def _main(ctx, targets_override=None, manifest_override=None):
    global LOGF
    os.makedirs('/workspace/output', exist_ok=True)
    LOGF = '/workspace/output/controller_log.txt'
    t0 = time.time()
    robot = Robot(ctx, log=log)
    robot.observe()
    instr = robot.last.get('instruction', '')
    log('instruction:', instr, 'step', robot.last.get('step_id'))
    goals = parse_instruction(instr)
    log('goals:', goals)
    man = manifest_override or get_manifest(robot)
    log('manifest:', man)
    cubes, frames = track_cubes(man, ROOT, log=log)
    for c in cubes:
        log('cube', c['color'], np.round(c['pos'], 3).tolist(), c['n_frames'], c['first'], c['last'], c['cams'])
    cubes = find_containers(frames, cubes)
    for c in cubes:
        log('container for', c['color'], c.get('container'))
    plan = pick_targets(goals, cubes)
    if targets_override is not None:
        plan = [p for p in plan if p['color'] in targets_override]
    log('plan:', json.dumps(plan))
    log(f'perception took {time.time()-t0:.1f}s')
    try:
        json.dump({'goals': goals, 'cubes': cubes, 'plan': plan}, open('/workspace/output/plan.json', 'w'), default=str)
    except Exception:
        pass
    for tgt in plan:
        if robot.ended:
            break
        try:
            res = lift_container(ctx, robot, tgt)
        except Exception as e:
            log('lift error', repr(e), traceback.format_exc())
            if robot.ended: break
            try:
                robot.observe()
            except Exception:
                break
            res = 'error'
        log('result', tgt['color'], res, 'step', robot.last.get('step_id'), 'ended', robot.ended,
            'complete', robot.task_complete)
    if not robot.ended:
        try:
            robot.observe()
        except Exception:
            pass
    log('done; ended', robot.ended, 'task_complete', robot.task_complete, f'time {time.time()-t0:.1f}s')
