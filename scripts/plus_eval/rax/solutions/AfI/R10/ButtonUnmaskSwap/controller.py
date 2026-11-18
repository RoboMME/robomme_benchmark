"""ButtonUnmaskSwap controller (v8 = v4 [freeze tracks once static] + exception guard).

Plan: step-0 colour/button detection -> press buttons early while tracking the 8 white
containers (front + wrist depth, occupancy grid + Hungarian tracker) -> observe from a
high pose until containers stop -> pick target containers in instruction order (yaw-aligned
body grasp), put each back down, next.
"""
import os, sys, json, time, re, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot import Robot
from percept import world_points, white_mask, detect, detect_cubes_buttons, box_yaw, container_top_points, detect2, upper_white, cells_of
from tracker import Tracker
from scipy.spatial.transform import Rotation as Rot

OUT = '/workspace/output'
COLORS = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'black', 'white', 'cyan', 'pink', 'gray']
HIGH = np.array([-0.06, 0.0, 0.45])
T0 = time.time()


def log(*a):
    msg = ' '.join(str(x) for x in a)
    print(f'[{time.time()-T0:7.1f}s] {msg}', flush=True)
    try:
        with open(os.path.join(OUT, 'controller_log.txt'), 'a') as f:
            f.write(msg + '\n')
    except Exception:
        pass


def parse_targets(instr):
    s = instr.lower()
    i = s.find('pick')
    part = s[i:] if i >= 0 else s
    found = []
    for m in re.finditer(r'\b(' + '|'.join(COLORS) + r')\b', part):
        found.append(m.group(1))
    return found


class Scene:
    def __init__(self, R):
        self.R = R
        self.buttons = []
        self.tracker = None
        self.labels = {}  # color -> track index
        self.n_cont = 8
        self.static = None
        self.frozen = False

    def cam_points(self, cam):
        R = self.R
        c = R.cams[cam]
        K = R.meta['camera_intrinsics'][cam]['matrix']; pose = R.meta['camera_poses'][cam]
        P, h, s, v = world_points(c['rgb'], c['depth'], K, pose)
        return P, h, s, v

    def detections(self, min_cells=25):
        R = self.R
        pts = []
        ee = R.pos
        for cam in ('front', 'wrist'):
            try:
                P, h, s, v = self.cam_points(cam)
            except Exception:
                continue
            p = upper_white(P, s, v)
            if len(p) and ee[2] < 0.25:
                dx = p[:, 0] - ee[0]; dy = p[:, 1] - ee[1]
                if cam == 'wrist':
                    keep = dx * dx + dy * dy > 0.05 ** 2
                else:
                    # gripper hand: elongated along y (finger axis), plus arm behind (smaller x)
                    keep = ~((np.abs(dy) < 0.10) & (dx > -0.12) & (dx < 0.045))
                p = p[keep]
            pts.append(p)
        out = detect2(pts, static_mask=self.static, ee=None, min_cells=min_cells)
        return [c for c, n in out]

    def update(self):
        if self.tracker is None or self.frozen:
            return
        d = self.detections()
        self.tracker.update(self.R.step_id, d)


def step_and_track(S, rows):
    S.R.step(rows)
    S.update()


def move_track(S, target, grip, tol=0.004, max_steps=80, vmax=0.8, chunk=4):
    R = S.R
    target = np.array(target, float); n = 0
    while n < max_steps and not R.ended:
        err = target - R.pos
        if np.linalg.norm(err) < tol:
            break
        rows = []; e = err.copy()
        for i in range(min(chunk, max_steps - n)):
            d = e / 0.05; m = np.max(np.abs(d))
            if m > vmax: d = d * vmax / m
            rot = list(R.rot_corr()) if i == 0 else [0, 0, 0]
            rows.append(list(d) + rot + [grip]); e = e - d * 0.05
            if np.linalg.norm(e) < tol * 0.5: break
        step_and_track(S, rows); n += len(rows)
    return np.linalg.norm(target - R.pos)


def press_button(S, b):
    R = S.R
    x, y = b
    move_track(S, [x, y, 0.065], 1.0, max_steps=30, tol=0.008, vmax=1.0)
    zs = []
    for k in range(10):
        tgt = np.array([x, y, 0.012])
        err = tgt - R.pos
        d = np.clip(err / 0.05, -0.6, 0.6)
        step_and_track(S, [list(d) + list(R.rot_corr()) + [1.0]])
        zs.append(R.pos[2])
        if R.pos[2] < 0.028 or (len(zs) >= 3 and R.pos[2] < 0.035 and abs(zs[-1] - zs[-2]) < 0.001):
            break
    zmin = min(zs)
    log('press', np.round(b, 3).tolist(), 'zmin', round(zmin, 4), 'step', R.step_id)
    move_track(S, [x, y, 0.065], 1.0, max_steps=6, tol=0.012, vmax=1.0)
    return zmin


def wait_static(S, min_step=150, reserve=420, still_steps=30):
    R = S.R
    limit = R.meta['episode_limits']['episode_step_limit']
    max_step = max(min_step + 50, limit - reserve)
    hist = []
    t_last = time.time()
    while not R.ended and R.step_id < max_step:
        step_and_track(S, [[0, 0, 0] + list(R.rot_corr()) + [1.0]] + [[0, 0, 0, 0, 0, 0, 1.0]] * 4)
        hist.append((R.step_id, [p.copy() for p in S.tracker.pos]))
        hist = hist[-40:]
        if time.time() - T0 > 900:
            log('wall-time guard in wait_static'); return False
        old = [h for h in hist if h[0] <= R.step_id - still_steps]
        if old and R.step_id >= min_step:
            p0 = old[-1][1]
            mv = max(np.linalg.norm(a - b) for a, b in zip(p0, S.tracker.pos))
            if mv < 0.003:
                log('static at step', R.step_id)
                return True
        if R.step_id % 100 < 5:
            log('waiting, step', R.step_id, 'max', max_step)
    log('wait_static hit max_step', R.step_id)
    return False


def refine_container(S, guess):
    """Use top-face points near guess from both cameras; returns center xy, yaw(rad) or None."""
    best = None
    allpts = []
    for cam in ('wrist', 'front'):
        P, h, s, v = S.cam_points(cam)
        pts = container_top_points(P, s, v, guess, radius=0.045, zlo=0.062, zhi=0.085)
        if len(pts) > 15:
            allpts.append((cam, pts))
    if not allpts:
        return None, None, 0
    # prefer wrist
    cam, pts = max(allpts, key=lambda t: (t[0] == 'wrist', len(t[1])))
    # second pass around refined center
    c = np.median(pts[:, :2], 0)
    P, h, s, v = S.cam_points(cam)
    pts = container_top_points(P, s, v, c, radius=0.035, zlo=0.062, zhi=0.085)
    if len(pts) < 15:
        return c, None, len(pts)
    yaw, ext = box_yaw(pts[:, :2])
    c = pts[:, :2].mean(0)
    # center from min-area rect
    ca, sa = np.cos(yaw), np.sin(yaw)
    Rm = np.array([[ca, -sa], [sa, ca]])
    r = (pts[:, :2] - c) @ Rm
    lo = np.percentile(r, 3, 0); hi = np.percentile(r, 97, 0)
    c = c + Rm @ ((lo + hi) / 2)
    return c, yaw, len(pts)


def set_yaw(R, yaw):
    R.rot_target = (Rot.from_euler('z', yaw) * Rot.from_rotvec([np.pi, 0, 0])).as_rotvec()


def pick_container(S, color):
    R = S.R
    ti = S.labels[color]
    for attempt in range(4):
        guess = S.tracker.pos[ti].copy()
        log('pick', color, 'attempt', attempt, 'track pos', guess.round(3))
        set_yaw(R, 0.0)
        move_track(S, [guess[0], guess[1], 0.24], -1.0, max_steps=60, tol=0.006)
        c, yaw, n = refine_container(S, guess)
        log('  refine1', None if c is None else c.round(3), None if yaw is None else round(np.degrees(yaw), 1), n)
        if c is None:
            c = guess
        if np.linalg.norm(c - guess) > 0.045:
            log('  refine far from track; using track'); c = guess
        if yaw is not None:
            set_yaw(R, yaw)
        move_track(S, [c[0], c[1], 0.16], -1.0, max_steps=40, tol=0.004)
        for _ in range(3):
            move_track(S, [c[0], c[1], 0.16], -1.0, max_steps=6, tol=0.002, chunk=2)
        c2, yaw2, n2 = refine_container(S, c)
        log('  refine2', None if c2 is None else c2.round(3), None if yaw2 is None else round(np.degrees(yaw2), 1), n2)
        if c2 is not None and np.linalg.norm(c2 - c) < 0.03:
            c = c2
            if yaw2 is not None:
                set_yaw(R, yaw2)
            move_track(S, [c[0], c[1], 0.14], -1.0, max_steps=20, tol=0.002, chunk=2)
        # descend
        move_track(S, [c[0], c[1], 0.09], -1.0, max_steps=20, tol=0.004, vmax=0.6)
        move_track(S, [c[0], c[1], 0.05], -1.0, max_steps=25, tol=0.003, vmax=0.35, chunk=2)
        zdown = R.pos[2]
        R.step([[0, 0, 0, 0, 0, 0, 1.0]] * 8)
        width = R.state[6] + R.state[7]
        log('  down z', round(zdown, 4), 'closed width', round(width, 4))
        move_track(S, [R.pos[0], R.pos[1], 0.20], 1.0, max_steps=30, vmax=0.5)
        width = R.state[6] + R.state[7]
        log('  lifted width', round(width, 4), 'ended', R.ended, 'complete', R.complete)
        if R.ended:
            return True
        if width > 0.01:
            # verify: container no longer at the table location (top points near c at z~0.07)
            return c
        # failed: open, back up, re-track
        R.step([[0, 0, 0, 0, 0, 0, -1.0]] * 4)
        move_track(S, [R.pos[0], R.pos[1], 0.24], -1.0, max_steps=20)
        S.update()
    return None


def place_back(S, c):
    R = S.R
    move_track(S, [c[0], c[1], 0.10], 1.0, max_steps=40, tol=0.005)
    move_track(S, [c[0], c[1], 0.058], 1.0, max_steps=25, tol=0.003, vmax=0.3, chunk=2)
    R.step([[0, 0, 0, 0, 0, 0, -1.0]] * 6)
    move_track(S, [c[0], c[1], 0.20], -1.0, max_steps=30)
    set_yaw(R, 0.0)


def main_inner(ctx):
    os.makedirs(OUT, exist_ok=True)
    R = Robot(ctx, log=log)
    R.observe()
    instr = R.meta.get('instruction', '')
    targets = parse_targets(instr)
    log('instruction:', instr)
    log('targets:', targets, 'step', R.step_id)
    S = Scene(R)
    # --- step-0 perception (wrist primary, front fallback)
    per = {}
    stat_pts = []
    for cam in ('wrist', 'front'):
        try:
            P, h, s, v = S.cam_points(cam)
        except Exception:
            continue
        per[cam] = detect_cubes_buttons(P, h, s, v)
        m = (P[..., 2] > 0.045) & (P[..., 2] < 0.095)
        stat_pts.append(P[m])
    S.static = cells_of(np.concatenate(stat_pts), dilate=3) if stat_pts else None
    log('static cells', int(S.static.sum()) if S.static is not None else None)
    objs = per.get('wrist') or per.get('front') or {}
    cubes = {}
    plates = []; caps = []
    for col, lst in objs.items():
        for xy, n, z95 in lst:
            if col in ('white', 'gray') and z95 < 0.026 and n > 30:
                plates.append(xy)
            elif col == 'gray' and z95 >= 0.026 and n > 20:
                caps.append(xy)
            elif z95 > 0.026 and n > 25 and col != 'gray':
                cubes.setdefault(col, []).append((n, xy))
    btn = []
    for c in caps:
        if any(np.linalg.norm(c - b) < 0.045 for b in plates) and all(np.linalg.norm(c - x) > 0.06 for x in btn):
            btn.append(c)
    if not btn:
        for b in plates:
            if all(np.linalg.norm(b - x) > 0.07 for x in btn):
                btn.append(b)
    S.buttons = btn
    # a 'white' cube that is a cap? remove white cubes within 0.04 of a button
    if 'white' in cubes:
        cubes['white'] = [t for t in cubes['white'] if all(np.linalg.norm(t[1] - b) > 0.04 for b in btn)]
        if not cubes['white']: del cubes['white']
    cube_xy = {c: max(l, key=lambda t: t[0])[1] for c, l in cubes.items()}
    log('buttons', [b.round(3).tolist() for b in S.buttons])
    log('cubes', {k: [t[1].round(3).tolist() for t in l] for k, l in cubes.items()})
    ncubes = sum(len(l) for l in cubes.values())
    S.n_cont = max(1, ncubes)
    # --- wait for containers to land (arm already moving toward first button), init tracker
    order = sorted(S.buttons, key=lambda b: b[1])
    if order:
        b0 = order[0]
        if np.linalg.norm(order[-1] - R.pos[:2]) < np.linalg.norm(b0 - R.pos[:2]):
            order = order[::-1]
    pre = np.array([order[0][0], order[0][1], 0.30]) if order else HIGH
    init = None
    for k in range(20):
        err = pre - R.pos
        d = np.clip(err / 0.05, -1, 1)
        R.step([list(d) + list(R.rot_corr()) + [1.0]])
        d = S.detections()
        if len(d) >= S.n_cont and R.step_id >= 7:
            init = d; break
    if init is None:
        init = S.detections()
    log('containers init', len(init), 'at step', R.step_id, [p.round(3).tolist() for p in init])
    S.tracker = Tracker(init, t0=R.step_id)
    for col in targets:
        if col in cube_xy:
            dists = [np.linalg.norm(p - cube_xy[col]) for p in S.tracker.pos]
            S.labels[col] = int(np.argmin(dists))
            log('label', col, '-> track', S.labels[col], 'dist', round(min(dists), 3))
        else:
            log('WARNING target colour not found at step 0:', col)
    # --- press buttons, nearest first
    for b in order:
        if R.step_id > 170: break
        press_button(S, b)
    move_track(S, HIGH, 1.0, max_steps=40, tol=0.01, vmax=1.0)
    log('at high pose step', R.step_id, 'tracks', [p.round(3).tolist() for p in S.tracker.pos])
    static_ok = wait_static(S)
    if static_ok:
        # average a few more frames, then freeze: containers no longer move (except ones we lift/place back)
        acc = [np.array([p.copy() for p in S.tracker.pos])]
        for _ in range(3):
            step_and_track(S, [[0, 0, 0] + list(R.rot_corr()) + [1.0]] + [[0, 0, 0, 0, 0, 0, 1.0]] * 2)
            acc.append(np.array([p.copy() for p in S.tracker.pos]))
        med = np.median(np.array(acc), 0)
        for i in range(len(S.tracker.pos)):
            S.tracker.pos[i] = med[i].copy()
        S.frozen = True
        log('tracks frozen')
    log('tracks final', [p.round(3).tolist() for p in S.tracker.pos])
    for col in targets:
        if R.ended: break
        if col not in S.labels:
            log('skip unknown', col); continue
        c = pick_container(S, col)
        if R.ended:
            log('episode ended after', col, 'complete', R.complete); break
        if c is None:
            log('failed to pick', col); continue
        place_back(S, c)
    log('done. ended', R.ended, 'complete', R.complete, 'step', R.step_id)


def main(ctx):
    try:
        main_inner(ctx)
    except Exception:
        log('EXCEPTION', traceback.format_exc())
