"""MoveCube autonomous controller (RoboMME-Plus, RAX).

Pipeline: classify the conditioning video's manner from wrist frames -> parse the
front RGB-D scene -> grasp the matching tool (hook / stick / plate / hammer) or the
ball -> plan a collision-checked push/pull (or a ramp roll) -> close-loop on the cube
position -> hold still until the native success signal.
"""
import json, math, os, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np
from PIL import Image

import perception as P
from classify import classify
from scene import Scene, CUBE_H, BASE_XY
from geom import wrap, R2, ang, unit, perp, ObstacleGrid
from motion import Robot, rotate_yaw, EpisodeEnded
import skills as S

OUT = Path('/workspace/output')


class Logger:
    def __init__(self):
        try:
            OUT.mkdir(parents=True, exist_ok=True)
            self.f = open(OUT / 'controller_log.txt', 'a')
        except Exception:
            self.f = None
        self.t0 = time.time()

    def __call__(self, *a):
        msg = f'[{time.time() - self.t0:7.1f}] ' + ' '.join(str(x) for x in a)
        print(msg, flush=True)
        if self.f:
            try:
                self.f.write(msg + '\n'); self.f.flush()
            except Exception:
                pass


def load_video_wrist(meta, log, max_frames=160):
    fs = meta.get('frame_sequence') or {}
    mp = fs.get('manifest_path')
    if not mp:
        return []
    cands = [Path('/workspace') / mp, Path.cwd() / mp, Path(mp)]
    man = None
    for c in cands:
        if c.exists():
            man = json.loads(c.read_text()); break
    if man is None:
        log('[video] manifest not found', mp); return []
    frames = [f for f in man.get('frames', []) if f.get('phase') == 'video_demo']
    frames.sort(key=lambda f: f.get('frame_index', 0))
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in idx]
    out = []
    for f in frames:
        files = {(x['camera'], x['kind']): x['path'] for x in f.get('files', [])}
        p = files.get(('wrist', 'rgb'))
        if not p: continue
        for root in (Path('/workspace'), Path.cwd(), Path('')):
            q = root / p
            if q.exists():
                out.append(np.array(Image.open(q).convert('RGB'))); break
    log(f'[video] {len(frames)} video frames, loaded {len(out)} wrist images')
    return out


# ----------------------------------------------------------------------------- teal tools
def teal_options(scene, kind):
    tools = [t for t in scene.tools if t['kind'] == kind]
    if not tools:
        # fall back on the most similar tool
        if kind == 'hook':
            tools = sorted(scene.tools, key=lambda t: -t['lat'])[:1]
        elif kind == 'plate':
            tools = sorted(scene.tools, key=lambda t: -t['occ'])[:1]
        else:
            tools = sorted(scene.tools, key=lambda t: t['lat'])[:1]
    return tools


def do_teal(R, scene, kind, log):
    cube, T = scene.cube, scene.target
    tools = teal_options(scene, kind)
    if not tools:
        raise S.Fail('no teal tool')
    best = None
    for t in tools:
        h = t['handle']
        if not S.reach_ok(h[:2]):
            log('[teal] handle out of reach', np.round(h, 3)); continue
        zg = float(np.clip(t['handle_top'] - 0.0075, 0.012, 0.03))
        grid = S.obstacle_grid(scene, exclude_pts=[t['pts']], exclude_xy=[(cube, 0.045), (T, 0.04)])
        for flip in (0, 1):
            yg = S.yaw_choice(wrap(ang(t['axis']) + math.pi * flip), 0.0)
            if yg is None: continue
            m0 = S.build_teal_model(t, np.r_[h[:2], zg], yg)
            plan = S.plan_push(m0, cube, T, grid, yg, zg, log=lambda *a: None, pitches=(0.0, 0.45, 0.7, 0.9) if kind == 'stick' else (0.0,))
            relaxed = False
            if plan is None:
                plan = S.plan_push(m0, cube, T, ObstacleGrid(np.zeros((0, 3))), yg, zg, log=lambda *a: None)
                relaxed = True
            if plan is None: continue
            cost = plan['cost'] + 0.1 * abs(yg) + 0.25 * abs(plan['yaw'] - yg) + (5 if relaxed else 0)
            cost += 3.0 if t['length'] < 0.24 else 0.0
            log(f"[teal] tool@{np.round(h[:2],3)} flip={flip} yg={math.degrees(yg):.0f} plan yaw={math.degrees(plan['yaw']):.0f} dev={math.degrees(plan['dev']):.0f} phi={math.degrees(plan['phi']):.0f} cost={cost:.2f} relaxed={relaxed}")
            if best is None or cost < best[0]:
                best = (cost, t, yg, zg, grid)
    if best is None:
        raise S.Fail('no feasible teal plan')
    _, t, yg, zg, grid = best
    scene.grid = grid
    h = t['handle']
    for attempt in range(2):
        op = S.descend_grasp(R, h[:2], zg, yg, label=kind)
        if 0.008 < op < 0.065:
            break
        log('[teal] grasp failed, retry'); R.set_gripper(-1.0, n=6)
        R.goto(np.r_[R.pos[:2], 0.12], speed=0.6)
        sc = S.look(R); tt = [x for x in sc.tools if np.linalg.norm(x['handle'][:2] - h[:2]) < 0.06]
        if tt: t = tt[0]; h = t['handle']; zg = float(np.clip(t['handle_top'] - 0.0075, 0.012, 0.03))
    else:
        raise S.Fail('grasp failed')
    tcp = R.pos.copy()
    model = S.build_teal_model(t, tcp, R.yaw_u)
    log(f'[teal] model kind={model.kind} contact_q={np.round(model.contact_q,3)} dir_q={np.round(model.dir_q,3)} '
        f'standoff={model.standoff:.3f} zbot={model.zbot:.3f} lat_tol={model.lat_tol:.3f}')
    # lift gently
    R.goto(np.r_[tcp[:2], tcp[2] + 0.03], speed=0.15, tol=0.004)
    model = remeasure(R, model, t, log) or model
    return push_and_finish(R, model, scene, tcp[2], kind, log)


def remeasure(R, model, t, log):
    """Re-measure the held teal tool from the front depth (slip check)."""
    try:
        sc = S.look(R)
        tcp = R.pos.copy()
        L = sc.L
        m = (L['teal'] | L['gray'] | L['salmon']) & sc.valid
        Pm = sc.Pimg
        rel = Pm[..., :2] - tcp[:2]
        dist = np.hypot(rel[..., 0], rel[..., 1])
        zz = Pm[..., 2]
        sel = m & (dist < model.extent + 0.05) & (zz > tcp[2] - 0.035) & (zz < tcp[2] + 0.02) & (zz > 0.012)
        pts = Pm[sel]
        # keep only points close to the predicted tool body
        xy_all, _ = model.pts_world(tcp, R.yaw_u, sub=1)
        g = ObstacleGrid(np.c_[xy_all, np.ones(len(xy_all))], res=0.01)
        near = np.array([g.max_height_near(p[None], 0.03) > 0 for p in pts[:, :2]]) if len(pts) else np.zeros(0, bool)
        pts = pts[near]
        if len(pts) < 30:
            log(f'[remeasure] few points {len(pts)}'); return None
        xy, _ = model.pts_world(tcp, R.yaw_u, sub=1)
        # predicted far end vs measured far end
        dpred = np.linalg.norm(xy - tcp[:2], axis=1); dmeas = np.hypot(*(pts[:, :2] - tcp[:2]).T)
        far_pred = xy[dpred > np.percentile(dpred, 92)].mean(0)
        far_meas = pts[dmeas > np.percentile(dmeas, 92)]
        fm = far_meas[:, :2].mean(0)
        err = np.linalg.norm(fm - far_pred)
        log(f'[remeasure] n={len(pts)} far_pred={np.round(far_pred,3)} far_meas={np.round(fm,3)} err={err:.3f} '
            f'far z={np.median(far_meas[:,2]):.3f} tcp z={tcp[2]:.3f}')
        if err > 0.02 and err < 0.12:
            tt = dict(t); tt['pts'] = pts; tt['axis'] = unit(fm - tcp[:2])
            m2 = S.build_teal_model(tt, tcp, R.yaw_u)
            log(f'[remeasure] adopting measured model contact_q={np.round(m2.contact_q,3)}')
            return m2
    except EpisodeEnded:
        raise
    except Exception as e:
        log('[remeasure] error', e)
    return None


def push_and_finish(R, model, scene, grasp_z, kind, log):
    T = scene.target
    for attempt in range(3):
        m, work_z = S.run_tool_push(R, model, scene, grasp_z, kind, log=log)
        cube, _ = S.cube_now(R)
        d = unit(T[:2] - cube[:2]) if cube is not None else m.dir_world(R.yaw_u)
        if cube is not None and np.linalg.norm(T[:2] - cube[:2]) < 0.004:
            d = m.dir_world(R.yaw_u)
        S.retreat(R, m, work_z, d, log)
        for w in range(4):
            if S.wait_success(R, 50):
                return True
            cube, _ = S.cube_now(R)
            if cube is None: return False
            dist = np.linalg.norm(T[:2] - cube[:2])
            log(f'[finish] no success yet, cube-target dist={dist:.4f}')
            if dist > 0.013:
                break
        else:
            return False
        scene.cube = cube
    return False


# ----------------------------------------------------------------------------- hammer
def do_hammer(R, scene, log):
    hm = scene.hammer
    if hm is None or 'tan_tip' not in hm:
        raise S.Fail('no hammer')
    cube, T = scene.cube, scene.target
    g = hm['grasp']; zg = float(max(0.03, hm['ztop'] - 0.015))
    allp = np.concatenate([hm['slate'], hm['tan']])
    grid = S.obstacle_grid(scene, exclude_pts=[allp], exclude_xy=[(cube, 0.045), (T, 0.04)])
    scene.grid = grid
    tipdir = unit(hm['tan_tip'][:2] - g[:2])
    best = None
    for flip in (0, 1):
        yg = S.yaw_choice(wrap(ang(tipdir) + math.pi * flip + math.pi / 2), 0.0)
        if yg is None: continue
        m0 = S.build_hammer_model(hm, np.r_[g[:2], zg], yg)
        plan = S.plan_push(m0, cube, T, grid, yg, zg, log=lambda *a: None)
        if plan is None: continue
        cost = plan['cost'] + 0.1 * abs(yg) + 0.25 * abs(plan['yaw'] - yg)
        log(f"[hammer] flip={flip} yg={math.degrees(yg):.0f} plan yaw={math.degrees(plan['yaw']):.0f} cost={cost:.2f}")
        if best is None or cost < best[0]: best = (cost, yg)
    if best is None:
        yg = S.yaw_choice(wrap(ang(tipdir) + math.pi / 2), 0.0) or 0.0
    else:
        yg = best[1]
    op = S.descend_grasp(R, g[:2], zg, yg, z_above=min(0.30, hm['ztop'] + 0.10), label='hammer')
    if not (0.005 < op < 0.07):
        raise S.Fail('hammer grasp failed')
    tcp = R.pos.copy()
    model = S.build_hammer_model(hm, tcp, R.yaw_u)
    log(f'[hammer] model contact_q={np.round(model.contact_q,3)} zbot={model.zbot:.3f}')
    R.goto(np.r_[tcp[:2], tcp[2] + 0.02], speed=0.15, tol=0.004)
    return push_and_finish(R, model, scene, tcp[2], 'hammer', log)


# ----------------------------------------------------------------------------- ball + ramp
def plan_release(scene, log):
    rp, cube = scene.ramp, scene.cube
    coef = rp['coef']; up = unit(rp['up'])
    zf = lambda p: coef[0] * p[0] + coef[1] * p[1] + coef[2]
    rpts = rp['pts'][:, :2]
    best = None
    for t in np.arange(0.02, 1.2, 0.004):
        p = cube[:2] + up * t
        dmin = np.min(np.hypot(rpts[:, 0] - p[0], rpts[:, 1] - p[1]))
        if dmin < 0.012 and zf(p) < rp['zmax'] + 0.005:
            best = t
    if best is None:
        return None
    t_rel = best - 0.035
    p = cube[:2] + up * t_rel
    th = math.atan(rp['slope'])
    zs = zf(p)
    log(f'[ball] release t={t_rel:.3f} p={np.round(p,3)} surface z={zs:.3f} slope={math.degrees(th):.1f}deg')
    return p, zs, th, up


def do_ball(R, scene, log):
    if scene.ball is None or scene.ramp is None:
        raise S.Fail('ball or ramp missing')
    rel = plan_release(scene, log)
    if rel is None:
        raise S.Fail('no release point')
    p, zs, th, up = rel
    T = scene.target
    fall = -up
    dct = unit(T[:2] - scene.cube[:2])
    log(f'[ball] fall dir {np.round(fall,3)} cube->target {np.round(dct,3)} angle {math.degrees(math.acos(np.clip(fall@dct,-1,1))):.1f}')
    b = scene.ball['c']
    yg = S.yaw_choice(wrap(ang(up)), 0.0)
    if yg is None or abs(yg) > math.pi / 2:
        yg = S.yaw_choice(wrap(ang(up) + math.pi), 0.0)
    for attempt in range(3):
        zb = float(np.clip(b[2] - 0.002, 0.018, 0.035))
        op = S.descend_grasp(R, b[:2], zb, yg, label='ball')
        if 0.025 < op < 0.075:
            break
        log('[ball] grasp failed, retry')
        R.set_gripper(-1.0, n=6); R.goto(np.r_[R.pos[:2], 0.14], speed=0.6)
        sc = S.look(R)
        if sc.ball is None: raise S.Fail('ball lost')
        b = sc.ball['c']
    else:
        raise S.Fail('ball grasp failed')
    r_ball = 0.0275
    z_rel = zs + r_ball / math.cos(th) + 0.012
    z_hi = max(0.30, rp_top(scene) + 0.12)
    R.goto(np.r_[R.pos[:2], z_hi], speed=0.5, tol=0.006)
    R.goto(np.r_[p, z_hi], speed=0.6, tol=0.006)
    R.goto(np.r_[p, z_rel + 0.03], speed=0.3, tol=0.004)
    R.goto(np.r_[p, z_rel], speed=0.1, tol=0.003, max_iter=6)
    log(f'[ball] releasing at {np.round(R.pos,3)}')
    R.set_gripper(-1.0, n=3)
    R.goto(np.r_[R.pos[:2], R.pos[2] + 0.10], speed=0.6, tol=0.01, max_iter=3)
    for k in range(20):
        R.hold(10)
        if k % 3 == 2:
            sc = S.look(R)
            if sc.cube is not None:
                log(f'[ball] wait {k} cube={np.round(sc.cube,3)} dist={np.linalg.norm(sc.cube[:2]-T[:2]):.3f}')
    return R.task_complete


def rp_top(scene):
    return scene.ramp['zmax'] if scene.ramp else 0.2


# ----------------------------------------------------------------------------- main
def main(ctx):
    log = Logger()
    R = Robot(ctx, log=log, step_budget_reserve=3)
    try:
        R.observe()
        log('[start] step', R.meta.get('step_id'), 'limits', R.meta.get('episode_limits'), 'state', np.round(R.state, 3))
        wr = load_video_wrist(R.meta, log)
        manner, info = classify(wr) if wr else ('unknown', {})
        log('[classify]', manner, info)
        scene = Scene(R.meta, R.imgs, log=log)
        log('[scene]', json.dumps(scene.summary()))
        if scene.cube is None or scene.target is None:
            raise S.Fail('cube/target not found')
        if manner == 'unknown':
            manner = 'stick'
        R.grip = -1.0
        if manner == 'ball':
            ok = do_ball(R, scene, log)
        elif manner == 'hammer':
            ok = do_hammer(R, scene, log)
        else:
            ok = do_teal(R, scene, manner, log)
        log('[done] result', ok, 'task_complete', R.task_complete)
    except EpisodeEnded as e:
        log('[end] episode ended', R.task_complete, e)
    except Exception as e:
        log('[error]', repr(e)); log(traceback.format_exc())
        # keep still: avoid further disturbances
        try:
            if not R.ended:
                R.hold(5)
        except Exception:
            pass
    log('[exit] task_complete', R.task_complete, 'step', (R.meta or {}).get('step_id'), 'calls', R.n_calls)
    return {'task_complete': bool(R.task_complete)}
