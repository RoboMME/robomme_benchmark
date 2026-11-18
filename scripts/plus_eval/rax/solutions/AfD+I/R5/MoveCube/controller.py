import sys, os, json, math, time, traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
from motion import Robot
from perception import *
from skills import *

OUT = Path('/workspace/output')

def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / 'controller_log.txt', 'a') as f: f.write(s + '\n')
    except Exception:
        pass

# ---------------------------------------------------------------- manner from video
def classify_manner(rb, m0):
    seq = m0.get('frame_sequence') or {}
    man = json.loads((WS / seq['manifest_path']).read_text())
    vid = [f for f in man['frames'] if f.get('phase') == 'video_demo']
    if len(vid) < 3:
        return 'unknown', None
    vid.sort(key=lambda f: f['frame_index'])
    sel = vid[::max(1, len(vid) // 32)] + [vid[-1]]
    imgs = [load_frame(f)['front_rgb'] for f in sel]
    res = moved_objects(imgs)
    scores = {}
    best_blob = {}
    for c, lst in res.items():
        for b, frac in lst:
            if frac >= 0.5:
                sc = b['n'] * frac
                if sc > scores.get(c, 0): scores[c] = sc; best_blob[c] = b
    log('video moved scores', {k: round(v, 1) for k, v in scores.items()})
    if not scores: return 'unknown', None
    cat = max(scores, key=scores.get)
    if cat in ('beige', 'dark'): return 'hammer', None
    if cat == 'yellow': return 'ramp', None
    # pink: classify by metric shape using video frame 0 depth
    F0 = load_frame(sel[0])
    P0 = world_map(F0['front_depth'], F0['K']['front'], F0['poses']['front'])
    b = best_blob['pink']
    pts = blob_world(b, P0)
    # merge nearby pink fragments
    m0c = color_masks(F0['front_rgb'])
    allp = [blob_world(q, P0) for q in blobs(m0c['pink'], 3)]
    c0 = pts.mean(0)
    pts = np.concatenate([q for q in allp if len(q) and np.linalg.norm(q.mean(0)[:2] - c0[:2]) < 0.07] + [pts])
    cls, ext = pink_class(pts)
    log('video pink tip', cls, np.round(ext, 3))
    return cls, ext

# ---------------------------------------------------------------- tool push
def push_with_tool(rb, sc, tool_kind, grasp_p, axis_yaw, z_grasp, tip_p, tip_r, tip_measure):
    yaw = wrap(axis_yaw)
    if yaw > math.pi / 2: yaw -= math.pi
    if yaw < -math.pi / 2: yaw += math.pi
    log('grasp', tool_kind, 'at', np.round(grasp_p, 3), 'yaw', round(yaw, 2), 'zg', round(z_grasp, 3))
    ok = grasp_at(rb, grasp_p, yaw, z_grasp, log)
    if not ok:
        log('grasp failed; retry slightly lower')
        rb.move_to([grasp_p[0], grasp_p[1], z_grasp + 0.05], -1, max_steps=30, yaw=yaw)
        ok = grasp_at(rb, grasp_p, yaw, max(0.008, z_grasp - 0.006), log)
        if not ok: return False
    return push_after_grasp(rb, sc, grasp_p, yaw, z_grasp, tip_p, tip_r, tip_measure)

def push_after_grasp(rb, sc, grasp_p, yaw, z_grasp, tip_p, tip_r, tip_measure):
    # lift
    rb.move_to(rb.state[:3] + [0, 0, 0.06], 1, max_steps=40, speed=0.3, yaw=yaw)
    tcp = rb.state[:3].copy()
    sc.grab()
    exp = np.array(tip_p) + (tcp - np.array(grasp_p)) * [0, 0, 1]
    tip = tip_measure(sc, tcp, exp)
    if tip is None:
        log('tip not seen after lift; using prior'); tip = exp
    off = tip - tcp
    log('tip', np.round(tip, 3), 'offset', np.round(off, 3))
    # rotate so the tool points along the push direction
    cube = sc.cube(); tgt = sc.target()
    if cube is None or tgt is None:
        log('cube/target missing', cube, tgt); return False
    u = (tgt - cube)[:2]; u /= np.linalg.norm(u) + 1e-9
    need = wrap(ang(u) - ang(off[:2]))
    y0 = cur_yaw(rb)
    tolr = math.radians(45)
    d = 0.0 if abs(need) <= tolr + math.radians(5) else need - math.copysign(tolr, need)
    if abs(wrap(y0 + d)) > math.radians(160):
        d = d - math.copysign(abs(wrap(y0 + d)) - math.radians(160), wrap(y0 + d))
    log('rotate by', round(math.degrees(d), 1), 'need', round(math.degrees(need), 1), 'yaw0', round(math.degrees(y0), 1))
    rb.move_to(tcp + [0, 0, 0.05], 1, max_steps=30, speed=0.4, yaw=y0)
    if abs(d) > 0.02:
        rb.move_to(rb.state[:3], 1, max_steps=120, yaw=wrap(y0 + d), tol=0.01, rot_rate=0.1)
    yawp = wrap(y0 + d)
    c, s_ = math.cos(d), math.sin(d)
    off = np.array([c * off[0] - s_ * off[1], s_ * off[0] + c * off[1], off[2]])
    sc.grab(); tcp = rb.state[:3].copy()
    t2 = tip_measure(sc, tcp, tcp + off)
    if t2 is not None:
        off = t2 - tcp
    log('offset after rotate', np.round(off, 3))
    # push loop
    zpush = max(0.013, z_grasp + 0.004)
    stall = 0; prev_d = None
    for it in range(25):
        if rb.ended: break
        sc.grab()
        cube = sc.cube(); tgt = sc.target() if it == 0 or tgt is None else tgt
        if cube is None:
            log('cube not visible'); stall += 1
            if stall > 3: break
            rb.move_to(rb.state[:3] + [0, 0, 0.04], 1, max_steps=20, yaw=yawp); continue
        e = (tgt - cube)[:2]; dist = float(np.linalg.norm(e))
        log(f'it{it} cube', np.round(cube[:2], 3), 'tgt', np.round(tgt[:2], 3), 'dist', round(dist, 3), 'step', rb.last.get('step_id'))
        if dist < 0.008 or rb.complete: break
        u = e / dist
        tcp = rb.state[:3].copy(); tipn = tcp + off
        rel = tipn[:2] - cube[:2]
        along = float(rel @ u); lat = float(rel[0] * u[1] - rel[1] * u[0])
        contact = 0.02 + tip_r
        behind = (-contact - 0.05) < along < (-contact + 0.012)
        at_h = abs(tcp[2] - zpush) < 0.012
        if behind and abs(lat) < 0.010 and at_h:
            stepd = min(dist, 0.04)
            goal_tip = cube[:2] + u * (stepd - contact + 0.003)
            g = np.array([goal_tip[0] - off[0], goal_tip[1] - off[1], zpush])
            rb.move_to(g, 1, max_steps=40, speed=0.12, tol=0.003, yaw=yawp)
        elif behind and abs(lat) < 0.03 and at_h:
            log('lateral fix, along', round(along, 3), 'lat', round(lat, 3))
            back = tipn[:2] - u * 0.02
            rb.move_to([back[0] - off[0], back[1] - off[1], zpush], 1, max_steps=20, speed=0.15, tol=0.003, yaw=yawp)
            start_tip = cube[:2] - u * (contact + 0.02)
            rb.move_to([start_tip[0] - off[0], start_tip[1] - off[1], zpush], 1, max_steps=30, speed=0.15, tol=0.003, yaw=yawp)
        else:
            start_tip = cube[:2] - u * (contact + 0.025)
            g = np.array([start_tip[0] - off[0], start_tip[1] - off[1], zpush])
            log('reposition, along', round(along, 3), 'lat', round(lat, 3), 'goal tcp', np.round(g, 3))
            rb.move_to([tcp[0], tcp[1], zpush + 0.10], 1, max_steps=40, speed=0.4, yaw=yawp)
            rb.move_to([g[0], g[1], zpush + 0.10], 1, max_steps=80, speed=0.5, yaw=yawp)
            rb.move_to(g, 1, max_steps=50, speed=0.25, tol=0.003, yaw=yawp)
        if rb.ik_errors >= 4:
            log('too many IK errors; stop'); break
        if rb.complete: break
    log('push done; complete', rb.complete)
    if not rb.ended:
        rb.move_to(rb.state[:3] + [0, 0, 0.06], 1, max_steps=30, speed=0.2, yaw=yawp)
    return True

# ---------------------------------------------------------------- scoop
def norm_yaw(axis_yaw):
    yaw = wrap(axis_yaw)
    if yaw > math.pi / 2: yaw -= math.pi
    if yaw < -math.pi / 2: yaw += math.pi
    return yaw

def scoop_carry(rb, sc, grasp_p, axis_yaw, z_grasp, tip_p, tip_measure):
    yaw = norm_yaw(axis_yaw)
    log('scoop grasp at', np.round(grasp_p, 3), 'yaw', round(yaw, 2), 'zg', round(z_grasp, 3))
    tgt0 = sc.target()
    ok = grasp_at(rb, grasp_p, yaw, z_grasp, log)
    if not ok:
        rb.move_to([grasp_p[0], grasp_p[1], z_grasp + 0.05], -1, max_steps=30, yaw=yaw)
        gp2 = np.array(grasp_p, float); d0 = np.array(tip_p)[:2] - gp2[:2]; d0 /= np.linalg.norm(d0) + 1e-9
        gp2[:2] -= d0 * 0.012
        ok = grasp_at(rb, gp2, yaw, max(0.010, z_grasp - 0.004), log)
        if not ok: log('scoop grasp failed'); return False
    zrest = float(rb.state[2])
    rb.move_to(rb.state[:3] + [0, 0, 0.06], 1, max_steps=40, speed=0.3, yaw=yaw)
    tcp = rb.state[:3].copy(); sc.grab()
    tip = tip_measure(sc, tcp, np.array(tip_p) + [0, 0, 0.06])
    if tip is None: tip = np.array(tip_p) + [0, 0, 0.06]
    off = tip - tcp; a = off[:2] / (np.linalg.norm(off[:2]) + 1e-9)
    log('scoop plate offset', np.round(off, 3))
    for attempt in range(3):
        if rb.ended or rb.complete: break
        sc.grab(); cube = sc.cube(); tgt = sc.target() if sc.target() is not None else tgt0
        if cube is None: log('cube not visible'); break
        log('scoop attempt', attempt, 'cube', np.round(cube, 3), 'tgt', np.round(tgt, 3))
        start_c = cube[:2] - a * (0.035 + 0.02 + 0.025)
        end_c = cube[:2] + a * 0.005
        zs = zrest - 0.002
        s0 = np.array([start_c[0] - off[0], start_c[1] - off[1], zs])
        s1 = np.array([end_c[0] - off[0], end_c[1] - off[1], zs])
        rb.move_to([rb.state[0], rb.state[1], zs + 0.10], 1, max_steps=40, speed=0.4, yaw=yaw)
        rb.move_to([s0[0], s0[1], zs + 0.10], 1, max_steps=80, speed=0.5, yaw=yaw)
        rb.move_to(s0, 1, max_steps=50, speed=0.2, tol=0.003, yaw=yaw)
        rb.move_to(s1, 1, max_steps=60, speed=0.08, tol=0.003, yaw=yaw)
        if rb.ended: break
        rb.move_to(rb.state[:3] + [0, 0, 0.045], 1, max_steps=40, speed=0.06, tol=0.003, yaw=yaw)
        sc.grab(); cube = sc.cube()
        lifted = cube is not None and sc.cube_top() > 0.07
        log('after scoop cube', None if cube is None else np.round(cube, 3), 'top', round(sc.cube_top(), 3), 'lifted', lifted)
        if not lifted:
            continue
        co = cube[:2] - rb.state[:2]
        g = np.array([tgt[0] - co[0], tgt[1] - co[1], rb.state[2] + 0.02])
        rb.move_to(g, 1, max_steps=100, speed=0.15, tol=0.004, yaw=yaw)
        rb.move_to([g[0], g[1], zs + 0.004], 1, max_steps=60, speed=0.06, tol=0.003, yaw=yaw)
        sc.grab(); c2 = sc.cube(); log('placed over target, cube', None if c2 is None else np.round(c2, 3), 'complete', rb.complete)
        if rb.complete or rb.ended: break
        # quick pull-back so the cube slides off the open front edge
        rb.step([[-a[0] * 1.0, -a[1] * 1.0, 0, 0, 0, 0, 1]] * 3)
        rb.step([[0, 0, 0, 0, 0, 0, 1]] * 10)
        sc.grab(); c3 = sc.cube(); log('after pull-back cube', None if c3 is None else np.round(c3, 3), 'complete', rb.complete)
        break
    return True

# ---------------------------------------------------------------- ramp + ball
def ramp_plan(sc):
    bls = blobs(sc.m['ramp'], 50)
    cube = sc.cube()
    if not bls or cube is None: return None
    rp = blob_world(bls[0], sc.P)
    rp = rp[rp[:, 2] > 0.004]
    if len(rp) < 30: return None
    A = np.c_[rp[:, 0], rp[:, 1], np.ones(len(rp))]
    keep = np.ones(len(rp), bool)
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(A[keep], rp[keep, 2], rcond=None)
        res = np.abs(A @ coef - rp[:, 2]); keep = res < 0.008
    grad = coef[:2]; dh = -grad / (np.linalg.norm(grad) + 1e-9)
    top = rp[keep]
    tgt = sc.target()
    aim = cube[:2].copy()
    if tgt is not None:
        w = (tgt - cube)[:2]; w = w / (np.linalg.norm(w) + 1e-9)
        if w @ dh > 0.5: aim = cube[:2] - w * 0.04 + dh * 0.04   # contact normal along cube->target
    cube = np.array([aim[0], aim[1], 0.02])
    log('ramp plane', np.round(coef, 3), 'slope', round(float(np.linalg.norm(grad)), 3), 'downhill', np.round(dh, 3), 'zmax', round(float(top[:, 2].max()), 3))
    # release ball-centre height calibrated on one run: TCP z 0.082 -> cube travel 0.18 m
    dneed = float(np.linalg.norm((tgt - sc.cube())[:2])) if tgt is not None else 0.18
    h_need = float(np.clip(0.082 * (dneed + 0.005) / 0.18, 0.05, 0.16))
    cands = []
    for s_ in np.linspace(0.03, 0.6, 115):
        p = cube[:2] - dh * s_
        near = top[np.linalg.norm(top[:, :2] - p, axis=1) < 0.012]
        if len(near) >= 3:
            zs = float(coef[0] * p[0] + coef[1] * p[1] + coef[2])
            cands.append((s_, p, zs))
    if not cands:
        log('no ramp point on the uphill line'); return None
    smax = max(c[0] for c in cands)
    ok_c = [c for c in cands if c[0] <= smax - 0.025 and c[1][0] > -0.37]
    if not ok_c: ok_c = cands
    s_, p, z = min(ok_c, key=lambda c: abs(c[2] + 0.035 - h_need))
    log('ramp release: dneed', round(dneed, 3), 'h_need', round(h_need, 3), 'surface z', round(z, 3), 's', round(s_, 3), 'smax', round(smax, 3))
    return {'p': p, 'z': z, 'dh': dh, 's': s_, 'cube': cube}

def ramp_ball(rb, sc):
    plan = ramp_plan(sc)
    log('ramp plan', plan)
    bl = blobs(sc.m['yellow'], 10)
    if not bl: log('no yellow ball'); return False
    pts = blob_world(bl[0], sc.P)
    ball = pts.mean(0); ztop = float(np.percentile(pts[:, 2], 95))
    zc = max(0.012, ztop - 0.022)
    log('ball', np.round(ball, 3), 'ztop', round(ztop, 3))
    if plan is None: return False
    by = norm_yaw(ang(plan['dh']))
    ok = grasp_at(rb, ball, by, zc, log)
    if not ok:
        rb.move_to([ball[0], ball[1], 0.10], -1, max_steps=30, yaw=by)
        ok = grasp_at(rb, ball, by, max(0.010, zc - 0.006), log)
        if not ok: log('ball grasp failed'); return False
    p, z = plan['p'], plan['z']
    zr = max(z + 0.035, 0.03)
    zc = max(zr + 0.05, 0.15)
    rb.move_to([rb.state[0], rb.state[1], zc], 1, max_steps=60, speed=0.4, yaw=by)
    rb.move_to([p[0], p[1], zc], 1, max_steps=80, speed=0.4, yaw=by)
    rb.move_to([p[0], p[1], zr], 1, max_steps=60, speed=0.15, yaw=by, tol=0.003)
    log('release at', np.round(rb.state[:3], 3))
    rb.step([[0, 0, 0, 0, 0, 0, -1]] * 4)
    rb.step([[0, 0, 0.6, 0, 0, 0, -1]] * 4)
    for _ in range(4):
        if rb.ended or rb.complete: break
        rb.step([[0, 0, 0, 0, 0, 0, -1]] * 20)
    sc.grab(); log('after release cube', sc.cube(), 'tgt', sc.target(), 'complete', rb.complete, 'ended', rb.ended)
    return True

# ---------------------------------------------------------------- main
def main(ctx):
    t0 = time.time()
    rb = Robot(ctx, log)
    m0 = rb.observe()
    log('instruction', m0.get('instruction'), 'limits', m0.get('episode_limits'))
    try:
        manner, ext = classify_manner(rb, m0)
    except Exception as e:
        log('classify error', repr(e)); traceback.print_exc(); manner = 'unknown'
    log('MANNER', manner, 'time', round(time.time() - t0, 1))
    sc = Scene(rb).grab()
    log('cube', sc.cube(), 'target', sc.target())
    try:
        run_manner(rb, sc, manner, ext)
    except Exception as e:
        log('manner error', repr(e)); traceback.print_exc()
    log('END complete', rb.complete, 'ended', rb.ended, 'steps_left', rb.steps_left, 'time', round(time.time() - t0, 1))

def run_manner(rb, sc, manner, ext=None):
    if manner in ('hook', 'ball', 'scoop'):
        tools, handles, tips = find_teal_tools(sc)
        log('tools', [(t['cls'], np.round(t['tip']['ext'], 3).tolist(), np.round(t['handle']['c'], 3).tolist()) for t in tools])
        if not tools: log('no teal tool'); return
        if ext is not None:
            t = min(tools, key=lambda t: np.hypot(t['tip']['ext'][0] - ext[0], t['tip']['ext'][1] - ext[1]))
        else:
            cand = [t for t in tools if t['cls'] == manner] or tools
            t = max(cand, key=lambda t: t['tip']['n'])
        log('chosen tool', t['cls'], np.round(t['tip']['ext'], 3), 'handle', np.round(h0 := t['handle']['c'], 3))
        h = t['handle']
        dv = t['tip']['c'][:2] - h['c'][:2]; dv = dv / (np.linalg.norm(dv) + 1e-9)
        axis = ang(dv)
        gp = h['c'].copy(); gp[:2] = gp[:2] - dv * 0.022   # far end of the handle (centre can be blocked)
        zg = max(0.012, h['zmax'] - 0.012)
        tip_meas = lambda sc_, tcp, exp: measure_pink_tip(sc_, exp)
        if manner == 'scoop' or t['cls'] == 'scoop':
            scoop_carry(rb, sc, gp, axis, zg, t['tip']['c'], tip_meas)
        else:
            push_with_tool(rb, sc, manner, gp, axis, zg, t['tip']['c'], TIP_R[manner], tip_meas)
    elif manner == 'hammer':
        hm = find_hammer(sc)
        log('hammer', hm)
        if hm is None: return
        axis = ang(hm['tip'][:2] - hm['grasp'][:2])
        zg = max(0.012, hm['grasp'][2] - 0.012)
        tip_meas = lambda sc_, tcp, exp: measure_beige_tip(sc_, tcp, exp)
        push_with_tool(rb, sc, 'hammer', hm['grasp'], axis, zg, hm['tip'], TIP_R['hammer'], tip_meas)
    elif manner == 'ramp':
        ramp_ball(rb, sc)
    else:
        log('unknown manner; trying any teal tool push')
        run_manner(rb, sc, 'ball', None)
