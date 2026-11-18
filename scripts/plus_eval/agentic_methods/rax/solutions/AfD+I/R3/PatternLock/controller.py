"""PatternLock autonomous controller (RAX).
1. Read the conditioning video from the reset frame sequence: fit the 5x5 grid, track which
   buttons turn red (front + wrist RGB-D -> world), giving the ordered keypoint pattern in grid indices.
2. Fit the execution grid from the step-0 observation (front, wrist fallback).
3. Drive the stick tip over each keypoint (travel just above button tops), press down and hold
   until the button is seen red, then continue. Stop when the episode ends.
"""
import json, math, os, sys, time, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pl

OUT = Path('/workspace/output')
T0 = time.time()
LOGF = None

def log(*a):
    s = ' '.join(str(x) for x in a)
    line = f"[{time.time()-T0:7.1f}s] {s}"
    print(line, flush=True)
    try:
        LOGF.write(line + '\n'); LOGF.flush()
    except Exception:
        pass

# ------------------------------------------------------------------ perception
def red_zband(g):
    top = g['z']; table = g.get('table')
    if table is None or not (top - 0.04 < table < top - 0.004):
        table = top - 0.016
    return (table + 0.4 * (top - table), top + 0.025)

def extract_pattern(frames, g):
    """Ordered list of nodes that turn red during the video (front + wrist)."""
    vids = [f for f in frames if f.get('phase') == 'video_demo']
    sp = g['spacing']
    zband = red_zband(g)
    seq = []; events = []
    base_front = None
    for fi, fr in enumerate(vids):
        try:
            im = pl.frame_images(fr, ('front', 'wrist'))
        except Exception as e:
            log('frame read fail', fi, e); continue
        counts = {}
        for cam, w in (('front', 1.0 / 6.0), ('wrist', 1.0 / 60.0)):
            rgb = im.get((cam, 'rgb')); dep = im.get((cam, 'depth'))
            if rgb is None or dep is None:
                continue
            rm = pl.red_mask(rgb)
            if cam == 'front':
                if base_front is None:
                    from scipy import ndimage
                    base_front = ndimage.binary_dilation(rm, iterations=2)
                rm = rm & ~base_front
            vs, us = np.nonzero(rm)
            if len(us) == 0:
                continue
            try:
                P = pl.backproject(dep, fr['camera_intrinsics'][cam]['matrix'], fr['camera_poses'][cam], us, vs)
            except Exception:
                continue
            ok = np.isfinite(P).all(1) & (P[:, 2] > zband[0]) & (P[:, 2] < zband[1])
            if ok.sum() == 0:
                continue
            N = g['nodes'].reshape(-1, 2)
            D = np.linalg.norm(P[ok][:, None, :2] - N[None], axis=2)
            k = D.argmin(1); dmin = D.min(1)
            for kk in k[dmin < 0.2 * sp]:
                node = (int(kk) // 5, int(kk) % 5)
                counts[node] = counts.get(node, 0) + w
        lit = [n for n, c in counts.items() if c >= 1.0]
        tip = np.asarray(fr.get('state', [0, 0, 0])[:3], float)
        new = [n for n in lit if n not in seq]
        # several new in one frame: farther from the tip came first
        new.sort(key=lambda n: -np.linalg.norm(g['nodes'][n] - tip[:2]))
        for n in new:
            seq.append(n)
            events.append(dict(frame=fi, ts=fr.get('timestamp_s'), node=n, score=round(counts[n], 2),
                               tip=np.round(tip, 4).tolist()))
    return seq, events

def grid_any(meta_like, imgs, tag):
    """Fit the 5x5 button lattice from one observation: depth-based (front, wrist), then color fallback."""
    res = []
    for method in ('depth', 'color'):
        for cam in ('front', 'wrist'):
            try:
                dep = imgs.get((cam, 'depth')); rgb = imgs.get((cam, 'rgb'))
                K = meta_like['camera_intrinsics'][cam]['matrix']; pose = meta_like['camera_poses'][cam]
                if method == 'depth':
                    g, c = pl.grid_from_depth(dep, K, pose)
                else:
                    g, c = pl.grid_from_view(rgb, dep, K, pose)
                if g is None:
                    log(f"{tag} grid {method}/{cam}: none ({len(c)} cands)"); continue
                log(f"{tag} grid {method}/{cam}: cells={g['ncells']} cands={len(c)} sp={g['spacing']:.4f} "
                    f"ang={g['angle']:.1f} resid={g['resid']:.4f} z={g['z']:.4f}")
                g['src'] = f'{method}/{cam}'
                res.append(g)
                if g['ncells'] >= 23 and 0.03 < g['spacing'] < 0.2 and g['resid'] < 0.006:
                    return g
            except Exception as e:
                log(f'{tag} grid {method}/{cam} fail', repr(e))
    if not res:
        return None
    return max(res, key=lambda g: (g['ncells'], -g['resid']))

# ------------------------------------------------------------------ main
def main(ctx):
    global LOGF
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        LOGF = open(OUT / 'controller_log.txt', 'a')
    except Exception:
        LOGF = None
    try:
        run(ctx)
    except Exception:
        log('FATAL', traceback.format_exc())
    finally:
        log('main exit')

def run(ctx):
    r = pl.Robot(ctx, log=log)
    meta = r.observe()
    log('instruction:', meta.get('instruction'), 'limits:', meta.get('episode_limits'), 'step', meta.get('step_id'))
    if r.ended:
        log('episode already ended'); return
    r.R_ref = pl.rotvec_to_R(r.state[3:6])
    g_exec = grid_any(r.meta, r.imgs, 'exec')
    # ---------------- video
    seq_desc = meta.get('frame_sequence') or {}
    man_rel = seq_desc.get('manifest_path')
    if not man_rel:
        try:
            import glob
            ms = sorted(glob.glob('/workspace/runtime/autonomous_controller/runs/*/frames/sequence_000000/manifest.json'),
                        key=os.path.getmtime)
            if ms:
                man_rel = os.path.relpath(ms[-1], '/workspace'); log('fallback manifest', man_rel)
        except Exception as e:
            log('manifest glob fail', repr(e))
    frames = []
    if man_rel:
        try:
            _, frames = pl.load_sequence(man_rel)
        except Exception as e:
            log('manifest load fail', repr(e))
    nvid = sum(1 for f in frames if f.get('phase') == 'video_demo')
    log('frames', len(frames), 'video frames', nvid)
    vids = [f for f in frames if f.get('phase') == 'video_demo']
    g_vid = None
    if vids:
        try:
            g_vid = grid_any(vids[0], pl.frame_images(vids[0], ('front', 'wrist')), 'video')
        except Exception as e:
            log('video grid fail', repr(e))
    if g_vid is None and g_exec is not None:
        g_vid = g_exec
    if g_exec is None:
        g_exec = g_vid
    if g_vid is None:
        log('NO GRID'); return
    seq, events = extract_pattern(frames, g_vid)
    for e in events:
        log('  video event', e)
    log('PATTERN', seq)
    # drawing height from video states at events
    zs = [e['tip'][2] for e in events if len(e['tip']) == 3]
    z_draw = float(np.median(zs)) if zs else 0.070
    if not (0.0 < z_draw < 0.25):
        log('z_draw suspicious', z_draw, 'grid z', g_exec['z'])
        z_draw = 0.070
    # check video tip near nodes at events
    for e in events:
        dxy = float(np.linalg.norm(np.asarray(e['tip'][:2]) - g_vid['nodes'][e['node']]))
        if dxy > 0.03:
            log('  WARN tip far from lit node at event', e['node'], round(dxy, 4))
    try:
        json.dump(dict(pattern=seq, events=events, z_draw=z_draw, exec_nodes=g_exec['nodes'].tolist(),
                       video_nodes=g_vid['nodes'].tolist()), open(OUT / 'pattern.json', 'w'), indent=1)
    except Exception:
        pass
    if not seq:
        log('EMPTY PATTERN'); return
    # ---------------- stick geometry: execution stick can differ from the video's stick
    delta = np.zeros(3)
    off_e_z = None
    try:
        te, ne = pl.stick_tip(r.imgs[('wrist', 'depth')], r.meta['camera_intrinsics']['wrist']['matrix'],
                              r.meta['camera_poses']['wrist'], r.state[:3])
        tv = None; nv = 0
        if vids:
            imv = pl.frame_images(vids[0], ('wrist',))
            tv, nv = pl.stick_tip(imv[('wrist', 'depth')], vids[0]['camera_intrinsics']['wrist']['matrix'],
                                  vids[0]['camera_poses']['wrist'], vids[0]['state'][:3])
        if te is not None and tv is not None:
            off_e = te - r.state[:3]; off_v = tv - np.asarray(vids[0]['state'][:3], float)
            off_e_z = float(off_e[2])
            d = off_e - off_v
            log(f"stick tip offset exec={np.round(off_e,4).tolist()} ({ne}px) video={np.round(off_v,4).tolist()} ({nv}px) delta={np.round(d,4).tolist()}")
            if abs(d[2]) < 0.10 and np.linalg.norm(d[:2]) < 0.05:
                delta = d
            else:
                log('stick delta out of range; ignoring')
        else:
            log('stick tip not measured', te is None, tv is None)
    except Exception as ex:
        log('stick measure fail', repr(ex))
    if np.linalg.norm(delta[:2]) < 0.002 and abs(delta[2]) < 0.002:
        delta = np.zeros(3)
    z_travel = z_draw + 0.002 - delta[2]
    z_press = z_draw - 0.010 - delta[2]
    # physical floor: keep the measured exec stick tip >= 2.5 cm above the button tops
    z_floor = g_exec['z'] + 0.025 - (off_e_z if off_e_z is not None else 0.001)
    if z_press - 0.008 < z_floor:
        log(f'z_press {z_press:.4f} raised to floor {z_floor + 0.008:.4f}')
        z_press = z_floor + 0.008
        z_travel = max(z_travel, z_press + 0.012)
    sp = g_exec['spacing']
    log(f"z_draw={z_draw:.4f} z_travel={z_travel:.4f} z_press={z_press:.4f} spacing={sp:.4f} delta={np.round(delta,4).tolist()}")
    dxy = delta[:2]
    nodes_t = g_exec['nodes'] - dxy[None, None, :]   # TCP xy that puts the stick tip where the video's tip was
    # ---------------- execution
    first = nodes_t[seq[0]]
    e, n = r.move_to(np.r_[first, z_travel + 0.03], vmax=0.06, tol=0.006, max_steps=60)
    log('approach', round(e, 4), n, 'step', r.meta.get('step_id'))
    e, n = r.move_to(np.r_[first, z_travel], vmax=0.015, tol=0.003, max_steps=20)
    log('descend', round(e, 4), n, 'step', r.meta.get('step_id'))
    for i, node in enumerate(seq):
        if r.ended:
            break
        xy = nodes_t[node]
        node_xy = g_exec['nodes'][node]
        if i > 0:
            e, n = r.move_to(np.r_[xy, z_travel], vmax=0.035, tol=0.0035, max_steps=50)
            log(f'travel to {node}', round(e, 4), n, 'step', r.meta.get('step_id'), 'rem', r.remaining)
        lit = False
        if np.linalg.norm(dxy) > 0.004:
            offsets = [(0.0, 0.0), tuple(0.5 * dxy), tuple(dxy)]   # full, half, no xy compensation
        else:
            offsets = [(0.0, 0.0), (0.004, 0.0), (-0.004, 0.0)]
        for a, off in enumerate(offsets):
            if r.ended or r.remaining <= 2:
                break
            pxy = xy + np.asarray(off)
            if a > 0:
                r.move_to(np.r_[pxy, z_travel], vmax=0.012, tol=0.002, max_steps=8)
            lit, info = press(r, pxy, z_travel, z_press - 0.004 * a, 0.2 * sp, red_zband(g_exec), max_steps=24,
                              look_xy=node_xy)
            if lit or r.ended:
                break
        log(f'node {i} {node} lit={lit} step={r.meta.get("step_id")} rem={r.remaining} ended={r.ended} complete={r.complete}')
        if r.ended:
            break
        # lift a little before travelling so the tip leaves the pressed button cleanly
        r.move_to(np.r_[xy, z_travel], vmax=0.015, tol=0.004, max_steps=3)
    log('DONE ended', r.ended, 'complete', r.complete, 'step', r.meta.get('step_id'))
    if not r.ended:
        # hold still briefly to let any pending evaluation settle
        try:
            r.step([[0, 0, 0] + r.rot_cmd() + [0]] * 3)
        except Exception as ex:
            log('final hold fail', repr(ex))
        log('after hold ended', r.ended, 'complete', r.complete)

def press(robot, xy, z_travel, z_press, radius, zref, max_steps=26, look_xy=None):
    xy = np.asarray(xy, float)
    look_xy = xy if look_xy is None else np.asarray(look_xy, float)
    lit = False; info = {}
    z_cmd = z_travel
    k = -1
    for k in range(max_steps):
        if robot.ended or robot.remaining <= 1:
            break
        z_cmd = max(z_press, z_cmd - 0.004)
        p = robot.state[:3]
        d = np.r_[xy, z_cmd] - p
        dxy = d[:2]; nn = np.linalg.norm(dxy)
        if nn > 0.008:
            dxy = dxy / nn * 0.008
        robot.step([[dxy[0] / 0.05, dxy[1] / 0.05, max(-0.4, min(0.4, d[2] / 0.05))] + robot.rot_cmd() + [0.0]])
        info = pl.red_near(robot.meta, robot.imgs, look_xy, radius, zref=zref)
        if info.get('front', 0) >= 6 or info.get('wrist', 0) >= 60:
            lit = True
            break
    log(f"   press xy={np.round(xy,4).tolist()} lit={lit} k={k} z={robot.state[2]:.4f} red={info}")
    return lit, info
