"""VideoRepick controller: identify the object picked in the conditioning video, track it
through the shuffle, pick & place it twice, then press the button."""
import os
import sys
import json
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
from mcp_util import log, elapsed, read_manifest, frame_files, load_rgb, load_depth, find_key, get_state
from robot import Robot, EpisodeOver
from motion import Motion
import geometry as G
import perception as P
from vision import rgb_to_hsv, components, adaptive_smin, mask_for_band

OUT = '/workspace/output'
SUMMARY = {}


def save_summary():
    try:
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, 'summary.json'), 'w') as f:
            json.dump(SUMMARY, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    except Exception:
        pass


# ====================================================================== video
def load_video(meta):
    man, path = read_manifest(meta)
    if not man:
        return None
    frames = sorted(man.get('frames', []), key=lambda f: f.get('frame_index', 0))
    vid = [f for f in frames if f.get('phase') == 'video_demo']
    ex = [f for f in frames if f.get('phase') == 'execution']
    out = {'video': [], 'exec': None}
    for f in vid:
        ff = frame_files(f)
        st = f.get('state')
        out['video'].append({'t': float(f.get('timestamp_s', 0.0)), 'files': ff,
                             'state': get_state({'state': st}) if st is not None else None})
    if ex:
        f = ex[-1]
        st = f.get('state')
        out['exec'] = {'files': frame_files(f),
                       'state': get_state({'state': st}) if st is not None else None}
    log('manifest %s: %d video frames, exec=%s' % (path, len(out['video']), out['exec'] is not None))
    return out


def img(entry, cam, kind='rgb'):
    ff = entry['files']
    if (cam, kind) not in ff:
        return None
    p, mpu = ff[(cam, kind)]
    try:
        return load_rgb(p) if kind == 'rgb' else load_depth(p, mpu)
    except Exception as e:
        log('load fail', p, repr(e))
        return None


# ====================================================================== geometry check
def check_geometry(robot, ex_imgs):
    """Choose camera convention by comparing own back-projection with pixel_to_position."""
    conv = {}
    cal = robot.cal
    sels = []
    for cam in ('front', 'wrist'):
        if cam in cal and ex_imgs.get(cam + '_depth') is not None:
            for u in (40, 100, 160, 220):
                for v in (60, 130, 200):
                    sels.append((cam, u, v))
    if not sels:
        log('geometry: no calibration/depth', list(cal.keys()))
        return conv
    pts = robot.pixel_to_world(sels, radius=0)
    for cam in ('front', 'wrist'):
        samples = []
        for (c, u, v), pw in zip(sels, pts):
            if c != cam or pw is None:
                continue
            d = float(ex_imgs[cam + '_depth'][int(v), int(u)])
            if d > 0:
                samples.append((u, v, d, pw))
        if len(samples) >= 3:
            best, err = G.choose_convention(cal[cam], samples)
            log('geometry %s: conv=%s err=%.4f n=%d' % (cam, best, err, len(samples)))
            if best is not None and err < 0.02:
                conv[cam] = best
    SUMMARY['geometry'] = {k: list(v) for k, v in conv.items()}
    return conv


# ====================================================================== video analysis
def analyse_video(V, robot, conv, ex_imgs):
    vid = V['video']
    n = len(vid)
    res = {'votes': [], 'release_world': None}
    if n < 10:
        return res
    T = [f['t'] for f in vid]
    t0 = time.time()
    fronts = [img(f, 'front') for f in vid]
    wrists = [img(f, 'wrist') for f in vid]
    log('loaded video images %.1fs' % (time.time() - t0))
    have_front = all(x is not None for x in fronts)
    have_wrist = all(x is not None for x in wrists)
    states = [f['state'] for f in vid]
    ev = {}
    if all(s is not None for s in states):
        try:
            ev = P.state_events(states, T)
        except Exception as e:
            log('state_events fail', repr(e))
    log('state events', {k: (np.round(v, 3).tolist() if hasattr(v, 'tolist') else v) for k, v in ev.items()})
    SUMMARY['state_events'] = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in ev.items()}
    if ev.get('release_pos') is not None:
        res['release_world'] = ev['release_pos']
    # ---------------- settle index
    settle = ev.get('settle_idx')
    if settle is None and have_front:
        settle = P.settle_from_images(fronts, T)
        log('settle from images', settle)
    if settle is None:
        settle = 0
    SUMMARY['settle_idx'] = settle
    exec_state = V['exec']['state'] if V.get('exec') else robot.state

    # ---------------- front
    if have_front:
        try:
            hue_f, sc = P.pick_hue([fronts[0], fronts[n // 2], fronts[-1]])
            smin_f = adaptive_smin(fronts[::max(1, n // 10)], hue_f)
            log('front hue', hue_f, 'smin', smin_f, sc[:3])
            res['hue_front'] = hue_f
            res['smin_front'] = smin_f
            masks = [P.object_mask(im, hue_f, smin_f) for im in fronts[settle:]]
            excl = P.static_exclusion(masks)
            init = None
            if ev.get('release_pos') is not None and 'front' in conv:
                pr = G.project(robot.cal['front'], ev['release_pos'], conv['front'])
                if pr is not None:
                    dets, a1 = P.split_components(masks[0] & ~excl)
                    if dets:
                        D = [np.hypot(d['x'] - pr[0][0], d['y'] - pr[0][1]) for d in dets]
                        j = int(np.argmin(D))
                        D2 = sorted(D)[1] if len(D) > 1 else 1e9
                        log('front release proj', np.round(pr[0], 1).tolist(), 'nearest det', round(D[j], 1), 'second', round(D2, 1))
                        if D[j] < 15 and D[j] < 0.6 * D2:
                            init = (dets[j]['x'], dets[j]['y'])
            mv, mconf = P.moved_object(fronts[0], fronts[settle], hue_f, smin_f, excl)
            log('front moved-object', mv, round(mconf, 2))
            if init is None and mv is not None and mconf > 0.2:
                init = mv
            res['front_init'] = init
            if init is not None:
                variants = [('pixel', hue_f, smin_f), ('blob', hue_f, smin_f),
                            ('pixel', (hue_f + 5) % 360, max(50, smin_f - 12)),
                            ('blob', (hue_f + 5) % 360, max(50, smin_f - 12))]
                for kind, h, s in variants:
                    if elapsed() > 420:
                        break
                    info = P.track(fronts, T, h, s, settle, init, log=log, name='front-%s-%d-%d' % (kind, h, s), kind=kind)
                    if info:
                        res['votes'].append({'cam': 'front', 'kind': kind, 'final_px': info['final_xy'],
                                             'lost_total': info['lost_total'], 'lost_end': info['lost_end'],
                                             'init_dist': info['init_dist']})
        except Exception as e:
            log('front analysis failed', repr(e), traceback.format_exc()[-800:])

    # ---------------- wrist hold view (requires states + wrist geometry)
    if have_wrist and 'wrist' in conv and ev.get('hold_state') is not None and exec_state is not None \
            and ev.get('release_pos') is not None:
        try:
            hold_end = ev.get('hold_end_idx', n - 1)
            if hold_end >= n - 2:
                cal_hold = G.transfer_camera(robot.cal['wrist'], exec_state, ev['hold_state'])
                W = wrists[settle:]
                TW = T[settle:]
                hue_w, sc = P.pick_hue([W[0], W[len(W) // 2], W[-1]])
                smin_w = adaptive_smin(W[::max(1, len(W) // 10)], hue_w)
                log('wrist hue', hue_w, 'smin', smin_w)
                pr = G.project(cal_hold, ev['release_pos'], conv['wrist'])
                if pr is not None:
                    log('wrist release proj', np.round(pr[0], 1).tolist())
                    masks = [P.object_mask(im, hue_w, smin_w) for im in W]
                    excl = P.static_exclusion(masks)
                    dets, a1 = P.split_components(masks[0] & ~excl)
                    if dets:
                        D = [np.hypot(d['x'] - pr[0][0], d['y'] - pr[0][1]) for d in dets]
                        j = int(np.argmin(D))
                        D2 = sorted(D)[1] if len(D) > 1 else 1e9
                        log('wrist nearest det', round(D[j], 1), 'second', round(D2, 1))
                        if D[j] < 30 and D[j] < 0.7 * D2:
                            init = (dets[j]['x'], dets[j]['y'])
                            dlast = img(vid[-1], 'wrist', 'depth')
                            for kind, h, s in [('pixel', hue_w, smin_w), ('blob', hue_w, smin_w)]:
                                if elapsed() > 450:
                                    break
                                info = P.track(W, TW, h, s, 0, init, excl=excl, log=log, name='wrist-%s' % kind, kind=kind)
                                if not info:
                                    continue
                                fx, fy = info['final_xy']
                                world = None
                                if dlast is not None:
                                    u, v = int(np.clip(round(fx), 0, 255)), int(np.clip(round(fy), 0, 255))
                                    win = dlast[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
                                    win = win[win > 0]
                                    if win.size:
                                        world = G.backproject(cal_hold, fx, fy, float(np.median(win)), conv['wrist'])
                                res['votes'].append({'cam': 'wrist', 'kind': kind, 'final_px': info['final_xy'],
                                                     'world': world, 'lost_total': info['lost_total'],
                                                     'lost_end': info['lost_end'], 'init_dist': info['init_dist']})
            else:
                log('wrist: robot moved during hold; skipping')
        except Exception as e:
            log('wrist analysis failed', repr(e), traceback.format_exc()[-800:])
    return res


# ====================================================================== exec scene objects
def detect_objects(im, hue, smin=None):
    if im is None or hue is None:
        return [], 0
    if smin is None:
        smin = adaptive_smin([im], hue)
    m = mask_for_band(im, hue, width=25, smin=smin, vmin=40)
    comps = components(m, minarea=3)
    H, W = m.shape
    comps = [c for c in comps if not (c['bbox'][0] <= 0 or c['bbox'][1] <= 0 or c['bbox'][2] >= W or c['bbox'][3] >= H)]
    if not comps:
        return [], smin
    med = float(np.median([c['area'] for c in comps]))
    comps = [c for c in comps if 0.25 * med <= c['area'] <= 6 * med]
    return comps, smin


def exec_objects(robot, ex_imgs, hue_f, hue_w):
    sels, meta = [], []
    for cam, hue in (('wrist', hue_w), ('front', hue_f)):
        dets, smin = detect_objects(ex_imgs.get(cam), hue)
        log('exec %s detections: %d (hue %s smin %s)' % (cam, len(dets), hue, smin))
        for d in dets:
            sels.append((cam, d['x'], d['y']))
            meta.append((cam, d))
    pts = robot.pixel_to_world(sels, radius=1) if sels else []
    objs = []
    for (cam, d), p in zip(meta, pts):
        if p is None:
            continue
        hit = None
        for o in objs:
            if np.linalg.norm(o['xyz'][:2] - p[:2]) < 0.02:
                hit = o
                break
        if hit is None:
            objs.append({'xyz': p.copy(), 'px': {cam: (d['x'], d['y'])}, 'area': {cam: d['area']}})
        else:
            hit['px'].setdefault(cam, (d['x'], d['y']))
            hit['area'].setdefault(cam, d['area'])
            if cam == 'wrist':
                hit['xyz'] = p.copy()
    log('exec objects:', [np.round(o['xyz'], 3).tolist() for o in objs])
    return objs


def table_height(robot, ex_imgs, objs):
    """Median z of table pixels in the wrist overview (fallback front)."""
    for cam in ('wrist', 'front'):
        im = ex_imgs.get(cam)
        if im is None:
            continue
        h, s, v = rgb_to_hsv(im)
        thue = np.median(h[(s > 60)])
        tab = (np.minimum(np.abs(h - thue), 360 - np.abs(h - thue)) < 15) & (s > 60)
        pts_px = [o['px'][cam] for o in objs if cam in o['px']]
        sels = []
        for (x, y) in pts_px[:8]:
            for dx, dy in ((14, 0), (-14, 0), (0, 14), (0, -14)):
                u, vv = int(x + dx), int(y + dy)
                if 0 <= u < 256 and 0 <= vv < 256 and tab[vv, u]:
                    sels.append((cam, u, vv))
        if len(sels) < 4:
            ys, xs = np.nonzero(tab[64:192, 64:192])
            if len(xs):
                k = np.linspace(0, len(xs) - 1, 16).astype(int)
                sels = [(cam, xs[i] + 64, ys[i] + 64) for i in k]
        pts = [p for p in robot.pixel_to_world(sels[:32], radius=1) if p is not None]
        if len(pts) >= 3:
            z = float(np.median([p[2] for p in pts]))
            log('table z (%s) = %.4f from %d' % (cam, z, len(pts)))
            return z
    return None


def choose_target(res, objs, front_exec_dets):
    votes = res.get('votes', [])
    scores = np.zeros(len(objs))
    details = []
    front_good = [v for v in votes if v['cam'] == 'front' and v['lost_total'] < 1.5 and v['lost_end'] < 0.3]
    wrist_good = [v for v in votes if v['cam'] == 'wrist' and v['lost_end'] < 0.3]
    use_front_only = len(front_good) >= 2
    for v in votes:
        j = None
        if v['cam'] == 'front' and v.get('world') is not None and len(objs):
            ds = [np.linalg.norm(o['xyz'][:2] - np.asarray(v['world'])[:2]) for o in objs]
            if min(ds) < 0.03:
                j = int(np.argmin(ds))
            w = 1.0 if (v['lost_total'] < 1.5 and v['lost_end'] < 0.3) else 0.2
            if v['kind'] == 'pixel':
                w *= 1.1
        elif v['cam'] == 'front':
            best = None
            for k, o in enumerate(objs):
                if 'front' in o['px']:
                    d = np.hypot(o['px']['front'][0] - v['final_px'][0], o['px']['front'][1] - v['final_px'][1])
                    if d < 10 and (best is None or d < best[0]):
                        best = (d, k)
            j = best[1] if best else None
            w = 1.0 if (v['lost_total'] < 1.5 and v['lost_end'] < 0.3) else 0.2
            if v['kind'] == 'pixel':
                w *= 1.1
        else:
            if v.get('world') is None:
                continue
            ds = [np.linalg.norm(o['xyz'][:2] - np.asarray(v['world'])[:2]) for o in objs]
            if ds and min(ds) < 0.03:
                j = int(np.argmin(ds))
            w = 0.9 if v['lost_end'] < 0.3 else 0.2
            if use_front_only:
                w *= 0.3
        details.append((v['cam'], v['kind'], j, w))
        if j is not None:
            scores[j] += w
    log('votes', details, 'scores', np.round(scores, 2).tolist())
    SUMMARY['votes'] = details
    if len(objs) and scores.max() > 0:
        return int(np.argmax(scores)), float(scores.max() / max(1e-6, scores.sum()))
    # fallback: release point (objects moved, so weak) or first object
    rw = res.get('release_world')
    if rw is not None and len(objs):
        return int(np.argmin([np.linalg.norm(o['xyz'][:2] - rw[:2]) for o in objs])), 0.0
    return (0 if objs else None), 0.0


# ====================================================================== live perception
def current_view(robot):
    meta, imgs = robot.observe()
    return robot.images_from_reply(meta, imgs)


def wrist_objects(robot, hue, smin=None):
    ims = current_view(robot)
    im = ims.get('wrist')
    dets, _ = detect_objects(im, hue, smin)
    if not dets:
        return [], ims
    pts = robot.pixel_to_world([('wrist', d['x'], d['y']) for d in dets], radius=1)
    out = []
    for d, p in zip(dets, pts):
        if p is not None:
            out.append({'xyz': p, 'px': (d['x'], d['y']), 'area': d['area']})
    return out, ims


def refine(robot, hue, smin, xy, maxd=0.05):
    objs, ims = wrist_objects(robot, hue, smin)
    if not objs:
        return None, objs
    ds = [np.linalg.norm(o['xyz'][:2] - xy[:2]) for o in objs]
    j = int(np.argmin(ds))
    log('refine: nearest %.3f m at %s (n=%d)' % (ds[j], np.round(objs[j]['xyz'], 3).tolist(), len(objs)))
    if ds[j] < maxd:
        return objs[j], objs
    return None, objs


def finger_axis(robot):
    cal = robot.cal.get('wrist')
    if cal is None:
        return np.array([1.0, 0.0])
    u = cal['R'][:2, 0]
    nrm = np.linalg.norm(u)
    return u / nrm if nrm > 1e-6 else np.array([1.0, 0.0])


def choose_yaw(robot, txy, neighbors, half_open=0.045, r_obj=0.015):
    u0 = finger_axis(robot)
    best = None
    for th in (0.0, np.pi / 4, -np.pi / 4, np.pi / 2):
        c, s = np.cos(th), np.sin(th)
        u = np.array([c * u0[0] - s * u0[1], s * u0[0] + c * u0[1]])
        clr = 1.0
        for nb in neighbors:
            for sg in (1, -1):
                f = txy[:2] + sg * half_open * u
                # distance from neighbour to finger segment (from f inward 2 cm)
                for a in (0.0, 0.01, 0.02):
                    q = f - sg * a * u
                    clr = min(clr, np.linalg.norm(nb[:2] - q) - r_obj)
        pen = 0.0 if th == 0.0 else 0.006
        sc = min(clr, 0.03) - pen
        if best is None or sc > best[0]:
            best = (sc, th, clr)
    log('yaw choice th=%.2f clearance=%.3f' % (best[1], best[2]))
    return best[1]


# ====================================================================== manipulation
class Ctl:
    def __init__(self, robot, mo, table_z, hue_w, smin_w, objs):
        self.r, self.m = robot, mo
        self.table_z = table_z
        self.hue_w, self.smin_w = hue_w, smin_w
        self.objs = objs
        self.travel_z = table_z + 0.16
        self.hold_ok = 0.004

    def go_travel(self, xy, grip):
        p = self.m.pos()
        if p[2] < self.travel_z - 0.01:
            self.m.move_to([p[0], p[1], self.travel_z], grip, label='lift', tol=0.01)
        self.m.move_to([xy[0], xy[1], self.travel_z], grip, label='travel', tol=0.006)

    def pick(self, target_xyz, attempts=3):
        """Returns (success, grasp_xyz)."""
        txyz = np.array(target_xyz, float)
        for att in range(attempts):
            if self.r.remaining() < 60:
                return False, txyz
            self.go_travel(txyz, -1)
            # close look from above
            ob, near = refine(self.r, self.hue_w, None, txyz, maxd=0.03 if att == 0 else 0.04)
            if ob is not None:
                txyz = ob['xyz'].copy()
                self.m.move_to([txyz[0], txyz[1], self.table_z + 0.10], -1, label='approach', tol=0.004)
                ob2, near2 = refine(self.r, self.hue_w, None, txyz, maxd=0.015)
                if near2:
                    near = near2
                if ob2 is not None:
                    txyz = ob2['xyz'].copy()
            nbs = [o['xyz'] for o in near if np.linalg.norm(o['xyz'][:2] - txyz[:2]) > 0.012]
            yaw = choose_yaw(self.r, txyz, nbs)
            self.m.set_yaw(yaw)
            h = txyz[2] - self.table_z
            if not (0.005 < h < 0.08):
                h = 0.02
            gz = self.table_z + float(np.clip(0.5 * h, 0.012, 0.03))
            log('pick attempt %d target %s h=%.3f grasp_z=%.3f' % (att, np.round(txyz, 3).tolist(), h, gz))
            self.m.move_to([txyz[0], txyz[1], self.table_z + 0.10], -1, label='align', tol=0.004)
            self.m.gripper(-1, n=6)
            self.m.move_to([txyz[0], txyz[1], gz + 0.03], -1, label='descend1', tol=0.004)
            self.m.move_to([txyz[0], txyz[1], gz], -1, label='descend2', tol=0.003, max_step=0.008)
            f = self.m.gripper(1, n=10)
            log('closed: fingers=%.4f at %s' % (f, np.round(self.m.pos(), 3).tolist()))
            gpos = self.m.pos().copy()
            if f > self.hold_ok:
                self.m.move_to([gpos[0], gpos[1], gz + 0.08], 1, label='lift-held', tol=0.006)
                f2 = self.m.fingers()
                log('after lift fingers=%.4f' % f2)
                if f2 > self.hold_ok:
                    return True, np.array([txyz[0], txyz[1], gz])
            log('grasp missed')
            self.m.gripper(-1, n=6)
            self.m.move_to([gpos[0], gpos[1], self.table_z + 0.10], -1, label='retreat', tol=0.01)
            ob, near = refine(self.r, self.hue_w, None, txyz, maxd=0.04)
            if ob is not None:
                txyz = ob['xyz'].copy()
        return False, txyz

    def place(self, g):
        p = self.m.pos()
        self.m.move_to([g[0], g[1], g[2] + 0.012], 1, label='place-down', tol=0.004, max_step=0.01)
        self.m.gripper(-1, n=8)
        q = self.m.pos()
        self.m.move_to([q[0], q[1], self.table_z + 0.10], -1, label='place-up', tol=0.008)
        self.m.set_yaw(0.0)
        return True


def find_button(im, cam, max_area=900):
    """Square low-saturation base with a darker round cap in its centre."""
    if im is None:
        return None
    from scipy import ndimage
    h, s, v = rgb_to_hsv(im)
    lowsat = (s < 55) & (v > 60)
    lowsat = ndimage.binary_opening(lowsat, iterations=1)
    lab, n = ndimage.label(lowsat)
    best = None
    H, W = lowsat.shape
    for i, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None:
            continue
        ys, xs = sl
        if xs.start <= 0 or ys.start <= 0 or xs.stop >= W or ys.stop >= H:
            continue
        bw, bh = xs.stop - xs.start, ys.stop - ys.start
        reg = lab[sl] == i + 1
        area = float(reg.sum())
        if area < 20 or area > max_area or bw < 5 or bh < 5:
            continue
        asp = min(bw, bh) / max(bw, bh)
        fill = area / float(bw * bh)
        if asp < 0.6 or fill < 0.55:
            continue
        vv = v[sl].astype(float)
        yy, xx = np.mgrid[0:bh, 0:bw]
        rr = np.hypot((xx - (bw - 1) / 2.0) / max(bw, 1), (yy - (bh - 1) / 2.0) / max(bh, 1))
        inner = reg & (rr < 0.2)
        outer = reg & (rr > 0.33)
        if inner.sum() < 2 or outer.sum() < 4:
            continue
        contrast = float(vv[outer].mean() - vv[inner].mean())
        sc = asp + fill + 0.04 * min(contrast, 40)
        if best is None or sc > best[0]:
            best = (sc, (xs.start + xs.stop - 1) / 2.0, (ys.start + ys.stop - 1) / 2.0, area, round(contrast, 1))
    return best


def locate_button(robot, ex_imgs):
    """Must be called while the exec observation is current."""
    for cam in ('wrist', 'front'):
        b = find_button(ex_imgs.get(cam), cam)
        log('button %s: %s' % (cam, b))
        if b is None:
            continue
        pts = robot.pixel_to_world([(cam, b[1], b[2])], radius=1)
        if pts and pts[0] is not None:
            return pts[0]
    return None


def press_button(robot, mo, ctl, bxyz):
    if bxyz is None:
        return False
    log('button world', np.round(bxyz, 3).tolist())
    SUMMARY['button'] = bxyz
    mo.gripper(1, n=6)
    ctl.go_travel(bxyz, 1)
    # refine with current wrist view
    ims = current_view(robot)
    b = find_button(ims.get('wrist'), 'wrist', max_area=20000)
    if b is not None:
        pts = robot.pixel_to_world([('wrist', b[1], b[2])], radius=1)
        if pts and pts[0] is not None and np.linalg.norm(pts[0][:2] - bxyz[:2]) < 0.05:
            bxyz = pts[0]
            log('button refined', np.round(bxyz, 3).tolist())
    top = float(bxyz[2])
    mo.move_to([bxyz[0], bxyz[1], top + 0.06], 1, label='btn-above', tol=0.004)
    mo.move_to([bxyz[0], bxyz[1], top + 0.01], 1, label='btn-touch', tol=0.004, max_step=0.01)
    for depth in (0.015, 0.025, 0.035):
        mo.move_to([bxyz[0], bxyz[1], top - depth], 1, label='btn-press', tol=0.003, max_step=0.006, max_chunks=6)
        mo.hold(1, n=4)
    mo.move_to([bxyz[0], bxyz[1], top + 0.08], 1, label='btn-up', tol=0.01)
    return True


# ====================================================================== main
def run(ctx):
    robot = Robot(ctx)
    meta, imgs = robot.observe()
    st = robot.status()
    log('step_limit', robot.step_limit, 'state', None if robot.state is None else np.round(robot.state, 3).tolist())
    SUMMARY['step_limit'] = robot.step_limit
    ex_step_id = robot.step_id
    V = load_video(meta)
    ex_imgs = {}
    if V and V.get('exec'):
        for cam in ('front', 'wrist'):
            ex_imgs[cam] = img(V['exec'], cam)
            ex_imgs[cam + '_depth'] = img(V['exec'], cam, 'depth')
    blk = robot.images_from_reply({}, imgs)
    # learn which RGB image block is which camera by comparing with the manifest exec images
    try:
        rg = blk.get('_rgb_blocks') or []
        if ex_imgs.get('front') is not None and ex_imgs.get('wrist') is not None and len(rg) >= 2:
            def dif(a, b):
                return float(np.abs(a.astype(int) - b.astype(int)).mean()) if a.shape == b.shape else 1e9
            idx = {}
            for cam in ('front', 'wrist'):
                ds = [dif(x, ex_imgs[cam]) for x in rg]
                idx[cam] = int(np.argmin(ds))
                log('block match', cam, [round(d, 1) for d in ds])
            if idx['front'] != idx['wrist']:
                robot.block_idx = idx
            log('image block idx', robot.block_idx)
    except Exception as e:
        log('block order check failed', repr(e))
    for cam in ('front', 'wrist'):
        if ex_imgs.get(cam) is None and blk.get(cam) is not None:
            ex_imgs[cam] = blk[cam]
        if ex_imgs.get(cam + '_depth') is None and blk.get(cam + '_depth') is not None:
            ex_imgs[cam + '_depth'] = blk[cam + '_depth']
    log('calibration cams', list(robot.cal.keys()))
    conv = {}
    try:
        conv = check_geometry(robot, ex_imgs)
    except Exception as e:
        log('geometry check failed', repr(e), traceback.format_exc()[-600:])
    res = {'votes': []}
    if V:
        try:
            res = analyse_video(V, robot, conv, ex_imgs)
        except Exception as e:
            log('video analysis failed', repr(e), traceback.format_exc()[-1200:])
    SUMMARY['votes_raw'] = [{k: v for k, v in x.items()} for x in res.get('votes', [])]
    save_summary()
    hue_f = res.get('hue_front')
    hue_w = None
    try:
        if ex_imgs.get('wrist') is not None:
            hue_w, _ = P.pick_hue([ex_imgs['wrist']])
        if hue_f is None and ex_imgs.get('front') is not None:
            hue_f, _ = P.pick_hue([ex_imgs['front']])
        if hue_w is None or (hue_f is not None and G is not None and min(abs(hue_w - hue_f), 360 - abs(hue_w - hue_f)) > 25):
            log('wrist hue %s differs from front %s; using front' % (hue_w, hue_f))
            hue_w = hue_f if hue_f is not None else hue_w
    except Exception as e:
        log('hue fail', repr(e))
    log('hues front %s wrist %s' % (hue_f, hue_w))
    robot.step_id = ex_step_id
    objs = exec_objects(robot, ex_imgs, hue_f, hue_w)
    table_z = table_height(robot, ex_imgs, objs)
    if table_z is None:
        zs = [o['xyz'][2] for o in objs]
        table_z = (min(zs) - 0.01) if zs else float(robot.state[2]) - 0.3
        log('table z fallback', table_z)
    SUMMARY['table_z'] = table_z
    bxyz = None
    try:
        bxyz = locate_button(robot, ex_imgs)
    except Exception as e:
        log('button locate failed', repr(e))
    try:
        fv = [v for v in res.get('votes', []) if v['cam'] == 'front']
        if fv:
            pts = robot.pixel_to_world([('front', float(np.clip(v['final_px'][0], 0, 255)),
                                         float(np.clip(v['final_px'][1], 0, 255))) for v in fv], radius=1)
            for v, p in zip(fv, pts):
                v['world'] = p
            log('front vote worlds', [None if p is None else np.round(p, 3).tolist() for p in pts])
    except Exception as e:
        log('front vote world failed', repr(e))
    ti, conf = choose_target(res, objs, None)
    log('target index', ti, 'conf', round(conf, 2), 'xyz', None if ti is None else np.round(objs[ti]['xyz'], 3).tolist())
    SUMMARY['target'] = None if ti is None else objs[ti]['xyz']
    SUMMARY['target_conf'] = conf
    SUMMARY['t_perception'] = elapsed()
    save_summary()

    mo = Motion(robot)
    if robot.state is None:
        log('no state; aborting')
        return
    mo.set_orientation_from_state()
    ctl = Ctl(robot, mo, table_z, hue_w, None, objs)
    picks = 0
    if ti is not None:
        target = objs[ti]['xyz'].copy()
        for k in range(2):
            ok, g = ctl.pick(target)
            log('pick %d success=%s' % (k, ok))
            if not ok:
                break
            ctl.place(g)
            picks += 1
            target = g.copy()
            target[2] = objs[ti]['xyz'][2]
            SUMMARY['picks'] = picks
            save_summary()
    mo.set_yaw(0.0)
    pressed = press_button(robot, mo, ctl, bxyz)
    SUMMARY['pressed'] = pressed
    for _ in range(3):
        if robot.ended:
            break
        mo.hold(1, n=5)
    log('done; steps used', robot.steps_used, 'complete', robot.complete)


def main(ctx):
    try:
        run(ctx)
    except EpisodeOver:
        log('episode over')
    except Exception as e:
        log('controller error', repr(e), traceback.format_exc()[-2000:])
    finally:
        SUMMARY['elapsed'] = elapsed()
        save_summary()
