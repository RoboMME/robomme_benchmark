"""VideoRepick controller v1.

1. Read the conditioning-video manifest (state + RGB-D per frame).
2. Find the pick/place event from the robot state in the video; target = object at place xy.
3. Track all objects through the shuffle in world XY using depth from front+wrist cameras.
4. In execution: pick the tracked object, place it back, twice; then press the button.
"""
import os
import sys
import json
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
from motion import Robot, packet, live_frame
from track import object_points, clusters, major_axis
from motion import rotvec_to_R
from frames import load_manifest
from analyze import analyze

OUT = Path('/workspace/output')
T0 = time.time()
LOGF = None


def log(*a):
    global LOGF
    msg = f'[{time.time() - T0:7.1f}s] ' + ' '.join(str(x) for x in a)
    print(msg, flush=True)
    try:
        if LOGF is None:
            OUT.mkdir(parents=True, exist_ok=True)
            LOGF = open(OUT / 'controller_log.txt', 'a')
        LOGF.write(msg + '\n'); LOGF.flush()
    except Exception:
        pass


def scene_clusters(frame, zhi=0.07):
    xy, z = object_points(frame, zhi=zhi)
    cl = clusters(xy)
    out = []
    for c, n, idx in cl:
        out.append(dict(c=c, n=n, cells=xy[idx], z=z[idx], ztop=float(np.percentile(z[idx], 95))))
    return out


def tall_mask(frame, cl_list):
    """Mark clusters that continue above 0.07 m (pillars / robot)."""
    xy, z = object_points(frame, zlo=0.07, zhi=0.5)
    for c in cl_list:
        if len(xy):
            d = np.linalg.norm(xy - c['c'], axis=1)
            c['tall'] = int((d < 0.03).sum())
        else:
            c['tall'] = 0
    return cl_list


def color_stats(frame, cl_list):
    P = object_points(frame, cams=('front', 'wrist'), want_rgb=True)
    for c in cl_list:
        if len(P) == 0:
            c['sat'] = 255; c['bright'] = 0; continue
        d = np.linalg.norm(P[:, :2] - c['c'], axis=1)
        sel = P[d < 0.025]
        if len(sel) == 0:
            c['sat'] = 255; c['bright'] = 0; continue
        rgb = sel[:, 3:6]
        c['sat'] = float((rgb.max(1) - rgb.min(1)).mean())
        c['bright'] = float(rgb.mean())
    return cl_list


class Ctl:
    def __init__(self, ctx):
        self.r = Robot(ctx, log=log)
        self.ysign = 1.0
        self.axis_checked = False
        self.default_closing = None

    def check_axis_sign(self, axis, grip):
        """Rotate a little toward axis and flip ysign if the error grew."""
        e0 = self.axis_err(axis)
        if abs(e0) < 0.1:
            return
        self.r.step([[0, 0, 0, 0, 0, self.ysign * np.sign(e0) * 0.3, grip]] * 2)
        e1 = self.axis_err(axis)
        log(f'axis sign check e0 {e0:.3f} e1 {e1:.3f}')
        if abs(e1) > abs(e0):
            self.ysign = -self.ysign
            log('flipped yaw sign')
        self.axis_checked = True

    def fresh_frame(self):
        reply = self.r.ctx.call('robomme_observe')
        p = live_frame(reply)
        self.r.last = {k: v for k, v in p.items() if k != '_arrays'}
        return p

    def closing_angle(self):
        R = rotvec_to_R(np.asarray(self.r.state()[3:6]))
        a = R[:, 1]
        return float(np.arctan2(a[1], a[0]))

    def axis_err(self, axis):
        if axis is None:
            return 0.0
        cur = self.closing_angle()
        d = self.default_closing if self.default_closing is not None else cur
        # representative of axis (mod pi) within +-pi/2 of the default closing angle -> bounded total yaw
        t = d + ((axis - d + np.pi / 2) % np.pi - np.pi / 2)
        e = t - cur
        return float((e + np.pi) % (2 * np.pi) - np.pi)

    def tilt_corr(self):
        R = rotvec_to_R(np.asarray(self.r.state()[3:6]))
        zc = R[:, 2]
        w = np.cross(zc, np.array([0.0, 0.0, -1.0]))
        return np.clip(w * 0.8 / 0.5, -0.3, 0.3)

    def move(self, xyz, grip, tol=0.004, max_steps=60, vmax=0.8, axis=None):
        """Closed-loop straight move. vmax in action units (1 = 5 cm/step). axis: target closing-axis angle."""
        xyz = np.asarray(xyz, float)
        used = 0
        while used < max_steps and not self.r.ended:
            s = self.r.state()
            err = xyz - s[:3]
            dist = float(np.linalg.norm(err))
            aerr = self.axis_err(axis)
            if dist < tol and abs(aerr) < 0.05:
                return True
            if dist < tol:
                # rotate in place
                n = max(1, min(6, int(np.ceil(abs(aerr) / 0.15))))
                tc = self.tilt_corr()
                self.r.step([[0, 0, 0, tc[0], tc[1], self.ysign * aerr / n / 0.5, grip]] + [[0, 0, 0, 0, 0, self.ysign * aerr / n / 0.5, grip]] * (n - 1))
                used += n
                continue
            # plan rows covering the remaining distance at <= vmax per step
            per = vmax * 0.05
            n = int(np.ceil(dist / per))
            n = max(1, min(n, 10, max_steps - used))
            rows = []
            rem = err.copy()
            nr = max(1, int(np.ceil(abs(aerr) / 0.15)))
            rz = self.ysign * aerr / max(n, nr) / 0.5
            tc = self.tilt_corr()
            for k in range(n):
                stepv = rem / max(1, (n - k))
                tx, ty = (tc[0], tc[1]) if k == 0 else (0.0, 0.0)
                rows.append([stepv[0] / 0.05, stepv[1] / 0.05, stepv[2] / 0.05, tx, ty, rz, grip])
                rem = rem - stepv
            self.r.step(rows)
            used += n
        return float(np.linalg.norm(xyz - self.r.state()[:3])) < tol * 3

    def grip(self, g, n):
        self.r.step([[0, 0, 0, 0, 0, 0, g]] * n)

    def width(self):
        s = self.r.state()
        return float(s[6] + s[7])

    def locate(self, near_xy, maxd=0.06, exclude=None):
        """Re-localize the object nearest near_xy in a fresh observation."""
        f = self.fresh_frame()
        cl = scene_clusters(f)
        if exclude is not None:
            cl = [c for c in cl if np.linalg.norm(c['c'] - exclude) > 0.03]
        if not cl:
            return None
        d = [np.linalg.norm(c['c'] - near_xy) for c in cl]
        j = int(np.argmin(d))
        if d[j] > maxd:
            log('locate: nearest cluster too far', d[j])
            return None
        return cl[j]

    def pick_place(self, obj_xy, button_xy, attempt_tag='', keep_yaw=False):
        """Pick object near obj_xy, lift, place back at same xy. Returns new xy or None."""
        for attempt in range(3):
            c = self.locate(obj_xy, exclude=button_xy)
            if c is None:
                log('pick_place: object not found near', obj_xy)
                return None
            xy = c['c']; ztop = c['ztop']
            gz = float(np.clip(ztop * 0.45, 0.012, 0.03))
            log(f'{attempt_tag} attempt {attempt}: obj {xy.round(3)} ztop {ztop:.3f} n {c["n"]} grasp z {gz:.3f}')
            hover = max(0.10, ztop + 0.06)
            ang, el = major_axis(c['cells'])
            axis = ang if (el > 1.15 and not (keep_yaw and attempt == 0)) else None
            log(f'major axis {np.degrees(ang):.0f} deg elong {el:.2f} -> axis {axis}')
            self.move([xy[0], xy[1], hover], -1, tol=0.006)
            if axis is not None and not self.axis_checked:
                self.check_axis_sign(axis, -1)
            self.move([xy[0], xy[1], hover], -1, tol=0.006, axis=axis)
            log(f'closing angle {np.degrees(self.closing_angle()):.0f}')
            self.move([xy[0], xy[1], gz], -1, tol=0.004, vmax=0.5, axis=axis)
            self.grip(1, 10)
            w = self.width()
            log(f'closed width {w:.4f}')
            if w < 0.004:
                log('grasp missed; reopening')
                self.grip(-1, 6)
                self.move([xy[0], xy[1], hover], -1, tol=0.01)
                obj_xy = xy
                continue
            self.move([xy[0], xy[1], hover], 1, tol=0.008)
            w2 = self.width()
            log(f'lifted width {w2:.4f}')
            if w2 < 0.004:
                log('object slipped')
                self.grip(-1, 4)
                obj_xy = xy
                continue
            self.move([xy[0], xy[1], gz + 0.008], 1, tol=0.004, vmax=0.5)
            self.grip(-1, 8)
            self.move([xy[0], xy[1], hover], -1, tol=0.01)
            for rec in range(2):
                if self.on_table(xy):
                    break
                log('object not on table after release: recovery', rec)
                self.unstick(xy, gz)
            return xy
        return None

    def on_table(self, xy):
        f = self.fresh_frame()
        cl = scene_clusters(f)
        d = [np.linalg.norm(c['c'] - xy) for c in cl]
        ok = len(d) > 0 and min(d) < 0.04
        return ok

    def unstick(self, xy, gz):
        self.grip(1, 8)
        w = self.width()
        log(f'unstick: closed width {w:.4f}')
        self.move([xy[0], xy[1], gz + 0.008], 1, tol=0.005, vmax=0.4)
        self.grip(-1, 8)
        s = self.r.state()
        for dy in (0.6, -1.2, 0.6):
            self.r.step([[0, 0, 0, 0, 0, dy, -1]] * 2)
        self.move([xy[0], xy[1], gz + 0.002], -1, tol=0.005, vmax=0.3, max_steps=10)
        self.move([xy[0], xy[1], 0.10], -1, tol=0.01, vmax=0.3)

    def press(self, b):
        bx, by = b['c']
        top = b.get('ztop', 0.034)
        log(f'press button at {np.round(b["c"],3)} top {top:.3f}')
        self.move([bx, by, max(0.08, top + 0.05)], 1, tol=0.006)
        for zt in [top - 0.02, top - 0.03, top - 0.04]:
            self.move([bx, by, max(zt, 0.003)], 1, tol=0.004, max_steps=20, vmax=0.4)
            if self.r.ended:
                return True
            self.grip(1, 3)
            if self.r.ended:
                return True
        return self.r.ended


def find_button(frame, clist, exclude_xy):
    cand = []
    for c in clist:
        if exclude_xy is not None and np.linalg.norm(c['c'] - exclude_xy) < 0.02:
            continue
        cand.append(c)
    tall_mask(frame, cand)
    color_stats(frame, cand)
    for c in cand:
        log(f"cluster {c['c'].round(3)} n {c['n']} ztop {c['ztop']:.3f} tall {c['tall']} sat {c['sat']:.0f} bright {c['bright']:.0f} static {c.get('static')}")
    # score: low saturation, bright, static, not tall
    best, bs = None, -1e9
    for c in cand:
        s = -c['sat'] + 0.3 * c['bright'] - 0.5 * c['tall']
        if c.get('static'):
            s += 60
        if s > bs:
            best, bs = c, s
    return best


def main(ctx):
    ctl = Ctl(ctx)
    try:
        reply = ctx.call('robomme_observe')
        p = packet(reply)
        log('instruction:', p.get('instruction'), 'limits', p.get('episode_limits'))
        seq = p.get('frame_sequence') or {}
        mpath = seq.get('manifest_path')
        res = None
        frames = []
        if mpath:
            M = load_manifest(mpath)
            frames = M['frames']
            log('frames', len(frames))
            try:
                res = analyze(frames, log=log)
            except Exception:
                log('analyze failed', traceback.format_exc())
        cur = ctl.fresh_frame()
        ctl.default_closing = ctl.closing_angle()
        log(f'default closing angle {np.degrees(ctl.default_closing):.0f}')
        clist = scene_clusters(cur)
        log('exec clusters', [(c['c'].round(3).tolist(), c['n']) for c in clist])
        # static flags from tracker
        target_xy = None
        if res is not None:
            fin = res['final']
            d = [np.linalg.norm(c['c'] - fin) for c in clist]
            j = int(np.argmin(d))
            log(f'tracked final {np.round(fin,3)} -> exec cluster {clist[j]["c"].round(3)} d={d[j]:.3f}')
            target_xy = clist[j]['c']
            start_c = res.get('start_centers')
            if start_c is not None:
                for c in clist:
                    ds = np.linalg.norm(start_c - c['c'], axis=1).min()
                    c['static'] = bool(ds < 0.01)
        button = find_button(cur, clist, target_xy)
        if button is None:
            log('no button found')
        if target_xy is None:
            # fallback: any non-button object
            others = [c for c in clist if button is None or np.linalg.norm(c['c'] - button['c']) > 0.03]
            target_xy = others[0]['c'] if others else None
        bxy = button['c'] if button is not None else None
        if target_xy is not None:
            xy = ctl.pick_place(target_xy, bxy, 'cycle1')
            if xy is not None and not ctl.r.ended:
                xy2 = ctl.pick_place(xy, bxy, 'cycle2', keep_yaw=True)
                log('cycle2 result', xy2)
        if button is not None and not ctl.r.ended:
            ctl.press(button)
        log('done; ended', ctl.r.ended, 'complete', ctl.r.complete, 'remaining', ctl.r.remaining())
    except Exception:
        log('controller exception', traceback.format_exc())
