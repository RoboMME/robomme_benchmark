"""RouteStick autonomous controller (RAX, demo&interaction, single rehearsal budget).

Pipeline (each step logs; nothing raises out of main):
  S1 perceive : observe -> read conditioning-video manifest -> parse route (disk visit sequence +
                pillar side per move) from front RGB frames (video.parse_route, validated offline
                on 3 demos / 22 moves).
  S2 localize : detect 5 disks in live front image, back-project disks, table and pillar pixels
                (robomme_pixel_to_position; fallback own depth back-projection).
  S3 plan     : half-ellipse arcs around the pillar between consecutive disks, on the parsed side.
  S4 execute  : approach first disk from above (calibrate tip contact height via stall / red disk),
                then for each move: lift slightly, follow arc at low height, dip onto next disk,
                verify red bullseye (front window or wrist) with a small search fallback.
  S5 finish   : hold still, report task_complete / episode_ended.
Transitions: S1 fail -> log and return (nothing safe to do); S2 fail -> fallbacks inside; S4 step
failures -> re-observe and continue; any terminal reply -> stop calling tools immediately.
"""
VERSION = 'routestick-v1.1'

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mcpio  # noqa: E402
from geometry import rot_error_world, backproject  # noqa: E402
from vision import red_mask  # noqa: E402
from video import detect_disks, parse_route, red_counts  # noqa: E402
import planner  # noqa: E402

POS_SCALE = 0.05
ROT_SCALE = 0.5
WALL_BUDGET_S = 1080.0          # controller hard limit is 1200 s
FAST_V = 0.015                  # m/step free-space
ARC_V = 0.009                   # m/step along arcs
DIP_V = 0.003                   # m/step final contact approach
TRAVEL_ABOVE_TABLE_M = 0.007    # tip height above table during arcs (min)
TRAVEL_ABOVE_DISK_M = 0.004     # tip height above disk top during arcs (min)
CONTACT_PUSH_M = 0.004          # target below disk top when dipping
SAFE_ABOVE_PILLAR_M = 0.05

OUT = Path('/workspace/output')


class Logger:
    def __init__(self):
        self.t0 = time.time()
        self.f = None
        try:
            OUT.mkdir(parents=True, exist_ok=True)
            self.f = open(OUT / 'routestick_log.txt', 'a')
        except Exception:
            self.f = None

    def __call__(self, *a):
        s = f"[{time.time() - self.t0:7.1f}s] " + ' '.join(str(x) for x in a)
        try:
            print(s, flush=True)
        except Exception:
            pass
        if self.f:
            try:
                self.f.write(s + '\n'); self.f.flush()
            except Exception:
                pass

    def elapsed(self):
        return time.time() - self.t0


def _short(obj, n=1500):
    try:
        s = json.dumps(obj, default=str)
    except Exception:
        s = str(obj)
    return s[:n]


def _strip_big(meta):
    """Copy of meta without long numeric lists, for logging the reply structure."""
    if isinstance(meta, dict):
        return {k: _strip_big(v) for k, v in meta.items()}
    if isinstance(meta, list):
        if len(meta) > 12:
            return [_strip_big(meta[0]), f'...({len(meta)} items)']
        return [_strip_big(v) for v in meta]
    if isinstance(meta, str) and len(meta) > 200:
        return meta[:200] + '...'
    return meta


class Robot:
    def __init__(self, ctx, log):
        self.ctx, self.log = ctx, log
        self.state = None
        self.step_id = 0
        self.ended = False
        self.complete = False
        self.calls = 0
        self.steps_sent = 0
        self.max_steps = None
        self.remaining = None
        self.images = []
        self.meta = {}
        self.front_ref = None
        self.r_ref = None
        self.rot_enabled = True
        self.rot_hist = []
        self.limits_logged = False
        self.call_times = []

    # ---------- parsing ----------
    def _absorb(self, result, what):
        meta = mcpio.meta_of(result)
        self.meta = meta
        st = mcpio.find_state(meta)
        if st is not None:
            self.state = st
        sid = mcpio.find_key(meta, 'step_id', int)
        if isinstance(sid, int):
            self.step_id = sid
        if mcpio.episode_ended(meta):
            if not self.ended:
                self.log(f"[robot] episode_ended reported after {what}: {_short(_strip_big(meta), 600)}")
            self.ended = True
        if mcpio.task_complete(meta):
            if not self.complete:
                self.log(f"[robot] task_complete reported after {what}")
            self.complete = True
        lim = mcpio.find_key(meta, 'episode_limits')
        if lim is not None:
            self._parse_limits(lim)
        try:
            self.images = mcpio.images_of(result)
        except Exception:
            self.images = []
        return meta

    def _parse_limits(self, lim):
        if not self.limits_logged:
            self.log('[robot] episode_limits:', _short(lim, 400))
            self.limits_logged = True
        if isinstance(lim, dict):
            for k, v in lim.items():
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                kl = k.lower()
                if 'remain' in kl or 'left' in kl:
                    self.remaining = int(v)
                elif 'max' in kl or 'limit' in kl or 'horizon' in kl:
                    self.max_steps = int(v)

    def steps_left(self):
        if self.remaining is not None:
            return self.remaining
        if self.max_steps is not None:
            return self.max_steps - self.step_id
        return 10 ** 6

    # ---------- calls ----------
    def call(self, tool, **kw):
        t = time.time()
        self.calls += 1
        try:
            return self.ctx.call(tool, **kw)
        finally:
            self.call_times.append(time.time() - t)

    def observe(self):
        try:
            res = self.call('robomme_observe')
            return self._absorb(res, 'observe'), res
        except Exception as e:
            self.log('[robot] observe failed:', repr(e)[:300])
            return None, None

    def status(self):
        try:
            res = self.call('robomme_status')
            meta = mcpio.meta_of(res)
            lim = mcpio.find_key(meta, 'episode_limits')
            if lim is not None:
                self._parse_limits(lim)
            if mcpio.episode_ended(meta):
                self.ended = True
            return meta
        except Exception as e:
            self.log('[robot] status failed:', repr(e)[:300])
            return {}

    def pos(self):
        return None if self.state is None else np.array(self.state[:3], float)

    def rot_rows(self, k):
        """Small world-frame rotation correction toward the initial orientation (guarded)."""
        if self.r_ref is None or self.state is None or not self.rot_enabled:
            return np.zeros(3)
        try:
            err = rot_error_world(self.r_ref, self.state[3:6])
        except Exception:
            return np.zeros(3)
        e = float(np.linalg.norm(err))
        self.rot_hist.append(e)
        if len(self.rot_hist) > 6 and e > 0.12 and e > self.rot_hist[-6] + 0.02:
            self.log(f"[robot] rotation error growing ({self.rot_hist[-6]:.3f}->{e:.3f}); disabling correction")
            self.rot_enabled = False
            return np.zeros(3)
        if e < 0.02:
            return np.zeros(3)
        per = 0.5 * err / max(k, 1)
        nn = np.linalg.norm(per)
        if nn > 0.04:
            per = per / nn * 0.04
        return per / ROT_SCALE

    def step(self, deltas, what=''):
        """deltas: list of 3-vectors (metres). Returns True if executed."""
        if self.ended:
            return False
        left = self.steps_left()
        if left <= 0:
            self.log('[robot] no steps left')
            self.ended = True
            return False
        deltas = deltas[:max(1, min(100, left))]
        rr = self.rot_rows(len(deltas))
        rows = []
        for d in deltas:
            d = np.asarray(d, float)
            row = list(np.clip(d / POS_SCALE, -1, 1)) + list(np.clip(rr, -1, 1)) + [0.0]
            rows.append([float(x) for x in row])
        try:
            res = self.call('robomme_step', actions=rows)
            self._absorb(res, f'step:{what}')
            self.steps_sent += len(rows)
            if self.remaining is not None and mcpio.find_key(self.meta, 'episode_limits') is None:
                self.remaining -= len(rows)
            return True
        except Exception as e:
            self.log(f'[robot] step failed ({what}):', repr(e)[:400])
            self.observe()
            return False

    # ---------- images ----------
    def front_image(self):
        """Pick the inline RGB image most similar to the reference front image."""
        rgb = [im for h, im in self.images if not h.startswith('depth') and im.ndim == 3]
        if not rgb:
            return None
        for h, im in self.images:
            hl = h.lower()
            if 'front' in hl and 'wrist' not in hl and im.ndim == 3:
                return im
        if self.front_ref is None or len(rgb) == 1:
            return rgb[0]
        best, bd = None, None
        for im in rgb:
            if im.shape != self.front_ref.shape:
                continue
            d = np.abs(im[140:].astype(int) - self.front_ref[140:].astype(int)).mean()
            if bd is None or d < bd:
                best, bd = im, d
        return best if best is not None else rgb[0]

    def wrist_image(self):
        fr = self.front_image()
        for h, im in self.images:
            hl = h.lower()
            if 'wrist' in hl and 'front' not in hl and im.ndim == 3 and im is not fr:
                return im
        for h, im in self.images:
            if im.ndim == 3 and not h.startswith('depth') and im is not fr:
                return im
        return None


# ------------------------------------------------------------------ perception helpers

def load_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.array(im.convert('RGB'))


def load_depth(path, mpu):
    from PIL import Image
    with Image.open(path) as im:
        a = np.array(im).astype(np.float64)
    return a * (mpu if mpu else 0.001)


def files_of(frame):
    out = {}
    for f in frame.get('files', []) or []:
        if isinstance(f, dict):
            out[(f.get('camera'), f.get('kind'))] = f
    return out


class Localizer:
    """World points for front-camera pixels of the current (step-0) observation."""

    def __init__(self, robot, log, step_id, depth=None, K=None, cam_pos=None, cam_quat=None):
        self.robot, self.log, self.step_id = robot, log, step_id
        self.depth, self.K, self.cam_pos, self.cam_quat = depth, K, cam_pos, cam_quat
        self.tool_ok = True
        self.convention = 'opengl'
        self.validated = False

    def own(self, u, v):
        if self.depth is None or self.K is None or self.cam_pos is None or self.cam_quat is None:
            return None
        H, W = self.depth.shape[:2]
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < W and 0 <= vi < H):
            return None
        win = self.depth[max(0, vi - 1):vi + 2, max(0, ui - 1):ui + 2]
        win = win[win > 0]
        if win.size == 0:
            return None
        d = float(np.median(win))
        return backproject(u, v, d, self.K, self.cam_pos, self.cam_quat, self.convention)

    def tool(self, pix, radius=1):
        out = []
        for i in range(0, len(pix), 32):
            chunk = pix[i:i + 32]
            sel = [{'camera': 'front', 'u': float(u), 'v': float(v)} for u, v in chunk]
            try:
                res = self.robot.call('robomme_pixel_to_position', observation_step_id=int(self.step_id),
                                      selections=sel, depth_window_radius=int(radius))
                meta = mcpio.meta_of(res)
                pts = mcpio.parse_points(meta, len(chunk))
                if pts is None:
                    self.log('[loc] could not parse pixel_to_position reply:', _short(_strip_big(meta), 800))
                    return None
                out.extend([p for p, _ in pts])
            except Exception as e:
                self.log('[loc] pixel_to_position failed:', repr(e)[:300])
                return None
        return out

    def points(self, pix, radius=1):
        pix = [(float(u), float(v)) for u, v in pix]
        res = None
        if self.tool_ok:
            res = self.tool(pix, radius)
            if res is None:
                self.tool_ok = False
        if res is not None:
            if not self.validated:
                self._validate(pix, res)
            return res
        return [self.own(u, v) for u, v in pix]

    def _validate(self, pix, res):
        """Cross-check own back-projection; pick the convention that matches the tool."""
        self.validated = True
        if self.depth is None or self.K is None or self.cam_pos is None:
            self.log('[loc] own back-projection unavailable (no depth/calibration parsed)')
            return
        best = None
        for conv in ('opengl', 'opencv'):
            self.convention = conv
            errs = []
            for (u, v), p in zip(pix, res):
                q = self.own(u, v)
                if p is not None and q is not None:
                    errs.append(np.linalg.norm(p - q))
            if errs:
                e = float(np.median(errs))
                if best is None or e < best[0]:
                    best = (e, conv)
        if best:
            self.convention = best[1]
            self.log(f'[loc] own back-projection convention={best[1]} median err vs tool={best[0]:.4f} m')


def parse_calibration(meta, log):
    """Find front camera intrinsics and pose in observation meta (flexible)."""
    K = cam_pos = cam_quat = None
    try:
        intr = mcpio.find_key(meta, 'camera_intrinsics')
        poses = mcpio.find_key(meta, 'camera_poses')
        fi = intr.get('front') if isinstance(intr, dict) else None
        if isinstance(fi, dict):
            for k in ('K', 'matrix', 'intrinsic_matrix', 'k'):
                if isinstance(fi.get(k), list):
                    K = np.array(fi[k], float).reshape(3, 3)
            if K is None and all(k in fi for k in ('fx', 'fy', 'cx', 'cy')):
                K = np.array([[fi['fx'], 0, fi['cx']], [0, fi['fy'], fi['cy']], [0, 0, 1]], float)
        elif isinstance(fi, list):
            K = np.array(fi, float).reshape(3, 3)
        fp = poses.get('front') if isinstance(poses, dict) else None
        if isinstance(fp, dict):
            for k in ('position_m', 'position', 'p', 'translation', 'xyz'):
                if mcpio.is_numlist(fp.get(k), 3):
                    cam_pos = np.array(fp[k], float)
            for k in ('quaternion_wxyz', 'quat_wxyz', 'q', 'quaternion'):
                if mcpio.is_numlist(fp.get(k), 4):
                    cam_quat = np.array(fp[k], float)
    except Exception as e:
        log('[cal] parse failed:', repr(e)[:200])
    log(f'[cal] K={None if K is None else K.round(2).tolist()} cam_pos={cam_pos} quat={cam_quat}')
    return K, cam_pos, cam_quat


# ------------------------------------------------------------------ main pipeline

class Controller:
    def __init__(self, ctx):
        self.log = Logger()
        self.robot = Robot(ctx, self.log)
        self.phase_t = {}

    def phase(self, name):
        self.phase_t[name] = (self.log.elapsed(), self.robot.step_id)
        self.log(f'===== PHASE {name} step={self.robot.step_id} calls={self.robot.calls}')

    def time_ok(self, need=30):
        return self.log.elapsed() + need < WALL_BUDGET_S

    # ---------------- S1 perceive
    def perceive(self):
        self.phase('perceive')
        meta, res = self.robot.observe()
        if meta is None:
            meta, res = self.robot.observe()
        if meta is None:
            raise RuntimeError('cannot observe')
        self.obs0_meta = meta
        self.log('[obs0] structure:', _short(_strip_big(meta), 2500))
        self.log('[obs0] images:', [(h[:60], getattr(im, 'shape', None)) for h, im in self.robot.images])
        self.log('[obs0] state:', self.robot.state, 'step_id:', self.robot.step_id)
        st = self.robot.status()
        self.log('[status]', _short(_strip_big(st), 1500))
        self.instruction = mcpio.find_key(meta, 'instruction', str)
        self.log('[obs0] instruction:', self.instruction)
        # manifest
        mpath = None
        fs = mcpio.find_key(meta, 'frame_sequence')
        if isinstance(fs, dict):
            mpath = fs.get('manifest_path')
        if mpath is None:
            mpath = mcpio.find_key(meta, 'manifest_path', str)
        self.log('[obs0] manifest_path:', mpath)
        manifest = None
        mdir = None
        rp = mcpio.resolve_path(mpath) if mpath else None
        if rp is not None:
            try:
                manifest = json.loads(rp.read_text())
                mdir = rp.parent
            except Exception as e:
                self.log('[obs0] manifest read failed:', repr(e)[:200])
        video_fronts, exec_front, exec_depth = [], None, None
        video_states = []
        if manifest is not None:
            frames = manifest.get('frames', []) if isinstance(manifest, dict) else []
            self.log(f'[manifest] {len(frames)} frames; keys={list(manifest.keys())[:12]}; '
                     f'video_sampling={_short(manifest.get("video_sampling"), 300)}')
            frames = sorted(frames, key=lambda f: (f.get('frame_index', 0)))
            vids = [f for f in frames if f.get('phase') == 'video_demo']
            execs = [f for f in frames if f.get('phase') == 'execution']
            self.log(f'[manifest] video frames={len(vids)} exec frames={len(execs)}')
            for f in vids:
                fl = files_of(f)
                e = fl.get(('front', 'rgb'))
                p = mcpio.resolve_path(e.get('path'), mdir) if e else None
                if p is None:
                    continue
                try:
                    video_fronts.append(load_rgb(p))
                    video_states.append(f.get('state'))
                except Exception as ex:
                    self.log('[manifest] bad frame', e.get('path'), repr(ex)[:100])
            if execs:
                fl = files_of(execs[-1])
                e = fl.get(('front', 'rgb'))
                p = mcpio.resolve_path(e.get('path'), mdir) if e else None
                if p is not None:
                    exec_front = load_rgb(p)
                e = fl.get(('front', 'depth'))
                p = mcpio.resolve_path(e.get('path'), mdir) if e else None
                if p is not None:
                    try:
                        exec_depth = load_depth(p, e.get('meters_per_unit', 0.001))
                    except Exception as ex:
                        self.log('[manifest] depth load failed', repr(ex)[:100])
        self.log(f'[perceive] loaded {len(video_fronts)} video fronts; exec_front={exec_front is not None} '
                 f'exec_depth={exec_depth is not None}')
        if exec_front is None:
            self.robot.front_ref = None
            exec_front = self.robot.front_image()
            self.log('[perceive] exec front from inline image:', None if exec_front is None else exec_front.shape)
        self.front_live = exec_front
        self.robot.front_ref = exec_front
        self.exec_depth = exec_depth
        self.video_fronts = video_fronts
        self.video_states = video_states
        try:
            from PIL import Image
            if exec_front is not None:
                Image.fromarray(exec_front).save(OUT / 'front_live0.png')
            if video_fronts:
                Image.fromarray(video_fronts[0]).save(OUT / 'video_front0.png')
                Image.fromarray(video_fronts[-1]).save(OUT / 'video_front_last.png')
        except Exception:
            pass

    def parse_video(self):
        self.phase('parse_video')
        Dv = Dl = None
        if self.video_fronts:
            Dv, cv = detect_disks(self.video_fronts[0])
            self.log('[video] disks in video frame0:', None if Dv is None else Dv.round(1).tolist(), 'cands', len(cv))
        if self.front_live is not None:
            Dl, cl = detect_disks(self.front_live)
            self.log('[video] disks in live frame:', None if Dl is None else Dl.round(1).tolist(), 'cands', len(cl))
        if Dv is None:
            Dv = Dl
        if Dl is None:
            Dl = Dv
        self.D_live = Dl
        self.route = None
        if Dv is not None and len(self.video_fronts) >= 5:
            r = parse_route(self.video_fronts, Dv, front0=self.video_fronts[0], log=self.log)
            self.log('[video] sequence:', r['sequence'])
            self.log('[video] moves:', r['moves'])
            self.route = r
        if self.route is None or len(self.route['sequence']) < 2:
            self.log('[video] route parse failed or too short:', None if self.route is None else self.route['sequence'])

    # ---------------- S2 localize
    def localize(self):
        self.phase('localize')
        K, cp, cq = parse_calibration(self.obs0_meta, self.log)
        self.loc = Localizer(self.robot, self.log, self.robot.step_id, self.exec_depth, K, cp, cq)
        D = self.D_live
        n = len(D)
        # disks: centre + 4 neighbours
        pix = []
        for u, v in D:
            pix += [(u, v), (u - 2, v), (u + 2, v), (u, v - 1), (u, v + 1)]
        pts = self.loc.points(pix, radius=1)
        disks = []
        for k in range(n):
            ps = [p for p in pts[5 * k:5 * k + 5] if p is not None]
            disks.append(np.median(np.array(ps), 0) if ps else None)
        self.log('[loc] disk points:', [None if d is None else d.round(4).tolist() for d in disks])
        # table points below the line (camera side) between and beside disks
        m, d = D.mean(0), None
        dd = D[-1] - D[0]
        dd = dd / (np.linalg.norm(dd) + 1e-9)
        nrm = np.array([-dd[1], dd[0]])
        if nrm[1] < 0:
            nrm = -nrm
        tpix = []
        for i in range(n - 1):
            mid = (D[i] + D[i + 1]) / 2
            tpix += [tuple(mid + 16 * nrm), tuple(mid + 24 * nrm)]
        tp = self.loc.points(tpix, radius=1)
        tps = np.array([p for p in tp if p is not None]) if tp else np.zeros((0, 3))
        if len(tps):
            self.table_z = float(np.median(tps[:, 2]))
        else:
            dz = [x[2] for x in disks if x is not None]
            self.table_z = float(np.median(dz)) - 0.005 if dz else 0.0
        self.log(f'[loc] table_z={self.table_z:.4f} from {len(tps)} pts')
        # fill missing disks by interpolation in world along the row
        good = [k for k in range(n) if disks[k] is not None and abs(disks[k][2] - self.table_z) < 0.04]
        if len(good) < 2:
            raise RuntimeError('fewer than 2 disks localized')
        for k in range(n):
            if k not in good:
                # linear fit in index
                ks = np.array(good, float)
                X = np.array([disks[g] for g in good])
                coef = np.polyfit(ks, X, 1)
                disks[k] = coef[0] * k + coef[1]
                self.log(f'[loc] disk {k} interpolated -> {disks[k].round(4).tolist()}')
        self.disks = np.array(disks)
        self.disk_top = self.disks[:, 2].copy()
        self.disk_h = float(np.clip(np.median(self.disk_top - self.table_z), 0.0, 0.03))
        spac = np.linalg.norm(np.diff(self.disks[:, :2], axis=0), axis=1)
        self.log(f'[loc] disk_h={self.disk_h:.4f} spacing={spac.round(3).tolist()}')
        # front direction (world, horizontal)
        fp = tps.mean(0) if len(tps) else None
        if fp is None:
            # fallback: camera position if parsed
            fp = cp if cp is not None else self.disks.mean(0) + np.array([0.3, 0, 0])
        self.front_dirs = []
        for i in range(n - 1):
            self.front_dirs.append(planner.front_direction(self.disks[i], self.disks[i + 1], fp))
        self.log('[loc] front dirs:', [f.round(3).tolist() for f in self.front_dirs])
        # pillars
        self.pillars = []
        for i in range(n - 1):
            mid = (D[i] + D[i + 1]) / 2
            gap = np.linalg.norm(D[i + 1] - D[i])
            us = np.linspace(-0.3, 0.3, 7) * gap
            vs = np.arange(-44, 6, 4)
            pix = []
            for vv in vs:
                for uu in us:
                    pix.append((float(np.clip(mid[0] + uu, 0, 255)), float(np.clip(mid[1] + vv, 0, 255))))
            pp = self.loc.points(pix, radius=0)
            P = np.array([p for p in pp if p is not None]) if pp else np.zeros((0, 3))
            pm = planner.pillar_model(P, self.disks[i], self.disks[i + 1], self.front_dirs[i], self.table_z, self.log)
            self.pillars.append(pm)
            self.log(f'[loc] pillar {i}: ' + _short({k: (round(v, 4) if isinstance(v, float) else v) for k, v in pm.items()}, 400))
        tops = [p['top_z'] for p in self.pillars if p.get('top_z') is not None]
        self.pillar_top = max(tops) if tops else self.table_z + 0.20
        self.log(f'[loc] pillar_top={self.pillar_top:.3f}')

    # ---------------- motion primitives
    def goto(self, target, vmax, tol=0.003, K=4, max_calls=40, what='goto'):
        target = np.asarray(target, float)
        for _ in range(max_calls):
            if self.robot.ended or not self.time_ok(10):
                return False
            p = self.robot.pos()
            if p is None:
                self.robot.observe(); continue
            err = target - p
            dist = np.linalg.norm(err)
            if dist < tol:
                return True
            rows, pp = [], p.copy()
            for _ in range(K):
                e = target - pp
                nn = np.linalg.norm(e)
                if nn < 1e-4:
                    break
                dlt = e if nn <= vmax else e / nn * vmax
                rows.append(dlt); pp = pp + dlt
            if not rows:
                return True
            self.robot.step(rows, what)
        p = self.robot.pos()
        return p is not None and np.linalg.norm(target - p) < 3 * tol

    def follow(self, pts, vmax, K=4, tol=0.004, what='follow'):
        pts = np.asarray(pts, float)
        idx = 0
        last = len(pts) - 1
        max_dev = 0.0
        ncalls = 0
        limit = int(3 * len(pts) / max(1, K)) + 20
        while ncalls < limit:
            if self.robot.ended or not self.time_ok(10):
                return False, max_dev
            p = self.robot.pos()
            if p is None:
                self.robot.observe(); ncalls += 1; continue
            # re-anchor: nearest path point between previous anchor and planned index (never skip ahead)
            lo = max(0, idx - 2 * K - 2)
            seg = pts[lo:idx + 1]
            j = int(np.argmin(np.linalg.norm(seg - p, axis=1)))
            dev = float(np.min(np.linalg.norm(seg - p, axis=1)))
            max_dev = max(max_dev, dev)
            idx = lo + j
            if idx >= last and np.linalg.norm(pts[-1] - p) < tol:
                return True, max_dev
            rows, pp = [], p.copy()
            for _ in range(K):
                while idx < last and np.linalg.norm(pts[idx] - pp) < 0.6 * vmax:
                    idx += 1
                e = pts[idx] - pp
                nn = np.linalg.norm(e)
                if nn < 1e-4:
                    break
                dlt = e if nn <= vmax else e / nn * vmax
                rows.append(dlt); pp = pp + dlt
            if not rows:
                if idx >= last:
                    return True, max_dev
                ncalls += 1
                continue
            self.robot.step(rows, what)
            ncalls += 1
        return False, max_dev

    def red_now(self, k, base_wrist):
        fr = self.robot.front_image()
        c = 0
        if fr is not None and self.front_live is not None and fr.shape == self.front_live.shape:
            try:
                c = red_counts(fr, self.red0_live, self.D_live[k:k + 1])[0]
            except Exception:
                c = 0
        w = 0
        wr = self.robot.wrist_image()
        if wr is not None:
            try:
                w = int(red_mask(wr).sum())
            except Exception:
                w = 0
        wrist_hit = base_wrist is not None and w > base_wrist + 60
        return (c >= 12) or wrist_hit, c, w

    def wrist_red(self):
        wr = self.robot.wrist_image()
        if wr is None:
            return None
        try:
            return int(red_mask(wr).sum())
        except Exception:
            return None

    def dip(self, k, xy, z_floor, dv=DIP_V, max_steps=None, what='dip'):
        """Descend at xy until red (disk k), confirmed stall (3 low-motion steps) or floor.

        Stall threshold is relative to the measured free-motion tracking ratio self.rho.
        Returns (result, tcp_z)."""
        base_red, c0, _ = self.red_now(k, None)
        base_wrist = self.wrist_red()
        p = self.robot.pos()
        if p is None:
            self.robot.observe(); p = self.robot.pos()
        if p is None:
            return 'nostate', None
        if max_steps is None:
            max_steps = int(max(0.0, p[2] - z_floor) / (dv * max(self.rho, 0.3))) + 8
        low = 0
        for i in range(max_steps):
            if self.robot.ended:
                return 'ended', None
            p = self.robot.pos()
            if p is None:
                self.robot.observe(); continue
            if p[2] <= z_floor + 0.0004:
                return 'floor', p[2]
            dz = -min(dv, p[2] - z_floor)
            exy = np.asarray(xy[:2], float) - p[:2]
            nn = np.linalg.norm(exy)
            if nn > 0.003:
                exy = exy / nn * 0.003
            self.robot.step([np.array([exy[0], exy[1], dz])], f'{what}{i}')
            q = self.robot.pos()
            if q is None:
                continue
            moved = p[2] - q[2]
            red, c, w = self.red_now(k, base_wrist)
            if red and not base_red:
                self.log(f'[dip] disk {k} red (front {c}, wrist {w}) at z={q[2]:.4f}')
                return 'red', q[2]
            if moved < 0.3 * self.rho * (-dz):
                low += 1
                if low >= 3:
                    self.log(f'[dip] stall at z={q[2]:.4f} (front red {c}, wrist {w}, base_red={base_red})')
                    return 'stall', q[2]
            else:
                low = 0
        p = self.robot.pos()
        return 'floor', None if p is None else p[2]

    def touch(self, k, first=False):
        """Bring tip into contact with disk k; verify (red bullseye, or stall at disk-top height)."""
        disk = self.disks[k]
        off = self.tip_offset
        z_floor = min(disk[2] - CONTACT_PUSH_M, self.table_z - 0.006) + off
        res, z = self.dip(k, disk, z_floor, what=f'dip{k}_')
        self.log(f'[touch] disk {k} result={res} z={z}')
        if res in ('red', 'ended'):
            self.red_seen += 1
            return True
        if res in ('stall', 'floor') and z is not None:
            # dwell briefly in case registration needs contact time
            self.robot.step([np.zeros(3)] * 2, 'dwell')
            red, c, w = self.red_now(k, None)
            if red and not self.red_before.get(k, False):
                self.red_seen += 1
                return True
            if res == 'stall' and abs(z - (disk[2] + off)) <= 0.003:
                self.log(f'[touch] disk {k}: stall at disk-top height (dz={z - disk[2] - off:+.4f}); accept')
                return True
        if self.robot.ended:
            return True
        if self.red_seen == 0 and res == 'stall':
            self.log(f'[touch] disk {k}: no red ever seen; accept stall without search')
            return True
        u = self.disks[min(k + 1, len(self.disks) - 1)] - self.disks[max(k - 1, 0)]
        u = u[:2] / (np.linalg.norm(u[:2]) + 1e-9)
        nrm = np.array([-u[1], u[0]])
        for j, o in enumerate([0.006 * nrm, -0.006 * nrm]):
            if self.robot.ended or not self.time_ok(40) or self.robot.steps_left() < 40:
                break
            p = self.robot.pos()
            self.goto([p[0], p[1], disk[2] + off + 0.006], FAST_V, tol=0.002, K=2, max_calls=4, what='lift_s')
            tgt = np.array([disk[0] + o[0], disk[1] + o[1], disk[2] + off + 0.006])
            self.goto(tgt, 0.006, tol=0.002, K=2, max_calls=5, what='search_xy')
            res, z = self.dip(k, tgt, z_floor, what=f'sdip{k}_{j}_')
            self.log(f'[touch] search {j} disk {k} result={res} z={z}')
            if res in ('red', 'ended'):
                self.red_seen += 1
                return True
            if res == 'stall' and z is not None and abs(z - (disk[2] + off)) <= 0.003:
                return True
        return False

    # ---------------- S4 execute
    def execute(self):
        self.phase('execute')
        seq = self.route['sequence']
        moves = self.route['moves']
        self.red0_live = red_mask(self.front_live) if self.front_live is not None else None
        p0 = self.robot.pos()
        if p0 is None:
            self.robot.observe(); p0 = self.robot.pos()
        if p0 is None:
            raise RuntimeError('no state')
        self.robot.r_ref = np.array(self.robot.state[3:6], float)
        self.log(f'[exec] start tcp={p0.round(4).tolist()} rot={self.robot.r_ref.round(3).tolist()} steps_left={self.robot.steps_left()}')
        self.tip_offset = 0.0
        self.rho = 0.8
        self.rho_samples = []
        self.red_seen = 0
        self.red_before = {}
        first = seq[0]
        disk = self.disks[first]
        z_safe = max(self.pillar_top + SAFE_ABOVE_PILLAR_M, self.table_z + 0.12)
        # 1) rise if needed, then translate above the first disk at safe height
        z_cruise = max(p0[2], z_safe)
        if p0[2] < z_safe - 0.005:
            self.goto([p0[0], p0[1], z_safe], FAST_V, tol=0.005, what='rise')
        p = self.robot.pos()
        t_before = p.copy()
        self.goto([disk[0], disk[1], z_cruise], FAST_V * 1.5, tol=0.005, K=5, what='to_first')
        p = self.robot.pos()
        dist_cmd = np.linalg.norm(np.array([disk[0], disk[1], z_cruise]) - t_before)
        self.log(f'[exec] above first disk {first}: tcp={p.round(4).tolist()} (moved {np.linalg.norm(p - t_before):.3f} of {dist_cmd:.3f})')
        # 2) descend fast to 3 cm above disk top, watching for an early stall (tip offset)
        self.phase('first_descent')
        z_pre = disk[2] + 0.03
        stall = 0
        early_contact = None
        for i in range(60):
            if self.robot.ended or not self.time_ok(60):
                break
            p = self.robot.pos()
            if p[2] <= z_pre + 0.002:
                break
            dz = -min(FAST_V, p[2] - z_pre)
            nrows = int(min(4, max(1, np.floor((p[2] - z_pre) / FAST_V))))
            exy = disk[:2] - p[:2]
            rows = [np.array([exy[0] / nrows if j == 0 else 0, exy[1] / nrows if j == 0 else 0, dz]) for j in range(nrows)]
            self.robot.step(rows, 'descend')
            q = self.robot.pos()
            moved = p[2] - q[2]
            if p[2] > disk[2] + 0.06:
                self.rho_samples.append(moved / (-dz * nrows))
            if moved < 0.15 * (-dz) * nrows and p[2] < self.pillar_top + 0.30:
                stall += 1
                if stall >= 2:
                    early_contact = q[2]
                    self.log(f'[exec] early stall during descent at tcp z={q[2]:.4f}')
                    break
            else:
                stall = 0
        if self.rho_samples:
            self.rho = float(np.clip(np.median(self.rho_samples[-6:]), 0.25, 1.0))
        self.log(f'[exec] tracking ratio rho={self.rho:.3f} from {len(self.rho_samples)} samples')
        if early_contact is not None:
            self.tip_offset = float(early_contact - disk[2])
            self.log(f'[exec] tip offset from early stall = {self.tip_offset:.4f}')
        # 3) slow dip to contact
        self.phase('first_touch')
        z_floor = min(disk[2] - 0.008, self.table_z - 0.006) + self.tip_offset
        res, z = self.dip(first, disk, z_floor, dv=DIP_V, what='first')
        self.log(f'[exec] first touch result={res} tcp z={z}')
        need_touch = res != 'red'
        if res == 'red':
            self.red_seen += 1
        if res in ('red', 'stall') and z is not None:
            off = float(z - disk[2])
            if early_contact is not None:
                self.tip_offset = float(z - disk[2])
            elif abs(off) <= 0.0035:
                self.tip_offset = 0.0          # TCP is the tip; contact at disk top
            elif self.disk_h >= 0.004 and abs(z - self.table_z) <= 0.0025:
                self.tip_offset = 0.0          # landed on the table beside the disk: search
                self.log('[exec] first stall at table height: missed disk; will search')
            elif abs(off) < 0.25:
                self.tip_offset = off
        self.log(f'[exec] tip_offset={self.tip_offset:.4f}')
        if need_touch and not (res == 'stall' and abs(z - disk[2] - self.tip_offset) <= 0.003):
            ok = self.touch(first, first=True)
            self.log(f'[exec] first touch verify={ok}')
        # 4) moves
        travel_tip = self.table_z + max(TRAVEL_ABOVE_TABLE_M, self.disk_h + TRAVEL_ABOVE_DISK_M)
        z_travel = travel_tip + self.tip_offset
        self.log(f'[exec] travel tcp z={z_travel:.4f} (tip {travel_tip:.4f})')
        for mi, (a, b, side, ev) in enumerate(moves):
            if self.robot.ended:
                break
            if not self.time_ok(45):
                self.log('[exec] wall budget nearly exhausted; stopping moves')
                break
            self.phase(f'move{mi}_{a}to{b}_{side}')
            try:
                self.do_move(mi, a, b, side, z_travel)
            except Exception:
                self.log(f'[exec] move {mi} EXCEPTION:\n' + traceback.format_exc())
                self.robot.observe()

    def do_move(self, mi, a, b, side, z_travel):
        if True:
            pi = min(a, b)
            if abs(a - b) != 1:
                self.log(f'[exec] non-adjacent move {a}->{b}; using pillar {pi}')
            A, B = self.disks[a], self.disks[b]
            pm = self.pillars[pi]
            half = np.linalg.norm((B - A)[:2]) / 2
            blat = planner.arc_lateral(pm, side, half)
            fd = self.front_dirs[pi]
            pts = planner.arc_points(A, B, fd, side, blat)
            pts[:, 2] = z_travel
            # lift at A
            p = self.robot.pos()
            self.goto([p[0], p[1], z_travel], 0.006, tol=0.0015, K=2, max_calls=4, what='lift')
            # steps check
            arc_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
            moves_left = max(1, len(self.route['moves']) - mi)
            budget = (self.robot.steps_left() - 15) / moves_left
            v = ARC_V
            if arc_len / ARC_V > 0.45 * budget:
                v = float(np.clip(arc_len / max(5.0, 0.45 * budget), ARC_V, 0.022))
                self.log(f'[exec] step budget tight ({self.robot.steps_left()} left, {moves_left} moves): v={v:.4f}')
            self.red_before = {}
            if self.front_live is not None:
                fr = self.robot.front_image()
                if fr is not None and fr.shape == self.front_live.shape:
                    rc = red_counts(fr, self.red0_live, self.D_live)
                    self.red_before = {kk: rc[kk] >= 12 for kk in range(len(rc))}
            ok, dev = self.follow(pts, v, K=4, what=f'arc{mi}')
            p = self.robot.pos()
            self.log(f'[exec] move {mi} {a}->{b} {side} b={blat:.3f} follow ok={ok} max_dev={dev:.4f} '
                     f'end tcp={None if p is None else p.round(4).tolist()} step={self.robot.step_id}')
            if self.robot.ended:
                return
            # align above B then dip
            self.goto([B[0], B[1], z_travel], 0.005, tol=0.0015, K=2, max_calls=4, what='alignB')
            okt = self.touch(b)
            self.log(f'[exec] move {mi} touch disk {b} verified={okt} step={self.robot.step_id} '
                     f'complete={self.robot.complete} ended={self.robot.ended}')

    def finish(self):
        self.phase('finish')
        if not self.robot.ended:
            self.robot.step([np.zeros(3)] * 5, 'hold')
        if not self.robot.ended:
            st = self.robot.status()
            self.log('[finish] status:', _short(_strip_big(st), 800))
        self.log(f'[finish] complete={self.robot.complete} ended={self.robot.ended} step={self.robot.step_id} '
                 f'calls={self.robot.calls} steps_sent={self.robot.steps_sent} '
                 f'mean_call_s={np.mean(self.robot.call_times) if self.robot.call_times else 0:.3f}')
        self.log('[finish] phases:', self.phase_t)

    def run(self):
        self.log(f'controller {VERSION} start')
        self.perceive()
        self.parse_video()
        if self.route is None or len(self.route['sequence']) < 2 or self.D_live is None:
            self.log('[run] no usable route; ending without motion')
            return
        self.localize()
        self.execute()


def main(ctx):
    c = None
    try:
        c = Controller(ctx)
        c.run()
    except Exception:
        try:
            (c.log if c else print)('[main] EXCEPTION:\n' + traceback.format_exc())
        except Exception:
            pass
    finally:
        try:
            if c is not None:
                c.finish()
        except Exception:
            try:
                print('[main] finish failed:\n' + traceback.format_exc())
            except Exception:
                pass
