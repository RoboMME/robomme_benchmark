"""VideoUnmask controller (RAX, interaction-only, R1) — written blind: no live scene was observed before the
single rehearsal, so every perception step self-checks and logs what it saw.

Pipeline / transition table
---------------------------
INIT      : robomme_status + robomme_observe(step 0); log packet structure.      -> CALIB
CALIB     : per camera choose back-projection convention that matches
            robomme_pixel_to_position on 12 pixels (fallback: flatness / tool).   -> VIDEO
VIDEO     : read conditioning-video manifest; per instruction colour track the
            colour blob over front frames; take the last position before it
            vanished (covered).                                                  -> SCENE
SCENE     : front point cloud -> table height, elevated objects; choose the
            object (container) nearest each target cube.                         -> PICK(i)
PICK(i)   : VIEW (wrist above container, refine centre/size/yaw) -> APPROACH ->
            DESCEND -> CLOSE -> check fingers (miss -> retry ladder, max 3) ->
            LIFT -> verify (fingers + task_complete) .                           -> PLACEBACK / HOLD
PLACEBACK : only if more targets follow: lower, open, retreat.                   -> PICK(i+1)
HOLD      : keep the last container lifted, watch task_complete, return.
Any exception inside a phase is logged; the pipeline proceeds to the next rung, never
retries a dispatched motion blindly, and stops as soon as the episode ends.
"""
import json
import math
import traceback

import numpy as np

import vu_geom as G
import vu_io as IO
import vu_percep as PC
from vu_io import log

VERSION = 'vunmask-v1'

# ---------------------------------------------------------------- tunables (untested live: interaction-only R1)
POS_SCALE = 0.05          # m per unit action (documented)
ROT_SCALE = 0.5           # rad per unit action (documented)
V_FREE = 0.6              # action units per step in free space (3 cm/step)
V_NEAR = 0.25             # near contact (1.25 cm/step)
POS_TOL = 0.006
ROT_TOL = 0.04
APPROACH_DZ = 0.09        # above container top before descending
VIEW_DZ = 0.22            # wrist inspection height above container top
LIFT_DZ = 0.18            # lift distance after grasp
CLOSE_STEPS = 10
OPEN_STEPS = 10
MAX_ATTEMPTS = 3
WALL_LIMIT = 1080.0       # s; controller hard limit is 1200 s
GRIP_MAX = 0.08


class EpisodeOver(Exception):
    pass


# =============================================================================== robot wrapper
class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.state = None
        self.step_id = 0
        self.ended = False
        self.complete = False
        self.remaining = None
        self.steps_sent = 0
        self.limits_logged = False
        self.last_meta = {}

    def _absorb(self, meta):
        if IO.any_true(meta, ['episode_ended']):
            self.ended = True
        if IO.any_true(meta, ['task_complete']):
            self.complete = True
        st = IO.find_state(meta)
        if st is not None:
            self.state = st
        sid = IO.find_step_id(meta)
        if sid is not None:
            self.step_id = sid
        lim = IO.find_first(meta, ['episode_limits'])
        if lim is not None:
            if not self.limits_logged:
                log(f"episode_limits: {json.dumps(IO.summarize(lim), default=str)[:400]}")
                self.limits_logged = True
            rem = None
            if isinstance(lim, dict):
                for k, v in lim.items():
                    if 'remain' in str(k).lower() and isinstance(v, (int, float)):
                        rem = int(v)
            if rem is not None:
                self.remaining = rem
        self.last_meta = meta

    def call(self, tool, **kw):
        res = self.ctx.call(tool, **kw)
        meta, imgs = IO.parse_reply(res)
        self._absorb(meta)
        return meta, imgs

    def observe(self):
        meta, imgs = self.call('robomme_observe')
        return meta, imgs

    def step(self, rows):
        if self.ended:
            raise EpisodeOver()
        if IO.elapsed() > WALL_LIMIT:
            raise EpisodeOver('wall')
        clean = []
        for r in rows:
            r = [float(np.clip(x if np.isfinite(x) else 0.0, -1, 1)) for x in r]
            clean.append(r)
        if self.remaining is not None:
            if self.remaining <= 0:
                raise EpisodeOver('steps')
            clean = clean[:max(1, self.remaining)]
        sid0 = self.step_id
        out = None
        for i in range(0, len(clean), 100):
            chunk = clean[i:i + 100]
            had_state = self.state
            self.state = None
            meta, _ = self.call('robomme_step', actions=chunk)
            if self.steps_sent == 0:
                IO.save_json('step_reply_summary.json', IO.summarize(meta))
            self.steps_sent += len(chunk)
            if self.state is None:
                self.state = had_state
                try:
                    self.observe()
                except Exception as e:
                    log(f"observe after step failed: {e!r}")
            out = meta
            if self.ended:
                log(f"episode ended at step {self.step_id} task_complete={self.complete}")
                raise EpisodeOver()
        if self.step_id == sid0:
            self.step_id = sid0 + len(clean)
        return out

    # ---------------------------------------------------------------- state helpers
    def pos(self):
        return np.array(self.state[:3], dtype=float)

    def R(self):
        return G.rotvec_to_R(self.state[3:6])

    def grip_width(self):
        return float(self.state[6] + self.state[7])


# =============================================================================== motion
ROT_FRAME = {'frame': 'world', 'bad': 0}


def move_to(rb, target, Rt, grip, vmax=V_FREE, tol=POS_TOL, max_calls=10, label=''):
    """Closed-loop relative moves until within tol. Returns final error (m)."""
    target = np.asarray(target, dtype=float)
    err = None
    for it in range(max_calls):
        p = rb.pos()
        e = target - p
        er = G.R_to_rotvec(Rt @ rb.R().T) if Rt is not None else np.zeros(3)
        err = float(np.linalg.norm(e))
        ern = float(np.linalg.norm(er))
        if err < tol and ern < ROT_TOL:
            break
        n_pos = err / (vmax * POS_SCALE)
        n_rot = ern / 0.12  # <=0.12 rad per step
        n = int(min(40, max(1, math.ceil(max(n_pos, n_rot)))))
        cmd = er if ROT_FRAME['frame'] == 'world' else rb.R().T @ er
        row = list(e / POS_SCALE / n) + list(cmd / ROT_SCALE / n) + [grip]
        rb.step([row] * n)
        if ern > 0.15 and Rt is not None:
            er2 = float(np.linalg.norm(G.R_to_rotvec(Rt @ rb.R().T)))
            if er2 > 0.9 * ern:
                ROT_FRAME['bad'] += 1
                log(f"rotation did not converge ({ern:.3f}->{er2:.3f}) frame={ROT_FRAME['frame']}")
                if ROT_FRAME['bad'] >= 2:
                    ROT_FRAME['frame'] = 'tool' if ROT_FRAME['frame'] == 'world' else 'world'
                    ROT_FRAME['bad'] = 0
                    log(f"switching rotation command frame to {ROT_FRAME['frame']}")
            else:
                ROT_FRAME['bad'] = 0
    p = rb.pos()
    err = float(np.linalg.norm(target - p))
    log(f"move[{label}] -> err {err*1000:.1f} mm pos {np.round(p,3).tolist()} step {rb.step_id}")
    return err


def hold(rb, n, grip):
    rb.step([[0, 0, 0, 0, 0, 0, grip]] * n)


# =============================================================================== perception wrapper
class Views:
    """Camera images/depth/calibration of one observation, with calibrated models."""

    def __init__(self):
        self.models = {}   # cam -> chosen convention dict(index, gl, half, ray, swap, mode)
        self.tool_ok = True

    # ---------------------------------------------------------------- assemble
    def assemble(self, rb, meta, imgs, manifest=None, mbase=None):
        views = {c: dict(rgb=None, depth=None, K=None, Ts=[], conv=None, step_id=rb.step_id)
                 for c in ('front', 'wrist')}
        unl_rgb, unl_depth = [], []
        desc = None
        # attachment descriptor list (same length as images)
        for lst in _lists_of_dicts(meta):
            if len(lst) == len(imgs) and all(('camera' in d) for d in lst):
                desc = lst
                break
        for i, b in enumerate(imgs):
            im = IO.decode_block(b)
            if im is None:
                continue
            cam = kind = None
            if desc is not None:
                cam = IO._cam_of(desc[i].get('camera', ''))
                kind = IO._kind_of(str(desc[i].get('kind', desc[i].get('type', desc[i].get('modality', '')))))
            cam = cam or IO._cam_of(b['hint'])
            kind = kind or IO._kind_of(b['hint'])
            mpu = 0.001
            if desc is not None:
                try:
                    mpu = float(desc[i].get('meters_per_unit', 0.001) or 0.001)
                except Exception:
                    pass
            d = IO.pil_to_depth(im, mpu)
            if d is not None:
                if cam and views[cam]['depth'] is None:
                    views[cam]['depth'] = d
                else:
                    unl_depth.append(d)
            elif kind == 'depth':
                continue
            else:
                rgb = IO.pil_to_rgb(im)
                if cam and views[cam]['rgb'] is None:
                    views[cam]['rgb'] = rgb
                else:
                    unl_rgb.append(rgb)
        # file references inside the packet
        for fe in IO.collect_file_entries(meta):
            cam, kind = fe['camera'], fe['kind']
            if cam not in views or kind not in ('rgb', 'depth') or views[cam][kind] is not None:
                continue
            im = IO.load_image_file(fe['path'])
            if im is None:
                continue
            if kind == 'depth':
                d = im if isinstance(im, np.ndarray) else IO.pil_to_depth(im, fe['mpu'])
                if d is not None:
                    views[cam]['depth'] = d.astype(float)
            else:
                views[cam]['rgb'] = IO.pil_to_rgb(im) if not isinstance(im, np.ndarray) else im
        # manifest execution frame matching this step
        if manifest is not None:
            frames = [f for f in manifest.get('frames', []) if f.get('phase') != 'video_demo']
            if frames:
                fr = frames[-1]
                if fr.get('step_id', rb.step_id) == rb.step_id:
                    ff = IO.frame_files(fr, mbase)
                    for (cam, kind), (p, mpu) in ff.items():
                        if cam in views and kind in ('rgb', 'depth') and views[cam][kind] is None:
                            im = IO.load_image_file(p)
                            if im is None:
                                continue
                            if kind == 'depth':
                                views[cam]['depth'] = IO.pil_to_depth(im, mpu)
                            else:
                                views[cam]['rgb'] = IO.pil_to_rgb(im)
        for cam in ('front', 'wrist'):
            if views[cam]['rgb'] is None and unl_rgb:
                views[cam]['rgb'] = unl_rgb.pop(0)
            if views[cam]['depth'] is None and unl_depth:
                views[cam]['depth'] = unl_depth.pop(0)
        intr, poses = IO.find_calibration(meta)
        for cam in ('front', 'wrist'):
            views[cam]['K'] = G.parse_K(IO.per_camera(intr, cam))
            Ts, conv = G.parse_pose(IO.per_camera(poses, cam))
            if not Ts:
                conv_all = None
                if isinstance(poses, dict):
                    for ck in ('convention', 'frame_convention'):
                        if isinstance(poses.get(ck), str):
                            conv_all = poses[ck].lower()
                conv = conv_all
            views[cam]['Ts'] = Ts
            views[cam]['conv'] = conv
        return views

    # ---------------------------------------------------------------- calibration
    def calibrate(self, rb, views):
        for cam in ('front', 'wrist'):
            v = views[cam]
            info = dict(mode='none')
            try:
                info = self._calib_cam(rb, cam, v)
            except Exception as e:
                log(f"calib {cam} error {e!r}\n{traceback.format_exc()}")
            self.models[cam] = info
            log(f"calib {cam}: {info}")

    def _calib_cam(self, rb, cam, v):
        d, K, Ts = v['depth'], v['K'], v['Ts']
        H, W = (d.shape if d is not None else (256, 256))
        # choose pixels on smooth depth
        pix = []
        if d is not None:
            for vv in np.linspace(20, H - 21, 4).astype(int):
                for uu in np.linspace(20, W - 21, 4).astype(int):
                    best = None
                    for du in (0, 6, -6, 12):
                        for dv in (0, 5, -5):
                            u2, v2 = int(uu + du + 3 * (vv % 3)), int(vv + dv)
                            if not (2 <= u2 < W - 2 and 2 <= v2 < H - 2):
                                continue
                            win = d[v2 - 2:v2 + 3, u2 - 2:u2 + 3]
                            if not np.isfinite(win).all():
                                continue
                            # locally planar: second differences small (no depth edge)
                            c0 = win[2, 2]
                            sd = max(abs(win[2, 0] + win[2, 4] - 2 * c0), abs(win[0, 2] + win[4, 2] - 2 * c0))
                            if sd < 0.003 and np.ptp(win) < 0.05:
                                best = (u2, v2)
                                break
                        if best:
                            break
                    if best:
                        pix.append(best)
        else:
            pix = [(int(u), int(v_)) for u in (40, 128, 210) for v_ in (60, 150, 230)]
        pix = pix[:16]
        tool_pts = self.tool_points(rb, cam, pix)
        ok = [i for i, p in enumerate(tool_pts) if p is not None]
        info = dict(mode='none', n_tool=len(ok))
        if d is not None and K is not None and Ts:
            cands = G.candidate_models(K, Ts, v['conv'], W)
            if len(ok) >= 4:
                uu = np.array([pix[i][0] for i in ok], float)
                vv_ = np.array([pix[i][1] for i in ok], float)
                dd = np.array([d[pix[i][1], pix[i][0]] for i in ok], float)
                TP = np.array([tool_pts[i] for i in ok])
                best, be = None, 1e9
                for ci, m in enumerate(cands):
                    P = m.backproject(uu, vv_, dd)
                    e = float(np.median(np.linalg.norm(P - TP, axis=1)))
                    if e < be:
                        best, be = ci, e
                m = cands[best]
                info = dict(mode='own' if be < 0.02 else 'tool', index=best, err=round(be, 4), **m.desc())
                if be >= 0.02:
                    log(f"calib {cam}: own back-projection mismatch {be:.3f} m -> tool mode")
            else:
                # no tool: flatness heuristic (front) or declared convention
                if cam == 'front':
                    gl_pref = not (v['conv'] and 'cv' in v['conv'])
                    scores = []
                    for m in cands:
                        if m.swap or m.ray or m.half != 0.5 or m.gl != gl_pref:
                            scores.append(-1.0)
                            continue
                        sc = G.flatness_score(m, d)
                        if m.T[2, 3] < 0:  # camera should be above the table
                            sc -= 0.5
                        scores.append(sc)
                    best = int(np.argmax(scores))
                    info = dict(mode='own_unverified', index=best, flat=round(scores[best], 3), **cands[best].desc())
                else:
                    fm = self.models.get('front', {})
                    if 'index' in fm:
                        # reuse front convention (same adapter) with this camera's direct pose
                        idx = [i for i, m in enumerate(cands) if m.gl == fm['gl'] and m.half == fm['half']
                               and m.ray == fm['ray'] and m.swap == fm['swap']]
                        best = idx[0] if idx else 0
                    else:
                        best = 0
                    info = dict(mode='own_unverified', index=best, **cands[best].desc())
        elif len(ok) >= 2:
            info = dict(mode='tool', n_tool=len(ok))
        return info

    def model(self, cam, v):
        info = self.models.get(cam, {})
        if 'index' not in info or v['K'] is None or not v['Ts']:
            return None
        W = v['depth'].shape[1] if v.get('depth') is not None else (v['rgb'].shape[1] if v.get('rgb') is not None else 256)
        cands = G.candidate_models(v['K'], v['Ts'], v['conv'], W)
        if info['index'] >= len(cands):
            return None
        return cands[info['index']]

    def own_ok(self, cam, v):
        return self.models.get(cam, {}).get('mode') in ('own', 'own_unverified') and v['depth'] is not None \
            and self.model(cam, v) is not None

    # ---------------------------------------------------------------- tool
    def tool_points(self, rb, cam, pix):
        out = [None] * len(pix)
        if not self.tool_ok or not pix:
            return out
        for s in range(0, len(pix), 32):
            sel = [{'camera': cam, 'u': float(u), 'v': float(v)} for u, v in pix[s:s + 32]]
            try:
                meta, _ = rb.call('robomme_pixel_to_position', observation_step_id=int(rb.step_id),
                                  selections=sel, depth_window_radius=0)
            except Exception as e:
                log(f"pixel_to_position failed: {e!r}")
                if s == 0:
                    self.tool_ok = 'failed_once' if self.tool_ok is True else False
                return out
            if not getattr(self, '_p2p_logged', False):
                IO.save_json('pixel_reply_summary.json', IO.summarize(meta))
                self._p2p_logged = True
            pts = _extract_points(meta, len(sel))
            for j, p in enumerate(pts):
                out[s + j] = p
        return out

    def cloud(self, rb, cam, v, roi=None, stride=1, tool_stride=6):
        """World points of a view. roi=(u0,u1,v0,v1). Returns (P, uu, vv)."""
        d = v['depth']
        H, W = (d.shape if d is not None else (256, 256))
        u0, u1, v0, v1 = roi if roi else (0, W, 0, H)
        u0, v0 = max(0, int(u0)), max(0, int(v0))
        u1, v1 = min(W, int(u1)), min(H, int(v1))
        if self.own_ok(cam, v):
            vv, uu = np.mgrid[v0:v1:stride, u0:u1:stride]
            dd = d[vv, uu]
            ok = np.isfinite(dd) & (dd > 0.02)
            P = self.model(cam, v).backproject(uu[ok], vv[ok], dd[ok])
            return P, uu[ok], vv[ok]
        if self.tool_ok:
            vv, uu = np.mgrid[v0:v1:tool_stride, u0:u1:tool_stride]
            pix = list(zip(uu.ravel().tolist(), vv.ravel().tolist()))[:1100]
            pts = self.tool_points(rb, cam, pix)
            keep = [(p, a, b) for p, (a, b) in zip(pts, pix) if p is not None]
            if keep:
                P = np.array([k[0] for k in keep])
                return P, np.array([k[1] for k in keep]), np.array([k[2] for k in keep])
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0)


def _lists_of_dicts(obj, depth=0, out=None):
    if out is None:
        out = []
    if depth > 5:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == 'frame_sequence':
                continue
            _lists_of_dicts(v, depth + 1, out)
    elif isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj):
            out.append(obj)
        for x in obj[:20]:
            _lists_of_dicts(x, depth + 1, out)
    return out


def _extract_points(meta, n):
    """Find n world points in a pixel_to_position reply (list of dicts with a 3-vector)."""
    best = None
    for lst in _lists_of_dicts(meta):
        if len(lst) != n:
            continue
        pts = []
        for d in lst:
            p = None
            if d.get('valid') is False:
                pts.append(None)
                continue
            for k in ('position', 'world_position', 'position_m', 'point', 'world', 'xyz', 'world_point',
                      'position_world', 'point_m'):
                if k in d:
                    try:
                        a = np.asarray(d[k], dtype=float).reshape(-1)
                        if a.size == 3 and np.all(np.isfinite(a)):
                            p = a
                            break
                    except Exception:
                        pass
            if p is None:
                for k, val in d.items():
                    if isinstance(val, list) and len(val) == 3 and all(isinstance(x, (int, float)) for x in val):
                        if any(s in k.lower() for s in ('pos', 'point', 'world', 'xyz')):
                            p = np.asarray(val, float)
                            break
            pts.append(p)
        if any(p is not None for p in pts):
            best = pts
            break
    if best is None:
        # single selection may be returned as a flat dict
        if n == 1:
            for k in ('position', 'world_position', 'point'):
                v = IO.find_first(meta, [k])
                try:
                    a = np.asarray(v, float).reshape(-1)
                    if a.size == 3:
                        return [a]
                except Exception:
                    pass
        return [None] * n
    return best


# =============================================================================== video analysis
def load_video(manifest, mbase):
    frames = manifest.get('frames', []) if manifest else []
    vid = [f for f in frames if f.get('phase') == 'video_demo']
    vid.sort(key=lambda f: (f.get('frame_index', 0), f.get('timestamp_s', 0)))
    exe = [f for f in frames if f.get('phase') != 'video_demo']
    return vid, (exe[-1] if exe else None)


def analyze_video(vid, mbase, colors, exec_rgb):
    """Return ({color: track result}, rgbs, seeds{color: result}) for instruction colours and
    for every palette colour (seeds: all covered cubes, used to split merged containers)."""
    targets = colors if colors else ['any']
    allc = list(dict.fromkeys(targets + PC.SEED_PALETTE))
    N = len(vid)
    seed_every = max(1, int(math.ceil(N / 120.0)))
    rgbs = []
    per = {c: [] for c in allc}
    t0 = IO.elapsed()
    for i, f in enumerate(vid):
        ff = IO.frame_files(f, mbase)
        p = ff.get(('front', 'rgb'))
        im = IO.load_image_file(p[0]) if p else None
        rgb = IO.pil_to_rgb(im) if im is not None and not isinstance(im, np.ndarray) else None
        rgbs.append(rgb)
        if rgb is None:
            continue
        hsv = PC.rgb_to_hsv(rgb)
        for c in allc:
            if c not in targets and (i % seed_every) and i != N - 1:
                continue
            try:
                m = PC.color_mask(rgb, c, hsv=hsv)
                bl = [b for b in PC.blobs(m, min_area=4, max_area=3000) if b['fill'] > 0.25 and b['aspect'] < 4.5]
            except Exception:
                bl = []
            per[c].append((i, bl))
        if IO.elapsed() - t0 > 240:
            log("video analysis time cap reached")
            break
    log(f"video: {len(vid)} frames, {sum(r is not None for r in rgbs)} front rgb loaded, "
        f"{IO.elapsed()-t0:.1f}s, seed_every={seed_every}")
    exec_hsv = PC.rgb_to_hsv(exec_rgb) if exec_rgb is not None else None
    results, seeds = {}, {}
    for c in allc:
        try:
            em = PC.color_mask(exec_rgb, c, hsv=exec_hsv) if exec_rgb is not None else None
            r = _rank_tracks(per[c], em, c, verbose=(c in targets))
        except Exception as e:
            log(f"track {c} failed {e!r}\n{traceback.format_exc()}")
            r = None
        if c in targets:
            results[c] = r
            log(f"video target {c}: {_short(r)}")
        if r is not None and r['vanished'] and c != 'any':
            seeds[c] = r
    log(f"seed colours (covered cubes): {[(c, round(r['u']), round(r['v'])) for c, r in seeds.items()]}")
    return results, rgbs, seeds


def _short(r):
    if not r:
        return None
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k not in ('rs', 'cs')}


def _rank_tracks(per, exec_mask, color, verbose=False):
    if not per:
        return None
    tracks = PC.track_blobs(per)
    N = len(per)
    cands = []
    for t in tracks:
        obs = t['obs']
        areas = np.array([b['area'] for _, b in obs])
        if len(obs) < max(2, int(0.02 * N)) or np.median(areas) < 4:
            continue
        med = float(np.median(areas))
        good = [(fi, b) for fi, b in obs if b['area'] >= 0.5 * med]
        fi, b = good[-1]
        tail = [(f2, b2) for f2, b2 in good[-8:] if math.hypot(b2['u'] - b['u'], b2['v'] - b['v']) < 3]
        u = float(np.mean([b2['u'] for _, b2 in tail]))
        v = float(np.mean([b2['v'] for _, b2 in tail]))
        vanished = True
        if exec_mask is not None:
            vanished = not PC.present_in(exec_mask, b, 0.3)
        cands.append(dict(u=u, v=v, frame=fi, last_fi=t['last_fi'], first_fi=obs[0][0], n=len(obs), area=med,
                          vanished=vanished, rs=b['rs'], cs=b['cs'], fill=float(b['fill'])))
    if not cands:
        return None
    # rank: vanished first, then long tracks (real cubes persist), then the latest-ending
    cands.sort(key=lambda c: (c['vanished'], min(c['n'], 0.25 * N), c['last_fi'], c['area']), reverse=True)
    if verbose:
        for c in cands[:5]:
            log(f"  cand {color}: {_short(c)}")
    best = dict(cands[0])
    best['n_cands'] = len(cands)
    return best


# =============================================================================== container choice
def pick_container(objs, xy):
    best, bd = None, 1e9
    for o in objs:
        dc, dctr = PC.dist_to_obj(o, xy)
        score = dc + 0.3 * dctr
        if score < bd:
            best, bd = o, score
    return best, bd


def obj_str(o):
    if o is None:
        return None
    return dict(center=np.round(o['center'], 3).tolist(), top=round(o['top'], 3), h=round(o['height'], 3),
                ang=round(math.degrees(o['angle']), 1), l1=round(o['l1'], 3), l2=round(o['l2'], 3),
                n=o['n_pts'], ext=round(o['foot_ext'], 3))


def grasp_yaw(R_ref, o, extra=0.0):
    """Yaw (rad, about world z) that puts the finger axis (tool y) on the container's narrow axis."""
    f = R_ref[:, 1].copy()
    f[2] = 0
    if np.linalg.norm(f) < 1e-6:
        f = np.array([0, 1.0, 0])
    f /= np.linalg.norm(f)
    a = o['angle']
    e1 = np.array([math.cos(a), math.sin(a)])
    e2 = np.array([-math.sin(a), math.cos(a)])
    l1, l2 = o['l1'], o['l2']
    if min(l1, l2) < 0.025:  # implausible fit (sparse/partial top): no yaw information
        l1 = l2 = max(l1, l2, 0.05)
        a = 0.0
        e1, e2 = np.array([1.0, 0]), np.array([0, 1.0])
        f_axis_yaw = math.atan2(f[1], f[0])
        a = f_axis_yaw  # align with current finger axis -> zero rotation
        e1 = np.array([math.cos(a), math.sin(a)])
        e2 = np.array([-math.sin(a), math.cos(a)])
    square = max(l1, l2) / max(1e-3, min(l1, l2)) < 1.2
    n = e1 if l1 <= l2 else e2
    phi = math.atan2(f[0] * n[1] - f[1] * n[0], f[0] * n[0] + f[1] * n[1])
    period = math.pi / 2 if square else math.pi
    phi = (phi + period / 2) % period - period / 2
    phi = G.wrap_pi(phi + extra)
    if abs(phi) > math.pi / 2 + 1e-6:
        phi = G.wrap_pi(phi - math.copysign(math.pi, phi))
    return phi, min(l1, l2), square


# =============================================================================== main
class Ctl:
    def __init__(self, ctx):
        self.rb = Robot(ctx)
        self.vw = Views()
        self.summary = dict(version=VERSION, phases=[])
        self.R_ref = None
        self.zt = None

    def phase(self, name, **kw):
        rec = dict(name=name, t=round(IO.elapsed(), 1), step=self.rb.step_id, **kw)
        self.summary['phases'].append(rec)
        log(f"PHASE {name} step={self.rb.step_id} {kw}")

    def fresh_views(self):
        meta, imgs = self.rb.observe()
        man, mp = IO.read_manifest(meta)
        return self.vw.assemble(self.rb, meta, imgs, man, mp.parent if mp else None)

    # ------------------------------------------------------------------ run
    def run(self):
        rb = self.rb
        self.phase('INIT')
        instr = ''
        try:
            smeta, _ = rb.call('robomme_status')
            IO.save_json('status_reply_summary.json', IO.summarize(smeta))
            instr = IO.find_first(smeta, ['instruction', 'language_instruction', 'task_instruction', 'goal', 'language_goal']) or ''
        except Exception as e:
            log(f"robomme_status failed {e!r}")
        meta0, imgs0 = rb.observe()
        IO.save_json('observe0_summary.json', IO.summarize(meta0))
        log(f"observe0 keys: {list(meta0.keys())} n_images={len(imgs0)} hints={[b['hint'][:60] for b in imgs0]}")
        i2 = IO.find_first(meta0, ['instruction', 'language_instruction', 'task_instruction', 'goal', 'language_goal'])
        if isinstance(i2, str) and i2:
            instr = i2
        if not isinstance(instr, str):
            instr = json.dumps(instr)
        log(f"instruction: {instr!r}")
        self.summary['instruction'] = instr
        if rb.state is None:
            # fall back to the state recorded with the manifest's execution frame
            try:
                m_, _ = IO.read_manifest(meta0)
                ex = [f for f in (m_ or {}).get('frames', []) if f.get('phase') != 'video_demo']
                if ex:
                    rb.state = IO.find_state(ex[-1])
            except Exception:
                pass
        if rb.state is None:
            log("no state found in observe0 or manifest -> abort (cannot control without measured pose)")
            return
        log(f"state0: {np.round(rb.state, 4).tolist()}")
        man, mp = IO.read_manifest(meta0)
        mbase = mp.parent if mp else None
        if man is None:
            man, mbase = _fallback_manifest()
        views = self.vw.assemble(rb, meta0, imgs0, man, mbase)
        for cam in ('front', 'wrist'):
            v = views[cam]
            log(f"view {cam}: rgb={None if v['rgb'] is None else v['rgb'].shape} depth="
                f"{None if v['depth'] is None else (v['depth'].shape, float(np.nanmin(v['depth'])) if np.isfinite(v['depth']).any() else None)}"
                f" K={None if v['K'] is None else np.round(v['K'], 2).tolist()} nT={len(v['Ts'])} conv={v['conv']}")
        self.phase('CALIB')
        self.vw.calibrate(rb, views)
        self.summary['calib'] = self.vw.models

        # orientation reference (pointing down)
        R0 = rb.R()
        if R0[:, 2] @ np.array([0, 0, -1.0]) > 0.85:
            self.R_ref = R0
        else:
            y = R0[:, 1].copy()
            y[2] = 0
            y = y / (np.linalg.norm(y) + 1e-9) if np.linalg.norm(y) > 1e-3 else np.array([0, -1.0, 0])
            z = np.array([0, 0, -1.0])
            x = np.cross(y, z)
            self.R_ref = np.stack([x, y, z], axis=1)
            log(f"initial tool z not down ({R0[:,2].round(2).tolist()}); using constructed down orientation")

        # ---------------- VIDEO
        self.phase('VIDEO')
        colors = PC.parse_colors(instr)
        cube_colors = [c for c in colors if c in PC.COLOR_HUES]
        log(f"colours in instruction: {colors} -> targets {cube_colors}")
        vid, exe = load_video(man, mbase)
        front = views['front']
        res, rgbs, seed_tracks = analyze_video(vid, mbase, cube_colors, front['rgb'])
        self._save_debug(rgbs, front['rgb'], res)
        # static-camera check (video front depth vs exec depth)
        same_cam = self._static_check(vid, mbase, front)
        targets = cube_colors if cube_colors else ['any']

        # ---------------- SCENE
        self.phase('SCENE')
        if self.vw.own_ok('front', front):
            Pf, uf, vf = self.vw.cloud(rb, 'front', front)
        else:
            parts = []
            rois = [(r['u'], r['v']) for r in res.values() if r] or [(128, 128)]
            for (u0, v0) in rois:
                P_, _, _ = self.vw.cloud(rb, 'front', front, roi=(u0 - 40, u0 + 40, v0 - 40, v0 + 40), tool_stride=3)
                parts.append(P_)
            Pf = np.concatenate(parts) if parts else np.zeros((0, 3))
        log(f"front cloud {len(Pf)} pts (own={self.vw.own_ok('front', front)})")
        tcp = rb.pos()
        plan = []
        for c in targets:
            r = res.get(c)
            xy = None
            if r is not None:
                xy = self._cube_world(r, vid, mbase, front, same_cam)
            plan.append(dict(color=c, video=_short(r), xy=None if xy is None else np.asarray(xy, float)))
        self.seeds = []
        for sc, r in seed_tracks.items():
            try:
                xy = self._cube_world(r, vid, mbase, front, same_cam, quiet=True)
            except Exception:
                xy = None
            if xy is not None and all(np.hypot(*(np.asarray(xy[:2]) - q[:2])) > 0.02 for q in self.seeds):
                self.seeds.append(np.asarray(xy, float))
        log(f"cube seeds (world): {[np.round(q,3).tolist() for q in self.seeds]}")
        # table height near the targets
        cxy = None
        for p in plan:
            if p['xy'] is not None:
                cxy = p['xy'][:2]
                break
        self.zt = PC.table_height(Pf, cxy) if len(Pf) else None
        log(f"table z = {self.zt}")
        if self.zt is None:
            self.zt = 0.0
        all_objs = PC.find_objects(Pf, self.zt, cxy if cxy is not None else tcp[:2], half=0.45,
                                   exclude=[tcp], excl_r=0.10) if len(Pf) else []
        all_objs = [o for o in all_objs if o['n_pts'] >= 8 and 0.015 < o['height'] < 0.25 and o['foot_ext'] < 0.25]
        for o in all_objs:
            log(f"  object {obj_str(o)}")
        self.summary['objects'] = [obj_str(o) for o in all_objs]
        for p in plan:
            if p['xy'] is not None and all_objs:
                o, sc = pick_container(all_objs, p['xy'])
                o2 = PC.split_by_seeds(o, self.seeds, p['xy'])
                if o2 is not o:
                    log(f"split merged object by cube seeds: {obj_str(o)} -> {obj_str(o2)}")
                o = o2
                p['obj'] = o
                p['score'] = sc
            else:
                p['obj'] = None
            log(f"plan {p['color']}: cube_xy={None if p['xy'] is None else np.round(p['xy'],3).tolist()} "
                f"-> container {obj_str(p['obj'])} score={p.get('score')}")
        if all(p['obj'] is None for p in plan) and all_objs:
            # no cube information: fall back to the object nearest the image centre ray (a guess)
            log("no cube localised: guessing the container nearest the table centre of objects")
            ctr = np.mean([o['center'] for o in all_objs], axis=0)
            o, _ = pick_container(all_objs, ctr)
            plan[0]['obj'] = o
        self.summary['plan'] = [dict(color=p['color'], xy=None if p['xy'] is None else p['xy'].tolist(),
                                     obj=obj_str(p['obj'])) for p in plan]
        IO.save_json('summary.json', self.summary)

        # ---------------- PICK each target
        for i, p in enumerate(plan):
            if p['obj'] is None and p['xy'] is None:
                log(f"target {p['color']}: nothing to pick")
                continue
            last = (i == len(plan) - 1)
            ok = self.pick(p, last)
            self.summary.setdefault('picks', []).append(dict(color=p['color'], ok=ok, complete=rb.complete))
            IO.save_json('summary.json', self.summary)
            if rb.complete:
                break
        # ---------------- HOLD
        self.phase('HOLD', complete=rb.complete)
        for k in range(3):
            if rb.complete or rb.ended:
                break
            hold(rb, 10, 1.0)
        if not rb.complete:
            # a little more height in case the success needs a higher lift
            p = rb.pos()
            move_to(rb, p + np.array([0, 0, 0.06]), None, 1.0, vmax=V_NEAR, label='extra-lift', max_calls=3)
            hold(rb, 10, 1.0)

    # ------------------------------------------------------------------ helpers
    def _static_check(self, vid, mbase, front):
        try:
            if not vid or front['depth'] is None:
                return True
            ff = IO.frame_files(vid[0], mbase)
            p = ff.get(('front', 'depth'))
            if not p:
                return True
            d0 = IO.pil_to_depth(IO.load_image_file(p[0]), p[1])
            if d0 is None or d0.shape != front['depth'].shape:
                return True
            m = np.isfinite(d0) & np.isfinite(front['depth'])
            diff = np.abs(d0[m] - front['depth'][m])
            frac = float(np.mean(diff < 0.01))
            log(f"static camera check: frac(|dDepth|<1cm)={frac:.2f}")
            return frac > 0.5
        except Exception as e:
            log(f"static check error {e!r}")
            return True

    def _cube_world(self, r, vid, mbase, front, same_cam, quiet=False):
        """World xy of the cube at its last visible video position."""
        u, v = r['u'], r['v']
        pts = []
        model = self.vw.model('front', front)
        own = self.vw.own_ok('front', front)
        # (1) video-frame depth on the blob pixels
        if own and same_cam and 0 <= r['frame'] < len(vid):
            ff = IO.frame_files(vid[r['frame']], mbase)
            p = ff.get(('front', 'depth'))
            if p:
                d = IO.pil_to_depth(IO.load_image_file(p[0]), p[1])
                if d is not None:
                    dd = d[r['rs'], r['cs']]
                    ok = np.isfinite(dd)
                    if ok.sum() >= 2:
                        P = model.backproject(r['cs'][ok], r['rs'][ok], dd[ok])
                        P = P[np.isfinite(P).all(axis=1)]
                        if len(P) >= 2:
                            zc = np.percentile(P[:, 2], 60)
                            Ptop = P[P[:, 2] >= zc] if (P[:, 2] >= zc).sum() >= 2 else P
                            c = np.median(Ptop, axis=0)
                            if not quiet:
                                log(f"cube world (video depth): {np.round(c,3).tolist()} n={len(P)}")
                            pts.append(c)
        if pts:
            return pts[0]
        # (2) same pixel in the execution frame (container surface)
        if own:
            d = front['depth']
            ui, vi = int(round(u)), int(round(v))
            win = d[max(0, vi - 1):vi + 2, max(0, ui - 1):ui + 2]
            if np.isfinite(win).any():
                P = model.backproject(np.array([u]), np.array([v]), np.array([np.nanmedian(win)]))[0]
                log(f"cube world (exec depth at pixel): {np.round(P,3).tolist()}")
                return P
        tp = self.vw.tool_points(self.rb, 'front', [(int(round(u)), int(round(v)))])
        if tp and tp[0] is not None:
            log(f"cube world (tool at pixel): {np.round(tp[0],3).tolist()}")
            return tp[0]
        return None

    def _save_debug(self, rgbs, exec_rgb, res):
        try:
            from PIL import Image, ImageDraw
            ims = []
            idx = [i for i, r in enumerate(rgbs) if r is not None]
            pick = []
            if idx:
                pick = [idx[0], idx[len(idx) // 2], idx[-1]]
            for c, r in res.items():
                if r and 0 <= r['frame'] < len(rgbs) and rgbs[r['frame']] is not None:
                    pick.append(r['frame'])
            tiles = [rgbs[i] for i in pick] + ([exec_rgb] if exec_rgb is not None else [])
            if not tiles:
                return
            W = sum(t.shape[1] for t in tiles)
            canvas = Image.new('RGB', (W, tiles[0].shape[0]))
            x = 0
            for t in tiles:
                im = Image.fromarray(t)
                dr = ImageDraw.Draw(im)
                for c, r in res.items():
                    if r:
                        dr.ellipse([r['u'] - 5, r['v'] - 5, r['u'] + 5, r['v'] + 5], outline=(255, 255, 0))
                canvas.paste(im, (x, 0))
                x += t.shape[1]
            canvas.save(IO.OUT / 'debug_video.png')
        except Exception as e:
            log(f"debug image failed {e!r}")

    # ------------------------------------------------------------------ pick one container
    def pick(self, p, last):
        rb = self.rb
        o = p['obj']
        if o is None:
            o = dict(center=np.asarray(p['xy'][:2]), top=float(p['xy'][2]) + 0.03, height=0.06, angle=0.0,
                     l1=0.06, l2=0.06, cells_xy=np.asarray([p['xy'][:2]]), n_pts=0, foot_ext=0.06)
        self.phase('PICK', color=p['color'], obj=obj_str(o))
        extra_yaw = 0.0
        deeper = 0.0
        for attempt in range(MAX_ATTEMPTS):
            if rb.ended or IO.elapsed() > WALL_LIMIT:
                return False
            try:
                o = self.view_refine(o, p)
                res = self.grasp_lift(o, extra_yaw, deeper, attempt)
            except EpisodeOver:
                raise
            except Exception as e:
                log(f"pick attempt {attempt} error {e!r}\n{traceback.format_exc()}")
                res = 'error'
            log(f"attempt {attempt}: {res} complete={rb.complete}")
            if res == 'lifted':
                if not last and not rb.complete:
                    self.place_back(o)
                return True
            # escalation ladder
            if res == 'miss':
                extra_yaw = math.pi / 2 if attempt == 0 else 0.0
                deeper = 0.0 if attempt == 0 else 0.01
            elif res == 'slip':
                deeper += 0.012
            try:
                p0 = rb.pos()
                move_to(rb, [p0[0], p0[1], max(p0[2], o['top'] + APPROACH_DZ)], None, -1.0, label='retreat',
                        max_calls=4)
            except EpisodeOver:
                raise
            except Exception as e:
                log(f"retreat error {e!r}")
        return False

    def view_refine(self, o, p):
        """Move the wrist camera above the container; re-segment; return refined object."""
        rb = self.rb
        c = np.asarray(o['center'], float)
        top = float(o['top'])
        cur = rb.pos()
        safe_z = max(cur[2], top + 0.12)
        if cur[2] < top + 0.10:
            move_to(rb, [cur[0], cur[1], safe_z], self.R_ref, -1.0, label='up')
        move_to(rb, [c[0], c[1], top + VIEW_DZ], self.R_ref, -1.0, vmax=V_FREE, tol=0.01, label='view')
        hold(rb, 3, -1.0)
        try:
            views = self.fresh_views()
            w = views['wrist']
            if w['rgb'] is not None:
                try:
                    from PIL import Image
                    Image.fromarray(w['rgb']).save(IO.OUT / f"wrist_view_{p['color']}_{rb.step_id}.png")
                except Exception:
                    pass
            P, uu, vv = self.vw.cloud(rb, 'wrist', w, tool_stride=12)
            if not self.vw.own_ok('wrist', w) and len(P) >= 20:
                # tool mode: coarse pass found where things are; densify around the object nearest c
                h_ = P[:, 2] - self.zt
                near = (np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) < 0.06) & (h_ > 0.012) & (h_ < (top - self.zt) + 0.035)
                if near.sum() >= 2:
                    u0, u1, v0, v1 = uu[near].min() - 24, uu[near].max() + 24, vv[near].min() - 24, vv[near].max() + 24
                    P2, _, _ = self.vw.cloud(rb, 'wrist', w, roi=(u0, u1, v0, v1), tool_stride=3)
                    if len(P2) > 30:
                        P = np.concatenate([P, P2])
            log(f"wrist cloud {len(P)} pts")
            if len(P) < 50:
                return o
            zt_w = PC.table_height(P, c, radius=0.2)
            zt = self.zt
            if zt_w is not None and abs(zt_w - self.zt) < 0.03:
                zt = zt_w
            objs = PC.find_objects(P, zt, c, half=0.13, cell=0.003, zmax=(top - zt) + 0.035, exclude=None,
                                   dilate=False)
            objs = [q for q in objs if q['n_pts'] >= 20 and q['height'] > 0.012]
            best, bd = None, 1e9
            for q in objs:
                dctr = float(np.hypot(*(q['center'] - c)))
                if dctr < bd and abs(q['top'] - top) < 0.035:
                    best, bd = q, dctr
            log(f"wrist objects: {[obj_str(q) for q in objs][:5]} chosen d={bd:.3f}")
            if best is not None and p.get('xy') is not None:
                b2 = PC.split_by_seeds(best, getattr(self, 'seeds', []), p['xy'])
                if b2 is not best:
                    log(f"wrist: split merged object by seeds -> {obj_str(b2)}")
                    best = b2
                    bd = float(np.hypot(*(best['center'] - c)))
            if best is not None and bd < 0.05:
                best['view_zt'] = zt
                return best
        except EpisodeOver:
            raise
        except Exception as e:
            log(f"wrist refine error {e!r}\n{traceback.format_exc()}")
        return o

    def grasp_lift(self, o, extra_yaw, deeper, attempt):
        rb = self.rb
        c = np.asarray(o['center'], float)
        top = float(o['top'])
        zt = float(o.get('view_zt', self.zt))
        h = max(0.01, top - zt)
        phi, width, square = grasp_yaw(self.R_ref, o, extra_yaw)
        if width > GRIP_MAX - 0.004:
            log(f"WARNING container narrow width {width:.3f} m exceeds gripper opening")
        depth_in = float(np.clip(0.45 * h, 0.012, 0.03)) + deeper
        zg = max(top - depth_in, zt + 0.014)
        Rt = G.Rz(phi) @ self.R_ref
        log(f"grasp plan: c={np.round(c,3).tolist()} top={top:.3f} zt={zt:.3f} h={h:.3f} width={width:.3f} "
            f"square={square} yaw={math.degrees(phi):.1f} zg={zg:.3f} attempt={attempt}")
        self.phase('APPROACH', attempt=attempt)
        move_to(rb, [c[0], c[1], top + APPROACH_DZ], Rt, -1.0, vmax=V_FREE, tol=0.005, max_calls=10, label='approach')
        hold(rb, 2, -1.0)
        self.phase('DESCEND')
        e = move_to(rb, [c[0], c[1], zg], Rt, -1.0, vmax=V_NEAR, tol=0.004, max_calls=8, label='descend')
        pz = rb.pos()[2]
        if pz > zg + 0.02:
            log(f"descend blocked at z={pz:.3f} (target {zg:.3f}) — likely landing on the container")
        self.phase('CLOSE')
        hold(rb, CLOSE_STEPS, 1.0)
        gw = rb.grip_width()
        for _ in range(6):  # wait until the fingers stop moving
            hold(rb, 3, 1.0)
            g2 = rb.grip_width()
            if abs(g2 - gw) < 0.0006:
                gw = g2
                break
            gw = g2
        log(f"grip width after close: {gw:.4f}")
        if gw < 0.006:
            return 'miss'
        self.phase('LIFT', grip=round(gw, 4))
        p0 = rb.pos()
        move_to(rb, [p0[0], p0[1], zg + LIFT_DZ], Rt, 1.0, vmax=0.4, tol=0.01, max_calls=6, label='lift')
        hold(rb, 3, 1.0)
        gw2 = rb.grip_width()
        log(f"grip width after lift: {gw2:.4f} complete={rb.complete}")
        if rb.complete:
            return 'lifted'
        if gw2 < 0.006:
            return 'slip'
        # visual check: is something still standing at the old place at the old height?
        try:
            views = self.fresh_views()
            Pf, _, _ = self.vw.cloud(rb, 'front', views['front'])
            objs = PC.find_objects(Pf, self.zt, c, half=0.06, exclude=[rb.pos()], excl_r=0.12)
            still = [q for q in objs if abs(q['top'] - top) < 0.012 and np.hypot(*(q['center'] - c)) < 0.025
                     and q['n_pts'] > 15]
            # log only: a revealed cube of similar height must not trigger a retry
            log(f"after-lift check (info only): {len(still)} objects at the old place/height")
        except EpisodeOver:
            raise
        except Exception as e:
            log(f"after-lift check error {e!r}")
        return 'lifted'

    def place_back(self, o):
        rb = self.rb
        self.phase('PLACEBACK')
        try:
            p = rb.pos()
            c = np.asarray(o['center'], float)
            top = float(o['top'])
            zt = float(o.get('view_zt', self.zt))
            depth_in = float(np.clip(0.45 * max(0.01, top - zt), 0.012, 0.03))
            zg = max(top - depth_in, zt + 0.014)
            move_to(rb, [c[0], c[1], p[2]], None, 1.0, vmax=0.4, tol=0.008, label='over')
            move_to(rb, [c[0], c[1], zg + 0.006], None, 1.0, vmax=V_NEAR, tol=0.006, label='lower')
            hold(rb, OPEN_STEPS, -1.0)
            p = rb.pos()
            move_to(rb, [p[0], p[1], top + 0.12], None, -1.0, vmax=0.5, tol=0.01, label='retreat')
        except EpisodeOver:
            raise
        except Exception as e:
            log(f"place_back error {e!r}")


def _fallback_manifest():
    try:
        import glob
        cands = sorted(glob.glob(str(IO.WS / 'runtime/autonomous_controller/runs/*/frames/**/*.json'), recursive=True),
                       key=lambda p: -IO.Path(p).stat().st_mtime)
        for c in cands[:50]:
            try:
                m = json.loads(IO.Path(c).read_text())
                if any(f.get('phase') == 'video_demo' for f in m.get('frames', [])):
                    log(f"fallback manifest {c}")
                    return m, IO.Path(c).parent
            except Exception:
                continue
    except Exception as e:
        log(f"fallback manifest error {e!r}")
    return None, None


def main(ctx):
    log(f"{VERSION} start")
    ctl = None
    try:
        ctl = Ctl(ctx)
        ctl.run()
    except EpisodeOver as e:
        log(f"episode over: {e!r}")
    except Exception as e:
        log(f"FATAL {e!r}\n{traceback.format_exc()}")
    finally:
        if ctl is not None:
            ctl.summary['final'] = dict(step=ctl.rb.step_id, complete=ctl.rb.complete, ended=ctl.rb.ended,
                                        t=round(IO.elapsed(), 1), steps_sent=ctl.rb.steps_sent)
            IO.save_json('summary.json', ctl.summary)
            log(f"final {ctl.summary['final']}")
