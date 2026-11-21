"""BinFill controller (RAX, RoboMME-Plus).  VERSION below is logged at start.

Task: "put <counts> <colour> cubes into the bin, then press the button to stop".  Cubes appear over
time; on some layouts they ride a conveyor belt (gray striped mat) and leave at its far end.

Pipeline / transition table (every step returns; main loop never raises):
  INIT      observe, parse instruction, detect bin/button/mat/table geometry (front cam)   -> WAIT
  WAIT      idle a few steps while estimating belt velocity from repeated front detections  -> SELECT
  SELECT    choose a cube of a still-needed colour that can be intercepted                  -> APPROACH | WAIT
  APPROACH  fly to hover above the predicted cube position (feed-forward on belt velocity)  -> REFINE | SELECT(timeout/lost)
  REFINE    wrist-camera localisation (xy, top z, yaw); rotate gripper to cube faces          -> GRASP
  GRASP     descend tracking the cube, close, lift; verify finger opening + wrist colour      -> CARRY | SELECT(miss)
  CARRY     carry over the bin (wrist refine of bin centre), lower, open, lift; count        -> SELECT | PRESS
  PRESS     when all counts done: press the button with the closed gripper until episode ends -> END
Global guards: episode_ended -> END; wall clock > WALL_STOP -> END; step budget checks.
"""
import json
import math
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perception as P  # noqa: E402
import io_utils as IO  # noqa: E402
from geom import rotvec_to_R, quat_wxyz_to_R, wrap_quarter, yaw_of  # noqa: E402

VERSION = 'binfill-v1'

# ---------------- tunables (untested on the live simulator; derived from demos + mock) -------------
POS_SCALE = 0.05          # m per action unit (docs)
ROT_SCALE = 0.5           # rad per action unit (docs)
V_FREE = 0.030            # m per step for free-space moves
V_NEAR = 0.015            # m per step near contact
HOVER_DZ = 0.10           # TCP height above cube top when hovering for wrist refinement
CARRY_DZ = 0.12           # carry clearance above bin rim (plus cube size)
DROP_CLEAR = 0.025        # cube bottom clearance above rim when releasing
DEFAULT_CUBE = 0.04       # fallback cube edge (m)
WALL_STOP = 960.0         # s; controller limit is 1200 s (one RPC may take up to 180 s)
STEP_RESERVE_PRESS = 45   # steps kept for the final button press
OUT_DIR = '/workspace/output'
MAX_SAVED_IMAGES = 60

NUM_WORDS = dict(a=1, an=1, one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, ten=10,
                 single=1, couple=2)
COLOUR_SYNONYM = dict(purple='magenta', violet='magenta', pink='magenta', cyan='cyan', teal='cyan',
                      red='red', green='green', blue='blue', yellow='yellow', orange='orange', magenta='magenta')


def parse_instruction(text):
    import re
    out = {}
    if not text:
        return out
    s = text.lower()
    s = s.split(', then')[0] if ', then' in s else s.split('then press')[0]
    for n, c in re.findall(r'\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|single)\s+'
                           r'(red|green|blue|yellow|orange|magenta|purple|violet|pink|cyan|teal)\s+(?:cubes?|blocks?)', s):
        cnt = int(n) if n.isdigit() else NUM_WORDS.get(n, 1)
        col = COLOUR_SYNONYM.get(c, c)
        out[col] = out.get(col, 0) + cnt
    return out


class EpisodeOver(Exception):
    pass


class Robot:
    """Thin wrapper over the MCP robot tools; tracks state, step id, and termination."""

    def __init__(self, ctx, log):
        self.ctx = ctx
        self.log = log
        self.meta = {}
        self.imgs = {}
        self.state = None
        self.step_id = 0
        self.ended = False
        self.complete = False
        self.limits = None
        self.max_steps = None
        self.track_ratio = 1.0
        self.n_calls = 0
        self.img_notes = ''
        self.first_meta_logged = False
        self.t0 = time.time()

    # ---- reply handling
    def _ingest(self, res, tool):
        meta = IO.parse_meta(res)
        self.meta = meta
        if not self.first_meta_logged or tool == 'robomme_status':
            self.log('meta_keys', tool=tool, keys=sorted(list(meta.keys()))[:60],
                     sample=json.dumps(meta, default=str)[:1500])
            self.first_meta_logged = True
        st = IO.find_first(meta, ('state', 'proprio', 'proprioception', 'robot_state'),
                           lambda v: IO.is_num_list(v, nmin=8))
        if st is not None:
            self.state = np.array(st[:8], float)
        sid = IO.find_first(meta, ('step_id', 'step', 'observation_step_id'),
                            lambda v: isinstance(v, int) and not isinstance(v, bool))
        if sid is not None:
            self.step_id = int(sid)
        for k in ('episode_ended', 'ended', 'done'):
            v = IO.find_first(meta, (k,), lambda v: isinstance(v, bool))
            if v:
                self.ended = True
        v = IO.find_first(meta, ('task_complete', 'success', 'task_success'), lambda v: isinstance(v, bool))
        if v:
            self.complete = True
        lim = IO.find_first(meta, ('episode_limits',))
        if lim is not None:
            self.limits = lim
            if isinstance(lim, dict):
                for k in ('max_steps', 'step_limit', 'max_episode_steps', 'horizon', 'limit', 'steps_limit'):
                    if isinstance(lim.get(k), int):
                        self.max_steps = lim[k]
                        break
        if tool in ('robomme_observe', 'robomme_step'):
            try:
                imgs, notes = IO.extract_images(res, meta, P.classify_camera)
            except Exception as e:  # noqa
                imgs, notes = {}, 'extract-failed %r' % e
            if 'front' not in imgs or 'wrist' not in imgs:
                try:
                    more = IO.images_from_files(meta)
                    for k, v in more.items():
                        imgs.setdefault(k, v)
                    if more:
                        notes += ';files'
                except Exception:
                    pass
            if imgs:
                self.imgs = imgs
                self.img_step = self.step_id
            if notes != self.img_notes:
                self.log('image_labels', notes=notes, cams=sorted(imgs.keys()))
                self.img_notes = notes

    def remaining_steps(self):
        if isinstance(self.limits, dict):
            for k in ('remaining_steps', 'remaining', 'steps_remaining'):
                v = self.limits.get(k)
                if isinstance(v, int):
                    return v
        if self.max_steps:
            return self.max_steps - self.step_id
        return 10 ** 6

    def call(self, tool, **kw):
        self.n_calls += 1
        return self.ctx.call(tool, **kw)

    def observe(self):
        res = self.call('robomme_observe')
        self._ingest(res, 'robomme_observe')
        return res

    def status(self):
        try:
            res = self.call('robomme_status')
            self._ingest(res, 'robomme_status')
        except Exception as e:
            self.log('status_error', err=repr(e)[:300])

    def step(self, rows):
        """Execute rows (list of 7-lists). Never retries a failed motion; re-syncs via observe."""
        if self.ended:
            raise EpisodeOver()
        if time.time() - self.t0 > WALL_STOP:
            self.log('wall_stop_in_step')
            raise EpisodeOver()
        rows = [[float(np.clip(x, -1.0, 1.0)) for x in r] for r in rows][:100]
        rem = self.remaining_steps()
        if rem <= 0:
            self.ended = True
            raise EpisodeOver()
        if len(rows) > rem:
            rows = rows[:max(1, rem)]
        p0 = None if self.state is None else self.state[:3].copy()
        cmd = np.sum([r[:3] for r in rows], 0) * POS_SCALE
        try:
            res = self.call('robomme_step', actions=rows)
            self._ingest(res, 'robomme_step')
        except Exception as e:
            msg = repr(e)
            self.log('step_error', err=msg[:500])
            if 'ended' in msg.lower() or 'terminat' in msg.lower() or 'done' in msg.lower():
                self.ended = True
            try:
                self.observe()
            except Exception as e2:
                self.log('observe_error', err=repr(e2)[:300])
                self.status()
            if self.ended:
                raise EpisodeOver()
            return
        if p0 is not None and self.state is not None:
            nc = float(np.linalg.norm(cmd))
            if nc > 0.02:
                meas = float(np.dot(self.state[:3] - p0, cmd / nc))
                r = float(np.clip(meas / nc, 0.3, 1.3))
                self.track_ratio = 0.7 * self.track_ratio + 0.3 * r
        if self.ended:
            raise EpisodeOver()

    def p2w(self, cam, pix, radius=1):
        """Back-project pixels of the current observation. Returns list of np.array|None."""
        if not pix:
            return []
        sels = [dict(camera=cam, u=float(int(round(u))), v=float(int(round(v)))) for u, v in pix][:32]
        try:
            res = self.call('robomme_pixel_to_position', observation_step_id=int(self.step_id),
                            selections=sels, depth_window_radius=int(radius))
            pts = IO.parse_points(res, sels)
            return [p for p, _ in pts]
        except Exception as e:
            self.log('p2w_error', err=repr(e)[:300])
            return [None] * len(sels)


class Controller:
    def __init__(self, ctx):
        self.t0 = time.time()
        try:
            os.makedirs(OUT_DIR, exist_ok=True)
        except Exception:
            pass
        self.logf = None
        try:
            self.logf = open(os.path.join(OUT_DIR, 'trace.jsonl'), 'a')
        except Exception:
            pass
        self.n_saved = 0
        self.r = Robot(ctx, self.log)
        self.need = {}
        self.done = {}
        self.table_z = None
        self.belt_z = None
        self.bin = None          # dict(c=xy, rim_z)
        self.button = None       # np.array xyz
        self.mat_pts = None
        self.vel = np.zeros(3)   # m per step
        self.vel_conf = 0
        self.vel_samples = []
        self.det_hist = []       # (step, [(colour, xyz)]) from front camera
        self.cube = DEFAULT_CUBE
        self.cube_samples = []
        self.approach_axis = None
        self.orient_ok = True
        self.yaw_total = 0.0
        self.yaw_sign = 1.0
        self.attempts = 0
        self.fail_streak = 0
        self.phase = 'INIT'
        self.phase_t = {}
        self.grip = -1.0
        self.blacklist = []      # (step, xy) recently failed targets

    # ------------------------------------------------------------------ logging
    def log(self, ev, **kw):
        rec = dict(t=round(time.time() - self.t0, 2), ev=ev)
        try:
            rec['step'] = self.r.step_id
        except Exception:
            pass
        rec.update(kw)
        try:
            s = json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
        except Exception:
            s = str(rec)
        print(s[:2000], flush=True)
        if self.logf:
            try:
                self.logf.write(s + '\n')
                self.logf.flush()
            except Exception:
                pass

    def save_img(self, name):
        if self.n_saved >= MAX_SAVED_IMAGES:
            return
        try:
            from PIL import Image
            ims = [self.r.imgs[k] for k in ('front', 'wrist') if k in self.r.imgs]
            if not ims:
                return
            im = np.concatenate(ims, 1)
            Image.fromarray(im).save(os.path.join(OUT_DIR, '%02d_s%04d_%s.png' % (self.n_saved, self.r.step_id, name)))
            self.n_saved += 1
        except Exception:
            pass

    def elapsed(self):
        return time.time() - self.t0

    # ------------------------------------------------------------------ helpers
    @property
    def ee(self):
        return self.r.state[:3].copy()

    def fingers(self):
        return float(self.r.state[6] + self.r.state[7])

    def orient_fix(self):
        """Small world-frame rotation vector keeping the approach axis pointing down."""
        if not self.orient_ok or self.approach_axis is None or self.r.state is None:
            return np.zeros(3)
        try:
            R = rotvec_to_R(self.r.state[3:6])
            idx, sgn = self.approach_axis
            a = R[:, idx] * sgn
            want = np.array([0, 0, -1.0])
            c = np.cross(a, want)
            s = np.linalg.norm(c)
            ang = math.atan2(s, float(np.dot(a, want)))
            if ang < 0.06 or s < 1e-6:
                return np.zeros(3)
            return c / s * min(ang * 0.5, 0.08)
        except Exception:
            return np.zeros(3)

    def tilt(self):
        try:
            R = rotvec_to_R(self.r.state[3:6])
            idx, sgn = self.approach_axis
            return math.degrees(math.acos(np.clip(-R[2, idx] * sgn, -1, 1)))
        except Exception:
            return -1

    def rows_for(self, disp_list, grip, yaw_list=None, fix=True):
        """Build action rows from per-step world displacements (m) and yaw increments (rad)."""
        rows = []
        k = 1.0 / max(0.5, min(1.2, self.r.track_ratio))
        of = self.orient_fix() if fix else np.zeros(3)
        for i, d in enumerate(disp_list):
            d = np.asarray(d, float) * k
            yaw = yaw_list[i] if yaw_list is not None else 0.0
            rot = (of if i < 2 else np.zeros(3)) + np.array([0, 0, yaw])
            rows.append([d[0] / POS_SCALE, d[1] / POS_SCALE, d[2] / POS_SCALE,
                         rot[0] / ROT_SCALE, rot[1] / ROT_SCALE, rot[2] / ROT_SCALE, grip])
        return rows

    def hold(self, n, grip=None):
        g = self.grip if grip is None else grip
        self.r.step([[0, 0, 0, 0, 0, 0, g]] * n)

    def move_to(self, target, grip, vmax=V_FREE, tol=0.01, max_steps=80, chunk=8, target_fn=None, z_first=False):
        """Closed-loop move of the TCP to target (or target_fn(step) for moving targets)."""
        used = 0
        while used < max_steps:
            tgt = np.asarray(target_fn(self.r.step_id + chunk) if target_fn else target, float)
            e = tgt - self.ee
            if np.linalg.norm(e) < tol:
                return True
            n = int(np.clip(math.ceil(np.max(np.abs(e)) / vmax), 1, chunk))
            if target_fn:
                tgt = np.asarray(target_fn(self.r.step_id + n), float)
                e = tgt - self.ee
            d = e / n
            m = np.max(np.abs(d))
            if m > vmax:
                d = d * (vmax / m)
            if z_first and e[2] > 0.02:
                d = np.array([0, 0, min(vmax, e[2] / n)])
            self.r.step(self.rows_for([d] * n, grip))
            used += n
        tgt = np.asarray(target_fn(self.r.step_id) if target_fn else target, float)
        return np.linalg.norm(tgt - self.ee) < tol * 2

    # ------------------------------------------------------------------ perception
    def front_detections(self, colours=None):
        img = self.r.imgs.get('front')
        if img is None:
            return []
        dets = P.detect_cubes(img, colours=colours, min_area=10, top_exclude=12)
        dets = [d for d in dets if d['area'] < 1500]
        if not dets:
            return []
        dets = dets[:32]
        pts = self.r.p2w('front', [(d['px'], d['py']) for d in dets], radius=1)
        out = []
        for d, p in zip(dets, pts):
            if p is None:
                continue
            if self.table_z is not None and (p[2] < self.table_z - 0.03 or p[2] > self.table_z + 0.12):
                continue
            if self.bin is not None and np.linalg.norm(p[:2] - self.bin['c']) < self.bin.get('r', 0.05):
                continue
            d = dict(d)
            d['p'] = p
            out.append(d)
        self.det_hist.append((self.r.step_id, [(d['colour'], d['p']) for d in out]))
        self.det_hist = self.det_hist[-40:]
        return out

    def update_velocity(self):
        """Vote for the common displacement of all cubes between two front detection sets."""
        if len(self.det_hist) < 2:
            return
        s2, B = self.det_hist[-1]
        if not B:
            return
        best = None
        for s1, A in reversed(self.det_hist[:-1]):
            dt = s2 - s1
            if dt < 6:
                continue
            if dt > 60:
                break
            best = (s1, A, dt)
            if dt >= 12:
                break
        if best is None:
            return
        s1, A, dt = best
        cand = []
        gate = 0.012 * dt + 0.02  # 0.24 m/s max belt speed
        for cb, pb in B:
            for ca, pa in A:
                if ca != cb:
                    continue
                d = pb[:2] - pa[:2]
                if np.linalg.norm(d) <= gate:
                    cand.append(d)
        if not cand:
            return
        cand = np.array(cand)
        tol = 0.012 + 0.0005 * dt
        votes = [(np.sum(np.linalg.norm(cand - c, axis=1) < tol), i) for i, c in enumerate(cand)]
        nv, i = max(votes)
        inl = cand[np.linalg.norm(cand - cand[i], axis=1) < tol]
        nb = len(B)
        if nv >= 2 or (nv == 1 and nb == 1 and len(A) <= 2):
            v = np.mean(inl, 0) / dt
            self.vel_samples.append((v, nv, dt))
            self.vel_samples = self.vel_samples[-10:]
            vs = np.array([s[0] for s in self.vel_samples])
            w = np.array([s[1] * s[2] for s in self.vel_samples], float)
            med = np.median(vs, 0)
            ok = np.linalg.norm(vs - med, axis=1) < 0.0015
            if ok.sum() >= 1:
                vv = np.average(vs[ok], 0, weights=w[ok])
                if np.linalg.norm(vv) < 0.0003:
                    vv = vv * 0.0
                self.vel = np.array([vv[0], vv[1], 0.0])
                self.vel_conf = int(ok.sum())

    def on_belt(self, xy):
        if self.mat_pts is None or len(self.mat_pts) < 4:
            return False
        if np.linalg.norm(self.vel[:2]) < 1e-4:
            return False
        u = self.vel[:2] / np.linalg.norm(self.vel[:2])
        n = np.array([-u[1], u[0]])
        s = self.mat_pts[:, :2] @ u
        l = self.mat_pts[:, :2] @ n
        return (s.min() - 0.03 <= xy @ u <= s.max() + 0.03) and (l.min() - 0.02 <= xy @ n <= l.max() + 0.02)

    def belt_end(self):
        """Downstream end coordinate along the belt direction (or None)."""
        if self.mat_pts is None or len(self.mat_pts) < 4 or np.linalg.norm(self.vel[:2]) < 1e-4:
            return None
        u = self.vel[:2] / np.linalg.norm(self.vel[:2])
        return float((self.mat_pts[:, :2] @ u).max()), u

    def predict(self, tgt, step):
        p = tgt['p'].copy()
        if tgt.get('moving', True):
            p[:2] = p[:2] + self.vel[:2] * (step - tgt['s'])
        return p

    # ------------------------------------------------------------------ phases
    def init_scene(self):
        self.r.observe()
        self.r.status()
        instr = IO.find_first(self.r.meta, ('instruction', 'language_instruction', 'goal', 'task_instruction'),
                              lambda v: isinstance(v, str) and len(v) > 5) or ''
        self.need = parse_instruction(instr)
        self.log('init', version=VERSION, instruction=instr, need=self.need, max_steps=self.r.max_steps,
                 limits=self.r.limits, state=self.r.state, cams=sorted(self.r.imgs.keys()))
        if not self.need:
            self.log('WARN_no_counts_parsed')
        self.done = {k: 0 for k in self.need}
        st = self.r.state
        if st is not None:
            R = rotvec_to_R(st[3:6])
            idx = int(np.argmax(np.abs(R[2, :])))
            sgn = -1.0 if R[2, idx] > 0 else 1.0  # make the axis point down
            self.approach_axis = (idx, sgn)
            tilt = self.tilt()
            if tilt > 25:
                self.orient_ok = False
            self.home = st[:3].copy()
            self.log('orientation', R=R.round(3).tolist(), approach_axis=self.approach_axis, tilt=tilt)
        self.save_img('init')
        self.detect_static()

    def detect_static(self):
        img = self.r.imgs.get('front')
        if img is None:
            self.log('WARN_no_front_image')
            return
        hc = P.hsv(img)
        # table height: wood-like pixels (hue 8-40, S 0.35-0.76) in lower image half
        h, s, v = hc
        wood = (h > 8) & (h < 40) & (s > 0.35) & (s < 0.76) & (v > 0.12)
        wood[:120] = False
        ys, xs = np.nonzero(wood)
        if len(xs) > 20:
            idx = np.linspace(0, len(xs) - 1, 16).astype(int)
            pts = [p for p in self.r.p2w('front', list(zip(xs[idx], ys[idx])), radius=1) if p is not None]
            if pts:
                self.table_z = float(np.median([p[2] for p in pts]))
        mat = P.detect_mat(img, hsv_cache=hc)
        if mat is not None:
            m = mat['mask'].copy()
            try:
                import cv2
                m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            except Exception:
                pass
            ys, xs = np.nonzero(m)
            if len(xs) > 30:
                order = np.lexsort((ys, xs))
                idx = np.linspace(0, len(xs) - 1, 24).astype(int)
                sel = list(zip(xs[order][idx], ys[order][idx]))
                pts = [p for p in self.r.p2w('front', sel, radius=1) if p is not None]
                if len(pts) >= 4:
                    self.mat_pts = np.array(pts)
                    self.belt_z = float(np.median(self.mat_pts[:, 2]))
        if self.belt_z is None:
            self.belt_z = self.table_z
        b = P.detect_bin(img, hsv_cache=hc)
        if b is not None:
            self.bin = self.locate_bin_from(img, b, 'front', hc)
        bt = P.detect_button(img, b, hsv_cache=hc)
        if bt is not None:
            self.button = self.locate_button_from(img, bt, 'front')
        self.log('scene', table_z=self.table_z, belt_z=self.belt_z, bin=self.bin, button=self.button,
                 mat=None if mat is None else mat['bbox'], n_mat_pts=None if self.mat_pts is None else len(self.mat_pts),
                 bin_px=None if b is None else (b['cx'], b['cy'], b['area']),
                 btn_px=None if bt is None else (bt['cx'], bt['cy'], bt['area']))

    def sample_window(self, cam, bbox, margin, max_pts=192, shape=(256, 256)):
        """Back-project a regular pixel grid over bbox expanded by margin. Returns Nx3 array."""
        x, y, w, h = bbox
        H, W = shape[:2]
        x0, x1 = max(0, int(x - margin)), min(W - 1, int(x + w + margin))
        y0, y1 = max(0, int(y - margin)), min(H - 1, int(y + h + margin))
        area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
        stp = max(1, int(math.ceil(math.sqrt(area / float(max_pts)))))
        pix = [(u, v) for v in range(y0, y1 + 1, stp) for u in range(x0, x1 + 1, stp)][:max_pts]
        pts = []
        for i in range(0, len(pix), 32):
            pts += [p for p in self.r.p2w(cam, pix[i:i + 32], radius=0) if p is not None]
        return np.array(pts) if pts else np.zeros((0, 3))

    def locate_bin_from(self, img, b, cam, hc, rim_ref=None):
        """Bin centre = mean xy of rim-top points (highest surface around the dark interior)."""
        x, y, w, h = b['bbox']
        margin = max(4, int(0.6 * max(w, h)))
        P3 = self.sample_window(cam, b['bbox'], margin, max_pts=192, shape=img.shape)
        if len(P3) < 8:
            return self.bin
        z = P3[:, 2]
        tz = self.table_z if self.table_z is not None else float(np.percentile(z, 10))
        ok = (z < tz + 0.20)
        if rim_ref is not None:
            ok &= np.abs(z - rim_ref) < 0.02
        if ok.sum() < 6:
            return self.bin
        zz = z[ok]
        rim_z = float(np.percentile(zz, 92)) if rim_ref is None else float(np.median(zz[np.abs(zz - rim_ref) < 0.012])) if (np.abs(zz - rim_ref) < 0.012).sum() > 3 else rim_ref
        top = ok & (np.abs(z - rim_z) < 0.008)
        if top.sum() < 5:
            return self.bin
        T = P3[top, :2]
        c = T.mean(0)
        for _ in range(2):
            r = np.linalg.norm(T - c, axis=1)
            keep = r < 2.0 * np.median(r) + 0.005
            if keep.sum() >= 5:
                c = T[keep].mean(0)
        rad = float(np.median(np.linalg.norm(T - c, axis=1)))
        # the interior should be markedly lower than the rim
        inner = P3[ok & (np.linalg.norm(P3[:, :2] - c, axis=1) < rad * 0.5)]
        depth = float(rim_z - np.median(inner[:, 2])) if len(inner) else None
        return dict(c=c, rim_z=rim_z, r=float(np.clip(rad + 0.015, 0.03, 0.10)), n=int(top.sum()), src=cam,
                    depth=depth, half=rad)

    def locate_button_from(self, img, bt, cam):
        P3 = self.sample_window(cam, bt['bbox'], 1, max_pts=96, shape=img.shape)
        pts = self.r.p2w(cam, [(bt['px'], bt['py'])], radius=1)
        if len(P3) < 4:
            return pts[0] if pts and pts[0] is not None else self.button
        z = P3[:, 2]
        tz = self.table_z if self.table_z is not None else float(np.percentile(z, 5))
        ok = z < tz + 0.12
        if ok.sum() < 4:
            return pts[0] if pts and pts[0] is not None else self.button
        ztop = float(np.percentile(z[ok], 95))
        cap = P3[ok & (z > ztop - 0.006)]
        base = P3[ok & (z > tz + 0.004)]
        c = base[:, :2].mean(0) if len(base) >= 4 else cap[:, :2].mean(0)
        if len(cap) >= 3:
            c = 0.5 * c + 0.5 * cap[:, :2].mean(0)
        self.log('button_fit', n=len(P3), n_cap=len(cap), n_base=len(base), ztop=ztop, c=c)
        return np.array([c[0], c[1], ztop])

    def wait_for_velocity(self, max_steps=60):
        """Idle in place (gripper open) collecting front detections until the belt velocity is known."""
        used = 0
        while used < max_steps:
            self.hold(6, -1.0)
            used += 6
            self.front_detections()
            self.update_velocity()
            if self.vel_conf >= 2:
                break
        self.log('velocity', vel=self.vel, conf=self.vel_conf, used=used,
                 n_samples=len(self.vel_samples))

    def remaining_need(self):
        return {k: self.need[k] - self.done.get(k, 0) for k in self.need if self.need[k] - self.done.get(k, 0) > 0}

    def select_target(self, dets):
        rem = self.remaining_need()
        if not rem:
            return None
        cands = [d for d in dets if d['colour'] in rem]
        if not cands:
            return None
        ee = self.ee
        end = self.belt_end()
        best = None
        s_now = self.r.step_id
        vspeed = V_FREE * 0.85
        for d in cands:
            tgt = dict(colour=d['colour'], p=d['p'].copy(), s=s_now, moving=self.on_belt(d['p'][:2]))
            if any(np.linalg.norm(tgt['p'][:2] - bl[1]) < 0.03 and s_now - bl[0] < 60 and np.linalg.norm(self.vel) < 1e-4
                   for bl in self.blacklist):
                continue
            T = 10
            for _ in range(4):
                p = self.predict(tgt, s_now + T)
                T = int(math.ceil(np.linalg.norm(p[:2] - ee[:2]) / vspeed)) + 14
            p_done = self.predict(tgt, s_now + T + 30)
            feasible = True
            margin = None
            if end is not None and tgt['moving']:
                smax, u = end
                margin = smax - float(p_done[:2] @ u)
                if margin < 0.04:
                    feasible = False
            if self.r.remaining_steps() < T + 60 + STEP_RESERVE_PRESS:
                feasible = False
            if not feasible:
                continue
            # clearance to other cubes at grasp time
            clear = 1.0
            for o in dets:
                if o is d:
                    continue
                dd = np.linalg.norm(o['p'][:2] - d['p'][:2])
                clear = min(clear, dd)
            score = T + (40 if clear < 0.05 else 0) + (15 if clear < 0.065 else 0)
            if best is None or score < best[0]:
                best = (score, tgt, T, clear, margin)
        if best is None:
            return None
        score, tgt, T, clear, margin = best
        self.log('select', colour=tgt['colour'], p=tgt['p'], moving=tgt['moving'], T=T, clear=clear, margin=margin)
        return tgt

    def hover_z(self):
        base = self.belt_z if self.belt_z is not None else (self.table_z if self.table_z is not None else self.ee[2] - 0.2)
        return base + self.cube + HOVER_DZ

    def approach(self, tgt):
        """Fly to hover above predicted cube position. Returns True when there."""
        hz = self.hover_z()
        steps0 = self.r.step_id
        # first make sure we're not too low (rise vertically if needed)
        if self.ee[2] < hz - 0.03:
            self.move_to(np.array([self.ee[0], self.ee[1], hz]), -1.0, tol=0.02, max_steps=12)

        def fn(step):
            p = self.predict(tgt, step)
            return np.array([p[0], p[1], hz])
        n_upd = 0
        while self.r.step_id - steps0 < 90:
            e = fn(self.r.step_id) - self.ee
            if np.linalg.norm(e[:2]) < 0.012 and abs(e[2]) < 0.025:
                return True
            dist = np.max(np.abs(e))
            n = int(np.clip(math.ceil(dist / V_FREE), 1, 8))
            tg = fn(self.r.step_id + n)
            d = (tg - self.ee) / n
            m = np.max(np.abs(d))
            if m > V_FREE * 1.3:
                d *= V_FREE * 1.3 / m
            self.r.step(self.rows_for([d] * n, -1.0))
            # opportunistic update of the target from the front camera (only when far)
            if np.linalg.norm(e[:2]) > 0.06 and n_upd < 4:
                dets = self.front_detections()
                self.update_velocity()
                pr = self.predict(tgt, self.r.step_id)
                best = None
                for dd in dets:
                    if dd['colour'] != tgt['colour']:
                        continue
                    dist2 = np.linalg.norm(dd['p'][:2] - pr[:2])
                    if dist2 < 0.04 and (best is None or dist2 < best[0]):
                        best = (dist2, dd)
                if best is not None:
                    tgt['p'] = 0.5 * pr + 0.5 * best[1]['p']
                    tgt['s'] = self.r.step_id
                    n_upd += 1
            end = self.belt_end()
            if end is not None and tgt.get('moving'):
                smax, u = end
                if smax - float(self.predict(tgt, self.r.step_id + 25)[:2] @ u) < 0.03:
                    self.log('approach_abort_belt_end')
                    return False
        self.log('approach_timeout', err=(fn(self.r.step_id) - self.ee).tolist())
        return False

    def wrist_find(self, colour, pred_xy, max_dist=0.06, want_yaw=False):
        """Locate a cube of colour in the wrist image near pred_xy. Returns dict or None."""
        img = self.r.imgs.get('wrist')
        if img is None:
            return None
        hc = P.hsv(img)
        dets = [d for d in P.detect_cubes(img, colours=[colour], min_area=25, hsv_cache=hc) if d['area'] < 30000]
        if not dets:
            return None
        dets.sort(key=lambda d: -d['area'])
        dets = dets[:6]
        found = []
        for d in dets:
            top, allm = P.blob_top_face(img, d, hc)
            if top is None or top.sum() < 6:
                top = None
            if top is not None:
                ys, xs = np.nonzero(top)
                c = (float(xs.mean()), float(ys.mean()))
                dd = (xs - c[0]) ** 2 + (ys - c[1]) ** 2
                j = int(np.argmin(dd))
                pix = (int(xs[j]), int(ys[j]))
            else:
                pix = (d['px'], d['py'])
            found.append((d, top, allm, pix))
        pts = self.r.p2w('wrist', [f[3] for f in found], radius=1)
        best = None
        for (d, top, allm, pix), p in zip(found, pts):
            if p is None:
                continue
            dist = np.linalg.norm(p[:2] - pred_xy)
            if dist < max_dist and (best is None or dist < best[0]):
                best = (dist, d, top, allm, pix, p)
        if best is None:
            self.log('wrist_find_none', colour=colour, n=len(found),
                     pts=[None if p is None else p.round(3).tolist() for p in pts], pred=pred_xy)
            return None
        dist, d, top, allm, pix, p = best
        out = dict(p=p, pix=pix, area=d['area'], dist=dist)
        # belt / table height around the cube (pixels just outside the blob)
        try:
            ring = P.ring_pixels(img.shape, allm, pix[0], pix[1], n_dirs=8, extra=4)
            rp = [q for q in self.r.p2w('wrist', ring, radius=0) if q is not None]
            zs = sorted(q[2] for q in rp if q[2] < p[2] - 0.01)
            if len(zs) >= 2:
                out['base_z'] = float(np.median(zs))
        except Exception:
            pass
        if want_yaw and top is not None:
            try:
                ma = P.min_area_angle(top)
                if ma is not None:
                    a, (w, h), (cx, cy) = ma
                    k = max(3.0, 0.3 * min(w, h)) if min(w, h) > 0 else 4.0
                    q = [(pix[0] + k * math.cos(a), pix[1] + k * math.sin(a)),
                         (pix[0] - k * math.cos(a), pix[1] - k * math.sin(a)),
                         (pix[0] + k, pix[1]), (pix[0] - k, pix[1])]
                    H, W = top.shape
                    q = [(min(W - 1, max(0, u)), min(H - 1, max(0, v))) for u, v in q]
                    qq = self.r.p2w('wrist', q, radius=0)
                    if all(x is not None for x in qq):
                        e_dir = qq[0] - qq[1]
                        x_dir = qq[2] - qq[3]
                        if np.linalg.norm(e_dir[:2]) > 0.004 and np.linalg.norm(x_dir[:2]) > 0.004:
                            out['yaw_delta'] = wrap_quarter(yaw_of(e_dir) - yaw_of(x_dir))
                            out['img_angle'] = a
                            out['xdir_yaw'] = yaw_of(x_dir)
            except Exception as e:
                self.log('yaw_err', err=repr(e)[:200])
        return out

    def choose_yaw(self, delta):
        """Pick an equivalent (mod 90 deg) rotation that keeps the cumulative yaw small."""
        opts = [delta + k * math.pi / 2 for k in (-1, 0, 1)]
        opts = [o for o in opts if abs(o) <= math.pi / 2 + 1e-6]
        return min(opts, key=lambda o: abs(self.yaw_total + o) + 0.3 * abs(o))

    def grasp(self, tgt):
        """Refine with the wrist camera, align yaw, descend while tracking, close, lift."""
        colour = tgt['colour']
        pred = self.predict(tgt, self.r.step_id)
        w = self.wrist_find(colour, pred[:2], max_dist=0.07, want_yaw=True)
        self.save_img('hover_%s' % colour)
        if w is None:
            # one more look after a short tracking move
            hz = self.hover_z()
            self.move_to(None, -1.0, tol=0.008, max_steps=4, chunk=4,
                         target_fn=lambda s: np.r_[self.predict(tgt, s)[:2], hz])
            pred = self.predict(tgt, self.r.step_id)
            w = self.wrist_find(colour, pred[:2], max_dist=0.08, want_yaw=True)
        if w is not None:
            tgt['p'] = np.array([w['p'][0], w['p'][1], w['p'][2]])
            tgt['s'] = self.r.step_id
            tgt['top_z'] = float(w['p'][2])
            if 'base_z' in w:
                size = tgt['top_z'] - w['base_z']
                if 0.015 < size < 0.08:
                    self.cube_samples.append(size)
                    self.cube = float(np.median(self.cube_samples))
                tgt['base_z'] = w['base_z']
        else:
            self.log('grasp_no_wrist_fix')
        base_z = tgt.get('base_z', tgt.get('top_z', (self.belt_z or 0) + self.cube) - self.cube)
        top_z = tgt.get('top_z', base_z + self.cube)
        # yaw alignment
        yaw = 0.0
        if w is not None and 'yaw_delta' in w:
            yaw = self.choose_yaw(w['yaw_delta'] * self.yaw_sign)
            if abs(yaw) < math.radians(5):
                yaw = 0.0
        hz = top_z + HOVER_DZ * 0.7
        self.log('grasp_plan', colour=colour, p=tgt['p'], top_z=top_z, base_z=base_z, cube=self.cube,
                 yaw_deg=math.degrees(yaw), wrist=None if w is None else dict(dist=w['dist'], area=w['area'],
                 yd=None if 'yaw_delta' not in w else math.degrees(w['yaw_delta'])))
        if yaw != 0.0:
            n = max(2, int(math.ceil(abs(yaw) / 0.35)))
            s0 = self.r.step_id
            disp = []
            cur = self.ee.copy()
            for k in range(1, n + 1):
                pk = self.predict(tgt, s0 + k)
                want = np.array([pk[0], pk[1], hz])
                dk = (want - cur) / max(1, (n - k + 1))
                dk = np.clip(dk, -V_FREE, V_FREE)
                disp.append(dk)
                cur = cur + dk
            self.r.step(self.rows_for(disp, -1.0, yaw_list=[yaw / n] * n))
            self.yaw_total += yaw
            # verify rotation sign once with a fresh wrist look
            w2 = self.wrist_find(colour, self.predict(tgt, self.r.step_id)[:2], max_dist=0.07, want_yaw=True)
            if w2 is not None:
                tgt['p'] = np.array([w2['p'][0], w2['p'][1], w2['p'][2]])
                tgt['s'] = self.r.step_id
                if 'yaw_delta' in w2:
                    resid = w2['yaw_delta']
                    self.log('yaw_check', before=math.degrees(w['yaw_delta']), after=math.degrees(resid), cmd=math.degrees(yaw))
                    if abs(yaw) > math.radians(15) and abs(resid) > abs(w['yaw_delta']) + math.radians(8):
                        self.yaw_sign *= -1.0
                        fix = self.choose_yaw(resid * 1.0)
                        self.log('yaw_sign_flip', fix=math.degrees(fix))
                        self.r.step(self.rows_for([np.zeros(3)] * 2, -1.0,
                                                  yaw_list=[fix / 2] * 2))
                        self.yaw_total += fix
                    elif abs(resid) > math.radians(10):
                        fix = resid
                        self.r.step(self.rows_for([np.zeros(3)] * 2, -1.0, yaw_list=[fix / 2] * 2))
                        self.yaw_total += fix
        # descend while tracking the cube
        gz = base_z + 0.5 * self.cube
        gz = max(gz, base_z + 0.012)
        s0 = self.r.step_id
        cur = self.ee.copy()
        z_start = float(cur[2])
        n_desc = int(np.clip(math.ceil((z_start - gz) / (V_NEAR * 1.3)), 3, 10))
        disp = []
        for k in range(1, n_desc + 1):
            pk = self.predict(tgt, s0 + k)
            want = np.array([pk[0], pk[1], z_start + (gz - z_start) * k / n_desc])
            dk = want - cur
            dk[:2] = np.clip(dk[:2], -V_FREE, V_FREE)
            disp.append(dk)
            cur = cur + dk
        self.r.step(self.rows_for(disp, -1.0))
        # final xy correction + close while moving with the belt
        s1 = self.r.step_id
        e0 = self.predict(tgt, s1 + 1) - self.ee
        e0[2] = gz - self.ee[2]
        n_close = 8
        disp = []
        for k in range(n_close):
            d = self.vel.copy() * 1.0
            if k < 2:
                d = d + np.array([e0[0], e0[1], e0[2]]) / 2
            disp.append(d)
        self.r.step(self.rows_for(disp, 1.0, fix=False))
        f1 = self.fingers()
        # a couple more closing steps if fingers still moving
        self.r.step(self.rows_for([self.vel.copy()] * 3, 1.0, fix=False))
        f2 = self.fingers()
        self.grip = 1.0
        # lift
        lift_z = max(self.ee[2] + 0.12, (self.bin['rim_z'] if self.bin else 0) + CARRY_DZ + self.cube)
        self.move_to(np.array([self.ee[0], self.ee[1], lift_z]), 1.0, tol=0.02, max_steps=14, chunk=7)
        f3 = self.fingers()
        held_colour = self.held_colour()
        self.save_img('lift_%s' % colour)
        empty_thr = max(0.006, 0.2 * self.cube)
        held = f3 > empty_thr
        self.log('grasp_result', f_close=f1, f_after=f2, f_lift=f3, held=held, held_colour=held_colour,
                 tilt=self.tilt(), ee=self.ee)
        if held and held_colour is not None and held_colour != colour:
            self.log('wrong_colour_held', want=colour, got=held_colour)
            return 'wrong'
        if held:
            return 'ok'
        if held_colour == colour and f3 > 0.003:
            return 'ok'
        return 'miss'

    def held_colour(self):
        img = self.r.imgs.get('wrist')
        if img is None:
            return None
        try:
            H, W = img.shape[:2]
            sub = img[H // 2:, W // 4:]
            hc = P.hsv(sub)
            best = None
            for col in P.CUBE_COLOURS:
                m = P.colour_mask(sub, col, hc)
                a = int(m.sum())
                if best is None or a > best[1]:
                    best = (col, a)
            if best and best[1] > 0.12 * sub.shape[0] * sub.shape[1]:
                return best[0]
        except Exception:
            pass
        return None

    def release_here(self):
        self.r.step(self.rows_for([np.zeros(3)] * 6, -1.0, fix=False))
        self.grip = -1.0
        self.move_to(self.ee + np.array([0, 0, 0.05]), -1.0, tol=0.02, max_steps=6)

    def carry_and_drop(self, colour):
        if self.bin is None:
            self.log('WARN_no_bin')
            return False
        rim = self.bin['rim_z']
        cz = rim + CARRY_DZ + self.cube
        cz = max(cz, self.ee[2] - 0.02)
        tgt = np.array([self.bin['c'][0], self.bin['c'][1], cz])
        self.move_to(np.array([self.ee[0], self.ee[1], cz]), 1.0, tol=0.03, max_steps=8, chunk=8)
        self.move_to(tgt, 1.0, vmax=V_FREE, tol=0.012, max_steps=60, chunk=8)
        # wrist refinement of bin centre
        try:
            img = self.r.imgs.get('wrist')
            if img is not None:
                hc = P.hsv(img)
                h, s, v = hc
                H = img.shape[0]
                b = P.detect_bin(img, top_exclude=0, hsv_cache=hc)
                if b is not None:
                    nb = self.locate_bin_from(img, b, 'wrist', hc, rim_ref=self.bin['rim_z'])
                    if nb is not None and nb is not self.bin and nb.get('src') == 'wrist' and nb.get('n', 0) >= 5:
                        dc = np.linalg.norm(nb['c'] - self.bin['c'])
                        self.log('bin_wrist', c=nb['c'], rim=nb['rim_z'], d=dc, n=nb['n'])
                        if dc < 0.04 and nb.get('n', 0) >= 10:
                            self.bin['c'] = 0.75 * self.bin['c'] + 0.25 * nb['c']
                            if abs(nb['rim_z'] - rim) < 0.04:
                                self.bin['rim_z'] = 0.5 * (rim + nb['rim_z'])
        except Exception as e:
            self.log('bin_refine_err', err=repr(e)[:200])
        self.save_img('overbin')
        rim = self.bin['rim_z']
        dz = rim + self.cube / 2 + DROP_CLEAR + 0.01
        self.move_to(np.array([self.bin['c'][0], self.bin['c'][1], dz]), 1.0, vmax=V_NEAR * 1.5, tol=0.008,
                     max_steps=20, chunk=6)
        # release
        self.r.step(self.rows_for([np.zeros(3)] * 6, -1.0, fix=False))
        self.grip = -1.0
        fo = self.fingers()
        self.move_to(np.array([self.ee[0], self.ee[1], cz]), -1.0, tol=0.02, max_steps=8, chunk=8)
        hc = self.held_colour()
        self.save_img('dropped_%s' % colour)
        self.log('drop', colour=colour, fingers_open=fo, still_held_colour=hc, bin=self.bin)
        return True

    def press_button(self):
        if self.button is None:
            self.log('WARN_no_button')
            return False
        b = self.button.copy()
        self.grip = 1.0
        top = b[2]
        above = np.array([b[0], b[1], top + 0.10])
        self.move_to(np.array([self.ee[0], self.ee[1], max(self.ee[2], top + 0.10)]), 1.0, tol=0.03, max_steps=10)
        self.move_to(above, 1.0, tol=0.01, max_steps=60)
        self.save_img('above_button')
        offsets = [(0, 0), (0.008, 0), (-0.008, 0), (0, 0.008), (0, -0.008), (0.012, 0.012)]
        for i, (ox, oy) in enumerate(offsets):
            depth = 0.02 + 0.01 * min(i, 2)
            self.move_to(np.array([b[0] + ox, b[1] + oy, top + 0.03]), 1.0, vmax=V_NEAR, tol=0.006, max_steps=16, chunk=6)
            zs = []
            for k in range(8):
                want = top - depth
                dz = np.clip(want - self.ee[2], -0.01, 0.0)
                self.r.step(self.rows_for([np.array([b[0] + ox - self.ee[0], b[1] + oy - self.ee[1], dz])], 1.0))
                zs.append(float(self.ee[2]))
            self.hold(3, 1.0)
            self.log('press_try', i=i, z=zs, top=top, ended=self.r.ended, complete=self.r.complete)
            self.move_to(np.array([self.ee[0], self.ee[1], top + 0.06]), 1.0, tol=0.01, max_steps=8)
        return False

    # ------------------------------------------------------------------ main loop
    def run(self):
        try:
            self.init_scene()
        except EpisodeOver:
            return
        except Exception as e:
            self.log('init_error', err=repr(e), tb=traceback.format_exc()[-1500:])
        if self.r.state is None:
            try:
                self.r.observe()
            except Exception:
                pass
        try:
            self.wait_for_velocity()
        except EpisodeOver:
            return
        except Exception as e:
            self.log('wait_error', err=repr(e), tb=traceback.format_exc()[-1500:])
        idle = 0
        n_err = 0
        self.no_state = 0
        while True:
            if self.r.ended:
                self.log('episode_ended', complete=self.r.complete)
                return
            if self.elapsed() > WALL_STOP:
                self.log('wall_stop')
                return
            try:
                rem = self.remaining_need()
                if not rem and self.need:
                    self.log('all_placed', done=self.done)
                    self.phase = 'PRESS'
                    self.press_button()
                    self.log('press_finished', ended=self.r.ended, complete=self.r.complete)
                    # if the episode is still running, keep trying the press a few more times
                    for _ in range(2):
                        if self.r.ended:
                            break
                        self.button = self.button + np.array([0, 0, -0.01]) if self.button is not None else None
                        self.press_button()
                    return
                if not self.need:
                    # instruction could not be parsed: nothing safe to do but idle
                    self.hold(20, -1.0)
                    continue
                if self.r.remaining_steps() < STEP_RESERVE_PRESS:
                    self.log('out_of_steps', rem=self.r.remaining_steps())
                    return
                if self.r.state is None:
                    self.no_state += 1
                    if self.no_state > 40:
                        self.log('no_state_giveup')
                        return
                    self.r.observe()
                    if self.r.state is None:
                        self.hold(2, -1.0)  # keep stepping (control progress) while state is missing
                    continue
                self.no_state = 0
                # make sure the gripper is empty and open
                if self.grip > 0:
                    self.release_here()
                self.phase = 'SELECT'
                dets = self.front_detections()
                self.update_velocity()
                tgt = self.select_target(dets)
                if tgt is None:
                    idle += 1
                    if idle % 10 == 1:
                        self.log('no_target', rem=rem, n_dets=len(dets), vel=self.vel,
                                 cols=sorted(set(d['colour'] for d in dets)))
                    # park high near home (keeps the front view clear) and wait
                    if idle == 1 or np.linalg.norm(self.ee - self.park()) > 0.05:
                        self.move_to(self.park(), -1.0, tol=0.03, max_steps=20)
                    else:
                        self.hold(5, -1.0)
                    continue
                idle = 0
                self.attempts += 1
                self.phase = 'APPROACH'
                ok = self.approach(tgt)
                if not ok:
                    self.fail_streak += 1
                    continue
                self.phase = 'GRASP'
                res = self.grasp(tgt)
                if res == 'ok':
                    self.phase = 'CARRY'
                    if self.carry_and_drop(tgt['colour']):
                        self.done[tgt['colour']] = self.done.get(tgt['colour'], 0) + 1
                        self.fail_streak = 0
                        self.log('count', done=self.done, need=self.need, attempts=self.attempts)
                elif res == 'wrong':
                    self.release_here()
                    self.fail_streak += 1
                else:
                    self.blacklist.append((self.r.step_id, tgt['p'][:2].copy()))
                    self.release_here()
                    self.fail_streak += 1
                    self.move_to(np.array([self.ee[0], self.ee[1], self.hover_z() + 0.03]), -1.0, tol=0.02, max_steps=8)
                n_err = 0
            except EpisodeOver:
                self.log('episode_over', complete=self.r.complete, done=self.done)
                return
            except Exception as e:
                n_err += 1
                self.log('loop_error', err=repr(e)[:400], tb=traceback.format_exc()[-2000:], n=n_err)
                if n_err > 25:
                    return
                try:
                    if self.r.ended:
                        return
                    self.r.observe()
                    if self.grip > 0:
                        pass
                    self.hold(2)
                except EpisodeOver:
                    return
                except Exception:
                    time.sleep(0.5)

    def park(self):
        p = getattr(self, 'home', None)
        if p is None:
            return self.ee
        return p.copy()


def main(ctx):
    c = None
    try:
        c = Controller(ctx)
        c.run()
    except BaseException as e:  # never propagate (a crashing controller is not a clean rehearsal)
        try:
            print(json.dumps(dict(ev='fatal', err=repr(e)[:500], tb=traceback.format_exc()[-2000:])), flush=True)
        except Exception:
            pass
    finally:
        try:
            if c is not None:
                c.log('end', elapsed=c.elapsed(), done=c.done, need=c.need, ended=c.r.ended, complete=c.r.complete,
                      calls=c.r.n_calls, step=c.r.step_id, track_ratio=c.r.track_ratio, vel=c.vel,
                      cube=c.cube, yaw_sign=c.yaw_sign)
        except Exception:
            pass
