"""PickXtimes controller v1 (built offline; R1 budget => the rehearsal is its first live run).

Transition table
  INIT      -> parse instruction (colour, N), front detection + depth points       -> TARGET_REF
  TARGET_REF-> hover above target, wrist refine target centre (fallback: front)    -> CUBE_REF
  CUBE_REF  -> hover above cube, wrist refine cube xyz + yaw (2 passes)            -> GRASP
  GRASP     -> align yaw, open, descend, close, lift; verify fingers + wrist view
               ok -> PLACE ; fail -> CUBE_REF (re-detect, max attempts)
  PLACE     -> above target at lift height, descend, open, rise; verify cube on target
               on target -> count += 1 ; off target -> CUBE_REF (not counted)
               count == N -> BUTTON else -> GRASP (cube re-measured from verify view)
  BUTTON    -> close gripper, hover, wrist refine button, press down in small chunks
               until episode ends / task_complete; retries with small xy offsets
Every robot reply is checked for episode end; main() never raises.
"""
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcpio  # noqa: E402
import vision  # noqa: E402

VERSION = 'pickx-v1'
OUT = Path('/workspace/output')

# ---- tunables (untested live: chosen from demo appearance and generic Panda geometry) ----
LIFT_H = 0.15          # TCP height above table while carrying (expert lifts high)
HOVER_CUBE_H = 0.14    # TCP height above table for wrist inspection of the cube
HOVER_TGT_H = 0.20     # TCP height above table for wrist inspection of the target
RISE_AFTER_PLACE = 0.09  # rise above cube top after release before verification view
V_FAST = 0.30          # action units (x0.05 m per step)
V_SLOW = 0.12
PLACE_DROP = 0.008     # cube bottom this far above target surface when released
ON_TARGET_TOL = 0.03   # m, cube centre vs target centre
MAX_GRASP_FAILS = 6
WALL_LIMIT = 1100.0    # s, controller budget guard (hard limit 1200)

NUMBERS = {'once': 1, 'twice': 2, 'thrice': 3, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
           'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12,
           'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
           'eighteen': 18, 'nineteen': 19, 'twenty': 20}


class EpisodeEnded(Exception):
    pass


def parse_instruction(text):
    t = (text or '').lower()
    color = 'green'
    m = re.search(r'pick up (?:the |a )?(\w+) cube', t)
    if m and m.group(1) in vision.COLOR_RULES:
        color = m.group(1)
    else:
        for c in ('green', 'red', 'blue', 'yellow', 'purple', 'orange', 'cyan'):
            if c + ' cube' in t:
                color = c; break
    n = None
    m = re.search(r'(\d+)\s*times', t)
    if m:
        n = int(m.group(1))
    if n is None:
        for w, v in NUMBERS.items():
            if re.search(r'\b' + w + r'\b\s*(times)?', t) and (w in ('once', 'twice', 'thrice') or re.search(r'\b' + w + r'\s+times', t)):
                n = v; break
    return color, n


# ---------------- rotation helpers ----------------
def rotvec_to_mat(r):
    r = np.asarray(r, float); th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def mat_to_rotvec(R):
    c = (np.trace(R) - 1) / 2.0
    c = min(1.0, max(-1.0, c)); th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: axis from the symmetric part
        A = (R + np.eye(3)) / 2.0
        k = np.sqrt(np.maximum(np.diag(A), 0))
        i = int(np.argmax(k))
        k = A[:, i] / max(math.sqrt(max(A[i, i], 1e-12)), 1e-12)
        k = k / np.linalg.norm(k)
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def wrap90(a):
    """wrap angle to (-pi/4, pi/4]"""
    q = math.pi / 2
    return (a + q / 2) % q - q / 2


# ---------------- controller ----------------
class Ctl:
    def __init__(self, ctx):
        self.ctx = ctx
        self.t0 = time.time()
        OUT.mkdir(parents=True, exist_ok=True)
        self.logf = open(OUT / 'controller_log.txt', 'a')
        self.meta_dumped = set()
        self.ended = False
        self.complete = False
        self.state = None
        self.step_id = None
        self.images = {}
        self.instruction = ''
        self.R_ref = None
        self.grip = -1.0
        self.n_steps_sent = 0
        self.max_steps = None
        self.state_seen = False
        self.rot_mode = 'world'
        self.rot_bad = 0
        self.rot_enabled = True

    # ---- logging ----
    def log(self, *a):
        s = '[%7.1fs] ' % (time.time() - self.t0) + ' '.join(str(x) for x in a)
        print(s, flush=True)
        try:
            self.logf.write(s + '\n'); self.logf.flush()
        except Exception:
            pass

    def dump_meta(self, tag, meta, reply=None):
        if tag in self.meta_dumped:
            return
        self.meta_dumped.add(tag)
        try:
            desc = []
            for b in (reply or {}).get('content', []) or []:
                if isinstance(b, dict):
                    desc.append({k: (v if k != 'data' else '<%d bytes>' % len(v)) for k, v in b.items()
                                 if k != 'text'} if b.get('type') != 'text' else {'type': 'text', 'len': len(b.get('text', ''))})
            with open(OUT / ('meta_%s.json' % tag), 'w') as f:
                json.dump({'meta': meta, 'blocks': desc}, f, indent=1, default=str)
        except Exception as e:
            self.log('dump_meta failed', e)

    # ---- reply handling ----
    def absorb(self, reply, tag):
        meta = mcpio.meta_of(reply)
        self.dump_meta(tag, meta, reply)
        st = mcpio.bfs_find(meta, 'state', lambda x: mcpio.is_num_list(x) and len(x) >= 8)
        if st is not None:
            self.state = np.array(st[:8], float)
            self.state_seen = True
        sid = mcpio.bfs_find(meta, 'step_id', lambda x: isinstance(x, int) and not isinstance(x, bool))
        if sid is not None:
            self.step_id = sid
        ins = mcpio.bfs_find(meta, 'instruction', lambda x: isinstance(x, str) and len(x) > 5)
        if ins:
            self.instruction = ins
        elif not self.instruction:
            for t in mcpio.text_blocks(reply):
                m = re.search(r'pick up[^.\n"]*times[^.\n"]*', t, re.I)
                if m:
                    self.instruction = m.group(0); break
        ended = mcpio.bfs_find(meta, 'episode_ended', lambda x: isinstance(x, bool))
        comp = mcpio.bfs_find(meta, 'task_complete', lambda x: isinstance(x, bool))
        if comp:
            self.complete = True
        if ended:
            self.ended = True
        imgs = mcpio.extract_images(reply, meta)
        if imgs:
            self.images = imgs
        return meta

    def check_end(self):
        if self.ended or self.complete:
            raise EpisodeEnded()

    def observe(self):
        r = self.ctx.call('robomme_observe')
        self.absorb(r, 'observe')
        return r

    @property
    def pos(self):
        return self.state[:3].copy()

    @property
    def R(self):
        return rotvec_to_mat(self.state[3:6])

    @property
    def finger(self):
        return float(self.state[6] + self.state[7])

    def send(self, rows):
        rows = [[float(max(-1.0, min(1.0, v))) for v in r] for r in rows]
        while rows:
            chunk, rows = rows[:100], rows[100:]
            rot_cmd = any(abs(v) > 1e-6 for r in chunk for v in r[3:6])
            e0 = self.rot_err() if rot_cmd else 0.0
            self.state_seen = False
            r = self.ctx.call('robomme_step', actions=chunk)
            self.n_steps_sent += len(chunk)
            self.absorb(r, 'step')
            self.check_end()
            if not self.state_seen:
                self.log('step reply without state; observing')
                self.observe()
                self.check_end()
            if rot_cmd and e0 > 0.05:
                e1 = self.rot_err()
                if e1 > e0 + 0.02:
                    self.rot_bad += 1
                    self.log('rotation monitor: err %.3f -> %.3f (mode %s, bad %d)' % (e0, e1, self.rot_mode, self.rot_bad))
                    if self.rot_bad >= 2:
                        if self.rot_mode == 'world':
                            self.rot_mode = 'tool'; self.rot_bad = 0
                        else:
                            self.rot_enabled = False
                            self.R_ref = self.R.copy()
                        self.log('rotation monitor -> mode %s enabled %s' % (self.rot_mode, self.rot_enabled))
                elif e1 < e0 - 0.02:
                    self.rot_bad = 0

    # ---- motion primitives ----
    def rot_err(self):
        if self.R_ref is None or self.state is None:
            return 0.0
        return float(np.linalg.norm(mat_to_rotvec(self.R_ref @ self.R.T)))

    def rot_rows(self):
        """Per-row rotation command list to bring orientation back to R_ref.
        Docs: rotation deltas are world-frame axis-angle. A monitor in send() flips to the
        tool-frame convention (or disables rotation) if commanded rotation increases the error."""
        if self.R_ref is None or not self.rot_enabled:
            return []
        rv = mat_to_rotvec(self.R_ref @ self.R.T)
        ang = np.linalg.norm(rv)
        if ang < 0.03:
            return []
        if self.rot_mode == 'tool':
            rv = self.R.T @ rv
        m = int(math.ceil(ang / 0.12))
        return [rv / m / 0.5] * min(m, 3)   # at most ~0.36 rad per call; re-measured each call

    def move_to(self, target, vmax=V_FAST, tol=0.004, max_calls=16, grip=None):
        """Closed-loop straight move of the TCP to a world target, holding R_ref.
        Stops on convergence, or after 3 calls without progress (blocked / contact)."""
        target = np.asarray(target, float)
        if grip is not None:
            self.grip = grip
        stalls = 0
        for _ in range(max_calls):
            err = target - self.pos
            d = float(np.linalg.norm(err))
            rr = self.rot_rows()
            if d < tol and not rr:
                return True
            per = vmax * 0.05
            n = min(max(1, int(math.ceil(d / per))), 40) if d >= tol else 0
            nrow = max(n, len(rr))
            rows = []
            for i in range(nrow):
                a = (err / n) / 0.05 if i < n else np.zeros(3)
                w = rr[i] if i < len(rr) else np.zeros(3)
                rows.append([a[0], a[1], a[2], w[0], w[1], w[2], self.grip])
            self.send(rows)
            d2 = float(np.linalg.norm(target - self.pos))
            if d >= tol and d - d2 < max(0.001, 0.15 * min(d, per * n)):
                stalls += 1
                if stalls >= 3:
                    self.log('move_to stalled at d=%.4f target %s' % (d2, np.round(target, 3).tolist()))
                    break
            else:
                stalls = 0
        d = float(np.linalg.norm(target - self.pos))
        return d < 3 * tol

    def hold(self, n, grip):
        self.grip = grip
        rows = []
        rr = self.rot_rows()
        for i in range(n):
            w = rr[i] if i < len(rr) else np.zeros(3)
            rows.append([0, 0, 0, w[0], w[1], w[2], grip])
        self.send(rows)

    def set_gripper(self, grip, max_rows=16):
        self.grip = grip
        last = self.finger
        sent = 0
        while sent < max_rows:
            self.hold(4, grip); sent += 4
            f = self.finger
            if sent >= 8 and abs(f - last) < 0.0008:
                break
            last = f
        return self.finger

    # ---- perception ----
    def points(self, cam, uvs, radius=1):
        if not uvs:
            return []
        sel = [{'camera': cam, 'u': float(u), 'v': float(v)} for (u, v) in uvs][:32]
        try:
            r = self.ctx.call('robomme_pixel_to_position', observation_step_id=int(self.step_id or 0),
                              selections=sel, depth_window_radius=int(radius))
        except Exception as e:
            self.log('pixel_to_position error', str(e)[:300])
            try:
                self.observe()
                r = self.ctx.call('robomme_pixel_to_position', observation_step_id=int(self.step_id or 0),
                                  selections=sel, depth_window_radius=int(radius))
            except Exception as e2:
                self.log('pixel_to_position retry error', str(e2)[:300])
                return [None] * len(sel)
        pts, meta = mcpio.parse_points(r, len(sel))
        self.dump_meta('p2p', meta, r)
        return pts

    def table_ring(self, u, v, rad, n=8, H=256):
        out = []
        for k in range(n):
            a = 2 * math.pi * k / n
            uu, vv = u + rad * math.cos(a), v + rad * math.sin(a)
            if 2 <= uu < H - 2 and 2 <= vv < H - 2:
                out.append((uu, vv))
        return out

    def front_elev(self):
        """Inverse-depth plane fit on the front depth image -> elevated mask (or None)."""
        d = self.images.get(('front', 'depth'))
        if d is None or d.ndim != 2:
            return None
        try:
            valid = d > 0
            if valid.mean() < 0.3:
                return None
            H, W = d.shape
            vv, uu = np.mgrid[0:H, 0:W]
            inv = np.zeros_like(d); inv[valid] = 1.0 / d[valid]
            sel = valid.copy(); sel[: H // 8] = False
            for _ in range(4):
                A = np.stack([uu[sel], vv[sel], np.ones(sel.sum())], 1)
                coef, *_ = np.linalg.lstsq(A, inv[sel], rcond=None)
                pred = coef[0] * uu + coef[1] * vv + coef[2]
                res = np.where(valid, d - 1.0 / np.maximum(pred, 1e-9), 0)
                scale = np.median(np.abs(res[sel])) + 1e-9
                sel = valid & (np.abs(res) < 3 * scale + 1e-6 * np.median(d[valid]))
                sel[: H // 8] = False
            # elevated = significantly closer than plane (raw units; mm typical)
            thr = max(4 * np.median(np.abs(res[sel])), 0.004 * np.median(d[valid]) if np.median(d[valid]) > 50 else 0.004)
            return valid & (res < -thr)
        except Exception as e:
            self.log('elev failed', e)
            return None

    def wrist_find(self, what, color, expect_xy=None, gate=0.06):
        """Wrist-camera detection -> (world point, blob, yaw_world or None)."""
        img = self.images.get(('wrist', 'rgb'))
        if img is None:
            return None
        cands = vision.wrist_candidates(img, what, color)
        if not cands:
            return None
        cands = cands[:4]
        uvs = [(c['u'], c['v']) for c in cands]
        pts = self.points('wrist', uvs, radius=2)
        best = None
        for c, p in zip(cands, pts):
            if p is None:
                continue
            dist = 0.0 if expect_xy is None else float(np.linalg.norm(p['p'][:2] - expect_xy[:2]))
            if expect_xy is not None and dist > gate:
                continue
            key = dist - 1e-5 * c['area']
            if best is None or key < best[0]:
                best = (key, c, p['p'])
        if best is None:
            return None
        _, c, p = best
        yaw = None
        if what == 'cube':
            try:
                t = vision.blob_axis_angle(c)
                half = 0.3 * math.sqrt(c['area'])
                q = self.points('wrist', [(c['u'], c['v']), (c['u'] + half * math.cos(t), c['v'] + half * math.sin(t)),
                                          (c['u'] - half * math.cos(t), c['v'] - half * math.sin(t))], radius=1)
                if q[1] is not None and q[2] is not None:
                    dv = q[1]['p'] - q[2]['p']
                    if np.linalg.norm(dv[:2]) > 0.004 and abs(dv[2]) < 0.6 * np.linalg.norm(dv[:2]):
                        yaw = math.atan2(dv[1], dv[0])
            except Exception as e:
                self.log('yaw est failed', e)
        return p, c, yaw

    def wrist_fraction(self, color):
        img = self.images.get(('wrist', 'rgb'))
        if img is None:
            return None
        return float(vision.color_mask(img, color).mean())

    def tool_yaw(self):
        R = self.R
        return math.atan2(R[1, 0], R[0, 0])


def main(ctx):
    c = Ctl(ctx)
    c.log('controller', VERSION, 'start')
    try:
        run(c)
    except EpisodeEnded:
        c.log('episode ended; complete=%s steps_sent=%d' % (c.complete, c.n_steps_sent))
    except Exception:
        c.log('FATAL', traceback.format_exc())
        # last-resort: nothing else safe to do
    c.log('main returns; complete=%s ended=%s steps=%d' % (c.complete, c.ended, c.n_steps_sent))
    try:
        c.logf.close()
    except Exception:
        pass


def run(c):
    # ---------- INIT ----------
    try:
        st = c.ctx.call('robomme_status')
        meta = mcpio.meta_of(st)
        c.dump_meta('status', meta, st)
        ins = mcpio.bfs_find(meta, 'instruction', lambda x: isinstance(x, str) and len(x) > 5)
        if ins:
            c.instruction = ins
        for k in ('max_steps', 'step_limit', 'max_episode_steps', 'horizon'):
            v = mcpio.bfs_find(meta, k, lambda x: isinstance(x, int))
            if v:
                c.max_steps = v; break
        c.log('status limits', json.dumps(mcpio.bfs_find(meta, 'episode_limits') or {}, default=str)[:300])
    except Exception as e:
        c.log('status failed', str(e)[:200])
    c.observe()
    c.check_end()
    color, N = parse_instruction(c.instruction)
    if N is None:
        N = 1
        c.log('WARNING: could not parse repetition count; default 1')
    c.log('instruction:', c.instruction, '| color', color, 'N', N, '| state', np.round(c.state, 4).tolist())
    c.log('images:', {k: getattr(v, 'shape', None) for k, v in c.images.items()})
    c.R_ref = c.R.copy()
    home = c.pos
    c.log('tool z-axis', np.round(c.R[:, 2], 3).tolist())

    front = c.images.get(('front', 'rgb'))
    if front is None:
        c.log('WARNING: no front image in observe reply; keys', list(c.images.keys()))
        c.hold(1, -1.0)
        front = c.images.get(('front', 'rgb'))
        if front is None:
            front = next((v for (cam, k), v in c.images.items() if k == 'rgb'), None)
    elev = c.front_elev()
    det = vision.front_detect(front, color, elev)
    try:
        from PIL import Image
        Image.fromarray(front).save(OUT / 'front0.png')
        if ('wrist', 'rgb') in c.images:
            Image.fromarray(c.images[('wrist', 'rgb')]).save(OUT / 'wrist0.png')
    except Exception:
        pass
    c.log('front det:', {k: (None if det[k] is None else (round(det[k]['u'], 1), round(det[k]['v'], 1), det[k]['area']))
                         for k in ('cube', 'target', 'button')}, 'elev', None if elev is None else int(elev.sum()))

    sel = []; tags = []
    for k in ('cube', 'target', 'button'):
        b = det[k]
        if b is None:
            continue
        for uv in vision.interior_pixels(b, 3):
            sel.append(uv); tags.append(k)
    for k in ('cube', 'target'):
        b = det[k]
        if b is not None:
            rad = max(b['umax'] - b['umin'], b['vmax'] - b['vmin']) * 0.5 + 8
            for uv in c.table_ring(b['u'], b['v'], rad, 7):
                sel.append(uv); tags.append('table')
    pts = c.points('front', sel[:32], radius=1)
    P = {}
    for t, p in zip(tags, pts):
        if p is not None:
            P.setdefault(t, []).append(p['p'])
    P = {k: np.median(np.array(v), 0) for k, v in P.items()}
    # table height robust
    table_z = None
    if 'table' in P:
        tz = [p['p'][2] for t, p in zip(tags, pts) if t == 'table' and p is not None]
        table_z = float(np.median(tz))
    c.log('front points:', {k: np.round(v, 4).tolist() for k, v in P.items()}, 'table_z', table_z)
    if table_z is None:
        # fall back: the lowest of object points minus a little
        zs = [v[2] for v in P.values()]
        table_z = (min(zs) - 0.01) if zs else home[2] - 0.3
        c.log('WARNING table_z fallback', table_z)

    target = P.get('target')
    cube = P.get('cube')
    button = P.get('button')

    # ---------- TARGET_REF ----------
    if target is not None:
        try:
            c.move_to([target[0], target[1], table_z + HOVER_TGT_H], V_FAST, tol=0.01)
            c.observe()
            r = c.wrist_find('target', color, target, gate=0.06)
            if r is not None:
                c.log('target refine', np.round(target, 4).tolist(), '->', np.round(r[0], 4).tolist())
                target = r[0]
            # second look directly above refined point
            c.move_to([target[0], target[1], table_z + HOVER_TGT_H - 0.04], V_FAST, tol=0.008)
            c.observe()
            r = c.wrist_find('target', color, target, gate=0.04)
            if r is not None:
                c.log('target refine2 ->', np.round(r[0], 4).tolist())
                target = r[0]
        except EpisodeEnded:
            raise
        except Exception:
            c.log('target refine error', traceback.format_exc()[-500:])
    else:
        c.log('WARNING: target not found in front view')
    target_z = target[2] if target is not None else table_z

    # ---------- main loop ----------
    count = 0
    fails = 0
    cube_h = None
    cube_yaw = None
    have_fresh_cube = False
    cycles = 0
    while count < N and cycles < N + MAX_GRASP_FAILS + 4:
        cycles += 1
        if time.time() - c.t0 > WALL_LIMIT:
            c.log('wall guard hit; stopping cycles'); break
        try:
            # ----- CUBE_REF -----
            if not have_fresh_cube or cube is None:
                if cube is None:
                    # re-detect in front view
                    c.move_to([home[0], home[1], max(home[2], table_z + 0.25)], V_FAST, tol=0.02)
                    c.observe()
                    d2 = vision.front_detect(c.images.get(('front', 'rgb')), color, None)
                    if d2['cube'] is not None:
                        q = c.points('front', vision.interior_pixels(d2['cube'], 3), 1)
                        q = [x['p'] for x in q if x is not None]
                        if q:
                            cube = np.median(np.array(q), 0)
                    if cube is None:
                        c.log('cube not found; retry'); fails += 1
                        if fails > MAX_GRASP_FAILS:
                            break
                        continue
                for k in range(2):
                    c.move_to([cube[0], cube[1], table_z + HOVER_CUBE_H - 0.03 * k], V_FAST, tol=0.008, grip=-1.0)
                    c.observe()
                    r = c.wrist_find('cube', color, cube, gate=0.07)
                    if r is not None:
                        c.log('cube refine', k, np.round(cube, 4).tolist(), '->', np.round(r[0], 4).tolist(), 'yaw', r[2])
                        cube = r[0]
                        if r[2] is not None:
                            cube_yaw = r[2]
                    else:
                        c.log('cube refine', k, 'no wrist detection')
            have_fresh_cube = False
            h = cube[2] - table_z
            if 0.012 < h < 0.09:
                cube_h = h if cube_h is None else 0.5 * (cube_h + h) if count == 0 else cube_h
            if cube_h is None:
                cube_h = 0.04
            # ----- GRASP -----
            if cube_yaw is not None and c.rot_enabled:
                dyaw = wrap90(cube_yaw - c.tool_yaw())
                if abs(dyaw) > math.radians(4):
                    c.R_ref = rotz(dyaw) @ c.R_ref
                    c.log('yaw align by %.1f deg' % math.degrees(dyaw))
                cube_yaw = None
            base_z = max(table_z, float(cube[2]) - cube_h) if cube[2] - table_z > 0.5 * cube_h else table_z
            grasp_z = base_z + max(0.5 * cube_h, 0.012)
            c.move_to([cube[0], cube[1], grasp_z + 0.06], V_FAST, tol=0.005, grip=-1.0)
            if c.finger < 0.06:
                c.set_gripper(-1.0)
            c.move_to([cube[0], cube[1], grasp_z], V_SLOW, tol=0.003, max_calls=6)
            f = c.set_gripper(1.0)
            c.log('closed fingers %.4f at z %.4f (grasp_z %.4f cube_h %.4f)' % (f, c.pos[2], grasp_z, cube_h))
            c.move_to([cube[0], cube[1], table_z + LIFT_H], V_FAST * 0.8, tol=0.008)
            f2 = c.finger
            c.observe()
            wf = c.wrist_fraction(color)
            held = f2 > 0.008 and (wf is None or wf > 0.02 or f2 > 0.6 * cube_h)
            c.log('lift check fingers %.4f wrist_frac %s held %s' % (f2, None if wf is None else round(wf, 3), held))
            if not held:
                fails += 1
                c.set_gripper(-1.0)
                c.move_to([cube[0], cube[1], table_z + HOVER_CUBE_H], V_FAST, tol=0.01)
                cube = None if fails % 2 == 0 else cube
                if fails > MAX_GRASP_FAILS:
                    c.log('too many grasp failures'); break
                continue
            # ----- PLACE -----
            tz = target if target is not None else cube
            c.move_to([tz[0], tz[1], table_z + LIFT_H], V_FAST * 0.8, tol=0.006)
            place_z = target_z + 0.5 * cube_h + PLACE_DROP
            c.move_to([tz[0], tz[1], place_z + 0.04], V_FAST * 0.8, tol=0.005)
            c.move_to([tz[0], tz[1], place_z], V_SLOW, tol=0.003, max_calls=6)
            c.set_gripper(-1.0)
            c.move_to([tz[0], tz[1], table_z + cube_h + RISE_AFTER_PLACE], V_FAST * 0.6, tol=0.008)
            c.observe()
            try:
                if cycles <= 12 and ('wrist', 'rgb') in c.images:
                    from PIL import Image
                    Image.fromarray(c.images[('wrist', 'rgb')]).save(OUT / ('place_%02d.png' % cycles))
            except Exception:
                pass
            r = c.wrist_find('cube', color, tz, gate=0.10)
            on = True
            if r is not None:
                cube = r[0]
                cube_yaw = r[2]
                dist = float(np.linalg.norm(cube[:2] - tz[:2]))
                on = dist < ON_TARGET_TOL and (cube[2] - target_z) < 0.10
                c.log('place check: cube %s dist %.4f top_z %.4f on=%s' % (np.round(cube, 4).tolist(), dist, cube[2], on))
                have_fresh_cube = True
            else:
                c.log('place check: cube not seen in wrist; assume placed')
                cube = np.array([tz[0], tz[1], target_z + cube_h])
                have_fresh_cube = True
            if on:
                count += 1
                c.log('=== placement %d / %d done (steps %d)' % (count, N, c.n_steps_sent))
            else:
                c.log('placement not on target; not counted')
                have_fresh_cube = False
        except EpisodeEnded:
            raise
        except Exception:
            c.log('cycle error', traceback.format_exc()[-800:])
            try:
                c.observe()
            except Exception:
                pass
            fails += 1
            if fails > MAX_GRASP_FAILS + 2:
                break

    c.log('placements done: %d / %d' % (count, N))
    # ---------- BUTTON ----------
    press_button(c, button, table_z, color)


def press_button(c, button, table_z, color):
    if button is None:
        c.log('WARNING: button unknown; re-detect from home view')
        c.move_to([c.pos[0], c.pos[1], table_z + 0.30], V_FAST, tol=0.02)
        c.observe()
        d = vision.front_detect(c.images.get(('front', 'rgb')), color, None)
        if d['button'] is not None:
            q = c.points('front', vision.interior_pixels(d['button'], 3), 1)
            q = [x['p'] for x in q if x is not None]
            if q:
                button = np.median(np.array(q), 0)
        if button is None:
            c.log('button not found; giving up'); return
    # lift clear first, close gripper
    c.move_to([c.pos[0], c.pos[1], table_z + LIFT_H], V_FAST, tol=0.01)
    c.set_gripper(1.0)
    c.move_to([button[0], button[1], table_z + 0.15], V_FAST, tol=0.008)
    c.observe()
    r = c.wrist_find('button', color, button, gate=0.05)
    if r is not None:
        c.log('button refine', np.round(button, 4).tolist(), '->', np.round(r[0], 4).tolist())
        button = r[0]
    c.move_to([button[0], button[1], table_z + 0.10], V_FAST, tol=0.006)
    c.observe()
    r = c.wrist_find('button', color, button, gate=0.03)
    if r is not None:
        c.log('button refine2 ->', np.round(r[0], 4).tolist())
        button = r[0]
    top = float(button[2])
    offsets = [(0, 0), (0.006, 0), (-0.006, 0), (0, 0.006), (0, -0.006), (0.01, 0.01), (-0.01, -0.01)]
    for k, (ox, oy) in enumerate(offsets):
        x, y = button[0] + ox, button[1] + oy
        c.move_to([x, y, top + 0.04], V_FAST * 0.6, tol=0.004, grip=1.0)
        # descend in short chunks, watching for the episode to end
        z_goal = top - 0.025 - 0.005 * k
        stall = 0
        while c.pos[2] > z_goal + 0.002:
            z0 = c.pos[2]
            rows = [[0, 0, -0.10, 0, 0, 0, 1.0]] * 2
            c.send(rows)
            if z0 - c.pos[2] < 0.001:
                stall += 1
                if stall >= 4:
                    break
            else:
                stall = 0
        c.log('press %d reached z %.4f (top %.4f) stall %d' % (k, c.pos[2], top, stall))
        c.hold(4, 1.0)
        c.move_to([x, y, top + 0.05], V_FAST * 0.6, tol=0.006)
        if time.time() - c.t0 > 1150:
            break
    c.log('button attempts exhausted')
