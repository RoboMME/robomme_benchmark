"""VideoPlaceOrder controller (RAX main(ctx) entrypoint).

1. Read the instruction and the conditioning video (frame_sequence manifest).
2. Vision (vision.py): slots, shuffle steps (target identity permutation), placements of the goal cube;
   the identity of the k-th placement -> its slot at the end of the video -> pixel in the front view.
3. Localise the goal cube and the answer target in 3-D (pixel_to_position on the current front view,
   own depth back-projection as fallback), then a top-down pick and place.
Never raises: every stage is guarded and logged to stdout and /workspace/output/vpo_log.txt.
"""
import base64
import io
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vision  # noqa: E402

WS = Path('/workspace')
OUT = WS / 'output'
T_START = time.time()
LOGF = None


def log(*a):
    global LOGF
    msg = f"[{time.time() - T_START:7.1f}s] " + ' '.join(str(x) for x in a)
    print(msg, flush=True)
    try:
        if LOGF is None:
            OUT.mkdir(parents=True, exist_ok=True)
            LOGF = open(OUT / 'vpo_log.txt', 'a')
        LOGF.write(msg + '\n'); LOGF.flush()
    except Exception:
        pass


# ----------------------------------------------------------------------------------- reply parsing
def reply_json(reply):
    """First JSON text block of an MCP reply (dict) or {}."""
    try:
        for b in reply.get('content', []):
            if b.get('type') == 'text':
                try:
                    return json.loads(b.get('text', ''))
                except Exception:
                    continue
    except Exception:
        pass
    sc = reply.get('structuredContent') if isinstance(reply, dict) else None
    return sc if isinstance(sc, dict) else {}


def deep_find(obj, key, depth=0):
    """First value stored under `key` anywhere in a nested dict/list (breadth-first-ish)."""
    if depth > 8:
        return None
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = deep_find(v, key, depth + 1)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj[:50]:
            r = deep_find(v, key, depth + 1)
            if r is not None:
                return r
    return None


def find_all(obj, key, depth=0, out=None):
    """All (container_dict, value) pairs for `key` in a nested structure, in traversal order."""
    if out is None:
        out = []
    if depth > 8:
        return out
    if isinstance(obj, dict):
        if key in obj:
            out.append((obj, obj[key]))
        for k, v in obj.items():
            if k != key:
                find_all(v, key, depth + 1, out)
    elif isinstance(obj, list):
        for v in obj[:400]:
            find_all(v, key, depth + 1, out)
    return out


def _as_state(s):
    if isinstance(s, dict):
        for k in ('values', 'vector', 'data'):
            if isinstance(s.get(k), list):
                s = s[k]
                break
    if isinstance(s, list) and len(s) >= 8 and all(isinstance(x, (int, float)) for x in s[:8]):
        return np.array(s[:8], float)
    return None


def get_state(j):
    """Current robot state: top-level / observation-level first, else the entry with the largest step_id."""
    if isinstance(j, dict):
        for path in (('state',), ('observation', 'state'), ('obs', 'state')):
            o = j
            for p in path:
                o = o.get(p) if isinstance(o, dict) else None
            st = _as_state(o)
            if st is not None:
                return st
    best = None
    for cont, v in find_all(j, 'state'):
        st = _as_state(v)
        if st is None:
            continue
        sid = cont.get('step_id') if isinstance(cont.get('step_id'), int) else -1
        if best is None or sid >= best[0]:
            best = (sid, st)
    return best[1] if best else None


def get_step_id(j):
    if isinstance(j, dict):
        for path in (('step_id',), ('observation', 'step_id')):
            o = j
            for p in path:
                o = o.get(p) if isinstance(o, dict) else None
            if isinstance(o, int):
                return o
    ids = [v for _c, v in find_all(j, 'step_id') if isinstance(v, int)]
    return max(ids) if ids else None


def episode_over(j):
    for k in ('episode_ended', 'terminated', 'done', 'task_complete', 'truncated'):
        v = deep_find(j, k)
        if v is True:
            return True
    return False


def summarize(j, n=600):
    try:
        s = json.dumps(j, default=str)
    except Exception:
        s = str(j)
    return s[:n]


# ----------------------------------------------------------------------------------- robot wrapper
class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.state = None
        self.step_id = None
        self.ended = False
        self.steps_used = 0
        self.step_limit = None
        self.last_j = {}
        self.R0 = None

    def status(self):
        try:
            j = reply_json(self.ctx.call('robomme_status'))
            log('status:', summarize(j, 1500))
            for k in ('step_limit', 'max_steps', 'episode_step_limit', 'limit'):
                v = deep_find(j, k)
                if isinstance(v, (int, float)) and v > 0:
                    self.step_limit = int(v)
                    break
            if episode_over(j):
                self.ended = True
            return j
        except Exception as e:
            log('status failed', e)
            return {}

    def observe(self):
        r = self.ctx.call('robomme_observe')
        j = reply_json(r)
        self._update(j)
        return r, j

    def _update(self, j):
        self.last_j = j
        s = get_state(j)
        if s is not None:
            self.state = s
        sid = get_step_id(j)
        if isinstance(sid, int):
            self.step_id = sid
        if episode_over(j):
            self.ended = True

    def step(self, rows):
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        out = None
        for i in range(0, len(rows), 100):
            if self.ended:
                break
            try:
                r = self.ctx.call('robomme_step', actions=rows[i:i + 100])
            except Exception as e:
                self.fail_count = getattr(self, 'fail_count', 0) + 1
                log(f'robomme_step failed ({self.fail_count}):', str(e)[:500])
                if self.fail_count >= 4:
                    raise
                try:
                    self.observe()
                    self.status()
                except Exception as e2:
                    log('observe/status after failed step failed', e2)
                continue
            j = reply_json(r)
            self.steps_used += len(rows[i:i + 100])
            self._update(j)
            out = (r, j)
            if get_state(j) is None:
                try:
                    self.observe()
                except Exception as e:
                    log('observe after step failed', e)
        return out

    # ------------------------------------------------------------------ orientation hold
    def rot_row(self):
        if self.R0 is None or self.state is None:
            return [0.0, 0.0, 0.0]
        try:
            from scipy.spatial.transform import Rotation as Rot
            Rc = Rot.from_rotvec(self.state[3:6])
            err = (self.R0 * Rc.inv()).as_rotvec()     # world-frame correction
            v = np.clip(err / 0.5, -0.3, 0.3)
            if np.linalg.norm(err) < 0.01:
                v = np.zeros(3)
            return [float(x) for x in v]
        except Exception:
            return [0.0, 0.0, 0.0]

    def move_to(self, target, grip, max_step=0.02, tol=0.004, iters=12, label=''):
        target = np.asarray(target, float)
        for it in range(iters):
            if self.ended or self.state is None:
                return False
            err = target - self.state[:3]
            d = float(np.linalg.norm(err))
            if d < tol:
                return True
            n = int(min(15, max(1, math.ceil(d / max_step))))
            # the adapter targets measured pose + delta at each step; distribute the error over n rows,
            # re-measure after the batch
            per = err / n / 0.05
            rows = []
            rr = self.rot_row()
            for k in range(n):
                rows.append([per[0], per[1], per[2]] + (rr if k == 0 else [0.0, 0.0, 0.0]) + [grip])
            self.step(rows)
        if self.state is not None:
            d = float(np.linalg.norm(target - self.state[:3]))
            log(f'move_to {label}: residual {d * 1000:.1f} mm')
            return d < 3 * tol
        return False

    def hold(self, grip, n):
        rows = [[0.0, 0.0, 0.0] + self.rot_row() + [grip]] + [[0, 0, 0, 0, 0, 0, grip] for _ in range(n - 1)]
        self.step(rows)

    def width(self):
        return float(self.state[6] + self.state[7]) if self.state is not None else None


# ----------------------------------------------------------------------------------- images / geometry
def load_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'))


def load_depth(path, mpu=0.001):
    from PIL import Image
    with Image.open(path) as im:
        a = np.asarray(im).astype(np.float64)
    return a * mpu


def manifest_frames(j):
    fs = deep_find(j, 'frame_sequence')
    mp = None
    if isinstance(fs, dict):
        mp = fs.get('manifest_path')
    if mp is None:
        mp = deep_find(j, 'manifest_path')
    if not mp:
        return None, None
    p = Path(mp)
    if not p.is_absolute():
        p = WS / p
    man = json.loads(p.read_text())
    return man, p


def frame_files(fr):
    files = {}
    for f in fr.get('files', []):
        files[(f.get('camera'), f.get('kind'))] = f
    return files


def decode_images(reply):
    """RGB image blocks of a reply -> list of numpy arrays (depth-like 16-bit / grey images skipped)."""
    out = []
    try:
        from PIL import Image
        for b in reply.get('content', []):
            if b.get('type') == 'image' and b.get('data'):
                im = Image.open(io.BytesIO(base64.b64decode(b['data'])))
                if im.mode not in ('RGB', 'RGBA', 'P'):
                    continue
                a = np.asarray(im.convert('RGB'))
                if a.shape[:2] != (256, 256):
                    continue
                sat = (a.max(2).astype(int) - a.min(2)).mean()
                if sat < 6:          # grey -> probably a depth visualisation
                    continue
                out.append(a)
    except Exception as e:
        log('decode_images failed', e)
    return out


REF_FRONT = None


def split_front_wrist(ims):
    """Front = the image most similar to the reference (step-0) front view."""
    if not ims:
        return None, None
    if len(ims) == 1:
        return ims[0], None
    if REF_FRONT is None:
        return ims[0], ims[1]
    ref = REF_FRONT[::4, ::4].astype(float)
    d = [np.abs(a[::4, ::4].astype(float) - ref).mean() for a in ims]
    i = int(np.argmin(d))
    rest = [a for k, a in enumerate(ims) if k != i]
    return ims[i], rest[0]


def current_images(robot, reply, j):
    """(front, wrist, front_depth or None) of the current observation."""
    front = wrist = depth = None
    try:
        man, _ = manifest_frames(j)
        if man and man.get('frames'):
            fr = sorted(man['frames'], key=lambda f: f.get('frame_index', 0))[-1]
            ff = frame_files(fr)
            fsid = fr.get('step_id')
            if isinstance(fsid, int) and robot.step_id is not None and fsid != robot.step_id:
                log(f'manifest last frame step {fsid} != current step {robot.step_id}; using inline images')
                ff = {}
            if ('front', 'rgb') in ff:
                front = load_rgb(WS / ff[('front', 'rgb')]['path'])
            if ('wrist', 'rgb') in ff:
                wrist = load_rgb(WS / ff[('wrist', 'rgb')]['path'])
            if ('front', 'depth') in ff:
                depth = load_depth(WS / ff[('front', 'depth')]['path'], ff[('front', 'depth')].get('meters_per_unit', 0.001))
    except Exception as e:
        log('manifest images failed', e)
    if front is None or wrist is None:
        f2, w2 = split_front_wrist(decode_images(reply))
        front = front if front is not None else f2
        wrist = wrist if wrist is not None else w2
    return front, wrist, depth


def parse_points(j, n):
    """Extract n world points from a pixel_to_position reply."""
    cands = None
    for key in ('results', 'points', 'positions', 'selections', 'world_points'):
        v = deep_find(j, key)
        if isinstance(v, list) and len(v) == n:
            cands = v
            break
    if cands is None:
        # any list of length n
        def walk(o, d=0):
            if d > 6:
                return None
            if isinstance(o, list) and len(o) == n and all(isinstance(x, dict) for x in o):
                return o
            if isinstance(o, dict):
                for v in o.values():
                    r = walk(v, d + 1)
                    if r is not None:
                        return r
            return None
        cands = walk(j)
    pts = []
    for c in (cands or []):
        p = None
        valid = True
        if isinstance(c, dict):
            if c.get('valid') is False:
                valid = False
            def as3(v):
                if isinstance(v, dict):
                    v = [v.get('x'), v.get('y'), v.get('z')]
                if isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
                    return np.array(v, float)
                return None
            for k in ('world_position', 'position_world', 'world_point', 'world', 'position', 'position_m',
                      'point', 'xyz'):
                p = as3(c.get(k))
                if p is not None:
                    break
            if p is None:
                keys = sorted(c.keys(), key=lambda k: (0 if 'world' in k else 1 if 'pos' in k else 2))
                for k in keys:
                    if 'camera' in k or 'pixel' in k or k in ('uv', 'u', 'v'):
                        continue
                    p = as3(c.get(k))
                    if p is not None:
                        break
        elif isinstance(c, list) and len(c) == 3:
            p = np.array(c, float)
        pts.append(p if (valid and p is not None and np.all(np.isfinite(p))) else None)
    while len(pts) < n:
        pts.append(None)
    return pts


def pix2world(robot, cam, uvs, radius=1):
    """World points for pixels of the current observation via the tool."""
    if not uvs:
        return []
    try:
        sel = [{'camera': cam, 'u': float(u), 'v': float(v)} for u, v in uvs]
        r = robot.ctx.call('robomme_pixel_to_position', observation_step_id=int(robot.step_id or 0),
                           selections=sel, depth_window_radius=int(radius))
        j = reply_json(r)
        pts = parse_points(j, len(uvs))
        if all(p is None for p in pts):
            log('pixel_to_position: no points parsed:', summarize(j, 800))
        return pts
    except Exception as e:
        log('pixel_to_position failed', e)
        return [None] * len(uvs)


def own_backproject(j, cam, depth, uvs):
    """Fallback: OpenGL camera pose (wxyz) + intrinsics + linear depth."""
    try:
        K = deep_find(j, 'camera_intrinsics')
        P = deep_find(j, 'camera_poses')
        if isinstance(K, dict):
            K = K.get(cam)
        if isinstance(P, dict):
            P = P.get(cam)
        if isinstance(K, dict):
            if 'fx' in K:
                K = [[K['fx'], 0, K['cx']], [0, K['fy'], K['cy']], [0, 0, 1]]
            else:
                K = K.get('K') or K.get('matrix') or K.get('intrinsics')
        K = np.array(K, float).reshape(3, 3)
        if isinstance(P, list) and len(P) == 7:
            P = {'position': P[:3], 'quaternion_wxyz': P[3:]}
        pos = np.array(P.get('position_m', P.get('position')), float)
        q = P.get('quaternion_wxyz') or P.get('orientation')
        w, x, y, z = [float(v) for v in q]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        out = []
        for u, v in uvs:
            ui, vi = int(round(u)), int(round(v))
            win = depth[max(0, vi - 1):vi + 2, max(0, ui - 1):ui + 2]
            win = win[win > 0]
            if not len(win):
                out.append(None); continue
            d = float(np.median(win))
            xc = (u - K[0, 2]) / K[0, 0] * d
            yc = -(v - K[1, 2]) / K[1, 1] * d
            out.append(R @ np.array([xc, yc, -d]) + pos)
        return out
    except Exception as e:
        log('own backprojection failed', e)
        return [None] * len(uvs)


# ----------------------------------------------------------------------------------- perception
def load_video(j0):
    man, mp = manifest_frames(j0)
    if not man:
        return None, None
    frs = sorted(man.get('frames', []), key=lambda f: f.get('frame_index', 0))
    vid, exe = [], None
    for f in frs:
        ff = frame_files(f)
        if ('front', 'rgb') not in ff:
            continue
        try:
            front = load_rgb(WS / ff[('front', 'rgb')]['path'])
            wrist = load_rgb(WS / ff[('wrist', 'rgb')]['path']) if ('wrist', 'rgb') in ff else None
        except Exception as e:
            log('frame load failed', e)
            continue
        if f.get('phase') == 'video_demo':
            vid.append((float(f.get('timestamp_s', len(vid) * 0.05)), front, wrist, f.get('state')))
        elif f.get('phase') == 'execution' and exe is None:
            exe = (float(f.get('timestamp_s', 0.0)), front, wrist, f.get('state'), ff)
    log(f'video: {len(vid)} conditioning frames, execution frame: {exe is not None}, manifest {mp}')
    return vid, exe


def run_vision(vid, exe, color, k):
    frames = [(t, f, w) for t, f, w, _ in vid]
    if exe is not None:
        t_last = frames[-1][0] if frames else 0.0
        frames.append((t_last + 0.05, exe[1], exe[2]))
    res = None
    try:
        res = vision.analyze_video2(frames, color, k, log=log)
    except Exception:
        log('analyze_video2 failed:\n' + traceback.format_exc())
    if res is None or res.get('answer_slot') is None:
        try:
            log('falling back to v1 pair-swap analysis')
            res = vision.analyze_video(frames, color, k, log=log)
        except Exception:
            log('analyze_video failed:\n' + traceback.format_exc())
    return res


def pick_target_pixel(res, front):
    blobs = [b for b in vision.target_blobs(front, 40)]
    if res is not None and res.get('answer_xy') is not None:
        ax = np.array(res['answer_xy'])
        if blobs:
            d = [np.hypot(b[0] - ax[0], b[1] - ax[1]) for b in blobs]
            j = int(np.argmin(d))
            if d[j] < 12:
                return (blobs[j][0], blobs[j][1]), blobs
        return (float(ax[0]), float(ax[1])), blobs
    if blobs:
        b = max(blobs, key=lambda b: b[2])
        return (b[0], b[1]), blobs
    return None, blobs


# ----------------------------------------------------------------------------------- task logic
def table_points(front, center, color, radius=16):
    """Pixels around a centre that look like bare table (not target, not cube, not arm)."""
    occ = vision.occluder_mask(front)
    tm = vision.target_mask(front)
    pts = []
    for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        for r in (radius, radius + 8):
            u = center[0] + r * math.cos(ang); v = center[1] + r * math.sin(ang)
            ui, vi = int(round(u)), int(round(v))
            if 2 <= ui < 254 and 2 <= vi < 254 and occ[vi, ui] < 0.05 and tm[max(0, vi - 2):vi + 3, max(0, ui - 2):ui + 3].sum() == 0:
                pts.append((u, v))
                break
    return pts[:10]


def locate(robot, reply, j, front, depth, pix_list):
    pts = pix2world(robot, 'front', pix_list)
    own = own_backproject(j, 'front', depth, pix_list) if depth is not None else [None] * len(pix_list)
    try:
        diffs = [float(np.linalg.norm(p - o)) for p, o in zip(pts, own) if p is not None and o is not None]
        if diffs:
            log(f'tool vs own backprojection: median diff {np.median(diffs) * 1000:.1f} mm over {len(diffs)} pts')
    except Exception:
        pass
    pts = [p if p is not None else o for p, o in zip(pts, own)]
    return pts


def cube_pixel(front, color):
    cb = vision.cube_blob(front, color)
    if cb is None:
        return None
    return (cb[3], cb[4]), cb[2]


def wrist_refine(robot, color, guess_xy, table_z):
    """Observe; locate the cube in the wrist view; return world xy (top face) or None."""
    try:
        r, j = robot.observe()
        _f, wrist, _d = current_images(robot, r, j)
        if wrist is None:
            return None
        cb = vision.cube_blob(wrist, color, min_area=40)
        if cb is None:
            return None
        pts = pix2world(robot, 'wrist', [(cb[3], cb[4])], radius=2)
        p = pts[0]
        if p is None:
            return None
        if np.linalg.norm(p[:2] - guess_xy) > 0.05 or p[2] < table_z - 0.01:
            log(f'wrist refine rejected: {p} vs guess {guess_xy}')
            return None
        return p
    except Exception as e:
        log('wrist refine failed', e)
        return None


def pick_and_place(robot, color, target_pix, rinfo):
    r, j = robot.observe()
    front, wrist, depth = current_images(robot, r, j)
    if front is None:
        front = rinfo.get('exec_front')
    if depth is None:
        depth = rinfo.get('exec_depth')
    cp = cube_pixel(front, color)
    if cp is None:
        log('cube not found in front view')
        return False
    (cu, cv), carea = cp
    tps = table_points(front, (cu, cv), color)
    pix = [(cu, cv), target_pix] + tps
    pts = locate(robot, r, j, front, depth, pix)
    cube_p, targ_p = pts[0], pts[1]
    tab = [p for p in pts[2:] if p is not None]
    log(f'cube pix ({cu:.1f},{cv:.1f}) area {carea} -> {cube_p}; target pix {target_pix} -> {targ_p}; table pts {len(tab)}')
    if cube_p is None or targ_p is None:
        log('localisation failed')
        return False
    table_z = float(np.median([p[2] for p in tab])) if tab else float(targ_p[2]) - 0.002
    h_c = float(np.clip(cube_p[2] - table_z, 0.015, 0.06))
    log(f'table z {table_z:.4f}, cube visible-surface height {h_c:.4f}, state {np.round(robot.state, 4).tolist()}')
    # the front camera sees the top and the front face: shift the point half a face away from the camera
    cube_xy = np.array(cube_p[:2])
    try:
        P = deep_find(j, 'camera_poses')
        if isinstance(P, dict):
            P = P.get('front')
        cam = np.array(P.get('position'), float)
        dvec = cube_xy - cam[:2]
        dvec /= max(np.linalg.norm(dvec), 1e-6)
        cube_xy = cube_xy + dvec * 0.25 * h_c
    except Exception:
        pass
    grasp_dz = float(np.clip(0.5 * h_c, 0.01, 0.022))
    z_safe = table_z + 0.14
    attempts = 0
    while attempts < 3 and not robot.ended:
        attempts += 1
        log(f'grasp attempt {attempts}: cube xy {cube_xy}')
        robot.hold(-1.0, 4)
        if robot.state is not None and robot.state[2] < z_safe - 0.02:
            robot.move_to([robot.state[0], robot.state[1], z_safe], -1.0, label='raise')
        robot.move_to([cube_xy[0], cube_xy[1], z_safe], -1.0, max_step=0.025, label='over cube')
        robot.move_to([cube_xy[0], cube_xy[1], table_z + 0.08], -1.0, label='pre-grasp')
        p = wrist_refine(robot, color, cube_xy, table_z)
        if p is not None:
            log(f'wrist refine: {p[:2]} (was {cube_xy})')
            cube_xy = 0.3 * cube_xy + 0.7 * np.array(p[:2])
            robot.move_to([cube_xy[0], cube_xy[1], table_z + 0.08], -1.0, label='pre-grasp2')
        robot.move_to([cube_xy[0], cube_xy[1], table_z + grasp_dz + 0.03], -1.0, max_step=0.012, label='approach')
        robot.move_to([cube_xy[0], cube_xy[1], table_z + grasp_dz], -1.0, max_step=0.008, tol=0.003, label='grasp')
        robot.hold(1.0, 12)
        w = robot.width()
        log(f'closed: width {w}, z {robot.state[2] if robot.state is not None else None}')
        robot.move_to([robot.state[0], robot.state[1], z_safe], 1.0, max_step=0.02, label='lift')
        robot.hold(1.0, 2)
        w2 = robot.width()
        log(f'after lift: width {w2}')
        if w2 is not None and w2 > 0.006:
            break
        log('grasp missed; relocalising cube')
        robot.hold(-1.0, 6)
        r, j = robot.observe()
        front2, _w, _d = current_images(robot, r, j)
        if front2 is not None:
            cp = cube_pixel(front2, color)
            if cp is not None:
                pts = locate(robot, r, j, front2, _d, [cp[0]])
                if pts[0] is not None:
                    cube_xy = np.array(pts[0][:2])
    else:
        if robot.ended:
            return False
    # place
    tx, ty = float(targ_p[0]), float(targ_p[1])
    place_z = max(float(targ_p[2]), table_z) + grasp_dz + 0.012
    log(f'placing at ({tx:.4f},{ty:.4f}) z {place_z:.4f}')
    robot.move_to([tx, ty, z_safe], 1.0, max_step=0.025, label='over target')
    robot.move_to([tx, ty, place_z + 0.04], 1.0, max_step=0.015, label='pre-place')
    robot.move_to([tx, ty, place_z], 1.0, max_step=0.008, tol=0.003, label='place')
    robot.hold(1.0, 3)
    robot.hold(-1.0, 10)
    log(f'released: ended={robot.ended} state {None if robot.state is None else np.round(robot.state, 4).tolist()}')
    if not robot.ended:
        robot.move_to([tx, ty, place_z + 0.08], -1.0, label='retreat')
        robot.hold(-1.0, 5)
    log(f'final: ended={robot.ended} reply {summarize(robot.last_j, 800)}')
    return True


def main(ctx):
    try:
        _main(ctx)
    except Exception:
        log('FATAL:\n' + traceback.format_exc())


def _main(ctx):
    global T_START, REF_FRONT
    T_START = time.time()
    log('VPO controller start; vision', getattr(vision, 'VERSION', '?'))
    robot = Robot(ctx)
    robot.status()
    r0, j0 = robot.observe()
    log('observe0:', summarize(j0, 1200))
    instr = deep_find(j0, 'instruction') or ''
    color, k = vision.parse_instruction(instr if isinstance(instr, str) else json.dumps(instr))
    log(f'instruction: {instr!r} -> color {color}, k {k}')
    if color is None:
        color = 'red'
    if robot.state is not None:
        try:
            from scipy.spatial.transform import Rotation as Rot
            robot.R0 = Rot.from_rotvec(robot.state[3:6])
        except Exception:
            pass
    vid, exe = load_video(j0)
    rinfo = {}
    res = None
    if vid:
        res = run_vision(vid, exe, color, k)
    else:
        log('no conditioning video found')
    exec_front = exe[1] if exe is not None else None
    exec_depth = None
    if exe is not None:
        ff = exe[4]
        try:
            if ('front', 'depth') in ff:
                exec_depth = load_depth(WS / ff[('front', 'depth')]['path'], ff[('front', 'depth')].get('meters_per_unit', 0.001))
        except Exception as e:
            log('exec depth load failed', e)
    REF_FRONT = exec_front
    front, wrist, depth = current_images(robot, r0, j0)
    if front is None:
        front = exec_front
    REF_FRONT = front
    rinfo['exec_front'] = exec_front
    rinfo['exec_depth'] = exec_depth if depth is None else depth
    if front is None:
        log('no front image; aborting')
        return
    tpix, blobs = pick_target_pixel(res, front)
    log(f'targets in current view: {[(round(b[0]), round(b[1]), b[2]) for b in blobs]}; chosen target pixel {tpix}')
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / 'vision_result.json', 'w') as f:
            json.dump({'color': color, 'k': k, 'instruction': instr, 'target_pix': tpix,
                       'res': {kk: vv for kk, vv in (res or {}).items() if kk in ('slots', 'final_perm', 'answer_slot', 'answer_xy', 'k_used', 'answer_ident')},
                       'steps': [{'t0': w['t0'], 'D': w['D'], 'moves': [m[:2] for m in w['moves']]} for w in (res or {}).get('steps', [])],
                       'events': (res or {}).get('events', [])}, f, default=str)
    except Exception as e:
        log('saving vision result failed', e)
    if tpix is None:
        log('no target pixel; aborting')
        return
    ok = pick_and_place(robot, color, tpix, rinfo)
    log(f'pick_and_place returned {ok}; steps used {robot.steps_used}')
    robot.status()
