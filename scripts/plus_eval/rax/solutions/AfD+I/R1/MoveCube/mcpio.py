"""Defensive MCP packet parsing for RoboMME-Plus observations (unknown exact schema)."""
import base64, io, json, os, time
import numpy as np
from PIL import Image

WS = '/workspace'
OUT = os.path.join(WS, 'output')
_LOG = None
T0 = time.time()


def log(*a):
    global _LOG
    s = '[%7.1f] ' % (time.time() - T0) + ' '.join(str(x) for x in a)
    try:
        print(s, flush=True)
    except Exception:
        pass
    try:
        if _LOG is None:
            os.makedirs(OUT, exist_ok=True)
            _LOG = open(os.path.join(OUT, 'controller_log.txt'), 'a')
        _LOG.write(s + '\n'); _LOG.flush()
    except Exception:
        pass


def dump_json(name, obj):
    try:
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, name), 'w') as f:
            json.dump(obj, f, indent=1, default=str)
    except Exception as e:
        log('dump fail', name, e)


def split_result(result):
    """MCP result -> (meta dict, list of image blocks)."""
    meta = {}
    imgs = []
    for b in (result or {}).get('content', []) or []:
        t = b.get('type')
        if t == 'text' and not meta:
            try:
                meta = json.loads(b.get('text', '') or '{}')
                if not isinstance(meta, dict):
                    meta = {'value': meta}
            except Exception:
                meta = {'text': b.get('text', '')}
        elif t == 'image':
            imgs.append(b)
    sc = (result or {}).get('structuredContent')
    if isinstance(sc, dict) and not meta:
        meta = sc
    return meta, imgs


def strip_big(o, depth=0):
    """Copy of a JSON object without long strings (image data) for logging."""
    if depth > 12:
        return '...'
    if isinstance(o, dict):
        return {k: strip_big(v, depth + 1) for k, v in o.items()}
    if isinstance(o, list):
        if len(o) > 40:
            return [strip_big(x, depth + 1) for x in o[:6]] + ['... %d items' % len(o)]
        return [strip_big(x, depth + 1) for x in o]
    if isinstance(o, str) and len(o) > 300:
        return o[:80] + '...<%d chars>' % len(o)
    return o


def find_key(o, key, depth=0):
    """First value for key anywhere in nested JSON (breadth-first)."""
    if depth > 10:
        return None
    if isinstance(o, dict):
        if key in o:
            return o[key]
        for v in o.values():
            r = find_key(v, key, depth + 1)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find_key(v, key, depth + 1)
            if r is not None:
                return r
    return None


def find_all(o, key, out=None, depth=0):
    if out is None:
        out = []
    if depth > 10:
        return out
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key:
                out.append(v)
            find_all(v, key, out, depth + 1)
    elif isinstance(o, list):
        for v in o:
            find_all(v, key, out, depth + 1)
    return out


def as_state(v):
    try:
        a = np.asarray(v, float).reshape(-1)
        if a.size >= 8:
            return a[:8]
    except Exception:
        pass
    if isinstance(v, dict):
        for k in ('values', 'vector', 'state'):
            if k in v:
                return as_state(v[k])
        try:
            p = v.get('ee_position') or v.get('position')
            o = v.get('ee_axis_angle') or v.get('orientation') or v.get('axis_angle')
            g = v.get('gripper') or v.get('fingers') or v.get('finger_positions')
            return np.asarray(list(p) + list(o) + list(g), float)[:8]
        except Exception:
            pass
    return None


def load_png(path):
    p = path if os.path.isabs(path) else os.path.join(WS, path)
    im = Image.open(p)
    im.load()
    return im


def decode_block(b):
    data = base64.b64decode(b.get('data', ''))
    im = Image.open(io.BytesIO(data)); im.load()
    return im


def parse_intrinsics(v):
    """-> 3x3 K or None."""
    if v is None:
        return None
    if isinstance(v, dict):
        if all(k in v for k in ('fx', 'fy', 'cx', 'cy')):
            return np.array([[v['fx'], 0, v['cx']], [0, v['fy'], v['cy']], [0, 0, 1]], float)
        for k in ('K', 'matrix', 'intrinsic_matrix', 'intrinsics', 'camera_matrix', 'k'):
            if k in v:
                return parse_intrinsics(v[k])
        return None
    try:
        a = np.asarray(v, float)
        if a.size == 9:
            return a.reshape(3, 3)
        if a.size == 4:
            return np.array([[a[0], 0, a[2]], [0, a[1], a[3]], [0, 0, 1]], float)
        if a.size == 12:
            return a.reshape(3, 4)[:, :3]
        if a.size == 16:
            return a.reshape(4, 4)[:3, :3]
    except Exception:
        pass
    return None


def quat_wxyz_to_R(q):
    w, x, y, z = [float(t) for t in q]
    n = np.sqrt(w * w + x * x + y * y + z * z) + 1e-12
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def parse_pose(v):
    """-> dict(R, t, q, convention) or None. R,t: camera->world (as published)."""
    if v is None:
        return None
    if isinstance(v, dict):
        conv = str(v.get('convention', v.get('frame_convention', 'opengl')))
        pos = v.get('position_m', v.get('position', v.get('p', v.get('translation', v.get('xyz')))))
        q = v.get('quaternion_wxyz', v.get('wxyz', v.get('quaternion', v.get('q', v.get('orientation')))))
        if pos is not None and q is not None:
            q = np.asarray(q, float).reshape(-1)
            if q.size == 4:
                return dict(R=quat_wxyz_to_R(q), t=np.asarray(pos, float).reshape(3), q=q, convention=conv)
        for k in ('matrix', 'pose', 'transform', 'camera_to_world', 'cam2world', 'extrinsic', 'T'):
            if k in v:
                r = parse_pose(v[k])
                if r is not None:
                    r['convention'] = conv
                    return r
        return None
    try:
        a = np.asarray(v, float)
        if a.size == 16:
            M = a.reshape(4, 4)
            return dict(R=M[:3, :3], t=M[:3, 3], q=None, convention='opengl')
        if a.size == 12:
            M = a.reshape(3, 4)
            return dict(R=M[:, :3], t=M[:, 3], q=None, convention='opengl')
        if a.size == 7:
            return dict(R=quat_wxyz_to_R(a[3:]), t=a[:3], q=a[3:], convention='opengl')
    except Exception:
        pass
    return None


def per_camera(v, cam):
    """Pick the entry for camera `cam` from a dict/list container."""
    if v is None:
        return None
    if isinstance(v, dict):
        if cam in v:
            return v[cam]
        for k, x in v.items():
            if cam in str(k).lower():
                return x
        return None
    if isinstance(v, list):
        for x in v:
            if isinstance(x, dict) and str(x.get('camera', x.get('name', ''))).lower().startswith(cam):
                return x
        if len(v) == 2 and cam in ('front', 'wrist'):
            return v[0 if cam == 'front' else 1]
    return None


class Obs:
    def __init__(self):
        self.meta = {}
        self.step_id = None
        self.state = None
        self.rgb = {}
        self.depth = {}
        self.depth_scale = {'front': 0.001, 'wrist': 0.001}
        self.K = {}
        self.pose = {}
        self.ended = False
        self.success = None
        self.manifest = None
        self.instruction = None

    def ok(self, cam='front'):
        return cam in self.rgb


def terminal_flags(meta):
    ended = False; success = None
    for k in ('episode_ended', 'terminated', 'done', 'truncated'):
        v = find_key(meta, k)
        if v is True:
            ended = True
    for k in ('task_complete', 'success', 'is_success'):
        v = find_key(meta, k)
        if v is True:
            success = True
        elif v is False and success is None:
            success = False
    return ended, success


def frames_from_manifest(path):
    try:
        p = path if os.path.isabs(path) else os.path.join(WS, path)
        with open(p) as f:
            man = json.load(f)
        return man
    except Exception as e:
        log('manifest read fail', path, e)
        return None


def frame_files(fr):
    files = {}
    for f in fr.get('files', []) or []:
        cam = str(f.get('camera', '')).lower(); kind = str(f.get('kind', '')).lower()
        files[(cam, kind)] = f
    return files


def fill_from_frame(obs, fr):
    files = frame_files(fr)
    for cam in ('front', 'wrist'):
        try:
            if (cam, 'rgb') in files:
                obs.rgb[cam] = np.array(load_png(files[(cam, 'rgb')]['path']).convert('RGB'))
            if (cam, 'depth') in files:
                im = load_png(files[(cam, 'depth')]['path'])
                obs.depth[cam] = np.array(im).astype(np.float64)
                mpu = files[(cam, 'depth')].get('meters_per_unit')
                if mpu:
                    obs.depth_scale[cam] = float(mpu)
        except Exception as e:
            log('frame file load fail', cam, e)


def parse_obs(result, prefer_manifest=True):
    meta, imgs = split_result(result)
    o = Obs()
    o.meta = meta
    o.step_id = find_key(meta, 'step_id')
    st = find_key(meta, 'state')
    o.state = as_state(st) if st is not None else None
    o.instruction = find_key(meta, 'instruction')
    o.ended, o.success = terminal_flags(meta)
    ki = find_key(meta, 'camera_intrinsics')
    kp = find_key(meta, 'camera_poses')
    for cam in ('front', 'wrist'):
        K = parse_intrinsics(per_camera(ki, cam))
        if K is not None:
            o.K[cam] = K
        P = parse_pose(per_camera(kp, cam))
        if P is not None:
            o.pose[cam] = P
    seq = find_key(meta, 'frame_sequence')
    mp = seq.get('manifest_path') if isinstance(seq, dict) else None
    if mp:
        man = frames_from_manifest(mp)
        o.manifest = man
        if man and prefer_manifest:
            frs = man.get('frames', []) or []
            ex = [f for f in frs if str(f.get('phase', '')) != 'video_demo']
            if ex:
                last = max(ex, key=lambda f: (f.get('step_id') or 0, f.get('frame_index') or 0))
                if o.step_id is None or last.get('step_id') in (None, o.step_id):
                    fill_from_frame(o, last)
    # inline images as fallback / complement
    if imgs and (('front' not in o.rgb) or ('front' not in o.depth)):
        rgbs, deps = [], []
        for b in imgs:
            try:
                im = decode_block(b)
                if im.mode in ('I;16', 'I', 'I;16B', 'F', 'I;16L') or b.get('mimeType', '').endswith('depth'):
                    deps.append(np.array(im).astype(np.float64))
                elif im.mode == 'L':
                    deps.append(np.array(im).astype(np.float64))
                else:
                    rgbs.append(np.array(im.convert('RGB')))
            except Exception as e:
                log('inline decode fail', e)
        if 'front' not in o.rgb and rgbs:
            o.rgb['front'] = rgbs[0]
        if 'wrist' not in o.rgb and len(rgbs) > 1:
            o.rgb['wrist'] = rgbs[1]
        if 'front' not in o.depth and deps:
            o.depth['front'] = deps[0]
        if 'wrist' not in o.depth and len(deps) > 1:
            o.depth['wrist'] = deps[1]
    return o


def video_frames(man, cam='front', max_frames=512):
    """Conditioning video RGB frames (sorted by frame_index), plus their states and depth paths."""
    out = []
    if not man:
        return out
    frs = [f for f in man.get('frames', []) or [] if str(f.get('phase', '')) == 'video_demo']
    frs.sort(key=lambda f: (f.get('frame_index') if f.get('frame_index') is not None else 0))
    if len(frs) > max_frames:
        idx = np.linspace(0, len(frs) - 1, max_frames).round().astype(int)
        frs = [frs[i] for i in idx]
    for f in frs:
        files = frame_files(f)
        if (cam, 'rgb') not in files:
            continue
        out.append(dict(frame=f, rgb_path=files[(cam, 'rgb')]['path'],
                        depth=files.get((cam, 'depth')), state=as_state(f.get('state')) if f.get('state') is not None else None,
                        t=f.get('timestamp_s')))
    return out
