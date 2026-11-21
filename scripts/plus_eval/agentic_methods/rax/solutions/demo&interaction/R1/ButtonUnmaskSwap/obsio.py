"""Defensive parsing of MCP replies into observations.

The exact reply layout was never seen offline, so every field is searched for
under several plausible names, and every decision is logged.
"""
import base64
import io
import json
import os
import re

import numpy as np
from PIL import Image

from geom import quat_wxyz_to_R

WS = '/workspace'
CAMS = ('front', 'wrist')


def log(*a):
    try:
        print(*a, flush=True)
    except Exception:
        pass


# ---------------------------------------------------------------- utilities
def walk(o, path=()):
    """yield (path, key, value) for all dict entries recursively"""
    if isinstance(o, dict):
        for k, v in o.items():
            yield path, k, v
            yield from walk(v, path + (k,))
    elif isinstance(o, list):
        for i, v in enumerate(o[:600]):
            yield from walk(v, path + (i,))


def find_key(o, names, want=None):
    """breadth-first search for the first key in names (shallowest wins)"""
    names = [n.lower() for n in names]
    queue = [o]
    n = 0
    while queue and n < 20000:
        cur = queue.pop(0)
        n += 1
        if isinstance(cur, dict):
            for k, v in cur.items():
                if isinstance(k, str) and k.lower() in names and (want is None or isinstance(v, want)):
                    return v
            queue.extend(v for v in cur.values() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            queue.extend(v for v in cur[:600] if isinstance(v, (dict, list)))
    return None


def strip_big(o, depth=0):
    """copy of JSON with long strings / lists shortened, for logging"""
    if depth > 6:
        return '...'
    if isinstance(o, dict):
        return {k: strip_big(v, depth + 1) for k, v in list(o.items())[:60]}
    if isinstance(o, list):
        if len(o) > 12:
            return [strip_big(v, depth + 1) for v in o[:4]] + ['...(%d)' % len(o)]
        return [strip_big(v, depth + 1) for v in o]
    if isinstance(o, str) and len(o) > 300:
        return o[:120] + '...(%d chars)' % len(o)
    return o


def resolve_path(p, base=None):
    if not isinstance(p, str):
        return None
    if p.startswith('file://'):
        p = p[7:]
    cands = [p] if os.path.isabs(p) else []
    cands += [os.path.join(WS, p)]
    if base:
        cands.append(os.path.join(base, p))
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def decode_image_bytes(b):
    im = Image.open(io.BytesIO(b))
    im.load()
    return im


def image_to_array(im, mpu=0.001):
    """returns (kind, array): kind 'rgb' uint8 HxWx3 or 'depth' float metres"""
    if im.mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F'):
        a = np.array(im).astype(np.float64)
        if im.mode == 'F' and a.max() < 50:
            return 'depth', a  # already metres
        return 'depth', a * mpu
    if im.mode == 'L':
        return 'gray', np.array(im)
    return 'rgb', np.array(im.convert('RGB'))


def load_file(path, mpu=0.001):
    try:
        im = Image.open(path)
        im.load()
        return image_to_array(im, mpu)
    except Exception as e:
        log('load_file failed', path, e)
        return None, None


# ---------------------------------------------------------------- calibration
def _to_K(v):
    if v is None:
        return None
    try:
        if isinstance(v, dict):
            for k in ('K', 'matrix', 'intrinsic_matrix', 'intrinsics', 'camera_matrix', 'k'):
                if k in v:
                    return _to_K(v[k])
            fx = v.get('fx'); fy = v.get('fy', fx)
            cx = v.get('cx', v.get('ppx')); cy = v.get('cy', v.get('ppy'))
            if fx is not None and cx is not None:
                return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], float)
            return None
        a = np.array(v, float)
        if a.shape == (3, 3):
            return a
        if a.size == 9:
            return a.reshape(3, 3)
        if a.size == 4:  # fx fy cx cy
            return np.array([[a[0], 0, a[2]], [0, a[1], a[3]], [0, 0, 1.0]])
    except Exception:
        return None
    return None


def _to_pose(v):
    """returns (R, t) or None"""
    if v is None:
        return None
    try:
        if isinstance(v, dict):
            t = None
            for k in ('position_m', 'position', 'p', 'translation', 'pos', 't', 'xyz'):
                if k in v:
                    t = np.array(v[k], float).reshape(3)
                    break
            q = None
            for k in ('quaternion_wxyz', 'wxyz', 'quaternion', 'q', 'orientation', 'quat', 'rotation_wxyz'):
                if k in v:
                    q = v[k]
                    break
            if t is not None and q is not None:
                if isinstance(q, dict):
                    q = [q.get('w'), q.get('x'), q.get('y'), q.get('z')]
                q = np.array(q, float)
                if q.size == 4:
                    if k in ('quaternion_xyzw', 'xyzw'):
                        q = q[[3, 0, 1, 2]]
                    return quat_wxyz_to_R(q), t
                if q.size == 9:
                    return q.reshape(3, 3), t
            for k in ('matrix', 'pose', 'T', 'transform', 'extrinsic', 'cam2world', 'camera_to_world', 'model_matrix'):
                if k in v:
                    r = _to_pose(v[k])
                    if r is not None:
                        return r
            return None
        a = np.array(v, float)
        if a.shape == (4, 4):
            return a[:3, :3], a[:3, 3]
        if a.size == 16:
            a = a.reshape(4, 4)
            return a[:3, :3], a[:3, 3]
        if a.size == 7:  # xyz + wxyz
            return quat_wxyz_to_R(a[3:]), a[:3]
    except Exception:
        return None
    return None


def per_camera(v, fn):
    out = {}
    if isinstance(v, dict):
        for cam in CAMS:
            for k, val in v.items():
                if isinstance(k, str) and cam in k.lower():
                    r = fn(val)
                    if r is not None:
                        out[cam] = r
                        break
    elif isinstance(v, list):
        for item in v:
            if isinstance(item, dict):
                name = str(item.get('camera', item.get('name', item.get('id', '')))).lower()
                for cam in CAMS:
                    if cam in name:
                        r = fn(item)
                        if r is not None:
                            out[cam] = r
    return out


def extract_calib(meta):
    Ks, poses = {}, {}
    for p, k, v in walk(meta):
        if not isinstance(k, str):
            continue
        kl = k.lower()
        if 'intrinsic' in kl:
            for c, r in per_camera(v, _to_K).items():
                Ks.setdefault(c, r)
        if ('pose' in kl or 'extrinsic' in kl) and 'camera' in kl:
            for c, r in per_camera(v, _to_pose).items():
                poses.setdefault(c, r)
    # per-camera dict layout: {"cameras": {"front": {"intrinsics":..., "pose":...}}}
    if len(Ks) < 2 or len(poses) < 2:
        for p, k, v in walk(meta):
            if isinstance(k, str) and k.lower() in CAMS and isinstance(v, dict):
                if k.lower() not in Ks:
                    for kk in v:
                        if 'intrinsic' in str(kk).lower() or kk in ('K',):
                            r = _to_K(v[kk])
                            if r is not None:
                                Ks[k.lower()] = r
                if k.lower() not in poses:
                    for kk in v:
                        if 'pose' in str(kk).lower() or 'extrinsic' in str(kk).lower():
                            r = _to_pose(v[kk])
                            if r is not None:
                                poses[k.lower()] = r
    return Ks, poses


# ---------------------------------------------------------------- main parse
def cam_of(text):
    t = str(text).lower()
    if 'wrist' in t or 'hand' in t:
        return 'wrist'
    if 'front' in t or 'third' in t or 'base' in t or 'agent' in t:
        return 'front'
    return None


def kind_of(text):
    t = str(text).lower()
    if 'depth' in t:
        return 'depth'
    if 'rgb' in t or 'color' in t or 'colour' in t:
        return 'rgb'
    return None


class Reply:
    """Parsed reply: meta dict, images {(cam,kind): array}, calib, state."""

    def __init__(self):
        self.meta = {}
        self.images = {}
        self.Ks = {}
        self.poses = {}
        self.state = None
        self.step = None
        self.ended = False
        self.complete = None
        self.manifest = None
        self.raw_blocks = []
        self.notes = []

    def summary(self):
        return dict(step=self.step, ended=self.ended, complete=self.complete,
                    images={str(k): (v.shape if hasattr(v, 'shape') else None) for k, v in self.images.items()},
                    Ks=list(self.Ks), poses=list(self.poses), state=None if self.state is None else [round(float(x), 4) for x in self.state],
                    notes=self.notes[-5:])


def parse(result):
    r = Reply()
    blocks = result.get('content', []) if isinstance(result, dict) else []
    texts = []
    for b in blocks:
        if b.get('type') == 'text':
            texts.append(b.get('text', ''))
    for t in texts:
        try:
            j = json.loads(t)
            if isinstance(j, dict):
                for k, v in j.items():
                    r.meta.setdefault(k, v)
        except Exception:
            pass
    if isinstance(result, dict) and isinstance(result.get('structuredContent'), dict):
        for k, v in result['structuredContent'].items():
            r.meta.setdefault(k, v)
    meta = r.meta
    mpu = find_key(meta, ['meters_per_unit', 'depth_scale', 'metres_per_unit'])
    try:
        mpu = float(mpu) if mpu is not None else 0.001
    except Exception:
        mpu = 0.001
    # ---- image blocks
    imgs = []
    hint = ''
    for b in blocks:
        typ = b.get('type')
        if typ == 'text':
            t = b.get('text', '')
            try:
                json.loads(t)
                hint_t = ''
            except Exception:
                hint_t = t[:200]
            hint = hint_t
            continue
        data = None
        label = hint
        ann = b.get('annotations') or b.get('_meta') or b.get('meta') or {}
        label = label + ' ' + json.dumps(ann)[:300] if ann else label
        try:
            if typ == 'image' and b.get('data'):
                data = base64.b64decode(b['data'])
            elif typ == 'resource':
                res = b.get('resource', {})
                label += ' ' + str(res.get('uri', '')) + ' ' + str(res.get('name', ''))
                if res.get('blob'):
                    data = base64.b64decode(res['blob'])
                elif res.get('uri'):
                    p = resolve_path(res['uri'])
                    if p:
                        data = open(p, 'rb').read()
            elif typ == 'resource_link':
                label += ' ' + str(b.get('uri', '')) + ' ' + str(b.get('name', ''))
                p = resolve_path(b.get('uri', ''))
                if p:
                    data = open(p, 'rb').read()
        except Exception as e:
            r.notes.append('block decode err %s' % e)
        if data is None:
            continue
        try:
            im = decode_image_bytes(data)
            kind, arr = image_to_array(im, mpu)
            imgs.append([label, kind, arr])
        except Exception as e:
            r.notes.append('img decode err %s' % e)
        hint = ''
    # labels from metadata lists aligned with images
    lab_list = None
    for p, k, v in walk(meta):
        if isinstance(v, list) and len(v) == len(imgs) and len(v) > 0 and all(isinstance(x, dict) for x in v):
            if all(any(cam_of(json.dumps(x)) for _ in [0]) for x in v):
                lab_list = v
                break
    if lab_list is not None:
        for it, x in zip(imgs, lab_list):
            it[0] = it[0] + ' ' + json.dumps(x)[:300]
    # assign cameras
    used = set()
    pending = []
    for label, kind, arr in imgs:
        cam = cam_of(label)
        k2 = kind_of(label)
        if kind in ('depth',):
            k2 = 'depth'
        elif kind == 'rgb':
            k2 = 'rgb'
        if kind == 'gray':
            continue
        if cam and (cam, k2) not in used:
            r.images[(cam, k2)] = arr
            used.add((cam, k2))
        else:
            pending.append((k2, arr))
    for k2, arr in pending:  # order fallback: front first, then wrist
        for cam in CAMS:
            if (cam, k2) not in used:
                r.images[(cam, k2)] = arr
                used.add((cam, k2))
                r.notes.append('camera by order %s %s' % (cam, k2))
                break
    # ---- files referenced in metadata (camera/kind/path dicts)
    for p, k, v in walk(meta):
        if isinstance(v, dict) and 'path' in v and ('camera' in v or 'kind' in v):
            cam = cam_of(v.get('camera', '')) or cam_of(v.get('path', ''))
            kind = kind_of(v.get('kind', '')) or kind_of(v.get('path', ''))
            if cam and kind and (cam, kind) not in r.images and 'frame_sequence' not in [str(x) for x in p]:
                fp = resolve_path(v['path'])
                if fp:
                    kk, arr = load_file(fp, float(v.get('meters_per_unit', mpu) or mpu))
                    if arr is not None:
                        r.images[(cam, kind)] = arr
                        r.notes.append('file %s %s' % (cam, kind))
    # ---- frame sequence manifest
    fs = find_key(meta, ['frame_sequence'])
    mp = None
    if isinstance(fs, dict):
        mp = fs.get('manifest_path') or fs.get('path')
    if mp is None:
        mp = find_key(meta, ['manifest_path'])
    if isinstance(mp, str):
        fp = resolve_path(mp)
        if fp:
            try:
                r.manifest = json.load(open(fp))
                r.manifest['_dir'] = os.path.dirname(fp)
            except Exception as e:
                r.notes.append('manifest err %s' % e)
    # ---- calibration
    r.Ks, r.poses = extract_calib(meta)
    # ---- state / step / end
    st = find_key(meta, ['state', 'proprio', 'proprioception'])
    if isinstance(st, dict):
        st = find_key(st, ['values', 'vector', 'data', 'state'])
    try:
        if st is not None:
            a = np.array(st, float).reshape(-1)
            if a.size >= 8:
                r.state = a[:8]
    except Exception:
        pass
    sid = find_key(meta, ['step_id', 'step', 'elapsed_steps'])
    try:
        r.step = int(sid) if sid is not None and not isinstance(sid, (dict, list)) else None
    except Exception:
        r.step = None
    e = find_key(meta, ['episode_ended', 'ended', 'done', 'terminated'])
    r.ended = bool(e) if isinstance(e, (bool, int)) else False
    c = find_key(meta, ['task_complete', 'success', 'task_success'])
    r.complete = c if isinstance(c, (bool, int)) else None
    if r.complete:
        r.ended = r.ended or True
    # manifest last frame as source if images missing
    if r.manifest and (('front', 'rgb') not in r.images or ('front', 'depth') not in r.images):
        frames = manifest_frames(r.manifest, load=False)
        if frames:
            last = load_frame_images(frames[-1])
            for (cam, kind), arr in last['images'].items():
                r.images.setdefault((cam, kind), arr)
            if r.state is None and last.get('state') is not None:
                r.state = last['state']
            r.notes.append('images from manifest last frame')
    return r


def manifest_frames(man, load=True, only=None):
    """list of dict(index, phase, step, state, images{(cam,kind):arr}, timestamp)"""
    out = []
    frames = man.get('frames', []) if isinstance(man, dict) else []
    base = man.get('_dir') if isinstance(man, dict) else None
    for i, f in enumerate(frames):
        rec = dict(index=f.get('frame_index', i), phase=f.get('phase'), step=f.get('step_id'),
                   timestamp=f.get('timestamp_s'), state=None, images={}, files={})
        try:
            if f.get('state') is not None:
                rec['state'] = np.array(f['state'], float)[:8]
        except Exception:
            pass
        for fi in f.get('files', []) or []:
            cam = cam_of(fi.get('camera', '')) or cam_of(fi.get('path', ''))
            kind = kind_of(fi.get('kind', '')) or kind_of(fi.get('path', ''))
            if not cam or not kind:
                continue
            fp = resolve_path(fi.get('path', ''), base)
            if fp:
                rec['files'][(cam, kind)] = (fp, float(fi.get('meters_per_unit', 0.001) or 0.001))
        if load:
            for key, (fp, mpu) in rec['files'].items():
                if only and key not in only:
                    continue
                kk, arr = load_file(fp, mpu)
                if arr is not None:
                    rec['images'][key] = arr
        out.append(rec)
    out.sort(key=lambda r: (r['index'] if isinstance(r['index'], (int, float)) else 0))
    return out


def load_frame_images(rec, only=None):
    for key, (fp, mpu) in rec['files'].items():
        if only and key not in only:
            continue
        if key not in rec['images']:
            kk, arr = load_file(fp, mpu)
            if arr is not None:
                rec['images'][key] = arr
    return rec


def parse_points(result):
    """robomme_pixel_to_position reply -> list of (xyz or None)"""
    meta = {}
    for b in (result.get('content', []) if isinstance(result, dict) else []):
        if b.get('type') == 'text':
            try:
                j = json.loads(b['text'])
                if isinstance(j, dict):
                    meta.update(j)
                elif isinstance(j, list):
                    meta['points'] = j
            except Exception:
                pass
    if isinstance(result, dict) and isinstance(result.get('structuredContent'), dict):
        meta.update(result['structuredContent'])
    lst = None
    for p, k, v in walk(meta):
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            if any(('u' in x and 'v' in x) or 'valid' in x or 'position' in x or 'world' in x for x in v):
                lst = v
                break
    out = []
    if lst is None:
        return out
    for x in lst:
        if x.get('valid') is False:
            out.append(None)
            continue
        pt = None
        for k in ('position', 'world_position', 'world', 'point', 'xyz', 'position_m', 'world_point', 'point_world'):
            if k in x:
                try:
                    a = np.array(x[k] if not isinstance(x[k], dict) else [x[k].get('x'), x[k].get('y'), x[k].get('z')], float).reshape(-1)
                    if a.size == 3:
                        pt = a
                        break
                except Exception:
                    pass
        if pt is None:
            for p2, k2, v2 in walk(x):
                if isinstance(v2, list) and len(v2) == 3 and all(isinstance(z, (int, float)) for z in v2):
                    pt = np.array(v2, float)
                    break
        out.append(pt)
    return out


def instruction_of(meta):
    ins = find_key(meta, ['instruction', 'goal', 'language_instruction', 'task_instruction'])
    return ins if isinstance(ins, str) else ''


COLOR_WORDS = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'black', 'white', 'pink', 'cyan', 'brown', 'gray', 'grey']


def target_colors(ins):
    s = ins.lower()
    # after "then" if present
    i = s.find('then')
    s2 = s[i:] if i >= 0 else s
    found = []
    for m in re.finditer(r'[a-z]+', s2):
        w = m.group(0)
        if w in COLOR_WORDS:
            found.append('gray' if w == 'grey' else w)
    return found


def remaining_steps(meta):
    lim = find_key(meta, ['episode_limits', 'limits'])
    src = lim if isinstance(lim, dict) else meta
    for k in ('remaining_steps', 'steps_remaining', 'remaining'):
        v = find_key(src, [k])
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    return None


def max_steps(meta):
    lim = find_key(meta, ['episode_limits', 'limits'])
    src = lim if isinstance(lim, dict) else meta
    for k in ('max_steps', 'step_limit', 'max_episode_steps', 'horizon', 'max_control_steps', 'steps'):
        v = find_key(src, [k])
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    return None
