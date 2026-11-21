"""MCP reply parsing and observation assembly, written defensively because the exact packet
layout was never observed live before the rehearsal (interaction-only, R1)."""
import base64
import glob
import io
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

WS = Path('/workspace')
OUT = WS / 'output'
_T0 = time.time()
_LOGF = None


def log(msg):
    global _LOGF
    line = f"[{time.time() - _T0:7.1f}s] {msg}"
    print(line, flush=True)
    try:
        if _LOGF is None:
            OUT.mkdir(parents=True, exist_ok=True)
            _LOGF = open(OUT / 'controller_log.txt', 'a')
        _LOGF.write(line + '\n')
        _LOGF.flush()
    except Exception:
        pass


def elapsed():
    return time.time() - _T0


def save_json(name, obj):
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / name, 'w') as f:
            json.dump(obj, f, indent=1, default=str)
    except Exception as e:
        log(f"save_json {name} failed: {e!r}")


def summarize(obj, depth=0, maxlen=200):
    """Structure summary of a JSON object (keys/types) for logging, image data elided."""
    if depth > 6:
        return '...'
    if isinstance(obj, dict):
        return {k: ('<%d chars>' % len(v) if isinstance(v, str) and len(v) > maxlen else summarize(v, depth + 1))
                for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 8 and all(isinstance(x, (int, float)) for x in obj):
            return f'<list {len(obj)} nums>'
        if len(obj) > 6:
            return [summarize(x, depth + 1) for x in obj[:3]] + [f'... {len(obj)} items']
        return [summarize(x, depth + 1) for x in obj]
    if isinstance(obj, str) and len(obj) > maxlen:
        return obj[:maxlen] + '...'
    return obj


# ----------------------------------------------------------------------------- reply parsing
def parse_reply(result):
    """-> (meta dict, list of image blocks with label hints)."""
    meta = {}
    images = []
    last_text = ''
    if not isinstance(result, dict):
        return meta, images
    content = result.get('content') or []
    for b in content:
        if not isinstance(b, dict):
            continue
        t = b.get('type')
        if t == 'text':
            txt = b.get('text', '') or ''
            parsed = None
            try:
                parsed = json.loads(txt)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                if not meta:
                    meta = parsed
                else:
                    meta.setdefault('_extra_json', []).append(parsed)
                last_text = ''
            else:
                last_text = txt
        elif t == 'image':
            hint = ' '.join(str(v) for k, v in b.items() if k not in ('data',) and isinstance(v, (str, int, float)))
            for k in ('annotations', '_meta', 'meta', 'metadata'):
                if k in b:
                    hint += ' ' + json.dumps(b[k], default=str)
            hint += ' ' + last_text
            images.append({'data': b.get('data'), 'mime': b.get('mimeType', ''), 'hint': hint.lower()})
            last_text = ''
        elif t == 'resource':
            res = b.get('resource') or {}
            if 'blob' in res and str(res.get('mimeType', '')).startswith('image'):
                images.append({'data': res.get('blob'), 'mime': res.get('mimeType', ''),
                               'hint': (str(res.get('uri', '')) + ' ' + last_text).lower()})
    if not meta and isinstance(result.get('structuredContent'), dict):
        meta = result['structuredContent']
    return meta, images


def find_all(obj, key, depth=0, out=None):
    if out is None:
        out = []
    if depth > 8:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            find_all(v, key, depth + 1, out)
    elif isinstance(obj, list):
        for x in obj[:600]:
            find_all(x, key, depth + 1, out)
    return out


def find_first(obj, keys):
    for k in keys:
        v = find_all(obj, k)
        if v:
            return v[0]
    return None


def any_true(obj, keys):
    for k in keys:
        for v in find_all(obj, k):
            if v is True:
                return True
    return False


def _vec(v, n):
    try:
        a = np.asarray(v, dtype=float).reshape(-1)
        if a.size == n and np.all(np.isfinite(a)):
            return a
    except Exception:
        pass
    return None


def _state_from_dict(d):
    pos = rot = fing = None
    for k, v in d.items():
        kl = str(k).lower()
        if pos is None and ('pos' in kl or 'xyz' in kl or 'translation' in kl) and 'finger' not in kl:
            pos = _vec(v, 3)
        elif rot is None and ('axis' in kl or 'rot' in kl or 'orient' in kl):
            rot = _vec(v, 3)
        elif fing is None and ('finger' in kl or 'gripper' in kl):
            fing = _vec(v, 2)
    if pos is not None and rot is not None:
        if fing is None:
            fing = np.array([0.04, 0.04])
        return np.concatenate([pos, rot, fing])
    return None


STATE_KEYS = ('state', 'proprio', 'proprioception', 'proprioceptive_state', 'robot_state', 'ee_state')


def find_state(meta):
    for key in STATE_KEYS:
        for v in find_all(meta, key):
            a = _vec(v, 8)
            if a is not None:
                return a
            if isinstance(v, dict):
                for kk in ('values', 'vector', 'data', 'value'):
                    if kk in v:
                        a = _vec(v[kk], 8)
                        if a is not None:
                            return a
                a = _state_from_dict(v)
                if a is not None:
                    return a
    return None


def find_step_id(meta):
    # prefer top-level / observation-level step_id
    for path in (('step_id',), ('observation', 'step_id'), ('obs', 'step_id')):
        cur = meta
        ok = True
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break
        if ok and isinstance(cur, (int, float)):
            return int(cur)
    vals = [v for v in find_all(meta, 'step_id') if isinstance(v, (int, float))]
    if vals:
        return int(max(vals))
    return None


# ----------------------------------------------------------------------------- images
def decode_block(b):
    try:
        raw = base64.b64decode(b['data'])
        im = Image.open(io.BytesIO(raw))
        im.load()
        return im
    except Exception:
        return None


def pil_to_rgb(im):
    return np.asarray(im.convert('RGB'), dtype=np.uint8)


def pil_to_depth(im, mpu=0.001):
    """16-bit PNG in millimetres -> float metres, NaN invalid. None if not a depth image."""
    if im is None:
        return None
    if im.mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F'):
        a = np.asarray(im).astype(np.float64)
        if im.mode == 'F' and np.nanmax(a) < 50:  # already metres
            d = a
        else:
            d = a * float(mpu)
        d[~np.isfinite(d) | (d <= 0)] = np.nan
        return d
    return None


def load_image_file(path):
    try:
        p = Path(path)
        if not p.is_absolute():
            p = WS / p
        if not p.exists():
            return None
        if p.suffix == '.npy':
            return np.load(p)
        im = Image.open(p)
        im.load()
        return im
    except Exception:
        return None


def resolve(path, base=None):
    p = Path(str(path))
    if p.is_absolute() and p.exists():
        return p
    q = WS / p
    if q.exists():
        return q
    if base is not None:
        q = Path(base) / p
        if q.exists():
            return q
    return WS / p


def _cam_of(text):
    t = str(text).lower()
    if 'wrist' in t or 'hand' in t:
        return 'wrist'
    if 'front' in t or 'third' in t or 'base' in t or 'main' in t:
        return 'front'
    return None


def _kind_of(text):
    t = str(text).lower()
    if 'depth' in t:
        return 'depth'
    if 'rgb' in t or 'color' in t:
        return 'rgb'
    return None


def collect_file_entries(meta, depth=0, keypath='', out=None):
    """Find dicts that reference image files; returns list of dict(camera, kind, path, mpu, step_id)."""
    if out is None:
        out = []
    if depth > 9:
        return out
    if isinstance(meta, dict):
        path = None
        for pk in ('path', 'file', 'filepath', 'file_path', 'uri'):
            v = meta.get(pk)
            if isinstance(v, str) and v.lower().split('?')[0].endswith(('.png', '.npy', '.jpg', '.jpeg')):
                path = v
                break
        if path is not None:
            cam = meta.get('camera') or _cam_of(keypath) or _cam_of(path)
            kind = meta.get('kind') or meta.get('modality') or _kind_of(keypath) or _kind_of(path)
            out.append(dict(camera=_cam_of(cam) if cam else None, kind=_kind_of(kind) if kind else None,
                            path=path, mpu=float(meta.get('meters_per_unit', 0.001) or 0.001),
                            keypath=keypath))
        for k, v in meta.items():
            if k in ('frame_sequence',):
                continue  # manifest frames handled separately
            if isinstance(v, str) and v.lower().endswith(('.png', '.npy')) and k not in ('path', 'file', 'filepath', 'file_path', 'uri'):
                cam = _cam_of(keypath + '/' + k) or _cam_of(v)
                kind = _kind_of(keypath + '/' + k) or _kind_of(v)
                out.append(dict(camera=cam, kind=kind, path=v, mpu=0.001, keypath=keypath + '/' + k))
            elif isinstance(v, (dict, list)):
                collect_file_entries(v, depth + 1, keypath + '/' + str(k), out)
    elif isinstance(meta, list):
        for i, x in enumerate(meta[:64]):
            collect_file_entries(x, depth + 1, keypath, out)
    return out


def read_manifest(meta):
    seq = meta.get('frame_sequence') if isinstance(meta, dict) else None
    if seq is None:
        seqs = find_all(meta, 'frame_sequence')
        seq = seqs[0] if seqs else None
    mp = None
    if isinstance(seq, dict):
        mp = seq.get('manifest_path') or seq.get('path') or seq.get('manifest')
    elif isinstance(seq, str):
        mp = seq
    if not mp:
        return None, None
    p = resolve(mp)
    try:
        return json.loads(p.read_text()), p
    except Exception as e:
        log(f"manifest read failed {p}: {e!r}")
        return None, p


def frame_files(frame, base=None):
    """manifest frame -> {(cam, kind): (path, mpu)}"""
    out = {}
    for f in frame.get('files', []) or []:
        cam = _cam_of(f.get('camera', '')) or f.get('camera')
        kind = _kind_of(f.get('kind', '')) or f.get('kind')
        if cam and kind and f.get('path'):
            out[(cam, kind)] = (resolve(f['path'], base), float(f.get('meters_per_unit', 0.001) or 0.001))
    return out


# ----------------------------------------------------------------------------- calibration in packet
def find_calibration(meta):
    intr = find_first(meta, ['camera_intrinsics', 'intrinsics', 'camera_intrinsic'])
    poses = find_first(meta, ['camera_poses', 'camera_pose', 'extrinsics', 'camera_extrinsics'])
    return intr, poses


def per_camera(v, cam):
    """Pick the entry for camera `cam` from a dict-of-cameras or list-of-cameras container."""
    if v is None:
        return None
    if isinstance(v, dict):
        for k, val in v.items():
            if _cam_of(k) == cam:
                return val
        # maybe a dict with 'camera' field
        if _cam_of(v.get('camera', '')) == cam:
            return v
        return None
    if isinstance(v, list):
        for e in v:
            if isinstance(e, dict) and _cam_of(e.get('camera', e.get('name', ''))) == cam:
                return e
    return None
