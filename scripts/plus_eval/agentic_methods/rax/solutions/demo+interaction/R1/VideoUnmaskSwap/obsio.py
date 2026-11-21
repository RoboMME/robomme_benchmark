"""Defensive parsing of MCP replies, frame manifests and image/depth files.

The exact nesting of observation packets is not documented, so everything here
searches the reply JSON recursively and validates what it finds.
"""
import base64
import io
import json
import os
import time

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

WS = '/workspace'
LOG = []
T0 = time.time()


def log(*a):
    s = ' '.join(str(x) for x in a)
    line = f'[{time.time() - T0:7.1f}] {s}'
    LOG.append(line)
    try:
        print(line, flush=True)
    except Exception:
        pass


class Reply:
    def __init__(self, raw):
        self.raw = raw if isinstance(raw, dict) else {}
        self.texts = []
        self.images = []
        for b in self.raw.get('content', []) or []:
            if not isinstance(b, dict):
                continue
            if b.get('type') == 'text':
                self.texts.append(b.get('text', ''))
            elif b.get('type') == 'image':
                self.images.append(b)
        self.meta = {}
        self.metas = []
        for t in self.texts:
            try:
                j = json.loads(t)
                self.metas.append(j)
            except Exception:
                continue
        if self.metas:
            self.meta = self.metas[0] if isinstance(self.metas[0], dict) else {'value': self.metas[0]}
        sc = self.raw.get('structuredContent')
        if isinstance(sc, dict):
            self.metas.append(sc)
            if not self.meta:
                self.meta = sc

    def all(self):
        return self.metas


def walk(obj, path=()):
    """Yield (path, key, value) for all dict items recursively."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k, v
            yield from walk(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, path + (i,))


def find_key(obj, key):
    return [v for _, k, v in walk(obj) if k == key]


def find_first(obj, keys, pred=None):
    for key in keys:
        for v in find_key(obj, key):
            if pred is None or pred(v):
                return v
    return None


def is_vec(v, n):
    return isinstance(v, (list, tuple)) and len(v) == n and all(isinstance(x, (int, float)) for x in v)


def find_dicts(obj, pred):
    out = []
    if isinstance(obj, dict):
        if pred(obj):
            out.append(obj)
        for v in obj.values():
            out.extend(find_dicts(v, pred))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(find_dicts(v, pred))
    return out


def abspath(p):
    if not isinstance(p, str):
        return None
    if os.path.isabs(p):
        return p
    return os.path.join(WS, p)


def read_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB')).copy()


def read_depth(path, mpu=0.001):
    a = None
    if cv2 is not None:
        try:
            a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            a = None
    if a is None:
        with Image.open(path) as im:
            a = np.array(im)
    if a.ndim == 3:
        a = a[..., 0]
    return a.astype(np.float64) * float(mpu or 0.001)


def decode_image_block(b):
    try:
        data = base64.b64decode(b.get('data', ''))
        im = Image.open(io.BytesIO(data))
        mode = im.mode
        if mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F'):
            return 'depth', np.array(im).astype(np.float64)
        return 'rgb', np.asarray(im.convert('RGB')).copy()
    except Exception:
        return None, None


def files_of(frame):
    """{(camera, kind): (path, meters_per_unit)} from a manifest frame."""
    out = {}
    for f in frame.get('files', []) or []:
        if not isinstance(f, dict):
            continue
        cam = str(f.get('camera', '')).lower()
        kind = str(f.get('kind', '')).lower()
        p = f.get('path')
        if not p:
            continue
        if 'wrist' in cam or 'hand' in cam:
            cam = 'wrist'
        elif cam:
            cam = 'front' if ('front' in cam or 'third' in cam or 'base' in cam) else cam
        if 'depth' in kind:
            kind = 'depth'
        elif 'rgb' in kind or 'color' in kind or 'image' in kind:
            kind = 'rgb'
        out[(cam, kind)] = (abspath(p), f.get('meters_per_unit', 0.001))
    return out


def load_manifest(path):
    try:
        with open(abspath(path)) as fh:
            return json.load(fh)
    except Exception as e:
        log('manifest load failed', path, repr(e))
        return None


def manifest_paths(reply):
    out = []
    for m in reply.all():
        for v in find_key(m, 'manifest_path'):
            if isinstance(v, str) and v not in out:
                out.append(v)
    return out


def frame_order_key(fr):
    for k in ('frame_index', 'index', 'source_frame_index'):
        if isinstance(fr.get(k), (int, float)):
            return fr[k]
    return 0


def get_state(obj):
    """8-D state from a reply meta (prefer key 'state')."""
    for v in find_key(obj, 'state'):
        if is_vec(v, 8):
            return np.asarray(v, float)
        if isinstance(v, dict):
            for vv in v.values():
                if is_vec(vv, 8):
                    return np.asarray(vv, float)
    for key in ('proprio', 'proprioception', 'robot_state'):
        for v in find_key(obj, key):
            if is_vec(v, 8):
                return np.asarray(v, float)
    return None


def get_step_id(obj):
    for key in ('step_id', 'step', 'elapsed_steps'):
        for v in find_key(obj, key):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
    return None


def episode_ended(obj):
    for key in ('episode_ended', 'terminated', 'done', 'task_complete', 'success', 'truncated'):
        for v in find_key(obj, key):
            if v is True:
                return True, key
    return False, None


DIHEDRAL = [(f, k) for f in (False, True) for k in range(4)]


def apply_tf(a, tf):
    """Apply a dihedral transform (flip_lr, rot90 count) to an image array."""
    if a is None or tf is None or tf == (False, 0):
        return a
    f, k = tf
    b = a[:, ::-1] if f else a
    return np.ascontiguousarray(np.rot90(b, k))


def _mad(a, b):
    try:
        if a.shape != b.shape:
            return 1e9
        return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
    except Exception:
        return 1e9


def best_tf(a, b):
    """Dihedral transform T minimising mean|T(a) - b| -> (T, diff)."""
    best = ((False, 0), _mad(a, b))
    if a is None or b is None:
        return best
    for tf in DIHEDRAL[1:]:
        d = _mad(apply_tf(a, tf), b)
        if d < best[1] - 1e-6:
            best = (tf, d)
    return best


def image_similarity(a, b):
    """Higher is more similar; tolerant to flips/rotations between sources."""
    try:
        if a is None or b is None:
            return -1e9
        d = _mad(a, b)
        if a.ndim >= 2 and a.shape[0] == a.shape[1]:
            d = min(d, best_tf(a, b)[1])
        return -d
    except Exception:
        return -1e9
