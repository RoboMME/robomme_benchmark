"""Defensive parsing of MCP replies, manifests and image files."""
import base64
import io
import json
import os
import numpy as np

WORKSPACE = '/workspace'

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None
try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def reply_json(reply):
    """First text block parsed as JSON (dict); {} if impossible."""
    if isinstance(reply, dict) and 'content' in reply:
        blocks = reply.get('content') or []
    elif isinstance(reply, list):
        blocks = reply
    else:
        blocks = []
    for b in blocks:
        if isinstance(b, dict) and b.get('type') == 'text':
            t = b.get('text', '')
            try:
                v = json.loads(t)
                if isinstance(v, dict):
                    return v
                if isinstance(v, list):
                    return {'_list': v}
            except Exception:
                continue
    if isinstance(reply, dict):
        sc = reply.get('structuredContent')
        if isinstance(sc, dict):
            return sc
    return {}


def reply_images(reply):
    out = []
    blocks = reply.get('content', []) if isinstance(reply, dict) else []
    for b in blocks:
        if isinstance(b, dict) and b.get('type') == 'image' and b.get('data'):
            out.append(b)
    return out


def find_key(obj, names, depth=0, maxdepth=6):
    """Depth-first search for the first value whose key is in names."""
    if depth > maxdepth:
        return None
    if isinstance(names, str):
        names = (names,)
    if isinstance(obj, dict):
        for n in names:
            if n in obj:
                return obj[n]
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                r = find_key(v, names, depth + 1, maxdepth)
                if r is not None:
                    return r
    elif isinstance(obj, list):
        for v in obj[:50]:
            if isinstance(v, (dict, list)):
                r = find_key(v, names, depth + 1, maxdepth)
                if r is not None:
                    return r
    return None


def is_num_list(v, n=None):
    if not isinstance(v, (list, tuple)):
        return False
    if n is not None and len(v) != n:
        return False
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


def parse_state(meta):
    s = find_key(meta, ('state', 'robot_state', 'proprio', 'proprioception'))
    if isinstance(s, dict):
        for k in ('values', 'vector', 'data', 'state'):
            if is_num_list(s.get(k), 8):
                return np.array(s[k], float)
        pos = s.get('ee_position') or s.get('position')
        ori = s.get('ee_axis_angle') or s.get('orientation') or s.get('axis_angle')
        fin = s.get('fingers') or s.get('gripper') or s.get('finger_positions')
        if is_num_list(pos, 3) and is_num_list(ori, 3):
            f = fin if is_num_list(fin, 2) else [0.0, 0.0]
            return np.array(list(pos) + list(ori) + list(f), float)
    if is_num_list(s) and len(s) >= 8:
        return np.array(s[:8], float)
    return None


def parse_step_id(meta):
    for k in ('step_id', 'step', 'elapsed_steps', 'steps'):
        v = find_key(meta, k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    return None


def episode_ended(meta):
    for k in ('episode_ended', 'terminated', 'done', 'episode_done'):
        v = find_key(meta, k)
        if v is True:
            return True
    return False


def task_complete(meta):
    for k in ('task_complete', 'success'):
        v = find_key(meta, k)
        if v is True:
            return True
    return False


def max_steps_from(meta):
    """Find the episode step limit in a status/step reply."""
    lim = find_key(meta, ('episode_limits', 'limits'))
    cands = []
    if isinstance(lim, dict):
        for k, v in lim.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and 'step' in k.lower():
                cands.append((k, int(v)))
    for k in ('max_steps', 'step_limit', 'max_episode_steps', 'horizon'):
        v = find_key(meta, k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            cands.append((k, int(v)))
    for k, v in cands:
        if 'remain' not in k.lower() and 'used' not in k.lower() and v > 0:
            return v
    return None


def remaining_steps_from(meta):
    lim = find_key(meta, ('episode_limits', 'limits'))
    if isinstance(lim, dict):
        for k, v in lim.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and 'remain' in k.lower():
                return int(v)
    v = find_key(meta, ('remaining_steps', 'steps_remaining'))
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return int(v)
    return None


def resolve_path(p, base_dirs=()):
    if not isinstance(p, str):
        return None
    cands = []
    if os.path.isabs(p):
        cands.append(p)
    cands.append(os.path.join(WORKSPACE, p))
    for b in base_dirs:
        cands.append(os.path.join(b, p))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def manifest_path_from(meta):
    fs = find_key(meta, ('frame_sequence',))
    p = None
    if isinstance(fs, dict):
        p = fs.get('manifest_path') or fs.get('path') or fs.get('manifest')
    elif isinstance(fs, str):
        p = fs
    if p is None:
        p = find_key(meta, ('manifest_path',))
    return resolve_path(p) if isinstance(p, str) else None


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def frame_files(frame, manifest_dir):
    """{(camera, kind): (abs_path, meters_per_unit)}"""
    out = {}
    files = frame.get('files') or []
    if isinstance(files, dict):
        # alternative layout {camera: {kind: path}}
        tmp = []
        for cam, v in files.items():
            if isinstance(v, dict):
                for kind, p in v.items():
                    tmp.append({'camera': cam, 'kind': kind, 'path': p if isinstance(p, str) else (p or {}).get('path')})
        files = tmp
    for f in files:
        if not isinstance(f, dict):
            continue
        cam = str(f.get('camera', '')).lower()
        kind = str(f.get('kind', '')).lower()
        p = f.get('path') or f.get('file')
        ap = resolve_path(p, (manifest_dir,))
        if ap is None:
            continue
        if 'depth' in kind:
            kind = 'depth'
        elif 'rgb' in kind or 'color' in kind or 'image' in kind:
            kind = 'rgb'
        if 'wrist' in cam or 'hand' in cam:
            cam = 'wrist'
        elif 'front' in cam or 'base' in cam or 'third' in cam:
            cam = 'front'
        mpu = f.get('meters_per_unit', 0.001)
        try:
            mpu = float(mpu)
        except Exception:
            mpu = 0.001
        out[(cam, kind)] = (ap, mpu)
    return out


def read_rgb(path):
    if cv2 is not None:
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is not None:
            return np.ascontiguousarray(im[:, :, ::-1])
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'))


def read_depth(path, mpu=0.001):
    arr = None
    if path.endswith('.npy'):
        arr = np.load(path).astype(np.float64)
        if arr.dtype.kind == 'f' and np.nanmax(arr) < 50:
            return arr
    if arr is None and cv2 is not None:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is not None:
            arr = im
    if arr is None:
        with Image.open(path) as im:
            arr = np.asarray(im)
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype.kind == 'f':
        d = arr.astype(np.float64)
        if np.nanmax(d) > 50:  # probably mm
            d = d * mpu
    else:
        d = arr.astype(np.float64) * mpu
    d[~np.isfinite(d)] = 0
    return d


def decode_image_block(b):
    data = base64.b64decode(b['data'])
    with Image.open(io.BytesIO(data)) as im:
        if im.mode in ('I;16', 'I;16B', 'I', 'F', 'I;16L'):
            return 'depth', np.asarray(im)
        return 'rgb', np.asarray(im.convert('RGB'))


def sorted_frames(manifest):
    fr = manifest.get('frames') or []
    def key(f):
        for k in ('frame_index', 'index', 'frame'):
            v = f.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0
    return sorted([f for f in fr if isinstance(f, dict)], key=key)
