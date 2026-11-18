"""Reply parsing, image decoding, calibration and back-projection for StopCube.

Written blind (no live reply seen before the rehearsal), so every accessor is
defensive: recursive key search, several encodings, explicit None on failure.
"""
import base64
import io
import json
import os

import numpy as np
from PIL import Image

WS = '/workspace'


# ----------------------------------------------------------------- generic JSON
def blocks(res):
    if isinstance(res, dict):
        c = res.get('content')
        if isinstance(c, list):
            return c
    return []


def meta_of(res):
    for b in blocks(res):
        if isinstance(b, dict) and b.get('type') == 'text':
            try:
                j = json.loads(b.get('text', ''))
            except Exception:
                continue
            if isinstance(j, dict):
                return j
    if isinstance(res, dict):
        sc = res.get('structuredContent')
        if isinstance(sc, dict):
            return sc
    return {}


def bfs_find(obj, names, max_nodes=20000):
    """First value (breadth first) whose key is in names."""
    names = set(names)
    queue = [obj]
    n = 0
    while queue and n < max_nodes:
        nxt = []
        for o in queue:
            n += 1
            if isinstance(o, dict):
                for k, v in o.items():
                    if k in names:
                        return v
                for k, v in o.items():
                    if isinstance(v, (dict, list)):
                        nxt.append(v)
            elif isinstance(o, list):
                for v in o:
                    if isinstance(v, (dict, list)):
                        nxt.append(v)
        queue = nxt
    return None


def bfs_find_all(obj, pred, max_nodes=20000):
    out = []
    queue = [obj]
    n = 0
    while queue and n < max_nodes:
        nxt = []
        for o in queue:
            n += 1
            if isinstance(o, dict):
                if pred(o):
                    out.append(o)
                for v in o.values():
                    if isinstance(v, (dict, list)):
                        nxt.append(v)
            elif isinstance(o, list):
                for v in o:
                    if isinstance(v, (dict, list)):
                        nxt.append(v)
        queue = nxt
    return out


def strip_meta(obj, depth=0, maxlen=300):
    """Copy of meta for logging, long strings / lists shortened."""
    if depth > 6:
        return '...'
    if isinstance(obj, dict):
        return {k: strip_meta(v, depth + 1, maxlen) for k, v in list(obj.items())[:60]}
    if isinstance(obj, list):
        if len(obj) > 12:
            return [strip_meta(v, depth + 1, maxlen) for v in obj[:6]] + ['...(%d items)' % len(obj)]
        return [strip_meta(v, depth + 1, maxlen) for v in obj]
    if isinstance(obj, str) and len(obj) > maxlen:
        return obj[:maxlen] + '...'
    return obj


def as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ('true', '1', 'yes')
    return False


def vec(v, n=None):
    try:
        a = np.asarray(v, dtype=float).reshape(-1)
    except Exception:
        return None
    if n is not None and a.size != n:
        return None
    if not np.all(np.isfinite(a)):
        return None
    return a


# ----------------------------------------------------------------- fields
def get_state(meta):
    s = bfs_find(meta, ['state'])
    a = vec(s)
    if a is not None and a.size >= 8:
        return a[:8]
    if isinstance(s, dict):
        for k in ('values', 'vector', 'data', 'state'):
            a = vec(s.get(k))
            if a is not None and a.size >= 8:
                return a[:8]
        for v in s.values():
            a = vec(v)
            if a is not None and a.size == 8:
                return a
        p = vec(s.get('ee_pos') or s.get('position') or s.get('end_effector_position'), 3)
        if p is not None:
            return np.concatenate([p, np.zeros(5)])
    return None


def get_step_id(meta):
    v = meta.get('step_id') if isinstance(meta, dict) else None
    if v is None:
        v = bfs_find(meta, ['step_id'])
    try:
        return int(v)
    except Exception:
        return None


def get_instruction(meta):
    for k in ('instruction', 'language_instruction', 'goal', 'task_instruction'):
        v = bfs_find(meta, [k])
        if isinstance(v, str) and v.strip():
            return v
    return ''


def get_flags(meta):
    ended = as_bool(bfs_find(meta, ['episode_ended']))
    tc = bfs_find(meta, ['task_complete'])
    complete = as_bool(tc)
    for k in ('terminated', 'truncated', 'done'):
        v = bfs_find(meta, [k])
        if isinstance(v, bool) and v:
            ended = True
    return ended, complete


def get_remaining_steps(meta):
    lim = bfs_find(meta, ['episode_limits'])
    if not isinstance(lim, dict):
        return None, None
    rem = None
    mx = None
    used = None
    for k, v in lim.items():
        kl = k.lower()
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if any(w in kl for w in ('second', 'wall', 'time', '_s', 'sec')) and 'step' not in kl:
                continue
            if 'remain' in kl and rem is None:
                rem = int(v)
            elif ('max' in kl or 'limit' in kl or 'horizon' in kl) and mx is None:
                mx = int(v)
            elif ('used' in kl or 'elapsed' in kl or 'taken' in kl or kl in ('step', 'steps', 'step_count')) and used is None:
                used = int(v)
    if rem is None and mx is not None and used is not None:
        rem = mx - used
    return rem, mx


# ----------------------------------------------------------------- images
def _decode_b64_image(data):
    raw = base64.b64decode(data)
    im = Image.open(io.BytesIO(raw))
    im.load()
    return im


def _im_to_arrays(im):
    """Return ('rgb', HxWx3 uint8) or ('depth', HxW float metres) or (None, None)."""
    mode = im.mode
    if mode in ('I;16', 'I;16B', 'I;16L', 'I;16N', 'I'):
        a = np.array(im).astype(np.float64)
        return 'depth', a * 0.001
    if mode in ('F',):
        a = np.array(im).astype(np.float64)
        return 'depth', a
    if mode in ('RGB', 'RGBA', 'P', 'L', 'LA'):
        return 'rgb', np.array(im.convert('RGB'))
    return None, None


def _camera_of(d):
    if not isinstance(d, dict):
        return None
    for k in ('camera', 'camera_name', 'view', 'name', 'sensor'):
        v = d.get(k)
        if isinstance(v, str):
            vl = v.lower()
            if 'wrist' in vl or 'hand' in vl:
                return 'wrist'
            if 'front' in vl or 'base' in vl or 'third' in vl or 'main' in vl:
                return 'front'
    return None


def _kind_of(d):
    if not isinstance(d, dict):
        return None
    for k in ('kind', 'type', 'modality', 'channel'):
        v = d.get(k)
        if isinstance(v, str):
            vl = v.lower()
            if 'depth' in vl:
                return 'depth'
            if 'rgb' in vl or 'color' in vl or 'colour' in vl:
                return 'rgb'
    return None


def load_file_image(path, meters_per_unit=None):
    p = path if os.path.isabs(path) else os.path.join(WS, path)
    if not os.path.exists(p):
        return None, None
    im = Image.open(p)
    im.load()
    kind, arr = _im_to_arrays(im)
    if kind == 'depth' and meters_per_unit:
        arr = np.array(im).astype(np.float64) * float(meters_per_unit)
    return kind, arr


def extract_images(res, meta):
    """Return dict with keys like ('front','rgb') -> array, plus notes."""
    out = {}
    notes = []
    imgs = []
    pending = None  # label from a short text block right before an image
    first_text = True
    for b in blocks(res):
        if isinstance(b, dict) and b.get('type') == 'text':
            txt = b.get('text', '') or ''
            lab = None
            try:
                j = json.loads(txt)
                if isinstance(j, dict) and not first_text:
                    lab = (_camera_of(j), _kind_of(j))
            except Exception:
                tl = txt.lower()
                if len(tl) < 200:
                    cam = 'wrist' if ('wrist' in tl or 'hand' in tl) else ('front' if ('front' in tl or 'third' in tl) else None)
                    kind = 'depth' if 'depth' in tl else ('rgb' if ('rgb' in tl or 'color' in tl) else None)
                    lab = (cam, kind)
            first_text = False
            pending = lab if lab and (lab[0] or lab[1]) else None
            continue
        if isinstance(b, dict) and b.get('type') == 'image' and b.get('data'):
            lab = pending
            pending = None
            try:
                im = _decode_b64_image(b['data'])
                kind, arr = _im_to_arrays(im)
                cam = _camera_of(b.get('_meta')) or _camera_of(b.get('annotations')) or _camera_of(b) or (lab[0] if lab else None)
                k2 = _kind_of(b.get('_meta')) or _kind_of(b.get('annotations')) or (lab[1] if lab else None)
                imgs.append({'kind': kind, 'arr': arr, 'cam': cam, 'k2': k2, 'mode': im.mode})
            except Exception as e:  # noqa
                notes.append('img_decode_err:%r' % (e,))
    # descriptor list in meta whose length matches the number of images
    descs = None
    if imgs:
        cands = bfs_find_all(meta, lambda d: any(isinstance(v, list) and len(v) == len(imgs)
                                                 and all(isinstance(x, dict) and _camera_of(x) for x in v)
                                                 for v in d.values()))
        for d in cands:
            for v in d.values():
                if isinstance(v, list) and len(v) == len(imgs) and all(isinstance(x, dict) and _camera_of(x) for x in v):
                    kinds = [_kind_of(x) for x in v]
                    consistent = all(k is None or k == it['kind'] for k, it in zip(kinds, imgs))
                    both = ('rgb' in kinds) and ('depth' in kinds)
                    if consistent or both:
                        descs = v
                        break
            if descs:
                break
    order_rgb = ['front', 'wrist']
    order_dep = ['front', 'wrist']
    for i, it in enumerate(imgs):
        cam = it['cam']
        kind = it['kind']
        if descs is not None:
            cam = cam or _camera_of(descs[i])
            dk = _kind_of(descs[i])
            if dk == 'depth' and kind == 'rgb':
                kind = 'depth_vis'  # a visualisation, not metric
        if it['k2'] == 'depth' and kind == 'rgb':
            kind = 'depth_vis'
        if kind == 'rgb':
            if cam is None:
                cam = order_rgb.pop(0) if order_rgb else None
            elif cam in order_rgb:
                order_rgb.remove(cam)
        elif kind == 'depth':
            if cam is None:
                cam = order_dep.pop(0) if order_dep else None
            elif cam in order_dep:
                order_dep.remove(cam)
        if cam and kind in ('rgb', 'depth') and (cam, kind) not in out:
            out[(cam, kind)] = it['arr']
        notes.append('blk%d:%s/%s/%s' % (i, it['mode'], kind, cam))
    # file descriptors (camera, kind, path) -- take the last occurrence per (cam, kind)
    fds = bfs_find_all(meta, lambda d: isinstance(d.get('path'), str) and _camera_of(d) and _kind_of(d))
    last = {}
    for d in fds:
        last[(_camera_of(d), _kind_of(d))] = d
    for key, d in last.items():
        if key[1] == 'depth' and key in out:
            continue
        if key[1] == 'rgb' and key in out:
            continue
        try:
            kind, arr = load_file_image(d['path'], d.get('meters_per_unit'))
            if kind == key[1] and arr is not None:
                out[key] = arr
                notes.append('file:%s/%s' % key)
        except Exception as e:  # noqa
            notes.append('file_err:%r' % (e,))
    return out, notes


# ----------------------------------------------------------------- calibration
def quat_wxyz_to_R(q):
    w, x, y, z = [float(t) for t in q]
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def parse_K(x):
    if x is None:
        return None
    if isinstance(x, dict):
        keys = {k.lower(): v for k, v in x.items()}
        if all(k in keys for k in ('fx', 'fy', 'cx', 'cy')):
            try:
                return np.array([[float(keys['fx']), 0, float(keys['cx'])],
                                 [0, float(keys['fy']), float(keys['cy'])], [0, 0, 1.0]])
            except Exception:
                return None
        for k in ('matrix', 'k', 'intrinsic_matrix', 'intrinsics', 'intrinsic_cv', 'camera_matrix', 'values', 'data'):
            if k in keys:
                r = parse_K(keys[k])
                if r is not None:
                    return r
        return None
    a = vec(x)
    if a is None:
        return None
    if a.size == 9:
        return a.reshape(3, 3)
    if a.size == 4:
        return np.array([[a[0], 0, a[2]], [0, a[1], a[3]], [0, 0, 1.0]])
    if a.size == 12:
        return a.reshape(3, 4)[:, :3]
    return None


def parse_pose(x):
    """Return (R, t, convention_string_or_None) or None."""
    if x is None:
        return None
    if isinstance(x, dict):
        keys = {k.lower(): v for k, v in x.items()}
        conv = None
        for k in ('convention', 'frame_convention', 'camera_convention', 'axes'):
            if isinstance(keys.get(k), str):
                conv = keys[k].lower()
        t = None
        for k in ('position_m', 'position', 'p', 'translation', 't', 'pos', 'xyz'):
            if k in keys:
                t = vec(keys[k], 3)
                if t is not None:
                    break
        q = None
        for k in ('quaternion_wxyz', 'wxyz', 'quaternion', 'q', 'quat', 'orientation', 'rotation'):
            if k in keys:
                q = vec(keys[k], 4)
                if q is not None:
                    break
        if t is not None and q is not None:
            return quat_wxyz_to_R(q), t, conv
        for k in ('matrix', 'cam2world', 'cam2world_gl', 'pose', 'transform', 'extrinsic', 'model_matrix', 'values'):
            if k in keys:
                r = parse_pose(keys[k])
                if r is not None:
                    return r[0], r[1], conv or r[2] or ('opencv' if 'cv' in k else None)
        return None
    a = vec(x)
    if a is None:
        return None
    if a.size == 16:
        M = a.reshape(4, 4)
        return M[:3, :3], M[:3, 3], None
    if a.size == 12:
        M = a.reshape(3, 4)
        return M[:, :3], M[:, 3], None
    if a.size == 7:
        return quat_wxyz_to_R(a[3:7]), a[:3], None
    return None


def _per_camera(container, parser):
    out = {}
    if isinstance(container, dict):
        for k, v in container.items():
            cam = 'wrist' if ('wrist' in k.lower() or 'hand' in k.lower()) else (
                'front' if ('front' in k.lower() or 'base' in k.lower() or 'third' in k.lower()) else None)
            if cam is None and isinstance(v, dict):
                cam = _camera_of(v)
            if cam is None:
                continue
            r = parser(v)
            if r is not None:
                out[cam] = r
        if not out:
            cam = _camera_of(container)
            r = parser(container)
            if cam and r is not None:
                out[cam] = r
    elif isinstance(container, list):
        for v in container:
            cam = _camera_of(v)
            if cam:
                r = parser(v)
                if r is not None:
                    out[cam] = r
    return out


def get_calib(meta):
    Ks = _per_camera(bfs_find(meta, ['camera_intrinsics', 'intrinsics']), parse_K)
    Ps = _per_camera(bfs_find(meta, ['camera_poses', 'camera_pose', 'extrinsics']), parse_pose)
    return Ks, Ps


# ----------------------------------------------------------------- back-projection
# Variants: (axes, pixel offset, invert pose)
VARIANTS = [(ax, off, inv) for ax in ('gl', 'cv') for off in (0.0, 0.5) for inv in (False, True)]


def backproject(u, v, d, K, pose, variant):
    """u, v pixel coords (arrays), d depth (m). Returns Nx3 world points."""
    ax, off, inv = variant
    R, t = pose[0], pose[1]
    if inv:
        R, t = R.T, -R.T @ t
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xc = (np.asarray(u, float) + off - cx) / fx * d
    yc = (np.asarray(v, float) + off - cy) / fy * d
    if ax == 'gl':
        pc = np.stack([xc, -yc, -np.asarray(d, float)], axis=-1)
    else:
        pc = np.stack([xc, yc, np.asarray(d, float)], axis=-1)
    return pc @ R.T + t


def dense_points(depth, K, pose, variant):
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W]
    P = backproject(uu, vv, depth, K, pose, variant)
    valid = np.isfinite(depth) & (depth > 1e-4) & (depth < 20.0)
    return P, valid


def parse_p2p(res):
    """Parse robomme_pixel_to_position reply -> list of (xyz or None, spread or None)."""
    meta = meta_of(res)
    lists = []

    def has_pt(d):
        return _pt_of(d) is not None or ('valid' in d and ('u' in d or 'camera' in d))
    for d in bfs_find_all(meta, lambda d: True):
        for v in d.values():
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v) and any(has_pt(x) for x in v):
                lists.append(v)
    if isinstance(meta, dict) is False:
        return []
    if not lists:
        return []
    L = max(lists, key=len)
    out = []
    for x in L:
        p = _pt_of(x)
        ok = x.get('valid', True)
        if not as_bool(ok) and ok is not True:
            p = None
        sp = x.get('depth_spread_m')
        out.append((p, sp))
    return out


def _pt_of(d):
    if not isinstance(d, dict):
        return None
    for k in ('position', 'world_position', 'point', 'xyz', 'world', 'world_point', 'position_m', 'world_xyz'):
        if k in d:
            p = vec(d[k], 3)
            if p is not None:
                return p
            if isinstance(d[k], dict):
                q = d[k]
                if all(c in q for c in ('x', 'y', 'z')):
                    return vec([q['x'], q['y'], q['z']], 3)
    if all(c in d for c in ('x', 'y', 'z')):
        return vec([d['x'], d['y'], d['z']], 3)
    return None
