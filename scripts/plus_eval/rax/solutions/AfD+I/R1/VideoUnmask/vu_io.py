"""Defensive parsing of RAX/RoboMME MCP replies and frame manifests.

The exact reply layout was never observed before the frozen run (R1 budget), so
every accessor searches the JSON recursively and has fallbacks.
"""
import base64
import io
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

WORKSPACE = Path('/workspace')


# ----------------------------------------------------------------- JSON utils
def reply_json(result):
    """Merge structuredContent and every JSON text block of an MCP result."""
    out = {}
    if not isinstance(result, dict):
        return out
    sc = result.get('structuredContent')
    if isinstance(sc, dict):
        out.update(sc)
    for b in result.get('content', []) or []:
        if isinstance(b, dict) and b.get('type') == 'text':
            t = b.get('text', '')
            try:
                j = json.loads(t)
            except Exception:
                continue
            if isinstance(j, dict):
                for k, v in j.items():
                    out.setdefault(k, v)
    return out


def walk(obj, depth=0, maxdepth=12):
    """Yield (key, value, parent) for every dict entry, breadth-first-ish."""
    if depth > maxdepth:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v, obj
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                yield from walk(v, depth + 1, maxdepth)
    elif isinstance(obj, list):
        for v in obj[:600]:
            if isinstance(v, (dict, list)):
                yield from walk(v, depth + 1, maxdepth)


def find_key(obj, names, pred=None):
    names = [n.lower() for n in names]
    for k, v, _ in walk(obj):
        if isinstance(k, str) and k.lower() in names:
            if pred is None or pred(v):
                return v
    return None


def is_num_list(v, n=None):
    if not isinstance(v, (list, tuple)):
        return False
    if n is not None and len(v) != n:
        return False
    try:
        [float(x) for x in v]
        return len(v) > 0 and not any(isinstance(x, (list, dict)) for x in v)
    except Exception:
        return False


def get_state(js):
    v = js.get('state') if isinstance(js, dict) else None
    if is_num_list(v, 8):
        return np.array(v, float)
    if isinstance(v, dict):
        for k in ('values', 'vector', 'data', 'state'):
            if is_num_list(v.get(k), 8):
                return np.array(v[k], float)
    v = find_key(js, ['state', 'proprio', 'proprioception', 'robot_state'], lambda x: is_num_list(x, 8))
    if v is not None:
        return np.array(v, float)
    return None


def get_step_id(js):
    for k in ('step_id', 'step', 'observation_step_id'):
        v = js.get(k) if isinstance(js, dict) else None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    v = find_key(js, ['step_id'], lambda x: isinstance(x, (int, float)) and not isinstance(x, bool))
    return int(v) if v is not None else None


def get_instruction(js):
    v = find_key(js, ['instruction', 'language_instruction', 'goal', 'task_instruction'],
                 lambda x: isinstance(x, str) and len(x) > 5)
    return v or ''


def get_flag(js, names):
    v = find_key(js, names, lambda x: isinstance(x, bool))
    return v


def episode_ended(js):
    v = get_flag(js, ['episode_ended', 'done', 'terminated'])
    return bool(v) if v is not None else False


def task_complete(js):
    v = get_flag(js, ['task_complete', 'success', 'is_success'])
    return bool(v) if v is not None else False


def remaining_steps(js):
    lim = find_key(js, ['episode_limits'])
    if isinstance(lim, dict):
        for k in ('remaining_steps', 'steps_remaining', 'remaining'):
            v = lim.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
        mx = None
        cur = None
        for k in ('max_steps', 'step_limit', 'max_episode_steps', 'limit'):
            if isinstance(lim.get(k), (int, float)):
                mx = int(lim[k])
        for k in ('steps_used', 'elapsed_steps', 'step', 'step_id', 'used'):
            if isinstance(lim.get(k), (int, float)):
                cur = int(lim[k])
        if mx is not None and cur is not None:
            return mx - cur
    v = find_key(js, ['remaining_steps', 'steps_remaining'],
                 lambda x: isinstance(x, (int, float)) and not isinstance(x, bool))
    return int(v) if v is not None else None


# ------------------------------------------------------------- calibration
def _as_K(v):
    try:
        if isinstance(v, dict):
            lower = {str(k).lower(): x for k, x in v.items()}
            if all(k in lower for k in ('fx', 'fy', 'cx', 'cy')):
                return np.array([[float(lower['fx']), 0, float(lower['cx'])],
                                 [0, float(lower['fy']), float(lower['cy'])], [0, 0, 1.0]])
            for k in ('k', 'matrix', 'intrinsic_cv', 'intrinsics', 'camera_matrix', 'intrinsic'):
                if k in lower:
                    K = _as_K(lower[k])
                    if K is not None:
                        return K
            return None
        a = np.array(v, float)
        if a.shape == (3, 3):
            return a
        if a.size == 9:
            return a.reshape(3, 3)
        if a.size == 4:
            fx, fy, cx, cy = a
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    except Exception:
        return None
    return None


def _cam_entry(container, cam):
    """container is a dict keyed by camera name or a list of dicts with a camera field."""
    if isinstance(container, dict):
        for k, v in container.items():
            if isinstance(k, str) and cam in k.lower():
                return v
        if any(isinstance(k, str) and k.lower() in ('camera', 'name') for k in container):
            nm = str(container.get('camera', container.get('name', ''))).lower()
            if cam in nm:
                return container
    if isinstance(container, list):
        for e in container:
            if isinstance(e, dict):
                nm = str(e.get('camera', e.get('name', e.get('id', '')))).lower()
                if cam in nm:
                    return e
    return None


def get_intrinsics(js, cam):
    c = find_key(js, ['camera_intrinsics', 'intrinsics'])
    e = _cam_entry(c, cam) if c is not None else None
    if e is not None:
        K = _as_K(e)
        if K is not None:
            return K
    # search any dict keyed by cam containing intrinsics
    for k, v, parent in walk(js):
        if isinstance(k, str) and cam in k.lower() and isinstance(v, dict):
            for kk in ('intrinsics', 'camera_intrinsics', 'K', 'intrinsic_cv'):
                if kk in v:
                    K = _as_K(v[kk])
                    if K is not None:
                        return K
    return None


def _as_pose(v):
    """Return (R_wxyz_candidates, t, convention_str) or None.  Returns dict."""
    try:
        if isinstance(v, dict):
            lower = {str(k).lower(): x for k, x in v.items()}
            conv = ''
            for k in ('convention', 'frame_convention', 'camera_convention', 'axes'):
                if isinstance(lower.get(k), str):
                    conv = lower[k].lower()
            pos = None
            for k in ('position_m', 'position', 'p', 'pos', 'translation', 't', 'xyz'):
                if is_num_list(lower.get(k), 3):
                    pos = np.array(lower[k], float)
                    break
            quat = None
            qorder = 'wxyz'
            for k in ('quaternion_wxyz', 'quat_wxyz', 'wxyz', 'orientation_wxyz'):
                if is_num_list(lower.get(k), 4):
                    quat = np.array(lower[k], float)
                    qorder = 'wxyz'
                    break
            if quat is None:
                for k in ('quaternion_xyzw', 'quat_xyzw', 'xyzw'):
                    if is_num_list(lower.get(k), 4):
                        quat = np.array(lower[k], float)
                        qorder = 'xyzw'
                        break
            if quat is None:
                for k in ('quaternion', 'quat', 'q', 'orientation', 'rotation'):
                    if is_num_list(lower.get(k), 4):
                        quat = np.array(lower[k], float)
                        qorder = 'wxyz?'
                        break
            for k in ('matrix', 'transform', 'pose', 'cam2world', 'cam2world_gl', 'T', 'extrinsic', 'model_matrix'):
                if k.lower() in lower:
                    a = np.array(lower[k.lower()], float)
                    if a.size == 16:
                        return {'T': a.reshape(4, 4), 'conv': conv}
            if pos is not None and quat is not None:
                return {'pos': pos, 'quat': quat, 'qorder': qorder, 'conv': conv}
            for k in ('pose', 'camera_pose', 'world_pose'):
                if k in lower:
                    r = _as_pose(lower[k])
                    if r is not None:
                        if not r.get('conv'):
                            r['conv'] = conv
                        return r
            return None
        a = np.array(v, float)
        if a.size == 16:
            return {'T': a.reshape(4, 4), 'conv': ''}
        if a.size == 7:
            return {'pos': a[:3], 'quat': a[3:], 'qorder': 'wxyz?', 'conv': ''}
    except Exception:
        return None
    return None


def get_pose(js, cam):
    c = find_key(js, ['camera_poses', 'camera_pose', 'extrinsics', 'camera_extrinsics'])
    e = _cam_entry(c, cam) if c is not None else None
    if e is not None:
        p = _as_pose(e)
        if p is not None:
            if not p.get('conv'):
                gc = find_key(js, ['convention', 'camera_convention', 'pose_convention'],
                              lambda x: isinstance(x, str))
                p['conv'] = (gc or '').lower()
            return p
    return None


# ------------------------------------------------------------------ images
def load_rgb(path):
    p = Path(path)
    if not p.is_absolute():
        p = WORKSPACE / p
    with Image.open(p) as im:
        return np.array(im.convert('RGB'))


def load_depth(path, mpu=0.001):
    p = Path(path)
    if not p.is_absolute():
        p = WORKSPACE / p
    with Image.open(p) as im:
        a = np.array(im)
    if a.ndim == 3:
        a = a[..., 0]
    is_float = a.dtype.kind == 'f'
    a = a.astype(np.float64)
    if is_float and np.nanmax(a) < 50:
        d = a.copy()            # already metres
    else:
        d = a * float(mpu if mpu else 0.001)
    d[~(a > 0)] = np.nan
    return d


def frame_files(frame):
    """{(camera, kind): (path, meters_per_unit)} for a manifest frame."""
    out = {}
    for f in frame.get('files', []) or []:
        if not isinstance(f, dict):
            continue
        cam = str(f.get('camera', '')).lower()
        kind = str(f.get('kind', '')).lower()
        path = f.get('path')
        if not path:
            continue
        cam = 'front' if 'front' in cam else ('wrist' if ('wrist' in cam or 'hand' in cam) else cam)
        kind = 'depth' if 'depth' in kind else ('rgb' if ('rgb' in kind or 'color' in kind) else kind)
        out[(cam, kind)] = (path, f.get('meters_per_unit', 0.001) or 0.001)
    return out


def load_manifest(js):
    fs = find_key(js, ['frame_sequence'])
    mp = None
    if isinstance(fs, dict):
        mp = fs.get('manifest_path') or find_key(fs, ['manifest_path', 'manifest'],
                                                 lambda x: isinstance(x, str))
    if mp is None:
        mp = find_key(js, ['manifest_path'], lambda x: isinstance(x, str))
    if not mp:
        return None, None
    p = Path(mp)
    if not p.is_absolute():
        p = WORKSPACE / p
    try:
        return json.loads(p.read_text()), str(p)
    except Exception:
        return None, str(p)


def sorted_frames(manifest):
    fr = [f for f in (manifest or {}).get('frames', []) if isinstance(f, dict)]

    def key(f):
        fi = f.get('frame_index')
        return (fi if isinstance(fi, (int, float)) else 0)
    return sorted(fr, key=key)


def decode_image_blocks(result):
    """Decode inline image blocks.  Returns list of (label, np.array, mode)."""
    out = []
    last_text = ''
    for b in (result or {}).get('content', []) or []:
        if not isinstance(b, dict):
            continue
        if b.get('type') == 'text':
            last_text = b.get('text', '')[:400].lower()
            continue
        if b.get('type') != 'image':
            continue
        try:
            raw = base64.b64decode(b.get('data', ''))
            with Image.open(io.BytesIO(raw)) as im:
                mode = im.mode
                arr = np.array(im)
        except Exception:
            continue
        label = ''
        for k in ('annotations', '_meta', 'meta', 'name', 'title', 'camera'):
            if k in b:
                label += json.dumps(b[k]).lower()
        label += ' ' + last_text[-120:]
        out.append((label, arr, mode))
    return out


def current_images(result, js):
    """Best effort: {'front': {'rgb':..., 'depth':...}, 'wrist': {...}} for the reply's observation."""
    imgs = {'front': {}, 'wrist': {}}
    step = get_step_id(js)
    # 1) manifest frame matching this step (last execution frame)
    man, _ = load_manifest(js)
    if man:
        frs = sorted_frames(man)
        cand = [f for f in frs if str(f.get('phase', '')).lower().startswith('exec')] or frs
        pick = None
        if step is not None:
            same = [f for f in cand if f.get('step_id') == step]
            if same:
                pick = same[-1]
        if pick is None and cand:
            pick = cand[-1]
            if step is not None and isinstance(pick.get('step_id'), int) and pick.get('step_id') != step:
                pick = None
        if pick is not None:
            ff = frame_files(pick)
            for (cam, kind), (path, mpu) in ff.items():
                if cam in imgs:
                    try:
                        imgs[cam][kind] = load_depth(path, mpu) if kind == 'depth' else load_rgb(path)
                    except Exception:
                        pass
    # 2) any explicit file entries inside the JSON
    if not all('depth' in imgs[c] and 'rgb' in imgs[c] for c in imgs):
        for k, v, parent in walk(js):
            if isinstance(v, dict) and 'path' in v and ('camera' in v or 'kind' in v):
                ff = frame_files({'files': [v]})
                for (cam, kind), (path, mpu) in ff.items():
                    if cam in imgs and kind not in imgs[cam]:
                        try:
                            imgs[cam][kind] = load_depth(path, mpu) if kind == 'depth' else load_rgb(path)
                        except Exception:
                            pass
    # 3) inline image blocks
    if not all('depth' in imgs[c] and 'rgb' in imgs[c] for c in imgs):
        blocks = decode_image_blocks(result)
        rgbs = [(l, a) for l, a, m in blocks if a.ndim == 3]
        deps = [(l, a) for l, a, m in blocks if a.ndim == 2 and a.dtype != np.uint8]
        for lst, kind in ((rgbs, 'rgb'), (deps, 'depth')):
            assigned = {}
            for l, a in lst:
                if 'wrist' in l or 'hand' in l:
                    assigned.setdefault('wrist', a)
                elif 'front' in l or 'third' in l:
                    assigned.setdefault('front', a)
            rest = [a for l, a in lst if not any(a is x for x in assigned.values())]
            for cam in ('front', 'wrist'):
                if cam not in assigned and rest:
                    assigned[cam] = rest.pop(0)
            for cam, a in assigned.items():
                if kind not in imgs[cam]:
                    if kind == 'depth':
                        d = a.astype(np.float64) * 0.001
                        d[a <= 0] = np.nan
                        imgs[cam][kind] = d
                    else:
                        imgs[cam][kind] = a[..., :3].astype(np.uint8)
    return imgs


def summarize(obj, maxlen=1500):
    """Compact structural summary for logs (no image data)."""
    def sk(o, d=0):
        if d > 4:
            return '…'
        if isinstance(o, dict):
            return {k: sk(v, d + 1) for k, v in list(o.items())[:30]}
        if isinstance(o, list):
            if len(o) > 8 and all(not isinstance(x, (dict, list)) for x in o):
                return f'[{len(o)} scalars: {o[:3]}…]'
            return [sk(x, d + 1) for x in o[:3]] + ([f'…{len(o)} items'] if len(o) > 3 else [])
        if isinstance(o, str):
            return o[:100]
        return o
    try:
        s = json.dumps(sk(obj), default=str)
    except Exception as e:
        s = f'<unserializable {e}>'
    return s[:maxlen]
