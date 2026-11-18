"""Defensive parsing of MCP replies, manifests and frame files."""
import json, os, time
import numpy as np
from PIL import Image

WS = '/workspace'
LOG_PATH = '/workspace/output/videorepick_log.txt'
_t0 = time.time()

def log(*a):
    msg = '[%7.1fs] ' % (time.time() - _t0) + ' '.join(str(x) for x in a)
    try:
        print(msg, flush=True)
    except Exception:
        pass
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, 'a') as f:
            f.write(msg + '\n')
    except Exception:
        pass

def reply_json(result):
    """First JSON text block of an MCP result (dict)."""
    if isinstance(result, dict) and 'content' in result:
        for b in result.get('content') or []:
            if isinstance(b, dict) and b.get('type') == 'text':
                try:
                    return json.loads(b.get('text', ''))
                except Exception:
                    continue
        if 'structuredContent' in result and isinstance(result['structuredContent'], dict):
            return result['structuredContent']
        return {}
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            return json.loads(result)
        except Exception:
            return {}
    return {}

def find(d, key, default=None, maxdepth=8):
    """Depth-first search for the first occurrence of key (breadth by level)."""
    level = [d]
    for _ in range(maxdepth):
        nxt = []
        for x in level:
            if isinstance(x, dict):
                if key in x:
                    return x[key]
                nxt.extend(v for v in x.values() if isinstance(v, (dict, list)))
            elif isinstance(x, list):
                nxt.extend(v for v in x if isinstance(v, (dict, list)))
        level = nxt
        if not level:
            break
    return default

def is_num_list(x, n=None):
    if not isinstance(x, (list, tuple)):
        return False
    if n is not None and len(x) != n:
        return False
    try:
        return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in x) and len(x) > 0
    except Exception:
        return False

def get_state(d):
    s = find(d, 'state')
    if is_num_list(s) and len(s) >= 8:
        return np.array(s[:8], float)
    if isinstance(s, dict):
        for k in ('values', 'vector', 'data', 'state'):
            if is_num_list(s.get(k)) and len(s[k]) >= 8:
                return np.array(s[k][:8], float)
        # composed
        pos = s.get('ee_position') or s.get('position')
        rot = s.get('ee_axis_angle') or s.get('axis_angle') or s.get('orientation')
        fin = s.get('gripper') or s.get('fingers') or s.get('finger_positions')
        if is_num_list(pos, 3) and is_num_list(rot, 3):
            f = list(fin) if is_num_list(fin) else [0.04, 0.04]
            if len(f) == 1:
                f = [f[0] / 2, f[0] / 2]
            return np.array(list(pos) + list(rot) + f[:2], float)
    return None

def get_step_id(d):
    for k in ('step_id', 'observation_step_id'):
        v = find(d, k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    return None

def get_bool(d, key):
    v = find(d, key)
    return bool(v) if isinstance(v, (bool, int)) else False

def episode_over(d):
    if get_bool(d, 'episode_ended') or get_bool(d, 'task_complete'):
        return True
    for k in ('terminated', 'truncated', 'done'):
        if find(d, k) is True:
            return True
    return False

def remaining_steps(d):
    lim = find(d, 'episode_limits')
    if isinstance(lim, dict):
        for k in ('remaining_steps', 'steps_remaining', 'remaining'):
            v = lim.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
        mx = None; used = None
        for k in ('max_steps', 'step_limit', 'max_episode_steps', 'limit'):
            if isinstance(lim.get(k), (int, float)):
                mx = int(lim[k]); break
        for k in ('steps_used', 'step', 'elapsed_steps', 'used'):
            if isinstance(lim.get(k), (int, float)):
                used = int(lim[k]); break
        if mx is not None and used is not None:
            return mx - used
        if mx is not None:
            sid = get_step_id(d)
            if sid is not None:
                return mx - sid
    return None

def manifest_path(d):
    fs = find(d, 'frame_sequence')
    if isinstance(fs, dict):
        p = fs.get('manifest_path') or fs.get('path')
        if isinstance(p, str):
            return p
    p = find(d, 'manifest_path')
    return p if isinstance(p, str) else None

def resolve(path, base=None):
    if os.path.isabs(path):
        return path
    p = os.path.join(WS, path)
    if os.path.exists(p) or base is None:
        return p
    q = os.path.join(base, path)
    return q if os.path.exists(q) else p

def load_manifest(path):
    full = resolve(path)
    with open(full) as f:
        m = json.load(f)
    frames = m.get('frames') if isinstance(m, dict) else m
    if not isinstance(frames, list):
        frames = []
    base = os.path.dirname(full)
    out = []
    for i, fr in enumerate(frames):
        if not isinstance(fr, dict):
            continue
        files = {}
        for fe in fr.get('files') or []:
            if not isinstance(fe, dict):
                continue
            cam = str(fe.get('camera', '')).lower(); kind = str(fe.get('kind', '')).lower()
            if 'path' in fe:
                files[(cam, kind)] = (resolve(fe['path'], base), fe.get('meters_per_unit', 0.001))
        idx = fr.get('frame_index', i)
        st = fr.get('state')
        st = np.array(st[:8], float) if is_num_list(st) and len(st) >= 8 else None
        out.append({'idx': idx if isinstance(idx, (int, float)) else i, 'phase': str(fr.get('phase', '')),
                    'ts': fr.get('timestamp_s'), 'src': fr.get('source_frame_index'), 'step_id': fr.get('step_id'),
                    'state': st, 'files': files, 'raw': fr})
    out.sort(key=lambda r: r['idx'])
    return out, m

def cam_key(files, cam):
    for (c, k) in files:
        if c == cam or c.startswith(cam):
            return c
    return None

def load_rgb(files, cam):
    c = cam_key(files, cam)
    if c is None or (c, 'rgb') not in files:
        return None
    with Image.open(files[(c, 'rgb')][0]) as im:
        return np.array(im.convert('RGB'))

def load_depth(files, cam):
    c = cam_key(files, cam)
    if c is None:
        return None
    key = (c, 'depth') if (c, 'depth') in files else None
    if key is None:
        return None
    path, mpu = files[key]
    try:
        mpu = float(mpu) if mpu is not None else 0.001
    except Exception:
        mpu = 0.001
    if path.endswith('.npy'):
        a = np.load(path).astype(np.float32)
        if a.max() > 50:
            a = a * mpu
        return a
    with Image.open(path) as im:
        a = np.array(im)
    if a.ndim == 3:
        a = a[..., 0]
    a = a.astype(np.float32)
    if a.dtype != np.float32 or a.max() > 50:
        a = a * mpu
    return a
