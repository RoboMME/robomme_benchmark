"""Defensive parsing of RoboMME-Plus MCP replies (formats only partly documented).

Everything here tolerates missing fields: it searches the JSON metadata recursively for
known keys and falls back to heuristics. Nothing raises on an unexpected layout; callers
check for None.
"""
import base64
import io
import json
import math
import os
import time

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

WORKSPACE = '/workspace'
LOG_PATH = '/workspace/output/controller_log.txt'
_T0 = time.time()


def log(*args):
    msg = '[%7.1fs] ' % (time.time() - _T0) + ' '.join(str(a) for a in args)
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


def elapsed():
    return time.time() - _T0


# ----------------------------------------------------------------------------- JSON utils

def _blocks(result):
    if isinstance(result, dict):
        c = result.get('content')
        if isinstance(c, list):
            return c
    if isinstance(result, list):
        return result
    return []


def _try_json(text):
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        i, j = text.index('{'), text.rindex('}')
        return json.loads(text[i:j + 1])
    except Exception:
        return None


def meta_of(result):
    """All JSON metadata in a reply merged into a list of objects (structured first)."""
    metas = []
    if isinstance(result, dict) and isinstance(result.get('structuredContent'), (dict, list)):
        metas.append(result['structuredContent'])
    for b in _blocks(result):
        if isinstance(b, dict) and b.get('type') == 'text' and isinstance(b.get('text'), str):
            j = _try_json(b['text'])
            if j is not None:
                metas.append(j)
    return metas


def walk(obj, depth=0, maxdepth=12):
    """Yield (key, value, depth) for every dict entry, breadth-ish (sorted by depth later)."""
    if depth > maxdepth:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v, depth
            yield from walk(v, depth + 1, maxdepth)
    elif isinstance(obj, list):
        for v in obj[:600]:
            yield from walk(v, depth + 1, maxdepth)


def find_all(metas, key):
    out = []
    for m in metas:
        for k, v, d in walk(m):
            if k == key:
                out.append((d, v))
    out.sort(key=lambda t: t[0])
    return [v for d, v in out]


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _numlist(v, n=None):
    if isinstance(v, (list, tuple)) and v and all(_is_num(x) for x in v):
        if n is None or len(v) == n:
            return [float(x) for x in v]
    return None


# ----------------------------------------------------------------------------- reply fields

def find_state(metas):
    for v in find_all(metas, 'state'):
        s = _numlist(v)
        if s is not None and len(s) >= 8:
            return s[:8]
        if isinstance(v, dict):
            for kk in ('values', 'vector', 'data', 'value'):
                s = _numlist(v.get(kk))
                if s is not None and len(s) >= 8:
                    return s[:8]
            # composed state
            p = None
            for kk in ('ee_position', 'position', 'tcp_position', 'end_effector_position'):
                p = _numlist(v.get(kk), 3) or p
            o = None
            for kk in ('ee_axis_angle', 'axis_angle', 'orientation', 'rotation', 'ee_orientation'):
                o = _numlist(v.get(kk), 3) or o
            g = None
            for kk in ('gripper', 'fingers', 'finger_positions', 'gripper_fingers'):
                g = _numlist(v.get(kk), 2) or g
            if p and o:
                return p + o + (g or [float('nan'), float('nan')])
    return None


def find_step_id(metas):
    ids = [v for v in find_all(metas, 'step_id') if isinstance(v, int) and not isinstance(v, bool)]
    if ids:
        return max(ids)
    for key in ('elapsed_steps', 'step', 'steps_taken', 'current_step'):
        ids = [v for v in find_all(metas, key) if isinstance(v, int) and not isinstance(v, bool)]
        if ids:
            return max(ids)
    return None


def find_flag(metas, keys):
    for key in keys:
        for v in find_all(metas, key):
            if v is True:
                return True
    return False


def episode_ended(metas):
    return find_flag(metas, ('episode_ended', 'terminated', 'truncated', 'done'))


def task_complete(metas):
    return find_flag(metas, ('task_complete', 'episode_success'))


def remaining_steps(metas):
    best = None
    for lim in find_all(metas, 'episode_limits'):
        if isinstance(lim, dict):
            for k, v in lim.items():
                if _is_num(v) and 'remain' in k.lower():
                    best = int(v) if best is None else min(best, int(v))
            if best is None:
                mx = None; used = None
                for k, v in lim.items():
                    kl = k.lower()
                    if _is_num(v) and ('max' in kl or 'limit' in kl):
                        mx = int(v)
                    if _is_num(v) and ('elapsed' in kl or 'used' in kl or kl == 'step' or 'taken' in kl):
                        used = int(v)
                if mx is not None and used is not None:
                    best = mx - used
    if best is None:
        for key in ('remaining_steps', 'steps_remaining'):
            for v in find_all(metas, key):
                if _is_num(v):
                    return int(v)
    return best


def max_steps(metas):
    for lim in find_all(metas, 'episode_limits'):
        if isinstance(lim, dict):
            for k, v in lim.items():
                kl = k.lower()
                if _is_num(v) and ('max' in kl or 'limit' in kl or 'horizon' in kl) and 'remain' not in kl:
                    return int(v)
    for key in ('max_steps', 'step_limit', 'max_episode_steps'):
        for v in find_all(metas, key):
            if _is_num(v):
                return int(v)
    return None


# ----------------------------------------------------------------------------- images

def _decode_image(data_b64):
    if Image is None:
        return None, None
    try:
        raw = base64.b64decode(data_b64)
        im = Image.open(io.BytesIO(raw))
        im.load()
        mode = im.mode
        if mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F'):
            return 'depth', np.asarray(im).astype(np.float32)
        if mode == 'L':
            return 'gray', np.asarray(im)
        return 'rgb', np.asarray(im.convert('RGB'))
    except Exception:
        return None, None


def _load_file_rgb(path):
    if Image is None:
        return None
    try:
        p = path if os.path.isabs(path) else os.path.join(WORKSPACE, path)
        with Image.open(p) as im:
            if im.mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F'):
                return None
            return np.asarray(im.convert('RGB'))
    except Exception:
        return None


def _camera_label(text):
    t = (text or '').lower()
    if 'wrist' in t or 'hand' in t:
        return 'wrist'
    if 'front' in t or 'third' in t or 'base' in t:
        return 'front'
    return None


def inline_images(result, metas):
    """Return list of dicts {kind, cam(label or None), img} in attachment order."""
    out = []
    last_text = None
    for b in _blocks(result):
        if not isinstance(b, dict):
            continue
        if b.get('type') == 'text':
            last_text = b.get('text')
            continue
        if b.get('type') == 'image' and isinstance(b.get('data'), str):
            kind, img = _decode_image(b['data'])
            if img is None:
                continue
            lab = None
            # annotations / meta on the block itself
            for k in ('annotations', '_meta', 'meta', 'name', 'title'):
                if k in b:
                    lab = _camera_label(json.dumps(b[k]) if not isinstance(b[k], str) else b[k]) or lab
            if lab is None and last_text and len(last_text) < 200:
                lab = _camera_label(last_text)
            out.append(dict(kind=kind, cam=lab, img=img))
            last_text = None
    # metadata list describing attachments, same count
    if out and any(o['cam'] is None for o in out):
        for m in metas:
            for k, v, d in walk(m):
                if isinstance(v, list) and len(v) == len(out) and all(isinstance(e, dict) and 'camera' in e for e in v):
                    for o, e in zip(out, v):
                        lab = _camera_label(str(e.get('camera')))
                        if lab:
                            o['cam'] = lab
                        kd = str(e.get('kind', e.get('type', ''))).lower()
                        if 'depth' in kd:
                            o['kind'] = 'depth' if o['kind'] != 'rgb' else o['kind']
                    break
    return out


def manifest_frames(metas):
    """Frames of any frame_sequence manifest referenced by the reply (sorted)."""
    frames = []
    for fs in find_all(metas, 'frame_sequence'):
        mp = fs.get('manifest_path') if isinstance(fs, dict) else None
        if not isinstance(mp, str):
            continue
        p = mp if os.path.isabs(mp) else os.path.join(WORKSPACE, mp)
        try:
            with open(p) as f:
                man = json.load(f)
        except Exception:
            continue
        fl = man.get('frames') if isinstance(man, dict) else None
        if isinstance(fl, list):
            for i, fr in enumerate(fl):
                if isinstance(fr, dict):
                    fr = dict(fr)
                    fr.setdefault('frame_index', i)
                    frames.append(fr)
    frames.sort(key=lambda f: (f.get('frame_index') if isinstance(f.get('frame_index'), int) else 0))
    return frames


def frame_rgb(frame, cam):
    files = frame.get('files') or []
    for f in files:
        if isinstance(f, dict) and f.get('camera') == cam and f.get('kind', 'rgb') == 'rgb' and isinstance(f.get('path'), str):
            return _load_file_rgb(f['path'])
    return None


def _dark_bottom(img):
    a = img.astype(np.int16)
    b = a[int(img.shape[0] * 0.7):]
    return float((b.max(axis=2) < 70).mean())


def assign_rgb(result, metas, ref_front=None, step_id=None, order_hint=None):
    """Return ({'front': img|None, 'wrist': img|None}, how, raw_unlabelled_list).

    how: 'labels' (explicit), 'manifest', 'hint' (remembered order), 'heuristic' or 'none'.
    order_hint: 'front_first' | 'wrist_first' | None for unlabelled attachment pairs.
    """
    cams = {'front': None, 'wrist': None}
    how = 'none'
    ims = [o for o in inline_images(result, metas) if o['kind'] == 'rgb']
    labelled = [o for o in ims if o['cam'] in cams]
    for o in labelled:
        if cams[o['cam']] is None:
            cams[o['cam']] = o['img']
            how = 'labels'
    unl = [o['img'] for o in ims if o['cam'] not in cams]
    frames = None
    if (cams['front'] is None or cams['wrist'] is None) and unl:
        if len(unl) >= 2 and cams['front'] is None and cams['wrist'] is None:
            a, b = unl[0], unl[1]
            swap = None
            # manifest files of this step identify the cameras exactly
            frames = manifest_frames(metas)
            cand = [f for f in frames if f.get('phase') != 'video_demo']
            if step_id is not None:
                cand = [f for f in cand if f.get('step_id') == step_id] or cand
            if cand:
                mf = frame_rgb(cand[-1], 'front')
                if mf is not None and mf.shape == a.shape:
                    da = np.abs(a.astype(np.int16) - mf).mean(); db = np.abs(b.astype(np.int16) - mf).mean()
                    if abs(da - db) > 1.0:
                        swap = db < da; how = 'manifest'
            if swap is None and order_hint in ('front_first', 'wrist_first'):
                swap = order_hint == 'wrist_first'; how = 'hint'
            if swap is None and ref_front is not None and a.shape == ref_front.shape:
                da = np.abs(a.astype(np.int16) - ref_front).mean()
                db = np.abs(b.astype(np.int16) - ref_front).mean()
                if abs(da - db) > 2.0:
                    swap = db < da; how = 'heuristic'
            if swap is None:
                swap = _dark_bottom(a) > _dark_bottom(b) + 0.05
                how = 'heuristic'
            cams['front'], cams['wrist'] = (b, a) if swap else (a, b)
        else:
            for img in unl:
                k = 'front' if cams['front'] is None else 'wrist'
                if cams[k] is None:
                    cams[k] = img
                    how = how if how != 'none' else 'heuristic'
    # manifest fallback (step 0 / sequences)
    if cams['front'] is None or cams['wrist'] is None:
        if frames is None:
            frames = manifest_frames(metas)
        cand = [f for f in frames if f.get('phase') not in ('video_demo',)]
        if step_id is not None:
            cand = [f for f in cand if f.get('step_id') == step_id] or cand
        if cand:
            fr = cand[-1]
            for k in cams:
                if cams[k] is None:
                    cams[k] = frame_rgb(fr, k)
                    if cams[k] is not None and how == 'none':
                        how = 'manifest'
    return cams, how, unl


# ----------------------------------------------------------------------------- back-projection results

_POS_KEYS = ('position', 'world_position', 'world_point', 'point', 'xyz', 'position_m',
             'world', 'world_xyz', 'point_world', 'position_world', 'world_position_m', 'p_world')


def _point_from(d):
    if not isinstance(d, dict):
        return _numlist(d, 3)
    if d.get('valid') is False:
        return None
    for k in _POS_KEYS:
        if k in d:
            v = d[k]
            p = _numlist(v, 3)
            if p:
                return p
            if isinstance(v, dict):
                if all(_is_num(v.get(c)) for c in 'xyz'):
                    return [float(v['x']), float(v['y']), float(v['z'])]
                for kk in _POS_KEYS:
                    p = _numlist(v.get(kk), 3)
                    if p:
                        return p
    if all(_is_num(d.get(c)) for c in 'xyz'):
        return [float(d['x']), float(d['y']), float(d['z'])]
    for k, v in d.items():
        kl = k.lower()
        if any(s in kl for s in ('camera', 'pixel', 'uv', 'intrinsic', 'quat', 'origin', 'ray', 'normal')):
            continue
        p = _numlist(v, 3)
        if p:
            return p
    return None


def parse_points(metas, n):
    """Return list of n (point or None) from a pixel_to_position reply."""
    best = None
    for m in metas:
        cands = []
        if isinstance(m, list) and len(m) == n:
            cands.append(m)
        for k, v, d in walk(m):
            if isinstance(v, list) and len(v) == n and all(isinstance(e, (dict, list)) for e in v):
                cands.append(v)
        for c in cands:
            pts = [_point_from(e) for e in c]
            good = sum(p is not None for p in pts)
            if best is None or good > best[0]:
                best = (good, pts)
    if best is None:
        return [None] * n
    return best[1]
