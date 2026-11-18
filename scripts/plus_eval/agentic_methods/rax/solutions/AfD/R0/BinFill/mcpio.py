"""Defensive parsing of RoboMME-Plus MCP replies (observations, steps, pixel queries)."""
import base64
import io
import json
import os
import time

import numpy as np
from PIL import Image

from geom import Camera, quat_wxyz_to_mat

WS = '/workspace'
LOG_PATH = '/workspace/output/binfill_log.txt'
_t0 = time.time()


def log(*a):
    msg = '[%7.1f] ' % (time.time() - _t0) + ' '.join(str(x) for x in a)
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
    return time.time() - _t0


# ---------------------------------------------------------------- generic helpers

def content_blocks(result):
    if isinstance(result, dict):
        c = result.get('content')
        if isinstance(c, list):
            return c
    if isinstance(result, list):
        return result
    return []


def parse_meta(result):
    meta = {}
    if isinstance(result, dict):
        sc = result.get('structuredContent')
        if isinstance(sc, dict):
            meta.update(sc)
    for b in content_blocks(result):
        if isinstance(b, dict) and b.get('type') == 'text':
            t = b.get('text', '')
            try:
                j = json.loads(t)
            except Exception:
                continue
            if isinstance(j, dict):
                for k, v in j.items():
                    meta.setdefault(k, v)
    return meta


def walk(obj, path=()):
    """Yield (path, key, value) for all dict entries recursively."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k, v
            yield from walk(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield path, i, v
            yield from walk(v, path + (i,))


def find_key(obj, pred, maxn=50):
    out = []
    for p, k, v in walk(obj):
        if isinstance(k, str) and pred(k.lower()):
            out.append((p, k, v))
            if len(out) >= maxn:
                break
    return out


def is_num_list(v, n=None):
    if not isinstance(v, (list, tuple)):
        return False
    if n is not None and len(v) != n:
        return False
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


def get_first(meta, names, default=None):
    for n in names:
        if n in meta:
            return meta[n]
    for p, k, v in walk(meta):
        if k in names:
            return v
    return default


def camera_of(s):
    s = str(s).lower()
    if 'wrist' in s or 'hand' in s or 'gripper' in s:
        return 'wrist'
    if 'front' in s or 'base' in s or 'third' in s or 'scene' in s or 'static' in s:
        return 'front'
    return None


def kind_of(s):
    s = str(s).lower()
    if 'depth' in s:
        return 'depth'
    if 'rgb' in s or 'color' in s or 'colour' in s or 'image' in s:
        return 'rgb'
    return None


def load_image_file(path):
    p = path if os.path.isabs(path) else os.path.join(WS, path)
    if not os.path.exists(p):
        return None
    if p.endswith('.npy'):
        return np.load(p)
    im = Image.open(p)
    im.load()
    return im


def pil_to_array(im, kind):
    if im is None:
        return None
    if isinstance(im, np.ndarray):
        return im
    if kind == 'depth' or im.mode in ('I;16', 'I;16B', 'I', 'F', 'I;16L'):
        return np.array(im).astype(np.float64)
    return np.array(im.convert('RGB'))


# ---------------------------------------------------------------- observation

class Obs:
    def __init__(self):
        self.meta = {}
        self.step_id = None
        self.state = None
        self.instruction = ''
        self.rgb = {}      # cam -> HxWx3 uint8
        self.depth = {}    # cam -> HxW float metres (0 invalid)
        self.cams = {}     # cam -> Camera
        self.ended = False
        self.success = None
        self.raw_keys = []

    def summary(self):
        return dict(step=self.step_id, state=None if self.state is None else [round(x, 4) for x in self.state],
                    rgb={k: v.shape for k, v in self.rgb.items()}, depth={k: v.shape for k, v in self.depth.items()},
                    cams=list(self.cams), ended=self.ended, success=self.success)


def _decode_image_blocks(result):
    """Return list of (label_hint, PIL image) for image blocks, with the preceding text as hint."""
    out = []
    last_text = ''
    for b in content_blocks(result):
        if not isinstance(b, dict):
            continue
        if b.get('type') == 'text':
            last_text = b.get('text', '')[:300]
            continue
        if b.get('type') == 'image' and b.get('data'):
            try:
                im = Image.open(io.BytesIO(base64.b64decode(b['data'])))
                im.load()
            except Exception as e:
                log('image decode failed', e)
                continue
            hint = ' '.join(str(b.get(k, '')) for k in b if k not in ('data', 'type'))
            out.append((hint + ' ' + (last_text if len(last_text) < 300 else ''), im))
            last_text = ''
    return out


def _parse_K(v):
    if isinstance(v, dict):
        for key in ('K', 'matrix', 'intrinsic_matrix', 'intrinsics', 'camera_matrix', 'k'):
            if key in v:
                return _parse_K(v[key])
        if all(k in v for k in ('fx', 'fy', 'cx', 'cy')):
            return np.array([[v['fx'], 0, v['cx']], [0, v['fy'], v['cy']], [0, 0, 1.0]])
        return None
    a = np.asarray(v, float) if isinstance(v, list) else None
    if a is None:
        return None
    if a.shape == (3, 3):
        return a
    if a.size == 9:
        return a.reshape(3, 3)
    if a.size == 4:
        return np.array([[a[0], 0, a[2]], [0, a[1], a[3]], [0, 0, 1.0]])
    return None


def _parse_pose(v):
    """Return (R, t, conv) or None."""
    conv = 'opengl'
    if isinstance(v, dict):
        for k, x in v.items():
            if isinstance(x, str) and ('opencv' in x.lower() or 'cv' == x.lower()):
                conv = 'opencv'
        pos = None
        for key in ('position_m', 'position', 'p', 'translation', 'pos', 't', 'xyz'):
            if key in v and is_num_list(v[key], 3):
                pos = np.asarray(v[key], float)
                break
        q = None
        for key in ('quaternion_wxyz', 'quat_wxyz', 'wxyz', 'quaternion', 'q', 'orientation', 'rotation'):
            if key in v and is_num_list(v[key], 4):
                q = v[key]
                break
        if pos is not None and q is not None:
            return quat_wxyz_to_mat(q), pos, conv
        for key in ('matrix', 'cam2world', 'camera_to_world', 'pose', 'transform', 'model_matrix', 'T'):
            if key in v:
                r = _parse_pose(v[key])
                if r is not None:
                    return r[0], r[1], conv
        return None
    if isinstance(v, list):
        a = np.asarray(v, float)
        if a.shape == (4, 4):
            return a[:3, :3], a[:3, 3], conv
        if a.size == 16:
            a = a.reshape(4, 4)
            return a[:3, :3], a[:3, 3], conv
        if a.size == 7:
            return quat_wxyz_to_mat(a[3:]), a[:3], conv
    return None


def _per_camera(v):
    """Split a value keyed by camera names into a dict."""
    out = {}
    if isinstance(v, dict):
        for k, x in v.items():
            c = camera_of(k)
            if c:
                out[c] = x
        if not out and 'camera' in v:
            c = camera_of(v['camera'])
            if c:
                out[c] = v
    elif isinstance(v, list):
        for x in v:
            if isinstance(x, dict):
                c = camera_of(x.get('camera', x.get('name', '')))
                if c:
                    out[c] = x
    return out


def parse_observation(result, prev=None):
    o = Obs()
    meta = parse_meta(result)
    o.meta = meta
    o.raw_keys = list(meta.keys())
    # nested observation dict?
    m = meta
    for key in ('observation', 'obs'):
        if isinstance(meta.get(key), dict):
            m = dict(meta)
            m.update(meta[key])
    sid = get_first(m, ['step_id', 'step', 'elapsed_steps'])
    try:
        o.step_id = int(sid) if sid is not None else None
    except Exception:
        o.step_id = None
    st = m.get('state')
    if isinstance(st, dict):
        for k in ('values', 'vector', 'state'):
            if is_num_list(st.get(k), 8):
                st = st[k]
                break
    if not is_num_list(st, 8):
        cands = [v for p, k, v in walk(m) if k in ('state', 'proprio', 'robot_state') and is_num_list(v, 8)]
        st = cands[0] if cands else None
    o.state = [float(x) for x in st] if st is not None else None
    ins = get_first(m, ['instruction', 'goal', 'language_instruction', 'task_instruction'])
    o.instruction = ins if isinstance(ins, str) else ''
    # termination
    for p, k, v in walk(meta):
        kl = str(k).lower()
        if kl in ('episode_ended', 'terminated', 'done', 'ended', 'truncated') and v is True:
            o.ended = True
        if kl in ('task_complete', 'success') and isinstance(v, bool):
            o.success = v if o.success is None else (o.success or v)
    # ---- images
    hinted = _decode_image_blocks(result)
    rgb_list, depth_list = [], []
    for hint, im in hinted:
        is_depth = im.mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F') or kind_of(hint) == 'depth'
        (depth_list if is_depth else rgb_list).append((camera_of(hint), im))
    # labels from metadata lists describing attachments
    att_desc = []
    for p, k, v in walk(meta):
        if isinstance(v, list) and v and all(isinstance(x, dict) and ('camera' in x) for x in v):
            att_desc.append(v)
    def assign(lst, kind):
        cams = [c for c, _ in lst]
        if all(cams) and len(set(cams)) == len(cams):
            return {c: im for c, im in lst}
        for desc in att_desc:
            dd = [x for x in desc if kind_of(x.get('kind', x.get('type', x.get('name', 'rgb')))) in (kind, None)]
            if len(dd) == len(lst):
                labs = [camera_of(x.get('camera')) for x in dd]
                if all(labs) and len(set(labs)) == len(labs):
                    return {c: im for c, (_, im) in zip(labs, lst)}
        order = ['front', 'wrist']
        return {order[i]: im for i, (_, im) in enumerate(lst[:2])}
    for c, im in assign(rgb_list, 'rgb').items():
        o.rgb[c] = pil_to_array(im, 'rgb')
    mpu_default = 0.001
    mpu_cands = [v for p, k, v in walk(meta) if k == 'meters_per_unit' and isinstance(v, (int, float))]
    if mpu_cands:
        mpu_default = float(mpu_cands[0])
    for c, im in assign(depth_list, 'depth').items():
        a = pil_to_array(im, 'depth')
        if a is not None and a.ndim == 2:
            o.depth[c] = a * (mpu_default if a.max() > 50 else 1.0)
    # files referenced in metadata (paths)
    for p, k, v in walk(meta):
        if isinstance(v, dict) and isinstance(v.get('path'), str) and ('camera' in v):
            c = camera_of(v['camera'])
            kd = kind_of(v.get('kind', v['path']))
            if not c or not kd:
                continue
            if (kd == 'rgb' and c in o.rgb) or (kd == 'depth' and c in o.depth):
                continue
            try:
                im = load_image_file(v['path'])
            except Exception as e:
                log('file load failed', v['path'], e)
                im = None
            if im is None:
                continue
            a = pil_to_array(im, kd)
            if kd == 'rgb':
                o.rgb[c] = a
            elif a.ndim == 2:
                mpu = float(v.get('meters_per_unit', mpu_default) or mpu_default)
                o.depth[c] = a * (mpu if a.max() > 50 else 1.0)
    # frame-sequence manifest: use its last frame if its step matches
    try:
        fs = get_first(meta, ['frame_sequence'])
        mp = fs.get('manifest_path') if isinstance(fs, dict) else None
        if mp and (('front' not in o.depth) or ('front' not in o.rgb) or ('wrist' not in o.depth)):
            man = json.load(open(os.path.join(WS, mp)))
            frames = man.get('frames', [])
            if frames:
                fr = frames[-1]
                ok = (o.step_id is None or fr.get('step_id') is None or int(fr.get('step_id')) == int(o.step_id))
                if ok:
                    for f in fr.get('files', []):
                        c, kd = camera_of(f.get('camera', '')), kind_of(f.get('kind', ''))
                        if not c or not kd:
                            continue
                        if (kd == 'rgb' and c in o.rgb) or (kd == 'depth' and c in o.depth):
                            continue
                        im = load_image_file(f['path'])
                        if im is None:
                            continue
                        a = pil_to_array(im, kd)
                        if kd == 'rgb':
                            o.rgb[c] = a
                        elif a.ndim == 2:
                            mpu = float(f.get('meters_per_unit', mpu_default) or mpu_default)
                            o.depth[c] = a * (mpu if a.max() > 50 else 1.0)
                    if o.state is None and is_num_list(fr.get('state'), 8):
                        o.state = [float(x) for x in fr['state']]
    except Exception as e:
        log('manifest parse failed', repr(e)[:200])
    # ---- calibration
    Ks, poses = {}, {}
    for p, k, v in walk(meta):
        kl = str(k).lower()
        if 'intrinsic' in kl:
            for c, x in _per_camera(v).items():
                K = _parse_K(x)
                if K is not None:
                    Ks.setdefault(c, K)
        elif ('pose' in kl or 'extrinsic' in kl) and 'camera' in kl:
            for c, x in _per_camera(v).items():
                r = _parse_pose(x)
                if r is not None:
                    poses.setdefault(c, r)
    # per-camera blocks e.g. cameras: {front: {intrinsics:..., pose:...}}
    for p, k, v in walk(meta):
        if isinstance(k, str) and camera_of(k) and isinstance(v, dict):
            c = camera_of(k)
            if c not in Ks:
                for kk, x in v.items():
                    if 'intrinsic' in str(kk).lower():
                        K = _parse_K(x)
                        if K is not None:
                            Ks[c] = K
            if c not in poses:
                for kk, x in v.items():
                    if 'pose' in str(kk).lower() or 'extrinsic' in str(kk).lower():
                        r = _parse_pose(x)
                        if r is not None:
                            poses[c] = r
    for c in ('front', 'wrist'):
        if c in Ks and c in poses:
            R, t, conv = poses[c]
            o.cams[c] = Camera(Ks[c], R, t, conv=conv)
    return o


def parse_points(result, n):
    """Parse robomme_pixel_to_position reply into list of dicts {pos: np.array|None, spread: float}."""
    meta = parse_meta(result)
    best = None
    for p, k, v in walk(meta):
        if isinstance(v, list) and len(v) == n and all(isinstance(x, dict) for x in v):
            best = v
            break
    if best is None and isinstance(meta, dict):
        for k in ('points', 'positions', 'results', 'selections'):
            if isinstance(meta.get(k), list):
                best = meta[k]
                break
    out = []
    if best is None:
        # maybe plain list of positions
        lists = [v for p, k, v in walk(meta) if isinstance(v, list) and len(v) == n and all(is_num_list(x, 3) or x is None for x in v)]
        if lists:
            return [{'pos': None if x is None else np.asarray(x, float), 'spread': 0.0} for x in lists[0]]
        return [{'pos': None, 'spread': None} for _ in range(n)]
    for x in best:
        pos = None
        valid = x.get('valid', True)
        for key in ('position', 'world_position', 'position_m', 'world', 'point', 'xyz', 'world_point', 'pos', 'p'):
            if key in x and is_num_list(x[key], 3):
                pos = np.asarray(x[key], float)
                break
        if pos is None:
            for kk, vv in x.items():
                if is_num_list(vv, 3) and ('pos' in kk or 'world' in kk or 'point' in kk):
                    pos = np.asarray(vv, float)
                    break
        spread = x.get('depth_spread_m', x.get('spread', 0.0))
        try:
            spread = float(spread)
        except Exception:
            spread = 0.0
        if valid is False:
            pos = None
        out.append({'pos': pos, 'spread': spread})
    while len(out) < n:
        out.append({'pos': None, 'spread': None})
    return out
