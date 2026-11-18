"""Defensive helpers around ctx.call replies (JSON text block + image blocks)."""
import base64
import io
import json
import os
import time
import numpy as np

WS = '/workspace'
LOG_PATH = '/workspace/output/controller_log.txt'
_T0 = time.time()


def log(*a):
    msg = '[%7.1fs] ' % (time.time() - _T0) + ' '.join(str(x) for x in a)
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


def parse_reply(result):
    """Return (meta_dict, list_of_text_blocks, list_of_image_blocks)."""
    meta, texts, images = {}, [], []
    content = result.get('content', []) if isinstance(result, dict) else []
    for b in content:
        t = b.get('type')
        if t == 'text':
            texts.append(b.get('text', ''))
        elif t == 'image':
            images.append(b)
    for tx in texts:
        try:
            j = json.loads(tx)
            if isinstance(j, dict):
                if not meta:
                    meta = j
                else:
                    meta.setdefault('_extra', []).append(j)
        except Exception:
            continue
    if not meta and isinstance(result, dict) and 'structuredContent' in result:
        sc = result.get('structuredContent')
        if isinstance(sc, dict):
            meta = sc
    return meta, texts, images


def find_key(obj, key, maxdepth=8):
    """First value for `key` found by BFS through nested dicts/lists."""
    q = [(obj, 0)]
    while q:
        o, d = q.pop(0)
        if isinstance(o, dict):
            if key in o:
                return o[key]
            if d < maxdepth:
                q.extend((v, d + 1) for v in o.values())
        elif isinstance(o, list) and d < maxdepth:
            q.extend((v, d + 1) for v in o[:200])
    return None


def find_all_keys(obj, key, maxdepth=8):
    out = []
    q = [(obj, 0)]
    while q:
        o, d = q.pop(0)
        if isinstance(o, dict):
            if key in o:
                out.append(o[key])
            if d < maxdepth:
                q.extend((v, d + 1) for v in o.values())
        elif isinstance(o, list) and d < maxdepth:
            q.extend((v, d + 1) for v in o[:200])
    return out


def get_state(meta):
    s = find_key(meta, 'state')
    if isinstance(s, dict):
        for k in ('values', 'state', 'vector', 'data'):
            if isinstance(s.get(k), list):
                s = s[k]
                break
    if isinstance(s, list) and len(s) >= 8:
        try:
            return np.array([float(x) for x in s[:8]])
        except Exception:
            return None
    return None


def episode_ended(meta):
    for k in ('episode_ended',):
        v = find_key(meta, k)
        if v is True:
            return True
    return False


def task_complete(meta):
    v = find_key(meta, 'task_complete')
    return v is True


def decode_image_block(b):
    from PIL import Image
    data = b.get('data')
    if not data:
        return None
    try:
        im = Image.open(io.BytesIO(base64.b64decode(data)))
        return im
    except Exception:
        return None


def load_rgb(path):
    from PIL import Image
    p = path if os.path.isabs(path) else os.path.join(WS, path)
    with Image.open(p) as im:
        return np.array(im.convert('RGB'))


def load_depth(path, mpu=0.001):
    from PIL import Image
    p = path if os.path.isabs(path) else os.path.join(WS, path)
    with Image.open(p) as im:
        a = np.array(im)
    return a.astype(np.float32) * float(mpu or 0.001)


def read_manifest(meta):
    fs = find_key(meta, 'frame_sequence')
    mp = None
    if isinstance(fs, dict):
        mp = fs.get('manifest_path') or find_key(fs, 'manifest_path')
    if mp is None:
        mp = find_key(meta, 'manifest_path')
    if not mp:
        return None, None
    p = mp if os.path.isabs(mp) else os.path.join(WS, mp)
    try:
        with open(p) as f:
            return json.load(f), p
    except Exception as e:
        log('manifest read failed', p, repr(e))
        return None, p


def frame_files(frame):
    """{(camera, kind): (path, meters_per_unit)} for one manifest frame."""
    out = {}
    for f in frame.get('files', []) or []:
        cam = str(f.get('camera', '')).lower()
        kind = str(f.get('kind', '')).lower()
        out[(cam, kind)] = (f.get('path'), f.get('meters_per_unit', 0.001))
    return out
