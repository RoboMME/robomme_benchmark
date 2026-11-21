"""Defensive parsing of MCP tool replies (format verified only against docs, not live)."""
import base64, io, json, os
import numpy as np
try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

WORKSPACE = '/workspace'


def text_blocks(res):
    out = []
    for b in (res or {}).get('content', []) or []:
        if isinstance(b, dict) and b.get('type') == 'text':
            out.append(b.get('text', ''))
    return out


def parse_meta(res):
    meta = {}
    if isinstance(res, dict) and isinstance(res.get('structuredContent'), dict):
        meta.update(res['structuredContent'])
    for t in text_blocks(res):
        try:
            d = json.loads(t)
        except Exception:
            continue
        if isinstance(d, dict):
            for k, v in d.items():
                meta.setdefault(k, v)
            break
    return meta


def bfs_find(obj, pred, max_nodes=20000):
    """Breadth-first search for (key, value) pairs satisfying pred(key, value); shallowest first."""
    q = [obj]
    out = []
    n = 0
    while q and n < max_nodes:
        nq = []
        for o in q:
            n += 1
            if isinstance(o, dict):
                for k, v in o.items():
                    try:
                        if pred(k, v):
                            out.append((k, v))
                    except Exception:
                        pass
                    if isinstance(v, (dict, list)):
                        nq.append(v)
            elif isinstance(o, list):
                for v in o:
                    if isinstance(v, (dict, list)):
                        nq.append(v)
        q = nq
    return out


def is_num_list(v, n=None, nmin=None):
    if not isinstance(v, (list, tuple)):
        return False
    if n is not None and len(v) != n:
        return False
    if nmin is not None and len(v) < nmin:
        return False
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


def find_first(meta, keys, pred=None):
    keys = set(keys)
    r = bfs_find(meta, lambda k, v: k in keys and (pred is None or pred(v)))
    return r[0][1] if r else None


def decode_image_block(b):
    data = b.get('data')
    if data is None and isinstance(b.get('resource'), dict):
        data = b['resource'].get('blob')
    if data is None:
        return None
    raw = base64.b64decode(data)
    im = Image.open(io.BytesIO(raw))
    im.load()
    return im


def extract_images(res, meta, classify):
    """Return dict camera -> RGB numpy array (and 'depth_<cam>' when depth attachments exist) plus
    a notes string describing how labels were assigned."""
    blocks = (res or {}).get('content', []) or []
    imgs = []  # (pil, hint)
    last_text = ''
    for b in blocks:
        if not isinstance(b, dict):
            continue
        t = b.get('type')
        if t == 'text':
            last_text = b.get('text', '') or ''
            continue
        if t == 'image' or (t == 'resource' and isinstance(b.get('resource'), dict)
                            and str(b['resource'].get('mimeType', '')).startswith('image')):
            try:
                pil = decode_image_block(b)
            except Exception:
                pil = None
            hint = ''
            lt = last_text.lower() if len(last_text) < 200 else ''
            for k in ('annotations', 'name', 'title', 'uri'):
                v = b.get(k)
                if isinstance(v, (str, dict)):
                    lt += ' ' + json.dumps(v).lower()
            if isinstance(b.get('resource'), dict):
                lt += ' ' + str(b['resource'].get('uri', '')).lower()
            if 'front' in lt or 'third' in lt:
                hint = 'front'
            elif 'wrist' in lt or 'hand' in lt:
                hint = 'wrist'
            kind = 'depth' if 'depth' in lt else ''
            imgs.append([pil, hint, kind])
            last_text = ''
    out = {}
    notes = []
    rgb = []
    for pil, hint, kind in imgs:
        if pil is None:
            continue
        mode = pil.mode
        if mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F') or kind == 'depth':
            arr = np.array(pil).astype(np.float32)
            if hint:
                out['depth_' + hint] = arr
            continue
        arr = np.array(pil.convert('RGB'))
        rgb.append([arr, hint])
    # metadata camera list labelling
    if rgb and any(not h for _, h in rgb):
        lists = bfs_find(meta, lambda k, v: isinstance(v, list) and len(v) > 0 and all(
            isinstance(e, dict) and ('camera' in e or 'name' in e) for e in v))
        for k, lst in lists:
            entries = [e for e in lst if str(e.get('kind', e.get('type', 'rgb'))).lower() in ('rgb', 'color', 'image', 'image/png')]
            cams = [str(e.get('camera', e.get('name', ''))).lower() for e in entries]
            if len(cams) == len(rgb) and all(('front' in c or 'wrist' in c) for c in cams):
                for i, c in enumerate(cams):
                    if not rgb[i][1]:
                        rgb[i][1] = 'front' if 'front' in c else 'wrist'
                notes.append('labels-from-meta:' + k)
                break
    # heuristic check / fill
    if len(rgb) >= 2:
        scores = [classify(a) for a, _ in rgb]
        hi = int(np.argmax(scores))
        labels = [h for _, h in rgb]
        if not all(labels) or len(set(labels)) < len(labels):
            for i in range(len(rgb)):
                rgb[i][1] = 'front' if i == hi else 'wrist'
            notes.append('labels-from-heuristic')
        else:
            fi = labels.index('front') if 'front' in labels else None
            if fi is not None and fi != hi and scores[hi] - scores[fi] > 0.3:
                notes.append('WARNING heuristic disagrees with labels %s' % [round(s, 2) for s in scores])
    elif len(rgb) == 1 and not rgb[0][1]:
        rgb[0][1] = 'front' if classify(rgb[0][0]) > 0.45 else 'wrist'
        notes.append('single-image-heuristic')
    for a, h in rgb:
        if h and h not in out:
            out[h] = a
    return out, ';'.join(notes)


def images_from_files(meta):
    """Fallback: load RGB frames from published file paths in metadata."""
    out = {}
    paths = bfs_find(meta, lambda k, v: isinstance(v, str) and v.lower().endswith(('.png', '.jpg', '.jpeg')))
    for k, p in paths:
        pl = p.lower()
        if 'depth' in pl:
            continue
        cam = 'front' if 'front' in pl else ('wrist' if 'wrist' in pl else None)
        if cam is None or cam in out:
            continue
        for base in (WORKSPACE, ''):
            fp = os.path.join(base, p) if base else p
            if os.path.isfile(fp):
                try:
                    out[cam] = np.array(Image.open(fp).convert('RGB'))
                except Exception:
                    pass
                break
    # dicts with camera/kind/path
    for k, lst in bfs_find(meta, lambda k, v: isinstance(v, list) and v and all(isinstance(e, dict) and 'path' in e for e in v)):
        for e in lst:
            cam = str(e.get('camera', '')).lower(); kind = str(e.get('kind', 'rgb')).lower()
            if cam in ('front', 'wrist') and kind == 'rgb' and cam not in out:
                fp = os.path.join(WORKSPACE, e['path'])
                if os.path.isfile(fp):
                    try:
                        out[cam] = np.array(Image.open(fp).convert('RGB'))
                    except Exception:
                        pass
    return out


VEC_KEYS = ('position', 'world_position', 'position_m', 'world_position_m', 'point', 'point_m', 'xyz',
            'world', 'world_point', 'world_xyz', 'position_world', 'p_world')


def _vec_of(d):
    for k in VEC_KEYS:
        v = d.get(k)
        if is_num_list(v, 3):
            return [float(x) for x in v]
        if isinstance(v, dict):
            if all(c in v for c in 'xyz'):
                return [float(v['x']), float(v['y']), float(v['z'])]
    for k, v in d.items():
        if is_num_list(v, 3) and 'pixel' not in k.lower() and 'uv' not in k.lower():
            return [float(x) for x in v]
    if all(c in d for c in 'xyz') and all(isinstance(d[c], (int, float)) for c in 'xyz'):
        return [float(d['x']), float(d['y']), float(d['z'])]
    return None


def parse_points(res, selections):
    """Return list (aligned with selections) of (xyz np.array or None, info dict)."""
    meta = parse_meta(res)
    n = len(selections)
    cands = bfs_find(meta, lambda k, v: isinstance(v, list) and len(v) > 0 and all(isinstance(e, dict) for e in v)
                     and any(_vec_of(e) is not None or e.get('valid') is False for e in v))
    out = [(None, {}) for _ in range(n)]
    if not cands:
        # maybe meta itself is a single point
        if n == 1 and isinstance(meta, dict) and _vec_of(meta) is not None:
            return [(np.array(_vec_of(meta)), meta)]
        return out
    lst = cands[0][1]
    used = set()
    for i, sel in enumerate(selections):
        j = None
        # match by u,v,camera when present
        for jj, e in enumerate(lst):
            if jj in used:
                continue
            if 'u' in e and 'v' in e:
                try:
                    if abs(float(e['u']) - sel['u']) < 0.6 and abs(float(e['v']) - sel['v']) < 0.6 and \
                            str(e.get('camera', sel['camera'])) == sel['camera']:
                        j = jj
                        break
                except Exception:
                    pass
        if j is None and len(lst) == n and i not in used:
            j = i
        if j is None:
            continue
        used.add(j)
        e = lst[j]
        v = _vec_of(e)
        valid = e.get('valid', True)
        if v is None or valid is False or not all(np.isfinite(v)):
            out[i] = (None, e)
        else:
            out[i] = (np.array(v, float), e)
    return out
