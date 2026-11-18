"""Defensive parsing of MCP replies (exact JSON layout was never observed live: R1 budget)."""
import base64
import io
import json
from pathlib import Path

import numpy as np

WS = Path('/workspace')


def bfs_find(obj, key, pred=None, max_nodes=20000):
    """Breadth-first search for the first value under `key` satisfying pred."""
    queue = [obj]; n = 0
    while queue and n < max_nodes:
        cur = queue.pop(0); n += 1
        if isinstance(cur, dict):
            if key in cur and (pred is None or pred(cur[key])):
                return cur[key]
            queue.extend(v for v in cur.values() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            queue.extend(v for v in cur if isinstance(v, (dict, list)))
    return None


def is_num_list(x, n=None):
    return isinstance(x, list) and (n is None or len(x) == n) and len(x) > 0 and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in x)


def text_blocks(reply):
    out = []
    for b in (reply or {}).get('content', []) or []:
        if isinstance(b, dict) and b.get('type') == 'text':
            out.append(b.get('text', ''))
    return out


def meta_of(reply):
    for t in text_blocks(reply):
        try:
            v = json.loads(t)
            if isinstance(v, dict):
                return v
        except Exception:
            continue
    return {}


def decode_image_block(b):
    from PIL import Image
    data = base64.b64decode(b.get('data', ''))
    im = Image.open(io.BytesIO(data))
    im.load()
    return im


def pil_to_arrays(im):
    """Return ('rgb', HxWx3 uint8) or ('depth', HxW float metres-ish raw)."""
    if im.mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F'):
        return 'depth', np.asarray(im).astype(np.float64)
    return 'rgb', np.asarray(im.convert('RGB'))


def _label_from_text(t):
    tl = t.lower()
    cam = 'wrist' if 'wrist' in tl else ('front' if 'front' in tl else None)
    kind = 'depth' if 'depth' in tl else None
    return cam, kind


def extract_images(reply, meta):
    """Return dict with keys like ('front','rgb'), ('wrist','rgb'), ('front','depth')..."""
    found = {}
    blocks = (reply or {}).get('content', []) or []
    seq = []  # (cam_hint, kind_hint, kind, array)
    last_text = ''
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get('type') == 'text':
            last_text = b.get('text', '') if len(b.get('text', '')) < 400 else ''
            continue
        if b.get('type') == 'image':
            try:
                im = decode_image_block(b)
                kind, arr = pil_to_arrays(im)
            except Exception:
                continue
            cam, khint = _label_from_text(last_text)
            # annotations or names on the block itself
            for k in ('name', 'camera', 'annotations', 'title', 'description'):
                if k in b and isinstance(b[k], (str, dict)):
                    c2, k2 = _label_from_text(json.dumps(b[k]) if isinstance(b[k], dict) else b[k])
                    cam = cam or c2; khint = khint or k2
            if khint == 'depth' and kind == 'rgb' and arr.ndim == 3:
                kind = 'depth_rgb'
            seq.append([cam, kind, arr])
            last_text = ''
    # metadata attachment list describing order
    desc = None
    for key in ('images', 'attachments', 'cameras', 'image_order'):
        v = bfs_find(meta, key, lambda x: isinstance(x, list) and len(x) > 0)
        if v is not None:
            desc = v; break
    if desc is not None:
        names = []
        for d in desc:
            if isinstance(d, dict):
                names.append(str(d.get('camera', d.get('name', ''))).lower() + ' ' + str(d.get('kind', '')).lower())
            else:
                names.append(str(d).lower())
        rgb_seq = [s for s in seq if s[1] == 'rgb']
        if len(names) == len(seq):
            for s, nm in zip(seq, names):
                if s[0] is None:
                    s[0] = 'wrist' if 'wrist' in nm else ('front' if 'front' in nm else None)
        elif len(names) == len(rgb_seq):
            for s, nm in zip(rgb_seq, names):
                if s[0] is None:
                    s[0] = 'wrist' if 'wrist' in nm else ('front' if 'front' in nm else None)
    # assign defaults: first rgb front, second wrist (tool doc order: third-person and wrist)
    rgbs = [s for s in seq if s[1] == 'rgb']
    if len(rgbs) > 2:
        # drop grayscale renderings (likely depth visualisations) when unlabeled
        keep = [s for s in rgbs if s[0] is not None or not _is_gray(s[2])]
        for s in rgbs:
            if s not in keep:
                s[1] = 'depth_rgb'
        rgbs = keep
    if len(rgbs) >= 2 and rgbs[0][0] is None and rgbs[1][0] is None:
        if dark_corner(rgbs[0][2]) > dark_corner(rgbs[1][2]) + 0.02:
            rgbs[0][0], rgbs[1][0] = 'wrist', 'front'
    deps = [s for s in seq if s[1] == 'depth']
    for i, s in enumerate(rgbs):
        if s[0] is None:
            used = {r[0] for r in rgbs if r[0]}
            s[0] = 'front' if 'front' not in used else 'wrist'
    for i, s in enumerate(deps):
        if s[0] is None:
            used = {r[0] for r in deps if r[0]}
            s[0] = 'front' if 'front' not in used else 'wrist'
    for s in seq:
        k = (s[0], 'rgb' if s[1] == 'rgb' else 'depth')
        if k not in found and s[1] != 'depth_rgb':
            found[k] = s[2]
    # file-path fallback
    if ('front', 'rgb') not in found or ('wrist', 'rgb') not in found or ('front', 'depth') not in found:
        try:
            files = _collect_files(meta)
            from PIL import Image
            for (cam, kind), path in files.items():
                if (cam, kind) in found:
                    continue
                p = WS / path if not str(path).startswith('/') else Path(path)
                if p.exists():
                    with Image.open(p) as im:
                        im.load()
                        k, arr = pil_to_arrays(im)
                    found[(cam, kind)] = arr
        except Exception:
            pass
    return found


def _is_gray(a):
    a = a.astype(np.int16)
    return bool(np.abs(a[..., 0] - a[..., 1]).mean() < 1.0 and np.abs(a[..., 1] - a[..., 2]).mean() < 1.0)


def dark_corner(a):
    """Fraction of near-black pixels in the bottom-right corner (gripper finger in wrist view)."""
    H, W = a.shape[:2]
    reg = a[int(H * 0.7):, int(W * 0.66):]
    return float((reg.max(-1) < 60).mean())


def _collect_files(meta):
    """Find {camera, kind, path} file records; keep the last one per (camera, kind)."""
    out = {}
    queue = [meta]; n = 0
    while queue and n < 20000:
        cur = queue.pop(0); n += 1
        if isinstance(cur, dict):
            if 'camera' in cur and 'path' in cur and isinstance(cur.get('path'), str):
                kind = str(cur.get('kind', 'rgb')).lower()
                out[(str(cur['camera']).lower(), 'depth' if 'depth' in kind else 'rgb')] = cur['path']
            queue.extend(v for v in cur.values() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            queue.extend(v for v in cur if isinstance(v, (dict, list)))
    return out


def vec3_of(d):
    if not isinstance(d, dict):
        return None
    for k in ('position', 'world_position', 'point', 'world_point', 'xyz', 'world', 'position_m',
              'world_xyz', 'point_world', 'world_m'):
        v = d.get(k)
        if is_num_list(v, 3):
            return np.array(v, float)
        if isinstance(v, dict):
            for kk in ('x',):
                if all(c in v for c in 'xyz'):
                    return np.array([v['x'], v['y'], v['z']], float)
    for k, v in d.items():
        if k in ('pixel', 'uv', 'camera_position', 'ray', 'direction'):
            continue
        if is_num_list(v, 3):
            return np.array(v, float)
    return None


def parse_points(reply, n):
    meta = meta_of(reply)
    lst = None
    queue = [meta]; cnt = 0
    while queue and cnt < 5000:
        cur = queue.pop(0); cnt += 1
        if isinstance(cur, list) and len(cur) == n and all(isinstance(e, dict) for e in cur):
            lst = cur; break
        if isinstance(cur, dict):
            queue.extend(v for v in cur.values() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            queue.extend(v for v in cur if isinstance(v, (dict, list)))
    out = []
    if lst is None:
        return [None] * n, meta
    for e in lst:
        valid = e.get('valid', True)
        p = vec3_of(e)
        spread = e.get('depth_spread_m')
        if valid is False or p is None or not np.all(np.isfinite(p)):
            out.append(None)
        else:
            out.append({'p': p, 'spread': spread if isinstance(spread, (int, float)) else None})
    return out, meta
