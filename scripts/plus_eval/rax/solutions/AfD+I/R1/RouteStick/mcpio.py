"""Defensive parsing of MCP tool replies (format never observed live before the rehearsal)."""
import base64
import io
import json
import numbers
import os
from pathlib import Path

import numpy as np

WORKSPACE = Path('/workspace')


def blocks(result):
    if isinstance(result, dict):
        c = result.get('content')
        if isinstance(c, list):
            return c
    if isinstance(result, list):
        return result
    return []


def meta_of(result):
    """First JSON dict found in text blocks, merged with structuredContent if present."""
    meta = None
    if isinstance(result, dict) and isinstance(result.get('structuredContent'), dict):
        meta = dict(result['structuredContent'])
    for b in blocks(result):
        if isinstance(b, dict) and b.get('type') == 'text':
            try:
                j = json.loads(b.get('text', ''))
            except Exception:
                continue
            if isinstance(j, dict):
                if meta is None:
                    meta = j
                else:
                    for k, v in j.items():
                        meta.setdefault(k, v)
                break
    return meta or {}


def images_of(result):
    """Decode image blocks -> list of (label_hint, np.uint8 HxWx3)."""
    from PIL import Image
    out = []
    last_text = ''
    for b in blocks(result):
        if not isinstance(b, dict):
            continue
        if b.get('type') == 'text':
            t = b.get('text', '') or ''
            # only short, non-JSON text blocks are treated as image labels
            last_text = t if (len(t) < 120 and not t.lstrip().startswith(('{', '['))) else ''
            continue
        if b.get('type') == 'image' and b.get('data'):
            try:
                im = Image.open(io.BytesIO(base64.b64decode(b['data'])))
                hint = ''
                for key in ('name', 'title', 'annotations', 'camera', 'uri'):
                    if key in b:
                        hint += json.dumps(b[key])
                hint += ' ' + last_text
                if im.mode in ('I;16', 'I', 'I;16B') or 'depth' in hint.lower()[:60]:
                    out.append(('depth:' + hint, np.array(im)))
                else:
                    out.append((hint, np.array(im.convert('RGB'))))
            except Exception:
                pass
    return out


def walk(obj, path=()):
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, path + (i,))


def is_numlist(v, n=None):
    return isinstance(v, list) and (n is None or len(v) == n) and len(v) > 0 and all(
        isinstance(x, numbers.Number) and not isinstance(x, bool) for x in v)


def find_state(meta):
    """8-number list under a key named 'state' (top-most; last if tie)."""
    best = None
    for p, v in walk(meta):
        if p and p[-1] == 'state' and is_numlist(v, 8):
            depth = len(p)
            if best is None or depth <= best[0]:
                best = (depth, v)
        elif p and p[-1] == 'state' and isinstance(v, dict):
            # maybe {'values': [...]} or {'ee_pos':..}
            for kk in ('values', 'vector', 'data'):
                if is_numlist(v.get(kk), 8):
                    if best is None or len(p) < best[0]:
                        best = (len(p), v[kk])
    return np.array(best[1], float) if best else None


def find_key(meta, key, types=None):
    best = None
    for p, v in walk(meta):
        if p and p[-1] == key and (types is None or isinstance(v, types)):
            if best is None or len(p) < best[0]:
                best = (len(p), v)
    return best[1] if best else None


def find_bool_any(meta, keys):
    for p, v in walk(meta):
        if p and p[-1] in keys and v is True:
            return True
    return False


def episode_ended(meta):
    return find_bool_any(meta, ('episode_ended', 'terminated', 'done', 'truncated'))


def task_complete(meta):
    return find_bool_any(meta, ('task_complete', 'success'))


def resolve_path(p, manifest_dir=None):
    if p is None:
        return None
    cands = [WORKSPACE / p, Path(p)]
    if manifest_dir is not None:
        cands.append(Path(manifest_dir) / p)
    cands.append(Path(os.getcwd()) / p)
    for c in cands:
        try:
            if c.exists():
                return c
        except Exception:
            pass
    return None


def parse_points(meta, n):
    """Extract n world points from a pixel_to_position reply. Returns list of (xyz or None, info)."""
    # find a list of n dicts
    cand = None
    for p, v in walk(meta):
        if isinstance(v, list) and len(v) == n and all(isinstance(x, dict) for x in v):
            if cand is None or len(p) < cand[0]:
                cand = (len(p), v)
    if cand is None:
        return None
    out = []
    for d in cand[1]:
        xyz = None
        pref = None
        for p, v in walk(d):
            if is_numlist(v, 3) and p:
                kname = str(p[-1]).lower()
                score = 0
                if 'world' in kname:
                    score = 3
                elif 'pos' in kname or 'point' in kname or 'xyz' in kname:
                    score = 2
                elif 'pixel' in kname or 'uv' in kname or 'cam' in kname:
                    score = -1
                else:
                    score = 1
                if pref is None or score > pref:
                    pref, xyz = score, v
        valid = d.get('valid', True)
        if valid is False or xyz is None or (pref is not None and pref < 0):
            out.append((None, d))
        else:
            out.append((np.array(xyz, float), d))
    return out
