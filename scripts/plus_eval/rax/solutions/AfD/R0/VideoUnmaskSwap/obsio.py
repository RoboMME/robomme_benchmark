"""Reply parsing and frame loading for RoboMME-Plus RAX tool replies (defensive)."""
import base64
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

WS = Path('/workspace')


def reply_meta(reply):
    """First JSON text block of an MCP tool result (merged with structuredContent)."""
    meta = {}
    if isinstance(reply, dict):
        sc = reply.get('structuredContent')
        if isinstance(sc, dict):
            meta.update(sc)
        for b in reply.get('content', []) or []:
            if b.get('type') == 'text':
                try:
                    j = json.loads(b.get('text', ''))
                except Exception:
                    continue
                if isinstance(j, dict):
                    for k, v in j.items():
                        meta.setdefault(k, v)
                    break
    return meta


def reply_images(reply):
    """List of decoded images from image content blocks, in order: (PIL.Image, block)."""
    out = []
    if not isinstance(reply, dict):
        return out
    for b in reply.get('content', []) or []:
        if b.get('type') == 'image' and b.get('data'):
            try:
                im = Image.open(io.BytesIO(base64.b64decode(b['data'])))
                im.load()
                out.append((im, b))
            except Exception:
                pass
    return out


def resolve(path, manifest_dir=None):
    p = Path(path)
    cands = [p] if p.is_absolute() else [WS / p]
    if manifest_dir is not None and not p.is_absolute():
        cands.append(Path(manifest_dir) / p)
    for c in cands:
        if c.exists():
            return c
    return None


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'))


def load_depth(path, mpu=0.001):
    with Image.open(path) as im:
        a = np.array(im)
    if a.ndim == 3:
        a = a[..., 0]
    a = a.astype(np.float64) * float(mpu if mpu else 0.001)
    return a


def find_all(obj, pred, depth=0, out=None):
    if out is None:
        out = []
    if depth > 10:
        return out
    if isinstance(obj, dict):
        if pred(obj):
            out.append(obj)
        for v in obj.values():
            find_all(v, pred, depth + 1, out)
    elif isinstance(obj, list):
        for v in obj:
            find_all(v, pred, depth + 1, out)
    return out


def files_to_frame(files, manifest_dir=None):
    """files: list of {camera, kind, path, meters_per_unit?} -> dict of arrays."""
    fr = {}
    for f in files:
        cam = str(f.get('camera', '')).lower()
        kind = str(f.get('kind', '')).lower()
        cam = 'front' if 'front' in cam or 'base' in cam or 'third' in cam else ('wrist' if 'wrist' in cam or 'hand' in cam else cam)
        p = resolve(f.get('path', ''), manifest_dir)
        if p is None:
            continue
        try:
            if kind == 'depth':
                fr[cam + '_depth'] = load_depth(p, f.get('meters_per_unit', 0.001))
            else:
                fr[cam + '_rgb'] = load_rgb(p)
        except Exception:
            pass
    return fr


def load_manifest(seq, base=None):
    """seq: frame_sequence descriptor or manifest path -> (manifest dict, manifest_dir)."""
    mp = seq.get('manifest_path') if isinstance(seq, dict) else seq
    if not mp:
        return None, None
    p = resolve(mp, base)
    if p is None:
        return None, None
    with open(p) as fh:
        return json.load(fh), p.parent


def frame_from_reply(reply, meta=None, allow_manifest=True):
    """Best-effort CURRENT frame from a reply: dict with front/wrist rgb/depth arrays.

    Priority: explicit file lists in the reply, inline image blocks, then the last
    frame of a frame_sequence manifest only if its step_id equals the reply's step_id.
    """
    meta = meta if meta is not None else reply_meta(reply)
    fr = {}
    cand = find_all({k: v for k, v in meta.items() if k != 'frame_sequence'},
                    lambda d: 'path' in d and 'camera' in d)
    if cand:
        fr.update(files_to_frame(cand))
    rgbs, depths = [], []
    for im, b in reply_images(reply):
        label = json.dumps({k: v for k, v in b.items() if k != 'data'}).lower()
        if im.mode in ('I;16', 'I', 'I;16B', 'I;16L', 'F') or 'depth' in label:
            a = np.array(im)
            if a.ndim == 3:
                a = a[..., 0]
            depths.append((a.astype(np.float64) * 0.001, label))
        else:
            rgbs.append((np.asarray(im.convert('RGB')), label))

    def pick(lst, cam, idx):
        for a, lab in lst:
            if cam in lab:
                return a
        return lst[idx][0] if len(lst) > idx else None
    for lst, kind in ((rgbs, 'rgb'), (depths, 'depth')):
        if lst:
            a = pick(lst, 'front', 0)
            if a is not None:
                fr.setdefault('front_' + kind, a)
            a = pick(lst, 'wrist', 1)
            if a is not None:
                fr.setdefault('wrist_' + kind, a)
    if allow_manifest and not all(k in fr for k in ('front_rgb', 'wrist_rgb', 'front_depth', 'wrist_depth')):
        seq = meta.get('frame_sequence')
        sid = get_step_id(meta)
        if seq:
            try:
                man, mdir = load_manifest(seq)
                frames = man.get('frames', []) if man else []
                if frames:
                    last = max(frames, key=lambda f: (f.get('frame_index', 0)))
                    fsid = last.get('step_id')
                    if sid is None or fsid is None or fsid == sid:
                        sub = files_to_frame(last.get('files', []), mdir)
                        for k, v in sub.items():
                            fr.setdefault(k, v)
            except Exception:
                pass
    return fr


def get_state(meta):
    for key in ('state', 'proprio', 'robot_state'):
        v = meta.get(key)
        if isinstance(v, dict):
            for k2 in ('values', 'state', 'vector'):
                if isinstance(v.get(k2), list):
                    v = v[k2]
                    break
        if isinstance(v, list) and len(v) >= 8 and all(isinstance(x, (int, float)) for x in v[:8]):
            return np.array(v[:8], float)
    # nested search
    hits = find_all(meta, lambda d: isinstance(d.get('state'), list) and len(d.get('state')) >= 8)
    if hits:
        try:
            return np.array(hits[-1]['state'][:8], float)
        except Exception:
            pass
    return None


def get_step_id(meta):
    for key in ('step_id', 'observation_step_id', 'step'):
        v = meta.get(key)
        if isinstance(v, int):
            return v
    hits = find_all(meta, lambda d: isinstance(d.get('step_id'), int))
    if hits:
        return hits[-1]['step_id']
    return None


def ended(meta):
    def b(k):
        v = meta.get(k)
        if v is None:
            hits = find_all(meta, lambda d: k in d and isinstance(d[k], bool))
            v = hits[-1][k] if hits else None
        return bool(v) if isinstance(v, bool) else False
    return b('episode_ended') or b('terminated') or b('truncated'), b('task_complete') or b('success')
