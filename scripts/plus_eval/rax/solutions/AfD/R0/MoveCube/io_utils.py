"""MCP reply parsing helpers (defensive: the exact packet layout is only partly documented)."""
import base64
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

WORKSPACE = Path('/workspace')


def first_json(result):
    for b in result.get('content', []) or []:
        if b.get('type') == 'text':
            try:
                return json.loads(b.get('text', ''))
            except Exception:
                continue
    return {}


def walk(obj, depth=0):
    """Yield (key, value, parent) for every dict entry recursively."""
    if depth > 12:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v, obj
            yield from walk(v, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:2000]:
            yield from walk(v, depth + 1)


def find_key(obj, names, pred=None):
    for k, v, _ in walk(obj):
        if k in names and (pred is None or pred(v)):
            return v
    return None


def is_vec(n):
    def p(v):
        return isinstance(v, (list, tuple)) and len(v) == n and all(isinstance(x, (int, float)) for x in v)
    return p


def get_state(meta):
    st = find_key(meta, ('state',), is_vec(8))
    if st is None:
        st = find_key(meta, ('proprio', 'robot_state', 'state_vector'), is_vec(8))
    return None if st is None else np.array(st, dtype=float)


def get_step_id(meta):
    v = meta.get('step_id') if isinstance(meta, dict) else None
    if isinstance(v, int):
        return v
    v = find_key(meta, ('step_id',), lambda x: isinstance(x, int))
    return v


def episode_ended(meta):
    for k, v, _ in walk(meta):
        if k in ('episode_ended', 'terminated', 'task_complete', 'done', 'success') and v is True:
            return True
    return False


def task_complete(meta):
    for k, v, _ in walk(meta):
        if k in ('task_complete', 'success') and v is True:
            return True
    return False


def _decode_image(b):
    data = b.get('data')
    if not data:
        return None
    try:
        im = Image.open(io.BytesIO(base64.b64decode(data))).convert('RGB')
        return np.array(im)
    except Exception:
        return None


def looks_like_wrist(im):
    """Wrist images show the dark gripper finger pads in the lower corners."""
    if im is None:
        return False
    h, w = im.shape[:2]
    reg = np.concatenate([im[int(h * 0.62):, :int(w * 0.2)].reshape(-1, 3), im[int(h * 0.62):, int(w * 0.8):].reshape(-1, 3)])
    dark = (reg.max(1) < 70).mean()
    return dark > 0.06


def get_images(result, meta=None):
    """Return {'front': arr, 'wrist': arr} from a tool result."""
    blocks = result.get('content', []) or []
    ims = []
    pending = None
    for i, b in enumerate(blocks):
        t = b.get('type')
        if t == 'text' and i > 0:
            txt = (b.get('text') or '').lower()
            if 'wrist' in txt and 'front' not in txt:
                pending = 'wrist'
            elif 'front' in txt and 'wrist' not in txt:
                pending = 'front'
        elif t == 'image':
            arr = _decode_image(b)
            label = pending
            for key in ('camera', 'name', 'label', 'annotations'):
                val = b.get(key)
                if isinstance(val, str):
                    if 'wrist' in val.lower():
                        label = 'wrist'
                    elif 'front' in val.lower():
                        label = 'front'
            ims.append([label, arr])
            pending = None
    # order hints in the JSON
    if meta and any(l is None for l, _ in ims):
        order = None
        for k, v, _ in walk(meta):
            if k in ('images', 'image_cameras', 'cameras', 'attachments') and isinstance(v, list):
                names = []
                for x in v:
                    if isinstance(x, str):
                        names.append(x)
                    elif isinstance(x, dict):
                        n = x.get('camera') or x.get('name')
                        k2 = x.get('kind', 'rgb')
                        if isinstance(n, str) and k2 in ('rgb', 'color', None):
                            names.append(n)
                if names and all(n in ('front', 'wrist') for n in names) and len(names) == len(ims):
                    order = names
                    break
        if order:
            for i, n in enumerate(order):
                if ims[i][0] is None:
                    ims[i][0] = n
    out = {}
    unl = []
    for l, a in ims:
        if a is None:
            continue
        if l in ('front', 'wrist') and l not in out:
            out[l] = a
        else:
            unl.append(a)
    for a in unl:
        if 'front' in out and 'wrist' in out:
            break
        if 'front' not in out and 'wrist' not in out:
            # decide by appearance
            if looks_like_wrist(a):
                out['wrist'] = a
            else:
                out['front'] = a
        elif 'front' not in out:
            out['front'] = a
        else:
            out['wrist'] = a
    # sanity: if front looks like wrist and wrist does not, swap
    if 'front' in out and 'wrist' in out:
        if looks_like_wrist(out['front']) and not looks_like_wrist(out['wrist']):
            out['front'], out['wrist'] = out['wrist'], out['front']
    # fallback: files listed in the packet
    if 'front' not in out and meta:
        p = find_rgb_path(meta, 'front')
        if p:
            try:
                out['front'] = np.array(Image.open(p).convert('RGB'))
            except Exception:
                pass
    return out


def find_rgb_path(meta, camera):
    for k, v, parent in walk(meta):
        if k == 'path' and isinstance(v, str) and isinstance(parent, dict):
            if parent.get('camera') == camera and parent.get('kind', 'rgb') == 'rgb':
                for base in (WORKSPACE, Path('.')):
                    p = base / v
                    if p.exists():
                        return p
    return None


def parse_points(meta, n):
    """Parse robomme_pixel_to_position reply into a list of n (xyz or None, spread)."""
    items = None
    for key in ('points', 'results', 'positions', 'selections', 'world_points'):
        v = find_key(meta, (key,), lambda x: isinstance(x, list) and len(x) == n)
        if v is not None:
            items = v
            break
    if items is None:
        # any list of length n of dicts
        for k, v, _ in walk(meta):
            if isinstance(v, list) and len(v) == n and all(isinstance(x, dict) for x in v):
                items = v
                break
    out = []
    if items is None:
        return [(None, None)] * n
    for it in items:
        xyz = None; spread = None; valid = True
        if isinstance(it, dict):
            if it.get('valid') is False:
                valid = False
            for key in ('position', 'world_position', 'world', 'xyz', 'point', 'position_m', 'world_m'):
                v = it.get(key)
                if is_vec(3)(v):
                    xyz = np.array(v, float)
                    break
                if isinstance(v, dict):
                    vv = [v.get('x'), v.get('y'), v.get('z')]
                    if all(isinstance(x, (int, float)) for x in vv):
                        xyz = np.array(vv, float)
                        break
            if xyz is None:
                for k2, v2, _ in walk(it):
                    if is_vec(3)(v2) and k2 not in ('pixel', 'uv'):
                        xyz = np.array(v2, float)
                        break
            sp = it.get('depth_spread_m')
            if isinstance(sp, (int, float)):
                spread = float(sp)
        elif is_vec(3)(it):
            xyz = np.array(it, float)
        if not valid or xyz is None or not np.all(np.isfinite(xyz)):
            out.append((None, spread))
        else:
            out.append((xyz, spread))
    return out


def load_video_frames(meta, max_frames=80, log=None):
    """Load front RGB frames of the conditioning video (phase video_demo), in order."""
    seq = find_key(meta, ('frame_sequence',), lambda x: isinstance(x, dict))
    mpath = None
    if seq:
        mp = seq.get('manifest_path') or find_key(seq, ('manifest_path',), lambda x: isinstance(x, str))
        if isinstance(mp, str):
            for base in (WORKSPACE, Path('.')):
                if (base / mp).exists():
                    mpath = base / mp
                    break
    if mpath is None:
        mp = find_key(meta, ('manifest_path',), lambda x: isinstance(x, str))
        if isinstance(mp, str) and (WORKSPACE / mp).exists():
            mpath = WORKSPACE / mp
    if mpath is None:
        if log:
            log('no conditioning manifest found')
        return [], []
    man = json.loads(mpath.read_text())
    frames = man.get('frames', [])
    vid = [f for f in frames if f.get('phase') == 'video_demo']
    vid.sort(key=lambda f: (f.get('frame_index', 0)))
    if log:
        log(f'manifest {mpath} frames={len(frames)} video={len(vid)}')
    if not vid:
        return [], []
    idx = sorted(set(int(round(x)) for x in np.linspace(0, len(vid) - 1, min(len(vid), max_frames))))
    out = []; wr = []
    for i in idx:
        f = vid[i]
        files = f.get('files', [])
        fp = None; wp = None
        for e in files:
            if e.get('kind', 'rgb') == 'rgb' and e.get('camera') == 'front':
                fp = e.get('path')
            if e.get('kind', 'rgb') == 'rgb' and e.get('camera') == 'wrist':
                wp = e.get('path')
        if fp is None:
            continue
        p = None
        for base in (WORKSPACE, mpath.parent, Path('.')):
            if (base / fp).exists():
                p = base / fp
                break
        if p is None:
            continue
        try:
            out.append(np.array(Image.open(p).convert('RGB')))
        except Exception:
            continue
    return out, idx
